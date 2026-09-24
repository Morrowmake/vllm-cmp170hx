# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GLM5_DRAFTER_SELECTOR_SHARD: vocab-sharded DFlash2 selector codebooks.

The DFlash2 candidate selector holds two (vocab, rank) bf16 codebooks
(154,880 x 256 each, 151.25 MiB per rank replicated). With the flag each
tensor-parallel rank keeps a contiguous vocab shard, looks up the rows it owns
(zero elsewhere) and one all-reduce of the stacked row sets rebuilds the
gathered rows. Each row has exactly one non-zero contributor, so the sum is
exact and the edge scores must be bit-identical to the replicated module.

These tests simulate TP=4 on CPU: four sharded modules, weights loaded through
their weight_loader from the full tensors, the all-reduce replaced by the sum
over the four ranks' local rows.

    CUDA_VISIBLE_DEVICES= pytest -q tests/models/glm5next/test_drafter_selector_shard.py
"""

import pytest
import torch

import vllm.model_executor.models.qwen3_dflash2 as dflash2
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode

VOCAB = 1000  # not a multiple of 4 * 64: exercises the padded last shard
RANK = 32
HIDDEN = 64
TOP_K = 8
STEPS = 3
TP = 4


def _full_weights(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "predecessor_codebook": torch.randn(VOCAB, RANK, generator=gen).bfloat16(),
        "successor_codebook": torch.randn(VOCAB, RANK, generator=gen).bfloat16(),
        "hidden_projection.weight": torch.randn(RANK, HIDDEN, generator=gen).bfloat16(),
    }


def _selector(monkeypatch, *, shard, tp_rank, tp_size):
    monkeypatch.setenv("VLLM_GLM5_DRAFTER_SELECTOR_SHARD", "1" if shard else "0")
    monkeypatch.setattr(
        dflash2, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(dflash2, "get_tensor_model_parallel_rank", lambda: tp_rank)
    # ReplicatedLinear and its parameters only record the rank; no process
    # group exists here.
    import vllm.model_executor.layers.linear as linear_mod
    import vllm.model_executor.parameter as parameter_mod

    for mod in (linear_mod, parameter_mod):
        for fn, value in (
            ("get_tensor_model_parallel_rank", tp_rank),
            ("get_tensor_model_parallel_world_size", tp_size),
        ):
            if hasattr(mod, fn):
                monkeypatch.setattr(mod, fn, lambda _v=value: _v)
    cfg = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(cfg):
        sel = dflash2.CandidateSelector(
            hidden_size=HIDDEN,
            vocab_size=VOCAB,
            rank=RANK,
            top_k=TOP_K,
            params_dtype=torch.bfloat16,
            prefix="candidate_selector",
        )
    for name, weight in _full_weights().items():
        param = dict(sel.named_parameters())[name]
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            assert param.shape == weight.shape
            param.data.copy_(weight)
        else:
            loader(param, weight)
    return sel


def _inputs(batch, seed=1):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (batch, STEPS, TOP_K), generator=gen)
    # Shard edges and the last id, where an off-by-one would show.
    edges = torch.tensor([0, 249, 250, 251, 499, 500, 750, VOCAB - 1])
    ids[0, 0, :] = edges[:TOP_K]
    anchor = torch.randint(0, VOCAB, (batch,), generator=gen)
    anchor[-1] = VOCAB - 1
    unary = torch.randn(batch, STEPS, TOP_K, generator=gen)
    hidden = torch.randn(batch, STEPS, HIDDEN, generator=gen).bfloat16()
    return ids, unary, hidden, anchor


def test_shard_bounds_cover_the_vocab_once():
    for vocab in (1000, 154_880, 7):
        seen = []
        for r in range(TP):
            start, end, per = dflash2.selector_vocab_shard(vocab, r, TP)
            assert end - start <= per
            seen.extend(range(start, end))
        assert seen == list(range(vocab))
    assert dflash2.selector_vocab_shard(154_880, 3, 4) == (116_160, 154_880, 38_720)


def test_sharded_bytes_at_deployment():
    full = 2 * 154_880 * 256 * 2
    _, _, per = dflash2.selector_vocab_shard(154_880, 0, 4)
    shard = 2 * per * 256 * 2
    assert round(full / 2**20, 2) == 151.25
    assert round((full - shard) / 2**20, 2) == 113.44


@pytest.mark.parametrize("batch", [1, 5, 8])
def test_sharded_scores_are_bit_identical(monkeypatch, batch):
    ids, unary, hidden, anchor = _inputs(batch)
    ref_sel = _selector(monkeypatch, shard=False, tp_rank=0, tp_size=1)
    assert not ref_sel.shard_vocab
    with torch.inference_mode():
        ref = ref_sel(ids, unary, hidden, anchor)

    shards = [
        _selector(monkeypatch, shard=True, tp_rank=r, tp_size=TP) for r in range(TP)
    ]
    assert all(s.shard_vocab for s in shards)
    assert shards[0].predecessor_codebook.shape == (VOCAB // TP, RANK)
    # The padded rows of every shard are zero and never looked up.
    for s in shards:
        n = s.vocab_end - s.vocab_start
        assert torch.count_nonzero(s.successor_codebook[n:]) == 0

    pred_ids = dflash2._predecessor_ids(ids, anchor, TOP_K)
    local = [
        torch.stack(
            (
                dflash2.local_codebook_rows(
                    s.predecessor_codebook, pred_ids, s.vocab_start, s.vocab_end
                ),
                dflash2.local_codebook_rows(
                    s.successor_codebook, ids, s.vocab_start, s.vocab_end
                ),
            )
        )
        for s in shards
    ]
    reduced = local[0].clone()
    for t in local[1:]:
        reduced = reduced + t
    # The reduction rebuilds the full-table rows bit for bit.
    full = _full_weights()
    assert torch.equal(reduced[0], full["predecessor_codebook"][pred_ids])
    assert torch.equal(reduced[1], full["successor_codebook"][ids])

    for r, s in enumerate(shards):
        calls = []

        def fake_all_reduce(x, _r=r):
            assert torch.equal(x, local[_r])
            calls.append(1)
            return reduced.clone()

        monkeypatch.setattr(
            dflash2, "tensor_model_parallel_all_reduce", fake_all_reduce
        )
        with torch.inference_mode():
            out = s(ids, unary, hidden, anchor)
        assert calls == [1], "exactly one collective per selector call"
        assert torch.equal(out, ref), f"rank {r}"


def test_flag_off_or_tp1_keeps_the_replicated_module(monkeypatch):
    s = _selector(monkeypatch, shard=True, tp_rank=0, tp_size=1)
    assert not s.shard_vocab
    assert s.predecessor_codebook.shape == (VOCAB, RANK)
    s = _selector(monkeypatch, shard=False, tp_rank=2, tp_size=TP)
    assert not s.shard_vocab
    assert s.successor_codebook.shape == (VOCAB, RANK)
