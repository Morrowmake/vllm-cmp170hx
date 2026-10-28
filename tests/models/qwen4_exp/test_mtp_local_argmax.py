# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``get_top_tokens`` on the Qwen4Exp MTP drafter, at two vocab shards (TP2).

With ``use_local_argmax_reduction`` the speculator drafts greedily with
``get_top_tokens`` instead of ``compute_logits(...).argmax(-1)``. Each rank
takes the argmax of its own vocab shard (padding masked to -inf); the (value,
index) pairs are all-gathered and the first rank holding the largest value
wins. These CPU checks run both rank programs with the all-gather they make
joined across the two, and require the same token as the full-vocab argmax,
including ties (first index wins), NaN rows, signed zeros, -inf rows and
padding columns that would win if they were not masked.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    pad_vocab_size,
)
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP

_LP = "vllm.model_executor.layers.logits_processor"
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"
QWEN_VOCAB = 248320  # qwen3.8-flash-next; a multiple of 64, so no padding
TP = 2


def _run_tp2(call):
    """Run ``call(rank)`` for both ranks, each all-gather joined across them.

    A first pass records what each rank contributes; the second returns the
    concatenation (rank 0 first, as the real all-gather) to both ranks.
    """
    sent = {}

    def recorder(rank):
        def gather(x, dim=-1):
            sent[rank] = x.clone()
            return torch.cat([x] * TP, dim=dim)

        return gather

    for rank in range(TP):
        with mock.patch(f"{_LP}.tensor_model_parallel_all_gather", recorder(rank)):
            call(rank)

    def joined(x, dim=-1):
        return torch.cat([sent[r] for r in range(TP)], dim=dim)

    with mock.patch(f"{_LP}.tensor_model_parallel_all_gather", joined):
        return [call(rank) for rank in range(TP)]


class _ShardHead:
    """What both paths read from a vocab-parallel head: the shard's logits come
    from ``quant_method.apply`` (an EXL3 or unquantized GEMM in service)."""

    def __init__(self, shard_logits, start, num_pad):
        self.tp_size = TP
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=num_pad
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: shard_logits.clone()
        )


def _shards(logits: torch.Tensor, pad_value: float):
    """Split full logits [B, V] the way VocabParallelEmbedding shards the
    vocab at TP2: pad V to a multiple of 64, equal shards, padding at the end
    of the last shard (filled with pad_value)."""
    rows, vocab = logits.shape
    padded = pad_vocab_size(vocab)
    per = padded // TP
    full = logits.new_full((rows, padded), pad_value)
    full[:, :vocab] = logits
    heads = []
    for rank in range(TP):
        start = rank * per
        real = max(0, min(vocab, start + per) - start)
        heads.append(
            _ShardHead(full[:, start : start + per].contiguous(), start, per - real)
        )
    return heads


def _both_paths(logits: torch.Tensor, pad_value: float = 1e30):
    """(full-vocab argmax per rank, get_top_tokens per rank)."""
    vocab = logits.shape[1]
    heads = _shards(logits, pad_value)
    lp = LogitsProcessor(vocab)
    lp.use_all_gather = True  # the CUDA platform's choice
    hidden = torch.zeros(logits.shape[0], 8, dtype=logits.dtype)
    full = _run_tp2(lambda r: lp(heads[r], hidden).argmax(dim=-1))
    local = _run_tp2(lambda r: lp.get_top_tokens(heads[r], hidden))
    return full, local


def _check(logits: torch.Tensor, pad_value: float = 1e30) -> torch.Tensor:
    want = logits.argmax(dim=-1)
    full, local = _both_paths(logits, pad_value)
    for rank in range(TP):
        assert local[rank].dtype == torch.int64
        assert torch.equal(full[rank], want), rank
        assert torch.equal(local[rank], want), (rank, local[rank], want)
    return want


@pytest.mark.parametrize("vocab", [QWEN_VOCAB, 500, 130])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_random_rows_match_full_argmax(default_vllm_config, vocab, dtype):
    g = torch.Generator().manual_seed(vocab)
    for rows in (1, 4, 32):
        _check(torch.randn(rows, vocab, generator=g).to(dtype))


def test_bf16_rows_with_many_ties(default_vllm_config):
    """Coarse bf16 values: the row maximum recurs, often in both shards."""
    g = torch.Generator().manual_seed(3)
    logits = torch.randint(-4, 5, (64, QWEN_VOCAB), generator=g).to(torch.bfloat16)
    want = _check(logits)
    assert (want < QWEN_VOCAB // TP).all()  # every row's first maximum is in shard 0


def test_ties_place_by_place(default_vllm_config):
    vocab = 500  # padded to 512: shards [0, 256) and [256, 500) + 12 padding
    per = pad_vocab_size(vocab) // TP
    logits = torch.full((10, vocab), -3.0, dtype=torch.bfloat16)
    logits[0, [17, 300]] = 5.0  # in both shards: shard 0's wins
    logits[1, [301, 499]] = 5.0  # twice in shard 1 only
    logits[2, [per - 1, per]] = 5.0  # either side of the shard boundary
    logits[3] = 1.0  # all equal: index 0
    logits[4] = -1.0
    logits[4, 400] = -0.0  # -0 in shard 1, +0 in shard 0: equal, first wins
    logits[4, 100] = 0.0
    logits[5, 450] = float("nan")  # NaN beats every number in both paths
    logits[5, 10] = 7.0
    logits[6, [260, 20]] = float("nan")  # two NaNs: the first index
    logits[7] = float("-inf")  # nothing finite: index 0
    logits[8, :per] = float("-inf")  # shard 0 empty: shard 1's first maximum
    logits[8, 333] = 2.0
    logits[9, vocab - 1] = 9.0  # last real column next to the padding
    want = _check(logits)
    assert want.tolist() == [17, 301, per - 1, 0, 100, 450, 20, 0, 333, vocab - 1]


@pytest.mark.parametrize("pad_value", [1e30, float("inf"), float("nan")])
def test_padding_never_wins(default_vllm_config, pad_value):
    """Padding columns are sliced off in the full path and masked to -inf in
    the local one, whatever the head computed for them."""
    g = torch.Generator().manual_seed(5)
    _check(torch.randn(16, 500, generator=g).to(torch.bfloat16), pad_value)


def test_indices_exact_through_fp32_pairs(default_vllm_config):
    """The pairs carry indices as fp32: exact for every Qwen vocab id."""
    assert QWEN_VOCAB < 2**24
    logits = torch.full((3, QWEN_VOCAB), -1.0, dtype=torch.bfloat16)
    for row, col in enumerate((QWEN_VOCAB - 1, QWEN_VOCAB // 2, QWEN_VOCAB // 2 - 1)):
        logits[row, col] = 1.0
    assert _check(logits).tolist() == [QWEN_VOCAB - 1, QWEN_VOCAB // 2, QWEN_VOCAB // 2 - 1]


def _real_heads(vocab: int, hidden: int, seed: int):
    """Two ParallelLMHead shards (rank 0 and 1 of TP2) over one weight, with
    large padding rows that would win the argmax unless masked."""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(vocab, hidden, generator=g) * 0.05
    heads = []
    for rank in range(TP):
        with (
            mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=rank),
            mock.patch(
                f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=TP
            ),
        ):
            head = ParallelLMHead(vocab, hidden, prefix="lm_head")
        idx = head.shard_indices
        with torch.no_grad():
            head.weight.fill_(1.0)  # padding rows: large logits
            n = idx.org_vocab_end_index - idx.org_vocab_start_index
            head.weight[:n] = weight[idx.org_vocab_start_index : idx.org_vocab_end_index]
        heads.append(head.to(torch.bfloat16))
    return heads


def _mtp(head) -> Qwen4ExpMTP:
    """Only what compute_logits and get_top_tokens read."""
    mtp = object.__new__(Qwen4ExpMTP)
    nn.Module.__init__(mtp)
    mtp.lm_head = head
    mtp.logits_processor = LogitsProcessor(head.org_vocab_size)
    mtp.logits_processor.use_all_gather = True
    return mtp


@pytest.mark.parametrize("vocab", [500, 512])
def test_qwen_mtp_get_top_tokens_matches_compute_logits(default_vllm_config, vocab):
    """The drafter's own method, through real vocab-parallel heads at TP2."""
    heads = _real_heads(vocab, hidden=64, seed=vocab)
    assert heads[1].shard_indices.num_org_vocab_padding == pad_vocab_size(vocab) - vocab
    models = [_mtp(h) for h in heads]
    g = torch.Generator().manual_seed(1)
    for rows in (1, 4, 32):
        hidden = torch.randn(rows, 64, generator=g).to(torch.bfloat16)
        full = _run_tp2(lambda r: models[r].compute_logits(hidden))
        greedy = full[0].argmax(dim=-1)
        assert torch.equal(full[0], full[1]) and full[0].shape == (rows, vocab)
        local = _run_tp2(lambda r: models[r].get_top_tokens(hidden))
        for rank in range(TP):
            assert torch.equal(local[rank], greedy), rank
        assert (greedy < vocab).all()


def test_speculator_probe_finds_the_method():
    """The proposer enables the reduction only if the draft model has it."""
    assert callable(getattr(Qwen4ExpMTP, "get_top_tokens", None))
