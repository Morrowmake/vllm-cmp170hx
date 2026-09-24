# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GLM5_PP_FOLD_DRAFT_FC, CPU only.

Each pipeline stage multiplies its own drafter aux states by their column
block of the drafter's fc.weight and forwards one fp32 partial sum; the last
stage hands the drafter the finished fc output. This must equal fc applied to
the concatenated aux states of a single-stage run, up to the fp32 summation
order (bf16 inputs, fp32 products and sums, one rounding at the end).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tests.models.glm5next.test_aux_hidden_states_pp import (
    HIDDEN,
    TOKENS,
    VOCAB,
    _build_stage,
    _set_pp,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle import aux_fc_fold, eagle3_utils
from vllm.v1.worker.gpu.spec_decode.eagle.aux_fc_fold import (
    AUX_FC_PARTIAL_KEY,
    load_fc_column_blocks,
    maybe_configure_aux_fc_fold,
)


def _fc_file(tmp_path, num_aux, dtype=torch.bfloat16):
    torch.manual_seed(1)
    w = (torch.randn(HIDDEN, num_aux * HIDDEN) / HIDDEN**0.5).to(dtype)
    path = tmp_path / "model.safetensors"
    save_file({"fc.weight": w, "norm.weight": torch.ones(HIDDEN)}, str(path))
    return w, [str(path)]


def _embed():
    torch.manual_seed(0)
    return nn.Embedding(VOCAB, HIDDEN).to(torch.bfloat16)


def _single_stage_fc(monkeypatch, embed, num_layers, aux_layers, ids, pos, w):
    _set_pp(monkeypatch, 0, 1)
    m = _build_stage(embed, num_layers, 0, num_layers, aux_layers)
    with torch.no_grad():
        _, aux = m(ids, pos, None)
    cat = torch.cat(aux, dim=-1)
    # The unfolded fc: one GEMM over K = n * hidden, fp32 accumulation.
    return torch.mm(cat.float(), w.float().t())


def _folded_pipeline(
    monkeypatch, embed, num_layers, partition, aux_layers, ids, pos, files
):
    monkeypatch.setenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "1")
    spec = SimpleNamespace(method="dflash")
    world = len(partition)
    bounds = [sum(partition[:r]) for r in range(world + 1)]
    received = None
    for rank in range(world):
        _set_pp(monkeypatch, rank, world)
        m = _build_stage(embed, num_layers, bounds[rank], bounds[rank + 1], aux_layers)
        m.device = "cpu"
        wrapper = SimpleNamespace(
            model=m, make_empty_intermediate_tensors=m.make_empty_intermediate_tensors
        )
        handler = SimpleNamespace(aux_hidden_state_relay_keys=("stale",))
        # Runner order: reserve aux slots, configure the relay, then the fold.
        eagle3_utils.reserve_aux_intermediate_tensor_slots(wrapper)
        assert maybe_configure_aux_fc_fold(wrapper, spec, handler, files=files)
        assert handler.aux_hidden_state_relay_keys == ()
        if rank > 0:
            buf = wrapper.make_empty_intermediate_tensors(TOKENS, torch.bfloat16, "cpu")
            # Only the streams and the fp32 partial sum travel.
            assert set(buf.tensors) == {"hidden_states", AUX_FC_PARTIAL_KEY}
            assert buf[AUX_FC_PARTIAL_KEY].dtype == torch.float32
            assert set(received.tensors) == set(buf.tensors)
        with torch.no_grad():
            out = m(ids, pos, received)
        if rank == world - 1:
            return out
        assert isinstance(out, IntermediateTensors)
        received = out
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "num_layers,partition,aux_layers",
    [
        (45, [13, 11, 11, 10], (6, 15, 25, 34, 43)),
        (45, [12, 12, 12, 9], (6, 15, 25, 34, 43)),
        (10, [3, 2, 3, 2], (2, 3, 5, 7, 8)),  # aux on boundaries; none on the last
        (12, [4, 4, 2, 2], (1, 2, 11)),  # middle stages without aux
    ],
)
def test_folded_fc_equals_single_stage_fc(
    monkeypatch, tmp_path, num_layers, partition, aux_layers
):
    embed = _embed()
    ids = torch.randint(0, VOCAB, (TOKENS,))
    pos = torch.arange(TOKENS)
    w, files = _fc_file(tmp_path, len(aux_layers))
    ref = _single_stage_fc(monkeypatch, embed, num_layers, aux_layers, ids, pos, w)
    hidden, aux = _folded_pipeline(
        monkeypatch, embed, num_layers, partition, aux_layers, ids, pos, files
    )
    assert len(aux) == 1
    folded = aux[0]
    assert folded.dtype == torch.float32 and folded.shape == (TOKENS, HIDDEN)
    # Same fp32 products and sums in a different order.
    torch.testing.assert_close(folded, ref, rtol=1e-5, atol=1e-5)
    # After the drafter's single rounding the values agree to within one bf16 ulp.
    torch.testing.assert_close(
        folded.to(torch.bfloat16).float(),
        ref.to(torch.bfloat16).float(),
        rtol=2**-7,
        atol=1e-6,
    )


def test_column_blocks_are_the_fc_slices(tmp_path):
    w, files = _fc_file(tmp_path, 5)
    blocks = load_fc_column_blocks(files, [0, 3, 4], HIDDEN, 5)
    for i, b in blocks.items():
        assert torch.equal(b, w[:, i * HIDDEN : (i + 1) * HIDDEN])
    with pytest.raises(ValueError):
        load_fc_column_blocks(files, [0], HIDDEN, 4)


def test_fold_off_or_unsupported_is_a_no_op(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "0")
    assert not maybe_configure_aux_fc_fold(SimpleNamespace(), None, None)
    monkeypatch.setenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "1")
    _set_pp(monkeypatch, 0, 1)
    assert not maybe_configure_aux_fc_fold(
        SimpleNamespace(model=SimpleNamespace()), SimpleNamespace(method="dflash"), None
    )


def test_drafter_takes_the_folded_fc_output():
    d = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(d)
    weight = torch.zeros(1, dtype=torch.bfloat16)
    fc = SimpleNamespace(output_size=HIDDEN, weight=weight)
    d.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    d.aux_fc_folded = True
    x = torch.randn(TOKENS, HIDDEN)
    out = d.combine_hidden_states(x)
    assert out.dtype == torch.bfloat16 and torch.equal(out, x.to(torch.bfloat16))
    with pytest.raises(AssertionError):
        d.combine_hidden_states(torch.randn(TOKENS, 5 * HIDDEN))


def test_mark_drafter(monkeypatch):
    spec = SimpleNamespace(model=SimpleNamespace())
    aux_fc_fold.mark_drafter_aux_fc_folded(spec)
    assert spec.model.aux_fc_folded
