# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE padding mask (VLLM_GLM5_MOE_MASK_PADDING).

Padding rows of a padded (CUDA-graph sized) batch carry stale hidden states;
routed like real tokens they change the expert block layout, and with it how
Marlin splits K for the real rows. The mask routes them to expert -1, which
the alignment drops. CPU only; the forward context is faked.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import moe_align_block_size as mab
from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr


@pytest.fixture
def fake_ctx(monkeypatch):
    state = {"is_padding": None, "available": True}
    monkeypatch.setattr(mr, "is_forward_context_available", lambda: state["available"])
    monkeypatch.setattr(mr, "get_forward_context",
                        lambda: SimpleNamespace(is_padding=state["is_padding"]))
    monkeypatch.setattr(mr, "_MASK_PADDING", None)
    monkeypatch.setattr(mr, "_MASK_MIN_ROWS", 0)
    return state


def _ids(n, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 288, (n, k), generator=g, dtype=torch.int32)


def _pad(n_real, n_total):
    p = torch.zeros(n_total, dtype=torch.bool)
    p[n_real:] = True
    return p


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_MOE_MASK_PADDING is False


def test_flag_is_part_of_the_compile_key():
    assert "VLLM_GLM5_MOE_MASK_PADDING" in envs.compile_factors()


def test_off_leaves_ids_untouched(fake_ctx, monkeypatch):
    monkeypatch.delenv("VLLM_GLM5_MOE_MASK_PADDING", raising=False)
    fake_ctx["is_padding"] = _pad(5, 16)
    ids = _ids(16)
    ref = ids.clone()
    assert mr.mask_padding_topk_ids(ids) is ids
    assert torch.equal(ids, ref)


def test_on_masks_padding_rows_in_place(fake_ctx, monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
    fake_ctx["is_padding"] = _pad(25, 32)
    ids = _ids(32)
    ref = ids.clone()
    out = mr.mask_padding_topk_ids(ids)
    assert out is ids                      # identity kept (fused decode handoff)
    assert torch.equal(ids[:25], ref[:25])
    assert (ids[25:] == -1).all()


def test_mask_view_of_the_persistent_buffer(fake_ctx, monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
    fake_ctx["is_padding"] = _pad(10, 64)[:16]   # the runner passes [:padded]
    ids = _ids(16)
    mr.mask_padding_topk_ids(ids)
    assert (ids[10:] == -1).all() and (ids[:10] >= 0).all()


def test_row_slice_of_the_batch_is_not_masked(fake_ctx, monkeypatch):
    # The prefill overlap calls the MoE on row slices; the batch mask does
    # not line up with a slice, so the call is left alone.
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
    fake_ctx["is_padding"] = _pad(30, 32)
    ids = _ids(16)
    ref = ids.clone()
    mr.mask_padding_topk_ids(ids)
    assert torch.equal(ids, ref)


@pytest.mark.parametrize("case", ["no_ctx", "no_mask", "short_mask"])
def test_no_usable_mask_is_a_no_op(fake_ctx, monkeypatch, case):
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
    if case == "no_ctx":
        fake_ctx["available"] = False
        fake_ctx["is_padding"] = _pad(1, 16)
    elif case == "short_mask":
        fake_ctx["is_padding"] = _pad(1, 8)     # shorter than the call
    ids = _ids(16)
    ref = ids.clone()
    mr.mask_padding_topk_ids(ids)
    assert torch.equal(ids, ref)


def test_fused_decode_sizes_are_skipped(fake_ctx, monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MOE_MAX_TOKENS", "8")
    fake_ctx["is_padding"] = _pad(3, 8)
    small = _ids(8)
    ref = small.clone()
    mr.mask_padding_topk_ids(small)
    assert torch.equal(small, ref)         # the fused kernel's own batch size
    fake_ctx["is_padding"] = _pad(12, 16)
    big = _ids(16)
    mr.mask_padding_topk_ids(big)
    assert (big[12:] == -1).all()


def test_masked_rows_leave_the_real_rows_layout_unchanged():
    """The deterministic alignment of the real rows does not depend on what
    the padding rows would have routed to once they are masked."""
    real = _ids(25, seed=1)
    outs = []
    for seed in (2, 3, 4):
        ids = torch.cat([real, _ids(7, seed=seed)])
        ids[25:] = -1
        n = ids.numel()
        bs = 16
        max_pad = n + 288 * (bs - 1)
        sorted_ids = torch.empty(max_pad, dtype=torch.int32)
        expert_ids = torch.empty((max_pad + bs - 1) // bs, dtype=torch.int32)
        ntp = torch.empty(1, dtype=torch.int32)
        mab.deterministic_moe_align_block_size(ids, 288, bs, sorted_ids, expert_ids, ntp, None, None)
        outs.append((sorted_ids[: int(ntp)].clone(), expert_ids.clone(), int(ntp)))
    for o in outs[1:]:
        assert o[2] == outs[0][2]
        assert torch.equal(o[0], outs[0][0]) and torch.equal(o[1], outs[0][1])


def test_runner_masks_before_the_experts():
    src = inspect.getsource(mr.MoERunner)
    sel = src.index("self.router.select_experts(")
    mask = src.index("topk_ids = mask_padding_topk_ids(topk_ids)")
    mod = src.index("self.routed_experts.forward_modular(")
    assert sel < mask < mod


def test_no_cuda_was_initialised():
    assert not torch.cuda.is_initialized()
