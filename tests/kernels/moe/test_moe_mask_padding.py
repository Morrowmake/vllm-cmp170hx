# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE padding mask (VLLM_MOE_MASK_PADDING).

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
from tests.kernels.moe_align_reference import align
from tests.sm80.tuning_rows import use_rows
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm_sm80.layers.moe import routed_experts as mr


@pytest.fixture
def fake_ctx(monkeypatch):
    state = {"is_padding": None, "available": True}
    monkeypatch.setattr(mr, "is_forward_context_available", lambda: state["available"])
    monkeypatch.setattr(mr, "get_forward_context",
                        lambda: SimpleNamespace(is_padding=state["is_padding"]))
    monkeypatch.setattr(mr, "_MASK_PADDING", None)
    use_rows(monkeypatch)
    return state


def _cutoff(monkeypatch, rows):
    """The moe_router/fused_route_max_rows row at top-8 (None: no row)."""
    if rows is None:
        use_rows(monkeypatch)
    else:
        use_rows(
            monkeypatch,
            ("moe_router", "fused_route_max_rows", {"top_k": 8}, int(rows)),
        )


def _ids(n, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 288, (n, k), generator=g, dtype=torch.int32)


def _pad(n_real, n_total):
    p = torch.zeros(n_total, dtype=torch.bool)
    p[n_real:] = True
    return p


def test_flag_defaults_off():
    assert envs.VLLM_MOE_MASK_PADDING is False


def test_flag_is_part_of_the_compile_key():
    assert "VLLM_MOE_MASK_PADDING" in envs.compile_factors()


def test_off_leaves_ids_untouched(fake_ctx, monkeypatch):
    monkeypatch.delenv("VLLM_MOE_MASK_PADDING", raising=False)
    fake_ctx["is_padding"] = _pad(5, 16)
    ids = _ids(16)
    ref = ids.clone()
    assert mr.mask_padding_topk_ids(ids) is ids
    assert torch.equal(ids, ref)


def test_on_masks_padding_rows_in_place(fake_ctx, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    _cutoff(monkeypatch, 0)
    fake_ctx["is_padding"] = _pad(25, 32)
    ids = _ids(32)
    ref = ids.clone()
    out = mr.mask_padding_topk_ids(ids)
    assert out is ids                      # identity kept (fused decode handoff)
    assert torch.equal(ids[:25], ref[:25])
    assert (ids[25:] == -1).all()


def test_mask_view_of_the_persistent_buffer(fake_ctx, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    _cutoff(monkeypatch, 0)
    fake_ctx["is_padding"] = _pad(10, 64)[:16]   # the runner passes [:padded]
    ids = _ids(16)
    mr.mask_padding_topk_ids(ids)
    assert (ids[10:] == -1).all() and (ids[:10] >= 0).all()


def test_row_slice_of_the_batch_is_not_masked(fake_ctx, monkeypatch):
    # A call on a row slice of the batch: the batch mask does not line up
    # with it, so the call is left alone.
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    _cutoff(monkeypatch, 0)
    fake_ctx["is_padding"] = _pad(30, 32)
    ids = _ids(16)
    ref = ids.clone()
    mr.mask_padding_topk_ids(ids)
    assert torch.equal(ids, ref)


@pytest.mark.parametrize("case", ["no_ctx", "no_mask", "short_mask"])
def test_no_usable_mask_is_a_no_op(fake_ctx, monkeypatch, case):
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    _cutoff(monkeypatch, 0)
    if case == "no_ctx":
        fake_ctx["available"] = False
        fake_ctx["is_padding"] = _pad(1, 16)
    elif case == "short_mask":
        fake_ctx["is_padding"] = _pad(1, 8)     # shorter than the call
    ids = _ids(16)
    ref = ids.clone()
    mr.mask_padding_topk_ids(ids)
    assert torch.equal(ids, ref)


def test_sizes_up_to_the_cutoff_are_skipped(fake_ctx, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    _cutoff(monkeypatch, 8)
    fake_ctx["is_padding"] = _pad(3, 8)
    small = _ids(8)
    ref = small.clone()
    mr.mask_padding_topk_ids(small)
    assert torch.equal(small, ref)         # the fused kernel's own batch size
    fake_ctx["is_padding"] = _pad(12, 16)
    big = _ids(16)
    mr.mask_padding_topk_ids(big)
    assert (big[12:] == -1).all()


def _old_cutoff(decode_kernels, cutoff):
    # The previous release: VLLM_AMPERE_DECODE_KERNELS=0 forced the cutoff to 0.
    return (8 if cutoff is None else int(cutoff)) if decode_kernels else 0


def _new_cutoff(cutoff):
    return 8 if cutoff is None else int(cutoff)


@pytest.mark.parametrize("rows", [1, 2, 8, 9, 11, 12, 16, 32])
@pytest.mark.parametrize("cutoff", [None, "0", "11"])
@pytest.mark.parametrize("decode_kernels", [0, 1])
@pytest.mark.parametrize("mask", [0, 1])
def test_mask_truth_table(fake_ctx, monkeypatch, mask, decode_kernels, cutoff, rows):
    """Every (mask, GLM decode switch, cutoff, rows) case. The cutoff no longer
    depends on VLLM_AMPERE_DECODE_KERNELS; the only cases that differ from the
    previous release are mask on, GLM decode switch off and rows at or below
    the cutoff, which were masked and are now left alone. The served row
    (mask on, GLM decode switch on, cutoff unset) is unchanged."""
    from vllm_sm80.kernels.moe import route_v2_masks_padding
    from vllm_sm80.kernels.moe.route import pad_supported

    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", str(mask))
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", str(decode_kernels))
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1")
    _cutoff(monkeypatch, cutoff)
    fake_ctx["is_padding"] = _pad(rows - 1, rows)
    ids = _ids(rows)
    ref = ids.clone()
    assert mr.mask_padding_topk_ids(ids) is ids

    masked = bool((ids[-1] == -1).all())
    assert masked == bool(mask and rows > _new_cutoff(cutoff))
    assert torch.equal(ids[:-1], ref[:-1])
    if not masked:
        assert torch.equal(ids, ref)
    assert route_v2_masks_padding(rows) == (masked and pad_supported(rows))

    was_masked = bool(mask and rows > _old_cutoff(decode_kernels, cutoff))
    changed = bool(mask and not decode_kernels and rows <= _new_cutoff(cutoff))
    assert (masked != was_masked) == changed
    if mask and decode_kernels and cutoff is None:
        assert masked == was_masked == (rows > 8)


def test_masked_rows_leave_the_real_rows_layout_unchanged():
    """The deterministic alignment of the real rows does not depend on what
    the padding rows would have routed to once they are masked."""
    real = _ids(25, seed=1)
    outs = []
    for seed in (2, 3, 4):
        ids = torch.cat([real, _ids(7, seed=seed)])
        ids[25:] = -1
        sorted_ids, expert_ids, ntp = align(ids, 288, 16)
        outs.append((sorted_ids[: int(ntp)].clone(), expert_ids.clone(), int(ntp)))
    for o in outs[1:]:
        assert o[2] == outs[0][2]
        assert torch.equal(o[0], outs[0][0]) and torch.equal(o[1], outs[0][1])


def test_runner_masks_before_the_experts():
    # Upstream's runner hands the router's ids straight to forward_modular,
    # and the fork's forward_modular masks them before upstream's.
    src = inspect.getsource(MoERunner._apply_quant_method)
    sel = src.index("self.router.select_experts(")
    mod = src.index("self.routed_experts.forward_modular(")
    assert sel < mod
    fork = inspect.getsource(mr.Sm80RoutedExperts.forward_modular)
    mask = fork.index("topk_ids = mask_padding_topk_ids(topk_ids)")
    assert mask < fork.index("super().forward_modular(")


def test_no_cuda_was_initialised():
    assert not torch.cuda.is_initialized()
