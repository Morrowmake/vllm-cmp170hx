# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE padding mask (VLLM_MOE_MASK_PADDING) together with the sm_80
fused decode router v2 (VLLM_MOE_DECODE_ROUTE_V2).

Route v2 computes the top-k AND the Marlin block alignment at the gate and
hands the alignment to moe_align_block_size through a stash keyed by the
topk_ids tensor's identity. The padding mask edits those ids in place after
routing, for batches above moe_router/fused_route_max_rows (8) rows. Without
dropping the stash, a padded batch of 9..32 rows would get the alignment of
the unmasked ids and the mask would do nothing. These tests drive the real
maybe_moe_route_v2 stash path with the router kernel replaced by a torch
stand-in (CPU only), then the real mask and the real alignment entry point.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_sm80.kernels.moe as moe_gates
import vllm_sm80.kernels.moe.route as router_module
from tests.kernels.moe_align_reference import align, chunked_align_reference
from tests.sm80.tuning_rows import use_rows
from vllm.model_executor.layers.fused_moe import moe_align_block_size as mab
from vllm_sm80.kernels.moe import align as moe_align_kernel
from vllm_sm80.layers.moe import routed_experts as mr
from vllm.triton_utils import tl, triton

E = 288
TOPK = 8


def _align(ids, bs):
    """A fresh deterministic alignment (the reference)."""
    return align(ids, E, bs)


def _same(a, b, bs):
    ntp = int(a[2])
    if ntp != int(b[2]):
        return False
    nb = ntp // bs
    return torch.equal(a[0][:ntp], b[0][:ntp]) and torch.equal(a[1][:nb], b[1][:nb])


@pytest.fixture
def env(monkeypatch):
    state = {"is_padding": None}
    monkeypatch.setattr(mr, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mr, "get_forward_context",
        lambda: SimpleNamespace(is_padding=state["is_padding"]),
    )
    monkeypatch.setattr(mr, "_MASK_PADDING", None)
    monkeypatch.setattr(moe_gates, "_PENDING", None)
    monkeypatch.setattr(moe_gates, "_PENDING_ROUTING", None)
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_MOE_DECODE_ROUTE_V2", "1")
    use_rows(
        monkeypatch,
        ("moe_router", "fused_route_max_rows", {"top_k": 8}, 8),
        ("moe_router", "route_v2_max_rows",
         {"n_experts": 288, "top_k": 8, "hidden": 4096}, 32),
    )
    # These tests cover the re-align fallback; the in-kernel mask
    # (VLLM_GLM5_MOE_ROUTE_V2_MASK) is tested in test_moe_route_v2_mask.py.
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "0")
    # 1 = the deterministic alignment; on CPU its kernel entry point runs the
    # torch reference of the same decomposition.
    monkeypatch.setattr(moe_align_kernel, "deterministic_moe_align_mode", lambda: 1)
    monkeypatch.setattr(moe_align_kernel, "kernel_moe_align_block_size",
                        chunked_align_reference)
    return state


def _route_v2(monkeypatch, m, seed):
    """Run the real maybe_moe_route_v2 with the kernel replaced by a torch
    stand-in that routes like the kernel would and aligns the routed ids
    (unmasked), exactly what the kernel hands over."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]
    ).to(torch.int32)
    w = torch.full((m, TOPK), 1.0 / TOPK)
    logits = torch.randn(m, E, generator=g)

    def fake_moe_route(x, weight, bias, *, topk, block_size, num_experts,
                       renormalize, routed_scaling_factor, is_padding=None):
        assert is_padding is None          # the re-align fallback path
        s, e, n = _align(ids, block_size)
        return logits, w, ids, s, e, n

    import vllm_sm80.kernels.moe.route as mrt

    monkeypatch.setattr(mrt, "moe_route", fake_moe_route)
    monkeypatch.setattr(moe_gates, "use_ampere_moe_route_v2", lambda *a: True)
    router = SimpleNamespace(top_k=TOPK, e_score_correction_bias=SimpleNamespace(
        data=torch.zeros(E)), routed_scaling_factor=1.0)
    gate = SimpleNamespace(weight=torch.empty(E, 16))
    out = moe_gates.maybe_moe_route_v2(gate, router, torch.empty(m, 16))
    assert out is logits
    w_ids = moe_gates.take_fused_routing(logits)
    assert w_ids is not None and w_ids[1] is ids
    return ids


@pytest.mark.parametrize("m", list(range(9, 33)))
def test_padded_route_v2_batch_is_realigned_from_masked_ids(env, monkeypatch, m):
    n_real = max(1, m - 1 - (m % 5))      # 1..(m-1) padding rows
    ids = _route_v2(monkeypatch, m, seed=m)
    unmasked = ids.clone()
    bs = moe_gates.marlin_block_size_m(m, TOPK, E)
    stale = _align(unmasked, bs)

    env["is_padding"] = torch.arange(m) >= n_real
    out = mr.mask_padding_topk_ids(ids)

    # The mask took effect, in place.
    assert out is ids
    assert torch.equal(ids[:n_real], unmasked[:n_real])
    assert (ids[n_real:] == -1).all()
    # The stash for these ids is gone ...
    assert moe_gates._PENDING is None
    # ... so the alignment Marlin asks for is a fresh deterministic one of the
    # masked ids, not the router's alignment of the unmasked ids.
    got = mab.moe_align_block_size(ids, bs, E)
    ref = _align(ids.clone(), bs)
    assert _same(got, ref, bs)
    assert not _same(got, stale, bs)
    # No slot in the used range refers to a padding row's routes.
    ntp = int(got[2])
    slots = got[0][:ntp]
    real = slots[slots < ids.numel()]
    assert (real < n_real * TOPK).all()


def test_unpadded_route_v2_batch_keeps_the_handoff_when_mask_is_off(env, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "0")
    ids = _route_v2(monkeypatch, 16, seed=1)
    bs = moe_gates.marlin_block_size_m(16, TOPK, E)
    stashed = moe_gates._PENDING[3]
    env["is_padding"] = torch.arange(16) >= 12
    mr.mask_padding_topk_ids(ids)
    got = mab.moe_align_block_size(ids, bs, E)
    assert got is stashed                  # handoff still taken, nothing masked


def test_no_usable_mask_keeps_the_handoff(env, monkeypatch):
    ids = _route_v2(monkeypatch, 16, seed=2)
    bs = moe_gates.marlin_block_size_m(16, TOPK, E)
    stashed = moe_gates._PENDING[3]
    env["is_padding"] = None               # no padding mask in this step
    mr.mask_padding_topk_ids(ids)
    assert mab.moe_align_block_size(ids, bs, E) is stashed


def test_fused_decode_sizes_keep_the_handoff(env, monkeypatch):
    # <= 8 rows: the mask skips the call, the router's alignment is used.
    ids = _route_v2(monkeypatch, 8, seed=3)
    bs = moe_gates.marlin_block_size_m(8, TOPK, E)
    stashed = moe_gates._PENDING[3]
    env["is_padding"] = torch.arange(8) >= 5
    ref = ids.clone()
    mr.mask_padding_topk_ids(ids)
    assert torch.equal(ids, ref)
    assert mab.moe_align_block_size(ids, bs, E) is stashed


def _route_v2_in_kernel_mask(monkeypatch, m, seed, state):
    """maybe_moe_route_v2 with the in-kernel mask on: the stand-in applies
    the is_padding it is given the way the kernel does (ids -1, alignment of
    the masked ids) and records what it was passed."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]
    ).to(torch.int32)
    w = torch.full((m, TOPK), 1.0 / TOPK)
    logits = torch.randn(m, E, generator=g)
    seen = {}

    def fake_moe_route(x, weight, bias, *, topk, block_size, num_experts,
                       renormalize, routed_scaling_factor, is_padding=None):
        seen["is_padding"] = is_padding
        if is_padding is not None:
            ids.masked_fill_(is_padding.unsqueeze(1), -1)
        s, e, n = _align(ids, block_size)
        return logits, w, ids, s, e, n

    import vllm_sm80.kernels.moe.route as mrt

    monkeypatch.setattr(mrt, "moe_route", fake_moe_route)
    monkeypatch.setattr(moe_gates, "use_ampere_moe_route_v2", lambda *a: True)
    import vllm.forward_context as fc

    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        fc, "get_forward_context",
        lambda: SimpleNamespace(is_padding=state["is_padding"]),
    )
    router = SimpleNamespace(top_k=TOPK, e_score_correction_bias=SimpleNamespace(
        data=torch.zeros(E)), routed_scaling_factor=1.0)
    gate = SimpleNamespace(weight=torch.empty(E, 16))
    out = moe_gates.maybe_moe_route_v2(gate, router, torch.empty(m, 16))
    assert out is logits
    w_ids = moe_gates.take_fused_routing(logits)
    assert w_ids is not None and w_ids[1] is ids
    return ids, seen


@pytest.mark.parametrize("m", list(range(9, 33)))
def test_in_kernel_mask_keeps_the_handoff(env, monkeypatch, m):
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1")
    n_real = max(1, m - 1 - (m % 5))
    env["is_padding"] = torch.arange(m) >= n_real
    ids, seen = _route_v2_in_kernel_mask(monkeypatch, m, m, env)
    assert seen["is_padding"] is env["is_padding"]
    bs = moe_gates.marlin_block_size_m(m, TOPK, E)
    stashed = moe_gates._PENDING[3]
    before = ids.clone()
    out = mr.mask_padding_topk_ids(ids)
    assert out is ids and torch.equal(ids, before)   # nothing left to mask
    assert (ids[n_real:] == -1).all()
    got = mab.moe_align_block_size(ids, bs, E)
    assert got is stashed                             # handoff taken
    assert _same(got, _align(ids.clone(), bs), bs)    # = fresh masked align


def test_in_kernel_mask_skips_fused_decode_sizes(env, monkeypatch):
    # <= 8 rows: neither the kernel nor the mask touches the batch.
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1")
    env["is_padding"] = torch.arange(8) >= 5
    ids, seen = _route_v2_in_kernel_mask(monkeypatch, 8, 5, env)
    assert seen["is_padding"] is None
    assert (ids >= 0).all()


def test_in_kernel_mask_other_mask_object_falls_back(env, monkeypatch):
    # The mask call sees a different is_padding object than the kernel did:
    # it must not trust the handoff; it masks and drops it.
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1")
    env["is_padding"] = torch.arange(16) >= 12
    ids, _ = _route_v2_in_kernel_mask(monkeypatch, 16, 6, env)
    env["is_padding"] = torch.arange(16) >= 10
    mr.mask_padding_topk_ids(ids)
    assert moe_gates._PENDING is None
    assert (ids[10:] == -1).all()


def test_in_kernel_mask_off_when_mask_padding_off(env, monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1")
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "0")
    env["is_padding"] = torch.arange(16) >= 12
    ids, seen = _route_v2_in_kernel_mask(monkeypatch, 16, 8, env)
    assert seen["is_padding"] is None
    assert (ids >= 0).all()


def test_drop_only_touches_its_own_ids():
    a = torch.zeros(2, TOPK, dtype=torch.int32)
    b = torch.zeros(2, TOPK, dtype=torch.int32)
    moe_gates.stash_fused_align(a, 8, E, ("x",))
    assert moe_gates.drop_fused_align(b) is False
    assert moe_gates._PENDING is not None
    assert moe_gates.drop_fused_align(a) is True
    assert moe_gates._PENDING is None
    assert moe_gates.drop_fused_align(a) is False


def test_no_cuda_was_initialised():
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("in_kernel", [False, True])
def test_graph_padding_has_zero_routed_slots_in_both_marlin_paths(
    env, monkeypatch, m, in_kernel
):
    """Trace graph padding through real route dispatch and align consumers."""
    from vllm_sm80.kernels.marlin_moe import decode as marlin_decode
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")
    monkeypatch.setenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", str(int(in_kernel)))
    nreq = m // 8
    padding = torch.arange(m) >= nreq * 2
    batch = SimpleNamespace(is_padding=padding)
    env["is_padding"] = batch.is_padding
    if in_kernel:
        ids, seen = _route_v2_in_kernel_mask(monkeypatch, m, m, env)
        assert seen["is_padding"] is batch.is_padding
    else:
        ids = _route_v2(monkeypatch, m, m)
    mr.mask_padding_topk_ids(ids)
    assert (ids[batch.is_padding] == -1).all()
    assert (ids[~batch.is_padding] >= 0).all()
    # Released Marlin obtains its work list at this entry point.
    released = mab.moe_align_block_size(ids, 8, E)
    slots = released[0][: int(released[2])]
    valid = slots[slots < ids.numel()].long()
    assert valid.numel() == nreq * 2 * TOPK
    assert not batch.is_padding[valid // TOPK].any()

    # Compiled decode consumes the same masked alignment without rebuilding.
    moe_gates.stash_fused_align(ids, 8, E, released, is_padding=batch.is_padding)
    launches = []

    def gemm(*args):
        sorted_ids, experts, ntpp = args[3:6]
        assert sorted_ids is released[0] and experts is released[1]
        used = sorted_ids[: int(ntpp)]
        used = used[used < ids.numel()].long()
        launches.append(used)
        return 1

    ops = SimpleNamespace(decode_gemm_orig=gemm, decode_act_orig=lambda *a: None)
    monkeypatch.setattr(
        "vllm_sm80.kernels.marlin_moe.extension.require_extension", lambda: ops
    )
    monkeypatch.setattr(
        marlin_decode,
        "_workspaces",
        lambda *a, **kw: {
            "h": torch.empty(m * TOPK * 512),
            "c3": torch.empty(m * TOPK * 4096),
            "ctr": None,
            "part": None,
        },
    )
    layer = SimpleNamespace(
        w1_scale=None,
        w2_scale=None,
        activation_config=SimpleNamespace(clamp_limit=10.0),
        moe_sum=lambda *a: None,
    )
    marlin_decode.run(
        layer,
        torch.empty(m, 4096),
        torch.empty(m, 4096),
        None,
        torch.empty(1, 32),
        torch.ones(m, TOPK),
        ids,
        None,
    )
    assert len(launches) == 2
    for used in launches:
        assert used.numel() == nreq * 2 * TOPK
        assert not batch.is_padding[used // TOPK].any()
    assert not torch.cuda.is_initialized()


def _check_interpreted_router_tile():
    @triton.jit
    def cpu_tanh(x):
        return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

    # The interpreter cannot call libdevice. This is exact at zero logits;
    # routing numerics are outside this mask test.
    router_module.libdevice = SimpleNamespace(tanh=cpu_tanh)

    @triton.jit
    def route(logits, bias, weights, ids, padding, keep_out):
        m = tl.program_id(0)
        _, keep = router_module._route_tile(
            logits,
            bias,
            weights,
            ids,
            m,
            288,
            1,
            2.5,
            M=4,
            E=288,
            TOPK=8,
            RENORM=True,
            BLOCK_M=1,
            BLOCK_K=8,
            EA=256,
            EB=32,
            BITONIC=True,
            pad_ptr=padding,
            HAS_PAD=True,
        )
        tl.store(keep_out + m * 8 + tl.arange(0, 8), keep.reshape([8]))

    batch = SimpleNamespace(is_padding=torch.tensor([False, False, True, True]))
    ids = torch.empty(4, TOPK, dtype=torch.int32)
    keep = torch.empty(4, TOPK, dtype=torch.bool)
    route[(4,)](
        torch.zeros(4, E),
        torch.zeros(E),
        torch.empty(4, TOPK),
        ids,
        batch.is_padding,
        keep,
    )
    assert (ids[2:] == -1).all() and not keep[2:].any()
    assert (ids[:2] >= 0).all() and keep[:2].all()
    assert not torch.cuda.is_initialized()


def test_interpreted_router_tile_drops_padding():
    import os
    import subprocess
    import sys

    code = (
        f"import runpy; runpy.run_path({__file__!r})"
        "['_check_interpreted_router_tile']()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
