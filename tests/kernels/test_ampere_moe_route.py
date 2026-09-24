# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sm_80 fused MoE router (vllm/ampere_decode/moe_route.py).

The gate, the flags and the routing/alignment handoff are tested on CPU. The
kernel itself is tested on an sm_80 GPU: logits against an FP64 reference, the
top-k against FP64 keys with the fp32 tie margin, the alignment against a
stable-sort reference, bitwise determinism, CUDA-graph replay, and agreement
with the production path (torch.mm + the grouped top-k + moe_align_block_size).
"""

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

E, TOPK, HIDDEN = 288, 8, 4096
TIE_DELTA = 2.0**-17


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MOE_ROUTE_V2", "1")
    monkeypatch.setattr("vllm.ampere_decode._SM80_CACHE", True)
    return monkeypatch


def _gate_router(device="cpu", **over):
    gate = SimpleNamespace(
        weight=torch.zeros(E, HIDDEN, dtype=torch.bfloat16, device=device),
        bias=None,
        out_dtype=torch.float32,
    )
    router = SimpleNamespace(
        scoring_func="sigmoid", num_expert_group=1, topk_group=1,
        renormalize=True, top_k=TOPK, num_fused_shared_experts=0,
        skip_padding=False, eplb_state=None, routed_scaling_factor=2.5,
        e_score_correction_bias=torch.zeros(E, dtype=torch.float32,
                                            device=device),
    )
    for k, v in over.items():
        setattr(gate if k in ("weight", "out_dtype") else router, k, v)
    return gate, router


def _x(m, device="cpu"):
    return torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=device)


# --- CPU: flags, gate, handoff --------------------------------------------------
def test_flags_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_GLM5_DECODE_MOE_ROUTE_V2", raising=False)
    monkeypatch.delenv("VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS", raising=False)
    assert envs.VLLM_GLM5_DECODE_MOE_ROUTE_V2 is False
    assert envs.VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS == 32


def test_gate_off_by_default(monkeypatch):
    from vllm.ampere_decode import use_ampere_moe_route_v2

    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    monkeypatch.delenv("VLLM_GLM5_DECODE_MOE_ROUTE_V2", raising=False)
    monkeypatch.setattr("vllm.ampere_decode._SM80_CACHE", True)
    g, r = _gate_router()
    assert not use_ampere_moe_route_v2(4, g, r, _x(4))


def test_master_flag_required(on):
    from vllm.ampere_decode import use_ampere_moe_route_v2

    on.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
    g, r = _gate_router()
    assert not use_ampere_moe_route_v2(4, g, r, _x(4))


def test_gate_token_range(on):
    from vllm.ampere_decode import use_ampere_moe_route_v2

    g, r = _gate_router()
    for m in (1, 2, 4, 8, 16, 32):
        assert use_ampere_moe_route_v2(m, g, r, _x(m))
    assert not use_ampere_moe_route_v2(33, g, r, _x(33))
    on.setenv("VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS", "4")
    assert use_ampere_moe_route_v2(4, g, r, _x(4))
    assert not use_ampere_moe_route_v2(5, g, r, _x(5))


def test_gate_off_on_non_sm80(on):
    from vllm.ampere_decode import use_ampere_moe_route_v2

    on.setattr("vllm.ampere_decode._SM80_CACHE", False)
    g, r = _gate_router()
    assert not use_ampere_moe_route_v2(4, g, r, _x(4))


@pytest.mark.parametrize("over", [
    {"scoring_func": "softmax"}, {"num_expert_group": 8}, {"topk_group": 4},
    {"renormalize": False}, {"top_k": 6}, {"num_fused_shared_experts": 1},
    {"skip_padding": True}, {"eplb_state": object()},
    {"e_score_correction_bias": None},
    {"e_score_correction_bias": torch.zeros(E, dtype=torch.bfloat16)},
    {"weight": torch.zeros(E, HIDDEN, dtype=torch.float32)},
    {"weight": torch.zeros(256, HIDDEN, dtype=torch.bfloat16)},
    {"weight": torch.zeros(HIDDEN, E, dtype=torch.bfloat16).T},
    {"out_dtype": torch.bfloat16},
])
def test_gate_pins_the_validated_router(on, over):
    from vllm.ampere_decode import use_ampere_moe_route_v2

    g, r = _gate_router(**over)
    assert not use_ampere_moe_route_v2(4, g, r, _x(4))


def test_gate_rejects_gate_bias_and_odd_x(on):
    from vllm.ampere_decode import use_ampere_moe_route_v2

    g, r = _gate_router()
    g.bias = torch.zeros(E)
    assert not use_ampere_moe_route_v2(4, g, r, _x(4))
    g, r = _gate_router()
    assert not use_ampere_moe_route_v2(4, g, r, _x(4).float())
    col_strided = torch.zeros(HIDDEN, 4, dtype=torch.bfloat16).T
    assert col_strided.stride(1) != 1
    assert not use_ampere_moe_route_v2(4, g, r, col_strided)


def test_off_path_does_not_import_the_kernel(monkeypatch):
    """With the flag off, maybe_moe_route_v2 returns None before importing."""
    code = (
        "import sys, vllm.ampere_decode as a\n"
        "from types import SimpleNamespace as S\n"
        "import torch\n"
        "assert a.maybe_moe_route_v2(S(weight=None), S(), "
        "torch.zeros(4, 4096, dtype=torch.bfloat16)) is None\n"
        "assert 'vllm.ampere_decode.moe_route' not in sys.modules\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    env = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
           "VLLM_GLM5_DECODE_KERNELS": "1"}
    import os

    env.update({k: v for k, v in os.environ.items()
                if k in ("HOME", "PYTHONPATH", "VIRTUAL_ENV")})
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_routing_handoff_is_identity_keyed():
    import vllm.ampere_decode as a

    logits = torch.zeros(4, E)
    routing = (torch.zeros(4, TOPK), torch.zeros(4, TOPK, dtype=torch.int32))
    a._PENDING_ROUTING = (logits, routing)
    assert a.take_fused_routing(logits.clone()) is None
    assert a.take_fused_routing(logits) is routing
    assert a.take_fused_routing(logits) is None


def test_import_does_not_initialise_cuda():
    code = ("import torch, vllm.ampere_decode.moe_route as m\n"
            "assert not torch.cuda.is_initialized()\n"
            "assert m.launches(1) == 2 and m.launches(2) == 1\n"
            "assert all(m.launches(k) == 1 for k in (4, 8, 16, 32, 64))\n"
            "assert m.launches(65) == 2\n")
    import os

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- GPU: the kernel ------------------------------------------------------------
@pytest.fixture
def sm80_gpu():
    """Skip unless an sm_80 GPU is visible. A fixture, not a collection-time
    skipif: probing the device while the module is collected would initialise
    CUDA before the CPU-only tests that assert it is not initialised."""
    if not torch.cuda.is_available() or \
            torch.cuda.get_device_capability(0) != (8, 0):
        pytest.skip("needs an sm_80 GPU")


gpu = pytest.mark.usefixtures("sm80_gpu")


def _inputs(m, seed, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(E, HIDDEN, generator=g) * 0.04).to(torch.bfloat16)
    b = (torch.randn(E, generator=g) * 0.05 + 8.0).float()
    x = (torch.randn(m, HIDDEN, generator=g) * 0.4).to(torch.bfloat16)
    return x.to(device), w.to(device), b.to(device)


def _truth(x, w, b):
    l64 = x.double() @ w.double().T
    key = 0.5 * torch.tanh(0.5 * l64) + 0.5 + b.double()
    return l64, key


def _admissible(ids, key, topk=TOPK):
    kmax = key.abs().amax(1)
    _, ex = torch.frexp(kmax)
    d = TIE_DELTA + 2.0 * torch.ldexp(torch.ones_like(kmax), (ex - 24).int())
    srt = key.sort(1, descending=True).values
    kth, nxt = srt[:, topk - 1], srt[:, topk]
    chosen = torch.zeros_like(key, dtype=torch.bool).scatter_(1, ids.long(), True)
    ok = chosen.sum(1) == topk
    ok &= ~((key > (nxt + d)[:, None]) & ~chosen).any(1)
    ok &= ~((key < (kth - d)[:, None]) & chosen).any(1)
    sel = key.gather(1, ids.long())
    ok &= (sel[:, :-1] >= sel[:, 1:] - d[:, None]).all(1)
    return ok, (kth - nxt) <= d


def _ref_align(ids, bs):
    numel = ids.numel()
    flat = ids.reshape(-1).long()
    cnt = torch.bincount(flat, minlength=E)
    pad = (cnt + bs - 1) // bs * bs
    return cnt, pad, torch.cumsum(pad, 0) - pad


def _check_align(ids, sorted_ids, expert_ids, ntpp, bs):
    numel = ids.numel()
    cnt, pad, excl = _ref_align(ids, bs)
    assert int(ntpp.item()) == int(pad.sum())
    eids = expert_ids.tolist()
    s = sorted_ids.tolist()
    flat = ids.reshape(-1).tolist()
    for e in torch.nonzero(pad).flatten().tolist():
        lo, hi = int(excl[e]), int(excl[e] + pad[e])
        assert all(v == e for v in eids[lo // bs:hi // bs])
        want = sorted(i for i, v in enumerate(flat) if v == e)
        got = sorted(v for v in s[lo:hi] if v != numel)
        assert got == want
        assert s[lo:hi].count(numel) == int(pad[e] - cnt[e])
    nb = int(pad.sum()) // bs
    assert all(v == -1 for v in eids[nb:])
    assert all(v == numel for v in s[int(pad.sum()):])


@gpu
@pytest.mark.parametrize("m", [1, 2, 3, 4, 7, 8, 16, 17, 32])
def test_kernel_exactness(m):
    from vllm.ampere_decode.moe_route import moe_route

    x, w, b = _inputs(m, 100 + m)
    logits, tw, ti, s, e, n = moe_route(x, w, b, topk=TOPK, block_size=8,
                                        num_experts=E)
    torch.cuda.synchronize()
    l64, key = _truth(x, w, b)
    inc = torch.mm(x, w.T, out_dtype=torch.float32)
    k_err = float((logits.double() - l64).abs().max())
    i_err = float((inc.double() - l64).abs().max())
    floor = 16 * 2.0**-24 * max(float(l64.abs().max()), 1.0)
    assert k_err <= max(floor, 1.5 * i_err)
    ok, _ = _admissible(ti, key)
    assert bool(ok.all())
    sc = 0.5 * torch.tanh(0.5 * l64.gather(1, ti.long())) + 0.5
    w64 = sc / (sc.sum(1, keepdim=True) + 1e-20) * 2.5
    assert float((tw.double() - w64).abs().max()) < 1e-6
    _check_align(ti, s, e, n, 8)


@gpu
@pytest.mark.parametrize("m", [4, 16, 32])
def test_matches_production_path(m):
    from vllm.ampere_decode.moe_route import moe_route
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        fused_grouped_topk,
    )

    x, w, b = _inputs(m, 7 * m)
    _, tw, ti, *_ = moe_route(x, w, b, topk=TOPK, block_size=8, num_experts=E)
    logits = torch.mm(x, w.T, out_dtype=torch.float32)
    iw, ii = fused_grouped_topk(
        hidden_states=x, gating_output=logits, topk=TOPK, renormalize=True,
        e_score_correction_bias=b, num_expert_group=1, topk_group=1,
        scoring_func="sigmoid", routed_scaling_factor=2.5)
    torch.cuda.synchronize()
    _, key = _truth(x, w, b)
    _, near = _admissible(ti, key)
    same = (ti.long().sort(1).values == ii.long().sort(1).values).all(1)
    assert bool((same | near).all())
    rows = same
    a = tw.gather(1, ti.long().argsort(1))[rows]
    c = iw.gather(1, ii.long().argsort(1))[rows]
    torch.testing.assert_close(a, c, rtol=2.0**-6, atol=2e-6)


@gpu
@pytest.mark.parametrize("m", [1, 4, 16, 32])
def test_deterministic_and_graph_replay(m):
    from vllm.ampere_decode.moe_route import align_sizes, moe_route, warmup

    warmup((m,), (8,), topk=TOPK, num_experts=E, hidden=HIDDEN)
    x, w, b = _inputs(m, 900 + m)
    first = [t.clone() for t in moe_route(x, w, b, topk=TOPK, block_size=8)]
    for _ in range(2):
        again = moe_route(x, w, b, topk=TOPK, block_size=8)
        assert all(torch.equal(p, q) for p, q in zip(first, again))
    mnp, nblk = align_sizes(m * TOPK, E, 8)
    out = (torch.empty(m, E, device="cuda"), torch.empty(m, TOPK, device="cuda"),
           torch.empty(m, TOPK, dtype=torch.int32, device="cuda"),
           torch.empty(mnp, dtype=torch.int32, device="cuda"),
           torch.empty(nblk, dtype=torch.int32, device="cuda"),
           torch.empty(1, dtype=torch.int32, device="cuda"))
    xin = x.clone()
    moe_route(xin, w, b, topk=TOPK, block_size=8, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_route(xin, w, b, topk=TOPK, block_size=8, out=out)
    for t in out:
        t.fill_(-7)
    m0 = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == m0
    assert all(torch.equal(p, q) for p, q in zip(first, out))
    x2, _, _ = _inputs(m, 901 + m)
    xin.copy_(x2)
    graph.replay()
    fresh = moe_route(x2, w, b, topk=TOPK, block_size=8)
    torch.cuda.synchronize()
    assert all(torch.equal(p, q) for p, q in zip(fresh, out))


@gpu
def test_end_to_end_handoff(on):
    """The gate call site returns the logits and stashes the routing for the
    router and the alignment for the Marlin experts."""
    import vllm.ampere_decode as a

    m = 4
    x, w, b = _inputs(m, 5)
    gate, router = _gate_router(device="cuda")
    gate.weight = w
    router.e_score_correction_bias = b
    logits = a.maybe_moe_route_v2(gate, router, x)
    assert logits is not None and logits.shape == (m, E)
    tw, ti = a.take_fused_routing(logits)
    aligned = a.take_fused_align(ti, a.marlin_block_size_m(m, TOPK, E), E,
                                 None, False)
    assert aligned is not None
    torch.cuda.synchronize()
    _, key = _truth(x, w, b)
    assert bool(_admissible(ti, key)[0].all())
    _check_align(ti, *aligned, 8)
    assert math.isfinite(float(tw.sum()))
