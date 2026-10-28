# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the 74-SM thin GEMM (vllm_sm80/kernels/thin_gemm/thin_gemm_v74.py).

CPU (runs anywhere):
    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_thin_gemm_v74.py
``impl()`` resolves to thin_gemm_v74 once, and every thin-GEMM caller
(dispatch, warmup, the PP draft-tail workspace lane, the indexer's merged GEMM)
goes through it; every schedule row is a well-formed config that
``_select_config`` returns as measured, and the heuristic is pinned at 74 SMs;
the rotation predicate; the rotated kernel compiles to different SASS than the
unrotated one.
GPU (sm_80, skipped otherwise): every production (N, K) at M 1..32 against an
FP64 recomputation: summed mean error <= 1.10x cuBLAS's and max error <= 1.25x
cuBLAS's (both are dominated by the final bf16 rounding); bitwise run to run
and out= == out=None; CUDA-graph replay == eager with zero allocation growth;
the merged indexer GEMM's bf16 columns stay bitwise the thin GEMM's.
"""

import contextlib
import types

import pytest
import torch

from vllm import envs


@pytest.fixture
def fresh_impl():
    import vllm_sm80.kernels.thin_gemm as atg

    atg.impl.cache_clear()
    yield atg
    # a test may have replaced impl; the next setup clears the real one
    getattr(atg.impl, "cache_clear", lambda: None)()


def _v74():
    from vllm_sm80.kernels.thin_gemm import thin_gemm_v74

    return thin_gemm_v74


# ------------------------------------------------------------------ CPU ----

def test_impl_is_v74_and_resolved_once(fresh_impl):
    mod = fresh_impl.impl()
    assert mod.__name__ == "vllm_sm80.kernels.thin_gemm.thin_gemm_v74"
    assert fresh_impl.impl() is mod  # resolved once
    for attr in ("thin_gemm", "warmup", "workspace_lane", "_select_config",
                 "_partials", "_locks", "_dummy_fp32"):
        assert callable(getattr(mod, attr)), attr


def test_dispatch_goes_through_impl(monkeypatch, fresh_impl):
    calls = []
    fake = types.SimpleNamespace(thin_gemm=lambda x, w: calls.append(x.shape) or "fake")
    monkeypatch.setattr(fresh_impl, "impl", lambda: fake)
    monkeypatch.setattr(fresh_impl, "use_ampere_thin_gemm", lambda: True)
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    w = torch.zeros(1024, 4096, dtype=torch.bfloat16)
    assert fresh_impl.ampere_thin_gemm(None, x, w, None) == "fake"
    assert fresh_impl.thin_linear(x, w) == "fake"
    assert calls == [(4, 4096), (4, 4096)]


def test_warmup_and_tail_lane_go_through_impl(monkeypatch, fresh_impl):
    from vllm_sm80.kernels.thin_gemm import warmup as wu
    from vllm_sm80.worker import pp_draft_tail as tail

    seen = []

    @contextlib.contextmanager
    def lane(n):
        seen.append(("lane", n))
        yield

    fake = types.SimpleNamespace(warmup=lambda nk, ms, device=None: seen.append(("warm", tuple(ms))),
                                 workspace_lane=lane)
    monkeypatch.setattr(fresh_impl, "impl", lambda: fake)
    monkeypatch.setattr(fresh_impl, "use_ampere_thin_gemm", lambda: True)
    monkeypatch.setattr(wu, "_discover_nk", lambda worker: {(4096, 4096)})
    wu.warmup_ampere_thin_gemm(types.SimpleNamespace(device="cpu"), [1, 4, 32, 64])
    assert seen[-1][0] == "warm" and seen[-1][1][-1] == 32
    # an attribute, not the environment: another test may have frozen envs
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM", True, raising=False)
    monkeypatch.setattr(tail, "private_flashinfer_topk_workspace",
                        lambda device, holder: contextlib.nullcontext())
    with tail.tail_side_stream_workspaces("cpu", {}):
        pass
    assert seen[-1] == ("lane", tail.DRAFT_TAIL_THIN_GEMM_LANE)


def test_merged_indexer_gemm_follows_impl(monkeypatch, fresh_impl):
    from vllm_sm80.models.glm5next.ops.idx_glue import thin_gemm_dual_supported

    x = torch.empty(4, 4096, dtype=torch.bfloat16)
    w = torch.empty(160, 4096, dtype=torch.bfloat16)
    seen = []
    v74 = _v74()

    def sel(M, N, K):
        seen.append((M, N, K))
        return v74._select_config(M, N, K)

    monkeypatch.setattr(fresh_impl, "impl", lambda: types.SimpleNamespace(_select_config=sel))
    assert thin_gemm_dual_supported(x, w, 128)
    assert seen == [(4, 160, 4096)]


def _pow2(v):
    return v > 0 and v & (v - 1) == 0


def test_schedule_rows_are_selected_as_measured(monkeypatch):
    v74 = _v74()
    monkeypatch.setattr(v74, "_num_sms_cache", 74)
    for (N, K, M), row in v74._CONFIG_OVERRIDES.items():
        bn, bk, sk, warps, stages = row
        assert _pow2(bn) and _pow2(bk) and sk >= 1, (N, K, M)
        assert warps in (2, 4, 8) and 2 <= stages <= 5, (N, K, M)
        for m in ((M,) if M is not None else (5, 33)):
            if (N, K, m) in v74._CONFIG_OVERRIDES and M is None:
                continue
            bm, *cfg, st = v74._select_config(m, N, K)
            assert bm == min(64, max(16, 1 << (m - 1).bit_length())), (N, K, m)
            assert tuple(cfg) == row[:4], (N, K, m)
            # the pipeline is trimmed only as far as shared memory requires
            assert st <= stages and (st == stages or st == 2
                                     or (st + 1) * (bn + bm) * bk * 2 > v74._MAX_SMEM)


def test_heuristic_is_pinned_at_74_sms(monkeypatch):
    v74 = _v74()
    monkeypatch.setattr(v74, "_num_sms_cache", 74)
    # (N, K) with no table row: the CTA-count rule at 1.8 CTAs per SM
    assert v74._select_config(4, 777, 4096) == (16, 16, 128, 3, 2, 4)
    assert v74._select_config(4, 4096, 200) == (16, 16, 64, 1, 2, 4)
    assert v74._select_config(32, 777, 4096) == (32, 16, 128, 3, 2, 4)


def test_rotation_predicate():
    v74 = _v74()
    assert v74._rotate_k(True, 1, 32, 64)
    assert v74._rotate_k(True, 1, 16, 128)
    assert not v74._rotate_k(True, 1, 16, 64)
    assert not v74._rotate_k(True, 2, 64, 256)    # split-K never rotates
    assert not v74._rotate_k(False, 1, 64, 256)   # ragged K never rotates


def _sass_args(kernel, consts):
    args = {}
    for name in kernel.arg_names:
        if name in consts:
            args[name] = (consts[name],)
        elif name in ("X", "W", "Y"):
            args[name] = "*bf16"
        elif name in ("P", "Y2"):
            args[name] = "*fp32"
        elif name == "LOCK":
            args[name] = "*i32"
        else:
            args[name] = 4096 * 3 + 7   # a runtime int without hints
    return args


def _sass(kernel, consts, num_warps, num_stages):
    from tests.kernels import ampere_sass

    sass, _, _ = ampere_sass.compile_sm80(kernel, _sass_args(kernel, consts),
                                          num_warps=num_warps, num_stages=num_stages)
    return "\n".join(line.split("*/", 1)[-1] for line in sass.splitlines()
                     if "/*" in line and "*/" in line and ".text." not in line)


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, EVEN_N)
@pytest.mark.parametrize("cfg", [(16, 32, 128, 2, 3, True), (32, 64, 128, 4, 4, True),
                                 (16, 64, 64, 8, 5, False)])
def test_rotation_changes_the_compiled_kernel(cfg):
    from tests.kernels import ampere_sass

    if not ampere_sass.available():
        pytest.skip("no cuobjdump")
    v74 = _v74()
    bm, bn, bk, warps, stages, even_n = cfg
    common = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=1, EVEN_K=True,
                  EVEN_N=even_n, TILED_M=False)
    a = _sass(v74._thin_gemm_kernel, dict(common, ROTATE_K=False), warps, stages)
    b = _sass(v74._thin_gemm_kernel, dict(common, ROTATE_K=True), warps, stages)
    assert len(a.splitlines()) > 100 and len(b.splitlines()) > 100
    assert a != b


# ------------------------------------------------------------------ GPU ----

def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 0)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")

# Every (N, K) with a schedule row (TP4 and PP4) plus the TP4 shapes that run
# on the heuristic or an any-M row only.
NK = sorted({
    (6416, 4096), (6144, 4096), (4096, 4096), (4096, 3072), (2048, 4096),
    (4096, 2048), (4096, 1536), (1024, 4096), (8192, 512), (4096, 512),
    (288, 4096), (160, 4096), (128, 4096), (4096, 20480), (38720, 4096),
    (1536, 4096), (4096, 1024), (2048, 128), (2560, 4096), (154880, 4096),
    (24896, 4096), (24576, 4096), (4096, 16384), (4096, 12288), (4096, 8192),
    (10240, 4096),
})
MS = (1, 2, 3, 4, 5, 8, 12, 16, 17, 18, 24, 32)


def _xw(M, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g) * K ** -0.5).to(torch.bfloat16)
    return x.cuda(), w.cuda(), x.double() @ w.double().t()


@gpu
@pytest.mark.parametrize("nk", NK)
def test_gpu_accuracy_vs_fp64(nk):
    v74 = _v74()
    N, K = nk
    v74.warmup([nk], MS)
    se_cublas = se74 = mx_cublas = mx74 = 0.0
    for M in MS:
        x, w, ref = _xw(M, N, K, seed=N + K + M)
        a = torch.nn.functional.linear(x, w)
        b = v74.thin_gemm(x, w)
        b2 = v74.thin_gemm(x, w)
        out = torch.full_like(b, float("nan"))
        v74.thin_gemm(x, w, out=out)
        torch.cuda.synchronize()
        assert torch.equal(b, b2) and torch.equal(b, out), (M, "run to run / out=")
        assert torch.isfinite(b).all()
        ref = ref.cpu()
        e_cublas = (a.double().cpu() - ref).abs()
        e74 = (b.double().cpu() - ref).abs()
        se_cublas += float(e_cublas.mean())
        se74 += float(e74.mean())
        mx_cublas = max(mx_cublas, float(e_cublas.max()))
        mx74 = max(mx74, float(e74.max()))
    assert se74 <= 1.10 * se_cublas, (nk, "mean", se74 / se_cublas)
    assert mx74 <= 1.25 * mx_cublas, (nk, "max", mx74 / mx_cublas)


@gpu
@pytest.mark.parametrize("M", [4, 16, 24, 32])
def test_gpu_graph_replay(M):
    v74 = _v74()
    shapes = [(6416, 4096), (38720, 4096), (2048, 4096), (4096, 20480)]
    v74.warmup(shapes, [M])
    cases = [_xw(M, N, K, seed=M + N) for N, K in shapes]
    eager = [v74.thin_gemm(x, w).clone() for x, w, _ in cases]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        outs = [v74.thin_gemm(x, w) for x, w, _ in cases]
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    # Read before the comparison: torch.equal allocates its eq and all()
    # temporaries, and whether they need a new allocator segment depends on
    # what earlier tests in the process left in the pool.
    grown = (torch.cuda.memory_allocated() - a0, torch.cuda.memory_reserved() - r0)
    assert all(torch.equal(a, b) for a, b in zip(outs, eager))
    assert grown == (0, 0), M


@gpu
def test_gpu_merged_indexer_gemm_matches_the_thin_gemm(fresh_impl):
    from vllm_sm80.models.glm5next.ops.idx_glue import thin_gemm_dual

    mod = fresh_impl.impl()
    for M in (1, 2, 4, 8, 16, 24, 32):
        x, w, _ = _xw(M, 160, 4096, seed=M)
        lo, _ = thin_gemm_dual(x, w, 128)
        full = mod.thin_gemm(x, w)
        torch.cuda.synchronize()
        assert torch.equal(lo, full[:, :128]), M
