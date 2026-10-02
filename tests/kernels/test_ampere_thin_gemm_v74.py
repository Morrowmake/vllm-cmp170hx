# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the 74-SM thin GEMM schedule (vllm/ampere_thin_gemm/thin_gemm_v74.py,
VLLM_GLM5_THIN_GEMM_V74, on by default; 0 is the kill switch).

CPU (runs anywhere):
    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_thin_gemm_v74.py
the flag; ``impl()`` resolves to thin_gemm_v74 when on and to thin_gemm (the
unchanged module) when off, and every thin-GEMM caller (dispatch, warmup, the
PP draft-tail workspace lane, the indexer's merged GEMM) goes through it; the
schedule table differs from thin_gemm's in exactly the re-swept rows; the
rotation predicate; with no rotation and no split-K the v74 kernel compiles to
the same SASS as thin_gemm's (and differs once rotation is on).
GPU (sm_80, skipped otherwise): every production (N, K) at M 1..32 against an
FP64 recomputation: summed mean error <= 1.10x thin_gemm's and max error <=
1.25x thin_gemm's (the 74-SM schedule reorders the K reduction, so single
elements may move by more than one bf16 ulp), bitwise equal to thin_gemm where
neither the config nor the rotation changed; bitwise run to run and out= == out=None; CUDA-graph
replay == eager with zero allocation growth; the merged indexer GEMM's bf16
columns stay bitwise the v74 thin GEMM's.
"""

import contextlib
import types

import pytest
import torch

from vllm import envs

FLAG = "VLLM_GLM5_THIN_GEMM_V74"


@pytest.fixture
def fresh_impl():
    import vllm.ampere_thin_gemm as atg

    atg.impl.cache_clear()
    yield atg
    # a test may have replaced impl; the next setup clears the real one
    getattr(atg.impl, "cache_clear", lambda: None)()


def _mods():
    from vllm.ampere_thin_gemm import thin_gemm as v1
    from vllm.ampere_thin_gemm import thin_gemm_v74 as v74

    return v1, v74


# ------------------------------------------------------------------ CPU ----

def test_flag_defaults_on_and_kill_switch(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert envs.VLLM_GLM5_THIN_GEMM_V74 is True
    monkeypatch.setenv(FLAG, "0")
    assert envs.VLLM_GLM5_THIN_GEMM_V74 is False


@pytest.mark.parametrize("flag,name", [("1", "thin_gemm_v74"), ("0", "thin_gemm")])
def test_impl_follows_the_flag(monkeypatch, fresh_impl, flag, name):
    monkeypatch.setenv(FLAG, flag)
    mod = fresh_impl.impl()
    assert mod.__name__ == f"vllm.ampere_thin_gemm.{name}"
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
    from vllm.ampere_thin_gemm import warmup as wu
    from vllm.v1.worker.gpu import pp_draft_tail as tail

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
    monkeypatch.setattr(envs, "VLLM_GLM5_THIN_GEMM", True, raising=False)
    monkeypatch.setattr(tail, "private_flashinfer_topk_workspace",
                        lambda device, holder: contextlib.nullcontext())
    with tail.tail_side_stream_workspaces("cpu", {}):
        pass
    assert seen[-1] == ("lane", tail.DRAFT_TAIL_THIN_GEMM_LANE)


def test_merged_indexer_gemm_follows_impl(monkeypatch, fresh_impl):
    from vllm.ampere_decode.idx_glue import thin_gemm_dual_supported

    x = torch.empty(4, 4096, dtype=torch.bfloat16)
    w = torch.empty(160, 4096, dtype=torch.bfloat16)
    seen = []
    v1, _ = _mods()

    def sel(M, N, K):
        seen.append((M, N, K))
        return v1._select_config(M, N, K)

    monkeypatch.setattr(fresh_impl, "impl", lambda: types.SimpleNamespace(_select_config=sel))
    assert thin_gemm_dual_supported(x, w, 128)
    assert seen == [(4, 160, 4096)]


# The (N, K, M) rows the 74-SM schedule re-swept; every other row is thin_gemm's.
CHANGED = {
    (6416, 4096, 4), (6416, 4096, 16), (6144, 4096, 4), (4096, 3072, 16),
    (2048, 4096, 24), (4096, 2048, 4), (1024, 4096, 16), (1024, 4096, 32),
    (4096, 512, 24), (4096, 512, 32), (4096, 20480, 4), (4096, 20480, 16),
    (38720, 4096, 1), (38720, 4096, 2), (38720, 4096, 3), (38720, 4096, 4),
    (38720, 4096, 8), (38720, 4096, 12), (38720, 4096, 16), (38720, 4096, 18),
    (38720, 4096, 24), (38720, 4096, 32), (1536, 4096, 4), (1536, 4096, 16),
    (1536, 4096, 24), (1536, 4096, 32), (4096, 1024, 24), (154880, 4096, 4),
    (154880, 4096, 8), (24896, 4096, 4), (24576, 4096, 8), (4096, 16384, 4),
    (4096, 16384, 8), (4096, 12288, 8), (4096, 12288, 16), (4096, 8192, 4),
    (4096, 8192, 8),
}


def test_schedule_differs_only_in_the_reswept_rows(monkeypatch):
    v1, v74 = _mods()
    monkeypatch.setattr(v1, "_num_sms_cache", 74)
    monkeypatch.setattr(v74, "_num_sms_cache", 74)
    a, b = v1._CONFIG_OVERRIDES, v74._CONFIG_OVERRIDES
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == CHANGED
    # heuristic and every other helper unchanged
    for M in (1, 4, 16, 32):
        for N, K in ((160, 4096), (777, 4096), (4096, 200)):
            if (N, K, M) not in CHANGED:
                assert v1._heuristic(M, N, K, 16 if M <= 16 else 32) == \
                    v74._heuristic(M, N, K, 16 if M <= 16 else 32)


def test_rotation_predicate():
    _, v74 = _mods()
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
def test_unrotated_v74_kernel_is_thin_gemms_sass(cfg):
    from tests.kernels import ampere_sass

    if not ampere_sass.available():
        pytest.skip("no cuobjdump")
    v1, v74 = _mods()
    bm, bn, bk, warps, stages, even_n = cfg
    common = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=1, EVEN_K=True,
                  EVEN_N=even_n, TILED_M=False)
    a = _sass(v1._thin_gemm_kernel, common, warps, stages)
    b = _sass(v74._thin_gemm_kernel, dict(common, ROTATE_K=False), warps, stages)
    assert len(a.splitlines()) > 100 and a == b
    assert _sass(v74._thin_gemm_kernel, dict(common, ROTATE_K=True), warps, stages) != a


# ------------------------------------------------------------------ GPU ----

def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 0)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")

NK = sorted({(n, k) for n, k, _ in CHANGED} | {(4096, 4096), (160, 4096), (128, 4096),
                                               (2048, 128), (8192, 512), (288, 4096),
                                               (2560, 4096), (4096, 1536), (10240, 4096)})
MS = (1, 2, 3, 4, 5, 8, 12, 16, 17, 18, 24, 32)


def _xw(M, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g) * K ** -0.5).to(torch.bfloat16)
    return x.cuda(), w.cuda(), x.double() @ w.double().t()


@gpu
@pytest.mark.parametrize("nk", NK)
def test_gpu_accuracy_vs_thin_gemm(nk):
    v1, v74 = _mods()
    N, K = nk
    v1.warmup([nk], MS)
    v74.warmup([nk], MS)
    se1 = se74 = mx1 = mx74 = 0.0
    for M in MS:
        x, w, ref = _xw(M, N, K, seed=N + K + M)
        a = v1.thin_gemm(x, w)
        b = v74.thin_gemm(x, w)
        b2 = v74.thin_gemm(x, w)
        out = torch.full_like(b, float("nan"))
        v74.thin_gemm(x, w, out=out)
        torch.cuda.synchronize()
        assert torch.equal(b, b2) and torch.equal(b, out), (M, "run to run / out=")
        assert torch.isfinite(b).all()
        bm, bn, bk, sk, _, _ = v74._select_config(M, N, K)
        same_cfg = v1._select_config(M, N, K) == v74._select_config(M, N, K)
        even_k = K % (bk * sk) == 0
        if same_cfg and sk == 1 and not v74._rotate_k(even_k, sk, bm, bn):
            assert torch.equal(a, b), (M, "unchanged config must be bitwise")
        ref = ref.cpu()
        e1 = (a.double().cpu() - ref).abs()
        e74 = (b.double().cpu() - ref).abs()
        se1 += float(e1.mean())
        se74 += float(e74.mean())
        mx1, mx74 = max(mx1, float(e1.max())), max(mx74, float(e74.max()))
    assert se74 <= 1.10 * se1, (nk, "mean", se74 / se1)
    assert mx74 <= 1.25 * mx1, (nk, "max", mx74 / mx1)


@gpu
@pytest.mark.parametrize("M", [4, 16, 24, 32])
def test_gpu_graph_replay(M):
    _, v74 = _mods()
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
    assert all(torch.equal(a, b) for a, b in zip(outs, eager))
    assert (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) == (a0, r0)


@gpu
@pytest.mark.parametrize("flag", ["1", "0"])
def test_gpu_merged_indexer_gemm_matches_the_thin_gemm(monkeypatch, fresh_impl, flag):
    from vllm.ampere_decode.idx_glue import thin_gemm_dual

    monkeypatch.setenv(FLAG, flag)
    mod = fresh_impl.impl()
    for M in (1, 2, 4, 8, 16, 24, 32):
        x, w, _ = _xw(M, 160, 4096, seed=M)
        lo, _ = thin_gemm_dual(x, w, 128)
        full = mod.thin_gemm(x, w)
        torch.cuda.synchronize()
        assert torch.equal(lo, full[:, :128]), M
