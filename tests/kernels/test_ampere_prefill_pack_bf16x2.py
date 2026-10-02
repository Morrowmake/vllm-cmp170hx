# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired bf16 rounding in the sm_80 prefill mHC pre-norm pack.

``VLLM_GLM5_PREFILL_PACK_BF16X2=1`` makes ``hc_prenorm_gemm`` pack ``fn`` with
``_pack_fn_x2_kernel``, which rounds fp32 to bf16 two elements per
``cvt.rn.bf16x2.f32`` instead of one per scalar ``cvt.rn.bf16.f32``. Both are
round-to-nearest-even, so the packed hi/mid/lo terms, and everything computed
from them, must be bitwise identical.

CPU tests (``CUDA_VISIBLE_DEVICES=""``): the flag, the dispatch with the flag
off and on, the banner through the real logger, and the sm_80 SASS of both
pack kernels (the scalar narrowing is gone). GPU tests (skip without an sm_80
device): pack and end-to-end on vs off bitwise on production shapes, rounding
ties and subnormals included, and CUDA-graph replay.
"""

import logging

import pytest
import torch

from tests.kernels import ampere_sass
from vllm.ampere_prefill import mhc_prenorm as mp

FLAG = "VLLM_GLM5_PREFILL_PACK_BF16X2"
K, N, BLOCK_N, PACK_K = 16384, 24, 32, 128
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")
needs_cuobjdump = pytest.mark.skipif(not ampere_sass.available(),
                                     reason="needs Triton's cuobjdump")


# --------------------------------------------------------------------- CPU
def test_flag_declared_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv(FLAG, raising=False)
    assert FLAG in envs.environment_variables
    assert envs.environment_variables[FLAG]() is False
    monkeypatch.setenv(FLAG, "1")
    assert envs.environment_variables[FLAG]() is True


def test_dispatch_follows_the_flag(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert mp._pack_bf16x2() is False
    monkeypatch.setenv(FLAG, "1")
    assert mp._pack_bf16x2() is True


def _banners(fn):
    """Messages logged by the real vLLM logger of mhc_prenorm during fn()."""
    import vllm.logger as vlog
    from vllm.logger import init_logger

    seen = []

    class Rec(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    logger = init_logger(mp.__name__)
    h = Rec(level=logging.DEBUG)
    old = logger.level
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    vlog._print_info_once.cache_clear()
    try:
        fn()
    finally:
        logger.removeHandler(h)
        logger.setLevel(old)
    return seen


def test_banner_goes_through_the_vllm_logger(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    seen = _banners(mp._pack_bf16x2)
    assert any("paired bf16 rounding" in m and FLAG + "=1" in m for m in seen), seen
    monkeypatch.delenv(FLAG, raising=False)
    assert _banners(mp._pack_bf16x2) == []


class _Launch:
    """Records which kernel object hc_prenorm_gemm launches."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        def run(*a, **k):
            self.log.append((self.name, k.get("BLOCK_N"), k.get("BLOCK_K")))
        return run


@pytest.mark.parametrize("on", [False, True])
def test_hc_prenorm_gemm_launches_the_selected_pack(monkeypatch, on):
    log = []
    monkeypatch.setattr(mp, "_pack_fn_kernel", _Launch("base", log))
    monkeypatch.setattr(mp, "_pack_fn_x2_kernel", _Launch("x2", log))
    monkeypatch.setattr(mp, "_prenorm_gemm_kernel", _Launch("gemm", log))
    monkeypatch.setattr(mp, "_prenorm_reduce_kernel", _Launch("reduce", log))
    monkeypatch.setattr(mp, "num_sms", lambda *_: 74)
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VLLM_GLM5_PRENORM_REDUCE_FP64", raising=False)
    x = torch.empty(1728, K, dtype=torch.bfloat16, device="meta")
    fn = torch.empty(N, K, dtype=torch.float32, device="meta")
    mp._FN_PACK.clear()
    mp._WORKSPACE.clear()
    mp.hc_prenorm_gemm(x, fn)
    assert log[0] == ("x2" if on else "base", BLOCK_N, PACK_K)
    assert [e[0] for e in log[1:]] == ["gemm", "reduce"]


def _pack_args():
    return dict(fn_ptr="*fp32", ft_ptr="*bf16", K=K, N=N, stride_fn_n=K,
                stride_fn_k=1, stride_tt=K * BLOCK_N, stride_tk=BLOCK_N,
                stride_tn=1, BLOCK_N=(BLOCK_N,), BLOCK_K=(PACK_K,))


@needs_cuobjdump
def test_sass_scalar_narrowing_gone(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _, _, base = ampere_sass.compile_sm80(mp._pack_fn_kernel, _pack_args())
    _, ptx, x2 = ampere_sass.compile_sm80(mp._pack_fn_x2_kernel, _pack_args())
    # 32 elements per thread x 3 roundings = 96 conversions
    assert ampere_sass.count(base, "F2F") > 0
    assert ampere_sass.count(x2, "F2F") == 0
    assert ampere_sass.count(x2, "F2FP") == 48
    assert "cvt.rn.bf16.f32" not in ptx
    assert ptx.count("cvt.rn.bf16x2.f32") == 48
    for ops in (base, x2):
        assert ampere_sass.count(ops, "LDL") == ampere_sass.count(ops, "STL") == 0


def test_rne_reference_matches_torch():
    """The rounding both PTX forms implement is IEEE round-to-nearest-even;
    torch's fp32 -> bf16 cast is the CPU reference used by the GPU tests."""
    bits = torch.tensor([0x3F808000, 0x3F818000, 0x3F80FFFF, 0x00008000,
                         0x00018000, 0x7F7FFFFF, 0x80008000], dtype=torch.int64)
    f = bits.to(torch.int32).view(torch.float32)
    got = f.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    want = torch.tensor([0x3F80, 0x3F82, 0x3F81, 0x0000, 0x0002, 0x7F80, 0x8000])
    assert torch.equal(got, want)


# --------------------------------------------------------------------- GPU
def _fn_cases(device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    yield "randn", torch.randn(N, K, generator=g, device=device) * 0.05
    # every value an exact bf16 rounding tie (both parities, both signs,
    # finite: exponent field < 0xFF), then fp32 subnormals of both signs
    hi = torch.randint(0, 0x7F80, (N, K), generator=g, device=device,
                       dtype=torch.int32)
    sign = torch.randint(0, 2, (N, K), generator=g, device=device,
                         dtype=torch.int32) << 31
    tie = (sign | (hi << 16) | 0x8000).view(torch.float32)
    yield "ties", tie
    sub = (sign | torch.randint(1, 1 << 23, (N, K), generator=g, device=device,
                                dtype=torch.int32)).view(torch.float32)
    yield "subnormal", sub
    mixed = torch.randn(N, K, generator=g, device=device) * torch.exp2(
        torch.randint(-140, 120, (N, K), generator=g, device=device).float())
    yield "wide", mixed


def _pack(kernel, fn):
    ft = torch.full((3, K, BLOCK_N), float("nan"), dtype=torch.bfloat16,
                    device=fn.device)
    kernel[(-(-K // PACK_K),)](
        fn, ft, K, N, fn.stride(0), fn.stride(1),
        ft.stride(0), ft.stride(1), ft.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=PACK_K, num_warps=4, num_stages=1)
    return ft


@needs_sm80
def test_pack_bitwise_on_equals_off():
    for name, fn in _fn_cases():
        a = _pack(mp._pack_fn_kernel, fn)
        b = _pack(mp._pack_fn_x2_kernel, fn)
        torch.cuda.synchronize()
        assert torch.equal(a[:, :, :N].view(torch.int16),
                           b[:, :, :N].view(torch.int16)), name
        if name not in ("randn", "ties"):
            continue
        # and both are the CPU split
        hi = fn.cpu().to(torch.bfloat16)
        r1 = fn.cpu() - hi.float()
        mid = r1.to(torch.bfloat16)
        lo = (r1 - mid.float()).to(torch.bfloat16)
        ref = torch.stack([hi, mid, lo]).transpose(1, 2)
        assert torch.equal(b[:, :, :N].cpu().view(torch.int16),
                           ref.contiguous().view(torch.int16)), name


@needs_sm80
@pytest.mark.parametrize("M", [512, 1152, 1728, 3456])
def test_hc_prenorm_gemm_on_equals_off(monkeypatch, M):
    g = torch.Generator(device="cuda").manual_seed(M)
    x = (torch.randn(M, K, generator=g, device="cuda")).to(torch.bfloat16)
    fn = torch.randn(N, K, generator=g, device="cuda") * 0.05
    monkeypatch.delenv(FLAG, raising=False)
    o0, s0 = mp.hc_prenorm_gemm(x, fn)
    o0, s0 = o0.clone(), s0.clone()
    monkeypatch.setenv(FLAG, "1")
    o1, s1 = mp.hc_prenorm_gemm(x, fn)
    torch.cuda.synchronize()
    assert torch.equal(o0, o1) and torch.equal(s0, s1)


@needs_sm80
def test_graph_replay_equals_eager(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    M = 1728
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16)
    fn = torch.randn(N, K, generator=g, device="cuda") * 0.05
    out = torch.empty(1, M, N, device="cuda")
    sq = torch.empty(1, M, device="cuda")
    mp.hc_prenorm_gemm(x, fn, out=out, sqrsum=sq)
    torch.cuda.synchronize()
    eager = (out.clone(), sq.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mp.hc_prenorm_gemm(x, fn, out=out, sqrsum=sq)
    out.fill_(float("nan"))
    sq.fill_(float("nan"))
    before = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager[0]) and torch.equal(sq, eager[1])
    assert torch.cuda.memory_allocated() == before
