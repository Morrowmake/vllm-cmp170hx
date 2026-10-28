# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired bf16 rounding in the sm_80 prefill mHC pre-norm pack.

``hc_prenorm_gemm`` packs ``fn`` with ``_pack_fn_x2_kernel``, which rounds fp32
to bf16 two elements per ``cvt.rn.bf16x2.f32``. That is round-to-nearest-even,
so the packed hi/mid/lo terms are bitwise the torch fp32 -> bf16 split.

CPU tests (``CUDA_VISIBLE_DEVICES=""``): the launch order through the shared
scratch owner, and sm_80 SASS of the pack kernel (no scalar narrowing). GPU
tests (skip without an sm_80 device): the pack against the CPU split on
production shapes, rounding ties included, and CUDA-graph replay.
"""

import pytest
import torch

from tests.kernels import ampere_sass
from vllm_sm80.kernels.mhc import prenorm as mp

K, N, BLOCK_N, PACK_K = 16384, 24, 32, 128
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")
needs_cuobjdump = pytest.mark.skipif(not ampere_sass.available(),
                                     reason="needs Triton's cuobjdump")


# --------------------------------------------------------------------- CPU
class _Launch:
    """Records which kernel object hc_prenorm_gemm launches."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        def run(*a, **k):
            self.log.append((self.name, k.get("BLOCK_N"), k.get("BLOCK_K")))
        return run


def test_hc_prenorm_gemm_launches_the_pack(monkeypatch):
    log = []
    monkeypatch.setattr(mp, "_pack_fn_x2_kernel", _Launch("pack", log))
    monkeypatch.setattr(mp, "_prenorm_gemm_kernel", _Launch("gemm", log))
    monkeypatch.setattr(mp, "_prenorm_reduce_kernel", _Launch("reduce", log))
    monkeypatch.setattr(mp, "num_sms", lambda *_: 74)
    monkeypatch.delenv("VLLM_GLM5_PRENORM_REDUCE_FP64", raising=False)
    x = torch.empty(1728, K, dtype=torch.bfloat16, device="meta")
    fn = torch.empty(N, K, dtype=torch.float32, device="meta")
    owner = mp.get_scratch_owner(x.device)
    monkeypatch.setattr(owner, "pack", None)
    monkeypatch.setattr(owner, "part", None)
    monkeypatch.setattr(owner, "psq", None)
    mp.hc_prenorm_gemm(x, fn)
    assert log[0] == ("pack", BLOCK_N, PACK_K)
    assert [e[0] for e in log[1:]] == ["gemm", "reduce"]


def _pack_args():
    return dict(fn_ptr="*fp32", ft_ptr="*bf16", K=K, N=N, stride_fn_n=K,
                stride_fn_k=1, stride_tt=K * BLOCK_N, stride_tk=BLOCK_N,
                stride_tn=1, BLOCK_N=(BLOCK_N,), BLOCK_K=(PACK_K,))


@needs_cuobjdump
def test_sass_has_no_scalar_narrowing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _, ptx, x2 = ampere_sass.compile_sm80(mp._pack_fn_x2_kernel, _pack_args())
    # 32 elements per thread x 3 roundings = 96 conversions, in pairs
    assert ampere_sass.count(x2, "F2F") == 0
    assert ampere_sass.count(x2, "F2FP") == 48
    assert "cvt.rn.bf16.f32" not in ptx
    assert ptx.count("cvt.rn.bf16x2.f32") == 48
    assert ampere_sass.count(x2, "LDL") == ampere_sass.count(x2, "STL") == 0


def test_rne_reference_matches_torch():
    """The rounding cvt.rn.bf16x2.f32 implements is IEEE round-to-nearest-even;
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
    # every value an exact bf16 rounding tie (both parities, both signs).
    # Normal exponents only, and below 0x7F (hi < 0x7F00): a tie at the top
    # binade rounds hi to inf, then mid = -inf and lo = NaN, and NaN bit
    # patterns differ between the GPU and torch's CPU cast.
    hi = torch.randint(0x0080, 0x7F00, (N, K), generator=g, device=device,
                       dtype=torch.int32)
    sign = torch.randint(0, 2, (N, K), generator=g, device=device,
                         dtype=torch.int32) << 31
    yield "ties", (sign | (hi << 16) | 0x8000).view(torch.float32)


def _pack(kernel, fn):
    ft = torch.full((3, K, BLOCK_N), float("nan"), dtype=torch.bfloat16,
                    device=fn.device)
    kernel[(-(-K // PACK_K),)](
        fn, ft, K, N, fn.stride(0), fn.stride(1),
        ft.stride(0), ft.stride(1), ft.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=PACK_K, num_warps=4, num_stages=1)
    return ft


@needs_sm80
def test_pack_is_the_cpu_split():
    for name, fn in _fn_cases():
        b = _pack(mp._pack_fn_x2_kernel, fn)
        torch.cuda.synchronize()
        hi = fn.cpu().to(torch.bfloat16)
        r1 = fn.cpu() - hi.float()
        mid = r1.to(torch.bfloat16)
        lo = (r1 - mid.float()).to(torch.bfloat16)
        ref = torch.stack([hi, mid, lo]).transpose(1, 2)
        diff = int((b[:, :, :N].cpu().view(torch.int16)
                    != ref.contiguous().view(torch.int16)).sum())
        assert diff == 0, (name, "vs CPU split", diff)


@needs_sm80
def test_graph_replay_equals_eager():
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
