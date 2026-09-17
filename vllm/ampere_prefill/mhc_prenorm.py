# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored kernel: do not edit here; keep in sync with the standalone source:
# edit the standalone harness, re-run its validate.py/bench.py, then re-port.
#!/usr/bin/env python
"""The mHC pre-norm GEMM, exact bf16x3 schedule.

    hc_prenorm_gemm(x[M, K] bf16, fn[N, K] fp32, out=None, sqrsum=None)
        -> (out[1, M, N] fp32, sqrsum[1, M] fp32)

exactly equal in meaning to

    out[0]    = x.float() @ fn.t()
    sqrsum[0] = x.float().square().sum(-1)

What the incumbent does wrong (from the 2026-09-16 trace, M=1152, K=16384,
N=24): `hc_prenorm_gemm_block_m_tilelang` launches `ceil(M/2) x ceil(N/12)`
= 576 x 2 CTAs of 512 threads, and every one of those 1152 CTAs streams its
own 12 x 16384 fp32 slice of `fn` (786 KB) to serve **two** tokens.  That is
884 MB of L2 traffic to move 37.9 MB of real data - a 12:1 read
amplification - and it lands at ~390 us against a ~75 us FP32-FMA floor.

This kernel tiles (BLOCK_M tokens) x (BLOCK_K of K), pads N=24 up to 32 so
`tl.dot` is legal, splits K across CTAs to fill 70 SMs, and reduces the fp32
split-K partials with a second kernel in a fixed ascending order - never with
atomics, so the result is bitwise reproducible run to run.

PRECISION - THE MULTIPLY IS EXACT, NOT APPROXIMATE
--------------------------------------------------
`x` is bf16 and `fn` is fp32.  A bf16 x bf16 -> fp32 tensor-core MMA is exact
(8-bit significands, fp32 accumulate), so decomposing only `fn` into three
bf16 terms

    hi  = bf16(fn)
    mid = bf16(fn - hi)
    lo  = bf16(fn - hi - mid)          (24 significand bits, residual 2^-25|fn|)

and issuing three MMAs against the *unmodified* bf16 `x` reproduces the fp32
product to better than fp32's own rounding.  Nothing is truncated: `x` is
already exactly representable and `fn` is carried to more bits than fp32 has.

Measured on this part (M=1152, K=16384, N=24, max|out| ~ 14, fp32
accumulation-order band 4.2e-4, TileLang incumbent's own error vs fp64
2.4e-6):

    tl.dot(fp32, fp32, input_precision=...)     max err vs fp64
      "ieee"                 1.30e-01   <- A NO-OP on sm_80 / Triton 3.7.1: TF32
      "tf32"                 2.54e-01
      "tf32x3"               4.88e-04   (3 x TF32 terms, 101 TFLOP/s pipe)
    bf16x3 against bf16 x    2.99e-05   <- this kernel, on the 202 TFLOP/s pipe

So `"ieee"` cannot be trusted here, and bf16x3 is both 16x more accurate than
tf32x3 and issued on a pipe with twice the throughput for a third of the
operand-splitting work (`x` needs no split at all).
`validate.py::tf32_trap` is built to catch a regression to a single-pass TF32
dot; it is what failed the first version of this file.

The three bf16 terms of `fn` are built **once per call** by `_pack_fn_kernel`
into a cached [3, K, BLOCK_N] bf16 workspace, already transposed into the
layout `tl.dot` wants for its B operand.  Doing the split inside the GEMM
instead costs 47 us at M=1152 (every one of the 18 M-tiles re-loads `fn` as
fp32, re-splits it and re-transposes it); the packing kernel costs 6 us, and
the GEMM's B loads then halve in bytes as well.

WHERE THE 80 us GO, AND WHAT IS ALREADY EXHAUSTED
-------------------------------------------------
Amortised over 60 calls at M=1152 (incumbent ~400 us in the same process):
GEMM 61-65 us, pack 7 us, split-K reduce 7 us.  Inside the GEMM, the same
kernel with the dots and the sqrsum removed still costs 51 us: **it is bound
by streaming `x`** (37.7 MB at ~740 GB/s), and the dots and sqrsum only add
~10 us on top because they hide behind that stream.

The load is 1.7x off what this part can do (an isolated 1D read of the same
37.7 MB runs at 1330 GB/s / 28 us, and an isolated *tiled* read reaches
1266 GB/s at BLOCK_M=64, BLOCK_K=64), but every reshaping that speeds the load
up costs more in the B-operand path, because B traffic scales with the number
of M-tiles (9 at BLOCK_M=128, 28 MB of L2; 18 at BLOCK_M=64, 56 MB):

    BM=128 BK=32 (this file)  GEMM 61-65 us     load alone 51 us
    BM=128 BK=64              GEMM 92 us        load alone ~33 us
    BM= 64 BK=64              GEMM 87-95 us     load alone 30 us
    BM= 64 BK=32              GEMM 77 us
    BM=256 BK=32              GEMM 101-124 us

Measured and rejected, do not retry without new information:
 * a 64-wide `x` load split into two 32-wide dot operands, either by
   `reshape`+`permute`+`tl.split` or by a 3D interleaved load + `tl.split`:
   82-108 us, the in-register shuffle costs more than the load saves;
 * 2 or 4 unrolled 32-wide loads per loop iteration (more MLP, same 64 B
   requests): 71-84 us;
 * dropping the always-true row mask when M % BLOCK_M == 0: 65.8 vs 64.8 us;
 * `.cg` on or off: identical;
 * BLOCK_M/SPLIT_K rebalanced at constant CTA count (BM=16/SK=4 ... BM=256/
   SK=64): 79 us is the best of the family, small BLOCK_M loses badly (BM=16
   186 us) because B traffic multiplies;
 * an m-major [M, SPLIT_K, BLOCK_N] partial layout: reduce unchanged, GEMM 8 us
   worse.

What is left is a schedule that streams `x` at full bandwidth *and* keeps the
B-operand path at BLOCK_M=128 - i.e. something other than a k-blocked
`tl.dot` loop.
"""

import torch

from vllm.triton_utils import tl, triton  # noqa: F401  (Triton 3.7.1)

_WORKSPACE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_FN_PACK: dict[tuple, torch.Tensor] = {}


def num_sms(device=0):
    return torch.cuda.get_device_properties(device).multi_processor_count


@triton.jit
def _pack_fn_kernel(
    fn_ptr, ft_ptr,
    K, N,
    stride_fn_n, stride_fn_k,
    stride_tt, stride_tk, stride_tn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """fn[N, K] fp32 -> ft[3, K, BLOCK_N] bf16 (hi, mid, lo), B-operand layout.

    Loaded n-major (K contiguous, coalesced), transposed in registers/shared,
    stored k-major so the GEMM's B tile needs no `tl.trans`.
    """
    k0 = tl.program_id(0) * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N

    fb = tl.load(
        fn_ptr + offs_n[:, None] * stride_fn_n + offs_k[None, :] * stride_fn_k,
        mask=mask_n[:, None] & mask_k[None, :], other=0.0,
    )
    hi = fb.to(tl.bfloat16)
    r1 = fb - hi.to(tl.float32)
    mid = r1.to(tl.bfloat16)
    lo = (r1 - mid.to(tl.float32)).to(tl.bfloat16)

    base = ft_ptr + offs_k[:, None] * stride_tk + offs_n[None, :] * stride_tn
    tl.store(base, tl.trans(hi), mask=mask_k[:, None])
    tl.store(base + stride_tt, tl.trans(mid), mask=mask_k[:, None])
    tl.store(base + 2 * stride_tt, tl.trans(lo), mask=mask_k[:, None])


@triton.jit
def _prenorm_gemm_kernel(
    x_ptr, ft_ptr, part_ptr, psq_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_tt, stride_tk, stride_tn,
    stride_pm, stride_pn, stride_ps,
    stride_qm, stride_qs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_per_split = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Every byte of `x` is read exactly once, so .cg (stream, do not keep in
        # L2 where the 3.1 MB of packed `fn` has to stay resident for every
        # M-tile) is the right intent - but MEASURED it is exactly neutral here
        # at BM=128/BK=32 (64.8 us either way; in an isolated 1D read it costs
        # 10 %).  Kept for the intent, not for a measured gain.
        xb = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            cache_modifier=".cg",
        )
        bp = ft_ptr + offs_k[:, None] * stride_tk + offs_n[None, :] * stride_tn
        # hi, then mid, then lo: fixed order, so the fp32 accumulation is
        # bitwise reproducible.
        acc = tl.dot(xb, tl.load(bp, mask=mask_k[:, None], other=0.0), acc)
        acc = tl.dot(xb, tl.load(bp + stride_tt, mask=mask_k[:, None], other=0.0), acc)
        acc = tl.dot(xb, tl.load(bp + 2 * stride_tt, mask=mask_k[:, None], other=0.0), acc)
        xf = xb.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)

    tl.store(
        part_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        + pid_k * stride_ps,
        acc, mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(psq_ptr + offs_m * stride_qm + pid_k * stride_qs, sq, mask=mask_m)


@triton.jit
def _prenorm_reduce_kernel(
    part_ptr, psq_ptr, out_ptr, sqrsum_ptr,
    M, N,
    stride_pm, stride_pn, stride_ps,
    stride_qm, stride_qs,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # Fixed ascending order over splits: deterministic, no atomics.
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(
            part_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
            + s * stride_ps, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        sq += tl.load(psq_ptr + offs_m * stride_qm + s * stride_qs,
                      mask=mask_m, other=0.0)

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(sqrsum_ptr + offs_m, sq, mask=mask_m)


# Measured on this part (70 SMs), M=1152, K=16384, N=24, packed-fn bf16x3,
# end-to-end (pack + GEMM + split-K reduce), amortised over 40 calls with the
# incumbent measured in the same process:
#     BM=128 BK= 32 SK=32 w=4 s=3   82 us  5.56x   <- chosen
#     BM=128 BK= 32 SK=32 w=4 s=4   84 us  5.43x
#     BM=128 BK= 32 SK=16 w=4 s=4   93 us  4.92x
#     BM=128 BK= 32 SK=64 w=4 s=3   93 us  4.91x
#     BM=128 BK= 16 SK=32 w=4 s=4   92 us  4.95x
#     BM= 64 BK= 32 SK=32 w=4 s=3   93 us  4.92x
#     BM=256 BK= 32 SK=32 w=4 s=3  101 us  4.53x
#     BM=128 BK= 32 SK=32 w=8 s=3  106 us  4.32x
#     BM=128 BK= 64 SK=32 w=4 s=3   96 us  4.36x
#     BM=128 BK=128 SK=32 w=4 s=3  100 us
#     BM=256 BK= 64 SK=16 w=4 s=3  167 us  2.52x
# BLOCK_K=32 beats 64/128/256: the B tile is 3 x [32, 32] bf16 and the shorter
# K step buys back occupancy, which is what this kernel is short of.  CTA count
# dominates everything else - SPLIT_K=32 puts 9 x 32 = 288 CTAs on 70 SMs, and
# dropping to 16 costs 13 %.  2 and 8 warps both lose to 4.
def _select_config(M, K, N, sms):
    """Pure function of the shape - no autotune, so capture is safe."""
    BLOCK_N = max(16, triton.next_power_of_2(N))
    BLOCK_M = 16 if M <= 16 else (32 if M <= 64 else 128)
    BLOCK_K = 32
    m_tiles = -(-M // BLOCK_M)
    # Aim for ~4 CTAs per SM; keep at least 2 BLOCK_K of real work per split.
    max_split = max(1, K // (2 * BLOCK_K))
    split_k = 1
    while split_k < 32 and split_k < max_split and m_tiles * split_k < 4 * sms:
        split_k *= 2
    return BLOCK_M, BLOCK_N, BLOCK_K, split_k, 4, 3


def _select_reduce_config(M, split_k, sms):
    """The reduce is its own, much smaller kernel and must NOT inherit the
    GEMM's BLOCK_M: ceil(1152/128) = 9 CTAs left it at 24 us of the 147 us
    total.  Measured cost of the reduce alone (GEMM subtracted, partials
    [SPLIT_K, M, BLOCK_N]): 4 tokens/CTA 7.0 us, 8 -> 9.9, 16 -> 16.7,
    32 -> 15.4, 64 -> 61.7, and one warp beats two or four at every size.
    It is latency-bound on 4.9 MB of partials, so what matters is CTA count:
    288 CTAs of one warp.  An m-major [M, SPLIT_K, BLOCK_N] partial layout was
    also measured - it does not help the reduce (8.2 us) and costs the GEMM
    8 us, because the GEMM's store then strides by SPLIT_K * BLOCK_N."""
    BLOCK_M = 4 if M >= 4 * sms else max(1, triton.next_power_of_2(max(M // sms, 1)))
    return BLOCK_M, 1


def _workspace(device, M, BLOCK_N, split_k):
    """Split-K scratch, cached per shape and never shrunk (see INTEGRATION.md)."""
    key = (device, M, BLOCK_N, split_k)
    ws = _WORKSPACE.get(key)
    if ws is None:
        part = torch.empty((split_k, M, BLOCK_N), dtype=torch.float32, device=device)
        psq = torch.empty((split_k, M), dtype=torch.float32, device=device)
        ws = (part, psq)
        _WORKSPACE[key] = ws
    return ws


def _fn_pack(device, K, BLOCK_N):
    """[3, K, BLOCK_N] bf16 hi/mid/lo scratch, cached (3.1 MB at K=16384)."""
    key = (device, K, BLOCK_N)
    ft = _FN_PACK.get(key)
    if ft is None:
        ft = torch.empty((3, K, BLOCK_N), dtype=torch.bfloat16, device=device)
        _FN_PACK[key] = ft
    return ft


def hc_prenorm_gemm(x, fn, out=None, sqrsum=None):
    assert x.dtype == torch.bfloat16, "x must be bf16"
    assert fn.dtype == torch.float32, "fn must be fp32 (no precision reduction)"
    assert x.ndim == 2 and fn.ndim == 2 and x.shape[1] == fn.shape[1]
    M, K = x.shape
    N = fn.shape[0]
    dev = x.device
    if out is None:
        out = torch.empty((1, M, N), dtype=torch.float32, device=dev)
    if sqrsum is None:
        sqrsum = torch.empty((1, M), dtype=torch.float32, device=dev)
    assert out.shape == (1, M, N) and sqrsum.shape == (1, M)

    sms = num_sms(dev.index or 0)
    BLOCK_M, BLOCK_N, BLOCK_K, split_k, nw, ns = _select_config(M, K, N, sms)
    part, psq = _workspace(dev, M, BLOCK_N, split_k)
    ft = _fn_pack(dev, K, BLOCK_N)
    m_tiles = -(-M // BLOCK_M)

    PACK_K = 128
    _pack_fn_kernel[(-(-K // PACK_K),)](
        fn, ft,
        K, N,
        fn.stride(0), fn.stride(1),
        ft.stride(0), ft.stride(1), ft.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=PACK_K, num_warps=4, num_stages=1,
    )
    _prenorm_gemm_kernel[(m_tiles, split_k)](
        x, ft, part, psq,
        M, K, N,
        x.stride(0), x.stride(1),
        ft.stride(0), ft.stride(1), ft.stride(2),
        part.stride(1), part.stride(2), part.stride(0),
        psq.stride(1), psq.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=split_k,
        num_warps=nw, num_stages=ns,
    )
    r_bm, r_warps = _select_reduce_config(M, split_k, sms)
    _prenorm_reduce_kernel[(-(-M // r_bm),)](
        part, psq, out, sqrsum,
        M, N,
        part.stride(1), part.stride(2), part.stride(0),
        psq.stride(1), psq.stride(0),
        out.stride(1), out.stride(2),
        BLOCK_M=r_bm, BLOCK_N=BLOCK_N, SPLIT_K=split_k,
        num_warps=r_warps, num_stages=1,
    )
    return out, sqrsum


def warmup(ms, K=16384, N=24, device="cuda:0"):
    """JIT-compile and size the workspaces for every M a graph may capture."""
    dev = torch.device(device)
    fn = torch.zeros(N, K, dtype=torch.float32, device=dev)
    for M in ms:
        x = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
        hc_prenorm_gemm(x, fn)
    torch.cuda.synchronize()
