# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC fused post+pre for GLM-5.3-Flash decode on sm_80, M <= 32 (v3).

Drop-in for `vllm.model_executor.kernels.mhc.tilelang.mhc_fused_post_pre_tilelang`:
same signature, same four outputs, same dtypes. Dispatched by
`vllm_sm80.kernels.mhc.use_ampere_mhc_decode_v3` (`VLLM_GLM5_DECODE_MHC_V2`,
1 <= M <= the `mhc/decode_v2_max_rows` row of the sm80 tuning table).

v3 is the earlier v2 kernel pair with a faster finish, measured on the 74-SM
part and held to an fp64 error gate (mean <= 1.10x, max <= 1.25x of v2) and
bitwise run-to-run determinism: the finish reduces the partials with one CTA
barrier at M <= 16 (`ONEBAR`, the same summation order), issues every load
before its first reduction, and runs the middle Sinkhorn iterations in
scale-vector form (inline PTX, eps-seeded FMA chains) with the final row norm
applied to V diag(C).
The text below describes everything else, unchanged from v2.

Both launches are Gluon kernels (`triton.experimental.gluon`, written against
the pinned Triton 3.7.1) for hc == 4 and hidden % 1024 == 0 (GLM-5.3-Flash),
the only shapes the gate admits:

  A residual_cur = post*x + comb^T @ residual for a 32-wide hidden slice,
    stored bf16, plus the prenorm GEMM partials on the tensor cores as a 3-pass
    (tf32x3-equivalent) mma_v2 with `fn` streamed global->shared by cp.async,
    and the sqrsum partials. M <= 8: `_glu_post_prenorm_tb_kernel`, grid
    (cdiv(M, 8), hidden / 32), transposed (fn^T @ rc^T) and K-split over warps;
    M > 8: `_glu_post_prenorm_kernel`, grid (cdiv(M, 16), hidden / 32).
  B `_glu_finish_kernel`, grid (M, 2): CTA (m, 0) reduces the partials, writes
    post_mix and runs the 20-iteration 4x4 Sinkhorn; CTA (m, 1) collapses the
    four streams and applies the RMSNorm. The two share only `pre`, so the
    Sinkhorn latency chain is off the collapse's critical path.

Measured per call on one CMP 170HX (graph replay, 90-deep L2 rotation), against
the faster of the v1 kernel and TileLang at each M:
    M=4 12.7 -> 7.8 us, M=16 18.2 -> 8.9 us (full table in the gate docstring).

NO PRECISION REDUCTION. bf16 in / fp32 accumulate / bf16 out, fp32 `fn`. On
real activations every output is as close to an FP64 recomputation as the
TileLang kernel's (mean and max error within ~1.05x of it at M = 4..25), and
`residual_cur` is bitwise equal to TileLang's. Three things make that true:
  * the prenorm GEMM on the tensor cores is split into three TF32 terms
    (hi*hi + hi*lo + lo*hi), and the big hi*hi term is accumulated OUTSIDE the
    tensor core, one fresh mma per 8-wide k block added into fp32 registers.
    An sm_80 TF32 mma truncates when it adds a product block into a non-zero
    accumulator, so chaining the big term through the mma accumulator cost 11x
    TileLang's error on the mixes (3x at one fresh mma per 32-wide block, 1.6x
    per 16-wide block, 1.05x per 8-wide block);
  * `residual_cur` is formed in TileLang's exact FMA order,
    fma(post, x, comb[0] * res[0]) then + comb[k] * res[k], k = 1..3;
  * the Sinkhorn normalisations use `rcp.approx`, except the final row and
    column normalisation, whose quotient gets one FMA correction (rounding
    errors of the earlier normalisations are absorbed by the later ones; the
    last one's are not), and the sigmoids use a Newton-refined reciprocal.
Gated as before on bitwise determinism and CUDA-graph capture. The two bf16
rounding points of the op are kept exactly:
  * `mixes` and `sqrsum` are accumulated from the pre-rounding fp32
    `residual_cur`, while the collapse reads the stored bf16 `residual_cur`;
  * the RMSNorm squared sum is taken from the fp32 collapse and the scale is
    applied to its bf16 rounding.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2


# --------------------------------------------------------------------------
# Kernel A in Gluon (explicit layouts), used for every M.
# Gluon (`triton.experimental.gluon`) is experimental API; this is written
# against the pinned Triton 3.7.1.
# --------------------------------------------------------------------------
@gluon.jit
def _clk_now():
    # A %clock64 read, used only to pin kernel A's schedule: ptxas keeps memory
    # instructions on their side of it, and a value derived from it (>= 0 for
    # the next ~2^63 cycles) makes later instructions depend on it.  Without the
    # pin ptxas sinks the fn LDGSTS below the mix, behind a full x/residual
    # round trip.
    return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], dtype=gl.int64,
                                     is_pure=False, pack=1)


@gluon.jit
def _tf32_hi(x):
    # x rounded to 10 explicit mantissa bits (nearest, ties away): add half an
    # ulp of the 13 dropped bits and clear them.  Bit-identical to
    # cvt.rna on finite inputs, and 1 instruction pair instead of ~4 SASS.
    return gl.inline_asm_elementwise("{ .reg .b32 t; add.u32 t, $1, 4096; and.b32 $0, t, -8192; }",
                                     "=r,r", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _mma3(a, b, acc_s, acc_t, acc_b, z, K: gl.constexpr):
    """a @ b to fp32 accuracy on the tensor cores, for the transposed kernel
    (a = fn^T [2, NP2, K], b = residual_cur^T [2, K, BT], K = 16): the three-pass
    split (hi*hi + hi*lo + lo*hi), each term in its own accumulator (summed as
    b + (s + t) at the end).  The two small terms chain through the mma
    accumulator (their truncation is 2^-11 below the result).  The big term is
    formed per 8-wide k block from a zero accumulator and added in fp32: a K=16
    mma is two chained m16n8k8 mmas, so b's hi part is split into its two k
    halves, each paired with a zero half (the second, all-zero mma adds nothing
    and truncates nothing)."""
    ah = _tf32_hi(a)
    bh = _tf32_hi(b)
    acc_s = mma_v2(a - ah, bh, acc_s)
    acc_t = mma_v2(ah, b - bh, acc_t)
    kk = gl.arange(0, K, layout=gl.SliceLayout(0, gl.SliceLayout(2, b.type.layout)))
    lo8 = (kk < 8)[None, :, None]
    acc_b = acc_b + mma_v2(ah, gl.where(lo8, bh, 0.0), z).to(acc_b.dtype)
    acc_b = acc_b + mma_v2(ah, gl.where(lo8, 0.0, bh), z).to(acc_b.dtype)
    return acc_s, acc_t, acc_b


@gluon.jit
def _mma3_smem(hsh, hsl, sb, acc_s, acc_t, acc_b, z, LA: gl.constexpr, LB: gl.constexpr,
               HB: gl.constexpr):
    """The same product for `_glu_post_prenorm_kernel`: operand A (residual_cur)
    already split into (hi, lo) in shared memory, operand B (fn) in shared
    memory, contracted in 16-wide k slices; the big term per 8-wide k block from a
    zero accumulator (A's hi part masked to one k half per mma), see `_mma3`."""
    kk = gl.arange(0, 16, layout=gl.SliceLayout(0, LA))
    lo8 = (kk < 8)[None, :]
    for kc in gl.static_range(HB // 16):
        ah = hsh.slice(kc * 16, 16, dim=1).load(LA)
        al = hsl.slice(kc * 16, 16, dim=1).load(LA)
        b = sb.slice(kc * 16, 16, dim=0).load(LB)
        bh = _tf32_hi(b)
        acc_t = mma_v2(ah, b - bh, acc_t)
        acc_s = mma_v2(al, bh, acc_s)
        acc_b = acc_b + mma_v2(gl.where(lo8, ah, 0.0), bh, z).to(acc_b.dtype)
        acc_b = acc_b + mma_v2(gl.where(lo8, 0.0, ah), bh, z).to(acc_b.dtype)
    return acc_s, acc_t, acc_b


@gluon.jit
def _glu_post_prenorm_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr, rc_ptr, mix_ptr, sqr_ptr,
                             M, HIDDEN: gl.constexpr, NOUT: gl.constexpr, HB: gl.constexpr,
                             BM: gl.constexpr, NP2: gl.constexpr, NW: gl.constexpr,
                             HAS_POST: gl.constexpr, ROUND_MIX: gl.constexpr,
                             FP64_ACC: gl.constexpr = False, HS_REUSE: gl.constexpr = False):
    """Kernel A for hc == 4: residual_cur + prenorm GEMM partials on the tensor cores.

    grid (cdiv(M, BM), HIDDEN // HB).  CTA (pm, ks) owns tokens [pm*BM, pm*BM+BM)
    (masked) and hidden[ks*HB : ks*HB+HB] of the 4 streams.  Written in Gluon
    because the Triton versions lost most of their time to layout conversions:
    the Triton mma kernel materialised residual_cur in three layouts and
    serialised ~6 global round trips behind shared-memory conversions.
    Used for M > BT_T (M <= BT_T takes `_glu_post_prenorm_tb_kernel`).  Here:
      * x, the 4 residual streams, post and comb are issued as independent
        loads in one blocked layout L2, THEN fn is copied global->shared with
        cp.async, one commit group per stream j, into swizzled [HB, NP2] buffers
        (rows n >= NOUT zero-filled).  The order is pinned with `_clk_now`:
        fn issued before the small loads delays them behind the whole fn
        stream (+0.5 us), fn issued after the mix (ptxas' own choice) exposes
        a full load round trip;
      * residual_cur_j = post_j*x + sum_k comb[k,j]*res_k in fp32, stored bf16;
        sqrsum from the fp32 values (R1);
      * operand A (residual_cur, replicated in all NW warps by the [1, NW] mma
        layout) is split hi/lo ONCE in L2 and staged in shared memory:
        splitting it in the dot-operand layout cost every warp 48 ALU ops per
        stream, and the mma phase is per-warp issue bound;
      * per stream j: wait for its fn group, then the 3-pass (tf32x3-equivalent)
        product [BM, HB] x [HB, NP2] into three accumulators, the big term per
        8-wide k block (`_mma3_smem`).
    """
    HC: gl.constexpr = 4
    L2: gl.constexpr = gl.BlockedLayout([1, 4], [BM // NW, 32 * NW // BM], [NW, 1], [1, 0])
    LF: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, NW], [0, 1])
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, NW],
                                                  instr_shape=[16, 8])
    LA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=1)
    LB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=1)
    SF: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 8, [0, 1])
    pm = gl.program_id(0)
    ks = gl.program_id(1)

    mi = pm * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L2))
    hh = gl.arange(0, HB, layout=gl.SliceLayout(0, L2))
    mm = mi < M
    m2 = mm[:, None] & (hh < HB)[None, :]
    h = ks * HB + hh
    rb = res_ptr + mi[:, None] * (HC * HIDDEN) + h[None, :]
    xv = gl.load(x_ptr + mi[:, None] * HIDDEN + h[None, :], mask=m2, other=0.0)
    r0 = gl.load(rb, mask=m2, other=0.0)
    r1 = gl.load(rb + HIDDEN, mask=m2, other=0.0)
    r2 = gl.load(rb + 2 * HIDDEN, mask=m2, other=0.0)
    r3 = gl.load(rb + 3 * HIDDEN, mask=m2, other=0.0)
    pb = post_ptr + mi * HC
    cb = comb_ptr + mi * (HC * HC)
    p0 = gl.load(pb, mask=mm, other=0.0)
    p1 = gl.load(pb + 1, mask=mm, other=0.0)
    p2 = gl.load(pb + 2, mask=mm, other=0.0)
    p3 = gl.load(pb + 3, mask=mm, other=0.0)
    c00 = gl.load(cb + 0, mask=mm, other=0.0)
    c01 = gl.load(cb + 1, mask=mm, other=0.0)
    c02 = gl.load(cb + 2, mask=mm, other=0.0)
    c03 = gl.load(cb + 3, mask=mm, other=0.0)
    c10 = gl.load(cb + 4, mask=mm, other=0.0)
    c11 = gl.load(cb + 5, mask=mm, other=0.0)
    c12 = gl.load(cb + 6, mask=mm, other=0.0)
    c13 = gl.load(cb + 7, mask=mm, other=0.0)
    c20 = gl.load(cb + 8, mask=mm, other=0.0)
    c21 = gl.load(cb + 9, mask=mm, other=0.0)
    c22 = gl.load(cb + 10, mask=mm, other=0.0)
    c23 = gl.load(cb + 11, mask=mm, other=0.0)
    c30 = gl.load(cb + 12, mask=mm, other=0.0)
    c31 = gl.load(cb + 13, mask=mm, other=0.0)
    c32 = gl.load(cb + 14, mask=mm, other=0.0)
    c33 = gl.load(cb + 15, mask=mm, other=0.0)

    # fn after the small loads, pinned there by the clock read in its mask
    kf = gl.arange(0, HB, layout=gl.SliceLayout(1, LF))
    nf = gl.arange(0, NP2, layout=gl.SliceLayout(0, LF))
    fmask = (kf < HB)[:, None] & (nf < NOUT)[None, :] & (_clk_now() >= 0)
    fbase = fn_ptr + nf[None, :] * (HC * HIDDEN) + ks * HB + kf[:, None]
    s0 = gl.allocate_shared_memory(gl.float32, [HB, NP2], SF)
    s1 = gl.allocate_shared_memory(gl.float32, [HB, NP2], SF)
    s2 = gl.allocate_shared_memory(gl.float32, [HB, NP2], SF)
    s3 = gl.allocate_shared_memory(gl.float32, [HB, NP2], SF)
    async_copy.async_copy_global_to_shared(s0, fbase, mask=fmask)
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(s1, fbase + HIDDEN, mask=fmask)
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(s2, fbase + 2 * HIDDEN, mask=fmask)
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(s3, fbase + 3 * HIDDEN, mask=fmask)
    async_copy.commit_group()

    # ... and the mix after the fn copies (else ptxas hoists it above them)
    pin = _clk_now()
    xf = gl.where(pin >= 0, xv.to(gl.float32), 0.0)
    rf0 = gl.where(pin >= 0, r0.to(gl.float32), 0.0)
    rf1 = r1.to(gl.float32)
    rf2 = r2.to(gl.float32)
    rf3 = r3.to(gl.float32)
    # residual_cur[j] = post[j]*x + sum_k comb[k, j] * residual[k], in TileLang's FMA
    # order (bitwise-equal output): comb[0,j]*res0, + post[j]*x, + comb[k,j]*res_k
    n0 = c00[:, None] * rf0
    n0 = gl.fma(p0[:, None], xf, n0)
    n0 = gl.fma(c10[:, None], rf1, n0)
    n0 = gl.fma(c20[:, None], rf2, n0)
    n0 = gl.fma(c30[:, None], rf3, n0)
    n1 = c01[:, None] * rf0
    n1 = gl.fma(p1[:, None], xf, n1)
    n1 = gl.fma(c11[:, None], rf1, n1)
    n1 = gl.fma(c21[:, None], rf2, n1)
    n1 = gl.fma(c31[:, None], rf3, n1)
    n2 = c02[:, None] * rf0
    n2 = gl.fma(p2[:, None], xf, n2)
    n2 = gl.fma(c12[:, None], rf1, n2)
    n2 = gl.fma(c22[:, None], rf2, n2)
    n2 = gl.fma(c32[:, None], rf3, n2)
    n3 = c03[:, None] * rf0
    n3 = gl.fma(p3[:, None], xf, n3)
    n3 = gl.fma(c13[:, None], rf1, n3)
    n3 = gl.fma(c23[:, None], rf2, n3)
    n3 = gl.fma(c33[:, None], rf3, n3)
    if not HAS_POST:
        # pre only (see `post_prenorm`): the streams are residual itself
        n0 = rf0
        n1 = rf1
        n2 = rf2
        n3 = rf3
    if ROUND_MIX:
        n0 = n0.to(gl.bfloat16).to(gl.float32)
        n1 = n1.to(gl.bfloat16).to(gl.float32)
        n2 = n2.to(gl.bfloat16).to(gl.float32)
        n3 = n3.to(gl.bfloat16).to(gl.float32)

    ob = rc_ptr + mi[:, None] * (HC * HIDDEN) + h[None, :]
    gl.store(ob, n0.to(gl.bfloat16), mask=m2)
    gl.store(ob + HIDDEN, n1.to(gl.bfloat16), mask=m2)
    gl.store(ob + 2 * HIDDEN, n2.to(gl.bfloat16), mask=m2)
    gl.store(ob + 3 * HIDDEN, n3.to(gl.bfloat16), mask=m2)
    if FP64_ACC:
        sq = gl.sum(n0.to(gl.float64) * n0 + n1.to(gl.float64) * n1
                    + n2.to(gl.float64) * n2 + n3.to(gl.float64) * n3, axis=1).to(gl.float32)
    else:
        sq = gl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, axis=1)
    gl.store(sqr_ptr + ks * M + mi, sq, mask=mm)

    acc_s = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    if FP64_ACC:
        acc_b = gl.zeros([BM, NP2], gl.float64, layout=MMA)
    else:
        acc_b = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    acc_t = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    z = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    # operand A split hi/lo once here (4 elements per thread) - see the docstring.
    # HS_REUSE (more than one row block): two hi/lo buffer pairs, each reused
    # once - stream 2 goes into stream 0's pair and stream 3 into stream 1's
    # after the barrier that ends the pair's mma. Shared memory 32 -> 24 KB per
    # CTA, so 5+ CTAs fit an SM (M = 32's 320 CTAs in one wave instead of 4 + a
    # straggler); in one row block the late stores only lengthen the chain. The
    # products and their order are the same either way; without HS_REUSE the
    # statements are the original ones, in their order.
    SA: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    hs0 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    hs1 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    hs2 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    hs3 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    if not HS_REUSE:
        hs4 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
        hs5 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
        hs6 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
        hs7 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    h0 = _tf32_hi(n0)
    h1 = _tf32_hi(n1)
    h2 = _tf32_hi(n2)
    h3 = _tf32_hi(n3)
    hs0.store(h0)
    hs1.store(n0 - h0)
    hs2.store(h1)
    hs3.store(n1 - h1)
    if HS_REUSE:
        async_copy.wait_group(3)
        gl.barrier()
        acc_s, acc_t, acc_b = _mma3_smem(hs0, hs1, s0, acc_s, acc_t, acc_b, z, LA, LB, HB)
        async_copy.wait_group(2)
        gl.barrier()
        hs0.store(h2)
        hs1.store(n2 - h2)
        acc_s, acc_t, acc_b = _mma3_smem(hs2, hs3, s1, acc_s, acc_t, acc_b, z, LA, LB, HB)
        async_copy.wait_group(1)
        gl.barrier()
        hs2.store(h3)
        hs3.store(n3 - h3)
        acc_s, acc_t, acc_b = _mma3_smem(hs0, hs1, s2, acc_s, acc_t, acc_b, z, LA, LB, HB)
        async_copy.wait_group(0)
        gl.barrier()
        acc_s, acc_t, acc_b = _mma3_smem(hs2, hs3, s3, acc_s, acc_t, acc_b, z, LA, LB, HB)
    else:
        hs4.store(h2)
        hs5.store(n2 - h2)
        hs6.store(h3)
        hs7.store(n3 - h3)
        async_copy.wait_group(3)
        gl.barrier()
        acc_s, acc_t, acc_b = _mma3_smem(hs0, hs1, s0, acc_s, acc_t, acc_b, z, LA, LB, HB)
        async_copy.wait_group(2)
        gl.barrier()
        acc_s, acc_t, acc_b = _mma3_smem(hs2, hs3, s1, acc_s, acc_t, acc_b, z, LA, LB, HB)
        async_copy.wait_group(1)
        gl.barrier()
        acc_s, acc_t, acc_b = _mma3_smem(hs4, hs5, s2, acc_s, acc_t, acc_b, z, LA, LB, HB)
        async_copy.wait_group(0)
        gl.barrier()
        acc_s, acc_t, acc_b = _mma3_smem(hs6, hs7, s3, acc_s, acc_t, acc_b, z, LA, LB, HB)
    if FP64_ACC:
        acc = (acc_b + (acc_s + acc_t).to(gl.float64)).to(gl.float32)
    else:
        acc = acc_b + (acc_s + acc_t)

    mo = pm * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, MMA))
    no = gl.arange(0, NP2, layout=gl.SliceLayout(0, MMA))
    gl.store(mix_ptr + (ks * M + mo[:, None]) * NOUT + no[None, :], acc,
             mask=(mo < M)[:, None] & (no < NOUT)[None, :])


@gluon.jit
def _glu_post_prenorm_tb_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr, rc_ptr, mix_ptr, sqr_ptr,
                                M, HIDDEN: gl.constexpr, NOUT: gl.constexpr, HB: gl.constexpr,
                                BT: gl.constexpr, NP2: gl.constexpr,
                                HAS_POST: gl.constexpr, ROUND_MIX: gl.constexpr,
                                FP64_ACC: gl.constexpr = False):
    """Kernel A for hc == 4 and M <= BT (= 8), transposed and K-split over warps:
    mix^T[n, t] = sum over the two k-halves h of fn^T[h][n, k] @ rc^T[h][k, t], as one
    batched mma with warps [2 (k-half), 2 (n-tile), 1].  The [16, 32] form wastes
    12 of 16 mma rows at M = 4 and replicates operand A in all four warps; here each
    warp issues a quarter of the CTA's HMMAs (24 instead of 48) - the mma phase is
    per-warp issue/latency bound (-0.5 us at M = 4).  Same fn/mix pinning as
    `_glu_post_prenorm_kernel`.  Measured worse: the same kernel at BT = 16 for
    M = 16 (+1.1 us), and the non-transposed K-split form (+1.1 us)."""
    HC: gl.constexpr = 4
    NW: gl.constexpr = 4
    KH: gl.constexpr = HB // 2
    # (k-half, k, n): k contiguous, FV = 4 fp32 = 16 B per cp.async
    FV: gl.constexpr = 4
    LF: gl.constexpr = gl.BlockedLayout([1, FV, 1], [1, KH // FV, 32 * FV // KH], [2, 1, 2],
                                        [1, 2, 0])
    # (k-half, k, t) for x / residual: 2 bf16 per thread along k
    LT: gl.constexpr = gl.BlockedLayout([1, 2, 1], [1, 8, 4], [1, 2, 2], [1, 2, 0])
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2, 1],
                                                  instr_shape=[1, 16, 8])
    LA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=1)
    LB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=1)
    SF: gl.constexpr = gl.SwizzledSharedLayout(FV, 1, 8, [1, 2, 0])
    pm = gl.program_id(0)
    ks = gl.program_id(1)

    fh = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, LF)))
    fk = gl.arange(0, KH, layout=gl.SliceLayout(0, gl.SliceLayout(2, LF)))
    fnn = gl.arange(0, NP2, layout=gl.SliceLayout(0, gl.SliceLayout(1, LF)))
    fmask = (fh[:, None, None] < 2) & (fk[None, :, None] < KH) & (fnn[None, None, :] < NOUT)
    fbase = (fn_ptr + fnn[None, None, :] * (HC * HIDDEN) + ks * HB
             + fh[:, None, None] * KH + fk[None, :, None])
    s0 = gl.allocate_shared_memory(gl.float32, [2, KH, NP2], SF)
    s1 = gl.allocate_shared_memory(gl.float32, [2, KH, NP2], SF)
    s2 = gl.allocate_shared_memory(gl.float32, [2, KH, NP2], SF)
    s3 = gl.allocate_shared_memory(gl.float32, [2, KH, NP2], SF)

    th = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, LT)))
    tk = gl.arange(0, KH, layout=gl.SliceLayout(0, gl.SliceLayout(2, LT)))
    ti = pm * BT + gl.arange(0, BT, layout=gl.SliceLayout(0, gl.SliceLayout(1, LT)))
    tm = ti < M
    h3 = ks * HB + th[:, None, None] * KH + tk[None, :, None]
    t3 = ti[None, None, :]
    m3 = (h3 < HIDDEN) & tm[None, None, :]
    rb = res_ptr + t3 * (HC * HIDDEN) + h3
    xv = gl.load(x_ptr + t3 * HIDDEN + h3, mask=m3, other=0.0)
    r0 = gl.load(rb, mask=m3, other=0.0)
    r1 = gl.load(rb + HIDDEN, mask=m3, other=0.0)
    r2 = gl.load(rb + 2 * HIDDEN, mask=m3, other=0.0)
    r3 = gl.load(rb + 3 * HIDDEN, mask=m3, other=0.0)
    pb = post_ptr + ti * HC
    cb = comb_ptr + ti * (HC * HC)
    p0 = gl.load(pb, mask=tm, other=0.0)[None, None, :]
    p1 = gl.load(pb + 1, mask=tm, other=0.0)[None, None, :]
    p2 = gl.load(pb + 2, mask=tm, other=0.0)[None, None, :]
    p3 = gl.load(pb + 3, mask=tm, other=0.0)[None, None, :]
    c00 = gl.load(cb + 0, mask=tm, other=0.0)[None, None, :]
    c01 = gl.load(cb + 1, mask=tm, other=0.0)[None, None, :]
    c02 = gl.load(cb + 2, mask=tm, other=0.0)[None, None, :]
    c03 = gl.load(cb + 3, mask=tm, other=0.0)[None, None, :]
    c10 = gl.load(cb + 4, mask=tm, other=0.0)[None, None, :]
    c11 = gl.load(cb + 5, mask=tm, other=0.0)[None, None, :]
    c12 = gl.load(cb + 6, mask=tm, other=0.0)[None, None, :]
    c13 = gl.load(cb + 7, mask=tm, other=0.0)[None, None, :]
    c20 = gl.load(cb + 8, mask=tm, other=0.0)[None, None, :]
    c21 = gl.load(cb + 9, mask=tm, other=0.0)[None, None, :]
    c22 = gl.load(cb + 10, mask=tm, other=0.0)[None, None, :]
    c23 = gl.load(cb + 11, mask=tm, other=0.0)[None, None, :]
    c30 = gl.load(cb + 12, mask=tm, other=0.0)[None, None, :]
    c31 = gl.load(cb + 13, mask=tm, other=0.0)[None, None, :]
    c32 = gl.load(cb + 14, mask=tm, other=0.0)[None, None, :]
    c33 = gl.load(cb + 15, mask=tm, other=0.0)[None, None, :]

    # fn after the small loads, pinned (see _glu_post_prenorm_kernel)
    fmask = fmask & (_clk_now() >= 0)
    async_copy.async_copy_global_to_shared(s0, fbase, mask=fmask)
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(s1, fbase + HIDDEN, mask=fmask)
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(s2, fbase + 2 * HIDDEN, mask=fmask)
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(s3, fbase + 3 * HIDDEN, mask=fmask)
    async_copy.commit_group()
    # ... and the mix after the fn copies (else ptxas hoists it above them)
    pin = _clk_now()
    xf = gl.where(pin >= 0, xv.to(gl.float32), 0.0)
    rf0 = gl.where(pin >= 0, r0.to(gl.float32), 0.0)
    rf1 = r1.to(gl.float32)
    rf2 = r2.to(gl.float32)
    rf3 = r3.to(gl.float32)
    # TileLang's FMA order, see _glu_post_prenorm_kernel
    n0 = c00 * rf0
    n0 = gl.fma(p0, xf, n0)
    n0 = gl.fma(c10, rf1, n0)
    n0 = gl.fma(c20, rf2, n0)
    n0 = gl.fma(c30, rf3, n0)
    n1 = c01 * rf0
    n1 = gl.fma(p1, xf, n1)
    n1 = gl.fma(c11, rf1, n1)
    n1 = gl.fma(c21, rf2, n1)
    n1 = gl.fma(c31, rf3, n1)
    n2 = c02 * rf0
    n2 = gl.fma(p2, xf, n2)
    n2 = gl.fma(c12, rf1, n2)
    n2 = gl.fma(c22, rf2, n2)
    n2 = gl.fma(c32, rf3, n2)
    n3 = c03 * rf0
    n3 = gl.fma(p3, xf, n3)
    n3 = gl.fma(c13, rf1, n3)
    n3 = gl.fma(c23, rf2, n3)
    n3 = gl.fma(c33, rf3, n3)
    if not HAS_POST:
        # pre only (see `post_prenorm`): the streams are residual itself
        n0 = rf0
        n1 = rf1
        n2 = rf2
        n3 = rf3
    if ROUND_MIX:
        n0 = n0.to(gl.bfloat16).to(gl.float32)
        n1 = n1.to(gl.bfloat16).to(gl.float32)
        n2 = n2.to(gl.bfloat16).to(gl.float32)
        n3 = n3.to(gl.bfloat16).to(gl.float32)

    ob = rc_ptr + t3 * (HC * HIDDEN) + h3
    gl.store(ob, n0.to(gl.bfloat16), mask=m3)
    gl.store(ob + HIDDEN, n1.to(gl.bfloat16), mask=m3)
    gl.store(ob + 2 * HIDDEN, n2.to(gl.bfloat16), mask=m3)
    gl.store(ob + 3 * HIDDEN, n3.to(gl.bfloat16), mask=m3)
    if FP64_ACC:
        sq = gl.sum(gl.sum(n0.to(gl.float64) * n0 + n1.to(gl.float64) * n1
                           + n2.to(gl.float64) * n2 + n3.to(gl.float64) * n3, axis=1),
                    axis=0).to(gl.float32)
    else:
        sq = gl.sum(gl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, axis=1), axis=0)
    gl.store(sqr_ptr + ks * M + ti, sq, mask=tm)

    acc_s = gl.zeros([2, NP2, BT], gl.float32, layout=MMA)
    acc_t = gl.zeros([2, NP2, BT], gl.float32, layout=MMA)
    if FP64_ACC:
        acc_b = gl.zeros([2, NP2, BT], gl.float64, layout=MMA)
    else:
        acc_b = gl.zeros([2, NP2, BT], gl.float32, layout=MMA)
    z = gl.zeros([2, NP2, BT], gl.float32, layout=MMA)
    b0 = gl.convert_layout(n0, LB)
    b1 = gl.convert_layout(n1, LB)
    b2 = gl.convert_layout(n2, LB)
    b3 = gl.convert_layout(n3, LB)
    async_copy.wait_group(3)
    gl.barrier()
    acc_s, acc_t, acc_b = _mma3(s0.permute([0, 2, 1]).load(LA), b0, acc_s, acc_t, acc_b, z, KH)
    async_copy.wait_group(2)
    gl.barrier()
    acc_s, acc_t, acc_b = _mma3(s1.permute([0, 2, 1]).load(LA), b1, acc_s, acc_t, acc_b, z, KH)
    async_copy.wait_group(1)
    gl.barrier()
    acc_s, acc_t, acc_b = _mma3(s2.permute([0, 2, 1]).load(LA), b2, acc_s, acc_t, acc_b, z, KH)
    async_copy.wait_group(0)
    gl.barrier()
    acc_s, acc_t, acc_b = _mma3(s3.permute([0, 2, 1]).load(LA), b3, acc_s, acc_t, acc_b, z, KH)
    # the two k-half partials meet in shared memory (one barrier); gl.sum over the
    # warp-distributed batch dim lowers to a longer shuffle + smem sequence.
    SR: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    LR: gl.constexpr = gl.BlockedLayout([2, 1, 1], [1, 4, 8], [1, 4, 1], [2, 1, 0])
    if FP64_ACC:
        red64 = gl.allocate_shared_memory(gl.float64, [2, NP2, BT], SR)
        red64.store(acc_b + (acc_s + acc_t).to(gl.float64))
        gl.barrier()
        acc = gl.sum(red64.load(LR), axis=0).to(gl.float32)
    else:
        red = gl.allocate_shared_memory(gl.float32, [2, NP2, BT], SR)
        red.store(acc_b + (acc_s + acc_t))
        gl.barrier()
        acc = gl.sum(red.load(LR), axis=0)

    LO: gl.constexpr = gl.SliceLayout(0, LR)
    no = gl.arange(0, NP2, layout=gl.SliceLayout(1, LO))
    to = pm * BT + gl.arange(0, BT, layout=gl.SliceLayout(0, LO))
    gl.store(mix_ptr + (ks * M + to[None, :]) * NOUT + no[:, None], acc,
             mask=(no < NOUT)[:, None] & (to < M)[None, :])


@gluon.jit
def _rcp_g(s):
    # MUFU.RCP, <= 1 ulp (x * rcp is within 2 ulp of x / s); the operands are
    # sums of positives >= eps, never denormal.  The IEEE `1.0 / s` or div.full
    # forms cost 2.5x the issue slots per norm.
    return gl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [s],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _rcp_nr(s):
    # rcp.approx refined by one Newton step, r + r * (1 - s * r): ~0.5 ulp.
    r = _rcp_g(s)
    return gl.fma(gl.fma(-s, r, 1.0), r, r)


@gluon.jit
def _div_c(x, s):
    # x / s as x * rcp.approx(s) plus one FMA correction of the quotient,
    # q + (x - s * q) * r: within rounding of the IEEE quotient.  A div.rn per
    # element cost ~12 us per call in the Sinkhorn loop, this ~1 us; it is used
    # only in the last iteration (see the finish kernel).
    r = _rcp_g(s)
    q = x * r
    return gl.fma(gl.fma(-s, q, x), r, q)


@gluon.jit
def _sigmoid_g(x):
    return _rcp_nr(1.0 + gl.exp(-x))


# The SINK - 2 middle (row, column) normalisation pairs of the Sinkhorn in scale-vector
# form: cm = diag(R) V diag(C) with V the matrix entering the loop.  A row norm sets
# R_r <- R_r / (R_r S_r + eps) with S_r = sum_c V_rc C_c, i.e. U = 1/R is updated as
# U <- S + eps * U (no division), then R = rcp(U); columns likewise (W = 1/C).  Each
# step's rounding acts as a row/column rescaling, absorbed by the later norms as in the
# elementwise form; on random logits the comb error is 0.75-0.85x the elementwise
# loop's, 126 vs 135 cycles per iteration (fewer FP32 issues on the critical path).
# Thread layout = the finish's L44: a thread holds rows 0..3 of two columns
# (inputs $16..$23 = r0c0, r0c1, r1c0, r1c1, ...), its xor-1 lane the other two.
# $24 = eps, $32 = eps / 2, $40 = iterations (>= 1).  The eps terms ride in the
# FMA chains (row partials start from eps/2 * U in each lane, column sums from
# eps * W, then add r0..r3 as their staggered rcps land): -12 cycles/iteration.
# Outputs: $0..$7 = V diag(C) (R is NOT applied) and $8..$15 = the final row
# norm's denominator in those units, S_r + eps * U_r (= (R S + eps) / R), so the
# last row normalisation divides V C directly - one rounding fewer than
# materialising diag(R) V diag(C) first (comb max error 1.15x -> 1.13x of v2).
_SINK_ASM = gl.constexpr("""{
.reg .pred lp;
.reg .u32 it;
.reg .f32 v00,v01,v10,v11,v20,v21,v30,v31,r0,r1,r2,r3,u0,u1,u2,u3,ca,cb,wa,wb;
.reg .f32 p0,p1,p2,p3,q0,q1,q2,q3,ta,tb,sa,sb,h0,h1,h2,h3;
mov.b32 v00, $16; mov.b32 v01, $17; mov.b32 v10, $18; mov.b32 v11, $19;
mov.b32 v20, $20; mov.b32 v21, $21; mov.b32 v30, $22; mov.b32 v31, $23;
mov.f32 u0, 0f3F800000; mov.f32 u1, 0f3F800000; mov.f32 u2, 0f3F800000; mov.f32 u3, 0f3F800000;
mov.f32 wa, 0f3F800000; mov.f32 wb, 0f3F800000; mov.f32 ca, 0f3F800000; mov.f32 cb, 0f3F800000;
mov.b32 h0, $32; mov.b32 h1, $32; mov.b32 h2, $32; mov.b32 h3, $32;
mov.u32 it, $40;
SLOOP:
fma.rn.f32 p0, v00, ca, h0; fma.rn.f32 p1, v10, ca, h1; fma.rn.f32 p2, v20, ca, h2; fma.rn.f32 p3, v30, ca, h3;
fma.rn.f32 p0, v01, cb, p0; fma.rn.f32 p1, v11, cb, p1; fma.rn.f32 p2, v21, cb, p2; fma.rn.f32 p3, v31, cb, p3;
shfl.sync.bfly.b32 q0, p0, 1, 31, -1;
shfl.sync.bfly.b32 q1, p1, 1, 31, -1;
shfl.sync.bfly.b32 q2, p2, 1, 31, -1;
shfl.sync.bfly.b32 q3, p3, 1, 31, -1;
add.f32 u0, p0, q0; add.f32 u1, p1, q1; add.f32 u2, p2, q2; add.f32 u3, p3, q3;
mul.f32 h0, $32, u0; mul.f32 h1, $32, u1; mul.f32 h2, $32, u2; mul.f32 h3, $32, u3;
rcp.approx.ftz.f32 r0, u0; rcp.approx.ftz.f32 r1, u1;
rcp.approx.ftz.f32 r2, u2; rcp.approx.ftz.f32 r3, u3;
mul.f32 ta, $24, wa; mul.f32 tb, $24, wb;
fma.rn.f32 ta, v00, r0, ta; fma.rn.f32 tb, v01, r0, tb;
fma.rn.f32 ta, v10, r1, ta; fma.rn.f32 tb, v11, r1, tb;
fma.rn.f32 ta, v20, r2, ta; fma.rn.f32 tb, v21, r2, tb;
fma.rn.f32 wa, v30, r3, ta; fma.rn.f32 wb, v31, r3, tb;
rcp.approx.ftz.f32 ca, wa; rcp.approx.ftz.f32 cb, wb;
sub.u32 it, it, 1;
setp.ne.u32 lp, it, 0;
@lp bra SLOOP;
fma.rn.f32 p0, v00, ca, h0; fma.rn.f32 p1, v10, ca, h1; fma.rn.f32 p2, v20, ca, h2; fma.rn.f32 p3, v30, ca, h3;
fma.rn.f32 p0, v01, cb, p0; fma.rn.f32 p1, v11, cb, p1; fma.rn.f32 p2, v21, cb, p2; fma.rn.f32 p3, v31, cb, p3;
shfl.sync.bfly.b32 q0, p0, 1, 31, -1;
shfl.sync.bfly.b32 q1, p1, 1, 31, -1;
shfl.sync.bfly.b32 q2, p2, 1, 31, -1;
shfl.sync.bfly.b32 q3, p3, 1, 31, -1;
add.f32 $8, p0, q0; add.f32 $10, p1, q1; add.f32 $12, p2, q2; add.f32 $14, p3, q3;
mov.b32 $9, $8; mov.b32 $11, $10; mov.b32 $13, $12; mov.b32 $15, $14;
mul.f32 $0, v00, ca; mul.f32 $1, v01, cb; mul.f32 $2, v10, ca; mul.f32 $3, v11, cb;
mul.f32 $4, v20, ca; mul.f32 $5, v21, cb; mul.f32 $6, v30, ca; mul.f32 $7, v31, cb;
}""")
_SINK_CONS = gl.constexpr("=r," * 16 + "r," * 8 + "r," * 8 + "r," * 8 + ",".join(["r"] * 8))


@gluon.jit
def _glu_finish_kernel(mix_ptr, sqr_ptr, scale_ptr, base_ptr, rc_ptr, nw_ptr,
                       post_out_ptr, comb_out_ptr, li_ptr, M,
                       rms_eps, hc_pre_eps, hc_sink_eps, post_mult, norm_eps,
                       HIDDEN: gl.constexpr, NOUT: gl.constexpr, KS: gl.constexpr,
                       KSP2: gl.constexpr, SINK: gl.constexpr, ONEBAR: gl.constexpr):
    """Finish for hc == 4 in Gluon: grid (M, 2), 4 warps.

    CTA (m, 0): the [KSP2, 8, 4] partial tile with columns remapped
    n' = (n + 8) mod 32, so row 2 = pre, row 3 = post, rows 4..7 = the 4x4 comb
    (rows 0, 1 masked).  The KS partials are reduced in a hybrid layout (KSP2/16
    in-thread, then a lane tree): a purely in-thread sequential sum pushed
    tf32_trap_m32's comb error to 7.0e-4 of an 8.0e-4 budget, a full lane tree
    cost +1.1 us.  post is stored from row 3's lanes; the affine-transformed
    rows go to a [8, 4] shared buffer and rows 4..7 come back replicated in
    every thread, so each Sinkhorn row/column sum is an in-thread sum and each
    norm one rcp.approx.
    CTA (m, 1): rc [4, HIDDEN] and the norm weight are loaded FIRST, then the
    pre partials; the collapse over the 4 streams is in-thread.
    """
    NW: gl.constexpr = 4
    HC: gl.constexpr = 4
    m = gl.program_id(0)
    part = gl.program_id(1)
    LS: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    kq = gl.arange(0, KSP2, layout=LS)
    if part == 0:
        L44: gl.constexpr = gl.BlockedLayout([4, 2], [1, 32], [1, NW], [1, 0])
        if ONEBAR:
            # The KS partials are reduced in exactly the order of a [KSP2, 8, 4] tile in
            # BlockedLayout([KSP2/16, 1, 4], [4, 8, 1], [NW, 1, 1]) summed over axis 0
            # (8 in-thread, lane xor 16/8, then warps as (w0 + w2) + (w1 + w3)), and the
            # sqrsum partials as a 1-D [KSP2] sum over 32 lanes x NW warps (lane xor 16..1,
            # then the same warp combine), but with ONE CTA barrier: every warp reduces its
            # own tile quarter to a per-warp partial, the partials meet once in shared
            # memory, and the sqrsum is reduced redundantly by every warp (all NW rows per
            # thread).  The default lowering took 5 barrier rounds (~600 cycles) here.
            r4 = gl.arange(0, 4, layout=gl.SliceLayout(1, L44))
            c4 = gl.arange(0, 4, layout=gl.SliceLayout(0, L44))
            # small loads first, pinned ahead of the partials (see `_clk_now`)
            bs_c = gl.load(base_ptr + 2 * HC + r4[:, None] * HC + c4[None, :])
            bs_p = gl.load(base_ptr + HC + c4[None, :] + r4[:, None] * 0)
            s1 = gl.load(scale_ptr + 1)
            s2 = gl.load(scale_ptr + 2)
            pin = _clk_now() >= 0
            LSQ: gl.constexpr = gl.BlockedLayout([NW, 1], [1, 32], [1, NW], [1, 0])
            wq = gl.arange(0, NW, layout=gl.SliceLayout(1, LSQ))
            lq = gl.arange(0, 32, layout=gl.SliceLayout(0, LSQ))
            kqs = wq[:, None] * 32 + lq[None, :]
            sqv = gl.load(sqr_ptr + kqs * M + m, mask=(kqs < KS) & pin, other=0.0)
            KW: gl.constexpr = KSP2 // NW
            L4: gl.constexpr = gl.BlockedLayout([1, KSP2 // 16, 1, 4], [1, 4, 8, 1], [NW, 1, 1, 1],
                                                [3, 2, 1, 0])
            wi = gl.arange(0, NW, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L4))))
            ji = gl.arange(0, KW, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L4))))
            ri = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L4))))
            ci = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L4))))
            ks4 = wi[:, None, None, None] * KW + ji[None, :, None, None]
            # columns remapped n' = (n + 8) mod 32: row 3 = post, rows 4..7 = the 4x4 comb
            src = (ri[None, None, :, None] * 4 + ci[None, None, None, :] + 24) % 32
            tile = gl.load(mix_ptr + (ks4 * M + m) * NOUT + src,
                           mask=(ks4 < KS) & (src < NOUT) & pin, other=0.0)
            wpart = gl.sum(tile, axis=1)
            SP: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
            pbuf = gl.allocate_shared_memory(gl.float32, [NW, 8, 4], SP)
            pbuf.store(wpart)
            sqw = gl.reshape(gl.sum(sqv, axis=1), [2, 2])
            sq = gl.sum(gl.sum(sqw, axis=0), axis=0)
            rms = gl.rsqrt(sq / (HC * HIDDEN) + rms_eps)
            gl.barrier()
            # rows 0..3 (row 3 = post) and 4..7 (comb) of the four warp partials
            pv = pbuf.reshape([NW * 8, 4])
            pp = ((pv.slice(0, 4, dim=0).load(L44) + pv.slice(16, 4, dim=0).load(L44))
                  + (pv.slice(8, 4, dim=0).load(L44)
                     + pv.slice(24, 4, dim=0).load(L44)))
            pc = ((pv.slice(4, 4, dim=0).load(L44) + pv.slice(20, 4, dim=0).load(L44))
                  + (pv.slice(12, 4, dim=0).load(L44)
                     + pv.slice(28, 4, dim=0).load(L44)))
            rr = r4[:, None] + c4[None, :] * 0
            vp = (pp * rms) * s1 + bs_p
            po = _sigmoid_g(vp) * post_mult
            gl.store(post_out_ptr + m * HC + c4[None, :] + rr * 0, po, mask=rr == 3)
            # Each thread holds two columns (lanes 0, 1 the four; the rest replicate):
            # row sums take one shuffle, column sums are in-thread, and each thread
            # issues half the FP32 work of a fully replicated 4x4 - the loop is
            # issue-bound (~147 cycles/iteration replicated); -0.13..-0.2 us.
            cm = (pc * rms) * s2 + bs_c
        else:
            # (M > 16: the default lowering measured 1.5 % faster at M = 32)
            LP: gl.constexpr = gl.BlockedLayout([KSP2 // 16, 1, 4], [4, 8, 1], [NW, 1, 1], [2, 1, 0])
            L2P: gl.constexpr = gl.SliceLayout(0, LP)
            ksi = gl.arange(0, KSP2, layout=gl.SliceLayout(1, gl.SliceLayout(2, LP)))
            ri = gl.arange(0, 8, layout=gl.SliceLayout(1, L2P))
            ci = gl.arange(0, 4, layout=gl.SliceLayout(0, L2P))
            src = (ri[:, None] * 4 + ci[None, :] + 24) % 32
            valid = src < NOUT
            ks3 = ksi[:, None, None]
            # every load before the first reduction: a gl.sum lowers to CTA barriers, and a
            # load placed after one is issued only after it (an extra L2 round trip)
            bs = gl.load(base_ptr + src, mask=valid, other=0.0)
            s1 = gl.load(scale_ptr + 1)
            s2 = gl.load(scale_ptr + 2)
            tile = gl.load(mix_ptr + (ks3 * M + m) * NOUT + src[None, :, :],
                           mask=(ks3 < KS) & valid[None, :, :], other=0.0)
            sq = gl.sum(gl.load(sqr_ptr + kq * M + m, mask=kq < KS, other=0.0), axis=0)
            rms = gl.rsqrt(sq / (HC * HIDDEN) + rms_eps)
            col = gl.sum(tile, axis=0) * rms
            rr = ri[:, None] + ci[None, :] * 0
            v = col * gl.where(rr == 3, s1, s2) + bs
            po = _sigmoid_g(v) * post_mult
            gl.store(post_out_ptr + m * HC + ci[None, :] + rr * 0, po, mask=rr == 3)
            SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            smem = gl.allocate_shared_memory(gl.float32, [8, 4], SL)
            smem.store(v)
            gl.barrier()
            # Each thread holds two columns (lanes 0, 1 the four; the rest replicate):
            # row sums take one shuffle, column sums are in-thread, and each thread
            # issues half the FP32 work of a fully replicated 4x4 - the loop is
            # issue-bound (~147 cycles/iteration replicated); -0.13..-0.2 us.
            cm = smem.slice(4, 4, dim=0).load(L44)
        # softmax over each row, + eps, then 1 + (SINK - 1) x (row, col) norms
        e = gl.exp(cm - gl.max(cm, axis=1)[:, None])
        cm = e * _rcp_g(gl.sum(e, axis=1))[:, None] + hc_sink_eps
        cm = cm * _rcp_g(gl.sum(cm, axis=0) + hc_sink_eps)[None, :]
        # The rounding of every normalisation but the last is absorbed by the
        # ones after it (it acts as a row or column rescaling); the last one's
        # is not, so only the final row and column normalisation divides with a
        # corrected quotient.  All iterations corrected measured the same error
        # at +1 us per call; none corrected, 1.09-1.14x TileLang's comb error.
        if SINK > 2:
            z = gl.zeros_like(cm)
            y, d = gl.inline_asm_elementwise(_SINK_ASM, _SINK_CONS,
                                             [cm, z + hc_sink_eps, z + hc_sink_eps * 0.5, (z + (SINK - 2)).to(gl.int32)],
                                             dtype=(gl.float32, gl.float32), is_pure=True, pack=8)
        if SINK > 2:
            cm = _div_c(y, d)
            cm = _div_c(cm, (gl.sum(cm, axis=0) + hc_sink_eps)[None, :] + gl.zeros_like(cm))
        elif SINK == 2:
            cm = _div_c(cm, (gl.sum(cm, axis=1) + hc_sink_eps)[:, None] + gl.zeros_like(cm))
            cm = _div_c(cm, (gl.sum(cm, axis=0) + hc_sink_eps)[None, :] + gl.zeros_like(cm))
        r4 = gl.arange(0, 4, layout=gl.SliceLayout(1, L44))
        c4 = gl.arange(0, 4, layout=gl.SliceLayout(0, L44))
        gl.store(comb_out_ptr + m * (HC * HC) + r4[:, None] * HC + c4[None, :], cm)
    else:
        LRC: gl.constexpr = gl.BlockedLayout([4, 8], [1, 32], [1, NW], [1, 0])
        jr = gl.arange(0, 4, layout=gl.SliceLayout(1, LRC))
        h = gl.arange(0, HIDDEN, layout=gl.SliceLayout(0, LRC))
        rct = gl.load(rc_ptr + m * (HC * HIDDEN) + jr[:, None] * HIDDEN + h[None, :])
        w = gl.load(nw_ptr + h)
        LQ: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NW, 1], [1, 0])
        kq2 = gl.arange(0, KSP2, layout=gl.SliceLayout(1, LQ))
        cq = gl.arange(0, 4, layout=gl.SliceLayout(0, LQ))
        bq = gl.load(base_ptr + cq)
        s0 = gl.load(scale_ptr)
        mixq = gl.load(mix_ptr + (kq2[:, None] * M + m) * NOUT + cq[None, :],
                       mask=(kq2 < KS)[:, None] & (cq < 4)[None, :], other=0.0)
        sq = gl.sum(gl.load(sqr_ptr + kq * M + m, mask=kq < KS, other=0.0), axis=0)
        rms = gl.rsqrt(sq / (HC * HIDDEN) + rms_eps)
        pre = _sigmoid_g(gl.sum(mixq, axis=0) * rms * s0 + bq) + hc_pre_eps
        pre = gl.convert_layout(pre, gl.SliceLayout(1, LRC))
        o = gl.sum(pre[:, None] * rct.to(gl.float32), axis=0)
        ob = o.to(gl.bfloat16).to(gl.float32)
        rn = gl.rsqrt(gl.sum(o * o, axis=0) / HIDDEN + norm_eps)
        gl.store(li_ptr + m * HIDDEN + h, (ob * rn * w.to(gl.float32)).to(gl.bfloat16))


# Kernel A tunables.  Module-level constants, not autotune: a graph capture must
# see one fixed config.  Swept for the Gluon kernel A: HB 16/64 +0.35..1.4 us,
# 8 warps and k_width 2 worse, plain register fn loads +0.3 us, unswizzled
# cp.async buffers +1.1 us (bank conflicts).
HB_GLU = 32
BT_T = 8            # token block of the transposed M <= 8 kernel (the mma's n8)
WARPS_GLU = 4
BM_A = 16           # bf16/tf32 mma is m16n8k16: 16 rows is the minimum unit


def post_prenorm(x, res, post, comb, fn, rc, mix, sqr, *, has_post=True, round_mix=False,
                 fp64_acc=False, hs_reuse=False):
    """Kernel A alone: ``rc`` [M, 4, H] bf16, the prenorm GEMM partials ``mix``
    [H / HB_GLU, M, 24] and the squared-sum partials ``sqr`` [H / HB_GLU, M].

    ``x`` [M, H] bf16, ``res`` [M, 4, H] bf16, ``post`` [M, 4] and ``comb``
    [M, 4, 4] fp32, ``fn`` [24, 4 * H] fp32, all contiguous; H a multiple of
    HB_GLU. Four compile-time options, all off for GLM-5.3-Flash (whose
    kernels are then the ones measured above, unchanged):
      * ``has_post=False``: no post step; the streams are ``res`` itself (the
        delayed ``pre``-only call). ``x``, ``post`` and ``comb`` are not used,
        but must still be tensors of at least the sizes above;
      * ``round_mix=True``: the GEMM and the squared sum read the streams as
        stored, rounded to bf16 (the DeepSeek-V4.1 reference computes its
        coefficients from the bf16 stream), not the fp32 values before the
        rounding;
      * ``fp64_acc=True``: inside each hidden slice the big (hi * hi)
        products' 8-wide mma blocks, the two small terms, the two k-halves
        and the squared sum are accumulated in FP64 (full rate on this card)
        and each partial is rounded once to FP32; the mma passes and their
        split are unchanged. At DeepSeek-V4.1's 4 x 5120 streams the FP32
        accumulation was kernel A's largest error source in the
        coefficients (CPU emulation: pre max error 1.3e-7 -> 7e-8).
      * ``hs_reuse=True``: from 17 rows (more than one 16-row block of
        `_glu_post_prenorm_kernel`) the hi/lo staging of operand A reuses two
        shared-memory buffer pairs (24 KB per CTA instead of 32, one wave at
        32 rows). The same products in the same order: the same bits.
    """
    M, hidden = res.shape[0], res.shape[-1]
    nout = fn.shape[0]
    ksb = hidden // HB_GLU
    assert hidden % HB_GLU == 0 and mix.shape == (ksb, M, nout), (hidden, mix.shape)
    if M <= BT_T:
        _glu_post_prenorm_tb_kernel[(triton.cdiv(M, BT_T), ksb)](
            x, res, post, comb, fn, rc, mix, sqr, M,
            HIDDEN=hidden, NOUT=nout, HB=HB_GLU, BT=BT_T,
            NP2=triton.next_power_of_2(nout), HAS_POST=has_post, ROUND_MIX=round_mix,
            FP64_ACC=fp64_acc, num_warps=4)
    else:
        _glu_post_prenorm_kernel[(triton.cdiv(M, BM_A), ksb)](
            x, res, post, comb, fn, rc, mix, sqr, M,
            HIDDEN=hidden, NOUT=nout, HB=HB_GLU, BM=BM_A,
            NP2=triton.next_power_of_2(nout), NW=WARPS_GLU, HAS_POST=has_post,
            ROUND_MIX=round_mix, FP64_ACC=fp64_acc,
            HS_REUSE=hs_reuse and triton.cdiv(M, BM_A) > 1, num_warps=WARPS_GLU)


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
):
    """Exactly `mhc_fused_post_pre_tilelang`.

    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur) with
    shapes (..., hc, H), (..., hc, 1), (..., hc, hc), (..., H).
    """
    assert residual.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32
    assert norm_weight is not None, "the decode path always fuses the input RMSNorm"

    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    # the use_ampere_mhc_decode_v3 shapes; no kernel here takes any other
    assert hc == 4 and hidden % 1024 == 0, (hc, hidden)
    nout = hc * 2 + hc * hc
    outer = residual.shape[:-2]

    res = residual.reshape(-1, hc, hidden).contiguous()
    M = res.shape[0]
    xf = x.reshape(M, hidden).contiguous()
    pf = post_layer_mix.reshape(M, hc).contiguous()
    cf = comb_res_mix.reshape(M, hc, hc).contiguous()
    fnf = fn.contiguous()
    nw = norm_weight
    if nw.dtype != torch.bfloat16:
        nw = nw.to(torch.bfloat16)
    nw = nw.contiguous()

    dev = res.device
    ksb = hidden // HB_GLU

    rc = torch.empty(M, hc, hidden, dtype=torch.bfloat16, device=dev)
    mix = torch.empty(ksb, M, nout, dtype=torch.float32, device=dev)
    sqr = torch.empty(ksb, M, dtype=torch.float32, device=dev)
    post_out = torch.empty(M, hc, dtype=torch.float32, device=dev)
    comb_out = torch.empty(M, hc * hc, dtype=torch.float32, device=dev)
    li = torch.empty(M, hidden, dtype=torch.bfloat16, device=dev)

    post_prenorm(xf, res, pf, cf, fnf, rc, mix, sqr)
    _glu_finish_kernel[(M, 2)](
        mix, sqr, hc_scale, hc_base, rc, nw, post_out, comb_out, li, M,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, norm_eps,
        HIDDEN=hidden, NOUT=nout, KS=ksb, KSP2=max(triton.next_power_of_2(ksb), 32),
        SINK=sinkhorn_repeat, ONEBAR=M <= 16, num_warps=4)

    return (rc.view(*outer, hc, hidden),
            post_out.view(*outer, hc, 1),
            comb_out.view(*outer, hc, hc),
            li.view(*outer, hidden))


def warmup(ms, hidden=4096, hc=4, sinkhorn=20, device="cuda"):
    """Compile every (M, config) this run will use, before any graph capture."""
    dev = torch.device(device)
    nout = hc * 2 + hc * hc
    for M in sorted(set(int(m) for m in ms)):
        case = dict(
            x=torch.zeros(M, hidden, device=dev, dtype=torch.bfloat16),
            residual=torch.zeros(M, hc, hidden, device=dev, dtype=torch.bfloat16),
            post_layer_mix=torch.zeros(M, hc, 1, device=dev, dtype=torch.float32),
            comb_res_mix=torch.zeros(M, hc, hc, device=dev, dtype=torch.float32),
            fn=torch.zeros(nout, hc * hidden, device=dev, dtype=torch.float32),
            hc_scale=torch.ones(3, device=dev, dtype=torch.float32),
            hc_base=torch.zeros(nout, device=dev, dtype=torch.float32),
            norm_weight=torch.ones(hidden, device=dev, dtype=torch.bfloat16))
        mhc_fused_post_pre(
            rms_eps=1e-5, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0, sinkhorn_repeat=sinkhorn, **case)
    torch.cuda.synchronize()
