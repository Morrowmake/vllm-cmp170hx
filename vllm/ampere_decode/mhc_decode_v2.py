# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC fused post+pre for GLM-5.3-Flash decode on sm_80, M <= 32 (v2).

Drop-in for `vllm.model_executor.kernels.mhc.tilelang.mhc_fused_post_pre_tilelang`:
same signature, same four outputs, same dtypes. Dispatched by
`vllm.ampere_decode.use_ampere_mhc_decode_v2` (env `VLLM_GLM5_DECODE_MHC_V2`,
default off) for 1 <= M <= `VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS` (32), where it
replaces both the v1 kernel (`mhc_decode.py`, M <= 8) and TileLang (M > 8).

For hc == 4 (GLM-5.3-Flash) both launches are Gluon kernels
(`triton.experimental.gluon`, written against the pinned Triton 3.7.1):

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

The Triton kernels below are the fallback for other hc / hidden and are not
reached by the gate.

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
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2


def num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _post_prenorm_kernel(
    x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr,
    rc_ptr, mix_ptr, sqr_ptr,
    M,
    HIDDEN: tl.constexpr, HC: tl.constexpr, NOUT: tl.constexpr,
    HB: tl.constexpr, NB: tl.constexpr, BMS: tl.constexpr,
):
    """residual_cur = post*x + comb^T @ residual, plus the prenorm GEMM partials.

    grid (M, KSPLIT, NOUT // NB).  CTA (m, ks, ng) owns token m,
    hidden[ks*HB : (ks+1)*HB] of all HC streams, and the NB prenorm outputs
    [ng*NB, ng*NB + NB).  Against the previous version (grid (M, KSPLIT), each
    CTA doing all 24 outputs) three things changed, each measured separately at
    M=4 with the 90-deep rotation under graph replay:

      1. The 24 outputs are split across CTAs (NOUT // NB of them).  Per-CTA
         work drops 3x and the CTA count triples; `fn` traffic is unchanged,
         because each CTA still reads only its own NB rows.
      2. The `fn` tile is addressed as a 3D [NB, HC, HB] block whose innermost
         index is a plain `arange`, instead of a 2D block indexed by a
         `tl.reshape`d offset vector.  Triton can then prove contiguity and
         emit 128-bit loads; with the reshaped index it emitted narrow ones and
         the kernel was latency-bound on them.  This was worth ~1.5 us.
      3. That `fn` load is issued FIRST, before the hc-mix, which depends on
         none of it.  16 KB per CTA against ~1 KB for everything else, so its
         HBM latency overlaps the mix rather than following it.

    Total at M=4: 15.58 -> 12.6 us for the pair, i.e. 1.03x the incumbent's
    13.0 us, and the family score over M = 1,2,4,8,16,32 goes 0.858 -> 0.99.

    What did NOT work, measured here, in case it looks tempting later:

      * `tl.dot(..., input_precision="tf32x3")` over K = HC*HB with all tokens
         padded into one BLOCK_M=16 mma - the fp32 [32, HC*HB] operand spills:
         17.4 us at HB=32, 22.1 at HB=64, 244 at HB=128.  At these shapes the
         prenorm reduction is 1.6 MFMA against a 6 MB read; it is memory-latency
         bound, not FMA-throughput bound, so the tensor cores buy nothing and
         the register cost of feeding them is fatal.  (This is the opposite of
         prefill/mhc_prenorm at M=1152, where the same op IS FLOP-bound.)
      * Moving the token loop ENTIRELY inside the CTA so `fn` is read once per
         call instead of once per token (1.57 MB vs 6.29 MB at M=4): 1.08x at
         M=1 but 0.79x at M=4 and 0.54x at M=16 - the serial token iterations
         cost ~1.5 us each, far more than the re-read they save.  A PARTIAL
         version of it does pay off once the grid is saturated, though: see
         `BMS_A` below, which blocks 2 tokens per CTA from M=32 and is worth
         8.4 % there (31.84 -> 29.22 us) while being neutral at M=16 and off
         entirely at the decode shapes.

    `residual_cur` and `sqrsum` are produced by the ng == 0 CTAs only; the other
    output groups recompute the mix in registers (a re-read of `residual` that
    L2 serves) and never write it.
    """
    pm = tl.program_id(0)
    ks = tl.program_id(1)
    ng = tl.program_id(2)
    h = ks * HB + tl.arange(0, HB)
    jr = tl.arange(0, HC)
    nr = ng * NB + tl.arange(0, NB)

    # Issue the big `fn` read FIRST: it is 16 KB per CTA against ~1 KB for
    # everything else, and the hc-mix below depends on none of it, so its
    # ~300-cycle HBM latency overlaps the mix instead of being paid after it.
    # The tile is addressed as a 3D [NB, HC, HB] block whose innermost index is
    # a plain arange, so Triton can prove contiguity and emit 128-bit loads;
    # going through `tl.reshape` of a computed index vector instead costs the
    # vectorisation and ~1.5 us of this kernel.
    f = tl.load(fn_ptr + nr[:, None, None] * (HC * HIDDEN)
                + jr[None, :, None] * HIDDEN + h[None, None, :])

    # BMS tokens share that one `fn` tile, so `fn` is read M/BMS times per call
    # instead of M times.  BMS=1 at the decode shapes (M <= 8), where the
    # earlier full-serial version lost; see the note above and `_config`.
    for b in tl.static_range(BMS):
        m = pm * BMS + b
        if m < M:
            xv = tl.load(x_ptr + m * HIDDEN + h).to(tl.float32)          # [HB]
            pv = tl.load(post_ptr + m * HC + jr)                         # [HC] fp32
            newr = pv[:, None] * xv[None, :]                             # [HC, HB]
            for k in tl.static_range(HC):
                ck = tl.load(comb_ptr + m * HC * HC + k * HC + jr)
                rk = tl.load(res_ptr + m * HC * HIDDEN + k * HIDDEN + h).to(tl.float32)
                newr += ck[:, None] * rk[None, :]

            if ng == 0:
                tl.store(rc_ptr + m * HC * HIDDEN + jr[:, None] * HIDDEN
                         + h[None, :], newr.to(tl.bfloat16))
                tl.store(sqr_ptr + ks * M + m, tl.sum(newr * newr))

            # NB prenorm-GEMM outputs from the fp32 residual_cur still in
            # registers (the semantic pre-rounding boundary), in a fixed
            # reduction order, so the result is bitwise reproducible and no
            # TF32-width mantissa is involved.
            tl.store(mix_ptr + (ks * M + m) * NOUT + nr,
                     tl.sum(tl.sum(f * newr[None, :, :], axis=2), axis=1))


@triton.jit
def _post_prenorm_mma_kernel(
    x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr,
    rc_ptr, mix_ptr, sqr_ptr,
    M,
    HIDDEN: tl.constexpr, HC: tl.constexpr, NOUT: tl.constexpr,
    HB: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, NP2: tl.constexpr,
    PREC: tl.constexpr,
):
    """Kernel A for the LARGE-M shapes: the prenorm GEMM on the tensor cores.

    grid (cdiv(M, BM), hidden // HB).  Each CTA owns BM tokens and an HB-wide
    slice of `hidden` across all HC streams, and contracts it against `fn` with
    `tl.dot(..., input_precision="tf32x3")` in BK-wide chunks.

    WHY THIS EXISTS ALONGSIDE THE SCALAR PATH.  Measurement showed tf32x3 losing
    badly and the scalar-FMA reduction was kept - but that measurement was at M=4,
    where a BLOCK_M=16 mma wastes 4x on padding, and with the whole K extent in
    one register tile, which spilled (17.4 us at HB=32, 244 at HB=128).  Two
    things are different here:
      * this path only runs from M = MMA_FROM_M (16), where BM=16 is exactly
        filled and the padding waste is zero;
      * K is consumed in BK-wide chunks (the shape `prefill/mhc_prenorm` found
        best at M=1152, where BLOCK_K=64 beat 128 and 256 everywhere), so the
        live operands are [BM, BK] + [NP2, BK] + the [BM, NP2] accumulator
        instead of a [NP2, HC*HB] monolith.
    Measurement showed that at M=16 the scalar path's reduction alone is 11.6 us of
    21.1 - the prenorm GEMM is 6.3 MFMA there, 0.95 us at the fp32 FMA peak and
    ~0.12 us at the TF32 mma peak - so this is the one shape where the tensor
    cores have something to win.

    PRECISION.  tf32x3 is three TF32 terms (3 x 10 >= 24 mantissa bits) and
    lands on the explicit-FMA path's error: 4.88e-04 against the incumbent's
    4.25e-04 on the calibrated `tf32_trap`, inside the gate's 1.5x budget, and
    the same rule `prefill/mhc_prenorm` passed under.  `residual_cur`, `sqrsum`
    and the bf16 rounding boundaries are computed exactly as the scalar path
    does - only the 24-wide contraction moves to the mma.
    """
    pm = tl.program_id(0)
    ks = tl.program_id(1)
    mi = pm * BM + tl.arange(0, BM)
    mm = mi < M
    nr = tl.arange(0, NP2)
    vn = nr < NOUT
    kk = tl.arange(0, BK)

    acc = tl.zeros([BM, NP2], tl.float32)
    sq = tl.zeros([BM], tl.float32)

    # One chunk is one hc stream and a BK-wide piece of this CTA's `hidden`
    # slice, so the flattened K index is contiguous within a chunk and `fn`
    # loads stay vectorised.
    for j in tl.static_range(HC):
        pj = tl.load(post_ptr + mi * HC + j, mask=mm, other=0.0)        # [BM]
        for h0 in tl.static_range(0, HB, BK):
            h = ks * HB + h0 + kk
            xv = tl.load(x_ptr + mi[:, None] * HIDDEN + h[None, :],
                         mask=mm[:, None], other=0.0).to(tl.float32)
            newr = pj[:, None] * xv                                      # [BM, BK]
            for k in tl.static_range(HC):
                ck = tl.load(comb_ptr + mi * (HC * HC) + k * HC + j,
                             mask=mm, other=0.0)
                rk = tl.load(res_ptr + mi[:, None] * (HC * HIDDEN)
                             + k * HIDDEN + h[None, :],
                             mask=mm[:, None], other=0.0).to(tl.float32)
                newr += ck[:, None] * rk

            tl.store(rc_ptr + mi[:, None] * (HC * HIDDEN) + j * HIDDEN
                     + h[None, :], newr.to(tl.bfloat16), mask=mm[:, None])
            sq += tl.sum(newr * newr, axis=1)

            fb = tl.load(fn_ptr + nr[:, None] * (HC * HIDDEN) + j * HIDDEN
                         + h[None, :], mask=vn[:, None], other=0.0)      # [NP2, BK]
            acc += tl.dot(newr, tl.trans(fb), input_precision=PREC)

    tl.store(sqr_ptr + ks * M + mi, sq, mask=mm)
    tl.store(mix_ptr + (ks * M + mi[:, None]) * NOUT + nr[None, :], acc,
             mask=mm[:, None] & vn[None, :])


# --------------------------------------------------------------------------
# Kernel A in Gluon (explicit layouts), used for every M when hc == 4.
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
    acc_b = acc_b + mma_v2(ah, gl.where(lo8, bh, 0.0), z)
    acc_b = acc_b + mma_v2(ah, gl.where(lo8, 0.0, bh), z)
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
        acc_s = mma_v2(al, bh, acc_s)
        acc_t = mma_v2(ah, b - bh, acc_t)
        acc_b = acc_b + mma_v2(gl.where(lo8, ah, 0.0), bh, z)
        acc_b = acc_b + mma_v2(gl.where(lo8, 0.0, ah), bh, z)
    return acc_s, acc_t, acc_b


@gluon.jit
def _glu_post_prenorm_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr, rc_ptr, mix_ptr, sqr_ptr,
                             M, HIDDEN: gl.constexpr, NOUT: gl.constexpr, HB: gl.constexpr,
                             BM: gl.constexpr, NP2: gl.constexpr, NW: gl.constexpr):
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

    ob = rc_ptr + mi[:, None] * (HC * HIDDEN) + h[None, :]
    gl.store(ob, n0.to(gl.bfloat16), mask=m2)
    gl.store(ob + HIDDEN, n1.to(gl.bfloat16), mask=m2)
    gl.store(ob + 2 * HIDDEN, n2.to(gl.bfloat16), mask=m2)
    gl.store(ob + 3 * HIDDEN, n3.to(gl.bfloat16), mask=m2)
    sq = gl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, axis=1)
    gl.store(sqr_ptr + ks * M + mi, sq, mask=mm)

    acc_s = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    acc_b = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    acc_t = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    z = gl.zeros([BM, NP2], gl.float32, layout=MMA)
    # operand A split hi/lo once here (4 elements per thread) - see the docstring
    SA: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    hs0 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    hs1 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    hs2 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
    hs3 = gl.allocate_shared_memory(gl.float32, [BM, HB], SA)
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
    acc = acc_b + (acc_s + acc_t)

    mo = pm * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, MMA))
    no = gl.arange(0, NP2, layout=gl.SliceLayout(0, MMA))
    gl.store(mix_ptr + (ks * M + mo[:, None]) * NOUT + no[None, :], acc,
             mask=(mo < M)[:, None] & (no < NOUT)[None, :])


@gluon.jit
def _glu_post_prenorm_tb_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr, rc_ptr, mix_ptr, sqr_ptr,
                                M, HIDDEN: gl.constexpr, NOUT: gl.constexpr, HB: gl.constexpr,
                                BT: gl.constexpr, NP2: gl.constexpr):
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
    # (k-half, k, n): k contiguous, 4 fp32 = 16 B per cp.async
    LF: gl.constexpr = gl.BlockedLayout([1, 4, 1], [1, 4, 8], [2, 1, 2], [1, 2, 0])
    # (k-half, k, t) for x / residual: 2 bf16 per thread along k
    LT: gl.constexpr = gl.BlockedLayout([1, 2, 1], [1, 8, 4], [1, 2, 2], [1, 2, 0])
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2, 1],
                                                  instr_shape=[1, 16, 8])
    LA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=1)
    LB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=1)
    SF: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 8, [1, 2, 0])
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

    ob = rc_ptr + t3 * (HC * HIDDEN) + h3
    gl.store(ob, n0.to(gl.bfloat16), mask=m3)
    gl.store(ob + HIDDEN, n1.to(gl.bfloat16), mask=m3)
    gl.store(ob + 2 * HIDDEN, n2.to(gl.bfloat16), mask=m3)
    gl.store(ob + 3 * HIDDEN, n3.to(gl.bfloat16), mask=m3)
    sq = gl.sum(gl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, axis=1), axis=0)
    gl.store(sqr_ptr + ks * M + ti, sq, mask=tm)

    acc_s = gl.zeros([2, NP2, BT], gl.float32, layout=MMA)
    acc_t = gl.zeros([2, NP2, BT], gl.float32, layout=MMA)
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
    red = gl.allocate_shared_memory(gl.float32, [2, NP2, BT], SR)
    red.store(acc_b + (acc_s + acc_t))
    gl.barrier()
    LR: gl.constexpr = gl.BlockedLayout([2, 1, 1], [1, 4, 8], [1, 4, 1], [2, 1, 0])
    acc = gl.sum(red.load(LR), axis=0)

    LO: gl.constexpr = gl.SliceLayout(0, LR)
    no = gl.arange(0, NP2, layout=gl.SliceLayout(1, LO))
    to = pm * BT + gl.arange(0, BT, layout=gl.SliceLayout(0, LO))
    gl.store(mix_ptr + (ks * M + to[None, :]) * NOUT + no[:, None], acc,
             mask=(no < NOUT)[:, None] & (to < M)[None, :])


@gluon.jit
def _rcp_g(s):
    # MUFU.RCP, <= 1 ulp; operands are sums of positives >= eps (see `_rcp`).
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


@gluon.jit
def _glu_finish_kernel(mix_ptr, sqr_ptr, scale_ptr, base_ptr, rc_ptr, nw_ptr,
                       post_out_ptr, comb_out_ptr, li_ptr, M,
                       rms_eps, hc_pre_eps, hc_sink_eps, post_mult, norm_eps,
                       HIDDEN: gl.constexpr, NOUT: gl.constexpr, KS: gl.constexpr,
                       KSP2: gl.constexpr, SINK: gl.constexpr):
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
        LP: gl.constexpr = gl.BlockedLayout([KSP2 // 16, 1, 4], [4, 8, 1], [NW, 1, 1], [2, 1, 0])
        L2P: gl.constexpr = gl.SliceLayout(0, LP)
        ksi = gl.arange(0, KSP2, layout=gl.SliceLayout(1, gl.SliceLayout(2, LP)))
        ri = gl.arange(0, 8, layout=gl.SliceLayout(1, L2P))
        ci = gl.arange(0, 4, layout=gl.SliceLayout(0, L2P))
        src = (ri[:, None] * 4 + ci[None, :] + 24) % 32
        valid = src < NOUT
        ks3 = ksi[:, None, None]
        tile = gl.load(mix_ptr + (ks3 * M + m) * NOUT + src[None, :, :],
                       mask=(ks3 < KS) & valid[None, :, :], other=0.0)
        sq = gl.sum(gl.load(sqr_ptr + kq * M + m, mask=kq < KS, other=0.0), axis=0)
        bs = gl.load(base_ptr + src, mask=valid, other=0.0)
        s1 = gl.load(scale_ptr + 1)
        s2 = gl.load(scale_ptr + 2)
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
        L44: gl.constexpr = gl.BlockedLayout([4, 2], [1, 32], [1, NW], [1, 0])
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
        for _ in range(SINK - 2):
            cm = cm * _rcp_g(gl.sum(cm, axis=1) + hc_sink_eps)[:, None]
            cm = cm * _rcp_g(gl.sum(cm, axis=0) + hc_sink_eps)[None, :]
        if SINK >= 2:
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
        mixq = gl.load(mix_ptr + (kq2[:, None] * M + m) * NOUT + cq[None, :],
                       mask=(kq2 < KS)[:, None] & (cq < 4)[None, :], other=0.0)
        sq = gl.sum(gl.load(sqr_ptr + kq * M + m, mask=kq < KS, other=0.0), axis=0)
        bq = gl.load(base_ptr + cq)
        s0 = gl.load(scale_ptr)
        rms = gl.rsqrt(sq / (HC * HIDDEN) + rms_eps)
        pre = _sigmoid_g(gl.sum(mixq, axis=0) * rms * s0 + bq) + hc_pre_eps
        pre = gl.convert_layout(pre, gl.SliceLayout(1, LRC))
        o = gl.sum(pre[:, None] * rct.to(gl.float32), axis=0)
        ob = o.to(gl.bfloat16).to(gl.float32)
        rn = gl.rsqrt(gl.sum(o * o, axis=0) / HIDDEN + norm_eps)
        gl.store(li_ptr + m * HIDDEN + h, (ob * rn * w.to(gl.float32)).to(gl.bfloat16))


@triton.jit
def _rcp(s):
    # MUFU.RCP, <= 1 ulp (x * rcp is within 2 ulp of x / s); the operands are
    # sums of positives >= eps, never denormal.  The IEEE `1.0 / s` or div.full
    # forms cost 2.5x the issue slots per norm and lost 2.2 us at M = 4.
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [s],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _norm4(a, b, c, d, eps):
    """x / (sum + eps) for a 4-vector as x * rcp; the sum as tl.sum's pairwise tree."""
    r = _rcp((a + b) + (c + d) + eps)
    return a * r, b * r, c * r, d * r


@triton.jit
def _softmax4(a, b, c, d, eps):
    mx = tl.maximum(tl.maximum(a, b), tl.maximum(c, d))
    a = tl.exp(a - mx)
    b = tl.exp(b - mx)
    c = tl.exp(c - mx)
    d = tl.exp(d - mx)
    r = _rcp((a + b) + (c + d))
    return a * r + eps, b * r + eps, c * r + eps, d * r + eps


@triton.jit
def _pre_finish_kernel(
    mix_ptr, sqr_ptr, scale_ptr, base_ptr, rc_ptr, nw_ptr,
    post_out_ptr, comb_out_ptr, li_ptr,
    M,
    rms_eps, hc_pre_eps, hc_sink_eps, post_mult, norm_eps,
    HIDDEN: tl.constexpr, HC: tl.constexpr, NOUT: tl.constexpr,
    KS: tl.constexpr, KSP2: tl.constexpr, NP2: tl.constexpr, SINK: tl.constexpr,
):
    """Reduce partials; Sinkhorn || collapse + RMSNorm.  grid (M, 2).

    CTA (m, 0) reduces the 24 mixes, writes `post_mix` and runs the Sinkhorn;
    CTA (m, 1) reduces only the four `pre` mixes and collapses + RMSNorms the
    whole row in one register pass.  The two only share `pre`, so the ~2.8 us
    Sinkhorn chain is off the collapse's critical path.  Splitting the collapse
    further over S CTAs (last arrival rescales, integer counter) measured no
    gain at M = 4 for S = 2..32 and lost at M >= 16 (S=8: m16 18.94 vs 17.92).
    """
    m = tl.program_id(0)
    part = tl.program_id(1)
    jr = tl.arange(0, HC)
    ks = tl.arange(0, KSP2)
    kmask = ks < KS

    if part == 0:
        nr = tl.arange(0, NP2)
        nmask = nr < NOUT
        mixp = tl.load(mix_ptr + (ks[:, None] * M + m) * NOUT + nr[None, :],
                       mask=kmask[:, None] & nmask[None, :], other=0.0)
        sq = tl.sum(tl.load(sqr_ptr + ks * M + m, mask=kmask, other=0.0))
        rms = tl.math.rsqrt(sq / (HC * HIDDEN) + rms_eps)
        # One contiguous [KSP2, NP2] load of the split partials, reduced in a
        # fixed order.  Three separate strided gathers cost ~2.3 us.
        mix = tl.sum(mixp, axis=0) * rms
        mix_post = tl.sum(tl.where(nr[None, :] == (jr + HC)[:, None], mix[None, :], 0.0),
                          axis=1)
        jk = jr[:, None] * HC + jr[None, :] + 2 * HC
        mix_comb = tl.sum(
            tl.where(nr[None, None, :] == jk[:, :, None], mix[None, None, :], 0.0), axis=2)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        po = _sigmoid(mix_post * s1 + tl.load(base_ptr + jr + HC)) * post_mult
        tl.store(post_out_ptr + m * HC + jr, po)

        cm = mix_comb * s2 + tl.load(base_ptr + jk)
        if HC == 4:
            # 16 per-thread scalars: every row/column sum is 3 in-register adds
            # instead of a shuffle tree, and each norm is one reciprocal.
            # Spread the tile into 16 per-thread scalars through comb_out
            # itself (it is overwritten below): one store, a CTA barrier and 16
            # broadcast loads, instead of 16 CTA-wide reductions.
            cb = comb_out_ptr + m * HC * HC
            tl.store(cb + jr[:, None] * HC + jr[None, :], cm)
            tl.debug_barrier()
            c00 = tl.load(cb + 0)
            c01 = tl.load(cb + 1)
            c02 = tl.load(cb + 2)
            c03 = tl.load(cb + 3)
            c10 = tl.load(cb + 4)
            c11 = tl.load(cb + 5)
            c12 = tl.load(cb + 6)
            c13 = tl.load(cb + 7)
            c20 = tl.load(cb + 8)
            c21 = tl.load(cb + 9)
            c22 = tl.load(cb + 10)
            c23 = tl.load(cb + 11)
            c30 = tl.load(cb + 12)
            c31 = tl.load(cb + 13)
            c32 = tl.load(cb + 14)
            c33 = tl.load(cb + 15)
            # softmax over each row, + eps
            c00, c01, c02, c03 = _softmax4(c00, c01, c02, c03, hc_sink_eps)
            c10, c11, c12, c13 = _softmax4(c10, c11, c12, c13, hc_sink_eps)
            c20, c21, c22, c23 = _softmax4(c20, c21, c22, c23, hc_sink_eps)
            c30, c31, c32, c33 = _softmax4(c30, c31, c32, c33, hc_sink_eps)
            c00, c10, c20, c30 = _norm4(c00, c10, c20, c30, hc_sink_eps)
            c01, c11, c21, c31 = _norm4(c01, c11, c21, c31, hc_sink_eps)
            c02, c12, c22, c32 = _norm4(c02, c12, c22, c32, hc_sink_eps)
            c03, c13, c23, c33 = _norm4(c03, c13, c23, c33, hc_sink_eps)
            for _ in range(SINK - 1):
                c00, c01, c02, c03 = _norm4(c00, c01, c02, c03, hc_sink_eps)
                c10, c11, c12, c13 = _norm4(c10, c11, c12, c13, hc_sink_eps)
                c20, c21, c22, c23 = _norm4(c20, c21, c22, c23, hc_sink_eps)
                c30, c31, c32, c33 = _norm4(c30, c31, c32, c33, hc_sink_eps)
                c00, c10, c20, c30 = _norm4(c00, c10, c20, c30, hc_sink_eps)
                c01, c11, c21, c31 = _norm4(c01, c11, c21, c31, hc_sink_eps)
                c02, c12, c22, c32 = _norm4(c02, c12, c22, c32, hc_sink_eps)
                c03, c13, c23, c33 = _norm4(c03, c13, c23, c33, hc_sink_eps)
            # No warp may overwrite the staging copy before every warp has read it.
            tl.debug_barrier()
            tl.store(cb + 0, c00)
            tl.store(cb + 1, c01)
            tl.store(cb + 2, c02)
            tl.store(cb + 3, c03)
            tl.store(cb + 4, c10)
            tl.store(cb + 5, c11)
            tl.store(cb + 6, c12)
            tl.store(cb + 7, c13)
            tl.store(cb + 8, c20)
            tl.store(cb + 9, c21)
            tl.store(cb + 10, c22)
            tl.store(cb + 11, c23)
            tl.store(cb + 12, c30)
            tl.store(cb + 13, c31)
            tl.store(cb + 14, c32)
            tl.store(cb + 15, c33)
        else:
            # softmax over k, then `sinkhorn_repeat` alternating row/col norms.
            cm = tl.exp(cm - tl.max(cm, axis=1)[:, None])
            cm = cm / tl.sum(cm, axis=1)[:, None] + hc_sink_eps
            cm = cm / (tl.sum(cm, axis=0)[None, :] + hc_sink_eps)
            for _ in tl.static_range(SINK - 1):
                cm = cm / (tl.sum(cm, axis=1)[:, None] + hc_sink_eps)
                cm = cm / (tl.sum(cm, axis=0)[None, :] + hc_sink_eps)
            tl.store(comb_out_ptr + m * HC * HC + jr[:, None] * HC + jr[None, :], cm)
    else:
        # Collapse the streams and RMSNorm, whole hidden in registers, one pass.
        h = tl.arange(0, HIDDEN)
        mixq = tl.load(mix_ptr + (ks[:, None] * M + m) * NOUT + jr[None, :],
                       mask=kmask[:, None], other=0.0)
        sq4 = tl.sum(tl.load(sqr_ptr + ks * M + m, mask=kmask, other=0.0))
        rms4 = tl.math.rsqrt(sq4 / (HC * HIDDEN) + rms_eps)
        s0 = tl.load(scale_ptr + 0)
        pre = _sigmoid(tl.sum(mixq, axis=0) * rms4 * s0 + tl.load(base_ptr + jr)) + hc_pre_eps
        o = tl.zeros([HIDDEN], tl.float32)
        for j in tl.static_range(HC):
            pj = tl.sum(tl.where(jr == j, pre, 0.0))
            o += pj * tl.load(rc_ptr + m * HC * HIDDEN + j * HIDDEN + h).to(tl.float32)
        ob = o.to(tl.bfloat16).to(tl.float32)
        rn = tl.math.rsqrt(tl.sum(o * o) / HIDDEN + norm_eps)
        w = tl.load(nw_ptr + h).to(tl.float32)
        tl.store(li_ptr + m * HIDDEN + h, (ob * rn * w).to(tl.bfloat16))


# Kernel A tunables.  Module-level constants, not autotune: a graph capture must
# see one fixed config (see INTEGRATION.md).  Swept on GPU 2 over all six bench
# M (1, 2, 4, 8, 16, 32) with the 90-deep rotation under graph replay; the entry
# is the family score, higher is better:
#
#   HB    NB  warps  stages   score      notes
#   128    8      4       3   0.994   <- chosen
#   128    8      4       5   0.993   same schedule
#   128    8      4       2   0.954   m1 14.3 us instead of 11.3
#   128    8      4       4   0.954   m1 regresses the same way
#   128    4      4       2   0.935   6 n-groups, m16/m32 worse
#   128    8      8       2   0.855   8 warps loses everywhere but M=1
#    64    8      4       3   0.875   2x the CTAs, 2x the per-CTA fn rows
#   256    8      4       2   0.715   tile too large
#
# HB=128 with NB=8 is one [8, 4, 128] fp32 fn tile = 16 KB per CTA and
# 32 * 3 * M CTAs; the ranking is dominated by that tile size, not by CTA count.
HB_A = 128          # hidden elements per CTA; KSPLIT = hidden // HB_A
# Token blocking for kernel A.  BMS tokens of one CTA share one `fn` tile, so
# `fn` is streamed ceil(M/BMS) times per call instead of M times.  Measured
# over three repeats (us, incumbent in the same run):
#
#              m16          m32        family score
#   BMS=1   20.66 (0.87)  31.84 (0.81)     0.980
#   BMS=2   20.98 (0.87)  29.22 (0.89)     0.997
#   BMS=4   20.73 (0.87)  29.91 (0.86)     0.972   (one repeat)
#
# So it is worth 8.4 % at M=32 and nothing at M=16, where 128 CTAs per token
# already saturate and the 25 MB of re-reads are absorbed.  Hence BMS_FROM_M=32:
# blocking is applied only where it was measured to win.  At M <= 8 it must stay
# 1 - measurement showed the fully serial form at 0.79x for M=4.
BMS_A = 2
BMS_FROM_M = 32

# From this M the prenorm GEMM runs on the tensor cores instead of as a scalar
# FMA reduction; see `_post_prenorm_mma_kernel` for why the crossover exists.
MMA_FROM_M = 16
BK_A = 64           # K chunk of the mma path (prefill measured 64 > 128 > 256)
# Swept on GPU 2 over all six bench M, three repeats (family score; m16/m32 us):
#
#   HB_MMA  warps     m16            m32          family
#      32       4   20.29 (0.90)   22.69 (1.14)   1.044  <- chosen
#      32       2   20.99 (0.87)   22.87 (1.13)   1.037
#       64      4   22.37 (0.81)   23.40 (1.10)   1.020
#       32      8   22.70 (0.80)   32.90 (0.79)   0.963
#      128      4   35.50 (0.52)   34.98 (0.74)   0.891
#       16      4   20.90 (0.88)   26.54 (0.99)   0.993
#
# HB_MMA=128 starves the grid: with no n-split the mma path launches only
# hidden/HB_MMA * cdiv(M, 16) CTAs, which is 32 at M=16.  32 gives 128 * 2.
# Gluon kernel A (hc == 4, every M): swept: HB 16/64
# +0.35..1.4 us, 8 warps and k_width 2 worse, plain register fn loads +0.3 us,
# unswizzled cp.async buffers +1.1 us (bank conflicts).
HB_GLU = 32
BT_T = 8            # token block of the transposed M <= 8 kernel (the mma's n8)
WARPS_GLU = 4
HB_MMA = 32         # hidden per CTA on the mma path; KSPLIT = hidden // HB_MMA
WARPS_MMA = 4
BM_A = 16           # bf16/tf32 mma is m16n8k16: 16 rows is the minimum unit
PREC_A = "tf32x3"   # the only fp32-accurate tl.dot mode on sm_80/Triton 3.7.1
NB_A = 8            # prenorm outputs per CTA; NOUT // NB_A CTAs in the n dim
WARPS_A = 4
STAGES_A = 3


def _config(M, hidden, hc):
    """(KSPLIT, HB, NB, warps_a, warps_b) - a pure function of the shape, so a
    capture always sees one config.  Kernel A's CTA count is
    M * (hidden // HB) * (NOUT // NB).
    """
    hb = min(HB_A, hidden)
    ks = hidden // hb
    # Token blocking: BMS tokens of one CTA share a single `fn` tile, so `fn`
    # is streamed ceil(M/BMS) times per call instead of M.  At M <= 8 the grid
    # is what is scarce and BMS stays 1 (Measurement showed the fully serial form
    # losing 0.79x at M=4); from M=16 the grid is already 1536+ CTAs and the
    # 25-50 MB of `fn` re-reads per call is what costs.  BMS_A is the swept cap.
    bms = 1
    if M >= BMS_FROM_M:
        bms = BMS_A
    return ks, hb, NB_A, bms, WARPS_A, 4


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
    ks, hb, nb, bms, wa, wb = _config(M, hidden, hc)

    rc = torch.empty(M, hc, hidden, dtype=torch.bfloat16, device=dev)
    ksa = max(ks, hidden // min(HB_MMA, hidden), hidden // HB_GLU)
    mix = torch.empty(ksa, M, nout, dtype=torch.float32, device=dev)
    sqr = torch.empty(ksa, M, dtype=torch.float32, device=dev)
    post_out = torch.empty(M, hc, dtype=torch.float32, device=dev)
    comb_out = torch.empty(M, hc * hc, dtype=torch.float32, device=dev)
    li = torch.empty(M, hidden, dtype=torch.bfloat16, device=dev)

    if hc == 4 and hidden % HB_GLU == 0 and M <= BT_T:
        ksb = hidden // HB_GLU
        _glu_post_prenorm_tb_kernel[(triton.cdiv(M, BT_T), ksb)](
            xf, res, pf, cf, fnf, rc, mix, sqr, M,
            HIDDEN=hidden, NOUT=nout, HB=HB_GLU, BT=BT_T,
            NP2=triton.next_power_of_2(nout), num_warps=4)
    elif hc == 4 and hidden % HB_GLU == 0:
        ksb = hidden // HB_GLU
        _glu_post_prenorm_kernel[(triton.cdiv(M, BM_A), ksb)](
            xf, res, pf, cf, fnf, rc, mix, sqr, M,
            HIDDEN=hidden, NOUT=nout, HB=HB_GLU, BM=BM_A,
            NP2=triton.next_power_of_2(nout), NW=WARPS_GLU, num_warps=WARPS_GLU)
    elif M >= MMA_FROM_M:
        ksb = hidden // min(HB_MMA, hidden)
        _post_prenorm_mma_kernel[(triton.cdiv(M, BM_A), ksb)](
            xf, res, pf, cf, fnf, rc, mix, sqr, M,
            HIDDEN=hidden, HC=hc, NOUT=nout, HB=min(HB_MMA, hidden),
            BK=min(BK_A, HB_MMA),
            BM=BM_A, NP2=triton.next_power_of_2(nout), PREC=PREC_A,
            num_warps=WARPS_MMA, num_stages=STAGES_A)
    else:
        ksb = ks
        _post_prenorm_kernel[(triton.cdiv(M, bms), ks, nout // nb)](
            xf, res, pf, cf, fnf, rc, mix, sqr, M,
            HIDDEN=hidden, HC=hc, NOUT=nout, HB=hb, NB=nb, BMS=bms,
            num_warps=wa, num_stages=STAGES_A)
    if hc == 4 and hidden % 1024 == 0:
        _glu_finish_kernel[(M, 2)](
            mix, sqr, hc_scale, hc_base, rc, nw, post_out, comb_out, li, M,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, norm_eps,
            HIDDEN=hidden, NOUT=nout, KS=ksb, KSP2=max(triton.next_power_of_2(ksb), 32),
            SINK=sinkhorn_repeat, num_warps=4)
    else:
        _pre_finish_kernel[(M, 2)](
            mix, sqr, hc_scale, hc_base, rc, nw, post_out, comb_out, li, M,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, norm_eps,
            HIDDEN=hidden, HC=hc, NOUT=nout, KS=ksb,
            KSP2=triton.next_power_of_2(ksb),
            NP2=triton.next_power_of_2(nout), SINK=sinkhorn_repeat,
            num_warps=wb, num_stages=1)

    return (rc.view(*outer, hc, hidden),
            post_out.view(*outer, hc, 1),
            comb_out.view(*outer, hc, hc),
            li.view(*outer, hidden))


LAUNCHES = 2


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
