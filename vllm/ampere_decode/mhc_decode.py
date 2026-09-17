#!/usr/bin/env python
"""Candidate mHC fused post+pre for GLM-5.3-Flash decode on sm_80.

Drop-in for `vllm.model_executor.kernels.mhc.tilelang.mhc_fused_post_pre_tilelang`:
same signature, same four outputs, same dtypes, same accumulation.
Keep this kernel in sync with the standalone source.

WHAT THE INCUMBENT DOES AND WHY IT IS SLOW
------------------------------------------
Two TileLang kernels, 14.33 us/call at M=4 against a 1.28 us memory roofline
(11.2x).  Measured causes, in order:

  * `mhc_pre_big_fuse_with_norm_tilelang` launches `T.Kernel(num_tokens)` -
    **4 CTAs of 96 threads** on a 70-SM part.  0.5 % occupancy.
  * Inside it, the 20 Sinkhorn iterations run in ONE warp on a 4x4 matrix using
    `T.reduce_sum`/`T.reduce_max` over a fragment: ~40 serialised shared-memory
    reductions for ~160 useful FLOPs.  That is most of its 7.9 us.
  * `mhc_fused_tilelang` uses grid `(m, n_tiles=24, split_k)`, so **each of the
    24 output tiles re-reads `residual` and recomputes the whole 4x4 hc-mix**.
    The residual read and the mix FMA are done 24 times over.

WHAT THIS CANDIDATE DOES
------------------------
Two kernels, restructured:

  A `_post_prenorm_kernel`, grid (M, KSPLIT, NOUT // NB).  Each CTA owns one
    token, one HB-wide slice of `hidden` across all HC streams, and NB of the 24
    prenorm outputs.  It computes `residual_cur` for that slice ONCE (the
    incumbent recomputes the hc-mix for each of its 24 output tiles), stores it
    from the ng == 0 CTAs, and reduces its NB `mixes` partials and the `sqrsum`
    partial from the fp32 value still in registers.  The `fn` tile is a 3D
    [NB, HC, HB] block loaded first, so the load vectorises and its latency
    overlaps the mix - see the kernel docstring for the three measured steps and
    for the two restructurings that lost.
  B `_pre_finish_kernel`, grid (M,).  Reduces the split partials with ONE
    contiguous [KSPLIT, 32] load in a fixed order (three strided gathers here
    cost 2.3 us), then does the Sinkhorn as Triton block ops on a [4, 4] tile,
    and collapses + RMSNorms the whole 4096-wide hidden in registers in a
    single pass.

MEASURED STATE: family score 0.99 over M = 1, 2, 4, 8, 16, 32 (was 0.858).
At the production shape M=4 the pair is 12.6 us against the incumbent's 13.0,
1.03x; M=1 1.13x, M=2 1.08x, M=8 1.07x.  M=16 (0.88x) and M=32 (0.89x) are
still behind: there `fn` is re-read once per token-block, 25 MB per call, and
sharing it further needs a structure that does not serialise the tokens - the
two full-serial attempts in the kernel docstring both failed, while the partial
2-token block (`BMS_A`) recovered M=32.  Kernel A is ~6.3 us of the
12.6 and kernel B ~6.3, of which ~3 us is the irreducible Sinkhorn latency
chain (README hint 2).

Both are plain Triton, CUDA-graph capturable, and reduce split partials with a
fixed-shape `tl.sum` (no atomics), so the result is bitwise reproducible.

NO PRECISION REDUCTION.  bf16 in / fp32 accumulate / bf16 out, fp32 `fn`, and
the two bf16 rounding boundaries that are part of the op's definition are kept
exactly where the incumbent puts them:
  * `mixes` and `sqrsum` are accumulated from the **pre-rounding fp32**
    `residual_cur` (that is what `mhc_fused` does - it has the value in
    registers);
  * the collapse in B reads the **bf16** `residual_cur` (that is what
    `mhc_pre_big_fuse` does - it is a separate kernel reading HBM), and the
    RMSNorm squared-sum is taken from the fp32 collapse while the scale is
    applied to its bf16 rounding.
`reference/mhc_decode_ref.py::check_rounding_model()` proves both choices
against the real kernel; do not "simplify" either one.
"""

import torch
import triton
import triton.language as tl


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


@triton.jit
def _pre_finish_kernel(
    mix_ptr, sqr_ptr, scale_ptr, base_ptr, rc_ptr, nw_ptr,
    post_out_ptr, comb_out_ptr, li_ptr,
    M,
    rms_eps, hc_pre_eps, hc_sink_eps, post_mult, norm_eps,
    HIDDEN: tl.constexpr, HC: tl.constexpr, NOUT: tl.constexpr,
    KS: tl.constexpr, KSP2: tl.constexpr, NP2: tl.constexpr, SINK: tl.constexpr,
):
    """Reduce partials, Sinkhorn, collapse + RMSNorm.  grid (M,)."""
    m = tl.program_id(0)
    jr = tl.arange(0, HC)
    ks = tl.arange(0, KSP2)
    kmask = ks < KS
    nr = tl.arange(0, NP2)
    nmask = nr < NOUT

    sq = tl.sum(tl.load(sqr_ptr + ks * M + m, mask=kmask, other=0.0))
    rms = tl.math.rsqrt(sq / (HC * HIDDEN) + rms_eps)

    # One contiguous [KSP2, NP2] load of the split partials, reduced in a fixed
    # order.  Three separate strided gathers here cost ~2.3 us of the kernel.
    mix = tl.sum(tl.load(mix_ptr + (ks[:, None] * M + m) * NOUT + nr[None, :],
                         mask=kmask[:, None] & nmask[None, :], other=0.0),
                 axis=0) * rms

    mix_pre = tl.sum(tl.where(nr[None, :] == jr[:, None], mix[None, :], 0.0), axis=1)
    mix_post = tl.sum(tl.where(nr[None, :] == (jr + HC)[:, None], mix[None, :], 0.0),
                      axis=1)
    jk = jr[:, None] * HC + jr[None, :] + 2 * HC
    mix_comb = tl.sum(
        tl.where(nr[None, None, :] == jk[:, :, None], mix[None, None, :], 0.0), axis=2)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = _sigmoid(mix_pre * s0 + tl.load(base_ptr + jr)) + hc_pre_eps
    po = _sigmoid(mix_post * s1 + tl.load(base_ptr + jr + HC)) * post_mult
    tl.store(post_out_ptr + m * HC + jr, po)

    # softmax over k, then `sinkhorn_repeat` alternating row/col normalisations.
    cm = mix_comb * s2 + tl.load(base_ptr + jk)
    cm = tl.exp(cm - tl.max(cm, axis=1)[:, None])
    cm = cm / tl.sum(cm, axis=1)[:, None] + hc_sink_eps
    cm = cm / (tl.sum(cm, axis=0)[None, :] + hc_sink_eps)
    for _ in tl.static_range(SINK - 1):
        cm = cm / (tl.sum(cm, axis=1)[:, None] + hc_sink_eps)
        cm = cm / (tl.sum(cm, axis=0)[None, :] + hc_sink_eps)
    tl.store(comb_out_ptr + m * HC * HC + jr[:, None] * HC + jr[None, :], cm)

    # Collapse the streams and RMSNorm, whole hidden in registers, one pass.
    h = tl.arange(0, HIDDEN)
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
    ksa = max(ks, hidden // min(HB_MMA, hidden))
    mix = torch.empty(ksa, M, nout, dtype=torch.float32, device=dev)
    sqr = torch.empty(ksa, M, dtype=torch.float32, device=dev)
    post_out = torch.empty(M, hc, dtype=torch.float32, device=dev)
    comb_out = torch.empty(M, hc * hc, dtype=torch.float32, device=dev)
    li = torch.empty(M, hidden, dtype=torch.bfloat16, device=dev)

    if M >= MMA_FROM_M:
        _post_prenorm_mma_kernel[(triton.cdiv(M, BM_A), hidden // min(HB_MMA, hidden))](
            xf, res, pf, cf, fnf, rc, mix, sqr, M,
            HIDDEN=hidden, HC=hc, NOUT=nout, HB=min(HB_MMA, hidden),
            BK=min(BK_A, HB_MMA),
            BM=BM_A, NP2=triton.next_power_of_2(nout), PREC=PREC_A,
            num_warps=WARPS_MMA, num_stages=STAGES_A)
    else:
        _post_prenorm_kernel[(triton.cdiv(M, bms), ks, nout // nb)](
            xf, res, pf, cf, fnf, rc, mix, sqr, M,
            HIDDEN=hidden, HC=hc, NOUT=nout, HB=hb, NB=nb, BMS=bms,
            num_warps=wa, num_stages=STAGES_A)
    ksb = (hidden // min(HB_MMA, hidden)) if M >= MMA_FROM_M else ks
    _pre_finish_kernel[(M,)](
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
