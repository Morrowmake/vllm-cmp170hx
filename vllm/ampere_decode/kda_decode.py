#!/usr/bin/env python
"""Fused KDA recurrent decode step -- ONE kernel for all three.

    kda_decode(qkv, conv_state, conv_weight, conv_bias, g, beta, gate2,
               norm_weight, rec_state, conv_state_indices, ssm_state_indices,
               num_accepted_tokens, cu_seqlens, max_query_len, a_log, g_bias,
               lower_bound=-5.0, eps=1e-5, scale=None, out=None)
        -> out [1, M, H, D] bf16

replaces, bit-for-bit in meaning,

    causal_conv1d_update(...)  ->  fused_recurrent_kda(...)  ->  FusedRMSNormGated(...)

and advances `conv_state` and `rec_state` in place exactly as they do.
`reference/kda_decode_ref.py` is the contract; read it first.

WHAT THE INCUMBENT DOES (measured on GPU 2, not guessed -- see README.md):

    _causal_conv1d_update_kernel        grid (nseq, 24)        256 thr, BLOCK_N=256
    fused_recurrent_gated_delta_rule    grid (1, 16, nseq*16)   32 thr,
                                        BK=128 BV=8 num_warps=1 num_stages=3
    layer_norm_gated_fwd_kernel         grid (ceil(M*H/16),)   256 thr, BT=16

Three launches x ~1.25 us of fixed in-graph cost, and the recurrent kernel's 256
single-warp CTAs each redo the same 128-wide q/k L2 norm and the same 128-wide
gate for their 8-row slice of the value dimension -- 16x redundant.  At M=4 the
three cost 4.08 + 11.96 + 2.38 us against a 1.65 us state-traffic roofline.

WHAT THIS DOES DIFFERENTLY

  * ONE launch instead of three: 2 x 1.25 us of fixed cost per layer removed
    (0.085 ms/step on its own), plus the qkv (48 KB) and core (16 KB) round
    trips through HBM that the kernel boundaries forced.
  * The value dimension is still split across CTAs (`BV` rows each, `NV = D/BV`
    CTAs per head), so occupancy is kept -- but the gated RMS norm reduces over
    ALL 128 value rows of a head, so a V-split CTA cannot finish the head on its
    own.  The fusion is closed by a counter: each CTA writes its BV-row slice of
    the (unnormalised, bf16) recurrence output -- which is exactly what the
    incumbent materialises between kernels 2 and 3 -- then bumps one int32 per
    (sequence, head) with acq_rel/gpu semantics; the CTA that observes NV-1
    normalises the whole head and resets the counter for the next launch.  No
    CTA ever waits, so there is no grid-size constraint and no deadlock, and the
    combine is a fixed-order reduction over a [128] tile, so the result does not
    depend on which CTA got there last.
  * EVERY TOKEN'S conv, silu, q/k L2 norm, gate and beta are computed UP FRONT,
    as one [BT, D] tile, before the recurrence starts.  None of that work
    depends on the recurrent state, and computing it per token put T separate
    cross-lane reduction trees on the dependent chain between state updates.
    Batched, the two L2 norms are ONE `tl.sum(.., 1)` over the token tile, and
    so is the gated RMS norm in the epilogue.  Measured at M=4: 18.84 -> 14.64
    us, and an ablation that simply deleted the per-token L2 norms scored 17.29,
    so the batched form is FASTER than not doing the reductions at all in the
    old schedule.
  * ONE WARP per CTA.  That is what makes the batching pay: with one warp every
    reduction - the norms, the [BV, D] -> [BV] sums, and the token-tile row
    selects - is a register shuffle instead of a shared-memory round trip with
    two barriers.  The same batched kernel at 2 warps is 23.25 us at M=4
    against 14.64 at one.  The incumbent's own recurrent kernel launches
    `num_warps=1` for this reason.
  * `BV` then trades redundancy against occupancy: the per-head work every
    V-slice repeats against the CTA count.  `_select` targets ~512 CTAs, which
    is the best column at every M in the measured sweep above it.
  * The conv window is read straight from the conv state and `qkv` per tap, so
    the conv output never reaches memory.
  * Redundant state stores are skipped.  The recurrence stores the state after
    every one of the T draft tokens, into `ssm_state_indices[n, t]`.  Whenever
    two consecutive tokens name the same slot -- which is every token at
    concurrency 1 -- all but the last store are dead.  Skipping them removes
    (T-1) x 64 KB of store traffic per head per call and changes nothing
    observable: the surviving store writes the newer value to the same address.
  * `qkv` is read, not overwritten (the incumbent's conv writes the conv output
    back over its input), saving 48 KB of stores per call.

NO PRECISION REDUCTION.  Same dtypes, same accumulation, same order:

  * the recurrent state is fp32 in memory and fp32 in registers;
  * the conv accumulates in fp32 from the fp32 weights in tap order 0,1,2,3,
    exactly as `_causal_conv1d_update_kernel` does;
  * the two bf16 roundings (conv output, recurrence output) are reproduced
    because the incumbent materialises both in bf16 and the next kernel reads
    them back -- dropping them would compute a different function (see the
    reference docstring).  They are the only roundings.
  * no `tl.dot` anywhere, so no TF32 mantissa loss is even possible;
  * every float reduction is a `tl.sum` over a fixed block, never an atomic
    (the only atomic is the int32 arrival counter), so the result is bitwise
    identical run to run.  Batching the reductions over the token tile changes
    which fixed order they run in, not that there is one: `validate.py` reports
    the state and output error unchanged from the per-token form on 84 checks.

MEASURED: 1.098x over M = 1, 2, 4, 8, 16, 32 (1.12 / 1.12 / 1.18 / 1.06 / 1.09
/ 1.02), against candidate 1's 0.967 and its 0.92 at the production shape.  At
M=4 the family is 14.64 us against the incumbent's 17.31, so the 34 KDA layers
go 0.583 -> 0.498 ms/step.

CUDA-GRAPH SAFE: no autotune on the timed path, no host sync, no `.item()`.  The
arrival counter is the one piece of state that outlives a call; `warmup()`
allocates it and it self-resets to zero at the end of every launch, so replaying
a captured graph any number of times is correct.  Call `warmup()` before capture
so Triton compiles outside it.

sm_80 NOTES: BF16 MMA is m16n8k16 so `tl.dot` would need BLOCK_M >= 16, and the
whole op is a matrix-vector chain -- there is nothing to feed the tensor cores
without padding 1 to 16, so this kernel is deliberately all CUDA-core FMA.  No
fp8e4nv, no TMA, no wgmma, no warp specialisation is used or available.
"""

import torch

from vllm.triton_utils import tl, triton  # noqa: F401  (Triton 3.7.1)

# Tunables.  MEASURED on GPU 2 with the L2-defeating rotation.  `num_warps` is
# ONE, always, and that is the whole story of this kernel's schedule: every
# token of the recurrence ends in cross-lane reductions (the two 128-wide q/k
# L2 norms, and the [BV, D] -> [BV] sums for `b_v` and `b_o`), they sit on the
# dependent chain, and with one warp they are register shuffles instead of a
# shared-memory round trip with two barriers.  This is why the incumbent
# launches `num_warps=1` too.  Candidate 1 used 4 warps and paid 1.55 us at
# M=4 for the L2 norms alone (measured by ablation: replacing them with a
# constant took 18.84 -> 17.29 us).
#
# Candidate microseconds, this kernel, num_warps=1, by BV x M:
#
#     M      BV=4    BV=8   BV=16   BV=32      incumbent
#     1     10.73    9.97   10.52   10.21          11.21
#     2     12.27   11.57   12.25   14.04          12.98
#     4     15.59   14.63   17.27   25.39          17.31
#     8     25.84   18.74   19.05   26.41          19.81
#    16     41.79   29.04   24.25   29.46          26.40
#    32     74.39   49.46   43.89   41.96          42.71
#
# and with 2 or 4 warps the same shapes lose 20-60 % (M=4, BV=8: 14.63 at one
# warp, 23.25 at two).  The BV ranking is pure CTA count: the best column is
# always the one that puts nseq * H * (D // BV) near 512 CTAs, i.e. ~7 per SM,
# which is what `_select` computes.  Below that the machine is empty; above it
# the per-head work that every V-slice repeats is what is being paid for.
TARGET_CTAS = 512          # nseq * H * (D // BV) aimed at this
BV_MIN = 8                 # BV=4 spills the token tiles at nseq >= 2
BV_MAX = 32
NUM_WARPS = 1
NUM_STAGES = 1             # the token loop is serial; pipelining it measured flat

_CTR_SLOTS = 8192  # arrival counters, one int32 per (sequence, head)
_SMS: dict[int, int] = {}


def num_sms(device=0):
    """Read at runtime -- never hard-code 70; the kernels target several parts."""
    idx = device if isinstance(device, int) else (device.index or 0)
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def _select(nseq, H, D, device):
    """(BV, num_warps) for this grid.  See the table above.

    A pure function of the shape, taken on the host before the launch, so the
    launch config is static and a captured graph always replays the same one.
    It is NOT an autotune on the timed path.
    """
    nv = max(1, min(D // BV_MIN, TARGET_CTAS // max(1, nseq * H)))
    bv = max(BV_MIN, min(BV_MAX, D // max(1, nv)))
    while D % bv:
        bv //= 2
    return bv, NUM_WARPS


@triton.jit
def _conv_silu_tile(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                    conv_w, conv_b, chan, off, o_t, m_t,
                    HAS_BIAS: tl.constexpr, CONV_K: tl.constexpr):
    """[BT, W] of conv + silu + bf16 outputs, one row per token.

    For token t and tap j the source row is the conv state at `off + t + j`
    while `t + j < CONV_K - 1` and the new input `x[t + j - (CONV_K - 1)]`
    after that - the same window the incumbent conv kernel slides, with the
    same fp32 tap order 0,1,2,3 and the same bf16 rounding of the result (the
    incumbent writes the conv output to the bf16 qkv buffer and the recurrent
    kernel reads it back, so that rounding is part of the function).
    """
    p_c = csb + chan * stride_cs_chan
    for j in tl.static_range(CONV_K):
        wj = tl.load(conv_w + chan * CONV_K + j).to(tl.float32)
        from_s = (o_t + j) < (CONV_K - 1)
        sv = tl.load(p_c[None, :] + (off + o_t + j)[:, None] * stride_cs_tok,
                     mask=from_s[:, None] & m_t[:, None], other=0.0)
        xv = tl.load(x + (bos + o_t + j - (CONV_K - 1))[:, None] * stride_x_t
                     + chan[None, :],
                     mask=(~from_s)[:, None] & m_t[:, None], other=0.0)
        src = tl.where(from_s[:, None], sv, xv).to(tl.float32)
        if j == 0:
            if HAS_BIAS:
                a = tl.load(conv_b + chan).to(tl.float32)[None, :] + wj[None, :] * src
            else:
                a = wj[None, :] * src
        else:
            a += wj[None, :] * src
    a = a / (1.0 + tl.exp(-a))
    return a.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _kda_decode_kernel(
    x,                       # [M, CONV_DIM] bf16, the conv input (q|k|v)
    conv_state,              # [nslot, CONV_DIM, SLEN] bf16 (the "DS" view)
    conv_w,                  # [CONV_DIM, CONV_K] fp32
    conv_b,                  # [CONV_DIM] fp32 or None
    g,                       # [1, M, H, D] raw gate logits
    beta,                    # [1, M, H] raw beta logits
    gate2,                   # [M, H, D] output-gate logits
    norm_w,                  # [D]
    out,                     # [1, M, H, D] bf16
    rec,                     # [nslot, H, D, D] fp32
    cu_seqlens,              # [nseq + 1] int32
    conv_idx,                # [nseq] int32
    ssm_idx,                 # [nseq, T] int32
    num_acc,                 # [nseq] int32
    a_log,                   # [H] fp32
    g_bias,                  # [PROJ] fp32
    ctr,                     # [nseq * H] int32, arrival counters (self-resetting)
    scale,
    eps,
    stride_x_t,
    stride_g_t,
    stride_beta_t,
    stride_g2_t,
    stride_out_t,
    stride_cs_slot,
    stride_cs_chan,
    stride_cs_tok,
    stride_rec_slot,
    stride_cidx,
    stride_sidx_seq,
    stride_sidx_tok,
    H: tl.constexpr,
    D: tl.constexpr,
    CONV_K: tl.constexpr,
    PROJ: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    L2_EPS: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    BS: tl.constexpr,        # next_pow2 of the conv state length
    BT: tl.constexpr,        # next_pow2 of max_query_len: the token tile
):
    # The rolling conv window below is written out for a width-4 kernel, which
    # is what `linear_conv_kernel_dim` is for every GLM-5.3-Flash KDA layer.
    tl.static_assert(CONV_K == 4, "conv window is specialised to width 4")

    pid = tl.program_id(0)
    i_v = pid % NV           # V-slices of one head are adjacent CTAs
    i_nh = pid // NV
    i_n = i_nh // H
    i_h = i_nh % H

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return

    # Initial state slot: the one the last ACCEPTED draft token left behind.
    acc = tl.load(num_acc + i_n).to(tl.int64)
    s_init = tl.load(ssm_idx + i_n * stride_sidx_seq
                     + (acc - 1) * stride_sidx_tok).to(tl.int64)
    if s_init <= 0:        # NULL_BLOCK_ID: padded batch entry, nothing to do
        return
    c_slot = tl.load(conv_idx + i_n * stride_cidx).to(tl.int64)
    if c_slot <= 0:
        return
    off = acc - 1          # sliding-window offset into the conv state

    o_d = tl.arange(0, D)                     # the key dimension, always full
    o_v = i_v * BV + tl.arange(0, BV)         # this CTA's value rows
    o_t = tl.arange(0, BT)                    # every token of this sequence
    m_t = o_t < T
    c_q = i_h * D + o_d
    c_k = PROJ + i_h * D + o_d
    c_v = 2 * PROJ + i_h * D + o_v
    csb = conv_state + c_slot * stride_cs_slot

    # ---- phase 1: EVERY token's conv, silu, L2 norm, gate and beta, up front
    # None of it depends on the recurrence, and doing all BT tokens as one tile
    # is what makes their reductions batch: `tl.sum(.., 1)` over a [BT, D] tile
    # is ONE cross-warp reduction tree where the per-token form was T of them,
    # each sitting on the dependent chain between two state updates.  Measured
    # by ablation at M=4: replacing just the two per-token 128-wide L2 norms
    # with a constant took the kernel from 18.84 to 17.29 us, so those eight
    # reduction trees were 1.55 us - most of the gap to the incumbent.
    qt = _conv_silu_tile(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                         conv_w, conv_b, c_q, off, o_t, m_t, HAS_BIAS, CONV_K)
    kt = _conv_silu_tile(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                         conv_w, conv_b, c_k, off, o_t, m_t, HAS_BIAS, CONV_K)
    vt = _conv_silu_tile(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                         conv_w, conv_b, c_v, off, o_t, m_t, HAS_BIAS, CONV_K)

    # Same op order as the incumbent: normalise, then scale.
    qt = (qt / tl.sqrt(tl.sum(qt * qt, 1) + L2_EPS)[:, None]) * scale
    kt = kt / tl.sqrt(tl.sum(kt * kt, 1) + L2_EPS)[:, None]

    b_a = tl.exp(tl.load(a_log + i_h).to(tl.float32))
    gb = tl.load(g_bias + i_h * D + o_d).to(tl.float32)
    gt = tl.load(g + (bos + o_t)[:, None] * stride_g_t + c_q[None, :],
                 mask=m_t[:, None], other=0.0).to(tl.float32)
    gt = tl.exp(LOWER_BOUND / (1.0 + tl.exp(-(b_a * (gt + gb[None, :])))))
    bt = tl.sigmoid(tl.load(beta + (bos + o_t) * stride_beta_t + i_h,
                            mask=m_t, other=0.0).to(tl.float32))

    # ---- phase 2: the recurrence.  This CTA's [BV, D] slice of the head tile.
    p_h = rec + s_init * stride_rec_slot + i_h * D * D
    b_h = tl.load(p_h + o_v[:, None] * D + o_d[None, :]).to(tl.float32)

    for t in tl.static_range(BT):
        if t < T:
            # Pick token t out of the phase-1 tiles.  A [BT, *] reduction over
            # BT <= 4 rows, not a cross-warp tree over D.
            sel = o_t == t
            b_q = tl.sum(tl.where(sel[:, None], qt, 0.0), 0)
            b_k = tl.sum(tl.where(sel[:, None], kt, 0.0), 0)
            b_e = tl.sum(tl.where(sel[:, None], gt, 0.0), 0)
            b_v = tl.sum(tl.where(sel[:, None], vt, 0.0), 0)
            b_beta = tl.sum(tl.where(sel, bt, 0.0))

            b_h *= b_e[None, :]
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_v *= b_beta
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)

            # Store the state, unless the next token overwrites the same slot.
            s_t = tl.load(ssm_idx + i_n * stride_sidx_seq
                          + t * stride_sidx_tok).to(tl.int64)
            t_n = tl.where(t + 1 < T, t + 1, t)   # clamped so the load is in range
            s_n = tl.load(ssm_idx + i_n * stride_sidx_seq
                          + t_n * stride_sidx_tok).to(tl.int64)
            live = (s_t > 0) & ((t_n == t) | (s_n != s_t))
            if live:
                p_ht = rec + s_t * stride_rec_slot + i_h * D * D
                tl.store(p_ht + o_v[:, None] * D + o_d[None, :], b_h)

            # Stage this slice of the (unnormalised) recurrence output in `out`,
            # in bf16 -- the same value the incumbent hands to the norm kernel.
            # `.cg` keeps it out of L1 so the epilogue CTA sees it.
            p_o = out + (bos + t) * stride_out_t + i_h * D + o_v
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), cache_modifier=".cg")

    # --- arrival: the last CTA of this (sequence, head) finishes the head ----
    tl.debug_barrier()          # this CTA's stores complete before its arrival
    do_epi = True
    if NV > 1:
        old = tl.atomic_add(ctr + i_nh, 1, sem="acq_rel", scope="gpu")
        do_epi = old == NV - 1

    if do_epi:
        if NV > 1:
            # Reset for the next launch (a captured graph replays this kernel).
            tl.atomic_xchg(ctr + i_nh, 0, sem="release", scope="gpu")

        # --- conv state writeback -------------------------------------------
        # Deferred to here: with NV > 1 the other V-slices are still reading the
        # q and k window when this CTA is halfway through its token loop, and
        # rows 0..CONV_K-3 of the new state overwrite rows this CTA does not
        # own.  The arrival counter is what makes it safe: every CTA of this
        # head has finished its conv-state loads before this point.
        #
        # New state = [window[off+1 .. off+CONV_K-2], x[0 .. T-1]].
        o_s = tl.arange(0, BS)
        keep: tl.constexpr = CONV_K - 2
        m_s = o_s < keep + T
        is_keep = o_s < keep
        for j in tl.static_range(3):
            chan = j * PROJ + i_h * D + o_d
            p_c = csb + chan * stride_cs_chan
            sv = tl.load(p_c[None, :] + (off + 1 + o_s)[:, None] * stride_cs_tok,
                         mask=is_keep[:, None] & m_s[:, None], other=0.0)
            xv = tl.load(x + (bos + o_s - keep)[:, None] * stride_x_t
                         + chan[None, :],
                         mask=(~is_keep)[:, None] & m_s[:, None], other=0.0)
            new = tl.where(is_keep[:, None], sv, xv)
            # Same load/store aliasing hazard as in the incumbent conv kernel:
            # without this the window read back its own rewritten rows whenever
            # off < CONV_K-1, and every head came out wrong at acc == 1.
            tl.debug_barrier()
            tl.store(p_c[None, :] + o_s[:, None] * stride_cs_tok, new,
                     mask=m_s[:, None])

        # --- 3. gated RMS norm over the whole head, all tokens at once -------
        # Batched for the same reason as phase 1: one reduction tree over a
        # [BT, D] tile instead of T of them.
        nw = tl.load(norm_w + o_d).to(tl.float32)
        p_ot = out + (bos + o_t)[:, None] * stride_out_t + c_q[None, :]
        b_o = tl.load(p_ot, mask=m_t[:, None], other=0.0,
                      cache_modifier=".cg").to(tl.float32)
        b_rstd = 1.0 / tl.sqrt(tl.sum(b_o * b_o, 1) / D + eps)
        b_y = b_o * b_rstd[:, None] * nw[None, :]
        b_g2 = tl.load(gate2 + (bos + o_t)[:, None] * stride_g2_t + c_q[None, :],
                       mask=m_t[:, None], other=0.0).to(tl.float32)
        b_y = b_y * tl.sigmoid(b_g2)
        tl.store(p_ot, b_y.to(p_ot.dtype.element_ty), mask=m_t[:, None])


_CTR: dict[tuple, torch.Tensor] = {}


def _counter(device):
    """The arrival counters.  Allocated once, never reallocated, always zero
    between launches (the last CTA of each head resets its own)."""
    key = (device.type, device.index or 0)
    c = _CTR.get(key)
    if c is None:
        c = torch.zeros(_CTR_SLOTS, device=device, dtype=torch.int32)
        _CTR[key] = c
    return c


def kda_decode(qkv, conv_state, conv_weight, conv_bias, g, beta, gate2,
               norm_weight, rec_state, conv_state_indices, ssm_state_indices,
               num_accepted_tokens, cu_seqlens, max_query_len, a_log, g_bias,
               lower_bound=-5.0, eps=1e-5, l2_eps=1e-6, scale=None, out=None):
    """One fused KDA decode step.  See the module docstring for the contract.

    `qkv` is consumed, not overwritten (the incumbent conv overwrites it in
    place; this kernel does not need to, and callers must not rely on it).
    `conv_state` and `rec_state` are advanced in place.  `out` is used as the
    staging buffer for the unnormalised recurrence output before it is
    normalised in place, so its contents are transient during the launch.
    """
    M, _ = qkv.shape
    H = a_log.shape[0]
    D = rec_state.shape[-1]
    CONV_K = conv_weight.shape[1]
    PROJ = H * D
    if scale is None:
        scale = D ** -0.5
    if out is None:
        out = torch.empty(1, M, H, D, dtype=qkv.dtype, device=qkv.device)
    assert rec_state.dtype == torch.float32, "the recurrent state stays fp32"

    nseq = cu_seqlens.numel() - 1
    assert nseq * H <= _CTR_SLOTS, "arrival counter buffer too small"
    bv, num_warps = _select(nseq, H, D, qkv.device)
    assert D % bv == 0, f"BV={bv} must divide the head dim {D}"
    ssm = ssm_state_indices
    if ssm.ndim == 1:
        st_seq, st_tok = ssm.stride(0), 1
    else:
        st_seq, st_tok = ssm.stride()
    nv = D // bv

    _kda_decode_kernel[(nv * nseq * H,)](
        qkv, conv_state, conv_weight, conv_bias, g, beta, gate2, norm_weight,
        out, rec_state, cu_seqlens, conv_state_indices, ssm, num_accepted_tokens,
        a_log, g_bias, _counter(qkv.device),
        scale, eps,
        qkv.stride(0),
        g.stride(1),
        beta.stride(1),
        gate2.stride(0),
        out.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        rec_state.stride(0),
        conv_state_indices.stride(0),
        st_seq, st_tok,
        H=H, D=D, CONV_K=CONV_K, PROJ=PROJ,
        LOWER_BOUND=lower_bound, HAS_BIAS=conv_bias is not None, L2_EPS=l2_eps,
        BV=bv, NV=nv,
        BS=triton.next_power_of_2(CONV_K - 2 + max_query_len),
        BT=triton.next_power_of_2(max_query_len),
        num_warps=num_warps, num_stages=NUM_STAGES,
    )
    return out


_WARMED = set()


def warmup(plans=((1, 4),), device=None):
    """Compile the kernel and allocate the arrival counters before capture.

    Triton compiles on first launch, which cannot happen during CUDA graph
    capture, and nothing may allocate a workspace on the timed path.
    """
    dev = torch.device("cuda") if device is None else device
    _counter(dev)
    for nseq, T in plans:
        key = (int(nseq), int(T), TARGET_CTAS, BV_MIN, BV_MAX, NUM_WARPS,
               NUM_STAGES)
        if key in _WARMED:
            continue
        H, D, CONV_K, PROJ = 16, 128, 4, 2048
        CONV_DIM = 3 * PROJ
        slen = CONV_K - 1 + (T - 1)
        M = nseq * T
        qkv = torch.zeros(M, CONV_DIM, device=dev, dtype=torch.bfloat16)
        cs = torch.zeros(2, slen, CONV_DIM, device=dev,
                         dtype=torch.bfloat16).transpose(-1, -2)
        cw = torch.zeros(CONV_DIM, CONV_K, device=dev, dtype=torch.float32)
        gg = torch.zeros(1, M, H, D, device=dev, dtype=torch.float32)
        bb = torch.zeros(1, M, H, device=dev, dtype=torch.bfloat16)
        g2 = torch.zeros(M, H, D, device=dev, dtype=torch.bfloat16)
        nw = torch.ones(D, device=dev, dtype=torch.bfloat16)
        rec = torch.zeros(2, H, D, D, device=dev, dtype=torch.float32)
        qsl = torch.arange(0, nseq + 1, device=dev, dtype=torch.int32) * T
        cidx = torch.ones(nseq, device=dev, dtype=torch.int32)
        sidx = torch.ones(nseq, T, device=dev, dtype=torch.int32)
        nacc = torch.ones(nseq, device=dev, dtype=torch.int32)
        al = torch.zeros(H, device=dev, dtype=torch.float32)
        gbi = torch.zeros(PROJ, device=dev, dtype=torch.float32)
        for gd in (torch.float32, torch.bfloat16):
            kda_decode(qkv, cs, cw, None, gg.to(gd), bb, g2, nw, rec, cidx,
                       sidx, nacc, qsl, T, al, gbi)
        _WARMED.add(key)
    torch.cuda.synchronize()
