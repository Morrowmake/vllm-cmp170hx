# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused KDA decode step with the gate projections inside (Gluon, sm_80).

    kda_decode_v2(qkv, beta, f_a, g_a, w_f, w_g, conv_state, conv_weight,
                  conv_bias, norm_weight, rec_state, conv_state_indices,
                  ssm_state_indices, num_accepted_tokens, cu_seqlens,
                  max_query_len, a_log, g_bias, lower_bound=-5.0, eps=1e-5,
                  l2_eps=1e-6, scale=None, out=None, records=None,
                  skip_norm=False, wide=False, fold=None) -> out [1, M, H, D] bf16

replaces, in meaning,

    g1 = thin_gemm(f_a, w_f); g2 = thin_gemm(g_a, w_g)
    causal_conv1d_update(...); fused_recurrent_kda(...); o_norm(o, g2)

and advances `conv_state` and `rec_state` in place exactly as they do.
The one KDA decode path of vllm/models/glm5next/common/kda.py whenever
VLLM_AMPERE_DECODE_KERNELS=1 and use_ampere_kda_decode_v2 covers the step
(16 heads of 128 under tensor parallel: up to 8 sequences of up to 5 tokens,
up to 4 sequences of 6..8; 64 heads of 128 under pipeline parallel: the
measured shapes of up to 8 sequences of up to 8 tokens) on pure verify steps;
under RecoverSSM also the spec rows of a mixed prefill + verify step
(SKIP_NORM).

SCHEDULE.  nseq * H * NV CTAs of 1 or 2 warps (see _select), NV V-slices of
BV value rows per (sequence, head), ~512 warps; the wide schedule (`wide`,
see wide_schedule) is 4 warps of 16 rows (BV = 64, NV = 2).  Per CTA:

  1. gate chunks: the head's two gate GEMMs (g1 = f_a @ w_f^T, g2 = g_a @ w_g^T,
     each [T, 128] x K = 128) are cut into 2 * D / 16 column chunks of
     [16 padded tokens, 16], dealt round-robin over the V-slices; each chunk is
     one bf16 mma_v2 (exact products, fp32 accumulate over K ascending),
     rounded to bf16, turned into the decay gate exp(lb * sigmoid(a * (g1 +
     bias))) or sigmoid(g2) and written to a workspace; then a release arrival
     on the head's gate counter.
  2. conv + silu (bf16-rounded) of every token's q, k (whole head) and v (this
     slice's rows), beta, the state load.
  3. spin (acquire) until the head's NV slices have arrived, read the gate.
  4. the recurrence over the [BV, D] slice, storing every token's state.
  5. arrival counter: the last slice of the head writes the conv window back
     and runs the gated RMSNorm over the whole head (reading sigmoid(g2)).

LAYOUTS ARE EXPLICIT (Gluon), because the recurrence is instruction-issue bound
and the per-token cost is the two row reductions over D.  The state is held
as [DJ, BV, DC] (D = DJ * DC, DC = 32) in L3: DC over 8 lanes x 4 registers
(16-B coalesced loads/stores), BV over the other 4 lanes, DJ in registers.  A
row reduction is DJ * 4 in-thread adds + 3 shuffles (was 5 shuffles per row
with D across the warp).  Token tiles q/k/gate are [DJ, BT, DC] in LT, the
same map with the tokens in registers (replicated over the 4 row lanes), so
picking a token is an in-thread select and its [DJ, DC] slice is already in
the state's broadcast layout: no layout conversion in the token loop.

The spin relies on in-order CTA dispatch: a head's slices are consecutive
program ids and a CTA only waits for slices of its own head, so it can only
stall if the NV (<= 16) slices of one head could not be resident at the same
time -- impossible on any part with more SMs than that.  There is no timeout.

NO PRECISION REDUCTION: fp32 accumulation of exact bf16 x bf16 products, the
same four bf16 roundings as the unfused path (g1, g2, conv output, recurrence
output), fp32 recurrent state, MMA only on bf16 x bf16 operands, every float
reduction fixed-shape -> bitwise deterministic run to run.  The only atomics
are int32 arrival counters.

RECOVER (RecoverSSM, see vllm_sm80/kernels/kda/recover.py): the
initial state comes from the request's single state slot (column 0) and, in
place of one recurrent state per token, every token stores its correction
c_t = (v_t - (h*e_t).k_t) * beta_t, its normalised key k_t and its decay gate
e_t (all fp32) to per-row records; no recurrent state is written.  The
rank-1 update is written as an explicit fma(c, k, h*e) (the same instruction
the unfused form compiles to), so the commit's replay of the accepted tokens
reproduces the per-token states bitwise.  SKIP_NORM leaves the recurrence
output unnormalised (the mixed prefill + verify step norms it afterwards).

FOLD (RecoverSSM commit fold, VLLM_AMPERE_KDA_FOLD, see recover.py): records
are kept per request slot (`rr`), and a row that still owes the previous
step's commit (`fl` tokens) replays those tokens from its records over the
loaded state first -- the commit's fma(c, k, h*e) sequence, so the state is
bitwise the committed one -- and stores the result back to the slot.  Its
conv window is read shifted by the owed count, as the commit's compaction
would have left it (the per-draft-slot step reads it the same way).  The
head's slices arrive on the gate counter only after reading the owed
records, so the first slice cannot overwrite them (step 4) before every
slice has read them.

CUDA-GRAPH SAFE: no autotune, no host sync, no .item(); `warmup()` compiles
and allocates the counters and the workspace before capture; counters
self-reset.

sm_80: 70 SMs (read at runtime, never hard-coded), no TMA / wgmma / fp8e4nv /
warp specialisation; BF16 MMA is m16n8k16, so the chunks pad tokens to 16.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from vllm.logger import init_logger
from vllm_sm80.kernels._platform import _is_sm80
from vllm_sm80.kernels.warmup_utils import find_module

logger = init_logger(__name__)

TARGET_CTAS = 512 # nseq * H * NV aimed at this (production value)
BV_MIN = 8
BV_MAX = 32
NUM_WARPS = 2
DC = 32                    # D chunk held across 8 lanes x 4 registers
# The wide schedule (64 heads per card, see wide_schedule): BV = 64 over 4
# warps, per-token state stores streamed (evict-first).
WIDE_HEADS = 64
WIDE_BV = 64
WIDE_NW = 4
_WIDE_ST_CACHE = ".cs"

_CTR_SLOTS = 8192
_WS_HEADS = 512            # default (sequence, head) capacity of the workspace
_WS_T = 8                  # token rows per head in the workspace
# At 32 rows per warp (one warp per CTA: nseq >= 7 at 16 heads) the RecoverSSM
# verify spills: 2,184 B of stack per thread, 2,624 B with the commit fold.
# The driver reserves the largest stack launched (at least its default
# 1,024 B) for every thread an SM can hold (2,048 per SM), and the first
# verify launch falls in the CUDA-graph memory profile, so the excess comes
# out of the KV cache. The small-stack config keeps one warp per CTA with
# SMALL_STACK_ROWS_MAX rows (more CTAs): the warp's layouts, hence every
# reduction order, stay those of the 32-row config (two warps would sum the
# q / k norms across warps), so 16 and 8 rows give bitwise the same outputs,
# records and states. At 16 rows the stack is at most 872 B per thread (the
# commit fold's verify, T 7-8 SKIP_NORM, sm_80), under the 1,024 B default, so
# nothing beyond the default is reserved; against 8 rows (twice the CTAs) the
# verify at 8 requests, T 4, measured 53.5 -> 39.9 us per call on a CMP 170HX.
SMALL_STACK_ROWS_MAX = 16


def _kda_tile(step_bound: int | None, window: int, keep_same_rows: bool = False) -> int:
    """max_query_len for the fused KDA decode kernel (per-draft-slot step and
    the recover verify): the step's bound on tokens per request when known
    (spec_max_query_len), else the full speculative window. The kernel's
    output does not depend on the tile (bitwise).

    `keep_same_rows` (recover verify): keep the window unless the token tile
    (next power of two) shrinks; with the same token tile only the conv tile
    would change, which measured slower for the recover verify at 8 requests
    (T 6: +0.214 ms/step over 34 layers on one card)."""
    if step_bound is None:
        return window
    t = max(1, min(int(step_bound), int(window)))
    if keep_same_rows and _next_pow2(t) == _next_pow2(window):
        return window
    return t


def _next_pow2(n: int) -> int:
    return 1 << (max(1, int(n)) - 1).bit_length()


def _select(nseq, H, D, small_stack=False):
    """(BV, num_warps): a pure function of the shape, taken on the host.

    Rows per warp are a power of two in [BV_MIN, BV_MAX] with nseq * H * D /
    rows close to TARGET_CTAS; BV = rows * num_warps, and NV = D / BV >= 2.
    `small_stack`: a one-warp config above SMALL_STACK_ROWS_MAX rows takes
    SMALL_STACK_ROWS_MAX rows instead.
    """
    nv = max(1, min(D // BV_MIN, TARGET_CTAS // max(1, nseq * H)))
    rw = BV_MIN
    while rw * 2 <= min(BV_MAX, D // nv):
        rw *= 2
    # Two warps per CTA (rows split across warps, the shared per-head prologue
    # split between them) pay off once there is more than one sequence and a
    # warp's state tile is at most 16 rows; one sequence is latency-bound and
    # wants the most CTAs, and 32-row warps are register-bound.
    nw = NUM_WARPS if (nseq > 1 and rw <= 16) else 1
    while nw > 1 and D // (rw * nw) < 2:
        nw //= 2
    if small_stack and nw == 1 and rw > SMALL_STACK_ROWS_MAX:
        return SMALL_STACK_ROWS_MAX, 1
    return rw * nw, nw


_SMALL_STACK = None        # VLLM_AMPERE_KDA_SMALL_STACK, read at the first launch


def _small_stack() -> bool:
    """VLLM_AMPERE_KDA_SMALL_STACK: RecoverSSM verify launches cap their rows
    per warp at SMALL_STACK_ROWS_MAX. Read once, at the first launch (the
    CUDA-graph memory profile, before any warmup)."""
    global _SMALL_STACK
    if _SMALL_STACK is None:
        from vllm import envs

        _SMALL_STACK = bool(envs.VLLM_AMPERE_KDA_SMALL_STACK)
    return _SMALL_STACK


def wide_schedule(nseq: int, tokens_per_seq: int, heads: int) -> bool:
    """Whether a 64-head step takes the wide schedule (host-side, per step).

    _select's register-bound 1- and 2-warp tiles lose to the unfused chain at
    64 heads beyond one sequence of up to 5 tokens (they spill at 8-token
    tiles); BV = 64 over 4 warps with streamed state stores is faster than
    them everywhere there and beats the unfused chain at the shapes the gate
    keeps (use_ampere_kda_decode_v2; graph replay, one CMP 170HX, us per
    layer, unfused chain -> wide; nseq x T):

        1x8  53.8 ->  43.5   2x4  54.1 ->  46.9   4x4  89.4 ->  83.9
        8x4 159.6 -> 157.1   8x8 300.2 -> 299.5   (4x8 150.7 -> 159.6: gated off)

    One sequence of up to 4 tokens keeps _select's schedule, which is faster
    there (bitwise the 1.7 output: the warp count sets the order of the D
    reductions in the conv tiles and the norm).
    """
    return heads == WIDE_HEADS and (nseq > 1 or tokens_per_seq > 4)


@gluon.jit
def _ar3(n: gl.constexpr, dim: gl.constexpr, L: gl.constexpr):
    """arange(n) laid along `dim` of the rank-3 layout L, shape [n,1,1] etc."""
    if dim == 0:
        r = gl.arange(0, n, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
        return gl.expand_dims(gl.expand_dims(r, 1), 2)
    elif dim == 1:
        r = gl.arange(0, n, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
        return gl.expand_dims(gl.expand_dims(r, 0), 2)
    else:
        r = gl.arange(0, n, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
        return gl.expand_dims(gl.expand_dims(r, 0), 1)


@gluon.jit
def _conv_silu(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
               conv_w, conv_b, chan, tok, m_tok, off,
               HAS_BIAS: gl.constexpr, CONV_K: gl.constexpr):
    """conv + silu, bf16-rounded, tap order 0..3.  `chan` and `tok` broadcast
    against each other to the tile shape (tokens x channels, any layout)."""
    p_c = csb + chan * stride_cs_chan
    for j in gl.static_range(CONV_K):
        wj = gl.load(conv_w + chan * CONV_K + j).to(gl.float32)
        from_s = (tok + j) < (CONV_K - 1)
        sv = gl.load(p_c + (off + tok + j) * stride_cs_tok,
                     mask=from_s & m_tok, other=0.0)
        xv = gl.load(x + (bos + tok + j - (CONV_K - 1)) * stride_x_t + chan,
                     mask=(~from_s) & m_tok, other=0.0)
        src = gl.where(from_s, sv, xv).to(gl.float32)
        if j == 0:
            if HAS_BIAS:
                a = gl.load(conv_b + chan).to(gl.float32) + wj * src
            else:
                a = wj * src
        else:
            a += wj * src
    a = a / (1.0 + gl.exp(-a))
    return a.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _mma_a(a_ptr, stride_a_t, bos, T, KA: gl.constexpr, MMA: gl.constexpr):
    """a[16 (tokens, masked to T), KA] as the MMA A operand."""
    AL: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    o_am = gl.arange(0, 16, layout=gl.SliceLayout(1, AL))
    o_ak = gl.arange(0, KA, layout=gl.SliceLayout(0, AL))
    return gl.load(a_ptr + gl.expand_dims(bos + o_am, 1) * stride_a_t
                   + gl.expand_dims(o_ak, 0),
                   mask=gl.expand_dims(o_am < T, 1), other=0.0)


@gluon.jit
def _mma_b(w_ptr, stride_w, row0, N: gl.constexpr, KA: gl.constexpr,
           MMA: gl.constexpr):
    """w[row0 .. row0+N-1, KA]^T as the [KA, N] MMA B operand."""
    BL: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    o_bk = gl.arange(0, KA, layout=gl.SliceLayout(1, BL))
    o_bn = gl.arange(0, N, layout=gl.SliceLayout(0, BL))
    return gl.load(w_ptr + gl.expand_dims(row0 + o_bn, 0) * stride_w
                   + gl.expand_dims(o_bk, 1))


@gluon.jit
def _gate_chunk(a_ptr, stride_a_t, bos, T, w_ptr, stride_w, row0,
                KA: gl.constexpr, MMA: gl.constexpr):
    """[16, 16] = a[16 (tokens, masked to T), KA] @ w[row0 .. row0+15, KA]^T.

    One bf16 mma_v2 chain over the whole K (ascending 16-wide steps, fp32
    accumulate), bf16-rounded.
    """
    AL: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    BL: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    o_am = gl.arange(0, 16, layout=gl.SliceLayout(1, AL))
    o_ak = gl.arange(0, KA, layout=gl.SliceLayout(0, AL))
    a = gl.load(a_ptr + gl.expand_dims(bos + o_am, 1) * stride_a_t
                + gl.expand_dims(o_ak, 0),
                mask=gl.expand_dims(o_am < T, 1), other=0.0)
    o_bk = gl.arange(0, KA, layout=gl.SliceLayout(1, BL))
    o_bn = gl.arange(0, 16, layout=gl.SliceLayout(0, BL))
    w = gl.load(w_ptr + gl.expand_dims(row0 + o_bn, 0) * stride_w
                + gl.expand_dims(o_bk, 1))
    acc = mma_v2(a, w, gl.zeros([16, 16], gl.float32, MMA))
    return acc.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _kda_decode_v2_kernel(
    x,                       # [M, CONV_DIM] bf16 view (row stride stride_x_t)
    fa,                      # [M, KA] bf16 view
    ga,                      # [M, KA] bf16 view
    wf,                      # [PROJ, KA] bf16
    wg,                      # [PROJ, KA] bf16
    conv_state,              # [nslot, CONV_DIM, SLEN] bf16 (DS view)
    conv_w,                  # [CONV_DIM, CONV_K] fp32
    conv_b,                  # [CONV_DIM] fp32 or None
    beta,                    # [1, M, H] raw beta logits (bf16 view)
    norm_w,                  # [D]
    out,                     # [1, M, H, D] bf16
    rec,                     # [nslot, H, D, D] fp32
    cu_seqlens,              # [nseq + 1] int32
    conv_idx,                # [nseq] int32 (strided view)
    ssm_idx,                 # [>= nseq, T] int32
    num_acc,                 # [nseq] int32
    a_log,                   # [H] fp32
    g_bias,                  # [PROJ] fp32
    ctr,                     # [nseq * H] int32 epilogue arrival counters (self-resetting)
    gctr,                    # [nseq * H] int32 gate arrival counters (reset by the epilogue)
    ws,                      # [heads, WS_T, 2 * D] fp32: decay gate | sigmoid(g2)
    rc,                      # RECOVER: [rows, WS_T, H, D] fp32 corrections
    rk,                      # RECOVER: [rows, WS_T, H, D] fp32 normalised keys
    re,                      # RECOVER: [rows, WS_T, H, D] fp32 decay gates
    fl,                      # FOLD: [>= nseq] int32 owed commit length of each row
    rr,                      # FOLD: [>= nseq] int32 record row (request slot), -1: none
    scale,
    eps,
    stride_x_t,
    stride_fa_t,
    stride_ga_t,
    stride_wf,
    stride_wg,
    stride_beta_t,
    stride_out_t,
    stride_cs_slot,
    stride_cs_chan,
    stride_cs_tok,
    stride_rec_slot,
    stride_cidx,
    stride_sidx_seq,
    stride_sidx_tok,
    n_sidx_tok,
    H: gl.constexpr,
    D: gl.constexpr,
    KA: gl.constexpr,
    CONV_K: gl.constexpr,
    PROJ: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    L2_EPS: gl.constexpr,
    BV: gl.constexpr,
    NV: gl.constexpr,
    BS: gl.constexpr,
    BT: gl.constexpr,
    WS_T: gl.constexpr,
    DC: gl.constexpr,
    SW: gl.constexpr,
    NW: gl.constexpr,
    RECOVER: gl.constexpr = False,
    SKIP_NORM: gl.constexpr = False,
    ST_CACHE: gl.constexpr = "",
    FOLD: gl.constexpr = False,
):
    gl.static_assert(CONV_K == 4, "conv window is specialised to width 4")
    gl.static_assert(NV > 1, "the gate chunks are shared across >= 2 slices")
    DJ: gl.constexpr = D // DC
    WS_W: gl.constexpr = 2 * D
    NJ: gl.constexpr = 2 * D // 16

    # [DJ, rows, DC]: DC over 8 lanes x 4 regs, rows over 4 lanes x NW warps,
    # DJ in regs.  Row reductions over D never leave a warp.
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 4, 8], [1, NW, 1], [2, 1, 0])
    # token tiles [BT, D]: D over LD lanes x 4 x NW warps, tokens over the
    # remaining lanes and registers (the [BT, D] conv work is split over warps)
    LD: gl.constexpr = D // (4 * NW)
    LC: gl.constexpr = gl.BlockedLayout([1, 4], [32 // LD, LD], [1, NW], [1, 0])
    SMEM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    # [BT, BV] v tile: tokens in registers, BV on the row lanes of L3
    LV: gl.constexpr = gl.BlockedLayout([BT, 1], [8, 4], [1, NW], [0, 1])
    SV: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(0, L3))     # [BV]
    # [rows, D] 2-D tiles of the epilogue
    LE: gl.constexpr = LC
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0],
                                                  warps_per_cta=[1, NW],
                                                  instr_shape=[16, 8])

    pid = gl.program_id(0)
    i_v = pid % NV
    i_nh = pid // NV
    i_n = i_nh // H
    i_h = i_nh % H

    # Every index in one round trip (no load waits on another load or on an
    # early-return branch), then one exit test.
    bos = gl.load(cu_seqlens + i_n).to(gl.int64)
    eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
    acc = gl.load(num_acc + i_n).to(gl.int64)
    if FOLD:
        n_f = gl.load(fl + i_n)
        r_row = gl.maximum(gl.load(rr + i_n), 0).to(gl.int64)
    c_slot = gl.load(conv_idx + i_n * stride_cidx).to(gl.int64)
    # the whole slot row: the token loop's slot lookups would otherwise be
    # global loads the compiler cannot hoist above the state stores
    o_sw = gl.arange(0, SW, layout=gl.SliceLayout(1, LC))
    srow = gl.load(ssm_idx + i_n * stride_sidx_seq + o_sw * stride_sidx_tok,
                   mask=o_sw < n_sidx_tok, other=0).to(gl.int64)
    T = eos - bos
    if RECOVER:
        s_init = gl.sum(gl.where(o_sw == 0, srow, 0), 0)
    else:
        s_init = gl.sum(gl.where(o_sw == acc - 1, srow, 0), 0)
    # T == 0, or NULL_BLOCK_ID (padded batch entry): nothing to do
    if (T == 0) | (s_init <= 0) | (c_slot <= 0):
        return
    if FOLD:
        # the conv window was not compacted: read it at the owed position
        off = gl.maximum(n_f, 1).to(gl.int64) - 1
    else:
        off = acc - 1
    csb = conv_state + c_slot * stride_cs_slot
    p_ws = ws + i_nh * (WS_T * WS_W)

    if FOLD:
        # ---- 0. commit fold: replay the owed tokens over the state and store
        # it back (the commit, bitwise); step 2 then loads it as usual.  Runs
        # before this slice's gate arrival: every slice of the head has read
        # the owed records before the head's first slice overwrites them.
        if n_f > 0:
            p_hf = rec + s_init * stride_rec_slot + i_h * D * D
            LW: gl.constexpr = gl.BlockedLayout([WS_T, 1], [8, 4], [1, NW], [0, 1])
            o_tw = gl.expand_dims(gl.arange(0, WS_T, layout=gl.SliceLayout(1, LC)), 1)
            o_dw = gl.expand_dims(gl.arange(0, D, layout=gl.SliceLayout(0, LC)), 0)
            p_rw = ((r_row * WS_T + o_tw) * H + i_h) * D
            k_o = gl.load(rk + p_rw + o_dw, mask=o_tw < n_f, other=0.0)
            e_o = gl.load(re + p_rw + o_dw, mask=o_tw < n_f, other=0.0)
            o_tw2 = gl.expand_dims(gl.arange(0, WS_T, layout=gl.SliceLayout(1, LW)), 1)
            o_vw = gl.expand_dims(i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, LW)), 0)
            c_o = gl.load(rc + ((r_row * WS_T + o_tw2) * H + i_h) * D + o_vw,
                          mask=o_tw2 < n_f, other=0.0)
            # Staged in shared memory, read back one token row at a time in a
            # runtime loop over the owed tokens (row views: an unswizzled
            # [WS_T, ...] buffer reinterpreted with a layout one rank lower).
            SM2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            SM1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
            ko_s = gl.allocate_shared_memory(
                gl.float32, [WS_T, DJ, DC], SMEM, gl.reshape(k_o, [WS_T, DJ, DC])
            )._reinterpret(gl.float32, [WS_T, DJ, DC], SM2)
            eo_s = gl.allocate_shared_memory(
                gl.float32, [WS_T, DJ, DC], SMEM, gl.reshape(e_o, [WS_T, DJ, DC])
            )._reinterpret(gl.float32, [WS_T, DJ, DC], SM2)
            co_s = gl.allocate_shared_memory(
                gl.float32, [WS_T, BV], SM2, c_o)._reinterpret(gl.float32, [WS_T, BV], SM1)
            LJC: gl.constexpr = gl.SliceLayout(1, L3)                 # [DJ, DC]
            gl.barrier()
            # The update is elementwise in the state, so the slice is replayed
            # in sub-tiles of 4 * NW value rows (one row per row lane: a few
            # registers per thread instead of the whole [DJ, BV, DC] tile).
            BVS: gl.constexpr = 4 * NW
            for j in gl.static_range(BV // BVS):
                f_off3 = (_ar3(DJ, 0, L3) * DC
                          + (i_v * BV + j * BVS + _ar3(BVS, 1, L3)) * D + _ar3(DC, 2, L3))
                h_f = gl.load(p_hf + f_off3)
                for t in range(0, n_f):
                    k3 = gl.expand_dims(ko_s.index(t).load(LJC), 1)     # [DJ, 1, DC]
                    e3 = gl.expand_dims(eo_s.index(t).load(LJC), 1)
                    c3 = gl.expand_dims(gl.expand_dims(
                        co_s.index(t).slice(j * BVS, BVS).load(SV), 1), 0)
                    h_f = h_f * e3
                    cb, kb = gl.broadcast(c3, k3)
                    h_f = gl.fma(cb, kb, h_f)
                gl.store(p_hf + f_off3, h_f)
            gl.barrier()

    # ---- 1. this slice's gate chunks -> workspace -> arrival ----------------
    b_a = gl.exp(gl.load(a_log + i_h).to(gl.float32))
    o_cm = gl.arange(0, 16, layout=gl.SliceLayout(1, MMA))
    m_c = gl.expand_dims(o_cm < T, 1)
    CW: gl.constexpr = D // NV
    if CW >= 16:
        # Each slice owns CW contiguous columns of both gates: all four
        # operands are loaded before any math or store (one round trip; a
        # store in between would stop the next loads from being hoisted).
        o_cw = gl.arange(0, CW, layout=gl.SliceLayout(0, MMA))
        a_f = _mma_a(fa, stride_fa_t, bos, T, KA, MMA)
        a_g = _mma_a(ga, stride_ga_t, bos, T, KA, MMA)
        w_f = _mma_b(wf, stride_wf, i_h * D + i_v * CW, CW, KA, MMA)
        w_g = _mma_b(wg, stride_wg, i_h * D + i_v * CW, CW, KA, MMA)
        gb = gl.load(g_bias + i_h * D + i_v * CW + o_cw).to(gl.float32)
        g1 = mma_v2(a_f, w_f, gl.zeros([16, CW], gl.float32, MMA))
        g2 = mma_v2(a_g, w_g, gl.zeros([16, CW], gl.float32, MMA))
        g1 = g1.to(gl.bfloat16).to(gl.float32)
        g2 = g2.to(gl.bfloat16).to(gl.float32)
        g1 = gl.exp(LOWER_BOUND / (1.0 + gl.exp(-(b_a * (g1 + gl.expand_dims(gb, 0))))))
        p_cw = p_ws + gl.expand_dims(o_cm, 1) * WS_W + i_v * CW + gl.expand_dims(o_cw, 0)
        gl.store(p_cw, g1, mask=m_c)
        gl.store(p_cw + D, 1.0 / (1.0 + gl.exp(-g2)), mask=m_c)
    else:
        # fewer than 16 columns per slice: 16-column chunks dealt round-robin
        o_cn = gl.arange(0, 16, layout=gl.SliceLayout(0, MMA))
        p_c16 = p_ws + gl.expand_dims(o_cm, 1) * WS_W + gl.expand_dims(o_cn, 0)
        for r in gl.static_range((NJ + NV - 1) // NV):
            job = i_v + r * NV
            if job < NJ:
                if job < D // 16:
                    col = job * 16
                    g = _gate_chunk(fa, stride_fa_t, bos, T, wf, stride_wf,
                                    i_h * D + col, KA, MMA)
                    gb = gl.load(g_bias + i_h * D + col + o_cn).to(gl.float32)
                    g = gl.exp(LOWER_BOUND / (1.0 + gl.exp(-(b_a * (g + gl.expand_dims(gb, 0))))))
                    gl.store(p_c16 + col, g, mask=m_c)
                else:
                    col = (job - D // 16) * 16
                    g = _gate_chunk(ga, stride_ga_t, bos, T, wg, stride_wg,
                                    i_h * D + col, KA, MMA)
                    gl.store(p_c16 + D + col, 1.0 / (1.0 + gl.exp(-g)), mask=m_c)
    gl.barrier()
    gl.atomic_add(gctr + i_nh, 1, sem="release", scope="gpu")

    # ---- 2. conv + silu of every token, beta, the state load ----------------
    o_tc = gl.expand_dims(gl.arange(0, BT, layout=gl.SliceLayout(1, LC)), 1)
    o_dc = gl.expand_dims(gl.arange(0, D, layout=gl.SliceLayout(0, LC)), 0)
    m_tc = o_tc < T
    qt = _conv_silu(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                    conv_w, conv_b, i_h * D + o_dc, o_tc, m_tc, off,
                    HAS_BIAS, CONV_K)
    kt = _conv_silu(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                    conv_w, conv_b, PROJ + i_h * D + o_dc, o_tc, m_tc, off,
                    HAS_BIAS, CONV_K)
    o_tv = gl.expand_dims(gl.arange(0, BT, layout=gl.SliceLayout(1, LV)), 1)
    o_vv = gl.expand_dims(i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, LV)), 0)
    vt = _conv_silu(x, stride_x_t, bos, csb, stride_cs_chan, stride_cs_tok,
                    conv_w, conv_b, 2 * PROJ + i_h * D + o_vv, o_tv, o_tv < T,
                    off, HAS_BIAS, CONV_K)
    o_t1 = gl.arange(0, BT, layout=gl.SliceLayout(1, LC))
    bt = gl.load(beta + (bos + o_t1) * stride_beta_t + i_h, mask=o_t1 < T,
                 other=0.0).to(gl.float32)
    bt = 1.0 / (1.0 + gl.exp(-bt))

    o_jh = _ar3(DJ, 0, L3)
    o_vh = i_v * BV + _ar3(BV, 1, L3)
    o_ch = _ar3(DC, 2, L3)
    off3 = o_jh * DC + o_vh * D + o_ch                       # [DJ, BV, DC]
    p_h = rec + s_init * stride_rec_slot + i_h * D * D
    b_h = gl.load(p_h + off3)

    # ---- 3. wait for the head's gate chunks ---------------------------------
    cnt = gl.atomic_add(gctr + i_nh, 0, sem="acquire", scope="gpu")
    while cnt < NV:
        cnt = gl.atomic_add(gctr + i_nh, 0, sem="acquire", scope="gpu")
    gl.barrier()
    gt = gl.load(p_ws + o_tc * WS_W + o_dc, mask=m_tc, other=1.0,
                 cache_modifier=".cg")

    # per-token L2 norms over D
    qt = qt / gl.expand_dims(gl.sqrt(gl.sum(qt * qt, 1) + L2_EPS), 1) * scale
    kt = kt / gl.expand_dims(gl.sqrt(gl.sum(kt * kt, 1) + L2_EPS), 1)
    # Stage the token tiles in shared memory as [BT, DJ, DC]: a token's row
    # then loads straight into the state's broadcast layout [DJ, 1, DC] (no
    # reduction, no replicated register tiles).
    q_s = gl.allocate_shared_memory(gl.float32, [BT, DJ, DC], SMEM,
                                    gl.reshape(qt, [BT, DJ, DC]))
    k_s = gl.allocate_shared_memory(gl.float32, [BT, DJ, DC], SMEM,
                                    gl.reshape(kt, [BT, DJ, DC]))
    g_s = gl.allocate_shared_memory(gl.float32, [BT, DJ, DC], SMEM,
                                    gl.reshape(gt, [BT, DJ, DC]))
    gl.barrier()

    # ---- 4. the recurrence over this CTA's [BV, D] slice --------------------
    o_vs = i_v * BV + gl.arange(0, BV, layout=SV)
    for t in gl.static_range(BT):
        if t < T:
            q3 = q_s.slice(t, 1, dim=0).permute([1, 0, 2]).load(L3)   # [DJ, 1, DC]
            k3 = k_s.slice(t, 1, dim=0).permute([1, 0, 2]).load(L3)
            e3 = g_s.slice(t, 1, dim=0).permute([1, 0, 2]).load(L3)
            b_v = gl.convert_layout(gl.sum(gl.where(o_tv == t, vt, 0.0), 0), SV)
            b_beta = gl.sum(gl.where(o_t1 == t, bt, 0.0), 0)

            b_h = b_h * e3
            b_v = (b_v - gl.sum(gl.sum(b_h * k3, 0), 1)) * b_beta
            if RECOVER:
                cb, kb = gl.broadcast(gl.expand_dims(gl.expand_dims(b_v, 1), 0), k3)
                b_h = gl.fma(cb, kb, b_h)
            else:
                b_h = b_h + gl.expand_dims(gl.expand_dims(b_v, 1), 0) * k3
            b_o = gl.sum(gl.sum(b_h * q3, 0), 1)

            if RECOVER:
                if FOLD:
                    p_r = ((r_row * WS_T + t) * H + i_h) * D
                else:
                    p_r = ((i_n * WS_T + t) * H + i_h) * D
                gl.store(rc + p_r + o_vs, b_v)
                if i_v == 0:
                    gl.store(rk + p_r + o_jh * DC + o_ch, k3)
                    gl.store(re + p_r + o_jh * DC + o_ch, e3)
            else:
                # Store the state unless the next token overwrites the same slot.
                s_t = gl.sum(gl.where(o_sw == t, srow, 0), 0)
                t_n = gl.where(t + 1 < T, t + 1, t)
                s_n = gl.sum(gl.where(o_sw == t_n, srow, 0), 0)
                live = (s_t > 0) & ((t_n == t) | (s_n != s_t))
                if live:
                    gl.store(rec + s_t * stride_rec_slot + i_h * D * D + off3, b_h,
                             cache_modifier=ST_CACHE)

            p_o = out + (bos + t) * stride_out_t + i_h * D + o_vs
            gl.store(p_o, b_o.to(gl.bfloat16), cache_modifier=".cg")

    # ---- 5. arrival: the last V-slice CTA of this (sequence, head) finishes --
    gl.barrier()
    old = gl.atomic_add(ctr + i_nh, 1, sem="acq_rel", scope="gpu")
    if old == NV - 1:
        gl.atomic_xchg(ctr + i_nh, 0, sem="release", scope="gpu")
        # every V-slice has passed its gate wait before arriving here
        gl.atomic_xchg(gctr + i_nh, 0, sem="release", scope="gpu")

        # Every load of the tail first (the three conv-window groups, the
        # staged output, sigmoid(g2), the norm weight), one barrier, then
        # every store: one memory round trip instead of four.
        o_s = gl.expand_dims(gl.arange(0, BS, layout=gl.SliceLayout(1, LE)), 1)
        o_de = gl.arange(0, D, layout=gl.SliceLayout(0, LE))
        o_d2 = gl.expand_dims(o_de, 0)
        keep: gl.constexpr = CONV_K - 2
        m_s = o_s < keep + T
        is_keep = o_s < keep
        p_c0 = csb + (i_h * D + o_d2) * stride_cs_chan
        p_c1 = csb + (PROJ + i_h * D + o_d2) * stride_cs_chan
        p_c2 = csb + (2 * PROJ + i_h * D + o_d2) * stride_cs_chan
        p_x0 = x + (bos + o_s - keep) * stride_x_t + i_h * D + o_d2
        m_k = is_keep & m_s
        m_x = (~is_keep) & m_s
        w0 = gl.where(is_keep, gl.load(p_c0 + (off + 1 + o_s) * stride_cs_tok, mask=m_k, other=0.0),
                      gl.load(p_x0, mask=m_x, other=0.0))
        w1 = gl.where(is_keep, gl.load(p_c1 + (off + 1 + o_s) * stride_cs_tok, mask=m_k, other=0.0),
                      gl.load(p_x0 + PROJ, mask=m_x, other=0.0))
        w2 = gl.where(is_keep, gl.load(p_c2 + (off + 1 + o_s) * stride_cs_tok, mask=m_k, other=0.0),
                      gl.load(p_x0 + 2 * PROJ, mask=m_x, other=0.0))

        if SKIP_NORM:
            gl.barrier()
            gl.store(p_c0 + o_s * stride_cs_tok, w0, mask=m_s)
            gl.store(p_c1 + o_s * stride_cs_tok, w1, mask=m_s)
            gl.store(p_c2 + o_s * stride_cs_tok, w2, mask=m_s)
        else:
            # gated RMSNorm over the whole head
            o_te = gl.expand_dims(gl.arange(0, BT, layout=gl.SliceLayout(1, LE)), 1)
            m_te = o_te < T
            g2s = gl.load(p_ws + o_te * WS_W + D + o_d2, mask=m_te, other=0.0,
                          cache_modifier=".cg")
            nw = gl.expand_dims(gl.load(norm_w + o_de).to(gl.float32), 0)
            p_ot = out + (bos + o_te) * stride_out_t + i_h * D + o_d2
            b_o = gl.load(p_ot, mask=m_te, other=0.0,
                          cache_modifier=".cg").to(gl.float32)
            gl.barrier()
            gl.store(p_c0 + o_s * stride_cs_tok, w0, mask=m_s)
            gl.store(p_c1 + o_s * stride_cs_tok, w1, mask=m_s)
            gl.store(p_c2 + o_s * stride_cs_tok, w2, mask=m_s)
            b_rstd = 1.0 / gl.sqrt(gl.sum(b_o * b_o, 1) / D + eps)
            b_y = b_o * gl.expand_dims(b_rstd, 1) * nw
            b_y = b_y * g2s
            gl.store(p_ot, b_y.to(gl.bfloat16), mask=m_te)


_CTR: dict = {}


def _counter(device, heads=_WS_HEADS):
    """(counters, workspace): 2 x _CTR_SLOTS int32 arrival counters (epilogue,
    gate) and the [>= heads, _WS_T, 2 * 128] fp32 gate workspace.  Grown only
    when a larger batch than any before is seen (warmup() does it up front)."""
    key = (device.type, device.index or 0)
    c = _CTR.get(key)
    if c is None or c[1].numel() < heads * _WS_T * 2 * 128:
        heads = max(heads, _WS_HEADS)
        ctr = c[0] if c is not None else torch.zeros(2 * _CTR_SLOTS, device=device,
                                                    dtype=torch.int32)
        c = (ctr, torch.zeros(heads * _WS_T * 2 * 128, device=device,
                              dtype=torch.float32))
        _CTR[key] = c
    return c


def kda_decode_v2(qkv, beta, f_a, g_a, w_f, w_g, conv_state, conv_weight,
                  conv_bias, norm_weight, rec_state, conv_state_indices,
                  ssm_state_indices, num_accepted_tokens, cu_seqlens,
                  max_query_len, a_log, g_bias, lower_bound=-5.0, eps=1e-5,
                  l2_eps=1e-6, scale=None, out=None, records=None,
                  skip_norm=False, wide=False, fold=None):
    """One fused KDA decode step including f_b_proj and g_b_proj.

    `qkv`, `beta`, `f_a`, `g_a` are read, never written.  `conv_state` and
    `rec_state` advance in place.  `out` doubles as the staging buffer for the
    unnormalised recurrence output, so its contents are transient during the
    launch.

    `records` (RecoverSSM verify): a (corrections, keys, gates) triple of
    fp32 [>= nseq, _WS_T, H, D] tensors; the initial state is read from
    column 0 of `ssm_state_indices` and no recurrent state is written.
    `skip_norm` (RecoverSSM only) leaves `out` unnormalised.  `wide` takes
    the 64-head wide schedule (the caller decides with wide_schedule).
    `fold` (RecoverSSM commit fold, needs `records`; default `records.fold`,
    see recover.LayerRecords): (owed commit lengths, record rows), int32
    [>= nseq] each; records are read and written at the record row and the
    owed tokens are replayed first (module docstring).
    """
    M, _ = qkv.shape
    H = a_log.shape[0]
    D = rec_state.shape[-1]
    KA = f_a.shape[1]
    CONV_K = conv_weight.shape[1]
    PROJ = H * D
    if scale is None:
        scale = D ** -0.5
    if out is None:
        out = torch.empty(1, M, H, D, dtype=qkv.dtype, device=qkv.device)
    assert rec_state.dtype == torch.float32, "the recurrent state stays fp32"
    assert w_f.shape == (PROJ, KA) and w_g.shape == (PROJ, KA)
    assert w_f.stride(1) == 1 and w_g.stride(1) == 1
    assert f_a.stride(1) == 1 and g_a.stride(1) == 1 and qkv.stride(1) == 1
    assert D % DC == 0 and D % 16 == 0 and KA % 16 == 0
    assert not skip_norm or records is not None, "skip_norm needs records"
    if records is not None:
        for r in records:
            assert r.dtype == torch.float32 and r.is_contiguous()
            assert r.shape[1:] == (_WS_T, H, D)
            assert r.shape[0] >= cu_seqlens.numel() - 1
    if fold is None and records is not None:
        fold = getattr(records, "fold", None)
    if fold is not None:
        assert records is not None, "fold needs records"
        for f in fold:
            assert f.dtype == torch.int32 and f.numel() >= cu_seqlens.numel() - 1

    nseq = cu_seqlens.numel() - 1
    assert nseq * H <= _CTR_SLOTS, "arrival counter buffer too small"
    if wide:
        assert H == WIDE_HEADS, "the wide schedule is for 64 heads"
        bv, num_warps, st_cache = WIDE_BV, WIDE_NW, _WIDE_ST_CACHE
    else:
        small = records is not None and _small_stack()
        bv, num_warps = _select(nseq, H, D, small_stack=small)
        st_cache = ""
    nv = D // bv
    BT = triton.next_power_of_2(max_query_len)
    assert BT <= _WS_T
    ctr, ws = _counter(qkv.device, nseq * H)
    ssm = ssm_state_indices
    if ssm.ndim == 1:
        st_seq, st_tok, n_tok = ssm.stride(0), 1, 1
    else:
        (st_seq, st_tok), n_tok = ssm.stride(), ssm.shape[1]

    _kda_decode_v2_kernel[(nv * nseq * H,)](
        qkv, f_a, g_a, w_f, w_g, conv_state, conv_weight, conv_bias, beta,
        norm_weight, out, rec_state, cu_seqlens, conv_state_indices, ssm,
        num_accepted_tokens, a_log, g_bias, ctr, ctr[_CTR_SLOTS:], ws,
        *(records if records is not None else (ws, ws, ws)),
        *(fold if fold is not None else (ctr, ctr)),
        scale, eps,
        qkv.stride(0), f_a.stride(0), g_a.stride(0),
        w_f.stride(0), w_g.stride(0),
        beta.stride(1),
        out.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        rec_state.stride(0),
        conv_state_indices.stride(0),
        st_seq, st_tok, n_tok,
        H=H, D=D, KA=KA, CONV_K=CONV_K, PROJ=PROJ,
        LOWER_BOUND=lower_bound, HAS_BIAS=conv_bias is not None, L2_EPS=l2_eps,
        BV=bv, NV=nv,
        BS=triton.next_power_of_2(CONV_K - 2 + max_query_len),
        BT=BT, WS_T=_WS_T, DC=DC, SW=triton.next_power_of_2(n_tok), NW=num_warps,
        RECOVER=records is not None, SKIP_NORM=bool(skip_norm), ST_CACHE=st_cache,
        FOLD=fold is not None,
        num_warps=num_warps,
    )
    return out


_WARMED = set()


def warmup(plans=((1, 4), (4, 4)), device=None, heads=16, recover=False,
           width=None, fold=False):
    """Compile every plan and allocate the counters and the gate workspace
    before capture.

    A plan is (nseq, T) or (nseq, T, wide): T is the token tile
    (max_query_len), `wide` the schedule (see wide_schedule).  `recover`
    compiles the RecoverSSM verify (normed and SKIP_NORM), otherwise the
    per-draft-slot step; `fold` (with `recover`) the commit-fold variants.
    `width` is the column count of the state-index rows the caller passes
    (num_speculative_tokens + 1; default T), a compile-time constant of the
    kernel.  Uses the production input layout (column views of one [M, 6416]
    buffer), so the stride specialisations Triton compiles are the ones a
    captured graph will replay.
    """
    dev = torch.device("cuda") if device is None else device
    H, D, KA, CONV_K = int(heads), 128, 128, 4
    plans = [(int(p[0]), int(p[1]), bool(p[2]) if len(p) > 2 else False)
             for p in plans]
    _counter(dev, max([n for n, _, _ in plans] + [1]) * H)
    PROJ = H * D
    CONV_DIM = 3 * PROJ
    PW = CONV_DIM + H + 2 * KA
    for nseq, T, wide in plans:
        W = max(T, int(width or T))
        key = (nseq, T, H, W, wide, TARGET_CTAS, BV_MIN, BV_MAX, NUM_WARPS, DC)
        if recover:
            key = key + ("recover",)
        if fold:
            key = key + ("fold",)
        if key in _WARMED:
            continue
        M = nseq * T
        slen = CONV_K - 1 + (W - 1)
        proj = torch.zeros(M, PW, device=dev, dtype=torch.bfloat16)
        qkv = proj[:, :CONV_DIM]
        beta = proj[:, CONV_DIM:CONV_DIM + H].unsqueeze(0)
        fa = proj[:, CONV_DIM + H:CONV_DIM + H + KA]
        ga = proj[:, CONV_DIM + H + KA:]
        wf = torch.zeros(PROJ, KA, device=dev, dtype=torch.bfloat16)
        cs = torch.zeros(2, slen, CONV_DIM, device=dev,
                         dtype=torch.bfloat16).transpose(-1, -2)
        cw = torch.zeros(CONV_DIM, CONV_K, device=dev, dtype=torch.float32)
        nw = torch.ones(D, device=dev, dtype=torch.bfloat16)
        rec = torch.zeros(2, H, D, D, device=dev, dtype=torch.float32)
        qsl = torch.arange(0, nseq + 1, device=dev, dtype=torch.int32) * T
        sidx = torch.ones(nseq, W, device=dev, dtype=torch.int32)
        nacc = torch.ones(nseq, device=dev, dtype=torch.int32)
        al = torch.zeros(H, device=dev, dtype=torch.float32)
        gbi = torch.zeros(PROJ, device=dev, dtype=torch.float32)
        recs = None
        if recover:
            recs = tuple(torch.zeros(nseq, _WS_T, H, D, device=dev,
                                     dtype=torch.float32) for _ in range(3))
        folds = None
        if fold:
            folds = (torch.zeros(nseq, device=dev, dtype=torch.int32),
                     torch.arange(nseq, device=dev, dtype=torch.int32))
        for skip in ((False, True) if recover else (False,)):
            kda_decode_v2(qkv, beta, fa, ga, wf, wf, cs, cw, None, nw, rec,
                          sidx[:, 0][:nseq], sidx, nacc, qsl, T, al, gbi,
                          records=recs, skip_norm=skip, wide=wide, fold=folds)
        _WARMED.add(key)
    if dev.type == "cuda":
        torch.cuda.synchronize()


# --- host gate ---------------------------------------------------------------
# Shapes the fused KDA decode step (kda_decode_v2 above) is validated for:
# head_dim 128, up to 8 sequences of up to 8 tokens (the gate
# workspace holds 8 token rows).
#
# 16 heads per card (tensor parallel 4): above 5 tokens per sequence only up
# to 4 sequences. There v2 was measured against the earlier fused kernel (one
# card, graph replay, us/layer: T=8 at 1/2/4 sequences 35.9->24.5,
# 45.8->29.0, 81.7->54.7; at 8 sequences T=6 124.5->146.4, T=8 151.9->172.7);
# that kernel is retired, so 5..8 sequences of 6..8 tokens take the unfused
# chain. RecoverSSM (vllm_sm80/kernels/kda/recover.py), the tensor-parallel
# default, verifies every pure step with v2 whatever this gate says.
#
# 64 heads per card (pipeline parallel): the (sequences, tokens per sequence)
# shapes in _KDA_V2_WIDE_SHAPES, where the fused step (the wide schedule
# beyond one sequence of up to 4 tokens, see kda_decode.wide_schedule)
# measured no slower than the unfused chain (one CMP 170HX, graph replay, 10
# layers per graph, 6 alternating rounds, mean <= 1.01x; the other shapes are
# 1.02-1.15x slower there and keep the unfused chain).
_KDA_V2_HEADS = 16
_KDA_V2_WIDE_HEADS = 64
_KDA_V2_HEAD_DIM = 128
_KDA_V2_MAX_TOKENS_PER_SEQ = 8
_KDA_V2_DEEP_FROM = 6          # 16 heads: tokens per sequence from here ...
_KDA_V2_DEEP_MAX_SEQS = 4      # ... only up to this many sequences
_KDA_V2_WIDE_SHAPES = frozenset({
    (1, 1), (2, 1), (3, 1), (4, 1),
    (1, 2), (2, 2), (3, 2), (4, 2),
    (1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
    (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4),
    (1, 5), (2, 5),
    (1, 6), (2, 6),
    (1, 7),
    (1, 8), (6, 8), (7, 8), (8, 8),
})
_KDA_V2_MAX_SEQS = 8


def use_ampere_kda_decode_v2(
    num_seqs: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    w_f: torch.Tensor | None = None,
    w_g: torch.Tensor | None = None,
) -> bool:
    """Gate for kda_decode_v2 above.

    Resolved on the host, once per layer call, before any launch. Outside the
    covered shapes, or with any condition unmet, the caller keeps the unfused
    chain (f_b/g_b GEMMs, causal_conv1d_update, fla fused_recurrent_kda,
    o_norm). ``w_f``/``w_g`` are the f_b_proj and g_b_proj weights: the
    kernel reads them directly, so they must be plain bf16 [H * D, D]
    row-major tensors (unquantized, as in the W4A16 checkpoint).
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS:
        return False
    if num_seqs < 1 or num_seqs > _KDA_V2_MAX_SEQS:
        return False
    if num_tokens < num_seqs or num_tokens % num_seqs:
        return False
    tokens_per_seq = num_tokens // num_seqs
    if tokens_per_seq > _KDA_V2_MAX_TOKENS_PER_SEQ:
        return False
    if head_dim != _KDA_V2_HEAD_DIM:
        return False
    if num_heads == _KDA_V2_HEADS:
        if tokens_per_seq >= _KDA_V2_DEEP_FROM and num_seqs > _KDA_V2_DEEP_MAX_SEQS:
            return False
    elif num_heads == _KDA_V2_WIDE_HEADS:
        if (num_seqs, tokens_per_seq) not in _KDA_V2_WIDE_SHAPES:
            return False
    else:
        return False
    for w in (w_f, w_g):
        if w is None:
            continue
        if (
            not isinstance(w, torch.Tensor)
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or tuple(w.shape) != (num_heads * head_dim, head_dim)
            or w.stride(1) != 1
        ):
            return False
    return _is_sm80()


# --- warmup from the worker ---------------------------------------------------
def warmup_from_worker(worker, capture_sizes) -> None:
    """No-op unless VLLM_AMPERE_DECODE_KERNELS. Runs before CUDA graph capture;
    see vllm_sm80/kernels/warmup_utils.py for why."""
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS:
        if envs.VLLM_AMPERE_KDA_SMALL_STACK:
            logger.info_once(
                "VLLM_AMPERE_KDA_SMALL_STACK=1 set but off: "
                "VLLM_AMPERE_DECODE_KERNELS is off, the fused KDA verify is not used."
            )
        return
    _warmup_kda_v2(worker, worker.get_model(), worker.device, capture_sizes)


def _warmup_kda_v2(worker, model, device, capture_sizes) -> None:
    """Compile kda_decode_v2 for every (nseq, tile, schedule) a pure verify
    step can take (RecoverSSM: also the SKIP_NORM variant of mixed steps) and
    allocate its arrival counters and gate workspace, before capture."""
    layer = find_module(model, "_conv_state_dim_first")
    if layer is None or not getattr(layer, "_kda_v2", False):
        if _small_stack():
            logger.info_once(
                "VLLM_AMPERE_KDA_SMALL_STACK=1 set but off: the fused KDA "
                "decode is not used by this model."
            )
        return
    heads = int(layer.local_num_heads)
    head_dim = int(layer.head_dim)
    num_spec = int(getattr(layer, "num_spec", 0) or 0)
    window = num_spec + 1
    max_seqs = int(worker.vllm_config.scheduler_config.max_num_seqs)
    step_tile = getattr(layer, "_kda_step_tile", False)
    wide = getattr(layer, "_kda_wide", False)
    if getattr(layer, "_kda_recover", False):
        # RecoverSSM: every spec-verify step takes the recover variants (pure
        # and mixed), for 1..max_num_seqs sequences; under the commit fold
        # their fold variants.
        from vllm_sm80.kernels.kda.recover import kda_fold_enabled

        fold = kda_fold_enabled(worker.vllm_config)
        rts = range(1, window + 1) if step_tile else (window,)
        rplans = tuple((n, t) for t in rts for n in range(1, max_seqs + 1))
        warmup(
            plans=rplans, device=torch.device(device), heads=heads, recover=True,
            width=window, fold=fold,
        )
        logger.info(
            "Warmed up sm_80 KDA recover verify%s for (nseq, T) in %s.",
            " (commit fold)" if fold else "", list(rplans),
        )
        if _small_stack():
            changed = [n for n in range(1, max_seqs + 1)
                       if _select(n, heads, head_dim, small_stack=True)
                       != _select(n, heads, head_dim)]
            if changed:
                logger.info_once(
                    "sm_80 KDA verify small-stack configs active "
                    "(VLLM_AMPERE_KDA_SMALL_STACK=1): one warp of %d rows per CTA, the "
                    "commit fold's verify included, where today's config has one "
                    "32-row warp (nseq 7-8 at 16 heads).",
                    SMALL_STACK_ROWS_MAX,
                )
            else:
                logger.info_once(
                    "VLLM_AMPERE_KDA_SMALL_STACK=1 set but off: no verify batch up to "
                    "max_num_seqs %d has a 32-row warp to replace.", max_seqs,
                )
        return
    if _small_stack():
        logger.info_once(
            "VLLM_AMPERE_KDA_SMALL_STACK=1 set but off: RecoverSSM is off for "
            "this configuration; the KDA decode keeps its block configs."
        )
    nseqs = {max(1, int(s) // window) for s in capture_sizes if int(s) >= 1}
    nseqs.update(range(1, max_seqs + 1))
    plans = set()
    for n in sorted(nseqs):
        for t in range(1, window + 1):
            if not use_ampere_kda_decode_v2(
                n, n * t, heads, head_dim, layer.f_b_proj.weight,
                layer.g_b_proj.weight,
            ):
                continue
            w = wide and wide_schedule(n, t, heads)
            plans.add((n, t if step_tile else window, w))
    if not plans:
        logger.info(
            "sm_80 KDA decode v2: its gate is closed at %d heads (unsupported "
            "shape); using the unfused path.",
            heads,
        )
        return
    warmup(
        plans=tuple(sorted(plans)), device=torch.device(device), heads=heads,
        width=window,
    )
    logger.info(
        "Warmed up sm_80 KDA decode v2 kernel for (nseq, T, wide) in %s at %d "
        "heads.",
        sorted(plans),
        heads,
    )
