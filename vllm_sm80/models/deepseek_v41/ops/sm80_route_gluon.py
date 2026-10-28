# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE route for decode rows (M <= sm80_router_mma.EXACT_ROWS) as one Gluon kernel (sm_80).

The selection of the served fused route (sm80_route.py) and of the prefill route (sm80_route_mma.py), with the
chosen experts' logits recomputed exactly for the weights: FP32 selection scores
``sqrt(softplus(l)) + bias`` (``bias_vl`` for image tokens), the ``K`` largest first (equal scores and NaNs lowest
expert first, NaN above every number), then the chosen experts' logits recomputed exactly from x and the BF16 weight
rows in FP64 and ``w = s / (sum + eps) * scale`` in FP64 (stable softplus), rounded once to FP32.

One program per (row t, K slice s of HIDDEN // SPLIT). Layouts are explicit:
- Keys: each expert's FP32 score becomes an order-preserving int32 (NaN = INT32_MAX, padding = INT32_MIN), computed
  spread over the CTA and handed once through shared memory into a copy held by every warp (NCH chunks of 32 * NW
  experts, NW keys per thread and chunk). A selection round is then two warp reductions (``redux.sync`` max of the
  keys, min of the expert index among the maxima) with no shared memory and no barrier; every warp knows every pick.
- Gather: the x slice and each pick's weight slice in one [NW, 32, 8] layout (8 contiguous BF16 per thread, no layout
  conversion); pick j's load is issued right after round j and its products taken after round j + 1. Products are
  summed per thread (8, in order), then within the warp.
- Hand-off: each warp's partials go to scratch ``part[t, s, warp, lane]``; after a CTA barrier an acquire-release
  counter picks the row's last program, which adds the partials (warps in order per slice, then slices in order),
  writes the weights and ids and resets the counter. Order-independent, deterministic, graph-replayable.
"""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

NUM_WARPS = 4
_INT32_MIN = gl.constexpr(-2147483648)


@gluon.jit
def _sqrt_softplus64(x):
    return libdevice.sqrt(gl.maximum(x, 0.0) + libdevice.log1p(libdevice.exp(-gl.abs(x))))


@gluon.jit
def _sqrt_softplus32(x):
    return gl.sqrt_rn(gl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x))))


@gluon.jit
def _route_kernel(
    gpart_ptr,
    logits_ptr,
    logits_stride,
    gate_rows,
    x_ptr,
    x_stride,
    w_ptr,
    bias_ptr,
    bias_vl_ptr,
    token_ptr,
    weights_ptr,
    ids_ptr,
    row_live_ptr,
    part_ptr,
    count_ptr,
    num_experts,
    sentinel_lo,
    sentinel_hi,
    scale,
    eps,
    HIDDEN: gl.constexpr,
    K: gl.constexpr,
    KP: gl.constexpr,
    NCH: gl.constexpr,
    NCHP: gl.constexpr,
    SPLIT: gl.constexpr,
    GSPLIT: gl.constexpr,
    GCOLS: gl.constexpr,
    NW: gl.constexpr,
    ROW_MASK: gl.constexpr = False,
):
    """GSPLIT > 1: row t's logits are the gate's FP64 partials ``gpart[g, t, :]`` ([GSPLIT, gate_rows, GCOLS]) added
    in order g = 0 .. GSPLIT-1 and rounded once to FP32; else the FP32 ``logits`` rows. ``part``: [M, SPLIT, NW, KP]
    FP64 scratch; ``count``: per-row counters, zero on entry and on exit."""
    DS: gl.constexpr = HIDDEN // SPLIT
    CH: gl.constexpr = 32 * NW                          # experts per key chunk (NCH of them, NCHP a power of two)
    gl.static_assert(DS == 8 * 32 * NW)
    LK: gl.constexpr = gl.BlockedLayout([NCHP], [32], [NW], [0])           # keys computed over the CTA
    LR: gl.constexpr = gl.BlockedLayout([CH // 32], [32], [NW], [0])       # a key chunk copied in every warp
    LW: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 32, 1], [NW, 1, 1], [2, 1, 0])   # x / weight slice
    LA: gl.constexpr = gl.BlockedLayout([1, 1, KP], [1, 32, 1], [NW, 1, 1], [2, 1, 0])  # per-thread pick sums
    LT: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])              # one pick lane per thread

    t = gl.program_id(0).to(gl.int64)
    s = gl.program_id(1)
    wi = gl.arange(0, NW, layout=gl.SliceLayout(1, gl.SliceLayout(2, LW)))
    li = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, LW)))
    ci = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, LW)))
    offs_d = (s * DS + gl.expand_dims(gl.expand_dims(wi, 1) * (32 * 8) + gl.expand_dims(li, 0) * 8, 2)
              + gl.expand_dims(gl.expand_dims(ci, 0), 1))
    xs = gl.load(x_ptr + t * x_stride + offs_d).to(gl.float64)

    offs = gl.arange(0, NCHP * CH, layout=LK)
    valid = offs < num_experts
    if GSPLIT > 1:
        lg64 = gl.zeros([NCHP * CH], gl.float64, LK)
        for g in gl.static_range(GSPLIT):
            lg64 += gl.load(gpart_ptr + (g * gate_rows + t) * GCOLS + offs, mask=valid, other=0.0)
        lg = lg64.to(gl.float32)
    else:
        lg = gl.load(logits_ptr + t * logits_stride + offs, mask=valid, other=0.0)
    token = gl.load(token_ptr + t)
    image = (token >= sentinel_lo) & (token < sentinel_hi)
    bias = gl.load(bias_ptr + offs, mask=valid, other=0.0)
    bias_vl = gl.load(bias_vl_ptr + offs, mask=valid, other=0.0)
    biased = _sqrt_softplus32(lg) + gl.where(image, bias_vl, bias)
    bits = biased.to(gl.int32, bitcast=True)
    order = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    order = gl.where(biased != biased, 0x7FFFFFFF, order)
    order = gl.where(valid, order, _INT32_MIN)
    keys = gl.allocate_shared_memory(gl.int32, [NCHP * CH], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    keys.store(order)
    gl.barrier()
    ords = ()
    experts = ()
    for c in gl.static_range(NCH):
        ords = ords + (keys.slice(c * CH, CH).load(LR),)
        experts = experts + (c * CH + gl.arange(0, CH, layout=LR),)

    lane_a = gl.expand_dims(gl.expand_dims(gl.arange(0, KP, layout=gl.SliceLayout(0, gl.SliceLayout(1, LA))), 0), 1)
    lane = gl.arange(0, KP, layout=LT)
    ids = gl.zeros([KP], gl.int32, LT)
    acc = gl.zeros([NW, 32, KP], gl.float64, LA)
    for j in gl.static_range(K):
        m = ords[0]
        for c in gl.static_range(1, NCH):
            m = gl.maximum(m, ords[c])
        top = gl.max(m, axis=0)
        cand = gl.where(ords[0] == top, experts[0], NCH * CH)
        for c in gl.static_range(1, NCH):
            cand = gl.minimum(cand, gl.where(ords[c] == top, experts[c], NCH * CH))
        pick = gl.min(cand, axis=0)
        removed = ()
        for c in gl.static_range(NCH):
            removed = removed + (gl.where(experts[c] == pick, _INT32_MIN, ords[c]),)
        ords = removed
        ids = gl.where(lane == j, pick, ids)
        w_cur = gl.load(w_ptr + pick.to(gl.int64) * HIDDEN + offs_d)
        if j > 0:
            acc = _accumulate(acc, w_prev, xs, lane_a, j - 1, LA)
        w_prev = w_cur
    acc = _accumulate(acc, w_prev, xs, lane_a, K - 1, LA)
    warp_sums = gl.sum(acc, axis=1)                                        # [NW, KP], within each warp

    wo = gl.expand_dims(gl.arange(0, NW, layout=gl.SliceLayout(1, gl.SliceLayout(1, LA))), 1)
    lo = gl.expand_dims(gl.arange(0, KP, layout=gl.SliceLayout(0, gl.SliceLayout(1, LA))), 0)
    gl.store(part_ptr + ((t * SPLIT + s) * NW + wo) * KP + lo, warp_sums, cache_modifier=".cg")
    gl.barrier()
    arrived = gl.atomic_add(count_ptr + t, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT - 1:
        keep = lane < K
        dots = gl.zeros([KP], gl.float64, LT)
        for i in gl.static_range(SPLIT):
            ps = gl.load(part_ptr + (t * SPLIT + i) * NW * KP + lane, cache_modifier=".cg")
            for w in gl.static_range(1, NW):
                ps += gl.load(part_ptr + ((t * SPLIT + i) * NW + w) * KP + lane, cache_modifier=".cg")
            dots += ps
        scores = _sqrt_softplus64(dots)
        total = gl.sum(gl.where(keep, scores, 0.0), axis=0)
        weights = scores / (total + eps) * scale
        if ROW_MASK:
            # The DSpark verification row mask: a masked row takes no routed expert.
            live = gl.load(row_live_ptr + t) != 0
            weights = gl.where(live, weights, 0.0)
            ids = gl.where(live, ids, -1)
        gl.store(weights_ptr + t * K + lane, weights.to(gl.float32), mask=keep)
        gl.store(ids_ptr + t * K + lane, ids, mask=keep)
        gl.store(count_ptr + t, 0)


@gluon.jit
def _accumulate(acc, w, xs, lane_a, j: gl.constexpr, LA: gl.constexpr):
    """Pick j's exact BF16 x BF16 products in FP64, summed per thread, into lane j of ``acc``."""
    pj = gl.convert_layout(gl.sum(w.to(gl.float64) * xs, axis=2), gl.SliceLayout(2, LA))
    return gl.where(lane_a == j, gl.expand_dims(pj, 2), acc)


def part_numel(M, split, kp_max):
    """FP64 scratch elements of the per-warp partials of M rows."""
    return M * split * NUM_WARPS * kp_max


def launch(x, w, bias, bias_vl, input_ids, out_w, out_ids, part, counts, gpart, logits, gate_rows, gsplit, gcols,
           split, sentinel_lo, sentinel_hi, scale, eps, options, row_live=None):
    """Route the M rows of x (one program per row and K slice); see _route_kernel for the buffers. ``options``:
    compile options of the launch. With ``row_live`` (int8 per row) a row whose entry is 0 gets ids -1 and
    weights 0."""
    M, hidden = x.shape
    E = w.shape[0]
    k = out_ids.shape[1]
    nch = triton.cdiv(E, 32 * NUM_WARPS)
    _route_kernel[(M, split)](
        gpart, logits, logits.stride(0), gate_rows, x, x.stride(0), w, bias, bias_vl, input_ids, out_w, out_ids,
        out_ids if row_live is None else row_live, part, counts, E, sentinel_lo, sentinel_hi, scale, eps, HIDDEN=hidden,
        K=k, KP=triton.next_power_of_2(k), NCH=nch, NCHP=triton.next_power_of_2(nch), SPLIT=split, GSPLIT=gsplit,
        GCOLS=gcols, NW=NUM_WARPS, ROW_MASK=row_live is not None, num_warps=NUM_WARPS, **options)
