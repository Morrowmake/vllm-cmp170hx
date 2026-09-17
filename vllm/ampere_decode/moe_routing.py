#!/usr/bin/env python
"""Candidate kernels for the GLM-5.3-Flash MoE routing preamble.

Keep this kernel in sync with the standalone source.
`bench.py`, `validate.py` and
`reference/` are the frozen contract.

At the production decode point (M = 4, E = 288, topk = 8, block_size = 48) the
incumbent is four kernels - `single_group_topk_block_kernel`,
`moe_align_block_size_kernel`, `count_and_sort_expert_tokens_kernel` and
`moe_sum_vec_kernel` - and the first three are adjacent and consume only
`topk_ids`, which the first one produces.  This file replaces them with:

  `fused_route_align`   ONE Triton kernel, grid (1,), doing routing + block
                        alignment + count-and-sort + the sentinel fill.
  `moe_sum`             the top-8 reduction, which must stay separate because
                        the Marlin GEMMs run between the routing and it.

so 4 (5 with the mask elementwise) launches per MoE layer become 2.

WHERE THE TIME ACTUALLY GOES (measured on this part, graph-replay, M=4)
    empty Triton kernel in a graph          0.98 us
    load logits + sigmoid + store, no top-k 1.09 us
    ... + one block reduction               3.0  us
    ... + eight block reductions            7.7  us
    the align half, one-hot histogram body  ~3   us   (removed)
    the align half, pairwise body           ~0.6 us   (this file, see below)
    the incumbent's routing kernel          4.35 us
    the incumbent's align (2 kernels)       5.55 us
A block reduction over a [4, 512] tile costs ~0.6-0.9 us in a single CTA on
this card, and the top-8 needs eight of them no matter how it is arranged.
That single number is now the whole cost of this family: the align half has
been taken out of the E dimension entirely (measurement: "it spends 6 us
arranging 32 pairs because its work is O(num_experts) regardless of M") by
computing every alignment output from PAIRWISE relations between the
M*topk pairs instead of from a 288-bin histogram.  At M=4 that is a 32x32 tile
instead of a 32x512 one, and the family goes 1.02x -> 1.25x at the production
shape.  See `_align_body`.

SHAPE DISPATCH.  One CTA is the right structure only while M is small: its cost
grows linearly in M while the incumbent's is flat (it uses one CTA per token).
`fused_route_align` therefore runs the fused single-CTA kernel for
M <= FUSE_MAX and otherwise splits the routing across ceil(M / BLOCK_M) CTAs
and leaves the alignment in its own single-CTA kernel - 3 launches instead of
2, still fewer than the incumbent's 4.  Both paths share the same two
`@triton.jit` device functions, so there is one implementation of the
semantics, not two.

A second dispatch, on the pair count rather than on M: the align body has two
implementations of the SAME semantics and `PAIRWISE_MAX` picks between them at
numel = M*topk = 32.

  <= 32 pairs   fully pairwise: no histogram at all, every output from
                relations between the pairs (see `_align_body`).
  >  32 pairs   an E-wide histogram again, because the pairwise body's
                full-width vectors stop fitting in registers - but built with
                NUMEL integer ATOMICS instead of a [BLOCK_NK, BLOCK_E] one-hot
                reduction, and with each pair's block offset GATHERED from a
                scratch [E] vector instead of masked out of an E-wide register
                vector.  That is 256 atomics plus one NUMEL^2 rank tile where
                the one-hot form was ~390 k tile elements, and it took M=32
                from 0.30x to 0.55x and M=16 from 0.51x to 0.85x.

Both are this file's own code; nothing here calls the incumbent op at any
shape.

CONSTRAINTS honoured here
  * sm_80: no fp8, no TMA, no wgmma, no warp specialization.  No `tl.dot` at
    all - there is no matmul in this op.
  * NO PRECISION REDUCTION: fp32 logits -> fp32 scores (the incumbent's exact
    `0.5*tanh(0.5*x)+0.5` form) -> fp32 weights; int32 ids; bf16 in / fp32
    accumulate / bf16 out for `moe_sum`, accumulated in slot order exactly like
    `moe_sum_vec_kernel`, which it matches bitwise.
  * CUDA-GRAPH CAPTURABLE: no autotune on the timed path, no host sync, no
    `.item()`, no allocation that has to happen during capture.
  * DETERMINISTIC: the top-k is a fixed sequence of single-operand maxima over
    an exactly packed (key, index) int64, and the renormalizing sum runs in
    slot order, so the routing outputs are bitwise reproducible.  The align
    outputs are reproducible too - pair ranks are an explicit "how many earlier
    pairs chose this expert" count - which is STRONGER than the incumbent,
    whose `count_and_sort_expert_tokens_kernel` ranks with a global atomicAdd.

TUNING SURFACE (available configuration choices)
  * `_config` picks BLOCK_M / BLOCK_E / BLOCK_NK / BLOCK_P / num_warps, and
    `FUSE_MAX` picks the structure.  Nothing else is shape-dependent.
  * The eight block reductions are the whole cost of the routing half.  A
    hierarchical selection (per-group top-8 over a narrow inner axis, then a
    merge) or a threshold-and-compact scheme that reduces their number or their
    width is the obvious next move, and the biggest single lever in this file.
  * `moe_sum` at M=4 moves 262 KB in ~1.6 us against a 0.18 us roofline and a
    ~1.25 us per-kernel floor; there is roughly 0.3 us in it, no more.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Above this many tokens the single-CTA fused kernel loses to splitting the
# routing across CTAs; see the module docstring.  Production decode is M=4.
FUSE_MAX = 4


def _next_pow2(n):
    return 1 << max(0, (int(n) - 1)).bit_length()


# The align body has two implementations of the same semantics and picks between
# them on the pair count (see `_align_body`).  Up to this many (token, slot)
# pairs the fully pairwise O(NUMEL^2) body wins; above it the atomic-histogram
# body does, because the pairwise body's full-width [BLOCK_N] vectors and its
# [BLOCK_NK, BLOCK_N] tiles stop fitting in one CTA's registers.
# Measured on GPU 2, whole family, `bench.py --repeats 1`:
#
#   M                   1     2     4     8    16    32
#   pairwise         1.34  1.50  1.28  0.87  0.53  0.21
#   atomic histogram 1.33  1.35  1.18  0.98  0.85  0.55
#   one-hot histogram (removed)  1.57  1.28  1.02  0.71  0.51  0.30
#
# so the crossover sits between 32 and 64 pairs: M=8 (64 pairs) prefers the
# atomic body, M=4 (32 pairs) the pairwise one.
PAIRWISE_MAX = 32

# Above this many pairs the align half runs as TWO multi-CTA kernels instead of
# one single-CTA one: `_align_prep_kernel` (the sentinel fill spread over CTAs,
# the meta phase on CTA 0) then `_align_scatter_kernel` (one CTA per pair
# chunk).  It costs one extra launch, so it only pays once the single CTA has
# enough to serialise.  Measured over three repeats, whole family:
#
#   SPLIT_FROM  BLOCK_R      m8     m16     m32   family
#      (none)         -    0.98    0.84    0.55    1.066
#         128         4    0.98    0.93    0.82    1.160
#         128         8    0.98    0.92    0.82    1.160   <- chosen
#         128        32    0.98    0.89    0.77    1.140
#         128        64    0.98    0.88    0.73    1.122
#         128       128    0.98    0.83    0.65    1.098
#          64        64    0.92    0.87    0.73    1.094
#
# SPLIT_FROM=64 costs M=8 six points for nothing: at 64 pairs the single CTA is
# already faster than the extra launch, which is the same launch-vs-parallelism
# trade as `FUSE_MAX`.  Narrow scatter chunks win because the rank tile is
# [BLOCK_R, BLOCK_N] and what matters is how many CTAs share the NUMEL^2
# comparisons: at M=32, BLOCK_R=8 is 32 CTAs against 2 at BLOCK_R=128.
SPLIT_FROM = 128
BLOCK_R = 8


def _config(M, topk, num_experts, block_size, mnp, nblk, fused):
    """Block shapes and warp count.  Pure heuristic; tune against measurements.

    BLOCK_M rows of the [BLOCK_M, BLOCK_E] score tile are selected at once, so
    the eight block reductions are amortised over BLOCK_M tokens; past 4 rows
    the packed int64 tile starts to spill on this part.

    `BLOCK_N` is the full pair count rounded up: the pairwise align body holds
    `cnt`/`rank`/`padded`/`first` as full [BLOCK_N] vectors (one element per
    thread at these sizes) and pairs them against a BLOCK_NK-wide chunk, so its
    tiles are [BLOCK_N, BLOCK_NK] and [BLOCK_NK, BLOCK_N], never E-wide.
    BLOCK_NK = 64 measured best for the pairwise body (M=1 7.64 us against 9.87
    at 32 and 16).  The atomic-histogram body wants the chunk as WIDE as the
    pair count instead - its per-chunk cost is dominated by one rank tile and
    fewer iterations is strictly better (M=32: 41.3 us at 16, 33.5 at 32,
    29.5 at 64, 27.3 at 128, 25.8 at 256) - so the two paths take different
    chunk widths.
    """
    # Both knobs below were re-swept after the bitonic rewrite, because both
    # of the old body's justifications were void: `num_warps=8` had been chosen
    # for a WIDE reduction that no longer exists, and `block_m`'s cap of 4 was
    # measured against an 80-register body that the bitonic form replaced with
    # a 40-register one.  Statically both looked like upside (nw=16 is 1493
    # instructions against nw=8's 2536; the bitonic body does not spill at
    # BLOCK_M=8/16).  On hardware, three repeats, whole family:
    #
    #   warps  block_m cap    family
    #       8            4    1.3997  <- both knobs unchanged
    #       8            8    1.3322
    #      16            4    1.3281
    #      16            8    1.2175
    #
    # so the static instruction count is not what this kernel is bound by and
    # the old values survive on their own merit.  Do not re-derive from PTX.
    numel = M * topk
    pairwise = numel <= PAIRWISE_MAX
    block_e = max(64, _next_pow2(num_experts))
    # Split the expert axis instead of padding it to a power of two: EA is the
    # largest power of two below E and EB covers the remainder.  E=288 -> 256+32
    # (288 reduction lanes) where one tile would be 512.
    e_a = max(16, _next_pow2(num_experts) // 2)
    if e_a >= num_experts:
        e_a, e_b = max(16, _next_pow2(num_experts)), 0
    else:
        e_b = max(16, _next_pow2(num_experts - e_a))
    block_m = min(_next_pow2(M), 4)
    block_k = _next_pow2(topk)
    block_n = max(16, _next_pow2(numel))
    block_nk = min(64 if pairwise else 256, _next_pow2(numel))
    split = (not fused) and numel >= SPLIT_FROM
    # `tl.topk` needs a power-of-two k no larger than either tile.
    bitonic = block_k >= 4 and e_a >= block_k and (e_b == 0 or e_b >= block_k)
    block_p = min(2048, _next_pow2(max(mnp, nblk)))
    num_warps = 8
    return (block_m, block_e, block_k, block_n, block_nk, block_p, pairwise,
            split, e_a, e_b, bitonic, num_warps)


@triton.jit
def _route_keys(logits_ptr, bias_ptr, offs_m, mask_m, e0,
                stride_lm, stride_le, E: tl.constexpr, EW: tl.constexpr):
    """The packed (key, index) tile for experts [e0, e0 + EW).

    key    = 0.5 * tanh(0.5 * logit) + 0.5 + bias      (the incumbent's exact
             sigmoid form; `tl.sigmoid` rounds differently and can flip a
             near-tie, which would be a precision reduction of the DECISION)
    packed = (monotone_bits(key) << 32) | (E - 1 - e)

    The int32 -> monotone map is exact, so no mantissa bit is sacrificed, and
    `E - 1 - e` in the low word makes the LOWER expert index win an exact tie,
    which is what `argsort(..., descending=True, stable=True)` does.
    """
    offs_e = e0 + tl.arange(0, EW)
    valid_e = offs_e < E
    bias = tl.load(bias_ptr + offs_e, mask=valid_e, other=0.0).to(tl.float32)
    x = tl.load(logits_ptr + offs_m[:, None] * stride_lm
                + offs_e[None, :] * stride_le,
                mask=mask_m[:, None] & valid_e[None, :],
                other=0.0).to(tl.float32)
    score = 0.5 * libdevice.tanh(0.5 * x) + 0.5
    key = tl.where(valid_e[None, :], score + bias[None, :], float("-inf"))
    b32 = key.to(tl.int32, bitcast=True)
    packed = (((b32 ^ ((b32 >> 31) & 0x7FFFFFFF)).to(tl.int64) << 32)
              | (E - 1 - offs_e).to(tl.int64)[None, :])
    return packed, x


@triton.jit
def _route_tile(logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr, m0,
                stride_lm, stride_le, routed_scaling_factor,
                M: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
                EA: tl.constexpr, EB: tl.constexpr, BITONIC: tl.constexpr,
                RENORM: tl.constexpr):
    """Route BLOCK_M tokens starting at m0: sigmoid, top-k, renormalize, scale.

    ONE REDUCTION PER SELECTED EXPERT.  The obvious top-k loop costs two block
    reductions per k (an argmax, then a masked sum to fetch the winner's
    score); `tl.max(..., return_indices=True)` is one reduction but a
    two-operand one, which Triton lowers through shared memory for both
    operands and is no cheaper (measured: 9.6 us vs 9.6 us at M=4).  Instead
    the corrected score and the expert index are PACKED into one int64 -

        key_i32 = bits(key) ^ ((bits(key) >> 31) & 0x7fffffff)     (monotone)
        packed  = (key_i32 << 32) | (E - 1 - e)

    - so a single-operand `tl.max` over int64 returns the winner's key and its
    index together, and `E - 1 - e` in the low word makes the LOWER expert
    index win an exact tie, which is what `argsort(..., descending=True,
    stable=True)` in the reference does.  The packing is exact - no mantissa
    bit is sacrificed - so the routing DECISION is bit-for-bit the fp32 one.
    The winner's score is then recovered with a [BLOCK_M] gather of its logit
    (the whole logit row is 1.2 KB and L1-resident) rather than a second block
    reduction.  Measured at M=4: 9.6 us for the two-reduction form, 7.7 us for
    this one.
    """
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # THE EXPERT AXIS IS SPLIT, NOT PADDED.  E = 288 and `tl.arange` needs a
    # power of two, so a single tile is 512 wide and 44 % of every one of the
    # eight top-k reductions is `-inf` padding.  Two tiles of 256 + 32 cover
    # exactly 288, and the eight reductions are the whole cost of the routing
    # half (7.7 us of the 10.9 at M=4 by phase ablation), so that padding was
    # ~2 us per call.  The maxima of the two tiles combine with one scalar
    # `tl.maximum`, and the packed key is unique per expert, so `packed == mx`
    # still matches exactly one lane across both tiles.
    pa, xa = _route_keys(logits_ptr, bias_ptr, offs_m, mask_m, 0,
                         stride_lm, stride_le, E, EA)
    if EB > 0:
        pb, xb = _route_keys(logits_ptr, bias_ptr, offs_m, mask_m, EA,
                             stride_lm, stride_le, E, EB)

    if BITONIC:
        # ZERO CROSS-LANE REDUCTIONS OVER THE EXPERT AXIS.  The eight-pass form
        # below costs eight dependent `tl.max` trees over the full 288 lanes,
        # plus eight full-width mask-out passes, plus eight dependent
        # single-token gathers - and measurement showed ~0.64 us of each pass's
        # 0.8 us as fixed tree/barrier cost, i.e. the thing that does not go
        # away by narrowing the candidate set.
        #
        # `tl.topk` (Triton 3.7.1, `language/standard.py`, = `sort_impl`
        # descending) is a BITONIC network: for k = 8 every compare-exchange it
        # emits eliminates only flat-index bits 0..3, so no exchange ever
        # crosses a 16-element window - and at BLOCK_M=4 x 256 lanes over 8
        # warps that is 4 elements per thread, so bits 0-1 are register-local
        # and 2-3 are one `shfl.bfly` inside one warp.  Nothing crosses a warp.
        # Statically, at BLOCK_M=4/nw=8: 35 -> 10 barriers for the selection,
        # 691 -> 335 shuffles, 4198 -> 2536 instructions, 80 -> 40 registers,
        # no spill.
        #
        # EXACTNESS IS INHERITED, NOT RE-ARGUED.  `tl.topk` compares the same
        # packed int64 with `maxsi`/`minsi`, and signed int64 order on
        # `(monotone(key) << 32) | (E - 1 - e)` is exactly lexicographic
        # (key descending, e ascending) - so the decision is the same fp32
        # comparison and an exact tie still goes to the LOWER expert index.
        # The results come out descending, so slot order and the renorm sum are
        # unchanged.  EA + EB = 288 exactly, so there are no padding lanes to
        # guard (a single padded 512 tile WOULD need `tl.where(valid_e, p,
        # INT64_MIN)`: `E - 1 - e` goes negative past e = 287 and a
        # sign-extended low word outranks every legitimate key).
        ta = tl.topk(pa, BLOCK_K)
        if EB > 0:
            tb = tl.topk(pb, BLOCK_K)
            top = tl.topk(tl.join(ta, tb).reshape([BLOCK_M, 2 * BLOCK_K]),
                          BLOCK_K)
        else:
            top = ta
        # One [BLOCK_M, BLOCK_K] gather where the loop did BLOCK_K dependent
        # [BLOCK_M] ones, each behind its own reduction.
        sel_i = (E - 1 - (top & 0xFFFFFFFF)).to(tl.int32)
        xg = tl.load(logits_ptr + offs_m[:, None] * stride_lm
                     + sel_i * stride_le,
                     mask=mask_m[:, None], other=0.0).to(tl.float32)
        sel_v = tl.where((offs_k < TOPK)[None, :],
                         0.5 * libdevice.tanh(0.5 * xg) + 0.5, 0.0)
    else:
        # topk <= 2: the bitonic network is barrier-flat at 18 while the loop
        # is 11 (topk=2) and 6 (topk=1), so the loop stays for those.
        sel_v = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
        sel_i = tl.zeros([BLOCK_M, BLOCK_K], tl.int32)
        for k in tl.static_range(TOPK):
            mx = tl.max(pa, axis=1)
            if EB > 0:
                mx = tl.maximum(mx, tl.max(pb, axis=1))
            idx = (E - 1 - (mx & 0xFFFFFFFF)).to(tl.int32)
            xg = tl.load(logits_ptr + offs_m * stride_lm + idx * stride_le,
                         mask=mask_m, other=0.0).to(tl.float32)
            v = 0.5 * libdevice.tanh(0.5 * xg) + 0.5
            slot = offs_k == k
            sel_v = tl.where(slot[None, :], v[:, None], sel_v)
            sel_i = tl.where(slot[None, :], idx[:, None], sel_i)
            neg = tl.full([1], -0x8000000000000000, tl.int64)
            pa = tl.where(pa == mx[:, None], neg, pa)
            if EB > 0:
                pb = tl.where(pb == mx[:, None], neg, pb)

    keep = mask_m[:, None] & (offs_k < TOPK)[None, :]
    w = sel_v
    if RENORM:
        w = w / (tl.sum(sel_v, axis=1)[:, None] + 1e-20)
    w = w * routed_scaling_factor
    off = offs_m[:, None] * TOPK + offs_k[None, :]
    tl.store(topk_ids_ptr + off, sel_i, mask=keep)
    tl.store(topk_w_ptr + off, w, mask=keep)


@triton.jit
def _align_meta(topk_ids_ptr, expert_ids_ptr, ntpp_ptr, cnt_ptr, excl_ptr,
                E: tl.constexpr, BS: tl.constexpr, NBLK: tl.constexpr,
                NUMEL: tl.constexpr, BLOCK_E: tl.constexpr,
                BLOCK_NK: tl.constexpr, BLOCK_P: tl.constexpr):
    """Counts, padded prefix, `expert_ids`, `num_tokens_post_pad`.  ONE CTA.

    The histogram is NUMEL integer atomics, not a [BLOCK_NK, BLOCK_E] one-hot
    reduction: 256 atomics against ~130 k tile elements at M=32, and at ~1-way
    contention over 288 bins, nowhere near the 25x collapse global atomics hit
    at 32-way.  `+1` integer atomics are order-independent, so the counts stay
    reproducible.

    `cnt_ptr` is left at zero on exit - the same self-resetting discipline as
    kda_decode's arrival counter - so a captured graph replays correctly.
    `excl_ptr` is published so the scatter can gather each pair's offset with
    one load instead of reducing an E-wide masked tile per pair.
    """
    offs_e = tl.arange(0, BLOCK_E)
    offs_p = tl.arange(0, BLOCK_P)
    offs_c = tl.arange(0, BLOCK_NK)
    valid_e = offs_e < E

    for b0 in tl.static_range(0, NBLK, BLOCK_P):
        b = b0 + offs_p
        tl.store(expert_ids_ptr + b, tl.full([BLOCK_P], -1, tl.int32),
                 mask=b < NBLK)

    for c0 in tl.static_range(0, NUMEL, BLOCK_NK):
        i = c0 + offs_c
        mi = i < NUMEL
        e = tl.load(topk_ids_ptr + i, mask=mi, other=0)
        tl.atomic_add(cnt_ptr + e, 1, mask=mi)
    tl.debug_barrier()

    cnt = tl.load(cnt_ptr + offs_e, mask=valid_e, other=0)
    padded = tl.where(valid_e, ((cnt + BS - 1) // BS) * BS, 0)
    excl = tl.cumsum(padded, axis=0) - padded
    tl.store(ntpp_ptr, tl.sum(padded))
    nblk_e = padded // BS
    blk0 = excl // BS

    tl.debug_barrier()          # the -1 fill above lands before these writes
    for j in tl.static_range(0, (NUMEL + BS - 1) // BS):
        tl.store(expert_ids_ptr + blk0 + j, offs_e.to(tl.int32),
                 mask=valid_e & (j < nblk_e))

    tl.store(excl_ptr + offs_e, excl, mask=valid_e)
    tl.store(cnt_ptr + offs_e, tl.zeros([BLOCK_E], tl.int32), mask=valid_e)


@triton.jit
def _align_scatter(topk_ids_ptr, sorted_ids_ptr, excl_ptr, r0,
                   NUMEL: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_R: tl.constexpr):
    """Place the BLOCK_R pairs starting at `r0`.  Independent of every other
    chunk, which is what lets the SPLIT path run one CTA per chunk.

    `rank(i) = #{j < i : e_j == e_i}` against ALL pairs (columns full width),
    an explicit count rather than the incumbent's global `atomicAdd`, so the
    layout is reproducible.  `excl[e_i]` is a gather from the scratch the meta
    phase published.
    """
    jj = tl.arange(0, BLOCK_N)
    vj = jj < NUMEL
    r = r0 + tl.arange(0, BLOCK_R)
    vr = r < NUMEL

    ej = tl.load(topk_ids_ptr + jj, mask=vj, other=-1)
    er = tl.load(topk_ids_ptr + r, mask=vr, other=0)
    rank = tl.sum(((ej[None, :] == er[:, None]) & vj[None, :]
                   & (jj[None, :] < r[:, None])).to(tl.int32), axis=1)
    ex = tl.load(excl_ptr + er, mask=vr, other=0)
    tl.store(sorted_ids_ptr + ex + rank, r.to(tl.int32), mask=vr)


@triton.jit
def _align_body(topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
                cnt_ptr, excl_ptr,
                E: tl.constexpr, BS: tl.constexpr, MNP: tl.constexpr,
                NBLK: tl.constexpr, NUMEL: tl.constexpr,
                BLOCK_E: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_NK: tl.constexpr, BLOCK_P: tl.constexpr,
                PAIRWISE: tl.constexpr):
    """`moe_align_block_size` + `count_and_sort_expert_tokens` in one CTA.

    THE WORK HERE IS O(NUMEL^2), NOT O(NUMEL x E).  The incumbent - and the
    first version of this body - build a 288-bin histogram and an exclusive
    prefix sum over it, so both spend O(E) per pair arranging M*topk = 32 pairs
    at the production shape; this was measured as 6 us of pure
    E-shaped work.  Nothing in the output actually needs the empty bins:

      * `rank(i)  = #{j < i : e_j == e_i}`                    - pairwise
      * `cnt(i)   = #{j     : e_j == e_i}`                    - pairwise
      * `first(i) = rank(i) == 0`, one pair per distinct expert
      * `excl(e_i) = sum_j [first(j) and e_j < e_i] pad(cnt(j))`  - pairwise,
        because summing the padded counts over the FIRST-OCCURRENCE pairs below
        e_i is exactly summing them over the distinct experts below e_i
      * `num_tokens_post_pad = sum_j [first(j)] pad(cnt(j))`
      * expert e_i owns `ceil(cnt(i)/BS)` blocks from `excl(e_i)/BS`, scattered
        from its first-occurrence pair instead of from an E-wide vector

    so E never appears.  At M=4 that is 32x32 = 1 k tile elements instead of
    32x512 = 16 k, and the whole align body stops scaling with `num_experts`.

    The two passes are shaped so that every pairwise tile stays small and no
    per-pair intermediate ever has to be scattered into registers:

      pass 1  rows FULL [BLOCK_N], columns chunked -> cnt, rank as full vectors
      pass 2  rows chunked, columns FULL           -> position and expert_ids,
                                                      consumed immediately

    NO GLOBAL SCRATCHPAD AND ONE BARRIER (a global round trip plus a barrier in
    a single CTA costs ~0.5 us on this part).  The single `tl.debug_barrier`
    orders the two sentinel fills - and, in the fused kernel, the `topk_ids`
    stores - before the pairwise passes read them back.

    Ranks are an explicit count, so the layout is reproducible: stronger than
    the incumbent, which ranks with a global `atomicAdd`.
    """
    offs_p = tl.arange(0, BLOCK_P)
    jj = tl.arange(0, BLOCK_N)
    vj = jj < NUMEL
    offs_c = tl.arange(0, BLOCK_NK)

    # ---- phase 0: the two padded outputs.  `sorted_ids` positions no pair
    # lands on must read back as `numel` (the Marlin GEMM treats that as
    # "skip"), and blocks past the last live one must read back as -1.
    for p0 in tl.static_range(0, MNP, BLOCK_P):
        p = p0 + offs_p
        tl.store(sorted_ids_ptr + p, tl.full([BLOCK_P], NUMEL, tl.int32),
                 mask=p < MNP)
    if PAIRWISE:
        for b0 in tl.static_range(0, NBLK, BLOCK_P):
            b = b0 + offs_p
            tl.store(expert_ids_ptr + b, tl.full([BLOCK_P], -1, tl.int32),
                     mask=b < NBLK)

    tl.debug_barrier()

    if PAIRWISE:
        # ---- pass 1: per-pair count and rank, rows full / columns chunked.
        ej = tl.load(topk_ids_ptr + jj, mask=vj, other=0)
        cnt = tl.zeros([BLOCK_N], tl.int32)
        rnk = tl.zeros([BLOCK_N], tl.int32)
        for c0 in tl.static_range(0, NUMEL, BLOCK_NK):
            c = c0 + offs_c
            vc = c < NUMEL
            ec = tl.load(topk_ids_ptr + c, mask=vc, other=0)
            eq = (ej[:, None] == ec[None, :]) & vc[None, :] & vj[:, None]
            cnt += tl.sum(eq.to(tl.int32), axis=1)
            rnk += tl.sum((eq & (c[None, :] < jj[:, None])).to(tl.int32), axis=1)

        padded = tl.where(vj, ((cnt + BS - 1) // BS) * BS, 0)
        first = vj & (rnk == 0)
        tl.store(ntpp_ptr, tl.sum(tl.where(first, padded, 0)))

        # ---- pass 2: position each pair and lay out `expert_ids`, rows
        # chunked / columns full, so the full-width `first`/`padded` vectors
        # from pass 1 are what the reduction runs over.
        for r0 in tl.static_range(0, NUMEL, BLOCK_NK):
            r = r0 + offs_c
            vr = r < NUMEL
            er = tl.load(topk_ids_ptr + r, mask=vr, other=0)

            excl = tl.sum(tl.where(first[None, :] & (ej[None, :] < er[:, None]),
                                   padded[None, :], 0), axis=1)
            eqr = (ej[None, :] == er[:, None]) & vj[None, :]
            rank_r = tl.sum((eqr & (jj[None, :] < r[:, None])).to(tl.int32),
                            axis=1)
            tl.store(sorted_ids_ptr + excl + rank_r, r.to(tl.int32), mask=vr)

            # One expert id per block of this expert's own padded region,
            # written by its first-occurrence pair.  At most cdiv(NUMEL, BS)
            # blocks can belong to one expert.
            cnt_r = tl.sum(eqr.to(tl.int32), axis=1)
            live = vr & (rank_r == 0)
            nblk_r = (cnt_r + BS - 1) // BS
            b0 = excl // BS
            for j2 in tl.static_range(0, (NUMEL + BS - 1) // BS):
                tl.store(expert_ids_ptr + b0 + j2, er.to(tl.int32),
                         mask=live & (j2 < nblk_r))
    else:
        # ---- the large-NUMEL body: an E-wide histogram again (above
        # PAIRWISE_MAX the pairwise full-width vectors stop fitting in one
        # CTA's registers), but built with integer ATOMICS and with each pair's
        # block offset GATHERED from scratch.  Both phases are device functions
        # so that the SPLIT path can run them as two multi-CTA kernels; see
        # `_align_meta`, `_align_scatter` and `SPLIT_FROM`.
        _align_meta(topk_ids_ptr, expert_ids_ptr, ntpp_ptr, cnt_ptr, excl_ptr,
                    E, BS, NBLK, NUMEL, BLOCK_E, BLOCK_NK, BLOCK_P)
        tl.debug_barrier()
        for r0 in tl.static_range(0, NUMEL, BLOCK_NK):
            _align_scatter(topk_ids_ptr, sorted_ids_ptr, excl_ptr, r0,
                           NUMEL, BLOCK_N, BLOCK_NK)


@triton.jit
def _fused_route_align_kernel(
    logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr, cnt_ptr, excl_ptr,
    stride_lm, stride_le, routed_scaling_factor,
    M: tl.constexpr, E: tl.constexpr, BS: tl.constexpr,
    MNP: tl.constexpr, NBLK: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_NK: tl.constexpr, BLOCK_P: tl.constexpr,
    PAIRWISE: tl.constexpr, EA: tl.constexpr, EB: tl.constexpr,
    BITONIC: tl.constexpr, RENORM: tl.constexpr,
):
    """Routing AND alignment in one CTA.  grid must be (1,).  Small M only."""
    for m0 in tl.static_range(0, M, BLOCK_M):
        _route_tile(logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr, m0,
                    stride_lm, stride_le, routed_scaling_factor,
                    M, E, TOPK, BLOCK_K, BLOCK_M, EA, EB, BITONIC, RENORM)
    _align_body(topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
                cnt_ptr, excl_ptr,
                E, BS, MNP, NBLK, M * TOPK, BLOCK_E, BLOCK_N, BLOCK_NK,
                BLOCK_P, PAIRWISE)


@triton.jit
def _route_kernel(logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr,
                  stride_lm, stride_le, routed_scaling_factor,
                  M: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                  BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
                  EA: tl.constexpr, EB: tl.constexpr, BITONIC: tl.constexpr,
                  RENORM: tl.constexpr):
    """Routing only, one CTA per BLOCK_M tokens.  The large-M path."""
    _route_tile(logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr,
                tl.program_id(0) * BLOCK_M, stride_lm, stride_le,
                routed_scaling_factor, M, E, TOPK, BLOCK_K, BLOCK_M, EA, EB,
                BITONIC, RENORM)


@triton.jit
def _align_kernel(topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
                  cnt_ptr, excl_ptr,
                  E: tl.constexpr, BS: tl.constexpr, MNP: tl.constexpr,
                  NBLK: tl.constexpr, NUMEL: tl.constexpr,
                  BLOCK_E: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_NK: tl.constexpr, BLOCK_P: tl.constexpr,
                  PAIRWISE: tl.constexpr):
    """Alignment only.  grid must be (1,).  The large-M path."""
    _align_body(topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
                cnt_ptr, excl_ptr,
                E, BS, MNP, NBLK, NUMEL, BLOCK_E, BLOCK_N, BLOCK_NK, BLOCK_P,
                PAIRWISE)


@triton.jit
def _align_prep_kernel(topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
                       cnt_ptr, excl_ptr,
                       E: tl.constexpr, BS: tl.constexpr, MNP: tl.constexpr,
                       NBLK: tl.constexpr, NUMEL: tl.constexpr,
                       BLOCK_E: tl.constexpr, BLOCK_NK: tl.constexpr,
                       BLOCK_P: tl.constexpr):
    """The SPLIT path's first align kernel.  grid (cdiv(MNP, BLOCK_P),).

    Every CTA fills its own BLOCK_P slice of the `sorted_ids` sentinel - at
    M=32 that is 12288 positions, which one CTA was serialising - and CTA 0
    additionally does the whole meta phase, which is a few hundred atomics and
    one E-wide cumsum and does not parallelise usefully.
    """
    pid = tl.program_id(0)
    p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    tl.store(sorted_ids_ptr + p, tl.full([BLOCK_P], NUMEL, tl.int32),
             mask=p < MNP)
    if pid == 0:
        _align_meta(topk_ids_ptr, expert_ids_ptr, ntpp_ptr, cnt_ptr, excl_ptr,
                    E, BS, NBLK, NUMEL, BLOCK_E, BLOCK_NK, BLOCK_P)


@triton.jit
def _align_scatter_kernel(topk_ids_ptr, sorted_ids_ptr, excl_ptr,
                          NUMEL: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_R: tl.constexpr):
    """The SPLIT path's second align kernel.  grid (cdiv(NUMEL, BLOCK_R),).

    The chunks are independent, so this is the phase the single-CTA body was
    serialising: 22.4 us of the family's 46.2 at M=32 by ablation before the
    atomic rewrite, and still the dominant align phase after it.
    """
    _align_scatter(topk_ids_ptr, sorted_ids_ptr, excl_ptr,
                   tl.program_id(0) * BLOCK_R, NUMEL, BLOCK_N, BLOCK_R)


@triton.jit
def _moe_sum_kernel(out_ptr, in_ptr, H, s_tok, s_topk, s_h,
                    TOPK: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[t, :] = sum_k in[t, k, :], fp32 accumulate in slot order.

    Slot order and fp32 accumulation are exactly what `moe_sum_vec_kernel`
    does, so this is bitwise identical to the incumbent, not merely close.
    """
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h < H
    base = in_ptr + t * s_tok + h * s_h
    acc = tl.zeros([BLOCK_H], tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(base + k * s_topk, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * H + h, acc.to(out_ptr.dtype.element_ty), mask=mask)


# --- public API -------------------------------------------------------------
_SCRATCH: dict[tuple, tuple] = {}


def _scratch(device, num_experts):
    """(counts, offsets) scratch for the large-NUMEL align body.

    Allocated once per (device, E) and never reallocated, so nothing allocates
    on the timed path or during a graph capture.  `counts` is left at zero by
    every launch that uses it, which is what makes a replay correct.
    """
    key = (torch.device(device).index or 0, int(num_experts))
    if key not in _SCRATCH:
        dev = torch.device(device)
        _SCRATCH[key] = (torch.zeros(num_experts, dtype=torch.int32, device=dev),
                         torch.zeros(num_experts, dtype=torch.int32, device=dev))
    return _SCRATCH[key]


def align_sizes(numel, num_experts, block_size):
    """`moe_align_block_size`'s output sizes (its host wrapper, verbatim)."""
    mnp = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        mnp = min(numel * block_size, mnp)
    return mnp, -(-mnp // block_size)


def launches(M, topk=8):
    """In-graph kernel launches this candidate costs at this M (bench uses it)."""
    if M <= FUSE_MAX:
        return 1
    return 3 if M * topk >= SPLIT_FROM else 2


def fused_route_align(logits, bias, topk=8, block_size=48, num_experts=288,
                      renormalize=True, routed_scaling_factor=2.5, out=None):
    """Routing + block alignment.

    logits [M, E] float32 (any row stride), bias [E] float32.
    Returns (topk_weights [M, topk] f32, topk_ids [M, topk] i32,
             sorted_ids [mnp] i32, expert_ids [nblk] i32,
             num_tokens_post_pad [1] i32) with exactly the shapes and
    semantics of `fused_grouped_topk` followed by `moe_align_block_size`.
    `out` may supply that 5-tuple to avoid allocating.
    """
    M, E = logits.shape
    assert E == num_experts, f"logits has {E} experts, expected {num_experts}"
    assert topk <= num_experts
    dev = logits.device
    numel = M * topk
    mnp, nblk = align_sizes(numel, num_experts, block_size)

    if out is None:
        topk_w = torch.empty((M, topk), dtype=torch.float32, device=dev)
        topk_ids = torch.empty((M, topk), dtype=torch.int32, device=dev)
        sorted_ids = torch.empty((mnp,), dtype=torch.int32, device=dev)
        expert_ids = torch.empty((nblk,), dtype=torch.int32, device=dev)
        ntpp = torch.empty((1,), dtype=torch.int32, device=dev)
    else:
        topk_w, topk_ids, sorted_ids, expert_ids, ntpp = out

    fused = M <= FUSE_MAX
    bm, be, bk, bn, bnk, bp, pw, sp, ea, eb, bit, nw = _config(M, topk, num_experts, block_size,
                                              mnp, nblk, fused)
    if fused:
        _fused_route_align_kernel[(1,)](
            logits, bias, topk_w, topk_ids, sorted_ids, expert_ids, ntpp,
            *_scratch(dev, num_experts),
            logits.stride(0), logits.stride(1), float(routed_scaling_factor),
            M=M, E=num_experts, BS=block_size, MNP=mnp, NBLK=nblk, TOPK=topk,
            BLOCK_K=bk, BLOCK_M=bm, BLOCK_E=be, BLOCK_N=bn, BLOCK_NK=bnk,
            BLOCK_P=bp, PAIRWISE=pw, EA=ea, EB=eb, BITONIC=bit,
            RENORM=bool(renormalize), num_warps=nw, num_stages=1)
    else:
        _route_kernel[(triton.cdiv(M, bm),)](
            logits, bias, topk_w, topk_ids,
            logits.stride(0), logits.stride(1), float(routed_scaling_factor),
            M=M, E=num_experts, TOPK=topk, BLOCK_K=bk, BLOCK_M=bm,
            EA=ea, EB=eb, BITONIC=bit,
            RENORM=bool(renormalize), num_warps=nw, num_stages=1)
        if sp:
            cnt_s, excl_s = _scratch(dev, num_experts)
            _align_prep_kernel[(triton.cdiv(mnp, bp),)](
                topk_ids, sorted_ids, expert_ids, ntpp, cnt_s, excl_s,
                E=num_experts, BS=block_size, MNP=mnp, NBLK=nblk, NUMEL=numel,
                BLOCK_E=be, BLOCK_NK=bnk, BLOCK_P=bp,
                num_warps=nw, num_stages=1)
            _align_scatter_kernel[(triton.cdiv(numel, BLOCK_R),)](
                topk_ids, sorted_ids, excl_s,
                NUMEL=numel, BLOCK_N=bn, BLOCK_R=BLOCK_R,
                num_warps=nw, num_stages=1)
        else:
            _align_kernel[(1,)](
                topk_ids, sorted_ids, expert_ids, ntpp,
                *_scratch(dev, num_experts),
                E=num_experts, BS=block_size, MNP=mnp, NBLK=nblk, NUMEL=numel,
                BLOCK_E=be, BLOCK_N=bn, BLOCK_NK=bnk, BLOCK_P=bp, PAIRWISE=pw,
                num_warps=nw, num_stages=1)
    return topk_w, topk_ids, sorted_ids, expert_ids, ntpp


def moe_sum(inp, out=None):
    """[M, topk, H] -> [M, H], fp32 accumulation in slot order, bf16 in/out."""
    M, topk, H = inp.shape
    if out is None:
        out = torch.empty((M, H), dtype=inp.dtype, device=inp.device)
    block_h = 256 if H >= 256 else max(16, _next_pow2(H))
    _moe_sum_kernel[(M, triton.cdiv(H, block_h))](
        out, inp, H, inp.stride(0), inp.stride(1), inp.stride(2),
        TOPK=topk, BLOCK_H=block_h, num_warps=4, num_stages=1,
    )
    return out


def warmup(ms, block_sizes=(48,), topk=8, num_experts=288, hidden=4096,
           device=None):
    """Compile every variant the caller will use.

    MUST be called before CUDA graph capture: Triton compiles on first launch,
    which may not happen during capture.  Every shape here is a `tl.constexpr`,
    so there is one variant per (M, topk, block_size, num_experts).
    """
    dev = device or torch.device("cuda:0")
    for M in sorted({int(m) for m in ms}):
        logits = torch.zeros((M, num_experts), dtype=torch.float32, device=dev)
        bias = torch.zeros((num_experts,), dtype=torch.float32, device=dev)
        for bs in sorted({int(b) for b in block_sizes}):
            fused_route_align(logits, bias, topk=topk, block_size=bs,
                              num_experts=num_experts)
        moe_sum(torch.zeros((M, topk, hidden), dtype=torch.bfloat16,
                            device=dev))
    torch.cuda.synchronize()
