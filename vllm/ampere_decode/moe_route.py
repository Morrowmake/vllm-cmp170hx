# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 MoE router: router GEMV + sigmoid/bias top-k + block alignment.

Replaces, per MoE layer, GateLinear's cuBLAS bf16 -> fp32 GEMM (a split-K
kernel plus a separate reduce on this part), the grouped top-k and
`moe_align_block_size`. Keep this file in sync with the standalone source.

THE OP (see reference/moe_route_ref.py for the exact semantics)

    x [M, 4096] bf16, W [288, 4096] bf16, bias [288] fp32
      -> logits [M, 288] fp32, topk_weights [M, 8] fp32, topk_ids [M, 8] i32,
         sorted_ids [mnp] i32, expert_ids [nblk] i32, num_tokens_post_pad [1] i32

ONE LAUNCH FOR 2 <= M <= 64, TWO OTHERWISE

  `_moe_route_kernel`  (2 <= M <= 64) the tensor-core GEMV CTAs
                   (`_gemv_tc_part`: split-K partials, release-counted) plus
                   one routing CTA per row (`_route_align_rows`), which waits
                   for every partial, sums its row in fixed split order,
                   routes it, and the last of which builds the alignment from
                   per-(row, expert) bitmask columns.
  M = 1 (or > 64)  `_gemv_kernel` (split-free FMA GEMV, faster at M = 1) then
                   `_fused_route_align_kernel` (ONE CTA: packed int64 key,
                   bitonic `tl.topk`, pairwise / histogram align).  In the older
                   docstrings "the incumbent" means upstream's CUDA kernels
                   (single_group_topk / moe_align / count_and_sort).

Whole op, 42-layer weight rotation, graph replay, routing variants (us):
    M                       1     2     4     8    16    32
    single-CTA route     9.87 11.48 13.02  20.8  34.2  68.3
    one CTA per row     11.52 11.99 11.82 12.37 12.93 14.20

CONSTRAINTS honoured here
  * NO PRECISION REDUCTION: bf16 x bf16 products are exact in fp32 and are
    accumulated in fp32; scores use the incumbent's exact 0.5*tanh(0.5x)+0.5
    form (`tl.sigmoid` rounds differently and can flip a decision); fp32
    weights, int32 ids.
  * DETERMINISTIC: fixed-order reductions only. Integer +1 / OR atomics
    commute, so counts and bitmasks are reproducible; ranks are explicit
    counts, never an atomic's return value.
  * CUDA-GRAPH CAPTURABLE: every shape is a constexpr compiled by `warmup()`,
    the align scratch is allocated once per (device, E) and left zeroed by
    every launch; no host sync, no `.item()`, no allocation when `out=` is
    given.
  * sm_80: no TMA, no wgmma, no fp8, no warp specialisation.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, out_ptr, stride_xm, stride_wn, stride_om,
                 M: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr):
    """logits[m, n] = sum_k x[m, k] * W[n, k], fp32, one fixed order.

    grid (cdiv(E, BLOCK_N), cdiv(M, BLOCK_M)). The K loop runs in order and
    each chunk is reduced by the same `tl.sum` tree, so the result is
    bitwise reproducible and independent of the grid.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < E
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        if EVEN_K:
            mx = mask_m[:, None]
            mw = mask_n[:, None]
        else:
            mx = mask_m[:, None] & (kk < K)[None, :]
            mw = mask_n[:, None] & (kk < K)[None, :]
        xk = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
                     mask=mx, other=0.0).to(tl.float32)
        wk = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :],
                     mask=mw, other=0.0).to(tl.float32)
        acc += tl.sum(xk[:, None, :] * wk[None, :, :], axis=2)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc,
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _gemv_tc_part(x_ptr, w_ptr, part_ptr, done_ptr,
                  pid_n, pid_s, stride_xm, stride_wn,
                  M: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
                  BLOCK_K: tl.constexpr, SPLIT: tl.constexpr):
    """One split-K partial of logits = x @ W.T on bf16 TENSOR CORES; this CTA
    is (expert tile pid_n, K split pid_s).

    Tokens sit on the MMA M dimension (padded to >= 16), BLOCK_E experts on N.
    CTA (n, s) owns the contiguous K range [s*K/SPLIT, (s+1)*K/SPLIT) and walks
    it in order through `tl.dot`, whose bf16 x bf16 products are exact in the
    fp32 accumulator.  It writes its fp32 partial to slab s of `part_ptr`
    ([SPLIT, M, E]) and publishes it with a release increment of `done_ptr`.
    NOTHING reduces here: the routing CTA of each row sums its row's SPLIT
    partials ALWAYS in order 0..SPLIT-1 (`_moe_route_kernel`), so the logits
    are bitwise reproducible and a per-tile last-arriving reduction's two
    dependent round trips stay off the critical path.
    """
    KS: tl.constexpr = K // SPLIT
    offs_m = tl.arange(0, BLOCK_M)
    offs_e = pid_n * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_e = offs_e < E
    x_m = tl.where(mask_m, offs_m, 0)
    w_e = tl.where(mask_e, offs_e, 0)
    k0 = pid_s * KS
    x_ptrs = x_ptr + x_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = w_ptr + w_e[:, None] * stride_wn + (k0 + offs_k)[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_E], tl.float32)
    for i in range(KS // BLOCK_K):
        xk = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        wk = tl.load(w_ptrs)
        # Every BLOCK_K chunk gets a FRESH accumulator and is added to `acc`
        # with an opaque round-to-nearest fp32 add.  sm_80 tensor cores
        # truncate inside the MMA accumulation, so one long chain drifts
        # (1024-term chain: 6.7e-6 max logit error vs cuBLAS's 2.1e-6); an
        # 8-step chain per chunk is 0.9e-6.  The asm add is what stops Triton's
        # combine pass from folding `acc + dot(a, b, 0)` back into the chain.
        d = tl.dot(xk, tl.trans(wk), out_dtype=tl.float32)
        acc = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r",
                                        [acc, d], dtype=tl.float32,
                                        is_pure=True, pack=1)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    p_off = offs_m[:, None] * E + offs_e[None, :]
    tl.store(part_ptr + pid_s * (M * E) + p_off, acc,
             mask=mask_m[:, None] & mask_e[None, :], cache_modifier=".cg")
    tl.debug_barrier()
    tl.atomic_add(done_ptr, 1, sem="release", scope="gpu")


_TC_MAX_SPLIT = 16
_TC_MAX_BM = 64
_TC_SCRATCH: dict = {}


def _tc_scratch(device, num_experts):
    """[_TC_MAX_SPLIT, _TC_MAX_BM, E] fp32 partials for `_gemv_tc_part`,
    allocated once per (device, E), so nothing allocates after warmup."""
    key = (torch.device(device).index or 0, int(num_experts))
    if key not in _TC_SCRATCH:
        _TC_SCRATCH[key] = torch.empty(
            _TC_MAX_SPLIT * _TC_MAX_BM * num_experts, dtype=torch.float32,
            device=torch.device(device))
    return _TC_SCRATCH[key]


def _gemv_tc_config(M, K):
    """(BLOCK_M, BLOCK_E, BLOCK_K, SPLIT, num_warps, num_stages) for
    `_moe_route_kernel`'s GEMV CTAs, or None where the FMA GEMV is used.

    Whole op, `_moe_route_kernel` (42-layer weight
    rotation + graph replay; be x split x bk x warps x stages, bm = pad(M)):

        be, bk, split, nw, ns     M=4     M=8    M=16    M=32   us
        32, 128,  8, 4, 3        10.3    10.4    10.7    11.8   <- all M
        32, 128, 16, 4, 3        11.1    11.3    11.6    12.6
        16, 128,  8, 4, 3        10.8    10.9    11.3    12.5
        64, 128,  8, 4, 3        10.4       -    10.8    12.0
        32, 128,  4, 4, 4        10.5       -    10.9    12.0
        32, 128,  8, 8, 3        11.3    11.4    11.7    12.6

    Four warps because the same CTAs' routing half wants them.  Precision does
    not constrain SPLIT: every 128-wide chunk is its own MMA chain (see
    `_gemv_tc_part`).  Whole op at M = 2 / 1: one launch 9.81 / 10.18
    us against the FMA GEMV + single-CTA routing's 11.42 / 9.87, so only M = 1
    keeps the two-launch path.
    """
    if M <= 1:
        return None
    bm = max(16, _next_pow2(M))
    if bm > _TC_MAX_BM:
        return None
    cfg = (bm, 32, 128, 8, 4, 3)
    if K % (cfg[2] * cfg[3]):
        return None
    return cfg


def _gemv_config(M, K):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps), from a sweep on this part
    (GPU graph replay, 42-layer weight rotation, bm x bn x bk x warps):

        M    best config       us      first heuristic      us
        1    1, 2, 2048, 4     6.80    1, 2, 1024, 4        9.13
        2    2, 2, 2048, 4     7.67    2, 2, 1024, 4        9.59
        4    4, 2, 2048, 4     9.87    4, 2, 1024, 4       12.11
        8    2, 2, 2048, 2    11.35    8, 4,  256, 4       48.42
       16    4, 2, 2048, 4    16.66   16, 4,  128, 4      ~130
       32    4, 4, 1024, 2    26.61   32, 4,   64, 4      ~240

    The [BM, BN, BK] product tile collapses once BLOCK_M goes past 4. Even the
    best entry is FMA-bound and slower than cuBLAS from M = 4 up (cuBLAS 7.9 us
    at M=4, 8.3 at M=16, split-K + reduce included): this structure is the
    starting point, not the answer."""
    table = {1: (1, 2, 2048, 4), 2: (2, 2, 2048, 4), 4: (4, 2, 2048, 4),
             8: (2, 2, 2048, 2), 16: (4, 2, 2048, 4), 32: (4, 4, 1024, 2)}
    p2 = _next_pow2(M)
    bm, bn, bk, nw = table.get(p2, table[32] if p2 > 32 else table[4])
    bm = min(bm, p2)
    bk = min(bk, _next_pow2(K))
    return bm, bn, bk, nw


def _next_pow2(n):
    return 1 << max(0, (int(n) - 1)).bit_length()


# The align body has two implementations of the same semantics and picks between
# them on the pair count (see `_align_body`).  Up to this many (token, slot)
# pairs the fully pairwise O(NUMEL^2) body wins; above it the atomic-histogram
# body does, because the pairwise body's full-width [BLOCK_N] vectors and its
# [BLOCK_NK, BLOCK_N] tiles stop fitting in one CTA's registers.
# Measured on this part, whole op, graph replay:
#
#   M                   1     2     4     8    16    32
#   pairwise         1.34  1.50  1.28  0.87  0.53  0.21
#   atomic histogram 1.33  1.35  1.18  0.98  0.85  0.55
#   one-hot histogram (removed)  1.57  1.28  1.02  0.71  0.51  0.30
#
# so the crossover sits between 32 and 64 pairs: M=8 (64 pairs) prefers the
# atomic body, M=4 (32 pairs) the pairwise one.
PAIRWISE_MAX = 32


def _config(M, topk, num_experts, block_size, mnp, nblk):
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
    # `tl.topk` needs a power-of-two k no larger than either tile.
    bitonic = block_k >= 4 and e_a >= block_k and (e_b == 0 or e_b >= block_k)
    block_p = min(2048, _next_pow2(max(mnp, nblk)))
    num_warps = 8
    return (block_m, block_e, block_k, block_n, block_nk, block_p, pairwise,
            e_a, e_b, bitonic, num_warps)


@triton.jit
def _route_load(logits_ptr, bias_ptr, offs_m, mask_m, e0,
                stride_lm, stride_le, E: tl.constexpr, EW: tl.constexpr):
    """The [BLOCK_M, EW] logits tile and the [EW] bias of experts
    [e0, e0 + EW)."""
    offs_e = e0 + tl.arange(0, EW)
    valid_e = offs_e < E
    bias = tl.load(bias_ptr + offs_e, mask=valid_e, other=0.0).to(tl.float32)
    x = tl.load(logits_ptr + offs_m[:, None] * stride_lm
                + offs_e[None, :] * stride_le,
                mask=mask_m[:, None] & valid_e[None, :],
                other=0.0).to(tl.float32)
    return x, bias


@triton.jit
def _route_keys(x, bias, e0, E: tl.constexpr, EW: tl.constexpr):
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
    score = 0.5 * libdevice.tanh(0.5 * x) + 0.5
    key = tl.where(valid_e[None, :], score + bias[None, :], float("-inf"))
    b32 = key.to(tl.int32, bitcast=True)
    packed = (((b32 ^ ((b32 >> 31) & 0x7FFFFFFF)).to(tl.int64) << 32)
              | (E - 1 - offs_e).to(tl.int64)[None, :])
    return packed


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
    #
    # BOTH TILES ARE LOADED BEFORE EITHER IS COMPUTED: with load+compute per
    # tile, the 32-wide tile's loads were issued only after tile A's branchy
    # libdevice `tanh`, a serialized L2 round trip on every routing row.
    xa, ba = _route_load(logits_ptr, bias_ptr, offs_m, mask_m, 0,
                         stride_lm, stride_le, E, EA)
    if EB > 0:
        xb, bb = _route_load(logits_ptr, bias_ptr, offs_m, mask_m, EA,
                             stride_lm, stride_le, E, EB)
    pa = _route_keys(xa, ba, 0, E, EA)
    if EB > 0:
        pb = _route_keys(xb, bb, EA, E, EB)

    if BITONIC:
        # ZERO CROSS-LANE REDUCTIONS OVER THE EXPERT AXIS.  The eight-pass form
        # below costs eight dependent `tl.max` trees over the full 288 lanes,
        # plus eight full-width mask-out passes, plus eight dependent
        # single-token gathers - and measurement showed ~0.64 us of each pass's
        # 0.8 us as fixed tree/barrier cost, i.e. the thing that does not go
        # away by narrowing the selection set.
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
    return sel_i, keep


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
        # block offset GATHERED from scratch.  Only the fused path's fallback
        # beyond 64 rows (never a decode batch) reaches it.
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
def _route_align_rows(
    pid, logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr, col_ptr, arrive_ptr,
    stride_lm, stride_le, routed_scaling_factor,
    M: tl.constexpr, E: tl.constexpr, BS: tl.constexpr,
    MNP: tl.constexpr, NBLK: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr, FILL_P: tl.constexpr,
    EA: tl.constexpr, EB: tl.constexpr, BITONIC: tl.constexpr,
    RENORM: tl.constexpr, G: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """Routing CTA `pid` of G = cdiv(M, BLOCK_M), and the alignment in
    whichever of them arrives last.

    ROW BITMASK COLUMNS.  A row selects each expert at most once, so the whole
    alignment follows from one bit per (row, expert): `col[g, e]` has bit
    `m % 32` set iff row `m = 32 g + bit` selected expert e.  Every CTA routes
    its BLOCK_M rows and sets its bits with integer `atomic_or` (OR commutes,
    so the words are reproducible), and fills its FILL_P slice of the two
    sentinel outputs.  Then one acq_rel arrival; the CTA that arrives last does
    the serial part on L2-hot data, all of it O(E + pairs):

        cnt[e]      = sum_g popc(col[g, e])
        excl[e]     = exclusive prefix of the padded counts
        rank(m, e)  = #{m' < m : row m' selected e}
                    = sum_{g < m/32} popc(col[g, e]) + popc(col[m/32, e] & low)
        sorted_ids[excl[e] + rank(m, e)] = m * TOPK + slot

    The rank is an explicit count, and nothing written depends on which CTA
    arrives last, so the outputs are bitwise deterministic.  (The pairwise
    rank this replaces was O(pairs^2) in one CTA: 23.6 us at M=32.)  The last
    CTA leaves `col` and the arrival counter at zero.
    """
    NUMEL: tl.constexpr = M * TOPK
    NG: tl.constexpr = (M + 31) // 32
    m0 = pid * BLOCK_M
    sel_i, keep = _route_tile(logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr,
                              m0, stride_lm, stride_le,
                              routed_scaling_factor, M, E, TOPK, BLOCK_K,
                              BLOCK_M, EA, EB, BITONIC, RENORM)
    p = pid * FILL_P + tl.arange(0, FILL_P)
    tl.store(sorted_ids_ptr + p, tl.full([FILL_P], NUMEL, tl.int32),
             mask=p < MNP)
    tl.store(expert_ids_ptr + p, tl.full([FILL_P], -1, tl.int32),
             mask=p < NBLK)
    rows = m0 + tl.arange(0, BLOCK_M)
    bits = (tl.full([BLOCK_M], 1, tl.int32) << (rows % 32))[:, None] \
        + tl.zeros_like(sel_i)
    tl.atomic_or(col_ptr + (rows // 32)[:, None] * E + sel_i, bits, mask=keep,
                 sem="relaxed", scope="gpu")
    tl.debug_barrier()
    arrived = tl.atomic_add(arrive_ptr, 1, sem="acq_rel", scope="gpu")
    if arrived == G - 1:
        # ONE dependent L2 round trip: the pair ids and the column words are
        # loaded together, and every per-pair lookup (col[., e], excl[e]) is a
        # `tl.gather` from registers instead of a scratch store + barrier +
        # global gather (which cost two more serialized round trips).
        offs_e = tl.arange(0, BLOCK_E)
        valid_e = offs_e < E
        i = tl.arange(0, BLOCK_N)
        vi = i < NUMEL
        e = tl.load(topk_ids_ptr + i, mask=vi, other=0, cache_modifier=".cg")
        m = i // TOPK
        low = (tl.full([BLOCK_N], 1, tl.int32) << (m % 32)) - 1
        cnt = tl.zeros([BLOCK_E], tl.int32)
        rank = tl.zeros([BLOCK_N], tl.int32)
        for g in tl.static_range(NG):
            w = tl.load(col_ptr + g * E + offs_e, mask=valid_e, other=0,
                        cache_modifier=".cg")
            cnt += libdevice.popc(w)
            wg = tl.gather(w, e, 0)
            wg = tl.where(g < m // 32, wg, tl.where(g == m // 32, wg & low, 0))
            rank += libdevice.popc(wg)
        padded = tl.where(valid_e, ((cnt + BS - 1) // BS) * BS, 0)
        excl = tl.cumsum(padded, axis=0) - padded
        tl.store(ntpp_ptr, tl.sum(padded))
        # An expert appears at most once per row, so at most cdiv(M, BS)
        # blocks: one 2-D masked store instead of a loop of layout changes.
        nblk_e = padded // BS
        blk0 = excl // BS
        jj = tl.arange(0, BLOCK_J)
        tl.store(expert_ids_ptr + blk0[:, None] + jj[None, :],
                 offs_e.to(tl.int32)[:, None]
                 + tl.zeros([BLOCK_E, BLOCK_J], tl.int32),
                 mask=valid_e[:, None] & (jj[None, :] < nblk_e[:, None]))
        ex = tl.gather(excl, e, 0)
        tl.store(sorted_ids_ptr + ex + rank, i.to(tl.int32), mask=vi)
        tl.debug_barrier()
        for g in tl.static_range(NG):
            tl.store(col_ptr + g * E + offs_e, tl.zeros([BLOCK_E], tl.int32),
                     mask=valid_e)
        # arrive_ptr[1] is the GEMV's partial-done counter in
        # `_moe_route_kernel`; every routing CTA is past its wait once all of
        # them have arrived.  Both resets are PLAIN stores: nothing in this
        # launch reads either counter again and the next launch is stream-
        # ordered after this one, so a release (a gpu-scope fence on the
        # critical path, 0.55 us measured) orders nothing.
        tl.store(arrive_ptr + 1, 0)
        tl.store(arrive_ptr, 0)


@triton.jit
def _ld_acquire(ptr):
    """`ld.acquire.gpu` of one int32, in every thread that executes it."""
    return tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l", [ptr],
        dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _moe_route_kernel(
    x_ptr, w_ptr, logits_ptr, part_ptr, bias_ptr,
    topk_w_ptr, topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    col_ptr, arrive_ptr,
    stride_xm, stride_wn, stride_lm, routed_scaling_factor,
    M: tl.constexpr, K: tl.constexpr, E: tl.constexpr, BS: tl.constexpr,
    MNP: tl.constexpr, NBLK: tl.constexpr, TOPK: tl.constexpr,
    GBM: tl.constexpr, GBE: tl.constexpr, GBK: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_N: tl.constexpr,
    FILL_P: tl.constexpr, EA: tl.constexpr, EB: tl.constexpr,
    BITONIC: tl.constexpr, RENORM: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """THE WHOLE OP IN ONE LAUNCH.  grid (cdiv(E, GBE) * SPLIT + M,).

    CTAs [0, NGEMV = NT * SPLIT) are the tensor-core GEMV (`_gemv_tc_part`),
    each publishing one split-K partial with a release increment of
    `arrive_ptr[1]`.  The last M CTAs are the routing rows: each waits until
    all NGEMV partials are published, sums ITS row's partials in the fixed
    order 0..SPLIT-1 into `logits`, then runs `_route_align_rows`.

    WHY THE WAIT CANNOT DEADLOCK: only the M <= 64 routing CTAs ever wait, and
    they wait only on GEMV CTAs, which never wait on anything.  The waiting
    CTAs occupy at most M block slots of a machine with >= 70 SMs x several
    slots each, so the GEMV CTAs always find room to run to completion,
    whatever order the hardware schedules the grid in.  (Polling with a
    `nanosleep` backoff of 50-1000 ns measured 0.2-0.6 us slower.)
    """
    NT: tl.constexpr = (E + GBE - 1) // GBE
    NGEMV: tl.constexpr = NT * SPLIT
    pid = tl.program_id(0)
    if pid < NGEMV:
        _gemv_tc_part(x_ptr, w_ptr, part_ptr, arrive_ptr + 1,
                      pid % NT, pid // NT, stride_xm, stride_wn,
                      M, K, E, GBM, GBE, GBK, SPLIT)
    else:
        row = pid - NGEMV
        # Every thread polls with `ld.acquire.gpu`, so each of them has
        # acquired the GEMV CTAs' release increments before it reads a
        # partial.  (A trailing `atomic_add(ptr, 0, sem="acquire")` is NOT an
        # acquire here: Triton drops an atomic whose result is unused.)
        done = _ld_acquire(arrive_ptr + 1)
        while done < NGEMV:
            done = _ld_acquire(arrive_ptr + 1)
        tl.debug_barrier()
        offs_e = tl.arange(0, BLOCK_E)
        ve = offs_e < E
        tot = tl.zeros([BLOCK_E], tl.float32)
        for s in tl.static_range(SPLIT):
            tot += tl.load(part_ptr + s * (M * E) + row * E + offs_e, mask=ve,
                           other=0.0, cache_modifier=".cg")
        tl.store(logits_ptr + row * stride_lm + offs_e, tot, mask=ve)
        tl.debug_barrier()
        _route_align_rows(pid - NGEMV, logits_ptr, bias_ptr, topk_w_ptr,
                          topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr,
                          ntpp_ptr, col_ptr, arrive_ptr,
                          stride_lm, 1, routed_scaling_factor,
                          M, E, BS, MNP, NBLK, TOPK, BLOCK_K, 1, BLOCK_E,
                          BLOCK_N, FILL_P, EA, EB, BITONIC, RENORM, M,
                          BLOCK_J)


# --- public API -------------------------------------------------------------
_SCRATCH: dict = {}


def _scratch(device, num_experts):
    """(bitmask columns / counts, offsets, arrival counter) align scratch.

    Allocated once per (device, E) and never reallocated, so nothing allocates
    on the timed path or during a graph capture. The first and last buffers
    are left at zero by every launch that uses them, which is what makes a
    replay correct (the fused path uses the first E words as its histogram).
    """
    key = (torch.device(device).index or 0, int(num_experts))
    if key not in _SCRATCH:
        dev = torch.device(device)
        _SCRATCH[key] = (
            torch.zeros(_TC_MAX_BM // 32 * num_experts, dtype=torch.int32,
                        device=dev),
            torch.zeros(num_experts, dtype=torch.int32, device=dev),
            torch.zeros(2, dtype=torch.int32, device=dev))
    return _SCRATCH[key]


def align_sizes(numel, num_experts, block_size):
    """`moe_align_block_size`'s output sizes (its host wrapper, verbatim)."""
    mnp = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        mnp = min(numel * block_size, mnp)
    return mnp, -(-mnp // block_size)


def launches(M, topk=8, hidden=4096):
    """In-graph kernel launches this op costs at M rows."""
    return 1 if _gemv_tc_config(M, hidden) is not None else 2


def moe_route(x, weight, bias, topk=8, block_size=8, num_experts=288,
              renormalize=True, routed_scaling_factor=2.5, out=None):
    """Router GEMV + sigmoid/bias top-k + Marlin block alignment.

    x [M, K] bf16 (row stride free, unit column stride), weight [E, K] bf16
    contiguous, bias [E] fp32. Returns
        (logits [M, E] f32, topk_weights [M, topk] f32, topk_ids [M, topk] i32,
         sorted_ids [mnp] i32, expert_ids [nblk] i32, num_tokens_post_pad [1] i32)
    with the semantics of torch.mm(x, W.T, out_dtype=fp32) + fused_grouped_topk
    + moe_align_block_size. `out` may supply that 6-tuple (logits contiguous).
    """
    M, K = x.shape
    E = weight.shape[0]
    assert E == num_experts, f"weight has {E} experts, expected {num_experts}"
    assert weight.shape[1] == K and weight.stride(1) == 1
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert bias.dtype == torch.float32 and bias.shape == (E,)
    assert x.stride(1) == 1, "x needs a unit column stride"
    assert topk <= num_experts
    dev = x.device
    numel = M * topk
    mnp, nblk = align_sizes(numel, num_experts, block_size)

    if out is None:
        logits = torch.empty((M, E), dtype=torch.float32, device=dev)
        topk_w = torch.empty((M, topk), dtype=torch.float32, device=dev)
        topk_ids = torch.empty((M, topk), dtype=torch.int32, device=dev)
        sorted_ids = torch.empty((mnp,), dtype=torch.int32, device=dev)
        expert_ids = torch.empty((nblk,), dtype=torch.int32, device=dev)
        ntpp = torch.empty((1,), dtype=torch.int32, device=dev)
    else:
        logits, topk_w, topk_ids, sorted_ids, expert_ids, ntpp = out

    (bm, be, bk, bn, bnk, bp, pw, ea, eb, bit,
     nw) = _config(M, topk, num_experts, block_size, mnp, nblk)
    s_lm, s_le = logits.stride(0), logits.stride(1)
    col_s, excl_s, arrive_s = _scratch(dev, num_experts)
    tc = _gemv_tc_config(M, K)
    if tc is not None:
        # ONE LAUNCH: tensor-core GEMV CTAs + one routing CTA per row.
        gbm, gbe, gbk, split, gnw, gns = tc
        part = _tc_scratch(dev, E)
        fill = _next_pow2(triton.cdiv(max(mnp, nblk), M))
        _moe_route_kernel[(triton.cdiv(E, gbe) * split + M,)](
            x, weight, logits, part, bias,
            topk_w, topk_ids, sorted_ids, expert_ids, ntpp,
            col_s, arrive_s,
            x.stride(0), weight.stride(0), s_lm, float(routed_scaling_factor),
            M=M, K=K, E=E, BS=block_size, MNP=mnp, NBLK=nblk, TOPK=topk,
            GBM=gbm, GBE=gbe, GBK=gbk, SPLIT=split,
            BLOCK_K=bk, BLOCK_E=be, BLOCK_N=bn, FILL_P=fill,
            EA=ea, EB=eb, BITONIC=bit, RENORM=bool(renormalize),
            BLOCK_J=_next_pow2(triton.cdiv(M, block_size)),
            num_warps=gnw, num_stages=gns)
    else:
        # M = 1 (and M > 64, never a decode batch): the split-free FMA GEMV,
        # then routing + alignment in one single-CTA kernel.
        gbm, gbn, gbk, gnw = _gemv_config(M, K)
        _gemv_kernel[(triton.cdiv(E, gbn), triton.cdiv(M, gbm))](
            x, weight, logits, x.stride(0), weight.stride(0), s_lm,
            M=M, K=K, E=E, BLOCK_M=gbm, BLOCK_N=gbn, BLOCK_K=gbk,
            EVEN_K=(K % gbk == 0), num_warps=gnw, num_stages=1)
        _fused_route_align_kernel[(1,)](
            logits, bias, topk_w, topk_ids, sorted_ids, expert_ids, ntpp,
            col_s, excl_s,
            s_lm, s_le, float(routed_scaling_factor),
            M=M, E=num_experts, BS=block_size, MNP=mnp, NBLK=nblk, TOPK=topk,
            BLOCK_K=bk, BLOCK_M=bm, BLOCK_E=be, BLOCK_N=bn, BLOCK_NK=bnk,
            BLOCK_P=bp, PAIRWISE=pw, EA=ea, EB=eb, BITONIC=bit,
            RENORM=bool(renormalize), num_warps=nw, num_stages=1)
    return logits, topk_w, topk_ids, sorted_ids, expert_ids, ntpp


def warmup(ms, block_sizes=(8,), topk=8, num_experts=288, hidden=4096,
           device=None):
    """Compile every variant the caller will use, and allocate the scratch.

    MUST run before CUDA-graph capture: Triton compiles on first launch, which
    must not happen inside a capture. One variant per (M, topk, block_size, E).
    """
    dev = device or torch.device("cuda:0")
    w = torch.zeros((num_experts, hidden), dtype=torch.bfloat16, device=dev)
    b = torch.zeros((num_experts,), dtype=torch.float32, device=dev)
    for M in sorted({int(m) for m in ms}):
        x = torch.zeros((M, hidden), dtype=torch.bfloat16, device=dev)
        for bs in sorted({int(v) for v in block_sizes}):
            moe_route(x, w, b, topk=topk, block_size=bs,
                      num_experts=num_experts)
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)
