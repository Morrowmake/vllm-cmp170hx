# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored kernel: keep in sync with the standalone kernel source.
# Revalidate changes with the standalone correctness and performance checks.
# Keep changes synchronized: the standalone repo owns the
# correctness gate (validate.py) and the benchmark (bench.py) that qualify this
# file, so a change made here and not there is a change that nothing measures.
# To update, re-copy the file and re-run the standalone gate.
"""Thin-M BF16 GEMM for sm_80, 74-SM schedule for the Qwen3.8 decode shapes.

Served behind ``VLLM_THIN_GEMM_GLUON`` (dispatch: ``gluon.py``). The schedule
is tuned on the 74-SM part: measured schedule rows for the Qwen3.8 TP2 decode
shapes (``_CONFIG_OVERRIDES``, `_heuristic` elsewhere; rows tagged with
WARPS_M run the Gluon kernels `_gluon_gemm_kernel`, `_gluon_sliced_kernel` or
`_gluon_balanced_kernel`), a GEMV for N = 1, a per-N-tile rotation of the K
start for wide tiles (``_rotate_k``; summation order changes,
deterministically), and a split-K arrival that either only releases, with the
last arriver acquiring through the counter reset, or (ACQ_REL rows) is one
acq_rel RMW with a relaxed reset. Every output is bitwise reproducible run to
run.

    thin_gemm(x[M, K] bf16, w[N, K] bf16) -> y[M, N] bf16      == F.linear(x, w)

Keep this kernel in sync with the standalone source.
The contract with the rest of the
tree is only the `thin_gemm` signature and its semantics (BF16 x BF16, FP32
accumulate, BF16 out; no precision reduction of any kind).

Why split-K
-----------
At M <= 32 the op is weight-bandwidth bound: it reads N*K*2 bytes of weight and
does 2*M*N*K flops, so the arithmetic intensity is ~M flop/byte - 4 flop/byte at
M=4 against ~75 flop/byte of bf16 balance on this part.  The only thing that
matters is streaming the weight once at full HBM rate, which needs enough
concurrent CTAs to saturate the memory system.  A plain (M, N) tiling gives
ceil(N/BLOCK_N) CTAs; for the narrow shapes that is 1..8 CTAs on
a 74-SM part.  Splitting K across CTAs multiplies the CTA count without reading
any weight byte twice - each CTA owns a disjoint set of K-blocks.

The partial sums are FP32 and are reduced in the same launch: each CTA stores
its partial and bumps a per-output-tile arrival counter, and the CTA that
arrives last sums the SPLIT_K partials in fixed order 0..SPLIT_K-1 and writes
the output.  No CTA spins, and the counter only elects the reducer (it never
accumulates values), so the result is bitwise deterministic run to run.

Constraints: sm_80 only, Triton 3.7.1 (its Gluon ampere API: async_copy,
mma_v2), no TMA/wgmma, no fp8, must be CUDA-graph capturable (see `warmup()`).
"""

import contextlib

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

# --------------------------------------------------------------------------
# Tuning knobs.  _select_config() below is where the schedule is tuned.
# --------------------------------------------------------------------------

# Target CTA count as a multiple of the SM count.  Measured on this part
# (74 SMs, sm_80): what a weight-streaming schedule wants is *many small* CTAs,
# not few large ones.  A tile that is narrow in N reads exactly the same weight
# bytes as a wide one, gives the scheduler finer-grained work so the tail of the
# grid does not idle SMs, and lets SPLIT_K stay at 1 - which drops the whole
# reduction launch.  1.8 is the smallest target that reproduces the measured
# winner for the wide shapes without over-splitting the narrow ones.
_CTA_PER_SM_TARGET = 1.8

# Never split K so finely that a CTA reads fewer than this many K elements;
# below it the fixed per-CTA cost and the reduction pass dominate.
_MIN_K_PER_SPLIT = 256

# Largest shared-memory allocation a CTA may request (sm_80 allows 163 KiB
# opt-in).  num_stages is trimmed rather than failing to compile at a large M.
_MAX_SMEM = 150 * 1024

# Measured schedules for the Qwen3.8 TP2 rank-0 decode shapes, keyed (N, K, M).
# A 5-tuple (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) runs
# `_thin_gemm_kernel`; a 6-tuple adds WARPS_M and runs `_gluon_gemm_kernel`
# (falling back to the Triton kernel with the first five fields for inputs the
# Gluon kernel does not take); a 7th field 1 selects the Gluon kernel's
# acq_rel split-K arrival (ACQ_REL).  BLOCK_M is never stored because it MUST
# cover M (the kernels mask rows against BLOCK_M).
#
# Produced by a schedule sweep on the 74-SM part: every config of a pruned
# grid timed in one CUDA graph rotating >= 256 MiB of weights, paired ABBA
# against the previous row; the top 6 re-timed over 5 paired repeats.  Triton
# rows: ratio over `_heuristic`.  Gluon rows: ratio over the Triton row they
# replaced; "acq_rel" ratios over the same row with the release arrival;
# "sliced" ratios over the acq_rel row; "uneven" rows split 40 K blocks 6 ways
# (time against the even sliced row, 3 swapped-order runs); "balanced" row:
# the persistent balanced-column Gluon kernel (same method).
# A row only replaced its predecessor at >= ~1 %.
# An M the table lacks uses the row of the next larger served M (`_SERVED_M`,
# same BLOCK_M); above the largest served M, `_heuristic`.

# WARPS_M value of a table row that selects `_gluon_sliced_kernel`.
SLICED_K = 0
# WARPS_M value of a table row that selects `_gluon_balanced_kernel`
# (SPLIT_K 1; BLOCK_N is the CTA's main tile, `_BALANCED_UNIT` its tail).
BALANCED = -1
_BALANCED_UNIT = 16

_CONFIG_OVERRIDES: dict = {
    # hc_down  N=336 K=10240 (heuristic: (16, 128, 7, 2, 4))
    (336, 10240, 1): (16, 128, 10, 2, 4, 1, 1),  # 1.064x; acq_rel 1.023x
    (336, 10240, 2): (16, 128, 10, 2, 4, 1, 1),  # 1.103x; acq_rel 1.022x
    (336, 10240, 4): (16, 128, 10, 2, 4, 1, 1),  # 1.100x; acq_rel 1.019x
    (336, 10240, 8): (16, 128, 10, 2, 4, SLICED_K, 1),   # sliced 1.023x
    (336, 10240, 16): (32, 256, 4, 2, 4, SLICED_K, 1),   # sliced 1.033x
    (336, 10240, 24): (32, 256, 6, 4, 4, SLICED_K, 1),   # uneven SK 0.968-0.971x time
    (336, 10240, 32): (32, 256, 6, 4, 3, SLICED_K, 1),   # uneven SK 0.975-0.980x time
    # final_down  N=320 K=10240 (heuristic: (16, 128, 7, 2, 4))
    (320, 10240, 1): (16, 128, 10, 2, 4, 1, 1),  # 1.077x; acq_rel 1.023x
    (320, 10240, 2): (16, 128, 10, 2, 4, 1, 1),  # 1.095x; acq_rel 1.019x
    (320, 10240, 4): (16, 128, 10, 2, 4, 1, 1),  # 1.090x; acq_rel 1.019x
    (320, 10240, 8): (16, 128, 10, 2, 4, SLICED_K, 1),   # sliced 1.028x
    (320, 10240, 16): (32, 256, 5, 2, 4, SLICED_K, 1),   # sliced 1.084x
    (320, 10240, 24): (32, 256, 5, 4, 4, SLICED_K, 1),   # sliced 1.077x
    (320, 10240, 32): (32, 256, 5, 4, 4, SLICED_K, 1),   # sliced 1.050x
    # hc_up  N=10240 K=320 (heuristic: (64, 64, 1, 4, 4))
    (10240, 320, 1): (32, 64, 1, 2, 4, 1),      # 1.072x over (32, 64, 1, 2, 4)
    (10240, 320, 2): (64, 64, 1, 2, 4, 1),      # 1.083x over (32, 64, 1, 2, 4)
    (10240, 320, 4): (64, 64, 1, 2, 4, 1),      # 1.091x over (32, 64, 1, 2, 4)
    (10240, 320, 8): (64, 64, 1, 2, 4, 1),      # 1.070x over (256, 64, 1, 4, 4)
    (10240, 320, 16): (64, 64, 1, 2, 4, 1),     # 1.022x over (256, 64, 1, 4, 4)
    (10240, 320, 24): (128, 64, 1, 4, 4, BALANCED),  # balanced 0.950-0.958x time over (64, 32, 1, 2, 4)
    (10240, 320, 32): (64, 64, 1, 2, 3),    # 1.106x
    # router  N=512 K=2560 (heuristic: (16, 128, 5, 2, 4))
    (512, 2560, 1): (16, 128, 5, 2, 4, 1, 1),    # 1.064x; acq_rel 1.042x
    (512, 2560, 2): (16, 128, 5, 2, 4, 1, 1),    # 1.064x; acq_rel 1.038x
    (512, 2560, 4): (16, 128, 5, 2, 4, SLICED_K, 1),     # sliced 1.016x
    (512, 2560, 8): (16, 128, 5, 2, 4, SLICED_K, 1),     # sliced 1.021x
    (512, 2560, 16): (16, 128, 5, 2, 4, SLICED_K, 1),    # sliced 1.027x
    (512, 2560, 24): (32, 128, 4, 2, 4, SLICED_K, 1),    # sliced 1.016x
    (512, 2560, 32): (32, 128, 4, 2, 4, SLICED_K, 1),    # sliced 1.041x
    # gdn_ba  N=48 K=2560 (heuristic: (16, 128, 10, 2, 4))
    (48, 2560, 1): (16, 128, 10, 2, 4, 1, 1),    # 1.032x; acq_rel 1.069x
    (48, 2560, 2): (16, 256, 10, 4, 4, SLICED_K, 1),     # sliced 1.019x
    (48, 2560, 4): (16, 256, 10, 4, 4, SLICED_K, 1),     # sliced 1.019x
    (48, 2560, 8): (16, 128, 10, 2, 3, SLICED_K, 1),     # sliced 1.020x
    (48, 2560, 16): (16, 128, 10, 2, 4, SLICED_K, 1),    # sliced 1.043x
    (48, 2560, 24): (16, 256, 10, 4, 3, SLICED_K, 1),    # sliced 1.024x
    (48, 2560, 32): (16, 256, 10, 4, 3, SLICED_K),       # sliced 1.023x
    # shared_gate  N=1 K=2560: `_gemv_kernel` (N <= _GEMV_MAX_N), no row here.
    # ple_kv  N=12800 K=2560 (heuristic: (64, 128, 1, 4, 4); M 1, 2 not swept)
    (12800, 2560, 1): (64, 64, 1, 4, 4),
    (12800, 2560, 2): (64, 64, 1, 4, 4),
    (12800, 2560, 4): (64, 64, 1, 4, 4),    # 1.183x
    (12800, 2560, 8): (64, 64, 1, 4, 4),    # 1.214x
    (12800, 2560, 16): (64, 64, 1, 4, 4),   # 1.245x
    (12800, 2560, 24): (64, 128, 1, 4, 3),  # 1.263x
    (12800, 2560, 32): (64, 128, 1, 4, 3),  # 1.354x
}

# Row counts the server dispatches through CUDA graphs (the table's M keys).
_SERVED_M = (1, 2, 4, 8, 16, 24, 32)

# Diagnostic schedule pin (``VLLM_THIN_GEMM_GLUON_FORCE_M``, set by
# ``gluon.py``): 0 (off) or a served row count S.  A call with M <= S then
# takes the schedule row and BLOCK_M of S instead of its own, so every row of
# every such call has the same summation order whatever the batch holds.
_FORCE_M = [0]

_num_sms_cache = None


def num_sms() -> int:
    global _num_sms_cache
    if _num_sms_cache is None:
        _num_sms_cache = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _num_sms_cache


def _pow2_at_least(x, lo=16):
    v = lo
    while v < x:
        v *= 2
    return v


def _schedule_m(M):
    """The row count whose schedule a call with M rows takes (``_FORCE_M``)."""
    return _FORCE_M[0] if 0 < M <= _FORCE_M[0] else M


def _heuristic(M, N, K, BLOCK_M):
    """Fallback for any (N, K) the sweep did not measure.

    Same principle the measured table encodes: pick the *largest* BLOCK_N that
    still yields ~_CTA_PER_SM_TARGET CTAs per SM from the N dimension alone, so
    SPLIT_K stays 1 and no reduction kernel runs; only split K when N cannot
    fill the machine by itself.
    """
    BLOCK_K = 128 if K >= 1024 else _pow2_at_least(min(K, 64), 16)
    want = max(1, int(_CTA_PER_SM_TARGET * num_sms()))

    BLOCK_N = 16
    for cand in (128, 64, 32):
        if triton.cdiv(N, cand) >= want:
            BLOCK_N = cand
            break

    tiles_n = triton.cdiv(N, BLOCK_N)
    max_split = max(1, min(triton.cdiv(K, BLOCK_K), K // _MIN_K_PER_SPLIT))
    SPLIT_K = max(1, min(max_split, -(-want // tiles_n)))

    num_warps = 4 if BLOCK_N >= 64 else 2
    num_stages = 4 if BLOCK_N * BLOCK_K <= 8192 else 3
    return BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages


def _table_row(M, N, K):
    """The measured row for (N, K) at the next served M >= M, or None.

    A row is (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) for the Triton
    kernel, or the same plus WARPS_M [, ACQ_REL] for a Gluon kernel (WARPS_M
    `SLICED_K` selects `_gluon_sliced_kernel`)."""
    sm = _schedule_m(M)
    served = next((s for s in _SERVED_M if s >= sm), None)
    return None if served is None else _CONFIG_OVERRIDES.get((N, K, served))


def _gluon_schedule(M, N, K):
    """-> (WARPS_M, ACQ_REL) of the row's Gluon schedule.  WARPS_M None: the
    Triton kernel (a 5-field row or no row); `SLICED_K`: the sliced-K kernel;
    `BALANCED`: the balanced persistent kernel; >= 1: `_gluon_gemm_kernel` with
    that many warps along M."""
    row = _table_row(M, N, K)
    if row is None or len(row) < 6:
        return None, False
    return row[5], len(row) > 6 and bool(row[6])


def _select_config(M, N, K):
    """-> (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)."""
    # tl.dot needs M >= 16 on Ampere (m16n8k16), so M is padded up to 16 and
    # masked.  The cap at 64 is a shared-memory bound, NOT a bound on M: the
    # kernel tiles M over grid dim 2, so any M is covered.
    BLOCK_M = min(64, _pow2_at_least(_schedule_m(M), 16))

    row = _table_row(M, N, K)
    cfg = row[:5] if row is not None else _heuristic(M, N, K, BLOCK_M)
    BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = cfg

    # A pinned config was measured at a small BLOCK_M; a larger M grows the
    # pipelined operand tiles, so trim the pipeline depth instead of asking for
    # more shared memory than an SM has.
    while num_stages > 2 and \
            num_stages * (BLOCK_N + BLOCK_M) * BLOCK_K * 2 > _MAX_SMEM:
        num_stages -= 1
    return BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

@triton.jit
def _thin_gemm_kernel(
    X, W, Y, P, LOCK,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    TILED_M: tl.constexpr,
    ROTATE_K: tl.constexpr,
):
    """One CTA per (M tile, N tile, K split).  Reads its K-blocks of W once.

    When SPLIT_K > 1 the FP32 partials are reduced by whichever CTA of the
    column arrives last, not by a second launch: the second launch cost 2-3.5 us
    per call (a bare in-graph launch is ~0.9 us on this part, and the rest is
    the drain plus one full HBM latency round with nothing left to overlap it).

    The reduction stays bitwise deterministic because the *order* is fixed -
    always partial 0, then 1, ... SPLIT_K-1 - even though *which* CTA performs
    it varies run to run.  That is the whole reason this is not an atomic_add
    into Y, which would not be reproducible.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # BLOCK_M is capped by shared memory, so a caller with more rows than one
    # tile gets several M tiles over grid dim 2.  Masking alone is NOT enough:
    # with a pinned BLOCK_M and no pid_m, rows past BLOCK_M were never written.
    # Every M the model uses (1..64) is a single tile, and there the row offset
    # must stay a compile-time zero -- a runtime `pid_m * BLOCK_M` in the
    # address arithmetic costs 0.5-4.5 % on every shape (measured), so the
    # one-tile case is specialized rather than paying for generality.
    if TILED_M:
        pid_m = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        pid_m = 0
        offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    # Clamping the row/column index keeps every address inside the tensor, so
    # the masked-off lanes of a ragged tile cannot fault even before the mask
    # is applied.
    x_m = tl.where(m_mask, offs_m, 0)
    w_n = tl.where(n_mask, offs_n, 0)

    x_ptrs = X + x_m[:, None] * stride_xm + \
        (pid_k * BLOCK_K + offs_k)[None, :] * stride_xk
    w_ptrs = W + w_n[:, None] * stride_wn + \
        (pid_k * BLOCK_K + offs_k)[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step = BLOCK_K * SPLIT_K
    n_iter = tl.cdiv(K, step)
    if ROTATE_K:
        # Every CTA streams the same x.  In lock step they would all read the
        # same x K-block from the same L2 lines at once; starting each N tile
        # at a different K-block spreads those reads (summation order changes,
        # deterministically: the start depends only on the tile).  Only taken
        # when EVEN_K, so every K-block is whole.
        shift = (pid_n * 3) % n_iter
        for i in range(n_iter):
            j = i + shift
            j = tl.where(j >= n_iter, j - n_iter, j)
            x = tl.load(x_ptrs + j * step * stride_xk, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs + j * step * stride_wk, mask=n_mask[:, None], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    else:
        for i in range(n_iter):
            if EVEN_K:
                x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
                w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            else:
                k_now = i * step + pid_k * BLOCK_K + offs_k
                k_mask = k_now < K
                x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
            x_ptrs += step * stride_xk
            w_ptrs += step * stride_wk

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    if SPLIT_K == 1:
        if EVEN_N:
            tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=m_mask[:, None])
        else:
            tl.store(y_ptrs, acc.to(Y.dtype.element_ty),
                     mask=m_mask[:, None] & n_mask[None, :])
    else:
        p_mask = m_mask[:, None] & n_mask[None, :]
        p_off = offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        # ".cg" keeps the partial out of the non-coherent per-SM L1 so the
        # reducing CTA, which runs on a different SM, can see it in L2.
        tl.store(P + pid_k * stride_pk + p_off, acc, mask=p_mask,
                 cache_modifier=".cg")
        # Release the store, then count arrivals for this N column.  No CTA
        # ever spins, so this cannot deadlock however the grid is scheduled.
        # One counter per (M tile, N tile) output block; grid dim 0 is the
        # number of N tiles, so this indexes the column within this M tile.
        if TILED_M:
            lock = LOCK + pid_m * tl.num_programs(0) + pid_n
        else:
            lock = LOCK + pid_n
        # The arrival only needs release (it publishes this CTA's partial).
        # Only the last arriver needs acquire, and it gets it from the counter
        # reset: an acquire RMW on the same counter, after the others' release
        # RMWs in its modification order, synchronizes with all of them.  The
        # non-last CTAs thus skip the acquire fence (measured +3-10 % on the
        # split-K shapes vs one acq_rel arrival).  Launches on one stream are
        # serialized, so resetting before the reads cannot race a later CTA.
        arrived = tl.atomic_add(lock, 1, sem="release", scope="gpu")
        if arrived == SPLIT_K - 1:
            tl.atomic_xchg(lock, 0, sem="acquire", scope="gpu")
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                tot += tl.load(P + k * stride_pk + p_off, mask=p_mask,
                               other=0.0, cache_modifier=".cv")
            tl.store(y_ptrs, tot.to(Y.dtype.element_ty), mask=p_mask)


# Largest N served by `_gemv_kernel` instead of the tensor-core kernel.
_GEMV_MAX_N = 1
# K elements one GEMV CTA loads per step (8 warps: 16 per thread at 4096).
_GEMV_MAX_BLOCK_K = 4096


@triton.jit
def _gemv_kernel(
    X, W, Y,
    K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """One CTA per output element (row m, column n), full K, no split.

    For N = 1 (shared_gate) the weight is a few KB and the tensor-core
    schedule is pure latency: a 16-row MMA tile wastes 15 of 16 columns and
    split-K adds a partial store, an arrival atomic and a reload (~1.3 us of
    a 3 us call, measured).  Here every CTA issues all its K loads at once
    (one HBM/L2 round trip when K <= BLOCK_K), multiplies in FP32 (a bf16 x
    bf16 product is exact in FP32) and reduces with one fixed-shape tl.sum,
    so the summation order is fixed and the result bitwise reproducible.
    """
    m = tl.program_id(0)
    n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = X + m * stride_xm + offs_k * stride_xk
    w_ptrs = W + n * stride_wn + offs_k * stride_wk
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if EVEN_K:
            xv = tl.load(x_ptrs)
            wv = tl.load(w_ptrs)
        else:
            k_mask = k0 + offs_k < K
            xv = tl.load(x_ptrs, mask=k_mask, other=0.0)
            wv = tl.load(w_ptrs, mask=k_mask, other=0.0)
        acc += xv.to(tl.float32) * wv.to(tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    tl.store(Y + m * stride_ym + n * stride_yn, tl.sum(acc).to(Y.dtype.element_ty))


@gluon.jit
def _gluon_gemm_kernel(
    X, W, Y, P, LOCK,
    M, N, K,
    stride_xm, stride_wn,
    stride_ym, stride_yn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WARPS_M: gl.constexpr,
    STAGES: gl.constexpr,
    THREADS_K: gl.constexpr,
    ACQ_REL: gl.constexpr,
):
    """`_thin_gemm_kernel` (EVEN_K, unit K strides, one M tile, no ROTATE_K)
    with explicit layouts.

    Same grid, same strided K partition, same MMA accumulation order and the
    same fixed-order last-arriver reduction, so a column's value is bitwise
    the Triton kernel's at the same (BLOCK_K, SPLIT_K).  What differs: the
    partial store, the reducer's reload and the output store all stay in the
    MMA accumulator layout.  Triton converts `acc` to a blocked layout through
    shared memory (with barriers) before those stores, which cost 0.5-0.8 us
    per split-K call (measured).  Operands come through a STAGES-slot cp.async
    ring, one commit group per stage holding both x and W.
    """
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WARPS_M, NUM_WARPS // WARPS_M],
        instr_shape=[16, 8])
    la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[32 // THREADS_K, THREADS_K],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    sl: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                               order=[1, 0])
    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)

    rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blk))
    rn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, blk))
    rk = pid_k * BLOCK_K + gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blk))
    m_ok = rm < M
    n_ok = rn < N
    # Clamped rows keep every address in bounds; the masks zero the padding.
    x_ptrs = X + gl.where(m_ok, rm, 0)[:, None] * stride_xm + rk[None, :]
    w_ptrs = W + gl.where(n_ok, rn, 0)[:, None] * stride_wn + rk[None, :]
    x_mask = m_ok[:, None] & (rk < K)[None, :]
    w_mask = n_ok[:, None] & (rk < K)[None, :]
    xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_M, BLOCK_K], sl)
    ws = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_N, BLOCK_K], sl)

    step = BLOCK_K * SPLIT_K
    n_iter = K // step
    for s in gl.static_range(STAGES - 1):
        if s < n_iter:
            cp.async_copy_global_to_shared(xs.index(s), x_ptrs + s * step, mask=x_mask)
            cp.async_copy_global_to_shared(ws.index(s), w_ptrs + s * step, mask=w_mask)
        cp.commit_group()
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma)
    for i in range(n_iter):
        cp.wait_group(STAGES - 2)
        # Every warp is past iteration i-1, so its slot can be refilled.
        gl.barrier()
        j = i + STAGES - 1
        if j < n_iter:
            cp.async_copy_global_to_shared(xs.index(j % STAGES), x_ptrs + j * step,
                                           mask=x_mask)
            cp.async_copy_global_to_shared(ws.index(j % STAGES), w_ptrs + j * step,
                                           mask=w_mask)
        cp.commit_group()
        a = xs.index(i % STAGES).load(la)
        b = ws.index(i % STAGES).permute([1, 0]).load(lb)
        acc = mma_v2(a, b, acc)

    _gluon_epilogue(acc, Y, P, LOCK, M, N, pid_n, pid_k,
                    stride_ym, stride_yn, stride_pk, stride_pm, stride_pn,
                    BLOCK_M, BLOCK_N, SPLIT_K, ACQ_REL, mma)


@gluon.jit
def _gluon_epilogue(acc, Y, P, LOCK, M, N, pid_n, pid_k,
                    stride_ym, stride_yn, stride_pk, stride_pm, stride_pn,
                    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                    SPLIT_K: gl.constexpr, ACQ_REL: gl.constexpr,
                    layout: gl.constexpr):
    """Store `acc` (SPLIT_K 1) or run the split-K handoff, all in `layout`
    (the accumulator's own layout: no conversion before the stores)."""
    om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    on = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    o_mask = (om < M)[:, None] & (on < N)[None, :]
    y_ptrs = Y + om[:, None] * stride_ym + on[None, :] * stride_yn
    if SPLIT_K == 1:
        gl.store(y_ptrs, acc.to(gl.bfloat16), mask=o_mask)
    else:
        p_off = om[:, None] * stride_pm + on[None, :] * stride_pn
        gl.store(P + pid_k * stride_pk + p_off, acc, mask=o_mask,
                 cache_modifier=".cg")
        if ACQ_REL:
            # One acq_rel arrival: the last arriver's RMW reads from the
            # release sequence of every earlier arrival, so it synchronizes
            # with all of them; its acquire precedes the CTA barrier that
            # broadcasts `arrived`, ordering every thread's reload after it.
            # The reset then needs no ordering of its own (stream order covers
            # the next launch), saving the acquire round trip.
            arrived = gl.atomic_add(LOCK + pid_n, 1, sem="acq_rel", scope="gpu")
        else:
            # Release arrival; the last arriver acquires through the reset.
            arrived = gl.atomic_add(LOCK + pid_n, 1, sem="release", scope="gpu")
        if arrived == SPLIT_K - 1:
            if ACQ_REL:
                gl.atomic_xchg(LOCK + pid_n, 0, sem="relaxed", scope="gpu")
            else:
                gl.atomic_xchg(LOCK + pid_n, 0, sem="acquire", scope="gpu")
            tot = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=layout)
            for k in gl.static_range(SPLIT_K):
                tot += gl.load(P + k * stride_pk + p_off, mask=o_mask, other=0.0,
                               cache_modifier=".cg")
            gl.store(y_ptrs, tot.to(gl.bfloat16), mask=o_mask)


@gluon.jit
def _gluon_sliced_kernel(
    X, W, Y, P, LOCK,
    M, N, K,
    stride_xm, stride_wn,
    stride_ym, stride_yn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    STAGES: gl.constexpr,
    THREADS_K: gl.constexpr,
    ACQ_REL: gl.constexpr,
    EVEN_SPLIT: gl.constexpr,
):
    """`_gluon_gemm_kernel` with the K-block sliced across warps.

    Warp w owns columns [w*KS, (w+1)*KS) of every BLOCK_K block (KS = BLOCK_K /
    NUM_WARPS) and a full [BLOCK_M, BLOCK_N] accumulator, so no operand byte is
    read from shared memory by more than one warp (with a 2-D warp grid each x
    byte is read by every N warp and each W byte by every M warp).  At
    BLOCK_M 32 that halves the ldmatrix traffic, which bounds those rows.  The
    NUM_WARPS accumulators are summed once after the K loop, through shared
    memory, by each thread over a fixed in-thread order: deterministic, but a
    different summation order from the other kernels (not bitwise theirs).
    """
    KS: gl.constexpr = BLOCK_K // NUM_WARPS
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[NUM_WARPS, 1, 1], instr_shape=[1, 16, 8])
    la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 8], threads_per_warp=[1, 32 // THREADS_K, THREADS_K],
        warps_per_cta=[NUM_WARPS, 1, 1], order=[2, 1, 0])
    sl: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                               order=[2, 1, 0])
    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)

    l0: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blk))
    l1: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blk))
    l2: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blk))
    rw = gl.arange(0, NUM_WARPS, layout=l0)
    rm = gl.arange(0, BLOCK_M, layout=l1)
    rn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=l1)
    rk = gl.arange(0, KS, layout=l2)
    kk = pid_k * BLOCK_K + rw[:, None, None] * KS + rk[None, None, :]
    m_ok = rm < M
    n_ok = rn < N
    x_ptrs = X + gl.where(m_ok, rm, 0)[None, :, None] * stride_xm + kk
    w_ptrs = W + gl.where(n_ok, rn, 0)[None, :, None] * stride_wn + kk
    x_mask = m_ok[None, :, None] & (kk < K)
    w_mask = n_ok[None, :, None] & (kk < K)
    xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, NUM_WARPS, BLOCK_M, KS], sl)
    ws = gl.allocate_shared_memory(gl.bfloat16, [STAGES, NUM_WARPS, BLOCK_N, KS], sl)

    step = BLOCK_K * SPLIT_K
    if EVEN_SPLIT:
        n_iter = K // step
    else:
        # K-block b goes to split b % SPLIT_K (the Triton kernel's partition,
        # bitwise its masked path); with a block count SPLIT_K does not divide,
        # e.g. 40 blocks over 6 splits run 7,7,7,7,6,6 iterations.
        n_iter = (K // BLOCK_K - pid_k + SPLIT_K - 1) // SPLIT_K
    for s in gl.static_range(STAGES - 1):
        if s < n_iter:
            cp.async_copy_global_to_shared(xs.index(s), x_ptrs + s * step, mask=x_mask)
            cp.async_copy_global_to_shared(ws.index(s), w_ptrs + s * step, mask=w_mask)
        cp.commit_group()
    acc = gl.zeros([NUM_WARPS, BLOCK_M, BLOCK_N], gl.float32, layout=mma)
    for i in range(n_iter):
        cp.wait_group(STAGES - 2)
        gl.barrier()
        j = i + STAGES - 1
        if j < n_iter:
            cp.async_copy_global_to_shared(xs.index(j % STAGES), x_ptrs + j * step,
                                           mask=x_mask)
            cp.async_copy_global_to_shared(ws.index(j % STAGES), w_ptrs + j * step,
                                           mask=w_mask)
        cp.commit_group()
        a = xs.index(i % STAGES).load(la)
        b = ws.index(i % STAGES).permute([0, 2, 1]).load(lb)
        acc = mma_v2(a, b, acc)

    # Warp partials -> one tile: through shared memory, then each thread sums
    # its NUM_WARPS values in-thread (gl.sum over the 3-D MMA tensor directly
    # spilled registers in the probe).
    rs: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1,
                                               order=[2, 1, 0])
    red = gl.allocate_shared_memory(gl.float32, [NUM_WARPS, BLOCK_M, BLOCK_N], rs)
    red.store(acc)
    gl.barrier()
    el3: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[NUM_WARPS, 1, 4],
        threads_per_warp=[1, 32 // (BLOCK_N // 4), BLOCK_N // 4],
        warps_per_cta=[1, NUM_WARPS, 1], order=[2, 1, 0])
    tile = gl.sum(red.load(el3), axis=0)
    _gluon_epilogue(tile, Y, P, LOCK, M, N, pid_n, pid_k,
                    stride_ym, stride_yn, stride_pk, stride_pm, stride_pn,
                    BLOCK_M, BLOCK_N, SPLIT_K, ACQ_REL, gl.SliceLayout(0, el3))


@gluon.jit
def _gluon_balanced_kernel(
    X, W, Y,
    M, N, K,
    stride_xm, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: gl.constexpr,
    BN0: gl.constexpr,
    BN1: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    STAGES: gl.constexpr,
    THREADS_K: gl.constexpr,
):
    """SPLIT_K 1 GEMM over a persistent grid of one CTA per SM with balanced
    column ranges.

    The N columns are cut into BN1-wide units spread evenly over the G CTAs;
    CTA c owns columns [BN1*floor(c*U/G), BN1*floor((c+1)*U/G)), at most
    BN0 + BN1 of them, computed as a BN0 tile plus a BN1 tile that share each
    x stage.  A tiled grid (e.g. 160 tiles of 64 over 74 SMs) leaves some SMs
    with an extra CTA that sets the end of the call.  A column's value never
    depends on which CTA computes it (K blocks are summed 0..K/BLOCK_K-1 in
    every CTA), so the result is independent of G; no CTA waits on another
    and there is no workspace.
    """
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, NUM_WARPS], instr_shape=[16, 8])
    la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[32 // THREADS_K, THREADS_K],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    sl: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8,
                                               order=[1, 0])
    pid = gl.program_id(0)
    G = gl.num_programs(0)
    U = (N + BN1 - 1) // BN1
    c0 = (pid * U // G) * BN1
    c1 = gl.minimum(((pid + 1) * U // G) * BN1, N)

    rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blk))
    rk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blk))
    m_ok = rm < M
    x_ptrs = X + gl.where(m_ok, rm, 0)[:, None] * stride_xm + rk[None, :]
    x_mask = m_ok[:, None] & (rk < K)[None, :]
    rn0 = c0 + gl.arange(0, BN0, layout=gl.SliceLayout(1, blk))
    n0_ok = rn0 < c1
    w0_ptrs = W + gl.where(n0_ok, rn0, 0)[:, None] * stride_wn + rk[None, :]
    w0_mask = n0_ok[:, None] & (rk < K)[None, :]
    rn1 = c0 + BN0 + gl.arange(0, BN1, layout=gl.SliceLayout(1, blk))
    n1_ok = rn1 < c1
    w1_ptrs = W + gl.where(n1_ok, rn1, 0)[:, None] * stride_wn + rk[None, :]
    w1_mask = n1_ok[:, None] & (rk < K)[None, :]
    xs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_M, BLOCK_K], sl)
    ws0 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BN0, BLOCK_K], sl)
    ws1 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BN1, BLOCK_K], sl)

    n_iter = K // BLOCK_K
    for s in gl.static_range(STAGES - 1):
        if s < n_iter:
            cp.async_copy_global_to_shared(xs.index(s), x_ptrs + s * BLOCK_K, mask=x_mask)
            cp.async_copy_global_to_shared(ws0.index(s), w0_ptrs + s * BLOCK_K, mask=w0_mask)
            cp.async_copy_global_to_shared(ws1.index(s), w1_ptrs + s * BLOCK_K, mask=w1_mask)
        cp.commit_group()
    acc0 = gl.zeros([BLOCK_M, BN0], gl.float32, layout=mma)
    acc1 = gl.zeros([BLOCK_M, BN1], gl.float32, layout=mma)
    for i in range(n_iter):
        cp.wait_group(STAGES - 2)
        gl.barrier()
        j = i + STAGES - 1
        if j < n_iter:
            cp.async_copy_global_to_shared(xs.index(j % STAGES), x_ptrs + j * BLOCK_K,
                                           mask=x_mask)
            cp.async_copy_global_to_shared(ws0.index(j % STAGES), w0_ptrs + j * BLOCK_K,
                                           mask=w0_mask)
            cp.async_copy_global_to_shared(ws1.index(j % STAGES), w1_ptrs + j * BLOCK_K,
                                           mask=w1_mask)
        cp.commit_group()
        a = xs.index(i % STAGES).load(la)
        acc0 = mma_v2(a, ws0.index(i % STAGES).permute([1, 0]).load(lb), acc0)
        acc1 = mma_v2(a, ws1.index(i % STAGES).permute([1, 0]).load(lb), acc1)

    om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma))
    on0 = c0 + gl.arange(0, BN0, layout=gl.SliceLayout(0, mma))
    gl.store(Y + om[:, None] * stride_ym + on0[None, :] * stride_yn, acc0.to(gl.bfloat16),
             mask=(om < M)[:, None] & (on0 < c1)[None, :])
    on1 = c0 + BN0 + gl.arange(0, BN1, layout=gl.SliceLayout(0, mma))
    gl.store(Y + om[:, None] * stride_ym + on1[None, :] * stride_yn, acc1.to(gl.bfloat16),
             mask=(om < M)[:, None] & (on1 < c1)[None, :])


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
# Buffers are cached forever and never shrunk.  That matters for CUDA-graph
# capture: a graph records the pointer it was captured with, so the buffer must
# outlive the graph.  Call warmup() before capture to force every size the
# model needs to exist first.

_WORKSPACE: dict = {}

# Workspace lanes.  The shared partials and arrival counters are safe because
# launches on one stream are serialized.  A caller that runs thin GEMMs on a
# second stream while the first one is busy selects another lane and gets
# buffers of its own.  Lane 0, the default, keeps the keys every other caller
# has always used.  A plain module-level slot (not a ContextVar) so that the
# lookup stays traceable inside torch.compile'd regions; a compiled region
# keeps the buffers of the lane it was traced in.
_LANE = [0]


@contextlib.contextmanager
def workspace_lane(lane: int):
    """Run the enclosed thin GEMMs on workspace lane ``lane`` (>= 0)."""
    if lane < 0:
        raise ValueError(f"workspace lane must be non-negative, got {lane}")
    prev = _LANE[0]
    _LANE[0] = lane
    try:
        yield
    finally:
        _LANE[0] = prev


def _lane_key(*key):
    lane = _LANE[0]
    return key if lane == 0 else ("lane", lane, *key)


def _partials(split_k, M, N, device):
    key = _lane_key(device.index, split_k * M * N)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.empty(split_k * M * N, dtype=torch.float32, device=device)
        _WORKSPACE[key] = buf
    return buf.view(split_k, M, N)


def _locks(tiles_n, device):
    """Zeroed arrival counters, one per N tile.

    Cached like every other workspace buffer so the pointer a captured graph
    recorded stays valid.  The kernel always leaves the counters at zero, so a
    buffer can be reused by any shape and across replays.
    """
    key = _lane_key(device.index, "lock", tiles_n)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.zeros(tiles_n, dtype=torch.int32, device=device)
        _WORKSPACE[key] = buf
    return buf


def _dummy_fp32(device):
    key = (device.index, "dummy")
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _WORKSPACE[key] = buf
    return buf


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _rotate_k(even_k: bool, split_k: int, block_m: int, block_n: int) -> bool:
    """Whether a launch rotates its K start per N tile (``ROTATE_K``)."""
    return even_k and split_k == 1 and (block_m >= 32 or block_n >= 128)

def thin_gemm(x: torch.Tensor, w: torch.Tensor,
              out: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ w.T for BF16 x [M, K] and BF16 w [N, K], FP32 accumulate."""
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, \
        "thin_gemm is BF16-only; no quantized or FP8 path exists by design"
    assert x.ndim == 2 and w.ndim == 2 and x.shape[1] == w.shape[1]
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M == 0 or N == 0:
        return out

    if N <= _GEMV_MAX_N:
        BLOCK_K = min(_GEMV_MAX_BLOCK_K, _pow2_at_least(K, 16))
        _gemv_kernel[(M, N)](
            x, w, out,
            K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_K=BLOCK_K, EVEN_K=(K % BLOCK_K) == 0,
            num_warps=8,
        )
        return out

    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = _select_config(M, N, K)
    even_k = (K % (BLOCK_K * SPLIT_K)) == 0
    even_n = (N % BLOCK_N) == 0

    tiles_n = triton.cdiv(N, BLOCK_N)
    # BLOCK_M is bounded by shared memory, so a caller with M > BLOCK_M needs
    # more than one M tile.  Every benchmarked M (1..32) gives tiles_m == 1.
    tiles_m = triton.cdiv(M, BLOCK_M)
    if SPLIT_K == 1:
        # Unused by the kernel (the SPLIT_K branch is constexpr-folded away),
        # but the arguments still have to be real pointers.
        p = _dummy_fp32(x.device)
        sp = (0, 0, 0)
        lock = _locks(1, x.device)
    else:
        p = _partials(SPLIT_K, M, N, x.device)
        sp = (p.stride(0), p.stride(1), p.stride(2))
        lock = _locks(tiles_n * tiles_m, x.device)

    warps_m, acq_rel = _gluon_schedule(M, N, K)
    if warps_m == BALANCED:
        # Every CTA's balanced column range must fit its BLOCK_N + unit tiles.
        units = triton.cdiv(N, _BALANCED_UNIT)
        gluon_k_ok = (SPLIT_K == 1 and K % BLOCK_K == 0 and _BALANCED_UNIT *
                      triton.cdiv(units, num_sms()) <= BLOCK_N + _BALANCED_UNIT)
    elif warps_m == SLICED_K:
        # The sliced kernel takes any split of whole K blocks; the 2-D one EVEN_K.
        gluon_k_ok = K % BLOCK_K == 0
    else:
        gluon_k_ok = even_k
    if (warps_m is not None and gluon_k_ok and tiles_m == 1
            and x.stride(1) == 1 and w.stride(1) == 1
            and x.stride(0) % 16 == 0 and w.stride(0) % 16 == 0
            and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0):
        # A Gluon row; awkward inputs (partial K blocks, strided K, a row
        # stride that is not a multiple of 16 elements, an unaligned base,
        # several M tiles) keep the Triton kernel below. The Gluon kernels'
        # cp.async loads need the 16-element row strides they compile for.
        if warps_m == BALANCED:
            _gluon_balanced_kernel[(num_sms(),)](
                x, w, out, M, N, K, x.stride(0), w.stride(0),
                out.stride(0), out.stride(1),
                BLOCK_M=BLOCK_M, BN0=BLOCK_N, BN1=_BALANCED_UNIT, BLOCK_K=BLOCK_K,
                NUM_WARPS=num_warps, STAGES=num_stages,
                THREADS_K=min(BLOCK_K // 8, 32),
                num_warps=num_warps,
            )
            return out
        args = (x, w, out, p, lock, M, N, K, x.stride(0), w.stride(0),
                out.stride(0), out.stride(1), sp[0], sp[1], sp[2])
        if warps_m == SLICED_K:
            _gluon_sliced_kernel[(tiles_n, SPLIT_K)](
                *args,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
                NUM_WARPS=num_warps, STAGES=num_stages,
                THREADS_K=min(BLOCK_K // num_warps // 8, 32), ACQ_REL=acq_rel,
                EVEN_SPLIT=even_k,
                num_warps=num_warps,
            )
        else:
            _gluon_gemm_kernel[(tiles_n, SPLIT_K)](
                *args,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
                NUM_WARPS=num_warps, WARPS_M=warps_m, STAGES=num_stages,
                THREADS_K=min(BLOCK_K // 8, 32), ACQ_REL=acq_rel,
                num_warps=num_warps,
            )
        return out

    _thin_gemm_kernel[(tiles_n, SPLIT_K, tiles_m)](
        x, w, out, p, lock,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        sp[0], sp[1], sp[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
        EVEN_K=even_k, EVEN_N=even_n, TILED_M=(tiles_m > 1),
        # Rotation measured on the 74-SM part: +2-5 % at BLOCK_M >= 32 (x tile
        # as large as a W tile row block) and on BLOCK_N >= 128 tiles at any M
        # (it also evens out the per-launch CTA finish spread); -0.5 to -3 %
        # on narrow BLOCK_N < 128 tiles at BLOCK_M = 16 and on split-K shapes.
        ROTATE_K=_rotate_k(even_k, SPLIT_K, BLOCK_M, BLOCK_N),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def warmup(shapes, ms, device=None) -> None:
    """JIT-compile and allocate for every (M, N, K) that will be replayed.

    Triton compiles on first launch and the workspace is allocated lazily, so
    neither may happen inside a `torch.cuda.graph` capture.  vLLM calls this
    once before capture (``vllm_sm80/kernels/thin_gemm/warmup.py``).
    """
    device = device or torch.device("cuda", torch.cuda.current_device())
    for N, K in shapes:
        w = torch.empty((N, K), dtype=torch.bfloat16, device=device)
        for M in ms:
            x = torch.empty((M, K), dtype=torch.bfloat16, device=device)
            thin_gemm(x, w)
    torch.cuda.synchronize(device)
