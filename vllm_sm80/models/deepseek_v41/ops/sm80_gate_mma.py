# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE gate logits ``x @ W^T`` on the BF16 tensor cores with exact products (sm_80).

x [M, K] is BF16 and the prepared weight [E, K] holds the gate's BF16 values, so every product of a BF16 x BF16
MMA is exact in its FP32 accumulator. Program (e, s, t) walks its K slice in chunks of ``BLOCK_K``: each chunk is
one ``tl.dot`` started from zero (the MMA sums at most ``BLOCK_K`` exact products in FP32), and the chunk sums are
added in FP64. The FP64 total is rounded once to FP32. The only error beyond the FP64 sum is the FP32 rounding
inside a chunk, far below the FP32 SGEMM's running-sum error and within the FP64 gate of the GEMV it replaces.

Split K (``SPLIT`` > 1): each program writes its FP64 partial to scratch and counts itself in with an acquire-release
atomic on the counter of its (token block, expert block) tile; the last to arrive sums the ``SPLIT`` partials in the
fixed order 0 .. SPLIT-1, writes the FP32 logits and resets the counter to zero (deterministic, graph-replayable).
"""
from vllm.triton_utils import tl, triton

# Counter slots per launch (int32): token blocks x expert blocks of every split launch fit.
LOCK_SLOTS = 256


@triton.jit(do_not_specialize=["M"])
def _gate_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    part_ptr,
    count_ptr,
    M,
    stride_xm,
    stride_om,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DEFER: tl.constexpr,
):
    """DEFER (SPLIT > 1): no completion step; program (e, s, t) writes its FP64 partial to ``part[s, row, expert]``
    ([SPLIT, TB * BLOCK_T, NB * BLOCK_E]) and the consumer adds the SPLIT partials in order 0 .. SPLIT-1 and rounds
    once (the same sum and rounding as the completion step)."""
    pid_e = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    KS: tl.constexpr = K // SPLIT
    NB: tl.constexpr = (N + BLOCK_E - 1) // BLOCK_E
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K)
    mask_e = offs_e < N
    mask_t = offs_t < M
    # Rows past N or M read weight row 0 or token row 0; their results are never stored.
    rows_e = tl.where(mask_e, offs_e, 0).to(tl.int64)
    rows_t = tl.where(mask_t, offs_t, 0).to(tl.int64)
    k0 = pid_s * KS
    w_ptrs = w_ptr + rows_e[:, None] * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + rows_t[:, None] * stride_xm + (k0 + offs_k)[None, :]
    acc = tl.zeros([BLOCK_E, BLOCK_T], dtype=tl.float64)
    for _ in range(KS // BLOCK_K):
        w = tl.load(w_ptrs)
        xt = tl.load(x_ptrs)
        acc += tl.dot(w, tl.trans(xt), out_dtype=tl.float32).to(tl.float64)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_t[None, :].to(tl.int64) * stride_om + offs_e[:, None]
    out_mask = mask_e[:, None] & mask_t[None, :]
    if SPLIT == 1:
        tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)
    elif DEFER:
        rows = pid_s * tl.num_programs(2) * BLOCK_T + offs_t
        tl.store(part_ptr + rows[None, :] * (NB * BLOCK_E) + offs_e[:, None], acc)
    else:
        TILE: tl.constexpr = NB * BLOCK_E * BLOCK_T
        tile = (pid_t * NB * BLOCK_E + offs_e[:, None]) * BLOCK_T + tl.arange(0, BLOCK_T)[None, :]
        tl.store(part_ptr + pid_s * TILE * tl.num_programs(2) + tile, acc, cache_modifier=".cg")
        # Every thread's partial is written before one thread's release increment; that thread's acquire is
        # ordered before every thread's loads below by the barrier that shares the atomic's result.
        tl.debug_barrier()
        counter = count_ptr + pid_t * NB + pid_e
        arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT - 1:
            total = tl.zeros([BLOCK_E, BLOCK_T], dtype=tl.float64)
            for s in range(SPLIT):
                total += tl.load(part_ptr + s * TILE * tl.num_programs(2) + tile, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(tl.float32), mask=out_mask)
            tl.store(counter, 0)


def config(M, E, K, sms):
    """(BLOCK_T, BLOCK_E, BLOCK_K, SPLIT, token blocks, num_warps, num_stages) for a call of M rows."""
    if M <= 64:
        block_t = min(32, max(16, triton.next_power_of_2(M)))
        block_e, block_k, warps, stages, per_sm = (16 if E <= 128 else 32), 128, 4, 4, 1
    else:
        block_t, block_e, block_k, warps, stages, per_sm = 64, 64, 64, 4, 3, 2
    tb = triton.cdiv(M, block_t)
    nb = triton.cdiv(E, block_e)
    chunks = K // block_k
    limit = max(1, per_sm * sms // (nb * tb))
    split = max(d for d in range(1, chunks + 1) if chunks % d == 0 and d <= limit)
    if tb * nb > LOCK_SLOTS:
        split = 1
    return block_t, block_e, block_k, split, tb, warps, stages


def part_numel(M, E, K, sms):
    """FP64 partial elements of a call (0 without split K)."""
    block_t, block_e, _, split, tb, _, _ = config(M, E, K, sms)
    if split == 1:
        return 0
    return split * tb * triton.cdiv(E, block_e) * block_e * block_t


def launch(x, w, logits, part, locks, sms, defer):
    """The gate for x; with ``defer`` (and split K) only the FP64 partials ``part[s, row, expert]`` are written."""
    M, K = x.shape
    E = w.shape[0]
    block_t, block_e, block_k, split, tb, warps, stages = config(M, E, K, sms)
    _gate_kernel[(triton.cdiv(E, block_e), split, tb)](
        x, w, logits, part, locks, M, x.stride(0), logits.stride(0),
        N=E, K=K, SPLIT=split, BLOCK_T=block_t, BLOCK_E=block_e, BLOCK_K=block_k, DEFER=defer, num_warps=warps,
        num_stages=stages)
