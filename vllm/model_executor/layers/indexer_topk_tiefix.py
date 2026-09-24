# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tie-consistent repair of a sparse-indexer top-k (VLLM_GLM5_TOPK_TIEFIX).

The stock top-k kernels (persistent_topk and its FilteredTopK/sampled_topk
branches for decode, topKPerRowJob for prefill) all return an exact top-k by
score, but place the entries that tie with the k-th score by atomic arrival
order, and which kernel runs depends on the number of rows and the row length.
When the tied group straddles the selection boundary the selected SET is then a
function of batch shape and timing, not of the scores.

This pass keeps the fast kernel and repairs only that boundary. Per row:

  T      = smallest selected score (the k-th score, since the set is exact)
  n_gt   = selected entries scoring above T (these are all entries above T)
  need   = selected entries equal to T
  n_eq   = entries in the row's window equal to T (one read of the row)

If ``n_eq == need`` every tied entry is already selected and the row is left
untouched, byte for byte. Otherwise the row is rewritten as the n_gt entries
above T (in the kernel's order) followed by the ``need`` lowest-index entries
equal to T, and -1 past the selected count. The result is the canonical set
(score descending, column index ascending) that ``canonical_topk`` selects,
at the cost of one extra read of the logits instead of an int64 sort.

Scores are compared through the same order-preserving integer image
``canonical_topk`` uses: -0.0 is folded onto +0.0 and NaN ranks above +inf
(a negative-signed NaN below -inf), so the two agree on every input.

One Triton program per row, grid (num_rows,), no host reads: safe inside CUDA
graph capture. Runs on CPU tensors under TRITON_INTERPRET=1 (the tests).
"""

import torch

from vllm.triton_utils import tl, triton

# Scan tile for the full-row passes and the warps per row program.
TIEFIX_BLOCK = 4096
TIEFIX_NUM_WARPS = 8


@triton.jit
def _ordered_key(x):
    """fp32 -> int32 with a > b iff key(a) > key(b); -0.0 folds onto +0.0."""
    bits = x.to(tl.int32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


# The row geometry (strides, width) is not specialised: one compiled variant per
# (K, HAS_STARTS, RELATIVE) whatever the batch, so nothing new compiles after
# warm_tiefix() has run, in particular not during CUDA graph capture.
@triton.jit(do_not_specialize=["stride_l", "stride_i", "n_cols"])
def _topk_tiefix_kernel(
    logits_ptr,
    idx_ptr,
    starts_ptr,
    ends_ptr,
    stride_l,
    stride_i,
    n_cols,
    K: tl.constexpr,
    K_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    RELATIVE: tl.constexpr,
    SORT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lrow = logits_ptr + row * stride_l
    irow = idx_ptr + row * stride_i

    end = tl.load(ends_ptr + row).to(tl.int32)
    if HAS_STARTS:
        start = tl.maximum(tl.load(starts_ptr + row).to(tl.int32), 0)
    else:
        start = end * 0
    end = tl.maximum(tl.minimum(end, n_cols), start)

    # Pass A: the selected scores, the boundary score T and the split of the
    # selection into "above T" and "equal to T".
    slots = tl.arange(0, K_P2)
    in_k = slots < K
    sel = tl.load(irow + slots, mask=in_k, other=-1)
    valid = in_k & (sel >= 0)
    col = sel + start if RELATIVE else sel
    key = _ordered_key(tl.load(lrow + col, mask=valid, other=0.0))
    thr = tl.min(tl.where(valid, key, 0x7FFFFFFF), axis=0)
    is_gt = valid & (key > thr)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    n_gt = tl.sum(is_gt.to(tl.int32), axis=0)
    need = n_valid - n_gt

    # Pass B: how many entries of the whole window equal T.
    n_eq = n_valid * 0
    for c0 in range(start, end, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        m = cols < end
        k2 = _ordered_key(tl.load(lrow + cols, mask=m, other=0.0))
        n_eq += tl.sum((m & (k2 == thr)).to(tl.int32), axis=0)

    # Pass C, only where the tied group straddles the boundary.
    if (n_valid > 0) & (n_eq > need):
        # Every thread's read of `sel` is complete before any slot is
        # overwritten (the reductions above already imply it; be explicit).
        tl.debug_barrier()
        pos = tl.cumsum(is_gt.to(tl.int32), axis=0) - 1
        tl.store(irow + pos, sel, mask=is_gt)
        tl.store(irow + slots, -1, mask=in_k & (slots >= n_valid))
        taken = n_valid * 0
        c0 = start
        while (c0 < end) & (taken < need):
            cols = c0 + tl.arange(0, BLOCK)
            m = cols < end
            k2 = _ordered_key(tl.load(lrow + cols, mask=m, other=0.0))
            is_eq = m & (k2 == thr)
            rank = taken + tl.cumsum(is_eq.to(tl.int32), axis=0) - 1
            out = cols - start if RELATIVE else cols
            tl.store(irow + n_gt + rank, out, mask=is_eq & (rank < need))
            taken += tl.sum(is_eq.to(tl.int32), axis=0)
            c0 += BLOCK

    # Optional: sort the row ascending with the -1 fill last, in registers
    # (VLLM_GLM5_TOPK_SORTED together with the tie fix), so the order is a
    # function of the set too; same bytes as sort_selected_topk_.
    if SORT:
        tl.debug_barrier()
        big = 0x7FFFFFFF
        v = tl.load(irow + slots, mask=in_k, other=big)
        v = tl.where(v < 0, big, v)
        v = tl.sort(v)
        v = tl.where(v == big, -1, v)
        tl.store(irow + slots, v, mask=in_k)


# ---------------------------------------------------------------------------
# Split path for small batches of wide rows (tie fix + sort only)
# ---------------------------------------------------------------------------
#
# One program per row leaves most SMs idle when a decode batch has a handful
# of rows over a long context (4 rows x 32,768 pools at 131K), and the
# contested-row repair then scans the row serially. With ``sort`` the result
# is fully determined by the row's threshold T (the smallest selected score),
# the selected count and ``need`` (selected entries equal to T): the canonical
# set is every entry above T plus the ``need`` lowest-index entries equal to
# T, and in ascending order each entry's slot is its running count. So two
# launches over (row, chunk) produce the sorted canonical row directly:
#
#   count  per chunk: entries above / equal to T (chunk 0 also records T,
#          the selected count and need); reads only.
#   write  per chunk: exclusive prefix of the counts, then one ascending scan
#          writing each chunk entry at its final slot (one int32 cumsum per
#          tile carries both running counts); chunk 0 writes the -1 tail.
#          Never reads the selection, so it may overwrite it.
#
# Rows whose window is no longer than k and fully selected are written as the
# identity. The bytes equal the one-program kernel with SORT (and so
# topk_tiefix_ + sort_selected_topk_). Taken only when enabled with
# VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS and the batch has at most that many rows.

# Split geometry: at most SPLIT_MAX_CHUNKS chunks per row, each a power of
# two >= SPLIT_CHUNK_MIN columns; tile and warps per chunk program.
SPLIT_CHUNK_MIN = 1024
SPLIT_MAX_CHUNKS = 64
SPLIT_BLOCK = 1024
SPLIT_NUM_WARPS = 4
# Narrower logits never split (one program per row is cheaper there).
SPLIT_MIN_COLS = 16384
_SCRATCH_INFO = 4  # thr, n_valid, need, length


@triton.jit
def _row_window(starts_ptr, ends_ptr, row, n_cols, HAS_STARTS: tl.constexpr):
    end = tl.load(ends_ptr + row).to(tl.int32)
    if HAS_STARTS:
        start = tl.maximum(tl.load(starts_ptr + row).to(tl.int32), 0)
    else:
        start = end * 0
    end = tl.maximum(tl.minimum(end, n_cols), start)
    return start, end


@triton.jit
def _selection_count(irow, K: tl.constexpr, K_P2: tl.constexpr):
    slots = tl.arange(0, K_P2)
    sel = tl.load(irow + slots, mask=slots < K, other=-1)
    return tl.sum(((slots < K) & (sel >= 0)).to(tl.int32), axis=0)


@triton.jit
def _selection_threshold(
    lrow, irow, start, K: tl.constexpr, K_P2: tl.constexpr, RELATIVE: tl.constexpr
):
    """(T, n_valid, need): the smallest selected key, the selected count and
    how many selected entries equal T."""
    slots = tl.arange(0, K_P2)
    in_k = slots < K
    sel = tl.load(irow + slots, mask=in_k, other=-1)
    valid = in_k & (sel >= 0)
    col = sel + start if RELATIVE else sel
    key = _ordered_key(tl.load(lrow + col, mask=valid, other=0.0))
    thr = tl.min(tl.where(valid, key, 0x7FFFFFFF), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    n_gt = tl.sum((valid & (key > thr)).to(tl.int32), axis=0)
    return thr, n_valid, n_valid - n_gt


@triton.jit
def _scan_write(
    lrow,
    irow,
    start,
    c_lo,
    c_hi,
    thr,
    n_valid,
    need,
    gt_seen,
    eq_seen,
    BLOCK: tl.constexpr,
    RELATIVE: tl.constexpr,
):
    """Write the canonical entries of columns [c_lo, c_hi) at their final
    ascending slots, given the above/equal counts of the columns before c_lo.
    Returns the counts including this range."""
    c0 = c_lo
    while (c0 < c_hi) & (gt_seen + tl.minimum(eq_seen, need) < n_valid):
        cols = c0 + tl.arange(0, BLOCK)
        m = cols < c_hi
        key = _ordered_key(tl.load(lrow + cols, mask=m, other=0.0))
        gt = m & (key > thr)
        eq = m & (key == thr)
        packed = gt.to(tl.int32) + (eq.to(tl.int32) << 16)
        inc = tl.cumsum(packed, axis=0)
        eq_run = eq_seen + (inc >> 16)
        pos = gt_seen + (inc & 0xFFFF) + tl.minimum(eq_run, need) - 1
        take = gt | (eq & (eq_run <= need))
        out = cols - start if RELATIVE else cols
        tl.store(irow + pos, out, mask=take & (pos < n_valid))
        tot = tl.sum(packed, axis=0)
        gt_seen += tot & 0xFFFF
        eq_seen += tot >> 16
        c0 += BLOCK
    return gt_seen, eq_seen


@triton.jit(do_not_specialize=["stride_l", "stride_i", "n_cols", "chunk"])
def _topk_tiefix_split_count_kernel(
    logits_ptr,
    idx_ptr,
    starts_ptr,
    ends_ptr,
    scratch_ptr,
    stride_l,
    stride_i,
    n_cols,
    chunk,
    K: tl.constexpr,
    K_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    NCH: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    RELATIVE: tl.constexpr,
):
    """Split pass 1, grid (rows, NCH): per chunk, entries above and equal to
    T; chunk 0 also records (T, n_valid, need, length). Reads only."""
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    lrow = logits_ptr + row * stride_l
    irow = idx_ptr + row * stride_i
    srow = scratch_ptr + row * (2 * NCH + 4)
    start, end = _row_window(starts_ptr, ends_ptr, row, n_cols, HAS_STARTS)
    length = end - start
    c_lo = start + ch * chunk
    c_hi = tl.minimum(c_lo + chunk, end)

    n_gt = length * 0
    n_eq = length * 0
    if c_lo < end:
        n_sel = _selection_count(irow, K, K_P2)
        if (length <= K) & (n_sel == length):
            # Whole window selected (only chunk 0 is non-empty, chunk >= K):
            # the write pass emits the identity.
            tl.store(srow + 2 * NCH + 3, -1)
        else:
            thr, n_valid, need = _selection_threshold(
                lrow, irow, start, K, K_P2, RELATIVE
            )
            if ch == 0:
                tl.store(srow + 2 * NCH + 0, thr)
                tl.store(srow + 2 * NCH + 1, n_valid)
                tl.store(srow + 2 * NCH + 2, need)
                tl.store(srow + 2 * NCH + 3, length)
            for c0 in range(c_lo, c_hi, BLOCK):
                cols = c0 + tl.arange(0, BLOCK)
                m = cols < c_hi
                key = _ordered_key(tl.load(lrow + cols, mask=m, other=0.0))
                n_gt += tl.sum((m & (key > thr)).to(tl.int32), axis=0)
                n_eq += tl.sum((m & (key == thr)).to(tl.int32), axis=0)
    elif ch == 0:
        # Empty window: nothing selected.
        tl.store(srow + 2 * NCH + 0, 0x7FFFFFFF)
        tl.store(srow + 2 * NCH + 1, 0)
        tl.store(srow + 2 * NCH + 2, 0)
        tl.store(srow + 2 * NCH + 3, 0)
    tl.store(srow + ch, n_gt)
    tl.store(srow + NCH + ch, n_eq)


@triton.jit(
    do_not_specialize=["stride_l", "stride_i", "n_cols", "chunk", "n_chunks"]
)
def _topk_tiefix_split_write_kernel(
    logits_ptr,
    idx_ptr,
    starts_ptr,
    ends_ptr,
    scratch_ptr,
    stride_l,
    stride_i,
    n_cols,
    chunk,
    n_chunks,
    K: tl.constexpr,
    K_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    NCH: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    RELATIVE: tl.constexpr,
):
    """Split pass 2, grid (rows, NCH): write each chunk's entries at their
    final slots from the exclusive prefix of pass 1's counts; chunk 0 writes
    the identity rows and the -1 tail. Never reads the selection."""
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    lrow = logits_ptr + row * stride_l
    irow = idx_ptr + row * stride_i
    srow = scratch_ptr + row * (2 * NCH + 4)
    start, end = _row_window(starts_ptr, ends_ptr, row, n_cols, HAS_STARTS)
    slots = tl.arange(0, K_P2)
    in_k = slots < K
    thr = tl.load(srow + 2 * NCH + 0)
    n_valid = tl.load(srow + 2 * NCH + 1)
    need = tl.load(srow + 2 * NCH + 2)
    length = tl.load(srow + 2 * NCH + 3)

    if length < 0:
        if ch == 0:
            n = end - start
            ident = slots if RELATIVE else slots + start
            tl.store(irow + slots, tl.where(slots < n, ident, -1), mask=in_k)
    else:
        chs = tl.arange(0, NCH)
        live = chs < n_chunks
        cgt = tl.load(srow + chs, mask=live, other=0)
        ceq = tl.load(srow + NCH + chs, mask=live, other=0)
        before = chs < ch
        gt_before = tl.sum(tl.where(before, cgt, 0), axis=0)
        eq_before = tl.sum(tl.where(before, ceq, 0), axis=0)
        c_lo = start + ch * chunk
        c_hi = tl.minimum(c_lo + chunk, end)
        if c_lo < end:
            _scan_write(
                lrow, irow, start, c_lo, c_hi, thr, n_valid, need,
                gt_before, eq_before, BLOCK, RELATIVE,
            )
        if ch == 0:
            tot_gt = tl.sum(cgt, axis=0)
            tot_eq = tl.sum(ceq, axis=0)
            written = tl.minimum(tot_gt + tl.minimum(tot_eq, need), n_valid)
            tl.store(irow + slots, -1, mask=in_k & (slots >= written))



def _split_chunk(n_cols: int, k: int) -> int:
    """Chunk width: a power of two, at least SPLIT_CHUNK_MIN, at least k (a
    fully selected short row must sit in chunk 0) and wide enough that the
    row fits in SPLIT_MAX_CHUNKS chunks."""
    per = triton.next_power_of_2(triton.cdiv(n_cols, SPLIT_MAX_CHUNKS))
    return max(SPLIT_CHUNK_MIN, triton.next_power_of_2(k), per)


def _topk_tiefix_sorted_split(
    logits, topk_indices, starts, ends, has_starts, relative, block, num_warps
):
    num_rows, k = topk_indices.shape
    n_cols = logits.shape[1]
    chunk = _split_chunk(n_cols, k)
    assert chunk >= k, f"split chunk {chunk} narrower than k={k}"
    nch = SPLIT_MAX_CHUNKS
    grid = (num_rows, triton.cdiv(n_cols, chunk))
    assert grid[1] <= nch
    scratch = torch.empty(
        (num_rows, 2 * nch + _SCRATCH_INFO), dtype=torch.int32, device=logits.device
    )
    common = dict(
        K=k,
        K_P2=triton.next_power_of_2(k),
        BLOCK=block or SPLIT_BLOCK,
        NCH=nch,
        HAS_STARTS=has_starts,
        RELATIVE=relative,
        num_warps=num_warps or SPLIT_NUM_WARPS,
    )
    args = (
        logits,
        topk_indices,
        starts,
        ends,
        scratch,
        logits.stride(0),
        topk_indices.stride(0),
        n_cols,
        chunk,
    )
    _topk_tiefix_split_count_kernel[grid](*args, **common)
    # Chunks past grid[1] are never launched; the write pass masks them.
    _topk_tiefix_split_write_kernel[grid](*args, grid[1], **common)
    return topk_indices


def use_split(num_rows: int, n_cols: int, split_rows: int) -> bool:
    """Whether the tie fix + sort takes the split path: enabled (split_rows
    > 0), at most split_rows rows, and logits at least SPLIT_MIN_COLS wide."""
    return 0 < num_rows <= split_rows and n_cols >= SPLIT_MIN_COLS


_WARM_KS = (512, 2048)


def warm_tiefix() -> None:
    """Compile the kernel variants the indexer uses (K 512 for kpool 4 and 2048
    for kpool 1; decode = absolute, prefill = relative with starts) before any
    CUDA graph capture; Triton otherwise compiles on first launch. A no-op
    without CUDA or inside a capture."""
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    for k in _WARM_KS:
        logits = torch.zeros((1, k + 1), dtype=torch.float32, device=dev)
        ends = torch.full((1,), k + 1, dtype=torch.int32, device=dev)
        starts = torch.zeros((1,), dtype=torch.int32, device=dev)
        for relative in (False, True):
            for sort, split in ((False, False), (True, False), (True, True)):
                ids = torch.arange(k, dtype=torch.int32, device=dev).reshape(1, k)
                topk_tiefix_(
                    logits,
                    ids,
                    row_ends=ends,
                    row_starts=starts if relative else None,
                    relative=relative,
                    sort=sort,
                    split=split,
                )


def topk_tiefix_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
    sort: bool = False,
    split: bool = False,
    block: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Make a top-k selection's set canonical on exact ties, in place.
    With ``sort`` each row is then sorted ascending, -1 fill last (the bytes
    of ``sort_selected_topk_``), in the same launch.

    Args:
        logits: (num_rows, num_cols) fp32, stride(1) == 1. The scores the
            selection in ``topk_indices`` was taken from.
        topk_indices: (num_rows, k) int32, stride(1) == 1 (may be a column
            slice of a wider buffer): an exact top-k per row, -1 filled.
        row_ends: (num_rows,) int, exclusive right boundary of each row.
        row_starts: (num_rows,) int, inclusive left boundary; default 0.
        relative: the indices are relative to ``row_starts`` (the
            ``top_k_per_row_prefill`` convention).
        sort: also sort each row ascending, -1 last.
        split: with ``sort``, run the two-launch (row, chunk) split path
            instead of one program per row (same bytes); see ``use_split``.
        block: scan tile override (default TIEFIX_BLOCK, SPLIT_BLOCK).
        num_warps: warps per program override (default TIEFIX_NUM_WARPS,
            SPLIT_NUM_WARPS).

    Returns:
        ``topk_indices``. Rows whose boundary is not contested are unchanged.

    """
    assert logits.dim() == 2 and topk_indices.dim() == 2
    assert logits.dtype == torch.float32, f"expected fp32, got {logits.dtype}"
    assert topk_indices.dtype == torch.int32
    assert logits.stride(1) == 1 and topk_indices.stride(1) == 1
    num_rows, k = topk_indices.shape
    assert logits.shape[0] >= num_rows
    if num_rows == 0 or k == 0:
        return topk_indices
    ends = row_ends.reshape(-1)
    assert ends.numel() >= num_rows
    starts = ends if row_starts is None else row_starts.reshape(-1)
    if split and sort:
        return _topk_tiefix_sorted_split(
            logits,
            topk_indices,
            starts,
            ends,
            row_starts is not None,
            relative,
            block,
            num_warps,
        )
    _topk_tiefix_kernel[(num_rows,)](
        logits,
        topk_indices,
        starts,
        ends,
        logits.stride(0),
        topk_indices.stride(0),
        logits.shape[1],
        K=k,
        K_P2=triton.next_power_of_2(k),
        BLOCK=block or TIEFIX_BLOCK,
        HAS_STARTS=row_starts is not None,
        RELATIVE=relative,
        SORT=sort,
        num_warps=num_warps or TIEFIX_NUM_WARPS,
    )
    return topk_indices
