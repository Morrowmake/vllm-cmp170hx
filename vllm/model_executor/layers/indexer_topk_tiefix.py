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
            ids = torch.arange(k, dtype=torch.int32, device=dev).reshape(1, k)
            topk_tiefix_(
                logits,
                ids,
                row_ends=ends,
                row_starts=starts if relative else None,
                relative=relative,
            )


def topk_tiefix_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
    block: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Make a top-k selection's set canonical on exact ties, in place.

    Args:
        logits: (num_rows, num_cols) fp32, stride(1) == 1. The scores the
            selection in ``topk_indices`` was taken from.
        topk_indices: (num_rows, k) int32, stride(1) == 1 (may be a column
            slice of a wider buffer): an exact top-k per row, -1 filled.
        row_ends: (num_rows,) int, exclusive right boundary of each row.
        row_starts: (num_rows,) int, inclusive left boundary; default 0.
        relative: the indices are relative to ``row_starts`` (the
            ``top_k_per_row_prefill`` convention).
        block: scan tile override (default TIEFIX_BLOCK).
        num_warps: warps per row program override (default TIEFIX_NUM_WARPS).

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
        num_warps=num_warps or TIEFIX_NUM_WARPS,
    )
    return topk_indices
