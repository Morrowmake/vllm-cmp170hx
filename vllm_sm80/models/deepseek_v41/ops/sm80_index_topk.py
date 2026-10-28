# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, stable radix selection for paged DeepSeek index scores.

Four byte passes find the exact FP32 cutoff. Segment counts and a prefix
scatter emit ascending positions, taking the earliest equal scores. No heap,
full-context sort, tensor-driven host loop or inter-CTA waiting is used.
Every pass reads only the segments that hold a visible column of its row
(``seg * SEGMENT < min(n_vis, width)``); the others hold no candidate, so
their programs return at once and their scratch is never read. The grid
keeps the capacity width (graph-safe); the memory traffic follows the
row's visible count instead.
The score spool remains live until BOTH token and candidate selectors finish.

With ``VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH`` a call of at most
``ONE_LAUNCH_MAX_ROWS`` rows is one launch (``_select_one_launch``) instead of
fifteen, with the same output bytes. A row's visible columns are cut into
parts of ``PART`` columns, one program each. A row of one part is selected
by its program alone: the four byte passes and the scatter loop over the
part in ``BLOCK`` columns, the cutoff and the running counts stay in
registers. A longer row's programs each keep their part's exact top
``min(n_vis, k)`` (score descending, earliest first) in ascending order in
workspace slots; the last of them to finish (a ticket counter, no waiting)
selects the row's top from those candidates in part order, which is the
row's column order. The row's selection is inside the union of its parts'
selections (a column in the row's top is in its part's top), so both levels
pick the same columns as the multi-kernel path.

Bounds. A program's work is serial: each byte pass reads ``4 n`` score bytes
of its ``n <= PART`` columns (the first pass reads the producer histogram
instead, ``n`` bytes) and runs a 256-bin histogram of about 2.7 warp
instructions a column on sm_80, so ``PART`` columns cost five 32 KiB reads
per program, near the fifteen launches of the multi-kernel path at decode.
The merge reads ``8 parts k`` candidate bytes per pass (128 KiB for
``k = 512`` over 262144 columns), the long-context limit of this path.
Rows of at most ``k`` visible columns keep every column in order and skip
the passes. A call is one launch when it has at most
``ONE_LAUNCH_MAX_ROWS`` rows (decode, verify and draft calls) and its
``parts * k`` candidate slots fit the ``PART_SLOTS`` reserved per part of
the workspace capacity (``k <= 512`` at any width, ``k <= 2048`` up to a
quarter of it); other calls keep the multi-kernel path.
"""
import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton

SEGMENT = 1024
RADIX = 256
ONE_LAUNCH_MAX_ROWS = 48
PART = 8192
BLOCK = 2048
# Candidate slots reserved per part: a call of ``parts * k`` slots fits.
PART_SLOTS = 512
_SEGMENT = tl.constexpr(SEGMENT)
_PART = tl.constexpr(PART)


@triton.jit
def float_key(x):
    # Numerical order, including -inf < -FLT_MAX; both zeros tie.
    bits = tl.where(x == 0.0, 0.0, x).to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits & 0x80000000) != 0, 0xffffffff, 0x80000000).to(tl.uint32)


@triton.jit
def _selection_init(out, state, nvis, k: tl.constexpr, os: tl.constexpr,
                    NS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * B + tl.arange(0, B)
    tl.store(out + row * os + j, -1, j < k)
    if tl.program_id(1) == 0:
        tl.store(state + row * 3, 0)
        tl.store(state + row * 3 + 1, tl.minimum(tl.load(nvis + row), k))
        tl.store(state + row * 3 + 2, 0)


@triton.jit
def _radix_hist(scores, nvis, state, hist, width, ss: tl.constexpr,
                NS: tl.constexpr, SHIFT: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    visible = tl.minimum(tl.load(nvis + row), width)
    if seg * B < visible:
        col = seg * B + tl.arange(0, B)
        valid = col < visible
        x = tl.load(scores + row * ss + col, col < width, float('-inf'))
        key = float_key(x)
        prefix = tl.load(state + row * 3).to(tl.uint32)
        if SHIFT < 24:
            valid = valid & ((key >> (SHIFT + 8)) == (prefix >> (SHIFT + 8)))
        byte = ((key >> SHIFT) & 255).to(tl.int32)
        counts = tl.histogram(tl.where(valid, byte, 256), 256)
        tl.store(hist + (row * NS + seg) * 256 + tl.arange(0, 256), counts)


@triton.jit
def _hist_reduce(hist, totals, nvis, width, NS: tl.constexpr, BS: tl.constexpr,
                 B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    byte = tl.program_id(1)
    seg = tl.arange(0, BS)
    used = (tl.minimum(tl.load(nvis + row), width) + B - 1) // B
    c = tl.load(hist + (row * NS + seg) * 256 + byte, (seg < NS) & (seg < used), 0)
    tl.store(totals + row * 256 + byte, tl.sum(c, 0))


@triton.jit
def _radix_cutoff(totals, state, SHIFT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    byte = tl.arange(0, 256)
    counts = tl.load(totals + row * 256 + byte)
    higher = tl.sum(counts, 0) - tl.cumsum(counts, 0)
    remaining = tl.load(state + row * 3 + 1)
    wins = (higher < remaining) & (higher + counts >= remaining)
    bucket = tl.max(tl.where(wins, byte, 0), 0).to(tl.uint32)
    above = tl.sum(tl.where(wins, higher, 0), 0)
    prefix = tl.load(state + row * 3).to(tl.uint32)
    tl.store(state + row * 3, prefix | (bucket << SHIFT))
    tl.store(state + row * 3 + 1, remaining - above)


@triton.jit
def _selection_counts(scores, nvis, state, counts, width, ss: tl.constexpr,
                      NS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    visible = tl.minimum(tl.load(nvis + row), width)
    if seg * B < visible:
        col = seg * B + tl.arange(0, B)
        valid = col < visible
        x = tl.load(scores + row * ss + col, col < width, float('-inf'))
        key = float_key(x)
        cutoff = tl.load(state + row * 3).to(tl.uint32)
        tl.store(counts + (row * NS + seg) * 2,
                 tl.sum((valid & (key > cutoff)).to(tl.int32), 0))
        tl.store(counts + (row * NS + seg) * 2 + 1,
                 tl.sum((valid & (key == cutoff)).to(tl.int32), 0))


@triton.jit
def _selection_prefix(counts, prefixes, state, nvis, width, NS: tl.constexpr,
                      BS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.arange(0, BS)
    used = (seg < NS) & (seg < (tl.minimum(tl.load(nvis + row), width) + B - 1) // B)
    gt = tl.load(counts + (row * NS + seg) * 2, used, 0)
    eq = tl.load(counts + (row * NS + seg) * 2 + 1, used, 0)
    tl.store(prefixes + (row * NS + seg) * 2, tl.cumsum(gt, 0) - gt, seg < NS)
    tl.store(prefixes + (row * NS + seg) * 2 + 1, tl.cumsum(eq, 0) - eq, seg < NS)


@triton.jit
def _selection_scatter(scores, nvis, state, prefixes, out, width,
                       ss: tl.constexpr, os: tl.constexpr, NS: tl.constexpr,
                       K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    visible = tl.minimum(tl.load(nvis + row), width)
    if seg * B < visible:
        col = seg * B + tl.arange(0, B)
        valid = col < visible
        x = tl.load(scores + row * ss + col, col < width, float('-inf'))
        key = float_key(x)
        cutoff = tl.load(state + row * 3).to(tl.uint32)
        equal_budget = tl.load(state + row * 3 + 1)
        gt_before = tl.load(prefixes + (row * NS + seg) * 2)
        eq_before = tl.load(prefixes + (row * NS + seg) * 2 + 1)
        equal = valid & (key == cutoff)
        keep = valid & ((key > cutoff) | (equal & (eq_before + tl.cumsum(equal.to(tl.int32), 0) <= equal_budget)))
        index = gt_before + tl.minimum(eq_before, equal_budget) + tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(out + row * os + index, col.to(tl.int32), keep & (index < K))


@triton.jit
def _list_block(scores, keys, cols, kept, start, lo, hi, stride, B: tl.constexpr,
                MERGE: tl.constexpr):
    """Keys, columns and validity of list positions ``start + [0, B)``: score
    columns ``lo + i < hi``, or (MERGE) candidate slot ``i < hi`` of part
    ``i // stride``, valid below that part's kept count."""
    i = start + tl.arange(0, B)
    if MERGE:
        part = i // stride
        valid = (i < hi) & (i - part * stride < tl.load(kept + part, i < hi, 0, cache_modifier='.cg'))
        key = tl.load(keys + i, valid, 0, cache_modifier='.cg').to(tl.uint32, bitcast=True)
        col = tl.load(cols + i, valid, 0, cache_modifier='.cg')
    else:
        col = lo + i
        valid = col < hi
        key = float_key(tl.load(scores + col, valid, float('-inf')))
    return key, col, valid


@triton.jit
def _list_pass(scores, keys, cols, kept, hist, lo, hi, stride, prefix, remaining,
               SHIFT: tl.constexpr, B: tl.constexpr, MERGE: tl.constexpr,
               FIRST_HIST: tl.constexpr):
    """One byte of the cutoff, as ``_radix_hist``/``_hist_reduce``/``_radix_cutoff``."""
    byte = tl.arange(0, 256)
    counts = tl.zeros((256,), tl.int32)
    if FIRST_HIST and SHIFT == 24:
        # The producer's per-segment histograms of this part's columns.
        seg = lo // _SEGMENT + tl.arange(0, _PART // _SEGMENT)
        c = tl.load(hist + seg[:, None] * 256 + byte[None, :],
                    (seg < (hi + _SEGMENT - 1) // _SEGMENT)[:, None], 0)
        counts += tl.sum(c, 0)
    else:
        for start in range(0, hi - lo, B):
            key, col, valid = _list_block(scores, keys, cols, kept, start, lo, hi,
                                          stride, B, MERGE)
            if SHIFT < 24:
                valid = valid & ((key >> (SHIFT + 8)) == (prefix >> (SHIFT + 8)))
            b = ((key >> SHIFT) & 255).to(tl.int32)
            counts += tl.histogram(tl.where(valid, b, 256), 256)
    higher = tl.sum(counts, 0) - tl.cumsum(counts, 0)
    wins = (higher < remaining) & (higher + counts >= remaining)
    prefix = prefix | (tl.max(tl.where(wins, byte, 0), 0).to(tl.uint32) << SHIFT)
    remaining = remaining - tl.sum(tl.where(wins, higher, 0), 0)
    return prefix, remaining


@triton.jit
def _list_select(scores, keys, cols, kept, hist, lo, hi, stride, total, remaining,
                 dst_cols, dst_keys, limit, B: tl.constexpr, MERGE: tl.constexpr,
                 FIRST_HIST: tl.constexpr, KEYS: tl.constexpr):
    """Exact top ``remaining`` of a list of ``total`` valid positions (key
    descending, earliest first), written in list order to ``dst_cols[:limit]``
    (and their keys to ``dst_keys``). Returns how many are kept. With
    ``total <= remaining`` every position is kept, as the passes would decide."""
    prefix = tl.full((), 0, tl.uint32)
    if total > remaining:
        prefix, remaining = _list_pass(scores, keys, cols, kept, hist, lo, hi, stride,
                                       prefix, remaining, 24, B, MERGE, FIRST_HIST)
        prefix, remaining = _list_pass(scores, keys, cols, kept, hist, lo, hi, stride,
                                       prefix, remaining, 16, B, MERGE, FIRST_HIST)
        prefix, remaining = _list_pass(scores, keys, cols, kept, hist, lo, hi, stride,
                                       prefix, remaining, 8, B, MERGE, FIRST_HIST)
        prefix, remaining = _list_pass(scores, keys, cols, kept, hist, lo, hi, stride,
                                       prefix, remaining, 0, B, MERGE, FIRST_HIST)
    gt_before = tl.full((), 0, tl.int32)
    eq_before = tl.full((), 0, tl.int32)
    for start in range(0, hi - lo, B):
        key, col, valid = _list_block(scores, keys, cols, kept, start, lo, hi, stride,
                                      B, MERGE)
        above = valid & (key > prefix)
        equal = valid & (key == prefix)
        keep = above | (equal & (eq_before + tl.cumsum(equal.to(tl.int32), 0) <= remaining))
        index = gt_before + tl.minimum(eq_before, remaining) + tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(dst_cols + index, col, keep & (index < limit))
        if KEYS:
            tl.store(dst_keys + index, key.to(tl.int32, bitcast=True), keep & (index < limit))
        gt_before += tl.sum(above.to(tl.int32), 0)
        eq_before += tl.sum(equal.to(tl.int32), 0)
    return gt_before + tl.minimum(eq_before, remaining)


@triton.jit
def _pad(out, kept, k, B: tl.constexpr):
    for start in range(0, k, B):
        j = start + tl.arange(0, B)
        tl.store(out + j, -1, (j >= kept) & (j < k))


@triton.jit
def _select_one_launch(scores, nvis, hist, keys, cols, kept, tickets, out, width, k,
                       ss: tl.constexpr, os: tl.constexpr, NS: tl.constexpr,
                       SLOTS: tl.constexpr, PARTS: tl.constexpr, BP: tl.constexpr,
                       FIRST_HIST: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    n = tl.load(nvis + row)
    visible = tl.minimum(n, width)
    remaining = tl.minimum(n, k)
    parts = (visible + _PART - 1) // _PART
    row_scores = scores + row * ss
    row_hist = hist + row * NS * 256
    row_out = out + row * os
    if part == 0 and parts <= 1:
        if visible <= remaining:
            for start in range(0, k, B):
                j = start + tl.arange(0, B)
                tl.store(row_out + j, tl.where(j < visible, j, -1), j < k)
        else:
            got = _list_select(row_scores, keys, cols, kept, row_hist, 0, visible, 1,
                               visible, remaining, row_out, row_out, k, B, False,
                               FIRST_HIST, False)
            _pad(row_out, got, k, B)
    elif part < parts:
        lo = part * _PART
        hi = tl.minimum(lo + _PART, visible)
        row_keys = keys + row * SLOTS
        row_cols = cols + row * SLOTS
        got = _list_select(row_scores, keys, cols, kept, row_hist, lo, hi, 1, hi - lo,
                           remaining, row_cols + part * k, row_keys + part * k, k, B,
                           False, FIRST_HIST, True)
        tl.store(kept + row * PARTS + part, got)
        tl.debug_barrier()
        ticket = tl.atomic_add(tickets + row, 1, sem='acq_rel', scope='gpu')
        if ticket == parts - 1:
            p = tl.arange(0, BP)
            total = tl.sum(tl.load(kept + row * PARTS + p, p < parts, 0,
                                   cache_modifier='.cg'), 0)
            got = _list_select(row_scores, row_keys, row_cols, kept + row * PARTS,
                               row_hist, 0, parts * k, k, total, remaining, row_out,
                               row_out, k, B, True, False, False)
            _pad(row_out, got, k, B)
            tl.store(tickets + row, 0)


class SelectionWorkspace:
    """Fixed capacity, one independently executable stream/runner's scratch."""
    def __init__(self, rows, width, device):
        self.rows, self.width = rows, width
        self.segments = triton.cdiv(max(width, 1), SEGMENT)
        self.hist = torch.empty((rows, self.segments, RADIX), dtype=torch.int32, device=device)
        self.totals = torch.empty((rows, RADIX), dtype=torch.int32, device=device)
        self.state = torch.empty((rows, 3), dtype=torch.int64, device=device)
        self.counts = torch.empty((rows, self.segments, 2), dtype=torch.int32, device=device)
        self.prefixes = torch.empty_like(self.counts)
        # One-launch path: each part's candidates, kept counts, and a ticket
        # per row that its last part resets to zero.
        self.parts = triton.cdiv(max(width, 1), PART)
        self.slots = self.parts * PART_SLOTS
        self.cand_keys = torch.empty((rows, self.slots), dtype=torch.int32, device=device)
        self.cand_cols = torch.empty_like(self.cand_keys)
        self.kept = torch.empty((rows, self.parts), dtype=torch.int32, device=device)
        self.tickets = torch.zeros(rows, dtype=torch.int32, device=device)

    @staticmethod
    def bytes(rows, width):
        segments = triton.cdiv(max(width, 1), SEGMENT)
        parts = triton.cdiv(max(width, 1), PART)
        return rows * (segments * (RADIX + 4) * 4 + RADIX * 4 + 3 * 8
                       + parts * (2 * PART_SLOTS + 1) * 4 + 4)


def one_launch(rows, width, k, workspace):
    """Whether ``select`` makes this call one launch."""
    parts = triton.cdiv(max(width, 1), PART)
    return bool(envs.VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH
                and rows <= ONE_LAUNCH_MAX_ROWS
                and (parts == 1 or parts * k <= workspace.slots))


def select(scores, n_vis, out, workspace, *, first_histogram=False):
    """Write canonical IDs directly into the persistent sharing buffer.

    ``first_histogram`` means the score producer has populated the high-byte
    histogram in exactly this workspace layout. Visibility is bounded by the
    caller; nonfinite arithmetic is rejected by validation, not repaired here.
    """
    rows, width = scores.shape
    if rows > workspace.rows or width > workspace.width:
        raise ValueError('selection exceeds reserved capacity')
    if rows == 0:
        return
    k = out.shape[1]
    if k == 0:
        return
    ns = triton.cdiv(max(width, 1), SEGMENT)
    if one_launch(rows, width, k, workspace):
        parts = triton.cdiv(max(width, 1), PART)
        _select_one_launch[(rows, parts)](scores, n_vis, workspace.hist,
            workspace.cand_keys, workspace.cand_cols, workspace.kept, workspace.tickets,
            out, width, k, scores.stride(0), out.stride(0), ns,
            workspace.slots, workspace.parts, triton.next_power_of_2(workspace.parts),
            first_histogram, BLOCK, num_warps=8)
        return
    _selection_init[(rows, triton.cdiv(k, 256))](out, workspace.state, n_vis, k,
        out.stride(0), ns, 256)
    for shift in (24, 16, 8, 0):
        if shift != 24 or not first_histogram:
            _radix_hist[(rows, ns)](scores, n_vis, workspace.state, workspace.hist,
                width, scores.stride(0), ns, shift, SEGMENT)
        _hist_reduce[(rows, 256)](workspace.hist, workspace.totals, n_vis, width, ns,
            triton.next_power_of_2(ns), SEGMENT)
        _radix_cutoff[(rows,)](workspace.totals, workspace.state, shift)
    _selection_counts[(rows, ns)](scores, n_vis, workspace.state, workspace.counts,
        width, scores.stride(0), ns, SEGMENT)
    _selection_prefix[(rows,)](workspace.counts, workspace.prefixes, workspace.state,
        n_vis, width, ns, triton.next_power_of_2(ns), SEGMENT)
    _selection_scatter[(rows, ns)](scores, n_vis, workspace.state, workspace.prefixes,
        out, width, scores.stride(0), out.stride(0), ns, k, SEGMENT)
