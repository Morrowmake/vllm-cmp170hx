# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection-logic tests for the canonical DSA indexer top-k.

The stock decode and prefill top-k kernels (``persistent_topk`` on sm_80,
``topKPerRowJob`` in the per-row fallback) select the correct top-k *scores*
but leave undefined which of several exactly-equal scores wins a contested
slot: both place threshold-bin entries with an ``atomicAdd``. The indexer
logits are a dot product against an fp8-e4m3 K cache, which puts many keys on
the same grid point, so ties are common and the selected index *set* -- hence
the KV page set, hence the attention output -- can differ between two
identical calls.

``canonical_topk`` fixes the order at (score descending, column index
ascending). These tests pin that logic on CPU; no CUDA, no engine.
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.indexer_topk import (
    canonical_topk,
    use_canonical_topk,
)


def reference_canonical_topk(
    logits: torch.Tensor,
    k: int,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
    identity_when_short: bool = False,
) -> torch.Tensor:
    """Independent per-row reference: a stable sort on (-score, column)."""
    rows, _ = logits.shape
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        start = 0 if row_starts is None else int(row_starts[r])
        end = int(row_ends[r])
        window = list(range(start, max(start, end)))
        if identity_when_short and len(window) <= k:
            picked = window
        else:
            picked = sorted(window, key=lambda c: (-float(logits[r, c]), c))[:k]
        for slot, col in enumerate(picked):
            out[r, slot] = col - start if relative else col
    return out


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_TOPK_CANONICAL is False
    use_canonical_topk.cache_clear()
    assert use_canonical_topk() is False


def test_flag_is_read_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_CANONICAL", "1")
    use_canonical_topk.cache_clear()
    try:
        assert use_canonical_topk() is True
    finally:
        use_canonical_topk.cache_clear()


def test_ties_resolve_to_the_lowest_index():
    # One row, every score identical: the answer must be 0..k-1.
    logits = torch.zeros(1, 64, dtype=torch.float32)
    out = canonical_topk(logits, 8, row_ends=torch.tensor([64], dtype=torch.int32))
    assert out.tolist() == [list(range(8))]


def test_tie_at_the_boundary_takes_the_lowest_indices():
    # Three clear winners, then a 5-way tie for the last two slots.
    logits = torch.full((1, 16), -1.0)
    logits[0, 9] = 3.0
    logits[0, 2] = 2.0
    logits[0, 13] = 1.0
    for c in (0, 4, 7, 11, 15):
        logits[0, c] = 0.5
    out = canonical_topk(logits, 5, row_ends=torch.tensor([16], dtype=torch.int32))
    assert out[0].tolist() == [9, 2, 13, 0, 4]


def test_matches_the_stable_sort_reference_with_heavy_ties():
    torch.manual_seed(7)
    rows, cols, k = 6, 512, 64
    # Quantise onto a coarse grid so exact ties are everywhere, the way the
    # fp8-e4m3 indexer K cache produces them.
    logits = (torch.randn(rows, cols) * 4).round() / 4
    ends = torch.tensor([cols, cols, cols, 400, 300, 64], dtype=torch.int32)
    got = canonical_topk(logits, k, row_ends=ends)
    want = reference_canonical_topk(logits, k, ends)
    assert torch.equal(got, want)


def test_result_set_equals_the_reference_set():
    torch.manual_seed(11)
    rows, cols, k = 4, 256, 32
    logits = (torch.randn(rows, cols) * 2).round() / 2
    ends = torch.full((rows,), cols, dtype=torch.int32)
    got = canonical_topk(logits, k, row_ends=ends)
    want = reference_canonical_topk(logits, k, ends)
    for r in range(rows):
        assert set(got[r].tolist()) == set(want[r].tolist())


def test_no_change_when_there_are_no_ties():
    # Distinct scores: the canonical answer is exactly torch.topk's.
    torch.manual_seed(3)
    rows, cols, k = 3, 128, 16
    logits = torch.randperm(rows * cols).float().reshape(rows, cols)
    ends = torch.full((rows,), cols, dtype=torch.int32)
    got = canonical_topk(logits, k, row_ends=ends)
    want = logits.topk(k, dim=-1).indices.to(torch.int32)
    assert torch.equal(got, want)


def test_selected_scores_are_descending():
    torch.manual_seed(5)
    logits = (torch.randn(2, 300) * 3).round() / 3
    ends = torch.tensor([300, 300], dtype=torch.int32)
    out = canonical_topk(logits, 40, row_ends=ends)
    for r in range(2):
        picked = logits[r, out[r].long()]
        assert torch.all(picked[:-1] >= picked[1:])


def test_short_rows_are_minus_one_filled():
    logits = torch.randn(3, 64)
    ends = torch.tensor([10, 0, 64], dtype=torch.int32)
    out = canonical_topk(logits, 16, row_ends=ends)
    assert (out[0, 10:] == -1).all() and (out[0, :10] >= 0).all()
    assert (out[1] == -1).all()
    assert (out[2] >= 0).all()


def test_negative_and_infinite_scores_order_correctly():
    logits = torch.tensor([[-3.0, float("-inf"), -0.5, 2.0, -0.5, 7.0]])
    out = canonical_topk(logits, 6, row_ends=torch.tensor([6], dtype=torch.int32))
    assert out[0].tolist() == [5, 3, 2, 4, 0, 1]


def test_prefill_convention_relative_and_identity_short():
    # Rows 0 and 1 are shorter than k -> identity order then -1 fill, which is
    # what top_k_per_row_prefill's rowLen <= topK shortcut emits.
    logits = torch.randn(3, 64)
    starts = torch.tensor([0, 8, 4], dtype=torch.int32)
    ends = torch.tensor([4, 14, 64], dtype=torch.int32)
    out = canonical_topk(
        logits,
        8,
        row_starts=starts,
        row_ends=ends,
        relative=True,
        identity_when_short=True,
    )
    want = reference_canonical_topk(
        logits, 8, ends, starts, relative=True, identity_when_short=True
    )
    assert torch.equal(out, want)
    assert out[0].tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert out[1].tolist() == [0, 1, 2, 3, 4, 5, -1, -1]
    # Row 2's window is [4, 64): indices are relative and in range.
    assert int(out[2].min()) >= 0 and int(out[2].max()) < 60


def test_relative_window_selects_only_inside_the_window():
    logits = torch.full((1, 32), 0.0)
    logits[0, 0] = 100.0  # outside the window, must never be selected
    logits[0, 31] = 100.0  # outside the window on the right
    starts = torch.tensor([8], dtype=torch.int32)
    ends = torch.tensor([24], dtype=torch.int32)
    out = canonical_topk(logits, 4, row_starts=starts, row_ends=ends, relative=True)
    assert out[0].tolist() == [0, 1, 2, 3]  # relative -> absolute 8, 9, 10, 11


def test_topk_repeated_calls_are_identical():
    torch.manual_seed(2)
    logits = (torch.randn(4, 1024) * 4).round() / 4
    ends = torch.full((4,), 1024, dtype=torch.int32)
    first = canonical_topk(logits, 128, row_ends=ends).clone()
    for _ in range(19):
        assert torch.equal(canonical_topk(logits, 128, row_ends=ends), first)


def test_row_chunking_does_not_change_the_answer(monkeypatch):
    import vllm.model_executor.layers.indexer_topk as mod

    torch.manual_seed(13)
    logits = (torch.randn(37, 600) * 3).round() / 3
    ends = torch.full((37,), 600, dtype=torch.int32)
    whole = canonical_topk(logits, 48, row_ends=ends).clone()
    monkeypatch.setattr(mod, "_CANONICAL_KEY_BUDGET_BYTES", 600 * 8 * 5)
    chunked = canonical_topk(logits, 48, row_ends=ends)
    assert torch.equal(whole, chunked)


def test_writes_into_a_wider_destination_buffer():
    logits = torch.randn(2, 64)
    dst = torch.full((2, 100), -7, dtype=torch.int32)
    ends = torch.tensor([64, 64], dtype=torch.int32)
    canonical_topk(logits, 16, row_ends=ends, out=dst)
    assert (dst[:, 16:] == -7).all()
    assert (dst[:, :16] >= 0).all()


def test_forward_takes_the_canonical_path(monkeypatch):
    """The dispatch is wired, and it is wired ahead of backend resolution.

    There is no CUDA here, so every backend SparseIndexerTopk could resolve to
    would raise. Getting an answer at all proves the canonical branch ran
    first -- which also means the flag-on path needs no top-k workspace.
    """
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

    monkeypatch.setenv("VLLM_GLM5_TOPK_CANONICAL", "1")
    use_canonical_topk.cache_clear()
    try:
        module = SparseIndexerTopk("auto")
        torch.manual_seed(17)
        logits = torch.randn(4, 256)
        seq_lens = torch.full((1, 4), 200, dtype=torch.int32)
        out = torch.empty((4, 64), dtype=torch.int32)
        module(logits, seq_lens, 4, out, 64, 200)
        want = canonical_topk(logits, 64, row_ends=module._row_ends(seq_lens, 4, 4))
        assert torch.equal(out, want)
    finally:
        use_canonical_topk.cache_clear()


def test_no_cuda_was_initialised():
    assert torch.cuda.is_initialized() is False
