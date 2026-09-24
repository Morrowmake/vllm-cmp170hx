# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the tie-consistent indexer top-k repair (VLLM_GLM5_TOPK_TIEFIX).

The fast top-k kernels return an exact top-k by score but pick among the
entries tied with the k-th score by atomic arrival order, and the kernel that
runs depends on the row count and length. ``topk_tiefix_`` rewrites only the
rows whose tied group straddles the boundary so that the selected SET equals
``canonical_topk``'s. The fast kernels are modelled here by an exact top-k
that breaks ties at random and returns the set in a random order.

Runs on CPU through the Triton interpreter; no CUDA, no engine.
"""

import importlib
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm.envs as envs  # noqa: E402
from vllm.model_executor.layers import indexer_topk  # noqa: E402
from vllm.model_executor.layers.indexer_topk import (  # noqa: E402
    canonical_topk,
    use_canonical_topk,
    use_tiefix_topk,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a GPU",
)


@pytest.fixture(scope="module")
def tiefix():
    mod = importlib.import_module("vllm.model_executor.layers.indexer_topk_tiefix")
    # Re-decorate under the interpreter in case triton was imported first.
    return importlib.reload(mod).topk_tiefix_


def fast_topk_model(logits, k, row_ends, row_starts=None, relative=False, seed=0):
    """An exact top-k with arbitrary tie choice and arbitrary output order,
    shortcut and -1 fill as the stock kernels: what they guarantee, no more."""
    g = torch.Generator().manual_seed(seed)
    rows = logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        s = 0 if row_starts is None else max(0, int(row_starts[r]))
        e = min(max(int(row_ends[r]), s), logits.shape[1])
        n = e - s
        if n <= k:
            picked = torch.arange(s, e)
        else:
            perm = torch.randperm(n, generator=g) + s
            vals = logits[r, perm]
            order = torch.sort(vals, descending=True, stable=True).indices
            picked = perm[order[:k]]
            picked = picked[torch.randperm(k, generator=g)]
        if relative:
            picked = picked - s
        out[r, : picked.numel()] = picked.to(torch.int32)
    return out


def row_sets(x):
    return [frozenset(v for v in r.tolist() if v >= 0) for r in x]


def tied_logits(rows, cols, levels, seed):
    """Scores on a coarse grid (like fp8-quantised keys): heavy exact ties."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, levels, (rows, cols), generator=g).float() - 3) / 4


def check_equals_canonical(tiefix, logits, k, ends, starts=None, relative=False):
    ref = canonical_topk(
        logits,
        k,
        row_ends=ends,
        row_starts=starts,
        relative=relative,
        identity_when_short=relative,
    )
    ref_sets = row_sets(ref)
    for seed in range(3):
        fast = fast_topk_model(logits, k, ends, starts, relative, seed=seed)
        fixed = tiefix(
            logits, fast.clone(), row_ends=ends, row_starts=starts, relative=relative
        )
        assert row_sets(fixed) == ref_sets
        # -1 only after the selected entries, never inside them.
        for r in fixed.tolist():
            n = sum(v >= 0 for v in r)
            assert all(v >= 0 for v in r[:n]) and all(v == -1 for v in r[n:])
        # Idempotent, byte for byte.
        again = tiefix(
            logits, fixed.clone(), row_ends=ends, row_starts=starts, relative=relative
        )
        assert torch.equal(again, fixed)


# --- flag ------------------------------------------------------------------


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_TOPK_TIEFIX is False
    use_tiefix_topk.cache_clear()
    try:
        assert use_tiefix_topk() is False
    finally:
        use_tiefix_topk.cache_clear()


def test_flag_is_read_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX", "1")
    use_tiefix_topk.cache_clear()
    use_canonical_topk.cache_clear()
    try:
        assert use_tiefix_topk() is True
    finally:
        use_tiefix_topk.cache_clear()


def test_canonical_takes_precedence(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX", "1")
    monkeypatch.setenv("VLLM_GLM5_TOPK_CANONICAL", "1")
    use_tiefix_topk.cache_clear()
    use_canonical_topk.cache_clear()
    try:
        assert use_tiefix_topk() is False
    finally:
        use_tiefix_topk.cache_clear()
        use_canonical_topk.cache_clear()


def test_flag_is_part_of_the_compile_key():
    assert "VLLM_GLM5_TOPK_TIEFIX" in envs.compile_factors()


def test_flag_off_does_not_touch_the_output(monkeypatch):
    """Unset, forward never calls the repair (byte-identical path)."""
    calls = []
    monkeypatch.setattr(indexer_topk, "tiefix_topk_", lambda *a, **k: calls.append(1))
    use_tiefix_topk.cache_clear()
    try:
        topk = indexer_topk.SparseIndexerTopk("torch")
        logits = tied_logits(2, 64, 4, seed=0)
        out = torch.empty(2, 8, dtype=torch.int32)
        seq = torch.tensor([[64], [40]], dtype=torch.int32)
        topk(logits, seq, 1, out, 8, 64)
        assert calls == []
    finally:
        use_tiefix_topk.cache_clear()


# --- selection -------------------------------------------------------------


@pytest.mark.parametrize("levels", [2, 3, 8, 64])
@pytest.mark.parametrize("rows,cols,k", [(1, 64, 8), (5, 300, 32), (16, 2048, 512)])
def test_decode_ties_equal_canonical(tiefix, levels, rows, cols, k):
    logits = tied_logits(rows, cols, levels, seed=levels * 7 + rows)
    g = torch.Generator().manual_seed(rows)
    ends = torch.randint(0, cols + 1, (rows,), generator=g, dtype=torch.int32)
    ends[0] = cols
    check_equals_canonical(tiefix, logits, k, ends)


@pytest.mark.parametrize("rows,cols,k", [(6, 256, 16), (12, 3000, 512)])
def test_prefill_relative_windows_equal_canonical(tiefix, rows, cols, k):
    logits = tied_logits(rows, cols, 5, seed=cols)
    g = torch.Generator().manual_seed(cols)
    starts = torch.randint(0, cols // 2, (rows,), generator=g, dtype=torch.int32)
    lens = torch.randint(0, cols // 2, (rows,), generator=g, dtype=torch.int32)
    lens[0] = k  # exactly k: shortcut row
    lens[1] = k + 1  # one over: always contested on a constant row
    logits[1] = 0.25
    ends = starts + lens
    check_equals_canonical(tiefix, logits, k, ends, starts, relative=True)


def test_short_rows_keep_the_kernels_identity_order(tiefix):
    logits = tied_logits(3, 128, 2, seed=1)
    ends = torch.tensor([0, 5, 16], dtype=torch.int32)
    fast = fast_topk_model(logits, 16, ends)
    fixed = tiefix(logits, fast.clone(), row_ends=ends)
    assert torch.equal(fixed, fast)


def test_uncontested_rows_are_untouched(tiefix):
    # Distinct scores: nothing to repair, every byte kept (including order).
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(8, 5000, generator=g)
    ends = torch.full((8,), 5000, dtype=torch.int32)
    fast = fast_topk_model(logits, 512, ends, seed=3)
    fixed = tiefix(logits, fast.clone(), row_ends=ends)
    assert torch.equal(fixed, fast)


def test_boundary_tie_takes_the_lowest_indices(tiefix):
    logits = torch.full((1, 16), -1.0)
    logits[0, 9], logits[0, 2], logits[0, 13] = 3.0, 2.0, 1.0
    for c in (0, 4, 7, 11, 15):
        logits[0, c] = 0.5
    ends = torch.tensor([16], dtype=torch.int32)
    fast = torch.tensor([[13, 15, 9, 11, 2]], dtype=torch.int32)  # a valid top-5
    fixed = tiefix(logits, fast.clone(), row_ends=ends)
    assert fixed.tolist() == [[13, 9, 2, 0, 4]]


def test_signed_zero_and_infinities_tie_like_canonical(tiefix):
    # -0.0 and +0.0 are one tie group; -inf padding inside the window ties.
    logits = torch.full((2, 40), float("-inf"))
    logits[0, :20] = torch.tensor([0.0, -0.0] * 10)
    logits[0, 30] = 1.0
    logits[1, 5] = 2.0  # one finite score, the rest -inf
    ends = torch.tensor([40, 40], dtype=torch.int32)
    check_equals_canonical(tiefix, logits, 8, ends)


def test_same_row_same_set_across_batch_shapes(tiefix):
    """A row's repaired set does not depend on the batch it runs in."""
    base = tied_logits(32, 1500, 4, seed=11)
    ends = torch.full((32,), 1500, dtype=torch.int32)
    k = 128
    ref = row_sets(canonical_topk(base, k, row_ends=ends))
    for m, seed in ((1, 0), (4, 1), (16, 2), (32, 3)):
        sub = base[:m]
        fast = fast_topk_model(sub, k, ends[:m], seed=seed)
        fixed = tiefix(sub, fast, row_ends=ends[:m])
        assert row_sets(fixed) == ref[:m]


def test_strided_output_slice(tiefix):
    # topk_dst is a column slice of the wider persistent buffer.
    logits = tied_logits(4, 700, 3, seed=5)
    ends = torch.full((4,), 700, dtype=torch.int32)
    buf = torch.full((4, 96), 7, dtype=torch.int32)
    view = buf[:, :64]
    view.copy_(fast_topk_model(logits, 64, ends))
    tiefix(logits, view, row_ends=ends)
    assert row_sets(view) == row_sets(canonical_topk(logits, 64, row_ends=ends))
    assert bool((buf[:, 64:] == 7).all())


def test_2d_seq_lens_through_forward(monkeypatch, tiefix):
    """forward() with the flag on repairs the torch backend's own choice."""
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX", "1")
    use_tiefix_topk.cache_clear()
    use_canonical_topk.cache_clear()
    try:
        topk = indexer_topk.SparseIndexerTopk("torch")
        logits = tied_logits(4, 900, 2, seed=2)
        seq = torch.tensor([[900], [850], [600], [3]], dtype=torch.int32)
        out = torch.empty(4, 64, dtype=torch.int32)
        topk(logits, seq, 1, out, 64, 900)
        ref = canonical_topk(logits, 64, row_ends=seq.reshape(-1))
        assert row_sets(out) == row_sets(ref)
    finally:
        use_tiefix_topk.cache_clear()
        use_canonical_topk.cache_clear()


@pytest.mark.parametrize("block", [16, 64])
def test_tie_group_spanning_many_tiles(tiefix, block):
    """Passes B and C carry their counts across scan tiles."""
    logits = tied_logits(3, 1000, 2, seed=9)
    starts = torch.tensor([0, 37, 500], dtype=torch.int32)
    ends = torch.tensor([1000, 999, 1000], dtype=torch.int32)
    for relative, st in ((False, None), (True, starts)):
        ref = canonical_topk(
            logits, 100, row_ends=ends, row_starts=st, relative=relative
        )
        fast = fast_topk_model(logits, 100, ends, st, relative, seed=4)
        fixed = tiefix(
            logits,
            fast,
            row_ends=ends,
            row_starts=st,
            relative=relative,
            block=block,
            num_warps=1,
        )
        assert row_sets(fixed) == row_sets(ref)
