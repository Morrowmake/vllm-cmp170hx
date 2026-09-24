# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the split tie fix + sort of the indexer top-k
(VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS).

``topk_tiefix_(..., sort=True, split=True)`` runs two launches over (row,
chunk) and must equal the one-program-per-row kernel with ``sort=True`` (and
so the canonical set in ascending order) byte for byte, for every input the
fast kernels can produce. The fast kernels are modelled by an exact top-k
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
    sort_selected_topk_,
    tiefix_split_rows,
    use_canonical_topk,
    use_sorted_topk,
    use_tiefix_topk,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a GPU",
)


@pytest.fixture(scope="module")
def mod():
    m = importlib.import_module("vllm.model_executor.layers.indexer_topk_tiefix")
    # Re-decorate under the interpreter in case triton was imported first.
    return importlib.reload(m)


@pytest.fixture
def small_chunks(mod, monkeypatch):
    """Shrink the split geometry so CPU-sized rows span many chunks."""
    monkeypatch.setattr(mod, "SPLIT_CHUNK_MIN", 64)
    return mod


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
            if seed % 2:
                picked = picked[torch.randperm(n, generator=g)]
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


def tied_logits(rows, cols, levels, seed):
    """Scores on a coarse grid (like fp8-quantised keys): heavy exact ties."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, levels, (rows, cols), generator=g).float() - 3) / 4


def check(mod, logits, k, ends, starts=None, relative=False, seeds=3, **kw):
    canon = sort_selected_topk_(
        canonical_topk(
            logits,
            k,
            row_ends=ends,
            row_starts=starts,
            relative=relative,
            identity_when_short=relative,
        )
    )
    for seed in range(seeds):
        fast = fast_topk_model(logits, k, ends, starts, relative, seed=seed)
        common = dict(row_ends=ends, row_starts=starts, relative=relative, sort=True)
        one = mod.topk_tiefix_(logits, fast.clone(), **common)
        # The one-program kernel's bytes = tie fix, then the separate sort.
        sep = sort_selected_topk_(
            mod.topk_tiefix_(
                logits, fast.clone(), row_ends=ends, row_starts=starts,
                relative=relative,
            )
        )
        assert torch.equal(one, sep)
        got = mod.topk_tiefix_(logits, fast.clone(), split=True, **common, **kw)
        assert torch.equal(got, one), f"seed={seed}"
        assert torch.equal(got, canon), f"seed={seed}"
        again = mod.topk_tiefix_(logits, got.clone(), split=True, **common, **kw)
        assert torch.equal(again, got)


# --- flag and dispatch ----------------------------------------------------------


def _clear():
    for f in (tiefix_split_rows, use_tiefix_topk, use_sorted_topk, use_canonical_topk):
        f.cache_clear()


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS == 0
    _clear()
    try:
        assert tiefix_split_rows() == 0
    finally:
        _clear()


def test_flag_is_part_of_the_compile_key():
    assert "VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS" in envs.compile_factors()


def test_gate_needs_tiefix_and_sorted(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS", "4")
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX", "1")
    _clear()
    try:
        assert tiefix_split_rows() == 0  # sort off
        monkeypatch.setenv("VLLM_GLM5_TOPK_SORTED", "1")
        _clear()
        assert tiefix_split_rows() == 4
    finally:
        _clear()


def test_use_split_bounds(mod):
    wide = mod.SPLIT_MIN_COLS
    assert not mod.use_split(4, wide, 0)
    assert mod.use_split(4, wide, 4)
    assert not mod.use_split(5, wide, 4)
    assert not mod.use_split(4, wide - 1, 4)
    assert not mod.use_split(0, wide, 4)


def test_dispatch_passes_split(monkeypatch, mod):
    seen = []

    def spy(*a, **k):
        seen.append((k["sort"], k["split"]))

    monkeypatch.setattr(mod, "topk_tiefix_", spy)
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS", "4")
    monkeypatch.setenv("VLLM_GLM5_TOPK_TIEFIX", "1")
    monkeypatch.setenv("VLLM_GLM5_TOPK_SORTED", "1")
    _clear()
    try:
        wide = torch.zeros(8, mod.SPLIT_MIN_COLS)
        ends = torch.full((8,), 10, dtype=torch.int32)
        idx = torch.zeros(8, 16, dtype=torch.int32)
        indexer_topk.tiefix_topk_(wide, idx[:4], row_ends=ends, sort=True)
        indexer_topk.tiefix_topk_(wide, idx, row_ends=ends, sort=True)
        indexer_topk.tiefix_topk_(wide[:, :100], idx[:4], row_ends=ends, sort=True)
        indexer_topk.tiefix_topk_(wide, idx[:4], row_ends=ends, sort=False)
        assert seen == [(True, True), (True, False), (True, False), (False, False)]
    finally:
        _clear()


def test_flag_off_dispatch_is_unchanged(monkeypatch, mod):
    seen = []
    monkeypatch.setattr(mod, "topk_tiefix_", lambda *a, **k: seen.append(k["split"]))
    _clear()
    try:
        wide = torch.zeros(4, mod.SPLIT_MIN_COLS)
        ends = torch.full((4,), 10, dtype=torch.int32)
        idx = torch.zeros(4, 16, dtype=torch.int32)
        indexer_topk.tiefix_topk_(wide, idx, row_ends=ends, sort=True)
        assert seen == [False]
    finally:
        _clear()


# --- split == one program ---------------------------------------------------------


@pytest.mark.parametrize("levels", [2, 3, 8, 64])
@pytest.mark.parametrize("rows,cols,k", [(1, 64, 8), (5, 300, 32), (4, 4096, 512)])
def test_decode_split_equals_one_program(small_chunks, levels, rows, cols, k):
    logits = tied_logits(rows, cols, levels, seed=levels * 7 + rows)
    g = torch.Generator().manual_seed(rows)
    ends = torch.randint(0, cols + 1, (rows,), generator=g, dtype=torch.int32)
    ends[0] = cols
    check(small_chunks, logits, k, ends)


@pytest.mark.parametrize("rows,cols,k", [(6, 256, 16), (4, 3000, 32)])
def test_prefill_relative_windows(small_chunks, rows, cols, k):
    logits = tied_logits(rows, cols, 5, seed=cols)
    g = torch.Generator().manual_seed(cols)
    starts = torch.randint(0, cols // 2, (rows,), generator=g, dtype=torch.int32)
    lens = torch.randint(0, cols // 2, (rows,), generator=g, dtype=torch.int32)
    lens[0] = k  # exactly k: shortcut row
    lens[1] = k + 1  # one over: always contested on a constant row
    logits[1] = 0.25
    ends = starts + lens
    check(small_chunks, logits, k, ends, starts, relative=True)


@pytest.mark.parametrize("block", [16, 64])
def test_ties_across_many_chunks_and_tiles(small_chunks, block):
    rows, cols, k = 5, 3000, 32
    logits = tied_logits(rows, cols, 2, seed=block)
    ends = torch.tensor([3000, 2999, 64, 65, 1500], dtype=torch.int32)
    check(small_chunks, logits, k, ends, block=block)
    starts = torch.tensor([0, 7, 100, 2900, 5], dtype=torch.int32)
    ends = torch.tensor([3000, 2000, 164, 3000, 2995], dtype=torch.int32)
    check(small_chunks, logits, k, ends, starts, relative=True, block=block)


def test_production_geometry_distinct_and_tied(mod):
    """Default geometry: k 512, 1024-column chunks, a row spanning 16."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 16384, generator=g)
    logits[2:] = torch.round(logits[2:] * 8) / 8
    ends = torch.tensor([16384, 9000, 16380, 700], dtype=torch.int32)
    check(mod, logits, 512, ends, seeds=2)


def test_signed_zero_and_inf_ties(small_chunks):
    logits = torch.zeros(4, 200)
    logits[0, ::2] = -0.0
    logits[1, :150] = float("-inf")
    logits[2, 50:] = float("inf")
    logits[3] = -0.0
    logits[3, 100:] = 0.0
    ends = torch.full((4,), 200, dtype=torch.int32)
    check(small_chunks, logits, 32, ends)


def test_empty_and_short_rows(small_chunks):
    logits = tied_logits(5, 128, 2, seed=1)
    ends = torch.tensor([0, 1, 5, 16, 17], dtype=torch.int32)
    check(small_chunks, logits, 16, ends, seeds=4)


def test_chunk_narrower_than_k(small_chunks):
    """Prefill chunk narrower than k (short requests beside a long one)."""
    rows, cols, k = 7, 250, 512
    logits = tied_logits(rows, cols, 3, seed=4)
    starts = torch.tensor([0, 0, 0, 100, 100, 100, 249], dtype=torch.int32)
    ends = torch.tensor([1, 60, 100, 101, 180, 250, 250], dtype=torch.int32)
    check(small_chunks, logits, k, ends, starts, relative=True)


def test_strided_output_slice(small_chunks):
    logits = tied_logits(6, 700, 3, seed=2)
    ends = torch.tensor([700, 600, 10, 0, 699, 64], dtype=torch.int32)
    buf = torch.full((6, 96), 12345, dtype=torch.int32)
    view = buf[:, :64]
    fast = fast_topk_model(logits, 64, ends, seed=1)
    view.copy_(fast)
    ref = small_chunks.topk_tiefix_(logits, fast.clone(), row_ends=ends, sort=True)
    small_chunks.topk_tiefix_(logits, view, row_ends=ends, sort=True, split=True)
    assert torch.equal(view, ref)
    assert bool((buf[:, 64:] == 12345).all())


def test_same_rows_in_any_batch(small_chunks):
    k, cols = 64, 1500
    base = tied_logits(4, cols, 3, seed=9)
    base_ends = torch.tensor([1500, 1000, 64, 900], dtype=torch.int32)
    outs = []
    for m in (4, 8, 16):
        extra = tied_logits(m - 4, cols, 3, seed=m) if m > 4 else base[:0]
        logits = torch.cat([base, extra])
        g = torch.Generator().manual_seed(m)
        ends = torch.cat(
            [base_ends, torch.randint(0, cols + 1, (m - 4,), generator=g).int()]
        )
        fast = fast_topk_model(logits, k, ends, seed=m)
        got = small_chunks.topk_tiefix_(
            logits, fast, row_ends=ends, sort=True, split=True
        )
        outs.append(got[:4].clone())
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


def test_chunk_is_never_narrower_than_k(mod):
    for k in (8, 512, 2048):
        for n_cols in (1, 100, 16384, 65536, 1 << 20):
            c = mod._split_chunk(n_cols, k)
            assert c >= k and c >= mod.SPLIT_CHUNK_MIN
            assert c & (c - 1) == 0
            assert -(-n_cols // c) <= mod.SPLIT_MAX_CHUNKS


def test_k_2048_with_default_geometry(mod):
    """kpool 1: k = 2048 is wider than the default 1024-column chunk."""
    logits = tied_logits(2, 5000, 3, seed=3)
    ends = torch.tensor([5000, 1500], dtype=torch.int32)
    check(mod, logits, 2048, ends, seeds=1)
