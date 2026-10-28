# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the tie-consistent indexer top-k repair
(VLLM_SM80_TOPK_CANONICAL_TIES).

The fast top-k kernels return an exact top-k by score but pick among the
entries tied with the k-th score by atomic arrival order, and the kernel that
runs depends on the row count and length. ``topk_tiefix_`` rewrites only the
rows whose tied group straddles the boundary so that the selected SET equals
``canonical_topk``'s. The fast kernels are modelled here by an exact top-k
that breaks ties at random and returns the set in a random order.

Runs on CPU through the Triton interpreter (in a child process when Triton was
imported without it); no CUDA, no engine.
"""

import importlib

import pytest
import torch

from tests.kernels.interpreter_child import run_in_interpreter_child
from tests.kernels.topk_reference import canonical_topk, sorted_rows
import vllm.envs as envs
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm_sm80.kernels.indexer import topk_tiefix_gate as gate
from vllm_sm80.kernels.indexer.topk_tiefix_gate import use_tiefix_topk


@pytest.fixture(scope="module")
def tiefix():
    mod = importlib.import_module("vllm_sm80.kernels.indexer.topk_tiefix")
    return mod.topk_tiefix_


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


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM80_TOPK_CANONICAL_TIES", raising=False)
    assert envs.VLLM_SM80_TOPK_CANONICAL_TIES is False
    use_tiefix_topk.cache_clear()
    try:
        assert use_tiefix_topk() is False
    finally:
        use_tiefix_topk.cache_clear()


def test_flag_is_read_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", "1")
    use_tiefix_topk.cache_clear()
    try:
        assert use_tiefix_topk() is True
    finally:
        use_tiefix_topk.cache_clear()


def test_flag_is_part_of_the_compile_key():
    assert "VLLM_SM80_TOPK_CANONICAL_TIES" in envs.compile_factors()


def test_flag_off_does_not_touch_the_output(monkeypatch):
    """Both switches unset, the decode hook never calls the repair or the
    sort (byte-identical path)."""
    calls = []
    monkeypatch.setattr(gate, "tiefix_topk_", lambda *a, **k: calls.append(1))
    monkeypatch.delenv("VLLM_SM80_TOPK_CANONICAL_TIES", raising=False)
    monkeypatch.delenv("VLLM_SM80_TOPK_SORTED", raising=False)
    for f in (use_tiefix_topk, gate.use_sorted_topk):
        f.cache_clear()
    try:
        topk = SparseIndexerTopk("torch")
        logits = tied_logits(2, 64, 4, seed=0)
        out = torch.empty(2, 8, dtype=torch.int32)
        seq = torch.tensor([[64], [40]], dtype=torch.int32)
        topk(logits, seq, 1, out, 8, 64)
        before = out.clone()
        gate.canonical_decode_topk_(logits, out, seq, 1, indexer="kpool")
        assert calls == []
        assert torch.equal(out, before)
    finally:
        for f in (use_tiefix_topk, gate.use_sorted_topk):
            f.cache_clear()


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


def test_2d_seq_lens_through_the_decode_tie_fix(monkeypatch, tiefix):
    """The decode tie fix with the flag on repairs the torch backend's own
    choice."""
    monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", "1")
    use_tiefix_topk.cache_clear()
    try:
        topk = SparseIndexerTopk("torch")
        logits = tied_logits(4, 900, 2, seed=2)
        seq = torch.tensor([[900], [850], [600], [3]], dtype=torch.int32)
        out = torch.empty(4, 64, dtype=torch.int32)
        topk(logits, seq, 1, out, 64, 900)
        gate.canonical_decode_topk_(logits, out, seq, 1, indexer="kpool")
        ref = canonical_topk(logits, 64, row_ends=seq.reshape(-1))
        assert row_sets(out) == row_sets(ref)
    finally:
        use_tiefix_topk.cache_clear()


@pytest.mark.parametrize("block", [16, 64])
def test_tie_group_spanning_many_tiles(tiefix, block, monkeypatch):
    """Passes B and C carry their counts across scan tiles."""
    mod = importlib.import_module("vllm_sm80.kernels.indexer.topk_tiefix")
    monkeypatch.setattr(mod, "TIEFIX_BLOCK", block)
    monkeypatch.setattr(mod, "TIEFIX_NUM_WARPS", 1)
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
        )
        assert row_sets(fixed) == row_sets(ref)


# --- pre-capture compile ----------------------------------------------------


@pytest.fixture
def tiefix_mod():
    return importlib.import_module("vllm_sm80.kernels.indexer.topk_tiefix")


def test_flag_on_warms_the_kernel_once(monkeypatch, tiefix_mod):
    calls = []
    monkeypatch.setattr(tiefix_mod, "warm_tiefix", lambda: calls.append(1))
    monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", "1")
    use_tiefix_topk.cache_clear()
    gate._warm.cache_clear()
    try:
        assert use_tiefix_topk() is True
        assert use_tiefix_topk() is True
        assert calls == [1]
    finally:
        use_tiefix_topk.cache_clear()
        gate._warm.cache_clear()


def test_flag_off_does_not_warm(monkeypatch, tiefix_mod):
    calls = []
    monkeypatch.setattr(tiefix_mod, "warm_tiefix", lambda: calls.append(1))
    monkeypatch.delenv("VLLM_SM80_TOPK_CANONICAL_TIES", raising=False)
    use_tiefix_topk.cache_clear()
    gate._warm.cache_clear()
    try:
        assert use_tiefix_topk() is False
        assert calls == []
    finally:
        use_tiefix_topk.cache_clear()


def test_warm_is_a_noop_without_cuda(monkeypatch, tiefix_mod):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tiefix_mod.warm_tiefix()  # must not launch or raise


def test_row_geometry_is_not_specialised(tiefix_mod):
    k = tiefix_mod._topk_tiefix_kernel
    if hasattr(k, "params"):  # compiled
        names = [p.name for p in k.params]
        dns = k.do_not_specialize
    else:  # interpreted: the decorator's own arguments
        names = k.arg_names
        dns = k.kwargs.get("do_not_specialize") or ()
    dns = {names[i] if isinstance(i, int) else i for i in dns}
    assert {"stride_l", "stride_i", "n_cols"} <= dns


# --- in-launch sort (TIEFIX + SORTED) -------------------------------------


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("levels", [2, 8, 1000])
@pytest.mark.parametrize("rows,cols,k", [(5, 300, 32), (8, 2048, 512)])
def test_sort_in_launch_equals_separate_sort(tiefix, relative, levels, rows, cols, k):
    logits = tied_logits(rows, cols, levels, seed=levels + rows)
    ends = torch.tensor([cols - 3 * i for i in range(rows)], dtype=torch.int32)
    ends[0] = min(k - 5, cols)  # a short row: identity + -1 fill
    starts = (
        torch.tensor([i for i in range(rows)], dtype=torch.int32) if relative else None
    )
    for seed in range(2):
        fast = fast_topk_model(logits, k, ends, starts, relative, seed=seed)
        a = tiefix(
            logits, fast.clone(), row_ends=ends, row_starts=starts, relative=relative
        )
        a = sorted_rows(a)
        b = tiefix(
            logits,
            fast.clone(),
            row_ends=ends,
            row_starts=starts,
            relative=relative,
            sort=True,
        )
        assert torch.equal(a, b)


def test_decode_tie_fix_passes_sort_when_both_flags(monkeypatch):
    seen = {}

    def fake(logits, ids, **kw):
        seen.update(kw)
        return ids

    monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", "1")
    monkeypatch.setenv("VLLM_SM80_TOPK_SORTED", "1")
    for f in (use_tiefix_topk, gate.use_sorted_topk):
        f.cache_clear()
    monkeypatch.setattr(gate, "tiefix_topk_", fake)
    try:
        logits = torch.zeros(4, 16)
        seq = torch.tensor([[16], [9]], dtype=torch.int32)
        gate.canonical_decode_topk_(
            logits, torch.zeros(4, 8, dtype=torch.int32), seq, 2, indexer="kpool"
        )
        assert seen.get("sort") is True
        # The per-row ends of the decode top-k kernels (next_n rows per request).
        assert torch.equal(seen["row_ends"], SparseIndexerTopk._row_ends(seq, 2, 4))
        assert seen["row_ends"].tolist() == [15, 16, 8, 9]
    finally:
        for f in (use_tiefix_topk, gate.use_sorted_topk):
            f.cache_clear()


# --- prefill chunk narrower than k ------------------------------------------


def _narrow_chunk():
    """A prefill chunk of short requests whose pool-granular logits are
    narrower than k = 512, while another, long request in the same step took
    the long-prefill path (so the short-prefill fill did not run)."""
    rows, cols, k = 7, 250, 512
    logits = tied_logits(rows, cols, 3, seed=4)
    starts = torch.tensor([0, 0, 0, 100, 100, 100, 249], dtype=torch.int32)
    ends = torch.tensor([1, 60, 100, 101, 180, 250, 250], dtype=torch.int32)
    return logits, k, starts, ends


def test_canonical_accepts_a_chunk_narrower_than_k():
    logits, k, starts, ends = _narrow_chunk()
    out = canonical_topk(
        logits,
        k,
        row_ends=ends,
        row_starts=starts,
        relative=True,
        identity_when_short=True,
    )
    assert out.shape == (logits.shape[0], k)
    for r, n in enumerate((ends - starts).tolist()):
        assert out[r, :n].tolist() == list(range(n))
        assert bool((out[r, n:] == -1).all())
    # Absolute ids, score order: the set is still the whole window.
    out = canonical_topk(logits, k, row_ends=ends, row_starts=starts)
    for r, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        assert sorted(v for v in out[r].tolist() if v >= 0) == list(range(s, e))


def test_canonical_zero_width_chunk():
    out = canonical_topk(
        torch.empty(3, 0), 8, row_ends=torch.zeros(3, dtype=torch.int32)
    )
    assert bool((out == -1).all())


@pytest.mark.parametrize("sort", [False, True])
def test_tiefix_on_a_chunk_narrower_than_k(tiefix, sort):
    logits, k, starts, ends = _narrow_chunk()
    ref = canonical_topk(
        logits,
        k,
        row_ends=ends,
        row_starts=starts,
        relative=True,
        identity_when_short=True,
    )
    for seed in range(3):
        fast = fast_topk_model(logits, k, ends, starts, True, seed=seed)
        fixed = tiefix(
            logits,
            fast.clone(),
            row_ends=ends,
            row_starts=starts,
            relative=True,
            sort=sort,
        )
        assert row_sets(fixed) == row_sets(ref)
        if sort:
            assert torch.equal(fixed, ref)
        else:
            assert torch.equal(fixed, fast)  # every row uncontested


run_in_interpreter_child(globals())
