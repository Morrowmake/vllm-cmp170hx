# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen QSA canonical top-k (VLLM_SM80_TOPK_CANONICAL_TIES, with
VLLM_SM80_TOPK_SORTED).

The QSA top-k (``persistent_topk`` on sm_80) returns an exact top-k by score
but places the blocks tied with the k-th score by atomic arrival order, so
identical scores can select a different block set. With the switch on,
``_topk`` passes the selection, through the shared gate
(``vllm_sm80/kernels/indexer/topk_tiefix_gate.py``), to the shared indexer
tie fix (``topk_tiefix.py``), which makes the set (score descending, block
index ascending). The stock kernel is modelled here by an exact top-k that
breaks ties at random and returns the set in a random order: what it
guarantees, no more.

Runs on CPU through the Triton interpreter (in a child process when Triton was
imported without it); no CUDA, no engine. The real kernel under ties is
exercised by test_qsa_canonical_topk_gpu.py.
"""

import importlib
import struct

import pytest
import torch

from tests.kernels.interpreter_child import run_in_interpreter_child
from tests.kernels.topk_reference import sorted_rows
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm_sm80.kernels.indexer import topk_tiefix_gate as gate

COMPRESS_RATIO = 4
BLOCK_TOPK = 512  # the shipped config: indexer_budget 2048 / ratio 4


def ordered_key(value: float) -> int:
    """persistent_topk's order (convert_to_uint32_v2): NaN above +inf, a
    negative-signed NaN below -inf. -0.0 is kept apart from +0.0 here, so the
    tests that use this oracle contain no negative zero."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def reference(logits: torch.Tensor, lengths: torch.Tensor, k: int) -> torch.Tensor:
    """Independent per-row oracle: the first k of the visible blocks ordered
    by (score descending, block index ascending), -1 filled."""
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for r, (row, n) in enumerate(zip(logits.tolist(), lengths.tolist())):
        n = min(n, len(row))
        picked = sorted(range(n), key=lambda c: (-ordered_key(row[c]), c))[:k]
        out[r, : len(picked)] = torch.tensor(picked, dtype=torch.int32)
    return out


def row_sets(x: torch.Tensor) -> list[frozenset[int]]:
    return [frozenset(v for v in r.tolist() if v >= 0) for r in x]


class FakeTopk:
    """Stand-in for torch.ops._C.persistent_topk / cooperative_topk: an exact
    top-k over each row's visible blocks (persistent_topk's key order), ties
    and output order chosen at random, min(visible, k) ids in the leading
    slots and -1 after."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.calls = 0

    def __call__(self, logits, lengths, output, workspace, k, max_len) -> None:
        self.calls += 1
        assert output.shape[1] == k and max_len == logits.shape[1]
        g = self.generator
        output.fill_(-1)
        for r, n in enumerate(lengths.tolist()):
            n = min(n, logits.shape[1])
            perm = torch.randperm(n, generator=g)
            keys = [ordered_key(v) for v in logits[r, perm].tolist()]
            order = sorted(range(n), key=lambda i: -keys[i])[:k]  # stable
            picked = perm[torch.tensor(order, dtype=torch.long)]
            picked = picked[torch.randperm(picked.numel(), generator=g)]
            output[r, : picked.numel()] = picked.to(torch.int32)


@pytest.fixture
def fake_topk(monkeypatch: pytest.MonkeyPatch) -> FakeTopk:
    fake = FakeTopk(seed=0)
    for name in ("persistent_topk", "cooperative_topk"):
        monkeypatch.setattr(torch.ops._C, name, fake, raising=False)
    return fake


def tied_logits(rows: int, cols: int, seed: int) -> torch.Tensor:
    """ReLU-floored scores on a coarse grid: most of each row is an exact 0.0
    tie (the floor) plus a few tied positive levels, so the k-th score is a
    tie that straddles the boundary."""
    g = torch.Generator().manual_seed(seed)
    levels = torch.randint(0, 6, (rows, cols), generator=g).float() - 2
    return levels.clamp_(min=0.0) / 4


def run_topk(logits, lengths, k=BLOCK_TOPK, out=None):
    if out is None:
        out = torch.full((logits.shape[0], k), -7, dtype=torch.int32)
    workspace = torch.empty(qsa_indexer_ops._TOPK_WORKSPACE_BYTES, dtype=torch.uint8)
    qsa_indexer_ops._topk(
        logits, lengths, k * COMPRESS_RATIO, COMPRESS_RATIO, out, workspace
    )
    return out


def _clear_gate() -> None:
    for f in (gate.use_tiefix_topk, gate.use_sorted_topk, gate.tiefix_split_rows):
        f.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_gate():
    """The gate caches its reads; drop them around each test (after the
    environment is restored)."""
    _clear_gate()
    yield
    _clear_gate()


def set_switches(monkeypatch, *, canonical: bool, sort: bool) -> None:
    monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", "1" if canonical else "0")
    monkeypatch.setenv("VLLM_SM80_TOPK_SORTED", "1" if sort else "0")
    _clear_gate()


# --- switch ----------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [(None, False), ("1", True)])
def test_switch_is_read_from_the_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_SM80_TOPK_CANONICAL_TIES", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", value)
    assert gate.use_tiefix_topk() is expected


@pytest.mark.parametrize("sort", [False, True])
def test_switch_off_keeps_todays_path(monkeypatch, fake_topk, sort):
    """Ties off: the native top-k, then the sort launch exactly when the sort
    is on; the tie fix is never called."""
    set_switches(monkeypatch, canonical=False, sort=sort)

    def forbidden(*args, **kwargs):
        raise AssertionError("tie fix requested with the switch off")

    tiefix_mod = importlib.import_module("vllm_sm80.kernels.indexer.topk_tiefix")
    monkeypatch.setattr(gate, "tiefix_topk_", forbidden)
    sorts = []
    monkeypatch.setattr(tiefix_mod, "sort_topk_rows_", lambda b: sorts.append(b))
    monkeypatch.setattr(tiefix_mod, "warm_tiefix", lambda: None)
    logits = tied_logits(4, 700, seed=1)
    lengths = torch.tensor([700, 600, 3, 0], dtype=torch.int32)
    native = torch.full((4, BLOCK_TOPK), -7, dtype=torch.int32)
    FakeTopk(seed=0)(logits, lengths, native, None, BLOCK_TOPK, 700)
    out = run_topk(logits, lengths)
    assert fake_topk.calls == 1
    assert torch.equal(out, native)  # same fake seed: same bytes
    assert len(sorts) == (1 if sort else 0)
    assert all(b is out for b in sorts)


# --- selection -------------------------------------------------------------


@pytest.mark.parametrize("sort", [False, True])
def test_exact_ties_resolve_score_desc_index_asc(monkeypatch, fake_topk, sort):
    set_switches(monkeypatch, canonical=True, sort=sort)
    logits = tied_logits(6, 1500, seed=2)
    lengths = torch.tensor([1500, 1400, 1100, 900, 700, 513], dtype=torch.int32)
    want = reference(logits, lengths, BLOCK_TOPK)
    for seed in range(3):
        fake_topk.generator.manual_seed(seed)
        got = run_topk(logits, lengths)
        assert row_sets(got) == row_sets(want)
        if sort:
            assert torch.equal(got, sorted_rows(want))
        else:
            # Score descending is not promised (the kernel's order is kept for
            # the blocks above the tie), but the set is, with -1 only after it.
            for row in got.tolist():
                n = sum(v >= 0 for v in row)
                assert all(v == -1 for v in row[n:])


def test_boundary_tie_takes_the_lowest_indices(monkeypatch, fake_topk):
    set_switches(monkeypatch, canonical=True, sort=True)
    logits = torch.zeros(1, 16)
    logits[0, 9], logits[0, 2], logits[0, 13] = 3.0, 2.0, 1.0
    for c in (0, 4, 7, 11, 15):
        logits[0, c] = 0.5
    got = run_topk(logits, torch.tensor([16], dtype=torch.int32), k=5)
    assert got.tolist() == [[0, 2, 4, 9, 13]]


def test_infinities_and_nan_rank_as_todays_topk(monkeypatch, fake_topk):
    """-inf ranks last, NaN above +inf, a negative-signed NaN below -inf: the
    persistent_topk key order. Ties among -inf break by lowest index."""
    set_switches(monkeypatch, canonical=True, sort=False)
    k = 8
    nan, neg_nan = float("nan"), -float("nan")
    logits = torch.full((3, 40), float("-inf"))
    logits[0, :12] = 0.25  # 12-way tie for 8 slots
    logits[0, 30] = float("inf")
    logits[1, 5], logits[1, 20], logits[1, 33] = nan, float("inf"), 1.0
    logits[1, 7] = neg_nan
    logits[2, 3] = 2.0  # one finite score, the -inf tie fills the rest
    lengths = torch.full((3,), 40, dtype=torch.int32)
    want = reference(logits, lengths, k)
    assert want[1, :3].tolist() == [5, 20, 33]
    assert 7 not in want[1].tolist()
    for seed in range(3):
        fake_topk.generator.manual_seed(seed)
        got = run_topk(logits, lengths, k=k)
        assert row_sets(got) == row_sets(want)


def test_masked_columns_are_never_selected(monkeypatch, fake_topk):
    """Columns at or past a row's visible count hold whatever the workspace
    held (the score kernels never write them): never read, never chosen."""
    set_switches(monkeypatch, canonical=True, sort=True)
    logits = tied_logits(4, 1024, seed=4)
    lengths = torch.tensor([1024, 700, 600, 0], dtype=torch.int32)
    columns = torch.arange(1024)
    garbage = columns >= lengths[:, None]
    logits[garbage] = 1e30
    logits[2, 650] = float("nan")
    got = run_topk(logits, lengths)
    want = reference(logits, lengths, BLOCK_TOPK)
    assert torch.equal(got, sorted_rows(want))
    for row, n in zip(got.tolist(), lengths.tolist()):
        assert all(v < n for v in row if v >= 0)
    assert (got[3] == -1).all()


@pytest.mark.parametrize("sort", [False, True])
def test_k_larger_than_visible(monkeypatch, fake_topk, sort):
    """min(visible, k) ids in the leading slots, -1 after; every visible block
    is selected, so the set is the whole window."""
    set_switches(monkeypatch, canonical=True, sort=sort)
    logits = tied_logits(5, 600, seed=5)
    lengths = torch.tensor([0, 1, 511, 512, 513], dtype=torch.int32)
    got = run_topk(logits, lengths)
    for row, n in zip(got.tolist(), lengths.tolist()):
        count = min(n, BLOCK_TOPK)
        assert all(v >= 0 for v in row[:count]) and all(v == -1 for v in row[count:])
        if n <= BLOCK_TOPK:
            assert set(row[:count]) == set(range(n))
            if sort:
                assert row[:count] == list(range(n))
    assert row_sets(got)[4] == row_sets(reference(logits, lengths, BLOCK_TOPK))[4]


def test_no_ties_same_set_as_todays_topk(monkeypatch, fake_topk):
    """Distinct scores: the canonical set is the native set, and with the sort
    off every byte of the native output is kept."""
    g = torch.Generator().manual_seed(6)
    logits = torch.randn(6, 3000, generator=g).abs_()
    lengths = torch.tensor([3000, 2900, 2048, 1024, 600, 100], dtype=torch.int32)
    native = torch.full((6, BLOCK_TOPK), -7, dtype=torch.int32)
    FakeTopk(seed=0)(logits, lengths, native, None, BLOCK_TOPK, 3000)
    set_switches(monkeypatch, canonical=True, sort=False)
    got = run_topk(logits, lengths)
    assert torch.equal(got, native)
    for r, n in enumerate(lengths.tolist()):
        want = torch.topk(logits[r, :n], min(n, BLOCK_TOPK)).indices
        assert row_sets(got)[r] == frozenset(want.tolist())


# --- wiring: prefill and decode selection -----------------------------------


def _fake_score_kernel(scores: torch.Tensor, logits_arg: int, prefill: bool):
    """Stand-in for a QSA score kernel launch: copies precomputed scores into
    the logits it is given and fills the columns the real kernel leaves
    unwritten with garbage."""

    class Launcher:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                logits = args[logits_arg]
                offset = args[13] if prefill else 0
                rows, cols = logits.shape
                logits.copy_(scores[offset : offset + rows, :cols])

            return launch

    return Launcher()


def _selection_case(rows: int, cols: int):
    scores = tied_logits(rows, cols, seed=rows)
    g = torch.Generator().manual_seed(cols)
    visible = torch.randint(BLOCK_TOPK - 16, cols + 1, (rows,), generator=g)
    visible = visible.to(torch.int32)
    scores[torch.arange(cols) >= visible[:, None]] = float("nan")
    return scores, visible


def test_decode_selection_takes_the_canonical_path(monkeypatch, fake_topk):
    set_switches(monkeypatch, canonical=True, sort=True)
    page_size, pages, dql = 64, 16, 2
    cols = page_size * pages
    scores, visible = _selection_case(6, cols)
    monkeypatch.setattr(
        qsa_indexer_ops,
        "_qsa_mqa_paged_uniform_kernel",
        _fake_score_kernel(scores, 4, prefill=False),
    )
    q = torch.zeros(6, 4, 32, dtype=torch.bfloat16)
    k_cache = torch.zeros(8, page_size, 1, 32, dtype=torch.bfloat16)
    page_table = torch.zeros(6 // dql, pages, dtype=torch.int32)
    out = torch.full((6, BLOCK_TOPK), -7, dtype=torch.int32)
    qsa_indexer_ops.qsa_select_paged_decode(
        q, k_cache, page_table, visible, BLOCK_TOPK * COMPRESS_RATIO,
        COMPRESS_RATIO, dql, out,
    )
    assert fake_topk.calls == 1
    assert torch.equal(out, sorted_rows(reference(scores, visible, BLOCK_TOPK)))


def test_prefill_selection_takes_the_canonical_path(monkeypatch, fake_topk):
    """Over several logits chunks (VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1)."""
    set_switches(monkeypatch, canonical=True, sort=True)
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    rows, page_size, pages = 150, 64, 64
    cols = page_size * pages  # 4096: 64 rows per 1 MiB chunk -> 3 chunks
    scores, visible = _selection_case(rows, cols)
    monkeypatch.setattr(
        qsa_indexer_ops,
        "_qsa_mqa_paged_prefill_kernel",
        _fake_score_kernel(scores, 5, prefill=True),
    )
    q = torch.zeros(rows, 4, 32, dtype=torch.bfloat16)
    k_cache = torch.zeros(8, page_size, 1, 32, dtype=torch.bfloat16)
    page_table = torch.zeros(1, pages, dtype=torch.int32)
    query_start_loc = torch.tensor([0, rows], dtype=torch.int32)
    out = torch.full((rows, BLOCK_TOPK), -7, dtype=torch.int32)
    qsa_indexer_ops.qsa_select_paged_prefill(
        q, k_cache, page_table, query_start_loc, visible,
        BLOCK_TOPK * COMPRESS_RATIO, COMPRESS_RATIO, rows, out,
        max_seq_len=cols * COMPRESS_RATIO,
    )
    assert fake_topk.calls == 3
    assert torch.equal(out, sorted_rows(reference(scores, visible, BLOCK_TOPK)))


run_in_interpreter_child(globals())
