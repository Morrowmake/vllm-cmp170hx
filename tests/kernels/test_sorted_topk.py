# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sorted sparse-indexer top-k (VLLM_SM80_TOPK_SORTED).

The stock prefill and decode top-k kernels return the correct index set in an
arrival-dependent order, and sparse attention accumulates in index order.
With VLLM_SM80_TOPK_CANONICAL_TIES the tie-fix launch also sorts each row
(ascending, -1 fill last; the bytes are checked in test_topk_tiefix.py);
without it ``sort_topk_rows_`` sorts each row and keeps the set. Both the GLM
kpool and the Qwen QSA indexer reach the switches through ``canonical_topk_``.

Runs on CPU through the Triton interpreter (in a child process when Triton was
imported without it); no CUDA, no engine.
"""

import ast
import importlib
import pathlib

import pytest
import torch

from tests.kernels.interpreter_child import run_in_interpreter_child
from tests.kernels.topk_reference import canonical_topk, sorted_rows
import vllm.envs as envs
from vllm_sm80.kernels.indexer import topk_tiefix_gate as gate
from vllm_sm80.kernels.indexer.topk_tiefix_gate import (
    use_sorted_topk,
    use_tiefix_topk,
)


def _clear():
    for f in (use_sorted_topk, use_tiefix_topk):
        f.cache_clear()


@pytest.fixture
def switches(monkeypatch):
    """Set both switches (None = unset) and re-read them."""

    def set_switches(*, canonical, sort):
        for name, value in (
            ("VLLM_SM80_TOPK_CANONICAL_TIES", canonical),
            ("VLLM_SM80_TOPK_SORTED", sort),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, "1" if value else "0")
        _clear()

    yield set_switches
    _clear()


@pytest.fixture(scope="module")
def tiefix_mod():
    return importlib.import_module("vllm_sm80.kernels.indexer.topk_tiefix")


# --- switch ----------------------------------------------------------------


def test_flag_defaults_off(switches):
    switches(canonical=None, sort=None)
    assert envs.VLLM_SM80_TOPK_SORTED is False
    assert use_sorted_topk() is False


def test_flag_is_read_when_set(switches):
    switches(canonical=None, sort=True)
    assert use_sorted_topk() is True


def test_flag_is_part_of_the_compile_key():
    # It changes what the indexers compute, so it must not be ignored.
    assert "VLLM_SM80_TOPK_SORTED" in envs.compile_factors()


def test_banner_names_the_switch(switches, monkeypatch, tiefix_mod):
    lines = []
    monkeypatch.setattr(gate.logger, "info_once", lambda msg, *a: lines.append(msg))
    monkeypatch.setattr(tiefix_mod, "warm_tiefix", lambda: None)
    gate._warm.cache_clear()
    try:
        switches(canonical=None, sort=True)
        assert use_sorted_topk() is True
    finally:
        gate._warm.cache_clear()
    assert len(lines) == 1 and "VLLM_SM80_TOPK_SORTED=1" in lines[0]


def test_sort_alone_is_accepted(switches):
    switches(canonical=None, sort=True)
    assert use_tiefix_topk() is False
    assert use_sorted_topk() is True


def test_either_switch_warms_the_kernels_once(switches, monkeypatch, tiefix_mod):
    calls = []
    monkeypatch.setattr(tiefix_mod, "warm_tiefix", lambda: calls.append(1))
    gate._warm.cache_clear()
    try:
        switches(canonical=True, sort=True)
        assert use_sorted_topk() and use_tiefix_topk()
        _clear()
        assert use_sorted_topk() and use_tiefix_topk()
    finally:
        gate._warm.cache_clear()
    assert calls == [1]


# --- dispatch --------------------------------------------------------------


@pytest.mark.parametrize(
    "canonical,sort,want",
    [
        (False, False, []),
        (False, True, ["sort"]),
        (True, False, [("tiefix", False)]),
        (True, True, [("tiefix", True)]),
    ],
)
def test_canonical_topk_dispatch(
    switches, monkeypatch, tiefix_mod, canonical, sort, want
):
    """Off: nothing runs (the top-k kernel's bytes). Sort alone: the sort
    launch. Ties: one tie-fix launch carrying the sort switch."""
    seen = []
    monkeypatch.setattr(
        gate, "tiefix_topk_", lambda *a, **k: seen.append(("tiefix", k["sort"]))
    )
    monkeypatch.setattr(tiefix_mod, "sort_topk_rows_", lambda ids: seen.append("sort"))
    monkeypatch.setattr(tiefix_mod, "warm_tiefix", lambda: None)
    switches(canonical=canonical, sort=sort)
    logits = torch.zeros(2, 16)
    ends = torch.full((2,), 16, dtype=torch.int32)
    gate.canonical_topk_(logits, torch.zeros(2, 4, dtype=torch.int32), indexer="kpool",
                         row_ends=ends)
    seq = torch.tensor([[16], [9]], dtype=torch.int32)
    gate.canonical_decode_topk_(logits, torch.zeros(2, 4, dtype=torch.int32), seq, 1,
                                indexer="kpool")
    assert seen == want * 2


def test_decode_off_computes_no_row_ends(switches, monkeypatch):
    """Both off: the decode hook does not even build the per-row ends."""
    monkeypatch.setattr(
        gate.SparseIndexerTopk,
        "_row_ends",
        staticmethod(lambda *a: pytest.fail("row ends built with both off")),
    )
    switches(canonical=False, sort=False)
    seq = torch.tensor([[16], [9]], dtype=torch.int32)
    gate.canonical_decode_topk_(
        torch.zeros(2, 16), torch.zeros(2, 4, dtype=torch.int32), seq, 1,
        indexer="kpool",
    )


# --- the sort launch --------------------------------------------------------


def _random_selection(rows, k, width, seed):
    """Distinct ids in random order, a random count per row, -1 after."""
    g = torch.Generator().manual_seed(seed)
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(torch.randint(0, k + 1, (1,), generator=g))
        out[r, :n] = torch.randperm(width, generator=g)[:n].to(torch.int32)
    return out


@pytest.mark.parametrize("rows,k,width", [(1, 1, 4), (5, 8, 40), (7, 512, 5000)])
def test_sort_rows_ascending_minus_one_last(tiefix_mod, rows, k, width):
    ids = _random_selection(rows, k, width, seed=rows * k)
    want = sorted_rows(ids)
    assert torch.equal(tiefix_mod.sort_topk_rows_(ids.clone()), want)


def test_sort_rows_on_a_strided_slice(tiefix_mod):
    # Column slice of a wider buffer and a column stride of 2: only the
    # selection's own slots change.
    storage = torch.full((5, 2 * 16 + 3), -99, dtype=torch.int32)
    view = storage[1:4, 1 : 2 * 16 + 1 : 2]
    view.copy_(_random_selection(3, 16, 100, seed=3))
    want = storage.clone()
    want[1:4, 1 : 2 * 16 + 1 : 2] = sorted_rows(view)
    tiefix_mod.sort_topk_rows_(view)
    assert torch.equal(storage, want)


def test_sort_rows_empty(tiefix_mod):
    for shape in ((0, 8), (3, 0)):
        ids = torch.empty(shape, dtype=torch.int32)
        assert tiefix_mod.sort_topk_rows_(ids) is ids


def test_sort_alone_equals_tiefix_sort_without_ties(tiefix_mod):
    """Distinct scores: the tie fix changes no set, so the sort alone writes
    the same bytes as the tie fix with the sort. Every row is wider than k or
    empty (shorter rows are the kernels' identity shortcut, already sorted)."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 900, generator=g)
    ends = torch.tensor([900, 800, 600, 65, 100, 0], dtype=torch.int32)
    k = 64
    fast = canonical_topk(logits, k, row_ends=ends)
    shuffled = fast.clone()
    for r in range(shuffled.shape[0]):
        n = int((shuffled[r] >= 0).sum())
        shuffled[r, :n] = shuffled[r, torch.randperm(n, generator=g)]
    alone = tiefix_mod.sort_topk_rows_(shuffled.clone())
    fixed = tiefix_mod.topk_tiefix_(logits, shuffled.clone(), row_ends=ends, sort=True)
    assert torch.equal(alone, fixed)
    assert torch.equal(alone, sorted_rows(fast))


# --- wiring ------------------------------------------------------------------


def _function_source(module, name):
    """Source of a top-level function or class, read from its file: importing
    the GLM-5 model module pulls in the vendored FLA ops, which query the
    device at import (and would initialise CUDA on a machine with a GPU)."""
    import importlib.util

    top, *parts = module.split(".")
    root = pathlib.Path(importlib.util.find_spec(top).origin).parent
    path = root.joinpath(*parts).with_suffix(".py")
    text = path.read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name == name)
    return ast.get_source_segment(text, node)


def test_glm_indexer_calls_the_gate_after_each_topk():
    # Prefill: right after the top-k selection and before pool expansion.
    # Decode: right after the shared decode top-k and before the DCP merge.
    src = _function_source("vllm.models.glm5next.nvidia.sparse_indexer",
                           "sparse_attn_indexer_kpool")
    assert src.count("sm80_indexer.prefill_topk_tiefix(") == 1
    pre = src.index("torch.ops._C.top_k_per_row_prefill")
    fix = src.index("sm80_indexer.prefill_topk_tiefix(")
    assert pre < fix < src.index("expand_pools_and_append_tail")
    assert src.count("sm80_indexer.decode_topk_tiefix(") == 1
    dec_topk = src.index("get_indexer_topk(topk_backend)(")
    dec_fix = src.index("sm80_indexer.decode_topk_tiefix(")
    assert dec_topk < dec_fix < src.index("_merge_dcp_topk_global(", dec_topk)
    # Each fork step runs its gate, with the kpool row of the tuning table as
    # its split row bound.
    for step, gate in (("prefill_topk_tiefix", "canonical_topk_("),
                       ("decode_topk_tiefix", "canonical_decode_topk_(")):
        step_src = _function_source("vllm_sm80.models.glm5next.sm80_indexer", step)
        assert step_src.count(gate) == 1 and 'indexer="kpool"' in step_src, step


def test_qsa_indexer_calls_the_gate_after_its_topk():
    src = _function_source("vllm.models.qwen4_exp.nvidia.ops.qsa_indexer", "_topk")
    assert src.count("canonical_topk_(") == 1
    assert src.index("topk_op(") < src.index("canonical_topk_(")
    assert 'indexer="qsa"' in src


run_in_interpreter_child(globals())
