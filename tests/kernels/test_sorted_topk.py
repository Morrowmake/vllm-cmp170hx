# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sorted sparse-indexer top-k (VLLM_GLM5_TOPK_SORTED).

The stock prefill and decode top-k kernels return the correct index set in an
arrival-dependent order, and the sparse MLA kernel accumulates in index order.
``sort_selected_topk_`` makes the order a function of the set: ascending, with
the -1 fill kept at the end of each row. CPU only; no CUDA, no engine.
"""

import inspect

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.indexer_topk import (
    sort_selected_topk_,
    use_sorted_topk,
)


def _rows(n_rows=6, k=16, universe=4000, seed=0, dtype=torch.int32):
    g = torch.Generator().manual_seed(seed)
    out = torch.full((n_rows, k), -1, dtype=dtype)
    for r in range(n_rows):
        valid = int(torch.randint(0, k + 1, (1,), generator=g))
        ids = torch.randperm(universe, generator=g)[:valid]
        out[r, :valid] = ids.to(dtype)
    return out


def _shuffle_rows(x, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([row[torch.randperm(row.numel(), generator=g)] for row in x])


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_TOPK_SORTED is False
    use_sorted_topk.cache_clear()
    try:
        assert use_sorted_topk() is False
    finally:
        use_sorted_topk.cache_clear()


def test_flag_is_read_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_SORTED", "1")
    use_sorted_topk.cache_clear()
    try:
        assert use_sorted_topk() is True
    finally:
        use_sorted_topk.cache_clear()


def test_flag_is_part_of_the_compile_key():
    # It changes what the indexer computes, so it must not be ignored.
    assert "VLLM_GLM5_TOPK_SORTED" in envs.compile_factors()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ascending_with_fill_last_and_set_unchanged(dtype):
    x = _rows(dtype=dtype)
    before = [set(r.tolist()) for r in x]
    sort_selected_topk_(x)
    for r, s in zip(x, before):
        vals = r.tolist()
        n = sum(v >= 0 for v in vals)
        assert vals[:n] == sorted(v for v in s if v >= 0)
        assert all(v == -1 for v in vals[n:])
        assert set(vals) == s


def test_any_order_of_the_same_set_gives_identical_bytes():
    base = _rows(n_rows=8, k=64, seed=3)
    ref = sort_selected_topk_(base.clone())
    for seed in range(5):
        y = sort_selected_topk_(_shuffle_rows(base, seed))
        assert torch.equal(y, ref)


def test_fill_scattered_inside_a_row_moves_to_the_end():
    x = torch.tensor([[7, -1, 3, -1, 5], [-1, -1, -1, -1, -1], [4, 3, 2, 1, 0]],
                     dtype=torch.int32)
    sort_selected_topk_(x)
    assert x.tolist() == [[3, 5, 7, -1, -1], [-1] * 5, [0, 1, 2, 3, 4]]


def test_strided_view_of_the_persistent_buffer_is_sorted_in_place():
    buf = torch.full((5, 24), 99, dtype=torch.int32)
    rows = _rows(n_rows=5, k=16, seed=7)
    buf[:, :16] = _shuffle_rows(rows, 1)
    view = buf[1:4, :16]
    assert not view.is_contiguous()
    sort_selected_topk_(view)
    assert torch.equal(buf[1:4, :16], sort_selected_topk_(rows[1:4].clone()))
    # Rows and columns outside the view are untouched.
    assert torch.equal(buf[0, :16], _shuffle_rows(rows, 1)[0])
    assert (buf[:, 16:] == 99).all()


def test_empty_and_zero_row_inputs():
    for shape in [(0, 16), (3, 0)]:
        x = torch.empty(shape, dtype=torch.int32)
        assert sort_selected_topk_(x) is x


def test_largest_valid_int32_index_is_not_mistaken_for_fill():
    big = 2**31 - 2
    x = torch.tensor([[big, 0, -1]], dtype=torch.int32)
    sort_selected_topk_(x)
    assert x.tolist() == [[0, big, -1]]


def test_indexer_calls_the_sort_on_both_paths():
    # The two call sites, prefill and decode, sit right after the top-k
    # selection and before pool expansion; guard against a refactor
    # dropping one of them.
    from vllm.models.glm5next.nvidia import sparse_indexer as si

    src = inspect.getsource(si.sparse_attn_indexer_kpool)
    assert src.count("if use_sorted_topk():") == 2
    assert src.count("sort_selected_topk_(topk_dst)") == 2
    pre = src.index("torch.ops._C.top_k_per_row_prefill")
    first = src.index("sort_selected_topk_(topk_dst)")
    expand = src.index("expand_pools_and_append_tail")
    assert pre < first < expand
    dec = src.index("get_indexer_topk(topk_backend)(")
    second = src.index("sort_selected_topk_(topk_dst)", first + 1)
    assert dec < second < src.index("pool_ids = pool_topk.to(torch.int64)", dec)


def test_no_cuda_was_initialised():
    assert not torch.cuda.is_initialized()
