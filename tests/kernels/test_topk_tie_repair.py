# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sparse-indexer top-k tie repair (VLLM_GLM5_TOPK_TIE_REPAIR).

The stock top-k kernels select every entry above the k-th score correctly but
place entries equal to it by atomic arrival. The repair re-picks only that
tied part, lowest column first, and sorts each row, which must give exactly
the canonical (score desc, index asc) set. The Triton kernel runs under the
CPU interpreter in a subprocess (TRITON_INTERPRET must be set before Triton is
imported); no CUDA.
"""

import os
import subprocess
import sys

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers import indexer_topk as it


def _stock_like(logits, k, starts, ends, seed):
    """A correct top-k that breaks exact ties at random, in random slot order,
    with relative indices and the identity shortcut for short windows."""
    g = torch.Generator().manual_seed(seed)
    rows = logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        s, e = int(starts[r]), int(ends[r])
        n = max(0, e - s)
        if n <= k:
            out[r, :n] = torch.arange(n, dtype=torch.int32)
            continue
        w = logits[r, s:e]
        noise = torch.rand(n, generator=g)
        pick = sorted(range(n), key=lambda i: (-float(w[i]), float(noise[i])))[:k]
        out[r] = torch.tensor(pick, dtype=torch.int32)[torch.randperm(k, generator=g)]
    return out


def _sets(x):
    return [set(v for v in row.tolist() if v >= 0) for row in x]


def _grid_logits(rows, cols, levels, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, levels, (rows, cols), generator=g).float() / 4.0


def test_flag_defaults_off_and_is_in_the_compile_key():
    assert envs.VLLM_GLM5_TOPK_TIE_REPAIR is False
    it.use_topk_tie_repair.cache_clear()
    try:
        assert it.use_topk_tie_repair() is False
    finally:
        it.use_topk_tie_repair.cache_clear()
    assert "VLLM_GLM5_TOPK_TIE_REPAIR" in envs.compile_factors()


def _check_prefill_convention_matches_the_canonical_set(seed):
    g = torch.Generator().manual_seed(seed)
    rows, cols, k = 6, 260, 24
    logits = _grid_logits(rows, cols, 10, seed)
    starts = torch.randint(0, 40, (rows,), generator=g, dtype=torch.int32)
    ends = (starts + torch.randint(8, 220, (rows,), generator=g, dtype=torch.int32)).clamp(max=cols)
    ref = it.canonical_topk(logits, k, row_starts=starts, row_ends=ends, relative=True,
                            identity_when_short=True)
    for s2 in range(2):
        buf = torch.full((rows, k + 3), 77, dtype=torch.int32)
        buf[:, :k] = _stock_like(logits, k, starts, ends, 10 * seed + s2)
        it.repair_topk_ties_(buf, logits, k, ends, starts, relative=True)
        assert _sets(buf[:, :k]) == _sets(ref)
        for row in buf[:, :k].tolist():
            n = sum(v >= 0 for v in row)
            assert row[:n] == sorted(row[:n]) and all(v == -1 for v in row[n:])
        assert (buf[:, k:] == 77).all()


def _check_different_tie_lotteries_give_identical_bytes():
    rows, cols, k = 4, 300, 32
    logits = _grid_logits(rows, cols, 6, 7)
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full((rows,), cols, dtype=torch.int32)
    outs = []
    for s in range(4):
        ids = _stock_like(logits, k, starts, ends, s)
        it.repair_topk_ties_(ids, logits, k, ends, starts, relative=True)
        outs.append(ids)
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


def _check_decode_convention_absolute_and_short_rows():
    rows, cols, k = 5, 200, 16
    logits = _grid_logits(rows, cols, 5, 3)
    ends = torch.tensor([200, 150, 17, 16, 0], dtype=torch.int32)
    ref = it.canonical_topk(logits, k, row_ends=ends)
    ids = _stock_like(logits, k, torch.zeros(rows, dtype=torch.int32), ends, 5)
    it.repair_topk_ties_(ids, logits, k, ends, None, relative=False)
    assert _sets(ids) == _sets(ref)
    assert (ids[4] == -1).all()


def _check_no_ties_is_only_a_sort():
    rows, cols, k = 3, 128, 16
    logits = torch.randn(rows, cols, generator=torch.Generator().manual_seed(1))
    ends = torch.full((rows,), cols, dtype=torch.int32)
    ids = _stock_like(logits, k, torch.zeros(rows, dtype=torch.int32), ends, 2)
    before = _sets(ids)
    it.repair_topk_ties_(ids, logits, k, ends, None, relative=False)
    assert _sets(ids) == before


def test_warm_up_is_a_no_op_without_cuda():
    it._warm_tie_repair()   # must not raise or touch CUDA
    assert not torch.cuda.is_initialized()


def _run_all_checks():
    for seed in range(4):
        _check_prefill_convention_matches_the_canonical_set(seed)
    _check_different_tie_lotteries_give_identical_bytes()
    _check_decode_convention_absolute_and_short_rows()
    _check_no_ties_is_only_a_sort()
    print("TIE_REPAIR_CHECKS_OK")


def test_kernel_matches_canonical_under_the_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, __file__], env=env, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert "TIE_REPAIR_CHECKS_OK" in proc.stdout


if __name__ == "__main__":
    _run_all_checks()
