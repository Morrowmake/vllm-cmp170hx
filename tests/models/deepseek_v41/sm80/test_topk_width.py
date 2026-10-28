# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA top-k of the sm_80 DeepSeek-V4.1 indexer (``torch.topk`` plus the
shared tie fix) over every row width: the selection equals the canonical
definition, and the tie-fix kernel is launched with one ``K`` (one compiled
variant) whatever the width.

Runs on CPU through the Triton interpreter (in a child process).
"""

import pytest
import torch

from tests.kernels.interpreter_child import run_in_interpreter_child
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.kernels.indexer import topk_tiefix

K = 24
WIDTHS = list(range(1, 40)) + [63, 64, 100, 257, 600]


@pytest.fixture
def launched_k(monkeypatch):
    seen: set[int] = set()
    kernel = topk_tiefix._topk_tiefix_kernel

    class Recorder:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(*args, **kwargs):
                seen.add(kwargs["K"])
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(topk_tiefix, "_topk_tiefix_kernel", Recorder())
    return seen


def test_every_width_canonical_with_one_variant(launched_k) -> None:
    g = torch.Generator().manual_seed(0)
    for n_cols in WIDTHS:
        rows = 4
        # Few distinct values, so the K-th score is tied across many columns.
        scores = torch.randint(-3, 4, (rows, n_cols), generator=g).float()
        n_vis = torch.tensor([n_cols, max(n_cols - 1, 0), n_cols // 2, 1])
        want = core.topk_canonical(scores, n_vis, K)
        got = core.topk_canonical_cuda(scores, n_vis, K)
        assert torch.equal(got, want), n_cols
    assert launched_k == {K}


run_in_interpreter_child(globals())
