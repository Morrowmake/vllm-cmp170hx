# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent stable-sort oracle for sparse-indexer top-k tests only.

For finite and infinite scores, select by (score descending, column ascending).
This deliberately uses a per-row Python sort, not the serving kernels' integer
keys or threshold scans. Inputs are copied to CPU once; results stay on the
logits' device so GPU metadata tests can use the same oracle.
"""

import torch


def canonical_topk(
    logits: torch.Tensor,
    k: int,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
    identity_when_short: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable per-row top-k with ragged windows and trailing -1 fill.

    ``relative`` emits offsets from each window's start. With
    ``identity_when_short``, windows no longer than k keep column order, as
    the stock prefill kernel's shortcut does. Widths below k, empty windows
    and zero-width chunks are supported. A wider ``out`` keeps its tail.
    """
    scores = logits.cpu().tolist()
    ends = row_ends.reshape(-1).cpu().tolist()
    starts = None if row_starts is None else row_starts.reshape(-1).cpu().tolist()
    rows, cols = logits.shape
    result = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        start = 0 if starts is None else starts[r]
        end = min(cols, max(start, ends[r]))
        window = list(range(start, end))
        if identity_when_short and len(window) <= k:
            picked = window
        else:
            picked = sorted(window, key=lambda c: (-scores[r][c], c))[:k]
        for slot, col in enumerate(picked):
            result[r, slot] = col - start if relative else col
    result = result.to(logits.device)
    if out is None:
        return result
    out[:, :k].copy_(result)
    return out


def sorted_rows(ids: torch.Tensor) -> torch.Tensor:
    """Each row ascending with the -1 fill last: the order the tie fix writes
    with ``sort``. Returns a new tensor on ``ids``' device."""
    rows = []
    for row in ids.cpu().tolist():
        vals = sorted(v for v in row if v >= 0)
        rows.append(vals + [-1] * (len(row) - len(vals)))
    return torch.tensor(rows, dtype=ids.dtype).reshape(ids.shape).to(ids.device)
