# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark row mask: reject the drafts on masked verify rows.

A logits row's draft is the input token of that row (row j of a request
holds draft j). When the step's target forward ran with the row mask
(``dspark_row_mask.py``), the rows past a request's ``live_len`` took no
routed expert, so their drafts must not be verified: ``cap_drafts`` adds
them to the rejection sampler's ``invalid_drafts`` mask, which verifies them
as -1 proposals (``RejectionSampler.__call__``: only the verification copy of
the draft ids is changed; sampling params, watermarking and the forward's
input ids keep the real drafts).

A -1 draft at row j + 1 stops verification at row j: greedy stores the
target argmax of row j, probabilistic samples row j's target distribution
with the request's seed at row j's position (no residual), exactly as for a
step that proposed only drafts 1..j. Rows 0..live_len are live, so every
emitted token comes from a live row, and greedy output is the unmasked
output truncated at ``live_len`` accepted drafts. The masked drafts count as
proposed and rejected: ``num_rejected`` (and so the drafter's next context
and the scheduler's rollback) covers them.
"""

from __future__ import annotations

import torch

from vllm_sm80.spec_decode.dspark_row_mask import get_row_mask_state


def cap_drafts(
    logits_indices: torch.Tensor, invalid_drafts: torch.Tensor | None
) -> torch.Tensor | None:
    """``invalid_drafts`` with the drafts on masked rows added.

    ``logits_indices`` [num_logits]: the token row of each logits row of the
    sampler's batch (with batch-sharded sampling, the local rows, which still
    index the forward's tokens). ``invalid_drafts``: None or bool
    [num_logits]. Returns the input unchanged (no kernel) unless this step's
    forward applied a mask.
    """
    state = get_row_mask_state()
    if state is None or not state.active:
        return invalid_drafts
    masked = state.row_live.index_select(0, logits_indices).eq(0)
    if invalid_drafts is None:
        return masked
    return masked.logical_or_(invalid_drafts)
