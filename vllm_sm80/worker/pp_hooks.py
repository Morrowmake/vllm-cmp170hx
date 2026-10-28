# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two calls vLLM's ``PPHandler`` (``vllm/v1/worker/gpu/pp_utils.py``)
makes into the plugin, on non-last pipeline stages. Both are no-ops for
vLLM's own handler; the fork handler (``vllm_sm80.worker.pp_handler``) sets
the state they read.

- :func:`drafts_src` (``receive``, before the draft-token broadcast): the
  broadcast's root. With the split draft event (``VLLM_PP_SPLIT_DRAFT_EVENT``)
  it first records the event the sampled tokens and counts complete with; with
  the moved draft tail (``VLLM_PP_DRAFT_TAIL_STAGE``) the drafts are rooted at
  the tail stage, which first receives the payload and computes them.
- :func:`defer_drafts` (``get_prev_sampled_outputs``, before the draft
  scatter): with the split draft event, keep the drafts for
  ``apply_pending_drafts`` right before the forward instead of scattering them.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class PendingDrafts:
    """Draft tokens consumed from a slot whose values may still be in flight."""

    event: torch.cuda.Event
    draft_tokens: torch.Tensor  # [num_reqs, num_speculative_steps]
    idx_mapping_np: np.ndarray  # [num_reqs]
    keep_np: np.ndarray  # [num_reqs] bool: rows to scatter
    gen_at_consume_np: np.ndarray  # [num_reqs]


def drafts_src(handler: Any, draft_tokens: torch.Tensor) -> int:
    """The root of this step's draft-token broadcast (``receive``, on the
    broadcast stream, after ``draft_tokens`` is allocated)."""
    if getattr(handler, "split_draft_event", False):
        # The sampled tokens and counts are all the next step's input
        # preparation needs; the draft values come after the drafter.
        handler.early_event = handler.broadcast_stream.record_event()
    step = getattr(handler, "draft_tail_step", None)
    if step is None:
        return handler.last_rank
    if step.compute is not None:
        # Tail stage: take the payload from the last stage, run the tail here
        # and root the draft broadcast.
        payload = torch.empty(
            step.payload_nbytes, dtype=torch.uint8, device=handler.device
        )
        torch.distributed.recv(payload, src=handler.last_rank, group=handler.tail_group)
        step.compute(payload, draft_tokens)
    return handler.draft_tail_src


def defer_drafts(
    handler: Any,
    slot: Any,
    exclude_mask: np.ndarray,
    draft_tokens_to_update: torch.Tensor | None,
) -> None:
    """Split draft event: defer the slot's draft scatter to
    ``apply_pending_drafts`` (right before the forward). A pending entry left
    by a step that never reached its forward is applied first."""
    draft_event = getattr(slot, "draft_event", None)
    if (
        draft_event is None
        or slot.draft_tokens is None
        or draft_tokens_to_update is None
    ):
        return
    handler.apply_pending_drafts(draft_tokens_to_update)
    handler.pending_drafts = PendingDrafts(
        event=draft_event,
        draft_tokens=slot.draft_tokens,
        idx_mapping_np=slot.idx_mapping_np,
        keep_np=~exclude_mask,
        gen_at_consume_np=handler.req_idx_gen_np[slot.idx_mapping_np].copy(),
    )
    # The scatter now waits for apply_pending_drafts.
    slot.draft_tokens = None
