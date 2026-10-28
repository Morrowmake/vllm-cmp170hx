# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's pipeline-parallel sampled-token handler with the fork's draft
transport. The V2 model runner builds it in place of vLLM's ``PPHandler``
(``vllm_sm80.worker.runner_hooks.pp_handler``).

- Split draft event (``VLLM_PP_SPLIT_DRAFT_EVENT``, non-last stages with
  speculative decoding): the draft tokens are waited for separately and as
  late as possible (:meth:`apply_pending_drafts`, right before the forward),
  so the input preparation before it overlaps the last stage's drafter.
- Moved draft tail (``VLLM_PP_DRAFT_TAIL_STAGE``): the last stage sends the
  drafter's tail inputs to the tail stage (:meth:`send_draft_tail`), which
  computes the drafts and roots their broadcast; the last stage lands them
  before a step reads them (:meth:`apply_remote_drafts`).

``receive`` and ``get_prev_sampled_outputs`` stay vLLM's; the two calls they
make into ``vllm_sm80.worker.pp_hooks`` read the state set here.
"""

from typing import Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed import get_pp_group
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm_sm80.worker.pp_draft_tail import RemoteDraftQueue, RemoteDrafts
from vllm_sm80.worker.pp_hooks import PendingDrafts


class Sm80PPHandler(PPHandler):
    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        *,
        draft_tail: Any = None,
        max_concurrent_batches: int = 1,
    ):
        super().__init__(
            max_num_reqs=max_num_reqs,
            num_speculative_steps=num_speculative_steps,
            device=device,
        )
        self.split_draft_event = (
            envs.VLLM_PP_SPLIT_DRAFT_EVENT
            and not self.is_last_rank
            and num_speculative_steps > 0
        )
        self.pending_drafts: PendingDrafts | None = None
        self.early_event: torch.cuda.Event | None = None

        # The stage (rank within the PP group) that runs the drafter's tail,
        # with its own communicator for the payload hop from the last stage.
        # Created on every rank (a new NCCL group).
        draft_tail_stage = draft_tail.gate.stage if draft_tail is not None else -1
        self.draft_tail_stage = draft_tail_stage
        self.draft_tail_src = -1
        self.tail_group = None
        self.remote_drafts: RemoteDraftQueue | None = None
        if draft_tail_stage >= 0 and num_speculative_steps > 0:
            pp_group = get_pp_group()
            self.draft_tail_src = pp_group.ranks[draft_tail_stage]
            self.tail_group = pp_group.make_sibling_device_group(
                group_desc="pp_draft_tail"
            )
            if self.is_last_rank:
                # An entry older than every step that can be in flight has
                # long landed; waiting on it costs nothing.
                self.remote_drafts = RemoteDraftQueue(
                    max_age=max(2, max_concurrent_batches + 1)
                )
        # Per step: the draft width the scheduler asked for (the runner sets
        # it before receive), this step's DraftTailStep while receive runs,
        # and the last stage's tail payload until broadcast_drafts sends it.
        self.draft_tail = draft_tail
        self.next_width: int | None = None
        self.draft_tail_step: Any = None
        self.tail_payload: torch.Tensor | None = None
        if draft_tail is not None:
            draft_tail.attach_handler(self)

    # ---- non-last stages ---------------------------------------------------

    def receive(self, input_batch: InputBatch) -> bool:
        """vLLM's receive; with the moved draft tail this step's drafts are
        rooted at the tail stage (which also computes them), and with the split
        draft event the slot's ``event`` covers only the sampled tokens and
        counts while ``draft_event`` covers the drafts."""
        tail = self.draft_tail
        self.draft_tail_step = (
            tail.step(input_batch, self.next_width) if tail is not None else None
        )
        self.early_event = None
        before = self.queue[-1]
        try:
            all_decode_next = super().receive(input_batch)
        finally:
            self.draft_tail_step = None
        slot = self.queue[-1]
        if self.early_event is not None and slot is not before:
            # vLLM's event was recorded after the draft broadcast.
            slot.draft_event = slot.event
            slot.event = self.early_event
        self.early_event = None
        return all_decode_next

    def has_pending_drafts(self) -> bool:
        return self.pending_drafts is not None

    def apply_pending_drafts(self, draft_tokens_to_update: torch.Tensor) -> None:
        """Wait (on the main stream) for the deferred draft tokens and scatter
        them into the request state. Rows whose request was freed since the
        results were consumed are skipped, so a re-used row keeps the state
        its new request wrote."""
        pending = self.pending_drafts
        if pending is None:
            return
        self.pending_drafts = None
        freed = self.req_idx_gen_np[pending.idx_mapping_np] != pending.gen_at_consume_np
        keep = pending.keep_np & ~freed
        if not keep.any():
            return
        self.main_stream.wait_event(pending.event)
        draft_tokens = pending.draft_tokens
        if keep.all():
            idx_mapping_np = pending.idx_mapping_np
        else:
            keep_t = torch.as_tensor(keep, device=draft_tokens.device)
            draft_tokens = draft_tokens[keep_t]
            idx_mapping_np = pending.idx_mapping_np[keep]
        idx_mapping = pp_utils.async_tensor_h2d(idx_mapping_np, device=self.device)
        draft_tokens_to_update[idx_mapping] = draft_tokens

    # ---- last stage ----------------------------------------------------------

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        payload, self.tail_payload = self.tail_payload, None
        if payload is not None:
            # This step's tail runs on the tail stage.
            self.send_draft_tail(payload, input_batch)
            return
        super().broadcast_drafts(draft_tokens, input_batch)

    def send_draft_tail(
        self,
        payload: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        """Last stage, a step whose tail runs on the tail stage: send the
        payload there, then receive the drafts it roots like any other stage.
        The drafts reach the request state through ``apply_remote_drafts``."""
        assert self.is_last_rank and self.remote_drafts is not None
        need_sampled_mask = pp_utils.compute_need_sampled_mask(input_batch)
        assert need_sampled_mask is not None
        num_reqs = input_batch.num_reqs
        gen_np = self.req_idx_gen_np[input_batch.idx_mapping_np].copy()
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # Allocated on the main stream (the drafter wrote it there).
            payload.record_stream(self.broadcast_stream)
            torch.distributed.send(
                payload, dst=self.draft_tail_src, group=self.tail_group
            )
            draft_tokens = torch.empty(
                num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=self.device,
            )
            torch.distributed.broadcast(
                draft_tokens, src=self.draft_tail_src, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            draft_tokens.record_stream(self.main_stream)
        self.remote_drafts.push(
            RemoteDrafts(
                step=self.remote_drafts.step,
                event=event,
                draft_tokens=draft_tokens,
                idx_mapping_np=input_batch.idx_mapping_np.copy(),
                keep_np=need_sampled_mask.copy(),
                gen_np=gen_np,
            )
        )

    def apply_remote_drafts(
        self, batch_idx_np: np.ndarray, draft_tokens_to_update: torch.Tensor
    ) -> None:
        """Last stage, before a step reads draft tokens: scatter the drafts of
        every earlier step that holds one of this batch's requests (waiting on
        the main stream only for those)."""
        queue = self.remote_drafts
        if queue is None:
            return
        queue.next_step()
        if not len(queue):
            return
        queue.apply(
            batch_idx_np,
            self.req_idx_gen_np,
            draft_tokens_to_update,
            self.main_stream.wait_event,
            lambda idx_np: pp_utils.async_tensor_h2d(idx_np, device=self.device),
        )
