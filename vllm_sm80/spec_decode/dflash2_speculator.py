# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's DFlash2 speculator with the load-following draft width and the
split draft tail.

- Load-following width (``adaptive_k.draft_by_load``): the drafter drafts the
  width the next step verifies (:class:`DraftWidthMixin`), with one set of
  drafter graphs per width.
- Split draft tail (``VLLM_PP_DRAFT_TAIL_STAGE``, last pipeline stage only):
  the drafter forward (and its CUDA graphs) stop after gathering the tail's
  inputs into a staging buffer; the tail then runs here eagerly from those
  inputs, or on the tail stage (``vllm_sm80.worker.pp_draft_tail``).

The speculator factory (``vllm_sm80.spec_decode.speculators``) builds this
class for the plain DFlash2 drafters; LiLiCorr keeps vLLM's class.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as dflash2
from vllm_sm80.spec_decode.adaptive_k import config_of as adaptive_k_config_of
from vllm_sm80.spec_decode.dflash_width import DraftWidthMixin
from vllm_sm80.worker.pp_draft_tail import (
    TailPayloadLayout,
    TailPayloadViews,
    draft_tail_gate,
    pack_tail_payload,
    run_draft_tail,
)

logger = init_logger(__name__)


def load_following_draft_widths(
    vllm_config: VllmConfig, log: bool = False
) -> tuple[int, ...]:
    """Draft widths for ``adaptive_k.draft_by_load``, or () when the
    drafter keeps its configured width. Taken from the config alone, so every
    pipeline stage reaches the same answer. ``log``: say why the gate is
    closed (the drafter's process only)."""
    spec = getattr(vllm_config, "speculative_config", None)
    config = adaptive_k_config_of(vllm_config) if spec is not None else None
    if config is None or not getattr(config, "draft_by_load", False):
        return ()
    reason = ""
    if spec.draft_sample_method != "greedy":
        reason = "needs greedy draft sampling"
    elif spec.enable_adaptive_verification:
        reason = "not with adaptive verification"
    elif vllm_config.parallel_config.data_parallel_size != 1:
        reason = "not with data parallelism"
    if reason:
        if log:
            # Plain warning, not *_once: the drafter lives on the last pipeline
            # stage, and *_once logs only on the local first rank.
            logger.warning(
                "adaptive_k.draft_by_load set but the drafter "
                "keeps its full block: %s.",
                reason,
            )
        return ()
    return tuple(config.allowed)


def log_draft_width_banner(widths) -> None:
    """The drafter's banner. A plain info line, once per drafter (one per
    process): the drafter lives on the last pipeline stage, where *_once
    (local first rank only) would never print it."""
    logger.info(
        "GLM-5 load-following DFlash draft width active "
        "(adaptive_k.draft_by_load): the drafter drafts the width "
        "the next step verifies; drafter graphs for widths %s",
        tuple(widths),
    )


class Sm80DFlash2Speculator(DraftWidthMixin, dflash2.DFlash2Speculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.split_tail = False
        self._tail_layout: TailPayloadLayout | None = None
        self._tail_staging: torch.Tensor | None = None
        self._tail_row_ids: torch.Tensor | None = None
        self._tail_packed_rows: int | None = None
        self.tail_rows: int | None = None
        self._tail_layouts: dict[int, TailPayloadLayout] = {}
        self._tail_row_ids_by_width: dict[int, torch.Tensor] = {}
        self._anchor_indices_by_width: dict[int, torch.Tensor] = {}
        # The stage's draft-tail controller (attach_speculator), which decides
        # per step whether the tail runs on the tail stage.
        self.draft_tail_controller: Any = None
        widths = load_following_draft_widths(vllm_config, log=True)
        if widths:
            self.enable_draft_widths(widths)
            log_draft_width_banner(self.widths)
        # The tail moves only from the last pipeline stage, the one that
        # builds the speculator; set before any forward or capture.
        from vllm import envs

        if draft_tail_gate(vllm_config, envs.VLLM_PP_DRAFT_TAIL_STAGE).enabled:
            self.enable_split_tail()

    def _on_width(self, width: int) -> None:
        self.candidate_sampler.num_steps = width
        if self.split_tail:
            self._tail_layout = self._tail_layouts[width]
            self._tail_row_ids = self._tail_row_ids_by_width[width]
            self._anchor_indices = self._anchor_indices_by_width[width]

    def enable_split_tail(self) -> None:
        """Split the drafter's tail off its forward (see VLLM_PP_DRAFT_TAIL_STAGE).

        Draft tokens are unchanged: the tail runs the same kernels on the same
        rows, only from a gathered copy of its inputs. One payload layout per
        draft width; the staging buffer is sized for the widest.
        """
        for k in self.widths:
            self._tail_layouts[k] = TailPayloadLayout(
                max_rows=self.max_num_reqs,
                num_steps=k,
                hidden_size=self.hidden_size,
                hidden_dtype=self.dtype,
                anchor_dtype=self.input_buffers.input_ids.dtype,
            )
            self._tail_row_ids_by_width[k] = (
                torch.arange(
                    self.max_num_reqs * k, dtype=torch.int32, device=self.device
                )
                // k
            )
            # Row r's anchor token sits at input_ids[r * (1 + k)], the rows
            # _score_candidates reads.
            self._anchor_indices_by_width[k] = torch.arange(
                self.max_num_reqs, dtype=torch.int64, device=self.device
            ) * (1 + k)
        self._tail_staging = torch.zeros(
            self._tail_layouts[self.max_width].nbytes(self.max_num_reqs),
            dtype=torch.uint8,
            device=self.device,
        )
        self.split_tail = True
        self._on_width(self.width)

    def tail_layout_for(self, width: int) -> TailPayloadLayout:
        return self._tail_layouts[width]

    @property
    def tail_layout(self) -> TailPayloadLayout:
        assert self._tail_layout is not None
        return self._tail_layout

    def tail_staging_views(self, rows: int) -> TailPayloadViews:
        assert self._tail_staging is not None
        return self.tail_layout.views(self._tail_staging, rows)

    def tail_payload(self, rows: int) -> torch.Tensor:
        """The staged tail inputs for ``rows`` rows, as the bytes to send."""
        assert self._tail_staging is not None
        return self._tail_staging[: self.tail_layout.nbytes(rows)]

    def _walk_from_payload(
        self,
        out_tokens: torch.Tensor,
        realized_scores: torch.Tensor,
    ):
        block_k = triton.next_power_of_2(self.top_k)

        def walk(candidate_ids, scores, views: TailPayloadViews, rows: int) -> None:
            dflash2._selector_walk_kernel[(rows,)](
                scores.contiguous(),
                candidate_ids.contiguous(),
                views.sample_pos,
                views.row_state,
                views.temperature,
                views.seeds,
                out_tokens,
                realized_scores,
                num_steps=self.num_speculative_steps,
                top_k=self.top_k,
                BLOCK_K=block_k,
                SAMPLE_PROBABILISTIC=self.draft_logits is not None,
                USE_FP64=self.use_fp64_gumbel,
                num_warps=1,
            )

        return walk

    def run_tail_local(self, rows: int) -> None:
        """The tail on this stage, from the staged inputs (split mode)."""
        assert self.split_tail
        candidate_ids, _ = run_draft_tail(
            self.model.compute_candidates,
            self.model.model.candidate_selector,
            self._walk_from_payload(self.draft_tokens, self.candidate_sampler.scores),
            self.tail_staging_views(rows),
            rows,
            self.num_speculative_steps,
            self.top_k,
        )
        # Same follow-ups as the fused path (the moved tail is gated to
        # configurations that need neither).
        num_sample = rows * self.num_speculative_steps
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:rows].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )
        if self.draft_logits is not None:
            self.candidate_sampler._cache_draft_logits(
                candidate_ids, num_sample, self.sample_idx_mapping, self.draft_logits
            )

    def _split_tail_rows(self, num_reqs: int, dummy_run: bool, is_profile: bool) -> int:
        """Rows the drafter forward just ran over (its CUDA-graph padding)."""
        if self._tail_packed_rows is not None:
            # An eager forward (or a capture) recorded its row count.
            return self._tail_packed_rows
        # A replayed graph: the dispatch is a pure function of the batch.
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_reqs * self.num_query_per_req,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )
        if batch_desc.num_reqs:
            return batch_desc.num_reqs
        return min(batch_desc.num_tokens, self.max_num_reqs)

    def graph_rows_table(self) -> dict[int, dict[int, int]]:
        """Draft width -> (request count -> rows the drafter forward runs
        over), for every batch size (sent to the tail stage once, after
        capture)."""
        tables: dict[int, dict[int, int]] = {}
        current = self.width
        for width in self.widths:
            self.set_width(width)
            table: dict[int, int] = {}
            for num_reqs in range(1, self.max_num_reqs + 1):
                batch_desc, _ = dispatch_cg_and_sync_dp(
                    self.query_cudagraph_manager,
                    num_reqs,
                    num_reqs * self.num_query_per_req,
                    uniform_token_count=self.num_query_per_req,
                    dp_size=self.dp_size,
                    dp_rank=self.dp_rank,
                )
                if batch_desc.cg_mode == CUDAGraphMode.FULL:
                    table[num_reqs] = batch_desc.num_reqs or min(
                        batch_desc.num_tokens, self.max_num_reqs
                    )
                else:
                    table[num_reqs] = num_reqs
            tables[width] = table
        self.set_width(current)
        return tables

    def propose(self, input_batch, *args, **kwargs):
        """vLLM's propose; in split mode the tail runs here afterwards, or on
        the tail stage (then the drafts are not known here: returns None)."""
        if not self.split_tail:
            return super().propose(input_batch, *args, **kwargs)
        controller = self.draft_tail_controller
        remote_tail = controller is not None and controller.remote_step(input_batch)
        self._tail_packed_rows = None
        super().propose(input_batch, *args, **kwargs)
        num_reqs = input_batch.num_reqs
        rows = self._split_tail_rows(
            num_reqs,
            kwargs.get("dummy_run", False),
            kwargs.get("is_profile", False),
        )
        self.tail_rows = rows
        if remote_tail:
            # The tail stage computes this step's drafts from the staged inputs.
            return None
        self.run_tail_local(rows)
        return self.draft_tokens[:num_reqs]

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if not self.split_tail:
            return super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        # Split mode: the drafter's layers as in vLLM's _generate_draft, then
        # gather the tail's inputs instead of running it. Runs inside the
        # captured graph (static shapes for a given num_reqs).
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        pack_tail_payload(
            self.tail_staging_views(num_reqs),
            num_reqs,
            self.num_speculative_steps,
            last_hidden_states,
            self.sample_indices,
            self.input_buffers.input_ids,
            self._anchor_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self._tail_row_ids,
        )
        self._tail_packed_rows = num_reqs
