# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark query blocks at the scheduler's requested speculative width."""

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_sm80.spec_decode.adaptive_k import config_of


def draft_widths(vllm_config: VllmConfig) -> tuple[int, ...]:
    spec = vllm_config.speculative_config
    assert spec is not None
    widths = {spec.num_speculative_tokens}
    if spec.uses_dynamic_speculative_decoding():
        widths.update(build_dynamic_sd_schedule_lookup(
            spec.num_speculative_tokens_per_batch_size,
            vllm_config.scheduler_config.max_num_seqs,
            spec.num_speculative_tokens,
        )[1:])
    adaptive = config_of(vllm_config)
    if adaptive is not None and adaptive.draft_by_load:
        widths.update(adaptive.allowed)
    # Zero means context-KV only, not a query block (DFlashSpeculator.propose).
    return tuple(sorted(w for w in widths if w > 0))


def create_dspark_speculator(
    vllm_config: VllmConfig, device: torch.device
) -> DSparkSpeculator:
    if envs.VLLM_DSPARK_ADAPTIVE_DEPTH and envs.VLLM_DSPARK_CONFIDENCE_DUMP:
        # The dump would record the drafts the mask cut as rejected by the target.
        raise ValueError(
            "VLLM_DSPARK_ADAPTIVE_DEPTH and VLLM_DSPARK_CONFIDENCE_DUMP cannot be "
            "set together: the dump would label drafts the mask cut as rejected."
        )
    cls = DSparkSpeculator
    if envs.VLLM_DSPARK_PER_K_WIDTH:
        cls = PerKDSparkSpeculator
    if envs.VLLM_DSPARK_CONFIDENCE_DUMP:
        from vllm_sm80.spec_decode.dspark_confidence import confidence_dump_class

        cls = confidence_dump_class(cls)
    if envs.VLLM_DSPARK_ADAPTIVE_DEPTH:
        from vllm_sm80.spec_decode.dspark_row_mask_conf import row_mask_class

        cls = row_mask_class(cls)
    return cls(vllm_config, device)


class PerKDSparkSpeculator(DSparkSpeculator):
    """Reuse DFlash's per-width graphs/views with DSpark's anchor layout.

    The maximum-width logits cache retains its request/column strides: the
    rejection sampler owns that tensor across width changes. Only the proposal
    buffer is packed, as in DFlash. Narrow blocks can change proposals for a
    non-causal backbone; they do not change the target's acceptance algorithm.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec = vllm_config.speculative_config
        assert spec is not None
        if not getattr(spec.draft_model_config.hf_config, "sample_from_anchor", True):
            raise ValueError(
                "VLLM_DSPARK_PER_K_WIDTH requires sample_from_anchor=True; "
                "fill-in graph descriptors are not restricted to the selected width"
            )
        super().__init__(vllm_config, device)
        self._anchor_indices_by_width = {self.max_width: self._anchor_idx}
        self._confidence_by_width = {
            self.max_width: self.draft_token_confidence_probs
        }
        self.enable_draft_widths(draft_widths(vllm_config))

    def enable_draft_widths(self, widths) -> None:
        super().enable_draft_widths(widths)
        full_confidence = self._confidence_by_width[self.max_width]
        for width in self.widths:
            if width in self._anchor_indices_by_width:
                continue
            self._anchor_indices_by_width[width] = (
                torch.arange(self.max_num_reqs, dtype=torch.int64, device=self.device)
                * width
            )
            self._confidence_by_width[width] = full_confidence[:, :width]

    def _on_width(self, width: int) -> None:
        # Persistent tensors are selected, not rewritten: each captured graph
        # keeps its own immutable anchors through arbitrary width transitions.
        self._anchor_idx = self._anchor_indices_by_width[width]

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # The head writes a (requests, K) block. Publish the original maximum
        # width to AdaptiveVerificationManager, whose request-slot cache keeps
        # that layout even when the scheduled proposals are a shorter prefix.
        full_confidence = self.draft_token_confidence_probs
        self.draft_token_confidence_probs = self._confidence_by_width[self.width]
        try:
            super()._sample_sequential(num_reqs, head_hidden)
        finally:
            self.draft_token_confidence_probs = full_confidence

