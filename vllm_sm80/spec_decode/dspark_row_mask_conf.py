# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark adaptive depth, draft side: the per-position features per request
slot.

With ``VLLM_DSPARK_ADAPTIVE_DEPTH`` set, ``create_dspark_speculator`` mixes
:class:`RowMaskConfidenceMixin` in front of the DSpark speculator class. It

* turns the drafter's confidence head on after the draft model loads (and
  refuses a drafter without one), and creates the shared ``RowMaskState``
  with the policy (``dspark_row_mask.py``), checking its cost table against
  the draft count each batch size verifies;
* when the policy uses the draft logits' top-1 probability or entropy,
  computes them in every draft step from the logits the greedy sampler
  takes (``_sample_logits``), or, with the per-rank argmax sampler, from
  each rank's softmax statistics carried in its exchange
  (``shard_argmax.py``); refused with draft top-k, which truncates them;
* at the end of every draft step's sequential sampling (the default, top-k
  and per-rank-argmax samplers alike, at any draft width) publishes the
  step's features into ``RowMaskState.conf_by_slot`` (``stats_by_slot``):
  request r's row goes to its request-state slot, its first ``width``
  columns are positions 1 .. width and the columns past the drafted width
  are 0 (a draft that does not exist is never accepted);
* on a ``propose`` call that drafts nothing (a zero-draft step stores only the
  context K/V, ``DFlashSpeculator.propose``), resets the batch's slots to 1.0, so no
  later verify step reads an older step's confidences.

The publish is one small kernel recorded inside the captured draft step. It
reads the slots from ``sample_idx_mapping``, the drafter's persistent
per-sample request-state index (column 0 of each request's ``width`` sample
rows): ``prepare_dflash_inputs`` writes it before every draft step, eager or
replayed, with -1 for every request past the batch up to ``max_num_reqs``,
and ``capture`` fills it with -1. A request with slot -1 (graph padding)
writes nothing. The width and the strides are fixed per captured graph: each
draft width has its own graphs, captured at that width.

Drafted tokens do not change: the head and the statistics run on the
sampler's existing inputs, and the publish only reads them.

Refused together with ``VLLM_DSPARK_CONFIDENCE_DUMP`` (the factory): the dump
would record the drafts the mask cut as rejected by the target. Refused with
sampled drafts: the live length would depend on the sampled tokens and bias the
verified distribution.
"""

from __future__ import annotations

import inspect
from functools import cache

import torch

from vllm.logger import init_logger
from vllm_sm80.spec_decode.dspark_row_mask import (
    SWITCH,
    RowMaskState,
    init_row_mask_state,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _publish_kernel(
    conf_ptr,
    conf_stride,
    stats_ptr,
    stats_stride,
    slot_ptr,
    slot_stride,
    out_ptr,
    out_stride,
    stats_out_ptr,
    stats_out_stride,
    num_slots,
    width,
    max_draft,
    stats_width,
    STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    slot = tl.load(slot_ptr + req * slot_stride)
    if (slot >= 0) & (slot < num_slots):
        cols = tl.arange(0, BLOCK)
        conf = tl.load(conf_ptr + req * conf_stride + cols, mask=cols < width, other=0.0)
        tl.store(out_ptr + slot * out_stride + cols, conf, mask=cols < max_draft)
        if STATS:
            src = stats_ptr + req * stats_stride + cols
            dst = stats_out_ptr + slot * stats_out_stride + cols
            top1 = tl.load(src, mask=cols < width, other=0.0)
            ent = tl.load(src + stats_width, mask=cols < width, other=0.0)
            tl.store(dst, top1, mask=cols < max_draft)
            tl.store(dst + max_draft, ent, mask=cols < max_draft)


def publish_confidences(
    confidences: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    num_reqs: int,
    width: int,
    conf_by_slot: torch.Tensor,
    stats: torch.Tensor | None = None,
    stats_by_slot: torch.Tensor | None = None,
) -> None:
    """Write ``confidences[r, :width]`` (zeros past it) into
    ``conf_by_slot[slot_r]`` for each request r < ``num_reqs`` whose slot
    ``slot_r = sample_idx_mapping[r * width]`` is not -1; with ``stats``
    ([>= num_reqs, 2, >= width], contiguous last two dims) likewise
    ``stats[r, :, :width]`` into ``stats_by_slot[slot_r]``.

    ``confidences``: [>= num_reqs, >= width] float32, unit column stride;
    ``conf_by_slot``: [num_slots, max_draft] float32, contiguous;
    ``stats_by_slot``: [num_slots, 2, max_draft] float32, contiguous."""
    if num_reqs == 0:
        return
    max_draft = conf_by_slot.shape[1]
    assert confidences.stride(1) == 1 and conf_by_slot.stride(1) == 1
    assert width <= max_draft
    use_stats = stats is not None
    if use_stats:
        assert stats_by_slot is not None and stats_by_slot.shape[1:] == (2, max_draft)
        assert stats.stride(2) == 1 and stats.stride(1) == stats.shape[2]
        assert stats.shape[2] >= width
    _publish_kernel[(num_reqs,)](
        confidences,
        confidences.stride(0),
        stats if use_stats else confidences,
        stats.stride(0) if use_stats else 0,
        sample_idx_mapping,
        width,
        conf_by_slot,
        conf_by_slot.stride(0),
        stats_by_slot if use_stats else conf_by_slot,
        stats_by_slot.stride(0) if use_stats else 0,
        conf_by_slot.shape[0],
        width,
        max_draft,
        stats.shape[2] if use_stats else 0,
        STATS=use_stats,
        BLOCK=triton.next_power_of_2(max_draft),
    )


def drafts_nothing(speculator, num_speculative_tokens: int | None) -> bool:
    """True when ``propose`` returns before the draft step: a step that
    schedules zero drafts stores only the context K/V (data-parallel ranks
    agree on it in the batch sync)."""
    return num_speculative_tokens == 0


def verify_drafts_by_batch(vllm_config, max_num_reqs: int) -> list[int]:
    """Index b: the most drafts a b-request step verifies: with adaptive_k
    the widest load width at a load of b or more requests (the load counts
    waiting requests too), else the per-batch-size counts, else
    num_speculative_tokens."""
    spec = vllm_config.speculative_config
    k = spec.num_speculative_tokens
    from vllm_sm80.spec_decode.adaptive_k import config_of

    adaptive = config_of(vllm_config)
    if adaptive is not None:
        by_load = list(adaptive.by_load)
        return [0] + [
            max(by_load[min(b, len(by_load)) - 1 :]) for b in range(1, max_num_reqs + 1)
        ]
    if spec.uses_dynamic_speculative_decoding():
        from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

        return build_dynamic_sd_schedule_lookup(
            spec.num_speculative_tokens_per_batch_size, max_num_reqs, k
        )
    return [0] + [k] * max_num_reqs


class RowMaskConfidenceMixin:
    """Mixed in front of a DSpark speculator class by ``row_mask_class``."""

    _row_mask: RowMaskState | None = None
    # [max_num_reqs, 2, max_width]: the draft logits' top-1 probability and
    # entropy per drafted position (only when the policy uses them).
    _row_stats: torch.Tensor | None = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        method = self.speculative_config.draft_sample_method
        if method != "greedy":
            raise ValueError(
                f"{SWITCH} needs greedy drafts (draft_sample_method "
                f"'greedy', got {method!r}): with sampled drafts the live length "
                "depends on the sampled tokens and biases verification."
            )
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if getattr(model.model, "confidence_head", None) is None:
            raise ValueError(
                f"{SWITCH} is set but the DSpark drafter has no "
                "confidence head (the checkpoint ships none, or it did not load)."
            )
        self.use_confidence_head = True
        # The target forward's token rows: the runner pads a step to at most
        # max_num_batched_tokens, the speculator's max_num_tokens.
        self._row_mask = state = init_row_mask_state(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_draft=self.max_width,
            device=self.device,
            verify_drafts=verify_drafts_by_batch(self.vllm_config, self.max_num_reqs),
        )
        assert state is not None
        if state.uses_stats:
            if getattr(self, "_draft_topk", None) is not None:
                raise ValueError(
                    f"{SWITCH}: the policy uses the draft logits' top-1 "
                    "probability or entropy, which draft top-k truncates."
                )
            self._row_stats = torch.zeros(
                self.max_num_reqs, 2, self.max_width, dtype=torch.float32, device=self.device
            )
        return model

    def _sample_logits(self, logits, idx_map, sample_pos, step):
        if self._row_stats is not None:
            from vllm_sm80.spec_decode.dspark_confidence import write_draft_stats

            write_draft_stats(self._row_stats, logits, step)
        return super()._sample_logits(logits, idx_map, sample_pos, step)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # The full [max_num_reqs, max_width] buffer: the per-width drafter
        # swaps in its [:, :width] view only while it samples.
        confidences = self.draft_token_confidence_probs
        super()._sample_sequential(num_reqs, head_hidden)
        state = self._row_mask
        if state is not None:
            publish_confidences(
                confidences,
                self.sample_idx_mapping,
                num_reqs,
                self.num_speculative_steps,
                state.conf_by_slot,
                self._row_stats,
                state.stats_by_slot if self._row_stats is not None else None,
            )

    def propose(self, input_batch, *args, **kwargs):
        result = super().propose(input_batch, *args, **kwargs)
        state = self._row_mask
        if state is None:
            return result
        if "num_speculative_tokens" in kwargs:
            width = kwargs["num_speculative_tokens"]
        else:
            signature = _propose_signature()
            known = {k: v for k, v in kwargs.items() if k in signature.parameters}
            width = signature.bind_partial(self, input_batch, *args, **known).arguments.get(
                "num_speculative_tokens"
            )
        if drafts_nothing(self, width):
            num_reqs = input_batch.num_reqs
            state.conf_by_slot.index_fill_(0, input_batch.idx_mapping[:num_reqs], 1.0)
        return result


@cache
def _propose_signature() -> inspect.Signature:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    return inspect.signature(DFlashSpeculator.propose)


@cache
def row_mask_class(base: type) -> type:
    """``base`` with the confidence head on, published per request slot."""
    return type(f"RowMask{base.__name__}", (RowMaskConfidenceMixin, base), {})
