# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The calls vLLM's V2 model runner (``vllm/v1/worker/gpu/model_runner.py``)
makes into the plugin, each from a point inside a long runner method where
no extension point exists. Everything else of the fork's runner is the runner
subclass (``vllm_sm80.worker.graph_mem_worker``), which the fork worker
always builds.

Every hook is a no-op for a runner and configuration without the feature it
serves, so vLLM's runner built without the plugin worker (tests) runs as
before.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm_sm80.failed_inputs import take_failed_tokens  # noqa: F401
from vllm_sm80.spec_decode.dspark_row_mask_kernel import (  # noqa: F401
    begin_target_step,
)
from vllm_sm80.spec_decode.dspark_row_mask_reject import cap_drafts  # noqa: F401
from vllm_sm80.worker.mem_attribution import ProfileStages  # noqa: F401

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

__all__ = [
    "ProfileStages",
    "after_drafter_load",
    "after_kv",
    "apply_late_drafts",
    "begin_target_step",
    "cap_drafts",
    "cg_miss",
    "configure_aux_fc_fold",
    "copy_hop_rows",
    "draft_combiner",
    "pp_handler",
    "store_drafts",
    "take_failed_tokens",
]


# ---- construction and load --------------------------------------------------


def pp_handler(runner: GPUModelRunner) -> Any:
    """The PP handler class the runner builds (``__init__``): the fork's
    ``Sm80PPHandler``, with this stage's draft tail
    (``VLLM_PP_DRAFT_TAIL_STAGE``), created here as ``runner.draft_tail``."""
    from vllm.distributed import get_pp_group
    from vllm_sm80.worker.pp_draft_tail import DraftTailController
    from vllm_sm80.worker.pp_handler import Sm80PPHandler

    draft_tail = None
    if runner.speculative_config is not None:
        pp = get_pp_group()
        draft_tail = DraftTailController.create(
            runner.vllm_config, runner.device, pp.rank_in_group, pp.world_size
        )
        if draft_tail is not None:
            draft_tail.attach_speculator(runner.speculator)
    runner.draft_tail = draft_tail
    return functools.partial(
        Sm80PPHandler,
        draft_tail=draft_tail,
        max_concurrent_batches=runner.vllm_config.max_concurrent_batches,
    )


def configure_aux_fc_fold(runner: GPUModelRunner, load_dummy_weights: bool) -> None:
    """``load_model``, after the aux hidden-state relay is configured
    (VLLM_GLM5_PP_FOLD_DRAFT_FC): each stage applies its slice of the
    drafter's input projection and forwards one fp32 partial sum instead of
    relaying the aux states."""
    from vllm_sm80.spec_decode.aux_fc_fold import maybe_configure_aux_fc_fold

    runner.aux_fc_folded = maybe_configure_aux_fc_fold(
        runner.model,
        runner.speculative_config,
        runner.pp_handler,
        load_dummy_weights=load_dummy_weights,
        dtype=runner.dtype,
    )


def after_drafter_load(runner: GPUModelRunner) -> None:
    """``load_model``, after the drafter is loaded, inside the load's memory
    window: the folded drafter takes the fc output, and the draft-tail stage
    allocates its lm_head and selector copy."""
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    if getattr(runner, "aux_fc_folded", False) and isinstance(
        runner.speculator, DraftModelSpeculator
    ):
        from vllm_sm80.spec_decode.aux_fc_fold import mark_drafter_aux_fc_folded

        mark_drafter_aux_fc_folded(runner.speculator)
    draft_tail = getattr(runner, "draft_tail", None)
    if draft_tail is not None:
        draft_tail.load(runner.model, runner.dtype)


def after_kv(is_profiling: bool) -> Any:
    """``initialize_kv_cache``: the context around the attention-metadata
    builders' initialization (VLLM_MEM_ATTRIBUTION)."""
    from vllm_sm80.worker.mem_attribution import AfterKvAllocations

    return AfterKvAllocations("graph-profile KV" if is_profiling else "serving KV")


# ---- per step ------------------------------------------------------------------


def _combine():
    # Looked up at call time: the runner module's binding is the one tests patch.
    from vllm.v1.worker.gpu import model_runner

    return model_runner.combine_sampled_and_draft_tokens


def draft_combiner(runner: GPUModelRunner, idx_mapping_np):
    """``prepare_inputs``: ``combine_sampled_and_draft_tokens`` as the runner
    calls it, with the fork's draft transport around it. A last stage with a
    moved draft tail first lands the drafts this batch reads; while the
    previous step's draft values are still in flight
    (``VLLM_PP_SPLIT_DRAFT_EVENT``) the call wrote stale ones, and
    :func:`apply_late_drafts` re-runs it (a pure function of the request
    state) once they land."""

    def combine(*args):
        handler = getattr(runner, "pp_handler", None)
        if getattr(handler, "remote_drafts", None) is not None:
            handler.apply_remote_drafts(idx_mapping_np, runner.req_states.draft_tokens)
        logits_indices = _combine()(*args)
        if getattr(handler, "pending_drafts", None) is not None:
            runner._late_combine_args = args
        return logits_indices

    return combine


def apply_late_drafts(runner: GPUModelRunner, dummy_run: bool) -> None:
    """``execute_model``, right before the forward (VLLM_PP_SPLIT_DRAFT_EVENT):
    once the previous step's draft tokens have landed, scatter them into the
    request state and rebuild this step's input ids. The stream waits only
    here, so the input preparation and attention metadata before it overlap
    the last stage's drafter."""
    handler = getattr(runner, "pp_handler", None)
    if dummy_run or getattr(handler, "pending_drafts", None) is None:
        return
    handler.apply_pending_drafts(runner.req_states.draft_tokens)
    args = getattr(runner, "_late_combine_args", None)
    if args is not None:
        _combine()(*args)
        runner._late_combine_args = None


def cg_miss(
    runner: GPUModelRunner,
    batch_desc: Any,
    num_reqs: int,
    num_toks: int,
    uniform_tok_count: int | None,
    dummy_run: bool,
) -> None:
    """``execute_model``, after the CUDA-graph dispatch: an adaptive-k uniform
    decode batch that found no graph of the configured decode mode runs
    PIECEWISE or eager, which would otherwise read as an unexplained
    slowdown."""
    if (
        not getattr(runner, "adaptive_k_enabled", False)
        or uniform_tok_count is None
        or dummy_run
    ):
        return
    from vllm_sm80.worker.adaptive_k_graphs import note_graph_miss

    note_graph_miss(runner, batch_desc, num_reqs, num_toks, uniform_tok_count)


def copy_hop_rows(buffer: torch.Tensor, received: torch.Tensor, n: int) -> torch.Tensor:
    """``execute_model`` on a non-first stage: the received intermediate
    tensor into the first ``n`` rows of the persistent buffer. A packed PP hop
    without metadata (``VLLM_PP_HOP_NO_METADATA``) sends only the scheduled
    rows; the CUDA-graph padding rows keep what the buffer holds (their
    outputs are never read)."""
    m = min(n, received.shape[0])
    buffer[:m].copy_(received[:m])
    return buffer[:n]


def store_drafts(
    runner: GPUModelRunner,
    input_batch: InputBatch,
    draft_tokens: torch.Tensor | None,
    num_spec_tokens: int,
) -> bool:
    """``sample_tokens``, after the drafter: store drafts vLLM's write cannot.
    Returns True when done here.

    - None: the draft tail runs on the tail stage this step; keep its payload
      for the PP handler's draft broadcast (``Sm80PPHandler.broadcast_drafts``).
    - A block narrower than the request state (the drafter captures one block
      per load-following width): write its prefix.
    """
    if draft_tokens is None:
        draft_tail = getattr(runner, "draft_tail", None)
        if draft_tail is None:
            return False
        runner.pp_handler.tail_payload = draft_tail.payload(input_batch.num_reqs)
        return True
    if draft_tokens.shape[1] == runner.req_states.draft_tokens.shape[1]:
        return False
    if num_spec_tokens < runner.num_speculative_steps:
        draft_tokens[:, num_spec_tokens:] = -1
    runner.req_states.draft_tokens[input_batch.idx_mapping, : draft_tokens.shape[1]] = (
        draft_tokens
    )
    return True
