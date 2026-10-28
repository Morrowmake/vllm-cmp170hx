# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler phase of the running forward step, for kernels that serve
only prefill steps.

A step is a prefill step when the scheduler gave some request more than
1 + num_speculative_tokens tokens (a prompt or a chunk of one); otherwise it
is a decode step (one token per request, or the verified draft of a
speculative step). This is the rule the Qwen4Exp QSA layer uses to pick its
prefill configuration (``max_query_len > 1 + num_speculative_tokens``). It
is read from the attention metadata of the current forward context, so it
reflects what the scheduler sent, never the row count a layer sees: a decode
step of many requests stays a decode step.

Drafters: an MTP or EAGLE drafter without parallel drafting runs with the
target step's query lengths, so its calls get the step's phase. A drafter
with parallel drafting slots (or DFlash) schedules more than 1 +
num_speculative_tokens tokens per request in decode steps; this rule would
call those steps prefill, so it does not cover such drafters. For them,
``max_decode_step_rows`` bounds the rows a decode step's forward passes can
carry: a kernel that serves only calls of more rows than that bound never
serves a decode step, whatever this rule says. A block drafter (DFlash,
DSpark) runs its block in a forward pass of its own, so its bound is the
larger of the target's verify rows and the drafter's block rows, not their
sum.

Without a forward context, without attention metadata (profiling runs and
piecewise CUDA-graph capture), or with per-microbatch metadata (a list,
under dual-batch overlap; the QSA layer reads the first microbatch), the
step counts as a decode step, so those paths keep the kernels that serve
decode. A captured CUDA graph replays the kernels it was captured with, so a
prefill step padded to a captured graph size runs the decode kernels.
"""

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)


def max_decode_query_len(vllm_config) -> int:
    """Most tokens one request is scheduled in a decode step: the sampled
    token plus the speculative draft being verified."""
    return 1 + vllm_config.num_speculative_tokens


# Drafters whose block runs in a forward pass separate from the target's.
BLOCK_DRAFTERS = ("dflash", "dspark")


def max_decode_step_rows(vllm_config) -> int:
    """Most rows one forward pass of a decode step can carry, target or
    drafter: for every batch size b up to max_num_seqs, b requests, each with
    the sampled token, its k drafts under verification and at most k slots of
    a parallel drafter, where k is the batch-size schedule's value for b
    (num_speculative_tokens_per_batch_size) or, without a schedule or with
    adaptive k, num_speculative_tokens. A block drafter's forward pass is
    separate: b requests of 1 + num_speculative_tokens rows at most (the
    block drafts the full width; DSpark's block is one row narrower), and the
    target's b * (1 + k), whichever is larger."""
    max_seqs = vllm_config.scheduler_config.max_num_seqs
    speculative = vllm_config.speculative_config
    k = 0 if speculative is None else speculative.num_speculative_tokens
    if getattr(speculative, "method", None) in BLOCK_DRAFTERS and k > 0:
        # Schedule values never exceed k, so the drafter's block bounds both.
        return max_seqs * (1 + k)
    by_batch = [k] * (max_seqs + 1)
    schedule = getattr(speculative, "num_speculative_tokens_per_batch_size", None)
    if schedule and k > 0 and getattr(speculative, "adaptive_k", None) is None:
        from vllm.v1.spec_decode.dynamic.utils import (
            build_dynamic_sd_schedule_lookup,
        )

        by_batch = build_dynamic_sd_schedule_lookup(
            schedule, vllm_max_batch_size=max_seqs, vllm_num_speculative_tokens=k
        )
    return max(b * (1 + 2 * by_batch[b]) for b in range(1, max_seqs + 1))


def prefill_step(max_decode_query_len: int) -> bool:
    """True when the running forward step is a prefill step (some request
    has more than ``max_decode_query_len`` tokens scheduled)."""
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return False
    for layer_metadata in metadata.values():
        query_len = getattr(layer_metadata, "max_query_len", None)
        if isinstance(query_len, int):
            return query_len > max_decode_query_len
    if metadata:
        logger.warning_once(
            "No attention metadata of this step carries an integer "
            "max_query_len: the step counts as a decode step and the EXL3 "
            "prefill kernels are not used."
        )
    return False
