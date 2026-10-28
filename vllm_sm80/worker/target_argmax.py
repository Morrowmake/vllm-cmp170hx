# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The idea (hand the sampler logits that hold only the vocabulary-parallel
# argmax when every request is greedy) follows GNmac-lab's greedy-argmax
# patch: Copyright (c) 2026 GNmac-lab, MIT.
"""Greedy target sampling with a per-rank argmax, without the all-gather of
the target logits.

``GPUModelRunner.sample`` computes the target logits once
(``self.model.compute_logits``: the vocabulary-parallel head, then an
all-gather over the TP group) before the grammar bitmask, the sampler and the
rejection sampler read them. When every request of the step is greedy and
nothing reads the logits beyond their argmax, that gather only feeds an
argmax.

With ``VLLM_SM80_SHARD_ARGMAX`` the fork's V2 model runner
(``vllm_sm80/worker/graph_mem_worker.py``) runs ``sample`` inside
:func:`step`, which decides from the step's batch, grammar output and sampler
states whether the step is greedy-only (:func:`greedy_only`). While such a
step runs (:func:`active`), a target class with :class:`ShardArgmaxLogits`
(the fork's DeepSeek-V4.1, GLM-5.3 and Qwen4Exp classes) returns
:func:`argmax_rows` from ``compute_logits`` in place of the gathered logits;
every other call (other steps, prompt logprobs, the dummy sampler run)
gathers. With batch-sharded sampling (``--enable-batch-sharded-sampling``) a
greedy-only step samples every row on every rank from the argmax rows, in
place of the all-to-all of the vocabulary shards; other steps shard as
before.

The same switch covers greedy drafting over a vocabulary-parallel head:
``vllm_sm80/spec_decode/shard_argmax.py`` (:func:`shard_argmax.drafting`)
says how the runner's drafter takes its greedy drafts, :func:`refusal`
refuses a drafter the switch cannot serve, and at load (:func:`load`, the
fork runner's ``load_model``) a drafter that samples through vLLM's
``sample_draft`` takes its greedy drafts per rank
(``shard_argmax.adopt_sample_draft``).

:func:`argmax_rows` keeps each rank's own columns
(``compute_logits_local``), takes their argmax per 4096-column chunk and
reduces the chunks over the TP group with the DSpark drafter's slot
all-reduce (``vllm_sm80/spec_decode/shard_argmax.py``: ``chunk_slots``,
``tensor_model_parallel_all_reduce``, ``pick``). The tokens are torch.argmax's
over the gathered row, ties and NaN included: the first index of the largest
value, the first NaN when any is NaN, -0.0 equal to +0.0. The samplers then
get rows that hold 0.0 at that token and -inf everywhere else, in the shape
and dtype ``compute_logits`` returns: the sampler's token, the rejection
sampler's target argmax and the bonus token are that token.

A step is greedy-only when there is no grammar bitmask, the sampler and
rejection sampler are the stock classes (no watermarking or model-specific
sampler), the sampler neither counts NaN in the logits nor returns a sampling
mask, no draft logits narrower than the target vocabulary cut the rejection
sampler's rows, and for every request of the batch: temperature 0, no logits
processing (penalties, logit bias, allowed tokens, bad words, min tokens,
thinking budget, custom processors), no logprobs and no top-k/top-p in the
sampling states (SamplingParams resets them for a greedy request; top-p can
drop the first of tied maxima). On such a step the stock samplers read the
logits only through their argmax: ``apply_sampling_params`` returns them as
they are and ``Sampler.sample`` applies no top-k/top-p. So the sampled
tokens are the gathered path's whenever its rows hold no NaN; on a row with
NaN the token is torch.argmax's first NaN. Every TP rank holds the same
sampling states and batch, so all ranks take the same path and enter the
same collective.

No host synchronization and no data-dependent shape: the slots, the token
ids and the rows have shapes fixed by the step's row count.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_sm80.spec_decode import shard_argmax
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

SWITCH = shard_argmax.SWITCH

# True while the runner samples a greedy-only step (:func:`step`).
_active = False


def _head(model):
    """The module holding the target's ``lm_head`` and ``logits_processor``."""
    return getattr(model, "language_model", model)


class ShardArgmaxLogits:
    """A target model class whose ``compute_logits`` returns
    :func:`argmax_rows` while a greedy-only step samples (:func:`active`),
    and the gathered logits otherwise. Listed before the upstream class; the
    class also needs ``compute_logits_local`` (this rank's vocabulary shard
    of the logits, padding columns kept)."""

    # refusal: compute_logits serves greedy-only steps.
    target_argmax_logits = True

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if _active:
            return argmax_rows(self, hidden_states)
        return super().compute_logits(hidden_states)


def refusal(runner) -> str | None:
    """Why the switch cannot serve this runner: its target, its head or its
    drafter (:func:`shard_argmax.drafting`)."""
    if not getattr(runner, "target_argmax_steps", False):
        return (
            f"the model runner {type(runner).__name__} does not sample in "
            "greedy steps (the fork V2 runner of vllm_sm80/worker/"
            f"graph_mem_worker.py, chosen at boot with {SWITCH}=1)"
        )
    if not getattr(runner, "is_last_pp_rank", True):
        return None  # samples nothing
    model = getattr(runner, "model", None)
    if not getattr(model, "target_argmax_logits", False):
        return (
            f"{type(model).__name__} does not return the argmax rows in a "
            "greedy step (compute_logits)"
        )
    if not hasattr(model, "compute_logits_local"):
        return (
            f"{type(model).__name__} has no vocabulary-shard logits "
            "(compute_logits_local)"
        )
    lm = _head(model)
    head = getattr(lm, "lm_head", None)
    processor = getattr(lm, "logits_processor", None)
    if not hasattr(head, "tp_rank") or processor is None:
        return f"{type(model).__name__} has no vocabulary-parallel LM head"
    if getattr(processor, "logits_as_input", False):
        return "the logits processor takes logits as input"
    if processor.org_vocab_size > shard_argmax.MAX_VOCAB:
        return f"a vocabulary above {shard_argmax.MAX_VOCAB} tokens"
    if head.tp_size == 1:
        return (
            "the LM head is not split over tensor-parallel ranks (TP 1): there "
            "is no all-gather to replace"
        )
    if getattr(runner, "lora_config", None):
        return "LoRA adapters replace the logits processor"
    _, reason = shard_argmax.drafting(getattr(runner, "speculator", None))
    return reason


def set_enabled(runner, on: bool) -> None:
    """Turn the per-rank argmax on or off for ``runner``; on is refused
    (ValueError) for a runner it cannot serve (:func:`refusal`), and logs
    the switch's startup line on the last pipeline stage."""
    if on:
        reason = refusal(runner)
        if reason is not None:
            raise ValueError(f"{SWITCH}: {reason}")
        if getattr(runner, "is_last_pp_rank", True):
            drafts, _ = shard_argmax.drafting(getattr(runner, "speculator", None))
            logger.info_once(
                "%s=1: greedy target sampling takes a per-rank argmax, no "
                "all-gather of the target logits when every request of a step "
                "is greedy; greedy drafts: %s.",
                SWITCH,
                drafts,
            )
    runner._target_shard_argmax = on


def load(runner) -> None:
    """At the fork runner's ``load_model``, with the switch on: refuse a
    runner it cannot serve (:func:`set_enabled`, which also logs the startup
    line) and, on the last pipeline stage, take the greedy drafts of a
    ``sample_draft`` drafter per rank before any draft graph is captured
    (``shard_argmax.adopt_sample_draft``)."""
    if not getattr(envs, SWITCH):
        return
    set_enabled(runner, True)
    if getattr(runner, "is_last_pp_rank", True):
        shard_argmax.adopt_sample_draft(getattr(runner, "speculator", None))


def enabled(runner) -> bool:
    on = getattr(runner, "_target_shard_argmax", None)
    if on is None:
        set_enabled(runner, bool(getattr(envs, SWITCH)))
        on = runner._target_shard_argmax
    return on


def greedy_only(runner, input_batch, grammar_output) -> bool:
    """True when the step's samplers read the target logits only through
    their argmax (see the module docstring). Host state only."""
    if grammar_output is not None:
        return False
    sampler = runner.sampler
    if type(sampler) is not Sampler:
        return False
    rejection = runner.rejection_sampler
    if rejection is not None and type(rejection) is not RejectionSampler:
        return False
    if sampler.compute_nans or sampler.return_sampling_mask:
        return False
    draft_logits = getattr(getattr(runner, "speculator", None), "draft_logits", None)
    if draft_logits is not None:
        vocab = _head(runner.model).logits_processor.org_vocab_size
        if draft_logits.shape[-1] < vocab:
            return False
    idx = input_batch.idx_mapping_np
    states = sampler.sampling_states
    return (
        bool(np.all(states.temperature.np[idx] == 0.0))
        and not np.any(sampler.needs_logits_processing[idx])
        and sampler.get_logprobs_dims(idx) is None
        # No top-k/top-p (get_top_k_top_p): top-p may drop the first of tied
        # maxima. SamplingParams resets both for a greedy request.
        and bool(np.all(states.top_k.np[idx] == states.vocab_size))
        and bool(np.all(states.top_p.np[idx] == 1.0))
    )


@contextmanager
def step(runner, input_batch, grammar_output) -> Iterator[None]:
    """Run one ``GPUModelRunner.sample``: while it runs, :func:`active` says
    whether the target's ``compute_logits`` returns the argmax rows. A
    greedy-only step of a runner with batch-sharded sampling samples
    unsharded: the argmax rows replace the shards' all-to-all, and every rank
    holds the same rows."""
    global _active
    _active = enabled(runner) and greedy_only(runner, input_batch, grammar_output)
    sharder = getattr(runner, "batch_sharder", None) if _active else None
    if sharder is not None:
        runner.batch_sharder = None
    try:
        yield
    finally:
        _active = False
        if sharder is not None:
            runner.batch_sharder = sharder


def active() -> bool:
    return _active


def argmax_rows(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Rows holding 0.0 at torch.argmax of ``model.compute_logits(
    hidden_states)`` and -inf elsewhere, in its shape and dtype, without
    gathering the logits (``shard_argmax.shard_tokens``)."""
    lm = _head(model)
    local = model.compute_logits_local(hidden_states)
    vocab = lm.logits_processor.org_vocab_size
    out = local.new_full((local.shape[0], vocab), float("-inf"))
    if local.shape[0] == 0:
        # Every rank has the same rows: none enters the all-reduce.
        return out
    tokens = shard_argmax.shard_tokens(
        local, lm.lm_head.tp_rank, lm.lm_head.tp_size, vocab
    )
    return out.scatter_(1, tokens[:, None], 0.0)


def warm(runner, hidden_states: torch.Tensor | None = None) -> None:
    """With the switch on, refuse a runner it cannot serve and compile the
    slot kernels on ``hidden_states`` (the dummy sampler run at boot), or on
    one zero row of the model's hidden size (the A/B switch's arm), outside
    graph capture. Every rank of the last pipeline stage calls it with the
    same rows."""
    if not (enabled(runner) and getattr(runner, "is_last_pp_rank", True)):
        return
    if hidden_states is None:
        config = runner.model_config
        hidden_states = torch.zeros(
            1, config.get_hidden_size(), dtype=config.dtype, device=runner.device
        )
    with torch.inference_mode():
        argmax_rows(runner.model, hidden_states)
