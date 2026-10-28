# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calls ``vllm/v1/core/sched/scheduler.py`` makes into the fork.

``Scheduler.schedule()`` has no extension point inside its two request loops,
where fair chunked prefill charges its prefill budget, PP decode spreading
skips decodes and a cached prompt tail keeps its prefill shape. The upstream
file makes one call per site into the per-step object :func:`step` returns.
For a scheduler without the fork mixin (``vllm_sm80.sched.scheduler``) that
object is :data:`UPSTREAM_STEP`, whose every method leaves the schedule as
upstream computes it.

This module is imported by the upstream scheduler module, so it imports
nothing from it.
"""

from typing import Any

_NO_FAILED_INPUTS: frozenset[str] = frozenset()


class UpstreamStep:
    """Per-step hooks of a scheduler without the fork mixin: no-ops."""

    __slots__ = ()

    def skip(self, request: Any) -> bool:
        return False

    def cap(self, request: Any, num_new_tokens: int) -> int:
        return num_new_tokens

    def admit(self, request: Any, num_computed_tokens: int, num_new_tokens: int) -> int:
        return num_new_tokens

    def ceil(self, token_budget: int, input_budget: int) -> int | None:
        return None

    def refund(self, request: Any, restored: int) -> None:
        pass

    def charge(self, request: Any, num_new_tokens: int) -> None:
        pass

    def prompt_tail(self, request: Any, num_computed_tokens: int) -> bool:
        return False

    def finish(self, total_num_scheduled_tokens: int, num_scheduled_tokens) -> None:
        pass


UPSTREAM_STEP = UpstreamStep()


def sm80_step(
    scheduler: Any,
    token_budget: int,
    draft_slots: int,
    long_prefill_token_threshold: int,
):
    """This step's hook object (called once per ``schedule()``, before the
    running loop)."""
    new_step = getattr(scheduler, "_sm80_new_step", None)
    if new_step is None:
        return UPSTREAM_STEP
    return new_step(token_budget, draft_slots, long_prefill_token_threshold)


def mamba_block_size(scheduler: Any) -> int:
    """Block grid of Mamba align-mode chunk ends: the Mamba group's own block
    size (``Sm80SchedulerMixin.mamba_state_block_size``), else upstream's
    ``cache_config.block_size``."""
    return (
        getattr(scheduler, "mamba_state_block_size", None)
        or scheduler.cache_config.block_size
    )


def failed_inputs(scheduler: Any, model_runner_output: Any):
    """Request IDs whose step output was computed from inputs a worker could
    not produce (``ModelRunnerOutput.failed_req_ids``); see
    ``vllm_sm80.failed_inputs``. Empty when none failed."""
    failed_req_ids = model_runner_output.failed_req_ids
    if not failed_req_ids:
        return _NO_FAILED_INPUTS
    from vllm_sm80.failed_inputs import handle_failed_inputs

    return handle_failed_inputs(scheduler, failed_req_ids)
