# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's scheduler: fair chunked prefill, PP decode spreading,
load-adaptive draft count and the Mamba state grid.

``vllm_sm80.sched.scheduler`` holds ``Sm80Scheduler`` /
``Sm80AsyncScheduler``, selected through ``SchedulerConfig.scheduler_cls``
(:func:`configure_scheduler`). ``vllm_sm80.sched.hooks`` holds the few calls
``vllm/v1/core/sched/scheduler.py`` makes from inside ``schedule()``'s
request loops, where no extension point exists; they do nothing for a
scheduler without the fork mixin.

This module imports no scheduler code: the API server configures the
scheduler without loading it.
"""

SCHEDULERS = (
    "vllm_sm80.sched.scheduler.Sm80Scheduler",
    "vllm_sm80.sched.scheduler.Sm80AsyncScheduler",
)


def configure_scheduler(vllm_config) -> None:
    """Check the fair chunked prefill knobs, then make ``Sm80Scheduler`` /
    ``Sm80AsyncScheduler`` the engine's scheduler when no scheduler class is
    configured (``--scheduler-cls`` unset) and the model is not a diffusion
    model (upstream picks its diffusion scheduler after the platform hook).
    Called last from the fork platform's ``check_and_update_config``, after
    upstream has settled async scheduling."""
    check_fair_prefill(vllm_config)
    scheduler_config = vllm_config.scheduler_config
    if (
        scheduler_config.scheduler_cls is not None
        or vllm_config.diffusion_config is not None
    ):
        return
    scheduler_config.scheduler_cls = SCHEDULERS[bool(scheduler_config.async_scheduling)]


def check_fair_prefill(vllm_config) -> None:
    """Refuse invalid fair chunked prefill knobs; warn that they do nothing
    without chunked prefill (the scheduler then ignores them)."""
    from vllm.logger import init_logger

    from vllm_sm80.config import FAIR_PREFILL_KEYS, section

    settings = section(vllm_config)
    for key in FAIR_PREFILL_KEYS:
        value = settings.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer, got {value!r}")
    cap, partial = (settings.get(key, 0) for key in FAIR_PREFILL_KEYS)
    if not (cap or partial):
        return
    if not vllm_config.scheduler_config.enable_chunked_prefill:
        init_logger(__name__).warning(
            "prefill_chunk_with_decodes / max_num_partial_prefills have no "
            "effect without chunked prefill; ignoring them."
        )
        return
    if 0 < cap < partial:
        raise ValueError(
            f"prefill_chunk_with_decodes ({cap}) must be at least "
            f"max_num_partial_prefills ({partial}) so every concurrent prefill "
            "can be given at least one token per step."
        )
