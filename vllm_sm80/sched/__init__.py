# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's scheduler: fair chunked prefill, PP decode spreading,
load-adaptive draft count and the Mamba state grid.

``vllm_sm80.sched.scheduler`` holds ``Sm80Scheduler`` /
``Sm80AsyncScheduler``, selected through ``SchedulerConfig.scheduler_cls``
(:func:`select_scheduler`). ``vllm_sm80.sched.hooks`` holds the few calls
``vllm/v1/core/sched/scheduler.py`` makes from inside ``schedule()``'s
request loops, where no extension point exists; they do nothing for a
scheduler without the fork mixin.

This module imports nothing: the API server selects the scheduler without
loading the scheduler code.
"""

SCHEDULERS = (
    "vllm_sm80.sched.scheduler.Sm80Scheduler",
    "vllm_sm80.sched.scheduler.Sm80AsyncScheduler",
)


def select_scheduler(vllm_config) -> None:
    """Make ``Sm80Scheduler`` / ``Sm80AsyncScheduler`` the engine's scheduler
    when no scheduler class is configured (``--scheduler-cls`` unset) and the
    model is not a diffusion model (upstream picks its diffusion scheduler
    after the platform hook). Called last from the fork platform's
    ``check_and_update_config``, after upstream has settled async
    scheduling."""
    scheduler_config = vllm_config.scheduler_config
    if (
        scheduler_config.scheduler_cls is not None
        or vllm_config.diffusion_config is not None
    ):
        return
    scheduler_config.scheduler_cls = SCHEDULERS[bool(scheduler_config.async_scheduling)]
