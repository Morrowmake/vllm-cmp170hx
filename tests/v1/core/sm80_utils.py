# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``create_scheduler`` for the fork scheduler: upstream's test helper
(tests/v1/core/utils.py) building ``Sm80Scheduler`` / ``Sm80AsyncScheduler``,
the classes the fork's CUDA platform selects."""

from vllm_sm80.sched.scheduler import Sm80AsyncScheduler, Sm80Scheduler

from .utils import create_requests  # noqa: F401
from .utils import create_scheduler as create_upstream_scheduler


def create_scheduler(*, async_scheduling: bool = False, scheduler_cls=None, **kwargs):
    if scheduler_cls is None:
        scheduler_cls = Sm80AsyncScheduler if async_scheduling else Sm80Scheduler
    return create_upstream_scheduler(
        async_scheduling=async_scheduling, scheduler_cls=scheduler_cls, **kwargs
    )
