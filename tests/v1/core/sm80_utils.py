# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``create_scheduler`` for the fork scheduler: upstream's test helper
(tests/v1/core/utils.py) building ``Sm80Scheduler`` / ``Sm80AsyncScheduler``,
the classes the fork's CUDA platform selects, with the fair chunked prefill
knobs in ``additional_config["sm80"]`` where the served flags put them."""

from vllm_sm80.config import with_settings
from vllm_sm80.sched.scheduler import Sm80AsyncScheduler, Sm80Scheduler

from .utils import create_requests  # noqa: F401
from .utils import create_scheduler as create_upstream_scheduler


def create_scheduler(
    *,
    async_scheduling: bool = False,
    scheduler_cls=None,
    prefill_chunk_with_decodes: int = 0,
    max_num_partial_prefills: int = 0,
    **kwargs,
):
    if scheduler_cls is None:
        scheduler_cls = Sm80AsyncScheduler if async_scheduling else Sm80Scheduler
    settings = {
        key: value
        for key, value in (
            ("prefill_chunk_with_decodes", prefill_chunk_with_decodes),
            ("max_num_partial_prefills", max_num_partial_prefills),
        )
        if value
    }

    def build(*, vllm_config, **scheduler_kwargs):
        if settings:
            vllm_config.additional_config = with_settings(
                vllm_config.additional_config, settings
            )
        return scheduler_cls(vllm_config=vllm_config, **scheduler_kwargs)

    return create_upstream_scheduler(
        async_scheduling=async_scheduling, scheduler_cls=build, **kwargs
    )
