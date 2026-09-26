# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP decode spreading (VLLM_PP_SPREAD_DECODES).

With the V2 runner, async scheduling and pipeline parallelism a decode request
is not eligible again until pp_size steps after it was scheduled. Without a
cap the first eligible step takes every decode and the next pp_size - 1
micro-batches get none; with the cap each step takes ceil(decoding / pp_size).

The scheduler is built with pipeline_parallel_size=1 (ParallelConfig checks
the world size against the visible GPUs) and switched to PP=4 afterwards;
only the scheduling arithmetic reads it.
"""

import pytest

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

PP = 4


def _pp_scheduler(spread: bool):
    scheduler = create_scheduler(
        async_scheduling=True, use_v2_model_runner=True, max_num_seqs=16
    )
    scheduler.parallel_config.pipeline_parallel_size = PP
    scheduler.pp_size = PP
    # The async scheduler's decode re-eligibility stride, set from the PP size
    # at construction.
    scheduler.decode_stagger = PP
    scheduler.use_pp = True
    scheduler.pp_spread_decodes = spread
    return scheduler


def _decodes_per_step(scheduler, num_requests: int, num_steps: int) -> list[int]:
    requests = create_requests(
        num_requests=num_requests, num_tokens=8, max_tokens=10_000, ignore_eos=True
    )
    for request in requests:
        scheduler.add_request(request)
    first = scheduler.schedule()  # every prompt prefills in one step
    assert len(first.num_scheduled_tokens) == num_requests
    counts = []
    for _ in range(num_steps):
        out = scheduler.schedule()
        counts.append(len(out.num_scheduled_tokens))
    return counts


def test_without_spreading_one_micro_batch_takes_every_decode():
    counts = _decodes_per_step(_pp_scheduler(spread=False), 8, 11)
    # Eligible again PP steps after the prefill step: all 8 at once, then the
    # other PP-1 micro-batches are empty.
    assert counts == [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0]


def test_spreading_shares_decodes_across_micro_batches():
    counts = _decodes_per_step(_pp_scheduler(spread=True), 8, 11)
    assert counts[:3] == [0, 0, 0]
    # From the first eligible step on, every micro-batch carries its share.
    assert counts[3:] == [2] * 8


@pytest.mark.parametrize("num_requests,share", [(1, 1), (3, 1), (5, 2), (16, 4)])
def test_spreading_share_is_ceil_of_decoding_over_pp(num_requests, share):
    counts = _decodes_per_step(_pp_scheduler(spread=True), num_requests, 12)
    steady = counts[3:]
    assert max(steady) == share
    # Every request is still advanced once per PP steps on average.
    assert sum(steady) >= num_requests * (len(steady) // PP)


def test_spreading_is_inert_without_pp():
    scheduler = create_scheduler(async_scheduling=True, use_v2_model_runner=True)
    assert not scheduler.pp_spread_decodes
    assert scheduler._pp_decode_cap() is None
