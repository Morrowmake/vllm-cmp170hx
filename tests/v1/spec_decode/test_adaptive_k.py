# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the load-adaptive per-step draft count."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm_sm80.sched.scheduler import Sm80AsyncScheduler, Sm80Scheduler
from vllm.v1.request import RequestStatus
from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig, AdaptiveKPolicy
from vllm.v1.structured_output import StructuredOutputManager

ACCEPT = {"by_load": [3, 1], "accept": {"costs": [1.0, 1.1, 1.2]}}

# --------------------------------------------------------------------- config


@pytest.mark.parametrize("raw", [{}, {"min": 1, "max": 4}])
def test_config_without_by_load_is_rejected(raw):
    with pytest.raises(ValueError, match="needs by_load"):
        AdaptiveKConfig.from_dict(raw, 4)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown adaptive_k keys"):
        AdaptiveKConfig.from_dict({"by_load": [4, 2], "ema": 1}, 4)


def test_config_rejects_a_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        AdaptiveKConfig.from_dict([4, 2], 4)


def test_snap_rounds_down_to_a_captured_draft_count():
    policy = AdaptiveKPolicy(AdaptiveKConfig.from_dict({"by_load": [7, 4, 2]}, 7))
    assert policy.config.allowed == (2, 4, 7)
    assert policy.snap(7) == 7
    assert policy.snap(6) == 4
    assert policy.snap(4) == 4
    assert policy.snap(3) == 2
    assert policy.snap(2) == 2
    # Below the smallest captured count, clamp up rather than run eager.
    assert policy.snap(1) == 2


# ------------------------------------------------------------------- feedback


def test_padding_marks_are_consumed_in_order_and_forgotten():
    policy = AdaptiveKPolicy(AdaptiveKConfig.from_dict(ACCEPT, 3))
    policy.mark_padded("a")
    policy.mark_padded("a")
    policy.observe("a", num_draft_tokens=3, num_accepted=0)
    policy.observe("a", num_draft_tokens=3, num_accepted=0)
    assert "a" not in policy._acc_s
    policy.observe("a", num_draft_tokens=3, num_accepted=2)
    assert policy._acc_s["a"] == pytest.approx(2.0)
    assert policy._acc_f["a"] == pytest.approx(1.0)

    policy.mark_padded("b")
    policy.forget("b")
    assert "b" not in policy._pending_padded


def test_steps_with_no_drafts_are_ignored():
    policy = AdaptiveKPolicy(AdaptiveKConfig.from_dict(ACCEPT, 3))
    policy.observe("a", num_draft_tokens=0, num_accepted=0)
    assert "a" not in policy._acc_s


# -------------------------------------------------------------------- logging


def test_logging_histogram_resets_only_after_the_interval():
    policy = AdaptiveKPolicy(
        AdaptiveKConfig.from_dict({"by_load": [3, 2], "log_interval": 3}, 3)
    )
    policy.record_choice(2)
    policy.record_choice(3)
    policy.maybe_log()
    assert policy._k_counts == {2: 1, 3: 1}
    policy.record_choice(3)
    policy.maybe_log()
    assert policy._k_counts == {}



# -------------------------------------------------------------- scheduler wiring


def _make_scheduler(adaptive_k: dict | None, num_speculative_tokens: int = 3,
                    cls=Sm80Scheduler):
    base = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        num_speculative_tokens=num_speculative_tokens,
    )
    speculative_config = base.vllm_config.speculative_config
    assert speculative_config is not None
    if adaptive_k is not None:
        base.vllm_config.additional_config = {"sm80": {"adaptive_k": adaptive_k}}
    return cls(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def test_scheduler_is_untouched_when_adaptive_k_is_off():
    scheduler = _make_scheduler(None)
    assert scheduler.adaptive_k is None
    for request in create_requests(num_requests=2):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == scheduler.num_spec_tokens == 3


def test_scheduler_truncates_proposed_drafts_to_the_step_k():
    scheduler = _make_scheduler({"by_load": [3, 1]})
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    for request in requests:
        request.spec_token_ids = [11, 12, 13]
    scheduler.schedule()
    # Verification must not be wider than the step's k, or the decode batch
    # leaves its captured uniform CUDA graph.
    assert scheduler.cur_num_spec_tokens == 1
    for request in requests:
        assert len(request.spec_token_ids) <= 1


def test_scheduler_never_verifies_more_drafts_than_the_batch_carries():
    scheduler = _make_scheduler({"by_load": [3, 3, 1]})
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    # Load says k=3, but one request only has 1 draft in hand. Verification
    # must drop to 1 for everyone, keeping the decode batch uniform...
    requests[0].spec_token_ids = [11, 12, 13]
    requests[1].spec_token_ids = [21]

    output = scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 1
    assert len(requests[0].spec_token_ids) <= 1
    # ...while the drafter is still asked for its full block, so the width
    # climbs back on the next step.
    assert output.num_spec_tokens_to_schedule == 3


def test_adaptive_k_does_not_corrupt_the_async_placeholder_list():
    # AsyncScheduler hands every request the *same* placeholder list object,
    # so trimming a request's drafts must rebind, never mutate in place.
    scheduler = _make_scheduler({"by_load": [3, 1]}, cls=Sm80AsyncScheduler)
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    shared = [-1, -1, -1]
    for request in requests:
        request.spec_token_ids = shared

    scheduler.schedule()
    assert shared == [-1, -1, -1], "the shared placeholder list was mutated"


def test_scheduler_forgets_finished_requests():
    scheduler = _make_scheduler(ACCEPT)
    assert scheduler.adaptive_k is not None
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.adaptive_k.observe(request.request_id, 3, 1)
    assert request.request_id in scheduler.adaptive_k._acc_s

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler.adaptive_k._acc_s


@pytest.mark.parametrize("log_interval,counted", [(0, False), (5, True)])
def test_scheduler_keeps_the_histogram_only_when_logging(log_interval, counted):
    scheduler = _make_scheduler({"by_load": [3, 1], "log_interval": log_interval})
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    scheduler.schedule()
    request.spec_token_ids = [11, 12, 13]
    scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 3
    assert (scheduler.adaptive_k._k_counts == {3: 1}) is counted
    assert scheduler.adaptive_k._steps_since_log == int(counted)


def test_served_speculative_config_key_reaches_the_fork_settings():
    """`--speculative-config {"adaptive_k": ...}` keeps working as served: the
    key leaves the speculative config (whose schema has no such field) for
    additional_config["sm80"], where config_of reads it."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm_sm80.spec_decode.adaptive_k import config_of

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--speculative-config",
            '{"method": "ngram", "num_speculative_tokens": 3,'
            ' "adaptive_k": {"by_load": [3, 1]}}',
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert "adaptive_k" not in engine_args.speculative_config
    assert engine_args.additional_config == {"sm80": {"adaptive_k": {"by_load": [3, 1]}}}
    scheduler = _make_scheduler({"by_load": [3, 1]})
    assert config_of(scheduler.vllm_config).by_load == (3, 1)
    assert config_of(_make_scheduler(None).vllm_config) is None
