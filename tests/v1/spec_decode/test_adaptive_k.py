# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the acceptance-adaptive per-step draft count."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.adaptive_k import AdaptiveKConfig, AdaptiveKPolicy
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


def _policy(num_speculative_tokens: int = 4, **overrides) -> AdaptiveKPolicy:
    raw = {"min": 1, "max": num_speculative_tokens}
    raw.update(overrides)
    return AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, num_speculative_tokens))


# --------------------------------------------------------------------- config


def test_config_defaults_to_full_contiguous_range():
    config = AdaptiveKConfig.from_dict({}, 4)
    assert config.min_k == 1
    assert config.max_k == 4
    assert config.allowed == (1, 2, 3, 4)
    assert config.ema == 0.8
    # The median, not the batch minimum -- see select_k.
    assert config.quantile == 0.5


def test_config_bounds_narrow_the_allowed_set():
    config = AdaptiveKConfig.from_dict({"min": 2, "max": 3}, 4)
    assert config.allowed == (2, 3)


def test_config_honours_an_explicit_allowed_set():
    # DFlash2's legal block sizes, as used by the MiaAI deployment.
    config = AdaptiveKConfig.from_dict({"min": 2, "max": 7, "allowed": [2, 4, 7]}, 7)
    assert config.allowed == (2, 4, 7)
    assert config.min_k == 2
    assert config.max_k == 7


def test_config_rejects_max_above_num_speculative_tokens():
    with pytest.raises(ValueError, match="must be <= num_speculative_tokens"):
        AdaptiveKConfig.from_dict({"max": 5}, 3)


def test_config_rejects_allowed_outside_the_drafter_range():
    with pytest.raises(ValueError, match="outside"):
        AdaptiveKConfig.from_dict({"allowed": [2, 9]}, 4)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown adaptive_k keys"):
        AdaptiveKConfig.from_dict({"decay": 0.9}, 4)


def test_config_rejects_bad_ranges_and_decays():
    with pytest.raises(ValueError, match="must be <= max"):
        AdaptiveKConfig.from_dict({"min": 3, "max": 2}, 4)
    with pytest.raises(ValueError, match=r"ema .* must be in"):
        AdaptiveKConfig.from_dict({"ema": 1.0}, 4)
    with pytest.raises(ValueError, match=r"quantile .* must be in"):
        AdaptiveKConfig.from_dict({"quantile": 1.5}, 4)
    with pytest.raises(ValueError, match="no value inside"):
        AdaptiveKConfig.from_dict({"min": 3, "max": 4, "allowed": [2]}, 4)


# ------------------------------------------------------------------ EMA update


def test_ema_first_observation_seeds_the_average():
    policy = _policy()
    policy.observe("a", num_draft_tokens=3, num_accepted=2)
    assert policy._ema["a"] == pytest.approx(2.0)


def test_ema_decays_towards_new_samples():
    policy = _policy(ema=0.5)
    policy.observe("a", num_draft_tokens=4, num_accepted=3)
    policy.observe("a", num_draft_tokens=4, num_accepted=0)
    assert policy._ema["a"] == pytest.approx(1.5)
    policy.observe("a", num_draft_tokens=4, num_accepted=0)
    assert policy._ema["a"] == pytest.approx(0.75)


def test_ema_ignores_steps_with_no_drafts():
    policy = _policy()
    policy.observe("a", num_draft_tokens=0, num_accepted=0)
    assert "a" not in policy._ema


def test_ema_clamps_accepted_to_the_drafts_offered():
    policy = _policy(margin=0.0)
    policy.observe("a", num_draft_tokens=2, num_accepted=5)
    assert policy._ema["a"] == pytest.approx(2.0)


def test_a_fully_accepted_step_is_recorded_above_its_ceiling():
    # Right-censored: every draft offered was accepted, so the sample carries
    # no upper bound and must not be averaged in at face value.
    policy = _policy(margin=1.0)
    policy.observe("a", num_draft_tokens=2, num_accepted=2)
    assert policy._ema["a"] == pytest.approx(3.0)
    # A partially accepted step is exact, so it is recorded as observed.
    policy.forget("a")
    policy.observe("a", num_draft_tokens=2, num_accepted=1)
    assert policy._ema["a"] == pytest.approx(1.0)


def test_censoring_correction_breaks_a_low_width_equilibrium():
    # A request whose drafts are always all accepted, fed back through the
    # policy's own choice. Every observation is censored, so without the
    # correction the average only ever confirms the width it is already at and
    # k never leaves it. With it the policy climbs to the configured maximum.
    policy = _policy(7)
    for _ in range(200):
        k = policy.select_k(["a"])
        policy.observe("a", num_draft_tokens=k, num_accepted=k)
    assert policy.select_k(["a"]) == 7


def test_width_settles_just_past_where_acceptance_tops_out():
    # Same feedback loop, but the content only ever supports two accepted
    # drafts. The policy must come to rest around there, not run away.
    policy = _policy(7)
    for _ in range(300):
        k = policy.select_k(["a"])
        policy.observe("a", num_draft_tokens=k, num_accepted=min(2, k))
    assert policy.select_k(["a"]) in (3, 4)


def test_forget_drops_per_request_state():
    policy = _policy()
    policy.observe("a", num_draft_tokens=3, num_accepted=3)
    policy.forget("a")
    assert "a" not in policy._ema


def test_new_requests_are_seeded_from_the_model_wide_mean():
    policy = _policy()
    # Drive the running mean down with a long stretch of poor acceptance.
    for _ in range(200):
        policy.observe("a", num_draft_tokens=4, num_accepted=0)
    # An unseen request inherits that prior rather than starting at max.
    assert policy.select_k(["fresh"]) == 1


# ------------------------------------------------------------------- selection


def test_k_tracks_acceptance_with_the_margin():
    policy = _policy()
    # k = clamp(round(ema_accepted + margin), min, max)
    for accepted, expected_k in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 4)):
        policy.forget("a")
        for _ in range(50):
            policy.observe("a", num_draft_tokens=4, num_accepted=accepted)
        assert policy.select_k(["a"]) == expected_k, accepted


def test_mixed_batch_takes_the_median_by_default():
    policy = _policy()
    for _ in range(50):
        policy.observe("easy", num_draft_tokens=4, num_accepted=4)  # censored
        policy.observe("hard", num_draft_tokens=4, num_accepted=0)
    assert policy.select_k(["easy"]) == 4
    assert policy.select_k(["hard"]) == 1
    # Median of [1, 4] with an even sample lands on the lower of the pair.
    assert policy.select_k(["easy", "hard"]) == 1
    # A batch the easy requests dominate follows them up.
    assert policy.select_k(["easy", "easy", "hard"]) == 4


def test_mixed_batch_quantile_zero_takes_the_minimum():
    policy = _policy(quantile=0.0)
    for _ in range(50):
        policy.observe("easy", num_draft_tokens=4, num_accepted=4)
        policy.observe("hard", num_draft_tokens=4, num_accepted=0)
    assert policy.select_k(["easy", "easy", "hard"]) == 1


def test_mixed_batch_quantile_trades_off_the_stragglers():
    policy = _policy(quantile=1.0)
    for _ in range(50):
        policy.observe("easy", num_draft_tokens=4, num_accepted=4)
        policy.observe("hard", num_draft_tokens=4, num_accepted=0)
    assert policy.select_k(["easy", "hard"]) == 4


def test_empty_batch_falls_back_to_max():
    assert _policy().select_k([]) == 4


# ----------------------------------------------- capture-size / allowed-set fit


def test_snap_rounds_down_to_a_captured_draft_count():
    policy = _policy(7, min=2, allowed=[2, 4, 7])
    assert policy.snap(7) == 7
    assert policy.snap(6) == 4
    assert policy.snap(4) == 4
    assert policy.snap(3) == 2
    assert policy.snap(2) == 2
    # Below the smallest captured count, clamp up rather than run eager.
    assert policy.snap(1) == 2


def test_selection_only_ever_returns_captured_counts():
    policy = _policy(7, min=2, allowed=[2, 4, 7])
    for accepted in range(8):
        policy.forget("a")
        for _ in range(50):
            policy.observe("a", num_draft_tokens=7, num_accepted=accepted)
        assert policy.select_k(["a"]) in policy.config.allowed


def test_logging_histogram_resets_only_after_the_interval():
    policy = _policy(log_interval=3)
    policy.record_choice(2)
    policy.record_choice(3)
    policy.maybe_log()
    assert policy._k_counts == {2: 1, 3: 1}
    policy.record_choice(3)
    policy.maybe_log()
    assert policy._k_counts == {}


def test_logging_is_off_by_default():
    policy = _policy()
    policy.record_choice(2)
    policy.maybe_log()
    assert policy._k_counts == {2: 1}


# -------------------------------------------------------------- scheduler wiring


def _make_scheduler(adaptive_k: dict | None, num_speculative_tokens: int = 3):
    base = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        num_speculative_tokens=num_speculative_tokens,
    )
    speculative_config = base.vllm_config.speculative_config
    assert speculative_config is not None
    if adaptive_k is not None:
        speculative_config.adaptive_k = adaptive_k
        speculative_config.adaptive_k_config = AdaptiveKConfig.from_dict(
            adaptive_k, num_speculative_tokens
        )
    return Scheduler(
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


def test_scheduler_reports_the_adaptive_k_for_the_step():
    scheduler = _make_scheduler({"min": 1, "max": 3})
    assert scheduler.adaptive_k is not None
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    # Hard content: nothing the drafter proposes is accepted.
    for request in requests:
        for _ in range(50):
            scheduler.adaptive_k.observe(request.request_id, 3, 0)
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == 1
    assert scheduler.cur_num_spec_tokens == 1


def test_scheduler_truncates_proposed_drafts_to_the_step_k():
    scheduler = _make_scheduler({"min": 1, "max": 3})
    assert scheduler.adaptive_k is not None
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    for request in requests:
        request.spec_token_ids = [11, 12, 13]
        for _ in range(50):
            scheduler.adaptive_k.observe(request.request_id, 3, 0)

    scheduler.schedule()
    # Verification must not be wider than the step's k, or the decode batch
    # leaves its captured uniform CUDA graph.
    for request in requests:
        assert len(request.spec_token_ids) <= 1


def test_scheduler_never_verifies_more_drafts_than_the_batch_carries():
    scheduler = _make_scheduler({"min": 1, "max": 3})
    assert scheduler.adaptive_k is not None
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    # Acceptance says k=3, but one request only has 1 draft in hand (e.g. it
    # was drafted while k was lower). Verification must drop to 1 for
    # everyone, keeping the decode batch uniform...
    for request in requests:
        for _ in range(50):
            scheduler.adaptive_k.observe(request.request_id, 3, 3)
    requests[0].spec_token_ids = [11, 12, 13]
    requests[1].spec_token_ids = [21]

    output = scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 1
    assert len(requests[0].spec_token_ids) <= 1
    # ...while the drafter is still asked for 3, so k can climb back next step
    # instead of being latched down forever.
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_k_can_climb_back_up():
    scheduler = _make_scheduler({"min": 1, "max": 3})
    assert scheduler.adaptive_k is not None
    requests = create_requests(num_requests=1)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    # Poor acceptance drives verification down to one draft.
    for _ in range(50):
        scheduler.adaptive_k.observe(requests[0].request_id, 3, 0)
    requests[0].spec_token_ids = [11, 12, 13]
    scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 1

    # Then acceptance recovers. The one draft on hand is accepted, which is a
    # censored observation -- acceptance is at least 1 -- so it is recorded as
    # 1 + margin and the ask goes straight back up. This step still verifies
    # the single draft it has, but asks for three.
    for _ in range(50):
        scheduler.adaptive_k.observe(requests[0].request_id, 1, 1)
    requests[0].spec_token_ids = [11]
    output = scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 1
    assert output.num_spec_tokens_to_schedule == 3

    # Once the wider drafts arrive, verification widens too.
    requests[0].spec_token_ids = [11, 12, 13]
    scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 3


def test_ema_climbs_one_notch_per_round_out_of_the_censored_region():
    # A request that has every draft accepted only proves acceptance >= k.
    # The margin is what lets k ratchet up; check it actually converges to the
    # configured maximum instead of latching at the width it happens to be on.
    policy = _policy(7)
    for _ in range(200):
        policy.observe("a", num_draft_tokens=7, num_accepted=7)
    assert policy.select_k(["a"]) == 7


def test_padding_placeholders_do_not_count_against_acceptance():
    # The scheduler pads a re-entering decode request with [-1] * k drafts to
    # keep the batch uniform. Those are rejected by construction and must not
    # look like a run of bad acceptance.
    policy = _policy(4)
    for _ in range(50):
        policy.observe("a", num_draft_tokens=4, num_accepted=3)
    before = policy._ema["a"]
    policy.mark_padded("a")
    policy.observe("a", num_draft_tokens=4, num_accepted=0)
    assert policy._ema["a"] == pytest.approx(before)
    # Only the marked step is skipped; the next real one still counts.
    policy.observe("a", num_draft_tokens=4, num_accepted=0)
    assert policy._ema["a"] < before


def test_padding_marks_are_consumed_in_order_and_forgotten():
    policy = _policy(4)
    policy.mark_padded("a")
    policy.mark_padded("a")
    policy.observe("a", num_draft_tokens=4, num_accepted=0)
    policy.observe("a", num_draft_tokens=4, num_accepted=0)
    assert "a" not in policy._ema
    policy.observe("a", num_draft_tokens=4, num_accepted=2)
    assert policy._ema["a"] == pytest.approx(2.0)

    policy.mark_padded("b")
    policy.forget("b")
    assert "b" not in policy._pending_padded


def test_scheduler_marks_padded_requests():
    # End-to-end through the scheduler: a padded step must leave the EMA alone.
    scheduler = _make_scheduler({"min": 1, "max": 3})
    assert scheduler.adaptive_k is not None
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.adaptive_k.mark_padded(request.request_id)
    scheduler.adaptive_k.observe(request.request_id, 3, 0)
    assert request.request_id not in scheduler.adaptive_k._ema


def test_scheduler_forgets_finished_requests():
    scheduler = _make_scheduler({"min": 1, "max": 3})
    assert scheduler.adaptive_k is not None
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.adaptive_k.observe(request.request_id, 3, 1)
    assert request.request_id in scheduler.adaptive_k._ema

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler.adaptive_k._ema


def test_adaptive_k_does_not_corrupt_the_async_placeholder_list():
    # AsyncScheduler hands every request the *same* placeholder list object,
    # so trimming a request's drafts must rebind, never mutate in place.
    base = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=8192, num_speculative_tokens=3
    )
    speculative_config = base.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.adaptive_k_config = AdaptiveKConfig.from_dict(
        {"min": 1, "max": 3}, 3
    )
    scheduler = AsyncScheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )
    assert scheduler.adaptive_k is not None

    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()

    shared = [-1, -1, -1]
    for request in requests:
        request.spec_token_ids = shared
        for _ in range(50):
            scheduler.adaptive_k.observe(request.request_id, 3, 0)

    scheduler.schedule()
    assert shared == [-1, -1, -1], "the shared placeholder list was mutated"
