# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-aware DFlash depth (``adaptive_k.accept``)."""

import random

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm_sm80.sched.scheduler import Sm80Scheduler
from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig, AdaptiveKPolicy
from vllm.v1.structured_output import StructuredOutputManager

TP_COSTS = [1.0, 1.08, 1.16]
ACC = {"by_load": [5, 4, 3], "accept": {"costs": TP_COSTS, "hysteresis": 0.03}}


def _policy(raw=ACC):
    return AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, 5))


def _feed(policy, req, p, steps, rng, depth_of=lambda: 5):
    """Verification results of a stream whose drafts are each accepted with
    probability p (given the earlier ones were)."""
    for _ in range(steps):
        k = depth_of()
        accepted = 0
        while accepted < k and rng.random() < p:
            accepted += 1
        policy.observe(req, k, accepted)


# ----------------------------------------------------------------- config


@pytest.mark.parametrize("raw,needle", [
    ({"by_load": [5, 4, 3], "accept": {"costs": [1.0, 1.1]}}, "one positive cost"),
    ({"by_load": [5, 4, 3], "accept": {"costs": [1.0, 0.0, 1.2]}}, "one positive cost"),
    ({"accept": {"costs": [1, 1.2]}}, "needs by_load"),
])
def test_config_rejects_bad_accept(raw, needle):
    with pytest.raises(ValueError, match=needle):
        AdaptiveKConfig.from_dict(raw, 5)


# ---------------------------------------------------------------- the policy


def test_expected_tokens():
    assert AdaptiveKPolicy.expected_tokens(0.0, 5) == 1.0
    assert AdaptiveKPolicy.expected_tokens(0.5, 3) == pytest.approx(1.875)
    assert AdaptiveKPolicy.expected_tokens(1.0, 3) == pytest.approx(4.0, rel=1e-2)


def test_low_acceptance_stream_falls_to_depth_3():
    policy, rng = _policy(), random.Random(0)
    _feed(policy, "prose", 0.45, 60, rng)
    assert policy.accept_rate("prose") < 0.55
    assert policy.select_by_acceptance(["prose"], cap=5) == 3


def test_high_acceptance_stream_reaches_depth_5():
    policy, rng = _policy(), random.Random(1)
    _feed(policy, "code", 0.9, 60, rng, depth_of=lambda: 3)  # even seen at depth 3
    assert policy.accept_rate("code") > 0.8
    assert policy.select_by_acceptance(["code"], cap=5) == 5


def test_the_load_width_caps_the_choice():
    policy, rng = _policy(), random.Random(2)
    _feed(policy, "code", 0.95, 60, rng)
    assert policy.select_by_acceptance(["code"], cap=4) == 4
    assert policy.select_by_acceptance(["code"], cap=3) == 3


def test_a_new_request_starts_at_the_fixed_prior():
    policy = _policy()
    assert policy.accept_rate("new") == pytest.approx(0.75)
    # At the 0.75 prior the deepest draft already pays at TP costs.
    assert policy.select_by_acceptance(["new"], cap=5) == 5
    rng = random.Random(3)
    for i in range(20):
        _feed(policy, f"old{i}", 0.4, 30, rng)
    # Other requests' acceptance never moves the prior a new request starts from.
    assert policy.accept_prior() == 0.75
    assert policy.accept_rate("fresh") == pytest.approx(0.75)
    assert policy.observed_acceptance() < 0.5  # logged only


def test_hysteresis_holds_a_near_tie():
    policy = _policy()
    # p near the depth-4 / depth-5 break-even at TP costs.
    policy._acc_s["r"], policy._acc_f["r"] = 72.0, 28.0
    first = policy.select_by_acceptance(["r"], cap=5)
    policy._acc_s["r"], policy._acc_f["r"] = 73.0, 27.0
    assert policy.select_by_acceptance(["r"], cap=5) == first
    # A clear change still moves it.
    policy._acc_s["r"], policy._acc_f["r"] = 20.0, 80.0
    assert policy.select_by_acceptance(["r"], cap=5) == 3


def test_batch_choice_weighs_every_request():
    policy, rng = _policy(), random.Random(4)
    _feed(policy, "a", 0.95, 60, rng)
    _feed(policy, "b", 0.3, 60, rng)
    k = policy.select_by_acceptance(["a", "b"], cap=4)
    scores = {d: sum(policy.expected_tokens(policy.accept_rate(r), d) for r in "ab") / c
              for d, c in ((3, 1.0), (4, 1.08))}
    assert k == max(scores, key=scores.get)


def test_padding_and_forget():
    policy = _policy()
    policy.mark_padded("r")
    policy.observe("r", 5, 0)  # placeholder drafts: no evidence
    assert "r" not in policy._acc_s
    policy.observe("r", 5, 5)
    assert policy._acc_s["r"] == 5 and policy._acc_f["r"] == 0
    policy.forget("r")
    assert "r" not in policy._acc_s and "r" not in policy._acc_f


# -------------------------------------------------------------- the scheduler


def _make_scheduler(raw):
    base = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192,
                            num_speculative_tokens=5)
    spec = base.vllm_config.speculative_config
    spec.adaptive_k = raw
    spec.adaptive_k_config = AdaptiveKConfig.from_dict(raw, 5)
    return Sm80Scheduler(vllm_config=base.vllm_config, kv_cache_config=base.kv_cache_config,
                     block_size=base.block_size, log_stats=True,
                     structured_output_manager=StructuredOutputManager(base.vllm_config))


def _decode_step(scheduler, requests, width=5):
    for r in requests:
        r.spec_token_ids = [11, 12, 13, 14, 15][:width]
    return scheduler.schedule()


def test_scheduler_one_prose_request_verifies_3():
    scheduler = _make_scheduler(ACC)
    [req] = create_requests(num_requests=1)
    scheduler.add_request(req)
    scheduler.schedule()
    _feed(scheduler.adaptive_k, req.request_id, 0.4, 60, random.Random(5))
    output = _decode_step(scheduler, [req])
    assert scheduler.cur_num_spec_tokens == 3
    assert output.num_spec_tokens_to_schedule == 5  # full block without the width flag


def test_scheduler_one_code_request_verifies_5():
    scheduler = _make_scheduler(ACC)
    [req] = create_requests(num_requests=1)
    scheduler.add_request(req)
    scheduler.schedule()
    _feed(scheduler.adaptive_k, req.request_id, 0.92, 60, random.Random(6))
    _decode_step(scheduler, [req])
    assert scheduler.cur_num_spec_tokens == 5


def test_scheduler_under_load_stays_at_3():
    scheduler = _make_scheduler(ACC)
    reqs = create_requests(num_requests=8)
    for r in reqs:
        scheduler.add_request(r)
    scheduler.schedule()
    for r in reqs:
        _feed(scheduler.adaptive_k, r.request_id, 0.95, 60, random.Random(7))
    _decode_step(scheduler, reqs)
    assert scheduler.cur_num_spec_tokens == 3


def test_scheduler_with_the_width_flag_drafts_the_chosen_depth():
    raw = dict(ACC, draft_by_load=True)
    scheduler = _make_scheduler(raw)
    [req] = create_requests(num_requests=1)
    scheduler.add_request(req)
    scheduler.schedule()
    _feed(scheduler.adaptive_k, req.request_id, 0.4, 60, random.Random(8))
    output = _decode_step(scheduler, [req])
    assert scheduler.cur_num_spec_tokens == 3
    assert output.num_spec_tokens_to_schedule == 3


def test_banners_go_through_the_real_logger(caplog_vllm):
    import vllm.logger as vl

    vl._print_info_once.cache_clear()
    _make_scheduler(ACC)
    messages = [r.getMessage() for r in caplog_vllm.records]
    assert any("GLM-5 acceptance-aware DFlash depth active" in m
               and "(1.0, 1.08, 1.16)" in m for m in messages), messages
    assert any("GLM-5 load-adaptive DFlash depth active" in m for m in messages)


# ------------------------------------------------------------- K_max 7


K7 = {"by_load": [7, 5, 3],
      "accept": {"costs": [1.0, 1.08, 1.16, 1.24, 1.32], "hysteresis": 0.03}}


def test_k7_allows_every_depth_from_3_to_7():
    config = AdaptiveKConfig.from_dict(K7, 7)
    assert config.allowed == (3, 4, 5, 6, 7)
    # Without the acceptance-aware choice only the load widths are captured.
    assert AdaptiveKConfig.from_dict({"by_load": [7, 5, 3]}, 7).allowed == (3, 5, 7)


def test_k7_graph_ceiling_per_depth():
    config = AdaptiveKConfig.from_dict(K7, 7)
    assert [config.max_reqs_for(k, 8) for k in (3, 4, 5, 6, 7)] == [8, 2, 2, 1, 1]
    assert config.max_reqs_for(2, 8) == 0 and config.max_reqs_for(8, 8) == 0


def test_k7_choice_never_leaves_the_captured_graphs():
    config = AdaptiveKConfig.from_dict(K7, 7)
    policy = AdaptiveKPolicy(config)
    rng = random.Random(9)
    for p in (0.2, 0.5, 0.7, 0.9, 0.99):
        for n in range(1, 9):
            reqs = [f"{p}-{i}" for i in range(n)]
            for r in reqs:
                _feed(policy, r, p, 40, rng)
            cap = config.k_for_load(n)
            k = policy.select_by_acceptance(reqs, cap)
            assert k in config.allowed and k <= cap
            assert config.max_reqs_for(k, 8) >= n, (p, n, k)


def test_k7_high_acceptance_reaches_7_and_prose_stays_at_most_5():
    policy, rng = AdaptiveKPolicy(AdaptiveKConfig.from_dict(K7, 7)), random.Random(10)
    _feed(policy, "code", 0.96, 80, rng, depth_of=lambda: 7)
    assert policy.select_by_acceptance(["code"], cap=7) == 7
    policy2 = AdaptiveKPolicy(AdaptiveKConfig.from_dict(K7, 7))
    _feed(policy2, "prose", 0.7, 80, random.Random(11), depth_of=lambda: 5)
    assert policy2.select_by_acceptance(["prose"], cap=7) <= 5
    policy3 = AdaptiveKPolicy(AdaptiveKConfig.from_dict(K7, 7))
    _feed(policy3, "chat", 0.45, 80, random.Random(12), depth_of=lambda: 3)
    assert policy3.select_by_acceptance(["chat"], cap=7) == 3


def test_multi_request_costs_are_used_for_batches():
    raw = {"by_load": [7, 5, 3], "accept": {
        "costs": [1.0, 1.08, 1.16, 1.24, 1.32],
        "costs_multi": [1.0, 1.5, 2.0, 2.5, 3.0], "hysteresis": 0.0}}
    policy = AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, 7))
    for r in ("a", "b"):
        policy._acc_s[r], policy._acc_f[r] = 95.0, 5.0
    assert policy.select_by_acceptance(["a"], cap=5) == 5
    assert policy.select_by_acceptance(["a", "b"], cap=5) == 3  # steep batch costs


# The served mappings: TP4 and PP4 with their measured step costs.
SERVED = {
    "tp4": {"by_load": [7, 5, 3], "draft_by_load": True, "accept": {
        "costs": [1.0, 1.1009, 1.1631, 1.2214, 1.2718],
        "costs_multi": [1.0, 1.1132, 1.1958, 1.2214, 1.2718], "hysteresis": 0.03},
        "log_interval": 0},
    "pp4": {"by_load": [7, 5, 3], "accept": {
        "costs": [1.0, 1.0894, 1.1375, 1.2001, 1.2239],
        "costs_multi": [1.0, 1.0889, 1.1194, 1.2001, 1.2239], "hysteresis": 0.03},
        "log_interval": 0},
}


@pytest.mark.parametrize("layout", sorted(SERVED))
def test_served_mapping(layout):
    raw = SERVED[layout]
    config = AdaptiveKConfig.from_dict(raw, 7)
    assert config.by_load == (7, 5, 3) and config.allowed == (3, 4, 5, 6, 7)
    assert config.draft_by_load is (layout == "tp4")
    assert config.accept and config.accept_costs == tuple(raw["accept"]["costs"])
    assert config.accept_costs_multi == tuple(raw["accept"]["costs_multi"])
    assert (config.accept_hysteresis, config.accept_prior) == (0.03, 0.75)
    assert config.log_interval == 0
    with pytest.raises(ValueError, match="must be <= num_speculative_tokens|outside"):
        AdaptiveKConfig.from_dict(raw, 3)


@pytest.mark.parametrize("depth", [3, 4, 5, 6, 7])
def test_force_file_pins_the_depth(tmp_path, monkeypatch, depth):
    force = tmp_path / "depth"
    force.write_text(str(depth))
    monkeypatch.setenv("VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE", str(force))
    base = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192,
                            num_speculative_tokens=7)
    spec = base.vllm_config.speculative_config
    spec.adaptive_k = K7
    spec.adaptive_k_config = AdaptiveKConfig.from_dict(K7, 7)
    scheduler = Sm80Scheduler(vllm_config=base.vllm_config, kv_cache_config=base.kv_cache_config,
                          block_size=base.block_size, log_stats=True,
                          structured_output_manager=StructuredOutputManager(base.vllm_config))
    [req] = create_requests(num_requests=1)
    scheduler.add_request(req)
    scheduler.schedule()
    req.spec_token_ids = list(range(11, 18))
    scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == depth
    force.write_text("9")  # not a captured depth: ignored
    scheduler._force_countdown = 0
    req.spec_token_ids = list(range(11, 18))
    scheduler.schedule()
    assert scheduler.cur_num_spec_tokens != 9


# ------------------------------------------------- level-1 determinism


def _run_alone(policy, req, pattern, cap=7):
    """One request alone: each step's accepted count is a fixed function of the
    step and the depth (an identical request repeats the same outcomes)."""
    depths = []
    for t, limit in enumerate(pattern):
        k = policy.select_by_acceptance([req], cap)
        depths.append(k)
        policy.observe(req, k, min(k, limit))
    policy.forget(req)
    return depths


@pytest.mark.parametrize("raw", [K7, ACC])
def test_identical_requests_choose_identical_depths_after_unrelated_traffic(raw):
    cap = 7 if raw is K7 else 5
    policy = AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, cap))
    rng = random.Random(13)
    pattern = [rng.choice([0, 1, 2, 3, 5, 7, 7, 7]) for _ in range(80)]
    first = _run_alone(policy, "a", pattern, cap)
    # Unrelated traffic: low- and high-acceptance requests, alone and batched,
    # leaving whatever hysteresis or statistics state they would.
    for i in range(30):
        _feed(policy, f"x{i}", rng.choice([0.2, 0.95]), 20, rng)
        policy.select_by_acceptance([f"x{i}"], cap)
        policy.select_by_acceptance([f"x{i}", f"y{i}"], min(cap, 5))
    second = _run_alone(policy, "b", pattern, cap)
    assert first == second
    if raw is K7:
        assert len(set(first)) > 1  # the pattern exercises depth changes
    # The same on a fresh policy: no hidden dependence on the policy's age.
    third = _run_alone(AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, cap)), "c", pattern, cap)
    assert first == third


def test_scheduler_depth_sequence_is_repeatable_alone():
    """Through the scheduler's own selection: a lone request's depths do not
    depend on requests that ran before it."""
    base = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192,
                            num_speculative_tokens=7)
    spec = base.vllm_config.speculative_config
    spec.adaptive_k, spec.adaptive_k_config = K7, AdaptiveKConfig.from_dict(K7, 7)
    scheduler = Sm80Scheduler(vllm_config=base.vllm_config, kv_cache_config=base.kv_cache_config,
                          block_size=base.block_size, log_stats=True,
                          structured_output_manager=StructuredOutputManager(base.vllm_config))
    rng = random.Random(14)
    pattern = [rng.choice([0, 2, 7, 7]) for _ in range(40)]

    def run(req):
        scheduler.running = [req]
        req.status = req.status
        depths = []
        for limit in pattern:
            req.spec_token_ids = list(range(11, 18))
            k, _ = scheduler._select_adaptive_k()
            depths.append(k)
            scheduler.adaptive_k.observe(req.request_id, k, min(k, limit))
        scheduler.adaptive_k.forget(req.request_id)
        scheduler.running = []
        return depths

    a, b = create_requests(num_requests=2)
    first = run(a)
    for i in range(10):
        _feed(scheduler.adaptive_k, f"z{i}", 0.3, 30, rng)
        scheduler.adaptive_k.select_by_acceptance([f"z{i}"], 7)
    assert run(b) == first
