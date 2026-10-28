# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the DSpark host width (dspark_host_width.py).

- The rule: on recorded p_i (``dspark_host_width_cases.json``: per case the
  p_i of b requests' earlier rounds, from a confidence dump through a fitted
  policy, and the K the policy writer's reference rule chose with that
  policy's block) ``choose_width`` picks the same K; hand-computed edges
  (all accepted: the widest K; none: the narrowest; exact ties: the smaller).
- The block's checks refuse malformed blocks; a policy without one has no
  host width, and the kernel returns no p_i.
- The scheduler wrappers: a step reads each request's round ``lag_rounds``
  back (two: the round before the step in flight), falls back without it
  (first rounds, placeholder drafts, a step whose output carried no p_i, a
  forgotten request), leaves other batch sizes and prefill steps alone, never
  verifies more than the drafts on hand (the best K within the cap, not the
  capped best), leaves a forced calibration depth alone, and refuses K
  choices without CUDA graphs and pipeline parallelism.
- The kernel's torch reference returns p_i past K without changing the
  decision; ``begin_step`` exposes them only for a uniform step.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace as S

import pytest
import torch

from vllm_sm80.spec_decode import dspark_host_width as hw_module
from vllm_sm80.spec_decode import dspark_row_mask
from vllm_sm80.spec_decode import dspark_row_mask_kernel as rmk
from vllm_sm80.spec_decode.dspark_host_width import (
    P_HAT_ATTR,
    choose_width,
    parse_host_width,
)
from vllm_sm80.spec_decode.dspark_row_mask import FEATURES, RowMaskState

pytestmark = pytest.mark.cpu_test

CASES = json.loads((Path(__file__).parent / "dspark_host_width_cases.json").read_text())


def block() -> dict:
    """The recorded policy's tau and host_width block."""
    return copy.deepcopy({"tau": CASES["tau"], "host_width": CASES["host_width"]})


@pytest.fixture(scope="module")
def hw():
    return parse_host_width("cases", block())


def test_recorded_cases_match_the_reference_rule(hw):
    cases = CASES["cases"]
    assert {c["b"] for c in cases} == set(range(3, 9))
    assert {c["k"] for c in cases} >= {1, 2, 3, 5}, sorted({c["k"] for c in cases})
    for c in cases:
        assert len(c["p_hat"]) == c["b"]
        assert choose_width(hw, c["b"], c["p_hat"]) == c["k"], c


def test_hand_computed_edges(hw):
    for b in range(3, 9):
        assert choose_width(hw, b, [[1.0] * 5] * b) == 5
        assert choose_width(hw, b, [[1e-3] * 5] * b) == 1
    raw = block()
    raw["host_width"]["s"]["4"] = {str(k): 0.0 for k in range(1, 6)}
    raw["host_width"]["cum_ms"]["4"] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    raw["host_width"]["t0_ms"]["4"] = 10.0
    free = parse_host_width("cases", raw)
    # Dead rows free and no request keeps a draft: every K costs t0 exactly.
    assert choose_width(free, 4, [[1e-3] * 5] * 4) == 1


def test_best_k_within_the_cap():
    # K5 scores best overall; under a cap of 3, K1 beats K3 (not min(5, 3)).
    raw = block()
    raw["host_width"]["t0_ms"]["4"] = 10.0
    raw["host_width"]["cum_ms"]["4"] = [0.0, 0.1, 3.0, 3.0, 3.0, 3.0]
    raw["host_width"]["s"]["4"] = {str(k): 0.0 for k in range(1, 6)}
    hw = parse_host_width("cases", raw)
    q = [[0.9] * 5] * 4
    assert choose_width(hw, 4, q) == 5
    assert choose_width(hw, 4, q, cap=3) == 1
    raw["host_width"]["k_choices"] = [2, 5]
    raw["host_width"]["fallback_k"] = 5
    assert choose_width(parse_host_width("cases", raw), 4, q, cap=1) == 1


def test_one_formula(hw):
    # T(b, K | d) and E as the block states them, on one step.
    b, q = 4, [[0.9, 0.8, 0.6, 0.5, 0.4], [0.3] * 5, [0.99] * 5, [0.7, 0.2, 0.9, 0.9, 0.9]]
    tau, cum = hw.tau[b], hw.cum[b]

    def value(k):
        ds, e = [], 0.0
        for row in q:
            prod, d = 1.0, 0
            while d < k and prod * row[d] >= tau:
                prod *= row[d]
                d += 1
            ds.append(d)
            e += 1.0 + sum(math.prod(row[: i + 1]) for i in range(d))
        t = hw.t0[b] + b * cum[k] - (1 - hw.share[b][k]) * sum(cum[k] - cum[d] for d in ds)
        return e / t

    want = max(hw.k_choices, key=lambda k: (value(k), -k))
    assert choose_width(hw, b, q) == want


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda r: r["host_width"].update(estimate="accepted"), "estimate"),
        (lambda r: r["host_width"].update(lag_rounds=0), "lag_rounds"),
        (lambda r: r["host_width"].update(fallback_k=7), "fallback_k"),
        (lambda r: r["host_width"].update(batch_sizes=[1, 3]), "start at 2"),
        (lambda r: r["host_width"].update(k_choices=[1, 1, 5]), "k_choices"),
        (lambda r: r["tau"].pop("5"), "batch size 5 has no tau"),
        (lambda r: r["tau"].update({"6": 0.0}), r"tau\[6\]"),
        (lambda r: r["host_width"]["cum_ms"].update({"3": [0.0, 1.0]}), r"cum_ms\[3\]"),
        (lambda r: r["host_width"]["cum_ms"].update({"3": [0.0, 2.0, 1.0, 3.0, 4.0, 5.0]}),
         "not decrease"),
        (lambda r: r["host_width"]["s"]["8"].pop("2"), "K = 2"),
        (lambda r: r["host_width"]["s"]["8"].update({"2": 1.5}), r"s\[8\]\[2\]"),
        (lambda r: r["host_width"]["t0_ms"].update({"7": -1.0}), r"t0_ms\[7\]"),
    ],
)
def test_malformed_blocks_refuse(edit, match):
    raw = block()
    edit(raw)
    with pytest.raises(ValueError, match=match):
        parse_host_width("cases", raw)


def _policy(tmp_path, with_block=True) -> Path:
    """A policy file with the recorded block (or none)."""
    raw = {
        "schema": "dspark-depth-policy/1",
        "max_depth": 7,
        "features": ["bias", "logit_conf"],
        "weights": [[0.0, 1.0]] * 7,
        "costs": {"1": {str(k): 10.0 + k for k in range(8)}},
        "tau": CASES["tau"],
    }
    if with_block:
        raw["host_width"] = CASES["host_width"]
    path = tmp_path / ("policy.json" if with_block else "plain.json")
    path.write_text(json.dumps(raw))
    return path


def test_policy_load(tmp_path, monkeypatch):
    path = _policy(tmp_path)
    policy = dspark_row_mask.load_policy(path, 8, 7, [0] + [7] * 8)
    assert policy.p_hat_width == 5
    with pytest.raises(ValueError, match="past the drafter's 4 drafts"):
        dspark_row_mask.load_policy(path, 8, 4, [0] + [4] * 8)
    plain = _policy(tmp_path, with_block=False)
    assert dspark_row_mask.load_policy(plain, 8, 7, [0] + [7] * 8).p_hat_width == 0
    monkeypatch.setattr(hw_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", str(plain))
    hw_module.configured.cache_clear()
    assert hw_module.configured() is None
    monkeypatch.setattr(hw_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", str(path))
    hw_module.configured.cache_clear()
    assert hw_module.configured().k_choices == (1, 2, 3, 4, 5)
    monkeypatch.setattr(hw_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", "")
    hw_module.configured.cache_clear()
    assert hw_module.configured() is None


# ------------------------------------------------------------------ scheduler

BY_LOAD = [7, 5, 5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5]
CAPTURE = sorted({c * (k + 1) for c in range(1, 9) for k in range(1, 6)} | {8})
LOW, HIGH = [0.05] * 5, [0.99] * 5


def _req(rid, prefill=False, drafts=5):
    return S(request_id=rid, is_prefill_chunk=prefill, spec_token_ids=[-1] * drafts)


class Engine:
    """The wrapped scheduler on stand-ins: ``step(running)`` runs one
    schedule (its width choice, then its output) and returns the width and
    the step's output; ``done(output, p_hat)`` processes a step's output."""

    def __init__(self, monkeypatch, tmp_path, by_load=BY_LOAD, capture=CAPTURE, pp=1):
        from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig, AdaptiveKPolicy
        from vllm.v1.core.sched.scheduler import Scheduler

        self.Scheduler = Scheduler
        for cls, name in (
            (Scheduler, "_select_adaptive_k"),
            (Scheduler, "schedule"),
            (Scheduler, "update_from_output"),
            (AdaptiveKPolicy, "mark_padded"),
            (AdaptiveKPolicy, "forget"),
        ):
            monkeypatch.setattr(cls, name, getattr(cls, name))

        def schedule(sched):
            # Stand-in for the real loop: the drafts each running request
            # gets this step, placeholder drafts for the "padded" ones.
            out = S(scheduled_spec_decode_tokens={})
            for r in sched.running:
                if not r.is_prefill_chunk:
                    out.scheduled_spec_decode_tokens[r.request_id] = list(r.spec_token_ids)
                    if r.request_id in sched.pad:
                        sched.adaptive_k.mark_padded(r.request_id)
            return out

        monkeypatch.setattr(Scheduler, "schedule", schedule)
        monkeypatch.setattr(Scheduler, "update_from_output", lambda sched, so, mro: "updated")
        monkeypatch.setattr(
            hw_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", str(_policy(tmp_path))
        )
        hw_module.configured.cache_clear()
        hw_module.install()
        hw_module.install()
        assert Scheduler._select_adaptive_k._dspark_host_width
        monkeypatch.setattr(hw_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", "")
        hw_module.configured.cache_clear()
        config = AdaptiveKConfig.from_dict({"by_load": by_load, "draft_by_load": True}, 7)
        self.policy = AdaptiveKPolicy(config)
        self.sched = S(
            running=[],
            waiting=[],
            kv_holding_waiting=[],
            num_spec_tokens=7,
            adaptive_k=self.policy,
            _forced_depth=lambda: None,
            max_num_running_reqs=8,
            vllm_config=S(
                compilation_config=S(cudagraph_capture_sizes=capture),
                parallel_config=S(pipeline_parallel_size=pp),
            ),
            pad=set(),
        )

    def step(self, running, pad=()):
        self.sched.running = running
        self.sched.pad = set(pad)
        choice = self.Scheduler._select_adaptive_k(self.sched)
        output = self.Scheduler.schedule(self.sched)
        return choice, output

    def done(self, output, p_hat=None, order=None):
        order = order or list(output.scheduled_spec_decode_tokens)
        mro = S(req_ids=order, req_id_to_index={r: i for i, r in enumerate(order)})
        if p_hat is not None:
            setattr(mro, P_HAT_ATTR, [p_hat[r] for r in order])
        assert self.Scheduler.update_from_output(self.sched, output, mro) == "updated"


def _running(n, **kw):
    return [_req(chr(ord("a") + i), **kw) for i in range(n)]


def test_lag_two_under_async_scheduling(monkeypatch, tmp_path):
    e = Engine(monkeypatch, tmp_path)
    reqs = "abcd"
    (k1, _), out1 = e.step(_running(4))
    (k2, _), out2 = e.step(_running(4))
    assert (k1, k2) == (5, 5)  # no round two back yet: the fallback
    e.done(out1, {r: LOW for r in reqs})
    run = _running(4)
    (k3, d3), out3 = e.step(run)
    assert (k3, d3) == (1, 5)  # round 1's low p_i: verify 1, draft the block
    assert [len(r.spec_token_ids) for r in run] == [1] * 4
    e.done(out2, {r: HIGH for r in reqs})
    (k4, _), out4 = e.step(_running(4))
    assert k4 == 5  # round 2's high p_i
    e.done(out3)  # a step whose output carried no p_i (no decision kernel)
    (k5, _), out5 = e.step(_running(4))
    assert k5 == 5  # fallback: round 3 has no p_i
    e.done(out4, {r: LOW for r in reqs})
    (k6, _), _ = e.step(_running(4))
    assert k6 == 1


def test_order_of_rows_is_the_outputs(monkeypatch, tmp_path):
    e = Engine(monkeypatch, tmp_path)
    _, out1 = e.step(_running(3))
    _, out2 = e.step(_running(3))
    # Rows in the runner's batch order, not the schedule's.
    e.done(out1, {"a": HIGH, "b": HIGH, "c": LOW}, order=["c", "a", "b"])
    hw = hw_module.parse_host_width("cases", block())
    (k, _), _ = e.step(_running(3))
    assert k == choose_width(hw, 3, [HIGH, HIGH, LOW])


def test_fallbacks(monkeypatch, tmp_path):
    e = Engine(monkeypatch, tmp_path)
    for _ in range(2):
        _, out = e.step(_running(3), pad={"b"})
        e.done(out, {r: LOW for r in "abc"})
    # Both of b's rounds had placeholder drafts: no estimate for b.
    (k, _), out = e.step(_running(3))
    assert k == 5
    e.done(out, {r: LOW for r in "abc"})
    _, out = e.step(_running(3))
    e.done(out, {r: LOW for r in "abc"})
    (k, _), _ = e.step(_running(3))
    assert k == 1
    e.policy.forget("c")  # a finished request's rounds go
    (k, _), _ = e.step(_running(3))
    assert k == 5


def test_other_steps_keep_their_width(monkeypatch, tmp_path):
    e = Engine(monkeypatch, tmp_path)
    for _ in range(2):
        _, out = e.step(_running(2))
        e.done(out, {r: LOW for r in "ab"})
    assert e.step(_running(2))[0] == (5, 5)  # b = 2 is not in the block
    for _ in range(2):
        _, out = e.step(_running(3))
        e.done(out, {r: LOW for r in "abc"})
    run = _running(3) + [_req("p", prefill=True)]
    assert e.step(run)[0] == (5, 5)  # a prefill row
    run = [_req("a", drafts=2), _req("b"), _req("c")]
    (k, _), _ = e.step(run)
    assert k == 1
    for r in "abc":
        e.policy.forget(r)
    for _ in range(2):
        _, out = e.step(_running(3))
        e.done(out, {r: HIGH for r in "abc"})
    run = [_req("a", drafts=3), _req("b"), _req("c")]
    (k, _), _ = e.step(run)
    assert k == 3  # never past the drafts on hand
    assert [len(r.spec_token_ids) for r in run] == [3, 3, 3]


def test_forced_depth_stands(monkeypatch, tmp_path):
    e = Engine(monkeypatch, tmp_path)
    for _ in range(2):
        _, out = e.step(_running(3))
        e.done(out, {r: LOW for r in "abc"})
    e.sched._forced_depth = lambda: 5
    assert e.step(_running(3))[0][0] == 5


def test_pipeline_parallel_refuses(monkeypatch, tmp_path):
    e = Engine(monkeypatch, tmp_path, pp=2)
    with pytest.raises(ValueError, match="pipeline"):
        e.step(_running(3))


@pytest.mark.parametrize(
    "by_load, capture, match",
    [
        ([7, 5, 5, 5, 5, 5, 5, 5], CAPTURE, r"K \[1, 2, 3, 4\]"),
        (BY_LOAD, [c for c in CAPTURE if c != 15], r"\[15\] tokens"),
    ],
)
def test_widths_without_graphs_refuse(monkeypatch, tmp_path, by_load, capture, match):
    e = Engine(monkeypatch, tmp_path, by_load=by_load, capture=capture)
    with pytest.raises(ValueError, match=match):
        e.step(_running(3))


def test_off_installs_nothing(monkeypatch, tmp_path):
    from vllm.v1.core.sched.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "_select_adaptive_k", Scheduler._select_adaptive_k)
    monkeypatch.setattr(
        hw_module.envs,
        "VLLM_DSPARK_ADAPTIVE_DEPTH",
        str(_policy(tmp_path, with_block=False)),
    )
    hw_module.configured.cache_clear()
    hw_module.install()
    assert not getattr(Scheduler._select_adaptive_k, "_dspark_host_width", False)
    hw_module.configured.cache_clear()


# --------------------------------------------------------------------- worker


def _state(width):
    state = RowMaskState(8, 96, 7, torch.device("cpu"))
    state.weights[:, FEATURES.index("bias")] = torch.linspace(0.5, -0.5, 7)
    state.weights[:, FEATURES.index("logit_conf")] = 1.0
    state.cost[:] = 20.0 + torch.arange(8.0)
    state.tau[2:] = 0.3
    state.p_hat_width = width
    gen = torch.Generator().manual_seed(3)
    state.conf_by_slot.copy_(torch.rand(8, 7, generator=gen))
    return state


@pytest.mark.parametrize("k", (1, 3, 5, 7))
@pytest.mark.parametrize("b", (1, 3, 8))
def test_reference_p_hat_past_k(k, b):
    plain, with_p = _state(0), _state(5)
    idx = torch.randperm(8, generator=torch.Generator().manual_seed(b))[:b]
    rmk.reference_row_mask(plain, idx, b, k)
    p_hat = torch.full((b, 5), -1.0)
    rmk.reference_row_mask(with_p, idx, b, k, p_hat)
    assert torch.equal(plain.live_len, with_p.live_len)
    assert torch.equal(plain.row_live, with_p.row_live)
    w = with_p.weights
    for r, slot in enumerate(idx.tolist()):
        for i in range(5):
            c = float(with_p.conf_by_slot[slot, i])
            c = min(max(c, 1e-6), 1 - 1e-6)
            z = float(w[i, 0]) + float(w[i, 1]) * (math.log(c) - math.log(1 - c))
            assert float(p_hat[r, i]) == pytest.approx(1 / (1 + math.exp(-z)), rel=1e-5)


def _batch(num_reqs, k, has_prefill=False):
    import numpy as np

    return S(
        dummy_run=False,
        has_prefill=has_prefill,
        max_query_len=None,
        num_draft_tokens_per_req=np.full(num_reqs, k, dtype=np.int32),
        num_reqs=num_reqs,
        num_tokens=num_reqs * (k + 1),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
    )


def test_begin_step_exposes_p_hat_once():
    state = _state(5)
    rmk.begin_step(state, _batch(4, 3), dummy_run=False)
    p_hat = state.take_p_hat()
    assert p_hat is not None and p_hat.shape == (4, 5)
    assert state.take_p_hat() is None
    rmk.begin_step(state, _batch(4, 3, has_prefill=True), dummy_run=False)
    assert state.take_p_hat() is None
    rmk.begin_step(state, _batch(4, 3), dummy_run=True)
    assert state.take_p_hat() is None
    off = _state(0)
    rmk.begin_step(off, _batch(4, 3), dummy_run=False)
    assert off.take_p_hat() is None
