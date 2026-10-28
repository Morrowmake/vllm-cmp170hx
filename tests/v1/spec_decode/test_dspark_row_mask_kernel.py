# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark adaptive-depth decision kernel (``dspark_row_mask_kernel``).

- The torch reference against hand-computed ``live_len`` at b = 1 and b = 8
  with the raw-confidence policy (weight 1 on logit_conf): all-one
  confidences keep every draft, a near-zero first confidence keeps none;
  exact ties (probabilities 0, 1/2 and 1 from bias weights) take the larger k.
- ``row_live``: request r's rows r*(K+1) .. r*(K+1)+K are live up to its
  ``live_len``; rows past the step that an earlier, wider step masked are
  live again; rows no step can mask are not written.
- ``begin_target_step``: a uniform speculative decode step computes the mask,
  any other step (prefill rows, mixed or zero drafts, dummy runs, GPU-side
  draft reallocation, a token count that is not num_reqs*(K+1)) leaves every
  row live; nothing happens without the state.
- A depth forced through the policy file alone (bias +20 up to k, -20 after,
  costs rising 1 % per draft) gives d = min(k, K) at every b and K.
- The Triton kernels in the interpreter (a subprocess with
  ``TRITON_INTERPRET=1``) against an independent numpy reference, bitwise:
  random and dump-like features (confidences near one, draft-logit
  statistics), random and fitted-like weights, drafts K = 1, 3, 5, 7 of a
  7-draft drafter at b = 1 .. 8; ``live_len`` covers 0 .. 7; ``live_len``
  past the batch and rows past the maskable span are never written; the
  raw-confidence policy equals the confidence-only rule
  (``E(k) / cost`` on the confidences themselves) wherever that rule's best
  score leads by more than 1e-5. With a ``p_hat`` output (the host width,
  half the cases) the kernel writes the numpy reference's ``p_i`` for
  positions 1 .. 5 whatever K is, nothing past the batch, and the same
  decisions.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_sm80.spec_decode import dspark_row_mask
from vllm_sm80.spec_decode import dspark_row_mask_kernel as rmk
from vllm_sm80.spec_decode.dspark_row_mask import FEATURES, PROB_EPS, RowMaskState

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
MAX_REQS = 8
MAX_DRAFT = 5
MAX_TOKENS = 64
F32 = np.float32


def make_state(max_draft: int = MAX_DRAFT, max_tokens: int = MAX_TOKENS) -> RowMaskState:
    """A CPU state with the raw-confidence policy."""
    state = RowMaskState(MAX_REQS, max_tokens, max_draft, torch.device("cpu"))
    state.weights[:, FEATURES.index("logit_conf")] = 1.0
    return state


def set_conf(state: RowMaskState, slots: list[int], rows: list[list[float]]) -> torch.Tensor:
    for slot, row in zip(slots, rows):
        state.conf_by_slot[slot, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return torch.tensor(slots, dtype=torch.int64)


def exact_probs(state: RowMaskState, probs: list[float]) -> None:
    """Bias-only weights giving position i exactly ``probs[i]`` (0, 1/2, 1)."""
    z = {0.0: -100.0, 0.5: 0.0, 1.0: 100.0}
    state.weights.zero_()
    for i, p in enumerate(probs):
        state.weights[i, 0] = z[p]


# b = 1: cost_r(k) = T_1(k).
T1 = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
# b = 8, K = 5: cost_r(k) = 80 - (80 - T_8(k)) / 8 = 75 + k.
T8 = [40.0, 48.0, 56.0, 64.0, 72.0, 80.0]


def test_reference_b1_hand_computed():
    state = make_state()
    state.cost[1] = torch.tensor(T1)
    cases = [
        # E(k) ~ 1 + k, score (1 + k) / (10 + k) rises with k: keep all 5.
        ([1.0] * 5, 5),
        # E(k) ~ 1: 1/10 beats (1 + 1e-6)/11.
        ([1e-6, 1.0, 1.0, 1.0, 1.0], 0),
        # E = 1, 1.9, 2.35, 2.395, ...: scores 0.1, 0.1727, 0.1958, 0.1842: 2.
        ([0.9, 0.5, 0.1, 0.1, 0.1], 2),
    ]
    for conf, want in cases:
        idx = set_conf(state, [3], [conf])
        rmk.reference_row_mask(state, idx, 1, 5)
        assert int(state.live_len[0]) == want, (conf, want, state.live_len)
        assert state.active


def test_bias_weights_give_exact_probabilities():
    state = make_state()
    exact_probs(state, [0.0, 0.5, 1.0, 1.0, 1.0])
    z = state.weights[:, 0]
    p = torch.div(torch.ones(5), 1.0 + torch.exp(-z))
    assert p.tolist() == [0.0, 0.5, 1.0, 1.0, 1.0]


def test_reference_b1_exact_ties_take_larger_k():
    state = make_state()
    # k=0: 1/1 = 1; k=1 with p_1 = 1: 2/2 = 1 (tie); k=2 with p_2 = 0: 2/3.
    state.cost[1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    exact_probs(state, [1.0, 0.0, 0.0, 0.0, 0.0])
    idx = set_conf(state, [0], [[0.3] * 5])
    rmk.reference_row_mask(state, idx, 1, 5)
    assert int(state.live_len[0]) == 1
    # k=0: 1/2; k=1 with p_1 = 0.5: 1.5/3 = 1/2 (tie); k>=2: 1.5/4 and less.
    state.cost[1] = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    exact_probs(state, [0.5, 0.0, 0.0, 0.0, 0.0])
    rmk.reference_row_mask(state, idx, 1, 5)
    assert int(state.live_len[0]) == 1


def test_reference_b8_hand_computed():
    state = make_state()
    state.cost[8] = torch.tensor(T8)
    slots = [7, 2, 5, 0, 1, 6, 3, 4]
    rows = [
        [1.0] * 5,  # (1 + k)/(75 + k) rises: 5
        [1e-6, 1.0, 1.0, 1.0, 1.0],  # 1/75 beats 1.000001/76: 0
        [0.5, 1e-6, 1.0, 1.0, 1.0],  # 1/75=.01333, 1.5/76=.01974, 1.5000005/77: 1
        [1.0, 1.0, 1e-6, 1.0, 1.0],  # 3/77 = .03896 beats 3.000001/78: 2
        [1.0, 1.0, 1.0, 1e-6, 1.0],  # 4/78 beats 4.000001/79: 3
        [1.0, 1.0, 1.0, 1.0, 1e-6],  # 5/79 beats 5.000001/80: 4
        [0.02, 1.0, 1.0, 1.0, 1.0],  # see below: 5
        [0.01, 0.01, 0.01, 0.01, 0.01],  # 1/75 = .013333 > 1.01/76 = .013289: 0
    ]
    idx = set_conf(state, slots, rows)
    rmk.reference_row_mask(state, idx, 8, 5)
    # Row 6: E(k) ~ 1 + 0.02k, scores 1/75 = .013333, 1.02/76 = .013421,
    # 1.04/77 = .013506, 1.06/78 = .013590, 1.08/79 = .013671, 1.1/80 = .01375.
    assert state.live_len.tolist() == [5, 0, 1, 2, 3, 4, 5, 0]


def test_reference_b8_k3_all_ties_take_k():
    state = make_state()
    # K = 3: cost_r(k) = 9 - (9 - T_8(k)) / 8 = [8, 9, 9, 9].
    state.cost[8] = torch.tensor([1.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    # p = 1/2, 0, 0: E = [1, 1.5, 1.5, 1.5]: scores 1/8, 1/6, 1/6, 1/6.
    exact_probs(state, [0.5, 0.0, 0.0, 1.0, 1.0])
    idx = set_conf(state, list(range(8)), [[0.3] * 5] * 8)
    rmk.reference_row_mask(state, idx, 8, 3)
    assert state.live_len.tolist() == [3] * 8


def test_reference_threshold_rule_at_b2():
    state = make_state()
    state.cost[2] = torch.tensor(T1)
    state.cost[1] = torch.tensor(T1)
    state.tau[2] = 0.25
    # p = 1/2, 1/2, 1, 0, ...: running products 0.5, 0.25, 0.25, 0: keep 3.
    exact_probs(state, [0.5, 0.5, 1.0, 0.0, 1.0])
    idx = set_conf(state, [1, 4], [[0.3] * 5] * 2)
    rmk.reference_row_mask(state, idx, 2, 5)
    assert state.live_len[:2].tolist() == [3, 3]
    # p_1 = 0 keeps none; at one request the cost rule runs whatever tau holds.
    exact_probs(state, [0.0, 1.0, 1.0, 1.0, 1.0])
    rmk.reference_row_mask(state, idx, 2, 5)
    assert state.live_len[:2].tolist() == [0, 0]
    state.tau[1] = 0.9
    exact_probs(state, [0.5, 0.5, 0.5, 0.5, 0.5])
    rmk.reference_row_mask(state, idx[:1], 1, 5)
    # Cost rule at b = 1: E = 1, 1.5, 1.75, 1.875: scores .1, .136, .146, .144: 2.
    assert int(state.live_len[0]) == 2


def test_row_live_layout_and_tail():
    state = make_state()
    state.cost[1] = torch.tensor(T1)
    state.cost[2] = torch.tensor(T1)
    span = MAX_REQS * (MAX_DRAFT + 1)
    # A sentinel past the maskable span: no step may write it.
    state.row_live[span:] = 7
    # A wide step masks rows across 0 .. 11.
    idx = set_conf(state, [4, 1], [[1.0] * 5, [1e-6] + [1.0] * 4])
    rmk.reference_row_mask(state, idx, 2, 5)
    assert state.live_len[:2].tolist() == [5, 0]
    assert state.row_live[:12].tolist() == [1] * 6 + [1, 0, 0, 0, 0, 0]
    # A narrower step (1 request, K = 3) rewrites rows 0 .. 3; rows 4 .. 11
    # of the earlier step are live again.
    idx = set_conf(state, [2], [[0.9, 0.5, 0.1]])
    rmk.reference_row_mask(state, idx, 1, 3)
    # cost_r = T_1 = 10, 11, 12, 13: E = 1, 1.9, 2.35, 2.395: keep 2.
    assert int(state.live_len[0]) == 2
    assert state.row_live[:4].tolist() == [1, 1, 1, 0]
    assert state.row_live[4:span].tolist() == [1] * (span - 4)
    assert state.row_live[span:].tolist() == [7] * (MAX_TOKENS - span)


def step(num_drafts, *, has_prefill=False, num_tokens=None, max_query_len=None, slots=None,
         req_ids=None):
    n = len(num_drafts)
    slots = list(range(n)) if slots is None else slots
    req_ids = [f"r{s}" for s in slots] if req_ids is None else req_ids
    drafts = None if num_drafts is None else np.asarray(num_drafts, dtype=np.int32)
    if num_tokens is None:
        num_tokens = int(sum(k + 1 for k in num_drafts))
    return SimpleNamespace(
        num_reqs=n,
        req_ids=req_ids,
        idx_mapping=torch.tensor(slots, dtype=torch.int64),
        idx_mapping_np=np.asarray(slots, dtype=np.int32),
        num_tokens=num_tokens,
        num_draft_tokens_per_req=drafts,
        has_prefill=has_prefill,
        max_query_len=max_query_len,
    )


def masked(state: RowMaskState) -> None:
    state.cost[2] = torch.tensor(T1)
    set_conf(state, [0, 1], [[1e-6] + [1.0] * 4, [1e-6] + [1.0] * 4])
    rmk.begin_step(state, step([5, 5]), dummy_run=False)
    assert state.active
    assert state.live_len[:2].tolist() == [0, 0]
    assert state.row_live[:12].tolist() == [1, 0, 0, 0, 0, 0] * 2


@pytest.mark.parametrize(
    "batch, dummy",
    [
        (dict(num_drafts=[5, 5], has_prefill=True), False),
        (dict(num_drafts=[5, 3]), False),
        (dict(num_drafts=[0, 0]), False),
        (dict(num_drafts=[5, 5]), True),
        (dict(num_drafts=[5, 5], max_query_len=6), False),
        (dict(num_drafts=[5, 5], num_tokens=11), False),
        (dict(num_drafts=[6, 6]), False),
    ],
    ids=["prefill", "mixed", "zero", "dummy", "reallocated", "tokens", "too-wide"],
)
def test_non_uniform_steps_reset(batch, dummy):
    state = make_state()
    masked(state)
    rmk.begin_step(state, step(**batch), dummy_run=dummy)
    assert not state.active
    assert bool((state.row_live == 1).all())


def test_no_drafts_scheduled_resets():
    state = make_state()
    masked(state)
    batch = step([0, 0])
    batch.num_draft_tokens_per_req = None
    rmk.begin_step(state, batch, dummy_run=False)
    assert not state.active
    assert bool((state.row_live == 1).all())


def test_uniform_step_masks_in_batch_order():
    state = make_state()
    state.cost[3] = torch.tensor(T1)
    # Slots 6, 0, 3 in batch order; only slot 0's request drops its drafts.
    set_conf(state, [6, 0, 3], [[1.0] * 3, [1e-6, 1.0, 1.0], [1.0] * 3])
    rmk.begin_step(state, step([3, 3, 3], slots=[6, 0, 3]), dummy_run=False)
    assert state.active
    assert state.live_len[:3].tolist() == [3, 0, 3]
    assert state.row_live[:12].tolist() == [1] * 4 + [1, 0, 0, 0] + [1] * 4


def test_runner_entry_is_a_no_op_without_state(monkeypatch):
    monkeypatch.setattr(dspark_row_mask, "_STATE", None)
    # A batch whose fields would fail if read: nothing is read.
    rmk.begin_target_step(object(), False)


def test_runner_entry_uses_the_shared_state(monkeypatch):
    state = make_state()
    monkeypatch.setattr(dspark_row_mask, "_STATE", state)
    state.cost[2] = torch.tensor(T1)
    set_conf(state, [0, 1], [[1e-6] + [1.0] * 4, [1.0] * 5])
    rmk.begin_target_step(step([5, 5]), False)
    assert state.live_len[:2].tolist() == [0, 5]
    rmk.begin_target_step(step([5, 5], has_prefill=True), False)
    assert not state.active and bool((state.row_live == 1).all())


# ---------------------------------------------------------------- numpy reference


def np_log_odds(q):
    # The kernel's bounds are constants folded in double, then rounded.
    q = np.minimum(np.maximum(F32(q), F32(PROB_EPS)), F32(1.0 - PROB_EPS))
    return F32(np.log(q) - np.log(F32(1.0) - q))


def np_decide(conf, stats, weights, cost_row, b, k_full, uses_stats, tau=0.0):
    """live_len and the decision's margin for one request (float32
    throughout, the kernel's operation order): the cost rule's best score's
    lead over the runner-up, or with ``tau`` > 0 the relative distance of
    the nearest running product to ``tau``."""
    t_full, t_zero, bb = F32(cost_row[k_full]), F32(cost_row[0]), F32(b)
    best = F32(F32(1.0) / F32(t_full - F32(F32(t_full - t_zero) / bb)))
    scores, prods = [best], []
    best_k, prod, expected = 0, F32(1.0), F32(1.0)
    for k in range(1, k_full + 1):
        w = weights[k - 1].astype(F32)
        z = F32(w[0] + F32(w[1] * np_log_odds(conf[k - 1])))
        if uses_stats:
            z = F32(z + F32(w[2] * np_log_odds(stats[0][k - 1])))
            z = F32(z + F32(w[3] * F32(stats[1][k - 1])))
        p = F32(F32(1.0) / F32(F32(1.0) + np.exp(F32(-z))))
        prod = F32(prod * p)
        expected = F32(expected + prod)
        cost = F32(t_full - F32(F32(t_full - F32(cost_row[k])) / bb))
        score = F32(expected / cost)
        scores.append(score)
        prods.append(float(prod))
        if (prod >= F32(tau)) if tau > 0 and b > 1 else (score >= best):
            best, best_k = score, k
    if tau > 0 and b > 1:
        return best_k, min([abs(p - tau) / tau for p in prods] or [1.0])
    others = [s for i, s in enumerate(scores) if i != best_k]
    lead = (best - max(others)) / best if others else 1.0
    return best_k, float(lead)


def np_raw_rule(conf, cost_row, b, k_full):
    """The confidence-only rule: p_i = conf_i (float64; its best k and lead)."""
    t_full = cost_row[k_full]
    scores, prod, expected = [], 1.0, 1.0
    for k in range(0, k_full + 1):
        if k:
            prod *= conf[k - 1]
            expected += prod
        scores.append(expected / (t_full - (t_full - cost_row[k]) / b))
    best_k = max(range(k_full + 1), key=lambda k: (scores[k], k))
    others = [s for i, s in enumerate(scores) if i != best_k]
    return best_k, (scores[best_k] - max(others)) / scores[best_k]


# ------------------------------------------------------------ interpreted kernels


def _random_case(gen, max_draft, b, k, trial):
    """Features, weights, costs and thresholds of one case; trials: 0
    random, 1 dump-like (confidences near one, fitted-like weights, the
    threshold rule at b >= 2), 2 raw-confidence policy, 3 draft-logit
    statistics with the threshold rule at b >= 2."""
    conf = torch.rand(MAX_REQS, max_draft, generator=gen)
    stats = torch.stack(
        (torch.rand(MAX_REQS, max_draft, generator=gen),
         torch.rand(MAX_REQS, max_draft, generator=gen) * 6.0), dim=1)
    weights = torch.randn(max_draft, len(FEATURES), generator=gen)
    uses_stats = trial == 3
    if not uses_stats:
        weights[:, 2:4] = 0.0
    if trial == 1:
        conf = conf.sqrt().sqrt()
        conf[0, :2] = torch.tensor([1.0, 0.0])  # clamped
        weights[:, 0] = torch.linspace(0.5, -1.0, max_draft)
        weights[:, 1] = torch.linspace(1.2, 0.6, max_draft)
    if trial == 2:
        weights.zero_()
        weights[:, 1] = 1.0
        conf = conf * 0.98 + 0.01
    steps = torch.rand(MAX_REQS + 1, max_draft + 1, generator=gen) + 0.05
    cost = 20.0 + steps.cumsum(1)
    cost[0] = 0.0
    idx = torch.randperm(MAX_REQS, generator=gen)[:b]
    if trial == 0:
        weights[:, 0] += 2.0
        conf[idx[0], 0] = 1e-4
        weights[0, 1] = 3.0  # the first request drops every draft
    if trial == 1 and b > 1:
        conf[idx[1]] = 1.0  # keeps every draft
        weights[:, 1] = weights[:, 1].clamp(min=1.0)
    tau = torch.zeros(MAX_REQS + 1)
    if trial in (1, 3):
        tau[2:] = torch.rand(MAX_REQS - 1, generator=gen) * 0.8 + 0.05
    return conf, stats, weights, cost, idx, uses_stats, tau


def interpreted_results() -> dict:
    """Run in the Triton interpreter: per random case, the kernels' buffers
    and their inputs."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.spec_decode import dspark_row_mask_kernel as kernel

    gen = torch.Generator().manual_seed(0)
    out = []
    max_draft, span_tokens = 7, 96
    for b in range(1, MAX_REQS + 1):
        for k in (1, 3, 5, 7):
            for trial in range(4):
                conf, stats, weights, cost, idx, uses_stats, tau = _random_case(
                    gen, max_draft, b, k, trial)
                s = RowMaskState(MAX_REQS, span_tokens, max_draft, torch.device("cpu"))
                s.conf_by_slot.copy_(conf)
                s.stats_by_slot.copy_(stats)
                s.weights.copy_(weights)
                s.cost.copy_(cost)
                s.tau.copy_(tau)
                s.uses_stats = uses_stats
                row_live = torch.randint(0, 2, (span_tokens,), generator=gen).to(torch.int8)
                s.row_live.copy_(row_live)
                s.live_len.fill_(-3)
                p_hat = torch.full((MAX_REQS, 5), -2.0) if trial in (1, 3) else None
                kernel.compute_row_mask(s, idx, b, k, p_hat)
                live_len = s.live_len.tolist()
                rows = s.row_live.tolist()
                out.append({
                    "case": [max_draft, b, k, trial],
                    "uses_stats": uses_stats,
                    "conf": conf.tolist(), "stats": stats.tolist(),
                    "weights": weights.tolist(), "cost": cost.tolist(), "tau": tau.tolist(),
                    "idx": idx.tolist(), "row_live_before": row_live.tolist(),
                    "live_len": live_len, "row_live": rows,
                    "active": s.active,
                    "p_hat": None if p_hat is None else p_hat.tolist(),
                })
    return {"cases": out, "span_tokens": span_tokens}


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.v1.spec_decode.test_dspark_row_mask_kernel import "
            "interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_interpreted_kernel_matches_numpy_reference(interpreted):
    cases = interpreted["cases"]
    span_tokens = interpreted["span_tokens"]
    assert len(cases) == MAX_REQS * 4 * 4
    seen = set()
    for c in cases:
        max_draft, b, k, _ = c["case"]
        assert c["active"], c["case"]
        weights = np.asarray(c["weights"], dtype=F32)
        want = []
        for slot in c["idx"]:
            d, _ = np_decide(c["conf"][slot], [c["stats"][slot][0], c["stats"][slot][1]],
                             weights, c["cost"][b], b, k, c["uses_stats"], c["tau"][b])
            want.append(d)
        assert c["live_len"][:b] == want, (c["case"], c["live_len"], want)
        seen.update(want)
        # live_len past the batch is never written.
        assert c["live_len"][b:] == [-3] * (MAX_REQS - b), c["case"]
        rows = []
        for d in want:
            rows += [1 if j <= d else 0 for j in range(k + 1)]
        span = min(MAX_REQS * (max_draft + 1), span_tokens)
        expect = rows + [1] * (span - len(rows)) + c["row_live_before"][span:]
        assert c["row_live"] == expect, c["case"]
    # d covers every depth 0 .. 7.
    assert seen == set(range(8)), sorted(seen)


def np_probs(conf, stats, weights, n, uses_stats):
    """p_i for positions 1 .. n (float32, the kernel's operation order)."""
    out = []
    for k in range(1, n + 1):
        w = weights[k - 1].astype(F32)
        z = F32(w[0] + F32(w[1] * np_log_odds(conf[k - 1])))
        if uses_stats:
            z = F32(z + F32(w[2] * np_log_odds(stats[0][k - 1])))
            z = F32(z + F32(w[3] * F32(stats[1][k - 1])))
        out.append(float(F32(F32(1.0) / F32(F32(1.0) + np.exp(F32(-z))))))
    return out


def test_interpreted_p_hat(interpreted):
    with_p = [c for c in interpreted["cases"] if c["p_hat"] is not None]
    assert len(with_p) == len(interpreted["cases"]) // 2
    for c in with_p:
        _, b, k, _ = c["case"]
        weights = np.asarray(c["weights"], dtype=F32)
        for r, slot in enumerate(c["idx"]):
            want = np_probs(c["conf"][slot], c["stats"][slot], weights, 5, c["uses_stats"])
            np.testing.assert_allclose(c["p_hat"][r], want, rtol=2e-6, err_msg=str(c["case"]))
        assert c["p_hat"][b:] == [[-2.0] * 5] * (MAX_REQS - b), c["case"]


def test_raw_confidence_policy_is_the_confidence_rule(interpreted):
    compared = total = 0
    for c in interpreted["cases"]:
        if c["case"][3] != 2:
            continue
        _, b, k, _ = c["case"]
        for r, slot in enumerate(c["idx"]):
            total += 1
            d, lead = np_raw_rule(c["conf"][slot], c["cost"][b], b, k)
            if lead > 1e-5:
                compared += 1
                assert c["live_len"][r] == d, (c["case"], r)
    assert compared >= 0.95 * total, (compared, total)


def test_torch_reference_matches_numpy_reference():
    gen = torch.Generator().manual_seed(1)
    agree = total = 0
    for b in range(1, MAX_REQS + 1):
        for k in (1, 3, 7):
            for trial in range(4):
                conf, stats, weights, cost, idx, uses_stats, tau = _random_case(
                    gen, 7, b, k, trial)
                s = RowMaskState(MAX_REQS, 96, 7, torch.device("cpu"))
                s.conf_by_slot.copy_(conf)
                s.stats_by_slot.copy_(stats)
                s.weights.copy_(weights)
                s.cost.copy_(cost)
                s.tau.copy_(tau)
                s.uses_stats = uses_stats
                rmk.reference_row_mask(s, idx, b, k)
                for r, slot in enumerate(idx.tolist()):
                    d, lead = np_decide(conf[slot].tolist(), stats[slot].tolist(),
                                        weights.numpy(), cost[b].tolist(), b, k, uses_stats,
                                        float(tau[b]))
                    total += 1
                    # torch's and numpy's exp/log may round differently.
                    if lead > 1e-5:
                        agree += int(int(s.live_len[r]) == d)
                    else:
                        agree += 1
    assert agree == total


def forced_depth_policy(k: int, max_depth: int = 7) -> dict:
    """The policy file that forces depth k: p_i = 1 up to k, ~2e-9 after;
    costs rising 1 % per draft so the larger depth wins while p = 1."""
    return {
        "schema": "dspark-depth-policy/1",
        "max_depth": max_depth,
        "features": ["bias"],
        "weights": [[20.0] if i < k else [-20.0] for i in range(max_depth)],
        "costs": {"1": {str(j): round(1.0 + 0.01 * j, 2) for j in range(max_depth + 1)}},
    }


@pytest.mark.parametrize("tau", (None, 0.5))
@pytest.mark.parametrize("k", range(8))
def test_policy_file_forces_a_depth(tmp_path, k, tau):
    path = tmp_path / "forced.json"
    policy = forced_depth_policy(k)
    if tau is not None:
        policy["tau"] = {"2": tau}
    path.write_text(json.dumps(policy))
    policy = dspark_row_mask.load_policy(path, MAX_REQS, 7, [0] + [7] * MAX_REQS)
    state = RowMaskState(MAX_REQS, 96, 7, torch.device("cpu"))
    state.set_policy(policy)
    gen = torch.Generator().manual_seed(k)
    state.conf_by_slot.copy_(torch.rand(MAX_REQS, 7, generator=gen))
    for b in range(1, MAX_REQS + 1):
        for big_k in (1, 3, 5, 7):
            idx = torch.randperm(MAX_REQS, generator=gen)[:b]
            rmk.reference_row_mask(state, idx, b, big_k)
            assert state.live_len[:b].tolist() == [min(k, big_k)] * b, (b, big_k)
            for slot in idx.tolist():
                d, lead = np_decide(state.conf_by_slot[slot].tolist(), [[0.0] * 7] * 2,
                                    state.weights.numpy(), state.cost[b].tolist(), b, big_k,
                                    False, float(state.tau[b]))
                assert d == min(k, big_k) and lead > 1e-6, (b, big_k, lead)
