# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair sites inside captured CUDA graphs (``vllm_sm80.hot_swap``) on one
card (skipped without CUDA).

- A toy pair: one graph holds both variants; replay gives the incumbent's
  bytes, the candidate's after ``_apply``, the incumbent's again after the
  switch back; eager calls follow the arm.
- The cost of carrying the other variant: the replay time of a chain of 300
  pair sites (incumbent arm, candidates disabled) against the same chain
  captured plainly, per disabled node (printed with ``-s``; bounded).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_sm80 import hot_swap

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SITES = 300
REPLAYS = 200
RUNNER = SimpleNamespace(
    vllm_config=SimpleNamespace(speculative_config=None),
    adaptive_verification=None, speculator=None,
)


@pytest.fixture
def hot(monkeypatch):
    fresh = hot_swap._HotState(listed=("toy",), resolved=True)
    monkeypatch.setattr(hot_swap, "_STATE", fresh)
    monkeypatch.setitem(hot_swap.HOT_PAIRS, "toy", hot_swap.HotPair(
        "toy", "VLLM_TOY", lambda r: None, collective_neutral=True))
    yield fresh
    torch.cuda.synchronize()


def _mul(x, k):
    """A pair whose incumbent scales by k and candidate by k + 1."""
    return hot_swap.pair(
        "toy", lambda: torch.mul(x, k),
        lambda out: torch.mul(x, k + 1) if out is None else torch.mul(x, k + 1, out=out))


def _capture(fn):
    """Capture fn() into one graph inside the hot capture window (an eager
    warm-up first, as the runner does); (graph, output)."""
    held = {}

    def body(runner, *, profile_only):
        fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            held["out"] = fn()
        held["graph"] = g
        return 0

    hot_swap.capture(RUNNER, body, False)
    return held["graph"], held["out"]


def _replay(g):
    g.replay()
    torch.cuda.synchronize()


def test_toy_pair_toggles_inside_the_graph(hot):
    x = torch.randn(4096, device="cuda")
    g, out = _capture(lambda: _mul(x, 2.0))
    _replay(g)
    assert torch.equal(out, x * 2)
    hot_swap._apply(RUNNER, frozenset({"toy"}))
    _replay(g)
    assert torch.equal(out, x * 3)
    assert torch.equal(_mul(x, 2.0), x * 3)  # eager follows the arm
    hot_swap._apply(RUNNER, frozenset())
    _replay(g)
    assert torch.equal(out, x * 2)
    assert torch.equal(_mul(x, 2.0), x * 2)
    site = hot_swap._sites()["toy"]
    assert (site["sites"], site["base_nodes"], site["cand_nodes"]) == (1, 1, 1)
    assert site["base_kernels"] and site["cand_kernels"]


def _median_replay_us(g) -> float:
    for _ in range(20):
        g.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(REPLAYS):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return sorted(times)[len(times) // 2]


def test_disabled_nodes_cost(hot):
    x = torch.randn(1024, device="cuda")

    def chain(pairs: bool):
        y = x
        for _ in range(SITES):
            y = _mul(y, 1.0) if pairs else torch.mul(y, 1.0)
        return y

    plain = torch.cuda.CUDAGraph()
    chain(False)
    with torch.cuda.graph(plain):
        chain(False)
    g, out = _capture(lambda: chain(True))
    t_plain = _median_replay_us(plain)
    t_base = _median_replay_us(g)
    hot_swap._apply(RUNNER, frozenset({"toy"}))
    t_cand = _median_replay_us(g)
    hot_swap._apply(RUNNER, frozenset())
    per_node = (t_base - t_plain) / SITES
    print(f"\n{SITES} sites: plain {t_plain:.1f} us, hot incumbent {t_base:.1f} us, "
          f"hot candidate {t_cand:.1f} us; {per_node * 1e3:.0f} ns per disabled node; "
          f"toggle {hot.toggle_ms:.2f} ms")
    assert per_node < 1.0
