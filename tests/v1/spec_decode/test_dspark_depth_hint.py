# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for VLLM_DSPARK_DEPTH_HINT (dspark_depth_hint.py)."""

from types import SimpleNamespace as S

import pytest

from vllm_sm80.spec_decode import dspark_depth_hint as hint

pytestmark = pytest.mark.cpu_test

TABLE = "full=7,7,5,5;base=5,5,5,3"


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_HINT", TABLE)
    hint.widths.cache_clear()
    yield
    hint.widths.cache_clear()


def _chat(tools=None, tool_choice="auto", fmt=None):
    return S(
        messages=[{"role": "user", "content": "Write Python code."}],
        tools=tools,
        tool_choice=tool_choice,
        response_format=fmt,
    )


@pytest.mark.parametrize(
    "request_, marked",
    [
        (_chat(), False),  # the prompt's words never decide
        (_chat(tools=[{"type": "function"}]), True),
        (_chat(tools=[{"type": "function"}], tool_choice="none"), False),
        (_chat(fmt=S(type="json_schema")), True),
        (_chat(fmt=S(type="text")), False),
    ],
)
def test_only_request_fields_mark(on, request_, marked):
    extra = {}
    hint.mark(request_, extra)
    assert extra == ({hint.KEY: "full"} if marked else {})


def test_off_marks_and_installs_nothing(monkeypatch):
    monkeypatch.delenv("VLLM_DSPARK_DEPTH_HINT", raising=False)
    hint.widths.cache_clear()
    extra = {}
    hint.mark(_chat(tools=[{"type": "function"}]), extra)
    assert extra == {} and not hint.enabled()
    hint.install()
    from vllm.v1.core.sched.scheduler import Scheduler

    assert not getattr(Scheduler._select_adaptive_k, "_dspark_depth_hint", False)


@pytest.mark.parametrize("raw", ["full=7", "full=7;base=5;deep=9"])
def test_bad_tables_refuse(monkeypatch, raw):
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_HINT", raw)
    hint.widths.cache_clear()
    with pytest.raises(ValueError):
        hint.widths()
    hint.widths.cache_clear()


def _req(rid, marked=False, prefill=False, drafts=7):
    return S(
        request_id=rid,
        is_prefill_chunk=prefill,
        spec_token_ids=list(range(drafts)),
        sampling_params=S(extra_args={hint.KEY: "full"} if marked else None),
    )


@pytest.mark.parametrize(
    "full, marked, load_width, expected",
    [
        ([True], [False], 7, (7, 7)),
        ([False], [False], 7, (5, 5)),
        ([False], [True], 7, (7, 7)),  # tools / JSON: full without history
        ([True, False], [False, False], 7, (5, 5)),  # one batch width
        ([True, True], [False, False], 7, (7, 7)),
        ([True] * 3, [False] * 3, 7, (5, 5)),  # full's third entry
        ([True] * 4, [False] * 4, 5, (5, 5)),
        ([False] * 4, [False] * 4, 5, (3, 5)),  # base verifies 3 of a 5-block
        ([True] * 8, [False] * 8, 5, (5, 5)),  # the last entry repeats
        ([True], [False], 5, (5, 5)),  # capped by the load width
    ],
)
def test_select(on, full, marked, load_width, expected):
    running = [_req(str(i), m) for i, m in enumerate(marked)]
    rounds = {str(i): f for i, f in enumerate(full)}
    assert hint.select(running, rounds, len(running), load_width) == expected


def test_first_and_prefill_steps_take_base(on):
    assert hint.select([_req("a")], {}, 1, 7) == (5, 5)
    assert hint.select([_req("a", prefill=True)], {"a": True}, 1, 7) == (5, 5)


def _installed(monkeypatch, by_load):
    """install() on a fresh adaptive-k policy; returns run(running) ->
    ((verify, draft), drafts left per request)."""
    from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig, AdaptiveKPolicy
    from vllm.v1.core.sched.scheduler import Scheduler

    for cls, name in (
        (Scheduler, "_select_adaptive_k"),
        (AdaptiveKPolicy, "observe"),
        (AdaptiveKPolicy, "forget"),
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    hint.install()
    hint.install()
    assert Scheduler._select_adaptive_k._dspark_depth_hint
    config = AdaptiveKConfig.from_dict({"by_load": by_load, "draft_by_load": True}, 7)
    policy = AdaptiveKPolicy(config)

    def run(running):
        sched = S(
            running=running,
            waiting=[],
            kv_holding_waiting=[],
            num_spec_tokens=7,
            adaptive_k=policy,
            _forced_depth=lambda: None,
        )
        choice = Scheduler._select_adaptive_k(sched)
        return choice, [len(r.spec_token_ids) for r in running]

    return policy, run


def test_a_width_without_graphs_refuses(on, monkeypatch):
    # base's 3 is in no by_load entry: its steps would run eager.
    _, run = _installed(monkeypatch, [7, 7, 5, 5, 5, 5, 5, 5])
    with pytest.raises(ValueError, match=r"widths \[3\]"):
        run([_req("a")])


def test_wrapper_follows_each_requests_last_round(on, monkeypatch):
    # The ninth entry, past max_num_seqs, captures 3 without serving it.
    policy, run = _installed(monkeypatch, [7, 7, 5, 5, 5, 5, 5, 5, 3])
    assert run([_req("a")]) == ((5, 5), [5])  # no history yet
    policy.observe("a", 5, 5)  # accepted every draft
    assert run([_req("a")]) == ((7, 7), [7])
    assert run([_req("a", drafts=5)]) == ((5, 7), [5])  # 5 on hand: ask for 7
    policy.observe("a", 7, 5)  # a rejection
    assert run([_req("a")]) == ((5, 5), [5])
    policy.mark_padded("a")
    policy.observe("a", 5, 0)  # placeholder drafts: no information
    assert run([_req("a")]) == ((5, 5), [5])
    for rid in "bcd":
        policy.observe(rid, 3, 3)
    policy.observe("a", 3, 3)
    assert run([_req(r) for r in "abcd"]) == ((5, 5), [5, 5, 5, 5])
    policy.forget("d")
    assert run([_req(r) for r in "abcd"]) == ((3, 5), [3, 3, 3, 3])
