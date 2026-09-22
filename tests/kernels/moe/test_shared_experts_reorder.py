# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the re-ordered multi-stream shared-expert overlap.

``VLLM_GLM5_SHARED_EXPERT_REORDER`` changes *when* the shared experts are
submitted to the aux stream, and nothing else. With the flag off the runner
submits them before the gate and the routed dispatch and joins afterwards
(upstream #52033); with it on the runner marks the aux stream's start point,
enqueues the routed experts, and only then runs the shared experts on the aux
stream before joining.

Everything here runs on the host. The streams, events, expert layers and input
tensors are fakes that append to one ordered log, and the ordering code under
test is the real ``MoERunner._forward_impl`` / ``_apply_quant_method`` /
``_maybe_apply_shared_experts`` and the real ``SharedExperts``, bound onto a
stub runner that supplies only the attributes those three methods touch. The
tests therefore assert the enqueue order the runner actually produces.

Covered: the exact order under each flag value, that the join precedes every
read of the shared output, that the output slot is returned between steps, the
token-threshold boundary and our real prefill widths, the disabled-stream and
modular-kernel guards, and that only event edges (no ``record_stream``) are
emitted while a cudagraph capture is underway.

Not covered, and deliberately so: that the two streams really run concurrently,
that a capture containing this fork/join is legal to the driver, and that the
step time moves. Those need the cards and are in the GPU validation plan.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)

HIDDEN = 8
# Pinned rather than read from envs, so the expectations below do not depend on
# the ambient environment. 256 is the upstream default.
THRESHOLD = 256
# Decode at c1 with the DFlash2 drafter at k=3 is M=4 tokens per step.
DECODE_TOKENS = 4
# Our production prefill chunk, and the micro-batches the TP prefill overlap
# splits it into at SPLITS=2 and SPLITS=4.
PREFILL_WIDTHS = (1152, 576, 288)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _Log(list):
    def ops(self) -> list[str]:
        return [entry[0] for entry in self]

    def index_of(self, op: str) -> int:
        assert op in self.ops(), f"{op} not in {self.ops()}"
        return self.ops().index(op)


class _FakeStream:
    """A CUDA stream that records the edges drawn to and from it."""

    def __init__(self, name: str, log: _Log) -> None:
        self.name = name
        self._log = log

    def wait_stream(self, other: "_FakeStream") -> None:
        self._log.append(("wait_stream", self.name, other.name))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<stream {self.name}>"


class _FakeEvent:
    """torch.cuda.Event stand-in, named by its slot within one harness."""

    def __init__(self, *args, **kwargs) -> None:
        # Filled in by the factory in _build; a bare construction is inert.
        self.name = "e?"
        self.log: _Log | None = None

    def record(self, stream: _FakeStream) -> None:
        assert self.log is not None
        self.log.append(("record", self.name, stream.name))

    def wait(self, stream: _FakeStream) -> None:
        assert self.log is not None
        self.log.append(("wait", self.name, stream.name))


class _StreamState:
    """Tracks which fake stream is 'current'."""

    def __init__(self, main: _FakeStream) -> None:
        self.main = main
        self.current = main


class _FakeTensor:
    """Enough tensor for the overlap decision and for record_stream."""

    def __init__(self, tokens: int, log: _Log, tag: str = "x") -> None:
        self.shape = (tokens, HIDDEN)
        self._log = log
        self.tag = tag

    def record_stream(self, stream: _FakeStream) -> None:
        self._log.append(("record_stream", self.tag, stream.name))


class _FakeSharedLayer(torch.nn.Module):
    """Stands in for the shared-expert MLP; logs the stream it ran on."""

    def __init__(self, log: _Log, state: _StreamState) -> None:
        super().__init__()
        self._log = log
        self._state = state
        self.calls = 0

    def forward(self, x):  # type: ignore[override]
        self.calls += 1
        self._log.append(("shared_experts", self._state.current.name))
        return _FakeTensor(x.shape[0], self._log, tag="shared_out")


class _LoggingSharedExperts(SharedExperts):
    """SharedExperts that records when its output is consumed."""

    def __init__(self, *args, log: _Log, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._log = log

    @property
    def output(self):
        self._log.append(("read_shared_output",))
        return SharedExperts.output.fget(self)


class _StubRunner:
    """The three real MoERunner methods over a minimal set of attributes."""

    _forward_impl = MoERunner._forward_impl
    _apply_quant_method = MoERunner._apply_quant_method
    _maybe_apply_shared_experts = MoERunner._maybe_apply_shared_experts

    def __init__(self, shared: SharedExperts | None, log: _Log) -> None:
        self._shared_experts = shared
        self.shared_experts = shared
        self._log = log
        self.gate = None
        self.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=True),
            _ensure_moe_quant_config_init=lambda: None,
            forward_monolithic=self._forward_monolithic,
        )

    # -- pieces _forward_impl leans on, faked ------------------------------- #

    def _forward_monolithic(self, x, router_logits, input_ids=None):
        self._log.append(("routed_experts",))
        return _FakeTensor(x.shape[0], self._log, tag="routed_out")

    def _sequence_parallel_context(self):
        return nullcontext()

    def _maybe_dispatch(self, hidden_states, router_logits):
        self._log.append(("dispatch",))
        return hidden_states, router_logits

    def _maybe_combine(self, shared_output, hidden_states):
        self._log.append(("combine", getattr(shared_output, "tag", None)))
        return shared_output, hidden_states


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _build(
    monkeypatch,
    *,
    reorder: bool,
    capturing: bool = False,
    disable_stream: bool = False,
    mk_overlap: bool = False,
    unsafe_backend: bool = False,
):
    log = _Log()
    main = _FakeStream("main", log)
    aux = _FakeStream("aux", log)
    state = _StreamState(main)

    # Events are named in creation order *within this harness*, so the expected
    # logs below do not depend on how many harnesses ran before this one.
    made: list[_FakeEvent] = []

    def _event_factory(*a, **kw):
        ev = _FakeEvent()
        ev.log = log
        ev.name = f"e{len(made) + 1}"
        made.append(ev)
        return ev

    def _stream_ctx(stream):
        class _Ctx:
            def __enter__(_self):
                log.append(("enter_stream", stream.name))
                state.current = stream

            def __exit__(_self, *exc):
                log.append(("exit_stream", stream.name))
                state.current = main
                return False

        return _Ctx()

    # Pin every environment input the overlap decision reads.
    monkeypatch.setenv("VLLM_GLM5_SHARED_EXPERT_REORDER", "1" if reorder else "0")
    monkeypatch.setenv(
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1" if disable_stream else "0"
    )
    monkeypatch.setenv(
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", str(THRESHOLD)
    )
    monkeypatch.setattr(se_mod, "aux_stream", lambda: aux)
    monkeypatch.setattr(se_mod, "current_stream", lambda: state.current)
    monkeypatch.setattr(se_mod, "_capture_underway", lambda: capturing)
    monkeypatch.setattr(
        se_mod, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    monkeypatch.setattr(torch.cuda, "Event", _event_factory)
    monkeypatch.setattr(torch.cuda, "stream", _stream_ctx)

    layer = _FakeSharedLayer(log, state)
    parallel = SimpleNamespace(
        enable_eplb=unsafe_backend,
        all2all_backend="naive" if unsafe_backend else "allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
    )
    shared = _LoggingSharedExperts(
        layer,
        moe_config=SimpleNamespace(moe_parallel_config=parallel),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: mk_overlap,
        log=log,
    )
    return _StubRunner(shared, log), log, layer


def _run(runner, log, tokens=DECODE_TOKENS):
    x = _FakeTensor(tokens, log, tag="x")
    return runner._forward_impl(x, router_logits=None, shared_experts_input=x)


def _serial_log():
    """The log of a step that never touches the aux stream.

    The dispatch happens in _forward_impl; the serial NO_OVERLAP shared-expert
    call is the first thing _apply_quant_method does, just before the routed
    experts.
    """
    return [
        ("dispatch",),
        ("shared_experts", "main"),
        ("routed_experts",),
        ("read_shared_output",),
        ("combine", "shared_out"),
    ]


# --------------------------------------------------------------------------- #
# Enqueue order
# --------------------------------------------------------------------------- #


def test_flag_off_keeps_upstream_order(monkeypatch):
    """Off: shared experts are submitted to the aux stream first, joined after."""
    runner, log, _ = _build(monkeypatch, reorder=False)
    _run(runner, log)

    assert log == [
        ("record", "e1", "main"),
        ("enter_stream", "aux"),
        ("wait", "e1", "aux"),
        ("shared_experts", "aux"),
        ("record", "e3", "aux"),
        ("exit_stream", "aux"),
        ("dispatch",),
        ("routed_experts",),
        ("wait", "e3", "main"),
        ("read_shared_output",),
        ("combine", "shared_out"),
    ]
    # The upstream path syncs by event; it draws no stream-to-stream edges and
    # leaves no allocator hint.
    assert "record_stream" not in log.ops()
    assert "wait_stream" not in log.ops()


def test_flag_on_runs_shared_experts_after_routed_dispatch(monkeypatch):
    """On: mark the aux stream, enqueue routed, then shared on aux, then join."""
    runner, log, _ = _build(monkeypatch, reorder=True)
    _run(runner, log)

    assert log == [
        ("record_stream", "x", "aux"),
        ("wait_stream", "aux", "main"),
        ("dispatch",),
        ("routed_experts",),
        ("enter_stream", "aux"),
        ("shared_experts", "aux"),
        ("exit_stream", "aux"),
        ("wait_stream", "main", "aux"),
        ("read_shared_output",),
        ("combine", "shared_out"),
    ]


def test_order_flips_only_the_enqueue_order(monkeypatch):
    """The same two units of work, on the same two streams, either way round."""
    off_runner, off_log, off_layer = _build(monkeypatch, reorder=False)
    _run(off_runner, off_log)
    on_runner, on_log, on_layer = _build(monkeypatch, reorder=True)
    _run(on_runner, on_log)

    for log in (off_log, on_log):
        assert ("shared_experts", "aux") in log
        assert ("routed_experts",) in log
        assert ("combine", "shared_out") in log
    assert off_layer.calls == on_layer.calls == 1

    assert off_log.index_of("shared_experts") < off_log.index_of("routed_experts")
    assert on_log.index_of("routed_experts") < on_log.index_of("shared_experts")


@pytest.mark.parametrize("reorder", [False, True])
def test_join_precedes_every_read_of_the_shared_output(monkeypatch, reorder):
    """Whichever order, the main stream joins before the output is consumed."""
    runner, log, _ = _build(monkeypatch, reorder=reorder)
    _run(runner, log)

    # The join is the edge that lands *on* the main stream, not the fork.
    if reorder:
        join = log.index(("wait_stream", "main", "aux"))
    else:
        join = log.index(next(e for e in log if e[0] == "wait" and e[2] == "main"))
    assert join < log.index_of("read_shared_output")
    assert log.index_of("read_shared_output") < log.index_of("combine")
    # The shared experts are enqueued before the join; the join is what makes
    # their result safe for the main stream to read, so nothing may be read
    # between the two.
    assert log.index_of("shared_experts") < join


@pytest.mark.parametrize("reorder", [False, True])
def test_output_slot_is_released_between_steps(monkeypatch, reorder):
    """Two steps in a row: a stale _output would trip the runner's assert."""
    runner, log, layer = _build(monkeypatch, reorder=reorder)
    _run(runner, log)
    log.clear()
    _run(runner, log)

    assert layer.calls == 2
    assert ("combine", "shared_out") in log
    assert runner._shared_experts._output == [None, None]


# --------------------------------------------------------------------------- #
# When the multi-stream order is not chosen
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("tokens", PREFILL_WIDTHS)
def test_prefill_widths_stay_serial_on_the_main_stream(monkeypatch, reorder, tokens):
    """Our real prefill widths are all above the threshold, either way round.

    1152 is the production chunk; 576 and 288 are what the TP prefill overlap
    splits it into at SPLITS=2 and SPLITS=4. None of them reaches the aux
    stream, so the collectives that region intercepts see no fork or join.
    """
    runner, log, _ = _build(monkeypatch, reorder=reorder)
    _run(runner, log, tokens=tokens)

    assert log == _serial_log()


@pytest.mark.parametrize("reorder", [False, True])
def test_threshold_is_inclusive(monkeypatch, reorder):
    """At exactly the threshold the aux path engages; one token more and it does not."""
    runner, log, _ = _build(monkeypatch, reorder=reorder)
    _run(runner, log, tokens=THRESHOLD)
    assert ("shared_experts", "aux") in log

    runner, log, _ = _build(monkeypatch, reorder=reorder)
    _run(runner, log, tokens=THRESHOLD + 1)
    assert ("shared_experts", "main") in log


@pytest.mark.parametrize("reorder", [False, True])
def test_disabled_stream_forces_the_serial_path(monkeypatch, reorder):
    """VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 keeps the flag inert."""
    runner, log, _ = _build(monkeypatch, reorder=reorder, disable_stream=True)
    _run(runner, log)

    assert runner._shared_experts._stream is None
    assert log == _serial_log()


@pytest.mark.parametrize("reorder", [False, True])
def test_modular_kernel_overlap_wins(monkeypatch, reorder):
    """When the kernel owns the overlap, neither ordering touches the aux stream."""
    runner, log, layer = _build(monkeypatch, reorder=reorder, mk_overlap=True)
    shared = runner._shared_experts
    x = _FakeTensor(DECODE_TOKENS, log, tag="x")

    assert shared.maybe_forward_async(x) is False
    shared.maybe_sync_shared_experts_stream(x)
    assert shared(x, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED) is None
    assert shared(x, SharedExpertsOrder.NO_OVERLAP) is None
    assert log == []
    # The kernel's own order is the one that runs.
    shared(x, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
    assert log == [("shared_experts", "main")]
    assert layer.calls == 1


@pytest.mark.parametrize("reorder", [False, True])
def test_unsafe_eplb_backend_forces_the_serial_path(monkeypatch, reorder):
    """The existing overlap veto still wins over the flag."""
    runner, log, _ = _build(monkeypatch, reorder=reorder, unsafe_backend=True)
    _run(runner, log)

    assert log == _serial_log()


def test_maybe_forward_async_never_enqueues_when_reordering(monkeypatch):
    """With the flag on, nothing is submitted before the routed dispatch."""
    runner, log, _ = _build(monkeypatch, reorder=True)
    shared = runner._shared_experts

    assert shared.maybe_forward_async(_FakeTensor(DECODE_TOKENS, log)) is False
    assert log == []


# --------------------------------------------------------------------------- #
# Cudagraph capture
# --------------------------------------------------------------------------- #


def test_capture_emits_event_edges_but_no_record_stream(monkeypatch):
    """Under capture the fork and join are still emitted, record_stream is not.

    ``wait_stream`` is an event record plus an event wait, and both edges are
    drawn inside the same custom op, so the capture region is entered and left
    once. ``record_stream`` is an allocator hint with no meaning for a replay
    and is skipped. This asserts the Python branch only -- whether the driver
    accepts the resulting capture is a GPU check.
    """
    runner, log, _ = _build(monkeypatch, reorder=True, capturing=True)
    _run(runner, log)

    assert "record_stream" not in log.ops()
    assert ("wait_stream", "aux", "main") in log
    assert ("wait_stream", "main", "aux") in log
    assert log.index_of("routed_experts") < log.index_of("shared_experts")
    # Fork before the work, join after it: a well-formed pair.
    assert log.index(("wait_stream", "aux", "main")) < log.index_of("shared_experts")
    assert log.index_of("shared_experts") < log.index(("wait_stream", "main", "aux"))
