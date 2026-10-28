# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the rank-ordered torch profiler trace export
(vllm_sm80/worker/trace_export.py, VLLM_SM80_SERIAL_TRACE_EXPORT)."""

import gzip
import json
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch

from vllm.config.profiler import ProfilerConfig
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm_sm80 import plugin
from vllm_sm80.worker import trace_export

FLAG = trace_export.FLAG
KEYS = ("_stop", "step", "shutdown", "_sm80_serial_trace_export")


@pytest.fixture(autouse=True)
def restore_wrapper():
    saved = {k: TorchProfilerWrapper.__dict__[k] for k in KEYS if k in TorchProfilerWrapper.__dict__}
    yield
    for k in KEYS:
        if k in saved:
            setattr(TorchProfilerWrapper, k, saved[k])
        elif k in TorchProfilerWrapper.__dict__:
            delattr(TorchProfilerWrapper, k)


def _fake_workers(directory: Path, count: int) -> list[TorchProfilerWrapper]:
    workers = []
    for rank in range(count):
        w = object.__new__(TorchProfilerWrapper)
        w.local_rank = rank
        w._torch_profiler_trace_dir = str(directory)
        w.profiler = object()
        workers.append(w)
    return workers


def test_off_by_default_patches_nothing(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    stop = TorchProfilerWrapper._stop
    plugin.register_serial_trace_export()
    assert TorchProfilerWrapper._stop is stop
    assert "_sm80_serial_trace_export" not in TorchProfilerWrapper.__dict__


def test_workers_stop_one_at_a_time_in_engine_order(monkeypatch, tmp_path):
    """Four workers entering the stop in reverse order export 0, 1, 2, 3, each
    only after the previous one released its profiler, with temporary files in
    the trace directory; no marker is left; the next session orders again."""
    events, lock = [], threading.Lock()

    def original(self):
        with lock:
            events.append(("start", self.local_rank, tempfile.gettempdir()))
            # the previous worker released its results before this one starts
            events.append(("held", sum(hasattr(w, "profiler") for w in workers)))
        time.sleep(0.05)
        with lock:
            events.append(("end", self.local_rank))

    monkeypatch.setattr(TorchProfilerWrapper, "_stop", original)
    monkeypatch.setenv(FLAG, "1")
    plugin.register_serial_trace_export()
    assert TorchProfilerWrapper._stop is not original
    position = threading.local()
    monkeypatch.setattr(trace_export, "engine_position", lambda: (position.rank, 4))
    workers = _fake_workers(tmp_path, 4)

    def run(w):
        position.rank = w.local_rank
        w._stop()

    for session in range(2):
        events.clear()
        for w in workers:
            w.profiler = object()
        threads = [threading.Thread(target=run, args=(w,)) for w in reversed(workers)]
        for t in threads:
            t.start()
            time.sleep(0.02)
        for t in threads:
            t.join(timeout=30)
        starts = [e[1] for e in events if e[0] == "start"]
        assert starts == [0, 1, 2, 3], (session, events)
        for rank in range(4):
            i = events.index(("start", rank, str(tmp_path)))
            assert events[i + 1] == ("held", 4 - rank)
            assert events[i + 2] == ("end", rank)
        assert not any(hasattr(w, "profiler") for w in workers)
        assert not list(tmp_path.glob(".trace-export.*"))
        assert tempfile.tempdir != str(tmp_path)


def _recorded_warnings(monkeypatch) -> list[str]:
    warned: list[str] = []
    monkeypatch.setattr(trace_export.logger, "warning", lambda msg, *args: warned.append(msg % args))
    return warned


def _cpu_wrapper(directory: Path, name: str, max_iterations: int = 0) -> TorchProfilerWrapper:
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(directory),
        torch_profiler_use_gzip=True,
        torch_profiler_with_stack=False,
        max_iterations=max_iterations,
    )
    return TorchProfilerWrapper(config, worker_name=name, local_rank=1, activities=["CPU"])


def test_the_stop_profile_rpc_waits_for_the_previous_worker_however_long(monkeypatch, tmp_path):
    """The worker's profile RPC stops through stop(): the second worker of an
    engine waits for the first one's marker with no time limit, warning while
    it waits, then exports, releases its profiler and removes the marker."""
    monkeypatch.setenv(FLAG, "1")
    plugin.register_serial_trace_export()
    assert not hasattr(trace_export, "WAIT_S")
    warned = _recorded_warnings(monkeypatch)
    monkeypatch.setattr(trace_export, "LOG_S", 0.1)
    monkeypatch.setattr(trace_export, "engine_position", lambda: (1, 2))
    wrapper = _cpu_wrapper(tmp_path, "rank1")
    stopped = threading.Event()

    def rpcs() -> None:  # the worker's start and stop RPCs run on its one busy-loop thread
        wrapper.start()
        torch.relu(torch.randn(16, 16)).sum()
        wrapper.stop()
        stopped.set()

    thread = threading.Thread(target=rpcs, daemon=True)
    thread.start()
    time.sleep(0.5)
    assert not stopped.is_set() and not list(tmp_path.glob("rank1.*.pt.trace.json.gz"))
    assert any(w.startswith("Serial trace export: waiting") for w in warned)
    first = trace_export.marker(str(tmp_path), 0, 0)
    first.touch()
    thread.join(timeout=60)
    assert stopped.is_set() and len(list(tmp_path.glob("rank1.*.pt.trace.json.gz"))) == 1
    assert not hasattr(wrapper, "profiler") and not first.exists()
    assert not list(tmp_path.glob(".trace-export.*"))


def test_a_stop_after_max_iterations_inside_a_model_step_is_not_ordered(monkeypatch, tmp_path):
    """The profiler stopping itself at its iteration limit stops inside the
    model step: the second worker exports at once without waiting for the
    first, nothing is released or marked, and a warning says why."""
    monkeypatch.setenv(FLAG, "1")
    plugin.register_serial_trace_export()
    warned = _recorded_warnings(monkeypatch)
    monkeypatch.setattr(trace_export, "engine_position", lambda: (1, 2))
    wrapper = _cpu_wrapper(tmp_path, "rank1", max_iterations=2)
    done = threading.Event()

    def steps() -> None:  # the start RPC, then model steps, on the worker's one thread
        wrapper.start()
        for _ in range(3):
            wrapper.step()
            torch.relu(torch.randn(16, 16)).sum()
        done.set()

    thread = threading.Thread(target=steps, daemon=True)
    thread.start()
    thread.join(timeout=60)
    assert done.is_set() and not wrapper.is_running
    assert len(list(tmp_path.glob("rank1.*.pt.trace.json.gz"))) == 1
    assert hasattr(wrapper, "profiler") and not list(tmp_path.glob(".trace-export.*"))
    assert any("inside a model step" in w and "not ordered" in w for w in warned)
    assert wrapper._sm80_stop_path is None


def test_real_session_writes_trace_releases_and_keeps_tmp_on_disk(monkeypatch, tmp_path):
    """A real CPU session outside a model-parallel worker (no ordering): the
    gzipped trace is written, torch's uncompressed temporary goes into the
    trace directory and is removed, the profiler is released, and a second
    session starts a new one."""
    monkeypatch.setenv(FLAG, "1")
    plugin.register_serial_trace_export()
    temp_dirs = []
    named = tempfile.NamedTemporaryFile

    def recording(*args, **kwargs):
        temp_dirs.append(tempfile.gettempdir())
        return named(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", recording)
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_use_gzip=True,
        torch_profiler_record_shapes=True,
        torch_profiler_with_stack=False,
    )
    wrapper = TorchProfilerWrapper(config, worker_name="rank0", local_rank=0, activities=["CPU"])
    for _ in range(2):
        wrapper.start()
        a = torch.randn(16, 16)
        torch.relu(a @ a).sum()
        wrapper.stop()
        assert not hasattr(wrapper, "profiler")
    traces = sorted(tmp_path.glob("rank0.*.pt.trace.json.gz"))
    assert len(traces) == 2
    with gzip.open(traces[0]) as f:
        names = {e.get("name") for e in json.load(f)["traceEvents"]}
    assert "aten::relu" in names
    assert temp_dirs and set(temp_dirs) == {str(tmp_path)}
    left = [p.name for p in tmp_path.iterdir() if not p.name.endswith((".pt.trace.json.gz", ".profiler_out.txt"))]
    assert left == []
