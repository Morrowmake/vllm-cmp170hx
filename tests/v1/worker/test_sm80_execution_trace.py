# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the execution trace beside the torch profiler trace
(vllm_sm80/worker/execution_trace.py, VLLM_SM80_EXECUTION_TRACE): a real
TorchProfilerWrapper session on CPU activity only."""

import gzip
import json
from pathlib import Path

import pytest
import torch

from vllm.config.profiler import ProfilerConfig
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm_sm80 import plugin
from vllm_sm80.worker import execution_trace

FLAG = execution_trace.FLAG


@pytest.fixture(autouse=True)
def restore_wrapper():
    saved = {
        k: TorchProfilerWrapper.__dict__[k]
        for k in ("_start", "_stop", "_sm80_execution_trace")
        if k in TorchProfilerWrapper.__dict__
    }
    yield
    for k in ("_start", "_stop", "_sm80_execution_trace"):
        if k in saved:
            setattr(TorchProfilerWrapper, k, saved[k])
        elif k in TorchProfilerWrapper.__dict__:
            delattr(TorchProfilerWrapper, k)


def _session(directory: Path, gzip_trace: bool) -> None:
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(directory),
        torch_profiler_use_gzip=gzip_trace,
        torch_profiler_record_shapes=True,
        torch_profiler_with_stack=False,
    )
    wrapper = TorchProfilerWrapper(
        config, worker_name="rank0", local_rank=0, activities=["CPU"]
    )
    wrapper.start()
    a = torch.randn(16, 16)
    b = torch.relu(a @ a)
    (b + a).sum()
    wrapper.stop()


def _load(path: Path):
    with gzip.open(path) if path.suffix == ".gz" else open(path) as f:
        return json.load(f)


def _outputs(directory: Path) -> list[Path]:
    """Files the session wrote, less the CPU-time table."""
    return sorted(
        p for p in directory.iterdir() if not p.name.endswith(".profiler_out.txt")
    )


def test_off_by_default_patches_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(FLAG, raising=False)
    start = TorchProfilerWrapper._start
    plugin.register_execution_trace()
    assert TorchProfilerWrapper._start is start
    assert "_sm80_execution_trace" not in TorchProfilerWrapper.__dict__
    _session(tmp_path, gzip_trace=True)
    names = [p.name for p in _outputs(tmp_path)]
    assert len(names) == 1 and names[0].endswith(".pt.trace.json.gz"), names


@pytest.mark.parametrize("gzip_trace", [True, False])
def test_on_writes_execution_trace_with_trace_stem(monkeypatch, tmp_path, gzip_trace):
    monkeypatch.setenv(FLAG, "1")
    plugin.register_execution_trace()
    _session(tmp_path, gzip_trace)
    traces = [p for p in tmp_path.iterdir() if ".pt.trace.json" in p.name]
    assert len(traces) == 1
    kineto = traces[0]
    if gzip_trace:
        stem = kineto.name[: -len(".pt.trace.json.gz")]
        expected = kineto.with_name(stem + ".et.json.gz")
    else:
        expected = kineto.with_name(kineto.stem + ".et.json")
    assert _outputs(tmp_path) == sorted([kineto, expected])

    et = _load(expected)
    events = _load(kineto)["traceEvents"]
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    assert et["pid"] in {e["pid"] for e in cpu_ops}
    rf_ids = {
        a["value"]
        for n in et["nodes"]
        for a in n.get("attrs") or ()
        if a.get("name") == "rf_id" and a["value"]
    }
    kineto_rf = {e["args"].get("Record function id") for e in cpu_ops}
    assert rf_ids & kineto_rf
    names = {n["name"] for n in et["nodes"]}
    assert "aten::relu" in names


def test_on_stop_before_trace_ready_leaves_no_part(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    plugin.register_execution_trace()
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        wait_iterations=2,
        active_iterations=1,
    )
    wrapper = TorchProfilerWrapper(
        config, worker_name="rank0", local_rank=0, activities=["CPU"]
    )
    wrapper.start()
    torch.randn(4, 4).sum()
    wrapper.stop()
    assert not [p for p in tmp_path.iterdir() if ".et.json" in p.name]
