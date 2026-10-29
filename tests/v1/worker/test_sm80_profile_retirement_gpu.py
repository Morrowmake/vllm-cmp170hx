# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-device allocator/replay proof; never part of the CPU check batch."""

import contextlib
import gc
from types import SimpleNamespace

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from tests.v1.worker.test_gpu_model_runner_v2_cudagraph_profiling import (
    _make_profiling_runner,
)
from vllm.config import CUDAGraphMode
from vllm_sm80 import capture
from vllm.v1.worker.gpu import cudagraph_utils as cgu


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_profile_registry_releases_pool_then_recaptures(monkeypatch):
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.compilation_config.static_forward_context = {}
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.kv_caches = []
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace()
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None
    monkeypatch.setattr(capture, "_pending", [])
    monkeypatch.setattr(cgu, "set_current_vllm_config", lambda _: contextlib.nullcontext())
    elements = (16 << 20) // 4
    source = torch.ones(elements, device="cuda")
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    before = torch.accelerator.memory_reserved()
    memory = {}

    def initialize(r):
        r.speculator = SimpleNamespace(query_cudagraph_manager=None, _cg_managers={})
        for width in (3, 5):
            manager = cgu.CudaGraphManager.__new__(cgu.CudaGraphManager)
            manager.pool = cgu.current_platform.get_global_graph_pool()
            manager.graphs = capture._BindingGraphs()
            r.speculator._cg_managers[width] = manager
        r.speculator.query_cudagraph_manager = r.speculator._cg_managers[5]

    def capture_one(manager, key):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=manager.pool):
            record = capture.Record("attention", "lifecycle", "layer", 1, {}, in_graph=True)
            record.put(snapshot=source * 2)
            record.done()
        manager.graphs[key] = graph

    def capture_model(*, profile_only=False):
        assert profile_only
        runner.cudagraph_manager.graphs = capture._BindingGraphs()
        capture_one(runner.cudagraph_manager, "target")
        for width, manager in runner.speculator._cg_managers.items():
            capture_one(manager, width)
        torch.accelerator.synchronize()
        assert not capture._pending
        memory["captured"] = torch.accelerator.memory_reserved()
        return memory["captured"] - before

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", initialize)
    runner.capture_model = capture_model
    cgu.profile_cudagraph_memory(runner)
    after = torch.accelerator.memory_reserved()
    assert memory["captured"] > before
    assert runner.speculator.query_cudagraph_manager is None
    assert runner.speculator._cg_managers == {}
    assert after == before

    # New graph handles bind the restored persistent pool. Their real replay
    # must refresh graph-owned snapshots from new input rather than stale state.
    initialize(runner)
    for width, manager in runner.speculator._cg_managers.items():
        capture_one(manager, width)
    source.fill_(7)
    for width, manager in runner.speculator._cg_managers.items():
        graph = manager.graphs[width]
        graph.replay()
        torch.accelerator.synchronize()
        observed = graph._sm80_capture_records[0].materialize()["snapshot"]
        torch.testing.assert_close(observed, torch.full_like(observed, 14))
    assert not capture._pending


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("preserve_layout", [False, True])
def test_graph_snapshots_do_not_retain_device_payload(preserve_layout):
    backing = torch.zeros((128, 4096), device="cuda")
    source = backing[:, :2048]
    source.add_(0)
    torch.cuda.synchronize()

    def reserved(graph):
        pool = tuple(graph.pool())
        return sum(
            int(segment["total_size"])
            for segment in torch.cuda.memory_snapshot(include_traces=False)
            if tuple(segment.get("segment_pool_id", ())) == pool
        )

    control = torch.cuda.CUDAGraph()
    with torch.cuda.graph(control):
        source.add_(1000)
    torch.cuda.synchronize()
    control_bytes = reserved(control)
    graph = torch.cuda.CUDAGraph()
    record = capture.Record("attention", "snapshot", "layer", 128, {}, in_graph=True)
    with torch.cuda.graph(graph):
        record.put(preserve_layout=preserve_layout, snapshot=source)
        source.add_(1000)
    torch.cuda.synchronize()
    # Count the live graph pool, including freed packing temporaries still reserved.
    assert reserved(graph) - control_bytes < source.numel() * source.element_size()
    saved = []
    for value in (3, 17, -9):
        source.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        observed = record.materialize()["snapshot"]
        torch.testing.assert_close(observed, torch.full(source.shape, float(value)), rtol=0, atol=0)
        torch.testing.assert_close(source.cpu(), torch.full(source.shape, float(value + 1000)), rtol=0, atol=0)
        if preserve_layout:
            assert observed.stride() == source.stride()
            span = (source.shape[0] - 1) * source.stride(0) + source.shape[1]
            assert observed.untyped_storage().nbytes() == span * source.element_size()
        saved.append((value, observed))
    # A later replay cannot overwrite a previously materialized export.
    for value, observed in saved:
        torch.testing.assert_close(observed, torch.full(source.shape, float(value)), rtol=0, atol=0)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_eager_snapshot_avoids_device_payload_and_preserves_prewrite_values():
    backing = torch.full((128, 4096), 3.0, device="cuda")
    source = backing[:, :2048]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    record = capture.Record("attention", "snapshot", "layer", 128, {})
    record.put(snapshot=source)
    observed = record.materialize()["snapshot"]
    source.add_(1000)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < source.numel() * source.element_size()
    torch.testing.assert_close(observed, torch.full(source.shape, 3.0), rtol=0, atol=0)
    torch.testing.assert_close(source.cpu(), torch.full(source.shape, 1003.0), rtol=0, atol=0)
    record.put(snapshot=source)
    torch.testing.assert_close(record.materialize()["snapshot"], torch.full(source.shape, 1003.0), rtol=0, atol=0)
    torch.testing.assert_close(observed, torch.full(source.shape, 3.0), rtol=0, atol=0)
