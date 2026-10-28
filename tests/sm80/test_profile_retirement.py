# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership and recapture contracts for profiling graph retirement."""

import gc
import importlib
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.spec_decode.test_dspark_width import _config, _init_cpu
from tests.v1.worker.test_gpu_model_runner_v2_cudagraph_profiling import (
    GLOBAL_POOL,
    THROWAWAY_POOL,
    _make_profiling_runner,
    _patch_module,
)
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm_sm80 import capture
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm_sm80.spec_decode.dflash_width import DraftWidthMixin
from vllm_sm80.spec_decode.dspark_width import Sm80DSparkSpeculator


class _WidthDFlashSpeculator(DraftWidthMixin, DFlashSpeculator):
    pass


pytestmark = pytest.mark.cpu_test


class _Graph:
    pass


def test_gpu_regression_has_reimportable_spawn_identity():
    module = importlib.import_module(
        "tests.v1.worker.test_sm80_profile_retirement_gpu")
    test = module.test_profile_registry_releases_pool_then_recaptures
    assert test.__module__ == module.__name__
    # Match the existing spawn helper's module/qualified-name lookup without
    # executing the GPU body or changing process isolation.
    target = importlib.import_module(test.__module__)
    for name in test.__qualname__.split("."):
        target = getattr(target, name)
    assert target is test


@pytest.mark.parametrize(
    "speculator_cls", [_WidthDFlashSpeculator, Sm80DSparkSpeculator]
)
@pytest.mark.parametrize("outcome", ["capture", "capture_error", "no_graphs"])
def test_real_width_managers_retire_and_reinitialize(monkeypatch, speculator_cls, outcome):
    _patch_module(monkeypatch)
    monkeypatch.setattr(capture, "_pending", [])
    monkeypatch.setattr(cgu, "get_pp_group", lambda: SimpleNamespace(
        is_first_rank=True, is_last_rank=True))
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL_AND_PIECEWISE, needs_capture=outcome != "no_graphs",
    )
    config = MagicMock(spec=VllmConfig)
    config.scheduler_config = SimpleNamespace(max_num_seqs=8)
    config.parallel_config = SimpleNamespace(tensor_parallel_size=4)
    config.model_config = SimpleNamespace(max_model_len=128)
    config.speculative_config = _config().speculative_config
    config.num_speculative_tokens = 5
    config.compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[3, 4, 5, 6, 8, 12, 15, 18, 20, 24, 30, 32, 40],
    )
    config.compilation_config.max_cudagraph_capture_size = 40
    speculator = speculator_cls.__new__(speculator_cls)
    _init_cpu(speculator, config, torch.device("cpu"))
    speculator.enable_draft_widths((3, 5))
    speculator.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH)
    speculator.attn_groups = speculator.kv_cache_config = None
    speculator._group_causal = False
    runner.speculator = speculator
    profile_refs = []
    captured = []
    expected_pool = THROWAWAY_POOL

    def initialize(r):
        r.events.append("init")
        # Use the real speculator initialization, not an imitation registry.
        speculator.block_tables = torch.arange(8)
        speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
        profile_refs.extend(weakref.ref(m) for m in speculator._cg_managers.values())

    def capture_graph(manager, forward, input_buffers, block_tables, *args, **kwargs):
        assert manager.pool == expected_pool
        assert block_tables is speculator.block_tables
        record = capture.Record("attention", "lifecycle", "layer", 1, {}, in_graph=True)
        record.put(snapshot=block_tables)
        record.done()
        graph = _Graph()
        manager.graphs = capture._BindingGraphs()
        manager.graphs[speculator.width] = graph
        manager._graphs_captured = True
        captured.append((expected_pool, speculator.width))
        if expected_pool == THROWAWAY_POOL:
            profile_refs.extend(weakref.ref(x) for x in (
                graph, record, record.tensors["snapshot"]))

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", initialize)
    monkeypatch.setattr(DFlashCudaGraphManager, "capture", capture_graph)
    original_capture = runner.capture_model

    def capture_model(*, profile_only=False):
        # As the fork runner's capture_model: the per-width managers are
        # dropped once the profiling capture returns or fails.
        assert profile_only
        try:
            speculator.capture()
            assert not capture._pending
            if outcome == "capture_error":
                raise RuntimeError("capture failed")
            return original_capture(profile_only=profile_only)
        finally:
            speculator.retire_width_managers()

    def teardown(r):
        r.events.append("teardown")
        gc.collect()
        assert speculator.query_cudagraph_manager is None
        if outcome == "no_graphs":
            # Nothing was captured, so nothing holds the profiling pool; the
            # real initialization replaces the managers.
            return
        assert speculator._cg_managers == {}
        assert all(ref() is None for ref in profile_refs)

    runner.capture_model = capture_model
    monkeypatch.setattr(cgu, "_teardown_profiling_state", teardown)
    if outcome == "capture_error":
        with pytest.raises(RuntimeError, match="capture failed"):
            cgu.profile_cudagraph_memory(runner)
    else:
        cgu.profile_cudagraph_memory(runner)
    assert cgu.current_platform.get_global_graph_pool() == GLOBAL_POOL
    assert captured == ([] if outcome == "no_graphs" else [
        (THROWAWAY_POOL, 3), (THROWAWAY_POOL, 5)])

    # The production initializer rebuilds managers and the active alias. The
    # production capture dispatch sees the new KV state, not profiling state.
    expected_pool = GLOBAL_POOL
    speculator.block_tables = torch.arange(8) + 100
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert speculator.query_cudagraph_manager is speculator._cg_managers[5]
    assert all(not m._graphs_captured for m in speculator._cg_managers.values())
    speculator.capture()
    assert captured[-2:] == [(GLOBAL_POOL, 3), (GLOBAL_POOL, 5)]
    for width, manager in speculator._cg_managers.items():
        assert manager.pool == GLOBAL_POOL
        assert manager._graphs_captured
        record = manager.graphs[width]._sm80_capture_records[0]
        torch.testing.assert_close(record.tensors["snapshot"], speculator.block_tables)
    assert not capture._pending

    # A disabled profiling path does not touch live real-capture managers.
    live = dict(speculator._cg_managers)
    runner.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    # Profiling is off only when no graph (the drafter's included) still needs
    # capturing; the live managers here are already captured.
    runner.needs_cudagraph_capture = lambda: False
    assert cgu.profile_cudagraph_memory(runner) == 0
    assert speculator._cg_managers == live
    assert all(m.graphs for m in live.values())
    assert not torch.cuda.is_initialized()
