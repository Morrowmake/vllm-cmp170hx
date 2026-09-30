# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-adaptive DFlash draft depth (VLLM_GLM5_DFLASH_ADAPTIVE_K)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import CompilationConfig, CUDAGraphMode, ParallelConfig
from vllm.config import SchedulerConfig, VllmConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.adaptive_k import AdaptiveKConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu import pp_draft_tail as dt

FLAG = "VLLM_GLM5_DFLASH_ADAPTIVE_K"
DEPTHS = "VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS"
LOG = "VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG"
BY_LOAD = {"by_load": [5, 4, 3]}


# ------------------------------------------------------------------ env flags


def test_flags_declared_default_off(monkeypatch):
    import vllm.envs as envs

    for name in (FLAG, DEPTHS, LOG):
        monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_K is False
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS == "5,4"
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG == 0
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DEPTHS, "6,4")
    monkeypatch.setenv(LOG, "500")
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_K is True
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS == "6,4"
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG == 500


# ---------------------------------------------------------- config rewriting


def _spec(**overrides) -> SimpleNamespace:
    """The attributes the rewrite reads, as a DFlash k=3 server sets them."""
    attrs = dict(
        method="dflash",
        num_speculative_tokens=3,
        adaptive_k=None,
        num_speculative_tokens_per_batch_size=None,
        enable_adaptive_verification=False,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _rewrite(spec: SimpleNamespace) -> SimpleNamespace:
    SpeculativeConfig._maybe_enable_glm5_load_adaptive_depth(spec)
    return spec


def test_flag_off_leaves_the_config_untouched(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    spec = _spec()
    before = dict(vars(spec))
    assert vars(_rewrite(spec)) == before


def test_flag_on_drafts_the_deepest_block_and_verifies_by_load(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv(DEPTHS, raising=False)
    monkeypatch.delenv(LOG, raising=False)
    spec = _rewrite(_spec())
    assert spec.num_speculative_tokens == 5
    assert spec.adaptive_k == {"by_load": [5, 4, 3], "log_interval": 0}
    config = AdaptiveKConfig.from_dict(spec.adaptive_k, spec.num_speculative_tokens)
    assert config.allowed == (3, 4, 5)
    assert [config.k_for_load(n) for n in range(1, 9)] == [5, 4, 3, 3, 3, 3, 3, 3]


def test_depths_and_log_interval_are_tunable(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DEPTHS, " 7, 5,4 ")
    monkeypatch.setenv(LOG, "1000")
    spec = _rewrite(_spec())
    assert spec.num_speculative_tokens == 7
    assert spec.adaptive_k == {"by_load": [7, 5, 4, 3], "log_interval": 1000}


def test_rewrite_is_idempotent(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv(DEPTHS, raising=False)
    spec = _rewrite(_spec())
    once = dict(vars(spec))
    assert vars(_rewrite(spec)) == once


@pytest.mark.parametrize(
    "overrides,depths,needle",
    [
        (dict(method="mtp"), "5,4", "needs the DFlash drafter"),
        (dict(adaptive_k={"min": 1}), "5,4", "adaptive_k is already configured"),
        (
            dict(num_speculative_tokens_per_batch_size=[(1, 8, 3)]),
            "5,4",
            "num_speculative_tokens_per_batch_size",
        ),
        (dict(enable_adaptive_verification=True), "5,4", "adaptive_verification"),
        (dict(num_speculative_tokens=None), "5,4", "not set"),
        (dict(), "3,2", "exceeds num_speculative_tokens=3"),
        (dict(), "five", "is not a list"),
        (dict(), "0,5", "empty or < 1"),
    ],
)
def test_gate_closed_leaves_the_config_and_says_why(
    monkeypatch, caplog_vllm, overrides, depths, needle
):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DEPTHS, depths)
    spec = _spec(**overrides)
    before = dict(vars(spec))
    # warning_once dedupes on its arguments; clear it so every case logs.
    _clear_once_caches()
    assert vars(_rewrite(spec)) == before
    messages = [r.getMessage() for r in caplog_vllm.records]
    assert any(
        "load-adaptive DFlash depth is off" in m and needle in m for m in messages
    ), messages


def _clear_once_caches() -> None:
    import vllm.logger as vllm_logger_module

    for name in ("_print_warning_once", "_print_info_once", "_print_debug_once"):
        fn = getattr(vllm_logger_module, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


# ------------------------------------------------------------------ the policy


def test_by_load_config_names_its_own_counts():
    config = AdaptiveKConfig.from_dict(BY_LOAD, 5)
    assert config.load_mode
    assert (config.min_k, config.max_k, config.allowed) == (3, 5, (3, 4, 5))
    assert config.k_for_load(0) == 5  # an empty server counts as one request
    assert config.k_for_load(1) == 5
    assert config.k_for_load(2) == 4
    assert config.k_for_load(3) == 3
    assert config.k_for_load(100) == 3


def test_by_load_graph_ceiling_per_count():
    config = AdaptiveKConfig.from_dict(BY_LOAD, 5)
    assert config.max_reqs_for(5, 8) == 1
    assert config.max_reqs_for(4, 8) == 2
    assert config.max_reqs_for(3, 8) == 8
    assert config.max_reqs_for(2, 8) == 0
    acceptance = AdaptiveKConfig.from_dict({"min": 1, "max": 5}, 5)
    assert not acceptance.load_mode
    assert acceptance.max_reqs_for(5, 8) == 8


@pytest.mark.parametrize(
    "raw,needle",
    [
        ({"by_load": []}, "non-empty list"),
        ({"by_load": [6, 3]}, "outside"),
        ({"by_load": [5, 3], "allowed": [3, 5]}, "drop"),
        ({"by_load": [5, 3], "max": 5}, "drop"),
    ],
)
def test_by_load_config_rejects_bad_input(raw, needle):
    with pytest.raises(ValueError, match=needle):
        AdaptiveKConfig.from_dict(raw, 5)


# ------------------------------------------------------------ scheduler wiring


def _make_scheduler(adaptive_k: dict | None, num_speculative_tokens: int = 5,
                    cls=Scheduler):
    base = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        num_speculative_tokens=num_speculative_tokens,
    )
    speculative_config = base.vllm_config.speculative_config
    assert speculative_config is not None
    if adaptive_k is not None:
        speculative_config.adaptive_k = adaptive_k
        speculative_config.adaptive_k_config = AdaptiveKConfig.from_dict(
            adaptive_k, num_speculative_tokens
        )
    return cls(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def _step_with_drafts(scheduler, requests):
    """One scheduled step in which every request holds a full draft block."""
    for request in requests:
        request.spec_token_ids = [11, 12, 13, 14, 15]
    return scheduler.schedule()


@pytest.mark.parametrize("num_requests,k", [(1, 5), (2, 4), (3, 3), (6, 3)])
def test_scheduler_verifies_by_load_and_always_drafts_the_full_block(
    num_requests, k
):
    scheduler = _make_scheduler(BY_LOAD)
    requests = create_requests(num_requests=num_requests)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()  # prefill

    output = _step_with_drafts(scheduler, requests)
    assert scheduler.cur_num_spec_tokens == k
    # The drafter is asked for its whole block regardless of the load.
    assert output.num_spec_tokens_to_schedule == 5
    for request in requests:
        # Verification takes the first k drafts of the block.
        assert output.scheduled_spec_decode_tokens[request.request_id] == [
            11, 12, 13, 14, 15
        ][:k]


def test_waiting_requests_count_as_load():
    scheduler = _make_scheduler(BY_LOAD)
    requests = create_requests(num_requests=1)
    scheduler.add_request(requests[0])
    scheduler.schedule()
    requests[0].spec_token_ids = [11, 12, 13, 14, 15]
    assert scheduler._select_adaptive_k() == (5, 5)
    # One more request queued (not yet admitted) already counts as load.
    late = create_requests(num_requests=2, req_ids=["w0", "w1"])
    scheduler.waiting.add_request(late[0])
    requests[0].spec_token_ids = [11, 12, 13, 14, 15]
    assert scheduler._select_adaptive_k() == (4, 5)
    scheduler.skipped_waiting.add_request(late[1])
    requests[0].spec_token_ids = [11, 12, 13, 14, 15]
    assert scheduler._select_adaptive_k() == (3, 5)


def test_depth_climbs_back_as_soon_as_load_drops():
    scheduler = _make_scheduler(BY_LOAD)
    requests = create_requests(num_requests=3)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()
    _step_with_drafts(scheduler, requests)
    assert scheduler.cur_num_spec_tokens == 3
    # Two requests leave; the survivor still holds a full block, so the next
    # step verifies all five instead of climbing one notch at a time.
    for request in requests[1:]:
        scheduler.running.remove(request)
    _step_with_drafts(scheduler, requests[:1])
    assert scheduler.cur_num_spec_tokens == 5


def test_short_draft_list_caps_the_step_for_everyone():
    scheduler = _make_scheduler(BY_LOAD)
    requests = create_requests(num_requests=1)
    scheduler.add_request(requests[0])
    scheduler.schedule()
    requests[0].spec_token_ids = [11, 12, 13, 14]
    scheduler.schedule()
    # Load asks for 5, the request carries 4: verify the widest captured
    # count it can fill.
    assert scheduler.cur_num_spec_tokens == 4


def test_async_scheduler_placeholders_are_full_width():
    scheduler = _make_scheduler(BY_LOAD, cls=AsyncScheduler)
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == 5
    output = scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 4
    assert output.num_spec_tokens_to_schedule == 5
    for request in requests:
        assert len(output.scheduled_spec_decode_tokens[request.request_id]) == 4


def test_scheduler_off_path_is_fixed_depth():
    scheduler = _make_scheduler(None, num_speculative_tokens=3)
    assert scheduler.adaptive_k is None
    requests = create_requests(num_requests=1)
    scheduler.add_request(requests[0])
    scheduler.schedule()
    requests[0].spec_token_ids = [11, 12, 13]
    output = scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 3
    assert output.num_spec_tokens_to_schedule == 3
    assert output.scheduled_spec_decode_tokens[requests[0].request_id] == [
        11, 12, 13
    ]


def test_banner_goes_through_the_real_logger(caplog_vllm):
    _clear_once_caches()
    _make_scheduler(BY_LOAD)
    messages = [r.getMessage() for r in caplog_vllm.records]
    banner = [m for m in messages if "load-adaptive DFlash depth active" in m]
    assert banner == [
        "GLM-5 load-adaptive DFlash depth active (VLLM_GLM5_DFLASH_ADAPTIVE_K): "
        "drafts 5 per step, verifies 5 at 1, 4 at 2, 3 at >= 3 requests"
    ], messages
    # The acceptance-mode banner stays silent in load mode.
    assert not any("Acceptance-adaptive" in m for m in messages)


# ----------------------------------------------------------- CUDA-graph sizes


def _graph_manager(monkeypatch, adaptive_k: dict, num_speculative_tokens: int,
                   max_num_seqs: int, capture_sizes: list[int]):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=capture_sizes,
    )
    compilation_config.max_cudagraph_capture_size = max(capture_sizes)
    compilation_config.post_init_cudagraph_sizes()
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.num_speculative_tokens = num_speculative_tokens
    config = AdaptiveKConfig.from_dict(adaptive_k, num_speculative_tokens)
    spec = MagicMock()
    spec.uses_dynamic_speculative_decoding.return_value = False
    spec.num_speculative_tokens_per_batch_size = None
    spec.uses_adaptive_k.return_value = True
    spec.adaptive_k_draft_counts.return_value = config.allowed
    spec.adaptive_k_config = config
    vllm_config.speculative_config = spec
    return gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=num_speculative_tokens + 1,
        adaptive_k_capture=True,
    )


def _full_decode_shapes(manager):
    return sorted(
        (d.uniform_token_count, d.num_reqs)
        for d in manager._capture_descs[CUDAGraphMode.FULL]
    )


# The capture sizes _set_cudagraph_sizes produces for max_num_seqs=8 with the
# load tiers: the k=3 grid up to 64 plus 5, 6 and 10.
LOAD_SIZES = [1, 2, 4, 5, 6, 8, 10, 16, 24, 32, 40, 48, 56, 64]


def test_load_mode_captures_each_depth_only_where_it_runs(monkeypatch):
    manager = _graph_manager(monkeypatch, BY_LOAD, 5, 8, LOAD_SIZES)
    shapes = _full_decode_shapes(manager)
    assert [s for s in shapes if s[0] == 6] == [(6, 1)]
    assert [s for s in shapes if s[0] == 5] == [(5, 1), (5, 2)]
    k3 = [n for q, n in shapes if q == 4]
    # k=3 covers every batch size up to max_num_seqs (padded to a graph).
    assert max(k3) == 8
    manager._graphs_captured = True
    for query_len, num_reqs in [(6, 1), (5, 1), (5, 2)] + [(4, n) for n in range(1, 9)]:
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs * query_len,
            uniform_token_count=query_len,
            num_active_loras=0,
        )
        # Dispatch pads up to a captured graph of the same query length.
        assert desc.cg_mode == CUDAGraphMode.FULL, (query_len, num_reqs)
        assert desc.uniform_token_count == query_len


def test_acceptance_mode_still_captures_every_count_everywhere(monkeypatch):
    manager = _graph_manager(
        monkeypatch, {"allowed": [3, 4, 5]}, 5, 8, list(range(1, 49))
    )
    shapes = _full_decode_shapes(manager)
    for query_len in (4, 5, 6):
        assert max(n for q, n in shapes if q == query_len) == 8


def test_set_cudagraph_sizes_load_tiers(monkeypatch):
    """The capture-size list a load-mode server gets: the configured depth's
    grid (ceiling from k=3, not k=5) plus the deep tiers' decode sizes."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        pytest.skip("capture sizes are computed for CUDA-like platforms only")
    base = create_scheduler(
        max_num_seqs=8, max_num_batched_tokens=3460, num_speculative_tokens=5
    )
    vllm_config = base.vllm_config
    spec = vllm_config.speculative_config
    spec.adaptive_k = dict(BY_LOAD)
    spec.adaptive_k_config = AdaptiveKConfig.from_dict(BY_LOAD, 5)
    comp = vllm_config.compilation_config
    comp.cudagraph_capture_sizes = None
    comp.max_cudagraph_capture_size = None
    if vllm_config.model_config.enforce_eager or comp.cudagraph_mode == (
        CUDAGraphMode.NONE
    ):
        pytest.skip("test model config runs eager")
    vllm_config._set_cudagraph_sizes()
    assert comp.max_cudagraph_capture_size == 64
    assert comp.cudagraph_capture_sizes == LOAD_SIZES


# ------------------------------------------------------------ PP draft tail


def _tail_config(adaptive_k_config):
    spec = SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        uses_adaptive_k=lambda: adaptive_k_config is not None,
        adaptive_k_config=adaptive_k_config,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=4,
            tensor_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        use_v2_model_runner=True,
        speculative_config=spec,
    )


def test_draft_tail_carries_load_mode_but_not_acceptance_mode():
    load = dt.draft_tail_gate(_tail_config(AdaptiveKConfig.from_dict(BY_LOAD, 5)), 2)
    assert load.enabled and load.stage == 2
    acceptance = dt.draft_tail_gate(
        _tail_config(AdaptiveKConfig.from_dict({"min": 1, "max": 5}, 5)), 2
    )
    assert not acceptance.enabled
    assert "variable draft counts" in acceptance.reason
    assert dt.draft_tail_gate(_tail_config(None), 2).enabled
