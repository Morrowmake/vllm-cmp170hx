# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-adaptive DFlash draft depth (``adaptive_k.by_load``)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import CompilationConfig, CUDAGraphMode, ParallelConfig
from vllm.config import SchedulerConfig, VllmConfig
from vllm_sm80.sched.scheduler import Sm80AsyncScheduler, Sm80Scheduler
from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm_sm80.worker import pp_draft_tail as dt

BY_LOAD = {"by_load": [5, 4, 3]}


def _clear_once_caches() -> None:
    import vllm.logger as vllm_logger_module

    for name in ("_print_warning_once", "_print_info_once", "_print_debug_once"):
        fn = getattr(vllm_logger_module, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


# ------------------------------------------------------------------ the policy


def test_by_load_config_names_its_own_counts():
    config = AdaptiveKConfig.from_dict(BY_LOAD, 5)
    assert (config.by_load, config.allowed) == ((5, 4, 3), (3, 4, 5))
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


@pytest.mark.parametrize(
    "raw,needle",
    [
        ({"by_load": []}, "non-empty list"),
        ({"by_load": [6, 3]}, "outside"),
        ({"by_load": [5, 3], "allowed": [3, 5]}, "Unknown adaptive_k keys"),
        ({"by_load": [5, 3], "max": 5}, "Unknown adaptive_k keys"),
    ],
)
def test_by_load_config_rejects_bad_input(raw, needle):
    with pytest.raises(ValueError, match=needle):
        AdaptiveKConfig.from_dict(raw, 5)


# ------------------------------------------------------------ scheduler wiring


def _make_scheduler(adaptive_k: dict | None, num_speculative_tokens: int = 5,
                    cls=Sm80Scheduler):
    base = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        num_speculative_tokens=num_speculative_tokens,
    )
    speculative_config = base.vllm_config.speculative_config
    assert speculative_config is not None
    if adaptive_k is not None:
        base.vllm_config.additional_config = {"sm80": {"adaptive_k": adaptive_k}}
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
    scheduler.kv_holding_waiting.add_request(late[1])
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
    scheduler = _make_scheduler(BY_LOAD, cls=Sm80AsyncScheduler)
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
        "GLM-5 load-adaptive DFlash depth active (adaptive_k.by_load): "
        "drafts 5 per step, verifies 5 at 1, 4 at 2, 3 at >= 3 requests"
    ], messages


# ----------------------------------------------------------- CUDA-graph sizes


def _graph_manager(
    monkeypatch,
    adaptive_k: dict,
    num_speculative_tokens: int,
    max_num_seqs: int,
    capture_sizes: list[int],
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_AND_PIECEWISE,
):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    compilation_config = CompilationConfig(
        cudagraph_mode=cudagraph_mode.name,
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
    vllm_config.additional_config = {"sm80": {"adaptive_k": adaptive_k}}
    spec = MagicMock()
    spec.uses_dynamic_speculative_decoding.return_value = False
    spec.num_speculative_tokens_per_batch_size = None
    spec.num_speculative_tokens = num_speculative_tokens
    vllm_config.speculative_config = spec
    return gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=cudagraph_mode,
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


def _dispatch_uniform(manager, num_reqs: int, query_len: int):
    return manager.dispatch(
        num_reqs=num_reqs,
        num_tokens=num_reqs * query_len,
        uniform_token_count=query_len,
        num_active_loras=0,
        max_query_len=query_len,
    )


# An explicit capture grid without the load tiers' extra sizes (the k=7 grid
# at max_num_seqs=8).
GRID_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]


@pytest.mark.parametrize(
    "adaptive_k,missed_before",
    [
        ({"by_load": [7, 5, 3]}, [(3, 3)]),
    ],
)
def test_every_decode_batch_reaches_a_full_graph_of_its_query_length(
    monkeypatch, adaptive_k, missed_before
):
    """Each query length 1+k gets its own candidate ladder. With one shared
    ladder, a batch whose next captured size held only other lengths' graphs
    missed the larger FULL graph of its own length and ran PIECEWISE;
    ``missed_before`` lists those (num_reqs, k)."""
    manager = _graph_manager(monkeypatch, adaptive_k, 7, 8, GRID_SIZES)
    manager._graphs_captured = True
    config = AdaptiveKConfig.from_dict(adaptive_k, 7)
    full = manager._capture_descs[CUDAGraphMode.FULL]
    checked = set()
    for k in config.allowed:
        query_len = k + 1
        for num_reqs in range(1, config.max_reqs_for(k, 8) + 1):
            servable = [
                d
                for d in full
                if d.uniform_token_count == query_len
                and d.num_reqs >= num_reqs
                and d.num_tokens >= num_reqs * query_len
            ]
            assert servable, (num_reqs, k)
            desc = _dispatch_uniform(manager, num_reqs, query_len)
            # The smallest FULL graph of this query length that fits.
            assert desc == min(servable, key=lambda d: d.num_tokens), (num_reqs, k)
            checked.add((num_reqs, k))
    assert set(missed_before) <= checked


# GLM tier 1: adaptive_k by_load 7,5,3 over 3 drafts with accept (TP costs),
# max_num_seqs=8, and the capture sizes it gets.
TIER1_ADAPTIVE_K = {
    "by_load": [7, 5, 3],
    "accept": {"costs": [1.0, 1.08, 1.16, 1.24, 1.32]},
}
TIER1_SIZES = [1, 2, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64]
# Dispatch for 1..8 requests at each query length, recorded with the shared
# ladder: "<tokens>F" is this length's FULL graph (tokens / length requests),
# "<tokens>P" the mixed PIECEWISE graph (shapes tier 1 never schedules).
TIER1_DISPATCH = {
    4: "4F 8F 12F 16F 24F 24F 32F 32F",
    5: "5F 10F 16P 24P 32P 32P 40P 40P",
    6: "6F 12F 24P 24P 32P 40P 48P 48P",
    7: "7F 16P 24P 32P 40P 48P 56P 56P",
    8: "8F 16P 24P 32P 40P 48P 56P 64P",
}


def test_tier1_dispatch_is_unchanged_by_per_length_ladders(monkeypatch):
    manager = _graph_manager(monkeypatch, TIER1_ADAPTIVE_K, 7, 8, TIER1_SIZES)
    manager._graphs_captured = True
    for query_len, row in TIER1_DISPATCH.items():
        for num_reqs, cell in enumerate(row.split(), start=1):
            num_tokens = int(cell[:-1])
            if cell[-1] == "F":
                graph_reqs = num_tokens // query_len
                expected = (CUDAGraphMode.FULL, num_tokens, graph_reqs, query_len)
            else:
                expected = (CUDAGraphMode.PIECEWISE, num_tokens, None, None)
            d = _dispatch_uniform(manager, num_reqs, query_len)
            got = (d.cg_mode, d.num_tokens, d.num_reqs, d.uniform_token_count)
            assert got == expected, (num_reqs, query_len)


@pytest.mark.parametrize(
    "mode,dispatched,missed",
    [
        (CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL, False),
        (CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.PIECEWISE, True),
        (CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.NONE, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, True),
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE, False),
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE, True),
        (CUDAGraphMode.NONE, CUDAGraphMode.NONE, False),
    ],
)
def test_decode_graph_miss_is_any_mode_below_the_decode_mode(
    monkeypatch, mode, dispatched, missed
):
    """The runner's miss counter counts a uniform decode step that ran below
    the configured decode mode, PIECEWISE included, not only eager."""
    manager = _graph_manager(
        monkeypatch, {"by_load": [7, 5, 3]}, 7, 8, GRID_SIZES, mode
    )
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=dispatched, num_tokens=12, num_reqs=3
    )
    assert manager.is_decode_graph_miss(desc) is missed


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
    vllm_config.additional_config = {"sm80": {"adaptive_k": dict(BY_LOAD)}}
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


def _tail_config(adaptive_k):
    spec = SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        num_speculative_tokens=5,
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
        additional_config={"sm80": {"adaptive_k": adaptive_k}} if adaptive_k else {},
    )


def test_draft_tail_carries_load_adaptive_depth():
    load = dt.draft_tail_gate(_tail_config(BY_LOAD), 2)
    assert load.enabled and load.stage == 2
    assert dt.draft_tail_gate(_tail_config(None), 2).enabled
