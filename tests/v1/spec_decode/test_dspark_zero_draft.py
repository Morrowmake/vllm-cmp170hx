# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark block geometry and speculation off under load.

DSpark (anchor-as-first) drafts its trained block of N query tokens per
request, not DFlash's 1 + N, whatever the draft width bookkeeping does. With
``num_speculative_tokens_per_batch_size`` [[1, 3, 5], [4, 8, 0]] it drafts 5
below 4 scheduled requests and nothing from 4 on: a zero-draft step writes
only the drafter's context K/V, and the decode CUDA graphs cover both regimes.
"""
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import CompilationConfig, CUDAGraphMode, ParallelConfig
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

pytestmark = pytest.mark.cpu_test

NUM_SPEC = 5
MAX_NUM_SEQS = 8
SCHEDULE = [(1, 3, NUM_SPEC), (4, MAX_NUM_SEQS, 0)]
CAPTURE_SIZES = [4, 5, 6, 7, 8, 12, 18]


def _scheduler() -> Scheduler:
    base = create_scheduler(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_SEQS * 10,
        num_speculative_tokens=NUM_SPEC,
    )
    base.vllm_config.speculative_config.num_speculative_tokens_per_batch_size = (
        SCHEDULE
    )
    return Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


@pytest.mark.parametrize("num_requests", range(1, MAX_NUM_SEQS + 1))
def test_drafts_five_below_four_requests_and_none_from_four(num_requests):
    scheduler = _scheduler()
    for request in create_requests(num_requests=num_requests, num_tokens=10):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == num_requests
    expected = NUM_SPEC if num_requests < 4 else 0
    assert output.num_spec_tokens_to_schedule == expected


def _graph_manager(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=CAPTURE_SIZES,
    )
    compilation_config.max_cudagraph_capture_size = CAPTURE_SIZES[-1]
    compilation_config.post_init_cudagraph_sizes()
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=MAX_NUM_SEQS
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.num_speculative_tokens = NUM_SPEC
    speculative_config = MagicMock()
    speculative_config.uses_dynamic_speculative_decoding.return_value = True
    speculative_config.num_speculative_tokens_per_batch_size = SCHEDULE
    speculative_config.uses_adaptive_k.return_value = False
    vllm_config.speculative_config = speculative_config
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=NUM_SPEC + 1,
    )
    manager._graphs_captured = True
    return manager


def test_capture_sizes_cover_both_regimes(monkeypatch):
    manager = _graph_manager(monkeypatch)
    # Drafting (1-3 requests): 6 tokens per request, graphs 6, 12, 18.
    for num_reqs in (1, 2, 3):
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs * (NUM_SPEC + 1),
            uniform_token_count=NUM_SPEC + 1,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_tokens == num_reqs * (NUM_SPEC + 1)
    # Plain decode (4-8 requests): 1 token per request, graphs 4..8.
    for num_reqs in range(4, MAX_NUM_SEQS + 1):
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            uniform_token_count=1,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_tokens == num_reqs


def _host_state(cls, num_query_per_req, widths=(NUM_SPEC,)):
    """A block drafter with its width bookkeeping only (no model, no device)."""
    s = object.__new__(cls)
    max_reqs = MAX_NUM_SEQS
    s.max_width = s.num_speculative_steps = s.width = max(widths)
    s.widths = tuple(widths)
    s._draft_token_views = {
        w: torch.full((max_reqs, w), 7, dtype=torch.int64) for w in widths
    }
    s._sample_cols = {
        w: torch.arange(w, dtype=torch.int32).repeat(max_reqs) for w in widths
    }
    s.draft_tokens = s._draft_token_views[s.max_width]
    s.sample_col = s._sample_cols[s.max_width]
    s._cg_managers = {}
    s.query_cudagraph_manager = None
    s.model = SimpleNamespace()
    s.num_query_per_req = num_query_per_req
    return s


def test_set_width_keeps_the_dspark_block():
    s = _host_state(DSparkSpeculator, num_query_per_req=NUM_SPEC)
    for width in (NUM_SPEC, 0, 3):
        # 0 and 3 are not DSpark widths: the trained block stays.
        s.set_width(width)
        assert (s.width, s.num_query_per_req) == (NUM_SPEC, NUM_SPEC)


def test_set_width_keeps_the_dflash_bonus_slot():
    # DFlash and DFlash2 keep the anchor as a bonus slot: 1 + width.
    s = _host_state(DFlashSpeculator, num_query_per_req=1 + NUM_SPEC, widths=(3, 5))
    block_sizes = []
    s._set_model_block_size = block_sizes.append
    for width in (3, 5, 3):
        s.set_width(width)
        assert s.num_query_per_req == 1 + width
    assert block_sizes == [4, 6, 4]


@pytest.mark.parametrize(
    "cls, num_query_per_req, widths, expected",
    [
        (DSparkSpeculator, NUM_SPEC, (NUM_SPEC,), [NUM_SPEC]),
        (DFlashSpeculator, 1 + NUM_SPEC, (3, NUM_SPEC), [4, 1 + NUM_SPEC]),
    ],
)
def test_drafter_graphs_use_the_block_length(
    monkeypatch, cls, num_query_per_req, widths, expected
):
    s = _host_state(cls, num_query_per_req, widths)
    s.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH, min_cg_attn_backend="x"
    )
    s.vllm_config = s.device = None
    lengths = []

    def manager(vllm_config, device, mode, decode_query_len):
        lengths.append(decode_query_len)
        return SimpleNamespace()

    monkeypatch.setattr(dflash_module, "DFlashCudaGraphManager", manager)
    s.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert lengths == expected
    assert s.num_query_per_req == num_query_per_req


# Runs the real input-preparation kernel under the Triton interpreter (CPU) in
# a child process: the interpreter is chosen when the kernel is defined.
_PREPARE = textwrap.dedent(
    """
    import json, sys
    from types import SimpleNamespace
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        prepare_dflash_inputs,
    )

    q, n, r, t = (int(a) for a in sys.argv[1:5])
    max_reqs, max_tokens = 4, 64
    def full(size, value, dtype=torch.int64):
        return torch.full((size,), value, dtype=dtype)
    buffers = SimpleNamespace(
        input_ids=full(max_tokens, -1, torch.int32),
        positions=full(max_tokens, -1),
        is_padding=torch.zeros(max_tokens, dtype=torch.bool),
        query_start_loc=full(max_reqs + 1, -1, torch.int32),
        seq_lens=full(max_reqs, -1, torch.int32),
    )
    positions = torch.cat(
        [torch.arange(100 * (i + 1), 100 * (i + 1) + t) for i in range(r)]
    )
    batch = SimpleNamespace(
        num_reqs=r,
        num_tokens=r * t,
        num_scheduled_tokens=np.full(r, t, dtype=np.int32),
        positions=positions,
        query_start_loc=torch.arange(0, (r + 1) * t, t, dtype=torch.int32),
        idx_mapping=torch.arange(r, dtype=torch.int32),
    )
    indices = full(max_reqs * n, -7)
    rows = full(max_reqs * n, -7)
    prepare_dflash_inputs(
        buffers, full(max_tokens, -2), full(max_tokens, -1),
        full(max_tokens, -2), indices, full(max_reqs * n, -7), rows,
        torch.zeros(max_reqs), torch.zeros(max_reqs, dtype=torch.int64), batch,
        torch.ones(r, dtype=torch.int32),
        torch.tensor([0, 2, 5, 1][:r], dtype=torch.int32),
        torch.tensor([11, 22, 33, 44]), torch.zeros(max_reqs, dtype=torch.int64),
        torch.ones(max_reqs), torch.zeros(max_reqs, dtype=torch.int64),
        torch.arange(1, 1 + max_reqs * 64, dtype=torch.int32).view(max_reqs, 64),
        16, 0, 1, 1, 999, q, n, max_reqs, max_tokens, 4096, True,
    )
    print(json.dumps({
        "indices": indices[: r * n].tolist(),
        "rows": rows[: r * n].tolist(),
        "input_ids": buffers.input_ids[: r * q].tolist(),
    }))
    """
)


def _prepare(num_query_per_req, num_reqs, target_tokens=NUM_SPEC + 1):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(
        [sys.executable, "-c", _PREPARE]
        + [str(v) for v in (num_query_per_req, NUM_SPEC, num_reqs, target_tokens)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _own_slots_only(result, num_query_per_req, num_reqs):
    """Every request samples N rows of its own query block, and its anchor
    (the request's bonus token) sits where DSpark's Markov head reads it."""
    indices, rows = result["indices"], result["rows"]
    anchors = [11, 22, 33, 44]
    for req in range(num_reqs):
        own = range(req * num_query_per_req, (req + 1) * num_query_per_req)
        mine = indices[req * NUM_SPEC : (req + 1) * NUM_SPEC]
        if sorted(mine) != list(own)[:NUM_SPEC] or set(mine) - set(own):
            return False
        if rows[req * NUM_SPEC : (req + 1) * NUM_SPEC] != [req] * NUM_SPEC:
            return False
        if result["input_ids"][req * NUM_SPEC] != anchors[req]:
            return False
    return True


@pytest.mark.parametrize("num_reqs", [1, 3])
def test_prepare_keeps_every_request_in_its_own_slots(num_reqs):
    s = _host_state(DSparkSpeculator, num_query_per_req=NUM_SPEC)
    s.set_width(NUM_SPEC)
    result = _prepare(s.num_query_per_req, num_reqs)
    assert _own_slots_only(result, s.num_query_per_req, num_reqs)


def test_a_six_token_block_mixes_requests():
    # The 1 + N block DSpark used to get: request r's last query row lands in
    # request r + 1's first sample slot.
    result = _prepare(NUM_SPEC + 1, num_reqs=3)
    assert not _own_slots_only(result, NUM_SPEC + 1, 3)


class _Stop(Exception):
    pass


def _dspark(monkeypatch, calls):
    """A DSpark drafter with host state only; every device step is recorded."""
    s = _host_state(DSparkSpeculator, num_query_per_req=NUM_SPEC)
    max_tokens = 64
    s.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))
    s.max_num_reqs = MAX_NUM_SEQS
    s.max_num_tokens = max_tokens
    s.hidden_states = torch.zeros(max_tokens, 4)
    s.pcp_manager = None
    s.draft_kv_cache_group_id = 0
    s.draft_kv_cache_group_ids = [0]
    s.block_tables = SimpleNamespace(
        slot_mappings=[None],
        input_block_tables=[None],
        block_sizes=[16],
        cp_rank=0,
        cp_interleave=1,
    )
    s.input_buffers = s.context_positions = s._context_slot_mappings = [None]
    s.sample_indices = s.sample_pos = s.sample_idx_mapping = None
    s.temperature = s.seeds = None
    s.dcp_size = 1
    s.dp_size = 1
    s.dp_rank = 0
    s.parallel_drafting_token_id = 0
    s.sample_from_anchor = True

    monkeypatch.setattr(
        dflash_module, "prepare_dflash_inputs", lambda *a: calls.append("prepare")
    )
    s._precompute_context_kv = lambda start, end, dummy_run=False: calls.append(
        ("context_kv", start, end)
    )

    def dispatch(*args, **kwargs):
        calls.append("draft_block")
        raise _Stop

    monkeypatch.setattr(dflash_module, "dispatch_cg_and_sync_dp", dispatch)
    s._generate_draft = lambda *a, **k: calls.append("draft_block")
    return s


def _propose(s, num_reqs, num_speculative_tokens):
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_reqs,
        seq_lens_cpu_upper_bound=np.full(num_reqs, 100),
    )
    zeros = torch.zeros(num_reqs, dtype=torch.int32)
    return s.propose(
        batch,
        {},
        {},
        torch.zeros(num_reqs, 4),
        None,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        num_speculative_tokens=num_speculative_tokens,
    )


def test_zero_drafts_store_context_kv_and_skip_the_draft_block(monkeypatch):
    calls: list = []
    s = _dspark(monkeypatch, calls)
    out = _propose(s, num_reqs=6, num_speculative_tokens=0)
    assert calls == ["prepare", ("context_kv", 0, 6)]
    assert out.shape == (6, NUM_SPEC)
    assert s.num_query_per_req == NUM_SPEC


def test_drafting_steps_run_the_draft_block(monkeypatch):
    calls: list = []
    s = _dspark(monkeypatch, calls)
    with pytest.raises(_Stop):
        _propose(s, num_reqs=3, num_speculative_tokens=NUM_SPEC)
    assert calls == ["prepare", "draft_block"]
    assert s.num_query_per_req == NUM_SPEC


def test_data_parallel_skips_the_draft_block_too(monkeypatch):
    # The data-parallel batch sync makes every rank agree on a zero-draft step.
    calls: list = []
    s = _dspark(monkeypatch, calls)
    s.dp_size = 2
    out = _propose(s, num_reqs=6, num_speculative_tokens=0)
    assert calls == ["prepare", ("context_kv", 0, 6)]
    assert out.shape == (6, NUM_SPEC)
