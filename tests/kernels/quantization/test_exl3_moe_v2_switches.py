# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 MoE switches VLLM_EXL3_MOE_V2_RING (moe_out_v2_try,
csrc/exl3/moe_v2_try.cu) and VLLM_EXL3_MOE_PREFILL_UNCHUNKED
(moe_out_prefill_try, csrc/exl3/moe_prefill_try.cu), both for the
DeepSeek-V4.1 target's whole-expert layers (hidden 5120, intermediate 2304,
top-6, 96 of 384 experts local, bits_x2 6 or 4).

CPU:
- both switches are declared, off by default and read through vllm.envs;
- the ring opens only for its configuration: not the hidden-2560 layers, the
  tensor-split chunks (640 or 512 columns), the DSpark drafter (top-3, bits_x2
  8) or bits_x2 8; the unchunked prefill likewise, read once at import;
- Exl3Experts.apply takes moe_out_v2_try for the calls moe_out_v2 serves when
  the layer's experts take it, before moe_out_v2_fold;
- Exl3MoEMethod._moe_v2_try requires the operator, sizes the workspace for
  it and warms it (trial) only with the switch, for the covered layer;
- the unchunked prefill workspace at 2048 rows fits the served moe_out
  workspace at its 2048-row chunk (the switch takes no KV cache memory), and
  the warm-up adds the 1025-row call (the GEMM shapes above 1024 rows).

GPU (SM 8.0): moe_out_v2_try returns moe_out_v2's output bytes for every
routing scenario at 1..64 rows (rows without local routes, hot experts,
masked and int64 ids, a 2-byte aligned weight set, the reached SwiGLU clamp),
with NaN in the output and workspace before each call; calls repeat bitwise
with a reused workspace holding garbage; a captured call replays bitwise
without allocator events. moe_out_prefill_try returns moe_out_prefill_v2's
bytes at 65, 1024, 1025 and 2048 rows and on the routing scenarios, and
repeats bitwise with a reused workspace.

    pytest -q tests/kernels/quantization/test_exl3_moe_v2_switches.py
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from tests.kernels.quantization import test_exl3_moe_v2 as h2560
from tests.kernels.quantization import test_exl3_moe_v2_hidden5120 as h5120
from vllm_sm80.kernels.exl3 import ops as exl3_ops

needs_sm80 = h2560.needs_sm80
SWITCHES = ("VLLM_EXL3_MOE_V2_RING", "VLLM_EXL3_MOE_PREFILL_UNCHUNKED")
CONFIG = (h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK, h5120.LOCAL, h5120.GLOBAL)

# ---------------------------------------------------------------- CPU tests


@pytest.mark.parametrize("name", SWITCHES)
def test_switch_declared_and_off_by_default(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False


def test_ring_opens_only_with_its_switch_for_its_configuration(monkeypatch):
    assert exl3_ops.MOE_V2_TRY_CONFIG == CONFIG
    monkeypatch.delenv("VLLM_EXL3_MOE_V2_RING", raising=False)
    assert not any(exl3_ops.moe_v2_try_open(*CONFIG, b) for b in (6, 4))
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_RING", "1")
    assert all(exl3_ops.moe_v2_try_open(*CONFIG, b) for b in (6, 4))
    assert not exl3_ops.moe_v2_try_open(*CONFIG, 8)
    for other in (
        (2560, 640, 10, 256, 512),  # the hidden-2560 layers
        (5120, 640, 6, 384, 384),  # the tensor-split chunks
        (5120, 512, 6, 384, 384),
        (5120, 2304, 3, 32, 128),  # the DSpark drafter's experts
        (5120, 640, 3, 128, 128),
    ):
        assert not exl3_ops.moe_v2_try_open(*other, 6), other


def test_unchunked_prefill_is_read_through_envs_once(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_MOE_PREFILL_UNCHUNKED", raising=False)
    assert exl3_ops.TRY_MOE_PREFILL is False
    # Read once at import: setting it later changes nothing in this process.
    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL_UNCHUNKED", "1")
    assert exl3_ops.TRY_MOE_PREFILL is False
    assert not exl3_ops._try_moe_prefill(5120, 2304, 6, 96)


def test_unchunked_prefill_covers_the_whole_expert_shape_only(monkeypatch):
    monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", True)
    assert exl3_ops._try_moe_prefill(5120, 2304, 6, 96)
    for other in ((5120, 640, 6, 384), (5120, 512, 6, 384), (5120, 2304, 3, 32),
                  (2560, 640, 10, 256)):
        assert not exl3_ops._try_moe_prefill(*other), other
    assert exl3_ops.TRY_MOE_PREFILL_ROWS == (65, 2048)


@pytest.mark.parametrize("ring,fold", [(False, False), (True, False), (True, True)])
def test_apply_takes_the_ring_operator_for_the_v2_calls(monkeypatch, ring, fold):
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_v2_fold", "moe_out_v2_try"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append(
                (_name, args[0].shape[0], kwargs)
            ),
        )
    experts = h2560._experts(64)
    experts.moe_v2_try, experts.moe_v2_fold = ring, fold
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    for rows in (1, 64, 65):
        x = torch.zeros(rows, h2560.HIDDEN, dtype=torch.bfloat16)
        h2560._apply(experts, x, "ids", "weights", "ptrs", "meta", "out", workspace2)
    v2 = "moe_out_v2_try" if ring else "moe_out_v2"
    assert [(name, rows) for name, rows, _ in calls] == [(v2, 1), (v2, 64), ("moe_out", 65)]
    for _, _, kwargs in calls[:2]:
        assert kwargs == {"swiglu_limit": 0.0, "bits_x2": h2560.BITS_X2}


def _ring_method(monkeypatch, warmed, required, local=h5120.LOCAL):
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.hidden_size = h5120.HIDDEN
    method.intermediate_size = h5120.INTERMEDIATE
    method.local_experts = local
    method.global_expert_ids = list(range(local))
    method.swiglu_limit = h5120.SWIGLU_LIMIT
    layer = SimpleNamespace(
        exl3_ptrs="ptrs", exl3_meta="meta", exl3_expert_map="map",
        top_k=h5120.TOPK, global_num_experts=h5120.GLOBAL,
    )  # fmt: skip
    monkeypatch.setattr(exl3_ops, "require_moe_v2_try", lambda: required.append(1))
    monkeypatch.setattr(exl3_ops, "moe_workspace_size_v2_try",
                        lambda rows, *args: 1000 * rows)
    monkeypatch.setattr(exl3_ops, "warmup_moe_v2",
                        lambda *args, **kwargs: warmed.append((args, kwargs)))
    return method, layer


def test_method_requires_sizes_and_warms_the_ring_only_with_its_switch(monkeypatch):
    warmed, required = [], []
    method, layer = _ring_method(monkeypatch, warmed, required)
    monkeypatch.delenv("VLLM_EXL3_MOE_V2_RING", raising=False)
    assert method._moe_v2_try(layer, 64, 6, 7) == (False, 7)
    assert not warmed and not required
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_RING", "1")
    assert method._moe_v2_try(layer, 64, 6, 7) == (True, 64000)
    assert method._moe_v2_try(layer, 64, 4, 99999) == (True, 99999)
    assert required == [1, 1]
    assert [kwargs for _, kwargs in warmed] == [{"trial": True}] * 2
    assert [args[7] for args, _ in warmed] == [6, 4]  # the layer's bits_x2
    # bits_x2 8 and another local expert count stay on moe_out_v2.
    assert method._moe_v2_try(layer, 64, 8, 7) == (False, 7)
    method, layer = _ring_method(monkeypatch, warmed, required, local=48)
    assert method._moe_v2_try(layer, 64, 6, 7) == (False, 7)
    assert len(warmed) == 2


def test_unchunked_prefill_workspace_fits_the_served_buffer(monkeypatch):
    h2560._extension_or_skip()
    args = (h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK, h5120.LOCAL)
    served = max(
        torch.ops._exl3_C.moe_workspace_size(2048, *args),
        torch.ops._exl3_C.moe_workspace_size_v2(64, *args),
    )
    sizes = {}
    for switch in (False, True):
        monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", switch)
        exl3_ops.moe_workspace_size_prefill_v2.cache_clear()
        sizes[switch] = exl3_ops.moe_workspace_size_prefill_v2(2048, *args)
    exl3_ops.moe_workspace_size_prefill_v2.cache_clear()
    try_size = torch.ops._exl3_C.moe_workspace_size_prefill_try(2048, *args)
    assert sizes[False] == torch.ops._exl3_C.moe_workspace_size_prefill_v2(2048, *args)
    assert sizes[True] == max(sizes[False], try_size)
    # Served at --max-num-batched-tokens 2048 and exl3_chunk_rows 2048:
    # the switch takes no KV cache memory.
    assert sizes[True] <= served


def test_unchunked_prefill_warmup_adds_the_long_call(monkeypatch):
    rows = []
    monkeypatch.setattr(exl3_ops, "moe_workspace_size_prefill_v2", lambda r, *a: 4)
    monkeypatch.setattr(exl3_ops, "moe_out_prefill_v2",
                        lambda x, *args, **kwargs: rows.append(x.shape[0]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None, raising=False)
    ptrs = torch.zeros(h5120.LOCAL, 3, 3, dtype=torch.int64)
    # The layer's stored formats: the warm-up checks them on the host first.
    meta = torch.tensor(
        [exl3_ops.moe_v2_expert_meta(h5120.HIDDEN, h5120.INTERMEDIATE, 6)] * h5120.LOCAL
    )
    for switch in (False, True):
        monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", switch)
        monkeypatch.setattr(exl3_ops, "_moe_prefill_v2_warm", set())
        rows.clear()
        exl3_ops.warmup_moe_prefill_v2(
            ptrs, meta, "map", list(range(h5120.LOCAL)), h5120.HIDDEN,
            h5120.INTERMEDIATE, h5120.TOPK, 6,
        )  # fmt: skip
        assert rows == ([exl3_ops.MOE_PREFILL_MIN_ROWS, 1025] if switch
                        else [exl3_ops.MOE_PREFILL_MIN_ROWS])


# ---------------------------------------------------------------- GPU tests


def _require(*names):
    h2560._extension_or_skip()
    for name in names:
        torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")


@pytest.fixture(scope="module", params=h5120.BITS, ids=lambda b: f"bits_x2={b}")
def synthetic(request):
    _require("moe_out_v2_try", "moe_workspace_size_v2_try")
    experts = h5120._synthetic(request.param)
    yield experts
    experts.release()


def _ring(experts, x, ids, weights, workspace=None, out=None):
    out = torch.full_like(x, float("nan")) if out is None else out
    if workspace is None:
        workspace = torch.full(
            (exl3_ops.moe_workspace_size_v2_try(x.shape[0], h5120.HIDDEN,
                                                h5120.INTERMEDIATE, h5120.TOPK,
                                                h5120.LOCAL),),
            float("nan"), device=x.device,
        )  # fmt: skip
    exl3_ops.moe_out_v2_try(
        x, ids, weights, experts.pointers, experts.meta, experts.expert_map,
        h5120.INTERMEDIATE, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
        bits_x2=experts.bits_x2,
    )  # fmt: skip
    return out


@needs_sm80
@pytest.mark.parametrize("scenario,rows,scale", h5120.SCENARIOS)
def test_ring_is_bitwise_v2(synthetic, scenario, rows, scale):
    x, ids, w = h5120._case(scenario, rows, 31 * rows + 5, scale)
    ring = _ring(synthetic, x, ids, w)
    h2560._bitwise(ring, h5120._v2(synthetic, x, ids, w), f"{scenario} rows={rows}")
    assert not torch.isnan(ring).any()


@needs_sm80
def test_ring_is_bitwise_v2_at_every_row_count(synthetic):
    for rows in range(1, exl3_ops.MOE_V2_MAX_ROWS + 1):
        x, ids, w = h5120._case("uniform", rows, rows)
        h2560._bitwise(_ring(synthetic, x, ids, w), h5120._v2(synthetic, x, ids, w),
                       f"rows={rows}")


@needs_sm80
def test_ring_repeats_bitwise_with_a_reused_workspace(synthetic):
    rows = 48
    workspace = torch.full(
        (exl3_ops.moe_workspace_size_v2_try(64, h5120.HIDDEN, h5120.INTERMEDIATE,
                                            h5120.TOPK, h5120.LOCAL),),
        float("nan"), device="cuda",
    )  # fmt: skip
    first = None
    for count in (rows, 64, 1, rows, rows):
        x, ids, w = h5120._case("uniform", count, 99 if count == rows else count)
        out = _ring(synthetic, x, ids, w, workspace)
        if count == rows:
            if first is None:
                first = out.clone()
                h2560._bitwise(out, h5120._v2(synthetic, x, ids, w), "ring vs v2")
            h2560._bitwise(out, first, "repeat")
        workspace.view(torch.int32)[::7] = -12345  # garbage between calls


@needs_sm80
@pytest.mark.parametrize("rows", [6, 48])
def test_ring_graph_replay_is_bitwise_without_allocations(synthetic, rows):
    exl3_ops.warmup_moe_v2(
        synthetic.pointers, synthetic.meta, synthetic.expert_map,
        list(range(h5120.LOCAL)), h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK,
        synthetic.bits_x2, h5120.SWIGLU_LIMIT, trial=True,
    )  # fmt: skip
    x, ids, weights = h5120._case("uniform", rows, 11)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_v2_try(rows, h5120.HIDDEN, h5120.INTERMEDIATE,
                                           h5120.TOPK, h5120.LOCAL),
        device=x.device,
    )  # fmt: skip

    def call():
        _ring(synthetic, x, ids, weights, workspace, out)

    torch.cuda.synchronize()
    before = h2560._allocations()
    call()
    torch.cuda.synchronize()
    assert h2560._allocations() == before, "eager call allocated"
    h2560._bitwise(out, h5120._v2(synthetic, x, ids, weights), "eager ring vs v2")
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = h2560._allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert h2560._allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, eager, "replay vs eager")
    x2, ids2, weights2 = h5120._case("masked", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, h5120._v2(synthetic, x2, ids2, weights2), "replay new inputs")


@pytest.fixture(scope="module", params=h5120.BITS, ids=lambda b: f"bits_x2={b}")
def aligned(request):
    """The synthetic experts with every trellis 16-byte aligned (the prefill
    operators' gate requires it)."""
    _require("moe_out_prefill_try", "moe_workspace_size_prefill_try",
             "moe_out_prefill_v2")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(h5120, "UNALIGNED_SET", -1)
        experts = h5120._synthetic(request.param)
    yield experts
    experts.release()


def _prefill(op, experts, x, ids, weights, workspace=None):
    args = (h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK, h5120.LOCAL)
    if workspace is None:
        size = max(torch.ops._exl3_C.moe_workspace_size_prefill_v2(x.shape[0], *args),
                   torch.ops._exl3_C.moe_workspace_size_prefill_try(x.shape[0], *args))
        workspace = torch.full((size,), float("nan"), device=x.device)
    out = torch.full_like(x, float("nan"))
    getattr(torch.ops._exl3_C, op)(
        x, ids, weights, experts.pointers, experts.meta, experts.expert_map,
        h5120.INTERMEDIATE, experts.bits_x2, out, workspace, h5120.SWIGLU_LIMIT,
    )  # fmt: skip
    return out


PREFILL_CASES = [
    ("uniform", 65, 1.0),
    ("uniform", 1024, 1.0),
    ("uniform", 1025, 1.0),
    ("uniform", 2048, 1.0),
    ("local", 2048, 1.0),
    ("masked", 200, 1.0),
    ("nonlocal", 130, 1.0),
    ("hot", 300, 1.0),
    ("single", 70, 1.0),
    ("int64", 130, 1.0),
    ("uniform", 517, 16.0),  # the gate/up clamp at 10 is reached
]


@needs_sm80
@pytest.mark.parametrize("scenario,rows,scale", PREFILL_CASES)
def test_unchunked_prefill_is_bitwise_prefill_v2(aligned, scenario, rows, scale):
    x, ids, w = h5120._case(scenario, rows, 13 * rows + 1, scale)
    got = _prefill("moe_out_prefill_try", aligned, x, ids, w)
    want = _prefill("moe_out_prefill_v2", aligned, x, ids, w)
    h2560._bitwise(got, want, f"{scenario} rows={rows}")
    assert not torch.isnan(got).any()


@needs_sm80
def test_unchunked_prefill_repeats_bitwise_with_a_reused_workspace(aligned):
    args = (h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK, h5120.LOCAL)
    workspace = torch.full(
        (torch.ops._exl3_C.moe_workspace_size_prefill_try(2048, *args),),
        float("nan"), device="cuda",
    )  # fmt: skip
    first = None
    for count in (1025, 2048, 65, 1025):
        x, ids, w = h5120._case("uniform", count, 7 if count == 1025 else count)
        out = _prefill("moe_out_prefill_try", aligned, x, ids, w, workspace)
        if count == 1025:
            if first is None:
                first = out.clone()
            h2560._bitwise(out, first, "repeat")
        workspace.view(torch.int32)[::7] = -12345  # garbage between calls


@needs_sm80
def test_unchunked_prefill_dispatch_takes_it_only_for_its_rows(aligned, monkeypatch):
    """moe_out_prefill_v2 with the switch: the operator's bits at 65..2048
    rows (the try), and the base kernels' everywhere (the same bits)."""
    monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", True)
    for rows in (65, 2048):
        x, ids, w = h5120._case("uniform", rows, rows)
        got = h5120._run(
            partial(exl3_ops.moe_out_prefill_v2, bits_x2=aligned.bits_x2),
            aligned, x, ids, w,
            lambda r, *a: max(torch.ops._exl3_C.moe_workspace_size_prefill_v2(r, *a),
                              torch.ops._exl3_C.moe_workspace_size_prefill_try(r, *a)),
        )  # fmt: skip
        h2560._bitwise(got, _prefill("moe_out_prefill_v2", aligned, x, ids, w),
                       f"dispatch rows={rows}")
