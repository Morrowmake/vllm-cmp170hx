# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE prefill kernel (VLLM_EXL3_MOE_PREFILL_TC, csrc/exl3/moe_prefill.cu).

CPU: the switch default, the configuration gate with its workspace, warm-up
and log lines (through vLLM's own logger), the dispatch in Exl3Experts.apply
by rows and by the scheduler phase of the step (calls of 65..3460 rows in
prefill steps only; decode steps of any size, calls of at most 64 rows and
larger calls keep their kernels; with the switch off the base rule), and the
operator's workspace contract. The configuration gate is the prefill's own: it
opens for both served expert partitions (256 and all 512 of 512 local) while
the decode gate (moe_out_v2) stays at 256, and the layout gate accepts the
validated (tensor, pipeline) sizes (2, 1) and (1, 2). The gate also closes
unless every local expert's trellis is 16-byte aligned, and the prefill
workspace does not grow with the local expert count.

GPU (SM 8.0 only): at the served shape (hidden 2560, intermediate 640, top-10,
bits_x2 10 / codebook 2) with 256 of 512 experts local, the off path is bitwise
moe_out; moe_out_prefill is held to moe_out's error against an FP64 reference
(pooled mean <= 1.10x, max <= 1.25x with no element excused, exact where
moe_out is exact) over awkward routings of 65..1001 rows; it is deterministic;
it refuses calls it does not cover; and a captured call replays bitwise like
the eager call without any allocator event. Every weight set's trellis is
16-byte aligned, as the gate requires. The same numerical, refusal, dispatch
and graph-replay coverage then runs with all 512 experts local and the identity
expert map (tensor parallel size 1, where pipeline parallelism splits the
layers), including routings that touch every expert exactly once, routings
through the local expert indices beyond 255 that only this partition has, and
hot experts.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
from tests.kernels.quantization.test_exl3_moe_v2 import (
    GLOBAL,
    HIDDEN,
    INTERMEDIATE,
    LOCAL,
    LOCAL_ALL,
    SERVED_META,
    TOPK,
    _apply,
    _bitwise,
    _case,
    _extension_or_skip,
    _Fp64Pool,
    _reference,
    _run,
    _v1,
    needs_sm80,
    served_experts,
)
from vllm.sm80.kernels.exl3 import ops as exl3_ops

# One sampled token plus three draft tokens under verification.
DECODE_LEN = 4


def _phase(monkeypatch, phase):
    """Forward context of a step of the given scheduler phase ("prefill",
    "decode") or none at all; the attention metadata carries the longest
    request's scheduled tokens, as the attention backends build it."""
    if phase is None:
        monkeypatch.setattr(forward_context, "_forward_context", None)
        return
    query_len = 309 if phase == "prefill" else DECODE_LEN
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            attn_metadata={
                "layers.3.self_attn": SimpleNamespace(max_query_len=query_len)
            }
        ),
    )


# ---------------------------------------------------------------- CPU tests


def _served_meta(local):
    """The per-expert [padded_in, padded_out, bits_x2, codebook] list of a
    layer with ``local`` local experts."""
    return [[list(projection) for projection in SERVED_META[0]] for _ in range(local)]


def _method(monkeypatch, warmed, num_speculative_tokens=3, local=LOCAL):
    """Exl3MoEMethod with only what the prefill gate reads, for a layer of
    ``local`` local experts; its workspace sizing and warm-up are recorded."""
    import vllm.config
    from vllm.sm80.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = local
    method.global_expert_ids = (
        list(range(GLOBAL)) if local == LOCAL_ALL else list(range(0, GLOBAL, 2))
    )
    method.swiglu_limit = 0.0
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(local, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(_served_meta(local)),
        exl3_expert_map=(
            torch.zeros(GLOBAL, dtype=torch.int32)
            if local == LOCAL
            else torch.arange(GLOBAL, dtype=torch.int32)
        ),
        top_k=TOPK,
        global_num_experts=GLOBAL,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        exl3_ops, "moe_workspace_size_prefill", lambda rows, *args: 1000 * rows
    )
    monkeypatch.setattr(
        exl3_ops, "warmup_moe_prefill", lambda *args: warmed.append(args)
    )
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: _config(num_speculative_tokens=num_speculative_tokens),
    )
    return method, layer


def _config(num_speculative_tokens=3, tp=2, pp=1, spec=None, graph_max=None):
    if spec is None and num_speculative_tokens:
        spec = SimpleNamespace(method="mtp", parallel_drafting=False)
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        speculative_config=spec,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=graph_max),
    )


@pytest.mark.parametrize("local, tp, pp", [(LOCAL, 2, 1), (LOCAL_ALL, 1, 2)])
def test_gate_opens_on_the_validated_layouts_and_partitions(monkeypatch, local, tp, pp):
    """Both served partitions on their validated layout: the gate opens, sizes
    the workspace for the layer's rows and warms up with the layer's own
    global expert ids (all 512 where every expert is local)."""
    import vllm.config

    warmed: list[tuple] = []
    method, layer = _method(monkeypatch, warmed, local=local)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _config(tp=tp, pp=pp)
    )
    assert method._moe_prefill(layer, _served_meta(local), 3460, 7) == (4, 3460000)
    assert len(warmed) == 1
    assert warmed[0][3] == method.global_expert_ids
    if local == LOCAL_ALL:
        assert warmed[0][0].shape == (GLOBAL, 3, 3)
        assert warmed[0][3][:TOPK] == list(range(TOPK))


@pytest.mark.parametrize(
    "config, words",
    [
        (dict(tp=4), "(4, 1)"),
        (dict(tp=1), "(1, 1)"),
        (dict(tp=2, pp=2), "(2, 2)"),
        (dict(tp=1, pp=4), "(1, 4)"),
        (
            dict(spec=SimpleNamespace(method="dflash", parallel_drafting=True)),
            "dflash drafts with parallel slots",
        ),
        (
            dict(spec=SimpleNamespace(method="eagle", parallel_drafting=True)),
            "eagle drafts with parallel slots",
        ),
    ],
)
def test_gate_closes_outside_the_validated_layouts_and_drafters(
    monkeypatch, config, words
):
    import vllm.config
    from vllm.sm80.kernels.exl3 import quant

    warmed: list[tuple] = []
    method, layer = _method(monkeypatch, warmed)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _config(**config)
    )
    lines = []
    monkeypatch.setattr(
        quant.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (0, 7)
    assert not warmed
    assert len(lines) == 1 and "gate is closed" in lines[0] and words in lines[0]


@pytest.mark.parametrize("tp, pp, local", [(2, 1, LOCAL_ALL), (1, 2, LOCAL)])
def test_gate_closes_when_the_partition_does_not_match_the_layout(
    monkeypatch, tp, pp, local
):
    import vllm.config

    warmed: list[tuple] = []
    method, layer = _method(monkeypatch, warmed, local=local)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _config(tp=tp, pp=pp)
    )
    assert method._moe_prefill(layer, _served_meta(local), 3460, 7) == (0, 7)
    assert not warmed


def test_gate_counts_the_speculative_draft(monkeypatch):
    method, layer = _method(monkeypatch, [], num_speculative_tokens=0)
    assert method._moe_prefill(layer, SERVED_META, 3460, 7)[0] == 1


@pytest.mark.parametrize("projection", [0, 1, 2])
def test_gate_closes_unless_every_expert_trellis_is_16_byte_aligned(
    monkeypatch, projection
):
    from vllm.sm80.kernels.exl3 import quant

    warmed: list[tuple] = []
    method, layer = _method(monkeypatch, warmed)
    lines = []
    monkeypatch.setattr(
        quant.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    # One 1,024,000-byte trellis (or scale vector) per address; the scale
    # vectors' alignment does not matter.
    pointers = 0x7F0000000000 + 1024000 * torch.arange(LOCAL * 9).view(LOCAL, 3, 3)
    pointers[:, :, 1:] += 2
    layer.exl3_ptrs = pointers.clone()
    assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (4, 3460000)
    assert len(warmed) == 1
    assert not any("gate is closed" in line for line in lines)
    pointers[37, projection, 0] += 2
    layer.exl3_ptrs = pointers
    assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (0, 7)
    assert len(warmed) == 1
    role = ("gate_proj", "up_proj", "down_proj")[projection]
    closed = [line for line in lines if "gate is closed" in line]
    assert len(closed) == 1
    assert f"local expert 37 {role} trellis is not 16-byte aligned" in closed[0]


def _prefill_reason(local=LOCAL, **changes):
    """moe_prefill_closed_reason of the served configuration of a layer with
    ``local`` local experts, with the given arguments changed."""
    args = dict(
        capability=(8, 0),
        dtype=torch.bfloat16,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        local_experts=local,
        global_experts=GLOBAL,
        metadata=_served_meta(local),
    )
    args.update(changes)
    return exl3_ops.moe_prefill_closed_reason(**args)


def test_prefill_gate_opens_for_both_served_partitions():
    assert _prefill_reason() is None  # 256 of 512 (tensor parallel size 2)
    assert _prefill_reason(local=LOCAL_ALL) is None  # all 512 (size 1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(local_experts=0),
        dict(local_experts=128),
        dict(local_experts=511),
        dict(local_experts=1024),
        dict(global_experts=256),
        dict(top_k=8),
        dict(hidden_size=4096),
        dict(intermediate_size=512),
        dict(capability=(8, 6)),
        dict(dtype=torch.float16),
    ],
)
@pytest.mark.parametrize("local", [LOCAL, LOCAL_ALL])
def test_prefill_gate_closes_outside_the_served_configuration(local, changes):
    assert _prefill_reason(local=local, **changes) is not None


@pytest.mark.parametrize("local, expert", [(LOCAL, 3), (LOCAL_ALL, LOCAL_ALL - 1)])
def test_prefill_gate_reads_every_local_expert_format(local, expert):
    metadata = _served_meta(local)
    assert _prefill_reason(local=local) is None
    metadata[expert][2] = [INTERMEDIATE, HIDDEN, 8, 2]
    reason = _prefill_reason(local=local, metadata=metadata)
    assert reason is not None


def test_prefill_workspace_does_not_grow_with_the_local_expert_count():
    """The workspace is bounded by the rows and top-k of the call (the routing
    table and the split operand rows are per assignment), not by how many
    experts the rank holds."""
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_prefill
    for rows in (65, 97, 1906, 3460):
        assert size(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL_ALL) == size(
            rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL
        )
    assert size(65, HIDDEN, INTERMEDIATE, TOPK, LOCAL_ALL) == size(
        65, HIDDEN, INTERMEDIATE, TOPK, 1
    )
    assert size(65, HIDDEN, INTERMEDIATE, TOPK, 0) == 0


def _experts(
    moe_v2_rows, prefill_decode_len, max_rows=8192, workspace=0, expert_map="layer map"
):
    """Exl3Experts without its constructor (see test_exl3_moe_v2._experts)."""
    from vllm.sm80.kernels.exl3.quant import Exl3Experts

    experts = object.__new__(Exl3Experts)
    experts.intermediate_size = INTERMEDIATE
    experts.expert_map = expert_map
    experts.workspace_elements = workspace
    experts.max_rows = max_rows
    experts.moe_v2_rows = moe_v2_rows
    experts.moe_out_rows = max_rows
    experts.moe_v2_bits_x2 = 10
    experts.prefill_decode_len = prefill_decode_len
    experts.prefill_v2_bits_x2 = 0
    experts.swiglu_limit = 0.0
    return experts


ROWS = (1, 4, 32, 63, 64, 65, 80, 97, 128, 256, 309, 1906, 3460, 3461, 8192)


def expected_operator(rows, phase, moe_v2_rows, prefill_decode_len):
    """The dispatch rule: the base rule, then calls of 65..3460 rows in a
    prefill step take moe_out_prefill when the gate is open."""
    if prefill_decode_len and 64 < rows <= 3460 and phase == "prefill":
        return "moe_out_prefill"
    return "moe_out_v2" if 0 < rows <= moe_v2_rows else "moe_out"


@pytest.mark.parametrize("moe_v2_rows", [0, 64])
@pytest.mark.parametrize("prefill_decode_len", [0, DECODE_LEN])
@pytest.mark.parametrize("phase", ["prefill", "decode", None])
def test_apply_dispatches_by_rows_and_scheduler_phase(
    monkeypatch, moe_v2_rows, prefill_decode_len, phase
):
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_prefill"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append(
                (_name, args[0].shape[0], args)
            ),
        )
    _phase(monkeypatch, phase)
    experts = _experts(moe_v2_rows, prefill_decode_len)
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    for rows in ROWS:
        x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        _apply(experts, x, "ids", "weights", "ptrs", "meta", "out", workspace2)
    expected = [
        (expected_operator(rows, phase, moe_v2_rows, prefill_decode_len), rows)
        for rows in ROWS
    ]
    assert [(name, rows) for name, rows, _ in calls] == expected
    if not prefill_decode_len or phase != "prefill":
        # Switch off, decode steps (80-256 rows: 20-64 requests verifying
        # three drafts) and steps without metadata never reach the new op.
        assert all(name != "moe_out_prefill" for name, _, _ in calls)
    for _, _, args in calls:
        # Same arguments for every operator: layer map, FP32 view of workspace2.
        assert args[1:6] == ("ids", "weights", "ptrs", "meta", "layer map")
        assert args[6] == INTERMEDIATE and args[7] == "out"
        assert args[8].dtype == torch.float32


def test_operator_list_and_workspace_contract():
    assert {"moe_out_prefill", "moe_workspace_size_prefill"} <= set(exl3_ops._OPERATORS)
    assert exl3_ops.MOE_PREFILL_MIN_ROWS == exl3_ops.MOE_V2_MAX_ROWS + 1 == 65
    assert exl3_ops.MOE_PREFILL_MAX_ROWS == 3460
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_prefill
    # 650 assignments: FP16 hi/lo gate/up operands (2 x 2 x 2560 halves, later
    # the down operands and the down results), gate/up results (2 x 640),
    # 2^-e per (assignment, projection) rounded to 4, then the routing table
    # (header 4, sorted 650 rounded to 4, tiles 4 x 650, positions 650)
    # rounded to 4.
    assignments = 650
    table = (4 + 652 + 4 * assignments + assignments + 3) // 4 * 4
    expected = assignments * (5120 + 1280) + (3 * assignments + 3) // 4 * 4 + table
    assert size(65, HIDDEN, INTERMEDIATE, TOPK, LOCAL) == expected
    assert size(65, HIDDEN, INTERMEDIATE, TOPK, 0) == 0
    with pytest.raises(RuntimeError, match="hidden 2560"):
        size(65, 4096, INTERMEDIATE, TOPK, LOCAL)


# ---------------------------------------------------------------- GPU tests


@pytest.fixture(scope="module")
def served():
    """The moe_out_v2 tests' expert sets with every trellis 16-byte aligned:
    moe_out_prefill requires it (its gate keeps moe_out otherwise). The
    "single" scenario still routes one assignment to one expert."""
    return served_experts(None)


def _p(served, x, ids, weights):
    return _run(
        exl3_ops.moe_out_prefill,
        served,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_prefill,
    )


@needs_sm80
@pytest.mark.parametrize("rows", [65, 1906])
def test_off_path_is_moe_out(served, monkeypatch, rows):
    _phase(monkeypatch, "prefill")
    x, ids, weights = _case(served, "uniform", rows, rows)
    expected = _v1(served, x, ids, weights)
    size = exl3_ops.moe_workspace_size(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL)
    for moe_v2_rows in (0, 64):
        experts = _experts(moe_v2_rows, 0, workspace=size, expert_map=served.expert_map)
        out = torch.full_like(x, float("nan"))
        workspace2 = torch.empty(2 * size, dtype=torch.bfloat16, device=x.device)
        _apply(experts, x, ids, weights, served.pointers, served.meta, out, workspace2)
        _bitwise(out, expected, f"off path rows={rows} moe_v2_rows={moe_v2_rows}")


@needs_sm80
def test_on_path_takes_the_prefill_op_only_in_prefill_steps(served, monkeypatch):
    rows = 97
    x, ids, weights = _case(served, "uniform", rows, 5)
    size = max(
        exl3_ops.moe_workspace_size(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        exl3_ops.moe_workspace_size_prefill(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
    )
    experts = _experts(64, DECODE_LEN, workspace=size, expert_map=served.expert_map)
    for phase, expected in (
        ("prefill", _p(served, x, ids, weights)),
        ("decode", _v1(served, x, ids, weights)),
    ):
        _phase(monkeypatch, phase)
        out = torch.full_like(x, float("nan"))
        workspace2 = torch.empty(2 * size, dtype=torch.bfloat16, device=x.device)
        _apply(experts, x, ids, weights, served.pointers, served.meta, out, workspace2)
        _bitwise(out, expected, f"{phase} step, {rows} rows")


SCENARIOS = [
    ("uniform", 65),
    ("uniform", 97),
    ("uniform", 129),
    ("uniform", 255),
    ("uniform", 511),
    ("uniform", 1001),
    ("masked", 777),
    ("hot", 512),
    ("hot", 70),
    ("single", 100),
    ("nonlocal", 66),
    ("int64", 81),
]


@needs_sm80
def test_prefill_op_holds_moe_outs_error_against_fp64(served):
    pool = _Fp64Pool()
    for index, (scenario, rows) in enumerate(SCENARIOS):
        x, ids, weights = _case(served, scenario, rows, 100 + index)
        name = f"{scenario} rows={rows}"
        cand = _p(served, x, ids, weights)
        inc = _v1(served, x, ids, weights)
        ref = _reference(served, x, ids, weights)
        if scenario in ("nonlocal", "single"):
            # Rows without a local route are exactly zero in both.
            local = ((ids >= 0) & (ids < GLOBAL) & (ids % 2 == 0)).any(-1)
            assert (cand[~local] == 0).all() and (inc[~local] == 0).all(), name
        # Deterministic: a second call is bitwise the first.
        _bitwise(_p(served, x, ids, weights), cand, f"{name}: repeat")
        pool.add(cand, inc, ref, name)
    pool.check("moe_out_prefill")


@needs_sm80
def test_prefill_op_refuses_calls_it_does_not_cover(served):
    x, ids, weights = _case(served, "uniform", 65, 3)
    out = torch.empty_like(x)
    args = (served.pointers, served.meta, served.expert_map)
    small = torch.empty(
        exl3_ops.moe_workspace_size_prefill(65, HIDDEN, INTERMEDIATE, TOPK, LOCAL) - 1,
        device=x.device,
    )
    with pytest.raises(RuntimeError, match="workspace is too small"):
        exl3_ops.moe_out_prefill(x, ids, weights, *args, INTERMEDIATE, out, small)
    workspace = torch.empty(small.numel() + 1, device=x.device)
    with pytest.raises(RuntimeError, match="at least 65 rows"):
        exl3_ops.moe_out_prefill(
            x[:64], ids[:64], weights[:64], *args, INTERMEDIATE, out[:64], workspace
        )
    with pytest.raises(RuntimeError, match="intermediate 640"):
        exl3_ops.moe_out_prefill(x, ids, weights, *args, 512, out, workspace)
    # Zero rows: nothing to do.
    exl3_ops.moe_out_prefill(
        x[:0], ids[:0], weights[:0], *args, INTERMEDIATE, out[:0], workspace
    )


def _allocations():
    return torch.accelerator.memory_stats()["allocation.all.allocated"]


def _graph_replay(served, rows, case, warm_ids):
    """Capture one prefill call of ``rows`` rows and replay it: the replay is
    bitwise the eager call, and with new inputs copied into the captured
    buffers it is bitwise a fresh eager call; no allocator event beyond the
    capture's own calibration. ``case(scenario, seed)`` builds the inputs of
    one call, ``warm_ids`` the layer's global expert ids."""
    exl3_ops.warmup_moe_prefill(
        served.pointers,
        served.meta,
        served.expert_map,
        warm_ids,
        HIDDEN,
        INTERMEDIATE,
        TOPK,
    )
    x, ids, weights = case("uniform", 21)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_prefill(
            rows, HIDDEN, INTERMEDIATE, TOPK, served.pointers.shape[0]
        ),
        device=x.device,
    )
    args = (served.pointers, served.meta, served.expert_map, INTERMEDIATE)

    def call():
        exl3_ops.moe_out_prefill(x, ids, weights, *args, out, workspace)

    torch.accelerator.synchronize()
    before = _allocations()
    call()
    torch.accelerator.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = _allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert _allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    _bitwise(out, eager, "replay vs eager")
    # New inputs in the captured buffers: the replay recomputes them.
    x2, ids2, weights2 = case("masked", 22)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.accelerator.synchronize()
    _bitwise(out, _p(served, x2, ids2, weights2), "replay with new inputs")


@needs_sm80
@pytest.mark.parametrize("rows", [65, 1906])
def test_prefill_graph_replay_is_bitwise_eager_without_allocations(served, rows):
    _graph_replay(
        served,
        rows,
        lambda scenario, seed: _case(served, scenario, rows, seed),
        list(range(0, GLOBAL, 2)),
    )


# ------------------------------------------------ all-local experts (512 local)


@pytest.fixture(scope="module")
def pp2():
    """All 512 experts local with the identity expert map: tensor parallel
    size 1, where pipeline parallelism splits the layers and every stage holds
    every expert. Every trellis is 16-byte aligned, as the gate requires."""
    return served_experts(None, local=LOCAL_ALL)


def _pp2_routes(scenario, rows, generator):
    """Routing of a call of the all-local partition: every global id is local,
    so the boundaries are the id range and the distribution over the 512
    experts, not another rank's experts."""
    ids = torch.randint(0, GLOBAL, (rows, TOPK), generator=generator)
    if scenario == "high":
        # Global ids 256..511, which under the identity map are the local
        # expert indices beyond 255: the pointer-table rows and route tables
        # only the all-local partition has (a tensor-parallel rank maps its
        # even global ids to local 0..255).
        ids = LOCAL + ids % (GLOBAL - LOCAL)
    elif scenario == "masked":  # padding and out of range, among local routes
        ids[:, 0] = -1
        ids[:, 1] = GLOBAL
        ids[:, 2] = -GLOBAL
    elif scenario == "hot":  # the two highest experts on every row
        ids[:, 0] = GLOBAL - 1
        ids[:, 1] = GLOBAL - 2
    elif scenario == "each":  # every expert exactly once, then any expert
        extra = rows * TOPK - GLOBAL
        ids = torch.cat(
            [
                torch.arange(GLOBAL),
                torch.randint(0, GLOBAL, (extra,), generator=generator),
            ]
        ).view(rows, TOPK)
    weights = torch.rand(rows, TOPK, generator=generator) + 0.01
    weights = weights / weights.sum(-1, keepdim=True)
    return ids if scenario == "int64" else ids.to(torch.int32), weights


def _pp2_case(scenario, rows, seed):
    generator = torch.Generator().manual_seed(seed)
    ids, weights = _pp2_routes(scenario, rows, generator)
    x = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
    return x.cuda(), ids.cuda(), weights.cuda()


PP2_SCENARIOS = [
    ("uniform", 65),  # 650 assignments over 512 experts: empty and partial tiles
    ("each", 65),
    ("high", 97),
    ("hot", 70),
    ("masked", 129),
    ("int64", 81),
    ("uniform", 512),
    ("high", 512),
]


@needs_sm80
def test_prefill_op_with_every_expert_local_holds_moe_outs_error_against_fp64(pp2):
    pool = _Fp64Pool()
    for index, (scenario, rows) in enumerate(PP2_SCENARIOS):
        x, ids, weights = _pp2_case(scenario, rows, 400 + index)
        name = f"{scenario} rows={rows}, 512 local experts"
        cand = _p(pp2, x, ids, weights)
        inc = _v1(pp2, x, ids, weights)
        ref = _reference(pp2, x, ids, weights)
        if scenario == "each":
            # One route per expert: every expert's contribution is consumed,
            # so a dropped or misplaced route moves its row by a whole route.
            # BF16 output rounding alone bounds a correct row at 0.002 of its
            # largest element; 0.01 is the same bound with its margin.
            assert ids.unique().numel() == GLOBAL, name
            error = (cand.double() - ref).abs()
            scale = ref.abs().amax(-1, keepdim=True).clamp_min(1.0)
            worst = float((error / scale).max())
            assert worst <= 0.01, f"{name}: worst row error {worst:.4f} of scale"
        # Deterministic: a second call is bitwise the first.
        _bitwise(_p(pp2, x, ids, weights), cand, f"{name}: repeat")
        pool.add(cand, inc, ref, name)
    pool.check("moe_out_prefill (512 local experts)")


@needs_sm80
def test_paths_with_every_expert_local_dispatch_on_rows_and_phase(pp2, monkeypatch):
    """A layer that holds every expert has no moe_out_v2 (its decode gate
    stays closed): prefill steps of 65..3460 rows take moe_out_prefill, decode
    steps keep moe_out bitwise."""
    rows = 97
    x, ids, weights = _pp2_case("high", rows, 11)
    size = max(
        exl3_ops.moe_workspace_size(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL_ALL),
        exl3_ops.moe_workspace_size_prefill(
            rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL_ALL
        ),
    )
    experts = _experts(0, DECODE_LEN, workspace=size, expert_map=pp2.expert_map)
    for phase, expected in (
        ("prefill", _p(pp2, x, ids, weights)),
        ("decode", _v1(pp2, x, ids, weights)),
    ):
        _phase(monkeypatch, phase)
        out = torch.full_like(x, float("nan"))
        workspace2 = torch.empty(2 * size, dtype=torch.bfloat16, device=x.device)
        _apply(experts, x, ids, weights, pp2.pointers, pp2.meta, out, workspace2)
        _bitwise(out, expected, f"{phase} step, {rows} rows, 512 local experts")


@needs_sm80
def test_prefill_op_with_every_expert_local_refuses_calls_it_does_not_cover(pp2):
    x, ids, weights = _pp2_case("uniform", 65, 12)
    out = torch.empty_like(x)
    args = (pp2.pointers, pp2.meta, pp2.expert_map)
    needed = exl3_ops.moe_workspace_size_prefill(
        65, HIDDEN, INTERMEDIATE, TOPK, LOCAL_ALL
    )
    small = torch.empty(needed - 1, device=x.device)
    with pytest.raises(RuntimeError, match="workspace is too small"):
        exl3_ops.moe_out_prefill(x, ids, weights, *args, INTERMEDIATE, out, small)
    workspace = torch.empty(needed, device=x.device)
    with pytest.raises(RuntimeError, match="at least 65 rows"):
        exl3_ops.moe_out_prefill(
            x[:64],
            ids[:64],
            weights[:64],
            *args,
            INTERMEDIATE,
            out[:64],
            workspace,
        )
    with pytest.raises(RuntimeError, match="intermediate 640"):
        exl3_ops.moe_out_prefill(x, ids, weights, *args, 512, out, workspace)
    # Zero rows: nothing to do.
    exl3_ops.moe_out_prefill(
        x[:0], ids[:0], weights[:0], *args, INTERMEDIATE, out[:0], workspace
    )


@needs_sm80
@pytest.mark.parametrize("rows", [65, 1906])
def test_prefill_graph_replay_with_every_expert_local_is_bitwise_eager(pp2, rows):
    _graph_replay(
        pp2,
        rows,
        lambda scenario, seed: _pp2_case(scenario, rows, seed),
        list(range(GLOBAL)),
    )
