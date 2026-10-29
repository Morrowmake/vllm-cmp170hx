# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE prefill kernel (VLLM_EXL3_MOE_PREFILL_TC, csrc/exl3/moe_prefill.cu).

CPU: the switch default, the configuration gate with its workspace, warm-up
and log lines (through vLLM's own logger), the dispatch in Exl3Experts.apply
by rows and by the scheduler phase of the step (calls of 65..3460 rows in
prefill steps only; decode steps of any size, calls of at most 64 rows and
larger calls keep their kernels; with the switch off the base rule), and the
operator's workspace contract. The gate also closes unless every local
expert's trellis is 16-byte aligned.

GPU (SM 8.0 only): at the served shape (hidden 2560, intermediate 640, top-10,
256 of 512 experts local, bits_x2 10 / codebook 2), decode steps are bitwise
moe_out; moe_out_prefill is held to moe_out's error against an FP64 reference
(pooled mean <= 1.10x, max <= 1.25x with no element excused, exact where
moe_out is exact) over awkward routings of 65..1001 rows; it is deterministic;
it refuses calls it does not cover; and a captured call replays bitwise like
the eager call without any allocator event. Every weight set's trellis is
16-byte aligned, as the gate requires.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
from tests.kernels.quantization.test_exl3_moe_v2 import (
    GLOBAL,
    HIDDEN,
    INTERMEDIATE,
    LOCAL,
    SERVED_META,
    TOPK,
    _apply,
    _bitwise,
    _case,
    _extension_or_skip,
    _Fp64Pool,
    _meta_with,
    _reference,
    _run,
    _v1,
    needs_sm80,
    served_experts,
)
from vllm_sm80.kernels.exl3 import ops as exl3_ops

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


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_MOE_PREFILL_TC", raising=False)
    assert envs.VLLM_EXL3_MOE_PREFILL_TC is False
    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL_TC", "1")
    assert envs.VLLM_EXL3_MOE_PREFILL_TC is True


def _method(monkeypatch, warmed, num_speculative_tokens=3):
    import vllm.config
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = LOCAL
    method.global_expert_ids = list(range(0, GLOBAL, 2))
    method.swiglu_limit = 0.0
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(LOCAL, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(SERVED_META),
        exl3_expert_map=torch.zeros(GLOBAL, dtype=torch.int32),
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


@pytest.mark.parametrize(
    "config, words",
    [
        (dict(tp=4), "(4, 1), not (2, 1)"),
        (dict(pp=2), "(2, 2), not (2, 1)"),
        (
            dict(spec=SimpleNamespace(method="eagle", parallel_drafting=True)),
            "eagle drafts with parallel slots",
        ),
    ],
)
def test_gate_closes_outside_the_validated_layout_and_drafters(
    monkeypatch, config, words
):
    import vllm.config
    from vllm_sm80.kernels.exl3 import quant

    warmed = []
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


def test_gate_sizes_workspace_warms_up_and_logs_through_vllm_logger(
    monkeypatch, caplog
):
    import vllm.logger as vllm_logger
    from vllm_sm80.kernels.exl3 import quant

    warmed = []
    method, layer = _method(monkeypatch, warmed)
    monkeypatch.setattr(quant, "logger", vllm_logger.init_logger(quant.__name__))
    # The vllm logger does not propagate to the root, where caplog listens.
    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    try:
        vllm_logger._print_info_once.cache_clear()
        # Open: decode steps schedule 1 + 3 tokens per request; workspace =
        # max(what the layer had, moe_out_prefill's at the layer's max rows).
        assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (4, 3460000)
        assert method._moe_prefill(layer, SERVED_META, 3460, 10**10) == (4, 10**10)
        # Larger batches: sized only for the validated 3460 rows.
        assert method._moe_prefill(layer, SERVED_META, 8192, 7) == (4, 3460000)
        assert method._moe_prefill(layer, SERVED_META, 1000, 7) == (4, 1000000)
        assert len(warmed) == 4 and warmed[0][3] == method.global_expert_ids
        # Closed: one line saying why; nothing sized or warmed.
        wrong = _meta_with(3, 0, [2560, 640, 8, 2])
        assert method._moe_prefill(layer, wrong, 3460, 7) == (0, 7)
        assert method._moe_prefill(layer, SERVED_META, 64, 7) == (0, 7)
        assert len(warmed) == 4
    finally:
        root.removeHandler(caplog.handler)
    lines = [(r.levelno, r.name, r.getMessage()) for r in caplog.records]
    assert (
        logging.INFO,
        quant.__name__,
        "EXL3 MoE prefill kernel active (VLLM_EXL3_MOE_PREFILL_TC): tensor-core "
        "routed experts for MoE calls of 65..3460 rows in prefill steps (a "
        "request scheduled more than 4 tokens); decode steps, calls of at most "
        "64 rows and larger calls keep their kernels.",
    ) in lines
    closed = sorted({m for _, _, m in lines if "gate is closed" in m})
    assert len(closed) == 2
    for line in closed:
        assert line.startswith(
            "VLLM_EXL3_MOE_PREFILL_TC is set but the gate is closed: "
        )
    assert any("local expert 3" in line for line in closed)
    assert any("at most 64 rows" in line for line in closed)


def test_gate_counts_the_speculative_draft(monkeypatch):
    method, layer = _method(monkeypatch, [], num_speculative_tokens=0)
    assert method._moe_prefill(layer, SERVED_META, 3460, 7)[0] == 1


@pytest.mark.parametrize("graph_max, said", [(64, False), (72, True)])
def test_gate_says_when_graphs_cover_prefill_sizes(monkeypatch, graph_max, said):
    import vllm.config
    from vllm_sm80.kernels.exl3 import quant

    method, layer = _method(monkeypatch, [])
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _config(graph_max=graph_max)
    )
    lines = []
    monkeypatch.setattr(
        quant.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    method._moe_prefill(layer, SERVED_META, 3460, 7)
    graph_lines = [m for m in lines if "CUDA graphs are captured up to" in m]
    assert len(graph_lines) == int(said)
    if said:
        assert "72 tokens" in graph_lines[0]


@pytest.mark.parametrize("projection", [0, 1, 2])
def test_gate_closes_unless_every_expert_trellis_is_16_byte_aligned(
    monkeypatch, projection
):
    from vllm_sm80.kernels.exl3 import quant

    warmed = []
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


def _experts(
    moe_v2_rows, prefill_decode_len, max_rows=8192, workspace=0, expert_map="layer map"
):
    """Exl3Experts without its constructor (see test_exl3_moe_v2._experts)."""
    from vllm_sm80.kernels.exl3.quant import Exl3Experts

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


@pytest.mark.parametrize(
    "moe_v2_rows, prefill_decode_len, phase",
    [
        (0, 0, "prefill"),
        (64, 0, "prefill"),
        (0, DECODE_LEN, "prefill"),
        (64, DECODE_LEN, "prefill"),
        (64, DECODE_LEN, "decode"),
        (64, DECODE_LEN, None),
    ],
)
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
    return torch.cuda.memory_stats()["allocation.all.allocated"]


@needs_sm80
@pytest.mark.parametrize("rows", [65, 1906])
def test_prefill_graph_replay_is_bitwise_eager_without_allocations(served, rows):
    exl3_ops.warmup_moe_prefill(
        served.pointers,
        served.meta,
        served.expert_map,
        list(range(0, GLOBAL, 2)),
        HIDDEN,
        INTERMEDIATE,
        TOPK,
    )
    x, ids, weights = _case(served, "uniform", rows, 21)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_prefill(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        device=x.device,
    )
    args = (served.pointers, served.meta, served.expert_map, INTERMEDIATE)

    def call():
        exl3_ops.moe_out_prefill(x, ids, weights, *args, out, workspace)

    torch.cuda.synchronize()
    before = _allocations()
    call()
    torch.cuda.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
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
    torch.cuda.synchronize()
    _bitwise(out, eager, "replay vs eager")
    # New inputs in the captured buffers: the replay recomputes them.
    x2, ids2, weights2 = _case(served, "masked", rows, 22)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _p(served, x2, ids2, weights2), "replay with new inputs")
