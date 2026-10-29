# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE prefill v2 (VLLM_DEEPSEEK_V41_MOE_PREFILL_TC,
csrc/exl3/moe_prefill_v2.cu): the routed experts of DeepSeek-V4.1-Flash
(hidden 5120, intermediate 2304, top-6, 96 of 384 experts local, bits_x2 6 or
4, codebook 2, SwiGLU clamp 10) on the tensor cores for prefill calls.

CPU: the switch default; the configuration gate (both widths open; the
drafter's bits_x2 8, the hidden-2560 shape and other layouts close); the
engine gate (tensor parallel size 4 without pipeline parallelism; a parallel
drafter only when no decode step can carry 65 rows; at least 65 rows; 16-byte
aligned trellis bases) with its workspace, warm-up and log lines through
vLLM's own logger; adopt_moe marking the routed experts only with the switch
set; the dispatch in Exl3Experts.apply by rows and scheduler phase at both
moe_out row chunks (512 and 2048): calls of 65..2048 rows in prefill steps
take the operator whole, everything else keeps the base rule; and the
workspace contract (no larger than the served MoE workspace at 512-row
chunks); the operator refusing on the host, before any launch, a call whose
experts are stored in another format or with an unaligned trellis base.

GPU (SM 8.0), each at bits_x2 6 and 4, on synthetic experts with 16-byte
aligned trellis bases (the gate requires it): the on path takes the operator
only in prefill steps and is bitwise the base kernels in decode steps at both
chunk sizes; the operator holds moe_out's error against an FP64 reference
(pooled mean <= 1.10x, max <= 1.25x with no element excused, exact where
moe_out is exact) over awkward routings of 65..2048 rows with the clamp
reached; rows without local routes are exact zero and calls repeat bitwise;
a row's bits do not depend on the call it is in; it refuses what it does not
cover; a captured call replays bitwise like the eager call without allocator
events. With VLLM_EXL3_TEST_CHECKPOINT: the same FP64 rule on the real experts
of layers 10 (bits_x2 6) and 20 (bits_x2 4) at 65, 517 and 2044 rows.
Captured served calls are checked by
tests/models/deepseek_v41/sm80/fp64_cases.py (family
exl3_moe_out_prefill_v2).
"""

import logging
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tests.kernels.quantization.test_exl3_moe_v2_hidden5120 as h5120
from tests.kernels.quantization.test_exl3_moe_prefill import DECODE_LEN, _phase
from tests.kernels.quantization.test_exl3_moe_v2 import (
    _allocations,
    _apply,
    _bitwise,
    _extension_or_skip,
    _Fp64Pool,
    needs_sm80,
)
from tests.kernels.quantization.test_exl3_moe_v2_hidden5120 import (
    BITS,
    HIDDEN,
    INTERMEDIATE,
    LOCAL,
    REAL_LAYERS,
    SWIGLU_LIMIT,
    TOPK,
    _case,
    _meta,
    _meta_with,
    _real_layer,
    _reason,
    _routes,
    _run,
    _v1,
)
from vllm_sm80.kernels.exl3 import ops as exl3_ops

MAX_ROWS = 2048

# ---------------------------------------------------------------- CPU tests


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_MOE_PREFILL_TC is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", "1")
    assert envs.VLLM_DEEPSEEK_V41_MOE_PREFILL_TC is True


def test_gate_opens_for_both_bit_widths_only():
    reason = partial(_reason, fn=exl3_ops.moe_prefill_v2_closed_reason)
    assert reason() is None
    assert reason(metadata=_meta(4)) is None
    closed = {
        "drafter bits_x2 8": reason(metadata=_meta(8)),
        "mixed widths": reason(metadata=_meta_with(6, 40, 2, [2304, 5120, 4, 2])),
        "codebook": reason(metadata=_meta_with(4, 95, 0, [5120, 2304, 4, 1])),
        "local": reason(local_experts=128, metadata=_meta(6) + _meta(6)[:32]),
        "global": reason(global_experts=256),
        "top_k": reason(top_k=8),
        "hidden 2560": reason(
            hidden_size=2560,
            intermediate_size=640,
            top_k=10,
            local_experts=256,
            global_experts=512,
        ),
        "capability": reason(capability=(8, 6)),
        "dtype": reason(dtype=torch.float16),
    }
    for name, why in closed.items():
        assert why, name
    assert "bits_x2 8" in closed["drafter bits_x2 8"]
    assert "local expert 40" in closed["mixed widths"]


def test_operator_refuses_other_formats_before_launch(monkeypatch):
    # A call whose experts are stored in another format than the call's
    # (hidden, intermediate, bits_x2), or with an unaligned trellis base, is
    # refused on the host; the kernels would stop the device on it.
    launched = []
    monkeypatch.setattr(exl3_ops, "ensure_extension", lambda: None)
    monkeypatch.setattr(
        torch.ops._exl3_C,
        "moe_out_prefill_v2",
        lambda *args: launched.append(args[6:8]),
        raising=False,
    )
    meta = torch.tensor(_meta(6), dtype=torch.int64)
    pointers = torch.full((LOCAL, 3, 3), 4096, dtype=torch.int64)
    x = torch.empty((65, HIDDEN), dtype=torch.bfloat16)
    ids = torch.zeros((65, TOPK), dtype=torch.int32)

    def call(intermediate, bits_x2, ptrs=pointers):
        exl3_ops.moe_out_prefill_v2(
            x, ids, None, ptrs, meta, None, intermediate, None, None, bits_x2=bits_x2
        )

    call(INTERMEDIATE, 6)
    assert launched == [(INTERMEDIATE, 6)]
    for intermediate, bits_x2 in ((INTERMEDIATE, 4), (640, 6), (512, 4)):
        with pytest.raises(RuntimeError, match="unsupported"):
            call(intermediate, bits_x2)
    unaligned = pointers.clone()
    unaligned[5, 1, 0] += 8
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        call(INTERMEDIATE, 6, unaligned)
    call(INTERMEDIATE, 6)
    assert launched == [(INTERMEDIATE, 6)] * 2


def _config(tp=4, pp=1, spec=None, max_num_seqs=8, k=3):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        num_speculative_tokens=k,
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
    )


def _dspark(k):
    return SimpleNamespace(
        method="dspark", parallel_drafting=True, num_speculative_tokens=k
    )


def _method(monkeypatch, warmed, required, config=None, bits_x2=6):
    import vllm.config
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = LOCAL
    method.global_expert_ids = list(range(LOCAL))
    method.swiglu_limit = SWIGLU_LIMIT
    method.prefill_v2 = True
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(LOCAL, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(_meta(bits_x2)),
        exl3_expert_map=torch.zeros(h5120.GLOBAL, dtype=torch.int32),
        top_k=TOPK,
        global_num_experts=h5120.GLOBAL,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        exl3_ops, "moe_workspace_size_prefill_v2", lambda rows, *args: 1000 * rows
    )
    monkeypatch.setattr(
        exl3_ops, "warmup_moe_prefill_v2", lambda *args: warmed.append(args)
    )
    monkeypatch.setattr(
        exl3_ops, "require_moe_prefill_v2", lambda: required.append(True)
    )
    config = _config() if config is None else config
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    return method, layer


def test_gate_sizes_workspace_warms_up_and_logs_through_vllm_logger(
    monkeypatch, caplog
):
    import vllm.logger as vllm_logger
    from vllm_sm80.kernels.exl3 import quant

    warmed, required = [], []
    method, layer = _method(monkeypatch, warmed, required, bits_x2=4)
    monkeypatch.setattr(quant, "logger", vllm_logger.init_logger(quant.__name__))
    # The vllm logger does not propagate to the root, where caplog listens.
    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    try:
        vllm_logger._print_info_once.cache_clear()
        # Open: decode steps schedule 1 + 3 tokens per request; workspace =
        # max(what the layer had, the operator's at the layer's max rows).
        assert method._moe_prefill_v2(layer, _meta(4), 2048, 7) == (4, 4, 2048000)
        assert method._moe_prefill_v2(layer, _meta(4), 2048, 10**10) == (
            4,
            4,
            10**10,
        )
        # Larger calls: sized only for the validated 2048 rows.
        assert method._moe_prefill_v2(layer, _meta(4), 8192, 7) == (4, 4, 2048000)
        assert method._moe_prefill_v2(layer, _meta(4), 512, 7) == (4, 4, 512000)
        assert len(warmed) == len(required) == 4
        args = warmed[0]
        assert args[3] == method.global_expert_ids
        assert args[4:9] == (HIDDEN, INTERMEDIATE, TOPK, 4, SWIGLU_LIMIT)
        # Closed: one line saying why; nothing required, sized or warmed.
        wrong = _meta_with(4, 3, 0, [5120, 2304, 8, 2])
        assert method._moe_prefill_v2(layer, wrong, 2048, 7) == (0, 0, 7)
        assert method._moe_prefill_v2(layer, _meta(4), 64, 7) == (0, 0, 7)
        assert len(warmed) == len(required) == 4
    finally:
        root.removeHandler(caplog.handler)
    lines = [(r.levelno, r.name, r.getMessage()) for r in caplog.records]
    assert (
        logging.INFO,
        quant.__name__,
        "EXL3 MoE prefill v2 tensor-core path active "
        "(VLLM_DEEPSEEK_V41_MOE_PREFILL_TC): routed experts (bits_x2 6 or 4) for "
        "MoE calls of 65..2048 rows in prefill steps (a request scheduled more "
        "than 4 tokens); decode steps, graph captures, calls of at most 64 rows "
        "and larger calls keep their kernels.",
    ) in lines
    closed = sorted({m for _, _, m in lines if "gate is closed" in m})
    assert len(closed) == 2
    for line in closed:
        assert line.startswith(
            "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC is set but the gate is closed: "
        )
    assert any("local expert 3" in line for line in closed)
    assert any("at most 64 rows" in line for line in closed)


@pytest.mark.parametrize(
    "config, words",
    [
        (_config(tp=2), "(2, 1), not (4, 1)"),
        (_config(pp=2), "(4, 2), not (4, 1)"),
    ],
)
def test_gate_closes_outside_the_validated_layout(monkeypatch, config, words):
    from vllm_sm80.kernels.exl3 import quant

    warmed, required = [], []
    method, layer = _method(monkeypatch, warmed, required, config=config)
    lines = []
    monkeypatch.setattr(
        quant.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    assert method._moe_prefill_v2(layer, _meta(6), 2048, 7) == (0, 0, 7)
    assert not warmed and not required
    assert len(lines) == 1 and "gate is closed" in lines[0] and words in lines[0]


@pytest.mark.parametrize(
    "config, decode_len",
    [
        (_config(), 4),
        (_config(spec=SimpleNamespace(method="mtp", num_speculative_tokens=3)), 4),
        # A DSpark decode pass carries 1 + k verified tokens and k parallel
        # slots per request, so a step is a prefill step above 1 + 2k tokens;
        # 8 requests x (1 + 2 x 5) = 88 rows: open, by the per-request bound.
        (_config(spec=_dspark(5), k=5), 11),
        (
            _config(
                spec=SimpleNamespace(
                    method="dflash", parallel_drafting=True, num_speculative_tokens=8
                ),
                k=8,
            ),
            17,
        ),
    ],
    ids=["no drafter", "mtp", "dspark k5 c8", "dflash"],
)
def test_gate_opens_for_the_served_layout_and_parallel_drafters(
    monkeypatch, config, decode_len
):
    warmed, required = [], []
    method, layer = _method(monkeypatch, warmed, required, config=config)
    assert method._moe_prefill_v2(layer, _meta(6), 2048, 7) == (6, decode_len, 2048000)
    assert len(warmed) == len(required) == 1


@pytest.mark.parametrize("projection", [2])
def test_gate_closes_unless_every_expert_trellis_is_16_byte_aligned(
    monkeypatch, projection
):
    from vllm_sm80.kernels.exl3 import quant

    warmed, required = [], []
    method, layer = _method(monkeypatch, warmed, required)
    lines = []
    monkeypatch.setattr(
        quant.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    pointers = 0x7F0000000000 + 13315584 * torch.arange(LOCAL * 9).view(LOCAL, 3, 3)
    pointers[:, :, 1:] += 2
    layer.exl3_ptrs = pointers.clone()
    assert method._moe_prefill_v2(layer, _meta(6), 2048, 7)[0] == 6
    pointers[37, projection, 0] += 8
    layer.exl3_ptrs = pointers
    assert method._moe_prefill_v2(layer, _meta(6), 2048, 7) == (0, 0, 7)
    assert len(warmed) == 1
    role = ("gate_proj", "up_proj", "down_proj")[projection]
    closed = [line for line in lines if "gate is closed" in line]
    assert len(closed) == 1
    assert f"local expert 37 {role} trellis is not 16-byte aligned" in closed[0]


def test_set_moe_prefill_v2_marks_only_exl3_routed_experts():
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod, set_moe_prefill_v2

    routed = nn.Module()
    routed.quant_method = object.__new__(Exl3MoEMethod)
    routed.quant_method.prefill_v2 = False
    assert set_moe_prefill_v2(routed) is None
    assert routed.quant_method.prefill_v2 is True
    other = nn.Module()
    other.quant_method = SimpleNamespace()
    reason = set_moe_prefill_v2(other)
    assert reason and "has no EXL3 MoE method" in reason


@pytest.mark.parametrize("switch", [False, True])
def test_adopt_moe_marks_the_routed_experts_only_with_the_switch(monkeypatch, switch):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm_sm80.models.deepseek_v41 import sm80_moe
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod

    if switch:
        monkeypatch.setenv("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", "1")
    else:
        monkeypatch.delenv("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", raising=False)
    monkeypatch.setattr(sm80_moe, "_fp32_gate_weight", lambda gate: None)
    router = object.__new__(FusedTopKBiasRouter)
    router.num_fused_shared_experts = 0
    routed = nn.Module()
    routed.quant_method = object.__new__(Exl3MoEMethod)
    routed.quant_method.prefill_v2 = False
    ffn = SimpleNamespace(
        use_native_mega_moe=False,
        gate=SimpleNamespace(tid2eid=None, e_score_correction_bias=torch.zeros(4)),
        scoring_func="sqrtsoftplus",
        experts=SimpleNamespace(router=router, routed_experts=routed),
    )
    sm80_moe.adopt_moe(ffn, SimpleNamespace())
    assert type(router) is sm80_moe.Sm80DeepseekV41Router
    assert routed.quant_method.prefill_v2 is switch


def _experts(prefill_v2_bits_x2, moe_out_rows, expert_map="layer map"):
    """Exl3Experts without its constructor (see test_exl3_moe_v2._experts)."""
    from vllm_sm80.kernels.exl3.quant import Exl3Experts

    experts = object.__new__(Exl3Experts)
    experts.intermediate_size = INTERMEDIATE
    experts.expert_map = expert_map
    experts.workspace_elements = 0
    experts.max_rows = MAX_ROWS
    experts.moe_v2_rows = 64
    experts.moe_out_rows = moe_out_rows
    experts.moe_v2_bits_x2 = 6
    experts.prefill_decode_len = 0
    experts.prefill_v2_bits_x2 = prefill_v2_bits_x2
    experts.prefill_v2_decode_len = DECODE_LEN if prefill_v2_bits_x2 else 0
    experts.swiglu_limit = SWIGLU_LIMIT
    return experts


ROWS = (1, 12, 64, 65, 97, 511, 512, 513, 1024, 1531, 2044, 2048)


def expected_calls(rows, phase, bits_x2, moe_out_rows):
    """The dispatch rule: calls of 65..2048 rows in a prefill step take
    moe_out_prefill_v2 whole when the gate is open; otherwise moe_out_v2 up
    to 64 rows and moe_out in launches of at most moe_out_rows rows."""
    if bits_x2 and 64 < rows <= MAX_ROWS and phase == "prefill":
        return [("moe_out_prefill_v2", rows)]
    if rows <= 64:
        return [("moe_out_v2", rows)]
    return [
        ("moe_out", min(moe_out_rows, rows - start))
        for start in range(0, rows, moe_out_rows)
    ]


@pytest.mark.parametrize(
    "moe_out_rows, bits_x2, phase",
    [
        (512, 0, "prefill"),
        (2048, 0, "prefill"),
        (512, 6, "prefill"),
        (512, 4, "prefill"),
        (512, 6, "decode"),
        (512, 6, None),
    ],
)
def test_apply_dispatches_by_rows_and_scheduler_phase(
    monkeypatch, moe_out_rows, bits_x2, phase
):
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_prefill", "moe_out_prefill_v2"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append(
                (_name, args[0].shape[0], args, kwargs)
            ),
        )
    _phase(monkeypatch, phase)
    experts = _experts(bits_x2, moe_out_rows)
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    expected = []
    for rows in ROWS:
        x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        ids = torch.zeros(rows, TOPK, dtype=torch.int32)
        weights = torch.zeros(rows, TOPK)
        _apply(experts, x, ids, weights, "ptrs", "meta", out, workspace2)
        expected += expected_calls(rows, phase, bits_x2, moe_out_rows)
    assert [(name, rows) for name, rows, _, _ in calls] == expected
    for name, _, args, kwargs in calls:
        # Same arguments for every operator: layer map, FP32 view of workspace2.
        assert args[3:6] == ("ptrs", "meta", "layer map")
        assert args[6] == INTERMEDIATE and args[8].dtype == torch.float32
        assert kwargs.pop("swiglu_limit") == SWIGLU_LIMIT
        if name == "moe_out_prefill_v2":
            assert kwargs == {"bits_x2": bits_x2}
        elif name == "moe_out_v2":
            assert kwargs == {"bits_x2": 6}
        else:
            assert kwargs == {}


def test_workspace_contract():
    assert exl3_ops.MOE_PREFILL_V2_MAX_ROWS == MAX_ROWS
    assert exl3_ops.MOE_PREFILL_V2_CONFIGS == {
        (HIDDEN, INTERMEDIATE, TOPK, LOCAL, 384): (6, 4),
        # Tensor-parallel routed experts (test_exl3_moe_tp_shapes.py).
        (HIDDEN, 640, TOPK, 384, 384): (6, 4),
        (HIDDEN, 512, TOPK, 384, 384): (6, 4),
    }
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_prefill_v2
    # Calls run in row chunks of 512 rows: the size stops growing there.
    assert size(MAX_ROWS, HIDDEN, INTERMEDIATE, TOPK, LOCAL) == size(
        512, HIDDEN, INTERMEDIATE, TOPK, LOCAL
    )
    assert size(65, HIDDEN, INTERMEDIATE, TOPK, LOCAL) < size(
        512, HIDDEN, INTERMEDIATE, TOPK, LOCAL
    )
    assert size(MAX_ROWS, HIDDEN, INTERMEDIATE, TOPK, 0) == 0
    # The served MoE workspace at 512-row moe_out chunks already holds it, so
    # the switch never takes KV cache memory.
    served = max(
        torch.ops._exl3_C.moe_workspace_size(512, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        torch.ops._exl3_C.moe_workspace_size_v2(64, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
    )
    assert size(MAX_ROWS, HIDDEN, INTERMEDIATE, TOPK, LOCAL) <= served


# ---------------------------------------------------------------- GPU helpers


def _synthetic(bits_x2):
    """test_exl3_moe_v2_hidden5120's synthetic experts with every trellis
    16-byte aligned (no unaligned weight set)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(h5120, "UNALIGNED_SET", -1)
        return h5120._synthetic(bits_x2)


@pytest.fixture(scope="module", params=BITS, ids=lambda b: f"bits_x2={b}")
def synthetic(request):
    experts = _synthetic(request.param)
    yield experts
    experts.release()


def _p(experts, x, ids, weights):
    return _run(
        partial(exl3_ops.moe_out_prefill_v2, bits_x2=experts.bits_x2),
        experts,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_prefill_v2,
    )


def _engine(experts, x, ids, weights, bits_x2, moe_out_rows):
    """Exl3Experts.apply on the layer, its workspace sized as the method
    sizes it (moe_out at its row chunk, moe_out_v2 at 64 rows, the operator
    at 2048 rows)."""
    engine = _experts(bits_x2, moe_out_rows, expert_map=experts.expert_map)
    size = max(
        exl3_ops.moe_workspace_size(moe_out_rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        exl3_ops.moe_workspace_size_v2(64, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        exl3_ops.moe_workspace_size_prefill_v2(
            MAX_ROWS, HIDDEN, INTERMEDIATE, TOPK, LOCAL
        ),
    )
    workspace2 = torch.full(
        (2 * size,), float("nan"), dtype=torch.bfloat16, device=x.device
    )
    out = torch.full_like(x, float("nan"))
    _apply(engine, x, ids, weights, experts.pointers, experts.meta, out, workspace2)
    return out


SCENARIOS = [
    ("uniform", 65, 1.0),
    ("uniform", 97, 1.0),
    ("local", 128, 1.0),
    ("masked", 200, 1.0),
    ("hot", 300, 1.0),
    ("int64", 130, 1.0),
    ("uniform", 513, 1.0),
    ("uniform", 1531, 1.0),
    ("uniform", 2044, 1.0),
    ("local", 2048, 1.0),
    ("uniform", 65, 16.0),  # the gate/up clamp at 10 is reached
    ("local", 517, 16.0),
]


def _pool_rule(experts, cases, name):
    """The operator and moe_out on every case, one FP64 reference pass over
    all of them (each local expert decoded once)."""
    refs = h5120.real.moe_fp64(
        cases, experts.expert_map, experts.decode, INTERMEDIATE, SWIGLU_LIMIT
    )
    pool = _Fp64Pool()
    for (x, ids, w), ref in zip(cases, refs):
        pool.add(
            _p(experts, x, ids, w),
            _v1(experts, x, ids, w),
            ref,
            f"{name} rows={x.shape[0]}",
        )
    return pool


# ---------------------------------------------------------------- GPU tests


@needs_sm80
@pytest.mark.parametrize("moe_out_rows", [512, 2048])
def test_on_path_takes_the_operator_only_in_prefill_steps(
    synthetic, monkeypatch, moe_out_rows
):
    bits = synthetic.bits_x2
    for rows in (65, 1531, 2044):
        x, ids, weights = _case("uniform", rows, 4 + rows)
        _phase(monkeypatch, "prefill")
        _bitwise(
            _engine(synthetic, x, ids, weights, bits, moe_out_rows),
            _p(synthetic, x, ids, weights),
            f"prefill step, {rows} rows",
        )
        _phase(monkeypatch, "decode")
        _bitwise(
            _engine(synthetic, x, ids, weights, bits, moe_out_rows),
            _v1(synthetic, x, ids, weights),
            f"decode step, {rows} rows",
        )


@needs_sm80
def test_operator_meets_the_fp64_gate_of_moe_out(synthetic, record_property):
    cases = [
        _case(scenario, rows, 300 + index, scale)
        for index, (scenario, rows, scale) in enumerate(SCENARIOS)
    ]
    mean, worst = _pool_rule(synthetic, cases, f"bits_x2={synthetic.bits_x2}").check()
    record_property("mean_ratio", mean)
    record_property("max_ratio", worst)
    print(
        f"synthetic bits_x2={synthetic.bits_x2}: mean x{mean:.3f} "
        f"max x{worst:.3f} of moe_out"
    )


@needs_sm80
def test_nonlocal_rows_are_exact_zero_and_calls_repeat_bitwise(synthetic):
    x, ids, weights = _case("nonlocal", 70, 7)
    assert (_p(synthetic, x, ids, weights) == 0).all()
    x, ids, weights = _case("hot", 600, 8)
    first = _p(synthetic, x, ids, weights)
    for _ in range(3):
        _bitwise(_p(synthetic, x, ids, weights), first, "repeat")


@needs_sm80
def test_rows_do_not_depend_on_the_call(synthetic):
    # Every row depends only on its own input and routes, whatever the call's
    # size, the row's position and the 512-row chunk it falls in.
    x, ids, weights = _case("local", MAX_ROWS, 13)
    whole = _p(synthetic, x, ids, weights)
    for start, stop in ((0, 65), (0, 511), (0, 513), (0, 1024), (7, 600), (1983, 2048)):
        part = _p(synthetic, x[start:stop], ids[start:stop], weights[start:stop])
        _bitwise(part, whole[start:stop], f"rows {start}..{stop} of {MAX_ROWS}")


@needs_sm80
def test_operator_refuses_calls_it_does_not_cover(synthetic):
    x, ids, weights = _case("uniform", 65, 9)
    out = torch.empty_like(x)
    args = (synthetic.pointers, synthetic.meta, synthetic.expert_map)
    size = exl3_ops.moe_workspace_size_prefill_v2(65, HIDDEN, INTERMEDIATE, TOPK, LOCAL)
    small = torch.empty(size - 1, device=x.device)
    with pytest.raises(RuntimeError, match="workspace is too small"):
        exl3_ops.moe_out_prefill_v2(
            x, ids, weights, *args, INTERMEDIATE, out, small, bits_x2=synthetic.bits_x2
        )
    workspace = torch.empty(size, device=x.device)
    # Shapes and widths without kernels, and those with kernels but not the
    # layer's stored format, are refused on the host before any launch.
    other = 4 if synthetic.bits_x2 == 6 else 6
    for bits_x2, intermediate in (
        (8, INTERMEDIATE),
        (10, INTERMEDIATE),
        (6, 640),
        (other, INTERMEDIATE),
    ):
        with pytest.raises(RuntimeError, match="unsupported"):
            exl3_ops.moe_out_prefill_v2(
                x, ids, weights, *args, intermediate, out, workspace, bits_x2=bits_x2
            )
    # Zero rows: nothing to do.
    exl3_ops.moe_out_prefill_v2(
        x[:0],
        ids[:0],
        weights[:0],
        *args,
        INTERMEDIATE,
        out[:0],
        workspace,
        bits_x2=synthetic.bits_x2,
    )


@needs_sm80
@pytest.mark.parametrize("rows", [65, 2044])
def test_graph_replay_is_bitwise_eager_without_allocations(synthetic, rows):
    exl3_ops.warmup_moe_prefill_v2(
        synthetic.pointers,
        synthetic.meta,
        synthetic.expert_map,
        list(range(LOCAL)),
        HIDDEN,
        INTERMEDIATE,
        TOPK,
        synthetic.bits_x2,
        SWIGLU_LIMIT,
    )
    x, ids, weights = _case("uniform", rows, 21)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_prefill_v2(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        device=x.device,
    )
    args = (synthetic.pointers, synthetic.meta, synthetic.expert_map, INTERMEDIATE)

    def call():
        exl3_ops.moe_out_prefill_v2(
            x,
            ids,
            weights,
            *args,
            out,
            workspace,
            swiglu_limit=SWIGLU_LIMIT,
            bits_x2=synthetic.bits_x2,
        )

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
    x2, ids2, weights2 = _case("masked", rows, 22)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _p(synthetic, x2, ids2, weights2), "replay with new inputs")


@needs_sm80
@pytest.mark.parametrize("layer", sorted(REAL_LAYERS))
def test_operator_holds_moe_outs_error_on_real_layer_weights(layer, record_property):
    experts, norm_weight = _real_layer(layer, REAL_LAYERS[layer])
    cases = []
    for rows in (65, 517, 2044):
        generator = torch.Generator().manual_seed(1000 * layer + rows)
        ids, weights = _routes("uniform", rows, generator)
        h = torch.randn(rows, HIDDEN, generator=generator, dtype=torch.float64)
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        x = (h.cuda().float() * norm_weight).to(torch.bfloat16)
        cases.append((x, ids.cuda(), weights.cuda()))
    mean, worst = _pool_rule(experts, cases, f"layer {layer}").check()
    record_property("mean_ratio", mean)
    record_property("max_ratio", worst)
    print(
        f"layer {layer} bits_x2={experts.bits_x2} rows (65, 517, 2044): "
        f"mean x{mean:.3f} max x{worst:.3f} of moe_out"
    )
    experts.release()
