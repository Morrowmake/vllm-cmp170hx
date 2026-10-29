# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense tensor-core blocks (VLLM_EXL3_DENSE_MMA, csrc/exl3/dense_mma.cu).

CPU: the switch and diagnostic override defaults, the gate (split-K active,
sm_80, 74 multiprocessors, TP2 and PP1), the override parsing, the one-time
resolution with its log lines (through vLLM's own logger) and library calls,
its place in ensure_extension, and the library's plan rule where it needs no
device (matrices and launches outside the blocks, the LM head's wide plan,
the override).

GPU (SM 8.0 only): on the tensor-core path the result is held to
linear_out's error against an FP64 reference (pooled mean <= 1.10x, max <=
1.25x, BF16 rounding ties forgiven, exact where linear_out is exact) for
every served matrix but the LM head (whose FP64 check is the real-input
test's), row counts 1-32 and every forced plan, and is deterministic;
launches outside the path (other matrices, formats, row counts) are the
split-K operator bit for bit; the LM head runs mma_kernel and neither
gemm_kernel nor pipe_kernel (profiler kernel names through exl3_linear's
body); row chunks of exl3_linear's body whose out is a slice at an offset
give each chunk's own launch bit for bit; a captured call replays bitwise
like the eager call without any allocator event.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_dense_splitk import (
    _bitwise,
    _extension_or_skip,
    _inputs,
    _matrix,
    _Pool,
    _run,
    _run_core,
    _sms,
    needs_sm80,
)
from vllm_sm80.kernels.exl3 import dense_mma
from vllm_sm80.kernels.exl3 import ops as exl3_ops

SMS = 74  # the multiprocessor count the plan rule was measured on
# The LM head of the TP2 rank: K 2560, N 124160 (half the vocabulary), 6 bpw.
HEAD = (2560, 124160, 12, 2)
# (padded_in, padded_out, bits_x2) of the matrices the blocks serve, codebook
# 2: one Qwen3.8-Flash-Next rank at TP2 (csrc/exl3/dense_mma.cu
# kServedShapes).
SERVED_DENSE = [
    (2560, 256, 14),  # attention k/v
    (2560, 640, 14),  # shared expert gate/up
    (2560, 640, 10),  # sparse-attention indexer q/k
    (2560, 1280, 12),  # drafter embedding / hidden-state projections
    (2560, 3072, 14),  # linear-attention z
    (2560, 5120, 14),  # linear-attention qkv
    (2560, 6144, 14),  # attention q (with output gate)
    (640, 2560, 14),  # shared expert down
    (3072, 2560, 14),  # output projections
]
SERVED_SHAPES = [*SERVED_DENSE, HEAD[:3]]
# N of a matrix just wide enough for the 4-tile plan at 74 SMs: not served.
WIDE_74 = 64 * 8 * SMS
# Matrices of the served formats that the blocks do not serve.
UNSERVED = [
    (2560, WIDE_74, 14),
    (2560, 12288, 14),  # attention q at TP1
    (6144, 2560, 14),  # attention output at TP1
    (2560, 248320, 12),  # the LM head at TP1
    (2560, 640, 12),  # a served shape in another served format
    (2560, 1280, 14),
    (4224, 256, 14),
    (640, 384, 14),
]


# ---------------------------------------------------------------- CPU tests


def test_switches_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_DENSE_MMA", raising=False)
    monkeypatch.delenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", raising=False)
    assert envs.VLLM_EXL3_DENSE_MMA is False
    assert envs.VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE == ""
    monkeypatch.setenv("VLLM_EXL3_DENSE_MMA", "1")
    monkeypatch.setenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", "2,8,2")
    assert envs.VLLM_EXL3_DENSE_MMA is True
    assert envs.VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE == "2,8,2"


def _config(tp=2, pp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        )
    )


def test_gate_needs_split_k_sm80_74_sms_and_the_tp2_layout():
    reason = dense_mma.closed_reason
    assert reason(True, (8, 0), SMS, _config()) is None
    assert "split-K operator is not active" in reason(False, (8, 0), SMS, _config())
    assert "(8, 6)" in reason(True, (8, 6), SMS, _config())
    assert "None" in reason(True, None, None, _config())
    assert "70 multiprocessors, not 74" in reason(True, (8, 0), 70, _config())
    assert "no vLLM config" in reason(True, (8, 0), SMS, None)
    assert "(1, 1), not (2, 1)" in reason(True, (8, 0), SMS, _config(tp=1))
    assert "(2, 2), not (2, 1)" in reason(True, (8, 0), SMS, _config(pp=2))
    assert "(4, 1), not (2, 1)" in reason(True, (8, 0), SMS, _config(tp=4))


def test_shape_override_parsing():
    assert dense_mma.parse_shape_override("") is None
    assert dense_mma.parse_shape_override("2,8,2") == (2, 8, 2)
    assert dense_mma.parse_shape_override("4,1,1") == (4, 1, 1)
    for bad in ("3,1,1", "2,3,1", "2,8,3", "2,8", "2,8,2,1", "a,b,c", "4"):
        with pytest.raises(ValueError, match="MT,KW,SPLITS"):
            dense_mma.parse_shape_override(bad)


def _device(monkeypatch, capability, sms=SMS, config=True, **layout):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sms),
    )
    import vllm.config

    value = _config(**layout) if config else None
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: value)


class _Library:
    """Records the library calls resolve() makes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))


def test_resolution_logs_through_vllm_logger(monkeypatch, caplog):
    import vllm.logger as vllm_logger

    monkeypatch.setattr(
        dense_mma, "logger", vllm_logger.init_logger(dense_mma.__name__)
    )
    monkeypatch.setattr(dense_mma, "active", False)
    library = _Library()
    monkeypatch.setattr(dense_mma, "_library", lambda: library)
    # The split-K operator's resolved gate (its own log lines are its test's).
    splitk = [True]
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: splitk[0])
    # The vllm logger does not propagate to the root, where caplog listens.
    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    try:
        vllm_logger._print_info_once.cache_clear()
        monkeypatch.delenv("VLLM_EXL3_DENSE_MMA", raising=False)
        monkeypatch.delenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", raising=False)
        _device(monkeypatch, (8, 0))
        dense_mma.resolve()
        assert not caplog.records and not library.calls, "the switch at 0 is silent"
        monkeypatch.setenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", "2,8,2")
        dense_mma.resolve()
        assert not library.calls and dense_mma.active is False
        monkeypatch.delenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE")
        monkeypatch.setenv("VLLM_EXL3_DENSE_MMA", "1")
        splitk[0] = False
        dense_mma.resolve()
        splitk[0] = True
        for capability, sms, config, layout in (
            ((8, 6), SMS, True, {}),
            ((8, 0), 70, True, {}),
            ((8, 0), SMS, False, {}),
            ((8, 0), SMS, True, {"tp": 1}),
        ):
            _device(monkeypatch, capability, sms, config, **layout)
            dense_mma.resolve()
            assert not library.calls and dense_mma.active is False
        _device(monkeypatch, (8, 0))
        dense_mma.resolve()
        assert dense_mma.active is True
        assert library.calls == [("dense_mma_prepare",), ("set_dense_mma", True)]
        library.calls.clear()
        monkeypatch.setenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", "4,1,1")
        dense_mma.resolve()
        assert library.calls == [
            ("set_dense_mma_shape_override", 4, 1, 1),
            ("dense_mma_prepare",),
            ("set_dense_mma", True),
        ]
        monkeypatch.setenv("VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", "4,1")
        with pytest.raises(ValueError, match="MT,KW,SPLITS"):
            dense_mma.resolve()
    finally:
        root.removeHandler(caplog.handler)
    lines = [(r.levelno, r.name, r.getMessage()) for r in caplog.records]
    assert {(level, name) for level, name, _ in lines} == {
        (logging.INFO, dense_mma.__name__)
    }
    closed = (
        "VLLM_EXL3_DENSE_MMA is set but the gate is closed: {}; every dense EXL3 "
        "launch keeps its current path."
    )
    # Each record may reach the capture handler twice (vllm and the root).
    assert list(dict.fromkeys(message for _, _, message in lines)) == [
        "VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE is set but VLLM_EXL3_DENSE_MMA is off; "
        "the override is ignored.",
        closed.format(
            "the split-K operator is not active (VLLM_EXL3_DENSE_SPLITK off or its "
            "gate closed; the tensor-core blocks run inside linear_out_splitk)"
        ),
        closed.format("device capability (8, 6), not (8, 0)"),
        closed.format("70 multiprocessors, not 74"),
        closed.format("no vLLM config is set, so the parallel layout is unknown"),
        closed.format("(tensor, pipeline) parallel size (1, 1), not (2, 1)"),
        "EXL3 dense MMA active (VLLM_EXL3_DENSE_MMA): dense launches of 1..32 rows "
        "of the validated TP2 matrices (bits_x2 10/12/14, codebook 2; the LM head "
        "included) run on tensor-core blocks inside linear_out_splitk; other "
        "launches keep the split-K operator's path.",
        "EXL3 dense MMA plan forced by VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE: mt=4 kw=1 "
        "splits=1 on every tensor-core launch (diagnostic).",
    ]


def test_library_without_the_functions_is_refused(monkeypatch):
    def missing(name, overload):
        raise RuntimeError(f"no {name}")

    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", missing)
    with pytest.raises(RuntimeError, match="built with csrc/exl3/dense_mma.cu"):
        dense_mma._library()


def test_resolved_once_by_ensure_extension_before_any_launch(monkeypatch):
    # Restored after the test: the module's real state.
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda fn: fn)
    order = []
    monkeypatch.setattr(dense_mma, "resolve", lambda: order.append("mma"))
    monkeypatch.setattr(
        exl3_ops, "_resolve_dense_splitk", lambda: order.append("splitk") or True
    )
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert order == ["mma", "splitk"], "resolved once per process"


def test_plan_rule_without_a_device():
    _extension_or_skip()
    plan = torch.ops._exl3_C.linear_mma_plan
    override = torch.ops._exl3_C.set_dense_mma_shape_override
    k, n, bits_x2, codebook = HEAD
    # The LM head has >= 8 four-tile column groups per SM: 4-tile groups over
    # all of K at every decode row count (N / 64 >= 8 x SMs up to 242 SMs).
    for rows in range(1, 33):
        assert plan(rows, k, n, SMS, bits_x2, codebook) == [4, 1, 1], rows
    assert plan(1, k, n, 242, bits_x2, codebook) == [4, 1, 1]
    # Outside the blocks: more than 32 rows, matrices outside the served set
    # (wide or not), other formats.
    for rows in (0, 33, 48, 64, 128, 3200):
        assert plan(rows, k, n, SMS, bits_x2, codebook) == [0, 0, 0], rows
    for padded_in, padded_out, other_bits in UNSERVED:
        for rows in (1, 8, 32):
            assert plan(rows, padded_in, padded_out, SMS, other_bits, 2) == [
                0,
                0,
                0,
            ], (padded_in, padded_out, other_bits, rows)
    for other in ((8, 2), (16, 2), (12, 1), (14, 1), (4, 1)):
        assert plan(8, k, n, SMS, *other) == [0, 0, 0], other
    with pytest.raises(RuntimeError, match="multiprocessor count"):
        plan(1, k, n, 0, bits_x2, codebook)
    with pytest.raises(RuntimeError, match="H128 multiples"):
        plan(1, k, 600, SMS, bits_x2, codebook)
    try:
        # The diagnostic override replaces the plan of every launch on the
        # blocks and of no other launch.
        override(2, 8, 2)
        assert plan(4, k, n, SMS, bits_x2, codebook) == [2, 8, 2]
        assert plan(33, k, n, SMS, bits_x2, codebook) == [0, 0, 0]
        assert plan(4, k, n, SMS, 8, 2) == [0, 0, 0]
        assert plan(4, 2560, WIDE_74, SMS, 14, 2) == [0, 0, 0]
        with pytest.raises(RuntimeError, match="plan override"):
            override(3, 8, 2)
        assert plan(4, k, n, SMS, bits_x2, codebook) == [2, 8, 2]
    finally:
        override(0, 0, 0)
    assert plan(4, k, n, SMS, bits_x2, codebook) == [4, 1, 1]


# ---------------------------------------------------------------- GPU tests


@pytest.fixture
def mma_switch():
    """The library switch, cleared (with any plan override) after the test."""
    _extension_or_skip()
    library = torch.ops._exl3_C
    yield library.set_dense_mma
    library.set_dense_mma(False)
    library.set_dense_mma_shape_override(0, 0, 0)


def _core(m, bits_x2, codebook=2):
    return real.core_weight(m["trellis"], bits_x2, codebook)


def _reference(m, core, x, logical_out):
    return real.apply_fp64(x, core, m["suh"], m["svh"])[:, :logical_out]


def _plan(rows, padded_in, padded_out, bits_x2, codebook=2):
    return torch.ops._exl3_C.linear_mma_plan(
        rows, padded_in, padded_out, _sms(), bits_x2, codebook
    )


@needs_sm80
@pytest.mark.parametrize("padded_in,padded_out,bits_x2", SERVED_DENSE)
def test_mma_holds_linear_outs_error_against_fp64(
    mma_switch, padded_in, padded_out, bits_x2
):
    codebook = 2
    m = _matrix(1000 * bits_x2 + padded_out, padded_in, padded_out, bits_x2)
    core = _core(m, bits_x2, codebook)
    for rows in (1, 3, 8, 9, 17, 24, 32):
        assert _plan(rows, padded_in, padded_out, bits_x2)[0] > 0, rows
        pool = _Pool()
        for seed in range(3):
            x = _inputs(100 * rows + seed, rows, padded_in)
            mma_switch(False)
            _, split_core = _run_core(
                exl3_ops.linear_out_splitk, m, x, bits_x2, padded_out, codebook
            )
            mma_switch(True)
            cand, cand_core = _run_core(
                exl3_ops.linear_out_splitk, m, x, bits_x2, padded_out, codebook
            )
            again = _run(
                exl3_ops.linear_out_splitk, m, x, bits_x2, padded_out, codebook
            )
            _bitwise(again, cand, f"rows {rows}: repeat call")
            # The tensor-core path ran: its FP32 result rows differ from the
            # split-K operator's sums.
            assert not torch.equal(cand_core, split_core), f"rows {rows}: no MMA"
            inc = _run(exl3_ops.linear_out, m, x, bits_x2, padded_out, codebook)
            pool.add(cand, inc, _reference(m, core, x, padded_out))
        pool.check(f"bits_x2={bits_x2} {padded_in}x{padded_out} rows {rows}")


@needs_sm80
@pytest.mark.parametrize("mt", [2, 4])
@pytest.mark.parametrize("kw", [1, 2, 4, 8])
@pytest.mark.parametrize("splits", [1, 2])
def test_every_forced_plan_holds_the_gate(mma_switch, mt, kw, splits):
    torch.ops._exl3_C.set_dense_mma_shape_override(mt, kw, splits)
    mma_switch(True)
    for padded_in, padded_out in ((2560, 640), (640, 2560)):
        m = _matrix(7 * padded_out + kw, padded_in, padded_out, 14)
        core = _core(m, 14)
        for rows in (1, 8, 32):
            assert _plan(rows, padded_in, padded_out, 14) == [mt, kw, splits]
            pool = _Pool()
            for seed in range(2):
                x = _inputs(10 * rows + seed, rows, padded_in)
                cand = _run(exl3_ops.linear_out_splitk, m, x, 14, padded_out)
                again = _run(exl3_ops.linear_out_splitk, m, x, 14, padded_out)
                _bitwise(again, cand, "repeat call")
                inc = _run(exl3_ops.linear_out, m, x, 14, padded_out)
                pool.add(cand, inc, _reference(m, core, x, padded_out))
            pool.check(
                f"forced {mt},{kw},{splits} {padded_in}x{padded_out} rows {rows}"
            )


@needs_sm80
@pytest.mark.parametrize(
    "rows,padded_in,padded_out,bits_x2,unaligned",
    [
        (33, 2560, 1280, 12, False),  # drafter fc rows (48-128 served)
        (4, 2560, 640, 8, False),  # a format outside the served set
        (4, 2560, 640, 14, True),  # 2-byte aligned trellis words
        # Matrices outside the served set:
        (1, 2560, WIDE_74, 14, False),  # wide (4-tile groups) at 74 SMs
        (4, 2560, 640, 12, False),  # a served shape in another served format
        (3, 640, 384, 14, False),
    ],
)
def test_launches_outside_the_blocks_are_the_split_k_operator(
    mma_switch, rows, padded_in, padded_out, bits_x2, unaligned
):
    m = _matrix(5 + rows, padded_in, padded_out, bits_x2, unaligned)
    x = _inputs(rows, rows, padded_in)
    mma_switch(False)
    split = _run(exl3_ops.linear_out_splitk, m, x, bits_x2, padded_out)
    mma_switch(True)
    _bitwise(
        _run(exl3_ops.linear_out_splitk, m, x, bits_x2, padded_out), split, "fallback"
    )


def _kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type.name == "CUDA"]


def _is_mma_kernel(name):
    return "exl3::dense_mma::" in name and "mma_kernel<" in name


@needs_sm80
def test_lm_head_runs_mma_kernel_not_gemm_kernel(monkeypatch, mma_switch):
    k, n, bits_x2, codebook = HEAD
    m = _matrix(31, k, n, bits_x2)
    scratch = torch.empty(exl3_ops.LINEAR_SCRATCH_ELEMENTS, device="cuda")
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: scratch)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", True)
    row_chunk = exl3_ops.linear_row_chunk(k, n)
    assert row_chunk >= 32

    def head(x):
        return exl3_ops._linear(
            x, m["trellis"], m["suh"], m["svh"], bits_x2, codebook, n, row_chunk
        )

    # Draft heads at rows c, the target head at rows 4c, c = 1..8.
    for rows in sorted({*range(1, 9), *range(4, 33, 4)}):
        x = _inputs(rows, rows, k)
        assert _plan(rows, k, n, bits_x2) == [4, 1, 1]
        mma_switch(False)
        names = _kernel_names(lambda x=x: head(x))
        assert any("gemm_kernel" in name for name in names), names
        mma_switch(True)
        names = _kernel_names(lambda x=x: head(x))
        assert any(_is_mma_kernel(name) for name in names), names
        assert any("linear_input_split" in name for name in names), names
        assert not any(
            "gemm_kernel" in name or "pipe_kernel" in name for name in names
        ), names


@needs_sm80
@pytest.mark.parametrize(
    "padded_in,padded_out,bits_x2,rows,row_chunk",
    [
        # Chunks of 8, 8 and 5 rows, all on the blocks: out slices at row
        # offsets 8 and 16.
        (2560, 256, 14, 21, 8),
        # The LM head's own row chunk (66 rows, the split-K operator's path)
        # and a 4-row tail on the blocks at row offset 66.
        (*HEAD[:3], None, None),
    ],
)
def test_mma_through_linear_on_row_chunk_tails(
    monkeypatch, mma_switch, padded_in, padded_out, bits_x2, rows, row_chunk
):
    if row_chunk is None:
        row_chunk = exl3_ops.linear_row_chunk(padded_in, padded_out)
        rows = row_chunk + 4
    m = _matrix(41, padded_in, padded_out, bits_x2)
    scratch = torch.empty(exl3_ops.LINEAR_SCRATCH_ELEMENTS, device="cuda")
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: scratch)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", True)
    x = _inputs(42, rows, padded_in)
    starts = list(range(0, rows, row_chunk))
    on_blocks = [s for s in starts if min(row_chunk, rows - s) <= 32]
    assert any(s > 0 for s in on_blocks), "a chunk on the blocks at an offset"
    mma_switch(True)

    def call():
        return exl3_ops._linear(
            x, m["trellis"], m["suh"], m["svh"], bits_x2, 2, padded_out, row_chunk
        )

    names = _kernel_names(call)
    assert sum(_is_mma_kernel(name) for name in names) == len(on_blocks), names
    out = call()
    torch.cuda.synchronize()
    for start in starts:
        chunk = x[start : start + row_chunk]
        if start in on_blocks:
            assert _plan(chunk.shape[0], padded_in, padded_out, bits_x2)[0] > 0
        _bitwise(
            out[start : start + row_chunk],
            _run(exl3_ops.linear_out_splitk, m, chunk, bits_x2, padded_out),
            f"rows {start}..{start + chunk.shape[0]} of {rows}",
        )


def _allocations():
    return torch.cuda.memory_stats()["allocation.all.allocated"]


@needs_sm80
@pytest.mark.parametrize(
    "rows,padded_out,bits_x2,expected",
    [
        (1, 256, 14, (2, None, 2)),  # 2-tile groups, K split over 2 blocks
        (32, 640, 14, (2, None, None)),  # 2-tile groups
        (8, HEAD[1], HEAD[2], (4, 1, 1)),  # the LM head: wide
    ],
)
def test_mma_graph_replay_is_bitwise_eager_without_allocations(
    mma_switch, rows, padded_out, bits_x2, expected
):
    plan = _plan(rows, 2560, padded_out, bits_x2)
    # The NaN-filled workspace below proves the captured zeroing only on a
    # plan that splits K over 2 blocks.
    assert all(e is None or p == e for p, e in zip(plan, expected)), plan
    m = _matrix(21, 2560, padded_out, bits_x2)
    x = _inputs(22, rows, 2560)
    out = torch.empty(rows, padded_out, dtype=torch.bfloat16, device=x.device)
    workspace = torch.empty(
        exl3_ops.linear_workspace_size(rows, 2560, padded_out), device=x.device
    )
    # resolve()'s warm-up: no capture below is an instantiation's first call.
    torch.ops._exl3_C.dense_mma_prepare()
    mma_switch(True)

    def call():
        exl3_ops.linear_out_splitk(
            x,
            m["trellis"],
            m["suh"],
            m["svh"],
            bits_x2,
            2,
            2560,
            padded_out,
            out,
            workspace,
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
    # New inputs in the captured buffer: the replay recomputes them.
    x2 = _inputs(23, rows, 2560)
    x.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(
        out,
        _run(exl3_ops.linear_out_splitk, m, x2, bits_x2, padded_out),
        "new inputs",
    )
