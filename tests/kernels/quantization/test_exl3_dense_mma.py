# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense tensor-core blocks (VLLM_EXL3_DENSE_MMA, csrc/exl3/dense_mma.cu).

CPU: split-K, SM80, SM-count and parallel-layout eligibility; diagnostic
override parsing; missing-library rejection; and device-independent plan
boundaries for both vocabulary-head widths and unsupported matrices.

GPU (SM 8.0 only): on the tensor-core path the result is held to
linear_out's error against an FP64 reference (pooled mean <= 1.10x, max <=
1.25x, BF16 rounding ties forgiven, exact where linear_out is exact) for
every served matrix, including both vocabulary-head widths, row counts
1-32 and every forced plan, and is deterministic;
launches outside the path (other matrices, formats, row counts) are the
split-K operator bit for bit; the LM head runs mma_kernel and neither
gemm_kernel nor pipe_kernel (profiler kernel names through exl3_linear's
body); row chunks of exl3_linear's body whose out is a slice at an offset
give each chunk's own launch bit for bit; a captured call replays bitwise
like the eager call without any allocator event.
"""

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
from vllm.sm80.kernels.exl3 import dense_mma
from vllm.sm80.kernels.exl3 import ops as exl3_ops

SMS = 74  # the multiprocessor count the plan rule was measured on
# Half-vocabulary TP2 and full-vocabulary PP2 heads, 6 bpw.
HEADS = ((2560, 124160, 12, 2), (2560, 248320, 12, 2))
# (padded_in, padded_out, bits_x2), codebook 2: the union of the TP2/PP1
# and TP1/PP2 dense projections in csrc/exl3/dense_mma.cu.
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
    (2560, 512, 14),  # PP2 attention k/v
    (2560, 2560, 12),  # PP2 drafter projections
    (6144, 2560, 14),  # PP2 output projections
    (2560, 10240, 14),  # PP2 linear-attention qkv
    (2560, 12288, 14),  # PP2 attention q
]
SERVED_SHAPES = [*SERVED_DENSE, *(head[:3] for head in HEADS)]
# N of a matrix just wide enough for the 4-tile plan at 74 SMs: not served.
WIDE_74 = 64 * 8 * SMS
# Matrices of the served formats that the blocks do not serve.
UNSERVED = [
    (2560, WIDE_74, 14),
    (2560, 640, 12),  # a served shape in another served format
    (2560, 1280, 14),
    (4224, 256, 14),
    (640, 384, 14),
]


# ---------------------------------------------------------------- CPU tests


def _config(tp=2, pp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        )
    )


@pytest.mark.parametrize(
    "splitk,capability,sms,config,allowed",
    [
        (True, (8, 0), SMS, _config(), True),
        (True, (8, 0), SMS, _config(tp=1, pp=2), True),
        (False, (8, 0), SMS, _config(tp=1, pp=2), False),
        (True, (8, 6), SMS, _config(tp=1, pp=2), False),
        (True, None, None, _config(), False),
        (True, (8, 0), 70, _config(tp=1, pp=2), False),
        (True, (8, 0), SMS, None, False),
        (True, (8, 0), SMS, _config(tp=1), False),
        (True, (8, 0), SMS, _config(pp=2), False),
        (True, (8, 0), SMS, _config(tp=4), False),
        (True, (8, 0), SMS, _config(tp=1, pp=4), False),
    ],
)
def test_gate_checks_device_split_k_and_parallel_layout(
    splitk, capability, sms, config, allowed
):
    assert (dense_mma.closed_reason(splitk, capability, sms, config) is None) == allowed


def test_shape_override_parsing():
    assert dense_mma.parse_shape_override("") is None
    assert dense_mma.parse_shape_override("2,8,2") == (2, 8, 2)
    assert dense_mma.parse_shape_override("4,1,1") == (4, 1, 1)
    for bad in ("3,1,1", "2,3,1", "2,8,3", "2,8", "2,8,2,1", "a,b,c", "4"):
        with pytest.raises(ValueError, match="MT,KW,SPLITS"):
            dense_mma.parse_shape_override(bad)


def test_library_without_the_functions_is_refused(monkeypatch):
    def missing(name, overload):
        raise RuntimeError(f"no {name}")

    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", missing)
    with pytest.raises(RuntimeError, match="built with csrc/exl3/dense_mma.cu"):
        dense_mma._library()


def test_plan_rule_without_a_device():
    _extension_or_skip()
    plan = torch.ops._exl3_C.linear_mma_plan
    override = torch.ops._exl3_C.set_dense_mma_shape_override
    # Both vocabulary widths use four-tile groups over all of K.
    for k, n, bits_x2, codebook in HEADS:
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
@pytest.mark.parametrize("padded_in,padded_out,bits_x2", SERVED_SHAPES)
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
        (128, 2560, 1280, 12, False),
        (963, 640, 384, 14, False),  # a prefill-sized launch
        (4, 2560, 640, 8, False),  # a format outside the served set
        (4, 2560, 640, 14, True),  # 2-byte aligned trellis words
        # Matrices outside the served set:
        (1, 2560, WIDE_74, 14, False),  # wide (4-tile groups) at 74 SMs
        (8, 2560, WIDE_74, 12, False),
        (4, 2560, 640, 12, False),  # a served shape in another served format
        (3, 640, 384, 14, False),
        (4, 4224, 256, 14, False),
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

    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    return [e.name for e in prof.events() if e.device_type.name == "CUDA"]


def _is_mma_kernel(name):
    return "exl3::dense_mma::" in name and "mma_kernel<" in name


@needs_sm80
@pytest.mark.parametrize("head_shape", HEADS)
def test_lm_head_runs_mma_kernel_not_gemm_kernel(monkeypatch, mma_switch, head_shape):
    k, n, bits_x2, codebook = head_shape
    m = _matrix(31, k, n, bits_x2)
    scratch = torch.empty(exl3_ops.linear_scratch_elements(), device="cuda")
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
        # Each head's own row chunk uses split-K, with a four-row MMA tail.
        (*HEADS[0][:3], None, None),
        (*HEADS[1][:3], None, None),
    ],
)
def test_mma_through_linear_on_row_chunk_tails(
    monkeypatch, mma_switch, padded_in, padded_out, bits_x2, rows, row_chunk
):
    if row_chunk is None:
        row_chunk = exl3_ops.linear_row_chunk(padded_in, padded_out)
        rows = row_chunk + 4
    m = _matrix(41, padded_in, padded_out, bits_x2)
    scratch = torch.empty(exl3_ops.linear_scratch_elements(), device="cuda")
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
    torch.accelerator.synchronize()
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
    return torch.accelerator.memory_stats()["allocation.all.allocated"]


@needs_sm80
@pytest.mark.parametrize(
    "rows,padded_in,padded_out,bits_x2,expected",
    [
        (1, 2560, 256, 14, (2, None, 2)),  # two blocks split K
        (32, 2560, 640, 14, (2, None, None)),
        (8, 6144, 2560, 14, (2, None, None)),  # full-width PP2 input
        (8, *HEADS[0][:3], (4, 1, 1)),
        (8, *HEADS[1][:3], (4, 1, 1)),
    ],
)
def test_mma_graph_replay_is_bitwise_eager_without_allocations(
    mma_switch, rows, padded_in, padded_out, bits_x2, expected
):
    plan = _plan(rows, padded_in, padded_out, bits_x2)
    # The NaN-filled workspace below proves the captured zeroing only on a
    # plan that splits K over 2 blocks.
    assert all(e is None or p == e for p, e in zip(plan, expected)), plan
    m = _matrix(21, padded_in, padded_out, bits_x2)
    x = _inputs(22, rows, padded_in)
    out = torch.empty(rows, padded_out, dtype=torch.bfloat16, device=x.device)
    workspace = torch.empty(
        exl3_ops.linear_workspace_size(rows, padded_in, padded_out), device=x.device
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
            padded_in,
            padded_out,
            out,
            workspace,
        )

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
    # New inputs in the captured buffer: the replay recomputes them.
    x2 = _inputs(23, rows, padded_in)
    x.copy_(x2)
    graph.replay()
    torch.accelerator.synchronize()
    _bitwise(
        out,
        _run(exl3_ops.linear_out_splitk, m, x2, bits_x2, padded_out),
        "new inputs",
    )
