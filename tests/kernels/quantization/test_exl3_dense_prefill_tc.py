# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense prefill on the tensor cores (VLLM_EXL3_DENSE_PREFILL_TC,
csrc/exl3/dense_prefill_tc.cu, vllm_sm80/kernels/exl3/dense_prefill_tc.py).

CPU: the switch and its schedule diagnostic default, the gate, the one-time
resolution with the active and gate-closed log lines (through vLLM's own
logger), resolution once per process by ensure_extension (with the gate open
through the real resolve and warm-up, which never call back into
ensure_extension), and _linear's operator per row chunk: with the switch
off exactly the base's operator;
with it on, linear_out_prefill only in prefill steps for launches the
operator covers and split-K does not split, never in a decode step of more
than 64 tokens, during a graph capture or without a forward context. The
library's host rule (linear_prefill_applies) and the served launches of the
3460 / 1906-token prefill chunks: the k/v 482-row tail keeps split-K.

GPU (SM 8.0, 74 multiprocessors): linear_out_prefill against an FP64
reference within linear_out's error (pooled mean <= 1.10x, max <= 1.25x,
BF16 rounding ties forgiven, exact where linear_out is exact) on whole-tile
and stream-K schedules; the k split ran (kernel names); bitwise
deterministic, also while another stream keeps multiprocessors busy (the
split tiles' last-arriving block changes, the sum order does not); the
whole-tile diagnostic; refusals; _linear end to end (off: linear_out; on:
each chunk the operator of its launch); graph replay bitwise like eager with
no allocator event and a NaN-filled workspace (the fixup counters are zeroed
inside the call); ensure_extension with the switch set at TP2 opens the gate
and warms up once.
"""

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
from tests.kernels.quantization.test_exl3_dense_splitk import (
    _allocations,
    _bitwise,
    _extension_or_skip,
    _inputs,
    _matrix,
    _Pool,
    _run,
    _run_core,
    needs_sm80,
)
from vllm_sm80.kernels.exl3 import dense_prefill_tc as tc
from vllm_sm80.kernels.exl3 import ops as exl3_ops

SMS = 74
DECODE_QUERY_LEN = 4  # one sampled token plus three drafts under verification
# (padded_in, padded_out, bits_x2) of the rank's served dense matrices.
KV, GATE, INDEXER, FC, DOWN, OUT, Z, QKV, Q = (
    (2560, 256, 14),
    (2560, 640, 14),
    (2560, 640, 10),
    (2560, 1280, 12),
    (640, 2560, 14),
    (3072, 2560, 14),
    (2560, 3072, 14),
    (2560, 5120, 14),
    (2560, 6144, 14),
)
# Every launch of the 8,826-token reference prompt in 3460 / 1906-token chunks
# (row chunk floor(8 Mi / (K + N))) -> the path with split-K on.
SERVED_LAUNCHES = [
    (KV, 2978, "tc"),
    (KV, 482, "splitk"),
    (KV, 1906, "tc"),
    (GATE, 2621, "tc"),
    (GATE, 839, "tc"),
    (GATE, 1906, "tc"),
    (INDEXER, 2621, "tc"),
    (INDEXER, 839, "tc"),
    (INDEXER, 1906, "tc"),
    (FC, 2184, "tc"),
    (FC, 1276, "tc"),
    (FC, 1906, "tc"),
    (FC, 736, "tc"),
    (FC, 1072, "tc"),
    (DOWN, 2621, "tc"),
    (DOWN, 839, "tc"),
    (DOWN, 1906, "tc"),
    (OUT, 1489, "tc"),
    (OUT, 482, "tc"),
    (OUT, 417, "tc"),
    (Z, 1489, "tc"),
    (Z, 482, "tc"),
    (Z, 417, "tc"),
    (QKV, 1092, "tc"),
    (QKV, 184, "tc"),
    (QKV, 814, "tc"),
    (Q, 963, "tc"),
    (Q, 571, "tc"),
    (Q, 943, "tc"),
]


def _attention(max_query_len):
    return SimpleNamespace(max_query_len=max_query_len)


def _step(monkeypatch, max_query_len):
    """A forward context whose scheduler gave its longest request
    max_query_len tokens (None: no forward context)."""
    context = (
        None
        if max_query_len is None
        else SimpleNamespace(
            attn_metadata={"layers.3.self_attn": _attention(max_query_len)}
        )
    )
    monkeypatch.setattr(forward_context, "_forward_context", context)


# ---------------------------------------------------------------- CPU tests


def test_switch_and_schedule_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC", raising=False)
    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC is False
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE == "planned"
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", "whole")
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC is True
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE == "whole"
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", "split")
    with pytest.raises(ValueError):
        _ = envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE
    assert set(tc.SCHEDULES) == {"planned", "whole"}


def _config(
    tp=2,
    pp=1,
    method="mtp",
    parallel_drafting=False,
    k=3,
    max_num_seqs=16,
    schedule=None,
):
    speculative = (
        None
        if method is None
        else SimpleNamespace(
            method=method,
            parallel_drafting=parallel_drafting,
            num_speculative_tokens=k,
            num_speculative_tokens_per_batch_size=schedule,
            adaptive_k=None,
        )
    )
    return SimpleNamespace(
        num_speculative_tokens=0 if method is None else k,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=speculative,
    )


def test_gate_opens_only_on_the_validated_configuration():
    config = _config()
    assert tc.closed_reason((8, 0), 74, config) is None
    assert tc.closed_reason((8, 0), 74, _config(method=None)) is None
    assert tc.closed_reason((8, 0), 74, _config(method="eagle3")) is None
    assert "(8, 6)" in tc.closed_reason((8, 6), 74, config)
    assert "None" in tc.closed_reason(None, None, config)
    assert "70 multiprocessors" in tc.closed_reason((8, 0), 70, config)
    assert "no vLLM config" in tc.closed_reason((8, 0), 74, None)
    assert tc.closed_reason((8, 0), 74, _config(tp=4)) is None
    # Other layouts give other matrices and row chunks.
    layouts = "not one of (2, 1), (4, 1)"
    assert f"(1, 1), {layouts}" in tc.closed_reason((8, 0), 74, _config(tp=1))
    assert f"(2, 2), {layouts}" in tc.closed_reason((8, 0), 74, _config(pp=2))
    # A drafter whose decode steps schedule more than 1 + k tokens per request
    # and can carry more than 64 rows. A block drafter (DFlash, DSpark) runs
    # its block in a forward pass of its own: 17 requests x (1 + 3) = 68.
    reason = tc.closed_reason((8, 0), 74, _config(method="dflash", max_num_seqs=17))
    assert "'dflash'" in reason and "carry 68 rows" in reason
    reason = tc.closed_reason((8, 0), 74, _config(parallel_drafting=True))
    assert "parallel drafting True" in reason
    # A block drafter whose forward passes never carry more than 64 rows: the
    # row floor keeps them on today's kernels, so the gate opens. 8 requests
    # x (1 + 5) = 48 rows with or without the batch-size schedule, and 8 x
    # (1 + 7) = 64 at seven drafts.
    dspark = dict(
        tp=4,
        method="dspark",
        parallel_drafting=True,
        k=5,
        max_num_seqs=8,
        schedule=[[1, 3, 5], [4, 8, 0]],
    )
    assert tc.closed_reason((8, 0), 74, _config(**dspark)) is None
    assert tc.closed_reason((8, 0), 74, _config(**dict(dspark, schedule=None))) is None
    assert tc.closed_reason((8, 0), 74, _config(**dict(dspark, schedule=None, k=7))) is None
    reason = tc.closed_reason((8, 0), 74, _config(**dict(dspark, schedule=None, k=8)))
    assert "'dspark'" in reason and "carry 72 rows" in reason


def test_only_validated_matrices_are_taken():
    _extension_or_skip()
    for padded_in, padded_out, bits_x2 in tc.VALIDATED_MATRICES:
        assert tc.takes_launch(1489, padded_in, padded_out, bits_x2, 2, False, SMS)
        assert not tc.takes_launch(1489, padded_in, padded_out, bits_x2, 1, False, SMS)
    # Full-width matrices of a one-card layout, and a validated shape stored
    # at another rate: the operator covers them, the gate does not.
    for padded_in, padded_out, bits_x2 in (
        (2560, 512, 14),
        (2560, 12288, 14),
        (2560, 2560, 14),
        (2560, 6144, 12),
    ):
        assert tc.linear_prefill_applies(2730, padded_in, padded_out, bits_x2, 2)
        assert not tc.takes_launch(2730, padded_in, padded_out, bits_x2, 2, False, SMS)


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


def test_resolution_logs_through_vllm_logger(monkeypatch, caplog):
    import vllm.logger as vllm_logger

    monkeypatch.setattr(tc, "logger", vllm_logger.init_logger(tc.__name__))
    for name in ("active", "_max_decode_query_len", "_sms", "_fake_registered"):
        monkeypatch.setattr(tc, name, getattr(tc, name))
    monkeypatch.setattr(tc, "_fake_registered", True)
    schedules = []
    monkeypatch.setattr(tc, "set_prefill_schedule", schedules.append)
    warmed = []
    monkeypatch.setattr(tc, "_warmup", warmed.append)
    monkeypatch.setattr(
        torch._C, "_dispatch_find_schema_or_throw", lambda name, overload: None
    )
    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    try:
        vllm_logger._print_info_once.cache_clear()
        monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC", raising=False)
        monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
        _device(monkeypatch, (8, 0))
        tc.resolve()
        assert tc.active is False and not caplog.records, "off: nothing logged"
        monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
        for capability, sms, config in (((8, 6), SMS, True), ((8, 0), 70, True)):
            _device(monkeypatch, capability, sms, config)
            tc.resolve()
            assert tc.active is False
        _device(monkeypatch, (8, 0), tp=1)
        tc.resolve()
        assert tc.active is False
        _device(monkeypatch, (8, 0), SMS, False)
        tc.resolve()
        assert tc.active is False and schedules == [] and warmed == []
        _device(monkeypatch, (8, 0))
        tc.resolve()
        assert tc.active is True and schedules == [0] and warmed == [SMS]
        assert tc._max_decode_query_len == 4 and tc._sms == SMS
    finally:
        root.removeHandler(caplog.handler)
    lines = [(r.levelno, r.name, r.getMessage()) for r in caplog.records]
    assert (
        logging.INFO,
        tc.__name__,
        "EXL3 dense prefill tensor-core path active (VLLM_EXL3_DENSE_PREFILL_TC): "
        "in prefill steps (a request with more than 4 tokens scheduled), dense "
        "launches of more than 64 rows and at least 16 output tiles of the 28 "
        "validated matrices (bits_x2 10/12/14, codebook 2) take "
        "linear_out_prefill, schedule planned; split-K launches, other matrices, "
        "decode steps and graph captures keep today's kernels.",
    ) in lines
    closed = sorted({m for _, _, m in lines if "gate is closed" in m})
    assert closed == [
        "VLLM_EXL3_DENSE_PREFILL_TC is set but the gate is closed: (tensor, "
        "pipeline) parallel size (1, 1), not one of (2, 1), (4, 1); every dense "
        "EXL3 launch keeps today's kernels.",
        "VLLM_EXL3_DENSE_PREFILL_TC is set but the gate is closed: 70 "
        "multiprocessors, not 74; every dense EXL3 launch keeps today's kernels.",
        "VLLM_EXL3_DENSE_PREFILL_TC is set but the gate is closed: device "
        "capability (8, 6), not (8, 0); every dense EXL3 launch keeps today's "
        "kernels.",
        "VLLM_EXL3_DENSE_PREFILL_TC is set but the gate is closed: no vLLM config "
        "is set, so the decode query length is unknown; every dense EXL3 launch "
        "keeps today's kernels.",
    ]


def test_a_stale_library_is_refused_with_the_rebuild_hint(monkeypatch):
    monkeypatch.setattr(tc, "active", False)
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    _device(monkeypatch, (8, 0))

    def missing(name, overload):
        raise RuntimeError(f"no schema {name}")

    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", missing)
    with pytest.raises(RuntimeError, match="build_standalone.py"):
        tc.resolve()
    assert tc.active is False


def test_switch_is_resolved_once_by_ensure_extension(monkeypatch):
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_fakes_registered", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda fn: fn)
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: False)
    resolved = []
    monkeypatch.setattr(tc, "resolve", lambda: resolved.append(1))
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert resolved == [1], "resolved once per process"


def test_ensure_extension_runs_the_open_gate_warmup_once(monkeypatch):
    """A process's first EXL3 use with the switch set and the gate open (TP2):
    ensure_extension runs the real resolve and warm-up once, and nothing they
    run calls back into ensure_extension, which marks the library loaded only
    after resolve returns. The library and the device are stood in for; the
    warm-up buffers live on the CPU."""
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_fakes_registered", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: False)
    for name in ("active", "_max_decode_query_len", "_sms"):
        monkeypatch.setattr(tc, name, getattr(tc, name))
    monkeypatch.setattr(tc, "_fake_registered", False)
    fakes = []
    monkeypatch.setattr(
        torch.library, "register_fake", lambda name: lambda fn: fakes.append(name)
    )
    monkeypatch.setattr(
        torch._C, "_dispatch_find_schema_or_throw", lambda name, overload: None
    )
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
    _device(monkeypatch, (8, 0))
    calls = []

    def workspace_size(*shape):
        calls.append(("workspace", shape))
        return 16

    def prefill(x, trellis, suh, svh, bits_x2, *rest):
        calls.append(("prefill", tuple(x.shape), bits_x2, rest[3].dtype))

    library = SimpleNamespace(
        set_prefill_schedule=lambda mode: calls.append(("schedule", mode)),
        linear_workspace_size=workspace_size,
        linear_out_prefill=prefill,
    )
    monkeypatch.setattr(torch.ops, "_exl3_C", library)
    for name in ("ones", "empty", "zeros"):
        make = getattr(torch, name)
        monkeypatch.setattr(
            torch, name, lambda *a, _make=make, device=None, **kw: _make(*a, **kw)
        )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    # A cached size would hide a call back into ensure_extension.
    exl3_ops.linear_workspace_size.cache_clear()
    resolve, resolved = tc.resolve, []
    monkeypatch.setattr(tc, "resolve", lambda: resolved.append(1) or resolve())
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert resolved == [1], "resolved once per process"
    assert exl3_ops._loaded and tc.active and tc._sms == SMS
    rows, padded_in, padded_out = tc.WARMUP_LAUNCH
    assert calls == [
        ("schedule", 0),
        ("workspace", (rows, padded_in, padded_out)),
        *(("prefill", (rows, padded_in), bits_x2, dtype)
          for bits_x2 in (10, 12, 14) for dtype in (torch.bfloat16, torch.float32)),
    ]
    assert fakes[-1] == "_exl3_C::linear_out_prefill"


def test_ensure_extension_retries_resolve_after_it_raised(monkeypatch):
    """A resolve that raised leaves the library unloaded; the next
    ensure_extension reaches resolve again instead of failing on the base fake
    implementations it registered the first time (torch refuses a second
    registration). The library is stood in for."""
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_fakes_registered", False, raising=False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: False)
    fakes = []

    def register_fake(name):
        if name in fakes:
            raise RuntimeError(f"{name} already has a fake impl registered")
        return lambda fn: fakes.append(name)

    monkeypatch.setattr(torch.library, "register_fake", register_fake)
    resolved = []

    def resolve():
        resolved.append(1)
        if len(resolved) == 1:
            raise RuntimeError("warm-up failed")

    monkeypatch.setattr(tc, "resolve", resolve)
    with pytest.raises(RuntimeError, match="warm-up failed"):
        exl3_ops.ensure_extension()
    assert not exl3_ops._loaded
    exl3_ops.ensure_extension()
    assert resolved == [1, 1] and exl3_ops._loaded
    assert "_exl3_C::linear_out" in fakes and len(fakes) == len(set(fakes))


def _recorders(monkeypatch):
    """Replace the three operators by recorders; returns the call list."""
    calls = []
    for module, name in (
        (exl3_ops, "linear_out"),
        (exl3_ops, "linear_out_splitk"),
        (tc, "linear_out_prefill"),
    ):
        monkeypatch.setattr(
            module, name, lambda *args, _name=name: calls.append((_name, args))
        )
    return calls


def _linear(monkeypatch, rows, matrix, row_chunk, splitk):
    padded_in, padded_out, bits_x2 = matrix
    workspace = torch.empty(16)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: workspace)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", splitk)
    trellis = torch.zeros(
        padded_in // 16, padded_out // 16, 8 * bits_x2, dtype=torch.int16
    )
    x = torch.zeros(rows, padded_in, dtype=torch.bfloat16)
    out = exl3_ops._linear(x, trellis, "suh", "svh", bits_x2, 2, padded_out, row_chunk)
    assert out.shape == (rows, padded_out)
    return trellis, workspace


@pytest.mark.parametrize("splitk", [False, True])
@pytest.mark.parametrize("max_query_len", [None, DECODE_QUERY_LEN, 3460])
def test_switch_off_keeps_the_base_operator(monkeypatch, splitk, max_query_len):
    calls = _recorders(monkeypatch)
    monkeypatch.setattr(tc, "active", False)
    monkeypatch.setattr(tc, "route", lambda *a: pytest.fail("route called while off"))
    _step(monkeypatch, max_query_len)
    trellis, workspace = _linear(monkeypatch, 3460, KV, 2978, splitk)
    expected = "linear_out_splitk" if splitk else "linear_out"
    assert [name for name, _ in calls] == [expected] * 2
    for _, args in calls:
        assert args[1] is trellis and args[9] is workspace
    assert [args[0].shape[0] for _, args in calls] == [2978, 482]


def _on(monkeypatch):
    monkeypatch.setattr(tc, "active", True)
    monkeypatch.setattr(tc, "_max_decode_query_len", DECODE_QUERY_LEN)
    monkeypatch.setattr(tc, "_sms", SMS)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.parametrize("splitk", [False, True])
@pytest.mark.parametrize(
    "max_query_len,rows,matrix",
    [
        # Decode steps of more than 64 tokens: 32 requests x 4 verify rows, and
        # the drafter's hidden-state projection at 4 x 64 verify rows.
        (DECODE_QUERY_LEN, 128, Q),
        (DECODE_QUERY_LEN, 256, FC),
        (1, 3460, OUT),
        # No forward context (profiling, tests, captures in piecewise mode).
        (None, 3460, OUT),
        # A prefill step's call of at most 64 rows.
        (3460, 64, Q),
    ],
)
def test_decode_steps_and_small_calls_keep_the_base_operator(
    monkeypatch, splitk, max_query_len, rows, matrix
):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    _step(monkeypatch, max_query_len)
    _linear(monkeypatch, rows, matrix, 4096, splitk)
    expected = "linear_out_splitk" if splitk else "linear_out"
    assert [name for name, _ in calls] == [expected]


def test_graph_capture_keeps_the_base_operator(monkeypatch):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _step(monkeypatch, 3460)
    _linear(monkeypatch, 1489, OUT, 1489, True)
    assert [name for name, _ in calls] == ["linear_out_splitk"]


@pytest.mark.parametrize(
    "splitk,expected",
    [
        # The k/v 482-row tail is a split-K launch: it keeps split-K.
        (True, ["linear_out_prefill", "linear_out_splitk"]),
        # 8 output tiles: below the operator's 16, so linear_out.
        (False, ["linear_out_prefill", "linear_out"]),
    ],
)
def test_prefill_step_routes_each_chunk_by_its_launch(monkeypatch, splitk, expected):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    _step(monkeypatch, 3460)
    trellis, workspace = _linear(monkeypatch, 3460, KV, 2978, splitk)
    assert [name for name, _ in calls] == expected
    assert [args[0].shape[0] for _, args in calls] == [2978, 482]
    for _, args in calls:
        assert args[1] is trellis and args[2:8] == ("suh", "svh", 14, 2, 2560, 256)
        assert args[8].shape == (args[0].shape[0], 256) and args[9] is workspace


def test_a_misaligned_trellis_keeps_the_base_operator(monkeypatch):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    _step(monkeypatch, 3460)
    padded_in, padded_out, bits_x2 = OUT
    base = torch.zeros(
        padded_in * padded_out // 256 * 8 * bits_x2 + 1, dtype=torch.int16
    )
    trellis = base[1:].view(padded_in // 16, padded_out // 16, 8 * bits_x2)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: torch.empty(16))
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    x = torch.zeros(1489, padded_in, dtype=torch.bfloat16)
    exl3_ops._linear(x, trellis, "suh", "svh", bits_x2, 2, padded_out, 1489)
    assert [name for name, _ in calls] == ["linear_out"]


def test_operator_rule_and_served_launches():
    _extension_or_skip()
    assert set(tc.OPERATORS) <= {
        name.split("::")[1].split(".")[0]
        for name in torch._C._dispatch_get_all_op_names()
        if name.startswith("_exl3_C::")
    }
    applies = tc.linear_prefill_applies
    # More than 64 rows, at least 16 output tiles, the covered formats.
    assert not applies(64, 2560, 2560, 14, 2)
    assert applies(65, 2560, 2048, 14, 2)  # 16 tiles
    assert not applies(65, 2560, 1920, 14, 2)  # 15 tiles
    assert not applies(1906, 2560, 128, 14, 2)  # 15 tiles
    assert applies(1921, 2560, 128, 14, 2)  # 16 tiles
    for bits_x2, codebook in ((10, 2), (12, 2), (14, 2)):
        assert applies(1489, 3072, 2560, bits_x2, codebook)
    for bits_x2, codebook in ((8, 2), (16, 2), (14, 1), (4, 1)):
        assert not applies(1489, 3072, 2560, bits_x2, codebook)
    with pytest.raises(RuntimeError):
        applies(1489, 3000, 2560, 14, 2)
    for (padded_in, padded_out, bits_x2), rows, path in SERVED_LAUNCHES:
        on = tc.takes_launch(rows, padded_in, padded_out, bits_x2, 2, True, SMS)
        off = tc.takes_launch(rows, padded_in, padded_out, bits_x2, 2, False, SMS)
        split = exl3_ops.linear_splitk_splits(rows, padded_out, SMS, bits_x2, 2) == 2
        got = "tc" if on else ("splitk" if split else "linear_out")
        assert got == path, (padded_in, padded_out, rows)
        # With split-K off every launch split-K does not take is unchanged.
        assert off == (
            on or (split and applies(rows, padded_in, padded_out, bits_x2, 2))
        )
    # The k/v tail without split-K: 8 tiles, so linear_out.
    assert not tc.takes_launch(482, 2560, 256, 14, 2, False, SMS)
    # Row chunks are the base's (the operator fits linear_workspace_size).
    assert exl3_ops.linear_row_chunk(2560, 256) == 2978
    assert exl3_ops.linear_row_chunk(3072, 2560) == 1489


# ---------------------------------------------------------------- GPU tests


def _gpu_or_skip():
    _extension_or_skip()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms != SMS:
        pytest.skip(f"{sms} multiprocessors: the schedule was validated at {SMS}")


def _prefill_kernels(fn):
    """Names of the prefill GEMM kernels fn launched."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if "prefill_gemm" in e.name}


def _split_ran(names):
    return any(", true>" in n or ",true>" in n or "(bool)1>" in n for n in names)


@needs_sm80
@pytest.mark.parametrize(
    "padded_in,padded_out,bits_x2,logical_in,logical_out,rows,split",
    [
        (2560, 256, 14, 2560, 256, 1906, False),  # k/v, 30 tiles: one wave
        (2560, 640, 10, 2560, 640, 839, False),  # indexer q/k, 35 tiles
        (3072, 2560, 14, 3072, 2560, 482, True),  # out/o_proj tail, 80 tiles
        (2560, 1280, 12, 2560, 1280, 1276, True),  # drafter fc, 100 tiles
        # Column-band splits (the whole-tile rectangle spans every row block).
        (2560, 3072, 14, 2560, 3072, 1300, True),  # linear-attention z, 264 tiles
        (2560, 640, 14, 2560, 640, 2200, True),  # shared gate/up, 90 tiles
        # q_proj, 74 tiles: 37 split blocks, as many as the workspace holds.
        (2560, 6144, 14, 2560, 6144, 200, True),
        (2560, 2048, 14, 2500, 1999, 200, False),  # cropped edges, 32 tiles
    ],
)
def test_prefill_holds_linear_outs_error_against_fp64(
    padded_in, padded_out, bits_x2, logical_in, logical_out, rows, split
):
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _gpu_or_skip()
    assert tc.linear_prefill_applies(rows, padded_in, padded_out, bits_x2, 2)
    m = _matrix(17 * bits_x2 + padded_out, padded_in, padded_out, bits_x2)
    weight = torch.from_numpy(effective_weight(*m["numpy"], bits_x2, 2)).cuda()
    weight = weight[:logical_in, :logical_out]
    pool = _Pool()
    for seed in range(2):
        x = _inputs(1000 + rows + seed, rows, logical_in)
        cand = _run(tc.linear_out_prefill, m, x, bits_x2, logical_out)
        names = _prefill_kernels(
            lambda x=x: _run(tc.linear_out_prefill, m, x, bits_x2, logical_out)
        )
        assert names, "the prefill GEMM did not run"
        assert _split_ran(names) == split, sorted(names)
        _bitwise(
            _run(tc.linear_out_prefill, m, x, bits_x2, logical_out), cand, "repeat"
        )
        inc = _run(exl3_ops.linear_out, m, x, bits_x2, logical_out)
        pool.add(cand, inc, x.double() @ weight)
    pool.check(f"{padded_in}x{padded_out} bits_x2={bits_x2} rows {rows}")


def _input_shapes():
    # The lab's shape-gate V2 (tools session.py plan shapes): a shape-gate-shapes/1
    # file, pinned input LAB_INPUT_SHAPES.
    path = os.environ.get("LAB_INPUT_SHAPES")
    if not path:
        return [
            pytest.param(
                0, 0, 0, marks=pytest.mark.skip(reason="LAB_INPUT_SHAPES not set")
            )
        ]
    with open(path) as f:
        doc = json.load(f)
    assert doc["schema"] == "shape-gate-shapes/1"
    return [tuple(s) for s in doc["shapes"]]


@needs_sm80
@pytest.mark.parametrize("rows", [482, 1906, 3460])
@pytest.mark.parametrize("padded_in,padded_out,bits_x2", _input_shapes())
def test_input_shapes_hold_linear_outs_error_against_fp64(
    padded_in, padded_out, bits_x2, rows
):
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _gpu_or_skip()
    # library arithmetic, not the VALIDATED_MATRICES table
    assert tc.linear_prefill_applies(rows, padded_in, padded_out, bits_x2, 2)
    m = _matrix(17 * bits_x2 + padded_out, padded_in, padded_out, bits_x2)
    weight = torch.from_numpy(effective_weight(*m["numpy"], bits_x2, 2)).cuda()
    pool = _Pool()
    for seed in range(2):
        x = _inputs(1000 + rows + seed, rows, padded_in)
        cand = _run(tc.linear_out_prefill, m, x, bits_x2, padded_out)
        names = _prefill_kernels(
            lambda x=x: _run(tc.linear_out_prefill, m, x, bits_x2, padded_out)
        )
        assert names, "the prefill GEMM did not run"
        _bitwise(
            _run(tc.linear_out_prefill, m, x, bits_x2, padded_out), cand, "repeat"
        )
        inc = _run(exl3_ops.linear_out, m, x, bits_x2, padded_out)
        pool.add(cand, inc, x.double() @ weight)
    pool.check(f"{padded_in}x{padded_out} bits_x2={bits_x2} rows {rows}")


@needs_sm80
@pytest.mark.parametrize(
    "padded_in,padded_out,rows",
    [
        (3072, 2560, 482),  # column band of 2 x 4 tiles
        (2560, 3072, 1300),  # column band of 4 x 11 tiles
        (2560, 640, 2200),  # column band of 1 x 18 tiles
        (2560, 6144, 200),  # column band of 11 x 2 tiles, 37 split blocks
        (2560, 1280, 1276),  # row band of 10 x 3 tiles
    ],
)
def test_stream_k_is_order_independent(padded_in, padded_out, rows):
    """A split tile's value is the sum of its segments' partials in k order,
    whichever block arrives last (the counter only elects the block that adds
    them). Busy multiprocessors change which block that is: the result must
    not change."""
    _gpu_or_skip()
    bits_x2 = 12 if padded_out == 1280 else 14
    m = _matrix(41 + rows, padded_in, padded_out, bits_x2)
    x = _inputs(42 + rows, rows, padded_in)

    def call():
        return _run(tc.linear_out_prefill, m, x, bits_x2, padded_out)

    assert _split_ran(_prefill_kernels(call))
    reference = call()
    for _ in range(10):
        _bitwise(call(), reference, "repeat")
    side = torch.cuda.Stream()
    a = torch.randn(2048, 2048, device="cuda")
    for delay in range(6):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(delay):
                a = a @ a / 2048
        out = call()
        torch.cuda.synchronize()
        _bitwise(out, reference, f"with {delay} matmuls beside it")


@needs_sm80
def test_whole_tile_diagnostic_runs_no_split():
    _gpu_or_skip()
    m = _matrix(51, 3072, 2560, 14)
    x = _inputs(52, 482, 3072)
    planned = _run(tc.linear_out_prefill, m, x, 14, 2560)
    try:
        tc.set_prefill_schedule(1)
        names = _prefill_kernels(lambda: _run(tc.linear_out_prefill, m, x, 14, 2560))
        whole = _run(tc.linear_out_prefill, m, x, 14, 2560)
    finally:
        tc.set_prefill_schedule(0)
    assert names and not _split_ran(names), sorted(names)
    # Only the split tiles' FP32 sums differ; the BF16 outputs stay close.
    assert (
        whole.float() - planned.float()
    ).abs().max() <= 2.0**-6 * planned.abs().max()
    with pytest.raises(RuntimeError):
        tc.set_prefill_schedule(2)


@needs_sm80
def test_uncovered_launches_are_refused():
    _gpu_or_skip()
    m = _matrix(61, 2560, 256, 14)
    for rows in (64, 482):  # the decode bound; 8 output tiles
        with pytest.raises(RuntimeError, match="does not cover"):
            _run(tc.linear_out_prefill, m, _inputs(rows, rows, 2560), 14, 256)
    unaligned = _matrix(62, 3072, 2560, 14, unaligned=True)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        _run(tc.linear_out_prefill, unaligned, _inputs(63, 482, 3072), 14, 2560)
    m = _matrix(64, 3072, 2560, 8)
    with pytest.raises(RuntimeError, match="does not cover"):
        _run(tc.linear_out_prefill, m, _inputs(65, 482, 3072), 8, 2560)


@needs_sm80
def test_ensure_extension_opens_the_gate_and_warms_up_on_the_card(monkeypatch):
    """A process's first EXL3 use with the switch set at TP2, on the card: the
    real library, gate and warm-up, resolved once without calling back into
    ensure_extension."""
    _gpu_or_skip()
    import vllm.config

    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", exl3_ops._dense_splitk)
    for name in ("active", "_max_decode_query_len", "_sms", "_fake_registered"):
        monkeypatch.setattr(tc, name, getattr(tc, name))
    # This process registered the operators' fake implementations already.
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda fn: fn)
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", _config)
    exl3_ops.linear_workspace_size.cache_clear()
    resolve, resolved = tc.resolve, []
    monkeypatch.setattr(tc, "resolve", lambda: resolved.append(1) or resolve())
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert resolved == [1] and exl3_ops._loaded
    assert tc.active and tc._sms == SMS
    assert tc._max_decode_query_len == DECODE_QUERY_LEN


@needs_sm80
@pytest.mark.parametrize("splitk", [False, True])
def test_linear_output_is_each_launchs_operator_bitwise(monkeypatch, splitk):
    _gpu_or_skip()
    m = _matrix(71, 2560, 256, 14)
    x = _inputs(72, 3460, 2560)
    scratch = torch.empty(exl3_ops.LINEAR_SCRATCH_ELEMENTS, device=x.device)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: scratch)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", splitk)
    default = exl3_ops.linear_out_splitk if splitk else exl3_ops.linear_out
    chunk = exl3_ops.linear_row_chunk(2560, 256)
    assert chunk == 2978

    def linear():
        return exl3_ops._linear(x, m["trellis"], m["suh"], m["svh"], 14, 2, 256, chunk)

    monkeypatch.setattr(tc, "active", False)
    _step(monkeypatch, 3460)
    off = linear()
    for start in (0, chunk):
        _bitwise(
            off[start : start + chunk],
            _run(default, m, x[start : start + chunk], 14, 256),
            "off",
        )
    _on(monkeypatch)
    on = linear()
    _bitwise(
        on[:chunk], _run(tc.linear_out_prefill, m, x[:chunk], 14, 256), "2978 rows"
    )
    _bitwise(on[chunk:], off[chunk:], "the 482-row tail keeps the base operator")
    _step(monkeypatch, DECODE_QUERY_LEN)
    _bitwise(linear(), off, "decode step")


@needs_sm80
def test_prefill_graph_replay_is_bitwise_eager_without_allocations():
    _gpu_or_skip()
    rows = 482
    m = _matrix(81, 3072, 2560, 14)
    x = _inputs(82, rows, 3072)
    out = torch.empty(rows, 2560, dtype=torch.bfloat16, device=x.device)
    workspace = torch.empty(
        exl3_ops.linear_workspace_size(rows, 3072, 2560), device=x.device
    )

    def call():
        tc.linear_out_prefill(
            x, m["trellis"], m["suh"], m["svh"], 14, 2, 3072, 2560, out, workspace
        )

    assert _split_ran(_prefill_kernels(call)), "the captured call must split k"
    torch.cuda.synchronize()
    before = _allocations()
    call()
    torch.cuda.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
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
    for _ in range(2):
        # NaN bits in the counters too: the call zeroes them itself.
        out.fill_(float("nan"))
        workspace.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _bitwise(out, eager, "replay vs eager")
    x.copy_(_inputs(83, rows, 3072))
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _run(tc.linear_out_prefill, m, x, 14, 2560), "new inputs")


def test_split_witness_matches_the_profiler_name_style():
    # The profiler demangles template arguments as true/false; cuobjdump as (bool)1.
    assert _split_ran({"void exl3::x::prefill_gemm<14, 2, __nv_bfloat16, true>(A, B*)"})
    assert not _split_ran(
        {"void exl3::x::prefill_gemm<14, 2, __nv_bfloat16, false>(A, B*)"}
    )
    assert _run_core is not None  # shared helper kept importable
