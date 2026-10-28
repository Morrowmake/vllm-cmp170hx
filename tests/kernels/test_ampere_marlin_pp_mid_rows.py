# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-expert compiled Marlin decode (VLLM_AMPERE_MARLIN_DECODE_CUDA) at 4..8
rows and for multi-request rows 9 up to the moe_routed_gemm/marlin_multi_max_rows
row of the sm80 tuning table (at most 64).

CPU: the regime at N = 2048 is M = 4..8 plus the multi-request rows, the TP4
regime unchanged; the former MID_ROWS and MULTI switches are retired and
change nothing; the banners go through the real vLLM logger.
GPU (sm_80 and the optional extension; skipped otherwise): every row's output
does not depend on M (rows of an M-row call equal the same rows of the 8-row
call) and stays inside the fp64 accuracy gate; CUDA-graph replay equals eager
with no allocation growth.
"""

import logging

import pytest
import torch

from vllm import envs
from vllm_sm80.kernels.marlin_moe import decode, extension
from tests.kernels import test_ampere_pp_marlin_prefill as pp_helpers
from tests.sm80.tuning_rows import use_rows

DECODE = "VLLM_AMPERE_MARLIN_DECODE_CUDA"
# The switches the whole-expert rows had before they joined the decode switch.
RETIRED = ("VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS",
           "VLLM_GLM5_MARLIN_DECODE_PP_MULTI")


@pytest.fixture(autouse=True)
def isolated_flags(monkeypatch):
    for name in (DECODE, "VLLM_SM80_MARLIN_PREFILL",
                 "VLLM_AMPERE_DECODE_KERNELS"):
        monkeypatch.setenv(name, "0")
    for name in RETIRED:
        monkeypatch.delenv(name, raising=False)
    use_rows(monkeypatch)
    monkeypatch.setattr(extension, "_OPS", None)


@pytest.mark.parametrize("value", ["0", "1"])
def test_retired_switches_change_nothing(monkeypatch, value):
    for name in RETIRED:
        assert name not in envs.environment_variables
        monkeypatch.setenv(name, value)
    assert [m for m in range(1, 70) if decode.compiled_regime(m, 2048)] == \
        list(range(4, 9)) + list(range(9, 33))


@pytest.mark.parametrize("tokens,width,released", [
    (1, 512, True), (5, 512, True), (32, 512, True), (33, 512, False),
    (0, 2048, False), (3, 2048, False), (4, 2048, True), (5, 2048, False),
    (6, 2048, False), (7, 2048, False), (8, 2048, True), (9, 2048, False),
    (16, 2048, False), (6, 1024, False),
])
def test_regime(monkeypatch, tokens, width, released):
    """Without multi-request rows: the released regime plus M = 5, 6, 7 at
    N = 2048."""
    _multi(monkeypatch, 0)
    expected = released or (width == 2048 and 5 <= tokens <= 7)
    assert decode.compiled_regime(tokens, width) is expected


@pytest.mark.parametrize("M,reason", [
    (6, "not sm_80"),  # past the regime: only the device check remains
    (3, "outside the compiled decode token/width regime"),
])
def test_gate_admits_mid_rows(monkeypatch, M, reason):
    monkeypatch.setenv(DECODE, "1")
    args = pp_helpers._meta_args(M=M, n=2048)
    layer = pp_helpers._layer(n=2048)
    assert reason in decode.gate_reason(layer, **args)
    output = torch.full((M, 4096), 5.0)
    assert not decode.maybe_apply(layer, output, **args)
    assert torch.equal(output, torch.full_like(output, 5.0))


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.mark.parametrize("width,logged", [(2048, True), (512, False)])
def test_warmup_banner_real_logger(monkeypatch, width, logged):
    """Whole experts name their rows once; TP4 shards have no such line."""
    from types import SimpleNamespace

    monkeypatch.setenv(DECODE, "1")
    _multi(monkeypatch, 64)
    monkeypatch.setattr(decode, "_is_sm80", lambda _device: True)
    monkeypatch.setattr(decode, "warmup", lambda *a, **k: None)
    from vllm.logger import _print_info_once

    _print_info_once.cache_clear()  # each case must log its own once-banner
    cfg = SimpleNamespace(n_routed_experts=288, num_experts_per_tok=8,
                          moe_intermediate_size=2048)
    tp = 2048 // width
    worker = SimpleNamespace(
        device="cuda:0",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=cfg, hf_config=cfg),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp)))
    handler = _Capture()
    lg = logging.getLogger(decode.logger.name)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        decode.warmup_from_worker(worker)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
    text = ("whole-expert compiled decode for multi-request rows 9..64 "
            "(moe_routed_gemm/marlin_multi_max_rows) and at 4..8 rows")
    assert any(text in line for line in handler.lines) is logged, handler.lines
    assert logged or not any("whole-expert" in line for line in handler.lines)


# ---------------------------------------------------------------- GPU ------

def _gpu_ok():
    import importlib.util
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (8, 0)
            and importlib.util.find_spec("vllm_sm80._ampere_marlin_C") is not None
            and torch.cuda.mem_get_info(0)[0] > 12 * 2**30)


gpu = pytest.mark.skipif(not _gpu_ok(),
                         reason="needs sm_80, the optional extension, ~12 GB free")
weights = pp_helpers.weights


def _run(layer, wd, x, tw, ids):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    out = torch.full_like(x, float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode.warmup(x.device, N=2048)
        decode.run(layer, out, x, wd["w1"], wd["w2"], tw, ids, MoEActivation.SILU)
    stream.synchronize()
    return out


@gpu
@pytest.mark.parametrize("M", [4, 5, 6, 7])
def test_rows_do_not_depend_on_m(weights, monkeypatch, M):
    from vllm_sm80.kernels.moe.align import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_MOE_DETERMINISTIC_ALIGN", "1")
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    x8, tw8, ids8 = pp_helpers._inputs(8, 900 + M)
    full = _run(layer, weights, x8, tw8, ids8)
    part = _run(layer, weights, x8[:M].contiguous(), tw8[:M].contiguous(),
                ids8[:M].contiguous())
    assert torch.equal(part, full[:M])
    rows = torch.arange(M, device=x8.device)
    ref = pp_helpers.reference_fp64(x8, weights, tw8, ids8, rows)
    released = pp_helpers._incumbent(weights, x8[:M].contiguous(), tw8[:M].contiguous(),
                                     ids8[:M].contiguous(), monkeypatch)
    cm, cx = pp_helpers._err(part, ref)
    im, ix = pp_helpers._err(released, ref)
    assert cm <= 1.10 * im and cx <= 1.25 * ix
    deterministic_moe_align_mode.cache_clear()


@gpu
@pytest.mark.parametrize("M", [5, 6, 7])
def test_graph_replay_mid_rows(weights, monkeypatch, M):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_sm80.kernels.moe.align import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_MOE_DETERMINISTIC_ALIGN", "1")
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    x, tw, ids = pp_helpers._inputs(M, 1300 + M)
    eager = _run(layer, weights, x, tw, ids)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # scratch is per stream: warm the capture stream
        decode.warmup(x.device, N=2048, max_tokens=M)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decode.run(layer, out, x, weights["w1"], weights["w2"], tw, ids,
                   MoEActivation.SILU)
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    assert torch.cuda.memory_allocated() == allocated
    deterministic_moe_align_mode.cache_clear()


# ------------------------------------- multi-request rows (CPU) -------------


def _multi(monkeypatch, hi):
    """The moe_routed_gemm/marlin_multi_max_rows row (None: no row)."""
    if hi is None:
        use_rows(monkeypatch)
    else:
        use_rows(
            monkeypatch,
            ("moe_routed_gemm", "marlin_multi_max_rows", {"gemm_n": 2048}, hi),
        )


@pytest.fixture
def multi_default(monkeypatch):
    use_rows(monkeypatch)


@pytest.mark.parametrize("hi,ranges", [
    (None, ((9, 32),)), (32, ((9, 32),)), (9, ((9, 9),)), (64, ((9, 64),)),
    (8, ()), (65, ()), (0, ()),
])
def test_multi_row_bounds(monkeypatch, hi, ranges):
    _multi(monkeypatch, hi)
    assert decode.pp_multi_ranges() == ranges


def test_multi_default_rows(multi_default):
    assert decode.multi_max_rows() == 32
    assert decode.pp_multi_ranges() == ((9, 32),)
    assert decode._scratch_tokens(2048) == 32
    assert decode._scratch_tokens(512) == 32
    assert decode.compiled_regime(32, 2048) and not decode.compiled_regime(33, 2048)


@pytest.mark.parametrize("hi,on", [
    (32, set(range(9, 33))),
    (16, set(range(9, 17))),
    (64, set(range(9, 65))),
    (8, set()), (65, set()),
])
def test_multi_regime(monkeypatch, hi, on):
    _multi(monkeypatch, hi)
    for m in range(1, 70):
        expected = 4 <= m <= 8 or m in on
        assert decode.compiled_regime(m, 2048) is expected, m
        assert decode.compiled_regime(m, 512) is (1 <= m <= 32), m


@pytest.mark.parametrize("hi,rows", [(None, 32), (32, 32), (40, 40), (64, 64),
                                     (8, 8)])
def test_multi_scratch_sizing(monkeypatch, hi, rows):
    from types import SimpleNamespace

    _multi(monkeypatch, hi)
    monkeypatch.setattr(decode, "_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda _device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert decode._scratch_tokens(2048) == rows
    ws = decode._workspaces(torch.device("cpu"), 2048, 4, create=True)
    assert ws["max_tokens"] == rows
    assert ws["part"].numel() == 4 * rows * 8 * 2 * 2048
    assert ws["c3"].numel() == rows * 8 * 4096


@pytest.mark.parametrize("width,hi,text", [
    (2048, 32, "whole-expert compiled decode for multi-request rows 9..32"),
    (2048, 8, "at 4..8 rows only: moe_routed_gemm/marlin_multi_max_rows = 8 "
     "is not within 9..64"),
])
def test_multi_banner_real_logger(monkeypatch, width, hi, text):
    from types import SimpleNamespace
    from vllm.logger import _print_info_once

    monkeypatch.setenv(DECODE, "1")
    _multi(monkeypatch, hi)
    monkeypatch.setattr(decode, "_is_sm80", lambda _device: True)
    monkeypatch.setattr(decode, "warmup", lambda *a, **k: None)
    _print_info_once.cache_clear()
    cfg = SimpleNamespace(n_routed_experts=288, num_experts_per_tok=8,
                          moe_intermediate_size=2048)
    worker = SimpleNamespace(
        device="cuda:0",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=cfg, hf_config=cfg),
            parallel_config=SimpleNamespace(tensor_parallel_size=2048 // width)))
    handler = _Capture()
    lg = logging.getLogger(decode.logger.name)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        decode.warmup_from_worker(worker)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
    assert any(text in line for line in handler.lines), handler.lines


# ------------------------------------- multi-request rows (GPU) -------------

@gpu
@pytest.mark.parametrize("M", [9, 16, 17, 32])
def test_multi_rows_do_not_depend_on_m(weights, monkeypatch, M):
    from vllm_sm80.kernels.moe.align import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_MOE_DETERMINISTIC_ALIGN", "1")
    _multi(monkeypatch, 64)
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    x, tw, ids = pp_helpers._inputs(64, 2100 + M)
    full = _run(layer, weights, x, tw, ids)
    part = _run(layer, weights, x[:M].contiguous(), tw[:M].contiguous(), ids[:M].contiguous())
    assert torch.equal(part, full[:M])
    rows = torch.arange(M, device=x.device)
    ref = pp_helpers.reference_fp64(x, weights, tw, ids, rows)
    released = pp_helpers._incumbent(weights, x[:M].contiguous(), tw[:M].contiguous(),
                                     ids[:M].contiguous(), monkeypatch)
    cm, cx = pp_helpers._err(part, ref)
    im, ix = pp_helpers._err(released, ref)
    assert cm <= 1.10 * im and cx <= 1.25 * ix
    deterministic_moe_align_mode.cache_clear()


@gpu
@pytest.mark.parametrize("M", [12, 32, 64])
def test_multi_graph_replay(weights, monkeypatch, M):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_sm80.kernels.moe.align import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_MOE_DETERMINISTIC_ALIGN", "1")
    _multi(monkeypatch, 64)
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    x, tw, ids = pp_helpers._inputs(M, 2500 + M)
    eager = _run(layer, weights, x, tw, ids)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # scratch is per stream: warm the capture stream
        decode.warmup(x.device, N=2048, max_tokens=M)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decode.run(layer, out, x, weights["w1"], weights["w2"], tw, ids,
                   MoEActivation.SILU)
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    assert torch.cuda.memory_allocated() == allocated
    deterministic_moe_align_mode.cache_clear()


# ---------------------- whole-expert scratch only after KV sizing (CPU) ----

@pytest.mark.parametrize("width,M", [(2048, 4), (2048, 6), (2048, 16), (512, 4), (512, 32)])
def test_decode_scratch_waits_for_kv_sizing(monkeypatch, width, M):
    from types import SimpleNamespace

    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setattr(decode, "_WORKSPACES", {})
    monkeypatch.setattr(decode, "_late_scratch_open", False)
    monkeypatch.setattr(decode, "gate_reason", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda _device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls = []
    monkeypatch.setattr(decode, "run", lambda *a, **k: calls.append(a))
    args = pp_helpers._meta_args(M=M, n=width, device="cpu")
    out = torch.zeros(M, 4096)

    def apply():
        return decode.maybe_apply(pp_helpers._layer(n=width), out, **args)

    # profile run / graph-memory profiling: released kernels, nothing allocated
    assert apply() is False and decode._WORKSPACES == {} and not calls
    decode.open_late_scratch()
    assert apply() is True and len(calls) == 1
    (ws,) = decode._WORKSPACES.values()
    assert ws["max_tokens"] == decode._scratch_tokens(width)


def test_kernel_warmup_opens_scratch_before_warmup():
    import inspect

    from vllm_sm80.kernels import warmup as sm80_warmup

    src = inspect.getsource(sm80_warmup.warmup_from_worker)
    assert 0 < src.index("open_late_scratch()") < src.index("warmup_from_worker(worker)")
