# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-expert compiled Marlin decode at 5-7 rows (VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS) and
for multi-request rows 9..64 (VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS).

CPU: the regime with the flag off is the released one; on, it adds exactly
M = 5, 6, 7 at N = 2048; the banners go through the real vLLM logger.
GPU (sm_80 and the optional extension; skipped otherwise): the ``exact`` variant
equals the released Marlin kernels bitwise at M = 4..8; under ``orig`` every
row's output does not depend on M (rows of an M-row call equal the same rows of
the 8-row call) and stays inside the fp64 accuracy gate; CUDA-graph replay equals
eager with no allocation growth.
"""

import logging

import pytest
import torch

from vllm import ampere_marlin, envs
from vllm.ampere_decode import marlin_moe as decode
from tests.kernels import test_ampere_pp_marlin_prefill as pp_helpers

DECODE = "VLLM_GLM5_MARLIN_DECODE_CUDA"
VARIANT = "VLLM_GLM5_MARLIN_DECODE_VARIANT"
MID = "VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS"


@pytest.fixture(autouse=True)
def isolated_flags(monkeypatch):
    for name in (DECODE, "VLLM_GLM5_MARLIN_PREFILL_CUDA", "VLLM_GLM5_PP_MARLIN_PREFILL",
                 "VLLM_GLM5_TP4_MARLIN_PREFILL", "VLLM_GLM5_DECODE_KERNELS"):
        monkeypatch.setenv(name, "0")
    monkeypatch.delenv(VARIANT, raising=False)
    monkeypatch.delenv(MID, raising=False)
    monkeypatch.delenv("VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS", raising=False)
    monkeypatch.delenv("VLLM_GLM5_MARLIN_DECODE_PP_MULTI", raising=False)
    monkeypatch.setattr(ampere_marlin, "_OPS", None)


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS is False


@pytest.mark.parametrize("mid", [False, True])
@pytest.mark.parametrize("tokens,width,released", [
    (1, 512, True), (5, 512, True), (32, 512, True), (33, 512, False),
    (0, 2048, False), (3, 2048, False), (4, 2048, True), (5, 2048, False),
    (6, 2048, False), (7, 2048, False), (8, 2048, True), (9, 2048, False),
    (16, 2048, False), (6, 1024, False),
])
def test_regime(monkeypatch, mid, tokens, width, released):
    monkeypatch.setenv(MID, str(int(mid)))
    expected = released or (mid and width == 2048 and 5 <= tokens <= 7)
    assert decode.compiled_regime(tokens, width) is expected


@pytest.mark.parametrize("mid,reason", [
    (False, "outside the compiled decode token/width regime"),
    (True, "not sm_80"),  # past the regime: only the device check remains
])
def test_gate_admits_mid_rows_only_with_flag(monkeypatch, mid, reason):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(MID, str(int(mid)))
    args = pp_helpers._meta_args(M=6, n=2048)
    layer = pp_helpers._layer(n=2048)
    assert reason in decode.gate_reason(layer, **args)
    output = torch.full((6, 4096), 5.0)
    assert not decode.maybe_apply(layer, output, **args)
    assert torch.equal(output, torch.full_like(output, 5.0))


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.mark.parametrize("width,text", [
    (2048, "whole-expert compiled decode at 4..8 rows"),
    (512, "is set but the gate is closed (expert width N=512"),
])
def test_warmup_banner_real_logger(monkeypatch, width, text):
    from types import SimpleNamespace

    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(MID, "1")
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
    assert any(text in line for line in handler.lines), handler.lines


# ---------------------------------------------------------------- GPU ------

def _gpu_ok():
    import importlib.util
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (8, 0)
            and importlib.util.find_spec("vllm._ampere_marlin_C") is not None
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
@pytest.mark.parametrize("M", [4, 5, 6, 7, 8])
def test_exact_variant_equals_released(weights, monkeypatch, M):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, "exact")
    monkeypatch.setenv(MID, "1")
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    for seed in (1, 2, 3):
        x, tw, ids = pp_helpers._inputs(M, 700 + 10 * M + seed)
        got = _run(layer, weights, x, tw, ids)
        released = pp_helpers._incumbent(weights, x, tw, ids, monkeypatch)
        assert torch.equal(got, released), (M, seed)
    deterministic_moe_align_mode.cache_clear()


@gpu
@pytest.mark.parametrize("M", [4, 5, 6, 7])
def test_orig_rows_do_not_depend_on_m(weights, monkeypatch, M):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, "orig")
    monkeypatch.setenv(MID, "1")
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
@pytest.mark.parametrize("name", ["orig", "exact"])
@pytest.mark.parametrize("M", [5, 6, 7])
def test_graph_replay_mid_rows(weights, monkeypatch, name, M):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, name)
    monkeypatch.setenv(MID, "1")
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    x, tw, ids = pp_helpers._inputs(M, 1300 + M)
    eager = _run(layer, weights, x, tw, ids)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
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

MULTI = "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS"
MULTI_ON = "VLLM_GLM5_MARLIN_DECODE_PP_MULTI"


@pytest.fixture
def multi_off(monkeypatch):
    monkeypatch.delenv(MULTI, raising=False)
    monkeypatch.delenv(MULTI_ON, raising=False)


@pytest.mark.parametrize("raw,ranges", [
    ("", ()), ("9-32", ((9, 32),)), ("9-16, 24-32", ((9, 16), (24, 32))),
    ("12", ((12, 12),)), ("9-64", ((9, 64),)),
    ("8-16", None), ("9-65", None), ("16-9", None), ("a-b", None), ("9-", None),
])
def test_multi_parse(raw, ranges):
    assert decode._parse_ranges(raw) == ranges


def test_multi_defaults_off(multi_off):
    assert envs.VLLM_GLM5_MARLIN_DECODE_PP_MULTI is False
    assert envs.VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS == "9-32"
    assert decode.pp_multi_ranges() == ()
    assert decode._scratch_tokens(2048) == 8
    assert decode._scratch_tokens(512) == 32
    for m in range(9, 65):
        assert not decode.compiled_regime(m, 2048)


def test_multi_rows_alone_do_nothing(monkeypatch):
    monkeypatch.setenv(MULTI, "9-64")
    assert decode.pp_multi_ranges() == ()
    assert decode._scratch_tokens(2048) == 8


def test_multi_switch_default_rows(monkeypatch, multi_off):
    monkeypatch.setenv(MULTI_ON, "1")
    assert decode.pp_multi_ranges() == ((9, 32),)
    assert decode._scratch_tokens(2048) == 32
    assert decode.compiled_regime(32, 2048) and not decode.compiled_regime(33, 2048)


@pytest.mark.parametrize("raw,on", [
    ("9-32", set(range(9, 33))),
    ("9-16,24-32", set(range(9, 17)) | set(range(24, 33))),
    ("9-64", set(range(9, 65))),
    ("bad", set()), ("4-8", set()),
])
def test_multi_regime(monkeypatch, raw, on):
    monkeypatch.setenv(MULTI, raw)
    monkeypatch.setenv(MULTI_ON, "1")
    for m in range(1, 70):
        expected = m in (4, 8) or m in on
        assert decode.compiled_regime(m, 2048) is expected, m
        assert decode.compiled_regime(m, 512) is (1 <= m <= 32), m


@pytest.mark.parametrize("raw,rows", [("", 8), ("9-32", 32), ("9-16,24-40", 40), ("9-64", 64),
                                      ("bad", 8)])
def test_multi_scratch_sizing(monkeypatch, raw, rows):
    from types import SimpleNamespace

    monkeypatch.setenv(MULTI, raw)
    monkeypatch.setenv(MULTI_ON, "1")
    monkeypatch.setattr(decode, "_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda _device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert decode._scratch_tokens(2048) == rows
    ws = decode._workspaces(torch.device("cpu"), 2048, 4, create=True)
    assert ws["max_tokens"] == rows
    assert ws["part"].numel() == 4 * rows * 8 * 2 * 2048
    assert ws["c3"].numel() == rows * 8 * 4096


@pytest.mark.parametrize("width,raw,text", [
    (2048, "9-32", "whole-expert compiled decode for multi-request rows 9..32"),
    (2048, "4-8", "is set but the gate is closed ('4-8' is not lo-hi ranges"),
    (512, "9-32", "is set but the gate is closed (expert width N=512"),
])
def test_multi_banner_real_logger(monkeypatch, width, raw, text):
    from types import SimpleNamespace
    from vllm.logger import _print_info_once

    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(MULTI, raw)
    monkeypatch.setenv(MULTI_ON, "1")
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

MULTI_MS = [9, 12, 16, 17, 24, 32, 40, 64]


@gpu
@pytest.mark.parametrize("M", MULTI_MS)
def test_multi_exact_equals_released(weights, monkeypatch, M):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, "exact")
    monkeypatch.setenv(MULTI, "9-64")
    monkeypatch.setenv(MULTI_ON, "1")
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    for seed in (1, 2):
        x, tw, ids = pp_helpers._inputs(M, 1700 + 10 * M + seed)
        got = _run(layer, weights, x, tw, ids)
        released = pp_helpers._incumbent(weights, x, tw, ids, monkeypatch)
        assert torch.equal(got, released), (M, seed)
    deterministic_moe_align_mode.cache_clear()


@gpu
@pytest.mark.parametrize("M", [9, 16, 17, 32])
def test_multi_orig_rows_do_not_depend_on_m(weights, monkeypatch, M):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, "orig")
    monkeypatch.setenv(MULTI, "9-64")
    monkeypatch.setenv(MULTI_ON, "1")
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
@pytest.mark.parametrize("name", ["orig", "exact"])
@pytest.mark.parametrize("M", [12, 32, 64])
def test_multi_graph_replay(weights, monkeypatch, name, M):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, name)
    monkeypatch.setenv(MULTI, "9-64")
    monkeypatch.setenv(MULTI_ON, "1")
    deterministic_moe_align_mode.cache_clear()
    layer = pp_helpers._gpu_layer(weights)
    x, tw, ids = pp_helpers._inputs(M, 2500 + M)
    eager = _run(layer, weights, x, tw, ids)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
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
