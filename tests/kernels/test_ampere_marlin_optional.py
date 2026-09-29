# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional extension startup contracts, guarded dispatch, and graph padding."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vllm import ampere_marlin, envs
from vllm.ampere_decode import marlin_moe as decode
from vllm.ampere_prefill import pp_marlin_prefill as prefill
from tests.kernels import test_ampere_pp_marlin_prefill as pp_helpers
from tests.kernels import test_ampere_tp4_marlin_prefill as tp_helpers

DECODE = "VLLM_GLM5_MARLIN_DECODE_CUDA"
PREFILL = "VLLM_GLM5_MARLIN_PREFILL_CUDA"


@pytest.fixture(autouse=True)
def isolated_optional_flags(monkeypatch):
    for name in (DECODE, PREFILL, "VLLM_GLM5_PP_MARLIN_PREFILL",
                 "VLLM_GLM5_TP4_MARLIN_PREFILL", "VLLM_GLM5_DECODE_IDX_GLUE"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(ampere_marlin, "_OPS", None)


def test_disabled_does_not_import_optional_library(monkeypatch):
    real_import = importlib.import_module

    def reject_optional(name, *args, **kwargs):
        if name == "vllm._ampere_marlin_C":
            pytest.fail("disabled optional library imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", reject_optional)
    importlib.reload(ampere_marlin)
    args = pp_helpers._meta_args(M=4, n=512)
    output = torch.full((4, 4096), 7.0)
    assert not decode.maybe_apply(pp_helpers._layer(n=512), output, **args)
    assert torch.equal(output, torch.full_like(output, 7.0))
    assert prefill.enabled_shapes() == {}


@pytest.mark.parametrize("decode_flag,prefill_flag", [(True, False), (False, True)])
def test_independent_flags(monkeypatch, decode_flag, prefill_flag):
    monkeypatch.setenv(DECODE, str(int(decode_flag)))
    monkeypatch.setenv(PREFILL, str(int(prefill_flag)))
    assert envs.VLLM_GLM5_MARLIN_DECODE_CUDA is decode_flag
    assert envs.VLLM_GLM5_MARLIN_PREFILL_CUDA is prefill_flag
    assert (2048 in prefill.enabled_shapes()) is prefill_flag


def test_missing_extension_is_actionable(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("optional module missing")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="VLLM_BUILD_AMPERE_MARLIN=1") as exc:
        ampere_marlin.require_extension()
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
    assert ampere_marlin._OPS is None


@pytest.mark.parametrize("key,bad", [
    ("abi_version", -1), ("torch_version", "0.0.0"),
    ("cuda_version", 99000), ("cxx11_abi", -1),
])
def test_incompatible_build_is_rejected_before_dispatch(monkeypatch, key, bad):
    info = ampere_marlin._runtime_build_info()
    info[key] = bad
    monkeypatch.setattr(importlib, "import_module",
                        lambda _name: SimpleNamespace(build_info=lambda: info))
    with pytest.raises(RuntimeError, match=key):
        ampere_marlin.require_extension()
    assert ampere_marlin._OPS is None


@pytest.mark.parametrize("tokens,width,expected", [
    (0, 512, False), (1, 512, True), (32, 512, True),
    (33, 512, False), (64, 512, False), (4, 2048, True),
    (8, 2048, True), (0, 2048, False), (1, 2048, False),
    (3, 2048, False), (5, 2048, False), (7, 2048, False),
    (9, 2048, False), (16, 2048, False), (32, 2048, False),
])
def test_decode_regime_boundaries(tokens, width, expected):
    assert decode.compiled_regime(tokens, width) is expected


@pytest.mark.parametrize("tokens,width,expected", [
    (383, 2048, False), (384, 2048, True), (2304, 2048, True),
    (2305, 2048, False), (2312, 2048, False),
    (1728, 512, False), (2304, 1024, False),
])
def test_prefill_regime_boundaries(tokens, width, expected):
    assert prefill.compiled_regime(tokens, width) is expected


@pytest.mark.parametrize("case,reason", [
    ("expert_map", "expert map"), ("fp8", "activation quantisation"),
    ("lora", "LoRA"), ("int64_ids", "int32"), ("bias", "w1_bias"),
])
def test_unsupported_decode_leaves_output_untouched(monkeypatch, case, reason):
    monkeypatch.setenv(DECODE, "1")
    args = pp_helpers._meta_args(M=4, n=512)
    layer = pp_helpers._layer(n=512)
    if case == "expert_map":
        args["expert_map"] = torch.empty(288, device="meta", dtype=torch.int32)
    elif case == "fp8":
        layer.input_dtype = torch.float8_e4m3fn
    elif case == "lora":
        layer._lora_context = object()
    elif case == "int64_ids":
        args["topk_ids"] = args["topk_ids"].to(torch.int64)
    elif case == "bias":
        layer.quant_config.w1_bias = torch.empty(1, device="meta")
    # Structural rejection must happen before the metadata-only device gate.
    assert reason in decode.gate_reason(layer, **args)
    output = torch.full((4, 4096), 3.0)
    assert not decode.maybe_apply(layer, output, **args)
    assert torch.equal(output, torch.full_like(output, 3.0))


def test_compiled_prefill_respects_released_threshold(monkeypatch):
    monkeypatch.setenv(PREFILL, "1")
    monkeypatch.setenv("VLLM_GLM5_PP_MARLIN_PREFILL", "1")
    monkeypatch.setenv("VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS", "1000")
    monkeypatch.setenv("VLLM_GLM5_TP4_MARLIN_PREFILL", "1")
    monkeypatch.setenv("VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS", "700")
    assert prefill.enabled_shapes()[2048][1] == 1000
    assert prefill.enabled_shapes()[512][1] == 700


@pytest.mark.parametrize("width,python_on,minimum,budget,expected", [
    (512, False, 384, 3460, (0, 0)),
    (512, True, 384, 3460, (3460, 0)),
    (2048, False, 384, 2312, (0, 2304)),
    (2048, True, 384, 2312, (2312, 2304)),
    (2048, False, 384, 383, (0, 0)),
    (2048, True, 2400, 3000, (3000, 0)),
    (1024, True, 384, 2312, (0, 0)),
])
def test_warmup_allocations_follow_admitted_width_and_tokens(
    monkeypatch, width, python_on, minimum, budget, expected,
):
    monkeypatch.setenv(PREFILL, "1")
    python_flag = (
        "VLLM_GLM5_TP4_MARLIN_PREFILL" if width == 512
        else "VLLM_GLM5_PP_MARLIN_PREFILL"
    )
    monkeypatch.setenv(python_flag, str(int(python_on)))
    monkeypatch.setenv(python_flag + "_MIN_TOKENS", str(minimum))
    assert prefill._warmup_capacities(width, budget) == expected


@pytest.mark.parametrize("width,requested,capacity", [
    (512, 4, 32), (2048, 4, 8), (2048, 64, 64),
])
def test_decode_scratch_reserves_only_admitted_capacity(
    monkeypatch, width, requested, capacity,
):
    # Exercise real allocation sizing on CPU, with only CUDA stream queries
    # replaced; PP4 must not inherit TP4's four-times-larger row reservation.
    monkeypatch.setattr(decode, "_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda _device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ws = decode._workspaces(torch.device("cpu"), width, requested, create=True)
    assert ws["max_tokens"] == capacity
    split_planes = 4 if width == 512 else 1
    assert ws["part"].numel() == split_planes * capacity * 8 * 2 * width
    assert ws["c3"].numel() == capacity * 8 * 4096


has_extension = importlib.util.find_spec("vllm._ampere_marlin_C") is not None
gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_extension,
    reason="requires CUDA and the optional prebuilt extension",
)
decode_weights = tp_helpers.hp.weights
prefill_weights = pp_helpers.weights


@gpu
@pytest.mark.parametrize("M,active", [(4, 4), (16, 12), (24, 20), (32, 28)])
def test_decode_accuracy_padding_and_graph(decode_weights, monkeypatch, M, active):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM 8.0 kernels")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    deterministic_moe_align_mode.cache_clear()
    hp = tp_helpers.hp
    layer = hp._gpu_layer(decode_weights)
    x, tw, ids = hp._inputs(M, M + 100)
    ids[active:] = -1
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode.warmup(x.device, max_tokens=M)
        decode.run(layer, out, x, decode_weights["w1"], decode_weights["w2"],
                   tw, ids, MoEActivation.SILU)
    stream.synchronize()
    expected = out.clone()
    incumbent = hp._incumbent(decode_weights, x, tw, ids, monkeypatch)
    rows = torch.arange(active, device=x.device)
    reference = hp.reference_fp64(x, decode_weights, tw, ids, rows)
    cm, cx = hp._err(expected[:active], reference)
    im, ix = hp._err(incumbent[:active], reference)
    assert cm <= 1.10 * im and cx <= 1.25 * ix
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decode.run(layer, out, x, decode_weights["w1"], decode_weights["w2"],
                   tw, ids, MoEActivation.SILU)
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[:active], expected[:active])
    assert torch.cuda.memory_allocated() == allocated
    deterministic_moe_align_mode.cache_clear()


@gpu
def test_prefill_wide_accuracy_and_graph_replay(prefill_weights):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM 8.0 kernels")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    hp = pp_helpers
    M = 384
    layer = hp._gpu_layer(prefill_weights)
    x, tw, ids = hp._inputs(M, 321)
    ws13 = torch.empty(M * 8, 2048, device=x.device, dtype=x.dtype)
    ws2 = torch.empty(M * 8 * 4096, device=x.device, dtype=x.dtype)
    buf = prefill._buffers(x.device, 288, M * 8, create=True)
    output = torch.empty_like(x)
    args = (layer, output, x, prefill_weights["w1"], prefill_weights["w2"],
            tw, ids, MoEActivation.SILU, ws13, ws2, buf)
    prefill.run(*args, wide=False)
    baseline = output.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prefill.run(*args, wide=True)
    stream.synchronize()
    eager = output.clone()
    rows = torch.arange(M, device=x.device)
    reference = hp.reference_fp64(x, prefill_weights, tw, ids, rows)
    candidate_mean, candidate_max = hp._err(eager, reference)
    baseline_mean, baseline_max = hp._err(baseline, reference)
    assert candidate_mean <= 1.10 * baseline_mean
    assert candidate_max <= 1.25 * baseline_max
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prefill.run(*args, wide=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, eager)


@gpu
def test_optional_scratch_is_separate_across_streams():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces, reductions = [], []
    device = torch.device("cuda:0")
    for stream in streams:
        with torch.cuda.stream(stream):
            decode.warmup(device)
            workspaces.append(decode._workspaces(device, 512, 32, create=False))
            reductions.append(prefill._reduce_scratch(device, create=True))
    for key in ("part", "h", "c3", "ctr", "sorted", "experts", "ntpp"):
        assert workspaces[0][key].data_ptr() != workspaces[1][key].data_ptr()
    assert reductions[0].data_ptr() != reductions[1].data_ptr()
    for stream in streams:
        stream.synchronize()
