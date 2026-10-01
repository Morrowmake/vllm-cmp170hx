# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional extension startup contracts, guarded decode dispatch, and graph padding.
The compiled prefill was removed: its old flag is recognised and ignored."""

import importlib
import importlib.util
import logging
from types import SimpleNamespace

import pytest
import torch

from vllm import ampere_marlin, envs
from vllm.ampere_decode import marlin_moe as decode
from vllm.ampere_prefill import pp_marlin_prefill as prefill
from tests.kernels import test_ampere_pp_marlin_prefill as pp_helpers
from tests.kernels import test_ampere_tp4_marlin_prefill as tp_helpers

DECODE = "VLLM_GLM5_MARLIN_DECODE_CUDA"
VARIANT = "VLLM_GLM5_MARLIN_DECODE_VARIANT"
PREFILL = "VLLM_GLM5_MARLIN_PREFILL_CUDA"


@pytest.fixture(autouse=True)
def isolated_optional_flags(monkeypatch):
    for name in (DECODE, PREFILL, "VLLM_GLM5_PP_MARLIN_PREFILL",
                 "VLLM_GLM5_TP4_MARLIN_PREFILL", "VLLM_GLM5_DECODE_IDX_GLUE"):
        monkeypatch.setenv(name, "0")
    monkeypatch.delenv(VARIANT, raising=False)
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


@pytest.mark.parametrize("value,warned", [("1", True), ("0", False), ("", False)])
def test_removed_prefill_flag_is_ignored(monkeypatch, value, warned):
    from vllm.logger import _print_warning_once

    monkeypatch.setenv(PREFILL, value)
    assert envs.VLLM_GLM5_MARLIN_PREFILL_CUDA == value
    assert PREFILL in envs.environment_variables  # no "Unknown vLLM environment variable"
    assert prefill.enabled_shapes() == {}
    assert not hasattr(prefill, "compiled_regime")
    assert "prefill_gemm" not in ampere_marlin._SCHEMA_ARGUMENTS
    _print_warning_once.cache_clear()
    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    lg = logging.getLogger(ampere_marlin.__name__)
    handler, level = Grab(), lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        ampere_marlin.note_removed_flags()
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
    assert any("VLLM_GLM5_MARLIN_PREFILL_CUDA=1 is ignored" in x for x in lines) is warned, lines


def test_decode_variant_defaults_to_orig_and_validates(monkeypatch):
    assert envs.VLLM_GLM5_MARLIN_DECODE_VARIANT == "orig"
    assert decode.variant() == "orig"
    monkeypatch.setenv(VARIANT, "exact")
    assert decode.variant() == "exact"
    monkeypatch.setenv(VARIANT, "fast")
    with pytest.raises(ValueError):
        decode.variant()


@pytest.mark.parametrize("name,suffix", [("orig", "_orig"), ("exact", "")])
def test_decode_variant_selects_operators(name, suffix):
    ops = SimpleNamespace(**{
        f"decode_{op}{s}": f"{op}{s}" for op in ("gemm", "act") for s in ("", "_orig")
    })
    assert decode._ops_for(ops, name) == (f"gemm{suffix}", f"act{suffix}")


@pytest.mark.parametrize("name,planes", [("orig", 4), ("exact", 1)])
def test_decode_scratch_holds_variant_partials(monkeypatch, name, planes):
    # Real allocation sizing on CPU with only CUDA stream queries replaced.
    monkeypatch.setenv(VARIANT, name)
    monkeypatch.setattr(decode, "_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda _device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ws = decode._workspaces(torch.device("cpu"), 512, 4, create=True)
    assert ws["part"].numel() == planes * 32 * 8 * 2 * 512


def test_older_library_without_orig_operators_is_rejected(monkeypatch):
    assert "decode_gemm_orig" in ampere_marlin._SCHEMA_ARGUMENTS
    assert "decode_act_orig" in ampere_marlin._SCHEMA_ARGUMENTS


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


def test_released_prefill_thresholds(monkeypatch):
    monkeypatch.setenv(PREFILL, "1")  # removed flag: changes nothing
    monkeypatch.setenv("VLLM_GLM5_PP_MARLIN_PREFILL", "1")
    monkeypatch.setenv("VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS", "1000")
    monkeypatch.setenv("VLLM_GLM5_TP4_MARLIN_PREFILL", "1")
    monkeypatch.setenv("VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS", "700")
    assert prefill.enabled_shapes()[2048][1] == 1000
    assert prefill.enabled_shapes()[512][1] == 700


@pytest.mark.parametrize("width,python_on,budget,expected", [
    (512, False, 3460, 0), (512, True, 3460, 3460),
    (2048, False, 2312, 0), (2048, True, 2312, 2312), (1024, True, 2312, 0),
])
def test_warmup_allocations_follow_admitted_width(monkeypatch, width, python_on, budget,
                                                  expected):
    monkeypatch.setenv(PREFILL, "1")  # removed flag: allocates nothing by itself
    python_flag = (
        "VLLM_GLM5_TP4_MARLIN_PREFILL" if width == 512
        else "VLLM_GLM5_PP_MARLIN_PREFILL"
    )
    monkeypatch.setenv(python_flag, str(int(python_on)))
    assert prefill._warmup_capacity(width, budget) == expected


has_extension = importlib.util.find_spec("vllm._ampere_marlin_C") is not None
gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_extension,
    reason="requires CUDA and the optional prebuilt extension",
)
decode_weights = tp_helpers.hp.weights


@gpu
@pytest.mark.parametrize("name", ["orig", "exact"])
@pytest.mark.parametrize("M,active", [(4, 4), (16, 12), (24, 20), (32, 28)])
def test_decode_accuracy_padding_and_graph(decode_weights, monkeypatch, M, active,
                                           name):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM 8.0 kernels")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv(VARIANT, name)
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
def test_optional_scratch_is_separate_across_streams():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = []
    device = torch.device("cuda:0")
    for stream in streams:
        with torch.cuda.stream(stream):
            decode.warmup(device)
            workspaces.append(decode._workspaces(device, 512, 32, create=False))
    for key in ("part", "h", "c3", "ctr", "sorted", "experts", "ntpp"):
        assert workspaces[0][key].data_ptr() != workspaces[1][key].data_ptr()
    for stream in streams:
        stream.synchronize()
