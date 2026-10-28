# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official-weight hybrid arithmetic and linked successor replay.

The reference is BF16 GEMM on the independently decoded official weight, never
EXL3 output or an analytical tolerance. All loader fingerprints precede repacking.
Consumer sources are exported with byte-checked producer views; absent links fail.
"""

import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

BF16 = torch.bfloat16


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def decode_fp8(weight, scale):
    """Independent E4M3FN and UE8M0 bit definitions, in FP64."""
    bits = weight.view(torch.uint8).to(torch.int32)
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    if bool(((exponent == 15) & (mantissa == 7)).any()):
        raise ValueError("nonfinite official FP8 weight")
    values = torch.where(exponent == 0, mantissa.double() / 8,
                         1 + mantissa.double() / 8)
    values = torch.ldexp(values, torch.where(exponent == 0, -6, exponent - 7))
    values = torch.where((bits & 128) != 0, -values, values)
    exp = scale.view(torch.uint8).to(torch.int32)
    if bool((exp == 255).any()):
        raise ValueError("nonfinite official UE8M0 scale")
    factors = torch.ldexp(torch.ones_like(exp, dtype=torch.float64), exp - 127)
    return (values.unflatten(-1, (-1, 32)) * factors.unsqueeze(-1)).flatten(-2)


def official_parts(checkpoint, facts):
    """Read the independently computed TP/group slice from original tensors."""
    root = Path(checkpoint)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    local_n, local_k = facts["shape"]
    rank, tp = facts["rank"], facts["tp"]
    weights, scales = [], []
    for prefix, shape in facts["sources"]:
        n, k = shape
        row_parallel = local_k != k
        if row_parallel:
            if local_k * tp != k:
                raise ValueError("input shard is not the recorded TP partition")
            rows, cols = slice(0, n), slice(rank * local_k, (rank + 1) * local_k)
        else:
            cols = slice(0, k)
            if facts.get("head_slice") is not None:
                rows = slice(*facts["head_slice"])
            elif "Replicated" in facts["layer_type"] or tp == 1:
                rows = slice(0, n)
            else:
                if n % tp:
                    raise ValueError("output shard does not divide TP")
                rows = slice(rank * (n // tp), (rank + 1) * (n // tp))
        with safe_open(str(root / index[prefix + ".weight"]), framework="pt") as f:
            w = f.get_slice(prefix + ".weight")[rows, cols]
        weights.append(w)
        if w.dtype == torch.float8_e4m3fn:
            if any(v % 32 for v in (rows.start, rows.stop, cols.start, cols.stop)):
                raise ValueError("official FP8 slice cuts a scale block")
            with safe_open(str(root / index[prefix + ".scale"]), framework="pt") as f:
                s = f.get_slice(prefix + ".scale")[
                    rows.start // 32:rows.stop // 32, cols.start // 32:cols.stop // 32]
            scales.append(s.view(torch.uint8).repeat_interleave(32, 0))
    weight = torch.cat(weights, dim=0)
    scale = torch.cat(scales, dim=0) if scales else None
    if facts.get("head_slice") is not None and weight.shape[0] < local_n:
        # This is checkpoint vocabulary padding, not a captured-input transformation.
        weight = torch.nn.functional.pad(weight, (0, 0, 0, local_n - weight.shape[0]))
    if list(weight.shape) != facts["shape"]:
        raise ValueError("official TP slice shape differs from actual loaded weight")
    if digest(weight) != facts["weight_sha256"]:
        raise ValueError("actual hybrid loader weight differs from official TP slice")
    if (digest(scale) if scale is not None else None) != facts["scale_sha256"]:
        raise ValueError("actual hybrid loader scale differs from official TP slice")
    exact = decode_fp8(weight, scale) if scale is not None else weight.double()
    if not torch.isfinite(exact).all():
        raise ValueError("nonfinite official decoded weight")
    return weight, scale, exact


# One coupled attention call loads four projections (wq_b, index_wq_b, wk, wo_a) of its layer.
@functools.lru_cache(maxsize=8)
def prepared(checkpoint, encoded, device):
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.modelopt import CkptCtx, ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic, kMxfp8Static
    from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

    facts = json.loads(encoded)
    weight, scales, exact = official_parts(checkpoint, facts)
    n, k = weight.shape
    layer = torch.nn.Module()
    layer.register_parameter("bias", None)
    if scales is not None:
        method = ModelOptLinearMethod(QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
                                      CkptCtx(scale_block_size=(32, 32)))
        method.input_dtype = method.out_dtype = BF16

        def load(param, source, *args):
            param.data.copy_(source)

        from pytest import MonkeyPatch
        from tests.models.deepseek_v41.sm80.construct import set_groups

        with MonkeyPatch.context() as patch, torch.device(device):
            set_groups(patch, 1, 0)
            method.create_weights(layer, k, [n], k, n, BF16, weight_loader=load)
        if type(method.kernel).__name__ != "MarlinMxfp8LinearKernel":
            raise ValueError("hybrid candidate did not dispatch Marlin W8A16")
        layer.weight.weight_loader(layer.weight, weight.to(device))
        layer.weight_scale.weight_loader(layer.weight_scale,
                                         scales[::32].to(device).view(torch.float8_e8m0fnu))
        if digest(layer.weight.detach().cpu()) != facts["weight_sha256"]:
            raise ValueError("replay weight loader changed official bytes")
        if digest(layer.weight_scale.detach().cpu()) != facts["scale_sha256"]:
            raise ValueError("replay scale loader changed official bytes")
        layer.quant_method = method
        if facts.get("grouped", False):
            GroupedLinearMethod.adopt(layer, facts["groups"])
        layer.quant_method.process_weights_after_loading(layer)
    else:
        layer.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
        layer.quant_method = UnquantizedLinearMethod()
    return layer, exact


def matmul64(x, weight, device, groups=1):
    chunks = []
    if groups > 1:
        width = weight.shape[0] // groups
        return torch.cat([matmul64(x[:, g], weight[g * width:(g + 1) * width], device)
                          for g in range(groups)], -1)
    # Keep the reference weight spool bounded, including the vocabulary head.
    xd = x.to(device=device, dtype=torch.float64)
    for start in range(0, weight.shape[0], 2048):
        chunks.append(xd @ weight[start:start + 2048].to(device=device, dtype=torch.float64).t())
    return torch.cat(chunks, -1)


def linear_outputs(a, device, meta, checkpoint):
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm_sm80.kernels.exl3.quant import OutDtypeLinearMethod

    facts = meta["linear"]
    layer, w64 = prepared(checkpoint, json.dumps(facts, sort_keys=True), str(device))
    x = a["x"].to(device)
    if x.dtype != BF16 or x.shape[-1] != w64.shape[-1]:
        raise ValueError("captured activation dtype/width mismatch; no slicing or padding")
    grouped = meta["op"] == "grouped"
    groups = facts.get("groups", 1) if grouped else 1
    dtype = a["out"].dtype

    def cand(inp):
        if meta["op"] == "head":
            return LogitsProcessor._apply_head(SimpleNamespace(head_dtype=torch.float32), layer, inp, None)
        if grouped:
            return layer.quant_method.apply_grouped(layer, inp)
        if dtype == torch.float32:
            method = OutDtypeLinearMethod(torch.float32)
            return method.apply(layer, inp)
        return layer.quant_method.apply(layer, inp)

    def ref(inp):
        if grouped:
            width = w64.shape[0] // groups
            return torch.cat([torch.mm(inp[:, g].contiguous(),
                w64[g * width:(g + 1) * width].to(device=device, dtype=BF16).t())
                for g in range(groups)], -1)
        weight = w64.to(device=device, dtype=BF16)
        return (torch.mm(inp, weight.t(), out_dtype=torch.float32) if dtype == torch.float32
                else torch.mm(inp, weight.t()))

    exact_x = x[:, 0] if grouped and groups == 1 else x
    return cand(x), ref(x), matmul64(exact_x, w64, device, groups), cand


def reference_settings():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    return {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False, "float32_matmul_precision": "highest"}


def hybrid_linear(a, device, src):
    from tests.models.deepseek_v41.sm80.fp64_cases import Call, bits_equal, picks

    reference_settings()
    cand, ref, exact, apply = linear_outputs(a, device, src.meta, src.checkpoint["hybrid"])
    if not bits_equal(cand.cpu(), a["out"]):
        raise ValueError("hybrid projection replay differs from its captured output")
    invariant = all(bits_equal(cand[i:i + 1], apply(a["x"][i:i + 1].to(device)))
                    for i in picks(cand.shape[0]))
    if not invariant:
        raise ValueError("hybrid projection violates row invariance")
    if not all(torch.isfinite(v).all() for v in (cand, ref, exact)):
        raise ValueError("nonfinite hybrid projection result")
    # The same owned input must also survive repeated graph replay.
    inp = a["x"].to(device)
    apply(inp)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = apply(inp)
    for _ in range(2):
        graph.replay()
        if not bits_equal(replay, cand):
            raise ValueError("hybrid projection graph replay changed output bits")
    return Call(cand, ref, exact, invariant)
