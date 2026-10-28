# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid EXL3 checkpoints (vllm_sm80/kernels/exl3/hybrid.py) on CPU.

The tiny hybrid checkpoint of ``hybrid_fixture`` (EXL3 routed experts, every
other module from a source in the official FP8 layout, built by
tools/sm80/build_hybrid_checkpoint.py) is built into the sm_80 target and
DSpark drafter per TP/EP rank and loaded by the EXL3 loader:

- every module takes the method its class should (EXL3 MoE, the FP8 method
  through Marlin W8A16, unquantized BF16, FP32-output BF16 compressor,
  grouped ``wo_a``) and every FP8 scale has the per-row [N, K/32] shape;
- the ledger accounts for every selected tensor once, the loaded FP8
  weights and expanded scales are the source's rows for the rank;
- a grouped ``wo_a`` cut into its groups projects as the official
  block-diagonal einsum of the dequantized weight;
- with the switch off, a hybrid config is refused and a plain EXL3 config
  builds the same config as before.

With ``LAB_INPUT_OFFICIAL_SHARD_3`` (the official shard holding layer 0), a
few real FP8 blocks loaded through the FP8 method dequantize bitwise as the
official convert.py dequantizes them.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from tests.models.deepseek_v41.sm80 import construct, hybrid_fixture
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)
from vllm.config import set_current_vllm_config


@pytest.fixture(scope="module")
def hybrid(tmp_path_factory):
    root = tmp_path_factory.mktemp("hybrid")
    return hybrid_fixture.write_hybrid(root, fixture.tiny_config())


@pytest.fixture
def hybrid_on(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_HYBRID", "1")


def _official_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The official convert.py dequantization of a block-scaled FP8 weight."""
    rows = weight.size(0) // scale.size(0)
    cols = weight.size(1) // scale.size(1)
    weight = (
        weight.unflatten(0, (-1, rows)).unflatten(-1, (-1, cols)).float()
        * scale[:, None, :, None].float()
    )
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a loaded FP8 layer: FP8 x its per-row UE8M0 scale per 32
    columns, in BF16 (the Marlin W8A16 kernel converts both to BF16; powers
    of two make the product exact)."""
    scales = scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    values = weight.to(torch.bfloat16).unflatten(-1, (-1, 32))
    return (values * scales.unsqueeze(-1)).flatten(-2)


# -- the built checkpoint ------------------------------------------------------


def test_builder_selects_one_copy_of_each_tensor(hybrid):
    directory, _, source = hybrid
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"])
    experts = {n for n in names if ".ffn.experts." in n}
    assert experts and all(
        n.endswith((".trellis", ".suh", ".svh", ".mul1")) for n in experts
    )
    assert not [n for n in names if ".engram.embed." in n]
    assert not [n for n in names - experts if n.endswith(".trellis")]
    assert names - experts == {
        n for n in source if ".ffn.experts." not in n and ".engram.embed." not in n
    }
    for path in directory.glob("*.safetensors"):
        assert path.is_symlink()
    quant = json.loads((directory / "config.json").read_text())["quantization_config"]
    assert quant == json.loads((directory / "quantization_config.json").read_text())
    assert quant["quant_method"] == "exl3"
    assert quant["non_routed_quantization"] == {
        **hybrid_fixture.SOURCE_QUANT,
        "quant_method": "deepseek_v4_fp8",
    }
    rules = quant["module_quantization"]["rules"]
    assert rules[0] == {"match": hybrid_fixture.EXL3_MODULES, "method": "exl3"}
    by_match = {r["match"]: r["method"] for r in rules}
    assert by_match[r"^layers\.\d+\.attn\.wq_a$"] == "deepseek_v4_fp8"
    assert by_match[r"^layers\.\d+\.attn\.compressor\.wkv$"] == "unquantized"
    assert by_match[r"^head$"] == "unquantized"
    assert "default" not in quant["module_quantization"]


def test_hybrid_config_needs_the_switch(monkeypatch, hybrid):
    from vllm_sm80.kernels.exl3.quant import Exl3Config

    monkeypatch.delenv("VLLM_EXL3_HYBRID", raising=False)
    quant = json.loads((hybrid[0] / "quantization_config.json").read_text())
    with pytest.raises(ValueError, match="VLLM_EXL3_HYBRID"):
        Exl3Config.from_config(quant)


@pytest.mark.parametrize("switch", ["0", "1"])
def test_plain_exl3_config_is_unchanged(monkeypatch, switch):
    from vllm_sm80.kernels.exl3.quant import Exl3Config

    monkeypatch.setenv("VLLM_EXL3_HYBRID", switch)
    config = Exl3Config.from_config(fixture.tiny_config()["quantization_config"])
    assert config.hybrid is None
    assert getattr(config, "weight_block_size", None) is None


def test_rules_refuse_an_unrouted_module(hybrid_on, hybrid):
    from vllm_sm80.kernels.exl3.hybrid import parse_hybrid

    quant = json.loads((hybrid[0] / "quantization_config.json").read_text())
    spec = parse_hybrid(quant)
    assert spec.method("layers.3.ffn.experts") == "exl3"
    assert spec.method("mtp.1.attn.wo_b") == "deepseek_v4_fp8"
    with pytest.raises(ValueError, match="No module_quantization rule"):
        spec.method("layers.3.attn.unknown")
    with pytest.raises(ValueError, match="different methods"):
        spec.method_for(["layers.2.attn.compressor.wkv", "layers.2.attn.wq_a"])


# -- every module's method, scales and loaded values ------------------------------


def _expected(name: str, module) -> str:
    """The method class a module of the hybrid checkpoint takes."""
    if name.startswith(("vision.", "aligner.")):
        # Built without a quantization config.
        return "UnquantizedLinearMethod"
    if re.search(r"\.ffn\.experts(\.routed_experts)?$", name):
        return "Exl3MoEMethod"
    if name.endswith(".wo_a"):
        return "GroupedLinearMethod"
    if name.endswith("compressor.fused_wkv_wgate"):
        two = len(module.output_sizes) == 2
        return "OutDtypeLinearMethod" if two else "UnquantizedLinearMethod"
    if name.endswith(
        (".indexer.wk", ".indexer.weights_proj", ".ffn.gate", "confidence_head.proj")
    ):
        return "UnquantizedLinearMethod"
    if name.endswith("lm_head"):
        return "UnquantizedEmbeddingMethod"
    if name.endswith(
        (
            ".fused_wqa_wkv",
            ".wq_b",
            ".wo_b",
            ".gate_up_proj",
            ".down_proj",
            ".engram.wkv",
            "main_proj",
        )
    ):
        return "ModelOptLinearMethod"
    raise AssertionError(f"unexpected quantized module {name}")


def _load(monkeypatch, model, config) -> dict[str, tuple[str, str]]:
    from vllm_sm80.kernels.exl3 import loader, ops, utils

    monkeypatch.setattr(ops, "ensure_extension", lambda: None)
    monkeypatch.setattr(loader, "require_cuda_parameters", lambda model: None)
    entries: dict[str, tuple[str, str]] = {}
    finish = utils.Exl3LoadLedger.finish

    def record(self):
        entries.update(self.entries)
        finish(self)

    with monkeypatch.context() as patch, set_current_vllm_config(config):
        patch.setattr(utils.Exl3LoadLedger, "finish", record)
        loader.load_exl3_weights(model, config.model_config)
    return entries


def _source_name(name: str) -> str:
    """Checkpoint module of a target/draft module name (tiny config: 8
    layers, the drafter's numbered after them)."""
    name = name.removeprefix("language_model.").removeprefix("model.")
    match = re.match(r"layers\.(\d+)\.(.*)", name)
    if match and int(match[1]) >= fixture.TEXT["num_hidden_layers"]:
        name = f"mtp.{int(match[1]) - fixture.TEXT['num_hidden_layers']}.{match[2]}"
    return name.replace(".shared_experts.down_proj", ".shared_experts.w2")


@pytest.mark.parametrize("tp", [1, 4])
def test_every_module_takes_its_method_and_loads(
    monkeypatch,
    meta_platform,  # noqa: F811
    hybrid_on,
    hybrid,
    tp,
):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm_sm80 import plugin

    directory, _, source = hybrid
    plugin._register_exl3()
    seen = defaultdict(set)
    for rank in range(tp):
        construct.set_groups(monkeypatch, tp, rank)
        construct.stub_engram_tables(monkeypatch, tp)
        target_config, draft_config = construct.vllm_configs(str(directory), tp)
        manifest = target_config.quant_config.manifest
        assert manifest.unselected > 0
        tensors = set(manifest.tensors)
        for config in (target_config, draft_config):
            model = construct.build_model(config)
            for name, module in model.named_modules():
                method = getattr(module, "quant_method", None)
                if method is None or not (
                    isinstance(module, LinearBase)
                    or name.endswith(("experts", "lm_head"))
                ):
                    continue
                kind = type(method).__name__
                assert kind == _expected(name, module), (name, kind)
                seen[kind].add(name.rsplit(".", 1)[-1])
                if kind == "GroupedLinearMethod":
                    assert type(method.inner).__name__ == "ModelOptLinearMethod"
                if kind == "OutDtypeLinearMethod":
                    assert method.out_dtype == torch.float32
                if kind in ("ModelOptLinearMethod", "GroupedLinearMethod"):
                    rows, cols = module.weight.shape
                    assert module.weight.dtype == torch.float8_e4m3fn
                    assert module.weight_scale.dtype == torch.uint8
                    assert tuple(module.weight_scale.shape) == (rows, cols // 32)
            ledger = _load(monkeypatch, model, config)
            # Every selected tensor is accounted for exactly once.
            assert set(ledger) == tensors
            if config is target_config:
                _check_values(model, source, tp, rank)
    assert {"Exl3MoEMethod", "ModelOptLinearMethod", "GroupedLinearMethod"} <= set(seen)
    assert {"OutDtypeLinearMethod", "UnquantizedLinearMethod"} <= set(seen)


def _check_values(model, source, tp, rank) -> None:
    """The loaded FP8 weights and expanded scales are the source's rows (or
    columns, row-parallel) for this rank."""
    modules = dict(model.named_modules())
    checks = {
        "language_model.model.layers.0.attn.wq_b": 0,
        "language_model.model.layers.0.attn.wo_b": 1,
        "language_model.model.layers.0.attn.wo_a": 0,
        "language_model.model.layers.1.engram.wkv": 0,
    }
    for name, dim in checks.items():
        module = modules[name]
        src = _source_name(name)
        weight, scale = source[f"{src}.weight"], source[f"{src}.scale"]
        scale = scale.view(torch.uint8).repeat_interleave(32, dim=0)
        width = weight.shape[dim] // tp
        part = slice(rank * width, (rank + 1) * width)
        scale_width = scale.shape[dim] // tp
        scale_part = slice(rank * scale_width, (rank + 1) * scale_width)
        if dim == 0:
            want_w, want_s = weight[part], scale[scale_part]
        else:
            want_w, want_s = weight[:, part], scale[:, scale_part]
        assert torch.equal(module.weight.view(torch.uint8), want_w.view(torch.uint8)), (
            name
        )
        assert torch.equal(module.weight_scale, want_s), name


# -- the grouped wo_a ------------------------------------------------------------


def test_grouped_wo_a_is_the_official_block_diagonal_einsum(monkeypatch):
    """Cut a loaded FP8 wo_a into its groups and project group by group (the
    Marlin repack and GEMM replaced by their CPU dequant equivalent): the
    result is the official einsum over the dequantized block-diagonal weight."""
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )
    from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

    construct.set_groups(monkeypatch, 1, 0)
    monkeypatch.setattr(
        MarlinMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, c=None: (True, None)),
    )
    processed = []
    monkeypatch.setattr(
        MarlinMxfp8LinearKernel,
        "process_weights_after_loading",
        lambda self, layer: processed.append(tuple(layer.weight.shape)),
    )
    monkeypatch.setattr(
        MarlinMxfp8LinearKernel,
        "apply_weights",
        lambda self, layer, x, bias=None: (
            x.double() @ _dequant(layer.weight, layer.weight_scale).double().t()
        ),
    )
    groups, rank_out, k = 2, 64, 96
    generator = torch.Generator().manual_seed(2)
    weight = (torch.rand(groups * rank_out, k, generator=generator) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    scale = torch.randint(
        118, 130, (groups * rank_out // 32, k // 32), generator=generator
    ).to(torch.uint8)

    layer = torch.nn.Module()
    method = ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
        CkptCtx(scale_block_size=(32, 32)),
    )

    def loader(param, loaded, *args):
        param.data.copy_(loaded)

    method.create_weights(
        layer, k, [groups * rank_out], k, groups * rank_out, torch.bfloat16,
        weight_loader=loader,
    )  # fmt: skip
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale.weight_loader(
        layer.weight_scale, scale.view(torch.float8_e8m0fnu)
    )
    layer.quant_method = method
    GroupedLinearMethod.adopt(layer, groups)
    layer.quant_method.process_weights_after_loading(layer)
    assert processed == [(rank_out, k)] * groups
    assert not hasattr(layer, "weight") and len(layer.grouped) == groups
    x = torch.randn(5, groups, k, generator=generator).bfloat16()
    got = layer.quant_method.apply_grouped(layer, x)
    official = _official_dequant(weight, scale.view(torch.float8_e8m0fnu))
    want = torch.einsum(
        "tgd,grd->tgr", x.double(), official.view(groups, rank_out, k).double()
    ).flatten(1)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_every_grouped_wo_a_runs_its_groups(
    monkeypatch,
    meta_platform,  # noqa: F811
    hybrid_on,
    hybrid,
):
    """Every grouped wo_a of the target and the drafter runs its groups as
    GEMMs of its own method (TP4: 2 local groups)."""
    from vllm_sm80 import plugin
    from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

    plugin._register_exl3()
    construct.set_groups(monkeypatch, 4, 0)
    construct.stub_engram_tables(monkeypatch, 4)
    methods = []
    for config in construct.vllm_configs(str(hybrid[0]), 4):
        model = construct.build_model(config)
        methods += [m.quant_method for n, m in model.named_modules()
                    if n.endswith(".wo_a")]  # fmt: skip
    assert methods and all(type(m) is GroupedLinearMethod for m in methods)
    assert {m.groups for m in methods} == {2}


@pytest.mark.parametrize(
    "flag,ab", [("1", ""), ("0", "dsv41_wo_a_grouped_gemm"), ("1", "dsv41_wo_a_grouped_gemm")]
)
def test_wo_a_grouped_gemm_switch_reaches_every_grouped_wo_a(
    monkeypatch,
    meta_platform,  # noqa: F811
    hybrid_on,
    hybrid,
    flag,
    ab,
):
    """VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM adopts every MXFP8 wo_a of the
    target and the drafter as the grouped GEMM (TP4: 2 local groups); the dev
    A/B switch keeps the per-group Marlin layers too and boots on the flag."""
    from vllm_sm80.models.deepseek_v41.ops.sm80_wo_a_grouped import (
        WoAGroupedGemmMethod,
    )
    from vllm_sm80 import plugin

    plugin._register_exl3()
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", flag)
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", ab)
    construct.set_groups(monkeypatch, 4, 0)
    construct.stub_engram_tables(monkeypatch, 4)
    methods = []
    for config in construct.vllm_configs(str(hybrid[0]), 4):
        model = construct.build_model(config)
        methods += [m.quant_method for n, m in model.named_modules()
                    if n.endswith(".wo_a")]  # fmt: skip
    assert methods and all(type(m) is WoAGroupedGemmMethod for m in methods)
    assert {m.groups for m in methods} == {2}
    assert {(m.gemm, m.keep_groups) for m in methods} == {(flag == "1", bool(ab))}


@pytest.mark.parametrize("device", ["meta", "cuda"])
def test_out_dtype_linear_keeps_the_fp32_sum(device):
    """FP32 output of BF16 x and weight (torch.mm out_dtype, a CUDA kernel;
    the meta device checks shape and dtype on CPU)."""
    from vllm_sm80.kernels.exl3.quant import OutDtypeLinearMethod, set_output_dtype

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda: the FP32-output GEMM runs only on a CUDA device")
    generator = torch.Generator().manual_seed(3)
    layer = torch.nn.Module()
    weight = torch.randn(48, 64, generator=generator).bfloat16()
    layer.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    layer.quant_method = UnquantizedLinearMethod()
    set_output_dtype(layer, torch.float32)
    assert type(layer.quant_method) is OutDtypeLinearMethod
    x = torch.randn(3, 7, 64, generator=generator).bfloat16()
    out = layer.quant_method.apply(layer, x.to(device))
    assert out.dtype == torch.float32 and out.shape == (3, 7, 48)
    if device == "cuda":
        want = (x.double() @ weight.double().t()).float()
        torch.testing.assert_close(out.cpu(), want, rtol=1e-6, atol=1e-6)


# -- real official blocks --------------------------------------------------------


def test_real_fp8_blocks_dequantize_as_official():
    """Real FP8 blocks (layer 0 wq_a and wo_a, the first 64 rows) loaded
    through the FP8 method's weight loaders dequantize bitwise as the
    official convert.py does."""
    from safetensors import safe_open

    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        KMxfp8Static,
    )

    path = os.environ.get("LAB_INPUT_OFFICIAL_SHARD_3")
    if not path or not Path(path).exists():
        pytest.skip("input: LAB_INPUT_OFFICIAL_SHARD_3 is not set")
    with safe_open(path, "pt") as shard:
        for name in ("layers.0.attn.wq_a", "layers.0.attn.wo_a"):
            weight = shard.get_slice(f"{name}.weight")[:64]
            scale = shard.get_slice(f"{name}.scale")[:2]
            assert weight.dtype == torch.float8_e4m3fn
            assert scale.dtype == torch.float8_e8m0fnu
            param = torch.empty(64, weight.shape[1] // 32, dtype=torch.uint8)
            load = KMxfp8Static.get_scale_weight_loader(
                lambda p, w: p.copy_(w), CkptCtx(scale_block_size=(32, 32))
            )
            load(param, scale)
            ours = _dequant(weight, param)
            assert torch.equal(
                ours.view(torch.int16),
                _official_dequant(weight, scale).view(torch.int16),
            ), name


def test_replay_uses_official_bytes_and_rejects_loader_mismatch(hybrid):
    from .hybrid_linear_cases import digest, official_parts

    directory, _, source = hybrid
    prefix = "layers.0.attn.wq_a"
    weight = source[prefix + ".weight"]
    scale = source[prefix + ".scale"].view(torch.uint8).repeat_interleave(32, 0)
    facts = {"shape": list(weight.shape), "sources": [(prefix, list(weight.shape))],
             "rank": 0, "tp": 1, "layer_type": "ReplicatedLinear", "groups": 1,
             "weight_sha256": digest(weight), "scale_sha256": digest(scale)}
    got, got_scale, exact = official_parts(directory, facts)
    assert torch.equal(got.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(got_scale, scale)
    assert torch.equal(exact.to(torch.bfloat16), _official_dequant(weight, source[prefix + ".scale"]))
    with pytest.raises(ValueError, match="loader weight differs"):
        official_parts(directory, dict(facts, weight_sha256="0" * 64))
