# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp EXL3 layout keeps the checkpoint name map, tensor ownership
and quant method choices of the EXL3 config.

``fixtures/exl3_names.json`` holds the tensor names of the Qwen4Exp EXL3
checkpoint index, reduced to decoder layers 0, 1, 3 and 47, routed experts 0
and 511 and vision blocks 0 and 26. For every module prefix the target and
MTP draft configs can be asked about, :data:`EXPECTED` pins a digest of what
the EXL3 config gave before the Qwen4Exp parts moved behind the layout hooks:
the checkpoint source, the quant method chosen for linear layers of 0, 3 and
4 outputs and for an LM head (with its matrices and uneven sharding), the
routed expert matrices, the ownership the loader assigns for the target and
for the draft with shared or dedicated embeddings, and the hooks the
Qwen4Exp model code reads from its quantization config.
"""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from torch import nn

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
from vllm_sm80.kernels.exl3.layout import exl3_layout
from vllm_sm80.kernels.exl3.quant import Exl3Config
from vllm_sm80.kernels.exl3.utils import Exl3LoadLedger, PackedMatrix

NAMES = Path(__file__).parent / "fixtures" / "exl3_names.json"
NUM_LAYERS = 48
EXPERTS = (0, 1, 511)
EXPERT_ROLES = ("gate_proj", "up_proj", "down_proj")
EXPECTED = {
    "target": "1982c1d215a99b255902e155a9e89602b4daab3f8a5feeedc5ad7bf47b7599de",
    "draft": "55d594722bb8699443d21ad5a29c6bb1b9caf291ad393669ee0e38f6feb373eb",
    "owners": "85a4ee2a9097ae9a24635e0e3ec4e39aba4f831a525ab3c0ea84857f5dbdb9e1",
    "hooks": "3a35e7093b992c1356ca4c7adc308fb9e99f7f8d7369c7032a2138f4f2b416a8",
}


class _Linear(LinearBase):
    def __init__(self, outputs: int):
        nn.Module.__init__(self)
        if outputs:
            self.output_sizes = [1] * outputs


class _Head(ParallelLMHead):
    def __init__(self):
        nn.Module.__init__(self)


def _manifest(names: list[str]) -> SimpleNamespace:
    present = set(names)
    matrices = {}
    for name in names:
        if name.endswith(".trellis"):
            prefix = name.removesuffix(".trellis")
            marker = "mcg" if f"{prefix}.mcg" in present else "mul1"
            matrices[prefix] = PackedMatrix(prefix, 128, 128, 4, 1, marker)
    return SimpleNamespace(
        tensors=dict.fromkeys(names), matrices=matrices, unused={}, ple_tables={}
    )


def _config(model_type: str, model_class: type, manifest) -> Exl3Config:
    """An EXL3 config as maybe_update_config and configure_quant_config
    leave it, over a manifest without files."""
    hf_config = SimpleNamespace(
        model_type=model_type,
        text_config=SimpleNamespace(num_hidden_layers=NUM_LAYERS),
    )
    config = Exl3Config()
    config.layout = exl3_layout(hf_config)
    config.manifest = manifest
    configure_quant_config(config, model_class)
    return config


def _variants(prefix: str) -> set[str]:
    """The checkpoint module and the construction prefixes that name it."""
    variants = {prefix}
    if prefix.startswith("model.language_model."):
        rest = prefix.removeprefix("model.language_model.")
        variants |= {
            f"model.{rest}",
            f"language_model.model.{rest}",
            f"model.language_model.model.{rest}",
        }
    elif prefix == "lm_head":
        variants |= {"language_model.lm_head", "model.language_model.lm_head"}
    elif prefix.startswith("model.visual."):
        variants.add(prefix.removeprefix("model."))
    elif match := re.match(r"mtp\.layers\.(\d+)\.(.*)", prefix):
        index, rest = int(match[1]) + NUM_LAYERS, match[2]
        variants |= {
            f"mtp.layers.{index}.{rest}",
            f"model.mtp.layers.{index}.{rest}",
            f"model.layers.{index}.{rest}",
        }
    return variants


def _prefixes(manifest, mapping: dict[str, list[str]]) -> tuple[set, set]:
    """Construction prefixes of every module and fused projection, and of
    every routed expert container."""
    modules = {name.rsplit(".", 1)[0] for name in manifest.tensors}
    for module in list(modules):
        for fused, members in mapping.items():
            if module.endswith(f".{members[0]}"):
                modules.add(module.removesuffix(members[0]) + fused)
    experts = {
        match[1]
        for module in modules
        if (match := re.match(r"(.*\.mlp\.experts)\.\d+\.", module))
    }
    return (
        {v for prefix in modules for v in _variants(prefix)},
        {v for prefix in experts for v in _variants(prefix)},
    )


def _method(config: Exl3Config, layer: nn.Module, prefix: str):
    try:
        method = config.get_quant_method(layer, prefix)
    except ValueError as error:
        return f"ValueError: {error}"
    if method is None:
        return None
    matrices = [matrix.prefix for matrix in getattr(method, "matrices", ())]
    return [type(method).__name__, matrices, getattr(method, "uneven", None)]


def _choices(config: Exl3Config, modules: set[str], experts: set[str]) -> dict:
    result = {}
    for prefix in sorted(modules):
        entry = {"source": config.source_prefix(prefix)}
        for outputs in (0, 3, 4):
            entry[f"linear{outputs}"] = _method(config, _Linear(outputs), prefix)
        entry["head"] = _method(config, _Head(), prefix)
        result[prefix] = entry
    for prefix in sorted(experts):
        result[f"{prefix} experts"] = [
            config.layout.expert_matrix(config, prefix, expert, role)
            for expert in EXPERTS
            for role in EXPERT_ROLES
        ]
    return result


def _owners(config: Exl3Config, manifest) -> dict:
    result = {}
    for model_type in ("qwen4_exp", "qwen4_exp_mtp"):
        for dedicated in (False, True):
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type),
                hf_text_config=SimpleNamespace(
                    mtp_use_dedicated_embeddings=dedicated
                ),
            )
            ledger = Exl3LoadLedger(manifest)
            is_draft = config.layout.is_draft(model_config)
            backed = config.layout.assign_owners(
                nn.Module(), config, ledger, model_config, is_draft
            )
            result[f"{model_type} dedicated={dedicated}"] = {
                "draft": is_draft,
                "entries": sorted(ledger.entries.items()),
                "file-backed": sorted(backed),
            }
    return result


def _hooks(config: Exl3Config) -> dict:
    return {
        "method": type(config.get_ple_embedding_method()).__qualname__,
        "table": config.ple_embedding_cls.__qualname__,
        "shares": [
            config.mtp_shares_target_embeddings(SimpleNamespace(**attrs))
            for attrs in (
                {},
                {"mtp_use_dedicated_embeddings": False},
                {"mtp_use_dedicated_embeddings": True},
            )
        ],
    }


def results(make_config=_config) -> dict:
    manifest = _manifest(json.loads(NAMES.read_text()))
    target = make_config("qwen4_exp", Qwen4ExpForConditionalGeneration, manifest)
    draft = make_config("qwen4_exp_mtp", Qwen4ExpMTP, manifest)
    # The MTP model numbers the draft layers after the target layers.
    draft.set_mtp_start_layer_idx(NUM_LAYERS)
    modules, experts = _prefixes(
        manifest, target.packed_modules_mapping | draft.packed_modules_mapping
    )
    return {
        "target": _choices(target, modules, experts),
        "draft": _choices(draft, modules, experts),
        "owners": _owners(target, manifest),
        "hooks": _hooks(target),
    }


def digests(found: dict) -> dict[str, str]:
    return {
        key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        for key, value in found.items()
    }


def test_layout_keeps_names_owners_and_methods():
    assert digests(results()) == EXPECTED
