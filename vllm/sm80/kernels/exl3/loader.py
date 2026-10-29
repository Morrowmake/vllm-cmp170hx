# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by AlexBrandes for CMP 170HX support; see repository history.
"""Load EXL3 constituents without materializing unassigned packed tensors."""

import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from torch import nn

from vllm.config import ModelConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.sm80.kernels.exl3.quant import (
    Exl3Config,
    Exl3LinearMethod,
    Exl3MoEMethod,
)
from vllm.sm80.kernels.exl3.utils import Exl3LoadLedger
from vllm.tracing import instrument

logger = init_logger(__name__)


def _verify_pipeline_parallel(
    model_config: ModelConfig, parallel_config: ParallelConfig
) -> None:
    if parallel_config.pipeline_parallel_size == 1:
        return
    if (
        model_config.hf_config.model_type
        not in ("qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp")
        or parallel_config.tensor_parallel_size != 1
        or parallel_config.pipeline_parallel_size != 2
    ):
        raise ValueError(
            "EXL3 pipeline parallelism supports Qwen4Exp at PP=2, TP=1; "
            "other EXL3 layouts require PP=1"
        )


class Exl3ModelLoader(DefaultModelLoader):
    """Load format "exl3" (--load-format exl3), registered by vllm/sm80/plugin.py.

    EXL3 checkpoints get the checks below and manifest-driven weight loading;
    anything else loads as with the default loader. Exl3Config refuses EXL3
    checkpoints under any other load format.
    """

    def download_model(self, model_config: ModelConfig) -> None:
        if model_config.quantization != "exl3":
            super().download_model(model_config)
        elif not Path(model_config.model).is_dir():
            raise ValueError("Download the EXL3 safetensors checkpoint before loading")

    def create_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        if model_config.quantization == "exl3":
            from vllm.sm80.kernels.exl3.ops import (
                ensure_extension,
            )

            offload = vllm_config.offload_config
            if offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size:
                raise ValueError("EXL3 permits runtime offload only for compressed PLE")
            _verify_pipeline_parallel(model_config, vllm_config.parallel_config)
            if vllm_config.lora_config is not None:
                raise ValueError("EXL3 does not support LoRA")
            ensure_extension()
        return super().create_model(vllm_config, model_config, prefix)

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        if model_config.quantization != "exl3":
            super().load_weights(model, model_config)
            return
        self.counter_before_loading_weights = time.perf_counter()
        load_exl3_weights(model, model_config)
        self.counter_after_loading_weights = time.perf_counter()


def require_cuda_parameters(model: nn.Module) -> None:
    """Refuse a loaded EXL3 model with a parameter off the GPU."""
    for name, parameter in model.named_parameters():
        if not parameter.is_cuda:
            raise ValueError(f"EXL3 parameter must remain CUDA-owned: {name}")


def manifest_still_needed(vllm_config, is_draft: bool) -> bool:
    """Whether a draft model will still be built from the target's EXL3 config.

    The MTP draft is constructed and loaded after the target, and its top
    module and the loader use the target's quantization config (the inner
    draft layers use a second config of their own). Under PP only the last
    target stage constructs that draft.
    """
    spec = vllm_config.speculative_config
    if is_draft or spec is None or spec.draft_model_config is None:
        return False
    return (
        vllm_config.parallel_config.pipeline_parallel_size == 1
        or get_pp_group().is_last_rank
    )


def release_manifests(model: torch.nn.Module, quant: Exl3Config) -> None:
    """Free every EXL3 checkpoint inventory the loaded model still references."""
    configs = {id(quant): quant}
    for module in model.modules():
        owner = getattr(getattr(module, "quant_method", None), "config", None)
        if isinstance(owner, Exl3Config):
            configs[id(owner)] = owner
    for owner in configs.values():
        owner.release_manifest()


def load_exl3_weights(model: torch.nn.Module, model_config) -> set[str]:
    from vllm.sm80.kernels.exl3.ops import ensure_extension

    ensure_extension()
    config = get_current_vllm_config()
    quant = cast(
        Exl3Config,
        model.quant_config if hasattr(model, "quant_config") else config.quant_config,
    )
    manifest = quant.manifest
    if manifest is None:
        raise ValueError("EXL3 checkpoint manifest was not initialized")
    _verify_pipeline_parallel(model_config, config.parallel_config)
    ledger = Exl3LoadLedger(manifest)
    layout = quant.layout
    is_draft = layout.is_draft(model_config)
    loaded = set()
    modules = dict(model.named_modules())
    # The model family's layout marks the tensors another model owns as unused
    # and joins file-backed sources, whose parameters need no tensor here.
    file_backed_parameters = layout.assign_owners(
        model, quant, ledger, model_config, is_draft
    )

    with ExitStack() as stack:
        files = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in sorted({header.path for header in manifest.tensors.values()})
        }
        for name, module in modules.items():
            method = getattr(module, "quant_method", None)
            if isinstance(method, Exl3LinearMethod):
                bias_offset = 0
                for key in method.nonlocal_keys:
                    ledger.mark(key, "unused", "group assigned to another TP rank")
                for weight in module.exl3_weights:
                    weight.load(files, ledger)
                    bias_key = f"{weight.matrix.prefix}.bias"
                    if bias_key in manifest.tensors:
                        bias = getattr(module, "bias", None)
                        if bias is None:
                            raise ValueError(
                                f"EXL3 checkpoint bias has no owner: {bias_key}"
                            )
                        header = manifest.tensors[bias_key]
                        if (
                            header.dtype not in ("F16", "BF16", "F32")
                            or len(header.shape) != 1
                        ):
                            raise ValueError(f"Invalid EXL3 bias: {bias_key}")
                        if (
                            header.shape[0]
                            not in (
                                weight.matrix.padded_out,
                                sum(s.stop - s.start for s in weight.output_slices),
                            )
                            and header.shape[0] != module.output_size
                        ):
                            raise ValueError(f"EXL3 bias geometry mismatch: {bias_key}")
                        data = files[header.path].get_slice(bias_key)
                        remaining = weight.logical_out
                        for part in weight.output_slices:
                            width = min(part.stop - part.start, remaining)
                            bias.data[bias_offset : bias_offset + width].copy_(
                                data[part.start : part.start + width]
                            )
                            bias_offset += width
                            remaining -= width
                        ledger.mark(bias_key, "consumed", name)
                        loaded.add(f"{name}.bias")
                    elif getattr(module, "bias", None) is not None:
                        raise ValueError(f"Missing EXL3 bias: {bias_key}")
                loaded.update(f"{name}.{key}" for key, _ in module.named_parameters())
            elif isinstance(method, Exl3MoEMethod):
                for key in method.nonlocal_keys:
                    ledger.mark(key, "unused", "expert assigned to another EP rank")
                for expert in module.exl3_experts:
                    for weight in expert:
                        weight.load(files, ledger)
                loaded.update(f"{name}.{key}" for key, _ in module.named_parameters())

        for name, header in manifest.tensors.items():
            if name in ledger.entries:
                continue
            if name.endswith((".trellis", ".suh", ".svh", ".mcg", ".mul1")):
                raise ValueError(f"EXL3 packed tensor has no runtime owner: {name}")
            tensor = files[header.path].get_tensor(name)
            consumed = model.load_weights(iter([(name, tensor)]))
            if not consumed:
                raise ValueError(f"EXL3 plain tensor has no runtime owner: {name}")
            loaded.update(consumed)
            ledger.mark(name, "consumed", ",".join(sorted(consumed)))
        ledger.finish()
    missing = {
        name
        for name, _ in model.named_parameters()
        if name not in loaded and name not in file_backed_parameters
    }
    if missing:
        raise ValueError(f"Unloaded EXL3 runtime parameters: {sorted(missing)[:16]}")
    require_cuda_parameters(model)
    from vllm.sm80 import capture

    if capture.active and "hybrid" in capture.config().families:
        from vllm.models.deepseek_v41.nvidia.hybrid_capture import register_model

        register_model(model)
    counts = Counter(disposition for disposition, _ in ledger.entries.values())
    logger.info(
        "EXL3 packed format, BF16 activations; %s model; "
        "TP rank=%d/%d, EP=%s; sources: %d consumed, %d file-backed, %d unused",
        "draft" if is_draft else "target",
        get_tensor_model_parallel_rank(),
        config.parallel_config.tensor_parallel_size,
        config.parallel_config.enable_expert_parallel,
        counts["consumed"],
        counts["file-backed"],
        counts["unused"],
    )
    assignments = {
        tuple(module.quant_method.global_expert_ids)
        for module in modules.values()
        if isinstance(getattr(module, "quant_method", None), Exl3MoEMethod)
    }
    for assignment in sorted(assignments):
        logger.info("EXL3 local routed expert IDs: %s", assignment)
    if not manifest_still_needed(config, is_draft):
        release_manifests(model, quant)
    return loaded
