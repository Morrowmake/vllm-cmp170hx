# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by AlexBrandes for CMP 170HX support; see repository history.
"""How a model family's EXL3 checkpoint maps onto its vLLM modules.

Exl3Config picks one layout per checkpoint from the HF ``model_type``
(:data:`LAYOUTS`, resolved lazily so no model code is imported until a
checkpoint of that family is loaded). The base :class:`Exl3Layout` keeps the
checkpoint names as module names and gives no tensor another owner; a model
family overrides only what its checkpoint does differently. The shared EXL3
config, manifest and loader reach model-specific behaviour only through
these hooks.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from vllm.sm80.kernels.exl3.quant import Exl3Config
    from vllm.sm80.kernels.exl3.utils import Exl3LoadLedger, Exl3Manifest

# HF model_type -> "<module>:<class>" of the layout its checkpoints use.
LAYOUTS = {
    "qwen4_exp": "vllm.models.qwen4_exp.nvidia.exl3_layout:Qwen4ExpExl3Layout",
    "qwen4_exp_text": "vllm.models.qwen4_exp.nvidia.exl3_layout:Qwen4ExpExl3Layout",
    "qwen4_exp_mtp": "vllm.models.qwen4_exp.nvidia.exl3_layout:Qwen4ExpExl3Layout",
    "deepseek_v41": ("vllm.models.deepseek_v41.nvidia.sm80_exl3:DeepseekV41Exl3Layout"),
}


class Exl3Layout:
    """Checkpoint names are module names; every tensor has a runtime owner
    on this stage."""

    # Layout attributes the family's own model code looks up on its
    # quantization config with getattr (Exl3Config.__getattr__ answers them),
    # such as the embedding methods of file-backed tables.
    config_hooks: frozenset[str] = frozenset()

    def __init__(self, hf_config=None):
        self.hf_config = hf_config

    def check_manifest(self, manifest: Exl3Manifest) -> None:
        """Refuse a checkpoint that lacks the auxiliary files or tensors this
        family needs beyond the EXL3 matrices."""

    def source_prefix(self, config: Exl3Config, prefix: str) -> str:
        """Checkpoint module name of a construction prefix that the model's
        HF-to-vLLM mapper does not name (Exl3Config.mapped_prefixes)."""
        return prefix

    def members(
        self, config: Exl3Config, leaf: str, source: str, outputs: int
    ) -> list[str] | None:
        """Checkpoint leaf names of the fused projection ``source`` with
        ``outputs`` logical outputs, or None for the model's
        packed_modules_mapping."""
        return None

    def expert_matrix(
        self, config: Exl3Config, experts_prefix: str, expert: int, role: str
    ) -> str:
        """Checkpoint matrix of one routed expert projection; ``role`` is
        gate_proj, up_proj or down_proj."""
        source = config.source_prefix(f"{experts_prefix}.0.gate_proj")
        return f"{source.removesuffix('.0.gate_proj')}.{expert}.{role}"

    def shards_unevenly(self, prefix: str) -> bool:
        """Whether this TP-sharded projection may give the leading ranks one
        H128 block more than the others when its blocks do not divide evenly
        (only for a column/row pair whose intermediate never leaves the rank,
        such as an MLP)."""
        return False

    def grouped_members(self, config: Exl3Config, source: str) -> list[str]:
        """Per-group checkpoint matrices of a block-diagonal projection that
        is stored as one matrix per group; empty for an ordinary one."""
        return []

    def is_draft(self, model_config) -> bool:
        """Whether ``model_config`` is the checkpoint's draft model."""
        return False

    def assign_owners(
        self,
        model: nn.Module,
        config: Exl3Config,
        ledger: Exl3LoadLedger,
        model_config,
        is_draft: bool,
    ) -> set[str]:
        """Mark in ``ledger`` the tensors another model owns and join any
        file-backed sources; return the runtime parameter names that load
        without a checkpoint tensor of this directory.

        The base marks only tensors whose module another pipeline stage
        builds (a PPMissingLayer after the model's own name mapping); with
        PP=1 none exist. Families extend this with their cross-model rules.
        """
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        mapper = getattr(model, "hf_to_vllm_mapper", None)
        assert config.manifest is not None
        for name in config.manifest.tensors:
            if name in ledger.entries:
                continue
            mapped_name = mapper.map_name(name) if mapper is not None else name
            if mapped_name is not None and is_pp_missing_parameter(mapped_name, model):
                ledger.mark(
                    name, "unused", "module assigned to another pipeline/model stage"
                )
        return set()


def exl3_layout(hf_config) -> Exl3Layout:
    """The layout of a checkpoint with this HF config (None: the base)."""
    qualified = LAYOUTS.get(getattr(hf_config, "model_type", ""))
    if qualified is None:
        return Exl3Layout(hf_config)
    module_name, _, class_name = qualified.partition(":")
    return getattr(importlib.import_module(module_name), class_name)(hf_config)
