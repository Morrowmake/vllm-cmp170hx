# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 EXL3 checkpoint layout (vllm_sm80/kernels/exl3/layout.py).

The checkpoint keeps DeepSeek's own module names: ``layers.<i>.*`` for the
decoder, ``embed``/``head``/``norm``, ``mtp.<j>.*`` for the DSpark draft
stages, ``vision.*``/``aligner.*``/``image_*`` for the vision tower. In vLLM
the target decoder layers sit under ``[language_model.]model.layers.<i>``, and
the DSpark drafter numbers its layers after the target's
(``model.layers.<num_hidden_layers + j>``) with ``main_proj`` at model level.

Fused vLLM projections run their checkpoint matrices as separate EXL3
constituents: ``fused_wqa_wkv`` = ``wq_a`` | ``wkv`` (replicated),
``compressor.fused_wkv_wgate`` = ``wkv`` [| ``wgate``], shared experts
``gate_up_proj`` = ``w1`` | ``w3`` and ``down_proj`` = ``w2``. Routed expert
roles gate/up/down are ``w1``/``w3``/``w2``. ``attn.wo_a`` is stored as one
matrix per output group (``wo_a.slice.<g>``), so it becomes a grouped EXL3
projection whose TP ranks own whole groups. The DSpark drafter's stacked
``context_wkv_proj`` (model level) is every draft stage's ``attn.wkv``, one
output per stage: ``mtp.<j>.attn.wkv``. It resolves so the projection gets
its stages' method when the drafter is built; the sm_80 drafter then drops
it (its stages write the context KV from their own ``wkv``), so the stage
matrices load once.

Shared experts (intermediate 2304 = 18 H128 blocks) do not split into four
equal block-aligned shards, so they shard unevenly: ranks 0-1 hold 5 blocks,
ranks 2-3 hold 4.

The Engram tables (``layers.<i>.engram.embed.*``) are not in this directory:
the sm_80 Engram storage reads them from the base checkpoint, so their
runtime parameters (if the embedding registers any) need no tensor here.
"""

import re

from vllm_sm80.kernels.exl3.layout import Exl3Layout

_FUSED_MEMBERS = {
    "fused_wqa_wkv": ("wq_a", "wkv"),
    "fused_wkv_wgate": ("wkv", "wgate"),
    "gate_up_proj": ("w1", "w3"),
}
_EXPERT_ROLES = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
# Runtime parameters the Engram storage fills from the base checkpoint.
_ENGRAM_TABLE = re.compile(r"\.engram\.embed_tokens\.")
DRAFT_ARCHITECTURE = "DSparkV41DraftModel"


class DeepseekV41Exl3Layout(Exl3Layout):
    def __init__(self, hf_config=None):
        super().__init__(hf_config)
        text = getattr(hf_config, "text_config", hf_config)
        self.num_layers = getattr(text, "num_hidden_layers", 0)

    def source_prefix(self, config, prefix):
        name = prefix.removeprefix("language_model.").removeprefix("model.")
        if name == "lm_head":
            return "head"
        if name == "embed_tokens":
            return "embed"
        if name == "main_proj":
            return "mtp.0.main_proj"
        match = re.match(r"layers\.(\d+)\.(.*)", name)
        if match is not None and int(match[1]) >= self.num_layers:
            name = f"mtp.{int(match[1]) - self.num_layers}.{match[2]}"
        return re.sub(r"\.shared_experts\.down_proj$", ".shared_experts.w2", name)

    def members(self, config, leaf, source, outputs):
        if source == "context_wkv_proj":
            return [f"mtp.{j}.attn.wkv" for j in range(outputs)]
        members = _FUSED_MEMBERS.get(leaf)
        if members is None:
            return None
        # The ratio-1 compressor has no gate: one output.
        return list(members[:outputs] if outputs else members)

    def expert_matrix(self, config, experts_prefix, expert, role):
        return (
            f"{config.source_prefix(experts_prefix)}.{expert}.{_EXPERT_ROLES[role]}"
        )

    def shards_unevenly(self, prefix):
        return ".shared_experts." in prefix

    def grouped_members(self, config, source):
        if not source.endswith(".attn.wo_a"):
            return []
        assert config.manifest is not None
        members = []
        while f"{source}.slice.{len(members)}" in config.manifest.matrices:
            members.append(f"{source}.slice.{len(members)}")
        return members

    def is_draft(self, model_config):
        return DRAFT_ARCHITECTURE in (model_config.hf_config.architectures or ())

    def assign_owners(self, model, config, ledger, model_config, is_draft):
        """The target owns everything but ``mtp.*``; the DSpark draft owns
        ``mtp.*`` and aliases the target's embedding and head."""
        for name in config.manifest.tensors:
            if name in ledger.entries:
                continue
            if name.startswith("mtp.") and not is_draft:
                ledger.mark(name, "unused", "DSpark draft model owner")
            elif is_draft and not name.startswith("mtp."):
                ledger.mark(name, "unused", "target model owner")
        return {
            name
            for name, _ in model.named_parameters()
            if _ENGRAM_TABLE.search(name)
        }
