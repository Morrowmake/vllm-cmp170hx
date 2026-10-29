# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by AlexBrandes for CMP 170HX support; see repository history.
"""Qwen4Exp EXL3 checkpoint layout (vllm/sm80/kernels/exl3/layout.py).

The checkpoint keeps the HF module names under ``model.language_model.``, the
MTP draft's layers under ``mtp.layers.<i>`` (numbered after the target's in
the model), the compressed PLE table in ``ngram_embedding.safetensors`` and
the MTP mixer in its own patch file.
"""

import re

from vllm.sm80.kernels.exl3.layout import Exl3Layout
from vllm.sm80.kernels.exl3.utils import PLE_AUXILIARIES

# Qwen4Exp QSA projects Q/gate, K, V and the replicated indexer Q/K in one
# layer. It first builds a three-output QKV layer, then replaces it with the
# four-output fused one, and the target, MTP and LoRA packed mappings disagree
# on the indexer, so each layer's output count selects its members.
_QSA_QKV_MEMBERS = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")
_MIXER_PATCH = "mtp_hyper_connection_mixer_patch.safetensors"
_AUXILIARY_FILES = ("ngram_embedding.safetensors", _MIXER_PATCH)
_MIXER = "mtp.hyper_connection_mixer"


class Qwen4ExpExl3Layout(Exl3Layout):
    # Read by common/ngram_embedding.py, nvidia/ngram_embedding.py and
    # nvidia/mtp.py.
    config_hooks = frozenset(
        {
            "get_ple_embedding_method",
            "ple_embedding_cls",
            "set_mtp_start_layer_idx",
            "mtp_shares_target_embeddings",
        }
    )

    def __init__(self, hf_config=None):
        super().__init__(hf_config)
        text = getattr(hf_config, "text_config", hf_config)
        self.mtp_start_layer_idx = getattr(text, "num_hidden_layers", 0)

    def check_manifest(self, manifest):
        """Both auxiliary files, the MTP mixer patch tensors and a compressed
        PLE table."""
        for name in _AUXILIARY_FILES:
            if not (manifest.directory / name).is_file():
                raise ValueError(f"Missing required EXL3 auxiliary file: {name}")
        down = manifest.tensors.get(f"{_MIXER}.input_mix_weight_down.weight")
        if down is None or len(down.shape) != 2 or min(down.shape) <= 0:
            raise ValueError("Invalid EXL3 MTP mixer down projection")
        rank, hidden = down.shape
        expected = {
            f"{_MIXER}.hc_norm.weight": (hidden,),
            f"{_MIXER}.input_mix_weight_down.weight": (rank, hidden),
            f"{_MIXER}.input_mix_weight_up.weight": (hidden, rank),
        }
        actual = {k for k, h in manifest.tensors.items() if h.path.name == _MIXER_PATCH}
        if actual != expected.keys():
            raise ValueError("Unexpected EXL3 MTP mixer patch keys")
        for name, shape in expected.items():
            manifest.expect(name, "F16", shape)
        if not manifest.ple_tables:
            raise ValueError("Missing compressed EXL3 PLE table")

    def get_ple_embedding_method(self):
        """PLE method (vllm/models/qwen4_exp/common/ngram_embedding.py)."""
        from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLEEmbeddingMethod

        return Exl3PLEEmbeddingMethod()

    @property
    def ple_embedding_cls(self):
        """PLE table (vllm/models/qwen4_exp/nvidia/ngram_embedding.py)."""
        from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLEEmbedding

        return Exl3PLEEmbedding

    def set_mtp_start_layer_idx(self, index: int) -> None:
        """MTP draft layers are numbered after the target's (nvidia/mtp.py)."""
        self.mtp_start_layer_idx = index

    @staticmethod
    def mtp_shares_target_embeddings(config) -> bool:
        """Whether the MTP draft reuses the target embedding and head
        (nvidia/mtp.py builds PPMissingLayer placeholders for them)."""
        return not getattr(config, "mtp_use_dedicated_embeddings", False)

    def source_prefix(self, config, prefix):
        if prefix.startswith("model.language_model.model."):
            prefix = prefix.replace(
                "model.language_model.model.", "model.language_model.", 1
            )
        elif prefix.startswith("language_model.model."):
            prefix = prefix.replace("language_model.model.", "model.language_model.", 1)
        elif prefix.startswith(("model.layers.", "model.embed_tokens.")):
            prefix = "model.language_model." + prefix.removeprefix("model.")
        for head_prefix in ("model.language_model.lm_head", "language_model.lm_head"):
            if prefix == head_prefix:
                prefix = "lm_head"
        if prefix.startswith("visual."):
            prefix = "model." + prefix
        if prefix.startswith("model.mtp."):
            prefix = prefix.removeprefix("model.")
        if prefix.startswith("mtp.layers."):
            match = re.match(r"mtp.layers.(\d+)\.(.*)", prefix)
            assert match is not None
            index = int(match[1])
            if index >= self.mtp_start_layer_idx:
                prefix = f"mtp.layers.{index - self.mtp_start_layer_idx}.{match[2]}"
        return prefix

    def members(self, config, leaf, source, outputs):
        if leaf == "qkv" and ".visual." in source:
            return ["q_proj", "k_proj", "v_proj"]
        if leaf == "qkv_proj" and outputs:
            return list(_QSA_QKV_MEMBERS[:outputs])
        return None

    def is_draft(self, model_config):
        return model_config.hf_config.model_type == "qwen4_exp_mtp"

    def assign_owners(self, model, config, ledger, model_config, is_draft):
        # The base marks tensors whose module another pipeline stage builds.
        super().assign_owners(model, config, ledger, model_config, is_draft)
        manifest = config.manifest
        shared: set[str] = set()
        if is_draft and self.mtp_shares_target_embeddings(model_config.hf_text_config):
            # The draft reuses the target's embedding and head: no draft owner.
            shared = {
                name
                for name in manifest.tensors
                if name.startswith(("lm_head.", "model.language_model.embed_tokens."))
            }
        for name in manifest.tensors:
            if name in ledger.entries:
                continue
            if name in shared:
                ledger.mark(name, "unused", "shared target embedding/head owner")
            elif name.startswith("mtp.") and not is_draft:
                ledger.mark(name, "unused", "native MTP model owner")
            elif is_draft and not name.startswith("mtp."):
                if name.startswith(("lm_head.", "model.language_model.embed_tokens.")):
                    continue
                ledger.mark(name, "unused", "target model owner")

        from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
            Qwen4ExpNGramEmbedding,
        )

        file_backed = set()
        for name, module in model.named_modules():
            if not isinstance(module, Qwen4ExpNGramEmbedding):
                continue
            source = config.source_prefix(f"{name}.ngram_embedding")
            table = manifest.ple_tables.get(source)
            if table is None:
                raise ValueError(f"Missing EXL3 PLE source for module: {name}")
            expected = {table.name} | {
                f"{source}.{suffix}" for suffix in PLE_AUXILIARIES
            }
            consumed = _setup_ple_file(module, str(table.path), table.name)
            if consumed != expected:
                raise ValueError(
                    f"Incomplete or unexpected EXL3 PLE source join: {source}"
                )
            for key in sorted(consumed):
                ledger.mark(
                    key, "file-backed" if key == table.name else "consumed", name
                )
            file_backed.add(f"{name}.ngram_embedding.weight")
        return file_backed


def _setup_ple_file(module, path: str, key: str) -> set[str]:
    """Join a file-backed manifest entry to a Qwen4ExpNGramEmbedding and
    consume its small auxiliaries. Reads the module's ngram_embedding,
    ngram_heads_offsets, ngram_heads_vocab_sizes and layer_multipliers
    (pinned in tests/sm80/seams.yaml)."""
    from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLEEmbedding

    if not isinstance(module.ngram_embedding, Exl3PLEEmbedding):
        raise TypeError("File-backed PLE setup requires EXL3 quantization")
    return module.ngram_embedding.setup_file(
        path,
        key,
        module.ngram_heads_offsets,
        module.ngram_heads_vocab_sizes,
        module.layer_multipliers,
    )
