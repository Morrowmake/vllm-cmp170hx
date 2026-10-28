# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A tiny DeepSeek-V4.1 EXL3 checkpoint with the real checkpoint's layout.

:func:`tiny_config` scales the real config down while keeping every kind of
layer (sliding-window, ratio-1 and ratio-2 compressed, KV/index/candidate
sources, Engram layers, three DSpark stages) and the TP4 sharding cases of
the real one: column/row cuts in whole H128 blocks, two ``wo_a`` groups per
rank, shared experts whose blocks do not split evenly over four ranks, and
a vocabulary whose four LM-head shards start inside H128 blocks.
:func:`checkpoint_tensors` lists the tensors such a checkpoint holds, named
and typed as in the real EXL3 shards (EXL3 matrices as trellis/suh/svh/mul1;
no Engram tables, which live in the base checkpoint).

The real checkpoint's config.json and its tensor headers (experts collapsed
to one entry per layer and role) are in ``fixtures/``.
"""

import json
import math
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

MUL1 = 0x83DCD12D
FIXTURES = Path(__file__).parent / "fixtures"
REAL_CONFIG = FIXTURES / "config.json"
_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I16": torch.int16,
    "I32": torch.int32,
}
_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4}

TEXT = {
    "vocab_size": 1408,
    "hidden_size": 512,
    "moe_intermediate_size": 640,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 256,
    "o_lora_rank": 128,
    "o_groups": 8,
    "max_position_embeddings": 8192,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 2,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 4096,
    },
    "n_routed_experts": 16,
    "num_experts_per_tok": 6,
    "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 0, 0, 0, 0],
    "kv_source_layer_ids": [2, 5],
    "index_source_layer_ids": [2, 5, 6],
    "index_n_heads": 8,
    "index_head_dim": 128,
    "index_topk": 64,
    "candidate_source_layer_id": 5,
    "candidate_topk_blocks": 16,
    "candidate_block_size": 8,
    "engram_layer_ids": [1, 4],
    "dspark_target_layer_ids": [5, 6, 7],
    "dspark_n_routed_experts": 8,
    "dspark_num_experts_per_tok": 3,
}
VISION = {
    "num_hidden_layers": 2,
    "hidden_size": 256,
    "num_attention_heads": 4,
    "intermediate_size": 512,
}
# bits_x2 per matrix kind (the real checkpoint's K: routed 3, shared and
# attention 5, wq_a/wkv and head 6, indexer wk 8, draft 4).
BITS_X2 = {
    "routed": 6,
    "shared": 10,
    "attn": 10,
    "attn_in": 12,
    "indexer_wk": 16,
    "head": 12,
    "engram": 10,
    "mtp": 8,
    "mtp_dense": 12,
}


def tiny_config() -> dict:
    """The real config.json with the tiny dimensions above."""
    config = json.loads(REAL_CONFIG.read_text())
    config["text_config"].update(json.loads(json.dumps(TEXT)))
    config["vision_config"].update(VISION)
    return config


def _matrix(tensors: dict, name: str, k: int, n: int, bits_x2: int) -> None:
    pad = lambda v: -(-v // 128) * 128  # noqa: E731
    k, n = pad(k), pad(n)
    tensors[f"{name}.trellis"] = ("I16", (k // 16, n // 16, 8 * bits_x2))
    tensors[f"{name}.suh"] = ("F16", (k,))
    tensors[f"{name}.svh"] = ("F16", (n,))
    tensors[f"{name}.mul1"] = ("I32", ())


def _block(tensors: dict, root: str, text: dict, layer: int, draft: bool) -> None:
    hidden, heads, head_dim = (
        text["hidden_size"],
        text["num_attention_heads"],
        text["head_dim"],
    )
    q_lora, o_lora, groups = text["q_lora_rank"], text["o_lora_rank"], text["o_groups"]
    dense = BITS_X2["mtp_dense" if draft else "attn"]
    attn_in = BITS_X2["mtp_dense" if draft else "attn_in"]
    _matrix(tensors, f"{root}.attn.wq_a", hidden, q_lora, attn_in)
    _matrix(tensors, f"{root}.attn.wkv", hidden, head_dim, attn_in)
    _matrix(tensors, f"{root}.attn.wq_b", q_lora, heads * head_dim, dense)
    for group in range(groups):
        _matrix(
            tensors,
            f"{root}.attn.wo_a.slice.{group}",
            heads * head_dim // groups,
            o_lora,
            dense,
        )
    _matrix(tensors, f"{root}.attn.wo_b", groups * o_lora, hidden, dense)
    tensors[f"{root}.attn.attn_sink"] = ("F32", (heads,))
    tensors[f"{root}.attn.q_norm.weight"] = ("BF16", (q_lora,))
    tensors[f"{root}.attn.kv_norm.weight"] = ("BF16", (head_dim,))
    if not draft:
        ratio = text["compress_ratios"][layer]
        if layer in text["kv_source_layer_ids"]:
            _matrix(tensors, f"{root}.attn.compressor.wkv", hidden, head_dim, dense)
            if ratio == 2:
                _matrix(
                    tensors, f"{root}.attn.compressor.wgate", hidden, head_dim, dense
                )
            tensors[f"{root}.attn.compressor.norm.weight"] = ("BF16", (head_dim,))
            index_dim = text["index_head_dim"]
            _matrix(
                tensors,
                f"{root}.attn.indexer.wk",
                head_dim,
                index_dim,
                BITS_X2["indexer_wk"],
            )
            tensors[f"{root}.attn.indexer.k_norm.weight"] = ("BF16", (index_dim,))
        if layer in text["index_source_layer_ids"]:
            index_heads = text["index_n_heads"]
            _matrix(
                tensors,
                f"{root}.attn.indexer.wq_b",
                q_lora,
                index_heads * text["index_head_dim"],
                dense,
            )
            tensors[f"{root}.attn.indexer.weights_proj.weight"] = (
                "F16",
                (index_heads, hidden),
            )
        if layer in text["engram_layer_ids"]:
            hc = text["hc_mult"]
            n_hash = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
            _matrix(
                tensors,
                f"{root}.engram.wkv",
                n_hash * text["engram_head_dim"],
                hidden * (hc + 1),
                BITS_X2["engram"],
            )
            tensors[f"{root}.engram.q_weight"] = ("F32", (hc, hidden))
            tensors[f"{root}.engram.k_weight"] = ("F32", (hc, hidden))
    experts = text["dspark_n_routed_experts"] if draft else text["n_routed_experts"]
    inter = text["moe_intermediate_size"]
    routed = BITS_X2["mtp" if draft else "routed"]
    for expert in range(experts):
        for weight, k, n in (("w1", hidden, inter), ("w3", hidden, inter)):
            _matrix(tensors, f"{root}.ffn.experts.{expert}.{weight}", k, n, routed)
        _matrix(tensors, f"{root}.ffn.experts.{expert}.w2", inter, hidden, routed)
    shared = BITS_X2["mtp" if draft else "shared"]
    _matrix(tensors, f"{root}.ffn.shared_experts.w1", hidden, inter, shared)
    _matrix(tensors, f"{root}.ffn.shared_experts.w3", hidden, inter, shared)
    _matrix(tensors, f"{root}.ffn.shared_experts.w2", inter, hidden, shared)
    tensors[f"{root}.ffn.gate.weight"] = ("F16", (experts, hidden))
    tensors[f"{root}.ffn.gate.bias"] = ("F16", (experts,))
    tensors[f"{root}.ffn.gate.bias_vl"] = ("F32", (experts,))
    mix = (2 + text["hc_mult"]) * text["hc_mult"]
    for part in ("attn", "ffn"):
        tensors[f"{root}.hc_{part}_fn"] = ("F32", (mix, text["hc_mult"] * hidden))
        tensors[f"{root}.hc_{part}_base"] = ("F32", (mix,))
        tensors[f"{root}.hc_{part}_scale"] = ("F32", (3,))
    tensors[f"{root}.attn_norm.weight"] = ("BF16", (hidden,))
    tensors[f"{root}.ffn_norm.weight"] = ("BF16", (hidden,))


def checkpoint_tensors(config: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Name -> (safetensors dtype, shape) of an EXL3 DeepSeek-V4.1 checkpoint
    with this config."""
    text, vision = config["text_config"], config["vision_config"]
    hidden, vocab = text["hidden_size"], text["vocab_size"]
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {
        "embed.weight": ("BF16", (vocab, hidden)),
        "norm.weight": ("BF16", (hidden,)),
    }
    _matrix(tensors, "head", hidden, vocab, BITS_X2["head"])
    for layer in range(text["num_hidden_layers"]):
        _block(tensors, f"layers.{layer}", text, layer, draft=False)
    stages = text["num_nextn_predict_layers"]
    for stage in range(stages):
        _block(tensors, f"mtp.{stage}", text, text["num_hidden_layers"] + stage, True)
    targets = len(text["dspark_target_layer_ids"])
    _matrix(tensors, "mtp.0.main_proj", targets * hidden, hidden, BITS_X2["mtp_dense"])
    tensors["mtp.0.main_norm.weight"] = ("BF16", (hidden,))
    last = f"mtp.{stages - 1}"
    rank = text["dspark_markov_rank"]
    tensors[f"{last}.norm.weight"] = ("BF16", (hidden,))
    tensors[f"{last}.markov_head.embed.weight"] = ("BF16", (vocab, rank))
    tensors[f"{last}.markov_head.head.weight"] = ("F16", (vocab, rank))
    tensors[f"{last}.confidence_head.proj.weight"] = ("F16", (1, hidden + rank))
    dim, inter = vision["hidden_size"], vision["intermediate_size"]
    patch = vision["patch_size"]
    for block in range(vision["num_hidden_layers"]):
        root = f"vision.blocks.{block}"
        tensors[f"{root}.attn.wqkv.weight"] = ("BF16", (3 * dim, dim))
        tensors[f"{root}.attn.wqkv.bias"] = ("BF16", (3 * dim,))
        tensors[f"{root}.attn.wo.weight"] = ("BF16", (dim, dim))
        tensors[f"{root}.attn.wo.bias"] = ("BF16", (dim,))
        tensors[f"{root}.mlp.w1.weight"] = ("BF16", (2 * inter, dim))
        tensors[f"{root}.mlp.w2.weight"] = ("BF16", (dim, inter))
        tensors[f"{root}.norm1.weight"] = ("BF16", (dim,))
        tensors[f"{root}.norm2.weight"] = ("BF16", (dim,))
    tensors["vision.norm.weight"] = ("BF16", (dim,))
    tensors["vision.patch_embed.proj.weight"] = ("BF16", (dim, 3 * patch * patch))
    tensors["vision.patch_embed.proj.bias"] = ("BF16", (dim,))
    ratio = vision["downsample_ratio"]
    tensors["aligner.w1.weight"] = ("BF16", (hidden, dim * ratio * ratio))
    tensors["aligner.w1.bias"] = ("BF16", (hidden,))
    tensors["aligner.w2.weight"] = ("BF16", (hidden, hidden))
    tensors["aligner.w2.bias"] = ("BF16", (hidden,))
    for name in ("image_start", "image_end", "image_newline"):
        tensors[name] = ("BF16", (hidden,))
    return tensors


def tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    return math.prod(shape) * _BYTES[dtype]


def write_checkpoint(directory: Path, config: dict, shard_bytes: int = 64 << 20):
    """Write ``config.json`` and the checkpoint's tensors (random values,
    the mul1 codebook marker) as safetensors shards."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(0)
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name, (dtype, shape) in sorted(checkpoint_tensors(config).items()):
        if name.endswith(".mul1"):
            value = struct.unpack("<i", struct.pack("<I", MUL1))[0]
            tensor = torch.tensor(value, dtype=torch.int32)
        elif dtype == "I16":
            tensor = torch.randint(
                -(2**15), 2**15, shape, dtype=torch.int16, generator=generator
            )
        else:
            tensor = (torch.rand(shape, generator=generator) * 0.02 - 0.01).to(
                _DTYPES[dtype]
            )
        if size and size + tensor.nbytes > shard_bytes:
            shards.append({})
            size = 0
        shards[-1][name] = tensor
        size += tensor.nbytes
    for index, shard in enumerate(shards):
        save_file(
            shard, str(directory / f"model-{index + 1:05d}-of-{len(shards):05d}.safetensors")
        )
