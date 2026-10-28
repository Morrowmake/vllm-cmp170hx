# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A tiny DeepSeek-V4.1 hybrid checkpoint: EXL3 routed experts with every
other module from a source checkpoint in the official FP8 layout.

:func:`source_tensors` turns the tiny EXL3 checkpoint of ``exl3_fixture``
into the official checkpoint's tensors for the same config: FP8 E4M3 linears
with one UE8M0 scale per 32x32 block (``wo_a`` one dense block-diagonal
matrix), BF16 compressor, indexer ``wk`` and head, FP4 (I8-packed) routed
experts with per-32 scales, the official dtypes of the plain tensors, and
Engram tables. :func:`write_hybrid` writes the export and the source (the
Engram shard in a second directory, as served) and builds the hybrid
directory with ``tools/sm80/build_hybrid_checkpoint.py``.
"""

import importlib.util
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.models.deepseek_v41.sm80 import exl3_fixture

REPO = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "build_hybrid_checkpoint", REPO / "tools" / "sm80" / "build_hybrid_checkpoint.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

SOURCE_QUANT = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}
EXL3_MODULES = r"^(layers|mtp)\.\d+\.ffn\.experts(\.|$)"
DROP = r"\.engram\.embed\."
METHOD = "deepseek_v4_fp8"
# Modules the official checkpoint stores as BF16 linears.
BF16_LINEARS = re.compile(r"\.compressor\.(wkv|wgate)$|\.indexer\.wk$|^head$")
# Plain tensors the official checkpoint stores at another dtype than the export.
PLAIN_DTYPES = {
    r"\.ffn\.gate\.weight$": "BF16",
    r"\.ffn\.gate\.bias$": "F32",
    r"\.weights_proj\.weight$": "BF16",
    r"\.markov_head\.head\.weight$": "BF16",
    r"\.confidence_head\.proj\.weight$": "BF16",
    r"\.engram\.(q|k)_weight$": "BF16",
}
ENGRAM_ROWS = 64
_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
}


def _blocks(v: int) -> int:
    return -(-v // 32)


def source_tensors(config: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Name -> (dtype, shape) of the official checkpoint for ``config``."""
    export = exl3_fixture.checkpoint_tensors(config)
    text = config["text_config"]
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    wo_a: dict[str, list[tuple[int, int]]] = {}
    for name, (dtype, shape) in export.items():
        if name.endswith((".suh", ".svh", ".mul1")):
            continue
        if not name.endswith(".trellis"):
            for pattern, source_dtype in PLAIN_DTYPES.items():
                if re.search(pattern, name):
                    dtype = source_dtype
            tensors[name] = (dtype, shape)
            continue
        prefix = name.removesuffix(".trellis")
        k, n = shape[0] * 16, shape[1] * 16
        group = re.match(r"(.*\.attn\.wo_a)\.slice\.\d+$", prefix)
        if group:
            wo_a.setdefault(group[1], []).append((k, n))
        elif ".ffn.experts." in prefix:
            tensors[f"{prefix}.weight"] = ("I8", (n, k // 2))
            tensors[f"{prefix}.scale"] = ("F8_E8M0", (n, k // 32))
        elif BF16_LINEARS.search(prefix):
            tensors[f"{prefix}.weight"] = ("BF16", (n, k))
        else:
            tensors[f"{prefix}.weight"] = ("F8_E4M3", (n, k))
            tensors[f"{prefix}.scale"] = ("F8_E8M0", (_blocks(n), _blocks(k)))
    for prefix, groups in wo_a.items():
        (k, n), count = groups[0], len(groups)
        tensors[f"{prefix}.weight"] = ("F8_E4M3", (count * n, k))
        tensors[f"{prefix}.scale"] = ("F8_E8M0", (_blocks(count * n), _blocks(k)))
    for layer in text["engram_layer_ids"]:
        if layer < text["num_hidden_layers"]:
            dim = text["engram_head_dim"]
            tensors[f"layers.{layer}.engram.embed.weight"] = (
                "F8_E4M3",
                (ENGRAM_ROWS, dim),
            )
            tensors[f"layers.{layer}.engram.embed.scale"] = (
                "F8_E8M0",
                (ENGRAM_ROWS, dim // 32),
            )
    return tensors


def _value(dtype: str, shape, generator) -> torch.Tensor:
    if dtype == "F8_E4M3":
        values = torch.rand(shape, generator=generator) * 2 - 1
        return values.to(torch.float8_e4m3fn)
    if dtype == "F8_E8M0":
        exponents = torch.randint(118, 130, shape, generator=generator)
        return exponents.to(torch.uint8).view(torch.float8_e8m0fnu)
    if dtype == "I8":
        return torch.randint(-128, 128, shape, dtype=torch.int8, generator=generator)
    return (torch.rand(shape, generator=generator) * 0.02 - 0.01).to(_TORCH[dtype])


def _write_shards(directory: Path, tensors: dict, names_per_file) -> dict[str, str]:
    """Write ``tensors`` (name -> tensor) into the files ``names_per_file``
    assigns; return the weight map."""
    weight_map = {}
    for filename, names in names_per_file.items():
        save_file({n: tensors[n] for n in names}, str(directory / filename))
        weight_map.update({n: filename for n in names})
    return weight_map


def _index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / builder.INDEX).write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def write_hybrid(root: Path, config: dict) -> tuple[Path, Path, dict]:
    """Write the export, the source and the hybrid built from them under
    ``root``; return (hybrid dir, source dir, source tensors by name)."""
    export = root / "export"
    exl3_fixture.write_checkpoint(export, config)
    weight_map = {}
    for path in sorted(export.glob("*.safetensors")):
        weight_map.update({n: path.name for n in builder.read_header(path)})
    _index(export, weight_map)
    (export / "tokenizer.json").write_text("{}")
    generator = torch.Generator().manual_seed(1)
    tensors = {
        name: _value(dtype, shape, generator)
        for name, (dtype, shape) in sorted(source_tensors(config).items())
    }
    source, engram = root / "source", root / "engram"
    source.mkdir()
    engram.mkdir()
    source_config = json.loads(json.dumps(config))
    source_config["quantization_config"] = SOURCE_QUANT
    for directory in (source, engram):
        (directory / "config.json").write_text(json.dumps(source_config))
    tables = sorted(n for n in tensors if ".engram.embed." in n)
    names = sorted(n for n in tensors if n not in tables)
    half = len(names) // 2
    files = {
        "model-00001-of-00003.safetensors": names[:half],
        "model-00002-of-00003.safetensors": names[half:],
    }
    weight_map = _write_shards(source, tensors, files)
    weight_map |= _write_shards(
        engram, tensors, {"model-00003-of-00003.safetensors": tables}
    )
    # Both directories carry the full index; each holds only its files.
    _index(source, weight_map)
    _index(engram, weight_map)
    hybrid = root / "hybrid"
    builder.main(
        [
            "--export", str(export),
            "--source", str(source),
            "--source", str(engram),
            "--out", str(hybrid),
            "--exl3-modules", EXL3_MODULES,
            "--method", METHOD,
            "--drop", DROP,
        ]
    )  # fmt: skip
    return hybrid, source, tensors
