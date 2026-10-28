# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header, ownership and transform-boundary failures need no tensor runtime."""

import json
import struct

import pytest

from vllm_sm80.models.qwen4_exp.exl3_layout import Qwen4ExpExl3Layout
from vllm_sm80.kernels.exl3.utils import (
    Exl3LoadLedger,
    Exl3Manifest,
    aligned_shard,
    ple_layout,
    read_headers,
    read_metadata,
)


def _write(path, tensors, metadata=None):
    header, payload = {}, bytearray()
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _matrix(prefix="proj", bits_x2=10, marker="mul1", padded_in=128, padded_out=256):
    multiplier = 0x83DCD12D if marker == "mul1" else 0xCBAC1FED
    return {
        f"{prefix}.trellis": (
            "I16",
            [padded_in // 16, padded_out // 16, 8 * bits_x2],
            bytes(padded_in * padded_out * bits_x2 // 16),
        ),
        f"{prefix}.suh": ("F16", [padded_in], bytes(padded_in * 2)),
        f"{prefix}.svh": ("F16", [padded_out], bytes(padded_out * 2)),
        f"{prefix}.{marker}": ("I32", [], struct.pack("<I", multiplier)),
    }


def test_manifest_uses_actual_headers_beyond_stale_index(tmp_path):
    _write(tmp_path / "model.safetensors", _matrix())
    _write(tmp_path / "aux.safetensors", {"aux": ("I64", [1], bytes(8))})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: "model.safetensors" for name in _matrix()},
            }
        )
    )
    manifest = Exl3Manifest(tmp_path)
    assert manifest.tensors["aux"].read_bytes() == bytes(8)
    assert manifest.matrices["proj"].bits_x2 == 10


@pytest.mark.parametrize("bits_x2,marker", [(10, "mul1"), (12, "mcg"), (7, "mul1")])
def test_matrix_format_is_derived_from_each_header(tmp_path, bits_x2, marker):
    _write(tmp_path / "model.safetensors", _matrix(bits_x2=bits_x2, marker=marker))
    matrix = Exl3Manifest(tmp_path).matrices["proj"]
    assert (matrix.padded_in, matrix.padded_out) == (128, 256)
    assert matrix.bits_x2 == bits_x2
    assert matrix.codebook == (2 if marker == "mul1" else 1)
    matrix.validate_logical(120, 250)
    with pytest.raises(ValueError, match="logical/padded"):
        matrix.validate_logical(129, 250)


@pytest.mark.parametrize("corruption", ["marker", "scale", "half_mcg", "both_markers"])
def test_matrix_rejects_incompatible_codec_metadata(tmp_path, corruption):
    tensors = _matrix()
    if corruption == "marker":
        tensors["proj.mul1"] = ("I32", [], bytes(4))
    elif corruption == "scale":
        tensors["proj.suh"] = ("F16", [127], bytes(254))
    elif corruption == "half_mcg":
        tensors = _matrix(bits_x2=7, marker="mcg")
    else:
        tensors["proj.mcg"] = ("I32", [], struct.pack("<I", 0xCBAC1FED))
    _write(tmp_path / "model.safetensors", tensors)
    with pytest.raises(ValueError):
        Exl3Manifest(tmp_path)


def test_manifest_rejects_duplicate_sources_and_truncation(tmp_path):
    path = tmp_path / "model.safetensors"
    _write(path, _matrix())
    _write(tmp_path / "duplicate.safetensors", _matrix())
    with pytest.raises(ValueError, match="Duplicate EXL3 source"):
        Exl3Manifest(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="extent"):
        read_headers(path)


def test_ledger_requires_every_source_exactly_once(tmp_path):
    _write(tmp_path / "model.safetensors", _matrix())
    manifest = Exl3Manifest(tmp_path)
    ledger = Exl3LoadLedger(manifest)
    ledger.mark("proj.trellis", "file-backed", "table")
    with pytest.raises(ValueError, match="more than once"):
        ledger.mark("proj.trellis", "consumed", "linear")
    with pytest.raises(ValueError, match="Unknown"):
        ledger.mark("absent", "consumed", "linear")
    with pytest.raises(ValueError, match="Unaccounted"):
        ledger.finish()
    for name in ("proj.suh", "proj.svh", "proj.mul1"):
        ledger.mark(name, "consumed", "linear")
    ledger.finish()


@pytest.mark.parametrize("width", [640, 1152, 128])
def test_tp_rejects_cuts_inside_hadamard_blocks(width):
    with pytest.raises(ValueError, match="H128"):
        aligned_shard(width, width, rank=1, size=2)


def test_tp_preserves_full_blocks_and_replicated_padding():
    assert aligned_shard(2560, 2560, 1, 2) == slice(1280, 2560)
    assert aligned_shard(4304, 4352, 0, 1) == slice(0, 4352)
    with pytest.raises(ValueError, match="H128"):
        aligned_shard(4304, 4352, 0, 2)


def test_qwen_manifest_requires_both_auxiliary_files(tmp_path):
    _write(tmp_path / "model.safetensors", _matrix())
    with pytest.raises(ValueError, match="ngram_embedding.safetensors"):
        Qwen4ExpExl3Layout().check_manifest(Exl3Manifest(tmp_path))


def test_vision_fused_fallback_has_explicit_unused_ownership(tmp_path):
    tensors = {}
    for role in ("q", "k", "v"):
        prefix = f"model.visual.blocks.0.attn.{role}_proj"
        tensors.update(_matrix(prefix))
        tensors[f"{prefix}.bias"] = ("F16", [256], bytes(512))
    root = "model.visual.blocks.0.attn.qkv"
    tensors[f"{root}.weight"] = ("F16", [768, 128], bytes(768 * 128 * 2))
    tensors[f"{root}.bias"] = ("F16", [768], bytes(768 * 2))
    _write(tmp_path / "model.safetensors", tensors)
    manifest = Exl3Manifest(tmp_path)
    assert set(manifest.unused) == {f"{root}.weight", f"{root}.bias"}


def test_loading_noncontiguous_h128_output_shards_preserves_packed_values(tmp_path):
    import torch
    from safetensors import safe_open

    from vllm_sm80.kernels.exl3.quant import Exl3PackedWeight

    tensors = _matrix(padded_in=256, padded_out=512)
    original = torch.arange(16 * 32 * 80, dtype=torch.int32).to(torch.int16)
    original = original.reshape(16, 32, 80)
    tensors["proj.trellis"] = ("I16", [16, 32, 80], original.numpy().tobytes())
    suh = torch.arange(256, dtype=torch.float16)
    svh = torch.arange(512, dtype=torch.float16)
    tensors["proj.suh"] = ("F16", [256], suh.numpy().tobytes())
    tensors["proj.svh"] = ("F16", [512], svh.numpy().tobytes())
    path = tmp_path / "model.safetensors"
    _write(path, tensors)
    manifest = Exl3Manifest(tmp_path)
    ledger = Exl3LoadLedger(manifest)
    with torch.device("cpu"):
        weight = Exl3PackedWeight(
            manifest.matrices["proj"],
            128,
            256,
            slice(128, 256),
            [slice(0, 128), slice(384, 512)],
        )
    with safe_open(str(path), framework="pt", device="cpu") as source:
        weight.load({path: source}, ledger)
    expected = torch.cat([original[8:, :8], original[8:, 24:]], dim=1)
    assert torch.equal(weight.trellis, expected)
    assert torch.equal(weight.suh, suh[128:])
    assert torch.equal(weight.svh, torch.cat([svh[:128], svh[384:]]))
    ledger.finish()


def _ple(bits, dim, sizes, rows, prefix="model.layers.0.ple.ngram_embedding"):
    heads = len(sizes)
    offsets = [sum(sizes[:i]) for i in range(heads)]
    words = 1 + dim * bits // 16
    tensors = {
        f"{prefix}.trellis": ("I16", [rows, words], bytes(rows * words * 2)),
        f"{prefix}.head_bias": ("F16", [heads, dim], bytes(heads * dim * 2)),
        f"{prefix}.head_offsets": ("I64", [heads], struct.pack(f"<{heads}q", *offsets)),
        f"{prefix}.head_vocab_sizes": (
            "I64",
            [heads],
            struct.pack(f"<{heads}q", *sizes),
        ),
        f"{prefix}.layer_multipliers": ("I64", [2], struct.pack("<2q", 3, 5)),
    }
    metadata = {
        "format": "exl3_ngram_trellis",
        "version": "1",
        "codebook": "mul1",
        "K": str(bits),
        "row_dim": str(dim),
        "rows": str(rows),
    }
    return tensors, metadata


# The real Qwen3.8 table (K=6, 160 channels, 16 heads) and synthetic layouts.
@pytest.mark.parametrize(
    "bits,dim,sizes,rows",
    [(6, 160, (100, 27) * 8, 1024), (3, 32, (7, 5, 3), 16), (5, 16, (9,), 10)],
)
def test_ple_layout_is_derived_from_headers(tmp_path, bits, dim, sizes, rows):
    tensors, metadata = _ple(bits, dim, sizes, rows)
    _write(tmp_path / "model.safetensors", _matrix())
    _write(tmp_path / "ngram_embedding.safetensors", tensors, metadata)
    manifest = Exl3Manifest(tmp_path)
    prefix = "model.layers.0.ple.ngram_embedding"
    assert manifest.ple_tables[prefix].name == f"{prefix}.trellis"
    path = tmp_path / "ngram_embedding.safetensors"
    layout = ple_layout(read_headers(path), read_metadata(path), prefix)
    assert (layout.bits, layout.dim, layout.heads) == (bits, dim, len(sizes))
    assert (layout.rows, layout.row_bytes) == (rows, 2 + dim * bits // 8)
    assert layout.head_vocab_sizes == sizes


@pytest.mark.parametrize(
    "change,match",
    [
        ({"K": "5"}, "do not hold"),
        ({"K": "0"}, "do not hold"),
        ({"K": "six"}, "metadata K"),
        ({"row_dim": "48"}, "disagrees"),
        ({"codebook": "mcg"}, "format"),
    ],
)
def test_ple_layout_rejects_inconsistent_headers(tmp_path, change, match):
    tensors, metadata = _ple(3, 32, (7, 5, 3), 16)
    _write(tmp_path / "model.safetensors", _matrix())
    _write(tmp_path / "ngram_embedding.safetensors", tensors, {**metadata, **change})
    with pytest.raises(ValueError, match=match):
        Exl3Manifest(tmp_path)
