# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Override sidecars: the builder restores only precision-only casts, the
loader reads the source bytes in the source dtype and refuses any entry whose
hash or cast check fails."""

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm_sm80.kernels.exl3.override import apply_override
from vllm_sm80.kernels.exl3.utils import Exl3Manifest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "build_override_sidecar", REPO / "tools" / "sm80" / "build_override_sidecar.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)
REVISION = "0123456789abcdef0123456789abcdef01234567"
RESTORED = ["layers.0.ffn.gate.bias", "layers.0.ffn.gate.weight"]


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


_DTYPE = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header, payload = {}, bytearray()
    for name, tensor in tensors.items():
        start = len(payload)
        payload.extend(_raw(tensor))
        header[name] = {
            "dtype": _DTYPE[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _pin(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkpoints(tmp_path):
    """A two-shard source and an export with every tensor class."""
    generator = torch.Generator().manual_seed(0)

    def rand(*shape, dtype):
        return (torch.randn(*shape, generator=generator) * 7).to(dtype)

    source = {
        "layers.0.ffn.gate.bias": rand(16, dtype=torch.float32),
        # Small enough that F16 holds them only as subnormals.
        "layers.0.ffn.gate.weight": (
            rand(16, 8, dtype=torch.float32) * 1e-7
        ).bfloat16(),
        "layers.0.ffn.tuned.weight": rand(16, 8, dtype=torch.bfloat16),
        "layers.0.attn_norm.weight": rand(8, dtype=torch.bfloat16),
        "embed.weight": rand(32, 8, dtype=torch.bfloat16),
        "layers.0.attn.wq.weight": rand(8, 8, dtype=torch.bfloat16),
    }
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shards = {
        "model-1.safetensors": ["layers.0.ffn.gate.bias", "embed.weight"],
        "model-2.safetensors": [
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.tuned.weight",
            "layers.0.attn_norm.weight",
            "layers.0.attn.wq.weight",
        ],
    }
    for shard, names in shards.items():
        _write(source_dir / shard, {name: source[name] for name in names})
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {n: shard for shard, ns in shards.items() for n in ns}}
        )
    )
    tuned = source["layers.0.ffn.tuned.weight"].half()
    tuned[3, 5] += 0.25
    # layers.0.attn.wq.weight is quantised in an export: no plain tensor of
    # that name.
    export = {
        # RESTORABLE: precision-only casts.
        "layers.0.ffn.gate.bias": source["layers.0.ffn.gate.bias"].half(),
        "layers.0.ffn.gate.weight": source["layers.0.ffn.gate.weight"].half(),
        # DIFFERENT: one value changed beyond the cast.
        "layers.0.ffn.tuned.weight": tuned,
        # EXACT: the same dtype, and a widening.
        "layers.0.attn_norm.weight": source["layers.0.attn_norm.weight"],
        "embed.weight": source["embed.weight"].float(),
        # UNMAPPED: no source tensor.
        "extra.scale": torch.ones(4, dtype=torch.float32),
    }
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write(export_dir / "model.safetensors", export)
    out = tmp_path / "override"
    argv = ["--repo", "org/model", "--revision", REVISION]
    argv += ["--export", str(export_dir), "--out", str(out)]
    assert builder.main(argv + ["--source-dir", str(source_dir)]) == 0
    manifest_sha = _pin(out / builder.MANIFEST)
    # A non-empty --out is refused without --replace and left untouched.
    with pytest.raises(ValueError, match="is not empty"):
        builder.main(argv + ["--source-dir", str(source_dir)])
    assert manifest_sha == _pin(out / builder.MANIFEST)
    argv += ["--replace", "--source-dir", str(source_dir)]
    assert builder.main(argv) == 0
    path = out / builder.MANIFEST
    return source, export, export_dir, path


def test_builder_restores_only_precision_only_casts(checkpoints):
    source, _, _, path = checkpoints
    spec = json.loads(path.read_text())
    assert spec["format"] == "override-sidecar/2"
    assert spec["source"] == {"repo": "org/model", "revision": REVISION}
    rows = [(t["name"], t["source_dtype"], t["export_dtype"]) for t in spec["tensors"]]
    assert rows == [
        ("layers.0.ffn.gate.bias", "F32", "F16"),
        ("layers.0.ffn.gate.weight", "BF16", "F16"),
    ]
    assert [t["cast_check"] for t in spec["tensors"]] == [
        {"export_dtype": "F16", "equal": True, "compared_elems": n} for n in (16, 128)
    ]
    for record in spec["tensors"]:
        expected = hashlib.sha256(_raw(source[record["name"]])).hexdigest()
        assert record["sha256"] == expected
    report_path = path.parent / spec["report"]
    assert spec["report_sha256"] == _pin(report_path)
    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "EXACT": 2,
        "RESTORABLE": 2,
        "DIFFERENT": 1,
        "UNMAPPED": 1,
    }
    rows = {row["name"]: row for row in report["tensors"]}
    assert rows["embed.weight"]["cls"] == "EXACT"
    assert rows["layers.0.attn_norm.weight"]["cls"] == "EXACT"
    assert rows["extra.scale"]["cls"] == "UNMAPPED"
    tuned = rows["layers.0.ffn.tuned.weight"]
    assert (tuned["cls"], tuned["reason"], tuned["n_diff"]) == (
        "DIFFERENT",
        "values",
        1,
    )
    assert tuned["max_abs"] > 0.2


def test_loaded_tensors_equal_source_bytes(checkpoints):
    source, _, export_dir, path = checkpoints
    manifest = Exl3Manifest(export_dir)
    assert apply_override(manifest, str(path), _pin(path)) == RESTORED
    for name in RESTORED:
        header = manifest.tensors[name]
        # The loader reads plain tensors through each header's file.
        with safe_open(str(header.path), framework="pt", device="cpu") as file:
            loaded = file.get_tensor(name)
        assert loaded.dtype == source[name].dtype
        assert _raw(loaded) == _raw(source[name])
    for name in ("layers.0.attn_norm.weight", "layers.0.ffn.tuned.weight"):
        assert manifest.tensors[name].path == export_dir / "model.safetensors"
    # The FP32 gate parameters keep every source bit.
    param = torch.empty(16, dtype=torch.float32)
    with safe_open(
        str(manifest.tensors["layers.0.ffn.gate.bias"].path), "pt", device="cpu"
    ) as file:
        param.copy_(file.get_tensor("layers.0.ffn.gate.bias"))
    assert _raw(param) == _raw(source["layers.0.ffn.gate.bias"])


def test_refuses_tensor_sha256_mismatch(checkpoints):
    _, _, export_dir, path = checkpoints
    sidecar = path.parent / builder.SIDECAR
    data = bytearray(sidecar.read_bytes())
    data[-1] ^= 1
    sidecar.write_bytes(bytes(data))
    manifest = Exl3Manifest(export_dir)
    before = dict(manifest.tensors)
    with pytest.raises(ValueError, match="sha256 mismatch: layers.0.ffn.gate"):
        apply_override(manifest, str(path), _pin(path))
    assert manifest.tensors == before


def test_refuses_unpinned_or_changed_manifest(checkpoints):
    _, _, export_dir, path = checkpoints
    manifest = Exl3Manifest(export_dir)
    with pytest.raises(ValueError, match="needs VLLM_EXL3_OVERRIDE_SHA256"):
        apply_override(manifest, str(path), "")
    with pytest.raises(ValueError, match="differs from the pinned"):
        apply_override(manifest, str(path), "0" * 64)


@pytest.mark.parametrize(
    "cast_check",
    [None, {"export_dtype": "F16", "equal": False, "compared_elems": 128}],
)
def test_refuses_entry_without_passing_cast_check(checkpoints, cast_check):
    _, _, export_dir, path = checkpoints
    spec = json.loads(path.read_text())
    entry = spec["tensors"][1]
    if cast_check is None:
        del entry["cast_check"]
    else:
        entry["cast_check"] = cast_check
    path.write_text(json.dumps(spec))
    manifest = Exl3Manifest(export_dir)
    with pytest.raises(ValueError, match="no passing cast check: layers.0.ffn.gate"):
        apply_override(manifest, str(path), _pin(path))


def test_refuses_export_that_is_not_the_source_cast(checkpoints):
    _, export, export_dir, path = checkpoints
    changed = dict(export)
    changed["layers.0.ffn.gate.weight"] = export["layers.0.ffn.gate.weight"].clone()
    changed["layers.0.ffn.gate.weight"][0, 0] += 1
    _write(export_dir / "model.safetensors", changed)
    manifest = Exl3Manifest(export_dir)
    before = dict(manifest.tensors)
    with pytest.raises(ValueError, match="not the source cast to F16"):
        apply_override(manifest, str(path), _pin(path))
    assert manifest.tensors == before


def test_refuses_export_other_than_recorded(checkpoints, tmp_path):
    _, export, _, path = checkpoints
    other = tmp_path / "other"
    other.mkdir()
    changed = dict(export)
    changed["layers.0.ffn.gate.bias"] = export["layers.0.ffn.gate.bias"].bfloat16()
    _write(other / "model.safetensors", changed)
    with pytest.raises(ValueError, match="differs from the override manifest"):
        apply_override(Exl3Manifest(other), str(path), _pin(path))
