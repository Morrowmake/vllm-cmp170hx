# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-checkpoint bytes for the plain tensors an export stored lossily.

An override sidecar (built by tools/sm80/build_override_sidecar.py) is a
safetensors file plus a JSON manifest naming the source repository and
revision and, per tensor, its name, shape, source shard and dtype, the
export's dtype, the cast check and the sha256 of the source bytes. A tensor
is restored only when the export's copy is the source tensor cast to the
export dtype, bit for bit: a precision-only difference, never a changed
value. VLLM_EXL3_OVERRIDE names the manifest and VLLM_EXL3_OVERRIDE_SHA256
pins it. Every listed tensor is verified and then read from the sidecar, in
the source dtype, wherever the loader would have read the export's copy.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm_sm80.kernels.exl3.utils import (
    Exl3Manifest,
    TensorHeader,
    _unique_object,
    read_headers,
)

logger = init_logger(__name__)
FORMAT = "override-sidecar/2"
_PACKED_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg", ".mul1")
_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "I64": torch.int64,
    "F64": torch.float64,
}


def _tensor(header: TensorHeader) -> torch.Tensor:
    data = bytearray(header.read_bytes())
    if not data:
        return torch.empty(header.shape, dtype=_DTYPES[header.dtype])
    return torch.frombuffer(data, dtype=_DTYPES[header.dtype]).reshape(header.shape)


def _is_cast_of(export: TensorHeader, source: TensorHeader) -> bool:
    """Whether the export's bytes are the source tensor cast to its dtype."""
    cast = _tensor(source).to(_DTYPES[export.dtype])
    return torch.equal(cast.view(torch.uint8), _tensor(export).view(torch.uint8))


def apply_override(manifest: Exl3Manifest, path: str, sha256: str) -> list[str]:
    """Point ``manifest`` at the sidecar's tensors; return the replaced names.

    Refuses, before replacing anything, a manifest that differs from the pin,
    a sidecar whose tensors differ from the manifest's list, a tensor the
    export lacks or holds at another dtype or shape than recorded, a packed
    tensor, an entry without a passing cast check, any tensor whose bytes do
    not hash to the recorded sha256, and any export tensor that is not the
    sidecar tensor cast to the export dtype.
    """
    if not sha256:
        raise ValueError("VLLM_EXL3_OVERRIDE needs VLLM_EXL3_OVERRIDE_SHA256")
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256.lower():
        raise ValueError(
            f"Override manifest sha256 {digest} differs from the pinned {sha256}"
        )
    spec = json.loads(raw, object_pairs_hook=_unique_object)
    if spec.get("format") != FORMAT:
        raise ValueError(f"Unsupported override manifest format: {spec.get('format')}")
    entries = spec["tensors"]
    names = [entry["name"] for entry in entries]
    sidecar = read_headers(manifest_path.parent / spec["sidecar"])
    if len(set(names)) != len(names) or set(names) != sidecar.keys():
        raise ValueError("Override sidecar and manifest list different tensors")
    for entry in entries:
        name = entry["name"]
        shape = tuple(entry["shape"])
        export = manifest.tensors.get(name)
        if export is None:
            raise ValueError(f"Override tensor is not in the checkpoint: {name}")
        if name.endswith(_PACKED_SUFFIXES):
            raise ValueError(f"Override tensor is a packed EXL3 part: {name}")
        if export.dtype != entry["export_dtype"] or export.shape != shape:
            raise ValueError(
                f"Checkpoint tensor differs from the override manifest: {name} "
                f"is {export.dtype}{list(export.shape)}, recorded "
                f"{entry['export_dtype']}{list(shape)}"
            )
        if entry.get("cast_check") != {
            "export_dtype": entry["export_dtype"],
            "equal": True,
            "compared_elems": math.prod(shape),
        }:
            raise ValueError(f"Override tensor has no passing cast check: {name}")
        source = sidecar[name]
        if source.dtype != entry["source_dtype"] or source.shape != shape:
            raise ValueError(f"Override sidecar tensor differs from manifest: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Override tensor sha256 mismatch: {name}")
        if not _is_cast_of(export, source):
            raise ValueError(
                f"Checkpoint tensor is not the source cast to {export.dtype}: {name}"
            )
    for entry in entries:
        manifest.tensors[entry["name"]] = sidecar[entry["name"]]
    logger.info(
        "EXL3 override from %s@%s: %d tensors loaded at source dtype: %s",
        spec["source"]["repo"],
        spec["source"]["revision"],
        len(entries),
        ", ".join(
            f"{e['name']} {e['export_dtype']}->{e['source_dtype']}" for e in entries
        ),
    )
    return names
