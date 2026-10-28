#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an override sidecar: source-checkpoint tensors an export stored lossily.

An export (EXL3 or any safetensors re-encode) may store unquantised tensors at
a dtype that cannot hold every source value (an FP32 router bias as FP16, a
BF16 gate as FP16). This tool compares every plain export tensor with the
pinned source checkpoint and classifies it:

    EXACT       the export holds the source values exactly
                (export cast to the source dtype equals the source bitwise);
    RESTORABLE  the export is the source cast to the export dtype, bitwise,
                and that cast lost values: a precision-only difference;
    DIFFERENT   neither (changed values, or another shape): reported with
                its largest absolute difference, never restored;
    UNMAPPED    no source tensor of that name: reported, never restored.

The RESTORABLE tensors' source bytes go into one safetensors sidecar, and a
manifest beside it records, per tensor, the source shard and dtype, the
export dtype, the cast check and the sha256 of the bytes. A report lists every
classified tensor. The loader (VLLM_EXL3_OVERRIDE, pinned by
VLLM_EXL3_OVERRIDE_SHA256) verifies the manifest, the hashes and the cast
check again, and loads the source bytes in place of the export's.

    python tools/sm80/build_override_sidecar.py \\
        --repo ORG/NAME --revision COMMIT --export EXPORT_DIR --out OUT_DIR \\
        [--names NAMES_FILE] [--map MAP_JSON] [--source-dir LOCAL_SOURCE] \
        [--replace]

--names restricts the classification to the listed export tensors (one per
line); by default every plain export tensor is classified. --map is a JSON
object of export tensor name -> source tensor name for exports that rename
tensors; unlisted names map to themselves. The source is read from the
repository's resolve URL at the pinned revision with HTTP Range requests
(only headers and the compared tensors), or from --source-dir, a local copy of
that revision.

A built sidecar is pinned by sha256 wherever it is served, so a new build
goes into a new directory: an existing non-empty --out is refused unless
--replace is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import torch

FORMAT = "override-sidecar/2"
REPORT_FORMAT = "override-report/1"
SIDECAR = "override.safetensors"
MANIFEST = "override.json"
REPORT = "override-report.json"
CLASSES = ("EXACT", "RESTORABLE", "DIFFERENT", "UNMAPPED")
DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "I64": torch.int64,
    "F64": torch.float64,
}
# Quantised-tensor parts: never plain tensors.
PACKED_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg", ".mul1")


def as_tensor(dtype: str, shape: list[int], data: bytes) -> torch.Tensor:
    if not data:
        return torch.empty(shape, dtype=DTYPES[dtype])
    return torch.frombuffer(bytearray(data), dtype=DTYPES[dtype]).reshape(shape)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


class Source:
    """Byte ranges of the source checkpoint's files at one revision."""

    def __init__(self, repo: str, revision: str, local: Path | None):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.local = local
        self.headers: dict[str, tuple[dict, int]] = {}

    def read(self, filename: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of one file."""
        if end <= start:
            return b""
        if self.local is not None:
            with (self.local / filename).open("rb") as file:
                file.seek(start)
                data = file.read(end - start)
        else:
            request = urllib.request.Request(
                self.base + filename, headers={"Range": f"bytes={start}-{end - 1}"}
            )
            with urllib.request.urlopen(request) as response:
                if response.status != 206:
                    raise ValueError(f"{filename}: range request not honoured")
                data = response.read()
        if len(data) != end - start:
            raise ValueError(f"{filename}: short read at {start}")
        return data

    def index(self) -> dict[str, str]:
        if self.local is not None:
            raw = (self.local / "model.safetensors.index.json").read_bytes()
        else:
            with urllib.request.urlopen(
                self.base + "model.safetensors.index.json"
            ) as response:
                raw = response.read()
        return json.loads(raw)["weight_map"]

    def header(self, filename: str) -> tuple[dict, int]:
        """A shard's parsed header and the offset of its payload."""
        if filename not in self.headers:
            length = struct.unpack("<Q", self.read(filename, 0, 8))[0]
            header = json.loads(self.read(filename, 8, 8 + length))
            self.headers[filename] = (header, 8 + length)
        return self.headers[filename]

    def tensor(self, filename: str, name: str) -> tuple[dict, bytes]:
        header, start = self.header(filename)
        entry = header[name]
        begin, end = entry["data_offsets"]
        return entry, self.read(filename, start + begin, start + end)


class Export:
    """Every tensor of a local safetensors checkpoint."""

    def __init__(self, directory: Path):
        self.tensors: dict[str, tuple[Path, dict, int]] = {}
        for path in sorted(directory.glob("*.safetensors")):
            with path.open("rb") as file:
                length = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensors:
                    raise ValueError(f"Duplicate export tensor: {name}")
                self.tensors[name] = (path, entry, 8 + length)
        if not self.tensors:
            raise ValueError(f"No safetensors files in {directory}")

    def plain(self) -> list[str]:
        return sorted(n for n in self.tensors if not n.endswith(PACKED_SUFFIXES))

    def tensor(self, name: str) -> tuple[dict, bytes]:
        path, entry, start = self.tensors[name]
        begin, end = entry["data_offsets"]
        with path.open("rb") as file:
            file.seek(start + begin)
            data = file.read(end - begin)
        if len(data) != end - begin:
            raise ValueError(f"{name}: short read")
        return entry, data


def classify(official: torch.Tensor, exported: torch.Tensor) -> dict:
    """The class of one export tensor against its source tensor."""
    row = {"numel": exported.numel()}
    if list(official.shape) != list(exported.shape):
        row.update(cls="DIFFERENT", reason="shape", source_shape=list(official.shape))
        return row
    if same_bits(exported.to(official.dtype), official):
        row["cls"] = "EXACT"
        return row
    cast = official.to(exported.dtype)
    if same_bits(cast, exported):
        row["cls"] = "RESTORABLE"
        row["cast_check"] = {
            "export_dtype": None,
            "equal": True,
            "compared_elems": exported.numel(),
        }
        return row
    diff = (exported.double() - official.double()).abs()
    row.update(
        cls="DIFFERENT",
        reason="values",
        n_diff=int((cast.view(torch.uint8) != exported.view(torch.uint8)).sum()),
        max_abs=float(diff.nan_to_num(nan=float("inf")).max()) if diff.numel() else 0.0,
    )
    return row


def write_safetensors(path: Path, tensors: dict[str, tuple[dict, bytes]], meta) -> None:
    header: dict = {"__metadata__": meta}
    offset = 0
    for name, (entry, data) in tensors.items():
        header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for _, data in tensors.values():
            file.write(data)
    partial.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.out.exists() and any(args.out.iterdir()) and not args.replace:
        raise ValueError(
            f"{args.out} is not empty; build into a new directory (or --replace)"
        )
    source = Source(args.repo, args.revision, args.source_dir)
    source_map = source.index()
    export = Export(args.export)
    mapping = json.loads(args.map.read_text()) if args.map is not None else {}
    names = export.plain()
    if args.names is not None:
        listed = sorted(
            {
                line.strip()
                for line in args.names.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            }
        )
        unknown = [n for n in listed if n not in names]
        if unknown:
            raise ValueError(f"Not plain export tensors: {unknown[:8]}")
        names = listed
    rows, tensors, records = [], {}, []
    for name in names:
        source_name = mapping.get(name, name)
        exported_entry, exported_data = export.tensor(name)
        row = {
            "name": name,
            "source_name": source_name,
            "export_dtype": exported_entry["dtype"],
            "shape": exported_entry["shape"],
        }
        if source_name not in source_map:
            row.update(cls="UNMAPPED", numel=None)
            rows.append(row)
            continue
        shard = source_map[source_name]
        entry, data = source.tensor(shard, source_name)
        row.update(source_shard=shard, source_dtype=entry["dtype"])
        row.update(
            classify(
                as_tensor(entry["dtype"], entry["shape"], data),
                as_tensor(
                    exported_entry["dtype"], exported_entry["shape"], exported_data
                ),
            )
        )
        rows.append(row)
        if row["cls"] != "RESTORABLE":
            continue
        row["cast_check"]["export_dtype"] = exported_entry["dtype"]
        sha256 = hashlib.sha256(data).hexdigest()
        tensors[name] = (entry, data)
        records.append(
            {
                "name": name,
                "source_name": source_name,
                "shape": entry["shape"],
                "source_shard": shard,
                "source_dtype": entry["dtype"],
                "export_shard": export.tensors[name][0].name,
                "export_dtype": exported_entry["dtype"],
                "cast_check": row["cast_check"],
                "sha256": sha256,
            }
        )
        row["sha256"] = sha256
    counts = {cls: sum(r["cls"] == cls for r in rows) for cls in CLASSES}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "format": REPORT_FORMAT,
        "source": {"repo": args.repo, "revision": args.revision},
        "export": args.export.name,
        "counts": counts,
        "tensors": rows,
    }
    (args.out / REPORT).write_text(json.dumps(report, indent=1) + "\n")
    if not records:
        raise ValueError(f"No RESTORABLE tensor; report in {args.out / REPORT}")
    sidecar = args.out / SIDECAR
    write_safetensors(
        sidecar,
        tensors,
        {"source_repo": args.repo, "source_revision": args.revision},
    )
    manifest = {
        "format": FORMAT,
        "source": {"repo": args.repo, "revision": args.revision},
        "export": args.export.name,
        "sidecar": SIDECAR,
        "sidecar_sha256": sha256_file(sidecar),
        "report": REPORT,
        "report_sha256": sha256_file(args.out / REPORT),
        "tensors": records,
    }
    (args.out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="source repository id")
    parser.add_argument("--revision", required=True, help="source commit")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--names", type=Path, help="export tensors to classify")
    parser.add_argument("--map", type=Path, help="JSON export name -> source name")
    parser.add_argument(
        "--source-dir", type=Path, help="local copy of the source at --revision"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="overwrite a non-empty --out (never one a served setting pins)",
    )
    args = parser.parse_args(argv)
    manifest, report = build(args)
    for row in report["tensors"]:
        if row["cls"] == "EXACT":
            continue
        detail = row.get("sha256") or row.get("reason") or ""
        if row.get("max_abs") is not None:
            detail += f" max_abs={row['max_abs']:.6g} n_diff={row['n_diff']}"
        print(
            f"{row['cls']}\t{row['name']}\t{row.get('source_dtype')} -> "
            f"{row['export_dtype']}\t{detail}"
        )
    print(" ".join(f"{cls}={n}" for cls, n in report["counts"].items()))
    print(f"{SIDECAR} sha256 {manifest['sidecar_sha256']}")
    print(f"{REPORT} sha256 {manifest['report_sha256']}")
    print(f"{MANIFEST} sha256 {sha256_file(args.out / MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
