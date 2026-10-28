#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a hybrid EXL3 checkpoint: chosen modules from an EXL3 export, the rest
from source checkpoints in their own format, as a directory of symlinks.

    python tools/sm80/build_hybrid_checkpoint.py \\
        --export EXL3_DIR --source SOURCE_DIR [--source SOURCE_DIR ...] \\
        --exl3-modules REGEX --method NAME --out OUT_DIR \\
        [--drop REGEX] [--link FILE ...]

--exl3-modules is searched (re.search) in checkpoint tensor and module names:
the export's matching tensors are taken and every other export tensor is
left out; the sources' matching tensors are left out too. Every other source
tensor is taken, except those matching --drop. A source directory offers the
tensors its model.safetensors.index.json maps to files it holds; a tensor two
sources offer is refused. Every export module left out must come from a
source (``<module>.weight``; an export's ``<module>.slice.<g>`` groups are
one module), and every plain export tensor left out must too.

OUT holds symlinks to the shard files the selection reads (whole files: the
index selects the tensors, vllm_sm80/kernels/exl3/hybrid.py), an index of
the selection, the export's config.json with its quantization_config
replaced by the hybrid one (also written as quantization_config.json), the
--link files from the export (default: its tokenizer files) and
hybrid-build.json (inputs, counts, bytes per origin, sha256 of every
written file). The hybrid quantization_config routes the --exl3-modules
modules to "exl3", each class of source module with a float weight to
"unquantized" and each other class to --method, the first source's own
quantization_config renamed to --method. A class is a module name with its
numeric path parts generalised; a class whose modules disagree gets one
rule per module. There is no default: a module no rule names is refused.
An existing non-empty OUT is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

INDEX = "model.safetensors.index.json"
PACKED_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg", ".mul1")
FLOAT_DTYPES = ("BF16", "F16", "F32")
DEFAULT_LINKS = ("tokenizer.json", "tokenizer_config.json")


def read_header(path: Path) -> dict:
    with path.open("rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(length))
    header.pop("__metadata__", None)
    return header


def offered(directory: Path) -> dict[str, str]:
    """Tensor -> file of a checkpoint directory: its index's entries whose
    file it holds."""
    weight_map = json.loads((directory / INDEX).read_text())["weight_map"]
    return {n: f for n, f in weight_map.items() if (directory / f).is_file()}


def module_of(name: str) -> str:
    """The module a tensor belongs to (an export's wo_a.slice.<g>.* -> wo_a)."""
    module = name.rsplit(".", 1)[0]
    return re.sub(r"\.slice\.\d+$", "", module)


def class_of(module: str) -> str:
    """The rule pattern of a module's class."""
    parts = [
        r"\d+" if part.isdigit() else re.escape(part) for part in module.split(".")
    ]
    return "^" + r"\.".join(parts) + "$"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args) -> dict:
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    exl3 = re.compile(args.exl3_modules)
    drop = re.compile(args.drop) if args.drop else None
    export: Path = args.export.resolve()
    export_map = offered(export)
    selected: dict[str, tuple[Path, str]] = {}
    for name, filename in export_map.items():
        if exl3.search(name):
            selected[name] = (export, filename)
    if not selected:
        raise SystemExit("--exl3-modules matches no export tensor")
    sources = [s.resolve() for s in args.source]
    origin: dict[str, Path] = {}
    for source in sources:
        for name, filename in offered(source).items():
            if exl3.search(name) or (drop is not None and drop.search(name)):
                continue
            if name in origin:
                raise SystemExit(f"{name} is offered by {origin[name]} and {source}")
            origin[name] = source
            selected[name] = (source, filename)
    # Coverage: every export module or plain tensor left out comes from a source.
    missing = set()
    for name in export_map:
        if exl3.search(name):
            continue
        if name.endswith(PACKED_SUFFIXES):
            if f"{module_of(name)}.weight" not in origin:
                missing.add(module_of(name))
        elif name not in origin:
            missing.add(name)
    if missing:
        raise SystemExit(f"Not offered by any source: {sorted(missing)[:20]}")
    # Headers of the selected tensors: dtypes for the rules, bytes per origin.
    headers: dict[Path, dict] = {}
    dtypes: dict[str, str] = {}
    nbytes: dict[str, int] = defaultdict(int)
    for name, (directory, filename) in selected.items():
        path = directory / filename
        if path not in headers:
            headers[path] = read_header(path)
        entry = headers[path][name]
        dtypes[name] = entry["dtype"]
        start, end = entry["data_offsets"]
        nbytes[str(directory)] += end - start
    # Rules: the EXL3 modules first, then one per source module class.
    kinds: dict[str, dict[str, str]] = defaultdict(dict)
    for name, dtype in dtypes.items():
        if name.endswith(".weight") and selected[name][0] != export:
            module = name.removesuffix(".weight")
            kind = "unquantized" if dtype in FLOAT_DTYPES else args.method
            kinds[class_of(module)][module] = kind
    rules = [{"match": args.exl3_modules, "method": "exl3"}]
    for pattern in sorted(kinds):
        by_module = kinds[pattern]
        if len(set(by_module.values())) == 1:
            rules.append({"match": pattern, "method": next(iter(by_module.values()))})
        else:
            rules.extend(
                {"match": class_of(module), "method": kind}
                for module, kind in sorted(by_module.items())
            )
    config = json.loads((export / "config.json").read_text())
    export_quant = dict(config.get("quantization_config") or {})
    if export_quant.get("quant_method") != "exl3":
        raise SystemExit("The export's quantization_config is not exl3")
    source_config = json.loads((sources[0] / "config.json").read_text())
    secondary = dict(source_config.get("quantization_config") or {})
    if not secondary:
        raise SystemExit(f"{sources[0]} has no quantization_config")
    secondary["quant_method"] = args.method
    export_quant.pop("quant_method")
    export_quant.pop("tensor_storage", None)
    quant = {
        "quant_method": "exl3",
        "module_quantization": {"rules": rules},
        "non_routed_quantization": secondary,
        "exl3_export": export_quant,
    }
    config["quantization_config"] = quant
    # Write the overlay.
    out.mkdir(parents=True, exist_ok=True)
    files = sorted({(d, f) for d, f in selected.values()})
    for directory, filename in files:
        os.symlink(directory / filename, out / filename)
    for filename in args.link or DEFAULT_LINKS:
        if (export / filename).exists():
            os.symlink(export / filename, out / filename)
    weight_map = {name: filename for name, (_, filename) in sorted(selected.items())}
    total = sum(nbytes.values())
    written = {
        INDEX: {"metadata": {"total_size": total}, "weight_map": weight_map},
        "config.json": config,
        "quantization_config.json": quant,
    }
    for filename, content in written.items():
        (out / filename).write_text(json.dumps(content, indent=2) + "\n")
    record = {
        "format": "hybrid-build/1",
        "export": str(export),
        "sources": [str(s) for s in sources],
        "exl3_modules": args.exl3_modules,
        "drop": args.drop,
        "method": args.method,
        "inputs_sha256": {
            str(d / name): sha256(d / name)
            for d in [export, *sources]
            for name in (INDEX, "config.json")
            if (d / name).exists()
        },
        "tensors": {
            str(d): sum(1 for o, _ in selected.values() if o == d)
            for d in [export, *sources]
        },
        "bytes": dict(nbytes),
        "files": [str(d / f) for d, f in files],
        "rules": len(rules),
        "written_sha256": {name: sha256(out / name) for name in written},
    }
    (out / "hybrid-build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", required=True, type=Path, help="EXL3 export dir")
    parser.add_argument(
        "--source", required=True, type=Path, action="append", help="source dir"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--exl3-modules", required=True, help="regex of EXL3 modules")
    parser.add_argument("--method", required=True, help="the sources' vLLM method")
    parser.add_argument("--drop", help="regex of source tensors to leave out")
    parser.add_argument("--link", action="append", help="export file to link")
    record = build(parser.parse_args(argv))
    json.dump(
        {k: record[k] for k in ("tensors", "bytes", "rules", "written_sha256")},
        sys.stdout,
        indent=2,
    )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
