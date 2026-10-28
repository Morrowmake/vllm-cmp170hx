# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure and rewrite the upstream seam pins in seams.yaml.

A seam is an upstream object the sm80 fork code relies on without editing
it. seams.yaml records a fingerprint of each one; test_seams.py fails when
an upstream change moves a fingerprint. After checking that change against
the row's `why`, run this script to record the new fingerprints:

    python tests/sm80/update_seams.py            # rewrite every row
    python tests/sm80/update_seams.py ID [ID..]  # rewrite only these rows
    python tests/sm80/update_seams.py --new-base # start the pins of a new
                                                 # upstream base (below)

The plugin runs on more than one upstream vLLM base, and a seam's
fingerprint is the object as one base has it. seams.yaml pins the base named
in FORK.md; seams.<base12>.yaml (the first 12 hex digits of the commit) is
the reviewed re-pin for another base. A tree is checked, and rewritten, by
the file of its upstream base when there is one, else by seams.yaml. The
base is that of the vLLM the tests import: the `upstream_commit` of the
assembly.json beside an assembled engine (<dir>/engine/vllm next to
<dir>/assembly.json), else the `Upstream base:` line of FORK.md.

Run it with the repository root importable (for example PYTHONPATH=.).
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

SEAMS_PATH = Path(__file__).with_name("seams.yaml")
REPO = Path(__file__).absolute().parents[2]
FORK_BASE = re.compile(r"^Upstream base: ([0-9a-f]{40})$", re.M)

# The repr of an object without its own __repr__ (a default-argument instance,
# e.g. a registry singleton) carries its memory address, which differs per
# process.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")

CODE_KINDS = ("function", "method")
ATTR_KINDS = ("module_attr", "class_attr")
KINDS = CODE_KINDS + ATTR_KINDS + ("dict_contract",)

FIELD_ORDER = ("id", "kind", "target", "why", "sha256", "signature", "type", "value")

HEADER = """\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Upstream seams the sm80 fork code relies on without editing them, pinned as
# one upstream base has them (seams.yaml: the base in FORK.md;
# seams.<base12>.yaml: the re-pin for another base).
# Checked by tests/sm80/test_seams.py. After checking an upstream change to
# a seam, rewrite the pins with: python tests/sm80/update_seams.py [ID ...]
#
# Fields:
#   id         unique row name
#   kind       function | method | class_attr | module_attr | dict_contract
#   target     'module:qualname'
#   why        what the fork code depends on
#   sha256     sha256 of textwrap.dedent(inspect.getsource(obj)) (code kinds)
#   signature  str(inspect.signature(obj)), object addresses removed (code kinds)
#   type       type name of the value (attr kinds and dict_contract)
#   value      exact value, for constants only (attr kinds, optional)
#   dict_contract rows also require every value to be a zero-arg callable.
"""


def assembled_engine(vllm_origin: str | None = None) -> tuple[Path, str] | None:
    """(engine directory, upstream_commit) of the assembled engine whose vLLM
    the tests import (``vllm_origin``: that package's ``__init__.py``, default
    the importable one): <dir>/engine/vllm next to <dir>/assembly.json. None
    for a single tree."""
    if vllm_origin is None:
        spec = importlib.util.find_spec("vllm")
        vllm_origin = spec.origin if spec is not None else None
    if vllm_origin is None:
        return None
    engine = Path(vllm_origin).resolve().parents[1]
    assembly = engine.parent / "assembly.json"
    if engine.name != "engine" or not assembly.is_file():
        return None
    return engine, json.loads(assembly.read_text())["upstream_commit"]


def upstream_base() -> str | None:
    """The upstream base of the vLLM the tests import: an assembled engine's
    assembly.json `upstream_commit`, else FORK.md's `Upstream base:`, else
    None."""
    engine = assembled_engine()
    if engine is not None:
        return engine[1]
    fork_map = REPO / "FORK.md"
    found = FORK_BASE.search(fork_map.read_text()) if fork_map.is_file() else None
    return found.group(1) if found else None


def base_seams_path(base: str) -> Path:
    """The reviewed re-pin for upstream `base` (it may not exist)."""
    return SEAMS_PATH.with_name(f"seams.{base[:12]}.yaml")


def seams_path(base: str | None) -> Path:
    """The pins a tree on upstream `base` is checked by: its base's re-pin
    when there is one, else seams.yaml."""
    if base is not None and base_seams_path(base).is_file():
        return base_seams_path(base)
    return SEAMS_PATH



def load_rows(path: Path = SEAMS_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    rows = data["seams"]
    ids = [row["id"] for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate seam ids in {path}: {duplicates}")
    for row in rows:
        if row["kind"] not in KINDS:
            raise ValueError(f"seam {row['id']}: unknown kind {row['kind']!r}")
    return rows


def resolve(target: str) -> Any:
    """Import 'module:qualname' and return the object it names."""
    module_name, _, qualname = target.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def source_sha256(obj: Any) -> str:
    source = textwrap.dedent(inspect.getsource(obj))
    return hashlib.sha256(source.encode()).hexdigest()


def signature_text(obj: Any) -> str:
    """str(inspect.signature(obj)) without the per-process object addresses
    of default-argument reprs ('<... object at 0x7f...>' -> '<... object>')."""
    return _OBJECT_ADDRESS.sub(">", str(inspect.signature(obj)))


def zero_arg_callable(value: Any) -> bool:
    if not callable(value):
        return False
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    return all(
        p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in signature.parameters.values()
    )


def measure(row: dict[str, Any]) -> dict[str, Any]:
    """Return the pinned fields of `row` as found in the importable tree.

    Raises whatever import or attribute lookup raises for a missing target.
    """
    obj = resolve(row["target"])
    kind = row["kind"]
    if kind in CODE_KINDS:
        return {
            "sha256": source_sha256(obj),
            "signature": signature_text(obj),
        }
    measured: dict[str, Any] = {"type": type(obj).__name__}
    if kind in ATTR_KINDS:
        if "value" in row:
            measured["value"] = obj
    else:  # dict_contract
        items = obj.items() if isinstance(obj, dict) else ()
        bad = sorted(str(k) for k, v in items if not zero_arg_callable(v))
        if bad:
            raise TypeError(
                f"{row['target']} values must be zero-arg callables; "
                f"offending keys: {bad}"
            )
    return measured


def dump_rows(rows: list[dict[str, Any]], path: Path = SEAMS_PATH) -> None:
    ordered = []
    for row in rows:
        known = [k for k in FIELD_ORDER if k in row]
        extra = [k for k in row if k not in FIELD_ORDER]
        ordered.append({k: row[k] for k in known + extra})
    body = yaml.safe_dump(
        {"seams": ordered}, sort_keys=False, width=4096, default_flow_style=False
    )
    path.write_text(HEADER + "\n" + body.replace("\n- id:", "\n\n- id:"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="seam ids to rewrite (default: all)")
    parser.add_argument(
        "--file",
        type=Path,
        help="the pins to rewrite (default: those the tree is checked by)",
    )
    parser.add_argument(
        "--new-base",
        action="store_true",
        help="write seams.<base12>.yaml for the tree's upstream base, "
        "started from seams.yaml when it does not exist yet",
    )
    args = parser.parse_args(argv)

    if args.file is not None:
        source = target = args.file
    elif args.new_base:
        base = upstream_base()
        if base is None:
            parser.error("--new-base: the tree's upstream base is unknown")
        target = base_seams_path(base)
        source = target if target.is_file() else SEAMS_PATH
    else:
        source = target = seams_path(upstream_base())
    print(f"pins: {target}")
    rows = load_rows(source)
    known_ids = {row["id"] for row in rows}
    unknown = sorted(set(args.ids) - known_ids)
    if unknown:
        parser.error(f"unknown seam ids: {unknown}")
    selected = set(args.ids) or known_ids

    failed = 0
    for row in rows:
        if row["id"] not in selected:
            continue
        try:
            measured = measure(row)
        except Exception as exc:
            failed += 1
            print(f"FAILED {row['id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        changed = [k for k, v in measured.items() if row.get(k) != v]
        row.update(measured)
        status = f"updated ({', '.join(changed)})" if changed else "same"
        print(f"{row['id']}: {status}")

    if failed:
        print(f"{failed} seam(s) could not be measured; file left unchanged",
              file=sys.stderr)
        return 1
    dump_rows(rows, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
