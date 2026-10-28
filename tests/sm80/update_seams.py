# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure and rewrite the upstream seam pins in seams.yaml.

A seam is an upstream object the sm80 fork code relies on without editing
it. seams.yaml records a fingerprint of each one; test_seams.py fails when
an upstream change moves a fingerprint. After checking that change against
the row's `why`, run this script to record the new fingerprints:

    python tests/sm80/update_seams.py            # rewrite every row
    python tests/sm80/update_seams.py ID [ID..]  # rewrite only these rows

Run it with the repository root importable (for example PYTHONPATH=.).
"""

import argparse
import hashlib
import importlib
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

SEAMS_PATH = Path(__file__).with_name("seams.yaml")

CODE_KINDS = ("function", "method")
ATTR_KINDS = ("module_attr", "class_attr")
KINDS = CODE_KINDS + ATTR_KINDS + ("dict_contract",)

FIELD_ORDER = ("id", "kind", "target", "why", "sha256", "signature", "type", "value")

HEADER = """\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Upstream seams the sm80 fork code relies on without editing them.
# Checked by tests/sm80/test_seams.py. After checking an upstream change to
# a seam, rewrite the pins with: python tests/sm80/update_seams.py [ID ...]
#
# Fields:
#   id         unique row name
#   kind       function | method | class_attr | module_attr | dict_contract
#   target     'module:qualname'
#   why        what the fork code depends on
#   sha256     sha256 of textwrap.dedent(inspect.getsource(obj)) (code kinds)
#   signature  str(inspect.signature(obj)) (code kinds)
#   type       type name of the value (attr kinds and dict_contract)
#   value      exact value, for constants only (attr kinds, optional)
#   dict_contract rows also require every value to be a zero-arg callable.
"""


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
            "signature": str(inspect.signature(obj)),
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
    parser.add_argument("--file", type=Path, default=SEAMS_PATH)
    args = parser.parse_args(argv)

    rows = load_rows(args.file)
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
    dump_rows(rows, args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
