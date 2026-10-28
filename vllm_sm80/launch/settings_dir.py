# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read a model settings directory safely.

Layout::

    settings/
      pins.env                 KEY=VALUE lines (ENGINE_COMMIT, TARGET_REPO, ...)
      model.toml               schema launch-model/1
      hardware/<name>.toml     schema launch-hardware/1
      layouts/<name>.toml      schema launch-layout/1
      variants/<name>.toml     schema launch-variant/1
      checkpoints/<name>.manifest.json   schema checkpoint-manifest/1
      validated -> validated.d/<generation>/   (or a plain directory)
        <layout>.json          schema validated-launch/1
        receipt.json           schema validation-receipt/1

Every file carries ``schema = "<name>/<major>"``; an unknown name or major
is refused. Non-finite floats, a newline or NUL inside a string, one key
spelled two ways, and any symlink other than the ``validated`` pointer (and
the publisher's transient ``validated.new``) are refused.
"""

from __future__ import annotations

import os
import re
import tomllib
from typing import Any

from canon import sha256_bytes
from errors import Refusal

SUPPORTED_SCHEMAS = {
    "launch-defaults": {1},
    "launch-model": {1},
    "launch-hardware": {1},
    "launch-layout": {1},
    "launch-variant": {1},
    "launch-overlay": {1},
    "checkpoint-manifest": {1},
    "validated-launch": {1},
    "validation-receipt": {1},
    "launch-resolved": {2},
    "tree-exempt": {1},
    "build-inputs": {4},
    "link-written": {1},
    "native-inputs": {1},
}

POINTER = "validated"
STAGING = "validated.new"          # a publisher's pointer before its rename
GENERATIONS = "validated.d"

_PIN_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def check_schema(data: dict, expected: str, where: str) -> int:
    value = data.get("schema")
    if not isinstance(value, str) or "/" not in value:
        raise Refusal(f"{where}: missing schema (expected {expected}/<major>)")
    name, _, major = value.partition("/")
    if name != expected:
        raise Refusal(f"{where}: schema {value!r} is not {expected}/<major>")
    if not major.isdigit() or int(major) not in SUPPORTED_SCHEMAS.get(name, ()):
        raise Refusal(f"{where}: unsupported schema major {value!r}")
    return int(major)


def _refuse_float(text: str) -> float:
    if text.lstrip("+-") in ("nan", "inf"):
        raise ValueError(f"non-finite float {text!r}")
    return float(text)


def check_strings(obj: Any, where: str, path: str = "") -> None:
    """Refuse newline or NUL in any string, key or value."""
    if isinstance(obj, str):
        if "\n" in obj or "\r" in obj or "\x00" in obj:
            raise Refusal(f"{where}: newline or NUL in value at {path or '<root>'}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_strings(k, where, path)
            check_strings(v, where, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_strings(v, where, f"{path}[{i}]")


def normalize_arg_name(name: str) -> str:
    return name.replace("_", "-")


def normalize_key(key: str) -> str:
    """args.max_model_len → args.max-model-len; other keys unchanged."""
    if key.startswith("args."):
        head, _, rest = key[5:].partition(".")
        return "args." + normalize_arg_name(head) + (f".{rest}" if rest else "")
    return key


def check_arg_spellings(table: dict, where: str) -> None:
    """Refuse one argument given under two spellings (a_b and a-b)."""
    seen: dict[str, str] = {}
    for key in table:
        norm = normalize_arg_name(key)
        if norm in seen and seen[norm] != key:
            raise Refusal(f"{where}: {seen[norm]!r} and {key!r} name the same argument")
        seen[norm] = key


def loads_toml(data: bytes, where: str) -> dict:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal(f"{where}: not UTF-8 ({exc})") from None
    try:
        parsed = tomllib.loads(text, parse_float=_refuse_float)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise Refusal(f"{where}: {exc}") from None
    check_strings(parsed, where)
    return parsed


def load_toml(path: str, expected_schema: str, where: str | None = None) -> dict:
    where = where or path
    with open(path, "rb") as f:
        data = loads_toml(f.read(), where)
    check_schema(data, expected_schema, where)
    return data


def parse_env_lines(text: str, where: str, *, allow_quotes: bool = False) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments and blank lines skipped; no expansion."""
    out: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _PIN_KEY.match(key):
            raise Refusal(f"{where}:{n}: not a KEY=VALUE line")
        if allow_quotes and len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if "\x00" in value:
            raise Refusal(f"{where}:{n}: NUL in value")
        if key in out:
            raise Refusal(f"{where}:{n}: {key} given twice")
        out[key] = value
    return out


class SettingsDir:
    """A settings directory with its symlink policy checked once."""

    def __init__(self, root: str, *, community: bool = False) -> None:
        self.root = os.path.abspath(root)
        self.community = community
        if not os.path.isdir(self.root):
            raise Refusal(f"settings directory {root} does not exist")
        self._check_symlinks()

    # -- structure ---------------------------------------------------------
    def _check_symlinks(self) -> None:
        root_real = os.path.realpath(self.root)
        gens = os.path.join(root_real, GENERATIONS) + os.sep
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            for name in list(dirnames) + filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.normpath(os.path.join(rel_dir, name))
                if os.path.islink(full):
                    if rel in (POINTER, STAGING) and not self.community:
                        target = os.path.realpath(full)
                        if not target.startswith(gens) or (rel == POINTER and not os.path.isdir(target)):
                            raise Refusal(
                                f"settings: {rel!r} must point to a generation "
                                f"directory under {GENERATIONS}/, not {os.readlink(full)!r}")
                        continue
                    raise Refusal(f"settings: symlink {rel} is not allowed")
                real = os.path.realpath(full)
                if not (real == root_real or real.startswith(root_real + os.sep)):
                    raise Refusal(f"settings: {rel} leaves the settings directory")
            # never descend into a symlinked directory
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def exists(self, *parts: str) -> bool:
        return os.path.exists(self.path(*parts))

    def toml(self, rel: str, schema: str) -> dict:
        return load_toml(self.path(rel), schema, where=f"settings:{rel}")

    def named(self, kind: str, name: str) -> str:
        if not _NAME.match(name):
            raise Refusal(f"{kind} name {name!r} is not a plain name")
        return os.path.join(kind, f"{name}.toml")

    def pins(self) -> dict[str, str]:
        p = self.path("pins.env")
        if not os.path.isfile(p):
            return {}
        with open(p, encoding="utf-8") as f:
            return parse_env_lines(f.read(), "settings:pins.env")

    # -- hashing -----------------------------------------------------------
    def file_hashes(self, validated: dict[str, bytes]) -> dict[str, str]:
        """sha256 of every settings file. ``validated/`` contributes the
        given generation snapshot (name → bytes), so one launch hashes the
        same generation it read; ``validated.d/`` and ``validated.new`` are
        never part of the settings."""
        out: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            rel_dir = os.path.relpath(dirpath, self.root)
            if rel_dir == ".":
                dirnames[:] = [d for d in dirnames if d not in (GENERATIONS, STAGING, POINTER)]
                filenames = [f for f in filenames if f not in (STAGING, POINTER)]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.normpath(os.path.join(rel_dir, name))
                with open(full, "rb") as f:
                    out[rel] = sha256_bytes(f.read())
        for name, raw in validated.items():
            out[f"{POINTER}/{name}"] = sha256_bytes(raw)
        return dict(sorted(out.items()))


def tree_sha256(file_hashes: dict[str, str]) -> str:
    """sha256 of the sorted ``path\\0sha256\\n`` list."""
    lines = "".join(f"{p}\0{h}\n" for p, h in sorted(file_hashes.items()))
    return sha256_bytes(lines.encode("utf-8"))
