# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the sm80 package of an engine tree lives, and the plugin's identity.

Three layouts:

* assembled engine: ``<dir>/engine`` (the tree: upstream plus the patch
  queue), ``<dir>/plugin`` (the plugin's source tree at its commit, with its
  compiled extensions and entry-point metadata) and ``<dir>/assembly.json``
  (schema engine-assembly/1: the upstream, queue and plugin commits, the
  native build and the sha256 of each extension). vLLM imports both
  directories (``PYTHONPATH=<dir>/engine:<dir>/plugin``);
* a tree that carries the package at its root (``<tree>/vllm_sm80``);
* a legacy tree (``<tree>/vllm/sm80``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from canon import HashCache, canonical_sha256
from errors import Refusal

ASSEMBLY = "assembly.json"
SCHEMA = "engine-assembly/1"
PACKAGE = "vllm_sm80"
PIN_FIELDS = ("upstream_commit", "queue_commit", "plugin_commit")
# Paths relative to the plugin directory, as assembly.json keys them.
NATIVES = ("vllm_sm80/_exl3_C.abi3.so", "vllm_sm80/_ampere_marlin_C.abi3.so")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_BUILD = re.compile(r"^native-[0-9a-f]{10}-t\S+$")
# Untracked files the assembly writes into the plugin's checkout: the
# entry-point metadata, the extensions and their provenance records.
_DIST_INFO = re.compile(r"^vllm_sm80-[^/]+\.dist-info/")
_PLUGIN_WRITTEN = re.compile(r"^(vllm_sm80-[^/]+\.dist-info/.+|vllm_sm80/_[a-z0-9_]+_C\.abi3\.so"
                             r"|\.sm80-build-owned\.json|\.sm80-build\.json)$")


def assembly_dir(tree: str) -> str | None:
    """``<dir>`` when ``tree`` is the ``engine`` directory of an assembled engine."""
    parent = os.path.dirname(os.path.abspath(tree))
    return parent if os.path.isfile(os.path.join(parent, ASSEMBLY)) else None


def plugin_dir(tree: str) -> str | None:
    """The plugin's directory of an assembled engine, else None."""
    parent = assembly_dir(tree)
    return os.path.join(parent, "plugin") if parent else None


def code_path(tree: str) -> str:
    """PYTHONPATH that imports vLLM and the sm80 package of ``tree``."""
    plugin = plugin_dir(tree)
    return f"{tree}:{plugin}" if plugin else tree


# Single-tree paths of the compiled extensions, which live in the plugin's
# package directory in an assembled engine.
_LEGACY_NATIVES = ("vllm/_exl3_C", "vllm/_ampere_marlin_C")


def plugin_pattern(pattern: str) -> str | None:
    """The plugin-relative form of a tree-relative path pattern that names a
    plugin file in an assembled engine (``vllm_sm80/...``, or an extension
    by its single-tree path ``vllm/_exl3_C*``, ``vllm/_ampere_marlin_C*``);
    None for an engine file."""
    if pattern.startswith(PACKAGE + "/"):
        return pattern
    if pattern.startswith(_LEGACY_NATIVES):
        return PACKAGE + "/" + pattern[len("vllm/"):]
    return None



def sm80_package_dir(tree: str) -> str | None:
    """Directory of the sm80 package that serves ``tree``, or None."""
    plugin = plugin_dir(tree)
    candidates = [os.path.join(plugin, PACKAGE)] if plugin else []
    candidates += [os.path.join(tree, PACKAGE), os.path.join(tree, "vllm", "sm80")]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def _git(cwd: str, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise Refusal(f"git {' '.join(args)} in {cwd}: {proc.stderr.strip()}")
    return proc.stdout


def read_assembly(parent: str) -> tuple[dict, list[str]]:
    """assembly.json of ``parent`` and its structural problems."""
    path = os.path.join(parent, ASSEMBLY)
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
    except (OSError, ValueError) as exc:
        return {}, [f"{ASSEMBLY}: {exc}"]
    if not isinstance(data, dict):
        return {}, [f"{ASSEMBLY}: not an object"]
    problems = []
    if data.get("schema") != SCHEMA:
        problems.append(f"{ASSEMBLY}: schema is not {SCHEMA}")
    for name in PIN_FIELDS:
        if not isinstance(data.get(name), str) or not _HEX40.match(data[name]):
            problems.append(f"{ASSEMBLY}: {name} is not a 40-hex commit")
    if not isinstance(data.get("native_build"), str) or not _NATIVE_BUILD.match(data["native_build"]):
        problems.append(f"{ASSEMBLY}: native_build is not native-<p10>-t<torch>")
    natives = data.get("natives")
    if not isinstance(natives, dict) or sorted(natives) != sorted(NATIVES) or not all(
            isinstance(v, str) and _HEX64.match(v) for v in natives.values()):
        problems.append(f"{ASSEMBLY}: natives must map {', '.join(NATIVES)} to sha256")
    return data, problems


def plugin_state(tree: str, cache: HashCache) -> dict | None:
    """Identity of the plugin that serves an assembled ``tree`` (None for any
    other layout): its commit, the files that make its checkout dirty, the
    hash of its entry-point metadata (``plugin_dist_sha256``), the sha256 of
    each extension on disk and the assembly's pins. ``problems`` lists every
    way it fails to match assembly.json; ``dirty`` is set by either."""
    parent = assembly_dir(tree)
    if parent is None:
        return None
    plugin = os.path.join(parent, "plugin")
    data, problems = read_assembly(parent)
    state: dict = {"dir": plugin, "commit": None, "dirty_paths": [], "problems": problems,
                   "assembly": {k: data.get(k) for k in (*PIN_FIELDS, "native_build")},
                   "plugin_dist_sha256": None, "natives": {}}
    if not os.path.exists(os.path.join(plugin, ".git")):
        problems.append(f"{plugin}: not a Git checkout")
        return _finish(state)
    state["commit"] = _git(plugin, "rev-parse", "HEAD").strip()
    if data.get("plugin_commit") and state["commit"] != data["plugin_commit"]:
        problems.append(f"plugin checkout is at {state['commit']}, assembly.json names "
                        f"{data['plugin_commit']}")
    for entry in _git(plugin, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                      "--ignored=matching").split("\0"):
        if not entry:
            continue
        code, rel = entry[:2], entry[3:]
        if code in ("??", "!!") and (_PLUGIN_WRITTEN.match(rel) or "/__pycache__/" in f"/{rel}"):
            continue
        state["dirty_paths"].append(entry)
    dist = {}
    for name in sorted(os.listdir(plugin)):
        info = os.path.join(plugin, name)
        if not (_DIST_INFO.match(name + "/") and os.path.isdir(info)):
            continue
        for root, _, files in os.walk(info):
            for file in files:
                path = os.path.join(root, file)
                dist[os.path.relpath(path, plugin)] = cache.sha256(path)
    if dist:
        state["plugin_dist_sha256"] = canonical_sha256(dist)
    else:
        problems.append("plugin has no vllm_sm80-*.dist-info entry-point metadata")
    want = data.get("natives") if isinstance(data.get("natives"), dict) else {}
    for rel in NATIVES:
        path = os.path.join(plugin, rel)
        state["natives"][rel] = cache.sha256(path) if os.path.isfile(path) else None
        if state["natives"][rel] is None or state["natives"][rel] != want.get(rel):
            problems.append(f"plugin native {rel} does not match assembly.json")
    return _finish(state)


def _finish(state: dict) -> dict:
    state["dirty"] = bool(state["dirty_paths"] or state["problems"])
    return state
