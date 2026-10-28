# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The validated statement: is this launch the one a release gate passed?

Inputs come only from the settings directory (one generation of
``validated/``), the target tree, the checkpoints and the environment;
nothing is read from any release state.

The validated layout file (schema validated-launch/1)::

    {"schema", "model", "layout", "argv_functional", "env_functional",
     "functional_sha256", "engine_commit", "requires", "checkpoints",
     "gate_record", "engine_equivalence"?}

The receipt (schema validation-receipt/1)::

    {"schema", "model",
     "layouts": {L: {"validated_sha256", "functional_sha256", "gate_record",
                     "verdict_revision", "gate_revision", "requires"}},
     "pins": {every pins.env key except ENGINE_IMAGE},
     "engines": [{"commit", "role", "native_manifest_sha256",
                  "build_owned_sha256", "engine_equivalence"?,
                  pin set only: "plugin_commit", "native_build",
                  "plugin_dist_sha256", "natives"}],
     "checkpoints": {name: {"repo", "revision", "manifest_sha256"}},
     "promoted_at", "user_yes", "promoter_commit"}

A model pins either one engine commit (ENGINE_COMMIT: a legacy single
tree) or a pin set (UPSTREAM_COMMIT, QUEUE_COMMIT, PLUGIN_COMMIT,
NATIVE_BUILD: an assembled engine whose tree is the queue commit and whose
plugin is the plugin commit with the natives of that build). The receipt's
pins hold whichever the model has, so the pin set keys the receipt; an
engines entry of a pin set also names the plugin's commit, entry-point
metadata hash and extension hashes, which must equal the launched plugin's.

Missing identity never equals missing identity: a receipt or a tree
without a build or native identity is NOT VALIDATED.

Roles: ``final`` settings must carry a receipt (a missing one is refused,
never read as a candidate); ``candidate`` settings must not; ``any`` reports
what it finds. Every receipt-independent check runs in every role.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from canon import canonical_sha256, sha256_bytes
from errors import Refusal
from settings_dir import GENERATIONS, POINTER, SettingsDir, check_schema

ROLES = ("any", "candidate", "final")
RECEIPT = "receipt.json"
REQUIRED_PINS = ("ENGINE_COMMIT", "TARGET_REPO", "TARGET_REVISION")
PIN_SET = ("UPSTREAM_COMMIT", "QUEUE_COMMIT", "PLUGIN_COMMIT", "NATIVE_BUILD")
PIN_SET_REQUIRED = PIN_SET + ("TARGET_REPO", "TARGET_REVISION")


def has_pin_set(pins: dict[str, str]) -> bool:
    """True when the pins name an assembled engine (any pin-set key)."""
    return any(k in pins for k in PIN_SET)


def required_pins(pins: dict[str, str]) -> tuple[str, ...]:
    return PIN_SET_REQUIRED if has_pin_set(pins) else REQUIRED_PINS


def engine_pin(pins: dict[str, str]) -> str | None:
    """The commit the engine tree must be at: the queue commit of a pin set,
    else the engine commit."""
    return pins.get("QUEUE_COMMIT") if has_pin_set(pins) else pins.get("ENGINE_COMMIT")


_HEX64 = set("0123456789abcdef")


@dataclass
class Generation:
    """Every file of one ``validated`` generation, read under one lock."""
    directory: str | None
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def receipt(self) -> bytes | None:
        return self.files.get(RECEIPT)

    def layout(self, name: str) -> bytes | None:
        return self.files.get(f"{name}.json")


def read_generation(sd: SettingsDir, between: Callable[[], None] | None = None) -> Generation:
    """Read every file of the current generation.

    In the lab ``validated`` is a symlink to ``validated.d/<gen>/``: the
    reader holds a shared flock on ``validated.d/.lock`` from resolving the
    pointer to its last read and closes it before returning, so the server
    never inherits it. ``between`` runs after the first file (tests)."""
    pointer = sd.path(POINTER)
    gens = sd.path(GENERATIONS)
    fd = None
    if os.path.isdir(gens):
        lock = os.path.join(gens, ".lock")
        try:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        except OSError:
            try:
                fd = os.open(lock, os.O_RDONLY | os.O_CLOEXEC)
            except OSError as exc:
                raise Refusal(f"cannot open {lock} for the reader lock: {exc.strerror}") from None
        fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        if not os.path.isdir(pointer):
            return Generation(None)
        directory = os.path.realpath(pointer)
        files: dict[str, bytes] = {}
        for i, name in enumerate(sorted(os.listdir(directory))):
            full = os.path.join(directory, name)
            if name.startswith(".") or not os.path.isfile(full):
                continue
            with open(full, "rb") as f:
                files[name] = f.read()
            if i == 0 and between is not None:
                between()
        return Generation(directory, files)
    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _json(raw: bytes, schema: str, where: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise Refusal(f"{where}: {exc}") from None
    if not isinstance(data, dict):
        raise Refusal(f"{where}: not an object")
    check_schema(data, schema, where)
    return data


def _sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX64


def receipt_problems(r: dict, pins: dict[str, str]) -> list[str]:
    """Structural problems of a receipt; any one makes the launch NOT VALIDATED."""
    out = []
    if not isinstance(r.get("layouts"), dict) or not r["layouts"]:
        out.append("receipt has no layouts")
    rp = r.get("pins")
    if not isinstance(rp, dict):
        out.append("receipt has no pins")
    else:
        for k in set(required_pins(pins)) | (set(pins) - {"ENGINE_IMAGE"}):
            if not isinstance(rp.get(k), str) or not rp[k]:
                out.append(f"receipt pins lack {k}")
    engines = r.get("engines")
    if not isinstance(engines, list) or not engines:
        out.append("receipt has no engines")
    else:
        for e in engines:
            if not isinstance(e, dict) or not isinstance(e.get("commit"), str) \
                    or e.get("role") not in ("validated", "carrier") \
                    or not _sha(e.get("build_owned_sha256")) or not _sha(e.get("native_manifest_sha256")):
                out.append("receipt engine entry lacks commit, role or build identity")
                break
            if has_pin_set(pins) and e.get("role") == "validated" and (
                    not isinstance(e.get("plugin_commit"), str)
                    or not isinstance(e.get("native_build"), str)
                    or not _sha(e.get("plugin_dist_sha256"))
                    or not isinstance(e.get("natives"), dict) or not e["natives"]
                    or not all(_sha(v) for v in e["natives"].values())):
                out.append("receipt engine entry lacks plugin identity")
                break
    cps = r.get("checkpoints")
    if not isinstance(cps, dict) or any(
            not isinstance(c, dict) or not _sha(c.get("manifest_sha256")) for c in cps.values()):
        out.append("receipt checkpoints malformed")
    return ["receipt malformed: " + p for p in out]


@dataclass
class Evidence:
    layout: str
    functional_sha256: str
    pins: dict[str, str]
    tree_commit: str | None
    tree_dirty: bool
    build_owned_sha256: str | None
    native_manifest_sha256: str | None
    cross_commit_artifacts: bool
    adopted: bool
    requires: dict[str, str]
    checkpoints: dict[str, dict]            # name -> {repo, revision, manifest_sha256, state, identity}
    not_validated: list[str] = field(default_factory=list)
    plugin: dict | None = None              # assembled engine: treestate plugin state


def evaluate(gen: Generation, ev: Evidence, *, role: str = "any") -> dict:
    """Criteria 1-5; every failing reason is listed."""
    if role not in ROLES:
        raise Refusal(f"unknown role {role!r}")
    reasons: list[str] = []
    out: dict = {"state": "NOT_VALIDATED", "reasons": reasons, "functional_match": False,
                 "role": role, "generation": gen.directory and os.path.basename(gen.directory)}
    if gen.receipt is None:
        if role == "final":
            raise Refusal("final settings carry no receipt (validated/receipt.json); "
                          "a final launch is never treated as a candidate")
        reasons.append("no receipt")
        out["receipt"] = None
        _independent(ev, reasons, None)
        out["reasons"] = list(dict.fromkeys(reasons))
        return out
    if role == "candidate":
        raise Refusal("candidate settings must not carry a receipt (validated/ must be empty)")
    receipt = _json(gen.receipt, "validation-receipt", "validated/receipt.json")
    out["receipt_sha256"] = sha256_bytes(gen.receipt)
    reasons += receipt_problems(receipt, ev.pins)
    layouts = receipt.get("layouts") if isinstance(receipt.get("layouts"), dict) else {}
    entry = layouts.get(ev.layout)
    raw = gen.layout(ev.layout)
    # 1. validated file ↔ receipt ↔ this launch
    if raw is None or not isinstance(entry, dict):
        reasons.append(f"layout {ev.layout} not validated")
    else:
        vfile = _json(raw, "validated-launch", f"validated/{ev.layout}.json")
        own = canonical_sha256({"argv": vfile.get("argv_functional"), "env": vfile.get("env_functional")})
        c1 = True
        if sha256_bytes(raw) != entry.get("validated_sha256"):
            reasons.append("validated file ≠ receipt")
            c1 = False
        if not _sha(vfile.get("functional_sha256")) \
                or vfile.get("functional_sha256") != entry.get("functional_sha256") \
                or own != vfile.get("functional_sha256"):
            reasons.append("receipt ≠ validated file")
            c1 = False
        if vfile.get("functional_sha256") != ev.functional_sha256:
            reasons.append("functional ≠ validated")
            c1 = False
        out["functional_match"] = c1
        out["gate_record"] = vfile.get("gate_record") or entry.get("gate_record")
        out["engine_commit"] = vfile.get("engine_commit")
        if vfile.get("requires") != ev.requires:
            reasons.append("requires ≠ validated")
    # 2. engine and pins bound to the receipt
    rpins = receipt.get("pins") if isinstance(receipt.get("pins"), dict) else {}
    if any(ev.pins.get(k) != v for k, v in rpins.items()) or \
            any(k not in rpins for k in ev.pins if k != "ENGINE_IMAGE"):
        reasons.append("pin ≠ receipt")
    engines = {e.get("commit"): e for e in receipt.get("engines") or [] if isinstance(e, dict)}
    pin = engine_pin(ev.pins)
    if pin not in engines or ev.tree_commit not in engines:
        reasons.append("engine not in receipt")
    # 3. build identity equal to the engines entry (missing never equals missing)
    eng = engines.get(ev.tree_commit)
    if eng is not None:
        if ev.build_owned_sha256 is None or eng.get("build_owned_sha256") != ev.build_owned_sha256:
            reasons.append("cross-commit artefacts" if ev.cross_commit_artifacts else "build ≠ receipt")
        if ev.native_manifest_sha256 is None or eng.get("native_manifest_sha256") != ev.native_manifest_sha256:
            reasons.append("natives ≠ receipt")
        if has_pin_set(ev.pins):
            plugin = ev.plugin or {}
            if plugin.get("commit") is None or eng.get("plugin_commit") != plugin.get("commit") \
                    or eng.get("native_build") != ev.pins.get("NATIVE_BUILD") \
                    or plugin.get("plugin_dist_sha256") is None \
                    or eng.get("plugin_dist_sha256") != plugin.get("plugin_dist_sha256") \
                    or eng.get("natives") != plugin.get("natives"):
                reasons.append("plugin ≠ receipt")
    # 4a. checkpoints bound to the receipt
    rcps = receipt.get("checkpoints") if isinstance(receipt.get("checkpoints"), dict) else {}
    for name, cp in ev.checkpoints.items():
        r = rcps.get(name)
        if not isinstance(r, dict) or r.get("manifest_sha256") != cp.get("manifest_sha256") \
                or r.get("repo") != cp.get("repo") or r.get("revision") != cp.get("revision"):
            reasons.append("checkpoint ≠ receipt")
            break
    if set(rcps) - set(ev.checkpoints):
        reasons.append("checkpoint ≠ receipt")
    _independent(ev, reasons, eng)
    out["reasons"] = list(dict.fromkeys(reasons))
    if not out["reasons"]:
        out["state"] = "VALIDATED"
    return out


def _independent(ev: Evidence, reasons: list[str], eng: dict | None) -> None:
    """Checks that need no receipt: they run for candidates too."""
    pin = engine_pin(ev.pins)
    if not pin:
        reasons.append("no engine pin")
    elif ev.tree_commit != pin:
        reasons.append("tree ≠ pin")
    if has_pin_set(ev.pins):
        if ev.plugin is None:
            reasons.append("no assembled plugin")
        else:
            assembly = ev.plugin.get("assembly") or {}
            if ev.plugin.get("commit") != ev.pins.get("PLUGIN_COMMIT"):
                reasons.append("plugin ≠ pin")
            if assembly.get("upstream_commit") != ev.pins.get("UPSTREAM_COMMIT") \
                    or assembly.get("queue_commit") != ev.pins.get("QUEUE_COMMIT") \
                    or assembly.get("plugin_commit") != ev.pins.get("PLUGIN_COMMIT") \
                    or assembly.get("native_build") != ev.pins.get("NATIVE_BUILD"):
                reasons.append("assembly ≠ pins")
    if ev.tree_dirty:
        reasons.append("dirty tree")
    if ev.adopted:
        reasons.append("adopted tree")
    if ev.cross_commit_artifacts and eng is None:
        reasons.append("cross-commit artefacts")
    for cp in ev.checkpoints.values():
        if cp.get("state") in ("not present", "incomplete"):
            reasons.append("checkpoint not present")
        elif cp.get("identity") != "verified":
            reasons.append("checkpoint not verified")
    reasons.extend(ev.not_validated)


def compare_expected(expected_raw: bytes, argv_functional: list[str],
                     env_functional: dict[str, str], where: str) -> dict:
    """Compare this launch with a frozen expected launch (a resolved launch
    or a validated layout file): candidate consistency, apart from VALIDATED."""
    try:
        exp = json.loads(expected_raw)
    except ValueError as exc:
        raise Refusal(f"{where}: {exc}") from None
    schema = exp.get("schema", "") if isinstance(exp, dict) else ""
    if schema not in ("launch-resolved/2", "validated-launch/1"):
        raise Refusal(f"{where}: expected a launch-resolved/2 or validated-launch/1 file")
    e_argv = exp.get("argv_functional")
    e_env = exp.get("env_functional")
    if not isinstance(e_argv, list) or not isinstance(e_env, dict):
        raise Refusal(f"{where}: no argv_functional/env_functional")
    sha = canonical_sha256({"argv": e_argv, "env": e_env})
    mine = canonical_sha256({"argv": argv_functional, "env": env_functional})
    env_diff = sorted(k for k in set(e_env) | set(env_functional)
                      if e_env.get(k) != env_functional.get(k))
    return {
        "source": where,
        "functional_sha256": sha,
        "functional_match": sha == mine,
        "argv_equal": e_argv == argv_functional,
        "env_differs": env_diff,
    }
