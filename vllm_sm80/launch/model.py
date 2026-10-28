# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""model.toml: what applies to every layout of one model.

Fields (schema launch-model/1)::

    model = "<model id>"
    hardware = "<hardware file name>"
    env_prefixes = ["MYMODEL_"]        # model-owned functional env prefixes
    arg_order = ["served-model-name", ...]   # argv order; other args follow
    baseline = { release = "...", gate_record = "...", bundle = "...", engine_commit = "..." }
    fixed = ["args.max-model-len", "tp", "pp", "cards"]
    controlled_args = ["max-num-seqs", ...]
    [layouts] allowed = [...], default = "..."
    [checkpoint.<name>] repo_key, revision_key, dir, container_dir?, index?,
                        weights?, required, manifest, kind? (model or sidecar)
    [args] / [env]                     # values: plain or { value = ..., why = "..." }
    [location]                         # plain, or { native = "...", container = "..." }
    [[refuse]] when = { "env.X" = "0", ... }, why = "..."
    [requires] files = [ { path = "vllm/_x*.so", fork_only = true } ]
    [user.<KNOB>]
        target = "args.a[.b]" | "env.X" | "card_ids" | "extra_args" | [several]
        type = "str" | "int" | "bool"      divide_by = N      choices = [...]
        secret = true                       variants = ["v"]  # only with one of these
        variant = "v"                       # 1/0 selects variant v
        variant = { a = "va", b = "" }      # the value picks a variant ("" none)

Model and drafter paths, Engram table directories and override sidecars must
use checkpoint references (``${checkpoint.<name>.path}``). A sidecar declares
``kind = "sidecar"`` and its ``required`` and ``weights`` files explicitly;
unlike a model it has no implicit config.json or model.safetensors. File
references append a manifest-listed relative path, for example
``${checkpoint.override.path}/override.json``. Paths move only through
``--checkpoint-path NAME=DIR``. Geometry is ``tp``/``pp`` only; requires paths
stay inside the tree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from errors import Refusal
from settings_dir import SettingsDir, normalize_key

_KNOB = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ENV_TARGET = re.compile(r"^env\.[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class ModelSettings:
    data: dict
    model: str
    hardware: str
    env_prefixes: list[str]
    arg_order: list[str]
    allowed_layouts: list[str]
    default_layout: str | None
    checkpoints: dict[str, dict]
    fixed: set[str]
    controlled_args: set[str]
    refuse: list[dict]
    requires: list[dict]
    user: dict[str, dict]
    baseline: dict = field(default_factory=dict)


def _check_knob(knob: str, spec) -> None:
    where = f"settings:model.toml: user.{knob}"
    if not _KNOB.match(knob):
        raise Refusal(f"settings:model.toml: user knob {knob!r} is not an env-style name")
    if not isinstance(spec, dict) or not (spec.get("target") or spec.get("variant")):
        raise Refusal(f"{where} needs target or variant")
    target = spec.get("target")
    if target is not None and not (isinstance(target, str) or (
            isinstance(target, list) and target and all(isinstance(t, str) for t in target))):
        raise Refusal(f"{where}: target is a key or a list of keys")
    variant = spec.get("variant")
    if variant is not None and not (isinstance(variant, str) or (
            isinstance(variant, dict) and all(isinstance(v, str) for v in variant.values()))):
        raise Refusal(f"{where}: variant is a name or a value-to-variant table")
    if spec.get("type", "str") not in ("str", "int", "bool"):
        raise Refusal(f"{where}: type is str, int or bool")
    secret = spec.get("secret", False)
    if not isinstance(secret, bool):
        raise Refusal(f"{where}: secret is true or false")
    if secret:
        # a secret value is masked and kept out of the identity only as an
        # environment value; anywhere else it would be printed and hashed
        targets = [target] if isinstance(target, str) else list(target or [])
        bad = [t for t in targets if not _ENV_TARGET.match(t)]
        if variant is not None or not targets or bad:
            raise Refusal(f"{where}: a secret knob may only target environment names (env.NAME)"
                          + (f"; not {', '.join(bad)}" if bad else ""))


def _inside(path: str) -> bool:
    return not os.path.isabs(path) and ".." not in path.split("/")


def load_model_settings(sd: SettingsDir) -> ModelSettings:
    d = sd.toml("model.toml", "launch-model")
    for key in ("model", "hardware"):
        if not isinstance(d.get(key), str) or not d[key]:
            raise Refusal(f"settings:model.toml: {key} is required")
    layouts = d.get("layouts", {})
    allowed = list(layouts.get("allowed", []))
    if not allowed:
        raise Refusal("settings:model.toml: [layouts] allowed is empty")
    default = layouts.get("default")
    if default is not None and default not in allowed:
        raise Refusal(f"settings:model.toml: default layout {default!r} is not allowed")
    cps = d.get("checkpoint", {})
    for name, spec in cps.items():
        if spec.get("kind", "model") not in ("model", "sidecar"):
            raise Refusal(f"settings:model.toml: checkpoint.{name}.kind must be model or sidecar")
        for key in ("dir", "manifest", "repo_key", "revision_key"):
            if not isinstance(spec.get(key), str):
                raise Refusal(f"settings:model.toml: checkpoint.{name}.{key} is required")
        if not _inside(spec["manifest"]):
            raise Refusal(f"settings:model.toml: checkpoint.{name}.manifest leaves the settings")
    user = d.get("user", {})
    for knob, spec in user.items():
        _check_knob(knob, spec)
    requires = []
    for ent in d.get("requires", {}).get("files", []):
        if isinstance(ent, str):
            ent = {"path": ent}
        if not isinstance(ent, dict) or not isinstance(ent.get("path"), str):
            raise Refusal("settings:model.toml: requires.files entries need a path")
        if not _inside(ent["path"]):
            raise Refusal(f"settings:model.toml: requires path {ent['path']!r} leaves the engine tree")
        requires.append(ent)
    return ModelSettings(
        data=d,
        model=d["model"],
        hardware=d["hardware"],
        env_prefixes=list(d.get("env_prefixes", [])),
        arg_order=[a.replace("_", "-") for a in d.get("arg_order", [])],
        allowed_layouts=allowed,
        default_layout=default,
        checkpoints=cps,
        fixed={normalize_key(k) for k in d.get("fixed", [])},
        controlled_args={a.lstrip("-").replace("_", "-") for a in d.get("controlled_args", [])},
        refuse=list(d.get("refuse", [])),
        requires=requires,
        user=user,
        baseline=dict(d.get("baseline", {})),
    )
