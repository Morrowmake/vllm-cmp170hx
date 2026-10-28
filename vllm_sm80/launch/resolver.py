# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve the settings layers into one final object.

Precedence, lowest first::

    defaults.toml[runtime] < model.toml < hardware < layout
      < variants (in the order given) < overlays (incl. the community
      settings directory) < user .env ([user] keys only)
      < ambient (merge policy only) < CLI --set

Every key of the final object records its value, layer, source file and
``why``. argv, the child environment, the printed configuration and
``functional_sha256`` are all derived from that object and nothing else.

Guards on the object: geometry has one representation (``tp``/``pp``;
the parallel-size arguments are derived from it); ``args.model`` is always
a checkpoint reference, as are drafter, Engram and override-sidecar paths;
the reference (not its directory) is part of the functional argv; [fixed] keys and controlled arguments change only
under --experiment, whatever the channel (overlay, .env, ambient, --set,
extra arguments, unset) and whatever the spelling: arguments are compared
by the engine argument they set (an alias, a negative form such as --no-X,
a dotted option or an abbreviation of a protected one sets that one); only the
launcher's own instrumentation list and the host/port location are
excluded from the functional hash, and their values must be plain scalars;
no argument value may carry an option token; every child environment name
is classified by the closed class table whichever table supplied it;
secret targets are fixed by the model's declarations before any input is
read.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from typing import Any

from assembly import code_path, plugin_dir
from canon import canonical_sha256
from envclass import ClassTable, always_refused
from errors import Refusal
from model import ModelSettings
from registry import ArgOptions, Registry
from settings_dir import check_arg_spellings, normalize_arg_name, normalize_key

LOCKED_LAYERS = ("overlay", "user", "ambient", "cli")   # [fixed] needs --experiment

# Functional-class names the container image's ENV may carry. They reach the
# server outside the functional hash, so only names that change no output
# are listed; any other functional name in the image ENV is refused.
IMAGE_FUNCTIONAL = {
    "CUDA_VERSION": "base-image package metadata; the engine and its libraries do not read it",
    "NCCL_VERSION": "base-image package metadata; the engine and its libraries do not read it",
    "PYTHONDONTWRITEBYTECODE": "the interpreter writes no .pyc files; nothing it runs changes",
}

RUNTIMES = ("native", "container")
GEOMETRY_ARGS = ("args.tensor-parallel-size", "args.pipeline-parallel-size")
GEOMETRY_ARGS_NAMES = tuple(k[5:] for k in GEOMETRY_ARGS)
# what only the launcher sets: geometry (tp/pp), the served checkpoint and
# no argument file (--config would load arguments past every check)
LAUNCHER_ARGS = frozenset(GEOMETRY_ARGS_NAMES + ("model", "config"))
CHECKPOINT_ARGS = {"model": (), "spec-model": None, "speculative-config": ("model",),
                   "hf-overrides": ("engram_table_dir",)}
LOCATION_ARGS = ("host", "port")
_SUBST = re.compile(r"\$\{([^}]*)\}")
_CHECKPOINT_REF = re.compile(r"^\$\{checkpoint\.([A-Za-z0-9_-]+)\.path\}$")
_CHECKPOINT_USE = re.compile(r"\$\{checkpoint\.([A-Za-z0-9_-]+)\.path\}")
_CHECKPOINT_FILE = re.compile(r"\$\{checkpoint\.([A-Za-z0-9_-]+)\.path\}/(.+)")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHORT_OPTION = re.compile(r"^-[A-Za-z]")
_TOP_KEYS = ("tp", "pp", "cards", "card_ids")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")


@dataclass
class Entry:
    value: Any
    layer: str
    source: str
    why: str | None = None


@dataclass
class Final:
    entries: dict[str, Entry] = field(default_factory=dict)
    unset: dict[str, str] = field(default_factory=dict)
    whys: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        e = self.entries.get(key)
        return default if e is None else e.value

    def put(self, key: str, value: Any, layer: str, source: str, why: str | None = None) -> None:
        old = self.entries.pop(key, None)
        if why is None and old is not None and old.value == value:
            why = old.why
        self.entries[key] = Entry(value, layer, source, why)
        self.unset.pop(key, None)

    def drop(self, key: str, layer: str) -> None:
        self.entries.pop(key, None)
        self.unset[key] = layer


def split_value(v: Any, where: str, whys: dict | None = None, path: str = "") -> tuple[Any, str | None]:
    """A plain value or { value = ..., why = "..." }; nested tables keep
    their own value/why forms, whose whys are collected into ``whys``."""
    if isinstance(v, dict) and "value" in v and set(v) <= {"value", "why"}:
        why = v.get("why")
        if why is not None and not isinstance(why, str):
            raise Refusal(f"{where}: why must be a string")
        inner, _ = split_value(v["value"], where, whys, path)
        return inner, why
    if isinstance(v, dict):
        out = {}
        for k, sub in v.items():
            sub_path = f"{path}.{k}" if path else k
            val, why = split_value(sub, where, whys, sub_path)
            if why and whys is not None:
                whys[sub_path] = why
            out[k] = val
        return out, None
    if isinstance(v, list):
        return [split_value(x, where, whys, path)[0] for x in v], None
    return v, None


def _env_value(v: Any, where: str) -> str:
    if isinstance(v, bool) or not isinstance(v, (str, int)):
        raise Refusal(f"{where}: environment values are strings or integers")
    return str(v)


def _set_and_unset(table: dict) -> tuple[list[str], list[str]]:
    """The keys one table sets and the keys it unsets (its own level only)."""
    sets = [key for key in _TOP_KEYS if key in table]
    sets += ["args." + normalize_arg_name(k) for k in table.get("args", {})]
    for section, prefix in (("env", "env."), ("location", "loc.")):
        sets += [prefix + k for k in table.get(section, {})]
    unsets = [normalize_key(k) for k in table.get("unset", []) if isinstance(k, str)]
    return sets, unsets


def _check_set_and_unset(table: dict, source: str) -> None:
    """A file that both sets and unsets a key (or a parent or child key of
    it) contradicts itself, wherever in the file the two are: its top level
    and every [layout.<L>] sub-table count together."""
    sets, unsets = [], []
    parts = [("", table)] + [(f"[layout.{name}]", sub) for name, sub in table.get("layout", {}).items()
                             if isinstance(sub, dict)]
    for where, part in parts:
        s, u = _set_and_unset(part)
        sets += [(key, where) for key in s]
        unsets += [(key, where) for key in u]
    for key, where in unsets:
        both = [f"{s}{w}" for s, w in sets if s == key or s.startswith(key + ".") or key.startswith(s + ".")]
        if both:
            raise Refusal(f"{source}: {key} is unset{where or ' at the top level'} and set "
                          f"({', '.join(both)}) in the same file; keep one")


def apply_table(final: Final, table: dict, layer: str, source: str, *,
                layout: str | None = None, runtime: str | None = None,
                whole_file: bool = True) -> None:
    """Apply one settings table (layout, variant, overlay, hardware, model).

    A [location] value may be ``{ native = "...", container = "..." }``:
    only the current runtime's entry applies (none if it is missing). A key
    the file both sets and unsets, in any of its tables, is refused; one
    file overriding another is the layering's job."""
    if whole_file:
        _check_set_and_unset(table, source)
    for key in _TOP_KEYS:
        if key in table:
            v = table[key]
            if key == "card_ids":
                v = str(v)
            elif isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise Refusal(f"{source}: {key} must be a positive integer")
            final.put(key, v, layer, source)
    args = table.get("args", {})
    check_arg_spellings(args, source)
    for k, v in args.items():
        key = "args." + normalize_arg_name(k)
        if key in GEOMETRY_ARGS:
            raise Refusal(f"{source}: set tp/pp, not --{key[5:]}")
        sub_whys: dict[str, str] = {}
        val, why = split_value(v, source, sub_whys)
        final.put(key, val, layer, source, why)
        for p, w in sub_whys.items():
            final.whys[f"{key}.{p}"] = w
    for section, prefix in (("env", "env."), ("location", "loc.")):
        for k, v in table.get(section, {}).items():
            if not _ENV_NAME.match(k):
                raise Refusal(f"{source}: {k!r} is not an environment name")
            val, why = split_value(v, source)
            if section == "location" and isinstance(val, dict):
                if not set(val) <= set(RUNTIMES):
                    raise Refusal(f"{source}: location {k} keys must be native/container")
                if runtime not in val:
                    continue
                val = val[runtime]
            final.put(prefix + k, _env_value(val, f"{source}:{k}"), layer, source, why)
    for key in table.get("unset", []):
        if not isinstance(key, str):
            raise Refusal(f"{source}: unset entries are key strings")
        key = normalize_key(key)
        if key in GEOMETRY_ARGS or key in ("tp", "pp", "cards"):
            raise Refusal(f"{source}: geometry cannot be unset")
        final.drop(key, layer)
    if layout is not None:
        sub = table.get("layout", {}).get(layout)
        if sub:
            apply_table(final, sub, layer, f"{source}[layout.{layout}]", runtime=runtime,
                        whole_file=False)


def substitute(value: Any, subst: dict[str, str], where: str) -> Any:
    if isinstance(value, str):
        def rep(m: re.Match) -> str:
            name = m.group(1)
            if name not in subst:
                raise Refusal(f"{where}: unknown substitution ${{{name}}}")
            return subst[name]
        return _SUBST.sub(rep, value)
    if isinstance(value, dict):
        return {k: substitute(v, subst, where) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, subst, where) for v in value]
    return value


def checkpoint_uses(value: Any) -> set[str]:
    """Names of the checkpoints a value refers to (``${checkpoint.X.path}``)."""
    if isinstance(value, str):
        return set(_CHECKPOINT_USE.findall(value))
    if isinstance(value, dict):
        return set().union(*(checkpoint_uses(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(checkpoint_uses(v) for v in value))
    return set()


def render_arg(name: str, value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        return [f"--{name}"]
    if isinstance(value, dict):
        return [f"--{name}", json.dumps(value, separators=(",", ":"), ensure_ascii=False)]
    if isinstance(value, list):
        items = [str(v) for v in value]
        if any(i.startswith("-") and not _number(i) for i in items):
            raise Refusal(f"--{name}: a list value may not carry an option token ({items})")
        return [f"--{name}", *items]
    if isinstance(value, float):
        return [f"--{name}", repr(value)]
    text = str(value)
    if text.startswith("-") and not _number(text):
        raise Refusal(f"--{name}: a value may not start with '-' ({text!r})")
    return [f"--{name}", text]


def _number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_set(item: str) -> tuple[str, Any]:
    key, sep, raw = item.partition("=")
    if not sep or not key:
        raise Refusal(f"--set {item!r}: expected KEY=VALUE")
    if key.startswith("args."):
        key = normalize_key(key)
    if not key.startswith("args.") and key not in ("tp", "pp"):
        return key, raw
    try:
        return key, tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return key, raw


def set_nested(final: Final, target: str, value: Any, layer: str, source: str) -> None:
    """args.a.b = value: set key b inside the table argument a."""
    parts = target.split(".")
    key = "args." + normalize_arg_name(parts[1])
    if key in GEOMETRY_ARGS:
        raise Refusal(f"{source}: set tp/pp, not --{key[5:]}")
    if len(parts) == 2:
        final.put(key, value, layer, source)
        return
    current = copy.deepcopy(final.get(key) or {})
    if not isinstance(current, dict):
        raise Refusal(f"{source}: {key} is not a table argument")
    node = current
    for p in parts[2:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    final.put(key, current, layer, source)


@dataclass
class Inputs:
    runtime: str
    runtime_cfg: dict
    layout: str
    variants: list[str]
    overlays: list[tuple[str, dict]]     # (source, table)
    community: list[tuple[str, dict]]    # files from the community settings directory
    env_file: dict[str, str]
    ambient: dict[str, str]
    sets: list[str]
    extra_args: list[str]
    policy: str
    experiment: str | None
    capability: str
    tree: str
    venv: str
    home: str
    checkpoint_paths: dict[str, str]
    arg_options: ArgOptions = field(default_factory=ArgOptions)


@dataclass
class Resolved:
    final: Final
    variants: list[str]
    argv: list[str]
    argv_functional: list[str]
    location_keys: dict[str, Any]
    harness: list[str]
    child_env: dict[str, dict]
    env_functional: dict[str, str]
    ambient_equal: list[str]
    ambient_layer: list[str]
    dropped: list[str]
    ignored: list[str]
    fork_absent: list[str]
    passthrough: dict[str, str]
    absent_args: dict[str, str]
    not_validated: list[str]
    functional_sha256: str
    exe: str
    secret_names: set[str]
    # the served checkpoint, then every other one the arguments or the
    # environment name (sorted): the checkpoints this launch reads
    checkpoint_refs: list[str]
    checkpoint_files: dict[str, list[str]]


def declared_secrets(ms: ModelSettings) -> set[str]:
    """Environment names the model declares secret through [user] knobs."""
    out: set[str] = set()
    for spec in ms.user.values():
        if spec.get("secret") and spec.get("target"):
            targets = spec["target"] if isinstance(spec["target"], list) else [spec["target"]]
            out |= {t[4:] for t in targets if t.startswith("env.")}
    return out


def check_knob_names(ms: ModelSettings, classes: ClassTable, secrets: set[str]) -> None:
    """Knob names are the user-facing namespace: they may not shadow any name
    the class table classifies (a runtime sets inert names such as TERM
    itself), CUDA_VISIBLE_DEVICES excepted; a secret knob may only target a
    secret or an otherwise unclassified name."""
    for knob in ms.user:
        cls = classes.classify(knob)
        if cls != "unclassified" and knob != "CUDA_VISIBLE_DEVICES":
            raise Refusal(f"model.toml: user knob {knob} is a {cls} environment name; "
                          "name knobs apart from engine and runtime variables")
    for name in secrets:
        cls = classes.classify(name)
        if cls not in ("secret", "unclassified"):
            raise Refusal(f"model.toml: a secret knob targets {name}, a {cls} name; "
                          "secret targets must be secret or otherwise unclassified names")


def user_values(ms: ModelSettings, inp: Inputs) -> dict[str, str]:
    """[user] knob values: the .env file, plus (in a container, where the
    .env arrives as the process environment) the ambient knobs."""
    values = dict(inp.env_file)
    if inp.runtime == "container":
        for knob in ms.user:
            if knob in inp.ambient and knob not in values:
                values[knob] = inp.ambient[knob]
    return values


def _knob_variant(knob: str, spec: dict, raw: str) -> str | None:
    """The variant a knob value selects (None: none)."""
    sel = spec["variant"]
    if isinstance(sel, dict):
        if raw not in sel:
            raise Refusal(f"{knob}={raw!r}: expected one of {', '.join(sel)}")
        return sel[raw] or None
    v = raw.strip().lower()
    if v in _TRUE:
        return sel
    if v in _FALSE:
        return None
    raise Refusal(f"{knob}={raw!r}: expected 1 or 0")


def selected_variants(ms: ModelSettings, inp: Inputs, values: dict[str, str]) -> list[str]:
    """--variant names, then the variants [user] selector knobs pick; a
    selector given with --set overrides the same knob from the .env."""
    variants = list(inp.variants)
    chosen = dict(values)
    for item in inp.sets:
        key, _, raw = item.partition("=")
        if key in ms.user and ms.user[key].get("variant"):
            chosen[key] = raw
    for knob, spec in ms.user.items():
        if spec.get("variant") and knob in chosen:
            name = _knob_variant(knob, spec, chosen[knob])
            if name and name not in variants:
                variants.append(name)
    if inp.capability == "plain-upstream" and "plain-upstream" not in variants:
        variants.append("plain-upstream")
    return variants


def resolve(ms: ModelSettings, files: dict[str, dict], registry: Registry,
            classes: ClassTable, harness_args: list[str], inp: Inputs) -> Resolved:
    """files: 'hardware', 'layout', 'variant:<name>' → parsed tables."""
    final = Final()
    rt = inp.runtime_cfg
    run = inp.runtime
    harness_set = {normalize_arg_name(h) for h in harness_args}
    secret_names = declared_secrets(ms)
    check_knob_names(ms, classes, secret_names)
    # runtime layer
    final.put("args.host", rt["host"], "runtime", f"defaults.toml[runtime.{run}]")
    for k, v in rt.get("location", {}).items():
        final.put("loc." + k, str(v), "runtime", f"defaults.toml[runtime.{run}.location]")
    # model, hardware, layout
    apply_table(final, {k: ms.data[k] for k in ("args", "env", "location") if k in ms.data},
                "model", "model.toml", runtime=run)
    hw = files["hardware"]
    apply_table(final, {k: hw[k] for k in ("env", "location") if k in hw}, "hardware",
                f"hardware/{ms.hardware}.toml", runtime=run)
    apply_table(final, files["layout"], "layout", f"layouts/{inp.layout}.toml", runtime=run)
    # variants (user knobs that select a variant are read first)
    values = user_values(ms, inp)
    variants = selected_variants(ms, inp, values)
    passthrough, absent_args = engine_declarations(files, variants, inp.capability)
    check_protected_spellings(ms, inp.arg_options, absent_args)
    for v in variants:
        table = files.get(f"variant:{v}")
        if table is None:
            raise Refusal(f"variant {v!r} has no variants/{v}.toml")
        apply_table(final, table, "variant", f"variants/{v}.toml", layout=inp.layout, runtime=run)
    # what the settings themselves decided; later layers may change protected keys only
    # under --experiment
    settled = {k: e.value for k, e in final.entries.items()}
    # overlays, then the community settings directory (same level)
    for source, table in inp.overlays:
        if table.get("class") == "harness":
            _check_harness_overlay(source, table, harness_set)
        apply_table(final, table, "overlay", source, layout=inp.layout, runtime=run)
    for source, table in inp.community:
        apply_table(final, table, "overlay", source, layout=inp.layout, runtime=run)
    # user layer
    extra_args = list(inp.extra_args)
    for knob, raw in values.items():
        spec = ms.user.get(knob)
        if spec is None:
            raise Refusal(f".env: {knob} is not a documented setting of this model "
                          f"(known: {', '.join(sorted(ms.user)) or 'none'})")
        if spec.get("target"):
            _apply_user(final, knob, spec, raw, extra_args, variants)
    # ambient
    amb = _ambient(final, ms, registry, classes, inp, secret_names)
    for key, value in amb["layer"].items():
        final.put(key, value, "ambient", "environment")
    # CLI layer
    for key, value in (parse_set(s) for s in inp.sets):
        if key in ms.user:
            if ms.user[key].get("target"):
                _apply_user(final, key, ms.user[key], str(value), extra_args, variants,
                            layer="cli", source="--set")
        elif key == "cards":
            final.put("card_ids", str(value), "cli", "--set")
        elif key.startswith("args."):
            set_nested(final, key, value, "cli", "--set")
        elif key.startswith("env."):
            if not _ENV_NAME.match(key[4:]):
                raise Refusal(f"--set {key}: not an environment name")
            final.put(key, str(value), "cli", "--set")
        elif key in ("tp", "pp"):
            apply_table(final, {key: value}, "cli", "--set")
        else:
            raise Refusal(f"--set {key}: expected env.NAME, args.NAME, cards, tp, pp or a [user] knob")
    _cards_against_lock(final, amb, inp)
    if extra_args:
        _check_extra(ms, extra_args, inp.experiment, inp.arg_options)
        final.put("extra_args", list(extra_args), "user", "EXTRA_ARGS")
    _check_protected(final, ms, inp, settled)

    _check_geometry(final, hw, inp.layout)
    for rule in ms.refuse:
        when = rule.get("when", {})
        if when and all(str(final.get(k, "")) == str(v) for k, v in when.items()):
            raise Refusal(f"refused combination {when}: {rule.get('why', '')}".strip())
    fork_absent = _check_env_names(final, registry, classes, inp.capability, secret_names, passthrough)
    return _render(final, ms, inp, rt, harness_set, amb, variants, fork_absent, passthrough,
                   absent_args, secret_names, classes)


def engine_declarations(files: dict[str, dict], variants: list[str],
                        capability: str) -> tuple[dict[str, str], dict[str, str]]:
    """What the applied variants declare about a plain-upstream engine tree,
    each name with its reason (NAME = "why" or { why = "..." }):

    [passthrough]  environment names a package of that engine reads which
                   its vllm/envs.py does not declare (a plugin's switches,
                   the model code's own names): forwarded as functional
                   switches, in the functional hash, never dropped;
    [absent_args]  fixed or controlled arguments of the model that the
                   engine's CLI does not declare.

    Only a variant declares them, and only under capability plain-upstream:
    the sm80 registry declares every name the fork reads."""
    passthrough: dict[str, str] = {}
    absent: dict[str, str] = {}
    for v in variants:
        table = files.get(f"variant:{v}") or {}
        source = f"variants/{v}.toml"
        for section, into in (("passthrough", passthrough), ("absent_args", absent)):
            decl = table.get(section)
            if decl is None:
                continue
            if capability != "plain-upstream":
                raise Refusal(f"{source}: [{section}] applies only under capability plain-upstream")
            if not isinstance(decl, dict):
                raise Refusal(f"{source}: [{section}] is a table of NAME = \"why\"")
            for name, why in decl.items():
                if isinstance(why, dict) and set(why) == {"why"}:
                    why = why["why"]
                if not isinstance(why, str) or not why.strip():
                    raise Refusal(f"{source}: [{section}] {name} needs its reason (a why string)")
                if section == "passthrough" and not _ENV_NAME.match(name):
                    raise Refusal(f"{source}: [passthrough] {name!r} is not an environment name")
                into[normalize_arg_name(name) if section == "absent_args" else name] = why
    return passthrough, absent


def _check_harness_overlay(source: str, table: dict, harness_set: set[str]) -> None:
    """A harness overlay may only add the launcher's instrumentation arguments."""
    extra = set(table) - {"schema", "class", "args"}
    bad = [k for k in table.get("args", {}) if normalize_arg_name(k) not in harness_set]
    if extra or bad:
        raise Refusal(f"{source}: a harness overlay may set only {', '.join(sorted(harness_set))}"
                      f" (found {', '.join(sorted(extra) + bad)})")


def _cards_against_lock(final: Final, amb: dict, inp: Inputs) -> None:
    """Ambient CUDA_VISIBLE_DEVICES (the lock's cards) against the final cards."""
    lock = amb["cvd"]
    if lock is None:
        return
    want = final.get("card_ids")
    if want is None:
        final.put("card_ids", lock, "ambient", "environment")
    elif str(want) == lock:
        amb["equal"].append("CUDA_VISIBLE_DEVICES")
    elif inp.policy == "strict":
        raise Refusal(f"CUDA_VISIBLE_DEVICES={lock!r} (the cards this process holds) differs from "
                      f"the resolved cards {want!r}")
    elif final.entries["card_ids"].layer != "cli":
        final.put("card_ids", lock, "ambient", "environment")


def _check_geometry(final: Final, hw: dict, layout: str) -> None:
    hw_count = hw.get("count")
    tp = final.get("tp", 1)
    pp = final.get("pp", 1)
    cards = final.get("cards", tp * pp)
    if cards != tp * pp:
        raise Refusal(f"layout {layout}: cards {cards} != tp {tp} x pp {pp}")
    if isinstance(hw_count, int) and cards > hw_count:
        raise Refusal(f"layout {layout}: needs {cards} cards, hardware has {hw_count}")
    card_ids = final.get("card_ids")
    if card_ids is not None:
        ids = [c.strip() for c in str(card_ids).split(",") if c.strip()]
        if len(ids) != cards or len(set(ids)) != len(ids):
            raise Refusal(f"cards {card_ids!r}: layout {layout} needs exactly {cards} distinct cards")
        if isinstance(hw_count, int) and any(not i.isdigit() or int(i) >= hw_count for i in ids):
            raise Refusal(f"cards {card_ids!r}: not cards of this hardware (0..{hw_count - 1})")


def _check_env_names(final: Final, registry: Registry, classes: ClassTable,
                     capability: str, secrets: set[str], passthrough: dict[str, str]) -> list[str]:
    """Settings env names: functional or secret class (declared secret
    targets: secret or unclassified); VLLM_* registered. Location names:
    location class. A [passthrough] name: functional or unclassified, not one
    the engine tree declares, set by some layer."""
    fork_absent: list[str] = []
    env_names = {key[4:] for key in final.entries if key.startswith("env.")}
    for key in list(final.entries):
        source = final.entries[key].source
        if key.startswith("loc."):
            name = key[4:]
            cls = classes.classify(name)
            if cls != "location" or name == "CUDA_VISIBLE_DEVICES":
                raise Refusal(f"{source}: {name} is class {cls}; [location] holds cache and "
                              "home locations only (cards are set with card_ids)")
            continue
        if not key.startswith("env."):
            continue
        name = key[4:]
        if name in passthrough:
            cls = classes.classify(name)
            if name in secrets or cls not in ("functional", "unclassified"):
                raise Refusal(f"{source}: {name} is class {cls}; [passthrough] forwards "
                              "functional switches only")
            if cls == "functional" and not (name.upper().startswith("VLLM_") and registry.check(name)):
                raise Refusal(f"{source}: {name} needs no [passthrough]: the engine tree declares it "
                              "or its class is functional")
            continue
        if name.upper().startswith("VLLM_"):
            problem = registry.check(name)
            if problem:
                if capability == "plain-upstream" and name == name.upper() \
                        and name not in registry.retired:
                    fork_absent.append(name)
                    final.drop(key, "capability")
                    continue
                raise Refusal(f"{source}: {problem}")
        cls = classes.classify(name)
        if name in secrets:
            continue                    # checked at load: secret or unclassified
        if cls not in ("functional", "secret"):
            raise Refusal(f"{source}: {name} is class {cls}; settings env holds functional switches only")
    unset = sorted(set(passthrough) - env_names)
    if unset:
        raise Refusal(f"[passthrough] {', '.join(unset)}: no settings layer sets it")
    return fork_absent


def _scalar(name: str, value: Any, kinds: tuple) -> None:
    if isinstance(value, bool) and bool not in kinds or not isinstance(value, kinds):
        raise Refusal(f"--{name}: expected a plain {'/'.join(k.__name__ for k in kinds)} value, "
                      f"got {type(value).__name__}")


def _checkpoint_fields(final: Final, ms: ModelSettings, inp: Inputs) -> dict[str, list[str]]:
    """Require tracked references wherever the engine consumes checkpoint paths.

    Directory fields name the entire checkpoint. File fields name a relative
    member whose presence in that checkpoint's manifest is checked by build().
    JSON string arguments are inspected without changing their rendered bytes.
    """
    files: dict[str, list[str]] = {}

    def reference(value: Any, where: str, *, file: bool = False) -> None:
        if value is None or (file and value == ""):
            return
        match = (_CHECKPOINT_FILE if file else _CHECKPOINT_REF).fullmatch(value) \
            if isinstance(value, str) else None
        if match is None or match.group(1) not in ms.checkpoints \
                or match.group(1) not in inp.checkpoint_paths:
            raise Refusal(f"{where} must be a manifest-backed checkpoint reference; "
                          "move checkpoints with --checkpoint-path NAME=DIR")
        if file:
            rel = match.group(2)
            if any(part in ("", ".", "..") for part in rel.split("/")):
                raise Refusal(f"{where}: file reference must stay inside its checkpoint manifest")
            files.setdefault(match.group(1), []).append(rel)

    for key, entry in final.entries.items():
        if key == "env.VLLM_EXL3_OVERRIDE":
            reference(entry.value, key, file=True)
        elif key.startswith("args."):
            name = key[5:]
            dest = reaches(name, CHECKPOINT_ARGS, inp.arg_options)
            if dest is None or dest == "model":
                continue  # args.model is required separately, including null/unset
            if name != dest:
                raise Refusal(f"{key}: use args.{dest} with manifest-backed checkpoint references, "
                              "not an alias, dotted option or abbreviation")
            value = entry.value
            if CHECKPOINT_ARGS[dest] is None:
                reference(value, key)
                continue
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    raise Refusal(f"{key}: expected a JSON object with checkpoint references") from None
            if value is None:
                continue
            if not isinstance(value, dict):
                raise Refusal(f"{key}: expected an object with checkpoint references")
            def check_fields(node: dict, where: str) -> None:
                for field, sub in node.items():
                    if field in CHECKPOINT_ARGS[dest]:
                        reference(sub, f"{where}.{field}")
                    elif dest == "hf-overrides" and isinstance(sub, dict):
                        check_fields(sub, f"{where}.{field}")
            check_fields(value, key)
    return files


def _render(final: Final, ms: ModelSettings, inp: Inputs, rt: dict, harness_set: set[str],
            amb: dict, variants: list[str], fork_absent: list[str], passthrough: dict[str, str],
            absent_args: dict[str, str], secret_names: set[str],
            classes: ClassTable) -> Resolved:
    subst = {"home": inp.home, "venv": inp.venv, "tree": inp.tree,
             "plugin": plugin_dir(inp.tree) or inp.tree,
             "models_dir": substitute(rt["models_dir"], {"home": inp.home}, "runtime")}
    for name, path in inp.checkpoint_paths.items():
        subst[f"checkpoint.{name}.path"] = path
    model_value = final.get("args.model")
    ref = _CHECKPOINT_REF.match(model_value) if isinstance(model_value, str) else None
    if ref is None or ref.group(1) not in inp.checkpoint_paths:
        raise Refusal(f"args.model must be a checkpoint reference such as ${{checkpoint.target.path}} "
                      f"(got {model_value!r}); serve another directory with "
                      "--checkpoint-path NAME=DIR, which is checked against the manifest")
    checkpoint_files = _checkpoint_fields(final, ms, inp)
    refs = [ref.group(1)]
    exe = os.path.join(inp.venv, "bin", "vllm")
    argv = [exe, "serve", substitute(model_value, subst, "args.model")]
    # the served checkpoint's reference (not its directory) is functional
    argv_functional: list[str] = [model_value]
    harness: list[str] = []
    location_keys: dict[str, Any] = {"args.model": argv[2]}
    tp, pp = final.get("tp", 1), final.get("pp", 1)
    geometry = {"args.tensor-parallel-size": tp, "args.pipeline-parallel-size": pp}
    keys = [k for k in final.entries if k.startswith("args.") and k != "args.model"] + list(geometry)
    ordered = [f"args.{k}" for k in ms.arg_order if k != "model" and f"args.{k}" in keys]
    ordered += [k for k in keys if k not in ordered]
    for key in ordered:
        name = key[5:]
        value = geometry[key] if key in geometry else final.get(key)
        if name in LOCATION_ARGS:
            _scalar(name, value, (str, int))
        elif name in harness_set:
            _scalar(name, value, (bool, str, int, float, dict))
        tokens = render_arg(name, substitute(value, subst, key))
        argv += tokens
        if name in LOCATION_ARGS:
            location_keys[key] = value
        elif name in harness_set:
            harness += tokens
        else:
            # placeholders stay unexpanded: paths are location, not function
            argv_functional += render_arg(name, value)
    for tok in final.get("extra_args", []) or []:
        argv.append(tok)
        argv_functional.append(tok)

    child: dict[str, dict] = {}
    env_functional: dict[str, str] = {}
    for key, e in final.entries.items():
        if key.startswith("env."):
            name = key[4:]
            cls = "secret" if name in secret_names else \
                "functional" if name in passthrough else classes.classify(name)
            child[name] = {"value": substitute(e.value, subst, key), "class": cls}
            if cls == "functional":
                env_functional[name] = e.value
            elif cls == "secret":
                secret_names.add(name)
        elif key.startswith("loc."):
            name = key[4:]
            child[name] = {"value": substitute(e.value, subst, key), "class": "location"}
            location_keys[key] = child[name]["value"]
    card_ids = final.get("card_ids")
    if card_ids is not None:
        child["CUDA_VISIBLE_DEVICES"] = {"value": str(card_ids), "class": "location"}
        location_keys["CUDA_VISIBLE_DEVICES"] = str(card_ids)
    for name, value in amb["code"].items():
        child[name] = {"value": value, "class": "code-loading"}
    for name, (value, cls) in amb["passed"].items():
        child.setdefault(name, {"value": value, "class": cls})
        if cls == "secret":
            secret_names.add(name)
    used = set().union(*(checkpoint_uses(e.value) for k, e in final.entries.items()
                         if k.split(".", 1)[0] in ("args", "env", "loc")))
    refs += sorted(used - set(refs))
    functional_sha256 = canonical_sha256({"argv": argv_functional, "env": env_functional})
    return Resolved(
        final=final, variants=variants, argv=argv, argv_functional=argv_functional,
        location_keys=location_keys, harness=harness, child_env=child,
        env_functional=env_functional, ambient_equal=amb["equal"],
        ambient_layer=sorted(amb["layer"]), dropped=amb["dropped"],
        ignored=amb["ignored"], fork_absent=fork_absent,
        passthrough=dict(passthrough), absent_args=dict(absent_args),
        not_validated=list(amb["not_validated"]),
        functional_sha256=functional_sha256, exe=exe, secret_names=secret_names,
        checkpoint_refs=refs, checkpoint_files=checkpoint_files,
    )


def _apply_user(final: Final, knob: str, spec: dict, raw: str, extra_args: list[str],
                variants: list[str], *, layer: str = "user", source: str | None = None) -> None:
    source = source or f".env:{knob}"
    only = spec.get("variants")
    if only and not set(only) & set(variants):
        raise Refusal(f"{knob} applies only with variant {' or '.join(only)} "
                      f"(active: {', '.join(variants) or 'none'})")
    choices = spec.get("choices")
    if choices is not None and raw not in [str(c) for c in choices]:
        raise Refusal(f"{knob}: value not one of {', '.join(map(str, choices))}")
    value: Any = raw
    kind = spec.get("type", "str")
    try:
        if spec.get("divide_by"):
            value = int(raw) // int(spec["divide_by"])
        elif kind == "int":
            value = int(raw)
    except ValueError:
        raise Refusal(f"{knob}: expected an integer") from None
    if kind == "bool":
        low = raw.strip().lower()
        if low not in _TRUE + _FALSE[1:]:
            raise Refusal(f"{knob}: expected 1 or 0")
        value = low in _TRUE
    targets = spec["target"] if isinstance(spec["target"], list) else [spec["target"]]
    for target in targets:
        target = normalize_key(target)
        if target == "extra_args":
            extra_args.extend(shlex.split(raw))
        elif target == "card_ids":
            final.put("card_ids", raw, layer, source)
        elif target == "args.model":
            raise Refusal(f"model.toml: user.{knob} may not target args.model; "
                          "checkpoints move with --checkpoint-path")
        elif target.startswith("args."):
            set_nested(final, target, value, layer, source)
        elif target.startswith("env."):
            final.put(target, str(raw if kind == "bool" else value), layer, source)
        else:
            raise Refusal(f"model.toml: user.{knob} target {target!r} is not args.*, env.*, card_ids or extra_args")


def arg_destination(name: str, opts: ArgOptions) -> str:
    """The engine argument a spelling sets: a dotted option (a.b) sets its
    parent a, an alias its first name, a negative form of the tree's parser
    (--no-X; --disable-X where it pairs --enable-X) the flag X."""
    return opts.destination(normalize_arg_name(name).split(".", 1)[0])


def reaches(name: str, guarded: set[str] | frozenset[str], opts: ArgOptions) -> str | None:
    """The guarded argument a spelling sets, if any: by its destination, or
    as an abbreviation argparse expands to a guarded option (or its alias or
    a negative form of either). A spelling the tree declares exactly is never an
    abbreviation. Both sides are compared as destinations: the guarded
    names are canonicalised the same way as the incoming one."""
    guarded = {arg_destination(g, opts) for g in guarded}
    dest = arg_destination(name, opts)
    if dest in guarded:
        return dest
    spelled = normalize_arg_name(name).split(".", 1)[0]
    if not spelled or spelled in opts.options:
        return None
    for g in sorted(guarded):
        forms = {g} | {a for a, c in opts.aliases.items() if c == g}
        if any(f.startswith(spelled) for f in forms | {n for f in forms for n in opts.negative_forms(f)}):
            return g
    return None


def check_protected_spellings(ms: ModelSettings, opts: ArgOptions,
                              absent: dict[str, str] | None = None) -> None:
    """A fixed or controlled argument is declared by a full option name the
    engine tree declares (its own name or an alias), so it names exactly one
    destination; an abbreviation, a negative form or a name the tree does not
    declare is refused rather than resolved, unless a plain-upstream variant
    declares it absent from this engine ([absent_args], with its reason)."""
    declared = [k[5:].split(".", 1)[0] for k in sorted(ms.fixed) if k.startswith("args.")]
    declared += sorted(ms.controlled_args)
    names = {normalize_arg_name(a) for a in declared}
    absent = absent or {}
    stale = sorted(set(absent) & opts.options)
    if stale:
        raise Refusal(f"[absent_args] {', '.join(stale)}: this engine tree declares it; drop the declaration")
    foreign = sorted(set(absent) - names)
    if foreign:
        raise Refusal(f"[absent_args] {', '.join(foreign)}: not a fixed or controlled argument of this model")
    bad = sorted(names - opts.options - set(absent))
    if bad:
        raise Refusal("model.toml: fixed and controlled arguments must be full option names of "
                      f"this engine tree; not {', '.join(bad)}")


def _protected_args(ms: ModelSettings) -> set[str]:
    return {k[5:].split(".", 1)[0] for k in ms.fixed if k.startswith("args.")} | set(ms.controlled_args)


def _check_extra(ms: ModelSettings, extra: list[str], experiment: str | None,
                 opts: ArgOptions) -> None:
    """Raw extra arguments: long options only; never what the launcher sets
    (geometry, the model, an argument file); fixed and controlled arguments
    only under --experiment, by the argument each spelling sets."""
    protected = _protected_args(ms)
    for tok in extra:
        if _SHORT_OPTION.match(tok):
            raise Refusal(f"extra argument {tok}: short options are not accepted; use the long form")
        if not tok.startswith("--"):
            continue
        name = tok[2:].split("=", 1)[0]
        hit = reaches(name, LAUNCHER_ARGS, opts)
        if hit:
            raise Refusal(f"extra argument --{name} sets --{hit}, which is set by the launcher; use "
                          "tp/pp, --checkpoint-path and the settings")
        hit = reaches(name, CHECKPOINT_ARGS, opts)
        if hit:
            raise Refusal(f"extra argument --{name} sets --{hit}, which can read checkpoints; "
                          "use settings with manifest-backed checkpoint references")
        hit = reaches(name, protected, opts)
        if hit and not experiment:
            raise Refusal(f"extra argument --{name} sets --{hit}, which is fixed by the settings; "
                          "change it with --set under --experiment")


def _protected(key: str, ms: ModelSettings, opts: ArgOptions) -> bool:
    if key in ms.fixed:
        return True
    return key.startswith("args.") and reaches(key[5:], _protected_args(ms), opts) is not None


def _check_protected(final: Final, ms: ModelSettings, inp: Inputs, settled: dict[str, Any]) -> None:
    """[fixed] keys and controlled arguments change only under --experiment,
    through any channel: a new value, a new key or an unset, under any
    spelling of the argument."""
    if final.get("args.model") != settled.get("args.model"):
        raise Refusal("args.model changes only through --checkpoint-path")
    opts = inp.arg_options
    for key in final.entries:
        if key.startswith("args.") and key != "args.model":
            hit = reaches(key[5:], LAUNCHER_ARGS, opts)
            if hit:
                raise Refusal(f"{key} sets --{hit}, which is set by the launcher; use tp/pp, "
                              "--checkpoint-path and the settings")
    if inp.experiment:
        return
    for key, e in final.entries.items():
        if e.layer in LOCKED_LAYERS and _protected(key, ms, opts):
            if key == "card_ids" and e.layer == "ambient":
                continue  # the lock's cards, checked against the settings above
            if key not in settled or settled[key] != e.value:
                raise Refusal(f"{key} is fixed for this model ({e.source}); "
                              "it changes only under --experiment")
    for key, layer in final.unset.items():
        if layer in LOCKED_LAYERS and _protected(key, ms, opts) and key in settled:
            raise Refusal(f"{key} is fixed for this model; unsetting it needs --experiment")


def _ambient(final: Final, ms: ModelSettings, registry: Registry, classes: ClassTable,
             inp: Inputs, secrets: set[str]) -> dict:
    """Classify the ambient environment and apply the policy."""
    out: dict = {"equal": [], "layer": {}, "dropped": [], "ignored": [], "passed": {},
                 "code": {}, "not_validated": [], "cvd": None}
    strict = inp.policy == "strict"
    container = inp.runtime == "container"
    rt = inp.runtime_cfg
    built_code = {
        "PATH": ":".join([os.path.join(inp.venv, "bin"), os.path.join(rt["cuda_home"], "bin"),
                          "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin",
                          "/sbin", "/bin"]),
        "PYTHONPATH": code_path(inp.tree),
        "PYTHONNOUSERSITE": "1",
        "VIRTUAL_ENV": inp.venv,
        "CUDA_HOME": rt["cuda_home"],
    }
    image_code, image_values = _image_env(rt.get("image_env", {}) if container else {}, classes)
    for name, value in image_code.items():
        built_code.setdefault(name, value)
    out["code"] = dict(built_code)
    for name, value in image_values.items():
        if "env." + name not in final.entries:
            out["passed"][name] = (value, classes.classify(name))
    refusals: list[str] = []
    for name in sorted(inp.ambient):
        value = inp.ambient[name]
        if container and name in ms.user:
            continue  # read as a [user] knob
        if name in image_values and "env." + name not in final.entries:
            want = image_values[name]
            if value == want:
                out["equal"].append(name)
            elif strict:
                refusals.append(f"{name}={value!r} in the environment differs from the image's "
                                f"{want!r} (strict policy)")
            else:
                out["passed"][name] = (value, classes.classify(name))
                out["not_validated"].append(f"ambient {name}")
            continue
        if name.upper().startswith("VLLM_") and name != name.upper():
            refusals.append(registry.check(name) or f"{name}: not a registered name")
            continue
        cls = "secret" if name in secrets else classes.classify(name)
        if cls == "secret":
            problem = registry.check(name) if name.startswith("VLLM_") else None
            if problem:
                refusals.append(problem)
            else:
                out["passed"][name] = (value, cls)
        elif cls == "ownership":
            if container:
                out["dropped"].append(name)
            else:
                out["passed"][name] = (value, cls)
        elif cls in ("container-runtime", "inert"):
            out["passed"][name] = (value, cls)
        elif cls == "code-loading":
            if always_refused(name):
                refusals.append(f"{name} in the environment is refused: it changes what the server loads")
            elif name == "PATH":
                out["ignored"].append(name)
            elif built_code.get(name) == value:
                out["equal"].append(name)
            elif strict:
                refusals.append(f"{name}={value!r} in the environment differs from the launcher's "
                                f"{built_code.get(name)!r} (strict policy)")
            else:
                out["code"][name] = value
                out["not_validated"].append(f"ambient {name}")
        elif cls == "location":
            if name == "CUDA_VISIBLE_DEVICES":
                out["cvd"] = value       # checked against the final cards after --set
                continue
            want = final.get("loc." + name)
            try:
                want = substitute(want, {"home": inp.home, "venv": inp.venv, "tree": inp.tree,
                                         "plugin": plugin_dir(inp.tree) or inp.tree}, name)
            except Refusal:
                pass
            if want == value:
                out["equal"].append(name)
            else:
                out["ignored"].append(name)
        elif cls == "functional":
            if name.startswith("VLLM_"):
                problem = registry.check(name)
                if problem:
                    refusals.append(problem)
                    continue
            key = "env." + name
            declared = key in final.entries
            if declared and final.get(key) == value:
                out["equal"].append(name)
            elif key in ms.fixed and not inp.experiment:
                refusals.append(f"{name} is fixed for this model; it changes only under --experiment")
            elif strict:
                if declared:
                    refusals.append(f"{name}={value!r} in the environment differs from the settings "
                                    f"value {final.get(key)!r} (strict policy: use --set or --overlay)")
                else:
                    refusals.append(f"{name} in the environment is not declared by the settings "
                                    "(strict policy: use --set or --overlay)")
            else:
                out["layer"][key] = value
                out["not_validated"].append(f"ambient {name}")
        else:
            out["dropped"].append(name)
    if refusals:
        raise Refusal(*refusals)
    return out


def _image_env(image: dict, classes: ClassTable) -> tuple[dict[str, str], dict[str, str]]:
    """The container image's ENV (``[runtime.container.image_env]``, its
    complete config Env) split into the code-loading values the launcher
    builds from it (all but PATH, which the launcher builds itself) and the
    functional or inert values the environment must hold unchanged.
    Container-runtime names are passed as the runtime sets them; functional
    names only from ``IMAGE_FUNCTIONAL``; anything else is refused, so every
    name of the image is classified and none it serves unhashed is read."""
    code: dict[str, str] = {}
    values: dict[str, str] = {}
    for name, value in image.items():
        cls = classes.classify(name)
        if not isinstance(value, str):
            raise Refusal(f"defaults: image_env {name} is not a string")
        if always_refused(name):
            raise Refusal(f"defaults: image_env {name} is refused: it changes what the server loads")
        if cls == "code-loading":
            if name != "PATH":
                code[name] = value
        elif cls == "inert" or (cls == "functional" and name in IMAGE_FUNCTIONAL):
            values[name] = value
        elif cls == "functional":
            raise Refusal(f"defaults: image_env {name} is functional and not a name the engine "
                          "leaves unread (" + ", ".join(sorted(IMAGE_FUNCTIONAL)) + ")")
        elif cls != "container-runtime":
            raise Refusal(f"defaults: image_env {name} is class {cls}; the image ENV holds only "
                          "code-loading, container-runtime, unread functional or inert names")
    return code, values
