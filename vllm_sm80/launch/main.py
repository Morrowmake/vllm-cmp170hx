# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch ``vllm serve`` from one model settings directory.

Run by path, under an isolated interpreter::

    python3 -I vllm_sm80/launch/main.py --runtime native --tree <engine tree> \\
        --venv <venv> --settings <dir> --layout <name> [--variant V ...] \\
        [--overlay FILE ...] [--set KEY=VALUE ...] [--env-file FILE] \\
        [--experiment ID] [--print-config [--json | --explain]] [-- extra args]

The registry of switch names is read from ``--tree``, the tree that will
run, never from this launcher's own commit. Without ``--print-config`` the
launcher checks the checkpoints and ``exec``s the server; it keeps no child.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import checkpoint  # noqa: E402
from assembly import PACKAGE, plugin_dir  # noqa: E402
from checkpoint import StopFlag  # noqa: E402
from canon import HashCache, atomic_write, canonical_sha256, sha256_file  # noqa: E402
from envclass import ClassTable  # noqa: E402
from errors import EXIT_CHECKPOINT_ABSENT, EXIT_CHECKPOINT_INCOMPLETE, EXIT_CHECKPOINT_MISMATCH, Refusal  # noqa: E402
from model import load_model_settings  # noqa: E402
from registry import has_sm80_plugin, read_arg_options, read_registry  # noqa: E402
from resolver import Inputs, resolve, substitute  # noqa: E402
from settings_dir import SettingsDir, load_toml, parse_env_lines, tree_sha256  # noqa: E402
from treestate import tree_state  # noqa: E402
from validated import Evidence, compare_expected, evaluate, read_generation  # noqa: E402

SCHEMA = "launch-resolved/2"
_GIT_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", required=True)
    p.add_argument("--settings-extra", help="community settings directory (overlay level)")
    p.add_argument("--layout")
    p.add_argument("--variant", action="append", default=[])
    p.add_argument("--overlay", action="append", default=[])
    p.add_argument("--set", dest="sets", action="append", default=[])
    p.add_argument("--env-file")
    p.add_argument("--experiment")
    p.add_argument("--runtime", choices=("native", "container"), default="native")
    p.add_argument("--tree")
    p.add_argument("--venv")
    p.add_argument("--capability", choices=("sm80", "plain-upstream"), default="sm80")
    p.add_argument("--policy", choices=("strict", "merge"))
    p.add_argument("--checkpoint-verify", choices=("stamp", "boot"))
    p.add_argument("--models-dir")
    p.add_argument("--checkpoint-path", action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--stamp-dir")
    p.add_argument("--role", choices=("any", "candidate", "final"), default="any",
                   help="final: a receipt is required; candidate: none may be present")
    p.add_argument("--expect", help="frozen expected launch (launch-resolved/2 or "
                                    "validated-launch/1); a launch that differs is refused")
    p.add_argument("--print-config", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--out", help="write the --json output here (atomic)")
    p.add_argument("--verify-checkpoints", action="store_true",
                   help="hash the checkpoints this launch would serve, write their stamps, exit")
    ns = p.parse_args(argv)
    ns.extra_args = extra
    return ns


def _home(runtime: str, environ: dict[str, str]) -> str:
    if runtime == "container":
        return "/cache"
    return environ.get("HOME") or os.path.expanduser("~")


def _launcher_impl() -> dict:
    files = sorted(glob.glob(os.path.join(_HERE, "*.py")) + glob.glob(os.path.join(_HERE, "*.toml"))
                   + glob.glob(os.path.join(_HERE, "data", "*.toml")))
    pkg = {os.path.relpath(f, _HERE): sha256_file(f) for f in files}
    out = {"kind": "sm80-launch", "path": _HERE, "package_sha256": canonical_sha256(pkg),
           "commit": None, "dirty": None}
    root = os.path.dirname(os.path.dirname(_HERE))
    if os.path.exists(os.path.join(root, ".git")):
        try:
            out["commit"] = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                check=True, env=_GIT_ENV).stdout.strip()
            status = subprocess.run(
                ["git", "-C", root, "status", "--porcelain", "--", "vllm_sm80/launch"],
                capture_output=True, text=True, check=True, env=_GIT_ENV).stdout
            out["dirty"] = bool(status.strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    return out


def _requires(ms, tree: str, capability: str, cache: HashCache) -> dict[str, str]:
    """Hashes of the required files. Patterns are tree-relative (checked at
    load); in an assembled engine a pattern under vllm_sm80/ is relative to
    the plugin directory, with the same key. A match may be a link (linked
    trees link their artefacts), whose ownership and content the tree state
    checks; a dangling one is refused."""
    out: dict[str, str] = {}
    plugin = plugin_dir(tree)
    for ent in ms.requires:
        if capability == "plain-upstream" and ent.get("fork_only"):
            continue
        base = plugin if plugin and ent["path"].startswith(PACKAGE + "/") else tree
        hits = sorted(glob.glob(os.path.join(base, ent["path"])))
        if not hits:
            raise Refusal(f"required file {ent['path']} is missing in {base}")
        for h in hits:
            rel = os.path.relpath(h, base)
            if rel.startswith(".."):
                raise Refusal(f"required file {h} is outside the engine tree")
            try:
                out[rel] = cache.sha256(h)
            except OSError as exc:
                raise Refusal(f"required file {rel} cannot be read: {exc.strerror}") from None
    return out


def _hub_snapshot(hub_dirs: list[str], repo: str, revision: str) -> str | None:
    name = "models--" + repo.replace("/", "--")
    for hub in hub_dirs:
        p = os.path.join(hub, name, "snapshots", revision)
        if os.path.isdir(p):
            return p
    return None


def _variant_files(sd: SettingsDir, extra_sd: SettingsDir | None, community: list) -> dict:
    """Every variant file, baked and community (a community one replaces
    the baked one of the same name)."""
    out: dict[str, dict] = {}
    for d in (sd, extra_sd):
        if d is None or not os.path.isdir(d.path("variants")):
            continue
        for name in sorted(os.listdir(d.path("variants"))):
            if not name.endswith(".toml"):
                continue
            rel = d.named("variants", name[:-5])
            out[f"variant:{name[:-5]}"] = d.toml(rel, "launch-variant")
            if d is extra_sd:
                community.append((f"extra:{rel}", {"schema": "launch-variant/1"}))
    return out


def checkpoint_paths(ns, ms, rt: dict, pins: dict, home: str) -> dict[str, str]:
    """Where each checkpoint is: --checkpoint-path, else <models_dir>/<dir>
    (``container_dir`` in a container), else in a container the HF-cache
    snapshot of the pinned revision. Launch and --verify-checkpoints share it."""
    models_dir = substitute(ns.models_dir or rt["models_dir"], {"home": home}, "models_dir")
    overrides: dict[str, str] = {}
    for item in ns.checkpoint_path:
        name, sep, path = item.partition("=")
        if not sep or name not in ms.checkpoints:
            raise Refusal(f"--checkpoint-path {item!r}: expected NAME=DIR with NAME one of "
                          f"{', '.join(ms.checkpoints)}")
        overrides[name] = os.path.abspath(path)
    out: dict[str, str] = {}
    for name, spec in ms.checkpoints.items():
        if name in overrides:
            out[name] = overrides[name]
            continue
        container = ns.runtime == "container"
        direct = os.path.join(models_dir, spec.get("container_dir", spec["dir"]) if container else spec["dir"])
        if container and not os.path.isdir(direct):
            snap = _hub_snapshot(rt.get("hub_dirs", []), pins.get(spec["repo_key"], ""),
                                 pins.get(spec["revision_key"], ""))
            direct = snap or direct
        out[name] = direct
    return out


def hardware_cuda_home(hw: dict, runtime: str, image_cuda_home: str | None) -> str | None:
    """The hardware file's CUDA for this runtime: ``cuda_home = "<dir>"``
    names the native CUDA only; ``cuda_home = { native = "<dir>",
    container = "<dir>" }`` names it per runtime (either may be left out).
    None leaves the runtime default. A container entry must name the
    image's CUDA_HOME (``[runtime.container.image_env]``): the image's own
    environment sets it, so any other directory would differ from every
    container's environment."""
    value = hw.get("cuda_home")
    if isinstance(value, dict):
        unknown = sorted(set(value) - {"native", "container"})
        if unknown:
            raise Refusal(f"hardware cuda_home: unknown runtime {', '.join(unknown)} (native, container)")
        for name, v in value.items():
            if not isinstance(v, str) or not v:
                raise Refusal(f"hardware cuda_home.{name}: not a directory name")
        if "container" in value and value["container"] != image_cuda_home:
            raise Refusal(f"hardware cuda_home.container {value['container']!r} differs from the "
                          f"image's CUDA_HOME {image_cuda_home!r}")
        return value.get(runtime)
    if value is not None and not isinstance(value, str):
        raise Refusal("hardware cuda_home: a directory name or a { native, container } table")
    return value if value and runtime == "native" else None


def _context(ns, environ):
    defaults = load_toml(os.path.join(_HERE, "defaults.toml"), "launch-defaults")
    rt = dict(defaults["runtime"][ns.runtime])
    home = _home(ns.runtime, environ)
    sd = SettingsDir(ns.settings)
    ms = load_model_settings(sd)
    stamp_dir = substitute(ns.stamp_dir or rt["stamp_dir"], {"home": home}, "stamp_dir")
    return defaults, rt, home, sd, ms, stamp_dir


def build(ns: argparse.Namespace, environ: dict[str, str]) -> dict:
    """Everything except exec: the launch-resolved/2 document plus private
    fields (prefixed with '_') used to exec."""
    defaults, rt, home, sd, ms, stamp_dir = _context(ns, environ)
    if not ns.tree and ns.runtime == "native":
        raise Refusal("--tree is required natively: name the engine tree that will run")
    tree = os.path.abspath(ns.tree or rt.get("tree") or "")
    venv = ns.venv or rt.get("venv") or ""
    if not venv:
        raise Refusal("--venv is required natively")
    policy = ns.policy or rt["policy"]
    hw = sd.toml(sd.named("hardware", ms.hardware), "launch-hardware")
    cuda_home = hardware_cuda_home(hw, ns.runtime,
                                   defaults["runtime"]["container"].get("image_env", {}).get("CUDA_HOME"))
    if cuda_home:
        rt["cuda_home"] = cuda_home
    layout = ns.layout or ms.default_layout
    if not layout:
        raise Refusal("no --layout and model.toml names no default layout")
    community: list[tuple[str, dict]] = []
    extra_sd = SettingsDir(ns.settings_extra, community=True) if ns.settings_extra else None
    layout_rel = sd.named("layouts", layout)
    extra_layout = extra_sd is not None and extra_sd.exists(layout_rel)
    if layout not in ms.allowed_layouts and not extra_layout:
        raise Refusal(f"layout {layout!r} is not allowed for {ms.model} "
                      f"(allowed: {', '.join(ms.allowed_layouts)})")
    if extra_layout:
        community.append((f"extra:{layout_rel}", extra_sd.toml(layout_rel, "launch-layout")))
    if sd.exists(layout_rel):
        layout_table = sd.toml(layout_rel, "launch-layout")
    elif extra_layout:
        # a layout only the community directory defines is that launch's layout
        # (always NOT VALIDATED: "community layer")
        layout_table = community[-1][1]
    else:
        raise Refusal(f"no layouts/{layout}.toml")
    files: dict[str, dict] = {"hardware": hw, "layout": layout_table}
    files.update(_variant_files(sd, extra_sd, community))
    overlays = [(f"overlay:{os.path.basename(p)}", load_toml(p, "launch-overlay")) for p in ns.overlay]
    env_file: dict[str, str] = {}
    if ns.env_file:
        with open(ns.env_file, encoding="utf-8") as f:
            env_file = parse_env_lines(f.read(), ns.env_file, allow_quotes=True)

    # engine target and its registry
    if not os.path.isdir(tree):
        raise Refusal(f"engine tree {tree} does not exist")
    if ns.capability == "plain-upstream" and has_sm80_plugin(tree):
        raise Refusal(f"plain-upstream arm on {tree}, which carries the sm80 plugin")
    registry = read_registry(tree, ns.capability)
    classes = ClassTable(defaults["classes"], ms.env_prefixes)
    harness_args = list(defaults.get("harness", {}).get("args", []))

    pins = sd.pins()
    cp_paths = checkpoint_paths(ns, ms, rt, pins, home)
    inp = Inputs(
        runtime=ns.runtime, runtime_cfg=rt, layout=layout, variants=list(ns.variant),
        overlays=overlays, community=community, env_file=env_file, ambient=dict(environ),
        sets=list(ns.sets), extra_args=list(ns.extra_args), policy=policy,
        experiment=ns.experiment, capability=ns.capability, tree=tree, venv=venv,
        home=home, checkpoint_paths=cp_paths, arg_options=read_arg_options(tree))
    res = resolve(ms, files, registry, classes, harness_args, inp)

    # a community layer is NOT VALIDATED unless it changes nothing functional
    community_changed = False
    if community:
        if not sd.exists(layout_rel):
            community_changed = True
        else:
            base_files = {"hardware": hw, "layout": layout_table}
            base_files.update(_variant_files(sd, None, []))
            try:
                base = resolve(ms, base_files, registry, classes, harness_args,
                               Inputs(**{**inp.__dict__, "community": []}))
                community_changed = base.functional_sha256 != res.functional_sha256
            except Refusal:
                community_changed = True

    cache = HashCache(os.path.join(stamp_dir, "hash-cache.json"))
    ts = tree_state(tree, image=ns.runtime == "container", cache=cache)
    requires = _requires(ms, tree, ns.capability, cache)
    cache.save()

    # only the checkpoints this launch reads: another role of the model (one
    # not downloaded, say) neither refuses nor marks this launch
    cps: dict[str, dict] = {}
    for name in res.checkpoint_refs:
        spec = ms.checkpoints[name]
        manifest, msha = checkpoint.load_manifest(sd.path(spec["manifest"]))
        for rel in res.checkpoint_files.get(name, []):
            if rel not in manifest["files"]:
                raise Refusal(f"checkpoint {name}: referenced file {rel!r} is not in its manifest")
        st = checkpoint.status(manifest, msha, cp_paths[name], stamp_dir)
        st.update(repo=pins.get(spec["repo_key"]), revision=pins.get(spec["revision_key"]))
        if manifest.get("repo") != st["repo"] or manifest.get("revision") != st["revision"]:
            st["manifest_pins_differ"] = True
        cps[name] = st

    not_validated = list(res.not_validated)
    if ns.experiment:
        not_validated.append(f"experiment {ns.experiment}")
    if community_changed:
        not_validated.append("community layer")
    if any(cp.get("manifest_pins_differ") for cp in cps.values()):
        not_validated.append("checkpoint manifest ≠ pins")
    if ts.build_owned_sha256 is None or ts.native_manifest_sha256 is None:
        not_validated.append("no build identity")      # candidates too, receipt or not
    gen = read_generation(sd)           # one generation for every validated read and hash
    ev = None
    if ns.capability == "plain-upstream":
        # a plain upstream arm is never a validated launch and never a
        # quality reference: it has no validated state at all
        validated = {"state": None, "reasons": ["plain upstream"], "functional_match": None,
                     "role": ns.role}
    else:
        ev = Evidence(
            layout=layout, functional_sha256=res.functional_sha256, pins=pins,
            tree_commit=ts.commit, tree_dirty=ts.dirty,
            build_owned_sha256=ts.build_owned_sha256,
            native_manifest_sha256=ts.native_manifest_sha256,
            cross_commit_artifacts=ts.cross_commit_artifacts, adopted=ts.adopted,
            requires=requires,
            checkpoints={n: {k: c.get(k) for k in ("repo", "revision", "manifest_sha256", "state", "identity")}
                         for n, c in cps.items()},
            not_validated=not_validated, plugin=ts.plugin)
        validated = evaluate(gen, ev, role=ns.role)
    expected = None
    if ns.expect:
        with open(ns.expect, "rb") as f:
            expected = compare_expected(f.read(), res.argv_functional, res.env_functional, ns.expect)
    settings_files = sd.file_hashes(gen.files)

    secrets = res.secret_names

    def mask_child(name: str, ent: dict) -> dict:
        return {"value": "***", "class": "secret"} if name in secrets else ent

    def mask_resolved(key: str, value):
        return "***" if key.startswith("env.") and key[4:] in secrets else value
    return {
        "schema": SCHEMA,
        "model": ms.model,
        "layout": layout,
        "variants": res.variants,
        "overlays": [s for s, _ in overlays],
        "community": [s for s, _ in community],
        "runtime": ns.runtime,
        "policy": policy,
        "experiment": ns.experiment,
        "engine_target": dict(ts.describe(), capability=ns.capability),
        "launcher_impl": _launcher_impl(),
        "registry": registry.describe(),
        "settings": {"files": settings_files, "tree_sha256": tree_sha256(settings_files),
                     "baseline": ms.baseline},
        "argv": res.argv,
        "argv_shell": shlex.join(res.argv),
        "argv_functional": res.argv_functional,
        "served_checkpoint": res.checkpoint_refs[0],
        "harness": res.harness,
        "location_keys": res.location_keys,
        "child_env": {k: mask_child(k, v) for k, v in sorted(res.child_env.items())},
        "env_functional": res.env_functional,
        "ambient_equal": res.ambient_equal,
        "ambient_layer": res.ambient_layer,
        "dropped": res.dropped,
        "ignored": res.ignored,
        "fork_switches_absent": res.fork_absent,
        "passthrough": res.passthrough,
        "absent_args": res.absent_args,
        "resolved": {k: {"value": mask_resolved(k, e.value), "layer": e.layer,
                         "source": e.source, "why": e.why}
                     for k, e in res.final.entries.items()},
        "unset": res.final.unset,
        "nested_why": res.final.whys,
        "requires": requires,
        "checkpoint": cps,
        "functional_sha256": res.functional_sha256,
        "validated": validated,
        "expected": expected,
        "_exe": res.exe,
        "_env": {k: v["value"] for k, v in res.child_env.items()},
        "_stamp_dir": stamp_dir,
        "_cp_specs": {n: (sd.path(ms.checkpoints[n]["manifest"]), cp_paths[n]) for n in res.checkpoint_refs},
        "_checkpoint_verify": ns.checkpoint_verify or rt["checkpoint_verify"],
        "_gen": gen,
        "_ev": ev,
        "_role": ns.role,
    }


def public(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def banner(doc: dict) -> str:
    v = doc["validated"]
    if v["state"] == "VALIDATED":
        return (f"VALIDATED {doc['model']} {doc['layout']} "
                f"(engine {str(v.get('engine_commit'))[:10]}, gate {v.get('gate_record')})")
    return f"NOT VALIDATED ({', '.join(v['reasons'])})"


def print_text(doc: dict, explain: bool) -> None:
    out = [banner(doc), "", "argv:", "  " + doc["argv_shell"], "", "environment:"]
    for name, ent in doc["child_env"].items():
        out.append(f"  [{ent['class']}] {name}={ent['value']}")
    if doc["dropped"]:
        out.append("dropped: " + " ".join(doc["dropped"]))
    if doc["ambient_layer"]:
        out.append("from the environment: " + " ".join(doc["ambient_layer"]))
    if explain:
        out += ["", "resolved:"]
        for k, e in doc["resolved"].items():
            why = f"  # {e['why']}" if e.get("why") else ""
            out.append(f"  {k} = {json.dumps(e['value'])}  [{e['layer']}: {e['source']}]{why}")
    print("\n".join(out))


def preflight_checkpoints(doc: dict, policy: str, stop: list | None = None) -> None:
    """Before exec: refuse incomplete checkpoints, apply the boot policy, and
    restate VALIDATED from the checkpoints as they are now. ``stop`` is the
    command's StopFlag; a requested stop is never turned into a start."""
    mode = doc["_checkpoint_verify"]
    stop = stop if stop is not None else [False]
    for name, (mpath, directory) in doc["_cp_specs"].items():
        if stop[0]:
            break
        manifest, msha = checkpoint.load_manifest(mpath)
        problems = checkpoint.completeness(manifest, directory)
        if problems:
            raise Refusal(f"checkpoint {name} at {directory} is not complete: " + "; ".join(problems[:10]))
        sfile = checkpoint.stamp_path(doc["_stamp_dir"], directory)
        state, stale, bad = checkpoint.identity(manifest, msha, directory, sfile)
        if stale and mode == "boot":
            try:
                state, bad, _ = checkpoint.verify(manifest, msha, directory, sfile, stop=stop)
            except checkpoint.Interrupted:
                raise Refusal("interrupted while verifying a checkpoint; no stamp written") from None
        elif stale:
            state = checkpoint.UNVERIFIED
        if state == checkpoint.UNVERIFIED:
            raise Refusal(f"checkpoint {name} at {directory} is not verified; run: python3 -I "
                          f"{os.path.join(_HERE, 'checkpoint.py')} verify --settings <settings> "
                          f"--checkpoint {name} --path {directory}")
        if state == checkpoint.MISMATCH and policy == "strict":
            raise Refusal(f"checkpoint {name} at {directory} differs from its manifest: {', '.join(bad[:10])}")
        doc["checkpoint"][name].update(identity=state, state="complete")
        doc["checkpoint"][name].pop("stale", None)
        if bad:
            doc["checkpoint"][name]["files"] = bad
        if doc["_ev"] is not None:
            doc["_ev"].checkpoints[name].update(identity=state, state="complete")
    if stop[0]:
        raise Refusal("stopped before start (SIGTERM/SIGINT)")
    if doc["_ev"] is not None:
        doc["validated"] = evaluate(doc["_gen"], doc["_ev"], role=doc["_role"])


def verify_checkpoints(ns, environ, stop: list) -> int:
    """--verify-checkpoints: stamp exactly the directories a launch would serve
    (the checkpoints its resolution reads). ``stop`` is the command's
    StopFlag, installed before anything is read."""
    doc = build(ns, environ)
    worst = 0
    report = {}
    for name, (mpath, directory) in doc["_cp_specs"].items():
        if stop[0]:
            raise Refusal("stopped (SIGTERM/SIGINT); no stamp written")
        manifest, msha = checkpoint.load_manifest(mpath)
        if not os.path.isdir(directory):
            report[name] = {"state": checkpoint.NOT_PRESENT, "path": directory}
            worst = max(worst, EXIT_CHECKPOINT_ABSENT)
            continue
        problems = checkpoint.completeness(manifest, directory)
        if problems:
            report[name] = {"state": checkpoint.INCOMPLETE, "path": directory, "files": problems}
            worst = max(worst, EXIT_CHECKPOINT_INCOMPLETE)
            continue
        try:
            state, bad, hashed = checkpoint.verify(manifest, msha, directory,
                                                   checkpoint.stamp_path(doc["_stamp_dir"], directory),
                                                   stop=stop)
        except checkpoint.Interrupted:
            raise Refusal("interrupted; no stamp written") from None
        report[name] = {"state": state, "path": directory, "hashed": hashed}
        if bad:
            report[name]["files"] = bad
            worst = max(worst, EXIT_CHECKPOINT_MISMATCH)
    if stop[0]:
        raise Refusal("stopped (SIGTERM/SIGINT)")
    print(json.dumps(report, indent=1, sort_keys=True))
    return worst


def _refuse(exc: Refusal, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"schema": "launch-refusal/1", "reasons": exc.reasons}, ensure_ascii=False))
    for r in exc.reasons:
        print(f"refused: {r}", file=sys.stderr)
    return exc.code


def main(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    environ = dict(os.environ) if environ is None else environ
    ns = parse_args(argv)
    as_json = ns.print_config and ns.json
    # one stop flag for the whole command: installed before anything is read,
    # released only after the last check before exec
    stop = None if ns.print_config else StopFlag()
    try:
        return _main(ns, environ, as_json, stop)
    finally:
        if stop is not None:
            stop.release()


def _main(ns: argparse.Namespace, environ: dict[str, str], as_json: bool,
          stop: StopFlag | None) -> int:
    try:
        if ns.verify_checkpoints:
            return verify_checkpoints(ns, environ, stop)
        doc = build(ns, environ)
    except Refusal as exc:
        return _refuse(exc, as_json)
    if ns.print_config:
        if ns.json:
            data = json.dumps(public(doc), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
            if ns.out:
                atomic_write(ns.out, data.encode("utf-8"))
            else:
                sys.stdout.write(data)
        else:
            print_text(doc, ns.explain)
        return 0
    try:
        if doc["expected"] is not None and not doc["expected"]["functional_match"]:
            raise Refusal(f"this launch differs from the expected launch {ns.expect} "
                          f"(env: {', '.join(doc['expected']['env_differs']) or 'none'}; "
                          f"argv equal: {doc['expected']['argv_equal']})")
        preflight_checkpoints(doc, doc["policy"], stop)
        # from here SIGTERM/SIGINT act as they would without the launcher (a
        # blocked banner write cannot swallow them); a stop that arrived
        # before is refused here
        stop.release()
        if stop[0]:
            raise Refusal("stopped before start (SIGTERM/SIGINT)")
    except Refusal as exc:
        return _refuse(exc, False)
    print(banner(doc), file=sys.stderr, flush=True)
    os.execve(doc["_exe"], doc["argv"], doc["_env"])
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main())
