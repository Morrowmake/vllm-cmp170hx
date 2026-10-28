# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint completeness and identity.

The manifest ``checkpoints/<name>.manifest.json`` (schema
checkpoint-manifest/1) lists every file the server reads from one pinned
revision: config.json, the checkpoint's ``required`` list, the index and
every shard it names (or the single weight file of an indexless
checkpoint), each with its size and sha256.

Completeness (availability): every manifest file present, non-empty and of
the manifest's size, and no ``.incomplete`` blob. Identity: every file's
sha256 equals the manifest's. Hashing is done once per file and recorded in
a stamp (schema checkpoint-stamp/1: path, device, inode, size, mtime_ns,
ctime_ns, sha256 and realpath per file, plus the manifest's sha256); later checks only stat files
against the stamp and never open a checkpoint file. A file the directory's
own stamp does not cover is taken from any stamp in the same stamp directory
that holds the same file (realpath, device, inode, size, mtime_ns and ctime_ns) with the
manifest's sha256, so a directory of links to stamped files, or a new
manifest naming stamped files, hashes only the files no stamp holds. Entries
without device or ctime identity are stale and must be hashed again.

Command line (CPU only; never needs a GPU)::

    python3 -I checkpoint.py verify --settings DIR [--checkpoint NAME] [--path DIR]
    python3 -I checkpoint.py manifest --settings DIR --checkpoint NAME --path DIR [--out FILE]

verify exit codes: 0 verified, 4 incomplete, 5 mismatch, 6 not present,
3 refused or interrupted.
"""

from __future__ import annotations

import json
import os
import signal
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canon import atomic_write, canonical, sha256_bytes, sha256_file  # noqa: E402
from errors import (EXIT_CHECKPOINT_ABSENT, EXIT_CHECKPOINT_INCOMPLETE,  # noqa: E402
                    EXIT_CHECKPOINT_MISMATCH, EXIT_REFUSED, Refusal)
from settings_dir import check_schema  # noqa: E402

STAMP_SCHEMA = "checkpoint-stamp/1"
STAMP_SUFFIX = ".sm80-verified.json"

VERIFIED = "verified"
UNVERIFIED = "unverified"
MISMATCH = "mismatch"
INCOMPLETE = "incomplete"
NOT_PRESENT = "not present"


class Interrupted(Exception):
    pass


def load_manifest(path: str) -> tuple[dict, str]:
    """(manifest, sha256 of its bytes)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise Refusal(f"checkpoint manifest {path}: {exc.strerror}") from None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise Refusal(f"{path}: {exc}") from None
    check_schema(data, "checkpoint-manifest", path)
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise Refusal(f"{path}: no files")
    for rel, ent in files.items():
        if os.path.isabs(rel) or ".." in rel.split("/") or not isinstance(ent, dict) \
                or not isinstance(ent.get("size"), int) or not isinstance(ent.get("sha256"), str):
            raise Refusal(f"{path}: bad entry {rel!r}")
    return data, sha256_bytes(raw)


def completeness(manifest: dict, directory: str) -> list[str]:
    """Problems that make the directory incomplete (stat only)."""
    if not os.path.isdir(directory):
        return [f"{directory}: not present"]
    problems = []
    for rel, exp in sorted(manifest["files"].items()):
        p = os.path.join(directory, rel)
        if not os.path.isfile(p):
            problems.append(f"{rel}: missing")
            continue
        size = os.path.getsize(p)
        if size == 0:
            problems.append(f"{rel}: empty")
        elif size != exp["size"]:
            problems.append(f"{rel}: size {size} != {exp['size']}")
    for dirpath, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".incomplete"):
                rel = os.path.relpath(os.path.join(dirpath, name), directory)
                problems.append(f"{rel}: incomplete download")
    return problems


def stamp_path(stamp_dir: str, directory: str) -> str:
    key = sha256_bytes(os.path.realpath(directory).encode("utf-8"))[:32]
    return os.path.join(stamp_dir, "stamps", key + STAMP_SUFFIX)


def read_stamp(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != STAMP_SCHEMA:
        return None
    return data


def _stat_key(path: str) -> dict:
    st = os.stat(path)
    return {"inode": st.st_ino, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "dev": st.st_dev}


def _own_entries(stamp: dict | None, manifest_sha: str, directory: str) -> dict:
    """The directory's own stamp entries, if the stamp is for this directory
    and manifest."""
    if stamp is None or stamp.get("manifest_sha256") != manifest_sha \
            or stamp.get("path") != os.path.realpath(directory):
        return {}
    files = stamp.get("files")
    return dict(files) if isinstance(files, dict) else {}


def _same_file(ent, key: dict, real: str) -> bool:
    return isinstance(ent, dict) and all(ent.get(k) == v for k, v in key.items()) \
        and ent.get("realpath", real) == real


def stamped_files(stamp_file: str) -> dict[str, list[dict]]:
    """Every file entry of every stamp in the directory of ``stamp_file``, by
    the file's realpath. An entry without a realpath (written before stamps
    recorded it) is keyed by its stamp's directory joined with its name.
    Reads stamps only, never a checkpoint file."""
    out: dict[str, list[dict]] = {}
    folder = os.path.dirname(stamp_file)
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return out
    for name in names:
        stamp = read_stamp(os.path.join(folder, name)) if name.endswith(STAMP_SUFFIX) else None
        files = stamp.get("files") if stamp else None
        if not isinstance(files, dict):
            continue
        base = stamp.get("path")
        for rel, ent in files.items():
            if not isinstance(ent, dict):
                continue
            real = ent.get("realpath")
            if not isinstance(real, str) and isinstance(base, str):
                real = os.path.realpath(os.path.join(base, rel))
            if isinstance(real, str):
                out.setdefault(real, []).append(ent)
    return out


def _reusable(pool: dict, real: str, key: dict, sha: str) -> dict | None:
    """A stamped entry of the same file whose sha256 is ``sha``, as an entry
    for this stamp; None if no stamp holds one."""
    for ent in pool.get(real, ()):
        if _same_file(ent, key, real) and ent.get("sha256") == sha:
            return dict(key, sha256=sha, realpath=real)
    return None


def identity(manifest: dict, manifest_sha: str, directory: str,
             stamp_file: str) -> tuple[str, list[str], list[str]]:
    """Stat-only check against the directory's stamp, and for a file it does
    not cover against every stamp beside it: (state, stale files, mismatched
    files)."""
    entries = _own_entries(read_stamp(stamp_file), manifest_sha, directory)
    pool = None
    stale, bad = [], []
    for rel in sorted(manifest["files"]):
        p = os.path.join(directory, rel)
        try:
            key = _stat_key(p)
        except OSError:
            stale.append(rel)
            continue
        real = os.path.realpath(p)
        want = manifest["files"][rel]["sha256"]
        ent = entries.get(rel)
        if _same_file(ent, key, real):
            if ent.get("sha256") != want:
                bad.append(rel)
            continue
        if pool is None:
            pool = stamped_files(stamp_file)
        if _reusable(pool, real, key, want) is None:
            stale.append(rel)
    if bad:
        return MISMATCH, stale, bad
    if stale:
        return UNVERIFIED, stale, []
    return VERIFIED, [], []


def verify(manifest: dict, manifest_sha: str, directory: str, stamp_file: str, *,
           stop: list | None = None) -> tuple[str, list[str], int]:
    """Hash every file no stamp covers (the directory's own stamp, or for a
    file it does not cover any stamp beside it holding the same file with
    the manifest's sha256), then publish the stamp.

    Returns (state, mismatched files, files hashed). A stop request
    between or during files, or before the stamp is published, raises
    Interrupted and writes no stamp."""
    entries = _own_entries(read_stamp(stamp_file), manifest_sha, directory)
    pool = None
    bad = []
    hashed = 0
    for rel in sorted(manifest["files"]):
        if stop and stop[0]:
            raise Interrupted()
        p = os.path.join(directory, rel)
        key = _stat_key(p)
        real = os.path.realpath(p)
        want = manifest["files"][rel]["sha256"]
        ent = entries.get(rel)
        if not _same_file(ent, key, real):
            if pool is None:
                pool = stamped_files(stamp_file)
            ent = _reusable(pool, real, key, want)
        if ent is None:
            digest = sha256_file(p)
            hashed += 1
            if stop and stop[0]:
                raise Interrupted()
            ent = dict(key, sha256=digest)
        entries[rel] = ent = dict(ent, realpath=real)
        if ent["sha256"] != want:
            bad.append(rel)
    entries = {k: v for k, v in entries.items() if k in manifest["files"]}
    stamp = {"schema": STAMP_SCHEMA, "path": os.path.realpath(directory),
             "manifest_sha256": manifest_sha, "files": entries}
    if stop and stop[0]:
        raise Interrupted()
    atomic_write(stamp_file, canonical(stamp))
    return (MISMATCH if bad else VERIFIED), bad, hashed


def needed_files(spec: dict, directory: str) -> list[str]:
    """The files the server reads, from the checkpoint spec and its index."""
    sidecar = spec.get("kind") == "sidecar"
    need = [] if sidecar else ["config.json"]
    need += [r for r in spec.get("required", []) if r not in need]
    index = spec.get("index")
    if index:
        if index not in need:
            need.append(index)
        with open(os.path.join(directory, index), encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        need += sorted(set(weight_map.values()) - set(need))
    else:
        for w in spec.get("weights", [] if sidecar else ["model.safetensors"]):
            if w not in need:
                need.append(w)
    return need


def make_manifest(repo: str, revision: str, spec: dict, directory: str) -> dict:
    files = {}
    for rel in needed_files(spec, directory):
        p = os.path.join(directory, rel)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            raise Refusal(f"{directory}: {rel} missing or empty; cannot write a manifest")
        files[rel] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    return {"schema": "checkpoint-manifest/1", "repo": repo, "revision": revision,
            "files": dict(sorted(files.items()))}


def status(manifest: dict, manifest_sha: str, directory: str, stamp_dir: str) -> dict:
    """Completeness plus stat-only identity; never opens a checkpoint file."""
    out = {"path": directory, "manifest_sha256": manifest_sha}
    if not os.path.isdir(directory):
        out.update(state=NOT_PRESENT, identity=UNVERIFIED)
        return out
    problems = completeness(manifest, directory)
    if problems:
        out.update(state=INCOMPLETE, identity=UNVERIFIED, files=problems)
        return out
    state, stale, bad = identity(manifest, manifest_sha, directory, stamp_path(stamp_dir, directory))
    out.update(state="complete", identity=state)
    if bad:
        out["files"] = bad
    elif stale:
        out["stale"] = stale
    return out


# -- command line --------------------------------------------------------------

class StopFlag(list):
    """``[requested]``, set by SIGTERM/SIGINT while installed. One flag lives
    for a whole command, so a stop is never forgotten; ``release`` gives
    both signals back the dispositions they had before, after which a
    signal acts as it would without the flag."""

    SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self) -> None:
        super().__init__([False])
        self._old = {}
        for sig in self.SIGNALS:
            old = signal.signal(sig, self._set)
            self._old[sig] = signal.SIG_DFL if old is None else old

    def _set(self, _sig, _frm) -> None:
        self[0] = True

    def release(self) -> None:
        old, self._old = self._old, {}
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _settings(path: str):
    from model import load_model_settings
    from settings_dir import SettingsDir
    sd = SettingsDir(path)
    return sd, load_model_settings(sd)


def _cli_verify(args) -> int:
    stop = StopFlag()
    try:
        return _cli_verify_all(args, stop)
    finally:
        stop.release()


def _cli_verify_all(args, stop: StopFlag) -> int:
    sd, ms = _settings(args.settings)
    names = [args.checkpoint] if args.checkpoint else sorted(ms.checkpoints)
    if args.path and len(names) != 1:
        raise Refusal("--path needs --checkpoint when the model has several checkpoints")
    home = os.path.expanduser("~")
    stamp_dir = args.stamp_dir or os.path.join(home, ".cache", "sm80-launch")
    models_dir = args.models_dir or os.path.join(home, "models")
    worst = 0
    report = {}
    for name in names:
        spec = ms.checkpoints.get(name)
        if spec is None:
            raise Refusal(f"no checkpoint {name!r} in model.toml")
        manifest, msha = load_manifest(sd.path(spec["manifest"]))
        directory = args.path or os.path.join(models_dir, spec["dir"])
        if not os.path.isdir(directory):
            report[name] = {"state": NOT_PRESENT, "path": directory}
            worst = max(worst, EXIT_CHECKPOINT_ABSENT)
            continue
        problems = completeness(manifest, directory)
        if problems:
            report[name] = {"state": INCOMPLETE, "path": directory, "files": problems}
            worst = max(worst, EXIT_CHECKPOINT_INCOMPLETE)
            continue
        try:
            state, bad, hashed = verify(manifest, msha, directory,
                                        stamp_path(stamp_dir, directory), stop=stop)
        except Interrupted:
            print(json.dumps({"interrupted": name}), file=sys.stderr)
            return EXIT_REFUSED
        report[name] = {"state": state, "path": directory, "manifest_sha256": msha,
                        "hashed": hashed}
        if bad:
            report[name]["files"] = bad
            worst = max(worst, EXIT_CHECKPOINT_MISMATCH)
    print(json.dumps(report, indent=1, sort_keys=True))
    return worst


def _cli_manifest(args) -> int:
    sd, ms = _settings(args.settings)
    spec = ms.checkpoints.get(args.checkpoint)
    if spec is None:
        raise Refusal(f"no checkpoint {args.checkpoint!r} in model.toml")
    pins = sd.pins()
    repo, revision = pins.get(spec["repo_key"]), pins.get(spec["revision_key"])
    if not repo or not revision:
        raise Refusal(f"pins.env lacks {spec['repo_key']} or {spec['revision_key']}")
    data = make_manifest(repo, revision, spec, args.path)
    out = args.out or sd.path(spec["manifest"])
    atomic_write(out, json.dumps(data, indent=1, sort_keys=True).encode() + b"\n")
    print(f"{len(data['files'])} files -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="checkpoint.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--settings", required=True)
    v.add_argument("--checkpoint")
    v.add_argument("--path")
    v.add_argument("--models-dir")
    v.add_argument("--stamp-dir")
    m = sub.add_parser("manifest")
    m.add_argument("--settings", required=True)
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--path", required=True)
    m.add_argument("--out")
    args = p.parse_args(argv)
    try:
        return _cli_verify(args) if args.cmd == "verify" else _cli_manifest(args)
    except Refusal as exc:
        for r in exc.reasons:
            print(f"refused: {r}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
