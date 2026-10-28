# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint completeness, identity, stamps and the verify command."""

from __future__ import annotations

import builtins
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

import checkpoint as ck
import main
from helpers import CHECKPOINT_FILES, LAUNCH, World, make_checkpoint, promote, write
from model import load_model_settings
from settings_dir import SettingsDir


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


class OpenCounter:
    """Counts opens of files under one directory (builtins.open and os.open)."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        self.hits: list[str] = []

    def __enter__(self):
        self._open, self._osopen = builtins.open, os.open
        counter = self

        def bopen(file, *a, **k):
            counter._see(file)
            return counter._open(file, *a, **k)

        def oopen(path, *a, **k):
            counter._see(path)
            return counter._osopen(path, *a, **k)
        builtins.open, os.open = bopen, oopen
        return self

    def _see(self, path) -> None:
        if isinstance(path, (str, bytes, os.PathLike)):
            p = os.path.realpath(os.fsdecode(path))
            if p.startswith(self.root + os.sep):
                self.hits.append(p)

    def __exit__(self, *exc):
        builtins.open, os.open = self._open, self._osopen


def verify_cli(w, *extra) -> tuple[int, dict]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = ck.main(["verify", "--settings", w.settings, "--models-dir", w.models,
                        "--stamp-dir", w.stamps, *extra])
    return code, json.loads(out.getvalue() or "{}")


def same_size_edit(path: str) -> None:
    data = open(path).read()
    write(path, data[:-1] + ("X" if data[-1] != "X" else "Y"))


def test_print_config_unstamped_reads_no_checkpoint_file(w):
    with OpenCounter(w.cp) as c:
        d = w.resolve()
    assert d["checkpoint"]["target"]["identity"] == "unverified"
    assert "checkpoint not verified" in d["validated"]["reasons"]
    assert c.hits == []


def test_verify_then_stamp_boot_hashes_nothing(w):
    code, rep = verify_cli(w)
    assert code == 0 and rep["target"]["state"] == "verified" and rep["target"]["hashed"] == 8
    import main
    doc = main.build(main.parse_args(w.base_args()), w.env())
    with OpenCounter(w.cp) as c:
        main.preflight_checkpoints(doc, "strict")
    assert c.hits == []
    assert w.resolve()["checkpoint"]["target"]["identity"] == "verified"


def test_verify_without_gpu(w):
    env = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": w.home}
    p = subprocess.run([sys.executable, "-I", os.path.join(LAUNCH, "checkpoint.py"), "verify",
                        "--settings", w.settings, "--models-dir", w.models, "--stamp-dir", w.stamps],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["target"]["state"] == "verified"


def test_mismatch_and_incomplete_exits(w):
    path = os.path.join(w.cp, "tokenizer.json")
    original = open(path).read()
    write(path, original + " ")
    code, rep = verify_cli(w)
    assert code == ck.EXIT_CHECKPOINT_INCOMPLETE and "tokenizer.json: size" in rep["target"]["files"][0]
    write(path, original)
    same_size_edit(path)
    code, rep = verify_cli(w)
    assert code == ck.EXIT_CHECKPOINT_MISMATCH and rep["target"]["files"] == ["tokenizer.json"]


@pytest.mark.parametrize("rel", ["chat_template.jinja", "tokenizer.json"])
def test_altered_file_is_mismatch(w, rel):
    assert verify_cli(w)[0] == 0
    same_size_edit(os.path.join(w.cp, rel))
    d = w.resolve()
    assert d["checkpoint"]["target"]["identity"] == "unverified"       # stat changed
    code, rep = verify_cli(w)
    assert code == ck.EXIT_CHECKPOINT_MISMATCH and rep["target"]["files"] == [rel]
    d = w.resolve()
    assert d["checkpoint"]["target"]["identity"] == "mismatch"
    p = w.run()
    assert p.returncode == 3 and "differs from its manifest" in p.stderr
    p = w.run("--policy", "merge")
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("case", ["empty", "truncated", "missing_tokenizer", "incomplete_blob"])
def test_incomplete_refused_every_mode(w, case):
    if case == "empty":
        write(os.path.join(w.cp, "model-00001-of-00002.safetensors"), "")
    elif case == "truncated":
        write(os.path.join(w.cp, "model-00001-of-00002.safetensors"), "A" * 10)
    elif case == "missing_tokenizer":
        os.unlink(os.path.join(w.cp, "tokenizer.json"))
    else:
        write(os.path.join(w.cp, "blob.incomplete"), "x")
    assert verify_cli(w)[0] == ck.EXIT_CHECKPOINT_INCOMPLETE
    for policy, mode in (("strict", "stamp"), ("merge", "boot")):
        p = w.run("--policy", policy, "--checkpoint-verify", mode)
        assert p.returncode == 3 and "not complete" in p.stderr
    d = w.resolve()
    assert d["checkpoint"]["target"]["state"] == "incomplete"
    assert "checkpoint not present" in d["validated"]["reasons"]


def test_other_revision_is_mismatch(w, tmp_path):
    files = dict(CHECKPOINT_FILES)
    files["config.json"] = files["config.json"].replace("synth", "SYNTH")
    other = make_checkpoint(str(tmp_path / "other"), files)
    code, rep = verify_cli(w, "--checkpoint", "target", "--path", other)
    assert code == ck.EXIT_CHECKPOINT_MISMATCH and rep["target"]["files"] == ["config.json"]
    d = w.resolve("--checkpoint-path", f"target={other}")
    assert d["checkpoint"]["target"]["identity"] == "mismatch"
    assert "checkpoint not verified" in d["validated"]["reasons"]
    assert w.run("--checkpoint-path", f"target={other}").returncode == 3


def test_not_present(w, tmp_path):
    code, _ = verify_cli(w, "--checkpoint", "target", "--path", str(tmp_path / "nope"))
    assert code == ck.EXIT_CHECKPOINT_ABSENT
    d = w.resolve("--checkpoint-path", f"target={tmp_path / 'nope'}")
    assert d["checkpoint"]["target"]["state"] == "not present"
    assert "checkpoint not present" in d["validated"]["reasons"]


def test_indexless_single_file(tmp_path):
    d = make_checkpoint(str(tmp_path / "drafter"), {"config.json": "{}", "model.safetensors": "W" * 32})
    os.unlink(os.path.join(d, "model.safetensors.index.json"))
    m = ck.make_manifest("r", "v", {"required": ["config.json", "model.safetensors"]}, d)
    assert sorted(m["files"]) == ["config.json", "model.safetensors"]
    sha = ck.sha256_bytes(json.dumps(m).encode())
    state, bad, _ = ck.verify(m, sha, d, str(tmp_path / "s.json"))
    assert state == ck.VERIFIED and bad == []


def test_stale_stamp_policies(w):
    assert verify_cli(w)[0] == 0
    shard = os.path.join(w.cp, "model-00002-of-00002.safetensors")
    os.utime(shard, ns=(1, 1))
    p = w.run()
    assert p.returncode == 3 and "checkpoint.py verify" in p.stderr
    import main
    doc = main.build(main.parse_args(w.base_args("--checkpoint-verify", "boot")), w.env())
    with OpenCounter(w.cp) as c:
        main.preflight_checkpoints(doc, "strict")
    assert c.hits == [os.path.realpath(shard)]


def test_interrupt_writes_no_stamp(w):
    m, sha = ck.load_manifest(os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json"))
    sfile = ck.stamp_path(w.stamps, w.cp)
    with pytest.raises(ck.Interrupted):
        ck.verify(m, sha, w.cp, sfile, stop=[True])
    assert not os.path.exists(sfile)


def test_new_pin_with_verified_weights_fails_receipt(w, tmp_path):
    assert verify_cli(w)[0] == 0
    promote(w, ["tp2"])
    assert w.resolve()["validated"]["state"] == "VALIDATED"
    files = dict(CHECKPOINT_FILES)
    files["config.json"] = files["config.json"].replace("synth", "rev02")
    other = make_checkpoint(str(tmp_path / "rev2"), files)
    pins = open(os.path.join(w.settings, "pins.env")).read().replace(
        "TARGET_REVISION=0123", "TARGET_REVISION=9123")
    write(os.path.join(w.settings, "pins.env"), pins)
    spec = load_model_settings(SettingsDir(w.settings)).checkpoints["target"]
    revision = pins.split("TARGET_REVISION=")[1].strip()
    m = ck.make_manifest("example/synth-dense-1b", revision, spec, other)
    write(os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json"),
          json.dumps(m, indent=1, sort_keys=True) + "\n")
    assert verify_cli(w, "--checkpoint", "target", "--path", other)[0] == 0
    d = w.resolve("--checkpoint-path", f"target={other}")
    assert d["checkpoint"]["target"]["identity"] == "verified"
    # TARGET_REVISION is a receipt pin, so criterion 2 fires beside 4a
    assert set(d["validated"]["reasons"]) == {"pin ≠ receipt", "checkpoint ≠ receipt"}


# -- a launch checks only the checkpoints it reads ------------------------------

def add_role(w, name: str, dirname: str) -> None:
    """A second checkpoint role of the model (same manifest as the target)."""
    with open(os.path.join(w.settings, "model.toml"), "a") as f:
        f.write(f'\n[checkpoint.{name}]\nrepo_key = "TARGET_REPO"\nrevision_key = "TARGET_REVISION"\n'
                f'dir = "{dirname}"\nindex = "model.safetensors.index.json"\n'
                f'manifest = "checkpoints/synth-dense-1b.manifest.json"\n')


def test_unread_role_with_missing_directory_is_not_checked(w):
    # a model whose settings carry a second served role (a variant serves it)
    # that is not downloaded: the target launch neither refuses nor is marked
    add_role(w, "hybrid", "synth-hybrid")
    write(os.path.join(w.settings, "variants", "hybrid.toml"),
          'schema = "launch-variant/1"\n[args]\nmodel = "${checkpoint.hybrid.path}"\n')
    assert verify_cli(w, "--checkpoint", "target")[0] == 0
    d = w.resolve()
    assert d["served_checkpoint"] == "target" and sorted(d["checkpoint"]) == ["target"]
    assert d["checkpoint"]["target"]["identity"] == "verified"
    assert not {"checkpoint not present", "checkpoint not verified"} & set(d["validated"]["reasons"])
    doc = main.build(main.parse_args(w.base_args()), w.env())
    assert list(doc["_cp_specs"]) == ["target"]
    main.preflight_checkpoints(doc, "strict")
    p = w.run()
    assert p.returncode == 0, p.stderr
    # the launch that serves the missing role is the one refused
    d = w.resolve("--variant", "hybrid")
    assert d["served_checkpoint"] == "hybrid" and sorted(d["checkpoint"]) == ["hybrid"]
    assert d["checkpoint"]["hybrid"]["state"] == "not present"
    p = w.run("--variant", "hybrid")
    assert p.returncode == 3 and "checkpoint hybrid" in p.stderr and "not complete" in p.stderr


def test_role_named_in_a_nested_argument_is_checked(w, tmp_path):
    # a checkpoint the server reads beside the served one (a table directory
    # named inside a dict argument) is checked with it
    add_role(w, "tables", "synth-tables")
    write(os.path.join(w.settings, "variants", "tables.toml"),
          'schema = "launch-variant/1"\n[args.hf-overrides]\n'
          'table_dir = "${checkpoint.tables.path}"\n')
    assert verify_cli(w, "--checkpoint", "target")[0] == 0
    d = w.resolve("--variant", "tables")
    assert d["served_checkpoint"] == "target" and sorted(d["checkpoint"]) == ["tables", "target"]
    assert d["checkpoint"]["tables"]["state"] == "not present"
    assert "checkpoint not present" in d["validated"]["reasons"]
    p = w.run("--variant", "tables")
    assert p.returncode == 3 and "checkpoint tables" in p.stderr
    shutil.copytree(w.cp, os.path.join(w.models, "synth-tables"))
    out = io.StringIO()
    with redirect_stdout(out):
        code = main.main(w.base_args("--variant", "tables", "--verify-checkpoints"), w.env())
    assert code == 0 and sorted(json.loads(out.getvalue())) == ["tables", "target"]
    assert w.run("--variant", "tables").returncode == 0


# -- stamps are reused across directories ----------------------------------------

def overlay(w, tmp_path, fresh=("config.json",)) -> str:
    """A directory of links to the target's files; ``fresh`` files are copies."""
    out = str(tmp_path / "overlay")
    os.makedirs(out)
    for rel in sorted(os.listdir(w.cp)):
        if rel in fresh:
            shutil.copy(os.path.join(w.cp, rel), os.path.join(out, rel))
        else:
            os.symlink(os.path.join(w.cp, rel), os.path.join(out, rel))
    return out


def manifest(w):
    return ck.load_manifest(os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json"))


def test_overlay_of_links_to_stamped_files_hashes_only_new_files(w, tmp_path):
    assert verify_cli(w)[0] == 0
    ov = overlay(w, tmp_path)
    with OpenCounter(w.cp) as c:
        code, rep = verify_cli(w, "--checkpoint", "target", "--path", ov)
    assert code == 0 and rep["target"]["state"] == "verified" and rep["target"]["hashed"] == 1
    assert c.hits == []
    stamp = ck.read_stamp(ck.stamp_path(w.stamps, ov))
    real = {rel: e["realpath"] for rel, e in stamp["files"].items()}
    assert real["tokenizer.json"] == os.path.realpath(os.path.join(w.cp, "tokenizer.json"))
    assert real["config.json"] == os.path.realpath(os.path.join(ov, "config.json"))
    assert w.resolve("--checkpoint-path", f"target={ov}")["checkpoint"]["target"]["identity"] == "verified"


def test_overlay_of_links_only_is_verified_without_its_own_stamp(w, tmp_path):
    assert verify_cli(w)[0] == 0
    ov = overlay(w, tmp_path, fresh=())
    with OpenCounter(w.cp) as c:
        d = w.resolve("--checkpoint-path", f"target={ov}")
        doc = main.build(main.parse_args(w.base_args("--checkpoint-path", f"target={ov}")), w.env())
        main.preflight_checkpoints(doc, "strict")
    assert c.hits == []
    assert d["checkpoint"]["target"]["identity"] == "verified"
    assert not os.path.exists(ck.stamp_path(w.stamps, ov))


def test_stamp_entries_without_realpath_are_reused(w, tmp_path):
    assert verify_cli(w)[0] == 0
    sfile = ck.stamp_path(w.stamps, w.cp)
    stamp = ck.read_stamp(sfile)
    assert all("realpath" in e for e in stamp["files"].values())
    for e in stamp["files"].values():
        del e["realpath"]
    write(sfile, json.dumps(stamp))
    ov = overlay(w, tmp_path)
    code, rep = verify_cli(w, "--checkpoint", "target", "--path", ov)
    assert code == 0 and rep["target"]["hashed"] == 1


@pytest.mark.parametrize("change", ["touched", "other_sha256", "other_file"])
def test_stamp_entry_of_another_file_or_hash_is_not_reused(w, tmp_path, change):
    assert verify_cli(w)[0] == 0
    shard = "model-00001-of-00002.safetensors"
    ov = overlay(w, tmp_path, fresh=("config.json", shard) if change == "other_file" else ("config.json",))
    if change == "touched":
        os.utime(os.path.join(w.cp, shard), ns=(1, 1))
    elif change == "other_sha256":
        sfile = ck.stamp_path(w.stamps, w.cp)
        stamp = ck.read_stamp(sfile)
        stamp["files"][shard]["sha256"] = "0" * 64
        write(sfile, json.dumps(stamp))
    m, sha = manifest(w)
    with OpenCounter(w.cp) as c:
        state, bad, hashed = ck.verify(m, sha, ov, ck.stamp_path(w.stamps, ov))
    assert state == ck.VERIFIED and bad == [] and hashed == 2
    assert c.hits == ([] if change == "other_file" else [os.path.realpath(os.path.join(w.cp, shard))])


def test_reused_entry_with_a_new_manifest(w):
    # a new manifest for the same directory (another file list) hashes nothing
    # the old stamp holds with the same sha256
    assert verify_cli(w)[0] == 0
    path = os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json")
    data = json.load(open(path))
    data["files"].pop("chat_template.jinja")
    write(path, json.dumps(data, indent=1, sort_keys=True) + "\n")
    m, sha = manifest(w)
    with OpenCounter(w.cp) as c:
        state, bad, hashed = ck.verify(m, sha, w.cp, ck.stamp_path(w.stamps, w.cp))
    assert state == ck.VERIFIED and hashed == 0 and c.hits == []
