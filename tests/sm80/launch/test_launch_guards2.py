# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""More guards: option injection through excluded values, the served
logical checkpoint, the linked-tree source contract, raw extra arguments,
missing build identity for candidates, secret knob targets, knob names,
linked required artefacts and stop requests before exec."""

from __future__ import annotations

import json
import os
import shutil
import signal

import pytest

from helpers import (CHECKPOINT_FILES, World, add_build, git, head, make_checkpoint, promote, write,
                     write_stamp)
from treestate import MANIFEST, STAMP, build_owned_sha256


@pytest.fixture
def w(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    promote(w, ["tp2"])
    assert w.resolve()["validated"]["state"] == "VALIDATED"
    return w


def reasons(d):
    return d["validated"]["reasons"]


def overlay(tmp_path, text, name="ov.toml"):
    return write(str(tmp_path / name), 'schema = "launch-overlay/1"\n' + text)


# -- excluded values cannot inject options -----------------------------------------------

@pytest.mark.parametrize("item", ['args.port=[8123,"--dtype","float32"]',
                                  'args.host=["0.0.0.0","--dtype","float32"]',
                                  'args.enable-prompt-tokens-details=["--dtype","float32"]',
                                  'args.served-model-name="--dtype"'])
def test_values_cannot_inject_options(w, tmp_path, item):
    w.refused("--set", item)
    frozen = tmp_path / "e.json"
    frozen.write_text(json.dumps(w.resolve()))
    assert w.run("--set", item, "--expect", str(frozen)).returncode == 3


def test_harness_overlay_value_shape(w, tmp_path):
    bad = overlay(tmp_path, 'class = "harness"\n[args]\nenable-prompt-tokens-details = ["--dtype", "float32"]\n')
    w.refused("--overlay", bad)
    ok = overlay(tmp_path, 'class = "harness"\n[args]\nprofiler-config = { profiler = "torch" }\n', "ok.toml")
    d = w.resolve("--overlay", ok)
    assert d["validated"]["state"] == "VALIDATED" and d["harness"][0] == "--profiler-config"


def test_scalar_location_change_allowed(w):
    d = w.resolve("--set", "args.port=9000", "--set", "args.host=0.0.0.0")
    assert d["validated"]["state"] == "VALIDATED"


# -- the served logical checkpoint is functional -----------------------------------------

def two_checkpoints(w, tmp_path):
    files = dict(CHECKPOINT_FILES, **{"config.json": CHECKPOINT_FILES["config.json"].replace("synth", "alter")})
    make_checkpoint(os.path.join(w.models, "alternate"), files)
    model = open(os.path.join(w.settings, "model.toml")).read()
    model = model.replace("[args]\n", '[checkpoint.alternate]\nrepo_key = "ALT_REPO"\n'
                          'revision_key = "ALT_REVISION"\ndir = "alternate"\n'
                          'index = "model.safetensors.index.json"\nrequired = ["config.json"]\n'
                          'manifest = "checkpoints/alternate.manifest.json"\n\n[args]\n', 1)
    write(os.path.join(w.settings, "model.toml"), model)
    pins = open(os.path.join(w.settings, "pins.env")).read() + "ALT_REPO=example/alt\nALT_REVISION=" + "1" * 40 + "\n"
    write(os.path.join(w.settings, "pins.env"), pins)
    import checkpoint as ck
    from model import load_model_settings
    from settings_dir import SettingsDir
    spec = load_model_settings(SettingsDir(w.settings)).checkpoints["alternate"]
    m = ck.make_manifest("example/alt", "1" * 40, spec, os.path.join(w.models, "alternate"))
    write(os.path.join(w.settings, "checkpoints", "alternate.manifest.json"), json.dumps(m))
    write(os.path.join(w.settings, "variants", "alt.toml"),
          'schema = "launch-variant/1"\n[args]\nmodel = "${checkpoint.alternate.path}"\n')
    assert w.verify() == 0
    promote(w, ["tp2"])


def test_switching_logical_checkpoint_breaks_validation(w, tmp_path):
    two_checkpoints(w, tmp_path)
    base = w.resolve()
    assert base["validated"]["state"] == "VALIDATED" and base["served_checkpoint"] == "target"
    d = w.resolve("--variant", "alt")
    assert d["served_checkpoint"] == "alternate"
    assert d["argv_functional"][0] == "${checkpoint.alternate.path}"
    assert "functional ≠ validated" in reasons(d)
    frozen = tmp_path / "e.json"
    frozen.write_text(json.dumps(base))
    assert w.resolve("--variant", "alt", "--expect", str(frozen))["expected"]["functional_match"] is False
    same = str(tmp_path / "copy")
    shutil.copytree(w.cp, same)
    import checkpoint as ck
    assert ck.main(["verify", "--settings", w.settings, "--checkpoint", "target", "--path", same,
                    "--stamp-dir", w.stamps]) == 0
    assert w.resolve("--checkpoint-path", f"target={same}")["validated"]["state"] == "VALIDATED"


# -- the linked-tree source contract --------------------------------------------------------

def linked(tmp_path):
    w = World(tmp_path, build=False)
    src = os.path.join(w.tmp, "src")
    shutil.copytree(w.tree, src, symlinks=True)
    src_man = add_build(src, {"vllm/_C.abi3.so": "native-lib", "vllm/_D.abi3.so": "d"}, stamp=False)
    for rel in ("vllm/_C.abi3.so", "vllm/_D.abi3.so"):
        os.symlink(os.path.join(src, rel), os.path.join(w.tree, rel))
    owned = {rel: src_man["files"][rel] for rel in ("vllm/_C.abi3.so", "vllm/_D.abi3.so")}
    man = {"schema": "build-owned/1", "source": "linked", "commit": head(w.tree), "files": owned,
           "native_manifest_sha256": src_man["native_manifest_sha256"],
           "native_manifest_path": src_man["native_manifest_path"],
           "build_owned_sha256": build_owned_sha256(owned),
           "linked_from": {"tree": src, "commit": head(src), "build_owned_sha256": src_man["build_owned_sha256"]}}
    write(os.path.join(w.tree, MANIFEST), json.dumps(man))
    write_stamp(w.tree, man)
    assert w.verify() == 0
    return w, src, man, src_man


def test_linked_clean_and_required_link(tmp_path):
    w, _, _, _ = linked(tmp_path)
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model.replace("[args]\n", '[requires]\nfiles = ["vllm/_C*.so"]\n\n[args]\n', 1))
    d = w.resolve()
    assert not d["engine_target"]["dirty"], d["engine_target"]["dirty_paths"]
    assert list(d["requires"]) == ["vllm/_C.abi3.so"]


def metadata(w):
    return " ".join(w.resolve()["engine_target"]["dirty_paths"].get("metadata_invalid", []))


def test_linked_source_record_missing(tmp_path):
    w, _, _, src_man = linked(tmp_path)
    os.unlink(src_man["build_record"]["path"])
    assert "build record" in metadata(w)


def test_linked_source_map_hash_wrong(tmp_path):
    w, src, _, src_man = linked(tmp_path)
    src_man["build_owned_sha256"] = "0" * 64
    write(os.path.join(src, MANIFEST), json.dumps(src_man))
    assert "build_owned_sha256 does not match its map" in metadata(w)


def test_linked_commit_disagrees(tmp_path):
    w, _, man, _ = linked(tmp_path)
    man["linked_from"]["commit"] = "1" * 40
    write(os.path.join(w.tree, MANIFEST), json.dumps(man))
    assert "source manifest changed since the link" in metadata(w)


def test_linked_missing_source_entry(tmp_path):
    w, _, man, _ = linked(tmp_path)
    os.unlink(os.path.join(w.tree, "vllm/_D.abi3.so"))
    del man["files"]["vllm/_D.abi3.so"]
    man["build_owned_sha256"] = build_owned_sha256(man["files"])
    write(os.path.join(w.tree, MANIFEST), json.dumps(man))
    assert "not linked" in metadata(w)


def test_required_dangling_link(tmp_path):
    w, src, _, _ = linked(tmp_path)
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model.replace("[args]\n", '[requires]\nfiles = ["vllm/_C*.so"]\n\n[args]\n', 1))
    os.unlink(os.path.join(src, "vllm/_C.abi3.so"))
    assert "cannot be read" in w.refused()


# -- raw extra arguments -------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [["-tp", "1"], ["-pp", "2"], ["--gpu-memory-utilization", "0.5"],
                                    ["--gpu-memory-utilization=0.5"]])
def test_extra_args_channels(w, tmp_path, tokens):
    w.refused("--", *tokens)
    env = write(str(tmp_path / "u.env"), "EXTRA_ARGS=" + " ".join(tokens) + "\n")
    w.refused("--env-file", env)


def test_unprotected_extra_allowed(w):
    d = w.resolve("--", "--enable-log-requests")
    assert d["argv"][-1] == "--enable-log-requests"
    w.resolve("--experiment", "e", "--", "--gpu-memory-utilization", "0.5")


# -- candidates without build identity --------------------------------------------------------

def test_candidate_without_native_identity(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    m = json.load(open(os.path.join(w.tree, MANIFEST)))
    m["native_manifest_sha256"] = None
    m.pop("native_manifest_path")
    write(os.path.join(w.tree, MANIFEST), json.dumps(m))
    s = json.load(open(os.path.join(w.tree, STAMP)))
    s["native_manifest_sha256"] = None
    write(os.path.join(w.tree, STAMP), json.dumps(s))
    d = w.resolve("--role", "candidate")
    assert not d["engine_target"]["dirty"]
    assert set(reasons(d)) == {"no receipt", "no build identity"}
    full = World(tmp_path / "full")
    assert full.verify() == 0
    assert reasons(full.resolve("--role", "candidate")) == ["no receipt"]


# -- secret knob targets and knob names --------------------------------------------------------

@pytest.mark.parametrize("target", ["env.VLLM_KERNEL_X", "env.SYNTH_SECRET", "env.CUDA_HOME"])
def test_secret_knob_cannot_hide_function(w, target):
    model = open(os.path.join(w.settings, "model.toml")).read() + \
        f'\n[user.KX]\ntarget = "{target}"\nsecret = true\n'
    write(os.path.join(w.settings, "model.toml"), model)
    assert "secret targets must be" in w.refused()


@pytest.mark.parametrize("knob", ["VLLM_KERNEL_X", "PYTHONPATH", "HF_TOKEN", "HOME"])
def test_knob_names_apart_from_variables(w, knob):
    model = open(os.path.join(w.settings, "model.toml")).read() + \
        f'\n[user.{knob}]\ntarget = "env.VLLM_PLACEMENT"\n'
    write(os.path.join(w.settings, "model.toml"), model)
    assert "name knobs apart" in w.refused()


# -- a stop request never becomes a start -----------------------------------------------------

@pytest.mark.parametrize("mode", ["stamp", "boot"])
def test_stop_before_exec(w, mode):
    import main
    from checkpoint import StopFlag
    from errors import Refusal
    doc = main.build(main.parse_args(w.base_args("--checkpoint-verify", mode)), w.env())
    old = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        flag = StopFlag()
        os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(Refusal, match="stopped before start"):
            main.preflight_checkpoints(doc, "strict", flag)
    finally:
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])


def test_stop_during_boot_verify_publishes_no_stamp(tmp_path):
    import checkpoint as ck
    w = World(tmp_path)
    m, sha = ck.load_manifest(os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json"))
    sfile = ck.stamp_path(w.stamps, w.cp)
    calls = {"n": 0}

    class LateStop(list):
        def __getitem__(self, i):
            calls["n"] += 1
            return calls["n"] > len(m["files"]) * 2   # set only after the last file
    with pytest.raises(ck.Interrupted):
        ck.verify(m, sha, w.cp, sfile, stop=LateStop([False]))
    assert not os.path.exists(sfile)
    _ = git
