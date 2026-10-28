# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-bearing fields and stat identities cannot bypass manifests."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import checkpoint as ck
import main
from helpers import World, write
from test_launch_checkpoint import add_role, manifest, overlay, same_size_edit, verify_cli


@pytest.fixture
def w(tmp_path):
    return World(tmp_path, upstream=["VLLM_PORT", "VLLM_API_KEY", "VLLM_EXL3_OVERRIDE"])


def test_preserved_mtime_hardlink_write_invalidates_stamps(w, tmp_path):
    shard = "model-00001-of-00002.safetensors"
    path = os.path.join(w.cp, shard)
    hardlink = tmp_path / "hardlink"
    os.link(path, hardlink)
    assert verify_cli(w)[0] == 0
    ov = overlay(w, tmp_path, fresh=())
    before = os.stat(path)
    same_size_edit(str(hardlink))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    m, sha = manifest(w)
    for directory in (w.cp, ov):
        state, stale, bad = ck.identity(m, sha, directory, ck.stamp_path(w.stamps, directory))
        assert (state, stale, bad) == (ck.UNVERIFIED, [shard], [])
        doc = main.build(main.parse_args(w.base_args("--checkpoint-path", f"target={directory}")), w.env())
        with pytest.raises(main.Refusal, match="not verified"):
            main.preflight_checkpoints(doc, "strict")
    state, bad, hashed = ck.verify(m, sha, ov, ck.stamp_path(w.stamps, ov))
    assert (state, bad, hashed) == (ck.MISMATCH, [shard], 1)


@pytest.mark.parametrize("field", ["ctime_ns", "dev"])
@pytest.mark.parametrize("change", ["missing", "different"])
@pytest.mark.parametrize("linked", [False, True])
def test_stamp_requires_ctime_and_device(w, tmp_path, field, change, linked):
    assert verify_cli(w)[0] == 0
    stamp_file = ck.stamp_path(w.stamps, w.cp)
    stamp = ck.read_stamp(stamp_file)
    for entry in stamp["files"].values():
        if change == "missing":
            entry.pop(field, None)
        else:
            entry[field] = -1
    write(stamp_file, json.dumps(stamp))
    directory = overlay(w, tmp_path, fresh=()) if linked else w.cp
    m, sha = manifest(w)
    target_stamp = ck.stamp_path(w.stamps, directory)
    assert ck.identity(m, sha, directory, target_stamp)[0] == ck.UNVERIFIED
    assert ck.verify(m, sha, directory, target_stamp) == (ck.VERIFIED, [], len(m["files"]))


def test_stat_key_records_ctime_and_device(monkeypatch):
    st = SimpleNamespace(st_ino=7, st_size=11, st_mtime_ns=13, st_ctime_ns=17, st_dev=19)
    monkeypatch.setattr(ck, "os", SimpleNamespace(stat=lambda path: st))
    assert ck._stat_key("file") == {"inode": 7, "size": 11, "mtime_ns": 13, "ctime_ns": 17, "dev": 19}


@pytest.mark.parametrize("key, channel", [
    (key, channel)
    for key in ("args.spec-model", "args.speculative-config.model", "args.hf-overrides.engram_table_dir",
                "args.hf-overrides.text_config.engram_table_dir",
                "env.VLLM_EXL3_OVERRIDE")
    for channel in ("set", "overlay", "user", "ambient")
    if channel != "ambient" or key.startswith("env.")
])
def test_literal_checkpoint_fields_refused(w, key, channel):
    value = "/missing/checkpoint"
    args = ["--experiment", "dependency", "--policy", "merge"]
    environ = w.env()
    if channel == "set":
        args += ["--set", f"{key}={value}"]
    elif channel == "overlay":
        parent, leaf = key.rsplit(".", 1)
        path = write(os.path.join(w.tmp, "literal.toml"),
                     f'schema = "launch-overlay/1"\n[{parent}]\n{leaf} = "{value}"\n')
        args += ["--overlay", path]
    elif channel == "user":
        with open(os.path.join(w.settings, "model.toml"), "a") as f:
            f.write(f'\n[user.CHECKPOINT]\ntarget = "{key}"\n')
        args += ["--set", f"CHECKPOINT={value}"]
    else:
        environ[key[4:]] = value
    assert "checkpoint reference" in w.refused(*args, environ=environ)


@pytest.mark.parametrize("option", ["--spec-model", "--speculative-config", "--speculative_config.model", "--specul",
                                    "--hf-overrides", "--hf_overrides.engram_table_dir", "--hf-over"])
def test_raw_extra_checkpoint_options_refused_even_experiment(w, option):
    assert "checkpoint" in w.refused("--experiment", "dependency", "--", option, "/missing")


@pytest.mark.parametrize("key", ["args.speculative-config", "args.hf-overrides"])
def test_json_literal_checkpoint_fields_refused(w, key):
    leaf = "model" if key.endswith("speculative-config") else "engram_table_dir"
    value = json.dumps({leaf: "/missing"})
    assert "checkpoint reference" in w.refused("--set", f'{key}={json.dumps(value)}')


def test_drafter_reference_is_checked_and_cannot_be_replaced_by_same_path(w):
    add_role(w, "drafter", "synth-drafter")
    settings = ["--set", "args.speculative-config.model=${checkpoint.drafter.path}"]
    doc = main.build(main.parse_args(w.base_args(*settings)), w.env())
    assert set(doc["_cp_specs"]) == {"target", "drafter"}
    assert doc["checkpoint"]["drafter"]["state"] == "not present"
    assert "checkpoint reference" in w.refused("--set", f"args.speculative-config.model={w.models}/synth-drafter")


def sidecar(w):
    directory = os.path.join(w.models, "sidecar")
    write(os.path.join(directory, "override.json"), '{"weights":"override.safetensors"}')
    write(os.path.join(directory, "override.safetensors"), "weights")
    spec = {"kind": "sidecar", "required": ["override.json"], "weights": ["override.safetensors"]}
    with open(os.path.join(w.settings, "model.toml"), "a") as f:
        f.write('\n[checkpoint.override]\nrepo_key = "TARGET_REPO"\nrevision_key = "TARGET_REVISION"\n'
                'dir = "sidecar"\nkind = "sidecar"\nrequired = ["override.json"]\nweights = ["override.safetensors"]\n'
                'manifest = "checkpoints/sidecar.manifest.json"\n')
    m = ck.make_manifest("example/synth-dense-1b", "0123456789abcdef0123456789abcdef01234567", spec, directory)
    write(os.path.join(w.settings, "checkpoints", "sidecar.manifest.json"), json.dumps(m))
    return directory


def test_sidecar_manifest_status_preflight_and_unset(w):
    directory = sidecar(w)
    args = ["--set", "env.VLLM_EXL3_OVERRIDE=${checkpoint.override.path}/override.json"]
    assert w.verify() == 0
    doc = main.build(main.parse_args(w.base_args(*args)), w.env())
    assert set(doc["_cp_specs"]) == {"target", "override"}
    assert doc["checkpoint"]["override"]["identity"] == "verified"
    assert doc["_env"]["VLLM_EXL3_OVERRIDE"] == directory + "/override.json"
    main.preflight_checkpoints(doc, "strict")
    os.unlink(os.path.join(directory, "override.safetensors"))
    assert w.resolve(*args)["checkpoint"]["override"]["state"] == "incomplete"
    with pytest.raises(main.Refusal, match="checkpoint override.*not complete"):
        main.preflight_checkpoints(doc, "strict")
    assert "override" not in w.resolve()["checkpoint"]


@pytest.mark.parametrize("suffix", ["/../outside.json", "/unlisted.json", "/override.json/../outside.json"])
def test_sidecar_reference_must_name_manifest_file(w, suffix):
    sidecar(w)
    assert "manifest" in w.refused("--set", "env.VLLM_EXL3_OVERRIDE=${checkpoint.override.path}" + suffix)
