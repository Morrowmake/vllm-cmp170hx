# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards on what the launch can be made to serve without leaving VALIDATED:
checkpoint binding, location and secret handling, instrumentation, geometry,
protected keys, cards, boot verification, Git-free image trees."""

from __future__ import annotations

import io
import json
import os
import shutil
from contextlib import redirect_stdout

import pytest

from helpers import CHECKPOINT_FILES, World, git, make_checkpoint, promote, read_record, write
from treestate import MANIFEST, STAMP


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


# -- the executed checkpoint is the checked checkpoint -----------------------------

def test_model_argument_cannot_be_redirected(w, tmp_path):
    assert "--checkpoint-path" in w.refused("--set", "args.model=/elsewhere")
    ov = overlay(tmp_path, '[args]\nmodel = "/elsewhere"\n')
    assert "checkpoint" in w.refused("--overlay", ov)
    assert "set by the launcher" in w.refused("--", "--model", "/elsewhere")


def test_relocated_checkpoint_is_the_one_checked(w, tmp_path):
    same = str(tmp_path / "copy")
    shutil.copytree(w.cp, same)
    d = w.resolve("--checkpoint-path", f"target={same}")
    assert d["argv"][2] == same
    assert d["checkpoint"]["target"]["path"] == same
    assert "checkpoint not verified" in reasons(d)           # unstamped copy
    out = io.StringIO()
    import checkpoint as ck
    with redirect_stdout(out):
        assert ck.main(["verify", "--settings", w.settings, "--checkpoint", "target",
                        "--path", same, "--stamp-dir", w.stamps]) == 0
    assert w.resolve("--checkpoint-path", f"target={same}")["validated"]["state"] == "VALIDATED"
    files = dict(CHECKPOINT_FILES, **{"config.json": CHECKPOINT_FILES["config.json"].replace("synth", "other")})
    other = make_checkpoint(str(tmp_path / "other"), files)
    d = w.resolve("--checkpoint-path", f"target={other}")
    assert d["validated"]["state"] == "NOT_VALIDATED"
    assert w.run("--checkpoint-path", f"target={other}").returncode == 3
    assert "NAME=DIR" in w.refused("--checkpoint-path", "nope=/x")


def test_user_knob_cannot_target_model(w, tmp_path):
    model = open(os.path.join(w.settings, "model.toml")).read() + '\n[user.MODEL]\ntarget = "args.model"\n'
    write(os.path.join(w.settings, "model.toml"), model)
    env = write(str(tmp_path / "u.env"), "MODEL=/elsewhere\n")
    assert "may not target args.model" in w.refused("--env-file", env)


# -- [location] holds locations only ---------------------------------------------------

@pytest.mark.parametrize("name,value", [("VLLM_FAST_PATH", "0"), ("VLLM_OLD_THIN_GEMM", "1"),
                                        ("VLLM_NOPE", "1"), ("FOO_DEBUG", "1"),
                                        ("LD_PRELOAD", "/x.so"), ("CUDA_VISIBLE_DEVICES", "3")])
def test_location_table_refuses_other_classes(w, tmp_path, name, value):
    ov = overlay(tmp_path, f'[location]\n{name} = "{value}"\n')
    assert "[location] holds" in w.refused("--overlay", ov)


def test_location_relocation_allowed(w, tmp_path):
    ov = overlay(tmp_path, '[location]\nTRITON_CACHE_DIR = "${home}/triton"\n')
    d = w.resolve("--overlay", ov)
    assert d["child_env"]["TRITON_CACHE_DIR"]["value"] == os.path.join(w.home, "triton")
    assert d["validated"]["state"] == "VALIDATED"
    p = w.run("--overlay", ov)
    assert p.returncode == 0, p.stderr
    _, env = read_record(w.venv)
    assert {k: env[k] for k in d["env_functional"]} == d["env_functional"]


# -- instrumentation is a closed list --------------------------------------------------

def test_harness_overlay_cannot_hide_function(w, tmp_path):
    ok = overlay(tmp_path, 'class = "harness"\n[args]\nenable-prompt-tokens-details = true\n')
    d = w.resolve("--overlay", ok)
    assert d["harness"] == ["--enable-prompt-tokens-details"] and d["validated"]["state"] == "VALIDATED"
    for body in ('[args]\ndtype = "float32"\n', '[args]\nmax-num-batched-tokens = 1\n',
                 '[env]\nVLLM_FAST_PATH = "0"\n'):
        bad = overlay(tmp_path, 'class = "harness"\n' + body, "bad.toml")
        assert "harness overlay may set only" in w.refused("--overlay", bad)


# -- secrets ------------------------------------------------------------------------------

def test_secrets_masked_everywhere(w, tmp_path, capsys):
    import main
    d = w.resolve("--set", "env.VLLM_API_KEY=s3cret")
    assert "s3cret" not in json.dumps(d)
    assert "VLLM_API_KEY" not in d["env_functional"]
    assert main.main(w.base_args("--print-config", "--explain", "--set", "env.VLLM_API_KEY=s3cret"),
                     w.env()) == 0
    assert "s3cret" not in capsys.readouterr().out
    d = w.resolve(environ=w.env(HF_TOKEN="hf_tok"))
    assert "hf_tok" not in json.dumps(d)
    model = open(os.path.join(w.settings, "model.toml")).read() + \
        '\n[user.CREDENTIAL]\ntarget = "env.MY_TOKEN"\nsecret = true\n'
    write(os.path.join(w.settings, "model.toml"), model)
    base = w.resolve()["functional_sha256"]
    ov = overlay(tmp_path, '[env]\nMY_TOKEN = "ov-token"\n')
    env = write(str(tmp_path / "u.env"), "CREDENTIAL=topsecret\n")
    for extra, environ in ((("--env-file", env), w.env()),
                           (("--set", "env.MY_TOKEN=cli-token"), w.env()),
                           (("--overlay", ov), w.env()),
                           ((), w.env(MY_TOKEN="amb-token"))):
        d = w.resolve(*extra, environ=environ)
        text = json.dumps(d)
        assert not any(t in text for t in ("topsecret", "cli-token", "ov-token", "amb-token"))
        assert "MY_TOKEN" not in d["env_functional"] and d["functional_sha256"] == base
        assert d["child_env"]["MY_TOKEN"]["class"] == "secret"
    p = w.run("--env-file", env)
    assert p.returncode == 0 and read_record(w.venv)[1]["MY_TOKEN"] == "topsecret"


# -- geometry and protected keys ------------------------------------------------------------

@pytest.mark.parametrize("extra,needle", [
    (("--set", "args.tensor-parallel-size=1"), "set tp/pp"),
    (("--set", "args.max-num-seqs=4"), "fixed"),
    (("--set", "args.max_model_len=1024"), "fixed"),
    (("--", "--pipeline-parallel-size", "2"), "set by the launcher"),
])
def test_protected_channels(w, extra, needle):
    assert needle in w.refused(*extra)


def test_protected_unset_and_overlay(w, tmp_path):
    assert "unsetting" in w.refused("--overlay", overlay(tmp_path, 'unset = ["args.max-model-len"]\n'))
    assert "fixed" in w.refused("--overlay", overlay(tmp_path, '[args]\nmax-num-seqs = 2\n', "b.toml"))
    d = w.resolve("--experiment", "e1", "--set", "args.max-num-seqs=4", "--set", "tp=1", "--set", "pp=2")
    a = d["argv"]
    assert a[a.index("--tensor-parallel-size") + 1] == "1"
    assert a[a.index("--pipeline-parallel-size") + 1] == "2"
    assert "experiment e1" in reasons(d)


def test_fixed_spellings_normalised(w):
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        '"args.max-model-len", "args.gpu', '"args.max_model_len", "args.gpu')
    write(os.path.join(w.settings, "model.toml"), model)
    assert "fixed" in w.refused("--set", "args.max-model-len=1024")


# -- cards against the lock --------------------------------------------------------------

def test_cards_set_on_cli_checked_against_lock(w):
    assert "cards this process holds" in w.refused("--set", "cards=2,3",
                                                   environ=w.env(CUDA_VISIBLE_DEVICES="0,1"))
    d = w.resolve("--set", "cards=2,3", environ=w.env(CUDA_VISIBLE_DEVICES="2,3"))
    assert "CUDA_VISIBLE_DEVICES" in d["ambient_equal"]
    assert d["child_env"]["CUDA_VISIBLE_DEVICES"]["value"] == "2,3"


# -- variant selector knobs on the CLI ---------------------------------------------------------

def test_variant_knob_on_cli(w, tmp_path):
    assert "mm-cap" in w.resolve("--set", "MM_CAP=1")["variants"]
    env = write(str(tmp_path / "u.env"), "MM_CAP=1\n")
    assert "mm-cap" not in w.resolve("--env-file", env, "--set", "MM_CAP=0")["variants"]
    assert "expected 1 or 0" in w.refused("--set", "MM_CAP=maybe")


# -- requires stays inside the tree --------------------------------------------------------

@pytest.mark.parametrize("path", ["/etc/hostname", "../outside.so"])
def test_requires_path_escape(w, path):
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model.replace("[args]\n", f'[requires]\nfiles = ["{path}"]\n\n[args]\n', 1))
    assert "leaves the engine tree" in w.refused()


# -- built code-loading values ---------------------------------------------------------------

def test_virtual_env_equal_is_accepted(w):
    d = w.resolve(environ=w.env(VIRTUAL_ENV=w.venv, HOME=w.home))
    assert "VIRTUAL_ENV" in d["ambient_equal"] and "HOME" in d["ambient_equal"]
    assert d["validated"]["state"] == "VALIDATED"


def test_equal_ambient_reaches_child_as_settings_value(w):
    p = w.run(environ=w.env(VLLM_THIN_GEMM="1"))
    assert p.returncode == 0, p.stderr
    assert read_record(w.venv)[1]["VLLM_THIN_GEMM"] == "1"


# -- tracked changes carry content identity ---------------------------------------------------

def test_tracked_edits_have_distinct_identity(w):
    path = os.path.join(w.tree, "csrc", "kernel.cu")
    write(path, "// a\n")
    a = w.resolve()["engine_target"]
    write(path, "// b\n")
    b = w.resolve()["engine_target"]
    assert a["dirty"] and a["state_sha256"] != b["state_sha256"]
    git(w.tree, "add", "csrc/kernel.cu")
    staged = w.resolve()["engine_target"]["dirty_paths"]["tracked"]
    assert staged[0]["path"] == "csrc/kernel.cu" and staged[0]["sha256"]
    git(w.tree, "reset", "-q", "--hard")
    git(w.tree, "mv", "csrc/kernel.cu", "csrc/moved.cu")
    moved = w.resolve()["engine_target"]["dirty_paths"]["tracked"]
    assert moved[0]["path"] == "csrc/moved.cu" and moved[0]["from"] == "csrc/kernel.cu"
    git(w.tree, "rm", "-q", "-f", "csrc/moved.cu")
    gone = w.resolve()["engine_target"]["dirty_paths"]["tracked"]
    assert gone[0]["kind"].startswith("D") and gone[0]["sha256"] is None


# -- boot verification restates VALIDATED ----------------------------------------------------

def test_first_boot_verifies_then_banner_validated(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    promote(w, ["tp2"])
    shutil.rmtree(w.stamps)
    p = w.run("--policy", "merge", "--checkpoint-verify", "boot")
    assert p.returncode == 0, p.stderr
    assert p.stderr.startswith("VALIDATED synth-dense tp2"), p.stderr


def test_boot_rehashes_stale_beside_known_mismatch(w):
    tok = os.path.join(w.cp, "tokenizer.json")
    data = open(tok).read()
    write(tok, data[:-1] + "X")
    assert w.verify() == 5
    cfg = os.path.join(w.cp, "config.json")
    data = open(cfg).read()
    write(cfg, data[:-1] + "X")
    p = w.run("--policy", "merge", "--checkpoint-verify", "boot")
    assert p.returncode == 0
    d = w.resolve()
    assert sorted(d["checkpoint"]["target"]["files"]) == ["config.json", "tokenizer.json"]


# -- --verify-checkpoints stamps what the launch serves -------------------------------------------

def test_verify_checkpoints_uses_launch_paths(w, tmp_path):
    import main
    copy = str(tmp_path / "copy")
    shutil.copytree(w.cp, copy)
    out = io.StringIO()
    with redirect_stdout(out):
        code = main.main(w.base_args("--verify-checkpoints", "--checkpoint-path", f"target={copy}"),
                         w.env())
    assert code == 0 and json.loads(out.getvalue())["target"]["path"] == copy
    d = w.resolve("--checkpoint-path", f"target={copy}")
    assert d["checkpoint"]["target"]["identity"] == "verified"


def test_container_paths_use_hub_snapshot(tmp_path):
    import main
    from model import load_model_settings
    from settings_dir import SettingsDir
    w = World(tmp_path)
    hub = tmp_path / "hub"
    snap = hub / "models--example--synth-dense-1b" / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
    shutil.copytree(w.cp, snap)
    ns = main.parse_args(w.base_args("--runtime", "container", "--models-dir", str(tmp_path / "empty")))
    sd = SettingsDir(w.settings)
    rt = {"models_dir": "/models", "hub_dirs": [str(hub)]}
    paths = main.checkpoint_paths(ns, load_model_settings(sd), rt, sd.pins(), "/cache")
    assert paths["target"] == str(snap)


# -- Git-free image trees ----------------------------------------------------------------------

def image_tree(w, tmp_path):
    """A Git-free copy of the tree with provenance.json and a tracked inventory."""
    from canon import sha256_file
    from treestate import build_owned_sha256
    img = str(tmp_path / "image" / "vllm-src")
    tracked = git(w.tree, "ls-files").split()
    shutil.copytree(w.tree, img, ignore=shutil.ignore_patterns(".git"))
    man = json.load(open(os.path.join(img, MANIFEST)))
    man["tracked"] = {p: sha256_file(os.path.join(img, p)) for p in tracked}
    man["build_owned_sha256"] = build_owned_sha256(man["files"])
    write(os.path.join(img, MANIFEST), json.dumps(man))
    commit = json.load(open(os.path.join(img, STAMP)))["commit"]
    write(os.path.join(img, "provenance.json"), json.dumps({"schema": 1, "engine_commit": commit}))
    return img


def test_image_tree_clean_then_unowned(w, tmp_path):
    from treestate import tree_state
    img = image_tree(w, tmp_path)
    st = tree_state(img, image=True)
    assert not st.dirty, st.dirty_paths
    write(os.path.join(img, "run.log"), "x")
    assert not tree_state(img, image=True).dirty
    for rel in (".sm80-extra.json", "sitecustomize.py", "vllm/model_executor/layers/fused_moe/configs/E=8.json"):
        write(os.path.join(img, rel), "{}")
        st = tree_state(img, image=True)
        assert any(u["path"] == rel for u in st.dirty_paths.get("unowned", [])), rel
        os.unlink(os.path.join(img, rel))
    write(os.path.join(img, "vllm", "envs.py"), "edited = 1\n")
    assert "tracked" in tree_state(img, image=True).dirty_paths


def test_image_tree_without_inventory_is_invalid(w, tmp_path):
    from treestate import tree_state
    img = image_tree(w, tmp_path)
    man = json.load(open(os.path.join(img, MANIFEST)))
    del man["tracked"]
    write(os.path.join(img, MANIFEST), json.dumps(man))
    assert "metadata_invalid" in tree_state(img, image=True).dirty_paths
