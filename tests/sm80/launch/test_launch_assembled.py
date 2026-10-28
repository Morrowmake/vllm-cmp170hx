# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An assembled engine (<dir>/engine, <dir>/plugin, <dir>/assembly.json)
pinned by a pin set: the plugin's commit, entry-point metadata and
extensions enter the tree state and the receipt; a legacy tree keeps its
engine-commit key."""

from __future__ import annotations

import json
import os

import pytest

from canon import sha256_file
from helpers import (FORK_VARS, RETIRED, World, add_build, git, head, make_tree, promote,
                     sm80_envs_py, write)
from treestate import tree_state

NATIVE_BUILD = "native-0123456789-t2.14.0"
NATIVES = ("vllm_sm80/_exl3_C.abi3.so", "vllm_sm80/_ampere_marlin_C.abi3.so")


def make_plugin(root: str) -> str:
    write(os.path.join(root, "vllm_sm80", "__init__.py"), "")
    write(os.path.join(root, "vllm_sm80", "envs.py"), sm80_envs_py(FORK_VARS, RETIRED))
    write(os.path.join(root, "pyproject.toml"), '[project]\nname = "vllm-sm80"\n')
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "plugin")
    info = os.path.join(root, "vllm_sm80-0.1.0.dist-info")
    write(os.path.join(info, "METADATA"), "Metadata-Version: 2.1\nName: vllm-sm80\nVersion: 0.1.0\n")
    write(os.path.join(info, "entry_points.txt"),
          "[vllm.general_plugins]\nsm80 = vllm_sm80.plugin:register\n")
    for rel in NATIVES:
        write(os.path.join(root, rel), f"native {rel}\n")
    return root


def write_assembly(parent: str, engine: str, plugin: str, **override) -> dict:
    data = {"schema": "engine-assembly/1", "upstream_commit": "1" * 40,
            "queue_commit": head(engine), "plugin_commit": head(plugin),
            "native_build": NATIVE_BUILD, "torch": "2.14.0",
            "natives": {rel: sha256_file(os.path.join(plugin, rel)) for rel in NATIVES}}
    data.update(override)
    write(os.path.join(parent, "assembly.json"), json.dumps(data))
    return data


def pin_set(w: World, assembly: dict) -> None:
    with open(os.path.join(w.settings, "pins.env")) as f:
        lines = [x for x in f.read().splitlines() if not x.startswith("ENGINE_COMMIT=")]
    lines += [f"UPSTREAM_COMMIT={assembly['upstream_commit']}",
              f"QUEUE_COMMIT={assembly['queue_commit']}",
              f"PLUGIN_COMMIT={assembly['plugin_commit']}",
              f"NATIVE_BUILD={assembly['native_build']}"]
    write(os.path.join(w.settings, "pins.env"), "\n".join(lines) + "\n")


@pytest.fixture
def aw(tmp_path):
    """A World whose tree is the engine of an assembled engine, pinned by a pin set."""
    w = World(tmp_path)
    parent = os.path.join(str(tmp_path), "assembled")
    engine = make_tree(os.path.join(parent, "engine"), sm80=False)
    add_build(engine)
    plugin = make_plugin(os.path.join(parent, "plugin"))
    pin_set(w, write_assembly(parent, engine, plugin))
    w.tree, w.parent, w.plugin = engine, parent, plugin
    assert w.verify() == 0
    return w


def reasons(d):
    return d["validated"]["reasons"]


def test_candidate_imports_engine_and_plugin(aw):
    d = aw.resolve()
    assert reasons(d) == ["no receipt"]
    assert d["child_env"]["PYTHONPATH"]["value"] == f"{aw.tree}:{aw.plugin}"
    plugin = d["engine_target"]["plugin"]
    assert plugin["commit"] == head(aw.plugin)
    assert plugin["dirty"] is False and plugin["natives"] == {
        rel: sha256_file(os.path.join(aw.plugin, rel)) for rel in NATIVES}
    # the fork switches come from the plugin's vllm_sm80/envs.py
    assert d["registry"]["sm80_envs_sha256"] == sha256_file(
        os.path.join(aw.plugin, "vllm_sm80", "envs.py"))


def test_receipt_binds_the_pin_set(aw):
    receipt = promote(aw, ["tp2"])
    assert {"UPSTREAM_COMMIT", "QUEUE_COMMIT", "PLUGIN_COMMIT", "NATIVE_BUILD"} <= set(receipt["pins"])
    assert "ENGINE_COMMIT" not in receipt["pins"]
    engine = receipt["engines"][0]
    assert engine["plugin_commit"] == head(aw.plugin) and engine["native_build"] == NATIVE_BUILD
    d = aw.resolve()
    assert d["validated"]["state"] == "VALIDATED", reasons(d)


def test_replaced_native_is_not_validated(aw):
    promote(aw, ["tp2"])
    write(os.path.join(aw.plugin, NATIVES[0]), "rebuilt\n")
    assert {"dirty tree", "plugin ≠ receipt"} <= set(reasons(aw.resolve()))


def test_plugin_at_another_commit_is_not_validated(aw):
    promote(aw, ["tp2"])
    write(os.path.join(aw.plugin, "vllm_sm80", "extra.py"), "X = 1\n")
    git(aw.plugin, "add", "-A")
    git(aw.plugin, "commit", "-q", "-m", "change")
    assert {"plugin ≠ pin", "plugin ≠ receipt", "dirty tree"} <= set(reasons(aw.resolve()))


def test_unowned_plugin_file_is_dirty(aw):
    promote(aw, ["tp2"])
    write(os.path.join(aw.plugin, "vllm_sm80", "stray.py"), "")
    assert reasons(aw.resolve()) == ["dirty tree"]


def test_assembly_not_matching_pins(aw):
    write_assembly(aw.parent, aw.tree, aw.plugin, native_build="native-fedcba9876-t2.14.0")
    assert "assembly ≠ pins" in reasons(aw.resolve())


def test_state_hash_covers_the_plugin(aw):
    before = tree_state(aw.tree).state_sha256()
    info = os.path.join(aw.plugin, "vllm_sm80-0.1.0.dist-info", "entry_points.txt")
    write(info, "[vllm.general_plugins]\n")
    assert tree_state(aw.tree).state_sha256() != before


def test_required_extensions_resolve_in_the_plugin(aw):
    """Settings shared with single-tree arms name the extensions by their
    single-tree paths; in an assembled engine they match the plugin's files
    under the same keys."""
    model = open(os.path.join(aw.settings, "model.toml")).read()
    write(os.path.join(aw.settings, "model.toml"), model.replace(
        "[args]\n", '[requires]\nfiles = ["vllm/_exl3_C*.so", "vllm/_ampere_marlin_C*.so", '
        '"vllm_sm80/envs.py"]\n\n[args]\n', 1))
    d = aw.resolve()
    assert d["requires"] == {
        "vllm/_exl3_C.abi3.so": sha256_file(os.path.join(aw.plugin, NATIVES[0])),
        "vllm/_ampere_marlin_C.abi3.so": sha256_file(os.path.join(aw.plugin, NATIVES[1])),
        "vllm_sm80/envs.py": sha256_file(os.path.join(aw.plugin, "vllm_sm80", "envs.py")),
    }


def test_legacy_tree_keeps_its_key(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    st = tree_state(w.tree)
    assert st.plugin is None and "plugin" not in st.describe()
    receipt = promote(w, ["tp2"])
    assert receipt["pins"]["ENGINE_COMMIT"] == head(w.tree)
    assert "plugin_commit" not in receipt["engines"][0]
    assert w.resolve()["validated"]["state"] == "VALIDATED"
