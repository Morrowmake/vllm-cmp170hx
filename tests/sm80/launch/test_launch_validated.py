# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The validated statement: receipt binding, tree state, generations,
candidate and final settings, adversarial inputs and schema majors."""

from __future__ import annotations

import json
import os
import shutil
import threading

import pytest

from canon import canonical_sha256
from helpers import (BUILD, VERSION_TEXT, World, add_build, git, head, promote, publish, read_record,
                     write, write_stamp)
from treestate import MANIFEST, STAMP, build_owned_sha256


@pytest.fixture
def w(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    return w


def reasons(d):
    return d["validated"]["reasons"]


def only(d, *expected):
    assert set(reasons(d)) == set(expected), reasons(d)


@pytest.fixture
def built(w):
    promote(w, ["tp2", "pp2"])
    d = w.resolve()
    assert d["validated"]["state"] == "VALIDATED", reasons(d)
    assert d["engine_target"]["build_owned"]["source"] == "built_in_place"
    return w


# -- receipt binding ------------------------------------------------------------

def test_hand_edited_validated_file(built):
    p = os.path.join(built.settings, "validated", "tp2.json")
    data = json.load(open(p))
    data["gate_record"] = "other"
    write(p, json.dumps(data))
    assert "validated file ≠ receipt" in reasons(built.resolve())


def test_hand_edited_receipt_entry(built):
    p = os.path.join(built.settings, "validated", "receipt.json")
    data = json.load(open(p))
    data["layouts"]["tp2"]["functional_sha256"] = "0" * 64
    write(p, json.dumps(data))
    d = built.resolve()
    assert "receipt ≠ validated file" in reasons(d) and d["validated"]["functional_match"] is False


def test_missing_receipt(built):
    os.unlink(os.path.join(built.settings, "validated", "receipt.json"))
    only(built.resolve(), "no receipt")


def test_pin_only_engine_change(built):
    """Another clean commit, same natives and build files, pin moved to it."""
    write(os.path.join(built.tree, "README"), "doc\n")
    git(built.tree, "add", "README")
    git(built.tree, "commit", "-q", "-m", "doc")
    add_build(built.tree)
    pins = open(os.path.join(built.settings, "pins.env")).read()
    old = pins.split("ENGINE_COMMIT=")[1].split("\n")[0]
    write(os.path.join(built.settings, "pins.env"), pins.replace(old, head(built.tree)))
    only(built.resolve(), "pin ≠ receipt", "engine not in receipt")


def test_tree_at_uncertified_commit(built):
    write(os.path.join(built.tree, "README"), "doc\n")
    git(built.tree, "add", "README")
    git(built.tree, "commit", "-q", "-m", "doc")
    add_build(built.tree)
    only(built.resolve(), "tree ≠ pin", "engine not in receipt")


def test_natives_replaced(built):
    add_build(built.tree, natives="natives-2")
    only(built.resolve(), "natives ≠ receipt")


def test_owned_byte_edited_then_manifest_regenerated(built):
    write(os.path.join(built.tree, "vllm/third_party/kern/compaction.py"), "x = 2\n")
    d = built.resolve()
    only(d, "dirty tree")
    assert d["engine_target"]["dirty_paths"]["modified_owned"][0]["path"] == "vllm/third_party/kern/compaction.py"
    add_build(built.tree, dict(BUILD, **{"vllm/third_party/kern/compaction.py": "x = 2\n"}))
    only(built.resolve(), "build ≠ receipt")


@pytest.mark.parametrize("rel", ["sitecustomize.py", "plugins/sm80/x.py", "notes.txt",
                                 "vllm/model_executor/layers/fused_moe/configs/E=8.json",
                                 ".sm80-extra.json"])
def test_unowned_is_dirty(built, rel):
    write(os.path.join(built.tree, rel), "{}\n")
    d = built.resolve()
    only(d, "dirty tree")
    first = d["engine_target"]["dirty_paths"]["unowned"][0]
    assert first["path"] == rel
    write(os.path.join(built.tree, rel), "{\"a\": 1}\n")
    again = built.resolve()["engine_target"]
    assert again["dirty_paths"]["unowned"][0]["sha256"] != first["sha256"]
    assert again["state_sha256"] != d["engine_target"]["state_sha256"]


@pytest.mark.parametrize("rel", ["x.py.bak-20261016", "run.log", "vllm/__pycache__/a.pyc", "docs/n.md"])
def test_exempt_is_clean(built, rel):
    write(os.path.join(built.tree, rel), "x")
    assert built.resolve()["validated"]["state"] == "VALIDATED"


def test_changed_requires_file_same_commit(built):
    model = open(os.path.join(built.settings, "model.toml")).read()
    write(os.path.join(built.settings, "model.toml"),
          model.replace("[args]\n", '[requires]\nfiles = ["vllm/_C*.so"]\n\n[args]\n', 1))
    promote(built, ["tp2"])
    assert built.resolve()["validated"]["state"] == "VALIDATED"
    add_build(built.tree, dict(BUILD, **{"vllm/_C.abi3.so": "other-lib"}))
    assert "requires ≠ validated" in reasons(built.resolve())


@pytest.mark.parametrize("tamper", ["map", "stamp_commit", "stamp_owned"])
def test_provenance_metadata_tamper(built, tamper):
    mp, sp = os.path.join(built.tree, MANIFEST), os.path.join(built.tree, STAMP)
    if tamper == "map":
        m = json.load(open(mp))
        m["files"]["vllm/_C.abi3.so"] = "f" * 64
        write(mp, json.dumps(m))
    elif tamper == "stamp_commit":
        s = json.load(open(sp))
        s["commit"] = "1" * 40
        write(sp, json.dumps(s))
    else:
        s = json.load(open(sp))
        s["build_owned_sha256"] = "2" * 64
        write(sp, json.dumps(s))
    d = built.resolve()
    assert "metadata_invalid" in d["engine_target"]["dirty_paths"]
    assert "dirty tree" in reasons(d)


def test_adopted_never_validated(built):
    add_build(built.tree, source="adopted")
    assert "adopted tree" in reasons(built.resolve())


def linked_world(tmp_path, *, change_build_input: bool):
    """vllm-src at commit A with a manifest; a worktree whose artefacts are
    symlinks into it, at commit A or a later commit B."""
    w = World(tmp_path, build=False)
    src = os.path.join(w.tmp, "src")
    shutil.copytree(w.tree, src, symlinks=True)
    src_man = add_build(src, {"vllm/_C.abi3.so": "native-lib"}, stamp=False)
    a = head(w.tree)
    rel = "csrc/kernel.cu" if change_build_input else "README"
    write(os.path.join(w.tree, rel), "// changed\n")
    git(w.tree, "add", rel)
    git(w.tree, "commit", "-q", "-m", "next")
    os.symlink(os.path.join(src, "vllm/_C.abi3.so"), os.path.join(w.tree, "vllm/_C.abi3.so"))
    write(os.path.join(w.tree, "vllm/_version.py"),
          VERSION_TEXT.replace("g0123456", "g" + head(w.tree)[:9]))
    from canon import sha256_file
    owned = {"vllm/_C.abi3.so": src_man["files"]["vllm/_C.abi3.so"],
             "vllm/_version.py": sha256_file(os.path.join(w.tree, "vllm/_version.py"))}
    man = {"schema": "build-owned/1", "source": "linked", "commit": a, "files": owned,
           "native_manifest_sha256": src_man["native_manifest_sha256"],
           "native_manifest_path": src_man["native_manifest_path"],
           "build_owned_sha256": build_owned_sha256(owned),
           "linked_from": {"tree": src, "commit": a, "build_owned_sha256": src_man["build_owned_sha256"]}}
    write(os.path.join(w.tree, MANIFEST), json.dumps(man))
    write_stamp(w.tree, man)
    pins = open(os.path.join(w.settings, "pins.env")).read()
    write(os.path.join(w.settings, "pins.env"), pins.replace(pins.split("ENGINE_COMMIT=")[1].split("\n")[0], head(w.tree)))
    assert w.verify() == 0
    return w, src, man


def test_linked_compatible_clean(tmp_path):
    w, _, _ = linked_world(tmp_path, change_build_input=False)
    et = w.resolve()["engine_target"]
    assert not et["dirty"] and not et["cross_commit_artifacts"], et["dirty_paths"]


def test_linked_cross_commit(tmp_path):
    w, _, _ = linked_world(tmp_path, change_build_input=True)
    d = w.resolve()
    assert d["engine_target"]["cross_commit_artifacts"] and not d["engine_target"]["dirty"]
    assert "cross-commit artefacts" in reasons(d)        # candidates see it too
    promote(w, ["tp2"])
    assert "cross-commit artefacts" not in reasons(w.resolve())   # the receipt names this build
    r = json.load(open(os.path.join(w.settings, "validated", "receipt.json")))
    r["engines"][0]["build_owned_sha256"] = "9" * 64
    write(os.path.join(w.settings, "validated", "receipt.json"), json.dumps(r))
    assert "cross-commit artefacts" in reasons(w.resolve())


def test_linked_source_rebuilt(tmp_path):
    w, src, _ = linked_world(tmp_path, change_build_input=False)
    write(os.path.join(src, "vllm/_C.abi3.so"), "rebuilt")
    assert "modified_owned" in w.resolve()["engine_target"]["dirty_paths"]


def test_linked_claims_extra_owned_file(tmp_path):
    """A self-consistent map plus file edit: an arbitrary runtime file added
    to the owned map, with map hash and file in step."""
    w, _, man = linked_world(tmp_path, change_build_input=False)
    from canon import sha256_file
    write(os.path.join(w.tree, "vllm/extra_hook.py"), "import os\n")
    man["files"]["vllm/extra_hook.py"] = sha256_file(os.path.join(w.tree, "vllm/extra_hook.py"))
    man["build_owned_sha256"] = build_owned_sha256(man["files"])
    write(os.path.join(w.tree, MANIFEST), json.dumps(man))
    bad = w.resolve()["engine_target"]["dirty_paths"]["metadata_invalid"]
    assert any("neither linked nor written" in b for b in bad)


def test_built_map_without_record_or_different_record(built):
    m = json.load(open(os.path.join(built.tree, MANIFEST)))
    rec = m.pop("build_record")
    write(os.path.join(built.tree, MANIFEST), json.dumps(m))
    assert "dirty tree" in reasons(built.resolve())
    add_build(built.tree)
    write(rec["path"], json.dumps({"schema": "build-record/1", "files": {"x": "0" * 64}}))
    bad = built.resolve()["engine_target"]["dirty_paths"]["metadata_invalid"]
    assert any("build record" in b for b in bad)


def test_native_manifest_file_modified(built):
    m = json.load(open(os.path.join(built.tree, MANIFEST)))
    write(m["native_manifest_path"], json.dumps({"natives": "edited"}))
    bad = built.resolve()["engine_target"]["dirty_paths"]["metadata_invalid"]
    assert any("native manifest" in b for b in bad)


# -- generations --------------------------------------------------------------------

def test_generation_pointer_and_swap_mid_read(w):
    import validated
    from settings_dir import SettingsDir
    shutil.rmtree(os.path.join(w.settings, "validated"))
    promote(w, ["tp2"], generation="1")
    assert w.resolve()["validated"]["state"] == "VALIDATED"
    sd = SettingsDir(w.settings)
    swapped = []

    def swap():
        if not swapped:
            swapped.append(1)
            t = threading.Thread(target=promote, args=(w, ["tp2"]),
                                 kwargs={"generation": "2", "gate": "gate-2"})
            t.start()
            t.join(0.5)   # the writer needs no lock: it renames the pointer
    gen = validated.read_generation(sd, between=swap)
    r = json.loads(gen.receipt)
    v = json.loads(gen.layout("tp2"))
    assert r["layouts"]["tp2"]["gate_record"] == v["gate_record"]
    assert os.path.realpath(os.path.join(w.settings, "validated")).endswith("validated.d/2")


def test_prune_waits_for_reader_and_lock_not_inherited(w):
    import fcntl

    import validated
    from settings_dir import SettingsDir
    shutil.rmtree(os.path.join(w.settings, "validated"))
    promote(w, ["tp2"], generation="1")
    lock = os.path.join(w.settings, "validated.d", ".lock")
    events = []

    def pruner():
        fd = os.open(lock, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        events.append("prune")
        os.close(fd)
    t = threading.Thread(target=pruner)

    def between():
        t.start()
        t.join(0.3)
        events.append("reader-done")
    validated.read_generation(SettingsDir(w.settings), between=between)
    t.join(5)
    assert events == ["reader-done", "prune"]
    p = w.run()
    assert p.returncode == 0, p.stderr
    fds = open(os.path.join(w.venv, "record.fds")).read()
    assert ".lock" not in fds


def test_tree_hash_ignores_pointer_form(w):
    promote(w, ["tp2"])
    plain = w.resolve()["settings"]["tree_sha256"]
    shutil.rmtree(os.path.join(w.settings, "validated"))
    promote(w, ["tp2"], generation="1")
    assert w.resolve()["settings"]["tree_sha256"] == plain


def test_full_build_reads_one_generation_with_staging_pointer(w):
    """A publisher paused with validated.new present: the launch still reads,
    and receipt, validated statement and settings hashes come from one
    generation."""
    import main
    shutil.rmtree(os.path.join(w.settings, "validated"))
    promote(w, ["tp2"], generation="1")
    files = {name: open(os.path.join(w.settings, "validated.d", "1", name), "rb").read()
             for name in os.listdir(os.path.join(w.settings, "validated.d", "1"))}
    files["tp2.json"] = files["tp2.json"].replace(b"gate-synth-1", b"gate-synth-2")
    seen = {}

    def during():
        seen["doc"] = main.build(main.parse_args(w.base_args()), w.env())
    publish(w.settings, files, "2", pause_before_rename=during)
    d = seen["doc"]
    assert d["validated"]["generation"] == "1" and d["validated"]["state"] == "VALIDATED"
    import hashlib
    assert d["settings"]["files"]["validated/tp2.json"] == hashlib.sha256(
        open(os.path.join(w.settings, "validated.d", "1", "tp2.json"), "rb").read()).hexdigest()


# -- candidate and final ---------------------------------------------------------------

def test_candidate_matches_its_frozen_launch(w, tmp_path):
    frozen = tmp_path / "expected.json"
    frozen.write_text(json.dumps(w.resolve()))
    d = w.resolve("--role", "candidate", "--expect", str(frozen))
    assert d["expected"]["functional_match"] is True
    only(d, "no receipt")


@pytest.mark.parametrize("damage", ["tracked", "metadata", "adopted"])
def test_candidate_still_reports_tree_problems(w, tmp_path, damage):
    frozen = tmp_path / "expected.json"
    frozen.write_text(json.dumps(w.resolve()))
    if damage == "tracked":
        write(os.path.join(w.tree, "csrc", "kernel.cu"), "// edited\n")
    elif damage == "metadata":
        s = json.load(open(os.path.join(w.tree, STAMP)))
        s["commit"] = "1" * 40
        write(os.path.join(w.tree, STAMP), json.dumps(s))
    else:
        add_build(w.tree, source="adopted")
    d = w.resolve("--role", "candidate", "--expect", str(frozen))
    assert d["expected"]["functional_match"] is True
    assert "no receipt" in reasons(d)
    assert ("adopted tree" if damage == "adopted" else "dirty tree") in reasons(d)


def test_intentional_change_versus_accident(w, tmp_path):
    frozen = tmp_path / "expected.json"
    frozen.write_text(json.dumps(w.resolve("--set", "env.VLLM_FAST_PATH=0")))
    write(os.path.join(w.settings, "layouts", "tp2.toml"),
          open(os.path.join(w.settings, "layouts", "tp2.toml")).read().replace(
              'VLLM_FAST_PATH = { value = "1"', 'VLLM_FAST_PATH = { value = "0"'))
    ok = w.resolve("--role", "candidate", "--expect", str(frozen))
    assert ok["expected"]["functional_match"] is True
    promote(w, ["tp2"])
    assert w.resolve("--role", "final")["validated"]["state"] == "VALIDATED"
    bad = w.resolve("--set", "env.VLLM_KERNEL_X=1", "--expect", str(frozen))
    assert bad["expected"]["functional_match"] is False
    assert bad["expected"]["env_differs"] == ["VLLM_KERNEL_X"]
    p = w.run("--set", "env.VLLM_KERNEL_X=1", "--expect", str(frozen))
    assert p.returncode == 3 and "differs from the expected launch" in p.stderr


def test_final_without_receipt_refused(w):
    promote(w, ["tp2"])
    os.unlink(os.path.join(w.settings, "validated", "receipt.json"))
    assert "never treated as a candidate" in w.refused("--role", "final")
    promote(w, ["tp2"])
    assert "must not carry a receipt" in w.refused("--role", "candidate")


# -- receipt structure: missing never equals missing ------------------------------------------

@pytest.mark.parametrize("field,edit", [
    ("pins", lambda r: r.update(pins={})),
    ("pins.TARGET_REVISION", lambda r: r["pins"].pop("TARGET_REVISION")),
    ("engines", lambda r: r.update(engines=[])),
    ("engines.build_owned", lambda r: r["engines"][0].update(build_owned_sha256=None)),
    ("engines.natives", lambda r: r["engines"][0].update(native_manifest_sha256=None)),
    ("checkpoints", lambda r: r.update(checkpoints={})),
    ("layouts", lambda r: r.update(layouts={})),
])
def test_receipt_field_removed(built, field, edit):
    p = os.path.join(built.settings, "validated", "receipt.json")
    r = json.load(open(p))
    edit(r)
    write(p, json.dumps(r))
    d = built.resolve()
    assert d["validated"]["state"] == "NOT_VALIDATED", field


def test_tree_without_build_identity_not_validated(tmp_path):
    w = World(tmp_path, build=False)
    assert w.verify() == 0
    promote(w, ["tp2"])
    d = w.resolve()
    assert d["validated"]["state"] == "NOT_VALIDATED"
    assert "build ≠ receipt" in reasons(d) or "receipt malformed" in " ".join(reasons(d))


# -- adversarial inputs and schemas ----------------------------------------------------

@pytest.mark.parametrize("text,needle", [
    ('[args]\nx = nan\n', "non-finite"),
    ('[args]\nx = inf\n', "non-finite"),
    ('[args]\nmax_num_seqs = 4\nmax-num-seqs = 4\n', "same argument"),
    ('[env]\nVLLM_THIN_GEMM = "1\\n2"\n', "newline or NUL"),
    ('[env]\nVLLM_THIN_GEMM = "1\\u0000"\n', "newline or NUL"),
])
def test_bad_values(w, text, needle):
    write(os.path.join(w.settings, "layouts", "tp2.toml"),
          'schema = "launch-layout/1"\ntp = 2\npp = 1\ncards = 2\n' + text)
    assert needle in w.refused()


def test_symlinks_refused(w, tmp_path):
    os.symlink(str(tmp_path), os.path.join(w.settings, "layouts", "out.toml"))
    assert "symlink" in w.refused()
    os.unlink(os.path.join(w.settings, "layouts", "out.toml"))
    os.symlink("tp2.toml", os.path.join(w.settings, "layouts", "alias.toml"))
    assert "symlink" in w.refused()


def test_community_layout(w, tmp_path):
    promote(w, ["tp2"])
    extra = tmp_path / "extra"
    write(str(extra / "layouts" / "tp2.toml"), open(os.path.join(w.settings, "layouts", "tp2.toml")).read())
    d = w.resolve("--settings-extra", str(extra))
    assert d["validated"]["state"] == "VALIDATED"          # functionally equal
    write(str(extra / "layouts" / "tp2.toml"),
          'schema = "launch-layout/1"\n[args]\nmax-num-batched-tokens = 1000\n')
    assert "community layer" in reasons(w.resolve("--settings-extra", str(extra)))
    write(str(extra / "layouts" / "tp1x.toml"), 'schema = "launch-layout/1"\ntp = 1\npp = 1\ncards = 1\ncard_ids = "1"\n')
    d = w.resolve("--settings-extra", str(extra), layout="tp1x")
    assert "community layer" in reasons(d)


@pytest.mark.parametrize("schema", ["launch-layout/2", "launch-model/1"])
def test_unknown_schema_major(w, schema):
    path = os.path.join(w.settings, "layouts", "tp2.toml")
    write(path, open(path).read().replace('schema = "launch-layout/1"', f'schema = "{schema}"'))
    assert "schema" in w.refused()


def test_receipt_unknown_major(w):
    promote(w, ["tp2"])
    p = os.path.join(w.settings, "validated", "receipt.json")
    r = json.load(open(p))
    r["schema"] = "validation-receipt/2"
    write(p, json.dumps(r))
    assert "unsupported schema" in w.refused()


def test_previous_major_fixtures_read():
    """Every schema the launcher reads has exactly one major today; this
    keeps the list honest when a second major is added."""
    from settings_dir import SUPPORTED_SCHEMAS
    assert all(len(v) == 1 for v in SUPPORTED_SCHEMAS.values())


def test_exec_matches_printed(w):
    promote(w, ["tp2"])
    d = w.resolve()
    p = w.run()
    assert p.returncode == 0, p.stderr
    assert p.stderr.startswith("VALIDATED synth-dense tp2")
    argv, env = read_record(w.venv)
    assert argv == d["argv"]
    assert set(env) == set(d["child_env"])
    assert {k: v for k, v in env.items() if d["child_env"][k]["class"] == "functional"} == d["env_functional"]
    assert canonical_sha256({"argv": d["argv_functional"], "env": d["env_functional"]}) == d["functional_sha256"]
