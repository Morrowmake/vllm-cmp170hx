# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution, precedence, refusals and the synthetic three-layout model."""

from __future__ import annotations

import glob
import json
import os
import re

import pytest

from helpers import LAUNCH, World, promote, read_record, write


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


# -- synthetic model --------------------------------------------------------------

@pytest.mark.parametrize("layout,tp,pp", [("tp1", 1, 1), ("tp2", 2, 1), ("pp2", 1, 2)])
def test_every_layout_resolves(w, layout, tp, pp):
    d = w.resolve(layout=layout)
    assert d["schema"] == "launch-resolved/2"
    a = d["argv"]
    assert a[1] == "serve" and a[2] == os.path.join(w.models, "synth-dense-1b")
    assert a[a.index("--tensor-parallel-size") + 1] == str(tp)
    assert a[a.index("--pipeline-parallel-size") + 1] == str(pp)
    assert d["engine_target"]["commit"] and d["registry"]["source"] == "arm-tree"
    assert d["validated"]["state"] == "NOT_VALIDATED" and "no receipt" in d["validated"]["reasons"]


def test_arg_order_and_functional_excludes_location(w):
    d = w.resolve()
    a = d["argv"]
    assert a[3:7] == ["--served-model-name", "synth-dense", "--tensor-parallel-size", "2"]
    f = d["argv_functional"]
    assert "--port" not in f and "--host" not in f and a[2] not in f
    assert d["location_keys"]["args.port"] == 8123


def test_why_and_layers_recorded(w):
    d = w.resolve()
    r = d["resolved"]
    assert r["args.max-num-batched-tokens"]["layer"] == "layout"
    assert r["args.max-model-len"]["why"].startswith("Long enough")
    assert r["args.host"]["layer"] == "runtime"


def test_text_and_explain(w, capsys):
    import main
    assert main.main(w.base_args("--print-config", "--explain"), w.env()) == 0
    out = capsys.readouterr().out
    assert out.startswith("NOT VALIDATED (no receipt")
    assert "# Faster small-M GEMMs" in out


def test_launcher_names_no_real_model():
    text = "".join(open(p, encoding="utf-8").read().lower()
                   for p in glob.glob(os.path.join(LAUNCH, "**", "*.*"), recursive=True)
                   if p.endswith((".py", ".toml")))
    for word in ("glm", "qwen", "deepseek", "llama", "dflash"):
        assert word not in text, word


def test_variant_with_layout_subtable_and_unset(w):
    d = w.resolve("--variant", "lean", layout="pp2")
    assert "--trust-remote-code" not in d["argv"]
    assert d["argv"][d["argv"].index("--block-size") + 1] == "32"
    assert d["unset"]["args.trust-remote-code"] == "variant"


SET_AND_UNSET = {
    # one table
    "env": ('unset = ["env.VLLM_FAST_PATH"]\n[env]\nVLLM_FAST_PATH = { value = "1", why = "x" }\n',
            "env.VLLM_FAST_PATH"),
    "args": ('unset = ["args.max_num_seqs"]\n[args]\nmax-num-seqs = 4\n', "args.max-num-seqs"),
    "nested": ('unset = ["args.compilation-config.level"]\n[args.compilation-config]\nlevel = 1\n',
               "args.compilation-config.level"),
    "location": ('unset = ["loc.TMPDIR"]\n[location]\nTMPDIR = "/x"\n', "loc.TMPDIR"),
    "layout": ('[layout.pp2]\nunset = ["args.block-size"]\n[layout.pp2.args]\nblock-size = 16\n',
               "args.block-size"),
    # across the tables of one file
    "top-unset-layout-set": ('unset = ["env.VLLM_FAST_PATH"]\n[layout.tp2.env]\nVLLM_FAST_PATH = "1"\n',
                             "env.VLLM_FAST_PATH"),
    "top-set-layout-unset": ('[args]\nmax-num-seqs = 4\n[layout.pp2]\nunset = ["args.max-num-seqs"]\n',
                             "args.max-num-seqs"),
    "two-layouts": ('[layout.tp2]\nunset = ["env.VLLM_KERNEL_X"]\n[layout.pp2.env]\nVLLM_KERNEL_X = "1"\n',
                    "env.VLLM_KERNEL_X"),
    "parent-child": ('unset = ["args.compilation-config"]\n[layout.tp2.args.compilation-config]\nlevel = 1\n',
                     "args.compilation-config"),
}


@pytest.mark.parametrize("layout", ["tp2", "pp2"])
@pytest.mark.parametrize("case", sorted(SET_AND_UNSET))
@pytest.mark.parametrize("kind", ["variant", "overlay"])
def test_key_both_set_and_unset_in_one_file_refused(w, tmp_path, case, kind, layout):
    """One file that both sets and unsets a key (or a parent or child key)
    contradicts itself, in one table or across its top level and its
    [layout.*] tables: refused for variant and overlay files alike, whichever
    layout is launched."""
    text, key = SET_AND_UNSET[case]
    text = f'schema = "launch-{kind}/1"\n' + text
    if kind == "variant":
        write(os.path.join(w.settings, "variants", "both.toml"), text)
        msg = w.refused("--variant", "both", layout=layout)
    else:
        msg = w.refused("--overlay", write(str(tmp_path / "o.toml"), text), "--experiment", "e",
                        layout=layout)
    assert "in the same file" in msg and key in msg, msg


def test_one_file_unsetting_what_another_sets_still_applies(w, tmp_path):
    """Across files the layering decides, as before: a variant unsets a key
    the layout file sets, and an overlay sets a key the variant unsets."""
    write(os.path.join(w.settings, "variants", "off.toml"),
          'schema = "launch-variant/1"\nunset = ["env.VLLM_FAST_PATH"]\n')
    d = w.resolve("--variant", "off", layout="tp2")
    assert "VLLM_FAST_PATH" not in d["child_env"] and d["unset"]["env.VLLM_FAST_PATH"] == "variant"
    ov = write(str(tmp_path / "on.toml"), 'schema = "launch-overlay/1"\n[env]\nVLLM_FAST_PATH = "1"\n')
    d = w.resolve("--variant", "off", "--overlay", ov, "--experiment", "e", layout="tp2")
    assert d["child_env"]["VLLM_FAST_PATH"]["value"] == "1"


def test_user_knobs(w, tmp_path):
    env = write(str(tmp_path / "user.env"),
                "PORT=9001\nPLACEMENT=ram\nMM_CAP=1\nMM_MAX_PIXELS=1568\nAPI_KEY='s3cret'\n"
                "EXTRA_ARGS=--enable-log-requests\n")
    d = w.resolve("--env-file", env)
    a = d["argv"]
    assert a[a.index("--port") + 1] == "9001"
    assert json.loads(a[a.index("--mm-processor-kwargs") + 1]) == {"max_image_tokens": 2}
    assert d["child_env"]["VLLM_PLACEMENT"]["value"] == "ram"
    assert d["child_env"]["VLLM_API_KEY"] == {"value": "***", "class": "secret"}
    assert "s3cret" not in json.dumps(d)
    assert a[-1] == "--enable-log-requests" and d["argv_functional"][-1] == "--enable-log-requests"
    assert "mm-cap" in d["variants"]


# -- precedence ------------------------------------------------------------------

def test_precedence_every_layer(w, tmp_path):
    """One key at every layer; each layer removed from the top in turn."""
    key = "VLLM_PLACEMENT"
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        'CUDA_DEVICE_ORDER = "PCI_BUS_ID"\n', f'CUDA_DEVICE_ORDER = "PCI_BUS_ID"\n{key} = "m"\n', 1)
    model = model.replace('choices = ["ssd", "ram", "device"]\n', "")
    write(os.path.join(w.settings, "model.toml"), model)
    hw = os.path.join(w.settings, "hardware", "two-card.toml")
    write(hw, open(hw).read() + f'\n[env]\n{key} = "h"\n')
    lay = os.path.join(w.settings, "layouts", "tp2.toml")
    write(lay, open(lay).read() + f'{key} = "l"\n')
    write(os.path.join(w.settings, "variants", "x.toml"), f'schema = "launch-variant/1"\n[env]\n{key} = "v"\n')
    ov = write(str(tmp_path / "ov.toml"), f'schema = "launch-overlay/1"\n[env]\n{key} = "o"\n')
    env = write(str(tmp_path / "u.env"), "PLACEMENT=u\n")
    full = ["--policy", "merge", "--variant", "x", "--overlay", ov, "--env-file", env,
            "--set", f"env.{key}=c"]
    cases = [(full, w.env(**{key: "a"}), "c", "cli"),
             (full[:-2], w.env(**{key: "a"}), "a", "ambient"),
             (full[:-2], w.env(), "u", "user"),
             (full[:-4], w.env(), "o", "overlay"),
             (full[:-6], w.env(), "v", "variant"),
             (full[:-8], w.env(), "l", "layout")]
    for extra, environ, value, layer in cases:
        d = w.resolve(*extra, environ=environ)
        assert d["child_env"][key]["value"] == value, layer
        assert d["resolved"][f"env.{key}"]["layer"] == layer
        assert d["env_functional"][key] == value
    write(lay, open(lay).read().replace(f'{key} = "l"\n', ""))
    assert w.resolve()["resolved"][f"env.{key}"]["layer"] == "hardware"
    write(hw, open(hw).read().replace(f'{key} = "h"\n', ""))
    assert w.resolve()["resolved"][f"env.{key}"]["layer"] == "model"


def test_ambient_vs_cli_conflict_reaches_child(w):
    """Ambient (merge) and --set on one key: the CLI value is printed, hashed and exec'd."""
    environ = w.env(VLLM_THIN_GEMM="7")
    d = w.resolve("--policy", "merge", "--set", "env.VLLM_THIN_GEMM=9", environ=environ)
    assert d["child_env"]["VLLM_THIN_GEMM"]["value"] == "9"
    assert d["env_functional"]["VLLM_THIN_GEMM"] == "9"
    ref = w.resolve("--set", "env.VLLM_THIN_GEMM=9")
    assert d["functional_sha256"] == ref["functional_sha256"]
    assert w.verify() == 0
    p = w.run("--policy", "merge", "--set", "env.VLLM_THIN_GEMM=9", environ=environ)
    assert p.returncode == 0, p.stderr
    argv, env = read_record(w.venv)
    assert env["VLLM_THIN_GEMM"] == "9"
    assert argv[1:] == d["argv"][1:]


# -- refusals --------------------------------------------------------------------

@pytest.mark.parametrize("extra,needle", [
    (("--set", "env.VLLM_NOPE=1"), "not registered"),
    (("--set", "env.VLLM_OLD_THIN_GEMM=1"), "renamed to VLLM_THIN_GEMM"),
    (("--set", "env.vllm_thin_gemm=1"), "case-sensitive"),
    (("--set", "args.max-model-len=1024"), "fixed"),
    (("--set", "cards=0"), "exactly 2"),
    (("--set", "cards=0,7"), "not cards of this hardware"),
    (("--set", "env.VLLM_MOE_SKIP_PADDING=0", "--set", "env.VLLM_MOE_MASK_PADDING=1"), "Masking"),
    (("--", "--tensor-parallel-size", "1"), "--tensor-parallel-size"),
])
def test_refusals(w, extra, needle):
    assert needle in w.refused(*extra)


def test_fixed_key_from_env_file(w, tmp_path):
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model + '\n[user.MAX_LEN]\ntarget = "args.max-model-len"\ntype = "int"\n')
    env = write(str(tmp_path / "u.env"), "MAX_LEN=1024\n")
    assert "fixed" in w.refused("--env-file", env)
    d = w.resolve("--env-file", env, "--experiment", "exp-1")
    assert any("experiment" in r for r in d["validated"]["reasons"])


def test_unknown_env_file_key(w, tmp_path):
    env = write(str(tmp_path / "u.env"), "VLLM_THIN_GEMM=0\n")
    assert "not a documented setting" in w.refused("--env-file", env)


def test_layout_cards_exceed_hardware(w):
    write(os.path.join(w.settings, "layouts", "tp8.toml"),
          'schema = "launch-layout/1"\ntp = 8\npp = 1\ncards = 8\n')
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        'allowed = ["tp1", "tp2", "pp2"]', 'allowed = ["tp1", "tp2", "pp2", "tp8"]')
    write(os.path.join(w.settings, "model.toml"), model)
    assert "hardware has 4" in w.refused(layout="tp8")


def test_card_count_not_tp_times_pp(w):
    write(os.path.join(w.settings, "layouts", "tp2.toml"),
          'schema = "launch-layout/1"\ntp = 2\npp = 1\ncards = 1\n')
    assert "cards 1 != tp 2" in w.refused()


def test_missing_requires_file(w):
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model.replace("[args]\n", '[requires]\nfiles = [{ path = "vllm/_x_C*.so", fork_only = true }]\n\n[args]\n', 1))
    assert "_x_C*.so is missing" in w.refused()
    write(os.path.join(w.tree, "vllm", "_x_C.abi3.so"), "lib")
    d = w.resolve()
    assert list(d["requires"]) == ["vllm/_x_C.abi3.so"]
    assert "unowned" in d["engine_target"]["dirty_paths"]


def test_sm80_capability_needs_fork_envs(tmp_path):
    w = World(tmp_path, sm80=False)
    assert "no vllm_sm80/envs.py" in w.refused()


def test_plain_upstream_on_fork_tree_refused(w):
    assert "carries the sm80 plugin" in w.refused("--capability", "plain-upstream")


def test_plain_upstream_drops_fork_names(tmp_path):
    w = World(tmp_path, sm80=False)
    d = w.resolve("--capability", "plain-upstream", layout="pp2")
    assert sorted(d["fork_switches_absent"]) == ["VLLM_PP_SPLIT", "VLLM_THIN_GEMM"]
    assert "--block-size" not in d["argv"]
    assert d["validated"]["state"] is None and d["validated"]["reasons"] == ["plain upstream"]
    assert d["registry"]["sm80_envs_sha256"] is None


def test_no_option_names_another_registry():
    import main
    with pytest.raises(SystemExit):
        main.parse_args(["--settings", "x", "--launcher-tree", "y"])
    src = open(os.path.join(LAUNCH, "registry.py")).read()
    assert re.search(r"def read_registry\(tree: str, capability: str\)", src)


# -- validated --------------------------------------------------------------------

def test_validated_after_promote(w):
    assert w.verify() == 0
    promote(w, ["tp2", "pp2"])
    d = w.resolve()
    assert d["validated"]["state"] == "VALIDATED", d["validated"]
    assert d["validated"]["functional_match"] is True
    import main
    assert main.banner(d).startswith("VALIDATED synth-dense tp2 (engine ")
    assert w.resolve(layout="tp1")["validated"]["reasons"] == ["layout tp1 not validated"]


def test_cli_change_breaks_functional_match(w):
    assert w.verify() == 0
    promote(w, ["tp2"])
    d = w.resolve("--set", "env.VLLM_FAST_PATH=0")
    assert d["validated"]["functional_match"] is False
    assert "functional ≠ validated" in d["validated"]["reasons"]


# -- hardware cuda_home per runtime ---------------------------------------------------------

def with_cuda_home(w, line: str) -> None:
    from helpers import HARDWARE_TOML
    write(os.path.join(w.settings, "hardware", "two-card.toml"), HARDWARE_TOML + line + "\n")


def cuda_homes(w) -> tuple[str, str]:
    native = w.resolve()["child_env"]["CUDA_HOME"]["value"]
    container = w.resolve("--runtime", "container", "--tree", w.tree)["child_env"]["CUDA_HOME"]["value"]
    return native, container


@pytest.mark.parametrize("line, native, container", [
    ("", "/usr/local/cuda", "/usr/local/cuda"),
    ('cuda_home = "/opt/cuda-13.3"', "/opt/cuda-13.3", "/usr/local/cuda"),
    ('cuda_home = { native = "/opt/cuda-13.3", container = "/usr/local/cuda" }',
     "/opt/cuda-13.3", "/usr/local/cuda"),
    ('cuda_home = { native = "/opt/cuda-13.3" }', "/opt/cuda-13.3", "/usr/local/cuda"),
])
def test_hardware_cuda_home_per_runtime(w, line, native, container):
    """A plain cuda_home names the native CUDA only; a { native, container }
    table names it per runtime (the container entry can only restate the
    image's CUDA_HOME); a runtime it leaves out keeps the default."""
    with_cuda_home(w, line)
    assert cuda_homes(w) == (native, container)


def test_plain_cuda_home_resolves_as_before(w):
    """The plain form and a table naming the same native CUDA resolve the
    same launch; PATH follows the CUDA chosen."""
    with_cuda_home(w, 'cuda_home = "/opt/cuda-13.3"')
    plain = w.resolve()
    with_cuda_home(w, 'cuda_home = { native = "/opt/cuda-13.3" }')
    table = w.resolve()
    for key in ("argv", "child_env", "functional_sha256"):
        assert plain[key] == table[key], key
    assert "/opt/cuda-13.3/bin" in plain["child_env"]["PATH"]["value"].split(":")


@pytest.mark.parametrize("line, expect", [
    ('cuda_home = { native = "/opt/cuda", gpu = "/x" }', "unknown runtime gpu"),
    ('cuda_home = { native = 13 }', "cuda_home.native: not a directory name"),
    ('cuda_home = { native = "" }', "cuda_home.native: not a directory name"),
    ("cuda_home = 13", "a directory name or a { native, container } table"),
    ('cuda_home = { container = "/usr/local/cuda-13.3" }', "differs from the image's CUDA_HOME"),
    ('cuda_home = { native = "/opt/cuda", container = "/usr/local/cuda-13.3" }',
     "differs from the image's CUDA_HOME"),
])
def test_hardware_cuda_home_malformed_refused(w, line, expect):
    """Refused natively too: a settings error shows before any container runs."""
    with_cuda_home(w, line)
    assert expect in w.refused()
    assert expect in w.refused("--runtime", "container", "--tree", w.tree)
