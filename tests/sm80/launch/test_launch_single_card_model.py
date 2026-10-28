# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A settings directory of another shape: LiquidAI/LFM2.5-24B-A2B (lfm2_moe,
BF16, one card, 131072 context, its own reasoning and tool-call parsers, no
draft model, no speculative decoding, no fork switches, no required fork
natives). It resolves and validates with only what it declares, and the
launcher execs the recording server stub with that command. The stub and
the synthetic argument file accept any parser name: whether the engine
registers the parsers named here is not checked by these tests."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from helpers import ARG_UTILS, World, head, promote, read_record, write

MODEL = """\
schema = "launch-model/1"
model = "lfm2.5-24b-a2b"
hardware = "one-card"
arg_order = ["served-model-name", "tensor-parallel-size", "max-model-len"]
fixed = ["args.max-model-len", "args.reasoning-parser", "args.tool-call-parser", "tp", "pp", "cards"]
controlled_args = ["max-num-seqs"]

[layouts]
allowed = ["tp1"]
default = "tp1"

[checkpoint.target]
repo_key = "TARGET_REPO"
revision_key = "TARGET_REVISION"
dir = "LFM2.5-24B-A2B"
index = "model.safetensors.index.json"
required = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
manifest = "checkpoints/lfm2.5-24b-a2b.manifest.json"

[args]
model = "${checkpoint.target.path}"
served-model-name = "LFM2.5-24B-A2B"
max-model-len = { value = 131072, why = "The model's full context." }
dtype = "bfloat16"
max-num-seqs = 8
reasoning-parser = { value = "deepseek_r1", why = "The chat template opens the thinking block in the generation prompt; this parser splits the thinking from the answer when the start token is absent." }
tool-call-parser = { value = "lfm2", why = "The model writes tool calls as Python calls." }
enable-auto-tool-choice = true
"""

HARDWARE = """\
schema = "launch-hardware/1"
card = "test-card"
sm = 80
count = 1
memory_gib = 80
"""

TP1 = """\
schema = "launch-layout/1"
name = "tp1"
tp = 1
pp = 1
cards = 1
card_ids = "0"

[args]
max-num-batched-tokens = { value = 8192, why = "One prefill chunk per step." }
"""

VARIANTS = {
    "enablement-only": 'schema = "launch-variant/1"\n',
    "first-light": 'schema = "launch-variant/1"\n[args]\nmax-model-len = 32768\n'
                   '[args.compilation-config]\ncudagraph_mode = "NONE"\n',
    "plain-upstream": 'schema = "launch-variant/1"\n[args]\nmax-num-seqs = 4\n',
}

REPO, REVISION = "LiquidAI/LFM2.5-24B-A2B", "89abcdef0123456789abcdef0123456789abcdef"

# the tree declares the options the settings protect
PARSER_ARGS = ARG_UTILS + """\
        parser.add_argument("--reasoning-parser", type=str)
        parser.add_argument("--tool-call-parser", type=str)
        parser.add_argument("--enable-auto-tool-choice", action="store_true")
"""

FILES = {
    "config.json": '{"model_type": "lfm2_moe", "num_experts": 64, "num_experts_per_tok": 4}',
    "generation_config.json": "{}",
    "tokenizer.json": '{"vocab": {}}',
    "tokenizer_config.json": "{}",
    "chat_template.jinja": "{{ messages }}",
    "model-00001-of-00002.safetensors": "A" * 64,
    "model-00002-of-00002.safetensors": "B" * 64,
}

# names and options that belong to other models' settings
FOREIGN_ARGS = ("speculative", "draft", "mtp", "prefix-match-unit", "prefill-chunk-with-decodes",
                "max-num-partial-prefills", "batch-sharded-sampling", "block-size", "trust-remote-code")


def single_card_world(tmp_path, **tree_kw) -> World:
    import checkpoint as ck
    from helpers import make_checkpoint
    from model import load_model_settings
    from settings_dir import SettingsDir
    tree_kw.setdefault("extra_files", {"vllm/engine/arg_utils.py": PARSER_ARGS})
    w = World(tmp_path, **tree_kw)
    shutil.rmtree(w.settings)
    shutil.rmtree(w.models)
    s = w.settings
    write(os.path.join(s, "model.toml"), MODEL)
    write(os.path.join(s, "hardware", "one-card.toml"), HARDWARE)
    write(os.path.join(s, "layouts", "tp1.toml"), TP1)
    for name, text in VARIANTS.items():
        write(os.path.join(s, "variants", f"{name}.toml"), text)
    write(os.path.join(s, "pins.env"), f"ENGINE_COMMIT={head(w.tree)}\nTARGET_REPO={REPO}\n"
          f"TARGET_REVISION={REVISION}\n")
    os.makedirs(os.path.join(s, "validated"))
    w.cp = make_checkpoint(os.path.join(w.models, "LFM2.5-24B-A2B"), FILES)
    ms = load_model_settings(SettingsDir(s))
    write(os.path.join(s, ms.checkpoints["target"]["manifest"]),
          json.dumps(ck.make_manifest(REPO, REVISION, ms.checkpoints["target"], w.cp),
                     indent=1, sort_keys=True) + "\n")
    assert w.verify() == 0
    return w


@pytest.fixture
def w(tmp_path):
    return single_card_world(tmp_path)


def options(argv: list[str]) -> list[str]:
    return [a[2:] for a in argv if a.startswith("--")]


def assert_own_shape(d: dict) -> None:
    """Nothing in argv or the environment the settings did not declare (the
    runtime's own locations and, in a container, the image's ENV aside)."""
    import main
    from settings_dir import load_toml
    image = load_toml(os.path.join(os.path.dirname(main.__file__), "defaults.toml"),
                      "launch-defaults")["runtime"]["container"]["image_env"]
    opts = options(d["argv"])
    assert not [o for o in opts if any(f in o for f in FOREIGN_ARGS)], opts
    assert d["env_functional"] == {}
    foreign = [n for n, e in d["child_env"].items() if e["class"] != "location"
               and (e["class"] == "functional" or n.startswith(("VLLM_", "NCCL_")))
               and not (n in image and e["value"] == image[n])]
    assert foreign == []
    assert d["requires"] == {} and list(d["checkpoint"]) == ["target"]


def test_resolves_with_only_its_own_settings(w):
    d = w.resolve(layout=None)
    assert d["layout"] == "tp1" and d["model"] == "lfm2.5-24b-a2b"
    assert d["argv"][1:3] == ["serve", w.cp]
    opts = options(d["argv"])
    assert opts[:3] == ["served-model-name", "tensor-parallel-size", "max-model-len"]
    assert set(opts) == {"served-model-name", "tensor-parallel-size", "pipeline-parallel-size",
                         "max-model-len", "dtype", "max-num-seqs", "reasoning-parser",
                         "tool-call-parser", "enable-auto-tool-choice", "max-num-batched-tokens",
                         "host"}
    a = d["argv"]
    assert a[a.index("--tensor-parallel-size") + 1] == "1"
    assert a[a.index("--reasoning-parser") + 1] == "deepseek_r1"
    assert a[a.index("--tool-call-parser") + 1] == "lfm2"
    assert d["child_env"]["CUDA_VISIBLE_DEVICES"]["value"] == "0"
    assert_own_shape(d)


def test_validates_and_execs_the_server_stub_on_one_card(w):
    """VALIDATED on one card; the exec'd stub records exactly the resolved
    command and no fork or NCCL name (a recording stub, not a server)."""
    promote(w, ["tp1"])
    d = w.resolve(layout="tp1")
    assert d["validated"]["state"] == "VALIDATED", d["validated"]
    assert_own_shape(d)
    proc = w.run(layout="tp1")
    assert proc.returncode == 0, proc.stderr
    argv, env = read_record(w.venv)
    assert argv[1:] == d["argv"][1:]
    assert not [n for n in env if n.startswith(("VLLM_", "NCCL_"))]


@pytest.mark.parametrize("variant", ["enablement-only", "first-light"])
def test_variants_resolve(w, variant):
    d = w.resolve("--variant", variant, layout="tp1")
    assert_own_shape(d)
    a = d["argv"]
    if variant == "first-light":
        assert a[a.index("--max-model-len") + 1] == "32768"
        assert json.loads(a[a.index("--compilation-config") + 1]) == {"cudagraph_mode": "NONE"}
    else:
        assert "--compilation-config" not in a and a[a.index("--max-model-len") + 1] == "131072"


def test_container_runtime_resolves(w):
    d = w.resolve("--runtime", "container", "--tree", w.tree, layout="tp1")
    assert_own_shape(d)
    assert d["child_env"]["CUDA_VISIBLE_DEVICES"]["value"] == "0"


def test_plain_upstream_resolves_on_a_tree_without_the_plugin(tmp_path):
    w = single_card_world(tmp_path, sm80=False)
    d = w.resolve("--capability", "plain-upstream", layout="tp1")
    assert d["validated"]["state"] is None and d["validated"]["reasons"] == ["plain upstream"]
    assert d["fork_switches_absent"] == []
    a = d["argv"]
    assert a[a.index("--max-num-seqs") + 1] == "4"
    assert_own_shape(d)


def test_plain_upstream_refused_on_the_plugin_tree(w):
    assert "carries the sm80 plugin" in w.refused("--capability", "plain-upstream", layout="tp1")


def test_protected_parsers_stay_fixed(w):
    assert "fixed by the settings" in w.refused("--", "--tool-call-parser", "hermes", layout="tp1")
    assert "fixed for this model" in w.refused("--set", "args.reasoning-parser=qwen3", layout="tp1")


def test_a_second_card_is_refused(w):
    assert "not cards of this hardware (0..0)" in w.refused("--set", "cards=1", layout="tp1")
    assert "needs exactly 1" in w.refused("--set", "cards=0,1", layout="tp1")
    assert "cards 1 != tp 2" in w.refused("--set", "tp=2", "--experiment", "e", layout="tp1")
