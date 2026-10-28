# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[user] knob forms: several targets, variant-gated knobs, value-to-variant
maps, booleans, and runtime-specific location values."""

from __future__ import annotations

import json
import os

import pytest

from helpers import World, write

EXTRA = """
[user.FRAMES]
target = ["args.limit-mm-per-prompt.video.num_frames", "args.media-io-kwargs.video.num_frames"]
type = "int"

[user.SPEC_N]
target = "args.speculative-config.num_speculative_tokens"
type = "int"
variants = ["spec"]

[user.SPEC_MODE]
variant = { spec = "spec", none = "" }

[user.ARGMAX]
target = "args.speculative-config.use_local_argmax"
type = "bool"
variants = ["spec"]

[user.FAIR]
variant = { "1" = "", "0" = "no-fair" }

[location]
WORKSPACE_PLACEHOLDER = "unused"
"""


@pytest.fixture
def w(tmp_path):
    w = World(tmp_path)
    model = open(os.path.join(w.settings, "model.toml")).read()
    model = model.replace("[[refuse]]", EXTRA.replace("[location]\nWORKSPACE_PLACEHOLDER = \"unused\"\n", "")
                          + "\n[[refuse]]", 1)
    model = model.replace("trust-remote-code = true\n",
                          'trust-remote-code = true\nprefill-chunk-with-decodes = 384\n'
                          'max-num-partial-prefills = 2\n', 1)
    model = model.replace('CUDA_DEVICE_ORDER = "PCI_BUS_ID"\n',
                          'CUDA_DEVICE_ORDER = "PCI_BUS_ID"\n\n[location]\n'
                          'FLASHINFER_WORKSPACE_BASE = { native = "${home}/fi-cache" }\n', 1)
    write(os.path.join(w.settings, "model.toml"), model)
    write(os.path.join(w.settings, "variants", "spec.toml"),
          'schema = "launch-variant/1"\n[args.speculative-config]\nmethod = "mtp"\nnum_speculative_tokens = 3\n')
    write(os.path.join(w.settings, "variants", "no-fair.toml"),
          'schema = "launch-variant/1"\nunset = ["args.prefill-chunk-with-decodes", "args.max-num-partial-prefills"]\n')
    return w


def env_file(tmp_path, text):
    return write(str(tmp_path / "u.env"), text)


def arg(d, name):
    return d["argv"][d["argv"].index(name) + 1]


def test_one_knob_two_targets(w, tmp_path):
    d = w.resolve("--env-file", env_file(tmp_path, "FRAMES=16\n"))
    assert json.loads(arg(d, "--limit-mm-per-prompt")) == {"video": {"num_frames": 16}}
    assert json.loads(arg(d, "--media-io-kwargs")) == {"video": {"num_frames": 16}}


def test_variant_gated_knob(w, tmp_path):
    assert "applies only with variant spec" in w.refused("--env-file", env_file(tmp_path, "SPEC_N=5\n"))
    d = w.resolve("--env-file", env_file(tmp_path, "SPEC_MODE=spec\nSPEC_N=5\nARGMAX=1\n"))
    assert json.loads(arg(d, "--speculative-config")) == {
        "method": "mtp", "num_speculative_tokens": 5, "use_local_argmax": True}
    assert d["variants"] == ["spec"]


def test_value_to_variant_map(w, tmp_path):
    assert w.resolve("--env-file", env_file(tmp_path, "SPEC_MODE=none\n"))["variants"] == []
    assert "expected one of" in w.refused("--env-file", env_file(tmp_path, "SPEC_MODE=eagle\n"))
    d = w.resolve("--env-file", env_file(tmp_path, "FAIR=0\n"))
    assert "--prefill-chunk-with-decodes" not in d["argv"] and "--max-num-partial-prefills" not in d["argv"]
    assert "--prefill-chunk-with-decodes" in w.resolve()["argv"]


def test_bool_knob_rejects_other_values(w, tmp_path):
    msg = w.refused("--env-file", env_file(tmp_path, "SPEC_MODE=spec\nARGMAX=maybe\n"))
    assert "expected 1 or 0" in msg


def test_runtime_specific_location(w):
    d = w.resolve()
    assert d["child_env"]["FLASHINFER_WORKSPACE_BASE"]["value"] == os.path.join(w.home, "fi-cache")
    c = w.resolve("--runtime", "container", "--tree", w.tree)
    assert c["child_env"]["FLASHINFER_WORKSPACE_BASE"]["value"] == "/cache"


def test_container_dir(w, tmp_path):
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        'dir = "synth-dense-1b"', 'dir = "synth-dense-1b"\ncontainer_dir = "Synth-Dense-1B"')
    write(os.path.join(w.settings, "model.toml"), model)
    c = w.resolve("--runtime", "container", "--tree", w.tree)
    assert c["argv"][2] == os.path.join(w.models, "Synth-Dense-1B")
    assert w.resolve()["argv"][2] == os.path.join(w.models, "synth-dense-1b")
