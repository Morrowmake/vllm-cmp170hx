# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A variant's declarations about a plain-upstream engine tree: [passthrough]
environment names its packages read but its vllm/envs.py does not declare
reach the server as functional switches (never dropped); [absent_args] names
a fixed or controlled argument its CLI does not declare. Each with a reason."""

from __future__ import annotations

import os

from helpers import World, write

# An engine CLI without --max-num-seqs (a controlled argument of the model).
FOREIGN_ARG_UTILS = """\
class AsyncEngineArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--max-model-len", type=int)
        parser.add_argument("--gpu-memory-utilization", type=float)
"""

ABSENT = '[absent_args]\nmax-num-seqs = "The engine\'s CLI declares no such option."\n'

FOREIGN = """\
schema = "launch-variant/1"
unset = ["env.VLLM_THIN_GEMM"]
[env]
VLLM_PLUGIN_SWITCH = "1"
MODEL_OWN_SWITCH = "2"
[passthrough]
VLLM_PLUGIN_SWITCH = "Read by the engine's plugin package, not by its vllm/envs.py."
MODEL_OWN_SWITCH = { why = "Read by the engine's model code." }
""" + ABSENT


def world(tmp_path, variant: str = FOREIGN, *, sm80: bool = False) -> World:
    w = World(tmp_path, sm80=sm80, extra_files={"vllm/engine/arg_utils.py": FOREIGN_ARG_UTILS})
    write(os.path.join(w.settings, "variants", "foreign.toml"), variant)
    return w


def test_declared_names_reach_the_server_as_functional_switches(tmp_path):
    w = world(tmp_path)
    d = w.resolve("--capability", "plain-upstream", "--variant", "foreign")
    for name, value in (("VLLM_PLUGIN_SWITCH", "1"), ("MODEL_OWN_SWITCH", "2")):
        assert d["child_env"][name] == {"value": value, "class": "functional"}
        assert d["env_functional"][name] == value
    assert d["passthrough"] == {
        "VLLM_PLUGIN_SWITCH": "Read by the engine's plugin package, not by its vllm/envs.py.",
        "MODEL_OWN_SWITCH": "Read by the engine's model code."}
    assert d["absent_args"] == {"max-num-seqs": "The engine's CLI declares no such option."}
    assert d["fork_switches_absent"] == ["VLLM_FAST_PATH"]      # the layout's, still reported


def test_without_the_declarations_the_names_are_dropped_or_refused(tmp_path):
    w = world(tmp_path / "a", FOREIGN.split("[passthrough]")[0] + ABSENT)
    assert "MODEL_OWN_SWITCH is class unclassified" in \
        w.refused("--capability", "plain-upstream", "--variant", "foreign")
    w = world(tmp_path / "b", FOREIGN.split("[absent_args]")[0])
    assert "full option names of this engine tree; not max-num-seqs" in \
        w.refused("--capability", "plain-upstream", "--variant", "foreign")


def test_declarations_are_refused_outside_their_contract(tmp_path):
    w = world(tmp_path / "sm80", sm80=True)
    assert "[passthrough] applies only under capability plain-upstream" in w.refused("--variant", "foreign")
    cases = {
        'VLLM_PORT = "1"\n[passthrough]\nVLLM_PORT = "x"\n' + ABSENT: "VLLM_PORT needs no [passthrough]",
        'LD_PRELOAD = "x.so"\n[passthrough]\nLD_PRELOAD = "x"\n' + ABSENT: "LD_PRELOAD is class code-loading",
        'A_SWITCH = "1"\n[passthrough]\nA_SWITCH = ""\n' + ABSENT: "[passthrough] A_SWITCH needs its reason",
        '[passthrough]\nNEVER_SET = "x"\n' + ABSENT: "[passthrough] NEVER_SET: no settings layer sets it",
        '[absent_args]\nmax-model-len = "x"\n': "[absent_args] max-model-len: this engine tree declares it",
        '[absent_args]\nmax-num-seqs = "x"\nblock-size = "x"\n':
            "[absent_args] block-size: not a fixed or controlled argument",
    }
    for i, (body, want) in enumerate(cases.items()):
        w = world(tmp_path / str(i), 'schema = "launch-variant/1"\n[env]\n' + body)
        assert want in w.refused("--capability", "plain-upstream", "--variant", "foreign"), body
