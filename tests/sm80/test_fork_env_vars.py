# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's environment variables (vllm_sm80/envs.py) as vLLM sees them once
the plugin registered them."""

import os
import re

import pytest
import vllm.envs as envs

from vllm_sm80.envs import FORK_ENV_VARS
from vllm_sm80.plugin import register_fork_env_vars

register_fork_env_vars()

# Where a production launcher script lives, if the caller wants it checked.
SERVE_SCRIPT_ENV = "GLM5_SERVE_SH"

# Whole names only: the serving-defaults constants embed a name in a longer one
# (GLM_DEFAULT_VLLM_...), which is not a variable the process sees.
_VLLM_NAME = re.compile(r"(?<![A-Z0-9_])VLLM_[A-Z0-9_]+")


def test_plugin_registers_the_fork_variables():
    assert set(FORK_ENV_VARS) <= set(envs.environment_variables)
    register_fork_env_vars()  # a second entry point registers nothing twice
    assert all(envs.environment_variables[n] is g for n, g in FORK_ENV_VARS.items())


def test_serve_script_uses_only_declared_variables():
    """Every VLLM_* name the launcher mentions must be declared.

    validate_environ() logs "Unknown vLLM environment variable detected" for
    every VLLM_* in the process environment that `environment_variables` has no
    entry for, on each worker at start. Some of the GLM-5.x flags are parsed by
    the code that consumes them (the C++ all-reduce kernel) instead of being
    read through `envs.X`, so declaring them is a separate step that is easy to
    forget; this is the check for it.

    Point GLM5_SERVE_SH at the launcher to run it. With the variable unset, or
    naming a file that is not there, the test skips.
    """
    path = os.environ.get(SERVE_SCRIPT_ENV, "")
    if not path or not os.path.isfile(path):
        pytest.skip(f"set {SERVE_SCRIPT_ENV} to a launcher script to run this")

    with open(path, encoding="utf-8") as f:
        names = set(_VLLM_NAME.findall(f.read()))
    assert names, f"{path} mentions no VLLM_* variable at all"

    undeclared = sorted(n for n in names if n not in envs.environment_variables)
    assert not undeclared, (
        f"{path} uses VLLM_* names that no envs table declares, so every"
        f" worker logs 'Unknown vLLM environment variable detected' for them at"
        f" start: {', '.join(undeclared)}"
    )


def test_custom_allreduce_algo_mirrors_the_kernel(monkeypatch: pytest.MonkeyPatch):
    """csrc/custom_all_reduce.cuh compares the raw bytes; so does the getter."""
    getter = envs.environment_variables["VLLM_CUSTOM_ALLREDUCE_ALGO"]

    monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
    assert getter() == ""

    for raw in ("1stage", "oneshot", "2stage", "twoshot"):
        monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raw)
        assert getter() == raw

    # No case folding and no stripping, because the kernel does neither.
    for raw in ("2STAGE", " 2stage", "3stage", ""):
        monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raw)
        with pytest.raises(ValueError, match="Valid values"):
            getter()
