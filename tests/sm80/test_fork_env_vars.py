# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's environment variables (vllm_sm80/envs.py) as vLLM sees them once
the plugin registered them, and in a process where no entry point ran."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import vllm.envs as envs

import vllm_sm80
from vllm_sm80.envs import FORK_ENV_VARS
from vllm_sm80.plugin import register_fork_env_vars

register_fork_env_vars()


class _ImportTimeReads(ast.NodeVisitor):
    """Fork variables read as ``envs.NAME`` outside any function body: in
    module and class bodies, decorators and default arguments."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        defaults = [d for d in node.args.kw_defaults if d is not None]
        for expr in [*node.decorator_list, *node.args.defaults, *defaults]:
            self.visit(expr)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "envs"
            and node.attr in FORK_ENV_VARS
        ):
            self.names.add(node.attr)
        self.generic_visit(node)


def _import_time_readers() -> list[str]:
    root = Path(vllm_sm80.__file__).parent
    modules = []
    for path in sorted(root.rglob("*.py")):
        reads = _ImportTimeReads()
        reads.visit(ast.parse(path.read_text(encoding="utf-8")))
        if reads.names:
            parts = path.relative_to(root).with_suffix("").parts
            parts = parts[:-1] if parts[-1] == "__init__" else parts
            modules.append(".".join(("vllm_sm80", *parts)))
    return modules


IMPORT_TIME_READERS = _import_time_readers()

# Where a production launcher script lives, if the caller wants it checked.
SERVE_SCRIPT_ENV = "GLM5_SERVE_SH"

# Whole names only: the serving-defaults constants embed a name in a longer one
# (GLM_DEFAULT_VLLM_...), which is not a variable the process sees.
_VLLM_NAME = re.compile(r"(?<![A-Z0-9_])VLLM_[A-Z0-9_]+")


def test_plugin_registers_the_fork_variables():
    assert set(FORK_ENV_VARS) <= set(envs.environment_variables)
    register_fork_env_vars()  # a second entry point registers nothing twice
    assert all(envs.environment_variables[n] is g for n, g in FORK_ENV_VARS.items())


def test_the_import_time_readers_are_found():
    # Reached from upstream's quantization config table through the
    # DeepSeek V4.1 model code, whatever the checkpoint.
    assert "vllm_sm80.models.deepseek_v41.ops.sm80_mhc_side" in IMPORT_TIME_READERS


@pytest.mark.parametrize("module", IMPORT_TIME_READERS)
def test_a_module_reading_fork_variables_at_import_needs_no_entry_point(module):
    """An upstream file may import the module where VLLM_PLUGINS keeps the
    plugin's entry points from running: the module registers the variables
    it reads itself."""
    env = dict(
        os.environ,
        VLLM_PLUGINS="some_other_plugin",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]


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
