# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""setup.py must load vLLM's environment variables without an importable vllm.

setup.py loads vllm/envs.py by file path, under the module name ``envs``. An
isolated PEP 517 build does not put the source tree on sys.path, so vllm/envs.py
registers the fork's variables (vllm_sm80/envs.py) only when it is imported as
``vllm.envs``. This runs setup.py's own loading statements in a clean
interpreter (-I: no PYTHONPATH, no script or working directory on sys.path) and
checks that every variable setup.py reads resolves without importing the vllm
package, and that the fork's variables are left out there.
"""

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "setup.py"


def _loading_statements() -> str:
    """load_module_from_path, ROOT_DIR and every statement that loads an envs.py."""
    source = SETUP.read_text(encoding="utf-8")
    keep = []
    for node in ast.parse(source).body:
        text = ast.get_source_segment(source, node)
        if isinstance(node, ast.FunctionDef) and node.name == "load_module_from_path":
            keep.append(text)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ROOT_DIR" for t in node.targets
        ):
            keep.append(text)
        elif "load_module_from_path(" in text and '"envs.py"' in text:
            keep.append(text)
    return "\n".join(keep)


def test_setup_loads_env_vars_without_vllm_package(tmp_path):
    statements = _loading_statements()
    read = sorted(set(re.findall(r"\benvs\.(VLLM_[A-Z0-9_]+)", SETUP.read_text())))
    assert read, "setup.py reads no envs.VLLM_* variable"
    script = tmp_path / "load_envs.py"
    script.write_text(
        "import importlib.util\nimport os\nimport sys\nfrom pathlib import Path\n"
        f"__file__ = {str(SETUP)!r}\n"
        f"READ = {read!r}\n"
        + statements
        + textwrap.dedent(
            """
            assert not [m for m in sys.modules if m.split(".")[0] == "vllm"], sorted(sys.modules)
            for name in READ:
                getattr(envs, name)
            print("FORK_VARS", "VLLM_AMPERE_MARLIN_DECODE_CUDA" in envs.environment_variables)
            """
        )
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_", "PYTHON"))
    }
    proc = subprocess.run(
        [sys.executable, "-I", str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FORK_VARS False" in proc.stdout, proc.stdout + proc.stderr


def test_package_import_registers_the_fork_variables():
    import vllm.envs as envs
    from vllm_sm80.envs import FORK_ENV_VARS

    assert set(FORK_ENV_VARS) <= set(envs.environment_variables)
