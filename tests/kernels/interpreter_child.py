# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a Triton-interpreter test module in a child process when it has to.

TRITON_INTERPRET takes effect only if it is set before Triton is imported, and
the repository conftest imports Triton (through vLLM) before any test module,
so a module cannot switch its own process to the interpreter. A module whose
tests need the interpreter ends with ``run_in_interpreter_child(globals())``.
With the interpreter active (TRITON_INTERPRET=1 exported) that does nothing.
Otherwise each test of the module reports the outcome of the same test in one
child pytest run of the module, with TRITON_INTERPRET=1 and no visible GPU.
"""

import functools
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest


def interpreter_active() -> bool:
    """Triton was imported with the interpreter on (its own library is
    interpreted, not compiled)."""
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction

    return isinstance(tl.standard.max, InterpretedFunction)


def _run_child(path: str) -> dict[str, tuple[str, str]]:
    """Run the module at ``path`` under the interpreter; test name ->
    (status, message)."""
    import vllm

    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    env = dict(
        os.environ,
        TRITON_INTERPRET="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(
            p for p in (root, os.environ.get("PYTHONPATH")) if p
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "child.xml")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider",
             f"--junitxml={report}"],
            cwd=root, env=env, capture_output=True, text=True, timeout=1800)
        if not os.path.exists(report):
            raise RuntimeError(
                f"interpreter child exited {r.returncode} without a report:\n"
                + r.stdout[-3000:] + r.stderr[-6000:])
        outcomes = {}
        for case in ET.parse(report).iter("testcase"):
            outcome = ("passed", "")
            for tag in ("failure", "error", "skipped"):
                el = case.find(tag)
                if el is not None:
                    outcome = (tag, f"{el.get('message', '')}\n{el.text or ''}")
                    break
            outcomes[case.get("name")] = outcome
    return outcomes


def run_in_interpreter_child(namespace: dict) -> None:
    """Replace the module's tests by reports from the interpreter child,
    unless this process already runs the interpreter."""
    if interpreter_active():
        return
    path = namespace["__file__"]
    cache: dict = {}

    def outcome(name: str) -> tuple[str, str]:
        if not cache:
            try:
                cache["outcomes"] = _run_child(path)
            except (OSError, RuntimeError, subprocess.SubprocessError,
                    ET.ParseError) as e:
                cache["error"] = f"{type(e).__name__}: {e}"
        if "error" in cache:
            pytest.fail(cache["error"], pytrace=False)
        if name not in cache["outcomes"]:
            pytest.fail(f"{name} did not run in the interpreter child",
                        pytrace=False)
        return cache["outcomes"][name]

    def delegate(fn):
        @functools.wraps(fn)
        def in_child(*args, **kwargs):
            # "<file>::<name>[<params>] (call)"
            current = os.environ["PYTEST_CURRENT_TEST"].rsplit(" ", 1)[0]
            status, text = outcome(current.split("::")[-1])
            if status == "skipped":
                pytest.skip(text)
            if status != "passed":
                pytest.fail(f"in the interpreter child ({status}): {text}",
                            pytrace=False)

        return in_child

    for name, obj in list(namespace.items()):
        if name.startswith("test_") and callable(obj):
            namespace[name] = delegate(obj)
