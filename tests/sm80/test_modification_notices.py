# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every upstream source file the fork modifies states that it was modified.

Apache-2.0 section 4(b) asks modified files to carry a prominent notice. The
fork's notice is a 'Modified by' line near the top, for example
`# Modified by Morrowmake for CMP 170HX support; see repository history.`
as line 3 of a Python file, after the two SPDX lines.

Checked: files that exist at the upstream base and differ from it in the tree
that holds the upstream-owned files (git status M), in the product tree (not
tests/, docs/ or .github/), whose kind is source or build code (Python, CUDA,
C/C++ and CMake). That tree and base are, in an assembled engine (the vLLM
the tests import is <dir>/engine next to <dir>/assembly.json), the engine
directory and the assembly's `upstream_commit`; in a single tree, the
repository root and the base in FORK.md.
"""
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "sm80_update_seams", _HERE / "update_seams.py"
)
seams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seams)

REPO = Path(__file__).resolve().parents[2]
FORK_MD = REPO / "FORK.md"
NOTICE = "Modified by"
HEAD_LINES = 15
NON_PRODUCT_DIRS = ("tests/", "docs/", ".github/")
SOURCE_SUFFIXES = frozenset(
    {".py", ".cu", ".cuh", ".h", ".hpp", ".c", ".cc", ".cpp", ".cmake"}
)
SOURCE_NAMES = frozenset({"CMakeLists.txt"})


def _git(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(tree), *args], capture_output=True, text=True
    )


def _fork_md_base() -> str:
    assert FORK_MD.is_file(), "FORK.md is missing"
    match = re.search(
        r"^Upstream base: `?([0-9a-f]{7,40})`?[ \t]*$",
        FORK_MD.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "FORK.md has no 'Upstream base: <sha>' line"
    return match.group(1)


def _checked_tree() -> tuple[Path, str]:
    """The tree holding the upstream-owned files, and its upstream base."""
    engine = seams.assembled_engine()
    if engine is not None:
        return engine
    return REPO, _fork_md_base()


def _needs_notice(path: str) -> bool:
    if path.startswith(NON_PRODUCT_DIRS):
        return False
    name = Path(path)
    return name.name in SOURCE_NAMES or name.suffix in SOURCE_SUFFIXES


def missing_notices(tree: Path, base: str) -> tuple[list[str], list[str]]:
    """(modified, missing): the upstream source files `tree` modifies against
    `base`, and those of them without a notice in their first lines."""
    proc = _git(
        tree, "diff", "--name-only", "--no-renames", "--diff-filter=M", "-z", base, "--"
    )
    assert proc.returncode == 0, proc.stderr
    modified = sorted(p for p in proc.stdout.split("\0") if p and _needs_notice(p))
    missing = []
    for path in modified:
        with open(tree / path, encoding="utf-8", errors="replace") as f:
            head = [f.readline() for _ in range(HEAD_LINES)]
        if not any(NOTICE in line for line in head):
            missing.append(path)
    return modified, missing


def test_modified_upstream_sources_carry_a_notice():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    tree, base = _checked_tree()
    if _git(tree, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip(f"{tree} is not a git checkout")
    if _git(tree, "cat-file", "-e", f"{base}^{{commit}}").returncode != 0:
        pytest.skip(f"upstream base {base} is not in this clone (shallow or not fetched)")
    modified, missing = missing_notices(tree, base)
    assert modified, f"no modified upstream source files found in {tree} against {base}"
    assert not missing, (
        f"{len(missing)} of {len(modified)} upstream files modified since {base} "
        f"have no '{NOTICE}' line in their first {HEAD_LINES} lines:\n"
        + "\n".join(missing)
    )


def test_an_assembled_engine_is_checked_in_its_engine_directory(tmp_path):
    """The engine directory beside assembly.json, at the assembly's upstream
    commit; a vLLM without that layout is a single tree."""
    engine = tmp_path / "engine"
    (engine / "vllm").mkdir(parents=True)
    origin = engine / "vllm" / "__init__.py"
    origin.write_text("")
    assert seams.assembled_engine(str(origin)) is None
    (tmp_path / "assembly.json").write_text(json.dumps({"upstream_commit": "c" * 40}))
    assert seams.assembled_engine(str(origin)) == (engine.resolve(), "c" * 40)
    single = tmp_path / "tree" / "vllm"
    single.mkdir(parents=True)
    (single / "__init__.py").write_text("")
    assert seams.assembled_engine(str(single / "__init__.py")) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_missing_notices_reads_the_tree_it_is_given(tmp_path):
    """Product source files modified against the base, without a notice; new
    files and test files are not checked."""

    def git(*args):
        proc = _git(
            tmp_path,
            "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null",
            "-c", "commit.gpgSign=false", *args,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    files = {"vllm/a.py": "a = 1\n", "vllm/b.py": "b = 1\n", "tests/t.py": "t = 1\n"}
    git("init", "-q")
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "vllm/a.py").write_text("# Modified by us.\na = 2\n")
    (tmp_path / "vllm/b.py").write_text("b = 2\n")
    (tmp_path / "tests/t.py").write_text("t = 2\n")
    (tmp_path / "vllm/new.py").write_text("n = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "change")
    assert missing_notices(tmp_path, base) == (["vllm/a.py", "vllm/b.py"], ["vllm/b.py"])
