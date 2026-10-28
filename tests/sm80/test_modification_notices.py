# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every upstream source file the fork modifies states that it was modified.

Apache-2.0 section 4(b) asks modified files to carry a prominent notice. The
fork's notice is a 'Modified by' line near the top, for example
`# Modified by Morrowmake for CMP 170HX support; see repository history.`
as line 3 of a Python file, after the two SPDX lines.

Checked: files that exist at the upstream base in FORK.md and differ from it
in the working tree (git status M), in the product tree (not tests/, docs/ or
.github/), whose kind is source or build code (Python, CUDA, C/C++ and CMake).
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FORK_MD = REPO / "FORK.md"
NOTICE = "Modified by"
HEAD_LINES = 15
NON_PRODUCT_DIRS = ("tests/", "docs/", ".github/")
SOURCE_SUFFIXES = frozenset(
    {".py", ".cu", ".cuh", ".h", ".hpp", ".c", ".cc", ".cpp", ".cmake"}
)
SOURCE_NAMES = frozenset({"CMakeLists.txt"})


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )


def _upstream_base() -> str:
    match = re.search(
        r"^Upstream base: `?([0-9a-f]{7,40})`?[ \t]*$",
        FORK_MD.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "FORK.md has no 'Upstream base: <sha>' line"
    return match.group(1)


def _needs_notice(path: str) -> bool:
    if path.startswith(NON_PRODUCT_DIRS):
        return False
    name = Path(path)
    return name.name in SOURCE_NAMES or name.suffix in SOURCE_SUFFIXES


def test_modified_upstream_sources_carry_a_notice():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip(f"{REPO} is not a git checkout")
    assert FORK_MD.is_file(), "FORK.md is missing"
    base = _upstream_base()
    if _git("cat-file", "-e", f"{base}^{{commit}}").returncode != 0:
        pytest.skip(f"upstream base {base} is not in this clone (shallow or not fetched)")
    proc = _git(
        "diff", "--name-only", "--no-renames", "--diff-filter=M", "-z", base, "--"
    )
    assert proc.returncode == 0, proc.stderr
    modified = sorted(p for p in proc.stdout.split("\0") if p and _needs_notice(p))
    assert modified, f"no modified upstream source files found against {base}"
    missing = []
    for path in modified:
        with open(REPO / path, encoding="utf-8", errors="replace") as f:
            head = [f.readline() for _ in range(HEAD_LINES)]
        if not any(NOTICE in line for line in head):
            missing.append(path)
    assert not missing, (
        f"{len(missing)} of {len(modified)} upstream files modified since {base} "
        f"have no '{NOTICE}' line in their first {HEAD_LINES} lines:\n"
        + "\n".join(missing)
    )
