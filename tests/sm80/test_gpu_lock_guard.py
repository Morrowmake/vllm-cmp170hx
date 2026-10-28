# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests run only under the GPU lock (gpu_lock_guard.py, called by
tests/conftest.py). Outside the lock a run that leaves a card visible refuses
the tests marked ``cuda`` at collection, a run that hides the cards skips them,
and CPU tests run either way with no card visible; inside the lock everything
runs and CUDA_VISIBLE_DEVICES is left alone.

Each case is a child pytest over gpu_lock_guard_cases.py, whose tests never
touch a card; the visible device given is not a real card's name.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.sm80 import gpu_lock_guard as guard

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = HERE / "gpu_lock_guard_cases.py"
NOT_A_CARD = "GPU-00000000-0000-0000-0000-000000000000"


def run_cases(tmp_path, *args, **env):
    """The child pytest over CASES with ENV (None: unset) on top of this
    environment without the lock, the guard's record or CUDA_VISIBLE_DEVICES.
    Returns (result, {case: CUDA_VISIBLE_DEVICES it saw})."""
    out = tmp_path / "seen.txt"
    drop = (guard.LOCK_ENV, guard.CALLER_ENV, "CUDA_VISIBLE_DEVICES")
    child = {k: v for k, v in os.environ.items() if k not in drop}
    child.update({k: v for k, v in env.items() if v is not None})
    child["GPU_LOCK_GUARD_CASES_OUT"] = str(out)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
         str(CASES), *args],
        cwd=REPO, env=child, capture_output=True, text=True, timeout=900,
    )
    seen = dict(line.split(" ", 1) for line in out.read_text().splitlines()) \
        if out.exists() else {}
    return result, seen


@pytest.mark.parametrize("visible", [NOT_A_CARD, None], ids=["named", "unset"])
@pytest.mark.parametrize("workers", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_outside_the_lock_a_visible_card_refuses_cuda_tests(tmp_path, visible, workers):
    result, seen = run_cases(tmp_path, *workers, CUDA_VISIBLE_DEVICES=visible)
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert guard.RULE in text and "test_cuda_case" in text, text
    assert seen == {}, seen   # refused at collection: nothing ran


def test_outside_the_lock_cpu_tests_run_without_a_card(tmp_path):
    result, seen = run_cases(tmp_path, "-m", "not cuda", CUDA_VISIBLE_DEVICES=NOT_A_CARD)
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen == {"cpu": "''"}, seen


def test_outside_the_lock_hidden_cards_skip_cuda_tests(tmp_path):
    result, seen = run_cases(tmp_path, CUDA_VISIBLE_DEVICES="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen == {"cpu": "''"}, seen
    assert "1 passed, 1 skipped" in result.stdout and guard.RULE in result.stdout, result.stdout


def test_inside_the_lock_everything_runs_untouched(tmp_path):
    result, seen = run_cases(tmp_path, GPU_LOCK_TOKENS="lock-token",
                             CUDA_VISIBLE_DEVICES=NOT_A_CARD)
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen == {"cpu": repr(NOT_A_CARD), "cuda": repr(NOT_A_CARD)}, seen
