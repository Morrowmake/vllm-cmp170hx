# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cases test_gpu_lock_guard.py runs in a child pytest (skipped when a run
names this file without that test). Neither touches a card: the guard around
them is what is tested. Each appends its name and the CUDA_VISIBLE_DEVICES it
sees to the file named by GPU_LOCK_GUARD_CASES_OUT."""

import os
from pathlib import Path

import pytest

if "GPU_LOCK_GUARD_CASES_OUT" not in os.environ:
    pytest.skip(
        "cases of test_gpu_lock_guard.py, which runs them in a child pytest",
        allow_module_level=True,
    )


def _seen(name: str) -> None:
    with Path(os.environ["GPU_LOCK_GUARD_CASES_OUT"]).open("a") as out:
        out.write(f"{name} {os.environ.get('CUDA_VISIBLE_DEVICES')!r}\n")


def test_cpu_case():
    _seen("cpu")


@pytest.mark.cuda
def test_cuda_case():
    _seen("cuda")
