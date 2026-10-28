# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cases test_gpu_lock_guard.py runs in a child pytest (not collected on their
own). Neither touches a card: the guard around them is what is tested. Each
appends its name and the CUDA_VISIBLE_DEVICES it sees to the file named by
GPU_LOCK_GUARD_CASES_OUT."""

import os
from pathlib import Path

import pytest


def _seen(name: str) -> None:
    with Path(os.environ["GPU_LOCK_GUARD_CASES_OUT"]).open("a") as out:
        out.write(f"{name} {os.environ.get('CUDA_VISIBLE_DEVICES')!r}\n")


def test_cpu_case():
    _seen("cpu")


@pytest.mark.cuda
def test_cuda_case():
    _seen("cuda")
