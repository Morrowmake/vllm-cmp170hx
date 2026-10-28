# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The batched sliding-window metadata of ``VLLM_DEEPSEEK_V41_SWA_META_BATCH``
(``nvidia/sm80_swa_meta_batch.py``, ``ops/sm80_swa_meta.py``) on CPU.

In the Triton interpreter (a subprocess with ``TRITON_INTERPRET=1``,
``swa_meta_batch_cases.py``), 43 window groups with their own block ids and
one other group go through the runner's ``build_attn_metadata`` per group
(switch off) and batched (switch on), on the scenarios of a served step: a
c1 verify step (6 rows), a c8 step (32 rows), c1 with two padding rows, a
step with prefill and decode rows (one request replaying), and requests
inside, at and across the window. For every group every field of the
metadata and every builder buffer is the per-group build's, byte for byte,
and the buffer views are the same buffers; the batched step launches one
kernel for the window part instead of the per-group ``torch.ge``, fill,
index and gather-length launches. A builder whose inputs changed since the
step that learned them, and DSpark's non-causal draft metadata, build per
group, and so does a group whose block table is a kernel-block table (filled
in the group's own build). A discarded layout (builders, block tables, slot
mappings) is freed once learned and launched for. The GPU half is
``test_sm80_swa_meta_batch_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_swa_meta_batch.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.models.deepseek_v41.sm80 import swa_meta_batch_cases as cases

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def interpreted() -> dict:
    env = dict(
        os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"
    )
    env.pop(cases.SWITCH, None)
    done = subprocess.run(
        [sys.executable, "-m", "tests.models.deepseek_v41.sm80.swa_meta_batch_cases"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


def test_batched_metadata_is_the_per_group_build_in_one_launch(interpreted):
    cases.check(interpreted)


def test_switch_default_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(cases.SWITCH, raising=False)
    assert envs.VLLM_DEEPSEEK_V41_SWA_META_BATCH is False
    monkeypatch.setenv(cases.SWITCH, "1")
    assert envs.VLLM_DEEPSEEK_V41_SWA_META_BATCH is True
