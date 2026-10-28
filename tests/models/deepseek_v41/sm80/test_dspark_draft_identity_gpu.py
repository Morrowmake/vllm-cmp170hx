# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 draft-ID equality, including cross-rank ties and CUDA graphs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from . import dspark_draft_identity as draft
from .test_wo_b_allreduce_bits_gpu import SERVED_ENV

SCRIPT = Path(draft.__file__).resolve()
TREE = SCRIPT.parents[4]
pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                                 reason="needs 4 GPUs")]


def test_actual_draft_ids_match_tp4(tmp_path):
    out = tmp_path / "draft_identity.json"
    env = dict(os.environ, **SERVED_ENV)
    env["VLLM_CACHE_ROOT"] = str(tmp_path / "cache")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(TREE), env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node", "4", str(SCRIPT), "--out", str(out)],
                          cwd=TREE, env=env, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-8000:]
    data = json.loads(out.read_text())
    assert data["verdict"] == "PASS" and data["world"] == 4
    assert set(data["modes"]) == {"eager", "graph"}
    draft.assert_identity(data["records"], 4, ("eager", "graph"))
