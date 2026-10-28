# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import dspark_draft_identity as draft


def _records():
    return [dict(rank=r, case=c, requests=n, steps=k, mode=m, on=on,
                 draft_ids=[[[0] * k for _ in range(n)] for _ in range(2)],
                 confidence=[[[0.0] * k for _ in range(n)] for _ in range(2)],
                 target_tokens=[42], expected_tie_id=0)
            for r in range(4) for c in draft.CASES for n, k in draft.SHAPES
            for m in ("eager", "graph") for on in (False, True)]


def test_changed_draft_ids_fail_even_when_target_outputs_are_identical():
    records = _records()
    draft.assert_identity(records, 4, ("eager", "graph"))
    target = copy.deepcopy([r["target_tokens"] for r in records])
    for repeat in records[-1]["draft_ids"]:
        repeat[0][0] = 1
    assert [r["target_tokens"] for r in records] == target
    with pytest.raises(AssertionError, match="draft IDs differ"):
        draft.assert_identity(records, 4, ("eager", "graph"))


def test_draft_gate_requires_every_rank_and_repeat():
    records = _records()
    with pytest.raises(AssertionError, match="coverage"):
        draft.assert_identity(records[:-1], 4, ("eager", "graph"))
    records[-1]["draft_ids"][1][0][0] = 1
    with pytest.raises(AssertionError, match="repeatability"):
        draft.assert_identity(records, 4, ("eager", "graph"))


def test_identical_wrong_tie_choices_still_fail():
    records = _records()
    for record in records:
        for repeat in record["draft_ids"]:
            for req in repeat:
                req[:] = [1] * len(req)
    with pytest.raises(AssertionError, match="lowest-index tie"):
        draft.assert_identity(records, 4, ("eager", "graph"))


def test_actual_draft_ids_match_on_gloo(tmp_path):
    script = Path(draft.__file__).resolve()
    root = script.parents[4]
    out = tmp_path / "draft_identity.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1", OMP_NUM_THREADS="1",
               PYTHONPATH=str(root))
    proc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node", "4", str(script), "--cpu", "--out", str(out)],
                          cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-8000:]
    data = json.loads(out.read_text())
    assert data["verdict"] == "PASS" and data["world"] == 4
    draft.assert_identity(data["records"], 4, ("eager",))
