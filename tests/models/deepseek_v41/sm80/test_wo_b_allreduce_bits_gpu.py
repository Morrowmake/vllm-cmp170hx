# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 bits of VLLM_SM80_ALLREDUCE_ROUND_ONCE on the wo_b partial (skip with
fewer than 4 GPUs).

Runs wo_b_allreduce_bits.py on four ranks (torchrun, 3 seeds, 20 timing
iterations) under the served all-reduce settings with the switch on, and
judges its JSON: every message the flags-in-data or the custom all-reduce
kernel takes whole is all-reduced in BF16 and gives the bits of the FP32
all-reduce. Every message NCCL would take is all-reduced in BF16 through the
custom all-reduce in pieces and gives the FP32 bits, or where an inexact FP32
sum rounds differently is no less accurate against the FP64 sum in mismatch
count and largest ulp distance. All three backends must occur over the row
counts the script sweeps.

The ranks build the fork's JIT all-reduce extensions in their own cache
(VLLM_CACHE_ROOT under tmp_path) with the interpreter's torch, so the run
never rewrites the extensions a served engine has loaded; the script puts
``ninja`` on PATH as the served venv has it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from . import wo_b_allreduce_bits as bits

SCRIPT = Path(bits.__file__).resolve()
TREE = SCRIPT.parents[4]
SEEDS = 3
ITERS = 20
TIMEOUT_S = 1200
# The served TP4 all-reduce settings the switch runs under, and the switch.
SERVED_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_P2P_LEVEL": "SYS",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "VLLM_CUSTOM_ALLREDUCE_ALGO": "2stage",
    "VLLM_SM80_ALLREDUCE": "1",
    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": "1",
    "VLLM_SM80_ALLREDUCE_ROUND_ONCE": "1",
}
BF16_BACKENDS = ("flags", "custom")
FP32_BACKENDS = ("nccl",)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        reason="needs 4 GPUs",
    ),
]


def test_wo_b_round_once_allreduce_bits_tp4(tmp_path):
    out = tmp_path / "wo_b_ar.json"
    env = dict(os.environ, **SERVED_ENV)
    env["VLLM_CACHE_ROOT"] = str(tmp_path / "vllm-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TREE), env.get("PYTHONPATH", "")) if p
    )
    argv = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node", "4", str(SCRIPT), "--out", str(out),
        "--seeds", str(SEEDS), "--iters", str(ITERS),
    ]  # fmt: skip
    proc = subprocess.run(
        argv,
        cwd=TREE,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    log = (proc.stdout + proc.stderr)[-6000:]
    assert proc.returncode == 0, f"4-rank run exited {proc.returncode}:\n{log}"
    data = json.loads(out.read_text())
    assert data["switch"] is True, "the run did not see the switch on"
    cases = data["cases"]
    assert sorted({c["rows"] for c in cases}) == sorted(bits.ROWS)
    assert data["world"] == 4
    assert {c["rank"] for c in cases} == set(range(4))
    assert len(cases) == 4 * len(bits.ROWS) * SEEDS
    assert data["switched_ranks_agree"] is True

    seen = {c["backend_bf16"] for c in cases}
    unknown = seen - set(BF16_BACKENDS) - set(FP32_BACKENDS)
    assert not unknown, f"messages took backends {sorted(unknown)}:\n{log}"
    missing = (set(BF16_BACKENDS) | set(FP32_BACKENDS)) - seen
    assert not missing, f"no row count reached {sorted(missing)}:\n{log}"

    for c in cases:
        where = f"rank {c['rank']} rows {c['rows']} seed {c['seed']} ({c['backend_bf16']})"
        assert c["switched_ranks_agree"], f"{where}: switched bits differ across ranks"
        if c["backend_bf16"] in BF16_BACKENDS:
            assert c["switched_path"] == "bf16", f"{where}: the switch kept FP32"
        else:
            assert c["switched_path"] == "bf16_pieces", f"{where}: not in pieces"
        if c["switched_path"] != "bf16_pieces":
            assert c["switched_eq_fp32"], f"{where}: switched bits differ from FP32"
        elif not c["switched_eq_fp32"]:
            assert c["switched_ne_exact"] <= c["fp32_ne_exact"], (
                f"{where}: more elements off the FP64 sum than FP32"
            )
            assert c["ulps_switched_exact"] <= c["ulps_fp32_exact"], (
                f"{where}: farther from the FP64 sum than FP32"
            )
    assert data["switched_no_less_accurate"] is True
