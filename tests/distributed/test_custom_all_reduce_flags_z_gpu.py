# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 bits of VLLM_SM80_ALLREDUCE_COMPRESS (skip with fewer than 4 GPUs).

Runs flags_z_bits.py on four ranks (torchrun) under the served DeepSeek-V4.1
all-reduce settings with the switch on (flags limit 512 KiB, one-shot up
to 8 KiB, raw cut-off 32 KiB, VLLM_CUSTOM_ALLREDUCE_ALGO=2stage) and judges
its JSON: on every captured served call of the all-reduce search's harness
(when its index is present: $VLLM_TEST_AR_REAL_INDEX, default
~/kernels/ampere-kernels/dsv41_allreduce_tp4/data/real_index.json, 96 calls),
on composed and generated messages at every size and in CUDA-graph replays,
the switch-on dispatch, the compressed kernel and the one-shot kernel give
the incumbent's bits (and the live output's), inputs stay unchanged, every
rank agrees and no wait times out. Every route must occur, and the
compressed kernel must replace upstream's two-stage kernel somewhere.

The ranks build the fork's JIT all-reduce extension in their own cache
(VLLM_CACHE_ROOT under tmp_path).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parent / "flags_z_bits.py"
TREE = SCRIPT.parents[2]
TIMEOUT_S = 1800
REAL_INDEX = Path(os.environ.get(
    "VLLM_TEST_AR_REAL_INDEX",
    os.path.expanduser("~/kernels/ampere-kernels/dsv41_allreduce_tp4/data/real_index.json"),
))
REAL_CALLS = 96
SERVED_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_P2P_LEVEL": "SYS",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "VLLM_CUSTOM_ALLREDUCE_ALGO": "2stage",
    "VLLM_SM80_ALLREDUCE": "1",
    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": "1",
    "VLLM_SM80_ALLREDUCE_COMPRESS": "1",
}
# The served DeepSeek-V4.1 rows, keyed by world size only: the script's
# communicator is built without a model config (no hidden size in the key).
TABLE = """schema = "sm80-tuning/1"
card = "test"

[[row]]
op = "allreduce"
param = "flags_max_bytes"
key = {world_size = 4}
value = 524288
evidence = "test: the DeepSeek-V4.1 row"

[[row]]
op = "allreduce"
param = "flags_raw_max_bytes"
key = {world_size = 4}
value = 32768
evidence = "test: the DeepSeek-V4.1 row"

[[row]]
op = "allreduce"
param = "flags_one_shot_max_bytes"
key = {world_size = 4}
value = 8192
evidence = "test: the DeepSeek-V4.1 row"
"""

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        reason="needs 4 GPUs",
    ),
]


def test_compressed_allreduce_bits_tp4(tmp_path):
    out = tmp_path / "z_bits.json"
    table = tmp_path / "served.toml"
    table.write_text(TABLE)
    env = dict(os.environ, **SERVED_ENV)
    env["VLLM_SM80_TUNING_FILE"] = str(table)
    env["VLLM_CACHE_ROOT"] = str(tmp_path / "vllm-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TREE), env.get("PYTHONPATH", "")) if p
    )
    argv = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node", "4", str(SCRIPT), "--out", str(out),
    ]  # fmt: skip
    if REAL_INDEX.is_file():
        argv += ["--real", str(REAL_INDEX)]
    proc = subprocess.run(
        argv, cwd=TREE, env=env, capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    log = (proc.stdout + proc.stderr)[-6000:]
    assert proc.returncode == 0, f"4-rank run exited {proc.returncode}:\n{log}"
    data = json.loads(out.read_text())
    config = data["config"]
    assert config["flags_max_bytes"] == 524288 and config["z_min_bytes"] == 32768
    assert config["one_shot_max_bytes"] == 8192
    assert config["z_upstream_from"] == 524304
    assert config["z_range_max_bytes"] == config["z_max_bytes"] == 8 * 1024 * 1024 - 16
    if REAL_INDEX.is_file():
        assert data["real_calls"] == REAL_CALLS, log
    assert data["no_errors"], f"a wait timed out:\n{log}"
    assert not data["bad"], f"{len(data['bad'])}+ cases differ:\n{data['bad'][:5]}\n{log}"
    assert data["z_for_upstream"] > 0, "no message took the compressed kernel above the flags limit"
    assert data["one_shot"] > 0 and data["one_forced"] > 0, "no message took the one-shot kernel"
    assert data["pass"] is True, log
    assert "Z_BITS PASS" in proc.stdout, log
    routes = {line.split()[0]: line for line in proc.stdout.splitlines() if " routes " in line}
    for route in ("'z':", "'flags':", "'one_shot':"):
        assert route in routes["randn"], log
