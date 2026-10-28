# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The buffer-lifecycle contracts producer (``lifecycle_contracts.py``):
every group holds all its checks on a seed and writes a schema-1 report, and
a lifecycle broken in the code under test turns the matching checks false.

The producer runs in a subprocess, as its callers run it: it selects the
Triton interpreter before any kernel is defined.

    pytest -q tests/models/deepseek_v41/sm80/test_lifecycle_contracts.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PRODUCER = "tests.models.deepseek_v41.sm80.lifecycle_contracts"

# Rewrites one statement of a module, then runs the producer.
MUTATED = """
import importlib, sys
from tests.models.deepseek_v41.sm80 import lifecycle_contracts as producer
mod = importlib.import_module({module!r})
src = open(mod.__file__).read()
assert src.count({old!r}) == 1, "mutation target not found"
exec(compile(src.replace({old!r}, {new!r}), mod.__file__, "exec"), mod.__dict__)
sys.exit(producer.main(sys.argv[1:]))
"""


def run_producer(tmp_path: Path, group: str, seed: int, mutation=None):
    out = tmp_path / f"{group}-{seed}.json"
    if mutation is None:
        argv = [sys.executable, "-m", PRODUCER]
    else:
        module, old, new = mutation
        argv = [sys.executable, "-c", MUTATED.format(module=module, old=old, new=new)]
    argv += ["--group", group, "--seed", str(seed), "--out", str(out)]
    # ROOT for the producer's own package, then every import root of this
    # process: in an assembled engine vllm and vllm_sm80 are not under ROOT.
    path = os.pathsep.join([str(ROOT), *sys.path])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=path)
    proc = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.is_file(), proc.stdout + proc.stderr
    return proc.returncode, json.loads(out.read_text())


@pytest.mark.parametrize("group", ["compressor_ring", "paged_kv", "engram_staging"])
def test_group_holds(tmp_path: Path, group: str) -> None:
    rc, report = run_producer(tmp_path, group, 7)
    assert report["schema"] == 1 and report["group"] == group and report["seed"] == 7
    assert report["failures"] == [], report["failures"][:3]
    assert all(report["checks"].values()) and rc == 0
    assert report["events"] and all(e["input"] and e["transition"] for e in report["events"])
    assert all(v for k, v in report["identity"].items())
    # Every counter the history is built to reach.
    zero = sorted(k for k, v in report["coverage"].items() if v == 0)
    assert set(zero) <= {"window_blocks_freed"}, zero


CORE = "vllm_sm80.models.deepseek_v41.ops.sm80_attn_core"
MUTATIONS = {
    # The ring keeps no rows: a group closed across steps pools a stale row.
    "ring_not_written": (
        "compressor_ring",
        (
            CORE,
            "    store_rows(ring, torch.where(tail, ring_slots, -1), kv_score[:T])",
            "    pass",
        ),
        {"ring_carry_reads_own_row", "compressed_rows_match_uninterrupted"},
    ),
    # The top-k keeps one compressed position past the visible ones: a row
    # the request has not written yet, or another token's future row.
    "topk_reads_ahead": (
        "paged_kv",
        (
            CORE,
            "keep = torch.arange(width, device=scores.device)[None, :] < n_vis[:, None]",
            "keep = torch.arange(width, device=scores.device)[None, :] <= n_vis[:, None]",
        ),
        {"compressed_reads_owned"},
    ),
    # The host fetch is queued before the copy of the ids to the host: it
    # fetches the previous lookup's ids.
    "ids_copied_after_fetch": (
        "engram_staging",
        (
            "vllm_sm80.models.deepseek_v41.sm80_engram",
            "        store.ids[:count].copy_(self._dev_ids[:count], non_blocking=True)\n"
            "        store.enqueue(count)\n",
            "        store.enqueue(count)\n"
            "        store.ids[:count].copy_(self._dev_ids[:count], non_blocking=True)\n",
        ),
        {"staging_reuse_ordered"},
    ),
    # The rows are copied to the device before the host fetch fills them.
    "rows_copied_before_fetch": (
        "engram_staging",
        (
            "vllm_sm80.models.deepseek_v41.sm80_engram",
            "        store.enqueue(count)\n"
            "        dev_values = self._dev_values[:count]\n"
            "        dev_scales = self._dev_scales[:count]\n"
            "        dev_values.copy_(store.values[:count], non_blocking=True)\n"
            "        dev_scales.copy_(store.scales_out[:count], non_blocking=True)\n",
            "        dev_values = self._dev_values[:count]\n"
            "        dev_scales = self._dev_scales[:count]\n"
            "        dev_values.copy_(store.values[:count], non_blocking=True)\n"
            "        dev_scales.copy_(store.scales_out[:count], non_blocking=True)\n"
            "        store.enqueue(count)\n",
        ),
        {"staging_reuse_ordered"},
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_broken_lifecycle_fails(tmp_path: Path, name: str) -> None:
    group, mutation, broken = MUTATIONS[name]
    rc, report = run_producer(tmp_path, group, 7, mutation)
    failed = {k for k, v in report["checks"].items() if not v}
    assert rc == 1 and broken <= failed, failed
    assert {f["check"] for f in report["failures"]} == failed
