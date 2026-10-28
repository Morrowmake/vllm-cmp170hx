# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The softmax statistics the per-rank DSpark argmax carries for the adaptive
depth policy (``shard_argmax.shard_stats`` / ``combine_stats``), in the
Triton interpreter (a subprocess with ``TRITON_INTERPRET=1``):

- every rank's statistics lanes are bytes (0 .. 255) in its own lanes only,
  and the chunk slots are those of the plain exchange;
- summed as the all-reduce sums them (FP32 then BF16, or BF16 rank by rank),
  the combined top-1 probability and entropy equal the gathered row's softmax
  statistics (float64 reference) to 2e-6 relative, at 1, 2 and 4 ranks, with
  small chunks, padding columns past the vocabulary, a rank without columns,
  -inf columns and peaked rows; the picked tokens are torch.argmax's;
- the combination writes the same bits from either sum.

    pytest -q tests/models/deepseek_v41/sm80/test_dspark_shard_argmax_stats.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[4]


def stats_cases() -> dict:
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from tests.models.deepseek_v41.sm80.test_dspark_shard_argmax import (
        _sum_slots,
        _sum_slots_bf16,
    )
    from vllm_sm80.spec_decode import shard_argmax as sa

    sa.BLOCK = 8
    g = torch.Generator().manual_seed(0)
    out = []
    for tp, width, vocab in ((1, 37, 37), (2, 20, 37), (4, 12, 37), (4, 12, 30)):
        rows = 6
        full = torch.randn(rows, vocab, generator=g) * 3.0
        full[1, :5] = float("-inf")
        full[2] *= 8.0  # peaked
        full[3] = 0.25  # flat
        chunks = -(-width // sa.BLOCK)
        base = tp * chunks * sa.LANES
        per_rank, lanes_ok, others_zero, plain_same = [], True, True, True
        for rank in range(tp):
            start = rank * width
            n = min(max(vocab - start, 0), width)
            cols = full[:, start : start + n].contiguous()
            slots = sa.chunk_slots(cols, start, rank, tp, chunks, stats=True)
            plain = sa.chunk_slots(cols, start, rank, tp, chunks)
            plain_same &= torch.equal(slots[:, :base], plain)
            lanes = slots.float()
            lanes_ok &= bool(torch.equal(lanes, lanes.round()) and lanes.min() >= 0
                             and lanes.max() <= 255)
            mine = slots[:, base:].view(rows, tp, sa.STATS_LANES)
            others = torch.cat([mine[:, :rank], mine[:, rank + 1 :]], dim=1)
            others_zero &= not others.float().any()
            per_rank.append(slots)
        got = []
        for total in (_sum_slots(per_rank), _sum_slots_bf16(per_rank)):
            res = torch.full((rows, 2), float("nan"))
            sa.combine_stats(total, base, tp, res)
            got.append(res)
            tokens = sa.pick(total, vocab, tp * chunks)
        p = torch.softmax(full.double(), dim=-1)
        top1 = p.max(-1).values
        ent = -torch.special.xlogy(p, p).sum(-1)
        out.append({
            "tp": tp, "lanes_ok": lanes_ok, "others_zero": others_zero,
            "plain_same": plain_same, "same_bits": torch.equal(got[0], got[1]),
            "top1": got[0][:, 0].tolist(), "ent": got[0][:, 1].tolist(),
            "want_top1": top1.tolist(), "want_ent": ent.tolist(),
            "tokens": tokens.tolist(), "argmax": full.argmax(-1).tolist(),
        })
    return {"cases": out}


@pytest.fixture(scope="module")
def cases():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_dspark_shard_argmax_stats "
            "import stats_cases as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])["cases"]


def test_lanes_are_bytes_in_the_ranks_own_lanes(cases):
    for c in cases:
        assert c["lanes_ok"] and c["others_zero"] and c["plain_same"], c["tp"]


def test_combined_statistics_are_the_gathered_softmax(cases):
    for c in cases:
        assert c["top1"] == pytest.approx(c["want_top1"], rel=2e-6), c["tp"]
        assert c["ent"] == pytest.approx(c["want_ent"], rel=2e-6, abs=2e-6), c["tp"]
        assert c["tokens"] == c["argmax"], c["tp"]


def test_both_sums_give_the_same_bits(cases):
    assert all(c["same_bits"] for c in cases)
