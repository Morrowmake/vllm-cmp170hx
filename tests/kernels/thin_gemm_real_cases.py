# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real captured inputs for the BF16 thin-GEMM tests.

The thin-GEMM counterpart of tests/kernels/quantization/exl3_real_cases.py
(same directory layout, sha256 checks and per-cell coverage), used by
test_ampere_thin_gemm_gluon_real_inputs.py.

Case set: the directory named by VLLM_THIN_GEMM_TEST_PORT_CASES.

    cases.json   {"schema": "thin_gemm_port_cases/1", "sm_count": 74,
                  "tp_size": 2, "tp_rank": 0, "cases": [case, ...]}
    x/<id>.pt    torch.save dict, read with weights_only=True: x [rows, K]
                 bf16 and, for live cases, out [rows, N] bf16
    w/<name>.pt  {"w": [N, K] bf16}, the layer's weight as the rank holds it

Every case: id, file, sha256 (of the file), weight_file, weight_sha256 (of
the weight file), shape (hc_down, hc_up, final_down, router, shared_gate,
gdn_ba, ple_kv), role (module prefix with the layer number replaced by "*"),
N, K, rows, phase ("decode", "prefill" or "mixed", from the scheduler step
that made the call, never from the row count), drafter (the call came from
the speculative drafter), source, record_sha256 (of the captured record) and
padded_from.

source "live": the whole call as it ran, with "out" = the output cuBLAS
(F.linear) gave in that call. source "padded": a row count the served engine
reaches only by padding a smaller call up to its CUDA-graph size (the
drafter's per-request rows in mixed steps): the real rows of a live call of
``padded_from`` rows plus real rows of other calls of the same shape and
phase; no stored output.
"""

import collections
import json
from pathlib import Path

import pytest
import torch

from tests.kernels.quantization.exl3_real_cases import directory_or_skip, file_sha256

CASES_ENV = "VLLM_THIN_GEMM_TEST_PORT_CASES"
SCHEMA = "thin_gemm_port_cases/1"
PHASES = ("decode", "prefill", "mixed")
SOURCES = ("live", "padded")
FIELDS = (
    "id", "file", "sha256", "weight_file", "weight_sha256", "shape", "role",
    "N", "K", "rows", "phase", "drafter", "source", "record_sha256", "padded_from",
)


def case_dir_or_skip() -> Path:
    path = directory_or_skip(CASES_ENV, "the thin-GEMM real-input case set")
    if not (path / "cases.json").is_file():
        pytest.skip(f"{CASES_ENV}={path} holds no cases.json")
    return path


def load_manifest(directory: Path) -> tuple[dict, list[dict]]:
    """(header, cases), each case checked for its fields."""
    manifest = json.loads((directory / "cases.json").read_text())
    assert manifest.get("schema") == SCHEMA, f"case set schema {manifest.get('schema')}"
    cases = manifest.get("cases")
    assert cases, "the case set has no cases"
    seen = set()
    for case in cases:
        missing = [k for k in FIELDS if k not in case]
        assert not missing, f"case {case.get('id')}: missing {missing}"
        assert case["id"] not in seen, f"duplicate case id {case['id']}"
        seen.add(case["id"])
        assert case["phase"] in PHASES, f"case {case['id']}: phase {case['phase']}"
        assert case["source"] in SOURCES, case["id"]
        assert case["rows"] > 0, case["id"]
        assert (case["source"] == "padded") == (case["padded_from"] is not None), case["id"]
    header = {k: v for k, v in manifest.items() if k != "cases"}
    return header, cases


def load_case(directory: Path, case: dict, weights: dict | None = None):
    """(x, w, out or None) on the CPU, after checking both files' sha256 and
    the shapes. ``weights`` caches loaded weights by weight_sha256."""
    path = directory / case["file"]
    assert file_sha256(path) == case["sha256"], f"case {case['id']}: file sha256"
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    x = tensors["x"]
    assert x.dtype == torch.bfloat16 and tuple(x.shape) == (case["rows"], case["K"]), case["id"]
    live = case["source"] == "live"
    assert ("out" in tensors) == live, f"case {case['id']}: out iff live"
    out = tensors.get("out")
    if live:
        assert out.dtype == torch.bfloat16 and tuple(out.shape) == (case["rows"], case["N"])
    cache = {} if weights is None else weights
    w = cache.get(case["weight_sha256"])
    if w is None:
        wpath = directory / case["weight_file"]
        assert file_sha256(wpath) == case["weight_sha256"], f"case {case['id']}: weight sha256"
        w = torch.load(wpath, map_location="cpu", weights_only=True)["w"]
        cache[case["weight_sha256"]] = w
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (case["N"], case["K"]), case["id"]
    return x, w, out


def pool_key(case: dict) -> tuple:
    return (case["shape"], case["rows"], case["phase"], case["drafter"], case["source"])


def pools(cases: list[dict]) -> dict[tuple, list[dict]]:
    grouped = collections.defaultdict(list)
    for case in cases:
        grouped[pool_key(case)].append(case)
    return dict(sorted(grouped.items()))


def coverage_report(
    cases: list[dict], required: set[tuple]
) -> tuple[list[str], list[tuple]]:
    """(one line per phase listing rows:shapes covered, the missing required
    cells). A cell (shape, rows, phase) is met by any case of that shape, row
    count and phase; a cell (shape, rows, phase, "target") only by one of the
    target model's own weights (not the drafter's), listed on one more line
    per phase that has such cells. "+" marks a row count beyond the
    required."""
    found = collections.defaultdict(collections.Counter)
    for case in cases:
        cell = (case["shape"], case["rows"], case["phase"])
        found[cell][case["source"]] += 1
        if not case["drafter"]:
            found[cell + ("target",)][case["source"]] += 1
    lines = []
    for phase in PHASES:
        for role in ((), ("target",)):
            want = {c for c in required if c[2] == phase and c[3:] == role}
            if role and not want:
                continue
            got = {c for c in found if c[2] == phase and c[3:] == role}
            rows = sorted({c[1] for c in want | (set() if role else got)})
            parts = []
            for r in rows:
                req = {c[0] for c in want if c[1] == r}
                have = {c[0] for c in got if c[1] == r}
                mark = "" if req else "+"
                miss = sorted(req - have)
                parts.append(f"{mark}{r}:{len(have)}"
                             + (f"(MISSING {','.join(miss)})" if miss else ""))
            name = " ".join((phase,) + role)
            lines.append(f"CELLS {name}: " + " ".join(parts))
    missing = sorted(c for c in required if c not in found)
    return lines, missing
