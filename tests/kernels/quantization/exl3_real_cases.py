# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real captured inputs for the EXL3 kernel tests.

Case loading, pooling, coverage of the (rows, phase) cells a dispatch rule
sends to a kernel, checkpoint weights as tensor-parallel rank 0 holds them,
and FP64 references. Used by test_exl3_dense_splitk_real_inputs.py and
test_exl3_moe_v2_real_inputs.py.

Case set: the directory named by VLLM_EXL3_TEST_PORT_CASES.

    cases.json        {"schema": "exl3_port_cases/1", "sm_count": 74,
                       "tp_size": 2, "tp_rank": 0,
                       "dense": [case, ...], "moe": [case, ...]}
    dense/<id>.pt     torch.save dicts, read with weights_only=True
    moe/<id>.pt

Every case: id, file (relative to the directory), sha256 (of the file),
prefix (checkpoint key stem of the matrix or the expert set), role (prefix
with the layer number replaced by "*"), layer, rows, phase ("decode" or
"prefill", from the scheduler step that made the call, never from the row
count), drafter (the call came from the speculative drafter), source and
record_sha256 (sha256 of the captured record the case came from).

source "live": the whole call as it ran, with "out" = the output the shipped
kernel produced in that call. source "sliced": the last rows of a live call
of the same matrix, phase and drafter, standing for a launch of that many
rows that no capture holds whole; it has no stored output. source "joined"
(MoE only): the rows of consecutive live calls of the same layer, phase and
drafter, in order, standing for a call of more rows than any capture holds
(the routed experts compute every row from its own input and routes alone);
joined_from lists [record_sha256, first row, end row] per part, and it has no
stored output.

Dense case fields: bits_x2, codebook, input_slice [start, stop] and
output_slices [[start, stop], ...] (the rank's H128 blocks of the checkpoint
matrix, in order), weight_sha256 (sha256 of the rank's trellis, suh and svh
bytes, or null), logical_out (N). File tensors: x [rows, K] bf16, out
[rows, N] bf16 (live).

MoE case file tensors: x [rows, 2560] bf16, topk_ids [rows, 10] int32,
topk_weights [rows, 10] fp32, expert_map [512] int32, packed_meta [256, 3, 4]
int64 (the layer's padded_in, padded_out, bits_x2, codebook per local expert
and projection), out [rows, 2560] bf16 (live).
"""

import collections
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.kernels.quantization.test_exl3_kernels import (
    _COLUMNS,
    _H128,
    _ROWS,
    _codebook,
)

CASES_ENV = "VLLM_EXL3_TEST_PORT_CASES"
CHECKPOINT_ENV = "VLLM_EXL3_TEST_CHECKPOINT"
SCHEMA = "exl3_port_cases/1"
PHASES = ("decode", "prefill")
_COMMON = (
    "id",
    "file",
    "sha256",
    "prefix",
    "role",
    "layer",
    "rows",
    "phase",
    "drafter",
    "source",
    "record_sha256",
)
_FIELDS = {
    "dense": _COMMON
    + (
        "bits_x2",
        "codebook",
        "input_slice",
        "output_slices",
        "weight_sha256",
        "logical_out",
    ),
    "moe": _COMMON,
}
_TENSORS = {
    "dense": ("x",),
    "moe": ("x", "topk_ids", "topk_weights", "expert_map", "packed_meta"),
}


# ---------------------------------------------------------------- the case set


def directory_or_skip(env: str, what: str) -> Path:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"set {env} to {what}")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{env}={value} is not a directory ({what})")
    return path


def case_dir_or_skip() -> Path:
    path = directory_or_skip(CASES_ENV, "the EXL3 real-input case set")
    if not (path / "cases.json").is_file():
        pytest.skip(f"{CASES_ENV}={path} holds no cases.json")
    return path


def checkpoint_or_skip() -> Path:
    return directory_or_skip(CHECKPOINT_ENV, "the EXL3 checkpoint the cases ran on")


def load_manifest(directory: Path, family: str) -> tuple[dict, list[dict]]:
    """(header, cases of the family), each case checked for its fields."""
    manifest = json.loads((directory / "cases.json").read_text())
    assert manifest.get("schema") == SCHEMA, f"case set schema {manifest.get('schema')}"
    cases = manifest.get(family)
    assert cases, f"the case set has no {family} cases"
    seen = set()
    for case in cases:
        missing = [k for k in _FIELDS[family] if k not in case]
        assert not missing, f"case {case.get('id')}: missing {missing}"
        assert case["id"] not in seen, f"duplicate case id {case['id']}"
        seen.add(case["id"])
        assert case["phase"] in PHASES, f"case {case['id']}: phase {case['phase']}"
        assert case["source"] in ("live", "sliced", "joined"), case["id"]
        assert case["source"] != "joined" or family == "moe", case["id"]
        assert case["rows"] > 0, case["id"]
    header = {k: v for k, v in manifest.items() if k not in _FIELDS}
    return header, cases


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(directory: Path, family: str, case: dict) -> dict:
    """The case's tensors (CPU), after checking the file's sha256 and shapes."""
    path = directory / case["file"]
    assert file_sha256(path) == case["sha256"], f"case {case['id']}: file sha256"
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    for name in _TENSORS[family]:
        assert name in tensors, f"case {case['id']}: no {name}"
    rows = case["rows"]
    assert tensors["x"].dtype == torch.bfloat16 and tensors["x"].shape[0] == rows
    live = case["source"] == "live"
    assert ("out" in tensors) == live, f"case {case['id']}: out iff live"
    if live:
        assert tensors["out"].dtype == torch.bfloat16
        assert tensors["out"].shape[0] == rows
    if family == "moe":
        assert tensors["topk_ids"].shape[0] == rows
        assert tensors["topk_weights"].shape == tensors["topk_ids"].shape
    return tensors


def pool_key(case: dict) -> tuple:
    return (case["role"], case["rows"], case["phase"], case["drafter"])


def pools(cases: list[dict]) -> dict[tuple, list[dict]]:
    grouped = collections.defaultdict(list)
    for case in cases:
        grouped[pool_key(case)].append(case)
    return dict(sorted(grouped.items()))


def cells(cases: list[dict]) -> dict[tuple[int, str], collections.Counter]:
    """(rows, phase) -> Counter of case sources."""
    found: dict = collections.defaultdict(collections.Counter)
    for case in cases:
        found[(case["rows"], case["phase"])][case["source"]] += 1
    return found


def coverage_report(
    cases: list[dict], required: set[tuple[int, str]]
) -> tuple[list[str], list[tuple[int, str]]]:
    """(one line per phase listing the covered cells, the missing required cells)."""
    found = cells(cases)
    lines = []
    for phase in PHASES:
        parts = []
        for rows in sorted(r for r, p in set(found) | required if p == phase):
            sources = found.get((rows, phase), collections.Counter())
            mark = "" if (rows, phase) in required else "+"
            if not sources:
                parts.append(f"{rows}:MISSING")
            else:
                parts.append(
                    f"{mark}{rows}:"
                    + "/".join(f"{n}{s[0]}" for s, n in sorted(sources.items()))
                )
        lines.append(f"CELLS {phase}: " + " ".join(parts))
    sliced = sorted(
        (cell for cell, sources in found.items() if sources.get("sliced")),
        key=lambda c: (c[1], c[0]),
    )
    lines.append(
        "CELLS sources: l = a live captured call, replayed bitwise; s = sliced, the"
        " last rows of a longer live call (real activations, no recorded output)"
    )
    lines.append("CELLS sliced: " + (" ".join(f"{r}/{p}" for r, p in sliced) or "none"))
    joined = sorted(
        (cell for cell, sources in found.items() if sources.get("joined")),
        key=lambda c: (c[1], c[0]),
    )
    if joined:
        lines.append(
            "CELLS joined (j, the rows of consecutive live calls of one layer, no "
            "recorded output): " + " ".join(f"{r}/{p}" for r, p in joined)
        )
    missing = sorted(required - set(found), key=lambda c: (c[1], c[0]))
    return lines, missing


def require_cells(cases: list[dict], required: set[tuple[int, str]], name: str):
    lines, missing = coverage_report(cases, required)
    for line in lines:
        print(f"{name} {line}")
    assert not missing, f"{name}: no case for required (rows, phase) cells {missing}"


# ---------------------------------------------------------------- checkpoint

_NUMPY = {"I16": np.int16, "F16": np.float16}


class Checkpoint:
    """Packed matrices of an EXL3 checkpoint, read through its headers."""

    def __init__(self, directory: Path):
        from vllm_sm80.kernels.exl3.hybrid import HYBRID_KEYS
        from vllm_sm80.kernels.exl3.utils import Exl3Manifest

        # A hybrid checkpoint (EXL3 for the modules its quantization_config
        # selects, another method for the rest) is read as the engine reads
        # it: the index names each tensor's file, FP8 dtypes included.
        config = Path(directory) / "config.json"
        quant = (
            json.loads(config.read_text()).get("quantization_config") or {}
            if config.exists()
            else {}
        )
        self.manifest = Exl3Manifest(
            directory, select_by_index=any(key in quant for key in HYBRID_KEYS)
        )

    def _array(self, name: str) -> np.ndarray:
        header = self.manifest.tensors[name]
        return np.memmap(
            header.path,
            dtype=_NUMPY[header.dtype],
            mode="r",
            offset=header.offset,
            shape=header.shape,
        )

    def matrix(self, prefix: str):
        return self.manifest.matrices[prefix]

    def shard(self, prefix: str, input_slice, output_slices):
        """(trellis, suh, svh) CPU tensors: the rank's blocks, as the engine
        loads them (Exl3PackedWeight.load)."""
        ks = slice(input_slice[0] // 16, input_slice[1] // 16)
        trellis = self._array(f"{prefix}.trellis")
        suh = self._array(f"{prefix}.suh")
        svh = self._array(f"{prefix}.svh")
        parts = [trellis[ks, a // 16 : b // 16] for a, b in output_slices]
        return (
            torch.from_numpy(np.ascontiguousarray(np.concatenate(parts, axis=1))),
            torch.from_numpy(np.array(suh[input_slice[0] : input_slice[1]])),
            torch.from_numpy(np.concatenate([svh[a:b] for a, b in output_slices])),
        )

    def whole(self, prefix: str):
        matrix = self.matrix(prefix)
        return self.shard(prefix, (0, matrix.padded_in), [(0, matrix.padded_out)])


def weight_sha256(trellis, suh, svh) -> str:
    digest = hashlib.sha256()
    for tensor in (trellis, suh, svh):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- FP64

_LUT: dict = {}


def _codebook_lut(codebook: int, device) -> torch.Tensor:
    key = (codebook, str(device))
    if key not in _LUT:
        values = _codebook(np.arange(65536, dtype=np.uint64), codebook)
        _LUT[key] = torch.from_numpy(values).to(device)
    return _LUT[key]


def core_weight(trellis: torch.Tensor, bits_x2: int, codebook: int, chunk=4096):
    """test_exl3_kernels.core_weight in torch on the trellis' device, in tile
    chunks: decoded codebook values, float64 [padded_in, padded_out]."""
    device = trellis.device
    k_tiles, n_tiles, halves = trellis.shape
    lut = _codebook_lut(codebook, device)
    ends = torch.arange(1, 257, device=device) * bits_x2 // 2
    window = (ends[:, None] - 16 + torch.arange(16, device=device)) % (16 * halves)
    shifts = torch.arange(31, -1, -1, device=device)
    position = torch.from_numpy(_ROWS * 16 + _COLUMNS).to(device)
    flat = trellis.reshape(-1, halves)
    tiles = torch.empty(flat.shape[0], 256, dtype=torch.float64, device=device)
    for start in range(0, flat.shape[0], chunk):
        h = flat[start : start + chunk].to(torch.int64) & 0xFFFF
        words = h[:, 0::2] | (h[:, 1::2] << 16)
        ring = ((words[..., None] >> shifts) & 1).reshape(words.shape[0], -1)
        states = torch.zeros(ring.shape[0], 256, dtype=torch.int64, device=device)
        for bit in range(16):
            states |= ring[:, window[:, bit]] << (15 - bit)
        block = torch.empty(states.shape, dtype=torch.float64, device=device)
        block[:, position] = lut[states]
        tiles[start : start + chunk] = block
    return (
        tiles.view(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
    )


def _blockwise_hadamard(v: torch.Tensor) -> torch.Tensor:
    h = torch.from_numpy(_H128).to(v.device)
    return (v.reshape(v.shape[0], -1, 128) @ h).reshape(v.shape)


def apply_fp64(x: torch.Tensor, core, suh, svh) -> torch.Tensor:
    """X [rows, logical_in] -> float64 [rows, padded_out] = x @ W, with
    W = diag(suh) H core H diag(svh) (test_exl3_kernels.effective_weight)."""
    padded_in = core.shape[0]
    t = torch.zeros(x.shape[0], padded_in, dtype=torch.float64, device=x.device)
    t[:, : x.shape[1]] = x.double() * suh[: x.shape[1]].double()
    return _blockwise_hadamard(_blockwise_hadamard(t) @ core) * svh.double()


def moe_fp64(calls, expert_map, decode_expert, intermediate: int, swiglu_limit=0.0):
    """FP64 routed-expert outputs of several calls of one layer.

    calls: [(x, topk_ids, topk_weights)], expert_map: global -> local id;
    decode_expert(local) -> three (core, suh, svh) for gate, up, down.
    out[t] = sum_j w[t, j] down(SiLU(gate x_t) * up x_t) over the local routes
    (ids outside the map or mapped to -1 are other ranks' or padding).
    swiglu_limit > 0 clamps gate x_t from above and up x_t on both sides
    before SiLU(gate) * up, as the operators' swiglu_limit does."""
    device = calls[0][0].device
    emap = expert_map.cpu().tolist()
    outs = [
        torch.zeros(x.shape, dtype=torch.float64, device=device) for x, _, _ in calls
    ]
    routes = collections.defaultdict(list)
    for index, (_, ids, weights) in enumerate(calls):
        ids, weights = ids.cpu().tolist(), weights.double().cpu().tolist()
        for row, (row_ids, row_weights) in enumerate(zip(ids, weights)):
            for g, w in zip(row_ids, row_weights):
                if 0 <= g < len(emap) and emap[g] >= 0:
                    routes[emap[g]].append((index, row, w))
    for local, items in sorted(routes.items()):
        gate, up, down = decode_expert(local)
        x = torch.stack([calls[i][0][r] for i, r, _ in items])
        hidden = x.shape[1]
        a = apply_fp64(x, *gate)[:, :intermediate]
        b = apply_fp64(x, *up)[:, :intermediate]
        if swiglu_limit > 0:
            a = a.clamp(max=swiglu_limit)
            b = b.clamp(-swiglu_limit, swiglu_limit)
        act = a / (1.0 + torch.exp(-a)) * b
        y = apply_fp64(act, *down)[:, :hidden]
        for (index, row, w), value in zip(items, y):
            outs[index][row] += w * value
    return outs
