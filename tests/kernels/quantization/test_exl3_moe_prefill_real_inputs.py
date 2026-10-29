# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE prefill kernel (VLLM_EXL3_MOE_PREFILL_TC) on real captured inputs.

Inputs as in test_exl3_moe_v2_real_inputs.py: VLLM_EXL3_TEST_PORT_CASES
names the case set (layout in exl3_real_cases.py) and
VLLM_EXL3_TEST_CHECKPOINT the Qwen3.8-Flash-Next 5.05 bpw checkpoint; with
either unset or missing, the tests that need it skip and say which. These
tests take the set's prefill calls of more than 64 rows (the calls
Exl3Experts.apply sends to moe_out_prefill); the dispatch check takes every
case.

The case set's layout (the manifest's tp_size and pp_size, the convention the
dense real-input tests use) picks the expert partition the set must hold: 256
of the 512 experts at tensor parallel size 2, all 512 at tensor parallel size 1
with pipeline parallel size 2; a set captured on another layout skips and says
so.

CPU: the required cells against the dispatch rule; the dispatch rule over
every case of the set, with the step phase the case was captured in (no
decode call, whatever its rows, and no call of at most 64 rows reaches
moe_out_prefill); a joined case on a synthetic set; and (case set given)
every file's sha256 and one case per required cell.

GPU (SM 8.0): per case the shipped operator (ops.moe_out), the port
(ops.moe_out_prefill) and an FP64 reference from the checkpoint weights.
moe_out must give each live case's recorded output bit for bit; per pool
(role, rows, phase, drafter) the port holds moe_out's error against FP64
(test_exl3_moe_v2._Fp64Pool: pooled mean <= 1.10x, max <= 1.25x with no
element excused, exact where moe_out is exact), and a second call is
bitwise the first. One line per pool is printed; a failing pool fails.

Coverage (REQUIRED_CELLS, prefill only): with 3460 tokens per step the served
prefill calls of more than 64 rows are the 3460-row chunks and each prompt's
tail: 1906 (the 8826-token reference prompt: 3460 + 3460 + 1906), and the
chunk sizes of the KL and drift prompts at that budget (2048; 2544 of a
16384-token prompt; 139..3412). 65 is the smallest call the dispatch sends.
Cells no capture holds whole are sliced from a longer live call of the same
layer, or joined from consecutive live calls of one layer above the longest
capture (3200 rows).
"""

import collections
import json

import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_moe_prefill import (
    DECODE_LEN,
    _experts,
    _phase,
    expected_operator,
)
from tests.kernels.quantization.test_exl3_moe_v2 import (
    GLOBAL,
    HIDDEN,
    INTERMEDIATE,
    TOPK,
    _apply,
    _bitwise,
    _extension_or_skip,
    _Fp64Pool,
    _v1,
    needs_sm80,
)
from tests.kernels.quantization.test_exl3_moe_v2_real_inputs import (
    _expert_set,
    _synthetic_case,
)
from vllm.sm80.kernels.exl3 import ops as exl3_ops

PREFILL_ROWS = sorted(
    {65, 1906, 2048, 2544, 3460}
    | {139, 309, 312, 316, 318, 326, 384, 632, 689, 756, 768, 827, 1237, 1452}
    | {2073, 2174, 2506, 2769, 3036, 3342, 3412}
)
REQUIRED_CELLS = {(rows, "prefill") for rows in PREFILL_ROWS}


def _prefill_cases(cases):
    return [
        case
        for case in cases
        if case["phase"] == "prefill" and case["rows"] >= exl3_ops.MOE_PREFILL_MIN_ROWS
    ]


# Expert partitions recorded by the Qwen checkpoint capture protocol.
PREFILL_PARTITION = {(2, 1): 256, (1, 2): 512}


def _local_experts_or_skip(header) -> int:
    """How many experts are local on the layout the cases were captured on
    (the manifest's tp_size and pp_size), skipping a set captured on a layout
    the prefill operator is not validated on: its expert map and pointer table
    would not be the served ones."""
    layout = (header["tp_size"], header.get("pp_size", 1))
    if layout not in PREFILL_PARTITION:
        pytest.skip(
            f"case set layout (tensor, pipeline) parallel size {layout} is not "
            f"a validated prefill partition, one of {list(PREFILL_PARTITION)}"
        )
    return PREFILL_PARTITION[layout]


# ---------------------------------------------------------------- CPU tests


def test_required_cells_are_moe_prefill_calls():
    for rows, phase in REQUIRED_CELLS:
        assert expected_operator(rows, phase, 64, 4) == "moe_out_prefill"
        assert expected_operator(rows, "decode", 64, 4) != "moe_out_prefill"


def test_joined_cases_load_and_show_in_the_coverage(tmp_path, capsys):
    cases = [
        _synthetic_case(tmp_path, 0, 70, "prefill", "live"),
        dict(_synthetic_case(tmp_path, 1, 90, "prefill", "sliced"), source="joined"),
    ]
    (tmp_path / "cases.json").write_text(
        json.dumps(dict(schema=real.SCHEMA, sm_count=74, moe=cases))
    )
    _, loaded = real.load_manifest(tmp_path, "moe")
    for case in loaded:
        real.load_case(tmp_path, "moe", case)
    real.require_cells(loaded, {(70, "prefill"), (90, "prefill")}, "synthetic")
    printed = capsys.readouterr().out
    assert "CELLS prefill: 70:1l 90:1j" in printed
    assert "CELLS joined" in printed and "90/prefill" in printed
    # A dense case may not be joined.
    dense = dict(loaded[1], bits_x2=10, codebook=2, input_slice=[0, 1])
    dense.update(output_slices=[[0, 1]], weight_sha256=None, logical_out=1)
    (tmp_path / "cases.json").write_text(
        json.dumps(dict(schema=real.SCHEMA, sm_count=74, dense=[dense]))
    )
    with pytest.raises(AssertionError):
        real.load_manifest(tmp_path, "dense")


@pytest.mark.parametrize("moe_v2_rows", [0, 64])
def test_case_set_dispatch_and_coverage(monkeypatch, moe_v2_rows):
    """Exl3Experts.apply with the switch on, per case in a step of the phase
    the case was captured in: only prefill calls of more than 64 rows reach
    moe_out_prefill."""
    directory = real.case_dir_or_skip()
    header, cases = real.load_manifest(directory, "moe")
    local = _local_experts_or_skip(header)
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_prefill"):
        monkeypatch.setattr(
            exl3_ops, name, lambda *args, _name=name, **kwargs: calls.append(_name)
        )
    experts = _experts(moe_v2_rows, DECODE_LEN)
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    reached: collections.Counter[tuple[str, str]] = collections.Counter()
    for case in cases:
        _phase(monkeypatch, case["phase"])
        x = torch.zeros(case["rows"], HIDDEN, dtype=torch.bfloat16)
        _apply(experts, x, "ids", "weights", "ptrs", "meta", "out", workspace2)
        operator = calls.pop()
        expected = expected_operator(case["rows"], case["phase"], moe_v2_rows, 4)
        assert operator == expected, case["id"]
        if case["phase"] == "decode" or case["rows"] <= 64:
            assert operator != "moe_out_prefill", case["id"]
        reached[(operator, case["phase"])] += 1
    print(
        f"MoE dispatch over the case set (moe_v2_rows={moe_v2_rows}): {dict(reached)}"
    )
    prefill = _prefill_cases(cases)
    for case in prefill:
        tensors = real.load_case(directory, "moe", case)
        assert tensors["topk_ids"].shape[1] == TOPK, case["id"]
        assert tensors["expert_map"].shape == (GLOBAL,), case["id"]
        assert int((tensors["expert_map"] >= 0).sum()) == local, case["id"]
    real.require_cells(prefill, REQUIRED_CELLS, "MoE prefill")


# ---------------------------------------------------------------- GPU tests


def _prefill(served, x, ids, weights):
    out = torch.full_like(x, float("nan"))
    workspace = torch.full(
        (
            exl3_ops.moe_workspace_size_prefill(
                x.shape[0], x.shape[1], INTERMEDIATE, TOPK, served.pointers.shape[0]
            ),
        ),
        float("nan"),
        device=x.device,
    )
    exl3_ops.moe_out_prefill(
        x,
        ids,
        weights,
        served.pointers,
        served.meta,
        served.expert_map,
        INTERMEDIATE,
        out,
        workspace,
    )
    return out


@needs_sm80
def test_prefill_holds_moe_outs_error_against_fp64_on_real_inputs():
    directory = real.case_dir_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "moe")
    local = _local_experts_or_skip(header)
    cases = _prefill_cases(cases)
    real.require_cells(cases, REQUIRED_CELLS, "MoE prefill")
    device = torch.device("cuda")
    layers = collections.defaultdict(list)
    for case in cases:
        tensors = real.load_case(directory, "moe", case)
        layers[(case["prefix"], tensors["expert_map"].numpy().tobytes())].append(case)
    grouped = real.pools(cases)
    pool_of = {key: _Fp64Pool() for key in grouped}
    live = 0
    for (prefix, _), group in sorted(layers.items()):
        first = real.load_case(directory, "moe", group[0])
        assert int((first["expert_map"] >= 0).sum()) == local, prefix
        served = _expert_set(checkpoint, prefix, first["expert_map"], device)
        calls, results = [], []
        for case in group:
            tensors = real.load_case(directory, "moe", case)
            name = (
                f"case {case['id']} ({prefix}, {case['rows']} rows, {case['source']})"
            )
            assert torch.equal(tensors["packed_meta"], served.meta.cpu()), (
                f"{name}: the checkpoint's expert layout differs from the recorded one"
            )
            x = tensors["x"].to(device)
            ids = tensors["topk_ids"].to(device)
            weights = tensors["topk_weights"].to(device)
            shipped = _v1(served, x, ids, weights)
            port = _prefill(served, x, ids, weights)
            _bitwise(_prefill(served, x, ids, weights), port, f"{name}: repeat")
            if case["source"] == "live":
                _bitwise(shipped.cpu(), tensors["out"], f"{name}: recorded output")
                live += 1
            calls.append((x.cpu(), ids.cpu(), weights.cpu()))
            results.append((case, name, port.cpu(), shipped.cpu()))
            del x, ids, weights, shipped, port
        decoded: dict[int, list[tuple]] = {}

        def decode(local, served=served, decoded=decoded):
            if local not in decoded:
                decoded.clear()  # one expert at a time bounds the FP64 memory
                parts = served.weights[3 * local : 3 * local + 3]
                decoded[local] = [
                    (real.core_weight(t, int(m[2]), int(m[3])), s, v)
                    for (t, s, v), m in zip(parts, served.meta[local].tolist())
                ]
            return decoded[local]

        gpu_calls = [tuple(t.to(device) for t in call) for call in calls]
        refs = real.moe_fp64(gpu_calls, served.expert_map, decode, INTERMEDIATE)
        for (case, name, port, shipped), ref in zip(results, refs):
            pool_of[real.pool_key(case)].add(port, shipped, ref.cpu(), name)
        del served, calls, gpu_calls, refs, decoded
        torch.accelerator.empty_cache()
    failures = []
    for key, pool in pool_of.items():
        role, rows, phase, drafter = key
        name = f"POOL moe_prefill {role} rows={rows} {phase} drafter={int(drafter)}"
        sources = collections.Counter(case["source"] for case in grouped[key])
        try:
            mean_ratio, max_ratio = pool.ratios()
            pool.check(name)
            print(
                f"{name} cases={len(grouped[key])} live={sources['live']} "
                f"sliced={sources['sliced']} joined={sources['joined']} "
                f"mean x{mean_ratio:.4f} max x{max_ratio:.4f} PASS"
            )
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"{name} cases={len(grouped[key])} FAIL {exc}")
    print(
        f"MoE prefill real inputs: {len(cases)} cases ({live} live replayed "
        f"bitwise), {len(pool_of)} pools, {len(failures)} failing"
    )
    assert not failures, failures
