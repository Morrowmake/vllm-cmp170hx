# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE v2 (VLLM_EXL3_MOE_V2) on real captured inputs.

Inputs: VLLM_EXL3_TEST_PORT_CASES names the case set (layout in
exl3_real_cases.py: cases.json plus one torch file per case) and
VLLM_EXL3_TEST_CHECKPOINT the EXL3 checkpoint the calls ran on (the
Qwen3.8-Flash-Next 5.05 bpw checkpoint). With either unset or missing, the
tests that need it skip and say which. Every case these tests take is a
routed-expert call of 1..64 rows (the set may hold larger calls for other
kernels) taken from the served model at tensor parallel size 2, rank 0 (256
of 512 experts local, top-10): its input rows, top-k ids and weights, the
layer's expert map and packed metadata, and for whole ("live") calls the
output moe_out gave. The layer's expert weights are read from the checkpoint
and laid out as Exl3MoEMethod lays them out.

CPU: skip behaviour, case loading, pooling and coverage on a synthetic case
set, the pooled rule on synthetic arrays with a failing negative control,
the FP64 routed-expert reference against a direct sum, the required cells
against the dispatch rule, and (case set given) every file's sha256 and one
case per required (rows, phase) cell.

GPU (SM 8.0): per case the shipped operator (ops.moe_out), the port
(ops.moe_out_v2) and an FP64 reference from the checkpoint weights. moe_out
must give each live case's recorded output bit for bit (the inputs and
weights are the served ones); per pool (role, rows, phase, drafter) the port
holds moe_out's error against FP64 (test_exl3_moe_v2._Fp64Pool: pooled mean
<= 1.10x, max <= 1.25x with no element excused, exact where moe_out is
exact). One line per pool and per phase's cells is printed; a failing pool
fails.

Coverage (REQUIRED_CELLS: Exl3Experts.apply sends every call of 1..64 rows
to moe_out_v2 whatever the phase; served traffic of 1-8 concurrent requests
with 3-token speculative decoding and prompts of 17-40 tokens gives):
- decode 1-8 (drafter rows) and 4c for c = 1-8 (verify rows): live calls,
  layers 0, 1, 2, 11, 23, 35, 47 and the drafter's layer;
- prefill 1-3 and 5-8 (the drafter's rows after a prefill of 1-8 requests)
  and the prompts' 17-40 rows (target and drafter): live calls;
- prefill 4: no capture holds a 4-row prefill call, so its cases are the last
  4 rows of two 5-row prefill calls of the drafter's layer ("sliced", no
  recorded output; both operators run them as 4-row calls).
"""

import collections
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_kernels import effective_weight
from tests.kernels.quantization.test_exl3_moe_v2 import (
    GLOBAL,
    HIDDEN,
    INTERMEDIATE,
    LOCAL,
    TOPK,
    _bitwise,
    _extension_or_skip,
    _Fp64Pool,
    _v1,
    _v2,
    needs_sm80,
)
from vllm.sm80.kernels.exl3 import ops as exl3_ops

_CONCURRENCY = range(1, 9)
DECODE_ROWS = sorted({*_CONCURRENCY} | {4 * c for c in _CONCURRENCY})
PREFILL_ROWS = sorted({*_CONCURRENCY} | {17, 18, 20, 24, 25, 28, 30, 32, 33, 40})
REQUIRED_CELLS = {(r, "decode") for r in DECODE_ROWS} | {
    (r, "prefill") for r in PREFILL_ROWS
}
ROLES = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- CPU tests


def test_unset_or_missing_inputs_skip_with_the_reason(monkeypatch, tmp_path):
    monkeypatch.delenv(real.CASES_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=real.CASES_ENV):
        real.case_dir_or_skip()
    monkeypatch.setenv(real.CASES_ENV, str(tmp_path / "absent"))
    with pytest.raises(pytest.skip.Exception, match="not a directory"):
        real.case_dir_or_skip()
    monkeypatch.delenv(real.CHECKPOINT_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=real.CHECKPOINT_ENV):
        real.checkpoint_or_skip()


def _synthetic_case(directory, index, rows, phase, source, drafter=False):
    g = torch.Generator().manual_seed(index)
    tensors = dict(
        x=torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16),
        topk_ids=torch.randint(0, GLOBAL, (rows, TOPK), generator=g).to(torch.int32),
        topk_weights=torch.rand(rows, TOPK, generator=g),
        expert_map=torch.arange(GLOBAL, dtype=torch.int32),
        packed_meta=torch.zeros(LOCAL, 3, 4, dtype=torch.int64),
    )
    if source == "live":
        tensors["out"] = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
    (directory / "moe").mkdir(exist_ok=True)
    path = directory / "moe" / f"{index}.pt"
    torch.save(tensors, path)
    return dict(
        id=str(index),
        file=f"moe/{index}.pt",
        sha256=real.file_sha256(path),
        prefix="layers.0.mlp.experts",
        role="layers.*.mlp.experts",
        layer=0,
        rows=rows,
        phase=phase,
        drafter=drafter,
        source=source,
        record_sha256="0" * 64,
    )


def test_case_loading_pooling_and_coverage_on_a_synthetic_set(tmp_path, capsys):
    import json

    cases = [
        _synthetic_case(tmp_path, 0, 4, "decode", "live"),
        _synthetic_case(tmp_path, 1, 4, "prefill", "sliced", drafter=True),
        _synthetic_case(tmp_path, 2, 17, "prefill", "live"),
    ]
    (tmp_path / "cases.json").write_text(
        json.dumps(dict(schema=real.SCHEMA, sm_count=74, moe=cases))
    )
    _, loaded = real.load_manifest(tmp_path, "moe")
    for case in loaded:
        tensors = real.load_case(tmp_path, "moe", case)
        assert tensors["topk_ids"].shape == (case["rows"], TOPK)
    assert len(real.pools(loaded)) == 3
    required = {(4, "decode"), (4, "prefill"), (17, "prefill")}
    real.require_cells(loaded, required, "synthetic")
    assert "CELLS prefill: 4:1s 17:1l" in capsys.readouterr().out
    with pytest.raises(AssertionError, match=r"\(64, 'decode'\)"):
        real.require_cells(loaded, required | {(64, "decode")}, "synthetic")
    # The case set must say which phase made the call.
    loaded[0]["phase"] = "mixed"
    (tmp_path / "cases.json").write_text(
        json.dumps(dict(schema=real.SCHEMA, sm_count=74, moe=loaded))
    )
    with pytest.raises(AssertionError, match="phase"):
        real.load_manifest(tmp_path, "moe")


def test_pool_rule_passes_equal_error_and_fails_a_larger_one():
    g = torch.Generator().manual_seed(6)
    passing, failing = _Fp64Pool(), _Fp64Pool()
    for index in range(3):
        ref = torch.randn(4, HIDDEN, generator=g, dtype=torch.float64)
        inc = ref.to(torch.bfloat16)
        passing.add(inc.clone(), inc, ref, f"call {index}")
        worse = (ref + 3.0 * (inc.double() - ref)).to(torch.bfloat16)
        failing.add(worse, inc, ref, f"call {index}")
    assert passing.check("equal error") == (1.0, 1.0)
    with pytest.raises(AssertionError, match="mean error"):
        failing.check("negative control")


def test_fp64_routed_reference_matches_a_direct_sum():
    rng = np.random.default_rng(4)
    hidden, intermediate, local = 256, 128, 3
    experts, dense = [], []
    for _ in range(local):
        parts, mats = [], []
        for k, n in (
            (hidden, intermediate),
            (hidden, intermediate),
            (intermediate, hidden),
        ):
            trellis = rng.integers(-(2**15), 2**15, (k // 16, n // 16, 80)).astype(
                np.int16
            )
            suh = rng.uniform(0.5, 1.5, k).astype(np.float16)
            svh = (rng.uniform(-1, 1, n) / 64).astype(np.float16)
            parts.append(
                (
                    real.core_weight(torch.from_numpy(trellis), 10, 2),
                    torch.from_numpy(suh),
                    torch.from_numpy(svh),
                )
            )
            mats.append(effective_weight(trellis, suh, svh, 10, 2))
        experts.append(parts)
        dense.append(mats)
    # Global experts 1, 3, 5 are local 0, 1, 2; -1 and 6 are padding / out of range.
    expert_map = torch.tensor([-1, 0, -1, 1, -1, 2], dtype=torch.int32)
    calls = []
    for rows in (1, 3):
        x = torch.from_numpy(rng.standard_normal((rows, hidden))).to(torch.bfloat16)
        ids = torch.from_numpy(rng.choice([-1, 0, 1, 3, 5, 6], (rows, 4))).to(
            torch.int32
        )
        weights = torch.from_numpy(rng.uniform(0.1, 0.4, (rows, 4)).astype(np.float32))
        calls.append((x, ids, weights))
    outs = real.moe_fp64(calls, expert_map, experts.__getitem__, intermediate)
    for (x, ids, weights), out in zip(calls, outs):
        x64 = x.double().numpy()
        for t in range(x.shape[0]):
            expected = np.zeros(hidden)
            for g, w in zip(ids[t].tolist(), weights[t].double().tolist()):
                if not 0 <= g < len(expert_map) or expert_map[g] < 0:
                    continue
                gate, up, down = dense[int(expert_map[g])]
                a = x64[t] @ gate
                expected += w * ((a / (1.0 + np.exp(-a)) * (x64[t] @ up)) @ down)
            np.testing.assert_allclose(out[t].numpy(), expected, rtol=1e-11, atol=1e-12)


def _v2_cases(directory):
    """The case set's MoE calls of 1..64 rows (it may also hold the larger
    calls other kernels serve)."""
    _, cases = real.load_manifest(directory, "moe")
    return [case for case in cases if case["rows"] <= exl3_ops.MOE_V2_MAX_ROWS]


def test_required_cells_are_moe_v2_calls():
    for rows, phase in REQUIRED_CELLS:
        assert 0 < rows <= exl3_ops.MOE_V2_MAX_ROWS, (rows, phase)


def test_case_set_files_and_coverage():
    directory = real.case_dir_or_skip()
    cases = _v2_cases(directory)
    for case in cases:
        tensors = real.load_case(directory, "moe", case)
        assert 0 < case["rows"] <= exl3_ops.MOE_V2_MAX_ROWS, case["id"]
        assert tensors["topk_ids"].shape[1] == TOPK, case["id"]
        assert tensors["expert_map"].shape == (GLOBAL,), case["id"]
        assert int((tensors["expert_map"] >= 0).sum()) == LOCAL, case["id"]
    real.require_cells(cases, REQUIRED_CELLS, "MoE v2")


# ---------------------------------------------------------------- GPU tests


def _expert_set(checkpoint, prefix, expert_map, device):
    """The layer's local experts as Exl3MoEMethod lays them out: local l holds
    global g where expert_map[g] == l; per projection trellis, suh, svh."""
    owner = {}
    for g, local in enumerate(expert_map.tolist()):
        if local >= 0:
            assert local not in owner, "expert map assigns a local expert twice"
            owner[local] = g
    # The partition follows the expert map: 256 of the 512 experts at tensor
    # parallel size 2, and all 512 where the pipeline splits the layers at
    # tensor parallel size 1.
    local_experts = len(owner)
    assert sorted(owner) == list(range(local_experts)), "expert map leaves a gap"
    weights, pointers, meta = [], [], []
    for local in range(local_experts):
        ptrs, rows = [], []
        for role in ROLES:
            name = f"{prefix}.{owner[local]}.{role}"
            matrix = checkpoint.matrix(name)
            tensors = [t.to(device) for t in checkpoint.whole(name)]
            weights.append(tensors)
            ptrs.append([t.data_ptr() for t in tensors])
            rows.append(
                [matrix.padded_in, matrix.padded_out, matrix.bits_x2, matrix.codebook]
            )
        pointers.append(ptrs)
        meta.append(rows)
    return SimpleNamespace(
        weights=weights,
        pointers=torch.tensor(pointers, dtype=torch.int64, device=device),
        meta=torch.tensor(meta, dtype=torch.int64, device=device),
        expert_map=expert_map.to(device),
    )


@needs_sm80
def test_v2_holds_moe_outs_error_against_fp64_on_real_inputs():
    directory = real.case_dir_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    _extension_or_skip()
    cases = _v2_cases(directory)
    real.require_cells(cases, REQUIRED_CELLS, "MoE v2")
    device = torch.device("cuda")
    loaded = {case["id"]: real.load_case(directory, "moe", case) for case in cases}
    layers = collections.defaultdict(list)
    for case in cases:
        expert_map = loaded[case["id"]]["expert_map"]
        layers[(case["prefix"], expert_map.numpy().tobytes())].append(case)
    grouped = real.pools(cases)
    pool_of = {key: _Fp64Pool() for key in grouped}
    live = 0
    for (prefix, _), group in sorted(layers.items()):
        expert_map = loaded[group[0]["id"]]["expert_map"]
        served = _expert_set(checkpoint, prefix, expert_map, device)
        calls, results = [], []
        for case in group:
            tensors = loaded[case["id"]]
            name = f"case {case['id']} ({prefix}, {case['rows']} rows, {case['phase']})"
            assert torch.equal(tensors["packed_meta"], served.meta.cpu()), (
                f"{name}: the checkpoint's expert layout differs from the recorded one"
            )
            x = tensors["x"].to(device)
            ids = tensors["topk_ids"].to(device)
            weights = tensors["topk_weights"].to(device)
            shipped = _v1(served, x, ids, weights)
            port = _v2(served, x, ids, weights)
            if case["source"] == "live":
                _bitwise(shipped.cpu(), tensors["out"], f"{name}: recorded output")
                live += 1
            calls.append((x, ids, weights))
            results.append((case, name, port.cpu(), shipped.cpu()))
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

        refs = real.moe_fp64(calls, served.expert_map, decode, INTERMEDIATE)
        for (case, name, port, shipped), ref in zip(results, refs):
            pool_of[real.pool_key(case)].add(port, shipped, ref.cpu(), name)
        del served, calls, refs, decoded
        torch.accelerator.empty_cache()
    failures = []
    for key, pool in pool_of.items():
        role, rows, phase, drafter = key
        name = f"POOL moe {role} rows={rows} {phase} drafter={int(drafter)}"
        sources = collections.Counter(case["source"] for case in grouped[key])
        try:
            mean_ratio, max_ratio = pool.check(name)
            print(
                f"{name} cases={len(grouped[key])} live={sources['live']} "
                f"sliced={sources['sliced']} mean x{mean_ratio:.4f} "
                f"max x{max_ratio:.4f} PASS"
            )
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"{name} cases={len(grouped[key])} FAIL {exc}")
    print(
        f"MoE v2 real inputs: {len(cases)} cases ({live} live replayed bitwise), "
        f"{len(pool_of)} pools, {len(failures)} failing"
    )
    assert not failures, failures
