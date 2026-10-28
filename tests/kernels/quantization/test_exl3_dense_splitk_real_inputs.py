# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense split-K (VLLM_EXL3_DENSE_SPLITK) on real captured inputs.

Inputs: VLLM_EXL3_TEST_PORT_CASES names the case set (layout in
exl3_real_cases.py: cases.json plus one torch file per case) and
VLLM_EXL3_TEST_CHECKPOINT the EXL3 checkpoint the calls ran on (the
Qwen3.8-Flash-Next 5.05 bpw checkpoint). With either unset or missing, the
tests that need it skip and say which. Every case is a dense launch taken
from the served model at tensor parallel size 2, rank 0, 74 multiprocessors:
the launch's input rows, its matrix (checkpoint key stem plus the rank's H128
blocks) and, for whole ("live") launches, the output linear_out gave.

CPU: skip behaviour, case loading, pooling and coverage on a synthetic case
set, the pooled rule on synthetic arrays with a failing negative control,
the FP64 decode against the NumPy one, the required cells against the split
rule, and (case set given) every file's sha256, every case on the split-K
path, and one case per required (rows, phase) cell.

GPU (SM 8.0, as many multiprocessors as the cases were recorded on): per case
the shipped operator (ops.linear_out), the port (ops.linear_out_splitk) and an
FP64 reference from the checkpoint weights. linear_out must give each live
case's recorded output bit for bit (the inputs and weights are the served
ones); per pool (role, rows, phase, drafter) the port holds linear_out's
error against FP64 (test_exl3_dense_splitk._Pool: pooled mean <= 1.10x, max
<= 1.25x, BF16 rounding ties forgiven, exact where linear_out is exact). One
line per pool and per phase's cells is printed; a failing pool fails.

Coverage (REQUIRED_CELLS, from the split rule over served traffic: 1-8
concurrent requests with 3-token speculative decoding, prompts of 17-40
tokens and longer prompts in 3460-token chunks):
- decode 1-8 (drafter rows), 4c and 16c for c = 1-8 (verify rows; the
  drafter's hidden-state projection takes 4 rows per token): live launches;
- prefill 1-8, 12-32 (drafter rows of 1-8 requests), the prompts' 17-40 rows
  and 4x those (68-132), and the prompts' later chunks and row-chunk tails
  (46, 110, 111, 222, 242, 309-330, 579, 639, 1272, 1585-1600): live
  launches;
- prefill 58, 66, 139, 148, 326, 364, 384, 434, 482, 632, 689, 1237, 1452
  (row-chunk tails of the 3460-token chunks): no capture holds such a launch
  whole, so these cases are the last rows of a longer live prefill launch of
  the same matrix ("sliced", no recorded output; both operators run them as
  launches of that many rows).
"""

import collections

import numpy as np
import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_dense_splitk import (
    _bitwise,
    _extension_or_skip,
    _Pool,
    _run,
    needs_sm80,
)
from tests.kernels.quantization.test_exl3_kernels import core_weight as np_core
from tests.kernels.quantization.test_exl3_kernels import effective_weight
from vllm_sm80.kernels.exl3 import ops as exl3_ops

_CONCURRENCY = range(1, 9)
DECODE_ROWS = sorted(
    {*_CONCURRENCY} | {4 * c for c in _CONCURRENCY} | {16 * c for c in _CONCURRENCY}
)
PREFILL_ROWS = sorted(
    {*_CONCURRENCY}
    | {4 * c for c in _CONCURRENCY}
    | {17, 18, 20, 24, 25, 28, 30, 32, 33, 40}
    | {68, 72, 80, 96, 100, 112, 120, 128, 132}
    | {46, 110, 111, 222, 242, 309, 312, 316, 318, 330, 579, 639}
    | {1272, 1585, 1599, 1600}
    | {58, 66, 139, 148, 326, 364, 384, 434, 482, 632, 689, 1237, 1452}
)
REQUIRED_CELLS = {(r, "decode") for r in DECODE_ROWS} | {
    (r, "prefill") for r in PREFILL_ROWS
}
# (padded_out, bits_x2, codebook) of the rank's served dense matrices.
SERVED_MATRICES = [
    (256, 14, 2),  # k/v projections
    (640, 14, 2),  # shared expert gate/up
    (640, 10, 2),  # sparse-attention indexer q/k
    (1280, 12, 2),  # drafter embedding / hidden-state projections
    (2560, 14, 2),  # shared expert down, output projections
    (3072, 14, 2),  # linear-attention z
    (5120, 14, 2),  # linear-attention qkv
    (6144, 14, 2),  # attention q (with output gate)
]
SMS = 74


# ---------------------------------------------------------------- CPU tests


def test_unset_or_missing_inputs_skip_with_the_reason(monkeypatch, tmp_path):
    monkeypatch.delenv(real.CASES_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=real.CASES_ENV):
        real.case_dir_or_skip()
    monkeypatch.setenv(real.CASES_ENV, str(tmp_path / "absent"))
    with pytest.raises(pytest.skip.Exception, match="not a directory"):
        real.case_dir_or_skip()
    monkeypatch.setenv(real.CASES_ENV, str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="no cases.json"):
        real.case_dir_or_skip()
    monkeypatch.delenv(real.CHECKPOINT_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=real.CHECKPOINT_ENV):
        real.checkpoint_or_skip()


def _synthetic_case(directory, index, rows, phase, source, role="r.*.k"):
    g = torch.Generator().manual_seed(index)
    tensors = {"x": torch.randn(rows, 256, generator=g).to(torch.bfloat16)}
    if source == "live":
        tensors["out"] = torch.randn(rows, 128, generator=g).to(torch.bfloat16)
    (directory / "dense").mkdir(exist_ok=True)
    path = directory / "dense" / f"{index}.pt"
    torch.save(tensors, path)
    return dict(
        id=str(index),
        file=f"dense/{index}.pt",
        sha256=real.file_sha256(path),
        prefix=f"r.{index}.k",
        role=role,
        layer=index,
        rows=rows,
        phase=phase,
        drafter=False,
        source=source,
        record_sha256="0" * 64,
        bits_x2=14,
        codebook=2,
        input_slice=[0, 256],
        output_slices=[[0, 128]],
        weight_sha256=None,
        logical_out=128,
    )


def test_case_loading_pooling_and_coverage_on_a_synthetic_set(tmp_path, capsys):
    import json

    cases = [
        _synthetic_case(tmp_path, 0, 4, "decode", "live"),
        _synthetic_case(tmp_path, 1, 4, "decode", "live"),
        _synthetic_case(tmp_path, 2, 17, "prefill", "sliced"),
        _synthetic_case(tmp_path, 3, 4, "prefill", "live", role="r.*.v"),
    ]
    manifest = dict(schema=real.SCHEMA, sm_count=SMS, dense=cases)
    (tmp_path / "cases.json").write_text(json.dumps(manifest))
    header, loaded = real.load_manifest(tmp_path, "dense")
    assert header["sm_count"] == SMS and loaded == cases
    for case in loaded:
        tensors = real.load_case(tmp_path, "dense", case)
        assert tensors["x"].shape == (case["rows"], 256)
    grouped = real.pools(loaded)
    assert [len(v) for v in grouped.values()] == [2, 1, 1]
    assert ("r.*.k", 4, "decode", False) in grouped
    required = {(4, "decode"), (17, "prefill"), (4, "prefill")}
    real.require_cells(loaded, required, "synthetic")
    printed = capsys.readouterr().out
    assert "CELLS decode: 4:2l" in printed
    assert "CELLS prefill: 4:1l 17:1s" in printed
    with pytest.raises(AssertionError, match=r"\(5, 'decode'\)"):
        real.require_cells(loaded, required | {(5, "decode")}, "synthetic")
    # A changed file no longer matches its sha256.
    torch.save(
        {"x": torch.zeros(4, 256, dtype=torch.bfloat16)}, tmp_path / "dense/0.pt"
    )
    with pytest.raises(AssertionError, match="sha256"):
        real.load_case(tmp_path, "dense", loaded[0])


def _rounded(ref, factor):
    """A BF16 output whose error against ref is about factor x BF16 rounding."""
    inc = ref.to(torch.bfloat16)
    return inc, (ref + factor * (inc.double() - ref)).to(torch.bfloat16)


def test_pool_rule_passes_equal_error_and_fails_a_larger_one():
    g = torch.Generator().manual_seed(5)
    passing, failing = _Pool(), _Pool()
    for _ in range(3):
        ref = torch.randn(8, 640, generator=g, dtype=torch.float64)
        inc, _ = _rounded(ref, 1.0)
        passing.add(inc.clone(), inc, ref)
        _, worse = _rounded(ref, 3.0)
        failing.add(worse, inc, ref)
    mean_ratio, max_ratio, ties = passing.check("equal error")
    assert mean_ratio == 1.0 and max_ratio == 1.0 and ties == 0
    with pytest.raises(AssertionError, match="mean ratio"):
        failing.check("negative control")


def test_fp64_decode_matches_the_numpy_decode():
    rng = np.random.default_rng(3)
    for bits_x2, codebook in ((14, 2), (10, 2), (12, 2), (4, 1)):
        trellis = rng.integers(-(2**15), 2**15, (8, 16, 8 * bits_x2)).astype(np.int16)
        core = real.core_weight(torch.from_numpy(trellis), bits_x2, codebook, chunk=37)
        assert torch.equal(core, torch.from_numpy(np_core(trellis, bits_x2, codebook)))
        suh = rng.uniform(0.5, 1.5, 128).astype(np.float16)
        svh = rng.uniform(-0.05, 0.05, 256).astype(np.float16)
        x = torch.from_numpy(rng.standard_normal((5, 100))).to(torch.bfloat16)
        expected = (
            x.double().numpy()
            @ effective_weight(trellis, suh, svh, bits_x2, codebook)[:100]
        )
        actual = real.apply_fp64(x, core, torch.from_numpy(suh), torch.from_numpy(svh))
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_required_cells_are_split_k_launches():
    _extension_or_skip()
    for rows, phase in sorted(REQUIRED_CELLS):
        assert any(
            exl3_ops.linear_splitk_splits(rows, n, SMS, bits_x2, codebook) == 2
            for n, bits_x2, codebook in SERVED_MATRICES
        ), (rows, phase)


def _padded_out(case):
    return sum(b - a for a, b in case["output_slices"])


def test_case_set_files_rule_and_coverage():
    directory = real.case_dir_or_skip()
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "dense")
    for case in cases:
        real.load_case(directory, "dense", case)
        assert (
            exl3_ops.linear_splitk_splits(
                case["rows"],
                _padded_out(case),
                header["sm_count"],
                case["bits_x2"],
                case["codebook"],
            )
            == 2
        ), f"case {case['id']} is not a split-K launch"
    real.require_cells(cases, REQUIRED_CELLS, "dense split-K")


# ---------------------------------------------------------------- GPU tests


@needs_sm80
def test_splitk_holds_linear_outs_error_against_fp64_on_real_inputs():
    directory = real.case_dir_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "dense")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms != header["sm_count"]:
        pytest.skip(
            f"the cases were recorded at {header['sm_count']} multiprocessors, "
            f"this device has {sms}: the split rule differs"
        )
    real.require_cells(cases, REQUIRED_CELLS, "dense split-K")
    device = torch.device("cuda")
    grouped = collections.defaultdict(list)
    for case in cases:
        key = (case["prefix"], tuple(case["input_slice"]))
        grouped[key + tuple(map(tuple, case["output_slices"]))].append(case)
    pool_of = {key: _Pool() for key in real.pools(cases)}
    live = 0
    for (prefix, *_), group in sorted(grouped.items()):
        first = group[0]
        trellis, suh, svh = checkpoint.shard(
            prefix, first["input_slice"], first["output_slices"]
        )
        if first["weight_sha256"]:
            assert real.weight_sha256(trellis, suh, svh) == first["weight_sha256"], (
                f"{prefix}: the checkpoint's shard differs from the recorded one"
            )
        bits_x2, codebook = first["bits_x2"], first["codebook"]
        matrix = checkpoint.matrix(prefix)
        assert (matrix.bits_x2, matrix.codebook) == (bits_x2, codebook), prefix
        m = dict(trellis=trellis.to(device), suh=suh.to(device), svh=svh.to(device))
        core = real.core_weight(m["trellis"], bits_x2, codebook)
        padded_out = m["trellis"].shape[1] * 16
        for case in group:
            tensors = real.load_case(directory, "dense", case)
            x = tensors["x"].to(device)
            rows, logical_out = x.shape[0], case["logical_out"]
            name = f"case {case['id']} ({prefix}, {rows} rows, {case['phase']})"
            assert (
                exl3_ops.linear_splitk_splits(rows, padded_out, sms, bits_x2, codebook)
                == 2
            ), f"{name}: not a split-K launch"
            shipped = _run(exl3_ops.linear_out, m, x, bits_x2, logical_out, codebook)
            port = _run(
                exl3_ops.linear_out_splitk, m, x, bits_x2, logical_out, codebook
            )
            if case["source"] == "live":
                _bitwise(shipped.cpu(), tensors["out"], f"{name}: recorded output")
                live += 1
            ref = real.apply_fp64(x, core, m["suh"], m["svh"])[:, :logical_out]
            pool_of[real.pool_key(case)].add(port.cpu(), shipped.cpu(), ref.cpu())
        del m, core
        torch.cuda.empty_cache()
    failures = []
    for key, pool in pool_of.items():
        role, rows, phase, drafter = key
        name = f"POOL dense {role} rows={rows} {phase} drafter={int(drafter)}"
        group = real.pools(cases)[key]
        sources = collections.Counter(case["source"] for case in group)
        try:
            mean_ratio, max_ratio, ties = pool.check(name)
            print(
                f"{name} cases={len(group)} live={sources['live']} "
                f"sliced={sources['sliced']} mean x{mean_ratio:.4f} "
                f"max x{max_ratio:.4f} ties={ties} PASS"
            )
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"{name} cases={len(group)} FAIL {exc}")
    print(
        f"dense split-K real inputs: {len(cases)} cases ({live} live replayed "
        f"bitwise), {len(pool_of)} pools, {len(failures)} failing"
    )
    assert not failures, failures
