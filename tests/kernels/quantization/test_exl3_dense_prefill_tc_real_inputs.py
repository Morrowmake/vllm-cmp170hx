# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense prefill on the tensor cores (VLLM_EXL3_DENSE_PREFILL_TC) on real
captured inputs.

Inputs: VLLM_EXL3_TEST_PORT_CASES names the case set (layout in
exl3_real_cases.py) and VLLM_EXL3_TEST_CHECKPOINT the EXL3 checkpoint the
calls ran on (the Qwen3.8-Flash-Next 5.05 bpw checkpoint); with either unset
or missing, the tests that need it skip and say which. Every case is a dense
launch of the served model at a validated (tensor, pipeline) layout:
tensor-parallel rank 0 at (2, 1), or the full-width rank 0 of tensor
parallel size 1 with two pipeline stages at (1, 2) (the manifest's
tp_size and pp_size), at 74 multiprocessors, its phase taken from the
scheduler step that made it.

CPU: the required cells against the operator's rule, and (case set given)
every file's sha256 and one case per required (rows, phase) cell.

GPU (SM 8.0, as many multiprocessors as the cases were recorded on), per
case: linear_out gives each live case's recorded output bit for bit; on
every prefill launch the operator covers, linear_out_prefill against an FP64
reference from the checkpoint weights, pooled per (role, rows, phase,
drafter) within linear_out's error (pooled mean <= 1.10x, max <= 1.25x, BF16
rounding ties forgiven, exact where linear_out is exact), with each pool's
mean and max absolute error of both operators printed (prefill launches that
stay on split-K, such as the k/v 482-row chunk tail, are pooled the same way
for linear_out_splitk); through _linear with
the switch on: decode-step calls (the drafter's hidden-state projection at
80-256 rows) and every launch the operator does not take (with split-K on,
the k/v 482-row chunk tail keeps linear_out_splitk) are bitwise today's
operator. A failing pool fails.

Coverage: per layout (the manifest's tp_size and pp_size), every prefill
launch row count the dispatch sends to linear_out_prefill with split-K off
(a superset of split-K on) among the launches of 3460 / 1906-token chunks
of an 8,826-token prompt split by the row chunk and every captured prefill
call of more than 64 rows, plus (TP2) the decode cells of more than 64 rows
(kept on today's path); a case set of a layout that is not validated skips.
Sliced cases (the last rows of a longer live launch of the same matrix and
phase) stand for chunk tails no capture holds whole.
"""

import collections
from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_dense_splitk import (
    _bitwise,
    _extension_or_skip,
    _Pool,
    _run,
    needs_sm80,
)
from vllm.sm80.kernels.exl3 import dense_prefill_tc as tc
from vllm.sm80.kernels.exl3 import ops as exl3_ops

TC_PREFILL_ROWS = [
    96, 110, 111, 132, 160, 172, 180, 184, 195, 215, 222, 242, 261, 273, 297,
    309, 311, 312, 316, 318, 330, 372, 417, 426, 442, 457, 465, 473, 482, 483,
    493, 500, 507, 508, 529, 531, 555, 571, 579, 592, 622, 636, 637, 639, 720,
    721, 736, 814, 823, 839, 854, 860, 862, 879, 880, 910, 937, 943, 952, 960,
    963, 966, 968, 992, 1016, 1072, 1092, 1108, 1188, 1232, 1236, 1248, 1264,
    1272, 1276, 1320, 1336, 1489, 1585, 1599, 1600, 1656, 1680, 1684, 1880,
    1906, 1915, 1946, 1954, 1972, 2028, 2032, 2184, 2368, 2399, 2426, 2457,
    2481, 2621, 2978,
]  # fmt: skip
DECODE_CONTROL_ROWS = [80, 96, 112, 128, 176, 192, 208, 224, 240, 256]
# Prefill launch rows of the reference prompt on the full-width matrices of
# the TP1 / PP2 layout: the 3460 and 1906 scheduler chunks split at each
# role's row chunk floor(8 Mi / (K + N)); see PP2_SERVED_LAUNCHES in
# test_exl3_dense_prefill_tc.
TC_PREFILL_ROWS_PP2 = [
    76, 184, 185, 214, 268, 564, 571, 596, 655, 730, 839, 943, 963, 1638,
    1906, 2621, 2730,
]  # fmt: skip


def layout_of(header):
    """The (tensor, pipeline) layout of a case set; manifests predating
    the pipeline field were captured at pipeline parallel size 1."""
    return (header["tp_size"], header.get("pp_size", 1))


def required_cells(layout):
    """The (rows, phase) cells a case set of a validated layout must
    cover: the served prefill launches of that layout, plus (TP2 only)
    the decode controls of more than 64 rows."""
    if layout == (2, 1):
        return {(r, "prefill") for r in TC_PREFILL_ROWS} | {
            (r, "decode") for r in DECODE_CONTROL_ROWS
        }
    return {(r, "prefill") for r in TC_PREFILL_ROWS_PP2}


DECODE_QUERY_LEN = 4
SMS = 74


def _padded(case):
    padded_in = case["input_slice"][1] - case["input_slice"][0]
    return padded_in, sum(b - a for a, b in case["output_slices"])


def _takes(case, splitk, matrices):
    padded_in, padded_out = _padded(case)
    return tc.takes_launch(
        case["rows"],
        padded_in,
        padded_out,
        case["bits_x2"],
        case["codebook"],
        splitk,
        SMS,
        matrices,
    )


# ---------------------------------------------------------------- CPU tests


def test_required_cells_follow_the_operator_rule():
    _extension_or_skip()
    # Every required prefill row count is covered for some matrix of its
    # layout; the decode controls are calls of more than 64 rows.
    cells = [
        (rows, sorted(tc.VALIDATED_MATRICES[layout]))
        for layout, row_counts in (
            ((2, 1), TC_PREFILL_ROWS),
            ((1, 2), TC_PREFILL_ROWS_PP2),
        )
        for rows in row_counts
    ]
    for rows, served in cells:
        assert any(tc.linear_prefill_applies(rows, k, n, b, 2) for k, n, b in served), (
            rows
        )
    assert min(DECODE_CONTROL_ROWS) > tc.MAX_SHIPPED_ROWS


def test_case_set_files_and_coverage():
    directory = real.case_dir_or_skip()
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "dense")
    assert header["sm_count"] == SMS
    layout = layout_of(header)
    if layout not in tc.VALIDATED_MATRICES:
        pytest.skip(f"case-set layout (tp, pp) {layout} is not validated")
    for case in cases:
        real.load_case(directory, "dense", case)
    real.require_cells(cases, required_cells(layout), "dense prefill tc")
    taken = {
        (c["rows"], c["phase"])
        for c in cases
        if c["phase"] == "prefill" and _takes(c, False, tc.VALIDATED_MATRICES[layout])
    }
    rows = TC_PREFILL_ROWS if layout == (2, 1) else TC_PREFILL_ROWS_PP2
    assert {(r, "prefill") for r in rows} <= taken


# ---------------------------------------------------------------- GPU tests


def _step(monkeypatch, max_query_len):
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            attn_metadata={
                "layers.3.self_attn": SimpleNamespace(max_query_len=max_query_len)
            }
        ),
    )


def _linear(m, x, case, splitk, monkeypatch, scratch):
    monkeypatch.setattr(exl3_ops, "_dense_splitk", splitk)
    padded_in, padded_out = _padded(case)
    return exl3_ops._linear(
        x,
        m["trellis"],
        m["suh"],
        m["svh"],
        case["bits_x2"],
        case["codebook"],
        case["logical_out"],
        exl3_ops.linear_row_chunk(padded_in, padded_out),
    )


@needs_sm80
def test_prefill_holds_linear_outs_error_against_fp64_on_real_inputs(
    monkeypatch, request
):
    directory = real.case_dir_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "dense")
    layout = layout_of(header)
    if layout not in tc.VALIDATED_MATRICES:
        pytest.skip(f"case-set layout (tp, pp) {layout} is not validated")
    cells = required_cells(layout)
    matrices = tc.VALIDATED_MATRICES[layout]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms != header["sm_count"]:
        pytest.skip(
            f"cases recorded at {header['sm_count']} multiprocessors, device has {sms}"
        )
    tc.set_prefill_arithmetic(int(layout == (1, 2)))
    request.addfinalizer(lambda: tc.set_prefill_arithmetic(0))
    real.require_cells(cases, cells, "dense prefill tc")
    device = torch.device("cuda")
    monkeypatch.setattr(tc, "active", True)
    monkeypatch.setattr(tc, "active_layout", layout)
    scratch = torch.empty(exl3_ops.linear_scratch_elements(), device=device)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: scratch)
    monkeypatch.setattr(tc, "_max_decode_query_len", DECODE_QUERY_LEN)
    monkeypatch.setattr(tc, "_sms", sms)
    monkeypatch.setattr(tc, "_served_matrices", matrices)
    grouped = collections.defaultdict(list)
    for case in cases:
        key = (case["prefix"], tuple(case["input_slice"]))
        grouped[key + tuple(map(tuple, case["output_slices"]))].append(case)
    pools = real.pools(cases)
    pool_of: dict[tuple, _Pool] = {}
    errors: collections.defaultdict[tuple, list[float]] = collections.defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    counts: collections.Counter[str] = collections.Counter()
    for (prefix, *_), group in sorted(grouped.items()):
        first = group[0]
        trellis, suh, svh = checkpoint.shard(
            prefix, first["input_slice"], first["output_slices"]
        )
        assert real.weight_sha256(trellis, suh, svh) == first["weight_sha256"], prefix
        bits_x2, codebook = first["bits_x2"], first["codebook"]
        m = dict(trellis=trellis.to(device), suh=suh.to(device), svh=svh.to(device))
        core = real.core_weight(m["trellis"], bits_x2, codebook)
        for case in group:
            tensors = real.load_case(directory, "dense", case)
            x = tensors["x"].to(device)
            rows, logical_out = x.shape[0], case["logical_out"]
            name = f"case {case['id']} ({prefix}, {rows} rows, {case['phase']})"
            shipped = _run(exl3_ops.linear_out, m, x, bits_x2, logical_out, codebook)
            if case["source"] == "live":
                _bitwise(shipped.cpu(), tensors["out"], f"{name}: recorded output")
                counts["live"] += 1
            # Through _linear with the switch on: everything the operator does
            # not take is today's operator, bit for bit.
            for splitk in (False, True):
                default = exl3_ops.linear_out_splitk if splitk else exl3_ops.linear_out
                _step(
                    monkeypatch,
                    rows if case["phase"] == "prefill" else DECODE_QUERY_LEN,
                )
                got = _linear(m, x, case, splitk, monkeypatch, scratch)
                padded_in, padded_out = _padded(case)
                chunk = exl3_ops.linear_row_chunk(padded_in, padded_out)
                assert rows <= chunk, f"{name}: one launch per case"
                if case["phase"] == "prefill" and _takes(case, splitk, matrices):
                    expected = _run(
                        tc.linear_out_prefill, m, x, bits_x2, logical_out, codebook
                    )
                    counts[f"tc splitk={int(splitk)}"] += 1
                else:
                    expected = _run(default, m, x, bits_x2, logical_out, codebook)
                    counts[f"kept {case['phase']} splitk={int(splitk)}"] += 1
                _bitwise(got.cpu(), expected.cpu(), f"{name}: _linear split-K {splitk}")
            if case["phase"] != "prefill":
                continue
            padded_in, padded_out = _padded(case)
            if _takes(case, False, matrices):
                operator, kind = tc.linear_out_prefill, "dense-prefill-tc"
            elif (
                exl3_ops.linear_splitk_splits(rows, padded_out, sms, bits_x2, codebook)
                == 2
            ):
                # Kept on split-K with split-K on (e.g. the k/v 482-row chunk tail).
                operator, kind = exl3_ops.linear_out_splitk, "dense-prefill-kept-splitk"
            else:
                continue
            port = _run(operator, m, x, bits_x2, logical_out, codebook)
            ref = real.apply_fp64(x, core, m["suh"], m["svh"])[:, :logical_out]
            key = (kind, *real.pool_key(case))
            pool_of.setdefault(key, _Pool()).add(port.cpu(), shipped.cpu(), ref.cpu())
            e = errors[key]
            ep, es = (port.double() - ref).abs(), (shipped.double() - ref).abs()
            e[0] += float(ep.sum())
            e[1] = max(e[1], float(ep.max()))
            e[2] += float(es.sum())
            e[3] = max(e[3], float(es.max()))
            e[4] += ep.numel()
        del m, core
        torch.accelerator.empty_cache()
    failures = []
    for pool_key, pool in sorted(pool_of.items()):
        kind, role, rows, phase, drafter = pool_key
        name = f"POOL {kind} {role} rows={rows} {phase} drafter={int(drafter)}"
        group = pools[pool_key[1:]]
        sources = collections.Counter(c["source"] for c in group)
        e = errors[pool_key]
        detail = (
            f"cases={len(group)} live={sources['live']} sliced={sources['sliced']} "
            f"port mean|err| {e[0] / e[4]:.3e} max {e[1]:.3e} "
            f"linear_out mean|err| {e[2] / e[4]:.3e} max {e[3]:.3e}"
        )
        try:
            mean_ratio, max_ratio, ties = pool.check(name)
            print(
                f"{name} {detail} mean x{mean_ratio:.4f} max x{max_ratio:.4f} "
                f"ties={ties} PASS"
            )
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"{name} {detail} FAIL {exc}")
    print(
        f"dense prefill tc real inputs: {len(cases)} cases ({counts['live']} live "
        f"replayed bitwise), {len(pool_of)} FP64 pools, paths {dict(counts)}, "
        f"{len(failures)} failing"
    )
    assert not failures, failures
