# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense tensor-core blocks (VLLM_EXL3_DENSE_MMA) on real captured inputs.

Inputs as in test_exl3_dense_splitk_real_inputs.py: VLLM_EXL3_TEST_PORT_CASES
names the case set (layout in exl3_real_cases.py) and
VLLM_EXL3_TEST_CHECKPOINT the EXL3 checkpoint the calls ran on. With either
unset or missing, the tests that need it skip and say which. Every case is a
dense launch of the served model at tensor parallel size 2, rank 0, 74
multiprocessors, that the tensor-core blocks take: 1..32 rows of one of the
served matrices, including the LM head (K 2560, N 124160, bits_x2 12),
which the target model calls at 4 rows per request and each of the three
drafter steps at 1 row per request.

CPU (case set given): every file's sha256, every case inside the blocks' rule
(a served matrix), the LM head's cases on the 4-tile plan, and one case per
required (rows, phase) cell for every matrix role (each role on its own), the
target head and the draft heads.

GPU (SM 8.0, as many multiprocessors as the cases were recorded on): per case
the shipped operator (ops.linear_out), the port (ops.linear_out_splitk with
the library switch on) and an FP64 reference from the checkpoint weights.
linear_out must give each live case's recorded output bit for bit; per pool
(role, rows, phase, drafter) the port holds linear_out's error against FP64
(test_exl3_dense_splitk._Pool: pooled mean <= 1.10x, max <= 1.25x, BF16
rounding ties forgiven, exact where linear_out is exact). One line per pool
and per phase's cells is printed; a failing pool fails.

Coverage, per role (the split-K test's traffic: 1-8 concurrent requests with
3-token speculative decoding, prompts of 17-40 tokens and longer prompts in
3460-token chunks; "sliced" cases are the last rows of a longer live call of
the same matrix, phase and drafter):
- target matrices: decode 4-32 by 4 (four rows per request), prefill 1-32
  (17, 18, 20, 24, 25, 28, 30, 32 live; the rest sliced: a last prompt chunk
  of up to 32 tokens, or a row-chunk tail);
- drafter matrices: decode and prefill 1-8 (one row per request), decode
  12-32 by 4, prefill 17, 18, 20, 24, 25, 28, 30, 32; the drafter's
  hidden-state projection decode and prefill 4-32 by 4;
- draft heads: decode and prefill 1-8 (one row per request);
- target head: decode 4-32 by 4 (four rows per request: the token and its
  three drafts), prefill 1-32 (the rows a prefill step samples: one per
  prefilling request and four per decoding one; 1 and 5-11 live, the rest
  sliced).
"""

import collections

import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_dense_mma import SERVED_SHAPES
from tests.kernels.quantization.test_exl3_dense_splitk import (
    _bitwise,
    _extension_or_skip,
    _Pool,
    _run,
    needs_sm80,
)
from vllm_sm80.kernels.exl3 import ops as exl3_ops

MAX_ROWS = 32
HEAD_ROLE = "lm_head"
HEAD = (2560, 124160, 12, 2)  # padded_in, padded_out, bits_x2, codebook
_CONCURRENCY = range(1, 9)
_VERIFY = {4 * c for c in _CONCURRENCY}  # four rows per request
_SHORT_PROMPTS = {17, 18, 20, 24, 25, 28, 30, 32}
TARGET_ROLES = {
    f"model.language_model.layers.*.{name}"
    for name in (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "mlp.shared_expert.down_proj",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "self_attn.indexer.index_qk_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )
}
DRAFT_HIDDEN_ROLE = "mtp.fc_hidden"
DRAFT_ROLES = {"mtp.fc_embedding", DRAFT_HIDDEN_ROLE} | {
    f"mtp.layers.*.{name}"
    for name in (
        "mlp.shared_expert.down_proj",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "self_attn.indexer.index_qk_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )
}
TARGET_REQUIRED_CELLS = {(r, "decode") for r in _VERIFY} | {
    (r, "prefill") for r in range(1, MAX_ROWS + 1)
}
DRAFT_REQUIRED_CELLS = (
    {(r, p) for r in _CONCURRENCY for p in real.PHASES}
    | {(r, "decode") for r in _VERIFY}
    | {(r, "prefill") for r in _SHORT_PROMPTS}
)
DRAFT_HIDDEN_REQUIRED_CELLS = {(r, p) for r in _VERIFY for p in real.PHASES}
HEAD_DRAFT_REQUIRED_CELLS = {(c, p) for c in _CONCURRENCY for p in real.PHASES}
HEAD_TARGET_REQUIRED_CELLS = {(r, "decode") for r in _VERIFY} | {
    (r, "prefill") for r in range(1, MAX_ROWS + 1)
}


def required_cells(role: str, drafter: bool) -> set[tuple[int, str]]:
    """The (rows, phase) cells the case set must hold for a role."""
    if role == HEAD_ROLE:
        return HEAD_DRAFT_REQUIRED_CELLS if drafter else HEAD_TARGET_REQUIRED_CELLS
    if drafter:
        assert role in DRAFT_ROLES, role
        if role == DRAFT_HIDDEN_ROLE:
            return DRAFT_HIDDEN_REQUIRED_CELLS
        return DRAFT_REQUIRED_CELLS
    assert role in TARGET_ROLES, role
    return TARGET_REQUIRED_CELLS


def _padded_out(case):
    return sum(b - a for a, b in case["output_slices"])


def _padded_in(case):
    return case["input_slice"][1] - case["input_slice"][0]


def case_format(case):
    return case["bits_x2"], case["codebook"]


def _require(cases):
    """Every role (drafter or target) holds its required cells on its own."""
    by_role = collections.defaultdict(list)
    for case in cases:
        by_role[(case["role"], case["drafter"])].append(case)
    expected = {(role, False) for role in TARGET_ROLES} | {
        (role, True) for role in DRAFT_ROLES
    }
    expected |= {(HEAD_ROLE, False), (HEAD_ROLE, True)}
    assert set(by_role) == expected, (
        f"roles without cases {sorted(expected - set(by_role))}, unexpected "
        f"roles {sorted(set(by_role) - expected)}"
    )
    for (role, drafter), group in sorted(by_role.items()):
        real.require_cells(
            group,
            required_cells(role, drafter),
            f"dense MMA {role} drafter={int(drafter)}",
        )


# ---------------------------------------------------------------- CPU tests


def test_case_set_files_rule_and_coverage():
    directory = real.case_dir_or_skip()
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "dense")
    plan = torch.ops._exl3_C.linear_mma_plan
    for case in cases:
        real.load_case(directory, "dense", case)
        name = f"case {case['id']}"
        assert 1 <= case["rows"] <= MAX_ROWS, f"{name}: {case['rows']} rows"
        assert case["codebook"] == 2, name
        shape = (_padded_in(case), _padded_out(case), case["bits_x2"])
        assert shape in SERVED_SHAPES, f"{name}: {shape} is not a served matrix"
        if case["role"] == HEAD_ROLE:
            assert (_padded_in(case), _padded_out(case), *case_format(case)) == HEAD
            assert plan(case["rows"], *HEAD[:2], header["sm_count"], *HEAD[2:]) == [
                4,
                1,
                1,
            ], name
    _require(cases)


# ---------------------------------------------------------------- GPU tests


@needs_sm80
def test_mma_holds_linear_outs_error_against_fp64_on_real_inputs():
    directory = real.case_dir_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    _extension_or_skip()
    header, cases = real.load_manifest(directory, "dense")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms != header["sm_count"]:
        pytest.skip(
            f"the cases were recorded at {header['sm_count']} multiprocessors, "
            f"this device has {sms}: the plan rule differs"
        )
    _require(cases)
    library = torch.ops._exl3_C
    library.dense_mma_prepare()
    device = torch.device("cuda")
    grouped = collections.defaultdict(list)
    for case in cases:
        key = (case["prefix"], tuple(case["input_slice"]))
        grouped[key + tuple(map(tuple, case["output_slices"]))].append(case)
    pool_of = {key: _Pool() for key in real.pools(cases)}
    live = 0
    plans = collections.Counter()
    try:
        for (prefix, *_), group in sorted(grouped.items()):
            first = group[0]
            trellis, suh, svh = checkpoint.shard(
                prefix, first["input_slice"], first["output_slices"]
            )
            if first["weight_sha256"]:
                assert (
                    real.weight_sha256(trellis, suh, svh) == first["weight_sha256"]
                ), f"{prefix}: the checkpoint's shard differs from the recorded one"
            bits_x2, codebook = case_format(first)
            matrix = checkpoint.matrix(prefix)
            assert (matrix.bits_x2, matrix.codebook) == (bits_x2, codebook), prefix
            m = dict(trellis=trellis.to(device), suh=suh.to(device), svh=svh.to(device))
            core = real.core_weight(m["trellis"], bits_x2, codebook)
            padded_in = m["trellis"].shape[0] * 16
            padded_out = m["trellis"].shape[1] * 16
            for case in group:
                tensors = real.load_case(directory, "dense", case)
                x = tensors["x"].to(device)
                rows, logical_out = x.shape[0], case["logical_out"]
                name = f"case {case['id']} ({prefix}, {rows} rows, {case['phase']})"
                plan = library.linear_mma_plan(
                    rows, padded_in, padded_out, sms, bits_x2, codebook
                )
                assert plan[0] > 0, f"{name}: not a tensor-core launch"
                plans[tuple(plan)] += 1
                shipped = _run(
                    exl3_ops.linear_out, m, x, bits_x2, logical_out, codebook
                )
                library.set_dense_mma(True)
                port = _run(
                    exl3_ops.linear_out_splitk, m, x, bits_x2, logical_out, codebook
                )
                library.set_dense_mma(False)
                if case["source"] == "live":
                    _bitwise(shipped.cpu(), tensors["out"], f"{name}: recorded output")
                    live += 1
                ref = real.apply_fp64(x, core, m["suh"], m["svh"])[:, :logical_out]
                pool_of[real.pool_key(case)].add(port.cpu(), shipped.cpu(), ref.cpu())
            del m, core
            torch.cuda.empty_cache()
    finally:
        library.set_dense_mma(False)
    failures = []
    for key, pool in pool_of.items():
        role, rows, phase, drafter = key
        name = f"POOL dense-mma {role} rows={rows} {phase} drafter={int(drafter)}"
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
        "dense MMA plans (mt, kw, splits): "
        + " ".join(f"{p}:{n}" for p, n in sorted(plans.items()))
    )
    print(
        f"dense MMA real inputs: {len(cases)} cases ({live} live replayed "
        f"bitwise), {len(pool_of)} pools, {len(failures)} failing"
    )
    assert not failures, failures
