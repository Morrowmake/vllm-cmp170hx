# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64_cases.py on synthetic sources in the captured format.

CPU: every family except the EXL3 MoE runs end to end through
``fp64_cases.run`` (on CPU the candidate is the torch path): one well-formed
line per case, the exact families exact against their independent
definitions, wrong or row-dependent candidates reported, a case that cannot
run left out and named. Each family's capture point takes the arguments the
driver reads, and the plain-argument points compose the layer functions.
The EXL3 MoE's expert matrices (target and drafter layers, from the module
prefix and expert map) and FP64 reference (with the SwiGLU clamp) run on
the tiny checkpoint against a direct sum. Routed experts split across the
tensor-parallel ranks: each rank's columns of every expert matrix, the
ranks' FP64 partials summing to the whole experts' output, and the split
cases derived from whole-expert calls, on the tiny checkpoint and (when
VLLM_DSV41_TEST_EXL3_CHECKPOINT, VLLM_DSV41_TEST_MOE_CAPTURE and, for the
prefill calls, VLLM_DSV41_TEST_MOE_PREFILL_CAPTURE are set) on the served
checkpoint and captures of its whole experts. GPU (skips without CUDA): the
same cases through the CUDA paths.

    pytest -q tests/models/deepseek_v41/sm80/test_fp64_cases.py
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

from . import exl3_fixture as fixture
from . import fp64_cases as F
from .test_sm80_kernels_gpu import _attn_case, _index_case

BF16 = torch.bfloat16
CPU = torch.device("cpu")
KEYS = {"cand_mean", "cand_max", "ref_mean", "ref_max"}
EXACT = ("qdq_fp8", "qdq_fp4:e8m0", "qdq_fp4:e4m3", "engram_dequant")


def _source(directory, name, family, tensors, schema=F.SOURCE_SCHEMA):
    path = directory / f"{name}.pt"
    meta = {"step": 0, "phase": "decode", "num_reqs": 1, "layer": 0, "rank": 0,
            "module_prefix": "model.layers.0.attn", "family": family}
    torch.save({"schema": schema, "family": family, "key": name, "seq": 0,
                "meta": meta, "tensors": tensors}, path)
    return str(path)


def _attention(g, compressed=True):
    q, swa, swa_idx, comp, table, local, req, sink = _attn_case(g)
    t = dict(q=q, swa_cache=swa, swa_indices=swa_idx, comp_cache=comp,
             comp_block_table=table, comp_local=local, req=req, sink=sink,
             scale=512**-0.5)
    if not compressed:
        t.update(comp_cache=None, comp_block_table=None, comp_local=None)
    return t


def _index(g):
    q, w, n_vis, req, idxk, table = _index_case(g)
    return dict(q=q, w=w, n_vis=n_vis, req=req, idxk=idxk, block_table=table,
                n_cols=int(n_vis.max()), heads_per_rank=8)


def _activations(g, d=512):
    x = torch.randn(13, 4, d, generator=g) * torch.logspace(-6, 3, 13)[:, None, None]
    x[0, 0] = 0.0
    x[1, 1, :4] = torch.tensor([448.0, -448.0, 6.0, 1e-30])
    return x.to(BF16)


def _compress(g, ratio):
    """Two requests: one whose groups close inside its chunk, one whose
    first token closes a group begun in the previous chunk (ring)."""
    d, capacity = 64, 4
    pos = torch.tensor([4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11])
    req = torch.tensor([0] * 6 + [1] * 5)
    t = dict(
        kv_score=torch.randn(pos.shape[0], ratio * d, generator=g),
        norm_weight=(1 + 0.1 * torch.randn(d, generator=g)).to(BF16),
        positions=pos,
        token_to_req=req,
        query_start_loc=torch.tensor([0, 6, 11]),
        comp_slots=torch.arange(pos.shape[0]),
        compress_ratio=ratio,
        head_dim=d,
        eps=1e-6,
    )
    if ratio == 1:
        t["comp_slots"] = torch.where(pos % 3 == 0, t["comp_slots"], -1)
        t["ring"] = t["ring_slots"] = None
    else:
        blocks = torch.tensor([2, 0])
        t["ring"] = torch.randn(3, capacity, 2 * d, generator=g)
        t["ring_slots"] = blocks[req] * capacity + pos % capacity
    return t


def _index_keys(g, n=9, d=128):
    return dict(
        k=torch.randn(n, d, generator=g).to(BF16),
        k_norm=(1 + 0.1 * torch.randn(d, generator=g)).to(BF16),
        freqs=core.rope_freqs(64, 0, 10000.0, 1.0, 32, 1),
        group_pos=torch.randint(0, 5000, (n,), generator=g),
        eps=1e-6,
    )


def _engram(g, n=21, dim=256):
    values = torch.randint(0, 256, (n, dim), generator=g).to(torch.uint8)
    values[0, :2] = torch.tensor([0x7F, 0xFF], dtype=torch.uint8)  # NaN patterns
    scales = torch.randint(110, 140, (n, dim // 32), generator=g).to(torch.uint8)
    scales[0, 1] = 0  # 2**-127: products in the FP32 denormals
    return dict(values=values, scales=scales)


def _mhc(g, pre_only=False):
    """A delayed mHC call (6 tokens: one kernel A form on CUDA), fused or
    pre-only, in the capture point's arguments."""
    from .test_sm80_mhc_fused import make_inputs

    a = make_inputs(6, 256, seed=int(torch.randint(0, 1 << 30, (1,), generator=g)))
    if pre_only:
        a.update(x=None, post_layer_mix=None, comb_res_mix=None)
    return {k: v for k, v in a.items() if v is not None}


def _rope_freqs():
    return core.rope_freqs(64, 0, 10000.0, 1.0, 32, 1)


def _attn_prologue(g, n=11):
    """A fused-prologue call: a token whose slot is -1 writes nothing."""
    slots = torch.randperm(5 * 64, generator=g)[:n] + 64
    slots[3] = -1
    return dict(
        qr_kv=(torch.randn(n, 1792, generator=g) * 3).to(BF16),
        q_norm=(1 + 0.1 * torch.randn(1280, generator=g)).to(BF16),
        kv_norm=(1 + 0.1 * torch.randn(512, generator=g)).to(BF16),
        positions=torch.randint(0, 9000, (n,), generator=g),
        freqs=_rope_freqs(),
        swa_cache=(torch.randn(6, 64, 512, generator=g) * 0.5).to(BF16),
        swa_slots=slots,
        eps=1e-20,
    )


def _attn_attend(g, compressed=True):
    q, swa, swa_idx, comp, table, local, req, sink = _attn_case(g)
    n = q.shape[0]
    t = dict(q=q, positions=torch.randint(0, 9000, (n,), generator=g),
             freqs=_rope_freqs(), swa_cache=swa, swa_indices=swa_idx, comp_cache=comp,
             comp_block_table=table, comp_local=local, req=req.int(), attn_sink=sink,
             scale=512**-0.5, n_groups=2, out=torch.zeros(n, 16, 512, dtype=BF16))
    if not compressed:
        t.update(comp_cache=None, comp_block_table=None, comp_local=None)
    return t


# mHC case ids: <family>:<output> (the output the case judges)
MHC = ("mhc_delayed:comb", "mhc_delayed:layer_input.rp", "mhc_pre_delayed:pre")
# Fused attention core case ids (the prologue's cases name their output)
FUSED_ATTN = ("attn_prologue:qr", "attn_prologue:window", "attn_attend")
MULTI = MHC + FUSED_ATTN[:2]


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    d = tmp_path_factory.mktemp("sources")
    g = torch.Generator().manual_seed(7)
    sources = {
        "sparse_attention": [
            _source(d, "attn0", "sparse_attention", _attention(g)),
            _source(d, "attn1", "sparse_attention", _attention(g, compressed=False)),
        ],
        "index_scores": [_source(d, "index", "index_scores", _index(g))],
        "qdq_fp8": [_source(d, "fp8", "qdq_fp8", dict(x=_activations(g), block=32))],
        "qdq_fp4:e8m0": [
            _source(d, "fp4a", "qdq_fp4", dict(x=_activations(g), block=32, scale="e8m0"))
        ],
        "qdq_fp4:e4m3": [
            _source(d, "fp4b", "qdq_fp4", dict(x=_activations(g), block=16, scale="e4m3"))
        ],
        "compress": [
            _source(d, "comp1", "compress", _compress(g, 1)),
            _source(d, "comp2", "compress", _compress(g, 2)),
        ],
        "index_keys": [_source(d, "keys", "index_keys", _index_keys(g))],
        "engram_dequant": [_source(d, "engram", "engram_dequant", _engram(g))],
        "mhc_delayed:comb": [_source(d, "mhc", "mhc_delayed", _mhc(g))],
        "mhc_delayed:layer_input.rp": [_source(d, "mhc_li", "mhc_delayed", _mhc(g))],
        "mhc_pre_delayed:pre": [_source(d, "mhc_pre", "mhc_pre_delayed", _mhc(g, True))],
        "attn_prologue:qr": [_source(d, "pro", "attn_prologue", _attn_prologue(g))],
        "attn_prologue:window": [
            _source(d, "pro_w", "attn_prologue", _attn_prologue(g))
        ],
        "attn_attend": [
            _source(d, "att0", "attn_attend", _attn_attend(g)),
            _source(d, "att1", "attn_attend", _attn_attend(g, compressed=False)),
        ],
    }
    return [
        {"id": k, "family": k.split(":")[0], "phase": "decode", "concurrency": 1,
         "calls": len(v), "captured_sources": v,
         **({"output": k.split(":")[1]} if k in MULTI else {})}
        for k, v in sources.items()
    ]


def _run(tmp_path, cases, device=CPU, schema="served_cases/1", target="/nonexistent"):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema": schema, "model": "m", "cases": cases,
                                "checkpoint": {"target": str(target)}}))
    out = tmp_path / "out.jsonl"
    rc = F.run(path, out, device)
    if rc not in (0, 1):
        return rc, None, None
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert set(lines[0]) == {"header"}
    return rc, lines[0]["header"], {r["case"]: r for r in lines[1:]}


def _well_formed(row):
    assert isinstance(row["bitwise"], bool) and isinstance(row["row_invariance"], bool)
    assert isinstance(row["numel"], int) and row["numel"] > 0
    assert set(row["fp64"]) == KEYS
    for v in row["fp64"].values():
        assert isinstance(v, float) and math.isfinite(v) and v >= 0


def test_every_family_runs_on_cpu(tmp_path, cases):
    rc, head, rows = _run(tmp_path, cases)
    assert rc == 0
    assert set(head) == {"torch", "triton", "cuda", "gpu", "sm_count", "reference_math"}
    assert head["reference_math"] == {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    assert set(rows) == {c["id"] for c in cases}
    for cid, row in rows.items():
        _well_formed(row)
        # mHC on CPU is the torch formulas, whose FP32 projection (a BLAS
        # GEMM) takes another path for one row than for six
        assert row["row_invariance"] or cid in MHC, cid
        e = row["fp64"]
        if cid == "sparse_attention":
            # The torch definition against the reference kernel's arithmetic:
            # both within BF16 rounding of the FP64 value.
            assert 0 < e["cand_max"] < 0.05 and 0 < e["ref_max"] < 0.05
            continue
        assert row["bitwise"], cid  # on CPU both paths are the torch definition
        assert (e["cand_mean"], e["cand_max"]) == (e["ref_mean"], e["ref_max"]), cid
        if cid.startswith("qdq"):
            # Exact: the definition equals the independent one everywhere.
            assert e["cand_max"] == 0.0, cid
        if cid in ("index_scores", "compress"):
            assert e["cand_max"] > 0, cid  # their BF16 rounding is measured
    assert rows["compress"]["family"] == "compress"
    for cid in MHC:
        assert rows[cid]["family"] == cid, cid
    for cid in FUSED_ATTN:
        assert rows[cid]["family"] == cid, cid


def test_an_mhc_case_judges_the_output_it_names(tmp_path, cases, capsys):
    """The case's output, else its family's or the cases file's suffix after
    ':'; none, or a name that is not an output: the case gets no line and
    the run is refused."""
    mhc = next(c for c in cases if c["id"] == "mhc_delayed:comb")
    bare = {k: v for k, v in mhc.items() if k != "output"}
    rc, _, rows = _run(tmp_path, [dict(bare, family="mhc_delayed:post.rp")])
    assert rc == 0 and rows[bare["id"]]["family"] == "mhc_delayed:post.rp"
    assert rows[bare["id"]]["numel"] == 6 * 4
    rc, _, rows = _run(tmp_path, [dict(bare, output="layer_input")])
    assert rows[bare["id"]]["family"] == "mhc_delayed:layer_input"
    assert rows[bare["id"]]["numel"] == 6 * 256
    rc, _, rows = _run(tmp_path, [dict(bare, id="none"), dict(bare, id="typo", output="comb.x")])
    assert rc == 1 and rows == {}
    err = capsys.readouterr().err
    assert "case none: ValueError: mHC cases name their output" in err
    assert "case typo: ValueError: mHC output 'comb.x'" in err
    pre = next(c for c in cases if c["id"] == "mhc_pre_delayed:pre")
    rc, _, rows = _run(tmp_path, [dict(pre, output="residual")])
    assert rc == 1 and rows == {}
    assert "the pre-only call has no residual output" in capsys.readouterr().err


def test_every_family_has_a_point_whose_parameters_are_its_arguments():
    import importlib
    import inspect

    assert set(F.POINTS) == set(F.FAMILIES)
    for family, (point, args) in F.POINTS.items():
        module, qualname = point.split(":")
        fn = importlib.import_module(module)
        for part in qualname.split("."):
            fn = getattr(fn, part)
        params = list(inspect.signature(fn).parameters)
        assert [p for p in params if p in args] == args, family


def test_the_points_compose_the_layer_functions():
    """compress / index_keys are compress_rows / index_key_rows of their
    arguments: the recorded call is the one the layer makes."""
    g = torch.Generator().manual_seed(11)
    a = _compress(g, 2)
    pos = a["positions"]
    batch = core.Sm80AttnBatch(
        positions=pos, token_to_req=a["token_to_req"],
        query_start_loc=a["query_start_loc"], swa_slots=torch.full_like(pos, -1),
        swa_indices=pos.new_empty((pos.shape[0], 0)), comp_slots=a["comp_slots"],
        ring_slots=a["ring_slots"],
    )
    cfg = type("Cfg", (), dict(compress_ratio=2, head_dim=a["head_dim"], eps=a["eps"]))
    ring = a["ring"].clone()
    rows, latent = core.compress(cfg, batch, a["kv_score"], a["norm_weight"], ring)
    ring2 = a["ring"].clone()
    rows2, latent2 = core.compress_rows(
        a["kv_score"], a["norm_weight"], ring2, pos, a["token_to_req"],
        a["query_start_loc"], a["comp_slots"], a["ring_slots"], 2, a["head_dim"], a["eps"],
    )
    assert torch.equal(rows, rows2) and torch.equal(latent, latent2)
    assert torch.equal(ring, ring2) and not torch.equal(ring, a["ring"])
    k = _index_keys(g)
    weights = type("Wts", (), dict(wk=staticmethod(lambda _x: k["k"]),
                                   k_norm=k["k_norm"], freqs=k["freqs"]))
    got = core.index_keys(type("Cfg", (), dict(eps=k["eps"])), weights, None, k["group_pos"])
    want = core.index_key_rows(k["k"], k["k_norm"], k["freqs"], k["group_pos"], k["eps"])
    assert torch.equal(got, want)


def test_a_wrong_candidate_shows_in_its_bits_and_error(tmp_path, cases, monkeypatch):
    real = core.qdq_fp8

    def off(x, block=32):
        y = real(x, block).clone()
        y.view(-1)[5] = y.view(-1)[5] * 2 + 1
        return y

    monkeypatch.setattr(core, "qdq_fp8", off)
    _, _, rows = _run(tmp_path, [c for c in cases if c["id"] == "qdq_fp8"])
    row = rows["qdq_fp8"]
    assert not row["bitwise"]
    assert row["fp64"]["cand_max"] > 0 == row["fp64"]["ref_max"]


def test_a_row_dependent_candidate_is_reported(tmp_path, cases, monkeypatch):
    real = core.sparse_attention

    def batch_dependent(q, *args):
        return real(q, *args) + (q.shape[0] > 1)

    monkeypatch.setattr(core, "sparse_attention", batch_dependent)
    _, _, rows = _run(tmp_path, [c for c in cases if c["id"] == "sparse_attention"])
    assert not rows["sparse_attention"]["row_invariance"]


def test_a_case_that_cannot_run_is_named_and_refuses_the_run(tmp_path, cases, capsys):
    good = [c for c in cases if c["id"] == "engram_dequant"]
    src = good[0]["captured_sources"][0]
    bad = _source(tmp_path, "bad", "unknown", {})
    old = _source(tmp_path, "old", "engram_dequant", {}, schema="lab-gemm-capture/1")
    moe = str(tmp_path / "moe.pt")
    torch.save({"schema": F.SOURCE_SCHEMA, "family": "exl3_moe_out", "meta": {"layer": 0},
                "tensors": {"x": torch.zeros(1, 8)}}, moe)

    def case(cid, family, sources):
        return {"id": cid, "family": family, "concurrency": 1, "phase": "decode",
                "captured_sources": sources}

    extra = [
        case("no_family", "unknown", [bad]),
        case("other_schema", "engram_dequant", [old]),
        case("wrong_family", "qdq_fp8", [src]),
        {"id": "no_sources", "concurrency": 2, "phase": "prefill"},
        case("moe_without_prefix", "exl3_moe_out", [moe]),
    ]
    rc, _, rows = _run(tmp_path, good + extra)
    # The run is refused (exit 1) after writing the cases that ran.
    assert rc == 1 and set(rows) == {"engram_dequant"}
    err = capsys.readouterr().err
    assert ("5 case(s) without a result: no_family, other_schema, wrong_family, "
            "no_sources, moe_without_prefix") in err
    assert "case no_family: ValueError" in err and "'unknown'" in err
    assert "case other_schema: ValueError" in err and "lab-gemm-capture/1" in err
    assert "case wrong_family: ValueError" in err and "'qdq_fp8'" in err
    assert "case no_sources: ValueError: no captured_sources" in err
    assert "case moe_without_prefix: ValueError" in err and "module_prefix" in err


def test_a_file_of_another_schema_is_refused(tmp_path, cases):
    assert _run(tmp_path, cases, schema="capml/1")[0] == 2


def test_abs_errors():
    exact = torch.tensor([1.5, float("nan"), float("inf")], dtype=torch.float64)
    assert F.abs_errors(torch.tensor([1.0, float("nan"), float("inf")]), exact) == (
        0.5, 0.5, 1,
    )
    assert F.abs_errors(torch.tensor([1.5, 0.0, float("inf")]), exact)[1] == math.inf
    assert F.abs_errors(torch.tensor([float("inf")]), torch.tensor([1.0]))[1] == math.inf
    with pytest.raises(ValueError, match="shape"):
        F.abs_errors(torch.zeros(2), torch.zeros(3))


def _bf16_nearest_even(v: float) -> float:
    """Independent definition: the nearest BF16 value of ``v`` by exact rational arithmetic, ties to even."""
    from fractions import Fraction

    if v == 0.0 or not math.isfinite(v):
        return v
    _, e = math.frexp(abs(v))  # |v| = m * 2**e, m in [0.5, 1)
    ulp = Fraction(2) ** (max(e, -125) - 8)
    q, rem = divmod(Fraction(abs(v)), ulp)
    if rem * 2 > ulp or (rem * 2 == ulp and q % 2):
        q += 1
    return math.copysign(float(q * ulp), v)


def test_bf16_rn_rounds_fp64_once():
    # The two window elements of the hybrid capture's verify M18 ratio-1 case: each FP64 value lies just
    # below a BF16 midpoint, FP32 rounds it onto the midpoint, and the tie then goes up.
    m18 = torch.tensor([0.48339843055547654, 0.12084960763886914], dtype=torch.float64)
    assert F.bf16_rn(m18).double().tolist() == [0.482421875, 0.12060546875]
    assert m18.to(BF16).double().tolist() == [0.484375, 0.12109375]  # torch: through FP32

    one = 1.0
    ties = torch.tensor([one + 2**-8, one + 3 * 2**-8, -(one + 2**-8), 2**-134, 3 * 2**-134],
                        dtype=torch.float64)
    assert F.bf16_rn(ties).double().tolist() == [1.0, one + 2**-6, -1.0, 0.0, 2**-132]
    special = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0, 3.4e38, -3.4e38],
                           dtype=torch.float64)
    got = F.bf16_rn(special).double()
    assert got[:2].tolist() == [math.inf, -math.inf] and got[2].isnan()
    assert got[3] == 0.0 and math.copysign(1.0, float(got[3])) == -1.0
    assert got[4:].tolist() == [math.inf, -math.inf]

    g = torch.Generator().manual_seed(5)
    wide = torch.randn(4096, generator=g, dtype=torch.float64) * torch.exp2(
        torch.randint(-140, 120, (4096,), generator=g).double())
    assert F.bf16_rn(wide).double().tolist() == [_bf16_nearest_even(v) for v in wide.tolist()]
    # Where FP32 holds the value exactly, the single and the double rounding agree.
    exact32 = wide.float().double()
    assert torch.equal(F.bf16_rn(exact32), exact32.float().to(BF16))



@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU refusal")
def test_main_refuses_without_cuda(tmp_path, capsys):
    assert F.main(["--cases", str(tmp_path / "c"), "--out", str(tmp_path / "o")]) == 2
    assert "no CUDA device" in capsys.readouterr().err


# ---------------------------------------------------------------- EXL3 MoE


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def test_moe_experts_from_the_checkpoint_and_their_fp64_output(checkpoint):
    from tests.kernels.quantization import exl3_real_cases as real
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    ckpt = real.Checkpoint(checkpoint)

    # A target layer, and a DSpark drafter layer (vLLM numbers it after the
    # target's layers; its experts are mtp.<stage>.* in the checkpoint).
    expert_map, meta = _recorded(
        ckpt, "layers.0.ffn.experts", (3, 7), fixture.TEXT["n_routed_experts"]
    )
    weights, pointers, got = F.expert_set(
        str(checkpoint), "model.layers.0.ffn.experts", expert_map, meta, CPU
    )
    assert torch.equal(got, meta) and pointers.shape == (2, 3, 3)
    assert torch.equal(weights[3][0], ckpt.whole("layers.0.ffn.experts.7.w1")[0])
    with pytest.raises(ValueError, match="layout"):
        F.expert_set(str(checkpoint), "model.layers.0.ffn.experts", expert_map, meta + 1, CPU)
    draft_map, draft_meta = _recorded(
        ckpt, "mtp.0.ffn.experts", (2, 5), fixture.TEXT["dspark_n_routed_experts"]
    )
    layers = fixture.TEXT["num_hidden_layers"]
    draft, _, _ = F.expert_set(
        str(checkpoint), f"model.layers.{layers}.ffn.experts", draft_map, draft_meta, CPU
    )
    assert torch.equal(draft[3][2], ckpt.whole("mtp.0.ffn.experts.5.w1")[2])
    with pytest.raises(ValueError, match="cover"):
        F.expert_set(str(checkpoint), "model.layers.0.ffn.experts", expert_map[:4], meta, CPU)

    g = torch.Generator().manual_seed(3)
    hidden, inter = fixture.TEXT["hidden_size"], fixture.TEXT["moe_intermediate_size"]
    x = torch.randn(4, hidden, generator=g).to(BF16)
    ids = torch.tensor([[3, 7, -1, 5, 0, 9], [7, 1, 2, 4, 6, 8],
                        [10, 11, 12, 13, 14, 15], [3, 3, 7, 16, -1, 2]], dtype=torch.int32)
    tw = torch.rand(4, 6, generator=g)
    dense = [
        [effective_weight(*(p.numpy() for p in weights[3 * e + j]), int(m[2]), int(m[3]))
         for j, m in enumerate(meta[e].tolist())]
        for e in range(2)
    ]
    plain = F.moe_exact(x, ids, tw, expert_map, weights, meta, inter, 0.0)
    gate_values = np.concatenate(
        [x.double().numpy() @ dense[e][0][:, :inter] for e in range(2)]
    )
    limit = float(np.median(np.abs(gate_values)))
    clamped = F.moe_exact(x, ids, tw, expert_map, weights, meta, inter, limit)
    assert not torch.equal(plain, clamped)
    x64 = x.double().numpy()
    for t in range(x.shape[0]):
        want = np.zeros(hidden)
        for gid, w in zip(ids[t].tolist(), tw[t].double().tolist()):
            if not 0 <= gid < len(expert_map) or expert_map[gid] < 0:
                continue
            gate, up, down = dense[int(expert_map[gid])]
            a = np.minimum(x64[t] @ gate[:, :inter], limit)
            b = np.clip(x64[t] @ up[:, :inter], -limit, limit)
            want += w * ((a / (1.0 + np.exp(-a)) * b) @ down[:inter, :hidden])
        np.testing.assert_allclose(clamped[t].numpy(), want, rtol=1e-11, atol=1e-13)


def _recorded(ckpt, stem, local, experts):
    """The expert map and packed meta a layer with these local experts records."""
    expert_map = torch.full((experts,), -1, dtype=torch.int32)
    for i, e in enumerate(local):
        expert_map[e] = i
    names = [[f"{stem}.{e}.{w}" for w in ("w1", "w3", "w2")] for e in local]
    meta = torch.tensor(
        [[[m.padded_in, m.padded_out, m.bits_x2, m.codebook]
          for m in map(ckpt.matrix, row)] for row in names]
    )
    return expert_map, meta


def test_moe_experts_from_a_hybrid_checkpoint(tmp_path):
    """A hybrid checkpoint (EXL3 routed experts, FP8 for the other modules,
    the index naming each tensor's file): the experts are read past the FP8
    tensors, as the engine's manifest reads them."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from tests.kernels.quantization import exl3_real_cases as real

    config = fixture.tiny_config()
    config["quantization_config"].update(
        module_quantization={"rules": [
            {"match": r"^(layers|mtp)\.\d+\.ffn\.experts(\.|$)", "method": "exl3"}]},
        non_routed_quantization={"quant_method": "fp8"},
    )
    fixture.write_checkpoint(tmp_path, config)
    save_file({
        "layers.0.attn.wkv.weight": torch.zeros(32, 32, dtype=torch.float8_e4m3fn),
        "layers.0.attn.wkv.scale": torch.full((1, 1), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu),
    }, str(tmp_path / "fp8.safetensors"))
    # As the hybrid build: the FP8 file replaces the EXL3 wkv, whose tensors
    # stay in their shard but are not in the index.
    weight_map = {}
    for path in sorted(tmp_path.glob("*.safetensors")):
        with safe_open(str(path), "pt") as f:
            weight_map.update({
                name: path.name for name in f.keys()
                if path.name == "fp8.safetensors" or not name.startswith("layers.0.attn.wkv.")
            })
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    ckpt = real.Checkpoint(tmp_path)
    expert_map, meta = _recorded(
        ckpt, "layers.0.ffn.experts", (3, 7), fixture.TEXT["n_routed_experts"]
    )
    weights, _, got = F.expert_set(
        str(tmp_path), "model.layers.0.ffn.experts", expert_map, meta, CPU
    )
    assert torch.equal(got, meta)
    assert torch.equal(weights[3][0], ckpt.whole("layers.0.ffn.experts.7.w1")[0])



@pytest.fixture(scope="module")
def mixed_checkpoint(tmp_path_factory):
    """The tiny checkpoint with layer 1's routed experts at bits_x2 4 (the
    others at 6), as the real one mixes the two widths by layer."""
    directory = tmp_path_factory.mktemp("mixed")
    tensors = fixture.checkpoint_tensors
    width = re.compile(r"layers\.1\.ffn\.experts\.\d+\.w\d\.trellis")

    def mixed(config):
        out = tensors(config)
        for name, (dtype, shape) in out.items():
            if width.fullmatch(name):
                out[name] = (dtype, (shape[0], shape[1], 8 * 4))
        return out

    fixture.checkpoint_tensors = mixed
    try:
        fixture.write_checkpoint(directory, fixture.tiny_config())
    finally:
        fixture.checkpoint_tensors = tensors
    return directory


def _cpu_operators(monkeypatch):
    """CPU stand-ins for the EXL3 MoE operators (CUDA only): the routed
    experts of the weights the driver loaded, in FP64 rounded to BF16. They
    record the bits_x2 each moe_out_v2 and moe_out_prefill_v2 call
    received."""
    from vllm_sm80.kernels.exl3 import ops

    loaded, seen = {}, []
    expert_set = F.expert_set

    def remember(*args):
        weights, pointers, meta = expert_set(*args)
        loaded[pointers.data_ptr()] = weights
        return weights, pointers, meta

    def operator(name):
        def run(x, ids, tw, ptrs, meta, emap, inter, out, workspace,
                swiglu_limit=0.0, *, bits_x2=None):
            assert (bits_x2 is not None) == name.endswith("_v2"), name
            if bits_x2 is not None:
                assert {int(b) for b in meta[..., 2].flatten()} == {bits_x2}
                seen.append(bits_x2)
            out.copy_(F.moe_exact(x, ids, tw, emap, loaded[ptrs.data_ptr()], meta,
                                  inter, swiglu_limit).to(out.dtype))
        return run

    monkeypatch.setattr(F, "expert_set", remember)
    for name in ("moe_out", "moe_out_v2", "moe_out_prefill", "moe_out_prefill_v2"):
        monkeypatch.setattr(ops, name, operator(name))
    for size in ("moe_workspace_size", "moe_workspace_size_v2",
                 "moe_workspace_size_prefill", "moe_workspace_size_prefill_v2"):
        monkeypatch.setattr(ops, size, lambda *args: 1)
    return seen


def test_moe_families_run_at_both_bit_widths(tmp_path, mixed_checkpoint, monkeypatch):
    """exl3_moe_out_v2 and exl3_moe_out_prefill_v2 cases of a bits_x2 6 and a
    bits_x2 4 layer, and an exl3_moe_out case, run through fp64_cases.run end
    to end (operators replaced by CPU stand-ins): one line each, the v2
    operators given each layer's width; a recorded width that is not the
    layer's refuses its case."""
    from tests.kernels.quantization import exl3_real_cases as real

    seen = _cpu_operators(monkeypatch)
    ckpt = real.Checkpoint(mixed_checkpoint)
    text = fixture.TEXT
    g = torch.Generator().manual_seed(5)

    def source(name, family, layer, bits=None):
        expert_map, meta = _recorded(
            ckpt, f"layers.{layer}.ffn.experts", range(4), text["n_routed_experts"]
        )
        args = dict(
            x=torch.randn(3, text["hidden_size"], generator=g).to(BF16),
            topk_ids=torch.randint(
                0, text["n_routed_experts"], (3, text["num_experts_per_tok"]), generator=g
            ).to(torch.int32),
            topk_weights=torch.rand(3, text["num_experts_per_tok"], generator=g),
            packed_meta=meta, expert_map=expert_map,
            intermediate_size=text["moe_intermediate_size"], swiglu_limit=10.0,
        )
        if bits is not None:
            args["bits_x2"] = bits
        path = _source(tmp_path, name, family, args)
        obj = torch.load(path, weights_only=True)
        obj["meta"]["module_prefix"] = f"model.layers.{layer}.ffn.experts"
        torch.save(obj, path)
        return {"id": name, "family": family, "phase": "decode", "concurrency": 3,
                "calls": 1, "captured_sources": [path]}

    cases = [
        source("v2_bits6", "exl3_moe_out_v2", 0, bits=6),
        source("v2_bits4", "exl3_moe_out_v2", 1, bits=4),
        source("pv2_bits6", "exl3_moe_out_prefill_v2", 0, bits=6),
        source("pv2_bits4", "exl3_moe_out_prefill_v2", 1, bits=4),
        source("v1_bits6", "exl3_moe_out", 0),
    ]
    family = {
        "v2": "exl3_moe_out_v2",
        "pv2": "exl3_moe_out_prefill_v2",
        "v1": "exl3_moe_out",
    }
    rc, _, rows = _run(tmp_path, cases, target=mixed_checkpoint)
    assert rc == 0 and set(rows) == {c["id"] for c in cases}
    for cid, row in rows.items():
        _well_formed(row)
        assert row["family"] == family[cid.split("_")[0]]
        assert row["bitwise"] and row["row_invariance"], cid
        assert row["fp64"]["cand_max"] == row["fp64"]["ref_max"] > 0, cid
    # Per v2 case the whole call, then three rows alone; per prefill v2 case
    # (3 rows, fewer than it takes alone) the whole call only.
    assert sorted(set(seen)) == [4, 6] and seen.count(6) == seen.count(4) == 5

    for name, fam in (("v2_wrong_bits", "exl3_moe_out_v2"),
                      ("pv2_wrong_bits", "exl3_moe_out_prefill_v2")):
        wrong = source(name, fam, 1, bits=6)
        rc, _, rows = _run(tmp_path, [wrong], target=mixed_checkpoint)
        assert rc == 1 and rows == {}, name


# ---------------------------------------------------------------- EXL3 MoE split over TP ranks

# The served routed experts' 2304 intermediate columns in 4 chunks of 5, 5,
# 4 and 4 H128 blocks.
CHUNKS = [(0, 640), (640, 1280), (1280, 1792), (1792, 2304)]
DSV41_CHECKPOINT_ENV = "VLLM_DSV41_TEST_EXL3_CHECKPOINT"
MOE_CAPTURE_ENV = "VLLM_DSV41_TEST_MOE_CAPTURE"
MOE_PREFILL_CAPTURE_ENV = "VLLM_DSV41_TEST_MOE_PREFILL_CAPTURE"


def test_split_columns_rotate_whole_blocks_by_layer():
    """Rank r of layer L holds chunk (r + L) mod 4, so every rank is 640
    wide in 20 of the 40 target layers; the drafter's layers continue the
    count; one rank holds every column."""
    heavy = [0] * 4
    for layer in range(43):
        prefix = f"language_model.model.layers.{layer}.ffn.experts"
        got = [F.expert_columns(2304, prefix, rank, 4) for rank in range(4)]
        assert [(s.start, s.stop) for s in got] == [CHUNKS[(r + layer) % 4] for r in range(4)]
        if layer < 40:
            for rank, s in enumerate(got):
                heavy[rank] += s.stop - s.start == 640
    assert heavy == [20] * 4
    assert F.expert_columns(2304, "model.layers.41.ffn.experts", 0, 4) == slice(640, 1280)
    assert F.expert_columns(2304, "model.layers.7.ffn.experts", 0, 1) == slice(0, 2304)


def _moe_call(ckpt, layer, local, g, rows=3):
    """The arguments of a whole-expert call of the tiny checkpoint's layer
    ``layer`` holding the experts ``local``."""
    text = fixture.TEXT
    expert_map, meta = _recorded(
        ckpt, f"layers.{layer}.ffn.experts", local, text["n_routed_experts"]
    )
    k = text["num_experts_per_tok"]
    return dict(
        x=torch.randn(rows, text["hidden_size"], generator=g).to(BF16),
        topk_ids=torch.randint(0, text["n_routed_experts"], (rows, k), generator=g).to(
            torch.int32
        ),
        topk_weights=torch.rand(rows, k, generator=g),
        packed_meta=meta, expert_map=expert_map,
        intermediate_size=text["moe_intermediate_size"], swiglu_limit=10.0,
    )


def _reassembled(pieces, down):
    """The matrix (trellis, suh, svh) of the ranks' pieces [(first column,
    piece)] in column order: down along its inputs, gate/up its outputs."""
    ordered = [p for _, p in sorted(pieces, key=lambda item: item[0])]
    trellis = torch.cat([p[0] for p in ordered], dim=0 if down else 1)
    if down:
        return trellis, torch.cat([p[1] for p in ordered]), ordered[0][2]
    return trellis, ordered[0][1], torch.cat([p[2] for p in ordered])


def _rank_piece_is_the_checkpoint_columns(piece, whole, cols, down):
    (t, s, v), (wt, ws, wv) = piece, whole
    if down:
        return (torch.equal(t, wt[cols.start // 16 : cols.stop // 16])
                and torch.equal(s, ws[cols]) and torch.equal(v, wv))
    return (torch.equal(t, wt[:, cols.start // 16 : cols.stop // 16])
            and torch.equal(s, ws) and torch.equal(v, wv[cols]))


def _sums_to(partials, whole):
    scale = float(whole.abs().max())
    assert scale > 0
    total = torch.stack(partials).sum(0)
    return float((total - whole).abs().max()) <= F.TP_SUM_RTOL * scale


def test_a_split_expert_set_holds_the_rank_columns_and_the_partials_sum(checkpoint):
    """Each rank's experts of a whole-expert call replayed as a split
    (tp_call): the identity map over every expert, the rank's columns of
    each matrix (the checkpoint's bytes there), the four ranks' pieces
    reassembling the whole matrix bit for bit, and their FP64 partial
    outputs summing to the whole experts' FP64 output."""
    from tests.kernels.quantization import exl3_real_cases as real

    ckpt = real.Checkpoint(checkpoint)
    g = torch.Generator().manual_seed(13)
    layer, prefix = 3, "model.layers.3.ffn.experts"
    inter, experts = fixture.TEXT["moe_intermediate_size"], fixture.TEXT["n_routed_experts"]
    a = _moe_call(ckpt, layer, (2, 9, 11), g, rows=5)
    a["topk_ids"][0, :3] = torch.tensor([9, 4, -1], dtype=torch.int32)
    stems = [f"layers.{layer}.ffn.experts.9.{w}" for w in ("w1", "w3", "w2")]
    whole = [ckpt.whole(stem) for stem in stems]
    pieces, partials = [[], [], []], []
    for rank in range(4):
        cols = F.expert_columns(inter, prefix, rank, 4)
        width = cols.stop - cols.start
        t = F.tp_call(a, prefix, rank, 4)
        assert torch.equal(t["expert_map"], torch.arange(experts, dtype=torch.int32))
        assert t["intermediate_size"] == width and t["packed_meta"].shape == (experts, 3, 4)
        assert all(t[k] is a[k] for k in ("x", "topk_ids", "topk_weights", "swiglu_limit"))
        weights, _, got = F.expert_set(
            str(checkpoint), prefix, t["expert_map"], t["packed_meta"], CPU, rank, 4
        )
        assert torch.equal(got, t["packed_meta"])
        for j, piece in enumerate(weights[3 * 9 : 3 * 9 + 3]):
            assert _rank_piece_is_the_checkpoint_columns(piece, whole[j], cols, j == 2)
            pieces[j].append((cols.start, piece))
        partials.append(F.moe_exact(a["x"], a["topk_ids"], a["topk_weights"],
                                    t["expert_map"], weights, got, width, 10.0))
    for j, w in enumerate(whole):
        assert all(torch.equal(p, q) for p, q in zip(_reassembled(pieces[j], j == 2), w))

    exact = F.whole_fp64(str(checkpoint), prefix, a, CPU)
    one = F.tp_call(a, prefix, 0, 1)
    weights, _, got = F.expert_set(str(checkpoint), prefix, one["expert_map"],
                                   one["packed_meta"], CPU)
    assert one["intermediate_size"] == inter and set(got[:, 0, 1].tolist()) == {inter}
    assert torch.equal(
        F.moe_exact(a["x"], a["topk_ids"], a["topk_weights"], one["expert_map"], weights,
                    got, inter, 10.0),
        exact,
    )
    assert _sums_to(partials, exact)
    assert not _sums_to(partials[:3], exact)

    with pytest.raises(ValueError, match="tp_rank"):
        F.expert_set(str(checkpoint), prefix, t["expert_map"], t["packed_meta"], CPU)


def test_split_cases_run_end_to_end_and_partials_that_do_not_sum_refuse(
    tmp_path, mixed_checkpoint, cases, monkeypatch
):
    """Through fp64_cases.run (operators replaced by CPU stand-ins): a case
    with expert_tp 4 replays its whole-expert source on every rank (four
    ranks' outputs, each the whole call and three rows alone), and a call
    captured on a split (meta.tp_rank, meta.tp_size) runs on its rank's
    columns. Ranks that do not tile the columns (each holding the same
    chunk) give partials that do not sum to the whole experts' output: the
    case is refused. expert_tp on a family that is not an EXL3 MoE refuses
    its case."""
    from tests.kernels.quantization import exl3_real_cases as real

    seen = _cpu_operators(monkeypatch)
    ckpt = real.Checkpoint(mixed_checkpoint)
    g = torch.Generator().manual_seed(17)
    hidden = fixture.TEXT["hidden_size"]

    def source(name, family, layer, args, **meta):
        path = _source(tmp_path, name, family, args)
        obj = torch.load(path, weights_only=True)
        obj["meta"].update(module_prefix=f"model.layers.{layer}.ffn.experts", **meta)
        torch.save(obj, path)
        return {"id": name, "family": family, "phase": "decode", "concurrency": 3,
                "captured_sources": [path]}

    ep = dict(_moe_call(ckpt, 1, range(4), g), bits_x2=4)
    derived = dict(source("derived", "exl3_moe_out_v2", 1, ep), expert_tp=4)
    split = F.tp_call(_moe_call(ckpt, 2, range(4), g), "model.layers.2.ffn.experts", 2, 4)
    captured = source("captured_split", "exl3_moe_out", 2, split, tp_rank=2, tp_size=4)
    rc, _, rows = _run(tmp_path, [derived, captured], target=mixed_checkpoint)
    assert rc == 0 and set(rows) == {"derived", "captured_split"}
    for cid, row in rows.items():
        _well_formed(row)
        assert row["bitwise"] and row["row_invariance"], cid
        assert row["fp64"]["cand_max"] == row["fp64"]["ref_max"] > 0, cid
    assert rows["derived"]["numel"] == 4 * 3 * hidden
    assert rows["captured_split"]["numel"] == 3 * hidden
    assert seen == [4] * 16

    columns = F.expert_columns
    monkeypatch.setattr(F, "expert_columns",
                        lambda inter, prefix, rank, size: columns(inter, prefix, 0, size))
    rc, _, rows = _run(tmp_path, [derived], target=mixed_checkpoint)
    assert rc == 1 and rows == {}
    monkeypatch.setattr(F, "expert_columns", columns)

    engram = dict(next(c for c in cases if c["id"] == "engram_dequant"), id="engram_tp",
                  expert_tp=4)
    rc, _, rows = _run(tmp_path, [engram], target=mixed_checkpoint)
    assert rc == 1 and rows == {}

    # A call of split experts is not replayed as the whole experts of a split.
    again = dict(captured, id="split_again", expert_tp=4)
    rc, _, rows = _run(tmp_path, [again], target=mixed_checkpoint)
    assert rc == 1 and rows == {}


def test_a_split_expert_of_the_served_checkpoint():
    """One routed expert of the served checkpoint at layer 3 on each of 4
    ranks: the rank holds chunk (r + 3) mod 4 of its intermediate (the
    checkpoint's bytes there), the pieces reassemble the expert bit for bit,
    and the four FP64 partial outputs of two rows routed to it sum to the
    whole expert's."""
    from tests.kernels.quantization import exl3_real_cases as real

    directory = str(real.directory_or_skip(
        DSV41_CHECKPOINT_ENV, "the served DeepSeek-V4.1 EXL3 checkpoint"))
    layout, resolver, ckpt = F._expert_layout(directory)
    layer, expert = 3, 5
    prefix = f"language_model.model.layers.{layer}.ffn.experts"
    names = [layout.expert_matrix(resolver, prefix, expert, role) for role in F.ROLES]
    matrices = [ckpt.matrix(name) for name in names]
    hidden, inter = matrices[0].padded_in, matrices[0].padded_out
    assert inter == CHUNKS[-1][1]
    config = json.loads((Path(directory) / "config.json").read_text())
    experts = config.get("text_config", config)["n_routed_experts"]
    emap = torch.full((experts,), -1, dtype=torch.int32)
    emap[expert] = 0
    whole = [ckpt.whole(name) for name in names]
    g = torch.Generator().manual_seed(23)
    x = torch.randn(2, hidden, generator=g).to(BF16)
    ids = torch.tensor([[expert], [expert]], dtype=torch.int32)
    tw = torch.tensor([[1.0], [0.5]])
    pieces, partials = [[], [], []], []
    for rank in range(4):
        start, stop = CHUNKS[(rank + layer) % 4]
        width = stop - start
        kinds = [[m.bits_x2, m.codebook] for m in matrices]
        meta = torch.tensor([[[hidden, width, *kinds[0]], [hidden, width, *kinds[1]],
                              [width, hidden, *kinds[2]]]])
        weights, _, got = F.expert_set(directory, prefix, emap, meta, CPU, rank, 4)
        for j, piece in enumerate(weights):
            assert _rank_piece_is_the_checkpoint_columns(
                piece, whole[j], slice(start, stop), j == 2)
            pieces[j].append((start, piece))
        partials.append(F.moe_exact(x, ids, tw, emap, weights, got, width, 10.0))
    for j, w in enumerate(whole):
        assert all(torch.equal(p, q) for p, q in zip(_reassembled(pieces[j], j == 2), w))
    a = dict(x=x, topk_ids=ids, topk_weights=tw, expert_map=emap,
             intermediate_size=inter, swiglu_limit=10.0)
    assert _sums_to(partials, F.whole_fp64(directory, prefix, a, CPU))


def _one_call_capture(capture, root, point, layer, rows=None):
    """A capture at ``root`` of 4 ranks holding one saved call of ``point``
    at ``layer`` (the fewest rows, or ``rows``) of ``capture``: each rank's
    file linked, rank 0's index line. (path of rank 0's file, its index
    line, its name under r<rank>)."""
    records = [json.loads(line) for line in
               (capture / "r0" / "index.jsonl").read_text().splitlines() if line]
    rec = min((r for r in records if r.get("saved") and r["point"] == point
               and r["prefix"].endswith(f"layers.{layer}.ffn.experts")
               and rows in (None, r["rows"])), key=lambda r: r["rows"])
    name = Path(rec["path"]).relative_to(capture / "r0")
    for rank in range(4):
        (root / f"r{rank}" / name).parent.mkdir(parents=True)
        (root / f"r{rank}" / name).symlink_to(capture / f"r{rank}" / name)
    path = str(root / "r0" / name)
    (root / "r0" / "index.jsonl").write_text(json.dumps(dict(rec, path=path)) + "\n")
    return path, rec, name


def _a_split_case_of(case, path, rec, capture, name, directory, layer):
    """The exported case of the one call: expert_tp 4, pinned to rank 0's
    file; each rank's derived arguments hold the same rows, the identity
    map and the rank's chunk of the layer, which the checkpoint's experts
    match."""
    obj = torch.load(path, weights_only=True)
    a, meta = obj["tensors"], obj["meta"]
    assert (case["family"], case["expert_tp"], case["scheduler_phase"], case["rows"]) == (
        rec["point"], 4, rec["phase"], rec["rows"])
    assert case["phase"] == ("prefill" if rec["phase"] == "prefill" else "decode")
    assert case["concurrency"] == len(meta["query_lens"])
    assert case["captured_sources"] == [path]
    assert case["captured_sha256"] == {path: "sha256:" + rec["sha256"]}
    experts = a["expert_map"].numel()
    assert int((a["expert_map"] >= 0).sum()) == a["packed_meta"].shape[0] < experts
    routed = sorted({int(e) for e in a["topk_ids"].flatten()})
    for rank in range(4):
        t = F.tp_call(a, meta["module_prefix"], rank, 4)
        start, stop = CHUNKS[(rank + layer) % 4]
        assert t["intermediate_size"] == stop - start
        assert torch.equal(t["expert_map"], torch.arange(experts, dtype=torch.int32))
        assert t["packed_meta"].shape == (experts, 3, 4)
        assert set(t["packed_meta"][:, :2, 1].flatten().tolist()) == {stop - start}
        other = torch.load(capture / f"r{rank}" / name, weights_only=True)["tensors"]
        assert all(F.bits_equal(other[k], t[k]) for k in ("x", "topk_ids", "topk_weights"))
        g = routed[rank % len(routed)]
        emap = torch.full((experts,), -1, dtype=torch.int32)
        emap[g] = 0
        F.expert_set(str(directory), meta["module_prefix"], emap,
                     t["packed_meta"][g : g + 1], CPU, rank, 4)


def _routed_other_rows(root, name):
    """Rank 2's file of the call at ``root`` with other rows than rank 0's."""
    bad = torch.load(root / "r2" / name, weights_only=True)
    bad["tensors"]["x"] = bad["tensors"]["x"] + 1
    (root / "r2" / name).unlink()
    torch.save(bad, root / "r2" / name)


def test_split_cases_from_a_whole_expert_capture(tmp_path):
    """The export of split cases from one decode-step call of a whole-expert
    capture of 4 ranks: one case of that call; an export is never replaced;
    a rank whose file routed other rows refuses the export."""
    from tests.kernels.quantization import exl3_real_cases as real

    directory = real.directory_or_skip(
        DSV41_CHECKPOINT_ENV, "the served DeepSeek-V4.1 EXL3 checkpoint")
    capture = real.directory_or_skip(
        MOE_CAPTURE_ENV, "a capture of the served whole experts on 4 ranks (r0..r3)")
    layer, root = 3, tmp_path / "capture"
    path, rec, name = _one_call_capture(capture, root, "exl3_moe_out_v2", layer)
    out = tmp_path / "cases.json"
    argv = ["--export-expert-tp", str(root), "--checkpoint", str(directory), "--out", str(out)]
    assert F.main(argv) == 0
    assert F.main(argv) == 2  # an export is never replaced
    doc = json.loads(out.read_text())
    assert doc["schema"] == "served_cases/1" and doc["checkpoint"] == {"target": str(directory)}
    (case,) = doc["cases"]
    _a_split_case_of(case, path, rec, capture, name, directory, layer)

    _routed_other_rows(root, name)
    with pytest.raises(ValueError, match="rank 2"):
        F.expert_tp_cases([root], directory)


def test_split_prefill_cases_join_the_decode_ones(tmp_path):
    """One export of two captures: a decode-step call of one and a prefill
    (exl3_moe_out_prefill_v2) call of the other give one cases file with a
    case of each, the prefill case a prefill case of its rows; a rank of the
    prefill capture that routed other rows refuses the export."""
    from tests.kernels.quantization import exl3_real_cases as real

    directory = real.directory_or_skip(
        DSV41_CHECKPOINT_ENV, "the served DeepSeek-V4.1 EXL3 checkpoint")
    decode = real.directory_or_skip(
        MOE_CAPTURE_ENV, "a capture of the served whole experts on 4 ranks (r0..r3)")
    prefill = real.directory_or_skip(
        MOE_PREFILL_CAPTURE_ENV,
        "a capture of the served whole experts' prefill calls on 4 ranks (r0..r3)")
    calls = {}
    for point, capture, layer in (("exl3_moe_out_v2", decode, 3),
                                  ("exl3_moe_out_prefill_v2", prefill, 11)):
        root = tmp_path / point
        calls[point] = (*_one_call_capture(capture, root, point, layer), capture, layer, root)
    out = tmp_path / "cases.json"
    roots = [str(c[-1]) for c in calls.values()]
    assert F.main(["--export-expert-tp", *roots, "--checkpoint", str(directory),
                   "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["source"] == roots
    by_family = {c["family"]: c for c in doc["cases"]}
    assert len(doc["cases"]) == 2 and set(by_family) == set(calls)
    for point, (path, rec, name, capture, layer, _) in calls.items():
        _a_split_case_of(by_family[point], path, rec, capture, name, directory, layer)
    assert by_family["exl3_moe_out_prefill_v2"]["phase"] == "prefill"
    assert by_family["exl3_moe_out_prefill_v2"]["rows"] >= 65

    path, rec, name, capture, layer, root = calls["exl3_moe_out_prefill_v2"]
    _routed_other_rows(root, name)
    with pytest.raises(ValueError, match="rank 2"):
        F.expert_tp_cases([Path(r) for r in roots], directory)


# ---------------------------------------------------------------- hybrid CPU contracts


def test_hybrid_independent_fp8_decode():
    from .hybrid_linear_cases import decode_fp8

    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(8, 32)
    bits[(bits & 127) == 127] = 0
    scale = torch.full((8, 1), 127, dtype=torch.uint8)
    got = decode_fp8(bits.view(torch.float8_e4m3fn), scale)
    want = bits.view(torch.float8_e4m3fn).double()
    assert torch.equal(got, want)
    assert torch.equal(got.signbit(), want.signbit())
    scale[0] = 255
    with pytest.raises(ValueError, match="nonfinite official UE8M0"):
        decode_fp8(bits.view(torch.float8_e4m3fn), scale)


def test_hybrid_capture_link_does_not_retain_dead_outputs():
    import gc
    from torch.multiprocessing.reductions import StorageWeakRef
    from vllm_sm80.models.deepseek_v41 import hybrid_capture as h

    x = torch.ones(2, 32, dtype=BF16)
    key = x.untyped_storage().data_ptr()
    h._outputs[key] = ({"dtype": str(BF16), "storage_offset": 0,
                        "call_id": 7, "output": "out", "prefix": "projection"}, StorageWeakRef(x.untyped_storage()))
    assert h._link(x)["call_id"] == 7
    del x
    gc.collect()
    assert h._outputs[key][1].expired()
    h._outputs.clear()


def test_hybrid_consumer_requires_real_link():
    from .hybrid_consumer_cases import evaluate

    obj = {"meta": {"call_id": 1, "op": "rms_norm", "outputs": ["out"]},
           "tensors": {"x": torch.ones(2, 32, dtype=BF16),
                       "weight": torch.ones(32, dtype=BF16), "eps": 1e-6,
                       "out": torch.ones(2, 32, dtype=BF16)}}
    with pytest.raises(ValueError, match="missing actual-consumer link"):
        evaluate(obj, CPU, "unused")


def test_hybrid_reference_disables_weak_reductions():
    from .hybrid_linear_cases import reference_settings

    settings = reference_settings()
    assert settings["allow_tf32"] is False
    assert settings["allow_bf16_reduced_precision_reduction"] is False
    assert settings["allow_fp16_reduced_precision_reduction"] is False
    assert settings["float32_matmul_precision"] == "highest"


def test_hybrid_consumer_defers_tp_collectives():
    from .hybrid_consumer_cases import evaluate

    obj = {"meta": {"call_id": 1, "tp_boundary": True}, "tensors": {}}
    with pytest.raises(ValueError, match="distributed consumer check"):
        evaluate(obj, torch.device("cpu"), "unused")


def test_hybrid_state_export_preserves_aliases_and_write_guards():
    import importlib.util
    from pathlib import Path
    from .hybrid_consumer_cases import _owned_state

    path = Path(__file__).resolve().parents[4] / "tools/sm80/export_hybrid_cases.py"
    spec = importlib.util.spec_from_file_location("hybrid_case_export", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    cache = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    after = cache.clone()
    after[1, 1] = -1
    blocks = torch.tensor([0, 1, 2, 1])
    args = {"cache": cache[blocks], "out": after[blocks],
            "snapshot_name": "cache", "snapshot_blocks": blocks,
            "slots": torch.tensor([3]), "rows": after[1, 1:2]}
    frozen = exporter.compact_state({"meta": {}, "tensors": args})
    assert torch.equal(frozen["tensors"]["cache"], cache)
    assert torch.equal(frozen["tensors"]["out"], after)
    assert frozen["meta"]["observed_blocks"] == [0, 1, 2]
    _owned_state("store_rows", frozen["tensors"], {"out": after})
    wrong = after.clone()
    wrong[2, 0] = -2
    with pytest.raises(ValueError, match="outside owned slots"):
        _owned_state("store_rows", frozen["tensors"], {"out": wrong})
    args["out"][-1, 0] = -3
    with pytest.raises(ValueError, match="aliased state blocks disagree"):
        exporter.compact_state({"meta": {}, "tensors": args})


def test_hybrid_explicit_draft_phase_during_prefill(monkeypatch, tmp_path):
    import json
    import re
    from vllm_sm80 import capture as c

    cfg = c._Config(str(tmp_path), frozenset({"hybrid"}), re.compile(".*"),
                    re.compile(".*"), frozenset({1}), 2, frozenset({1}), 65,
                    1 << 20, 0, 0, 1, prefill_per_key=0)
    monkeypatch.setattr(c, "_cfg", cfg)
    monkeypatch.setattr(c, "_step", {"real": True, "step": 1, "phase": "prefill",
                                   "query_lens": [128], "context_lens": [0], "context": "short"})
    monkeypatch.setattr(c, "_counts", {})
    monkeypatch.setattr(c, "_prefill_calls", {})
    monkeypatch.setattr(c, "_bytes", [0])
    monkeypatch.setattr(c, "_capturing", lambda: False)
    monkeypatch.setattr(c.torch.cuda, "is_available", lambda: False)
    assert c.begin("hybrid", "hybrid_linear", "head", 1) is None
    with c.phase_scope("draft"):
        rec = c.begin("hybrid", "hybrid_linear", "head", 1)
        assert rec is not None and rec.point_phase == "draft"
        rec.put(x=torch.ones(1, 32)).done()
    row = json.loads((tmp_path / "index.jsonl").read_text())
    assert row["phase"] == "draft" and row["step_phase"] == "prefill"
    assert "-draft-eager-" in row["path"]
    assert c._point_phase.get() is None
    with pytest.raises(ValueError, match="unknown capture point phase"):
        c.begin("hybrid", "hybrid_linear", "head", 1, point_phase="invented")


def test_hybrid_fp32_router_is_not_a_changed_projection(monkeypatch):
    import re
    from types import SimpleNamespace
    from vllm_sm80.models.deepseek_v41 import hybrid_capture as h

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(32, 32, dtype=torch.float32), requires_grad=False)
    model._hybrid_capture_sources = [("layers.0.ffn.gate", [32, 32])]
    model.prefix = "layers.0.ffn.gate"
    monkeypatch.setattr(h.capture, "config", lambda: SimpleNamespace(
        match=re.compile(".*"), prefill_match=re.compile(".*")))
    h.register_model(model)
    assert not hasattr(model, "_hybrid_capture")


# ---------------------------------------------------------------- GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_paths_on_the_synthetic_cases(tmp_path, cases):
    rc, head, rows = _run(tmp_path, cases, torch.device("cuda"))
    assert rc == 0 and head["sm_count"] > 0
    assert set(rows) == {c["id"] for c in cases}
    for cid, row in rows.items():
        _well_formed(row)
        assert row["row_invariance"], cid
        e = row["fp64"]
        if cid in EXACT or cid == "index_scores":
            assert row["bitwise"], cid
            assert e["cand_mean"] == e["ref_mean"] and e["cand_max"] == e["ref_max"], cid
        if cid in EXACT[:3]:
            assert e["cand_max"] == 0.0, cid
        if cid in FUSED_ATTN:
            # V2: the fused core against the served path, both against FP64
            if e["ref_max"] == 0.0:
                assert e["cand_max"] == 0.0, cid
            else:
                assert e["cand_mean"] <= 1.10 * e["ref_mean"], (cid, e)
                assert e["cand_max"] <= 1.25 * e["ref_max"], (cid, e)


def test_call_order_keeps_a_call_together_and_alternates_strata():
    def case(i, sources, phase, rank):
        return {"id": f"c{i}", "family": "attn_pipeline", "phase": phase, "context": "short",
                "drafter": False, "rank": rank, "captured_sources": sources}
    cases = [case(0, ["a"], "verify", 0), case(1, ["b", "c"], "verify", 0), case(2, ["d"], "decode", 1),
             case(3, ["a"], "verify", 0), case(4, ["b", "c"], "verify", 0), case(5, ["d"], "decode", 1)]
    assert [c["id"] for c in F.call_order(cases)] == ["c2", "c5", "c0", "c3", "c1", "c4"]
    # The pipeline export also holds index_select cases (capture r3: 180 of 3072).
    select = dict(case(6, ["e"], "verify", 0), family="index_select")
    order = ["c2", "c5", "c0", "c3", "c6", "c1", "c4"]
    assert [c["id"] for c in F.call_order(cases + [select])] == order
    assert [c["id"] for c in F.call_order([select] + cases)] == order


def test_the_outputs_of_one_call_replay_it_once(monkeypatch):
    from . import attn_pipeline_cases as P
    calls = []
    def replay(a, device, src):
        calls.append(id(a))
        if a.get("bad"):
            raise ValueError("nonfinite pipeline comp")
        return "outputs"
    monkeypatch.setattr(P, "_pipeline_outputs", replay)
    monkeypatch.setattr(P, "_REPLAYED", {})
    good, bad = {}, {"bad": True}
    for _ in range(3):
        assert P.pipeline_outputs(good, CPU, None) == "outputs"
        with pytest.raises(ValueError, match="nonfinite"):
            P.pipeline_outputs(bad, CPU, None)
    assert calls == [id(good), id(bad)]
    assert P.pipeline_outputs(dict(good), CPU, None) == "outputs"
    assert len(calls) == 3


def test_pipeline_caches_are_judged_on_the_slots_the_call_touches():
    """A served capture (pipeline-359, r3) gathered a write-guard comp page
    whose unwritten entries held NaN memory; the call neither reads nor
    writes them. Only touched slots are judged; a NaN there still fails."""
    from types import SimpleNamespace
    from . import attn_pipeline_cases as P
    e = 4
    a = {"comp": torch.zeros(4, e, 8),
         # two requests; page 1 holds request 0's entries 0-3, page 2 entry 4
         "batch.comp_block_table": torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
         "batch.positions": torch.tensor([3, 4, 1]),
         "batch.token_to_req": torch.tensor([0, 0, 1], dtype=torch.int32),
         "batch.swa_slots": torch.tensor([0, 1, 2]),
         "batch.comp_slots": torch.tensor([7, 8, 13])}
    cfg = SimpleNamespace(compress_ratio=1)
    mask = P.touched(a, "comp", 1)
    assert mask.nonzero().tolist() == [[1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [3, 0], [3, 1]]
    guard = a["comp"].clone()
    guard[0] = float("nan")
    guard[2, 1:] = float("nan")
    assert P.finite_where_read(a, cfg, "comp", guard)
    for page, entry in ((1, 2), (2, 0), (3, 1)):
        read = a["comp"].clone()
        read[page, entry, 3] = float("nan")
        assert not P.finite_where_read(a, cfg, "comp", read)
    assert not P.finite_where_read(a, cfg, "qr", torch.tensor([0.0, float("nan")]))
