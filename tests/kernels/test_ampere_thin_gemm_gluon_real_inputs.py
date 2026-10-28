# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Gluon thin GEMM (VLLM_THIN_GEMM_GLUON) on real captured inputs.

Inputs: VLLM_THIN_GEMM_TEST_PORT_CASES names the case set (layout in
thin_gemm_real_cases.py). With it unset or missing, the tests that need it
skip and say which. Every case is a call of an unquantized BF16 linear taken
from the served Qwen3.8-Flash-Next at tensor parallel size 2, rank 0, 74
multiprocessors: its input rows, the layer's weight and, for whole ("live")
calls, the output cuBLAS gave.

CPU: skip behaviour, case loading (sha256, fields, shapes) and coverage on a
synthetic case set with a tampered file as negative control; with the case set
given, every file's sha256, every case on the Gluon path (its (N, K) in the
table, at most 32 rows), and one case per required (shape, rows, phase) cell.

GPU (SM 8.0 with 74 multiprocessors): every case goes through the dispatch
(``wrap_dispatch`` after ``warmup_table`` opened the gate) captured into a CUDA
graph, which is where it runs the Gluon kernels, and eagerly, where it must give
F.linear's bits. F.linear must give each live case's recorded output bit for
bit (the inputs and weights are the served ones); per pool (shape, rows, phase,
drafter, source) the captured dispatch holds
cuBLAS's error against FP64 (pooled mean <= 1.10x, max <= 1.25x, BF16 rounding
ties forgiven, exact where cuBLAS is exact) and is bitwise replay to replay.
One line per pool and per phase's cells is printed; a failing pool fails.

Required cells (REQUIRED_CELLS: the row counts the thin-GEMM dispatch takes at
1-8 concurrent requests with 3-token speculative decoding or without drafts,
each padded to its CUDA-graph size 1, 2, 4, 8, 16, 24 or 32; phase from the
scheduler):
- decode: drafter rows 1-8 (hyper-connection, final mixer, router and
  shared-expert gate shapes) and verify rows 4-32 (all seven shapes);
- decode with the target model's weights at 1 and 2 rows (all seven shapes):
  a verify step whose drafts the scheduler dropped (structured-output
  requests whose drafts the grammar rejects, preemption) or a launch without
  speculative decoding;
- prefill: all seven shapes at every graph size up to 32;
- mixed: the drafter shapes at 2 and 4, all seven shapes at 8-32 (row count 4
  of a mixed step comes only from 3 real rows padded: "padded" cases).
"""

import json

import pytest
import torch

from tests.kernels import thin_gemm_real_cases as real

DRAFTER_SHAPES = ("final_down", "hc_down", "hc_up", "router", "shared_gate")
ALL_SHAPES = DRAFTER_SHAPES + ("gdn_ba", "ple_kv")
REQUIRED_CELLS = (
    {(s, r, "decode") for s in DRAFTER_SHAPES for r in (1, 2, 4, 8)}
    | {(s, r, "decode") for s in ALL_SHAPES for r in (4, 8, 16, 24, 32)}
    | {(s, r, "decode", "target") for s in ALL_SHAPES for r in (1, 2)}
    | {(s, r, "prefill") for s in ALL_SHAPES for r in (1, 2, 4, 8, 16, 24, 32)}
    | {(s, r, "mixed") for s in DRAFTER_SHAPES for r in (2, 4)}
    | {(s, r, "mixed") for s in ALL_SHAPES for r in (8, 16, 24, 32)}
)


def _gpu_ok():
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (8, 0)
            and torch.cuda.get_device_properties(0).multi_processor_count == 74)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU with 74 SMs")


# ------------------------------------------------------------------ CPU ----
def test_skips_without_the_case_set(monkeypatch):
    monkeypatch.delenv(real.CASES_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=real.CASES_ENV):
        real.case_dir_or_skip()


def _synthetic(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "w").mkdir()
    w = torch.randn(48, 2560).to(torch.bfloat16)
    torch.save({"w": w}, tmp_path / "w" / "a.pt")
    wsha = real.file_sha256(tmp_path / "w" / "a.pt")
    cases = []
    for i, (rows, phase, source) in enumerate(
            [(4, "decode", "live"), (8, "mixed", "padded")]):
        x = torch.randn(rows, 2560).to(torch.bfloat16)
        t = {"x": x}
        if source == "live":
            t["out"] = torch.nn.functional.linear(x, w)
        torch.save(t, tmp_path / "x" / f"c{i}.pt")
        cases.append(dict(
            id=f"c{i}", file=f"x/c{i}.pt", sha256=real.file_sha256(tmp_path / "x" / f"c{i}.pt"),
            weight_file="w/a.pt", weight_sha256=wsha, shape="gdn_ba",
            role="model.layers.*.linear_attn.in_proj_ba", N=48, K=2560, rows=rows, phase=phase,
            drafter=False, source=source, record_sha256="0" * 64,
            padded_from=None if source == "live" else 3))
    (tmp_path / "cases.json").write_text(json.dumps(
        {"schema": real.SCHEMA, "sm_count": 74, "tp_size": 2, "tp_rank": 0, "cases": cases}))
    return cases


def test_synthetic_case_set_loads_pools_and_reports(tmp_path, monkeypatch):
    _synthetic(tmp_path)
    monkeypatch.setenv(real.CASES_ENV, str(tmp_path))
    directory = real.case_dir_or_skip()
    header, cases = real.load_manifest(directory)
    assert header["sm_count"] == 74 and len(cases) == 2
    weights = {}
    x, w, out = real.load_case(directory, cases[0], weights)
    assert out is not None and len(weights) == 1
    x, w, out = real.load_case(directory, cases[1], weights)
    assert out is None and x.shape == (8, 2560)
    assert list(real.pools(cases)) == [("gdn_ba", 4, "decode", False, "live"),
                                       ("gdn_ba", 8, "mixed", False, "padded")]
    lines, missing = real.coverage_report(
        cases, {("gdn_ba", 4, "decode"), ("gdn_ba", 16, "prefill")})
    assert missing == [("gdn_ba", 16, "prefill")]
    assert lines[0] == "CELLS decode: 4:1"
    assert lines[1] == "CELLS prefill: 16:0(MISSING gdn_ba)"
    assert lines[2] == "CELLS mixed: +8:1"


def test_target_cells_need_the_target_weights(tmp_path):
    """A (shape, rows, phase, "target") cell is not met by a drafter case."""
    base = _synthetic(tmp_path)[0]
    drafter = dict(base, id="d", shape="hc_down", rows=1, drafter=True)
    target = dict(base, id="t", rows=2)
    required = {("hc_down", 1, "decode", "target"), ("gdn_ba", 2, "decode", "target")}
    lines, missing = real.coverage_report([drafter, target], required)
    assert missing == [("hc_down", 1, "decode", "target")]
    assert lines[:2] == ["CELLS decode: +1:1 +2:1",
                         "CELLS decode target: 1:0(MISSING hc_down) 2:1"]
    lines, missing = real.coverage_report([drafter], {("hc_down", 1, "decode")})
    assert missing == [] and lines[0] == "CELLS decode: 1:1"
    assert real.coverage_report([drafter, dict(drafter, id="d2", drafter=False)],
                                {("hc_down", 1, "decode", "target")})[1] == []


def test_tampered_case_fails(tmp_path):
    cases = _synthetic(tmp_path)
    path = tmp_path / cases[0]["file"]
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(AssertionError, match="file sha256"):
        real.load_case(tmp_path, cases[0])


def test_case_set_files_and_coverage():
    """Every file intact, every case on the Gluon path, every required cell."""
    from vllm_sm80.kernels.thin_gemm import gluon

    directory = real.case_dir_or_skip()
    header, cases = real.load_manifest(directory)
    assert (header["sm_count"], header["tp_size"], header["tp_rank"]) == (74, 2, 0)
    weights = {}
    for case in cases:
        real.load_case(directory, case, weights)
        assert (case["N"], case["K"]) in gluon.SHAPES, case["id"]
        assert case["rows"] <= gluon.MAX_M, case["id"]
    lines, missing = real.coverage_report(cases, REQUIRED_CELLS)
    print("\n".join(lines))
    assert not missing, missing


# ------------------------------------------------------------------ GPU ----
@gpu
def test_dispatch_holds_cublas_error_on_real_inputs(monkeypatch):
    import types

    from tests.kernels.quantization.test_exl3_dense_splitk import _Pool
    from tests.sm80.tuning_rows import use_rows
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm_sm80.kernels.thin_gemm import gluon

    directory = real.case_dir_or_skip()
    _, cases = real.load_manifest(directory)
    use_rows(monkeypatch)
    monkeypatch.setattr(gluon, "_LIVE", [False])
    gluon.impl.cache_clear()
    device = torch.device("cuda", torch.cuda.current_device())
    worker = types.SimpleNamespace(
        device=device,
        vllm_config=types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(
                mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.PIECEWISE),
            model_config=types.SimpleNamespace(enforce_eager=False)))
    assert gluon.warmup_table(worker, list(range(1, 33))) == gluon.SHAPES
    left, launched = [], []

    def shipped(layer, x, w, bias=None):
        # Recorded, not raised: an exception inside the capture would surface
        # as the capture's error instead of this one.
        left.append(tuple(x.shape))
        return torch.nn.functional.linear(x, w, bias)

    kernels = gluon.impl()
    monkeypatch.setattr(gluon, "impl", lambda: types.SimpleNamespace(
        thin_gemm=lambda x, w: launched.append(x.shape[0]) or kernels.thin_gemm(x, w)))
    dispatch = gluon.wrap_dispatch(shipped)
    stream = torch.cuda.Stream()

    def captured(x, w):
        """The dispatch captured into a CUDA graph: two replays' outputs."""
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = dispatch(None, x, w)
        graph.replay()
        first = out.clone()
        graph.replay()
        torch.cuda.synchronize()
        return first, out.clone()

    weights, live_bad, failed = {}, [], []
    for key, group in real.pools(cases).items():
        pool = _Pool()
        for case in group:
            x, w, out = real.load_case(directory, case, weights)
            x, w = x.cuda(), w.cuda()
            inc = torch.nn.functional.linear(x, w)
            if out is not None and not torch.equal(inc.view(torch.int16),
                                                    out.cuda().view(torch.int16)):
                live_bad.append(case["id"])
            n_launched = len(launched)
            eager = dispatch(None, x, w)
            assert torch.equal(eager.view(torch.int16), inc.view(torch.int16)), case["id"]
            assert len(launched) == n_launched, f"{case['id']}: an eager call launched the kernels"
            c1, c2 = captured(x, w)
            assert len(launched) == n_launched + 1, f"{case['id']}: the captured call missed the kernels"
            assert torch.equal(c1.view(torch.int16), c2.view(torch.int16)), case["id"]
            pool.add(c1, inc, x.double() @ w.double().T)
        name = "%s rows %d %s%s %s" % (key[0], key[1], key[2], " drafter" if key[3] else "", key[4])
        try:
            mean, worst, ties = pool.check(name)
            print(f"POOL {name}: {len(group)} cases, mean {mean:.3f}x max {worst:.3f}x ties {ties}")
        except AssertionError as exc:
            failed.append(str(exc))
            print(f"POOL {name}: FAIL {exc}")
    lines, _ = real.coverage_report(cases, REQUIRED_CELLS)
    print("\n".join(lines))
    assert not left, f"real cases left the Gluon path: {left[:4]}"
    assert not live_bad, f"F.linear replay differs from the recorded output: {live_bad[:4]}"
    assert not failed, failed[:4]
