# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled Marlin MoE prefill tiles (VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED and
VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED, on by default; 0 is the kill switch): the
split-list GEMMs of the TP4 (N=512) and PP whole-expert (N=2048) Marlin prefill
through the optional library's prefill_tile_gemm; at N=2048 with the optimistic
fast dequant and its per-GEMM redo areas.

CPU: the switch; ``compiled_tiles`` reasons (switch off, no table for the
width, library/op missing, no scratch during a capture); the per-GEMM tile
table reaches the op exactly (w13 (64,512,1) and w2 (64,256,2) on the
64/48/32-row lists, Marlin's own choice on the 16-row list); without the op the
released kernels run with THREAD_CFG; ``prefill_tile_op`` never raises; the
library sources declare the same 21 kernels in the selector and the
instantiation file, including the six explicit tiles; the in-tree library, when
present, exports prefill_tile_gemm.
GPU (sm_80 with ~12 GB free and the library; skipped otherwise): the compiled
tiles run (spied) and against an fp64 reference stay within 1.10x mean /
1.25x max of the incumbent's error at the TP4 call sizes and small M; bitwise
run to run; with the op missing the result is bitwise the released split path
(switch 0); CUDA-graph replay equals eager with no allocation growth.
At N=2048 (PP whole experts): every list GEMM gets its redo row; the fp64
gate against the released split path and the unsplit incumbent, with
in-range scales and with scales above 2^-5 (redo pass listed tiles);
in-range scales give bitwise the regular kernels' output; graph replay
equals eager (the replay zeroes stale counts) with no allocation growth.
"""

import re
from pathlib import Path

import pytest
import torch

from tests.kernels import test_ampere_tp4_marlin_prefill as tp4
from vllm.ampere_prefill import pp_marlin_prefill as pmp

SWITCH = "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED"
PP_SWITCH = "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED"
FLAG = tp4.FLAG
hp = tp4.hp
E, TOPK, K, N = hp.E, hp.TOPK, hp.K, hp.N
gpu = hp.gpu
weights = hp.weights
ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "csrc" / "libtorch_stable" / "moe" / "ampere_marlin"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.delenv(PP_SWITCH, raising=False)
    monkeypatch.delenv("VLLM_GLM5_PP_MARLIN_PREFILL", raising=False)


# ------------------------------------------------------------------ CPU ----

def test_switch_defaults_on_and_kill_switch(monkeypatch):
    import vllm.envs as envs

    assert envs.VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED is True
    monkeypatch.setenv(SWITCH, "0")
    assert envs.VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED is False
    assert envs.VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED is True
    monkeypatch.setenv(PP_SWITCH, "0")
    assert envs.VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED is False


def test_tile_tables():
    flag, tables = pmp.TILE_TABLES[512]
    assert flag == SWITCH
    assert tables["w13"] == {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)}
    assert tables["w2"] == {64: (64, 256, 2), 48: (64, 256, 2), 32: (64, 256, 2)}
    flag, tables = pmp.TILE_TABLES[2048]
    assert flag == PP_SWITCH
    wide = {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)}
    assert tables == {"w13": wide, "w2": wide}
    assert pmp.REDO_N == (2048,)


def test_compiled_tiles_reasons(monkeypatch):
    from vllm import ampere_marlin

    dev = torch.device("cuda", 0)   # a device object only; nothing is allocated
    monkeypatch.setenv(SWITCH, "0")
    assert pmp.compiled_tiles(512, dev, create=False) == (None, None, None, None)
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv(PP_SWITCH, "0")
    assert pmp.compiled_tiles(2048, dev, create=False) == (None, None, None, None)
    assert pmp.compiled_tiles(1024, dev, create=False) == (None, None, None, None)
    monkeypatch.setattr(ampere_marlin, "prefill_tile_op", lambda: (None, "no library"))
    op, _, _, why = pmp.compiled_tiles(512, dev, create=False)
    assert op is None and SWITCH in why and "no library" in why
    monkeypatch.setattr(ampere_marlin, "prefill_tile_op", lambda: ("OP", None))
    op, _, _, why = pmp.compiled_tiles(512, torch.device("cpu"), create=True)
    assert op is None and "not a CUDA device" in why
    monkeypatch.setattr(pmp, "_TILE_SCRATCH", {})
    op, _, _, why = pmp.compiled_tiles(512, dev, create=False)
    assert op is None and "graph capture" in why
    monkeypatch.setattr(pmp, "_TILE_SCRATCH", {str(dev): "SCRATCH"})
    op, tables, c_tmp, why = pmp.compiled_tiles(512, dev, create=False)
    assert (op, c_tmp, why) == ("OP", "SCRATCH", None)
    assert tables is pmp.TILE_TABLES[512][1]


def test_prefill_tile_op_never_raises(monkeypatch):
    from vllm import ampere_marlin

    def boom(name):
        raise ImportError("not built")

    monkeypatch.setattr(ampere_marlin, "_PREFILL_TILE", None)
    monkeypatch.setattr(ampere_marlin.importlib, "import_module", boom)
    op, why = ampere_marlin.prefill_tile_op()
    assert op is None and "not built" in why


class _Layer:
    w1_scale = "s1"
    w2_scale = "s2"

    def __init__(self, log):
        self.log = log

    def activation(self, *a, **k):
        self.log.append("act")

    def moe_sum(self, *a, **k):
        self.log.append("sum")


def _run(monkeypatch, compiled, n=N, redo=None):
    """Run pmp.run on CPU with fake lists, op and workspaces; returns the log."""
    import vllm.ampere_prefill.moe_split_align as sa
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    log = []
    lists = [(bs, f"sid{bs}", f"eid{bs}", f"n{bs}") for bs in (64, 48, 32, 16)]
    monkeypatch.setattr(sa, "split_align", lambda ids, e, buf: lists)
    monkeypatch.setattr(marlin_utils, "get_marlin_workspace", lambda dev: "locks")

    def released(*a, **k):
        log.append(("released", k["size_n"], k["size_k"], k["moe_block_size"],
                    k["thread_k"], k["thread_n"], k["blocks_per_sm"]))

    monkeypatch.setattr(pmp.ops, "moe_wna16_marlin_gemm", released)
    M = 8
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    w1 = torch.zeros(E, 1, 1)
    w2 = torch.zeros(E, n // 16, 1)
    ids = torch.zeros(M, TOPK, dtype=torch.int32)
    ws13 = torch.empty(M * TOPK * K, dtype=torch.bfloat16)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16)
    if compiled:
        def op(*a):
            assert a[-2] == "C_TMP" and a[8] == "locks"
            r = a[-1]
            log.append(("tile", a[18], a[19], a[13], a[23], a[24], a[25],
                        None if r is None else int(r[0])))
        compiled = (op, pmp.TILE_TABLES[n][1], "C_TMP", redo)
    pmp.run(_Layer(log), torch.empty(M, K), x, w1, w2, torch.zeros(M, TOPK),
            ids, "silu", ws13, ws2, buf=None, compiled=compiled or None)
    return log


def test_compiled_run_passes_the_tile_table(monkeypatch):
    log = _run(monkeypatch, compiled=True)
    w13 = [(e[3], e[4:7]) for e in log if e[0] == "tile" and e[1] == 2 * N]
    w2 = [(e[3], e[4:7]) for e in log if e[0] == "tile" and e[1] == K]
    assert w13 == [(64, (64, 512, 1)), (48, (64, 512, 1)), (32, (64, 512, 1)),
                   (16, (-1, -1, -1))]
    assert w2 == [(64, (64, 256, 2)), (48, (64, 256, 2)), (32, (64, 256, 2)),
                  (16, (-1, -1, -1))]
    assert all(e[7] is None for e in log if e[0] == "tile")   # no redo at N=512
    assert [e for e in log if isinstance(e, str)] == ["act", "sum"]
    assert not [e for e in log if e[0] == "released"]


def test_pp_run_passes_the_wide_table_and_one_redo_row_per_gemm(monkeypatch):
    """N=2048: (64,512,1) on the 64/48/32 lists of both GEMMs, Marlin's choice
    on the 16 list, and redo row i for w13 list i, row 4 + i for w2 list i,
    every count zeroed before the first launch."""
    redo = torch.arange(8 * 5, dtype=torch.int32).view(8, 5) + 1
    for i in range(8):
        redo[i, 1] = 100 + i            # row tag (not a count)
    log = _run(monkeypatch, compiled=True, n=2048, redo=redo)
    tiles = [e for e in log if e[0] == "tile"]
    assert [e[1] for e in tiles] == [4096] * 8          # 2N = K = 4096
    wide = (64, 512, 1)
    assert [(e[3], e[4:7]) for e in tiles] == (
        [(64, wide), (48, wide), (32, wide), (16, (-1, -1, -1))] * 2)
    assert [e[7] for e in tiles] == [0] * 8              # counts zeroed first
    assert redo[:, 0].tolist() == [0] * 8
    assert redo[:, 1].tolist() == [100 + i for i in range(8)]


def test_pp_run_passes_the_rows_in_order(monkeypatch):
    seen = []
    import vllm.ampere_prefill.moe_split_align as sa
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    lists = [(bs, f"sid{bs}", f"eid{bs}", f"n{bs}") for bs in (64, 48, 32, 16)]
    monkeypatch.setattr(sa, "split_align", lambda ids, e, buf: lists)
    monkeypatch.setattr(marlin_utils, "get_marlin_workspace", lambda dev: "locks")
    redo = torch.zeros(8, 3, dtype=torch.int32)
    redo[:, 1] = torch.arange(8, dtype=torch.int32)
    op = lambda *a: seen.append(int(a[-1][1]))   # noqa: E731
    M = 8
    pmp.run(_Layer([]), torch.empty(M, K), torch.zeros(M, K, dtype=torch.bfloat16),
            torch.zeros(E, 1, 1), torch.zeros(E, 2048 // 16, 1), torch.zeros(M, TOPK),
            torch.zeros(M, TOPK, dtype=torch.int32), "silu",
            torch.empty(M * TOPK * K, dtype=torch.bfloat16),
            torch.empty(M * TOPK * K, dtype=torch.bfloat16), buf=None,
            compiled=(op, pmp.TILE_TABLES[2048][1], "C_TMP", redo))
    assert seen == list(range(8))


def test_redo_area_bound_covers_the_op_check():
    """_redo_len >= 1 + the op's bound (ops.cu: ceil((rows + E*bs)/bs) m-blocks
    x prob_n/thread_n n-tiles + one per CTA) for every list size, both GEMMs,
    thread_n >= 64 and up to 4 CTAs per SM."""
    for rows in (8, 3072, 18432, 27648, 27704):
        for sms in (70, 74, 108):
            have = pmp._redo_len(rows, E, sms)
            for bs in (16, 32, 48, 64):
                for tn in (64, 128, 256, 512):
                    for bps in (1, 2, 4):
                        need = (-(-(rows + E * bs) // bs) * (4096 // tn)
                                + sms * bps)
                        assert have >= 1 + need, (rows, sms, bs, tn, bps)


def test_tile_redo_allocates_only_outside_a_capture(monkeypatch):
    dev = torch.device("cpu")
    monkeypatch.setattr(pmp, "_TILE_REDO", {})
    monkeypatch.setattr(pmp, "_SMS", {str(dev): 74})
    assert pmp._tile_redo(dev, 1024, E, create=False) is None
    a = pmp._tile_redo(dev, 1024, E, create=True)
    assert a.shape == (8, pmp._redo_len(1024, E, 74)) and a.dtype == torch.int32
    assert pmp._tile_redo(dev, 512, E, create=False) is a     # big enough
    assert pmp._tile_redo(dev, 4096, E, create=False) is None  # would grow
    b = pmp._tile_redo(dev, 4096, E, create=True)
    assert b is not a and b.size(1) > a.size(1)
    assert pmp._RETIRED[-1] is a                               # never freed


def test_without_the_op_the_released_kernels_run(monkeypatch):
    log = _run(monkeypatch, compiled=False)
    assert not [e for e in log if e[0] == "tile"]
    rel = [(e[3], e[4:]) for e in log if e[0] == "released" and e[1] == 2 * N]
    assert rel == [(64, (64, 256, 1)), (48, (-1, -1, -1)), (32, (-1, -1, -1)),
                   (16, (-1, -1, -1))]


@pytest.mark.parametrize("op", ["OP", None])
def test_open_gate_hands_the_compiled_tiles_to_run(monkeypatch, op):
    """maybe_apply with the gate open: the compiled tiles (or None when the op
    is missing) reach run(), and the banners log (hashable arguments)."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setattr(pmp, "gate_reason", lambda *a, **k: None)
    monkeypatch.setattr(pmp, "_buffers", lambda *a, **k: {"t_max": 1 << 20})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    tables = pmp.TILE_TABLES[512][1]
    got = (op, tables, "C_TMP", None) if op else (None, None, None, "absent")
    monkeypatch.setattr(pmp, "compiled_tiles", lambda *a, **k: got)
    monkeypatch.setattr(pmp, "_tile_redo", lambda *a, **k: 1 / 0)  # not at N=512
    ran = []
    monkeypatch.setattr(pmp, "run", lambda *a, **k: ran.append(k["compiled"]))
    assert pmp.maybe_apply(hp._layer(n=512), **tp4._maybe_args(1728)) is True
    assert ran == [("OP", tables, "C_TMP", None) if op else None]


@pytest.mark.parametrize("capturing", [False, True])
def test_open_pp_gate_hands_the_redo_area_to_run(monkeypatch, capturing):
    """maybe_apply at N=2048: the redo area for M * 8 rows reaches run(); during
    a capture it is only looked up (None when it would have to grow)."""
    pp = tp4.PP_FLAG
    monkeypatch.setenv(pp, "1")
    monkeypatch.setenv(pp + "_MIN_TOKENS", "384")
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(pmp, "gate_reason", lambda *a, **k: None)
    monkeypatch.setattr(pmp, "_buffers", lambda *a, **k: {"t_max": 1 << 20})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    tables = pmp.TILE_TABLES[2048][1]
    monkeypatch.setattr(pmp, "compiled_tiles",
                        lambda *a, **k: ("OP", tables, "C_TMP", None))
    asked = []

    def redo(device, rows, e, create):
        asked.append((rows, e, create))
        return None if capturing else "REDO"

    monkeypatch.setattr(pmp, "_tile_redo", redo)
    ran = []
    monkeypatch.setattr(pmp, "run", lambda *a, **k: ran.append(k["compiled"]))
    M = 2304
    a = hp._meta_args(M=M, n=2048)
    args = dict(
        output=torch.empty(M, K, dtype=torch.bfloat16, device="meta"),
        hidden_states=a["hidden_states"], w1=a["w1"], w2=a["w2"],
        topk_weights=a["topk_weights"], topk_ids=a["topk_ids"],
        activation=a["activation"], global_num_experts=E, expert_map=None,
        workspace13=torch.empty(M * TOPK, K, dtype=torch.bfloat16, device="meta"),
        workspace2=torch.empty(M * TOPK * K, dtype=torch.bfloat16, device="meta"),
        apply_router_weight_on_input=False)
    assert pmp.maybe_apply(hp._layer(n=2048), **args) is True
    assert asked == [(M * TOPK, E, not capturing)]
    assert ran == [("OP", tables, "C_TMP", None if capturing else "REDO")]


def _kernels(text):
    return sorted(set(re.findall(r"Marlin<([^>]*)>", text)))


def test_library_declares_the_explicit_tiles():
    sel = (LIB / "kernel_selector_tiles.h").read_text()
    inst = (LIB / "kernels_tiles_sm80.cu").read_text()
    ks, ki = _kernels(sel), _kernels(inst)
    assert len(ks) == 27 and ks == ki
    pre = ("vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), "
           "vllm::kBFloat16.id(), ")
    for m in (2, 3, 4):
        for threads, n_blocks in ((256, 32), (128, 16)):   # (64,512) and (64,256)
            want = f"{pre}{threads}, {m}, {n_blocks}, 4, false, 4, 8, false, false"
            assert want in ks, (m, threads)
        # the fast_dequant twin of the (64,512) tile (PP whole experts)
        assert f"{pre}256, {m}, 32, 4, false, 4, 8, false, true" in ks, m
    fast = [k for k in ks if k.endswith("false, true")]
    assert len(fast) == 6
    # every fast twin has its regular kernel (the redo pass)
    assert all(k[: -len("true")] + "false" in ks for k in fast)
    assert "#define MARLIN_MOE_FAST_DEQUANT_REDO" in (LIB / "common_tiles.h").read_text()
    for f in ("CMakeLists.txt",):
        src = (ROOT / f).read_text()
        assert "ampere_marlin/prefill_tiles.cu" in src
        assert "ampere_marlin/kernels_tiles_sm80.cu" in src
    build = (LIB / "build_standalone.py").read_text()
    assert '"prefill_tiles.cu", "kernels_tiles_sm80.cu"' in build


def test_installed_library_exports_the_op():
    import importlib.util

    if importlib.util.find_spec("vllm._ampere_marlin_C") is None:
        pytest.skip("optional library not installed")
    from vllm import ampere_marlin

    op, why = ampere_marlin.prefill_tile_op()
    assert op is not None, why


# ------------------------------------------------------------------ GPU ----

def _have_op():
    from vllm import ampere_marlin

    return ampere_marlin.prefill_tile_op()[0] is not None


TP4_M = [384, 640, 642, 1728, 1732, 3456]


@gpu
@pytest.mark.parametrize("M,min_tokens", [(m, 384) for m in TP4_M]
                         + [(1, 1), (17, 1), (299, 1)])
def test_compiled_accuracy_vs_fp64(weights, monkeypatch, M, min_tokens):
    if not _have_op():
        pytest.skip("library without prefill_tile_gemm")
    from vllm import ampere_marlin

    calls = []
    real = ampere_marlin.prefill_tile_op()

    def spy(*a):
        calls.append(a[13])
        return real[0](*a)

    monkeypatch.setattr(ampere_marlin, "prefill_tile_op", lambda: (spy, None))
    layer = tp4._gpu_layer(weights)
    x, tw, ti = hp._inputs(M, 100 + M)
    monkeypatch.setenv(SWITCH, "1")
    on = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1", min_tokens)
    assert calls, "compiled tiles ran"
    again = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1", min_tokens)
    assert torch.equal(on, again), "bitwise run to run"
    inc = hp._incumbent(weights, x, tw, ti, monkeypatch)
    rows = hp._sample_rows(M)
    ref = hp.reference_fp64(x, weights, tw, ti, rows)
    hp._check_accuracy(on[rows], inc[rows], ref, f"M={M}")


@gpu
@pytest.mark.parametrize("M", [640, 1728])
def test_missing_op_is_bitwise_the_released_split_path(weights, monkeypatch, M):
    from vllm import ampere_marlin

    layer = tp4._gpu_layer(weights)
    x, tw, ti = hp._inputs(M, 7 + M)
    monkeypatch.setenv(SWITCH, "0")
    released = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1")
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(ampere_marlin, "prefill_tile_op", lambda: (None, "absent"))
    fallback = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1")
    assert torch.equal(fallback, released)


@gpu
def test_compiled_graph_replay_equals_eager(weights, monkeypatch):
    if not _have_op():
        pytest.skip("library without prefill_tile_gemm")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    M = 1728
    layer = tp4._gpu_layer(weights)
    x, tw, ti = hp._inputs(M, 3)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setenv(SWITCH, "1")
    pmp.warmup(x.device, E, 3460, TOPK, N)
    assert str(x.device) in pmp._TILE_SCRATCH
    eager = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1")
    out = torch.empty_like(x)
    ws13 = torch.empty(M * TOPK, K, dtype=torch.bfloat16, device=x.device)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16, device=x.device)

    def step():
        layer.apply(out, x, weights["w1"], weights["w2"], tw, ti,
                    MoEActivation.SILU, E, None, None, None, ws13, ws2, None,
                    False)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        out.zero_()
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert torch.equal(out, eager)


# ------------------------------------------------- GPU, PP whole experts ----

def _pp():
    from tests.kernels import test_ampere_pp_marlin_prefill as pp

    return pp


@pytest.fixture(scope="module")
def pp_weights():
    """PP (N=2048) weights as test_ampere_pp_marlin_prefill's, plus a copy whose
    scales exceed 2^-5 (the fast dequant's exact range) in a few groups of a
    few experts, so the redo pass runs."""
    pp = _pp()
    En, Kn, Nn, G = pp.E, pp.K, pp.N, pp.G
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(0)
    info = torch.iinfo(torch.int32)

    def packed(*shape):
        return torch.randint(info.min, info.max, shape, generator=g, device=dev,
                             dtype=torch.int32)

    def scales(*shape):
        return ((torch.rand(*shape, generator=g, device=dev) * 0.2 + 0.9)
                * 0.0034).to(torch.bfloat16)

    wd = {"w1": packed(En, Kn // 16, 2 * Nn * 2), "w2": packed(En, Nn // 16, Kn * 2),
          "w1_scale": scales(En, Kn // G, 2 * Nn), "w2_scale": scales(En, Nn // G, Kn)}
    big = dict(wd)
    big["w1_scale"] = wd["w1_scale"].clone()
    big["w2_scale"] = wd["w2_scale"].clone()
    # 0.0334 is the largest real checkpoint scale seen above 2^-5 (stage 0 L0 w1)
    for e in (0, 7, 130, 287):
        big["w1_scale"][e, 3, 100:140] = 0.0334
        big["w2_scale"][e, 1, 2000:2010] = -0.05
    yield wd, big
    del wd, big
    torch.cuda.empty_cache()


def _pp_spy(monkeypatch):
    from vllm import ampere_marlin

    real = ampere_marlin.prefill_tile_op()
    if real[0] is None:
        pytest.skip("library without prefill_tile_gemm")
    calls = []

    def spy(*a):
        calls.append(a[-1] is not None)
        return real[0](*a)

    monkeypatch.setattr(ampere_marlin, "prefill_tile_op", lambda: (spy, None))
    return calls


@gpu
@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("M,min_tokens", [(1282, 384), (2304, 384), (3456, 384),
                                          (17, 1), (299, 1)])
def test_pp_compiled_accuracy_vs_fp64(pp_weights, monkeypatch, M, min_tokens, big):
    """PP whole-expert compiled tiles + fast dequant: every list GEMM gets a redo
    row; bitwise run to run; within 1.10x mean / 1.25x max of the released
    split path's fp64 error; with out-of-range scales the redo pass lists
    tiles and the result still passes."""
    pp = _pp()
    wd = pp_weights[1 if big else 0]
    calls = _pp_spy(monkeypatch)
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 100 + M)
    monkeypatch.setenv(PP_SWITCH, "1")
    on = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", min_tokens)
    assert calls and all(calls), "every list GEMM ran with a redo row"
    counts = pmp._TILE_REDO[str(x.device)][:, 0].tolist()
    if big and M >= 384:
        assert sum(counts) > 0, "out-of-range scales listed tiles"
    if not big:
        assert sum(counts) == 0, counts
    again = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", min_tokens)
    assert torch.equal(on, again), "bitwise run to run"
    monkeypatch.setenv(PP_SWITCH, "0")
    released = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", min_tokens)
    rows = pp._sample_rows(M)
    ref = pp.reference_fp64(x, wd, tw, ti, rows)
    pp._check_accuracy(on[rows], released[rows], ref, f"M={M} big={big}")
    inc = pp._incumbent(wd, x, tw, ti, monkeypatch)
    pp._check_accuracy(on[rows], inc[rows], ref, f"M={M} big={big} vs unsplit")


@gpu
@pytest.mark.parametrize("M", [1282, 2304, 3456])
def test_pp_fast_dequant_is_bitwise_the_regular_kernels(pp_weights, monkeypatch, M):
    """In-range scales: the fast_dequant kernels (one-HFMA2 dequant, overlapped
    write-out) list nothing and give bitwise the regular kernels' output
    (the same op and tiles without a redo area)."""
    pp = _pp()
    wd = pp_weights[0]
    _pp_spy(monkeypatch)
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 40 + M)
    monkeypatch.setenv(PP_SWITCH, "1")
    fast = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1")
    assert sum(pmp._TILE_REDO[str(x.device)][:, 0].tolist()) == 0
    monkeypatch.setattr(pmp, "_tile_redo", lambda *a, **k: None)
    regular = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1")
    assert torch.equal(fast, regular)


@gpu
def test_pp_compiled_graph_replay_equals_eager(pp_weights, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    pp = _pp()
    if not _have_op():
        pytest.skip("library without prefill_tile_gemm")
    wd = pp_weights[1]
    M = 2304
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 3)
    monkeypatch.setenv(pp.FLAG, "1")
    monkeypatch.setenv(pp.FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setenv(PP_SWITCH, "1")
    monkeypatch.setattr(pmp, "_TILE_REDO", {})
    pmp.warmup(x.device, pp.E, 3460, pp.TOPK, pp.N)
    redo = pmp._TILE_REDO[str(x.device)]
    eager = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1")
    assert sum(redo[:, 0].tolist()) > 0
    out = torch.empty_like(x)
    ws13 = torch.empty(M * pp.TOPK, pp.K, dtype=torch.bfloat16, device=x.device)
    ws2 = torch.empty(M * pp.TOPK * pp.K, dtype=torch.bfloat16, device=x.device)

    def step():
        layer.apply(out, x, wd["w1"], wd["w2"], tw, ti, MoEActivation.SILU,
                    pp.E, None, None, None, ws13, ws2, None, False)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    assert pmp._TILE_REDO[str(x.device)] is redo
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        out.zero_()
        redo[:, 0].fill_(-5)     # stale counts: the replay zeroes them
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert torch.equal(out, eager)
