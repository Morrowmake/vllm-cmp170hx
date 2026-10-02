# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled Marlin MoE prefill tiles (VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED,
on by default; 0 is the kill switch): the split-list GEMMs of the TP4 Marlin
prefill through the optional library's prefill_tile_gemm.

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
"""

import re
from pathlib import Path

import pytest
import torch

from tests.kernels import test_ampere_tp4_marlin_prefill as tp4
from vllm.ampere_prefill import pp_marlin_prefill as pmp

SWITCH = "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED"
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
    monkeypatch.delenv("VLLM_GLM5_PP_MARLIN_PREFILL", raising=False)


# ------------------------------------------------------------------ CPU ----

def test_switch_defaults_on_and_kill_switch(monkeypatch):
    import vllm.envs as envs

    assert envs.VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED is True
    monkeypatch.setenv(SWITCH, "0")
    assert envs.VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED is False


def test_tile_tables():
    flag, tables = pmp.TILE_TABLES[512]
    assert flag == SWITCH
    assert tables["w13"] == {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)}
    assert tables["w2"] == {64: (64, 256, 2), 48: (64, 256, 2), 32: (64, 256, 2)}
    assert 2048 not in pmp.TILE_TABLES   # whole experts: none yet


def test_compiled_tiles_reasons(monkeypatch):
    from vllm import ampere_marlin

    dev = torch.device("cuda", 0)   # a device object only; nothing is allocated
    monkeypatch.setenv(SWITCH, "0")
    assert pmp.compiled_tiles(512, dev, create=False) == (None, None, None, None)
    monkeypatch.setenv(SWITCH, "1")
    assert pmp.compiled_tiles(2048, dev, create=False) == (None, None, None, None)
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


def _run(monkeypatch, compiled):
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
    w2 = torch.zeros(E, N // 16, 1)
    ids = torch.zeros(M, TOPK, dtype=torch.int32)
    ws13 = torch.empty(M * TOPK * K, dtype=torch.bfloat16)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16)
    if compiled:
        def op(*a):
            assert a[-1] == "C_TMP" and a[8] == "locks"
            log.append(("tile", a[18], a[19], a[13], a[23], a[24], a[25]))
        compiled = (op, pmp.TILE_TABLES[512][1], "C_TMP")
    pmp.run(_Layer(log), torch.empty(M, K), x, w1, w2, torch.zeros(M, TOPK),
            ids, "silu", ws13, ws2, buf=None, compiled=compiled or None)
    return log


def test_compiled_run_passes_the_tile_table(monkeypatch):
    log = _run(monkeypatch, compiled=True)
    w13 = [(e[3], e[4:]) for e in log if e[0] == "tile" and e[1] == 2 * N]
    w2 = [(e[3], e[4:]) for e in log if e[0] == "tile" and e[1] == K]
    assert w13 == [(64, (64, 512, 1)), (48, (64, 512, 1)), (32, (64, 512, 1)),
                   (16, (-1, -1, -1))]
    assert w2 == [(64, (64, 256, 2)), (48, (64, 256, 2)), (32, (64, 256, 2)),
                  (16, (-1, -1, -1))]
    assert [e for e in log if isinstance(e, str)] == ["act", "sum"]
    assert not [e for e in log if e[0] == "released"]


def test_without_the_op_the_released_kernels_run(monkeypatch):
    log = _run(monkeypatch, compiled=False)
    assert not [e for e in log if e[0] == "tile"]
    rel = [(e[3], e[4:]) for e in log if e[0] == "released" and e[1] == 2 * N]
    assert rel == [(64, (64, 256, 1)), (48, (-1, -1, -1)), (32, (-1, -1, -1)),
                   (16, (-1, -1, -1))]


def _kernels(text):
    return sorted(set(re.findall(r"Marlin<([^>]*)>", text)))


def test_library_declares_the_explicit_tiles():
    sel = (LIB / "kernel_selector_tiles.h").read_text()
    inst = (LIB / "kernels_tiles_sm80.cu").read_text()
    ks, ki = _kernels(sel), _kernels(inst)
    assert len(ks) == 21 and ks == ki
    for m in (2, 3, 4):
        for threads, n_blocks in ((256, 32), (128, 16)):   # (64,512) and (64,256)
            want = (f"vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), "
                    f"vllm::kBFloat16.id(), {threads}, {m}, {n_blocks}, 4, false, 4, 8, false")
            assert want in ks, (m, threads)
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
