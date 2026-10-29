# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled Marlin MoE prefill tiles for PP whole experts
(VLLM_SM80_MARLIN_PREFILL, off by default): the split-list GEMMs of the
N=2048 Marlin prefill through the optional library's
prefill_tile_gemm, with the optimistic fast dequant and its per-GEMM redo
areas. The TP4 (N=512) lists have no tile table and run on the released
kernels.

CPU: the switch and the retired names of the separate tiles switch;
``compiled_tiles`` reasons (switch off,
no table for the width, library/op missing, not a CUDA device, no scratch
during a capture); the per-GEMM tile table reaches the op exactly ((64,512,1)
on the 64/48/32-row lists of both GEMMs, Marlin's own choice on the 16-row
list) with one redo row per list GEMM; without the op the released kernels
run with THREAD_CFG; the redo area bound and its allocation outside a
capture; ``prefill_tile_op`` never raises, refuses a library without the
start-ordered schedule and accepts one that registers the op; the startup
banners (split path, compiled tiles, fast dequant) and the kill switch
through ``maybe_apply``; a TP4 call never reaches the op; the library
sources declare the same 12 kernels in the selector and the instantiation
file and CMake and the standalone build compile exactly the library's
translation units; the in-tree library, when present, exports
prefill_tile_gemm.

GPU (sm_80 with ~12 GB free and the library; skipped otherwise): every list
GEMM gets its redo row; against an fp64 reference within 1.10x mean / 1.25x
max of the released split path's and the unsplit incumbent's error, with
in-range scales and with scales above 2^-5 (the redo pass lists tiles);
bitwise run to run; with the op missing the result is bitwise the released
split path (the lists without a tile table); in-range scales give bitwise
the regular kernels'
output; graph replay equals eager (the replay zeroes stale counts) with no
allocation growth; with every SM but one held by another kernel, as a
collective on another stream can hold them, the compiled tiles finish on the
one free SM, bitwise the result without the hold, and leave every semaphore
and the start counter at zero.
"""

import contextlib
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm

from tests.kernels import test_ampere_pp_marlin_prefill as pp
from vllm_sm80.kernels.marlin_moe import prefill as pmp

SWITCH = "VLLM_SM80_MARLIN_PREFILL"
# The names of the former separate tiles switch.
RETIRED_SWITCHES = ("VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED",
                    "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED")
FLAG = pp.FLAG
E, TOPK, K, N = pp.E, pp.TOPK, pp.K, pp.N
gpu = pp.gpu
ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "csrc" / "ampere_marlin"
ENGINE = Path(vllm.__file__).parents[1]
WIDE = {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)}
TILES_BANNER = f"Marlin MoE prefill compiled tiles active ({SWITCH}, N=2048)"
FAST_BANNER = f"Marlin MoE prefill fast dequant active ({SWITCH}, N=2048)"
assert N == 2048 and FLAG == SWITCH


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (SWITCH, *RETIRED_SWITCHES):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ CPU ----

def test_switch_defaults_off_and_turns_on(monkeypatch):
    import vllm.envs as envs

    assert envs.VLLM_SM80_MARLIN_PREFILL is False
    monkeypatch.setenv(SWITCH, "0")
    assert envs.VLLM_SM80_MARLIN_PREFILL is False
    monkeypatch.setenv(SWITCH, "1")
    assert envs.VLLM_SM80_MARLIN_PREFILL is True
    from vllm_sm80 import envs as fork_envs  # typed declarations

    src = open(fork_envs.__file__).read()
    assert f"    {SWITCH}: bool = False\n" in src


def test_retired_names_are_ignored(monkeypatch):
    """The names of the former tiles switch only warn; they never turn the
    tiles or the split path on or off, and the TP4 tiles switch has no
    successor."""
    import vllm.envs as envs
    from vllm_sm80 import envs as fork_envs

    for name in RETIRED_SWITCHES:
        monkeypatch.setenv(name, "1")
        assert name not in envs.environment_variables
        assert fork_envs.RETIRED_ENV_VARS[name].startswith(f"merged into {SWITCH}")
    assert envs.VLLM_SM80_MARLIN_PREFILL is False
    monkeypatch.setenv(SWITCH, "1")
    assert envs.VLLM_SM80_MARLIN_PREFILL is True
    assert fork_envs.RETIRED_ENV_VARS[
        "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED"
    ].startswith("removed;")


def test_tile_tables():
    assert set(pmp.TILE_TABLES) == {2048}
    flag, tables = pmp.TILE_TABLES[2048]
    assert flag == SWITCH
    assert tables == {"w13": WIDE, "w2": WIDE}
    assert pmp.REDO_N == (2048,)


def test_compiled_tiles_reasons(monkeypatch):
    from vllm_sm80.kernels.marlin_moe import extension

    dev = torch.device("cuda", 0)   # a device object only; nothing is allocated
    monkeypatch.setenv(SWITCH, "0")
    assert pmp.compiled_tiles(2048, dev, create=False) == (None, None, None, None)
    monkeypatch.setenv(SWITCH, "1")
    # TP4 shards and any other width: no table, not requested
    assert pmp.compiled_tiles(512, dev, create=False) == (None, None, None, None)
    assert pmp.compiled_tiles(1024, dev, create=False) == (None, None, None, None)
    monkeypatch.setattr(extension, "prefill_tile_op", lambda: (None, "no library"))
    op, _, _, why = pmp.compiled_tiles(2048, dev, create=False)
    assert op is None and SWITCH in why and "no library" in why
    monkeypatch.setattr(extension, "prefill_tile_op", lambda: ("OP", None))
    op, _, _, why = pmp.compiled_tiles(2048, torch.device("cpu"), create=True)
    assert op is None and "not a CUDA device" in why
    monkeypatch.setattr(pmp, "_TILE_SCRATCH", {})
    op, _, _, why = pmp.compiled_tiles(2048, dev, create=False)
    assert op is None and "graph capture" in why
    monkeypatch.setattr(pmp, "_TILE_SCRATCH", {str(dev): "SCRATCH"})
    op, tables, c_tmp, why = pmp.compiled_tiles(2048, dev, create=False)
    assert (op, c_tmp, why) == ("OP", "SCRATCH", None)
    assert tables is pmp.TILE_TABLES[2048][1]


def test_prefill_tile_op_never_raises(monkeypatch):
    from vllm_sm80.kernels.marlin_moe import extension

    def boom(name):
        raise ImportError("not built")

    monkeypatch.setattr(extension, "_PREFILL_TILE", None)
    monkeypatch.setattr(extension.importlib, "import_module", boom)
    op, why = extension.prefill_tile_op()
    assert op is None and "not built" in why


@pytest.mark.parametrize("version", [None, 1])
def test_library_without_ordered_schedule_keeps_the_released_kernels(
        monkeypatch, version):
    """A library whose prefill tiles need the whole grid resident (no
    prefill_schedule_version 2) is not used for prefill."""
    from vllm_sm80.kernels.marlin_moe import extension

    info = {} if version is None else {"prefill_schedule_version": version}
    module = SimpleNamespace(build_info=lambda: info)
    monkeypatch.setattr(extension, "_PREFILL_TILE", None)
    monkeypatch.setattr(extension.importlib, "import_module", lambda n: module)
    monkeypatch.setattr(extension, "_validate_extension", lambda m: None)
    op, why = extension.prefill_tile_op()
    assert op is None and "every CTA resident" in why


def _schema():
    from vllm_sm80.kernels.marlin_moe import extension

    return extension._PREFILL_TILE_SCHEMA


def _fake_library(monkeypatch, names, has_cuda=True):
    """vllm_sm80._ampere_marlin_C as a module whose decode operators validate and
    whose prefill_tile_gemm has argument ``names``."""
    from vllm_sm80.kernels.marlin_moe import extension

    module = SimpleNamespace(
        build_info=lambda: {"prefill_schedule_version": 2})
    monkeypatch.setattr(extension, "_PREFILL_TILE", None)
    monkeypatch.setattr(extension.importlib, "import_module", lambda n: module)
    monkeypatch.setattr(extension, "_validate_extension", lambda m: None)
    schema = SimpleNamespace(
        arguments=[SimpleNamespace(name=n) for n in names.split()])
    found = []

    def find(qualified, overload):
        found.append(qualified)
        return SimpleNamespace(schema=lambda: schema)

    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", find)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key",
                        lambda q, k: has_cuda)
    monkeypatch.setattr(torch.ops, "_ampere_marlin_C",
                        SimpleNamespace(prefill_tile_gemm="OP"), raising=False)
    return extension, found


def test_prefill_tile_op_accepts_a_library_with_the_op(monkeypatch):
    extension, found = _fake_library(monkeypatch, _schema())
    assert extension.prefill_tile_op() == ("OP", None)
    assert found == ["_ampere_marlin_C::prefill_tile_gemm"]
    # cached: a second call does not look again
    assert extension.prefill_tile_op() == ("OP", None)
    assert len(found) == 1


@pytest.mark.parametrize("case", ["schema", "no_cuda"])
def test_prefill_tile_op_refuses_an_incompatible_op(monkeypatch, case):
    names = _schema()
    if case == "schema":
        names = names.replace(" redo", "")
    extension, _ = _fake_library(monkeypatch, names, has_cuda=case != "no_cuda")
    op, why = extension.prefill_tile_op()
    assert op is None
    assert ("incompatible operator schema" if case == "schema"
            else "missing CUDA implementation") in why


def test_prefill_tile_schema_matches_the_registration():
    """The loader's expected argument names are those prefill_tiles.cu
    registers."""
    src = (LIB / "prefill_tiles.cu").read_text()
    decl = "".join(re.findall(r'"([^"]*)"', src.split("m.def(", 1)[1]
                              .split(");", 1)[0]))
    args = decl.split("(", 1)[1].rsplit(")", 1)[0]
    names = [a.split("=")[0].split()[-1] for a in args.split(",")]
    assert " ".join(names) == _schema()


class _Layer:
    w1_scale = "s1"
    w2_scale = "s2"

    def __init__(self, log):
        self.log = log

    def activation(self, *a, **k):
        self.log.append("act")

    def moe_sum(self, *a, **k):
        self.log.append("sum")


def _run(monkeypatch, compiled, redo=None):
    """Run pmp.run on CPU at N=2048 with fake lists, op and workspaces; returns
    the log.  The compiled op gets its own semaphores, never the shared Marlin
    locks."""
    import vllm_sm80.kernels.marlin_moe.split_align as sa
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
            assert a[-2] == "C_TMP" and a[8] == "TILE_LOCKS"
            r = a[-1]
            log.append(("tile", a[18], a[19], a[13], a[23], a[24], a[25],
                        None if r is None else int(r[0])))
        compiled = (op, pmp.TILE_TABLES[N][1], ("C_TMP", "TILE_LOCKS"), redo)
    pmp.run(_Layer(log), torch.empty(M, K), x, w1, w2, torch.zeros(M, TOPK),
            ids, "silu", ws13, ws2, buf=None, compiled=compiled or None)
    return log


def test_run_passes_the_wide_table_and_one_redo_row_per_gemm(monkeypatch):
    """(64,512,1) on the 64/48/32 lists of both GEMMs, Marlin's choice on the
    16 list, and redo row i for w13 list i, row 4 + i for w2 list i, every
    count zeroed before the first launch."""
    redo = torch.arange(8 * 5, dtype=torch.int32).view(8, 5) + 1
    for i in range(8):
        redo[i, 1] = 100 + i            # row tag (not a count)
    log = _run(monkeypatch, compiled=True, redo=redo)
    tiles = [e for e in log if e[0] == "tile"]
    assert [e[1] for e in tiles] == [4096] * 8          # 2N = K = 4096
    assert [e[2] for e in tiles] == [K] * 4 + [N] * 4
    wide = (64, 512, 1)
    assert [(e[3], e[4:7]) for e in tiles] == (
        [(64, wide), (48, wide), (32, wide), (16, (-1, -1, -1))] * 2)
    assert [e[7] for e in tiles] == [0] * 8              # counts zeroed first
    assert redo[:, 0].tolist() == [0] * 8
    assert redo[:, 1].tolist() == [100 + i for i in range(8)]
    assert [e for e in log if isinstance(e, str)] == ["act", "sum"]
    assert not [e for e in log if e[0] == "released"]


def test_run_without_a_redo_area_runs_the_regular_tiles(monkeypatch):
    log = _run(monkeypatch, compiled=True, redo=None)
    tiles = [e for e in log if e[0] == "tile"]
    assert len(tiles) == 8 and all(e[7] is None for e in tiles)


def test_run_passes_the_rows_in_order(monkeypatch):
    seen = []
    import vllm_sm80.kernels.marlin_moe.split_align as sa
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    lists = [(bs, f"sid{bs}", f"eid{bs}", f"n{bs}") for bs in (64, 48, 32, 16)]
    monkeypatch.setattr(sa, "split_align", lambda ids, e, buf: lists)
    monkeypatch.setattr(marlin_utils, "get_marlin_workspace", lambda dev: "locks")
    redo = torch.zeros(8, 3, dtype=torch.int32)
    redo[:, 1] = torch.arange(8, dtype=torch.int32)
    op = lambda *a: seen.append(int(a[-1][1]))   # noqa: E731
    M = 8
    pmp.run(_Layer([]), torch.empty(M, K), torch.zeros(M, K, dtype=torch.bfloat16),
            torch.zeros(E, 1, 1), torch.zeros(E, N // 16, 1), torch.zeros(M, TOPK),
            torch.zeros(M, TOPK, dtype=torch.int32), "silu",
            torch.empty(M * TOPK * K, dtype=torch.bfloat16),
            torch.empty(M * TOPK * K, dtype=torch.bfloat16), buf=None,
            compiled=(op, pmp.TILE_TABLES[N][1], ("C_TMP", "LOCKS"), redo))
    assert seen == list(range(8))


def test_without_the_op_the_released_kernels_run(monkeypatch):
    """THREAD_CFG on both GEMMs (w13: K = 4096 deep, w2: N = 2048 deep; at
    N = 2048 both are 4096 wide)."""
    log = _run(monkeypatch, compiled=False)
    assert not [e for e in log if e[0] == "tile"]
    want = [(64, (64, 256, 1)), (48, (-1, -1, -1)), (32, (-1, -1, -1)),
            (16, (-1, -1, -1))]
    for size_k in (K, N):
        rel = [(e[3], e[4:]) for e in log
               if e[0] == "released" and e[1:3] == (4096, size_k)]
        assert rel == want, size_k
    assert [e for e in log if isinstance(e, str)] == ["act", "sum"]


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


def _maybe_args(M, n=N):
    a = pp._meta_args(M=M, n=n)
    return dict(
        output=torch.empty(M, K, dtype=torch.bfloat16, device="meta"),
        hidden_states=a["hidden_states"], w1=a["w1"], w2=a["w2"],
        topk_weights=a["topk_weights"], topk_ids=a["topk_ids"],
        activation=a["activation"], global_num_experts=E, expert_map=None,
        workspace13=torch.empty(M * TOPK, K, dtype=torch.bfloat16, device="meta"),
        workspace2=torch.empty(M * TOPK * K, dtype=torch.bfloat16, device="meta"),
        apply_router_weight_on_input=False)


def _open_gate(monkeypatch, capturing=False):
    """Open the gate on CPU (meta tensors; the device check is the only
    closed condition here) and record run()'s ``compiled`` and the banners."""
    monkeypatch.setenv(FLAG, "1")
    pp._min_tokens(monkeypatch, 384)
    monkeypatch.setattr(pmp, "gate_reason", lambda *a, **k: None)
    monkeypatch.setattr(pmp, "_buffers", lambda *a, **k: {"t_max": 1 << 20})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: capturing)
    ran, banners = [], []
    monkeypatch.setattr(pmp, "run", lambda *a, **k: ran.append(k["compiled"]))
    monkeypatch.setattr(pmp.logger, "info_once",
                        lambda msg, *args, **k: banners.append(msg % args))
    return ran, banners


@pytest.mark.parametrize("capturing", [False, True])
def test_open_gate_hands_the_tiles_and_redo_area_to_run(monkeypatch, capturing):
    """maybe_apply at N=2048: the compiled tiles and the redo area for M * 8
    rows reach run(); during a capture the area is only looked up (None when
    it would have to grow).  The banners name the switch and the width."""
    ran, banners = _open_gate(monkeypatch, capturing)
    tables = pmp.TILE_TABLES[2048][1]
    monkeypatch.setattr(pmp, "compiled_tiles",
                        lambda *a, **k: ("OP", tables, "C_TMP", None))
    asked = []

    def redo(device, rows, e, create):
        asked.append((rows, e, create))
        return None if capturing else "REDO"

    monkeypatch.setattr(pmp, "_tile_redo", redo)
    M = 2304
    assert pmp.maybe_apply(pp._layer(n=2048), **_maybe_args(M)) is True
    assert asked == [(M * TOPK, E, not capturing)]
    assert ran == [("OP", tables, "C_TMP", None if capturing else "REDO")]
    assert banners[0].startswith(
        f"PP Marlin MoE prefill active ({FLAG}): split 64/48/32/16-row")
    assert banners[1].startswith(TILES_BANNER + ": w13 ")
    assert sum(b.startswith(FAST_BANNER) for b in banners) == (0 if capturing else 1)
    assert sum("no redo area" in b for b in banners) == (1 if capturing else 0)


def test_missing_op_keeps_the_released_kernels_and_says_why(monkeypatch):
    from vllm_sm80.kernels.marlin_moe import extension

    monkeypatch.setenv(SWITCH, "1")
    ran, banners = _open_gate(monkeypatch)
    monkeypatch.setattr(extension, "prefill_tile_op", lambda: (None, "absent"))
    monkeypatch.setattr(pmp, "_tile_redo", lambda *a, **k: pytest.fail("redo"))
    # The meta tensors stand in for a CUDA device; without the op
    # compiled_tiles returns before it would allocate anything.
    real = pmp.compiled_tiles
    monkeypatch.setattr(pmp, "compiled_tiles", lambda N, dev, create: real(
        N, torch.device("cuda", 0), create))
    assert pmp.maybe_apply(pp._layer(n=2048), **_maybe_args(2304)) is True
    assert ran == [None]
    assert not any(b.startswith((TILES_BANNER, FAST_BANNER)) for b in banners)
    assert any(f"{SWITCH} is set but" in b and "absent" in b
               and "released Marlin kernels" in b for b in banners)


def test_kill_switch_leaves_the_call_to_fused_marlin_moe(monkeypatch):
    """=0: the split path declines before any gate, library lookup or
    banner; the caller runs fused_marlin_moe."""
    from vllm_sm80.kernels.marlin_moe import extension

    ran, banners = _open_gate(monkeypatch)
    monkeypatch.setenv(SWITCH, "0")
    monkeypatch.setattr(extension, "prefill_tile_op",
                        lambda: pytest.fail("library consulted"))
    assert pmp.maybe_apply(pp._layer(n=2048), **_maybe_args(2304)) is False
    assert ran == []
    assert banners == []


def test_tp4_call_never_reaches_the_op(monkeypatch):
    """The TP4 split path (N=512) with the switch on runs on the released
    kernels: no table, no library lookup, no tile banner."""
    from vllm_sm80.kernels.marlin_moe import extension

    ran, banners = _open_gate(monkeypatch)
    monkeypatch.setattr(extension, "prefill_tile_op",
                        lambda: pytest.fail("library consulted"))
    assert pmp.maybe_apply(pp._layer(n=512), **_maybe_args(1728, n=512)) is True
    assert ran == [None]
    assert len(banners) == 1 and banners[0].startswith("TP4 Marlin MoE prefill active")


@pytest.mark.parametrize("tp", [1, 4])
def test_warmup_from_worker_passes_the_width(monkeypatch, tp):
    """The pre-capture hook hands N to warmup, so the op's scratch and the
    redo area exist before any capture. At TP4 the width is N/4 = 512, which
    has no tile table: compiled_tiles returns nothing without asking for the
    op or allocating scratch or a redo area."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(pmp, "_is_sm80", lambda d: True)
    calls = []
    monkeypatch.setattr(pmp, "warmup", lambda *a: calls.append(a))
    cfg = SimpleNamespace(n_routed_experts=E, num_experts_per_tok=TOPK,
                          moe_intermediate_size=N)
    worker = SimpleNamespace(
        device=torch.device("cuda", 0),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=cfg),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2304))
    assert pmp.warmup_from_worker(worker) == E
    assert calls == [(torch.device("cuda", 0), E, 2304, TOPK, N // tp)]
    if tp == 4:
        from vllm_sm80.kernels.marlin_moe import extension

        def unreachable(*args, **kwargs):
            raise AssertionError("the N=512 width must not reach the tiles")

        monkeypatch.setattr(extension, "prefill_tile_op", unreachable)
        monkeypatch.setattr(pmp, "_tile_scratch", unreachable)
        monkeypatch.setattr(pmp, "_tile_redo", unreachable)
        assert pmp.compiled_tiles(N // tp, torch.device("cuda", 0),
                                  create=True) == (None, None, None, None)


def _kernels(text):
    return sorted(set(re.findall(r"Marlin<([^>]*)>", text)))


def test_library_declares_the_explicit_tiles():
    sel = (LIB / "kernel_selector_tiles.h").read_text()
    inst = (LIB / "kernels_tiles_sm80.cu").read_text()
    ks, ki = _kernels(sel), _kernels(inst)
    assert len(ks) == 12 and ks == ki
    pre = ("vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), "
           "vllm::kBFloat16.id(), ")
    # 16-row lists: Marlin's own choice among its small-batch configs
    for threads, n_blocks, k_blocks in ((256, 8, 8), (128, 8, 4), (128, 4, 8)):
        for fast in ("false", "true"):
            want = (f"{pre}{threads}, 1, {n_blocks}, {k_blocks}, false, 4, 8, "
                    f"false, {fast}")
            assert want in ks, (threads, n_blocks, k_blocks, fast)
    for m in (2, 3, 4):
        # the (64,512) tile and its fast_dequant twin
        for fast in ("false", "true"):
            assert f"{pre}256, {m}, 32, 4, false, 4, 8, false, {fast}" in ks, m
    fast = [k for k in ks if k.endswith("false, true")]
    assert len(fast) == 6
    # every fast twin has its regular kernel (the redo pass)
    assert all(k[: -len("true")] + "false" in ks for k in fast)
    # every TILE_TABLES tile has its kernel (thread_k 64: thread_n / 2 threads,
    # 128 to 256, all warps along N)
    for _, tables in pmp.TILE_TABLES.values():
        for table in tables.values():
            for bs, (tk, tn, _) in table.items():
                threads = min(256, max(128, tn // 2))
                want = (f"{pre}{threads}, {bs // 16}, {tn // 16}, {tk // 16}, "
                        "false, 4, 8, false, false")
                assert want in ks, (bs, tk, tn)
    template = (LIB / "marlin_moe_template.h").read_text()
    assert "const bool fast_dequant" in template and "int* __restrict__ redo" in template
    units = {"decode_orig.cu", "decode_v2.cu", "prefill_tiles.cu",
             "kernels_tiles_sm80.cu", "module.cpp"}
    assert {p.name for p in LIB.iterdir() if p.suffix in (".cu", ".cpp")} == units
    # Built only by its own builder, from this directory's copy of the Marlin
    # MoE GEMM: vLLM's CMake build names none of it, and no unit reaches
    # vLLM's own Marlin MoE sources.
    assert "ampere_marlin" not in (ENGINE / "CMakeLists.txt").read_text()
    for path in LIB.iterdir():
        if path.suffix in (".cu", ".cuh", ".h", ".cpp"):
            assert "marlin_moe_wna16/" not in "".join(re.findall(
                r'#include\s+[<"]([^>"]+)[>"]', path.read_text())), path.name
    build = (LIB / "build_standalone.py").read_text()
    assert set(re.findall(r'"(\w+\.(?:cu|cpp))"', build)) == units
    assert '"prefill_schedule_version", 2' in (LIB / "module.cpp").read_text()


@pytest.mark.parametrize("unit", ["decode_orig.cu", "decode_v2.cu"])
def test_decode_source_includes_no_tile_header(unit):
    """The decode sources compile from their own text only: the tile settings
    and the marlin_moe_wna16 template cannot change the decode operators."""
    src = (LIB / unit).read_text()
    includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', src)
    assert not [i for i in includes if "marlin" in i or "tiles" in i], includes


def test_installed_library_exports_the_op():
    import importlib.util

    if importlib.util.find_spec("vllm_sm80._ampere_marlin_C") is None:
        pytest.skip("optional library not installed")
    from vllm_sm80.kernels.marlin_moe import extension

    op, why = extension.prefill_tile_op()
    assert op is not None, why


# ------------------------------------------------------------------ GPU ----

def _have_op():
    from vllm_sm80.kernels.marlin_moe import extension

    return extension.prefill_tile_op()[0] is not None


@pytest.fixture(scope="module")
def pp_weights():
    """PP (N=2048) weights as test_ampere_pp_marlin_prefill's, plus a copy whose
    scales exceed 2^-5 (the fast dequant's exact range) in a few groups of a
    few experts, so the redo pass runs."""
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


def _spy(monkeypatch):
    from vllm_sm80.kernels.marlin_moe import extension

    real = extension.prefill_tile_op()
    if real[0] is None:
        pytest.skip("library without prefill_tile_gemm")
    calls = []

    def spy(*a):
        calls.append(a[-1] is not None)
        return real[0](*a)

    monkeypatch.setattr(extension, "prefill_tile_op", lambda: (spy, None))
    return calls


@gpu
@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("M,min_tokens", [(1282, 384), (2304, 384), (3456, 384),
                                          (17, 1), (299, 1)])
def test_compiled_accuracy_vs_fp64(pp_weights, monkeypatch, M, min_tokens, big):
    """Compiled tiles + fast dequant: every list GEMM gets a redo row; bitwise
    run to run; within 1.10x mean / 1.25x max of the released split path's
    and the unsplit incumbent's fp64 error; with out-of-range scales the redo
    pass lists tiles and the result still passes."""
    wd = pp_weights[1 if big else 0]
    calls = _spy(monkeypatch)
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 100 + M)
    on = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", min_tokens, tiles=True)
    assert calls and all(calls), "every list GEMM ran with a redo row"
    counts = pmp._TILE_REDO[str(x.device)][:, 0].tolist()
    if big and M >= 384:
        assert sum(counts) > 0, "out-of-range scales listed tiles"
    if not big:
        assert sum(counts) == 0, counts
    again = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", min_tokens, tiles=True)
    assert torch.equal(on, again), "bitwise run to run"
    released = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", min_tokens)
    rows = pp._sample_rows(M)
    ref = pp.reference_fp64(x, wd, tw, ti, rows)
    pp._check_accuracy(on[rows], released[rows], ref, f"M={M} big={big}")
    inc = pp._incumbent(wd, x, tw, ti, monkeypatch)
    pp._check_accuracy(on[rows], inc[rows], ref, f"M={M} big={big} vs unsplit")


@gpu
@pytest.mark.parametrize("M", [1282, 2304])
def test_missing_op_is_bitwise_the_released_split_path(pp_weights, monkeypatch, M):
    from vllm_sm80.kernels.marlin_moe import extension

    wd = pp_weights[0]
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 7 + M)
    released = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1")
    monkeypatch.setattr(extension, "prefill_tile_op", lambda: (None, "absent"))
    fallback = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", tiles=True)
    assert torch.equal(fallback, released)


@gpu
@pytest.mark.parametrize("M", [1282, 2304, 3456])
def test_fast_dequant_is_bitwise_the_regular_kernels(pp_weights, monkeypatch, M):
    """In-range scales: the fast_dequant kernels (one-HFMA2 dequant, overlapped
    write-out) list nothing and give bitwise the regular kernels' output
    (the same op and tiles without a redo area)."""
    wd = pp_weights[0]
    _spy(monkeypatch)
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 40 + M)
    fast = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", tiles=True)
    assert sum(pmp._TILE_REDO[str(x.device)][:, 0].tolist()) == 0
    monkeypatch.setattr(pmp, "_tile_redo", lambda *a, **k: None)
    regular = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", tiles=True)
    assert torch.equal(fast, regular)


@gpu
def test_compiled_graph_replay_equals_eager(pp_weights, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if not _have_op():
        pytest.skip("library without prefill_tile_gemm")
    wd = pp_weights[1]
    M = 2304
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(M, 3)
    monkeypatch.setenv(FLAG, "1")
    pp._min_tokens(monkeypatch, 384)
    monkeypatch.setattr(pmp, "_TILE_REDO", {})
    pmp.warmup(x.device, E, 3460, TOPK, N)
    assert str(x.device) in pmp._TILE_SCRATCH
    redo = pmp._TILE_REDO[str(x.device)]
    eager = pp._apply(layer, wd, x, tw, ti, monkeypatch, "1", tiles=True)
    assert sum(redo[:, 0].tolist()) > 0
    out = torch.empty_like(x)
    ws13 = torch.empty(M * TOPK, K, dtype=torch.bfloat16, device=x.device)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16, device=x.device)

    def step():
        layer.apply(out, x, wd["w1"], wd["w2"], tw, ti, MoEActivation.SILU,
                    E, None, None, None, ws13, ws2, None, False)

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


# One CTA per SM (the dynamic shared memory request is the whole opt-in
# budget). flags[i] = 1 once CTA i holds its SM; a host write to
# flags[gridDim.x], or the time limit, releases them all.
_HOLD_SM_SRC = r"""
extern "C" __global__ void hold_sm(volatile int* flags, unsigned long long limit_ns) {
  extern __shared__ volatile unsigned char budget[];
  if (threadIdx.x != 0) return;
  budget[0] = 1;
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  flags[blockIdx.x] = 1;
  __threadfence_system();
  do {
    __nanosleep(10000);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (flags[gridDim.x] == 0 && now - start < limit_ns);
}
"""


@contextlib.contextmanager
def _all_but_one_sm_held(device, limit_s=60.0):
    """Hold every SM of ``device`` but one with a kernel on another stream, as
    a collective waiting for its peers can; yields once all of them run."""
    import ctypes

    from cuda.bindings import driver, nvrtc

    def ok(result):
        err, *rest = result
        assert int(err) == 0, err
        return rest[0] if len(rest) == 1 else None

    torch.cuda.synchronize(device)
    ok(driver.cuInit(0))
    prog = ok(nvrtc.nvrtcCreateProgram(_HOLD_SM_SRC.encode(), b"hold_sm.cu", 0,
                                       [], []))
    opts = [b"--gpu-architecture=sm_80"]
    ok(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    cubin = b" " * ok(nvrtc.nvrtcGetCUBINSize(prog))
    ok(nvrtc.nvrtcGetCUBIN(prog, cubin))
    ok(nvrtc.nvrtcDestroyProgram(prog))
    module = ok(driver.cuModuleLoadData(cubin))
    func = ok(driver.cuModuleGetFunction(module, b"hold_sm"))
    dev = ok(driver.cuDeviceGet(torch.device(device).index or 0))
    smem = ok(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        dev))
    ok(driver.cuFuncSetAttribute(
        func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem))
    assert ok(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, 32, smem)) == 1
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    flags = torch.zeros(sms, dtype=torch.int32, pin_memory=True)
    stream = torch.cuda.Stream(device)
    ok(driver.cuLaunchKernel(
        func, sms - 1, 1, 1, 32, 1, 1, smem, driver.CUstream(stream.cuda_stream),
        ((flags.data_ptr(), int(limit_s * 1e9)), (ctypes.c_void_p, ctypes.c_ulonglong)),
        0))
    try:
        deadline = time.monotonic() + 10
        while int(flags[:sms - 1].sum()) != sms - 1:
            assert time.monotonic() < deadline, "not every other SM was held"
            time.sleep(0.001)
        yield
    finally:
        flags[sms - 1] = 1
        stream.synchronize()
        ok(driver.cuModuleUnload(module))


@gpu
def test_compiled_progress_with_one_free_sm(pp_weights, monkeypatch):
    """Whole experts with scales above 2^-5: the fast_dequant kernels and the
    regular kernels' redo pass both finish while every SM but one is held,
    bitwise as without the hold, with every semaphore back at zero.  M=388
    (two 192-token resumed prefill chunks plus a 4-token speculative decode)
    has small block lists whose tiles are split across many stream-K CTAs."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_sm80.kernels.marlin_moe import extension

    if not _have_op():
        pytest.skip("library without prefill_tile_gemm")
    wd = pp_weights[1]
    layer = pp._gpu_layer(wd)
    x, tw, ti = pp._inputs(388, 6)
    monkeypatch.setattr(pmp, "_TILE_SCRATCH", {})
    monkeypatch.setattr(pmp, "_TILE_REDO", {})
    device, M = x.device, x.size(0)
    pmp.warmup(device, E, M, TOPK, N)
    scratch = pmp._tile_scratch(device, create=False)
    redo = pmp._tile_redo(device, M * TOPK, E, create=False)
    compiled = (extension.prefill_tile_op()[0], pmp.TILE_TABLES[N][1],
                scratch, redo)
    buf = pmp._buffers(device, E, M * TOPK, create=False)
    out = torch.empty_like(x)
    ws13 = torch.empty(M * TOPK, K, dtype=torch.bfloat16, device=device)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16, device=device)

    def step():
        pmp.run(layer, out, x, wd["w1"], wd["w2"], tw, ti, MoEActivation.SILU,
                ws13, ws2, buf, compiled=compiled)

    step()
    torch.cuda.synchronize(device)
    free = out.clone()
    out.fill_(float("nan"))
    done = torch.cuda.Event()
    with _all_but_one_sm_held(device):
        step()
        done.record()
        deadline = time.monotonic() + 15
        while not done.query() and time.monotonic() < deadline:
            time.sleep(0.001)
        finished = done.query()
    torch.cuda.synchronize(device)
    assert finished, "the compiled tiles waited for SMs held by another kernel"
    assert torch.equal(out, free)
    assert torch.count_nonzero(scratch[1]).item() == 0, "semaphores not reset"
    assert sum(redo[:, 0].tolist()) > 0
