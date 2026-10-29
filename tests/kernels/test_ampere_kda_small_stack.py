# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the small-stack RecoverSSM verify configs (VLLM_AMPERE_KDA_SMALL_STACK,
vllm_sm80/models/glm5next/ops/kda_decode.py).

CPU part: the switch, read once at the first launch; the block configs it
changes (nseq 7-8 at 16 heads, every RecoverSSM verify, the commit fold's
included) and the ones it leaves, pinned for nseq 1..8 with and without it;
the banner and the "set but off" lines; the verify's stack per thread compiled
for sm_80 (Triton's own toolchain, no GPU; skipped without it) with and
without the switch, today's verify and the fold's, and the 16-row config's
at every token tile.
GPU part (skipped without CUDA): with the switch against without, on the same
inputs, bitwise (outputs normed and SKIP_NORM, records, conv window, state
slot), today's verify and the fold's, for nseq 7-8 (the only batch sizes the
switch changes) and T 1..8; the 16-row
config against the 8-row one, today's verify and the fold's, nseq 7-8, T 1..8.

    python tests/kernels/test_ampere_kda_small_stack.py           # CPU part
    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_small_stack.py
"""

import os
import re
import subprocess
import sys
import tempfile
import types
import unittest

try:
    import pytest
except ImportError:
    pytest = None
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

H, D = 16, 128


def _fresh_switch(value):
    """Set the switch and forget the value read at the first launch."""
    from vllm_sm80.models.glm5next.ops import kda_decode as k2

    if value is None:
        os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    else:
        os.environ["VLLM_AMPERE_KDA_SMALL_STACK"] = value
    k2._SMALL_STACK = None
    return k2


def _launch_config(k2, nseq, recover, fold=False, T=4):
    """The (BV, num_warps) kda_decode_v2 launches with (its launch replaced),
    at max_query_len T (token tile BT = next_pow2(T))."""
    seen = []

    class _Grid:
        def __getitem__(self, grid):
            return lambda *a, **kw: seen.append((kw["BV"], kw["num_warps"]))

    M = nseq * T
    f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
    qkv = torch.zeros(M, 3 * H * D, dtype=bf16)
    records = tuple(torch.zeros(nseq, k2._WS_T, H, D, dtype=f32) for _ in range(3)) \
        if recover else None
    orig, k2._kda_decode_v2_kernel = k2._kda_decode_v2_kernel, _Grid()
    orig_ctr, k2._counter = k2._counter, lambda device, heads: (
        torch.zeros(2 * k2._CTR_SLOTS, dtype=i32), torch.zeros(1, dtype=f32))
    try:
        k2.kda_decode_v2(
            qkv, torch.zeros(1, M, H, dtype=bf16), torch.zeros(M, 128, dtype=bf16),
            torch.zeros(M, 128, dtype=bf16), torch.zeros(H * D, 128, dtype=bf16),
            torch.zeros(H * D, 128, dtype=bf16), torch.zeros(4, 3 * H * D, 10, dtype=bf16),
            torch.zeros(3 * H * D, 4, dtype=f32), None, torch.zeros(D, dtype=bf16),
            torch.zeros(4, H, D, D, dtype=f32), torch.zeros(nseq, dtype=i32),
            torch.zeros(nseq, 8, dtype=i32), torch.ones(nseq, dtype=i32),
            torch.arange(nseq + 1, dtype=i32) * T, T, torch.zeros(H, dtype=f32),
            torch.zeros(H * D, dtype=f32), out=torch.empty(1, M, H, D, dtype=bf16),
            records=records,
            fold=(torch.zeros(nseq, dtype=i32), torch.arange(nseq, dtype=i32)) if fold else None)
    finally:
        k2._kda_decode_v2_kernel, k2._counter = orig, orig_ctr
    return seen[0]


# _select at 16 heads, D 128, nseq 1..8: (BV, num_warps), without and with
# small_stack. Rows per warp / warps: nseq 1 8/1, nseq 2-3 8/2, nseq 4-6 16/2,
# nseq 7-8 32/1 (small_stack: 16/1). The select does not depend on the token tile.
SELECT_H16 = {
    False: {1: (8, 1), 2: (16, 2), 3: (16, 2), 4: (32, 2), 5: (32, 2), 6: (32, 2),
            7: (32, 1), 8: (32, 1)},
    True: {1: (8, 1), 2: (16, 2), 3: (16, 2), 4: (32, 2), 5: (32, 2), 6: (32, 2),
           7: (16, 1), 8: (16, 1)},
}


def test_select_pinned():
    """Every (small_stack, nseq) at 16 heads for nseq 1..8 gives exactly the
    pinned (BV, num_warps), from _select and from the launch at token tiles
    BT 1, 2, 4 and 8 (RecoverSSM verify for small_stack, the fold's included)."""
    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    try:
        for small in (False, True):
            k2 = _fresh_switch("1" if small else None)
            for n in range(1, 9):
                want = SELECT_H16[small][n]
                assert k2._select(n, H, D, small_stack=small) == want, (small, n)
                for T in (1, 2, 4, 8):
                    assert _launch_config(k2, n, True, T=T) == want, (small, n, T)
                    assert _launch_config(k2, n, True, fold=True, T=T) == want, (small, n, T)
                    assert _launch_config(k2, n, False, T=T) == SELECT_H16[False][n], (n, T)
    finally:
        _fresh_switch(saved)


def test_switch_read_once():
    """VLLM_AMPERE_KDA_SMALL_STACK is read once, at the first launch: set to 0
    afterwards, the RecoverSSM verify keeps the small-stack config. The configs
    with and without the switch: test_select_pinned."""
    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    try:
        k2 = _fresh_switch("1")
        assert _launch_config(k2, 8, True) == (16, 1)
        os.environ["VLLM_AMPERE_KDA_SMALL_STACK"] = "0"      # read once: stays on
        assert _launch_config(k2, 8, True) == (16, 1)
    finally:
        _fresh_switch(saved)


def test_banner_and_set_but_off():
    """The warmup names the switch when it is on and a verify config changes,
    and says why it is off when the fused verify is not used or no batch size
    up to max_num_seqs has a 32-row warp."""
    from vllm_sm80.models.glm5next.ops import kda_decode as k2

    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    saved_dk = os.environ.get("VLLM_AMPERE_DECODE_KERNELS")
    msgs = []
    orig_info, orig_once = k2.logger.info, k2.logger.info_once
    k2.logger.info_once = lambda msg, *a, **k: msgs.append(msg % a if a else msg)
    k2.logger.info = lambda *a, **k: None
    orig_warm, k2.warmup = k2.warmup, lambda *a, **k: None
    layer = types.SimpleNamespace(_conv_state_dim_first=True, _kda_v2=True, local_num_heads=16,
                                  head_dim=128, num_spec=7, _kda_step_tile=True,
                                  _kda_wide=False, _kda_recover=True)
    worker = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=8)))
    orig_find, k2.find_module = k2.find_module, lambda model, attr: model
    try:
        _fresh_switch("1")
        k2._warmup_kda_v2(worker, layer, "cpu", [8])
        worker.vllm_config.scheduler_config.max_num_seqs = 6
        k2._warmup_kda_v2(worker, layer, "cpu", [8])
        k2._warmup_kda_v2(worker, None, "cpu", [8])
        os.environ["VLLM_AMPERE_DECODE_KERNELS"] = "0"
        k2.warmup_from_worker(worker, [8])
    finally:
        k2.logger.info, k2.logger.info_once = orig_info, orig_once
        k2.warmup, k2.find_module = orig_warm, orig_find
        if saved_dk is None:
            os.environ.pop("VLLM_AMPERE_DECODE_KERNELS", None)
        else:
            os.environ["VLLM_AMPERE_DECODE_KERNELS"] = saved_dk
        _fresh_switch(saved)
    assert any("small-stack configs active (VLLM_AMPERE_KDA_SMALL_STACK=1)" in m
               for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_SMALL_STACK=1 set but off" in m
               and "no verify batch up to max_num_seqs 6" in m for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_SMALL_STACK=1 set but off" in m and "not used by this model" in m
               for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_SMALL_STACK=1 set but off" in m
               and "VLLM_AMPERE_DECODE_KERNELS is off" in m for m in msgs), msgs


def _skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise unittest.SkipTest(reason)


def _cuobjdump():
    import triton

    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin",
                        "cuobjdump")


def _need_sm80_toolchain():
    """Skip unless Triton can compile for sm_80 here (Gluon, ptxas, cuobjdump)."""
    try:
        from triton import knobs
        from triton.backends.compiler import GPUTarget  # noqa: F401
        from triton.experimental.gluon._runtime import GluonASTSource  # noqa: F401

        tools = (knobs.nvidia.ptxas.path, _cuobjdump())
    except Exception as e:  # noqa: BLE001
        _skip(f"no Triton sm_80 toolchain: {type(e).__name__}: {e}")
    for tool in tools:
        if not tool or not os.path.isfile(tool):
            _skip(f"no Triton sm_80 toolchain: {tool!r} missing")


def _verify_stack(nseq, T, skip, small_stack, fold):
    """Stack bytes per thread of the RecoverSSM verify compiled for sm_80 on
    the CPU, with the specialisation a served launch gets (pointers and the
    divisible strides 16-aligned, the unit strides constant)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource

    from vllm_sm80.models.glm5next.ops import kda_decode as k2

    KA, CK = 128, 4
    ptr = {"x": "*bf16", "fa": "*bf16", "ga": "*bf16", "wf": "*bf16", "wg": "*bf16",
           "conv_state": "*bf16", "conv_w": "*fp32", "beta": "*bf16", "norm_w": "*bf16",
           "out": "*bf16", "rec": "*fp32", "cu_seqlens": "*i32", "conv_idx": "*i32",
           "ssm_idx": "*i32", "num_acc": "*i32", "a_log": "*fp32", "g_bias": "*fp32",
           "ctr": "*i32", "gctr": "*i32", "ws": "*fp32", "rc": "*fp32", "rk": "*fp32",
           "re": "*fp32", "fl": "*i32", "rr": "*i32", "scale": "fp32", "eps": "fp32"}
    div16 = {"stride_x_t", "stride_fa_t", "stride_ga_t", "stride_wf", "stride_wg",
             "stride_beta_t", "stride_out_t", "stride_cs_slot", "stride_cs_tok",
             "stride_rec_slot"}
    fn = k2._kda_decode_v2_kernel
    bv, nw = k2._select(nseq, H, D, small_stack=small_stack)
    const = dict(conv_b=None, H=H, D=D, KA=KA, CONV_K=CK, PROJ=H * D, LOWER_BOUND=-5.0,
                 HAS_BIAS=False, L2_EPS=1e-6, BV=bv, NV=D // bv,
                 BS=triton.next_power_of_2(CK - 2 + T), BT=triton.next_power_of_2(T),
                 WS_T=8, DC=32, SW=8, NW=nw, RECOVER=True, SKIP_NORM=skip, ST_CACHE="",
                 FOLD=fold, stride_cs_chan=1, stride_sidx_tok=1)
    sig = {a: ("constexpr" if a in const else ptr.get(a, "i32")) for a in fn.arg_names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(fn.arg_names)
             if a not in const and (ptr.get(a, "").startswith("*") or a in div16)}
    k = triton.compile(GluonASTSource(fn, sig, const, attrs),
                       target=GPUTarget("cuda", 80, 32), options={"num_warps": nw})
    cu = _cuobjdump()
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        out = subprocess.run([cu, "--dump-resource-usage", f.name], capture_output=True,
                             text=True, check=True).stdout
    return int(re.search(r"STACK:(\d+)", out).group(1))


def test_verify_stack():
    """Compiled for sm_80, the largest-stack verify plans (nseq 7-8: T 5
    SKIP_NORM, T 7 normed), today's verify and the fold's, stay under the
    driver's default 1,024 B of stack per thread with the small-stack config,
    against more than 2,000 B without it."""
    _need_sm80_toolchain()
    for fold in (False, True):
        for nseq, T, skip in ((8, 5, True), (7, 7, False)):
            today = _verify_stack(nseq, T, skip, False, fold)
            small = _verify_stack(nseq, T, skip, True, fold)
            print(f"  {'fold' if fold else 'off'} n{nseq}_t{T}_{'skip' if skip else 'norm'}: "
                  f"stack {today} B -> {small} B")
            assert today > 2000 and small < 1024, (fold, nseq, T, skip, today, small)


def test_small_stack_config_stack():
    """The small-stack config nseq 7-8 take (16 rows, one warp), compiled for
    sm_80 at every token tile (T 1..8, normed and SKIP_NORM, today's verify and
    the fold's), stays within the driver's default 1,024 B of stack per thread,
    so its launch reserves nothing beyond the default. T enters the compile
    only through BT = next_pow2(T) and BS = next_pow2(T + 2): T 4, 6 and 8
    compile the kernels of T 3, 5 and 7."""
    _need_sm80_toolchain()
    from vllm_sm80.models.glm5next.ops import kda_decode as k2

    assert k2._select(8, H, D, small_stack=True) == k2._select(7, H, D, small_stack=True) \
        == (16, 1)
    worst = {}
    for fold in (False, True):
        for T in (1, 2, 3, 5, 7):
            for skip in (False, True):
                stack = _verify_stack(8, T, skip, True, fold)
                assert stack <= 1024, (fold, T, skip, stack)
                if stack >= worst.get(fold, (-1,))[0]:
                    worst[fold] = (stack, T, skip)
    for fold, (stack, T, skip) in sorted(worst.items()):
        print(f"  {'fold' if fold else 'off'}: max stack {stack} B at T {T}"
              f"{' SKIP_NORM' if skip else ''}")


# ------------------------------------------------------------------ GPU
def gpu_test_small_stack_bitwise():
    """Real kernel, one card: with the switch against without, on the same
    inputs, bitwise: outputs (normed and SKIP_NORM), records, the conv window
    and the state slot, for nseq 7-8 (nseq 1-6 launch the same config either
    way: test_select_pinned), T 1..8 and n accepted 1, T // 2, T.
    Today's verify: verify k -> commit(n) -> verify k + 1. The fold's: FOLD
    verify k (nothing owed) -> FOLD verify k + 1 owing n."""
    from tests.kernels.test_ampere_kda_recover import _context, _inputs
    from vllm_sm80.kernels.kda import recover as kr

    def v2(k2, p, conv, rec, ssm, acc, T, records, fold=None, skip=False):
        nseq = ssm.shape[0]
        out = torch.empty(1, nseq * T, H, D, device="cuda", dtype=torch.bfloat16)
        return k2.kda_decode_v2(p.qkv, p.beta, p.fa, p.ga, p.wf, p.wg, conv.transpose(-1, -2),
                                p.cw, None, p.nw, rec, ssm[:, 0], ssm, acc, p.qsl, 8, p.al,
                                p.gb, out=out, records=records, skip_norm=skip, fold=fold)

    def run(small, fold, nseq, T, n, skip):
        k2 = _fresh_switch("1" if small else None)
        k2.warmup(plans=((nseq, T),), recover=True, width=8, fold=fold)
        nslot = 2 + nseq * 8
        p = _inputs(nseq, T, 7000 + 10 * T + nseq, nslot)
        p2 = _inputs(nseq, T, 9000 + 10 * T + nseq, nslot)
        p.qsl = p2.qsl = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * T
        ssm = torch.tensor([[1 + 8 * s] for s in range(nseq)], device="cuda",
                           dtype=torch.int32)
        ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
        pool = torch.full((1, 3, 8, kr.WS_T, H, D), float("nan"), device="cuda")
        rec, conv = p.rec.clone(), p.conv.clone()
        recs = kr.layer_records(pool, 0)
        if fold:
            slots = torch.tensor([(3 * s + 5) % 8 for s in range(nseq)], device="cuda",
                                 dtype=torch.int32)
            zero = torch.zeros(nseq, device="cuda", dtype=torch.int32)
            v2(k2, p, conv, rec, ssm, ones, T, recs, fold=(zero, slots))
            owed = torch.full((nseq,), n, device="cuda", dtype=torch.int32)
            y = v2(k2, p2, conv, rec, ssm, ones, T, recs, fold=(owed, slots), skip=skip)
        else:
            v2(k2, p, conv, rec, ssm, ones, T, recs)
            ctx = _context(conv, rec, pool)
            ctx.commit(torch.full((nseq,), n, device="cuda", dtype=torch.int32),
                       ssm[:, 0].contiguous(), p.qsl)
            y = v2(k2, p2, conv, rec, ssm, ones, T, recs, skip=skip)
        torch.cuda.synchronize()
        rows = [1 + 8 * s for s in range(nseq)]
        return (y.view(torch.int16), pool.view(torch.int32), conv[rows].view(torch.int16),
                rec[rows].view(torch.int32))

    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    nfail = checks = 0
    try:
        for fold in (False, True):
            for nseq in (7, 8):
                for T in range(1, 9):
                    for n in sorted({1, max(1, T // 2), T}):
                        for skip in (False, True):
                            a = run(False, fold, nseq, T, n, skip)
                            b = run(True, fold, nseq, T, n, skip)
                            checks += 1
                            if not all(torch.equal(x, y) for x, y in zip(a, b)):
                                nfail += 1
                                print(f"  FAIL fold={fold} nseq={nseq} T={T} n={n} skip={skip}")
    finally:
        _fresh_switch(saved)
    print(f"  {checks - nfail}/{checks} bitwise")
    assert nfail == 0


# Every shape where the small-stack rule changed from 8 to 16 rows at 16 heads:
# nseq 7-8 (the only one-warp 32-row configs), every token tile, today's
# RecoverSSM verify and the fold's.
GRID_16_VS_8 = [(fold, nseq, T) for fold in (False, True) for nseq in (7, 8)
                for T in range(1, 9)]


def _bitwise_16_vs_8(fold, nseq, T):
    """Failures of the 16-row small-stack config against the 8-row one (switch
    on, one warp each) at (nseq, T), max_query_len T, width-8 state index
    rows, normed and SKIP_NORM, n accepted / owed in {1, T}. Today's verify:
    verify k -> commit(n) -> verify k + 1. The fold's: FOLD verify k (nothing
    owed) -> FOLD verify k + 1 owing n. After every step, bitwise (integer
    views): out, records c / k / e, the whole conv cache and state pool."""
    from tests.kernels.test_ampere_kda_recover import _context, _inputs
    from vllm_sm80.kernels.kda import recover as kr

    def snap(out, pool, conv, rec):
        torch.cuda.synchronize()
        s = {"c": pool[0, 0].view(torch.int32).clone(),
             "k": pool[0, 1].view(torch.int32).clone(),
             "e": pool[0, 2].view(torch.int32).clone(),
             "conv": conv.view(torch.int16).clone(), "state": rec.view(torch.int32).clone()}
        if out is not None:
            s["out"] = out.view(torch.int16).clone()
        return s

    def run(k2, rows, n, skip):
        k2.SMALL_STACK_ROWS_MAX = rows
        assert k2._select(nseq, H, D, small_stack=True) == (rows, 1)
        nslot = 2 + nseq * 8
        p = _inputs(nseq, T, 7100 + 10 * T + nseq, nslot)
        p2 = _inputs(nseq, T, 9100 + 10 * T + nseq, nslot)
        p.qsl = p2.qsl = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * T
        ssm = torch.tensor([[1 + 8 * s + t for t in range(8)] for s in range(nseq)],
                           device="cuda", dtype=torch.int32)
        ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
        pool = torch.full((1, 3, 8, kr.WS_T, H, D), float("nan"), device="cuda")
        rec, conv = p.rec.clone(), p.conv.clone()
        recs = kr.layer_records(pool, 0)
        slots = torch.tensor([(3 * s + 5) % 8 for s in range(nseq)], device="cuda",
                             dtype=torch.int32)

        def verify(q, owed):
            out = torch.empty(1, nseq * T, H, D, device="cuda", dtype=torch.bfloat16)
            k2.kda_decode_v2(q.qkv, q.beta, q.fa, q.ga, q.wf, q.wg, conv.transpose(-1, -2),
                             q.cw, None, q.nw, rec, ssm[:, 0], ssm, ones, q.qsl, T, q.al,
                             q.gb, out=out, records=recs, skip_norm=skip,
                             fold=None if owed is None else (
                                 torch.full((nseq,), owed, device="cuda", dtype=torch.int32),
                                 slots))
            return snap(out, pool, conv, rec)

        if fold:
            return [("verify owed 0", verify(p, 0)), (f"verify owed {n}", verify(p2, n))]
        steps = [("verify", verify(p, None))]
        _context(conv, rec, pool).commit(
            torch.full((nseq,), n, device="cuda", dtype=torch.int32),
            ssm[:, 0].contiguous(), p.qsl)
        steps.append((f"commit {n}", snap(None, pool, conv, rec)))
        steps.append(("verify", verify(p2, None)))
        return steps

    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    k2 = _fresh_switch("1")
    saved_rows = k2.SMALL_STACK_ROWS_MAX
    fails, checks = [], 0
    try:
        for n in sorted({1, T}):
            for skip in (False, True):
                a, b = run(k2, 8, n, skip), run(k2, 16, n, skip)
                for (step, sa), (_, sb) in zip(a, b):
                    checks += 1
                    bad = [key for key in sa if not torch.equal(sa[key], sb[key])]
                    if bad:
                        fails.append(f"fold={fold} nseq={nseq} T={T} n={n} skip={skip} "
                                     f"{step}: {bad}")
    finally:
        k2.SMALL_STACK_ROWS_MAX = saved_rows
        _fresh_switch(saved)
    return fails, checks


def gpu_test_16_rows_bitwise_8_rows():
    """Real kernel, one card: _bitwise_16_vs_8 over GRID_16_VS_8."""
    nfail = checks = 0
    for case in GRID_16_VS_8:
        fails, n = _bitwise_16_vs_8(*case)
        checks += n
        nfail += len(fails)
        for f in fails:
            print(f"  FAIL {f}")
    print(f"  {checks - nfail}/{checks} bitwise (16 rows vs 8 rows)")
    assert nfail == 0


CPU_TESTS = (test_select_pinned, test_switch_read_once, test_banner_and_set_but_off,
             test_verify_stack, test_small_stack_config_stack)
GPU_TESTS = (gpu_test_small_stack_bitwise, gpu_test_16_rows_bitwise_8_rows)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an sm_80 GPU")
    test_gpu_small_stack_bitwise = _needs_gpu(gpu_test_small_stack_bitwise)

    @_needs_gpu
    @pytest.mark.parametrize("fold,nseq,T", GRID_16_VS_8, ids=[
        f"{'fold' if f else 'nofold'}-n{n}-t{t}" for f, n, t in GRID_16_VS_8])
    def test_gpu_16_rows_bitwise_8_rows(fold, nseq, T):
        fails, checks = _bitwise_16_vs_8(fold, nseq, T)
        print(f"  {checks - len(fails)}/{checks} bitwise (16 rows vs 8 rows)")
        assert not fails, fails


def _main():
    import traceback

    tests = list(CPU_TESTS)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available():
        tests += list(GPU_TESTS)
    failed = skipped = 0
    skips = (unittest.SkipTest,) + ((pytest.skip.Exception,) if pytest is not None else ())
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except skips as e:
            skipped += 1
            print(f"SKIP {t.__name__}: {e}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
