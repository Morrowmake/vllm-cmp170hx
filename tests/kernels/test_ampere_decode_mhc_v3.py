# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sm_80 mHC decode v3 path (vllm/ampere_decode/mhc_decode_v3.py,
VLLM_GLM5_DECODE_MHC_V3, on by default; 0 is the kill switch).

CPU (runs anywhere):
    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_decode_mhc_v3.py
the flag and its kill switch; the v3 gate is exactly the v2 gate plus the flag;
the dispatch in `mhc_fused_post_pre_tilelang` reaches v3 when it is on and v2
(unchanged) when it is off; the blocked bf16 `fn` copy addresses the same
element as the plain layout for every (n, stream, hidden slice) kernel A reads;
the copy is read only for M <= 12; registration rules; warmup registers only
the module that runs; every kernel specialisation compiles for sm_80.
GPU (sm_80, skipped otherwise): error against an FP64 recomputation no worse
than v2's (summed mean <= 1.10x, max <= 1.25x on post_mix and comb_mix;
layer_input max within one bf16 ulp or 1.25x of v2's) with residual_cur bitwise
equal to v2's; with the registered copy every output is bitwise equal to the
unregistered fp32 run at every M (the skipped lo(fn) term is exactly zero);
bitwise run-to-run; CUDA-graph replay == eager with zero allocation growth.
Inputs are generated on the CPU (CUDA randn depends on the SM count).
"""

import logging

import pytest
import torch
from torch import nn

from vllm import envs
from vllm.ampere_decode import mhc_decode_v3 as v3

FLAG = "VLLM_GLM5_DECODE_MHC_V3"
HIDDEN, HC = 4096, 4
NOUT = HC * 2 + HC * HC
MS = (1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 32)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    import vllm.ampere_decode as ad

    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS", raising=False)
    monkeypatch.setattr(v3, "_FN_BF16", {})
    monkeypatch.setattr(ad, "_SM80_CACHE", True)


def _on(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", "1")


class _Layer(nn.Module):
    def __init__(self, fn_attn, fn_ffn):
        super().__init__()
        self.hc_attn_fn = nn.Parameter(fn_attn, requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(fn_ffn, requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.zeros(NOUT), requires_grad=False)


class _Model(nn.Module):
    def __init__(self, n, dev="cpu", exact=True, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def fn():
            t = torch.randn(NOUT, HC * HIDDEN, generator=g) * 0.02
            return (t.to(torch.bfloat16).float() if exact else t).to(dev)

        self.layers = nn.ModuleList(_Layer(fn(), fn()) for _ in range(n))


def _logged(fn, *args):
    lines = []

    class _H(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    h = _H()
    lg = logging.getLogger(v3.logger.name)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    try:
        return fn(*args), lines
    finally:
        lg.removeHandler(h)
        lg.setLevel(level)


# ------------------------------------------------------------------ CPU ----

def test_flag_defaults_on_and_kill_switch(monkeypatch):
    assert envs.VLLM_GLM5_DECODE_MHC_V3 is True
    monkeypatch.setenv(FLAG, "0")
    assert envs.VLLM_GLM5_DECODE_MHC_V3 is False


def test_gate_is_v2_gate_plus_flag(monkeypatch):
    from vllm.ampere_decode import use_ampere_mhc_decode_v2, use_ampere_mhc_decode_v3

    cases = [(m, hc, hid, nw) for m in (0, 1, 4, 12, 16, 32, 33)
             for hc in (3, 4) for hid in (2048, 4000, 4096) for nw in (None, object())]
    for kernels, v2flag in (("0", "1"), ("1", "0"), ("1", "1")):
        monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", kernels)
        monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", v2flag)
        for flag in ("1", "0"):
            monkeypatch.setenv(FLAG, flag)
            for m, hc, hid, nw in cases:
                want = flag == "1" and use_ampere_mhc_decode_v2(m, hc, hid, norm_weight=nw)
                assert use_ampere_mhc_decode_v3(m, hc, hid, norm_weight=nw) is want
    _on(monkeypatch)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS", "8")
    assert use_ampere_mhc_decode_v3(8, HC, HIDDEN) and not use_ampere_mhc_decode_v3(9, HC, HIDDEN)
    import vllm.ampere_decode as ad

    monkeypatch.setattr(ad, "_SM80_CACHE", False)
    assert not use_ampere_mhc_decode_v3(4, HC, HIDDEN)


def _cpu_kw(M):
    return dict(
        x=torch.zeros(M, HIDDEN, dtype=torch.bfloat16),
        residual=torch.zeros(M, HC, HIDDEN, dtype=torch.bfloat16),
        post_layer_mix=torch.zeros(M, HC, 1),
        comb_res_mix=torch.zeros(M, HC, HC),
        fn=torch.zeros(NOUT, HC * HIDDEN),
        hc_scale=torch.ones(3),
        hc_base=torch.zeros(NOUT),
        rms_eps=1e-5, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
        norm_weight=torch.ones(HIDDEN, dtype=torch.bfloat16), norm_eps=1e-5)


@pytest.mark.parametrize("flag,want", [("1", "v3"), ("0", "v2")])
def test_dispatch_picks_v3_then_v2(monkeypatch, flag, want):
    """Host dispatch only: the module entry points are replaced by spies."""
    from vllm.ampere_decode import mhc_decode_v2 as v2
    from vllm.model_executor.kernels.mhc import tilelang as tlmod

    seen = []
    monkeypatch.setattr(v3, "mhc_fused_post_pre", lambda *a, **k: seen.append("v3") or "v3")
    monkeypatch.setattr(v2, "mhc_fused_post_pre", lambda *a, **k: seen.append("v2") or "v2")
    _on(monkeypatch)
    monkeypatch.setenv(FLAG, flag)
    for M in (1, 4, 16, 32):
        assert tlmod.mhc_fused_post_pre_tilelang(**_cpu_kw(M)) == want
    assert seen == [want] * 4


def test_blocked_layout_addresses_the_same_elements():
    """Kernel A reads element (n, j, ks*HB + k) at n*FN + j*FJ + ks*FK + k; the
    blocked copy with its strides must hold the plain fn's value there."""
    hidden = 1024
    fn = (torch.randn(NOUT, HC * hidden) * 0.02).to(torch.bfloat16)
    blk = v3._block_fn(fn, HC)
    assert blk.is_contiguous() and blk.numel() == fn.numel()
    hb = v3.HB_GLU
    n = torch.arange(NOUT)[:, None, None, None]
    j = torch.arange(HC)[None, :, None, None]
    ks = torch.arange(hidden // hb)[None, None, :, None]
    k = torch.arange(hb)[None, None, None, :]
    for blocked, src in ((False, fn.reshape(-1)), (True, blk.reshape(-1))):
        fs_n, fs_j, fs_k = v3._fn_strides(blocked, HC, hidden, NOUT)
        got = src[n * fs_n + j * fs_j + ks * fs_k + k]
        want = fn.reshape(NOUT, HC, hidden // hb, hb)
        assert torch.equal(got, want), blocked
    # one CTA (ks) reads its four stream tiles as one contiguous run
    fs_n, fs_j, fs_k = v3._fn_strides(True, HC, hidden, NOUT)
    assert (fs_n, fs_j, fs_k) == (hb, NOUT * hb, HC * NOUT * hb)


def test_bf16_copy_only_up_to_twelve_rows():
    assert v3.FN_BF16_MAX_TOKENS == 12
    model = _Model(1)
    v3.register_fn_bf16(model)
    fn = model.layers[0].hc_attn_fn
    for M in range(1, 65):
        got, blocked = v3._select_fn(fn, M, HC, HIDDEN)
        if M <= 12:
            assert blocked and got is v3._fn_bf16(fn) and got.dtype == torch.bfloat16, M
        else:
            assert not blocked and got is fn, M
    assert v3._select_fn(fn, 4, 2, HIDDEN) == (fn, False)
    v3._FN_BF16.clear()
    assert v3._select_fn(fn, 4, HC, HIDDEN) == (fn, False)


def test_registers_blocked_exact_copies():
    model = _Model(2)
    n, lines = _logged(v3.register_fn_bf16, model)
    assert n == 4 and len(v3._FN_BF16) == 4
    assert any("mHC decode v3 reads exact blocked bf16 fn copies for M <= 12" in line
               for line in lines), lines
    for layer in model.layers:
        for p in (layer.hc_attn_fn, layer.hc_ffn_fn):
            bf = v3._fn_bf16(p)
            assert bf is not None and bf.dtype == torch.bfloat16
            assert torch.equal(bf, v3._block_fn(p.to(torch.bfloat16), HC))
    assert v3._fn_bf16(model.layers[0].hc_attn_base) is None


def test_inexact_fn_registers_none():
    model = _Model(2, exact=False)
    n, lines = _logged(v3.register_fn_bf16, model)
    assert n == 0 and v3._FN_BF16 == {}
    assert any("not bf16" in line for line in lines), lines


def test_fn_of_other_shape_is_not_registered():
    model = nn.Module()
    model.hc_attn_fn = nn.Parameter(torch.zeros(NOUT, 100), requires_grad=False)
    n, _ = _logged(v3.register_fn_bf16, model)
    assert n == 0 and v3._FN_BF16 == {}


@pytest.mark.parametrize("flag", ["1", "0"])
def test_warmup_registers_only_the_module_that_runs(monkeypatch, flag):
    from vllm.ampere_decode import mhc_decode_v2 as v2
    from vllm.ampere_decode import use_ampere_mhc_decode_v2, warmup as wu

    class _Op(nn.Module):
        hidden_size, n, mhc_sinkhorn_iterations = HIDDEN, HC, 20

    model = _Model(1)
    model.op = _Op()
    model.op.mhc_fused_post_pre_op = True
    calls = []
    monkeypatch.setattr(v2, "_FN_BF16", {})
    monkeypatch.setattr(v2, "warmup", lambda ms, **k: calls.append(("v2", tuple(ms))))
    monkeypatch.setattr(v3, "warmup", lambda ms, **k: calls.append(("v3", tuple(ms))))
    _on(monkeypatch)
    monkeypatch.setenv(FLAG, flag)
    wu._warmup_mhc_v2(model, "cpu", [1, 4, 8, 16, 32, 64], use_ampere_mhc_decode_v2)
    name = "v3" if flag == "1" else "v2"
    assert [c[0] for c in calls] == [name]
    assert calls[0][1][-1] == 32 and 64 not in calls[0][1]
    assert len(v3._FN_BF16) == (2 if flag == "1" else 0)
    assert len(v2._FN_BF16) == (0 if flag == "1" else 2)


def test_kernels_compile_for_sm80():
    """Every kernel A / finish specialisation the dispatch can pick (fp32 or
    bf16 fn, plain or blocked strides, EXACT on/off, ONEBAR on/off) compiles
    for sm_80 without a GPU."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource

    hb = v3.HB_GLU

    def build(kern, sig_types, consts):
        params = list(kern.arg_names)
        sig = {p: sig_types.get(p, "constexpr") for p in params}
        attrs = {(i,): [["tt.divisibility", 16]] for i, p in enumerate(params)
                 if sig[p].startswith("*")}
        c = triton.compile(GluonASTSource(kern, sig, constexprs=consts, attrs=attrs),
                           target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
        assert c.asm["cubin"]

    for fdt in ("fp32", "bf16"):
        types = {"x_ptr": "*bf16", "res_ptr": "*bf16", "post_ptr": "*fp32",
                 "comb_ptr": "*fp32", "fn_ptr": f"*{fdt}", "rc_ptr": "*bf16",
                 "mix_ptr": "*fp32", "sqr_ptr": "*fp32", "M": "i32"}
        for blocked in ((False, True) if fdt == "bf16" else (False,)):
            fs = v3._fn_strides(blocked, HC, HIDDEN, NOUT)
            for exact in ((True,) if fdt == "bf16" else (False, True)):
                strides = {"FN": fs[0], "FJ": fs[1], "FK": fs[2], "EXACT": exact}
                build(v3._glu_post_prenorm_tb_kernel, types,
                      dict(HIDDEN=HIDDEN, NOUT=NOUT, HB=hb, BT=8, NP2=32,
                           FV=4 if fdt == "fp32" else 8, **strides))
                build(v3._glu_post_prenorm_kernel, types,
                      dict(HIDDEN=HIDDEN, NOUT=NOUT, HB=hb, BM=16, NP2=32, NW=4, **strides))
    fin = {"mix_ptr": "*fp32", "sqr_ptr": "*fp32", "scale_ptr": "*fp32", "base_ptr": "*fp32",
           "rc_ptr": "*bf16", "nw_ptr": "*bf16", "post_out_ptr": "*fp32",
           "comb_out_ptr": "*fp32", "li_ptr": "*bf16", "M": "i32", "rms_eps": "fp32",
           "hc_pre_eps": "fp32", "hc_sink_eps": "fp32", "post_mult": "fp32",
           "norm_eps": "fp32"}
    ks = HIDDEN // hb
    for onebar in (True, False):
        build(v3._glu_finish_kernel, fin,
              dict(HIDDEN=HIDDEN, NOUT=NOUT, KS=ks, KSP2=max(triton.next_power_of_2(ks), 32),
                   SINK=20, ONEBAR=onebar))


# ------------------------------------------------------------------ GPU ----

def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 0)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")


def _case(M, fn, seed, outliers=False):
    from tests.kernels import test_ampere_decode_mhc_v2 as base

    c = base._case(M, seed)
    if outliers:
        g = torch.Generator().manual_seed(seed + 7)
        idx = torch.randint(0, HIDDEN, (16,), generator=g).cuda()
        res = c["residual"].float()
        res[:, :, idx] *= 1000.0
        c["residual"] = res.to(torch.bfloat16)
    c["fn"] = fn
    return c


@gpu
def test_gpu_accuracy_vs_v2_no_floor(monkeypatch):
    """v3 and v2 on identical inputs (production-like bf16-exact fn, registered in
    both), errors against FP64 summed over M x 2 input kinds x 3 seeds."""
    from tests.kernels import test_ampere_decode_mhc_v2 as base
    from vllm.ampere_decode import mhc_decode_v2 as v2

    monkeypatch.setattr(v2, "_FN_BF16", {})
    model = _Model(1, dev="cuda", seed=3)
    fn = model.layers[0].hc_attn_fn
    v2.register_fn_bf16(model)
    v3.register_fn_bf16(model)
    v2.warmup(MS)
    v3.warmup(MS)
    sums = {i: [0.0, 0.0, 0.0, 0.0] for i in (1, 2, 3)}
    ulps = [0.0, 0.0]
    for M in MS:
        for outl in (False, True):
            for seed in range(3):
                c = _case(M, fn, 1000 * M + seed, outl)
                kw = base._kw(c)
                a = v3.mhc_fused_post_pre(**kw)
                a2 = v3.mhc_fused_post_pre(**kw)
                b = v2.mhc_fused_post_pre(**kw)
                torch.cuda.synchronize()
                assert all(torch.equal(p, q) for p, q in zip(a, a2)), (M, "run-to-run")
                assert torch.equal(a[0], b[0]), (M, "residual_cur")
                ref = base._ref64(c)
                for i in (1, 2, 3):
                    g_max, g_mean = base._err(a[i], ref[i])
                    t_max, t_mean = base._err(b[i], ref[i])
                    s = sums[i]
                    s[0] += g_mean
                    s[1] += t_mean
                    s[2] = max(s[2], g_max)
                    s[3] = max(s[3], t_max)
                ulps[0] = max(ulps[0], base._bf16_ulps(a[3], ref[3]))
                ulps[1] = max(ulps[1], base._bf16_ulps(b[3], ref[3]))
    for i, name in ((1, "post_mix"), (2, "comb_mix"), (3, "layer_input")):
        gm, tm, gx, tx = sums[i]
        assert gm <= 1.10 * tm, (name, "mean", gm / tm)
        if i < 3:
            assert gx <= 1.25 * tx, (name, "max", gx / tx)
    assert ulps[0] <= max(1.25 * ulps[1], 1.0), ("layer_input", "max ulps", ulps)


@gpu
def test_gpu_registered_copy_is_bitwise_the_fp32_run():
    """EXACT and the blocked copy change no bit: registered (blocked bf16 at
    M <= 12, EXACT fp32 above) vs unregistered fp32 fn at every M."""
    from tests.kernels import test_ampere_decode_mhc_v2 as base

    model = _Model(1, dev="cuda", seed=5)
    fn = model.layers[0].hc_ffn_fn
    v3.warmup(MS)
    plain = {M: [t.clone() for t in v3.mhc_fused_post_pre(**base._kw(_case(M, fn, 40 + M)))]
             for M in MS}
    assert v3.register_fn_bf16(model) == 2
    v3.warmup(MS)
    for M in MS:
        got = v3.mhc_fused_post_pre(**base._kw(_case(M, fn, 40 + M)))
        for a, b in zip(got, plain[M]):
            assert torch.equal(a, b), M
        # a bf16 fn passed directly (plain layout, EXACT) gives the same bits
        if M <= 12:
            direct = v3.mhc_fused_post_pre(**base._kw(_case(M, fn.to(torch.bfloat16), 40 + M)))
            for a, b in zip(direct, plain[M]):
                assert torch.equal(a, b), M


@gpu
def test_gpu_dispatch_reaches_v3(monkeypatch):
    from tests.kernels import test_ampere_decode_mhc_v2 as base
    from vllm.model_executor.kernels.mhc import tilelang as tlmod

    model = _Model(1, dev="cuda", seed=7)
    fn = model.layers[0].hc_attn_fn
    v3.register_fn_bf16(model)
    v3.warmup((4, 16, 32))
    real, seen = v3.mhc_fused_post_pre, []

    def spy(*a, **k):
        seen.append(a[1].shape[0])
        return real(*a, **k)

    monkeypatch.setattr(v3, "mhc_fused_post_pre", spy)
    _on(monkeypatch)
    for M in (4, 16, 32, 33):
        kw = base._kw(_case(M, fn, 70 + M))
        out = tlmod.mhc_fused_post_pre_tilelang(**kw)
        if M <= 32:
            assert seen[-1] == M
            want = real(**kw)
            assert all(torch.equal(a, b) for a, b in zip(out, want)), M
        else:
            assert seen[-1:] != [M]


@gpu
@pytest.mark.parametrize("M", [1, 4, 8, 12, 13, 16, 24, 32])
def test_gpu_graph_replay(M):
    from tests.kernels import test_ampere_decode_mhc_v2 as base

    model = _Model(1, dev="cuda", seed=9)
    fn = model.layers[0].hc_attn_fn
    v3.register_fn_bf16(model)
    v3.warmup((M,))
    static = _case(M, fn, 40 + M)
    kw = base._kw(static)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        v3.mhc_fused_post_pre(**kw)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = v3.mhc_fused_post_pre(**kw)
    new = _case(M, fn, 90 + M)
    for k in static:
        if k != "fn":
            static[k].copy_(new[k])
    g.replay()
    torch.cuda.synchronize()
    want = v3.mhc_fused_post_pre(**base._kw(new))
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(outs, want)), M
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    assert (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) == (a0, r0)
