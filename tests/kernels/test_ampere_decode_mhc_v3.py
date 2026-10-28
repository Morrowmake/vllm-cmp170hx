# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sm_80 mHC decode v3 path (vllm_sm80/kernels/mhc/decode.py,
opted in with VLLM_GLM5_DECODE_MHC_V2).

CPU (runs anywhere):
    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_decode_mhc_v3.py
the gate (off by default, needs the master flag, 1 <= M <= 32 and its bound,
the hc == 4 / hidden % 1024 shapes, sm_80 only); the dispatch in
`vllm::sm80_mhc_fused_post_pre` (fused_op.py) reaches v3, and the plugin
replaces MHCFusedPostPreOp only with both switches; warmup covers M <= 32;
every kernel
specialisation compiles for sm_80.
GPU (sm_80, skipped otherwise): error against an FP64 recomputation no worse
than TileLang's (summed mean <= 1.2x, max <= 1.25x on post_mix and comb_mix;
layer_input max within one bf16 ulp or 1.25x of TileLang's) with residual_cur
bitwise equal to TileLang's; bitwise run-to-run; a closed gate never reaches
v3; CUDA-graph replay == eager with zero allocation growth.
Inputs are generated on the CPU (CUDA randn depends on the SM count).
"""

import pytest
import torch
from torch import nn

from tests.sm80.tuning_rows import use_rows
from vllm_sm80.kernels.mhc import decode as v3
from vllm_sm80.kernels.mhc import use_ampere_mhc_decode_v3

HIDDEN, HC = 4096, 4
NOUT = HC * 2 + HC * HC
MS = (1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 32)
SINK = 20
EPS = 1e-6
RMS_EPS = 1e-5
NORM_EPS = 1e-5
POST_MULT = 2.0


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    from vllm_sm80.kernels import _platform

    monkeypatch.delenv("VLLM_GLM5_DECODE_MHC_V2", raising=False)
    use_rows(monkeypatch)
    monkeypatch.setattr(_platform, "_SM80_CACHE", True)


def _on(monkeypatch):
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", "1")


def _gate(m, hc=HC, hidden=HIDDEN, **kw):
    return use_ampere_mhc_decode_v3(m, hc, hidden, **kw)


# ------------------------------------------------------------------ CPU ----

def test_gate_off_by_default(monkeypatch):
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    for m in (1, 4, 8, 16, 32):
        assert _gate(m) is False, m


def test_gate_needs_master_flag(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "0")
    assert _gate(4) is False


def test_gate_range_is_1_to_32(monkeypatch):
    _on(monkeypatch)
    for m in (1, 2, 3, 4, 8, 12, 16, 17, 24, 32):
        assert _gate(m) is True, m
    for m in (0, 33, 64, 1152):
        assert _gate(m) is False, m
    use_rows(
        monkeypatch,
        ("mhc", "decode_v2_max_rows", {"hidden": HIDDEN, "hc_mult": HC}, 8),
    )
    assert _gate(8) is True and _gate(9) is False
    # Keyed by the shape: another hidden size keeps the default 32.
    assert _gate(9, hidden=2048) is True


def test_gate_shape_conditions(monkeypatch):
    _on(monkeypatch)
    assert _gate(4, norm_weight=None) is False
    assert _gate(4, norm_weight=object()) is True
    assert _gate(4, hc=3) is False
    assert _gate(4, hc=8) is False
    assert _gate(4, hidden=4000) is False
    assert _gate(4, hidden=2048) is True


def test_gate_off_on_non_sm80(monkeypatch):
    from vllm_sm80.kernels import _platform

    _on(monkeypatch)
    monkeypatch.setattr(_platform, "_SM80_CACHE", False)
    assert _gate(4) is False


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


def test_dispatch_reaches_v3(monkeypatch):
    """Host dispatch only: the v3 entry point is replaced by a spy."""
    from vllm_sm80.kernels.mhc import fused_op

    seen = []
    monkeypatch.setattr(v3, "mhc_fused_post_pre",
                        lambda *a, **k: seen.append(a[1].shape[0]) or "v3")
    _on(monkeypatch)
    for M in (1, 4, 16, 32):
        assert fused_op.sm80_mhc_fused_post_pre(**_cpu_kw(M)) == "v3"
    assert seen == [1, 4, 16, 32]


@pytest.mark.parametrize(
    ("decode", "mhc_v2", "replaced"),
    [("1", "1", True), ("1", "0", False), ("0", "1", False)],
)
def test_plugin_replaces_the_op_only_with_both_switches(
    monkeypatch, decode, mhc_v2, replaced
):
    from vllm.model_executor import custom_op
    from vllm_sm80 import plugin
    from vllm_sm80.kernels.mhc import fused_op

    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", decode)
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", mhc_v2)
    plugin.register_mhc_decode_op()
    plugin.register_mhc_decode_op()  # idempotent
    registered = custom_op.op_registry_oot.get("MHCFusedPostPreOp")
    assert (registered is fused_op.Sm80MHCFusedPostPreOp) is replaced


def test_warmup_covers_the_bound(monkeypatch):
    from vllm_sm80.kernels.mhc import warmup as wu

    class _Op(nn.Module):
        hidden_size, n, mhc_sinkhorn_iterations = HIDDEN, HC, 20

    model = nn.Module()
    model.op = _Op()
    model.op.mhc_fused_post_pre_op = True
    calls = []
    monkeypatch.setattr(v3, "warmup", lambda ms, **k: calls.append(tuple(ms)))
    _on(monkeypatch)
    wu._warmup_mhc_v3(model, "cpu", [1, 4, 8, 16, 32, 64], use_ampere_mhc_decode_v3)
    assert len(calls) == 1
    assert calls[0][-1] == 32 and 64 not in calls[0]


def test_kernels_compile_for_sm80():
    """Every kernel A / finish specialisation the dispatch can pick (ONEBAR on
    and off) compiles for sm_80 without a GPU."""
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

    types = {"x_ptr": "*bf16", "res_ptr": "*bf16", "post_ptr": "*fp32",
             "comb_ptr": "*fp32", "fn_ptr": "*fp32", "rc_ptr": "*bf16",
             "mix_ptr": "*fp32", "sqr_ptr": "*fp32", "M": "i32"}
    # GLM's options (see decode.post_prenorm)
    glm = dict(HAS_POST=True, ROUND_MIX=False, FP64_ACC=False)
    build(v3._glu_post_prenorm_tb_kernel, types,
          dict(HIDDEN=HIDDEN, NOUT=NOUT, HB=hb, BT=8, NP2=32, **glm))
    build(v3._glu_post_prenorm_kernel, types,
          dict(HIDDEN=HIDDEN, NOUT=NOUT, HB=hb, BM=16, NP2=32, NW=4, **glm))
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


def _fn(seed):
    """A production-like fp32 `fn` whose values are all bf16 values (the
    checkpoint stores `fn` in bf16)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(NOUT, HC * HIDDEN, generator=g) * 0.02
    return t.to(torch.bfloat16).float().cuda()


def _inputs(M, seed, dev="cuda"):
    # Inputs come from the CPU generator: CUDA randn sizes its grid from the
    # SM count, so seeded CUDA inputs differ between GPUs (and SM overrides).
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, mul=1.0):
        return (torch.randn(*shape, generator=g) * mul).to(dev)

    return dict(
        x=rn(M, HIDDEN).to(torch.bfloat16),
        residual=rn(M, HC, HIDDEN).to(torch.bfloat16),
        post_layer_mix=torch.sigmoid(rn(M, HC, 1)) * POST_MULT,
        comb_res_mix=torch.softmax(rn(M, HC, HC), dim=-1),
        fn=rn(NOUT, HC * HIDDEN, mul=0.02),
        hc_scale=torch.ones(3, device=dev),
        hc_base=rn(NOUT, mul=0.1),
        norm_weight=(1.0 + rn(HIDDEN, mul=0.1)).to(torch.bfloat16),
    )


def _case(M, fn, seed, outliers=False):
    """`_inputs` with the given `fn`; `outliers` gives production-like residual
    magnitudes: most channels O(1), a few hidden positions ~1e3 larger (real
    residual streams reach ~5e3 there), where an accumulation of the 24 mixes
    that truncates loses precision."""
    c = _inputs(M, seed)
    if outliers:
        g = torch.Generator().manual_seed(seed + 7)
        idx = torch.randint(0, HIDDEN, (16,), generator=g).cuda()
        res = c["residual"].float()
        res[:, :, idx] *= 1000.0
        c["residual"] = res.to(torch.bfloat16)
    c["fn"] = fn
    return c


def _kw(c):
    return dict(c, rms_eps=RMS_EPS, hc_pre_eps=EPS, hc_sinkhorn_eps=EPS,
                hc_post_mult_value=POST_MULT, sinkhorn_repeat=SINK,
                norm_eps=NORM_EPS)


def _ref64(c):
    """FP64 op with the two bf16 rounding points (mixes from fp32 rc, collapse
    from bf16 rc; RMSNorm sumsq of fp32 o, scale on bf16(o))."""
    d = torch.float64
    x, res = c["x"].to(d), c["residual"].to(d)
    post = c["post_layer_mix"].reshape(-1, HC).to(d)
    comb, fn = c["comb_res_mix"].to(d), c["fn"].to(d)
    sc, base, nw = c["hc_scale"].to(d), c["hc_base"].to(d), c["norm_weight"].to(d)
    M = x.shape[0]
    rc = torch.einsum("mkj,mkh->mjh", comb, res) + post[:, :, None] * x[:, None, :]
    rcb = rc.to(torch.bfloat16)
    flat = rc.reshape(M, -1)
    mixes = (flat @ fn.t()) * torch.rsqrt(flat.square().sum(-1, keepdim=True)
                                          / flat.shape[1] + RMS_EPS)
    pre = torch.sigmoid(mixes[:, :HC] * sc[0] + base[:HC]) + EPS
    po = torch.sigmoid(mixes[:, HC:2 * HC] * sc[1] + base[HC:2 * HC]) * POST_MULT
    cm = mixes[:, 2 * HC:].reshape(M, HC, HC) * sc[2] + base[2 * HC:].reshape(1, HC, HC)
    cm = torch.softmax(cm, -1) + EPS
    cm = cm / (cm.sum(-2, keepdim=True) + EPS)
    for _ in range(SINK - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + EPS)
        cm = cm / (cm.sum(-2, keepdim=True) + EPS)
    o = (pre[:, :, None] * rcb.to(d)).sum(1)
    rn = torch.rsqrt(o.square().sum(-1, keepdim=True) / HIDDEN + NORM_EPS)
    li = (o.to(torch.bfloat16).to(d) * rn * nw).to(torch.bfloat16)
    return rcb, po.unsqueeze(-1), cm, li


def _err(a, b):
    d = (a.to(torch.float64) - b.to(torch.float64)).abs()
    return float(d.max()), float(d.mean())


def _bf16_ulps(a, ref):
    """Max |a - ref| in units of one bf16 ulp of ref (normal range)."""
    r = ref.to(torch.float64).abs().clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(r)) - 7)
    return float(((a.to(torch.float64) - ref.to(torch.float64)).abs() / ulp).max())


@gpu
def test_gpu_accuracy_vs_tilelang_no_floor(monkeypatch):
    """v3 and TileLang on identical inputs (production-like bf16-exact fn),
    errors against FP64 summed over M x 2 input kinds x 3 seeds with no ulp
    floor. layer_input is a bf16 output whose fp64 reference is itself rounded
    to bf16, so its max error is counted in bf16 ulps of the reference and may
    reach one ulp."""
    from vllm.model_executor.kernels.mhc import tilelang as tlmod

    fn = _fn(3)
    v3.warmup(MS)
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "0")
    sums = {i: [0.0, 0.0, 0.0, 0.0] for i in (1, 2, 3)}
    ulps = [0.0, 0.0]
    for M in MS:
        for outl in (False, True):
            for seed in range(3):
                c = _case(M, fn, 1000 * M + seed, outl)
                kw = _kw(c)
                a = v3.mhc_fused_post_pre(**kw)
                a2 = v3.mhc_fused_post_pre(**kw)
                b = tlmod.mhc_fused_post_pre_tilelang(**kw)
                torch.cuda.synchronize()
                assert all(torch.equal(p, q) for p, q in zip(a, a2)), (M, "run-to-run")
                assert torch.equal(a[0], b[0]), (M, "residual_cur")
                ref = _ref64(c)
                for i in (1, 2, 3):
                    g_max, g_mean = _err(a[i], ref[i])
                    t_max, t_mean = _err(b[i], ref[i])
                    s = sums[i]
                    s[0] += g_mean
                    s[1] += t_mean
                    s[2] = max(s[2], g_max)
                    s[3] = max(s[3], t_max)
                ulps[0] = max(ulps[0], _bf16_ulps(a[3], ref[3]))
                ulps[1] = max(ulps[1], _bf16_ulps(b[3], ref[3]))
    for i, name in ((1, "post_mix"), (2, "comb_mix"), (3, "layer_input")):
        gm, tm, gx, tx = sums[i]
        assert gm <= 1.2 * tm, (name, "mean", gm / tm)
        if i < 3:
            assert gx <= 1.25 * tx, (name, "max", gx / tx)
    assert ulps[0] <= max(1.25 * ulps[1], 1.0), ("layer_input", "max ulps", ulps)


@gpu
def test_gpu_dispatch_reaches_v3(monkeypatch):
    from vllm_sm80.kernels.mhc import fused_op

    fn = _fn(7)
    v3.warmup((4, 16, 32))
    real, seen = v3.mhc_fused_post_pre, []

    def spy(*a, **k):
        seen.append(a[1].shape[0])
        return real(*a, **k)

    monkeypatch.setattr(v3, "mhc_fused_post_pre", spy)
    _on(monkeypatch)
    for M in (4, 16, 32, 33):
        kw = _kw(_case(M, fn, 70 + M))
        out = fused_op.sm80_mhc_fused_post_pre(**kw)
        if M <= 32:
            assert seen[-1] == M
            want = real(**kw)
            assert all(torch.equal(a, b) for a, b in zip(out, want)), M
        else:
            assert seen[-1:] != [M]


@gpu
def test_gpu_closed_gate_never_reaches_v3(monkeypatch):
    from vllm_sm80.kernels.mhc import fused_op

    def boom(*a, **k):
        raise AssertionError("v3 called with VLLM_GLM5_DECODE_MHC_V2 off")

    monkeypatch.setattr(v3, "mhc_fused_post_pre", boom)
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", "0")
    for M in (4, 16):
        fused_op.sm80_mhc_fused_post_pre(**_kw(_inputs(M, seed=5)))
    torch.cuda.synchronize()


@gpu
@pytest.mark.parametrize("M", [1, 4, 8, 12, 13, 16, 24, 32])
def test_gpu_graph_replay(M):
    fn = _fn(9)
    v3.warmup((M,))
    static = _case(M, fn, 40 + M)
    kw = _kw(static)
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
    want = v3.mhc_fused_post_pre(**_kw(new))
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(outs, want)), M
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    assert (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) == (a0, r0)
