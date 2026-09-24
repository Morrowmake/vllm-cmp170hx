# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in sm_80 mHC decode v2 path (vllm/ampere_decode/mhc_decode_v2.py).

The gate tests are pure host logic and run anywhere:

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_decode_mhc_v2.py

The GPU tests (skipped without an sm_80 device) check, through the real
dispatch in `mhc_fused_post_pre_tilelang`: that V2 on routes M <= 32 to the v2
kernels and nothing else, that V2 off never touches them, the error against an
FP64 recomputation relative to TileLang's own (per case with a one-ulp floor, and
summed over many cases with NO floor, on random and on outlier-channel inputs,
with `residual_cur` bitwise equal to TileLang's), bitwise determinism, and CUDA
graph capture + replay with zero allocation growth.
"""

try:
    import pytest
except ImportError:  # the vllm-dev venv has no pytest; see __main__ below
    pytest = None
import torch

if pytest is None:
    class _Mark:
        def __getattr__(self, _name):
            return lambda *a, **k: (lambda f: f)

    class _Pytest:
        mark = _Mark()

        @staticmethod
        def fixture(f):
            return f

    pytest = _Pytest()

from vllm.ampere_decode import use_ampere_mhc_decode, use_ampere_mhc_decode_v2

HIDDEN = 4096
HC = 4
HC3 = HC * 2 + HC * HC
SINK = 20
EPS = 1e-6
RMS_EPS = 1e-5
NORM_EPS = 1e-5
POST_MULT = 2.0


class _Env:
    """Minimal monkeypatch stand-in that works with and without pytest."""

    def __init__(self):
        import os
        self._os = os
        self._saved = {}

    def setenv(self, k, v):
        self._saved.setdefault(k, self._os.environ.get(k))
        self._os.environ[k] = v

    def delenv(self, k):
        self._saved.setdefault(k, self._os.environ.get(k))
        self._os.environ.pop(k, None)

    def restore(self):
        for k, v in self._saved.items():
            if v is None:
                self._os.environ.pop(k, None)
            else:
                self._os.environ[k] = v
        self._saved.clear()


def _on(env, sm80=True):
    import vllm.ampere_decode as ad
    env.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    env.setenv("VLLM_GLM5_DECODE_MHC_V2", "1")
    env.delenv("VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS")
    env._saved_cache = ad._SM80_CACHE
    ad._SM80_CACHE = sm80


def _restore(env):
    import vllm.ampere_decode as ad
    if hasattr(env, "_saved_cache"):
        ad._SM80_CACHE = env._saved_cache
    env.restore()


def _gate(m, hc=HC, hidden=HIDDEN, **kw):
    return use_ampere_mhc_decode_v2(m, hc, hidden, **kw)


# ------------------------------------------------------------- CPU: gates ---

def test_v2_off_by_default():
    env = _Env()
    try:
        env.delenv("VLLM_GLM5_DECODE_MHC_V2")
        env.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
        import vllm.ampere_decode as ad
        saved, ad._SM80_CACHE = ad._SM80_CACHE, True
        try:
            for m in (1, 4, 8, 16, 32):
                assert _gate(m) is False, m
        finally:
            ad._SM80_CACHE = saved
    finally:
        env.restore()


def test_v2_needs_master_flag():
    env = _Env()
    try:
        _on(env)
        env.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
        assert _gate(4) is False
    finally:
        _restore(env)


def test_v2_range_is_1_to_32():
    env = _Env()
    try:
        _on(env)
        for m in (1, 2, 3, 4, 8, 12, 16, 17, 24, 32):
            assert _gate(m) is True, m
        for m in (0, 33, 64, 1152):
            assert _gate(m) is False, m
        env.setenv("VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS", "8")
        assert _gate(8) is True and _gate(16) is False
    finally:
        _restore(env)


def test_v2_independent_of_v1_flag():
    """V2 replaces v1 and TileLang in its range whatever VLLM_GLM5_DECODE_MHC says."""
    env = _Env()
    try:
        _on(env)
        env.setenv("VLLM_GLM5_DECODE_MHC", "0")
        assert _gate(4) is True
        # and v1's own gate is untouched by the v2 flag
        env.setenv("VLLM_GLM5_DECODE_MHC", "1")
        assert use_ampere_mhc_decode(4, HC, HIDDEN) is True
        assert use_ampere_mhc_decode(16, HC, HIDDEN) is False
    finally:
        _restore(env)


def test_v2_shape_conditions():
    env = _Env()
    try:
        _on(env)
        assert _gate(4, norm_weight=None) is False
        assert _gate(4, norm_weight=object()) is True
        assert _gate(4, hc=3) is False
        assert _gate(4, hc=8) is False
        assert _gate(4, hidden=4000) is False
        assert _gate(4, hidden=2048) is True
    finally:
        _restore(env)


def test_v2_off_on_non_sm80():
    env = _Env()
    try:
        _on(env, sm80=False)
        assert _gate(4) is False
    finally:
        _restore(env)


# ------------------------------------------------------------- GPU tests ----

def _gpu_ok():
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (8, 0))


def _case(M, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)

    def rn(*shape, mul=1.0):
        return torch.randn(*shape, generator=g, device=dev) * mul

    return dict(
        x=rn(M, HIDDEN).to(torch.bfloat16),
        residual=rn(M, HC, HIDDEN).to(torch.bfloat16),
        post_layer_mix=torch.sigmoid(rn(M, HC, 1)) * POST_MULT,
        comb_res_mix=torch.softmax(rn(M, HC, HC), dim=-1),
        fn=rn(HC3, HC * HIDDEN, mul=0.02),
        hc_scale=torch.ones(3, device=dev),
        hc_base=rn(HC3, mul=0.1),
        norm_weight=(1.0 + rn(HIDDEN, mul=0.1)).to(torch.bfloat16),
    )


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


def _band(t, bits):
    return max(float(t.abs().float().max()), 1.0) * 2.0 ** -bits


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")
def test_gpu_dispatch_and_accuracy():
    from vllm.ampere_decode import mhc_decode_v2
    from vllm.model_executor.kernels.mhc import tilelang as tlmod
    env = _Env()
    calls = []
    real = mhc_decode_v2.mhc_fused_post_pre

    def spy(*a, **k):
        calls.append(a[1].shape[0])
        return real(*a, **k)

    try:
        mhc_decode_v2.warmup((1, 3, 4, 8, 16, 17, 32))
        for M in (1, 3, 4, 8, 16, 17, 32, 33):
            c = _case(M, seed=11 + M)
            env.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
            tl_out = tlmod.mhc_fused_post_pre_tilelang(**_kw(c))
            _on(env)
            mhc_decode_v2.mhc_fused_post_pre = spy
            got = tlmod.mhc_fused_post_pre_tilelang(**_kw(c))
            again = tlmod.mhc_fused_post_pre_tilelang(**_kw(c))
            mhc_decode_v2.mhc_fused_post_pre = real
            _restore(env)
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(got, again)), M
            if M > 32:
                assert calls[-1:] != [M], "M=33 must not reach v2"
                continue
            assert calls[-1] == M
            ref = _ref64(c)
            for i, bits in enumerate((8, 20, 20, 8)):
                g_max, g_mean = _err(got[i], ref[i])
                t_max, t_mean = _err(tl_out[i], ref[i])
                band = _band(ref[i], bits)
                assert g_max <= max(band, 1.5 * t_max), (M, i, g_max, t_max)
                assert g_mean <= max(band, 1.5 * t_mean), (M, i, g_mean, t_mean)
                assert got[i].shape == tl_out[i].shape
                assert got[i].dtype == tl_out[i].dtype
    finally:
        mhc_decode_v2.mhc_fused_post_pre = real
        _restore(env)


def _case_outliers(M, seed, dev="cuda"):
    """Production-like residual magnitudes: most channels O(1), a few hidden
    positions ~1e3 larger (real residual streams reach ~5e3 there).  This is
    where the accumulation of the 24 mixes loses precision if it truncates."""
    c = _case(M, seed, dev)
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    idx = torch.randint(0, HIDDEN, (16,), generator=g, device=dev)
    res = c["residual"].float()
    res[:, :, idx] *= 1000.0
    c["residual"] = res.to(torch.bfloat16)
    return c


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")
def test_gpu_accuracy_vs_tilelang_no_floor():
    """Error against FP64 no worse than TileLang's, with no ulp floor: summed over
    7 token counts x 2 input kinds x 3 seeds, mean error <= 1.2x TileLang's and
    max error <= 1.25x TileLang's on post_mix, comb_mix and layer_input, and
    residual_cur bitwise equal to TileLang's.  (A per-case one-ulp floor let a
    2.8x mean-error regression on the fp32 mixes through.)"""
    from vllm.ampere_decode import mhc_decode_v2
    from vllm.model_executor.kernels.mhc import tilelang as tlmod
    env = _Env()
    ms = (1, 4, 8, 16, 17, 25, 32)
    sums = {i: [0.0, 0.0, 0.0, 0.0] for i in (1, 2, 3)}  # got mean, tl mean, got max, tl max
    try:
        mhc_decode_v2.warmup(ms)
        for M in ms:
            for kind, make in (("randn", _case), ("outliers", _case_outliers)):
                for seed in range(3):
                    c = make(M, seed=1000 * M + seed)
                    env.setenv("VLLM_GLM5_DECODE_KERNELS", "0")
                    tl_out = tlmod.mhc_fused_post_pre_tilelang(**_kw(c))
                    _restore(env)
                    got = mhc_decode_v2.mhc_fused_post_pre(**_kw(c))
                    torch.cuda.synchronize()
                    assert torch.equal(got[0], tl_out[0]), (M, kind, seed, "residual_cur")
                    ref = _ref64(c)
                    for i in (1, 2, 3):
                        g_max, g_mean = _err(got[i], ref[i])
                        t_max, t_mean = _err(tl_out[i], ref[i])
                        s = sums[i]
                        s[0] += g_mean
                        s[1] += t_mean
                        s[2] = max(s[2], g_max)
                        s[3] = max(s[3], t_max)
    finally:
        _restore(env)
    for i, name in ((1, "post_mix"), (2, "comb_mix"), (3, "layer_input")):
        gm, tm, gx, tx = sums[i]
        assert gm <= 1.2 * tm, (name, "mean", gm / tm)
        assert gx <= 1.25 * tx, (name, "max", gx / tx)


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")
def test_gpu_off_path_never_touches_v2():
    from vllm.ampere_decode import mhc_decode_v2
    from vllm.model_executor.kernels.mhc import tilelang as tlmod
    env = _Env()
    real = mhc_decode_v2.mhc_fused_post_pre

    def boom(*a, **k):
        raise AssertionError("v2 called with the flag off")

    try:
        mhc_decode_v2.mhc_fused_post_pre = boom
        env.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
        env.setenv("VLLM_GLM5_DECODE_MHC_V2", "0")
        for M in (4, 16):
            tlmod.mhc_fused_post_pre_tilelang(**_kw(_case(M, seed=5)))
        torch.cuda.synchronize()
    finally:
        mhc_decode_v2.mhc_fused_post_pre = real
        env.restore()


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")
def test_gpu_graph_capture():
    from vllm.ampere_decode import mhc_decode_v2
    for M in (1, 4, 16, 32):
        mhc_decode_v2.warmup((M,))
        static = _case(M, seed=40 + M)
        kw = _kw(static)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mhc_decode_v2.mhc_fused_post_pre(**kw)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = mhc_decode_v2.mhc_fused_post_pre(**kw)
        new = _case(M, seed=90 + M)
        for k in static:
            static[k].copy_(new[k])
        g.replay()
        torch.cuda.synchronize()
        want = mhc_decode_v2.mhc_fused_post_pre(**_kw(new))
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(outs, want)), M
        a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        assert (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) == (a0, r0)
        del g


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        if t.__name__.startswith("test_gpu") and not _gpu_ok():
            print(f"SKIP {t.__name__}")
            continue
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    sys.exit(1 if failed else 0)
