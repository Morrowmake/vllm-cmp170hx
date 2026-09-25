# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 KDA chunked prefill with 64 heads per card (VLLM_GLM5_PP_KDA_PREFILL).

CPU tests (run with ``CUDA_VISIBLE_DEVICES=""``): the flag is declared and off
by default, the layer and call gates and their fallbacks, and the warm-up
hook is a no-op with the flag unset.

GPU tests (skip without an sm_80 device): with the gate closed the upstream
chunk path runs unchanged (bitwise); with it open, error against an exact
fp64 token-sequential recomputation is at most the upstream path's (mean
<= 1.10x, max <= 1.25x per output; on o, elements that are one of the two
bf16 neighbours of the exact value may be set aside for the max, at most
0.1 % of them), on production-magnitude random inputs, awkward shapes and,
when ``KDA_PRE_CAPTURE_DIR`` points at a directory of captured live prefill
calls (``p<pp>_L<layer>_b<T>_<n>.pt``), on real inputs against the live
outputs; run-to-run determinism; CUDA-graph replay bitwise equal to eager
with no allocation growth.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_pp_kda_prefill.py
    KDA_PRE_CAPTURE_DIR=<dir> pytest -q tests/kernels/test_ampere_pp_kda_prefill.py
"""

import glob
import itertools
import os

import pytest
import torch

from vllm.ampere_prefill import kda_prefill as kp

FLAG = "VLLM_GLM5_PP_KDA_PREFILL"
H, D = kp.HEADS, kp.HEAD_DIM
P = H * D
LB = kp.LOWER_BOUND
L2_EPS = 1e-6
MEAN_RATIO, MAX_RATIO, TIE_CAP = 1.10, 1.25, 1e-3
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")
BF16, F32 = torch.bfloat16, torch.float32


# ------------------------------------------------------------------ CPU: flag
def test_flag_declared_default_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(FLAG, raising=False)
    assert envs.environment_variables[FLAG]() is False
    monkeypatch.setenv(FLAG, "1")
    assert envs.environment_variables[FLAG]() is True
    monkeypatch.setenv(FLAG, "0")
    assert envs.environment_variables[FLAG]() is False
    src = open(envs.__file__).read()
    assert f"    {FLAG}: bool = False\n" in src


# ------------------------------------------------------------------ CPU: gates
OPEN_LAYER = dict(backend="triton", num_heads=64, head_dim=128, dtype=BF16,
                  safe_gate=True, lower_bound=-5.0, capability=(8, 0))


def test_layer_gate_open():
    assert kp.layer_closed_reason(**OPEN_LAYER) == ""


@pytest.mark.parametrize("change,needle", [
    (dict(capability=(9, 0)), "sm_80"),
    (dict(capability=(8, 6)), "sm_80"),
    (dict(capability=None), "sm_80"),
    (dict(backend="flashkda"), "backend"),
    (dict(num_heads=16), "16 heads"),
    (dict(num_heads=32), "32 heads"),
    (dict(head_dim=64), "head_dim"),
    (dict(dtype=torch.float16), "dtype"),
    (dict(dtype=F32), "dtype"),
    (dict(safe_gate=False), "safe"),
    (dict(lower_bound=-3.0), "lower_bound"),
    (dict(lower_bound=None), "lower_bound"),
])
def test_layer_gate_closed(change, needle):
    why = kp.layer_closed_reason(**{**OPEN_LAYER, **change})
    assert why and needle in why


OPEN_CALL = dict(num_tokens=2304, num_seqs=1, qkv_dtype=BF16, g_dtype=BF16,
                 state_dtype=F32, a_log_dtype=F32, bias_dtype=F32)


@pytest.mark.parametrize("change", [
    {}, dict(num_tokens=1), dict(num_tokens=2312), dict(num_tokens=1282),
    dict(num_seqs=16), dict(num_tokens=2312, num_seqs=16),
])
def test_call_gate_open(change):
    assert kp.call_closed_reason(**{**OPEN_CALL, **change}) == ""


@pytest.mark.parametrize("change", [
    dict(num_tokens=0), dict(num_tokens=2313), dict(num_tokens=3456),
    dict(num_seqs=0), dict(num_seqs=17),
    dict(qkv_dtype=torch.float16), dict(g_dtype=F32),
    dict(state_dtype=BF16), dict(a_log_dtype=BF16), dict(bias_dtype=None),
])
def test_call_gate_closed(change):
    assert kp.call_closed_reason(**{**OPEN_CALL, **change}) != ""


def test_select_chunk_fn_fallback_and_open():
    def upstream(**kw):
        raise AssertionError("not called")

    args = list(OPEN_CALL.values())
    assert kp.select_chunk_fn(upstream, *args) is kp.chunk_kda_with_fused_gate
    closed = dict(OPEN_CALL, num_tokens=4000)
    assert kp.select_chunk_fn(upstream, *closed.values()) is upstream
    closed = dict(OPEN_CALL, num_seqs=17)
    assert kp.select_chunk_fn(upstream, *closed.values()) is upstream


def test_use_for_layer_closed_on_cpu_or_wrong_heads(monkeypatch):
    from vllm.platforms import current_platform

    class Cap:
        major, minor = 8, 0

    monkeypatch.setattr(current_platform, "get_device_capability", lambda *a, **k: Cap)
    assert kp.use_for_layer("triton", 64, 128, BF16, True, -5.0) is True
    assert kp.use_for_layer("triton", 16, 128, BF16, True, -5.0) is False
    assert kp.use_for_layer("flashkda", 64, 128, BF16, True, -5.0) is False
    monkeypatch.setattr(current_platform, "get_device_capability", lambda *a, **k: None)
    assert kp.use_for_layer("triton", 64, 128, BF16, True, -5.0) is False


def test_warmup_hook_noop_without_flag(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)

    class Worker:
        def get_model(self):
            raise AssertionError("must not touch the model with the flag off")

    kp.warmup_from_worker(Worker())


def test_warmup_hook_noop_without_open_layer(monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    class Layer(torch.nn.Module):
        _pp_kda_prefill = False

    class Worker:
        device = "cpu"

        def get_model(self):
            return torch.nn.Sequential(Layer())

    called = []
    monkeypatch.setattr(kp, "warmup", lambda *a, **k: called.append(1))
    kp.warmup_from_worker(Worker())
    assert not called


# ------------------------------------------------------------------ GPU helpers
def upstream_fn():
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
    return chunk_kda_with_fused_gate


def cu_from(seqlens, device="cuda"):
    return torch.tensor([0] + list(itertools.accumulate(seqlens)),
                        dtype=torch.int32, device=device)


def make_case(seqlens, init="real", seed=0, layout="production", qkv_scale=1.0,
              gate_shift=0.0, beta_shift=0.0, device="cuda"):
    """Random inputs with production magnitudes (per-channel pre-SiLU conv
    statistics, gate projection, beta logits, A_log, dt_bias and per-head
    state rms of the 64-head layers) in the production layout: q, k, v are
    column views of one [T, 3 * H * D] conv-output buffer."""
    g = torch.Generator().manual_seed(1000 + seed)
    T = int(sum(seqlens))
    N = len(seqlens)
    std = torch.exp(torch.randn(3 * P, generator=g) * 0.8 + torch.log(torch.tensor(0.015)))
    std = std.clamp(0.005, 0.93)
    std[2 * P:] *= 2.3                                          # v runs ~2x q, k
    mean = torch.randn(3 * P, generator=g) * 0.0026
    pre = (mean + std * torch.randn(T, 3 * P, generator=g)) * qkv_scale
    buf = (pre * torch.sigmoid(pre)).to(BF16).to(device)
    if layout == "production":
        q, k, v = (buf[:, i * P:(i + 1) * P].view(1, T, H, D) for i in range(3))
    else:
        q, k, v = (buf[:, i * P:(i + 1) * P].contiguous().view(1, T, H, D)
                   for i in range(3))
    gm = torch.randn(P, generator=g) * 0.044 + 0.0215
    gs = (torch.randn(P, generator=g) * 0.075 + 0.384).clamp(0.2, 0.7)
    raw_g = (gm + gs * torch.randn(T, P, generator=g) + gate_shift).to(BF16)
    raw_g = raw_g.to(device).view(1, T, H, D)
    bm = torch.randn(H, generator=g) * 0.16 + 1.31
    bs = torch.randn(H, generator=g) * 0.17 + 1.53
    beta_raw = (bm + bs * torch.randn(T, H, generator=g) + beta_shift).to(BF16)
    beta = beta_raw.float().sigmoid().to(device).unsqueeze(0)
    A_log = (torch.randn(H, generator=g) * 0.45 + 1.69).view(1, 1, H, 1).to(device)
    dt_bias = (torch.randn(P, generator=g) * 1.13 - 1.0).to(device)
    rms = torch.exp(torch.randn(H, generator=g) * 0.6 + torch.log(torch.tensor(0.008)))
    h0 = torch.zeros(N, H, D, D, dtype=F32)
    for n in range(N):
        mode = init if init != "mixed" else ("zero" if n % 2 == 0 else "real")
        if mode == "real":
            h0[n] = torch.randn(H, D, D, generator=g) * rms.view(H, 1, 1)
    return dict(q=q, k=k, v=v, raw_g=raw_g, beta=beta, A_log=A_log, g_bias=dt_bias,
                initial_state=h0.to(device), cu_seqlens=cu_from(seqlens, device),
                buf=buf)


def call(fn, c, initial_state="case"):
    # upstream writes o into v.contiguous(): with a contiguous v that IS v
    v = c["v"].clone() if c["v"].is_contiguous() else c["v"]
    h0 = c["initial_state"] if initial_state == "case" else initial_state
    return fn(q=c["q"], k=c["k"], v=v, raw_g=c["raw_g"], beta=c["beta"],
              A_log=c["A_log"], g_bias=c["g_bias"], initial_state=h0,
              output_final_state=True, use_qk_l2norm_in_kernel=True,
              cu_seqlens=c["cu_seqlens"], safe_gate=True, lower_bound=LB)


def reference_fp64(c):
    """Exact float64 token-sequential recomputation: (o [1, T, H, D], S [N, H, V, K])."""
    q = c["q"][0].double()
    k = c["k"][0].double()
    v = c["v"][0].double()
    beta = c["beta"][0].double()
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * D ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
    a = torch.exp(c["A_log"].double().reshape(-1))[:, None]
    x = c["raw_g"][0].double() + c["g_bias"].double().view(H, D)
    eg = torch.exp(LB / (1.0 + torch.exp(-(a * x))))
    cu = c["cu_seqlens"].tolist()
    o = torch.empty(cu[-1], H, D, dtype=torch.float64, device=q.device)
    fin = torch.empty(len(cu) - 1, H, D, D, dtype=torch.float64, device=q.device)
    for n in range(len(cu) - 1):
        S = c["initial_state"][n].double().clone()
        for t in range(cu[n], cu[n + 1]):
            S.mul_(eg[t][:, None, :])
            u = (v[t] - torch.bmm(S, k[t][:, :, None])[..., 0]) * beta[t][:, None]
            S.add_(u[:, :, None] * k[t][:, None, :])
            o[t] = torch.bmm(S, q[t][:, :, None])[..., 0]
        fin[n] = S
    return o.unsqueeze(0), fin


def errors(y, r):
    e = (y.double() - r).abs()
    return (float(e.max()), float(e.mean())) if e.numel() else (0.0, 0.0)


def tie_adjusted_max(y, r):
    """Max error with rounding-limited elements (y is one of the two bf16
    neighbours of the exact value) set aside, and how many of those are above it."""
    yd = y.double()
    e = (yd - r).abs()
    n = r.to(BF16)
    nf = n.double()
    bits = n.view(torch.int16)
    up, dn = r > nf, r < nf
    step_up = torch.where(nf >= 0, bits + 1, bits - 1).view(BF16).double()
    step_dn = torch.where(nf > 0, bits - 1, bits + 1).view(BF16).double()
    other = torch.where(up, step_up, torch.where(dn, step_dn, nf))
    lim = (yd == torch.minimum(nf, other)) | (yd == torch.maximum(nf, other))
    rest = e[~lim]
    m = float(rest.max()) if rest.numel() else 0.0
    return m, int((e[lim] > m).sum())


def ratio(a, b):
    return (0.0 if a == 0 else float("inf")) if b == 0 else a / b


def check_accuracy(new, inc, ref):
    """Per output: (mean ratio, max ratio, ok) vs the incumbent's own error."""
    out = {}
    for name, y, yi, r in (("o", new[0], inc[0], ref[0]), ("state", new[1], inc[1], ref[1])):
        cmx, cmn = errors(y, r)
        imx, imn = errors(yi, r)
        rm, rx = ratio(cmn, imn), ratio(cmx, imx)
        ok_max = rx <= MAX_RATIO
        if name == "o" and not ok_max:
            cadj, n_above = tie_adjusted_max(y, r)
            iadj, _ = tie_adjusted_max(yi, r)
            ok_max = (ratio(cadj, iadj) <= MAX_RATIO
                      and n_above <= max(1, int(TIE_CAP * y.numel())))
        out[name] = (rm, rx, rm <= MEAN_RATIO and ok_max)
    return out


@pytest.fixture(scope="module")
def warmed():
    if not IS_SM80:
        pytest.skip("needs an sm_80 GPU")
    kp.warmup(kp.WARMUP_PLANS)
    return True


# ------------------------------------------------------------------ GPU: off path
@needs_sm80
@pytest.mark.parametrize("seqlens,init", [([2304], "real"), ([2304], "zero"),
                                          ([1282], "real"), ([576] * 4, "mixed")])
def test_gate_closed_is_upstream_bitwise(seqlens, init):
    up = upstream_fn()
    c = make_case(seqlens, init, seed=7)
    ref_out = call(up, c)
    # 2313 tokens: the call gate closes and hands back the upstream function
    fn = kp.select_chunk_fn(up, 2313, len(seqlens), BF16, BF16, F32, F32, F32)
    assert fn is up
    out = call(fn, c)
    torch.cuda.synchronize()
    assert torch.equal(out[0], ref_out[0]) and torch.equal(out[1], ref_out[1])


# ------------------------------------------------------------------ GPU: on path
ACCURACY_CASES = [
    ("cont2304", [2304], "real", {}),
    ("first2304", [2304], "zero", {}),
    ("tail1282", [1282], "real", {}),
    ("t2312", [2312], "real", {}),
    ("varlen4", [576] * 4, "mixed", {}),
    ("varlen2_unaligned", [1000, 1304], "mixed", {}),
    ("varlen16", [144] * 16, "mixed", {}),
    ("t1", [1], "real", {}),
    ("t63", [63], "zero", {}),
    ("t64", [64], "real", {}),
    ("t65", [65], "zero", {}),
    ("t127", [127], "real", {}),
    ("varlen_short12", [1, 2, 3, 5, 8, 13, 21, 34, 55, 64, 65, 127], "mixed", {}),
    ("contiguous_qkv", [700], "real", dict(layout="contiguous")),
    ("large_qkv_x8", [512], "real", dict(qkv_scale=8.0)),
    ("gate_saturated", [512], "real", dict(gate_shift=12.0)),
    ("gate_open", [512], "real", dict(gate_shift=-12.0)),
    ("beta_near0", [256], "real", dict(beta_shift=-10.0)),
    ("beta_near1", [256], "real", dict(beta_shift=10.0)),
]


@needs_sm80
@pytest.mark.parametrize("name,seqlens,init,kw", ACCURACY_CASES,
                         ids=[c[0] for c in ACCURACY_CASES])
def test_accuracy_vs_fp64_relative_to_upstream(warmed, name, seqlens, init, kw):
    seed = [c[0] for c in ACCURACY_CASES].index(name)
    c = make_case(seqlens, init, seed=seed, **kw)
    keep = {k: c[k].clone() for k in ("buf", "raw_g", "beta", "initial_state",
                                      "A_log", "g_bias", "cu_seqlens")}
    inc = call(upstream_fn(), c)
    new = call(kp.chunk_kda_with_fused_gate, c)
    new2 = call(kp.chunk_kda_with_fused_gate, c)
    torch.cuda.synchronize()
    T, N = sum(seqlens), len(seqlens)
    assert new[0].shape == (1, T, H, D) and new[0].dtype == BF16
    assert new[1].shape == (N, H, D, D) and new[1].dtype == F32
    assert bool(torch.isfinite(new[0]).all() and torch.isfinite(new[1]).all())
    assert torch.equal(new[0], new2[0]) and torch.equal(new[1], new2[1]), "not deterministic"
    for k_, v_ in keep.items():
        assert torch.equal(c[k_], v_), f"input {k_} modified"
    res = check_accuracy(new, inc, reference_fp64(c))
    assert all(r[2] for r in res.values()), f"{name}: {res}"


@needs_sm80
def test_initial_state_none_equals_zeros(warmed):
    c = make_case([300], "zero", seed=41)
    a = call(kp.chunk_kda_with_fused_gate, c, initial_state=None)
    b = call(kp.chunk_kda_with_fused_gate, c)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _capture_files():
    d = os.environ.get("KDA_PRE_CAPTURE_DIR", "")
    files = sorted(glob.glob(os.path.join(d, "*.pt"))) if d else []
    cap = int(os.environ.get("KDA_PRE_CAPTURE_MAX", "0") or 0)
    if cap and len(files) > cap:
        step = len(files) / cap
        files = [files[int(i * step)] for i in range(cap)]
    return files


def load_record(path):
    r = torch.load(path, map_location="cpu", weights_only=False)
    cu = r["cu_seqlens"].to(torch.int32)
    c = dict(q=r["q"].cuda(), k=r["k"].cuda(), v=r["v"].cuda(), raw_g=r["raw_g"].cuda(),
             beta=r["beta"].cuda(), A_log=r["A_log"].cuda(), g_bias=r["dt_bias"].cuda(),
             initial_state=r["initial_state"].cuda(), cu_seqlens=cu.cuda())
    assert bool(r.get("safe_gate", True)) and float(r.get("lower_bound", LB) or LB) == LB
    return c, (r["o"].cuda(), r["final_state"].cuda()), r.get("cls") or "all"


@needs_sm80
@pytest.mark.skipif(not _capture_files(), reason="KDA_PRE_CAPTURE_DIR not set or empty")
def test_real_captured_inputs(warmed):
    """Per class of captured call (first / cont / multi): sum of per-call mean
    errors <= 1.10x, average per-call max <= 1.25x and worst max <= 1.25x the
    live (upstream) outputs' errors against fp64."""
    agg = {}
    for f in _capture_files():
        c, live, cls = load_record(f)
        new = call(kp.chunk_kda_with_fused_gate, c)
        ref = reference_fp64(c)
        torch.cuda.synchronize()
        for name, y, yi, r in (("o", new[0], live[0], ref[0]),
                               ("state", new[1], live[1], ref[1])):
            cmx, cmn = errors(y, r)
            imx, imn = errors(yi, r)
            for key in ((cls, name), ("all", name)):
                a = agg.setdefault(key, [0.0] * 6)
                a[0] += cmn; a[1] += imn; a[2] += cmx; a[3] += imx  # noqa: E702
                a[4] = max(a[4], cmx); a[5] = max(a[5], imx)  # noqa: E702
        del c, live, new, ref
    bad = {}
    for key, (gm, tm, gx, tx, gw, tw) in agg.items():
        rm, rx, rw = ratio(gm, tm), ratio(gx, tx), ratio(gw, tw)
        if not (rm <= MEAN_RATIO and rx <= MAX_RATIO and rw <= MAX_RATIO):
            bad[key] = (rm, rx, rw)
    assert not bad, bad


# ------------------------------------------------------------------ GPU: graphs
@needs_sm80
@pytest.mark.parametrize("seqlens,init", [([2304], "real"), ([576] * 4, "mixed")])
def test_cuda_graph_replay_bitwise_no_growth(warmed, seqlens, init):
    c = make_case(seqlens, init, seed=61)
    fn = kp.chunk_kda_with_fused_gate
    for _ in range(2):
        eager = call(fn, c)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(fn, c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = call(fn, c)
    g.replay()
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    assert torch.equal(outs[0], eager[0]) and torch.equal(outs[1], eager[1])
    assert m1 == m0, f"allocation growth {m1 - m0} B over 3 replays"
    del g
