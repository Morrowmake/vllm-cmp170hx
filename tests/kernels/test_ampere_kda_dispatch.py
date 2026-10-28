# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA decode dispatch of Glm5NextLinearAttention over every served shape,
against an FP64 reference.

One decode path: the fused sm_80 step (vllm_sm80/models/glm5next/ops/kda_decode.py),
in RECOVER mode under RecoverSSM (tensor parallel, 16 heads per card: pure
verify steps and the spec rows of mixed steps), with per-draft-slot states
otherwise (pipeline parallel, 64 heads per card: pure verify steps of the
measured shapes, the wide schedule beyond one sequence of up to 4 tokens); the
unfused chain (f_b/g_b GEMMs, causal_conv1d_update, fla fused_recurrent_kda,
o_norm) elsewhere. Served shapes: up to 8 sequences of 1..8 tokens (draft depth
up to 7), pure verify and mixed steps.

CPU part: the dispatch table (64 heads take every measured shape under
VLLM_AMPERE_DECODE_KERNELS), the wide-schedule choice.
GPU part (skipped without CUDA), driven through the real forward()/_forward():
for every served shape, the step's normed output o, its consumer y = o @ Wo^T
(FP64 consumer on the kernel's bf16 output) and the per-token recurrent states
are compared with an FP64 model of the layer. The dispatched path passes when
it is no worse than the unfused chain on the same inputs (mean <= 1.10x,
max <= 1.25x its error, for o and y), so are its per-token states, and its
conv state is bitwise the unfused chain's. Also: mixed-step spec rows,
CUDA-graph replay bitwise eager, a breakable (PIECEWISE) graph captured on one
kind of step and replayed on another applying o_norm by the replayed step, and
the warmup plans.

    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_dispatch.py
"""

import os
import types

try:
    import pytest
except ImportError:  # the vllm-dev venv has no pytest; see __main__ below
    pytest = None
import torch

from vllm_sm80.kernels import _platform
from vllm_sm80.models.glm5next.ops import kda_decode
from vllm_sm80.models.glm5next.ops.kda_decode import use_ampere_kda_decode_v2

D, KA, CONV_K = 128, 128, 4
WINDOW = 8                      # draft depth 7 + 1: the served speculative window
HIDDEN = 512                    # synthetic o_proj rows (the consumer y)
ENV = ("VLLM_AMPERE_DECODE_KERNELS", "VLLM_GLM5_DECODE_MOE_ROUTING",
       "VLLM_USE_BREAKABLE_CUDAGRAPH")
MEAN_X, MAX_X = 1.10, 1.25
REF_DEVICE = "cpu"              # where the FP64 model runs ("cuda" for large runs)


class _Env:
    """setenv/undo for the standalone runner and pytest alike."""

    def __init__(self):
        self._saved = {k: os.environ.get(k) for k in ENV}
        self._sm80 = _platform._SM80_CACHE

    def set(self, sm80=True, **kv):
        for k, v in kv.items():
            os.environ[k] = str(v)
        _platform._SM80_CACHE = sm80
        return self

    def undo(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _platform._SM80_CACHE = self._sm80


def _w(H, dtype=torch.bfloat16, device="cpu"):
    return torch.zeros((H * D, KA), dtype=dtype, device=device)


def expected_path(H, nseq, T):
    """The 1.8 dispatch of a pure verify step: 'v2' (per-draft-slot step,
    _select schedule), 'v2w' (wide schedule) or 'unfused'. RecoverSSM layers
    (16 heads under tensor parallel) verify every step with v2 RECOVER."""
    from vllm_sm80.models.glm5next.ops.kda_decode import wide_schedule

    if nseq < 1 or nseq > 8 or T < 1 or T > 8:
        return "unfused"
    if H == 16:
        return "unfused" if (T >= 6 and nseq > 4) else "v2"
    if H == 64:
        if (nseq, T) not in kda_decode._KDA_V2_WIDE_SHAPES:
            return "unfused"
        return "v2w" if wide_schedule(nseq, T, H) else "v2"
    return "unfused"


# ------------------------------------------------------------------ CPU tests

def test_dispatch_table_every_served_shape():
    e = _Env()
    try:
        e.set(VLLM_AMPERE_DECODE_KERNELS=1)
        for H in (16, 64):
            w = _w(H)
            for nseq in range(1, 10):
                for T in range(1, 10):
                    want = expected_path(H, nseq, T) != "unfused"
                    got = use_ampere_kda_decode_v2(nseq, nseq * T, H, D, w, w)
                    assert got == want, (H, nseq, T, got)
        # ragged token counts, zero sequences, other head counts and widths
        assert not use_ampere_kda_decode_v2(2, 7, 16, D)
        assert not use_ampere_kda_decode_v2(0, 4, 16, D)
        assert not use_ampere_kda_decode_v2(1, 4, 32, D)
        assert not use_ampere_kda_decode_v2(1, 4, 16, 64)
        # the retired 64-head switch changes nothing
        os.environ["VLLM_GLM5_DECODE_KDA_WIDE"] = "0"
        assert use_ampere_kda_decode_v2(8, 32, 64, D)
        del os.environ["VLLM_GLM5_DECODE_KDA_WIDE"]
        # the master switch, the device
        e.set(VLLM_AMPERE_DECODE_KERNELS=0)
        assert not use_ampere_kda_decode_v2(1, 4, 16, D)
        assert not use_ampere_kda_decode_v2(1, 4, 64, D)
        e.set(VLLM_AMPERE_DECODE_KERNELS=1, sm80=False)
        assert not use_ampere_kda_decode_v2(1, 4, 16, D)
    finally:
        e.undo()


def test_gate_checks_the_weights():
    e = _Env()
    try:
        e.set(VLLM_AMPERE_DECODE_KERNELS=1)
        ok = _w(16)
        assert use_ampere_kda_decode_v2(1, 4, 16, D, ok, ok)
        assert not use_ampere_kda_decode_v2(1, 4, 16, D, _w(16, torch.float16), ok)
        assert not use_ampere_kda_decode_v2(1, 4, 16, D, ok, _w(16, torch.int32))
        assert not use_ampere_kda_decode_v2(1, 4, 16, D, torch.zeros(16 * D, 64), ok)
        assert not use_ampere_kda_decode_v2(1, 4, 16, D, torch.zeros(KA, 16 * D).t(), ok)
        # 16-head weights on a 64-head layer
        assert not use_ampere_kda_decode_v2(1, 4, 64, D, ok, ok)
    finally:
        e.undo()


def test_wide_schedule():
    """64 heads: the wide schedule everywhere except one sequence of up to 4
    tokens, which keeps the 1.7 schedule (bitwise the 1.7 output)."""
    from vllm_sm80.models.glm5next.ops.kda_decode import wide_schedule

    for nseq in range(1, 9):
        for T in range(1, 9):
            assert wide_schedule(nseq, T, 64) == (nseq > 1 or T > 4), (nseq, T)
            assert not wide_schedule(nseq, T, 16)


def test_import_does_not_initialise_cuda():
    import importlib

    importlib.reload(kda_decode)
    assert torch.cuda.is_initialized() is False


# ------------------------------------------------------------------ the harness
# (reusable with real captured weights, inputs, states and conv windows in
# place of the synthetic ones below)

def synthetic_weights(H, seed, dev="cuda"):
    """Weights with the statistics of the GLM-5.3-Flash checkpoint's KDA layers."""
    g = torch.Generator(device=dev).manual_seed(seed)

    def n(shape, mean, std, dtype=torch.float32):
        t = torch.empty(shape, device=dev, dtype=torch.float32)
        return t.normal_(mean, std, generator=g).to(dtype)

    P = H * D
    return {
        "w_f": n((P, KA), 0.0, 0.02344, torch.bfloat16),
        "w_g": n((P, KA), 0.0, 0.02858, torch.bfloat16),
        "conv_w": n((3 * P, CONV_K), 0.0, 0.5),
        "o_w": n((D,), 0.1325, 0.0126, torch.bfloat16),
        "a_log": n((H,), 1.527, 0.411),
        "dt_bias": n((P,), -0.815, 1.211),
        "o_proj": n((HIDDEN, P), 0.0, 0.02, torch.bfloat16),
        "eps": 1e-5,
        "lower_bound": -5.0,
    }


def synthetic_inputs(H, M, seed, dev="cuda"):
    """[M, 3P + H + 2 KA] bf16 projection rows (q|k|v|beta|f_a|g_a), scaled so
    the f_b/g_b outputs have unit variance."""
    P = H * D
    g = torch.Generator(device=dev).manual_seed(seed)
    b = torch.empty(M, 3 * P + H + 2 * KA, device=dev).normal_(0.0, 1.0, generator=g)
    b[:, 3 * P + H:3 * P + H + KA] *= 1.0 / (0.02344 * KA ** 0.5)
    b[:, 3 * P + H + KA:] *= 1.0 / (0.02858 * KA ** 0.5)
    return b.to(torch.bfloat16)


class _Lin:
    """f_b_proj / g_b_proj with the production GEMM dispatch (thin GEMM to 32
    rows, cuBLAS above)."""

    def __init__(self, w):
        self.weight = w

    def __call__(self, x):
        from vllm_sm80.kernels.thin_gemm import ampere_thin_gemm

        return (ampere_thin_gemm(None, x, self.weight),)


class _Norm:
    """self.o_norm with the production forward_cuda dispatch. ``calls`` counts
    the unfused passes: a fused sm_80 kernel applies the norm itself."""

    def __init__(self, m):
        self.m, self.weight, self.eps = m, m.weight, m.eps
        self.calls = 0

    def __call__(self, x, g):
        self.calls += 1
        return self.m.forward_cuda(x, g)


def _fused(L):
    """The last run_layer took a fused sm_80 path (o_norm inside the kernel)."""
    return L.o_norm.calls == 0


def build_layer(H, w, *, kernels, nslot, recover=False, window=WINDOW,
                dev="cuda"):
    """A Glm5NextLinearAttention carrying only what forward()/_forward() read,
    with the decode attributes __init__ derives from the switches."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_sm80.models.glm5next.sm80_kda import Glm5NextLinearAttention as K

    with set_current_vllm_config(VllmConfig()):
        from vllm.third_party.flash_linear_attention.ops.kda import (
            FusedRMSNormGated,
        )

        norm = FusedRMSNormGated(D, activation="sigmoid", eps=float(w["eps"]))
    norm = norm.to(device=dev, dtype=torch.bfloat16)
    norm.weight.data = w["o_w"].to(dev).to(torch.bfloat16).contiguous()
    P = H * D
    L = types.SimpleNamespace()
    L.in_proj_qkvbfg_a = lambda x: (x,)
    L.local_projection_size, L.local_num_heads, L.head_dim = P, H, D
    L.num_spec = window - 1
    L.f_b_proj = _Lin(w["w_f"].to(dev).to(torch.bfloat16).contiguous())
    L.g_b_proj = _Lin(w["w_g"].to(dev).to(torch.bfloat16).contiguous())
    L.o_norm = _Norm(norm)
    L.o_proj = lambda x: (x,)
    L.prefix = "kda.0"
    conv = torch.zeros(nslot, CONV_K - 1 + window - 1, 3 * P, device=dev,
                       dtype=torch.bfloat16)
    rec = torch.zeros(nslot, H, D, D, device=dev, dtype=torch.float32)
    L.kv_cache = (conv, rec)
    L.kda_safe_gate, L.kda_lower_bound = True, float(w["lower_bound"])
    L._conv_state_dim_first = False
    L._merged_conv_weight = w["conv_w"].to(dev).float().reshape(3 * P, CONV_K).contiguous()
    L.q_conv1d = types.SimpleNamespace(bias=None)
    L.A_log = w["a_log"].to(dev).float().reshape(1, 1, H, 1).contiguous()
    L.dt_bias = w["dt_bias"].to(dev).float().reshape(-1).contiguous()
    L._kda_v2 = bool(kernels)
    L._kda_wide = bool(kernels and H == 64)
    L._kda_step_tile = bool(kernels)
    L._kda_recover = bool(kernels and recover)
    L._kda_recover_records = None
    if L._kda_recover:
        from vllm_sm80.kernels.kda import recover as kr

        L._kda_recover_pool = torch.zeros(1, 3, 8, kr.WS_T, H, D, device=dev)
        L._kda_recover_records = kr.layer_records(L._kda_recover_pool, 0)
    L._pp_kda_prefill = False
    for name in ("forward", "_forward", "_forward_core", "_fill_deferred_g2"):
        setattr(L, name, types.MethodType(getattr(K, name), L))
    return L


def step_metadata(nseq, T, slots, acc, *, decodes=(), tile=None, dev="cuda"):
    """GDN metadata of a pure verify step (decodes empty) or of a mixed step:
    `decodes` plain one-token rows (their state slots) first, then the spec
    rows, in token order (the order the builder's stable argsort gives)."""
    from vllm_sm80.kernels.kda.recover import Glm5NextGDNAttentionMetadata

    nd = len(decodes)
    M = nd + nseq * T
    i32 = dict(device=dev, dtype=torch.int32)
    common = dict(
        num_prefills=0, num_prefill_tokens=0, num_decodes=nd, num_decode_tokens=nd,
        num_spec_decodes=nseq, num_spec_decode_tokens=nseq * T, num_actual_tokens=M,
        spec_query_start_loc=torch.arange(0, nseq + 1, **i32) * T,
        spec_state_indices_tensor=torch.as_tensor(slots, **i32),
        spec_sequence_masks=torch.ones(nseq + nd, device=dev, dtype=torch.bool),
        num_accepted_tokens=torch.as_tensor(acc, **i32),
        spec_max_query_len=tile if tile is not None else T,
    )
    if nd:
        common.update(
            non_spec_query_start_loc=torch.arange(0, nd + 1, **i32),
            non_spec_state_indices_tensor=torch.as_tensor(list(decodes), **i32),
            spec_token_indx=torch.arange(nd, M, device=dev),
            non_spec_token_indx=torch.arange(0, nd, device=dev),
        )
    return Glm5NextGDNAttentionMetadata(**common)


def decode_metadata(n, dev="cuda"):
    """A plain decode step: n one-token rows on state slots 1 .. n."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    i32 = dict(device=dev, dtype=torch.int32)
    return GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=n, num_decode_tokens=n,
        num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=n,
        non_spec_query_start_loc=torch.arange(0, n + 1, **i32),
        non_spec_state_indices_tensor=torch.arange(1, n + 1, **i32),
    )


def run_layer(L, x, meta, conv0, rec0, calls=None):
    """forward() from a fixed state -> (normed o [M, P] bf16, conv, rec, records).
    `calls` (a list) collects the fused kernel's (wide, skip_norm, recover)
    per launch."""
    import vllm.models.glm5next.common.kda as kmod
    from vllm_sm80.models.glm5next.ops import kda_decode as v2mod

    conv, rec = L.kv_cache
    conv.copy_(conv0)
    rec.copy_(rec0)
    if L._kda_recover_records is not None:
        L._kda_recover_pool.zero_()
    L.o_norm.calls = 0
    saved = kmod.get_forward_context
    kmod.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={L.prefix: meta})
    fused = v2mod.kda_decode_v2
    if calls is not None:
        def counted(*a, **k):
            calls.append((bool(k.get("wide", False)), bool(k.get("skip_norm", False)),
                          k.get("records") is not None))
            return fused(*a, **k)
        v2mod.kda_decode_v2 = counted
    try:
        # a fresh copy: the unfused conv path overwrites its input in place
        y = L.forward(x.clone(), None)
        torch.cuda.synchronize()
    finally:
        kmod.get_forward_context = saved
        v2mod.kda_decode_v2 = fused
    recs = None if L._kda_recover_records is None else L._kda_recover_pool[0].clone()
    return y.clone(), conv.clone(), rec.clone(), recs


def _f64(t):
    return t.detach().to(REF_DEVICE, torch.float64)


def _bf16(t):
    return t.to(torch.bfloat16).to(torch.float64)


def fp64_sequence(w, H, x_rows, win, h0):
    """FP64 model of one sequence's verify rows: x_rows [T, 3P+H+2KA] bf16,
    win [3, 3P] the conv window before the first row, h0 [H, V, K] the state.
    Returns (normed o [T, H, D], states after each token [T, H, V, K])."""
    P = H * D
    x = _f64(x_rows)
    T = x.shape[0]
    cw = _f64(w["conv_w"]).reshape(3 * P, CONV_K)
    src = torch.cat([_f64(win), x[:, :3 * P]], 0)
    y = torch.zeros(T, 3 * P, dtype=torch.float64, device=REF_DEVICE)
    for j in range(CONV_K):
        y = y + src[j:j + T] * cw[:, j]
    y = y / (1.0 + torch.exp(-y))
    q, k, v = (y[:, i * P:(i + 1) * P].reshape(T, H, D) for i in range(3))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(x[:, 3 * P:3 * P + H])
    g1 = x[:, 3 * P + H:3 * P + H + KA] @ _f64(w["w_f"]).t()
    g2 = x[:, 3 * P + H + KA:] @ _f64(w["w_g"]).t()
    a = _f64(w["a_log"]).reshape(H).exp()
    e = torch.exp(float(w["lower_bound"]) * torch.sigmoid(
        a[None, :, None] * (g1.reshape(T, H, D) + _f64(w["dt_bias"]).reshape(H, D))))
    h = _f64(h0).clone()
    outs, states = [], []
    for t in range(T):
        h = h * e[t][:, None, :]
        c = (v[t] - torch.einsum("hvk,hk->hv", h, k[t])) * beta[t][:, None]
        h = h + c[:, :, None] * k[t][:, None, :]
        outs.append(torch.einsum("hvk,hk->hv", h, q[t]))
        states.append(h.clone())
    o = torch.stack(outs)
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + float(w["eps"])) \
        * _f64(w["o_w"]) * torch.sigmoid(g2.reshape(T, H, D))
    return o, torch.stack(states)


def step_states(res, slots, acc, T_list, recover, rec0):
    """Per-sequence states after each token from a path's result: the slots it
    wrote (per-draft-slot step) or the commit replay of its records."""
    out = []
    for n, T in enumerate(T_list):
        if recover:
            from tests.kernels.kda_recover_reference import commit_reference

            rows = []
            for t in range(T):
                rows.append(commit_reference(rec0, res[3].to(rec0.device),
                                             int(slots[n][0]), t + 1, n))
            out.append(torch.stack(rows))
        else:
            out.append(torch.stack([res[2][int(slots[n][t])] for t in range(T)]))
    return out


class Stats:
    """Absolute-error mean and max of a path vs FP64, pooled over a shape."""

    def __init__(self):
        self.s, self.n, self.m = 0.0, 0, 0.0

    def add(self, got, ref):
        d = (_f64(got) - ref).abs()
        self.s += float(d.sum())
        self.n += d.numel()
        self.m = max(self.m, float(d.max()))

    @property
    def mean(self):
        return self.s / max(1, self.n)


def compare_shape(w, H, x, slots, acc, T, res_new, res_old, conv0, rec0, *,
                  recover_new, recover_old, decodes=0):
    """FP64 comparison of one step for both paths. Returns a dict of Stats per
    (path, output) plus the floors."""
    P = H * D
    nseq = len(slots)
    Wo = _f64(w["o_proj"])
    st = {k: Stats() for k in ("o_new", "o_old", "o_floor", "y_new", "y_old", "y_floor",
                               "s_new", "s_old", "s_floor")}
    s_new = step_states(res_new, slots, acc, [T] * nseq, recover_new, rec0)
    s_old = step_states(res_old, slots, acc, [T] * nseq, recover_old, rec0)
    conv_sd = conv0                                  # [slot, rows, 3P] (SD storage)
    for n in range(nseq):
        rows = slice(decodes + n * T, decodes + (n + 1) * T)
        a = int(acc[n])
        s0 = int(slots[n][a - 1])
        win = conv_sd[int(slots[n][0]), a - 1:a + 2]
        o_ref, h_ref = fp64_sequence(w, H, x[rows], win, rec0[s0])
        o_ref = o_ref.reshape(T, P)
        y_ref = o_ref @ Wo.t()
        for tag, res, sts in (("new", res_new, s_new), ("old", res_old, s_old)):
            o = res[0][rows]
            st["o_" + tag].add(o, o_ref)
            st["y_" + tag].add(_f64(o) @ Wo.t(), y_ref)
            st["s_" + tag].add(sts[n], h_ref)
        st["o_floor"].add(_bf16(o_ref), o_ref)
        st["y_floor"].add(_bf16(o_ref) @ Wo.t(), y_ref)
        st["s_floor"].add(_bf16(h_ref), h_ref)
    return st


def verdicts(st):
    """(passed, reasons): o, y and the states no worse than the unfused chain
    (on real captured inputs the states are also held to the bf16 floor;
    random inputs carry no such guarantee for either path)."""
    bad = []
    for o in ("o", "y", "s"):
        if st[o + "_new"].mean > MEAN_X * st[o + "_old"].mean:
            bad.append(f"{o} mean {st[o + '_new'].mean / st[o + '_old'].mean:.3f}x")
        if st[o + "_new"].m > MAX_X * st[o + "_old"].m:
            bad.append(f"{o} max {st[o + '_new'].m / st[o + '_old'].m:.3f}x")
    return not bad, bad


def _setup(H, nseq, T, seed, *, mixed=False, recover=False):
    """Synthetic step: slots 1 + n * WINDOW + t, accepted counts spread over the
    window (RecoverSSM: 1, state in column 0), conv/state pools at random."""
    nd = 2 if mixed else 0
    nslot = 2 + (nseq + nd) * WINDOW
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    P = H * D
    conv0 = torch.empty(nslot, CONV_K - 1 + WINDOW - 1, 3 * P, device="cuda").normal_(
        0.0, 1.0, generator=g).to(torch.bfloat16)
    rec0 = torch.empty(nslot, H, D, D, device="cuda").normal_(0.0, 0.05, generator=g)
    slots = [[1 + n * WINDOW + t for t in range(WINDOW)] for n in range(nseq)]
    acc = [1] * nseq if recover else [1 + (3 * n) % WINDOW for n in range(nseq)]
    decodes = tuple(1 + (nseq + i) * WINDOW for i in range(nd))
    x = synthetic_inputs(H, nd + nseq * T, seed)
    meta = step_metadata(nseq, T, slots, acc, decodes=decodes,
                         tile=WINDOW if mixed else T)
    return x, slots, acc, conv0, rec0, meta, nd


def _check_shape(H, nseq, T, *, recover=False, mixed=False, seed=0, w=None, report=None):
    w = w or synthetic_weights(H, seed=1000 + H)
    x, slots, acc, conv0, rec0, meta, nd = _setup(H, nseq, T, seed, mixed=mixed,
                                                  recover=recover)
    nslot = conv0.shape[0]
    new = build_layer(H, w, kernels=True, nslot=nslot, recover=recover)
    # the unfused chain (decode kernels off): the 1.7 path of every moved shape
    old = build_layer(H, w, kernels=False, nslot=nslot)
    calls, calls_old = [], []
    res_new = run_layer(new, x, meta, conv0, rec0, calls)
    res_old = run_layer(old, x, meta, conv0, rec0, calls_old)
    want = "v2" if recover else ("unfused" if mixed else expected_path(H, nseq, T))
    assert not calls_old
    if want == "unfused":
        assert not calls, (H, nseq, T, mixed, calls)
    else:
        assert calls == [(want == "v2w", mixed, recover)], (H, nseq, T, mixed, calls)
    assert _fused(new) == (want != "unfused" and not mixed), (H, nseq, T, mixed)
    assert torch.equal(res_new[1], res_old[1]), ("conv state", H, nseq, T, mixed)
    if nd:
        assert torch.equal(res_new[0][:nd], res_old[0][:nd]), ("decode rows", H, nseq, T)
        for d in meta.non_spec_state_indices_tensor.tolist():
            assert torch.equal(res_new[2][d], res_old[2][d]), ("decode state", H, nseq, T)
    st = compare_shape(w, H, x, slots, acc, T, res_new, res_old, conv0, rec0,
                       recover_new=recover, recover_old=False, decodes=nd)
    ok, why = verdicts(st)
    if report is not None:
        report.append(((H, nseq, T, recover, mixed), st, ok, why))
    return ok, why


# ------------------------------------------------------------------ GPU tests

def _warm(H, plans, recover=False):
    from vllm_sm80.models.glm5next.ops import kda_decode

    kda_decode.warmup(plans=plans, heads=H, recover=recover, width=WINDOW)


def gpu_test_every_served_shape_vs_fp64():
    """Pure verify steps: 16 heads (RecoverSSM, and the per-draft-slot step
    used where RecoverSSM is off) and 64 heads, 1..8 sequences of 1..8 tokens."""
    from vllm_sm80.models.glm5next.ops.kda_decode import wide_schedule

    e = _Env().set(VLLM_AMPERE_DECODE_KERNELS=1)
    report, failed = [], []
    try:
        for H, recover in ((16, True), (16, False), (64, False)):
            w = synthetic_weights(H, seed=1000 + H)
            for nseq in range(1, 9):
                for T in range(1, 9):
                    if recover:
                        _warm(H, ((nseq, T),), recover=True)
                    elif expected_path(H, nseq, T) != "unfused":
                        _warm(H, ((nseq, T, wide_schedule(nseq, T, H)),))
                    ok, why = _check_shape(H, nseq, T, recover=recover,
                                           seed=nseq * 10 + T, w=w, report=report)
                    if not ok:
                        failed.append(((H, recover, nseq, T), why))
    finally:
        e.undo()
    worst = {}
    for (key, st, ok, _) in report:
        r = (st["o_new"].mean / st["o_old"].mean, st["y_new"].mean / st["y_old"].mean,
             st["s_new"].mean / st["s_old"].mean)
        k = (key[0], key[3])
        worst[k] = tuple(max(a, b) for a, b in zip(worst.get(k, (0, 0, 0)), r))
    for k, r in sorted(worst.items()):
        print(f"  H={k[0]} recover={k[1]}: worst mean ratio o {r[0]:.3f} "
              f"y {r[1]:.3f} state {r[2]:.3f} (vs the unfused chain)")
    assert not failed, failed


def gpu_test_mixed_step_spec_rows():
    """Mixed steps (plain decode rows + verify rows): under RecoverSSM (16
    heads) the spec rows run the fused kernel unnormalised; at 64 heads the
    whole step is the unfused chain. The plain rows and the conv state are
    bitwise the unfused chain's."""

    e = _Env().set(VLLM_AMPERE_DECODE_KERNELS=1)
    failed = []
    try:
        for H, recover in ((16, True), (64, False)):
            w = synthetic_weights(H, seed=1000 + H)
            for nseq, T in ((1, 4), (2, 4), (1, 8), (3, 6), (8, 8)):
                if recover:
                    _warm(H, ((nseq, WINDOW),), recover=True)
                ok, why = _check_shape(H, nseq, T, recover=recover, mixed=True,
                                       seed=500 + nseq * 10 + T, w=w)
                if not ok:
                    failed.append(((H, nseq, T), why))
    finally:
        e.undo()
    assert not failed, failed


def gpu_test_graph_replay_is_eager():
    """The 64-head wide shapes capture into a CUDA graph and replay bitwise."""
    import vllm.models.glm5next.common.kda as kmod
    from vllm_sm80.models.glm5next.ops.kda_decode import wide_schedule

    e = _Env().set(VLLM_AMPERE_DECODE_KERNELS=1)
    try:
        H = 64
        w = synthetic_weights(H, seed=1064)
        for nseq, T in ((1, 8), (2, 4), (8, 8)):
            _warm(H, ((nseq, T, wide_schedule(nseq, T, H)),))
            x, slots, acc, conv0, rec0, meta, _ = _setup(H, nseq, T, 900 + nseq)
            L = build_layer(H, w, kernels=True, nslot=conv0.shape[0])
            ya = run_layer(L, x, meta, conv0, rec0)
            yb = run_layer(L, x, meta, conv0, rec0)
            assert all(torch.equal(a, b) for a, b in zip(ya[:3], yb[:3])), (nseq, T)
            conv, rec = L.kv_cache
            saved = kmod.get_forward_context
            kmod.get_forward_context = lambda: types.SimpleNamespace(
                attn_metadata={L.prefix: meta})
            try:
                xg = x.clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    L.forward(xg.clone(), None)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    yg = L.forward(xg, None)
            finally:
                kmod.get_forward_context = saved
            for _ in range(2):
                conv.copy_(conv0)
                rec.copy_(rec0)
                gr.replay()
                torch.cuda.synchronize()
                assert torch.equal(yg, ya[0]) and torch.equal(conv, ya[1]) \
                    and torch.equal(rec, ya[2]), (nseq, T)
            del gr
    finally:
        e.undo()


def gpu_test_piecewise_replay_norms_by_replay_path():
    """A breakable (PIECEWISE) graph replays forward() for any step of its
    token count, so whether o_norm runs must follow the replayed step, not the
    captured one. Capture a pure verify step (8 x 4 tokens, the shape the
    runner captures PIECEWISE graphs with: the fused sm_80 kernel norms
    in-kernel) and replay a plain decode step of the same 32 tokens (the
    unfused chain norms after the kernels), and the reverse, at 16 heads
    (RecoverSSM) and 64 heads: each replay must be bitwise eager forward() of
    the replayed step."""
    import vllm.models.glm5next.common.kda as kmod
    from vllm_sm80.models.glm5next.ops.kda_decode import wide_schedule
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    e = _Env().set(VLLM_AMPERE_DECODE_KERNELS=1, VLLM_USE_BREAKABLE_CUDAGRAPH=1)
    saved = kmod.get_forward_context
    try:
        nseq, T = 8, 4
        for H, recover in ((16, True), (64, False)):
            assert recover or expected_path(H, nseq, T) != "unfused", H
            if recover:
                _warm(H, ((nseq, T),), recover=True)
            else:
                _warm(H, ((nseq, T, wide_schedule(nseq, T, H)),))
            w = synthetic_weights(H, seed=1000 + H)
            x, _, _, conv0, rec0, spec, _ = _setup(H, nseq, T, 91, recover=recover)
            L = build_layer(H, w, kernels=True, nslot=conv0.shape[0], recover=recover)
            dec = decode_metadata(nseq * T)
            want_spec = run_layer(L, x, spec, conv0, rec0)
            assert _fused(L), H
            want_dec = run_layer(L, x, dec, conv0, rec0)
            assert not _fused(L), H
            # Kernels around the break, as the real projections are: the
            # captured segments then hold work and their own static buffers.
            L.in_proj_qkvbfg_a = lambda t: (t.clone(),)
            L.o_proj = lambda t: (t.clone(),)
            # The class was decorated at import, when breaks may have been off.
            fn = kmod.Glm5NextLinearAttention._forward
            L._forward = types.MethodType(
                eager_break_during_capture(getattr(fn, "__wrapped__", fn)), L)
            conv, rec = L.kv_cache
            step = {}
            kmod.get_forward_context = lambda: types.SimpleNamespace(
                attn_metadata={L.prefix: step["meta"]})
            for captured, replayed, want, label in (
                (spec, dec, want_dec, "verify capture, decode replay"),
                (dec, spec, want_spec, "decode capture, verify replay"),
            ):
                step["meta"] = captured
                xg = x.clone()
                conv.copy_(conv0)
                rec.copy_(rec0)
                cap = BreakableCUDAGraphCapture(pool=torch.cuda.graph_pool_handle())
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), cap:
                    yg = L.forward(xg, None)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                assert cap.num_eager_breaks == 1, cap
                step["meta"] = replayed
                xg.copy_(x)
                conv.copy_(conv0)
                rec.copy_(rec0)
                cap.replay()
                torch.cuda.synchronize()
                dy = float((yg.float() - want[0].float()).abs().max())
                assert torch.equal(yg, want[0]), (H, label, dy)
                assert torch.equal(conv, want[1]) and torch.equal(rec, want[2]), (H, label)
                print(f"  H={H} {label}: bitwise eager")
                del cap
            kmod.get_forward_context = saved
    finally:
        kmod.get_forward_context = saved
        e.undo()


def gpu_test_warmup_plans_64_heads():
    """The 64-head warmup compiles every (nseq, T, schedule) the gate admits."""
    from vllm_sm80.models.glm5next.ops.kda_decode import (
        _warmup_kda_v2,
        wide_schedule,
    )

    wt = torch.zeros(64 * D, KA, device="cuda", dtype=torch.bfloat16)
    layer = types.SimpleNamespace(
        _conv_state_dim_first=False, local_num_heads=64, head_dim=D, num_spec=7,
        _kda_v2=True, _kda_wide=True, _kda_step_tile=True, _kda_recover=False,
        f_b_proj=types.SimpleNamespace(weight=wt), g_b_proj=types.SimpleNamespace(weight=wt))
    worker = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=8)))
    saved = kda_decode.find_module
    kda_decode.find_module = lambda model, attr: layer
    e = _Env().set(VLLM_AMPERE_DECODE_KERNELS=1)
    try:
        kda_decode._WARMED.clear()
        _warmup_kda_v2(worker, torch.nn.Module(), "cuda", [1, 2, 4, 8, 16, 32, 64])
        got = sorted((k[0], k[1], k[4]) for k in kda_decode._WARMED)
        assert got == sorted((n, t, wide_schedule(n, t, 64))
                             for n, t in kda_decode._KDA_V2_WIDE_SHAPES), got
    finally:
        kda_decode.find_module = saved
        e.undo()


CPU_TESTS = (test_dispatch_table_every_served_shape,
             test_gate_checks_the_weights, test_wide_schedule,
             test_import_does_not_initialise_cuda)
GPU_TESTS = (gpu_test_every_served_shape_vs_fp64, gpu_test_mixed_step_spec_rows,
             gpu_test_graph_replay_is_eager,
             gpu_test_piecewise_replay_norms_by_replay_path,
             gpu_test_warmup_plans_64_heads)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(),
                                    reason="needs an sm_80 GPU")
    test_gpu_every_served_shape_vs_fp64 = _needs_gpu(gpu_test_every_served_shape_vs_fp64)
    test_gpu_mixed_step_spec_rows = _needs_gpu(gpu_test_mixed_step_spec_rows)
    test_gpu_graph_replay_is_eager = _needs_gpu(gpu_test_graph_replay_is_eager)
    test_gpu_piecewise_replay_norms_by_replay_path = _needs_gpu(
        gpu_test_piecewise_replay_norms_by_replay_path)
    test_gpu_warmup_plans_64_heads = _needs_gpu(gpu_test_warmup_plans_64_heads)


def _main():
    import sys
    import traceback

    tests = list(CPU_TESTS)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available():
        tests += list(GPU_TESTS)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
