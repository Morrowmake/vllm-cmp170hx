# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused sm_80 KDA decode step v2 (VLLM_GLM5_DECODE_KDA_V2).

CPU part (no device): the dispatch gate and the env default.
GPU part (skipped without CUDA): the Glm5NextLinearAttention forward with the
switch off and on, driven through the real forward()/_forward() with a
spec-decode metadata object:

  * ON vs OFF agree (output within 2 bf16 ulp, state within the fp32 band
    apart from rare bf16 rounding ties of the gate projections);
  * ON outside the covered shapes is BITWISE identical to OFF, on both the
    v1 fused path and the upstream path;
  * OFF matches a direct thin_gemm + kda_decode composition bitwise;
  * ON is deterministic and CUDA-graph capturable (replay == eager, bitwise);
  * the pre-capture warmup compiles exactly the admitted (nseq, T) plans.

    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_decode_v2.py
"""

import os
import types

try:
    import pytest
except ImportError:  # the vllm-dev venv has no pytest; see __main__ below
    pytest = None
import torch

import vllm.ampere_decode as ad
from vllm.ampere_decode import use_ampere_kda_decode_v2

H, D, KA, CONV_K = 16, 128, 128, 4
PROJ = H * D
CONV_DIM = 3 * PROJ
PW = CONV_DIM + H + 2 * KA
ENV = ("VLLM_GLM5_DECODE_KERNELS", "VLLM_GLM5_DECODE_KDA",
       "VLLM_GLM5_DECODE_KDA_V2", "VLLM_GLM5_DECODE_KDA_MAX_TOKENS",
       "VLLM_GLM5_DECODE_MHC", "VLLM_GLM5_DECODE_MOE_ROUTING")


class _Env:
    """setenv/undo for the standalone runner and pytest alike."""

    def __init__(self):
        self._saved = {k: os.environ.get(k) for k in ENV}
        self._sm80 = ad._SM80_CACHE

    def set(self, sm80=True, **kv):
        for k, v in kv.items():
            os.environ[k] = str(v)
        ad._SM80_CACHE = sm80
        return self

    def undo(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ad._SM80_CACHE = self._sm80


def _w(dtype=torch.bfloat16, shape=(PROJ, D), device="cpu"):
    return torch.zeros(shape, dtype=dtype, device=device)


# ------------------------------------------------------------------ CPU tests

def test_env_default_is_off():
    e = _Env()
    try:
        os.environ.pop("VLLM_GLM5_DECODE_KDA_V2", None)
        from vllm import envs

        assert envs.VLLM_GLM5_DECODE_KDA_V2 is False
        e.set(VLLM_GLM5_DECODE_KERNELS=1)
        assert use_ampere_kda_decode_v2(1, 4, H, D, _w(), _w()) is False
    finally:
        e.undo()


def test_gate_needs_master_and_family_flag():
    e = _Env()
    try:
        e.set(VLLM_GLM5_DECODE_KERNELS=0, VLLM_GLM5_DECODE_KDA_V2=1)
        assert use_ampere_kda_decode_v2(1, 4, H, D, _w(), _w()) is False
        e.set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA_V2=1)
        assert use_ampere_kda_decode_v2(1, 4, H, D, _w(), _w()) is True
        e.set(sm80=False)
        assert use_ampere_kda_decode_v2(1, 4, H, D, _w(), _w()) is False
    finally:
        e.undo()


def test_gate_covers_exactly_the_validated_shapes():
    e = _Env()
    try:
        e.set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA_V2=1)
        w = _w()
        for nseq in range(1, 9):
            for t in range(1, 6):
                assert use_ampere_kda_decode_v2(nseq, nseq * t, H, D, w, w), (nseq, t)
        assert not use_ampere_kda_decode_v2(9, 36, H, D, w, w)      # > 8 seqs
        assert not use_ampere_kda_decode_v2(1, 6, H, D, w, w)       # T > 5
        assert not use_ampere_kda_decode_v2(2, 7, H, D, w, w)       # ragged
        assert not use_ampere_kda_decode_v2(0, 4, H, D, w, w)
        assert not use_ampere_kda_decode_v2(1, 4, 64, D, w, w)      # 64 heads
        assert not use_ampere_kda_decode_v2(1, 4, H, 64, w, w)
        e.set(VLLM_GLM5_DECODE_KDA_MAX_TOKENS=8)
        assert not use_ampere_kda_decode_v2(4, 16, H, D, w, w)      # bound
        assert use_ampere_kda_decode_v2(2, 8, H, D, w, w)
    finally:
        e.undo()


def test_gate_checks_the_weights():
    e = _Env()
    try:
        e.set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA_V2=1)
        ok = _w()
        assert use_ampere_kda_decode_v2(1, 4, H, D, ok, ok)
        assert not use_ampere_kda_decode_v2(1, 4, H, D, _w(torch.float16), ok)
        assert not use_ampere_kda_decode_v2(1, 4, H, D, ok, _w(torch.int32))
        assert not use_ampere_kda_decode_v2(1, 4, H, D, _w(shape=(PROJ, 64)), ok)
        assert not use_ampere_kda_decode_v2(1, 4, H, D, _w(shape=(D, PROJ)).t(), ok)
    finally:
        e.undo()


def test_import_does_not_initialise_cuda():
    import importlib

    importlib.reload(ad)
    assert torch.cuda.is_initialized() is False


# ------------------------------------------------------------------ GPU tests

def _layer(kda_v2, seed=0, nslot=64, T=4, dev="cuda"):
    """A Glm5NextLinearAttention carrying only what forward()/_forward() read."""
    from vllm.ampere_thin_gemm.thin_gemm import thin_gemm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention as K

    g = torch.Generator(device=dev).manual_seed(seed)

    def n(shape, mean, std, dtype):
        t = torch.empty(shape, device=dev, dtype=torch.float32)
        return t.normal_(mean, std, generator=g).to(dtype)

    class _Lin:
        def __init__(self, w):
            self.weight = w

        def __call__(self, x):
            return (thin_gemm(x, self.weight),)

    with set_current_vllm_config(VllmConfig()):
        from vllm.third_party.flash_linear_attention.ops.kda import (
            FusedRMSNormGated,
        )

        norm = FusedRMSNormGated(D, activation="sigmoid", eps=1e-5)
    norm = norm.to(device=dev, dtype=torch.bfloat16)
    norm.weight.data = n((D,), 0.1325, 0.0126, torch.bfloat16)

    L = types.SimpleNamespace()
    L.in_proj_qkvbfg_a = lambda x: (x,)
    L.local_projection_size, L.local_num_heads, L.head_dim = PROJ, H, D
    L.f_b_proj = _Lin(n((PROJ, KA), 0.0, 0.02344, torch.bfloat16))
    L.g_b_proj = _Lin(n((PROJ, KA), 0.0, 0.02858, torch.bfloat16))
    L.o_norm = _Norm(norm)
    L.o_proj = lambda x: (x,)
    L.prefix = "kda.0"
    slen = CONV_K - 1 + (T - 1)
    conv = n((nslot, slen, CONV_DIM), 0.0, 1.0, torch.bfloat16)
    rec = n((nslot, H, D, D), 0.0, 0.5, torch.float32)
    L.kv_cache = (conv, rec)
    L.kda_safe_gate, L.kda_lower_bound = True, -5.0
    L._conv_state_dim_first = False
    L._merged_conv_weight = n((CONV_DIM, CONV_K), 0.0, 0.5, torch.float32)
    L.q_conv1d = types.SimpleNamespace(bias=None)
    L.A_log = n((1, 1, H, 1), 1.527, 0.411, torch.float32)
    L.dt_bias = n((PROJ,), -0.815, 1.211, torch.float32)
    L._kda_v2 = kda_v2
    L._ampere_kda_normed = False
    L._forward = types.MethodType(K._forward, L)
    L._fill_deferred_g2 = types.MethodType(K._fill_deferred_g2, L)
    L.forward = types.MethodType(K.forward, L)
    return L


class _Norm:
    """self.o_norm with the production forward_cuda dispatch."""

    def __init__(self, m):
        self.m, self.weight, self.eps = m, m.weight, m.eps

    def __call__(self, x, g):
        return self.m.forward_cuda(x, g)


def _meta(nseq, T, acc, slots, dev="cuda"):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    M = nseq * T
    return GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=nseq, num_spec_decode_tokens=M, num_actual_tokens=M,
        spec_query_start_loc=torch.arange(0, nseq + 1, device=dev,
                                          dtype=torch.int32) * T,
        spec_state_indices_tensor=torch.as_tensor(slots, device=dev,
                                                  dtype=torch.int32),
        spec_sequence_masks=torch.ones(nseq, device=dev, dtype=torch.bool),
        num_accepted_tokens=torch.as_tensor(acc, device=dev, dtype=torch.int32),
    )


def _projected(M, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    b = torch.empty(M, PW, device=dev).normal_(0.0, 1.0, generator=g)
    b[:, CONV_DIM + H:CONV_DIM + H + KA] *= 1.0 / (0.02344 * KA ** 0.5)
    b[:, CONV_DIM + H + KA:] *= 1.0 / (0.02858 * KA ** 0.5)
    return b.to(torch.bfloat16)


def _run(layer, x, meta, pools0):
    """forward() from a fixed state -> (out, conv, rec)."""
    import vllm.models.glm5next.common.kda as kmod

    conv, rec = layer.kv_cache
    conv.copy_(pools0[0])
    rec.copy_(pools0[1])
    saved = kmod.get_forward_context
    kmod.get_forward_context = lambda: types.SimpleNamespace(
        attn_metadata={layer.prefix: meta})
    try:
        # a fresh copy: the upstream conv path overwrites its input in place
        y = layer.forward(x.clone(), None)
        torch.cuda.synchronize()
    finally:
        kmod.get_forward_context = saved
    return y.clone(), conv.clone(), rec.clone()


def _pair(nseq, T=4, seed=1, env=None):
    on, off = _layer(True, seed=seed, T=T), _layer(False, seed=seed, T=T)
    pools0 = (on.kv_cache[0].clone(), on.kv_cache[1].clone())
    slots = [[1 + s * T + t for t in range(T)] for s in range(nseq)]
    acc = [1 + s % T for s in range(nseq)]
    return on, off, pools0, _meta(nseq, T, acc, slots), _projected(nseq * T, seed)


def _warm(nseq, T):
    from vllm.ampere_decode import kda_decode, kda_decode_v2

    kda_decode.warmup(plans=((nseq, T),))
    if nseq <= 8:
        kda_decode_v2.warmup(plans=((nseq, T),))


def gpu_test_on_matches_off():
    e = _Env().set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA=1,
                   VLLM_GLM5_DECODE_KDA_V2=1)
    try:
        for nseq in (1, 4, 8):
            _warm(nseq, 4)
            on, off, p0, meta, x = _pair(nseq, seed=10 + nseq)
            y1, c1, r1 = _run(on, x, meta, p0)
            y0, c0, r0 = _run(off, x, meta, p0)
            assert on._ampere_kda_normed and off._ampere_kda_normed
            tol = 2.0 * 2 ** -8 * float(y0.float().abs().max())
            dy = float((y1.float() - y0.float()).abs().max())
            assert dy <= tol, (nseq, dy, tol)
            assert torch.equal(c1, c0), nseq
            d = (r1 - r0).abs()
            frac = float((d > 1e-4).float().mean())
            assert float(d.max()) < 3e-2 and frac < 1e-3, (nseq, float(d.max()), frac)
            print(f"  nseq={nseq}: out diff {dy:.2e} (tol {tol:.2e}), "
                  f"state max {float(d.max()):.2e}, >1e-4 {100 * frac:.4f} %")
    finally:
        e.undo()


def gpu_test_fallback_is_bitwise_off():
    """ON outside the covered shapes (9 sequences) takes exactly the OFF path:
    the v1 fused kernel, and with VLLM_GLM5_DECODE_KDA=0 the upstream kernels."""
    for kda_v1 in (1, 0):
        e = _Env().set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA=kda_v1,
                       VLLM_GLM5_DECODE_KDA_V2=1)
        try:
            _warm(9, 4)
            on, off, p0, meta, x = _pair(9, seed=31)
            y1, c1, r1 = _run(on, x, meta, p0)
            y0, c0, r0 = _run(off, x, meta, p0)
            assert on._ampere_kda_normed == bool(kda_v1)
            assert torch.equal(y1, y0) and torch.equal(c1, c0) and torch.equal(r1, r0)
            print(f"  VLLM_GLM5_DECODE_KDA={kda_v1}: bitwise equal")
        finally:
            e.undo()


def gpu_test_off_is_the_unfused_composition():
    from vllm.ampere_decode.kda_decode import kda_decode
    from vllm.ampere_thin_gemm.thin_gemm import thin_gemm

    e = _Env().set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA=1,
                   VLLM_GLM5_DECODE_KDA_V2=0)
    try:
        _warm(4, 4)
        _, off, p0, meta, x = _pair(4, seed=41)
        y0, c0, r0 = _run(off, x, meta, p0)
        conv, rec = off.kv_cache
        conv.copy_(p0[0])
        rec.copy_(p0[1])
        M = x.shape[0]
        g1 = thin_gemm(x[:, CONV_DIM + H:CONV_DIM + H + KA],
                       off.f_b_proj.weight).reshape(1, M, H, D)
        g2 = thin_gemm(x[:, CONV_DIM + H + KA:], off.g_b_proj.weight).reshape(M, H, D)
        ssi = meta.spec_state_indices_tensor
        out = torch.empty(1, M, H, D, device="cuda", dtype=torch.bfloat16)
        kda_decode(x[:, :CONV_DIM], conv.transpose(-1, -2), off._merged_conv_weight,
                   None, g1, x[:, CONV_DIM:CONV_DIM + H].unsqueeze(0), g2,
                   off.o_norm.weight, rec, ssi[:, 0][:4], ssi,
                   meta.num_accepted_tokens, meta.spec_query_start_loc, 4,
                   off.A_log.view(-1), off.dt_bias, lower_bound=-5.0, eps=1e-5,
                   out=out)
        torch.cuda.synchronize()
        assert torch.equal(out.reshape(M, -1), y0)
        assert torch.equal(conv, c0) and torch.equal(rec, r0)
    finally:
        e.undo()


def gpu_test_on_deterministic_and_capturable():
    import vllm.models.glm5next.common.kda as kmod

    e = _Env().set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA=1,
                   VLLM_GLM5_DECODE_KDA_V2=1)
    try:
        for nseq in (1, 4):
            _warm(nseq, 4)
            on, _, p0, meta, x = _pair(nseq, seed=51 + nseq)
            ya, ca, ra = _run(on, x, meta, p0)
            yb, cb, rb = _run(on, x, meta, p0)
            assert torch.equal(ya, yb) and torch.equal(ca, cb) and torch.equal(ra, rb)

            conv, rec = on.kv_cache
            saved = kmod.get_forward_context
            kmod.get_forward_context = lambda: types.SimpleNamespace(
                attn_metadata={on.prefix: meta})
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    on.forward(x, None)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    yg = on.forward(x, None)
            finally:
                kmod.get_forward_context = saved
            for _ in range(2):
                conv.copy_(p0[0])
                rec.copy_(p0[1])
                gr.replay()
                torch.cuda.synchronize()
                assert torch.equal(yg, ya) and torch.equal(conv, ca) \
                    and torch.equal(rec, ra), nseq
            del gr
    finally:
        e.undo()


def gpu_test_warmup_covers_the_capture_plans():
    """warmup_ampere_decode compiles v2 for every (nseq, T) the gate admits and
    allocates its counters/workspace before capture, and nothing else."""
    from vllm.ampere_decode import kda_decode_v2
    from vllm.ampere_decode.warmup import warmup_ampere_decode

    class _KDA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._conv_state_dim_first = False
            self.local_num_heads, self.head_dim, self.num_spec = H, D, 3
            w = torch.zeros(PROJ, KA, device="cuda", dtype=torch.bfloat16)
            self.f_b_proj = types.SimpleNamespace(weight=w)
            self.g_b_proj = types.SimpleNamespace(weight=w)

    model = torch.nn.Sequential(_KDA())
    worker = types.SimpleNamespace(
        get_model=lambda: model, device="cuda",
        vllm_config=types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=8)))
    e = _Env().set(VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_MHC=0,
                   VLLM_GLM5_DECODE_MOE_ROUTING=0, VLLM_GLM5_DECODE_KDA=0,
                   VLLM_GLM5_DECODE_KDA_V2=1)
    try:
        kda_decode_v2._WARMED.clear()
        warmup_ampere_decode(worker, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])
        got = sorted({(k[0], k[1]) for k in kda_decode_v2._WARMED})
        assert got == [(n, 4) for n in range(1, 9)], got
        ctr, ws = kda_decode_v2._CTR[("cuda", torch.cuda.current_device())]
        assert ws.numel() >= 8 * H * kda_decode_v2._WS_T * 2 * D
    finally:
        e.undo()


CPU_TESTS = (test_env_default_is_off, test_gate_needs_master_and_family_flag,
             test_gate_covers_exactly_the_validated_shapes,
             test_gate_checks_the_weights, test_import_does_not_initialise_cuda)
GPU_TESTS = (gpu_test_on_matches_off, gpu_test_fallback_is_bitwise_off,
             gpu_test_off_is_the_unfused_composition,
             gpu_test_on_deterministic_and_capturable,
             gpu_test_warmup_covers_the_capture_plans)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(),
                                    reason="needs an sm_80 GPU")
    test_gpu_on_matches_off = _needs_gpu(gpu_test_on_matches_off)
    test_gpu_fallback_is_bitwise_off = _needs_gpu(gpu_test_fallback_is_bitwise_off)
    test_gpu_off_is_the_unfused_composition = _needs_gpu(
        gpu_test_off_is_the_unfused_composition)
    test_gpu_on_deterministic_and_capturable = _needs_gpu(
        gpu_test_on_deterministic_and_capturable)
    test_gpu_warmup_covers_the_capture_plans = _needs_gpu(
        gpu_test_warmup_covers_the_capture_plans)


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
