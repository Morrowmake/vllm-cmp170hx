# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 copies of the mHC prenorm projection `fn` for decode v2
(VLLM_GLM5_DECODE_MHC_V2_FN_BF16, on by default; read only for M <= 12).

CPU: the flag is on by default; kernel A gets the bf16 copy only for M <= FN_BF16_MAX_TOKENS; copies are registered only when every `fn`
value is a bf16 value; lookups reject other tensors; the banners go through the
real vLLM logger; both kernel-A variants compile for sm_80 with a bf16 `fn`.
GPU (sm_80, skipped otherwise): every output is bitwise equal between the fp32
`fn` and its bf16 copy, through the real dispatch, eager and under CUDA-graph
replay, with no allocation growth; with the flag off nothing is registered.
"""

import logging

import pytest
import torch
from torch import nn

from vllm import envs
from vllm.ampere_decode import mhc_decode_v2 as v2

FLAG = "VLLM_GLM5_DECODE_MHC_V2_FN_BF16"
HIDDEN, HC = 4096, 4
NOUT = HC * 2 + HC * HC


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(v2, "_FN_BF16", {})


class _Layer(nn.Module):
    def __init__(self, fn_attn, fn_ffn):
        super().__init__()
        self.hc_attn_fn = nn.Parameter(fn_attn, requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(fn_ffn, requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.zeros(NOUT), requires_grad=False)


class _Model(nn.Module):
    def __init__(self, n, dev="cpu", exact=True):
        super().__init__()
        g = torch.Generator(device=dev).manual_seed(0)

        def fn():
            t = torch.randn(NOUT, HC * HIDDEN, generator=g, device=dev) * 0.02
            return t.to(torch.bfloat16).float() if exact else t

        self.layers = nn.ModuleList(_Layer(fn(), fn()) for _ in range(n))


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _logged(fn, *args):
    handler = _Capture()
    lg = logging.getLogger(v2.logger.name)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        result = fn(*args)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
    return result, handler.lines


def test_flag_defaults_on_and_kill_switch(monkeypatch):
    assert envs.VLLM_GLM5_DECODE_MHC_V2_FN_BF16 is True
    monkeypatch.setenv(FLAG, "0")
    assert envs.VLLM_GLM5_DECODE_MHC_V2_FN_BF16 is False


def test_bf16_only_up_to_twelve_rows():
    assert v2.FN_BF16_MAX_TOKENS == 12
    model = _Model(1)
    v2.register_fn_bf16(model)
    fn = model.layers[0].hc_attn_fn
    for M in range(1, 65):
        got = v2._select_fn(fn, M, HC, HIDDEN)
        if M <= 12:
            assert got.dtype == torch.bfloat16 and got is v2._fn_bf16(fn), M
        else:
            assert got is fn, M
    assert v2._select_fn(fn, 4, 2, HIDDEN) is fn  # hc != 4: no Gluon kernel A
    v2._FN_BF16.clear()
    assert v2._select_fn(fn, 4, HC, HIDDEN) is fn  # nothing registered


def test_registers_exact_copies():
    model = _Model(3)
    n, lines = _logged(v2.register_fn_bf16, model)
    assert n == 6 and len(v2._FN_BF16) == 6
    assert any("reads exact bf16 fn copies for M <= 12" in line for line in lines), lines
    for layer in model.layers:
        for p in (layer.hc_attn_fn, layer.hc_ffn_fn):
            bf = v2._fn_bf16(p)
            assert bf is not None and bf.dtype == torch.bfloat16
            assert torch.equal(bf.float(), p)
            assert v2._fn_bf16(p.contiguous()) is bf
    # not an fn: a base vector and an unrelated tensor are never looked up
    assert v2._fn_bf16(model.layers[0].hc_attn_base) is None
    assert v2._fn_bf16(torch.zeros(NOUT, HC * HIDDEN)) is None


def test_same_storage_other_shape_is_rejected():
    model = _Model(1)
    v2.register_fn_bf16(model)
    p = model.layers[0].hc_attn_fn
    assert v2._fn_bf16(p[:4]) is None


def test_non_bf16_values_close_the_gate():
    model = _Model(2, exact=False)
    n, lines = _logged(v2.register_fn_bf16, model)
    assert n == 0 and v2._FN_BF16 == {}
    assert any("gate is closed (an mHC fn holds values that are not bf16)" in line
               for line in lines), lines


def test_one_inexact_tensor_registers_none():
    model = _Model(2)
    with torch.no_grad():
        model.layers[1].hc_ffn_fn[0, 0] = 1.0 + 2.0 ** -12
    n, _ = _logged(v2.register_fn_bf16, model)
    assert n == 0 and v2._FN_BF16 == {}


def test_no_fn_closes_the_gate():
    n, lines = _logged(v2.register_fn_bf16, nn.Linear(4, 4))
    assert n == 0
    assert any("no fp32 mHC fn on this rank" in line for line in lines), lines


def test_bf16_kernels_compile_for_sm80():
    """Both kernel-A specialisations compile for sm_80 with a bf16 fn (no GPU)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource

    common = {"x_ptr": "*bf16", "res_ptr": "*bf16", "post_ptr": "*fp32",
              "comb_ptr": "*fp32", "fn_ptr": "*bf16", "rc_ptr": "*bf16",
              "mix_ptr": "*fp32", "sqr_ptr": "*fp32", "M": "i32"}
    cases = [(v2._glu_post_prenorm_tb_kernel,
              {"HIDDEN": HIDDEN, "NOUT": NOUT, "HB": 32, "BT": 8, "NP2": 32}),
             (v2._glu_post_prenorm_kernel,
              {"HIDDEN": HIDDEN, "NOUT": NOUT, "HB": 32, "BM": 16, "NP2": 32, "NW": 4})]
    for kern, consts in cases:
        params = list(kern.arg_names)
        sig = {p: common.get(p, "constexpr") for p in params}
        attrs = {(i,): [["tt.divisibility", 16]] for i, p in enumerate(params)
                 if sig[p].startswith("*")}
        c = triton.compile(GluonASTSource(kern, sig, constexprs=consts, attrs=attrs),
                           target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
        assert c.asm["cubin"]


# ---------------------------------------------------------------- GPU ------

def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 0)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")


def _case(M, fn, seed):
    from tests.kernels import test_ampere_decode_mhc_v2 as base

    c = base._case(M, seed)
    c["fn"] = fn
    return base._kw(c)


def _dispatch(monkeypatch, kw):
    from vllm.model_executor.kernels.mhc import tilelang as tlmod
    import vllm.ampere_decode as ad

    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", "1")
    # v3 (default on) takes v2's place in the dispatch; these tests are v2's
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V3", "0")
    monkeypatch.setattr(ad, "_SM80_CACHE", True)
    return tlmod.mhc_fused_post_pre_tilelang(**kw)


MS = (1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 32)


@gpu
def test_bf16_fn_bitwise_equal(monkeypatch):
    model = _Model(1, dev="cuda")
    fn = model.layers[0].hc_attn_fn
    v2.warmup(MS)
    outs = {}
    for M in MS:
        outs[M] = [t.clone() for t in _dispatch(monkeypatch, _case(M, fn, 40 + M))]
    assert v2._FN_BF16 == {}
    monkeypatch.setenv(FLAG, "1")
    assert v2.register_fn_bf16(model) == 2
    v2.warmup(MS)
    seen = []
    real = v2._select_fn

    def spy(f, M, hc, hidden):
        out = real(f, M, hc, hidden)
        seen.append((M, out.dtype))
        return out

    monkeypatch.setattr(v2, "_select_fn", spy)
    for M in MS:
        got = _dispatch(monkeypatch, _case(M, fn, 40 + M))
        assert seen[-1] == (M, torch.bfloat16 if M <= 12 else torch.float32), seen[-1]
        for a, b in zip(got, outs[M]):
            assert torch.equal(a, b), M
        # the bf16 copy passed straight to the v2 kernel entry (the TileLang
        # dispatch accepts only the model's fp32 fn) gives the same results
        direct = v2.mhc_fused_post_pre(**_case(M, v2._fn_bf16(fn), 40 + M))
        for a, b in zip(direct, outs[M]):
            assert torch.equal(a, b), M


@gpu
@pytest.mark.parametrize("M", [4, 8, 16, 32])
def test_bf16_fn_graph_replay(monkeypatch, M):
    model = _Model(1, dev="cuda")
    fn = model.layers[0].hc_ffn_fn
    v2.register_fn_bf16(model)
    v2.warmup((M,))
    kw = _case(M, fn, 90 + M)
    eager = [t.clone() for t in _dispatch(monkeypatch, kw)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = _dispatch(monkeypatch, kw)
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(outs, eager):
        assert torch.equal(a, b)
    assert torch.cuda.memory_allocated() == allocated
