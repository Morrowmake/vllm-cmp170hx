# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""sm_80 KDA chunked prefill with 16 heads per card (VLLM_GLM5_TP4_KDA_PREFILL).

The kernels are the ones of VLLM_GLM5_PP_KDA_PREFILL (64 heads); this file
checks the tensor-parallel-4 gate and runs the same GPU checks at 16 heads
and the TP4 chunk sizes (up to 3460 tokens). The GPU helpers (production
input layout, fp64 reference, accuracy rule) are those of
``test_ampere_pp_kda_prefill.py``, loaded as a separate module with the head
count set to 16.

CPU tests (run with ``CUDA_VISIBLE_DEVICES=""``): the flag is declared and off
by default; the 16-head layer gate and the 3460-token call gate open only
under this flag, and the 64-head gates are unchanged; the warm-up hook
compiles at the layer's head count.

GPU tests (skip without an sm_80 device): gate closed == upstream bitwise;
error against an exact fp64 token-sequential recomputation at most the
upstream path's (mean <= 1.10x, max <= 1.25x per output, as in the PP file)
on production-magnitude random inputs and awkward shapes, and, when
``KDA_PRE_CAPTURE_DIR`` points at captured live TP4 prefill calls (16
heads), on real inputs against the live outputs; determinism; CUDA-graph
replay bitwise equal to eager with no allocation growth.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_tp4_kda_prefill.py
    KDA_PRE_CAPTURE_DIR=<dir> pytest -q tests/kernels/test_ampere_tp4_kda_prefill.py
"""

import importlib.util
import os

import pytest
import torch

from vllm.ampere_prefill import kda_prefill as kp

FLAG = "VLLM_GLM5_TP4_KDA_PREFILL"
PP_FLAG = "VLLM_GLM5_PP_KDA_PREFILL"
H, D = kp.TP4_HEADS, kp.HEAD_DIM
BF16, F32 = torch.bfloat16, torch.float32


def _helpers():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_ampere_pp_kda_prefill.py")
    spec = importlib.util.spec_from_file_location("_kda_prefill_helpers_h16", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.H, mod.P = H, H * D
    return mod


hp = _helpers()
IS_SM80 = hp.IS_SM80
needs_sm80 = hp.needs_sm80


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


@pytest.mark.parametrize("pp,tp4,want", [
    ("0", "0", ()), ("1", "0", (64,)), ("0", "1", (16,)), ("1", "1", (64, 16)),
])
def test_enabled_heads(monkeypatch, pp, tp4, want):
    monkeypatch.setenv(PP_FLAG, pp)
    monkeypatch.setenv(FLAG, tp4)
    assert kp.enabled_heads() == want


# ------------------------------------------------------------------ CPU: gates
OPEN_LAYER = dict(backend="triton", num_heads=16, head_dim=128, dtype=BF16,
                  safe_gate=True, lower_bound=-5.0, capability=(8, 0),
                  allowed_heads=(16,))


def test_layer_gate_open():
    assert kp.layer_closed_reason(**OPEN_LAYER) == ""
    assert kp.layer_closed_reason(**dict(OPEN_LAYER, allowed_heads=(64, 16))) == ""


@pytest.mark.parametrize("change,needle", [
    (dict(allowed_heads=(64,)), "16 heads"),          # only the PP flag set
    (dict(allowed_heads=()), "16 heads"),
    (dict(num_heads=64), "64 heads"),                 # 64 heads, only the TP4 flag set
    (dict(num_heads=32), "32 heads"),
    (dict(num_heads=8, allowed_heads=(64, 16)), "8 heads"),
    (dict(capability=(8, 6)), "sm_80"),
    (dict(backend="flashkda"), "backend"),
    (dict(head_dim=64), "head_dim"),
    (dict(dtype=torch.float16), "dtype"),
    (dict(safe_gate=False), "safe"),
    (dict(lower_bound=-3.0), "lower_bound"),
])
def test_layer_gate_closed(change, needle):
    why = kp.layer_closed_reason(**{**OPEN_LAYER, **change})
    assert why and needle in why


def test_pp_layer_gate_unchanged():
    base = dict(OPEN_LAYER, num_heads=64)
    base.pop("allowed_heads")
    assert kp.layer_closed_reason(**base) == ""
    why = kp.layer_closed_reason(**dict(base, num_heads=16))
    assert why == "16 heads on this rank (needs 64: pipeline parallel, TP=1)"


OPEN_CALL = dict(num_tokens=3456, num_seqs=1, qkv_dtype=BF16, g_dtype=BF16,
                 state_dtype=F32, a_log_dtype=F32, bias_dtype=F32)


@pytest.mark.parametrize("num_tokens,num_seqs", [
    (3456, 1), (3460, 1), (1282, 1), (1, 1), (2312, 1), (2313, 1), (3460, 16),
])
def test_call_gate_open_16_heads(num_tokens, num_seqs):
    c = dict(OPEN_CALL, num_tokens=num_tokens, num_seqs=num_seqs)
    assert kp.call_closed_reason(**c, max_tokens=kp.TP4_MAX_TOKENS) == ""

    def upstream(**kw):
        raise AssertionError("not called")

    fn = kp.select_chunk_fn(upstream, *c.values(), num_heads=16)
    assert fn is kp.chunk_kda_with_fused_gate


@pytest.mark.parametrize("change", [
    dict(num_tokens=0), dict(num_tokens=3461), dict(num_tokens=4608),
    dict(num_seqs=0), dict(num_seqs=17), dict(qkv_dtype=torch.float16),
    dict(state_dtype=BF16), dict(bias_dtype=None),
])
def test_call_gate_closed_16_heads(change):
    c = {**OPEN_CALL, **change}
    assert kp.call_closed_reason(**c, max_tokens=kp.TP4_MAX_TOKENS) != ""

    def upstream(**kw):
        raise AssertionError("not called")

    assert kp.select_chunk_fn(upstream, *c.values(), num_heads=16) is upstream


def test_pp_call_gate_unchanged():
    def upstream(**kw):
        raise AssertionError("not called")

    # 64 heads keep the 2312-token limit, with or without the keyword
    for kw in ({}, dict(num_heads=64)):
        c = dict(OPEN_CALL, num_tokens=2313)
        assert kp.select_chunk_fn(upstream, *c.values(), **kw) is upstream
        c = dict(OPEN_CALL, num_tokens=2312)
        assert kp.select_chunk_fn(upstream, *c.values(), **kw) is kp.chunk_kda_with_fused_gate


def test_use_for_layer(monkeypatch):
    from vllm.platforms import current_platform

    class Cap:
        major, minor = 8, 0

    monkeypatch.setattr(current_platform, "get_device_capability", lambda *a, **k: Cap)
    assert kp.use_for_layer("triton", 16, 128, BF16, True, -5.0, allowed_heads=(16,)) is True
    assert kp.use_for_layer("triton", 16, 128, BF16, True, -5.0, allowed_heads=(64,)) is False
    assert kp.use_for_layer("triton", 16, 128, BF16, True, -5.0) is False
    assert kp.use_for_layer("triton", 64, 128, BF16, True, -5.0, allowed_heads=(16,)) is False
    assert kp.use_for_layer("triton", 64, 128, BF16, True, -5.0, allowed_heads=(64, 16)) is True
    monkeypatch.setattr(current_platform, "get_device_capability", lambda *a, **k: None)
    assert kp.use_for_layer("triton", 16, 128, BF16, True, -5.0, allowed_heads=(16,)) is False


def test_warmup_hook_noop_without_flags(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(PP_FLAG, raising=False)

    class Worker:
        def get_model(self):
            raise AssertionError("must not touch the model with the flags off")

    kp.warmup_from_worker(Worker())


def test_warmup_hook_uses_layer_heads(monkeypatch):
    monkeypatch.delenv(PP_FLAG, raising=False)
    monkeypatch.setenv(FLAG, "1")

    class Layer(torch.nn.Module):
        _pp_kda_prefill = True
        local_num_heads = 16

    class Worker:
        device = "cpu"

        def get_model(self):
            return torch.nn.Sequential(Layer())

    called = []
    monkeypatch.setattr(kp, "warmup", lambda plans, device=None, heads=None: called.append(heads))
    kp.warmup_from_worker(Worker())
    assert called == [16]


def test_call_site_passes_head_count():
    import vllm.models.glm5next.common.kda as kda_mod

    src = open(kda_mod.__file__).read()
    assert "num_heads=self.local_num_heads," in src
    assert "allowed_heads=enabled_heads()," in src
    assert "_envs.VLLM_GLM5_PP_KDA_PREFILL or _envs.VLLM_GLM5_TP4_KDA_PREFILL" in src


# ------------------------------------------------------------------ GPU
@pytest.fixture(scope="module")
def warmed():
    if not IS_SM80:
        pytest.skip("needs an sm_80 GPU")
    kp.warmup(kp.WARMUP_PLANS, heads=H)
    return True


@needs_sm80
@pytest.mark.parametrize("seqlens,init", [([3456], "real"), ([3456], "zero"),
                                          ([1282], "real"), ([864] * 4, "mixed")])
def test_gate_closed_is_upstream_bitwise(seqlens, init):
    up = hp.upstream_fn()
    c = hp.make_case(seqlens, init, seed=7)
    ref_out = hp.call(up, c)
    fn = kp.select_chunk_fn(up, 3461, len(seqlens), BF16, BF16, F32, F32, F32, num_heads=16)
    assert fn is up
    out = hp.call(fn, c)
    torch.cuda.synchronize()
    assert torch.equal(out[0], ref_out[0]) and torch.equal(out[1], ref_out[1])


ACCURACY_CASES = [
    ("cont3456", [3456], "real", {}),
    ("first3456", [3456], "zero", {}),
    ("tail1282", [1282], "real", {}),
    ("t3460", [3460], "real", {}),
    ("cont2304", [2304], "real", {}),
    ("varlen4", [864] * 4, "mixed", {}),
    ("varlen2_unaligned", [1500, 1956], "mixed", {}),
    ("varlen16", [216] * 16, "mixed", {}),
    ("t1", [1], "real", {}),
    ("t63", [63], "zero", {}),
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
    c = hp.make_case(seqlens, init, seed=seed, **kw)
    keep = {k: c[k].clone() for k in ("buf", "raw_g", "beta", "initial_state",
                                      "A_log", "g_bias", "cu_seqlens")}
    inc = hp.call(hp.upstream_fn(), c)
    new = hp.call(kp.chunk_kda_with_fused_gate, c)
    new2 = hp.call(kp.chunk_kda_with_fused_gate, c)
    torch.cuda.synchronize()
    T, N = sum(seqlens), len(seqlens)
    assert new[0].shape == (1, T, H, D) and new[0].dtype == BF16
    assert new[1].shape == (N, H, D, D) and new[1].dtype == F32
    assert bool(torch.isfinite(new[0]).all() and torch.isfinite(new[1]).all())
    assert torch.equal(new[0], new2[0]) and torch.equal(new[1], new2[1]), "not deterministic"
    for k_, v_ in keep.items():
        assert torch.equal(c[k_], v_), f"input {k_} modified"
    res = hp.check_accuracy(new, inc, hp.reference_fp64(c))
    assert all(r[2] for r in res.values()), f"{name}: {res}"


def _tp4_capture_files():
    files = hp._capture_files()
    out = []
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        if r["q"].shape[2] == H:
            out.append(f)
    return out


@needs_sm80
@pytest.mark.skipif(not hp._capture_files(), reason="KDA_PRE_CAPTURE_DIR not set or empty")
def test_real_captured_inputs(warmed):
    """TP4 records (16 heads) only; per class (first / cont / multi) and over
    all: sum of per-call mean errors <= 1.10x, average and worst per-call max
    <= 1.25x the live (upstream) outputs' errors against fp64."""
    files = _tp4_capture_files()
    if not files:
        pytest.skip("no 16-head records in KDA_PRE_CAPTURE_DIR")
    agg = {}
    for f in files:
        c, live, cls = hp.load_record(f)
        new = hp.call(kp.chunk_kda_with_fused_gate, c)
        ref = hp.reference_fp64(c)
        torch.cuda.synchronize()
        for name, y, yi, r in (("o", new[0], live[0], ref[0]),
                               ("state", new[1], live[1], ref[1])):
            cmx, cmn = hp.errors(y, r)
            imx, imn = hp.errors(yi, r)
            for key in ((cls, name), ("all", name)):
                a = agg.setdefault(key, [0.0] * 6)
                a[0] += cmn; a[1] += imn; a[2] += cmx; a[3] += imx  # noqa: E702
                a[4] = max(a[4], cmx); a[5] = max(a[5], imx)  # noqa: E702
        del c, live, new, ref
    bad = {}
    for key, (gm, tm, gx, tx, gw, tw) in agg.items():
        rm, rx, rw = hp.ratio(gm, tm), hp.ratio(gx, tx), hp.ratio(gw, tw)
        if not (rm <= hp.MEAN_RATIO and rx <= hp.MAX_RATIO and rw <= hp.MAX_RATIO):
            bad[key] = (rm, rx, rw)
    assert not bad, bad


@needs_sm80
@pytest.mark.parametrize("seqlens,init", [([3456], "real"), ([864] * 4, "mixed")])
def test_cuda_graph_replay_bitwise_no_growth(warmed, seqlens, init):
    c = hp.make_case(seqlens, init, seed=61)
    fn = kp.chunk_kda_with_fused_gate
    for _ in range(2):
        eager = hp.call(fn, c)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        hp.call(fn, c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = hp.call(fn, c)
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
