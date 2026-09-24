# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side tests for the opt-in sm_80 prefill kernels (vllm/ampere_prefill/).

Everything but the GPU tests at the end runs without a GPU: the dispatch gates
are pure host logic, and the bf16 hi/mid/lo split the mHC pre-norm GEMM relies
on is exercised against a CPU reference. The GPU tests (skipped without a GPU;
the standalone runner below hides the GPU unless CUDA_VISIBLE_DEVICES is set)
gate the pre-norm GEMM's error against an fp64 recomputation relative to the
TileLang kernel it replaces, with no ulp floor, and its CUDA-graph replay. The
full GPU correctness of the kernels is covered by the standalone GPU
correctness suite.

    pytest -q tests/kernels/test_ampere_prefill.py
"""

try:
    import pytest
except ImportError:  # the vllm-dev venv has no pytest; see __main__ below
    pytest = None
import torch

if pytest is None:  # minimal stand-ins so the module still imports and runs
    class _Mark:
        def __getattr__(self, _name):
            return lambda *a, **k: (lambda f: f)

    class _Pytest:
        mark = _Mark()

        @staticmethod
        def fixture(f):
            return f

    pytest = _Pytest()

from vllm.model_executor.kernels.mhc.tilelang import _use_ampere_prefill_prenorm
from vllm.v1.attention.ops.triton_mla_sparse import _use_ampere_prefill_sparse_mla


class _FakePlatform:
    def __init__(self, cuda=True, cap=80):
        self._cuda = cuda
        self._cap = cap

    def is_cuda(self):
        return self._cuda

    def is_device_capability(self, cap):
        return self._cuda and self._cap == cap


def _as_sm80(monkeypatch):
    """Point vllm at a fake sm_80 CUDA platform, without touching a real device."""
    monkeypatch.setattr("vllm.platforms.current_platform", _FakePlatform())
    return monkeypatch


@pytest.fixture
def sm80(monkeypatch):
    return _as_sm80(monkeypatch)


# ----------------------------------------------------------------- gates OFF

@pytest.mark.parametrize("num_tokens", [1, 4, 64, 512, 1152, 2304])
@pytest.mark.parametrize("seq_kv", [1024, 2304, 8192, 99328])
def test_gates_are_off_by_default(sm80, num_tokens, seq_kv):
    """With VLLM_GLM5_PREFILL_KERNELS unset, nothing dispatches -- ever."""
    sm80.delenv("VLLM_GLM5_PREFILL_KERNELS", raising=False)
    assert _use_ampere_prefill_prenorm(num_tokens) is False
    assert _use_ampere_prefill_sparse_mla(num_tokens, seq_kv, 2048) is False


def test_gates_off_on_non_sm80(monkeypatch):
    """Even with the flag on, a non-sm_80 part keeps the upstream kernels."""
    monkeypatch.setenv("VLLM_GLM5_PREFILL_KERNELS", "1")
    monkeypatch.setattr("vllm.platforms.current_platform", _FakePlatform(cap=90))
    assert _use_ampere_prefill_prenorm(1152) is False
    assert _use_ampere_prefill_sparse_mla(1152, 99328, 2048) is False

    monkeypatch.setattr("vllm.platforms.current_platform", _FakePlatform(cuda=False))
    assert _use_ampere_prefill_prenorm(1152) is False
    assert _use_ampere_prefill_sparse_mla(1152, 99328, 2048) is False


# ------------------------------------------------------------------ gates ON

def test_prenorm_token_threshold(sm80):
    sm80.setenv("VLLM_GLM5_PREFILL_KERNELS", "1")
    sm80.setenv("VLLM_GLM5_PREFILL_MIN_TOKENS", "512")
    # decode sizes -- these are the captured CUDA-graph sizes, must stay upstream
    for n in (1, 2, 4, 8, 16, 32, 64, 511):
        assert _use_ampere_prefill_prenorm(n) is False, n
    # the production prefill chunk is 1152
    for n in (512, 1152, 1153, 2304):
        assert _use_ampere_prefill_prenorm(n) is True, n


def test_sparse_mla_context_gate(sm80):
    """The measured regression case (ctx ~= topk) must fall back."""
    sm80.setenv("VLLM_GLM5_PREFILL_KERNELS", "1")
    sm80.setenv("VLLM_GLM5_PREFILL_MIN_TOKENS", "512")
    sm80.setenv("VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT", "2.0")
    topk = 2048
    # 0.93x measured here -- must NOT dispatch
    assert _use_ampere_prefill_sparse_mla(1152, 2304, topk) is False
    assert _use_ampere_prefill_sparse_mla(1152, 4095, topk) is False
    # 1.39-1.42x measured here -- must dispatch
    assert _use_ampere_prefill_sparse_mla(1152, 4096, topk) is True
    assert _use_ampere_prefill_sparse_mla(1152, 8192, topk) is True
    assert _use_ampere_prefill_sparse_mla(1152, 99328, topk) is True
    # decode stays upstream whatever the context
    assert _use_ampere_prefill_sparse_mla(4, 99328, topk) is False


def test_sparse_mla_gate_scales_with_topk(sm80):
    """The gate is relative to index_topk, not an absolute row count."""
    sm80.setenv("VLLM_GLM5_PREFILL_KERNELS", "1")
    sm80.setenv("VLLM_GLM5_PREFILL_MIN_TOKENS", "512")
    assert _use_ampere_prefill_sparse_mla(1152, 4096, 4096) is False
    assert _use_ampere_prefill_sparse_mla(1152, 8192, 4096) is True


def test_thresholds_are_configurable(sm80):
    sm80.setenv("VLLM_GLM5_PREFILL_KERNELS", "1")
    sm80.setenv("VLLM_GLM5_PREFILL_MIN_TOKENS", "2048")
    assert _use_ampere_prefill_prenorm(1152) is False
    sm80.setenv("VLLM_GLM5_PREFILL_MIN_TOKENS", "1024")
    assert _use_ampere_prefill_prenorm(1152) is True
    sm80.setenv("VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT", "0.0")
    assert _use_ampere_prefill_sparse_mla(1152, 2304, 2048) is True


# ------------------------------------------------- the bf16 hi/mid/lo split

def _split3(fn: torch.Tensor):
    """CPU reference for the pack in ampere_prefill/mhc_prenorm.py."""
    hi = fn.to(torch.bfloat16)
    r1 = fn - hi.float()
    mid = r1.to(torch.bfloat16)
    lo = (r1 - mid.float()).to(torch.bfloat16)
    return hi, mid, lo


def test_split3_reconstructs_fp32():
    """hi+mid+lo must carry ~24 significand bits, i.e. fp32 to a 2^-25 residual."""
    g = torch.Generator().manual_seed(0)
    fn = torch.randn(24, 16384, generator=g, dtype=torch.float32)
    hi, mid, lo = _split3(fn)
    rec = hi.float() + mid.float() + lo.float()
    rel = ((rec - fn).abs() / fn.abs().clamp_min(1e-30)).max().item()
    assert rel < 2.0**-24, rel
    # and a single bf16 would be ~2^-8: the split is doing real work
    rel1 = ((hi.float() - fn).abs() / fn.abs().clamp_min(1e-30)).max().item()
    assert rel1 > 2.0**-9


def test_split3_is_injective_for_distinct_fn():
    """PORT HAZARD GUARD.

    There are 90 distinct `fn` tensors per prefill chunk (45 layers x
    attn/ffn) and they all share the pack buffer's (device, K, BLOCK_N) key.
    The pack is re-run on every call precisely so that is safe. If anyone ever
    turns that scratch buffer into a shape-keyed memo, every layer after the
    first would silently reuse the first layer's weights -- and nothing in the
    numerics would look obviously wrong, because the result stays finite and
    well-scaled. This asserts the property that would break.
    """
    g = torch.Generator().manual_seed(1)
    a = torch.randn(24, 4096, generator=g, dtype=torch.float32)
    b = torch.randn(24, 4096, generator=g, dtype=torch.float32)
    assert not torch.equal(a, b)
    for xa, xb in zip(_split3(a), _split3(b)):
        assert not torch.equal(xa, xb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_distinct_fn_give_distinct_results_on_gpu():
    """The same hazard, end to end, on the real kernel.

    Deliberately calls hc_prenorm_gemm back-to-back with two different `fn` and
    the same `x`, which is exactly the 90-calls-per-chunk pattern.
    """
    from vllm.ampere_prefill.mhc_prenorm import hc_prenorm_gemm

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(2)
    x = torch.randn(1152, 16384, generator=g, device=dev,
                    dtype=torch.float32).to(torch.bfloat16)
    fa = torch.randn(24, 16384, generator=g, device=dev, dtype=torch.float32)
    fb = torch.randn(24, 16384, generator=g, device=dev, dtype=torch.float32)

    ya, _ = hc_prenorm_gemm(x, fa)
    yb, _ = hc_prenorm_gemm(x, fb)
    torch.cuda.synchronize()
    assert not torch.allclose(ya, yb)
    torch.testing.assert_close(ya[0], x.float() @ fa.t(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(yb[0], x.float() @ fb.t(), rtol=2e-3, atol=2e-3)

    # and re-running the first one still gives the first answer
    ya2, _ = hc_prenorm_gemm(x, fa)
    torch.cuda.synchronize()
    assert torch.equal(ya, ya2)


def _prenorm_cases(dev):
    """(name, x, fn) inputs shaped like the prefill calls: unnormalised residual
    streams with a per-token scale spread, and the same with a few massive
    channels (real activations have them); fn ~ N(0, 1/K)."""
    g = torch.Generator(device=dev).manual_seed(11)
    K = 16384
    fn = torch.randn(24, K, generator=g, device=dev, dtype=torch.float32) / K ** 0.5
    cases = []
    for M in (384, 1152, 3456):
        scale = 0.25 * 16.0 ** torch.rand(M, 1, generator=g, device=dev)
        x = torch.randn(M, K, generator=g, device=dev, dtype=torch.float32) * scale
        cases.append((f"spread_m{M}", x.to(torch.bfloat16), fn))
        xo = x.clone()
        ch = torch.randint(0, K, (8,), generator=g, device=dev)
        xo[:, ch] *= 200.0
        cases.append((f"outlier_m{M}", xo.to(torch.bfloat16), fn))
    return cases


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_prenorm_accuracy_vs_tilelang_no_floor():
    """Error against fp64 relative to TileLang's, NO ulp floor.

    The first version chained every k block through the tensor-core mma
    accumulator, which truncates on sm_80: 23x TileLang's mean error on the
    mixes of a real 16K-token prefill (9-31x here), while passing every
    absolute-tolerance check. Gate, for the mixes and for sqrsum: per case mean
    <= 1.10x TileLang's and max <= 1.50x; summed over the spread cases, max
    <= 1.10x. The outlier cases' max is looser on purpose: a 200x channel in
    a 16-wide mma k group sets the alignment the other 15 products are
    truncated to, which a scalar FMA chain does not do, so this kernel's
    worst element there is up to ~1.4x TileLang's (its mean stays <= 1.0x).
    On real inputs the per-element tail is 0.83x on average and is gated at
    1.25x by the standalone suite's real-input check.
    """
    from vllm.ampere_prefill.mhc_prenorm import hc_prenorm_gemm
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
    )

    dev = torch.device("cuda")
    bad = []
    sums = {"out": [0.0, 0.0], "sqrsum": [0.0, 0.0]}
    for name, x, fn in _prenorm_cases(dev):
        M, N = x.shape[0], fn.shape[0]
        o_t = torch.empty(1, M, N, dtype=torch.float32, device=dev)
        s_t = torch.empty(1, M, dtype=torch.float32, device=dev)
        _HC_PRENORM_GEMM_TILELANG_KERNEL(x, fn, o_t, s_t, 4096, 4)
        o_k, s_k = hc_prenorm_gemm(x, fn)
        torch.cuda.synchronize()
        xd = x.double()
        ref = {"out": xd @ fn.double().t(), "sqrsum": xd.square().sum(-1)}
        for key, got, tl_ in (("out", o_k[0], o_t[0]), ("sqrsum", s_k[0], s_t[0])):
            e_k = (got.double() - ref[key]).abs()
            e_t = (tl_.double() - ref[key]).abs()
            r_mean = float(e_k.mean() / e_t.mean())
            r_max = float(e_k.max() / e_t.max())
            if name.startswith("spread"):
                sums[key][0] += float(e_k.max())
                sums[key][1] += float(e_t.max())
            if r_mean > 1.10 or r_max > 1.50:
                bad.append(f"{name} {key}: mean {r_mean:.2f}x max {r_max:.2f}x")
    for key, (k_sum, t_sum) in sums.items():
        if k_sum > 1.10 * t_sum:
            bad.append(f"{key}: summed max {k_sum / t_sum:.2f}x")
    assert not bad, "; ".join(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_prenorm_graph_replay():
    """Captured in a CUDA graph, replayed on new inputs: bitwise equal to the
    eager call, and replays allocate nothing."""
    from vllm.ampere_prefill.mhc_prenorm import hc_prenorm_gemm, warmup

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(12)
    K, N = 16384, 24
    for M in (384, 1152):
        warmup((M,), K=K, N=N, device=str(dev))
        x = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
        fn = torch.zeros(N, K, dtype=torch.float32, device=dev)
        out = torch.empty(1, M, N, dtype=torch.float32, device=dev)
        sq = torch.empty(1, M, dtype=torch.float32, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            hc_prenorm_gemm(x, fn, out=out, sqrsum=sq)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hc_prenorm_gemm(x, fn, out=out, sqrsum=sq)
        torch.cuda.synchronize()
        for _ in range(3):
            x.copy_((torch.randn(M, K, generator=g, device=dev) * 3.0).to(torch.bfloat16))
            fn.copy_(torch.randn(N, K, generator=g, device=dev) / K ** 0.5)
            torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated(dev)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(dev) == mem0, M
            e_out, e_sq = hc_prenorm_gemm(x, fn)
            torch.cuda.synchronize()
            assert torch.equal(out, e_out), M
            assert torch.equal(sq, e_sq), M


# --------------------------------------------------------------------------
# Standalone runner: `python tests/kernels/test_ampere_prefill.py`.
# The vllm-dev venv has no pytest and this suite must be runnable there, so
# this reimplements just enough of monkeypatch/parametrize to execute it.
# Under real pytest this block does not run.
# --------------------------------------------------------------------------
class _MonkeyPatch:
    def __init__(self):
        self._env = []
        self._attr = []

    def setenv(self, k, v):
        import os
        self._env.append((k, os.environ.get(k)))
        os.environ[k] = str(v)

    def delenv(self, k, raising=True):
        import os
        self._env.append((k, os.environ.get(k)))
        os.environ.pop(k, None)

    def setattr(self, target, value):
        import importlib
        mod, _, attr = target.rpartition(".")
        m = importlib.import_module(mod)
        self._attr.append((m, attr, getattr(m, attr)))
        setattr(m, attr, value)

    def undo(self):
        import os
        for k, v in reversed(self._env):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for m, a, v in reversed(self._attr):
            setattr(m, a, v)
        self._env, self._attr = [], []



# ------------------------------------------------- sparse-MLA prefill schedule
def test_sparse_mla_prefill_schedule_by_head_count():
    """16 heads (TP=4) keep the measured 2-warp schedule; a 32-row head tile
    (all 64 heads on one card) takes 4 warps so its accumulator does not spill."""
    from vllm.ampere_prefill.sparse_prefill_mla import _select_config

    for tokens in (384, 1152, 2304):
        assert _select_config(tokens, 2048, 16, 512, 70) == (16, 32, 1, 2, 2)
        assert _select_config(tokens, 2048, 64, 512, 70) == (32, 32, 1, 4, 2)


# --------------------------------------------------- sparse-MLA decode schedule
def test_sparse_mla_decode_schedule_by_head_count(monkeypatch):
    """16 heads (TP=4) keep the retuned schedule exactly; 64 heads on one card
    take the measured wide-head table up to 16 rows and the rule past it."""
    import vllm.v1.attention.ops.triton_mla_sparse as m

    monkeypatch.setattr(m, "_smem_budget", lambda _d: 166912)
    monkeypatch.setattr(m, "_num_sms", lambda _d: 70)
    tp4 = {1: (16, 64, 8, 4, 2), 2: (16, 64, 8, 4, 2), 4: (16, 64, 8, 4, 2),
           8: (16, 64, 8, 4, 2), 12: (16, 64, 4, 4, 2), 16: (16, 64, 4, 4, 2),
           24: (16, 64, 2, 4, 2), 32: (16, 64, 2, 4, 2), 64: (16, 64, 1, 4, 2)}
    for rows, cfg in tp4.items():
        assert m._pick_config(rows, 2048, 16, 512, 0) == cfg, rows
    wide = {1: (16, 64, 4, 4, 2), 4: (16, 64, 4, 4, 2), 8: (16, 32, 4, 4, 2),
            12: (32, 32, 4, 4, 2), 16: (32, 32, 4, 4, 2), 32: (32, 64, 1, 4, 2)}
    for rows, cfg in wide.items():
        assert m._pick_config(rows, 2048, 64, 512, 0) == cfg, rows
    # A part that cannot stage 64 keys twice falls back to the rule.
    monkeypatch.setattr(m, "_smem_budget", lambda _d: 101376)
    assert m._pick_config(4, 2048, 64, 512, 0)[:2] == (32, 32)


def _main():
    import os
    import sys
    import traceback

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    cases = []
    for n in (1, 4, 64, 512, 1152, 2304):
        for s_kv in (1024, 2304, 8192, 99328):
            cases.append((f"test_gates_are_off_by_default[{n},{s_kv}]",
                          lambda mp, n=n, s=s_kv: test_gates_are_off_by_default(
                              _as_sm80(mp), n, s)))
    for fn in (test_gates_off_on_non_sm80,):
        cases.append((fn.__name__, lambda mp, f=fn: f(mp)))
    for fn in (test_prenorm_token_threshold, test_sparse_mla_context_gate,
               test_sparse_mla_gate_scales_with_topk, test_thresholds_are_configurable):
        cases.append((fn.__name__, lambda mp, f=fn: f(_as_sm80(mp))))
    for fn in (test_split3_reconstructs_fp32, test_split3_is_injective_for_distinct_fn):
        cases.append((fn.__name__, lambda mp, f=fn: f()))
    gpu_tests = (test_distinct_fn_give_distinct_results_on_gpu,
                 test_gpu_prenorm_accuracy_vs_tilelang_no_floor,
                 test_gpu_prenorm_graph_replay)
    for fn in gpu_tests:
        if torch.cuda.is_available():
            cases.append((fn.__name__, lambda mp, f=fn: f()))
        else:
            print(f"SKIP {fn.__name__} (no GPU visible)")

    failed = 0
    for name, run in cases:
        mp = _MonkeyPatch()
        try:
            run(mp)
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
