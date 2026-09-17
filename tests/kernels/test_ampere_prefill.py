# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side tests for the opt-in sm_80 prefill kernels (vllm/ampere_prefill/).

Everything here runs without a GPU: the dispatch gates are pure host logic, and
the bf16 hi/mid/lo split the mHC pre-norm GEMM relies on is exercised against a
CPU reference. The GPU correctness of the kernels themselves is covered by
the standalone GPU correctness suite (37 checks, three families).

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
    if torch.cuda.is_available():
        cases.append((test_distinct_fn_give_distinct_results_on_gpu.__name__,
                      lambda mp: test_distinct_fn_give_distinct_results_on_gpu()))
    else:
        print("SKIP test_distinct_fn_give_distinct_results_on_gpu (no GPU visible)")

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
