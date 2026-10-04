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


# ------------------------------------------------- real-Torch scratch owners

def test_prenorm_scratch_variable_tails_bound_live_storage():
    """Actual selector tails retain three owners with current contiguous views."""
    import weakref

    from vllm.ampere_prefill.mhc_prenorm import (
        _fn_pack,
        _select_config,
        _workspace,
    )
    from vllm.ampere_prefill.mhc_prenorm_workspace import ScratchOwner

    owner = ScratchOwner("cpu")
    shapes = [(m, 2048, 24) for m in (
        1, 16, 17, 64, 65, 384, 511, 512, *range(513, 577),
        1152, 1153, 2304, 2305, 3456,
    )]
    shapes += [(17, 64, 7), (65, 128, 33), (513, 1024, 24)]
    shapes += list(reversed(shapes))
    maxima = [0, 0, 0]
    live = {}
    seen_splits = set()
    for M, K, N in shapes:
        _, bn, _, sk, _, _ = _select_config(M, K, N, 70)
        seen_splits.add(sk)
        previous = (owner.part, owner.psq, owner.pack)
        part, psq = _workspace("cpu", M, bn, sk, owner=owner)
        ft = _fn_pack("cpu", K, bn, owner=owner)
        assert part.shape == (sk, M, bn) and part.stride() == (M * bn, bn, 1)
        assert psq.shape == (sk, M) and psq.stride() == (M, 1)
        assert ft.shape == (3, K, bn) and ft.stride() == (K * bn, bn, 1)
        assert part.is_contiguous() and psq.is_contiguous() and ft.is_contiguous()
        for i, (view, storage) in enumerate(zip(
            (part, psq, ft), (owner.part, owner.psq, owner.pack)
        )):
            maxima[i] = max(maxima[i], view.numel())
            assert maxima[i] <= storage.numel() < 2 * maxima[i]
            assert view.data_ptr() == storage.data_ptr()
            if previous[i] is not None and previous[i].numel() >= view.numel():
                assert storage is previous[i]
            live[id(storage)] = weakref.ref(storage)

        del previous, part, psq, ft, storage, view
        assert sum(ref() is not None for ref in live.values()) == 3
    assert len(seen_splits) > 1


def test_prenorm_scratch_bounds_even_unreleased_growth_storage():
    """Even retaining every old allocation cannot accumulate variable tails."""
    from vllm.ampere_prefill.mhc_prenorm import (
        _fn_pack,
        _select_config,
        _workspace,
    )
    from vllm.ampere_prefill.mhc_prenorm_workspace import ScratchOwner

    owner = ScratchOwner("cpu")
    pending = [{}, {}, {}]
    for M in (1, 2, 3, 16, 17, 64, 65, 128, 129, 512, 513, 1152, 1153):
        K = 64 + M
        _, bn, _, sk, _, _ = _select_config(M, K, 24, 70)
        _workspace("cpu", M, bn, sk, owner=owner)
        _fn_pack("cpu", K, bn, owner=owner)
        for held, storage in zip(pending, (owner.part, owner.psq, owner.pack)):
            held[id(storage)] = storage
            assert sum(t.numel() for t in held.values()) < 2 * storage.numel()


def test_prenorm_scratch_growth_keeps_consumers_then_releases_storage():
    """Growth preserves held consumer views without retaining old owners."""
    import weakref

    from vllm.ampere_prefill.mhc_prenorm import _fn_pack, _workspace
    from vllm.ampere_prefill.mhc_prenorm_workspace import ScratchOwner

    owner = ScratchOwner("cpu")
    part, psq = _workspace("cpu", 3, 16, 2, owner=owner)
    ft = _fn_pack("cpu", 8, 16, owner=owner)
    part.fill_(3)
    psq.fill_(5)
    ft.fill_(7)
    old = [weakref.ref(t) for t in (owner.part, owner.psq, owner.pack)]
    bigger, bigger_sq = _workspace("cpu", 17, 32, 4, owner=owner)
    bigger_ft = _fn_pack("cpu", 17, 32, owner=owner)
    bigger.fill_(-1)
    bigger_sq.fill_(-2)
    bigger_ft.fill_(-3)
    torch.testing.assert_close(part.sum(0), torch.full((3, 16), 6.0))
    torch.testing.assert_close(psq.sum(0), torch.full((3,), 10.0))
    torch.testing.assert_close(
        torch.mv(ft[0].float().t(), torch.ones(8)), torch.full((16,), 56.0)
    )
    assert all(ref() is not None for ref in old)
    del part, psq, ft
    assert all(ref() is None for ref in old)
    current = [weakref.ref(t) for t in (owner.part, owner.psq, owner.pack)]
    del bigger, bigger_sq, bigger_ft, owner
    assert all(ref() is None for ref in current)


def test_prenorm_scratch_cpu_leases_serialize_real_consumers():
    """Canonical device leases keep concurrent writes exclusive."""
    from concurrent.futures import ThreadPoolExecutor

    from vllm.ampere_prefill.mhc_prenorm_workspace import (
        get_scratch_owner,
        scratch_owner,
    )

    assert get_scratch_owner("cpu") is get_scratch_owner("cpu:0")

    def consume(value):
        for _ in range(20):
            with scratch_owner("cpu") as owner:
                part, psq = owner.workspace(13, 16, 4)
                ft = owner.fn_pack(8, 16)
                part.fill_(value)
                psq.fill_(value + 1)
                ft.fill_(value + 2)
                torch.testing.assert_close(
                    part.sum(0), torch.full((13, 16), 4.0 * value)
                )
                torch.testing.assert_close(
                    psq.sum(0), torch.full((13,), 4.0 * (value + 1))
                )
                torch.testing.assert_close(
                    ft.float().sum(0), torch.full((8, 16), 3.0 * (value + 2))
                )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(consume, (1, 7)))


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
    attn/ffn), sharing the leased high-water pack within each eager device.
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


def _gpu_prenorm_growth_m(owner, K, N):
    from vllm.ampere_prefill.mhc_prenorm import _select_config, num_sms

    M = 1024
    capacity = 0 if owner.part is None else owner.part.numel()
    while True:
        _, bn, _, sk, _, _ = _select_config(M, K, N, num_sms(owner.device.index))
        if sk * M * bn > capacity:
            return M
        M *= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_prenorm_pending_growth_and_stream_retirement():
    """Pending old reads survive growth and repeated short-lived streams."""
    import weakref

    from vllm.ampere_prefill.mhc_prenorm import hc_prenorm_gemm
    from vllm.ampere_prefill.mhc_prenorm_workspace import (
        _OWNERS,
        get_scratch_owner,
    )

    owner = get_scratch_owner("cuda")
    dev = owner.device
    N = 24
    pack_capacity = 0 if owner.pack is None else owner.pack.numel()
    K = max(2048, 128 + pack_capacity // (3 * 32))
    large_k = 2 * K + 128
    M = _gpu_prenorm_growth_m(owner, large_k, N)
    g = torch.Generator(device=dev).manual_seed(31)
    small = torch.randn(17, K, generator=g, device=dev).to(torch.bfloat16)
    large = torch.randn(M, large_k, generator=g, device=dev).to(torch.bfloat16)
    weights = [
        torch.randn(N, k, generator=g, device=dev) / k ** 0.5
        for k in (K, large_k, K, K, K, K, K, K)
    ]
    torch.cuda.synchronize()
    ready = torch.cuda.Event()
    ready.record()
    owners_before = len(_OWNERS)
    streams = [torch.cuda.Stream() for _ in range(2)]
    with torch.cuda.stream(streams[0]):
        streams[0].wait_event(ready)
        torch.cuda._sleep(10_000_000)
        first, _ = hc_prenorm_gemm(small, weights[0])
    old = [weakref.ref(t) for t in (owner.part, owner.psq, owner.pack)]
    with torch.cuda.stream(streams[1]):
        streams[1].wait_event(ready)
        grown, _ = hc_prenorm_gemm(large, weights[1])
    del streams
    tails = []
    for fn in weights[2:]:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            result, _ = hc_prenorm_gemm(small, fn)
            tails.append(result)
        del stream
    torch.cuda.synchronize()
    assert len(_OWNERS) == owners_before
    assert get_scratch_owner(dev) is owner
    assert all(ref() is None for ref in old)
    torch.testing.assert_close(
        first[0], small.float() @ weights[0].t(), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        grown[0], large.float() @ weights[1].t(), rtol=2e-3, atol=2e-3
    )
    for result, fn in zip(tails, weights[2:]):
        torch.testing.assert_close(
            result[0], small.float() @ fn.t(), rtol=2e-3, atol=2e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_prenorm_capture_pools_survive_eager_growth():
    """Two-layer graphs own pointers across eager growth and concurrent replay.

    Independent pools replay on separate streams. Shared pools use the same
    stream and serialized replay, matching vLLM's supported pool contract.
    """
    from vllm.ampere_prefill.mhc_prenorm import hc_prenorm_gemm, warmup
    from vllm.ampere_prefill.mhc_prenorm_workspace import (
        _OWNERS,
        get_scratch_owner,
    )

    owner = get_scratch_owner("cuda")
    dev = owner.device
    M, K, N = 65, 1024, 24
    warmup((M,), K=K, N=N, device=str(dev))
    for shared in (False, True):
        x = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
        weights = [
            [torch.zeros(N, K, device=dev) for _ in range(2)]
            for _ in range(2)
        ]
        outputs = [
            [torch.empty(1, M, N, device=dev) for _ in range(2)]
            for _ in range(2)
        ]
        sums = [
            [torch.empty(1, M, device=dev) for _ in range(2)]
            for _ in range(2)
        ]
        stream = torch.cuda.Stream()
        streams = [stream, stream if shared else torch.cuda.Stream()]
        pool = torch.cuda.graph_pool_handle() if shared else None
        graphs = []
        owners_before = len(_OWNERS)
        for i, capture_stream in enumerate(streams):
            capture_stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                for layer in range(2):
                    hc_prenorm_gemm(
                        x, weights[i][layer],
                        out=outputs[i][layer], sqrsum=sums[i][layer],
                    )
            graphs.append(graph)
        torch.cuda.synchronize()
        assert len(_OWNERS) == owners_before
        x.normal_()
        for layer_weights in weights:
            for fn in layer_weights:
                fn.normal_(std=K ** -0.5)
        growth_m = _gpu_prenorm_growth_m(owner, K, N)
        eager_x = torch.ones(growth_m, K, dtype=torch.bfloat16, device=dev)
        eager_fn = torch.ones(N, K, device=dev) / K
        ready = torch.cuda.Event()
        ready.record()
        eager_stream = torch.cuda.Stream()
        with torch.cuda.stream(eager_stream):
            eager_stream.wait_event(ready)
            eager_out, _ = hc_prenorm_gemm(eager_x, eager_fn)
        for graph, replay_stream in zip(graphs, streams):
            with torch.cuda.stream(replay_stream):
                replay_stream.wait_event(ready)
                graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            eager_out[0], torch.ones(growth_m, N, device=dev),
            rtol=2e-3, atol=2e-3,
        )
        for i in range(2):
            assert not torch.equal(outputs[i][0], outputs[i][1])
            for layer in range(2):
                eager, eager_sq = hc_prenorm_gemm(x, weights[i][layer])
                torch.cuda.synchronize()
                assert torch.equal(outputs[i][layer], eager)
                assert torch.equal(sums[i][layer], eager_sq)
        mem0 = torch.cuda.memory_allocated(dev)
        for graph, replay_stream in zip(graphs, streams):
            with torch.cuda.stream(replay_stream):
                graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(dev) == mem0


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
    for fn in (
        test_split3_reconstructs_fp32,
        test_split3_is_injective_for_distinct_fn,
        test_prenorm_scratch_variable_tails_bound_live_storage,
        test_prenorm_scratch_bounds_even_unreleased_growth_storage,
        test_prenorm_scratch_growth_keeps_consumers_then_releases_storage,
        test_prenorm_scratch_cpu_leases_serialize_real_consumers,
    ):
        cases.append((fn.__name__, lambda mp, f=fn: f()))
    gpu_tests = (test_distinct_fn_give_distinct_results_on_gpu,
                 test_gpu_prenorm_accuracy_vs_tilelang_no_floor,
                 test_gpu_prenorm_graph_replay,
                 test_gpu_prenorm_pending_growth_and_stream_retirement,
                 test_gpu_prenorm_capture_pools_survive_eager_growth)
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
