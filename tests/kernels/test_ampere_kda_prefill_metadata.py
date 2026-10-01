# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk metadata of the KDA chunked prefill: ``prepare_chunk_indices`` and
the ``tensor_cache`` in front of it, and the sm_80 fused path
(``vllm.ampere_prefill.kda_prefill``) on resumed-prefill batches.

CPU tests (``CUDA_VISIBLE_DEVICES=""``):
  * chunk indices name the sequence of ``cu_seqlens`` they belong to, also
    after zero-length sequences;
  * the cache returns a stored result only while every tensor argument is the
    same object with the same contents (an in-place rewrite recomputes), and
    accepts inference tensors;
  * the fused path refuses an initial state with fewer rows than sequences;
  * the fused kernels, run in the Triton interpreter (a child process with
    ``TRITON_INTERPRET=1``) on resumed-prefill batches (non-zero initial state,
    decode rows, zero-length rows in the middle and at the back): every
    buffer they allocate starts as NaN with NaN guard zones around it, the
    guards stay intact (no write outside an allocation), outputs are finite
    and each sequence's output and final state equal the same sequence run
    alone (bitwise). Addressing only: the interpreter does not reproduce the
    bf16 split products faithfully, so numerics are checked on the GPU.

GPU tests (skip without an sm_80 device): the same batch-vs-alone bitwise
check and guard check at 64 and 16 heads with production chunk sizes plus an
exact fp64 recurrence on the small batches, for the fused path; the upstream
chunk path against the fp64 recurrence.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_kda_prefill_metadata.py
"""

import importlib.util
import itertools
import json
import os
import subprocess
import sys
import types

import pytest
import torch

HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")
D = 128
LB = -5.0
GUARD = 4096


def _no_pin(monkeypatch):
    import vllm.utils.torch_utils as tu

    monkeypatch.setattr(tu, "PIN_MEMORY", False)


def cu_from(seqlens, device="cpu"):
    return torch.tensor([0] + list(itertools.accumulate(seqlens)),
                        dtype=torch.int32, device=device)


def brute_chunk_indices(seqlens, bt):
    return [[n, i] for n, L in enumerate(seqlens) for i in range(-(-L // bt))]


# ------------------------------------------------------------ CPU: chunk indices
@pytest.mark.parametrize("seqlens", [
    [64], [1], [65, 63], [2304], [1, 1, 1, 2304], [5, 0, 64], [0, 130],
    [70, 0, 5, 130], [0, 0, 1], [3, 0, 0, 200, 0], [2304, 0, 0, 0], [0], [],
])
def test_prepare_chunk_indices_names_cu_seqlens_rows(monkeypatch, seqlens):
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    _no_pin(monkeypatch)
    cu = cu_from(seqlens)
    ci = prepare_chunk_indices(cu, 64)
    assert ci.dtype == cu.dtype and ci.shape == (len(brute_chunk_indices(seqlens, 64)), 2)
    assert ci.tolist() == brute_chunk_indices(seqlens, 64)
    # chunk_offsets[n] + i is the row of (n, i): the state pass relies on it
    off = prepare_chunk_offsets(cu, 64).tolist()
    for row, (n, i) in enumerate(ci.tolist()):
        assert off[n] + i == row


def test_chunk_indices_unchanged_without_empty_sequences():
    from vllm.third_party.flash_linear_attention.ops.index import (
        chunk_indices_from_counts,
    )

    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        counts = torch.randint(1, 40, (int(torch.randint(1, 20, (1,), generator=g)),),
                               generator=g).tolist()
        indices = torch.cat([torch.arange(n) for n in counts])
        old = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        assert torch.equal(chunk_indices_from_counts(counts), old)


# ------------------------------------------------------------ CPU: the cache
def _counted(fn):
    from vllm.third_party.flash_linear_attention.ops.utils import tensor_cache

    calls = []

    @tensor_cache
    def wrapped(x, n):
        calls.append(1)
        return fn(x, n)

    return wrapped, calls


def test_cache_hit_for_same_unchanged_object():
    f, calls = _counted(lambda x, n: x.sum() * n)
    x = torch.arange(9, dtype=torch.int32)
    a = f(x, 2)
    b = f(x, 2)
    assert a is b and len(calls) == 1


def test_cache_misses_after_in_place_rewrite():
    """A persistent buffer rewritten in place between steps must not get the
    result computed for its old contents."""
    f, calls = _counted(lambda x, n: x.clone() * n)
    buf = torch.tensor([0, 64, 128], dtype=torch.int32)
    first = f(buf, 1)
    buf.copy_(torch.tensor([0, 5, 69], dtype=torch.int32))
    second = f(buf, 1)
    assert len(calls) == 2
    assert second.tolist() == [0, 5, 69] and first.tolist() == [0, 64, 128]


def test_cache_misses_after_in_place_rewrite_of_base_through_view():
    f, calls = _counted(lambda x, n: x.clone())
    base = torch.zeros(9, dtype=torch.int32)
    view = base[:4]
    f(view, 0)
    base.copy_(torch.arange(9, dtype=torch.int32))
    assert f(view, 0).tolist() == [0, 1, 2, 3] and len(calls) == 2


def test_cache_runner_pattern_fresh_view_per_step(monkeypatch):
    """The model runner's pattern: one persistent buffer, a fresh slice per
    step. Every step must see its own contents (no zero-length rows here:
    those are covered by the chunk-index tests above)."""
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices

    _no_pin(monkeypatch)
    buf = torch.zeros(17, dtype=torch.int32)
    for step, seqlens in enumerate([[2304], [1, 1, 2304], [70, 5], [130], [2304], [1, 1, 2304]]):
        n = len(seqlens)
        buf[: n + 1].copy_(cu_from(seqlens))
        buf[n + 1:] = buf[n]
        cu = buf[: n + 1]
        for _layer in range(3):
            assert prepare_chunk_indices(cu, 64).tolist() == brute_chunk_indices(seqlens, 64)


def test_cache_same_object_rewritten_between_layers_of_steps(monkeypatch):
    """The exact hazard: the same tensor object passed on two steps with
    different contents."""
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices

    _no_pin(monkeypatch)
    cu = cu_from([2304])
    assert prepare_chunk_indices(cu, 64).shape[0] == 36
    cu.copy_(cu_from([1152]))
    assert prepare_chunk_indices(cu, 64).tolist() == brute_chunk_indices([1152], 64)


def test_cache_accepts_inference_tensors():
    f, calls = _counted(lambda x, n: x.sum() + n)
    with torch.inference_mode():
        x = torch.arange(5)
        assert x.is_inference()
        a = f(x, 1)
        b = f(x, 1)
    assert a is b and len(calls) == 1
    # a different tensor with the same contents is a different key
    with torch.inference_mode():
        f(torch.arange(5), 1)
    assert len(calls) == 2


def test_cache_kwargs_snapshot():
    from vllm.third_party.flash_linear_attention.ops.utils import tensor_cache

    calls = []

    @tensor_cache
    def f(*, x):
        calls.append(1)
        return x.clone()

    x = torch.zeros(3)
    f(x=x)
    f(x=x)
    x.add_(1)
    assert f(x=x).tolist() == [1.0, 1.0, 1.0] and len(calls) == 2


# ------------------------------------------------------------ CPU: shape guard
def test_fused_refuses_short_initial_state():
    from vllm.ampere_prefill import kda_prefill as kp

    H = 2
    T = 10
    x = torch.zeros(1, T, H, D, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="initial_state has 1 rows for 2 sequences"):
        kp.chunk_kda_with_fused_gate(
            q=x, k=x, v=x, raw_g=x, beta=torch.zeros(1, T, H),
            A_log=torch.zeros(1, 1, H, 1), g_bias=torch.zeros(H * D),
            initial_state=torch.zeros(1, H, D, D), output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_from([4, 6]),
            safe_gate=True, lower_bound=LB)


# ------------------------------------------------------------ shared harness
def load_interpreted_kda_prefill(tmpdir):
    """The fused module with the Triton interpreter (TRITON_INTERPRET=1 must be
    set before triton is imported). Two textual substitutions the interpreter
    needs (module-level ``tl`` aliases called from kernels, ``.to`` on a
    Python loop counter); the index arithmetic is untouched."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    spec = importlib.util.find_spec("vllm.ampere_prefill.kda_prefill")
    src = open(spec.origin).read()
    swaps = [
        ("i_t.to(tl.int64)", "tl.full([], i_t, tl.int64)"),
        ("exp = tl.exp\nexp2 = tl.exp2\nlog = tl.log\n",
         "def exp(x):\n    return tl.exp(x)\n\n\ndef exp2(x):\n    return tl.exp2(x)\n\n\n"
         "def log(x):\n    return tl.log(x)\n"),
    ]
    for a, b in swaps:
        assert a in src, a
        src = src.replace(a, b)
    path = os.path.join(tmpdir, "kda_prefill_interpreted.py")
    with open(path, "w") as f:
        f.write(src)
    spec2 = importlib.util.spec_from_file_location("kda_prefill_interpreted", path)
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)
    return mod


class Guarded:
    """Allocations with NaN (float) / sentinel (int) contents and guard zones:
    a read of an element nobody wrote propagates NaN, a write outside an
    allocation changes a guard."""

    def __init__(self):
        self.blocks = []

    @staticmethod
    def _fill_value(dtype):
        return float("nan") if dtype.is_floating_point else -(2 ** 30)

    def empty(self, *shape, dtype=None, device=None, **_):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        dtype = dtype or torch.get_default_dtype()
        n = 1
        for s in shape:
            n *= int(s)
        big = torch.full((n + 2 * GUARD,), self._fill_value(dtype), dtype=dtype,
                         device=device)
        self.blocks.append((big, n))
        return big[GUARD:GUARD + n].view(*shape)

    def empty_like(self, x, dtype=None, **_):
        return self.empty(*x.shape, dtype=dtype or x.dtype, device=x.device)

    def place(self, x):
        y = self.empty(*x.shape, dtype=x.dtype, device=x.device)
        y.copy_(x)
        return y

    def guards_intact(self):
        for big, n in self.blocks:
            for part in (big[:GUARD], big[GUARD + n:]):
                if part.is_floating_point():
                    if not torch.isnan(part).all():
                        return False
                elif not (part == -(2 ** 30)).all():
                    return False
        return True


def install_guarded(mod, guarded):
    shim = types.SimpleNamespace(
        **{n: getattr(torch, n) for n in dir(torch) if not n.startswith("__")})
    shim.empty = guarded.empty
    shim.empty_like = guarded.empty_like
    mod.torch = shim
    orig = torch.Tensor.new_empty

    def new_empty(self, *size, dtype=None, **kw):
        return guarded.empty(*size, dtype=dtype or self.dtype, device=self.device)

    torch.Tensor.new_empty = new_empty
    return lambda: setattr(torch.Tensor, "new_empty", orig)


def make_case(seqlens, H, seed=0, device="cpu", guarded=None):
    """Production layout (q, k, v column views of one [T, 3 H D] buffer),
    production-like magnitudes, a non-zero initial state on every row."""
    g = torch.Generator().manual_seed(seed)
    T, N, P = int(sum(seqlens)), len(seqlens), H * D
    std = torch.exp(torch.randn(3 * P, generator=g) * 0.8 + torch.log(torch.tensor(0.015)))
    pre = std.clamp(0.005, 0.93) * torch.randn(T, 3 * P, generator=g)
    buf = (pre * torch.sigmoid(pre)).to(torch.bfloat16)
    raw_g = ((torch.randn(T, P, generator=g) * 0.38 + 0.02).to(torch.bfloat16)).view(1, T, H, D)
    beta = (torch.randn(T, H, generator=g) * 1.5 + 1.3).sigmoid().unsqueeze(0)
    A_log = (torch.randn(H, generator=g) * 0.45 + 1.69).view(1, 1, H, 1)
    dt = torch.randn(P, generator=g) * 1.13 - 1.0
    h0 = torch.randn(N, H, D, D, generator=g) * 0.008
    c = dict(buf=buf, raw_g=raw_g, beta=beta, A_log=A_log, g_bias=dt, initial_state=h0,
             cu_seqlens=cu_from(seqlens))
    c = {k: v.to(device) for k, v in c.items()}
    if guarded is not None:
        c = {k: guarded.place(v) for k, v in c.items()}
    b = c["buf"]
    c["q"], c["k"], c["v"] = (b[:, i * P:(i + 1) * P].view(1, T, H, D) for i in range(3))
    return c


def run(fn, c):
    v = c["v"].clone() if c["v"].is_contiguous() else c["v"]
    return fn(q=c["q"], k=c["k"], v=v, raw_g=c["raw_g"], beta=c["beta"], A_log=c["A_log"],
              g_bias=c["g_bias"], initial_state=c["initial_state"], output_final_state=True,
              use_qk_l2norm_in_kernel=True, cu_seqlens=c["cu_seqlens"], safe_gate=True,
              lower_bound=LB)


def sub_case(c, n, H):
    """Sequence n of case c alone (its rows, its initial state)."""
    cu = c["cu_seqlens"].tolist()
    a, b = cu[n], cu[n + 1]
    P = H * D
    buf = c["buf"][a:b].clone()
    s = dict(buf=buf, raw_g=c["raw_g"][:, a:b].clone(), beta=c["beta"][:, a:b].clone(),
             A_log=c["A_log"], g_bias=c["g_bias"],
             initial_state=c["initial_state"][n:n + 1].clone(),
             cu_seqlens=cu_from([b - a], device=buf.device))
    s["q"], s["k"], s["v"] = (buf[:, i * P:(i + 1) * P].view(1, b - a, H, D) for i in range(3))
    return s


def reference_fp64(c, H):
    q = c["q"][0].double()
    k = c["k"][0].double()
    v = c["v"][0].double()
    beta = c["beta"][0].double()
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    a = torch.exp(c["A_log"].double().reshape(-1))[:, None]
    x = c["raw_g"][0].double() + c["g_bias"].double().view(H, D)
    eg = torch.exp(LB / (1.0 + torch.exp(-(a * x))))
    cu = c["cu_seqlens"].tolist()
    o = torch.zeros(cu[-1], H, D, dtype=torch.float64, device=q.device)
    fin = torch.empty(len(cu) - 1, H, D, D, dtype=torch.float64, device=q.device)
    for n in range(len(cu) - 1):
        S = c["initial_state"][n].double().clone()
        for t in range(cu[n], cu[n + 1]):
            S.mul_(eg[t][:, None, :])
            u = (v[t] - torch.bmm(S, k[t][:, :, None])[..., 0]) * beta[t][:, None]
            S.add_(u[:, :, None] * k[t][:, None, :])
            o[t] = torch.bmm(S, q[t][:, :, None])[..., 0]
        fin[n] = S
    return o, fin


def check_batch(fn, seqlens, H, device, guarded=None, with_reference=True, seed=0,
                bitwise_alone=True):
    """Returns a list of failure strings (empty: pass)."""
    fails = []
    c = make_case(seqlens, H, seed=seed, device=device, guarded=guarded)
    o, S = run(fn, c)
    if guarded is not None and not guarded.guards_intact():
        fails.append("a write landed outside an allocation")
    cu = c["cu_seqlens"].tolist()
    for n, L in enumerate(seqlens):
        if L == 0:
            if not torch.equal(S[n], c["initial_state"][n]):
                fails.append(f"seq {n} (empty): final state is not its initial state")
            continue
        if not bitwise_alone:
            continue
        so, sS = run(fn, sub_case(c, n, H))
        if not torch.equal(o[0, cu[n]:cu[n + 1]], so[0]):
            fails.append(f"seq {n} (len {L}): output differs from the sequence alone")
        if not torch.equal(S[n], sS[0]):
            fails.append(f"seq {n} (len {L}): final state differs from the sequence alone")
    if not torch.isfinite(o.float()).all() or not torch.isfinite(S).all():
        fails.append("non-finite output or state")
    if with_reference and not fails:
        ro, rS = reference_fp64(c, H)
        eo = float((o[0].double() - ro).abs().max()) / max(float(ro.abs().max()), 1e-30)
        eS = float((S.double() - rS).abs().max()) / max(float(rS.abs().max()), 1e-30)
        if eo > 2e-2 or eS > 2e-3:
            fails.append(f"error vs fp64: o {eo:.3g}, state {eS:.3g} (relative to max)")
    return fails


# Resumed-prefill batches: decode rows (length 1, with state) first, then the
# resumed chunk(s) with state, zero-length padding at the back or (not
# produced by the scheduler today) in the middle.
INTERP_CASES = [
    [70, 0, 5, 130],
    [5, 0, 64],
    [0, 130],
    [1, 1, 1, 200, 0, 0],
    [1, 1, 64, 63, 1, 0],
    [3, 0, 0, 129, 0],
]


def _interp_child(cases, H, tmpdir):
    kp = load_interpreted_kda_prefill(tmpdir)
    results = {}
    for seqlens in cases:
        g = Guarded()
        restore = install_guarded(kp, g)
        try:
            # addressing only: the interpreter's emulation of the bf16 split
            # products is not faithful, so no comparison with the fp64 recurrence
            results[json.dumps(seqlens)] = check_batch(
                kp.chunk_kda_with_fused_gate, seqlens, H, "cpu", guarded=g,
                with_reference=False)
        finally:
            restore()
    return results


def _spawn_interp(cases, H, tmp_path):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, __file__, "--interp", json.dumps(cases), str(H), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=3600)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    if r.returncode != 0 or not line:
        pytest.skip(f"Triton interpreter unavailable: {r.stderr[-800:]}")
    return json.loads(line[-1][len("RESULT "):])


@pytest.mark.parametrize("seqlens", INTERP_CASES, ids=[json.dumps(s) for s in INTERP_CASES])
def test_fused_interpreted_resumed_batches(tmp_path, seqlens):
    res = _spawn_interp([seqlens], 2, tmp_path)
    assert res[json.dumps(seqlens)] == []


def test_fused_interpreted_production_resumed_chunk(tmp_path):
    """A 2,304-token chunk resumed with state behind three decode rows and two
    padding rows (the PP4 shape of a prefix-cache resume with concurrent
    decode)."""
    seqlens = [1, 1, 1, 2304, 0, 0]
    res = _spawn_interp([seqlens], 2, tmp_path)
    assert res[json.dumps(seqlens)] == []


# ------------------------------------------------------------ GPU
GPU_CASES = [
    [2304],
    [1, 1, 1, 2304, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2297],
    [1152, 1152, 0],
    [70, 0, 5, 130],
    [5, 0, 64],
    [3, 0, 0, 2200, 0],
]
GPU_CASES_TP4 = [[3456], [1, 1, 3454, 0], [1152, 0, 2304]]


def _upstream():
    from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate

    return chunk_kda_with_fused_gate


@needs_sm80
@pytest.mark.parametrize("H,seqlens", [(64, s) for s in GPU_CASES]
                         + [(16, s) for s in GPU_CASES + GPU_CASES_TP4])
def test_gpu_fused_resumed_batches(H, seqlens):
    from vllm.ampere_prefill import kda_prefill as kp

    g = Guarded()
    restore = install_guarded(kp, g)
    try:
        fails = check_batch(kp.chunk_kda_with_fused_gate, seqlens, H, "cuda", guarded=g,
                            with_reference=sum(seqlens) <= 300)
        torch.cuda.synchronize()
    finally:
        restore()
    assert fails == []


@needs_sm80
@pytest.mark.parametrize("H,seqlens", [(64, s) for s in GPU_CASES])
def test_gpu_upstream_resumed_batches(H, seqlens):
    # autotuned launches: not bitwise across batch layouts, so judged against
    # the fp64 recurrence (small batches) and for finite outputs
    fails = check_batch(_upstream(), seqlens, H, "cuda", bitwise_alone=False,
                        with_reference=sum(seqlens) <= 300)
    torch.cuda.synchronize()
    assert fails == []


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--interp":
    import vllm.utils.torch_utils as _tu

    _tu.PIN_MEMORY = False
    out = _interp_child(json.loads(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    print("RESULT " + json.dumps(out))
