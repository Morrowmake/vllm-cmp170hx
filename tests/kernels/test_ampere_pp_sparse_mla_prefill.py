# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The 64-head sparse-MLA prefill kernel (vllm/ampere_prefill/sparse_prefill_mla_pp.py).

``VLLM_GLM5_PP_SPARSE_MLA_PREFILL=1`` sends ``sparse_mla_fwd`` calls with all
64 heads on one card (pipeline parallel, TP=1) to a Gluon kernel; every other
call keeps the Triton kernel, and with the flag off nothing changes.

CPU tests (run with ``CUDA_VISIBLE_DEVICES=""``): the flag, the gate and each
of its fallbacks, the dispatch, the banner lines and the warmup gate.
GPU tests (skip without an sm_80 device): the off path and the gate-closed
fallbacks bitwise equal to the flag-off path; on-path accuracy against an
fp64 recomputation, relative to the Triton kernel with the predicated gather
and to the fp32 reference that rounds P to bf16 before P.V; awkward shapes;
determinism; CUDA-graph replay equal to eager with no allocation growth; real
captured prefill records when ``SMLA_PRE_CAPTURE_DIR`` points at them.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_pp_sparse_mla_prefill.py
    pytest -q tests/kernels/test_ampere_pp_sparse_mla_prefill.py
"""

import glob
import os
import sys

import pytest
import torch

from vllm.ampere_prefill import sparse_prefill_mla as spm

FLAG = "VLLM_GLM5_PP_SPARSE_MLA_PREFILL"
PRED = "VLLM_GLM5_SMLA_PREFILL_PRED_LOAD"
PP_MOD = "vllm.ampere_prefill.sparse_prefill_mla_pp"
HEADS, DIM, WIDTH, TOPK, TAIL, GROUP = 64, 512, 2176, 2048, 128, 4
SM_SCALE = 0.0625
MEAN_RATIO, MAX_RATIO = 1.10, 1.25
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")


def _pp():
    from vllm.ampere_prefill import sparse_prefill_mla_pp
    return sparse_prefill_mla_pp


# ------------------------------------------------------------------ inputs
def make_indices(T, prior, seed, width=WIDTH, topk=TOPK):
    """The prefill index layout (logical rows 0 .. prior + T - 1): row r at
    absolute position a = prior + r holds the (a + 1) % 4 rows of its current
    group in the local section and the earlier rows in the top-k section (all
    of them in order while they fit, else a sorted random subset); every
    other slot is -1."""
    front_w = min(topk, width)
    tail = width - front_w
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(prior, prior + T, dtype=torch.int64)
    t = torch.minimum((a + 1) % GROUP, torch.full_like(a, tail))
    n = a + 1 - t
    idx = torch.full((T, width), -1, dtype=torch.int32)
    cols = torch.arange(front_w, dtype=torch.int64)
    for r0 in range(0, T, 256):
        r1 = min(T, r0 + 256)
        nn = n[r0:r1]
        front = torch.where(cols[None, :] < nn[:, None], cols[None, :],
                            torch.full_like(cols[None, :], -1)).clone()
        big = (nn > front_w).nonzero().flatten()
        if len(big):
            nmax = int(nn.max())
            keys = torch.rand(len(big), nmax, generator=g)
            keys[torch.arange(nmax)[None, :] >= nn[big][:, None]] = 2.0
            pick = keys.topk(front_w, dim=1, largest=False).indices
            front[big] = pick.sort(dim=1).values
        idx[r0:r1, :front_w] = front.to(torch.int32)
    if tail:
        tt = torch.arange(tail, dtype=torch.int64)
        loc = a[:, None] - t[:, None] + 1 + tt[None, :]
        loc = torch.where(tt[None, :] < t[:, None], loc, torch.full_like(loc, -1))
        idx[:, front_w:] = loc.to(torch.int32)
    return idx.view(T, 1, width)


def make_case(T, prior, seed, heads=HEADS, width=WIDTH, dim=DIM, q_scale=1.0,
              device="cuda", pages=True):
    """q, kv, indices: the context on 4608-row pages scattered over a cache
    with two spare pages (physical rows), latent std 0.4-0.8 per channel."""
    g = torch.Generator(device=device).manual_seed(seed)
    ctx = prior + T
    page = 4608
    n_blocks = -(-ctx // page) + 2
    ch = 0.4 + 0.4 * torch.rand(dim, generator=g, device=device)
    kv = (torch.randn(n_blocks * page, 1, dim, generator=g, device=device) * ch).to(torch.bfloat16)
    q = (torch.randn(T, heads, dim, generator=g, device=device) * 1.2 * q_scale).to(torch.bfloat16)
    il = make_indices(T, prior, seed, width).long()
    if pages:
        gc = torch.Generator().manual_seed(10_000 + seed)
        blocks = torch.randperm(n_blocks, generator=gc)[:-(-ctx // page)]
        pos = torch.arange(ctx, dtype=torch.int64)
        phys = blocks[pos // page] * page + pos % page
        il = torch.where(il >= 0, phys[il.clamp_min(0)], torch.full_like(il, -1))
    return q, kv, il.to(torch.int32).to(device)


def test_first_chunk_layout_has_many_invalid_slots():
    idx = make_indices(2304, 0, 0)[:, 0, :]
    valid = (idx >= 0).sum(-1)
    assert idx.shape == (2304, WIDTH)
    assert int(valid[0]) == 1 and int(valid[2047]) == 2048
    assert int((WIDTH - valid).float().mean()) > 1000


# --------------------------------------------------------------------- CPU
def test_flag_declared_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv(FLAG, raising=False)
    assert FLAG in envs.environment_variables
    assert envs.environment_variables[FLAG]() is False
    monkeypatch.setenv(FLAG, "1")
    assert envs.environment_variables[FLAG]() is True


@pytest.fixture
def sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (8, 0))


def _meta(T=2304, heads=HEADS, dim=DIM, width=WIDTH, dtype=torch.bfloat16,
          idx_dtype=torch.int32):
    q = torch.empty(T, heads, dim, dtype=dtype, device="meta")
    kv = torch.empty(8192, 1, dim, dtype=dtype, device="meta")
    idx = torch.empty(T, 1, width, dtype=idx_dtype, device="meta")
    return q, kv, idx


@pytest.mark.parametrize("T", [1, 17, 384, 385, 1282, 2304, 2311, 2312])
def test_gate_open_on_the_validated_configuration(sm80, T):
    q, kv, idx = _meta(T)
    assert _pp().closed(q, kv, idx, DIM, 0, None) == ""
    assert _pp().closed(q, kv, idx, DIM, None, None) == ""       # block_dpe from the shapes
    out = torch.empty(T, HEADS, DIM, dtype=torch.bfloat16, device="meta")
    assert _pp().closed(q, kv, idx, DIM, 0, out) == ""


def test_gate_open_on_a_head_strided_q_view(sm80):
    buf = torch.empty(700, HEADS + 3, DIM, dtype=torch.bfloat16, device="meta")
    _, kv, idx = _meta(700)
    assert _pp().closed(buf[:, 1:HEADS + 1], kv, idx, DIM, 0, None) == ""


@pytest.mark.parametrize("case,expect", [
    ("heads16", "16 query heads"),
    ("heads32", "32 query heads"),
    ("heads128", "128 query heads"),
    ("fp16", "dtype"),
    ("dpe", "layout"),
    ("dim576", "layout"),
    ("width2048", "indices"),
    ("width2160", "indices"),
    ("idx_int64", "indices"),
    ("t0", "0 query rows"),
    ("t2313", "2313 query rows"),
    ("out_fp32", "out"),
    ("out_strided", "out"),
    ("q_last_dim", "not contiguous"),
    ("idx_last_dim", "not contiguous"),
])
def test_gate_closed_fallbacks(sm80, case, expect):
    q, kv, idx = _meta()
    dpe, dv, out = 0, DIM, None
    if case.startswith("heads"):
        q, kv, idx = _meta(heads=int(case[5:]))
    elif case == "fp16":
        q, kv, idx = _meta(dtype=torch.float16)
    elif case == "dpe":
        dpe, dv = 64, 448
    elif case == "dim576":
        q, kv, idx = _meta(dim=576)
        dpe = 64
    elif case.startswith("width"):
        q, kv, idx = _meta(width=int(case[5:]))
    elif case == "idx_int64":
        q, kv, idx = _meta(idx_dtype=torch.int64)
    elif case.startswith("t"):
        q, kv, idx = _meta(T=int(case[1:]))
    elif case == "out_fp32":
        out = torch.empty(2304, HEADS, DIM, dtype=torch.float32, device="meta")
    elif case == "out_strided":
        out = torch.empty(2304, HEADS, 2 * DIM, dtype=torch.bfloat16, device="meta")[..., ::2]
    elif case == "q_last_dim":
        q = torch.empty(2304, HEADS, 2 * DIM, dtype=torch.bfloat16, device="meta")[..., ::2]
    elif case == "idx_last_dim":
        idx = torch.empty(2304, 1, 2 * WIDTH, dtype=torch.int32, device="meta")[..., ::2]
    why = _pp().closed(q, kv, idx, dv, dpe, out)
    assert expect in why, why


def test_gate_closed_off_sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (8, 6))
    q, kv, idx = _meta()
    assert _pp().closed(q, kv, idx, DIM, 0, None) == "not sm_80"


def test_banner_lines(monkeypatch, sm80):
    pp = _pp()
    lines = []
    monkeypatch.setattr(pp.logger, "info_once",
                        lambda msg, *a, **k: lines.append(msg % a))
    q, kv, idx = _meta()
    assert pp.use(q, kv, idx, DIM, 0, None) is True
    q16, kv16, idx16 = _meta(heads=16)
    assert pp.use(q16, kv16, idx16, DIM, 0, None) is False
    assert "PP sparse-MLA prefill kernel active" in lines[0] and FLAG in lines[0]
    assert "is off for this call: 16 query heads" in lines[1] and FLAG in lines[1]


class _Recorder:
    """Stands in for the Triton kernel: records the launch."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kw):
            self.calls.append((grid, kw))
        return launch


def _dispatch(monkeypatch, T, heads, width=WIDTH, dim=DIM, block_dpe=0, d_v=DIM):
    """(triton launches, pp launches) of one sparse_mla_fwd call on CPU."""
    rec = _Recorder()
    monkeypatch.setattr(spm, "_sparse_mla_kernel", rec)
    monkeypatch.setattr(spm, "num_sms", lambda *_a: 70)
    pp_calls = []
    pp = _pp()
    monkeypatch.setattr(pp, "sparse_mla_prefill",
                        lambda *a, **k: pp_calls.append((a, k)) or ("pp",))
    q = torch.zeros(T, heads, dim, dtype=torch.bfloat16)
    kv = torch.zeros(4096, 1, dim, dtype=torch.bfloat16)
    idx = torch.zeros(T, 1, width, dtype=torch.int32)
    res = spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=d_v, block_dpe=block_dpe)
    launches = [(g, {k: v for k, v in kw.items() if not isinstance(v, torch.Tensor)})
                for g, kw in rec.calls]
    return launches, pp_calls, res


def test_flag_off_never_consults_the_new_module(monkeypatch, sm80):
    monkeypatch.delenv(FLAG, raising=False)
    pp = _pp()

    def boom(*_a, **_k):
        raise AssertionError("the flag-off path must not reach the new kernel")

    monkeypatch.setattr(pp, "use", boom)
    monkeypatch.setattr(pp, "closed", boom)
    launches, pp_calls, _ = _dispatch(monkeypatch, 2304, HEADS)
    assert len(launches) == 1 and not pp_calls


def test_flag_off_does_not_import_the_new_module(monkeypatch, sm80):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delitem(sys.modules, PP_MOD, raising=False)
    import vllm.ampere_prefill as pkg
    monkeypatch.delattr(pkg, "sparse_prefill_mla_pp", raising=False)
    rec = _Recorder()
    monkeypatch.setattr(spm, "_sparse_mla_kernel", rec)
    monkeypatch.setattr(spm, "num_sms", lambda *_a: 70)
    spm.sparse_mla_fwd(torch.zeros(2304, HEADS, DIM, dtype=torch.bfloat16),
                       torch.zeros(4096, 1, DIM, dtype=torch.bfloat16),
                       torch.zeros(2304, 1, WIDTH, dtype=torch.int32), SM_SCALE,
                       d_v=DIM, block_dpe=0)
    assert PP_MOD not in sys.modules
    assert len(rec.calls) == 1


@pytest.mark.parametrize("pred", [False, True])
@pytest.mark.parametrize("T,heads,width,dim,dpe,dv,live", [
    (2304, 64, WIDTH, DIM, 0, DIM, True),
    (1282, 64, WIDTH, DIM, 0, DIM, True),
    (2312, 64, WIDTH, DIM, 0, DIM, True),
    (3456, 64, WIDTH, DIM, 0, DIM, False),
    (2304, 16, WIDTH, DIM, 0, DIM, False),
    (2304, 32, WIDTH, DIM, 0, DIM, False),
    (2304, 64, 2048, DIM, 0, DIM, False),
    (2304, 64, WIDTH, 576, 64, DIM, False),
])
def test_dispatch_and_unchanged_fallback(monkeypatch, sm80, pred, T, heads, width, dim, dpe,
                                         dv, live):
    """Flag on: the validated configuration goes to the new kernel and never
    launches the Triton kernel; everything else launches exactly what the
    flag-off call launches (with or without the predicated gather)."""
    if pred:
        monkeypatch.setenv(PRED, "1")
    else:
        monkeypatch.delenv(PRED, raising=False)
    monkeypatch.delenv(FLAG, raising=False)
    off, pp_off, _ = _dispatch(monkeypatch, T, heads, width, dim, dpe, dv)
    assert len(off) == 1 and not pp_off
    monkeypatch.setenv(FLAG, "1")
    on, pp_on, res = _dispatch(monkeypatch, T, heads, width, dim, dpe, dv)
    if live:
        assert on == [] and len(pp_on) == 1 and res == ("pp",)
        (a, k), = pp_on
        assert a[3] == SM_SCALE and k["d_v"] == DIM and k["out"] is None
    else:
        assert not pp_on and on == off


class _FakeWorker:
    def __init__(self, heads):
        self.device = torch.device("cuda:0")
        self.parallel_config = object()
        self.model_config = type("M", (), {
            "get_num_attention_heads": lambda _s, _p: heads})()


@pytest.mark.parametrize("flag,prefill,heads", [(0, 1, 64), (1, 0, 64), (1, 1, 16)])
def test_warmup_gate(monkeypatch, sm80, flag, prefill, heads):
    monkeypatch.setenv(FLAG, str(flag))
    monkeypatch.setenv("VLLM_GLM5_PREFILL_KERNELS", str(prefill))
    pp = _pp()
    monkeypatch.setattr(pp, "warmup", lambda *_a, **_k: pytest.fail("must not compile"))
    assert pp.warmup_from_worker(_FakeWorker(heads)) is False


def test_warmup_runs_when_open(monkeypatch, sm80):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_GLM5_PREFILL_KERNELS", "1")
    pp = _pp()
    seen = []
    monkeypatch.setattr(pp, "warmup", lambda dev: seen.append(dev))
    assert pp.warmup_from_worker(_FakeWorker(64)) is True and len(seen) == 1


# --------------------------------------------------------------------- GPU
def _fwd(q, kv, idx, monkeypatch, flag, pred=True, out=None, block_dpe=0, d_v=DIM):
    monkeypatch.setenv(FLAG, "1") if flag else monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setenv(PRED, "1") if pred else monkeypatch.delenv(PRED, raising=False)
    res = spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=d_v, block_dpe=block_dpe, out=out)
    torch.cuda.synchronize()
    return tuple(t.clone() for t in res)


def _math(q, kv, idx, dt, round_p):
    """fp64 / fp32 recomputation: fp32 P rounded to bf16 before P.V when
    round_p (the tensor-core convention), normalised by the unrounded sum."""
    U = kv.shape[0]
    ii = idx[:, 0, :].long()
    valid = (ii >= 0) & (ii < U)
    kk = kv[ii.clamp(0, U - 1), 0, :].to(dt)
    s = torch.einsum("thd,tkd->thk", q.to(dt), kk) * SM_SCALE
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(s - m)
    lsum = e.sum(-1)
    ep = e.to(torch.bfloat16).to(dt) if round_p else e
    return torch.einsum("thk,tkd->thd", ep, kk[..., :DIM]) / lsum.clamp_min(1e-300)[..., None]


def _refs(q, kv, idx, rows, chunk=8):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q, idx = q[rows], idx[rows]
        r64 = torch.empty(len(rows), q.shape[1], DIM, dtype=torch.float64, device=q.device)
        bp = torch.empty(len(rows), q.shape[1], DIM, dtype=torch.bfloat16, device=q.device)
        for i in range(0, len(rows), chunk):
            j = min(len(rows), i + chunk)
            r64[i:j] = _math(q[i:j], kv, idx[i:j], torch.float64, False)
            bp[i:j] = _math(q[i:j], kv, idx[i:j], torch.float32, True).to(torch.bfloat16)
        ii = idx[:, 0, :]
        live = ((ii >= 0) & (ii < kv.shape[0])).sum(-1) > 0
        return r64, bp, live
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _err(a, ref, live):
    d = (a.double() - ref).abs()[live]
    return (float(d.mean()), float(d.max())) if d.numel() else (0.0, 0.0)


def _check_rows(T):
    if T <= 128:
        return list(range(T))
    return sorted(set(range(64)) | set(range(T // 2 - 32, T // 2 + 32)) | set(range(T - 64, T)))


def _accuracy(q, kv, idx, monkeypatch, out=None):
    """(new, incumbent, errors) on a row subset: the new kernel vs the Triton
    kernel with the predicated gather, both against fp64."""
    new = _fwd(q, kv, idx, monkeypatch, True, out=out)
    inc = _fwd(q, kv, idx, monkeypatch, False)
    rows = torch.tensor(_check_rows(q.shape[0]), device=q.device)
    r64, bp, live = _refs(q, kv, idx, rows)
    e = dict(new=_err(new[0][rows], r64, live), inc=_err(inc[0][rows], r64, live),
             bp=_err(bp, r64, live))
    return new, inc, e, rows, live


def _assert_within(e):
    """The accuracy gate, exactly: against fp64 on the same inputs, the new
    kernel's mean error <= 1.10x and max error <= 1.25x of the incumbent's
    (the Triton kernel with the predicated gather) and of the fp32 reference
    with P rounded to bf16. No absolute or per-element allowance."""
    for base in ("inc", "bp"):
        m, x = e[base]
        assert e["new"][0] <= MEAN_RATIO * m, (base, e)
        assert e["new"][1] <= MAX_RATIO * x, (base, e)


@needs_sm80
@pytest.mark.parametrize("T,prior", [(2304, 0), (2304, 2304), (2304, 4608), (1282, 6912),
                                     (385, 2304), (2312, 2304), (2304, 63232)])
def test_accuracy_vs_fp64_relative_to_incumbent(monkeypatch, T, prior):
    q, kv, idx = make_case(T, prior, seed=T + prior)
    new, inc, e, rows, live = _accuracy(q, kv, idx, monkeypatch)
    _assert_within(e)
    assert torch.isfinite(new[0]).all() and torch.isfinite(new[2]).all()
    # lse is not used at prefill (only with DCP), but must track the incumbent
    torch.testing.assert_close(new[2], inc[2], rtol=0, atol=1e-3)
    torch.testing.assert_close(new[1], inc[1], rtol=0, atol=1e-3)


@needs_sm80
@pytest.mark.parametrize("case", ["t1", "t17", "t63", "dead_and_single", "out_of_range",
                                  "strided_q", "peaky"])
def test_awkward(monkeypatch, case):
    if case == "t1":
        q, kv, idx = make_case(1, 0, 64)
    elif case == "t17":
        q, kv, idx = make_case(17, 4096, 65)
    elif case == "t63":
        q, kv, idx = make_case(63, 2304, 66)
    elif case == "dead_and_single":
        q, kv, idx = make_case(640, 2304, 71)
        idx[::7] = -1
        one = torch.arange(3, 640, 7, device="cuda")
        keep = idx[one, 0, 5].clone()
        idx[one] = -1
        idx[one, 0, 5] = keep
    elif case == "out_of_range":
        q, kv, idx = make_case(512, 2304, 72)
        g = torch.Generator().manual_seed(72)
        m = (torch.rand(idx.shape, generator=g) < 0.1).cuda() & (idx >= 0)
        big = kv.shape[0] + torch.randint(0, 1 << 20, idx.shape, generator=g).cuda().int()
        idx = torch.where(m, big, idx)
    elif case == "strided_q":
        q0, kv, idx = make_case(515, 2304, 73)
        buf = torch.zeros(515, HEADS + 3, DIM, dtype=q0.dtype, device="cuda")
        buf[:, 1:HEADS + 1] = q0
        q = buf[:, 1:HEADS + 1]
    else:
        q, kv, idx = make_case(512, 2304, 69, q_scale=3.0)
    snap = (q.clone(), kv.clone(), idx.clone())
    new, inc, e, rows, live = _accuracy(q, kv, idx, monkeypatch)
    _assert_within(e)
    assert torch.equal(snap[0], q) and torch.equal(snap[1], kv) and torch.equal(snap[2], idx)
    assert torch.isfinite(new[0]).all()
    if case == "dead_and_single":
        assert torch.count_nonzero(new[0][::7]) == 0
        assert torch.count_nonzero(inc[0][::7]) == 0


@needs_sm80
def test_caller_out_buffer_and_determinism(monkeypatch):
    q, kv, idx = make_case(777, 2304, 81)
    a = _fwd(q, kv, idx, monkeypatch, True)
    b = _fwd(q, kv, idx, monkeypatch, True)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    buf = torch.full((777, HEADS, DIM), float("nan"), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setenv(FLAG, "1")
    res = spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0, out=buf)
    torch.cuda.synchronize()
    assert res[0].data_ptr() == buf.data_ptr() and torch.equal(buf, a[0])


@needs_sm80
@pytest.mark.parametrize("pred", [False, True])
@pytest.mark.parametrize("T,prior", [(2304, 0), (2304, 4608), (1282, 6912)])
def test_off_path_bitwise(monkeypatch, pred, T, prior):
    """Flag off vs flag on with the gate forced closed: the fallthrough runs
    exactly the flag-off code."""
    q, kv, idx = make_case(T, prior, seed=5 + T + prior)
    off = _fwd(q, kv, idx, monkeypatch, False, pred=pred)
    monkeypatch.setattr(_pp(), "use", lambda *_a, **_k: False)
    on = _fwd(q, kv, idx, monkeypatch, True, pred=pred)
    for x, y in zip(on, off):
        assert torch.equal(x, y)


@needs_sm80
@pytest.mark.parametrize("case", ["heads16", "dpe64", "width2048", "t2313"])
def test_gate_closed_fallbacks_bitwise(monkeypatch, case):
    dpe, dv = 0, DIM
    if case == "heads16":
        q, kv, idx = make_case(2304, 2304, 3, heads=16)
    elif case == "dpe64":
        q, kv, idx = make_case(1152, 2304, 4, dim=576)
        dpe = 64
    elif case == "width2048":
        q, kv, idx = make_case(1152, 4608, 6, width=2048)
    else:
        q, kv, idx = make_case(2313, 2304, 8)
    assert _pp().closed(q, kv, idx, dv, dpe, None) != ""
    off = _fwd(q, kv, idx, monkeypatch, False, block_dpe=dpe, d_v=dv)
    on = _fwd(q, kv, idx, monkeypatch, True, block_dpe=dpe, d_v=dv)
    for x, y in zip(on, off):
        assert torch.equal(x, y)


def _graph(fn):
    """(allocation during capture beyond an empty capture, growth over 3
    replays) of one captured call."""
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    b0 = torch.cuda.memory_allocated()
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        pass
    base = torch.cuda.memory_allocated() - b0
    before = torch.cuda.memory_allocated()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    after = torch.cuda.memory_allocated()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    growth = torch.cuda.memory_allocated() - after
    del g, g0
    return after - before - base, growth


@needs_sm80
@pytest.mark.parametrize("T,prior", [(2304, 2304), (1282, 6912)])
def test_graph_replay_equals_eager_no_growth(monkeypatch, T, prior):
    """Through sparse_mla_fwd: the capture allocates what the flag-off capture
    allocates (the returned stats), replays allocate nothing, and replay
    output equals eager."""
    q, kv, idx = make_case(T, prior, seed=13 + T)
    out = torch.empty(T, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    cap = {}
    for flag in (False, True):
        eager = _fwd(q, kv, idx, monkeypatch, flag, out=out)
        holder = {}

        def fn():
            holder["r"] = spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0,
                                             out=out)

        out.fill_(float("nan"))
        cap[flag], growth = _graph(fn)
        assert growth == 0, growth
        assert torch.equal(out, eager[0])
        assert torch.equal(holder["r"][2], eager[2])
    assert cap[True] == cap[False], cap


def _capture_files():
    d = os.environ.get("SMLA_PRE_CAPTURE_DIR", "")
    return sorted(glob.glob(os.path.join(d, "p*_b*_*.pt"))) if d else []


@needs_sm80
@pytest.mark.skipif(not _capture_files(), reason="SMLA_PRE_CAPTURE_DIR has no records")
def test_real_records(monkeypatch):
    """Each captured record replayed at its original row positions inside a
    full-size chunk (other rows -1). Summed mean error and worst max error
    over all records, against fp64: new <= 1.10x / 1.25x of the Triton
    kernel's and of the bf16-P reference's; padding rows give 0."""
    tot = dict(new=[0.0, 0.0], inc=[0.0, 0.0], bp=[0.0, 0.0])
    n = 0
    for f in _capture_files():
        c = torch.load(f, map_location="cpu", weights_only=False)
        if int(c["in_num_heads"]) != HEADS or c["in_indices"].shape[2] != WIDTH:
            continue
        NT = int(c["in_num_tokens"])
        rows = c["in_rows"].cuda()
        q = torch.zeros(NT, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        q[rows] = c["in_q"].cuda()
        idx = torch.full((NT, 1, WIDTH), -1, dtype=torch.int32, device="cuda")
        idx[rows] = c["in_indices"].cuda()
        kv = c["in_kv"].cuda()
        new = _fwd(q, kv, idx, monkeypatch, True)
        inc = _fwd(q, kv, idx, monkeypatch, False)
        assert torch.equal(inc[0][rows], c["out_out"].cuda()), f
        other = torch.ones(NT, dtype=torch.bool, device="cuda")
        other[rows] = False
        assert torch.count_nonzero(new[0][other]) == 0
        r64, bp, live = _refs(q, kv, idx, rows)
        for k, v in (("new", new[0][rows]), ("inc", inc[0][rows]), ("bp", bp)):
            m, x = _err(v, r64, live)
            tot[k][0] += m
            tot[k][1] = max(tot[k][1], x)
        n += 1
    if n == 0:
        pytest.skip("no 64-head records in SMLA_PRE_CAPTURE_DIR")
    _assert_within({k: tuple(v) for k, v in tot.items()})
