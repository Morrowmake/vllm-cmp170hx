# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predicated row gather of the sm_80 sparse-MLA prefill kernel.

``VLLM_GLM5_SMLA_PREFILL_PRED_LOAD=1`` masks the kernel's row gather with the
index mask, so invalid (-1) top-k slots load nothing instead of all reading
cache row 0.  The output must be bitwise the plain gather's: an invalid slot's
score is forced to -1e30 and its pv weight to 0 either way.

CPU tests (run with ``CUDA_VISIBLE_DEVICES=""``): the flag, the gate and its
fallbacks, and that the launch is unchanged whenever the gate is closed.
GPU tests (skip without a device): on vs off bitwise on production-layout
inputs at 64 heads (pipeline parallel) and 16 heads (tensor parallel 4), on
real captured prefill records when ``SMLA_PRE_CAPTURE_DIR`` points at them,
and CUDA-graph replay.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_smla_prefill_pred_load.py
    pytest -q tests/kernels/test_ampere_smla_prefill_pred_load.py
"""

import glob
import os

import pytest
import torch

from vllm.ampere_prefill import sparse_prefill_mla as spm

FLAG = "VLLM_GLM5_SMLA_PREFILL_PRED_LOAD"
HEADS, DIM, WIDTH, TOPK, TAIL, GROUP = 64, 512, 2176, 2048, 128, 4
TP4_HEADS = 16
LIVE_HEADS = (HEADS, TP4_HEADS)
SM_SCALE = 0.0625
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")


# ------------------------------------------------------------------ inputs
def make_indices(T, prior, seed, width=WIDTH, topk=TOPK):
    """The prefill index layout: row r (absolute position a = prior + r) holds
    the (a + 1) % 4 rows of its current group in the local section and the
    earlier rows in the top-k section (all of them in order while they fit,
    else a sorted random subset); every other slot is -1."""
    tail = width - topk
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(prior, prior + T, dtype=torch.int64)
    t = (a + 1) % GROUP
    n = a + 1 - t
    idx = torch.full((T, width), -1, dtype=torch.int32)
    cols = torch.arange(topk, dtype=torch.int64)
    for r0 in range(0, T, 256):
        r1 = min(T, r0 + 256)
        nn = n[r0:r1]
        front = torch.where(cols[None, :] < nn[:, None], cols[None, :],
                            torch.full_like(cols[None, :], -1)).clone()
        big = (nn > topk).nonzero().flatten()
        if len(big):
            nmax = int(nn.max())
            keys = torch.rand(len(big), nmax, generator=g)
            keys[torch.arange(nmax)[None, :] >= nn[big][:, None]] = 2.0
            pick = keys.topk(topk, dim=1, largest=False).indices
            front[big] = pick.sort(dim=1).values
        idx[r0:r1, :topk] = front.to(torch.int32)
    tt = torch.arange(tail, dtype=torch.int64)
    loc = a[:, None] - t[:, None] + 1 + tt[None, :]
    loc = torch.where(tt[None, :] < t[:, None], loc, torch.full_like(loc, -1))
    idx[:, topk:] = loc.to(torch.int32)
    return idx.view(T, 1, width)


def make_case(T, prior, seed, heads=HEADS, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.randn(T, heads, DIM, generator=g, device=device) * 1.2).to(torch.bfloat16)
    kv = (torch.randn(prior + T, 1, DIM, generator=g, device=device) * 0.6).to(torch.bfloat16)
    return q, kv, make_indices(T, prior, seed).to(device)


def test_first_chunk_layout_has_many_invalid_slots():
    idx = make_indices(2304, 0, 0)[:, 0, :]
    valid = (idx >= 0).sum(-1)
    assert idx.shape == (2304, WIDTH)
    # row r sees r + 1 rows (front + local), capped by the 2048-slot front.
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


def _meta(T=2304, heads=HEADS, dim=DIM, width=WIDTH, dtype=torch.bfloat16):
    q = torch.empty(T, heads, dim, dtype=dtype, device="meta")
    kv = torch.empty(8192, 1, dim, dtype=dtype, device="meta")
    return q, kv, width


def _cfg(T, heads, width=WIDTH, dim=DIM):
    cfg = spm._select_config(T, width, heads, dim, 70)
    return cfg, cfg[2] == 1 and width % cfg[1] == 0


@pytest.fixture
def sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (8, 0))


@pytest.mark.parametrize("heads", LIVE_HEADS)
@pytest.mark.parametrize("T", [512, 1152, 1282, 2304, 3456])
def test_gate_open_on_the_validated_configuration(monkeypatch, sm80, T, heads):
    monkeypatch.setenv(FLAG, "1")
    q, kv, w = _meta(T, heads=heads)
    cfg, rot = _cfg(T, heads)
    assert spm._pred_load_closed(q, kv, w, 0, DIM, cfg, rot) == ""
    assert spm._use_pred_load(q, kv, w, 0, DIM, cfg, rot) is True


def test_gate_closed_when_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)

    def boom(*_a, **_k):
        raise AssertionError("the flag-off path must not look at the device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    q, kv, w = _meta()
    cfg, rot = _cfg(2304, HEADS)
    assert spm._use_pred_load(q, kv, w, 0, DIM, cfg, rot) is False


@pytest.mark.parametrize("case,expect", [
    ("heads8", "8 query heads"),
    ("heads32", "32 query heads"),
    ("heads16_64schedule", "schedule"),
    ("heads64_16schedule", "schedule"),
    ("fp16", "dtype"),
    ("dpe", "layout"),
    ("dim576", "layout"),
    ("width2048", "2048 index slots"),
    ("norotate", "schedule"),
    ("config", "schedule"),
])
def test_gate_closed_fallbacks(monkeypatch, sm80, case, expect):
    monkeypatch.setenv(FLAG, "1")
    q, kv, w = _meta()
    cfg, rot = _cfg(2304, HEADS)
    dpe, dv = 0, DIM
    if case == "heads8":
        q, kv, w = _meta(heads=8)
        cfg, rot = _cfg(2304, 8)
    elif case == "heads16_64schedule":
        q, kv, w = _meta(heads=16)
    elif case == "heads64_16schedule":
        cfg, rot = _cfg(2304, 16)
    elif case == "heads32":
        q, kv, w = _meta(heads=32)
    elif case == "fp16":
        q, kv, w = _meta(dtype=torch.float16)
    elif case == "dpe":
        dpe, dv = 64, 448
    elif case == "dim576":
        q, kv, w = _meta(dim=576)
        dpe = 64
    elif case == "width2048":
        q, kv, w = _meta(width=2048)
        cfg, rot = _cfg(2304, HEADS, width=2048)
    elif case == "norotate":
        rot = False
    elif case == "config":
        cfg = (32, 32, 2, 4, 2)
    why = spm._pred_load_closed(q, kv, w, dpe, dv, cfg, rot)
    assert expect in why, why
    assert spm._use_pred_load(q, kv, w, dpe, dv, cfg, rot) is False


def test_gate_closed_off_sm80(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (8, 6))
    q, kv, w = _meta()
    cfg, rot = _cfg(2304, HEADS)
    assert spm._pred_load_closed(q, kv, w, 0, DIM, cfg, rot) == "not sm_80"


class _Recorder:
    """Stands in for the Triton kernel: records the launch."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kw):
            self.calls.append((grid, kw))
        return launch


def _launch_kwargs(monkeypatch, T, heads, width=WIDTH):
    rec = _Recorder()
    monkeypatch.setattr(spm, "_sparse_mla_kernel", rec)
    monkeypatch.setattr(spm, "num_sms", lambda *_a: 70)
    q = torch.zeros(T, heads, DIM, dtype=torch.bfloat16)
    kv = torch.zeros(4096, 1, DIM, dtype=torch.bfloat16)
    idx = torch.zeros(T, 1, width, dtype=torch.int32)
    spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0)
    (grid, kw), = rec.calls
    return grid, {k: v for k, v in kw.items() if not isinstance(v, torch.Tensor)}


@pytest.mark.parametrize("T,heads,width", [
    (2304, 64, WIDTH), (1282, 64, WIDTH), (2304, 16, WIDTH), (3456, 16, WIDTH),
    (1282, 16, WIDTH), (2304, 64, 2048), (3456, 16, 2048), (2304, 8, WIDTH),
    (2304, 32, WIDTH)])
def test_launch_unchanged_when_gate_closed(monkeypatch, sm80, T, heads, width):
    """Flag off: the launch never carries PRED_LOAD.  Flag on: only the
    validated configuration adds it, and nothing else in the launch moves."""
    monkeypatch.delenv(FLAG, raising=False)
    g0, off = _launch_kwargs(monkeypatch, T, heads, width)
    assert "PRED_LOAD" not in off
    monkeypatch.setenv(FLAG, "1")
    g1, on = _launch_kwargs(monkeypatch, T, heads, width)
    live = heads in LIVE_HEADS and width == WIDTH
    assert on.pop("PRED_LOAD", False) is live
    assert g0 == g1 and on == off


def test_kernel_source_predicates_only_the_nope_gather():
    src = spm._sparse_mla_kernel.src
    assert "PRED_LOAD: tl.constexpr = False" in src
    assert "mask=mask_kv[:, None], other=0.0)" in src
    # the clamp target stays row 0
    assert "tl.where(mask_kv, indices, 0)" in src
    assert src.count("mask=mask_kv[:, None]") == 1


# --------------------------------------------------------------------- GPU
def _run(q, kv, idx, flag, monkeypatch, out=None):
    if flag:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    o, mx, lse = spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0, out=out)
    torch.cuda.synchronize()
    return o, mx, lse


def _assert_same(a, b):
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@needs_sm80
@pytest.mark.parametrize("T,prior", [(2304, 0), (2304, 2304), (2304, 4608), (1282, 6912),
                                     (1152, 0), (600, 0)])
def test_on_equals_off_bitwise(monkeypatch, T, prior):
    q, kv, idx = make_case(T, prior, seed=T + prior)
    cfg, rot = _cfg(T, HEADS)
    assert spm._pred_load_closed(q, kv, WIDTH, 0, DIM, cfg, rot) == ""
    off = _run(q, kv, idx, False, monkeypatch)
    off = tuple(t.clone() for t in off)
    on = _run(q, kv, idx, True, monkeypatch)
    _assert_same(on, off)
    assert torch.isfinite(on[0]).all() and torch.isfinite(on[2]).all()


@needs_sm80
def test_on_equals_off_with_fully_masked_rows(monkeypatch):
    """Rows with no valid slot at all keep the finite-zero result."""
    q, kv, idx = make_case(1152, 0, seed=7)
    idx[::5] = -1
    off = tuple(t.clone() for t in _run(q, kv, idx, False, monkeypatch))
    on = _run(q, kv, idx, True, monkeypatch)
    _assert_same(on, off)
    assert torch.count_nonzero(on[0][::5]) == 0


@needs_sm80
@pytest.mark.parametrize("T,prior", [(3456, 0), (3456, 3456), (3456, 6912), (1282, 6912),
                                     (1282, 0), (600, 0)])
def test_16_heads_on_equals_off_bitwise(monkeypatch, T, prior):
    """The tensor-parallel-4 layout (16 heads per rank, 3456-row chunks)."""
    q, kv, idx = make_case(T, prior, seed=3 + T + prior, heads=TP4_HEADS)
    cfg, rot = _cfg(T, TP4_HEADS)
    assert spm._pred_load_closed(q, kv, WIDTH, 0, DIM, cfg, rot) == ""
    off = tuple(t.clone() for t in _run(q, kv, idx, False, monkeypatch))
    on = _run(q, kv, idx, True, monkeypatch)
    _assert_same(on, off)
    assert torch.isfinite(on[0]).all() and torch.isfinite(on[2]).all()


@needs_sm80
def test_8_heads_fall_through(monkeypatch):
    q, kv, idx = make_case(2304, 0, seed=3, heads=8)
    off = tuple(t.clone() for t in _run(q, kv, idx, False, monkeypatch))
    on = _run(q, kv, idx, True, monkeypatch)
    _assert_same(on, off)


def _capture_dir():
    d = os.environ.get("SMLA_PRE_CAPTURE_DIR", "")
    files = sorted(glob.glob(os.path.join(d, "p*_b*_*.pt"))) if d else []
    return files


@needs_sm80
@pytest.mark.skipif(not _capture_dir(), reason="SMLA_PRE_CAPTURE_DIR has no records")
def test_real_records_on_equals_off_equals_live(monkeypatch):
    """Each captured record replayed at its original row positions inside a
    full-size chunk (the other rows all -1), so every row keeps its rotated
    start: off must reproduce the live output and on must equal off."""
    n = 0
    for f in _capture_dir():
        c = torch.load(f, map_location="cpu", weights_only=False)
        heads = int(c["in_num_heads"])
        if heads not in LIVE_HEADS or c["in_indices"].shape[2] != WIDTH:
            continue
        NT = int(c["in_num_tokens"])
        rows = c["in_rows"].cuda()
        q = torch.zeros(NT, heads, DIM, dtype=torch.bfloat16, device="cuda")
        q[rows] = c["in_q"].cuda()
        idx = torch.full((NT, 1, WIDTH), -1, dtype=torch.int32, device="cuda")
        idx[rows] = c["in_indices"].cuda()
        kv = c["in_kv"].cuda()
        off = tuple(t.clone() for t in _run(q, kv, idx, False, monkeypatch))
        on = _run(q, kv, idx, True, monkeypatch)
        _assert_same(on, off)
        assert torch.equal(off[0][rows], c["out_out"].cuda()), f
        n += 1
    if n == 0:
        pytest.skip("no 64- or 16-head records in SMLA_PRE_CAPTURE_DIR")


def _graph_growth(fn):
    """(eager outputs, replay outputs, allocation growth) of one captured call,
    net of the fixed per-capture allocation of an empty graph."""
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
    grow = (after - before - base) + (torch.cuda.memory_allocated() - after)
    del g, g0
    return grow


@needs_sm80
def test_graph_replay_equals_eager_no_growth(monkeypatch):
    from vllm.triton_utils import LOG2E, LOGE2

    q, kv, idx = make_case(2304, 0, seed=11)
    T = q.shape[0]
    out = torch.empty(T, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    stats = torch.empty(2, T, HEADS, dtype=torch.float32, device="cuda")
    BLOCK_H, BLOCK_N, _, warps, stages = spm._select_config(
        T, WIDTH, HEADS, DIM, spm.num_sms(0))
    sq, skv, so, si = q.stride(), kv.stride(), out.stride(), idx.stride()

    def launch():
        spm._sparse_mla_kernel[(T, HEADS // BLOCK_H, 1)](
            q_buffer=q, k_buffer=kv, indices_ptr=idx, out_ptr=out,
            softmax_lse_ptr=stats[1], part_acc_ptr=stats[0], part_sum_ptr=stats[0],
            part_max_ptr=stats[0], counter_ptr=stats[0], max_logits_ptr=stats[0],
            seq_kv=kv.shape[0], h_q=HEADS, dim_qk=DIM, dim_v=DIM,
            stride_q_token=sq[0], stride_q_head=sq[1],
            stride_k_token=skv[0], stride_k_head=skv[1],
            stride_out_token=so[0], stride_out_head=so[1], stride_lse=HEADS,
            stride_indices_token=si[0], stride_indices_head=si[1],
            sm_scale=SM_SCALE * LOG2E, split_len=WIDTH, kv_group_num=HEADS,
            index_topk=WIDTH, NUM_SPLITS=1, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N,
            BLOCK_DV=DIM, BLOCK_DMODEL=DIM, BLOCK_DPE=0, V_IS_K=True,
            ROTATE=True, LOGE2=LOGE2, PRED_LOAD=True,
            num_warps=warps, num_stages=stages)

    launch()
    torch.cuda.synchronize()
    eager = (out.clone(), stats.clone())
    # the direct launch is the production call with the gate open
    ref = _run(q, kv, idx, True, monkeypatch)
    assert torch.equal(ref[0], eager[0]) and torch.equal(ref[2], eager[1][1])
    out.fill_(float("nan"))
    stats.fill_(float("nan"))
    grow = _graph_growth(launch)
    assert torch.equal(out, eager[0]) and torch.equal(stats, eager[1])
    assert grow == 0, grow


@needs_sm80
@pytest.mark.parametrize("heads,T", [(HEADS, 2304), (TP4_HEADS, 3456)])
def test_graph_capture_through_the_launcher(monkeypatch, heads, T):
    """Through sparse_mla_fwd (which allocates its stats per call) the capture
    allocates exactly what the flag-off capture allocates, and replay equals
    eager."""
    q, kv, idx = make_case(T, 0, seed=13, heads=heads)
    out = torch.empty(T, heads, DIM, dtype=torch.bfloat16, device="cuda")
    growth, res = {}, {}
    for flag in (False, True):
        eager = _run(q, kv, idx, flag, monkeypatch, out=out)[0].clone()
        holder = {}

        def fn():
            holder["r"] = spm.sparse_mla_fwd(q, kv, idx, SM_SCALE, d_v=DIM,
                                             block_dpe=0, out=out)

        out.fill_(float("nan"))
        growth[flag] = _graph_growth(fn)
        assert torch.equal(out, eager)
        res[flag] = eager
    assert torch.equal(res[True], res[False])
    assert growth[True] == growth[False], growth
