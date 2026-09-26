# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PP_DRAFT_TAIL_STAGE, CPU only.

The DFlash2 drafter's tail (candidate lm_head pass, top-k, selector, walk)
moves from the last pipeline stage to an earlier one. These tests check the
pieces that decide and carry it -- the gate, the payload, the per-step role
agreement, the last stage's pending-draft queue, the private workspaces --
and run a simulated 4-stage pipeline through the real PPHandler methods with
an in-process communicator, showing that every stage ends up with the same
draft tokens with the flag on and off.
"""

import ast
import contextlib
import pathlib
import sys
import types
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pp_utils as pp_utils
from vllm.v1.worker.gpu import pp_draft_tail as dt
from vllm.v1.worker.gpu.pp_utils import PPHandler

FLAG = "VLLM_PP_DRAFT_TAIL_STAGE"

# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def _config(pp=4, tp=1, dp=1, method="dflash", arch="DFlash2DraftModel",
            sample="greedy", adaptive=False, adaptive_k=False, v2=True):
    spec = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(architectures=[arch]),
        draft_sample_method=sample,
        enable_adaptive_verification=adaptive,
        uses_adaptive_k=lambda: adaptive_k,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            prefill_context_parallel_size=1,
        ),
        use_v2_model_runner=v2,
        speculative_config=spec,
    )


def test_gate_default_is_off_and_silent():
    gate = dt.draft_tail_gate(_config(), -1)
    assert not gate.enabled and gate.reason == ""


def test_gate_accepts_the_production_layout():
    gate = dt.draft_tail_gate(_config(), 2)
    assert gate.enabled and gate.stage == 2 and gate.reason == ""


@pytest.mark.parametrize(
    "kwargs,stage,needle",
    [
        (dict(pp=1), 0, "pipeline parallelism"),
        (dict(), 3, "earlier stage than the last"),
        (dict(tp=2), 2, "tensor parallel size 1"),
        (dict(dp=2), 2, "data parallelism"),
        (dict(v2=False), 2, "V2 model runner"),
        (dict(method="mtp"), 2, "DFlash2"),
        (dict(arch="DFlashDraftModel"), 2, "candidate selector"),
        (dict(sample="probabilistic"), 2, "probabilistic"),
        (dict(adaptive=True), 2, "adaptive verification"),
        (dict(adaptive_k=True), 2, "variable draft counts"),
    ],
)
def test_gate_refuses_with_a_reason(kwargs, stage, needle):
    gate = dt.draft_tail_gate(_config(**kwargs), stage)
    assert not gate.enabled and needle in gate.reason


def test_flag_declared_default_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(FLAG, raising=False)
    assert envs.VLLM_PP_DRAFT_TAIL_STAGE == -1
    monkeypatch.setenv(FLAG, "2")
    assert envs.VLLM_PP_DRAFT_TAIL_STAGE == 2


def test_controller_not_created_when_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert dt.DraftTailController.create(_config(), "cpu", 3, 4) is None


# --------------------------------------------------------------------------
# Per-step decision, roles and rows
# --------------------------------------------------------------------------


def test_step_decision():
    mask = np.array([True, False])
    assert dt.step_uses_remote_tail(True, mask, False)
    assert not dt.step_uses_remote_tail(False, mask, False)  # not live yet
    assert not dt.step_uses_remote_tail(True, None, False)  # nothing sampled
    assert not dt.step_uses_remote_tail(True, mask, True)  # structured output


@pytest.mark.parametrize("remote", [False, True])
def test_roles_have_one_root_and_one_payload_pair(remote):
    roles = [dt.draft_role(r, 4, 2, remote) for r in range(4)]
    src = dt.draft_broadcast_src(4, 2, remote)
    roots = [r for r, role in enumerate(roles)
             if role in (dt.ROLE_LOCAL_ROOT, dt.ROLE_TAIL_ROOT)]
    assert roots == [src]
    if remote:
        assert roles == [dt.ROLE_RECV, dt.ROLE_RECV, dt.ROLE_TAIL_ROOT,
                         dt.ROLE_SEND_THEN_RECV]
    else:
        assert roles == [dt.ROLE_RECV] * 3 + [dt.ROLE_LOCAL_ROOT]


def test_tail_rows_follow_the_table():
    assert dt.tail_rows(3, None) == 3
    assert dt.tail_rows(3, {3: 4}) == 4
    assert dt.tail_rows(5, {3: 4}) == 5


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------


def _layout(max_rows=8, k=3, h=16):
    return dt.TailPayloadLayout(max_rows, k, h, torch.bfloat16, torch.int32)


def test_layout_is_aligned_and_sized_from_rows():
    lay = _layout()
    for rows in range(1, 9):
        buf = torch.zeros(lay.nbytes(rows), dtype=torch.uint8)
        v = lay.views(buf, rows)
        assert v.hidden.shape == (rows * 3, 16) and v.anchor.shape == (rows,)
        for t in (v.hidden, v.sample_pos, v.seeds, v.anchor, v.row_state,
                  v.temperature):
            assert t.storage_offset() * t.element_size() % 16 == 0
        assert lay.nbytes(rows) % 16 == 0
    assert lay.nbytes(1) < lay.nbytes(8)


def _last_stage_buffers(rows, num_reqs, k=3, h=16, max_reqs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    num_tokens = rows * (k + 1)
    last_hidden = torch.randn(num_tokens + 4, h, generator=g).to(torch.bfloat16)
    sample_indices = torch.zeros(max_reqs * k, dtype=torch.int64)
    sample_pos = torch.zeros(max_reqs * k, dtype=torch.int64)
    sample_idx_mapping = torch.full((max_reqs * k,), -1, dtype=torch.int32)
    slots = torch.randperm(max_reqs, generator=g)[:num_reqs].to(torch.int32)
    for r in range(num_reqs):
        for s in range(k):
            f = r * k + s
            sample_indices[f] = r * (k + 1) + 1 + s
            sample_pos[f] = 100 + 7 * r + s
            sample_idx_mapping[f] = slots[r]
    input_ids = torch.randint(0, 97, (num_tokens + 4,), generator=g,
                              dtype=torch.int32)
    anchor_indices = torch.arange(max_reqs, dtype=torch.int64) * (k + 1)
    temperature = torch.rand(max_reqs, generator=g)
    seeds = torch.randint(0, 1 << 40, (max_reqs,), generator=g)
    row_ids = torch.arange(max_reqs * k, dtype=torch.int32) // k
    return SimpleNamespace(
        last_hidden=last_hidden, sample_indices=sample_indices,
        sample_pos=sample_pos, sample_idx_mapping=sample_idx_mapping,
        input_ids=input_ids, anchor_indices=anchor_indices,
        temperature=temperature, seeds=seeds, row_ids=row_ids,
    )


def _pack(lay, b, rows):
    buf = torch.zeros(lay.nbytes(rows), dtype=torch.uint8)
    dt.pack_tail_payload(
        lay.views(buf, rows), rows, lay.num_steps, b.last_hidden,
        b.sample_indices, b.input_ids, b.anchor_indices, b.sample_pos,
        b.sample_idx_mapping, b.temperature, b.seeds, b.row_ids,
    )
    return buf


@pytest.mark.parametrize("num_reqs,rows", [(1, 1), (3, 4), (5, 6), (8, 8)])
def test_pack_carries_what_the_fused_tail_reads(num_reqs, rows):
    lay = _layout()
    b = _last_stage_buffers(rows, num_reqs)
    # Bytes survive the hop unchanged.
    v = lay.views(_pack(lay, b, rows).clone(), rows)
    n = rows * 3
    assert torch.equal(v.hidden, b.last_hidden[b.sample_indices[:n]])
    assert torch.equal(v.anchor, b.input_ids[b.anchor_indices[:rows]])
    assert torch.equal(v.sample_pos, b.sample_pos[:n])
    for f in range(n):
        slot = int(b.sample_idx_mapping[f])
        assert int(v.row_state[f]) == (f // 3 if slot >= 0 else -1)
    for r in range(num_reqs):
        slot = int(b.sample_idx_mapping[r * 3])
        assert v.temperature[r] == b.temperature[slot]
        assert v.seeds[r] == b.seeds[slot]


# --------------------------------------------------------------------------
# A CPU tail with the fused path's structure
# --------------------------------------------------------------------------

TOP_K = 4
VOCAB = 97


class _CpuDrafterHead:
    """lm_head + top-k + selector + greedy walk with the fused path's
    data flow (``DFlash2Speculator._generate_draft``), on CPU."""

    def __init__(self, h=16, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(VOCAB, h, generator=g).to(torch.bfloat16)
        self.pair = torch.randn(VOCAB, VOCAB, generator=g)
        self.proj = torch.randn(h, generator=g).to(torch.bfloat16)

    def compute_candidates(self, hidden):
        logits = (hidden.float() @ self.w.float().T)
        values, ids = torch.topk(logits, TOP_K, dim=-1)
        return ids.to(torch.int64), values

    def select(self, cand, unary, hidden, anchor):
        # [rows, k, top_k(previous), top_k(next)] like _score_edges.
        rows, k, top_k = cand.shape
        prev = torch.cat(
            [anchor.to(torch.int64).view(rows, 1, 1).expand(rows, 1, top_k),
             cand[:, :-1, :]], dim=1)
        hid = (hidden.float() @ self.proj.float()).view(rows, k, 1, 1)
        edges = self.pair[prev.unsqueeze(-1), cand.unsqueeze(-2)]
        return edges + unary.unsqueeze(-2) + hid

    @staticmethod
    def walk_into(out_tokens, row_state_of, temperature_of, seed_of):
        def walk(cand, scores, views_or_none, rows):
            k = cand.shape[1]
            for r in range(rows):
                state = row_state_of(views_or_none, r * k)
                if state < 0:
                    continue
                # temperature and seed are read (as the kernel does) but a
                # greedy walk ignores them.
                temperature_of(views_or_none, state)
                seed_of(views_or_none, state)
                previous = 0
                for s in range(k):
                    idx = int(torch.argmax(scores[r, s, previous]))
                    out_tokens[r, s] = cand[r, s, idx]
                    previous = idx
        return walk

    def fused(self, b, num_reqs, rows, out_tokens):
        """Flag-off: gather in place, slot-indexed sampling state."""
        k = 3
        n = rows * k
        hidden = b.last_hidden[b.sample_indices[:n]].view(rows, k, -1)
        cand, unary = self.compute_candidates(hidden.flatten(0, 1))
        cand = cand.view(rows, k, TOP_K)
        unary = unary.view_as(cand)
        anchor = b.input_ids[b.anchor_indices[:rows]]
        scores = self.select(cand, unary, hidden, anchor)
        walk = self.walk_into(
            out_tokens,
            lambda _, f: int(b.sample_idx_mapping[f]),
            lambda _, st: b.temperature[st],
            lambda _, st: b.seeds[st],
        )
        walk(cand, scores, None, rows)

    def from_payload(self, lay, payload, rows, out_tokens):
        """Flag-on: the tail stage, from the received bytes only."""
        views = lay.views(payload, rows)
        walk = self.walk_into(
            out_tokens,
            lambda v, f: int(v.row_state[f]),
            lambda v, st: v.temperature[st],
            lambda v, st: v.seeds[st],
        )
        dt.run_draft_tail(self.compute_candidates, self.select,
                          lambda c, s, v, r: walk(c, s, v, r),
                          views, rows, lay.num_steps, TOP_K)


@pytest.mark.parametrize("num_reqs,rows", [(1, 1), (2, 2), (3, 4), (7, 8)])
def test_moved_tail_gives_identical_drafts(num_reqs, rows):
    lay = _layout()
    head = _CpuDrafterHead()
    b = _last_stage_buffers(rows, num_reqs, seed=num_reqs)
    fused = torch.zeros(8, 3, dtype=torch.int64)
    head.fused(b, num_reqs, rows, fused)
    moved = torch.zeros(8, 3, dtype=torch.int64)
    head.from_payload(lay, _pack(lay, b, rows).clone(), rows, moved)
    assert torch.equal(fused[:num_reqs], moved[:num_reqs])


def test_row_mismatch_repack_keeps_real_rows():
    lay = _layout()
    b = _last_stage_buffers(6, 5)
    src = lay.views(_pack(lay, b, 6), 6)
    out = torch.zeros(lay.nbytes(5), dtype=torch.uint8)
    dst = lay.views(out, 5)
    dst.row_state.fill_(-1)
    dt.copy_payload_rows(src, dst, 5, 3)
    assert torch.equal(dst.hidden, src.hidden[:15])
    assert torch.equal(dst.row_state, src.row_state[:15])


# --------------------------------------------------------------------------
# Last stage: pending drafts
# --------------------------------------------------------------------------


def _entry(step, idx, drafts, keep=None, gen=None, event=None):
    n = len(idx)
    return dt.RemoteDrafts(
        step=step,
        event=event or f"e{step}",
        draft_tokens=torch.tensor(drafts, dtype=torch.int64),
        idx_mapping_np=np.array(idx, dtype=np.int32),
        keep_np=np.ones(n, dtype=bool) if keep is None else np.array(keep),
        gen_np=np.zeros(n, dtype=np.int32) if gen is None else np.array(gen),
    )


def _apply(q, batch, gen, state, waited):
    return q.apply(np.array(batch, dtype=np.int32), gen, state, waited.append,
                   lambda a: torch.as_tensor(a, dtype=torch.int64))


def test_queue_waits_only_for_the_rows_the_batch_reads():
    q = dt.RemoteDraftQueue(max_age=6)
    q.next_step()
    q.push(_entry(1, [0], [[1, 2, 3]]))
    q.push(_entry(1, [1], [[4, 5, 6]], event="e1b"))
    gen = np.zeros(8, dtype=np.int32)
    state = torch.zeros(8, 3, dtype=torch.int64)
    waited = []
    q.next_step()
    _apply(q, [5], gen, state, waited)  # neither row read: nothing waits
    assert waited == [] and len(q) == 2
    _apply(q, [0], gen, state, waited)
    assert waited == ["e1"] and state[0].tolist() == [1, 2, 3] and len(q) == 1
    _apply(q, [1], gen, state, waited)
    assert waited == ["e1", "e1b"] and state[1].tolist() == [4, 5, 6]


def test_queue_applies_in_order_so_the_newest_drafts_win():
    q = dt.RemoteDraftQueue(max_age=6)
    q.push(_entry(0, [2], [[1, 1, 1]]))
    q.push(_entry(0, [3], [[9, 9, 9]]))
    q.push(_entry(0, [2], [[2, 2, 2]]))
    state = torch.zeros(8, 3, dtype=torch.int64)
    waited = []
    _apply(q, [2], np.zeros(8, dtype=np.int32), state, waited)
    assert state[2].tolist() == [2, 2, 2] and state[3].tolist() == [9, 9, 9]
    assert len(q) == 0


def test_queue_skips_freed_slots_and_non_sampling_rows():
    q = dt.RemoteDraftQueue(max_age=6)
    q.push(_entry(0, [4, 5, 6], [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                  keep=[True, True, False]))
    gen = np.zeros(8, dtype=np.int32)
    gen[5] += 1  # slot 5 freed and reused since the send
    state = torch.full((8, 3), 7, dtype=torch.int64)
    waited = []
    _apply(q, [4], gen, state, waited)
    assert state[4].tolist() == [1, 1, 1]
    assert state[5].tolist() == [7, 7, 7] and state[6].tolist() == [7, 7, 7]


def test_queue_ages_out_entries_nobody_reads():
    q = dt.RemoteDraftQueue(max_age=2)
    q.push(_entry(0, [1], [[1, 2, 3]]))
    state = torch.zeros(8, 3, dtype=torch.int64)
    waited = []
    gen = np.zeros(8, dtype=np.int32)
    q.next_step()
    _apply(q, [7], gen, state, waited)
    assert len(q) == 1
    q.next_step()
    _apply(q, [7], gen, state, waited)
    assert len(q) == 0 and state[1].tolist() == [1, 2, 3]


def test_queue_verify_counts_differing_rows():
    q = dt.RemoteDraftQueue(max_age=6)
    e = _entry(0, [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
               keep=[True, True, False])
    e.check = torch.tensor([[1, 2, 3], [4, 0, 6], [0, 0, 0]])
    q.push(e)
    assert q.verify_report() is None
    _apply(q, [1], np.zeros(8, dtype=np.int32),
           torch.zeros(8, 3, dtype=torch.int64), [])
    # Row 3 does not sample, so it is neither checked nor counted.
    assert q.verify_report() == (2, 1)


def test_verify_flag_declared_default_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_PP_DRAFT_TAIL_VERIFY", raising=False)
    assert envs.VLLM_PP_DRAFT_TAIL_VERIFY is False


# --------------------------------------------------------------------------
# Private workspaces
# --------------------------------------------------------------------------


def test_thin_gemm_lanes_get_private_buffers():
    tg = pytest.importorskip("vllm.ampere_thin_gemm.thin_gemm")
    dev = torch.device("cpu")
    a = tg._partials(2, 4, 8, dev)
    lock0 = tg._locks(3, dev)
    with tg.workspace_lane(1):
        b = tg._partials(2, 4, 8, dev)
        lock1 = tg._locks(3, dev)
    assert a.data_ptr() != b.data_ptr() and lock0.data_ptr() != lock1.data_ptr()
    # Lane 0 keeps the keys every other caller uses.
    assert (dev.index, 2 * 4 * 8) in tg._WORKSPACE
    assert (dev.index, "lock", 3) in tg._WORKSPACE
    assert tg._partials(2, 4, 8, dev).data_ptr() == a.data_ptr()
    with pytest.raises(ValueError):
        with tg.workspace_lane(-1):
            pass


def test_thin_gemm_workspace_lookup_still_traces_in_one_graph():
    # The target model's compiled regions call thin_gemm; the lane lookup
    # must not break a full-graph trace (a ContextVar does).
    tg = pytest.importorskip("vllm.ampere_thin_gemm.thin_gemm")

    def f(x):
        buf = tg._partials(2, x.shape[0], 8, x.device)
        lock = tg._locks(3, x.device)
        return x + buf.view(-1)[: x.numel()].view_as(x) * 0 + lock.sum()

    x = torch.zeros(4, 16)
    torch._dynamo.reset()
    torch.compile(f, fullgraph=True, backend="eager")(x)
    torch._dynamo.reset()
    with tg.workspace_lane(1):
        torch.compile(f, fullgraph=True, backend="eager")(x)
    torch._dynamo.reset()


def test_flashinfer_topk_buffer_is_swapped_and_restored(monkeypatch):
    fake = types.ModuleType("flashinfer.utils")
    fake._cache_buf = {}
    pkg = types.ModuleType("flashinfer")
    pkg.utils = fake
    monkeypatch.setitem(sys.modules, "flashinfer", pkg)
    monkeypatch.setitem(sys.modules, "flashinfer.utils", fake)
    dev = torch.device("cpu")
    key = (f"radix_topk_row_states_{dev}", dev)
    shared = torch.zeros(4, dtype=torch.uint8)
    fake._cache_buf[key] = shared
    holder = {}
    with dt.private_flashinfer_topk_workspace(dev, holder):
        inside = fake._cache_buf[key]
        assert inside is not shared and inside.numel() == 1024 * 1024
        grown = torch.zeros(2 * 1024 * 1024, dtype=torch.uint8)
        fake._cache_buf[key] = grown  # FlashInfer grew it
    assert fake._cache_buf[key] is shared
    assert holder["buf"] is grown
    with dt.private_flashinfer_topk_workspace(dev, holder):
        assert fake._cache_buf[key] is grown  # reused, not re-zeroed


def test_agree_names_the_mismatch():
    good = {"ok": True, "sig": [1], "mod": {"a": 1}, "layout": (8,),
            "fp64": False, "probabilistic": False, "adaptive": False}
    assert dt.DraftTailController.agree(good, dict(good)) == (True, "")
    ok, reason = dt.DraftTailController.agree(good, dict(good, sig=[2]))
    assert not ok and "sig" in reason
    ok, reason = dt.DraftTailController.agree(dict(good, ok=False, reason="x"),
                                              good)
    assert not ok and reason == "x"
    ok, reason = dt.DraftTailController.agree(dict(good, probabilistic=True),
                                              good)
    assert not ok


# --------------------------------------------------------------------------
# Memory accounting: the copy is inside the model's memory profile, and what
# runs after the KV pool is sized (the copy check) stays small
# --------------------------------------------------------------------------


def test_tail_copy_is_allocated_inside_the_model_memory_profile():
    """The tail stage's lm_head + selector copy is allocated by
    DraftTailController.load inside load_model's DeviceMemoryProfiler, so it
    is part of model_memory_usage and the KV pool is sized without it."""
    src = pathlib.Path(dt.__file__).with_name("model_runner.py").read_text()
    tree = ast.parse(src)
    load_model = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    profiled = [w for w in ast.walk(load_model) if isinstance(w, ast.With)
                and any(isinstance(i.context_expr, ast.Call)
                        and getattr(i.context_expr.func, "id", "")
                        == "DeviceMemoryProfiler" for i in w.items)]
    assert len(profiled) == 1

    def is_tail_load(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "load"
                and isinstance(n.func.value, ast.Attribute)
                and n.func.value.attr == "draft_tail")

    inside = [n for n in ast.walk(profiled[0]) if is_tail_load(n)]
    everywhere = [n for n in ast.walk(load_model) if is_tail_load(n)]
    assert len(inside) == 1 and len(everywhere) == 1


class _Fp64Peak(torch.utils._python_dispatch.TorchDispatchMode):
    """Largest new tensor (bytes) any op creates, and largest fp64 one."""

    def __init__(self):
        super().__init__()
        self.peak = 0
        self.peak64 = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        # outputs sharing an input's storage (views, in-place ops) are not new
        seen = {a.untyped_storage().data_ptr()
                for a in torch.utils._pytree.tree_leaves((args, kwargs))
                if isinstance(a, torch.Tensor)}
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if (isinstance(t, torch.Tensor)
                    and t.untyped_storage().data_ptr() not in seen):
                n = t.numel() * t.element_size()
                self.peak = max(self.peak, n)
                if t.dtype == torch.float64:
                    self.peak64 = max(self.peak64, n)
        return out


@pytest.mark.parametrize("shape,transpose", [((1000, 64), False),
                                              ((64, 1000), True),
                                              ((3, 5000), False),
                                              ((777,), False),
                                              ((), False)])
def test_weight_checksum_bounds_the_fp64_transient(shape, transpose):
    g = torch.Generator().manual_seed(1)
    t = torch.randn(shape, generator=g).to(torch.bfloat16)
    if transpose:
        t = t.t()
    chunk = 4096
    with _Fp64Peak() as mode:
        got = dt.weight_checksum(t, chunk)
    assert mode.peak64 <= chunk * 8
    assert got == dt.weight_checksum(t.contiguous(), chunk)
    assert got == pytest.approx(float(t.double().sum()), rel=1e-12, abs=1e-9)


class _Lin(torch.nn.Module):
    def __init__(self, n, h, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(
            torch.randn(n, h, generator=g).to(torch.bfloat16),
            requires_grad=False)


def _copy_pair(monkeypatch, chunk):
    """The last stage's and the tail stage's _copy_weights on CPU, with the
    send/recv and object channel in process."""
    monkeypatch.setattr(dt, "CHECKSUM_CHUNK", chunk)
    monkeypatch.setattr(dt.torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(dt.torch.cuda, "empty_cache", lambda *a, **k: None)
    wire: deque = deque()
    objs: deque = deque()
    monkeypatch.setattr(dt.torch.distributed, "send",
                        lambda t, dst, group: wire.append(t.clone()))
    monkeypatch.setattr(dt.torch.distributed, "recv",
                        lambda t, src, group: t.copy_(wire.popleft()))
    pp = SimpleNamespace(ranks=list(range(PP)),
                         send_object=lambda o, dst: objs.append(o),
                         recv_object=lambda src: objs.popleft())
    gate = dt.DraftTailGate(requested=TAIL, stage=TAIL, reason="")
    cfg = SimpleNamespace()
    last = dt.DraftTailController(cfg, torch.device("cpu"), gate, PP - 1, PP)
    tail = dt.DraftTailController(cfg, torch.device("cpu"), gate, TAIL, PP)
    for c in (last, tail):
        c.handler = SimpleNamespace(tail_group=object())
    last.speculator = SimpleNamespace(model=SimpleNamespace(
        lm_head=_Lin(512, 64, 1),
        model=SimpleNamespace(candidate_selector=_Lin(32, 64, 2))))
    tail.module = SimpleNamespace(lm_head=_Lin(512, 64, 3),
                                  candidate_selector=_Lin(32, 64, 4))
    return pp, last, tail


def test_copy_check_after_kv_sizing_allocates_no_weight_sized_tensor(monkeypatch):
    """The copy check runs after the KV pool is sized: neither stage may
    allocate an fp64 image of the weights (the 4.73 GiB lm_head .double()
    that ran the last stage out of memory), only chunk-sized partials; the
    tail stage receives into the copy load() already allocated."""
    chunk = 2048
    pp, last, tail = _copy_pair(monkeypatch, chunk)
    lm_bytes = 512 * 64 * 2
    with _Fp64Peak() as m_last:
        assert last._copy_weights(pp, TAIL) == (True, "")
    with _Fp64Peak() as m_tail:
        assert tail._copy_weights(pp, TAIL) == (True, "")
    assert m_last.peak64 <= chunk * 8 and m_tail.peak64 <= chunk * 8
    # the only weight-sized tensors are the last stage's in-process send
    # clones (the wire here, NCCL on a GPU); the tail stage creates none
    assert m_tail.peak < lm_bytes
    assert torch.equal(tail.module.lm_head.weight,
                       last.speculator.model.lm_head.weight)


def test_copy_check_detects_a_corrupt_copy(monkeypatch):
    pp, last, tail = _copy_pair(monkeypatch, 2048)
    real = dt.torch.distributed.recv

    def bad(t, src, group):
        real(t, src, group)
        if t.shape[0] == 512:
            t[7, 3] += 1.0

    monkeypatch.setattr(dt.torch.distributed, "recv", bad)
    assert last._copy_weights(pp, TAIL) == (True, "")
    ok, reason = tail._copy_weights(pp, TAIL)
    assert not ok and "does not match" in reason


# --------------------------------------------------------------------------
# Start-up banners through the real vLLM logger (its *_once helpers cache
# their arguments, so every argument must be hashable)
# --------------------------------------------------------------------------


def _real_logger():
    from vllm.logger import _VllmLogger  # noqa: F401  (the patched methods)

    assert type(dt.logger).__name__ != "MagicMock"
    assert callable(dt.logger.info_once) and callable(dt.logger.warning_once)
    return dt.logger


def _run_finalize(monkeypatch, rank, verdict_ok):
    """finalize() on one rank with an in-process PP group: the last stage's
    info, the tail stage's info and the verdict arrive in finalize's order."""
    _real_logger()
    import vllm.distributed.parallel_state as ps

    info = {"ok": True, "sig": [("w", (4, 2), "torch.bfloat16", (2, 1))],
            "mod": {"top_k": 16}, "layout": (8, K, 16, "torch.bfloat16",
                                             "torch.int32"),
            "fp64": False, "probabilistic": False, "adaptive": False}
    last = dict(info, rows_table={1: 1, 2: 2, 3: 4, 4: 4})
    verdict = (verdict_ok, "" if verdict_ok else "the copy does not match")
    sent = deque([(PP - 1, last), (TAIL, dict(info)), (TAIL, verdict)])

    class _Group:
        def broadcast_object(self, obj, src):
            want_src, value = sent.popleft()
            assert src == want_src
            return value

    monkeypatch.setattr(ps, "get_pp_group", lambda: _Group())
    gate = dt.DraftTailGate(requested=TAIL, stage=TAIL, reason="")
    c = dt.DraftTailController(SimpleNamespace(), torch.device("cpu"), gate,
                               rank, PP)
    mine = last if rank == PP - 1 else dict(info)
    monkeypatch.setattr(c, "_local_info", lambda: mine)
    monkeypatch.setattr(c, "_copy_weights", lambda pp, idx: (verdict_ok, ""))
    monkeypatch.setattr(c, "_warm_up", lambda: None)
    c.finalize()
    assert not sent
    return c


@pytest.mark.parametrize("rank", range(4))  # PP
def test_finalize_banner_on_every_rank(monkeypatch, rank):
    c = _run_finalize(monkeypatch, rank, True)
    assert c.live and c.rows_table == {1: 1, 2: 2, 3: 4, 4: 4}


@pytest.mark.parametrize("rank", [0, 2, 3])  # other, TAIL, last
def test_finalize_off_banner(monkeypatch, rank):
    c = _run_finalize(monkeypatch, rank, False)
    assert not c.live


def test_create_refusal_banner(monkeypatch):
    _real_logger()
    monkeypatch.setenv(FLAG, str(TAIL))
    assert dt.DraftTailController.create(_config(tp=2), "cpu", 0, PP) is None


# --------------------------------------------------------------------------
# The real DFlash2Speculator: split mode (gather + tail from the payload)
# gives the fused path's drafts. CPU, with a Python stand-in for the Triton
# selector walk (same indexing), so any attribute the split path reads from
# the speculator or its CandidateSampler is exercised.
# --------------------------------------------------------------------------


def test_gate_refuses_lilicorr():
    cfg = _config()
    cfg.speculative_config.draft_model_config.architectures = [
        "DFlash2DraftModel", "LiLiCorrDraftModel"]
    gate = dt.draft_tail_gate(cfg, TAIL)
    assert not gate.enabled and "LiLiCorr" in gate.reason


class _WalkStub:
    """_selector_walk_kernel[(rows,)](...) in Python, greedy; logs what each
    row reads through its request-state index."""

    def __init__(self):
        self.log = []

    def __getitem__(self, grid):
        def run(scores, cand, sample_pos, req_state, temperature, seeds, tokens,
                realized, num_steps, top_k, BLOCK_K, SAMPLE_PROBABILISTIC,
                USE_FP64, num_warps):
            sc, ca = scores.reshape(-1), cand.reshape(-1)
            rs, tok, real = req_state.reshape(-1), tokens.reshape(-1), \
                realized.reshape(-1)
            for row in range(grid[0]):
                st = int(rs[row * num_steps])
                if st < 0:
                    continue
                prev = 0
                for step in range(num_steps):
                    flat = row * num_steps + step
                    base = (flat * top_k + prev) * top_k
                    v = sc[base:base + top_k]
                    idx = int(torch.argmax(v))
                    real[flat * top_k:(flat + 1) * top_k] = v
                    tok[flat] = ca[flat * top_k + idx]
                    self.log.append((row, step, float(temperature[st]),
                                     int(seeds[st]), int(sample_pos[flat])))
                    prev = idx
        return run


def _speculator(num_reqs, pad, split, seed=0):
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as sm

    k, top_k, h, nq, max_reqs, vocab = K, 4, 16, K + 1, 8, 97
    g = torch.Generator().manual_seed(seed)
    s = object.__new__(sm.DFlash2Speculator)
    s.device = torch.device("cpu")
    s.dtype = torch.float32
    s.hidden_size = h
    s.max_num_reqs = max_reqs
    s.num_speculative_steps = k
    s.num_query_per_req = nq
    s.top_k = top_k
    s.candidate_sampler = sm.CandidateSampler(max_reqs, k, top_k, s.device)
    s.draft_tokens = torch.zeros(max_reqs, k, dtype=torch.int64)
    s.draft_logits = None
    s.use_fp64_gumbel = False
    s.enable_adaptive_verification = False
    s.split_tail = False
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(0, vocab, (max_reqs * nq,), generator=g,
                                dtype=torch.int32))
    rows = num_reqs + pad
    slots = torch.randperm(max_reqs, generator=g)[:rows].int()
    slots[num_reqs:] = -1
    s.sample_idx_mapping = torch.full((max_reqs * k,), -1, dtype=torch.int32)
    s.sample_idx_mapping[:rows * k] = slots.repeat_interleave(k)
    s.sample_indices = (torch.arange(rows * k) // k * nq + torch.arange(rows * k)
                        % k + 1).to(torch.int64)
    s.sample_indices = torch.cat([s.sample_indices, torch.zeros(
        max_reqs * k - rows * k, dtype=torch.int64)])
    s.sample_pos = torch.randint(1, 5000, (max_reqs * k,), generator=g)
    s.temperature = torch.rand(max_reqs, generator=g)
    s.seeds = torch.randint(0, 1 << 30, (max_reqs,), generator=g)
    hidden = torch.randn(rows * nq, h, generator=g)
    w = torch.randn(h, top_k, generator=g)

    def compute_candidates(x):
        unary = x @ w
        ids = (x.abs().sum(-1, keepdim=True) * 1000).long() % vocab
        return (ids + torch.arange(top_k)) % vocab, unary

    def selector(cand, unary, hid, anchor):
        return (unary[..., None, :] + 0.5 * unary[..., :, None]
                + 0.01 * anchor.float()[:, None, None, None]
                + 0.1 * hid.sum(-1)[..., None, None])

    s.model = SimpleNamespace(compute_candidates=compute_candidates,
                              model=SimpleNamespace(candidate_selector=selector))
    s._run_model = lambda *a, **kw: hidden
    if split:
        s.enable_split_tail()
    return s, rows


@pytest.mark.parametrize("num_reqs,pad", [(1, 0), (3, 1), (5, 3)])
def test_split_tail_matches_the_fused_speculator(monkeypatch, num_reqs, pad):
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as sm

    out = {}
    for split in (False, True):
        stub = _WalkStub()
        monkeypatch.setattr(sm, "_selector_walk_kernel", stub)
        s, rows = _speculator(num_reqs, pad, split)
        s._generate_draft(rows, rows * s.num_query_per_req, None, None, None)
        if split:
            s.run_tail_local(rows)
        out[split] = (s.draft_tokens[:num_reqs].clone(),
                      s.candidate_sampler.scores[:num_reqs].clone(), stub.log)
    assert torch.equal(out[True][0], out[False][0])
    assert torch.equal(out[True][1], out[False][1])
    assert out[True][2] == out[False][2] and out[False][2]


# --------------------------------------------------------------------------
# Simulated 4-stage pipeline through the real PPHandler methods
# --------------------------------------------------------------------------

PP = 4
TAIL = 2
K = 3
MAX_REQS = 8


class _FakeComm:
    """In-process collectives with GPU-like deferred semantics: a receive
    registers its tensor and is filled when the root (or sender) runs,
    whichever comes first. Records every rank's collective order."""

    def __init__(self):
        self.bcast: dict[int, dict] = {}
        self.p2p: dict[tuple[int, int], deque] = {}
        self.pending_p2p: dict[tuple[int, int], deque] = {}
        self.seq = [0] * PP
        self.log = [[] for _ in range(PP)]
        self.rank = 0

    def broadcast(self, tensor, src, group=None):
        i = self.seq[self.rank]
        self.seq[self.rank] += 1
        self.log[self.rank].append(("bcast", i, src, tuple(tensor.shape)))
        slot = self.bcast.setdefault(i, {"data": None, "waiting": []})
        if self.rank == src:
            slot["data"] = tensor.clone()
            for t in slot["waiting"]:
                t.copy_(slot["data"])
        elif slot["data"] is not None:
            tensor.copy_(slot["data"])
        else:
            slot["waiting"].append(tensor)

    def send(self, tensor, dst, group=None):
        self.log[self.rank].append(("send", dst, tensor.numel()))
        key = (self.rank, dst)
        waiting = self.pending_p2p.get(key)
        if waiting:
            waiting.popleft().copy_(tensor)
        else:
            self.p2p.setdefault(key, deque()).append(tensor.clone())

    def recv(self, tensor, src, group=None):
        self.log[self.rank].append(("recv", src, tensor.numel()))
        key = (src, self.rank)
        queued = self.p2p.get(key)
        if queued:
            tensor.copy_(queued.popleft())
        else:
            self.pending_p2p.setdefault(key, deque()).append(tensor)


class _Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)

    def wait_stream(self, other):
        pass

    def record_event(self):
        return object()


def _make_handler(rank, remote_on):
    h = PPHandler.__new__(PPHandler)
    h.is_last_rank = rank == PP - 1
    h.last_rank = PP - 1
    h.max_sample_len = K + 1
    h.num_speculative_steps = K
    h.device = torch.device("cpu")
    h.main_stream = _Stream()
    h.broadcast_stream = _Stream()
    h.queue = deque() if h.is_last_rank else deque([None] * PP)
    h.req_idx_gen_np = np.zeros(MAX_REQS, dtype=np.int32)
    h.broadcast_group = "bcast"
    h.aux_hidden_state_relay_keys = ()
    h.split_draft_event = False
    h.pending_drafts = None
    h.draft_tail_stage = TAIL if remote_on else -1
    h.draft_tail_src = TAIL if remote_on else -1
    h.tail_group = "tail" if remote_on else None
    h.remote_drafts = (dt.RemoteDraftQueue(max_age=PP + 2)
                       if remote_on and h.is_last_rank else None)
    return h


@contextlib.contextmanager
def _as_rank(comm, rank):
    comm.rank = rank
    yield


def _batch(idx, computed, prefill, scheduled, structured=False):
    idx = np.array(idx, dtype=np.int32)
    return SimpleNamespace(
        num_reqs=len(idx),
        idx_mapping=torch.as_tensor(idx),
        idx_mapping_np=idx,
        num_computed_tokens_np=np.array(computed),
        prefill_len_np=np.array(prefill),
        num_scheduled_tokens=np.array(scheduled),
        has_structured_output_reqs=structured,
    )


def _schedule(seed, steps=40):
    """Micro-batches of decodes (spread over the in-flight slots), final and
    non-final prefill chunks, structured-output steps, and request slots that
    are freed and reused."""
    rng = np.random.default_rng(seed)
    out = []
    for step in range(steps):
        n = int(rng.integers(1, 4))
        idx = rng.choice(MAX_REQS, size=n, replace=False)
        kind = rng.random()
        if kind < 0.15:  # non-final prefill chunk for everyone: nothing sampled
            computed, prefill, sched = [0] * n, [100] * n, [10] * n
        elif kind < 0.3:  # mixed: first row mid-prefill
            computed = [0] + [50] * (n - 1)
            prefill = [100] + [10] * (n - 1)
            sched = [10] + [K + 1] * (n - 1)
        else:
            computed, prefill, sched = [50] * n, [10] * n, [K + 1] * n
        structured = rng.random() < 0.1
        freed = [int(x) for x in rng.choice(MAX_REQS, size=int(rng.integers(0, 2)))]
        out.append((idx.tolist(), computed, prefill, sched, structured, freed))
    return out


def _run_pipeline(seed, remote_on, monkeypatch):
    comm = _FakeComm()
    monkeypatch.setattr(pp_utils, "async_tensor_h2d",
                        lambda data, device: torch.as_tensor(data))
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None,
                        raising=False)
    monkeypatch.setattr(torch.distributed, "broadcast", comm.broadcast)
    monkeypatch.setattr(torch.distributed, "send", comm.send)
    monkeypatch.setattr(torch.distributed, "recv", comm.recv)

    handlers = [_make_handler(r, remote_on) for r in range(PP)]
    states = [torch.zeros(MAX_REQS, K, dtype=torch.int64) for _ in range(PP)]
    lay = dt.TailPayloadLayout(MAX_REQS, K, 16, torch.bfloat16, torch.int32)
    head = _CpuDrafterHead()
    rows_table = {n: n + (n % 2) if n < MAX_REQS else n for n in range(1, 9)}
    tail_out = torch.zeros(MAX_REQS, K, dtype=torch.int64)
    reads = [[] for _ in range(PP)]  # the draft rows each stage reads
    # Rows whose drafts a step may read: the request's previous step sampled
    # (a decode never follows a non-final chunk without the final one).
    sampled_last = np.zeros(MAX_REQS, dtype=bool)

    ctl = dt.DraftTailController.__new__(dt.DraftTailController)
    ctl.live = remote_on
    ctl.rows_table = rows_table
    ctl.layout = lay
    ctl.is_tail = False  # set per rank below

    def tail_compute(rows, num_reqs):
        def compute(payload, draft_tokens):
            head.from_payload(lay, payload, rows, tail_out)
            draft_tokens.copy_(tail_out[:num_reqs])
        return compute

    for step, (idx, computed, prefill, sched, structured, freed) in enumerate(
        _schedule(seed)
    ):
        batch = _batch(idx, computed, prefill, sched, structured)
        # Every stage frees the same slots at the start of the step.
        for h in handlers:
            for slot in freed:
                h.on_req_idx_freed(slot)
        sampled_last[freed] = False
        comparable = torch.as_tensor(sampled_last[batch.idx_mapping_np])
        # Each stage lands earlier drafts, then reads this batch's rows.
        for r, h in enumerate(handlers):
            with _as_rank(comm, r):
                if h.is_last_rank:
                    h.apply_remote_drafts(batch.idx_mapping_np, states[r])
                else:
                    h.get_prev_sampled_outputs(states[r])
            reads[r].append(states[r][batch.idx_mapping][comparable].clone())
        # Last stage: sample, broadcast counts, draft.
        rows = rows_table[batch.num_reqs]
        b = _last_stage_buffers(rows, batch.num_reqs, seed=1000 + step)
        b.sample_idx_mapping[: batch.num_reqs * K] = torch.as_tensor(
            np.repeat(batch.idx_mapping_np, K))
        last = handlers[PP - 1]
        with _as_rank(comm, PP - 1):
            last.broadcast(torch.zeros(batch.num_reqs, K + 1, dtype=torch.int64),
                           torch.ones(batch.num_reqs, dtype=torch.int32),
                           torch.zeros(batch.num_reqs, dtype=torch.int32), batch)
            ctl.is_tail = False
            remote = ctl.remote_step(batch)
            if remote:
                buf = torch.zeros(lay.nbytes(rows), dtype=torch.uint8)
                dt.pack_tail_payload(
                    lay.views(buf, rows), rows, K, b.last_hidden,
                    b.sample_indices, b.input_ids, b.anchor_indices,
                    b.sample_pos, b.sample_idx_mapping, b.temperature,
                    b.seeds, b.row_ids)
                last.send_draft_tail(buf, batch)
            else:
                out = torch.zeros(MAX_REQS, K, dtype=torch.int64)
                head.fused(b, batch.num_reqs, rows, out)
                states[PP - 1][batch.idx_mapping] = out[: batch.num_reqs]
                last.broadcast_drafts(states[PP - 1], batch)
        # Earlier stages receive (the tail stage computes and roots).
        for r in [TAIL] + [x for x in range(PP - 1) if x != TAIL]:
            ctl.is_tail = r == TAIL
            step_obj = ctl.step(batch) if remote_on else None
            if step_obj is not None and step_obj.compute is not None:
                step_obj.compute = tail_compute(rows, batch.num_reqs)
            with _as_rank(comm, r):
                handlers[r].receive(batch, step_obj)
        need = pp_utils.compute_need_sampled_mask(batch)
        sampled_last[batch.idx_mapping_np] = (
            need if need is not None else np.zeros(batch.num_reqs, dtype=bool)
        )
    return reads, comm


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_simulated_pipeline_drafts_identical_with_flag_on_and_off(
    seed, monkeypatch
):
    with monkeypatch.context() as m:
        off_reads, off_comm = _run_pipeline(seed, False, m)
    with monkeypatch.context() as m:
        on_reads, on_comm = _run_pipeline(seed, True, m)
    compared = 0
    for r in range(PP):
        assert len(off_reads[r]) == len(on_reads[r])
        for a, b in zip(off_reads[r], on_reads[r]):
            assert torch.equal(a, b)
            compared += a.shape[0]
    assert compared > 0
    # Every stage issued the same broadcasts in the same order with the same
    # root and size; the payload hop pairs the last stage with the tail stage.
    for comm in (off_comm, on_comm):
        bcasts = [[e for e in log if e[0] == "bcast"] for log in comm.log]
        assert all(len(x) == len(bcasts[0]) for x in bcasts)
        for ops in zip(*bcasts):
            assert len({(o[2], o[3]) for o in ops}) == 1
    sends = [e for e in on_comm.log[PP - 1] if e[0] == "send"]
    recvs = [e for e in on_comm.log[TAIL] if e[0] == "recv"]
    assert len(sends) == len(recvs) > 0
    assert [s[2] for s in sends] == [r[2] for r in recvs]
    assert all(not [e for e in on_comm.log[r] if e[0] in ("send", "recv")]
               for r in range(PP) if r not in (TAIL, PP - 1))
    # With the flag on, some drafts are rooted at the tail stage.
    roots = {e[2] for e in on_comm.log[0] if e[0] == "bcast"}
    assert TAIL in roots and PP - 1 in roots


# --------------------------------------------------------------------------
# GPU (skipped without one)
# --------------------------------------------------------------------------

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@_cuda
@pytest.mark.parametrize("fp64", [False, True])
def test_walk_from_payload_matches_slot_indexed_walk(fp64):
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _selector_walk_kernel,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    rows, k, top_k, max_reqs = 6, 3, 16, 8
    num_reqs = 5  # one padded row
    n = rows * k
    scores = torch.randn(rows, k, top_k, top_k, device=dev, generator=g)
    cand = torch.randint(0, 150000, (rows, k, top_k), device=dev, generator=g)
    sample_pos = torch.randint(10, 5000, (n,), device=dev, generator=g)
    slots = torch.tensor([6, 1, 4, 0, 3], dtype=torch.int32, device=dev)
    idx_map = torch.full((n,), -1, dtype=torch.int32, device=dev)
    idx_map[: num_reqs * k] = slots.repeat_interleave(k)
    temperature = torch.rand(max_reqs, device=dev, generator=g) + 0.5
    seeds = torch.randint(0, 1 << 40, (max_reqs,), device=dev, generator=g)
    row_ids = torch.arange(n, dtype=torch.int32, device=dev) // k
    row_state = torch.where(idx_map >= 0, row_ids, torch.full_like(row_ids, -1))
    slot_rows = idx_map[0:n:k].clamp(min=0).long()
    for probabilistic in (False, True):
        a = torch.zeros(max_reqs, k, dtype=torch.int64, device=dev)
        b = torch.zeros_like(a)
        sa = torch.zeros(n * top_k, device=dev)
        sb = torch.zeros_like(sa)
        common = dict(num_steps=k, top_k=top_k, BLOCK_K=16,
                      SAMPLE_PROBABILISTIC=probabilistic, USE_FP64=fp64,
                      num_warps=1)
        _selector_walk_kernel[(rows,)](scores, cand, sample_pos, idx_map,
                                       temperature, seeds, a, sa, **common)
        _selector_walk_kernel[(rows,)](scores, cand, sample_pos, row_state,
                                       temperature[slot_rows], seeds[slot_rows],
                                       b, sb, **common)
        assert torch.equal(a, b) and torch.equal(sa, sb)


@_cuda
def test_thin_gemm_lane_on_a_side_stream_is_bitwise_serial():
    tg = pytest.importorskip("vllm.ampere_thin_gemm.thin_gemm")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("thin GEMM is sm_80 only")
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    # lm_head-shaped (candidate pass) and a stage-shaped GEMM, same M.
    w_head = torch.randn(154880, 4096, device=dev, generator=g).bfloat16()
    w_stage = torch.randn(8192, 4096, device=dev, generator=g).bfloat16()
    x = torch.randn(12, 4096, device=dev, generator=g).bfloat16()
    ref_head = tg.thin_gemm(x, w_head).clone()
    ref_stage = tg.thin_gemm(x, w_stage).clone()
    side = torch.cuda.Stream()
    for _ in range(20):
        outs = []
        with torch.cuda.stream(side), tg.workspace_lane(1):
            outs.append(tg.thin_gemm(x, w_head))
        for _ in range(4):
            outs.append(tg.thin_gemm(x, w_stage))
        torch.cuda.synchronize()
        assert torch.equal(outs[0], ref_head)
        assert all(torch.equal(o, ref_stage) for o in outs[1:])
