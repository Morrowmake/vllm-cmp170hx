# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the RecoverSSM commit fold (VLLM_AMPERE_KDA_FOLD,
vllm_sm80/kernels/kda/recover.py).

CPU part (no GPU; the Triton kernels through the Triton interpreter in a
subprocess): the switch and its banner; the fold plan against the upstream
commit plan; the per-step gather; multi-step sequences through the engine's
commit / gather / pre-copy / postprocess path (kda_fold_world.py, the Gluon
verify modelled) with byte identity of everything a consumer reads, on
random schedules at block size 16 and at the 1,152-token boundary, in align
and in "none" mode, and on schedules replayed from captured commits
(VLLM_SM80_REPLAY_DIR/kda, optional); the same sequences with deliberately broken
folds must fail; request lifecycle: streaming-input re-adds (the owed commit
is written first), a preempted request resumed in a later step and one
preempted and resumed in the same step (never written); the host cost of a
step (launches, no sync, no allocation); the off path; the CUDA-graph
buffers and warmup; the real TP4 config's metadata.
GPU part (skipped without CUDA): the real verify kernel, fold against
recover + commit, bitwise on one card, per kernel and over sequences, and
under CUDA-graph capture.

    python tests/kernels/test_ampere_kda_fold.py           # CPU part
    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_fold.py
    python tests/kernels/test_ampere_kda_fold.py test_gather   # named cases
    (under pytest, select cases with -k)
"""

import contextlib
import os
import subprocess
import sys
import types

try:
    import pytest
except ImportError:
    pytest = None
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.kernels import kda_fold_world as W  # noqa: E402
from tests.kernels import replay_inputs as replay  # noqa: E402


def _interpret(name, timeout=3600):
    """Run `name` of INTERPRETED in a fresh interpreter with TRITON_INTERPRET=1
    and no GPU (Triton reads the variable at import)."""
    import vllm

    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "interpret", name],
                       env=env, text=True, capture_output=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-8000:]
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")


# ------------------------------------------------------------------ switch
def test_switch_and_banner():
    """Off unless VLLM_AMPERE_KDA_FOLD=1; on only where RecoverSSM is on and
    without data parallelism; the banner names the switch."""
    from vllm import envs
    from vllm_sm80.kernels.kda import recover as kr

    assert "VLLM_AMPERE_KDA_FOLD" in envs.environment_variables
    saved = os.environ.pop("VLLM_AMPERE_KDA_FOLD", None)
    undo = W.env(False)
    try:
        os.environ.pop("VLLM_AMPERE_KDA_FOLD", None)
        assert not envs.VLLM_AMPERE_KDA_FOLD
        assert not kr.kda_fold_enabled(W.config())
    finally:
        undo()
    undo = W.env(True)
    msgs = []
    orig = kr.logger.info_once
    kr.logger.info_once = lambda msg, *a, **k: msgs.append(msg % a if a else msg)
    try:
        assert kr.kda_fold_enabled(W.config())
        assert kr.kda_fold_enabled(W.config(mode="none"))
        assert not kr.kda_fold_enabled(W.config(pp=4, tp=1))
        assert not kr.kda_fold_enabled(W.config(dp=2))
        os.environ["VLLM_AMPERE_DECODE_KERNELS"] = "0"
        assert not kr.kda_fold_enabled(W.config())
    finally:
        kr.logger.info_once = orig
        undo()
        if saved is not None:
            os.environ["VLLM_AMPERE_KDA_FOLD"] = saved
    assert any("commit fold active (VLLM_AMPERE_KDA_FOLD=1)" in m for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_FOLD=1 set but off" in m and "data parallel" in m
               for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_FOLD=1 set but off" in m and "RecoverSSM is off" in m
               for m in msgs), msgs


# ------------------------------------------------------------------ interpreted
def _plan_vs_upstream():
    """The fold plan is the upstream plan for every row (true lengths, final
    and boundary owners); eager rows keep the upstream commit length, deferred
    rows owe it in pend and commit nothing now; the split is the documented
    rule."""
    from vllm_sm80.kernels.kda import recover as kr
    from vllm_sm80.kernels.kda.commit_plan import _prepare_commit_plan_kernel

    B, sql = 16, 8
    i32 = dict(dtype=torch.int32)
    checked = deferred = 0
    for nc in range(0, 3 * B + 1):
        for q in range(0, sql + 1):
            for ns in (-1, 0, 1, 2, q - 1, q, q + 3):
                for src_null in (False, True):
                    for req in (0, -1):
                        for slot in (3, -1):
                            fnc = nc + max(0, min(ns, q, sql))
                            ncols = max(fnc, nc + q, 1)
                            table = torch.zeros(1, 8, **i32)
                            table[0, :(ncols - 1) // B + 1] = torch.arange(
                                10, 10 + (ncols - 1) // B + 1, **i32)
                            col = (nc + q - 1) // B if nc + q > 0 else 0
                            src = torch.tensor([0 if src_null else int(table[0, col])], **i32)
                            args = (torch.tensor([ns], **i32),
                                    torch.tensor([req], **i32) if req < 0 else None, src,
                                    torch.tensor([0, q], **i32), table, torch.tensor([nc], **i32))
                            up = [torch.full((1,), -7, **i32) for _ in range(4)]
                            _prepare_commit_plan_kernel[(1,)](
                                *args, *up, 0, B, 8, 1, 1, 1, 1, 8, 1, 1,
                                SPEC_QUERY_LEN=sql, num_warps=1)
                            fo = [torch.full((1,), -7, **i32) for _ in range(5)]
                            pend = torch.full((8,), -5, **i32)
                            srcs = torch.full((8,), -5, **i32)
                            rows = torch.tensor([slot], **i32)
                            kr._kda_fold_plan_kernel[(1,)](
                                *args, rows, pend, srcs, fo[0], fo[1], fo[2], fo[3], fo[4],
                                0, B, 8, 1, 1, 1, 1, 8, 1, 1,
                                SPEC_QUERY_LEN=sql, NEXT_WINDOW=sql, num_warps=1)
                            cl, fin, bnd, blen = (int(t) for t in up)
                            eager, true, ffin, fbnd, fblen = (int(t) for t in fo)
                            assert true == cl and ffin == fin and fbnd == bnd and fblen == blen
                            n = cl
                            rule = (n > 0 and fin == int(src) and fnc < (nc // B + 1) * B
                                    and (fnc + sql - 1) // B == (fnc - 1) // B)
                            assert eager == (0 if rule else n), (nc, q, ns, eager, n)
                            owed = int(pend[3])
                            if slot >= 0 and (req >= 0):
                                assert owed == (n if rule else 0), (nc, q, ns, owed)
                                if rule:      # the block a flush writes
                                    assert int(srcs[3]) == int(src), (nc, q, ns)
                            else:
                                assert owed == -5
                            deferred += rule
                            checked += 1
    # none mode: every valid row defers
    for ns in (0, 1, 5, 9):
        fo = [torch.full((1,), -7, **i32) for _ in range(5)]
        pend = torch.full((8,), -5, **i32)
        srcs = torch.full((8,), -5, **i32)
        kr._kda_fold_plan_kernel[(1,)](
            torch.tensor([ns], **i32), None, torch.tensor([4], **i32),
            torch.tensor([0, 6], **i32), None, None, torch.tensor([2], **i32), pend, srcs,
            *fo, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
            SPEC_QUERY_LEN=8, NEXT_WINDOW=8, num_warps=1)
        n = max(0, min(ns, 6))
        assert int(fo[0]) == 0 and int(fo[1]) == n and int(pend[2]) == n
        assert n == 0 or int(srcs[2]) == 4
    assert deferred > 0
    print(f"PASS plan: {checked} align rows ({deferred} deferred) + none mode")


def _gather():
    """Per verify row: the request slot and each group's owed length; rows
    beyond the step, filtered rows and padding owe nothing."""
    from vllm_sm80.kernels.kda import recover as kr

    kr.FoldState._states.clear()
    f = kr.FoldState("cpu", 8, 64)
    for names in (("a",), ("b",), ("a",)):
        f.group(names)
    assert f.groups == {("a",): 0, ("b",): 1}
    f.pend[0] = torch.arange(8, dtype=torch.int32) + 1
    f.pend[1] = torch.arange(8, dtype=torch.int32) + 11
    f.lens.fill_(-3)
    f.record_rows.fill_(-3)
    idx = torch.tensor([5, 2, 7, -1, 4], dtype=torch.int32)
    f.gather(idx, None, 4)                              # pure: rows are batch rows
    assert f.record_rows[:6].tolist() == [5, 2, 7, -1, -1, -1]
    assert f.lens[0, :6].tolist() == [6, 3, 8, 0, 0, 0]
    assert f.lens[1, :6].tolist() == [16, 13, 18, 0, 0, 0]
    assert bool((f.lens[:2, 6:] == 0).all()) and bool((f.record_rows[6:] == -1).all())
    f.gather(idx, torch.tensor([4, 0, 2], dtype=torch.int32), 3)   # mixed
    assert f.record_rows[:4].tolist() == [4, 5, 7, -1]
    assert f.lens[0, :4].tolist() == [5, 6, 8, 0]
    f.reset_slot(4)
    assert int(f.pend[0, 4]) == 0 and int(f.pend[1, 4]) == 0
    print("PASS gather")


def _scenarios():
    yield "align B16 seed1", dict(block_size=16), W.random_schedule(1, 40, 16)
    yield "align B16 seed2", dict(block_size=16), W.random_schedule(2, 40, 16, p_mixed=0.5)
    yield ("align B1152 boundary", dict(block_size=1152),
           W.random_schedule(3, 40, 1152, near_boundary=True))
    yield "none B16", dict(block_size=16, mode="none"), W.random_schedule(4, 25, 16)
    # short requests, nobody skipped: slots are reused at once and every new
    # request's first step is a spec row (a prompt tail run as a spec row,
    # prefill_runs_as_decode), right after a request that owed a commit
    yield ("slot reuse B16", dict(block_size=16),
           W.random_schedule(5, 40, 16, p_skip=0.0, p_mixed=0.0, max_len=12))
    files = replay.files("kda", "*-cm-*.pt")
    if files:
        rows = _captured_commits(files[: replay.limit(120) or None])
        yield f"replay {len(rows)} captured steps", dict(block_size=1152), \
            W.replay_schedule(rows, 1152, len(rows))


def _captured_commits(paths):
    """[(qlen, num_sampled, nc mod 1152)] per captured commit ('cm' payloads,
    in capture order)."""
    out = []
    for path in paths:
        p = torch.load(path, map_location="cpu", weights_only=False)
        t = p["tensors"]
        q = (t["query_start_loc"][1:] - t["query_start_loc"][:-1]).tolist()
        ns = t["num_sampled"].tolist()
        blen = t["boundary_lens"].tolist()
        bs = int(p["meta"]["block_size"])
        out.append([(q[i], ns[i], (bs - blen[i]) % bs) for i in range(len(q))])
    return out


def _sequences():
    for name, kw, events in _scenarios():
        a = W.run(W.World(fold=False, **kw), events)
        b = W.run(W.World(fold=True, **kw), events)
        problems = W.diff(a, b)
        assert not problems, (name, problems[:10])
        assert not W.checkpoints_unchanged(a) and not W.checkpoints_unchanged(b), name
        steps = sum(1 for e in events if e[0] == "step")
        print(f"  {name}: {steps} steps, {len(a.log)} verify reads, "
              f"{len(a.snaps)} checkpoints identical", flush=True)
    print("PASS sequences")


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class _ForceNoRecordRows:
    def __init__(self, k):
        self.k = k

    def __getitem__(self, grid):
        def run(*a, **kw):
            kw["HAS_RECORD_ROWS"] = False
            return self.k[grid](*a, **kw)
        return run


class _ConvOffByOne(W.ModelVerify):
    """A verify that reads the uncompacted window one row late."""

    def __call__(self, world, layer, meta, rows, log):
        fold = world.records[layer].fold
        if fold is not None:
            conv = world.conv[layer]
            saved = conv.clone()
            for i_n in range(len(rows)):
                s = int(meta.spec_state_indices_tensor[i_n, 0])
                if s > 0 and int(fold[0][i_n]) >= 1:
                    conv[s, :-1] = saved[s, 1:]
        return super().__call__(world, layer, meta, rows, log)


def _broken():
    """Each deliberately broken fold must change what a consumer reads."""
    from vllm_sm80.kernels.kda import recover as kr

    events = W.random_schedule(7, 30, 16, p_mixed=0.6)
    base = W.run(W.World(fold=False, block_size=16), events)
    orig_gather = kr.FoldState.gather
    orig_commit = kr.KDARecoverCommitContext.commit

    def gather_no_request_indices(self, idx, ri, n):
        return orig_gather(self, idx, None, n)

    def plan_without_blocks(self, *a, fold=None, **kw):
        # the fold plan without the block table defers every row, boundary
        # crossings included (their checkpoint is never written)
        if fold is not None:
            kw["block_table"] = None
        return orig_commit(self, *a, fold=fold, **kw)

    mutations = {
        "defers rows that cross a block boundary":
            lambda: _patched(kr.KDARecoverCommitContext, "commit", plan_without_blocks),
        "gather ignores the mixed step's row map":
            lambda: _patched(kr.FoldState, "gather", gather_no_request_indices),
        "a reused slot keeps the old request's debt":
            lambda: _patched(kr.FoldState, "reset_slot", lambda self, slot: None),
        "eager commit reads records by verify row":
            lambda: _patched(kr, "_kda_recover_commit_kernel",
                             _ForceNoRecordRows(kr._kda_recover_commit_kernel)),
    }
    for what, mutate in mutations.items():
        with mutate():
            b = W.run(W.World(fold=True, block_size=16), events)
        problems = W.diff(base, b)
        assert problems, f"not detected: {what}"
        print(f"  detected ({len(problems)} differences): {what}", flush=True)
    b = W.run(W.World(fold=True, block_size=16, verify=_ConvOffByOne()), events)
    assert W.diff(base, b), "not detected: conv window read one row late"
    print("  detected: conv window read one row late (verify contract)")
    b = W.run(W.World(fold=True, block_size=16), events)
    assert not W.diff(base, b)
    print("PASS broken folds detected")


def _streaming():
    """A streaming-input session: the worker removes the paused session and
    re-adds it under the same id with its state kept; the next chunk's
    prefill reads its state block and conv window. With the fold the last
    step's commit is usually still owed then: the re-add must write it
    (bitwise the commit without the fold). A preempted request removed in one
    step and re-added later must not be flushed: its freed blocks may already
    hold another request's state."""
    def scenario(world):
        rng = __import__("random").Random(11)
        for s in range(4):
            world.add(s, 5 + 3 * s, 10**6)
        for k in range(12):
            sched = [(s, rng.randint(2, W.SQL), 0) for s in range(4)]
            sched = [(s, q, rng.randint(1, q)) for s, q, _ in sched]
            world.step(sched)
            if k % 3 == 2:
                for s in range(4):
                    world.stream_readd(s)          # every session, owed or not
        # preemption: slot 3's request (owing its last step's commit) is
        # removed; a new request on another slot takes its freed blocks at once
        world.step([(s, 5, 4) for s in range(4)])
        gone = world.finish(3, front=True)
        world.step([(s, 3, 2) for s in range(3)])
        world.add(4, 9, 10**6, tag="reuse")
        world.step([(s, 4, 3) for s in (0, 1, 2, 4)])
        # the preempted request comes back on its old slot in a later step;
        # the next verifies read every state (the reuse request's included)
        world.add(3, 1, 10**6, rid=gone.rid, tag="resume")
        world.step([(s, 3, 2) for s in range(5)])
        world.step([(s, 2, 1) for s in range(5)])
        # Scheduler.reset_prefix_cache(reset_running_requests=True): a running
        # request (owing its last commit) is preempted and resumed in the same
        # step: removed in finish_requests, removed again and re-added by
        # add_requests, on its freed blocks, with computed tokens kept (a KV
        # connector hit). Its state is the resumed one, never the old commit.
        world.step([(s, 6, 5) for s in range(5)])
        again = world.finish(1, front=True)
        world.add(1, 7, 10**6, rid=again.rid, tag="reset")
        world.step([(s, 3, 2) for s in range(5)])
        world.step([(s, 2, 1) for s in range(5)])
        return world

    for kw in (dict(block_size=1152), dict(block_size=16, mode="none")):
        a = scenario(W.World(fold=False, **kw))
        b = scenario(W.World(fold=True, **kw))
        problems = W.diff(a, b)
        assert not problems, (kw, problems[:10])
        n = sum(1 for e in a.log if e[0] == "stream")
        print(f"  {kw}: {n} streaming re-add reads, {len(a.log)} reads identical")
    from vllm_sm80.kernels.kda import recover as kr

    base = scenario(W.World(fold=False, block_size=1152))
    orig_add = kr.FoldHooks.add_request

    def reset_only(self, slot, req_id, nc):        # the 8bf7b2baa9 behaviour
        fold = kr.FoldState.peek(self.device)
        fold.removed.clear()
        return orig_add(self, slot, req_id, nc)

    def flush_any_readd(self, slot, req_id, nc):   # flush across steps too
        seen = self.__dict__.setdefault("_seen", {})
        fold = kr.FoldState.peek(self.device)
        old = fold.removed.pop(req_id, None)
        if old is None:
            old = seen.get(req_id)
        if old is not None and nc > 0:
            fold.flush(old)
        fold.reset_slot(slot)
        fold.owner[slot] = req_id
        seen[req_id] = slot

    orig_remove = kr.FoldHooks.remove_request

    def remember_once(self, req_id):               # the 13f9e61da2 behaviour
        fold = kr.FoldState.peek(self.device)
        if req_id in fold.owner:
            return orig_remove(self, req_id)

    for what, attr, fn in (
            ("re-add resets the debt", "add_request", reset_only),
            ("a preempted request's re-add flushes", "add_request", flush_any_readd),
            ("a same-step preempt + re-add flushes (second removal kept)", "remove_request",
             remember_once)):
        with _patched(kr.FoldHooks, attr, fn):
            b = scenario(W.World(fold=True, block_size=1152))
        problems = W.diff(base, b)
        assert problems, f"not detected: {what}"
        print(f"  detected ({len(problems)} differences): {what}", flush=True)
    print("PASS streaming")


INTERPRETED = {"plan": _plan_vs_upstream, "gather": _gather, "sequences": _sequences,
               "broken": _broken, "streaming": _streaming}


def test_fold_plan_matches_upstream():
    _interpret("plan")


def test_gather():
    _interpret("gather")


def test_sequences_byte_identical():
    _interpret("sequences")


def test_broken_folds_fail():
    _interpret("broken")


def test_streaming_readd_keeps_owed_commit():
    _interpret("streaming")


# ------------------------------------------------------------------ host cost
class _Launch:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        def run(*a, **kw):
            self.log.append(self.name)
        return run


def _host_step(fold):
    """One align step through wrap + record_step + commit_step with every
    kernel replaced by a recorder: (launches, aten ops) of the engine code."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from vllm_sm80.kernels.kda import recover as kr
    from vllm_sm80.kernels.kda import commit_plan as cp
    from vllm.models.kimi_k3.nvidia.ops import recoverssm as up

    launches, ops = [], []

    class Ops(TorchDispatchMode):
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    world = W.World(fold=fold, block_size=16, nblocks=40)
    for s in range(4):
        world.add(s, 5 + s, 100)
    sched = [(3, 4, 2), (1, 8, 8), (0, 1, 1), (2, 6, 3)]
    names = ("_kda_fold_gather_kernel", "_kda_fold_plan_kernel", "_kda_recover_commit_kernel")
    stack = contextlib.ExitStack()
    with stack:
        for n in names:
            stack.enter_context(_patched(kr, n, _Launch(n, launches)))
        stack.enter_context(_patched(cp, "_prepare_commit_plan_kernel",
                                     _Launch("_prepare_commit_plan_kernel", launches)))
        stack.enter_context(_patched(up, "_compact_conv_state_kernel",
                                     _Launch("_compact_conv_state_kernel", launches)))
        stack.enter_context(_patched(cp, "_postprocess_recoverssm_align_kernel",
                                     _Launch("_postprocess_recoverssm_align_kernel", launches)))
        stack.enter_context(_patched(world, "verify", lambda *a: None))
        stack.enter_context(_patched(world, "_preprocess", lambda *a: None))
        orig_rec, orig_commit = world.state.record_step, world.state.commit_step
        if world.hooks is not None:
            orig_prep = world.hooks.prepare_step

            def prep(*a, **kw):
                with Ops():
                    return orig_prep(*a, **kw)

            world.hooks.prepare_step = prep

        def rec(*a, **kw):
            with Ops():
                return orig_rec(*a, **kw)

        def commit(*a, **kw):
            with Ops():
                return orig_commit(*a, **kw)

        world.state.record_step, world.state.commit_step = rec, commit
        world.step(sched)
    return launches, ops


def test_host_cost_of_a_step():
    """Per step the fold adds one launch (the gather) and replaces the plan
    one for one; it adds no host sync, no allocation and no copy: the only
    extra aten ops are views of the persistent fold buffers."""
    off_l, off_ops = _host_step(False)
    on_l, on_ops = _host_step(True)
    G = 2
    assert off_l.count("_prepare_commit_plan_kernel") == G and "_kda_fold_plan_kernel" not in off_l
    assert on_l.count("_kda_fold_plan_kernel") == G and "_prepare_commit_plan_kernel" not in on_l
    assert on_l.count("_kda_fold_gather_kernel") == 1
    assert len(on_l) == len(off_l) + 1, (off_l, on_l)
    from collections import Counter

    extra = Counter(on_ops) - Counter(off_ops)
    allowed = {"select", "slice", "alias", "view", "as_strided"}
    assert set(extra) <= allowed, extra
    for bad in ("_local_scalar_dense", "item", "empty", "zeros", "full", "copy_",
                "_to_copy", "clone"):
        assert Counter(on_ops)[bad] <= Counter(off_ops)[bad], bad
    print(f"  launches per step at 2 groups: off {len(off_l)}, fold {len(on_l)}; "
          f"extra aten ops: {dict(extra)}")


# ------------------------------------------------------------------ graph buffers
def test_cuda_graph_buffers():
    """The verify reads fold buffers that never move (a captured graph replays
    the same addresses): views at offset 0 of FoldState's persistent tensors,
    sized for every row the builder can hand the verify (the largest capture);
    the warmup compiles the fold variants for every (nseq, T) plan."""
    from vllm_sm80.kernels.kda import recover as kr

    world = W.World(fold=True, block_size=16, nblocks=40)
    f = kr.FoldState.peek("cpu")
    assert f.rows == W.MAX_SEQS * W.SQL >= world.builders[0].ones.shape[0]
    ptrs = set()
    for s in range(3):
        world.add(s, 3 + s, 100)
    with _patched(world, "verify", lambda w, layer, meta, rows, log: ptrs.add(
            (layer, w.records[layer].fold[0].data_ptr(), w.records[layer].fold[1].data_ptr(),
             w.records[layer].fold[0].shape[0]))), \
            _patched(world, "_preprocess", lambda *a: None), \
            _patched(world.state, "commit_step", lambda *a, **k: None), \
            _patched(kr, "_kda_fold_gather_kernel", _Launch("g", [])):
        world.step([(0, 3, 1), (1, 2, 2)])
        world.step([(2, 8, 4), (0, 1, 1), (1, 5, 5)], mixed_pos=1)
    for g in range(world.G):
        got = {p[1:] for p in ptrs if p[0] == world.layer_of(g, 0)}
        assert got == {(f.lens[g].data_ptr(), f.record_rows.data_ptr(), f.rows)}, got
    from vllm_sm80.models.glm5next.ops import kda_decode as k2

    calls = []
    layer = types.SimpleNamespace(_conv_state_dim_first=True, _kda_v2=True, local_num_heads=16,
                                  head_dim=128, num_spec=7, _kda_step_tile=True, _kda_wide=False,
                                  _kda_recover=True)
    for on in (True, False):
        calls = []
        undo = W.env(on)
        try:
            worker = types.SimpleNamespace(vllm_config=W.config())
            with _patched(k2, "find_module", lambda m, a: layer), \
                    _patched(k2, "warmup", lambda **kw: calls.append(kw)):
                k2._warmup_kda_v2(worker, None, "cpu", [1, 8, 64])
        finally:
            undo()
        assert len(calls) == 1 and calls[0]["fold"] is on and calls[0]["recover"] is True
        assert set(calls[0]["plans"]) == {(n, t) for n in range(1, 9) for t in range(1, 9)}


def test_off_path_untouched():
    """Switch off: no hooks (the model state makes no call into the fold),
    no FoldState, layer records without fold buffers, the upstream plan,
    verify rows as record rows (the recover control-path test in
    test_ampere_kda_recover.py runs this path end to end). The shared model
    state module imports nothing of vllm_sm80 at import time."""
    import ast
    import inspect
    import subprocess

    from vllm_sm80.kernels.kda import recover as kr
    from vllm.v1.worker.gpu.model_states import mamba_hybrid

    world = W.World(fold=False, block_size=16, nblocks=40)
    assert world.hooks is None
    assert kr.FoldState.peek("cpu") is None and not kr.FoldState._states
    assert all(b.fold is None and b.fold_group == -1 for b in world.builders)
    assert all(r.fold is None for r in world.records)
    seen = []
    world.add(0, 4, 50)
    with _patched(world, "verify", lambda w, layer, meta, rows, log: seen.append(
            w.records[layer].fold)), \
            _patched(world, "_preprocess", lambda *a: None), \
            _patched(world.state, "commit_step", lambda *a, **k: None):
        world.step([(0, 3, 1)])
    assert seen and all(s is None for s in seen)
    tree = ast.parse(inspect.getsource(mamba_hybrid))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [getattr(n, "module", None) or n.names[0].name for n in top]
    assert not [m for m in names if m and m.startswith("vllm_sm80")], names
    code = ("import sys, vllm.v1.worker.gpu.model_states.mamba_hybrid; "
            "print('vllm_sm80.kernels.kda.recover' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "False", r.stdout[-500:]


def _real_config_build(monkeypatch, fold, cache_mode, query_lens, prefill):
    """GLM-5.3-Flash TP4 checkpoint config (byte-exact, DFlash), the real GDN
    builders of the GLM KDA layers for 4 KDA groups and build_attn_metadata
    (the test_glm_metadata_reuse.py setup), then the fold's step hook."""
    from tests.models.test_native_checkpoint_config_meta import _vllm_config
    from tests.v1.attention.test_glm_metadata_reuse import _glm_kda_backend
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config.compilation import CUDAGraphMode
    from vllm_sm80.kernels.kda import recover as kr
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
    from vllm.v1.worker.utils import AttentionGroup

    kr.FoldState._states.clear()
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_AMPERE_KDA_RECOVER_OFF", "0")
    # FoldState allocates with zeros/full (the fixture redirects empty/ones)
    for fn in ("zeros", "full"):
        def allocate(*args, _original=getattr(torch, fn), **kwargs):
            if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, fn, allocate)
    monkeypatch.setenv("VLLM_AMPERE_KDA_FOLD", str(int(fold)))
    config = _vllm_config("glm53_flash", 4, 1, with_dflash=True)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.compilation_config.max_cudagraph_capture_size = 64
    config.cache_config.mamba_cache_mode = cache_mode
    text = config.model_config.hf_text_config
    num_spec = config.num_speculative_tokens
    spec = MambaSpec(
        block_size=16,
        shapes=MambaStateShapeCalculator.kda_state_shape(
            tp_world_size=4, num_heads=text.linear_num_heads,
            head_dim=text.linear_head_dim, num_spec=num_spec),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=cache_mode, num_speculative_blocks=0)
    fc = config.compilation_config.static_forward_context
    pool = torch.zeros(8, 3, 1, kr.WS_T, 1, 1)
    names = [[f"layer.{g}.{i}" for i in range(2)] for g in range(4)]
    records = {}
    for g, ns in enumerate(names):
        for i, n in enumerate(ns):
            records[n] = kr.layer_records(pool, 2 * g + i)
            fc[n] = types.SimpleNamespace(_kda_recover_records=records[n])
    backend = _glm_kda_backend()
    builder_cls = backend.get_builder_cls()
    builders = [builder_cls(spec, ns, config, torch.device("cuda")) for ns in names]
    assert all(b._kda_recover is not None for b in builders)
    for b in builders:
        # a mixed step's row map is copied to the builder's device: CPU here
        b.device = torch.device("cpu")
    batch = BatchSpec(seq_lens=[64] * len(query_lens), query_lens=query_lens)
    common = create_common_attn_metadata(batch, 16, torch.device("cpu"))
    tables = [torch.arange(batch.batch_size * 16, dtype=torch.int32).reshape(
        batch.batch_size, 16) + 1000 * g for g in range(4)]
    drafts = torch.tensor([-1 if i in prefill else q - 1 for i, q in enumerate(query_lens)],
                          dtype=torch.int32)
    model_meta = MambaHybridAttnMetadata(
        is_prefilling=torch.tensor([i in prefill for i in range(len(query_lens))]),
        num_accepted_tokens=torch.ones(batch.batch_size, dtype=torch.int32),
        num_decode_draft_tokens_cpu=drafts)
    groups = [AttentionGroup(backend, b.layer_names, spec, g, metadata_builders=[b])
              for g, b in enumerate(builders)]
    kv_config = KVCacheConfig(10000, [], [KVCacheGroupSpec(g.layer_names, spec)
                                          for g in groups])
    attn = build_attn_metadata(
        attn_groups=[[g] for g in groups], num_reqs=batch.batch_size,
        num_tokens=common.num_actual_tokens, query_start_loc_gpu=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu, max_query_len=common.max_query_len,
        seq_lens=common.seq_lens, max_seq_len=common.max_seq_len, block_tables=tables,
        slot_mappings=[common.slot_mapping] * 4, kv_cache_config=kv_config,
        seq_lens_cpu_upper_bound=common.seq_lens_cpu_upper_bound,
        model_specific_attn_metadata=model_meta, for_cudagraph_capture=False)
    gathered = []
    state = kr.FoldState._states.get(("cuda", 0))
    idx = torch.arange(batch.batch_size, dtype=torch.int32) + 2
    hooks = kr.kda_fold_hooks(config, torch.device("cuda"))
    assert (hooks is not None) == fold
    if hooks is not None:
        with _patched(kr, "_kda_fold_gather_kernel", _Launch("gather", gathered)), \
                _patched(kr.FoldState, "peek", classmethod(lambda cls, d: state)):
            hooks.prepare_step(attn, idx, False)
            hooks.prepare_step(attn, idx, True)            # capture: nothing
    return attn, builders, records, state, gathered, num_spec


_META_FIELDS = (
    "num_decodes", "num_prefills", "num_spec_decodes", "num_spec_decode_tokens",
    "spec_state_indices_tensor", "non_spec_state_indices_tensor", "spec_query_start_loc",
    "spec_sequence_masks", "spec_token_indx", "non_spec_token_indx", "num_accepted_tokens",
    "spec_query_start_loc")


if pytest is not None:
    from tests.models.test_native_checkpoint_config_meta import meta_platform  # noqa: F401
    from tests.v1.attention.test_glm_metadata_reuse import (  # noqa: F401
        cpu_cuda_allocations,
    )

    @pytest.mark.parametrize("cache_mode", ["none", "align"])
    @pytest.mark.parametrize("query_lens,prefill", [
        ([8, 8, 8, 8], ()),                 # pure verify, 4 requests
        ([4, 1, 8, 2, 6, 3, 8, 5], ()),     # 8 requests, uneven drafts
        # (a prefill chunk among spec rows needs the GDN chunk metadata, which
        # allocates on CUDA in the base builder: the World sequences cover the
        # fold's mixed-step row map instead)
        ([3, 8], ()),                       # a prompt tail run as a spec row
    ])
    def test_real_config_meta_build(monkeypatch, cpu_cuda_allocations, cache_mode,
                                    query_lens, prefill):
        """Real GLM-5.3-Flash TP4 config and builders: the step metadata is
        identical with the fold on and off (the fold adds nothing to it); on,
        one FoldState with 4 groups and rows for every captured verify row,
        every KDA layer's records carry its group's buffers, and the step hook
        runs one gather with the step's row map (none at capture)."""
        off = _real_config_build(monkeypatch, False, cache_mode, query_lens, prefill)
        on = _real_config_build(monkeypatch, True, cache_mode, query_lens, prefill)
        attn_off, attn_on = off[0], on[0]
        assert attn_off.keys() == attn_on.keys()
        for name in attn_off:
            a, b = attn_off[name], attn_on[name]
            assert type(a) is type(b)
            for f in _META_FIELDS:
                x, y = getattr(a, f, None), getattr(b, f, None)
                if isinstance(x, torch.Tensor):
                    assert torch.equal(x, y), (name, f)
                else:
                    assert x == y, (name, f)
            ca, cb = getattr(a, "recover_commit", None), getattr(b, "recover_commit", None)
            assert (ca is None) == (cb is None)
            if ca is not None:
                assert torch.equal(ca.state_indices, cb.state_indices)
                assert (ca.request_indices is None) == (cb.request_indices is None)
        _, builders_off, records_off, state_off, gathered_off, _ = off
        _, builders, records, state, gathered, num_spec = on
        assert state_off is None and not gathered_off
        assert all(r.fold is None for r in records_off.values())
        assert state is not None and len(state.groups) == 4
        assert state.rows == 8 * (1 + num_spec)
        assert all(b._kda_recover.ones.shape[0] <= state.rows for b in builders)
        for b in builders:
            g = b._kda_recover.fold_group
            for n in b.layer_names:
                lens, rows = records[n].fold
                assert lens.data_ptr() == state.lens[g].data_ptr()
                assert rows.data_ptr() == state.record_rows.data_ptr()
        assert gathered == ["gather"]
        assert not torch.cuda.is_initialized()


# ------------------------------------------------------------------ GPU
def _gpu_inputs(nseq, T, seed, nslot):
    from tests.kernels.test_ampere_kda_recover import _inputs

    return _inputs(nseq, T, seed, nslot)


def _v2(p, conv, rec, ssm, acc, T, records, fold=None, skip_norm=False):
    from vllm_sm80.models.glm5next.ops.kda_decode import kda_decode_v2

    nseq = ssm.shape[0]
    out = torch.empty(1, nseq * T, 16, 128, device="cuda", dtype=torch.bfloat16)
    return kda_decode_v2(p.qkv, p.beta, p.fa, p.ga, p.wf, p.wg, conv.transpose(-1, -2), p.cw,
                         None, p.nw, rec, ssm[:, 0], ssm, acc, p.qsl, 8, p.al, p.gb, out=out,
                         records=records, skip_norm=skip_norm, fold=fold)


def gpu_test_fold_verify_bitwise():
    """Real kernel, one card: step k verify, then step k + 1.
    today: RECOVER verify k (records at verify rows) -> commit(n) -> RECOVER
    verify k + 1; fold: FOLD verify k (records at permuted slots, nothing
    owed) -> FOLD verify k + 1 owing n. Equal bitwise: step k + 1 outputs
    (normed and SKIP_NORM), its records, the state slot after k + 1 (fold:
    the committed state the prologue stored == today's commit) and the conv
    window; for T 1..8, nseq 1/2/4/7/8, n 1..T."""
    from vllm_sm80.kernels.kda import recover as kr
    from vllm_sm80.models.glm5next.ops import kda_decode as k2
    from tests.kernels.test_ampere_kda_recover import _context

    nfail = checks = 0
    for T in range(1, 9):
        for nseq in (1, 2, 4, 7, 8):
            k2.warmup(plans=((nseq, T),), recover=True, width=8)
            k2.warmup(plans=((nseq, T),), recover=True, width=8, fold=True)
            nslot = 2 + nseq * 8
            p = _gpu_inputs(nseq, T, 7000 + 10 * T + nseq, nslot)
            p2 = _gpu_inputs(nseq, T, 9000 + 10 * T + nseq, nslot)
            p.qsl = p2.qsl = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * T
            ssm = torch.tensor([[1 + 8 * s] for s in range(nseq)], device="cuda",
                               dtype=torch.int32)
            ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
            slots = torch.tensor([(3 * s + 5) % 8 for s in range(nseq)], device="cuda",
                                 dtype=torch.int32)
            for n in range(1, T + 1):
                for skip in (False, True):
                    pool_a = torch.full((1, 3, 8, kr.WS_T, 16, 128), float("nan"), device="cuda")
                    pool_b = pool_a.clone()
                    ra, ca = p.rec.clone(), p.conv.clone()
                    rb, cb = p.rec.clone(), p.conv.clone()
                    _v2(p, ca, ra, ssm, ones, T, kr.layer_records(pool_a, 0))
                    ctx = _context(ca, ra, pool_a)
                    ctx.commit(torch.full((nseq,), n, device="cuda", dtype=torch.int32),
                               ssm[:, 0].contiguous(), p.qsl)
                    ya = _v2(p2, ca, ra, ssm, ones, T, kr.layer_records(pool_a, 0),
                             skip_norm=skip).clone()
                    zero = torch.zeros(nseq, device="cuda", dtype=torch.int32)
                    _v2(p, cb, rb, ssm, ones, T, kr.layer_records(pool_b, 0), fold=(zero, slots))
                    owed = torch.full((nseq,), n, device="cuda", dtype=torch.int32)
                    yb = _v2(p2, cb, rb, ssm, ones, T, kr.layer_records(pool_b, 0),
                             fold=(owed, slots), skip_norm=skip).clone()
                    torch.cuda.synchronize()
                    ok = torch.equal(ya, yb) and torch.equal(ra.view(torch.int32),
                                                             rb.view(torch.int32))
                    for s in range(nseq):
                        sl = int(slots[s])
                        ok &= torch.equal(pool_a[0, :, s, :T], pool_b[0, :, sl, :T])
                        ok &= torch.equal(ca[1 + 8 * s, :2 + T], cb[1 + 8 * s, :2 + T])
                    checks += 1
                    if not ok:
                        nfail += 1
                        print(f"  FAIL T={T} nseq={nseq} n={n} skip={skip}")
    print(f"  {checks - nfail}/{checks} bitwise")
    assert nfail == 0


class KernelVerify:
    """kda_decode_v2 itself as the World's verify (GPU; real conv width)."""

    def __init__(self, world):
        from tests.kernels.test_ampere_kda_recover import _inputs

        self.p = [_inputs(1, 1, 31 + l, 2) for l in range(world.L)]

    def __call__(self, world, layer, meta, rows, log):
        from vllm_sm80.models.glm5next.ops.kda_decode import kda_decode_v2

        p = self.p[layer]
        T_all = [T for _, _, T in rows]
        M = sum(T_all)
        g = torch.Generator(device="cuda").manual_seed(
            W.seed(layer, tuple(r[:2] for r in rows)))
        buf = torch.empty(M, 3 * 2048 + 16 + 256, device="cuda").normal_(generator=g)
        buf[:, 6144 + 16:] *= 1.0 / (0.025 * 128 ** 0.5)
        proj = buf.to(torch.bfloat16)
        qkv, beta = proj[:, :6144], proj[:, 6144:6160].unsqueeze(0)
        fa, ga = proj[:, 6160:6288], proj[:, 6288:]
        n = len(rows)
        ssm = meta.spec_state_indices_tensor
        out = torch.empty(1, M, 16, 128, device="cuda", dtype=torch.bfloat16)
        rec, conv = world.rec[layer], world.conv[layer]
        records = world.records[layer]          # the layer's records: .fold if on
        fold = records.fold
        kda_decode_v2(qkv, beta, fa, ga, p.wf, p.wg, conv.transpose(-1, -2), p.cw, None, p.nw,
                      rec, ssm[:, 0][:n], ssm, meta.num_accepted_tokens,
                      meta.spec_query_start_loc[:n + 1], max(T_all), p.al, p.gb, out=out,
                      records=records)
        torch.cuda.synchronize()
        for i_n, (req, pos0, T) in enumerate(rows):
            s = int(ssm[i_n, 0])
            r_row = max(int(fold[1][i_n]), 0) if fold is not None else i_n
            log.append(("out", layer, req, pos0,
                        out[0, int(meta.spec_query_start_loc[i_n]):
                            int(meta.spec_query_start_loc[i_n + 1])].clone(),
                        world.pool[layer][:, r_row, :T].clone(),
                        conv[s, :2 + T].clone(),
                        self.fingerprint(rec[s])))

    _w = None

    def fingerprint(self, x):
        """The state slot after the verify (fold: the committed state its
        prologue stored; today: the committed state) as a position-weighted
        int64 sum of its bits (a 1 MiB copy per entry would not fit)."""
        if self._w is None or self._w.numel() != x.numel():
            g = torch.Generator(device="cuda").manual_seed(7)
            self._w = torch.randint(1, 2**31 - 1, (x.numel(),), device="cuda", generator=g)
        return (x.reshape(-1).view(torch.int32).to(torch.int64) * self._w).sum().reshape(1)


def gpu_test_fold_sequences_bitwise():
    """The CPU sequence check with the real verify kernel and the real
    compiled commit / plan / gather / pre-copy kernels on one card: every
    output, record, conv window and boundary checkpoint bitwise equal with and
    without the fold (block size 16 and the 1,152-token boundary)."""
    for name, kw, events in (
            ("align B16", dict(block_size=16), W.random_schedule(11, 40, 16, p_mixed=0.4)),
            ("align B1152", dict(block_size=1152),
             W.random_schedule(12, 40, 1152, near_boundary=True)),
            ("none", dict(block_size=16, mode="none"), W.random_schedule(13, 20, 16))):
        worlds = []
        for fold in (False, True):
            w = W.World(fold=fold, device="cuda", conv_dim=6144, conv_dtype=torch.bfloat16,
                        nblocks=200, **kw)
            w.verify = KernelVerify(w)
            worlds.append(W.run(w, events))
        problems = W.diff(*worlds)
        assert not problems, (name, problems[:10])
        print(f"  {name}: {len(worlds[0].log)} verify outputs, "
              f"{len(worlds[0].snaps)} checkpoints bitwise")


def gpu_test_fold_graph_zero_growth():
    """Gather + FOLD verify + fold plan + eager commit capture into one CUDA
    graph with no allocation beyond torch's own, replay allocates nothing and
    is bitwise the eager run."""
    from vllm_sm80.kernels.kda import recover as kr
    from vllm_sm80.models.glm5next.ops import kda_decode as k2
    from tests.kernels.test_ampere_kda_recover import _context

    T, nseq = 8, 4
    k2.warmup(plans=((nseq, 8),), recover=True, width=8, fold=True)
    p = _gpu_inputs(nseq, T, 21, 2 + nseq * 8)
    ssm = torch.tensor([[1 + 8 * s] for s in range(nseq)], device="cuda", dtype=torch.int32)
    ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
    kr.FoldState._states.clear()
    f = kr.FoldState("cuda", 8, 64)
    f.group(("l0",))
    f.pend[0, :nseq] = torch.tensor([1, 4, 8, 6], device="cuda", dtype=torch.int32)
    idx = torch.tensor([6, 1, 3, 0], device="cuda", dtype=torch.int32)
    pool = torch.zeros(1, 3, 8, kr.WS_T, 16, 128, device="cuda")
    ctx = _context(p.conv, p.rec, pool)
    nsamp = torch.tensor([2, 8, 1, 5], device="cuda", dtype=torch.int32)
    # finite owed records (the previous step's corrections, unit keys, decay
    # gates in (0.5, 1)), so the replay under capture does real arithmetic
    g = torch.Generator(device="cuda").manual_seed(5)
    pool[0, 0].normal_(0.0, 0.05, generator=g)
    keys = torch.randn(pool[0, 1].shape, device="cuda", generator=g)
    pool[0, 1].copy_(keys / keys.norm(dim=-1, keepdim=True))
    pool[0, 2].uniform_(0.5, 1.0, generator=g)
    # every input the step reads and writes: the state, the conv window, the
    # owed lengths and the record pool (the FOLD verify replays owed tokens
    # from the records and stores this step's records over them)
    rec0, conv0, pend0, pool0 = p.rec.clone(), p.conv.clone(), f.pend.clone(), pool.clone()

    recs = kr.layer_records(pool, 0)
    recs.fold = (f.lens[0], f.record_rows)

    def step():
        f.gather(idx, None, nseq)
        _v2(p, p.conv, p.rec, ssm, ones, T, recs)
        ctx.commit(nsamp, ssm[:, 0].contiguous(), p.qsl, fold=(f, 0))

    step()
    torch.cuda.synchronize()
    want = (p.rec.clone(), p.conv.clone(), f.pend.clone(), pool.clone())
    for t, v in ((p.rec, rec0), (p.conv, conv0), (f.pend, pend0), (pool, pool0)):
        t.copy_(v)
    scratch = torch.zeros(16, device="cuda")
    torch.cuda.synchronize()
    c0 = torch.cuda.memory_allocated()
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        scratch.add_(1.0)
    cap = torch.cuda.memory_allocated() - c0
    del g0
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - m0 - cap <= 0
    m1 = torch.cuda.memory_allocated()
    for _ in range(2):
        for t, v in ((p.rec, rec0), (p.conv, conv0), (f.pend, pend0), (pool, pool0)):
            t.copy_(v)
        g.replay()
        torch.cuda.synchronize()
        got = (p.rec, p.conv, f.pend, pool)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert torch.cuda.memory_allocated() <= m1


CPU_TESTS = (test_switch_and_banner, test_fold_plan_matches_upstream, test_gather,
             test_sequences_byte_identical, test_broken_folds_fail,
             test_streaming_readd_keeps_owed_commit, test_host_cost_of_a_step,
             test_cuda_graph_buffers, test_off_path_untouched)
GPU_TESTS = (gpu_test_fold_verify_bitwise, gpu_test_fold_sequences_bitwise,
             gpu_test_fold_graph_zero_growth)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an sm_80 GPU")
    test_gpu_fold_verify_bitwise = _needs_gpu(gpu_test_fold_verify_bitwise)
    test_gpu_fold_sequences_bitwise = _needs_gpu(gpu_test_fold_sequences_bitwise)
    test_gpu_fold_graph_zero_growth = _needs_gpu(gpu_test_fold_graph_zero_growth)


def _main():
    import traceback

    if len(sys.argv) == 3 and sys.argv[1] == "interpret":
        assert os.environ.get("TRITON_INTERPRET") == "1"
        INTERPRETED[sys.argv[2]]()
        return
    tests = list(CPU_TESTS)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available():
        tests += list(GPU_TESTS)
    only = sys.argv[1:]
    if only:
        tests = [t for t in tests if t.__name__ in only]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
