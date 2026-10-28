# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Gloo workers must agree after faults, or terminate before another RPC."""
import json
import multiprocessing
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def _worker(rank, rendezvous, folder, fault):
    import torch.distributed as dist

    from vllm.distributed import parallel_state
    from vllm_sm80 import dev_ab as ab
    from vllm.v1.executor.multiproc_executor import WorkerProc

    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=15))
    parallel_state._WORLD = NS(cpu_group=dist.group.WORLD)
    runner = NS(actual_arm=False, req_states=NS(req_id_to_index={}))
    sync_calls, apply_failed = 0, False

    def synchronize():
        nonlocal sync_calls
        sync_calls += 1
        if rank != 1:
            return
        if (fault == "before" and sync_calls == 1
                or fault == "after" and sync_calls == 2
                or fault == "rollback" and sync_calls >= 2):
            raise RuntimeError("rank-local synchronization fault")

    def apply(_runner, arm):
        nonlocal apply_failed
        runner.actual_arm = arm
        if rank == 1 and arm and fault == "adapter" and not apply_failed:
            apply_failed = True
            raise RuntimeError("rank-local partial adapter fault")

    class Manager:
        def __init__(self, failing=False):
            self.value, self.failing = True, failing
        @property
        def _graphs_captured(self):
            return self.value
        @_graphs_captured.setter
        def _graphs_captured(self, value):
            self.value = value
            if self.failing and not value:
                self.failing = False
                raise RuntimeError("rank-local replay flag fault")

    managers = [Manager(), Manager(rank == 1 and fault == "replay")]
    plans = [NS(manager=m, check_graphs=lambda arms: None) for m in managers]
    switch = ab.Switch("fake", "VLLM_FAKE_SWITCH", True, True, lambda r: None,
                       lambda r: runner.actual_arm, apply)
    ab._STATE = ab._ArmState(switch=switch, boot=False, arm=False, captured=(False, True),
                             saved={"capture_contract": (None, plans)})
    ab._synchronize = synchronize
    ab._graphs_on = lambda r: True
    ab._check_policy = lambda *a: None
    ab._check_complete = lambda *a: None
    original_vote = ab._all_ranks_agree
    votes = 0
    def vote(ok):
        nonlocal votes
        votes += 1
        if fault == "collective" and rank == 1 and votes == 2:
            raise RuntimeError("rank-local commit acknowledgement fault")
        return original_vote(ok)
    ab._all_ranks_agree = vote

    class Worker(ab.DevABWorkerExtension):
        pass
    worker = Worker()
    worker.rank, worker.model_runner = rank, runner
    replies = []
    proc = NS(worker=worker, rank=rank, handle_output=replies.append)
    # This exact handler catches ordinary exceptions and leaves the worker alive.
    WorkerProc._execute_worker_rpc(proc, ("set_ab_arm", ("on", 1, False), {}, None))
    reply = replies[0]
    Path(folder, f"rank{rank}.json").write_text(json.dumps({
        "reply": reply if isinstance(reply, dict) else {"error": type(reply).__name__},
        "arm": ab._STATE.arm, "actual_arm": runner.actual_arm,
        "generation": ab._STATE.generation, "replay": ab._STATE.replay,
        "graph_flags": [m._graphs_captured for m in managers],
    }))
    dist.destroy_process_group()


def _run(tmp_path, fault):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(rank, str(tmp_path / "gloo"), str(tmp_path), fault))
             for rank in range(2)]
    try:
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(60)
        assert all(not p.is_alive() for p in procs), "RPC transaction hung"
        return [p.exitcode for p in procs]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()  # only this test's own CPU children
            proc.join()


@pytest.mark.parametrize("fault", ["before", "after", "adapter", "replay"])
def test_rank_local_fault_rolls_back_every_worker(tmp_path, fault):
    assert _run(tmp_path, fault) == [0, 0]
    states = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    for state in states:
        assert state["reply"].get("ok") is False, state
        assert "rolled back" in state["reply"]["reason"], state
        assert state["arm"] is state["actual_arm"] is False
        assert state["generation"] == 0 and state["replay"] is True
        assert state["graph_flags"] == [True, True]


@pytest.mark.parametrize("fault", ["rollback", "collective"])
def test_unrecoverable_consistency_fault_stops_every_worker(tmp_path, fault):
    assert _run(tmp_path, fault) == [1, 1]
    assert not list(tmp_path.glob("rank*.json")), "worker returned to RPC handling after consistency failure"


def test_collective_commit_acknowledges_every_worker(tmp_path):
    assert _run(tmp_path, "success") == [0, 0]
    for rank in range(2):
        state = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert state["reply"]["ok"] is True
        assert state["arm"] is state["actual_arm"] is True
        assert state["generation"] == 1 and state["replay"] is False
        assert state["graph_flags"] == [False, False]
