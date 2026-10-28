# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission barriers, actual dispatch traces and fail-closed row pins."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode as Mode
from vllm_sm80 import dev_ab as ab
from vllm_sm80 import dev_ab_batch as batch
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor as Desc
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "exl3_dense_splitk")
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    monkeypatch.setattr(ab, "_STATE", ab._ArmState(arm=False, boot=False, captured=(False, True)))
    monkeypatch.setattr(ab, "_transaction_vote", lambda ok: ok)
    monkeypatch.setattr(ab, "_synchronize", lambda: None)


def manifest(name="batch"):
    return dict(id=name, rows=[dict(tokens=2, sha256=batch.token_digest([10, 11])) for _ in range(2)])


def request(index, declared=None):
    import json
    declared = declared or manifest()
    return NS(request_id=f"cmpl-{declared['id']}-{index}-1234abcd", prompt_token_ids=[10, 11],
              sampling_params=NS(extra_args={"dev_ab_batch": json.dumps(declared)}))


class SchedulerBase:
    def __init__(self):
        self.requests, self.running, self.finished_req_ids = {}, [], set()
        self.waiting = FCFSRequestQueue()
        self.policy = SchedulingPolicy.FCFS
        self.max_num_running_reqs = self.max_num_active_reqs = 8
        self.max_num_scheduled_tokens = 16
        self.scheduler_config = NS(max_num_batched_tokens=16)
        self.vllm_config = NS(speculative_config=None)

    def add_request(self, req):
        self.requests[req.request_id] = req
        self.waiting.add_request(req)

    def get_request_counts(self):
        return len(self.running), len(self.waiting)

    def has_finished_requests(self):
        return bool(self.finished_req_ids)

    def has_requests(self):
        return bool(self.running or self.waiting or self.finished_req_ids)

    def schedule(self):
        while self.waiting:
            self.running.append(self.waiting.pop_request())
        return [r.request_id for r in self.running]

    def finish_requests(self, ids=None, status=None):
        finished = []
        for rid in list(self.requests) if ids is None else ids:
            req = self.requests.pop(rid)
            if req in self.waiting:
                self.waiting.remove_request(req)
            self.finished_req_ids.add(rid)
            finished.append(req)
        self.running = [r for r in self.running if r.request_id in self.requests]
        return finished


class Ordered(batch.OrderedAdmission, SchedulerBase):
    pass


def test_out_of_order_arrival_cannot_execute_a_partial_batch():
    scheduler = Ordered()
    scheduler.add_request(request(1))
    assert scheduler.get_request_counts() == (0, 1)
    assert not scheduler.has_requests() and not scheduler.waiting
    scheduler.add_request(request(0))
    assert scheduler.has_requests()
    expected = [request(i).request_id for i in range(2)]
    assert scheduler.schedule() == expected
    scheduler.running.reverse()
    assert scheduler.schedule() == expected


@pytest.mark.parametrize("name", ["VLLM_DEV_AB_SWITCH", "VLLM_SERVER_DEV_MODE"])
def test_barrier_refuses_without_both_development_gates(monkeypatch, name):
    monkeypatch.delenv(name)
    with pytest.raises(ValueError, match="require"):
        Ordered()


def test_unset_capability_never_inspects_runner(monkeypatch):
    monkeypatch.delenv("VLLM_DEV_AB_SWITCH")
    class Trap:
        def __getattr__(self, name):
            raise AssertionError(name)
    assert batch.capability(Trap()) is None


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_configuration_preserves_scheduler_mode(async_scheduling):
    config = NS(scheduler_config=NS(async_scheduling=async_scheduling, scheduler_cls=None),
                parallel_config=NS(worker_extension_cls=""))
    ab.configure(config)
    assert config.scheduler_config.scheduler_cls == batch.SCHEDULERS[async_scheduling]
    ab.configure(config)
    assert config.scheduler_config.async_scheduling is async_scheduling


def test_other_scheduler_is_not_silently_replaced():
    config = NS(scheduler_config=NS(async_scheduling=True, scheduler_cls="other.Scheduler"))
    with pytest.raises(ValueError, match="another scheduler"):
        batch.configure(config)


def test_cancelled_partial_batch_remains_cancellable_and_releases_barrier():
    scheduler = Ordered()
    scheduler.add_request(request(0))
    assert len(scheduler.finish_requests()) == 1
    assert scheduler.get_request_counts() == (0, 0)
    assert not scheduler._ordered_held and scheduler._ordered_manifest is None
    ordinary = request(0)
    ordinary.sampling_params.extra_args.clear()
    scheduler.add_request(ordinary)
    assert list(scheduler.waiting) == [ordinary]


@pytest.mark.parametrize("mutation", ["prompt", "duplicate", "foreign", "oversize"])
def test_barrier_rejects_mixed_or_unpinned_admission(mutation):
    scheduler = Ordered()
    scheduler.add_request(request(0))
    req = request(1)
    if mutation == "prompt":
        req.prompt_token_ids = [12, 13]
    elif mutation == "duplicate":
        req = request(0)
    elif mutation == "foreign":
        req = request(1, manifest("foreign"))
    else:
        scheduler.max_num_active_reqs = 1
    with pytest.raises(ValueError):
        scheduler.add_request(req)
    assert not scheduler.waiting


class Manager:
    dispatch = CudaGraphManager.dispatch
    def __init__(self, desc):
        self._candidates = {(desc.num_tokens, 0): [desc]}
        self._graphs_captured = True
    def _resolve_effective_loras(self, value):
        return value


class Runner:
    def __init__(self):
        self.cudagraph_manager = Manager(Desc(Mode.FULL, 8, 2, 4))
        self.speculator = NS(_cg_managers={5: Manager(Desc(Mode.FULL, 10, 2, 5))})
        self.vllm_config = NS(scheduler_config=NS(scheduler_cls=batch.SCHEDULERS[0]))
        self.req_states = NS(req_id_to_index={})
        self.forward_calls = 0
        self.corrupt_input = False

    def gather_batch_req_state(self, output, dummy_run):
        ids = list(output.num_scheduled_tokens)
        return NS(req_ids=ids, num_scheduled_tokens=np.array([4, 4]),
                  idx_mapping_np=np.array([0, 1]), has_prefill=False), 4

    def prepare_inputs(self, output, state, desc, loras):
        ids = torch.arange(8)
        if self.corrupt_input:
            ids[0] += 1
        return NS(input_ids=ids, positions=torch.arange(8), num_tokens=8)


def execute(runner, name="batch", reverse=False):
    ids = [f"cmpl-{name}-{i}-1234abcd" for i in range(2)]
    if reverse:
        ids.reverse()
    output = NS(num_scheduled_tokens={r: 4 for r in ids},
                scheduled_spec_decode_tokens={r: [1, 2, 3] for r in ids}, num_spec_tokens_to_schedule=3)
    state, uniform = runner.gather_batch_req_state(output, False)
    desc = runner.cudagraph_manager.dispatch(2, 8, uniform, 0)
    runner.prepare_inputs(output, state, desc, 0)
    runner.forward_calls += 1
    runner.speculator._cg_managers[5].dispatch(2, 10, 5, 0)


def test_actual_target_and_draft_descriptors_and_restoration():
    runner = Runner()
    trace = batch.BatchTrace(runner, manifest(), None)
    trace.install()
    execute(runner)
    pins = deepcopy(trace.pins())
    assert pins[0]["rows"] == [0, 1] and pins[0]["draft_k"] == 3
    assert pins[0]["verify_k"] == [3, 3]
    assert trace.steps[0]["dispatches"][1]["actual"]["num_tokens"] == 10
    trace.restore()
    assert "prepare_inputs" not in vars(runner)
    assert "dispatch" not in vars(runner.cudagraph_manager)
    # Logical graph descriptors remain pinned when actual execution is eager.
    runner.cudagraph_manager._graphs_captured = False
    runner.speculator._cg_managers[5]._graphs_captured = False
    repeat = batch.BatchTrace(runner, manifest("repeat"), pins)
    repeat.install()
    execute(runner, "repeat")
    assert repeat.pins() == pins
    assert repeat.steps[0]["target_descriptor"]["cg_mode"] == "NONE"
    repeat.restore()


@pytest.mark.parametrize("reverse,change_input", [(True, False), (False, True)])
def test_deliberately_changed_batch_is_pin_mismatch_before_forward(reverse, change_input):
    runner = Runner()
    first = batch.BatchTrace(runner, manifest(), None)
    first.install()
    execute(runner)
    pins = deepcopy(first.pins())
    first.restore()
    runner.forward_calls = 0
    runner.corrupt_input = change_input
    repeat = batch.BatchTrace(runner, manifest("repeat"), pins)
    repeat.install()
    with pytest.raises(RuntimeError, match="pin mismatch"):
        execute(runner, "repeat", reverse)
    assert runner.forward_calls == 0
    repeat.restore()


def test_rpc_establish_repeat_and_incomplete_end_restore_methods():
    runner = Runner()
    worker = NS(rank=0, model_runner=runner)
    assert batch.begin(worker, manifest(), "c2", True)["ok"]
    execute(runner)
    assert batch.end(worker)["ok"]
    assert batch.begin(worker, manifest("repeat"), "c2", False)["ok"]
    reply = batch.end(worker)
    assert not reply["ok"] and "no actual execution" in reply["reason"]
    assert "prepare_inputs" not in vars(runner) and "batch_trace" not in ab._STATE.saved


def test_partial_install_failure_restores_preexisting_methods(monkeypatch):
    runner = Runner()
    original = runner.prepare_inputs
    runner.prepare_inputs = original
    class Broken:
        @property
        def dispatch(self):
            raise RuntimeError("missing dispatch")
    runner.speculator._cg_managers[5] = Broken()
    reply = batch.begin(NS(rank=0, model_runner=runner), manifest(), "c2", True)
    assert not reply["ok"]
    assert runner.prepare_inputs is original
    assert "gather_batch_req_state" not in vars(runner)
    assert "dispatch" not in vars(runner.cudagraph_manager)


@pytest.mark.parametrize("prompt_tokens,input_budget,scheduled_budget,spec_tokens,admitted", [
    (256, 2048, None, 5, False),
    (254, 2048, None, 5, False),
    (252, 2048, None, 5, True),
    (256, 2080, None, 5, True),
    (252, 2048, 2015, 5, False),
    (256, 2048, None, None, True),
])
def test_real_scheduler_reserves_draft_input_capacity(
    tmp_path, monkeypatch, prompt_tokens, input_budget, scheduled_budget, spec_tokens, admitted
):
    import json

    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids

    # The existing model-free parallel-drafting fixture validates under MRV1.
    # Exercise real V2 scheduling and its sorter without constructing a runner;
    # the real SpeculativeConfig supplies Scheduler.schedule's reservation.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    scheduler = create_scheduler(
        model=str(tmp_path), max_num_seqs=8, max_num_batched_tokens=input_budget,
        max_model_len=2048, num_speculative_tokens=spec_tokens,
        parallel_drafting=spec_tokens is not None, skip_tokenizer_init=True,
        num_speculative_tokens_per_batch_size=[(1, 3, 5), (4, 8, 3)] if spec_tokens else None,
        scheduler_cls=batch.OrderedScheduler, use_v2_model_runner=True,
    )
    if scheduled_budget is not None:
        scheduler.max_num_scheduled_tokens = scheduled_budget
    spec = scheduler.vllm_config.speculative_config
    if spec is not None:
        assert spec.max_num_new_slots_for_drafting == 4
        assert scheduler.dynamic_sd_lookup[8] == 3
    ids = [f"cmpl-capacity-{i}-1234abcd" for i in range(8)]
    requests = create_requests(
        num_requests=8, num_tokens=prompt_tokens, same_prompt=True, req_ids=ids,
        max_tokens=32, ignore_eos=True,
    )
    declared = dict(id="capacity", rows=[
        dict(tokens=len(req.prompt_token_ids), sha256=batch.token_digest(req.prompt_token_ids))
        for req in requests
    ])
    for req in requests:
        req.sampling_params.extra_args = {"dev_ab_batch": json.dumps(declared)}
    if not admitted:
        with pytest.raises(ValueError, match="including draft reservations"):
            scheduler.add_request(requests[-1])
        assert not scheduler.requests and not scheduler.waiting and not scheduler._ordered_held
        return
    for req in reversed(requests):
        scheduler.add_request(req)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {rid: prompt_tokens for rid in ids}
    assert not any(req.is_prefill_chunk for req in requests)
    assert sort_batch_req_ids(
        output.num_scheduled_tokens, output.scheduled_spec_decode_tokens, (spec_tokens or 0) + 1
    ) == ids
    if spec is not None:
        assert output.num_spec_tokens_to_schedule == 3
