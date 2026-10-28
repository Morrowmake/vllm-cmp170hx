# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests whose model inputs failed on a worker finish with an error, the
rest of the batch keeps running, and no request is served from KV computed
from those inputs."""

from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler


def _step(scheduler, failed=()):
    scheduler_output = scheduler.schedule()
    req_ids = list(scheduler_output.num_scheduled_tokens)
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[7] for _ in req_ids],
        failed_req_ids=set(failed) or None,
    )
    engine_outputs = scheduler.update_from_output(scheduler_output, output)
    reasons = {
        out.request_id: out.finish_reason
        for outputs in engine_outputs.values()
        for out in outputs.outputs
        if out.finish_reason is not None
    }
    return scheduler_output, reasons


def _blocks(scheduler, request):
    return scheduler.kv_cache_manager.coordinator.get_blocks(request.request_id)[0]


def test_failed_request_errors_and_the_batch_keeps_running():
    scheduler = create_scheduler(enable_prefix_caching=True)
    a, b = create_requests(num_requests=2, num_tokens=40)
    for request in (a, b):
        scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    a_blocks = list(_blocks(scheduler, a))
    assert all(block.block_hash is not None for block in a_blocks[:2])
    req_ids = list(scheduler_output.num_scheduled_tokens)
    engine_outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[7], [7]],
            failed_req_ids={a.request_id},
        ),
    )
    outputs = {o.request_id: o for e in engine_outputs.values() for o in e.outputs}
    assert a.status == RequestStatus.FINISHED_ERROR
    assert outputs[a.request_id].finish_reason == FinishReason.ERROR
    assert outputs[a.request_id].new_token_ids == []
    assert list(a.output_token_ids) == []
    assert all(block.block_hash is None for block in a_blocks)
    assert list(b.output_token_ids) == [7]
    assert b.status == RequestStatus.RUNNING
    # The batch is not stuck: B keeps decoding and a request with A's prompt
    # recomputes it instead of hitting A's dropped blocks.
    (c,) = create_requests(num_requests=1, num_tokens=40, req_ids=["c"])
    assert c.prompt_token_ids == a.prompt_token_ids
    scheduler.add_request(c)
    scheduler_output, reasons = _step(scheduler)
    assert scheduler_output.num_scheduled_tokens == {b.request_id: 1, "c": 40}
    assert reasons == {}
    assert list(b.output_token_ids) == [7, 7]
    assert list(c.output_token_ids) == [7]


def test_requests_reusing_failed_kv_also_fail():
    scheduler = create_scheduler(enable_prefix_caching=True)
    a, b = create_requests(num_requests=2, num_tokens=40, same_prompt=True)
    other = create_requests(num_requests=2, num_tokens=40, req_ids=["y", "x"])[1]
    assert other.prompt_token_ids != a.prompt_token_ids
    for request in (a, b, other):
        scheduler.add_request(request)
    scheduler_output, reasons = _step(scheduler, failed={a.request_id})
    # B reused A's blocks in the same step, so it saw A's wrong KV.
    assert scheduler_output.num_scheduled_tokens[b.request_id] == 8
    assert reasons == dict.fromkeys((a.request_id, b.request_id), FinishReason.ERROR)
    assert list(b.output_token_ids) == []
    assert other.status == RequestStatus.RUNNING
    assert list(other.output_token_ids) == [7]


def test_reuse_of_kv_computed_before_the_failed_step_is_kept():
    scheduler = create_scheduler(enable_prefix_caching=True)
    (a,) = create_requests(num_requests=1, num_tokens=40)
    scheduler.add_request(a)
    _step(scheduler)
    (b,) = create_requests(num_requests=1, num_tokens=40, req_ids=["b"])
    scheduler.add_request(b)
    shared = list(_blocks(scheduler, a)[:2])
    scheduler_output, reasons = _step(scheduler, failed={a.request_id})
    # B reused A's prompt blocks, which A computed in an earlier, good step.
    assert scheduler_output.num_scheduled_tokens == {a.request_id: 1, "b": 8}
    assert [blk.block_id for blk in _blocks(scheduler, b)[:2]] == [
        blk.block_id for blk in shared
    ]
    assert reasons == {a.request_id: FinishReason.ERROR}
    assert b.status == RequestStatus.RUNNING
    assert list(b.output_token_ids) == [7]
    assert all(block.block_hash is not None for block in shared)
