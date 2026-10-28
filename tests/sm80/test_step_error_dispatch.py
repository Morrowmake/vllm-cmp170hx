# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A rank whose successful output is discarded must still observe device errors."""
from queue import Queue
from types import SimpleNamespace

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput


class CheckedOutput(AsyncModelRunnerOutput):
    _sm80_step_errors = True

    def __init__(self, fail):
        self.fail = fail
        self.reads = 0

    def get_output(self):
        self.reads += 1
        if self.fail:
            raise ValueError("copy-engine ring all-reduce: a peer flag did not arrive")
        return "valid output"

    def get_kv_connector_output(self):
        return None


def _worker(output, rank=3):
    worker = object.__new__(WorkerProc)
    worker.rank = rank
    worker.worker = SimpleNamespace(sample_tokens=lambda: output)
    worker.use_async_scheduling = True
    worker.async_output_queue = Queue()
    replies = []
    worker.worker_response_mq = SimpleNamespace(enqueue=replies.append)
    return worker, replies


@pytest.mark.parametrize("fail", [False, True])
def test_non_output_rank_consumes_existing_async_output(fail):
    output = CheckedOutput(fail)
    worker, replies = _worker(output)
    worker._execute_worker_rpc(("sample_tokens", (), {}, 0))
    assert not worker.async_output_queue.empty(), "the non-output rank dropped its device error check"
    assert output.reads == 0, "the RPC thread must not wait for the output copy"
    queued = worker.async_output_queue.get_nowait()
    if fail:
        with pytest.raises(ValueError, match="a peer flag did not arrive"):
            worker.enqueue_output(queued)
    else:
        worker.enqueue_output(queued)
    assert output.reads == 1
    assert replies == [], "a discarded result must not corrupt the next RPC response"


def test_default_non_output_behavior_is_unchanged():
    worker, replies = _worker(object())
    worker._execute_worker_rpc(("sample_tokens", (), {}, 0))
    assert worker.async_output_queue.empty()
    assert replies == []


def test_reply_rank_keeps_the_normal_failure_response():
    output = CheckedOutput(True)
    worker, replies = _worker(output, rank=0)
    worker._execute_worker_rpc(("sample_tokens", (), {}, 0))
    worker.enqueue_output(worker.async_output_queue.get_nowait())
    assert output.reads == 1
    assert replies == [(WorkerProc.ResponseStatus.FAILURE,
                        "copy-engine ring all-reduce: a peer flag did not arrive")]
