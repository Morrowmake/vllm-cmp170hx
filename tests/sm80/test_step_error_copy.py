# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.distributed import step_errors as errors
from vllm.v1.worker.gpu import async_utils


class Event:
    def __init__(self, **kwargs):
        self.records = self.syncs = 0
    def record(self, *args):
        self.records += 1
    def synchronize(self):
        self.syncs += 1


@pytest.mark.parametrize("flag", [None, 0, errors.CE_RING_TIMEOUT])
def test_error_joins_the_existing_copy_event(monkeypatch, flag):
    monkeypatch.setattr(errors, "_buffers", {} if flag is None else {(0, 1): torch.tensor(flag)})
    events = []
    def event(**kwargs):
        value = Event()
        events.append(value)
        return value
    stream = SimpleNamespace(device=torch.device("cuda", 0), wait_stream=lambda _: None)
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())
    output = SimpleNamespace(prompt_logprobs_dict={}, prompt_token_id_logprobs_dict={}, sampled_token_ids=None)
    sampler = SimpleNamespace(sampled_token_ids=torch.tensor([[7]]), logprobs_tensors=None,
                              num_nans=None, sampling_mask_tensors=None)
    result = async_utils.AsyncOutput(output, sampler, torch.ones(1, dtype=torch.int32),
                                     stream, stream, False, None)
    assert len(events) == 1 and events[0].syncs == 0
    assert events[0].records == (1 if flag is None else 2)
    if flag:
        with pytest.raises(ValueError, match="copy-engine ring"):
            result.get_output()
        assert output.sampled_token_ids is None, "invalid tokens escaped before error checking"
    else:
        assert result.get_output().sampled_token_ids == [[7]]
    assert events[0].syncs == 1
    assert hasattr(result, "_sm80_step_errors") is (flag is not None)


def test_sync_runner_reuses_its_existing_event(monkeypatch):
    monkeypatch.setattr(errors, "_buffers", {(None, 1): torch.tensor(errors.CE_RING_TIMEOUT)})
    class Runner:
        def __init__(self):
            self.transfer_event = Event()
        def _to_list(self, tensor):
            self.transfer_event.record()
            self.transfer_event.synchronize()
            return tensor.tolist()
    errors.install_output_checks(sync_runner=Runner)
    runner = Runner()
    original = runner.transfer_event
    with pytest.raises(ValueError, match="copy-engine ring"):
        runner._to_list(torch.tensor([7]))
    assert runner.transfer_event is original
    assert original.records == original.syncs == 1
