# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sm80.failed_inputs import failed_request_ids, relay_failed_tokens


def test_failed_tokens_map_to_their_requests():
    # Requests a=[0,3), b empty, c=[3,5), d=[5,7); tokens 7.. are padding.
    query_start_loc = np.array([0, 3, 3, 5, 7], dtype=np.int32)
    req_ids = ["a", "b", "c", "d"]
    mask = np.zeros(9, dtype=bool)
    assert failed_request_ids(mask, query_start_loc, req_ids) == set()
    mask[[3, 4, 8]] = True
    assert failed_request_ids(mask, query_start_loc, req_ids) == {"c"}
    mask[[0, 6]] = True
    assert failed_request_ids(mask, query_start_loc, req_ids) == {"a", "c", "d"}
    mask[:] = False
    mask[2] = True
    assert failed_request_ids(mask, query_start_loc, req_ids) == {"a"}


class _FailureSource(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def take_failed_tokens(self):
        mask, self.mask = self.mask, None
        return mask


def test_pp_failures_cross_hops_without_poisoning_the_next_batch():
    first = _FailureSource(torch.tensor([False, True, False, False, True]))
    middle = _FailureSource(torch.tensor([False, False, True, False, False]))
    last = torch.nn.Module()
    hop1 = SimpleNamespace(tensors={"failed_tokens": torch.empty(5, dtype=torch.bool)})
    hop2 = SimpleNamespace(tensors={"failed_tokens": torch.empty(5, dtype=torch.bool)})
    starts = np.array([0, 2, 4], dtype=np.int32)

    relay_failed_tokens(first, None, hop1)
    relay_failed_tokens(middle, hop1, hop2)
    result = relay_failed_tokens(last, hop2, None)
    # Reuse the persistent transport buffers before consuming the async result.
    # Stale failures must clear, and reuse must not erase the previous result.
    relay_failed_tokens(first, None, hop1)
    relay_failed_tokens(middle, hop1, hop2)
    healthy = relay_failed_tokens(last, hop2, None)
    assert failed_request_ids(result.numpy(), starts, ["a", "b"]) == {"a", "b"}
    assert failed_request_ids(healthy.numpy(), starts, ["c", "d"]) == set()


def test_pp_fails_closed_without_a_failure_transport_field():
    source = _FailureSource(torch.tensor([True]))
    with pytest.raises(RuntimeError, match="failed-token transport field"):
        relay_failed_tokens(source, None, SimpleNamespace(tensors={}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("failed", [False, True])
def test_failed_tokens_copy_outlives_the_callers_mask(failed: bool):
    """The runner drops its mask right after building AsyncOutput, while the
    copy to the host may still be queued behind other copies. The mask's
    memory must not go back to the allocator before that copy has run:
    reuse could fail healthy requests or clear a real failure."""
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    device = torch.device("cuda")
    main_stream = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device)
    num_tokens = 2048
    query_start_loc = np.array([0, num_tokens], dtype=np.int32)
    zeros = torch.zeros(1, 1, dtype=torch.int64, device=device)
    sampler_output = SamplerOutput(
        sampled_token_ids=zeros,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=zeros[0],
        num_rejected=zeros[0],
    )
    model_runner_output = ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0})
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    mask[7] = failed
    torch.accelerator.synchronize(device)
    # Hold the copy stream back, as earlier output copies can.
    with torch.cuda.stream(copy_stream):
        torch.cuda._sleep(200_000_000)
    output = AsyncOutput(
        model_runner_output=model_runner_output,
        sampler_output=sampler_output,
        num_sampled_tokens=zeros[0],
        main_stream=main_stream,
        copy_stream=copy_stream,
        check_ep_fault=False,
        pending_aux_output=None,
        failed_tokens=mask,
        query_start_loc_np=query_start_loc,
    )
    del mask
    # Later work on the main stream takes the freed block if it was released,
    # and writes the opposite of the mask over it.
    later = torch.empty(num_tokens, dtype=torch.bool, device=device)
    later.fill_(not failed)
    expected = {"a"} if failed else None
    assert output.get_output().failed_req_ids == expected
    del later
