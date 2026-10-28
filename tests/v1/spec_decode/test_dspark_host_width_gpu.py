# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark host width's device side on one sm_80 card.

- ``compute_row_mask`` with a ``p_hat`` output: the same ``live_len`` and
  ``row_live`` as without it, bitwise, and the torch reference's ``p_i`` for
  positions 1 .. 5 at every K (random and draft-logit-statistics policies,
  K = 1, 3, 5, 7, b = 1 .. 8).
- The GPU runner's ``AsyncOutput`` (wrapped by ``install_output_copy``):
  a step with ``p_hat`` carries its rows, in batch order, to the
  ``ModelRunnerOutput`` after ``get_output``; the step's tensor is taken once;
  a step without one carries nothing.

    pytest -q tests/v1/spec_decode/test_dspark_host_width_gpu.py
"""

import pytest
import torch

from vllm_sm80.spec_decode import dspark_row_mask
from vllm_sm80.spec_decode import dspark_row_mask_kernel as rmk
from vllm_sm80.spec_decode.dspark_host_width import P_HAT_ATTR, install_output_copy
from vllm_sm80.spec_decode.dspark_row_mask import FEATURES, RowMaskState

MAX_REQS, MAX_TOKENS, MAX_DRAFT, WIDTH = 8, 96, 7, 5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80 card",
)


def _state(gen, device, uses_stats):
    s = RowMaskState(MAX_REQS, MAX_TOKENS, MAX_DRAFT, torch.device(device))
    s.conf_by_slot.copy_(torch.rand(MAX_REQS, MAX_DRAFT, generator=gen).sqrt())
    stats = torch.stack(
        (torch.rand(MAX_REQS, MAX_DRAFT, generator=gen),
         torch.rand(MAX_REQS, MAX_DRAFT, generator=gen) * 4.0), dim=1)
    s.stats_by_slot.copy_(stats)
    w = torch.randn(MAX_DRAFT, len(FEATURES), generator=gen)
    if not uses_stats:
        w[:, 2:] = 0.0
    s.weights.copy_(w)
    s.cost.copy_(20.0 + (torch.rand(MAX_REQS + 1, MAX_DRAFT + 1, generator=gen) + 0.05).cumsum(1))
    tau = torch.zeros(MAX_REQS + 1)
    tau[2:] = torch.rand(MAX_REQS - 1, generator=gen) * 0.6 + 0.05
    s.tau.copy_(tau)
    s.uses_stats = uses_stats
    return s


@pytest.mark.parametrize("uses_stats", (False, True))
def test_kernel_p_hat(uses_stats):
    for b in range(1, MAX_REQS + 1):
        for k in (1, 3, 5, 7):
            gen = torch.Generator().manual_seed(100 * b + k)
            idx = torch.randperm(MAX_REQS, generator=gen)[:b].to(torch.int32)
            plain = _state(gen.manual_seed(7 * b + k), "cuda", uses_stats)
            with_p = _state(gen.manual_seed(7 * b + k), "cuda", uses_stats)
            ref = _state(gen.manual_seed(7 * b + k), "cpu", uses_stats)
            rmk.compute_row_mask(plain, idx.cuda(), b, k)
            p_hat = torch.full((MAX_REQS, WIDTH), -2.0, device="cuda")
            rmk.compute_row_mask(with_p, idx.cuda(), b, k, p_hat)
            want = torch.empty(b, WIDTH)
            rmk.reference_row_mask(ref, idx, b, k, want)
            assert torch.equal(plain.live_len, with_p.live_len), (b, k)
            assert torch.equal(plain.row_live, with_p.row_live), (b, k)
            torch.testing.assert_close(p_hat[:b].cpu(), want, rtol=1e-5, atol=1e-7)
            assert (p_hat[b:] == -2.0).all(), (b, k)


def _async_output(req_ids):
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    n = len(req_ids)
    dev = torch.device("cuda")
    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=None,  # type: ignore[arg-type]
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
    )
    sampled = torch.arange(n * 6, dtype=torch.int32, device=dev).view(n, 6)
    ones = torch.ones(n, dtype=torch.int32, device=dev)
    sampler_output = SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=ones,
        num_rejected=ones * 0,
    )
    return AsyncOutput(
        model_runner_output=mro,
        sampler_output=sampler_output,
        num_sampled_tokens=ones,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        check_ep_fault=False,
        pending_aux_output=None,
    )


def test_async_output_carries_p_hat(monkeypatch):
    from vllm.v1.worker.gpu import async_utils

    monkeypatch.setattr(async_utils.AsyncOutput, "__init__", async_utils.AsyncOutput.__init__)
    monkeypatch.setattr(async_utils.AsyncOutput, "get_output", async_utils.AsyncOutput.get_output)
    install_output_copy()
    install_output_copy()
    state = RowMaskState(MAX_REQS, MAX_TOKENS, MAX_DRAFT, torch.device("cuda"))
    state.p_hat_width = WIDTH
    monkeypatch.setattr(dspark_row_mask, "_STATE", state)
    p_hat = torch.rand(3, WIDTH, device="cuda")
    # Written on the main stream right before the output is built.
    state.p_hat_step = p_hat * 1.0
    output = _async_output(["x", "y", "z"]).get_output()
    assert state.p_hat_step is None
    assert getattr(output, P_HAT_ATTR) == p_hat.cpu().tolist()
    assert output.sampled_token_ids == [[0], [6], [12]]
    output = _async_output(["x", "y"]).get_output()
    assert not hasattr(output, P_HAT_ATTR)
