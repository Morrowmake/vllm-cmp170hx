# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 prepare_inputs keeps the padding buffer the sm80 MoE padding mask reads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner


def _prepare_padding_batch(monkeypatch, lengths, padding):
    """Run prepare_inputs on CPU; replace only device input-building kernels."""
    nreq, ntok = len(lengths), sum(lengths)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = nreq
    runner.decode_query_len = 8
    runner.input_buffers = InputBuffers(nreq, len(padding), runner.device)
    runner.input_buffers.is_padding.copy_(padding)
    runner.input_buffers.positions[:ntok] = torch.cat(
        [torch.arange(10, 10 + length) for length in lengths]
    )
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.adaptive_verification = None
    runner.fast_prefill = None
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.req_states = SimpleNamespace(
        num_computed_tokens_np=np.full(nreq, 10, dtype=np.int32),
        num_computed_tokens=SimpleNamespace(gpu=torch.full((nreq,), 10)),
        last_sampled_tokens=None,
        prefill_len=SimpleNamespace(gpu=torch.zeros(nreq)),
        draft_tokens=None,
    )

    def h2d(values, device=None, dtype=None, out=None):
        tensor = torch.as_tensor(values, dtype=dtype)
        return out.copy_(tensor) if out is not None else tensor

    def expand(mapping, total, boundaries, _decode_len):
        counts = boundaries[1:] - boundaries[:-1]
        return mapping.repeat_interleave(counts), torch.cat(
            [torch.arange(int(count)) for count in counts]
        )

    monkeypatch.setattr(model_runner_module, "async_tensor_h2d", h2d)
    monkeypatch.setattr(model_runner_module, "expand_idx_mapping", expand)
    monkeypatch.setattr(model_runner_module, "prepare_pos_seq_lens", lambda *a: None)
    monkeypatch.setattr(
        model_runner_module,
        "combine_sampled_and_draft_tokens",
        lambda *a: torch.arange(ntok),
    )
    req_ids = [str(i) for i in range(nreq)]
    scheduler = SimpleNamespace(
        scheduled_spec_decode_tokens={
            req_id: [1] * (length - 1)
            for req_id, length in zip(req_ids, lengths)
            if length > 1
        },
        has_structured_output_requests=False,
    )
    state = BatchReqState(
        num_tokens=ntok,
        req_ids=req_ids,
        num_scheduled_tokens=np.array(lengths, dtype=np.int32),
        num_draft_tokens_np=np.array(lengths, dtype=np.int32) - 1,
        idx_mapping_np=np.arange(nreq),
        has_prefill=False,
        prefill_len_np=np.zeros(nreq, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(nreq, dtype=np.int32),
        is_prefilling_np=np.zeros(nreq, dtype=np.bool_),
        max_seq_len_np=None,
        prefill_runs_as_decode_np=None,
        decode_graph_eligible=True,
    )
    desc = SimpleNamespace(num_tokens=len(padding), num_reqs=nreq)
    return runner, runner.prepare_inputs(scheduler, state, desc, 0)


@pytest.mark.parametrize("generic_skip", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_prepare_inputs_preserves_base_padding_and_routes(
    monkeypatch, generic_skip, seed
):
    """MASK_PADDING alone must preserve the base buffer/routing contract."""
    from vllm_sm80.layers.moe import routed_experts as mr

    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", str(int(generic_skip)))
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    from tests.sm80.tuning_rows import use_rows

    use_rows(
        monkeypatch, ("moe_router", "fused_route_max_rows", {"top_k": 8}, 8)
    )
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 9, (1 + seed % 4,), generator=g).tolist()
    ntok = sum(lengths)
    padding = torch.randint(0, 2, (ntok + seed % 5 + 3,), generator=g).bool()
    expected = padding.clone()
    # Only the upstream padding switch initializes this buffer.
    if generic_skip:
        expected[:ntok] = False
        expected[ntok:] = True
    runner, batch = _prepare_padding_batch(monkeypatch, lengths, padding)
    assert batch.is_padding.data_ptr() == runner.input_buffers.is_padding.data_ptr()
    assert torch.equal(batch.is_padding, expected)
    assert torch.equal(runner.input_buffers.is_padding, expected)

    monkeypatch.setattr(mr, "_MASK_PADDING", None)
    monkeypatch.setattr(mr, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mr, "get_forward_context", lambda: batch)
    ids = torch.randint(0, 288, (len(padding), 8), generator=g, dtype=torch.int32)
    expected_ids = ids.clone()
    if len(padding) > 8:
        expected_ids.masked_fill_(expected[:, None], -1)
    assert mr.mask_padding_topk_ids(ids) is ids
    assert torch.equal(ids, expected_ids)
    assert not torch.cuda.is_initialized()
