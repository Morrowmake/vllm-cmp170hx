# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the address and layout contracts of breakable PLE graphs."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def _layout_case():
    from vllm.models.qwen4_exp.common.ops.ple import _ple_ngram_ids_kernel
    from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState

    state = object.__new__(Qwen4ExpModelState)
    state.rope_state = None
    state.supports_mm_inputs = False
    state.prompt_embeds_state = None
    state.uses_ngram_embedding = True
    state.ngram_eos_token_id = 0
    state.ngram_context = torch.zeros((8, 2), dtype=torch.int32)
    state.ngram_context_offsets = torch.tensor([-2, -1])
    state.ple_query_start_loc = torch.zeros(9, dtype=torch.int32)
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.full((8,), 2)),
        all_token_ids=SimpleNamespace(gpu=torch.arange(1, 129).reshape(8, 16)),
    )
    multipliers = torch.tensor([11, 37, 101], dtype=torch.int64)
    sizes = torch.tensor([997, 991], dtype=torch.int64)
    offsets = torch.tensor([0, 997], dtype=torch.int64)

    def ids(tokens, layout, context):
        out = torch.empty((len(tokens), 2), dtype=torch.int64)
        n = layout.numel() - 1
        _ple_ngram_ids_kernel[((len(tokens) + 7) // 8,)](
            tokens, layout, context, multipliers, sizes, offsets, out,
            len(tokens), n, 0, n.bit_length(),
            NGRAM_CONTEXT_LEN=2, HEADS_PER_NGRAM=1, BLOCK_T=8, launch_pdl=False,
        )
        return out

    captured = state.prepare_dummy_inputs(8, 8)
    captured_layout = captured["query_start_loc"]
    captured_context = captured["ngram_context"]
    ptrs = (captured_layout.data_ptr(), captured_context.data_ptr())
    for lengths in ([1] * 8, [7], [3, 4], [1] * 8, [2]):
        n = len(lengths)
        qsl = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        batch = SimpleNamespace(
            num_reqs=n, num_reqs_after_padding=n, query_start_loc=qsl,
            idx_mapping=torch.arange(7, 7 - n, -1, dtype=torch.int32),
        )
        inputs = state.prepare_inputs(batch, req_states)
        assert (inputs["query_start_loc"].data_ptr(), inputs["ngram_context"].data_ptr()) == ptrs
        assert inputs["query_start_loc"].numel() == 9
        tokens = torch.arange(101, 101 + sum(lengths), dtype=torch.int32)
        expected = ids(tokens, qsl, captured_context[:n])
        torch.testing.assert_close(ids(tokens, captured_layout, captured_context), expected)
        assert captured_layout[n + 1 :].tolist() == [sum(lengths)] * (8 - n)
        if lengths == [7]:
            stale = torch.arange(9, dtype=torch.int32)
            stale[:n + 1].copy_(qsl)
            assert not torch.equal(ids(tokens, stale, captured_context), expected)
    # Dummy capture layouts must reset the unused tail as well.
    state.prepare_dummy_inputs(1, 7)
    assert captured_layout.tolist() == [0] + [7] * 8


def _replay_case():
    from torch import nn

    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture, eager_break_during_capture,
    )
    from vllm_sm80.models.qwen4_exp.exl3_ple import Exl3PLEEmbedding, Exl3PLESlot

    class CPUCapture(BreakableCUDAGraphCapture):
        # Only stream capture is simulated; add_eager, the decorator's closure
        # binding, and replay are the production implementations.
        def _begin_segment(self):
            self._capturing = True

        def _end_segment(self):
            self._capturing = False

    event = SimpleNamespace(record=lambda stream: None)
    stream = SimpleNamespace(wait_event=lambda event: None)
    module = object.__new__(Exl3PLEEmbedding)
    nn.Module.__init__(module)
    module._decoded = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    module._output_dim = 8
    module._heads = 2
    module.embedding_dim = 4
    module._gathered_tokens = module._local_tokens = 3
    module._slot_offset = 0
    module.tp_size = module.etp_data_parallel_size = 1
    module._deferred = None
    module._ready = module._released = event
    module._slot = Exl3PLESlot(6, event)
    module._slot.state = "ready"
    with (
        patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
        patch.object(torch.cuda, "current_stream", lambda: stream),
        # The native non-owning Tensor wrapper has no CPU dispatch. Keep the
        # exact same Tensor/address alive while exercising the replay closure.
        patch("vllm.compilation.breakable_cudagraph.weak_ref_tensor",
              lambda value: value),
    ):
        capture = CPUCapture()
        with capture:
            output = module.finalize_prefetch(3)
        assert capture.num_eager_breaks == 1
        address = output.data_ptr()
        for value in (101, 211):
            module._decoded.fill_(value)
            module._slot.state = "ready"
            capture.replay()
            assert output.data_ptr() == address
            assert output.tolist() == [[value] * 8] * 3

        # A newly allocated eager return is the historical failure mode:
        # replay ignores it and the consumer keeps reading the captured tensor.
        @eager_break_during_capture
        def allocating_lookup():
            return module._decoded.clone()

        broken = CPUCapture()
        with broken:
            stale_output = allocating_lookup()
        module._decoded.fill_(317)
        broken.replay()
        assert not torch.equal(stale_output, module._decoded)


@pytest.mark.parametrize("case", ["layout", "replay"])
def test_ple_capture_contract_on_cpu(case):
    result = subprocess.run(
        [sys.executable, "-m", __name__, case],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
                 VLLM_USE_BREAKABLE_CUDAGRAPH="1"),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    {"layout": _layout_case, "replay": _replay_case}[sys.argv[1]]()
