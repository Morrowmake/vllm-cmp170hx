# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the GLM-5.x batch-sharded logits path
(``--enable-batch-sharded-sampling``).

The model runner selects batch-sharded sampling when the flag is on and
``hasattr(model, "compute_logits_local")``. Both fork GLM classes (which the
fork plugin registers for the two GLM architectures) define the method
unconditionally, as upstream models do, so the flag alone decides. These tests
pin that contract and the one-line semantics of the method itself, without a
GPU or a loaded checkpoint.
"""

import torch
from torch import nn

from vllm_sm80.models.glm5next.sm80_model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)


class _RecordingLogitsProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, lm_head, hidden_states, *args, **kwargs):
        self.calls.append((lm_head, hidden_states, args, kwargs))
        return torch.zeros(hidden_states.shape[0], 8)


def _bare_model() -> Glm5NextForCausalLM:
    """An instance with just the two attributes the logits path touches."""
    m = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(m)
    m.lm_head = object()
    m.logits_processor = _RecordingLogitsProcessor()
    return m


def test_both_classes_expose_compute_logits_local():
    assert hasattr(Glm5NextForCausalLM, "compute_logits_local")
    assert hasattr(Glm5NextForConditionalGeneration, "compute_logits_local")


def test_compute_logits_local_skips_the_gather():
    m = _bare_model()
    hidden = torch.zeros(4, 16)
    out = m.compute_logits_local(hidden)
    assert out.shape == (4, 8)
    assert len(m.logits_processor.calls) == 1
    lm_head, hs, args, kwargs = m.logits_processor.calls[0]
    assert lm_head is m.lm_head
    assert hs is hidden
    assert kwargs == {"skip_gather": True}, kwargs


def test_compute_logits_still_gathers():
    m = _bare_model()
    hidden = torch.zeros(2, 16)
    m.compute_logits(hidden)
    _, _, args, kwargs = m.logits_processor.calls[0]
    assert args == () and kwargs == {}


def test_wrapper_forwards_to_the_text_model():
    wrapper = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = _bare_model()
    hidden = torch.zeros(3, 16)
    assert wrapper.compute_logits_local(hidden).shape == (3, 8)
    _, hs, _, kwargs = wrapper.language_model.logits_processor.calls[0]
    assert hs is hidden and kwargs == {"skip_gather": True}
