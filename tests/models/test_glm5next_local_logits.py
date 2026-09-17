# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the GLM-5.x batch-sharded logits path (VLLM_GLM5_LOCAL_LOGITS).

The model runner selects batch-sharded sampling with
``hasattr(model, "compute_logits_local")``, checked once at load time, so the
env gate has to hide the *attribute*, not just change what the method does.
These tests pin that contract and the one-line semantics of the method itself,
without a GPU or a loaded checkpoint.
"""

import pytest
import torch
from torch import nn

from vllm.models.glm5next.common import model as glm5_model
from vllm.models.glm5next.common.model import Glm5NextForCausalLM


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


def test_class_does_not_expose_compute_logits_local():
    """Default off: the runner's hasattr probe must fail on the class."""
    assert not hasattr(Glm5NextForCausalLM, "compute_logits_local")
    assert hasattr(Glm5NextForCausalLM, "_compute_logits_local")


def test_instance_does_not_expose_it_either_by_default():
    assert not hasattr(_bare_model(), "compute_logits_local")


def test_compute_logits_local_skips_the_gather():
    m = _bare_model()
    hidden = torch.zeros(4, 16)
    out = m._compute_logits_local(hidden)
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


def test_binding_makes_hasattr_true():
    """What __init__ does when the flag is on."""
    m = _bare_model()
    m.compute_logits_local = m._compute_logits_local
    assert hasattr(m, "compute_logits_local")
    m.compute_logits_local(torch.zeros(1, 16))
    assert m.logits_processor.calls[0][3] == {"skip_gather": True}


@pytest.mark.parametrize(
    "flag,tp,expected",
    [(False, 4, False), (True, 1, False), (True, 4, True), (False, 1, False)],
)
def test_use_local_logits_gate(monkeypatch, flag, tp, expected):
    monkeypatch.setattr(glm5_model.envs, "VLLM_GLM5_LOCAL_LOGITS", flag, raising=False)
    monkeypatch.setattr(
        glm5_model, "get_tensor_model_parallel_world_size", lambda: tp
    )
    assert glm5_model.use_local_logits() is expected
