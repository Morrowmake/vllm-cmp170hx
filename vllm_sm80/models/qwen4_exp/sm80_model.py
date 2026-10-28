# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model classes of the sm_80 fork.

The fork plugin (``vllm_sm80.plugin``) registers the classes of this module in
vLLM's model registry in place of the upstream
``Qwen4ExpForConditionalGeneration`` (``vllm/models/qwen4_exp/nvidia/model.py``)
and ``Qwen4ExpMTP`` (``mtp.py``). Each subclasses the upstream class and keeps
its name, so the registry reports the same model information and checkpoints
load into the same parameter names.

What the subclasses add, for ``VLLM_SM80_SHARD_ARGMAX``
(``vllm_sm80/worker/target_argmax.py``,
``vllm_sm80/spec_decode/shard_argmax.py``):

- ``compute_logits_local``: this rank's vocabulary shard of the logits,
  ungathered, padding columns kept: the logits processor on the head without
  the gather, the same call each class's ``compute_logits`` makes. The
  upstream wrapper inherits a ``compute_logits_local`` that asks its text
  model for one, which ``Qwen4ExpForCausalLM`` does not have;
- the target: the argmax rows from ``compute_logits`` in a greedy-only
  sampling step of the fork runner (``ShardArgmaxLogits``).
"""

import torch

from vllm.models.qwen4_exp.nvidia import model as qwen_model
from vllm.models.qwen4_exp.nvidia import mtp as qwen_mtp
from vllm_sm80.worker.target_argmax import ShardArgmaxLogits


class Qwen4ExpForConditionalGeneration(
    ShardArgmaxLogits, qwen_model.Qwen4ExpForConditionalGeneration
):
    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm = self.language_model
        return lm.logits_processor(lm.lm_head, hidden_states, skip_gather=True)


class Qwen4ExpMTP(qwen_mtp.Qwen4ExpMTP):
    def compute_logits_local(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)
