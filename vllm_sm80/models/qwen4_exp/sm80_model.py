# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model classes of the sm_80 fork.

The fork plugin (``vllm_sm80.plugin``) registers the classes of this module in
vLLM's model registry in place of the upstream ``Qwen4ExpForCausalLM`` and
``Qwen4ExpForConditionalGeneration`` (``vllm/models/qwen4_exp/nvidia/model.py``)
and ``Qwen4ExpMTP`` (``mtp.py``). Each subclasses the upstream class and keeps
its name, so the registry reports the same model information and checkpoints
load into the same parameter names.

Each class builds the upstream model and then, before any weight loads:

- sets the router logits dtype of every sparse MoE block
  (``precision.adopt_router_precision``, text config ``moe_router_dtype``);
- the text model: builds the LM head with the model's quantization config
  (upstream builds it unquantized; a quantized checkpoint's head is packed);
- the MTP drafter: the per-stream pre-FC hidden norm under
  ``mtp_hidden_norm="per_stream"`` (``precision.adopt_mtp_hidden_norm``), and
  no embedding or LM head of its own when the quantization config says the
  drafter shares the target's (``mtp_shares_target_embeddings``; the loader
  aliases them).

What the classes add, for ``VLLM_SM80_SHARD_ARGMAX``
(``vllm_sm80/worker/target_argmax.py``,
``vllm_sm80/spec_decode/shard_argmax.py``):

- ``compute_logits_local``: this rank's vocabulary shard of the logits,
  ungathered, padding columns kept: the logits processor on the head without
  the gather, the same call each class's ``compute_logits`` makes. The
  upstream wrapper inherits a ``compute_logits_local`` that asks its text
  model for one, which ``Qwen4ExpForCausalLM`` does not have;
- the target: the argmax rows from ``compute_logits`` in a greedy-only
  sampling step of the fork runner (``ShardArgmaxLogits``);
- the drafter: ``get_top_tokens``, the greedy draft tokens from the per-rank
  argmax over the vocabulary shard.
"""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.models.qwen4_exp.nvidia import model as qwen_model
from vllm.models.qwen4_exp.nvidia import mtp as qwen_mtp
from vllm_sm80.models.qwen4_exp.precision import (
    adopt_mtp_hidden_norm,
    adopt_router_precision,
)
from vllm_sm80.worker.target_argmax import ShardArgmaxLogits


def adopt_text_model(
    lm: qwen_model.Qwen4ExpForCausalLM, vllm_config: VllmConfig, prefix: str
) -> None:
    """The sm_80 changes to an upstream-built ``Qwen4ExpForCausalLM`` whose
    prefix is ``prefix``."""
    adopt_router_precision(vllm_config, lm)
    if lm.quant_config is None:
        return
    config = lm.config
    # Free the unquantized head before building the quantized one in its place.
    lm.lm_head = None
    lm.lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        quant_config=lm.quant_config,
        prefix=maybe_prefix(prefix, "lm_head"),
    )


class Qwen4ExpForCausalLM(qwen_model.Qwen4ExpForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        adopt_text_model(self, vllm_config, prefix)


class Qwen4ExpForConditionalGeneration(
    ShardArgmaxLogits, qwen_model.Qwen4ExpForConditionalGeneration
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Not built in --mm-encoder-only mode (a StageMissingLayer instead).
        if isinstance(self.language_model, qwen_model.Qwen4ExpForCausalLM):
            adopt_text_model(
                self.language_model,
                vllm_config,
                maybe_prefix(prefix, "language_model"),
            )

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm = self.language_model
        return lm.logits_processor(lm.lm_head, hidden_states, skip_gather=True)


class Qwen4ExpMTP(qwen_mtp.Qwen4ExpMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        adopt_router_precision(vllm_config, self)
        adopt_mtp_hidden_norm(vllm_config, self.model)
        shares = getattr(self.quant_config, "mtp_shares_target_embeddings", None)
        if shares is not None and shares(self.config):
            self.model.embed_tokens = None
            self.model.embed_tokens = PPMissingLayer()
            self.lm_head = None
            self.lm_head = PPMissingLayer()

    def compute_logits_local(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        """Greedy draft token ids via per-rank argmax over the vocab shard.
        Saves the full-vocab all-gather ``compute_logits`` does; same tokens.
        Name is fixed by the protocol the proposer probes for
        (``use_local_argmax_reduction``).
        """
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
