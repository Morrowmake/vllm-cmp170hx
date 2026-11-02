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
  aliases them);
- an EXL3 checkpoint under pipeline parallelism (``adopt_exl3_stages``): a
  PPMissingLayer for the final HC mixer of a stage without it, by which the
  EXL3 loader accounts for that stage's unowned tensors; the target
  embedding on the last stage too when the MTP drafter there shares it
  (upstream builds it on the first stage only); and the ``failed_tokens``
  field of the intermediate tensors, which carries the SSD read failures of
  the PLE stage to the sampling stage (``vllm_sm80.failed_inputs``).

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
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.models.qwen4_exp.nvidia import model as qwen_model
from vllm.models.qwen4_exp.nvidia import mtp as qwen_mtp
from vllm_sm80.models.qwen4_exp.precision import (
    adopt_mtp_hidden_norm,
    adopt_router_precision,
)
from vllm_sm80.worker.target_argmax import ShardArgmaxLogits


def _exl3_pipeline(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    return (
        vllm_config.parallel_config.pipeline_parallel_size > 1
        and model_config is not None
        and model_config.quantization == "exl3"
    )


def adopt_exl3_stages(
    lm: qwen_model.Qwen4ExpForCausalLM, vllm_config: VllmConfig
) -> None:
    """The stage-local modules of an EXL3 checkpoint under pipeline
    parallelism, before any weight loads."""
    if not _exl3_pipeline(vllm_config):
        return
    model = lm.model
    if model.hyper_connection_mixer is None:
        model.hyper_connection_mixer = PPMissingLayer()
    spec = vllm_config.speculative_config
    shares = getattr(lm.quant_config, "mtp_shares_target_embeddings", None)
    if (
        get_pp_group().is_last_rank
        and isinstance(model.embed_tokens, PPMissingLayer)
        and spec is not None
        and spec.method == "mtp"
        and shares is not None
        and shares(lm.config)
    ):
        # The drafter on this stage takes the target embedding
        # (Qwen4ExpMTP.has_own_embed_tokens = False).
        model.embed_tokens = VocabParallelEmbedding(
            model.vocab_size, lm.config.hidden_size
        )


def adopt_pp_failed_tokens(model: torch.nn.Module, vllm_config: VllmConfig) -> None:
    """Add the ``failed_tokens`` field to the intermediate tensors of an EXL3
    PLE model under pipeline parallelism: each stage's receive buffers get
    it, and ``failed_inputs.relay_failed_tokens`` fills it on each sending
    stage (``pp_failed_tokens``)."""
    if not _exl3_pipeline(vllm_config) or not getattr(
        vllm_config.model_config.hf_text_config, "ple_layer_ids", None
    ):
        return
    make = model.make_empty_intermediate_tensors

    def make_with_failed_tokens(batch_size, dtype, device):
        tensors = make(batch_size, dtype, device)
        tensors.tensors["failed_tokens"] = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        return tensors

    model.make_empty_intermediate_tensors = make_with_failed_tokens
    model.pp_failed_tokens = True


def adopt_text_model(
    lm: qwen_model.Qwen4ExpForCausalLM, vllm_config: VllmConfig, prefix: str
) -> None:
    """The sm_80 changes to an upstream-built ``Qwen4ExpForCausalLM`` whose
    prefix is ``prefix``."""
    adopt_router_precision(vllm_config, lm)
    adopt_exl3_stages(lm, vllm_config)
    # A pipeline stage without the head (upstream builds a PPMissingLayer
    # there) keeps it.
    if lm.quant_config is None or isinstance(lm.lm_head, PPMissingLayer):
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
        adopt_pp_failed_tokens(self, vllm_config)


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
            adopt_pp_failed_tokens(self, vllm_config)

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
            # Declared, so the target embedding is shared under pipeline
            # parallelism too (eagle.utils.maybe_share_target_embed).
            self.has_own_embed_tokens = False

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
