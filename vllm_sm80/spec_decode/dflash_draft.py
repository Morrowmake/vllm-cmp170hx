# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the sm_80 fork adds to every DFlash-family drafter it registers.

:class:`DFlashDraftMixin` goes before the upstream class of each fork drafter
class (``DFlashQwen3ForCausalLM`` here, ``dflash2_model.py``,
``lilicorr_model.py``), all of which build ``DFlashQwen3DecoderLayer``s
(``vllm/model_executor/models/qwen3_dflash.py``):

- ``VLLM_GLM5_DRAFTER_ROPE_FIT``: the drafter is built with the RoPE cache
  cut to the positions the draft can reach (:func:`dflash_rope_max_position`);
- ``VLLM_GLM5_PP_FOLD_DRAFT_FC`` (``aux_fc_fold.py``): when the pipeline
  stages already applied the drafter's ``fc`` to the target's aux hidden
  states, ``combine_hidden_states`` only rounds their fp32 sum to the model
  dtype.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import Qwen3Config

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.models import qwen3_dflash

logger = init_logger(__name__)

# Alignment of the fitted drafter RoPE cache length.
_ROPE_FIT_ALIGN = 64


def dflash_rope_max_position(vllm_config: VllmConfig, config: Qwen3Config) -> int:
    """Number of RoPE cache rows the draft attention needs.

    By default this is the draft config's ``max_position_embeddings``. With
    ``VLLM_GLM5_DRAFTER_ROPE_FIT=1`` it is cut to the positions the draft can
    reach. Context positions are the target's (always < max_model_len). Query
    positions are the last context position + 1 + offset, with offset below
    1 + num_speculative_tokens; one input path clamps them to max_model_len - 1,
    the other does not, so the largest index is max_model_len +
    num_speculative_tokens. The cache keeps max_model_len + 1 +
    num_speculative_tokens rows, rounded up to ``_ROPE_FIT_ALIGN``.

    Only plain RoPE is fitted: there each row depends on its position and the
    base alone, so every kept row is identical to the unfitted cache. Scaled
    variants keep the configured size.
    """
    configured = int(config.max_position_embeddings)
    if not envs.VLLM_GLM5_DRAFTER_ROPE_FIT:
        return configured

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if (
        rope_parameters.get("rope_type", "default") != "default"
        or "mrope_section" in rope_parameters
        or rope_parameters.get("use_fope", False)
        or getattr(config, "dual_chunk_attention_config", None) is not None
    ):
        logger.info_once(
            "VLLM_GLM5_DRAFTER_ROPE_FIT: draft RoPE is not plain (%s); "
            "keeping %d positions.",
            rope_parameters.get("rope_type", "default"),
            configured,
        )
        return configured

    max_model_len = int(vllm_config.model_config.max_model_len)
    speculative_config = vllm_config.speculative_config
    num_query_per_req = 1
    if speculative_config is not None:
        num_query_per_req += int(speculative_config.num_speculative_tokens or 0)
        target_model_config = speculative_config.target_model_config
        if target_model_config is not None:
            max_model_len = max(max_model_len, int(target_model_config.max_model_len))

    reach = max_model_len + num_query_per_req
    fitted = -(-reach // _ROPE_FIT_ALIGN) * _ROPE_FIT_ALIGN
    fitted = min(configured, fitted)
    logger.info_once(
        "VLLM_GLM5_DRAFTER_ROPE_FIT: draft RoPE cache %d -> %d positions "
        "(max_model_len %d, %d query tokens per request).",
        configured,
        fitted,
        max_model_len,
        num_query_per_req,
    )
    return fitted


@contextmanager
def fitted_rope_positions(config: Qwen3Config) -> Iterator[None]:
    """Build a drafter with :func:`dflash_rope_max_position` RoPE positions.

    ``config`` is the draft ``hf_config`` the drafter's decoder layers are
    built from. During construction only ``DFlashQwen3DecoderLayer`` reads its
    ``max_position_embeddings`` (the RoPE cache size), so the fitted value is
    set there while the drafter is built, from the vLLM config the layers see
    (``get_current_vllm_config()``), and restored afterwards.
    """
    configured = config.max_position_embeddings
    fitted = dflash_rope_max_position(get_current_vllm_config(), config)
    if fitted == configured:
        yield
        return
    config.max_position_embeddings = fitted
    try:
        yield
    finally:
        config.max_position_embeddings = configured


class DFlashDraftMixin:
    """Put before the upstream ``DFlashQwen3ForCausalLM`` (sub)class."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        with fitted_rope_positions(speculative_config.draft_model_config.hf_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.model.use_aux_hidden_state and getattr(self, "aux_fc_folded", False):
            # Pipeline stages already applied fc (VLLM_GLM5_PP_FOLD_DRAFT_FC):
            # this is its fp32 output; round once to the model dtype.
            assert hidden_states.shape[-1] == self.model.fc.output_size, (
                hidden_states.shape
            )
            return hidden_states.to(self.model.fc.weight.dtype)
        return super().combine_hidden_states(hidden_states)


class DFlashQwen3ForCausalLM(DFlashDraftMixin, qwen3_dflash.DFlashQwen3ForCausalLM):
    """The upstream DFlash drafter (``DFlashDraftModel``) with the mixin."""
