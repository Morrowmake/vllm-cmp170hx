# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state of the fork's GLM-5.3-Flash classes.

The fork GLM classes (``sm80_model.py``) return this class from
``get_model_state_cls`` wherever vLLM's own resolution would pick
``MambaHybridModelState`` (hybrid or attention-free configs), and that
resolution's choice otherwise (``resolve_glm_model_state_cls``). It adds the
RecoverSSM state whenever the fork's KDA RecoverSSM is enabled
(``vllm_sm80/kernels/kda/recover.py``), next to upstream's
``cache_config.use_kda_recoverssm`` switch. The cache config itself is left
alone: upstream reads ``use_kda_recoverssm`` in other places too (CUDA graph
capture sizes, config checks), which must not change.

It also carries the KDA commit fold's hooks (``VLLM_AMPERE_KDA_FOLD``; None
with the fold off, so the off path makes no call): per step after the
attention metadata is built, on add_request and on remove_request.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.utils import AttentionGroup


class Glm5NextModelState(MambaHybridModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.recoverssm is None:
            from vllm_sm80.kernels.kda.recover import kda_recover_enabled

            if kda_recover_enabled(vllm_config):
                self.recoverssm = RecoverSSMState()
        # KDA commit fold (VLLM_AMPERE_KDA_FOLD): request and step hooks, or
        # None. The fold needs RecoverSSM on, so self.recoverssm is set with it.
        from vllm_sm80.kernels.kda.recover import kda_fold_hooks

        self._kda_fold = kda_fold_hooks(vllm_config, device)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self._kda_fold is not None:
            self._kda_fold.add_request(
                req_index, new_req_data.req_id, new_req_data.num_computed_tokens
            )

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        if self._kda_fold is not None:
            self._kda_fold.remove_request(req_id)

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        if self._kda_fold is not None:
            self._kda_fold.prepare_step(
                attn_metadata, input_batch.idx_mapping, for_capture
            )
        return attn_metadata


class _WithoutModelStateHook:
    """A model as vLLM's model-state resolution sees it, minus the
    ``get_model_state_cls`` hook, so the resolution takes its own path."""

    def __init__(self, model: nn.Module) -> None:
        self.modules = model.modules


def resolve_glm_model_state_cls(vllm_config: VllmConfig, model: nn.Module) -> type:
    """vLLM's model-state choice for ``model`` without the hook, with
    :class:`Glm5NextModelState` in place of ``MambaHybridModelState``."""
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls

    cls = resolve_model_state_cls(vllm_config, _WithoutModelStateHook(model))
    return Glm5NextModelState if cls is MambaHybridModelState else cls
