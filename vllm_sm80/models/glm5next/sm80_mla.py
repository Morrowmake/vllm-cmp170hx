# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA attention of the GLM-5.3-Flash model on the sm_80 fork.

``vllm_sm80.plugin.register_glm_layers`` registers
:class:`MultiHeadLatentAttentionWrapper` as the out-of-tree replacement of
upstream's pluggable MLA layer of the same name (``vllm/model_executor/layers/
mla.py``) when the fork GLM model module is imported, so every MLA layer of a
GLM process (the backbone's and the MTP drafter's) is this subclass. It is
built by the upstream constructor; then its attention child (upstream
``MLAAttention``) is given :class:`MLAAttention` of this module. Module names,
parameters and construction order stay upstream's.

:class:`MLAAttention` adds the ``VLLM_AMPERE_DECODE_IDX_GLUE`` "cache" fold:
on a decode-only batch the latent cache write moves into the Triton sparse MLA
backend's index-remap launch (``TritonMLASparseImpl.forward_mqa`` in
``vllm_sm80/attention/triton_mla_sparse.py``, which also writes an outstanding
hand-off before its MHA path, ``forward_mha``).
"""

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers import mla
from vllm.model_executor.layers.attention import mla_attention


class MLAAttention(mla_attention.MLAAttention):
    def update_kv_cache(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        attn_metadata: "mla_attention.MLACommonMetadata | None",
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # A HiSparse cache never defers (_idx_glue_defer_kv), so upstream's
        # early return for its dummy batch is upstream's to take.
        if (
            slot_mapping is not None
            and envs.VLLM_AMPERE_DECODE_IDX_GLUE
            and self._idx_glue_defer_kv(kv_cache, attn_metadata, kv_cache_dtype)
        ):
            # The sparse backend writes this cache in its index-remap launch,
            # which runs before anything reads it (see _idx_glue_defer_kv).
            self._idx_glue_flush_kv()
            self.impl._idx_glue_pending_kv = (  # type: ignore[attr-defined]
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
            return
        super().update_kv_cache(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            attn_metadata,
            kv_cache_dtype,
            k_scale,
        )

    def _idx_glue_defer_kv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: "mla_attention.MLACommonMetadata | None",
        kv_cache_dtype: str,
    ) -> bool:
        """VLLM_AMPERE_DECODE_IDX_GLUE "cache": may the latent cache write move
        into the sparse backend's index-remap launch?

        Only for a decode-only batch (the MQA path is then the sole reader of
        the cache, after the remap), a bf16 cache (the write is a copy), no
        HiSparse/PCP/DCP, a backend that consumes the hand-off, and never
        while a breakable piecewise graph is being captured: there the write
        would be recorded while the backend runs eagerly on replay.
        """
        if (
            self.hisparse_cache is not None
            or self.use_pcp
            or kv_cache_dtype != "auto"
            or kv_cache.numel() == 0
            or attn_metadata is None
            or getattr(attn_metadata, "num_prefills", 1) != 0
            or not getattr(self.impl, "supports_idx_glue_kv_defer", False)
            or getattr(self.impl, "dcp_world_size", 1) > 1
        ):
            return False
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        capture = BreakableCUDAGraphCapture.current()
        if capture is not None and capture._capturing:
            from vllm.config import CUDAGraphMode

            if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL:
                return False
        from vllm_sm80.kernels.idx_glue_gate import use_idx_glue

        return use_idx_glue("cache")

    def _idx_glue_flush_kv(self) -> None:
        """Write a deferred latent cache update now (no-op when none)."""
        pending = getattr(self.impl, "_idx_glue_pending_kv", None)
        if pending is None:
            return
        self.impl._idx_glue_pending_kv = None  # type: ignore[attr-defined]
        self.impl.do_kv_cache_update(*pending)  # type: ignore[attr-defined]


class MultiHeadLatentAttentionWrapper(mla.MultiHeadLatentAttentionWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Same object, same place in the module tree: only its class changes.
        assert type(self.mla_attn) is mla_attention.MLAAttention, type(
            self.mla_attn
        )
        self.mla_attn.__class__ = MLAAttention
