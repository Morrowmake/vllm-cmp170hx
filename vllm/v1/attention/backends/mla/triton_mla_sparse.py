# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA attention backend backed by a pure-Triton MQA kernel.

Serves DSA-style sparse MLA (DeepSeek V3.2, GLM-5.3-Flash) on any CUDA GPU
with compute capability >= 8.0, i.e. also on Ampere where FlashMLA-sparse,
FA3 and FlashInfer-sparse are unavailable. Dense-MHA prefill and the
metadata plumbing come from ``SparseMLACommon``; only the top-k MQA path
(decode and prefill-as-MQA) runs the Triton kernel in
``vllm/v1/attention/ops/triton_mla_sparse.py``.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        # The kernel gathers individual cache rows through a flat row view of
        # the paged cache, so it has no block-size requirement of its own.
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return TritonMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512: NoPE latent (GLM-5.3-Flash); 576: 512 NoPE + 64 RoPE (DeepSeek).
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return "Triton MLA Sparse currently supports only FP16/BF16 KV cache"

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            if vllm_config.parallel_config.decode_context_parallel_size > 1:
                return "Triton MLA Sparse does not support DCP for now"

            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "Triton MLA Sparse requires model with index_topk"
        return None


@dataclass
class TritonMLASparseMetadata(SparseMLACommonMetadata):
    pass


class TritonMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[TritonMLASparseMetadata]
):
    metadata_cls = TritonMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        self._init_reorder_batch_threshold(threshold, supports_spec_as_decode=True)


class TritonMLASparseImpl(SparseMLACommonImpl[TritonMLASparseMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "TritonMLASparseImpl currently supports only FP16/BF16 KV cache."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        assert head_size == self.kv_lora_rank + self.qk_rope_head_dim, (
            f"head_size ({head_size}) must equal kv_lora_rank + qk_rope_head_dim "
            f"({self.kv_lora_rank} + {self.qk_rope_head_dim})"
        )
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The shared MLA layer hands over (ql_nope, q_pe); the kernel wants the
        # concatenated [tokens, heads, kv_lora_rank + qk_rope_head_dim] query.
        # For NoPE models q_pe is zero-width, so no concat is needed.
        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = q_nope if q_pe.shape[-1] == 0 else torch.cat((q_nope, q_pe), dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            num_decode_tokens = attn_metadata.num_decode_tokens
            outputs = []
            if num_decode_tokens:
                physical_topk = index_group.convert_decode_logical_to_physical_topk(
                    self.index_group_index,
                    topk_indices[:num_decode_tokens],
                    attn_metadata,
                    return_valid_counts=False,
                )
                outputs.append(
                    self._run_mqa_kernel(
                        q[:num_decode_tokens],
                        index_group.physical_kv_cache(self.index_group_index).view(
                            kv_c_and_k_pe_cache.dtype
                        ),
                        physical_topk,
                        attn_metadata.block_size,
                    )[0]
                )
            if num_decode_tokens < num_actual_toks:
                cache = index_group.cache(self.index_group_index)
                if num_decode_tokens == 0 and cache.all_context_pages_resident:
                    physical_topk = index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices,
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=False,
                    )
                    prefill_cache = index_group.physical_kv_cache(
                        self.index_group_index
                    ).view(kv_c_and_k_pe_cache.dtype)
                else:
                    prefill_cache, block_table, req_ids = (
                        index_group.stage_prefill_rows(
                            self.index_group_index,
                            kv_c_and_k_pe_cache,
                            attn_metadata,
                        )
                    )
                    physical_topk = triton_convert_req_index_to_global_index(
                        req_ids,
                        block_table,
                        topk_indices[num_decode_tokens:],
                        BLOCK_SIZE=attn_metadata.block_size,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                    )
                outputs.append(
                    self._run_mqa_kernel(
                        q[num_decode_tokens:],
                        prefill_cache,
                        physical_topk,
                        attn_metadata.block_size,
                    )[0]
                )
            return torch.cat(outputs) if len(outputs) > 1 else outputs[0], None

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        # Per-request logical positions -> global cache rows; -1 stays -1.
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )
        out, lse = self._run_mqa_kernel(
            q,
            kv_rows,
            topk_indices,
            attn_metadata.block_size,
            cache_is_flat=True,
        )
        return out, lse if self.need_to_return_lse_for_decode else None

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        block_size: int,
        *,
        cache_is_flat: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_rows = (
            kv_cache if cache_is_flat else flat_kv_row_view(kv_cache, block_size)[0]
        )
        num_tokens = q.shape[0]
        out, _, lse = triton_mla_sparse_fwd(
            q,
            kv_rows.unsqueeze(1),
            topk_indices.view(num_tokens, 1, -1),
            sm_scale=self.scale,
            d_v=self.kv_lora_rank,
            block_dpe=self.qk_rope_head_dim,
        )
        return out, lse
