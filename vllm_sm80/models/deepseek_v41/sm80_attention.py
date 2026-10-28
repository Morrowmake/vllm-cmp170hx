# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 attention on sm_80 (compute capability 8.x).

None of the vendor kernels the upstream layers use (FlashMLA, FlashInfer
sparse MLA, DeepGEMM, the FP8/FP4 conversion instructions) exist on Ampere.
This layer keeps upstream's cache topology -- a sliding-window cache per
layer, compressed KV and index keys owned by the kv-source layers, the shared
top-k and candidate-block buffers -- with every cache holding plain BF16 rows,
and computes the reference model's attention through
``ops/sm80_attn_core.py``: the FP8/FP4 rounding of the reference is emulated
in software, so the cached rows carry the reference's quantize-dequantize
values exactly.

Selection is by the platform: :func:`sm80_select_attn_cls` wraps upstream's
``_select_dsv4_attn_cls`` and returns :class:`DeepseekV41Sm80Attention` on
compute capability 8.x; the DSpark draft layers are built by the same
selector. :func:`sm80_dspark_context_kv` wraps the drafter's context-KV
insertion so its window rows go through the same rounding.

The attention region (KV writes, compressor, indexer, top-k, sparse attention,
output RoPE and quantize-dequantize) runs eagerly under a piecewise CUDA-graph
break and is captured whole in FULL graphs (uniform decode under
FULL_AND_PIECEWISE): it reads only the metadata builders' persistent buffers
and does no host reads, and in a FULL graph the indexer scores the fixed
width ``max_model_len // ratio``. The input projections and ``wo_a`` /
``wo_b`` are in the graph either way.
"""

import dataclasses
import functools
import os
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm_sm80.models.deepseek_v41 import sm80_swa_meta_batch
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4SparseMLABackend,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm_sm80 import capture
from vllm_sm80.kernels.exl3.quant import set_output_dtype
from vllm_sm80.kernels.exl3.step_phase import max_decode_query_len, prefill_step
from vllm_sm80.layers.grouped_linear import GroupedLinearMethod
from vllm_sm80.distributed import ce_ring_all_reduce
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Backends: upstream's metadata builders, plain BF16 rows, compute capability 8
# ---------------------------------------------------------------------------


class Sm80SparseDsv41Backend(DeepseekV4SparseMLABackend):
    """The kv-source compressed cache: upstream's DSv4.1 sparse metadata
    (block table + compressed slot mapping), BF16 rows, any even block size
    (so the kernel blocks are the manager blocks)."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "SM80_SPARSE_DSV41"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(2)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class Sm80IndexKBackend(Sm80SparseDsv41Backend):
    """The index-key cache: same metadata, 128-dim BF16 rows."""

    @staticmethod
    def get_name() -> str:
        return "SM80_INDEX_K_DSV41"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]


class Sm80SWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """Upstream's sliding-window metadata without FlashMLA tile plans.

    With ``VLLM_DEEPSEEK_V41_SWA_META_BATCH`` the builds of one
    ``build_attn_metadata`` call share one launch for every SWA group
    (``sm80_swa_meta_batch``); each build returns what its own build does."""

    def build_tile_scheduler(self, num_decode_tokens: int) -> dict[str, Any]:
        return super().build_tile_scheduler(0)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ):
        if envs.VLLM_DEEPSEEK_V41_SWA_META_BATCH:
            return sm80_swa_meta_batch.build(
                self, common_prefix_len, common_attn_metadata, fast_build, replay_start
            )
        return self.build_per_group(
            common_prefix_len, common_attn_metadata, fast_build, replay_start
        )

    def build_per_group(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ):
        """Upstream's build of this group alone."""
        return super().build(
            common_prefix_len, common_attn_metadata, fast_build, replay_start
        )

    # Capture and drafting builds stay per group (outside the step's batch).
    def build_for_cudagraph_capture(self, common_attn_metadata: Any):
        return self.build_per_group(0, common_attn_metadata)

    def build_for_drafting(self, common_attn_metadata: Any, draft_index: int):
        return self.build_per_group(0, common_attn_metadata, fast_build=True)


class Sm80SWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_name() -> str:
        return "SM80_DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[Sm80SWAMetadataBuilder]:
        return Sm80SWAMetadataBuilder


class Sm80IndexKCache(DeepseekV4IndexerCache):
    """Index keys as BF16 rows of ``index_head_dim`` (256 B per key)."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
            alignment=512,
        )

    def get_attn_backend(self) -> type[Sm80IndexKBackend]:
        return Sm80IndexKBackend


# ---------------------------------------------------------------------------
# Indexer: the reference's modules, no upstream scoring op
# ---------------------------------------------------------------------------


class DeepseekV41Sm80Indexer(DeepseekV4Indexer):
    """Holds the indexer's weights and index-key cache; the scoring and the
    top-k run in ``sm80_attn_core`` from :class:`DeepseekV41Sm80Attention`.

    ``wq_b`` and ``weights_proj`` stay replicated (all heads on every rank,
    as upstream); the reference's TP split of the heads is reproduced
    numerically by the head grouping of the scores. ``wq_b`` and ``wk`` take
    the model's quantization config (both are quantized in the EXL3
    checkpoint); ``weights_proj`` stays an unquantized BF16 linear."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config,
        cache_config,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int,
        prefix: str,
        *,
        owns_k: bool,
        k_cache: DeepseekV4IndexerCache,
        main_head_dim: int,
        candidate_block_buffer: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ):
        nn.Module.__init__(self)
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk_tokens = config.index_topk
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.prefix = prefix
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        context = vllm_config.compilation_config.static_forward_context
        if owns_k:
            self.wk = ReplicatedLinear(
                main_head_dim,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            # Replace the packed-record cache the base layer registered.
            context.pop(k_cache.prefix)
            k_cache = Sm80IndexKCache(
                head_dim=self.head_dim,
                dtype=torch.bfloat16,
                prefix=k_cache.prefix,
                cache_config=cache_config,
                compress_ratio=compress_ratio,
            )
        elif not isinstance(k_cache, Sm80IndexKCache):
            raise NotImplementedError(
                f"index-key cache {k_cache.prefix} is not an sm_80 cache"
            )
        self.k_cache = k_cache


# ---------------------------------------------------------------------------
# The attention layer
# ---------------------------------------------------------------------------


def _rope_params(config, compress_ratio: int, rope_dim: int) -> tuple:
    """``(dim, original_seq_len, base, factor, beta_fast, beta_slow)`` of the
    reference's ``precompute_freqs_cis`` for this layer type: compressed
    layers use ``compress_rope_theta`` with YaRN, sliding-window layers plain
    RoPE at ``rope_theta``."""
    rp = getattr(config, "rope_parameters", None) or getattr(
        config, "rope_scaling", None
    ) or {}
    if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
        rp = rp["compress" if compress_ratio else "main"]
    factor = float(rp.get("factor", 1.0))
    beta_fast = rp.get("beta_fast", 32)
    beta_slow = rp.get("beta_slow", 1)
    if not compress_ratio:
        return (rope_dim, 0, float(config.rope_theta), factor, beta_fast, beta_slow)
    yarn = rp.get("rope_type", "default") != "default"
    original = int(rp.get("original_max_position_embeddings", 0)) if yarn else 0
    return (
        rope_dim,
        original,
        float(config.compress_rope_theta),
        factor,
        beta_fast,
        beta_slow,
    )


class DeepseekV41Sm80Attention(DeepseekV4Attention):
    """DeepSeek-V4.1 attention layer for compute capability 8.x."""

    backend_cls = Sm80SparseDsv41Backend
    swa_backend_cls = Sm80SWABackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _indexer_cls(self, k_cache) -> type[DeepseekV4Indexer]:
        return DeepseekV41Sm80Indexer

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise ValueError(
                "DeepSeek V4.1 on sm_80 stores BF16 KV rows; got kv-cache dtype "
                f"{self.kv_cache_dtype}."
            )
        config = vllm_config.model_config.hf_config
        if self.compressor is not None:
            # The compressor projection takes the model's quantization config
            # (an EXL3 checkpoint quantizes it; upstream builds it unquantized
            # for its fp32 GEMM).
            c = self.compressor
            c.fused_wkv_wgate = MergedColumnParallelLinear(
                c.hidden_size,
                [c.head_dim] * (2 if c.has_gate else 1),
                bias=False,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                disable_tp=True,
                prefix=f"{c.prefix}.fused_wkv_wgate",
            )
            if c.has_gate:
                # Above ratio 1 the reference projects in FP32 for its FP32
                # softmax pooling (wkv and wgate are FP32 linears applied to
                # x.float(), exactly the BF16 x): the EXL3 or BF16 linear
                # returns its FP32 result unrounded. Ratio 1 stays BF16, as
                # there.
                set_output_dtype(c.fused_wkv_wgate, torch.float32)
        wo_a_method = self.wo_a.quant_method
        if getattr(wo_a_method, "apply_grouped", None) is None and not isinstance(
            wo_a_method, UnquantizedLinearMethod
        ):
            # A quantized dense wo_a (an FP8 one in a hybrid checkpoint) runs
            # its groups as separate GEMMs of its own method (no batched
            # kernel on sm_80; _o_proj calls apply_grouped). With
            # VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM the MXFP8 wo_a runs as
            # one grouped Triton GEMM instead (both arms kept for its dev
            # A/B switch).
            ab_gemm = os.getenv("VLLM_DEV_AB_SWITCH") == "dsv41_wo_a_grouped_gemm"
            if envs.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM or ab_gemm:
                from vllm_sm80.models.deepseek_v41.ops.sm80_wo_a_grouped import (
                    WoAGroupedGemmMethod,
                )

                WoAGroupedGemmMethod.adopt(
                    self.wo_a,
                    self.n_local_groups,
                    gemm=envs.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM,
                    keep_groups=ab_gemm,
                )
            else:
                GroupedLinearMethod.adopt(self.wo_a, self.n_local_groups)
        tp_size = self.n_heads // self.n_local_heads
        index_n_heads = getattr(config, "index_n_heads", 0)
        cand_source = self.candidate_source_layer
        self.sm80_cfg = core.Sm80AttnConfig(
            n_local_heads=self.n_local_heads,
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            is_kv_source=self.is_kv_source,
            is_index_source=self.is_index_source,
            is_candidate_source=self.is_index_source and self.layer_id == cand_source,
            uses_candidates=self.is_index_source and 0 <= cand_source < self.layer_id,
            index_n_heads=index_n_heads,
            index_head_dim=getattr(config, "index_head_dim", 0),
            index_topk=getattr(config, "index_topk", 0),
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
            tp_size=tp_size,
            eps=self.eps,
        )
        if self.is_index_source and index_n_heads % tp_size:
            raise ValueError(
                f"{index_n_heads} index heads do not split over TP {tp_size}"
            )
        self._sm80_fused_attn = self._sm80_fused_attn_gate(vllm_config)
        self._sm80_draft_prologue_eligible = self._sm80_draft_prologue_gate()
        self._sm80_draft_prologue = self._sm80_draft_prologue_eligible and bool(
            envs.VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED
        )
        self._sm80_rope_params = _rope_params(
            config, self.compress_ratio, self.rope_head_dim
        )
        # Built here, eagerly, on the device: a first use inside a graph
        # capture would allocate it from that graph's private pool.
        self._sm80_freqs = core.rope_freqs(
            *self._sm80_rope_params, device=current_platform.device_type
        )
        self._sm80_chunk_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        # Eligibility is independent of the active arm; in-boot capture must
        # be able to select either arm on both target and drafter modules.
        from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline

        self._sm80_pipeline_max_rows = sm80_attn_pipeline.MAX_ROWS
        self._sm80_max_decode_query_len = max_decode_query_len(vllm_config)
        self._sm80_is_draft_layer = (
            self.layer_id >= vllm_config.model_config.hf_text_config.num_hidden_layers
        )
        self._sm80_pipeline_owner = (id(vllm_config), self._sm80_is_draft_layer)
        self._sm80_pipeline_execution_stream = None
        self._sm80_attn_pipeline_enabled = bool(envs.VLLM_DEEPSEEK_V41_ATTN_PIPELINE)
        self._sm80_pipeline_reserved = self._sm80_attn_pipeline_enabled or (
            os.getenv("VLLM_DEV_AB_SWITCH") == "dsv41_attn_pipeline"
        )
        self._sm80_attn_pipeline_eligible = self._sm80_pipeline_gate()
        if self._sm80_pipeline_reserved and self._sm80_attn_pipeline_eligible:
            from vllm_sm80.models.deepseek_v41.ops.sm80_attn_pipeline import PIPELINE_WARMUP

            PIPELINE_WARMUP.register_warmup(self)
        if self.indexer is not None and vllm_config.kernel_config.enable_jit_warmup:
            from vllm_sm80.models.deepseek_v41.ops.sm80_index_scores import (
                INDEX_SCORES_WARMUP,
            )

            INDEX_SCORES_WARMUP.register_warmup(
                self.indexer.k_cache,
                self.sm80_cfg.index_n_heads,
                self.sm80_cfg.index_head_dim,
                self.sm80_cfg.index_heads_per_rank,
            )
        self._wo_b_round_once = wo_b_round_once_switch()

    # -- pieces -------------------------------------------------------------

    def _sm80_draft_prologue_gate(self) -> bool:
        """Whether this layer's DSpark context-KV prologue may run as one
        kernel (``VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED``, taken by
        :meth:`sm80_store_context_kv`): the layer has the configuration the
        kernel was validated for."""
        from vllm_sm80.models.deepseek_v41.ops import sm80_draft_prologue

        reason = sm80_draft_prologue.supports(
            self.head_dim,
            self.rope_head_dim,
            self.kv_cache_torch_dtype,
            current_platform.get_device_capability(),
        )
        if reason is not None and envs.VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED:
            logger.warning_once(
                "DeepSeek-V4.1 sm_80: VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED=1 but "
                "the draft context-KV prologue stays unfused: %s", reason,
                scope="global",
            )
        return reason is None

    def _sm80_fused_attn_gate(self, vllm_config: VllmConfig) -> bool:
        """Whether this layer may take the fused attention core
        (``VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE``): the switch is on and the
        layer has the configuration the fused kernels were validated for.
        Which calls take it is :meth:`_sm80_fused_attn_call`."""
        if not envs.VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE:
            return False
        from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused

        reason = sm80_attn_fused.supports(
            self.n_local_heads,
            self.head_dim,
            self.rope_head_dim,
            self.n_local_groups,
            self.window_size,
            self.sm80_cfg.index_topk,
            self.compress_ratio,
            self.kv_cache_torch_dtype,
            current_platform.get_device_capability(),
        )
        if reason is not None:
            logger.warning_once(
                "DeepSeek-V4.1 sm_80: VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE=1 but the "
                "attention core stays unfused: %s", reason, scope="global",
            )
            return False
        self._sm80_fused_max_rows = sm80_attn_fused.MAX_ROWS
        self._sm80_max_decode_query_len = max_decode_query_len(vllm_config)
        # The speculative drafter's layers are numbered after the target's.
        n_target = vllm_config.model_config.hf_text_config.num_hidden_layers
        self._sm80_is_draft_layer = self.layer_id >= n_target
        logger.info_once(
            "DeepSeek-V4.1 sm_80: the attention core of decode, verify and draft "
            "steps of up to %d rows runs fused (VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE=1)",
            self._sm80_fused_max_rows, scope="global",
        )
        return True

    def _sm80_fused_attn_call(self, rows: int) -> bool:
        """A call of ``rows`` (padded) rows takes the fused core when the layer
        may and the rows are within the bound, at a drafter layer (its calls
        are draft blocks) or in a step the scheduler did not make a prefill
        step (decode or speculative verify)."""
        if not self._sm80_fused_attn or rows > self._sm80_fused_max_rows:
            return False
        return self._sm80_is_draft_layer or not prefill_step(
            self._sm80_max_decode_query_len
        )

    def _sm80_pipeline_gate(self) -> bool:
        from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused, sm80_attn_pipeline

        reason = sm80_attn_fused.supports(self.n_local_heads, self.head_dim,
            self.rope_head_dim, self.n_local_groups, self.window_size,
            self.sm80_cfg.index_topk, self.compress_ratio, self.kv_cache_torch_dtype,
            current_platform.get_device_capability())
        reason = reason or sm80_attn_pipeline.supports(self.sm80_cfg,
            self._sm80_pipeline_max_rows, self.max_model_len, self.q_lora_rank,
            min(self._sm80_chunk_bytes, core.DEFAULT_SCORE_CHUNK_BYTES))
        if reason is not None and self._sm80_pipeline_reserved:
            logger.warning_once("DeepSeek-V4.1 attention pipeline disabled: %s",
                                reason, scope="global")
        return reason is None

    def _sm80_pipeline_workspace(self, device):
        from vllm_sm80.models.deepseek_v41.ops.sm80_attn_pipeline import workspace

        # A runner/lane executes serially. Its temporary profiling/capture
        # streams join the execution stream and are not independent runners.
        # Bind that execution owner before graph profiling, not one scratch
        # allocation for every short-lived capture stream. Separate runners,
        # target/drafter owners and WorkspaceManager ubatch/lanes stay disjoint.
        if self._sm80_pipeline_execution_stream is None:
            self._sm80_pipeline_execution_stream = torch.cuda.current_stream(device).cuda_stream
        return workspace(self._sm80_pipeline_max_rows, self.max_model_len,
                         self.q_lora_rank, device, self._sm80_pipeline_owner,
                         execution_stream=self._sm80_pipeline_execution_stream)

    def _sm80_pipeline_call(self, qr_kv, kv_score, index_weights, batch, caches):
        if not (self._sm80_attn_pipeline_enabled and self._sm80_attn_pipeline_eligible):
            return False
        if qr_kv.shape[0] > self._sm80_pipeline_max_rows:
            return False
        if not self._sm80_is_draft_layer and prefill_step(self._sm80_max_decode_query_len):
            return False
        reason = None
        if qr_kv.dtype != torch.bfloat16 or not qr_kv.is_cuda:
            reason = "queries must be CUDA BF16"
        elif batch.max_seq_len > self.max_model_len:
            reason = "context exceeds reserved workspace"
        elif batch.swa_indices.shape[1] > self.window_size:
            reason = "window exceeds supported width"
        elif self.is_kv_source and (kv_score is None or
                kv_score.dtype != torch.float32 or kv_score.stride(1) != 1):
            reason = "compressor projection must be contiguous FP32 rows"
        elif self.is_index_source and (index_weights is None or
                index_weights.dtype != torch.bfloat16 or index_weights.stride(1) != 1):
            reason = "index weights must have contiguous heads"
        for cache in (caches.swa, caches.comp, caches.idxk):
            if cache is not None and (cache.dtype != torch.bfloat16 or cache.stride(2) != 1):
                reason = "cache rows must be BF16 with contiguous channels"
        if caches.ring is not None and (caches.ring.dtype != torch.float32 or
                                       caches.ring.stride(2) != 1):
            reason = "ring rows must be contiguous FP32"
        if reason is not None:
            logger.warning_once("DeepSeek-V4.1 attention pipeline call stays unfused: %s",
                                reason, scope="global")
        return reason is None

    def _sm80_weights(self, device: torch.device) -> core.Sm80AttnWeights:
        freqs = self._sm80_freqs.to(device)
        w = core.Sm80AttnWeights(
            q_norm=self.q_norm.weight,
            kv_norm=self.kv_norm.weight,
            attn_sink=self.attn_sink,
            freqs=freqs,
            wq_b=self.wq_b,
        )
        if self.compressor is not None:
            w.comp_norm = self.compressor.norm.weight
        indexer = self.indexer
        if indexer is not None:
            w.index_wq_b = _first(indexer.wq_b)
            if indexer.owns_k:
                w.wk = _first(indexer.wk)
                w.k_norm = indexer.k_norm.weight
        return w

    def _sm80_caches(self) -> core.Sm80AttnCaches:
        assert self.topk_indices_buffer is not None
        return core.Sm80AttnCaches(
            swa=self.swa_cache_layer.kv_cache,
            topk=self.topk_indices_buffer,
            comp=self._compressed_kv_cache() if self.compress_ratio else None,
            idxk=self.indexer.k_cache.kv_cache if self.indexer is not None else None,
            ring=(
                self.compressor.state_cache.kv_cache
                if self.compressor is not None
                and self.compressor.state_cache is not None
                else None
            ),
            candidates=self.candidate_block_buffer,
        )

    def _sm80_batch(self, attn_metadata: dict[str, Any], positions: torch.Tensor):
        """Gather this step's token metadata from the upstream builders."""
        swa = cast(Any, attn_metadata[self.swa_cache_layer.prefix])
        nd, npf = swa.num_decode_tokens, swa.num_prefill_tokens
        T = nd + npf
        parts = []
        if nd:
            parts.append(swa.decode_swa_indices.reshape(nd, -1))
        if npf:
            parts.append(swa.prefill_swa_indices.reshape(npf, -1))
        if len(parts) == 1:
            # Decode-only or prefill-only: the builder's buffer itself.
            swa_indices = parts[0]
        else:
            width = max([p.shape[1] for p in parts] + [1])
            swa_indices = torch.cat(
                [
                    torch.nn.functional.pad(p, (0, width - p.shape[1]), value=-1)
                    for p in parts
                ]
            ) if parts else torch.empty(
                0, 1, dtype=torch.int32, device=positions.device
            )
        batch = core.Sm80AttnBatch(
            positions=positions[:T],
            token_to_req=swa.token_to_req_indices[:T],
            query_start_loc=swa.query_start_loc,
            swa_slots=swa.slot_mapping[:T],
            swa_indices=swa_indices.contiguous(),
        )
        if self.compress_ratio:
            comp = cast(Any, attn_metadata[self.compressed_cache_prefix])
            batch.comp_block_table = comp.block_table
            # A FULL graph replays its capture's score width for every step.
            batch.max_seq_len = (
                self.max_model_len if _in_full_graph() else comp.max_seq_len
            )
            if self.is_kv_source:
                batch.comp_slots = comp.slot_mapping[:T]
        if self.indexer is not None:
            idxk = cast(Any, attn_metadata[self.indexer.k_cache.prefix])
            batch.idxk_block_table = idxk.block_table
            batch.idxk_slots = idxk.slot_mapping[:T]
        if self.compressor is not None and self.compressor.state_cache is not None:
            ring = cast(Any, attn_metadata[self.compressor.state_cache.prefix])
            batch.ring_slots = ring.slot_mapping[:T]
        return batch

    def _sm80_projections(self, hidden_states: torch.Tensor):
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = None
        if self.compressor is not None:
            kv_score = self.compressor.fused_wkv_wgate(hidden_states).float()
        index_weights = None
        if self.indexer is not None:
            index_weights, _ = self.indexer.weights_proj(hidden_states)
        return qr_kv, kv_score, index_weights

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._o_proj(self.attend(positions, hidden_states), positions)

    def attend(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The input projections and the attention region of :meth:`forward`:
        the ``wo_a`` input (see :meth:`forward_mqa`), one row per token."""
        attn_out = torch.empty(
            (hidden_states.shape[0], self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, index_weights = self._sm80_projections(hidden_states)
        self._sm80_attention(qr_kv, kv_score, index_weights, positions, attn_out)
        return attn_out

    @eager_break_during_capture
    def _sm80_attention(
        self,
        qr_kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profile run: no caches yet. Reserve what the indexer's score
            # chunk (scores, candidate mask, sort) can take at its bound.
            out.zero_()
            if self._sm80_pipeline_reserved and self._sm80_attn_pipeline_eligible:
                self._sm80_pipeline_workspace(out.device)
            if self.is_index_source and out.is_cuda:
                torch.empty(
                    3 * self._sm80_chunk_bytes, dtype=torch.uint8, device=out.device
                )
            return
        batch = core.pad_batch(
            self._sm80_batch(attn_metadata, positions), qr_kv.shape[0]
        )
        weights = self._sm80_weights(out.device)
        caches = self._sm80_caches()
        pipeline = self._sm80_pipeline_call(qr_kv, kv_score, index_weights, batch, caches)
        workspace = self._sm80_pipeline_workspace(out.device) if pipeline else None
        fused = not pipeline and self._sm80_fused_attn_call(qr_kv.shape[0])
        record = None
        if capture.active:
            record = capture.begin(
                "attn", "dsv41_attn_layer", self.prefix, qr_kv.shape[0]
            )
        if record is not None:
            weights = _capture_attention_inputs(record, batch, caches, weights, qr_kv)
        q = core.prepare_layer(
            self.sm80_cfg,
            weights,
            caches,
            batch,
            qr_kv,
            kv_score,
            index_weights,
            self._sm80_chunk_bytes,
            fused=fused, pipeline=pipeline, workspace=workspace,
        )
        self._sm80_attend(batch, q, out, fused=fused, pipeline=pipeline, workspace=workspace)
        if record is not None:
            _capture_attention_outputs(record, self.sm80_cfg, batch, caches, out)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse attention of the post-RoPE ``q`` over this step's window
        rows and selected compressed rows, already in the caches (``kv`` is
        not read). ``output`` receives the ``wo_a`` input: inverse RoPE and
        FP8 qdq applied, as :meth:`_o_proj` expects."""
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            output.zero_()
            return
        batch = core.pad_batch(self._sm80_batch(attn_metadata, positions), q.shape[0])
        self._sm80_attend(batch, q, output)

    def _sm80_attend(
        self, batch, q: torch.Tensor, out: torch.Tensor, fused: bool = False,
        pipeline: bool = False, workspace=None,
    ) -> None:
        core.attend_layer(
            self.sm80_cfg, self._sm80_weights(out.device), self._sm80_caches(),
            batch, q, out, fused=fused, pipeline=pipeline, workspace=workspace,
        )

    def forward_kv(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        qr_kv, kv_score, _ = self._sm80_projections(hidden_states)
        self._sm80_store_kv(qr_kv, kv_score, positions)

    @eager_break_during_capture
    def _sm80_store_kv(
        self,
        qr_kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        batch = self._sm80_batch(attn_metadata, positions)
        T = batch.num_tokens
        kv = core.rms_norm(qr_kv[:T, self.q_lora_rank :], self.kv_norm.weight, self.eps)
        core.store_kv(
            self.sm80_cfg,
            self._sm80_weights(qr_kv.device),
            self._sm80_caches(),
            batch,
            kv,
            None if kv_score is None else kv_score[:T],
        )

    @eager_break_during_capture
    def sm80_store_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        """DSpark context KV: ``kv_norm(wkv(main_x))`` RoPE'd and FP8 qdq'd
        into this layer's window cache at ``slot_mapping`` (the reference
        ``DSparkAttention`` main KV). With
        ``VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED`` everything after the
        projection is one kernel with the same bytes
        (``ops/sm80_draft_prologue.py``)."""
        qr_kv = self._fused_wqa_wkv_gemm(main_x)
        if slot_mapping is None:
            return
        if self._sm80_draft_prologue:
            from vllm_sm80.models.deepseek_v41.ops import sm80_draft_prologue

            sm80_draft_prologue.store_context_kv(
                qr_kv[:, self.q_lora_rank :], self.kv_norm.weight, self.eps,
                positions, self._sm80_freqs, self.swa_cache_layer.kv_cache,
                slot_mapping,
            )
            return
        n = min(slot_mapping.shape[0], qr_kv.shape[0])
        kv = core.rms_norm(qr_kv[:n, self.q_lora_rank :], self.kv_norm.weight, self.eps)
        freqs = self._sm80_weights(kv.device).freqs
        rows = core.window_kv(kv, positions[:n], freqs)
        core.store_rows(self.swa_cache_layer.kv_cache, slot_mapping[:n], rows)

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``wo_a`` per group (the reference's einsum over its block-diagonal
        weight) on the already inverse-RoPE'd, FP8-qdq'd output, then
        ``wo_b``; with TP the ``wo_b`` partials are summed in fp32 and rounded
        once, as the reference's RowParallelLinear does (in BF16 on a
        transport that does exactly that, with
        VLLM_SM80_ALLREDUCE_ROUND_ONCE; see
        :func:`wo_b_all_reduce`)."""
        if not self.reduces_wo_b():
            return self._wo_b_proj(self._wo_a_proj(attn_out))
        return self.reduce_wo_b(self.wo_b_partial(attn_out))

    def _wo_a_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        T = attn_out.shape[0]
        o = attn_out.view(T, self.n_local_groups, -1)
        wo_a = self.wo_a
        apply_grouped = getattr(wo_a.quant_method, "apply_grouped", None)
        if apply_grouped is not None:
            return apply_grouped(wo_a, o)
        w = wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        return torch.einsum("tgd,grd->tgr", o, w).flatten(1)

    def reduces_wo_b(self) -> bool:
        """Whether :meth:`_o_proj` all-reduces this rank's ``wo_b`` partial
        through :meth:`reduce_wo_b` (TP, no fused GEMM + reduce-scatter)."""
        wo_b = self.wo_b
        return self.gemm_rs is None and wo_b.reduce_results and wo_b.tp_size > 1

    def wo_b_partial(self, attn_out: torch.Tensor) -> torch.Tensor:
        """This rank's ``wo_b`` partial of ``attn_out`` (``wo_a`` then
        ``wo_b``, not yet all-reduced); each row depends only on its own."""
        wo_b = self.wo_b
        return wo_b.quant_method.apply(wo_b, self._wo_a_proj(attn_out), None)

    def reduce_wo_b(self, partial: torch.Tensor) -> torch.Tensor:
        """The TP all-reduce of :meth:`wo_b_partial`'s output, with the mHC
        coefficients the decoder layer deferred beside it
        (``ops/sm80_mhc_side.py``)."""
        return sm80_mhc_side.beside(
            wo_b_all_reduce, partial, self._wo_b_round_once
        )


# Backends the communicator tries before the custom all-reduce. When one is
# enabled the custom all-reduce's own predicate no longer says which kernel
# reduces a message, so the BF16 switch stays inert.
_PREEMPTING_BACKENDS = (
    "fi_ar_comm",
    "qr_comm",
    "fi_pcie_ipc_ar_comm",
    "aiter_ar_comm",
)


def _window_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Rows of window slots ``slots`` (-1: row of slot 0, masked by the
    slots themselves)."""
    slots = slots.clamp_min(0).to(torch.int64)
    return cache[slots // cache.shape[1], slots % cache.shape[1]]


def _capture_attention_inputs(record, batch, caches, weights, qr_kv: torch.Tensor):
    """Record the attention core's inputs before the layer writes its caches
    (the window rows the step attends to, as they were), and return weights
    whose ``wq_b`` records its input (the normalised query latent) and its
    output (the query before RoPE)."""
    record.put(
        qr_kv=qr_kv,
        positions=batch.positions,
        token_to_req=batch.token_to_req,
        query_start_loc=batch.query_start_loc,
        swa_slots=batch.swa_slots,
        swa_indices=batch.swa_indices,
        swa_rows_before=_window_rows(caches.swa, batch.swa_indices),
        q_norm=weights.q_norm,
        kv_norm=weights.kv_norm,
        attn_sink=weights.attn_sink,
        freqs=weights.freqs,
    )
    wq_b = weights.wq_b

    def recording_wq_b(qr: torch.Tensor) -> torch.Tensor:
        q = wq_b(qr)
        record.put(qr=qr, q=q)
        return q

    return dataclasses.replace(weights, wq_b=recording_wq_b)


def _capture_attention_outputs(record, cfg, batch, caches, out: torch.Tensor) -> None:
    """Record what attention read after the layer's cache writes (window
    rows; at compressed layers the selected positions and their rows) and
    the ``wo_a`` input it wrote."""
    T = batch.num_tokens
    values: dict[str, Any] = dict(
        swa_rows=_window_rows(caches.swa, batch.swa_indices),
        out=out,
        num_tokens=T,
        compress_ratio=int(cfg.compress_ratio),
        n_groups=int(cfg.n_local_groups),
        scale=float(cfg.softmax_scale),
        eps=float(cfg.eps),
    )
    if cfg.compress_ratio:
        local = caches.topk[:T]
        req = batch.token_to_req.to(torch.int64)[:, None].expand_as(local)
        values.update(
            comp_local=local,
            comp_rows=core.gather_compressed_rows(
                caches.comp, batch.comp_block_table, req, local.clamp_min(0)
            ),
        )
    record.put(**values).done()


def _preempting_backends(comm: Any) -> list[str]:
    from vllm.distributed.device_communicators.pynccl_allocator import (
        is_symmetric_memory_enabled,
    )

    names = [
        name
        for name in _PREEMPTING_BACKENDS
        if getattr(comm, name, None) is not None
        and not getattr(getattr(comm, name), "disabled", False)
    ]
    pynccl = getattr(comm, "pynccl_comm", None)
    if pynccl is not None and is_symmetric_memory_enabled():
        names.append("nccl_symm_mem")
    return names


def custom_all_reduce_takes(t: torch.Tensor) -> bool:
    """Whether the TP group's custom all-reduce reduces ``t``: the host-staged
    path (dispatched first by the fork communicator) does not take it, no
    backend dispatched before the custom all-reduce is enabled, and the
    custom all-reduce's own ``should_custom_ar`` accepts it."""
    comm = get_tp_group().device_communicator
    host = getattr(comm, "hostshm_comm", None)
    if host is not None and host.should_host_ar(t):
        return False
    ca = getattr(comm, "ca_comm", None)
    if ca is None or ca.disabled or _preempting_backends(comm):
        return False
    return bool(ca.should_custom_ar(t))


def custom_all_reduce_pieces(t: torch.Tensor) -> list[torch.Tensor] | None:
    """The pieces in which the TP group's custom all-reduce reduces ``t`` when
    it does not take ``t`` whole: flat views of ``t`` in storage order, each
    the largest whole number of 16-byte units below its buffer size (the last
    one shorter), every one accepted by its own ``should_custom_ar``, with the
    host-staged path not taking ``t`` and no backend dispatched before the
    custom all-reduce enabled. None when it takes ``t`` whole (see
    :func:`custom_all_reduce_takes`), declines a piece, ``t`` is not
    contiguous, or a CUDA graph is being captured or warmed up: the pieces go
    through the eager path's copy into the registered buffer."""
    if t.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    comm = get_tp_group().device_communicator
    host = getattr(comm, "hostshm_comm", None)
    if host is not None and host.should_host_ar(t):
        return None
    ca = getattr(comm, "ca_comm", None)
    if ca is None or ca.disabled or _preempting_backends(comm):
        return None
    if getattr(ca, "_IS_CAPTURING", False) or not t.is_contiguous():
        return None
    if ca.should_custom_ar(t):
        return None
    numel = (ca.max_size - 1) // 16 * (16 // t.element_size())
    if numel <= 0:
        return None
    pieces = list(t.view(-1).split(numel))
    if not all(ca.should_custom_ar(p) for p in pieces):
        return None
    return pieces


def large_round_once_all_reduce(t: torch.Tensor) -> torch.Tensor | None:
    """The round-once all-reduce of a message the TP group's fast paths do not
    take whole (the host-staged path and the custom all-reduce; prefill-sized
    messages): on the copy-engine ring's owner sum when it is built
    (VLLM_SM80_ALLREDUCE; the FP32 sum in rank order rounded once, no SMs held
    during transfer), else through the custom all-reduce in pieces
    (:func:`custom_all_reduce_pieces`, the FP32 sum in a fixed order rounded
    once). None when neither applies: the caller keeps its own path."""
    ring = _ce_ring_for(t)
    if ring is not None:
        return ring.round_once(t)
    pieces = custom_all_reduce_pieces(t)
    return None if pieces is None else all_reduce_in_pieces(t, pieces)


def round_once_all_reduce(t: torch.Tensor) -> torch.Tensor:
    """The TP sum of the BF16 partials ``t`` with one rounding per element
    (VLLM_SM80_ALLREDUCE_ROUND_ONCE; written for any consumer, today the
    ``wo_b`` partial): sent as ``t``'s dtype where the transport accumulates
    in FP32 in a fixed order and rounds once (the custom all-reduce taking
    ``t`` whole, else :func:`large_round_once_all_reduce`), else all-reduced
    in FP32 and rounded once (also where the host-staged path takes ``t``:
    the FP32 message may exceed its size limit and go elsewhere)."""
    if custom_all_reduce_takes(t):
        return tensor_model_parallel_all_reduce(t)
    out = large_round_once_all_reduce(t)
    if out is not None:
        return out
    return prefill_all_reduce(t.float()).to(t.dtype)


def wo_b_round_once_switch() -> bool:
    """VLLM_SM80_ALLREDUCE_ROUND_ONCE, announced once at start when on."""
    on = bool(envs.VLLM_SM80_ALLREDUCE_ROUND_ONCE)
    if on:
        logger.info_once(
            "DeepSeek-V4.1 sm_80: wo_b partials are summed with one rounding "
            "per element, in BF16 where the transport rounds once, else in "
            "FP32 (VLLM_SM80_ALLREDUCE_ROUND_ONCE=1; set 0 to all-reduce "
            "every wo_b partial in FP32)",
            scope="global",
        )
    return on


def wo_b_all_reduce(partial: torch.Tensor, round_once: bool) -> torch.Tensor:
    """Sum the ``wo_b`` partials of the TP ranks, rounding once to BF16: with
    ``round_once`` (VLLM_SM80_ALLREDUCE_ROUND_ONCE) through
    :func:`round_once_all_reduce`, else all-reduced in FP32."""
    if round_once:
        return round_once_all_reduce(partial)
    return prefill_all_reduce(partial.float()).to(partial.dtype)


def prefill_all_reduce(t: torch.Tensor) -> torch.Tensor:
    """``t``'s TP all-reduce: through the copy-engine ring
    (VLLM_SM80_ALLREDUCE, ``ce_ring_all_reduce.py``) when it
    takes ``t`` and neither the host-staged nor the custom all-reduce does
    (eager prefill-sized messages); otherwise
    ``tensor_model_parallel_all_reduce``."""
    ring = _ce_ring_for(t)
    if ring is not None:
        return ring.all_reduce(t)
    return tensor_model_parallel_all_reduce(t)


def _ce_ring_for(t: torch.Tensor):
    """The copy-engine ring when it is built and takes ``t`` and neither the
    host-staged nor the custom all-reduce does; else None."""
    ring = ce_ring_all_reduce.ce_ring()
    if ring is None or not ring.takes(t):
        return None
    comm = get_tp_group().device_communicator
    host = getattr(comm, "hostshm_comm", None)
    if (host is not None and host.should_host_ar(t)) or custom_all_reduce_takes(t):
        return None
    return ring


def all_reduce_in_pieces(t: torch.Tensor, pieces: list[torch.Tensor]) -> torch.Tensor:
    """All-reduce ``t`` through the TP group's custom all-reduce, one call per
    piece of :func:`custom_all_reduce_pieces`, into one output like ``t``."""
    ca = get_tp_group().device_communicator.ca_comm
    out = torch.empty_like(t)
    for piece, dst in zip(pieces, out.view(-1).split(pieces[0].numel())):
        ca.all_reduce(piece, out=dst)
    return out

def _in_full_graph() -> bool:
    """Whether this forward runs (or is captured) as a FULL CUDA graph."""
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
    )


def _first(linear: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Call a ``(output, bias)``-returning linear for its output."""

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear(x)[0]

    return call


# ---------------------------------------------------------------------------
# Hooks into upstream (one decorator each)
# ---------------------------------------------------------------------------


def _is_sm80() -> bool:
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 8


def sm80_select_attn_cls(
    select: Callable[[VllmConfig], type[DeepseekV4Attention]],
) -> Callable[[VllmConfig], type[DeepseekV4Attention]]:
    """Decorates ``_select_dsv4_attn_cls``: compute capability 8.x always
    takes :class:`DeepseekV41Sm80Attention` (no other backend runs there)."""

    @functools.wraps(select)
    def wrapper(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
        if _is_sm80():
            return DeepseekV41Sm80Attention
        return select(vllm_config)

    return wrapper


def sm80_dspark_context_kv(store: Callable[..., None]) -> Callable[..., None]:
    """Decorates the DSpark drafter's ``precompute_and_store_context_kv``:
    sm_80 draft layers write their context window rows themselves."""

    @functools.wraps(store)
    def wrapper(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        layers = list(self.layers)
        if not layers or not isinstance(layers[0].attn, DeepseekV41Sm80Attention):
            return store(self, main_x, context_positions, context_slot_mappings)
        for i, layer in enumerate(layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            layer.attn.sm80_store_context_kv(main_x, context_positions, slot_mapping)
        return None

    return wrapper
