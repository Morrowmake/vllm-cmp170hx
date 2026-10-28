# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash sparse attention indexer on the sm_80 fork.

Upstream's ``Indexer.forward`` (``vllm/models/glm5next/common/attention.py``)
and ``sparse_attn_indexer_kpool`` (``vllm/models/glm5next/nvidia/
sparse_indexer.py``) call the functions of this module at the sites they
change, one call each; the rest of both is upstream's code:

- the ``VLLM_AMPERE_DECODE_IDX_GLUE`` folds: the k / fp32-weights pair from
  one thin GEMM (``weights``), the weight scale in the Hadamard+quant kernel
  (``fwht``), the decode positions without their int32 cast and the whole
  top-k buffer fill from the pool expansion (``glue``);
- ``VLLM_THIN_GEMM`` on the pool gate projection (``thin_linear``);
- the Triton fp8 MQA-logits kernels where DeepGEMM is not usable (sm_80), for
  prefill (uncompressed rows) and the paged decode;
- the top-k determinism switches after both top-k selections;
- the fp32 prefill logits freed before the next chunk allocates its own.

The op class itself, registered out of tree in place of upstream's, is in
``sm80_indexer_op.py``. This module imports neither upstream module at
import time: both import it.
"""

import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, is_deep_gemm_supported
from vllm_sm80.kernels.indexer.topk_tiefix_gate import (
    canonical_decode_topk_,
    canonical_topk_,
)
from vllm_sm80.kernels.thin_gemm import thin_linear
from vllm_sm80.models.glm5next.ops.idx_glue_gates import (
    use_glue_pos,
    use_glue_tail,
    use_idx_glue_fwht,
    use_idx_glue_weights,
)


# ---------------------------------------------------------------------------
# Indexer.forward (glm5next/common/attention.py)
# ---------------------------------------------------------------------------


def k_weights(indexer, hidden_states: torch.Tensor):
    """``k`` and the fp32 head-gate ``weights`` of ``Indexer.forward``.

    The gate is computed in fp32; bf16 error can change near-tie pool
    rankings on long-context tasks (the fp32 weight block is cached after
    weights are loaded). With the ``weights`` fold, one thin-GEMM launch
    stores the k columns bf16 (bitwise the merged GEMM's) and the weight
    columns as its fp32 accumulator, instead of recomputing them with a
    separate fp32 sgemm.
    """
    if use_idx_glue_weights(indexer, hidden_states):
        from vllm_sm80.models.glm5next.ops.idx_glue import thin_gemm_dual

        return thin_gemm_dual(
            hidden_states, indexer.wk_weights_proj.weight, indexer.head_dim
        )
    kw, _ = indexer.wk_weights_proj(hidden_states)
    k = kw[:, : indexer.head_dim]
    if indexer._wp_fp32 is None:
        indexer._wp_fp32 = (
            indexer.wk_weights_proj.weight.data[indexer.head_dim :, :]
            .t()
            .contiguous()
            .float()
        )
    weights = torch.mm(hidden_states.float(), indexer._wp_fp32)
    return k, weights


def quant_q(indexer, q: torch.Tensor, weights: torch.Tensor):
    """The Hadamard-rotated fp8 Q ``[T, n_head, head_dim]`` and the weights
    scaled by its per-(token, head) scale and ``softmax_scale / sqrt(n_head)``.
    With the ``fwht`` fold: the same rotation/quant and the same fp32
    ``(w * q_scale) * wscale`` in one launch with more CTAs (q_scale has no
    other consumer)."""
    from vllm.models.glm5next.common import attention as glm_attention
    wscale = indexer.softmax_scale * indexer.n_head**-0.5
    if use_idx_glue_fwht(indexer, q, weights, wscale):
        from vllm_sm80.models.glm5next.ops.idx_glue import fwht128_quant_fp8_wscale

        q_fp8, weights = fwht128_quant_fp8_wscale(q, weights, wscale)
        return q_fp8.view(-1, indexer.n_head, indexer.head_dim), weights
    q_fp8, q_scale = fwht128_quant_fp8(q)
    q_fp8 = q_fp8.view(-1, indexer.n_head, indexer.head_dim)
    q_scale = q_scale.view(-1, indexer.n_head, 1)
    weights = glm_attention._fused_indexer_weight_scale(weights, q_scale, wscale)
    return q_fp8, weights


def gate_score(indexer, hidden_states: torch.Tensor) -> torch.Tensor:
    """kpool's per-token gate score, ``F.linear(hidden_states, gate)``:
    ``thin_linear`` is F.linear unless VLLM_THIN_GEMM is on and the call is
    inside the measured win region. This site never builds a LinearBase, so
    it cannot be reached by the dispatch_unquantized_gemm hook that covers
    every other shape (N=128, K=4096, 11 calls/step, 1.27x vs cuBLAS)."""
    return thin_linear(hidden_states, indexer.index_kpool_compress_gate)


# ---------------------------------------------------------------------------
# sparse_attn_indexer_kpool (glm5next/nvidia/sparse_indexer.py)
# ---------------------------------------------------------------------------


def fill_topk_indices(
    topk_indices: torch.Tensor,
    attn_metadata,
    positions: torch.Tensor | None,
    index_kpool: int,
    use_fp4_cache: bool,
    topk_tokens: int,
) -> bool:
    """Upstream's ``-1`` fill of this step's top-k buffer rows
    (``topk_indices``), unless the ``glue`` fold applies: on a pure-decode
    batch whose rows all go through the pool expansion, one kernel writes
    every buffer row this fill covers (the expanded indices, -1 elsewhere),
    so the fill, the int64/int32 casts, the +1 and the copy after the decode
    top-k are folded into it (:func:`expand_tail`). Returns whether it does.
    """
    fuse_tail = use_glue_tail(
        attn_metadata,
        attn_metadata.num_decodes > 0,
        attn_metadata.num_prefills > 0,
        index_kpool,
        positions,
        use_fp4_cache,
        topk_tokens,
    )
    if not fuse_tail:
        topk_indices[:] = -1
    return fuse_tail


def triton_prefill_logits(chunk, use_fp4_cache: bool, dcp_world_size: int) -> bool:
    """Whether this prefill chunk's logits come from the Triton kernel: CUDA
    without a usable DeepGEMM (e.g. Ampere), and context to score."""
    if chunk.local_total_seq_lens == 0 or is_deep_gemm_supported():
        return False
    assert not use_fp4_cache and dcp_world_size == 1, (
        "The MXFP4 indexer cache and DCP need DeepGEMM"
    )
    return True


def prefill_logits(
    q: torch.Tensor,
    k: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    """(logits, row_starts, row_ends) of a prefill chunk from the Triton
    kernel, which dequantizes the fp8 bytes in-kernel (bf16 MMA, fp32
    accumulate). Its logits are not compressed: row i spans K positions
    ``[ks[i], ke[i])``, which the DCP merge does not take."""
    from vllm_sm80.kernels.attention.triton_mqa_logits import (
        fp8_mqa_logits as triton_fp8_mqa_logits,
    )

    logits = triton_fp8_mqa_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke)
    return logits, cu_seqlen_ks, cu_seqlen_ke


def prefill_topk_tiefix(
    logits: torch.Tensor,
    topk: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
) -> None:
    """The top-k determinism switches after the prefill top-k: boundary ties
    to the lowest indices whatever the chunk shape, rows in ascending order."""
    canonical_topk_(
        logits,
        topk,
        indexer="kpool",
        row_starts=row_starts,
        row_ends=row_ends,
        relative=True,
    )


def decode_positions(positions: torch.Tensor, num_decode_tokens: int):
    """The decode tokens' positions for the kpool tail kernel: with the
    ``glue`` fold as they are (the kernel truncates its position load to
    int32), else cast to int32 as upstream does."""
    if use_glue_pos():
        return positions[:num_decode_tokens]
    return positions[:num_decode_tokens].to(torch.int32)


def paged_mqa_logits(attn_metadata):
    """The paged decode MQA-logits function: DeepGEMM's where it is usable,
    else an adapter with its signature over the Triton paged kernel, which
    reads the uint8 (D + 4)-byte cache with next_n Q rows per request and the
    same (B, next_n) seq_lens layout. It needs no scheduler metadata;
    ``attn_metadata.max_seq_len`` (token-granular, so an upper bound on the
    pool-granular seq_lens) trims the tile grid. Its logits are fp32."""
    if is_deep_gemm_supported():
        return fp8_fp4_paged_mqa_logits
    max_seq_len = attn_metadata.max_seq_len

    def triton_paged_mqa_logits(
        q_fp8_and_scale,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_model_len,
        indices=None,
    ):
        from vllm_sm80.kernels.attention.triton_mqa_logits import (
            fp8_paged_mqa_logits as triton_fp8_paged_mqa_logits,
        )

        q, q_scale = q_fp8_and_scale
        assert q_scale is None, "MXFP4 indexer cache requires DeepGEMM (SM100/SM120)"
        return triton_fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            max_model_len,
            max_seq_len=max_seq_len,
        )

    return triton_paged_mqa_logits


def decode_topk_tiefix(
    logits: torch.Tensor, topk: torch.Tensor, seq_lens: torch.Tensor, next_n: int
) -> None:
    """The top-k determinism switches after the decode top-k (as after the
    prefill top-k)."""
    canonical_decode_topk_(logits, topk, seq_lens, next_n, indexer="kpool")


def expand_tail(
    pool_topk: torch.Tensor,
    positions: torch.Tensor | None,
    topk_indices_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
    index_kpool: int,
) -> torch.Tensor:
    """The ``glue`` fold's pool expansion (see :func:`fill_topk_indices`):
    every top-k buffer row of the step, in one kernel."""
    from vllm_sm80.models.glm5next.ops.idx_glue import expand_pools_into_buffer

    assert positions is not None
    expand_pools_into_buffer(
        pool_topk,
        positions[: pool_topk.shape[0]],
        topk_indices_buffer,
        hidden_states.shape[0],
        index_kpool,
    )
    return topk_indices_buffer
