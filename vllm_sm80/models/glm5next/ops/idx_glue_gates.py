# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host gates of the ``VLLM_AMPERE_DECODE_IDX_GLUE`` folds in the GLM indexer.

The GLM indexer (``common/attention.py``: weights, fwht) and the kpool sparse
indexer (``nvidia/sparse_indexer.py``: glue) ask these before taking a fold.
They import the fold kernels (``idx_glue.py``) only once their part is on, so
with the switch unset no fold module is loaded.
"""

import torch

from vllm.platforms import current_platform


def use_idx_glue_weights(indexer, hidden_states: torch.Tensor) -> bool:
    """Host gate for the fp32-weights fold (VLLM_AMPERE_DECODE_IDX_GLUE).

    Only where the merged wk+weights GEMM itself runs on the thin GEMM, so
    the k columns stay bitwise what that GEMM stores today.
    """
    from vllm_sm80.kernels.idx_glue_gate import use_idx_glue

    if not use_idx_glue("weights"):
        return False
    from vllm_sm80.models.glm5next.ops.idx_glue import thin_gemm_dual_supported
    from vllm_sm80.kernels.thin_gemm import use_ampere_thin_gemm

    w = indexer.wk_weights_proj.weight
    return (
        use_ampere_thin_gemm()
        and getattr(indexer.wk_weights_proj, "bias", None) is None
        and thin_gemm_dual_supported(hidden_states, w, indexer.head_dim)
    )


def use_idx_glue_fwht(
    indexer, q: torch.Tensor, weights: torch.Tensor, wscale: float
) -> bool:
    """Host gate for the weight-scale fold into the Hadamard+quant kernel."""
    from vllm_sm80.kernels.idx_glue_gate import use_idx_glue

    if not use_idx_glue("fwht"):
        return False
    from vllm_sm80.models.glm5next.ops.idx_glue import is_pow2_float

    # A power-of-two constant makes (w * q_scale) * wscale exact whatever
    # the constant's representation; the unfused leaf gets the same value.
    return (
        is_pow2_float(wscale)
        and q.is_contiguous()
        and weights.dtype == torch.float32
        and weights.ndim == 2
        and weights.shape[1] == indexer.n_head
        and q.shape[0] == weights.shape[0] * indexer.n_head
    )


def use_glue_pos() -> bool:
    from vllm_sm80.kernels.idx_glue_gate import use_idx_glue

    return use_idx_glue("glue")


def use_glue_tail(
    attn_metadata_narrowed,
    has_decode: bool,
    has_prefill: bool,
    index_kpool: int,
    positions: torch.Tensor | None,
    use_fp4_cache: bool,
    topk_tokens: int,
) -> bool:
    """True when the decode pool expansion can own the whole top-k fill.

    Every condition is host metadata: a decode-only batch, the kpool path with
    positions, the flat (unpadded) decode layout, and a context long enough
    that the short-decode causal fill does not return early.
    """
    if (
        not has_decode
        or has_prefill
        or index_kpool <= 1
        or positions is None
        or positions.numel() == 0
        or use_fp4_cache
    ):
        return False
    from vllm_sm80.kernels.idx_glue_gate import use_idx_glue

    if not use_idx_glue("glue"):
        return False
    decode_metadata = attn_metadata_narrowed.decode
    return (
        decode_metadata is not None
        and not decode_metadata.requires_padding
        and attn_metadata_narrowed.max_seq_len > topk_tokens
        and current_platform.is_cuda_alike()
    )
