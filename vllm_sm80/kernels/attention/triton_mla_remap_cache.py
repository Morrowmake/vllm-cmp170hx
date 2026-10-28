# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The "cache" decode fold of ``VLLM_AMPERE_DECODE_IDX_GLUE`` for the Triton
sparse-MLA backend: the MLA latent cache write merged with the sparse top-k
index remap in one launch.

Called from ``vllm_sm80/attention/triton_mla_sparse.py``; whether
the fold is on is decided by the idx-glue gate (``use_idx_glue("cache")``).
"""

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# cache: MLA latent cache write merged with the sparse top-k index remap
# ---------------------------------------------------------------------------
@triton.jit
def _remap_and_cache_mla_kernel(
    req_id_ptr,  # int32 [n_q]
    block_table_ptr,  # int32 [R, max_blocks]
    ti_ptr,  # int32 [n_q, NUM_TOPK]
    out_ptr,  # int32 [n_q, NUM_TOPK]
    kvc_ptr,  # [n_w, D_C] latent
    kpe_ptr,  # [n_w, D_PE] rope part (unused when D_PE == 0)
    slot_ptr,  # int [n_w]
    cache_ptr,  # [num_blocks, CACHE_BLOCK, D_C + D_PE]
    n_q,
    n_w,
    max_num_blocks_per_req,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    out_stride0,
    kvc_stride0,
    kpe_stride0,
    cache_stride0,
    cache_stride1,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE_ROWS: tl.constexpr,
    NUM_TOPK: tl.constexpr,
    REMAP_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    D_C: tl.constexpr,
    D_PE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Programs ``tile < REMAP_TILES``: ``sparse_mla_index_remap_kernel`` with
    no prefill workspace, no valid count, DCP 1 (where its de-interleave is
    the identity). Programs ``tile >= REMAP_TILES``: one ``BLOCK_D`` chunk of
    ``concat_and_cache_mla`` (``kv_cache_dtype="auto"``: a copy)."""
    token = tl.program_id(0)
    tile = tl.program_id(1)
    if tile < REMAP_TILES:
        if token < n_q:
            idx = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            req = tl.load(req_id_ptr + token)
            tok = tl.load(ti_ptr + token * ti_stride0 + idx, mask=idx < NUM_TOPK,
                          other=-1)
            is_invalid_tok = tok < 0
            block_id = tok // BLOCK_SIZE
            inblock_off = tok % BLOCK_SIZE
            valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
            bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
            is_invalid_tok |= ~valid_block
            base = tl.load(bt_ptr, mask=valid_block, other=0)
            out_val = base * BLOCK_STRIDE_ROWS + inblock_off
            out_val = tl.where(is_invalid_tok, -1, out_val)
            tl.store(out_ptr + token * out_stride0 + idx, out_val,
                     mask=idx < NUM_TOPK)
    else:
        if token < n_w:
            slot = tl.load(slot_ptr + token).to(tl.int64)
            if slot >= 0:
                blk = slot // CACHE_BLOCK
                off = slot % CACHE_BLOCK
                dst = cache_ptr + blk * cache_stride0 + off * cache_stride1
                d = (tile - REMAP_TILES) * BLOCK_D + tl.arange(0, BLOCK_D)
                src_c = tl.load(kvc_ptr + token * kvc_stride0 + d, mask=d < D_C)
                tl.store(dst + d, src_c, mask=d < D_C)
                if D_PE > 0:
                    dp = d - D_C
                    pmask = (dp >= 0) & (dp < D_PE)
                    src_p = tl.load(kpe_ptr + token * kpe_stride0 + dp, mask=pmask)
                    tl.store(dst + d, src_p, mask=pmask)


def remap_and_cache_mla(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    block_stride_rows: int,
) -> torch.Tensor:
    """``concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, "auto")`` +
    ``triton_convert_req_index_to_global_index(req_id, block_table,
    token_indices, BLOCK_SIZE=block_size, BLOCK_STRIDE_ROWS=block_stride_rows,
    NUM_TOPK_TOKENS=token_indices.shape[1])`` in one launch; returns the remap.
    """
    assert req_id.dtype == torch.int32 and block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert req_id.shape[0] == token_indices.shape[0]
    num_topk = token_indices.shape[1]
    BLOCK_N = 128
    assert num_topk % BLOCK_N == 0
    assert kv_cache.ndim == 3 and kv_cache.stride(2) == 1
    assert kv_c.dtype == kv_cache.dtype and kv_c.stride(-1) == 1
    n_w = slot_mapping.shape[0]
    d_c = kv_c.shape[-1]
    pe_width = 1
    for d in k_pe.shape[1:]:
        pe_width *= d
    k_pe2 = k_pe.reshape(k_pe.shape[0], pe_width)
    d_pe = k_pe2.shape[-1]
    assert kv_cache.shape[2] == d_c + d_pe
    assert kv_c.shape[0] >= n_w and (d_pe == 0 or k_pe2.shape[0] >= n_w)
    if d_pe:
        assert k_pe2.dtype == kv_cache.dtype and k_pe2.stride(-1) == 1
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    ti = token_indices.contiguous()
    slots = slot_mapping.contiguous()
    out = torch.empty_like(ti)
    n_q = ti.shape[0]
    remap_tiles = num_topk // BLOCK_N
    BLOCK_D = 128
    write_tiles = triton.cdiv(d_c + d_pe, BLOCK_D)
    rows = max(n_q, n_w)
    if rows == 0:
        return out
    _remap_and_cache_mla_kernel[(rows, remap_tiles + write_tiles)](
        req_id_c,
        block_table_c,
        ti,
        out,
        kv_c,
        k_pe2 if d_pe else kv_c,
        slots,
        kv_cache,
        n_q,
        n_w,
        block_table_c.shape[1],
        block_table_c.stride(0),
        block_table_c.stride(1),
        ti.stride(0),
        out.stride(0),
        kv_c.stride(0),
        k_pe2.stride(0) if d_pe else 0,
        kv_cache.stride(0),
        kv_cache.stride(1),
        BLOCK_SIZE=block_size,
        BLOCK_STRIDE_ROWS=block_stride_rows,
        NUM_TOPK=num_topk,
        REMAP_TILES=remap_tiles,
        BLOCK_N=BLOCK_N,
        CACHE_BLOCK=kv_cache.shape[1],
        D_C=d_c,
        D_PE=d_pe,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out
