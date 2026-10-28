# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sliding-window metadata of every SWA KV cache group of a step in one
Triton launch (sm_80 DeepSeek-V4.1).

Upstream's ``DeepseekSparseSWAMetadataBuilder.build``
(``vllm/v1/attention/backends/mla/sparse_swa.py``) runs per KV cache group:
``torch.ge(slot_mapping, 0)`` into ``is_valid_token``, the fill of
``decode_swa_lens`` past the decode rows, ``_compute_swa_indices_and_lens_kernel``
for the decode rows and again for the prefill rows, and the prefill gather
lengths (``ComputePrefillMetadataKernel``). DeepSeek-V4.1 has one SWA cache per
layer, each its own group with its own block table, so a step builds this 43
times. :func:`swa_meta_batch` writes all of it for every group at once, with
the same integer arithmetic, into each group's builder buffers (addressed
through pointer tables), so the buffers hold the per-group build's bytes.

Grid: (tokens + 1, groups). Program ``(t, g)``, ``t`` below the token count,
writes token ``t``'s validity and, for a decode or prefill token, its index row
and length in group ``g``'s buffers. Program ``(tokens, g)`` zeroes group
``g``'s decode lengths past the decode rows (when there are decode rows), and
for ``g == 0`` writes the prefill gather lengths. The image visibility of the
V4 vision variant is not served (its builders keep the per-group build).
"""

import torch

from vllm.triton_utils import tl, triton

__all__ = ["swa_meta_batch"]

_BLOCK = 1024
# Rows of the output pointer table (out_ptrs).
OUT_ROWS = ("is_valid_token", "decode_swa_indices", "decode_swa_lens",
            "prefill_swa_indices", "prefill_swa_lens")


@triton.jit
def _ptr(table_ptr, DTYPE: tl.constexpr):
    return tl.cast(tl.load(table_ptr), tl.pointer_type(DTYPE))


@triton.jit
def _window_row(
    indices_ptr,
    lens_ptr,
    row,
    width,
    token,
    valid,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_ptr,
    replay_start_ptr,
    window_size,
    block_size,
    BLOCK: tl.constexpr,
):
    """One row of ``_compute_swa_indices_and_lens_kernel`` (no image span)."""
    if not valid:
        tl.store(lens_ptr + row, 0)
        for i in range(0, width, BLOCK):
            offset = i + tl.arange(0, BLOCK)
            tl.store(indices_ptr + row * width + offset, -1, mask=offset < width)
    else:
        req_idx = tl.load(token_to_req_ptr + token)
        query_start = tl.load(query_start_loc_ptr + req_idx)
        query_end = tl.load(query_start_loc_ptr + req_idx + 1)
        query_len = query_end - query_start
        seq_len = tl.load(seq_lens_ptr + req_idx)
        prefix_len = seq_len - query_len
        pos = prefix_len + token - query_start
        start_pos = tl.maximum(pos - (window_size - 1), 0)
        start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
        end_pos = pos + 1
        swa_len = end_pos - start_pos
        tl.store(lens_ptr + row, swa_len)
        for i in range(0, width, BLOCK):
            offset = i + tl.arange(0, BLOCK)
            pos_offset = start_pos + offset
            block_numbers = tl.load(
                block_table_ptr
                + req_idx * block_table_stride
                + pos_offset // block_size,
                mask=pos_offset < end_pos,
            )
            slot_ids = block_numbers * block_size + pos_offset % block_size
            slot_ids = tl.where(offset < swa_len, slot_ids, -1)
            tl.store(
                indices_ptr + row * width + offset, slot_ids, mask=offset < width
            )


@triton.jit(
    do_not_specialize=[
        "num_groups",
        "num_slots",
        "num_decode_tokens",
        "num_prefill_tokens",
        "num_decodes",
        "num_prefills",
        "lens_capacity",
        "window_size",
        "decode_width",
        "prefill_width",
        "block_size",
    ]
)
def _swa_meta_batch_kernel(
    slot_ptrs,  # int64 [groups]: each group's slot mapping (int64 [num_slots])
    block_table_ptrs,  # int64 [groups]: each group's block table (int32)
    block_table_strides,  # int64 [groups]
    out_ptrs,  # int64 [5, groups]: is_valid, decode idx/lens, prefill idx/lens
    gather_lens_ptr,  # int32 [num_prefills]
    query_start_loc_ptr,  # int32 [num_reqs + 1]
    seq_lens_ptr,  # int32 [num_reqs]
    token_to_req_ptr,  # int32 [num_slots]
    replay_start_ptr,  # int32 [num_reqs]
    num_groups,
    num_slots,
    num_decode_tokens,
    num_prefill_tokens,
    num_decodes,
    num_prefills,
    lens_capacity,
    window_size,
    decode_width,
    prefill_width,
    block_size,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.program_id(1)
    if t == num_slots:
        if num_decode_tokens > 0:
            lens_ptr = _ptr(out_ptrs + 2 * num_groups + g, tl.int32)
            for i in range(num_decode_tokens, lens_capacity, BLOCK):
                offset = i + tl.arange(0, BLOCK)
                tl.store(lens_ptr + offset, 0, mask=offset < lens_capacity)
        if g == 0:
            # ComputePrefillMetadataKernel: per prefill request.
            for i in range(0, num_prefills, BLOCK):
                offset = i + tl.arange(0, BLOCK)
                mask = offset < num_prefills
                req = num_decodes + offset
                seq_len = tl.load(seq_lens_ptr + req, mask=mask, other=0)
                qsl_start = tl.load(query_start_loc_ptr + req, mask=mask, other=0)
                qsl_end = tl.load(query_start_loc_ptr + req + 1, mask=mask, other=0)
                query_len = qsl_end - qsl_start
                prefix_len = seq_len - query_len
                replay_start = tl.load(replay_start_ptr + req, mask=mask, other=0)
                visible_prefix = tl.maximum(prefix_len - replay_start, 0)
                gather_len = query_len + tl.minimum(visible_prefix, window_size - 1)
                tl.store(gather_lens_ptr + offset, gather_len, mask=mask)
    else:
        slot = tl.load(_ptr(slot_ptrs + g, tl.int64) + t)
        valid = slot >= 0
        tl.store(_ptr(out_ptrs + 0 * num_groups + g, tl.int8) + t, valid.to(tl.int8))
        block_table_ptr = _ptr(block_table_ptrs + g, tl.int32)
        block_table_stride = tl.load(block_table_strides + g)
        if t < num_decode_tokens:
            _window_row(
                _ptr(out_ptrs + 1 * num_groups + g, tl.int32),
                _ptr(out_ptrs + 2 * num_groups + g, tl.int32),
                t,
                decode_width,
                t,
                valid,
                block_table_ptr,
                block_table_stride,
                query_start_loc_ptr,
                seq_lens_ptr,
                token_to_req_ptr,
                replay_start_ptr,
                window_size,
                block_size,
                BLOCK,
            )
        elif t < num_decode_tokens + num_prefill_tokens:
            _window_row(
                _ptr(out_ptrs + 3 * num_groups + g, tl.int32),
                _ptr(out_ptrs + 4 * num_groups + g, tl.int32),
                t - num_decode_tokens,
                prefill_width,
                t,
                valid,
                block_table_ptr,
                block_table_stride,
                query_start_loc_ptr,
                seq_lens_ptr,
                token_to_req_ptr,
                replay_start_ptr,
                window_size,
                block_size,
                BLOCK,
            )


def swa_meta_batch(
    slot_ptrs: torch.Tensor,
    block_table_ptrs: torch.Tensor,
    block_table_strides: torch.Tensor,
    out_ptrs: torch.Tensor,
    gather_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    replay_start: torch.Tensor,
    *,
    num_slots: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
    num_decodes: int,
    num_prefills: int,
    lens_capacity: int,
    window_size: int,
    decode_width: int,
    prefill_width: int,
    block_size: int,
) -> None:
    """One launch for every group (the tables' length). The caller keeps
    ``num_decode_tokens + num_prefill_tokens <= num_slots``, every group's slot
    mapping at least ``num_slots`` long and its block table covering the
    batch's requests."""
    num_groups = slot_ptrs.shape[0]
    _swa_meta_batch_kernel[(num_slots + 1, num_groups)](
        slot_ptrs,
        block_table_ptrs,
        block_table_strides,
        out_ptrs,
        gather_lens,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        replay_start,
        num_groups,
        num_slots,
        num_decode_tokens,
        num_prefill_tokens,
        num_decodes,
        num_prefills,
        lens_capacity,
        window_size,
        decode_width,
        prefill_width,
        block_size,
        BLOCK=_BLOCK,
    )
