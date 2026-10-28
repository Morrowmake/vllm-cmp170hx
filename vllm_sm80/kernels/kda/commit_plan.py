# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM commit plan and align postprocess kernels for KDA layers.

Upstream RecoverSSM plans each spec-decode row's commit in
``_prepare_commit_plan_kernel`` (vllm/models/kimi_k3/nvidia/ops/recoverssm.py)
and, after sampling, moves the request's running state column in
``_postprocess_recoverssm_align_kernel``
(vllm/v1/worker/gpu/model_states/recoverssm.py). Both clamp the final state
column to the last block-table column. These versions replace them at the
upstream launch sites (the upstream definitions stay in place, unused):

  * plan: padding/null sources and filtered mappings (request index < 0)
    exit before dereferencing request arrays or a block-table row, as does a
    zero-length commit; the final state column ``max((n - 1) // B, 0)`` is
    read with a mask instead of being clamped, so a missing owner is never
    replaced by a different checkpoint, and the row is valid only when that
    owner exists; the boundary load is masked to the table; device asserts
    flag a missing owner in TRITON_DEBUG builds;
  * align postprocess: the running column follows the plan's commit length
    (the sampler count clipped to the query and the speculative window), not
    the sampled count; rows with a filtered mapping or a null source are left
    alone, and a zero-length commit keeps the source column (the copy bias is
    still reset to the neutral value).

Callers: the Kimi-K3 commit context (kimi_k3/nvidia/ops/recoverssm.py), the
GLM KDA recover commit (``recover``) and RecoverSSMState.commit_step.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    # Padding/null sources and filtered mappings do not own a checkpoint.
    # Exit before dereferencing request arrays or a block-table row.
    if (source_state_idx <= null_block_id) | (request_idx < 0):
        tl.store(commit_lens_ptr + spec_idx, 0)
        tl.store(final_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_recovery_lens_ptr + spec_idx, 0)
        return
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)
    if commit_len <= 0:
        tl.store(commit_lens_ptr + spec_idx, 0)
        tl.store(final_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_recovery_lens_ptr + spec_idx, 0)
        return

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        # Columns own tokens (col * B, (col + 1) * B], just as the allocator's
        # ceil division does. At an exact boundary the checkpoint column is
        # also the final state; the next column need not exist yet.
        final_state_col = tl.maximum((final_num_computed - 1) // mamba_block_size, 0)
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col,
            mask=final_state_col < block_table_width,
            other=null_block_id,
        ).to(tl.int64)
        # Diagnostic in TRITON_DEBUG builds; never clamp a missing owner to
        # a different checkpoint. Masked loads/stores remain safe in release.
        tl.device_assert(
            final_state_idx > null_block_id, "RecoverSSM final state has no owner"
        )
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=(
                crosses_boundary
                & (next_boundary // mamba_block_size - 1 >= 0)
                & (next_boundary // mamba_block_size - 1 < block_table_width)
            ),
            other=null_block_id,
        ).to(tl.int64)
        tl.device_assert(
            (~crosses_boundary) | (boundary_state_idx > null_block_id),
            "RecoverSSM boundary state has no owner",
        )
    valid = (final_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


def launch_align_postprocess(
    postprocess_meta: RecoverSSMPostprocessMetadata,
    idx_mapping: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Move each committed request's running state column to the plan's
    final column and reset its next-step copy bias to the neutral value."""
    _postprocess_recoverssm_align_kernel[(postprocess_meta.num_spec_decodes,)](
        idx_mapping,
        postprocess_meta.commit_lens,
        postprocess_meta.source_state_indices,
        postprocess_meta.request_indices,
        postprocess_meta.num_computed_tokens,
        state_indices,
        num_accepted_tokens,
        MAMBA_BLOCK_SIZE=postprocess_meta.block_size,
        NULL_BLOCK_ID=NULL_BLOCK_ID,
        stride_source_state=postprocess_meta.source_state_indices.stride(0),
    )


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    commit_lens_ptr,
    source_state_indices_ptr,
    request_indices_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    stride_source_state,
):
    spec_idx = tl.program_id(0)
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    if batch_idx < 0:
        return
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    source_state = tl.load(source_state_indices_ptr + spec_idx * stride_source_state)
    if source_state <= NULL_BLOCK_ID:
        return
    commit_len = tl.load(commit_lens_ptr + spec_idx)
    if commit_len > 0:
        num_computed = tl.load(num_computed_ptr + batch_idx)
        # Match the commit destination and allocator even when query/window
        # clipping made the effective count smaller than the sampler count.
        tl.store(
            state_idx_ptr + req_state_idx,
            tl.maximum((num_computed + commit_len - 1) // MAMBA_BLOCK_SIZE, 0),
        )
    # Zero acceptance leaves the existing source column intact. Its checkpoint
    # has not advanced, so the next pre-copy must still use the neutral bias.
    tl.store(num_accepted_ptr + req_state_idx, 1)
