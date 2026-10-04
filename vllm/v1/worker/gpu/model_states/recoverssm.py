# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.utils import AttentionGroup


class RecoverSSMState:
    """Coordinates RecoverSSM metadata between attention and postprocessing."""

    def __init__(self) -> None:
        self._step: tuple[RecoverSSMMetadata, ...] | None = None

    def record_step(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        *,
        for_capture: bool,
    ) -> None:
        if for_capture:
            self._step = None
            return

        step: list[RecoverSSMMetadata] = []
        for group_list in attn_groups:
            for group in group_list:
                metadata = attn_metadata[group.layer_names[0]]
                if isinstance(metadata, RecoverSSMMetadata):
                    step.append(metadata)
        self._step = tuple(step)

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        step = self._step
        self._step = None
        if isinstance(num_sampled, int) or step is None:
            return

        for metadata in step:
            postprocess_meta = metadata.commit_recoverssm_state(num_sampled)
            if postprocess_meta is None:
                continue
            assert state_indices is not None
            # RecoverSSM already restored the accepted state. Update its running
            # column and reset the next-step copy bias to the neutral value.
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
