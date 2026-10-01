# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.triton_utils import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def chunk_indices_from_counts(chunk_counts: list[int]) -> torch.Tensor:
    """[NT, 2] int64 rows (sequence index into ``cu_seqlens``, chunk index
    within that sequence), in sequence order. Sequences with no chunks (zero
    length) contribute no rows and keep their index, so the rows after them
    still name the right sequence."""
    counts = torch.tensor(chunk_counts, dtype=torch.int64).clamp_(min=0)
    seq = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    return torch.stack([seq, torch.arange(seq.numel()) - starts], 1)


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # This will be fixed by https://github.com/vllm-project/vllm/pull/51540.
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    chunk_indices = chunk_indices_from_counts(chunk_counts)
    return async_tensor_h2d(
        chunk_indices, device=cu_seqlens.device, dtype=cu_seqlens.dtype
    )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
