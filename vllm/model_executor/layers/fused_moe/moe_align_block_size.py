# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import Literal, overload

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


@functools.cache
def use_deterministic_moe_align() -> bool:
    """Whether to take the deterministic alignment. Cached: a boot-time
    choice (tests call cache_clear)."""
    from vllm import envs

    return envs.VLLM_GLM5_DETERMINISTIC_MOE_ALIGN


# ---------------------------------------------------------------------------
# Deterministic alignment -- VLLM_GLM5_DETERMINISTIC_MOE_ALIGN
# ---------------------------------------------------------------------------
#
# csrc/libtorch_stable/moe/moe_align_sum_kernels.cu takes its single-kernel
# "small batch expert" path only when topk_ids.numel() < 1024 AND
# num_experts <= 64. GLM-5.3-Flash has 288 routed experts, so every call goes
# down the two-kernel path, and its second kernel
# (count_and_sort_expert_tokens_kernel) computes each token's slot with
#
#     rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
#
# i.e. the order of tokens inside an expert segment is whatever order the
# threads happened to arrive in. The fused Marlin MoE's per-expert
# accumulation follows that order, so two identical calls differ in the low
# bits. This path ranks by ascending flat routed-row index instead.
#
# There is no other deterministic option in this tree: there is no
# moe_align_block_size_triton and no upstream determinism flag. The fused
# sm_80 routing kernel (vllm/ampere_decode/moe_routing.py, behind
# VLLM_GLM5_DECODE_KERNELS) does contain a count-based scatter that should
# already be deterministic, but it is a different, separately gated code path
# and it only covers M <= 8.


def deterministic_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
) -> None:
    """Drop-in replacement for ``ops.moe_align_block_size`` with a defined
    within-segment order.

    Writes the same three (four with scatter_idx) out-params with the same
    conventions as the CUDA op:

    * ``sorted_token_ids`` is pre-filled with ``topk_ids.numel()`` (the
      padding sentinel) and then, for each valid routed row i, holds i at
      ``cumsum[expert(i)] + rank(i)`` where ``cumsum`` is the exclusive
      prefix sum of ``ceil(count[e] / block_size) * block_size``.
    * ``rank(i)`` here is i's position among that expert's rows ordered by
      ascending i -- this is the whole change.
    * ``expert_ids[b]`` is the (local) expert owning block b, -1 past the end.
    * ``num_tokens_post_pad`` is ``cumsum[num_experts]``.
    * ``scatter_idx[i]`` is i for valid routes and -1 otherwise.

    Every step is order-independent: ``bincount`` sums +1 integer atomics
    (commutative), ``cumsum`` is a fixed-order scan, and ``argsort(stable)``
    is a radix sort with a defined tie order.

    Shapes are static and nothing is read back to the host -- no ``.item()``,
    no boolean-mask indexing, and no op that sizes an output from device data
    (which rules out ``torch.bincount``) -- so the path stays CUDA-graph
    capturable, which the decode path requires.
    """
    device = topk_ids.device
    numel = topk_ids.numel()
    flat = topk_ids.reshape(-1).to(torch.int64)

    # get_local_expert_id(): out-of-range ids are invalid, then the map is
    # applied and may itself return -1.
    valid = (flat >= 0) & (flat < num_experts)
    eid = torch.where(valid, flat, torch.zeros_like(flat))
    if expert_map is not None:
        eid = expert_map.to(torch.int64)[eid]
        valid = valid & (eid >= 0)
    # Invalid routes go to a sentinel bucket that sorts after every expert.
    eid = torch.where(valid, eid, torch.full_like(eid, num_experts))

    # A scatter_add_ histogram, not torch.bincount: bincount sizes its output
    # from the device data, which is a host sync and is therefore rejected
    # during CUDA graph capture. Integer +1 accumulation is order-independent
    # either way.
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    counts.scatter_add_(0, eid, torch.ones_like(eid))
    counts = counts[:num_experts]
    padded = ((counts + block_size - 1) // block_size) * block_size

    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum[1:] = torch.cumsum(padded, 0)
    unpadded_start = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    unpadded_start[1:] = torch.cumsum(counts, 0)

    # Ascending expert, then ascending flat routed-row index within it.
    order = torch.argsort(eid, stable=True)
    sorted_eid = eid[order]
    pos = torch.arange(numel, device=device, dtype=torch.int64)
    dest = cumsum[sorted_eid] + (pos - unpadded_start[sorted_eid])
    keep = sorted_eid < num_experts

    # Invalid routes land in the padded tail (cumsum[num_experts] onwards) and
    # write the padding sentinel there, so the scatter is a no-op for them and
    # the destination index needs only a clamp, not a mask.
    buf_len = sorted_token_ids.numel()
    sorted_token_ids.fill_(numel)
    sorted_token_ids.reshape(-1).scatter_(
        0,
        dest.clamp_(min=0, max=buf_len - 1),
        torch.where(keep, order, torch.full_like(order, numel)).to(
            sorted_token_ids.dtype
        ),
    )

    # Block b belongs to the last expert whose padded segment starts at or
    # before b; blocks past the end are -1 (the CUDA op's inactive id).
    block_start = cumsum // block_size
    blocks = torch.arange(expert_ids.numel(), device=device, dtype=torch.int64)
    owner = torch.searchsorted(block_start, blocks, right=True) - 1
    expert_ids.reshape(-1).copy_(
        torch.where(blocks < block_start[num_experts], owner, -1).to(expert_ids.dtype)
    )

    num_tokens_post_pad.reshape(-1)[:1] = cumsum[num_experts : num_experts + 1].to(
        num_tokens_post_pad.dtype
    )

    if scatter_idx is not None:
        scatter_idx.reshape(-1).copy_(
            torch.where(valid, pos, torch.full_like(pos, -1)).to(scatter_idx.dtype)
        )


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: bool,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
): ...


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Args:
        topk_ids: A tensor of shape [total_tokens, top_k] representing the
            top-k expert indices for each token.
        block_size: The block size used in block matrix multiplication.
        num_experts: The total number of experts.
        expert_map: A tensor of shape [num_experts] that maps the expert index
            from the global space to the local index space of the current
            expert parallel shard. If the expert is not in the current expert
            parallel shard, the mapping is set to -1.
        pad_sorted_ids: A flag indicating whether the sorted_token_ids length
            should be padded to a multiple of block_size,
        ignore_invalid_experts: A flag indicating whether to ignore invalid
            experts. When False, all expert_ids in topk_ids will participate in
            counting and ranking, but invalid experts in expert_ids will be marked
            as -1. When True, all invalid expert_ids in topk_ids will be ignored
            and will not participate in counting or ranking, and there will be no
            -1 in expert_ids.
        return_scatter_idx: Whether to additionally return a masked identity
            mapping for the original routed rows.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    - scatter_idx: Only returned when return_scatter_idx=True. An int32 tensor
        shaped [topk_ids.numel(), 1], containing the original routed row index
        for valid routes and -1 otherwise. With expert_map, this requires
        ignore_invalid_experts=True.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    """
    if return_scatter_idx and expert_map is not None and not ignore_invalid_experts:
        raise ValueError("scatter_idx with expert_map requires ignore_invalid_experts")

    from vllm import envs

    if not return_scatter_idx and envs.VLLM_GLM5_DECODE_KERNELS:
        from vllm.ampere_decode import take_fused_align

        # `fused_route_align` in vllm/ampere_decode/moe_routing.py produced
        # this alignment in the same launch as the routing -- the fusion is
        # exact (count_and_sort is phase 2 of moe_align_block_size and both
        # consume only topk_ids, which the routing kernel produced, and
        # nothing reads topk_ids in between). A miss here is harmless: the
        # upstream path below runs exactly as before.
        _fused = take_fused_align(
            topk_ids, block_size, num_experts, expert_map, pad_sorted_ids
        )
        if _fused is not None:
            return _fused

    scatter_idx = None
    if return_scatter_idx:
        scatter_idx = torch.empty(
            (topk_ids.numel(), 1),
            dtype=torch.int32,
            device=topk_ids.device,
        )
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    if use_deterministic_moe_align():
        deterministic_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )
    else:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    if scatter_idx is not None:
        return sorted_ids, expert_ids, num_tokens_post_pad, scatter_idx
    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Args:
        max_tokens_per_batch (int): Number of tokens in each batch (both
            valid and invalid).
        block_size (int): block_size to align the data to.
        expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
            the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.

    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.

    """
    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
