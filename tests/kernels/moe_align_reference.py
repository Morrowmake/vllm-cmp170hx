# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference for the deterministic MoE alignment, tests only.

``chunked_align_reference`` runs the decomposition of
``vllm_sm80/kernels/moe/align.py`` (per-chunk
histogram, scan over chunks, within-chunk pairwise rank) in torch on any
device, writing the same out-params as ``kernel_moe_align_block_size``. With
``block_c`` at least the number of routed rows it is the plain "rank by
ascending flat routed-row index" alignment.
"""

import torch

from vllm_sm80.kernels.moe.align import BLOCK_C


def chunked_align_reference(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
    *,
    block_c: int = BLOCK_C,
) -> None:
    """Execute the histogram, chunk scan and local pairwise rank in torch."""
    assert block_c > 0
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = flat.numel()
    nchunk = (numel + block_c - 1) // block_c
    ok = (flat >= 0) & (flat < num_experts)
    e = flat
    if expert_map is not None:
        e = expert_map[torch.where(ok, flat, 0)].to(torch.int64)
        ok = ok & (e >= 0)
    eid = torch.where(ok, e, num_experts)
    chunk_cnt = torch.zeros(
        (nchunk, num_experts + 1), dtype=torch.int64, device=device
    )
    for c in range(nchunk):
        er = eid[c * block_c : (c + 1) * block_c]
        chunk_cnt[c].scatter_add_(0, er, torch.ones_like(er))
    chunk_cnt = chunk_cnt[:, :num_experts]
    chunk_base = chunk_cnt.cumsum(0) - chunk_cnt
    counts = chunk_cnt.sum(0)
    padded = ((counts + block_size - 1) // block_size) * block_size
    excl = padded.cumsum(0) - padded
    ntpp = padded.sum()
    sorted_token_ids.fill_(numel)
    for c in range(nchunk):
        i = torch.arange(c * block_c, min((c + 1) * block_c, numel), device=device)
        er = eid[i]
        valid = er < num_experts
        cond = ((er[:, None] == er[None, :])
                & (i[None, :] < i[:, None]) & valid[None, :])
        local_rank = cond.to(torch.int64).sum(1)
        dest = excl[er[valid]] + chunk_base[c, er[valid]] + local_rank[valid]
        sorted_token_ids.reshape(-1)[dest] = i[valid].to(sorted_token_ids.dtype)
    b = torch.arange(expert_ids.numel(), device=device)
    owner = ((excl // block_size)[None, :] <= b[:, None]).sum(1) - 1
    expert_ids.reshape(-1).copy_(torch.where(b < ntpp // block_size, owner, -1))
    num_tokens_post_pad.reshape(-1)[:1] = ntpp
    if scatter_idx is not None:
        i = torch.arange(numel, device=device)
        scatter_idx.reshape(-1).copy_(torch.where(ok, i, -1))


def align(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """Fresh (sorted_ids, expert_ids, num_tokens_post_pad) of ``topk_ids``,
    sized as ``moe_align_block_size`` sizes them."""
    n = topk_ids.numel()
    max_pad = n + num_experts * (block_size - 1)
    if n < num_experts:
        max_pad = min(n * block_size, max_pad)
    dev = topk_ids.device
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
    expert_ids = torch.empty((max_pad + block_size - 1) // block_size,
                             dtype=torch.int32, device=dev)
    ntpp = torch.empty(1, dtype=torch.int32, device=dev)
    chunked_align_reference(topk_ids, num_experts, block_size, sorted_ids,
                            expert_ids, ntpp, None, None)
    return sorted_ids, expert_ids, ntpp
