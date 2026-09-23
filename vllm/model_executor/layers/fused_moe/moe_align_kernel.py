# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic MoE alignment by chunked counting sort.

WHAT THIS REPLACES.  ``deterministic_moe_align_block_size`` in
``moe_align_block_size.py`` gets the same layout out of ``torch.argsort`` over
every routed row.  At prefill that is 9216 rows per MoE layer, 45 layers, and
it priced out at +13.9 % of a decode step at concurrency 4 -- the order is
right, the cost is not.  A counting sort over 288 buckets does not need a sort
at all, and this file is that counting sort.

THE DECOMPOSITION.  Split the NUMEL routed rows into contiguous chunks of
BLOCK_C.  Then

    rank(i) = #{j < i : e_j == e_i}
            = chunk_base[chunk(i), e_i]  +  #{j in chunk(i), j < i : e_j == e_i}

where ``chunk_base`` is the exclusive scan over chunks of the per-chunk
histogram.  The right-hand count is local to one chunk, so it costs
O(NUMEL * BLOCK_C) pairwise comparisons instead of the O(NUMEL^2) that
``ampere_decode/moe_routing.py::_align_scatter`` pays -- 1.2 M comparisons at
NUMEL = 9216 against 85 M.  With BLOCK_C = NUMEL there is one chunk,
``chunk_base`` is zero and this degenerates exactly to ``_align_scatter``, so
this is that kernel generalised to arbitrary M, not a second algorithm.

THREE LAUNCHES.
    _align_hist_kernel     per-chunk histogram + the sentinel fill of
                           ``sorted_token_ids`` + the local expert id per row
                           + ``scatter_idx``.  grid covers both jobs.
    _align_scan_kernel     ONE CTA: the exclusive scan over chunks (which also
                           yields the per-expert totals in registers, so the
                           counts never round-trip through global memory), then
                           the padded exclusive prefix over experts and
                           ``num_tokens_post_pad``.
    _align_scatter_kernel  the within-chunk rank and the scatter, plus
                           ``expert_ids`` on the same grid.

DETERMINISM.  Three things could set an order and none of them does.  The
histogram is ``+1`` integer atomics, which commute, so the counts are
reproducible whatever order the threads arrive in.  Both scans are
``tl.cumsum`` over a fixed axis.  The within-chunk rank is an explicit "how
many earlier rows chose this expert" count, never an ``atomicAdd`` whose
return value depends on arrival order -- which is exactly the property the
upstream ``count_and_sort_expert_tokens_kernel`` lacks.

CUDA-GRAPH CAPTURABLE.  NUMEL, E, BS, MNP, NBLK and NCHUNK are runtime
scalars with specialization disabled, so there is one compiled variant per
(HAS_MAP, HAS_SCATTER) pair and no recompile when M moves -- a constexpr NUMEL
would both explode the variant count and risk compiling inside a capture.
Nothing syncs, nothing calls ``.item()``, and the scratch is allocated once
per (device, E) by ``warmup()``, which must run before capture.  The
``_align_scan_kernel`` resets every histogram entry it consumed to zero, the
same self-resetting discipline ``_align_meta`` uses, so a replay is correct.

Scratch is shared per (device, E), so calls using it must be serialized on
one stream.  Capacity growth is allowed only outside capture.  No CUDA work
and no allocation happens at import or on the CPU reference path.
"""

import torch

from vllm.triton_utils import tl, triton

BLOCK_C = 128
BLOCK_E = 512
BLOCK_NC = 16
BLOCK_P = 1024
BLOCK_B = 32
NUMEL_CAP = 16384


@triton.jit(
    do_not_specialize=["NUMEL", "E", "MNP", "NCHUNK"],
    do_not_specialize_on_alignment=[
        "topk_ptr", "sorted_ptr", "eid_ptr", "cnt_ptr", "map_ptr", "scatter_ptr"
    ],
)
def _align_hist_kernel(
    topk_ptr, sorted_ptr, eid_ptr, cnt_ptr, map_ptr, scatter_ptr,
    NUMEL, E, MNP, NCHUNK,
    BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr,
    HAS_MAP: tl.constexpr, HAS_SCATTER: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    tl.store(sorted_ptr + p, NUMEL, mask=p < MNP)
    if pid < NCHUNK:
        i = pid * BLOCK_C + tl.arange(0, BLOCK_C)
        mi = i < NUMEL
        raw = tl.load(topk_ptr + i, mask=mi, other=-1)
        ok = mi & (raw >= 0) & (raw < E)
        if HAS_MAP:
            e = tl.load(map_ptr + tl.where(ok, raw, 0), mask=ok, other=-1)
            ok = ok & (e >= 0)
        else:
            e = raw
        e = tl.where(ok, e, E)
        tl.store(eid_ptr + i, e, mask=mi)
        tl.atomic_add(cnt_ptr + pid * E + e, 1, mask=ok)
        if HAS_SCATTER:
            tl.store(scatter_ptr + i, tl.where(ok, i, -1), mask=mi)


@triton.jit(
    do_not_specialize=["E", "BS", "NCHUNK"],
    do_not_specialize_on_alignment=["cnt_ptr", "base_ptr", "excl_ptr", "ntpp_ptr"],
)
def _align_scan_kernel(
    cnt_ptr, base_ptr, excl_ptr, ntpp_ptr, E, BS, NCHUNK,
    BLOCK_E: tl.constexpr, BLOCK_NC: tl.constexpr,
):
    offs_e = tl.arange(0, BLOCK_E)
    valid_e = offs_e < E
    carry = tl.zeros([BLOCK_E], tl.int32)
    for c0 in range(0, NCHUNK, BLOCK_NC):
        rows = c0 + tl.arange(0, BLOCK_NC)
        m = (rows < NCHUNK)[:, None] & valid_e[None, :]
        offsets = rows[:, None] * E + offs_e[None, :]
        t = tl.load(cnt_ptr + offsets, mask=m, other=0)
        base = tl.cumsum(t, axis=0) - t + carry[None, :]
        tl.store(base_ptr + offsets, base, mask=m)
        tl.store(cnt_ptr + offsets, tl.zeros([BLOCK_NC, BLOCK_E], tl.int32), mask=m)
        carry += tl.sum(t, axis=0)
    padded = tl.where(valid_e, ((carry + BS - 1) // BS) * BS, 0)
    excl = tl.cumsum(padded, axis=0) - padded
    tl.store(excl_ptr + offs_e, excl, mask=valid_e)
    tl.store(ntpp_ptr, tl.sum(padded))


@triton.jit(
    do_not_specialize=["NUMEL", "E", "BS", "NBLK", "NCHUNK"],
    do_not_specialize_on_alignment=[
        "eid_ptr", "base_ptr", "excl_ptr", "sorted_ptr", "expert_ptr", "ntpp_ptr"
    ],
)
def _align_scatter_kernel(
    eid_ptr, base_ptr, excl_ptr, sorted_ptr, expert_ptr, ntpp_ptr,
    NUMEL, E, BS, NBLK, NCHUNK,
    BLOCK_C: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NCHUNK:
        i = pid * BLOCK_C + tl.arange(0, BLOCK_C)
        mi = i < NUMEL
        er = tl.load(eid_ptr + i, mask=mi, other=E)
        ok = mi & (er < E)
        cond = ((er[:, None] == er[None, :])
                & (i[None, :] < i[:, None]) & ok[None, :])
        lrank = tl.sum(cond.to(tl.int32), axis=1)
        base = tl.load(base_ptr + pid * E + er, mask=ok, other=0)
        ex = tl.load(excl_ptr + er, mask=ok, other=0)
        tl.store(sorted_ptr + ex + base + lrank, i.to(tl.int32), mask=ok)
    # The grid is max(NCHUNK, cdiv(NBLK, BLOCK_B)); whichever job is shorter
    # leaves its CTAs idle rather than computing a [BLOCK_B, BLOCK_E] tile it
    # would then mask away.
    if pid * BLOCK_B < NBLK:
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_e = tl.arange(0, BLOCK_E)
        valid_e = offs_e < E
        bstart = tl.load(excl_ptr + offs_e, mask=valid_e, other=0) // BS
        nblk_live = tl.load(ntpp_ptr) // BS
        # block_start is non-decreasing, so the count of experts that start at
        # or before b, minus one, is the last one that does -- searchsorted
        # right=True minus one, which is what the torch path computes. Experts
        # with an empty (zero-length) segment share their successor's start and
        # so are stepped over.
        owner = tl.sum(((bstart[None, :] <= b[:, None])
                        & valid_e[None, :]).to(tl.int32), axis=1) - 1
        tl.store(expert_ptr + b, tl.where(b < nblk_live, owner, -1),
                 mask=b < NBLK)


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


_SCRATCH: dict[tuple[torch.device, int], tuple[torch.Tensor, ...]] = {}
_RETIRED_SCRATCH: list[tuple[torch.Tensor, ...]] = []


def _scratch(device: torch.device, num_experts: int, numel: int):
    key = (device, num_experts)
    scratch = _SCRATCH.get(key)
    if scratch is None or scratch[0].numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoE alignment scratch must be warmed before capture")
        if scratch is not None:
            # Captured graphs may still reference the old allocation.
            _RETIRED_SCRATCH.append(scratch)
        capacity = max(NUMEL_CAP, triton.next_power_of_2(numel))
        nchunk = triton.cdiv(capacity, BLOCK_C)
        scratch = (
            torch.empty(capacity, dtype=torch.int32, device=device),
            torch.zeros(nchunk * num_experts, dtype=torch.int32, device=device),
            torch.empty(nchunk * num_experts, dtype=torch.int32, device=device),
            torch.empty(num_experts, dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = scratch
    return scratch


def kernel_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
) -> None:
    """Write deterministic alignment out-params, using torch on CPU."""
    assert 0 < num_experts <= BLOCK_E
    if not topk_ids.is_cuda:
        chunked_align_reference(
            topk_ids, num_experts, block_size, sorted_token_ids, expert_ids,
            num_tokens_post_pad, expert_map, scatter_idx,
        )
        return
    numel = topk_ids.numel()
    nchunk = triton.cdiv(numel, BLOCK_C)
    mnp, nblk = sorted_token_ids.numel(), expert_ids.numel()
    with torch.cuda.device(topk_ids.device):
        eid, cnt, base, excl = _scratch(topk_ids.device, num_experts, numel)
        _align_hist_kernel[(max(1, nchunk, triton.cdiv(mnp, BLOCK_P)),)](
            topk_ids, sorted_token_ids, eid, cnt,
            expert_map if expert_map is not None else eid,
            scatter_idx if scatter_idx is not None else eid,
            numel, num_experts, mnp, nchunk,
            BLOCK_C, BLOCK_P, expert_map is not None, scatter_idx is not None,
            num_warps=8, num_stages=1,
        )
        _align_scan_kernel[(1,)](
            cnt, base, excl, num_tokens_post_pad, num_experts, block_size, nchunk,
            BLOCK_E, BLOCK_NC, num_warps=8, num_stages=1,
        )
        _align_scatter_kernel[(max(1, nchunk, triton.cdiv(nblk, BLOCK_B)),)](
            eid, base, excl, sorted_token_ids, expert_ids, num_tokens_post_pad,
            numel, num_experts, block_size, nblk, nchunk,
            BLOCK_C, BLOCK_B, BLOCK_E, num_warps=8, num_stages=1,
        )


def warmup(device: torch.device | str, num_experts: int) -> None:
    """Allocate default scratch and compile all flag variants before capture.

    Larger capacities must be primed by a call at the required size outside
    capture. Existing captured graphs retain their original scratch tensors.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return
    assert 0 < num_experts <= BLOCK_E
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoE alignment warmup must run before capture")
        topk_ids = torch.zeros((1, 8), dtype=torch.int32, device=device)
        sorted_ids = torch.empty(8 * 8, dtype=torch.int32, device=device)
        experts = torch.empty(8, dtype=torch.int32, device=device)
        ntpp = torch.empty(1, dtype=torch.int32, device=device)
        expert_map = torch.arange(num_experts, dtype=torch.int32, device=device)
        scatter = torch.empty((8, 1), dtype=torch.int32, device=device)
        for mapping in (None, expert_map):
            for scattering in (None, scatter):
                kernel_moe_align_block_size(
                    topk_ids, num_experts, 8, sorted_ids, experts, ntpp,
                    mapping, scattering,
                )


def warmup_from_worker(worker) -> None:
    """Pre-capture hook, called from ``model_executor/warmup/kernel_warmup.py``.

    No-op unless ``VLLM_GLM5_DETERMINISTIC_MOE_ALIGN`` selects this path. The
    expert count comes from the model config, because the scratch is keyed by
    it; if the count cannot be read the warmup is skipped and the first call
    allocates, which is correct outside a capture and raises inside one.
    """
    from vllm import envs

    if envs.VLLM_GLM5_DETERMINISTIC_MOE_ALIGN != 1:
        return
    model_config = worker.vllm_config.model_config
    cfg = getattr(model_config, "hf_text_config", None) or model_config.hf_config
    num_experts = 0
    for attr in ("n_routed_experts", "num_experts", "num_local_experts",
                 "moe_num_experts"):
        value = getattr(cfg, attr, None)
        if value:
            num_experts = int(value)
            break
    if not 0 < num_experts <= BLOCK_E:
        return
    warmup(worker.device, num_experts)
    return num_experts
