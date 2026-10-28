# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 index scores on sm_80 (Triton), over a paged BF16 key cache.

Same contract and rounding points as ``sm80_attn_core.index_scores_torch``:
one program per (query, tile of compressed positions) computes the 32 heads'
fp32 dot products with one MMA, rounds each to BF16, rectifies, multiplies by
the head's BF16 weight and rounds again, then sums each rank's heads in head
order (fp32, rounded to BF16) and adds the ranks in order with BF16 rounding.
Positions at or past the query's visible count score ``-inf``. No atomics;
every output element is written by exactly one program.

One compiled variant per index-key cache: the score width ``n_cols`` and the
block-table row stride are not specialised on, and the output rows are padded
to a multiple of 16 so their stride always falls in Triton's divisible-by-16
class. The rest of the specialisation key (dtypes, constexprs, the cache's
block stride and alignment, the fixed q/w strides) is the same for every call
on a cache, so :data:`INDEX_SCORES_WARMUP`, registered by each index-source
layer, compiles everything a served run can reach at boot. No boot dummy run
reaches the kernel: dummy and capture batches have padding slots or position
0, where no compressed position is visible yet.
"""

from dataclasses import dataclass, field

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton

_BLOCK_N = 64
# Output row stride granule: a multiple of 16 keeps one specialisation class.
_ROW_ALIGN = 16


@triton.jit
def _row(s, hidx, h: tl.constexpr):
    """Row ``h`` of ``s`` ([H, N]) as an [N] tensor (adds only zeros)."""
    return tl.sum(tl.where(hidx[:, None] == h, s, 0.0), axis=0)


@triton.jit
def score_tile(q_ptr, w_ptr, req_ptr, k_ptr, bt_ptr, t, cols, n_vis,
               stride_qt, stride_qh, stride_wt, stride_kb, stride_ke, stride_bt,
               ENTRIES: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               HPR: tl.constexpr, BLOCK_N: tl.constexpr):
    valid = cols < n_vis
    req = tl.load(req_ptr + t).to(tl.int64)
    blk = tl.load(bt_ptr + req * stride_bt + cols // ENTRIES, mask=valid, other=0)
    koff = blk.to(tl.int64) * stride_kb + (cols % ENTRIES).to(tl.int64) * stride_ke
    d = tl.arange(0, D)
    k = tl.load(k_ptr + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0)
    hidx = tl.arange(0, H)
    q = tl.load(q_ptr + t * stride_qt + hidx[:, None] * stride_qh + d[None, :])
    s = tl.dot(q, tl.trans(k))  # [H, BLOCK_N] fp32
    s = tl.maximum(s.to(tl.bfloat16).to(tl.float32), 0.0)
    w = tl.load(w_ptr + t * stride_wt + hidx).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    total = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in tl.static_range(H // HPR):
        part = _row(s, hidx, r * HPR)
        for i in tl.static_range(1, HPR):
            part = part + _row(s, hidx, r * HPR + i)
        part = part.to(tl.bfloat16).to(tl.float32)
        if r == 0:
            total = part
        else:
            total = (total + part).to(tl.bfloat16).to(tl.float32)
    total = tl.where(valid, total, float("-inf"))
    return total


@triton.jit(do_not_specialize=["n_cols", "stride_bt"])
def _index_scores_kernel(
    q_ptr,
    w_ptr,
    nvis_ptr,
    req_ptr,
    k_ptr,
    bt_ptr,
    out_ptr,
    n_cols,
    stride_qt,
    stride_qh,
    stride_wt,
    stride_kb,
    stride_ke,
    stride_bt,
    stride_ot,
    ENTRIES: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HPR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_vis = tl.load(nvis_ptr + t).to(tl.int32)
    if tile * BLOCK_N >= n_vis:
        # Nothing visible in this tile: in a fixed-width (graph) launch most
        # tiles are past the context, so skip their keys and the MMA.
        tl.store(
            out_ptr + t * stride_ot + cols,
            tl.full([BLOCK_N], float("-inf"), tl.float32),
            mask=cols < n_cols,
        )
        return
    total = score_tile(q_ptr, w_ptr, req_ptr, k_ptr, bt_ptr, t, cols, n_vis,
        stride_qt, stride_qh, stride_wt, stride_kb, stride_ke, stride_bt,
        ENTRIES, H, D, HPR, BLOCK_N)
    tl.store(out_ptr + t * stride_ot + cols, total, mask=cols < n_cols)


def index_scores_triton(
    q: torch.Tensor,
    w: torch.Tensor,
    n_vis: torch.Tensor,
    req: torch.Tensor,
    idxk: torch.Tensor,
    block_table: torch.Tensor,
    n_cols: int,
    heads_per_rank: int,
) -> torch.Tensor:
    """``[n, n_cols]`` fp32 index scores; see ``index_scores_torch``.

    q: [n, H, D] BF16 (stride 1 on D); w: [n, H] BF16; n_vis, req: [n];
    idxk: [blocks, E, D] BF16 (any block stride); block_table: [R, nb] int32.
    """
    n, h, d = q.shape
    assert q.stride(2) == 1 and idxk.stride(2) == 1 and w.stride(1) == 1
    assert h % heads_per_rank == 0
    padded = triton.cdiv(max(n_cols, 1), _ROW_ALIGN) * _ROW_ALIGN
    out = torch.empty((n, padded), dtype=torch.float32, device=q.device)[:, :n_cols]
    if n == 0 or n_cols == 0:
        return out
    grid = (n, triton.cdiv(n_cols, _BLOCK_N))
    _index_scores_kernel[grid](
        q,
        w,
        n_vis.to(torch.int32).contiguous(),
        req.to(torch.int32).contiguous(),
        idxk,
        block_table,
        out,
        n_cols,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        idxk.stride(0),
        idxk.stride(1),
        block_table.stride(0),
        out.stride(0),
        ENTRIES=idxk.shape[1],
        H=h,
        D=d,
        HPR=heads_per_rank,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
    )
    return out


class IndexScoresWarmup(VllmJitKernel["IndexScoresWarmup.Key"]):
    """Boot-time compile of :func:`index_scores_triton` for one index-key
    cache. Registered by every index-source layer (whose cache is bound by the
    time the warmup runs); layers whose caches share a specialisation key
    compile once."""

    @dataclass(frozen=True)
    class Key:
        entries: int
        heads: int
        dim: int
        heads_per_rank: int
        block_stride_class: int
        aligned: bool
        cache: torch.Tensor = field(compare=False, hash=False, repr=False)

    def get_warmup_keys(
        self, k_cache_layer, heads: int, dim: int, heads_per_rank: int
    ) -> list["IndexScoresWarmup.Key"]:
        cache = k_cache_layer.kv_cache
        if cache.numel() == 0 or not cache.is_cuda:
            return []
        return [
            self.Key(
                entries=cache.shape[1],
                heads=heads,
                dim=dim,
                heads_per_rank=heads_per_rank,
                block_stride_class=triton_scalar_specialization_rep(cache.stride(0)),
                aligned=cache.data_ptr() % 16 == 0,
                cache=cache,
            )
        ]

    def compile(self, compile_key: "IndexScoresWarmup.Key") -> None:
        # Block 0 (the null block) of the real cache, two queries seeing one
        # and two positions; the launch matches a served one in every
        # specialised argument.
        cache = compile_key.cache
        dev = cache.device
        n = 2
        q = torch.zeros(
            (n, compile_key.heads, compile_key.dim), dtype=torch.bfloat16, device=dev
        )
        w = torch.zeros((n, compile_key.heads), dtype=torch.bfloat16, device=dev)
        n_vis = torch.tensor([1, 2], dtype=torch.int64, device=dev)
        req = torch.zeros(n, dtype=torch.int64, device=dev)
        block_table = torch.zeros((1, 2), dtype=torch.int32, device=dev)
        index_scores_triton(
            q, w, n_vis, req, cache, block_table, 2, compile_key.heads_per_rank
        )


INDEX_SCORES_WARMUP = IndexScoresWarmup()
