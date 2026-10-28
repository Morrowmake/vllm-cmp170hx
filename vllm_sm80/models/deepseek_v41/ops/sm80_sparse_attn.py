# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 sparse attention with sinks on sm_80 (Triton).

One program per (query, block of 16 heads) walks the query's window slots in
the layer's sliding BF16 cache, then its selected compressed positions in the
kv source's BF16 cache (paged through that request's block table), with the
online softmax of the reference ``sparse_attn`` kernel: fp32 scores times the
softmax scale, a finite running-max floor (-1e30) so a query with no valid
row ends at zero output, probabilities rounded to BF16 before they weight the
rows (the running sum keeps them in fp32), and the per-head sink added to the
denominator at the end. Every paged offset is int64. No split-K, no atomics.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_H = 16
_BLOCK_N = 32


@triton.jit
def _attend(q, kv, ok, m, l, acc, scale):
    s = tl.dot(q, tl.trans(kv)) * scale
    s = tl.where(ok[None, :], s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, axis=1))
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    return m_new, l, acc


@triton.jit(do_not_specialize=["n_window", "n_comp"])
def _sparse_attn_kernel(
    q_ptr,
    swa_ptr,
    swa_idx_ptr,
    comp_ptr,
    comp_bt_ptr,
    comp_idx_ptr,
    req_ptr,
    sink_ptr,
    out_ptr,
    n_heads,
    n_window,
    n_comp,
    scale,
    stride_qt,
    stride_qh,
    stride_sb,
    stride_se,
    stride_wi,
    stride_cb,
    stride_ce,
    stride_bt,
    stride_ci,
    stride_ot,
    stride_oh,
    SWA_E: tl.constexpr,
    COMP_E: tl.constexpr,
    HAS_COMP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = heads < n_heads
    d = tl.arange(0, D)
    q = tl.load(
        q_ptr + t * stride_qt + heads[:, None] * stride_qh + d[None, :],
        mask=hmask[:, None],
        other=0.0,
    )
    m = tl.full([BLOCK_H], -1.0e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    n = tl.arange(0, BLOCK_N)
    for n0 in range(0, n_window, BLOCK_N):
        cols = n0 + n
        slot = tl.load(swa_idx_ptr + t * stride_wi + cols, mask=cols < n_window, other=-1)
        ok = slot >= 0
        s64 = tl.maximum(slot, 0).to(tl.int64)
        off = (s64 // SWA_E) * stride_sb + (s64 % SWA_E) * stride_se
        kv = tl.load(swa_ptr + off[:, None] + d[None, :], mask=ok[:, None], other=0.0)
        m, l, acc = _attend(q, kv, ok, m, l, acc, scale)
    if HAS_COMP:
        req = tl.load(req_ptr + t).to(tl.int64)
        for n0 in range(0, n_comp, BLOCK_N):
            cols = n0 + n
            j = tl.load(comp_idx_ptr + t * stride_ci + cols, mask=cols < n_comp, other=-1)
            ok = j >= 0
            j64 = tl.maximum(j, 0).to(tl.int64)
            blk = tl.load(comp_bt_ptr + req * stride_bt + j64 // COMP_E, mask=ok, other=0)
            off = blk.to(tl.int64) * stride_cb + (j64 % COMP_E) * stride_ce
            kv = tl.load(comp_ptr + off[:, None] + d[None, :], mask=ok[:, None], other=0.0)
            m, l, acc = _attend(q, kv, ok, m, l, acc, scale)
    sink = tl.load(sink_ptr + heads, mask=hmask, other=float("-inf"))
    l = l + tl.exp(sink - m)
    o = acc / l[:, None]
    tl.store(
        out_ptr + t * stride_ot + heads[:, None] * stride_oh + d[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


def sparse_attention_triton(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    comp_cache: torch.Tensor | None,
    comp_block_table: torch.Tensor | None,
    comp_local: torch.Tensor | None,
    req: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """See ``sm80_attn_core.sparse_attention_torch`` (same contract)."""
    n, h, d = q.shape
    out = torch.empty((n, h, d), dtype=torch.bfloat16, device=q.device)
    if n == 0:
        return out
    assert q.stride(2) == 1 and swa_cache.stride(2) == 1
    assert swa_indices.stride(1) == 1
    has_comp = comp_local is not None and comp_local.shape[1] > 0
    if has_comp:
        assert comp_cache is not None and comp_block_table is not None
        assert comp_cache.stride(2) == 1 and comp_local.stride(1) == 1
        comp_args = (
            comp_cache,
            comp_block_table,
            comp_local,
            comp_cache.stride(0),
            comp_cache.stride(1),
            comp_block_table.stride(0),
            comp_local.stride(0),
            comp_local.shape[1],
            comp_cache.shape[1],
        )
    else:
        comp_args = (swa_cache, swa_indices, swa_indices, 0, 0, 0, 0, 0, 1)
    comp, comp_bt, comp_idx, s_cb, s_ce, s_bt, s_ci, n_comp, comp_e = comp_args
    grid = (n, triton.cdiv(h, _BLOCK_H))
    _sparse_attn_kernel[grid](
        q,
        swa_cache,
        swa_indices,
        comp,
        comp_bt,
        comp_idx,
        req.to(torch.int32).contiguous(),
        attn_sink,
        out,
        h,
        swa_indices.shape[1],
        n_comp,
        scale,
        q.stride(0),
        q.stride(1),
        swa_cache.stride(0),
        swa_cache.stride(1),
        swa_indices.stride(0),
        s_cb,
        s_ce,
        s_bt,
        s_ci,
        out.stride(0),
        out.stride(1),
        SWA_E=swa_cache.shape[1],
        COMP_E=comp_e,
        HAS_COMP=has_comp,
        D=d,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        num_warps=8,
    )
    return out
