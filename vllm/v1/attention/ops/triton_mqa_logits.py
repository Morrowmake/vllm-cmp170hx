# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MQA-logits kernels for the DSA indexer on CUDA devices without DeepGEMM.

DeepGEMM only ships SM90/SM100/SM120 kernels, so on Ampere (and any other
device where ``is_deep_gemm_supported()`` is False) the sparse indexer
computes its logits here. Two entry points mirror the DeepGEMM API used by
``sparse_attn_indexer_kpool``:

* :func:`fp8_mqa_logits` -- prefill, varlen: ``q [M, H, D]`` against a packed
  ``k [N, D]`` with per-row ``[ks, ke)`` bounds; returns ``[M, N]`` fp32.
* :func:`fp8_paged_mqa_logits` -- decode over the paged uint8 cache
  (``D`` e4m3 bytes + 4 fp32-scale bytes per entry) with ``next_n`` Q rows per
  request and ``(B, next_n)`` context lengths; returns ``[B*next_n,
  max_model_len]`` fp32.

Both dequantize e4m3 bytes to bf16 inside the kernel (exact: every e4m3 value
is representable in bf16), run the ``q @ k^T`` product on bf16 tensor cores
with fp32 accumulation, and reduce over heads as DeepGEMM does:
``sum_h w[m, h] * relu(acc[m, h, n]) * k_scale[n]``. Like DeepGEMM with
``clean_logits=False``, entries outside the per-row valid range are only
written (as ``-inf``) inside tiles the kernel visits; the downstream top-k
kernels only read ``[ks, ke)`` / ``[0, seq_len)``.

``VLLM_DSA_INDEXER_REF=1`` swaps in the pure-torch references for debugging.
"""

import functools
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_e4m3 import (
    e4m3_bits_to_bf16_raw,
    e4m3_bits_to_f32_fast,
)

# K tiles are dequantized as raw bf16 bits = value * 2^-120 (see
# e4m3_bits_to_bf16_raw). Half of that scale goes into Q, the other half into
# the head weights, so q @ k^T sits 2^-60 below its true value (fp32 normal
# range for every |logit| >= 2^-66) and w * 2^60 * relu(acc) is exact again.
_TWO60 = tl.constexpr(2.0**60)


_NEG_INF = float("-inf")


def _use_torch_reference() -> bool:
    return os.environ.get("VLLM_DSA_INDEXER_REF", "0") == "1"


def _as_uint8(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.uint8 else x.view(torch.uint8)


def _as_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float8_e4m3fn else x.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Prefill (varlen, unpaged)
# ---------------------------------------------------------------------------


@triton.jit
def _mqa_logits_kernel(
    q_ptr,  # [M, H, D] uint8 (e4m3 bits)
    k_ptr,  # [N, D] uint8 (e4m3 bits)
    kscale_ptr,  # [N] fp32
    w_ptr,  # [M, H] fp32
    ks_ptr,  # [M] int32
    ke_ptr,  # [M] int32
    out_ptr,  # [M, N] fp32
    M,
    N,
    tiles_per_prog,
    stride_qm,
    stride_qh,
    stride_wm,
    stride_om,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program: BLOCK_M query rows x ``tiles_per_prog`` consecutive K
    tiles. Q is dequantized once (scaled by 2^60) and reused across the
    (pipelined) K loop; K tiles are raw bf16 bits (value * 2^-120); tiles no
    row of the block can see (causal / chunk bounds) are skipped."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    ks = tl.load(ks_ptr + rows, mask=rmask, other=2147483647)
    ke = tl.load(ke_ptr + rows, mask=rmask, other=0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    t_begin = tl.maximum(pid_n * tiles_per_prog, tl.min(ks, axis=0) // BLOCK_N)
    t_end = tl.minimum(pid_n * tiles_per_prog + tiles_per_prog, n_tiles)
    t_end = tl.minimum(t_end, tl.cdiv(tl.max(ke, axis=0), BLOCK_N))
    if t_begin >= t_end:
        return

    d = tl.arange(0, D)
    mh = tl.arange(0, BLOCK_M * H)
    q_rows = pid_m * BLOCK_M + mh // H
    q_heads = mh % H
    q_u8 = tl.load(
        q_ptr
        + q_rows[:, None] * stride_qm
        + q_heads[:, None] * stride_qh
        + d[None, :],
        mask=(q_rows < M)[:, None],
        other=0,
    )
    q = (e4m3_bits_to_f32_fast(q_u8) * _TWO60).to(tl.bfloat16)  # [BLOCK_M*H, D]
    w = tl.load(
        w_ptr + rows[:, None] * stride_wm + tl.arange(0, H)[None, :],
        mask=rmask[:, None],
        other=0.0,
    ) * _TWO60  # [BLOCK_M, H]

    for t in tl.range(t_begin, t_end):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        cmask = cols < N
        k_u8 = tl.load(
            k_ptr + cols[:, None] * D + d[None, :], mask=cmask[:, None], other=0
        )
        k = e4m3_bits_to_bf16_raw(k_u8)  # [BLOCK_N, D], value * 2^-120
        acc = tl.dot(q, tl.trans(k))  # [BLOCK_M*H, BLOCK_N] fp32
        acc = tl.reshape(acc, (BLOCK_M, H, BLOCK_N))
        s = tl.sum(tl.maximum(acc, 0.0) * w[:, :, None], axis=1)
        kscale = tl.load(kscale_ptr + cols, mask=cmask, other=0.0)
        s = s * kscale[None, :]
        valid = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
        s = tl.where(valid, s, float("-inf"))
        tl.store(
            out_ptr + rows[:, None] * stride_om + cols[None, :],
            s,
            mask=rmask[:, None] & cmask[None, :],
        )


def _tiles_per_program(num_row_blocks: int, num_tiles: int, cap: int = 16) -> int:
    """K tiles per program: amortize the Q dequant, but keep >= ~4 programs
    per SM so small / causal problems still fill the GPU."""
    target_programs = 4 * _num_sms()
    per = (num_row_blocks * num_tiles) // max(1, target_programs)
    return max(1, min(cap, per))


@functools.cache
def _num_sms() -> int:
    dev = torch.cuda.current_device()
    return torch.cuda.get_device_properties(dev).multi_processor_count


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    block_n: int = 128,
    block_mh: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    tiles_per_prog: int | None = None,
) -> torch.Tensor:
    """Triton varlen MQA logits. See :func:`fp8_mqa_logits` for the contract.

    ``block_mh`` is the number of (row, head) pairs per tile, i.e. the M of
    the underlying ``[block_mh, D] @ [D, block_n]`` MMA; ``block_m =
    block_mh // H`` query rows are handled per program. Defaults were swept
    on an SM80 part (CMP 170HX): 128x64 tiles, 4 warps, K loop pipelined 2
    stages.
    """
    k_fp8, k_scale = kv
    q8 = _as_uint8(q)
    k8 = _as_uint8(k_fp8)
    assert q8.ndim == 3 and k8.ndim == 2
    M, H, D = q8.shape
    N = k8.shape[0]
    assert k8.shape[1] == D and D in (64, 128, 256), (k8.shape, D)
    assert H & (H - 1) == 0 and H >= 8, f"num_heads: power of 2 >= 8, got {H}"
    assert q8.stride(2) == 1
    k8 = k8.contiguous()
    k_scale = k_scale.reshape(-1).contiguous()
    assert k_scale.numel() == N and k_scale.dtype == torch.float32
    weights = weights.contiguous()
    assert weights.shape == (M, H) and weights.dtype == torch.float32
    assert cu_seqlen_ks.dtype == torch.int32 and cu_seqlen_ke.dtype == torch.int32
    cu_seqlen_ks = cu_seqlen_ks.contiguous()
    cu_seqlen_ke = cu_seqlen_ke.contiguous()

    logits = torch.empty((M, N), dtype=torch.float32, device=q8.device)
    if M == 0 or N == 0:
        return logits
    block_m = max(1, block_mh // H)
    num_row_blocks = cdiv(M, block_m)
    num_tiles = cdiv(N, block_n)
    if tiles_per_prog is None:
        tiles_per_prog = _tiles_per_program(num_row_blocks, num_tiles)
    grid = (num_row_blocks, cdiv(num_tiles, tiles_per_prog))
    _mqa_logits_kernel[grid](
        q8,
        k8,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        M,
        N,
        tiles_per_prog,
        q8.stride(0),
        q8.stride(1),
        weights.stride(0),
        logits.stride(0),
        H=H,
        D=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference (fp32 math, DeepGEMM operation order)."""
    k_fp8, k_scale = kv
    qf = _as_e4m3(q).float()
    kf = _as_e4m3(k_fp8).float()
    M, H, D = qf.shape
    N = kf.shape[0]
    score = torch.matmul(qf.reshape(M * H, D), kf.t()).view(M, H, N)
    logits = (score.relu_() * weights.float().unsqueeze(-1)).sum(dim=1)
    logits = logits * k_scale.reshape(-1).float()[None, :]
    ar = torch.arange(N, device=qf.device, dtype=torch.int32)
    mask = (ar[None, :] >= cu_seqlen_ks[:, None]) & (
        ar[None, :] < cu_seqlen_ke[:, None]
    )
    return logits.masked_fill_(~mask, _NEG_INF)


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Non-DeepGEMM replacement for ``deep_gemm.fp8_fp4_mqa_logits`` (FP8 only).

    Args:
        q: ``[M, H, D]`` float8_e4m3fn (or its uint8 view); the per-token Q
            scale is already folded into ``weights``.
        kv: ``(k [N, D] float8_e4m3fn or uint8, k_scale [N] fp32)``.
        weights: ``[M, H]`` fp32.
        cu_seqlen_ks / cu_seqlen_ke: ``[M]`` int32 per-row K bounds.

    Returns:
        ``[M, N]`` fp32 logits, ``-inf`` outside ``[ks, ke)`` within visited
        tiles (undefined elsewhere, like DeepGEMM with ``clean_logits=False``).
    """
    if _use_torch_reference():
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    return fp8_mqa_logits_triton(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


# ---------------------------------------------------------------------------
# Decode (paged, next_n rows per request)
# ---------------------------------------------------------------------------


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,  # [B, next_n, H, D] uint8 (e4m3 bits)
    kv_u8_ptr,  # paged cache as flat uint8
    kv_f32_ptr,  # same memory as flat fp32 (for the per-entry scales)
    w_ptr,  # [B*next_n, H] fp32
    ctx_ptr,  # [B, next_n] int32 (per-row context length)
    bt_ptr,  # [B, max_blocks] int32
    out_ptr,  # [B*next_n, max_model_len] fp32
    max_model_len,
    tiles_per_prog,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_wm,
    stride_ctx_b,
    stride_ctx_t,
    stride_bt,
    stride_om,
    NEXT_N: tl.constexpr,
    NEXT_N_P2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
):
    """One program: one request's next_n Q rows x ``tiles_per_prog``
    consecutive KV pages (one page = one BLOCK_SIZE tile). Q is dequantized
    once; pages past the request's longest row are skipped."""
    b = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = tl.arange(0, NEXT_N_P2)
    tmask = t < NEXT_N
    ctx = tl.load(ctx_ptr + b * stride_ctx_b + t * stride_ctx_t, mask=tmask, other=0)
    n_tiles = tl.cdiv(tl.max(ctx, axis=0), BLOCK_SIZE)
    t_begin = pid_n * tiles_per_prog
    t_end = tl.minimum(t_begin + tiles_per_prog, n_tiles)
    if t_begin >= t_end:
        return

    d = tl.arange(0, D)
    rows = tl.arange(0, NEXT_N_P2 * H)
    rt = rows // H
    rh = rows % H
    q_u8 = tl.load(
        q_ptr
        + b * stride_qb
        + rt[:, None] * stride_qt
        + rh[:, None] * stride_qh
        + d[None, :],
        mask=(rt < NEXT_N)[:, None],
        other=0,
    )
    q = e4m3_bits_to_f32_fast(q_u8).to(tl.bfloat16)  # [NEXT_N_P2*H, D]
    orow = b * NEXT_N + t
    w = tl.load(
        w_ptr + orow[:, None] * stride_wm + tl.arange(0, H)[None, :],
        mask=tmask[:, None],
        other=0.0,
    )  # [NEXT_N_P2, H]
    pos = tl.arange(0, BLOCK_SIZE)

    for tile in tl.range(t_begin, t_end):
        blk = tl.load(bt_ptr + b * stride_bt + tile).to(tl.int64)
        k_base = blk * PAGE_BYTES
        k_u8 = tl.load(kv_u8_ptr + k_base + pos[:, None] * D + d[None, :])
        k = e4m3_bits_to_f32_fast(k_u8).to(tl.bfloat16)  # [BLOCK_SIZE, D]
        kscale = tl.load(kv_f32_ptr + (k_base + BLOCK_SIZE * D) // 4 + pos)
        acc = tl.dot(q, tl.trans(k))  # [NEXT_N_P2*H, BLOCK_SIZE] fp32
        acc = tl.reshape(acc, (NEXT_N_P2, H, BLOCK_SIZE))
        s = tl.sum(tl.maximum(acc, 0.0) * w[:, :, None], axis=1)
        s = s * kscale[None, :]
        kidx = tile * BLOCK_SIZE + pos
        valid = kidx[None, :] < ctx[:, None]
        s = tl.where(valid, s, float("-inf"))
        tl.store(
            out_ptr + orow[:, None] * stride_om + kidx[None, :],
            s,
            mask=tmask[:, None] & (kidx < max_model_len)[None, :],
        )


def _per_row_context_lens(context_lens: torch.Tensor, batch_size: int, next_n: int):
    """Normalize ``context_lens`` to a ``[B, next_n]`` int32 tensor.

    ``(B, next_n)`` is used as-is (row ``t`` sees ``[0, ctx[b, t])``). A 1-D or
    ``(B, 1)`` tensor is the DeepGEMM 1-D convention: row ``t`` of request
    ``b`` sees ``[0, ctx[b] - (next_n - 1 - t))``.
    """
    ctx = context_lens
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(-1)
    assert ctx.dim() == 2 and ctx.shape[0] >= batch_size, ctx.shape
    ctx = ctx[:batch_size]
    if ctx.shape[1] == next_n:
        return ctx.to(torch.int32)
    assert ctx.shape[1] == 1, (ctx.shape, next_n)
    if next_n == 1:
        return ctx.to(torch.int32)
    offs = torch.arange(next_n - 1, -1, -1, device=ctx.device, dtype=torch.int32)
    return (ctx.to(torch.int32) - offs[None, :]).clamp_(min=0)


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    max_seq_len: int | None = None,
    num_warps: int = 2,
    num_stages: int = 2,
    tiles_per_prog: int | None = None,
) -> torch.Tensor:
    """Triton paged MQA logits. See :func:`fp8_paged_mqa_logits`."""
    q8 = _as_uint8(q)
    assert q8.ndim == 4, q8.shape
    B, next_n, H, D = q8.shape
    assert q8.stride(3) == 1
    assert H & (H - 1) == 0 and H >= 8, f"num_heads: power of 2 >= 8, got {H}"
    assert kv_cache.dtype == torch.uint8 and kv_cache.is_contiguous()
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    entry_bytes = (
        kv_cache.numel() // (num_blocks * block_size) if num_blocks else D + 4
    )
    assert entry_bytes == D + 4, (kv_cache.shape, D)
    assert block_size & (block_size - 1) == 0 and block_size >= 16, block_size
    page_bytes = block_size * (D + 4)
    kv_u8 = kv_cache.view(-1)
    kv_f32 = kv_u8.view(torch.float32)
    weights = weights.contiguous()
    assert weights.shape[0] >= B * next_n and weights.shape[1] == H
    assert weights.dtype == torch.float32
    ctx = _per_row_context_lens(context_lens, B, next_n)
    assert block_tables.dtype == torch.int32 and block_tables.stride(1) == 1

    logits = torch.empty(
        (B * next_n, max_model_len), dtype=torch.float32, device=q8.device
    )
    if B == 0:
        return logits
    if max_seq_len is None:
        max_seq_len = min(block_tables.shape[1] * block_size, max_model_len)
    n_tiles = min(cdiv(max_seq_len, block_size), block_tables.shape[1])
    if n_tiles == 0:
        return logits
    if tiles_per_prog is None:
        tiles_per_prog = _tiles_per_program(B, n_tiles, cap=8)
    _paged_mqa_logits_kernel[(B, cdiv(n_tiles, tiles_per_prog))](
        q8,
        kv_u8,
        kv_f32,
        weights,
        ctx,
        block_tables,
        logits,
        max_model_len,
        tiles_per_prog,
        q8.stride(0),
        q8.stride(1),
        q8.stride(2),
        weights.stride(0),
        ctx.stride(0),
        ctx.stride(1),
        block_tables.stride(0),
        logits.stride(0),
        NEXT_N=next_n,
        NEXT_N_P2=triton.next_power_of_2(next_n),
        H=H,
        D=D,
        BLOCK_SIZE=block_size,
        PAGE_BYTES=page_bytes,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits


def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Pure-torch reference (fp32 math, DeepGEMM operation order).

    Adapted from DeepGEMM's ``tests/test_attention.py`` reference via
    ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py``; syncs once per request.
    """
    B, next_n, H, D = q.shape
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    # Page-flat layout: [block_size * D e4m3 bytes | block_size fp32 scales].
    kv = kv_cache.reshape(num_blocks, block_size * (D + 4))
    k_region = block_size * D
    k_all = kv[:, :k_region].view(torch.float8_e4m3fn).float()
    k_all = k_all.view(num_blocks, block_size, D)
    s_all = kv[:, k_region:].contiguous().view(torch.float32)
    s_all = s_all.view(num_blocks, block_size)
    ctx = _per_row_context_lens(context_lens, B, next_n)
    qf = _as_e4m3(q).float()
    logits = torch.full(
        (B * next_n, max_model_len), _NEG_INF, dtype=torch.float32, device=q.device
    )
    ctx_cpu = ctx.cpu()
    for b in range(B):
        limit = int(ctx_cpu[b].max().item())
        if limit <= 0:
            continue
        n_pages = cdiv(limit, block_size)
        pages = block_tables[b, :n_pages].long()
        kb = k_all[pages].reshape(-1, D)  # [L, D]
        sb = s_all[pages].reshape(-1)  # [L]
        L = kb.shape[0]
        score = torch.matmul(qf[b].reshape(next_n * H, D), kb.t()).view(next_n, H, L)
        w = weights[b * next_n : (b + 1) * next_n].float()
        s = (score.relu_() * w.unsqueeze(-1)).sum(dim=1) * sb[None, :]
        kidx = torch.arange(L, device=q.device, dtype=torch.int32)
        s = s.masked_fill_(kidx[None, :] >= ctx[b][:, None], _NEG_INF)
        L = min(L, max_model_len)
        logits[b * next_n : (b + 1) * next_n, :L] = s[:, :L]
    return logits


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Non-DeepGEMM replacement for ``deep_gemm.fp8_fp4_paged_mqa_logits``.

    Args:
        q: ``[B, next_n, H, D]`` float8_e4m3fn (or uint8 view).
        kv_cache: ``[num_blocks, block_size, 1, D+4]`` (or 3-D) uint8, page-
            flat: ``block_size * D`` e4m3 bytes (entry-major) followed by
            ``block_size`` fp32 scales (vLLM's ``indexer_k_quant_and_cache`` /
            kpool writer layout).
        weights: ``[B*next_n, H]`` fp32.
        context_lens: ``[B, next_n]`` int32 (row ``t`` of request ``b`` sees
            ``[0, context_lens[b, t])``); ``[B]`` / ``[B, 1]`` follow the
            DeepGEMM 1-D convention (see :func:`_per_row_context_lens`).
        block_tables: ``[B, max_blocks]`` int32.
        max_model_len: width of the output.
        max_seq_len: optional upper bound on ``context_lens`` (avoids
            launching tiles past the longest request); any over-estimate is
            safe. No scheduler metadata is needed on this path.

    Returns:
        ``[B*next_n, max_model_len]`` fp32; ``-inf`` for ``k >= ctx`` inside
        visited tiles, undefined beyond (``clean_logits=False`` semantics).
    """
    if _use_torch_reference():
        return fp8_paged_mqa_logits_torch(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    return fp8_paged_mqa_logits_triton(
        q, kv_cache, weights, context_lens, block_tables, max_model_len, max_seq_len
    )
