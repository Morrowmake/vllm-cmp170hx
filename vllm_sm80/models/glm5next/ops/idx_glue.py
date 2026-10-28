# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 decode folds for the GLM-5.3-Flash MLA indexer chain and the MoE tail.

Opt-in with ``VLLM_AMPERE_DECODE_IDX_GLUE=1``; ``VLLM_GLM5_DECODE_IDX_GLUE_PARTS``
selects the folds (default: all five). With the switch unset nothing here is
called and every call site keeps its existing code path.

=========  ==================================================================
part       what it replaces (per MLA layer unless stated)
=========  ==================================================================
weights    ``hidden.float()`` + fp32 sgemm split-K + splitKreduce that
           recompute the 32 indexer head weights the wk+weights GEMM already
           produced in bf16: the same thin GEMM now stores those columns in
           fp32. The k columns keep the bf16 store they had (bitwise); the
           weight columns change only accumulation order (tensor-core
           fp32 accumulate in a fixed split-K order instead of cuBLAS's fp32
           sgemm order), so they are NOT bitwise vs the sgemm -- floor check.
glue       ``positions.to(int32)`` for the kpool update, the ``-1`` fill of
           the top-k buffer, ``pool_topk.to(int64)``, ``positions.to(int32)``
           + ``1`` for the tail seq lens and the strided copy of the expanded
           indices into the buffer. The casts move into the kernels' loads and
           the fill/copy into one expand kernel that writes the buffer rows
           directly. Bitwise.
fwht       ``weights * q_scale * softmax_scale * n_head**-0.5`` (one inductor
           kernel) folded into the Hadamard + fp8 quant kernel, which also runs
           on 4-row CTAs instead of 32-row ones (M*32/4 CTAs instead of M).
           Same fp32 operations in the same order; bitwise.
cache      ``concat_and_cache_mla`` (latent cache write) merged into the sparse
           top-k index remap launch that precedes ``_mla_sparse``. Pure data
           movement plus the same integer remap; bitwise. Decode-only batches,
           bf16 cache, no HiSparse/PCP/DCP, never under a piecewise capture.
moesum     routed ``moe_sum`` (fp32 slot-order sum, bf16 round) fused with the
           shared-expert add that follows it: the sum is rounded to bf16
           exactly as before, then added in fp32 and rounded, which is what the
           out-of-place bf16 add did. Bitwise. MoE layers, M <= the decode
           MoE bound.
=========  ==================================================================

The gate for every part (``use_idx_glue``, the parts list and the banner) is
``vllm_sm80/kernels/idx_glue_gate.py``. The kernels of the two parts shared code
calls live with their callers: ``cache`` in
``vllm_sm80/kernels/attention/triton_mla_remap_cache.py`` (the Triton sparse-MLA
backend) and ``moesum`` in ``vllm_sm80/kernels/moe/sum.py`` (the MoE runner and
the Marlin experts).
"""

from __future__ import annotations

import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import _fwht_stage
from vllm_sm80.kernels.attention.triton_e4m3 import (
    store_fp8_e4m3,
    triton_fp8_e4m3_native,
)
from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# weights: wk+weights thin GEMM with a bf16 low block and an fp32 high block
# ---------------------------------------------------------------------------
@triton.jit
def _add_rn(a, b):
    # Plain fp32 add the compiler cannot fold into an mma accumulator or
    # reassociate (a tensor-core mma adding into a non-zero accumulator
    # truncates on sm_80).
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _kahan_add(s, c, v):
    """(s, c) += v with Kahan compensation; returns the new (s, c)."""
    y = _sub_rn(v, c)
    t = _add_rn(s, y)
    c = _sub_rn(_sub_rn(t, s), y)
    return t, c


@triton.jit
def _thin_gemm_dual_kernel(
    X, W, Y, Y2, P, LOCK,
    M, N, K, N_LO,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_y2m, stride_y2n,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    TILED_M: tl.constexpr,
    ROTATE_K: tl.constexpr = False,
    HI_SUB_K: tl.constexpr = 0,
    HI_KAHAN_LOOP: tl.constexpr = False,
    HI_KAHAN_REDUCE: tl.constexpr = False,
):
    """``thin_gemm.thin_gemm_v74._thin_gemm_kernel`` with a split store.

    Columns ``< N_LO`` run the incumbent's main loop and fixed-order split-K
    reduction line for line and are stored bf16 into ``Y`` (bitwise the
    incumbent's store). Columns ``>= N_LO`` replace a fp32 sgemm, so they are
    accumulated for fp32 accuracy and stored unrounded into ``Y2``: a fresh
    mma accumulator per ``HI_SUB_K``-wide k slice, summed in fp32 registers
    (an sm_80 mma that adds products into a non-zero accumulator truncates
    them to the running sum's exponent), and a Kahan-compensated split-K
    reduce. ``HI_SUB_K == 0`` keeps the incumbent loop for them too.
    ``N_LO % BLOCK_N == 0`` (checked by the wrapper), so every N tile is
    entirely one or the other.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    if TILED_M:
        pid_m = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        pid_m = 0
        offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    x_m = tl.where(m_mask, offs_m, 0)
    w_n = tl.where(n_mask, offs_n, 0)

    x_ptrs = X + x_m[:, None] * stride_xm + \
        (pid_k * BLOCK_K + offs_k)[None, :] * stride_xk
    w_ptrs = W + w_n[:, None] * stride_wn + \
        (pid_k * BLOCK_K + offs_k)[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step = BLOCK_K * SPLIT_K
    n_iter = tl.cdiv(K, step)
    is_lo = pid_n * BLOCK_N < N_LO
    if ROTATE_K and is_lo:
        # bf16 columns under the thin GEMM's K-start rotation
        # (thin_gemm_v74._thin_gemm_kernel, line for line; EVEN_K only).
        shift = (pid_n * 3) % n_iter
        for i in range(n_iter):
            j = i + shift
            j = tl.where(j >= n_iter, j - n_iter, j)
            x = tl.load(x_ptrs + j * step * stride_xk, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs + j * step * stride_wk, mask=n_mask[:, None], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    elif is_lo or HI_SUB_K == 0:
        # bf16 columns: the incumbent's loop, line for line (bitwise the
        # merged GEMM's k columns).
        for i in range(n_iter):
            if EVEN_K:
                x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
                w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            else:
                k_now = i * step + pid_k * BLOCK_K + offs_k
                k_mask = k_now < K
                x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
            x_ptrs += step * stride_xk
            w_ptrs += step * stride_wk
    else:
        # fp32 columns: a fresh mma accumulator per HI_SUB_K-wide slice,
        # summed in fp32 registers (optionally Kahan-compensated), so no
        # product is truncated to the running sum's exponent inside the mma.
        comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        offs_s = tl.arange(0, HI_SUB_K)
        for i in range(n_iter):
            for j in tl.static_range(BLOCK_K // HI_SUB_K):
                kk = i * step + pid_k * BLOCK_K + j * HI_SUB_K + offs_s
                xs = X + x_m[:, None] * stride_xm + kk[None, :] * stride_xk
                ws = W + w_n[:, None] * stride_wn + kk[None, :] * stride_wk
                if EVEN_K:
                    x = tl.load(xs, mask=m_mask[:, None], other=0.0)
                    w = tl.load(ws, mask=n_mask[:, None], other=0.0)
                else:
                    k_mask = kk < K
                    x = tl.load(xs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                    w = tl.load(ws, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
                blk = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
                if HI_KAHAN_LOOP:
                    acc, comp = _kahan_add(acc, comp, blk)
                else:
                    acc = _add_rn(acc, blk)
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y2_ptrs = Y2 + offs_m[:, None] * stride_y2m + \
        (offs_n - N_LO)[None, :] * stride_y2n
    if SPLIT_K == 1:
        if is_lo:
            tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=m_mask[:, None])
        else:
            tl.store(y2_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
    else:
        p_mask = m_mask[:, None] & n_mask[None, :]
        p_off = offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        tl.store(P + pid_k * stride_pk + p_off, acc, mask=p_mask,
                 cache_modifier=".cg")
        if TILED_M:
            lock = LOCK + pid_m * tl.num_programs(0) + pid_n
        else:
            lock = LOCK + pid_n
        arrived = tl.atomic_add(lock, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if HI_KAHAN_REDUCE and not is_lo:
                tc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in tl.static_range(SPLIT_K):
                    v = tl.load(P + k * stride_pk + p_off, mask=p_mask,
                                other=0.0, cache_modifier=".cv")
                    tot, tc = _kahan_add(tot, tc, v)
            else:
                for k in tl.static_range(SPLIT_K):
                    tot += tl.load(P + k * stride_pk + p_off, mask=p_mask,
                                   other=0.0, cache_modifier=".cv")
            if is_lo:
                tl.store(y_ptrs, tot.to(Y.dtype.element_ty), mask=p_mask)
            else:
                tl.store(y2_ptrs, tot, mask=p_mask)
            tl.atomic_xchg(lock, 0, sem="release", scope="gpu")


def thin_gemm_dual_supported(x: torch.Tensor, w: torch.Tensor, n_lo: int) -> bool:
    """Shape gate for :func:`thin_gemm_dual` (host only, no allocation)."""
    from vllm_sm80.kernels.thin_gemm import impl, thin_gemm_supported

    if not thin_gemm_supported(x, w, None):
        return False
    M, K = x.shape
    N = w.shape[0]
    if M < 1 or not (0 < n_lo < N):
        return False
    BLOCK_N = impl()._select_config(M, N, K)[1]
    return n_lo % BLOCK_N == 0


# fp32 (high) block accumulation: (k width of each fresh mma accumulator, Kahan
# in the k loop, Kahan in the split-K reduce); a width of 0 keeps the
# incumbent's running accumulator. On real decode inputs (M 4-32, K 4096) the
# running accumulator was 5.6x the fp32 sgemm's mean error against an exact
# product; (64, False, True) is 0.62x mean, 0.51x max, for <= 0.7 us per call.
_HI_ACC = (64, False, True)


def thin_gemm_dual(
    x: torch.Tensor, w: torch.Tensor, n_lo: int, _hi_acc=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``F.linear(x, w)`` split at column ``n_lo``: (bf16 [M, n_lo], fp32 [M, N-n_lo]).

    The bf16 block is bitwise what ``thin_gemm(x, w)[:, :n_lo]`` returns and is
    returned as a view of an ``[M, N]`` buffer so its strides match that slice.
    The fp32 block is accumulated as described in ``_thin_gemm_dual_kernel``;
    its error against an exact product is no worse than the fp32 sgemm it
    replaces (``tests/kernels/test_ampere_idx_glue.py``).
    """
    from vllm_sm80.kernels.thin_gemm import impl

    tg = impl()
    _dummy_fp32, _locks, _partials, _select_config = (
        tg._dummy_fp32, tg._locks, tg._partials, tg._select_config)
    M, K = x.shape
    N = w.shape[0]
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = _select_config(M, N, K)
    assert n_lo % BLOCK_N == 0, (n_lo, BLOCK_N)
    hi = _HI_ACC if _hi_acc is None else _hi_acc
    lo_buf = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    y_lo = lo_buf[:, :n_lo]
    y_hi = torch.empty((M, N - n_lo), dtype=torch.float32, device=x.device)
    if M == 0:
        return y_lo, y_hi
    even_k = (K % (BLOCK_K * SPLIT_K)) == 0
    tiles_n = triton.cdiv(N, BLOCK_N)
    tiles_m = triton.cdiv(M, BLOCK_M)
    if SPLIT_K == 1:
        p = _dummy_fp32(x.device)
        sp = (0, 0, 0)
        lock = _locks(1, x.device)
    else:
        p = _partials(SPLIT_K, M, N, x.device)
        sp = (p.stride(0), p.stride(1), p.stride(2))
        lock = _locks(tiles_n * tiles_m, x.device)
    _thin_gemm_dual_kernel[(tiles_n, SPLIT_K, tiles_m)](
        x, w, y_lo, y_hi, p, lock,
        M, N, K, n_lo,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y_lo.stride(0), y_lo.stride(1),
        y_hi.stride(0), y_hi.stride(1),
        sp[0], sp[1], sp[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
        EVEN_K=even_k, TILED_M=(tiles_m > 1),
        # the bf16 columns follow the thin GEMM's K-start rotation (74-SM module)
        ROTATE_K=(hasattr(tg, "_rotate_k")
                  and tg._rotate_k(even_k, SPLIT_K, BLOCK_M, BLOCK_N)),
        HI_SUB_K=min(hi[0], BLOCK_K), HI_KAHAN_LOOP=hi[1], HI_KAHAN_REDUCE=hi[2],
        num_warps=num_warps, num_stages=num_stages,
    )
    return y_lo, y_hi


# ---------------------------------------------------------------------------
# fwht: Hadamard-128 + ue8m0 fp8 quant with the head-weight scale folded in
# ---------------------------------------------------------------------------
@triton.jit
def _fwht_quant_wscale_kernel(
    q_ptr,
    qout_ptr,
    w_ptr,
    wout_ptr,
    n_rows,
    w_s0,
    w_s1,
    wscale,
    NH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    FP8_NATIVE: tl.constexpr,
):
    """``kpool_compress._fwht_quant_kernel`` + the indexer weight scaling.

    Rows are ``token * NH + head``. The rotation/quant body is the incumbent's
    (same butterflies via the shared ``_fwht_stage``, same per-row max, same
    ue8m0 scale); every op is per row, so the row block size does not change
    any value. The per-row scale is not stored: its only consumer was
    ``weights * q_scale * scale``, computed here as ``(w * s) * wscale`` in
    fp32, the incumbent inductor kernel's order.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    offs = tl.arange(0, 128)
    x = tl.load(
        q_ptr + rows[:, None] * 128 + offs[None, :], mask=rmask[:, None], other=0.0
    ).to(tl.float32)

    N: tl.constexpr = BLOCK_R * 128
    x = tl.reshape(x, (N,))
    x = _fwht_stage(x, N, BLOCK_R * 64, 1)
    x = _fwht_stage(x, N, BLOCK_R * 32, 2)
    x = _fwht_stage(x, N, BLOCK_R * 16, 4)
    x = _fwht_stage(x, N, BLOCK_R * 8, 8)
    x = _fwht_stage(x, N, BLOCK_R * 4, 16)
    x = _fwht_stage(x, N, BLOCK_R * 2, 32)
    x = _fwht_stage(x, N, BLOCK_R, 64)
    x = x * 0.08838834764831845  # 1/sqrt(128), exact in fp32

    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, 128))

    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)

    store_fp8_e4m3(
        qout_ptr + rows[:, None] * 128 + offs[None, :], y, rmask[:, None], FP8_NATIVE
    )

    tok = rows // NH
    head = rows % NH
    w = tl.load(w_ptr + tok * w_s0 + head * w_s1, mask=rmask, other=0.0)
    tl.store(wout_ptr + rows, (w * scale) * wscale, mask=rmask)


def _fwht_block_rows(n_rows: int) -> tuple[int, int]:
    """(BLOCK_R, num_warps). The incumbent runs 32 rows per CTA, i.e. M CTAs
    at 32 heads; small row blocks give M*8 CTAs at decode sizes."""
    if n_rows <= 4096:
        return 4, 1
    return 32, 2


def fwht128_quant_fp8_wscale(
    q: torch.Tensor, weights: torch.Tensor, wscale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``fwht128_quant_fp8(q)`` + ``weights * q_scale * wscale``.

    Args:
        q: ``[tokens * NH, 128]`` bf16, contiguous (head-major within a token).
        weights: ``[tokens, NH]`` fp32, any strides.
        wscale: python float whose fp32 rounding is a power of two
            (:func:`is_pow2_float`); the kernel multiplies by that fp32 value.

    Returns:
        (q_fp8 ``[tokens * NH, 128]`` float8_e4m3fn,
         scaled weights ``[tokens, NH]`` fp32 contiguous).
    """
    assert q.ndim == 2 and q.shape[1] == 128 and q.dtype == torch.bfloat16
    assert q.is_contiguous()
    assert weights.ndim == 2 and weights.dtype == torch.float32
    tokens, nh = weights.shape
    n_rows = q.shape[0]
    assert n_rows == tokens * nh, (n_rows, tokens, nh)
    q_fp8 = torch.empty((n_rows, 128), dtype=torch.float8_e4m3fn, device=q.device)
    w_out = torch.empty((tokens, nh), dtype=torch.float32, device=q.device)
    if n_rows == 0:
        return q_fp8, w_out
    block_r, num_warps = _fwht_block_rows(n_rows)
    native = triton_fp8_e4m3_native()
    _fwht_quant_wscale_kernel[(triton.cdiv(n_rows, block_r),)](
        q,
        q_fp8 if native else q_fp8.view(torch.uint8),
        weights,
        w_out,
        n_rows,
        weights.stride(0),
        weights.stride(1),
        fp32_value(float(wscale)),
        NH=nh,
        BLOCK_R=block_r,
        FP8_NATIVE=native,
        num_warps=num_warps,
    )
    return q_fp8, w_out


def fp32_value(x: float) -> float:
    """``x`` rounded to fp32, as a Python float."""
    import struct

    return struct.unpack("f", struct.pack("f", x))[0]


def is_pow2_float(x: float) -> bool:
    """Is ``x``, rounded to fp32, a power of two?

    The production constant ``head_dim**-0.5 * n_head**-0.5`` is
    0.015625000000000003 in fp64 and exactly 2**-6 in fp32. With a power-of-two
    fp32 constant, ``(w * q_scale) * c`` is exact (q_scale is ue8m0), and it is
    the same fp32 value whether the unfused leaf multiplies by the fp32 or the
    fp64 constant: the fp64 product is ``v * (1 + 2**-52 k)`` for an fp32
    value ``v`` and rounds back to ``v``.
    """
    import math

    if not math.isfinite(x) or x <= 0.0:
        return False
    x32 = fp32_value(x)
    if not (x32 > 0.0) or not math.isfinite(x32):
        return False
    m, _ = math.frexp(x32)
    return m == 0.5


# ---------------------------------------------------------------------------
# glue: expand pools + tail straight into the top-k buffer rows
# ---------------------------------------------------------------------------
@triton.jit
def _expand_into_buffer_kernel(
    pool_ids_ptr,  # [n_rows, n_groups] int32 (pool top-k, -1 = none)
    pos_ptr,  # [n_rows] any int (token positions)
    out_ptr,  # topk_indices_buffer, int32
    n_rows,  # rows that get indices; rows [n_rows, grid0) get -1
    topk,  # n_groups * POOL_SIZE
    out_cols,  # topk + POOL_SIZE - 1
    width,  # buffer row width
    pid_s0,
    out_s0,
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """``buf[:T] = -1``; ``buf[:n, :out_cols] = expand(pool_ids, pos + 1)``.

    The expansion is ``kpool_compress._expand_pools_and_append_tail_kernel``
    with the int64 pool-id cast and the int32 ``pos + 1`` folded into the loads
    (pool ids < 2**29, positions < 2**31, so no value changes).
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    wmask = cols < width
    result = tl.full((BLOCK_COLS,), -1, tl.int32)
    if row < n_rows:
        seq_len = tl.load(pos_ptr + row).to(tl.int32) + 1
        pool_len = seq_len // POOL_SIZE
        tail_start = pool_len * POOL_SIZE
        tail_count = seq_len - tail_start

        mask = cols < out_cols
        is_history = cols < topk
        g = cols // POOL_SIZE
        o = cols % POOL_SIZE
        pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history,
                      other=-1)
        hist_val = (pid * POOL_SIZE + o).to(tl.int32)
        hist_out = tl.where(pid >= 0, hist_val, -1)

        tail_off = cols - topk
        is_tail = (tail_off >= 0) & (tail_off < tail_count)
        tail_val = (tail_start + tail_off).to(tl.int32)
        tail_out = tl.where(is_tail, tail_val, -1)

        result = tl.where(is_history, hist_out, tail_out)
        result = tl.where(mask, result, -1)
    tl.store(out_ptr + row * out_s0 + cols, result, mask=wmask)


def expand_pools_into_buffer(
    pool_topk: torch.Tensor,
    positions: torch.Tensor,
    buf: torch.Tensor,
    n_fill_rows: int,
    pool_size: int,
) -> None:
    """In place: rows ``[0, n)`` of ``buf`` get the expanded indices (``n`` =
    ``pool_topk.shape[0]``), every other column and rows ``[n, n_fill_rows)``
    get -1. Equivalent to the fill + cast + expand + copy sequence it replaces.
    """
    assert pool_topk.ndim == 2 and pool_topk.dtype == torch.int32
    assert buf.dtype == torch.int32 and buf.stride(1) == 1
    n, n_groups = pool_topk.shape
    assert positions.shape[0] >= n and positions.is_contiguous()
    assert n <= n_fill_rows <= buf.shape[0]
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    width = buf.shape[1]
    assert out_cols <= width
    if n_fill_rows == 0:
        return
    BLOCK_COLS = 128
    _expand_into_buffer_kernel[(n_fill_rows, triton.cdiv(width, BLOCK_COLS))](
        pool_topk,
        positions,
        buf,
        n,
        topk,
        out_cols,
        width,
        pool_topk.stride(0),
        buf.stride(0),
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
    )
