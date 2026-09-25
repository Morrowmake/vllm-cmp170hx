# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# mypy: ignore-errors
#
# The kernels below are derived from the flash-linear-attention project
# (solve_tril, chunk_delta_h, the KDA chunk kernels). The original source code
# was licensed under the MIT license and included the following copyright
# notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
"""KDA chunked prefill for GLM-5.3-Flash with 64 heads per card, sm_80.

``VLLM_GLM5_PP_KDA_PREFILL=1`` replaces ``chunk_kda_with_fused_gate`` of
``vllm/models/glm5next/nvidia/ops/third_party/kda`` at the prefill call site
of ``vllm/models/glm5next/common/kda.py`` when all 64 heads live on one card
(pipeline parallel, TP=1). Same signature, same outputs: ``o [1, T, H, V]``
in the dtype of ``v`` and ``final_state [N, H, V, K]`` fp32 (K contiguous,
the layout the decode kernels read from the state pool).

Six launches instead of nine, and more precision where the recurrence needs
it:

  ``kda_gate_kernel``   gate ``lower_bound * sigmoid(exp(A_log) (g + bias))``
                        and its chunk-local cumsum (log2 units) in [64, 32]
                        column slices on one warp; also the gate at each
                        chunk's last valid row (the state pass's decay).
  ``kda_kkt_kernel``    per (chunk, 16-row block, head): l2-normalises q and
                        k on load, ``A = beta k k^T`` (strictly lower) and
                        ``Aqk = scale q k^T`` (lower) with the decay folded
                        into the operands (finite, normal references per
                        block), as bf16x3 tensor-core products; writes
                        ``qg`` (bf16) and ``kg`` as a bf16 hi/lo pair.
  ``merge_16x16_to_64x64_inverse_kernel``  ``(I + A)^-1`` per chunk: the four
                        diagonal 16x16 inverses by column substitution, the
                        off-diagonal blocks as bf16x3 products; fp32 out.
  ``kda_wu_kernel``     ``w = Ai beta k exp2(g)`` (bf16) and
                        ``u = Ai beta v`` (fp32), bf16x3 products.
  ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``  the sequential state
                        pass with ``h`` split into bf16 hi + lo for
                        ``w h^T`` and ``kg``, ``v_new`` split for the update:
                        the only bf16 rounding the recurrence sees is ``w``.
  ``chunk_gla_fwd_kernel_o``  ``o = qg h^T + Aqk v_new``.

Measured on one CMP 170HX (70 SMs), CUDA-graph replay of a rotation of real
inputs: 1.65x the incumbent over the production prefill chunks (weighted
geomean of 2304-token continuing and first chunks and a 1282-token tail).
Error against an exact fp64 token-sequential recomputation is at most the
incumbent's on every output (final_state well below it).

Every launch configuration is fixed (no runtime autotune), so the result is a
pure function of the inputs and bitwise reproducible run to run and boot to
boot. No float atomics. Nothing is allocated outside the call and there is no
host sync apart from ``prepare_chunk_indices``, which is cached per
``cu_seqlens`` tensor (once per step, shared with the upstream path).
"""

import itertools

import torch

from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import input_guard
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

FLA_CHUNK_SIZE = 64
RCP_LN2 = 1.4426950216
RCP_LN2_C = tl.constexpr(RCP_LN2)
# Plain fp32 exponentials (the fast-ops variants are off in production).
exp = tl.exp
exp2 = tl.exp2
log = tl.log

# chunk_gla_fwd_kernel_o: one fixed config for every shape (the config the
# timing autotuner settled on at 64 heads for the production chunk sizes).
O_BK, O_BV, O_WARPS, O_STAGES = 64, 128, 4, 2


# ============================================================================
# (I + A)^-1 per 64-token chunk
# ============================================================================

@triton.jit
def _mm3m(a, b):
    # ~fp32 product of fp32 16x16 tiles from bf16 tensor-core dots (hi*hi + hi*lo + lo*hi)
    a_hi = a.to(tl.bfloat16, fp_downcast_rounding="rtne")
    a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    b_hi = b.to(tl.bfloat16, fp_downcast_rounding="rtne")
    b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    acc = tl.dot(a_lo, b_hi)
    acc = tl.dot(a_hi, b_lo, acc)
    return tl.dot(a_hi, b_hi, acc)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_A_22 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_A_33 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_A_44 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    # [16, 16] diagonal inverses X = (I + A_cc)^-1 by column substitution, right to left:
    # X[:, m] = e_m - X A_cc[:, m] (A_cc strictly lower, so only the already-final columns
    # j > m contribute).  The reduction runs along axis 1 (lanes: shuffles only, no
    # shared-memory round trip or barrier), and the column loads do not depend on the
    # loop-carried X (L1 hits).  The four chains are interleaved.  Rows past T load 0.
    o_r = tl.arange(0, 16)
    b_Ai_11 = tl.where(m_I, 1.0, 0.0)
    b_Ai_22 = tl.where(m_I, 1.0, 0.0)
    b_Ai_33 = tl.where(m_I, 1.0, 0.0)
    b_Ai_44 = tl.where(m_I, 1.0, 0.0)
    p_c = A + (i_t * BT + o_r) * H * BT
    for s in range(15):
        m = 14 - s
        a_1 = tl.load(p_c + m, mask=(i_t * BT + o_r) < T, other=0.0)[None, :]
        a_2 = tl.load(p_c + 16 * H * BT + 16 + m, mask=(i_t * BT + 16 + o_r) < T, other=0.0)[None, :]
        a_3 = tl.load(p_c + 32 * H * BT + 32 + m, mask=(i_t * BT + 32 + o_r) < T, other=0.0)[None, :]
        a_4 = tl.load(p_c + 48 * H * BT + 48 + m, mask=(i_t * BT + 48 + o_r) < T, other=0.0)[None, :]
        sel = (o_i == m)[None, :]
        e_m = tl.where(o_i == m, 1.0, 0.0)
        b_Ai_11 = tl.where(sel, (e_m - tl.sum(b_Ai_11 * a_1, 1))[:, None], b_Ai_11)
        b_Ai_22 = tl.where(sel, (e_m - tl.sum(b_Ai_22 * a_2, 1))[:, None], b_Ai_22)
        b_Ai_33 = tl.where(sel, (e_m - tl.sum(b_Ai_33 * a_3, 1))[:, None], b_Ai_33)
        b_Ai_44 = tl.where(sel, (e_m - tl.sum(b_Ai_44 * a_4, 1))[:, None], b_Ai_44)

    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -_mm3m(
        _mm3m(b_Ai_22, b_A_21),
        b_Ai_11
    )
    b_Ai_32 = -_mm3m(
        _mm3m(b_Ai_33, b_A_32),
        b_Ai_22
    )
    b_Ai_43 = -_mm3m(
        _mm3m(b_Ai_44, b_A_43),
        b_Ai_33
    )

    b_Ai_31 = -_mm3m(
        b_Ai_33,
        _mm3m(b_A_31, b_Ai_11)
        + _mm3m(b_A_32, b_Ai_21)
    )
    b_Ai_42 = -_mm3m(
        b_Ai_44,
        _mm3m(b_A_42, b_Ai_22)
        + _mm3m(b_A_43, b_Ai_32)
    )
    b_Ai_41 = -_mm3m(
        b_Ai_44,
        _mm3m(b_A_41, b_Ai_11)
        + _mm3m(b_A_42, b_Ai_21)
        + _mm3m(b_A_43, b_Ai_31)
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_11,
        b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """(I + A)^-1 for strictly lower triangular A [1, T, H, 64] (varlen)."""
    assert A.shape[-1] == 64
    B, T, H, BT = A.shape
    NT = len(chunk_indices)
    # only the lower triangle is written; the consumer (kda_wu_kernel) masks the rest
    Ai = torch.empty_like(A, dtype=output_dtype)
    merge_16x16_to_64x64_inverse_kernel[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        num_warps=1,
        num_stages=1,
    )
    return Ai


# ============================================================================
# The sequential state pass. gk = per-chunk last-row gate [NT, H, K]; k (= kg)
# comes as a bf16 hi/lo pair and v (= u) is fp32: v - w h^T with h split into
# bf16 hi + lo, the k^T v_new update with k and v_new split (bf16x3), so the
# only bf16 rounding in the recurrence is w; v_new is stored in bf16 for the
# o pass.
# ============================================================================


@triton.jit
def _split_bf16(x):
    hi = x.to(tl.bfloat16, fp_downcast_rounding="rtne")
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    return hi, lo


@triton.jit
def _dot3(a_hi, a_lo, b_hi, b_lo, acc):
    # ~fp32 product from bf16 tensor-core dots: lo*hi + hi*lo + hi*hi (small terms first)
    acc = tl.dot(a_lo, b_hi, acc)
    acc = tl.dot(a_hi, b_lo, acc)
    acc = tl.dot(a_hi, b_hi, acc)
    return acc


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    k_lo,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    k_lo += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, 64),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 64),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 128),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 192),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        h_hi = b_h1.to(b_w.dtype)
        b_v = tl.dot(b_w, tl.trans((b_h1 - h_hi.to(tl.float32)).to(b_w.dtype)))
        b_v = tl.dot(b_w, tl.trans(h_hi), b_v)
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            h_hi = b_h2.to(b_w.dtype)
            b_v = tl.dot(b_w, tl.trans((b_h2 - h_hi.to(tl.float32)).to(b_w.dtype)), b_v)
            b_v = tl.dot(b_w, tl.trans(h_hi), b_v)
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            h_hi = b_h3.to(b_w.dtype)
            b_v = tl.dot(b_w, tl.trans((b_h3 - h_hi.to(tl.float32)).to(b_w.dtype)), b_v)
            b_v = tl.dot(b_w, tl.trans(h_hi), b_v)
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            h_hi = b_h4.to(b_w.dtype)
            b_v = tl.dot(b_w, tl.trans((b_h4 - h_hi.to(tl.float32)).to(b_w.dtype)), b_v)
            b_v = tl.dot(b_w, tl.trans(h_hi), b_v)
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + ((boh + i_t) * H + i_h) * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + ((boh + i_t) * H + i_h) * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + ((boh + i_t) * H + i_h) * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + ((boh + i_t) * H + i_h) * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        vn_hi, vn_lo = _split_bf16(b_v)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kl = tl.load(tl.make_block_ptr(k_lo, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)), boundary_check=(0, 1))
        b_h1 += tl.trans(_dot3(b_k, b_kl, vn_hi, vn_lo, tl.zeros([64, BV], dtype=tl.float32)))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kl = tl.load(tl.make_block_ptr(k_lo, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)), boundary_check=(0, 1))
            b_h2 += tl.trans(_dot3(b_k, b_kl, vn_hi, vn_lo, tl.zeros([64, BV], dtype=tl.float32)))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kl = tl.load(tl.make_block_ptr(k_lo, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)), boundary_check=(0, 1))
            b_h3 += tl.trans(_dot3(b_k, b_kl, vn_hi, vn_lo, tl.zeros([64, BV], dtype=tl.float32)))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kl = tl.load(tl.make_block_ptr(k_lo, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)), boundary_check=(0, 1))
            b_h4 += tl.trans(_dot3(b_k, b_kl, vn_hi, vn_lo, tl.zeros([64, BV], dtype=tl.float32)))
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    k_lo: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K, dtype=torch.bfloat16)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u, dtype=torch.bfloat16) if save_new_value else None

    # fixed: BV 32, 4 warps, 2 stages (fastest on this part among BV 32/64/128
    # x 4/8 warps; deeper stages exceed shared memory with the fp32 operands)
    BV = 32
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[(triton.cdiv(V, BV), N * H)](
        k=k,
        k_lo=k_lo,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, final_state


# ============================================================================
# o = qg h^T + Aqk v_new
# ============================================================================

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    qg,
    v,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_qg = tl.make_block_ptr(
            qg + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )
        # [BT, BK]
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(
    qg: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    grid = (triton.cdiv(V, O_BV), NT, B * H)
    chunk_gla_fwd_kernel_o[grid](
        qg=qg,
        v=v,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=O_BK,
        BV=O_BV,
        num_warps=O_WARPS,
        num_stages=O_STAGES,
    )
    return o


# ============================================================================
# Intra-chunk path.  Precision: the state recurrence only ever sees bf16
# roundings of w and v_new (as upstream); k (l2-normalised in fp32, never
# rounded), A, (I + A)^-1, u = Ai beta v and kg stay fp32 / ~fp32 (bf16x3-split
# tensor-core products), and the state pass splits h and kg into bf16 hi + lo.
# ============================================================================

@triton.jit(do_not_specialize=["T"])
def kda_gate_kernel(
    g_raw, A_log, g_bias, g, g_last,
    cu_seqlens, chunk_indices,
    stride_gt, stride_gh,
    H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    SAFE_GATE: tl.constexpr, LOWER_BOUND: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    # gate + chunk-local cumsum (log2 units) and g at each chunk's last valid row; one
    # program per (chunk, head, BD-column slice): the cumsum runs down columns, so slices
    # are independent and small tiles keep register use (and residency) sane.
    i_tg, i_h, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)
    t0 = i_t * BT
    o_d = i_d * BD + tl.arange(0, BD)
    o_t = tl.arange(0, BT)
    b_a = tl.load(A_log + i_h).to(tl.float32)
    if SAFE_GATE:
        b_a = tl.exp(b_a)
    else:
        b_a = -tl.exp(b_a)
    p_g = tl.make_block_ptr(g_raw + bos * stride_gt + i_h * stride_gh, (T, D), (stride_gt, 1),
                            (t0, i_d * BD), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(g_bias + i_h * D + o_d).to(tl.float32)[None, :]
    if SAFE_GATE:
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_gate = b_a * tl.where(b_g > 20.0, b_g, log(1.0 + tl.exp(b_g)))
    b_y = tl.cumsum(b_gate, 0) * RCP_LN2_C
    p_y = tl.make_block_ptr(g + (bos * H + i_h) * D, (T, D), (H * D, 1), (t0, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, b_y, boundary_check=(0, 1))
    last = tl.minimum(T - t0, BT) - 1
    tl.store(g_last + (i_tg * H + i_h) * D + o_d, tl.sum(tl.where((o_t == last)[:, None], b_y, 0.0), 0))


@triton.jit
def _ld_rows(p, rows, T, stride_t, C: tl.constexpr):
    o_c = tl.arange(0, C)
    return tl.load(p + rows[:, None] * stride_t + o_c[None, :], mask=(rows < T)[:, None], other=0.0).to(tl.float32)


@triton.jit
def _l2n(x, eps, USE_L2NORM: tl.constexpr):
    if USE_L2NORM:
        x = x * tl.rsqrt(tl.sum(x * x, 1) + eps)[:, None]
    return x


@triton.jit
def _rnorm(p, rows, T, stride_t, eps, K: tl.constexpr, BK: tl.constexpr, R: tl.constexpr,
           USE_L2NORM: tl.constexpr):
    # 1 / ||x_row|| (rsqrt(sum x^2 + eps), fp32), accumulated over BK-column slices
    if USE_L2NORM:
        ss = tl.zeros([R], dtype=tl.float32)
        for kk in tl.static_range(0, K, BK):
            x = _ld_rows(p + kk, rows, T, stride_t, BK)
            ss += tl.sum(x * x, 1)
        return tl.rsqrt(ss + eps)
    else:
        return tl.full([R], 1.0, dtype=tl.float32)


@triton.jit(do_not_specialize=["T"])
def kda_kkt_kernel(
    q, k, g, g_last, beta, A, Aqk, qg, kg, kg_lo,
    cu_seqlens, chunk_indices, scale, eps,
    stride_qt, stride_qh, stride_kt, stride_kh,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, USE_L2NORM: tl.constexpr,
):
    # One program per (chunk, 16-row block I, head): row block I of
    #   A = beta k k^T (strictly lower) and Aqk = scale q k^T (lower incl. diagonal),
    # with the per-channel decay folded into the operands.  References keep every
    # factor finite and normal: off-diagonal blocks J < I use g at the last row of
    # block I-1 (both factors <= 1); the diagonal block uses g at its row 7
    # (|exponent| <= 8 steps * 5 / ln2 < 58).  Also writes qg = q scale exp2 g (bf16)
    # and kg = k exp2(g_last - g) as a bf16 hi/lo pair for the rows of block I; Aqk is
    # stored in bf16 (the o kernel rounds it to bf16 anyway).
    i_tg, I, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)
    t0 = i_t * BT
    r0 = t0 + I * 16
    if r0 >= T:
        return
    HK: tl.constexpr = H * K
    o_i = tl.arange(0, 16)
    o_k = tl.arange(0, K)
    rw = r0 + o_i
    m_w = rw < T
    p_q = q + bos * stride_qt + i_h * stride_qh
    p_k = k + bos * stride_kt + i_h * stride_kh
    p_g = g + (bos * H + i_h) * K
    p_A = A + (bos * H + i_h) * BT + rw[:, None] * (H * BT) + o_i[None, :]
    p_S = Aqk + (bos * H + i_h) * BT + rw[:, None] * (H * BT) + o_i[None, :]

    k_i = _l2n(_ld_rows(p_k, rw, T, stride_kt, K), eps, USE_L2NORM)
    q_i = _l2n(_ld_rows(p_q, rw, T, stride_qt, K), eps, USE_L2NORM) * scale
    g_i = _ld_rows(p_g, rw, T, HK, K)
    b_b = tl.load(beta + (bos + rw) * H + i_h, mask=m_w, other=0.0).to(tl.float32)
    gll = tl.load(g_last + (i_tg * H + i_h) * K + o_k)
    p_o = (bos * H + i_h) * K + rw[:, None] * HK + o_k[None, :]
    tl.store(qg + p_o, (q_i * exp2(g_i)).to(tl.bfloat16), mask=m_w[:, None])
    # kg as a bf16 hi + lo pair (the rtne split the state pass used to do itself)
    kg_hi, kg_l = _split_bf16(k_i * exp2(gll[None, :] - g_i))
    tl.store(kg + p_o, kg_hi, mask=m_w[:, None])
    tl.store(kg_lo + p_o, kg_l, mask=m_w[:, None])

    # diagonal block
    rd = tl.load(p_g + tl.minimum(r0 + 7, T - 1) * HK + o_k)
    e = exp2(tl.where(m_w[:, None], g_i - rd[None, :], 0.0))
    l_hi, l_lo = _split_bf16(k_i * e)
    s_hi, s_lo = _split_bf16(q_i * e)
    r_hi, r_lo = _split_bf16(tl.trans(k_i * exp2(tl.where(m_w[:, None], rd[None, :] - g_i, 0.0))))
    zero = tl.zeros([16, 16], dtype=tl.float32)
    b_A = _dot3(l_hi, l_lo, r_hi, r_lo, zero) * b_b[:, None]
    b_S = _dot3(s_hi, s_lo, r_hi, r_lo, zero)
    tl.store(p_A + I * 16, tl.where(o_i[:, None] > o_i[None, :], b_A, 0.0), mask=m_w[:, None])
    tl.store(p_S + I * 16, tl.where(o_i[:, None] >= o_i[None, :], b_S, 0.0).to(tl.bfloat16), mask=m_w[:, None])

    # off-diagonal blocks J < I
    if I > 0:
        ro = tl.load(p_g + (r0 - 1) * HK + o_k)
        e = exp2(tl.where(m_w[:, None], g_i - ro[None, :], 0.0))
        l_hi, l_lo = _split_bf16(k_i * e)
        s_hi, s_lo = _split_bf16(q_i * e)
        for J in range(0, I):
            rj = t0 + J * 16 + o_i
            k_j = _l2n(_ld_rows(p_k, rj, T, stride_kt, K), eps, USE_L2NORM)
            g_j = _ld_rows(p_g, rj, T, HK, K)
            r_hi, r_lo = _split_bf16(tl.trans(k_j * exp2(ro[None, :] - g_j)))
            b_A = _dot3(l_hi, l_lo, r_hi, r_lo, zero) * b_b[:, None]
            b_S = _dot3(s_hi, s_lo, r_hi, r_lo, zero)
            tl.store(p_A + J * 16, b_A, mask=m_w[:, None])
            tl.store(p_S + J * 16, b_S.to(tl.bfloat16), mask=m_w[:, None])


@triton.jit(do_not_specialize=["T"])
def kda_wu_kernel(
    k, v, g, beta, Ai, w, u,
    cu_seqlens, chunk_indices, eps,
    stride_kt, stride_kh, stride_vt, stride_vh,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    # w = Ai (beta k exp2 g) -> bf16,  u = Ai (beta v) -> fp32; Ai fp32 (lower triangle
    # of the tile is read, the rest masked), products bf16x3-split.
    i_tg, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)
    t0 = i_t * BT
    HK: tl.constexpr = H * K
    o_t = tl.arange(0, BT)
    rows = t0 + o_t
    m_r = rows < T
    p_k = k + bos * stride_kt + i_h * stride_kh
    p_v = v + bos * stride_vt + i_h * stride_vh
    p_g = g + (bos * H + i_h) * K

    pa = tl.make_block_ptr(Ai + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (t0, 0), (BT, BT), (1, 0))
    b_Ai = tl.load(pa, boundary_check=(0, 1))
    b_Ai = tl.where(o_t[:, None] >= o_t[None, :], b_Ai, 0.0)
    a_hi, a_lo = _split_bf16(b_Ai)
    b_b = tl.load(beta + (bos + rows) * H + i_h, mask=m_r, other=0.0).to(tl.float32)
    rk = _rnorm(p_k, rows, T, stride_kt, eps, K, BK, BT, USE_L2NORM) * b_b
    for kk in tl.static_range(0, K, BK):
        pk = tl.make_block_ptr(p_k, (T, K), (stride_kt, 1), (t0, kk), (BT, BK), (1, 0))
        pg = tl.make_block_ptr(p_g, (T, K), (HK, 1), (t0, kk), (BT, BK), (1, 0))
        b_x = tl.load(pk, boundary_check=(0, 1)).to(tl.float32) * rk[:, None]
        b_x = b_x * exp2(tl.load(pg, boundary_check=(0, 1)))
        x_hi, x_lo = _split_bf16(b_x)
        b_w = _dot3(a_hi, a_lo, x_hi, x_lo, tl.zeros([BT, BK], dtype=tl.float32))
        pw = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (HK, 1), (t0, kk), (BT, BK), (1, 0))
        tl.store(pw, b_w.to(tl.bfloat16), boundary_check=(0, 1))
    for vv in tl.static_range(0, V, BK):
        pv = tl.make_block_ptr(p_v, (T, V), (stride_vt, 1), (t0, vv), (BT, BK), (1, 0))
        b_x = tl.load(pv, boundary_check=(0, 1)).to(tl.float32) * b_b[:, None]
        x_hi, x_lo = _split_bf16(b_x)
        b_u = _dot3(a_hi, a_lo, x_hi, x_lo, tl.zeros([BT, BK], dtype=tl.float32))
        pu = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (t0, vv), (BT, BK), (1, 0))
        tl.store(pu, b_u, boundary_check=(0, 1))


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    **kwargs,
):
    """Run chunk KDA from raw gate projection using fused gate+cumsum."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if cu_seqlens is None:
        # equal-length batch == varlen over the flattened batch
        cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=k.device)
        q, k, v, raw_g = (x.reshape(1, B * T, H, x.shape[-1]) for x in (q, k, v, raw_g))
        beta = beta.reshape(1, B * T, H)
    assert q.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    BT = FLA_CHUNK_SIZE
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)
    TT = q.shape[1]
    dev = k.device
    l2 = bool(use_qk_l2norm_in_kernel)

    def _lastdim(x):
        return x if x.stride(-1) == 1 else x.contiguous()

    q, k, v, raw_g = _lastdim(q), _lastdim(k), _lastdim(v), _lastdim(raw_g)
    beta = beta.contiguous()

    g = torch.empty(1, TT, H, K, device=dev, dtype=torch.float32)
    g_last = torch.empty(NT, H, K, device=dev, dtype=torch.float32)
    GBD = 32  # [64, 32] slices on 1 warp: the cumsum stays inside the warp
    kda_gate_kernel[(NT, H, K // GBD)](
        raw_g, A_log.reshape(-1), g_bias.reshape(-1) if g_bias is not None else None, g, g_last,
        cu_seqlens, chunk_indices, raw_g.stride(1), raw_g.stride(2),
        H=H, D=K, BT=BT, BD=GBD, SAFE_GATE=bool(safe_gate), LOWER_BOUND=float(lower_bound),
        HAS_BIAS=g_bias is not None, num_warps=1,
    )
    A = torch.empty(1, TT, H, BT, device=dev, dtype=torch.float32)
    Aqk = torch.empty(1, TT, H, BT, device=dev, dtype=torch.bfloat16)
    qg = torch.empty(1, TT, H, K, device=dev, dtype=torch.bfloat16)
    kg = torch.empty(1, TT, H, K, device=dev, dtype=torch.bfloat16)
    kg_lo = torch.empty(1, TT, H, K, device=dev, dtype=torch.bfloat16)
    kda_kkt_kernel[(NT, BT // 16, H)](
        q, k, g, g_last, beta, A, Aqk, qg, kg, kg_lo, cu_seqlens, chunk_indices, float(scale), 1e-6,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2),
        # num_stages=1: the default 3 double-buffers the J loop's k/g tiles (48 KB
        # smem, 3 CTAs/SM); 24 KB lets 6 CTAs/SM hide the per-pair latency chain
        H=H, K=K, BT=BT, USE_L2NORM=l2, num_warps=2, num_stages=1,
    )
    Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=torch.float32)
    del A
    w = torch.empty(1, TT, H, K, device=dev, dtype=torch.bfloat16)
    u = torch.empty(1, TT, H, V, device=dev, dtype=torch.float32)
    kda_wu_kernel[(NT, H)](
        k, v, g, beta, Ai, w, u, cu_seqlens, chunk_indices, 1e-6,
        k.stride(1), k.stride(2), v.stride(1), v.stride(2),
        H=H, K=K, V=V, BT=BT, BK=32, USE_L2NORM=l2, num_warps=4,
    )
    del Ai, g
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        k_lo=kg_lo,
        w=w,
        u=u,
        gk=g_last,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    del w, u, kg, kg_lo
    o = chunk_gla_fwd_o_gk(
        qg=qg,
        v=v_new,
        A=Aqk,
        h=h,
        o=torch.empty(1, TT, H, V, device=dev, dtype=v.dtype),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )
    if o.shape[1] != T:
        o = o.reshape(B, T, H, V)
    return o, final_state


# ============================================================================
# Gate, dispatch and warm-up
# ============================================================================

# The configuration this path was validated on: all 64 heads on one card, head
# dim 128, bf16 activations, fp32 recurrent state, A_log and dt_bias, the
# bounded (safe) gate at lower_bound -5, l2-normalised q and k, varlen
# cu_seqlens, and prefill chunks of 1..2312 tokens holding at most 16
# sequences. Anything else keeps the upstream path.
HEADS = 64
HEAD_DIM = 128
LOWER_BOUND = -5.0
MAX_TOKENS = 2312
MAX_SEQS = 16

BANNER = (
    "GLM5 PP KDA prefill: sm_80 fused chunk path live (64 heads); "
    "VLLM_GLM5_PP_KDA_PREFILL=1"
)


def layer_closed_reason(
    backend: str,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    safe_gate: bool,
    lower_bound: float | None,
    capability: tuple[int, int] | None,
) -> str:
    """Why a KDA layer cannot take this path, or '' when it can."""
    if capability != (8, 0):
        return f"device capability {capability} (needs sm_80)"
    if backend != "triton":
        return f"prefill backend {backend} (needs triton)"
    if num_heads != HEADS:
        return f"{num_heads} heads on this rank (needs {HEADS}: pipeline parallel, TP=1)"
    if head_dim != HEAD_DIM:
        return f"head_dim {head_dim} (needs {HEAD_DIM})"
    if dtype != torch.bfloat16:
        return f"model dtype {dtype} (needs bfloat16)"
    if not safe_gate or lower_bound is None or float(lower_bound) != LOWER_BOUND:
        return f"gate safe={safe_gate} lower_bound={lower_bound} (needs the safe gate at {LOWER_BOUND})"
    return ""


def use_for_layer(
    backend: str,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    safe_gate: bool,
    lower_bound: float | None,
) -> bool:
    """Resolve the layer part of the gate once (called with the flag set)."""
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability()
    cap = None if cap is None else (int(cap.major), int(cap.minor))
    why = layer_closed_reason(
        backend, num_heads, head_dim, dtype, safe_gate, lower_bound, cap
    )
    if why:
        logger.info_once(
            "VLLM_GLM5_PP_KDA_PREFILL=1 but the sm_80 KDA prefill path is off: "
            "%s; using the upstream chunk path",
            why,
        )
        return False
    logger.info_once(BANNER)
    return True


def call_closed_reason(
    num_tokens: int,
    num_seqs: int,
    qkv_dtype: torch.dtype,
    g_dtype: torch.dtype,
    state_dtype: torch.dtype,
    a_log_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
) -> str:
    """Why this prefill call cannot take the path, or '' when it can. Host
    values only (shapes, dtypes): no device sync."""
    if not 1 <= num_tokens <= MAX_TOKENS:
        return f"prefill chunk outside 1..{MAX_TOKENS} tokens"
    if not 1 <= num_seqs <= MAX_SEQS:
        return f"more than {MAX_SEQS} sequences in the chunk"
    if qkv_dtype != torch.bfloat16 or g_dtype != torch.bfloat16:
        return "q/k/v or gate input not bfloat16"
    if state_dtype != torch.float32:
        return "recurrent state not float32"
    if a_log_dtype != torch.float32 or bias_dtype != torch.float32:
        return "A_log or dt_bias not float32"
    return ""


def select_chunk_fn(
    upstream,
    num_tokens: int,
    num_seqs: int,
    qkv_dtype: torch.dtype,
    g_dtype: torch.dtype,
    state_dtype: torch.dtype,
    a_log_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
):
    """This module's chunk_kda_with_fused_gate when the call is in the
    validated range, else ``upstream`` (called with identical arguments)."""
    why = call_closed_reason(
        num_tokens, num_seqs, qkv_dtype, g_dtype, state_dtype, a_log_dtype, bias_dtype
    )
    if why:
        logger.info_once(
            "VLLM_GLM5_PP_KDA_PREFILL=1 but this prefill call keeps the upstream "
            "chunk path: %s",
            why,
        )
        return upstream
    return chunk_kda_with_fused_gate


def warmup(plans, device="cuda", heads: int = HEADS, head_dim: int = HEAD_DIM):
    """Compile every kernel variant: one call per plan (a list of sequence
    lengths) in the production input layout (q, k, v column views of one
    [T, 3 * H * D] buffer, fp32 beta, fp32 initial state). All launch
    configurations are fixed, so one small plan already compiles everything;
    Triton specialises on none of the shape values that differ between plans."""
    P = heads * head_dim
    for seqlens in plans:
        T = int(sum(seqlens))
        cu = torch.tensor(
            [0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device=device
        )
        buf = torch.randn(T, 3 * P, device=device, dtype=torch.bfloat16) * 0.1
        q, k, v = (buf[:, i * P : (i + 1) * P].view(1, T, heads, head_dim) for i in range(3))
        raw_g = torch.randn(1, T, heads, head_dim, device=device, dtype=torch.bfloat16)
        beta = torch.rand(1, T, heads, device=device, dtype=torch.float32)
        a_log = torch.zeros(1, 1, heads, 1, device=device, dtype=torch.float32)
        dt = torch.zeros(P, device=device, dtype=torch.float32)
        h0 = torch.zeros(
            len(seqlens), heads, head_dim, head_dim, device=device, dtype=torch.float32
        )
        chunk_kda_with_fused_gate(
            q=q, k=k, v=v, raw_g=raw_g, beta=beta, A_log=a_log, g_bias=dt,
            initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu, safe_gate=True, lower_bound=LOWER_BOUND,
        )
    torch.cuda.synchronize(device)


WARMUP_PLANS = ([64], [65, 63])


def warmup_from_worker(worker) -> None:
    """Compile the path before serving when any KDA layer of this worker took
    it (the first prefill then pays no Triton compilation). Prefill runs
    eagerly, outside CUDA graphs; nothing here is kept after the call. No-op
    unless VLLM_GLM5_PP_KDA_PREFILL=1 and the layer gate is open."""
    from vllm import envs

    if not envs.VLLM_GLM5_PP_KDA_PREFILL:
        return
    layer = None
    for module in worker.get_model().modules():
        if getattr(module, "_pp_kda_prefill", False):
            layer = module
            break
    if layer is None:
        return
    warmup(WARMUP_PLANS, device=worker.device)
    logger.info(
        "Warmed up the sm_80 KDA prefill kernels (64 heads) for plans %s.",
        WARMUP_PLANS,
    )
