# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DeepSeek-V4.1 attention core for decode-sized steps on sm_80
(``VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE``).

Two calls per layer replace the torch/Triton pieces of
``sm80_attn_core.prepare_layer`` and ``attend_layer`` around ``wq_b``:

``prologue(qr_kv, q_norm, kv_norm, positions, freqs, swa_cache, swa_slots, eps)``
    one launch: ``qr = rms_norm(qr_kv[:T, :q_lora], q_norm)`` (the ``wq_b``
    input, returned) and, for every ``swa_slots[t] >= 0``, the window row
    ``qdq_fp8(rope(rms_norm(qr_kv[t, q_lora:], kv_norm)))`` written into
    ``swa_cache``; no other cache row changes (not even the null block).
``attend(q, positions, freqs, swa_cache, swa_indices, comp_cache,
comp_block_table, comp_local, req, attn_sink, scale, n_groups, out)``
    ``q`` is ``wq_b``'s output BEFORE RoPE ([T, heads, head_dim]); the query
    RoPE, the sparse attention with sinks over the window slots and the
    selected compressed positions, the inverse RoPE and the per-group FP8
    (32, UE8M0) qdq; writes ``out[:T]`` (the ``wo_a`` input) and zeros
    ``out[T:]``.

The rounding points are the reference's (``sm80_attn_core``): the RMSNorm
outputs, the RoPE outputs, the attention output and the inverse-RoPE output
are BF16; the FP8 qdq is exact. RMSNorm and the window row's RoPE run in FP64
between those points (full rate on this card); the query and output RoPE keep
FP32 angles and products without FMA contraction. The attention runs as a
score pass and an ordered value pass that walks the window, then the
compressed rows, in the reference's order with an online softmax, a BF16
probability before the value dot and the sink added to the denominator once.
No host sync, no data-dependent host branching; the scratch tensors come from
the caching allocator (CUDA-graph pool under capture).

Validated configuration (:func:`supports`): 16 local heads of 512 with a
64-dim RoPE tail, 2 local output groups, a 128-slot window, compressed
top-k width 512 or none, BF16 caches, sm_80, calls of 1..``MAX_ROWS`` rows
(the caller limits it to decode-sized steps of at most that many rows).
"""
import torch
from triton.language.extra import libdevice

from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import qdq_fp8_tile
from vllm.triton_utils import tl, triton

# The validated configuration (see :func:`supports`).
N_LOCAL_HEADS = 16
HEAD_DIM = 512
ROPE_DIM = 64
N_LOCAL_GROUPS = 2
WINDOW = 128
TOPK = 512
# The most rows (tokens of the padded step) a call was validated at: decode,
# verify and draft steps of 1..48 rows.
MAX_ROWS = 48

# ---------------------------------------------------------------- prologue

@triton.jit
def _prologue_kernel(
    x_ptr, q_norm_ptr, kv_norm_ptr, positions_ptr, freqs_ptr, cache_ptr,
    slots_ptr, qr_ptr, eps,
    stride_xt: tl.constexpr, stride_xd: tl.constexpr,
    stride_qn: tl.constexpr, stride_kn: tl.constexpr,
    stride_pos: tl.constexpr, stride_freq: tl.constexpr,
    stride_cb: tl.constexpr, stride_ce: tl.constexpr, stride_cd: tl.constexpr,
    stride_slot: tl.constexpr,
    QD: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
    RD: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 0:
        d = tl.arange(0, BQ)
        x = tl.load(x_ptr + t * stride_xt + d * stride_xd, d < QD, 0).to(tl.float64)
        w = tl.load(q_norm_ptr + d * stride_qn, d < QD, 0).to(tl.float64)
        inv = libdevice.rsqrt(tl.sum(x * x, axis=0) / QD + eps)
        qr = (x * inv * w).to(tl.bfloat16)
        tl.store(qr_ptr + t * QD + d, qr, d < QD)
    else:
        slot = tl.load(slots_ptr + t * stride_slot)
        if slot >= 0:
            d = tl.arange(0, BK)
            x = tl.load(x_ptr + t * stride_xt + (QD + d) * stride_xd, d < D, 0).to(tl.float64)
            w = tl.load(kv_norm_ptr + d * stride_kn, d < D, 0).to(tl.float64)
            inv = libdevice.rsqrt(tl.sum(x * x, axis=0) / D + eps)
            kv = (x * inv * w).to(tl.bfloat16).to(tl.float32)

            r = tl.arange(0, RD)
            a = tl.gather(kv, D - RD + r, axis=0).to(tl.float64)
            b = tl.gather(kv, D - RD + (r ^ 1), axis=0).to(tl.float64)
            pos = tl.load(positions_ptr + t * stride_pos).to(tl.float32)
            freq = tl.load(freqs_ptr + (r // 2) * stride_freq)
            angle = (pos * freq).to(tl.float64)
            signed_sin = tl.where(r % 2 == 0, -libdevice.sin(angle), libdevice.sin(angle))
            rotated = (a * libdevice.cos(angle) + b * signed_sin).to(tl.bfloat16).to(tl.float32)
            tail = tl.gather(rotated, tl.maximum(d - (D - RD), 0), axis=0)
            kv = tl.where(d >= D - RD, tail, kv)
            rows = qdq_fp8_tile(tl.reshape(kv, (BK // 32, 32)))
            rows = tl.reshape(rows, (BK,))
            off = (slot // E) * stride_cb + (slot % E) * stride_ce
            tl.store(cache_ptr + off + d * stride_cd, rows.to(tl.bfloat16), d < D)


def fused_prologue(qr_kv, q_norm, kv_norm, positions, freqs, swa_cache, swa_slots, eps,
                   *, qr_out=None):
    t = positions.shape[0]
    d = swa_cache.shape[-1]
    qd = qr_kv.shape[1] - d
    qr = qr_out if qr_out is not None else torch.empty(
        (t, qd), dtype=qr_kv.dtype, device=qr_kv.device)
    if t:
        _prologue_kernel[(t, 2)](
            qr_kv, q_norm, kv_norm, positions, freqs, swa_cache, swa_slots, qr, eps,
            qr_kv.stride(0), qr_kv.stride(1), q_norm.stride(0), kv_norm.stride(0),
            positions.stride(0), freqs.stride(0),
            swa_cache.stride(0), swa_cache.stride(1), swa_cache.stride(2),
            swa_slots.stride(0), QD=qd, D=d, E=swa_cache.shape[1], RD=2 * freqs.numel(),
            BQ=triton.next_power_of_2(qd), BK=triton.next_power_of_2(d),
            num_warps=4, enable_fp_fusion=False,
        )
    return qr


# ---------------------------------------------------------------- attention

_BLOCK_H = 16
_BLOCK_N = 32
_BLOCK_D = 64


@triton.jit
def _attend(s, kv, ok, m, l, acc):
    s = tl.where(ok[None, :], s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, axis=1))
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    return m_new, l, acc


@triton.jit
def _query_rope_kernel(
    q_ptr, positions_ptr, freqs_ptr, rotated_ptr,
    stride_qt: tl.constexpr, stride_qh: tl.constexpr,
    stride_pos: tl.constexpr, stride_freq: tl.constexpr,
    stride_rt: tl.constexpr, stride_rh: tl.constexpr,
    D: tl.constexpr, RD: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + t * stride_qt + h * stride_qh + d)
    r = tl.arange(0, RD)
    a = tl.gather(q, D - RD + r, axis=0).to(tl.float32)
    b = tl.gather(q, D - RD + (r ^ 1), axis=0).to(tl.float32)
    pos = tl.load(positions_ptr + t * stride_pos).to(tl.float32)
    freq = tl.load(freqs_ptr + (r // 2) * stride_freq)
    angle = pos * freq
    sine = libdevice.sin(angle)
    signed_sine = tl.where(r % 2 == 0, -sine, sine)
    real_prod = libdevice.mul_rn(a, libdevice.cos(angle))
    imag_prod = libdevice.mul_rn(b, signed_sine)
    rotated = libdevice.add_rn(real_prod, imag_prod).to(tl.bfloat16)
    tail = tl.gather(rotated, tl.maximum(d - (D - RD), 0), axis=0)
    q = tl.where(d >= D - RD, tail, q)
    tl.store(rotated_ptr + t * stride_rt + h * stride_rh + d, q)


@triton.jit(do_not_specialize=["n_window", "n_comp"])
def _score_prepass_kernel(
    q_ptr,
    swa_ptr,
    swa_idx_ptr,
    comp_ptr,
    comp_bt_ptr,
    comp_idx_ptr,
    req_ptr,
    scores_ptr,
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
    score_width,
    WINDOW_TILES: tl.constexpr,
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
    tile = tl.program_id(2)
    n = tl.arange(0, BLOCK_N)
    ok = tl.full([BLOCK_N], False, tl.int1)
    off = tl.zeros([BLOCK_N], tl.int64)
    if tile < WINDOW_TILES:
        cols = tile * BLOCK_N + n
        slot = tl.load(swa_idx_ptr + t * stride_wi + cols, mask=cols < n_window, other=-1)
        ok = slot >= 0
        s64 = tl.maximum(slot, 0).to(tl.int64)
        off = (s64 // SWA_E) * stride_sb + (s64 % SWA_E) * stride_se
    elif HAS_COMP:
        cols = (tile - WINDOW_TILES) * BLOCK_N + n
        j = tl.load(comp_idx_ptr + t * stride_ci + cols, mask=cols < n_comp, other=-1)
        ok = j >= 0
        req = tl.load(req_ptr + t).to(tl.int64)
        j64 = tl.maximum(j, 0).to(tl.int64)
        blk = tl.load(comp_bt_ptr + req * stride_bt + j64 // COMP_E, mask=ok, other=0)
        off = blk.to(tl.int64) * stride_cb + (j64 % COMP_E) * stride_ce
    s = tl.full([BLOCK_H, BLOCK_N], float("-inf"), tl.float32)
    if tl.sum(ok.to(tl.int32), axis=0) > 0:
        q = tl.load(
            q_ptr + t * stride_qt + heads[:, None] * stride_qh + d[None, :],
            mask=hmask[:, None],
            other=0.0,
        )
        if tile < WINDOW_TILES:
            kv = tl.load(swa_ptr + off[:, None] + d[None, :], mask=ok[:, None], other=0.0)
        else:
            kv = tl.load(comp_ptr + off[:, None] + d[None, :], mask=ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        s = tl.where(ok[None, :], s, float("-inf"))
    tl.store(
        scores_ptr + (t * n_heads + heads[:, None]) * score_width + tile * BLOCK_N + n[None, :],
        s,
        mask=hmask[:, None],
    )


@triton.jit(do_not_specialize=["n_window", "n_comp"])
def _ordered_value_kernel(
    scores_ptr,
    swa_ptr,
    swa_idx_ptr,
    comp_ptr,
    comp_bt_ptr,
    comp_idx_ptr,
    req_ptr,
    sink_ptr,
    out_ptr,
    positions_ptr,
    freqs_ptr,
    n_tokens,
    n_heads,
    n_window,
    n_comp,
    score_width,
    stride_sb,
    stride_se,
    stride_wi,
    stride_cb,
    stride_ce,
    stride_bt,
    stride_ci,
    stride_ot,
    stride_oh,
    stride_pos: tl.constexpr,
    stride_freq: tl.constexpr,
    RD: tl.constexpr,
    EPILOGUE: tl.constexpr,
    SWA_E: tl.constexpr,
    COMP_E: tl.constexpr,
    HAS_COMP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WINDOW_TILES: tl.constexpr,
    WINDOW_INDEX_BLOCK: tl.constexpr,
    COMP_INDEX_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = heads < n_heads
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < D
    if EPILOGUE and t >= n_tokens:
        tl.store(out_ptr + t * stride_ot + heads[:, None] * stride_oh + d[None, :],
                 0.0, hmask[:, None] & dmask[None, :])
        return
    m = tl.full([BLOCK_H], -1.0e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    n = tl.arange(0, BLOCK_N)
    wcols = tl.arange(0, WINDOW_INDEX_BLOCK)
    windices = tl.load(swa_idx_ptr + t * stride_wi + wcols, mask=wcols < n_window, other=-1)
    window_end = tl.max(tl.where(windices >= 0, wcols + 1, 0), axis=0)
    for n0 in range(0, window_end, BLOCK_N):
        cols = n0 + n
        slot = tl.load(swa_idx_ptr + t * stride_wi + cols, mask=cols < n_window, other=-1)
        ok = slot >= 0
        s64 = tl.maximum(slot, 0).to(tl.int64)
        off = (s64 // SWA_E) * stride_sb + (s64 % SWA_E) * stride_se
        kv = tl.load(
            swa_ptr + off[:, None] + d[None, :],
            mask=ok[:, None] & dmask[None, :],
            other=0.0,
        )
        s = tl.load(
            scores_ptr + (t * n_heads + heads[:, None]) * score_width + cols[None, :],
            mask=hmask[:, None],
            other=float("-inf"),
        )
        m, l, acc = _attend(s, kv, ok, m, l, acc)
    if HAS_COMP:
        req = tl.load(req_ptr + t).to(tl.int64)
        ccols = tl.arange(0, COMP_INDEX_BLOCK)
        cindices = tl.load(comp_idx_ptr + t * stride_ci + ccols, mask=ccols < n_comp, other=-1)
        comp_end = tl.max(tl.where(cindices >= 0, ccols + 1, 0), axis=0)
        for n0 in range(0, comp_end, BLOCK_N):
            cols = n0 + n
            j = tl.load(comp_idx_ptr + t * stride_ci + cols, mask=cols < n_comp, other=-1)
            ok = j >= 0
            j64 = tl.maximum(j, 0).to(tl.int64)
            blk = tl.load(comp_bt_ptr + req * stride_bt + j64 // COMP_E, mask=ok, other=0)
            off = blk.to(tl.int64) * stride_cb + (j64 % COMP_E) * stride_ce
            kv = tl.load(
                comp_ptr + off[:, None] + d[None, :],
                mask=ok[:, None] & dmask[None, :],
                other=0.0,
            )
            s = tl.load(
                scores_ptr + (t * n_heads + heads[:, None]) * score_width
                + WINDOW_TILES * BLOCK_N + cols[None, :],
                mask=hmask[:, None],
                other=float("-inf"),
            )
            m, l, acc = _attend(s, kv, ok, m, l, acc)
    sink = tl.load(sink_ptr + heads, mask=hmask, other=float("-inf"))
    l = l + tl.exp(sink - m)
    o = acc / l[:, None]
    if EPILOGUE:
        # The value result is stored to BF16 BEFORE inverse RoPE and QDQ.
        o = o.to(tl.bfloat16).to(tl.float32)
        if tl.program_id(2) * BLOCK_D >= D - RD:
            local = tl.arange(0, BLOCK_D)
            partner = tl.broadcast_to((local ^ 1)[None, :], (BLOCK_H, BLOCK_D))
            b = tl.gather(o, partner, 1)
            angle = tl.load(positions_ptr + t * stride_pos).to(tl.float32) * tl.load(
                freqs_ptr + ((d - (D - RD)) // 2) * stride_freq)
            sine = libdevice.sin(angle)
            rotated = libdevice.add_rn(libdevice.mul_rn(o, libdevice.cos(angle)[None, :]),
                libdevice.mul_rn(b, tl.where(d % 2 == 0, sine, -sine)[None, :]))
            o = rotated.to(tl.bfloat16).to(tl.float32)
        o = tl.reshape(qdq_fp8_tile(tl.reshape(o, (BLOCK_H * BLOCK_D // 32, 32))),
                       (BLOCK_H, BLOCK_D))
    tl.store(
        out_ptr + t * stride_ot + heads[:, None] * stride_oh + d[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None] & dmask[None, :],
    )


def sparse_attention_triton(
    q: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    comp_cache: torch.Tensor | None,
    comp_block_table: torch.Tensor | None,
    comp_local: torch.Tensor | None,
    req: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
    *, post_rope: bool = False, final_out=None, scores_out=None,
) -> torch.Tensor:
    """Attend with the seed schedule; optionally consume post-RoPE queries
    and absorb the output rounding/RoPE/QDQ into the ordered-value store."""
    n, h, d = q.shape
    out = final_out if final_out is not None else torch.empty(
        (n, h, d), dtype=torch.bfloat16, device=q.device)
    if n == 0:
        if final_out is not None:
            out.zero_()
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
    n_window = swa_indices.shape[1]
    window_tiles = triton.cdiv(n_window, _BLOCK_N)
    score_tiles = max(window_tiles + triton.cdiv(n_comp, _BLOCK_N), 1)
    score_width = score_tiles * _BLOCK_N
    scores = (scores_out[:n, :, :score_width] if scores_out is not None else
              torch.empty((n, h, score_width), dtype=torch.float32, device=q.device))
    # The kernels address scores with this physical head stride.
    score_width = scores.stride(1)
    rotated_q = q if post_rope else torch.empty((n, h, d), dtype=torch.bfloat16, device=q.device)
    req_i32 = req if post_rope else req.to(torch.int32).contiguous()
    grid = (n, triton.cdiv(h, _BLOCK_H))
    if not post_rope:
        _query_rope_kernel[(n, h)](
            q, positions, freqs, rotated_q,
            q.stride(0), q.stride(1), positions.stride(0), freqs.stride(0),
            rotated_q.stride(0), rotated_q.stride(1),
            D=d, RD=2 * freqs.numel(), num_warps=4,
        )
    _score_prepass_kernel[(*grid, score_tiles)](
        rotated_q,
        swa_cache,
        swa_indices,
        comp,
        comp_bt,
        comp_idx,
        req_i32,
        scores,
        h,
        n_window,
        n_comp,
        scale,
        rotated_q.stride(0),
        rotated_q.stride(1),
        swa_cache.stride(0),
        swa_cache.stride(1),
        swa_indices.stride(0),
        s_cb,
        s_ce,
        s_bt,
        s_ci,
        score_width,
        WINDOW_TILES=window_tiles,
        SWA_E=swa_cache.shape[1],
        COMP_E=comp_e,
        HAS_COMP=has_comp,
        D=d,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    value_grid = (out.shape[0], grid[1]) if final_out is not None else grid
    _ordered_value_kernel[(*value_grid, triton.cdiv(d, _BLOCK_D))](
        scores,
        swa_cache,
        swa_indices,
        comp,
        comp_bt,
        comp_idx,
        req_i32,
        attn_sink,
        out,
        positions,
        freqs,
        n,
        h,
        n_window,
        n_comp,
        score_width,
        swa_cache.stride(0),
        swa_cache.stride(1),
        swa_indices.stride(0),
        s_cb,
        s_ce,
        s_bt,
        s_ci,
        out.stride(0),
        out.stride(1),
        stride_pos=positions.stride(0),
        stride_freq=freqs.stride(0),
        RD=2 * freqs.numel(),
        EPILOGUE=final_out is not None,
        SWA_E=swa_cache.shape[1],
        COMP_E=comp_e,
        HAS_COMP=has_comp,
        D=d,
        BLOCK_D=_BLOCK_D,
        WINDOW_TILES=window_tiles,
        WINDOW_INDEX_BLOCK=triton.next_power_of_2(max(n_window, 1)),
        COMP_INDEX_BLOCK=triton.next_power_of_2(max(n_comp, 1)),
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    return out


# ---------------------------------------------------------------- output


@triton.jit(do_not_specialize=["n_tokens"])
def _output_kernel(
    o_ptr, positions_ptr, freqs_ptr, out_ptr, n_tokens,
    stride_ot: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
    stride_pos: tl.constexpr, stride_freq: tl.constexpr,
    stride_yt: tl.constexpr, stride_yh: tl.constexpr, stride_yd: tl.constexpr,
    D: tl.constexpr, RD: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, D)
    if t < n_tokens:
        x = tl.load(o_ptr + t * stride_ot + h * stride_oh + d * stride_od).to(tl.float32)
        r = tl.arange(0, RD)
        a = tl.gather(x, D - RD + r, axis=0)
        b = tl.gather(x, D - RD + (r ^ 1), axis=0)
        pos = tl.load(positions_ptr + t * stride_pos).to(tl.float32)
        freq = tl.load(freqs_ptr + (r // 2) * stride_freq)
        angle = pos * freq
        sine = libdevice.sin(angle)
        signed_sine = tl.where(r % 2 == 0, sine, -sine)
        rotated = (a * libdevice.cos(angle) + b * signed_sine).to(tl.bfloat16).to(tl.float32)
        tail = tl.gather(rotated, tl.maximum(d - (D - RD), 0), axis=0)
        x = tl.where(d >= D - RD, tail, x)
        y = qdq_fp8_tile(tl.reshape(x, (D // 32, 32)))
        y = tl.reshape(y, (D,))
    else:
        y = tl.full((D,), 0, tl.float32)
    tl.store(out_ptr + t * stride_yt + h * stride_yh + d * stride_yd, y.to(tl.bfloat16))


def output_qdq(o, positions, freqs, out):
    if out.shape[0]:
        _output_kernel[(out.shape[0], o.shape[1])](
            o, positions, freqs, out, positions.shape[0],
            o.stride(0), o.stride(1), o.stride(2), positions.stride(0), freqs.stride(0),
            out.stride(0), out.stride(1), out.stride(2), D=o.shape[2], RD=2 * freqs.numel(),
            num_warps=4, enable_fp_fusion=False,
        )


# ---------------------------------------------------------------- entry points


def supports(
    n_local_heads: int,
    head_dim: int,
    rope_dim: int,
    n_local_groups: int,
    window_size: int,
    index_topk: int,
    compress_ratio: int,
    kv_dtype: torch.dtype,
    capability: tuple[int, int] | None,
) -> str | None:
    """None when a layer has the validated configuration, else why not."""
    if capability is None or capability[0] != 8:
        return f"needs compute capability 8.x, got {capability}"
    got = (n_local_heads, head_dim, rope_dim, n_local_groups, window_size)
    want = (N_LOCAL_HEADS, HEAD_DIM, ROPE_DIM, N_LOCAL_GROUPS, WINDOW)
    if got != want:
        return (
            "validated for (local heads, head dim, rope dim, local groups, window) "
            f"= {want}, got {got}"
        )
    if compress_ratio and index_topk != TOPK:
        return f"validated for a compressed top-k of {TOPK}, got {index_topk}"
    if kv_dtype != torch.bfloat16:
        return f"validated for BF16 caches, got {kv_dtype}"
    return None


def prologue(qr_kv, q_norm, kv_norm, positions, freqs, swa_cache, swa_slots, eps):
    """``qr`` [T, q_lora] BF16, and the window rows of ``swa_slots`` written
    into ``swa_cache`` (module docstring)."""
    return fused_prologue(
        qr_kv, q_norm, kv_norm, positions, freqs, swa_cache, swa_slots, eps
    )


def attend(
    q,
    positions,
    freqs,
    swa_cache,
    swa_indices,
    comp_cache,
    comp_block_table,
    comp_local,
    req,
    attn_sink,
    scale,
    n_groups,
    out,
):
    """``out[:T]`` = the ``wo_a`` input of the pre-RoPE query ``q``,
    ``out[T:]`` = 0 (module docstring). ``n_groups`` is part of the call's
    contract: the FP8 blocks of 32 never cross a head, so the group split
    does not change any block or scale."""
    T = positions.shape[0]
    o = sparse_attention_triton(
        q[:T], positions, freqs, swa_cache, swa_indices, comp_cache,
        comp_block_table, comp_local, req, attn_sink, scale,
    )
    output_qdq(o, positions, freqs, out)
