# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 delayed mHC finish v2 on sm_80 (``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2``).

The finish after kernel A (``ops/sm80_mhc_fused.py``) as a Gluon kernel,
``_delayed_finish_v2_kernel``, grid (tokens, 2), 4 warps, with the contract,
outputs and rounding points of the Triton ``_delayed_finish_kernel`` it
replaces. New arithmetic: the FP64 reciprocals and the reciprocal square root
of the coefficient CTA are MUFU approximations refined by Newton steps (not
``div.rn.f64``), the mean of the squares is a product with the FP64 constant
1 / (4 hidden), and the Sinkhorn runs in scale-vector form. Held to the FP64
rule against the served kernel on real captured inputs (errors 1.000x of it
at every dispatched row count) for exactly the configuration below
(:data:`HIDDEN`, :data:`SINK`, :data:`MAX_ROWS`, hc 4, with the norm, sm_80);
the wrappers take it nowhere else. Kernel A runs with ``hs_reuse`` on this path (the same
bits, less shared memory from 17 rows).

CTA (t, 0), the coefficients, in FP64 from kernel A's FP32 partials, each output
rounded once to FP32:

- every load (the KS x 24 projection partials, the KS squared-sum partials, the
  base and scale) is issued before the first reduction; each warp reduces its
  quarter of the partials, the four warp partials meet once in shared memory
  (one CTA barrier) and every thread sums the four it needs in-thread;
- pre/post: one value per lane, ``sigmoid(a m + b)`` with libdevice's exp;
- comb: the softmax with one value per lane, then the SINK - 1 (row, column)
  Sinkhorn pairs in scale-vector form, A = diag(R) V diag(C): a row norm sets
  R_r <- R_r / (R_r S_r + eps) with S = V C, i.e. U = 1/R <- S + eps U, the
  columns likewise with W = 1/C. One reciprocal per scale (MUFU's approximate
  FP64 reciprocal refined by Newton) instead of one rounding per element; a
  thread holds a 2 x 2 block, so each row or column sum is one shuffle.

CTA (t, 1), the layer input: the streams collapsed with the CARRIED pre-mix,
exact in FP64 and rounded to FP32 then BF16 (where the reference rounds its FP32
sum), then the RMSNorm in FP32, its statistic from that BF16 value and
``weight * (x * rn)`` rounded to BF16. One pass over the streams: every chunk's
stream and norm-weight loads issued first, the BF16 collapse parked in shared
memory for the scaling.

``PART`` 1 or 2 launches only the coefficient or only the layer-input CTA of
each row (grid (tokens, 1)), as the Triton finish does
(``VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM``, ``ops/sm80_mhc_side.py``).

CUDA graphs: :data:`FINISH_V2_WARMUP` compiles every specialisation the
wrappers can launch on this path before capture (registered by ``mhc_ops``
in ``sm80_model.py``); the kernels keep no state and allocate nothing.
"""

import contextlib
from dataclasses import dataclass

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel

# The validated configuration: hidden 5120, 20 Sinkhorn iterations, 1..2048
# rows (the served prefill chunk; every decode, verify and draft row count).
HIDDEN = 5120
SINK = 20
MAX_ROWS = 2048
# Row counts compiled before capture besides the capture sizes: one of each
# specialisation of kernel A (transposed form, 9..16 rows, 17+ rows with the
# staging reuse) and of the finish (1 row, rows divisible by 16, others).
WARMUP_ROWS = (1, 2, 4, 8, 9, 16, 17, 32)
# Hidden chunk of the collapse CTA.
BH_FIN = 1024


@gluon.jit
def _rcp_approx64(x):
    return gl.inline_asm_elementwise("rcp.approx.ftz.f64 $0, $1;", "=d,d", [x],
                                     dtype=gl.float64, is_pure=True, pack=1)


@gluon.jit
def _rcp64(x):
    """1 / ``x`` in FP64 for a positive normal ``x``: rcp.approx.ftz.f64 refined
    by one cubic Newton step r + r (e + e^2), e = 1 - x r, within an ulp of the
    IEEE quotient (``div.rn.f64`` is a long software sequence)."""
    r = _rcp_approx64(x)
    e = gl.fma(-x, r, 1.0)
    return gl.fma(r, gl.fma(e, e, e), r)


@gluon.jit
def _rcp64_q(x):
    """1 / ``x`` with one quadratic Newton step: about 2^-40 relative
    (rcp.approx.ftz.f64 reads only 20 mantissa bits of ``x``, about 2^-20
    before the step): the Sinkhorn scales, far below the FP32 outputs'
    resolution."""
    r = _rcp_approx64(x)
    return gl.fma(r, gl.fma(-x, r, 1.0), r)


@gluon.jit
def _rsqrt64(x):
    """1 / sqrt(``x``) in FP64 for a positive ``x``: rsqrt.approx.ftz.f64 and
    two Newton steps y (1.5 - x/2 y^2); +inf gives 0 as the IEEE form does."""
    y = gl.inline_asm_elementwise("rsqrt.approx.ftz.f64 $0, $1;", "=d,d", [x],
                                  dtype=gl.float64, is_pure=True, pack=1)
    h = 0.5 * x
    y = y * gl.fma(-h * y, y, 1.5)
    y = y * gl.fma(-h * y, y, 1.5)
    return gl.where(x == float("inf"), 0.0, y)


@gluon.jit
def _collapse(x0, x1, x2, x3, w0, w1, w2, w3):
    """sum_j pre_mix[j] * stream_j: the products (BF16 x FP32) and their sum
    exact in FP64, rounded to FP32 then BF16; returned as FP32."""
    f64: gl.constexpr = gl.float64
    p = ((x0.to(f64) * w0 + x1.to(f64) * w1) + x2.to(f64) * w2) + x3.to(f64) * w3
    return p.to(gl.float32).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _partials(mix_ptr, sqr_ptr, M, m, K0: gl.constexpr, KN: gl.constexpr, KS: gl.constexpr,
              NO: gl.constexpr, L3: gl.constexpr):
    """Partial rows K0 .. K0+KN-1 (masked at KS) of token ``m``: the [NW, KN/NW, 32]
    projection tile and the [NW, KN/NW] squared sums (FP32)."""
    NW: gl.constexpr = 4
    KW: gl.constexpr = KN // NW
    LK: gl.constexpr = gl.SliceLayout(2, L3)
    wi = gl.arange(0, NW, layout=gl.SliceLayout(1, LK))
    ki = gl.arange(0, KW, layout=gl.SliceLayout(0, LK))
    ci = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    ks2 = K0 + wi[:, None] * KW + ki[None, :]
    ks3 = ks2[:, :, None]
    n3 = (ci[None, None, :] + 24) % 32
    tile = gl.load(mix_ptr + (ks3 * M + m) * NO + n3, mask=(ks3 < KS) & (n3 < NO), other=0.0)
    sqv = gl.load(sqr_ptr + ks2 * M + m, mask=ks2 < KS, other=0.0)
    return tile, sqv


@gluon.jit
def _delayed_finish_v2_kernel(
    mix_ptr, sqr_ptr, scale_ptr, base_ptr, rc_ptr, pre_in_ptr, nw_ptr,
    post_ptr, comb_ptr, pre_ptr, li_ptr, M,
    rms_eps, hc_eps, post_mult, norm_eps,
    HIDDEN: gl.constexpr, KS: gl.constexpr, KA: gl.constexpr, KB: gl.constexpr, SINK: gl.constexpr,
    HAS_PRE_IN: gl.constexpr, HAS_NORM: gl.constexpr, BH: gl.constexpr,
    PART: gl.constexpr = 0,
):
    NW: gl.constexpr = 4
    HC: gl.constexpr = 4
    NO: gl.constexpr = 24
    f64: gl.constexpr = gl.float64
    m = gl.program_id(0)
    # PART 0: CTA (m, 0) the coefficients, (m, 1) the layer input; PART 1 or
    # 2: only the coefficient or only the layer-input CTA of row m
    if PART == 0:
        coefficients = gl.program_id(1) == 0
    else:
        coefficients = PART == 1
    if coefficients:
        # The KS partial rows in two tiles, KA (a power of two) + KB (the rest, padded to a
        # power of two), so a warp converts no more padded rows than needed: [NW, KA/NW, 32]
        # and [NW, KB/NW, 32], warp w holding rows w*KA/NW .. of each, a thread 4 columns;
        # columns remapped n = (c + 24) mod 32, so c 8..15 = pre, post and c 16..31 = the
        # comb (c 0..7, n >= 24, masked). sqr in the [NW, K*/NW] slices.
        LA3: gl.constexpr = gl.BlockedLayout([1, KA // NW // 4, 4], [1, 4, 8], [NW, 1, 1], [2, 1, 0])
        tile, sqv = _partials(mix_ptr, sqr_ptr, M, m, 0, KA, KS, NO, LA3)
        wp = gl.sum(tile.to(f64), axis=1)  # [NW, 32]
        wsq = gl.sum(sqv.to(f64), axis=1)  # [NW]
        if KB > 0:
            LB3: gl.constexpr = gl.BlockedLayout([1, KB // NW // 4, 4], [1, 4, 8], [NW, 1, 1], [2, 1, 0])
            tb, sb = _partials(mix_ptr, sqr_ptr, M, m, KA, KB, KS, NO, LB3)
            wp = wp + gl.convert_layout(gl.sum(tb.to(f64), axis=1), gl.SliceLayout(1, LA3))
            wsq = wsq + gl.convert_layout(gl.sum(sb.to(f64), axis=1), gl.SliceLayout(1, gl.SliceLayout(2, LA3)))
        # pre / post: [2, 4], row 0 = pre, row 1 = post, one value per lane
        L3P: gl.constexpr = gl.BlockedLayout([NW, 1, 1], [1, 2, 16], [1, 1, NW], [2, 1, 0])
        LPP: gl.constexpr = gl.SliceLayout(0, L3P)
        rP = gl.arange(0, 2, layout=gl.SliceLayout(1, LPP))
        cP = gl.arange(0, HC, layout=gl.SliceLayout(0, LPP))
        pidx = rP[:, None] * HC + cP[None, :]
        bp = gl.load(base_ptr + pidx).to(f64)
        # comb for the softmax: [4, 4], one value per lane (lanes r * 8 + c, c < 4)
        L3C: gl.constexpr = gl.BlockedLayout([NW, 1, 1], [1, 4, 8], [1, 1, NW], [2, 1, 0])
        LC: gl.constexpr = gl.SliceLayout(0, L3C)
        rC = gl.arange(0, HC, layout=gl.SliceLayout(1, LC))
        cC = gl.arange(0, HC, layout=gl.SliceLayout(0, LC))
        bc = gl.load(base_ptr + 2 * HC + rC[:, None] * HC + cC[None, :]).to(f64)
        s0 = gl.load(scale_ptr).to(f64)
        s1 = gl.load(scale_ptr + 1).to(f64)
        s2 = gl.load(scale_ptr + 2).to(f64)

        SH2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        SH1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        pbuf = gl.allocate_shared_memory(f64, [NW, 32], SH2)
        qbuf = gl.allocate_shared_memory(f64, [NW], SH1)
        pbuf.store(wp)
        qbuf.store(wsq)
        gl.barrier()
        LQ: gl.constexpr = gl.BlockedLayout([NW], [32], [NW], [0])
        sq = gl.sum(qbuf.load(LQ), axis=0)
        # the mean as a product with the FP64 constant 1/(4 HIDDEN) (a div.rn.f64 is a
        # long software sequence on the critical path)
        rsq = _rsqrt64(sq * (1.0 / (HC * HIDDEN)) + rms_eps.to(f64))
        eps = hc_eps.to(f64)
        pv = pbuf.reshape([NW, 8, HC])
        pp = gl.sum(pv.slice(2, 2, dim=1).load(L3P), axis=0)  # [2, 4] in LPP
        pc = gl.sum(pv.slice(4, 4, dim=1).load(L3C), axis=0)  # [4, 4] in LC

        v = (pp * rsq) * gl.where(pidx < HC, s0, s1) + bp
        # exp(-v) kept finite for the reciprocal: below -708 the sigmoid is < 1e-307,
        # an FP32 0 (post) and pre = eps either way; NaN stays NaN
        sg = _rcp64(1.0 + libdevice.exp(-gl.where(v < -708.0, -708.0, v)))
        is_pre = pidx < HC
        pre = (sg + eps).to(gl.float32)
        post = (post_mult.to(f64) * sg).to(gl.float32)
        gl.store(pre_ptr + m * HC + gl.where(is_pre, pidx, 0), pre, mask=is_pre)
        gl.store(post_ptr + m * HC + gl.where(is_pre, 0, pidx - HC), post, mask=~is_pre)

        # comb: softmax over each row + eps, a column normalisation, SINK - 1 pairs
        c = (pc * rsq) * s2 + bc
        c = libdevice.exp(c - gl.max(c, axis=1)[:, None])
        c = c * _rcp64(gl.sum(c, axis=1))[:, None] + eps
        c = c * _rcp64(gl.sum(c, axis=0) + eps)[None, :]
        LS: gl.constexpr = gl.BlockedLayout([2, 2], [2, 16], [1, NW], [1, 0])
        V = gl.convert_layout(c, LS)
        U = gl.full([HC], 1.0, f64, gl.SliceLayout(1, LS))
        W = gl.full([HC], 1.0, f64, gl.SliceLayout(0, LS))
        R = U
        C = W
        for _ in gl.static_range(SINK - 1):
            U = gl.sum(V * C[None, :], axis=1) + eps * U
            R = _rcp64_q(U)
            W = gl.sum(V * R[:, None], axis=0) + eps * W
            C = _rcp64_q(W)
        c = (V * R[:, None]) * C[None, :]
        rS = gl.arange(0, HC, layout=gl.SliceLayout(1, LS))
        cS = gl.arange(0, HC, layout=gl.SliceLayout(0, LS))
        gl.store(comb_ptr + m * (HC * HC) + rS[:, None] * HC + cS[None, :], c.to(gl.float32))
    else:
        if HAS_PRE_IN:
            w0 = gl.load(pre_in_ptr + m * HC)
            w1 = gl.load(pre_in_ptr + m * HC + 1)
            w2 = gl.load(pre_in_ptr + m * HC + 2)
            w3 = gl.load(pre_in_ptr + m * HC + 3)
        else:
            # the identity pre-mix: stream 0, the others weighted by zero
            w0 = 1.0
            w1 = 0.0
            w2 = 0.0
            w3 = 0.0
        NCH: gl.constexpr = (HIDDEN + BH - 1) // BH
        LH: gl.constexpr = gl.BlockedLayout([8], [32], [NW], [0])
        rc_row = rc_ptr + m.to(gl.int64) * (HC * HIDDEN)
        li_row = li_ptr + m.to(gl.int64) * HIDDEN
        hb = gl.arange(0, BH, layout=LH)
        # every chunk's stream loads issued before any arithmetic
        xs = ()
        for i in gl.static_range(NCH):
            h = i * BH + hb
            hm = h < HIDDEN
            xs = xs + ((gl.load(rc_row + h, mask=hm, other=0.0),
                        gl.load(rc_row + HIDDEN + h, mask=hm, other=0.0),
                        gl.load(rc_row + 2 * HIDDEN + h, mask=hm, other=0.0),
                        gl.load(rc_row + 3 * HIDDEN + h, mask=hm, other=0.0)),)
        if HAS_NORM:
            ws = ()
            for i in gl.static_range(NCH):
                h = i * BH + hb
                ws = ws + (gl.load(nw_ptr + h, mask=h < HIDDEN, other=0.0),)
        if HAS_NORM:
            # the BF16 collapse parked in shared memory (each thread reads back its own)
            SHB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
            obuf = gl.allocate_shared_memory(gl.bfloat16, [triton.next_power_of_2(NCH) * BH], SHB)
            acc = gl.zeros([BH], gl.float32, LH)
            for i in gl.static_range(NCH):
                x = xs[i]
                ob = _collapse(x[0], x[1], x[2], x[3], w0, w1, w2, w3)
                acc += ob * ob
                obuf.slice(i * BH, BH).store(ob.to(gl.bfloat16))
            ms = gl.div_rn(gl.sum(acc, axis=0), HIDDEN)
            rn = gl.div_rn(1.0, gl.sqrt_rn(ms + norm_eps))
            for i in gl.static_range(NCH):
                h = i * BH + hb
                hm = h < HIDDEN
                ob = obuf.slice(i * BH, BH).load(LH).to(gl.float32)
                w = ws[i].to(gl.float32)
                gl.store(li_row + h, (w * (ob * rn)).to(gl.bfloat16), mask=hm)
        else:
            for i in gl.static_range(NCH):
                h = i * BH + hb
                x = xs[i]
                ob = _collapse(x[0], x[1], x[2], x[3], w0, w1, w2, w3)
                gl.store(li_row + h, ob.to(gl.bfloat16), mask=h < HIDDEN)


def _ka(ks: int) -> int:
    """The finish's first partial tile: the largest power of two <= ``ks``, at least
    16 rows (4 warps x 4)."""
    return max(16, 1 << (ks.bit_length() - 1))


def _kb(ks: int) -> int:
    """The finish's second partial tile: the rows past ``_ka``, padded to a power of
    two of at least 16 rows; 0 when there are none."""
    rest = ks - _ka(ks)
    return 0 if rest <= 0 else max(16, triton.next_power_of_2(rest))


def gate_closed(hidden: int, sinkhorn_repeat: int, rows: int) -> str | None:
    """Why a call is outside the validated configuration (None: inside it)."""
    if hidden != HIDDEN:
        return f"hidden size {hidden}, validated for {HIDDEN} only"
    if sinkhorn_repeat != SINK:
        return f"{sinkhorn_repeat} Sinkhorn iterations, validated for {SINK} only"
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows, validated for 1..{MAX_ROWS}"
    return None


def delayed_finish_v2(
    mix: torch.Tensor,
    sqr: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rc: torch.Tensor,
    pre_in: torch.Tensor,
    nw: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_pre: torch.Tensor,
    layer_input: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    hc_post_mult_value: float,
    norm_eps: float,
    sinkhorn_repeat: int,
    has_pre_in: bool,
    has_norm: bool,
    part: int = 0,
) -> None:
    """The finish v2 into ``post``, ``comb``, ``next_pre`` and ``layer_input``,
    the arguments of the served finish kernel (``sm80_mhc_fused._delayed``);
    ``part`` as its ``PART``."""
    ksb, T = mix.shape[0], mix.shape[1]
    _delayed_finish_v2_kernel[(T, 2 if part == 0 else 1)](
        mix, sqr, hc_scale, hc_base, rc, pre_in, nw,
        post, comb, next_pre, layer_input, T,
        float(rms_eps), float(hc_eps), float(hc_post_mult_value), float(norm_eps),
        HIDDEN=rc.shape[-1],
        KS=ksb,
        KA=_ka(ksb),
        KB=_kb(ksb),
        SINK=int(sinkhorn_repeat),
        HAS_PRE_IN=has_pre_in,
        HAS_NORM=has_norm,
        BH=BH_FIN,
        PART=part,
        num_warps=4,
    )


class FinishV2Warmup(VllmJitKernel["FinishV2Warmup.Key"]):
    """Boot-time compile of the finish v2 path (kernel A with ``hs_reuse`` and
    :func:`delayed_finish_v2`) in every form the decoder layers call (fused
    post + pre and pre only, with and without a carried pre-mix, with the
    norm), at the graph capture sizes and :data:`WARMUP_ROWS`."""

    @dataclass(frozen=True)
    class Key:
        rows: int

    def get_warmup_keys(self, vllm_config) -> list["FinishV2Warmup.Key"]:
        sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
        rows = {int(s) for s in sizes if 1 <= int(s) <= MAX_ROWS}
        return [self.Key(r) for r in sorted(rows.union(WARMUP_ROWS))]

    def compile(self, compile_key: "FinishV2Warmup.Key") -> None:
        from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side
        from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

        if not (fused.FINISH_V2 and fused.FINISH_KERNEL):
            raise RuntimeError("the mHC finish v2 warmup runs with the finish v2 selected")
        dev = torch.device("cuda", torch.cuda.current_device())
        T, hc, bf16 = compile_key.rows, 4, torch.bfloat16
        nout = (2 + hc) * hc
        common = (
            torch.zeros(nout, hc * HIDDEN, dtype=torch.float32, device=dev),
            torch.ones(3, dtype=torch.float32, device=dev),
            torch.zeros(nout, dtype=torch.float32, device=dev),
            1e-20,
            1e-6,
            1e-6,
            2.0,
            SINK,
        )
        residual = torch.zeros(T, hc, HIDDEN, dtype=bf16, device=dev)
        norm = torch.ones(HIDDEN, dtype=bf16, device=dev)
        carried = torch.full((T, hc), 0.25, dtype=torch.float32, device=dev)
        # with VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM taken by the decoder layers
        # (sm80_mhc_side.layer_scope) also the two kinds of CTA launched
        # apart, as inside a decoder layer's forward
        scopes = [contextlib.nullcontext()]
        if sm80_mhc_side.IN_USE:
            scopes.append(sm80_mhc_side.LAYER)
        for scope in scopes:
            with scope:
                for pre_mix in (None, carried):
                    fused.mhc_pre_delayed(residual, *common, pre_mix=pre_mix, norm_weight=norm)
                    fused.mhc_fused_post_pre_delayed(
                        torch.zeros(T, HIDDEN, dtype=bf16, device=dev),
                        residual,
                        torch.ones(T, hc, 1, dtype=torch.float32, device=dev),
                        torch.full((T, hc, hc), 0.25, dtype=torch.float32, device=dev),
                        *common,
                        pre_mix=pre_mix,
                        norm_weight=norm,
                    )
        torch.cuda.synchronize(dev)


FINISH_V2_WARMUP = FinishV2Warmup()
