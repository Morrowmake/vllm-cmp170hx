# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 delayed mHC on sm_80 with the sm_80 mHC kernel A.

The same contracts, arguments and BF16 rounding points as ``ops/sm80_mhc.py``
(the reference's FP32 formulas in torch, about 155 kernels per call, 134 of
them the Sinkhorn loop), selected with ``VLLM_DEEPSEEK_V41_FUSED_MHC`` in
``sm80_model.py``. Kernel A (below) always runs; the finish after it is the
Triton kernel B with ``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH`` (two kernels per call),
else the reference's torch formulas on kernel A's partials
(:func:`_torch_finish`: ``hc_split_sinkhorn`` in FP64 on the partials summed
in a fixed FP64 order, then ``hc_collapse`` and ``rms_norm`` of
``ops/sm80_mhc.py`` in FP32). The two switches are separate so that kernel B,
a new kernel, is judged on its own. Ops:

- ``mhc_fused_post_pre_delayed``: the previous sublayer's post, then the
  delayed pre of the new streams;
- ``mhc_pre_delayed``: the delayed pre alone (the model entry and the Engram
  layers, whose post runs before the Engram injection);
- ``mhc_post``: ``sm80_mhc.mhc_post`` itself (the backbone's final post and the
  Engram layers' post stay torch: one call each per layer that has them).

Kernel A is the GLM-5.3-Flash sm_80 decode kernel
(``vllm_sm80/kernels/mhc/decode.py``, ``post_prenorm``): the post step
``post * x + comb^T @ residual`` stored BF16, and the projection on the
streams onto the 24 mixes as split-K partials, three TF32 passes on the
tensor cores (FP32-equivalent), with the squared-sum partials. Three of its
compile-time options serve this model: ``round_mix`` (the projection and
the squared sum read the BF16 streams, as the reference computes its
coefficients from the BF16 stream), ``has_post=False`` (the pre-only
call) and ``fp64_acc`` (each slice's partials accumulated in FP64 and
rounded once to FP32: at 4 x 5120 streams the FP32 accumulation was
kernel A's largest error source in the coefficients).

Kernel B (``_delayed_finish_kernel``, grid ``(tokens, 2)``) is the delayed
finish, in Triton:

- CTA ``(t, 0)``, in FP64 (full rate on this card) from the FP32 partials,
  each output rounded once to FP32: the partials reduced,
  ``rsqrt(mean(x^2) + rms_eps)``, pre ``sigmoid(a0 m + b) + eps`` (the next
  pre-mix), post ``mult * sigmoid(a1 m + b)``, comb
  ``softmax(a2 m + b) + eps``, a column normalisation and ``iters - 1``
  row/column Sinkhorn passes. These few values per token carry the torch
  path's FP32 rounding error at the ulp level, so the FP64 rule's 1.25x
  bound on their maximum would otherwise hang on single ulps;
- CTA ``(t, 1)``: the streams collapsed with the CARRIED pre-mix (the
  delayed step: this call's pre is only an output), exact in FP64 and
  rounded to FP32 then BF16 (where the reference rounds its FP32 sum), then
  the RMSNorm in FP32 as torch does, its statistic from that BF16 value and
  ``weight * x`` rounded to BF16. The two CTAs share nothing.

``PART`` 1 or 2 launches only the coefficient or only the layer-input CTA of
each row (grid ``(tokens, 1)``): with ``VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM`` a
decoder layer's call of up to 64 rows launches the two apart, the coefficients
beside the sublayer's all-reduce (``ops/sm80_mhc_side.py``). Both finish
kernels take it.

With ``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2`` (and the finish kernel
selected) a call inside the validated configuration (hidden 5120, 20
Sinkhorn iterations, up to 2048 rows, with the norm, sm_80; :func:`finish_v2_unavailable`
at model construction) runs kernel A with its staging reuse and the Gluon
finish of ``ops/sm80_mhc_finish_v2.py`` instead of ``_delayed_finish_kernel``;
every other call runs the kernels above unchanged.

The rounding points are the reference's; the arithmetic differs from the
torch path in summation order, in the projection (split-K tensor core
passes against cuBLAS SGEMM) and in the FP64 coefficients, and is held to
the FP64 rule against it
(``tests/models/deepseek_v41/sm80/test_sm80_mhc_fused_gpu.py``,
``fp64_cases.py`` family ``mhc_delayed``). Shapes: ``hc == 4``, hidden a
multiple of 32, BF16 streams, FP32 coefficients and ``fn``.
"""

import os

import torch
from triton.language.extra import libdevice

import vllm.envs as envs
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side
from vllm_sm80.models.deepseek_v41.ops.sm80_mhc import mhc_post
from vllm_sm80.kernels.mhc.decode import HB_GLU, post_prenorm
from vllm.triton_utils import tl, triton

__all__ = ["mhc_fused_post_pre_delayed", "mhc_post", "mhc_pre_delayed", "supports"]

HC = 4
NOUT = (2 + HC) * HC
# Kernel B for the finish (VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH), read once per
# process; the tests set it directly.
FINISH_KERNEL = bool(envs.VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH)
# The finish v2 (VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2), read once per process;
# cleared by ``mhc_ops`` when finish_v2_unavailable gives a reason; the tests
# set it directly.
FINISH_V2 = bool(envs.VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2)
# Hidden chunk of the collapse CTA (two passes over the streams: the norm's
# statistic, then the scaled output).
_BH = 1024
# libdevice's exp on CUDA; the Triton interpreter (CPU tests)
# has no libdevice and takes tl.exp.
_ACCURATE_EXP = os.environ.get("TRITON_INTERPRET", "0") != "1"
# The CTAs a finish launch runs: each row's two (grid (T, 2)), or only its
# coefficient or only its layer-input CTA (grid (T, 1)).
PART_BOTH, PART_COEFFICIENTS, PART_LAYER_INPUT = 0, 1, 2


def supports(hc_mult: int, hidden_size: int) -> str | None:
    """None when the fused kernels serve these shapes, else the reason."""
    if hc_mult != HC:
        return f"hc_mult {hc_mult}, not {HC}"
    if hidden_size <= 0 or hidden_size % HB_GLU:
        return f"hidden size {hidden_size} is not a multiple of {HB_GLU}"
    return None


def finish_v2_unavailable(hc_mult: int, hidden_size: int, sinkhorn_repeat: int) -> str | None:
    """None when the finish v2 can serve the model's delayed mHC on this
    device, else why not: it replaces the finish kernel in the configuration
    it was validated for, on sm_80."""
    if not FINISH_KERNEL:
        return "it replaces the finish kernel: VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1 is required"
    reason = supports(hc_mult, hidden_size) or finish_v2.gate_closed(
        hidden_size, sinkhorn_repeat, 1
    )
    if reason is not None:
        return reason
    if torch.cuda.get_device_capability() != (8, 0):
        return "validated on sm_80 only"
    return None


@triton.jit
def _exp(v, ACCURATE: tl.constexpr):
    if ACCURATE:
        return libdevice.exp(v)
    else:
        return tl.exp(v)


@triton.jit
def _bf16(v):
    """``v`` rounded to BF16 (nearest, ties to even) as FP32, NaN kept: the
    value of ``v.to(tl.bfloat16)`` on CUDA, written out because the Triton
    interpreter (the CPU tests) truncates that conversion. Stores of the
    result to BF16 are then exact on both."""
    b = v.to(tl.int32, bitcast=True)
    r = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
    return tl.where(v == v, r.to(tl.float32, bitcast=True), v)


@triton.jit
def _collapse(rc_row, h, hm, w0, w1, w2, w3, HIDDEN: tl.constexpr):
    """sum_j pre_mix[j] * stream_j over ``h``, rounded to BF16 (as FP32). The
    products (FP32 x BF16) and their sum are exact in FP64; the sum is rounded
    to FP32 and then to BF16, where the reference rounds its FP32 sum."""
    f64 = tl.float64
    p0 = tl.load(rc_row + h, mask=hm, other=0.0).to(f64) * w0
    p1 = tl.load(rc_row + HIDDEN + h, mask=hm, other=0.0).to(f64) * w1
    p2 = tl.load(rc_row + 2 * HIDDEN + h, mask=hm, other=0.0).to(f64) * w2
    p3 = tl.load(rc_row + 3 * HIDDEN + h, mask=hm, other=0.0).to(f64) * w3
    return _bf16((((p0 + p1) + p2) + p3).to(tl.float32))


@triton.jit
def _delayed_finish_kernel(
    mix_ptr,
    sqr_ptr,
    scale_ptr,
    base_ptr,
    rc_ptr,
    pre_in_ptr,
    nw_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    li_ptr,
    M,
    rms_eps,
    hc_eps,
    post_mult,
    norm_eps,
    HIDDEN: tl.constexpr,
    KS: tl.constexpr,
    KSP2: tl.constexpr,
    SINK: tl.constexpr,
    HAS_PRE_IN: tl.constexpr,
    HAS_NORM: tl.constexpr,
    BH: tl.constexpr,
    ACCURATE: tl.constexpr,
    PART: tl.constexpr = 0,
):
    HCK: tl.constexpr = 4
    NO: tl.constexpr = 24
    m = tl.program_id(0)
    # PART 0: CTA (m, 0) the coefficients, (m, 1) the layer input; PART 1 or
    # 2: only the coefficient or only the layer-input CTA of row m
    if PART == 0:
        coefficients = tl.program_id(1) == 0
    else:
        coefficients = PART == 1
    if coefficients:
        k = tl.arange(0, KSP2)
        kmask = k < KS
        rows = k * M + m
        f64 = tl.float64
        sq = tl.sum(tl.load(sqr_ptr + rows, mask=kmask, other=0.0).to(f64), axis=0)
        rsq = 1.0 / tl.sqrt(sq / (HCK * HIDDEN) + tl.cast(rms_eps, f64))
        s0 = tl.load(scale_ptr).to(f64)
        s1 = tl.load(scale_ptr + 1).to(f64)
        s2 = tl.load(scale_ptr + 2).to(f64)
        eps = tl.cast(hc_eps, f64)
        # pre (mixes 0..3) and post (4..7)
        j = tl.arange(0, 2 * HCK)
        pp = tl.load(
            mix_ptr + rows[:, None] * NO + j[None, :], mask=kmask[:, None], other=0.0
        )
        pp = tl.sum(pp.to(f64), axis=0)
        v = (pp * rsq) * tl.where(j < HCK, s0, s1) + tl.load(base_ptr + j).to(f64)
        sg = 1.0 / (1.0 + _exp(-v, ACCURATE))
        is_pre = j < HCK
        pre = (sg + eps).to(tl.float32)
        post = (tl.cast(post_mult, f64) * sg).to(tl.float32)
        tl.store(pre_ptr + m * HCK + tl.where(is_pre, j, 0), pre, mask=is_pre)
        tl.store(
            post_ptr + m * HCK + tl.where(is_pre, 0, j - HCK), post, mask=~is_pre
        )
        # comb (8..23) as [row, col], then the Sinkhorn (in FP64 a product
        # with the reciprocal is as good as the quotient for an FP32 result)
        r = tl.arange(0, HCK)
        cidx = 2 * HCK + r[:, None] * HCK + r[None, :]
        ct = tl.load(
            mix_ptr + rows[:, None, None] * NO + cidx[None, :, :],
            mask=kmask[:, None, None],
            other=0.0,
        )
        c = (tl.sum(ct.to(f64), axis=0) * rsq) * s2 + tl.load(base_ptr + cidx).to(f64)
        c = _exp(c - tl.max(c, axis=1)[:, None], ACCURATE)
        c = c * (1.0 / tl.sum(c, axis=1))[:, None] + eps
        c = c * (1.0 / (tl.sum(c, axis=0) + eps))[None, :]
        for _ in tl.static_range(SINK - 1):
            c = c * (1.0 / (tl.sum(c, axis=1) + eps))[:, None]
            c = c * (1.0 / (tl.sum(c, axis=0) + eps))[None, :]
        cidx_out = m * (HCK * HCK) + r[:, None] * HCK + r[None, :]
        tl.store(comb_ptr + cidx_out, c.to(tl.float32))
    else:
        if HAS_PRE_IN:
            w0 = tl.load(pre_in_ptr + m * HCK)
            w1 = tl.load(pre_in_ptr + m * HCK + 1)
            w2 = tl.load(pre_in_ptr + m * HCK + 2)
            w3 = tl.load(pre_in_ptr + m * HCK + 3)
        else:
            # the identity pre-mix: stream 0, the others weighted by zero
            w0 = 1.0
            w1 = 0.0
            w2 = 0.0
            w3 = 0.0
        rc_row = rc_ptr + m.to(tl.int64) * (HCK * HIDDEN)
        li_row = li_ptr + m.to(tl.int64) * HIDDEN
        hb = tl.arange(0, BH)
        if HAS_NORM:
            acc = tl.zeros([BH], dtype=tl.float32)
            for h0 in tl.static_range(0, HIDDEN, BH):
                h = h0 + hb
                hm = h < HIDDEN
                ob = _collapse(rc_row, h, hm, w0, w1, w2, w3, HIDDEN)
                acc += ob * ob
            ms = tl.div_rn(tl.sum(acc, axis=0), HIDDEN)
            rn = tl.div_rn(1.0, tl.sqrt_rn(ms + norm_eps))
        for h0 in tl.static_range(0, HIDDEN, BH):
            h = h0 + hb
            hm = h < HIDDEN
            ob = _collapse(rc_row, h, hm, w0, w1, w2, w3, HIDDEN)
            if HAS_NORM:
                w = tl.load(nw_ptr + h, mask=hm, other=0.0).to(tl.float32)
                ob = w * (ob * rn)
            tl.store(li_row + h, _bf16(ob).to(tl.bfloat16), mask=hm)


def _sum_partials(p: torch.Tensor) -> torch.Tensor:
    """``p`` [KS, ...] summed over its first axis in FP64, in a fixed
    pairwise order of elementwise adds: the same bits for a token whatever
    the other tokens of the call (a torch reduction picks its order from the
    whole shape)."""
    p = p.double()
    while p.shape[0] > 1:
        half = p.shape[0] // 2
        rest = p[2 * half :]
        p = p[0 : 2 * half : 2] + p[1 : 2 * half : 2]
        if rest.shape[0]:
            p = torch.cat((p, rest))
    return p[0]


def _torch_finish(
    mix: torch.Tensor,
    sqr: torch.Tensor,
    streams: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
):
    """The finish in the reference's torch formulas on kernel A's partials:
    the projection and the squared sum reduced over the hidden slices, then
    ``Block.hc_mixes``' scaling and ``hc_split_sinkhorn`` in FP64, each
    coefficient rounded once to FP32 (in FP32 these few values carry the
    torch formulas' own rounding at the ulp level); the collapse with the
    carried pre-mix and the RMSNorm in FP32 as ``ops/sm80_mhc.py``."""
    T, hc, hidden = streams.shape
    rsqrt = torch.rsqrt(_sum_partials(sqr) / (hc * hidden) + rms_eps)
    mixes = _sum_partials(mix) * rsqrt.unsqueeze(-1)
    pre, post, comb = sm80_mhc.hc_split_sinkhorn(
        mixes,
        hc_scale.double(),
        hc_base.double(),
        hc,
        sinkhorn_repeat,
        hc_eps,
        hc_post_mult_value,
    )
    layer_input = sm80_mhc.hc_collapse(streams, pre_mix)
    if norm_weight is not None:
        layer_input = sm80_mhc.rms_norm(layer_input, norm_weight, norm_eps)
    return post.float().unsqueeze(-1), comb.float(), layer_input, pre.float()


def _delayed(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor | None,
    comb_res_mix: torch.Tensor | None,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
):
    """The post (``x`` given) or not, then the delayed pre. Returns the
    streams (with ``x``; else ``residual``), post [T, hc, 1], comb
    [T, hc, hc], the layer input [T, hidden] BF16 and the next pre-mix
    [T, hc]."""
    if hc_pre_eps != hc_sinkhorn_eps:
        raise ValueError("The DeepSeek-V4.1 reference uses one hc_eps")
    if sinkhorn_repeat < 1:
        raise ValueError(f"sinkhorn_repeat {sinkhorn_repeat} < 1")
    if residual.dim() != 3:
        raise ValueError(f"streams must be [tokens, hc, hidden], got {tuple(residual.shape)}")
    T, hc, hidden = residual.shape
    reason = supports(hc, hidden)
    if reason is not None:
        raise ValueError(f"fused delayed mHC: {reason}")
    assert residual.dtype == torch.bfloat16, residual.dtype
    assert fn.dtype == torch.float32 and fn.shape == (NOUT, hc * hidden), (fn.dtype, fn.shape)
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32
    # a previous call's deferred coefficients, before kernel A reads them
    sm80_mhc_side.flush()
    dev = residual.device
    res = residual.contiguous()
    post = torch.empty(T, hc, 1, dtype=torch.float32, device=dev)
    comb = torch.empty(T, hc, hc, dtype=torch.float32, device=dev)
    layer_input = torch.empty(T, hidden, dtype=residual.dtype, device=dev)
    next_pre = torch.empty(T, hc, dtype=torch.float32, device=dev)
    if T == 0:
        return (res if x is None else res.clone()), post, comb, layer_input, next_pre
    ksb = hidden // HB_GLU
    mix = torch.empty(ksb, T, NOUT, dtype=torch.float32, device=dev)
    sqr = torch.empty(ksb, T, dtype=torch.float32, device=dev)
    rc = torch.empty_like(res)
    fnc = fn.contiguous()
    # the finish v2: only inside its validated configuration, which includes
    # the finish kernel being selected and the norm (every decoder layer call
    # has one)
    v2 = (
        FINISH_V2
        and FINISH_KERNEL
        and norm_weight is not None
        and finish_v2.gate_closed(hidden, sinkhorn_repeat, T) is None
    )
    if x is None:
        # pre only: the kernel reads the streams themselves; x, post and comb
        # are placeholders of sufficient size (not used)
        post_prenorm(
            res, res, mix, mix, fnc, rc, mix, sqr, has_post=False, round_mix=True,
            fp64_acc=True, hs_reuse=v2,
        )
        streams = res
    else:
        assert x.dtype == torch.bfloat16 and x.shape == (T, hidden), (x.dtype, x.shape)
        assert post_layer_mix is not None and comb_res_mix is not None
        assert post_layer_mix.dtype == torch.float32 and comb_res_mix.dtype == torch.float32
        post_prenorm(
            x.contiguous(),
            res,
            post_layer_mix.reshape(T, hc).contiguous(),
            comb_res_mix.reshape(T, hc, hc).contiguous(),
            fnc,
            rc,
            mix,
            sqr,
            round_mix=True,
            fp64_acc=True,
            hs_reuse=v2,
        )
        streams = rc
    if not FINISH_KERNEL:
        return (
            streams,
            *_torch_finish(
                mix,
                sqr,
                rc,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                pre_mix,
                norm_weight,
                norm_eps,
            ),
        )
    if pre_mix is not None:
        assert pre_mix.dtype == torch.float32, pre_mix.dtype
        pre_in = pre_mix.reshape(T, hc).contiguous()
    else:
        pre_in = next_pre  # not read
    nw = layer_input if norm_weight is None else norm_weight.contiguous()
    scale, base = hc_scale.contiguous(), hc_base.contiguous()

    def finish(part: int) -> None:
        """The finish's CTAs of ``part`` (:data:`PART_BOTH`: the launch as
        before the option)."""
        if v2:
            finish_v2.delayed_finish_v2(
                mix,
                sqr,
                scale,
                base,
                rc,
                pre_in,
                nw,
                post,
                comb,
                next_pre,
                layer_input,
                rms_eps,
                hc_pre_eps,
                hc_post_mult_value,
                norm_eps,
                sinkhorn_repeat,
                has_pre_in=pre_mix is not None,
                has_norm=norm_weight is not None,
                **({} if part == PART_BOTH else {"part": part}),
            )
            return
        _delayed_finish_kernel[(T, 2 if part == PART_BOTH else 1)](
            mix,
            sqr,
            scale,
            base,
            rc,
            pre_in,
            nw,
            post,
            comb,
            next_pre,
            layer_input,
            T,
            float(rms_eps),
            float(hc_pre_eps),
            float(hc_post_mult_value),
            float(norm_eps),
            HIDDEN=hidden,
            KS=ksb,
            KSP2=triton.next_power_of_2(ksb),
            SINK=int(sinkhorn_repeat),
            HAS_PRE_IN=pre_mix is not None,
            HAS_NORM=norm_weight is not None,
            BH=_BH,
            ACCURATE=_ACCURATE_EXP,
            num_warps=4,
            **({} if part == PART_BOTH else {"PART": part}),
        )

    if sm80_mhc_side.defers(T, dev):
        # the sublayer reads only the layer input; the coefficients go beside
        # its all-reduce (ops/sm80_mhc_side.py)
        finish(PART_LAYER_INPUT)
        sm80_mhc_side.defer(lambda: finish(PART_COEFFICIENTS))
    else:
        finish(PART_BOTH)
    return streams, post, comb, layer_input, next_pre


def mhc_pre_delayed(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sm80_mhc.mhc_pre_delayed``: post [T, hc, 1], comb [T, hc, hc], the
    layer input [T, hidden] (BF16) and the next pre-mix [T, hc]."""
    _, post, comb, layer_input, next_pre = _delayed(
        None,
        residual,
        None,
        None,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix,
        norm_weight,
        norm_eps,
    )
    return post, comb, layer_input, next_pre


def mhc_fused_post_pre_delayed(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """``sm80_mhc.mhc_fused_post_pre_delayed``: the new streams, post, comb,
    the layer input, the next pre-mix and the streams' BF16 mean (empty
    unless ``capture_aux``)."""
    streams, post, comb, layer_input, next_pre = _delayed(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix,
        norm_weight,
        norm_eps,
    )
    aux = (
        streams.mean(dim=1)
        if capture_aux
        else streams.new_empty(0, streams.shape[-1])
    )
    return streams, post, comb, layer_input, next_pre, aux
