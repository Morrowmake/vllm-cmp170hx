# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 delayed mHC on sm_80, in the reference's own FP32 formulas.

The upstream DSv4.1 path runs these steps as TileLang kernels
(``mhc_pre_delayed_tilelang``, ``mhc_fused_post_pre_delayed_tilelang``,
``mhc_post_tilelang``) with DeepGEMM TF32 or overlap variants on SM100; the
fork's sm_80 mHC kernels (vllm_sm80/kernels/mhc) cover only the non-delayed
GLM ops. These are the reference's operations (``Block.hc_mixes``,
``hc_pre``, ``hc_post``, ``RMSNorm`` and ``kernel.hc_split_sinkhorn``),
written with the same torch ops in the same order, so they run on any device
in FP32 and carry the reference's BF16 rounding points:

- coefficients from the BF16 stream promoted to FP32: ``x @ fn.T`` times
  ``rsqrt(mean(x^2) + rms_eps)``, pre ``sigmoid(a0 m + b) + eps``, post
  ``mult * sigmoid(a1 m + b)``, comb ``softmax(a2 m + b) + eps`` then a
  column normalisation and ``iters - 1`` row/column Sinkhorn passes;
- collapse ``sum(pre * x.float())`` rounded to BF16, then RMSNorm (FP32
  statistic, ``weight * x`` rounded to BF16);
- post ``post * x + sum_j comb[j, k] * residual_j`` rounded to BF16.

The first layer's stream is the embedding repeated ``hc_mult`` times and its
coefficients come from that whole stream, as in the reference (upstream
projects the single embedding with the summed weights instead, which differs
in rounding).
"""

import torch
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    post_mult: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``kernel.hc_split_sinkhorn``: FP32 mixes [..., (2 + hc) * hc] ->
    pre [..., hc], post [..., hc], comb [..., hc, hc]."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = post_mult * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = mixes[..., 2 * hc :].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[
        2 * hc :
    ].view(hc, hc)
    # The kernel's softmax: exp of the max-shifted row over its sum, + eps.
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Block.hc_mixes`` of the BF16 streams [tokens, hc, hidden]."""
    x = residual.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn) * rsqrt
    return hc_split_sinkhorn(
        mixes, hc_scale, hc_base, residual.shape[-2], sinkhorn_iters, hc_eps, post_mult
    )


def identity_pre_mix(residual: torch.Tensor) -> torch.Tensor:
    """``make_identity_pre_mix``: the one-hot pre-mix that selects stream 0."""
    pre_mix = residual.new_zeros(residual.shape[:-1], dtype=torch.float32)
    pre_mix[..., 0] = 1.0
    return pre_mix


def hc_collapse(residual: torch.Tensor, pre_mix: torch.Tensor | None) -> torch.Tensor:
    """``Block.hc_pre``: the streams weighted by ``pre_mix`` and summed, in
    BF16; None is the identity pre-mix, applied the same way."""
    if pre_mix is None:
        pre_mix = identity_pre_mix(residual)
    return torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).to(
        residual.dtype
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference ``RMSNorm``: FP32 statistic, ``weight * x`` in BF16."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """``Block.hc_post``: sublayer output [tokens, hidden], streams
    [tokens, hc, hidden], post [tokens, hc(, 1)], comb [tokens, hc, hc] ->
    the next streams in BF16."""
    post = post_layer_mix.reshape(residual.shape[:-1])
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
    )
    return y.type_as(x)


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
    """``mhc_pre_delayed_tilelang``'s contract: the sublayer input collapsed
    with the carried ``pre_mix`` (None: stream zero) and normalised, and
    this stream's coefficients. Returns post [tokens, hc, 1], comb
    [tokens, hc, hc], layer input [tokens, hidden] (BF16) and the next
    pre-mix [tokens, hc]."""
    if hc_pre_eps != hc_sinkhorn_eps:
        raise ValueError("The DeepSeek-V4.1 reference uses one hc_eps")
    pre, post, comb = hc_mixes(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        sinkhorn_repeat,
        hc_post_mult_value,
    )
    layer_input = hc_collapse(residual, pre_mix)
    if norm_weight is not None:
        layer_input = rms_norm(layer_input, norm_weight, norm_eps)
    return post.unsqueeze(-1), comb, layer_input, pre


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
    """``mhc_fused_post_pre_delayed_tilelang``'s contract: the previous
    sublayer's post, then the delayed pre of the new streams. Returns the
    streams, post, comb, layer input, next pre-mix and the streams' BF16
    mean (empty unless ``capture_aux``)."""
    residual = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    aux = (
        residual.mean(dim=1)
        if capture_aux
        else residual.new_empty(0, residual.shape[-1])
    )
    outputs = mhc_pre_delayed(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return residual, *outputs, aux
