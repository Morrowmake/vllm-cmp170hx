# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 DeepSeek-V4.1 mHC and MoE routing against a port of the
reference (``inference/model.py`` Block.hc_mixes / hc_pre / hc_post /
RMSNorm / Gate / Expert and ``kernel.py`` hc_split_sinkhorn), on CPU in FP32.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_sm80.models.deepseek_v41.ops import sm80_mhc
from vllm_sm80.models.deepseek_v41.sm80_moe import image_token_mask, route

HC, HIDDEN, ITERS, EPS, NORM_EPS = 4, 256, 20, 1e-6, 1e-20
MIX = (2 + HC) * HC


# --- reference port -------------------------------------------------------


def ref_sinkhorn(mixes, hc_scale, hc_base):
    """kernel.hc_split_sinkhorn_kernel, statement by statement per row."""
    pre = torch.sigmoid(mixes[:, :HC] * hc_scale[0] + hc_base[:HC]) + EPS
    post = 2 * torch.sigmoid(mixes[:, HC : 2 * HC] * hc_scale[1] + hc_base[HC : 2 * HC])
    comb = (mixes[:, 2 * HC :] * hc_scale[2] + hc_base[2 * HC :]).view(-1, HC, HC)
    row_max = comb.amax(dim=2, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + EPS
    comb = comb / (comb.sum(dim=1, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=1, keepdim=True) + EPS)
    return pre, post, comb


def ref_hc_mixes(x, fn, scale, base):
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)
    return ref_sinkhorn(F.linear(x, fn) * rsqrt, scale, base)


def ref_hc_pre(x, pre_mix):
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def ref_hc_post(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1
    )
    return y.type_as(x)


def ref_rms_norm(x, weight):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)
    return (weight * x).to(dtype)


def ref_gate(x, weight, bias, bias_vl, image_mask, top_k, route_scale):
    scores = F.linear(x.float(), weight.float())
    scores = F.softplus(scores).sqrt()
    if image_mask is not None:
        bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
    indices = (scores + bias).topk(top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    if top_k > 1:
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    weights *= route_scale
    return weights, indices


# --- fixtures -------------------------------------------------------------


def _mix_params(generator):
    fn = torch.randn(MIX, HC * HIDDEN, generator=generator) * 0.02
    scale = torch.rand(3, generator=generator) + 0.5
    base = torch.randn(MIX, generator=generator) * 0.1
    return fn, scale, base


def _mixes(params):
    return (*params, NORM_EPS, EPS, EPS, 2.0, ITERS)


# --- mHC ------------------------------------------------------------------


def test_split_sinkhorn_matches_the_reference_kernel():
    generator = torch.Generator().manual_seed(0)
    mixes = torch.randn(64, MIX, generator=generator) * 3
    fn, scale, base = _mix_params(generator)
    ours = sm80_mhc.hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
    for got, want in zip(ours, ref_sinkhorn(mixes, scale, base)):
        torch.testing.assert_close(got, want, rtol=2e-6, atol=1e-7)
    # The doubly stochastic comb: rows and columns sum to one.
    comb = ours[2]
    torch.testing.assert_close(comb.sum(-2), torch.ones(64, HC), rtol=1e-4, atol=1e-4)


def _layer(x, fn):
    """A deterministic stand-in sublayer: BF16 in, BF16 out."""
    return torch.tanh(x.float() @ fn).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 7])
def test_delayed_mhc_layers_match_the_reference_block(tokens):
    """Three layers of the delayed schedule the sm_80 decoder layer runs
    (first layer from the repeated embedding, fused post + pre after), then
    the final post and collapse, against the reference Block.forward."""
    generator = torch.Generator().manual_seed(1)
    embed = torch.randn(tokens, HIDDEN, generator=generator).to(torch.bfloat16)
    layers = []
    for _ in range(3):
        layers.append(
            {
                "attn": _mix_params(generator),
                "ffn": _mix_params(generator),
                "attn_norm": (torch.rand(HIDDEN, generator=generator) + 0.5).to(
                    torch.bfloat16
                ),
                "ffn_norm": (torch.rand(HIDDEN, generator=generator) + 0.5).to(
                    torch.bfloat16
                ),
                "w_attn": torch.randn(HIDDEN, HIDDEN, generator=generator) * 0.05,
                "w_ffn": torch.randn(HIDDEN, HIDDEN, generator=generator) * 0.05,
            }
        )

    # Reference: Transformer.forward over Block.forward.
    h = embed.unsqueeze(1).repeat(1, HC, 1)
    pre_mix = torch.zeros(tokens, HC)
    pre_mix[:, 0] = 1.0
    for p in layers:
        residual = h
        attn_pre, attn_post, attn_comb = ref_hc_mixes(h, *p["attn"])
        x = ref_rms_norm(ref_hc_pre(h, pre_mix), p["attn_norm"])
        x = _layer(x, p["w_attn"])
        h = ref_hc_post(x, residual, attn_post, attn_comb)
        residual = h
        ffn_pre, ffn_post, ffn_comb = ref_hc_mixes(h, *p["ffn"])
        x = ref_rms_norm(ref_hc_pre(h, attn_pre), p["ffn_norm"])
        x = _layer(x, p["w_ffn"])
        h = ref_hc_post(x, residual, ffn_post, ffn_comb)
        pre_mix = ffn_pre
    reference = ref_hc_pre(h, pre_mix)

    # sm_80: the decoder layer's delayed schedule.
    x, residual, post, comb, pre = embed, None, None, None, None
    for p in layers:
        if residual is None:
            residual = x.unsqueeze(1).expand(-1, HC, -1).contiguous()
            post, comb, x, attn_pre = sm80_mhc.mhc_pre_delayed(
                residual, *_mixes(p["attn"]), norm_weight=p["attn_norm"], norm_eps=NORM_EPS
            )
        else:
            residual, post, comb, x, attn_pre, _ = sm80_mhc.mhc_fused_post_pre_delayed(
                x,
                residual,
                post,
                comb,
                *_mixes(p["attn"]),
                pre_mix=pre,
                norm_weight=p["attn_norm"],
                norm_eps=NORM_EPS,
            )
        x = _layer(x, p["w_attn"])
        residual, post, comb, x, pre, _ = sm80_mhc.mhc_fused_post_pre_delayed(
            x,
            residual,
            post,
            comb,
            *_mixes(p["ffn"]),
            pre_mix=attn_pre,
            norm_weight=p["ffn_norm"],
            norm_eps=NORM_EPS,
        )
        x = _layer(x, p["w_ffn"])
    streams = sm80_mhc.mhc_post(x, residual, post, comb)
    torch.testing.assert_close(streams, h, rtol=0, atol=0)
    assert torch.equal(sm80_mhc.hc_collapse(streams, pre), reference)


# --- routing --------------------------------------------------------------

SENTINEL = 1000


@pytest.mark.parametrize(
    ("experts", "top_k", "with_images"), [(384, 6, True), (128, 3, True), (384, 6, False)]
)
def test_routing_matches_the_reference_gate(experts, top_k, with_images):
    generator = torch.Generator().manual_seed(2)
    tokens = 33
    x = torch.randn(tokens, HIDDEN, generator=generator).to(torch.bfloat16)
    # The checkpoint's gate weight and bias are FP16, bias_vl FP32.
    weight = (torch.randn(experts, HIDDEN, generator=generator) * 0.05).half()
    bias = (torch.randn(experts, generator=generator) * 0.1).half()
    bias_vl = torch.randn(experts, generator=generator) * 0.1
    input_ids = torch.randint(0, 2000, (tokens,), generator=generator)
    input_ids[::3] = SENTINEL + torch.arange(0, tokens, 3) % 5
    mask = image_token_mask(input_ids, SENTINEL if with_images else 0)
    # The adopted gate: FP32 weight, FP32 GEMM of the promoted input.
    logits = F.linear(x.float(), weight.float())
    got_w, got_ids = route(
        logits,
        bias.float(),
        top_k,
        bias_vl=bias_vl,
        image_mask=mask,
        routed_scaling_factor=1.5,
    )
    want_w, want_ids = ref_gate(
        x, weight, bias.float(), bias_vl, mask, top_k, route_scale=1.5
    )
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_w, want_w)
    if with_images:
        assert mask is not None and mask[::3].all() and not mask[1::3].any()


def test_swiglu_clamp_matches_the_reference_expert():
    """SiluAndMulWithClamp (the shared expert's activation) applies the
    reference clamp: up on both sides, the gate from above."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.config import VllmConfig, set_current_vllm_config


    generator = torch.Generator().manual_seed(3)
    gate_up = torch.randn(16, 2 * 64, generator=generator) * 20
    gate, up = gate_up.chunk(2, dim=-1)
    want = F.silu(torch.clamp(gate, max=10.0)) * torch.clamp(up, min=-10.0, max=10.0)
    with set_current_vllm_config(VllmConfig()):
        got = SiluAndMulWithClamp(10.0).forward_native(gate_up)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
