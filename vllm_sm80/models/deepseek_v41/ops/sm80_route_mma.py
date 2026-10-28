# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE routing after the gate logits as one Triton kernel (sm_80), for prefill rows (M >
sm80_router_mma.EXACT_ROWS; decode rows use sm80_route_gluon.py, same selection, weights from exactly recomputed logits).

Per token (one program, the row's experts in one block), from the FP32 gate logits ``l``: the selection scores
``s = sqrt(softplus(l))`` in FP32 as the served route computes them (``l`` above 20, else ``log1p(exp(l))``, correctly
rounded square root); ``b = s + bias`` (``bias_vl`` for tokens in ``[sentinel_lo, sentinel_hi)``); the ``K`` largest
``b``, largest first, equal values lowest expert first. ``w = s[ids] / (sum + eps) * scale`` is evaluated in FP64
(stable softplus ``max(l, 0) + log1p(exp(-|l|))``) from the gate's FP32 logits of the chosen experts (the incumbent's
prefill logits come from its FP32 SGEMM) and rounded once. A NaN biased score ranks above every number (as
torch.topk), NaNs lowest expert first, so ids stay in range.
"""
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _sqrt_softplus64(x):
    return libdevice.sqrt(tl.maximum(x, 0.0) + libdevice.log1p(libdevice.exp(-tl.abs(x))))


@triton.jit
def _sqrt_softplus32(x):
    return tl.sqrt_rn(tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x))))


@triton.jit
def _route_kernel(
    logits_ptr,
    logits_stride,
    bias_ptr,
    bias_vl_ptr,
    token_ptr,
    weights_ptr,
    ids_ptr,
    row_live_ptr,
    num_experts,
    sentinel_lo,
    sentinel_hi,
    scale,
    eps,
    K: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_E: tl.constexpr,
    ROW_MASK: tl.constexpr = False,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    valid = offs < num_experts
    lg = tl.load(logits_ptr + t * logits_stride + offs, mask=valid, other=0.0)
    token = tl.load(token_ptr + t)
    image = (token >= sentinel_lo) & (token < sentinel_hi)
    bias = tl.load(bias_ptr + offs, mask=valid, other=0.0)
    bias_vl = tl.load(bias_vl_ptr + offs, mask=valid, other=0.0)
    bias = tl.where(image, bias_vl, bias)
    biased = _sqrt_softplus32(lg) + bias
    # One int64 key per expert: the FP32 score's bits mapped to an order-preserving int32 (NaN above every number)
    # in the high half, BLOCK_E - 1 - expert in the low half, so one max per pick takes the largest score, lowest
    # expert first among equal scores and NaNs.
    bits = biased.to(tl.int32, bitcast=True)
    order = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    order = tl.where(biased != biased, 0x7FFFFFFF, order)
    key = (order.to(tl.int64) << 32) | (BLOCK_E - 1 - offs).to(tl.int64)
    key = tl.where(valid, key, -0x7FFFFFFFFFFFFFFF)

    lane = tl.arange(0, KP)
    ids = tl.zeros([KP], dtype=tl.int32)
    keep = lane < K
    for j in tl.static_range(K):
        top = tl.max(key, 0)
        pick = tl.minimum(BLOCK_E - 1 - (top & 0xFFFFFFFF).to(tl.int32), num_experts - 1)
        ids = tl.where(lane == j, pick, ids)
        key = tl.where(offs == pick, -0x7FFFFFFFFFFFFFFF, key)
    chosen = tl.load(logits_ptr + t * logits_stride + ids).to(tl.float64)
    scores = _sqrt_softplus64(chosen)
    total = tl.sum(tl.where(keep, scores, 0.0), 0)
    weights = scores / (total + eps) * scale
    if ROW_MASK:
        # The DSpark verification row mask: a masked row takes no routed expert.
        live = tl.load(row_live_ptr + t) != 0
        weights = tl.where(live, weights, 0.0)
        ids = tl.where(live, ids, -1)
    tl.store(weights_ptr + t * K + lane, weights.to(tl.float32), mask=keep)
    tl.store(ids_ptr + t * K + lane, ids, mask=keep)
