# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE routing after the gate GEMM as one Triton kernel (sm_80).

:func:`route_fused` returns what ``sm80_moe.route`` returns for FP32 gate
logits, with ``gate_temp`` 1, in one launch instead of fifteen (image-token
mask, softplus, sqrt, bias select, bias add, top-k and its sort, gather,
sum, ``+ 1e-20``, divide, scale and the int32 cast). Every value goes
through the same FP32 operations in the same order as the torch path:

- ``softplus(x)``: ``x`` above 20, else ``log1p(exp(x))``, with libdevice's
  ``exp`` and ``log1p`` (the functions torch's CUDA kernel calls) compiled
  without flush-to-zero, as torch is;
- the square root and the divide correctly rounded (``sqrt_rn``,
  ``div_rn``);
- per token, ``bias_vl`` for ids in ``[sentinel_lo, sentinel_hi)`` (the
  image span) else ``bias``, added to the score;
- the ``k`` largest biased scores, largest first; equal biased scores are
  taken lowest expert first;
- the chosen experts' unbiased scores summed in the order of torch's
  reduction kernel for a row of ``k`` contiguous FP32 values: with ``W`` the
  largest power of two not above ``k``, lane ``x`` holds
  ``w[x] + w[x + W]`` (``w[x]`` alone past ``k``) and the lanes fold in
  halves, so ``k == 6`` sums ``((w0 + w4) + w2) + ((w1 + w5) + w3)`` and
  ``k == 3`` sums ``(w0 + w2) + w1``; then ``+ 1e-20``, each weight divided
  by that sum, and every weight multiplied by the routed scaling factor.

The outputs are therefore the torch path's bytes for finite logits, except
where torch's own result is unspecified: the order of equal biased scores
within the top-k (torch's sort is not stable) and which of several equal
scores at the k-th place is kept. A NaN biased score ranks above every number,
as in torch.topk (NaNs lowest expert first), so a NaN row keeps in-range
expert ids and NaN weights instead of selecting past the experts. One program per token; the experts of a
row fit one block.

With ``row_live`` (the DSpark verification row mask, int8 per row, read in
place) a row whose entry is 0 gets expert ids -1 and weights +0.0 (no routed
expert); every other row's outputs are unchanged. Without it the kernel
never reads the mask.
"""

import os

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton

__all__ = ["route_fused", "supports"]

_MAX_EXPERTS = 1024
_MAX_TOP_K = 16
# The Triton interpreter (CPU tests) has no libdevice: it takes tl.exp and
# tl.log, and launches without the CUDA compile options.
_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


def supports(num_experts: int, top_k: int, gate_temp: float) -> str | None:
    """None when :func:`route_fused` serves this router, else the reason."""
    if gate_temp != 1.0:
        return f"gate_temp {gate_temp} is not 1 (torch scales by its reciprocal)"
    if not 0 < num_experts <= _MAX_EXPERTS:
        return f"{num_experts} experts (1 to {_MAX_EXPERTS} served)"
    if not 0 < top_k <= min(num_experts, _MAX_TOP_K):
        return f"top_k {top_k} (1 to {min(num_experts, _MAX_TOP_K)} served)"
    return None


@triton.jit
def _sqrt_softplus(x, ACCURATE: tl.constexpr):
    if ACCURATE:
        sp = libdevice.log1p(libdevice.exp(x))
    else:
        sp = tl.log(1.0 + tl.exp(x))
    return tl.sqrt_rn(tl.where(x > 20.0, x, sp))


@triton.jit
def _torch_row_sum(v, W: tl.constexpr, LOG2W: tl.constexpr):
    """Sum of ``v`` ([2 W], zeros past k) in torch's reduction order for a
    contiguous row of k values (W <= k < 2 W): lane x = v[x] + v[x + W], then
    lane x += lane x + h for h = W / 2, ..., 1."""
    lanes = tl.sum(tl.reshape(v, (2, W)), 0)
    for level in tl.static_range(LOG2W):
        lanes = tl.sum(tl.reshape(lanes, (2, W >> (level + 1))), 0)
    return tl.sum(lanes, 0)


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
    W: tl.constexpr,
    LOG2W: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_VL: tl.constexpr,
    RENORM: tl.constexpr,
    ACCURATE: tl.constexpr,
    ROW_MASK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    valid = offs < num_experts
    x = tl.load(logits_ptr + t * logits_stride + offs, mask=valid, other=0.0)
    scores = _sqrt_softplus(x, ACCURATE)
    bias = tl.load(bias_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    if HAS_VL:
        token = tl.load(token_ptr + t)
        image = (token >= sentinel_lo) & (token < sentinel_hi)
        bias_vl = tl.load(bias_vl_ptr + offs, mask=valid, other=0.0).to(tl.float32)
        bias = tl.where(image, bias_vl, bias)
    biased = tl.where(valid, scores + bias, float("-inf"))

    lane = tl.arange(0, 2 * W)
    weights = tl.zeros([2 * W], dtype=tl.float32)
    ids = tl.zeros([2 * W], dtype=tl.int32)
    for j in tl.static_range(K):
        # torch.topk ranks NaN above every number; tl.max skips NaN, so a NaN
        # is taken first here (lowest expert first), keeping the ids in range
        # and the row's weights NaN as torch's are. Finite rows take the
        # largest value, lowest expert first, as before.
        is_nan = biased != biased
        first_nan = tl.min(tl.where(is_nan, offs, BLOCK_E), 0)
        top = tl.max(tl.where(is_nan, float("-inf"), biased), 0)
        pick = tl.min(tl.where(biased == top, offs, BLOCK_E), 0)
        pick = tl.where(first_nan < BLOCK_E, first_nan, pick)
        pick = tl.minimum(pick, num_experts - 1)
        chosen = offs == pick
        weights = tl.where(lane == j, tl.sum(tl.where(chosen, scores, 0.0), 0), weights)
        ids = tl.where(lane == j, pick, ids)
        biased = tl.where(chosen, float("-inf"), biased)
    if RENORM:
        total = _torch_row_sum(weights, W, LOG2W) + eps
        weights = tl.math.div_rn(weights, total)
    weights = weights * scale
    if ROW_MASK:
        # A masked row takes no routed expert.
        live = tl.load(row_live_ptr + t) != 0
        weights = tl.where(live, weights, 0.0)
        ids = tl.where(live, ids, -1)
    keep = lane < K
    tl.store(weights_ptr + t * K + lane, weights, mask=keep)
    tl.store(ids_ptr + t * K + lane, ids, mask=keep)


def route_fused(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    *,
    bias_vl: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    sentinel_lo: int = 0,
    sentinel_hi: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    row_live: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``sm80_moe.route`` with ``gate_temp`` 1 and the image mask of
    ``input_ids`` (``sm80_moe.image_token_mask``: ids in ``[sentinel_lo,
    sentinel_hi)``, none when ``sentinel_lo`` is 0) for FP32 ``logits``
    [tokens, experts]: FP32 weights and int32 expert ids [tokens, top_k].
    With ``row_live`` (int8, at least ``tokens`` entries) rows whose entry is
    0 get ids -1 and weights 0. No host synchronisation (graph-safe; the
    mask is read when the kernel runs)."""
    assert logits.dtype == torch.float32 and logits.dim() == 2, (logits.dtype, logits.shape)
    assert logits.stride(1) == 1, logits.stride()
    T, E = logits.shape
    reason = supports(E, top_k, 1.0)
    assert reason is None, reason
    assert bias.shape == (E,) and bias.is_contiguous(), bias.shape
    weights = torch.empty(T, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(T, top_k, dtype=torch.int32, device=logits.device)
    if T == 0:
        return weights, ids
    has_vl = input_ids is not None and sentinel_lo > 0 and bias_vl is not None
    if has_vl:
        assert input_ids.shape == (T,) and input_ids.is_contiguous(), input_ids.shape
        assert bias_vl.shape == (E,) and bias_vl.is_contiguous(), bias_vl.shape
    if row_live is not None:
        assert (
            row_live.dtype == torch.int8 and row_live.dim() == 1
            and row_live.is_contiguous() and row_live.numel() >= T
            and row_live.device == logits.device
        ), (row_live.dtype, row_live.shape, row_live.device)
    w = 1 << (top_k.bit_length() - 1)
    options = {} if _INTERPRET else {"enable_reflect_ftz": False, "enable_fp_fusion": False}
    _route_kernel[(T,)](
        logits,
        logits.stride(0),
        bias,
        bias_vl if has_vl else bias,
        input_ids if has_vl else ids,
        weights,
        ids,
        ids if row_live is None else row_live,
        E,
        int(sentinel_lo),
        int(sentinel_hi),
        float(routed_scaling_factor),
        1e-20,
        K=top_k,
        W=w,
        LOG2W=w.bit_length() - 1,
        BLOCK_E=triton.next_power_of_2(E),
        HAS_VL=has_vl,
        RENORM=renormalize and top_k > 1,
        ACCURATE=not _INTERPRET,
        ROW_MASK=row_live is not None,
        num_warps=4,
        **options,
    )
    return weights, ids
