# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 MoE routing kernels called from the shared fused-MoE layers.

  ``routing``  routing + block alignment in one kernel, plus ``moe_sum``
               (``VLLM_GLM5_DECODE_MOE_ROUTING``)
  ``route``    router GEMV + top-k + block alignment, one launch for
               2 <= M <= 64 (``VLLM_MOE_DECODE_ROUTE_V2``)
  ``sum``      the routed sum fused with the shared-expert add (the "moesum"
               part of ``VLLM_AMPERE_DECODE_IDX_GLUE``)
  ``warmup``   compiles ``routing`` and ``route`` before CUDA graph capture

The gates and the routing -> alignment handoff below are evaluated on the host
at their call sites (grouped top-k router, MoE runner, Marlin experts and the
MoE block alignment). Both kernel families need ``VLLM_AMPERE_DECODE_KERNELS``
plus their own switch; with those unset the gates return False, no kernel
module is imported and every call site keeps the upstream code path exactly.

TOKEN BOUND. Measured us/call of ``routing`` (incumbent -> candidate) on GPU 2,
graph replay:

    M        moe_routing
    1    13.18 -> 5.92 2.23
    2    13.63 -> 7.20 1.89
    4    13.91 -> 8.86 1.57
    8    14.02 -> 12.17 1.15
   16    14.50 -> 13.66 1.06
   32    14.02 -> 15.14 0.93

The moe_router/fused_route_max_rows row of the sm80 tuning table
(:func:`fused_route_max_rows`, 8 without a row) bounds it at the top of that
win region; moe_router/route_v2_max_rows (:func:`route_v2_max_rows`) bounds
``route``.
"""

import torch

from vllm_sm80 import tuning
from vllm_sm80.kernels._platform import _is_sm80

__all__ = [
    "use_ampere_moe_routing",
    "fused_route_max_rows",
    "marlin_block_size_m",
    "stash_fused_align",
    "take_fused_align",
    "drop_fused_align",
    "fused_align_is_masked",
    "route_v2_masks_padding",
    "route_v2_max_rows",
    "use_ampere_moe_route_v2",
    "maybe_moe_route_v2",
    "take_fused_routing",
]

# The router shape route.py is built and validated for
# (use_ampere_moe_route_v2), and the most rows it has a path for.
_ROUTE_V2_SHAPE = {"n_experts": 288, "top_k": 8, "hidden": 4096}
_ROUTE_V2_ROW_LIMIT = 64


def fused_route_max_rows(top_k: int) -> int:
    """Largest batch of the fused decode routing, which is also the largest
    batch the MoE padding mask leaves unmasked: the
    moe_router/fused_route_max_rows row of the sm80 tuning table, else 8."""
    return tuning.lookup(
        "moe_router", "fused_route_max_rows", {"top_k": int(top_k)}, 8
    )


def route_v2_max_rows() -> int:
    """Largest batch of the fused router v2: the moe_router/route_v2_max_rows
    row of the sm80 tuning table, else 32 (the largest batch it was validated
    and timed at). Raises above 64, the most rows route.py handles."""
    rows = tuning.lookup("moe_router", "route_v2_max_rows", _ROUTE_V2_SHAPE, 32)
    if rows > _ROUTE_V2_ROW_LIMIT:
        raise ValueError(
            f"moe_router/route_v2_max_rows = {rows} exceeds "
            f"{_ROUTE_V2_ROW_LIMIT}, the most rows the sm_80 fused MoE router "
            "handles"
        )
    return rows


def use_ampere_moe_routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    scoring_func: str = "sigmoid",
    num_expert_group: int = 1,
    topk_group: int = 1,
    renormalize: bool = True,
) -> bool:
    """Gate for vllm_sm80/kernels/moe/routing.py.

    Covers ``fused_route_align`` (routing + block alignment) and ``moe_sum``.
    The keyword arguments are the routing-method assumptions the kernel bakes
    in; their defaults are the GLM-5.3-Flash values, so the three-positional
    call form used by the artifact-link script means "the production router".
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MOE_ROUTING:
        return False
    if num_tokens < 1 or num_tokens > fused_route_max_rows(topk):
        return False
    # The fused kernel is a single-CTA sigmoid top-k with no group phase.
    if scoring_func != "sigmoid":
        return False
    if num_expert_group != 1 or topk_group != 1:
        return False
    if not renormalize:
        return False
    if num_experts != 288:
        return False
    # `tl.topk` inside the kernel needs a power-of-two k that fits the expert
    # tile split (E=288 -> 256 + 32).
    if topk < 4 or topk > 32 or (topk & (topk - 1)):
        return False
    return _is_sm80()


def marlin_block_size_m(
    num_tokens: int,
    topk: int,
    num_experts: int,
    input_dtype: torch.dtype | None = None,
) -> int:
    """``fused_marlin_moe``'s host-side ``block_size_m`` choice, verbatim.

    The fused routing kernel produces the block alignment too, but the routing
    runs in ``GroupedTopKRouter`` and the alignment is requested much later,
    inside ``fused_marlin_moe``, which is where the block size is picked. This
    reproduces that pick so the alignment can be computed early. If the
    prediction is ever wrong the handoff below simply misses and upstream's
    ``moe_align_block_size`` runs as before -- correct, just not fused.
    """
    block_size_m = 8
    for block_size_m in (8, 16, 32, 48, 64):
        if num_tokens * topk / num_experts / block_size_m < 0.9:
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)
    return block_size_m


# --- routing -> alignment handoff -------------------------------------------
# `fused_route_align` replaces `fused_grouped_topk` AND `moe_align_block_size`,
# but those are called from two places a long way apart (the router, then the
# Marlin experts' apply). One slot is enough: the 42 MoE layers run strictly
# one after another, routing then alignment, and the slot holds a reference to
# the `topk_ids` it belongs to, so identity comparison can never alias a freed
# tensor.
_PENDING: tuple | None = None


def stash_fused_align(
    topk_ids, block_size, num_experts, aligned, is_padding=None
) -> None:
    """Record the alignment the fused routing kernel already computed.

    `is_padding` is the padding mask the kernel already applied to these ids
    (route v2 with VLLM_GLM5_MOE_ROUTE_V2_MASK), or None."""
    global _PENDING
    _PENDING = (topk_ids, int(block_size), int(num_experts), aligned, is_padding)


def take_fused_align(
    topk_ids,
    block_size: int,
    num_experts: int,
    expert_map,
    pad_sorted_ids: bool,
):
    """Return the stashed alignment for exactly this request, or None."""
    global _PENDING
    pending = _PENDING
    if pending is None:
        return None
    # `ignore_invalid_experts` deliberately has no guard: upstream passes
    # `expert_map if ignore_invalid_experts else None` down to the kernel and
    # applies the post-hoc gather only `if expert_map is not None and not
    # ignore_invalid_experts`, so once expert_map is None -- which the line
    # below requires -- the flag changes nothing on either branch. Marlin's
    # ignore_invalid_experts=True is therefore covered by the expert_map test.
    if expert_map is not None or pad_sorted_ids:
        return None
    ids, bs, ne, aligned, _ = pending
    if ids is not topk_ids or bs != int(block_size) or ne != int(num_experts):
        return None
    _PENDING = None
    return aligned


def drop_fused_align(topk_ids) -> bool:
    """Forget the stashed alignment for exactly this `topk_ids`, if any.

    The alignment was computed from the ids as routed. A caller that edits
    them in place afterwards (the padding mask sets padding rows to -1 and
    keeps the tensor's identity) must drop it, or the handoff would return an
    alignment of ids that no longer exist and the experts would run on the
    unmasked layout. The next alignment request then misses and computes a
    fresh one from the edited ids. Host-side only; returns whether a stash
    was dropped."""
    global _PENDING
    pending = _PENDING
    if pending is None or pending[0] is not topk_ids:
        return False
    _PENDING = None
    return True


def fused_align_is_masked(topk_ids, is_padding) -> bool:
    """Whether the stashed alignment for exactly this `topk_ids` was computed
    by a kernel that already routed the rows of exactly this `is_padding`
    mask to no expert (ids -1, alignment of the masked ids). The padding mask
    then has nothing left to do: the ids and the alignment are already what
    it would produce. Host-side identity checks only."""
    pending = _PENDING
    return (
        pending is not None
        and pending[0] is topk_ids
        and pending[4] is not None
        and pending[4] is is_padding
    )


# --- router GEMV + routing + alignment (sm80/kernels/moe/route.py) ----------
# `moe_route` computes the router logits, the top-k and the Marlin alignment in
# one launch at the gate's call site (moe_runner.py). The router and the Marlin
# experts run later and far away, so the results are handed over through two
# single-slot stashes keyed by tensor IDENTITY: the routing by the
# `router_logits` tensor returned in place of the gate's, the alignment by the
# `topk_ids` (the existing `stash_fused_align`). A consumer that sees a
# different tensor misses and computes its own result from the logits, which
# are the same logits either way.
_PENDING_ROUTING: tuple | None = None
_V2_ANNOUNCED = False


def use_ampere_moe_route_v2(num_tokens: int, gate, router, x) -> bool:
    """Gate for vllm_sm80/kernels/moe/route.py, evaluated on the host.

    Pinned to what the kernel was validated for: GLM-5.x noaux_tc sigmoid
    routing, one expert group, renormalised, fp32 correction bias, 288 experts,
    top-8, hidden 4096, bf16 gate weight with fp32 logits, no gate bias.
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS or not envs.VLLM_MOE_DECODE_ROUTE_V2:
        return False
    if num_tokens < 1 or num_tokens > route_v2_max_rows():
        return False
    if gate is None or router is None:
        return False
    if getattr(router, "scoring_func", None) != "sigmoid":
        return False
    if getattr(router, "num_expert_group", None) != 1:
        return False
    if getattr(router, "topk_group", None) != 1:
        return False
    if not getattr(router, "renormalize", False):
        return False
    if getattr(router, "top_k", None) != 8:
        return False
    if getattr(router, "num_fused_shared_experts", 0) != 0:
        return False
    if getattr(router, "skip_padding", False):
        return False
    if getattr(router, "eplb_state", None) is not None:
        return False
    bias = getattr(router, "e_score_correction_bias", None)
    if bias is None or bias.dtype != torch.float32 or tuple(bias.shape) != (288,):
        return False
    w = getattr(gate, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != (288, 4096):
        return False
    if not w.is_contiguous():
        return False
    if getattr(gate, "bias", None) is not None:
        return False
    if getattr(gate, "out_dtype", None) != torch.float32:
        return False
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != 4096:
        return False
    if x.stride(1) != 1:
        return False
    return _is_sm80()


def route_v2_masks_padding(num_tokens: int) -> bool:
    """Whether route v2 applies the padding mask inside its launch at this
    batch size: VLLM_MOE_MASK_PADDING=1 and VLLM_GLM5_MOE_ROUTE_V2_MASK=1
    (default), a batch the mask covers (above fused_route_max_rows rows at
    route v2's top-k) and the one-launch kernel's range.
    Outside it the mask edits the ids afterwards and drops the stash
    (`drop_fused_align`), as before. A static function of the shape, so eager
    runs, warmup and captured graphs agree."""
    from vllm import envs

    if not envs.VLLM_MOE_MASK_PADDING or not envs.VLLM_GLM5_MOE_ROUTE_V2_MASK:
        return False
    min_rows = fused_route_max_rows(_ROUTE_V2_SHAPE["top_k"])
    if num_tokens <= min_rows:
        return False
    from vllm_sm80.kernels.moe.route import pad_supported

    return pad_supported(num_tokens)


_V2_MASK_ANNOUNCED = False


def _route_v2_padding(num_tokens: int):
    """The forward context's padding mask for a route v2 call of this size
    when the mask is applied inside the kernel, else None. The same
    conditions as `mask_padding_topk_ids`, which then finds the ids masked."""
    if not route_v2_masks_padding(num_tokens):
        return None
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None or tuple(is_padding.shape) != (num_tokens,):
        return None
    if not is_padding.is_contiguous():
        return None
    global _V2_MASK_ANNOUNCED
    if not _V2_MASK_ANNOUNCED:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "sm_80 fused MoE router masks padding rows in-kernel: padded "
            "batches keep the fused block alignment "
            "[VLLM_GLM5_MOE_ROUTE_V2_MASK=1; set 0 for the re-align path]"
        )
        _V2_MASK_ANNOUNCED = True
    return is_padding


def maybe_moe_route_v2(gate, router, x):
    """Run the fused router for `x` if the gate allows it.

    Returns the fp32 router logits (in place of `gate(x)[0]`) and stashes the
    top-k and the alignment for the router and the Marlin experts, or returns
    None, in which case the caller runs the gate exactly as before.
    """
    num_tokens = x.shape[0]
    if not use_ampere_moe_route_v2(num_tokens, gate, router, x):
        return None
    from vllm_sm80.kernels.moe.route import moe_route

    num_experts = int(gate.weight.shape[0])
    topk = int(router.top_k)
    block_size = marlin_block_size_m(num_tokens, topk, num_experts)
    is_padding = _route_v2_padding(num_tokens)
    logits, topk_w, topk_ids, sorted_ids, expert_ids, ntpp = moe_route(
        x,
        gate.weight,
        router.e_score_correction_bias.data,
        topk=topk,
        block_size=block_size,
        num_experts=num_experts,
        renormalize=True,
        routed_scaling_factor=float(router.routed_scaling_factor),
        is_padding=is_padding,
    )
    global _PENDING_ROUTING, _V2_ANNOUNCED
    _PENDING_ROUTING = (logits, (topk_w, topk_ids))
    stash_fused_align(topk_ids, block_size, num_experts,
                      (sorted_ids, expert_ids, ntpp), is_padding=is_padding)
    if not _V2_ANNOUNCED:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "sm_80 fused MoE router active: gate GEMV + top-k + block "
            "alignment in one op for M <= %d "
            "[VLLM_MOE_DECODE_ROUTE_V2=1]",
            route_v2_max_rows(),
        )
        _V2_ANNOUNCED = True
    return logits


def take_fused_routing(router_logits):
    """(topk_weights, topk_ids) computed with exactly these logits, or None."""
    global _PENDING_ROUTING
    pending = _PENDING_ROUTING
    if pending is None:
        return None
    logits, routing = pending
    if logits is not router_logits:
        return None
    _PENDING_ROUTING = None
    return routing
