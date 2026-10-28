# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE padding mask (``VLLM_MOE_MASK_PADDING``).

Registered out of tree for ``RoutedExperts`` (``vllm_sm80.plugin``) while
VLLM_MOE_MASK_PADDING is set, so every
``RoutedExperts`` is built as :class:`Sm80RoutedExperts`. Its
``forward_modular`` - called by ``MoERunner._apply_quant_method`` right after
``router.select_experts`` - masks the top-k ids of the padding rows
(:func:`mask_padding_topk_ids`) and then runs upstream's ``forward_modular``.
With the switch off the ids pass through untouched.
"""

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

_MASK_PADDING: bool | None = None


def mask_padding_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    """With VLLM_MOE_MASK_PADDING=1, set the expert ids of padding rows
    (forward context is_padding) to -1, in place, so the alignment drops them;
    the real rows then get the same expert blocks whatever the padding rows
    hold. In place keeps the tensor's identity, which the sm_80 fused decode
    routing uses to hand over its own alignment. Batches of at most
    fused_route_max_rows rows at this top-k (the fused router's range, the
    moe_router/fused_route_max_rows row of the sm80 tuning table) are skipped
    whether or not VLLM_AMPERE_DECODE_KERNELS is on; a row of 0 masks every
    batch. A larger batch that the fused router v2 covered (up to
    32 rows) had its alignment computed before the mask; that handoff is
    dropped here so the alignment is recomputed from the masked ids, unless
    the router already masked exactly these rows inside its launch
    (VLLM_GLM5_MOE_ROUTE_V2_MASK), in which case the ids and its alignment
    are already the masked ones and nothing is done. Returns topk_ids."""
    global _MASK_PADDING
    if _MASK_PADDING is None:
        import vllm.envs as envs

        _MASK_PADDING = bool(envs.VLLM_MOE_MASK_PADDING)
        if _MASK_PADDING:
            logger.info_once(
                "MoE padding mask active: padding rows are routed to no "
                "expert (VLLM_MOE_MASK_PADDING=1; set 0 to disable)"
            )
    if not _MASK_PADDING:
        return topk_ids
    from vllm_sm80.kernels.moe import fused_route_max_rows

    if topk_ids.shape[0] <= fused_route_max_rows(topk_ids.shape[1]):
        return topk_ids
    if not is_forward_context_available():
        return topk_ids
    is_padding = get_forward_context().is_padding
    # Only a call that sees the whole (padded) batch can be masked row for
    # row: a caller that runs the MoE on row slices of an unpadded batch must
    # not take the first rows of the batch's mask.
    if is_padding is None or is_padding.shape[0] != topk_ids.shape[0]:
        return topk_ids
    from vllm_sm80.kernels.moe import drop_fused_align, fused_align_is_masked

    if fused_align_is_masked(topk_ids, is_padding):
        # Route v2 routed these padding rows to no expert in its own launch
        # and aligned the masked ids: the fill below would rewrite the same
        # -1s and the handoff is valid.
        return topk_ids
    topk_ids.masked_fill_(is_padding.unsqueeze(1), -1)
    # The sm_80 fused decode router (VLLM_MOE_DECODE_ROUTE_V2, up to 32
    # rows) stashed a block alignment of these ids before the mask; drop it
    # so the alignment is recomputed from the masked ids. Which rows are
    # padding is only known on the device, so this is decided by the call
    # shape alone and is the same in eager mode and in a captured graph.
    if drop_fused_align(topk_ids):
        logger.info_once(
            "MoE padding mask: the fused router's block alignment is "
            "recomputed from the masked ids (VLLM_MOE_MASK_PADDING=1)"
        )
    return topk_ids


class Sm80RoutedExperts(RoutedExperts):
    """``RoutedExperts`` that masks the padding rows' top-k ids."""

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ):
        topk_ids = mask_padding_topk_ids(topk_ids)
        return super().forward_modular(
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
