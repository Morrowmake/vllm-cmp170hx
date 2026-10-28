# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 fused decode routing behind upstream's grouped top-k router.

:class:`Sm80GroupedTopKRouter` adds two paths in front of upstream's
``GroupedTopKRouter._compute_routing``:

- ``VLLM_MOE_DECODE_ROUTE_V2``: the fused router already routed these exact
  logits at the gate's call site (``vllm_sm80.kernels.moe.maybe_moe_route_v2``
  via ``vllm_sm80.layers.moe.gate``); its top-k is taken from the stash.
- ``VLLM_GLM5_DECODE_MOE_ROUTING``: one fused kernel for the routing and the
  block alignment that ``fused_marlin_moe`` asks for later
  (``vllm/model_executor/layers/fused_moe/moe_align_block_size.py``: the two
  call sites are far apart, so the alignment is handed over through the
  single-slot stash keyed by the topk_ids identity).

Every other call is upstream's. ``vllm_sm80.layers.moe.runner.Sm80MoERunner``
swaps every router of exactly this upstream class to it.
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)


class Sm80GroupedTopKRouter(GroupedTopKRouter):
    """``GroupedTopKRouter`` with the sm_80 fused decode routing paths."""

    @classmethod
    def adopt(cls, router: GroupedTopKRouter) -> "Sm80GroupedTopKRouter":
        """Turn an upstream ``GroupedTopKRouter`` into this class in place."""
        assert type(router) is GroupedTopKRouter, type(router)
        router.__class__ = cls
        return router

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_MOE_DECODE_ROUTE_V2 and indices_type in (None, torch.int32):
            from vllm_sm80.kernels.moe import take_fused_routing

            pre = take_fused_routing(router_logits)
            if pre is not None:
                return pre

        fused = self._maybe_fused_route_align(router_logits, indices_type)
        if fused is not None:
            return fused
        return super()._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )

    def _valid_grouping(self, router_logits: torch.Tensor) -> bool:
        # The same test as upstream's `_compute_routing`: an invalid grouping
        # takes upstream's fused_topk(_bias) branch before the grouped path.
        num_experts = router_logits.shape[-1]
        if num_experts <= self.num_expert_group:
            return False
        return num_experts % self.num_expert_group == 0

    def _maybe_fused_route_align(
        self, router_logits: torch.Tensor, indices_type: torch.dtype | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Route and align in one kernel when its gate allows, else None."""
        if not self._valid_grouping(router_logits):
            return None
        from vllm_sm80.kernels.moe import (
            marlin_block_size_m,
            stash_fused_align,
            use_ampere_moe_routing,
        )

        bias = self.e_score_correction_bias
        if not (
            bias is not None
            and not self.skip_padding
            and self.eplb_state is None
            and self.num_fused_shared_experts == 0
            and indices_type in (None, torch.int32)
            and router_logits.dim() == 2
            and router_logits.dtype == torch.float32
            and bias.dtype == torch.float32
            and use_ampere_moe_routing(
                router_logits.shape[0],
                router_logits.shape[-1],
                self.top_k,
                scoring_func=self.scoring_func,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                renormalize=self.renormalize,
            )
        ):
            return None
        from vllm_sm80.kernels.moe.routing import fused_route_align

        num_experts = router_logits.shape[-1]
        block_size = marlin_block_size_m(
            router_logits.shape[0], self.top_k, num_experts
        )
        (
            topk_weights,
            topk_ids,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        ) = fused_route_align(
            router_logits,
            bias.data,
            topk=self.top_k,
            block_size=block_size,
            num_experts=num_experts,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        stash_fused_align(
            topk_ids,
            block_size,
            num_experts,
            (sorted_ids, expert_ids, num_tokens_post_pad),
        )
        return topk_weights, topk_ids
