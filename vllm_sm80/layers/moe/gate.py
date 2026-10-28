# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 fused decode router (``VLLM_MOE_DECODE_ROUTE_V2``) at the gate.

Registered out of tree for ``GateLinear`` (``vllm_sm80.plugin``) while
VLLM_MOE_DECODE_ROUTE_V2 is set, so every
``GateLinear`` is built as :class:`Sm80GateLinear`. Only a gate that a
``MoERunner`` holds and calls itself (``MoERunner._forward_impl``, the
unfused-gate branch) is given its router (:meth:`attach_router`, called by
``vllm_sm80.layers.moe.runner.Sm80MoERunner`` with the switch on). Such a gate
runs gate GEMV + top-k + block alignment in one op when
``vllm_sm80.kernels.moe.maybe_moe_route_v2`` allows it, returning the fp32
logits in place of its GEMM output; the router and the Marlin experts pick up
the stashed top-k and alignment. Every other call, and every gate without a
router, is upstream's ``GateLinear.forward``.
"""

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


class Sm80GateLinear(GateLinear):
    """``GateLinear`` that can run the sm_80 fused decode router."""

    # Set per instance by `attach_router`; never a submodule (routers are
    # not modules), so parameter names and the module tree are unchanged.
    _route_v2_router = None

    def attach_router(self, router) -> None:
        self._route_v2_router = router

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        router = self._route_v2_router
        if router is not None:
            from vllm_sm80.kernels.moe import maybe_moe_route_v2

            router_logits = maybe_moe_route_v2(self, router, x)
            if router_logits is not None:
                return self._return(router_logits)
        return super().forward(x)
