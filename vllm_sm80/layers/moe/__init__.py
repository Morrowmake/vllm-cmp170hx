# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's MoE layer classes.

  ``runner``          ``Sm80MoERunner`` (out of tree for ``MoERunner``): swaps
                      the children below in, wraps the entry for the idx-glue
                      "moesum" (``VLLM_AMPERE_DECODE_IDX_GLUE``)
  ``shared_experts``  ``Sm80SharedExperts``: re-ordered aux-stream overlap
                      (``VLLM_SHARED_EXPERT_REORDER``)
  ``router``          ``Sm80GroupedTopKRouter``: sm_80 fused decode routing
                      (``VLLM_GLM5_DECODE_MOE_ROUTING``, route v2 stash)
  ``gate``            ``Sm80GateLinear`` (out of tree for ``GateLinear``):
                      fused router v2 at the gate (``VLLM_MOE_DECODE_ROUTE_V2``)
  ``routed_experts``  ``Sm80RoutedExperts`` (out of tree for
                      ``RoutedExperts``): padding mask (``VLLM_MOE_MASK_PADDING``)

The kernels live in ``vllm_sm80.kernels.moe``. Every class is upstream's
behaviour while its switches are off.
"""
