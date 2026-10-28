# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 paths of ``MarlinExperts`` (``fused_moe/experts/marlin_moe.py``).

``MarlinExperts.apply`` calls :func:`apply` on its no-LoRA path, ahead of
``fused_marlin_moe``; ``MarlinExperts.moe_sum`` calls :func:`moe_sum` ahead of
``ops.moe_sum``. Each returns True when it wrote ``output``, and False (with
nothing done) when upstream's code runs as before.
"""

import torch

import vllm.envs as envs


def apply(
    experts,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> bool:
    """The compiled Marlin decode (``VLLM_AMPERE_MARLIN_DECODE_CUDA``), then
    the sm_80 Marlin prefill (``VLLM_SM80_MARLIN_PREFILL``)."""
    if envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
        from vllm_sm80.kernels.marlin_moe.decode import (
            maybe_apply as decode_apply,
        )

        if decode_apply(
            experts, output, hidden_states, w1, w2, topk_weights,
            topk_ids, activation, global_num_experts, expert_map,
            apply_router_weight_on_input,
        ):
            return True
    if envs.VLLM_SM80_MARLIN_PREFILL:
        from vllm_sm80.kernels.marlin_moe.prefill import maybe_apply

        if maybe_apply(
            experts,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            workspace13,
            workspace2,
        ):
            return True
    return False


def moe_sum(
    experts,
    input: torch.Tensor,
    output: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> bool:
    """The deferred sum hand-off (``VLLM_AMPERE_DECODE_IDX_GLUE``), then the
    sm_80 routing sum (``use_ampere_moe_routing``)."""
    from vllm_sm80.kernels.moe import use_ampere_moe_routing

    if (
        envs.VLLM_AMPERE_DECODE_IDX_GLUE
        and expert_map is None
        and input.dim() == 3
    ):
        # VLLM_AMPERE_DECODE_IDX_GLUE "moesum": when the MoE runner armed the
        # hand-off it fuses this sum with the shared-expert add (same fp32
        # slot-order sum as both upstream branches, then the same add).
        from vllm_sm80.kernels.moe.sum import MOE_SUM_DEFERRAL

        if MOE_SUM_DEFERRAL.offer(input, output):
            return True

    _moe_cfg = getattr(experts, "moe_config", None)
    if (
        expert_map is None
        and _moe_cfg is not None
        and input.dim() == 3
        and use_ampere_moe_routing(
            input.shape[0],
            getattr(_moe_cfg, "num_experts", 0),
            input.shape[1],
        )
    ):
        # fp32 accumulation in slot order, same as ops.moe_sum.
        from vllm_sm80.kernels.moe.routing import moe_sum as ampere_moe_sum

        ampere_moe_sum(input, out=output)
        return True
    return False
