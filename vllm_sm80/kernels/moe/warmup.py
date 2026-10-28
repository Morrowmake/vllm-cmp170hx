# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-capture warmup of the sm_80 MoE routing kernels (see
``vllm_sm80/kernels/warmup_utils.py`` for why it must run before capture)."""

from vllm.logger import init_logger
from vllm_sm80.kernels.moe import (
    fused_route_max_rows,
    marlin_block_size_m,
    route_v2_masks_padding,
    route_v2_max_rows,
    use_ampere_moe_routing,
)
from vllm_sm80.kernels.warmup_utils import find_module, token_sizes

logger = init_logger(__name__)


def warmup_routing_from_worker(worker, capture_sizes) -> None:
    """No-op unless VLLM_AMPERE_DECODE_KERNELS and VLLM_GLM5_DECODE_MOE_ROUTING."""
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MOE_ROUTING:
        return
    _warmup_moe(
        worker, worker.get_model(), worker.device, capture_sizes,
        use_ampere_moe_routing,
    )


def warmup_route_from_worker(worker, capture_sizes) -> None:
    """No-op unless VLLM_AMPERE_DECODE_KERNELS and VLLM_MOE_DECODE_ROUTE_V2."""
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS or not envs.VLLM_MOE_DECODE_ROUTE_V2:
        return
    _warmup_moe_route(worker, worker.get_model(), worker.device, capture_sizes)


def _moe_shape(worker, model) -> tuple[int, int] | None:
    """(num_experts, topk) from the live router, or from the model config."""
    router = find_module(model, "global_num_experts")
    if router is not None and getattr(router, "top_k", None) is not None:
        return int(router.global_num_experts), int(router.top_k)
    cfg = getattr(worker.vllm_config.model_config, "hf_config", None)
    cfg = getattr(cfg, "text_config", cfg)
    experts = getattr(cfg, "n_routed_experts", None)
    topk = getattr(cfg, "num_experts_per_tok", None)
    if experts is None or topk is None:
        return None
    return int(experts), int(topk)


def _warmup_moe(worker, model, device, capture_sizes, gate) -> None:
    from vllm_sm80.kernels.moe import routing

    shape = _moe_shape(worker, model)
    if shape is None:
        return
    num_experts, topk = shape
    hidden = int(worker.vllm_config.model_config.get_hidden_size())
    ms = [
        m
        for m in token_sizes(capture_sizes, fused_route_max_rows(topk))
        if gate(m, num_experts, topk)
    ]
    if not ms:
        return
    # Every block size `fused_marlin_moe` can ask for: block_size is a
    # tl.constexpr, so each one is its own compiled variant, and the choice is
    # made after routing (see marlin_block_size_m).
    routing.warmup(
        ms,
        block_sizes=(8, 16, 32, 48, 64),
        topk=topk,
        num_experts=num_experts,
        hidden=hidden,
        device=device,
    )
    logger.info("Warmed up sm_80 MoE routing kernels for M in %s.", ms)


def _warmup_moe_route(worker, model, device, capture_sizes) -> None:
    """Compile every (M, block size) variant of route.py and allocate its
    scratch (split-K partials, bitmask columns, arrival counters) before any
    graph is captured."""
    from vllm_sm80.kernels.moe import route

    shape = _moe_shape(worker, model)
    if shape is None:
        return
    num_experts, topk = shape
    hidden = int(worker.vllm_config.model_config.get_hidden_size())
    if (num_experts, topk, hidden) != (288, 8, 4096):
        return
    ms = token_sizes(capture_sizes, route_v2_max_rows())
    if not ms:
        return
    for m in ms:
        route.warmup(
            (m,),
            block_sizes=(marlin_block_size_m(m, topk, num_experts),),
            topk=topk,
            num_experts=num_experts,
            hidden=hidden,
            device=device,
            # the in-kernel padding mask variant (VLLM_GLM5_MOE_ROUTE_V2_MASK)
            padded_ms=(m,) if route_v2_masks_padding(m) else (),
        )
    logger.info("Warmed up sm_80 fused MoE router for M in %s.", ms)
