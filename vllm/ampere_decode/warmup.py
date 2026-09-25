# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-capture warmup for the sm_80 decode kernels.

WHY THIS MUST RUN BEFORE CUDA GRAPH CAPTURE. Two of the three families keep an
allocate-once module-level buffer -- ``moe_routing._scratch(device, E)`` and
``kda_decode._counter(device)``. Both are left zeroed by every launch, which is
what makes a graph replay correct, and neither allocates on the timed path.
But if the first call happened *during* capture the allocation would land in
that graph's private memory pool and a second capture would reuse freed
memory. Triton also compiles on first launch, which cannot happen inside a
capture at all. So every ``warmup()`` here runs from
``vllm/model_executor/warmup/kernel_warmup.py``, which
``Worker.compile_or_warm_up_model`` calls immediately before
``capture_model()``.

Covers the batch sizes the decode graphs are captured for, intersected with
each family's token bound (a size above the bound can never dispatch).
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Small M values that a non-captured (eager) decode step can hit. Compiling
# them here costs a few seconds of startup and avoids a first-launch stall.
_EXTRA_MS = (1, 2, 4, 8)


def _token_sizes(capture_sizes, bound: int) -> list[int]:
    sizes = {int(s) for s in capture_sizes if 1 <= int(s) <= bound}
    sizes.update(m for m in _EXTRA_MS if m <= bound)
    return sorted(sizes)


def _find_module(model, attr: str):
    for module in model.modules():
        if hasattr(module, attr):
            return module
    return None


def warmup_ampere_decode(worker, capture_sizes) -> None:
    """Compile and allocate everything the three families need, per family."""
    from vllm import envs
    from vllm.ampere_decode import (
        use_ampere_kda_decode,
        use_ampere_mhc_decode,
        use_ampere_mhc_decode_v2,
        use_ampere_moe_routing,
    )

    if not envs.VLLM_GLM5_DECODE_KERNELS:
        return

    model = worker.get_model()
    device = worker.device

    if envs.VLLM_GLM5_DECODE_MHC_V2:
        _warmup_mhc_v2(model, device, capture_sizes, use_ampere_mhc_decode_v2)
    if envs.VLLM_GLM5_DECODE_MHC:
        _warmup_mhc(model, device, capture_sizes, use_ampere_mhc_decode)
    if envs.VLLM_GLM5_DECODE_MOE_ROUTING:
        _warmup_moe(worker, model, device, capture_sizes, use_ampere_moe_routing)
    if envs.VLLM_GLM5_DECODE_KDA:
        _warmup_kda(worker, model, device, capture_sizes, use_ampere_kda_decode)
    if envs.VLLM_GLM5_DECODE_KDA_V2:
        _warmup_kda_v2(worker, model, device, capture_sizes)
    if envs.VLLM_GLM5_DECODE_MOE_ROUTE_V2:
        _warmup_moe_route(worker, model, device, capture_sizes)


def _warmup_mhc(model, device, capture_sizes, gate) -> None:
    from vllm import envs
    from vllm.ampere_decode import mhc_decode

    layer = _find_module(model, "mhc_fused_post_pre_op")
    if layer is None:
        return
    hidden = int(layer.hidden_size)
    hc = int(layer.n)
    sinkhorn = int(layer.mhc_sinkhorn_iterations)
    ms = [
        m
        for m in _token_sizes(capture_sizes, envs.VLLM_GLM5_DECODE_MHC_MAX_TOKENS)
        if gate(m, hc, hidden)
    ]
    if not ms:
        return
    mhc_decode.warmup(ms, hidden=hidden, hc=hc, sinkhorn=sinkhorn, device=device)
    logger.info("Warmed up sm_80 mHC decode kernels for M in %s.", ms)


def _warmup_mhc_v2(model, device, capture_sizes, gate) -> None:
    """Compile the v2 mHC kernels for every capture size its gate accepts.

    The v2 kernels keep no module-level buffers (outputs and split partials are
    allocated per call, from the graph pool during capture), so compiling
    before capture is all that is needed.
    """
    from vllm import envs
    from vllm.ampere_decode import mhc_decode_v2

    layer = _find_module(model, "mhc_fused_post_pre_op")
    if layer is None:
        return
    hidden = int(layer.hidden_size)
    hc = int(layer.n)
    sinkhorn = int(layer.mhc_sinkhorn_iterations)
    bound = envs.VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS
    ms = [m for m in _token_sizes(capture_sizes, bound) if gate(m, hc, hidden)]
    if not ms:
        logger.info("sm_80 mHC decode v2 requested but its gate is closed "
                    "(hc=%d, hidden=%d); using the previous paths.", hc, hidden)
        return
    mhc_decode_v2.warmup(ms, hidden=hidden, hc=hc, sinkhorn=sinkhorn, device=device)
    logger.info("sm_80 mHC decode v2 on for M <= %d; warmed up M in %s.", bound, ms)


def _moe_shape(worker, model) -> tuple[int, int] | None:
    """(num_experts, topk) from the live router, or from the model config."""
    router = _find_module(model, "global_num_experts")
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
    from vllm import envs
    from vllm.ampere_decode import moe_routing

    shape = _moe_shape(worker, model)
    if shape is None:
        return
    num_experts, topk = shape
    hidden = int(worker.vllm_config.model_config.get_hidden_size())
    ms = [
        m
        for m in _token_sizes(capture_sizes, envs.VLLM_GLM5_DECODE_MOE_MAX_TOKENS)
        if gate(m, num_experts, topk)
    ]
    if not ms:
        return
    # Every block size `fused_marlin_moe` can ask for: block_size is a
    # tl.constexpr, so each one is its own compiled variant, and the choice is
    # made after routing (see marlin_block_size_m).
    moe_routing.warmup(
        ms,
        block_sizes=(8, 16, 32, 48, 64),
        topk=topk,
        num_experts=num_experts,
        hidden=hidden,
        device=device,
    )
    logger.info("Warmed up sm_80 MoE routing kernels for M in %s.", ms)


def _warmup_kda(worker, model, device, capture_sizes, gate) -> None:
    from vllm import envs
    from vllm.ampere_decode import kda_decode

    layer = _find_module(model, "_conv_state_dim_first")
    if layer is None:
        return
    heads = int(layer.local_num_heads)
    head_dim = int(layer.head_dim)
    num_spec = int(getattr(layer, "num_spec", 0) or 0)
    tokens_per_seq = num_spec + 1
    max_seqs = int(worker.vllm_config.scheduler_config.max_num_seqs)
    bound = envs.VLLM_GLM5_DECODE_KDA_MAX_TOKENS
    # `_select(nseq, H, D)` picks BV from the sequence count, and BS/BT are
    # constexprs of max_query_len, so one variant per (nseq, tokens_per_seq).
    # Capture sizes are token counts; cover every sequence count that can
    # produce one, and every sequence count the scheduler can admit.
    nseqs = {
        max(1, int(s) // tokens_per_seq) for s in capture_sizes if int(s) >= 1
    }
    nseqs.update(range(1, max_seqs + 1))
    plans = [
        (n, tokens_per_seq)
        for n in sorted(nseqs)
        if n * tokens_per_seq <= bound
        and gate(n, n * tokens_per_seq, heads, head_dim)
    ]
    if not plans:
        return
    kda_decode.warmup(plans=tuple(plans), device=torch.device(device))
    logger.info("Warmed up sm_80 KDA decode kernel for (nseq, T) in %s.", plans)


def _warmup_kda_v2(worker, model, device, capture_sizes) -> None:
    """Compile kda_decode_v2 for every (nseq, T) a decode graph can hold and
    allocate its arrival counters and gate workspace, before capture."""
    from vllm.ampere_decode import kda_decode_v2, use_ampere_kda_decode_v2

    layer = _find_module(model, "_conv_state_dim_first")
    if layer is None:
        return
    heads = int(layer.local_num_heads)
    head_dim = int(layer.head_dim)
    num_spec = int(getattr(layer, "num_spec", 0) or 0)
    tokens_per_seq = num_spec + 1
    max_seqs = int(worker.vllm_config.scheduler_config.max_num_seqs)
    nseqs = {
        max(1, int(s) // tokens_per_seq) for s in capture_sizes if int(s) >= 1
    }
    nseqs.update(range(1, max_seqs + 1))
    plans = [
        (n, tokens_per_seq)
        for n in sorted(nseqs)
        if use_ampere_kda_decode_v2(
            n,
            n * tokens_per_seq,
            heads,
            head_dim,
            layer.f_b_proj.weight,
            layer.g_b_proj.weight,
        )
    ]
    if not plans:
        logger.info(
            "sm_80 KDA decode v2 requested but its gate is closed at %d heads "
            "(VLLM_GLM5_DECODE_KDA_V2_WIDE_MAX_SEQS=0 or an unsupported shape); "
            "using the previous path.",
            heads,
        )
        return
    kda_decode_v2.warmup(
        plans=tuple(plans), device=torch.device(device), heads=heads
    )
    logger.info(
        "Warmed up sm_80 KDA decode v2 kernel for (nseq, T) in %s at %d heads.",
        plans,
        heads,
    )


def _warmup_moe_route(worker, model, device, capture_sizes) -> None:
    """Compile every (M, block size) variant of moe_route.py and allocate its
    scratch (split-K partials, bitmask columns, arrival counters) before any
    graph is captured."""
    from vllm import envs
    from vllm.ampere_decode import (
        marlin_block_size_m,
        moe_route,
        route_v2_masks_padding,
    )

    shape = _moe_shape(worker, model)
    if shape is None:
        return
    num_experts, topk = shape
    hidden = int(worker.vllm_config.model_config.get_hidden_size())
    if (num_experts, topk, hidden) != (288, 8, 4096):
        return
    ms = _token_sizes(capture_sizes, envs.VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS)
    if not ms:
        return
    for m in ms:
        moe_route.warmup(
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
