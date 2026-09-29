# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prebuilt sm_80 W4A16 MoE decode with the released routing/sum contract.

Sharded and whole-expert GEMMs use the admitted block-8 Marlin two-chain/stripe
reduction and a single full-sum W13 plane, retaining the released bf16 rounding
points. This descriptor is not a universal Marlin auto-selector; equivalence
admission must bind the ordinary binary and device. ``layer.moe_sum`` retains slot-order sum,
shared-expert deferral, and the subsequent shared add/all-reduce. Fully masked
CUDA-graph padding rows remain don't-care; no routing or padding flag changes.
Persistent buffers are shared by serialized calls on each device, as are the
released alignment buffers. No CUDA source compilation occurs here.
"""

import torch

import vllm.envs as envs
from vllm.ampere_prefill.pp_marlin_prefill import (
    E_GATE,
    K_GATE,
    TOPK_GATE,
    _is_sm80,
    gate_reason as _shape_gate_reason,
)
from vllm.logger import init_logger

logger = init_logger(__name__)
MAX_TOKENS = 32
_SCRATCH_TOKENS = {512: MAX_TOKENS, 2048: 8}
W13_ROWS = 64
_WORKSPACES: dict = {}
_RETIRED: list = []


def compiled_regime(num_tokens: int, intermediate_size: int) -> bool:
    """TP4 small batches and the measured whole-expert PP4 decode batches."""
    return (
        intermediate_size == 512 and 1 <= num_tokens <= MAX_TOKENS
        or intermediate_size == 2048 and num_tokens in (4, 8)
    )


def _config(num_tokens: int) -> tuple[int, int]:
    return (49 if num_tokens >= 16 else 0, 49 if num_tokens > 16 else 0)


def _workspaces(device, N: int, max_tokens: int, *, create: bool):
    key = (str(device), torch.cuda.current_stream(device).cuda_stream, N)
    ws = _WORKSPACES.get(key)
    if ws is not None and ws["max_tokens"] >= max_tokens:
        return ws
    if not create:
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Marlin decode scratch must be warmed before CUDA capture")
    capacity = max(_SCRATCH_TOKENS.get(N, 0), max_tokens)
    rows = capacity * TOPK_GATE
    padded = (rows + E_GATE * 7 + 7) // 8 * 8
    if ws is not None:
        _RETIRED.append(ws)  # Existing graphs may still reference these tensors.
    ws = {
        "max_tokens": capacity,
        "part": torch.zeros(
            rows * 2 * N,
            device=device, dtype=torch.float32,
        ),
        "h": torch.zeros(rows * N, device=device, dtype=torch.bfloat16),
        "c3": torch.zeros(rows * K_GATE, device=device, dtype=torch.bfloat16),
        "ctr": torch.zeros(2, device=device, dtype=torch.int32),
        "sorted": torch.empty(padded, device=device, dtype=torch.int32),
        "experts": torch.empty(padded // 8, device=device, dtype=torch.int32),
        "ntpp": torch.empty(1, device=device, dtype=torch.int32),
    }
    _WORKSPACES[key] = ws
    return ws


def _align(topk_ids, ws):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    if envs.VLLM_GLM5_DECODE_KERNELS:
        from vllm.ampere_decode import take_fused_align

        fused = take_fused_align(topk_ids, 8, E_GATE, None, False)
        if fused is not None:
            return fused
    mode = deterministic_moe_align_mode()
    if mode == 1:
        from vllm.model_executor.layers.fused_moe.moe_align_kernel import (
            kernel_moe_align_block_size as align,
        )
    elif mode == 0:
        from vllm._custom_ops import moe_align_block_size as align
    else:
        raise ValueError("Compiled decode does not use the allocating torch alignment")
    align(topk_ids, E_GATE, 8, ws["sorted"], ws["experts"], ws["ntpp"], None, None)
    return ws["sorted"], ws["experts"], ws["ntpp"]


def gate_reason(layer, hidden_states, w1, w2, topk_weights, topk_ids,
                activation, global_num_experts, expert_map,
                apply_router_weight_on_input):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    N = w2.size(1) * 16
    if not compiled_regime(hidden_states.size(0), N):
        return "outside the compiled decode token/width regime"
    if deterministic_moe_align_mode() == 2:
        return "torch deterministic alignment"
    if getattr(layer, "_lora_context", None) is not None:
        return "LoRA"
    if (
        tuple(w1.shape) != (E_GATE, K_GATE // 16, N * 4)
        or tuple(w2.shape) != (E_GATE, N // 16, K_GATE * 2)
    ):
        return "unsupported packed weight layout"
    tensors = (w1, w2, layer.w1_scale, layer.w2_scale, topk_weights, topk_ids)
    if any(t is None or not t.is_contiguous() for t in tensors):
        return "noncontiguous weights, scales, or routing"
    if any(t.device != hidden_states.device for t in tensors):
        return "weights and routing must be on the activation device"
    if topk_ids.dtype != torch.int32 or w1.dtype != torch.int32 or w2.dtype != torch.int32:
        return "packed weights and routing must be int32"
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        return "routing shape mismatch"
    if topk_ids.size(0) != hidden_states.size(0):
        return "routing token count mismatch"
    return _shape_gate_reason(
        layer, hidden_states, w1, w2, topk_weights, topk_ids, activation,
        global_num_experts, expert_map, apply_router_weight_on_input,
        allowed_n=(512, 2048),
    )


def maybe_apply(layer, output, hidden_states, w1, w2, topk_weights,
                topk_ids, activation, global_num_experts, expert_map,
                apply_router_weight_on_input) -> bool:
    if not envs.VLLM_GLM5_MARLIN_DECODE_CUDA:
        return False
    why = gate_reason(
        layer, hidden_states, w1, w2, topk_weights, topk_ids, activation,
        global_num_experts, expert_map, apply_router_weight_on_input,
    )
    if why is None:
        ws = _workspaces(
            hidden_states.device, w2.size(1) * 16, hidden_states.size(0),
            create=not torch.cuda.is_current_stream_capturing(),
        )
        if ws is None:
            why = "scratch not warmed before CUDA capture"
    if why is not None:
        logger.info_once(
            "VLLM_GLM5_MARLIN_DECODE_CUDA is set but the gate is closed (%s); "
            "using released Marlin kernels.", why,
        )
        return False
    logger.info_once(
        "Compiled Marlin MoE decode active (VLLM_GLM5_MARLIN_DECODE_CUDA): "
        "sm_80, N=512 with 1 <= M <= 32 or N=2048 with M in {4, 8}."
    )
    run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)
    return True


def run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, align=None):
    """Run into caller storage; direct calls also support N=2048 and M<=64.

    Call ``warmup`` first. ``align`` optionally supplies the released block-8
    alignment (three tensors and the integer block size).
    """
    from vllm.ampere_marlin import require_extension

    ops = require_extension()
    M, K = hidden_states.shape
    N = w2.size(1) * 16
    ws = _workspaces(hidden_states.device, N, M, create=False)
    if ws is None:
        raise RuntimeError("Call Marlin decode warmup before run/capture")
    if align is None:
        sorted_ids, expert_ids, ntpp = _align(topk_ids, ws)
    else:
        sorted_ids, expert_ids, ntpp, block_size = align
        if block_size != 8:
            raise ValueError("Compiled Marlin decode requires block-8 alignment")
    slots = M * TOPK_GATE
    h = ws["h"][:slots * N].view(slots, N)
    c3 = ws["c3"][:slots * K].view(slots, K)
    tw = topk_weights.view(-1)
    c13, c2 = _config(M)
    ksplit = ops.decode_gemm(
        hidden_states, w1, layer.w1_scale, sorted_ids, expert_ids, ntpp, tw,
        TOPK_GATE, slots, K, 2 * N, True, W13_ROWS, c13, ws["ctr"], ws["part"],
    )
    ops.decode_act(ws["part"], topk_ids.view(-1), ksplit, slots, N,
                   layer.activation_config.clamp_limit, h)
    ops.decode_gemm(
        h, w2, layer.w2_scale, sorted_ids, expert_ids, ntpp, tw,
        TOPK_GATE, slots, N, K, False, N // 16, c2, ws["ctr"], c3,
    )
    layer.moe_sum(c3.view(M, TOPK_GATE, K), output, topk_ids, None)
    return output


def warmup(device, N=512, max_tokens=None):
    """Reserve scratch and warm the selected released alignment implementation."""
    from vllm.ampere_marlin import require_extension
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    require_extension()
    if max_tokens is None:
        max_tokens = _SCRATCH_TOKENS[N]
    device = torch.device(device)
    with torch.cuda.device(device):
        ws = _workspaces(device, N, max_tokens, create=True)
        if deterministic_moe_align_mode() != 2:
            ids = torch.zeros(max_tokens, TOPK_GATE, device=device, dtype=torch.int32)
            _align(ids, ws)


def warmup_from_worker(worker):
    if not envs.VLLM_GLM5_MARLIN_DECODE_CUDA:
        return
    device = torch.device(worker.device)
    if device.type != "cuda" or not _is_sm80(device):
        return
    model_config = worker.vllm_config.model_config
    cfg = getattr(model_config, "hf_text_config", None) or model_config.hf_config
    if (getattr(cfg, "n_routed_experts", 0) != E_GATE
            or getattr(cfg, "num_experts_per_tok", 0) != TOPK_GATE):
        return
    tp = worker.vllm_config.parallel_config.tensor_parallel_size
    N = int(getattr(cfg, "moe_intermediate_size", 0)) // tp
    capacity = _SCRATCH_TOKENS.get(N, 0)
    if compiled_regime(capacity, N):
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            deterministic_moe_align_mode,
        )

        if deterministic_moe_align_mode() != 2:
            warmup(device, N, capacity)
