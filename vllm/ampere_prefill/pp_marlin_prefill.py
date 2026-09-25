# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-block Marlin W4A16 MoE prefill on sm_80 (PP whole experts, TP4 shards).

WHAT.  Under pipeline parallelism with TP=1 every stage holds all 288 experts
whole (N=2048, K=4096; ``VLLM_GLM5_PP_MARLIN_PREFILL``); under tensor
parallel 4 every card holds all 288 sharded to N=512
(``VLLM_GLM5_TP4_MARLIN_PREFILL``).  ``fused_marlin_moe`` aligns the ~64 rows per expert
to one block size, so the last block of each expert is mostly padding (~1.55x
the useful rows at 2304 tokens).  This path replaces the alignment with
``moe_split_align`` (one block list per size 64/48/32/16, cheapest cover per
expert) and runs each Marlin GEMM as one launch per list, with the 64-row
list on a (thread_k 64, thread_n 256, 1 block/SM) tile.  Everything else is
the incumbent's: the same compiled Marlin kernels with fp32 reduce, the same
activation (``layer.activation``), the same slot-order sum
(``layer.moe_sum``), the same workspaces.  Marlin's per-row result does not
depend on the block a row lands in, so the output is bitwise equal to
``fused_marlin_moe`` on the measured PP shapes.  At N=512 it is not bitwise
equal (measured); its error against an fp64 reference equals the
incumbent's (1.00x mean and max on captured TP4 calls).  Under
TP4 the prefill overlap (two micro-batches) halves each chunk, so the calls
see M = 1728 (x2 per 3456-token chunk) and 640/642 for a 1282-token tail,
where the incumbent pads to 1.76x / 2.04x the routed rows and the split
cover to 1.16x / 1.45x (captured TP4 routing).  The cover's cost table was
fitted at N=2048; on the captured TP4 routing any per-block cost affine in
the block size gives the same covers, so it is kept.

EMPTY LISTS.  A list with no block (e.g. no 64-row block at a few hundred
tokens) is still launched: sizes stay on the device, so skipping it would
need a host sync.  The kernel reads ``num_tokens_past_padded`` = 0, so it has
zero m-blocks (``parallel`` = 0) and zero tiles; the slice setup returns
before reading any sorted id, expert id, weight or scale, ``slice_iters`` is
0, the main loop never runs and nothing is written.

GATE.  Taken only for exactly the validated configuration: sm_80, bf16
activations, uint4b8 group-128 weights without zero points, bias, global
scales or activation quantisation, E = 288 local = global (no expert map),
top-8, K = 4096, N = 2048, SiLU with clamp limit 10.0, router weight applied
on the output, and M >= VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS; or the same
with N = 512 and M >= VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS under
VLLM_GLM5_TP4_MARLIN_PREFILL.  Everything else falls through to
``fused_marlin_moe`` unchanged.

CUDA GRAPHS.  No host sync and no per-call allocation: the list buffers are
allocated once per (device, E) for max_num_batched_tokens rows by ``warmup``
(called before capture from ``kernel_warmup``), and a call that would need a
new or larger buffer while a capture is running falls through to the
incumbent instead.  Like the deterministic MoE alignment's scratch, one
buffer set serves every call on the device, so calls must be serialised on
one stream.
"""

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

E_GATE = 288
TOPK_GATE = 8
K_GATE = 4096
N_GATE = 2048
N_GATE_TP4 = 512
GROUP_SIZE = 128
CLAMP_LIMIT = 10.0

# (thread_k, thread_n, blocks_per_sm) per block-list size; absent = Marlin's
# own exec-config choice.  Only configurations the compiled Marlin MoE library
# already instantiates: (64, 256) at 256 threads exists for m-blocks 2..4.
THREAD_CFG = {64: (64, 256, 1)}

_BUFFERS: dict = {}
_RETIRED: list = []
_CAP_OK: dict = {}


def _thread_cfg(bs: int, size_n: int, size_k: int, table: dict) -> tuple:
    c = table.get(bs)
    if c is not None and size_n % c[1] == 0 and size_k % c[0] == 0:
        return c
    return (-1, -1, -1)


def _is_sm80(device: torch.device) -> bool:
    key = device.index if device.index is not None else torch.cuda.current_device()
    ok = _CAP_OK.get(key)
    if ok is None:
        ok = torch.cuda.get_device_capability(key) == (8, 0)
        _CAP_OK[key] = ok
    return ok


def enabled_shapes() -> dict:
    """{N: (flag, min_tokens)} for the flags that are set: N=2048 under
    VLLM_GLM5_PP_MARLIN_PREFILL, N=512 under VLLM_GLM5_TP4_MARLIN_PREFILL."""
    out = {}
    if envs.VLLM_GLM5_PP_MARLIN_PREFILL:
        out[N_GATE] = ("VLLM_GLM5_PP_MARLIN_PREFILL",
                       envs.VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS)
    if envs.VLLM_GLM5_TP4_MARLIN_PREFILL:
        out[N_GATE_TP4] = ("VLLM_GLM5_TP4_MARLIN_PREFILL",
                           envs.VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS)
    return out


def _buffers(device: torch.device, E: int, rows: int, create: bool):
    from vllm.ampere_prefill import moe_split_align

    key = (str(device), E)
    buf = _BUFFERS.get(key)
    if buf is not None and buf["t_max"] >= rows:
        return buf
    if not create:
        return None
    if buf is not None:
        # a captured graph may still point at the old set: never free it
        _RETIRED.append(buf)
    buf = moe_split_align.buffers(rows, E, device)
    _BUFFERS[key] = buf
    return buf


def gate_reason(layer, hidden_states: torch.Tensor, w1: torch.Tensor,
                w2: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor, activation, global_num_experts: int,
                expert_map, apply_router_weight_on_input: bool,
                allowed_n: tuple = (N_GATE,)) -> str | None:
    """None when the split path may run, else why not (for the banner).
    ``allowed_n``: the intermediate sizes whose flag is set.

    Structural conditions first, the device last, so the gate is testable
    without a GPU."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_intermediate_size,
    )
    from vllm.scalar_type import scalar_types

    if expert_map is not None:
        return "expert map (expert parallelism)"
    E = w1.size(0)
    if E != E_GATE or global_num_experts not in (-1, E):
        return f"E={E} (global {global_num_experts}), validated only E={E_GATE}"
    if topk_ids.dim() != 2 or topk_ids.size(1) != TOPK_GATE:
        return f"top-k {tuple(topk_ids.shape)[1:]} != {TOPK_GATE}"
    if hidden_states.dim() != 2 or hidden_states.size(1) != K_GATE:
        return f"hidden size {tuple(hidden_states.shape)[1:]} != {K_GATE}"
    N = marlin_moe_intermediate_size(w1, w2)
    if N not in allowed_n:
        if tuple(allowed_n) == (N_GATE,):
            return f"intermediate size N={N} != {N_GATE} (TP-sharded experts)"
        return f"intermediate size N={N} not in {tuple(allowed_n)}"
    if getattr(layer, "input_dtype", None) is not None:
        return f"activation quantisation to {layer.input_dtype}"
    if hidden_states.dtype != torch.bfloat16:
        return f"activation dtype {hidden_states.dtype}"
    for name in ("w1_zp", "w2_zp", "w1_bias", "w2_bias", "g1_alphas",
                 "g2_alphas", "a1_gscale", "a2_gscale"):
        if getattr(layer, name, None) is not None:
            return f"{name} is set"
    if layer.quant_type_id != scalar_types.uint4b8.id:
        return "weight type is not uint4b8"
    s1, s2 = layer.w1_scale, layer.w2_scale
    if (s1 is None or s2 is None or s1.dtype != torch.bfloat16
            or tuple(s1.shape) != (E, K_GATE // GROUP_SIZE, 2 * N)
            or tuple(s2.shape) != (E, N // GROUP_SIZE, K_GATE)):
        return "scales are not bf16 group-128"
    if apply_router_weight_on_input:
        return "router weight applied on the input"
    cfg = layer.activation_config
    if (activation != MoEActivation.SILU or cfg.clamp_limit != CLAMP_LIMIT
            or cfg.alpha != 1.0 or cfg.beta != 0.0):
        return f"activation {activation} clamp {cfg.clamp_limit}"
    if topk_weights.dtype != torch.float32 or not hidden_states.is_contiguous():
        return "topk weights not fp32 or hidden states not contiguous"
    if not hidden_states.is_cuda or not _is_sm80(hidden_states.device):
        return "not sm_80"
    return None


def maybe_apply(layer, output: torch.Tensor, hidden_states: torch.Tensor,
                w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor, activation, global_num_experts: int,
                expert_map, apply_router_weight_on_input: bool,
                workspace13: torch.Tensor, workspace2: torch.Tensor) -> bool:
    """Run the split path into ``output`` and return True, or return False
    (nothing touched) so the caller runs ``fused_marlin_moe``."""
    M = hidden_states.size(0)
    shapes = enabled_shapes()
    if not shapes or M < min(m for _, m in shapes.values()):
        return False
    why = gate_reason(layer, hidden_states, w1, w2, topk_weights, topk_ids,
                      activation, global_num_experts, expert_map,
                      apply_router_weight_on_input, tuple(shapes))
    flag = " / ".join(f for f, _ in shapes.values())
    min_tokens = 0
    if why is None:
        flag, min_tokens = shapes[w2.size(1) * 16]
        if M < min_tokens:
            return False
        rows = M * topk_ids.size(1)
        buf = _buffers(hidden_states.device, w1.size(0), rows,
                       create=not torch.cuda.is_current_stream_capturing())
        if buf is None:
            why = f"no list buffers for {rows} rows during a graph capture"
    if why is not None:
        logger.info_once(
            "%s is set but the gate is closed (%s); using fused_marlin_moe.",
            flag, why)
        return False
    logger.info_once(
        "%s Marlin MoE prefill active (%s): split 64/48/32/16-row block lists "
        "for M >= %d.", "PP" if flag == "VLLM_GLM5_PP_MARLIN_PREFILL" else "TP4",
        flag, min_tokens)
    run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, workspace13, workspace2, buf)
    return True


def run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, workspace13, workspace2, buf, thread_cfg=None):
    """The split path. Workspaces as ``MarlinExperts.apply`` passes them to
    ``fused_marlin_moe`` (workspace2 holds w13's and w2's outputs, workspace13
    the activation)."""
    from vllm.ampere_prefill.moe_split_align import split_align
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_marlin_workspace,
    )
    from vllm.scalar_type import scalar_types

    table = THREAD_CFG if thread_cfg is None else thread_cfg
    M, K = hidden_states.shape
    E = w1.size(0)
    topk = topk_ids.size(1)
    N = w2.size(1) * 16
    rows = M * topk
    qt = scalar_types.uint4b8
    workspace = get_marlin_workspace(hidden_states.device)
    c1 = _resize_cache(workspace2, (rows, 2 * N))
    c3 = _resize_cache(workspace2, (rows, K))
    c2 = _resize_cache(workspace13, (rows, N))

    lists = split_align(topk_ids, E, buf)
    for bs, sorted_ids, expert_ids, ntpp in lists:
        tk, tn, bps = _thread_cfg(bs, 2 * N, K, table)
        ops.moe_wna16_marlin_gemm(
            hidden_states, c1, w1, None, layer.w1_scale, None, None, None,
            workspace, sorted_ids, expert_ids, ntpp, topk_weights,
            moe_block_size=bs, top_k=topk, mul_topk_weights=False,
            b_q_type=qt, size_m=M, size_n=2 * N, size_k=K,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
            thread_k=tk, thread_n=tn, blocks_per_sm=bps)
    layer.activation(activation, c2, c1, topk_ids=topk_ids, expert_map=None)
    for bs, sorted_ids, expert_ids, ntpp in lists:
        tk, tn, bps = _thread_cfg(bs, K, N, table)
        ops.moe_wna16_marlin_gemm(
            c2, c3, w2, None, layer.w2_scale, None, None, None,
            workspace, sorted_ids, expert_ids, ntpp, topk_weights,
            moe_block_size=bs, top_k=1, mul_topk_weights=True,
            b_q_type=qt, size_m=rows, size_n=K, size_k=N,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
            thread_k=tk, thread_n=tn, blocks_per_sm=bps)
    layer.moe_sum(c3.view(M, topk, K), output, topk_ids, None)


def warmup(device, num_experts: int, max_tokens: int, topk: int) -> None:
    """Allocate the list buffers for the current stream and compile the
    split-alignment kernels.  Must run before any CUDA-graph capture."""
    from vllm.ampere_prefill.moe_split_align import split_align

    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return
    with torch.cuda.device(device):
        buf = _buffers(device, num_experts, max_tokens * topk, create=True)
        ids = torch.zeros(1, topk, device=device, dtype=torch.int32)
        split_align(ids, num_experts, buf)
        torch.cuda.synchronize(device)


def warmup_from_worker(worker) -> int:
    """Pre-capture hook from ``model_executor/warmup/kernel_warmup.py``.
    No-op unless VLLM_GLM5_PP_MARLIN_PREFILL or VLLM_GLM5_TP4_MARLIN_PREFILL
    is set on an sm_80 device and the model has 288 routed experts; returns
    the expert count warmed (0 if not)."""
    if not (envs.VLLM_GLM5_PP_MARLIN_PREFILL or envs.VLLM_GLM5_TP4_MARLIN_PREFILL):
        return 0
    device = torch.device(worker.device)
    if device.type != "cuda" or not _is_sm80(device):
        return 0
    model_config = worker.vllm_config.model_config
    cfg = getattr(model_config, "hf_text_config", None) or model_config.hf_config
    E = int(getattr(cfg, "n_routed_experts", 0) or 0)
    topk = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    if E != E_GATE or topk != TOPK_GATE:
        return 0
    max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    warmup(device, E, max_tokens, topk)
    return E
