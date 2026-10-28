# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prebuilt sm_80 W4A16 MoE decode with the released routing/sum contract.

The W13 projection is split four ways along K into fixed-order fp32 partials;
this is faster than the released kernels and changes the fp32 summation
order, so decoded text may differ from them while staying within their fp64
accuracy bounds. The released bf16 rounding points are retained. This
descriptor is not a universal Marlin auto-selector; equivalence
admission must bind the ordinary binary and device. ``layer.moe_sum`` retains slot-order sum,
shared-expert deferral, and the subsequent shared add/all-reduce. Fully masked
CUDA-graph padding rows remain don't-care; no routing or padding flag changes.
Persistent buffers are shared by serialized calls on each device, as are the
released alignment buffers. No CUDA source compilation occurs here.

VLLM_AMPERE_MARLIN_DECODE_V2 swaps decode_gemm_orig for decode_gemm_v2 on the
tensor-parallel shard (N=512): the same partial planes and per-output order,
an exact one-FMA dequant and per-row-count tiles. Its dequant is exact for
weight scales 0 or in [2^-110, 2^10) (checked; up to the sign of zero
weights); it carries the fp32 sums at 2^-16 scale, so the output equals
decode_gemm_orig's bit for bit as long as no nonzero weight-activation
product falls below 2^-110 in magnitude (a subnormal there, which the mma is
assumed to keep unflushed): far below production magnitudes.
"""

import torch

import vllm.envs as envs
from vllm_sm80 import tuning
from vllm_sm80.kernels.marlin_moe.prefill import (
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
# Whole-expert multi-request rows: PP_MULTI_MIN up to the
# moe_routed_gemm/marlin_multi_max_rows row of the sm80 tuning table.
PP_MULTI_MIN, PP_MULTI_MAX = 9, 64
W13_ROWS = 64
_WORKSPACES: dict = {}
# W13 fp32 partial planes (the kernel splits K four ways).
_W13_PLANES = 4
_RETIRED: list = []
# decode_gemm_v2: the expert width it is built and validated for, and its
# exact-dequant scale range as bf16 magnitude bits (2^-110 = 0x0880,
# 2^10 = 0x4480); scales are checked in chunks of _V2_SCALE_CHUNK.
_V2_N = 512
_V2_SCALE_MIN_BITS, _V2_SCALE_END_BITS = 0x0880, 0x4480
_V2_SCALE_CHUNK = 1 << 20
# The decode scratch (TP4 shards N=512 and whole experts N=2048) is created only
# after the KV cache is sized (kernel_warmup opens it). Earlier calls -- the
# profile run and the CUDA-graph memory profiling, which runs capture warmups
# before KV sizing -- take the released kernels, so the scratch is never
# charged against the KV cache.
_LATE_SCRATCH_N = (512, 2048)
_late_scratch_open = False


def open_late_scratch() -> None:
    """Allow the decode scratch from now on (called after KV sizing)."""
    global _late_scratch_open
    _late_scratch_open = True


def multi_max_rows() -> int:
    """Largest multi-request row count of the whole-expert (N=2048) compiled
    decode: the moe_routed_gemm/marlin_multi_max_rows row of the sm80 tuning
    table, else 32 (the most rows one PP4 step verifies with DEPTHS 7,5 and
    8 sequences)."""
    return tuning.lookup(
        "moe_routed_gemm", "marlin_multi_max_rows", {"gemm_n": 2048}, 32
    )


def pp_multi_ranges() -> tuple[tuple[int, int], ...]:
    """((PP_MULTI_MIN, multi_max_rows()),), or () when the row lies outside
    PP_MULTI_MIN..PP_MULTI_MAX (the gate then stays closed)."""
    hi = multi_max_rows()
    if not PP_MULTI_MIN <= hi <= PP_MULTI_MAX:
        return ()
    return ((PP_MULTI_MIN, hi),)


def _scratch_tokens(N: int) -> int:
    """Rows the persistent scratch of width N is sized for."""
    base = _SCRATCH_TOKENS.get(N, 0)
    if N == 2048:
        base = max([base] + [hi for _, hi in pp_multi_ranges()])
    return base


def compiled_regime(num_tokens: int, intermediate_size: int) -> bool:
    """TP4 small batches and the measured whole-expert PP4 decode batches
    (4..8 rows, and the multi-request rows of pp_multi_ranges)."""
    return (
        intermediate_size == 512 and 1 <= num_tokens <= MAX_TOKENS
        or intermediate_size == 2048 and (
            4 <= num_tokens <= 8
            or num_tokens >= PP_MULTI_MIN and any(
                lo <= num_tokens <= hi for lo, hi in pp_multi_ranges())
        )
    )


def _config(num_tokens: int) -> tuple[int, int]:
    return (49 if num_tokens >= 16 else 0, 49 if num_tokens > 16 else 0)


def _config_v2(num_tokens: int) -> tuple[int, int]:
    """decode_gemm_v2 cfg codes (w13, w2): index into its AMPERE_MARLIN_CFGS +
    16 * max CTAs per SM. 2-3 rows: 4-row chunks (cfg 2); 4-7 rows: 4-row
    chunks x two tiles (cfg 3); 32 rows: cfg 9 (cfg 1 with zero activation
    fragments for padding tokens), at most 3 CTAs per SM; otherwise
    ``_config``'s codes."""
    if 4 <= num_tokens < 8:
        return (3, 3)
    if 2 <= num_tokens < 4:
        return (2, 2)
    if num_tokens >= 32:
        return (57, 57)
    return _config(num_tokens)


def _scales_in_v2_range(scale: torch.Tensor) -> bool:
    """True when every bf16 scale is 0 or 2^-110 <= |s| < 2^10, where
    decode_gemm_v2's dequant is exact. One host sync; chunked so the
    temporaries stay small."""
    if scale.dtype != torch.bfloat16:
        return False
    bad = []
    for chunk in scale.reshape(-1).split(_V2_SCALE_CHUNK):
        mag = chunk.view(torch.int16) & 0x7FFF
        bad.append(((mag != 0) & ((mag < _V2_SCALE_MIN_BITS)
                                  | (mag >= _V2_SCALE_END_BITS))).any())
    return not bool(torch.stack(bad).any())


def _v2_gemm(layer, N: int):
    """decode_gemm_v2 for this layer, or None (decode_gemm_orig runs; the
    reason is logged once). The scale range is checked on the first call
    outside CUDA capture and cached on the layer per scale tensor object:
    scales rewritten in place afterwards are not rechecked."""
    from vllm_sm80.kernels.marlin_moe.extension import decode_v2_op

    op, why = decode_v2_op()
    if op is not None and N != _V2_N:
        op, why = None, "expert width N=%d per rank; validated for N=%d" % (N, _V2_N)
    if op is not None:
        s1, s2 = layer.w1_scale, layer.w2_scale
        checked = getattr(layer, "_ampere_marlin_decode_v2_scales", None)
        if checked is None or checked[0] is not s1 or checked[1] is not s2:
            if torch.cuda.is_current_stream_capturing():
                checked = None
                op, why = None, "weight scales not checked before CUDA capture"
            else:
                checked = (s1, s2, _scales_in_v2_range(s1)
                           and _scales_in_v2_range(s2))
                layer._ampere_marlin_decode_v2_scales = checked
        if checked is not None and not checked[2]:
            op, why = None, "a weight scale lies outside [2^-110, 2^10)"
    if op is None:
        logger.info_once(
            "VLLM_AMPERE_MARLIN_DECODE_V2 is set but the gate is closed (%s); "
            "using decode_gemm_orig.", why,
        )
        return None
    logger.info_once(
        "Compiled Marlin MoE decode v2 active (VLLM_AMPERE_MARLIN_DECODE_V2): "
        "sm_80, N=%d, 1 <= M <= %d.", _V2_N, MAX_TOKENS,
    )
    return op


def _workspaces(device, N: int, max_tokens: int, *, create: bool):
    key = (str(device), torch.cuda.current_stream(device).cuda_stream, N)
    ws = _WORKSPACES.get(key)
    if ws is not None and ws["max_tokens"] >= max_tokens:
        return ws
    if not create:
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Marlin decode scratch must be warmed before CUDA capture")
    capacity = max(_scratch_tokens(N), max_tokens)
    rows = capacity * TOPK_GATE
    padded = (rows + E_GATE * 7 + 7) // 8 * 8
    if ws is not None:
        _RETIRED.append(ws)  # Existing graphs may still reference these tensors.
    ws = {
        "max_tokens": capacity,
        "part": torch.zeros(
            _W13_PLANES * rows * 2 * N,
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
    from vllm_sm80.kernels.moe.align import (
        deterministic_moe_align_mode,
    )

    if envs.VLLM_AMPERE_DECODE_KERNELS:
        from vllm_sm80.kernels.moe import take_fused_align

        fused = take_fused_align(topk_ids, 8, E_GATE, None, False)
        if fused is not None:
            return fused
    if deterministic_moe_align_mode() == 1:
        from vllm_sm80.kernels.moe.align import (
            kernel_moe_align_block_size as align,
        )
    else:
        from vllm._custom_ops import moe_align_block_size as align
    align(topk_ids, E_GATE, 8, ws["sorted"], ws["experts"], ws["ntpp"], None, None)
    return ws["sorted"], ws["experts"], ws["ntpp"]


def gate_reason(layer, hidden_states, w1, w2, topk_weights, topk_ids,
                activation, global_num_experts, expert_map,
                apply_router_weight_on_input):
    N = w2.size(1) * 16
    if not compiled_regime(hidden_states.size(0), N):
        return "outside the compiled decode token/width regime"
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
    if not envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
        return False
    why = gate_reason(
        layer, hidden_states, w1, w2, topk_weights, topk_ids, activation,
        global_num_experts, expert_map, apply_router_weight_on_input,
    )
    if why is None:
        N = w2.size(1) * 16
        early = N in _LATE_SCRATCH_N and not _late_scratch_open
        ws = _workspaces(
            hidden_states.device, N, hidden_states.size(0),
            create=not early and not torch.cuda.is_current_stream_capturing(),
        )
        if ws is None:
            why = ("decode scratch is allocated after KV-cache sizing"
                   if early else "scratch not warmed before CUDA capture")
    if why is not None:
        logger.info_once(
            "VLLM_AMPERE_MARLIN_DECODE_CUDA is set but the gate is closed (%s); "
            "using released Marlin kernels.", why,
        )
        return False
    logger.info_once(
        "Compiled Marlin MoE decode active (VLLM_AMPERE_MARLIN_DECODE_CUDA): "
        "sm_80, N=512 with 1 <= M <= 32 or N=2048 with M in 4..8%s.",
        "".join(", %d..%d" % r for r in pp_multi_ranges()),
    )
    gemm_v2 = None
    if envs.VLLM_AMPERE_MARLIN_DECODE_V2:
        gemm_v2 = _v2_gemm(layer, w2.size(1) * 16)
    run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
        gemm_v2=gemm_v2)
    return True


def run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, gemm_v2=None):
    """Run into caller storage; direct calls also support N=2048 and M<=64.
    ``gemm_v2`` (from ``_v2_gemm``) replaces decode_gemm_orig and its configs
    for both GEMMs.

    Call ``warmup`` first.
    """
    from vllm_sm80.kernels.marlin_moe.extension import require_extension

    ops = require_extension()
    M, K = hidden_states.shape
    N = w2.size(1) * 16
    ws = _workspaces(hidden_states.device, N, M, create=False)
    if ws is None:
        raise RuntimeError("Call Marlin decode warmup before run/capture")
    sorted_ids, expert_ids, ntpp = _align(topk_ids, ws)
    slots = M * TOPK_GATE
    h = ws["h"][:slots * N].view(slots, N)
    c3 = ws["c3"][:slots * K].view(slots, K)
    tw = topk_weights.view(-1)
    if gemm_v2 is None:
        gemm = ops.decode_gemm_orig
        c13, c2 = _config(M)
    else:
        gemm = gemm_v2
        c13, c2 = _config_v2(M)
    ksplit = gemm(
        hidden_states, w1, layer.w1_scale, sorted_ids, expert_ids, ntpp, tw,
        TOPK_GATE, slots, K, 2 * N, True, W13_ROWS, c13, ws["ctr"], ws["part"],
    )
    ops.decode_act_orig(ws["part"], topk_ids.view(-1), ksplit, slots, N,
                        layer.activation_config.clamp_limit, h)
    gemm(
        h, w2, layer.w2_scale, sorted_ids, expert_ids, ntpp, tw,
        TOPK_GATE, slots, N, K, False, N // 16, c2, ws["ctr"], c3,
    )
    layer.moe_sum(c3.view(M, TOPK_GATE, K), output, topk_ids, None)
    return output


def _warm_v2(device, ws) -> None:
    """Launch every decode_gemm_v2 variant ``_config_v2`` can pick once with
    no work items (ntpp = 0), so kernel loading and the per-variant attribute
    setup happen before CUDA-graph capture. Writes nothing but the zeroed
    work counters."""
    from vllm_sm80.kernels.marlin_moe.extension import decode_v2_op

    op, _ = decode_v2_op()
    if op is None:
        return
    K, N = K_GATE, _V2_N
    w = torch.zeros(4, device=device, dtype=torch.int32)
    s = torch.zeros(8, device=device, dtype=torch.bfloat16)
    a13 = torch.zeros(1, K, device=device, dtype=torch.bfloat16)
    a2 = torch.zeros(1, N, device=device, dtype=torch.bfloat16)
    ntpp = torch.zeros(1, device=device, dtype=torch.int32)
    tw = torch.zeros(MAX_TOKENS * TOPK_GATE, device=device, dtype=torch.float32)
    for m in (1, 2, 4, 16, 17, MAX_TOKENS):
        c13, c2 = _config_v2(m)
        slots = m * TOPK_GATE
        op(a13, w, s, ws["sorted"], ws["experts"], ntpp, tw, TOPK_GATE, slots,
           K, 2 * N, True, W13_ROWS, c13, ws["ctr"], ws["part"])
        op(a2, w, s, ws["sorted"], ws["experts"], ntpp, tw, TOPK_GATE, slots,
           N, K, False, N // 16, c2, ws["ctr"], ws["c3"])


def warmup(device, N=512, max_tokens=None):
    """Reserve scratch and warm the selected released alignment implementation
    (and decode_gemm_v2's variants when VLLM_AMPERE_MARLIN_DECODE_V2 is set)."""
    from vllm_sm80.kernels.marlin_moe.extension import require_extension

    require_extension()
    if max_tokens is None:
        max_tokens = _scratch_tokens(N)
    device = torch.device(device)
    with torch.cuda.device(device):
        ws = _workspaces(device, N, max_tokens, create=True)
        ids = torch.zeros(max_tokens, TOPK_GATE, device=device, dtype=torch.int32)
        _align(ids, ws)
        if envs.VLLM_AMPERE_MARLIN_DECODE_V2 and N == _V2_N:
            _warm_v2(device, ws)


def warmup_from_worker(worker):
    if not envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
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
    if N == 2048:
        ranges = pp_multi_ranges()
        if not ranges:
            logger.info_once(
                "[ampere-marlin] whole-expert compiled decode at 4..8 rows only: "
                "moe_routed_gemm/marlin_multi_max_rows = %d is not within %d..%d.",
                multi_max_rows(), PP_MULTI_MIN, PP_MULTI_MAX,
            )
        else:
            logger.info_once(
                "[ampere-marlin] whole-expert compiled decode for multi-request "
                "rows %s (moe_routed_gemm/marlin_multi_max_rows) and at 4..8 "
                "rows, scratch %d rows.",
                ",".join("%d..%d" % r for r in ranges), _scratch_tokens(N),
            )
    capacity = _scratch_tokens(N)
    if N in _LATE_SCRATCH_N and _late_scratch_open:
        logger.info_once(
            "[ampere-marlin] decode scratch active after KV-cache sizing "
            "(N=%d, %d rows); profiling used the released kernels.", N, capacity,
        )
    if compiled_regime(capacity, N):
        warmup(device, N, capacity)
