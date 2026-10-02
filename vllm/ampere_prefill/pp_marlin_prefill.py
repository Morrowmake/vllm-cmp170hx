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
(``layer.moe_sum``), the same workspaces.  Split-cover scheduling can change
stream-K boundaries and reduction order: measured PP outputs differ bitwise
from unsplit ``fused_marlin_moe``.  At N=512
split-cover outputs also differ bitwise (measured); error against an fp64
reference equals the incumbent's (1.00x mean and max on captured TP4 calls).  Under
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

COMPILED TILES.  With VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED (default on)
and the optional library's ``prefill_tile_gemm`` present, every list GEMM
runs through that op (the same Marlin template, its own instantiations) with
the per-GEMM tile table ``TILE_TABLES[N]``: at N=512 w13 on the 64/48/32-row
lists uses the (64,512) all-warps-along-N tile and w2 the (64,256) tile at
128 threads, 2 CTAs per SM; the 16-row list keeps Marlin's own choice.
Measured against this path's released kernels on captured 74-SM TP4 calls:
mean and max error vs fp64 within 1.10x / 1.25x, bitwise run to run.
At N=2048 (PP whole experts, VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED, default
on) both GEMMs use the (64,512) tile on the 64/48/32-row lists, and every list
GEMM runs the op's optimistic dequant: a fast_dequant kernel (one HFMA2 per
bf16x2, exact while every scale of a tile is <= 2^-5) that lists the tiles it
could not compute exactly in a per-GEMM ``redo`` area, then the regular kernel
on just those tiles (it exits at once when none are listed).  The redo area
(one row per list GEMM, its count zeroed on the device per call) is allocated
in ``warmup``; without it (e.g. a larger call during a capture) the regular
kernels alone run.  The op's fp32 reduction scratch is allocated in
``warmup`` too; without the library, the op or the scratch the released
kernels run with ``THREAD_CFG``.

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

# Compiled tiles (prefill_tile_gemm): N -> (switch, {"w13": table, "w2": table}),
# tables as THREAD_CFG.
TILE_TABLES = {
    N_GATE_TP4: ("VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED", {
        "w13": {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)},
        "w2": {64: (64, 256, 2), 48: (64, 256, 2), 32: (64, 256, 2)},
    }),
    N_GATE: ("VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED", {
        "w13": {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)},
        "w2": {64: (64, 512, 1), 48: (64, 512, 1), 32: (64, 512, 1)},
    }),
}
# Widths whose list GEMMs run the optimistic fast dequant with a redo area.
REDO_N = (N_GATE,)
# Redo areas per device: [one row per list GEMM (w13 and w2 per block size),
# count + listed tiles].
_TILE_REDO: dict = {}
_SMS: dict = {}
MAX_BLOCKS_PER_SM = 4
# fp32 reduction scratch of prefill_tile_gemm per device: Marlin needs at most
# sms * 4 blocks * moe_block_size (64) * thread_n floats; sized for thread_n 512.
_TILE_SCRATCH: dict = {}
_TILE_MAX_N = 512


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


def _tile_scratch(device: torch.device, create: bool):
    key = str(device)
    buf = _TILE_SCRATCH.get(key)
    if buf is None and create:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        buf = torch.empty(sms * 4 * 64 * _TILE_MAX_N, dtype=torch.float32,
                          device=device)
        _TILE_SCRATCH[key] = buf
    return buf


def _sms(device: torch.device) -> int:
    key = str(device)
    n = _SMS.get(key)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _SMS[key] = n
    return n


def _redo_len(rows: int, E: int, sms: int) -> int:
    """One redo row: the count plus one entry per listed slice; at most every
    m-n tile (16-row blocks, thread_n >= 64: <= 64 n-tiles at N, K <= 4096)
    plus one per CTA.  The op checks its own bound against the row length."""
    blocks = (rows + 16 * E) // 16 + 1
    return 1 + blocks * 64 + sms * MAX_BLOCKS_PER_SM


def _tile_redo(device: torch.device, rows: int, E: int, create: bool):
    """The redo area for calls of up to ``rows`` routed rows, or None when
    it would have to be (re)allocated and ``create`` is False."""
    from vllm.ampere_prefill.moe_split_align import SIZES

    key = str(device)
    need = _redo_len(rows, E, _sms(device))
    buf = _TILE_REDO.get(key)
    if buf is not None and buf.size(1) >= need:
        return buf
    if not create:
        return None
    if buf is not None:
        _RETIRED.append(buf)      # a captured graph may still point at it
    buf = torch.empty(2 * len(SIZES), need, dtype=torch.int32, device=device)
    _TILE_REDO[key] = buf
    return buf


def compiled_tiles(N: int, device: torch.device, create: bool):
    """(op, tables, c_tmp, None) when the compiled tiles serve width N, else
    (None, None, None, reason); reason None means simply not requested."""
    entry = TILE_TABLES.get(N)
    if entry is None:
        return None, None, None, None
    flag, tables = entry
    if not getattr(envs, flag):
        return None, None, None, None
    if torch.device(device).type != "cuda":
        return None, None, None, f"{flag} is set but {device} is not a CUDA device"
    from vllm.ampere_marlin import prefill_tile_op

    op, why = prefill_tile_op()
    if op is None:
        return None, None, None, f"{flag} is set but {why}"
    c_tmp = _tile_scratch(device, create)
    if c_tmp is None:
        return None, None, None, (f"{flag} is set but its scratch is not "
                                  "allocated during a graph capture")
    return op, tables, c_tmp, None


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
        "for M >= %d.", "PP" if w2.size(1) * 16 == N_GATE else "TP4",
        flag, min_tokens)
    N = w2.size(1) * 16
    capturing = torch.cuda.is_current_stream_capturing()
    op, tables, c_tmp, why_not = compiled_tiles(
        N, hidden_states.device, create=not capturing)
    if op is not None:
        logger.info_once(
            "Marlin MoE prefill compiled tiles active (%s, N=%d): w13 %s, w2 %s "
            "by block-list size; other lists Marlin's own choice.",
            TILE_TABLES[N][0], N, str(tables["w13"]), str(tables["w2"]))
        redo = None
        if N in REDO_N:
            redo = _tile_redo(hidden_states.device, rows, w1.size(0),
                              create=not capturing)
            if redo is None:
                logger.info_once(
                    "Marlin MoE prefill fast dequant: no redo area for %d rows "
                    "during a graph capture; the regular kernels run.", rows)
            else:
                logger.info_once(
                    "Marlin MoE prefill fast dequant active (%s, N=%d): one-HFMA2 "
                    "dequant, tiles with a scale above 2^-5 recomputed by the "
                    "regular kernel.", TILE_TABLES[N][0], N)
        compiled = (op, tables, c_tmp, redo)
    else:
        if why_not is not None:
            logger.info_once("%s; using the released Marlin kernels.", why_not)
        compiled = None
    run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, workspace13, workspace2, buf, compiled=compiled)
    return True


def run(layer, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, workspace13, workspace2, buf, thread_cfg=None,
        compiled=None):
    """The split path. Workspaces as ``MarlinExperts.apply`` passes them to
    ``fused_marlin_moe`` (workspace2 holds w13's and w2's outputs, workspace13
    the activation). ``compiled`` = (prefill_tile_gemm, {"w13", "w2"} tables,
    c_tmp, redo or None) runs the list GEMMs on the compiled tiles instead;
    with a redo area every list GEMM runs the optimistic fast dequant."""
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
    if compiled is not None:
        op, tables, c_tmp, redo = compiled
        nl = len(lists)
        if redo is not None:
            redo[:, 0].zero_()       # per-GEMM tile counts (kernel.h)
        for i, (bs, sorted_ids, expert_ids, ntpp) in enumerate(lists):
            op(hidden_states, c1, w1, None, layer.w1_scale, None, None, None,
               workspace, sorted_ids, expert_ids, ntpp, topk_weights, bs, topk,
               False, qt.id, M, 2 * N, K, False, True, False,
               *_thread_cfg(bs, 2 * N, K, tables["w13"]), c_tmp,
               None if redo is None else redo[i])
        layer.activation(activation, c2, c1, topk_ids=topk_ids, expert_map=None)
        for i, (bs, sorted_ids, expert_ids, ntpp) in enumerate(lists):
            op(c2, c3, w2, None, layer.w2_scale, None, None, None,
               workspace, sorted_ids, expert_ids, ntpp, topk_weights, bs, 1,
               True, qt.id, rows, K, N, False, True, False,
               *_thread_cfg(bs, K, N, tables["w2"]), c_tmp,
               None if redo is None else redo[nl + i])
        layer.moe_sum(c3.view(M, topk, K), output, topk_ids, None)
        return
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


def warmup(device, num_experts: int, max_tokens: int, topk: int,
           N: int | None = None) -> None:
    """Allocate the list buffers for the current stream and compile the
    split-alignment kernels; for width ``N`` with compiled tiles, also load the
    op and allocate its scratch.  Must run before any CUDA-graph capture."""
    from vllm.ampere_prefill.moe_split_align import split_align

    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return
    with torch.cuda.device(device):
        buf = _buffers(device, num_experts, max_tokens * topk, create=True)
        ids = torch.zeros(1, topk, device=device, dtype=torch.int32)
        split_align(ids, num_experts, buf)
        if N is not None:
            op, _, _, why = compiled_tiles(N, device, create=True)
            if why is not None:
                logger.info_once("%s; using the released Marlin kernels.", why)
            if op is not None and N in REDO_N:
                _tile_redo(device, max_tokens * topk, num_experts, create=True)
        torch.cuda.synchronize(device)


def _warmup_capacity(N: int, max_tokens: int) -> int:
    """List capacity (rows of tokens) to allocate for width N; 0 = none."""
    return max_tokens if N in enabled_shapes() else 0


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
    tp = worker.vllm_config.parallel_config.tensor_parallel_size
    N = int(getattr(cfg, "moe_intermediate_size", 0)) // tp
    max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    capacity = _warmup_capacity(N, max_tokens)
    if capacity:
        warmup(device, E, capacity, topk, N)
    return E if capacity else 0
