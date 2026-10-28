# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE gate + routing on the BF16 tensor cores (sm_80), as two
launches that share their scratch: :func:`gate` (``sm80_gate_mma.py``: exact
BF16 products, FP32 sums per K chunk, FP64 across chunks, one FP32 rounding)
and :func:`route`: FP32 selection as the served route, weights in FP64
rounded once; up to :data:`EXACT_ROWS` rows the Gluon kernel
(``sm80_route_gluon.py``) recomputes the chosen experts' logits exactly,
above it the Triton kernel (``sm80_route_mma.py``) takes them from the gate's
logits. For decode rows with split K the gate leaves its FP64 partials and
the route completes them, so :func:`route` must consume the :class:`Pending`
of the :func:`gate` call that produced its logits.

Served configuration: hidden 5120, 384 experts top 6 (target) or 128 experts
top 3 (drafter), routed scaling 1.5, renormalised, gate_temp 1, image sentinel
ids [129264, 129269) or none, BF16 x of 1 to :data:`MAX_ROWS` rows
(:func:`serves`). A call of 2048 rows (a whole 2048-token prefill chunk)
takes the served gate and route (``sm80_gate_gemv`` / the FP32 GEMM, then
``sm80_route.route_fused``): the same FP32 selection rules on the FP32 GEMM's
logits, weights rounded in FP32, so one prefill may mix the two. Both routes
compile without FTZ and FP contraction, as ``sm80_route.py``, so equal FP32
logits select the same experts. Both kernels read the layer's gate weight
itself, held once in BF16 (``sm80_moe``).

CUDA graphs and the first eager prefill: :data:`ROUTER_WARMUP` compiles every
form of both kernels a served call can reach (one call per
:func:`launch_variant` in 1..MAX_ROWS) for every adopted layer shape, target
and drafter, before capture (registered by :func:`register_warmup` from
``sm80_moe._adopt_router_mma``), and logs the device bytes the BF16-held gate
weights free on the rank.
"""

import functools
import os
import weakref
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    triton_scalar_specialization_rep,
)
from vllm_sm80.models.deepseek_v41.ops import sm80_gate_mma as G
from vllm_sm80.models.deepseek_v41.ops import sm80_route_gluon as RG
from vllm_sm80.models.deepseek_v41.ops import sm80_route_mma as R
from vllm.triton_utils import triton

logger = init_logger(__name__)

HIDDEN = 5120
EXPERTS_TOP_K = {384: 6, 128: 3}
SCALE = 1.5
SENTINEL_LO = 129264
SENTINEL_HI = SENTINEL_LO + 5
MAX_ROWS = 2047
# Rows whose weights come from the exactly recomputed logits; above, from the
# gate's FP32 logits.
EXACT_ROWS = 64
# K slices of the decode route's recompute (5 x 1024, one program per row and
# slice).
_ROUTE_SPLIT = 5
# Prefill route warps: one row per single-warp program above _LARGE_ROWS.
_ROUTE_WARPS, _ROUTE_WARPS_LARGE, _LARGE_ROWS = 4, 1, 256
_KP_MAX = 8
# The served route's compile options (sm80_route.py): no FTZ, no FP
# contraction, so its FP32 selection scores are this route's.
_ROUTE_OPTIONS = ({} if os.environ.get("TRITON_INTERPRET", "0") == "1"
                  else {"enable_reflect_ftz": False, "enable_fp_fusion": False})
_SMS: dict = {}
_LOCKS: dict = {}


def supports(num_experts: int, hidden: int, top_k: int, renormalize: bool,
             scale: float, gate_temp: float, sentinel_lo: int) -> str | None:
    """None when this gate + route serves the layer, else the reason."""
    if EXPERTS_TOP_K.get(num_experts) != top_k or hidden != HIDDEN:
        return f"{num_experts} experts top {top_k}, hidden {hidden}"
    if not renormalize or scale != SCALE or gate_temp != 1.0:
        return f"renormalize {renormalize}, scale {scale}, gate_temp {gate_temp}"
    if sentinel_lo not in (0, SENTINEL_LO):
        return f"image sentinel {sentinel_lo}"
    return None


def serves(x: torch.Tensor) -> bool:
    """A call this path takes: CUDA sm_80, BF16 [1..MAX_ROWS, 5120], unit
    column stride, even row stride."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.shape[1] == HIDDEN
        and x.stride(1) == 1
        and x.stride(0) % 2 == 0
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    )


def _sms(device: torch.device) -> int:
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SMS[key]


def _locks(device: torch.device) -> torch.Tensor:
    """The per-device counters (zero between calls); allocated on the first,
    eager call, before any graph capture."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _LOCKS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoE gate counters must be allocated before graph capture")
        _LOCKS[key] = torch.zeros(G.LOCK_SLOTS, dtype=torch.int32, device=device)
    return _LOCKS[key]


@dataclass
class Pending:
    """One gate call's state for its :func:`route`."""

    x: torch.Tensor
    logits: torch.Tensor
    gpart: torch.Tensor
    rpart: torch.Tensor
    locks: torch.Tensor
    exact: bool
    gsplit: int
    gate_rows: int
    gcols: int


def gate(x: torch.Tensor, weight_bf16: torch.Tensor) -> tuple[torch.Tensor, Pending]:
    """FP32 logits [M, E] of ``x @ weight^T`` (complete unless the returned
    :class:`Pending` defers them to :func:`route`)."""
    M, K = x.shape
    E = weight_bf16.shape[0]
    assert K % _ROUTE_SPLIT == 0 and (K // _ROUTE_SPLIT) & (K // _ROUTE_SPLIT - 1) == 0, K
    sms = _sms(x.device)
    exact = M <= EXACT_ROWS
    ng = G.part_numel(M, E, K, sms)
    nr = RG.part_numel(M, _ROUTE_SPLIT, _KP_MAX) if exact else 0
    scratch = torch.empty(2 * (ng + nr) + M * E, dtype=torch.float32, device=x.device)
    gpart = scratch[: 2 * ng].view(torch.float64)
    rpart = scratch[2 * ng : 2 * (ng + nr)].view(torch.float64)
    logits = scratch[2 * (ng + nr) :].view(M, E)
    block_t, block_e, _, gsplit, tb, _, _ = G.config(M, E, K, sms)
    defer = exact and gsplit > 1
    locks = _locks(x.device)
    G.launch(x, weight_bf16, logits, gpart, locks, sms, defer)
    return logits, Pending(
        x=x, logits=logits, gpart=gpart, rpart=rpart, locks=locks, exact=exact,
        gsplit=gsplit if defer else 1, gate_rows=tb * block_t,
        gcols=triton.cdiv(E, block_e) * block_e,
    )


def route(
    p: Pending,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    top_k: int,
    sentinel_lo: int,
    row_live: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 weights and int32 expert ids [M, top_k] of the gate call ``p``
    (image tokens: ``input_ids`` in [sentinel_lo, sentinel_lo + 5) select with
    ``bias_vl``; none when ``sentinel_lo`` is 0). With ``row_live`` (the DSpark
    verification row mask: int8, at least M entries, read when the kernel
    runs) a row whose entry is 0 gets ids -1 and weights +0.0."""
    M, E = p.logits.shape
    if row_live is not None:
        assert (
            row_live.dtype == torch.int8 and row_live.dim() == 1
            and row_live.is_contiguous() and row_live.numel() >= M
        ), (row_live.dtype, row_live.shape)
    weights = torch.empty(M, top_k, dtype=torch.float32, device=p.x.device)
    ids = torch.empty(M, top_k, dtype=torch.int32, device=p.x.device)
    if input_ids is None or bias_vl is None or sentinel_lo <= 0:
        # No image tokens: the sentinel range is empty and the token reads
        # (of ``ids``' bytes) select nothing.
        tokens, lo, hi, bias_vl = ids, 0, 0, bias
    else:
        assert input_ids.shape == (M,) and input_ids.is_contiguous(), input_ids.shape
        tokens, lo, hi = input_ids, sentinel_lo, sentinel_lo + (SENTINEL_HI - SENTINEL_LO)
    kp = triton.next_power_of_2(top_k)
    assert kp <= _KP_MAX, top_k
    if p.exact:
        RG.launch(p.x, weight_bf16, bias, bias_vl, tokens, weights, ids, p.rpart, p.locks, p.gpart, p.logits,
                  p.gate_rows, p.gsplit, p.gcols, _ROUTE_SPLIT, lo, hi, SCALE, 1e-20, _ROUTE_OPTIONS,
                  row_live=row_live)
    else:
        R._route_kernel[(M,)](
            p.logits, p.logits.stride(0), bias, bias_vl, tokens, weights, ids,
            ids if row_live is None else row_live, E, lo, hi, SCALE, 1e-20,
            K=top_k, KP=kp, BLOCK_E=triton.next_power_of_2(E), ROW_MASK=row_live is not None,
            num_warps=_ROUTE_WARPS_LARGE if M > _LARGE_ROWS else _ROUTE_WARPS, **_ROUTE_OPTIONS)
    return weights, ids


def gate_route(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor | None,
    top_k: int,
    input_ids: torch.Tensor | None = None,
    sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`gate` then :func:`route` of one call; ``weight`` [E, 5120] BF16
    (or FP32 holding BF16 values)."""
    w16 = (weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)).contiguous()
    _, pending = gate(x, w16)
    return route(pending, w16, bias, bias_vl, input_ids, top_k, sentinel_lo)


def launch_variant(rows: int, experts: int, sms: int) -> tuple:
    """The part of the Triton cache keys of a call's two launches that depends
    on its row count (``sms`` SMs): the gate's tiling, split K, DEFER, warps and
    stages (:func:`sm80_gate_mma.config`; its ``M`` is not specialised), and
    the route: the Gluon kernel with its GSPLIT, GCOLS and the class of its
    ``gate_rows`` argument (Triton specialises an integer on the value 1 and
    on divisibility by 16) up to :data:`EXACT_ROWS` rows, else the Triton
    kernel's warps. Every other launch argument is a layer tensor, a constant,
    a fresh output or a scratch view at a 16-byte aligned offset."""
    block_t, block_e, block_k, split, tb, warps, stages = G.config(rows, experts, HIDDEN, sms)
    defer = rows <= EXACT_ROWS and split > 1
    gate_key = (block_t, block_e, block_k, split, defer, warps, stages)
    if rows <= EXACT_ROWS:
        gcols = triton.cdiv(experts, block_e) * block_e
        route_key = ("gluon", split if defer else 1, gcols, triton_scalar_specialization_rep(tb * block_t))
    else:
        route_key = ("triton", _ROUTE_WARPS_LARGE if rows > _LARGE_ROWS else _ROUTE_WARPS)
    return gate_key, route_key


@functools.cache
def warmup_rows(experts: int, sms: int) -> tuple[int, ...]:
    """The smallest row count of every :func:`launch_variant` in 1..MAX_ROWS."""
    first: dict = {}
    for rows in range(1, MAX_ROWS + 1):
        first.setdefault(launch_variant(rows, experts, sms), rows)
    return tuple(sorted(first.values()))


# The routers whose layers run this gate and route, the target's and the
# drafter's: the drafter loads outside the runner's warmup registry, so the
# one registration (made while the target loads) reads the set at warmup time.
_ADOPTED: "weakref.WeakSet[torch.nn.Module]" = weakref.WeakSet()


def register_warmup(router: torch.nn.Module) -> None:
    """Compile this path's kernels for ``router`` (its ``mma_gate`` adopted)
    in the boot warmup."""
    _ADOPTED.add(router)
    ROUTER_WARMUP.register_warmup()


def _sm80(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 0)


def _tensor_class(t: torch.Tensor | None) -> tuple | None:
    """A tensor argument's part of a Triton cache key."""
    return None if t is None else (t.dtype, t.data_ptr() % 16 == 0)


def held_gate_bytes(routers) -> tuple[int, int, int]:
    """(gates, bytes held, bytes freed) of the adopted routers' gate weights:
    each is held once, in BF16, where the gate held its FP32 weight and a
    BF16 copy of it (the freed bytes: 4 a value)."""
    weights = {id(r.mma_gate): r.mma_gate.weight for r in routers}.values()
    held = sum(w.numel() * w.element_size() for w in weights)
    return len(weights), held, sum(w.numel() * 4 for w in weights)


class RouterWarmup(VllmJitKernel["RouterWarmup.Key"]):
    """Boot-time compile of :func:`gate` and :func:`route` for each adopted
    layer shape at :func:`warmup_rows` (one call per :func:`launch_variant`
    reachable on the device: every graph capture size and every eager row
    count share the cache entries of one of them), through those entry points
    with the layer's gate weight, biases and image sentinel, as the layers
    call them (``sm80_moe``), and the runner's int32 token ids."""

    @dataclass(frozen=True)
    class Key:
        rows: int
        experts: int
        top_k: int
        sentinel_lo: int
        biases: tuple
        # The router marked for the DSpark verification row mask (its route
        # launches compile with ROW_MASK).
        row_mask: bool
        router: object = field(compare=False, hash=False, repr=False)

    def get_warmup_keys(self) -> list["RouterWarmup.Key"]:
        keys: dict = {}
        routers = list(_ADOPTED)
        if routers:
            gates, held, freed = held_gate_bytes(routers)
            logger.info_once(
                "DeepSeek-V4.1 sm_80: %d MoE gate weights held once, in BF16 (%d bytes); "
                "%d bytes of FP32 gate weights freed on this rank "
                "(VLLM_DEEPSEEK_V41_ROUTER_MMA=1)",
                gates, held, freed,
            )
        for router in routers:
            weight = router.mma_gate.weight
            if not _sm80(weight.device):
                continue
            experts = weight.shape[0]
            biases = (_tensor_class(router.e_score_correction_bias), _tensor_class(router.bias_vl))
            for rows in warmup_rows(experts, _sms(weight.device)):
                key = self.Key(rows, experts, router.top_k, router.image_sentinel_lo, biases,
                               bool(getattr(router, "row_mask", False)), router)
                keys.setdefault(key, None)
        return list(keys)

    def compile(self, compile_key: "RouterWarmup.Key") -> None:
        router, rows = compile_key.router, compile_key.rows
        w16 = router.mma_gate.weight
        dev = w16.device
        x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        input_ids = torch.zeros(rows, dtype=torch.int32, device=dev)
        row_live = torch.ones(rows, dtype=torch.int8, device=dev) if compile_key.row_mask else None
        _, pending = gate(x, w16)
        route(pending, w16, router.e_score_correction_bias, router.bias_vl, input_ids,
              router.top_k, router.image_sentinel_lo, row_live=row_live)
        torch.cuda.synchronize(dev)


ROUTER_WARMUP = RouterWarmup()
