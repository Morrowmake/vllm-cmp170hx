# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE routing on sm_80, as the reference ``Gate`` computes it.

Upstream routes DSv4.1 with an FP32-weight-less gate (on sm_80 GateLinear
stores the FP16 checkpoint weight in BF16 and multiplies in BF16) and fused
kernels (``dsv4_topk``, ``topk_hash_softplus_sqrt``) whose softplus,
normalisation and scale differ from the reference in rounding (fast
``__logf``/``__expf``, ``log(1 + exp)``, ``w * (scale / sum)``, no ``+1e-20``).
:func:`adopt_moe` keeps the upstream modules, parameter names and EXL3
experts, and changes two things before the weights load:

- the gate weight is kept in FP32 (the FP16 checkpoint values exactly) and
  the gate (:class:`Sm80DeepseekV41Gate`) computes ``x.float() @ weight.T``
  in FP32; with ``VLLM_DEEPSEEK_V41_GATE_GEMV`` its decode-sized calls take
  the BF16 input and compute the logits in FP64, rounded once to FP32, in one
  kernel (``ops/sm80_gate_gemv.py``);
- the runner's router is :class:`Sm80DeepseekV41Router`, whose
  :func:`route` is the reference ``Gate.forward``: ``sqrt(softplus(logits /
  gate_temp))``, the correction bias (``bias_vl`` for image tokens) used only
  to select the top-k, the weights gathered from the unbiased scores,
  divided by their sum ``+ 1e-20`` and multiplied by the routed scaling
  factor, all in FP32 torch ops (graph-safe, no host synchronisation).
  With ``VLLM_DEEPSEEK_V41_FUSED_ROUTE`` the same arithmetic after the gate
  GEMM runs as one kernel (``ops/sm80_route.py``, the torch ops' bytes).
  With ``VLLM_DEEPSEEK_V41_ROUTER_MMA`` the gate and the routing of calls of
  1 to 2047 BF16 rows run on the BF16 tensor cores
  (``ops/sm80_router_mma.py``: FP64 sums rounded once, closer to the FP64
  routing weights than the FP32 GEMM, not its bytes).

With ``VLLM_DEEPSEEK_V41_ROUTER_MMA`` the gate weight is held once, in BF16:
the checkpoint's gate weights are BF16-valued (a value BF16 does not hold
exactly is refused as it loads), so the BF16 weight holds the FP32 weight's
values and every other reader widens it exactly: the gate GEMV to FP64 in
the kernel, the FP32 GEMM of the calls the tensor cores do not serve to FP32
for the call (:meth:`Sm80DeepseekV41Gate._fp32_fallback`). Their logits are
the bytes of the FP32-held weight's.

With ``VLLM_DSPARK_ADAPTIVE_DEPTH`` (the DSpark verification row mask,
``vllm_sm80/spec_decode/dspark_row_mask.py``) the routers of the target's
layers (:func:`adopt_row_mask`; never the drafter's) give every row whose
``row_live`` entry is 0 expert id -1 and weight 0: the EXL3 MoE kernels skip
such a route on every rank, as they skip a route to another rank's expert,
so the row takes no routed expert. Every other row's routing is unchanged,
and the mask is read in place (CUDA graphs see each step's mask).

The experts and the SwiGLU clamp run in the EXL3 MoE kernels
(vllm_sm80/kernels/exl3/quant.py, ``swiglu_limit``). With
``VLLM_DEEPSEEK_V41_MOE_PREFILL_TC`` the experts may take the tensor-core
prefill operator ``moe_out_prefill_v2`` for prefill calls (its gate decides
when the weights have loaded).
"""

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm_sm80 import capture
from vllm_sm80.kernels.exl3.quant import set_moe_prefill_v2
from vllm.models.deepseek_v41.nvidia import model as upstream_model
from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv, sm80_route, sm80_router_mma
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side
from vllm_sm80.models.deepseek_v41.sm80_attention import (
    large_round_once_all_reduce,
    prefill_all_reduce,
)
from vllm_sm80.spec_decode import dspark_row_mask

logger = init_logger(__name__)

# Image-span tokens carry five consecutive sentinel ids from this one on
# (common/mm_preprocess.py), the ids the upstream routers test for bias_vl.
IMAGE_SENTINEL_IDS = 5
# GateLinear's specialised GEMM tiers; none may run an FP32 gate weight with
# BF16 input, so the adopted gate takes the FP32 fallback.
_GATE_TIERS = (
    "allow_ll_bf16_gemm",
    "allow_fp32_router_gemm",
    "allow_ll_fp32w_gemm",
    "allow_bf16x3_router_gemm",
    "allow_cublas_router_gemm",
    "allow_rocm_bf16x3_router_gemm",
)


def image_token_mask(input_ids: torch.Tensor | None, sentinel_lo: int):
    """Tokens inside an image span, or None for a text-only model."""
    if input_ids is None or sentinel_lo <= 0:
        return None
    return (input_ids >= sentinel_lo) & (input_ids < sentinel_lo + IMAGE_SENTINEL_IDS)


def route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    *,
    bias_vl: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    gate_temp: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference ``Gate.forward`` after its GEMM, for FP32 ``logits``
    [tokens, experts]: FP32 weights and int64 expert ids [tokens, top_k]."""
    scores = logits if gate_temp == 1.0 else logits / gate_temp
    scores = F.softplus(scores).sqrt()
    if image_mask is not None and bias_vl is not None:
        bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
    indices = (scores + bias).topk(top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    if renormalize and top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return weights, indices


class Sm80DeepseekV41Router(FusedTopKBiasRouter):
    """The upstream DSv4.1 router with the reference routing arithmetic."""

    gate_temp = 1.0
    # VLLM_DEEPSEEK_V41_FUSED_ROUTE: route() as one kernel (ops/sm80_route.py).
    fused_route = False
    # VLLM_DEEPSEEK_V41_ROUTER_MMA: the adopted gate whose pending call this router routes.
    mma_gate = None
    # VLLM_DSPARK_ADAPTIVE_DEPTH: a target layer's router (adopt_row_mask).
    row_mask = False

    def _row_live(self, tokens: int, *, is_cuda: bool) -> torch.Tensor | None:
        """The row mask's per-row flags (int8, 0: no routed expert) for a call
        of ``tokens`` rows, or None when this router does not mask: not a
        target layer's, or no mask state yet (nothing can have masked a row)."""
        if not self.row_mask:
            return None
        state = dspark_row_mask.get_row_mask_state()
        if state is None:
            if is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "VLLM_DSPARK_ADAPTIVE_DEPTH: create the row mask state before CUDA "
                    "graph capture (a graph captured without it never masks)"
                )
            return None
        if tokens > state.row_live.numel():
            raise ValueError(
                f"VLLM_DSPARK_ADAPTIVE_DEPTH: {tokens} rows exceed the row mask's "
                f"{state.row_live.numel()}"
            )
        return state.row_live

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if router_logits.dtype != torch.float32:
            raise ValueError("DeepSeek-V4.1 routing needs FP32 router logits")
        row_live = self._row_live(router_logits.shape[0], is_cuda=router_logits.is_cuda)
        gate = self.mma_gate
        if gate is not None and gate.pending is not None:
            pending, gate.pending = gate.pending, None
            if pending.logits is not router_logits:
                raise RuntimeError("VLLM_DEEPSEEK_V41_ROUTER_MMA: router logits are not the gate call's")
            weights, ids = sm80_router_mma.route(
                pending,
                gate.weight,
                self.e_score_correction_bias,
                self.bias_vl,
                input_ids,
                self.top_k,
                self.image_sentinel_lo,
                row_live=row_live,
            )
            if indices_type is None or indices_type == torch.int32:
                return weights, ids
            return weights, ids.to(indices_type)
        if self.fused_route and router_logits.is_cuda:
            weights, ids = sm80_route.route_fused(
                router_logits,
                self.e_score_correction_bias,
                self.top_k,
                bias_vl=self.bias_vl,
                input_ids=input_ids,
                sentinel_lo=self.image_sentinel_lo,
                sentinel_hi=self.image_sentinel_lo + IMAGE_SENTINEL_IDS,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
                row_live=row_live,
            )
            if indices_type is None or indices_type == torch.int32:
                return weights, ids
            return weights, ids.to(indices_type)
        weights, ids = route(
            router_logits,
            self.e_score_correction_bias,
            self.top_k,
            bias_vl=self.bias_vl,
            image_mask=image_token_mask(input_ids, self.image_sentinel_lo),
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            gate_temp=self.gate_temp,
        )
        ids = ids.to(torch.int32 if indices_type is None else indices_type)
        if row_live is not None:
            # In place on this call's own outputs; live rows keep their bytes.
            masked = (row_live[: ids.shape[0]] == 0).unsqueeze(1)
            weights.masked_fill_(masked, 0.0)
            ids.masked_fill_(masked, -1)
        return weights, ids


class Sm80DeepseekV41Gate(GateLinear):
    """The adopted gate: FP32 weight, FP32 logits. With ``gemv``
    (VLLM_DEEPSEEK_V41_GATE_GEMV) a call that ``sm80_gate_gemv.serves`` (CUDA,
    BF16, at most ``sm80_gate_gemv.MAX_ROWS`` rows) runs ``gate_gemv``; every
    other call is ``GateLinear.forward`` (its FP32 fallback). With ``mma``
    (VLLM_DEEPSEEK_V41_ROUTER_MMA) the weight is held in BF16 (its FP32
    values exactly) and the calls neither kernel takes run
    :meth:`_fp32_fallback`."""

    # VLLM_DEEPSEEK_V41_GATE_GEMV: decode-sized calls run gate_gemv.
    gemv = False

    # VLLM_DEEPSEEK_V41_ROUTER_MMA: calls that sm80_router_mma.serves run its gate; the
    # router (``mma_gate``) consumes ``pending``.
    mma = False
    pending = None

    def forward(self, x: torch.Tensor):
        if self.mma and self.bias is None and sm80_router_mma.serves(x):
            if self.pending is not None:
                # Only the router consumes a call's Pending; a decode call's
                # logits stay unwritten until it does.
                raise RuntimeError(
                    "VLLM_DEEPSEEK_V41_ROUTER_MMA: the previous gate call was not "
                    "routed: the expert method reads the router logits itself "
                    "(a monolithic method), which the decode gate leaves unwritten"
                )
            logits, self.pending = sm80_router_mma.gate(x, self.weight)
            return self._return(logits)
        if (
            self.gemv
            and self.bias is None
            and sm80_gate_gemv.serves(x, self.weight)
        ):
            return self._return(sm80_gate_gemv.gate_gemv(x, self.weight))
        if self.weight.dtype != torch.float32:
            return self._return(self._fp32_fallback(x))
        return super().forward(x)

    def _fp32_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """``GateLinear.forward``'s FP32 fallback (``x`` in FP32, then
        ``UnquantizedLinearMethod.apply``) on the FP32 values of the BF16-held
        weight, widened for this call: the widening is exact, so the GEMM
        reads the bytes of the FP32-held weight."""
        x = x.to(torch.float32)
        weight = self.weight.float()
        if envs.VLLM_BATCH_INVARIANT and (
            current_platform.is_cuda_alike() or current_platform.is_xpu()
        ):
            return linear_batch_invariant(x, weight, None)
        return self.quant_method._gemm_impl(self, x, weight, None)


def _hold_gate_weight(gate: nn.Module, dtype: torch.dtype) -> None:
    """Hold the gate weight in ``dtype``, route the gate's forward to its FP32
    fallback, and make an upstream ``GateLinear`` a
    :class:`Sm80DeepseekV41Gate` (``gemv`` off)."""
    if type(gate) is GateLinear:
        gate.__class__ = Sm80DeepseekV41Gate
    weight = gate.weight
    if weight.dtype != dtype:
        held = nn.Parameter(torch.empty_like(weight, dtype=dtype), requires_grad=False)
        for key, value in vars(weight).items():
            setattr(held, key, value)
        gate.weight = held
    for tier in _GATE_TIERS:
        if getattr(gate, tier, False):
            setattr(gate, tier, False)
    if gate.out_dtype != torch.float32:
        raise ValueError("DeepSeek-V4.1 gates produce FP32 logits")


def _fp32_gate_weight(gate: nn.Module) -> None:
    """:func:`_hold_gate_weight` in FP32, so the checkpoint's FP16 values
    load exactly."""
    _hold_gate_weight(gate, torch.float32)


def _bf16_valued_loader(loader):
    """``loader`` for the BF16-held gate weight, refusing a checkpoint tensor
    whose values BF16 does not hold exactly (the copy would round them)."""

    def load(param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
        if loaded_weight.dtype != torch.bfloat16 and not torch.equal(
            loaded_weight.to(torch.bfloat16).float(), loaded_weight.float()
        ):
            raise ValueError("VLLM_DEEPSEEK_V41_ROUTER_MMA: the gate weight is not BF16-valued")
        return loader(param, loaded_weight, *args, **kwargs)

    return load


def _adopt_router_mma(ffn: nn.Module, router: "Sm80DeepseekV41Router") -> None:
    """VLLM_DEEPSEEK_V41_ROUTER_MMA: the layer's gate and routing run
    ``ops/sm80_router_mma.py`` for every call it serves; the calls it does not
    serve take the gate GEMV or the FP32 gate and the fused route, so both
    switches are required. The gate weight is held in BF16 (by
    :func:`adopt_moe`) and loads only BF16-valued."""
    missing = [
        name
        for name in ("VLLM_DEEPSEEK_V41_FUSED_ROUTE", "VLLM_DEEPSEEK_V41_GATE_GEMV")
        if not getattr(envs, name)
    ]
    if missing:
        raise ValueError(
            "VLLM_DEEPSEEK_V41_ROUTER_MMA needs "
            + " and ".join(f"{name}=1" for name in missing)
            + " (the path of the calls it does not serve)"
        )
    gate = ffn.gate
    if not isinstance(gate, Sm80DeepseekV41Gate) or gate.bias is not None:
        raise TypeError(f"VLLM_DEEPSEEK_V41_ROUTER_MMA: unexpected gate {type(gate).__name__}")
    if not isinstance(gate.quant_method, UnquantizedLinearMethod):
        raise TypeError(
            "VLLM_DEEPSEEK_V41_ROUTER_MMA: unexpected gate method "
            f"{type(gate.quant_method).__name__}"
        )
    if gate.weight.dtype != torch.bfloat16:
        raise TypeError(f"VLLM_DEEPSEEK_V41_ROUTER_MMA: the gate weight is held in {gate.weight.dtype}")
    moe_config = ffn.experts.moe_config
    if moe_config.dp_size > 1 or moe_config.pcp_size > 1 or moe_config.is_sequence_parallel:
        # The route consumes the gate call's own logits tensor and scratch.
        raise ValueError("VLLM_DEEPSEEK_V41_ROUTER_MMA: the MoE runner dispatches the router logits")
    reason = sm80_router_mma.supports(
        router.global_num_experts, gate.weight.shape[1], router.top_k, router.renormalize,
        router.routed_scaling_factor, router.gate_temp, router.image_sentinel_lo,
    )
    if reason is not None:
        raise ValueError(f"VLLM_DEEPSEEK_V41_ROUTER_MMA: {reason}")
    if capture.captures("hybrid"):
        raise ValueError("VLLM_DEEPSEEK_V41_ROUTER_MMA: the hybrid capture records the FP32 gate GEMM; capture with it off")
    weight = gate.weight
    weight.weight_loader = _bf16_valued_loader(
        getattr(weight, "weight_loader", default_weight_loader)
    )
    gate.mma = True
    router.mma_gate = gate
    sm80_router_mma.register_warmup(router)
    logger.info_once(
        "DeepSeek-V4.1 sm_80: MoE gate and routing on the BF16 tensor cores for calls "
        "of 1 to %d rows (VLLM_DEEPSEEK_V41_ROUTER_MMA=1)",
        sm80_router_mma.MAX_ROWS,
        scope="global",
    )


def adopt_row_mask(layers) -> None:
    """With ``VLLM_DSPARK_ADAPTIVE_DEPTH``, mark the routers of these decoder
    layers (the target model's, after :func:`adopt_moe`) to apply the
    verification row mask. The DSpark drafter's layers are never given: its
    rows are not the target's verify rows."""
    if not dspark_row_mask.enabled():
        return
    for layer in layers:
        router = layer.ffn.experts.router
        if type(router) is not Sm80DeepseekV41Router:
            raise TypeError(f"Unexpected DeepSeek-V4.1 router: {type(router).__name__}")
        router.row_mask = True
    logger.info_once(
        "DeepSeek-V4.1 sm_80: the target's MoE routing gives verify rows the "
        "DSpark row mask holds masked no routed expert (VLLM_DSPARK_ADAPTIVE_DEPTH)",
        scope="global",
    )


def adopt_moe(ffn: nn.Module, config) -> None:
    """Give an upstream-built DeepseekV4MoE the reference gate and router
    (before its weights load)."""
    if ffn.use_native_mega_moe:
        raise ValueError("DeepSeek-V4.1 on sm_80 runs the modular EXL3 MoE")
    if ffn.gate.tid2eid is not None or ffn.gate.e_score_correction_bias is None:
        raise ValueError("DeepSeek-V4.1 routes every layer with noaux_tc")
    if ffn.scoring_func != "sqrtsoftplus":
        raise ValueError(f"Unsupported DeepSeek-V4.1 scoring: {ffn.scoring_func}")
    if envs.VLLM_DEEPSEEK_V41_ROUTER_MMA:
        _hold_gate_weight(ffn.gate, torch.bfloat16)
    else:
        _fp32_gate_weight(ffn.gate)
    router = ffn.experts.router
    if type(router) is not FusedTopKBiasRouter or router.num_fused_shared_experts:
        raise TypeError(f"Unexpected DeepSeek-V4.1 router: {type(router).__name__}")
    router.__class__ = Sm80DeepseekV41Router
    router.gate_temp = float(getattr(config, "gate_temp", 1.0))
    if envs.VLLM_DEEPSEEK_V41_FUSED_ROUTE:
        reason = sm80_route.supports(
            router.global_num_experts, router.top_k, router.gate_temp
        )
        if reason is not None:
            raise ValueError(f"VLLM_DEEPSEEK_V41_FUSED_ROUTE: {reason}")
        router.fused_route = True
        logger.info_once(
            "DeepSeek-V4.1 sm_80: MoE routing after the gate GEMM runs as one "
            "kernel (VLLM_DEEPSEEK_V41_FUSED_ROUTE=1)",
            scope="global",
        )
    if envs.VLLM_DEEPSEEK_V41_ROUTER_MMA:
        _adopt_router_mma(ffn, router)
    if envs.VLLM_DEEPSEEK_V41_GATE_GEMV:
        gate = ffn.gate
        if not isinstance(gate, Sm80DeepseekV41Gate):
            raise TypeError(
                "VLLM_DEEPSEEK_V41_GATE_GEMV: unexpected DeepSeek-V4.1 gate "
                f"{type(gate).__name__}"
            )
        reason = sm80_gate_gemv.supports(*gate.weight.shape)
        if reason is not None:
            raise ValueError(f"VLLM_DEEPSEEK_V41_GATE_GEMV: {reason}")
        if capture.captures("hybrid"):
            # The hybrid capture records the gate through GateLinear.forward,
            # which the kernel bypasses: its gate and route cases would vanish.
            raise ValueError(
                "VLLM_DEEPSEEK_V41_GATE_GEMV: the hybrid capture records the FP32 "
                "gate GEMM, not this kernel; capture with the flag off"
            )
        gate.gemv = True
        logger.info_once(
            "DeepSeek-V4.1 sm_80: MoE gate calls of at most %d rows compute the "
            "logits in FP64 in one kernel (VLLM_DEEPSEEK_V41_GATE_GEMV=1)",
            sm80_gate_gemv.MAX_ROWS,
            scope="global",
        )
    if envs.VLLM_DEEPSEEK_V41_MOE_PREFILL_TC:
        reason = set_moe_prefill_v2(ffn.experts.routed_experts)
        if reason is not None:
            logger.info_once(
                "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC is set but the gate is closed: "
                "%s; these MoE layers keep their kernels in prefill steps.",
                reason,
            )
    if (
        envs.VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE
        or envs.VLLM_SM80_PREFILL_REDUCE_OVERLAP
        or envs.VLLM_SM80_ALLREDUCE
    ):
        _adopt_layer_all_reduce(
            ffn, round_once=envs.VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE
        )


def moe_all_reduce(out: torch.Tensor) -> torch.Tensor:
    """The MoE output's TP all-reduce: rounded once per element when the fast
    paths do not take ``out`` whole (``large_round_once_all_reduce`` of
    sm80_attention); otherwise the runner's own ``tensor_model_parallel_all_reduce``."""
    reduced = large_round_once_all_reduce(out)
    if reduced is not None:
        return reduced
    return tensor_model_parallel_all_reduce(out)


class Sm80DeepseekV41MoE(upstream_model.DeepseekV4MoE):
    """The upstream DSv4.1 MoE with its runner's final all-reduce moved here
    (``skip_final_all_reduce``): :meth:`forward` is :meth:`reduce` of
    :meth:`forward_partial`, so a caller may run the two apart (the prefill
    split reduction, ``sm80_split_reduce.py``)."""

    # VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE: :func:`moe_all_reduce`;
    # otherwise the runner's own all-reduce.
    _round_once: bool = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        mega_gate_metadata=None,
    ) -> torch.Tensor:
        return self.reduce(
            self.forward_partial(hidden_states, input_ids, mega_gate_metadata)
        )

    def forward_partial(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        mega_gate_metadata=None,
    ) -> torch.Tensor:
        """This rank's MoE output (routed experts plus the shared-expert
        shard), not yet all-reduced. Each row depends only on its own input."""
        return super().forward(hidden_states, input_ids, mega_gate_metadata)

    def reduce(self, out: torch.Tensor) -> torch.Tensor:
        """The TP all-reduce of :meth:`forward_partial`'s output, with the mHC
        coefficients the decoder layer deferred beside it
        (``ops/sm80_mhc_side.py``)."""
        if self._round_once:
            return sm80_mhc_side.beside(moe_all_reduce, out)
        return sm80_mhc_side.beside(prefill_all_reduce, out)


def _adopt_layer_all_reduce(ffn: nn.Module, *, round_once: bool) -> None:
    """The layer, not the runner, all-reduces the MoE output, where the runner
    would: rounded once with VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE
    (``round_once``), else through the runner's all-reduce
    (VLLM_SM80_PREFILL_REDUCE_OVERLAP or VLLM_SM80_ALLREDUCE alone)."""
    if type(ffn) is not upstream_model.DeepseekV4MoE:
        raise TypeError(f"Unexpected DeepSeek-V4.1 MoE: {type(ffn).__name__}")
    moe_config = ffn.experts.moe_config
    if moe_config.is_sequence_parallel or moe_config.skip_final_all_reduce:
        raise ValueError("the MoE output is not all-reduced by the runner")
    if moe_config.tp_size == 1 and moe_config.ep_size == 1:
        return
    moe_config.skip_final_all_reduce = True
    ffn.__class__ = Sm80DeepseekV41MoE
    ffn._round_once = round_once
    if round_once:
        logger.info_once(
            "DeepSeek-V4.1 sm_80: MoE outputs too large for the custom all-reduce "
            "whole are all-reduced with one rounding per element on eager steps, "
            "over the copy-engine ring's owner sum (VLLM_SM80_ALLREDUCE=1) or "
            "else the custom all-reduce in pieces "
            "(VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE=1; set 0 for the "
            "runner's all-reduce)",
            scope="global",
        )
