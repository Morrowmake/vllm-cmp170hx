# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.communication_op import get_tp_all_reduce_interceptor
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)

logger = init_logger(__name__)


_MASK_PADDING: bool | None = None
_MASK_MIN_ROWS = 0


def mask_padding_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    """With VLLM_GLM5_MOE_MASK_PADDING=1, set the expert ids of padding rows
    (forward context is_padding) to -1, in place, so the alignment drops them;
    the real rows then get the same expert blocks whatever the padding rows
    hold. In place keeps the tensor's identity, which the sm_80 fused decode
    routing uses to hand over its own alignment; batches it covers
    (<= VLLM_GLM5_DECODE_MOE_MAX_TOKENS rows, captured at exact sizes) are
    skipped. A larger batch that the fused router v2 covered (up to 32 rows)
    had its alignment computed before the mask; that handoff is dropped here
    so the alignment is recomputed from the masked ids. Returns topk_ids."""
    global _MASK_PADDING, _MASK_MIN_ROWS
    if _MASK_PADDING is None:
        import vllm.envs as envs

        _MASK_PADDING = bool(envs.VLLM_GLM5_MOE_MASK_PADDING)
        _MASK_MIN_ROWS = (
            int(envs.VLLM_GLM5_DECODE_MOE_MAX_TOKENS)
            if envs.VLLM_GLM5_DECODE_KERNELS
            else 0
        )
        if _MASK_PADDING:
            logger.info_once(
                "GLM-5 MoE padding mask active: padding rows are routed to no "
                "expert (VLLM_GLM5_MOE_MASK_PADDING=1; set 0 to disable)"
            )
    if not _MASK_PADDING or topk_ids.shape[0] <= _MASK_MIN_ROWS:
        return topk_ids
    if not is_forward_context_available():
        return topk_ids
    is_padding = get_forward_context().is_padding
    # Only a call that sees the whole (padded) batch can be masked row for
    # row: the prefill overlap runs the MoE on row slices of an unpadded
    # batch, and a slice must not take the first rows of the batch's mask.
    if is_padding is None or is_padding.shape[0] != topk_ids.shape[0]:
        return topk_ids
    topk_ids.masked_fill_(is_padding.unsqueeze(1), -1)
    # The sm_80 fused decode router (VLLM_GLM5_DECODE_MOE_ROUTE_V2, up to 32
    # rows) stashed a block alignment of these ids before the mask; drop it
    # so the alignment is recomputed from the masked ids. Which rows are
    # padding is only known on the device, so this is decided by the call
    # shape alone and is the same in eager mode and in a captured graph.
    from vllm.ampere_decode import drop_fused_align

    if drop_fused_align(topk_ids):
        logger.info_once(
            "GLM-5 MoE padding mask: the fused router's block alignment is "
            "recomputed from the masked ids (VLLM_GLM5_MOE_MASK_PADDING=1)"
        )
    return topk_ids


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: "MoERunner",
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


def get_layer_from_name(layer_name: str) -> MoERunnerInterface:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MoERunnerInterface)
    return layer


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return cast(
        torch.Tensor,
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ),
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ),
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# NOTE: `moe_forward` and `moe_forward_shared` being opaque custom ops is a
# load-bearing assumption for the MoE-LoRA dual-stream path.
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor
    | UnfinalizedMoEOutput
    | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput],
) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


class MoERunner(MoERunnerInterface):
    """Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # GEMM produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
            )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        return self.routed_experts.load_weights(weights)

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        if current_platform.is_cpu():
            # CPU never touches the workspace manager (Monolithic experts
            # skip it entirely; Modular experts' _allocate_buffers bypasses
            # it too, see modular_kernel.py) -- the ContextVar-based lane
            # lookup was the only part of this call graph Dynamo can't
            # trace, so CPU can always call the fused-MoE op directly.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts._replace_quant_method(quant_method)

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.routed_experts._set_moe_config(new_moe_config)
        if self._shared_experts is not None:
            self._shared_experts._set_moe_config(new_moe_config)

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by MoERunner.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    def inline_all_reduce_reason(self) -> str | None:
        """Why ``forward`` reads a TP all-reduce result before returning, or None.

        Two paths reduce *inside* the runner and consume the result on the
        spot: the shared-expert all-reduce taken when the MoE kernel reports
        ``output_is_reduced`` (the sum ``shared + fused`` follows at once), and
        the early routed all-reduce ahead of a non-commutative routed output
        transform. A caller that defers TP all-reduces through
        ``set_tp_all_reduce_interceptor`` (the GLM-5 prefill overlap) hands
        back a buffer that is not valid until it joins, so either path would
        read unfinished data there. Ask before opening such a region. When the
        MoE kernel reduces its own output it also runs collectives of its own
        inside the fused op, which the interceptor does not see either.

        Returns None on the ordinary TP path (the only reduction is the final
        one, whose result is returned to the caller unread).
        """
        mc = self.moe_config
        if mc.is_sequence_parallel:
            return None
        fused_reduced = self._fused_output_is_reduced
        if fused_reduced:
            return (
                "the MoE kernel reduces its own output (output_is_reduced), so "
                "the shared-expert output is all-reduced and consumed inside "
                "the runner"
            )
        if (
            self.routed_output_transform is not None
            and not getattr(self.routed_output_transform, "reduce_commutative", False)
            and (mc.tp_size > 1 or mc.ep_size > 1)
        ):
            return (
                "the routed output transform needs the routed output all-reduced "
                "inside the runner before it is applied"
            )
        return None

    @staticmethod
    def _refuse_deferred_all_reduce(what: str) -> None:
        """Raise if a deferring all-reduce interceptor is installed.

        Called only on the two inline-consumed reductions (see
        ``inline_all_reduce_reason``), never on the ordinary TP path.
        """
        if get_tp_all_reduce_interceptor() is not None:
            raise RuntimeError(
                f"MoE runner: the {what} is all-reduced and read inside the "
                "runner, but a TP all-reduce interceptor is installed, which "
                "returns the result before it is valid. This would compute on "
                "unfinished data. Open the deferring region only when "
                "MoERunner.inline_all_reduce_reason() is None (the GLM-5 "
                "prefill overlap checks this and stays off)."
            )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced

        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and fused_output_is_reduced
        ):
            self._refuse_deferred_all_reduce("shared-expert output")
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        """All-reduce latent routed output before its output transform.

        Latent MoE output transforms may contain non-linear ops, e.g. RMSNorm.
        TP partial routed outputs must be summed in latent space before such
        transforms are applied.

        A transform that commutes with the TP sum is exempt: if
        ``sum_r T(x_r) == T(sum_r x_r)``, applying the transform to the local
        partial output and letting the existing late all-reduce sum the
        combined result is equivalent, and costs one collective instead of two.
        Such a transform opts out by setting ``reduce_commutative = True``.
        The default is False, so transforms that do not declare themselves
        keep being reduced early.
        """
        if (
            self.routed_output_transform is not None
            and not getattr(self.routed_output_transform, "reduce_commutative", False)
            and not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not fused_output_is_reduced
        ):
            self._refuse_deferred_all_reduce("routed output (before its transform)")
            fused_output = tensor_model_parallel_all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        """All-reduce the combined output if needed.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # skip_final_all_reduce must not coexist with a pre-reduced fused
        # output. This should be enforced by MoE config initialization.
        if self.moe_config.skip_final_all_reduce:
            assert not self._fused_output_is_reduced, (
                "skip_final_all_reduce requires an un-reduced fused output"
            )

        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced

        if (
            not self.moe_config.is_sequence_parallel
            and not self.moe_config.skip_final_all_reduce
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not output_is_reduced
        ):
            states = tensor_model_parallel_all_reduce(states)

        return states[..., :trunc_size] if trunc_size is not None else states

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim: int | None = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            assert transformed_hidden_dim is not None
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # Truncation sizes for stripping kernel padding from the output.
        # None means no truncation needed (no padding was applied).
        #
        # Two truncation points exist in forward():
        #   pre_xform:  applied to fused_output BEFORE routed_output_transform
        #   post_xform: applied to the final result AFTER all-reduce
        #
        # MoE with routed output transform or shared experts:
        #   - pre_xform applies if the transform needs unpadded routed output
        #     or shared+routed add needs matching hidden dims. For Nemotron-3
        #     Nano, TRTLLM NVFP4 pads routed MoE hidden dim 2688->2816, while
        #     shared output stays 2688.
        #   - post_xform uses shared_experts_hidden_dim when transform and shared
        #     experts make the final output full hidden dim.
        #
        # Standard MoE / MoE without transforms (GPT-OSS, Mixtral):
        #   - pre_xform is None (no early truncation)
        #   - post_xform strips padding after all-reduce (or None if unpadded)
        if transformed_hidden_dim == hidden_states.shape[-1]:
            transformed_hidden_dim = None

        pre_xform_trunc_size = None
        if self.routed_output_transform is not None or shared_experts_hidden_dim > 0:
            pre_xform_trunc_size = transformed_hidden_dim
        post_xform_trunc_size = transformed_hidden_dim
        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            post_xform_trunc_size = shared_experts_hidden_dim

        return hidden_states, pre_xform_trunc_size, post_xform_trunc_size

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(shared_experts_input, order)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_overlapping: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).

        `shared_experts_overlapping` should be True only if using multi-stream
        overlap. Then the shared expert was already launched in a separate
        stream, so the results only have to be awaited here.
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        if self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

            topk_ids = mask_padding_topk_ids(topk_ids)

            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )

        if shared_experts_overlapping:
            assert self._shared_experts is not None
            self._shared_experts.wait()
        else:
            # Re-ordered aux-stream overlap: run the shared experts now, on
            # the aux stream, behind the routed experts just enqueued; the
            # call joins the aux stream before it returns. A no-op unless
            # VLLM_GLM5_SHARED_EXPERT_REORDER is set and the overlap decision
            # chose the multi-stream order, since every other order was
            # already served by the NO_OVERLAP call above or by the kernel.
            self._maybe_apply_shared_experts(
                shared_experts_input, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
            )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.
        """
        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        glue_moe_sum = envs.VLLM_GLM5_DECODE_IDX_GLUE and self._idx_glue_arm_moe_sum(
            hidden_states, og_hidden_dim_pre_xform
        )
        if glue_moe_sum:
            from vllm.ampere_decode.idx_glue import MOE_SUM_DEFERRAL

            MOE_SUM_DEFERRAL.arm()
        try:
            result = self._forward_entry(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                self._encode_layer_name(),
                self.moe_config.hidden_dim_unpadded
                if self._quant_method.has_unpadded_output
                else 0,
            )
        finally:
            pending_moe_sum = MOE_SUM_DEFERRAL.take() if glue_moe_sum else None

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)
        fused_output = cast(torch.Tensor, fused_output)
        if pending_moe_sum is not None and not self._idx_glue_can_fuse_moe_sum(
            pending_moe_sum[0], shared_output
        ):
            # Not fusable after all: do the deferred sum where it would have
            # gone (and into fused_output, should the experts' finalize have
            # copied the unsummed buffer), then carry on unchanged.
            from vllm.ampere_decode.moe_routing import moe_sum as ampere_moe_sum

            ampere_moe_sum(pending_moe_sum[0], out=pending_moe_sum[1])
            if fused_output.data_ptr() != pending_moe_sum[1].data_ptr():
                ampere_moe_sum(pending_moe_sum[0], out=fused_output)
            pending_moe_sum = None

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_output_is_reduced = self._fused_output_is_reduced

        # Latent routed output has to be reduced before output transform,
        # because the transform may include non-linear normalization.
        fused_output, fused_output_is_reduced = (
            self._maybe_reduce_routed_output_before_transform(
                fused_output,
                fused_output_is_reduced,
            )
        )

        # If routed output is already reduced, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(
            shared_output, fused_output_is_reduced
        )

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if pending_moe_sum is not None:
            # VLLM_GLM5_DECODE_IDX_GLUE "moesum": the routed sum (bf16-rounded
            # as before) and this add in one kernel; bitwise the same result.
            from vllm.ampere_decode.idx_glue import moe_sum_add

            assert shared_output is not None
            result = moe_sum_add(pending_moe_sum[0], shared_output)
        elif shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, fused_output_is_reduced
        )

        return self._maybe_add_zero_expert_output(result)

    def _idx_glue_arm_moe_sum(
        self, hidden_states: torch.Tensor, og_hidden_dim_pre_xform
    ) -> bool:
        """Host gate: may the routed moe_sum be deferred into the shared add?

        Everything between the experts' sum and ``shared + routed`` must be the
        identity: no routed scale, no output transform, no reduction of the
        routed output before the add, no hidden-dim padding. Decode sizes only
        (M <= 32, the captured thin-GEMM region).
        """
        if (
            self._shared_experts is None
            or self.routed_scaling_factor != 1.0
            or self.routed_output_transform is not None
            or og_hidden_dim_pre_xform is not None
            or self._fused_output_is_reduced
            or hidden_states.dim() != 2
            or hidden_states.shape[0] > 32
            or hidden_states.dtype != torch.bfloat16
        ):
            return False
        from vllm.ampere_decode import use_idx_glue

        return use_idx_glue("moesum")

    @staticmethod
    def _idx_glue_can_fuse_moe_sum(
        moe_inp: torch.Tensor, shared_output: torch.Tensor | None
    ) -> bool:
        return (
            shared_output is not None
            and shared_output.dim() == 2
            and moe_inp.dim() == 3
            and shared_output.dtype == moe_inp.dtype
            and shared_output.shape[0] == moe_inp.shape[0]
            and shared_output.shape[1] == moe_inp.shape[2]
        )

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 or self.moe_config.is_sequence_parallel
        ) and not self._quant_method.supports_internal_mk

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor | UnfinalizedMoEOutput,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]
    ):
        if self.do_naive_dispatch_combine:
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "Naive expert-parallel combine cannot consume a deferred "
                    "MoE output."
                )
            hidden_states = get_ep_group().combine(
                hidden_states,
                self.moe_config.is_sequence_parallel,
            )

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "PCP reduce-scatter cannot consume a deferred MoE output."
                )
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput]
    ):
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns routed output, optionally paired with shared-expert output. A
        fused consumer may request the routed output in deferred-finalize form.
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        # Multi-stream overlap for the shared experts. Upstream launches them
        # here, before the gate and the routed dispatch, and joins after the
        # routed experts. Under VLLM_GLM5_SHARED_EXPERT_REORDER this call
        # enqueues nothing and only marks the aux stream's start point; the
        # shared experts are then run on the aux stream inside
        # _apply_quant_method, after the routed experts have been enqueued.
        shared_experts_overlapping = False
        if self._shared_experts is not None:
            shared_experts_overlapping = self._shared_experts.maybe_forward_async(
                shared_experts_input
            )
            if not shared_experts_overlapping and shared_experts_input is not None:
                self._shared_experts.maybe_sync_shared_experts_stream(
                    shared_experts_input
                )

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = dispatch_unquantized_gemm()(
                    self, hidden_states, self._combined_gate_weight, None
                )
            else:
                router_logits = None
                if envs.VLLM_GLM5_DECODE_MOE_ROUTE_V2:
                    # sm_80: gate GEMV + top-k + alignment in one op; the
                    # router and Marlin pick up the stashed results.
                    from vllm.ampere_decode import maybe_moe_route_v2

                    router_logits = maybe_moe_route_v2(
                        self.gate, self.router, hidden_states
                    )
                if router_logits is None:
                    router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states,
                router_logits,
            )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
                shared_experts_overlapping=shared_experts_overlapping,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )

    #########################################################
    #
    # Old methods from FusedMoE layer. Remove when possible.
    #
    #########################################################

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self.routed_experts.activation

    #
    # Expert maps
    #

    @property
    def expert_map_manager(self):
        """Forward to routed_experts.expert_map_manager for backward compatibility."""
        return self.routed_experts.expert_map_manager

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.routed_experts.expert_map

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.routed_experts._expert_routing_tables()

    def update_expert_map(self):
        self.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.routed_experts._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return self.routed_experts.get_expert_weights()

    #
    # EPLB
    #

    @property
    def eplb_state(self) -> EplbLayerState | None:
        return self.router.eplb_state

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        if self.router.eplb_state is not None:
            self.router.eplb_state.set_layer_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )
