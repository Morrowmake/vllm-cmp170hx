# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's ``MoERunner`` (registered out of tree by ``vllm_sm80.plugin``).

While one of its switches is set (``vllm_sm80.plugin.MOE_LAYERS``), every
``MoERunner`` is built as :class:`Sm80MoERunner`. Its ``__init__`` runs
upstream's unchanged and then changes the class of children upstream built or
was handed, so no module is constructed in a different order and no parameter
is renamed:

- the ``SharedExperts`` -> ``vllm_sm80.layers.moe.shared_experts`` (the
  re-ordered aux-stream overlap, ``VLLM_SHARED_EXPERT_REORDER``);
- a router of exactly ``GroupedTopKRouter`` -> ``vllm_sm80.layers.moe.router``
  (the sm_80 fused decode routing);
- a gate it calls itself gets its router (``vllm_sm80.layers.moe.gate``,
  ``VLLM_MOE_DECODE_ROUTE_V2``).

With ``VLLM_AMPERE_DECODE_IDX_GLUE`` set, :meth:`_select_forward` wraps the
custom-op entry for the "moesum" part: the routed experts' final ``moe_sum``
is deferred and fused with the shared-expert add (``moe_sum_add``). Without it
the entry is upstream's.

Subclasses of ``MoERunner`` (the Transformers backend's runner, Kimi's latent
runners) are looked up under their own names and stay upstream's.
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _unpack,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm_sm80.layers.moe.gate import Sm80GateLinear
from vllm_sm80.layers.moe.router import Sm80GroupedTopKRouter
from vllm_sm80.layers.moe.shared_experts import Sm80SharedExperts


class Sm80MoERunner(MoERunner):
    """``MoERunner`` with the fork's MoE paths (see the module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self._shared_experts) is SharedExperts:
            Sm80SharedExperts.adopt(self._shared_experts)
        if type(self.router) is GroupedTopKRouter:
            Sm80GroupedTopKRouter.adopt(self.router)
        if (
            envs.VLLM_MOE_DECODE_ROUTE_V2
            and isinstance(self.gate, Sm80GateLinear)
            and not self._fse_fuse_gate
        ):
            self.gate.attach_router(self.router)

    def _select_forward(self) -> Callable:
        entry = super()._select_forward()
        # The hidden-dim truncation recorded by `_maybe_pad_hidden_states` for
        # the moesum gate; set before the first forward.
        self._moe_sum_og_pre = None
        self._moe_sum_glue = bool(envs.VLLM_AMPERE_DECODE_IDX_GLUE)
        if not self._moe_sum_glue:
            return entry
        return partial(self._moe_sum_forward_entry, entry)

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        result = super()._maybe_pad_hidden_states(shared_experts_input, hidden_states)
        if self._moe_sum_glue:
            self._moe_sum_og_pre = result[1]
        return result

    def _moe_sum_forward_entry(
        self,
        entry: Callable,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        layer_name,
        hidden_dim_unpadded: int,
    ):
        """`entry` with the routed moe_sum deferred into the shared add.

        When :meth:`_idx_glue_arm_moe_sum` allows it, the experts leave the
        routed sum to this wrapper, which returns ``(None, shared + routed)``
        computed by one ``moe_sum_add`` (bitwise the routed sum, bf16-rounded
        as before, plus the add). Under that gate everything ``forward`` does
        between the entry and ``shared + routed`` is the identity, so the
        result and the kernels are those of doing the add there.
        """
        glue_moe_sum = self._idx_glue_arm_moe_sum(
            hidden_states, self._moe_sum_og_pre
        )
        if glue_moe_sum:
            from vllm_sm80.kernels.moe.sum import MOE_SUM_DEFERRAL

            MOE_SUM_DEFERRAL.arm()
        try:
            result = entry(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                layer_name,
                hidden_dim_unpadded,
            )
        finally:
            pending_moe_sum = MOE_SUM_DEFERRAL.take() if glue_moe_sum else None
        if pending_moe_sum is None:
            return result
        shared_output, fused_output = _unpack(result)
        fused_output = cast(torch.Tensor, fused_output)
        if not self._idx_glue_can_fuse_moe_sum(pending_moe_sum[0], shared_output):
            # Not fusable after all: do the deferred sum where it would have
            # gone (and into fused_output, should the experts' finalize have
            # copied the unsummed buffer), then carry on unchanged.
            from vllm_sm80.kernels.moe.routing import moe_sum as ampere_moe_sum

            ampere_moe_sum(pending_moe_sum[0], out=pending_moe_sum[1])
            if fused_output.data_ptr() != pending_moe_sum[1].data_ptr():
                ampere_moe_sum(pending_moe_sum[0], out=fused_output)
            return result
        from vllm_sm80.kernels.moe.sum import moe_sum_add

        assert shared_output is not None
        return None, moe_sum_add(pending_moe_sum[0], shared_output)

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
        from vllm_sm80.kernels.idx_glue_gate import use_idx_glue

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
