# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The re-ordered multi-stream shared-expert overlap (``VLLM_SHARED_EXPERT_REORDER``).

Upstream (#52033) submits the shared experts to the aux stream in
``maybe_forward_async``, which ``MoERunner._forward_impl`` calls before the
gate and the routed dispatch, and joins in ``wait``, which
``MoERunner._apply_quant_method`` calls after the routed experts. That leaves
the aux branch resident long before the routed kernels are submitted, so it
barely overlaps under breakable cudagraphs.

With the flag on, :class:`Sm80SharedExperts` keeps those two call sites and
changes what they do: ``maybe_forward_async`` only marks the aux stream's start
point (the aux stream waits for everything the main stream has submitted so
far) and returns True, and ``wait`` runs the shared experts on the aux stream,
now behind the gate, the router and the routed experts, then joins the main
stream back onto it. Enqueue order only: the same kernels read the same
tensors, and the join still precedes every read of the shared-expert output.

``vllm_sm80.layers.moe.runner.Sm80MoERunner`` swaps every ``SharedExperts`` it
builds to this class. With the flag off both methods are upstream's.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)


def _capture_underway() -> bool:
    """True while the current stream is capturing a CUDA graph.

    Returns False when CUDA is unavailable or not initialised, so the callers
    below stay importable and testable on a host with no visible device.
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _log_shared_experts_order_once(reordered: bool) -> None:
    """Announce which multi-stream enqueue order this process is using."""
    if reordered:
        logger.info_once(
            "MoE shared experts: re-ordered aux-stream overlap "
            "(mark aux stream, enqueue routed experts, then run shared "
            "experts on the aux stream and join) "
            "[VLLM_SHARED_EXPERT_REORDER=1]"
        )
    else:
        logger.info_once(
            "MoE shared experts: upstream aux-stream overlap "
            "(shared experts enqueued on the aux stream before routed "
            "dispatch, joined after) "
            "[VLLM_SHARED_EXPERT_REORDER=0]"
        )


class Sm80SharedExperts(SharedExperts):
    """``SharedExperts`` with the re-ordered aux-stream overlap.

    Instances are made by changing the class of an upstream ``SharedExperts``
    (:meth:`adopt`); nothing else about the module changes.
    """

    @classmethod
    def adopt(cls, shared: SharedExperts) -> "Sm80SharedExperts":
        """Turn an upstream ``SharedExperts`` into this class in place."""
        assert type(shared) is SharedExperts, type(shared)
        shared.__class__ = cls
        # Read once, like every other switch of this module.
        shared._reorder_after_routed = bool(envs.VLLM_SHARED_EXPERT_REORDER)
        # One input per DBO ubatch id, held from the aux-stream mark to `wait`.
        shared._pending_input = [None, None]
        return shared

    def maybe_forward_async(self, shared_experts_input: torch.Tensor) -> bool:
        """Upstream's enqueue, or under VLLM_SHARED_EXPERT_REORDER only the
        aux stream's start point; True when `wait` has work to finish."""
        if not self._reorder_after_routed:
            enqueued = super().maybe_forward_async(shared_experts_input)
            if enqueued:
                _log_shared_experts_order_once(False)
            return enqueued
        if shared_experts_input is None or (
            self._determine_shared_experts_order(shared_experts_input)
            != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False
        assert self._stream is not None
        _log_shared_experts_order_once(True)
        idx = self._output_idx
        assert self._output[idx] is None and self._pending_input[idx] is None
        # The aux stream reads the input after the main stream has queued
        # more work, so hint the caching allocator not to hand the block to
        # another main-stream allocation in the meantime. The caller holds
        # the tensor alive across the whole region anyway, so this is belt
        # and braces; it is skipped while a cudagraph is being captured,
        # where it is an allocator hint with no counterpart in the replayed
        # graph. The fork below, and the join in `wait`, are ordinary event
        # edges and are what actually order the two streams.
        if not _capture_underway():
            shared_experts_input.record_stream(self._stream)
        self._stream.wait_stream(current_stream())
        self._pending_input[idx] = shared_experts_input
        return True

    def wait(self) -> None:
        """Upstream's join, or under VLLM_SHARED_EXPERT_REORDER the shared
        experts on the aux stream behind the routed experts, then the join."""
        if not self._reorder_after_routed:
            super().wait()
            return
        assert self._stream is not None
        idx = self._output_idx
        shared_experts_input = self._pending_input[idx]
        assert shared_experts_input is not None and self._output[idx] is None
        self._pending_input[idx] = None
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        # Join before the caller reads `output`. The output block belongs to
        # the aux stream, exactly as in upstream's path; the next layer's aux
        # fork waits on the main stream, so a later aux-stream reuse of the
        # block cannot precede the main stream's reads of it.
        current_stream().wait_stream(self._stream)
        self._output[idx] = output
