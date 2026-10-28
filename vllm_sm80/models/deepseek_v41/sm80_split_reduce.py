# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 prefill all-reduces beside compute, in two row slices
(``VLLM_SM80_PREFILL_REDUCE_OVERLAP``).

A decoder layer sums two TP partials: ``wo_b`` after the attention and the
MoE output (routed experts plus the shared-expert shard) after the FFN. In a
prefill step each is a large NCCL all-reduce between the compute that makes
it and the compute that needs it, so the step serialises them. Everything
from ``wo_a`` to the MoE all-reduce depends only on its own row; with the
switch, an eager step of enough rows runs that part of the layer in two row
slices A = ``[0, cut)`` and B = ``[cut, rows)`` and issues the four
all-reduces on a side stream, in program order:

    main: attention (all rows) | wo_b(A) | wo_b(B) | FFN(A) | FFN(B) | join
    side:                       sum wo_b(A) | sum wo_b(B) | sum FFN(A) | sum FFN(B)

``wo_b(B)`` runs beside the sum of ``wo_b(A)``, ``FFN(A)`` beside the sum of
``wo_b(B)`` and ``FFN(B)`` beside the sum of ``FFN(A)``. ``FFN`` is the FFN's
mHC post + pre and the MoE. The layer's outputs are whole again at the join,
so the next layer, the Engram layers and the decoder replay see the unsplit
layer's tensors row for row.

The cut is a multiple of :data:`SLICE_ALIGN` = 512 rows, the routed experts'
row chunk (``moe_out_prefill_v2``, ``csrc/exl3/moe_prefill_v2.cu``
``kChunkRows``) and a multiple of Marlin's 64-row blocks, so the expert
kernels run the row chunks they run unsplit and decode no weight twice. With
``VLLM_EXL3_MOE_PREFILL_UNCHUNKED`` a call of up to 2048 rows runs as one
launch sequence, so a split step decodes each touched expert once per slice
(twice per step) where the unsplit step would decode it once; the results
are the same. Each slice's sum takes the path the whole message would
(``DeepseekV41Sm80Attention.reduce_wo_b``, ``Sm80DeepseekV41MoE.reduce``). A
step splits only when the host-staged and the custom all-reduce take neither
slice: their messages stay whole on the main stream.

A split step is not bit-equal to the unsplit one: both all-reduce paths sum
an element in an order set by its place in the message. NCCL's ring does, and
so does the custom all-reduce's two-stage kernel that carries the round-once
``wo_b`` pieces (each rank owns a share of a piece by position, and splitting
moves every element of slice B), so a sum can move by one BF16 ulp. Splitting
is judged by the V2 prefill dNLL check, not bitwise. On TP4 at 2048-row
chunks, steps split only at 1844-2048 rows (the custom all-reduce takes a
whole 512-row slice, so shorter steps run whole).

Eager steps only (never during CUDA graph capture or replay), target decoder
layers (``DeepseekV4Model``), TP > 1.
"""

from collections.abc import Callable, Iterable

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm_sm80.models.deepseek_v41 import sm80_attention
from vllm_sm80.models.deepseek_v41.sm80_moe import Sm80DeepseekV41MoE
from vllm_sm80 import capture
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

# The routed experts' row chunk (moe_out_prefill_v2 kChunkRows).
SLICE_ALIGN = 512


def splits(layer) -> bool:
    """Whether the decoder layer issues both of its sums itself (TP ``wo_b``
    and the adopted MoE's :meth:`~Sm80DeepseekV41MoE.reduce`), so its eager
    steps may split. The hot pair ``prefill_reduce_overlap``
    (``vllm_sm80.hot_swap``) sets ``_split_reduce`` on these layers."""
    return isinstance(layer.ffn, Sm80DeepseekV41MoE) and layer.attn.reduces_wo_b()


def adopt_split_reduce(layers: Iterable) -> None:
    """Let the target decoder layers that :func:`splits` names split."""
    count = 0
    for layer in layers:
        if splits(layer):
            layer._split_reduce = True
            count += 1
    logger.info_once(
        "DeepSeek-V4.1 sm_80: %d decoder layers run the wo_b, FFN and both "
        "all-reduces of eager steps of at least %d rows in two row slices, the "
        "all-reduces on a side stream (VLLM_SM80_PREFILL_REDUCE_OVERLAP=1)",
        count,
        2 * SLICE_ALIGN,
        scope="global",
    )


def cut_rows(rows: int) -> int | None:
    """Slice A's rows: the multiple of :data:`SLICE_ALIGN` nearest half of
    ``rows`` (halves round up); None when a slice would be shorter than it."""
    cut = (rows + SLICE_ALIGN) // (2 * SLICE_ALIGN) * SLICE_ALIGN
    if cut < SLICE_ALIGN or rows - cut < SLICE_ALIGN:
        return None
    return cut


def _whole_path_takes(t: torch.Tensor) -> bool:
    """Whether the host-staged or the custom all-reduce takes ``t``."""
    comm = sm80_attention.get_tp_group().device_communicator
    host = getattr(comm, "hostshm_comm", None)
    if host is not None and host.should_host_ar(t):
        return True
    return sm80_attention.custom_all_reduce_takes(t)


def _workspace_locked() -> bool:
    """Whether the worker's scratch workspace is locked: true only after the
    profile and warm-up runs. Those runs must call every layer whole, so the
    workspace (sized by the largest call they make, then locked) holds the
    largest MoE call any later step makes, split or not: a step's slices are
    never larger than the whole calls the profile run made."""
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    return is_workspace_manager_initialized() and current_workspace_manager().is_locked()


def split_point(x: torch.Tensor, mega_gate_metadata) -> int | None:
    """Where an adopted layer splits this call's rows (``x``: the attention
    input), or None to run the call whole. Every rank decides the same: the
    inputs are the step's row count and the TP group's settings."""
    if not x.is_cuda or mega_gate_metadata is not None or capture.active:
        return None
    if not _workspace_locked():
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    if (
        not is_forward_context_available()
        or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE
    ):
        return None
    cut = cut_rows(x.shape[0])
    if cut is None:
        return None
    # x has the MoE output's shape and dtype; wo_b's message is as large or
    # larger (FP32 unless a BF16 switch sends it in BF16).
    if _whole_path_takes(x[:cut]) or _whole_path_takes(x[cut:]):
        return None
    return cut


class _Pending:
    """A side-stream result; :meth:`wait` hands it to the current stream."""

    __slots__ = ("out", "done")

    def __init__(self, out: torch.Tensor, done: torch.cuda.Event | None):
        self.out = out
        self.done = done

    def wait(self) -> torch.Tensor:
        if self.done is not None:
            stream = current_stream()
            stream.wait_event(self.done)
            # Allocated on the side stream, last read on this one.
            self.out.record_stream(stream)
        return self.out


class SideStream:
    """The device's stream for the split layers' all-reduces: one at a time,
    in program order, so every rank issues the same collectives in the same
    order on one stream. High priority: a ready all-reduce's blocks are
    dispatched ahead of the compute blocks still queued, so a sum starts when
    its input is ready rather than after the running grid drains."""

    def __init__(self, device: torch.device):
        self.stream = torch.cuda.Stream(device=device, priority=-1)

    def issue(
        self, fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> _Pending:
        """``fn(t)`` on the side stream, after the current stream's work so far."""
        self.stream.wait_stream(current_stream())
        with torch.cuda.stream(self.stream):
            out = fn(t)
            done = torch.cuda.Event()
            done.record(self.stream)
        # Allocated on the current stream, last read on the side stream.
        t.record_stream(self.stream)
        return _Pending(out, done)


class _Inline:
    """CPU stand-in for :class:`SideStream` (tests): ``fn(t)`` at once."""

    @staticmethod
    def issue(fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor) -> _Pending:
        return _Pending(fn(t), None)


_SIDE: dict[torch.device, SideStream] = {}


def side_stream(device: torch.device) -> SideStream | _Inline:
    if device.type != "cuda":
        return _Inline()
    side = _SIDE.get(device)
    if side is None:
        side = _SIDE[device] = SideStream(device)
    return side


def post_attention(
    layer,
    positions: torch.Tensor,
    x: torch.Tensor,
    cut: int,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    attn_pre: torch.Tensor,
    input_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``layer``'s attention on every row of ``x`` (its normalised attention
    input), then ``wo_b``, the FFN's mHC post + pre, the MoE and both
    all-reduces in the row slices ``[0, cut)`` and ``[cut, rows)`` (see the
    module docstring). Returns the layer's FFN output and mHC state, whole:
    ``(x, residual, post_mix, res_mix, ffn_pre)`` as the unsplit forward."""
    attn, ffn, ops = layer.attn, layer.ffn, layer._mhc_ops
    attn_out = attn.attend(positions, x)
    halves = (slice(0, cut), slice(cut, attn_out.shape[0]))
    side = side_stream(attn_out.device)
    # wo_b(B) is issued before wo_b(A)'s sum is awaited: it runs beside it.
    wo_b = [side.issue(attn.reduce_wo_b, attn.wo_b_partial(attn_out[h])) for h in halves]
    mixes = layer._mixes(False)
    moe: list[_Pending] = []
    mhc: list[tuple[torch.Tensor, ...]] = []
    for h, attn_sum in zip(halves, wo_b):
        res_h, post_h, comb_h, x_h, pre_h, _ = ops.mhc_fused_post_pre_delayed(
            attn_sum.wait(),
            residual[h],
            post_mix[h],
            res_mix[h],
            *mixes,
            pre_mix=attn_pre[h],
            norm_weight=layer.ffn_norm.weight,
            norm_eps=layer.ffn_norm.variance_epsilon,
        )
        ids = None if input_ids is None else input_ids[h]
        moe.append(side.issue(ffn.reduce, ffn.forward_partial(x_h, ids)))
        mhc.append((res_h, post_h, comb_h, pre_h))
    out = torch.cat([m.wait() for m in moe])
    residual, post_mix, res_mix, ffn_pre = (torch.cat(parts) for parts in zip(*mhc))
    return out, residual, post_mix, res_mix, ffn_pre
