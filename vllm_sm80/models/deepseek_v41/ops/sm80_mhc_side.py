# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The fork point (the mHC Sinkhorn beside the next all-reduce, not beside the
# sublayer's GEMMs) follows GNmac-lab's fused mHC patch: Copyright (c) 2026
# GNmac-lab, MIT.
"""DeepSeek-V4.1 delayed mHC coefficients beside the sublayer's all-reduce
(``VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM``, with ``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH``).

The finish after kernel A (``sm80_mhc_fused._delayed``, either finish kernel)
runs two kinds of CTA that share nothing: per row, the coefficients (post,
comb with its Sinkhorn, the next pre-mix) from kernel A's partials, and the
layer input from the streams and the carried pre-mix. The sublayer reads only
the layer input; the coefficients are first read at the next sublayer
boundary (its kernel A and finish), after the sublayer's TP all-reduce.

With the switch, inside a decoder layer's forward (:data:`LAYER`), a finish
of at most :data:`MAX_ROWS` rows launches its layer-input CTAs in place and
defers its coefficient CTAs (:func:`defer`: the same kernel with the same
arguments, ``PART`` selecting the CTAs). The sublayer's all-reduce
(:func:`beside`, around the attention's ``wo_b`` sum and the MoE output's)
launches them on a side stream after the current stream's work so far, and
the current stream waits for them after the all-reduce's work. Whatever comes
first otherwise launches them inline on the current stream (:func:`flush`):
the next mHC call or the end of the layer's forward. Nothing deferred outlives
a layer's forward.

A deferred launch keeps the stream it was deferred on (the stream whose work
wrote its inputs and owns their memory): an all-reduce or a flush on another
stream launches it inline there, and that stream waits for it. The attention
half goes beside the all-reduce where the layer sums ``wo_b`` itself
(``reduces_wo_b``), the FFN half where the layer all-reduces the MoE output
(``VLLM_SM80_ALLREDUCE_ROUND_ONCE``, ``VLLM_SM80_PREFILL_REDUCE_OVERLAP`` or
``VLLM_SM80_ALLREDUCE``); otherwise the next mHC call launches it inline.

Same bytes as the one-launch finish: every CTA runs the same code on the same
inputs. Kernel A wrote the partials before the fork point; nothing else reads
or writes the partials or the coefficient outputs before the join (they are
the call's own tensors, held by the deferred launch, so their memory is not
reused before it).

CUDA graphs: the fork and the join are event edges inside the all-reduce's
call, so one capture (or one breakable segment) holds both.
"""

from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

# Read once per process; the tests set it directly.
ACTIVE = bool(envs.VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM)
# The largest finish whose coefficient CTAs go beside the all-reduce: decode,
# verify and draft steps (the graph sizes reach 40 rows). Larger steps (prefill
# chunks, the split reduction's) keep the one-launch finish.
MAX_ROWS = 64

# Whether a decoder layer took :data:`LAYER` (:func:`layer_scope`): the
# finish warmup compiles the launches apart only then.
IN_USE = False
_open = False
# The deferred launch and the stream it was deferred on.
_deferred: tuple[Callable[[], None], Any] | None = None
_side: dict[int, tuple[torch.cuda.Stream, torch.cuda.Event]] = {}


class _Layer:
    """A decoder layer's forward (``with LAYER:``): finishes inside it may
    defer their coefficient CTAs; whatever is still deferred at its end is
    launched inline."""

    def __enter__(self) -> None:
        global _open
        _open = True

    def __exit__(self, *exc: object) -> None:
        global _open
        _open = False
        flush()


LAYER = _Layer()


def layer_scope(ops: Any) -> _Layer | None:
    """:data:`LAYER` for a decoder layer whose mHC ops are ``ops`` when the
    switch moves its finish's coefficient CTAs, else None; the line says
    which, once. Builds the device's side stream (before any capture)."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    global IN_USE
    if ops is not sm80_mhc_fused:
        why = (
            "the decoder layers' mHC does not run the fused path "
            "(VLLM_DEEPSEEK_V41_FUSED_MHC=1), or the kernel call capture records it"
        )
    elif not sm80_mhc_fused.FINISH_KERNEL:
        why = "it moves the finish kernel's CTAs: VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1 is required"
    else:
        if _microbatching():
            # The deferred launch is one process-wide slot per forward.
            raise ValueError(
                "VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM does not support microbatching"
            )
        if torch.cuda.is_available():
            _side_stream(torch.cuda.current_device())
        logger.info_once(
            "DeepSeek-V4.1 sm_80: the delayed mHC finish's coefficient CTAs of up "
            "to %d rows run on a side stream beside the sublayer's all-reduce "
            "(VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM=1): the attention half where the "
            "layer all-reduces wo_b itself, the FFN half where "
            "VLLM_SM80_ALLREDUCE_ROUND_ONCE, "
            "VLLM_SM80_PREFILL_REDUCE_OVERLAP or "
            "VLLM_SM80_ALLREDUCE is set; otherwise inline at the "
            "next mHC call",
            MAX_ROWS,
            scope="global",
        )
        IN_USE = True
        return LAYER
    logger.info_once(
        "DeepSeek-V4.1 sm_80: VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM is set but inert: %s",
        why,
        scope="global",
    )
    return None


def _microbatching() -> bool:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config is not None and bool(config.parallel_config.use_ubatching)


def defers(rows: int, device: torch.device) -> bool:
    """Whether a finish of ``rows`` rows on ``device`` launches its
    coefficient CTAs through :func:`defer`."""
    return _open and device.type == "cuda" and 0 < rows <= MAX_ROWS


def defer(launch: Callable[[], None]) -> None:
    """Hold ``launch`` (a finish's coefficient CTAs) for the next
    :func:`beside` or :func:`flush`, with the current stream."""
    global _deferred
    assert _deferred is None, "a deferred mHC finish was not launched"
    _deferred = (launch, current_stream())


def flush() -> None:
    """Launch the deferred coefficient CTAs, if any, inline on the stream
    they were deferred on."""
    global _deferred
    pending, _deferred = _deferred, None
    if pending is not None:
        _inline(*pending)


def _inline(launch: Callable[[], None], origin: Any) -> None:
    """``launch`` on ``origin``; the current stream, if another, waits for
    it."""
    current = current_stream()
    if origin == current:
        launch()
        return
    with torch.cuda.stream(origin):
        launch()
    current.wait_stream(origin)


def beside(fn: Callable[..., torch.Tensor], *args: Any) -> torch.Tensor:
    """``fn(*args)``, a sublayer's TP all-reduce, with the deferred
    coefficient CTAs, if any, on the side stream beside it: they start after
    the current stream's work so far, and the current stream waits for them
    after ``fn``'s work. Deferred on another stream, they launch inline there
    first (:func:`flush`)."""
    global _deferred
    pending, _deferred = _deferred, None
    if pending is None:
        return fn(*args)
    launch, origin = pending
    main = current_stream()
    if origin != main:
        _inline(launch, origin)
        return fn(*args)
    side, done = _side_stream(main.device.index)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        launch()
        done.record(side)
    try:
        return fn(*args)
    finally:
        main.wait_event(done)


def _side_stream(index: int) -> tuple[torch.cuda.Stream, torch.cuda.Event]:
    pair = _side.get(index)
    if pair is None:
        device = torch.device("cuda", index)
        pair = _side[index] = (torch.cuda.Stream(device=device), torch.cuda.Event())
    return pair
