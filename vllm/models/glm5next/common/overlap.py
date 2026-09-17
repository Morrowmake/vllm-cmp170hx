# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Communication/computation overlap for tensor-parallel GLM-5.3-Flash prefill.

Opt-in through ``VLLM_GLM5_PREFILL_OVERLAP=1``. Default off; with it off the
only change to the model is a single ``is None`` test per decoder layer, so the
off path is byte-identical to the unmodified fork.

The problem
-----------
On a PCIe-Gen2 x16, no-P2P box (4x CMP 170HX, NCCL SHM transport) a TP=4
prefill chunk of 1152 tokens costs ~587 ms per rank: ~357 ms (61%) in the 90
per-layer ``all_reduce``s and ~203 ms in compute. The two are strictly
serialised because ``RowParallelLinear`` reduces on the same stream that
produced its input. Each message is 1152 x 4096 x 2 B = 9.4 MB and already runs
at the link ceiling (~2.5 GB/s algbw, see ``chain20_ar_sweep.txt``), so the
bytes cannot be made cheaper -- only the serialisation can be removed.

What this does
--------------
Inside the post-attention part of an mHC decoder layer, the token dimension is
split into ``S`` micro-batches (``VLLM_GLM5_PREFILL_OVERLAP_SPLITS``, default
2). The attention-output all-reduce is issued as ``S`` independent slice
all-reduces on a dedicated comm stream; the main stream then joins slice ``i``,
runs the ffn mHC mix + MoE/MLP for that slice, and submits *its* all-reduce to
the same comm stream. The comm stream therefore stays back-to-back busy while
the main stream computes, and per layer the MLP-side compute disappears under
the collectives:

    today     Ca + AR_a + Cm + AR_m
    overlap   Ca + max(AR_a + AR_m, Cm + something) ~= Ca + AR_a + AR_m

Attention itself is **not** split (see ``overlap_NOTES.md``): the KDA chunked
scan carries state across tokens and the DSA indexer maintains kpool/tail
state, so an intra-chunk split there needs attention-metadata surgery in the
worker. That is left as a follow-up; it is the difference between the ~480 ms
this can reach and the ~380 ms ceiling.

Mechanics
---------
``RowParallelLinear`` and ``MoERunner`` both reduce through
``vllm.distributed.tensor_model_parallel_all_reduce``. Rather than thread a
"don't reduce" flag through both, the region installs an interceptor on that
function (``set_tp_all_reduce_interceptor``) for the exact duration of the two
calls whose collectives it wants to own. The interceptor returns the (not yet
valid) destination buffer immediately and hands back a handle carrying one CUDA
event per slice; the driver waits on those events before the buffer is read.

Ordering (per layer, S=2), which is identical on every rank because every rank
runs the same program:

    comm  | AR_a[0] AR_a[1] . wait(ready0) AR_m[0] . wait(ready1) AR_m[1]
    main  | attn .. join(AR_a[0]) mix+mlp[0] ready0 .. join(AR_a[1])
          |    mix+mlp[1] ready1 .. join(AR_m[0]) join(AR_m[1]) cat

There is no cycle: the comm stream's only dependencies are events recorded on
the main stream *before* the main stream waits on any later comm event. Only
one stream ever has NCCL work in flight, because the layer fully joins before
returning -- so the existing TP communicator is reused and no second NCCL
communicator (and no extra device memory) is needed.

Exactness
---------
Every operation that is split is per-token: the mHC pre/post mixes are
``einsum``s over a leading token dim, RMSNorm is per row, MoE routing is
top-k per row, and the expert/shared-expert GEMMs are row-independent. The
values fed to each slice all-reduce are therefore bit-identical to the values
the unsplit path would have fed to the corresponding rows of the full
all-reduce. Two second-order differences remain and are why the contract is
"BF16 summation-order tolerance" rather than bit-exactness:

1. NCCL ring all-reduce chooses which rank starts the accumulation of a given
   element from that element's offset within the message, so a 4.7 MB message
   sums the same four contributions in a different order than a 9.4 MB one.
2. cuBLASLt and Marlin may select a different tile/split-K schedule at M=576
   than at M=1152, which changes the K-reduction order of the shared-expert
   and o_proj GEMMs.

Both are last-bit BF16 effects, of the same kind and size as vLLM's own custom
all-reduce versus NCCL. No precision is reduced anywhere: dtypes, quantisation
and the KV cache are untouched.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.communication_op import set_tp_all_reduce_interceptor
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "OverlapSettings",
    "read_settings",
    "split_bounds",
    "PrefillOverlapRegion",
    "PendingAllReduce",
    "OverlapExecutor",
    "CudaOverlapExecutor",
    "get_active_region",
    "maybe_open_region",
    "close_region",
]

# Keep micro-batch boundaries on a multiple of this many tokens so the split
# GEMMs stay on tidy tile counts. 1152 / 2 = 576 and 1152 / 4 = 288 are both
# already multiples of 8, so this never bites at the production chunk size.
TOKEN_ALIGN = 8


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _env_flag(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: dict[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if value < minimum:
        raise ValueError(f"{name}={value} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class OverlapSettings:
    """Parsed ``VLLM_GLM5_PREFILL_OVERLAP*`` environment."""

    enabled: bool = False
    splits: int = 2
    # MEASURED 2026-09-17: the scheduler issues ~280-380 token prefill forwards
    # in the production serving config, NOT the 1152-token chunks the KV block
    # size suggests, and NOT the token budget (MAX_BATCHED=2304 changes
    # nothing). A 512 default therefore keeps this feature permanently dormant
    # on the real path -- which is how the first PHASE 2 run measured +0.8%
    # on a feature that never ran. See overlap_NOTES.md R2/R3/R11.
    min_tokens: int = 512
    # Emit a one-line summary of what the first overlapped chunk did.
    debug: bool = False

    @property
    def active(self) -> bool:
        return self.enabled and self.splits > 1


def read_settings(env: dict[str, str] | None = None) -> OverlapSettings:
    """Parse the overlap environment. Pure; safe to call on CPU."""
    env = dict(os.environ) if env is None else env
    enabled = _env_flag(env, "VLLM_GLM5_PREFILL_OVERLAP", False)
    settings = OverlapSettings(
        enabled=enabled,
        splits=_env_int(env, "VLLM_GLM5_PREFILL_OVERLAP_SPLITS", 2, 1),
        min_tokens=_env_int(env, "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS", 512, 1),
        debug=_env_flag(env, "VLLM_GLM5_PREFILL_OVERLAP_DEBUG", False),
    )
    if enabled and settings.splits == 1:
        logger.warning(
            "VLLM_GLM5_PREFILL_OVERLAP=1 with SPLITS=1 does nothing; "
            "the unmodified path is used."
        )
    return settings


def split_bounds(
    num_tokens: int, splits: int, align: int = TOKEN_ALIGN
) -> list[tuple[int, int]]:
    """Contiguous ``[lo, hi)`` micro-batch bounds covering ``num_tokens``.

    Every slice but the last is a multiple of ``align``; the last absorbs the
    remainder. Falls back to a single slice when the split would produce empty
    or degenerate micro-batches.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if splits <= 1 or num_tokens < splits * max(align, 1):
        return [(0, num_tokens)]
    base = (num_tokens // splits // align) * align
    if base <= 0:
        return [(0, num_tokens)]
    bounds: list[tuple[int, int]] = []
    lo = 0
    for i in range(splits):
        hi = num_tokens if i == splits - 1 else lo + base
        bounds.append((lo, hi))
        lo = hi
    return bounds


# --------------------------------------------------------------------------- #
# Stream / event plumbing (injectable so the ordering logic is CPU-testable)
# --------------------------------------------------------------------------- #


class OverlapExecutor:
    """The CUDA-stream operations the region needs, as an injectable seam."""

    def fork(self) -> None:
        """Make the comm stream wait for everything queued on the main stream."""
        raise NotImplementedError

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Enqueue an in->out all-reduce on the comm stream."""
        raise NotImplementedError

    def signal(self) -> Any:
        """Record and return an event on the comm stream."""
        raise NotImplementedError

    def join(self, token: Any) -> None:
        """Make the main stream wait for ``token``."""
        raise NotImplementedError

    def keepalive(self, *tensors: torch.Tensor) -> None:
        """Tell the caching allocator these are in use by the comm stream."""

    def empty_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(tensor)


class CudaOverlapExecutor(OverlapExecutor):
    """Real implementation: one side stream plus the existing TP NCCL comm."""

    def __init__(self, comm_stream: Any, nccl: Any) -> None:
        self._comm_stream = comm_stream
        self._nccl = nccl
        self._events: list[Any] = []
        self._next_event = 0

    def reset_events(self) -> None:
        """Recycle the event pool. Only safe once every event has been joined."""
        self._next_event = 0

    def _event(self) -> Any:
        if self._next_event == len(self._events):
            self._events.append(torch.cuda.Event())
        event = self._events[self._next_event]
        self._next_event += 1
        return event

    def fork(self) -> None:
        event = self._event()
        event.record(torch.cuda.current_stream())
        self._comm_stream.wait_event(event)

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        self._nccl.all_reduce(inp, out, stream=self._comm_stream)

    def signal(self) -> Any:
        event = self._event()
        event.record(self._comm_stream)
        return event

    def join(self, token: Any) -> None:
        torch.cuda.current_stream().wait_event(token)

    def keepalive(self, *tensors: torch.Tensor) -> None:
        for tensor in tensors:
            tensor.record_stream(self._comm_stream)


# --------------------------------------------------------------------------- #
# The region
# --------------------------------------------------------------------------- #


@dataclass
class PendingAllReduce:
    """One intercepted all-reduce, possibly split into slices."""

    out: torch.Tensor
    bounds: list[tuple[int, int]]
    tokens: list[Any] = field(default_factory=list)
    region: "PrefillOverlapRegion | None" = None
    # True when the interceptor could not go async and reduced inline; the
    # result is already valid and joining is a no-op.
    synchronous: bool = False

    def join(self, index: int) -> None:
        if self.synchronous or self.region is None:
            return
        self.region.executor.join(self.tokens[index])

    def join_all(self) -> None:
        if self.synchronous or self.region is None:
            return
        for token in self.tokens:
            self.region.executor.join(token)


class PrefillOverlapRegion:
    """Owns the comm stream and the interceptor for one model forward."""

    def __init__(
        self,
        settings: OverlapSettings,
        executor: OverlapExecutor,
        fallback_all_reduce: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.settings = settings
        self.executor = executor
        self._fallback = fallback_all_reduce
        self._sink: list[PendingAllReduce] | None = None
        self._splits_now = 1
        # Counters, for the debug line and for tests.
        self.submitted = 0
        self.async_submitted = 0
        self.slices = 0
        self.degraded = False

    @property
    def splits(self) -> int:
        return self.settings.splits

    # -- driver-facing API -------------------------------------------------- #

    def begin_layer(self) -> None:
        reset = getattr(self.executor, "reset_events", None)
        if reset is not None:
            reset()

    @contextlib.contextmanager
    def capture(self, splits: int) -> Iterator[list[PendingAllReduce]]:
        """Own every TP all-reduce issued inside the block."""
        handles: list[PendingAllReduce] = []
        prev_sink, prev_splits = self._sink, self._splits_now
        self._sink, self._splits_now = handles, splits
        prev_hook = set_tp_all_reduce_interceptor(self.submit)
        try:
            yield handles
        finally:
            set_tp_all_reduce_interceptor(prev_hook)
            self._sink, self._splits_now = prev_sink, prev_splits

    def single(
        self, handles: Sequence[PendingAllReduce], what: str
    ) -> PendingAllReduce | None:
        """Return the one handle a block was expected to produce.

        If a block produced a different number of collectives than this file
        assumes -- a model change, an unexpected fused path -- everything is
        joined immediately and ``None`` is returned, so the layer falls back to
        unsplit consumption instead of reading a buffer that is still in
        flight. Correct, just not overlapped.
        """
        if len(handles) == 1:
            return handles[0]
        for handle in handles:
            handle.join_all()
        if not self.degraded:
            self.degraded = True
            logger.warning(
                "GLM5 prefill overlap: expected exactly 1 tensor-parallel "
                "all-reduce in %s but saw %d; falling back to the "
                "non-overlapped consumption for this region.",
                what,
                len(handles),
            )
        return None

    # -- interceptor -------------------------------------------------------- #

    def submit(self, tensor: torch.Tensor) -> torch.Tensor:
        """Interceptor body: enqueue ``tensor``'s all-reduce on the comm stream."""
        self.submitted += 1
        sink = self._sink
        splits = self._splits_now
        num_tokens = tensor.shape[0] if tensor.dim() >= 1 else 0
        if sink is None or num_tokens < splits:
            # Not inside a capture (or nothing to slice): behave as before.
            out = self._fallback(tensor)
            if sink is not None:
                sink.append(
                    PendingAllReduce(
                        out=out,
                        bounds=[(0, num_tokens)],
                        region=self,
                        synchronous=True,
                    )
                )
            return out

        tensor = tensor.contiguous()
        out = self.executor.empty_like(tensor)
        bounds = (
            split_bounds(num_tokens, splits) if splits > 1 else [(0, num_tokens)]
        )
        self.executor.fork()
        tokens: list[Any] = []
        for lo, hi in bounds:
            self.executor.all_reduce(tensor[lo:hi], out[lo:hi])
            tokens.append(self.executor.signal())
        self.executor.keepalive(tensor, out)
        self.async_submitted += 1
        self.slices += len(bounds)
        handle = PendingAllReduce(out=out, bounds=bounds, tokens=tokens, region=self)
        sink.append(handle)
        return out


# --------------------------------------------------------------------------- #
# Process-wide activation
# --------------------------------------------------------------------------- #

_SETTINGS: OverlapSettings | None = None
_OBSERVED: dict[int, int] = {}
_OBSERVED_TOTAL = 0
_SKIPPED = 0
_OPENED = 0
_WARNED_DORMANT = False
_REGION: PrefillOverlapRegion | None = None
_ACTIVE: PrefillOverlapRegion | None = None
_LOGGED = False


def get_settings() -> OverlapSettings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = read_settings()
    return _SETTINGS


def reset_for_testing() -> None:
    """Drop cached settings/region. Tests only."""
    global _SETTINGS, _REGION, _ACTIVE, _LOGGED
    global _SKIPPED, _OPENED, _WARNED_DORMANT, _OBSERVED_TOTAL
    _OBSERVED_TOTAL = 0
    _SETTINGS = None
    _REGION = None
    _ACTIVE = None
    _LOGGED = False
    _SKIPPED = 0
    _OPENED = 0
    _WARNED_DORMANT = False
    _OBSERVED.clear()


def observe_forward(num_tokens: int) -> None:
    """Log the distribution of prefill-forward token counts.

    VLLM_GLM5_PREFILL_OBSERVE=1, independent of the overlap feature and
    with no effect on behaviour. Exists because the scheduled chunk size is not
    something any log line states directly: attention block size/kv lcm
    block sizes describe the KV block, and the token budget is only an upper
    bound (an upstream --long-prefill-token-threshold, for one, caps it below
    both). Measuring it beats inferring it.
    """
    global _OBSERVED_TOTAL
    if not _env_flag(dict(os.environ), "VLLM_GLM5_PREFILL_OBSERVE", False):
        return
    if num_tokens < 64:  # decode / drafter steps are not chunks
        return
    _OBSERVED[num_tokens] = _OBSERVED.get(num_tokens, 0) + 1
    _OBSERVED_TOTAL += 1
    if _OBSERVED_TOTAL % 100 == 0:
        top = sorted(_OBSERVED.items(), key=lambda kv: -kv[1])[:6]
        logger.info(
            "GLM5 prefill chunk observer: %d forwards, most common sizes %s",
            _OBSERVED_TOTAL,
            ", ".join(f"{n} tok x{c}" for n, c in top),
        )


def get_active_region() -> PrefillOverlapRegion | None:
    """The region a decoder layer should use, or ``None`` for the normal path."""
    return _ACTIVE


def _build_region(settings: OverlapSettings) -> PrefillOverlapRegion | None:
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    communicator = getattr(tp_group, "device_communicator", None)
    nccl = getattr(communicator, "pynccl_comm", None)
    if nccl is None or getattr(nccl, "disabled", True):
        logger.warning(
            "GLM5 prefill overlap requested but this TP group has no usable "
            "PyNccl communicator; the feature stays off."
        )
        return None
    # The interceptor goes straight to PyNccl, skipping CudaCommunicator's
    # backend selection. On a no-P2P box that is exactly what the selector
    # picks anyway ("Using ['PYNCCL'] all-reduce backends"), but if a faster
    # backend is live we would silently downgrade it, so refuse instead.
    for attribute in (
        "ca_comm",
        "qr_comm",
        "fi_ar_comm",
        "fi_pcie_ipc_ar_comm",
        "symm_mem_comm",
        "aiter_ar_comm",
    ):
        other = getattr(communicator, attribute, None)
        if other is not None and not getattr(other, "disabled", False):
            logger.warning(
                "GLM5 prefill overlap stays off: the TP group has an active "
                "'%s' all-reduce backend, which this path would bypass.",
                attribute,
            )
            return None
    # A dedicated, low-priority-agnostic side stream. The collectives are the
    # long pole, so it is created with default priority and simply kept busy.
    comm_stream = torch.cuda.Stream()
    executor = CudaOverlapExecutor(comm_stream, nccl)
    return PrefillOverlapRegion(settings, executor, tp_group.all_reduce)


def maybe_open_region(
    *,
    num_tokens: int,
    mhc: bool,
    sequence_parallel: bool,
) -> PrefillOverlapRegion | None:
    """Activate the overlap for this forward, or return ``None``.

    Guards, in order of cheapness: the feature flag, a prefill-sized batch,
    mHC (the only layer shape this file knows how to split), no sequence
    parallelism (its reduce-scatter/all-gather pair is a different collective
    schedule), TP > 1, and not inside a CUDA-graph capture -- decode is
    captured and must stay untouched.
    """
    global _ACTIVE, _REGION, _LOGGED, _SETTINGS

    observe_forward(num_tokens)
    settings = get_settings()
    if not settings.active:
        return None
    if num_tokens < settings.min_tokens:
        note_forward_skipped(num_tokens)
        return None
    if not mhc or sequence_parallel:
        return None
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return None

    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() <= 1:
        return None

    if _REGION is None:
        _REGION = _build_region(settings)
        if _REGION is None:
            # Permanently disable rather than retrying every chunk.
            _SETTINGS = OverlapSettings(enabled=False)
            return None
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "GLM5 prefill comm/compute overlap active: splits=%d, "
            "min_tokens=%d (this chunk: %d tokens)",
            settings.splits,
            settings.min_tokens,
            num_tokens,
        )
    globals()["_OPENED"] = _OPENED + 1
    _ACTIVE = _REGION
    return _REGION


def note_forward_skipped(num_tokens: int) -> None:
    """Record a prefill-sized forward that did NOT open a region.

    Guards against the failure mode that cost a whole PHASE 2 leg: the feature
    switched on, never engaging, and reporting as "no effect" rather than
    "never ran". After enough skipped forwards we say so, once, loudly.
    """
    global _SKIPPED, _WARNED_DORMANT
    settings = get_settings()
    if not settings.active or _WARNED_DORMANT:
        return
    _SKIPPED += 1
    if _SKIPPED >= 64 and _OPENED == 0:
        _WARNED_DORMANT = True
        logger.warning(
            "GLM5 prefill overlap is ENABLED but has never engaged: %d "
            "forwards were all below min_tokens=%d (most recent: %d tokens). "
            "The feature is doing nothing. Lower "
            "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS below the real chunk size, "
            "or turn the feature off.",
            _SKIPPED,
            settings.min_tokens,
            num_tokens,
        )


def close_region(region: PrefillOverlapRegion | None) -> None:
    global _ACTIVE
    if region is None:
        return
    _ACTIVE = None
    if region.settings.debug:
        logger.info(
            "GLM5 prefill overlap: %d all-reduces intercepted, %d issued "
            "asynchronously as %d slices%s",
            region.submitted,
            region.async_submitted,
            region.slices,
            " (degraded)" if region.degraded else "",
        )
    region.submitted = 0
    region.async_submitted = 0
    region.slices = 0


@contextlib.contextmanager
def prefill_overlap(
    *, num_tokens: int, mhc: bool, sequence_parallel: bool
) -> Iterator[PrefillOverlapRegion | None]:
    region = maybe_open_region(
        num_tokens=num_tokens, mhc=mhc, sequence_parallel=sequence_parallel
    )
    try:
        yield region
    finally:
        close_region(region)
