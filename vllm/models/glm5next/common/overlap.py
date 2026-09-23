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

Which backend carries the split collectives
-------------------------------------------
The interceptor does not go through ``CudaCommunicator.all_reduce``'s dispatch
chain -- it needs a collective it can place on a named stream, and that chain
takes the current stream and picks a backend per call. It therefore chooses a
backend itself, once, at region build time:

``VLLM_GLM5_PREFILL_OVERLAP_BACKEND`` = ``auto`` (default) | ``custom`` |
``nccl`` | ``hostshm``.

``auto`` means ``custom`` when the TP group has a live ``ca_comm`` and ``nccl``
otherwise. Whatever is chosen, a slice the backend declines falls through to
PyNccl for that slice, so the choice can never fail a forward -- it can only
be slower.

This used to be a blanket refusal: if the TP group had *any* other active
all-reduce backend the overlap switched itself off for the whole process, on
the grounds that going straight to PyNccl would silently bypass something
faster. On this box that reasoning does not survive the numbers. With
``VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1`` the group is ``['CUSTOM','PYNCCL']``
and ``CustomAllreduce.max_size`` is 8 MiB, while an unsplit 1152-token chunk is
1152 x 4096 x 2 = 9.4 MB -- over the cap, so ``should_custom_ar`` rejects it and
the "faster backend" the overlap was standing aside for never carried that
message at all. The 4.7 MB halves the overlap issues are *under* the cap, so
routing them through ``ca_comm`` is an upgrade rather than a bypass. Measured
cost of the old refusal: cold prefill 2222 -> 2083 tok/s (-6.3%) and TTFT@23K
+9.6% (``p2p_tp4_validation.md`` section 3).

``VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA=0`` restores the old stand-down, so the
three arms -- stand down, split via CUSTOM, split via PyNccl -- are an A/B and
not a rewrite. Backends nobody has analysed on a side stream (``qr_comm``,
``fi_ar_comm``, ``fi_pcie_ipc_ar_comm``, ``symm_mem_comm``, ``aiter_ar_comm``)
still cause the old blanket stand-down; only ``ca_comm`` has been worked
through.

Why ``ca_comm`` is legal on the side stream
-------------------------------------------
* **Stream.** ``ops.all_reduce`` takes no stream argument; it reads
  ``get_current_cuda_stream(device)``. Issuing it inside
  ``torch.cuda.stream(comm_stream)`` therefore puts both the staging
  ``cudaMemcpyAsync`` and the reduce kernel on the comm stream. There is no
  hidden default-stream work.
* **No ``capture()``.** ``CustomAllreduce.capture()`` exists only to collect
  graph-buffer IPC handles during CUDA-graph capture. The overlap refuses to
  open inside a capture (``maybe_open_region``), so the eager path is the only
  one reached, and ``register_graph_buffers`` never sees these tensors.
* **No IPC registration of the slices.** With ``registered=False`` the input is
  copied into the pre-registered ``buffer_ptrs[rank]`` staging buffer first, so
  an arbitrary weakly-contiguous input pointer is fine. Our slices are
  contiguous (a leading-dim slice of a contiguous tensor), which satisfies
  ``is_weak_contiguous``.
* **Back-to-back slices on one stream are safe**, but the staging buffer and
  the two-shot scratch are protected by *different* barriers, which is worth
  separating:

  - *Staging.* Call N+1's ``cudaMemcpyAsync`` into ``buffer_ptrs[rank]``
    happens **before** its start barrier, so that barrier cannot be what
    protects it. What protects it is call N's *end* barrier: in
    ``cross_device_reduce_2stage`` every block finishes reading the peers'
    staging buffers in stage 1 and only then publishes its end flag, so once
    my own kernel N has retired -- which the stream guarantees before N+1's
    memcpy is issued -- every peer has finished reading my staging buffer.
    ``cross_device_reduce_1stage`` gives the same guarantee from its final
    ``barrier_at_end``.
  - *Two-shot scratch.* A peer may still be gathering from my scratch when my
    kernel N retires; the two-shot kernel has no final barrier. But call N+1
    writes scratch only **after** its start barrier, and a peer publishes its
    N+1 start flag only once its own kernel N -- scratch reads included -- has
    retired.

  Both arguments need every rank to issue the same collectives in the same
  order, which holds because every rank runs the same program and the backend
  choice is a pure function of size and dtype.
* **Two streams are NOT safe.** Precisely because that state is shared, a
  ``ca_comm`` all-reduce on the main stream concurrent with one on the comm
  stream would interleave the flag protocol. The interceptor owns every TP
  all-reduce inside the region, but its own inline-fallback path (a message too
  small to slice) does run on the main stream, so that path first drains the
  comm stream (``PrefillOverlapRegion.drain``). The same drain removes a
  pre-existing latent hazard on the PyNccl path, where two concurrent
  collectives on one communicator can deadlock.
* **The 8 MiB cap is the trap.** ``should_custom_ar`` is a strict ``<``, so at
  2048 tokens with SPLITS=2 each half is *exactly* 8 MiB and is declined -- the
  arm would quietly measure PyNccl. The executor counts what each slice
  actually got and says so, once, rather than leaving that to a profiler.

Exactness under ``custom``
--------------------------
``cross_device_reduce_2stage`` gives element ``i`` to rank ``i / (size/ngpus)``
and that rank accumulates in the rotated order ``(rank + j) % ngpus``. The
order is a fixed function of ``(size, world_size)``, so it is deterministic
across calls and bitwise identical on every rank -- but it does change when a
9.4 MB message becomes two 4.7 MB ones, because the ownership boundaries move.
That is the same second-order BF16 effect as item 1 above, not a new class of
difference, and it is judged the same way: against the same-server logprob
floor. ``1stage`` has no such dependence (fixed order 0..N-1 at every size).
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
    "SlicedMHCState",
    "OverlapSettings",
    "SPLIT_BACKENDS",
    "UNANALYSED_BACKENDS",
    "select_split_backend",
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

# Backends the region knows how to drive on its own stream. "auto" resolves at
# build time to "custom" when the TP group has a live ca_comm, else "nccl".
SPLIT_BACKENDS = ("auto", "custom", "nccl", "hostshm")

# Other all-reduce backends on the TP group. Nobody has worked out what these
# do when driven from a side stream, so their presence still switches the whole
# feature off, exactly as before. ``ca_comm`` is deliberately not in this list.
UNANALYSED_BACKENDS = (
    "qr_comm",
    "fi_ar_comm",
    "fi_pcie_ipc_ar_comm",
    "symm_mem_comm",
    "aiter_ar_comm",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _env_flag(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_choice(
    env: dict[str, str], name: str, default: str, allowed: Sequence[str]
) -> str:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value not in allowed:
        raise ValueError(f"{name}={raw!r} must be one of {', '.join(allowed)}")
    return value


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
    # Carry the micro-batch split across the layer boundary so the NEXT layer's
    # pre-attention mHC mix runs per slice, under the previous layer's MLP
    # collective, instead of after a full materialisation. Sub-flag so the
    # validated S=2 production path is unaffected while this is A/B'd.
    cross_layer: bool = False
    # MEASURED 2026-09-17: the scheduler issues ~280-380 token prefill forwards
    # in the production serving config, NOT the 1152-token chunks the KV block
    # size suggests, and NOT the token budget (MAX_BATCHED=2304 changes
    # nothing). A 512 default therefore keeps this feature permanently dormant
    # on the real path -- which is how the first PHASE 2 run measured +0.8%
    # on a feature that never ran. See overlap_NOTES.md R2/R3/R11.
    min_tokens: int = 512
    # Emit a one-line summary of what the first overlapped chunk did.
    debug: bool = False
    # Run the overlap even when the TP group has a live `ca_comm`, instead of
    # standing the whole feature down. 0 restores the historic refusal, which
    # is the control arm of the A/B.
    under_ca: bool = True
    # Which communicator carries the split collectives; see the module
    # docstring. "auto" -> "custom" with a live ca_comm, else "nccl".
    backend: str = "auto"

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
        cross_layer=_env_flag(env, "VLLM_GLM5_PREFILL_OVERLAP_CROSS_LAYER", False),
        debug=_env_flag(env, "VLLM_GLM5_PREFILL_OVERLAP_DEBUG", False),
        under_ca=_env_flag(env, "VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA", True),
        backend=_env_choice(
            env, "VLLM_GLM5_PREFILL_OVERLAP_BACKEND", "auto", SPLIT_BACKENDS
        ),
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


def select_split_backend(
    settings: OverlapSettings,
    *,
    ca_active: bool,
    hostshm_active: bool = False,
    unanalysed: str | None = None,
) -> tuple[str | None, str]:
    """Decide which communicator carries the region's split collectives.

    Pure, so the policy can be tested without a TP group. Returns
    ``(backend, reason)``; a ``None`` backend means stand the feature down and
    ``reason`` says why, in words fit for the log line.
    """
    if unanalysed is not None:
        return None, (
            f"the TP group has an active '{unanalysed}' all-reduce backend, "
            "which this path has not been analysed against"
        )
    if ca_active and not settings.under_ca:
        return None, (
            "the TP group has an active 'ca_comm' all-reduce backend and "
            "VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA=0"
        )
    choice = settings.backend
    if choice == "auto":
        choice = "custom" if ca_active else "nccl"
        return choice, f"auto -> {choice}"
    if choice == "custom" and not ca_active:
        return "nccl", (
            "VLLM_GLM5_PREFILL_OVERLAP_BACKEND=custom but this TP group has no "
            "active 'ca_comm'; the split collectives use PyNccl"
        )
    if choice == "hostshm" and not hostshm_active:
        return "nccl", (
            "VLLM_GLM5_PREFILL_OVERLAP_BACKEND=hostshm but this TP group has no "
            "active host-shm all-reduce; the split collectives use PyNccl"
        )
    return choice, f"requested {choice}"


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
    """Real implementation: one side stream plus the TP group's communicators.

    ``backend`` names the communicator the split collectives are *offered* to;
    anything it declines (size cap, dtype, alignment) falls through to PyNccl
    for that slice, which is always present -- the region refuses to build
    without it. ``slice_backends`` records what each slice actually got, which
    is the only way to tell a "ran on CUSTOM" leg from a "CUSTOM declined every
    slice because they were 8 MiB" leg without a profiler. It is cumulative for
    the life of the process -- ``close_region`` resets the per-forward counters
    but deliberately not this one.
    """

    def __init__(
        self,
        comm_stream: Any,
        nccl: Any,
        *,
        backend: str = "nccl",
        ca: Any = None,
        hostshm: Any = None,
    ) -> None:
        self._comm_stream = comm_stream
        self._nccl = nccl
        self._events: list[Any] = []
        self._next_event = 0
        self.backend = backend
        self._ca = ca
        self._hostshm = hostshm
        self.slice_backends: dict[str, int] = {"custom": 0, "hostshm": 0, "nccl": 0}
        self._declined_warned = False

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

    def choose(self, inp: torch.Tensor) -> str:
        """Which backend will carry ``inp``. Pure; no CUDA calls."""
        if self.backend == "custom":
            ca = self._ca
            if ca is not None and not getattr(ca, "disabled", False):
                if ca.should_custom_ar(inp):
                    return "custom"
        elif self.backend == "hostshm":
            hostshm = self._hostshm
            if hostshm is not None and not getattr(hostshm, "disabled", False):
                if hostshm.should_host_ar(inp):
                    return "hostshm"
        return "nccl"

    def _note(self, inp: torch.Tensor, chosen: str) -> None:
        self.slice_backends[chosen] = self.slice_backends.get(chosen, 0) + 1
        if chosen == "nccl" and self.backend != "nccl" and not self._declined_warned:
            self._declined_warned = True
            logger.warning(
                "GLM5 prefill overlap: the '%s' backend declined a %d-byte "
                "slice, so this slice went to PyNccl. CustomAllreduce's cap is "
                "a strict '< max_size' (8 MiB by default), so e.g. a 2048-token "
                "chunk at SPLITS=2 is exactly at the cap and is refused. Raise "
                "VLLM_GLM5_PREFILL_OVERLAP_SPLITS, or lower the chunk size, if "
                "this arm was meant to measure '%s'.",
                self.backend,
                inp.numel() * inp.element_size(),
                self.backend,
            )

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        chosen = self.choose(inp)
        self._note(inp, chosen)
        if chosen == "custom":
            # ops.all_reduce reads the *current* stream, so the context manager
            # is what puts the staging memcpy and the reduce kernel on the comm
            # stream. registered=False -> the input is staged through the
            # pre-registered IPC buffer, so no slice needs registering.
            with torch.cuda.stream(self._comm_stream):
                self._ca.all_reduce(inp, out=out, registered=False)
            return
        if chosen == "hostshm":
            with torch.cuda.stream(self._comm_stream):
                out.copy_(self._hostshm.host_all_reduce(inp))
            return
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
class SlicedMHCState:
    """A layer's pre-attention mHC mix, already applied, still in micro-batches.

    Handed from layer L to layer L+1 in place of the usual
    ``(residual, post, comb)`` triple when cross-layer mode is on. Layer L
    computed it slice by slice as each of its MLP collectives landed, so layer
    L+1 skips its own pre-attention mix and goes straight to attention.

    Only the attention input needs to be contiguous; these three stay sliced,
    which is what removes the 37.7 MB ``residual`` concatenation per layer.
    """

    residual: list[torch.Tensor]
    post: list[torch.Tensor]
    comb: list[torch.Tensor]
    bounds: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.bounds)


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
        # The most recent signal the comm stream recorded. Joining it makes the
        # main stream wait for every collective queued on the comm stream
        # before it, because they are all on that one stream in issue order.
        self._last_token: Any = None
        # Counters, for the debug line and for tests.
        self.submitted = 0
        self.async_submitted = 0
        self.slices = 0
        self.degraded = False
        # Settings are frozen and shared; this is per-forward, because
        # aux-hidden-state capture has to suppress cross-layer handoff.
        self.cross_layer = settings.cross_layer

    @property
    def splits(self) -> int:
        return self.settings.splits

    # -- driver-facing API -------------------------------------------------- #

    def begin_layer(self) -> None:
        self._last_token = None
        reset = getattr(self.executor, "reset_events", None)
        if reset is not None:
            reset()

    def drain(self) -> None:
        """Make the main stream wait for everything the comm stream holds.

        Needed before any collective the main stream issues itself, because the
        backends underneath are not safe to drive from two streams at once: the
        custom all-reduce shares one staging buffer and one ``Signal`` block
        between calls, and two concurrent PyNccl collectives on one
        communicator can deadlock. In practice only the inline fallback below
        reaches this, and only for a message too small to slice.
        """
        token = self._last_token
        if token is None:
            return
        self._last_token = None
        self.executor.join(token)

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
            # Not inside a capture (or nothing to slice): reduce inline on the
            # main stream, but only once the comm stream is clear -- see
            # `drain`.
            self.drain()
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
        self._last_token = tokens[-1]
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
    # The interceptor does not go through CudaCommunicator's dispatch chain --
    # it needs a collective it can place on a named stream -- so it picks a
    # backend itself. Backends nobody has analysed on a side stream still
    # switch the whole feature off; `ca_comm` has been worked through (see the
    # module docstring) and is driven directly.
    def _live(attribute: str) -> Any:
        other = getattr(communicator, attribute, None)
        if other is not None and not getattr(other, "disabled", False):
            return other
        return None

    unanalysed = next((a for a in UNANALYSED_BACKENDS if _live(a) is not None), None)
    ca = _live("ca_comm")
    hostshm = _live("hostshm_comm")
    backend, reason = select_split_backend(
        settings,
        ca_active=ca is not None,
        hostshm_active=hostshm is not None,
        unanalysed=unanalysed,
    )
    if backend is None:
        logger.warning("GLM5 prefill overlap stays off: %s.", reason)
        return None
    logger.info(
        "GLM5 prefill overlap: split collectives go through %s (%s); "
        "anything that backend declines falls through to PyNccl.",
        backend.upper(),
        reason,
    )
    # A dedicated, low-priority-agnostic side stream. The collectives are the
    # long pole, so it is created with default priority and simply kept busy.
    comm_stream = torch.cuda.Stream()
    executor = CudaOverlapExecutor(
        comm_stream, nccl, backend=backend, ca=ca, hostshm=hostshm
    )
    return PrefillOverlapRegion(settings, executor, tp_group.all_reduce)


def maybe_open_region(
    *,
    num_tokens: int,
    mhc: bool,
    sequence_parallel: bool,
    allow_cross_layer: bool = True,
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
    _REGION.cross_layer = settings.cross_layer and allow_cross_layer
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
        counts = getattr(region.executor, "slice_backends", None)
        logger.info(
            "GLM5 prefill overlap: %d all-reduces intercepted, %d issued "
            "asynchronously as %d slices%s%s",
            region.submitted,
            region.async_submitted,
            region.slices,
            " (degraded)" if region.degraded else "",
            ""
            if counts is None
            else (
                "; slices by backend, cumulative: "
                + ", ".join(f"{k}={v}" for k, v in counts.items())
            ),
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
