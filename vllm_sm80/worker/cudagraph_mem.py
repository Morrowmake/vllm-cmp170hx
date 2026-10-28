# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph memory estimate as the bytes the real capture keeps
(VLLM_CUDAGRAPH_ESTIMATE_SPLIT).

vLLM's estimate (``cudagraph_utils.profile_cudagraph_memory``) is the drop in
free device memory over one profiling ``capture_model()``. That capture is the
first time attention and the capture warmups run at all, so the drop also holds
one-time costs (allocator cache of the warmup transients, lazily created
buffers and driver state) that the real capture, after the KV cache is sized,
does not pay again. With the flag set, the profiling run is split:

* ``pool``: bytes of the segments of the throwaway graph pool at the end of the
  capture -- what the real capture's pool holds for the same capture sizes;
* ``objects``: device memory outside the torch allocator that the teardown of
  the profiling graphs gives back (the instantiated graphs);
* ``buffers``: torch blocks outside the pool created during the capture and
  live at its end (persistent buffers a capture creates, per-stream
  workspaces); a real capture may create them again;
* ``held``: the drop in free memory from before the minimal KV cache is built
  to after the teardown -- whatever profiling leaves allocated for good
  (one-time driver state, buffers that outlive the teardown);
* ``cache``: the allocator cache the capture warmups left, released after the
  capture (vLLM counts it; it is free memory).

The profiling capture captures every FULL graph. vLLM captures only the two
largest and extrapolates the rest linearly from the second one's cost, which
assumes every smaller graph costs the same. It does not: on DeepSeek-V4.1 a
graph size's cost depends on the kernels and buffers its batch size selects
(FULL_DECODE_ONLY: 0.55 GiB extrapolated against a 0.32 GiB pool at sizes
1-8; 0.68 GiB against 1.04 GiB at the DSpark verify sizes 6-48). With every
graph captured the pool is measured, not extrapolated.

The estimate is ``pool + objects + buffers + held``. ``buffers`` that outlive
the teardown are
counted twice (in ``buffers`` and ``held``), so a buffer the real capture
creates again is covered either way. After the real capture the allocator
cache is released and what the capture keeps is logged against the estimate.

After the teardown one more line attributes ``held``: the torch segments
that profiling left reserved (live blocks, the free space those blocks keep
from being released, and the CUDA stream each segment belongs to) and the
rest, outside torch. With ``VLLM_CUDAGRAPH_ESTIMATE_SPLIT=2`` the allocation
history is recorded for the length of the profiling run and the live blocks
behind ``held`` are listed by their vLLM call stack (diagnostics: the
recording slows the profiling capture down).

The sm_80 tuning table's ``cudagraph`` / ``capture_max_split_mb`` value
(default 0, off) sets the allocator's ``max_split_size_mb`` for the length of
both captures, as the worker already does for ``profile_run``: blocks above
the limit are never split, so a small long-lived allocation made during a
capture cannot keep a large free segment from being released.

The worker and model runner subclasses that call in here are in
``graph_mem_worker``; the fork platform always selects that worker in place of
vLLM's GPU worker, and without the flag nothing here runs.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_sm80 import hot_swap

logger = init_logger(__name__)

_MiB = float(1 << 20)

UPSTREAM_WORKER = "vllm.v1.worker.gpu_worker.Worker"
WORKER = "vllm_sm80.worker.graph_mem_worker.GraphMemWorker"


def enabled() -> bool:
    return envs.VLLM_CUDAGRAPH_ESTIMATE_SPLIT >= 1


def tracing() -> bool:
    return envs.VLLM_CUDAGRAPH_ESTIMATE_SPLIT >= 2


def select_worker(parallel_config: Any) -> None:
    """Run the fork worker (``graph_mem_worker``) in place of vLLM's GPU
    worker (called by the fork platform's ``check_and_update_config``); its
    runner features are no-ops with their switches off. Another worker is
    kept: the greedy target argmax (``VLLM_SM80_SHARD_ARGMAX``, or its hot
    pair ``shard_argmax``) refuses it, the estimate split only warns."""
    if parallel_config.worker_cls == UPSTREAM_WORKER:
        parallel_config.worker_cls = WORKER
    if parallel_config.worker_cls == WORKER:
        return
    if bool(envs.VLLM_SM80_SHARD_ARGMAX) or hot_swap.listed("shard_argmax"):
        raise ValueError(
            "VLLM_SM80_SHARD_ARGMAX: the greedy target "
            f"argmax needs the fork GPU worker, not {parallel_config.worker_cls}"
        )
    if enabled():
        logger.warning(
            "VLLM_CUDAGRAPH_ESTIMATE_SPLIT=1 is set but the worker is %s, not "
            "vLLM's GPU worker; the CUDA-graph memory estimate stays vLLM's.",
            parallel_config.worker_cls,
        )


@dataclass
class _Window:
    """Readings of one profiling capture."""

    start_blocks: frozenset[tuple[int, int]] | None = None
    measured: int | None = None
    pool: int = 0
    buffers: int = 0
    cache: int = 0
    free_end: int = 0
    reserved_end: int = 0
    graphs: int = 0
    full_graphs: int = 0


_window: _Window | None = None
_estimate: int | None = None


def _free() -> int:
    return int(torch.accelerator.get_memory_info()[0])


def _reserved() -> int:
    return int(torch.accelerator.memory_reserved())


def _release() -> int:
    """Synchronize, empty the allocator cache; return the bytes it freed."""
    torch.accelerator.synchronize()
    before = _free()
    torch.accelerator.empty_cache()
    return max(_free() - before, 0)


def _segments() -> list[dict[str, Any]]:
    device = torch.accelerator.current_device_index()
    return [
        seg
        for seg in torch.cuda.memory_snapshot(include_traces=False)
        if seg.get("device", device) == device
    ]


def _graph_pool() -> Any:
    """The global graph pool; the profiling run points it at its throwaway
    pool for the length of its capture."""
    from vllm.platforms import current_platform

    return current_platform.get_global_graph_pool()


def _in_pool(segment: dict[str, Any], pool: Any) -> bool:
    return tuple(segment.get("segment_pool_id", ())) == tuple(pool)


def _active_blocks(
    segments: list[dict[str, Any]], pool: Any = None
) -> dict[tuple[int, int], int]:
    """Active blocks ``(address, size) -> size`` outside ``pool``."""
    out: dict[tuple[int, int], int] = {}
    for seg in segments:
        if pool is not None and _in_pool(seg, pool):
            continue
        for block in seg.get("blocks", ()):
            if block.get("state") != "inactive":
                size = int(block["size"])
                out[(int(block["address"]), size)] = size
    return out


def _count_graphs() -> int:
    """CUDA graphs held by the piecewise wrappers: every graph segment of
    every breakable capture, and every torch-compiled piecewise graph."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    count = 0
    for wrapper in list(BreakableCUDAGraphWrapper._all_instances):
        for entry in wrapper.entries.values():
            if entry.capture is not None:
                count += entry.capture.num_graphs
    for wrapper in list(CUDAGraphWrapper._all_instances):
        for entry in wrapper.concrete_cudagraph_entries.values():
            if entry.cudagraph is not None:
                count += 1
    return count


def measure_full_graphs(manager: Any) -> None:
    """Inside the profiling run, before its capture: capture every FULL graph
    of ``manager`` instead of vLLM's sample of the largest two, and take no
    per-graph samples, so vLLM's extrapolation adds nothing and its return
    value is the measured drop."""
    if _window is None or manager is None:
        return
    from vllm.config.compilation import CUDAGraphMode

    manager._max_full_descs_to_capture = None
    manager._capture_mem_samples = None
    _window.full_graphs = len(manager._capture_descs.get(CUDAGraphMode.FULL, ()))


def _window_start(w: _Window) -> None:
    torch.accelerator.synchronize()
    w.start_blocks = frozenset(_active_blocks(_segments()))


def _window_end(w: _Window, pool: Any, measured: int) -> None:
    assert w.start_blocks is not None
    w.cache += _release()
    w.free_end = _free()
    w.reserved_end = _reserved()
    segments = _segments()
    w.pool = sum(int(seg["total_size"]) for seg in segments if _in_pool(seg, pool))
    w.buffers = sum(
        size
        for key, size in _active_blocks(segments, pool).items()
        if key not in w.start_blocks
    )
    w.measured = int(measured)
    w.graphs = _count_graphs()


def capture(capture_fn: Callable[..., int], profile_only: bool) -> int:
    """``capture_fn(profile_only=...)`` (vLLM's ``capture_model``). Inside the
    profiling run, take the readings around it; for the real capture, release
    the allocator cache after it and log what it keeps against the estimate.
    Returns vLLM's value unchanged."""
    w = _window
    if profile_only:
        if w is None:
            with _capture_max_split():
                return capture_fn(profile_only=True)
        _window_start(w)
        with _capture_max_split():
            measured = int(capture_fn(profile_only=True))
        _window_end(w, _graph_pool(), measured)
        return measured
    _release()
    before = _free()
    with _capture_max_split():
        result = capture_fn(profile_only=False)
    cache = _release()
    kept = before - _free()
    graphs = _count_graphs()
    if _estimate is None:
        logger.info(
            "CUDA graph capture keeps %.1f MiB after releasing %.1f MiB of "
            "allocator cache (vLLM reports %.1f MiB, %d graph segments); no "
            "estimate was made",
            kept / _MiB,
            cache / _MiB,
            result / _MiB,
            graphs,
        )
        return result
    log = logger.info if kept <= _estimate else logger.warning
    log(
        "CUDA graph capture keeps %.1f MiB after releasing %.1f MiB of allocator "
        "cache (vLLM reports %.1f MiB, %d graph segments); estimate %.1f MiB: %s",
        kept / _MiB,
        cache / _MiB,
        result / _MiB,
        graphs,
        _estimate / _MiB,
        "within the reserve" if kept <= _estimate else "OVER the reserve",
    )
    return result


def profile(profile_fn: Callable[[Any], int], runner: Any) -> int:
    """``profile_fn(runner)`` (vLLM's profiling run, which calls
    ``runner.capture_model(profile_only=True)`` and so :func:`capture`); the
    bytes the real capture keeps instead of the profiling capture's drop."""
    global _window, _estimate
    gc.collect()
    _release()
    free0 = _free()
    reserved0 = _reserved()
    start_segments = _segments()
    trace = tracing() and _record_history(True)
    _window = window = _Window()
    try:
        upstream = int(profile_fn(runner))
    finally:
        _window = None
        if trace and window.measured is None:
            _record_history(False)
    if window.measured is None:
        # No capture ran (no graphs to capture): vLLM's value stands.
        _estimate = upstream
        return upstream
    _release()
    free1 = _free()
    reserved1 = _reserved()
    end_segments = _segments()
    if trace:
        _record_history(False)
    held = max(free0 - free1, 0)
    teardown_freed = free1 - window.free_end
    objects = max(teardown_freed - (window.reserved_end - reserved1), 0)
    # Every FULL graph was captured (measure_full_graphs): vLLM's return value
    # is the measured drop, with no extrapolated term.
    assert upstream == window.measured, (upstream, window.measured)
    estimate = window.pool + objects + window.buffers + held
    logger.info(
        "CUDA graph memory estimate split: graph pool %.1f MiB, graph objects "
        "%.1f MiB, capture buffers %.1f MiB, held after teardown (cold "
        "one-time) %.1f MiB; estimate %.1f MiB; profiling capture drop %.1f "
        "MiB with all %d FULL graphs captured, allocator cache %.1f MiB",
        window.pool / _MiB,
        objects / _MiB,
        window.buffers / _MiB,
        held / _MiB,
        estimate / _MiB,
        upstream / _MiB,
        window.full_graphs,
        window.cache / _MiB,
    )
    _log_held(
        free0 - free1,
        reserved1 - reserved0,
        start_segments,
        end_segments,
        trace,
        window.graphs,
    )
    _estimate = estimate
    return estimate


# ---------------------------------------------------------------------------
# Diagnostics: what profiling leaves allocated, and the capture split limit
# ---------------------------------------------------------------------------

_TOP = 12
_FRAMES = 3


def _record_history(on: bool) -> bool:
    """Start (python stacks of allocations) or stop the allocator's history."""
    try:
        if on:
            torch.cuda.memory._record_memory_history(
                enabled="state", context="alloc", stacks="python"
            )
        else:
            torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        logger.warning("CUDA allocation history unavailable: %s", e)
        return False
    return True


def _stack(block: dict[str, Any]) -> str:
    """The innermost vLLM frames of a block's allocation stack."""
    frames = [
        f"{f['filename'].rsplit('/vllm/', 1)[-1]}:{f['line']} {f['name']}"
        for f in block.get("frames", ())
        if "/vllm/" in f.get("filename", "")
        and "sm80/worker/cudagraph_mem" not in f.get("filename", "")
    ]
    return " < ".join(frames[:_FRAMES]) if frames else "(no vLLM frame)"


def _log_held(
    held: int,
    torch_growth: int,
    start_segments: list[dict[str, Any]],
    end_segments: list[dict[str, Any]],
    trace: bool,
    graphs: int = 0,
) -> None:
    """Attribute the bytes profiling leaves allocated: new torch segments
    (their live blocks, the free space those keep, their stream) and the rest
    outside torch; with ``trace``, the live blocks by allocation stack."""
    old = {int(seg["address"]) for seg in start_segments}
    new = [seg for seg in end_segments if int(seg["address"]) not in old]
    by_stream: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    live_total = 0
    for seg in new:
        total = int(seg["total_size"])
        live = [b for b in seg.get("blocks", ()) if b.get("state") != "inactive"]
        live_bytes = sum(int(b["size"]) for b in live)
        live_total += live_bytes
        row = by_stream[int(seg.get("stream", 0))]
        row[0] += 1
        row[1] += total
        row[2] += live_bytes
        # The segment is charged to its largest live block's stack.
        owner = max(live, key=lambda b: int(b["size"]), default=None)
        for block in live:
            g = groups[_stack(block) if trace else "-"]
            g[0] += 1
            g[1] += int(block["size"])
            if block is owner:
                g[2] += total
    reserved_new = sum(int(seg["total_size"]) for seg in new)
    logger.info(
        "CUDA graph memory held after teardown %.1f MiB: torch reserved +%.1f MiB "
        "(%d new segments %.1f MiB: live blocks %.1f MiB, free inside them "
        "%.1f MiB; by stream %s), outside torch %.1f MiB (%d graph segments "
        "captured: %.2f MiB each)",
        held / _MiB,
        torch_growth / _MiB,
        len(new),
        reserved_new / _MiB,
        live_total / _MiB,
        (reserved_new - live_total) / _MiB,
        ", ".join(
            f"{stream:#x}: {n} seg {total / _MiB:.1f} MiB live {live / _MiB:.1f}"
            for stream, (n, total, live) in sorted(
                by_stream.items(), key=lambda kv: -kv[1][1]
            )
        )
        or "none",
        (held - torch_growth) / _MiB,
        graphs,
        (held - torch_growth) / _MiB / max(graphs, 1),
    )
    if not trace:
        return
    ranked = sorted(groups.items(), key=lambda kv: -(kv[1][2] or kv[1][1]))
    for stack, (count, live, segs) in ranked[:_TOP]:
        logger.info(
            "CUDA graph memory held block: %d x, live %.1f MiB, segments %.1f "
            "MiB: %s",
            count,
            live / _MiB,
            segs / _MiB,
            stack,
        )


@contextmanager
def _capture_max_split() -> Iterator[None]:
    """``max_split_size_mb`` = the tuning table's capture_max_split_mb for the
    length of a capture (0, the default: unchanged), restoring the settings on
    exit."""
    from vllm_sm80 import tuning

    limit = tuning.lookup("cudagraph", "capture_max_split_mb", {}, 0)
    if limit <= 0 or torch.cuda.memory.get_allocator_backend() != "native":
        yield
        return
    get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if get_settings is not None:
        original = get_settings()
    else:
        original = torch.cuda.memory._snapshot()["allocator_settings"][
            "PYTORCH_CUDA_ALLOC_CONF"
        ]
    # The setter resets the other options unless they are included.
    prefix = original.strip().rstrip(",")
    settings = f"max_split_size_mb:{limit}"
    if prefix:
        settings = f"{prefix},{settings}"
    try:
        torch._C._accelerator_setAllocatorSettings(settings)
        yield
    finally:
        torch._C._accelerator_setAllocatorSettings(original)
