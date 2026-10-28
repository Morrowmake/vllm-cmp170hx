# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-ordered torch profiler trace export (``VLLM_SM80_SERIAL_TRACE_EXPORT``).

Stopping a torch profiler session turns the collected activity into a trace in
host memory, writes it out and keeps the results until the next session
starts. Every worker of an engine stops at the same moment, so the host holds
every rank's trace at once.

With the switch set, the workers of one engine stop their sessions one after
another, in the order of their position in the engine (pipeline stage, prefill
context rank, tensor-parallel rank). A worker waits until the worker before it
has left a done-marker in the trace directory; the marker is written only after
that worker has written its trace, dropped the stopped profiler with its
results and returned the freed heap pages to the system (glibc
``malloc_trim``). So the host holds one rank's trace at a time. During the stop
torch's temporary uncompressed JSON (a gzipped trace is written uncompressed
first) goes into the trace directory instead of the temporary directory, which
may be in memory (tmpfs).

Only the stop the stop_profile request makes is ordered: it runs in the
workers' profile RPC, which the engine waits on without a timeout, so a worker
waits for the one before it as long as that worker's export takes (a warning
every ``LOG_S`` names the marker it waits for). A session the profiler stops
itself after its iteration limit stops inside a model step, under the
engine's step timeout: that stop is not ordered, nothing else changes, and a
warning says so. Neither is a stop at shutdown.

Markers are named by the engine (the workers' parent process), the session
number and the position; the next worker removes the marker it waited for, the
last worker writes none. A process that is no model-parallel worker (the
front-end profiler) is not ordered; its results are still released. Off by
default: nothing is patched.
"""

import ctypes
import ctypes.util
import gc
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

FLAG = "VLLM_SM80_SERIAL_TRACE_EXPORT"
LOG_S = 60.0
POLL_S = 0.05


def enabled() -> bool:
    import vllm.envs as envs

    return bool(getattr(envs, FLAG))


def install() -> bool:
    """Wrap ``TorchProfilerWrapper._stop`` (and ``step`` and ``shutdown``,
    which mark the stops not to order) when the switch is set."""
    if not enabled():
        return False
    from vllm.profiler.wrapper import TorchProfilerWrapper

    if getattr(TorchProfilerWrapper, "_sm80_serial_trace_export", False):
        return True
    stop = TorchProfilerWrapper._stop
    step = TorchProfilerWrapper.step
    shutdown = TorchProfilerWrapper.shutdown

    def _stop(self) -> None:
        from vllm.config.profiler import _is_uri_path

        session = getattr(self, "_sm80_export_session", 0)
        self._sm80_export_session = session + 1
        path = getattr(self, "_sm80_stop_path", None)
        if path is not None:
            if path == "step":
                logger.warning(
                    "Serial trace export: the profiler stopped itself after "
                    "max_iterations inside a model step; this export is not "
                    "ordered (profile with max_iterations = 0 and stop with "
                    "stop_profile)"
                )
            stop(self)
            return
        directory = self._torch_profiler_trace_dir
        local = bool(directory) and not _is_uri_path(directory)
        where = engine_position() if local else None
        if where is not None and where[0] > 0:
            wait_for(marker(directory, session, where[0] - 1))
        try:
            with temp_dir(directory if local else None):
                stop(self)
        finally:
            release(self)
            if where is not None and where[0] < where[1] - 1:
                mark(marker(directory, session, where[0]))

    def _step(self) -> None:
        self._sm80_stop_path = "step"
        try:
            step(self)
        finally:
            self._sm80_stop_path = None

    def _shutdown(self) -> None:
        self._sm80_stop_path = "shutdown"
        try:
            shutdown(self)
        finally:
            self._sm80_stop_path = None

    TorchProfilerWrapper._stop = _stop
    TorchProfilerWrapper.step = _step
    TorchProfilerWrapper.shutdown = _shutdown
    TorchProfilerWrapper._sm80_serial_trace_export = True
    logger.warning_once(
        "Serial trace export on: the workers of an engine stop their torch "
        "profiler sessions one at a time inside stop_profile (%s=1)",
        FLAG,
    )
    return True


def engine_position() -> tuple[int, int] | None:
    """(position, workers) of this worker in its engine: pipeline stage, then
    prefill context rank, then tensor-parallel rank. None outside a
    model-parallel worker."""
    from vllm.distributed import parallel_state as ps

    if not ps.model_parallel_is_initialized():
        return None
    tp, pp = ps.get_tp_group(), ps.get_pp_group()
    pcp_rank, pcp_size = 0, 1
    if ps._PCP is not None:
        pcp_rank, pcp_size = ps._PCP.rank_in_group, ps._PCP.world_size
    position = (pp.rank_in_group * pcp_size + pcp_rank) * tp.world_size
    return position + tp.rank_in_group, pp.world_size * pcp_size * tp.world_size


def marker(directory: str, session: int, position: int) -> Path:
    """The done-marker of ``position``'s ``session`` in this engine."""
    name = f".trace-export.{os.getppid()}.{session}.{position}.done"
    return Path(directory) / name


def wait_for(path: Path) -> None:
    """Wait for the previous worker's marker, however long its export takes,
    and remove it."""
    start = time.monotonic()
    logged = start
    while not path.exists():
        now = time.monotonic()
        if now - logged >= LOG_S:
            logger.warning(
                "Serial trace export: waiting %.0f s for %s",
                now - start,
                path,
            )
            logged = now
        time.sleep(POLL_S)
    path.unlink(missing_ok=True)


def mark(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@contextmanager
def temp_dir(directory: str | None) -> Iterator[None]:
    """Temporary files of the block go into ``directory`` (None: unchanged)."""
    if directory is None:
        yield
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    saved = tempfile.tempdir
    tempfile.tempdir = str(directory)
    try:
        yield
    finally:
        tempfile.tempdir = saved


def release(wrapper: Any) -> None:
    """Drop the stopped profiler (its results) and return freed heap pages to
    the system. The next session's start creates a new profiler."""
    wrapper.__dict__.pop("profiler", None)
    gc.collect()
    name = ctypes.util.find_library("c")
    if name is None:
        return
    libc = ctypes.CDLL(name)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
