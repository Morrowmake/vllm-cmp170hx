# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution trace beside each torch profiler trace (``VLLM_SM80_EXECUTION_TRACE``).

With the switch set, every ``TorchProfilerWrapper`` session also runs torch's
``ExecutionTraceObserver``: the profiler starts and stops it with the kineto
trace (schedule included), and when the kineto trace has been written the
observer is finalised and its file moved beside it with the same stem:
``<stem>.et.json`` for ``<stem>.pt.trace.json`` and ``<stem>.et.json.gz`` for
``<stem>.pt.trace.json.gz`` (``<stem>`` for an uncompressed trace keeps its
``.pt.trace``, as ``Path.stem`` gives it). Each execution-trace node carries
the record function id (``rf_id``) of the kineto event it belongs to and the
tensor and storage ids of its inputs and outputs.

The observer sees ops the host executes. Ops replayed inside a CUDA graph
never run on the host, so a replayed step contributes only its launch; the
ops inside a graph are recorded only when the capture itself runs inside a
profiled session (``capture_torch_profiler``). Eager steps and the eager
pieces between piecewise graphs are recorded op by op.

Off by default: nothing is patched and no file is written.
"""

import gzip
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

FLAG = "VLLM_SM80_EXECUTION_TRACE"
_KINETO = ".pt.trace.json"
_PART = ".et.json.part"


def enabled() -> bool:
    import vllm.envs as envs

    return bool(getattr(envs, FLAG))


def install() -> bool:
    """Wrap ``TorchProfilerWrapper`` start/stop when the switch is set."""
    if not enabled():
        return False
    from vllm.profiler.wrapper import TorchProfilerWrapper

    if getattr(TorchProfilerWrapper, "_sm80_execution_trace", False):
        return True
    start = TorchProfilerWrapper._start
    stop = TorchProfilerWrapper._stop

    def _start(self) -> None:
        kwargs = self._profiler_kwargs
        observer, handler = _session(self, kwargs["on_trace_ready"])
        self._sm80_et_observer = observer
        self._profiler_kwargs = {
            **kwargs,
            "execution_trace_observer": observer,
            "on_trace_ready": handler,
        }
        try:
            start(self)
        finally:
            self._profiler_kwargs = kwargs

    def _stop(self) -> None:
        try:
            stop(self)
        finally:
            # A session stopped before its trace was ready writes no kineto
            # trace: drop its execution trace too.
            observer = getattr(self, "_sm80_et_observer", None)
            self._sm80_et_observer = None
            if observer is not None and observer.is_registered:
                part = observer.get_output_file_path()
                observer.unregister_callback()
                if part:
                    Path(part).unlink(missing_ok=True)

    TorchProfilerWrapper._start = _start
    TorchProfilerWrapper._stop = _stop
    TorchProfilerWrapper._sm80_execution_trace = True
    logger.warning_once(
        "Execution trace on: each torch profiler trace gets <stem>.et.json "
        "beside it (%s=1)",
        FLAG,
    )
    return True


def et_path(kineto: Path) -> Path:
    """The execution-trace path beside a kineto trace file."""
    name = kineto.name
    if name.endswith(_KINETO + ".gz"):
        return kineto.with_name(name[: -len(_KINETO + ".gz")] + ".et.json.gz")
    return kineto.with_name(kineto.stem + ".et.json")


def _traces(directory: Path, worker_name: str) -> set[Path]:
    if not directory.is_dir():
        return set()
    return {
        p
        for p in directory.iterdir()
        if p.name.startswith(worker_name + ".")
        and (p.name.endswith(_KINETO) or p.name.endswith(_KINETO + ".gz"))
    }


def _session(wrapper: Any, on_trace_ready: Callable[[Any], None]):
    """A registered observer writing a part file in the trace directory, and
    the trace handler that runs ``on_trace_ready`` and then finalises the
    observer beside the trace it wrote."""
    import torch

    directory = Path(wrapper._torch_profiler_trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    worker_name = wrapper._worker_name
    part = directory / f"{worker_name}.{os.getpid()}{_PART}"
    observer = torch.profiler.ExecutionTraceObserver().register_callback(str(part))

    def handler(prof: Any) -> None:
        before = _traces(directory, worker_name)
        on_trace_ready(prof)
        observer.unregister_callback()
        new = sorted(_traces(directory, worker_name) - before)
        if len(new) != 1:
            logger.warning(
                "Execution trace: %d new traces of %s in %s; left at %s",
                len(new),
                worker_name,
                directory,
                part,
            )
            return
        target = et_path(new[0])
        if target.name.endswith(".gz"):
            with open(part, "rb") as src, gzip.open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            part.unlink()
        else:
            os.replace(part, target)
        logger.info("Execution trace written to %s", target)

    return observer, handler
