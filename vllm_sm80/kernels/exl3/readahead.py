# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-cache readahead of the EXL3 checkpoint bytes the loader copies next
(VLLM_EXL3_FAST_LOAD).

The loader copies each packed weight and plain tensor from the memory-mapped
checkpoint; a copy that touches pages not yet in the page cache waits on the
kernel's fault read-around, one 128 KiB read at a time per faulting thread. A
rank of the tensor-split routed experts copies a column chunk of every gate
and up trellis, a short run of bytes in every row, so it faults in the whole
tensor that way. A CheckpointReadahead thread instead asks the kernel to read
the spans of the next loads (posix_fadvise WILLNEED in pieces the kernel reads
whole) up to a window ahead of the loader, in the loader's order, so the
copies find their pages cached. It changes no copy: the loader reads the same
bytes from the same mapped files with or without it, so the loaded tensors
are byte-identical.
"""

import os
import threading
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

# How far the readahead runs ahead of the loader, in bytes of planned spans.
WINDOW_BYTES = 4 << 30


def readahead_piece(path: Path) -> int:
    """The most bytes one posix_fadvise(WILLNEED) call reads on the block
    device holding ``path``: max(read_ahead_kb, max_sectors_kb) of its queue
    (the kernel's force_page_cache_ra limit); 128 KiB when sysfs does not say."""
    dev = os.stat(path).st_dev
    base = Path(f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}")
    for queue in (base / "queue", base / ".." / "queue"):
        try:
            return 1024 * max(
                int((queue / name).read_text())
                for name in ("read_ahead_kb", "max_sectors_kb")
            )
        except (OSError, ValueError):
            continue
    return 128 << 10


def coalesce(
    extents: list[tuple[Path, int, int]], gap: int
) -> list[tuple[Path, int, int]]:
    """``extents`` sorted by file and offset, with spans of one file closer than
    ``gap`` bytes merged."""
    merged: list[tuple[Path, int, int]] = []
    for path, start, end in sorted(extents):
        if merged and merged[-1][0] == path and start - merged[-1][2] < gap:
            merged[-1] = (path, merged[-1][1], max(end, merged[-1][2]))
        else:
            merged.append((path, start, end))
    return merged


class CheckpointReadahead:
    """A thread reading into the page cache the checkpoint spans of planned
    loads, in plan order, at most ``window`` bytes ahead of the loads done.

    ``plan``: per load, in the order the loader runs them, the (file, first
    byte, end byte) spans it copies from. The loader calls ``loaded()`` after
    each. A context manager: leaving it stops and joins the thread."""

    def __init__(
        self, plan: list[list[tuple[Path, int, int]]], window: int = WINDOW_BYTES
    ):
        self.window = window
        paths = {path for spans in plan for path, _, _ in spans}
        self.pieces = {path: readahead_piece(path) for path in paths}
        self.plan = [coalesce(spans, 1 << 16) for spans in plan]
        # Bytes planned before each load.
        self.before = [0]
        for spans in self.plan:
            size = sum(end - start for _, start, end in spans)
            self.before.append(self.before[-1] + size)
        self.done = 0
        self.stopped = False
        self.condition = threading.Condition()
        self.thread = threading.Thread(
            target=self._run, name="exl3-readahead", daemon=True
        )

    def __enter__(self) -> "CheckpointReadahead":
        logger.info(
            "EXL3 load readahead (VLLM_EXL3_FAST_LOAD): %.1f GiB of checkpoint "
            "spans in %d loads read into the page cache up to %.1f GiB ahead of "
            "the loader; the loaded bytes are unchanged.",
            self.before[-1] / 2**30,
            len(self.plan),
            self.window / 2**30,
        )
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()

    def loaded(self) -> None:
        with self.condition:
            self.done += 1
            self.condition.notify_all()

    def _run(self) -> None:
        files: dict[Path, int] = {}
        try:
            for index, spans in enumerate(self.plan):
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.stopped
                        or self.before[index] - self.before[min(self.done, index)]
                        < self.window
                    )
                    if self.stopped:
                        return
                    if index < self.done:
                        # The loader got there first.
                        continue
                for path, start, end in spans:
                    if path not in files:
                        files[path] = os.open(path, os.O_RDONLY)
                    piece = self.pieces[path]
                    for offset in range(start, end, piece):
                        os.posix_fadvise(
                            files[path],
                            offset,
                            min(piece, end - offset),
                            os.POSIX_FADV_WILLNEED,
                        )
        except OSError as error:
            # Only a hint: the loader reads the spans itself.
            logger.warning("EXL3 load readahead stopped: %s", error)
        finally:
            for fd in files.values():
                os.close(fd)
