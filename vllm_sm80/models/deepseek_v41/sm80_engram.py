# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram embedding storage on sm_80: SSD rows, a bounded host row cache and
resident scales.

A DeepSeek V4.1 Engram table holds ~384M rows of 256 E4M3 bytes (~92 GiB)
plus one E8M0 byte per 32 values (~2.9 GiB). The EXL3 checkpoint carries no
``engram.embed.*``; the tables are read from the base checkpoint's
safetensors shards in the directory named by the model config field
``engram_table_dir`` (through its ``model.safetensors.index.json``).

Each TP rank owns whole hash heads, as upstream ``ParallelEngramEmbedding``
shards them, i.e. one contiguous row range ``[vocab_start, vocab_end)``:

- its scales are read once into host memory of their exact size, past the
  page cache;
- its values stay in the file (placement ``ssd``) and are read on demand at
  absolute int64 byte offsets through a bounded set-associative LRU host row
  cache, pre-faulted at startup and optionally filled there by one
  sequential pass over the file (``VLLM_ENGRAM_ROW_CACHE_PRELOAD``), or are
  read whole into host memory of their exact size (placement ``ram``), past
  the page cache, after a startup check that the host (and every memory
  limit of the process's cgroups) can hold them, which refuses to start
  rather than fall back;
- with ``VLLM_ENGRAM_SEGMENT`` (placement ``ram`` only) the scales and
  values are not read: the rank maps its rows of the files a table segment
  holder (``vllm_sm80.table_segment``) already holds in sealed memory,
  read-only; a missing holder, or a file other than the one it read, is
  refused at startup;
- with ``VLLM_ENGRAM_DIRECT_READ`` the rows the cache misses are read with
  O_DIRECT through aligned 4 KiB spans, so they leave no pages in the page
  cache; a file system that refuses O_DIRECT is refused at startup, never
  read buffered instead;
- a lookup stages the requested rows' values and scales in pinned host
  memory, copies them to the device and dequantizes them there exactly as
  the reference (``ops/sm80_engram_dequant.py``).

The resident scales and values are read only by the host gather into the
pinned staging, so they are neither pinned nor registered with CUDA: the
caching pinned allocator rounds a block up to a power of two (a 22.9 GiB
value share would take 32 GiB), and registration buys nothing for host
reads. They are anonymous mappings of their exact size, advised to use
transparent huge pages, which the random row gathers walk with fewer misses;
attached ones are read-only shared mappings of the holder's sealed memory
files, populated at startup and advised the same way (shared memory takes
huge pages only where the host's ``shmem_enabled`` allows them).

The host part of a lookup (owned mask, scales gather, cache probe, reads of
the missed rows, cache insert, values gather) is one native call without the
GIL (csrc/exl3/engram_fetch.cpp, ``EngramHostFetch``). On CUDA it is queued
as a host function on the lookup's stream between the copy of the ids to
the host and the copy of the rows to the device, so the host thread never
waits for the ids and the reads overlap decoder compute. Staging reuse is
ordered on the device by two events: the next lookup's copies wait for the
previous host-to-device copy and for the previous dequant.

The module registers no parameter or buffer, so no loader looks for the
table in the checkpoint.

Lookups can be captured into a CUDA graph (FULL graphs, where the upstream
Engram's ``_start_prefetch`` runs inside the capture). With full graphs a
table has one fixed staging, host and device, as large as the eager one, so
every token count the runner captures fits (the capture sizes and the
spec-decode counts rounded to the decode query length alike, all within
``max_num_batched_tokens``). A captured lookup stages its ids on the device,
then queues with fixed arguments the copy of the ids to the fixed host ids,
the native fetch of its row count as a host-function node on the fixed host
staging, the copies of its rows to the fixed device staging and the
dequant. A replay therefore fetches that replay's ids; graph launches on one
stream run one after another, so the graphs of every count share the
staging and a replay never overwrites it before the previous one has read
it. Lookups outside a capture, or from an eager break of a breakable graph,
use the eager staging above.

A row that cannot be read (after the retries) fails only the requests that
use it, as the EXL3 PLE rows do (``vllm_sm80/failed_inputs.py``): the
owning rank stages it poisoned, an E4M3 NaN as its first value and zeros
after, which survives the TP gather of the rows. After the gather every rank
marks the tokens holding a NaN first value as failed, zeros those tokens'
rows, and reports them through ``take_failed_tokens``; the runner finishes
their requests with an error. The failure is logged when the finished
fetches are taken (a later lookup, or the runner's per-step take of the
failed tokens for captured lookups). Only startup and the table locator
raise.
"""

import atexit
import errno
import functools
import json
import mmap
import os
import socket
import struct
import time
import warnings
import weakref
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

ENGRAM_QUANT_BLOCK = 32
# Read attempts after the first for a failed or truncated row read (1 ms
# backoff, doubling), as the PLE reader's default.
READ_RETRIES = 3
# Threads that read the rows not in the page cache.
READ_THREADS = 8
# Associativity of the host row cache.
CACHE_WAYS = 8
# Host fetches queued or finished but not yet taken, per table.
HOST_FETCH_SLOTS = 4
# Chunk of the sequential startup reads (scales, ram placement, row-cache
# preload), and the offset and length alignment of their direct reads.
_PRELOAD_CHUNK_BYTES = 32 << 20
_DIRECT_ALIGN = 4096
_MAX_HEADER_BYTES = 100 << 20
GiB = 1 << 30
# Host memory a worker still takes after its Engram tables are built (CUDA
# graphs, runtime buffers): DeepSeek V4.1 Flash at TP4 grew 5.8 GiB over its
# four workers after model construction. The ram placement keeps it free.
RAM_PLACEMENT_MARGIN_PER_RANK = 2 * GiB
# EngramHostFetch.counters(), in order (csrc/exl3/engram_fetch.cpp).
NATIVE_COUNTERS = (
    "lookups",
    "rows_requested",
    "rows_unowned",
    "rows_unique",
    "cache_hits",
    "cache_misses",
    "cache_inserts",
    "cache_evictions",
    "rows_failed",
    "page_cache_rows",
    "storage_rows",
    "read_ns",
    "fetch_ns",
)
# Host time of the lookups themselves, counted in Python.
_LOOKUP_COUNTERS = ("gpu_wait_ns", "host_ns")


# ---------------------------------------------------------------------------
# Table files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedStaging:
    """The pinned host staging of captured fetches, and the CPU counter of
    their failed rows the host functions add to."""

    ids: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor
    failed_total: torch.Tensor

    @property
    def nbytes(self) -> int:
        return self.ids.nbytes + self.values.nbytes + self.scales.nbytes


@dataclass(frozen=True)
class EngramTableFiles:
    """Where one Engram table's values and scales live (absolute offsets)."""

    values_path: str
    values_offset: int
    scales_path: str
    scales_offset: int
    rows: int
    dim: int


def read_safetensors_header(path: str | os.PathLike) -> tuple[dict, int]:
    """Return a safetensors file's header and the absolute offset of its data."""
    with open(path, "rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if length > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: safetensors header of {length} bytes")
        text = file.read(length)
        if len(text) != length:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(text), 8 + length


def locate_engram_table(
    table_dir: str | os.PathLike, layer_id: int, rows: int, dim: int
) -> EngramTableFiles:
    """Find and validate ``layers.<layer_id>.engram.embed.{weight,scale}``."""
    root = Path(table_dir)
    index_path = root / "model.safetensors.index.json"
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Engram table index {index_path} unreadable: {exc}") from exc
    found = {}
    for suffix, dtype, width in (
        ("weight", "F8_E4M3", dim),
        ("scale", "F8_E8M0", dim // ENGRAM_QUANT_BLOCK),
    ):
        name = f"layers.{layer_id}.engram.embed.{suffix}"
        if name not in weight_map:
            raise ValueError(f"{index_path} lists no {name}")
        path = root / weight_map[name]
        header, data_start = read_safetensors_header(path)
        info = header.get(name)
        if info is None:
            raise ValueError(f"{path} holds no {name}")
        begin, end = info["data_offsets"]
        if (
            info["dtype"] != dtype
            or list(info["shape"]) != [rows, width]
            or end - begin != rows * width
        ):
            raise ValueError(
                f"{name} in {path} is {info['dtype']} {info['shape']}, "
                f"expected {dtype} {[rows, width]}"
            )
        offset = data_start + begin
        if os.stat(path).st_size < offset + rows * width:
            raise ValueError(f"{path} is truncated before the end of {name}")
        found[suffix] = (str(path), offset)
    return EngramTableFiles(
        values_path=found["weight"][0],
        values_offset=found["weight"][1],
        scales_path=found["scale"][0],
        scales_offset=found["scale"][1],
        rows=rows,
        dim=dim,
    )


def engram_owner_range(
    head_sizes: Sequence[int], tp_rank: int, tp_size: int
) -> tuple[int, int, int, int]:
    """(heads per rank, first head, vocab_start, vocab_end) of a TP rank, as
    upstream ``ParallelEngramEmbedding`` shards whole hash heads."""
    heads = len(head_sizes)
    part = -(-heads // tp_size)
    if (tp_size - 1) * part >= heads:
        raise ValueError(
            f"Engram sharding leaves ranks without hash heads: {heads} heads "
            f"over {tp_size} shards"
        )
    head_start = tp_rank * part
    return (
        part,
        head_start,
        sum(head_sizes[:head_start]),
        sum(head_sizes[: head_start + part]),
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def format_engram_telemetry(counts: dict[str, int]) -> str:
    """One line of counters: rows, cache, page-cache and storage bytes, and
    host time (native fetch, its reads, wait for the GPU, whole lookup)."""
    mib = 1 << 20
    hits, misses = counts["cache_hits"], counts["cache_misses"]
    return (
        f"lookups {counts['lookups']} rows {counts['rows_requested']} "
        f"(unowned {counts['rows_unowned']}, unique {counts['rows_unique']}) "
        f"cache hits {hits} misses {misses} "
        f"({100.0 * hits / max(hits + misses, 1):.1f}% hit) "
        f"evictions {counts['cache_evictions']}; "
        f"failed rows {counts['rows_failed']}; read MiB page cache "
        f"{counts['page_cache_bytes'] / mib:.1f} storage "
        f"{counts['storage_bytes'] / mib:.1f}; host ms fetch "
        f"{counts['fetch_ns'] / 1e6:.1f} read {counts['read_ns'] / 1e6:.1f} "
        f"gpu wait {counts['gpu_wait_ns'] / 1e6:.1f} "
        f"lookup {counts['host_ns'] / 1e6:.1f}"
    )


# The live sm_80 Engram tables of this process.
_TABLES: "weakref.WeakSet[Sm80EngramEmbedding]" = weakref.WeakSet()


def engram_telemetry() -> dict[str, dict[str, int]]:
    """The counters of every live sm_80 Engram table in this process, by
    label (``layer <id>``)."""
    return {table.label: table.telemetry() for table in list(_TABLES)}


# ---------------------------------------------------------------------------
# Owned rows of one table
# ---------------------------------------------------------------------------


def _host_empty(shape: tuple[int, ...], dtype: torch.dtype, pin: bool) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin)


def _resident_empty(rows: int, width: int) -> torch.Tensor:
    """A [rows, width] uint8 host tensor of exactly rows * width bytes: an
    anonymous mapping advised to use transparent huge pages (best effort).
    The tensor keeps the mapping alive."""
    nbytes = rows * width
    if nbytes == 0:
        return torch.empty((rows, width), dtype=torch.uint8)
    mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        mapping.madvise(mmap.MADV_HUGEPAGE)
    except (AttributeError, OSError) as error:
        logger.debug("Engram tables without transparent huge pages: %s", error)
    table = torch.frombuffer(mapping, dtype=torch.uint8).view(rows, width)
    if table.untyped_storage().nbytes() != nbytes:
        raise RuntimeError("Engram host table is not of its exact size")
    return table


def _open_preload(path: str) -> tuple[int, bool]:
    """A descriptor for the sequential startup reads and whether they bypass
    the page cache: O_DIRECT, or buffered (each chunk's pages dropped after
    use) where the file system refuses O_DIRECT."""
    try:
        return os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT), True
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
    return os.open(path, os.O_RDONLY | os.O_CLOEXEC), False


def _open_direct(path: str) -> int:
    """A descriptor for the direct reads of missed rows (the native fetch
    reads aligned spans through it). A file system that refuses O_DIRECT is
    refused: these reads never fall back to the page cache."""
    try:
        return os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
            raise
        raise ValueError(
            f"VLLM_ENGRAM_DIRECT_READ=1: the file system of {path} refuses "
            f"O_DIRECT ({os.strerror(error.errno)}); move the Engram tables to "
            "one that supports it or unset VLLM_ENGRAM_DIRECT_READ for "
            "buffered reads"
        ) from error


def _read_at_least(fd: int, view: memoryview, offset: int, need: int) -> None:
    """Read from ``offset`` into ``view`` until at least ``need`` bytes are
    in; a direct read ends short only at the end of the file."""
    done = 0
    while done < need:
        count = os.preadv(fd, [view[done:]], offset + done)
        if count == 0:
            raise EOFError(f"Truncated Engram data at byte {offset + done}")
        done += count


def _stream_extent(
    path: str,
    offset: int,
    nbytes: int,
    consume: Callable[[int, torch.Tensor], bool],
    chunk_bytes: int = _PRELOAD_CHUNK_BYTES,
) -> tuple[int, bool]:
    """Read bytes [offset, offset + nbytes) of ``path`` in one sequential
    pass of ``chunk_bytes`` pieces past the page cache (``_open_preload``),
    reading the next piece while ``consume(begin, data)`` takes the current
    one: ``begin`` is the piece's offset from ``offset`` and ``data`` a
    uint8 view valid only during the call. Stops early when ``consume``
    returns True. Returns (bytes read, whether the reads were direct)."""
    if chunk_bytes <= 0 or chunk_bytes % _DIRECT_ALIGN:
        raise ValueError("Engram startup read chunks must be aligned blocks")
    if nbytes <= 0:
        return 0, True
    # Aligned reads of a piece take at most one block more.
    buffers = [mmap.mmap(-1, chunk_bytes + _DIRECT_ALIGN) for _ in range(2)]
    fd, direct = _open_preload(path)

    def read(buffer: mmap.mmap, begin: int) -> tuple[int, int, int, int]:
        end = min(begin + chunk_bytes, nbytes)
        first = offset + begin - (offset + begin) % _DIRECT_ALIGN
        length = -(-(offset + end - first) // _DIRECT_ALIGN) * _DIRECT_ALIGN
        _read_at_least(fd, memoryview(buffer)[:length], first, offset + end - first)
        if not direct:
            os.posix_fadvise(fd, first, length, os.POSIX_FADV_DONTNEED)
        return begin, end, offset + begin - first, length

    read_bytes = 0
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(read, buffers[0], 0)
            turn = 0
            while pending is not None:
                begin, end, skip, length = pending.result()
                pending = None
                if end < nbytes:
                    pending = pool.submit(read, buffers[1 - turn], end)
                read_bytes += length
                data = torch.frombuffer(
                    buffers[turn], dtype=torch.uint8, count=end - begin, offset=skip
                )
                if consume(begin, data):
                    break
                turn = 1 - turn
    finally:
        os.close(fd)
    return read_bytes, direct


def _read_resident(path: str, offset: int, dest: torch.Tensor) -> bool:
    """Fill ``dest`` (contiguous uint8) with the file bytes at ``offset`` in
    one sequential pass past the page cache; return whether it was direct."""
    flat = dest.view(-1)

    def store(begin: int, data: torch.Tensor) -> bool:
        flat[begin : begin + data.numel()].copy_(data)
        return False

    return _stream_extent(path, offset, flat.numel(), store)[1]


@dataclass(frozen=True)
class HostHeadroom:
    """What this process can still commit, and what limits it."""

    bytes: int
    # The binding limit: "MemAvailable" or the cgroup path with memory.max.
    limit: str
    # Host MemTotal and that cgroup's memory.max (0 for MemAvailable).
    host_total: int
    limit_bytes: int

    def describe(self) -> str:
        if not self.limit_bytes:
            return (
                f"{self.bytes / GiB:.2f} GiB available (host MemAvailable of "
                f"{self.host_total / GiB:.2f} GiB total)"
            )
        return (
            f"{self.bytes / GiB:.2f} GiB available under the memory.max of "
            f"{self.limit_bytes / GiB:.2f} GiB of cgroup {self.limit}, which "
            f"holds back {(self.host_total - self.limit_bytes) / GiB:.2f} GiB of "
            f"the host's {self.host_total / GiB:.2f} GiB"
        )


def host_memory_headroom(
    proc: str = "/proc", cgroup_root: str = "/sys/fs/cgroup"
) -> HostHeadroom:
    """What this process can still commit: the least of the host's
    MemAvailable and, for its cgroup (v2) and each ancestor with a
    memory.max, that limit less the group's use other than file pages
    (memory.current - file)."""
    with open(os.path.join(proc, "meminfo")) as file:
        fields = dict(line.split(":", 1) for line in file if ":" in line)
    total = int(fields["MemTotal"].split()[0]) * 1024
    best = HostHeadroom(
        int(fields["MemAvailable"].split()[0]) * 1024, "MemAvailable", total, 0
    )
    try:
        with open(os.path.join(proc, "self", "cgroup")) as file:
            unified = [
                line for line in file.read().splitlines() if line.startswith("0::")
            ]
    except OSError:
        unified = []
    if not unified:
        return best
    group = Path(cgroup_root) / unified[0][3:].lstrip("/")
    root = Path(cgroup_root)
    while True:
        try:
            limit = (group / "memory.max").read_text().strip()
            if limit != "max":
                current = int((group / "memory.current").read_text())
                stat = dict(
                    line.split()
                    for line in (group / "memory.stat").read_text().splitlines()
                )
                left = int(limit) - (current - int(stat.get("file", 0)))
                if left < best.bytes:
                    name = "/" + str(group.relative_to(root)).lstrip(".")
                    best = HostHeadroom(left, name, total, int(limit))
        except (OSError, ValueError):
            pass
        if group == root or root not in group.parents:
            break
        group = group.parent
    return best


def check_ram_placement_fits(
    need_bytes: int, gather: Callable[[object], list] | None = None
) -> None:
    """Refuse to start when the hosts cannot hold one more ram-placement
    table of every rank. Each rank gives its ``need_bytes`` and its host
    memory headroom, measured before any rank allocates; ``gather`` collects
    them from the ranks building the table together (default: this process
    only). On each host the needs plus RAM_PLACEMENT_MARGIN_PER_RANK per rank
    must fit the smallest headroom its ranks measured. Every rank reaches
    the same verdict; the refusal names the shortfall and the binding limit."""
    mine = (socket.gethostname(), need_bytes, host_memory_headroom())
    entries = gather(mine) if gather is not None else [mine]
    hosts: dict[str, tuple[int, HostHeadroom, int]] = {}
    for host, need, headroom in entries:
        total, smallest, ranks = hosts.get(host, (0, headroom, 0))
        if headroom.bytes < smallest.bytes:
            smallest = headroom
        hosts[host] = (total + need, smallest, ranks + 1)
    for host, (total, smallest, ranks) in sorted(hosts.items()):
        margin = ranks * RAM_PLACEMENT_MARGIN_PER_RANK
        short = total + margin - smallest.bytes
        if short > 0:
            raise ValueError(
                f"VLLM_ENGRAM_PLACEMENT=ram: host {host} is {short / GiB:.2f} GiB "
                f"short for the Engram table: its {ranks} rank(s) need "
                f"{total / GiB:.2f} GiB plus {margin / GiB:.0f} GiB margin, and "
                f"{smallest.describe()}. Free host memory, raise that limit "
                "(hold back less) or use placement ssd"
            )


def _gather_over_tp(obj: object) -> list:
    """``obj`` of every rank of this rank's TP group (CPU group)."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size == 1:
        return [obj]
    out: list = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def _attached(attachment, path: str, offset: int, rows: int, width: int) -> torch.Tensor:
    """A read-only [rows, width] uint8 view of bytes at ``offset`` of the
    held image of ``path``. The tensor keeps the mapping alive; nothing
    writes it (the native fetch only reads its tables)."""
    if rows * width == 0:
        return torch.empty((rows, width), dtype=torch.uint8)
    mapping, skip = attachment.map(path, offset, rows * width)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        table = torch.frombuffer(
            mapping, dtype=torch.uint8, count=rows * width, offset=skip
        )
    return table.view(rows, width)


class EngramRowStore:
    """One rank's rows of one Engram table on the host, fetched by the
    native ``EngramHostFetch`` into this store's staging.

    Scales are always resident. Values come from the file through the row
    cache (``ssd``) or from a resident copy (``ram``); resident data is of
    its exact size and read past the page cache, or, with ``segment`` (the
    socket of a table segment holder, ``ram`` only), mapped read-only from
    the holder's images of the files. A fetch fills the
    staging for global row ids: unowned ids get zero values and the scale
    byte of 1.0, which dequantize to +0 as the reference's mask does; rows
    that cannot be read are poisoned. With ``direct_read`` (``ssd`` only)
    the native fetch reads missed rows with O_DIRECT; startup refuses a
    file system where that fails.
    """

    def __init__(
        self,
        files: EngramTableFiles,
        owner_lo: int,
        owner_hi: int,
        *,
        placement: str,
        cache_bytes: int,
        capacity: int,
        pin: bool = False,
        direct_read: bool = False,
        segment: str = "",
    ):
        from vllm_sm80.models.qwen4_exp.exl3_ple import (
            _read_error,
            page_cache_residency,
        )
        from vllm_sm80.kernels.exl3.ops import engram_host_fetch

        if placement not in ("ssd", "ram"):
            raise ValueError(f"Engram placement must be ssd or ram, not {placement!r}")
        if direct_read and placement != "ssd":
            raise ValueError("Engram direct reads apply to placement ssd only")
        if segment and placement != "ram":
            raise ValueError("VLLM_ENGRAM_SEGMENT applies to placement ram only")
        if not 0 <= owner_lo <= owner_hi <= files.rows:
            raise ValueError("Engram owner range exceeds the table rows")
        if files.dim % ENGRAM_QUANT_BLOCK:
            raise ValueError("Engram row width must be a multiple of 32")
        self._read_error = _read_error
        self._residency = page_cache_residency
        self.placement = placement
        self.path = files.values_path
        self.base = files.values_offset
        self.direct_read = direct_read
        self.owner_lo, self.owner_hi = owner_lo, owner_hi
        self.row_bytes = files.dim
        self.scale_bytes = files.dim // ENGRAM_QUANT_BLOCK
        rows = owner_hi - owner_lo
        self.pin = pin
        self.fixed: FixedStaging | None = None
        # The failed rows of fixed fetches already taken.
        self._fixed_failed_seen = 0
        self.fetcher = None
        # Fetches queued or finished but not yet taken. Counted here, so a
        # lookup calls into the fetch once (each call may hand off the GIL).
        self.in_flight = 0
        # The table segment lease (its connection), when attached.
        self.attachment = None
        self.fd = os.open(files.values_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            # Rows are read at random; readahead would only fetch unused rows.
            os.posix_fadvise(
                self.fd, self.extent_offset, self.extent_bytes, os.POSIX_FADV_RANDOM
            )
            self.table: torch.Tensor | None = None
            scales_at = files.scales_offset + owner_lo * self.scale_bytes
            if segment:
                from vllm_sm80.table_segment import Attachment

                self.attachment = Attachment(
                    segment,
                    sorted({files.values_path, files.scales_path}),
                    log=logger.info,
                )
                self.scales = _attached(
                    self.attachment, files.scales_path, scales_at, rows,
                    self.scale_bytes,
                )
                self.table = _attached(
                    self.attachment, self.path, self.extent_offset, rows,
                    self.row_bytes,
                )
                self.direct_load = True
            else:
                self.scales = _resident_empty(rows, self.scale_bytes)
                self.direct_load = _read_resident(
                    files.scales_path, scales_at, self.scales
                )
                if placement == "ram":
                    self.table = _resident_empty(rows, self.row_bytes)
                    self.direct_load &= _read_resident(
                        self.path, self.extent_offset, self.table
                    )
            self.ids = _host_empty((capacity,), torch.int64, pin)
            self.values = _host_empty((capacity, self.row_bytes), torch.uint8, pin)
            self.scales_out = _host_empty(
                (capacity, self.scale_bytes), torch.uint8, pin
            )
            # The fetch keeps its own duplicate; an O_DIRECT descriptor
            # makes its reads direct.
            fetch_fd = _open_direct(self.path) if direct_read else self.fd
            try:
                self.fetcher = engram_host_fetch(
                    fetch_fd,
                    self.base,
                    self.row_bytes,
                    owner_lo,
                    owner_hi,
                    self.scales,
                    self.table
                    if self.table is not None
                    else torch.empty(0, dtype=torch.uint8, device="cpu"),
                    self.ids,
                    self.values,
                    self.scales_out,
                    cache_bytes if placement == "ssd" else 0,
                    CACHE_WAYS,
                    READ_RETRIES,
                    READ_THREADS,
                    HOST_FETCH_SLOTS,
                )
            finally:
                if fetch_fd != self.fd:
                    os.close(fetch_fd)
        except BaseException:
            self.close()
            raise

    @property
    def extent_offset(self) -> int:
        return self.base + self.owner_lo * self.row_bytes

    @property
    def extent_bytes(self) -> int:
        return (self.owner_hi - self.owner_lo) * self.row_bytes

    @property
    def cache_bytes(self) -> int:
        return self.fetcher.cache_bytes()

    @property
    def host_bytes(self) -> int:
        """Host memory of this store: scales, resident table or cache, and
        the staging."""
        resident = self.scales.nbytes + self.cache_bytes
        if self.table is not None:
            resident += self.table.nbytes
        staging = self.ids.nbytes + self.values.nbytes + self.scales_out.nbytes
        fixed = self.fixed.nbytes if self.fixed is not None else 0
        return resident + staging + fixed

    def add_fixed(self, capacity: int) -> FixedStaging:
        """The fixed host staging for captured fetches of up to
        ``capacity`` rows; added once, before any fetch is queued."""
        ids = _host_empty((capacity,), torch.int64, self.pin)
        values = _host_empty((capacity, self.row_bytes), torch.uint8, self.pin)
        scales = _host_empty((capacity, self.scale_bytes), torch.uint8, self.pin)
        failed_total = torch.zeros(1, dtype=torch.int64, device="cpu")
        self.fetcher.add_fixed(ids, values, scales, failed_total)
        self.fixed = FixedStaging(ids, values, scales, failed_total)
        self._fixed_failed_total = failed_total.numpy()
        return self.fixed

    def page_cache_residency(self) -> tuple[int, int]:
        """(resident, total) page-cache pages of the owned value extent;
        sampling does not change residency."""
        return self._residency(self.fd, self.extent_offset, self.extent_bytes)

    def prefault(self) -> int:
        """Touch every page of the row cache; return the time in ns."""
        return self.fetcher.prefault()

    def preload(self, chunk_bytes: int = _PRELOAD_CHUNK_BYTES) -> tuple[int, int, bool]:
        """Fill the empty row-cache ways from one sequential pass over the
        owned value extent, before any fetch, stopping once every way holds
        a row. Returns (rows stored, bytes read, whether the reads bypassed
        the page cache).

        A row goes to the first empty way of its set, in file order, so a
        cache smaller than the extent holds an even spread of it. The reads
        use O_DIRECT; where the file system refuses it they are buffered and
        each chunk's pages are dropped from the page cache after it is read,
        including pages that were resident before. The next chunk is read
        while the native call stores the current one. The fetch counters do
        not change.
        """
        capacity = self.fetcher.cache_rows()
        rows = self.owner_hi - self.owner_lo
        if not capacity or not rows:
            return 0, 0, True
        if (
            chunk_bytes <= 0
            or chunk_bytes % _DIRECT_ALIGN
            or chunk_bytes % self.row_bytes
        ):
            raise ValueError("Engram preload chunks hold whole rows and aligned blocks")
        stored = 0

        def insert(begin: int, data: torch.Tensor) -> bool:
            nonlocal stored
            stored += self.fetcher.preload(data, begin // self.row_bytes)
            return stored == capacity

        read_bytes, direct = _stream_extent(
            self.path, self.extent_offset, self.extent_bytes, insert, chunk_bytes
        )
        return stored, read_bytes, direct

    def fetch(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch global row ids on this thread; return views of the staged
        values [n, row_bytes] and scales [n, scale_bytes]."""
        count = len(ids)
        self.ids.numpy()[:count] = ids
        self.run(count)
        return self.values.numpy()[:count], self.scales_out.numpy()[:count]

    def run(self, count: int) -> None:
        """Fetch the staged ids[0:count) on this thread (GIL released)."""
        self.fetcher.run(count)
        self.in_flight += 1
        self.take()

    def enqueue(self, count: int) -> None:
        """Queue the fetch of the staged ids[0:count) as a host function on
        the current CUDA stream."""
        self.fetcher.enqueue(count)
        self.in_flight += 1

    def take(self) -> int:
        """Log the finished fetches' failed rows (already poisoned, so their
        requests fail in their own step); return how many were taken."""
        results = self.fetcher.take().tolist()
        self.in_flight -= len(results)
        for _, count, status, failed, row, code, offset in results:
            if status:
                self._log_stopped(1, count)
            elif failed:
                self._log_failed(failed, row, code, offset)
        return len(results)

    def take_fixed(self) -> int:
        """Log the failed rows of the fixed (captured) fetches since the
        last call; never waits for a fetch in flight. Returns their count.

        The host functions add every failed row to a CPU counter, read here
        without calling into the native fetch (each call hands the GIL off),
        which is called only when that counter moved.
        """
        if self.fixed is None:
            return 0
        total = int(self._fixed_failed_total[0])
        if total == self._fixed_failed_seen:
            return 0
        self._fixed_failed_seen = total
        failed, row, code, offset, stopped = self.fetcher.take_fixed().tolist()
        if stopped:
            self._log_stopped(stopped, failed)
        elif failed:
            self._log_failed(failed, row, code, offset)
        return failed

    def _log_stopped(self, fetches: int, rows: int) -> None:
        logger.error(
            "sm_80 Engram table %s: %d host fetch(es) stopped on an unexpected "
            "error; their %d rows were poisoned and the requests using them "
            "will be finished with an error",
            self.path,
            fetches,
            rows,
        )

    def _log_failed(self, failed: int, row: int, code: int, offset: int) -> None:
        logger.error(
            "sm_80 Engram table %s: %d row read(s) failed after %d retries "
            "(first: row %d: %s); the requests using them will be finished "
            "with an error",
            self.path,
            failed,
            READ_RETRIES,
            row,
            self._read_error(code, offset),
        )

    def counters(self) -> dict[str, int]:
        """The native counters of every finished fetch."""
        return dict(zip(NATIVE_COUNTERS, self.fetcher.counters().tolist()))

    def reset_counters(self) -> None:
        self.fetcher.reset_counters()

    def close(self) -> None:
        if self.fetcher is not None:
            self.take()
            self.take_fixed()
            # The fetch keeps its own descriptor until it is dropped.
            self.fetcher = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        if self.attachment is not None:
            # Detach; the mappings stay valid while the tensors live.
            self.attachment.close()
            self.attachment = None


# ---------------------------------------------------------------------------
# The embedding module
# ---------------------------------------------------------------------------


class Sm80EngramEmbedding(nn.Module):
    """``ParallelEngramEmbedding`` for sm_80 with SSD rows (or the ram
    comparison path), no table parameters and the same lookup interface.

    The attributes the upstream ``Engram`` reads keep their meaning;
    ``cpu_offload`` is set so ``Engram`` runs lookups on its prefetch stream
    from its eager graph break.
    """

    cpu_offload = True
    dp_shared_memory = False
    use_thp = False
    dp_size = 1

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        files: EngramTableFiles,
        *,
        tp_rank: int,
        tp_size: int,
        max_tokens: int,
        placement: str,
        cache_bytes: int,
        direct_read: bool = False,
        segment: str = "",
        device: torch.device | str = "cpu",
        block_size: int = ENGRAM_QUANT_BLOCK,
        label: str = "engram",
        log_every: int = 0,
        graph_capture: bool = False,
    ) -> None:
        super().__init__()
        if block_size != ENGRAM_QUANT_BLOCK or dim % block_size:
            raise ValueError("Engram rows must hold 32-wide scale blocks")
        if (files.rows, files.dim) != (num_embeddings, dim):
            raise ValueError(
                f"Engram table is [{files.rows}, {files.dim}], the model expects "
                f"[{num_embeddings}, {dim}]"
            )
        if not head_sizes or sum(head_sizes) > num_embeddings:
            raise ValueError("Engram hash heads exceed the table rows")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = tp_size
        (
            self.part_n_hash_cols,
            self.head_start,
            self.vocab_start_idx,
            self.vocab_end_idx,
        ) = engram_owner_range(head_sizes, tp_rank, tp_size)
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.device = torch.device(device)
        self.label = label
        # Steps between telemetry log lines; 0 never logs.
        self.log_every = log_every
        self._since_log = 0
        self._lookup_counts = dict.fromkeys(_LOOKUP_COUNTERS, 0)
        self._logged = self.telemetry_fields()
        cuda = self.device.type == "cuda"
        self.capacity_rows = max_tokens * self.part_n_hash_cols
        self.store = EngramRowStore(
            files,
            self.vocab_start_idx,
            self.vocab_end_idx,
            placement=placement,
            cache_bytes=cache_bytes,
            capacity=self.capacity_rows,
            pin=cuda,
            direct_read=direct_read,
            segment=segment,
        )
        # Tokens of the last forward whose rows failed, on any rank.
        self._failed_tokens = torch.zeros(
            max_tokens, dtype=torch.bool, device=self.device
        )
        if cuda:
            cap, scale_dim = self.capacity_rows, dim // block_size
            self._dev_ids = torch.empty(cap, dtype=torch.int64, device=self.device)
            self._dev_values = torch.empty(
                cap, dim, dtype=torch.uint8, device=self.device
            )
            self._dev_scales = torch.empty(
                cap, scale_dim, dtype=torch.uint8, device=self.device
            )
            # The last host-to-device copy of the host staging finished.
            self._staged = torch.cuda.Event()
            # The last dequant finished reading the device staging.
            self._consumed = torch.cuda.Event()
        # With full CUDA graphs: the fixed staging, host (native) and device,
        # as large as the eager one. Every captured lookup's copies, host
        # fetch and dequant run on it on every replay.
        self._graph_staging: (
            tuple[FixedStaging, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None
        if graph_capture:
            cap = self.capacity_rows
            device_staging = (
                torch.empty(cap, dtype=torch.int64, device=self.device),
                torch.empty(cap, dim, dtype=torch.uint8, device=self.device),
                torch.empty(
                    cap, dim // block_size, dtype=torch.uint8, device=self.device
                ),
            )
            self._graph_staging = (self.store.add_fixed(cap), device_staging)

    @property
    def placement(self) -> str:
        return self.store.placement

    def host_bytes(self) -> int:
        """Host memory of this table: scales, cache or resident values, and
        the host staging."""
        return self.store.host_bytes

    def device_bytes(self) -> int:
        """Device memory of this table's staging (the tables use none)."""
        graphs = 0
        if self._graph_staging is not None:
            graphs = sum(tensor.nbytes for tensor in self._graph_staging[1])
        if self.device.type != "cuda":
            return graphs
        eager = self._dev_ids.nbytes + self._dev_values.nbytes + self._dev_scales.nbytes
        return eager + graphs

    @staticmethod
    def telemetry_fields() -> dict[str, int]:
        return dict.fromkeys(
            (*NATIVE_COUNTERS, "page_cache_bytes", "storage_bytes", *_LOOKUP_COUNTERS),
            0,
        )

    def telemetry(self) -> dict[str, int]:
        """Counters since start or the last :meth:`reset_telemetry`; fetches
        still queued are not counted yet."""
        counts = self.store.counters()
        counts["page_cache_bytes"] = counts["page_cache_rows"] * self.dim
        counts["storage_bytes"] = counts["storage_rows"] * self.dim
        counts.update(self._lookup_counts)
        return counts

    def reset_telemetry(self) -> None:
        self.store.reset_counters()
        self._lookup_counts = dict.fromkeys(_LOOKUP_COUNTERS, 0)
        self._logged = self.telemetry_fields()

    def log_telemetry(self, when: str = "") -> None:
        """Log the counters since start and since the last line."""
        total = self.telemetry()
        window = {name: total[name] - self._logged[name] for name in total}
        self._logged = total
        logger.info(
            "sm_80 Engram %s telemetry%s: %s; last %s",
            self.label,
            f" {when}" if when else "",
            format_engram_telemetry(total),
            format_engram_telemetry(window),
        )

    def _count_lookup(self, gpu_wait_ns: int, host_ns: int) -> None:
        self._lookup_counts["gpu_wait_ns"] += gpu_wait_ns
        self._lookup_counts["host_ns"] += host_ns

    def page_cache_residency(self) -> tuple[int, int]:
        """(resident, total) page-cache pages of this rank's value extent."""
        return self.store.page_cache_residency()

    def _stage_ids(
        self, indices: torch.Tensor, num_tokens: int, target: torch.Tensor
    ) -> int:
        """Write this rank's head columns of ``indices`` as int64 into the
        id staging ``target`` (-1 for padded heads); return the row count."""
        part = self.part_n_hash_cols
        count = num_tokens * part
        columns = indices[:, self.head_start : self.head_start + part]
        width = columns.shape[1]
        staged = target[:count].view(num_tokens, part)
        staged[:, :width].copy_(columns)
        if width < part:
            staged[:, width:].fill_(-1)
        return count

    def _make_room(self) -> int:
        """With every slot taken, take the finished fetches, waiting for the
        last one if none has finished. Returns the ns waited. Read failures
        are therefore logged up to HOST_FETCH_SLOTS lookups late; their
        requests already failed in their own step."""
        store = self.store
        if store.in_flight < HOST_FETCH_SLOTS or store.take():
            return 0
        start = time.perf_counter_ns()
        self._staged.synchronize()
        store.take()
        return time.perf_counter_ns() - start

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up this rank's heads of ``indices`` [T, n_hash_cols] into
        ``out`` [T, part_n_hash_cols, dim] BF16 on the current stream.

        ``background`` is accepted for the upstream interface; the dequant
        kernel is small either way.
        """
        from vllm_sm80.models.deepseek_v41.ops.sm80_engram_dequant import (
            dequant_engram_rows,
        )

        if out.dtype != torch.bfloat16:
            raise ValueError("sm_80 Engram lookups produce BF16 rows only")
        num_tokens = indices.shape[0]
        count = num_tokens * self.part_n_hash_cols
        if not count:
            return
        if count > self.capacity_rows:
            raise ValueError("Engram lookup exceeds the staging capacity")
        rows_out = out.view(count, self.dim)
        if self._capturing():
            self._captured_lookup(indices, num_tokens, count, rows_out)
            return
        store = self.store
        start = time.perf_counter_ns()
        if self.device.type != "cuda":
            self._stage_ids(indices, num_tokens, store.ids)
            store.run(count)
            dequant_engram_rows(
                store.values[:count], store.scales_out[:count], rows_out
            )
            self._count_lookup(0, time.perf_counter_ns() - start)
            return
        stream = torch.cuda.current_stream()
        gpu_wait = self._make_room()
        # The host staging is free once the previous copy has read it, the
        # device staging once the previous dequant has.
        stream.wait_event(self._staged)
        stream.wait_event(self._consumed)
        self._stage_ids(indices, num_tokens, self._dev_ids)
        store.ids[:count].copy_(self._dev_ids[:count], non_blocking=True)
        store.enqueue(count)
        dev_values = self._dev_values[:count]
        dev_scales = self._dev_scales[:count]
        dev_values.copy_(store.values[:count], non_blocking=True)
        dev_scales.copy_(store.scales_out[:count], non_blocking=True)
        self._staged.record(stream)
        dequant_engram_rows(dev_values, dev_scales, rows_out)
        self._consumed.record(stream)
        self._count_lookup(gpu_wait, time.perf_counter_ns() - start)

    def _capturing(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()

    def _captured_lookup(
        self,
        indices: torch.Tensor,
        num_tokens: int,
        count: int,
        rows_out: torch.Tensor,
    ) -> None:
        """The lookup inside a CUDA graph capture, on the fixed staging:
        every queued argument is fixed, so each replay fetches its own ids.
        The host time of a replay is not counted."""
        from vllm_sm80.models.deepseek_v41.ops.sm80_engram_dequant import (
            dequant_engram_rows,
        )

        if self._graph_staging is None:
            raise RuntimeError(
                "sm_80 Engram: a lookup is captured but the table was built "
                "without full CUDA graphs (no fixed staging)"
            )
        dev_ids, dev_values, dev_scales = self._graph_staging[1]
        self._stage_ids(indices, num_tokens, dev_ids)
        self.store.fetcher.enqueue_fixed(count, dev_ids, dev_values, dev_scales)
        dev_values, dev_scales = dev_values[:count], dev_scales[:count]
        dequant_engram_rows(dev_values, dev_scales, rows_out)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("sm_80 Engram tables are read through Engram.lookup")

    def mask_failed_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Zero the tokens of gathered rows [T, n_hash_cols, dim] that hold a
        poisoned row and remember them for :meth:`take_failed_tokens`.

        Runs on every rank after the TP gather, so each rank sees the rows
        any rank failed to read. Graph-capturable.
        """
        failed = rows[:, :, 0].isnan().any(-1)
        self._failed_tokens[: rows.shape[0]].copy_(failed)
        return rows.masked_fill(failed[:, None, None], 0)

    def take_failed_tokens(self) -> torch.Tensor:
        """The failed-token mask over the last forward's tokens (padded to
        the staging capacity), cleared once taken (vllm_sm80/failed_inputs.py).

        The runner takes it once per step, so this also logs the captured
        fetches' failed rows and, every ``log_every`` steps, the telemetry.
        """
        self.store.take_fixed()
        if self.log_every:
            self._since_log += 1
            if self._since_log >= self.log_every:
                self._since_log = 0
                self.log_telemetry()
        failed = self._failed_tokens.clone()
        self._failed_tokens.zero_()
        return failed

    def close(self) -> None:
        """Wait for queued fetches and staging copies, then close the table."""
        if self.device.type == "cuda":
            self._staged.synchronize()
        if self.log_every:
            self.log_telemetry("at close")
        self.store.close()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def engram_table_dir(model_config) -> str:
    """The base-checkpoint directory holding the Engram tables."""
    for config in (model_config.hf_config, model_config.hf_text_config):
        table_dir = getattr(config, "engram_table_dir", None)
        if table_dir:
            return str(table_dir)
    raise ValueError(
        "sm_80 Engram tables are read from the base checkpoint: set the model "
        'config field engram_table_dir (--hf-overrides \'{"engram_table_dir": '
        '"<dir>"}\') to the directory holding its model.safetensors.index.json '
        "and Engram shards"
    )


def captures_full_graphs(mode) -> bool:
    """Whether the cudagraph mode captures full graphs (FULL,
    FULL_DECODE_ONLY, FULL_AND_PIECEWISE): their Engram lookups need the
    fixed staging."""
    return mode is not None and mode.has_full_cudagraphs()


def preload_row_cache(store: EngramRowStore) -> str:
    """Fill a store's row cache from its file (:meth:`EngramRowStore.preload`);
    return the startup log's account of it."""
    start = time.perf_counter_ns()
    stored, read, direct = store.preload()
    seconds = (time.perf_counter_ns() - start) / 1e9
    rows = store.fetcher.cache_rows()
    return (
        f", preloaded {stored} of {rows} rows "
        f"({100.0 * stored / max(rows, 1):.1f}%) from {read / GiB:.2f} GiB "
        f"{'direct' if direct else 'buffered'} reads in {seconds:.2f} s"
    )


def build_sm80_engram_embedding(layout, layer_hash_index: int) -> Sm80EngramEmbedding:
    """The sm_80 Engram table of one Engram layer of the current model."""
    from vllm import envs
    from vllm.config import get_current_vllm_config
    from vllm.distributed import (
        get_engram_dp_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    vllm_config = get_current_vllm_config()
    engram_config = vllm_config.engram_config
    if engram_config is not None and (
        engram_config.dp_shared_memory or engram_config.use_thp
    ):
        raise ValueError(
            "sm_80 Engram tables support neither dp_shared_memory nor use_thp"
        )
    if get_engram_dp_size() > 1:
        raise NotImplementedError("sm_80 Engram tables are sharded over TP only")
    layer_id = layout.layer_ids[layer_hash_index]
    rows = layout.num_embeddings[layer_hash_index]
    files = locate_engram_table(
        engram_table_dir(vllm_config.model_config), layer_id, rows, layout.head_dim
    )
    placement = envs.VLLM_ENGRAM_PLACEMENT
    cache_gib = envs.VLLM_ENGRAM_ROW_CACHE_GIB
    if cache_gib < 0:
        raise ValueError("VLLM_ENGRAM_ROW_CACHE_GIB must be nonnegative")
    log_every = envs.VLLM_ENGRAM_TELEMETRY_STEPS
    if log_every < 0:
        raise ValueError("VLLM_ENGRAM_TELEMETRY_STEPS must be nonnegative")
    graph_capture = captures_full_graphs(vllm_config.compilation_config.cudagraph_mode)
    head_sizes = tuple(
        size for order in layout.primes[layer_hash_index] for size in order
    )
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    segment = envs.VLLM_ENGRAM_SEGMENT
    if placement == "ram" and not segment:
        # Every rank measures its host before any allocates this table.
        _, _, lo, hi = engram_owner_range(head_sizes, tp_rank, tp_size)
        check_ram_placement_fits(
            (hi - lo) * (layout.head_dim + layout.head_dim // ENGRAM_QUANT_BLOCK),
            _gather_over_tp,
        )
    table = Sm80EngramEmbedding(
        rows,
        layout.head_dim,
        head_sizes,
        files,
        tp_rank=tp_rank,
        tp_size=tp_size,
        max_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        placement=placement,
        cache_bytes=int(cache_gib * GiB) // len(layout.layer_ids),
        direct_read=placement == "ssd" and envs.VLLM_ENGRAM_DIRECT_READ,
        segment=segment,
        device=torch.device("cuda", torch.cuda.current_device()),
        label=f"layer {layer_id}",
        log_every=log_every,
        graph_capture=graph_capture,
    )
    _TABLES.add(table)
    if log_every:
        # Engine shutdown does not close modules: log the totals at exit.
        atexit.register(table.log_telemetry, "at exit")
    prefault_ns = table.store.prefault()
    preloaded = ""
    if placement == "ssd" and envs.VLLM_ENGRAM_ROW_CACHE_PRELOAD:
        preloaded = preload_row_cache(table.store)
    resident, total = table.page_cache_residency()
    if placement == "ram" and segment:
        logger.info_once(
            "VLLM_ENGRAM_SEGMENT on: the ram Engram tables are mapped from the "
            "table segment held at %s, not read from storage",
            segment,
        )
        values = (
            f"attached {table.store.table.nbytes / GiB:.2f} GiB from the table "
            f"segment {segment} (manifest {table.store.attachment.manifest[:12]})"
        )
    elif placement == "ram":
        values = (
            f"resident {table.store.table.nbytes / GiB:.2f} GiB, read "
            f"{'direct' if table.store.direct_load else 'buffered'}"
        )
    else:
        values = (
            f"row cache {table.store.cache_bytes / GiB:.2f} GiB, pre-faulted "
            f"in {prefault_ns / 1e9:.2f} s{preloaded}, missed rows read "
            f"{'direct' if table.store.direct_read else 'buffered'}"
        )
    logger.info(
        "sm_80 Engram layer %d (%s): placement=%s rows [%d, %d) heads [%d, %d); "
        "host %.2f GiB (scales %.2f GiB, values %s); device staging "
        "%.1f MiB (fixed graph staging %s); page cache %d/%d value pages "
        "resident",
        layer_id,
        files.values_path,
        placement,
        table.vocab_start_idx,
        table.vocab_end_idx,
        table.head_start,
        table.head_start + table.part_n_hash_cols,
        table.host_bytes() / GiB,
        table.store.scales.nbytes / GiB,
        values,
        table.device_bytes() / (1 << 20),
        "on" if graph_capture else "off",
        resident,
        total,
    )
    return table


def sm80_engram_embedding(create: Callable) -> Callable:
    """Decorates upstream ``Engram._create_embedding``: on sm_80 the layer's
    table is a :class:`Sm80EngramEmbedding`; elsewhere upstream's."""

    @functools.wraps(create)
    def build(engram, layout, layer_hash_index: int):
        from vllm_sm80.kernels._platform import _is_sm80

        if not _is_sm80():
            return create(engram, layout, layer_hash_index)
        table = build_sm80_engram_embedding(layout, layer_hash_index)
        mask_failed_after_gather(engram, table)
        return table

    return build


def mask_failed_after_gather(engram, table: Sm80EngramEmbedding) -> None:
    """Route ``engram.embed`` (the lookup's rows after the TP gather) through
    :meth:`Sm80EngramEmbedding.mask_failed_rows`."""
    gather = engram.embed

    def embed(hash_ids: torch.Tensor) -> torch.Tensor:
        return table.mask_failed_rows(gather(hash_ids))

    engram.embed = embed
