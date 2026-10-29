# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by AlexBrandes for CMP 170HX support; see repository history.
"""Bounded compressed PLE storage and selected-row decoding."""

import atexit
import bisect
import ctypes
import functools
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
)
from vllm.sm80.kernels.exl3.utils import (
    PLE_AUXILIARIES,
    TensorHeader,
    ple_layout,
    read_headers,
    read_metadata,
)
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


# mincore is called per window so its residency vector stays small.
_RESIDENCY_WINDOW = 1 << 30


@functools.cache
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
    )
    libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    return libc


def page_cache_residency(fd: int, start: int, length: int) -> tuple[int, int]:
    """Return (resident, total) page-cache pages of a file extent.

    Maps the extent without touching it and asks mincore, so sampling does
    not change residency.
    """
    libc = _libc()
    page = mmap.PAGESIZE
    first = start // page * page
    end = start + length
    resident = 0
    for window in range(first, end, _RESIDENCY_WINDOW):
        size = min(_RESIDENCY_WINDOW, end - window)
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, window)
        if address is None or address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap of the PLE table failed")
        try:
            vector = (ctypes.c_ubyte * ((size + page - 1) // page))()
            if libc.mincore(address, size, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore of the PLE table failed")
            resident += int((np.frombuffer(vector, dtype=np.uint8) & 1).sum())
        finally:
            libc.munmap(address, size)
    return resident, (end - first + page - 1) // page


def pread_exact_into(fd: int, target: memoryview, offset: int) -> None:
    """Fill a byte view without changing the descriptor's seek position.

    Partial reads continue; errors, EINTR included, and end of file are
    raised for the caller's bounded retry.
    """
    if offset < 0 or offset + len(target) > (1 << 63) - 1:
        raise ValueError("PLE file offset exceeds signed 64-bit range")
    done = 0
    while done < len(target):
        count = os.preadv(fd, [target[done:]], offset + done)
        if count == 0:
            raise EOFError(f"Truncated PLE data at byte {offset + done}")
        done += count


def _read_error(code: int, offset: int) -> BaseException:
    """The error pread_exact_into raises for a failure the native reader reports."""
    if code == 0:
        return EOFError(f"Truncated PLE data at byte {offset}")
    return OSError(code, os.strerror(code))


# The first retry waits 1 ms, each later one twice as long.
_READ_BACKOFF_S = 0.001
# An FP16 quiet NaN as a row's scale makes every decoded channel NaN.
_POISONED_SCALE = 0x7E00
# Host reads queued on the PLE stream or finished but not yet taken.
_HOST_READ_SLOTS = 4


def poison_rows(rows: np.ndarray, indices) -> None:
    """Mark staged packed rows that could not be read.

    The decoder turns them into NaN rows; NaN survives the cross-rank sum,
    so every rank sees which tokens failed, whichever rank owns the row.
    """
    rows[list(indices), 0] = _POISONED_SCALE


def copy_masking_failed(rows: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Copy [tokens, heads, dim] rows into output, zeroing poisoned tokens.

    Returns the per-token failure mask. Zero rows keep NaN out of the
    forward; the runner fails the requests that own the masked tokens.
    """
    failed = rows[..., 0].isnan().any(-1)
    output.copy_(rows.flatten(-2))
    output.masked_fill_(failed.unsqueeze(-1), 0)
    return failed


class Exl3PLETiming:
    """Host time of each SSD prefetch (VLLM_EXL3_PLE_TIMING): the wait for the
    token ids on the host, then the row reads; one log line per window."""

    WINDOW = 256

    def __init__(self, label: str):
        self.label = label
        self.samples: list[tuple[int, int, int]] = []

    def add(self, rows: int, wait_ns: int, read_ns: int) -> None:
        self.samples.append((rows, wait_ns, read_ns))
        if len(self.samples) >= self.WINDOW:
            self.flush()

    def flush(self) -> None:
        if not self.samples:
            return
        rows, wait, read = (np.array(c, dtype=np.float64) for c in zip(*self.samples))
        wait_us = np.percentile(wait / 1e3, [50, 90])
        read_us = np.percentile(read / 1e3, [50, 90])
        logger.info(
            "EXL3 PLE timing %s: %d prefetches, rows mean %.1f max %d; "
            "ids wait us p50 %.0f p90 %.0f max %.0f; "
            "row reads us p50 %.0f p90 %.0f max %.0f mean %.0f; us per row %.2f",
            self.label,
            len(rows),
            rows.mean(),
            rows.max(),
            wait_us[0],
            wait_us[1],
            wait.max() / 1e3,
            read_us[0],
            read_us[1],
            read.max() / 1e3,
            read.mean() / 1e3,
            read.sum() / 1e3 / max(rows.sum(), 1.0),
        )
        self.samples.clear()


class Exl3PLERowReader:
    """Read owned rows of a validated compressed PLE table into caller storage."""

    def __init__(
        self,
        path: str,
        key: str,
        owner_lo: int,
        owner_hi: int,
        retries: int | None = None,
    ):
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            if not key.endswith(".trellis"):
                raise ValueError("PLE source key must name the packed trellis")
            source = Path(path)
            self.layout = ple_layout(
                read_headers(source),
                read_metadata(source),
                key.removesuffix(".trellis"),
            )
            if not 0 <= owner_lo <= owner_hi <= self.layout.rows:
                raise ValueError("PLE owner range exceeds stored padded rows")
            self.owner_lo, self.owner_hi = owner_lo, owner_hi
            self.base = self.layout.trellis.offset
            self.row_bytes = self.layout.row_bytes
            if retries is None:
                retries = envs.VLLM_EXL3_PLE_READ_RETRIES
            self.retries = retries
            if self.retries < 0:
                raise ValueError("PLE read retries must be nonnegative")
            # Rows are read at random; readahead would only fetch unused rows.
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        except BaseException:
            self.close()
            raise

    def drop_page_cache(self) -> None:
        """Evict the table's clean, unmapped page-cache pages (no root needed)."""
        os.posix_fadvise(
            self.fd,
            self.base,
            self.layout.trellis.nbytes,
            os.POSIX_FADV_DONTNEED,
        )

    def page_cache_residency(self) -> tuple[int, int]:
        """Return (resident, total) page-cache pages of the table extent."""
        return page_cache_residency(self.fd, self.base, self.layout.trellis.nbytes)

    def read_exact(self, target: memoryview, offset: int) -> None:
        """Read with bounded retries and exponential backoff; raise the last
        error once retries are exhausted."""
        for attempt in range(self.retries + 1):
            try:
                pread_exact_into(self.fd, target, offset)
                return
            except (OSError, EOFError):
                if attempt == self.retries:
                    raise
                time.sleep(_READ_BACKOFF_S * 2**attempt)

    def read_tensor(self, header: TensorHeader) -> bytearray:
        data = bytearray(header.nbytes)
        self.read_exact(memoryview(data), header.offset)
        return data

    def owns(self, row: int) -> bool:
        if not self.owner_lo <= row < self.owner_hi:
            return False
        offsets = self.layout.head_offsets
        head = bisect.bisect_right(offsets, row) - 1
        return head >= 0 and row < offsets[head] + self.layout.head_vocab_sizes[head]

    def read_rows(
        self, ids, target: memoryview, begin: int = 0, end: int | None = None
    ) -> list[tuple[int, BaseException]]:
        """Preserve order/repetitions; mask only nonowned or padding rows.

        Returns (index, error) for owned rows that still failed after the
        retries; their bytes in target are unspecified. Other rows are read
        regardless, and the reader stays usable.
        """
        end = len(ids) if end is None else end
        if not 0 <= begin <= end <= len(ids) or len(target) < end * self.row_bytes:
            raise ValueError("PLE row destination is too small")
        zero = bytes(self.row_bytes)
        failed = []
        for index in range(begin, end):
            row = int(ids[index])
            dest = target[index * self.row_bytes : (index + 1) * self.row_bytes]
            if not self.owns(row):
                dest[:] = zero
                continue
            try:
                self.read_exact(dest, self.base + row * self.row_bytes)
            except (OSError, EOFError) as error:
                failed.append((index, error))
        return failed

    @functools.cached_property
    def _native_heads(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.layout.head_offsets, dtype=torch.int64),
            torch.tensor(self.layout.head_vocab_sizes, dtype=torch.int64),
        )

    def read_rows_native(
        self, ids: torch.Tensor, target: torch.Tensor, count: int, threads: int
    ) -> list[tuple[int, BaseException]]:
        """read_rows(ids, target, 0, count) in one native call without the GIL.

        Same rows and bytes in target; the failures are a subset of
        read_rows' (one extra read attempt for rows not in the page cache),
        with equal errors for rows both report. Rows in the page cache are
        read on the calling thread, the others on up
        to `threads` threads. ids is a 1-D CPU int64 tensor, target
        contiguous CPU memory.
        """
        from vllm.sm80.kernels.exl3.ops import ple_read_rows

        offsets, sizes = self._native_heads
        failed = ple_read_rows(
            self.fd,
            ids,
            target,
            count,
            self.base,
            self.row_bytes,
            self.owner_lo,
            self.owner_hi,
            offsets,
            sizes,
            self.retries,
            threads,
        )
        return [
            (index, _read_error(code, offset))
            for index, code, offset in failed.tolist()
        ]

    def host_reader(
        self, ids: torch.Tensor, rows: torch.Tensor, threads: int, slots: int
    ):
        """A PleHostReader (csrc/exl3/ple_read.cpp) of this table: reads
        ids[0:count) into rows as read_rows_native does, in a host function
        queued on a CUDA stream, and poisons the rows that fail. It keeps a
        duplicate of fd and the ids and rows tensors."""
        from vllm.sm80.kernels.exl3.ops import ple_host_reader

        offsets, sizes = self._native_heads
        return ple_host_reader(
            self.fd,
            ids,
            rows,
            self.base,
            self.row_bytes,
            self.owner_lo,
            self.owner_hi,
            offsets,
            sizes,
            self.retries,
            threads,
            slots,
        )

    def close(self) -> None:
        fd = getattr(self, "fd", -1)
        if fd >= 0:
            os.close(fd)
            self.fd = -1

    def __del__(self):
        self.close()


class Exl3PLESlot:
    """The PLE step state: one bounded staging lease whose reuse waits for the
    previous output copy."""

    def __init__(self, capacity, completion):
        self.capacity = capacity
        self.completion = completion
        self.state = "idle"

    def acquire(self, count, stream=None):
        """Wait for the previous output copy: on the host, or on `stream`
        when given (work queued there after this call waits instead)."""
        if self.state != "idle":
            raise RuntimeError(f"PLE prefetch slot is {self.state}")
        if not 0 <= count <= self.capacity:
            raise ValueError("PLE request exceeds configured staging capacity")
        self.state = "waiting"
        try:
            if stream is None:
                self.completion.synchronize()
            else:
                stream.wait_event(self.completion)
        except BaseException:
            self.state = "failed"
            raise
        self.state = "reading"

    def ready(self):
        if self.state != "reading":
            raise RuntimeError("PLE reads must complete before decode readiness")
        self.state = "ready"

    def release(self):
        if self.state != "ready":
            raise RuntimeError("PLE finalize has no matching prefetched rows")
        self.state = "idle"

    def fail(self):
        self.state = "failed"


class Exl3PLEEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Keep the table compressed independently of BF16 activations."""

    def __init__(self, quant_config=None):
        self.placement = envs.VLLM_EXL3_PLE_PLACEMENT
        if self.placement not in ("ssd", "ram", "device"):
            raise ValueError("EXL3 PLE placement must be ssd, ram or device")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("EXL3 PLE requires BF16 output activations")
        # A zero-sized storage marker satisfies the embedding module interface.
        # The loader joins the file extent instead of loading a dense parameter.
        layer.register_parameter(
            "weight",
            nn.Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
        )

    def process_weights_after_loading(self, layer):
        layer.check_ready()

    def embedding(self, layer, input_):
        raise RuntimeError("EXL3 PLE lookup requires start_prefetch and finalize")

    def dequantize(self, layer, embeddings, output_dtype):
        if output_dtype != torch.bfloat16 or embeddings.dtype != output_dtype:
            raise ValueError("EXL3 PLE decoder output must be BF16")
        return embeddings


class Exl3PLEEmbedding(Qwen4ExpPLEEmbedding):
    """One bounded slot, reused only after its graph-output copy completes."""

    supports_prefetch = True

    def __init__(self, *args, num_ngram_heads=1, max_total_tokens=0, **kwargs):
        super().__init__(
            *args,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            **kwargs,
        )
        if max_total_tokens <= 0:
            raise ValueError("EXL3 PLE needs a positive staging token capacity")
        if not self.reduce_results:
            # Sequence parallelism defers the ETP sum to a reduce-scatter, but
            # read-failure masking needs the summed rows of every token here.
            raise NotImplementedError(
                "EXL3 PLE does not support sequence parallelism (reduce_results=False)"
            )
        self.placement = self.embedding_method.placement
        self._heads = num_ngram_heads
        self._capacity = max_total_tokens * self.etp_data_parallel_size * self._heads
        self._output_dim = self._heads * self.embedding_dim
        self._reader = None
        self._pool = None
        self._slot: Exl3PLESlot | None = None
        self._failed_tokens: torch.Tensor | None = None
        self._staged_rows: torch.Tensor | None = None
        self._timing: Exl3PLETiming | None = None
        # SSD reads run at finalize, after the layers before this one queue.
        self.overlap_reads = envs.VLLM_EXL3_PLE_OVERLAP
        self._deferred: int | None = None
        # SSD rows are read by one native call instead of the thread pool.
        self.native_reads = envs.VLLM_EXL3_PLE_NATIVE_READ
        # SSD rows are read by a host function queued on the PLE stream.
        self.hostfn_reads = envs.VLLM_EXL3_PLE_HOSTFN_READ
        self._host_reader = None
        self._host_reads_in_flight = 0

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        raise RuntimeError("EXL3 PLE must never allocate a dense table")

    def check_ready(self):
        if self._slot is None:
            raise RuntimeError("EXL3 PLE file/auxiliary loader join is missing")

    def setup_file(
        self, path, key, head_offsets, head_vocab_sizes, layer_multipliers
    ) -> set[str]:
        """Load auxiliaries exactly once; return consumed source keys.

        The manifest must exclude these keys and the trellis from ordinary
        tensor iteration. Hash buffers belong to the parent n-gram module.
        """
        if self._reader is not None:
            raise RuntimeError("EXL3 PLE file was already set up")
        device = self.weight.device
        if device.type != "cuda" or any(
            t.device != device
            for t in (head_offsets, head_vocab_sizes, layer_multipliers)
        ):
            raise ValueError("PLE hash auxiliaries must be CUDA-resident")
        reader = Exl3PLERowReader(
            path,
            key,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )
        self._reader = reader
        try:
            layout = reader.layout
            if (
                layout.rows != self.org_vocab_size
                or layout.dim != self.embedding_dim
                or layout.heads != self._heads
            ):
                raise ValueError("PLE checkpoint rows/dim/heads differ from model")
            self._log_residency("at open")
            if envs.VLLM_EXL3_PLE_COLD:
                # Cold run: start from storage, so SSD runs can be labelled.
                reader.drop_page_cache()
                self._log_residency("after cold drop")
            tensors = {}
            for name, header in zip(PLE_AUXILIARIES, layout.auxiliaries):
                dtype = torch.float16 if header.dtype == "F16" else torch.int64
                tensors[name] = torch.frombuffer(
                    reader.read_tensor(header), dtype=dtype
                ).reshape(header.shape)
            for name, target in (
                ("head_offsets", head_offsets),
                ("head_vocab_sizes", head_vocab_sizes),
                ("layer_multipliers", layer_multipliers),
            ):
                if tensors[name].shape != target.shape:
                    raise ValueError(f"Invalid PLE auxiliary {name}")
            consumed = {key} | {header.name for header in layout.auxiliaries}
            head_offsets.copy_(tensors["head_offsets"])
            head_vocab_sizes.copy_(tensors["head_vocab_sizes"])
            layer_multipliers.copy_(tensors["layer_multipliers"])
            self.register_buffer("head_bias", tensors["head_bias"].to(device))
            self.head_offsets = head_offsets
            self.head_vocab_sizes = head_vocab_sizes
            self._stream = torch.cuda.Stream(device=device)
            self._ids_ready = torch.cuda.Event()
            self._ready = torch.cuda.Event()
            self._released = torch.cuda.Event()
            self._released.record(torch.cuda.current_stream())
            self._ids = torch.empty(self._capacity, dtype=torch.int64, device=device)
            self._packed = torch.empty(
                self._capacity, layout.words, dtype=torch.int16, device=device
            )
            self._decoded = torch.empty(
                self._capacity, layout.dim, dtype=torch.bfloat16, device=device
            )
            if self.placement == "ssd":
                self._host_ids = torch.empty(
                    self._capacity, dtype=torch.int64, device="cpu", pin_memory=True
                )
                self._host_rows = torch.empty(
                    self._capacity,
                    layout.words,
                    dtype=torch.int16,
                    device="cpu",
                    pin_memory=True,
                )
                self._host_ids_np = self._host_ids.numpy()
                self._host_rows_np = self._host_rows.numpy()
                self._host_bytes = memoryview(self._host_rows_np).cast("B")
                self._workers = min(8, os.cpu_count() or 1)
                self._pool = ThreadPoolExecutor(
                    max_workers=self._workers, thread_name_prefix="ple-read"
                )
                if envs.VLLM_EXL3_PLE_TIMING:
                    self._timing = Exl3PLETiming(
                        f"owner=[{reader.owner_lo},{reader.owner_hi}) "
                        f"workers={self._workers}"
                    )
            else:
                self._local_ids = torch.empty_like(self._ids)
                count = reader.owner_hi - reader.owner_lo
                if count <= 0:
                    raise ValueError("PLE comparison shard must contain rows")
                if self.placement == "ram":
                    if not is_uva_available():
                        raise RuntimeError("RAM PLE comparison requires CUDA UVA")
                    self._table = torch.empty(
                        count,
                        layout.words,
                        dtype=torch.int16,
                        device="cpu",
                        pin_memory=True,
                    )
                    view = memoryview(self._table.numpy()).cast("B")
                    reader.read_exact(
                        view,
                        reader.base + reader.owner_lo * reader.row_bytes,
                    )
                    self._table_view = get_accelerator_view_from_cpu_tensor(self._table)
                else:
                    self._table = torch.empty(
                        count, layout.words, dtype=torch.int16, device=device
                    )
                    block = min(count, self._capacity)
                    staging = torch.empty(
                        block,
                        layout.words,
                        dtype=torch.int16,
                        device="cpu",
                        pin_memory=True,
                    )
                    view = memoryview(staging.numpy()).cast("B")
                    for begin in range(0, count, block):
                        length = min(block, count - begin)
                        reader.read_exact(
                            view[: length * reader.row_bytes],
                            reader.base + (reader.owner_lo + begin) * reader.row_bytes,
                        )
                        self._table[begin : begin + length].copy_(staging[:length])
                    self._table_view = self._table
            self._slot = Exl3PLESlot(self._capacity, self._released)
            # Engine shutdown does not close modules; log residency at exit.
            atexit.register(self.close)
            logger.info(
                "EXL3 PLE: placement=%s storage=int16 output=bfloat16 "
                "owner=[%d,%d) row_bytes=%d capacity=%d",
                self.placement,
                reader.owner_lo,
                reader.owner_hi,
                reader.row_bytes,
                self._capacity,
            )
            hostfn = self.hostfn_reads and self.placement == "ssd"
            for name, on, banner in (
                (
                    "VLLM_EXL3_PLE_OVERLAP",
                    self.overlap_reads,
                    "EXL3 PLE: SSD row reads overlap the preceding decoder "
                    "layer ON (VLLM_EXL3_PLE_OVERLAP=1; same rows and kernels)",
                ),
                (
                    "VLLM_EXL3_PLE_NATIVE_READ",
                    self.native_reads,
                    "EXL3 PLE: native batched SSD row reader ON "
                    "(VLLM_EXL3_PLE_NATIVE_READ=1; same rows and bytes)",
                ),
            ):
                if not on or self.placement != "ssd":
                    continue
                if hostfn:
                    logger.info(
                        "EXL3 PLE: %s=1 is ignored for SSD rows while "
                        "VLLM_EXL3_PLE_HOSTFN_READ=1",
                        name,
                    )
                else:
                    logger.info(banner)
            if hostfn:
                self._open_host_reader(at_boot=True)
                logger.info(
                    "EXL3 PLE: SSD rows read by a host function on the PLE "
                    "stream ON (VLLM_EXL3_PLE_HOSTFN_READ=1; same rows and "
                    "bytes, no host wait; supersedes VLLM_EXL3_PLE_OVERLAP "
                    "and VLLM_EXL3_PLE_NATIVE_READ for SSD rows; read "
                    "failures are logged one step later)"
                )
            return consumed
        except BaseException:
            reader.close()
            raise

    def page_cache_residency(self) -> tuple[int, int]:
        """Return (resident, total) page-cache pages of the PLE table.

        Sampling does not change residency; lab scripts call this between
        runs to tell page-cache hits from SSD reads.
        """
        if self._reader is None or self._reader.fd < 0:
            raise RuntimeError("EXL3 PLE file is not open")
        return self._reader.page_cache_residency()

    def _log_residency(self, when: str) -> None:
        resident, total = self.page_cache_residency()
        logger.info(
            "EXL3 PLE page cache %s: %d/%d table pages resident (%.1f%%)",
            when,
            resident,
            total,
            100.0 * resident / max(total, 1),
        )

    def close(self):
        if self._slot is not None:
            self._stream.synchronize()
            self._released.synchronize()
        if self._host_reader is not None:
            try:
                self._take_host_reads()
            except RuntimeError as error:
                logger.error("EXL3 PLE: %s", error)
            # Every queued read has run: the reader closes its descriptor.
            self._host_reader = None
            self._host_reads_in_flight = 0
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._timing is not None:
            self._timing.flush()
        if self._reader is not None and self._reader.fd >= 0:
            self._log_residency("at close")
            self._reader.close()
        self._slot = None

    @eager_break_during_capture
    def start_prefetch(self, ngram_ids):
        self.check_ready()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 PLE requires eager or split CUDA graphs")
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered = self._gather_dp_ids(ngram_ids, slot_size)
        count = gathered.numel()
        hostfn = self.hostfn_reads and self.placement == "ssd"
        stream = self._stream
        # With host reads the slot's wait for the previous output copy is
        # queued on the PLE stream instead of made on the host.
        self._slot.acquire(count, stream if hostfn else None)
        try:
            if self._host_reader is not None:
                # Also after a switch to another reader: its last reads
                # finished before this step's ids wait.
                self._take_host_reads(room=hostfn)
            self._local_tokens = ngram_ids.shape[0]
            self._slot_offset = slot_offset
            self._gathered_tokens = gathered.shape[0]
            stream.wait_stream(torch.cuda.current_stream())
            gathered.record_stream(stream)
            with torch.cuda.stream(stream):
                self._ids[:count].copy_(gathered.reshape(-1))
                if self.placement == "ssd":
                    self._host_ids[:count].copy_(self._ids[:count], non_blocking=True)
                    if not hostfn:
                        self._ids_ready.record(stream)
        except BaseException:
            self._slot.fail()
            raise
        if hostfn:
            # The read is queued behind the ids copy; nothing waits on the host.
            self._queue_read_and_decode(count)
            return
        if self.overlap_reads and self.placement == "ssd":
            # The read waits for the ids on the host; at finalize that wait
            # no longer holds back the layers queued in between.
            self._deferred = count
            return
        self._read_and_decode(count)

    def _read_and_decode(self, count):
        """Read the staged rows (SSD), then queue their copy and decode.

        The slot is reading on entry and ready on return, or failed.
        """
        try:
            if self.placement == "ssd":
                timing = self._timing
                if timing is not None:
                    t0 = time.perf_counter_ns()
                self._ids_ready.synchronize()
                if timing is not None:
                    t1 = time.perf_counter_ns()
                ids = self._host_ids_np[:count]
                if self.native_reads:
                    failed = self._reader.read_rows_native(
                        self._host_ids, self._host_rows, count, self._workers
                    )
                else:
                    step = (count + self._workers - 1) // self._workers
                    futures = []
                    if step:
                        for begin in range(0, count, step):
                            futures.append(
                                self._pool.submit(
                                    self._reader.read_rows,
                                    ids,
                                    self._host_bytes,
                                    begin,
                                    min(begin + step, count),
                                )
                            )
                    # Join every read even if one fails, before the slot can close.
                    errors = []
                    failed = []
                    for future in futures:
                        try:
                            failed.extend(future.result())
                        except BaseException as error:
                            errors.append(error)
                    if errors:
                        raise errors[0]
                if failed:
                    poison_rows(self._host_rows_np, (index for index, _ in failed))
                    index, error = failed[0]
                    logger.error(
                        "EXL3 PLE: %d row read(s) failed after %d retries "
                        "(first: row %d: %s); the requests using them will be "
                        "finished with an error",
                        len(failed),
                        self._reader.retries,
                        int(ids[index]),
                        error,
                    )
                if timing is not None:
                    timing.add(count, t1 - t0, time.perf_counter_ns() - t1)
            self._queue_decode(count)
            self._slot.ready()
        except BaseException:
            self._slot.fail()
            raise

    def _open_host_reader(self, at_boot=False):
        """The host-function reader of the staging buffers, made on first
        use so the reader can be switched on after boot."""
        if self._host_reader is None:
            self._host_reader = self._reader.host_reader(
                self._host_ids, self._host_rows, self._workers, _HOST_READ_SLOTS
            )
            if not at_boot:
                logger.info(
                    "EXL3 PLE: SSD rows read by a host function on the PLE "
                    "stream switched ON after boot (same rows and bytes, no "
                    "host wait; read failures are logged one step later)"
                )
        return self._host_reader

    def _queue_read_and_decode(self, count):
        """Queue the row read as a host function on the PLE stream, then the
        copy and decode of its rows behind it. Every reader and writer of the
        staging buffers runs on that stream in order. The slot is reading on
        entry and ready on return, or failed."""
        try:
            reader = self._open_host_reader()
            with torch.cuda.stream(self._stream):
                reader.enqueue(count)
            self._host_reads_in_flight += 1
            self._queue_decode(count)
            self._slot.ready()
        except BaseException:
            self._slot.fail()
            raise

    def _take_host_reads(self, room=False):
        """Log the finished host reads: failed rows (already poisoned, so
        their tokens failed in their own step) and timing. With room, first
        wait for the PLE stream if every read slot is still taken. Raises
        when a read stopped on an unexpected error."""
        reader = self._host_reader
        results = reader.take().tolist()
        if room and self._host_reads_in_flight - len(results) >= _HOST_READ_SLOTS:
            logger.warning_once(
                "EXL3 PLE: every host read slot is in flight; waiting for the "
                "PLE stream"
            )
            self._stream.synchronize()
            results += reader.take().tolist()
        self._host_reads_in_flight -= len(results)
        stopped = 0
        for _, count, status, failed, _, row, code, offset, read_ns in results:
            if self._timing is not None:
                self._timing.add(count, 0, read_ns)
            if status:
                stopped += 1
            elif failed:
                logger.error(
                    "EXL3 PLE: %d row read(s) failed after %d retries "
                    "(first: row %d: %s); the requests using them will be "
                    "finished with an error",
                    failed,
                    self._reader.retries,
                    row,
                    _read_error(code, offset),
                )
        if stopped:
            raise RuntimeError(
                f"{stopped} PLE host read(s) stopped on an unexpected error; "
                "their rows were poisoned"
            )

    def _queue_decode(self, count):
        """Queue the copy of the staged rows and their decode on the PLE
        stream, then record _ready."""
        stream = self._stream
        with torch.cuda.stream(stream):
            packed = self._packed[:count]
            if self.placement == "ssd":
                packed.copy_(self._host_rows[:count], non_blocking=True)
            else:
                local = self._local_ids[:count]
                torch.sub(self._ids[:count], self._reader.owner_lo, out=local)
                # Invalid indices use row zero; the decoder masks before bias.
                local.clamp_(0, self._reader.owner_hi - self._reader.owner_lo - 1)
                torch.index_select(self._table_view, 0, local, out=packed)
            from vllm.sm80.kernels.exl3.ops import (
                ple_decode_out,
            )

            ple_decode_out(
                packed,
                self._ids[:count],
                self.head_bias,
                self.head_offsets,
                self.head_vocab_sizes,
                self._reader.owner_lo,
                self._reader.owner_hi,
                self._reader.layout.bits,
                self._decoded[:count],
            )
            self._ready.record(stream)

    @eager_break_during_capture
    def _finalize_prefetch(self, output):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 PLE finalize requires an eager graph break")
        count, self._deferred = self._deferred, None
        if count is not None:
            self._read_and_decode(count)
        if self._slot.state != "ready" or output.shape[0] != self._local_tokens:
            raise RuntimeError("PLE finalize has no matching prefetched rows")
        current = torch.cuda.current_stream()
        current.wait_event(self._ready)
        rows = self._decoded[: self._gathered_tokens * self._heads].view(
            self._gathered_tokens, self._heads, self.embedding_dim
        )
        if self.tp_size > 1:
            rows = self.parallel_group.all_reduce(rows)
        rows = self._select_embeddings(rows, self._local_tokens, self._slot_offset)
        self._failed_tokens = copy_masking_failed(rows, output)
        self._released.record(current)
        self._slot.release()

    def take_failed_tokens(self) -> torch.Tensor | None:
        """Return the last forward's per-token read-failure mask, once."""
        failed, self._failed_tokens = self._failed_tokens, None
        return failed

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Finish the prefetched lookup into new output storage."""
        output = self._decoded.new_empty((total_tokens, self._output_dim))
        self._finalize_prefetch(output)
        return output

    @property
    def supports_external_staging(self) -> bool:
        # Cross-DP ETP needs metadata from the forward context, which is not
        # available yet when the runner prepares model inputs.
        return self.etp_data_parallel_size == 1

    def view(self, total_tokens: int) -> torch.Tensor:
        """Persistent rows consumed by full graph replays."""
        self.check_ready()
        if not self.supports_external_staging:
            raise RuntimeError(
                "External EXL3 PLE staging requires ETP within one DP rank"
            )
        capacity = self._capacity // self._heads
        if not 0 <= total_tokens <= capacity:
            raise ValueError(f"PLE rows capacity exceeded: {total_tokens}/{capacity}")
        if self._staged_rows is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "EXL3 PLE rows must be allocated before graph capture"
                )
            # Separate from the reader's decoded slot: that slot can be reused
            # once the copy completes, while the model still consumes these rows.
            self._staged_rows = self._decoded.new_empty((capacity, self._output_dim))
        return self._staged_rows[:total_tokens]

    def prepare(self, ngram_ids: torch.Tensor, num_tokens_after_padding: int) -> None:
        """Read/decode outside capture with the existing slot and failure rules."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 PLE rows must be prepared outside graph capture")
        if ngram_ids.ndim != 2 or ngram_ids.shape[1] != self._heads:
            raise ValueError("PLE staging requires one row index per head and token")
        tokens = ngram_ids.shape[0]
        if tokens > num_tokens_after_padding:
            raise ValueError(
                "PLE rows cannot be padded to fewer than the active tokens"
            )
        rows = self.view(num_tokens_after_padding)
        if tokens:
            self.start_prefetch(ngram_ids)
            self._finalize_prefetch(rows[:tokens])
        else:
            self._failed_tokens = None
        rows[tokens:].zero_()

    def zero_rows(self, total_tokens: int) -> None:
        """Prepare dummy inputs without retaining a previous request's failures."""
        self.view(total_tokens).zero_()
        self._failed_tokens = None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "EXL3 PLE embedding requires start_prefetch() and finalize_prefetch()."
        )
