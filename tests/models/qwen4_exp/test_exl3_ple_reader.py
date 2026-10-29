# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 PLE row reader: positional reads, retries, truncation, read
faults, failed-token masks, page-cache residency, slot reuse, the
prefetch timing summary, the SSD read overlapped with the layer ahead, the
native reader and the read run as a host function on the PLE stream."""

import errno
import json
import logging
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import accumulate

import numpy as np
import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import exl3_ple
from vllm.models.qwen4_exp.nvidia.exl3_ple import (
    Exl3PLEEmbedding,
    Exl3PLERowReader,
    Exl3PLESlot,
    Exl3PLETiming,
    copy_masking_failed,
    poison_rows,
    pread_exact_into,
)

_PLE_PREFIX = "model.layers.0.ple.ple_embedding.ngram_embedding"


def _packed_ple_file(
    tmp_path,
    offset=32,
    *,
    bits=6,
    dim=160,
    sizes=(5, 4),
    rows=10,
    offsets=None,
    metadata=None,
    written_rows=None,
):
    """Small real PLE file: two heads, a sparse gap before the trellis and one
    padded row. Row r holds the byte r + 1 repeated."""
    words = 1 + dim * bits // 16
    heads = len(sizes)
    offsets = list(accumulate((0, *sizes[:-1]))) if offsets is None else offsets
    rows_data = [bytes([row + 1]) * (2 * words) for row in range(rows)]
    parts = [
        ("head_bias", "F16", [heads, dim], bytes(range(256)) * (heads * dim)),
        ("head_offsets", "I64", [heads], struct.pack(f"<{heads}q", *offsets)),
        ("head_vocab_sizes", "I64", [heads], struct.pack(f"<{heads}q", *sizes)),
        ("layer_multipliers", "I64", [3], struct.pack("<3q", 3, 5, 7)),
        ("gap", "I16", [offset // 2], None),
        ("trellis", "I16", [rows, words], b"".join(rows_data)),
    ]
    header = {
        "__metadata__": {
            "format": "exl3_ngram_trellis",
            "version": "1",
            "K": str(bits),
            "codebook": "mul1",
            "row_dim": str(dim),
            "rows": str(rows),
            **(metadata or {}),
        }
    }
    end = 0
    for name, dtype, shape, _ in parts:
        size = (
            (2 if dtype in ("F16", "I16") else 8)
            * shape[0]
            * (shape[1] if len(shape) > 1 else 1)
        )
        header[f"{_PLE_PREFIX}.{name}"] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [end, end + size],
        }
        end += size
    raw = json.dumps(header).encode()
    path = tmp_path / "ngram_embedding.safetensors"
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        for name, _, _, data in parts:
            info = header[f"{_PLE_PREFIX}.{name}"]["data_offsets"]
            size = info[1] - info[0]
            if name == "trellis" and written_rows is not None:
                data = data[: written_rows * 2 * words]
            if data is None:
                file.seek(size, os.SEEK_CUR)
            else:
                file.write(data[:size])
        file.truncate(8 + len(raw) + end)
    base = 8 + len(raw) + header[f"{_PLE_PREFIX}.trellis"]["data_offsets"][0]
    return path, f"{_PLE_PREFIX}.trellis", rows_data, base


_PLE_LAYOUTS = [
    pytest.param(6, 160, id="k6x160"),
    pytest.param(3, 32, id="k3x32"),
    pytest.param(5, 16, id="k5x16"),
]


@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
@pytest.mark.parametrize("offset", [32, (1 << 32) + 16])
def test_exl3_ple_positional_rows_preserve_order_and_masks(tmp_path, offset, bits, dim):
    path, key, rows, base = _packed_ple_file(tmp_path, offset, bits=bits, dim=dim)
    reader = Exl3PLERowReader(str(path), key, 4, 10)
    try:
        ids = [8, 4, 5, 4, 9, -1, 3, 10]
        row_bytes = reader.row_bytes
        target = bytearray(len(ids) * row_bytes)
        reader.read_rows(ids, memoryview(target))
        assert reader.base == base
        assert row_bytes == 2 + dim * bits // 8 == len(rows[0])
        assert reader.layout.bits == bits
        assert target == b"".join(
            rows[i] if i in (4, 5, 8) else bytes(row_bytes) for i in ids
        )
        assert os.lseek(reader.fd, 0, os.SEEK_CUR) == 0
    finally:
        reader.close()


@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
def test_exl3_ple_read_retries_interruptions_and_partial_reads(
    tmp_path, monkeypatch, bits, dim
):
    path, key, rows, _ = _packed_ple_file(tmp_path, bits=bits, dim=dim)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    real_preadv = os.preadv
    attempts = 0

    def interrupted_read(fd, buffers, offset):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InterruptedError()
        return real_preadv(fd, [buffers[0][:7]], offset)

    monkeypatch.setattr(os, "preadv", interrupted_read)
    try:
        target = bytearray(2 * reader.row_bytes)
        reader.read_rows([8, 1], memoryview(target))
        assert target == rows[8] + rows[1]
    finally:
        reader.close()


def test_exl3_ple_truncation_after_open_fails_only_cut_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(exl3_ple.time, "sleep", lambda _: None)
    path, key, rows, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=2)
    try:
        os.truncate(path, base + 8 * reader.row_bytes + 10)
        target = bytearray(2 * reader.row_bytes)
        failed = reader.read_rows([8, 1], memoryview(target))
        assert [(index, type(error)) for index, error in failed] == [(0, EOFError)]
        assert "Truncated PLE" in str(failed[0][1])
        assert target[reader.row_bytes :] == rows[1]
    finally:
        reader.close()
    with pytest.raises(ValueError, match="extent"):
        Exl3PLERowReader(str(path), key, 0, 10)


def test_exl3_ple_rejects_invalid_ranges_and_destinations(tmp_path):
    path, key, _, _ = _packed_ple_file(tmp_path)
    with pytest.raises(ValueError, match="owner range"):
        Exl3PLERowReader(str(path), key, 0, 11)
    with pytest.raises(ValueError, match="trellis"):
        Exl3PLERowReader(str(path), key.removesuffix(".trellis"), 0, 10)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    try:
        with pytest.raises(ValueError, match="destination"):
            reader.read_rows([1, 2], memoryview(bytearray(reader.row_bytes)))
        with pytest.raises(ValueError, match="64-bit"):
            pread_exact_into(reader.fd, memoryview(bytearray(1)), (1 << 63) - 1)
    finally:
        reader.close()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"offsets": [0, 6]}, "head range"),
        ({"sizes": (5, 0)}, "head range"),
        ({"rows": 140}, "padding"),
        ({"metadata": {"codebook": "mcg"}}, "format"),
        ({"metadata": {"rows": "11"}}, "metadata"),
        ({"metadata": {"K": "5"}}, "do not hold"),
        ({"bits": 3, "dim": 24}, "do not hold"),
    ],
)
def test_exl3_ple_rejects_invalid_layouts(tmp_path, kwargs, match):
    path, key, _, _ = _packed_ple_file(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        Exl3PLERowReader(str(path), key, 0, 10)


def _faulty_preadv(monkeypatch, fault, rows_to_fail, base, row_bytes, times=None):
    """Make reads of the given rows fail `times` times (always if None)."""
    real_preadv = os.preadv
    attempts = {}

    def preadv(fd, buffers, offset):
        row, within = divmod(offset - base, row_bytes)
        if row in rows_to_fail:
            if within == 0:
                attempts[row] = attempts.get(row, 0) + 1
            if times is None or attempts[row] <= times:
                if fault != "short":
                    raise OSError(fault, os.strerror(fault))
                # Part of the row, then end of file.
                if within == 0:
                    return real_preadv(fd, [buffers[0][:5]], offset)
                return 0
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", preadv)
    return attempts


_READ_FAULTS = [
    pytest.param(errno.EINTR, id="eintr"),
    pytest.param(errno.EIO, id="eio"),
    pytest.param("short", id="short"),
]


@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
@pytest.mark.parametrize("fault", _READ_FAULTS)
def test_exl3_ple_transient_read_error_is_retried(
    tmp_path, monkeypatch, fault, bits, dim
):
    sleeps = []
    monkeypatch.setattr(exl3_ple.time, "sleep", sleeps.append)
    path, key, rows, base = _packed_ple_file(tmp_path, bits=bits, dim=dim)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=3)
    try:
        attempts = _faulty_preadv(
            monkeypatch, fault, {4}, base, reader.row_bytes, times=1
        )
        ids = [4, 2, 4, 8]
        target = bytearray(len(ids) * reader.row_bytes)
        assert reader.read_rows(ids, memoryview(target)) == []
        assert target == b"".join(rows[i] for i in ids)
        assert sleeps == [0.001]
        assert attempts[4] >= 2
    finally:
        reader.close()


@pytest.mark.parametrize("fault", _READ_FAULTS)
def test_exl3_ple_persistent_read_error_fails_only_its_rows(
    tmp_path, monkeypatch, fault
):
    sleeps = []
    monkeypatch.setattr(exl3_ple.time, "sleep", sleeps.append)
    path, key, rows, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=3)
    try:
        _faulty_preadv(monkeypatch, fault, {5}, base, reader.row_bytes)
        ids = [5, 2, 9, 5, 7]
        target = bytearray(len(ids) * reader.row_bytes)
        failed = reader.read_rows(ids, memoryview(target))
        assert [index for index, _ in failed] == [0, 3]
        error = EOFError if fault == "short" else OSError
        assert all(isinstance(e, error) for _, e in failed)
        # Bounded: retries + 1 attempts per failed row, doubling backoff.
        assert sleeps == [0.001, 0.002, 0.004] * 2
        size = reader.row_bytes
        for index, row in ((1, rows[2]), (2, bytes(size)), (4, rows[7])):
            assert target[index * size : (index + 1) * size] == row
        # The reader is not stuck: later reads of other rows succeed...
        target = bytearray(2 * reader.row_bytes)
        assert reader.read_rows([1, 8], memoryview(target)) == []
        assert target == rows[1] + rows[8]
        # ...and the failed row reads again once the fault clears.
        monkeypatch.undo()
        target = bytearray(reader.row_bytes)
        assert reader.read_rows([5], memoryview(target)) == []
        assert target == rows[5]
    finally:
        reader.close()


def test_exl3_ple_poisoned_rows_become_zero_failed_tokens():
    """A poisoned row decodes to NaN (NaN scale times any codebook value);
    the NaN survives the rank sum and only its token is zeroed and flagged."""
    packed = np.zeros((6, 61), dtype=np.int16)
    packed[:, 0] = np.float16(0.5).view(np.int16)
    poison_rows(packed, [3])
    scales = packed[:, 0].view(np.float16).astype(np.float32)
    assert np.isnan(scales[3]) and not np.isnan(np.delete(scales, 3)).any()
    # Rows are [tokens=3, heads=2, dim]; row 3 is token 1, head 1.
    owner = torch.from_numpy(scales).repeat_interleave(4).view(3, 2, 4)
    other = torch.ones(3, 2, 4)
    summed = (owner * 2 + other).to(torch.bfloat16)
    output = torch.full((3, 8), 7.0, dtype=torch.bfloat16)
    failed = copy_masking_failed(summed, output)
    assert failed.tolist() == [False, True, False]
    assert torch.equal(output[1], torch.zeros(8, dtype=torch.bfloat16))
    assert torch.equal(output[[0, 2]], summed[[0, 2]].flatten(-2))


def test_exl3_ple_advises_random_reads_and_cold_drop(tmp_path, monkeypatch):
    path, key, _, base = _packed_ple_file(tmp_path)
    calls = []
    real_fadvise = os.posix_fadvise

    def record(fd, offset, length, advice):
        calls.append((offset, length, advice))
        real_fadvise(fd, offset, length, advice)

    monkeypatch.setattr(os, "posix_fadvise", record)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    try:
        assert calls == [(0, 0, os.POSIX_FADV_RANDOM)]
        reader.drop_page_cache()
        assert calls[1] == (base, 10 * reader.row_bytes, os.POSIX_FADV_DONTNEED)
    finally:
        reader.close()


def test_exl3_ple_page_cache_residency_counts_table_pages(tmp_path):
    """Only written table pages are resident; holes and other tensors are not
    counted, and sampling itself faults nothing in."""
    path, key, _, base = _packed_ple_file(
        tmp_path, offset=(1 << 20), sizes=(150, 40), rows=200, written_rows=10
    )
    page = mmap.PAGESIZE
    reader = Exl3PLERowReader(str(path), key, 0, 200)
    try:
        end = base + 200 * reader.row_bytes
        total = (end - 1) // page - base // page + 1
        written = (base + 10 * reader.row_bytes - 1) // page - base // page + 1
        assert written < total
        assert reader.page_cache_residency() == (written, total)
        assert reader.page_cache_residency() == (written, total)
        os.fsync(reader.fd)
        reader.drop_page_cache()
        resident, _ = reader.page_cache_residency()
        assert resident in (0, written)  # tmpfs keeps pages; disks drop them
    finally:
        reader.close()


def test_exl3_ple_slot_waits_for_copy_before_reuse():
    """A released lease still waits for the asynchronous output copy."""

    class Completion:
        def __init__(self):
            self.entered = threading.Event()
            self.done = threading.Event()

        def synchronize(self):
            self.entered.set()
            if not self.done.wait(5):
                raise TimeoutError("test copy did not complete")

    completion = Completion()
    completion.done.set()
    slot = Exl3PLESlot(8, completion)
    with pytest.raises(ValueError, match="capacity"):
        slot.acquire(9)
    slot.acquire(8)
    with pytest.raises(RuntimeError, match="reading"):
        slot.acquire(1)
    with pytest.raises(RuntimeError, match="no matching"):
        slot.release()
    slot.ready()
    with pytest.raises(RuntimeError, match="ready"):
        slot.acquire(1)
    completion.done.clear()
    completion.entered.clear()
    slot.release()
    acquired = threading.Event()

    def reuse():
        slot.acquire(1)
        acquired.set()

    worker = threading.Thread(target=reuse)
    worker.start()
    try:
        assert completion.entered.wait(2)
        assert not acquired.is_set()
        with pytest.raises(RuntimeError, match="waiting"):
            slot.acquire(1)
    finally:
        completion.done.set()
        worker.join(5)
    assert acquired.is_set()
    slot.ready()
    slot.release()


def test_exl3_ple_timing_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_EXL3_PLE_TIMING", raising=False)
    assert envs.VLLM_EXL3_PLE_TIMING is False
    monkeypatch.setenv("VLLM_EXL3_PLE_TIMING", "1")
    assert envs.VLLM_EXL3_PLE_TIMING is True


def test_exl3_ple_timing_logs_one_line_per_window():
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    log = logging.getLogger(exl3_ple.__name__)
    handler, level = Capture(logging.INFO), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        timing = Exl3PLETiming("owner=[0,10) workers=8")
        timing.WINDOW = 4
        timing.flush()  # nothing yet: no line
        for i in range(3):
            timing.add(64, 10_000, 800_000 + 100_000 * i)
        assert messages == []
        timing.add(64, 50_000, 1_200_000)  # fourth: the window is full
        assert len(messages) == 1 and timing.samples == []
        line = messages[0]
        # reads 800/900/1000/1200 us over 4 x 64 rows; waits 10/10/10/50 us
        for part in (
            "EXL3 PLE timing owner=[0,10) workers=8",
            "4 prefetches",
            "rows mean 64.0 max 64",
            "ids wait us p50 10 p90 38 max 50;",
            "row reads us p50 950 ",
            "max 1200 mean 975",
            "us per row 15.23",
        ):
            assert part in line, (part, line)
        timing.add(512, 1_000, 2_400_000)
        timing.flush()  # a partial window at close
        assert len(messages) == 2 and "1 prefetches" in messages[1]
    finally:
        log.removeHandler(handler)
        log.setLevel(level)


def test_exl3_ple_overlap_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_EXL3_PLE_OVERLAP", raising=False)
    assert envs.VLLM_EXL3_PLE_OVERLAP is False
    monkeypatch.setenv("VLLM_EXL3_PLE_OVERLAP", "1")
    assert envs.VLLM_EXL3_PLE_OVERLAP is True


class _Event:
    """A CUDA event stand-in that writes what the host asks of it to a trace;
    CPU tensors finish their work at once."""

    def __init__(self, name, trace):
        self.name, self.trace = name, trace

    def record(self, stream=None):
        self.trace.append(f"record {self.name}")

    def synchronize(self):
        self.trace.append(f"sync {self.name}")


class _Stream(_Event):
    def wait_stream(self, other):
        self.trace.append(f"{self.name} waits {other.name}")

    def wait_event(self, event):
        self.trace.append(f"{self.name} waits {event.name}")


_HEADS, _TOKENS = 2, 8
_REAL_PREADV = os.preadv


class _HostReaderOnStream:
    """A PleHostReader on a host without CUDA: enqueue traces the queued
    host function and runs the real one at once, as the stream stand-ins
    do with queued work; with hold set it keeps them until run_held."""

    def __init__(self, reader, trace):
        self.reader, self.trace = reader, trace
        self.hold, self.held = False, []

    def enqueue(self, count):
        self.trace.append(f"host read on {torch.cuda.current_stream().name}")
        if self.hold:
            self.held.append(count)
        else:
            self.reader.run(count)

    def run_held(self):
        for count in self.held:
            self.reader.run(count)
        self.held.clear()

    def take(self):
        return self.reader.take()


def _embedding_module(reader, device, *, overlap, native, hostfn):
    """An SSD PLE embedding over reader's table without its streams, events
    and slot; staging is zeroed, so a failed row's unread bytes match between
    modules."""
    layout = reader.layout
    capacity = _TOKENS * _HEADS
    module = object.__new__(Exl3PLEEmbedding)
    nn.Module.__init__(module)
    module.placement = "ssd"
    module.overlap_reads = overlap
    module.native_reads = native
    module.hostfn_reads = hostfn
    module._host_reader = None
    module._host_reads_in_flight = 0
    module._deferred = None
    module.tp_size = module.etp_data_parallel_size = 1
    module.embedding_dim = layout.dim
    module._heads = _HEADS
    module._output_dim = _HEADS * layout.dim
    module._capacity = capacity
    module._reader = reader
    module._workers = 2
    module._pool = ThreadPoolExecutor(max_workers=2)
    module._timing = None
    module._failed_tokens = None
    module._staged_rows = None
    module.head_bias = torch.zeros(
        _HEADS, layout.dim, dtype=torch.float16, device=device
    )
    module.head_offsets = torch.tensor(layout.head_offsets, device=device)
    module.head_vocab_sizes = torch.tensor(layout.head_vocab_sizes, device=device)
    module._ids = torch.zeros(capacity, dtype=torch.int64, device=device)
    module._packed = torch.zeros(
        capacity, layout.words, dtype=torch.int16, device=device
    )
    module._decoded = torch.zeros(
        capacity, layout.dim, dtype=torch.bfloat16, device=device
    )
    pin = device != "cpu"
    module._host_ids = torch.zeros(capacity, dtype=torch.int64, pin_memory=pin)
    module._host_rows = torch.zeros(
        capacity, layout.words, dtype=torch.int16, pin_memory=pin
    )
    module._host_ids_np = module._host_ids.numpy()
    module._host_rows_np = module._host_rows.numpy()
    module._host_bytes = memoryview(module._host_rows_np).cast("B")
    return module


def _ssd_embedding(
    tmp_path,
    monkeypatch,
    trace,
    *,
    overlap,
    owner_lo=3,
    native=False,
    hostfn=False,
    copies=False,
):
    """An SSD PLE embedding over a small real table, on the host: streams and
    events trace the host's calls, the decoder records the rows it gets. A
    stream context makes its stream the current one; with copies, the copy
    of the ids to the host and of the rows to the device are traced with
    the stream they are queued on."""
    path, key, _, _ = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, owner_lo, 10, retries=0)
    read_rows = reader.read_rows

    def traced_read_rows(*args):
        trace.append("read")
        return read_rows(*args)

    reader.read_rows = traced_read_rows
    read_rows_native = reader.read_rows_native

    def traced_read_rows_native(*args):
        trace.append("read")
        return read_rows_native(*args)

    reader.read_rows_native = traced_read_rows_native
    host_reader = reader.host_reader
    reader.host_reader = lambda *args: _HostReaderOnStream(host_reader(*args), trace)
    streams = [_Stream("current", trace)]

    @contextmanager
    def on_stream(stream):
        streams.append(stream)
        try:
            yield
        finally:
            streams.pop()

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: streams[-1])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    decoded = []

    def decode(packed, ids, bias, offsets, sizes, lo, hi, bits, out):
        trace.append("decode")
        decoded.append((packed.clone(), ids.clone()))
        # Any function of the packed bytes; a poisoned scale gives NaN.
        scale = packed[:, :1].view(torch.float16).float()
        out.copy_((scale * packed[:, 1:].float().sum(-1, keepdim=True)).expand_as(out))

    from vllm.sm80.kernels.exl3 import ops

    monkeypatch.setattr(ops, "ple_decode_out", decode)
    module = _embedding_module(
        reader, "cpu", overlap=overlap, native=native, hostfn=hostfn
    )
    module._stream = _Stream("side", trace)
    module._ids_ready = _Event("ids", trace)
    module._ready = _Event("ready", trace)
    module._released = _Event("released", trace)
    module._slot = exl3_ple.Exl3PLESlot(module._capacity, module._released)
    if copies:
        real_copy = torch.Tensor.copy_
        host_ids = module._host_ids.untyped_storage().data_ptr()
        host_rows = module._host_rows.untyped_storage().data_ptr()

        def traced_copy(self, src, non_blocking=False):
            if non_blocking:
                on = torch.cuda.current_stream().name
                if self.untyped_storage().data_ptr() == host_ids:
                    trace.append(f"ids to host on {on}")
                elif src.untyped_storage().data_ptr() == host_rows:
                    trace.append(f"rows to device on {on}")
            return real_copy(self, src, non_blocking=non_blocking)

        monkeypatch.setattr(torch.Tensor, "copy_", traced_copy)
    return module, decoded


def _step(module, ids, trace):
    """One forward's PLE part: prefetch, the layer ahead, finalize."""
    module.start_prefetch(torch.tensor(ids, dtype=torch.int64))
    trace.append("layer ahead")
    output = module.finalize_prefetch(len(ids))
    return output, module.take_failed_tokens()


def _reads_once(trace):
    """Pool workers each trace their chunk; keep one entry per read."""
    return [e for i, e in enumerate(trace) if e != "read" or trace[i - 1] != "read"]


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_overlap_keeps_order_and_moves_only_the_layer(
    tmp_path, monkeypatch, overlap
):
    """The deferred step makes the same host calls in the same order; only
    the layer ahead is queued before the ids wait instead of after the decode.
    """
    trace = []
    module, _ = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=overlap)
    try:
        for _ in range(2):  # the second step reuses the slot
            trace.clear()
            module.start_prefetch(torch.tensor([[3, 6], [4, 9]]))
            assert module._slot.state == ("reading" if overlap else "ready")
            trace.append("layer ahead")
            module.finalize_prefetch(2)
            assert module._slot.state == "idle"
            assert module.take_failed_tokens().tolist() == [False, False]
            calls = [
                "sync released",  # the slot waits for the previous copy
                "side waits current",
                "record ids",
                "sync ids",
                "read",
                "decode",
                "record ready",
                "current waits ready",
                "record released",
            ]
            calls.insert(3 if overlap else 7, "layer ahead")
            assert _reads_once(trace) == calls
    finally:
        module.close()


def test_exl3_ple_overlap_gives_the_consumer_the_same_bytes(tmp_path, monkeypatch):
    """Rows, ids, outputs and failure masks match bit for bit across token
    counts, unowned and padding rows, a failed row, and arms switched between
    steps as the in-boot A/B does."""
    monkeypatch.setattr(exl3_ple.time, "sleep", lambda seconds: None)
    steps = [
        [[3, 5]],
        [[4, 8], [0, 9], [3, 6]],  # row 0 is unowned, row 9 padding
        [[r % 5, 5 + r % 4] for r in range(_TOKENS)],  # rows 4 and 6 fail below
        [[2, 7]],
    ]
    results = {}
    for arms in ("off", "on", "alternating"):
        trace = []
        (tmp_path / arms).mkdir()
        module, decoded = _ssd_embedding(
            tmp_path / arms, monkeypatch, trace, overlap=arms == "on"
        )
        outputs = []
        try:
            for index, ids in enumerate(steps):
                if arms == "alternating":
                    module.overlap_reads = bool(index % 2)
                if index == 2:
                    _faulty_preadv(
                        monkeypatch,
                        errno.EIO,
                        {4, 6},
                        module._reader.base,
                        module._reader.row_bytes,
                    )
                output, failed = _step(module, ids, trace)
                if index == 2:
                    monkeypatch.setattr(os, "preadv", _REAL_PREADV)
                outputs.append((output.view(torch.int16).clone(), failed.clone()))
        finally:
            module.close()
        results[arms] = (decoded, outputs)
    reference_decoded, reference_outputs = results["off"]
    assert reference_outputs[2][1].tolist() == [
        r % 5 == 4 or 5 + r % 4 == 6 for r in range(_TOKENS)
    ]
    assert not any(failed.any() for _, failed in reference_outputs[:2])
    for arms in ("on", "alternating"):
        decoded, outputs = results[arms]
        assert len(decoded) == len(outputs) == len(steps)
        for (packed, ids), (ref_packed, ref_ids) in zip(decoded, reference_decoded):
            assert torch.equal(packed, ref_packed) and torch.equal(ids, ref_ids)
        for (output, failed), (ref_output, ref_failed) in zip(
            outputs, reference_outputs
        ):
            assert torch.equal(output, ref_output)
            assert torch.equal(failed, ref_failed)


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_overlap_read_error_fails_the_slot(tmp_path, monkeypatch, overlap):
    """A read that raises fails the slot after every chunk is joined; with the
    overlap it surfaces at finalize instead of at the prefetch."""
    trace = []
    module, decoded = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=overlap)
    joined = []

    def broken(ids, target, begin, end):
        joined.append(begin)
        if begin == 0:
            raise RuntimeError("storage gone")
        return []

    module._reader.read_rows = broken
    ids = torch.tensor([[3, 5], [4, 6], [3, 7], [4, 8]])
    try:
        if overlap:
            module.start_prefetch(ids)
            assert joined == [] and module._slot.state == "reading"
            with pytest.raises(RuntimeError, match="storage gone"):
                module.finalize_prefetch(4)
        else:
            with pytest.raises(RuntimeError, match="storage gone"):
                module.start_prefetch(ids)
        assert sorted(joined) == [0, 4]  # both chunks ran before the failure
        assert module._slot.state == "failed" and decoded == []
        assert module._deferred is None
        with pytest.raises(RuntimeError, match="slot is failed"):
            module.start_prefetch(ids)
        with pytest.raises(RuntimeError, match="no matching"):
            module.finalize_prefetch(4)
    finally:
        module.close()


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_overlap_cancelled_step_reads_nothing(tmp_path, monkeypatch, overlap):
    """A forward that ends between prefetch and finalize leaves the slot held:
    the next prefetch is refused as before. With the overlap no row was read
    and nothing is queued, so close has nothing to wait for."""
    trace = []
    module, decoded = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=overlap)
    try:
        with pytest.raises(RuntimeError, match="no matching"):
            module.finalize_prefetch(1)  # finalize before any prefetch
        assert module._slot.state == "idle"
        module.start_prefetch(torch.tensor([[3, 5]]))
        state = "reading" if overlap else "ready"
        with pytest.raises(RuntimeError, match=f"slot is {state}"):
            module.start_prefetch(torch.tensor([[4, 6]]))
        assert ("read" in trace) is not overlap
        assert len(decoded) == (0 if overlap else 1)
        assert module._slot.state == state
    finally:
        module.close()
    assert module._pool is None and module._reader.fd == -1


@pytest.fixture
def native_reader():
    from vllm.sm80.kernels.exl3 import ops

    try:
        ops.ensure_extension()
    except RuntimeError as exc:
        pytest.skip(f"vllm._exl3_C unavailable: {exc}")


def test_exl3_ple_native_read_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_EXL3_PLE_NATIVE_READ", raising=False)
    assert envs.VLLM_EXL3_PLE_NATIVE_READ is False
    monkeypatch.setenv("VLLM_EXL3_PLE_NATIVE_READ", "1")
    assert envs.VLLM_EXL3_PLE_NATIVE_READ is True


def _python_and_native(reader, ids, threads, count=None):
    """Destinations and failures of read_rows and read_rows_native over the
    same ids, both into one extra row of the same garbage."""
    count = len(ids) if count is None else count
    host_ids = torch.tensor(ids, dtype=torch.int64)
    garbage = torch.randint(
        -(1 << 15), 1 << 15, (len(ids) + 1, reader.row_bytes // 2), dtype=torch.int16
    )
    python, native = garbage.clone(), garbage.clone()
    python_failed = reader.read_rows(
        host_ids.numpy(), memoryview(python.numpy()).cast("B"), 0, count
    )
    native_failed = reader.read_rows_native(host_ids, native, count, threads)
    assert torch.equal(python[count:], garbage[count:])
    assert torch.equal(native[count:], garbage[count:])
    return python, native, python_failed, native_failed


def _errors(failed):
    return [(index, type(error), str(error)) for index, error in failed]


@pytest.mark.parametrize("threads", [1, 3, 8])
@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
@pytest.mark.parametrize("offset", [32, (1 << 32) + 16])
def test_exl3_ple_native_read_gives_the_same_bytes(
    tmp_path, native_reader, offset, bits, dim, threads
):
    """Order, repetitions, unowned, padding and out-of-range rows, a count
    short of the ids and owner ranges that cut a head."""
    path, key, rows, _ = _packed_ple_file(
        tmp_path, offset, bits=bits, dim=dim, sizes=(100, 120), rows=230
    )
    generator = np.random.default_rng(threads)
    for owner_lo, owner_hi in ((0, 230), (57, 181), (100, 100)):
        reader = Exl3PLERowReader(str(path), key, owner_lo, owner_hi)
        try:
            ids = [8, 4, 5, 4, 225, -1, 3, 230, 1 << 40, -(1 << 40), 99, 100]
            ids += generator.integers(-5, 240, 500).tolist()
            for count in (0, 37, len(ids)):
                python, native, python_failed, native_failed = _python_and_native(
                    reader, ids, threads, count
                )
                assert python_failed == native_failed == []
                assert torch.equal(python, native)
            owned = [owner_lo <= i < min(owner_hi, 220) for i in ids]
            expected = b"".join(
                rows[i] if own else bytes(reader.row_bytes)
                for i, own in zip(ids, owned)
            )
            assert native[: len(ids)].numpy().tobytes() == expected
            assert os.lseek(reader.fd, 0, os.SEEK_CUR) == 0
        finally:
            reader.close()


@pytest.mark.parametrize("threads", [1, 4])
def test_exl3_ple_native_read_from_storage_gives_the_same_bytes(
    tmp_path, native_reader, threads
):
    """Rows not in the page cache take the threaded read path."""
    path, key, rows, _ = _packed_ple_file(tmp_path, sizes=(100, 120), rows=230)
    reader = Exl3PLERowReader(str(path), key, 0, 230)
    try:
        os.fsync(reader.fd)
        os.posix_fadvise(reader.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if reader.page_cache_residency()[0]:
            pytest.skip("this file system keeps the table in the page cache")
        ids = np.random.default_rng(7).integers(0, 230, 300).tolist()
        target = torch.zeros(len(ids), reader.row_bytes // 2, dtype=torch.int16)
        failed = reader.read_rows_native(torch.tensor(ids), target, len(ids), threads)
        assert failed == []
        assert target.numpy().tobytes() == b"".join(
            rows[i] if i < 220 else bytes(reader.row_bytes) for i in ids
        )
    finally:
        reader.close()


@pytest.mark.parametrize("threads", [1, 3])
def test_exl3_ple_native_read_fails_the_same_rows_with_equal_errors(
    tmp_path, native_reader, threads
):
    """End of file inside and before a row, and a descriptor that cannot be
    read: the same rows fail after the retries, with equal errors, and the
    other rows are read."""
    path, key, rows, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=2)
    try:
        os.truncate(path, base + 8 * reader.row_bytes + 10)
        ids = [8, 1, 9, 0, 8, 7]
        python, native, python_failed, native_failed = _python_and_native(
            reader, ids, threads
        )
        assert _errors(native_failed) == _errors(python_failed)
        assert [index for index, _ in native_failed] == [0, 4]  # row 9 is padding
        assert all(isinstance(error, EOFError) for _, error in native_failed)
        good = [1, 2, 3, 5]
        assert torch.equal(native[good], python[good])
        file_fd = reader.fd
        reader.fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _, _, python_failed, native_failed = _python_and_native(
                reader, ids, threads
            )
        finally:
            os.close(reader.fd)
            reader.fd = file_fd
        assert _errors(native_failed) == _errors(python_failed)
        assert [index for index, _ in native_failed] == [0, 1, 3, 4, 5]
        assert all(isinstance(error, IsADirectoryError) for _, error in native_failed)
    finally:
        reader.close()


def test_exl3_ple_native_read_releases_the_gil(tmp_path, native_reader):
    """Python threads run while the native call waits (here on its retry
    backoff: 1 + 2 + 4 + 8 ms for each of four failing rows)."""
    path, key, _, _ = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=4)
    file_fd = reader.fd
    reader.fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    window = []

    def call():
        target = torch.empty(4, reader.row_bytes // 2, dtype=torch.int16)
        window.append(time.perf_counter())
        reader.read_rows_native(torch.tensor([1, 2, 3, 4]), target, 4, 1)
        window.append(time.perf_counter())

    try:
        worker = threading.Thread(target=call)
        ticks = []
        worker.start()
        while worker.is_alive():
            ticks.append(time.perf_counter())
        worker.join()
    finally:
        os.close(reader.fd)
        reader.fd = file_fd
        reader.close()
    start, end = window
    assert end - start >= 0.05
    inside = [start] + [t for t in ticks if start < t < end] + [end]
    assert max(b - a for a, b in zip(inside, inside[1:])) < (end - start) / 2


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_native_read_keeps_the_host_order(tmp_path, monkeypatch, overlap):
    """The native read takes the pooled read's place in the step; nothing
    else moves."""
    trace = []
    module, _ = _ssd_embedding(
        tmp_path, monkeypatch, trace, overlap=overlap, native=True
    )
    module._reader.read_rows = None  # the pool must not be used
    module._reader.read_rows_native = lambda *args: trace.append("read") or []
    try:
        for _ in range(2):
            trace.clear()
            _, failed = _step(module, [[3, 6], [4, 9]], trace)
            assert failed.tolist() == [False, False]
            calls = [
                "sync released",
                "side waits current",
                "record ids",
                "sync ids",
                "read",
                "decode",
                "record ready",
                "current waits ready",
                "record released",
            ]
            calls.insert(3 if overlap else 7, "layer ahead")
            assert trace == calls
    finally:
        module.close()


def test_exl3_ple_native_read_gives_the_consumer_the_same_bytes(
    tmp_path, monkeypatch, native_reader
):
    """Rows, ids, outputs and failure masks match bit for bit between the
    pooled and the native reader, with the overlap off and on, and with the
    reader switched between steps as the in-boot A/B does. In one step every
    owned row fails (a descriptor that cannot be read)."""
    monkeypatch.setattr(exl3_ple.time, "sleep", lambda seconds: None)
    steps = [
        [[3, 5]],
        [[4, 8], [0, 9], [3, 6]],  # row 0 is unowned, row 9 padding
        [[r % 5, 5 + r % 4] for r in range(_TOKENS)],
        [[2, 7]],
    ]
    results = {}
    for overlap in (False, True):
        for arms in ("pooled", "native", "alternating"):
            name = f"{arms}-{overlap}"
            (tmp_path / name).mkdir()
            module, decoded = _ssd_embedding(
                tmp_path / name,
                monkeypatch,
                [],
                overlap=overlap,
                native=arms == "native",
            )
            outputs = []
            try:
                for index, ids in enumerate(steps):
                    if arms == "alternating":
                        module.native_reads = bool(index % 2)
                    if index == 2:
                        file_fd = module._reader.fd
                        module._reader.fd = os.open(
                            tmp_path, os.O_RDONLY | os.O_DIRECTORY
                        )
                    try:
                        output, failed = _step(module, ids, [])
                    finally:
                        if index == 2:
                            os.close(module._reader.fd)
                            module._reader.fd = file_fd
                    outputs.append((output.view(torch.int16).clone(), failed.clone()))
            finally:
                module.close()
            results[name] = (decoded, outputs)
    reference_decoded, reference_outputs = results["pooled-False"]
    assert reference_outputs[2][1].tolist() == [True] * _TOKENS
    assert not any(failed.any() for _, failed in reference_outputs[:2])
    for name, (decoded, outputs) in results.items():
        assert len(decoded) == len(outputs) == len(steps)
        for (packed, ids), (ref_packed, ref_ids) in zip(decoded, reference_decoded):
            assert torch.equal(packed, ref_packed) and torch.equal(ids, ref_ids)
        for (output, failed), (ref_output, ref_failed) in zip(
            outputs, reference_outputs
        ):
            assert torch.equal(output, ref_output)
            assert torch.equal(failed, ref_failed)


def test_exl3_ple_hostfn_read_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_EXL3_PLE_HOSTFN_READ", raising=False)
    assert envs.VLLM_EXL3_PLE_HOSTFN_READ is False
    monkeypatch.setenv("VLLM_EXL3_PLE_HOSTFN_READ", "1")
    assert envs.VLLM_EXL3_PLE_HOSTFN_READ is True


_RESULT_COLUMNS = (
    "sequence",
    "count",
    "status",
    "failed",
    "index",
    "row",
    "errno",
    "offset",
    "read_ns",
)


def _results(host):
    """take() as one dict per finished read."""
    return [dict(zip(_RESULT_COLUMNS, row)) for row in host.take().tolist()]


def _python_and_host(reader, ids, threads, count=None):
    """Destinations of read_rows (failed rows poisoned) and of the host
    function over the same ids, both into one extra row of the same garbage;
    the failures of read_rows and the host read's result."""
    count = len(ids) if count is None else count
    host_ids = torch.tensor(ids, dtype=torch.int64)
    garbage = torch.randint(
        -(1 << 15), 1 << 15, (len(ids) + 1, reader.row_bytes // 2), dtype=torch.int16
    )
    python, host_rows = garbage.clone(), garbage.clone()
    python_failed = reader.read_rows(
        host_ids.numpy(), memoryview(python.numpy()).cast("B"), 0, count
    )
    if python_failed:
        poison_rows(python.numpy(), (index for index, _ in python_failed))
    host = reader.host_reader(host_ids, host_rows, threads, 2)
    host.run(count)
    (result,) = _results(host)
    assert torch.equal(python[count:], garbage[count:])
    assert torch.equal(host_rows[count:], garbage[count:])
    return python, host_rows, python_failed, result


@pytest.mark.parametrize("threads", [1, 3, 8])
@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
@pytest.mark.parametrize("offset", [32, (1 << 32) + 16])
def test_exl3_ple_hostfn_read_gives_the_same_bytes(
    tmp_path, native_reader, offset, bits, dim, threads
):
    """The host function's body: order, repetitions, unowned, padding and
    out-of-range rows, a count short of the ids and owner ranges that cut a
    head."""
    path, key, rows, _ = _packed_ple_file(
        tmp_path, offset, bits=bits, dim=dim, sizes=(100, 120), rows=230
    )
    generator = np.random.default_rng(threads)
    for owner_lo, owner_hi in ((0, 230), (57, 181), (100, 100)):
        reader = Exl3PLERowReader(str(path), key, owner_lo, owner_hi)
        try:
            ids = [8, 4, 5, 4, 225, -1, 3, 230, 1 << 40, -(1 << 40), 99, 100]
            ids += generator.integers(-5, 240, 500).tolist()
            for count in (0, 37, len(ids)):
                python, host, python_failed, result = _python_and_host(
                    reader, ids, threads, count
                )
                assert python_failed == []
                assert result["count"] == count
                assert result["status"] == result["failed"] == 0
                assert result["read_ns"] > 0
                assert torch.equal(python, host)
            owned = [owner_lo <= i < min(owner_hi, 220) for i in ids]
            expected = b"".join(
                rows[i] if own else bytes(reader.row_bytes)
                for i, own in zip(ids, owned)
            )
            assert host[: len(ids)].numpy().tobytes() == expected
            assert os.lseek(reader.fd, 0, os.SEEK_CUR) == 0
        finally:
            reader.close()


@pytest.mark.parametrize("threads", [1, 3])
def test_exl3_ple_hostfn_read_poisons_the_failed_rows(tmp_path, native_reader, threads):
    """End of file inside and before a row, and a descriptor that cannot be
    read: the rows read_rows fails are poisoned, the first failure is
    reported with an equal error and the other rows are read."""
    path, key, rows, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=2)
    try:
        os.truncate(path, base + 8 * reader.row_bytes + 10)
        ids = [8, 1, 9, 0, 8, 7]
        python, host, python_failed, result = _python_and_host(reader, ids, threads)
        assert [index for index, _ in python_failed] == [0, 4]  # row 9 is padding
        assert result["status"] == 0 and result["failed"] == 2
        assert (result["index"], result["row"]) == (0, 8)
        error = exl3_ple._read_error(result["errno"], result["offset"])
        assert _errors([(0, error)]) == _errors(python_failed[:1])
        good = [1, 2, 3, 5]
        assert torch.equal(host[good], python[good])
        poisoned = exl3_ple._POISONED_SCALE
        assert host[[0, 4], 0].tolist() == [poisoned] * 2
        assert python[[0, 4], 0].tolist() == [poisoned] * 2
        # The host reader duplicates the descriptor it is built with.
        file_fd = reader.fd
        reader.fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            python, host, python_failed, result = _python_and_host(reader, ids, threads)
        finally:
            os.close(reader.fd)
            reader.fd = file_fd
        assert [index for index, _ in python_failed] == [0, 1, 3, 4, 5]
        assert result["failed"] == 5 and result["errno"] == errno.EISDIR
        error = exl3_ple._read_error(result["errno"], result["offset"])
        assert _errors([(0, error)]) == _errors(python_failed[:1])
        assert host[[0, 1, 3, 4, 5], 0].tolist() == [poisoned] * 5
        assert torch.equal(host[2], python[2])
    finally:
        reader.close()


def test_exl3_ple_hostfn_read_from_storage_gives_the_same_bytes(
    tmp_path, native_reader
):
    """Rows not in the page cache take the threaded read path."""
    path, key, rows, _ = _packed_ple_file(tmp_path, sizes=(100, 120), rows=230)
    reader = Exl3PLERowReader(str(path), key, 0, 230)
    try:
        os.fsync(reader.fd)
        os.posix_fadvise(reader.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if reader.page_cache_residency()[0]:
            pytest.skip("this file system keeps the table in the page cache")
        ids = np.random.default_rng(7).integers(0, 230, 300).tolist()
        target = torch.zeros(len(ids), reader.row_bytes // 2, dtype=torch.int16)
        host = reader.host_reader(torch.tensor(ids), target, 4, 1)
        host.run(len(ids))
        assert [r["failed"] for r in _results(host)] == [0]
        assert target.numpy().tobytes() == b"".join(
            rows[i] if i < 220 else bytes(reader.row_bytes) for i in ids
        )
    finally:
        reader.close()


def test_exl3_ple_hostfn_read_ring_keeps_each_result(tmp_path, native_reader):
    """Results come back in queue order, each with its own failures although
    the next read rewrote the rows; a full ring and a read beyond the
    staging capacity are refused without taking a slot."""
    path, key, _, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=0)
    try:
        os.truncate(path, base + 8 * reader.row_bytes)
        ids = torch.tensor([8, 1, 8, 2])
        host_rows = torch.zeros(4, reader.row_bytes // 2, dtype=torch.int16)
        host = reader.host_reader(ids, host_rows, 1, 2)
        host.run(4)
        ids.copy_(torch.tensor([1, 2, 3, 4]))
        host.run(4)
        with pytest.raises(RuntimeError, match="in flight or not taken"):
            host.run(1)
        results = _results(host)
        assert [r["sequence"] for r in results] == [0, 1]
        assert [r["failed"] for r in results] == [2, 0]
        assert results[0]["row"] == 8
        assert _results(host) == []
        with pytest.raises(RuntimeError, match="staging capacity"):
            host.run(5)
        host.run(1)
        assert [r["sequence"] for r in _results(host)] == [2]
    finally:
        reader.close()


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a host without CUDA")
def test_exl3_ple_hostfn_enqueue_without_cuda_queues_nothing(tmp_path, native_reader):
    path, key, _, _ = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    try:
        host_rows = torch.zeros(2, reader.row_bytes // 2, dtype=torch.int16)
        host = reader.host_reader(torch.tensor([1, 2]), host_rows, 1, 1)
        with pytest.raises(RuntimeError):
            host.enqueue(2)
        assert _results(host) == []
        host.run(2)
        assert [r["sequence"] for r in _results(host)] == [0]
    finally:
        reader.close()


def test_exl3_ple_slot_acquire_on_a_stream_waits_there():
    trace = []
    slot = Exl3PLESlot(8, _Event("released", trace))
    slot.acquire(2, _Stream("side", trace))
    assert trace == ["side waits released"] and slot.state == "reading"
    slot.ready()
    slot.release()
    slot.acquire(2)
    assert trace[-1] == "sync released"


def _hostfn_calls():
    return [
        "side waits released",  # queued: the host does not wait
        "side waits current",
        "host read on side",
        "decode",
        "record ready",
        "layer ahead",
        "current waits ready",
        "record released",
    ]


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_hostfn_read_never_waits_on_the_host(
    tmp_path, monkeypatch, native_reader, overlap
):
    """Everything is queued at the prefetch, read included; the host makes
    no synchronize call, with the overlap off or on."""
    trace = []
    module, _ = _ssd_embedding(
        tmp_path, monkeypatch, trace, overlap=overlap, hostfn=True
    )
    module._reader.read_rows = None  # neither the pool
    module._reader.read_rows_native = None  # nor the native call
    try:
        for _ in range(2):
            trace.clear()
            _, failed = _step(module, [[3, 6], [4, 9]], trace)
            assert failed.tolist() == [False, False]
            assert trace == _hostfn_calls()
        assert module._host_reads_in_flight == 1
    finally:
        module.close()
    assert module._host_reader is None and module._host_reads_in_flight == 0


@pytest.mark.parametrize("hostfn", [False, True])
def test_exl3_ple_reads_and_copies_stay_on_the_side_stream(
    tmp_path, monkeypatch, native_reader, hostfn
):
    """The copy of the ids to the host, the read and the copy of the rows to
    the device are queued in that order on the PLE stream: a host read queued
    on another stream would race the two copies around it."""
    trace = []
    module, _ = _ssd_embedding(
        tmp_path, monkeypatch, trace, overlap=False, hostfn=hostfn, copies=True
    )
    if hostfn:
        calls = _hostfn_calls()
        calls.insert(2, "ids to host on side")
        calls.insert(4, "rows to device on side")
    else:
        calls = [
            "sync released",
            "side waits current",
            "ids to host on side",
            "record ids",
            "sync ids",
            "read",
            "rows to device on side",
            "decode",
            "record ready",
            "layer ahead",
            "current waits ready",
            "record released",
        ]
    try:
        for _ in range(2):
            trace.clear()
            _, failed = _step(module, [[3, 6], [4, 9]], trace)
            assert failed.tolist() == [False, False]
            assert _reads_once(trace) == calls
    finally:
        module.close()


def _truncated(path, size):
    """Truncate the table for one step, then write it back."""
    data = path.read_bytes()
    os.truncate(path, size)

    def restore():
        with path.open("r+b") as file:
            file.write(data)

    return restore


def test_exl3_ple_hostfn_read_gives_the_consumer_the_same_bytes(
    tmp_path, monkeypatch, native_reader
):
    """Rows, ids, outputs and failure masks match bit for bit between the
    pooled, native and host-function readers, the last with the overlap off
    and on, and with the reader switched between steps as the in-boot A/B
    does (both parities). In one step the table ends inside row 6."""
    monkeypatch.setattr(exl3_ple.time, "sleep", lambda seconds: None)
    steps = [
        [[3, 5]],
        [[4, 8], [0, 9], [3, 6]],  # row 0 is unowned, row 9 padding
        [[r % 5, 5 + r % 4] for r in range(_TOKENS)],  # rows 6, 7, 8 fail
        [[2, 7]],
        [[8, 3]],
    ]
    arms = {
        "pooled": lambda index: {},
        "native": lambda index: {"native_reads": True},
        "hostfn": lambda index: {"hostfn_reads": True},
        "hostfn-overlap": lambda index: {"hostfn_reads": True, "overlap_reads": True},
        "alternating-0": lambda index: {"hostfn_reads": index % 2 == 0},
        "alternating-1": lambda index: {"hostfn_reads": index % 2 == 1},
    }
    results = {}
    for name, switches in arms.items():
        (tmp_path / name).mkdir()
        module, decoded = _ssd_embedding(
            tmp_path / name, monkeypatch, [], overlap=False
        )
        path = tmp_path / name / "ngram_embedding.safetensors"
        outputs = []
        try:
            for index, ids in enumerate(steps):
                module.native_reads = module.hostfn_reads = False
                module.overlap_reads = False
                for attribute, value in switches(index).items():
                    setattr(module, attribute, value)
                restore = None
                if index == 2:
                    restore = _truncated(
                        path, module._reader.base + 6 * module._reader.row_bytes + 10
                    )
                try:
                    output, failed = _step(module, ids, [])
                finally:
                    if restore is not None:
                        restore()
                outputs.append((output.view(torch.int16).clone(), failed.clone()))
        finally:
            module.close()
        results[name] = (decoded, outputs)
    reference_decoded, reference_outputs = results["pooled"]
    assert reference_outputs[2][1].tolist() == [r % 4 != 0 for r in range(_TOKENS)]
    assert not any(failed.any() for _, failed in reference_outputs[:2])
    assert not any(failed.any() for _, failed in reference_outputs[3:])
    for name, (decoded, outputs) in results.items():
        assert len(decoded) == len(outputs) == len(steps), name
        for (packed, ids), (ref_packed, ref_ids) in zip(decoded, reference_decoded):
            assert torch.equal(packed, ref_packed) and torch.equal(ids, ref_ids), name
        for (output, failed), (ref_output, ref_failed) in zip(
            outputs, reference_outputs
        ):
            assert torch.equal(output, ref_output), name
            assert torch.equal(failed, ref_failed), name


@contextmanager
def _captured_log():
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append((record.levelno, record.getMessage()))

    log = logging.getLogger(exl3_ple.__name__)
    handler, level = Capture(logging.INFO), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        yield messages
    finally:
        log.removeHandler(handler)
        log.setLevel(level)


def test_exl3_ple_hostfn_read_failure_masks_its_step_and_logs_at_the_next(
    tmp_path, monkeypatch, native_reader
):
    trace = []
    module, _ = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=False, hostfn=True)
    path = tmp_path / "ngram_embedding.safetensors"
    try:
        with _captured_log() as messages:
            restore = _truncated(
                path, module._reader.base + 8 * module._reader.row_bytes + 10
            )
            try:
                _, failed = _step(module, [[3, 8], [4, 5]], trace)
            finally:
                restore()
            assert failed.tolist() == [True, False]
            ((level, message),) = messages  # made after boot: one line
            assert level == logging.INFO and "switched ON after boot" in message
            messages.clear()
            _, failed = _step(module, [[3, 8], [4, 5]], trace)
            assert failed.tolist() == [False, False]
            ((level, message),) = messages
            assert level == logging.ERROR
            assert "1 row read(s) failed after 0 retries" in message
            assert "(first: row 8: Truncated PLE data at byte" in message
    finally:
        module.close()


def test_exl3_ple_hostfn_read_close_logs_late_failures(
    tmp_path, monkeypatch, native_reader
):
    trace = []
    module, _ = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=False, hostfn=True)
    path = tmp_path / "ngram_embedding.safetensors"
    restore = _truncated(path, module._reader.base + 8 * module._reader.row_bytes)
    try:
        with _captured_log() as messages:
            _, failed = _step(module, [[8, 3]], trace)
            assert failed.tolist() == [True]
            module.close()
        assert any(
            level == logging.ERROR and "first: row 8:" in message
            for level, message in messages
        )
        assert module._host_reader is None and module._reader.fd == -1
    finally:
        restore()
        module.close()


def test_exl3_ple_hostfn_read_stopped_fails_the_slot(
    tmp_path, monkeypatch, native_reader
):
    """A read stopped by an unexpected error (every row poisoned) fails the
    slot at the next prefetch, after its line is logged."""
    trace = []
    module, _ = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=False, hostfn=True)
    host = module._open_host_reader()
    take = host.take
    try:
        _step(module, [[3, 6]], trace)
        stopped = take()
        assert stopped.shape == (1, len(_RESULT_COLUMNS))
        stopped[0, 2] = 1  # status: stopped
        stopped[0, 3] = 2  # every row of the read failed
        host.take = lambda: stopped
        with pytest.raises(RuntimeError, match="stopped on an unexpected error"):
            module.start_prefetch(torch.tensor([[4, 5]]))
        assert module._slot.state == "failed"
    finally:
        host.take = take
        module.close()


def test_exl3_ple_hostfn_read_waits_once_when_the_ring_is_full(
    tmp_path, monkeypatch, native_reader
):
    """Reads that have not run by the time every slot is taken make the
    prefetch wait for the PLE stream once; then the ring has room again."""
    trace = []
    module, _ = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=False, hostfn=True)
    try:
        host = module._open_host_reader()
        host.hold = True

        def synchronize():
            trace.append("sync side")
            host.run_held()

        module._stream.synchronize = synchronize
        for step in range(exl3_ple._HOST_READ_SLOTS + 2):
            trace.clear()
            _step(module, [[3, 6]], trace)
            calls = _hostfn_calls()
            if step == exl3_ple._HOST_READ_SLOTS:
                calls.insert(1, "sync side")  # after the slot's queued wait
            assert trace == calls
        assert module._host_reads_in_flight == 2
        host.hold = False
        synchronize()
    finally:
        module.close()
    assert module._host_reads_in_flight == 0


_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cuda_host_reader(tmp_path, count=64, slots=4):
    path, key, _, _ = _packed_ple_file(tmp_path, sizes=(100, 120), rows=230)
    reader = Exl3PLERowReader(str(path), key, 0, 230)
    ids = torch.tensor(
        np.random.default_rng(3).integers(0, 230, count), dtype=torch.int64
    ).pin_memory()
    rows = torch.zeros(count, reader.row_bytes // 2, dtype=torch.int16).pin_memory()
    expected = torch.zeros_like(rows)
    reader.read_rows(ids.numpy(), memoryview(expected.numpy()).cast("B"))
    return reader, ids, rows, expected, reader.host_reader(ids, rows, 4, slots)


@_needs_cuda
def test_exl3_ple_hostfn_enqueue_runs_in_stream_order(tmp_path, native_reader):
    reader, ids, rows, expected, host = _cuda_host_reader(tmp_path)
    try:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(500_000_000)  # holds the stream well past enqueue
            host.enqueue(len(ids))
            device = rows.to("cuda", non_blocking=True)
        assert _results(host) == []
        stream.synchronize()
        (result,) = _results(host)
        assert (result["count"], result["status"], result["failed"]) == (64, 0, 0)
        assert torch.equal(device.cpu(), expected)
    finally:
        reader.close()


@_needs_cuda
def test_exl3_ple_hostfn_reader_dropped_while_queued(tmp_path, native_reader):
    """A reader dropped with a read still queued is left allocated; the read
    runs when the stream gets to it."""
    import gc

    reader, ids, rows, expected, host = _cuda_host_reader(tmp_path)
    try:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(500_000_000)
            host.enqueue(len(ids))
        del host
        gc.collect()
        stream.synchronize()
        assert torch.equal(rows, expected)
    finally:
        reader.close()


@_needs_cuda
def test_exl3_ple_hostfn_enqueue_refuses_capture(tmp_path, native_reader):
    reader, ids, rows, _, host = _cuda_host_reader(tmp_path)
    try:
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(RuntimeError, match="cannot be captured"),
            torch.cuda.graph(graph),
        ):
            host.enqueue(len(ids))
        assert _results(host) == []
        host.run(len(ids))
        assert [r["sequence"] for r in _results(host)] == [0]
    finally:
        reader.close()


def _cuda_ssd_embedding(directory, *, native, hostfn):
    """An SSD PLE embedding over a small real table on the GPU: real streams,
    events, pinned staging and ple_decode_out."""
    path, key, _, _ = _packed_ple_file(directory)
    reader = Exl3PLERowReader(str(path), key, 3, 10, retries=0)
    module = _embedding_module(
        reader, "cuda", overlap=False, native=native, hostfn=hostfn
    )
    module._stream = torch.cuda.Stream()
    module._ids_ready = torch.cuda.Event()
    module._ready = torch.cuda.Event()
    module._released = torch.cuda.Event()
    module._released.record(torch.cuda.current_stream())
    module._slot = Exl3PLESlot(module._capacity, module._released)
    return module, path


@_needs_cuda
def test_exl3_ple_hostfn_module_matches_the_native_reader_on_the_gpu(
    tmp_path, native_reader
):
    """Real streams, with the current stream held before every prefetch, so
    the host queues the ids copy, read and rows copy of later steps while
    earlier reads have not run (until the ring is full). The ids change every
    step and the table ends inside row 6 for one step. Outputs and failure
    masks match the native reader byte for byte."""
    steps = [
        [[3, 5]],
        [[4, 8], [0, 9], [3, 6]],  # row 0 is unowned, row 9 padding
        [[5, 7], [8, 3]],
        [[6, 4]],
        [[7, 7], [5, 3], [8, 4]],
        [[r % 5, 5 + r % 4] for r in range(_TOKENS)],  # rows 6, 7, 8 fail
        [[8, 6], [3, 4]],
    ]
    results = {}
    for name in ("native", "hostfn"):
        (tmp_path / name).mkdir()
        module, path = _cuda_ssd_embedding(
            tmp_path / name, native=name == "native", hostfn=name == "hostfn"
        )
        ids = [torch.tensor(s, dtype=torch.int64, device="cuda") for s in steps]
        outputs, in_flight = [], []
        try:
            for index, step_ids in enumerate(ids):
                restore = None
                if index == 5:
                    torch.accelerator.synchronize()  # earlier reads see the whole table
                    restore = _truncated(
                        path, module._reader.base + 6 * module._reader.row_bytes + 10
                    )
                torch.cuda._sleep(50_000_000)
                module.start_prefetch(step_ids)
                in_flight.append(module._host_reads_in_flight)
                output = module.finalize_prefetch(step_ids.shape[0])
                outputs.append((output, module.take_failed_tokens()))
                if restore is not None:
                    torch.accelerator.synchronize()  # this step's read has run
                    restore()
            torch.accelerator.synchronize()
            results[name] = [
                (output.view(torch.int16).cpu(), failed.cpu())
                for output, failed in outputs
            ]
        finally:
            module.close()
        if name == "hostfn":
            assert max(in_flight) >= 2  # the host ran ahead of the reads
    reference = results["native"]
    assert reference[5][1].tolist() == [r % 4 != 0 for r in range(_TOKENS)]
    assert not any(failed.any() for i, (_, failed) in enumerate(reference) if i != 5)
    for (output, failed), (ref_output, ref_failed) in zip(results["hostfn"], reference):
        assert torch.equal(output, ref_output)
        assert torch.equal(failed, ref_failed)


@_needs_cuda
@pytest.mark.parametrize("reader", ["python", "native", "hostfn"])
def test_exl3_ple_external_rows_follow_graph_replays(tmp_path, native_reader, reader):
    """New IDs, padding and read failures must not reuse a prior replay's rows."""
    module, path = _cuda_ssd_embedding(
        tmp_path, native=reader == "native", hostfn=reader == "hostfn"
    )
    steps = [
        ([[3, 5]], 5, False),
        ([[4, 8], [0, 9], [3, 6]], 5, False),
        ([[5, 7], [8, 3]], 3, False),
        ([[6, 4]], 3, True),
        ([], 5, False),
        ([[8, 6], [3, 4]], 5, False),
    ]
    observed = []
    try:
        module.zero_rows(5)
        address = module.view(5).data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        with torch.cuda.stream(stream):
            for padded in (3, 5):
                for _ in range(3):
                    module.view(padded).float().sum(-1)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = module.view(padded).float().sum(-1)
                graphs[padded] = graph, output
        torch.cuda.current_stream().wait_stream(stream)

        for values, padded, truncate in steps:
            ids = torch.tensor(values, dtype=torch.int64, device="cuda").reshape(
                -1, _HEADS
            )
            restore = None
            if truncate:
                torch.accelerator.synchronize()
                restore = _truncated(
                    path, module._reader.base + 6 * module._reader.row_bytes + 10
                )
            try:
                expected = torch.zeros(padded, device="cuda")
                expected_failed = None
                if values:
                    module.start_prefetch(ids)
                    rows = module.finalize_prefetch(len(values))
                    expected[: len(values)] = rows.float().sum(-1)
                    expected_failed = module.take_failed_tokens()
                module.prepare(ids, padded)
                assert module.view(padded).data_ptr() == address
                graph, output = graphs[padded]
                graph.replay()
                observed.append(
                    (
                        output.clone(),
                        expected,
                        module.take_failed_tokens(),
                        expected_failed,
                    )
                )
                assert module.take_failed_tokens() is None
            finally:
                if restore is not None:
                    torch.accelerator.synchronize()
                    restore()
        module.zero_rows(5)
        graphs[5][0].replay()
        torch.accelerator.synchronize()
        assert not graphs[5][1].any()
        assert observed[3][2].tolist() == [True]
        for output, expected, failed, expected_failed in observed:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            if expected_failed is None:
                assert failed is None
            else:
                assert torch.equal(failed, expected_failed)
    finally:
        module.close()


@_needs_cuda
def test_exl3_ngram_rows_survive_full_piecewise_and_eager_transitions(
    tmp_path, native_reader
):
    from vllm.config import VllmConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    backend, _ = _cuda_ssd_embedding(tmp_path, native=True, hostfn=False)
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.ngram_embedding = backend
    module.staged_rows_enabled = True
    module.use_sequence_parallel = False
    module.heads_per_ngram = 1
    module.eos_token_id = 99
    module.register_buffer("layer_multipliers", torch.tensor([3, 5, 7], device="cuda"))
    module.register_buffer("ngram_heads_vocab_sizes", backend.head_vocab_sizes)
    module.register_buffer("ngram_heads_offsets", backend.head_offsets)
    module._prefetch_ids = torch.empty(_TOKENS, _HEADS, dtype=torch.long, device="cuda")
    ids = torch.zeros(2, dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    context = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
    hidden = torch.zeros(2, backend._output_dim, device="cuda")
    config = VllmConfig()
    try:
        backend.zero_rows(2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            backend.view(2).clone()
            with (
                set_forward_context(
                    None,
                    config,
                    num_tokens=2,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ),
                torch.cuda.graph(graph, stream=stream),
            ):
                module.start_prefetch(ids, starts, context)
                captured = module(hidden, ids, starts, context).clone()
        torch.cuda.current_stream().wait_stream(stream)
        steps = [
            (CUDAGraphMode.NONE, [11, 12]),
            (CUDAGraphMode.FULL, [13, 17]),
            (CUDAGraphMode.PIECEWISE, [21, 22]),
            (CUDAGraphMode.FULL, [23, 29]),
            (CUDAGraphMode.NONE, [31, 34]),
        ]
        for mode, tokens in steps:
            ids.copy_(torch.tensor(tokens, dtype=ids.dtype, device="cuda"))
            lookup = module.compute_ngram_ids(ids, starts, context)
            backend.start_prefetch(lookup)
            expected = backend.finalize_prefetch(2)
            backend.take_failed_tokens()
            if mode == CUDAGraphMode.FULL:
                module.prepare_staged_rows(ids, starts, context, 2, 2)
                graph.replay()
                actual = captured
            else:
                with set_forward_context(
                    None, config, num_tokens=2, cudagraph_runtime_mode=mode
                ):
                    module.start_prefetch(ids, starts, context)
                    actual = module(hidden, ids, starts, context)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            backend.take_failed_tokens()
    finally:
        backend.close()
