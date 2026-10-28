# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 Engram table storage: the native row fetch (int64 offsets, the
host row cache and its LRU semantics, TP head ownership, failed rows),
telemetry, dequantization against the reference formula, tables attached
from a table segment holder, the real shards (VLLM_ENGRAM_TEST_TABLE_DIR)
and CUDA graph replay with changing ids (GPU only)."""

import errno
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from torch import nn

from vllm_sm80.models.deepseek_v41 import sm80_engram
from vllm_sm80.models.deepseek_v41.ops.sm80_engram_dequant import (
    dequant_engram_rows,
    dequant_engram_rows_reference,
)
from vllm_sm80.models.deepseek_v41.sm80_engram import (
    EngramRowStore,
    Sm80EngramEmbedding,
    engram_owner_range,
    engram_table_dir,
    locate_engram_table,
)

DIM = 256
SCALE_DIM = DIM // 32
TP = 4

# A table past 2**32 bytes of values: rows from 2**23 sit past byte 2**31,
# rows from 2**24 past byte 2**32.
BIG_HEADS = (2_097_161,) * 8
BIG_ROWS = sum(BIG_HEADS) + 12
BIG_LAYER = 1
SMALL_HEADS = (600,) * 8
SMALL_ROWS = sum(SMALL_HEADS) + 200
SMALL_LAYER = 14


def _row_bytes(rows: np.ndarray, width: int, salt: int) -> np.ndarray:
    """Deterministic bytes of the given rows (every byte value occurs)."""
    rows = rows.astype(np.uint64)[:, None]
    cols = np.arange(width, dtype=np.uint64)[None, :]
    mixed = (
        rows * np.uint64(0x9E3779B97F4A7C15)
        + cols * np.uint64(0xBF58476D1CE4E5B9)
        + np.uint64(salt)
    )
    mixed ^= mixed >> np.uint64(31)
    mixed *= np.uint64(0x94D049BB133111EB)
    return (mixed >> np.uint64(56)).astype(np.uint8)


def _write_table(directory, layer, rows, written, file_name, extra_index=None):
    """A base-checkpoint shard in the real layout (values, then scales, then
    a small tensor), sparse: only `written` rows hold data, the rest are
    holes that read as zeros."""
    values_name = f"layers.{layer}.engram.embed.weight"
    scales_name = f"layers.{layer}.engram.embed.scale"
    q_name = f"layers.{layer}.engram.q_weight"
    values_end = rows * DIM
    scales_end = values_end + rows * SCALE_DIM
    header = {
        values_name: {
            "dtype": "F8_E4M3",
            "shape": [rows, DIM],
            "data_offsets": [0, values_end],
        },
        scales_name: {
            "dtype": "F8_E8M0",
            "shape": [rows, SCALE_DIM],
            "data_offsets": [values_end, scales_end],
        },
        q_name: {
            "dtype": "BF16",
            "shape": [4, 16],
            "data_offsets": [scales_end, scales_end + 128],
        },
    }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    start = 8 + len(raw)
    path = directory / file_name
    written = np.unique(np.asarray(written, dtype=np.int64))
    values = _row_bytes(written, DIM, 1)
    scales = _row_bytes(written, SCALE_DIM, 2)
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        file.truncate(start + scales_end + 128)
        fd = file.fileno()
        for i, row in enumerate(written.tolist()):
            os.pwrite(fd, values[i].tobytes(), start + row * DIM)
            os.pwrite(fd, scales[i].tobytes(), start + values_end + row * SCALE_DIM)
    index_path = directory / "model.safetensors.index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"metadata": {}, "weight_map": {}}
    )
    for name in (values_name, scales_name, q_name):
        index["weight_map"][name] = file_name
    index["weight_map"].update(extra_index or {})
    index_path.write_text(json.dumps(index))
    return path


def _owner_ranges(heads):
    return [engram_owner_range(heads, rank, TP) for rank in range(TP)]


def _big_written():
    rng = np.random.default_rng(0)
    rows = [rng.integers(0, BIG_ROWS, 4000)]
    boundary = [
        0,
        1,
        2**23 - 1,
        2**23,
        2**23 + 1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        BIG_ROWS - 1,
    ]
    for _, _, lo, hi in _owner_ranges(BIG_HEADS):
        boundary += [lo, lo + 1, hi - 1]
    return np.concatenate([np.concatenate(rows), np.asarray(boundary)])


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    directory = tmp_path_factory.mktemp("engram_tables")
    big_written = _big_written()
    _write_table(
        directory, BIG_LAYER, BIG_ROWS, big_written, "model-00047-of-00048.safetensors"
    )
    _write_table(
        directory,
        SMALL_LAYER,
        SMALL_ROWS,
        np.arange(SMALL_ROWS),
        "model-00048-of-00048.safetensors",
    )
    return SimpleNamespace(
        dir=directory,
        big=locate_engram_table(directory, BIG_LAYER, BIG_ROWS, DIM),
        small=locate_engram_table(directory, SMALL_LAYER, SMALL_ROWS, DIM),
        big_written=np.unique(big_written),
    )


def _independent_rows(path, layer, rows):
    """Values and scales bytes of `rows` through the safetensors library."""
    values, scales = [], []
    with safe_open(str(path), framework="pt") as file:
        weight = file.get_slice(f"layers.{layer}.engram.embed.weight")
        scale = file.get_slice(f"layers.{layer}.engram.embed.scale")
        for row in np.asarray(rows).tolist():
            values.append(weight[row : row + 1].view(torch.uint8))
            scales.append(scale[row : row + 1].view(torch.uint8))
    return torch.cat(values).numpy(), torch.cat(scales).numpy()


def _expected(files, layer, ids, lo, hi):
    ids = np.asarray(ids, dtype=np.int64)
    values = np.zeros((len(ids), DIM), dtype=np.uint8)
    scales = np.full((len(ids), SCALE_DIM), 127, dtype=np.uint8)
    owned = (ids >= lo) & (ids < hi)
    if owned.any():
        v, s = _independent_rows(files.values_path, layer, ids[owned])
        values[owned], scales[owned] = v, s
    return values, scales


# ---------------------------------------------------------------------------
# Table files
# ---------------------------------------------------------------------------


def test_locate_engram_table_offsets(tables):
    files = tables.big
    with open(files.values_path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
    assert files.values_offset == 8 + length
    assert files.scales_offset == 8 + length + BIG_ROWS * DIM
    assert files.values_path == files.scales_path
    assert files.scales_offset > 2**32


def test_locate_engram_table_refuses_mismatch(tables, tmp_path):
    with pytest.raises(ValueError, match="expected"):
        locate_engram_table(tables.dir, BIG_LAYER, BIG_ROWS + 1, DIM)
    with pytest.raises(ValueError, match="lists no"):
        locate_engram_table(tables.dir, 2, BIG_ROWS, DIM)
    path = _write_table(tmp_path, 3, 64, np.arange(64), "short.safetensors")
    os.truncate(path, os.path.getsize(path) - 200)
    with pytest.raises(ValueError, match="truncated"):
        locate_engram_table(tmp_path, 3, 64, DIM)


def test_engram_table_dir_from_model_config():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(engram_table_dir=None),
        hf_text_config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="engram_table_dir"):
        engram_table_dir(config)
    config.hf_config.engram_table_dir = "/tables"
    assert engram_table_dir(config) == "/tables"


def test_owner_ranges_match_upstream_head_sharding():
    heads = tuple(range(100, 124))
    ranges = _owner_ranges(heads)
    assert [r[:2] for r in ranges] == [(6, 0), (6, 6), (6, 12), (6, 18)]
    assert ranges[0][2] == 0 and ranges[-1][3] == sum(heads)
    for (_, _, _, hi), (_, _, lo, _) in zip(ranges, ranges[1:]):
        assert hi == lo
    with pytest.raises(ValueError, match="without hash heads"):
        engram_owner_range((1, 2, 3), 0, 4)


# ---------------------------------------------------------------------------
# Native fetch: parity, cache, telemetry
# ---------------------------------------------------------------------------


def _store(files, lo, hi, cache_bytes=1 << 20, placement="ssd", capacity=4096):
    return EngramRowStore(
        files, lo, hi, placement=placement, cache_bytes=cache_bytes, capacity=capacity
    )


def _fetch(store, ids):
    values, scales = store.fetch(np.asarray(ids, dtype=np.int64))
    return values.copy(), scales.copy()


@pytest.mark.parametrize("rank", range(TP))
def test_fetch_row_parity_per_rank(tables, rank):
    _, _, lo, hi = engram_owner_range(BIG_HEADS, rank, TP)
    store = _store(tables.big, lo, hi)
    try:
        written = tables.big_written
        mine = written[(written >= lo) & (written < hi)]
        holes = np.array([lo + 7, hi - 7])
        others = written[(written < lo) | (written >= hi)][:50]
        ids = np.concatenate([mine, holes, others, mine[:20], [-1, BIG_ROWS + 5]])
        for _ in range(2):  # cold, then warm through the cache
            values, scales = _fetch(store, ids)
            want_values, want_scales = _expected(tables.big, BIG_LAYER, ids, lo, hi)
            np.testing.assert_array_equal(values, want_values)
            np.testing.assert_array_equal(scales, want_scales)
        counts = store.counters()
        assert counts["cache_hits"] > 0
        assert counts["rows_unowned"] == 2 * (len(others) + 2)
    finally:
        store.close()


def test_fetch_reads_past_int32_offsets(tables):
    rows = np.array(
        [2**23 - 1, 2**23, 2**23 + 1, 2**24 - 1, 2**24, 2**24 + 1, BIG_ROWS - 1]
    )
    store = _store(tables.big, 0, BIG_ROWS, cache_bytes=0)
    try:
        values, scales = _fetch(store, rows)
        want_values, want_scales = _independent_rows(
            tables.big.values_path, BIG_LAYER, rows
        )
        np.testing.assert_array_equal(values, want_values)
        np.testing.assert_array_equal(scales, want_scales)
        assert values.any()
    finally:
        store.close()


@contextmanager
def _default_device_set(device):
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(None)


@pytest.mark.parametrize("context", [torch.device, _default_device_set])
def test_fetch_ignores_the_default_device(tables, context):
    """Model construction runs under a default device (CUDA when served):
    the fetch's tensors must stay on the CPU. The meta device reproduces a
    non-CPU default without a GPU."""
    _, _, lo, hi = engram_owner_range(BIG_HEADS, 1, TP)
    written = tables.big_written
    ids = np.concatenate([written[(written >= lo) & (written < hi)], [lo - 1]])
    with context("meta"):
        assert torch.empty(1).device.type == "meta"
        store = _store(tables.big, lo, hi, cache_bytes=0)
        try:
            values, scales = _fetch(store, ids)
        finally:
            store.close()
    want_values, want_scales = _expected(tables.big, BIG_LAYER, ids, lo, hi)
    np.testing.assert_array_equal(values, want_values)
    np.testing.assert_array_equal(scales, want_scales)


def test_ram_placement_matches_ssd(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 1, TP)
    ssd = _store(tables.small, lo, hi, cache_bytes=1 << 16)
    ram = _store(tables.small, lo, hi, cache_bytes=1 << 30, placement="ram")
    try:
        ids = np.random.default_rng(3).integers(-5, SMALL_ROWS + 5, 3000)
        for a, b in zip(_fetch(ssd, ids), _fetch(ram, ids)):
            np.testing.assert_array_equal(a, b)
        assert ram.cache_bytes == 0 and ram.table.shape == (hi - lo, DIM)
    finally:
        ssd.close()
        ram.close()


class _LruModel:
    """The row cache semantics the native fetch keeps: Fibonacci-hashed
    sets, one clock tick per probe with hits and per insert, a set takes at
    most `ways` new rows per insert, in LRU order of its stamps."""

    def __init__(self, num_sets, ways):
        self.num_sets, self.ways, self.clock = num_sets, ways, 0
        self.tags = np.full((num_sets, ways), -1, dtype=np.int64)
        self.stamps = np.zeros((num_sets, ways), dtype=np.int64)

    def set_of(self, rows):
        mixed = (rows.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(
            32
        )
        return (mixed % np.uint64(self.num_sets)).astype(np.int64)

    def fetch(self, local):
        unique = np.unique(local)
        sets = self.set_of(unique)
        match = self.tags[sets] == unique[:, None]
        hit = match.any(axis=1)
        if hit.any():
            self.clock += 1
            self.stamps[sets[hit], match.argmax(axis=1)[hit]] = self.clock
        miss, miss_sets = unique[~hit], sets[~hit]
        inserted = evicted = overflow = 0
        if len(miss):
            order = np.argsort(miss_sets, kind="stable")
            self.clock += 1
            lru = {}
            ranks = {}
            for m in order:
                s = int(miss_sets[m])
                if s not in lru:
                    lru[s] = np.argsort(self.stamps[s], kind="stable")
                rank = ranks.get(s, 0)
                ranks[s] = rank + 1
                if rank >= self.ways:
                    overflow += 1
                    continue
                way = lru[s][rank]
                evicted += int(self.tags[s, way] >= 0)
                self.tags[s, way] = miss[m]
                self.stamps[s, way] = self.clock
                inserted += 1
        return int(hit.sum()), len(miss), inserted, evicted, overflow


def test_cache_semantics_match_the_lru_model(tables):
    """Per-fetch hits, misses, inserts and evictions of a tiny cache equal
    the LRU model's over random batches with collisions and overflow."""
    lo, hi = 0, 600
    num_sets, ways = 5, 8
    store = _store(tables.small, lo, hi, cache_bytes=num_sets * ways * (DIM + 16))
    model = _LruModel(num_sets, ways)
    rng = np.random.default_rng(11)
    seen = {"evicted": 0, "overflow": 0}
    try:
        assert store.fetcher.cache_rows() == num_sets * ways
        previous = store.counters()
        for _ in range(300):
            ids = rng.integers(lo - 20, lo + 180, int(rng.integers(1, 120)))
            values, _ = _fetch(store, ids)
            want, _ = _expected(tables.small, SMALL_LAYER, ids, lo, hi)
            np.testing.assert_array_equal(values, want)
            owned = ids[(ids >= lo) & (ids < hi)] - lo
            hits, misses, inserted, evicted, overflow = model.fetch(owned)
            counts = store.counters()
            delta = {k: counts[k] - previous[k] for k in counts}
            previous = counts
            assert (
                delta["cache_hits"],
                delta["cache_misses"],
                delta["cache_inserts"],
                delta["cache_evictions"],
            ) == (hits, misses, inserted, evicted)
            seen["evicted"] += evicted
            seen["overflow"] += overflow
        assert seen["evicted"] > 0 and seen["overflow"] > 0
    finally:
        store.close()


def test_cache_bound(tables):
    for cache_bytes in (0, 1000, 1 << 20, (1 << 20) + 12345):
        store = _store(tables.small, 0, 600, cache_bytes=cache_bytes)
        try:
            rows = cache_bytes // (8 * (DIM + 16)) * 8
            assert store.fetcher.cache_rows() == rows
            assert store.cache_bytes == rows * (DIM + 16) <= cache_bytes
            assert store.prefault() >= 0
            ids = np.arange(0, 64)
            _fetch(store, ids)
            _fetch(store, ids)
            assert store.counters()["cache_hits"] == (64 if rows >= 64 else 0)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Row-cache preload
# ---------------------------------------------------------------------------


def _set_of(local: int, num_sets: int) -> int:
    """The native cache's set of a local row (Fibonacci hashing)."""
    return (((local * 0x9E3779B97F4A7C15) % (1 << 64)) >> 32) % num_sets


def _preload_model(rows: int, num_sets: int, ways: int = 8):
    """The local rows a preload stores, by set (each set keeps its first
    `ways` rows in file order), and how many rows it reads before the cache
    is full (None: it never fills)."""
    sets: dict[int, list[int]] = {}
    stored = 0
    for local in range(rows):
        kept = sets.setdefault(_set_of(local, num_sets), [])
        if len(kept) < ways:
            kept.append(local)
            stored += 1
            if stored == num_sets * ways:
                return sets, local + 1
    return sets, None


def _buffered_preload(monkeypatch):
    monkeypatch.setattr(
        sm80_engram,
        "_open_preload",
        lambda path: (os.open(path, os.O_RDONLY | os.O_CLOEXEC), False),
    )


@pytest.mark.parametrize("direct", [True, False])
def test_preload_stores_file_rows_until_full(tables, monkeypatch, direct):
    """A cache smaller than the share: every set takes its first rows in file
    order, the bytes are the file's, the pass stops once the cache is full
    and no counter moves."""
    if not direct:
        _buffered_preload(monkeypatch)
    files = tables.small
    # Chunks start off the direct-read alignment.
    assert files.values_offset % sm80_engram._DIRECT_ALIGN
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 1, TP)
    num_sets, chunk = 16, 8192
    store = _store(files, lo, hi, cache_bytes=num_sets * 8 * (DIM + 16))
    try:
        sets, full_at = _preload_model(hi - lo, num_sets)
        assert full_at is not None and full_at < hi - lo
        stored, read, used_direct = store.preload(chunk_bytes=chunk)
        assert used_direct is direct
        assert stored == num_sets * 8
        chunks = -(-full_at // (chunk // DIM))
        assert chunks * chunk <= read <= chunks * (chunk + sm80_engram._DIRECT_ALIGN)
        assert not any(store.counters().values())
        ids = np.sort(np.concatenate([np.asarray(v) for v in sets.values()])) + lo
        values, scales = _fetch(store, ids)
        want_values, want_scales = _expected(files, SMALL_LAYER, ids, lo, hi)
        np.testing.assert_array_equal(values, want_values)
        np.testing.assert_array_equal(scales, want_scales)
        counts = store.counters()
        assert counts["cache_hits"] == len(ids)
        assert counts["cache_misses"] == counts["cache_inserts"] == 0
        assert counts["cache_evictions"] == 0
    finally:
        store.close()


@pytest.mark.parametrize("direct", [True, False])
def test_preload_of_a_cache_larger_than_the_share(tables, monkeypatch, direct):
    """Every owned row is stored once, the whole extent is read, the other
    ways stay empty and every owned row is then a hit."""
    if not direct:
        _buffered_preload(monkeypatch)
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 3, TP)
    store = _store(tables.small, lo, hi, cache_bytes=1 << 20)
    try:
        assert store.fetcher.cache_rows() > hi - lo
        stored, read, _ = store.preload(chunk_bytes=4096)
        assert stored == hi - lo
        assert (hi - lo) * DIM <= read
        assert not any(store.counters().values())
        ids = np.arange(lo, hi)
        values, _ = _fetch(store, ids)
        np.testing.assert_array_equal(
            values, _expected(tables.small, SMALL_LAYER, ids, lo, hi)[0]
        )
        counts = store.counters()
        assert counts["cache_hits"] == hi - lo and counts["cache_misses"] == 0
    finally:
        store.close()


def test_demand_rows_evict_preloaded_rows_first(tables):
    """Preloaded rows are older than any demand row: a set's demand inserts
    replace its preloaded rows before one another."""
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 2, TP)
    num_sets = 16
    store = _store(tables.small, lo, hi, cache_bytes=num_sets * 8 * (DIM + 16))
    try:
        sets, _ = _preload_model(hi - lo, num_sets)
        assert store.preload(chunk_bytes=8192)[0] == num_sets * 8
        target = _set_of(0, num_sets)
        others = [
            local
            for local in range(hi - lo)
            if _set_of(local, num_sets) == target and local not in sets[target]
        ]
        demand = others[:8]
        assert len(demand) == 8
        for local in demand:
            _fetch(store, [lo + local])
            # The first demand row is never the one evicted.
            _fetch(store, [lo + demand[0]])
        counts = store.counters()
        assert counts["cache_misses"] == 8 and counts["cache_evictions"] == 8
        assert counts["cache_hits"] == 8
        # The set now holds exactly the demand rows.
        before = store.counters()["cache_hits"]
        values, _ = _fetch(store, lo + np.asarray(demand))
        assert store.counters()["cache_hits"] - before == 8
        np.testing.assert_array_equal(
            values,
            _expected(tables.small, SMALL_LAYER, lo + np.asarray(demand), lo, hi)[0],
        )
        _fetch(store, [lo + sets[target][0]])
        assert store.counters()["cache_misses"] == 9
    finally:
        store.close()


def test_preload_runs_only_before_fetches(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 0, TP)
    store = _store(tables.small, lo, hi)
    try:
        _fetch(store, [lo])
        with pytest.raises(RuntimeError, match="preloaded before any fetch"):
            store.preload()
    finally:
        store.close()
    for placement, cache_bytes in (("ssd", 0), ("ram", 1 << 20)):
        store = _store(tables.small, lo, hi, cache_bytes, placement)
        try:
            assert store.preload() == (0, 0, True)
        finally:
            store.close()


def test_preload_flag_and_log(tables, monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_ENGRAM_ROW_CACHE_PRELOAD", raising=False)
    assert envs.VLLM_ENGRAM_ROW_CACHE_PRELOAD is False
    monkeypatch.setenv("VLLM_ENGRAM_ROW_CACHE_PRELOAD", "1")
    assert envs.VLLM_ENGRAM_ROW_CACHE_PRELOAD is True
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 1, TP)
    store = _store(tables.small, lo, hi, cache_bytes=1 << 20)
    try:
        line = sm80_engram.preload_row_cache(store)
        assert line.startswith(f", preloaded {hi - lo} of ")
        assert "direct reads in " in line
    finally:
        store.close()


def test_page_cache_classification(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 3, TP)
    store = _store(tables.small, lo, hi, cache_bytes=0)
    try:
        resident, total = store.page_cache_residency()
        assert 0 <= resident <= total
        _fetch(store, np.arange(lo, lo + 64))
        counts = store.counters()
        assert counts["page_cache_rows"] + counts["storage_rows"] == 64
        assert counts["read_ns"] > 0 and counts["fetch_ns"] >= counts["read_ns"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Direct (O_DIRECT) reads of missed rows
# ---------------------------------------------------------------------------

ALIGN = 4096
# Values start 128 bytes before a 4 KiB boundary, so row 0 and every 16th
# row after it straddle two blocks; they are the file's last bytes and end
# off a block boundary, so the last rows' spans run past the end of the file.
DIRECT_ROWS = 300
DIRECT_VALUES_PHASE = ALIGN - 128


def _direct_table(directory, layer=5):
    """A shard with the scales first and the values last, the values at
    DIRECT_VALUES_PHASE within a block; every row written."""
    rows = DIRECT_ROWS
    scales_end = rows * SCALE_DIM
    values_end = scales_end + rows * DIM
    header = {
        f"layers.{layer}.engram.embed.scale": {
            "dtype": "F8_E8M0",
            "shape": [rows, SCALE_DIM],
            "data_offsets": [0, scales_end],
        },
        f"layers.{layer}.engram.embed.weight": {
            "dtype": "F8_E4M3",
            "shape": [rows, DIM],
            "data_offsets": [scales_end, values_end],
        },
    }
    raw = json.dumps(header).encode()
    raw += b" " * ((DIRECT_VALUES_PHASE - 8 - len(raw) - scales_end) % ALIGN)
    start = 8 + len(raw)
    ids = np.arange(rows)
    path = directory / "direct.safetensors"
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        file.write(_row_bytes(ids, SCALE_DIM, 2).tobytes())
        file.write(_row_bytes(ids, DIM, 1).tobytes())
    assert os.path.getsize(path) == start + values_end
    index = {"metadata": {}, "weight_map": {name: path.name for name in header}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    files = locate_engram_table(directory, layer, rows, DIM)
    assert files.values_offset % ALIGN == DIRECT_VALUES_PHASE
    assert os.path.getsize(path) % ALIGN
    return files


def _direct_expected(ids, lo, hi):
    ids = np.asarray(ids, dtype=np.int64)
    values = np.zeros((len(ids), DIM), dtype=np.uint8)
    scales = np.full((len(ids), SCALE_DIM), 127, dtype=np.uint8)
    owned = (ids >= lo) & (ids < hi)
    values[owned] = _row_bytes(ids[owned], DIM, 1)
    scales[owned] = _row_bytes(ids[owned], SCALE_DIM, 2)
    return values, scales


def _straddling_rows(files):
    offsets = files.values_offset + np.arange(files.rows) * DIM
    return np.flatnonzero(offsets // ALIGN != (offsets + DIM - 1) // ALIGN)


def _direct_store(files, lo, hi, cache_bytes=0, capacity=4096):
    return EngramRowStore(
        files,
        lo,
        hi,
        placement="ssd",
        cache_bytes=cache_bytes,
        capacity=capacity,
        direct_read=True,
    )


@pytest.mark.parametrize("cache_bytes", [0, 1 << 16])
@pytest.mark.parametrize("owner", [(0, DIRECT_ROWS), (5, DIRECT_ROWS - 3)])
def test_direct_reads_match_buffered_reads(tmp_path, owner, cache_bytes):
    """Rows straddling a 4 KiB block, the owner range's first and last rows
    and the file's last rows (spans past the end of the file) come back
    byte for byte as the buffered reads give them."""
    files = _direct_table(tmp_path)
    lo, hi = owner
    straddling = _straddling_rows(files)
    assert straddling[0] == 0 and len(straddling) == -(-DIRECT_ROWS // 16)
    rng = np.random.default_rng(7)
    ids = np.concatenate(
        [
            straddling,
            [lo, hi - 1, DIRECT_ROWS - 2, DIRECT_ROWS - 1, -1, DIRECT_ROWS],
            rng.integers(-3, DIRECT_ROWS + 3, 500),
        ]
    )
    buffered = _store(files, lo, hi, cache_bytes=cache_bytes)
    direct = _direct_store(files, lo, hi, cache_bytes=cache_bytes)
    try:
        want = _direct_expected(ids, lo, hi)
        for _ in range(2):  # cold, then through the cache when there is one
            got = _fetch(direct, ids)
            for a, b, c in zip(got, _fetch(buffered, ids), want):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(a, b)
        counts = direct.counters()
        assert counts["rows_failed"] == 0
        assert counts["page_cache_rows"] == 0
        assert counts["storage_rows"] == counts["cache_misses"] > 0
    finally:
        buffered.close()
        direct.close()


def test_direct_reads_on_every_read_thread(tmp_path):
    """A fetch large enough for every read thread: each thread reads through
    its own span."""
    files = _direct_table(tmp_path)
    store = _direct_store(files, 0, DIRECT_ROWS)
    try:
        ids = np.random.default_rng(11).permutation(DIRECT_ROWS)
        assert len(ids) >= 4 * sm80_engram.READ_THREADS
        got = _fetch(store, ids)
        for a, b in zip(got, _direct_expected(ids, 0, DIRECT_ROWS)):
            np.testing.assert_array_equal(a, b)
        assert store.counters()["storage_rows"] == DIRECT_ROWS
    finally:
        store.close()


def test_direct_reads_leave_the_page_cache_alone(tmp_path):
    files = _direct_table(tmp_path)
    fd = os.open(files.values_path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    ids = np.arange(0, DIRECT_ROWS, 7)
    direct = _direct_store(files, 0, DIRECT_ROWS)
    try:
        before, total = direct.page_cache_residency()
        assert before < total
        _fetch(direct, ids)
        assert direct.page_cache_residency()[0] == before
    finally:
        direct.close()
    buffered = _store(files, 0, DIRECT_ROWS, cache_bytes=0)
    try:
        _fetch(buffered, ids)
        assert buffered.page_cache_residency()[0] > before
    finally:
        buffered.close()


def test_direct_reads_refuse_instead_of_falling_back(tmp_path, monkeypatch):
    files = _direct_table(tmp_path)
    with pytest.raises(ValueError, match="placement ssd only"):
        EngramRowStore(
            files,
            0,
            DIRECT_ROWS,
            placement="ram",
            cache_bytes=0,
            capacity=16,
            direct_read=True,
        )
    # A file whose first owned row lies past its end: the startup read fails.
    past_end = sm80_engram.EngramTableFiles(
        values_path=files.values_path,
        values_offset=os.path.getsize(files.values_path),
        scales_path=files.scales_path,
        scales_offset=files.scales_offset,
        rows=DIRECT_ROWS,
        dim=DIM,
    )
    with pytest.raises(RuntimeError, match="Direct reads of the Engram table fail"):
        _direct_store(past_end, 0, DIRECT_ROWS)
    # A file system that refuses O_DIRECT at open.
    real_open = os.open

    def no_direct(path, flags, *args, **kwargs):
        if flags & os.O_DIRECT:
            raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_direct)
    with pytest.raises(ValueError, match="refuses O_DIRECT"):
        _direct_store(files, 0, DIRECT_ROWS)


def test_direct_read_flag(tmp_path, monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_ENGRAM_DIRECT_READ", raising=False)
    assert envs.VLLM_ENGRAM_DIRECT_READ is False
    monkeypatch.setenv("VLLM_ENGRAM_DIRECT_READ", "1")
    assert envs.VLLM_ENGRAM_DIRECT_READ is True
    files = _direct_table(tmp_path)
    table = Sm80EngramEmbedding(
        DIRECT_ROWS,
        DIM,
        (DIRECT_ROWS // 4,) * 4,
        files,
        tp_rank=2,
        tp_size=TP,
        max_tokens=8,
        placement="ssd",
        cache_bytes=0,
        direct_read=True,
    )
    try:
        assert table.store.direct_read
        ids = np.array([table.vocab_start_idx, table.vocab_end_idx - 1])
        for a, b in zip(
            _fetch(table.store, ids),
            _direct_expected(ids, table.vocab_start_idx, table.vocab_end_idx),
        ):
            np.testing.assert_array_equal(a, b)
    finally:
        table.close()


# ---------------------------------------------------------------------------
# Resident tables: exact size, past the page cache, startup refusal
# ---------------------------------------------------------------------------


def _evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _file_residency(path):
    from vllm_sm80.models.qwen4_exp.exl3_ple import page_cache_residency

    fd = os.open(path, os.O_RDONLY)
    try:
        return page_cache_residency(fd, 0, os.path.getsize(path))
    finally:
        os.close(fd)


@pytest.mark.parametrize("owner", [(0, DIRECT_ROWS), (5, DIRECT_ROWS - 3)])
def test_ram_placement_reads_exact_tables_past_the_page_cache(tmp_path, owner):
    """The resident values and scales are the file's bytes, take exactly
    their size and leave the page cache as it was; lookups match ssd."""
    files = _direct_table(tmp_path)
    lo, hi = owner
    _evict(files.values_path)
    before, total = _file_residency(files.values_path)
    assert before < total
    ram = _store(files, lo, hi, cache_bytes=0, placement="ram")
    try:
        assert _file_residency(files.values_path)[0] == before
        assert ram.direct_load
        want_values, want_scales = _direct_expected(np.arange(lo, hi), lo, hi)
        np.testing.assert_array_equal(ram.table.numpy(), want_values)
        np.testing.assert_array_equal(ram.scales.numpy(), want_scales)
        assert ram.table.untyped_storage().nbytes() == (hi - lo) * DIM
        assert ram.scales.untyped_storage().nbytes() == (hi - lo) * SCALE_DIM
        ids = np.random.default_rng(5).integers(-3, DIRECT_ROWS + 3, 700)
        ssd = _store(files, lo, hi, cache_bytes=0)
        try:
            for a, b, c in zip(
                _fetch(ram, ids), _fetch(ssd, ids), _direct_expected(ids, lo, hi)
            ):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(a, b)
        finally:
            ssd.close()
    finally:
        ram.close()


def test_ram_placement_buffered_fallback_drops_its_pages(tmp_path, monkeypatch):
    _buffered_preload(monkeypatch)
    files = _direct_table(tmp_path)
    _evict(files.values_path)
    before = _file_residency(files.values_path)[0]
    ram = _store(files, 0, DIRECT_ROWS, cache_bytes=0, placement="ram")
    try:
        assert not ram.direct_load
        assert _file_residency(files.values_path)[0] == before
        np.testing.assert_array_equal(
            ram.table.numpy(),
            _direct_expected(np.arange(DIRECT_ROWS), 0, DIRECT_ROWS)[0],
        )
    finally:
        ram.close()


def test_resident_tables_bypass_the_pinned_allocator(tmp_path, monkeypatch):
    """Only the small staging goes through the (power-of-two rounding)
    pinned allocator: resident values and scales never do."""
    calls = []

    def recording(shape, dtype, pin):
        calls.append((tuple(shape), pin))
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(sm80_engram, "_host_empty", recording)
    files = _direct_table(tmp_path)
    store = EngramRowStore(
        files, 0, DIRECT_ROWS, placement="ram", cache_bytes=0, capacity=16, pin=True
    )
    try:
        assert {shape[0] for shape, _ in calls} == {16}
        assert all(pin for _, pin in calls)
    finally:
        store.close()


def _fake_host(tmp_path, available_gib, limits):
    """A /proc and cgroup tree: MemTotal 256 GiB, MemAvailable as given and
    cgroups from the root down to the process's, each (memory.max or
    "max", memory.current, file) in GiB."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(
        f"MemTotal: {256 << 20} kB\nMemAvailable: {int(available_gib * (1 << 20))} kB\n"
    )
    root = tmp_path / "cgroup"
    group = root
    for name, (limit, current, file) in limits:
        group = group / name
        group.mkdir(parents=True)
        (group / "memory.max").write_text(
            "max\n" if limit == "max" else f"{int(limit * GiB)}\n"
        )
        (group / "memory.current").write_text(f"{int(current * GiB)}\n")
        (group / "memory.stat").write_text(f"anon 0\nfile {int(file * GiB)}\n")
    (proc / "self" / "cgroup").write_text(f"0::/{group.relative_to(root).as_posix()}\n")
    return str(proc), str(root)


GiB = 1 << 30


def test_host_memory_headroom_takes_the_binding_limit(tmp_path):
    proc, root = _fake_host(
        tmp_path,
        200,
        [("pool.slice", (222, 60, 20)), ("job.scope", ("max", 30, 10))],
    )
    headroom = sm80_engram.host_memory_headroom(proc, root)
    # 222 - (60 - 20) = 182 < 200.
    assert headroom.bytes == 182 * GiB
    assert headroom.limit == "/pool.slice"
    assert headroom.limit_bytes == 222 * GiB and headroom.host_total == 256 * GiB
    assert "holds back 34.00 GiB" in headroom.describe()
    proc, root = _fake_host(tmp_path / "b", 150, [("pool.slice", (222, 10, 5))])
    headroom = sm80_engram.host_memory_headroom(proc, root)
    assert (headroom.bytes, headroom.limit) == (150 * GiB, "MemAvailable")


def test_ram_placement_refuses_what_the_host_cannot_hold(tmp_path, monkeypatch):
    proc, root = _fake_host(tmp_path, 250, [("pool.slice", (222, 40, 0))])
    measure = sm80_engram.host_memory_headroom
    monkeypatch.setattr(
        sm80_engram,
        "host_memory_headroom",
        lambda: measure(proc, root),
    )

    def four_ranks(entry):
        return [entry] * 4

    # 182 GiB left: four 43 GiB shares plus 2 GiB each fit, 44 GiB do not.
    sm80_engram.check_ram_placement_fits(43 * GiB, four_ranks)
    with pytest.raises(ValueError) as refusal:
        sm80_engram.check_ram_placement_fits(44 * GiB, four_ranks)
    message = str(refusal.value)
    assert "is 2.00 GiB short" in message
    assert "need 176.00 GiB plus 8 GiB margin" in message
    assert "holds back 34.00 GiB" in message
    # The builder checks before it allocates the table.
    monkeypatch.setattr(
        sm80_engram,
        "check_ram_placement_fits",
        lambda need, gather: (_ for _ in ()).throw(ValueError("refused")),
    )
    monkeypatch.setenv("VLLM_ENGRAM_PLACEMENT", "ram")
    layout = SimpleNamespace(
        layer_ids=(14,),
        num_embeddings=(SMALL_ROWS,),
        primes=((SMALL_HEADS,),),
        head_dim=DIM,
    )
    config = SimpleNamespace(
        engram_config=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(engram_table_dir=str(tmp_path)),
            hf_text_config=SimpleNamespace(),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    built = []
    monkeypatch.setattr(
        sm80_engram, "Sm80EngramEmbedding", lambda *a, **k: built.append(1)
    )
    monkeypatch.setattr(sm80_engram, "locate_engram_table", lambda *a: None)
    import vllm.config
    import vllm.distributed

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(vllm.distributed, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: TP
    )
    with pytest.raises(ValueError, match="refused"):
        sm80_engram.build_sm80_engram_embedding(layout, 0)
    assert not built


def test_telemetry_readable_and_logged(tables, monkeypatch):
    lines = []
    monkeypatch.setattr(
        sm80_engram.logger, "info", lambda msg, *args: lines.append(msg % args)
    )
    table = Sm80EngramEmbedding(
        SMALL_ROWS,
        DIM,
        SMALL_HEADS,
        tables.small,
        tp_rank=1,
        tp_size=TP,
        max_tokens=16,
        placement="ssd",
        cache_bytes=1 << 20,
        label="layer 14",
        log_every=2,
    )
    sm80_engram._TABLES.add(table)
    try:
        for seed in range(5):
            indices = _hash_like_indices(SMALL_HEADS, np.arange(SMALL_ROWS), 16, seed)
            table.lookup(indices, torch.empty(16, 2, DIM, dtype=torch.bfloat16))
            table.take_failed_tokens()  # the runner's once-per-step take
        counts = sm80_engram.engram_telemetry()["layer 14"]
        assert counts["lookups"] == 5 and counts["rows_requested"] == 5 * 32
        assert counts["host_ns"] >= counts["fetch_ns"] > 0
        assert counts["cache_hits"] + counts["cache_misses"] == counts["rows_unique"]
        assert counts["cache_misses"] == (
            counts["page_cache_rows"] + counts["storage_rows"]
        )
        assert counts["page_cache_bytes"] + counts["storage_bytes"] == (
            counts["cache_misses"] * DIM
        )
        assert len(lines) == 2
        assert all(
            line.startswith("sm_80 Engram layer 14 telemetry: ") for line in lines
        )
        assert "lookups 4 " in lines[1] and "; last lookups 2 " in lines[1]
    finally:
        table.close()
    assert lines[-1].startswith("sm_80 Engram layer 14 telemetry at close: lookups 5 ")
    assert "; last lookups 1 " in lines[-1]


# ---------------------------------------------------------------------------
# The module and dequantization
# ---------------------------------------------------------------------------


def _reference_lookup(files, layer, heads, rank, indices):
    """ParallelEngramEmbedding.forward of the reference for this rank's heads:
    out-of-range ids give zero rows."""
    part, head_start, lo, hi = engram_owner_range(heads, rank, TP)
    cols = indices[:, head_start : head_start + part].reshape(-1).numpy()
    values, scales = _expected(files, layer, cols, lo, hi)
    values = torch.from_numpy(values).view(torch.float8_e4m3fn)
    scales = torch.from_numpy(scales).view(torch.float8_e8m0fnu)
    rows = values.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)
    owned = torch.from_numpy((cols >= lo) & (cols < hi))
    rows = rows.flatten(-2).to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
    return rows.view(indices.shape[0], part, DIM)


def _hash_like_indices(heads, written, tokens, seed):
    """Per-head ids inside each head's bucket range, from the written rows,
    with dead (-1) entries."""
    rng = np.random.default_rng(seed)
    offsets = np.cumsum((0, *heads[:-1]))
    columns = []
    for head, (offset, size) in enumerate(zip(offsets, heads)):
        inside = written[(written >= offset) & (written < offset + size)]
        pick = rng.choice(inside, tokens) if len(inside) else np.full(tokens, offset)
        columns.append(pick)
    indices = np.stack(columns, axis=1)
    indices[rng.random(indices.shape) < 0.05] = -1
    return torch.from_numpy(indices.astype(np.int32))


@pytest.mark.parametrize("rank", range(TP))
def test_module_lookup_matches_reference(tables, rank):
    table = Sm80EngramEmbedding(
        BIG_ROWS,
        DIM,
        BIG_HEADS,
        tables.big,
        tp_rank=rank,
        tp_size=TP,
        max_tokens=64,
        placement="ssd",
        cache_bytes=1 << 20,
    )
    try:
        assert list(table.parameters()) == [] and table.state_dict() == {}
        assert (table.cpu_offload, table.dp_size, table.part_n_hash_cols) == (
            True,
            1,
            2,
        )
        for seed in range(3):
            indices = _hash_like_indices(BIG_HEADS, tables.big_written, 64, seed)
            out = torch.full((64, 2, DIM), float("nan"), dtype=torch.bfloat16)
            table.lookup(indices, out)
            want = _reference_lookup(tables.big, BIG_LAYER, BIG_HEADS, rank, indices)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16))
        with pytest.raises(ValueError, match="capacity"):
            table.lookup(
                torch.zeros(65, 8, dtype=torch.int32),
                torch.empty(65, 2, DIM, dtype=torch.bfloat16),
            )
        assert table.telemetry()["lookups"] == 3
    finally:
        table.close()


def _kernel_decode_emulation(value_bytes, scale_bytes):
    """The Triton kernel's bit formula in numpy FP32 (no torch float8)."""
    x = value_bytes.astype(np.uint32)
    bits = ((x & 0x7F) << 20) | ((x & 0x80) << 24)
    value = bits.view(np.float32) * np.float32(2.0**120)
    value = np.where((x & 0x7F) == 0x7F, np.float32(np.nan), value)
    s = scale_bytes.astype(np.uint32)
    scale = (s << 23).view(np.float32)
    scale = np.where(s == 0, np.array([0x400000], np.uint32).view(np.float32)[0], scale)
    scale = np.where(s == 0xFF, np.float32(np.nan), scale)
    with np.errstate(over="ignore", invalid="ignore"):
        product = (value * scale).astype(np.float32)
    # Round to nearest even on the bits, as the kernel's conversion does;
    # every NaN is written as 0xFFFF.
    bits = product.view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    bits = np.where(np.isnan(product), 0xFFFF, bits).astype(np.uint16)
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)


def _assert_bitwise(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape
    bits = torch.int32 if want.dtype == torch.float32 else torch.int16
    nan = want.isnan()
    assert torch.equal(got.isnan(), nan)
    assert torch.equal(got.view(bits)[~nan], want.view(bits)[~nan])


def _all_byte_pairs():
    """Every (value byte, scale byte) pair, laid out as [256 scales, 256]
    rows: row s holds value bytes 0..255 with every block scale s."""
    values = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    scales = np.repeat(np.arange(256, dtype=np.uint8)[:, None], SCALE_DIM, axis=1)
    return values, scales


def test_kernel_decode_formula_matches_reference_exhaustively():
    values, scales = _all_byte_pairs()
    want = dequant_engram_rows_reference(
        torch.from_numpy(values), torch.from_numpy(scales)
    )
    got = _kernel_decode_emulation(values, np.repeat(scales, 32, axis=1))
    # Every bit, NaN outputs included: the reference writes NaN as 0xFFFF.
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    nan = want.isnan()
    assert nan.sum() == 2 * 256 + 2 * 127 and (want.view(torch.int16)[nan] == -1).all()
    # Denormal products, the 2**-127 scale and saturation are all exercised.
    assert (want[0] != 0).any() and want.isinf().any()


def test_cpu_dequant_is_the_reference_formula():
    values, scales = _all_byte_pairs()
    out = torch.empty(256, DIM, dtype=torch.bfloat16)
    dequant_engram_rows(torch.from_numpy(values), torch.from_numpy(scales), out)
    ref = (
        torch.from_numpy(values).view(torch.float8_e4m3fn).float().view(256, 8, 32)
        * torch.from_numpy(scales).view(torch.float8_e8m0fnu).float()[..., None]
    )
    _assert_bitwise(out, ref.view(256, DIM).to(torch.bfloat16))


def test_create_embedding_hook_dispatch(monkeypatch):
    calls = []

    def upstream(engram, layout, index):
        calls.append(("upstream", index))
        return "upstream"

    wrapped = sm80_engram.sm80_engram_embedding(upstream)
    monkeypatch.setattr("vllm_sm80.kernels._platform._SM80_CACHE", False)
    assert wrapped(None, None, 0) == "upstream"
    monkeypatch.setattr("vllm_sm80.kernels._platform._SM80_CACHE", True)
    table = SimpleNamespace(mask_failed_rows=lambda rows: -rows)
    monkeypatch.setattr(
        sm80_engram,
        "build_sm80_engram_embedding",
        lambda layout, index: table,
    )
    engram = SimpleNamespace(embed=lambda ids: ids + 1)
    assert wrapped(engram, None, 1) is table
    # The gathered rows pass through the table's failure mask.
    assert engram.embed(torch.ones(2)).tolist() == [-2.0, -2.0]
    assert calls == [("upstream", 0)]


def test_unreadable_rows_fail_only_their_requests(tmp_path):
    """A short read on a chosen row: its owner's token comes out zero and is
    reported on rank 0, the other tokens stay byte-exact, nothing raises."""
    from vllm_sm80.failed_inputs import failed_request_ids, take_failed_tokens

    heads, rows, layer, tokens = (300,) * 8, 2400, 14, 12
    _write_table(tmp_path, layer, rows, np.arange(rows), "model.safetensors")
    files = locate_engram_table(tmp_path, layer, rows, DIM)
    short_row = 2397  # rank 3 (head 7)
    good = np.arange(2396)
    # A table row whose first value dequantizes to NaN (an E4M3 NaN value or
    # an E8M0 NaN scale) would fail its token too.
    good = good[(_row_bytes(good, DIM, 1)[:, 0] & 0x7F) != 0x7F]
    good = good[_row_bytes(good, SCALE_DIM, 2)[:, 0] != 0xFF]
    indices = _hash_like_indices(heads, good, tokens, seed=5)
    indices[5, 7] = short_row  # request b
    want = torch.cat(
        [_reference_lookup(files, layer, heads, r, indices) for r in range(TP)], 1
    )
    tables = [
        Sm80EngramEmbedding(
            rows,
            DIM,
            heads,
            files,
            tp_rank=rank,
            tp_size=TP,
            max_tokens=tokens,
            placement="ssd",
            cache_bytes=1 << 20,
        )
        for rank in range(TP)
    ]
    try:
        # Cut the file inside row 2396: rows from there on read short.
        os.truncate(files.values_path, files.values_offset + 2396 * DIM + 100)
        outs = []
        for table in tables:
            outs.append(torch.empty(tokens, 2, DIM, dtype=torch.bfloat16))
            table.lookup(indices, outs[-1])
        engram = SimpleNamespace(embed=lambda ids: torch.cat(outs, dim=1))
        sm80_engram.mask_failed_after_gather(engram, tables[0])
        got = engram.embed(indices)
        keep = [t for t in range(tokens) if t != 5]
        assert not got[5].any()
        assert torch.equal(got[keep].view(torch.int16), want[keep].view(torch.int16))
        model = nn.Module()
        model.table = tables[0]
        query_start_loc, req_ids = np.array([0, 4, 8, 12]), ["a", "b", "c"]
        mask = take_failed_tokens(model).numpy()
        assert failed_request_ids(mask, query_start_loc, req_ids) == {"b"}
        # Taking clears the mask.
        mask = take_failed_tokens(model).numpy()
        assert failed_request_ids(mask, query_start_loc, req_ids) == set()
        assert tables[3].telemetry()["rows_failed"] == 1
        assert sum(t.telemetry()["rows_failed"] for t in tables[:3]) == 0
        # A failed row is never cached: the next lookup reads it again.
        tables[3].lookup(indices, outs[3])
        assert tables[3].telemetry()["rows_failed"] == 2
    finally:
        for table in tables:
            table.close()


def _table_2400(tmp_path):
    _write_table(tmp_path, 14, 2400, np.arange(2400), "model.safetensors")
    return locate_engram_table(tmp_path, 14, 2400, DIM)


def _cut_after_row_2396(files):
    """Value rows from 2396 on read short (the scales were read at open)."""
    os.truncate(files.values_path, files.values_offset + 2396 * DIM + 100)


def test_fixed_staging_matches_the_eager_fetch(tmp_path):
    """The fixed (graph) fetch gives the eager fetch's bytes and counters for
    any row count up to its capacity, poisons the same rows, publishes them
    on its CPU counter (take_fixed reads the native record only when it
    moved), and leaves the eager staging alone."""
    files = _table_2400(tmp_path)
    store = _store(files, 0, 2400, capacity=64)
    other = _store(files, 0, 2400, capacity=64)
    fresh = _store(files, 0, 2400, capacity=64)
    _cut_after_row_2396(files)
    try:
        assert store.take_fixed() == 0  # no fixed staging: nothing to take
        fixed = store.add_fixed(48)
        assert store.fixed is fixed and fixed.failed_total.item() == 0
        rng = np.random.default_rng(7)
        for step, count in enumerate((48, 1, 17, 48, 30, 6)):
            ids = rng.integers(-3, 2396, count)
            if step == 3:
                ids[5] = 2397  # past the cut: a short read
            store.ids.numpy()[:48] = -7
            fixed.ids.numpy()[:count] = ids
            store.fetcher.run_fixed(count)
            want_values, want_scales = _fetch(other, ids)
            np.testing.assert_array_equal(fixed.values.numpy()[:count], want_values)
            np.testing.assert_array_equal(fixed.scales.numpy()[:count], want_scales)
            assert (store.ids.numpy()[:48] == -7).all()
            assert fixed.failed_total.item() == (1 if step >= 3 else 0)
            failed = store.take_fixed()
            assert failed == (1 if step == 3 else 0)
            if step == 3:
                assert fixed.values.numpy()[5, 0] == 0x7F
                assert not fixed.values.numpy()[5, 1:].any()
        a, b = store.counters(), other.counters()
        for name in (
            "lookups",
            "cache_hits",
            "cache_misses",
            "cache_inserts",
            "rows_failed",
            "rows_unowned",
            "rows_unique",
        ):
            assert a[name] == b[name], name
        with pytest.raises(RuntimeError, match="exceeds the fixed staging"):
            store.fetcher.run_fixed(49)
        with pytest.raises(RuntimeError, match="already added"):
            store.add_fixed(8)
    finally:
        store.close()
        other.close()
    try:
        with pytest.raises(RuntimeError, match="exceeds the fetch capacity"):
            fresh.add_fixed(65)
        with pytest.raises(RuntimeError, match="No Engram fixed staging"):
            fresh.fetcher.run_fixed(1)
    finally:
        fresh.close()


class _GraphEmulator:
    """Stands in for EngramHostFetch.enqueue_fixed on CPU: records the queued
    (fixed) arguments and does what the queued copies and host function do."""

    def __init__(self, store):
        self.store, self.fetcher, self.queued = store, store.fetcher, []

    def __getattr__(self, name):
        return getattr(self.fetcher, name)

    def enqueue_fixed(self, count, dev_ids, dev_values, dev_scales):
        self.queued.append(
            (count, dev_ids.data_ptr(), dev_values.data_ptr(), dev_scales.data_ptr())
        )
        fixed = self.store.fixed
        fixed.ids[:count].copy_(dev_ids[:count])
        self.fetcher.run_fixed(count)
        dev_values[:count].copy_(fixed.values[:count])
        dev_scales[:count].copy_(fixed.scales[:count])


def _graph_tables(tables, max_tokens):
    kwargs = dict(
        tp_rank=2,
        tp_size=TP,
        max_tokens=max_tokens,
        placement="ssd",
        cache_bytes=1 << 20,
    )
    table = Sm80EngramEmbedding(
        BIG_ROWS, DIM, BIG_HEADS, tables.big, graph_capture=True, **kwargs
    )
    eager = Sm80EngramEmbedding(BIG_ROWS, DIM, BIG_HEADS, tables.big, **kwargs)
    return table, eager


def _check_captured(tables, table, eager, emulator, counts, seeds):
    for tokens in counts:
        for seed in seeds:
            indices = _hash_like_indices(BIG_HEADS, tables.big_written, tokens, seed)
            got = torch.full((tokens, 2, DIM), float("nan"), dtype=torch.bfloat16)
            table.lookup(indices, got)
            want = torch.empty_like(got)
            eager.lookup(indices, want)
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), tokens
            ref = _reference_lookup(tables.big, BIG_LAYER, BIG_HEADS, 2, indices)
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), tokens
            assert emulator.queued[-1][0] == tokens * table.part_n_hash_cols
    # Every captured lookup queues the one fixed staging.
    assert len({q[1:] for q in emulator.queued}) == 1


def test_captured_lookup_is_the_eager_lookup(tables, monkeypatch):
    """A lookup inside a capture uses the one fixed staging with the same
    buffers every time, for any token count, and dequantizes to the eager
    lookup's bits (the reference)."""
    table, eager = _graph_tables(tables, 64)
    emulator = _GraphEmulator(table.store)
    table.store.fetcher = emulator
    try:
        # On CPU only the fixed device staging is counted: one capacity.
        cap = table.capacity_rows
        assert table.device_bytes() == cap * (8 + DIM + DIM // 32)
        monkeypatch.setattr(table, "_capturing", lambda: True)
        _check_captured(tables, table, eager, emulator, (4, 16, 64, 1, 33), range(3))
        # The eager staging was never used by the captured lookups.
        assert table.telemetry()["host_ns"] == 0
        # A table built without full graphs refuses a captured lookup.
        monkeypatch.setattr(eager, "_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="without full CUDA graphs"):
            eager.lookup(
                torch.zeros(8, 8, dtype=torch.int32),
                torch.empty(8, 2, DIM, dtype=torch.bfloat16),
            )
    finally:
        table.store.fetcher = emulator.fetcher
        table.close()
        eager.close()


def _round_up(x, q):
    return -(-x // q) * q


def test_captured_lookup_covers_spec_decode_graph_counts(tables, monkeypatch):
    """DSpark (k = 5) with FULL_DECODE_ONLY graphs at capture sizes 6..48:
    the V2 runner captures uniform-decode graphs of round_up(size, q) tokens
    for decode query lengths q = 5 and 6 (cudagraph_utils), up to the
    largest capture size. Every such count, and every capture size, is
    captured on the fixed staging and equals the eager lookup."""
    sizes = range(6, 49)
    counts = set(sizes)
    for q in (5, 6):
        counts |= {_round_up(s, q) for s in sizes if _round_up(s, q) <= max(sizes)}
    counts = sorted(counts)
    assert not set(counts) <= {1, 2, 4, 8, 16, 24, 32, 40, 48}  # more than sizes
    table, eager = _graph_tables(tables, max(sizes))
    emulator = _GraphEmulator(table.store)
    table.store.fetcher = emulator
    try:
        monkeypatch.setattr(table, "_capturing", lambda: True)
        _check_captured(tables, table, eager, emulator, counts, (0,))
    finally:
        table.store.fetcher = emulator.fetcher
        table.close()
        eager.close()


@pytest.mark.parametrize(
    "mode, full",
    [
        ("FULL_DECODE_ONLY", True),
        ("FULL", True),
        ("FULL_AND_PIECEWISE", True),
        ("PIECEWISE", False),
        ("NONE", False),
    ],
)
def test_fixed_staging_follows_the_cudagraph_mode(mode, full):
    """The builder gives a table the fixed staging whenever the cudagraph
    mode captures full graphs, FULL_DECODE_ONLY included."""
    from vllm.config import CUDAGraphMode

    assert sm80_engram.captures_full_graphs(getattr(CUDAGraphMode, mode)) is full
    assert sm80_engram.captures_full_graphs(None) is False


def test_captured_lookup_fails_only_its_requests(tmp_path, monkeypatch):
    """ENG-1 through the fixed staging: a short read poisons the row, the
    gathered rows mask its token, the per-step take reports it once."""
    from vllm_sm80.failed_inputs import failed_request_ids, take_failed_tokens

    heads, tokens = (300,) * 8, 12
    files = _table_2400(tmp_path)
    good = np.arange(2396)
    good = good[(_row_bytes(good, DIM, 1)[:, 0] & 0x7F) != 0x7F]
    good = good[_row_bytes(good, SCALE_DIM, 2)[:, 0] != 0xFF]
    indices = _hash_like_indices(heads, good, tokens, seed=5)
    indices[5, 7] = 2397  # rank 3, request b
    tables = [
        Sm80EngramEmbedding(
            2400,
            DIM,
            heads,
            files,
            tp_rank=rank,
            tp_size=TP,
            max_tokens=tokens,
            placement="ssd",
            cache_bytes=1 << 20,
            graph_capture=True,
        )
        for rank in range(TP)
    ]
    _cut_after_row_2396(files)
    try:
        outs = []
        for table in tables:
            table.store.fetcher = _GraphEmulator(table.store)
            monkeypatch.setattr(table, "_capturing", lambda: True)
            outs.append(torch.empty(tokens, 2, DIM, dtype=torch.bfloat16))
            table.lookup(indices, outs[-1])
        engram = SimpleNamespace(embed=lambda ids: torch.cat(outs, dim=1))
        sm80_engram.mask_failed_after_gather(engram, tables[0])
        got = engram.embed(indices)
        assert not got[5].any() and got[[4, 6]].any()
        model = nn.Module()
        model.table = tables[0]
        mask = take_failed_tokens(model).numpy()
        assert failed_request_ids(mask, np.array([0, 4, 8, 12]), ["a", "b", "c"]) == {
            "b"
        }
        assert tables[3].store.take_fixed() == 1
        assert tables[3].store.take_fixed() == 0
    finally:
        for table in tables:
            table.store.fetcher = table.store.fetcher.fetcher
            table.close()


def test_upstream_engram_carries_the_hook():
    from vllm.models.deepseek_v41.common.engram import Engram

    assert Engram._create_embedding.__wrapped__.__qualname__ == (
        "Engram._create_embedding"
    )


# ---------------------------------------------------------------------------
# Real shards
# ---------------------------------------------------------------------------

_REAL_LAYERS = {1: 384_006_168, 14: 384_016_682}
_REAL_FILES = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
# The base checkpoint directory of the real-shard tests: both Engram shards
# and its model.safetensors.index.json.
_REAL_DIR_ENV = "VLLM_ENGRAM_TEST_TABLE_DIR"


def _real_dir():
    directory = os.environ.get(_REAL_DIR_ENV)
    if not directory or not all(
        os.path.isfile(os.path.join(directory, name)) for name in _REAL_FILES
    ):
        pytest.skip(f"{_REAL_DIR_ENV} with both Engram shards is not set")
    return directory


def _real_head_sizes():
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    config = SimpleNamespace(
        engram_layer_ids=list(_REAL_LAYERS),
        engram_num_embeddings=list(_REAL_LAYERS.values()),
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=DIM,
        engram_vocab_size=16_000_000,
        engram_compressed_vocab_size=99092,
        engram_pad_token_id=2,
    )
    layout = EngramLayout(config)
    return [tuple(p for order in layer for p in order) for layer in layout.primes]


def test_real_shards_parity():
    """1024 seeded random rows plus the rank's first and last rows (the
    table's first and last among them), per layer and rank, cold and then
    through the cache, against the safetensors read."""
    directory = _real_dir()
    rng = np.random.default_rng(1017)
    checked = 0
    for (layer, rows), heads in zip(_REAL_LAYERS.items(), _real_head_sizes()):
        files = locate_engram_table(directory, layer, rows, DIM)
        assert sum(heads) == rows
        for rank in range(TP):
            _, _, lo, hi = engram_owner_range(heads, rank, TP)
            sample = np.concatenate([rng.integers(lo, hi, 1024), [lo, hi - 1]])
            assert rank or lo == 0
            assert rank < TP - 1 or hi == rows
            want_values, want_scales = _independent_rows(
                files.values_path, layer, sample
            )
            store = _store(files, lo, hi, cache_bytes=64 << 20)
            try:
                for _ in range(2):
                    values, scales = _fetch(store, sample)
                    np.testing.assert_array_equal(values, want_values)
                    np.testing.assert_array_equal(scales, want_scales)
                assert store.counters()["cache_hits"] >= len(np.unique(sample))
            finally:
                store.close()
            got = dequant_engram_rows_reference(
                torch.from_numpy(values), torch.from_numpy(scales)
            )
            assert not got.isnan().any()
            checked += len(sample)
    assert checked == 2 * TP * 1026


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_gpu_dequant_kernel_bitwise_exhaustive():
    values, scales = _all_byte_pairs()
    values_t = torch.from_numpy(values).cuda()
    scales_t = torch.from_numpy(scales).cuda()
    out = torch.empty(256, DIM, dtype=torch.bfloat16, device="cuda")
    dequant_engram_rows(values_t, scales_t, out)
    want = dequant_engram_rows_reference(
        torch.from_numpy(values), torch.from_numpy(scales)
    )
    # Every bit, NaN outputs (0xFFFF) included.
    assert torch.equal(out.cpu().view(torch.int16), want.view(torch.int16))
    # Odd row counts take the masked tail.
    tail = torch.empty(37, DIM, dtype=torch.bfloat16, device="cuda")
    dequant_engram_rows(values_t[:37].contiguous(), scales_t[:37].contiguous(), tail)
    _assert_bitwise(tail.cpu(), want[:37])


@needs_cuda
def test_gpu_captured_lookup_replays_the_eager_lookup(tables):
    """Captured with torch.cuda.graph at several token counts on the one fixed
    staging: each replay fetches that replay's ids, byte-equal to the
    reference, with graphs of different counts replayed in turn."""
    table = Sm80EngramEmbedding(
        BIG_ROWS,
        DIM,
        BIG_HEADS,
        tables.big,
        tp_rank=1,
        tp_size=TP,
        max_tokens=16,
        placement="ssd",
        cache_bytes=1 << 20,
        device="cuda",
        graph_capture=True,
    )
    try:
        graphs = {}
        for tokens in (5, 8, 12):
            static_ids = torch.full((tokens, 8), -1, dtype=torch.int32, device="cuda")
            rows = torch.empty(tokens, 2, DIM, dtype=torch.bfloat16, device="cuda")
            table.lookup(static_ids, rows)  # warm-up compiles the dequant
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                table.lookup(static_ids + 0, rows)
            graphs[tokens] = (graph, static_ids, rows)
        for seed in range(4):
            for tokens, (graph, static_ids, rows) in graphs.items():
                indices = _hash_like_indices(
                    BIG_HEADS, tables.big_written, tokens, seed
                )
                static_ids.copy_(indices.cuda())
                graph.replay()
                torch.cuda.synchronize()
                want = _reference_lookup(tables.big, BIG_LAYER, BIG_HEADS, 1, indices)
                assert torch.equal(rows.cpu().view(torch.int16), want.view(torch.int16))
    finally:
        table.close()


@needs_cuda
@pytest.mark.parametrize("rank", [0, 3])
def test_gpu_breakable_graph_replay_with_changing_ids(tables, monkeypatch, rank):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    tokens = 16
    table = Sm80EngramEmbedding(
        BIG_ROWS,
        DIM,
        BIG_HEADS,
        tables.big,
        tp_rank=rank,
        tp_size=TP,
        max_tokens=tokens,
        placement="ssd",
        cache_bytes=1 << 20,
        device="cuda",
    )
    prefetch = torch.cuda.Stream()
    done = torch.cuda.Event()
    static_ids = torch.zeros(tokens, 8, dtype=torch.int32, device="cuda")
    rows = torch.empty(tokens, 2, DIM, dtype=torch.bfloat16, device="cuda")

    # The upstream Engram's prefetch and finish, as eager breaks.
    @eager_break_during_capture
    def start(ids, out):
        prefetch.wait_stream(torch.cuda.current_stream())
        ids.record_stream(prefetch)
        with torch.cuda.stream(prefetch):
            table.lookup(ids, out, background=True)
        done.record(prefetch)

    @eager_break_during_capture
    def finish():
        torch.cuda.current_stream().wait_event(done)

    try:
        torch.cuda.synchronize()
        main = torch.cuda.Stream()
        main.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(main):
            capture = BreakableCUDAGraphCapture()
            with capture:
                ids = static_ids + 0  # stands in for the hash kernel
                start(ids, rows)
                finish()
                result = rows.float() * 2
            assert capture.num_eager_breaks == 2
            steps = [
                _hash_like_indices(BIG_HEADS, tables.big_written, tokens, seed)
                for seed in range(10, 16)
            ]
            outputs = []
            for indices in steps:
                static_ids.copy_(indices.to("cuda", non_blocking=False))
                capture.replay()
                outputs.append(result.clone())  # queued; no host sync between
        torch.cuda.synchronize()
        for indices, got in zip(steps, outputs):
            want = _reference_lookup(tables.big, BIG_LAYER, BIG_HEADS, rank, indices)
            _assert_bitwise(got.cpu(), want.float() * 2)
        assert table.telemetry()["lookups"] == len(steps) + 1
    finally:
        table.close()


# ---------------------------------------------------------------------------
# Tables attached from a table segment holder (VLLM_ENGRAM_SEGMENT)
# ---------------------------------------------------------------------------


@contextmanager
def _segment(tmp_path, paths):
    """A table segment holder of ``paths`` (subprocess), ready."""
    from vllm_sm80 import table_segment

    sock = tmp_path / "seg.sock"
    argv = [sys.executable, table_segment.__file__, "hold", "--socket", str(sock),
            "--label", "test@0"]
    for path in sorted(set(paths)):
        with open(path, "rb") as file:
            argv += ["--file", f"{path}={hashlib.sha256(file.read()).hexdigest()}"]
    proc = subprocess.Popen(argv, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if table_segment.request(str(sock), {"op": "status"})["state"] == "ready":
                    break
            except table_segment.SegmentError:
                pass
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        yield str(sock)
    finally:
        proc.terminate()
        proc.wait(30)


@pytest.mark.parametrize("owner", [(0, DIRECT_ROWS), (5, DIRECT_ROWS - 3)])
def test_ram_placement_attached_matches_read(tmp_path, owner):
    """Attached values and scales are the file's bytes, mapped read-only
    without a copy; lookups match the read ram placement and the file."""
    from vllm_sm80 import table_segment

    files = _direct_table(tmp_path)
    lo, hi = owner
    with _segment(tmp_path, [files.values_path, files.scales_path]) as sock:
        read = _store(files, lo, hi, cache_bytes=0, placement="ram")
        attached = EngramRowStore(
            files, lo, hi, placement="ram", cache_bytes=0, capacity=4096, segment=sock
        )
        try:
            status = table_segment.request(sock, {"op": "status"})
            assert len(status["clients"]) == 1
            np.testing.assert_array_equal(attached.table.numpy(), read.table.numpy())
            np.testing.assert_array_equal(attached.scales.numpy(), read.scales.numpy())
            assert attached.table.shape == (hi - lo, DIM)
            assert attached.cache_bytes == 0
            ids = np.random.default_rng(9).integers(-3, DIRECT_ROWS + 3, 900)
            for a, b, c in zip(
                _fetch(attached, ids), _fetch(read, ids), _direct_expected(ids, lo, hi)
            ):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(a, b)
        finally:
            read.close()
            attached.close()
        deadline = time.monotonic() + 10
        while table_segment.request(sock, {"op": "status"})["clients"]:
            assert time.monotonic() < deadline, "close did not detach"
            time.sleep(0.05)


def test_builder_passes_the_segment_and_skips_the_ram_fit_check(tmp_path, monkeypatch):
    """With VLLM_ENGRAM_SEGMENT the builder hands the socket to the table
    and does not run the ram fit check: an attached table allocates no
    table memory (a host holding the segment would fail the check)."""
    import vllm.config
    import vllm.distributed

    def refuse(*args, **kwargs):
        raise AssertionError("ram fit check ran for an attached table")

    built = []

    def build(*args, **kwargs):
        built.append(kwargs)
        raise RuntimeError("built")

    monkeypatch.setattr(sm80_engram, "check_ram_placement_fits", refuse)
    monkeypatch.setattr(sm80_engram, "Sm80EngramEmbedding", build)
    monkeypatch.setattr(sm80_engram, "locate_engram_table", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    config = SimpleNamespace(
        engram_config=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(engram_table_dir=str(tmp_path)),
            hf_text_config=SimpleNamespace(),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(vllm.distributed, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: TP
    )
    layout = SimpleNamespace(
        layer_ids=(14,),
        num_embeddings=(SMALL_ROWS,),
        primes=((SMALL_HEADS,),),
        head_dim=DIM,
    )
    monkeypatch.setenv("VLLM_ENGRAM_PLACEMENT", "ram")
    sock = str(tmp_path / "holder.sock")
    monkeypatch.setenv("VLLM_ENGRAM_SEGMENT", sock)
    with pytest.raises(RuntimeError, match="built"):
        sm80_engram.build_sm80_engram_embedding(layout, 0)
    assert built[0]["segment"] == sock
    assert built[0]["placement"] == "ram"


def test_segment_refusals(tmp_path, monkeypatch):
    from vllm import envs
    from vllm_sm80.table_segment import SegmentError

    monkeypatch.delenv("VLLM_ENGRAM_SEGMENT", raising=False)
    assert envs.VLLM_ENGRAM_SEGMENT == ""
    files = _direct_table(tmp_path)
    with pytest.raises(ValueError, match="placement ram only"):
        EngramRowStore(files, 0, 8, placement="ssd", cache_bytes=0, capacity=8,
                       segment=str(tmp_path / "seg.sock"))
    with pytest.raises(SegmentError, match="no table segment holder"):
        EngramRowStore(files, 0, 8, placement="ram", cache_bytes=0, capacity=8,
                       segment=str(tmp_path / "seg.sock"))
    with _segment(tmp_path, [files.values_path]) as sock:
        st = os.stat(files.values_path)
        os.utime(files.values_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        with pytest.raises(SegmentError, match="not the file the holder read"):
            EngramRowStore(files, 0, 8, placement="ram", cache_bytes=0, capacity=8,
                           segment=sock)
