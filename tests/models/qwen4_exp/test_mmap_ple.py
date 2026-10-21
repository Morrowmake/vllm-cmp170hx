# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for the mmap PLE backend: mappings, indexing, staging."""

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm.models.qwen4_exp.common.ngram_embedding as ngram_embedding_module
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.nvidia.mmap_ple import Qwen4ExpPLEMmapEmbedding

ROWS, DIM, HEADS, SHARDS, CAPACITY = 10, 4, 2, 4, 6
ROW_BYTES = DIM * 2  # BF16 storage
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _mock_parallel(monkeypatch: pytest.MonkeyPatch, world_size: int = 1) -> None:
    group = SimpleNamespace(
        rank_in_group=0, world_size=world_size, all_reduce=lambda tensor: tensor
    )
    monkeypatch.setattr(ngram_embedding_module, "get_etp_group", lambda: group)
    monkeypatch.setattr(
        ngram_embedding_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size),
    )


def _make_embedding(monkeypatch: pytest.MonkeyPatch, **overrides):
    world_size = overrides.pop("world_size", 1)
    method = overrides.pop(
        "method",
        None,
    )
    _mock_parallel(monkeypatch, world_size)
    return Qwen4ExpPLEMmapEmbedding(
        overrides.pop("rows", ROWS),
        overrides.pop("dim", DIM),
        params_dtype=overrides.pop("dtype", torch.bfloat16),
        padding_size=1,
        prefix="test.ple.ngram_embedding",
        embedding_method=(
            method if method is not None else Qwen4ExpPLEUnquantizedEmbeddingMethod()
        ),
        num_ngram_heads=overrides.pop("heads", HEADS),
        max_total_tokens=overrides.pop("capacity", CAPACITY),
        num_shards=overrides.pop("num_shards", SHARDS),
        **overrides,
    )


def _dense_table(**kwargs) -> torch.Tensor:
    # Integer-valued rows: row r, channel c -> 16*r + c (< 256, exact in BF16).
    rows = kwargs.pop("rows", ROWS)
    dim = kwargs.pop("dim", DIM)
    values = torch.arange(rows)[:, None] * 16 + torch.arange(dim)[None, :]
    return values.to(torch.bfloat16)


def _shard_tensors(table: torch.Tensor, num_shards: int) -> list[torch.Tensor]:
    shard_rows = (table.shape[0] + num_shards - 1) // num_shards
    return [
        table[shard * shard_rows : (shard + 1) * shard_rows].clone()
        for shard in range(num_shards)
    ]


def _load_all(embedding, shards) -> None:
    for index, tensor in enumerate(shards):
        embedding.load_shard(index, tensor)
    embedding.finalize_load()


def _table_bytes(table: torch.Tensor) -> np.ndarray:
    return table.view(torch.uint8).numpy()


def test_reads_survive_mapping_scope_and_preserve_raw_bf16(tmp_path, monkeypatch):
    """Retained safetensors mappings stay valid, bit-identical, after gc."""
    embedding = _make_embedding(monkeypatch)
    names = [
        f"model.layers.0.ple.ngram_embedding.shard_{i}.weight" for i in range(SHARDS)
    ]
    save_file(
        dict(zip(names, _shard_tensors(_dense_table(), SHARDS))),
        tmp_path / "model.safetensors",
    )
    tensors = {}
    with safe_open(tmp_path / "model.safetensors", framework="pt") as handle:
        for name in names:
            tensors[name] = handle.get_tensor(name)
    for index, name in enumerate(names):
        embedding.load_shard(index, tensors[name])
    embedding.finalize_load()
    del tensors
    gc.collect()

    ids = np.array([2, 5, 5, 9, 0])
    destination = np.zeros((ids.shape[0], ROW_BYTES), np.uint8)
    embedding.gather_into(ids, destination)
    truth = _table_bytes(_dense_table())
    assert np.array_equal(destination, truth[ids])
    assert np.array_equal(destination[1], destination[2])
    assert embedding.weight.numel() == DIM


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dtype": torch.float32}, "BF16 and FP16"),
        ({"num_shards": 0}, "positive shard count"),
        ({"capacity": 0}, "positive staging"),
        ({"world_size": 2}, "ETP=TP=1"),
        ({"reduce_results": False}, "sequence parallelism"),
        ({"method": Qwen4ExpPLEFp8EmbeddingMethod()}, "unquantized"),
    ],
)
def test_constructor_rejects_unsupported_deployments(monkeypatch, overrides, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _make_embedding(monkeypatch, **overrides)


def test_direct_lookup_paths_are_rejected(monkeypatch):
    embedding = _make_embedding(monkeypatch)
    ids = torch.zeros(2, HEADS, dtype=torch.int64)
    weight = torch.zeros(ROWS, DIM, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="prepare"):
        embedding.forward(ids)
    with pytest.raises(RuntimeError, match="prepare"):
        embedding.start_prefetch(ids)
    with pytest.raises(RuntimeError, match="prepare"):
        embedding.finalize_prefetch(2)
    with pytest.raises(RuntimeError, match="load_shard"):
        embedding.weight_loader(embedding.weight, weight)


def test_shard_geometry_validation_and_missing_shards(monkeypatch):
    """Partial last shard loads; invalid geometry, dtype and gaps fail."""
    embedding = _make_embedding(monkeypatch)
    shards = _shard_tensors(_dense_table(), SHARDS)
    assert [tensor.shape[0] for tensor in shards] == [3, 3, 3, 1]
    destination = np.zeros((1, ROW_BYTES), np.uint8)
    with pytest.raises(RuntimeError, match="finalize_load"):
        embedding.gather_into(np.array([0]), destination)

    embedding.load_shard(3, shards[3])
    with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
        embedding.load_shard(4, shards[0])
    with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
        embedding.load_shard(-1, shards[0])
    with pytest.raises(ValueError, match="shape"):
        embedding.load_shard(0, shards[0][:2])
    with pytest.raises(ValueError, match="dtype"):
        embedding.load_shard(0, shards[0].float())
    strided = torch.zeros(3, 2 * DIM, dtype=torch.bfloat16)[:, :DIM]
    with pytest.raises(ValueError, match="row-contiguous"):
        embedding.load_shard(0, strided)
    with pytest.raises(RuntimeError, match="missing 3 of 4"):
        embedding.finalize_load()

    for index in (0, 1, 2):
        embedding.load_shard(index, shards[index])
    embedding.finalize_load()
    embedding.finalize_load()
    embedding.gather_into(np.array([4]), destination)
    assert np.array_equal(destination[0], _table_bytes(_dense_table())[4])


def test_gather_preserves_order_duplicates_and_masks(monkeypatch):
    """Rows follow id order including repetitions; invalid ids are zero."""
    embedding = _make_embedding(monkeypatch)
    table = _dense_table()
    _load_all(embedding, _shard_tensors(table, SHARDS))

    ids = np.array([9, 0, 3, 3, 10, -1, 2, 7, 8, 4], dtype=np.int64)
    destination = np.zeros((ids.shape[0], ROW_BYTES), np.uint8)
    embedding.gather_into(ids, destination)
    truth = _table_bytes(table)
    expected = np.zeros_like(destination)
    valid = (ids >= 0) & (ids < ROWS)
    expected[valid] = truth[ids[valid]]
    assert np.array_equal(destination, expected)

    rng = np.random.default_rng(0)
    fuzz = rng.integers(-2, ROWS + 2, size=512)
    fuzz_dest = np.zeros((fuzz.shape[0], ROW_BYTES), np.uint8)
    embedding.gather_into(fuzz, fuzz_dest)
    fuzz_expected = np.zeros_like(fuzz_dest)
    fuzz_valid = (fuzz >= 0) & (fuzz < ROWS)
    fuzz_expected[fuzz_valid] = truth[fuzz[fuzz_valid]]
    assert np.array_equal(fuzz_dest, fuzz_expected)


def test_shard_count_above_vocab_loads_empty_shards(monkeypatch):
    embedding = _make_embedding(monkeypatch, num_shards=13)
    table = _dense_table()
    shards = _shard_tensors(table, 13)
    assert sum(tensor.shape[0] for tensor in shards) == ROWS
    assert shards[12].shape[0] == 0
    _load_all(embedding, shards)
    destination = np.zeros((2, ROW_BYTES), np.uint8)
    embedding.gather_into(np.array([9, 10]), destination)
    assert np.array_equal(destination[0], _table_bytes(table)[9])
    assert not destination[1].any()


def test_shard_reload_rules(monkeypatch):
    """Repeated references are safe; a different mapping is rejected."""
    embedding = _make_embedding(monkeypatch)
    table = _dense_table()
    shards = _shard_tensors(table, SHARDS)
    embedding.load_shard(0, shards[0])
    embedding.load_shard(0, shards[0])
    retried = shards[0].clone()
    with pytest.raises(ValueError, match="already loaded"):
        embedding.load_shard(0, retried)
    _load_all(embedding, shards)
    destination = np.zeros((1, ROW_BYTES), np.uint8)
    embedding.gather_into(np.array([0]), destination)
    assert np.array_equal(destination[0], _table_bytes(table)[0])


def test_prepare_validates_batch_without_cuda(monkeypatch):
    embedding = _make_embedding(monkeypatch, capacity=3)
    with pytest.raises(ValueError, match="capacity"):
        embedding.prepare(torch.zeros(4, HEADS, dtype=torch.int64), 4)
    with pytest.raises(ValueError, match="padded to"):
        embedding.prepare(torch.zeros(3, HEADS, dtype=torch.int64), 2)
    with pytest.raises(ValueError, match="heads per token"):
        embedding.prepare(torch.zeros(2, HEADS + 1, dtype=torch.int64), 2)
    with pytest.raises(ValueError, match="whole number"):
        embedding.prepare(torch.zeros(3, dtype=torch.int64), 2)
    with pytest.raises(ValueError, match="integer row indices"):
        embedding.prepare(torch.zeros(2, HEADS), 2)
    with pytest.raises(ValueError, match="capacity"):
        embedding.view(4)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            embedding.zero_rows(2)


def _staged_truth(table, cpu_ids, padded) -> torch.Tensor:
    out = torch.zeros(padded, cpu_ids.shape[1] * DIM, dtype=table.dtype)
    valid = (cpu_ids >= 0) & (cpu_ids < table.shape[0])
    rows = table[cpu_ids.clamp(0, table.shape[0] - 1)]
    out[: cpu_ids.shape[0]] = (rows * valid.unsqueeze(-1)).flatten(1)
    return out


@requires_cuda
def test_prepare_stages_exact_rows_and_zero_pads(monkeypatch):
    """GPU and CPU ids stage bit-exact rows; padding stays zero."""
    embedding = _make_embedding(monkeypatch)
    table = _dense_table()
    _load_all(embedding, _shard_tensors(table, SHARDS))

    ids = torch.tensor([[9, 0], [2, 3], [-1, 10]], dtype=torch.int64, device="cuda")
    embedding.prepare(ids, 5)
    staged = embedding.view(5)
    torch.testing.assert_close(
        staged.cpu(), _staged_truth(table, ids.cpu(), 5), atol=0, rtol=0
    )
    persistent = staged.data_ptr()

    cpu_ids = torch.tensor([[4, 4], [1, 7]], dtype=torch.int64)
    embedding.prepare(cpu_ids, 5)
    assert embedding.view(5).data_ptr() == persistent
    torch.testing.assert_close(
        embedding.view(5).cpu(),
        _staged_truth(table, cpu_ids, 5),
        atol=0,
        rtol=0,
    )

    embedding.prepare(torch.empty(0, HEADS, dtype=torch.int64, device="cuda"), 3)
    assert not embedding.view(3).any()
    embedding.close()


@requires_cuda
def test_full_graph_replay_reflects_ids_under_sync_guard(monkeypatch):
    """Replayed graphs read the persistent rows a new prepare overwrote."""
    embedding = _make_embedding(monkeypatch)
    table = _dense_table()
    _load_all(embedding, _shard_tensors(table, SHARDS))
    ids = torch.tensor([[9, 0], [2, 3]], dtype=torch.int64, device="cuda")
    embedding.prepare(ids, 5)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            embedding.view(5).clone()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = embedding.view(5).clone()

    new_ids = torch.tensor([[1, 7], [4, 4]], dtype=torch.int64, device="cuda")
    sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        embedding.prepare(new_ids, 5)
        graph.replay()
        embedding.zero_rows(2)
    finally:
        torch.cuda.set_sync_debug_mode(sync_mode)
    torch.testing.assert_close(
        output.cpu(), _staged_truth(table, new_ids.cpu(), 5), atol=0, rtol=0
    )
    embedding.close()
