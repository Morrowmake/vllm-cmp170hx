# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid snapshots must follow consumed rows, not paged-table capacity."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_sm80.models.deepseek_v41 import hybrid_capture as hybrid
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.capture import Record


class NoCacheSnapshot(TorchDispatchMode):
    """The capture-time path may copy metadata, never a reader's KV payload."""

    def __init__(self, *caches):
        super().__init__()
        self.pointers = {x.untyped_storage().data_ptr() for x in caches if x is not None}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.index.Tensor, torch.ops.aten.clone.default):
            assert args[0].untyped_storage().data_ptr() not in self.pointers
        return func(*args, **(kwargs or {}))


def index_args(device="cpu"):
    table = torch.full((2, 65536), 999999, dtype=torch.int32, device=device)
    table[:, :2] = torch.tensor([[2, 4], [2, 5]], device=device)
    return dict(
        q=torch.arange(32, device=device).reshape(4, 2, 4).to(torch.bfloat16) / 8,
        w=torch.ones(4, 2, device=device),
        n_vis=torch.tensor([5, 3, 7, 0], device=device),
        req=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device),
        idxk=torch.arange(96, device=device).reshape(6, 4, 4).to(torch.bfloat16) / 32,
        block_table=table, n_cols=8, heads_per_rank=2,
    )


def attention_args(kind="compressed", device="cpu"):
    table = torch.full((2, 65536), 999999, dtype=torch.int32, device=device)
    table[:, :2] = torch.tensor([[2, 4], [2, 5]], device=device)
    cache = torch.arange(96, device=device).reshape(6, 4, 4).to(torch.bfloat16) / 32
    local = torch.tensor([[0, 5, -1, 0], [1, 6, 4, -1]], dtype=torch.int32, device=device)
    if kind in ("absent", "none_selection"):
        local = None
    elif kind == "empty_selection":
        local = local[:, :0]
    elif kind == "invalid_selection":
        local.fill_(-1)
    return dict(
        q=torch.arange(16, device=device).reshape(2, 2, 4).to(torch.bfloat16) / 8,
        swa_cache=cache,
        swa_indices=torch.tensor([[8, 9, -1, 8], [8, 21, 9, -1]],
                                 dtype=torch.int32, device=device),
        comp_cache=None if kind == "absent" else cache.clone() + 1,
        comp_block_table=None if kind == "absent" else table,
        comp_local=local,
        req=torch.tensor([0, 1], dtype=torch.int32, device=device),
        sink=torch.zeros(2, device=device), scale=0.5,
    )


def reader_record(op, args):
    record = Record("hybrid", "hybrid_linear_consumer", "layers.8.attn", args["q"].shape[0], {})
    caches = [args[name] for name in ("idxk", "swa_cache", "comp_cache") if name in args]
    with NoCacheSnapshot(*caches):
        hybrid._defer_reader(record, op, args)
    return record


def test_hybrid_index_snapshot_uses_visible_unique_blocks_and_owns_metadata():
    args = index_args()
    expected = core.index_scores_torch(**args)
    record = reader_record("index_scores", args)
    assert "idxk" not in record.tensors and "block_table" not in record.tensors
    # These caller-owned buffers can be reused after the point was recorded.
    args["block_table"].fill_(999999)
    args["req"].zero_()
    args["n_vis"].zero_()
    args["q"].zero_()
    saved = record.materialize()
    assert saved["idxk"].shape == (4, 4, 4)  # null + physical blocks 2,4,5
    assert saved["block_table"][0, 0] == saved["block_table"][1, 0]
    assert bool((saved["block_table"][:, 2:] == 0).all())
    assert torch.equal(core.index_scores_torch(**saved), expected)


@pytest.mark.parametrize("visible", [0, 1, 4, 5, 8])
def test_hybrid_index_snapshot_visibility_boundaries(visible):
    args = index_args()
    args["n_vis"].fill_(visible)
    expected = core.index_scores_torch(**args)
    saved = reader_record("index_scores", args).materialize()
    want_blocks = 1 if visible == 0 else (2 if visible <= 4 else 4)
    assert saved["idxk"].shape[0] == want_blocks
    assert torch.equal(core.index_scores_torch(**saved), expected)


@pytest.mark.parametrize("kind", ["compressed", "absent", "none_selection", "empty_selection", "invalid_selection"])
def test_hybrid_sparse_snapshot_preserves_consumed_rows_and_negative_slots(kind):
    args = attention_args(kind)
    expected = core.sparse_attention(**args)
    record = reader_record("sparse_attention", args)
    assert not {"swa_cache", "comp_cache", "swa_indices", "comp_block_table"} & record.tensors.keys()
    args["swa_indices"].fill_(-1)
    args["req"].zero_()
    if args["comp_block_table"] is not None:
        args["comp_block_table"].fill_(999999)
    if args["comp_local"] is not None:
        args["comp_local"].fill_(-1)
    saved = record.materialize()
    assert saved["swa_cache"].shape == (3, 4, 4)  # null + physical blocks 2,5
    assert saved["swa_indices"][0, 0] == saved["swa_indices"][0, 3] == saved["swa_indices"][1, 0]
    assert saved["swa_indices"][0, 2] == saved["swa_indices"][1, 3] == -1
    if kind == "absent":
        assert saved["comp_cache"] is None and saved["comp_block_table"] is None
    else:
        assert saved["comp_cache"].shape[0] == (4 if kind == "compressed" else 1)
        assert bool((saved["comp_block_table"][:, 2:] == 0).all())
    assert torch.equal(core.sparse_attention(**saved), expected)


def intercept_records(monkeypatch):
    records = []
    monkeypatch.setattr(hybrid, "_outputs", {})

    def begin(family, point, prefix, rows, point_phase=None, **meta):
        return Record(family, point, prefix, rows, meta, point_phase=point_phase)

    monkeypatch.setattr(hybrid.capture, "begin", begin)
    monkeypatch.setattr(Record, "done", lambda record: records.append(record))
    return records


@pytest.mark.parametrize("op", ["store_rows", "compress_rows"])
def test_hybrid_mutating_snapshot_still_keeps_pre_post_and_guard_blocks(monkeypatch, op):
    records = intercept_records(monkeypatch)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    produced = hybrid._record("linear", {"x": x}, lambda: x + 1,
                              linear={"prefix": "layers.8.attn.wkv"})
    cache = torch.arange(192, dtype=torch.float32).reshape(6, 4, 8)
    before = cache.clone()
    slots = torch.tensor([8, 9, -1])
    name = "ring" if op == "compress_rows" else "cache"
    slot_name = "ring_slots" if op == "compress_rows" else "slots"
    input_name = "kv_score" if op == "compress_rows" else "rows"

    def run():
        valid = slots >= 0
        cache[slots[valid] // 4, slots[valid] % 4] = produced[valid]
        return produced.clone() if op == "compress_rows" else None

    hybrid._record(op, {name: cache, slot_name: slots, input_name: produced}, run)
    after = cache.clone()
    cache.add_(10000)
    record = records[-1]
    saved = record.materialize()
    ids = saved["snapshot_blocks"]
    assert set(ids.tolist()) == {0, 1, 2, 3}
    assert saved[name].shape[0] == 4
    assert torch.equal(saved[name], before[ids])
    assert torch.equal(saved["out.ring" if op == "compress_rows" else "out"], after[ids])
    assert record.meta["parents"][input_name]["call_id"] == records[0].meta["call_id"]


@pytest.mark.parametrize("op", ["index_scores", "sparse_attention"])
def test_hybrid_reader_record_preserves_real_parent_and_successor_links(monkeypatch, op):
    records = intercept_records(monkeypatch)
    args = index_args() if op == "index_scores" else attention_args()
    q = args["q"]
    args["q"] = hybrid._record("linear", {"x": q}, lambda: q.clone(),
                               linear={"prefix": "layers.8.attn.wq_b"})
    run = core.index_scores if op == "index_scores" else core.sparse_attention
    result = hybrid._record(op, args, lambda: run(**args))
    link = hybrid._link(result)
    assert link["call_id"] == records[-1].meta["call_id"]
    assert link["output"] == "out"
    assert records[-1].meta["parents"]["q"]["call_id"] == records[0].meta["call_id"]
    saved = records[-1].materialize()
    observed = saved.pop("out")
    assert torch.equal(run(**saved), observed)
    assert torch.equal(observed, result)


def test_hybrid_unselected_call_does_not_snapshot_or_defer(monkeypatch):
    monkeypatch.setattr(hybrid, "_outputs", {})
    args = index_args()
    expected = core.index_scores_torch(**args)
    with NoCacheSnapshot(args["idxk"]):
        # No producer means no capture. The dummy run avoids confusing the
        # actual reader's legitimate cache reads with diagnostic snapshots.
        result = hybrid._record("index_scores", args, lambda: expected)
    assert result is expected


@pytest.mark.parametrize("n_cols,origin,offset_view", [(8, 0, False), (8, 3, True), (131071, 0, False)])
def test_hybrid_padded_scores_export_and_select(monkeypatch, tmp_path, n_cols, origin, offset_view):
    """CPU layout/link fixture, not serving coverage or numerical qualification.

    Use the real writer, exporter and selection consumer. Deliberately retain
    the exporter's complete-matrix refusal instead of inventing unrelated
    projection roles/phases to make this small fixture look complete.
    """
    import importlib.util
    import json
    from pathlib import Path
    import re

    from vllm_sm80 import capture as cap

    root = tmp_path / "capture"
    for rank in range(4):
        cfg = cap._Config(str(root / f"r{rank}"), frozenset({"hybrid"}),
                          re.compile(".*"), re.compile(".*"), frozenset({2, 4}),
                          2, frozenset({1}), 65, 1 << 26, rank, rank, 4, prefill_per_key=0)
        monkeypatch.setattr(cap, "_cfg", cfg)
        monkeypatch.setattr(cap, "_capturing", lambda: False)
        monkeypatch.setattr(cap.torch.cuda, "is_available", lambda: False)
        for name, value in (("_counts", {}), ("_prefill_calls", {}), ("_bytes", [0]), ("_seen", {})):
            monkeypatch.setattr(cap, name, value)
        monkeypatch.setattr(cap, "_step", dict(real=True, step=1, phase="verify",
                                              query_lens=[2, 2], context_lens=[16, 16], context="short"))
        monkeypatch.setattr(hybrid, "_outputs", {})
        args = index_args()
        args["n_cols"] = n_cols
        q = args["q"]
        # Keep this producer view live: this test isolates padded geometry
        # from the separately covered alias-lifetime defect.
        query_output = hybrid._record("linear", {"x": q.reshape(4, -1)},
            lambda: q.clone().reshape(4, -1),
            linear={"prefix": "layers.8.attn.indexer.wq_b",
                    "sources": [("layers.8.attn.indexer.wq_b.weight", [8, 8])]})
        args["q"] = query_output.view_as(q)
        logical = core.index_scores_torch(**args)
        padded_cols = ((n_cols + origin + 15) // 16) * 16
        backing = torch.full((4, padded_cols), 12345.0)
        padded = backing[:, origin:origin + n_cols]
        padded.copy_(logical)
        scores = hybrid._record("index_scores", args, lambda: padded)
        selected = scores[1:3, 2:6] if offset_view else scores
        n_vis = ((args["n_vis"][1:3] - 2).clamp(0, 4) if offset_view else args["n_vis"])
        inputs = {"scores": selected, "n_vis": n_vis, "k": 2}
        hybrid._record("select_topk", inputs, lambda: core.select_topk(**inputs))

    path = Path(hybrid.__file__).resolve().parents[4] / "tools/sm80/export_hybrid_cases.py"
    spec = importlib.util.spec_from_file_location("hybrid_link_export", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    out = tmp_path / "export"
    # Every real source link is checked/frozen before the unchanged coverage
    # gate rejects this intentionally partial fixture. Old padded clones fail
    # earlier here with out-of-bounds as_strided storage, not this refusal.
    with pytest.raises(ValueError, match="missing hybrid captures: rank 0 roles"):
        exporter.export(root, tmp_path / "unused-checkpoint", out)
    assert not (out / "hybrid-linear.json").exists()
    assert not (out / "hybrid-consumer.json").exists()

    from tests.models.deepseek_v41.sm80.hybrid_consumer_cases import (
        _producer_view, actual, ideal, load_source,
    )

    consumers = []
    for path in (out / "sources").glob("*.pt"):
        obj = torch.load(path, weights_only=True, mmap=True)
        if obj["meta"]["op"] == "select_topk":
            consumers.append(obj)
    assert len(consumers) == 4
    for obj in consumers:
        source = obj["meta"]["parent_sources"]["scores"]
        parent = load_source(source["path"], source["sha256"])
        stored, view = parent["tensors"]["out"], source["view"]
        assert stored.stride() == (padded_cols, 1)
        assert stored.storage_offset() == 0
        assert stored.untyped_storage().nbytes() == ((4 - 1) * padded_cols + n_cols) * stored.element_size()
        assert view["offset"] == (padded_cols + 2 if offset_view else 0)
        linked = stored.as_strided(view["shape"], view["stride"], view["offset"])
        assert exporter.same(linked, obj["tensors"]["scores"])
        args = {k: v for k, v in parent["tensors"].items() if not k.startswith("out")}
        consumer_args = {k: v for k, v in obj["tensors"].items() if not k.startswith("out")}
        assert torch.equal(actual("select_topk", {**consumer_args, "scores": linked}), obj["tensors"]["out"])
        # The actual CPU producer and FP64 reference are packed; their logical
        # values must acquire the saved strides before applying the same link.
        for replayed in (actual("index_scores", args), ideal("index_scores", args)):
            relinked = _producer_view(replayed, stored, view)
            assert relinked.shape == linked.shape
            selected = actual("select_topk", {**consumer_args, "scores": relinked})
            assert torch.equal(selected, obj["tensors"]["out"])
    # The native payloads, rather than a hand-written geometry echo, supplied
    # every exporter input and frozen consumer's parent source.
    rows = [json.loads(line) for line in (root / "r0/index.jsonl").read_text().splitlines()]
    assert {row["point"] for row in rows} == {"hybrid_linear", "hybrid_linear_consumer"}
    assert all(row["saved"] for row in rows)


def test_hybrid_live_alias_preserves_producer_link(monkeypatch):
    """Match a reshape result immediately viewed again by its successor."""
    import gc
    import weakref

    records = intercept_records(monkeypatch)
    base = torch.ones(4, 2, 8)
    producer = hybrid._record("linear", {"x": base}, lambda: base.reshape(4, 16),
                              linear={"prefix": "layers.8.attn.indexer.wq_b"})
    query = producer.view(4, 2, 8)
    before = hybrid._link(query)
    producer_ref = weakref.ref(producer)
    assert before is not None and query._base is base
    del producer
    gc.collect()
    assert producer_ref() is None
    assert query.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    after = hybrid._link(query)
    assert after is not None and after["call_id"] == records[-1].meta["call_id"]
    assert after["shape"] == list(query.shape) and after["stride"] == list(query.stride())


def test_hybrid_storage_identity_rejects_stale_entries(monkeypatch):
    import gc
    import weakref

    intercept_records(monkeypatch)
    value = hybrid._record("linear", {"x": torch.ones(2, 8)}, lambda: torch.ones(2, 8),
                           linear={"prefix": "layers.8.attn.wq_b"})
    key = value.untyped_storage().data_ptr()
    entry = hybrid._outputs[key]
    other = torch.zeros_like(value)
    # A pointer lookup alone cannot authorize another storage identity, even
    # while the original storage is still live.
    hybrid._outputs[other.untyped_storage().data_ptr()] = entry
    assert hybrid._link(other) is None
    reference = weakref.ref(value)
    del value
    gc.collect()
    assert reference() is None and entry[1].expired()
    assert hybrid._link(other) is None


@pytest.mark.parametrize("preserve", [False, True])
def test_pageable_snapshot_exports_only_required_storage(preserve):
    owned = torch.arange(74, dtype=torch.float32).as_strided((8, 4), (10, 1))
    record = Record("hybrid", "snapshot", "layer", 8, {}, tensors={"x": owned})
    if preserve:
        record.layout_fields.add("x")
    exported = record.materialize()["x"]
    assert torch.equal(exported, owned)
    expected_elements = 74 if preserve else 32
    assert exported.untyped_storage().nbytes() == expected_elements * owned.element_size()
    assert exported.stride() == ((10, 1) if preserve else (4, 1))
