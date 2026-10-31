# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_EXL3_FAST_LOAD: the EXL3 loader's page-cache readahead
(readahead.CheckpointReadahead over Exl3PackedWeight.source_extents) and its
plain-tensor loads without the routed experts in the module tree
(loader.routed_experts_hidden), on CPU.

Neither changes a copy. Checked here, on the synthetic tensor-split
checkpoints of test_exl3_expert_split (every rank) and with whole experts:

- source_extents covers every byte load() reads: loading from a copy of the
  checkpoint whose bytes outside the rank's extents are overwritten gives the
  same tensors;
- with the readahead the loaded tensors are the bytes loaded without it, and
  its posix_fadvise calls cover the extents in pieces the kernel reads whole;
- it stays within its window of the loader and stops when the loader leaves;
- routed_experts_hidden takes exactly the routed experts' packed parameters
  out of named_parameters() and puts back the same modules, also on an error.
"""

import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from tests.kernels.quantization.test_exl3_expert_split import (  # noqa: F401
    EXPERTS,
    LAYERS,
    _Layer,
    _moe_config,
    synthetic,
)
from vllm_sm80.kernels.exl3 import quant, readahead
from vllm_sm80.kernels.exl3.utils import Exl3LoadLedger

KINDS = ("trellis", "suh", "svh")


def _method_and_layer(config, prefix, intermediate, size, rank, whole):
    """The rank's Exl3MoEMethod and layer after create_weights, unloaded: the
    tensor split, or whole experts (one rank holding all) with ``whole``."""
    moe = _moe_config(intermediate, 1 if whole else size, 0 if whole else rank)
    method = quant.Exl3MoEMethod(config, prefix, moe)
    hidden, width = method.maybe_roundup_sizes(
        moe.hidden_dim,
        moe.intermediate_size_per_partition,
        moe.in_dtype,
        moe.moe_parallel_config,
    )
    layer = _Layer(EXPERTS)
    method.create_weights(layer, EXPERTS, hidden, width, torch.bfloat16)
    return method, layer


def _create(config, prefix, intermediate, size, rank, whole):
    """The rank's Exl3PackedWeight of every expert, in load order."""
    _, layer = _method_and_layer(config, prefix, intermediate, size, rank, whole)
    return [weight for expert in layer.exl3_experts for weight in expert]


def _load(manifest, weights, prefetch, window=readahead.WINDOW_BYTES):
    """The loader's per-weight load (load_exl3_weights), with the readahead
    when ``prefetch``; returns the loaded tensors."""
    ledger = Exl3LoadLedger(manifest)
    with ExitStack() as stack:
        files = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in sorted({h.path for h in manifest.tensors.values()})
        }
        ahead = None
        if prefetch:
            ahead = stack.enter_context(
                readahead.CheckpointReadahead(
                    [w.source_extents(manifest) for w in weights], window
                )
            )
        for weight in weights:
            weight.load(files, ledger)
            if ahead is not None:
                ahead.loaded()
    return [getattr(w, kind).detach().clone() for w in weights for kind in KINDS]


def _layers(synthetic):
    config, intermediate, size = synthetic
    for layer_index in range(LAYERS):
        prefix = f"model.layers.{layer_index}.mlp.experts"
        for whole in (False, True):
            for rank in range(1 if whole else size):
                yield config, prefix, intermediate, size, rank, whole


def test_extents_cover_every_byte_load_reads(synthetic, tmp_path):
    config, _, _ = synthetic
    manifest = config.manifest
    (path,) = {h.path for h in manifest.tensors.values()}
    data_start = min(h.data_start for h in manifest.tensors.values())
    for config, prefix, intermediate, size, rank, whole in _layers(synthetic):
        weights = _create(config, prefix, intermediate, size, rank, whole)
        reference = _load(manifest, weights, prefetch=False)
        # The checkpoint with every payload byte outside the extents changed
        # (but the codebook markers, which the manifest reads).
        payload = np.fromfile(path, dtype=np.uint8)
        keep = np.zeros(payload.size, dtype=bool)
        keep[:data_start] = True
        for header in manifest.tensors.values():
            if header.name.endswith((".mul1", ".mcg")):
                keep[header.offset : header.offset + header.nbytes] = True
        for weight in weights:
            for _, start, end in weight.source_extents(manifest):
                assert data_start <= start < end <= payload.size
                keep[start:end] = True
        assert not keep.all()
        payload[~keep] ^= 0x5A
        masked_dir = tmp_path / f"{prefix}-{rank}-{whole}"
        masked_dir.mkdir()
        payload.tofile(masked_dir / path.name)
        masked = quant.Exl3Config()
        masked.maybe_update_config(str(masked_dir))
        weights = _create(masked, prefix, intermediate, size, rank, whole)
        loaded = _load(masked.manifest, weights, prefetch=False)
        assert all(torch.equal(a, b) for a, b in zip(reference, loaded, strict=True))


def test_readahead_loads_the_same_bytes_and_covers_the_extents(
    synthetic, monkeypatch
):
    calls = []
    advise = os.posix_fadvise

    def record(fd, offset, length, advice):
        calls.append((os.readlink(f"/proc/self/fd/{fd}"), offset, length, advice))
        advise(fd, offset, length, advice)

    monkeypatch.setattr(readahead, "readahead_piece", lambda path: 4096)
    monkeypatch.setattr(readahead.os, "posix_fadvise", record)
    for config, prefix, intermediate, size, rank, whole in _layers(synthetic):
        manifest = config.manifest
        weights = _create(config, prefix, intermediate, size, rank, whole)
        without = _load(manifest, weights, prefetch=False)
        weights = _create(config, prefix, intermediate, size, rank, whole)
        calls.clear()
        with_readahead = _load(manifest, weights, prefetch=True)
        assert all(
            torch.equal(a, b) for a, b in zip(without, with_readahead, strict=True)
        )
        assert all(
            length <= 4096 and advice == os.POSIX_FADV_WILLNEED
            for _, _, length, advice in calls
        )
    # The loader may overtake the thread (which then skips the loads done), so
    # coverage is checked on a thread that runs alone.
    config, prefix, intermediate, size, rank, whole = next(_layers(synthetic))
    manifest = config.manifest
    weights = _create(config, prefix, intermediate, size, rank, whole)
    calls.clear()
    plan = [w.source_extents(manifest) for w in weights]
    ahead = readahead.CheckpointReadahead(plan, window=1 << 40)
    with ahead:
        ahead.thread.join(timeout=30)
    covered = {}
    for name, offset, length, _ in calls:
        covered.setdefault(name, set()).update(range(offset, offset + length))
    for path, start, end in (e for spans in plan for e in spans):
        assert set(range(start, end)) <= covered[str(Path(path).resolve())]


def test_readahead_stays_within_its_window_and_stops_with_the_loader(
    tmp_path, monkeypatch
):
    path = tmp_path / "blob"
    path.write_bytes(bytes(1 << 20))
    issued = []
    monkeypatch.setattr(readahead, "readahead_piece", lambda path: 1024)
    monkeypatch.setattr(
        readahead.os,
        "posix_fadvise",
        lambda fd, offset, length, advice: issued.append(offset + length),
    )
    # Ten loads of 10 KiB each, a window of 25 KiB.
    plan = [[(path, i * 10240, (i + 1) * 10240)] for i in range(10)]
    ahead = readahead.CheckpointReadahead(plan, window=25 * 1024)
    with ahead:
        time.sleep(0.2)
        # Loads 0, 1 and 2 start within the window of the loader at load 0.
        assert max(issued) == 3 * 10240
        ahead.loaded()
        time.sleep(0.2)
        assert max(issued) == 4 * 10240
    assert not ahead.thread.is_alive()
    # A loader that runs ahead of the thread: the thread skips the loads done.
    issued.clear()
    ahead = readahead.CheckpointReadahead(plan, window=1 << 30)
    ahead.done = 7
    with ahead:
        ahead.thread.join(timeout=10)
    assert min(issued) > 7 * 10240


def test_readahead_piece_reads_the_device_limits(tmp_path):
    piece = readahead.readahead_piece(tmp_path)
    assert piece >= 4096 and piece % 4096 == 0


def test_coalesce_merges_close_spans_of_one_file():
    a, b = Path("/a"), Path("/b")
    spans = [(a, 100, 200), (b, 0, 10), (a, 0, 50), (a, 210, 300), (a, 500, 600)]
    assert readahead.coalesce(spans, 16) == [
        (a, 0, 50),
        (a, 100, 300),
        (a, 500, 600),
        (b, 0, 10),
    ]


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_FAST_LOAD", raising=False)
    assert envs.VLLM_EXL3_FAST_LOAD is False
    monkeypatch.setenv("VLLM_EXL3_FAST_LOAD", "1")
    assert envs.VLLM_EXL3_FAST_LOAD is True


@pytest.mark.parametrize("window", [1, 1 << 40])
def test_any_window_loads_the_same_bytes(synthetic, window):
    config, prefix, intermediate, size, rank, whole = next(_layers(synthetic))
    weights = _create(config, prefix, intermediate, size, rank, whole)
    without = _load(config.manifest, weights, prefetch=False)
    weights = _create(config, prefix, intermediate, size, rank, whole)
    loaded = _load(config.manifest, weights, prefetch=True, window=window)
    assert all(torch.equal(a, b) for a, b in zip(without, loaded, strict=True))


def test_routed_experts_hidden_takes_out_only_the_packed_experts(synthetic):
    from vllm_sm80.kernels.exl3 import loader

    config, intermediate, size = synthetic
    model = torch.nn.Module()
    model.norm = torch.nn.Linear(4, 4)
    model.layers = torch.nn.ModuleList()
    for layer_index, whole in ((0, False), (1, True)):
        method, layer = _method_and_layer(
            config,
            f"model.layers.{layer_index}.mlp.experts",
            intermediate,
            size,
            0,
            whole,
        )
        layer.quant_method = method
        layer.gate = torch.nn.Linear(4, EXPERTS)
        model.layers.append(layer)
    before = list(model.named_parameters())
    experts = [layer.exl3_experts for layer in model.layers]
    kept = [name for name, _ in before if ".exl3_experts." not in name]
    assert len(kept) < len(before)
    modules = dict(model.named_modules())
    with loader.routed_experts_hidden(modules):
        assert [name for name, _ in model.named_parameters()] == kept
    with pytest.raises(RuntimeError), loader.routed_experts_hidden(modules):
        raise RuntimeError
    after = list(model.named_parameters())
    assert [n for n, _ in after] == [n for n, _ in before]
    assert all(a is b for (_, a), (_, b) in zip(after, before, strict=True))
    assert all(
        layer.exl3_experts is held
        for layer, held in zip(model.layers, experts, strict=True)
    )
