# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_EXL3_EXPERTS_CONTIGUOUS: a MoE layer's local expert matrices as views
into one buffer per tensor kind (quant.expert_buffer_views), on CPU.

With the switch off every trellis, suh and svh is its own allocation; with it
on, every local expert matrix's trellis of the layer is a view of one int16
buffer, its suh of one FP16 buffer and its svh of another, each view
contiguous, of the same shape and dtype, at a 512-byte-aligned offset and
not overlapping. Loading through the EXL3 loader
(Exl3PackedWeight.load) gives every expert the bytes of the checkpoint in
both modes:

- a synthetic checkpoint with random payloads (always);
- the checkpoint named by VLLM_EXL3_TEST_CHECKPOINT (any EXL3 MoE model:
  DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next), layers VLLM_EXL3_TEST_EXPERT_LAYERS
  (default 10,20), the experts expert-parallel rank 0 of
  VLLM_EXL3_TEST_EP_SIZE (default 4) holds.
"""

import json
import os
import re
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from tests.model_executor.layers.test_exl3_layers import _write_sparse
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm_sm80.kernels.exl3 import quant
from vllm_sm80.kernels.exl3.utils import Exl3LoadLedger

ALIGN = quant.EXPERT_VIEW_ALIGN_BYTES
EXPERTS, HIDDEN, INTERMEDIATE = 8, 384, 256
KINDS = ("trellis", "suh", "svh")


class _Layer(torch.nn.Module):
    """The RoutedExperts members create_weights reads."""

    def __init__(self, global_experts, expert_map):
        super().__init__()
        self.global_num_experts = global_experts
        self.apply_router_weight_on_input = False
        self.expert_map = expert_map

    def _map_global_expert_id_to_local_expert_id(self, expert):
        # No map (one EP rank): every expert is local under its own id.
        return expert if self.expert_map is None else int(self.expert_map[expert])


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    return value


def _config(directory: Path, hf_config=None):
    config = quant.Exl3Config()
    config.maybe_update_config(str(directory), hf_config)
    return config


def _build_and_load(monkeypatch, config, experts_prefix, ep_size, contiguous):
    """create_weights for EP rank 0 of ep_size, then the loader's per-weight
    load; returns the layer."""
    monkeypatch.setenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", "1" if contiguous else "0")
    global_experts = len(
        {
            name.split(".")[-2]
            for name in config.manifest.matrices
            if name.startswith(f"{experts_prefix}.")
            and name.count(".") == experts_prefix.count(".") + 2
        }
    )
    local, expert_map, _ = determine_expert_map(ep_size, 0, global_experts)
    method = object.__new__(quant.Exl3MoEMethod)
    method.config = config
    method.prefix = experts_prefix
    layer = _Layer(global_experts, expert_map)
    # Hidden and intermediate sizes from the first expert's gate projection
    # (multiples of 128 in the checkpoints these tests read).
    gate = config.manifest.matrices[
        config.layout.expert_matrix(config, experts_prefix, 0, "gate_proj")
    ]
    method.create_weights(layer, local, gate.padded_in, gate.padded_out, torch.bfloat16)
    ledger = Exl3LoadLedger(config.manifest)
    manifest = config.manifest
    with ExitStack() as stack:
        files = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in sorted({h.path for h in manifest.tensors.values()})
        }
        for expert in layer.exl3_experts:
            for weight in expert:
                weight.load(files, ledger)
    return layer


def _tensors(layer, kind):
    return [getattr(w, kind) for expert in layer.exl3_experts for w in expert]


def _check_layout(layer, contiguous):
    for kind in KINDS:
        tensors = _tensors(layer, kind)
        storages = {t.untyped_storage().data_ptr() for t in tensors}
        assert all(t.is_contiguous() for t in tensors), kind
        if not contiguous:
            assert len(storages) == len(tensors), kind
            continue
        assert len(storages) == 1, f"{kind}: {len(storages)} storages"
        size = tensors[0].element_size()
        starts = sorted(t.storage_offset() * size for t in tensors)
        assert all(start % ALIGN == 0 for start in starts), kind
        spans = sorted(
            (t.storage_offset() * size, (t.storage_offset() + t.numel()) * size)
            for t in tensors
        )
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), f"{kind} overlaps"
        buffer = tensors[0].untyped_storage().nbytes()
        padded = sum(-(-(e - s) // ALIGN) * ALIGN for s, e in spans)
        assert buffer == padded, f"{kind}: buffer {buffer} B for {padded} B of views"


def _file_array(manifest, name):
    header = manifest.tensors[name]
    dtype = {"I16": np.int16, "F16": np.float16}[header.dtype]
    return np.memmap(
        header.path, dtype=dtype, mode="r", offset=header.offset, shape=header.shape
    )


def _check_bytes(layer, manifest):
    """Every loaded tensor equals the checkpoint's (whole matrices)."""
    for expert in layer.exl3_experts:
        for weight in expert:
            for kind in KINDS:
                expected = torch.from_numpy(
                    np.array(_file_array(manifest, f"{weight.matrix.prefix}.{kind}"))
                )
                actual = getattr(weight, kind).detach()
                assert torch.equal(
                    actual.view(torch.int16), expected.view(torch.int16)
                ), f"{weight.matrix.prefix}.{kind}"


# ---------------------------------------------------------------- synthetic


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A checkpoint of one MoE layer with random payloads; the down
    projection's suh is 256 halves (512 B) and svh 384 (768 B, not a
    multiple of the alignment)."""
    directory = tmp_path_factory.mktemp("experts")
    rng = np.random.default_rng(1019)
    tensors = {}
    for expert in range(EXPERTS):
        for role, k, n, bits_x2 in (
            ("gate_proj", HIDDEN, INTERMEDIATE, 6),
            ("up_proj", HIDDEN, INTERMEDIATE, 6),
            ("down_proj", INTERMEDIATE, HIDDEN, 4),
        ):
            prefix = f"model.layers.0.mlp.experts.{expert}.{role}"
            trellis = rng.integers(-(2**15), 2**15, (k // 16, n // 16, 8 * bits_x2))
            tensors[f"{prefix}.trellis"] = (
                "I16",
                list(trellis.shape),
                trellis.astype(np.int16).tobytes(),
            )
            for kind, size in (("suh", k), ("svh", n)):
                values = rng.uniform(-1.5, 1.5, size).astype(np.float16)
                tensors[f"{prefix}.{kind}"] = ("F16", [size], values.tobytes())
            tensors[f"{prefix}.mul1"] = ("I32", [], np.uint32(0x83DCD12D).tobytes())
    _write_sparse(directory / "model.safetensors", tensors)
    return directory


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_synthetic_layer_loads_the_same_bytes(
    monkeypatch, synthetic, contiguous, ep_size
):
    config = _config(synthetic)
    layer = _build_and_load(
        monkeypatch, config, "model.layers.0.mlp.experts", ep_size, contiguous
    )
    assert len(layer.exl3_experts) == EXPERTS // ep_size
    _check_layout(layer, contiguous)
    _check_bytes(layer, config.manifest)


def test_mismatched_views_are_refused(synthetic):
    config = _config(synthetic)
    matrix = config.manifest.matrices["model.layers.0.mlp.experts.0.gate_proj"]
    trellis, suh, svh = quant.expert_buffer_views([matrix])[0]
    whole = (slice(0, matrix.padded_in), [slice(0, matrix.padded_out)])
    quant.Exl3PackedWeight(
        matrix, HIDDEN, INTERMEDIATE, *whole, views=(trellis, suh, svh)
    )
    for views in (
        (trellis, svh, suh),
        (trellis.float(), suh, svh),
        (trellis.transpose(0, 1), suh, svh),
    ):
        with pytest.raises(ValueError, match="views do not match"):
            quant.Exl3PackedWeight(matrix, HIDDEN, INTERMEDIATE, *whole, views=views)


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", raising=False)
    assert envs.VLLM_EXL3_EXPERTS_CONTIGUOUS is False
    monkeypatch.setenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", "1")
    assert envs.VLLM_EXL3_EXPERTS_CONTIGUOUS is True


# ---------------------------------------------------------------- real checkpoint


def _real_checkpoint():
    value = os.environ.get("VLLM_EXL3_TEST_CHECKPOINT")
    if not value or not Path(value).is_dir():
        pytest.skip("set VLLM_EXL3_TEST_CHECKPOINT to an EXL3 MoE checkpoint")
    return Path(value)


def _experts_prefix(config, layer_index):
    """The target model's routed-experts prefix of a layer (not the MTP's)."""
    found = set()
    for name in config.manifest.matrices:
        match = re.match(r"(.*\.experts)\.\d+\.[^.]+$", name)
        if (
            match
            and not name.startswith("mtp")
            and re.search(rf"(^|\.)layers\.{layer_index}\.", name)
        ):
            found.add(match[1])
    if len(found) != 1:
        pytest.skip(f"layer {layer_index}: routed-experts prefixes {sorted(found)}")
    return found.pop()


REAL_LAYERS = [
    int(v) for v in os.environ.get("VLLM_EXL3_TEST_EXPERT_LAYERS", "10,20").split(",")
]


@pytest.fixture(scope="module")
def real_config():
    directory = _real_checkpoint()
    hf_config = _namespace(json.loads((directory / "config.json").read_text()))
    return _config(directory, hf_config)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("layer_index", REAL_LAYERS)
def test_real_layer_loads_the_checkpoint_bytes(
    monkeypatch, real_config, layer_index, contiguous
):
    ep_size = int(os.environ.get("VLLM_EXL3_TEST_EP_SIZE", "4"))
    prefix = _experts_prefix(real_config, layer_index)
    layer = _build_and_load(monkeypatch, real_config, prefix, ep_size, contiguous)
    _check_layout(layer, contiguous)
    _check_bytes(layer, real_config.manifest)
    tensors = [t for kind in KINDS for t in _tensors(layer, kind)]
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors
    }
    print(
        f"{prefix} EP rank 0 of {ep_size}: {len(layer.exl3_experts)} experts, "
        f"{len(tensors)} tensors, {nbytes} B, {len(storages)} allocations of "
        f"{sum(storages.values())} B (contiguous={contiguous})"
    )
