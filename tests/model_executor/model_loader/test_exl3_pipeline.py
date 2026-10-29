# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP checkpoint ownership must not hide genuinely unassigned EXL3 weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.sm80.kernels.exl3 import loader, ops
from vllm.sm80.kernels.exl3.quant import Exl3Config
from vllm.sm80.kernels.exl3.utils import Exl3Manifest

from .test_exl3_manifest import _matrix, _write


class _Stage(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )

    def __init__(self, manifest):
        super().__init__()
        self.quant_config = Exl3Config()
        self.quant_config.manifest = manifest
        self.model = nn.Module()
        local = nn.Module()
        # Buffers let the CPU test exercise real file loading without pretending
        # CPU parameters satisfy the loader's separate CUDA-residency contract.
        local.register_buffer("weight", torch.zeros(2))
        self.model.layers = nn.ModuleList([local, PPMissingLayer()])
        self.model.hyper_connection_mixer = PPMissingLayer()

    def load_weights(self, weights):
        buffers = dict(self.named_buffers())
        loaded = set()
        for name, value in weights:
            mapped = self.hf_to_vllm_mapper.map_name(name)
            if mapped in buffers:
                buffers[mapped].copy_(value)
                loaded.add(mapped)
        return loaded


def _load_stage(tmp_path, monkeypatch, extra=None):
    import struct

    tensors = _matrix(prefix="model.language_model.layers.1.proj")
    tensors["model.language_model.layers.0.weight"] = (
        "F32",
        [2],
        struct.pack("<ff", 3.0, 7.0),
    )
    tensors["model.language_model.hyper_connection_mixer.weight"] = (
        "F32",
        [1],
        struct.pack("<f", 11.0),
    )
    tensors.update(extra or {})
    _write(tmp_path / "model.safetensors", tensors)
    stage = _Stage(Exl3Manifest(tmp_path))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=1,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen4_exp"),
        multimodal_config=SimpleNamespace(language_model_only=True),
    )
    monkeypatch.setattr(ops, "ensure_extension", lambda: None)
    monkeypatch.setattr(loader, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(loader, "get_tensor_model_parallel_rank", lambda: 0)
    loader.load_exl3_weights(stage, model_config)
    return stage


def test_exl3_pp_loads_local_weights_and_skips_remote_stage(tmp_path, monkeypatch):
    stage = _load_stage(tmp_path, monkeypatch)
    torch.testing.assert_close(stage.model.layers[0].weight, torch.tensor([3.0, 7.0]))


def test_exl3_pp_does_not_hide_unowned_packed_weights(tmp_path, monkeypatch):
    # Layer 1 is a declared remote stage; layer 11 is not. Prefix matching must
    # not turn an out-of-model checkpoint tensor into a permitted remote one.
    with pytest.raises(ValueError, match="packed tensor has no runtime owner"):
        _load_stage(
            tmp_path,
            monkeypatch,
            _matrix(prefix="model.language_model.layers.11.proj"),
        )


def test_exl3_pp_does_not_hide_unowned_plain_weights(tmp_path, monkeypatch):
    import struct

    with pytest.raises(ValueError, match="plain tensor has no runtime owner"):
        _load_stage(
            tmp_path,
            monkeypatch,
            {
                "model.language_model.layers.11.weight": (
                    "F32",
                    [1],
                    struct.pack("<f", 1.0),
                )
            },
        )


@pytest.mark.parametrize("language_model_only", [False, True])
def test_exl3_pp_accepts_qwen4_exp_vision_and_text(language_model_only):
    loader._verify_pipeline_parallel(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp"),
            multimodal_config=SimpleNamespace(language_model_only=language_model_only),
        ),
        SimpleNamespace(pipeline_parallel_size=2, tensor_parallel_size=1),
    )


@pytest.mark.parametrize(
    "model_type,pp,tp",
    [
        ("qwen4_exp", 2, 2),
        ("glm4_moe", 2, 1),
        ("qwen4_exp", 3, 1),
    ],
)
def test_exl3_pp_rejects_unimplemented_layouts(model_type, pp, tp):
    with pytest.raises(ValueError):
        loader._verify_pipeline_parallel(
            SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type),
                multimodal_config=SimpleNamespace(language_model_only=False),
            ),
            SimpleNamespace(pipeline_parallel_size=pp, tensor_parallel_size=tp),
        )
