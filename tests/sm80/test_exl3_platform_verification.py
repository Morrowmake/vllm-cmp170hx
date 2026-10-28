# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization verification of an EXL3 checkpoint on real platform objects.

ModelConfig._verify_quantization asks every registered method's
override_quantization_method, then current_platform.verify_quantization. For
an EXL3 checkpoint the whole step must pass on a CUDA platform and refuse
any other platform with the EXL3 message. The platforms are real
CudaPlatform and CpuPlatform objects; their is_cuda is not replaced, so a
platform check that calls an instance method on the class (a TypeError)
fails here. Runs on CPU; nothing initialises CUDA.
"""
from types import SimpleNamespace

import pytest

import vllm.config.model as model_config
import vllm.platforms
from vllm.config.model import ModelConfig
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import PlatformEnum
from vllm_sm80 import plugin

EXL3_QUANT_CFG = {"quant_method": "exl3", "bits": 4.0}


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(plugin, "_registered", False)
    plugin.register()


def _use_platform(monkeypatch, platform):
    # vllm.config.model binds current_platform at import; the EXL3 config
    # resolves it from vllm.platforms when it runs.
    monkeypatch.setattr(vllm.platforms, "_current_platform", platform)
    monkeypatch.setattr(model_config, "current_platform", platform)


def _verify_exl3_checkpoint():
    config = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(quantization_config=EXL3_QUANT_CFG),
        hf_config=None,
        allow_deprecated_quantization=False,
    )
    ModelConfig._verify_quantization(config)
    return config.quantization


def test_exl3_checkpoint_verifies_on_cuda(registered, monkeypatch):
    platform = CudaPlatform()
    assert platform._enum == PlatformEnum.CUDA
    _use_platform(monkeypatch, platform)
    assert _verify_exl3_checkpoint() == "exl3"
    type(platform).verify_quantization("exl3")


def test_exl3_checkpoint_is_refused_on_cpu(registered, monkeypatch):
    platform = CpuPlatform()
    assert platform._enum != PlatformEnum.CUDA
    _use_platform(monkeypatch, platform)
    with pytest.raises(ValueError, match="EXL3 execution requires the NVIDIA CUDA"):
        _verify_exl3_checkpoint()
