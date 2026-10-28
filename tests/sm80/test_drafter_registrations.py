# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork plugin's drafter registrations (DFlash2 and LiLiCorr).

After :func:`vllm_sm80.plugin.register_drafter_models` the model registry
resolves each drafter architecture to the fork class, which describes itself
exactly as the upstream class it replaces (the same model information).
"""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.registry import ModelRegistry, _ModelInfo
from vllm_sm80 import plugin
from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig


def _load(qualified: str) -> type:
    module_name, _, class_name = qualified.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


@pytest.fixture(autouse=True)
def _drafters_registered():
    plugin.register_drafter_models()


@pytest.mark.parametrize("architecture", sorted(plugin.DRAFTER_OVERRIDES))
def test_drafter_resolves_to_the_fork_class(architecture):
    qualified, upstream_module = plugin.DRAFTER_OVERRIDES[architecture]
    fork_cls = _load(qualified)
    assert ModelRegistry._try_load_model_cls(architecture) is fork_cls
    upstream_cls = getattr(importlib.import_module(upstream_module), fork_cls.__name__)
    assert fork_cls.__bases__ == (upstream_cls,)
    assert _ModelInfo.from_model_cls(fork_cls) == _ModelInfo.from_model_cls(
        upstream_cls
    )


def test_register_drafter_models_is_idempotent():
    plugin.register_drafter_models()
    plugin.register_drafter_models()
    for architecture, (qualified, _) in plugin.DRAFTER_OVERRIDES.items():
        assert ModelRegistry._try_load_model_cls(architecture) is _load(qualified)


def test_model_info_cache_key_covers_the_upstream_module(tmp_path, monkeypatch):
    """The cached model information of a fork drafter class is keyed on its
    module and on the upstream module its class extends."""
    from vllm.model_executor.models.registry import _LazyRegisteredModel
    from vllm_sm80.model_registry import UpstreamHashedLazyModel

    for architecture in plugin.DRAFTER_OVERRIDES:
        assert type(ModelRegistry.models[architecture]) is UpstreamHashedLazyModel
    upstream = tmp_path / "sm80_upstream_drafter_probe.py"
    upstream.write_text("X = 1\n")
    fork_module = tmp_path / "fork_drafter.py"
    fork_module.write_text("Y = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = UpstreamHashedLazyModel(
        "fork_drafter", "Drafter", "sm80_upstream_drafter_probe"
    )
    key = entry._get_modelinfo_module_hash(fork_module)
    module_only = _LazyRegisteredModel._get_modelinfo_module_hash(fork_module)
    upstream.write_text("X = 2\n")
    assert entry._get_modelinfo_module_hash(fork_module) != key
    assert _LazyRegisteredModel._get_modelinfo_module_hash(fork_module) == module_only
    key = entry._get_modelinfo_module_hash(fork_module)
    fork_module.write_text("Y = 2\n")
    assert entry._get_modelinfo_module_hash(fork_module) != key
    assert "sm80_upstream_drafter_probe" not in sys.modules


def test_dynamic_conv_op_keeps_its_name():
    from vllm.model_executor.models import qwen3_dflash2
    from vllm_sm80.spec_decode import dflash2_model

    assert not hasattr(qwen3_dflash2, "dflash2_grouped_conv_dyn_impl")
    schema = str(torch.ops.vllm.dflash2_grouped_conv_dyn.default._schema)
    assert schema.startswith("vllm::dflash2_grouped_conv_dyn(")
    assert dflash2_model.dflash2_grouped_conv_dyn_impl is not None


def _width_cfg(draft_by_load):
    adaptive_k = {"by_load": [5, 4, 3], "draft_by_load": draft_by_load}
    if not draft_by_load:
        adaptive_k.pop("draft_by_load")
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            adaptive_k_config=AdaptiveKConfig.from_dict(adaptive_k, 5),
            draft_sample_method="greedy",
            enable_adaptive_verification=False,
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )


@pytest.mark.parametrize("draft_by_load", [False, True])
def test_lilicorr_convs_follow_the_draft_width(monkeypatch, draft_by_load):
    """The fork LiLiCorr gives its DFlash2 convolutions the runtime block
    length exactly when the draft width follows the load."""
    from vllm.model_executor.models import lilicorr, qwen3_dflash2
    from vllm_sm80.spec_decode import dflash2_model, lilicorr_model

    def upstream_init(self, *, vllm_config, start_layer_id=0, prefix=""):
        torch.nn.Module.__init__(self)
        conv = object.__new__(qwen3_dflash2.DFlashGroupedConv)
        torch.nn.Module.__init__(conv)
        conv.block_size = 6
        self.conv = conv

    monkeypatch.setattr(lilicorr.LiLiCorr, "__init__", upstream_init)
    model = lilicorr_model.LiLiCorr(vllm_config=_width_cfg(draft_by_load))
    if draft_by_load:
        assert type(model.conv) is dflash2_model.DFlashGroupedConv
        assert model.conv.block_size_tensor.tolist() == [6]
        assert model.conv.block_size_tensor.dtype == torch.int32
        assert "conv.block_size_tensor" not in model.state_dict()
    else:
        assert type(model.conv) is qwen3_dflash2.DFlashGroupedConv
        assert getattr(model.conv, "block_size_tensor", None) is None
    assert lilicorr_model.LiLiCorrForCausalLM.model_cls is lilicorr_model.LiLiCorr
