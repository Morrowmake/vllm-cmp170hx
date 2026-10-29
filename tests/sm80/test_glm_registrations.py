# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork plugin's GLM model registrations.

After :func:`vllm_sm80.plugin.register` the model registry resolves each
overridden architecture to the fork class, and describes it exactly as the
upstream class it replaces (the same model information), so vLLM treats the
model the same way (pipeline parallel, hybrid KV cache, inner state).
"""

import importlib

import pytest
import torch

from tests.models.test_native_checkpoint_config_meta import (
    _parallel_groups,
    _vllm_config,
    meta_platform,  # noqa: F401
)
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.registry import ModelRegistry, _ModelInfo
from vllm_sm80 import plugin


def _load(qualified: str) -> type:
    module_name, _, class_name = qualified.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


@pytest.fixture(autouse=True)
def _models_registered():
    plugin.register_models()


def test_register_is_idempotent():
    plugin.register_models()
    plugin.register_models()
    for architecture, qualified in plugin.MODEL_OVERRIDES.items():
        assert ModelRegistry._try_load_model_cls(architecture) is _load(qualified)


@pytest.mark.parametrize(
    "architecture", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
def test_glm_model_info_matches_upstream(architecture):
    from vllm.models.glm5next.common import model as upstream

    fork_cls = _load(plugin.MODEL_OVERRIDES[architecture])
    upstream_cls = getattr(upstream, architecture)
    assert fork_cls.__bases__[-1] is upstream_cls
    assert fork_cls.__name__ == upstream_cls.__name__ == architecture
    info = _ModelInfo.from_model_cls(fork_cls)
    assert info == _ModelInfo.from_model_cls(upstream_cls)
    assert info.supports_pp and info.is_hybrid and info.has_inner_state
    # Drafters read auxiliary hidden states from the registered classes, whose
    # backbone (the fork one, see adopt) also carries them across PP stages.
    assert supports_eagle3(fork_cls)
    fork_backbone = importlib.import_module(fork_cls.__module__).Glm5NextModel
    assert issubclass(fork_backbone, upstream.Glm5NextModel)
    assert fork_backbone.supports_aux_hidden_states_over_pp



@pytest.mark.parametrize(
    "architecture", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
@pytest.mark.parametrize(
    ("is_hybrid", "is_attention_free", "expected"),
    [
        (True, False, "Glm5NextModelState"),
        (False, True, "Glm5NextModelState"),
        (False, False, "DefaultModelState"),
    ],
)
def test_glm_classes_pick_the_glm_model_state(
    architecture, is_hybrid, is_attention_free, expected
):
    """Glm5NextModelState wherever vLLM would pick MambaHybridModelState,
    otherwise vLLM's own choice."""
    from types import SimpleNamespace

    from torch import nn

    from vllm_sm80.models.glm5next.model_state import Glm5NextModelState
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=is_hybrid, is_attention_free=is_attention_free
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    fork_cls = _load(plugin.MODEL_OVERRIDES[architecture])
    model = object.__new__(fork_cls)
    nn.Module.__init__(model)
    model.vllm_config = model._sm80_vllm_config = config
    assert resolve_model_state_cls(config, model).__name__ == expected
    assert issubclass(Glm5NextModelState, MambaHybridModelState)


@pytest.mark.parametrize(
    ("use_kda_recoverssm", "fork_recover", "expected"),
    [(False, False, False), (False, True, True), (True, False, True)],
)
def test_glm_model_state_enables_recoverssm(
    monkeypatch, use_kda_recoverssm, fork_recover, expected
):
    from types import SimpleNamespace

    from vllm_sm80.models.glm5next.model_state import Glm5NextModelState
    from vllm_sm80.kernels.kda import recover
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

    def base_init(self, vllm_config, model, encoder_cache, device):
        self.recoverssm = RecoverSSMState() if use_kda_recoverssm else None

    monkeypatch.setattr(MambaHybridModelState, "__init__", base_init)
    monkeypatch.setattr(recover, "kda_recover_enabled", lambda config: fork_recover)
    # The commit fold is not under test: pin it off, so a fold switch set in
    # the environment does not reach kda_fold_enabled with this bare config.
    monkeypatch.setenv("VLLM_AMPERE_KDA_FOLD", "0")
    state = Glm5NextModelState(SimpleNamespace(), None, None, None)
    assert isinstance(state.recoverssm, RecoverSSMState) == expected
    assert state._kda_fold is None


@pytest.mark.parametrize("fold", [False, True])
def test_glm_model_state_calls_the_fold_hooks(monkeypatch, fold):
    """The KDA commit fold's hooks run after the hybrid state's own work on
    add_request, remove_request and prepare_attn (on the metadata it built);
    with the fold off there are no hooks and the hybrid results pass through."""
    from types import SimpleNamespace

    from vllm_sm80.models.glm5next.model_state import Glm5NextModelState
    from vllm_sm80.kernels.kda import recover
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    calls = []
    metadata = {"layer": object()}

    def base_init(self, vllm_config, model, encoder_cache, device):
        self.recoverssm = None

    def base_prepare_attn(self, input_batch, *args, **kwargs):
        calls.append(("base.prepare_attn", kwargs["for_capture"]))
        return metadata

    class Hooks:
        def __init__(self, vllm_config, device):
            calls.append(("hooks", device))

        def add_request(self, slot, req_id, num_computed_tokens):
            calls.append(("fold.add", slot, req_id, num_computed_tokens))

        def remove_request(self, req_id):
            calls.append(("fold.remove", req_id))

        def prepare_step(self, attn_metadata, idx_mapping, for_capture):
            calls.append(("fold.step", attn_metadata is metadata, idx_mapping,
                          for_capture))

    monkeypatch.setattr(MambaHybridModelState, "__init__", base_init)
    monkeypatch.setattr(
        MambaHybridModelState, "add_request",
        lambda self, i, data: calls.append(("base.add", i)))
    monkeypatch.setattr(
        MambaHybridModelState, "remove_request",
        lambda self, req_id: calls.append(("base.remove", req_id)))
    monkeypatch.setattr(MambaHybridModelState, "prepare_attn", base_prepare_attn)
    monkeypatch.setattr(recover, "kda_recover_enabled", lambda config: fold)
    monkeypatch.setattr(
        recover, "kda_fold_hooks",
        lambda config, device: Hooks(config, device) if fold else None)

    state = Glm5NextModelState(SimpleNamespace(), None, None, "dev")
    assert (state._kda_fold is None) != fold
    state.add_request(3, SimpleNamespace(req_id="r", num_computed_tokens=17))
    state.remove_request("r")
    batch = SimpleNamespace(idx_mapping="idx")
    for for_capture in (False, True):
        out = state.prepare_attn(batch, None, (), None, [], None,
                                 for_capture=for_capture)
        assert out is metadata
    if fold:
        assert calls == [
            ("hooks", "dev"),
            ("base.add", 3), ("fold.add", 3, "r", 17),
            ("base.remove", "r"), ("fold.remove", "r"),
            ("base.prepare_attn", False), ("fold.step", True, "idx", False),
            ("base.prepare_attn", True), ("fold.step", True, "idx", True),
        ]
    else:
        assert calls == [
            ("base.add", 3), ("base.remove", "r"),
            ("base.prepare_attn", False), ("base.prepare_attn", True),
        ]


def test_kda_layer_is_replaced_by_the_fork_layer():
    """Importing the fork GLM model module registers the KDA layer override;
    the upstream layer class then builds the fork layer, whose attention
    backend is the GLM GDN backend."""
    from vllm.model_executor.custom_op import maybe_get_oot_by_class
    from vllm.models.glm5next.common import kda as upstream
    from vllm_sm80.models.glm5next import sm80_kda
    from vllm_sm80.kernels.kda.recover import (
        Glm5KDARecoverMetadata,
        Glm5NextGDNAttentionMetadata,
    )
    from vllm.v1.attention.backends.gdn_attn import (
        GDNAttentionBackend,
        GDNAttentionMetadata,
        GDNAttentionMetadataBuilder,
    )

    _load(plugin.MODEL_OVERRIDES["Glm5NextForCausalLM"])
    plugin.register_glm_layers()
    fork_cls = sm80_kda.Glm5NextLinearAttention
    upstream_cls = upstream.Glm5NextLinearAttention
    assert fork_cls.__bases__ == (upstream_cls,)
    assert maybe_get_oot_by_class(upstream_cls) is fork_cls
    layer = upstream_cls.__new__(upstream_cls)
    assert type(layer) is fork_cls

    backend = layer.get_attn_backend()
    assert backend is sm80_kda.Glm5NextGDNAttentionBackend
    assert backend.__bases__ == (GDNAttentionBackend,)
    assert backend.get_name() == GDNAttentionBackend.get_name()
    builder_cls = backend.get_builder_cls()
    assert builder_cls.__bases__ == (GDNAttentionMetadataBuilder,)
    assert Glm5NextGDNAttentionMetadata.__bases__ == (GDNAttentionMetadata,)
    assert issubclass(Glm5KDARecoverMetadata, Glm5NextGDNAttentionMetadata)


def test_mla_layer_is_replaced_by_the_fork_layer(monkeypatch, meta_platform):  # noqa: F811
    """Importing the fork GLM model module also registers the MLA wrapper
    override, so every MLA layer the GLM model builds is the fork wrapper
    (built by the upstream constructor) holding the fork MLAAttention.

    ``PluggableLayer.register_oot`` sets the fork wrapper's ``name`` to the
    registered name ("MultiHeadLatentAttentionWrapper", upstream's is
    "multi_head_latent_attention"); nothing reads a pluggable layer's
    ``name`` today."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.custom_op import maybe_get_oot_by_class
    from vllm.model_executor.layers import mla
    from vllm.model_executor.layers.attention import mla_attention
    from vllm_sm80.models.glm5next import sm80_mla
    from vllm.utils.torch_utils import set_default_torch_dtype

    model_cls = _load(plugin.MODEL_OVERRIDES["Glm5NextForCausalLM"])
    upstream_cls = mla.MultiHeadLatentAttentionWrapper
    fork_cls = sm80_mla.MultiHeadLatentAttentionWrapper
    assert fork_cls.__bases__ == (upstream_cls,)
    assert sm80_mla.MLAAttention.__bases__ == (mla_attention.MLAAttention,)
    assert maybe_get_oot_by_class(upstream_cls) is fork_cls
    assert fork_cls.name == "MultiHeadLatentAttentionWrapper"
    assert upstream_cls.name == "multi_head_latent_attention"

    # First pipeline stage of PP4 (two MLA layers), built on the meta device.
    _parallel_groups(monkeypatch, 1, 0, 4, 0)
    config = _vllm_config("glm53_flash", 1, 4)
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            model = model_cls(vllm_config=config)
    wrappers = [m for m in model.modules() if isinstance(m, upstream_cls)]
    assert len(wrappers) == 2
    assert all(type(w) is fork_cls for w in wrappers)
    assert all(type(w.mla_attn) is sm80_mla.MLAAttention for w in wrappers)


def test_model_info_cache_key_covers_the_model_package(tmp_path, monkeypatch):
    """The cached model information of the fork GLM classes is keyed on the
    fork package vllm_sm80/models/glm5next and the whole upstream package
    vllm/models/glm5next, so a change to the upstream model code
    (common/model.py) or to another fork module changes the key; a key over
    sm80_model.py alone would not."""
    import shutil
    from pathlib import Path

    import vllm_sm80.model_registry as model_registry
    from vllm.model_executor.models.registry import _LazyRegisteredModel
    from vllm.models import glm5next
    from vllm_sm80.model_registry import PackageHashedLazyModel
    from vllm_sm80.models import glm5next as fork_glm5next

    for architecture in plugin.MODEL_OVERRIDES:
        assert type(ModelRegistry.models[architecture]) is PackageHashedLazyModel

    upstream_copy = tmp_path / "vllm" / "models" / "glm5next"
    fork_copy = tmp_path / "vllm_sm80" / "models" / "glm5next"
    for source, copy in ((glm5next, upstream_copy), (fork_glm5next, fork_copy)):
        shutil.copytree(
            Path(source.__file__).parent,
            copy,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    monkeypatch.setattr(model_registry, "upstream_model_package_init",
                        lambda model: tmp_path / "vllm" / "models" / model / "__init__.py")
    fork_module = fork_copy / "sm80_model.py"
    key = PackageHashedLazyModel._get_modelinfo_module_hash(fork_module)
    module_only = _LazyRegisteredModel._get_modelinfo_module_hash(fork_module)

    upstream_model = upstream_copy / "common" / "model.py"
    upstream_model.write_bytes(upstream_model.read_bytes() + b"\n")
    changed_upstream = PackageHashedLazyModel._get_modelinfo_module_hash(fork_module)
    assert changed_upstream != key
    fork_kda = fork_copy / "sm80_kda.py"
    fork_kda.write_bytes(fork_kda.read_bytes() + b"\n")
    assert PackageHashedLazyModel._get_modelinfo_module_hash(fork_module) != changed_upstream
    assert _LazyRegisteredModel._get_modelinfo_module_hash(fork_module) == module_only
