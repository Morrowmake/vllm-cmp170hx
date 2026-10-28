# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real native-config constructor regression, with no weights or GPU context.

The fixtures are byte-exact deployed checkpoint configs (see provenance.json).
Unlike tiny synthetic configs, these preserve all nested text/vision configs,
layer kinds, PLE tables, MoE dimensions and quantization metadata. Loading uses
vLLM get_config/get_hf_text_config, not direct Transformers construction.

ModelConfig/VllmConfig below are typed constructor-runtime views. Their engine
post-init is deliberately NOT run: it resolves checkpoint weight quantizers,
inspects model classes in subprocesses, and configures engine executors. In
particular EXL3 is not registered in every branch that shares these tests. Only
pure runtime defaults are supplied; native HF attributes are never inserted,
removed, overridden, or replaced with a mock. Quantized weight creation is out
of scope: quant_config=None, while checkpoint quantization metadata is asserted
separately. Every model/layer constructor remains real and executes on meta.
GLM and drafter classes are resolved through vLLM's model registry after the
fork plugin's registrations, which is how a serving process picks them.

Requires the CUDA vLLM Python dependencies/extensions (including FA2), but no
visible GPU. Static SM80 platform traits select real backend implementations;
CUDA lazy initialization is forbidden. Run in a fresh process with
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q tests/models/test_native_checkpoint_config_meta.py
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from pydantic_core import PydanticUndefined

from vllm.config import (
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.utils.torch_utils import set_default_torch_dtype

_FIXTURES = Path(__file__).parent / "fixtures" / "native_checkpoint_configs"


def _load_checkpoint(name):
    provenance = json.loads((_FIXTURES / "provenance.json").read_text())
    checkpoint = _FIXTURES / name
    raw = (checkpoint / "config.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance["fixtures"][name]["sha256"]
    hf_config = get_config(str(checkpoint), trust_remote_code=False)
    text_config = get_hf_text_config(hf_config)
    return checkpoint, json.loads(raw), hf_config, text_config


def _runtime_defaults(cls, **overrides):
    """Use declared runtime defaults without engine/quantizer post-init.

    This is not an HF-config mock: all objects retain their real methods and
    properties, and the native configs are attached unchanged by the caller.
    Missing required runtime fields remain missing rather than being invented.
    """
    instance = object.__new__(cls)
    for name, field in cls.__pydantic_fields__.items():
        if name in overrides:
            continue
        value = field.get_default(call_default_factory=True)
        if value is not PydanticUndefined:
            setattr(instance, name, value)
    for name, value in overrides.items():
        setattr(instance, name, value)
    return instance


def _model_config(name):
    checkpoint, raw, hf_config, text_config = _load_checkpoint(name)
    config = _runtime_defaults(
        ModelConfig,
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        hf_config=hf_config,
        hf_text_config=text_config,
        dtype=torch.bfloat16,
        max_model_len=128,
        original_max_model_len=128,
        runner_type="generate",
        convert_type="none",
        multimodal_config=None,
        quantization=None,
        skip_tokenizer_init=True,
    )
    config.model_arch_config = config.get_model_arch_config()
    # ModelConfig.__post_init__ (not run here) resolves the registered
    # architecture's interfaces, which properties such as
    # supports_kv_cache_grouping read.
    _register_fork_plugin()
    config._model_info, config._architecture = config.registry.inspect_model_cls(
        config.architectures, config
    )
    return config


def _vllm_config(name, tp_size, pp_size=1, *, with_dflash=False):
    model_config = _model_config(name)
    parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="mp",
    )
    scheduler_config = SchedulerConfig(
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        is_encoder_decoder=False,
    )
    compilation_config = CompilationConfig(
        mode=0, cudagraph_mode="NONE", custom_ops=["all"]
    )
    config = _runtime_defaults(
        VllmConfig,
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
        quant_config=None,
    )
    config.kernel_config.enable_jit_warmup = False
    # VllmConfig.__post_init__ (not run here) resolves the sparse MLA prefill
    # workspace factor from max_model_len: 5 up to 256K tokens.
    config.attention_config.sparse_mla_prefill_workspace_factor = 5
    if with_dflash:
        draft_model_config = _model_config("glm53_dflash2")
        config.speculative_config = _runtime_defaults(
            SpeculativeConfig,
            model=draft_model_config.model,
            method="dflash",
            num_speculative_tokens=7,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            draft_model_config=draft_model_config,
            draft_parallel_config=parallel_config,
        )
    return config


@dataclass
class _ConstructionGroup:
    """Rank metadata only; no process group or collective is initialized."""

    world_size: int = 1
    rank_in_group: int = 0
    device_communicator: object = None

    @property
    def is_first_rank(self):
        return self.rank_in_group == 0

    @property
    def is_last_rank(self):
        return self.rank_in_group == self.world_size - 1

    @property
    def device_group(self):
        return self

    def size(self):
        return self.world_size


def _parallel_groups(monkeypatch, tp_size, tp_rank, pp_size=1, pp_rank=0):
    import vllm.distributed.parallel_state as ps

    tp = _ConstructionGroup(tp_size, tp_rank)
    pp = _ConstructionGroup(pp_size, pp_rank)
    singleton = _ConstructionGroup()
    for name, group in {
        "_TP": tp,
        "_ETP": tp,
        "_PP": pp,
        "_DP": singleton,
        "_EP": singleton,
        "_DCP": singleton,
        "_PCP": singleton,
        "_WORLD": _ConstructionGroup(tp_size * pp_size, pp_rank * tp_size + tp_rank),
    }.items():
        monkeypatch.setattr(ps, name, group)


@pytest.fixture
def meta_platform(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability, PlatformEnum

    assert not torch.cuda.is_initialized(), "Run this CPU test in a fresh process"

    def forbid_cuda_init(*args, **kwargs):
        raise AssertionError("Meta constructor attempted native GPU initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_cuda_init)
    # Third-party kernel imports query Torch hardware metadata, independently
    # of vLLM's platform. Answer those queries statically, not by opening CUDA.
    @dataclass(frozen=True)
    class SM80Properties:
        name: str = "Static SM80 meta-construction device"
        major: int = 8
        minor: int = 0
        multi_processor_count: int = 108
        total_memory: int = 80 * 1024**3
        warp_size: int = 32

    properties = SM80Properties()
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device=None: properties
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (8, 0)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    platform_cls = type(current_platform)
    monkeypatch.setattr(platform_cls, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(platform_cls, "device_type", "meta")
    monkeypatch.setattr(platform_cls, "dispatch_key", "CUDA")
    monkeypatch.setattr(
        platform_cls,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(8, 0)),
    )
    monkeypatch.setattr(
        platform_cls, "current_device", staticmethod(lambda: torch.device("meta")),
        raising=False,
    )

    def backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        if attn_selector_config.use_mla:
            assert attn_selector_config.use_sparse
            return "vllm_sm80.attention.triton_mla_sparse.TritonMLASparseBackend"
        return "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    monkeypatch.setattr(platform_cls, "get_attn_backend_cls", classmethod(backend_cls))
    # Shared-expert stream/event allocation is execution state, not model config.
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    yield
    assert not torch.cuda.is_initialized()


def _register_fork_plugin():
    """Run the fork plugin's registrations, as vLLM does in every serving
    process, so the registry resolves the classes a real run builds."""
    from vllm_sm80 import plugin

    plugin.register()


def _registered_model_cls(architecture):
    from vllm.model_executor.models.registry import ModelRegistry

    _register_fork_plugin()
    model_cls = ModelRegistry._try_load_model_cls(architecture)
    assert model_cls is not None, architecture
    return model_cls


def _assert_meta_model(model):
    parameters = list(model.named_parameters())
    assert parameters, "Real module constructors must create model parameters"
    for name, parameter in parameters:
        assert parameter.is_meta, name
    for name, buffer in model.named_buffers():
        assert buffer.is_meta, name


@pytest.mark.parametrize("name", ["glm53_flash", "qwen38_flash_next", "glm53_dflash2"])
def test_real_native_checkpoint_config_contract(name):
    from transformers import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Qwen3Config,
        Qwen4ExpConfig,
        Qwen4ExpTextConfig,
    )

    _, raw, hf_config, text = _load_checkpoint(name)
    if name == "glm53_flash":
        assert type(hf_config) is Glm5NextConfig
        assert type(text) is Glm5NextTextConfig
        assert text is hf_config.text_config
        assert (text.hidden_size, text.num_hidden_layers, text.hc_mult) == (4096, 45, 4)
        assert text.mhc is True
        assert not hasattr(text, "mhc_num_residual_streams")
        assert text.layer_types == raw["text_config"]["layer_types"]
        assert hf_config.quantization_config == raw["quantization_config"]
        assert hf_config.quantization_config["quant_method"] == "compressed-tensors"
    elif name == "qwen38_flash_next":
        assert type(hf_config) is Qwen4ExpConfig
        assert type(text) is Qwen4ExpTextConfig
        assert text is hf_config.text_config
        assert (text.hidden_size, text.num_hidden_layers, text.hc_count) == (2560, 48, 4)
        assert text.mtp_use_dedicated_embeddings is False
        assert text.mtp_num_hidden_layers == 1
        assert text.ple_layer_ids == [2]
        # Native Transformers may canonicalize the serialized full_attention
        # spelling, but must retain every layer and sparse-indexer setting.
        assert len(text.layer_types) == len(raw["text_config"]["layer_types"])
        assert text.indexer_n_heads == 4
        assert hf_config.quantization_config == raw["quantization_config"]
        assert hf_config.quantization_config["quant_method"] == "exl3"
    else:
        assert type(hf_config) is Qwen3Config
        assert text is hf_config
        assert (text.hidden_size, text.num_hidden_layers, text.num_target_layers) == (4096, 5, 45)
        assert text.architectures == ["DFlash2DraftModel"]
        assert text.dflash_config == raw["dflash_config"]
        assert text.dflash_config["target_layer_ids"] == [5, 14, 24, 33, 42]


_GLM_TOPOLOGIES = (
    [(4, rank, 1, 0) for rank in range(4)]
    + [(1, 0, 4, stage) for stage in range(4)]
    + [(4, 0, 4, stage) for stage in range(4)]
)


@pytest.mark.parametrize("tp_size,tp_rank,pp_size,pp_rank", _GLM_TOPOLOGIES)
def test_glm_real_config_meta_constructor(
    monkeypatch, meta_platform, tp_size, tp_rank, pp_size, pp_rank
):
    from vllm.model_executor.models.utils import PPMissingLayer

    Glm5NextForCausalLM = _registered_model_cls("Glm5NextForCausalLM")
    _parallel_groups(monkeypatch, tp_size, tp_rank, pp_size, pp_rank)
    config = _vllm_config("glm53_flash", tp_size, pp_size, with_dflash=True)
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            model = Glm5NextForCausalLM(vllm_config=config)
            boundary = model.make_empty_intermediate_tensors(2, torch.bfloat16, torch.device("meta"))
    # The original head fails inside Glm5NextModel.__init__, reading the absent
    # config.mhc_num_residual_streams. Do not catch or xfail that regression.
    _assert_meta_model(model)
    backbone = model.model
    text = config.model_config.hf_text_config
    assert backbone.config is text
    assert backbone.mhc is True
    assert backbone.hc_mult == text.hc_mult
    assert boundary["hidden_states"].shape == (2, 4, 4096)
    assert boundary["hidden_states"].is_meta
    expected_range = (0, 45) if pp_size == 1 else [(0, 11), (11, 22), (22, 34), (34, 45)][pp_rank]
    assert (backbone.start_layer, backbone.end_layer) == expected_range
    active = list(backbone.layers[backbone.start_layer:backbone.end_layer])
    assert len(active) == expected_range[1] - expected_range[0]
    assert {layer.layer_kind for layer in active} == {"kda", "mla"}
    assert all(layer.n == text.hc_mult for layer in active)
    assert isinstance(backbone.norm, PPMissingLayer) == (pp_rank != pp_size - 1)
    assert isinstance(model.lm_head, PPMissingLayer) == (pp_rank != pp_size - 1)
    # DFlash needs an embedding on the last PP stage, including with TP4+PP4.
    assert isinstance(backbone.embed_tokens, PPMissingLayer) == (pp_rank not in (0, pp_size - 1))


@pytest.mark.parametrize("tp_rank", range(2))
def test_qwen_real_config_tp2_meta_constructor(monkeypatch, meta_platform, tp_rank):
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM

    _parallel_groups(monkeypatch, 2, tp_rank)
    config = _vllm_config("qwen38_flash_next", 2)
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            model = Qwen4ExpForCausalLM(vllm_config=config)
            boundary = model.make_empty_intermediate_tensors(2, torch.bfloat16, torch.device("meta"))
    _assert_meta_model(model)
    backbone = model.model
    assert backbone.config is config.model_config.hf_text_config
    assert (backbone.start_layer, backbone.end_layer) == (0, 48)
    assert len(backbone.layers) == 48
    assert sum(hasattr(layer, "linear_attn") for layer in backbone.layers) == 36
    assert sum(hasattr(layer, "self_attn") for layer in backbone.layers) == 12
    assert backbone.layers[1].ple is not None
    assert all(layer.ple is None for idx, layer in enumerate(backbone.layers) if idx != 1)
    assert backbone.hyper_connection_mixer is not None
    assert boundary["hidden_states"].shape == (2, 4 * 2560)
    assert boundary["hidden_states"].is_meta


@pytest.mark.parametrize("tp_rank", range(4))
def test_real_dflash2_tp4_meta_constructor(monkeypatch, meta_platform, tp_rank):
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm_sm80 import plugin

    # The class vLLM resolves for the architecture once the fork plugin ran.
    plugin.register()
    EntryClass = ModelRegistry._try_load_model_cls("DFlash2DraftModel")

    _parallel_groups(monkeypatch, 4, tp_rank)
    config = _vllm_config("glm53_flash", 4, with_dflash=True)
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            model = EntryClass(vllm_config=config)
    _assert_meta_model(model)
    draft = config.speculative_config.draft_model_config.hf_config
    assert model.config is draft
    assert model.model.config is draft
    assert len(model.model.layers) == 5
    assert model.model.fc.weight.shape == (4096, 5 * 4096)
    assert model.model.candidate_selector.predecessor_codebook.shape[1] == 256
    assert model.model.candidate_selector.top_k == 16
    for layer in model.model.layers:
        assert layer.attention_conv.block_size == 8
        assert layer.mlp_conv.taps == 2
        assert layer.attention_conv.group_size == 16
        assert layer.self_attn.sliding_window == 2048
        assert layer.self_attn.causal is False
