# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct the real checkpoint-declared multimodal wrappers on CPU/meta.

This supplements, rather than substitutes for, the text/drafter regressions in
``test_native_checkpoint_config_meta``. Both deployed configs actually declare
ForConditionalGeneration wrappers. Construction resolves those exact names
through vLLM's registry and initialize_model, including their real vision tower,
input normalization, language-model and MoE metadata constructors.

Deployment provenance (2026-10-06): GLM serving_defaults.sh has MM_CAP=0, and
serve.sh adds no modality-limit flags in that mode. The Qwen launcher likewise
has no language-model-only/limit-mm flags and explicitly reports vision=on.
Therefore default positive image/video limits are the canonical deployed mode.
No modality is disabled to make construction cheaper. The full HF config.json
is copied byte-for-byte from the existing provenance-checked fixtures. Complete
real processor metadata below is copied from those deployed checkpoints, not
synthetic normalization defaults, and is staged next to the untouched config.

The existing typed runtime views/static SM80/rank fixtures remain unchanged.
Native ModelConfig metadata methods and VllmConfig.with_hf_config are real; its
nested submodel replacement takes the library's normal is_submodel_config path.
Tokenizer initialization remains disabled by the existing constructor-runtime
flag; the real tokenizer helper returns None and no tokenizer is mocked.
GPU initialization, weights, inference and profiling are forbidden/out of scope.
"""

import json
import shutil
from contextlib import contextmanager

import pytest
import torch

from vllm.config import MultiModalConfig, SpeculativeConfig, set_current_vllm_config
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.utils.torch_utils import set_default_torch_dtype

from .test_native_checkpoint_config_meta import (
    _FIXTURES,
    _GLM_TOPOLOGIES,
    _assert_meta_model,
    _parallel_groups,
    _register_fork_plugin,
    _vllm_config,
    meta_platform,
)

# Complete deployed processor files. Capture provenance: checkpoint basenames
# GLM-5.3-Flash-W4A16-MTP/processor_config.json and
# qwen38-flash-next-exl3-5.05/{preprocessor,video_preprocessor}_config.json.
_PROCESSOR_FILES = {
    "glm53_flash": {
        "processor_config.json": {
            "image_processor": {
                "do_rescale": True,
                "patch_expand_factor": 1,
                "merge_size": 2,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "temporal_patch_size": 2,
                "patch_size": 14,
                "min_image_tokens": 16,
                "max_image_tokens": 8000,
                "image_processor_type": "Glm5NextImageProcessor",
            },
            "video_processor": {
                "do_rescale": True,
                "video_processor_type": "Glm5NextVideoProcessor",
                "patch_expand_factor": 1,
                "merge_size": 2,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "temporal_patch_size": 2,
                "patch_size": 14,
                "min_image_tokens": 16,
                "max_image_tokens": 240000,
                "fps": 2,
            },
            "processor_class": "Glm5NextProcessor",
        },
    },
    "qwen38_flash_next": {
        "preprocessor_config.json": {
            "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
        },
        "video_preprocessor_config.json": {
            "size": {"longest_edge": 25165824, "shortest_edge": 4096},
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "processor_class": "Qwen3VLProcessor",
            "video_processor_type": "Qwen3VLVideoProcessor",
        },
    },
}

_ARCHITECTURES = {
    "glm53_flash": "Glm5NextForConditionalGeneration",
    "qwen38_flash_next": "Qwen4ExpForConditionalGeneration",
}


def _wrapper_runtime_config(tmp_path, name, tp_size, pp_size):
    """Attach actual multimodal metadata to the shared typed runtime view."""
    from vllm.model_executor.models.registry import _ModelInfo

    config = _vllm_config(
        name, tp_size, pp_size, with_dflash=name == "glm53_flash"
    )
    spec = config.speculative_config
    if spec is not None:
        # The real drafter runs on PP=1 (on the target's tail stage), not on
        # the target's PP topology. Reuse the production builders so nested
        # with_hf_config verification checks the actual DFlash2 deployment.
        spec.draft_tensor_parallel_size = SpeculativeConfig._verify_and_get_draft_tp(
            spec.target_parallel_config,
            spec.draft_tensor_parallel_size,
        )
        spec.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
            spec.target_parallel_config,
            spec.draft_tensor_parallel_size,
            draft_model_config=spec.draft_model_config,
        )
        assert spec.draft_parallel_config.pipeline_parallel_size == 1
        assert spec.draft_parallel_config.tensor_parallel_size == tp_size
    model_config = config.model_config
    checkpoint = tmp_path / name
    checkpoint.mkdir()
    shutil.copyfile(_FIXTURES / name / "config.json", checkpoint / "config.json")
    for filename, metadata in _PROCESSOR_FILES[name].items():
        (checkpoint / filename).write_text(json.dumps(metadata))

    # Reload through the real vLLM path with the real processor metadata now
    # adjacent to the unchanged checkpoint config; never replace native attrs.
    model_config.model = str(checkpoint)
    model_config.tokenizer = str(checkpoint)
    model_config.hf_config = get_config(str(checkpoint), trust_remote_code=False)
    model_config.hf_text_config = get_hf_text_config(model_config.hf_config)
    declared_architectures = model_config.hf_config.architectures
    _register_fork_plugin()
    assert declared_architectures == [_ARCHITECTURES[name]]
    model_cls, architecture = model_config.registry.resolve_model_cls(
        declared_architectures, model_config=model_config
    )
    assert architecture == _ARCHITECTURES[name]
    assert model_cls.__name__ == architecture
    model_config._architecture = architecture
    model_config._model_info = _ModelInfo.from_model_cls(model_cls)
    assert model_config._model_info.supports_multimodal

    # Both real launchers keep image and video enabled with the stock limits.
    model_config.multimodal_config = MultiModalConfig()
    model_config.multimodal_config.mm_device_do_normalize = (
        model_config._resolve_mm_device_do_normalize(None)
    )
    assert model_config.multimodal_config.mm_device_do_normalize is True
    assert model_config.multimodal_config.language_model_only is False
    assert model_config.multimodal_config.get_limit_per_prompt("image") == 999
    assert model_config.multimodal_config.get_limit_per_prompt("video") == 999
    model_config.model_arch_config = model_config.get_model_arch_config()
    assert model_config.supports_multimodal_inputs
    return config, model_cls


@contextmanager
def _meta_default_device():
    # FusedMMInputNorm explicitly uses torch.get_default_device when moving its
    # three-channel CPU normalization constants. A torch.device context alone
    # is not a replacement for that process-wide default-device setting.
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.set_default_device(previous)


_WRAPPER_CASES = [
    pytest.param(
        "glm53_flash", tp_size, tp_rank, pp_size, pp_rank,
        id=f"glm-tp{tp_size}-r{tp_rank}-pp{pp_size}-r{pp_rank}",
    )
    for tp_size, tp_rank, pp_size, pp_rank in _GLM_TOPOLOGIES
] + [
    pytest.param("qwen38_flash_next", 2, rank, 1, 0, id=f"qwen-tp2-r{rank}")
    for rank in range(2)
]


@pytest.mark.parametrize("name,tp_size,tp_rank,pp_size,pp_rank", _WRAPPER_CASES)
def test_declared_checkpoint_architecture_with_vision_meta(
    monkeypatch, meta_platform, tmp_path, name, tp_size, tp_rank, pp_size, pp_rank
):
    from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.model_executor.models.utils import StageMissingLayer

    _parallel_groups(monkeypatch, tp_size, tp_rank, pp_size, pp_rank)
    config, model_cls = _wrapper_runtime_config(tmp_path, name, tp_size, pp_size)
    root_config = config.model_config.hf_config
    root_before = root_config.to_dict()
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with _meta_default_device():
            # Do not pass model_class: initialize_model must resolve the real
            # checkpoint-declared wrapper through the production registry path.
            model = initialize_model(vllm_config=config)
            boundary = model.make_empty_intermediate_tensors(
                2, torch.bfloat16, torch.device("meta")
            )

    assert type(model) is model_cls
    assert model.config is root_config
    assert model.config.text_config is config.model_config.hf_text_config
    assert root_config.to_dict() == root_before
    _assert_meta_model(model)
    assert boundary["hidden_states"].is_meta
    if pp_rank > 0:
        # Model Runner V2 runs the encoders on the first stage only, so later
        # stages build no vision tower.
        assert config.use_v2_model_runner
        assert isinstance(model.visual, StageMissingLayer)
        assert model.visual.stage_name == "vision_tower"
    else:
        assert not isinstance(model.visual, StageMissingLayer)
        _assert_meta_model(model.visual)
        assert isinstance(model.visual.input_norm, FusedMMInputNorm)
        assert model.visual.input_norm.channel == 3
        assert len(model.visual.blocks) == root_config.vision_config.depth
        assert model.visual.hidden_size == root_config.vision_config.hidden_size
        assert (
            model.visual.out_hidden_size == root_config.vision_config.out_hidden_size
        )
        assert model.visual.patch_size == root_config.vision_config.patch_size
        assert (
            model.visual.spatial_merge_size
            == root_config.vision_config.spatial_merge_size
        )
        assert type(model.visual).__name__ == {
            "glm53_flash": "Glm5NextVisionTransformer",
            "qwen38_flash_next": "Qwen3_VisionTransformer",
        }[name]
        assert len(model.visual.blocks) == (24 if name == "glm53_flash" else 27)

    backbone = model.language_model.model
    if name == "glm53_flash":
        expected_range = (0, 45) if pp_size == 1 else [
            (0, 11), (11, 22), (22, 34), (34, 45)
        ][pp_rank]
        assert (backbone.start_layer, backbone.end_layer) == expected_range
        assert backbone.hc_mult == root_config.text_config.hc_mult == 4
        assert boundary["hidden_states"].shape == (2, 4, 4096)
        assert model.num_moe_layers > 0
    else:
        assert model.language_model_only is False
        assert (backbone.start_layer, backbone.end_layer) == (0, 48)
        assert backbone.layers[1].ple is not None
        assert boundary["hidden_states"].shape == (2, 4 * 2560)
