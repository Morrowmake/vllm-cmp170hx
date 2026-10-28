# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV cache plan of GLM-5.3-Flash with its DFlash2 drafter at PP4, per
stage, as the engine builds it.

Each stage's layers come from the real target built on meta from the deployed
checkpoint config (TP1, PP4, the default layer partition); the drafter is
built on the last stage, where it runs. The hybrid block geometry is the one
the platform sets at engine start for this layout (4608-token blocks, KDA
pages padded to the attention page). The plan must give every worker cache
tensors for exactly its own layers, each owned by one of its groups (a worker
allocates a tensor only for a layer of one of its groups).

Run in a fresh process with CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1.
"""
import torch

from tests.models.test_native_checkpoint_config_meta import (  # noqa: F401
    _parallel_groups,
    _registered_model_cls,
    _vllm_config,
    meta_platform,
)
from vllm.config import set_current_vllm_config
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import SlidingWindowSpec

_PP = 4
_BLOCK_SIZE = 4608
_MLA_ROW_BYTES = 1024


def _stage(monkeypatch, stage):
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm_sm80 import plugin
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    _parallel_groups(monkeypatch, 1, 0, _PP, stage)
    config = _vllm_config("glm53_flash", 1, _PP, with_dflash=True)
    cache = config.cache_config
    cache.kv_cache_layout = "BLHNC"
    cache.mamba_cache_mode = "align"
    cache.block_size = cache.mamba_block_size = _BLOCK_SIZE
    cache.mamba_page_size_padded = _BLOCK_SIZE * _MLA_ROW_BYTES
    target_cls = _registered_model_cls("Glm5NextForCausalLM")
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            target_cls(vllm_config=config)
            if stage == _PP - 1:
                plugin.register()
                draft_cls = ModelRegistry._try_load_model_cls("DFlash2DraftModel")
                draft_cls(vllm_config=config, prefix="draft_model")
        specs = get_kv_cache_spec(config)
    return config, specs


def test_every_pp4_stage_gets_tensors_for_exactly_its_layers(
    monkeypatch, meta_platform
):
    from tests.v1.core.utils import get_kv_cache_configs

    workers = []
    config = None
    for stage in range(_PP):
        with monkeypatch.context() as patch:
            stage_config, specs = _stage(patch, stage)
        config = config or stage_config
        workers.append(specs)
    drafter = {
        name for name, spec in workers[-1].items() if type(spec) is SlidingWindowSpec
    }
    assert len(drafter) == 5
    assert not any(type(s) is SlidingWindowSpec for w in workers[:-1] for s in w.values())

    configs = get_kv_cache_configs(config, workers, [16 << 30] * _PP)

    assert len({c.num_blocks for c in configs}) == 1
    for specs, kv_cache_config in zip(workers, configs):
        grouped = [n for g in kv_cache_config.kv_cache_groups for n in g.layer_names]
        tensors = [n for t in kv_cache_config.kv_cache_tensors for n in t.layers]
        assert sorted(grouped) == sorted(specs)
        assert sorted(tensors) == sorted(specs)
    draft_groups = [
        g
        for g in configs[-1].kv_cache_groups
        if g.layer_names and g.layer_names[0] in drafter
    ]
    assert draft_groups
    assert all(type(g.kv_cache_spec) is SlidingWindowSpec for g in draft_groups)
