# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm80 Triton sparse MLA backend reads the cache as flat rows, so packed
KV blocks must sit a whole number of rows apart.

With a block-outermost layout every layer's page shares one physical block,
and the block stride is rounded only to the alignments the backends publish
through ``customize_spec``. The cache view must then pass
``flat_kv_row_view``. The geometry is GLM-5.3-Flash's TP4 rank-0 target, built
on meta from the deployed checkpoint config, with the hybrid block geometry
the platform sets at engine start for it (1152-token blocks, KDA pages padded
to the attention page).

Run in a fresh process with CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1.
"""
import pytest
import torch

from tests.models.test_native_checkpoint_config_meta import (  # noqa: F401
    _parallel_groups,
    _registered_model_cls,
    _vllm_config,
    meta_platform,
)
from vllm.config import set_current_vllm_config
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# Platform._align_hybrid_block_size for this model: the smallest multiple of
# 128 tokens whose MLA page holds a KDA state.
_BLOCK_SIZE = 1152
_HEAD_DIM = 512  # the latent row GLM-5.3-Flash caches per token
_ROW_BYTES = _HEAD_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()


def _served_glm_tp4_specs(monkeypatch):
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    _parallel_groups(monkeypatch, 4, 0)
    config = _vllm_config("glm53_flash", 4, with_dflash=True)
    cache = config.cache_config
    cache.kv_cache_layout = "BLHNC"
    cache.mamba_cache_mode = "align"
    cache.block_size = cache.mamba_block_size = _BLOCK_SIZE
    cache.mamba_page_size_padded = _BLOCK_SIZE * _ROW_BYTES
    Glm5NextForCausalLM = _registered_model_cls("Glm5NextForCausalLM")
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            Glm5NextForCausalLM(vllm_config=config)
        specs = get_kv_cache_spec(config)
    return config, specs


def _block_stride_bytes(config, specs):
    from vllm.v1.core import kv_cache_planning

    groups = kv_cache_planning.get_kv_cache_groups(config, specs)
    layout = config.cache_config.get_resolved_kv_cache_layout()
    assert layout.is_block_outermost
    return kv_cache_planning._get_kv_cache_bytes_per_block(groups, layout)


def _mla_specs(specs):
    mla = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, MLAAttentionSpec) and spec.dtype == torch.bfloat16
    }
    assert mla, "no MLA attention layer in the served GLM target"
    assert {spec.head_size for spec in mla.values()} == {_HEAD_DIM}
    assert {spec.block_size for spec in mla.values()} == {_BLOCK_SIZE}
    return mla


def _view_of_packed_cache(stride_bytes):
    """A two-block cache view with the packed block stride, as the runner binds
    it to an MLA layer (bf16 latent rows)."""
    elem = torch.tensor([], dtype=torch.bfloat16).element_size()
    assert stride_bytes % elem == 0
    stride = stride_bytes // elem
    storage = torch.zeros(stride + _BLOCK_SIZE * _HEAD_DIM, dtype=torch.bfloat16)
    return storage.as_strided((2, _BLOCK_SIZE, _HEAD_DIM), (stride, _HEAD_DIM, 1))


def test_served_glm_mla_layers_ask_for_whole_row_blocks(monkeypatch, meta_platform):
    from vllm_sm80.attention.triton_mla_sparse import TritonMLASparseBackend

    config, specs = _served_glm_tp4_specs(monkeypatch)
    from vllm.v1.worker.gpu.attn_utils import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(config, AttentionLayerBase)
    for name, spec in _mla_specs(specs).items():
        assert layers[name].get_attn_backend() is TritonMLASparseBackend
        assert spec.get_block_stride_alignment() == _ROW_BYTES


def test_packed_block_stride_is_whole_rows(monkeypatch, meta_platform):
    config, specs = _served_glm_tp4_specs(monkeypatch)
    _mla_specs(specs)
    stride = _block_stride_bytes(config, specs)
    assert stride % _ROW_BYTES == 0
    rows, stride_rows = flat_kv_row_view(_view_of_packed_cache(stride), _BLOCK_SIZE)
    assert stride_rows == stride // _ROW_BYTES
    assert rows.shape == (stride_rows + _BLOCK_SIZE, _HEAD_DIM)


def test_without_the_alignment_the_packed_stride_misaddresses(
    monkeypatch, meta_platform
):
    """The served failure: without ``customize_spec`` the packed stride is not
    a whole number of rows and ``flat_kv_row_view`` refuses the cache."""
    from vllm_sm80.attention.triton_mla_sparse import TritonMLASparseBackend

    monkeypatch.setattr(
        TritonMLASparseBackend,
        "customize_spec",
        classmethod(AttentionBackend.customize_spec.__func__),
    )
    config, specs = _served_glm_tp4_specs(monkeypatch)
    _mla_specs(specs)
    stride = _block_stride_bytes(config, specs)
    assert stride % _ROW_BYTES != 0
    with pytest.raises(AssertionError, match="not a whole number of rows"):
        flat_kv_row_view(_view_of_packed_cache(stride), _BLOCK_SIZE)


def test_qwen_layers_do_not_use_the_sm80_triton_mla_backend(
    monkeypatch, meta_platform
):
    """Qwen3.8-Flash-Next's caches (QSA included) are served by other
    backends, so the alignment above does not change its KV layout."""
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM
    from vllm_sm80.attention.triton_mla_sparse import TritonMLASparseBackend
    from vllm.v1.worker.gpu.attn_utils import get_layers_from_vllm_config

    _parallel_groups(monkeypatch, 2, 0)
    config = _vllm_config("qwen38_flash_next", 2)
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device("meta"):
            Qwen4ExpForCausalLM(vllm_config=config)
    layers = get_layers_from_vllm_config(config, AttentionLayerBase)
    assert layers
    assert not [
        name
        for name, layer in layers.items()
        if layer.get_attn_backend() is TritonMLASparseBackend
    ]
