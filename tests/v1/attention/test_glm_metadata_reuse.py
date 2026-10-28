# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of GLM graph/dummy and variable-draft metadata across groups.

Use byte-exact checkpoint configs, the real GDN builders of the GLM KDA
layers, Mamba hybrid kwargs and MRV2 dispatch. Only CUDA allocation is
redirected to CPU; no builder, metadata, recovery wrapper or reuse
implementation is mocked.
"""

import pytest
import torch

from tests.models.test_native_checkpoint_config_meta import (
    _vllm_config,
    meta_platform,
)
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


_SHAPES = [
    pytest.param([1] * 8, True, id="capture-zero-draft"),
    pytest.param([4] * 8, True, id="capture-short-window"),
    pytest.param([8] * 8, True, id="capture-max-window"),
    pytest.param([4, 4, 0, 0], True, id="capture-padded"),
    pytest.param([4] * 8, False, id="dummy-warmup"),
    pytest.param([8, 4, 1, 0], False, id="variable-draft-padded"),
    pytest.param([1, 1, 0, 0], False, id="zero-draft-padded"),
]


def _glm_kda_backend():
    """The attention backend of the GLM KDA layers as a serving process builds
    them: the fork plugin's registrations run and the GLM architecture is
    resolved before any layer exists."""
    from vllm.model_executor.custom_op import maybe_get_oot_by_class
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm_sm80 import plugin

    plugin.register()
    assert ModelRegistry._try_load_model_cls("Glm5NextForCausalLM") is not None
    layer = object.__new__(maybe_get_oot_by_class(Glm5NextLinearAttention))
    return layer.get_attn_backend()


@pytest.fixture
def cpu_cuda_allocations(monkeypatch, meta_platform):
    """Keep CUDA builder admission checks while forbidding GPU initialization."""
    from vllm_sm80.kernels import _platform
    from vllm.utils import torch_utils

    monkeypatch.setattr(_platform, "_SM80_CACHE", True)
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    for name in ("empty", "ones"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)


@pytest.mark.parametrize(
    "tp,pp,kernels,recover_off",
    [
        (4, 1, False, None),
        (4, 1, True, None),
        (4, 1, True, "0"),
        (4, 1, True, "1"),
        (1, 4, True, None),
    ],
    ids=[
        "tp4-16-head-kernels-off",
        "tp4-16-head-recover",
        "tp4-16-head-recover-off-0",
        "tp4-16-head-recover-off-1",
        "pp4-64-head",
    ],
)
@pytest.mark.parametrize("cache_mode", ["none", "align"])
@pytest.mark.parametrize("query_lens,capture", _SHAPES)
def test_glm_metadata_group_ownership(
    monkeypatch,
    cpu_cuda_allocations,
    tp,
    pp,
    kernels,
    recover_off,
    cache_mode,
    query_lens,
    capture,
):
    # RecoverSSM follows the decode kernels at 16 KDA heads per card, unless
    # the diagnostic VLLM_AMPERE_KDA_RECOVER_OFF=1 keeps the per-draft-position
    # states there (the builders then match the kernels-off case).
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", str(int(kernels)))
    if recover_off is None:
        monkeypatch.delenv("VLLM_AMPERE_KDA_RECOVER_OFF", raising=False)
    else:
        monkeypatch.setenv("VLLM_AMPERE_KDA_RECOVER_OFF", recover_off)
    recover = kernels and tp == 4 and recover_off != "1"
    config = _vllm_config("glm53_flash", tp, pp, with_dflash=True)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.compilation_config.max_cudagraph_capture_size = 64
    config.cache_config.mamba_cache_mode = cache_mode
    assert config.use_v2_model_runner
    text = config.model_config.hf_text_config
    assert text.linear_num_heads // tp == (16 if tp == 4 else 64)
    num_spec = config.num_speculative_tokens
    spec = MambaSpec(
        block_size=16,
        shapes=MambaStateShapeCalculator.kda_state_shape(
            tp_world_size=tp,
            num_heads=text.linear_num_heads,
            head_dim=text.linear_head_dim,
            num_spec=num_spec,
        ),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=cache_mode,
        num_speculative_blocks=0 if recover else num_spec,
    )
    backend = _glm_kda_backend()
    builder_cls = backend.get_builder_cls()
    builders = [
        builder_cls(spec, [f"layer.{i}"], config, torch.device("cuda"))
        for i in range(3)
    ]
    assert all((b._kda_recover is not None) == recover for b in builders)
    batch = BatchSpec(seq_lens=[64] * len(query_lens), query_lens=query_lens)
    common = create_common_attn_metadata(batch, 16, torch.device("cpu"))
    # Tables must hold the current aligned state and all speculative columns.
    tables = [
        torch.arange(batch.batch_size * 16, dtype=torch.int32).reshape(
            batch.batch_size, 16
        ) + 1000 * i
        for i in range(3)
    ]
    accepted = torch.tensor(
        [min(i + 1, num_spec + 1) for i in range(batch.batch_size)],
        dtype=torch.int32,
    )
    drafts = torch.tensor([q - 1 for q in query_lens], dtype=torch.int32)
    model_meta = MambaHybridAttnMetadata(
        is_prefilling=torch.zeros(batch.batch_size, dtype=torch.bool),
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=drafts,
    )
    groups = [
        AttentionGroup(backend, b.layer_names, spec, i, metadata_builders=[b])
        for i, b in enumerate(builders)
    ]
    if cache_mode == "align":
        for b, table in zip(builders, tables):
            if hasattr(b, "mamba_aligned_state_indices"):
                b.mamba_aligned_state_indices = mamba_get_block_table_tensor(
                    table, common.seq_lens, spec, cache_mode
                )
    kv_config = KVCacheConfig(
        10000, [], [KVCacheGroupSpec(g.layer_names, spec) for g in groups]
    )
    actual = build_attn_metadata(
        attn_groups=[[g] for g in groups],
        num_reqs=batch.batch_size,
        num_tokens=common.num_actual_tokens,
        query_start_loc_gpu=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        max_query_len=common.max_query_len,
        seq_lens=common.seq_lens,
        max_seq_len=common.max_seq_len,
        block_tables=tables,
        slot_mappings=[common.slot_mapping] * 3,
        kv_cache_config=kv_config,
        seq_lens_cpu_upper_bound=common.seq_lens_cpu_upper_bound,
        model_specific_attn_metadata=model_meta,
        for_cudagraph_capture=capture,
    )
    for b, table in zip(builders, tables):
        meta = actual[b.layer_names[0]]
        assert meta.spec_sequence_masks_cpu is not None
        torch.testing.assert_close(meta.spec_sequence_masks_cpu, drafts >= 0)
        # Compare against an independent real eager build for this group.
        ref = builder_cls(spec, b.layer_names, config, torch.device("cuda"))
        if cache_mode == "align" and hasattr(ref, "mamba_aligned_state_indices"):
            ref.mamba_aligned_state_indices = mamba_get_block_table_tensor(
                table, common.seq_lens, spec, cache_mode
            )
        ref_common = common.replace(block_table_tensor=table)
        expected = (
            ref.build_for_cudagraph_capture(ref_common)
            if capture else ref.build(0, ref_common, accepted, drafts)
        )
        for field in (
            "num_decodes", "num_prefills", "num_spec_decodes", "num_spec_decode_tokens",
            "spec_state_indices_tensor", "non_spec_state_indices_tensor",
            "spec_query_start_loc", "spec_sequence_masks", "spec_sequence_masks_cpu",
            "spec_token_indx", "non_spec_token_indx", "num_accepted_tokens",
        ):
            got, want = getattr(meta, field), getattr(expected, field)
            torch.testing.assert_close(got, want, msg=field)
        if recover:
            assert not b.supports_update_block_table
            assert meta.recover_context is b._kda_recover
            commit_table = table if cache_mode == "align" else None
            assert meta.recover_commit.block_table is commit_table
            torch.testing.assert_close(
                meta.recover_commit.state_indices,
                expected.recover_commit.state_indices,
            )
        else:
            assert b.supports_update_block_table
            assert (
                meta.spec_query_start_loc.data_ptr()
                == actual["layer.0"].spec_query_start_loc.data_ptr()
            )
    assert not torch.cuda.is_initialized()
