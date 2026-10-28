# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for VLLM_GLM5_DRAFTER_ROPE_FIT.

The DFlash drafter builds its RoPE cos/sin cache for the draft config's
max_position_embeddings. With the flag on, the cache is cut to the positions
the draft can reach (max_model_len + 1 + num_speculative_tokens, rounded up to
64). Rows are indexed by position, so every kept row must be bit-identical to
the unfitted cache, and the target's caches must not be touched.

No GPU and no weights.

    CUDA_VISIBLE_DEVICES= pytest -q tests/models/glm5next/test_drafter_rope_fit.py
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope
from vllm.model_executor.models.qwen3_dflash import dflash_rope_max_position

FLAG = "VLLM_GLM5_DRAFTER_ROPE_FIT"
PLAIN_ROPE = {"rope_theta": 10000.0, "rope_type": "default"}


def _configs(
    configured: int,
    max_model_len: int,
    num_spec: int | None,
    rope_parameters: dict | None = None,
    target_max_model_len: int | None = None,
):
    draft = SimpleNamespace(
        max_position_embeddings=configured,
        rope_parameters=dict(rope_parameters or PLAIN_ROPE),
    )
    speculative_config = None
    if num_spec is not None:
        target = (
            SimpleNamespace(max_model_len=target_max_model_len)
            if target_max_model_len is not None
            else None
        )
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_spec, target_model_config=target
        )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        speculative_config=speculative_config,
    )
    return vllm_config, draft


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert envs.VLLM_GLM5_DRAFTER_ROPE_FIT is False
    vllm_config, draft = _configs(1 << 20, 262_144, 3)
    assert dflash_rope_max_position(vllm_config, draft) == 1 << 20


def test_flag_zero_is_off(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    vllm_config, draft = _configs(1 << 20, 262_144, 3)
    assert dflash_rope_max_position(vllm_config, draft) == 1 << 20


@pytest.mark.parametrize(
    "max_model_len,num_spec,expected",
    [
        # Production: 262,144 + 1 + 3 -> 262,208 (next multiple of 64).
        (262_144, 3, 262_208),
        (262_144, 7, 262_208),
        (262_144, 63, 262_208),
        (262_144, 64, 262_272),
        (131_072, 3, 131_136),
        (1_000, 3, 1_024),
        (1_020, 3, 1_024),
        (1_021, 3, 1_088),
    ],
)
def test_fitted_length(flag_on, max_model_len, num_spec, expected):
    vllm_config, draft = _configs(1 << 20, max_model_len, num_spec)
    fitted = dflash_rope_max_position(vllm_config, draft)
    assert fitted == expected
    # Margin covers the largest unclamped query index: max_model_len - 1 as
    # the last context position, + 1 + num_spec as the last query offset.
    assert fitted >= max_model_len + num_spec + 1


def test_never_grows_past_configured(flag_on):
    vllm_config, draft = _configs(4_096, 1_000_000, 3)
    assert dflash_rope_max_position(vllm_config, draft) == 4_096


def test_uses_larger_of_draft_and_target_len(flag_on):
    vllm_config, draft = _configs(1 << 20, 4_096, 3, target_max_model_len=262_144)
    assert dflash_rope_max_position(vllm_config, draft) == 262_208


def test_no_speculative_config(flag_on):
    vllm_config, draft = _configs(1 << 20, 1_000, None)
    assert dflash_rope_max_position(vllm_config, draft) == 1_024


@pytest.mark.parametrize(
    "rope_parameters",
    [
        {"rope_theta": 10000.0, "rope_type": "yarn", "factor": 4.0},
        {"rope_theta": 10000.0, "rope_type": "linear", "factor": 2.0},
        {"rope_theta": 10000.0, "rope_type": "default", "mrope_section": [16, 24, 24]},
    ],
)
def test_scaled_rope_keeps_configured(flag_on, rope_parameters):
    vllm_config, draft = _configs(1 << 20, 262_144, 3, rope_parameters)
    assert dflash_rope_max_position(vllm_config, draft) == 1 << 20


@pytest.mark.parametrize("is_neox_style", [True, False])
def test_kept_rows_identical(flag_on, default_vllm_config, is_neox_style):
    head_size = 128
    configured = 8_192
    max_model_len, num_spec = 2_000, 3
    vllm_config, draft = _configs(configured, max_model_len, num_spec)
    fitted = dflash_rope_max_position(vllm_config, draft)
    assert fitted == 2_048

    full = get_rope(
        head_size,
        max_position=configured,
        is_neox_style=is_neox_style,
        rope_parameters=PLAIN_ROPE,
        dtype=torch.float32,
    )
    full_cache_before = full.cos_sin_cache.clone()
    small = get_rope(
        head_size,
        max_position=fitted,
        is_neox_style=is_neox_style,
        rope_parameters=PLAIN_ROPE,
        dtype=torch.float32,
    )

    # Separate cache entries: fitting the draft never resizes a cache that
    # another module (the target) got from get_rope with the configured size.
    assert small is not full
    assert small.cos_sin_cache.shape == (fitted, head_size)
    assert full.cos_sin_cache.shape == (configured, head_size)
    assert torch.equal(full.cos_sin_cache, full_cache_before)

    # Row-for-row identical over every kept position.
    assert torch.equal(small.cos_sin_cache, full.cos_sin_cache[:fitted])

    # Same rotation for every reachable position, including the unclamped
    # query tail past max_model_len.
    torch.manual_seed(0)
    positions = torch.cat(
        [
            torch.arange(0, 64),
            torch.arange(max_model_len - 64, max_model_len + num_spec + 1),
        ]
    )
    q = torch.randn(positions.numel(), 4 * head_size)
    k = torch.randn(positions.numel(), 1 * head_size)
    q_full, k_full = full.forward_native(positions, q.clone(), k.clone())
    q_small, k_small = small.forward_native(positions, q.clone(), k.clone())
    assert torch.equal(q_full, q_small)
    assert torch.equal(k_full, k_small)


def test_bf16_rows_identical_at_production_size(flag_on, default_vllm_config):
    """Production shape: 1,048,576 -> 262,208 rows of 128 bf16 values."""
    configured, head_size = 1 << 20, 128
    vllm_config, draft = _configs(configured, 262_144, 3)
    fitted = dflash_rope_max_position(vllm_config, draft)
    full = get_rope(
        head_size, configured, True, PLAIN_ROPE, dtype=torch.bfloat16
    ).cos_sin_cache
    small = get_rope(
        head_size, fitted, True, PLAIN_ROPE, dtype=torch.bfloat16
    ).cos_sin_cache
    assert torch.equal(small, full[:fitted])
    freed = (full.numel() - small.numel()) * full.element_size()
    assert full.numel() * full.element_size() == 256 * 2**20
    assert freed == (configured - 262_208) * head_size * 2  # 191.98 MiB
    # Drop the large entries so later tests in the session do not keep them.
    for key in [
        k
        for k in _ROPE_DICT
        if k[2] in (configured, fitted) and k[-1] == torch.bfloat16
    ]:
        del _ROPE_DICT[key]
