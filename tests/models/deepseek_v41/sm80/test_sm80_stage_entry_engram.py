# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pipeline stage that starts on an Engram layer injects the Engram into
the hc stream it receives before the sm_80 layer's mHC pre, as the upstream
layer does."""

from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.models.deepseek_v41 import sm80_model


class _StopAfterMhcPre(Exception):
    pass


def test_stage_entry_injects_engram_before_mhc_pre() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls = []
    mhc_pre_calls = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre_delayed(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise _StopAfterMhcPre

    layer = SimpleNamespace(
        _mhc_ops=SimpleNamespace(mhc_pre_delayed=mhc_pre_delayed),
        _mixes=lambda attn: (),
        engram=Engram(),
        hc_mult=4,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )

    with pytest.raises(_StopAfterMhcPre):
        sm80_model.DeepseekV4DecoderLayer._forward_layer(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix
