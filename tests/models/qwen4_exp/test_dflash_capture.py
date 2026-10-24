# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DFlash / EAGLE-3 capture of Qwen4ExpModel's auxiliary hidden states."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel

HC, H, T = 4, 8, 3


def _model() -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(hc_count=HC, hidden_size=H))


def test_aux_hidden_state_is_the_stream_mean() -> None:
    """With no pending combine the capture is the mean over the HC streams."""
    streams = torch.randn(T, HC * H)
    aux = Qwen4ExpModel._aux_hidden_state(_model(), streams, None, None, None)
    assert aux.shape == (T, H)
    torch.testing.assert_close(aux, streams.view(T, HC, H).mean(dim=1))


def test_aux_hidden_state_completes_a_pending_combine_out_of_place() -> None:
    """A pending combine is materialised with the previous layer's HC module,
    and the main stream handed to the next layer is left untouched."""
    streams = torch.randn(T, HC * H)
    block_output = torch.randn(T, H)
    injection = torch.randn(T, HC)
    combined = torch.randn(T, HC * H)
    layer = MagicMock()
    layer.mlp_hyper_connection.combine.return_value = combined
    before = streams.clone()

    aux = Qwen4ExpModel._aux_hidden_state(
        _model(), streams, block_output, injection, layer
    )

    layer.mlp_hyper_connection.combine.assert_called_once_with(
        streams, block_output, injection
    )
    torch.testing.assert_close(aux, combined.view(T, HC, H).mean(dim=1))
    torch.testing.assert_close(streams, before)
