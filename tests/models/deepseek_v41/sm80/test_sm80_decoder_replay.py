# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder replay's KV write runs the sm_80 layer's ``write_kv`` (the
reference mHC), not the upstream one it was bound to before adoption."""

from types import SimpleNamespace

import torch.nn as nn

from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.models.deepseek_v41.nvidia import model as upstream_model
from vllm_sm80.models.deepseek_v41 import sm80_model


def _bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def test_adopt_rebinds_the_replay_kv_write(monkeypatch) -> None:
    monkeypatch.setattr(sm80_model, "_adopt_layers", lambda layers, config: None)
    model = _bare(upstream_model.DeepseekV4Model)
    layers = [_bare(sm80_model.DeepseekV4DecoderLayer) for _ in range(3)]
    model.layers = layers
    model.start_layer, model.end_layer = 0, 3
    model.config = SimpleNamespace()
    model.use_sequence_parallel = model.fuse_mhc_all_reduce = False
    model.decoder_replay_start = 1
    upstream_write = upstream_model.DeepseekV4DecoderLayer.write_kv.__get__(layers[1])
    model.decoder_replay_layers = DecoderReplayLayers(
        128, lambda *a: a, upstream_write, set(), "swa"
    )
    sm80_model.DeepseekV4Model.adopt(model)
    write = model.decoder_replay_layers.write_batch_kv
    assert write.__self__ is layers[1]
    assert write.__func__ is sm80_model.DeepseekV4DecoderLayer.write_kv
