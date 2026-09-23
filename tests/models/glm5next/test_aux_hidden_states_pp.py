# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glm5Next relays EAGLE3/DFlash auxiliary hidden states across pipeline
stages: a simulated N-stage run (stage-by-stage forward, the runner's relay of
upstream keys on middle stages, the receiver's reserved slots) must hand the
drafter the same aux tensors, in the same order, as a single-stage run.

CPU only. The decoder layers are stubs with the same deferred-mHC contract as
Glm5NextDecoderLayer: a layer returns (branch, streams, post, comb) and leaves
its hc_post to the next layer, the first layer on a stage takes materialised
streams, and the model's last layer contracts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.distributed.parallel_state as ps
import vllm.models.glm5next.common.model as glm_model
from vllm.models.glm5next.common.model import Glm5NextModel
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle import eagle3_utils

N_STREAMS = 4
HIDDEN = 16
TOKENS = 5
VOCAB = 32


class _StubLayer(nn.Module):
    def __init__(self, layer_idx: int, is_last: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_last = is_last

    @staticmethod
    def hc_post(branch, residual, post, comb):
        return residual * comb + branch[:, None, :] * post

    def forward(self, positions, hidden_states, residual, post, comb):
        if residual is None:
            if hidden_states.dim() == 2:  # layer 0: expand the embedding
                streams = hidden_states[:, None, :].expand(-1, N_STREAMS, -1)
                streams = streams.contiguous()
            else:  # first layer of a later stage: materialised streams
                streams = hidden_states
        else:  # fused path: apply the previous layer's deferred hc_post
            streams = self.hc_post(hidden_states, residual, post, comb)
        i = self.layer_idx
        branch = torch.tanh(streams.mean(1) * (1.0 + 0.1 * i)) + 0.01 * i
        p = torch.tensor(1.0 + 0.05 * i)
        c = torch.tensor(0.9 - 0.01 * i)
        if self.is_last:
            out = glm_model.hc_contract(self.hc_post(branch, streams, p, c), N_STREAMS)
            return out, None, None, None
        return branch, streams, p, c


class _FakePP:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1


def _set_pp(monkeypatch, rank: int, world_size: int) -> None:
    fake = _FakePP(rank, world_size)
    monkeypatch.setattr(glm_model, "get_pp_group", lambda: fake)
    monkeypatch.setattr(ps, "get_pp_group", lambda: fake)
    monkeypatch.setattr(ps, "model_parallel_is_initialized", lambda: True)


def _build_stage(embed, num_layers, start, end, aux_layers):
    m = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(m)
    m.config = SimpleNamespace(hidden_size=HIDDEN)
    m.mhc = True
    m.mhc_num_residual_streams = N_STREAMS
    m.is_sequence_parallel = False
    m.aux_hidden_state_mode = "stream_mean"
    m.embed_tokens = embed
    m.norm = nn.Identity()
    m.start_layer, m.end_layer = start, end
    m._active_layers = [
        _StubLayer(i, is_last=(i == num_layers - 1)) for i in range(start, end)
    ]
    m.make_empty_intermediate_tensors = m._make_empty_mhc_intermediate_tensors
    m._set_aux_hidden_state_layers(aux_layers)
    return m


def _run_single(monkeypatch, embed, num_layers, aux_layers, input_ids, positions):
    _set_pp(monkeypatch, 0, 1)
    m = _build_stage(embed, num_layers, 0, num_layers, aux_layers)
    with torch.no_grad():
        return m(input_ids, positions, None)


def _run_pipeline(
    monkeypatch, embed, num_layers, partition, aux_layers, input_ids, positions
):
    world = len(partition)
    bounds = [sum(partition[:r]) for r in range(world + 1)]
    received = None
    for rank in range(world):
        _set_pp(monkeypatch, rank, world)
        m = _build_stage(embed, num_layers, bounds[rank], bounds[rank + 1], aux_layers)
        wrapper = SimpleNamespace(
            model=m, make_empty_intermediate_tensors=m.make_empty_intermediate_tensors
        )
        if rank > 0:
            # The receiver's persistent buffer (reserve_aux_intermediate_tensor_slots)
            # must name exactly the keys the sender produced: the runner copies
            # every buffer key out of the received tensors.
            eagle3_utils.reserve_aux_intermediate_tensor_slots(wrapper)
            buf = wrapper.make_empty_intermediate_tensors(TOKENS, torch.float32, "cpu")
            assert set(buf.tensors) == set(received.tensors)
            for k, v in buf.tensors.items():
                assert v.shape == received[k].shape, k
        relay_keys = eagle3_utils.aux_hidden_state_relay_keys(wrapper)
        with torch.no_grad():
            out = m(input_ids, positions, received)
        if rank == world - 1:
            return out
        assert isinstance(out, IntermediateTensors)
        # PPHandler.relay_aux_hidden_states on middle stages
        if relay_keys:
            out = IntermediateTensors(
                out.tensors | {k: received[k] for k in relay_keys}
            )
        received = out
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "num_layers,partition,aux_layers",
    [
        # aux ids on a stage boundary (3, 5, 8), several on one stage, and a
        # last stage with none of its own
        (10, [3, 2, 3, 2], (2, 3, 5, 7, 8)),
        # GLM-5.3-Flash + DFlash2: target_layer_ids 5,14,24,33,42 (+1)
        (45, [12, 11, 11, 11], (6, 15, 25, 34, 43)),
        (45, [13, 11, 11, 10], (6, 15, 25, 34, 43)),
        # a middle stage with no aux layer of its own
        (12, [4, 4, 2, 2], (1, 2, 11)),
        (8, [4, 4], (4, 7)),
    ],
)
def test_aux_hidden_states_match_single_stage(
    monkeypatch, num_layers, partition, aux_layers
):
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (TOKENS,))
    positions = torch.arange(TOKENS)

    ref_hidden, ref_aux = _run_single(
        monkeypatch, embed, num_layers, aux_layers, input_ids, positions
    )
    pp_hidden, pp_aux = _run_pipeline(
        monkeypatch, embed, num_layers, partition, aux_layers, input_ids, positions
    )

    assert len(ref_aux) == len(aux_layers)
    assert len(pp_aux) == len(ref_aux)
    torch.testing.assert_close(pp_hidden, ref_hidden, rtol=0, atol=1e-6)
    for got, want in zip(pp_aux, ref_aux):
        assert got.shape == (TOKENS, HIDDEN)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_glm5next_declares_aux_relay_support():
    inner = Glm5NextModel.__new__(Glm5NextModel)
    eagle3_utils.verify_supports_aux_hidden_states_over_pp(
        SimpleNamespace(model=inner), "dflash"
    )


def test_no_aux_layers_leaves_boundary_unchanged(monkeypatch):
    """Without a drafter the boundary carries only the mHC streams."""
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    _set_pp(monkeypatch, 0, 2)
    m = _build_stage(embed, 4, 0, 2, ())
    with torch.no_grad():
        out = m(torch.randint(0, VOCAB, (TOKENS,)), torch.arange(TOKENS), None)
    assert isinstance(out, IntermediateTensors)
    assert set(out.tensors) == {"hidden_states"}
    assert out["hidden_states"].shape == (TOKENS, N_STREAMS, HIDDEN)
