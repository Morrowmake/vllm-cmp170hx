# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE: the adopted MoE
all-reduces its own output where the runner would, through the custom
all-reduce in pieces when the message is too large for it whole.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_moe_allreduce.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v41.nvidia import model as dsv41_model
from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
from vllm_sm80.models.deepseek_v41 import sm80_moe

from .test_sm80_attention_wiring import _MAX, _piece_comm, _PieceCustomAllreduce

BF16 = torch.bfloat16


def _comm(monkeypatch, ca, **kw):
    comm = _piece_comm(ca, **kw)
    monkeypatch.setattr(
        sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )
    seen = []

    def all_reduce(t):
        seen.append(t.dtype)
        return t * 4

    monkeypatch.setattr(sm80_moe, "tensor_model_parallel_all_reduce", all_reduce)
    return seen


@pytest.mark.parametrize(
    "case,rows,in_pieces",
    [
        ("too_large", 40, True),
        ("whole_fits", 7, False),
        ("host_takes", 40, False),
        ("not_fully_connected", 40, False),
    ],
)
def test_moe_all_reduce_pieces_or_runner_path(monkeypatch, case, rows, in_pieces):
    """A BF16 MoE output too large for the custom all-reduce whole goes through
    it in pieces (FP32 sum rounded once); any other message keeps the runner's
    all-reduce, in BF16 as before."""
    ca = _PieceCustomAllreduce(_MAX, fully_connected=case != "not_fully_connected")
    seen = _comm(monkeypatch, ca, host_takes=case == "host_takes")
    g = torch.Generator().manual_seed(4)
    out = torch.randn(rows, 64, generator=g).to(BF16)
    got = sm80_moe.moe_all_reduce(out)
    assert got.dtype == BF16 and got.shape == out.shape
    if in_pieces:
        assert not seen and len(ca.pieces) > 1
        assert torch.equal(got, (out.float() * 4).to(BF16))
    else:
        assert seen == [BF16] and not ca.pieces
        assert torch.equal(got, out * 4)


def _bare_moe(tp_size=4, ep_size=4, sequence_parallel=False, skip=False):
    ffn = object.__new__(dsv41_model.DeepseekV4MoE)
    nn.Module.__init__(ffn)
    moe_config = SimpleNamespace(
        is_sequence_parallel=sequence_parallel,
        skip_final_all_reduce=skip,
        tp_size=tp_size,
        ep_size=ep_size,
    )
    object.__setattr__(ffn, "experts", SimpleNamespace(moe_config=moe_config))
    return ffn


@pytest.mark.parametrize("round_once", [True, False])
def test_layer_all_reduce_moves_the_final_all_reduce_into_the_layer(
    monkeypatch, round_once
) -> None:
    """The adopted MoE's forward is its reduce of its forward_partial: the
    round-once all-reduce with VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE,
    else the runner's own all-reduce (VLLM_SM80_PREFILL_REDUCE_OVERLAP or
    VLLM_SM80_ALLREDUCE alone)."""
    ffn = _bare_moe()
    sm80_moe._adopt_layer_all_reduce(ffn, round_once=round_once)
    assert type(ffn) is sm80_moe.Sm80DeepseekV41MoE
    assert ffn.experts.moe_config.skip_final_all_reduce is True

    calls = []

    def upstream_forward(self, hidden_states, input_ids=None, mega=None):
        calls.append((hidden_states.shape, input_ids, mega))
        return hidden_states + 1

    monkeypatch.setattr(dsv41_model.DeepseekV4MoE, "forward", upstream_forward)
    monkeypatch.setattr(sm80_moe, "moe_all_reduce", lambda t: t * 2)
    monkeypatch.setattr(sm80_moe, "prefill_all_reduce", lambda t: t * 3)
    x = torch.ones(3, 8, dtype=BF16)
    ids = torch.arange(3)
    got = ffn(x, ids, "meta")
    assert calls == [((3, 8), ids, "meta")]
    assert torch.equal(got, (x + 1) * (2 if round_once else 3))
    assert torch.equal(ffn.forward_partial(x, ids), x + 1)
    assert torch.equal(ffn.reduce(x), x * (2 if round_once else 3))


def test_layer_all_reduce_leaves_single_rank_layers(monkeypatch) -> None:
    ffn = _bare_moe(tp_size=1, ep_size=1)
    sm80_moe._adopt_layer_all_reduce(ffn, round_once=True)
    assert type(ffn) is dsv41_model.DeepseekV4MoE
    assert ffn.experts.moe_config.skip_final_all_reduce is False


@pytest.mark.parametrize("kw", [{"sequence_parallel": True}, {"skip": True}])
def test_layer_all_reduce_refuses_layers_the_runner_does_not_reduce(kw) -> None:
    with pytest.raises(ValueError):
        sm80_moe._adopt_layer_all_reduce(_bare_moe(**kw), round_once=True)


def test_layer_all_reduce_refuses_other_moe_classes() -> None:
    class Other(dsv41_model.DeepseekV4MoE):
        pass

    ffn = _bare_moe()
    ffn.__class__ = Other
    with pytest.raises(TypeError):
        sm80_moe._adopt_layer_all_reduce(ffn, round_once=False)


def test_moe_round_once_allreduce_default_off(monkeypatch) -> None:
    import vllm.envs as envs

    name = "VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
