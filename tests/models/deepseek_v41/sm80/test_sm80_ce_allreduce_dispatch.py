# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of VLLM_SM80_ALLREDUCE's copy-engine ring dispatch: which TP sums go to the
copy-engine ring (sm80_attention.prefill_all_reduce), and the MoE adoption under the switch.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_ce_allreduce_dispatch.py
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
from vllm_sm80.distributed import ce_ring_all_reduce as ce

from .test_sm80_attention_wiring import _piece_comm, _PieceCustomAllreduce

BF16 = torch.bfloat16


class _Ring:
    def __init__(self, takes):
        self._takes, self.calls = takes, []

    def takes(self, t):
        return self._takes

    def all_reduce(self, t):
        self.calls.append((t.dtype, tuple(t.shape)))
        return t * 4

    def round_once(self, t):
        self.calls.append(("round_once", t.dtype, tuple(t.shape)))
        return t * 5


@pytest.mark.parametrize(
    "case,ring_on,ring_takes,host,custom,to_ring",
    [
        ("prefill_message", True, True, False, False, True),
        ("switch_off", False, True, False, False, False),
        ("ring_declines", True, False, False, False, False),
        ("host_takes", True, True, True, False, False),
        ("custom_takes", True, True, False, True, False),
    ],
)
def test_prefill_all_reduce_dispatch(monkeypatch, case, ring_on, ring_takes, host, custom,
                                     to_ring) -> None:
    ring = _Ring(ring_takes)
    monkeypatch.setattr(ce, "_RING", ring if ring_on else None)
    ca = _PieceCustomAllreduce(1 << 30 if custom else 16)
    comm = _piece_comm(ca, host_takes=host)
    monkeypatch.setattr(sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm))
    nccl = []
    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", lambda t: nccl.append(t.dtype) or t * 3)
    t = torch.ones(8, 64)
    got = sm80.prefill_all_reduce(t)
    assert torch.equal(got, t * (4 if to_ring else 3))
    assert bool(ring.calls) == to_ring and bool(nccl) != to_ring


@pytest.mark.parametrize(
    "case,ring_on,custom,owner",
    [("ring_on", True, False, True), ("ring_off", False, False, False),
     ("custom_takes", True, True, False)],
)
def test_wo_b_bf16_large_path_uses_the_owner_sum(monkeypatch, case, ring_on, custom, owner) -> None:
    """With the ring built, a BF16 wo_b partial too large for the custom all-reduce whole goes to
    the ring's owner sum (round once), not the custom all-reduce in pieces."""
    ring = _Ring(True)
    monkeypatch.setattr(ce, "_RING", ring if ring_on else None)
    ca = _PieceCustomAllreduce(1 << 30 if custom else 1024)
    comm = _piece_comm(ca)
    monkeypatch.setattr(sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm))
    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", lambda t: t * 3)
    partial = torch.ones(40, 64, dtype=BF16)
    got = sm80.wo_b_all_reduce(partial, True)
    assert bool(ring.calls) == owner
    if owner:
        assert ring.calls == [("round_once", BF16, (40, 64))]
        assert torch.equal(got, partial * 5)
    elif not custom:
        assert len(ca.pieces) > 1 and torch.equal(got, (partial.float() * 4).to(BF16))


def test_wo_b_fp32_path_goes_through_prefill_all_reduce(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(sm80, "prefill_all_reduce", lambda t: seen.append(t.dtype) or t * 3)
    partial = torch.ones(4, 8, dtype=BF16)
    got = sm80.wo_b_all_reduce(partial, False)
    assert seen == [torch.float32] and got.dtype == BF16
    assert torch.equal(got, (partial.float() * 3).to(BF16))


def test_switch_adopts_the_layer_reduced_moe(monkeypatch) -> None:
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    calls = []
    monkeypatch.setattr(sm80_moe, "_adopt_layer_all_reduce",
                        lambda ffn, round_once: calls.append(round_once))
    for name in ("VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE",
                 "VLLM_SM80_PREFILL_REDUCE_OVERLAP", "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE", "1")
    ffn = SimpleNamespace(use_native_mega_moe=False,
                          gate=SimpleNamespace(tid2eid=None, e_score_correction_bias=object()),
                          scoring_func="sqrtsoftplus")
    monkeypatch.setattr(sm80_moe, "_fp32_gate_weight", lambda gate: None)
    router = sm80_moe.FusedTopKBiasRouter.__new__(sm80_moe.FusedTopKBiasRouter)
    router.num_fused_shared_experts = 0
    ffn.experts = SimpleNamespace(router=router)
    sm80_moe.adopt_moe(ffn, SimpleNamespace())
    assert calls == [False]


def test_allreduce_default_off(monkeypatch) -> None:
    import vllm.envs as envs

    name = "VLLM_SM80_ALLREDUCE"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
