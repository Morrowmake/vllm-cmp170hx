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


def _adopt_moe_round_once(monkeypatch) -> list[bool]:
    """adopt_moe on a stub DSv4.1 MoE under the current environment; the
    round_once argument of each layer all-reduce adoption it makes."""
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    calls = []
    monkeypatch.setattr(sm80_moe, "_adopt_layer_all_reduce",
                        lambda ffn, round_once: calls.append(round_once))
    monkeypatch.delenv("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", raising=False)
    ffn = SimpleNamespace(use_native_mega_moe=False,
                          gate=SimpleNamespace(tid2eid=None, e_score_correction_bias=object()),
                          scoring_func="sqrtsoftplus")
    monkeypatch.setattr(sm80_moe, "_fp32_gate_weight", lambda gate: None)
    router = sm80_moe.FusedTopKBiasRouter.__new__(sm80_moe.FusedTopKBiasRouter)
    router.num_fused_shared_experts = 0
    ffn.experts = SimpleNamespace(router=router)
    sm80_moe.adopt_moe(ffn, SimpleNamespace())
    return calls


def test_switch_adopts_the_layer_reduced_moe(monkeypatch) -> None:
    for name in ("VLLM_SM80_ALLREDUCE_ROUND_ONCE", "VLLM_SM80_PREFILL_REDUCE_OVERLAP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE", "1")
    assert _adopt_moe_round_once(monkeypatch) == [False]


def test_allreduce_default_off(monkeypatch) -> None:
    import vllm.envs as envs

    name = "VLLM_SM80_ALLREDUCE"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True


def _ring_model():
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=2048))
    return SimpleNamespace(vllm_config=cfg, config=SimpleNamespace(hidden_size=4096))


@pytest.mark.parametrize("pcie_p2p,built", [("0", False), ("1", True)])
def test_ring_is_built_only_with_pcie_peer_access(monkeypatch, pcie_p2p, built) -> None:
    """Issue #3 (dsv41 recipe): the ring maps the other cards' device memory, so with the recipe's
    P2P check failed (VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0) it is not built, rather than failing the
    boot in cudaIpcOpenMemHandle; with peer access it is built as before."""
    from vllm.distributed import parallel_state
    from vllm_sm80.models.deepseek_v41 import sm80_model

    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", pcie_p2p)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=4))
    calls = []

    def init(group, capacity_bytes):
        calls.append(capacity_bytes)
        return SimpleNamespace(capacity_bytes=capacity_bytes, device_bytes=capacity_bytes)

    monkeypatch.setattr(ce, "init_ce_ring", init)
    sm80_model._init_ce_ring(_ring_model())
    assert calls == ([2048 * 4096 * 4] if built else [])


def test_ring_unavailable_is_not_fatal(monkeypatch) -> None:
    """Where the ranks cannot share device IPC buffers although the P2P switch is on, the model
    starts without the ring (the sums take the other transports) instead of failing the boot."""
    from vllm.distributed import parallel_state
    from vllm_sm80.models.deepseek_v41 import sm80_model

    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", "1")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=4))

    def init(group, capacity_bytes):
        raise ce.PeerBuffersUnavailable("rank 1 could not open rank 0's device IPC buffer")

    monkeypatch.setattr(ce, "init_ce_ring", init)
    sm80_model._init_ce_ring(_ring_model())
    assert ce.ce_ring() is None
