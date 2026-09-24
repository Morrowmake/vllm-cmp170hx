# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed PP hop (VLLM_PP_PACKED_HOP / VLLM_PP_HOP_NO_METADATA), CPU only.

The transport is an in-memory stand-in for the pipeline group and
torch.distributed, so the tests check the protocol: one device transfer per
hop, metadata every step or only on the first hop, the receiver's layout
check, and that the tensors arrive intact.
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_hop import HopLayout, PackedHop

T, N, H = 7, 4, 16


def _boundary(rows=T, with_partial=False):
    torch.manual_seed(0)
    tensors = {
        "hidden_states": torch.randn(rows, N, H).to(torch.bfloat16),
        "aux_hidden_states_1": torch.randn(rows, H).to(torch.bfloat16),
        "aux_hidden_states_0": torch.randn(rows, H).to(torch.bfloat16),
    }
    if with_partial:
        tensors["aux_fc_partial"] = torch.randn(rows, H, dtype=torch.float32)
    return tensors


def test_layout_round_trip_mixed_dtypes_and_row_slice():
    tensors = _boundary(rows=10, with_partial=True)
    layout = HopLayout.from_tensors(tensors)
    # Order is by key, independent of dict order.
    assert [e.key for e in layout.entries] == sorted(tensors)
    flat = layout.pack(tensors, rows=T)
    assert flat.dtype == torch.uint8 and flat.numel() == layout.nbytes(T)
    for _, offset, _ in layout.segments(T):
        assert offset % 16 == 0
    out = layout.unpack(flat, T)
    for k, v in tensors.items():
        assert out[k].dtype == v.dtype
        assert torch.equal(out[k], v[:T])


def test_layout_rejects_other_keys():
    layout = HopLayout.from_tensors(_boundary())
    bad = _boundary()
    bad.pop("aux_hidden_states_1")
    with pytest.raises(AssertionError):
        layout.pack(bad, T)


class _Wire:
    """In-memory pipeline group + torch.distributed for two ranks."""

    def __init__(self):
        self.meta: deque = deque()
        self.data: deque = deque()
        self.n_isend = 0
        self.n_isend_object = 0

    def group(self, rank):
        wire = self

        class _Handle:
            def is_completed(self):
                return True

            def wait(self):
                pass

        def isend_object(obj, dst):
            wire.n_isend_object += 1
            wire.meta.append(obj)
            return _Handle()

        def recv_object(src):
            return wire.meta.popleft()

        return SimpleNamespace(
            ranks=[0, 1],
            rank_in_group=rank,
            world_size=2,
            device_group=None,
            isend_object=isend_object,
            recv_object=recv_object,
        ), SimpleNamespace(
            isend=lambda t, dst, group: (
                setattr(wire, "n_isend", wire.n_isend + 1),
                wire.data.append(t.clone()),
                _Handle(),
            )[-1],
            irecv=lambda t, src, group: (t.copy_(wire.data.popleft()), _Handle())[-1],
        )


@pytest.mark.parametrize("no_metadata", [False, True])
def test_one_device_op_per_hop_and_metadata_policy(no_metadata):
    wire = _Wire()
    g0, d0 = wire.group(0)
    g1, d1 = wire.group(1)
    sender = PackedHop(g0, no_metadata=no_metadata, dist=d0)
    receiver = PackedHop(g1, no_metadata=no_metadata, dist=d1)
    layout = HopLayout.from_tensors(_boundary())
    steps = 4
    for step in range(steps):
        tensors = _boundary(rows=9)  # padded rows beyond the scheduled T
        handles = sender.send(tensors, rows=T)
        assert len(handles) == 1
        got, rh = receiver.recv(layout, rows=T)
        assert len(rh) == 1
        for k, v in tensors.items():
            assert got[k].shape == v[:T].shape
            assert torch.equal(got[k], v[:T])
    assert wire.n_isend == steps  # one device transfer per hop
    # Metadata every step, or only the first-hop handshake.
    assert wire.n_isend_object == (1 if no_metadata else steps)
    assert not wire.meta and not wire.data


def test_unknown_rows_force_metadata_even_without_metadata_mode():
    wire = _Wire()
    g0, d0 = wire.group(0)
    g1, d1 = wire.group(1)
    sender = PackedHop(g0, no_metadata=True, dist=d0)
    receiver = PackedHop(g1, no_metadata=True, dist=d1)
    layout = HopLayout.from_tensors(_boundary())
    for _ in range(2):
        tensors = _boundary(rows=9)
        sender.send(tensors, rows=None)  # e.g. adaptive verification
        got, _ = receiver.recv(layout, rows=None)
        assert got["hidden_states"].shape[0] == 9
    assert wire.n_isend_object == 2


def test_receiver_rejects_mismatched_layout_and_rows():
    wire = _Wire()
    g0, d0 = wire.group(0)
    g1, d1 = wire.group(1)
    sender = PackedHop(g0, no_metadata=False, dist=d0)
    receiver = PackedHop(g1, no_metadata=False, dist=d1)
    tensors = _boundary()
    other = HopLayout.from_tensors(_boundary(with_partial=True))
    sender.send(tensors, rows=T)
    with pytest.raises(RuntimeError, match="layout"):
        receiver.recv(other, rows=T)
    sender.send(tensors, rows=T)
    with pytest.raises(RuntimeError, match="rows"):
        receiver.recv(HopLayout.from_tensors(tensors), rows=T - 1)


def test_pp_hop_rows_only_when_every_stage_runs_the_scheduled_count():
    sched = SimpleNamespace(total_num_scheduled_tokens=37)
    runner = SimpleNamespace(
        adaptive_verification=None, pcp_manager=None, ubatch_runner=None
    )
    assert GPUModelRunner.pp_hop_rows(runner, sched) == 37
    for attr in ("adaptive_verification", "pcp_manager", "ubatch_runner"):
        r = SimpleNamespace(**{**vars(runner), attr: object()})
        assert GPUModelRunner.pp_hop_rows(r, sched) is None


def _worker(monkeypatch, *, flag, no_meta=False, v2=True, pp=4, tp=1):
    import vllm.v1.worker.gpu_worker as gw

    monkeypatch.setenv("VLLM_PP_PACKED_HOP", "1" if flag else "0")
    monkeypatch.setenv("VLLM_PP_HOP_NO_METADATA", "1" if no_meta else "0")
    monkeypatch.setattr(
        gw, "get_pp_group", lambda: SimpleNamespace(world_size=pp, rank_in_group=0)
    )
    monkeypatch.setattr(gw, "get_tp_group", lambda: SimpleNamespace(world_size=tp))
    w = SimpleNamespace(use_v2_model_runner=v2, _pp_hop=None, _pp_hop_resolved=False)
    return gw.Worker._get_pp_hop(w)


def test_worker_hop_gate(monkeypatch):
    assert _worker(monkeypatch, flag=False) is None
    hop = _worker(monkeypatch, flag=True, no_meta=True)
    assert isinstance(hop, PackedHop) and hop.no_metadata
    assert _worker(monkeypatch, flag=True, v2=False) is None
    assert _worker(monkeypatch, flag=True, tp=2) is None
    assert _worker(monkeypatch, flag=True, pp=1) is None
