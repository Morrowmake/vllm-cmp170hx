# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PP_SPLIT_DRAFT_EVENT, CPU only: a non-last stage consumes a step's
sampled tokens and counts behind one event and defers the draft tokens
(broadcast after the drafter ran) behind their own event until right before
the forward, where the input ids are rebuilt.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as mr
import vllm.v1.worker.gpu.pp_utils as pp_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler


class _Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


def _handler(monkeypatch, split=True):
    monkeypatch.setattr(
        pp_utils, "async_tensor_h2d", lambda data, device: torch.as_tensor(data)
    )
    h = PPHandler.__new__(PPHandler)
    h.device = torch.device("cpu")
    h.main_stream = _Stream()
    h.req_idx_gen_np = np.zeros(8, dtype=np.int32)
    h.split_draft_event = split
    h.pending_drafts = None
    h.queue = deque()
    return h


def _slot(idx, drafts, split=True):
    n = len(idx)
    return PendingRecv(
        event="counts",
        sampled_tokens=torch.zeros(n, 4, dtype=torch.int64),
        num_sampled=torch.ones(n, dtype=torch.int32),
        num_rejected=torch.zeros(n, dtype=torch.int32),
        idx_mapping=torch.tensor(idx, dtype=torch.int32),
        idx_mapping_np=np.array(idx, dtype=np.int32),
        need_sampled_mask=np.ones(n, dtype=bool),
        gen_at_receive_np=np.zeros(n, dtype=np.int32),
        draft_tokens=torch.tensor(drafts, dtype=torch.int64),
        draft_event="drafts" if split else None,
    )


def test_consume_waits_counts_only_and_defers_drafts(monkeypatch):
    h = _handler(monkeypatch)
    h.queue.append(_slot([3, 5], [[11, 12, 13], [21, 22, 23]]))
    state = torch.zeros(8, 3, dtype=torch.int64)
    out = h.get_prev_sampled_outputs(state)
    assert out is not None
    assert h.main_stream.waited == ["counts"]
    assert not state.any()  # drafts not written yet
    assert h.has_pending_drafts()
    h.apply_pending_drafts(state)
    assert h.main_stream.waited == ["counts", "drafts"]
    assert state[3].tolist() == [11, 12, 13] and state[5].tolist() == [21, 22, 23]
    assert not h.has_pending_drafts()


def test_rows_freed_before_the_late_scatter_are_skipped(monkeypatch):
    h = _handler(monkeypatch)
    h.queue.append(_slot([3, 5], [[11, 12, 13], [21, 22, 23]]))
    state = torch.zeros(8, 3, dtype=torch.int64)
    h.get_prev_sampled_outputs(state)
    # Request slot 5 finishes and is re-used by a new request in between.
    h.on_req_idx_freed(5)
    state[5] = 99
    h.apply_pending_drafts(state)
    assert state[3].tolist() == [11, 12, 13]
    assert state[5].tolist() == [99, 99, 99]


def test_unapplied_pending_is_flushed_before_the_next_consume(monkeypatch):
    h = _handler(monkeypatch)
    h.queue.extend([_slot([1], [[1, 2, 3]]), _slot([2], [[4, 5, 6]])])
    state = torch.zeros(8, 3, dtype=torch.int64)
    h.get_prev_sampled_outputs(state)  # step without a forward: never applied
    h.get_prev_sampled_outputs(state)
    assert state[1].tolist() == [1, 2, 3]
    h.apply_pending_drafts(state)
    assert state[2].tolist() == [4, 5, 6]


def test_without_split_drafts_are_written_at_consume(monkeypatch):
    h = _handler(monkeypatch, split=False)
    h.queue.append(_slot([3], [[7, 8, 9]], split=False))
    state = torch.zeros(8, 3, dtype=torch.int64)
    h.get_prev_sampled_outputs(state)
    assert state[3].tolist() == [7, 8, 9]
    assert not h.has_pending_drafts()


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_filtered_drafts_preserve_freed_and_unsampled_slots(monkeypatch, split, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    transfer = pp_utils.async_tensor_h2d
    handler = _handler(monkeypatch, split=split)
    handler.device = torch.device(device)
    slot = _slot([1, 3, 5], [[11, 12, 13], [21, 22, 23], [31, 32, 33]], split)
    slot.need_sampled_mask[2] = False
    handler.on_req_idx_freed(3)
    state = torch.full((8, 3), 99, dtype=torch.int64, device=device)
    if device == "cuda":
        monkeypatch.setattr(pp_utils, "async_tensor_h2d", transfer)
        handler.main_stream = torch.cuda.current_stream()
        for name in (
            "sampled_tokens",
            "num_sampled",
            "num_rejected",
            "idx_mapping",
            "draft_tokens",
        ):
            setattr(slot, name, getattr(slot, name).to(device))
        slot.event = torch.cuda.Event()
        slot.event.record()
        slot.draft_event = slot.event if split else None
    handler.queue.append(slot)
    sync_mode = torch.cuda.get_sync_debug_mode() if device == "cuda" else None
    try:
        if sync_mode is not None:
            torch.cuda.set_sync_debug_mode("error")
        result = handler.get_prev_sampled_outputs(state)
        if split:
            handler.apply_pending_drafts(state)
    finally:
        if sync_mode is not None:
            torch.cuda.set_sync_debug_mode(sync_mode)
    expected = torch.full((8, 3), 99, dtype=torch.int64)
    expected[1] = torch.tensor([11, 12, 13])
    torch.testing.assert_close(state.cpu(), expected)
    assert result["idx_mapping"].cpu().tolist() == [1, -1, -1]


def test_runner_rebuilds_input_ids_after_the_late_scatter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mr, "combine_sampled_and_draft_tokens", lambda *a: calls.append(a)
    )
    applied: list[str] = []
    handler = SimpleNamespace(
        has_pending_drafts=lambda: not applied,
        apply_pending_drafts=lambda state: applied.append(state),
    )
    runner = SimpleNamespace(
        pp_handler=handler,
        req_states=SimpleNamespace(draft_tokens="state"),
        _late_combine_args=("args",),
    )
    GPUModelRunner.apply_late_drafts(runner)
    assert applied == ["state"]
    assert calls == [("args",)]
    assert runner._late_combine_args is None
    GPUModelRunner.apply_late_drafts(runner)  # nothing pending: no-op
    assert calls == [("args",)]
