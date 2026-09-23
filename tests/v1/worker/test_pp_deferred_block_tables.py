# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-last pipeline stages apply a step's sampled results pp_size steps late.
The mamba align postprocess reads block tables by batch row, so before it runs
the runner must restore that step's batch-order block tables: by then the
persistent batch-order tables hold a later step's (different) batch.

CPU only: the handler and the runner method are exercised with stand-ins.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler


class _Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


def _handler_with_slot(idx_mapping_np, gen_now, gen_at_receive, need):
    h = PPHandler.__new__(PPHandler)
    h.device = torch.device("cpu")
    h.main_stream = _Stream()
    h.req_idx_gen_np = np.asarray(gen_now, dtype=np.int32)
    n = len(idx_mapping_np)
    slot = PendingRecv(
        event=object(),
        sampled_tokens=torch.zeros(n, 4, dtype=torch.int64),
        num_sampled=torch.ones(n, dtype=torch.int32),
        num_rejected=torch.zeros(n, dtype=torch.int32),
        idx_mapping=torch.as_tensor(idx_mapping_np, dtype=torch.int32),
        idx_mapping_np=np.asarray(idx_mapping_np, dtype=np.int32),
        need_sampled_mask=np.asarray(need, dtype=bool),
        gen_at_receive_np=np.asarray(gen_at_receive, dtype=np.int32),
    )
    h.queue = deque([slot])
    return h, slot


def test_prev_outputs_carry_the_steps_unfiltered_mapping():
    h, slot = _handler_with_slot(
        [5, 2, 7], gen_now=[0] * 8, gen_at_receive=[0, 0, 0], need=[1, 1, 1]
    )
    out = h.get_prev_sampled_outputs()
    assert out is not None
    assert out["step_idx_mapping"] is slot.idx_mapping
    assert out["idx_mapping"] is slot.idx_mapping
    assert h.main_stream.waited == [slot.event]


def test_prev_outputs_mask_freed_rows_but_keep_step_mapping(monkeypatch):
    import vllm.v1.worker.gpu.pp_utils as pp_utils

    # The masked mapping is uploaded with a pinned H2D copy; stay on the CPU.
    monkeypatch.setattr(
        pp_utils, "async_tensor_h2d", lambda data, device: torch.as_tensor(data)
    )
    # Request slot 2 was freed (generation moved on) since the step ran.
    gen_now = [0] * 8
    gen_now[2] = 1
    h, slot = _handler_with_slot(
        [5, 2, 7], gen_now=gen_now, gen_at_receive=[0, 0, 0], need=[1, 1, 1]
    )
    out = h.get_prev_sampled_outputs()
    assert out is not None
    assert out["idx_mapping"].tolist() == [5, -1, 7]
    # The block-table restore needs valid request indices for every row.
    assert out["step_idx_mapping"].tolist() == [5, 2, 7]


class _Recorder:
    def __init__(self):
        self.calls = []


def _fake_runner(align_mode: bool, step_idx_mapping: torch.Tensor):
    rec = _Recorder()
    outputs = dict(
        sampled_tokens=torch.zeros(3, 4, dtype=torch.int64),
        num_sampled=torch.ones(3, dtype=torch.int32),
        num_rejected=torch.zeros(3, dtype=torch.int32),
        idx_mapping=torch.tensor([5, -1, 7], dtype=torch.int32),
        step_idx_mapping=step_idx_mapping,
    )
    runner = SimpleNamespace(
        pp_handler=SimpleNamespace(get_prev_sampled_outputs=lambda _d: dict(outputs)),
        req_states=SimpleNamespace(draft_tokens=None),
        model_state=SimpleNamespace(_align_mode=align_mode),
        block_tables=SimpleNamespace(
            gather_block_tables=lambda idx, num_reqs_padded: rec.calls.append(
                ("gather", idx, num_reqs_padded)
            )
        ),
        postprocess_sampled=lambda **kw: rec.calls.append(("postprocess", kw)),
    )
    return runner, rec


def test_align_mode_restores_step_block_tables_before_postprocess():
    step = torch.tensor([5, 2, 7], dtype=torch.int32)
    runner, rec = _fake_runner(True, step)
    GPUModelRunner.update_pp_decode_requests(runner)
    assert [c[0] for c in rec.calls] == ["gather", "postprocess"]
    _, idx, padded = rec.calls[0]
    assert idx is step and padded == 3
    # postprocess_sampled gets exactly its own arguments.
    assert "step_idx_mapping" not in rec.calls[1][1]
    assert rec.calls[1][1]["idx_mapping"].tolist() == [5, -1, 7]


def test_no_restore_without_align_mode():
    runner, rec = _fake_runner(False, torch.tensor([5, 2, 7], dtype=torch.int32))
    GPUModelRunner.update_pp_decode_requests(runner)
    assert [c[0] for c in rec.calls] == ["postprocess"]
    assert "step_idx_mapping" not in rec.calls[0][1]
