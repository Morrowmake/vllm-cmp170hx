# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-last pipeline stages apply a step's sampled results pp_size steps late.
The mamba align postprocess reads block tables by batch row, so before it runs
the runner must restore that step's batch-order block tables: by then the
persistent batch-order tables hold a later step's (different) batch.

CPU only: the handler and the runner method are exercised with stand-ins, and
the deferred update once more with the real handler and the real gather kernel
in the Triton interpreter.
"""

import json
import os
import subprocess
import sys
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


_CHILD = r"""
import json, os
os.environ["TRITON_INTERPRET"] = "1"
from types import SimpleNamespace
import torch
import vllm.v1.worker.gpu.pp_utils as pp_utils
from tests.v1.worker.test_pp_deferred_block_tables import _handler_with_slot
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

# The masked mapping is uploaded with a pinned H2D copy; stay on the CPU.
pp_utils.async_tensor_h2d = lambda data, device: torch.as_tensor(data)
R, COLS = 8, 4
ptrs = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64)
# Per-request-slot tables: every slot owns its own blocks (block 0 is null).
req_bt = torch.arange(1, R * COLS + 1, dtype=torch.int32).view(R, COLS)
batch_order = torch.zeros(R, COLS, dtype=torch.int32)
tables = SimpleNamespace(
    num_kv_cache_groups=1, max_num_reqs=R,
    input_block_tables=[batch_order], input_block_table_ptrs=ptrs([batch_order]),
    block_table_ptrs=ptrs([req_bt]),
    block_table_strides=torch.tensor([COLS], dtype=torch.int64),
    num_blocks=SimpleNamespace(gpu=torch.full((1, R), COLS, dtype=torch.int32)),
    block_sizes_tensor=torch.tensor([16], dtype=torch.int32),
    has_sliding_window_groups=False,
    sliding_window_reach=torch.zeros(1, dtype=torch.int32),
    _num_dirty_rows=0)
gather = lambda idx, num_reqs_padded: BlockTables.gather_block_tables(
    tables, idx, num_reqs_padded=num_reqs_padded)

# The step whose results arrive now ran slots 5, 2, 7, 3; slot 2 was freed
# since, so the handler masks its row. A later step has gathered its own
# batch (slots 1, 6, 4, 0) into the batch-order tables meanwhile.
step, later = [5, 2, 7, 3], [1, 6, 4, 0]
gen_now = [0] * R
gen_now[2] = 1
handler, _ = _handler_with_slot(step, gen_now=gen_now,
                                gen_at_receive=[0] * 4, need=[1] * 4)
gather(torch.tensor(later, dtype=torch.int32), len(later))
stale = batch_order.clone()
seen = []
runner = SimpleNamespace(
    pp_handler=handler,
    req_states=SimpleNamespace(draft_tokens=None),
    model_state=SimpleNamespace(_align_mode=True),
    block_tables=SimpleNamespace(gather_block_tables=gather),
    postprocess_sampled=lambda **kw: seen.append(
        (kw["idx_mapping"].tolist(), batch_order.clone())))
GPUModelRunner.update_pp_decode_requests(runner)

mapping, rows = seen[0] if len(seen) == 1 else (None, None)
print("RESULT " + json.dumps({
    "postprocess_once": len(seen) == 1,
    "freed_row_masked": mapping == [5, -1, 7, 3],
    "later_step_rows_were_there": all(
        torch.equal(stale[i], req_bt[s]) for i, s in enumerate(later)),
    "rows_are_the_steps_own": rows is not None and all(
        torch.equal(rows[i], req_bt[s]) for i, s in enumerate(step)),
    "no_null_row": rows is not None and bool((rows[: len(step)] != 0).all()),
}))
"""


def test_deferred_postprocess_sees_the_steps_own_rows():
    """When a step's results reach a non-last stage, the batch-order tables
    hold a later step's rows. With the real handler (one freed request, masked
    to -1) and the real gather kernel in the Triton interpreter, by the time
    postprocess_sampled runs every batch row must hold that step's own request
    row again: not the later step's row, and not the null row a -1 mapping
    would gather for the freed request."""
    env = dict(
        os.environ,
        TRITON_INTERPRET="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(sys.path),
    )
    r = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    assert r.returncode == 0 and line, r.stderr[-3000:]
    result = json.loads(line[-1][len("RESULT ") :])
    assert result == {k: True for k in result}, result
