# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU address/ownership checks, not CUDA allocator or arithmetic proofs."""
import gc
import weakref

import numpy as np
import pytest
import torch
from triton.runtime import interpreter

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
from vllm.v1.worker import workspace as wm


class _RingReadsChecked(BaseException):
    """End the address probe before unsupported libdevice arithmetic."""


@pytest.mark.parametrize('position', [0, 1, 7, 8, 15, 16, 262143])
@pytest.mark.parametrize('dummy', [False, True])
@pytest.mark.parametrize('row_padding', [0, 16])
def test_compressor_predecessor_reads_own_ring(
    monkeypatch, position, dummy, row_padding
):
    # Use the actual launch wrapper, workspace allocation and Triton function.
    # The interpreter lowers signed remainder to numpy.fmod, like LLVM srem.
    # Intercept the two predecessor loads BEFORE dereferencing their addresses:
    # an out-of-bounds baseline must fail safely, not read arbitrary CPU memory.
    d, capacity = 512, 8
    row_stride = 2 * d + row_padding
    ring = torch.empty_strided(
        (2, capacity, 2 * d),
        (capacity * row_stride + row_padding, row_stride, 1),
        dtype=torch.float32,
    ).zero_()
    raw = torch.zeros((1, 2 * d), dtype=torch.float32)
    norm = torch.ones(d, dtype=torch.bfloat16)
    positions = torch.tensor([position], dtype=torch.int64)
    req = torch.zeros(1, dtype=torch.int32)
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    slots = torch.tensor([-1 if dummy else 0], dtype=torch.int64)
    ring_slots = torch.tensor(
        [-1 if dummy else capacity + position % capacity], dtype=torch.int64
    )
    manager = wm.WorkspaceManager(torch.device('cpu'))
    monkeypatch.setattr(wm, 'current_workspace_manager', lambda: manager)
    ws = pipe.workspace(1, 8, 1280, 'cpu', 'target', execution_stream=17)
    kernel = interpreter.InterpretedFunction(pipe._compress_latent_kernel.fn)
    monkeypatch.setattr(pipe, '_compress_latent_kernel', kernel)

    # The Torch consumer's circular indexing selects the expected physical row.
    previous = torch.remainder(positions[0] - 1, capacity).item()
    expected = ring[0 if dummy else 1, previous]
    load = interpreter.interpreter_builder.create_masked_load
    reads = []
    ring_slots_seen = False

    def checked_load(ptrs, mask, other, *args):
        nonlocal ring_slots_seen
        addresses = ptrs.data[mask.data]
        if ring_slots_seen:
            offset = len(reads) * d
            wanted = expected.data_ptr() + (
                np.arange(d, dtype=np.uint64) + offset
            ) * expected.element_size()
            assert np.array_equal(addresses, wanted), (
                'predecessor read escapes its circular row',
                int(addresses[0]) - ring.data_ptr(),
                int(wanted[0]) - ring.data_ptr(),
            )
            value = load(ptrs, mask, other, *args)
            reads.append(addresses.copy())
            if len(reads) == 2:
                raise _RingReadsChecked
            return value
        value = load(ptrs, mask, other, *args)
        if addresses.size == 1 and int(addresses[0]) == ring_slots.data_ptr():
            ring_slots_seen = True
        return value

    monkeypatch.setattr(
        interpreter.interpreter_builder, 'create_masked_load', checked_load
    )
    with pytest.raises(_RingReadsChecked):
        pipe.compress_rows(
            raw, norm, ring, positions, req, qsl, slots, ring_slots,
            2, d, 1e-6, latent=ws.latent, closes=ws.closes,
        )
    assert len(reads) == 2


def test_pipeline_workspace_retains_production_allocations_by_owner(monkeypatch):
    manager = wm.WorkspaceManager(torch.device('cpu'), num_lanes=2)
    monkeypatch.setattr(wm, 'current_workspace_manager', lambda: manager)

    def get(owner='target', stream=17):
        return pipe.workspace(48, 262144, 1280, 'cpu', owner,
                              execution_stream=stream)

    ws = get()
    tensors = [v for v in vars(ws).values() if isinstance(v, torch.Tensor)]
    tensors += [v for v in vars(ws.selection).values()
                if isinstance(v, torch.Tensor)]
    assert sum(t.numel() * t.element_size() for t in tensors) == (
        pipe.PipelineWorkspace.bytes(48, 262144, 1280)
    )
    pointers = [t.data_ptr() for t in tensors]
    saved = weakref.ref(ws)
    del tensors, ws
    gc.collect()
    # Retirement of temporary caller references does not retire manager storage.
    ws = get()
    assert ws is saved()
    retained = [v for v in vars(ws).values() if isinstance(v, torch.Tensor)]
    retained += [v for v in vars(ws.selection).values()
                 if isinstance(v, torch.Tensor)]
    assert [t.data_ptr() for t in retained] == pointers
    assert get('draft').scores.data_ptr() != ws.scores.data_ptr()
    assert get(stream=19).scores.data_ptr() != ws.scores.data_ptr()
    with wm.use_workspace_lane(1):
        assert get().scores.data_ptr() != ws.scores.data_ptr()
    manager.lock()
    assert get() is ws
    with pytest.raises(AssertionError, match='not allocated'):
        get('unreserved')
