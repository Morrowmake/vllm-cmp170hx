# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-water mHC scratch with eager-stream and capture-pool lifetimes."""

from contextlib import contextmanager
from threading import Lock

import torch


class ScratchOwner:
    def __init__(self, device):
        self.device = torch.device(device)
        self.part: torch.Tensor | None = None
        self.psq: torch.Tensor | None = None
        self.pack: torch.Tensor | None = None
        self.lock = Lock()
        self.event: torch.cuda.Event | None = None
        self.last_stream: int | None = None

    def workspace(self, M, BLOCK_N, split_k):
        part_size = split_k * M * BLOCK_N
        psq_size = split_k * M
        part_capacity = 0 if self.part is None else self.part.numel()
        psq_capacity = 0 if self.psq is None else self.psq.numel()
        # Doubling also bounds all superseded, still-pending allocations by
        # the current capacity, rather than accumulating one per growing tail.
        if self.part is None or part_capacity < part_size:
            self.part = torch.empty(
                max(part_size, 2 * part_capacity),
                dtype=torch.float32, device=self.device,
            )
        if self.psq is None or psq_capacity < psq_size:
            self.psq = torch.empty(
                max(psq_size, 2 * psq_capacity),
                dtype=torch.float32, device=self.device,
            )
        # Prefix views, not dimension slices: changing M or split_k must also
        # change the split strides to the CURRENT contiguous layout.
        return (
            self.part[:part_size].view(split_k, M, BLOCK_N),
            self.psq[:psq_size].view(split_k, M),
        )

    def fn_pack(self, K, BLOCK_N):
        pack_size = 3 * K * BLOCK_N
        pack_capacity = 0 if self.pack is None else self.pack.numel()
        if self.pack is None or pack_capacity < pack_size:
            self.pack = torch.empty(
                max(pack_size, 2 * pack_capacity),
                dtype=torch.bfloat16, device=self.device,
            )
        return self.pack[:pack_size].view(3, K, BLOCK_N)

    def record_stream(self, stream):
        for storage in (self.part, self.psq, self.pack):
            if storage is not None:
                storage.record_stream(stream)


_OWNERS: dict[torch.device, ScratchOwner] = {}
_OWNERS_LOCK = Lock()


def get_scratch_owner(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif device.type == "cpu":
        device = torch.device("cpu")
    with _OWNERS_LOCK:
        owner = _OWNERS.get(device)
        if owner is None:
            owner = ScratchOwner(device)
            _OWNERS[device] = owner
        return owner


@contextmanager
def scratch_owner(device):
    """Lease all three buffers until the last consumer has been enqueued.

    Eager calls share one owner per device. A host lock prevents interleaved
    launches; one CUDA event orders writes after the previous call's reduce,
    even on another stream. record_stream protects superseded storage during
    growth. Neither stream objects nor stream IDs accumulate in a cache.

    Captured calls allocate inside PyTorch's capture pool, never in the eager
    owner. The allocator retains captured addresses for the live graph/pool,
    and can reuse dead temporaries within that pool. vLLM's shared graph pools
    require serialized replay on their capture stream (cuda_graph.py); this
    scratch follows the same contract as other captured torch.empty buffers.
    Independent pools may replay concurrently, including with eager calls.
    """
    device = torch.device(device)
    stream = None
    if device.type == "cuda":
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            yield ScratchOwner(device)
            return

    owner = get_scratch_owner(device)
    with owner.lock:
        if stream is not None and owner.event is not None:
            if owner.last_stream != stream.cuda_stream:
                stream.wait_event(owner.event)
        try:
            yield owner
        finally:
            if stream is not None:
                owner.record_stream(stream)
                if owner.event is None:
                    owner.event = torch.cuda.Event()
                owner.event.record(stream)
                owner.last_stream = stream.cuda_stream
