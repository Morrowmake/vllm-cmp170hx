# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-engine ring all-reduce for large tensor-parallel messages.

The design follows the PCIe DMA all-reduce of b12x (Apache-2.0,
``b12x/comm/pcie/pcie_dma.py``, local-inference-lab): a reduce-scatter +
all-gather ring whose data moves by copy-engine copies
(``cudaMemcpyAsync`` into the next rank's IPC scratch), while the SMs only
signal, wait and add. An NCCL ring moves the same bytes with SM kernels that
stay resident for the whole transfer; the compute beside them then runs
slower. Here the SM work per step is a one-block flag store, a one-block flag
wait and one elementwise add (reduce-scatter) or nothing (all-gather).

Per call, on the caller's stream, with ``epoch`` the call count (the same on
every rank: every rank issues the same calls in the same order):

1. ready: store ``epoch`` into the previous rank's ready flag (this rank's
   scratch is free: its last reader is earlier on this stream), wait until the
   next rank's store reaches this rank's flag;
2. ``2 (world - 1)`` steps ``k``: copy the send shard into the next rank's
   scratch slot ``k``, store ``epoch`` into its flag ``k``, wait for this
   rank's flag ``k`` (the previous rank's copy into this rank's slot ``k`` has
   landed), then add the slot into the received shard (reduce-scatter:
   ``out = input + slot``, the first and only touch of that shard) or copy it
   there (all-gather).

Each element is summed in the order its owner's reduce-scatter visits the
ranks, in the message dtype (FP32 adds for FP32, BF16 rounding per add for
BF16, as NCCL's ring); every rank ends with the same bits.
:func:`emulate` computes those bits on the host. Flags are monotonic epochs
(no reset); scratch slots are per step, so no slot is rewritten within a call.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

FLAG_STRIDE_BYTES = 128
SCRATCH_ALIGN = 256
DTYPES = (torch.float32, torch.bfloat16)


@dataclass(frozen=True)
class Step:
    k: int
    reduce: bool
    send: int
    recv: int


def ring_steps(world: int, rank: int) -> list[Step]:
    """The ring's steps for ``rank``: shard indices it sends to ``rank + 1``
    and receives from ``rank - 1``; after the reduce-scatter it owns shard
    ``rank + 1``."""
    steps = []
    for k in range(2 * (world - 1)):
        if k < world - 1:
            steps.append(Step(k, True, (rank - k) % world, (rank - k - 1) % world))
        else:
            j = k - (world - 1)
            steps.append(Step(k, False, (rank + 1 - j) % world, (rank - j) % world))
    return steps


def emulate(inputs: list[torch.Tensor]) -> torch.Tensor:
    """The ring's result (every rank's) for per-rank ``inputs``, on the host,
    with the device's per-add rounding."""
    world = len(inputs)
    shards = [list(t.reshape(world, -1).unbind(0)) for t in inputs]
    out = [[None] * world for _ in range(world)]
    for k in range(world - 1):
        sent = []
        for r in range(world):
            s = ring_steps(world, r)[k]
            src = shards[r][s.send] if k == 0 else out[r][s.send]
            sent.append(src)
        for r in range(world):
            s = ring_steps(world, r)[k]
            out[r][s.recv] = shards[r][s.recv] + sent[(r - 1) % world]
    return torch.cat([out[(c - 1) % world][c] for c in range(world)])


def _ready_slot(world: int) -> int:
    return 2 * (world - 1)


def flag_bytes(world: int) -> int:
    # Ring: one flag per step plus its ready flag. Owner sum (round_once_ops):
    # the same step flags (reduce-scatter, then all-gather, indexed by the
    # sender's offset) plus one ready flag per receiver, after the ring's.
    return (_ready_slot(world) + world) * FLAG_STRIDE_BYTES


def emulate_owner(inputs: list[torch.Tensor]) -> torch.Tensor:
    """:func:`round_once_ops`' result for per-rank ``inputs``: every element
    summed in FP32 in rank order, rounded once to the input dtype."""
    acc = inputs[0].float()
    for t in inputs[1:]:
        acc = acc + t.float()
    return acc.to(inputs[0].dtype)


def _slot_index(world: int, sender: int, receiver: int) -> int:
    """The receiver-side slot (and flag) index of ``sender``'s shard."""
    return (sender - receiver) % world - 1


def round_once_ops(backend, world: int, rank: int, epoch: int, inp: int, out: int,
                   shard_bytes: int, shard_numel: int, dtype: torch.dtype,
                   scratch: list[int], flags: list[int], slot_bytes: int) -> Iterator[None]:
    """One owner-sum call: every rank copies shard ``j`` to rank ``j``'s slot
    for it (copy engine), rank ``j`` sums the ``world`` partials of shard ``j``
    in FP32 in rank order and rounds once, then copies the result to every
    other rank (copy engine). Same bytes as the ring; the sum depends only on
    the element's partials, so a message split anywhere gets the same bits.
    Slots and flags 0..world-2 carry the reduce-scatter, world-1..2world-3 the
    all-gather; the ready flags follow the ring's."""
    ready0 = (_ready_slot(world) + 1) * FLAG_STRIDE_BYTES
    peers = [(rank + o) % world for o in range(1, world)]
    for q in peers:  # my slots in q's buffer: q has freed them
        backend.set_flag(flags[q] + ready0 + _slot_index(world, q, rank) * FLAG_STRIDE_BYTES, epoch)
    for q in peers:
        yield from backend.wait_flag(
            flags[rank] + ready0 + _slot_index(world, rank, q) * FLAG_STRIDE_BYTES, epoch)
    for q in peers:
        k = _slot_index(world, rank, q)
        backend.copy(scratch[q] + k * slot_bytes, inp + q * shard_bytes, shard_bytes)
        backend.set_flag(flags[q] + k * FLAG_STRIDE_BYTES, epoch)
    parts = []
    for s in range(world):
        if s == rank:
            parts.append(inp + rank * shard_bytes)
            continue
        k = _slot_index(world, s, rank)
        yield from backend.wait_flag(flags[rank] + k * FLAG_STRIDE_BYTES, epoch)
        parts.append(scratch[rank] + k * slot_bytes)
    mine = out + rank * shard_bytes
    backend.sum_round(mine, parts, shard_numel, dtype)
    gather = world - 1
    for q in peers:
        k = gather + _slot_index(world, rank, q)
        backend.copy(scratch[q] + k * slot_bytes, mine, shard_bytes)
        backend.set_flag(flags[q] + k * FLAG_STRIDE_BYTES, epoch)
    for s in peers:
        k = gather + _slot_index(world, s, rank)
        yield from backend.wait_flag(flags[rank] + k * FLAG_STRIDE_BYTES, epoch)
        backend.copy(out + s * shard_bytes, scratch[rank] + k * slot_bytes, shard_bytes)


def ring_ops(backend, world: int, rank: int, epoch: int, inp: int, out: int,
             shard_bytes: int, shard_numel: int, dtype: torch.dtype,
             scratch: list[int], flags: list[int], slot_bytes: int) -> Iterator[None]:
    """One call's operations on ``backend`` (pointers are integers). A
    generator: a backend's ``wait_flag`` may yield until the flag is set (the
    host simulation); a device backend queues the wait and never yields."""
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    ready = _ready_slot(world) * FLAG_STRIDE_BYTES
    backend.set_flag(flags[prv] + ready, epoch)
    yield from backend.wait_flag(flags[rank] + ready, epoch)
    for s in ring_steps(world, rank):
        src = (inp if s.k == 0 else out) + s.send * shard_bytes
        backend.copy(scratch[nxt] + s.k * slot_bytes, src, shard_bytes)
        backend.set_flag(flags[nxt] + s.k * FLAG_STRIDE_BYTES, epoch)
        yield from backend.wait_flag(flags[rank] + s.k * FLAG_STRIDE_BYTES, epoch)
        slot = scratch[rank] + s.k * slot_bytes
        dst = out + s.recv * shard_bytes
        if s.reduce:
            backend.add(dst, inp + s.recv * shard_bytes, slot, shard_numel, dtype)
        else:
            backend.copy(dst, slot, shard_bytes)


# --- device backend -------------------------------------------------------------

class _Ptr:
    """A raw device pointer with a dtype, as Triton takes a tensor argument."""

    __slots__ = ("_ptr", "dtype")

    def __init__(self, ptr: int, dtype: torch.dtype):
        self._ptr, self.dtype = ptr, dtype

    def data_ptr(self) -> int:
        return self._ptr


# A flag wait gives up after this long and marks the step failed (bit
# CE_RING_TIMEOUT of the stream's sticky error word, step_errors.py) instead of
# spinning forever: a peer that never arrives fails the step at its output
# copy, it does not hang the server. Prefill sums take tens of milliseconds and
# ranks skew by at most a layer's expert imbalance, so a minute only trips on a
# real desynchronisation.
WAIT_TIMEOUT_NS = 60_000_000_000


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _store_release_sys(ptr, value):
        # System scope: the peer reads the flag over PCIe and must see the
        # earlier copy-engine writes of the slot first.
        tl.inline_asm_elementwise("st.release.sys.global.u32 [$1], $2; mov.u32 $0, 0;", "=r,l,r",
                                  [ptr, value], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _load_acquire_sys(ptr):
        return tl.inline_asm_elementwise("ld.acquire.sys.global.u32 $0, [$1];", "=r,l",
                                         [ptr], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _now_ns(ptr):
        # %globaltimer; the pointer only gives the asm an operand.
        return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,l",
                                         [ptr], dtype=tl.int64, is_pure=False, pack=1)

    @triton.jit
    def set_flag(flag_ptr, value):
        # Queued after the copy on the same stream: the copy has completed.
        _store_release_sys(flag_ptr, value)

    @triton.jit
    def wait_flag(flag_ptr, value, err_ptr, timeout_ns, code):
        start = _now_ns(flag_ptr)
        cur = _load_acquire_sys(flag_ptr)
        late = _now_ns(flag_ptr) - start > timeout_ns
        while (cur < value) & (late == 0):
            cur = _load_acquire_sys(flag_ptr)
            late = _now_ns(flag_ptr) - start > timeout_ns
        if cur < value:
            tl.atomic_or(err_ptr, code, sem="relaxed")

    @triton.jit
    def add(dst_ptr, a_ptr, b_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, (a.to(tl.float32) + b.to(tl.float32)).to(a.dtype), mask=mask)

    @triton.jit
    def sum_round(dst_ptr, p0, p1, p2, p3, p4, p5, p6, p7, n, W: tl.constexpr,
                  BLOCK: tl.constexpr):
        # FP32 sum of W partials in rank order, rounded once to dst's dtype.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        acc = tl.load(p0 + offs, mask=mask).to(tl.float32)
        if W > 1:
            acc = acc + tl.load(p1 + offs, mask=mask).to(tl.float32)
        if W > 2:
            acc = acc + tl.load(p2 + offs, mask=mask).to(tl.float32)
        if W > 3:
            acc = acc + tl.load(p3 + offs, mask=mask).to(tl.float32)
        if W > 4:
            acc = acc + tl.load(p4 + offs, mask=mask).to(tl.float32)
        if W > 5:
            acc = acc + tl.load(p5 + offs, mask=mask).to(tl.float32)
        if W > 6:
            acc = acc + tl.load(p6 + offs, mask=mask).to(tl.float32)
        if W > 7:
            acc = acc + tl.load(p7 + offs, mask=mask).to(tl.float32)
        tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask)

    return set_flag, wait_flag, add, sum_round


class DeviceBackend:
    """Queues a call's operations on the current CUDA stream."""

    ADD_BLOCK = 2048

    def __init__(self, memcpy_async: Callable[[int, int, int, int], None], kernels,
                 error: torch.Tensor, timeout_ns: int = WAIT_TIMEOUT_NS):
        self._memcpy, (self._set, self._wait, self._add, self._sum) = memcpy_async, kernels
        self.stream = torch.cuda.current_stream().cuda_stream
        self._error, self._timeout_ns = error, timeout_ns

    def copy(self, dst: int, src: int, nbytes: int) -> None:
        self._memcpy(dst, src, nbytes, self.stream)

    def set_flag(self, ptr: int, value: int) -> None:
        self._set[(1,)](_Ptr(ptr, torch.int32), value)

    def wait_flag(self, ptr: int, value: int) -> Iterator[None]:
        from vllm_sm80.distributed.step_errors import CE_RING_TIMEOUT

        self._wait[(1,)](_Ptr(ptr, torch.int32), value, self._error, self._timeout_ns, CE_RING_TIMEOUT)
        return iter(())

    def add(self, dst: int, a: int, b: int, n: int, dtype: torch.dtype) -> None:
        grid = (triton_cdiv(n, self.ADD_BLOCK),)
        self._add[grid](_Ptr(dst, dtype), _Ptr(a, dtype), _Ptr(b, dtype), n, BLOCK=self.ADD_BLOCK)

    def sum_round(self, dst: int, parts: list[int], n: int, dtype: torch.dtype) -> None:
        if not 1 <= len(parts) <= 8:
            raise ValueError("the owner sum takes 1 to 8 partials")
        ptrs = [_Ptr(p, dtype) for p in parts] + [_Ptr(parts[0], dtype)] * (8 - len(parts))
        grid = (triton_cdiv(n, self.ADD_BLOCK),)
        self._sum[grid](_Ptr(dst, dtype), *ptrs, n, W=len(parts), BLOCK=self.ADD_BLOCK)


def triton_cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _memcpy_async_fn() -> Callable[[int, int, int, int], None]:
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary().lib
    fn = lib.cudaMemcpyAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]

    def memcpy_async(dst: int, src: int, nbytes: int, stream: int) -> None:
        rc = fn(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, 4, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed: cudaError {rc}")

    return memcpy_async


class CeRingAllReduce:
    """The TP group's copy-engine ring: per rank one scratch slot per ring step
    for shards of messages up to ``capacity_bytes`` and one flag block, shared
    across ranks through CUDA IPC (collective: every rank constructs it)."""

    def __init__(self, group, capacity_bytes: int):
        import torch.distributed as dist

        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

        self.world, self.rank = group.world_size, group.rank_in_group
        if self.world < 2:
            raise ValueError("the copy-engine ring needs at least two ranks")
        shard = -(-capacity_bytes // self.world)
        self.slot_bytes = -(-shard // SCRATCH_ALIGN) * SCRATCH_ALIGN
        self.capacity_bytes = self.slot_bytes * self.world
        steps = 2 * (self.world - 1)
        cpu = group.cpu_group
        self.scratch = CustomAllreduce.create_shared_buffer(self.slot_bytes * steps, group=cpu)
        self.flags = CustomAllreduce.create_shared_buffer(flag_bytes(self.world), group=cpu)
        CudaRTLibrary().cudaMemset(ctypes.c_void_p(self.flags[self.rank]), 0, flag_bytes(self.world))
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        self.device_bytes = self.slot_bytes * steps + flag_bytes(self.world)
        self.epoch = 0
        self._memcpy = _memcpy_async_fn()
        self._kernels = _kernels()

    def takes(self, t: torch.Tensor) -> bool:
        """Whether this ring reduces ``t``: an FP32 or BF16 contiguous CUDA
        tensor within the capacity whose shards are 16-byte multiples, outside
        CUDA graph capture."""
        if not t.is_cuda or t.dtype not in DTYPES or not t.is_contiguous():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        nbytes = t.numel() * t.element_size()
        return 0 < nbytes <= self.capacity_bytes and nbytes % (16 * self.world) == 0

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """The sum of every rank's ``t`` around the ring (each add in ``t``'s
        dtype), queued on the current stream."""
        return self._run(ring_ops, t)

    def round_once(self, t: torch.Tensor) -> torch.Tensor:
        """The sum of every rank's ``t``, each element summed in FP32 in rank
        order and rounded once (:func:`round_once_ops`), queued on the current
        stream. The same bits for any split of the message."""
        if self.world > 8:
            raise ValueError("the owner sum takes at most eight ranks")
        return self._run(round_once_ops, t)

    def _run(self, ops, t: torch.Tensor) -> torch.Tensor:
        from vllm_sm80.distributed.step_errors import error_buffer

        out = torch.empty_like(t)
        self.epoch += 1
        shard_numel = t.numel() // self.world
        backend = DeviceBackend(self._memcpy, self._kernels, error_buffer(t.device))
        for _ in ops(backend, self.world, self.rank, self.epoch, t.data_ptr(), out.data_ptr(),
                     shard_numel * t.element_size(), shard_numel, t.dtype,
                     self.scratch, self.flags, self.slot_bytes):
            raise RuntimeError("a device wait yielded")
        return out


_RING: CeRingAllReduce | None = None


def init_ce_ring(group, capacity_bytes: int) -> CeRingAllReduce:
    """Build the process's ring once (collective over ``group``)."""
    global _RING
    if _RING is None:
        _RING = CeRingAllReduce(group, capacity_bytes)
    return _RING


def ce_ring() -> CeRingAllReduce | None:
    return _RING
