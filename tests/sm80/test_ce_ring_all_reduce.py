# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the copy-engine ring all-reduce (vllm_sm80/distributed/ce_ring_all_reduce.py):
the ring schedule, the host emulation of its bits, and ``ring_ops`` run for every rank against a
host model of the shared memory, with every copy's content checked where it lands.

    pytest -q tests/sm80/test_ce_ring_all_reduce.py
"""

import pytest
import torch

from vllm_sm80.distributed import ce_ring_all_reduce as ce

DT = {torch.float32: 4, torch.bfloat16: 2}


@pytest.mark.parametrize("world", [2, 3])
def test_ring_schedule_reduces_every_shard_once_and_gathers_all(world) -> None:
    for rank in range(world):
        steps = ce.ring_steps(world, rank)
        assert len(steps) == 2 * (world - 1)
        rs = [s for s in steps if s.reduce]
        ag = [s for s in steps if not s.reduce]
        # Reduce-scatter: every shard but this rank's own is received once.
        assert sorted(s.recv for s in rs) == sorted(set(range(world)) - {rank})
        # The shard sent at step k+1 is the one received at step k.
        for a, b in zip(steps, steps[1:]):
            assert b.send == a.recv
        assert steps[0].send == rank
        # All-gather: the owned shard (rank + 1) goes out first; every other shard comes in once.
        assert ag[0].send == (rank + 1) % world
        assert sorted(s.recv for s in ag) == sorted(set(range(world)) - {(rank + 1) % world})
        # What this rank receives at step k is what the previous rank sends then.
        prev = ce.ring_steps(world, (rank - 1) % world)
        assert [s.recv for s in steps] == [p.send for p in prev]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_emulation_is_the_sum(dtype) -> None:
    g = torch.Generator().manual_seed(1)
    parts = [torch.randn(4 * 96, generator=g).to(dtype) for _ in range(4)]
    want = torch.stack([p.double() for p in parts]).sum(0)
    got = ce.emulate(parts).double()
    tol = 1e-5 if dtype == torch.float32 else 0.06
    assert (got - want).abs().max().item() < tol


class _Memory:
    """Host model of the ranks' device memory: regions at disjoint fake addresses."""

    def __init__(self):
        self.regions = []

    def alloc(self, nbytes: int) -> int:
        base = (len(self.regions) + 1) << 40
        self.regions.append((base, torch.zeros(nbytes, dtype=torch.uint8)))
        return base

    def put(self, t: torch.Tensor) -> int:
        b = t.contiguous().view(torch.uint8).flatten()
        base = self.alloc(b.numel())
        self.at(base, b.numel()).copy_(b)
        return base

    def at(self, addr: int, nbytes: int) -> torch.Tensor:
        for base, buf in self.regions:
            if base <= addr and addr + nbytes <= base + buf.numel():
                return buf[addr - base:addr - base + nbytes]
        raise AssertionError(f"address {addr:#x}+{nbytes} outside every region")


class _HostBackend:
    """One rank's operations on the host model. Copies are checked byte for byte where they land."""

    def __init__(self, mem: _Memory, log: list):
        self.mem, self.log = mem, log

    def copy(self, dst, src, nbytes):
        data = self.mem.at(src, nbytes).clone()
        self.mem.at(dst, nbytes).copy_(data)
        assert torch.equal(self.mem.at(dst, nbytes), data)
        self.log.append(("copy", dst, data))

    def set_flag(self, ptr, value):
        self.mem.at(ptr, 4).view(torch.int32)[0] = value

    def wait_flag(self, ptr, value):
        while int(self.mem.at(ptr, 4).view(torch.int32)[0]) < value:
            yield

    def add(self, dst, a, b, n, dtype):
        size = DT[dtype] * n
        x = self.mem.at(a, size).view(dtype).float()
        y = self.mem.at(b, size).view(dtype).float()
        self.mem.at(dst, size).view(dtype).copy_((x + y).to(dtype))

    def sum_round(self, dst, parts, n, dtype):
        size = DT[dtype] * n
        acc = self.mem.at(parts[0], size).view(dtype).float()
        for p in parts[1:]:
            acc = acc + self.mem.at(p, size).view(dtype).float()
        self.mem.at(dst, size).view(dtype).copy_(acc.to(dtype))


def _run_ring(world, inputs_per_call, slot_bytes, ops=None):
    """``ops[i]``: the call's algorithm (ce.ring_ops by default)."""
    mem = _Memory()
    steps = 2 * (world - 1)
    scratch = [mem.alloc(slot_bytes * steps) for _ in range(world)]
    flags = [mem.alloc(ce.flag_bytes(world)) for _ in range(world)]
    results = []
    logs = [[] for _ in range(world)]
    for epoch, inputs in enumerate(inputs_per_call, start=1):
        dtype = inputs[0].dtype
        shard_numel = inputs[0].numel() // world
        inp = [mem.put(t) for t in inputs]
        out = [mem.alloc(t.numel() * DT[dtype]) for t in inputs]
        algo = ops[epoch - 1] if ops else ce.ring_ops
        gens = [algo(_HostBackend(mem, logs[r]), world, r, epoch, inp[r], out[r],
                            shard_numel * DT[dtype], shard_numel, dtype, scratch, flags, slot_bytes)
                for r in range(world)]
        live, spins = set(range(world)), 0
        while live:
            for r in sorted(live):
                try:
                    next(gens[r])
                except StopIteration:
                    live.discard(r)
            spins += 1
            assert spins < 10_000, "ring deadlocked"
        results.append([mem.at(out[r], inputs[r].numel() * DT[dtype]).view(dtype).clone()
                        for r in range(world)])
    return results, logs


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ring_ops_give_the_emulated_bits_on_every_rank(world, dtype) -> None:
    """Three calls back to back (fresh data, monotonic epochs, reused scratch slots): every rank ends
    with the emulation's bits, and every copy put exactly its source bytes where it landed."""
    g = torch.Generator().manual_seed(world)
    calls = [[torch.randn(world * 40, generator=g).to(dtype) for _ in range(world)] for _ in range(3)]
    slot_bytes = 256 * ((40 * 4 + 255) // 256)
    results, logs = _run_ring(world, calls, slot_bytes)
    for inputs, outs in zip(calls, results):
        want = ce.emulate(inputs)
        for o in outs:
            assert torch.equal(o, want)
    # Each call: 2(world-1) peer copies plus (world-1) all-gather local copies per rank.
    for log in logs:
        assert len(log) == 3 * (2 * (world - 1) + (world - 1))


@pytest.mark.parametrize("world", [2, 3, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_owner_sum_rounds_once_on_every_rank_between_ring_calls(world, dtype) -> None:
    """Owner-sum calls interleaved with ring calls on the same scratch, flags and epochs: every rank
    ends each owner call with the FP32 rank-order sum rounded once, each ring call with the ring's
    bits, and every copy put exactly its source bytes where it landed."""
    g = torch.Generator().manual_seed(10 + world)
    ops = [ce.round_once_ops, ce.ring_ops, ce.round_once_ops, ce.round_once_ops, ce.ring_ops]
    calls = [[torch.randn(world * 40, generator=g).to(dtype) for _ in range(world)] for _ in ops]
    slot_bytes = 256 * ((40 * 4 + 255) // 256)
    results, _ = _run_ring(world, calls, slot_bytes, ops)
    for algo, inputs, outs in zip(ops, calls, results):
        want = ce.emulate_owner(inputs) if algo is ce.round_once_ops else ce.emulate(inputs)
        for o in outs:
            assert torch.equal(o, want)


def test_owner_sum_is_split_invariant() -> None:
    """A message summed whole and in two slices gives the same bits (the sum of an element depends
    only on its partials), which the ring's position-dependent order does not guarantee."""
    world = 4
    g = torch.Generator().manual_seed(3)
    parts = [(torch.randn(world * 64, generator=g) * 2.0 ** torch.randint(-8, 8, (world * 64,),
              generator=g)).to(torch.bfloat16) for _ in range(world)]
    slot_bytes = 256
    whole, _ = _run_ring(world, [parts], slot_bytes, [ce.round_once_ops])
    halves = [[p[:world * 16] for p in parts], [p[world * 16:] for p in parts]]
    split_res, _ = _run_ring(world, halves, slot_bytes, [ce.round_once_ops] * 2)
    assert torch.equal(torch.cat([split_res[0][0], split_res[1][0]]), whole[0][0])
    want = torch.stack([p.double() for p in parts]).sum(0)
    assert (whole[0][0].double() - want).abs().max() <= want.abs().max() * 2.0 ** -7


def test_owner_sum_waits_for_every_receiver_before_writing() -> None:
    """A rank copies into no peer's scratch before every peer's ready store for it."""
    world, mem, events = 4, _Memory(), []
    scratch = [mem.alloc(256 * 2 * (world - 1)) for _ in range(world)]
    flags = [mem.alloc(ce.flag_bytes(world)) for _ in range(world)]

    class Logged(_HostBackend):
        def copy(self, dst, src, nbytes):
            events.append("copy")
            super().copy(dst, src, nbytes)

    x = torch.ones(world * 4)
    inp = [mem.put(x) for _ in range(world)]
    out = [mem.alloc(x.numel() * 4) for _ in range(world)]
    g0 = ce.round_once_ops(Logged(mem, []), world, 0, 1, inp[0], out[0], 4, 1, torch.float32,
                           scratch, flags, 256)
    next(g0)  # rank 0 published its ready flags and waits for its peers'
    assert events == []


def test_ring_ops_waits_for_the_ready_flag_before_writing() -> None:
    """A rank never copies into the next rank's scratch before that rank's ready store."""
    world, mem, events = 2, _Memory(), []
    scratch = [mem.alloc(256 * 2) for _ in range(world)]
    flags = [mem.alloc(ce.flag_bytes(world)) for _ in range(world)]

    class Logged(_HostBackend):
        def copy(self, dst, src, nbytes):
            events.append(("copy", dst))
            super().copy(dst, src, nbytes)

        def set_flag(self, ptr, value):
            events.append(("flag", ptr))
            super().set_flag(ptr, value)

    x = torch.ones(world * 4)
    inp = [mem.put(x) for _ in range(world)]
    out = [mem.alloc(x.numel() * 4) for _ in range(world)]
    g0 = ce.ring_ops(Logged(mem, []), world, 0, 1, inp[0], out[0], 8, 2, torch.float32,
                     scratch, flags, 256)
    next(g0)  # rank 0 has published its ready flag and is waiting on rank 1's
    ready = (2 * (world - 1)) * ce.FLAG_STRIDE_BYTES
    assert events == [("flag", flags[1] + ready)]


def _ptx(kernel, signature, constexprs=None):
    """The kernel compiled for sm_80 on the host (no GPU needed)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs or {})
    return triton.compile(src, target=GPUTarget("cuda", 80, 32)).asm["ptx"]


WAIT_SIGNATURE = {"flag_ptr": "*i32", "value": "i32", "err_ptr": "*i32", "timeout_ns": "i64", "code": "i32"}


def test_flag_kernels_use_system_scope_release_acquire() -> None:
    """The flag store a peer reads over PCIe is a system-scope release (ordered after the copy-engine
    writes it announces); the wait reads with system-scope acquire."""
    set_flag, wait_flag, _, _ = ce._kernels()
    assert "st.release.sys.global.u32" in _ptx(set_flag, {"flag_ptr": "*i32", "value": "i32"})
    assert "ld.acquire.sys.global.u32" in _ptx(wait_flag, WAIT_SIGNATURE)


def test_flag_wait_is_bounded_and_reports_on_the_error_word() -> None:
    """The wait reads the global timer, leaves its loop on the bound, and ORs its code into the
    stream's sticky error word: it never spins forever and never traps."""
    _, wait_flag, _, _ = ce._kernels()
    ptx = _ptx(wait_flag, WAIT_SIGNATURE)
    assert "%globaltimer" in ptx
    assert "atom" in ptx and ".or" in ptx
    assert "trap" not in ptx
    assert 0 < ce.WAIT_TIMEOUT_NS <= 120_000_000_000


def test_owner_sum_kernel_compiles_and_accumulates_in_fp32() -> None:
    *_, sum_round = ce._kernels()
    sig = {"dst_ptr": "*bf16", **{f"p{i}": "*bf16" for i in range(8)}, "n": "i32"}
    ptx = _ptx(sum_round, sig, {"W": 4, "BLOCK": 1024})
    assert "add.f32" in ptx or "add.rn.f32" in ptx
    assert "cvt.rn.bf16.f32" in ptx


def test_step_error_word_names_the_ring_timeout() -> None:
    from vllm_sm80.distributed import step_errors

    for word, text in ((step_errors.CE_RING_TIMEOUT, "copy-engine ring"),
                       (1, "device step error word 0x1"),
                       (3, "copy-engine ring")):
        check = step_errors.ErrorCheck([torch.tensor(word, dtype=torch.int32)])
        with pytest.raises(ValueError, match=text):
            check.raise_if_failed()
    step_errors.ErrorCheck([torch.tensor(0, dtype=torch.int32)]).raise_if_failed()
