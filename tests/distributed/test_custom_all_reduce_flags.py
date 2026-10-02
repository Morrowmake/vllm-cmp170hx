# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the flags-in-data two-shot all-reduce
(vllm/distributed/device_communicators/custom_all_reduce_flags.py).

The bitwise and timing checks against cross_device_reduce_2stage need four
peer-connected GPUs and run outside pytest; everything here runs without a GPU.
"""

import ctypes
import logging
import os
import re
import shutil
import subprocess

import pytest
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators import custom_all_reduce_flags as flags

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
HAVE_NVCC = os.path.exists(NVCC)
CUOBJDUMP = os.path.join(os.path.dirname(NVCC), "cuobjdump")
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")


# ---------------------------------------------------------------- env and gate
def test_env_defaults(monkeypatch):
    for name in (
        "VLLM_CUSTOM_ALLREDUCE_FLAGS",
        "VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES",
        "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_CUSTOM_ALLREDUCE_FLAGS is False
    assert envs.VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES == 262144
    assert envs.VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR is None
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_FLAGS", "1")
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES", "65536")
    assert envs.VLLM_CUSTOM_ALLREDUCE_FLAGS is True
    assert envs.VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES == 65536


def test_build_dir_not_hashed():
    assert "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR" not in envs.compile_factors()


@pytest.mark.parametrize(
    "kw,expect",
    [
        (dict(), None),
        (dict(world_size=2), "world size 2"),
        (dict(world_size=8), "world size 8"),
        (dict(same_node=False), "spans nodes"),
        (dict(capability=(9, 0)), "compute capability (9, 0)"),
        (dict(capability=(8, 6)), "compute capability (8, 6)"),
        (dict(capability=None), "compute capability None"),
        (dict(is_cuda=False), "not a CUDA platform"),
        (dict(pcie_p2p_allowed=False), "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE"),
    ],
)
def test_gate(kw, expect):
    args = dict(
        world_size=4,
        same_node=True,
        capability=(8, 0),
        is_cuda=True,
        pcie_p2p_allowed=True,
    )
    args.update(kw)
    reason = flags.gate_reason(**args)
    if expect is None:
        assert reason is None
    else:
        assert reason is not None and expect in reason


def test_sizes_and_launch():
    # one 4096-wide bf16 row = 8 KiB = 512 packs; partitions 128 x 4
    assert flags.largest_part_packs(8192) == 128
    assert flags.largest_part_packs(16) == 1  # one pack: rank 3 owns it
    assert flags.largest_part_packs(48) == 3
    assert flags.largest_part_packs(80) == 2  # 5 packs: 1,1,1,2
    assert flags.launch_blocks(16) == 1
    assert flags.launch_blocks(8192) == 1
    assert flags.launch_blocks(65536) == 4
    assert flags.launch_blocks(262144) == 16
    assert flags.launch_blocks(8 << 20) == flags.MAX_BLOCKS


# ------------------------------------------------------------------- dispatch
class _CA(car.CustomAllreduce):
    """Built with object.__new__ around fake state: nothing to dispose."""

    def __del__(self):
        pass


class FakeFlags:
    def __init__(self, max_bytes=262144):
        self.max_bytes = max_bytes
        self.calls = []

    eligible = flags.FlagsAllreduce.eligible

    def all_reduce(self, inp, out=None):
        self.calls.append(inp.nbytes)
        return torch.full_like(inp, 7)


def _fake_ca(monkeypatch, with_flags):
    ca = object.__new__(_CA)
    ca.disabled = False
    ca.world_size = 4
    ca.fully_connected = True
    ca.max_size = 8192 * 1024
    ca._IS_CAPTURING = False
    ca.rank = 0
    ca._ptr = 1234
    ca.buffer_ptrs = [11, 12, 13, 14]
    ca._flags = FakeFlags() if with_flags else None
    ops_calls = []

    def fake_all_reduce(ptr, inp, out, reg, reg_sz):
        ops_calls.append((ptr, inp.nbytes, reg, reg_sz))
        out.fill_(3)

    monkeypatch.setattr(car.ops, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(car.torch.cuda, "is_current_stream_capturing", lambda: False)
    return ca, ops_calls


def test_off_path_unchanged(monkeypatch):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=False)
    x = torch.zeros(4096, dtype=torch.bfloat16)
    out = ca.custom_all_reduce(x)
    assert ops_calls == [(1234, 8192, 11, 8192 * 1024)]  # staging buffer as before
    assert torch.all(out == 3)


@pytest.mark.parametrize(
    "dtype,numel,flagged",
    [
        (torch.bfloat16, 8, True),  # 16 bytes
        (torch.bfloat16, 4096, True),  # one row
        (torch.bfloat16, 32 * 4096, True),  # 256 KiB, the limit
        (torch.bfloat16, 32 * 4096 + 8, False),  # above the limit
        (torch.float16, 4096, False),
        (torch.float32, 4096, False),
    ],
)
def test_dispatch(monkeypatch, dtype, numel, flagged):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    x = torch.zeros(numel, dtype=dtype)
    out = ca.custom_all_reduce(x)
    if flagged:
        assert ca._flags.calls == [x.nbytes] and not ops_calls
        assert torch.all(out == 7)
    else:
        assert not ca._flags.calls and len(ops_calls) == 1


def test_dispatch_respects_should_custom_ar(monkeypatch):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    ca.fully_connected = False  # incumbent refuses: the flags path must too
    assert ca.custom_all_reduce(torch.zeros(4096, dtype=torch.bfloat16)) is None
    assert not ca._flags.calls and not ops_calls


def test_capture_paths(monkeypatch):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    ca._IS_CAPTURING = True
    x = torch.zeros(4096, dtype=torch.bfloat16)
    out = ca.custom_all_reduce(x)  # warm-up before capture: no collective
    assert out.shape == x.shape and not ca._flags.calls and not ops_calls
    monkeypatch.setattr(car.torch.cuda, "is_current_stream_capturing", lambda: True)
    ca.custom_all_reduce(x)
    assert ca._flags.calls == [8192] and not ops_calls  # no registration either


# ---------------------------------------------------------------- init + banner
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def log_capture():
    h = _Capture()
    lg = logging.getLogger(car.__name__)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    yield h
    lg.removeHandler(h)
    lg.setLevel(level)


def _init_ca(monkeypatch, world_size=4, capability=(8, 0), p2p=True, load_ok=True):
    import vllm.platforms.cuda as pcuda

    ca = object.__new__(_CA)
    ca._flags = None
    ca.world_size = world_size
    ca.rank = 0
    ca.group = object()
    ca.device = torch.device("cpu")
    monkeypatch.setattr(
        car.current_platform, "get_device_capability", lambda *a, **k: capability
    )
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(pcuda, "pcie_p2p_custom_allreduce_allowed", lambda: p2p)

    def load():
        if not load_ok:
            raise RuntimeError("no compiler")

    monkeypatch.setattr(flags, "load_module", load)
    monkeypatch.setattr(car, "_all_ranks_true", lambda group, v: v)
    made = []

    class FakeFA:
        def __init__(self, group, device, rank, max_bytes, create, free):
            self.max_bytes = max_bytes
            self.limit_ns = int(60e9)
            made.append(self)

    monkeypatch.setattr(flags, "FlagsAllreduce", FakeFA)
    return ca, made


def test_init_on_banner(monkeypatch, log_capture):
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES", "131072")
    ca, made = _init_ca(monkeypatch)
    ca._init_flags(same_node=True)
    assert ca._flags is made[0] and made[0].max_bytes == 131072
    assert any(
        "flags-in-data two-shot kernel on" in m and "131072" in m and "60 s" in m
        for m in log_capture.messages
    ), log_capture.messages


@pytest.mark.parametrize(
    "kw,why",
    [
        (dict(world_size=2), "world size 2"),
        (dict(capability=(8, 6)), "compute capability (8, 6)"),
        (dict(p2p=False), "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE"),
        (dict(load_ok=False), "did not build on every rank"),
    ],
)
def test_init_gate_closed_banner(monkeypatch, log_capture, kw, why):
    ca, made = _init_ca(monkeypatch, **kw)
    ca._init_flags(same_node=True)
    assert ca._flags is None and not made
    assert any(
        "flags-in-data path requested" in m and why in m for m in log_capture.messages
    ), log_capture.messages


def test_init_not_called_when_off():
    src = open(car.__file__).read()
    # the setup is reached only under the flag
    assert re.search(
        r"if envs\.VLLM_CUSTOM_ALLREDUCE_FLAGS:\n\s+self\._init_flags\(same_node\)", src
    )


# ------------------------------------------------------------ kernel source
@pytest.mark.skipif(not HAVE_NVCC, reason="nvcc not available")
def test_kernel_compiles_without_local_memory(tmp_path):
    src = tmp_path / "k.cu"
    src.write_text(flags.KERNEL_SRC)
    cubin = tmp_path / "k.cubin"
    r = subprocess.run(
        [NVCC, "-arch=sm_80", "-O3", "-std=c++17", "-Xptxas", "-v", "-cubin",
         "-o", str(cubin), str(src)],
        capture_output=True, text=True, env=CPU_ENV,
    )
    assert r.returncode == 0, r.stderr
    frames = re.findall(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        r.stderr,
    )
    assert frames and all(f == ("0", "0", "0") for f in frames), frames
    sass = subprocess.run(
        [CUOBJDUMP, "-sass", str(cubin)], capture_output=True, text=True, check=True
    ).stdout
    assert "STL" not in sass and "LDL" not in sass
    assert "STG.E.128.STRONG.SYS" in sass and "LDG.E.128.STRONG.SYS" in sass
    assert "BPT.TRAP" in sass  # an overrun traps; it never falls through


HOST_TEST = r"""
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include "k.cuh"
using namespace arflags;
static int fails = 0;
static uint16_t bits(__nv_bfloat16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }
static __nv_bfloat16 frombits(uint16_t b) { __nv_bfloat16 v; memcpy(&v, &b, 2); return v; }
// cross_device_reduce_2stage, written out from its loop bounds and packed_reduce
static __nv_bfloat16 incumbent(const __nv_bfloat16* x, int idx, int size) {
  int part = size / 4;
  for (int rank = 0; rank < 4; rank++) {
    int start = rank * part, end = rank == 3 ? size : start + part;
    if (idx >= start && idx < end) {
      float acc = __bfloat162float(x[rank]);
      for (int i = 1; i < 4; i++) acc += __bfloat162float(x[(rank + i) % 4]);
      return __float2bfloat16(acc);
    }
  }
  return frombits(0xdead);
}
int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  long checked = 0, control = 0;
  for (int size = 1; size <= 20000; size += (size < 2048 ? 1 : 97)) {
    int sum = 0, L = part_len(3, size);
    for (int q = 0; q < 4; q++) { sum += part_len(q, size); if (part_len(q, size) > L) fails++; }
    if (sum != size) fails++;
    for (int k = 0; k < 8; k++) {
      int idx = (int)(rng() % size);
      int part = size / 4;
      int owner = part == 0 ? 3 : (idx / part < 3 ? idx / part : 3);
      __nv_bfloat16 x[4];
      float B = ldexpf(nd(rng), 20);
      for (int r = 0; r < 4; r++) x[r] = __float2bfloat16(nd(rng));
      x[k % 4] = __float2bfloat16(B);
      x[(k + 1) % 4] = __float2bfloat16(-B);
      if (bits(sum4_owner_order(x, owner)) != bits(incumbent(x, idx, size))) fails++;
      if (bits(sum4_owner_order(x, 0)) != bits(incumbent(x, idx, size))) control++;
      checked++;
    }
  }
  if (sanitize_word(0xFFFFFFFFu) != 0x7FFF7FFFu) fails++;
  if (sanitize_word(0x80000000u) != 0x80000000u) fails++;
  if (sanitize_word(0u) != 0u) fails++;
  // the replacement keeps both halves NaN
  if (!std::isnan(__bfloat162float(frombits(0x7FFF))) || !std::isnan(__bfloat162float(frombits(0xFFFF)))) fails++;
  printf("HOST_ORDER %s checked=%ld control=%ld\n", fails ? "FAIL" : "PASS", checked, control);
  return fails || control == 0;
}
"""


@pytest.mark.skipif(not HAVE_NVCC, reason="nvcc not available")
def test_host_order_matches_incumbent(tmp_path):
    (tmp_path / "k.cuh").write_text(flags.KERNEL_SRC)
    (tmp_path / "t.cu").write_text(HOST_TEST)
    exe = tmp_path / "t"
    r = subprocess.run(
        [NVCC, "-arch=sm_80", "-std=c++17", "-O2", "-I", str(tmp_path),
         str(tmp_path / "t.cu"), "-o", str(exe)],
        capture_output=True, text=True, env=CPU_ENV,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=CPU_ENV)
    assert r.returncode == 0 and "HOST_ORDER PASS" in r.stdout, r.stdout


@pytest.mark.skipif(
    not HAVE_NVCC or os.environ.get("VLLM_TEST_FLAGS_JIT", "0") != "1",
    reason="JIT build test: set VLLM_TEST_FLAGS_JIT=1 (needs nvcc and ninja)",
)
def test_jit_build(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(flags, "_MODULE", None)
    m = flags.load_module()
    assert hasattr(m, "all_reduce") and hasattr(m, "fill_sentinel")



# ------------------------------------------------------------ wait overrun
def test_wait_limit_env(monkeypatch):
    monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S", raising=False)
    assert envs.VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S == 60.0
    assert flags.wait_limit_ns() == 60 * 10**9
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S", "2.5")
    assert flags.wait_limit_ns() == 2_500_000_000
    for bad in ("0", "-1"):
        monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S", bad)
        with pytest.raises(ValueError):
            flags.wait_limit_ns()


def test_kernel_source_never_falls_through():
    src = flags.KERNEL_SRC
    assert "spin_expired" not in src and "clock64" not in src
    body = src[src.index("void wait_failed"):]
    body = body[: body.index("\n}\n")]
    assert body.rstrip().endswith("__trap();")
    # the record is published with a system-scope release before anyone traps,
    # and the trap waits (bounded) for the host acknowledgement
    assert "st_release_sys_u64(c.err, 1)" in body and "c.err + 11" in body
    assert "*published == 0u" in body
    # both waits call it on overrun, with their phase
    assert "wait_failed(c, 1, peer, waited)" in src
    assert "wait_failed(c, 2, peer, waited)" in src


REC = [1, 2, 0, 41, 1536, 1, 3, 61_000_000_000, 5, 17, 60_000_000_000, 0, 0, 0, 0, 0]


def test_describe():
    assert flags.describe([0] * 16) is None
    msg = flags.describe(REC)
    for part in ("timed out on rank 2", "61.0 s", "limit 60.0 s", "call 41",
                 "stage 0", "1536 packs", "reduce-scatter wait for rank 3",
                 "block 5, thread 17", "trapped", "VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S"):
        assert part in msg, (part, msg)
    assert "all-gather" in flags.describe(REC[:5] + [2] + REC[6:])


class _FakeModule:
    """alloc_error_record backed by a plain host buffer."""

    def __init__(self):
        self.buf = (ctypes.c_int64 * 16)()
        self.calls = []

    def alloc_error_record(self):
        a = ctypes.addressof(self.buf)
        return [a, a + 1]

    def all_reduce(self, *args):
        self.calls.append(args)


def test_error_record_watcher_reports_once():
    import time

    m = _FakeModule()
    seen = []
    exits = []
    rec = flags.ErrorRecord(m, poll_s=0.01, on_error=seen.append, exit_fn=exits.append)
    try:
        time.sleep(0.05)
        assert seen == []
        for i, v in enumerate(REC):
            m.buf[i] = v
        deadline = time.time() + 5
        while not seen and time.time() < deadline:
            time.sleep(0.01)
        assert len(seen) == 1 and "timed out on rank 2" in seen[0]
        assert m.buf[flags.ACK_WORD] == 1  # acknowledged after reporting
        assert exits == [flags.EXIT_CODE]
        rec.check()  # reported once only
        assert len(seen) == 1 and exits == [flags.EXIT_CODE]
    finally:
        rec.stop()


def test_report_logs_before_ack_and_exit(log_capture):
    """Order matters: message out, then acknowledgement, then exit."""
    m = _FakeModule()
    order = []
    lg = logging.getLogger(flags.__name__)
    lg.addHandler(log_capture)
    try:
        def report(msg):
            flags._report(msg)
            order.append(("report", m.buf[flags.ACK_WORD]))

        rec = flags.ErrorRecord(
            m, poll_s=60, on_error=report,
            exit_fn=lambda c: order.append(("exit", m.buf[flags.ACK_WORD], c)))
        for i, v in enumerate(REC):
            m.buf[i] = v
        rec.check()
        rec.stop()
    finally:
        lg.removeHandler(log_capture)
    assert order == [("report", 0), ("exit", 1, flags.EXIT_CODE)]
    assert any("timed out on rank 2" in x for x in log_capture.messages)


def test_all_reduce_passes_limit_and_record(monkeypatch):
    m = _FakeModule()
    fa = object.__new__(flags.FlagsAllreduce)
    fa.module, fa.ptrs, fa.rank, fa.stage_packs = m, [1, 2, 3, 4], 0, 32768
    fa.epoch = torch.zeros(16, dtype=torch.int32)
    fa.limit_ns = 1234
    seen = []
    fa.errors = flags.ErrorRecord(m, poll_s=60, on_error=seen.append,
                                  exit_fn=lambda c: None)
    fa._calls = 0
    monkeypatch.setattr(flags.torch.cuda, "is_current_stream_capturing", lambda: False)
    try:
        x = torch.zeros(4096, dtype=torch.bfloat16)
        fa.all_reduce(x)
        assert m.calls[0][-2:] == (1234, fa.errors.dev_ptr)
        m.buf[0], m.buf[1] = 1, 3  # a record appears: the periodic check reports it
        for _ in range(flags.CHECK_EVERY - 1):
            fa.all_reduce(x)
        assert len(seen) == 1 and "rank 3" in seen[0]
    finally:
        fa.errors.stop()


# CPU model of the overrun path: four ranks run the two-shot protocol with a
# global clock; a rank that never pushes (dead) or pushes late (slow) is
# modelled. Every wait is bounded by LIMIT ticks and an overrun ends that rank
# (trap). Claims: with a dead rank, every live rank traps and no rank returns
# from the call; with a late rank inside the limit, every rank returns the
# exact sums and none traps.
LIMIT = 50


def _model(dead=None, late=None, delay=0, size=7, seed=0):
    import random

    rng = random.Random(seed)
    R = 4
    part = size // R
    L = size - 3 * part
    owner_len = [part, part, part, L]
    x = [[rng.randint(-9, 9) for _ in range(size)] for _ in range(R)]
    rs = [dict() for _ in range(R)]  # (src, idx) -> value
    ag = [dict() for _ in range(R)]  # (q, idx) -> value
    out = [[None] * size for _ in range(R)]
    state = {r: "push" for r in range(R)}
    waited = {r: 0 for r in range(R)}
    clock = 0
    trapped, returned = set(), set()
    inflight = []

    def start_time(r):
        return delay if r == late else 0

    while len(trapped) + len(returned) < R - (1 if dead is not None else 0):
        clock += 1
        if clock > 10_000:
            raise AssertionError("model did not settle")
        for r in range(R):
            if r == dead or r in trapped or r in returned or clock < start_time(r):
                continue
            if state[r] == "push":
                for q in range(R):
                    if q != r:
                        for i in range(owner_len[q]):
                            inflight.append((clock + rng.randint(1, 5), "rs", q, (r, i), x[r][q * part + i]))
                state[r] = "reduce"
            elif state[r] == "reduce":
                need = [(s, i) for s in range(R) if s != r for i in range(owner_len[r])]
                if all(k in rs[r] for k in need):
                    for i in range(owner_len[r]):
                        v = sum(x[r][r * part + i] if s == r else rs[r][(s, i)] for s in range(R))
                        out[r][r * part + i] = v
                        for q in range(R):
                            if q != r:
                                inflight.append((clock + rng.randint(1, 5), "ag", q, (r, i), v))
                    state[r], waited[r] = "gather", 0
                else:
                    waited[r] += 1
                    if waited[r] > LIMIT:
                        trapped.add(r)
            elif state[r] == "gather":
                need = [(q, i) for q in range(R) if q != r for i in range(owner_len[q])]
                if all(k in ag[r] for k in need):
                    for q, i in need:
                        out[r][q * part + i] = ag[r][(q, i)]
                    returned.add(r)
                else:
                    waited[r] += 1
                    if waited[r] > LIMIT:
                        trapped.add(r)
        for item in [t for t in inflight if t[0] <= clock]:
            inflight.remove(item)
            _, kind, dst, key, v = item
            (rs if kind == "rs" else ag)[dst][key] = v
    exact = [sum(x[r][j] for r in range(R)) for j in range(size)]
    return trapped, returned, out, exact


@pytest.mark.parametrize("dead", [0, 1, 2, 3])
@pytest.mark.parametrize("size", [1, 5, 8, 13])
def test_model_dead_rank_traps_everyone(dead, size):
    for seed in range(20):
        trapped, returned, _, _ = _model(dead=dead, size=size, seed=seed)
        assert returned == set(), "a rank returned without a peer's data"
        assert trapped == set(range(4)) - {dead}


@pytest.mark.parametrize("late", [0, 3])
def test_model_late_rank_within_limit_is_exact(late):
    for seed in range(20):
        trapped, returned, out, exact = _model(late=late, delay=LIMIT // 2, seed=seed)
        assert trapped == set() and returned == set(range(4))
        assert all(o == exact for o in out)


def test_model_late_rank_beyond_limit_traps():
    trapped, returned, _, _ = _model(late=2, delay=4 * LIMIT)
    assert returned == set() and {0, 1, 3} <= trapped


# ------------------------------------------------------------ GPU: forced overrun
OVERRUN_CHILD = r"""
import sys, torch
from vllm.distributed.device_communicators import custom_all_reduce_flags as f
m = f.load_module()
rec = f.ErrorRecord(m)
size_bytes = 8192
stage = 8 * f.largest_part_packs(262144)
bufs = [torch.full((3 * stage * 16,), 0xFF, dtype=torch.uint8, device="cuda") for _ in range(4)]
epoch = torch.zeros(16, dtype=torch.int32, device="cuda")
x = torch.randn(size_bytes // 2, device="cuda").bfloat16()
out = torch.empty_like(x)
# one card plays rank 1 of 4; ranks 0, 2, 3 never push: every wait must overrun
m.all_reduce(x, out, [b.data_ptr() for b in bufs], epoch.data_ptr(), 1, stage,
             256, 36, f.wait_limit_ns(), rec.dev_ptr)
torch.cuda.synchronize()
print("RESULT RETURNED", out.float().sum().item(), flush=True)
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 GPU",
)
def test_gpu_overrun_kills_process_with_message(tmp_path):
    import sys
    import time

    script = tmp_path / "child.py"
    script.write_text(OVERRUN_CHILD)
    env = dict(os.environ, VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S="2")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=env, timeout=600)
    took = time.time() - t0
    assert "RESULT RETURNED" not in r.stdout
    assert r.returncode != 0
    assert "flags-in-data wait timed out on rank 1" in r.stderr, r.stderr[-2000:]
    assert "limit 2.0 s" in r.stderr and "wait for rank 0" in r.stderr
    print("child exit", r.returncode, "after %.1f s" % took)
