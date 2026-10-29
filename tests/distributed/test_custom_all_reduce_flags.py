# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the flags-in-data two-shot all-reduce
(vllm_sm80/distributed/custom_all_reduce_flags.py).

The bitwise and timing checks against cross_device_reduce_2stage at four ranks
need four peer-connected GPUs and run outside pytest. The two-rank bitwise check
against upstream's custom all-reduce needs two peer-connected sm_80 GPUs and
nvcc. Everything else runs without a GPU except the overrun test, which needs
one sm_80 GPU; the compile tests need nvcc.
"""

import ctypes
import logging
import os
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from tests.sm80.tuning_rows import use_rows
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm_sm80.distributed import communicator as comm
from vllm_sm80.distributed import custom_all_reduce_flags as flags

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
HAVE_NVCC = os.path.exists(NVCC)
CUOBJDUMP = os.path.join(os.path.dirname(NVCC), "cuobjdump")
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")


# ---------------------------------------------------------------- env and gate
def test_env_defaults(monkeypatch):
    monkeypatch.delenv("VLLM_SM80_ALLREDUCE", raising=False)
    assert envs.VLLM_SM80_ALLREDUCE is False
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE", "1")
    assert envs.VLLM_SM80_ALLREDUCE is True


def _current_config(monkeypatch, hidden):
    import vllm.config

    config = None
    if hidden is not None:
        config = SimpleNamespace(
            model_config=SimpleNamespace(get_hidden_size=lambda: hidden)
        )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)


def test_max_bytes_default_without_a_row(monkeypatch):
    use_rows(monkeypatch)
    _current_config(monkeypatch, 4096)
    assert comm.allreduce_max_bytes("flags_max_bytes", 4, 262144) == 262144


@pytest.mark.parametrize(
    "hidden,world_size,want",
    [(4096, 4, 262144), (5120, 4, 524288), (5120, 2, 65536), (None, 4, 65536)],
)
def test_max_bytes_keyed_by_world_size_and_hidden(
    monkeypatch, hidden, world_size, want
):
    use_rows(
        monkeypatch,
        ("allreduce", "flags_max_bytes", {"world_size": 4, "hidden": 5120}, 524288),
        ("allreduce", "flags_max_bytes", {}, 65536),
        ("allreduce", "flags_max_bytes", {"world_size": 4, "hidden": 4096}, 262144),
    )
    _current_config(monkeypatch, hidden)
    assert comm.allreduce_max_bytes("flags_max_bytes", world_size, 1) == want


def test_build_dir_follows_vllm_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    assert flags._build_dir() == str(tmp_path / "custom_all_reduce_flags")
    # A location only: VLLM_CACHE_ROOT is not a compile-cache factor.
    assert "VLLM_CACHE_ROOT" not in envs.compile_factors()


@pytest.mark.parametrize(
    "kw,expect",
    [
        (dict(), None),
        (dict(world_size=2), None),
        (dict(world_size=3), "world size 3"),
        (dict(world_size=6), "world size 6"),
        (dict(world_size=8), "world size 8"),
        (dict(same_node=False), "spans nodes"),
        (dict(capability=(9, 0)), "compute capability (9, 0)"),
        (dict(capability=(8, 6)), "compute capability (8, 6)"),
        (dict(capability=None), "compute capability None"),
        (dict(is_cuda=False), "not a CUDA platform"),
        (dict(pcie_p2p_allowed=False), "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P"),
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
    assert flags.largest_part_packs(8192, 4) == 128
    assert flags.largest_part_packs(16, 4) == 1  # one pack: rank 3 owns it
    assert flags.largest_part_packs(48, 4) == 3
    assert flags.largest_part_packs(80, 4) == 2  # 5 packs: 1,1,1,2
    assert flags.launch_blocks(16, 4) == 1
    assert flags.launch_blocks(8192, 4) == 1
    assert flags.launch_blocks(65536, 4) == 4
    assert flags.launch_blocks(262144, 4) == 16
    assert flags.launch_blocks(8 << 20, 4) == flags.MAX_BLOCKS
    # four ranks keep the buffer they had: 8 slots of the largest partition
    assert flags.stage_packs(262144, 4) == 8 * 4096


def test_sizes_and_launch_two_ranks():
    # 8 KiB = 512 packs; partitions 256 x 2
    assert flags.largest_part_packs(8192, 2) == 256
    assert flags.largest_part_packs(16, 2) == 1  # one pack: rank 1 owns it
    assert flags.largest_part_packs(48, 2) == 2  # 3 packs: 1,2
    assert flags.launch_blocks(16, 2) == 1
    assert flags.launch_blocks(8192, 2) == 1
    assert flags.launch_blocks(65536, 2) == 8
    assert flags.launch_blocks(262144, 2) == 32
    assert flags.launch_blocks(8 << 20, 2) == flags.MAX_BLOCKS
    # 2 reduce-scatter + 2 all-gather slots of 8192 packs: the same 1.5 MiB
    # (three stages) as at four ranks
    assert flags.stage_packs(262144, 2) == 4 * 8192
    assert 3 * flags.stage_packs(262144, 2) * 16 == 3 * flags.stage_packs(262144, 4) * 16
    for nbytes in range(16, 4096, 16):
        size = nbytes // 16
        parts = [size // 2, size - size // 2]
        assert sum(parts) == size and max(parts) == flags.largest_part_packs(nbytes, 2)


# ------------------------------------------------------------------- dispatch
class _CA(comm.Sm80CustomAllreduce):
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


def _fake_ca(monkeypatch, with_flags, world_size=4):
    ca = object.__new__(_CA)
    ca.disabled = False
    ca.world_size = world_size
    # Two PCIe ranks are not "fully connected"; upstream serves them anyway.
    ca.fully_connected = world_size != 2
    ca.max_size = 8192 * 1024
    ca._IS_CAPTURING = False
    ca.rank = 0
    ca._ptr = 1234
    ca.buffer_ptrs = [11, 12, 13, 14][:world_size]
    ca._flags = FakeFlags() if with_flags else None
    ops_calls = []

    def fake_all_reduce(ptr, inp, out, reg, reg_sz):
        ops_calls.append((ptr, inp.nbytes, reg, reg_sz))
        out.fill_(3)

    monkeypatch.setattr(car.ops, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(car.torch.cuda, "is_current_stream_capturing", lambda: False)
    return ca, ops_calls


@pytest.mark.parametrize("world_size", [2, 4])
def test_off_path_unchanged(monkeypatch, world_size):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=False, world_size=world_size)
    x = torch.zeros(4096, dtype=torch.bfloat16)
    out = ca.custom_all_reduce(x)
    assert ops_calls == [(1234, 8192, 11, 8192 * 1024)]  # staging buffer as before
    assert torch.all(out == 3)


@pytest.mark.parametrize("world_size", [2, 4])
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
def test_dispatch(monkeypatch, dtype, numel, flagged, world_size):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True, world_size=world_size)
    x = torch.zeros(numel, dtype=dtype)
    out = ca.custom_all_reduce(x)
    if flagged:
        assert ca._flags.calls == [x.nbytes] and not ops_calls
        assert torch.all(out == 7)
    else:
        assert not ca._flags.calls and len(ops_calls) == 1


def test_dispatch_four_ranks_deepseek_rows(monkeypatch):
    """DeepSeek-V4.1 TP4 messages (5120-wide bf16 rows: 32 rows at eight
    requests, 40 in the largest decode graph, 128 in the prefill replay tail)
    take the flags path whenever the allreduce/flags_max_bytes limit of the
    sm80 tuning table covers them: 256 KiB (the default) keeps 26 rows and
    more on upstream's kernel, 512 KiB (the served DeepSeek value) takes up
    to 51 rows, and 1.25 MiB takes the 128-row prefill tail."""
    for max_bytes, rows, flagged in (
        (262144, 25, True),
        (262144, 26, False),
        (262144, 32, False),
        (524288, 51, True),
        (524288, 52, False),
        (1310720, 32, True),
        (1310720, 40, True),
        (1310720, 128, True),
        (1310720, 129, False),
    ):
        ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
        ca._flags = FakeFlags(max_bytes)
        ca.custom_all_reduce(torch.zeros(rows, 5120, dtype=torch.bfloat16))
        assert bool(ca._flags.calls) is flagged and len(ops_calls) == (not flagged)


def test_dispatch_respects_should_custom_ar(monkeypatch):
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    ca.fully_connected = False  # incumbent refuses: the flags path must too
    assert ca.custom_all_reduce(torch.zeros(4096, dtype=torch.bfloat16)) is None
    assert not ca._flags.calls and not ops_calls


@pytest.mark.parametrize("numel", [4096, 32 * 4096 + 8])
def test_should_custom_ar_runs_once(monkeypatch, numel):
    """One check inside custom_all_reduce on the flags path and on the
    existing kernel's path (all_reduce makes the other one)."""
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    calls = []
    check = car.CustomAllreduce.should_custom_ar
    monkeypatch.setattr(
        car.CustomAllreduce, "should_custom_ar",
        lambda self, inp: calls.append(inp.nbytes) or check(self, inp),
    )
    ca.custom_all_reduce(torch.zeros(numel, dtype=torch.bfloat16))
    assert len(calls) == 1


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
    lg = logging.getLogger(comm.__name__)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    yield h
    lg.removeHandler(h)
    lg.setLevel(level)


def _init_ca(monkeypatch, world_size=4, capability=(8, 0), p2p=True, load_ok=True):
    import vllm_sm80.distributed.pcie_p2p as pcie_p2p

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
    monkeypatch.setattr(pcie_p2p, "pcie_p2p_custom_allreduce_allowed", lambda: p2p)

    def load():
        if not load_ok:
            raise RuntimeError("no compiler")

    monkeypatch.setattr(flags, "load_module", load)
    monkeypatch.setattr(car, "_all_ranks_true", lambda group, v: v)
    made = []

    class FakeFA:
        def __init__(self, group, device, rank, world_size, max_bytes, create, free):
            self.world_size = world_size
            self.max_bytes = max_bytes
            self.limit_ns = int(60e9)
            made.append(self)

    monkeypatch.setattr(flags, "FlagsAllreduce", FakeFA)
    return ca, made


@pytest.mark.parametrize("world_size", [2, 4])
def test_init_on_banner(monkeypatch, log_capture, world_size):
    # distinct sizes per case: info_once would drop a repeated banner
    max_bytes = 65536 * world_size
    use_rows(
        monkeypatch,
        ("allreduce", "flags_max_bytes", {"world_size": world_size}, max_bytes),
    )
    ca, made = _init_ca(monkeypatch, world_size=world_size)
    ca._init_flags(same_node=True)
    assert ca._flags is made[0] and made[0].max_bytes == max_bytes
    assert made[0].world_size == world_size
    assert any(
        "flags-in-data two-shot kernel on" in m and str(max_bytes) in m and "60 s" in m
        for m in log_capture.messages
    ), log_capture.messages


@pytest.mark.parametrize(
    "kw,why",
    [
        (dict(world_size=8), "world size 8"),
        (dict(capability=(8, 6)), "compute capability (8, 6)"),
        (dict(p2p=False), "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P"),
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
    import inspect

    src = inspect.getsource(comm.Sm80CudaCommunicator.__init__)
    # the class swap and the setup are reached only under the flag
    assert re.search(
        r"if envs\.VLLM_SM80_ALLREDUCE and ca_comm is not None", src
    )
    assert "ca_comm.__class__ = Sm80CustomAllreduce" in src


def _bare_communicator(**attrs):
    c = object.__new__(comm.Sm80CudaCommunicator)
    for name in ("fi_pcie_ipc_ar_comm", "fi_ar_comm", "pynccl_comm", "qr_comm",
                 "aiter_ar_comm", "hostshm_comm", "ca_comm", "symm_mem_comm"):
        setattr(c, name, None)
    c.use_aiter_allreduce = False
    c._defer_backend_log = False
    c.world_size = 4
    for name, value in attrs.items():
        setattr(c, name, value)
    return c


@pytest.mark.parametrize("on", [False, True])
def test_backend_line_names_the_transport_switch(monkeypatch, log_capture, on):
    """The backend-selection line says when VLLM_SM80_ALLREDUCE picks the
    transport per message (one line per group: distinct names per case)."""
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE", "1" if on else "0")
    monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", raising=False)
    name = f"tp:transport-note-{on}"
    _bare_communicator(unique_name=name)._log_all_reduce_backend_selection()
    lines = [m for m in log_capture.messages if name in m]
    assert len(lines) == 1, log_capture.messages
    assert ("VLLM_SM80_ALLREDUCE=1: the fork's all-reduce transports" in lines[0]) is on


def test_switch_selects_the_fork_communicator(monkeypatch):
    monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", raising=False)
    monkeypatch.delenv("VLLM_SM80_ALLREDUCE", raising=False)
    assert comm.needs_fork_communicator() is False
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE", "1")
    assert comm.needs_fork_communicator() is True


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
// One element, the incumbent's arithmetic: fp32 sum from the owner onwards.
static __nv_bfloat16 sum4_owner_order(const __nv_bfloat16 v[4], int owner) {
  float acc = __bfloat162float(v[owner]);
  for (int j = 1; j < 4; j++) acc += __bfloat162float(v[(owner + j) % 4]);
  return __float2bfloat16(acc);
}
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
// Two ranks: the flags kernel adds x_owner + x_other over the partitions of
// part_len<2>; cross_device_reduce_1stage (upstream's kernel at world size 2)
// adds x_0 + x_1 for every element.
static __nv_bfloat16 sum2_owner_order(const __nv_bfloat16 v[2], int owner) {
  float acc = __bfloat162float(v[owner]);
  acc += __bfloat162float(v[owner ^ 1]);
  return __float2bfloat16(acc);
}
static __nv_bfloat16 one_stage2(const __nv_bfloat16 v[2]) {
  float acc = __bfloat162float(v[0]);
  acc += __bfloat162float(v[1]);
  return __float2bfloat16(acc);
}
int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  long checked = 0, control = 0;
  // Four ranks: every size up to 2048 packs, then strides up to 2 MiB, and the
  // 5120-wide bf16 messages (640 packs per row) of 24 to 40 decode rows and of
  // the 128-row prefill tail, which exceed the default flags limit and run on
  // the incumbent unless the tuning table's allreduce/flags_max_bytes is raised.
  auto check4 = [&](int size) {
    int sum = 0, L = part_len<4>(3, size);
    for (int q = 0; q < 4; q++) { sum += part_len<4>(q, size); if (part_len<4>(q, size) > L) fails++; }
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
  };
  for (int size = 1; size <= 131072; size += (size < 2048 ? 1 : (size < 20000 ? 97 : 1021)))
    check4(size);
  for (int rows : {24, 28, 32, 40, 51, 128}) check4(rows * 640);
  // Two ranks, finite and special values (signed zeros, infinities, subnormals,
  // the largest finite value); NaN results are the GPU test's.
  const uint16_t specials[] = {0x0000, 0x8000, 0x7F80, 0xFF80, 0x0001, 0x8001, 0x7F7F, 0xFF7F};
  long checked2 = 0;
  for (int size = 1; size <= 20000; size += (size < 2048 ? 1 : 97)) {
    int L = part_len<2>(1, size);
    if (part_len<2>(0, size) + L != size || part_len<2>(0, size) > L) fails++;
    for (int k = 0; k < 8; k++) {
      int idx = (int)(rng() % size);
      int part = size / 2;
      int owner = part == 0 ? 1 : (idx < part ? 0 : 1);
      __nv_bfloat16 x[2];
      float B = ldexpf(nd(rng), 20);
      x[0] = __float2bfloat16((k & 1) ? B : nd(rng));
      x[1] = __float2bfloat16((k & 1) ? -B * (1.f + ldexpf(nd(rng), -9)) : nd(rng));
      if (k == 6) x[1] = __float2bfloat16(-__bfloat162float(x[0]));
      if (k == 7) { x[0] = frombits(specials[rng() % 8]); x[1] = frombits(specials[rng() % 8]); }
      if (bits(sum2_owner_order(x, owner)) != bits(one_stage2(x))) fails++;
      checked2++;
    }
  }
  if (sanitize_word(0xFFFFFFFFu) != 0x7FFF7FFFu) fails++;
  if (sanitize_word(0x80000000u) != 0x80000000u) fails++;
  if (sanitize_word(0u) != 0u) fails++;
  // the replacement keeps both halves NaN
  if (!std::isnan(__bfloat162float(frombits(0x7FFF))) || !std::isnan(__bfloat162float(frombits(0xFFFF)))) fails++;
  printf("HOST_ORDER %s checked=%ld checked2=%ld control=%ld\n", fails ? "FAIL" : "PASS",
         checked, checked2, control);
  return fails || control == 0 || checked2 == 0;
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
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(flags, "_MODULE", None)
    m = flags.load_module()
    assert hasattr(m, "all_reduce") and hasattr(m, "fill_sentinel")



# ------------------------------------------------------------ wait overrun
def test_wait_limit_constant(monkeypatch):
    assert flags.WAIT_S == 60.0
    assert flags.wait_limit_ns() == 60 * 10**9
    # read at call time, so a test can shorten it
    monkeypatch.setattr(flags, "WAIT_S", 2.5)
    assert flags.wait_limit_ns() == 2_500_000_000


REC = [1, 2, 0, 41, 1536, 1, 3, 61_000_000_000, 5, 17, 60_000_000_000, 0, 0, 0, 0, 0]


def test_describe():
    assert flags.describe([0] * 16) is None
    msg = flags.describe(REC)
    for part in ("timed out on rank 2", "61.0 s", "limit 60.0 s", "call 41",
                 "stage 0", "1536 packs", "reduce-scatter wait for rank 3",
                 "block 5, thread 17", "trapped", "custom_all_reduce_flags.WAIT_S"):
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
        rec._stop.set()


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
        rec._stop.set()
    finally:
        lg.removeHandler(log_capture)
    assert order == [("report", 0), ("exit", 1, flags.EXIT_CODE)]
    assert any("timed out on rank 2" in x for x in log_capture.messages)


def test_all_reduce_passes_limit_and_record(monkeypatch):
    m = _FakeModule()
    fa = object.__new__(flags.FlagsAllreduce)
    fa.module, fa.ptrs, fa.rank, fa.stage_packs = m, [1, 2, 3, 4], 0, 32768
    fa.world_size = 4
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
        fa.errors._stop.set()


# ------------------------------------------------------------ GPU: forced overrun
OVERRUN_CHILD = r"""
import sys, torch
from vllm_sm80.distributed import custom_all_reduce_flags as f
f.WAIT_S = 2.0
m = f.load_module()
rec = f.ErrorRecord(m)
size_bytes = 8192
stage = f.stage_packs(262144, 4)
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
    env = dict(os.environ)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=env, timeout=600)
    took = time.time() - t0
    assert "RESULT RETURNED" not in r.stdout
    assert r.returncode != 0
    assert "flags-in-data wait timed out on rank 1" in r.stderr, r.stderr[-2000:]
    assert "limit 2.0 s" in r.stderr and "wait for rank 0" in r.stderr
    print("child exit", r.returncode, "after %.1f s" % took)


# ------------------------------------------------------- GPU: two-rank bitwise
def _tp2_inputs(nbytes: int, seed: int) -> torch.Tensor:
    """Both ranks' bf16 inputs, shape (2, n): normals over 2^-20..2^20, exactly
    cancelling pairs, and special words (signed zeros, infinities, subnormals,
    NaNs including 0xFFFF halves and whole sentinel words)."""
    g = torch.Generator().manual_seed(seed)
    n = nbytes // 2
    scale = torch.exp2(torch.randint(-20, 21, (2, n), generator=g).float())
    x = (torch.randn(2, n, generator=g) * scale).bfloat16()
    cancel = torch.rand(n, generator=g) < 0.1
    x[1][cancel] = -x[0][cancel]
    special = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x0001, 0x8001, 0x7FC0, 0xFFC1, 0xFFFF,
               0x7FFF]
    special = torch.tensor([s - 65536 if s >= 32768 else s for s in special],
                           dtype=torch.int16)
    bits = x.view(torch.int16)
    pick = torch.rand(2, n, generator=g) < 0.02
    bits[pick] = special[torch.randint(0, len(special), (2, n), generator=g)[pick]]
    x.view(torch.int32)[0, ::97] = -1  # sentinel words on rank 0
    x.view(torch.int32)[1, 5::101] = -1  # and on rank 1
    return x


@pytest.mark.skipif(
    not HAVE_NVCC
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or any(torch.cuda.get_device_capability(d) != (8, 0) for d in range(2)),
    reason="needs nvcc and two sm_80 GPUs",
)
@pytest.mark.parametrize("algo", ["", "2stage"])
def test_gpu_two_ranks_bitwise_vs_custom_allreduce(monkeypatch, algo):
    """One process drives both ranks (cards 0 and 1, peer access on): the flags
    kernel at two ranks against upstream's custom all-reduce with its default
    kernel at world size 2 (one-stage) and with the two-stage kernel forced.
    Every size runs twice so that each of the three stages is reused."""
    from vllm import _custom_ops as ops

    if not (torch.cuda.can_device_access_peer(0, 1)
            and torch.cuda.can_device_access_peer(1, 0)):
        pytest.skip("cards 0 and 1 have no peer access")
    if algo:
        monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", algo)
    else:
        monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
    monkeypatch.setattr(flags, "WAIT_S", 20.0)
    devs = [torch.device("cuda", r) for r in range(2)]
    # a device-to-device copy makes torch enable peer access, one direction each
    for a, b in ((0, 1), (1, 0)):
        torch.zeros(1, device=devs[b]).copy_(torch.zeros(1, device=devs[a]))

    max_bytes = 262144
    reg_bytes = 1 << 20
    meta = [torch.zeros(ops.meta_size() + reg_bytes, dtype=torch.uint8, device=d)
            for d in devs]
    reg = [torch.empty(reg_bytes, dtype=torch.uint8, device=d) for d in devs]
    rank_data = [torch.empty(1 << 20, dtype=torch.uint8, device=d) for d in devs]
    fa = []
    for r, d in enumerate(devs):
        with torch.cuda.device(d):
            fa.append(ops.init_custom_ar([t.data_ptr() for t in meta], rank_data[r],
                                         r, True))
            ops.register_buffer(fa[r], [t.data_ptr() for t in reg])

    m = flags.load_module()
    sp = flags.stage_packs(max_bytes, 2)
    nbuf = 3 * sp * 16
    fbuf = [torch.empty(nbuf, dtype=torch.uint8, device=d) for d in devs]
    epoch = [torch.zeros(flags.EPOCH_WORDS, dtype=torch.int32, device=d) for d in devs]
    errors = []
    recs = []
    for r, d in enumerate(devs):
        with torch.cuda.device(d):
            m.fill_sentinel(fbuf[r].data_ptr(), nbuf)
            recs.append(flags.ErrorRecord(m, on_error=errors.append,
                                          exit_fn=lambda code: None))

    def run(x, launch):
        outs = [torch.empty_like(x[r]) for r in range(2)]
        for r, d in enumerate(devs):  # both launched before either is awaited
            with torch.cuda.device(d):
                launch(r, x[r], outs[r])
        for d in devs:
            torch.cuda.synchronize(d)
        return [o.view(torch.int16).cpu() for o in outs]

    def upstream(r, inp, out):
        ops.all_reduce(fa[r], inp, out, reg[r].data_ptr(), reg_bytes)

    def flagged(r, inp, out):
        m.all_reduce(inp, out, [t.data_ptr() for t in fbuf], epoch[r].data_ptr(), r,
                     sp, flags.THREADS, flags.launch_blocks(inp.nbytes, 2),
                     flags.wait_limit_ns(), recs[r].dev_ptr)

    sizes = [16, 48, 4096, 5120, 20480, 24592, 65536, 163840, 262144]
    try:
        for call, nbytes in enumerate(sizes * 2):
            host = _tp2_inputs(nbytes, seed=call)
            x = [host[r].to(devs[r]) for r in range(2)]
            want = run(x, upstream)
            got = run(x, flagged)
            assert not errors, errors
            assert torch.equal(want[0], want[1]), nbytes
            for r in range(2):
                diff = (got[r] != want[r]).nonzero()
                assert diff.numel() == 0, (nbytes, r, diff[:8].flatten().tolist())
    finally:
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                ops.dispose(fa[r])
        for rec in recs:
            rec._stop.set()
