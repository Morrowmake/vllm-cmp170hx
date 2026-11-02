# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the flags-in-data two-shot all-reduce
(vllm_sm80/distributed/custom_all_reduce_flags.py).

The bitwise and timing checks against cross_device_reduce_2stage at four ranks
need four peer-connected GPUs and run outside pytest. The two-rank bitwise check
against upstream's custom all-reduce needs two peer-connected sm_80 GPUs and
nvcc. Everything else runs without a GPU except the overrun test, which needs
one sm_80 GPU; the compile tests need nvcc. VLLM_SM80_ALLREDUCE_COMPRESS (the
compressed wire format v2, the one-shot kernel): dispatch, sizes and a
lossless round trip of the format's codec (host C++ compiler) here; its
four-rank bitwise check is test_custom_all_reduce_flags_z_gpu.py.
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
    # the one-shot: one thread per pack of the whole message
    assert flags.launch_blocks(16, 4, one_shot=True) == 1
    assert flags.launch_blocks(4096, 4, one_shot=True) == 1
    assert flags.launch_blocks(8192, 4, one_shot=True) == 2
    assert flags.launch_blocks(8 << 20, 4, one_shot=True) == flags.MAX_BLOCKS
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
    """The flags-in-data FlagsAllreduce: two-shot calls (``calls``, writes 7s)
    and one-shot calls (``one_shot_calls``, writes 5s)."""

    def __init__(self, max_bytes=262144):
        self.max_bytes = max_bytes
        self.calls = []
        self.one_shot_calls = []

    eligible = flags.FlagsAllreduce.eligible

    def all_reduce(self, inp, out=None, one_shot=False):
        if out is None:
            out = torch.empty_like(inp)
        if one_shot:
            self.one_shot_calls.append(inp.nbytes)
            return out.fill_(5)
        self.calls.append(inp.nbytes)
        return out.fill_(7)


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


# ------------------------------------------- dispatch: compressed wire format
class FakeZ(FakeFlags):
    """The compressed kernel's FlagsAllreduce: records calls, writes 9s."""

    def __init__(self, group, device, rank, world_size, max_bytes, create, free,
                 compressed=False):
        super().__init__(max_bytes)
        assert compressed and world_size == 4
        self.buffer_bytes = 3 * flags.z_stage_words(max_bytes) * 4

    def all_reduce(self, inp, out=None):
        self.calls.append(inp.nbytes)
        if out is None:
            return torch.full_like(inp, 9)
        return out.fill_(9)


def _z_ca(monkeypatch, algo="2stage", flags_max=524288, hot=False, rows=()):
    """A four-rank custom all-reduce with the flags path (FakeFlags) and the
    compressed one (FakeZ) set up by _init_flags_z, upstream's algorithm
    choice ``algo`` ("" = unset) and these tuning rows."""
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    ca._flags = FakeFlags(flags_max)
    ca.group = object()
    ca.device = torch.device("cpu")
    if algo:
        monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", algo)
    else:
        monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
    use_rows(monkeypatch, *rows)
    _current_config(monkeypatch, 5120)
    ca._init_flags_z(SimpleNamespace(Z_WORLD_SIZE=4, FlagsAllreduce=FakeZ), hot)
    return ca, ops_calls


# (message bytes, path) at the served DeepSeek-V4.1 settings: flags limit
# 512 KiB, VLLM_CUSTOM_ALLREDUCE_ALGO=2stage, one-shot up to 8 KiB, raw
# cut-off 32 KiB (defaults).
Z_SERVED = [
    (16, "one_shot"),
    (640, "one_shot"),  # a draft slot sum (c1: 320 lanes)
    (8192, "one_shot"),  # the one-shot limit
    (8208, "flags"),
    (32768, "flags"),  # the cut-off: uncompressed
    (32784, "z"),
    (8 * 10240, "z"),  # 8 rows of 5120
    (48 * 10240, "z"),  # the largest decode message
    (524288, "z"),  # the flags limit
    (524304, "z"),  # cross_device_reduce_2stage's range from here
    (128 * 10240, "z"),  # the prefill replay tail
    (8 * 1024 * 1024 - 16, "z"),  # the largest the custom all-reduce takes
    (8 * 1024 * 1024, None),  # upstream refuses it (should_custom_ar)
]


@pytest.mark.parametrize("nbytes,path", Z_SERVED)
def test_z_dispatch_served(monkeypatch, nbytes, path):
    ca, ops_calls = _z_ca(monkeypatch)
    assert ca._flags_z.max_bytes == 8 * 1024 * 1024 - 16
    out = ca.custom_all_reduce(torch.zeros(nbytes // 2, dtype=torch.bfloat16))
    took = {
        "one_shot": bool(ca._flags.one_shot_calls),
        "flags": bool(ca._flags.calls),
        "z": bool(ca._flags_z.calls),
        "upstream": bool(ops_calls),
    }
    if path is None:
        assert out is None and not any(took.values())
    else:
        assert [k for k, v in took.items() if v] == [path]
        assert torch.all(out == {"one_shot": 5, "flags": 7, "z": 9}[path])


@pytest.mark.parametrize(
    "algo,flags_max,nbytes,path",
    [
        # unset: upstream runs the one-stage kernel (rank order) below 512 KiB
        ("", 262144, 262144, "z"),
        ("", 262144, 262160, "upstream"),
        ("", 262144, 524272, "upstream"),
        ("", 262144, 524288, "z"),
        ("", 524288, 524304, "z"),
        ("twoshot", 262144, 262160, "z"),
        # one-stage forced: only the flags kernel's range
        ("1stage", 524288, 524288, "z"),
        ("1stage", 524288, 524304, "upstream"),
        ("oneshot", 524288, 128 * 10240, "upstream"),
        # the one-shot gives the two-shot's bits: under every algorithm
        ("1stage", 524288, 4096, "one_shot"),
        ("", 262144, 8192, "one_shot"),
    ],
)
def test_z_replaces_only_owner_order_kernels(monkeypatch, algo, flags_max, nbytes, path):
    ca, ops_calls = _z_ca(monkeypatch, algo=algo, flags_max=flags_max)
    ca.custom_all_reduce(torch.zeros(nbytes // 2, dtype=torch.bfloat16))
    assert bool(ca._flags_z.calls) is (path == "z")
    assert bool(ca._flags.one_shot_calls) is (path == "one_shot")
    assert bool(ops_calls) is (path == "upstream") and not ca._flags.calls


def test_z_one_stage_buffer_is_the_flags_limit(monkeypatch):
    ca, _ = _z_ca(monkeypatch, algo="1stage")
    assert ca._flags_z.max_bytes == 524288


def test_one_shot_cut_off_from_the_tuning_table(monkeypatch):
    ca, ops_calls = _z_ca(
        monkeypatch,
        rows=[("allreduce", "flags_one_shot_max_bytes", {"world_size": 4, "hidden": 5120}, 4096)],
    )
    assert ca._flags_one_shot_max_bytes == 4096
    ca.custom_all_reduce(torch.zeros(2048, dtype=torch.bfloat16))
    ca.custom_all_reduce(torch.zeros(2056, dtype=torch.bfloat16))
    assert ca._flags.one_shot_calls == [4096] and ca._flags.calls == [4112] and not ops_calls


def test_one_shot_needs_the_switch(monkeypatch):
    """Without the compressed kernel (switch off, or two ranks) no message
    takes the one-shot."""
    ca, ops_calls = _fake_ca(monkeypatch, with_flags=True)
    ca.custom_all_reduce(torch.zeros(8, dtype=torch.bfloat16))
    assert ca._flags.calls == [16] and not ca._flags.one_shot_calls


def test_z_raw_cut_off_from_the_tuning_table(monkeypatch):
    ca, ops_calls = _z_ca(
        monkeypatch,
        rows=[("allreduce", "flags_raw_max_bytes", {"world_size": 4, "hidden": 5120}, 65536)],
    )
    assert ca._flags_z_min_bytes == 65536
    ca.custom_all_reduce(torch.zeros(32768, dtype=torch.bfloat16))  # 64 KiB
    assert ca._flags.calls == [65536] and not ca._flags_z.calls
    ca.custom_all_reduce(torch.zeros(32776, dtype=torch.bfloat16))
    assert ca._flags_z.calls == [65552]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_z_bf16_only(monkeypatch, dtype):
    ca, ops_calls = _z_ca(monkeypatch)
    ca.custom_all_reduce(torch.zeros(65536, dtype=dtype))
    assert not ca._flags_z.calls and not ca._flags.calls and len(ops_calls) == 1


def test_z_off_while_the_flags_path_is(monkeypatch):
    """dev_ab.py's custom_allreduce_flags toggle sets _flags to None: every
    message then takes upstream's kernel, the compressed and one-shot paths
    included."""
    ca, ops_calls = _z_ca(monkeypatch)
    flags_z = ca._flags_z
    ca._flags = None
    for numel in (512, 8192, 65536, 128 * 5120):
        ca.custom_all_reduce(torch.zeros(numel, dtype=torch.bfloat16))
    assert not flags_z.calls and len(ops_calls) == 4


def test_z_respects_should_custom_ar_once(monkeypatch):
    ca, ops_calls = _z_ca(monkeypatch)
    calls = []
    check = car.CustomAllreduce.should_custom_ar
    monkeypatch.setattr(
        car.CustomAllreduce, "should_custom_ar",
        lambda self, inp: calls.append(inp.nbytes) or check(self, inp),
    )
    ca.custom_all_reduce(torch.zeros(65536, dtype=torch.bfloat16))
    assert calls == [131072] and ca._flags_z.calls == [131072]
    ca.fully_connected = False  # incumbent refuses: so does the compressed path
    assert ca.custom_all_reduce(torch.zeros(65536, dtype=torch.bfloat16)) is None
    assert ca._flags_z.calls == [131072] and not ops_calls


@pytest.mark.parametrize("numel,route", [(65536, "z"), (2048, "one_shot")])
def test_z_capture_paths(monkeypatch, numel, route):
    ca, ops_calls = _z_ca(monkeypatch)
    ca._IS_CAPTURING = True
    x = torch.zeros(numel, dtype=torch.bfloat16)
    out = ca.custom_all_reduce(x)  # warm-up before capture: no collective
    assert out.shape == x.shape and not ca._flags_z.calls and not ops_calls
    assert not ca._flags.one_shot_calls
    monkeypatch.setattr(car.torch.cuda, "is_current_stream_capturing", lambda: True)
    ca.custom_all_reduce(x)
    got = ca._flags_z.calls if route == "z" else ca._flags.one_shot_calls
    assert got == [2 * numel] and not ops_calls and not ca._flags.calls


@pytest.mark.parametrize(
    "nbytes,base,cand",
    [(65536, "flags", 9), (128 * 10240, "upstream", 9), (640, "flags", 5)],
)
def test_z_hot_pair_site(monkeypatch, nbytes, base, cand):
    """Carried by the hot pair, every message the compressed or the one-shot
    kernel takes is a pair site: the incumbent's kernel, then the candidate
    into its output; messages they do not take are not sites."""
    from vllm_sm80 import hot_swap

    ca, ops_calls = _z_ca(monkeypatch, hot=True)
    sites = []

    def pair(name, base_fn, cand_fn):
        out = base_fn()
        got = cand_fn(out)
        assert got.data_ptr() == out.data_ptr()
        sites.append(name)
        return out

    monkeypatch.setattr(hot_swap, "pair", pair)
    out = ca.custom_all_reduce(torch.zeros(nbytes // 2, dtype=torch.bfloat16))
    assert sites == [comm.COMPRESS_HOT_PAIR] and torch.all(out == cand)
    assert (ca._flags_z.calls if cand == 9 else ca._flags.one_shot_calls) == [nbytes]
    assert bool(ca._flags.calls) is (base == "flags")
    assert bool(ops_calls) is (base == "upstream")
    ca.custom_all_reduce(torch.zeros(8192, dtype=torch.bfloat16))  # 16 KiB: neither
    assert sites == [comm.COMPRESS_HOT_PAIR]


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
    ca.max_size = 8192 * 1024
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
        def __init__(self, group, device, rank, world_size, max_bytes, create, free,
                     compressed=False):
            self.world_size = world_size
            self.max_bytes = max_bytes
            self.compressed = compressed
            self.buffer_bytes = 3 * flags.z_stage_words(max_bytes) * 4 if compressed else 0
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


def test_init_compress_off_builds_nothing(monkeypatch):
    monkeypatch.delenv("VLLM_SM80_ALLREDUCE_COMPRESS", raising=False)
    monkeypatch.delenv("VLLM_HOT_SWAP", raising=False)
    ca, made = _init_ca(monkeypatch)
    ca._init_flags(same_node=True)
    assert len(made) == 1 and not made[0].compressed and ca._flags_z is None


@pytest.mark.parametrize("algo,want_max", [("2stage", 8192 * 1024 - 16), ("1stage", 524288)])
def test_init_compress_on_banner(monkeypatch, log_capture, algo, want_max):
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE_COMPRESS", "1")
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", algo)
    use_rows(
        monkeypatch,
        ("allreduce", "flags_max_bytes", {"world_size": 4}, 524288),
        # a cut-off no other test logs with: info_once drops a repeated banner
        ("allreduce", "flags_raw_max_bytes", {"world_size": 4}, 49152),
    )
    ca, made = _init_ca(monkeypatch)
    ca._init_flags(same_node=True)
    assert [fa.compressed for fa in made] == [False, True]
    assert ca._flags is made[0] and ca._flags_z is made[1] and not ca._flags_z_hot
    assert made[1].max_bytes == want_max and ca._flags_z_min_bytes == 49152
    assert ca._flags_one_shot_max_bytes == 8192
    assert any(
        "compressed flags kernel on (VLLM_SM80_ALLREDUCE_COMPRESS=1)" in m
        and f"to {want_max} bytes" in m and repr(algo) in m
        and "one-shot kernel for those up to 8192 bytes" in m
        for m in log_capture.messages
    ), log_capture.messages


def test_init_compress_two_ranks_off(monkeypatch, log_capture):
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE_COMPRESS", "1")
    ca, made = _init_ca(monkeypatch, world_size=2)
    ca._init_flags(same_node=True)
    assert len(made) == 1 and ca._flags_z is None
    assert any(
        "compressed flags kernel requested" in m and "world size 2" in m
        for m in log_capture.messages
    ), log_capture.messages


def test_init_compress_off_with_the_flags_path(monkeypatch, log_capture):
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE_COMPRESS", "1")
    ca, made = _init_ca(monkeypatch, capability=(8, 9))
    ca._init_flags(same_node=True)
    assert not made and ca._flags_z is None
    assert any(
        "compute capability (8, 9)" in m and "so is its compressed wire format" in m
        for m in log_capture.messages
    ), log_capture.messages


def test_init_compress_hot_pair(monkeypatch):
    """Listed in VLLM_HOT_SWAP with the switch off, the compressed kernel is
    built beside the flags one, and its calls become pair sites."""
    from vllm_sm80 import hot_swap

    monkeypatch.delenv("VLLM_SM80_ALLREDUCE_COMPRESS", raising=False)
    monkeypatch.setenv("VLLM_HOT_SWAP", comm.COMPRESS_HOT_PAIR)
    monkeypatch.setattr(
        hot_swap, "_STATE", hot_swap._HotState(listed=(comm.COMPRESS_HOT_PAIR,))
    )
    ca, made = _init_ca(monkeypatch)
    ca._init_flags(same_node=True)
    assert [fa.compressed for fa in made] == [False, True] and ca._flags_z_hot


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


# ------------------------------------------- compressed wire format on the CPU
# The codec's own source text (KERNEL_SRC: encode2, decode2 and their
# helpers) compiled for the host, its warp run as 32 threads: every shuffle, vote and reduction is
# a store, a barrier, a read and a barrier, __syncwarp a barrier, the warp's
# shared-memory scratch one array. store_block (a volatile PTX store) is
# replaced by a plain store into the slot; recv_blocks2's polling is the GPU
# tests'.
CXX = shutil.which("g++") or shutil.which("c++")

Z_CODEC_PRELUDE = r"""
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
using std::max;
using std::min;
#define AFD static inline
struct alignas(16) uint4 { uint32_t x, y, z, w; };
static inline uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return uint4{a, b, c, d};
}
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
static thread_local int t_lane = 0;
static uint32_t g_vals[32];
static std::atomic<unsigned> g_arrived{0}, g_phase{0};
static void warp_bar() {
  const unsigned phase = g_phase.load(std::memory_order_acquire);
  if (g_arrived.fetch_add(1, std::memory_order_acq_rel) == 31) {
    g_arrived.store(0, std::memory_order_relaxed);
    g_phase.store(phase + 1, std::memory_order_release);
  } else {
    while (g_phase.load(std::memory_order_acquire) == phase) std::this_thread::yield();
  }
}
static void __syncwarp() { warp_bar(); }
static uint32_t __shfl_sync(unsigned, uint32_t v, int src) {
  g_vals[t_lane] = v;
  warp_bar();
  const uint32_t r = g_vals[src & 31];
  warp_bar();
  return r;
}
static uint32_t __reduce_max_sync(unsigned, uint32_t v) {
  g_vals[t_lane] = v;
  warp_bar();
  uint32_t r = g_vals[0];
  for (int i = 1; i < 32; i++) r = max(r, g_vals[i]);
  warp_bar();
  return r;
}
static uint32_t __ballot_sync(unsigned, bool p) {
  g_vals[t_lane] = p;
  warp_bar();
  uint32_t r = 0;
  for (int i = 0; i < 32; i++) r |= (g_vals[i] ? 1u : 0u) << i;
  warp_bar();
  return r;
}
static bool __any_sync(unsigned m, bool p) { return __ballot_sync(m, p) != 0; }
static int __popc(uint32_t v) { return __builtin_popcount(v); }
static uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, uint32_t shift) {
  return uint32_t(((uint64_t(hi) << 32) | lo) >> (shift & 31));
}
"""

Z_CODEC_MAIN = r"""
using Block = std::array<uint4, 32>;
static std::vector<Block> g_in, g_out[2], g_slots[2];
alignas(16) static uint32_t g_sm[kZ2Words];
static uint4 g_slot[32];

// One warp-collective round per block, as two_shot_z's sender and receiver
// (pass 0: encode2, at most 24 quads; pass 1: encode2<23>, a 23-quad limit).
static void lane_main(int lane) {
  t_lane = lane;
  for (int pass = 0; pass < 2; pass++)
    for (size_t b = 0; b < g_in.size(); b++) {
      if (lane == 0)
        for (auto& q : g_slot) q = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
      warp_bar();
      const Quad q = pass ? encode2<23>(g_in[b][lane], lane, g_sm)
                          : encode2(g_in[b][lane], lane, g_sm);
      if (q.store) g_slot[q.at] = q.v;
      warp_bar();
      const uint4 got = g_slot[lane];
      g_out[pass][b][lane] = decode2(got, g_slot[0].x, g_slot[0].y, lane, g_sm);
      if (lane == 0)
        for (int i = 0; i < 32; i++) g_slots[pass][b][i] = g_slot[i];
      warp_bar();
    }
}

static uint16_t half(const uint4& v, int i) {
  const uint32_t w = i < 2 ? v.x : i < 4 ? v.y : i < 6 ? v.z : v.w;
  return (i & 1) ? (w >> 16) : (w & 0xFFFFu);
}
static uint32_t word(const uint4& v, int k) {
  return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}

// Block from 248 BF16 bit patterns (lanes 0..30; lane 31 and lanes past
// `valid` zero), sanitized as the kernel sanitizes its input packs.
static Block make_block(const std::vector<uint16_t>& h, int valid) {
  Block blk{};
  for (int l = 0; l < 31 && l < valid; l++) {
    uint32_t w[4];
    for (int k = 0; k < 4; k++) {
      w[k] = uint32_t(h[8 * l + 2 * k]) | (uint32_t(h[8 * l + 2 * k + 1]) << 16);
      if (w[k] == kSentinel) w[k] = 0x7FFF7FFFu;
    }
    blk[l] = make_uint4(w[0], w[1], w[2], w[3]);
  }
  return blk;
}

int main() {
  std::mt19937 rng(1031);
  auto r = [&](uint32_t n) { return uint32_t(rng() % n); };
  auto val = [&](uint32_t e) { return uint16_t((r(2) << 15) | (e << 7) | r(128)); };
  // Every BF16 bit pattern once, in order: 248 consecutive patterns share a
  // sign and span at most three exponents.
  for (uint32_t s = 0; s < 65536; s += 248) {
    std::vector<uint16_t> h(248);
    for (int i = 0; i < 248; i++) h[i] = uint16_t((s + i) & 0xFFFF);
    g_in.push_back(make_block(h, 31));
  }
  const uint16_t specials[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080,
                               0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF};
  for (int b = 0; b < 4000; b++) {
    std::vector<uint16_t> h(248);
    const int kind = b % 12;
    const uint32_t T = 16 + r(239);  // the block's largest exponent, 16..254
    for (int i = 0; i < 248; i++) {
      switch (kind) {
        case 0: h[i] = r(8) ? val(T - r(6)) : 0; break;  // served-like
        case 1: h[i] = uint16_t(r(65536)); break;  // any bits
        case 2:  // +-0, subnormals: wide lanes
          h[i] = r(3) ? val(T - r(14)) : uint16_t((r(2) << 15) | r(128));
          break;
        case 3: h[i] = val(i == 0 ? T : T - r(8)); break;  // within 7 octaves: narrow lanes
        case 4: h[i] = val(i == 0 ? T : i == 9 ? T - 8 : T - r(8)); break;  // one wide lane
        case 5: h[i] = val(i == 0 ? T : i == 100 ? T - 14 : T - r(15)); break;  // code 15
        case 6: h[i] = val(i == 0 ? T : i == 100 ? T - 15 : T - r(15)); break;  // code 16: raw
        case 7: h[i] = r(64) ? val(T - r(4)) : specials[r(15)]; break;  // Inf, NaN, extremes
        case 8: h[i] = r(2) ? 0x8000 : 0; break;  // only +-0
        case 9: h[i] = val(i < 8 * int(1 + r(31)) ? T - 8 - r(6) : T - r(8)); break;  // wide lanes
        case 10:  // sign / mantissa bytes 0xFF: a compressed word would be the sentinel
          h[i] = (i % 8) < 4 && i < 16 ? uint16_t(0x8000 | ((T - r(3)) << 7) | 0x7F)
                                       : val(T - r(3));
          break;
        default: h[i] = val(T - r(6)); break;  // partial block below
      }
    }
    if (kind == 7 && b % 24 == 7) h[0] = h[1] = 0xFFFF;  // a sentinel word
    g_in.push_back(make_block(h, kind == 11 ? 1 + int(r(30)) : 31));
  }
  const int nb = int(g_in.size());
  for (int p = 0; p < 2; p++) g_out[p].resize(nb), g_slots[p].resize(nb);
  std::vector<std::thread> warp;
  for (int l = 0; l < 32; l++) warp.emplace_back(lane_main, l);
  for (auto& t : warp) t.join();

  long fails = 0, compressed = 0, raw = 0, wide_lanes = 0, narrow_blocks = 0, q24 = 0;
  long sentinel_raw = 0, raw23 = 0, sanitized = 0;
  for (int b = 0; b < nb; b++) {
    // Expected format, from the definition: T the largest exponent field; a
    // lane is wide when it holds exponent field 0 or a value more than 7
    // octaves below T; nq whole quads.
    uint32_t T = 0, maxc = 0, mask = 0;
    for (int l = 0; l < 31; l++)
      for (int i = 0; i < 8; i++) T = max(T, uint32_t(half(g_in[b][l], i) >> 7) & 0xFF);
    for (int l = 0; l < 32; l++) {
      bool zero = false;
      uint32_t lc = 0;
      for (int i = 0; i < 8; i++) {
        const uint32_t e = (half(g_in[b][l], i) >> 7) & 0xFF;
        zero |= e == 0;
        lc = max(lc, e ? T + 1 - e : 0u);
      }
      maxc = max(maxc, lc);
      if (l < 31 && (zero || lc > 8)) mask |= 1u << l;
    }
    const int nq = (kZ2FixedBytes + __builtin_popcount(mask) + 15) >> 4;
    bool raw_at[2];
    for (int p = 0; p < 2; p++) {
      const bool must_raw = T == 255 || maxc > 15 || nq > (p ? 23 : 24);
      const Block& s = g_slots[p][b];
      const bool is_raw = s[0].x == 0 && s[0].y == kZ2RawW1 && s[0].z == 0 && s[0].w == 0;
      raw_at[p] = is_raw;
      if (must_raw && !is_raw) fails++;
      if (is_raw) {
        // quad l + 1 = lane l's sanitized pack, all 32 quads sent
        for (int l = 0; l < 31; l++)
          for (int k = 0; k < 4; k++)
            if (word(s[l + 1], k) != word(g_in[b][l], k)) fails++;
      } else {
        if (s[0].x != (T | (mask << 8)) || (s[0].y & 0xFFu) != (mask >> 24)) fails++;
        // nq whole quads sent, no word of them the sentinel; the rest still is
        for (int qd = 0; qd < 32; qd++)
          for (int k = 0; k < 4; k++)
            if ((word(s[qd], k) == kSentinel) == (qd < nq)) fails++;
      }
      if (p == 0) {
        // raw with every code in range: a compressed word was the sentinel
        sentinel_raw += is_raw && !must_raw;
        (is_raw ? raw : compressed)++;
        if (!is_raw) {
          wide_lanes += __builtin_popcount(mask);
          narrow_blocks += mask == 0;
          q24 += nq == 24;
        }
      }
      // Lossless: every lane gets the pack that was sent, bit for bit.
      for (int l = 0; l < 31; l++)
        for (int k = 0; k < 4; k++)
          if (word(g_out[p][b][l], k) != word(g_in[b][l], k)) fails++;
    }
    raw23 += raw_at[1] && !raw_at[0];  // 24 quads: raw under the 23-quad limit only
    for (int l = 0; l < 31; l++)
      for (int k = 0; k < 4; k++) sanitized += word(g_in[b][l], k) == 0x7FFF7FFFu;
  }
  printf("Z_ROUNDTRIP %s blocks=%d compressed=%ld raw=%ld sentinel_raw=%ld narrow_blocks=%ld "
         "q24=%ld wide_lanes=%ld raw23=%ld sanitized=%ld\n",
         fails ? "FAIL" : "PASS", nb, compressed, raw, sentinel_raw, narrow_blocks, q24,
         wide_lanes, raw23, sanitized);
  return fails || !compressed || !raw || !sentinel_raw || !narrow_blocks || !q24 ||
         !raw23 || !sanitized;
}
"""


def _z_codec_source() -> str:
    """The format's section of KERNEL_SRC, less add_pack (BF16 arithmetic)
    and the device-only stores, polling and kernels."""
    src = flags.KERNEL_SRC
    a = src.index("constexpr int kLanes = 32;")
    b = src.index("AFD void add_pack(")
    c = src.index("// Format v2: one block of 31 packs per warp")
    d = src.index("AFD void store_block(")
    assert a < b < c < d
    return src[a:b] + src[c:d]


@pytest.mark.skipif(CXX is None, reason="no host C++ compiler")
def test_z_codec_round_trip_is_lossless(tmp_path):
    """encode2 -> slot -> decode2 gives every lane its pack back, bit for bit,
    in two_shot_z's slots (up to 24 quads) and under a 23-quad limit: every
    BF16 pattern (both signs, +-0, every subnormal, every exponent, Inf, every
    NaN payload), random and served-like blocks, all-narrow blocks (22 quads),
    wide lanes up to 24 quads, codes 15 (compressed) and 16 (raw), sign /
    mantissa bytes that would make a word the sentinel (raw), partial blocks,
    the sanitized sentinel word. The header, the quads sent (no word the
    sentinel) and the quads not sent (still the sentinel) are checked against
    the format's definition."""
    (tmp_path / "z.cc").write_text(Z_CODEC_PRELUDE + _z_codec_source() + Z_CODEC_MAIN)
    exe = tmp_path / "z"
    r = subprocess.run(
        [CXX, "-std=c++17", "-O2", "-pthread", str(tmp_path / "z.cc"), "-o", str(exe)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "Z_ROUNDTRIP PASS" in r.stdout, r.stdout


def test_z_sizes():
    # 8 MiB - 16 (the largest message the custom all-reduce takes): the last
    # rank's partition is 131074 packs = 4229 blocks of 31.
    assert flags.z_block_slots(8 * 1024 * 1024 - 16) == 4229
    assert flags.z_block_slots(8 * 1024 * 1024) == 4229
    assert flags.z_stage_words(8 * 1024 * 1024 - 16) == 8 * 4229 * 128
    assert 3 * flags.z_stage_words(8 * 1024 * 1024 - 16) * 4 == 51_965_952  # ~49.6 MiB
    assert flags.z_block_slots(524288) == 265  # 8192 packs in the last partition
    assert flags.z_block_slots(16) == 1 and flags.z_block_slots(31 * 16 * 4) == 1
    assert flags.z_block_slots(32 * 16 * 4) == 2
    # the binding's bound, mirrored: nb <= lb_max for every message up to the max
    for nbytes in range(16, 200_000, 16):
        L = flags.largest_part_packs(nbytes, 4)
        assert -(-L // flags.Z_BLOCK_PACKS) <= flags.z_block_slots(200_000)


def test_z_kernel_source_is_the_search_candidate():
    """The candidate's kernels and launch rules are present; the uncompressed
    kernel and its binding are kept for two and four ranks."""
    src = flags.KERNEL_SRC
    assert "two_shot_z(Ptrs bufs" in src and "template __global__ void two_shot<2>" in src
    assert "one_shot(Ptrs bufs" in src and "template __global__ void one_shot<4>" in src
    assert "recv_blocks2<kRanks - 1>(src, active, peer, lane, 1, v, sm, wc)" in src
    assert "encode_block" not in src and "recv_blocks(" not in src  # one format
    for name in ("all_reduce", "all_reduce_z", "fill_sentinel", "alloc_error_record"):
        assert flags.BIND_SRC.count(f'm.def("{name}",') == 1
    assert "(3 * nb + warps_per_block - 1) / warps_per_block" in flags.BIND_SRC
    assert "const int work = one_shot ? size : L;" in flags.BIND_SRC
    assert flags.CUDA_SRC == flags.KERNEL_SRC + flags.BIND_SRC
    assert flags.Z_BLOCK_PACKS == 31 and "kBlockPacks = 31" in src
    assert flags.Z_SLOT_WORDS == 128 and "kSlotWords = 4 * kLanes" in src


@pytest.mark.skipif(
    not HAVE_NVCC or os.environ.get("VLLM_TEST_FLAGS_JIT", "0") != "1",
    reason="JIT build test: set VLLM_TEST_FLAGS_JIT=1 (needs nvcc and ninja)",
)
def test_jit_build(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(flags, "_MODULE", None)
    m = flags.load_module()
    assert hasattr(m, "all_reduce") and hasattr(m, "fill_sentinel")
    assert hasattr(m, "all_reduce_z")



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
    fa.compressed = False
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
        assert m.calls[0][-3:] == (1234, fa.errors.dev_ptr, False)
        assert m.calls[0][-4] == flags.launch_blocks(8192, 4)
        fa.all_reduce(x, one_shot=True)  # the one-shot: its launch rule and flag
        assert m.calls[1][-4:] == (flags.launch_blocks(8192, 4, True), 1234,
                                   fa.errors.dev_ptr, True)
        m.buf[0], m.buf[1] = 1, 3  # a record appears: the periodic check reports it
        for _ in range(flags.CHECK_EVERY - 2):
            fa.all_reduce(x)
        assert len(seen) == 1 and "rank 3" in seen[0]
    finally:
        fa.errors._stop.set()


def test_all_reduce_z_passes_its_buffer_and_launch_rule(monkeypatch):
    m = _FakeModule()
    m.all_reduce_z = lambda *args: m.calls.append(("z",) + args)
    fa = object.__new__(flags.FlagsAllreduce)
    fa.module, fa.ptrs, fa.rank, fa.world_size = m, [1, 2, 3, 4], 2, 4
    fa.compressed = True
    fa.lb_max = flags.z_block_slots(8 << 20)
    fa.stage_words = flags.z_stage_words(8 << 20)
    fa.epoch = torch.zeros(16, dtype=torch.int32)
    fa.limit_ns = 1234
    fa.errors = SimpleNamespace(dev_ptr=99, check=lambda: None)
    fa._calls = 0
    monkeypatch.setattr(flags.torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.zeros(65536, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    assert fa.all_reduce(x, out) is out
    (call,) = m.calls
    assert call[0] == "z" and call[1] is x and call[2] is out
    assert call[3:] == ([1, 2, 3, 4], fa.epoch.data_ptr(), 2, 8 * 4229 * 128, 4229,
                        flags.THREADS, flags.MAX_BLOCKS, 1234, 99)


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
