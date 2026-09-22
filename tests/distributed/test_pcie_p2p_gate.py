# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the opt-in PCIe-P2P custom all-reduce gate.

    CUDA_VISIBLE_DEVICES="" pytest tests/distributed/test_pcie_p2p_gate.py

Nothing here touches a GPU: NVML is a stub object and every torch peer query is
patched out.

What is worth testing without hardware:

  * `is_fully_connected` answers the four cases -- NVLink connected, PCIe P2P
    connected with the env on, PCIe P2P connected with the env off, neither --
    and the env only ever *adds* a way to say yes;
  * the NVML READ/WRITE pair is required in both directions, and a pynvml
    without those caps indices falls back to `can_device_access_peer`;
  * `_can_p2p` keeps `gpu_p2p_access_check` mandatory under the new env, since
    a driver that advertises peer access it cannot route is exactly the failure
    this opt-in invites;
  * the host-staged all-reduce stands aside on the operator's say-so and not
    because a driver started reporting peer access.
"""

from __future__ import annotations

import pytest
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators import host_shm_all_reduce as hsm
from vllm.platforms import cuda as cuda_platform

ENV = "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE"
IDS = [0, 1, 2, 3]


class _FakeNVMLError(Exception):
    pass


class _FakeNvml:
    """Just enough pynvml for `is_fully_connected`."""

    def __init__(self, *, nvlink_ok: bool, pcie_ok: bool, pcie_indices: bool = True):
        self.NVMLError = _FakeNVMLError
        self.NVML_P2P_STATUS_OK = 0
        self.NVML_P2P_STATUS_NOT_SUPPORTED = 4
        self.NVML_P2P_CAPS_INDEX_NVLINK = 2
        if pcie_indices:
            self.NVML_P2P_CAPS_INDEX_READ = 0
            self.NVML_P2P_CAPS_INDEX_WRITE = 1
        self._nvlink_ok = nvlink_ok
        self._pcie_ok = pcie_ok
        self.queries: list[tuple[str, str, int]] = []
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, i):
        return f"handle{i}"

    def nvmlDeviceGetP2PStatus(self, a, b, index):
        self.queries.append((a, b, index))
        if index == self.NVML_P2P_CAPS_INDEX_NVLINK:
            ok = self._nvlink_ok
        else:
            ok = self._pcie_ok
        return self.NVML_P2P_STATUS_OK if ok else self.NVML_P2P_STATUS_NOT_SUPPORTED


def _install(monkeypatch, fake):
    monkeypatch.setattr(cuda_platform, "pynvml", fake)
    return fake


def _no_torch_peer(monkeypatch):
    """Make the torch fallback leg answer False, so NVML alone decides."""
    monkeypatch.setattr(cuda_platform, "_cuda_device_count_stateless", lambda *a: 4)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: False)


def _torch_peer(monkeypatch, value=True):
    monkeypatch.setattr(cuda_platform, "_cuda_device_count_stateless", lambda *a: 4)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: value)


# --------------------------------------------------------------------------
# envs
# --------------------------------------------------------------------------
def test_env_is_declared_and_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert ENV in envs.environment_variables
    assert getattr(envs, ENV) is False
    assert ENV in envs.__annotations__ or hasattr(envs, ENV)


def test_env_parses_one_as_true(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert getattr(envs, ENV) is True
    monkeypatch.setenv(ENV, "0")
    assert getattr(envs, ENV) is False


# --------------------------------------------------------------------------
# is_fully_connected: the four combinations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("env_on", [False, True])
def test_nvlink_connected_is_fully_connected(monkeypatch, env_on):
    """NVLink keeps answering yes whatever the new env says."""
    monkeypatch.setenv(ENV, "1" if env_on else "0")
    fake = _install(monkeypatch, _FakeNvml(nvlink_ok=True, pcie_ok=False))
    _no_torch_peer(monkeypatch)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True
    # and the PCIe leg was never consulted
    assert all(q[2] == fake.NVML_P2P_CAPS_INDEX_NVLINK for q in fake.queries)


def test_pcie_connected_with_env_on_is_fully_connected(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    fake = _install(monkeypatch, _FakeNvml(nvlink_ok=False, pcie_ok=True))
    _no_torch_peer(monkeypatch)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True
    indices = {q[2] for q in fake.queries}
    assert fake.NVML_P2P_CAPS_INDEX_READ in indices
    assert fake.NVML_P2P_CAPS_INDEX_WRITE in indices


def test_pcie_connected_with_env_off_is_not_fully_connected(monkeypatch):
    """The default is unchanged from upstream: PCIe P2P is not enough."""
    monkeypatch.setenv(ENV, "0")
    fake = _install(monkeypatch, _FakeNvml(nvlink_ok=False, pcie_ok=True))
    _torch_peer(monkeypatch, True)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False
    assert all(q[2] == fake.NVML_P2P_CAPS_INDEX_NVLINK for q in fake.queries)


def test_neither_link_with_env_on_is_not_fully_connected(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    _install(monkeypatch, _FakeNvml(nvlink_ok=False, pcie_ok=False))
    _no_torch_peer(monkeypatch)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False


# --------------------------------------------------------------------------
# the PCIe leg in detail
# --------------------------------------------------------------------------
def test_read_alone_is_not_enough(monkeypatch):
    """READ OK but WRITE refused must not count as connected."""
    monkeypatch.setenv(ENV, "1")
    fake = _FakeNvml(nvlink_ok=False, pcie_ok=True)

    def read_only(a, b, index):
        if index == fake.NVML_P2P_CAPS_INDEX_WRITE:
            return fake.NVML_P2P_STATUS_NOT_SUPPORTED
        if index == fake.NVML_P2P_CAPS_INDEX_NVLINK:
            return fake.NVML_P2P_STATUS_NOT_SUPPORTED
        return fake.NVML_P2P_STATUS_OK

    fake.nvmlDeviceGetP2PStatus = read_only
    _install(monkeypatch, fake)
    _no_torch_peer(monkeypatch)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False


def test_one_bad_pair_is_not_enough(monkeypatch):
    """Fully connected means every pair; 0<->3 refusing is a veto."""
    monkeypatch.setenv(ENV, "1")
    fake = _FakeNvml(nvlink_ok=False, pcie_ok=True)

    def one_bad_pair(a, b, index):
        if index == fake.NVML_P2P_CAPS_INDEX_NVLINK:
            return fake.NVML_P2P_STATUS_NOT_SUPPORTED
        if {a, b} == {"handle0", "handle3"}:
            return fake.NVML_P2P_STATUS_NOT_SUPPORTED
        return fake.NVML_P2P_STATUS_OK

    fake.nvmlDeviceGetP2PStatus = one_bad_pair
    _install(monkeypatch, fake)
    _no_torch_peer(monkeypatch)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False


def test_pcie_leg_falls_back_to_torch_without_caps_indices(monkeypatch):
    """An older pynvml has no READ/WRITE indices; torch answers instead."""
    monkeypatch.setenv(ENV, "1")
    _install(
        monkeypatch, _FakeNvml(nvlink_ok=False, pcie_ok=False, pcie_indices=False)
    )
    _torch_peer(monkeypatch, True)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True

    _torch_peer(monkeypatch, False)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False


def test_nvml_error_on_the_pcie_leg_falls_back_to_torch(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    fake = _FakeNvml(nvlink_ok=False, pcie_ok=True)

    def boom(a, b, index):
        if index == fake.NVML_P2P_CAPS_INDEX_NVLINK:
            return fake.NVML_P2P_STATUS_NOT_SUPPORTED
        raise _FakeNVMLError("nvml is unhappy")

    fake.nvmlDeviceGetP2PStatus = boom
    _install(monkeypatch, fake)
    _torch_peer(monkeypatch, True)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True


def test_tuple_valued_caps_index_is_unwrapped(monkeypatch):
    """A caps index declared as the 1-tuple `(0,)` still queries index 0.

    Regression: vllm/third_party/pynvml.py declares
    `NVML_P2P_CAPS_INDEX_READ = (0,)` -- a stray trailing comma. ctypes cannot
    convert a tuple, so nvmlDeviceGetP2PStatus raised ctypes.ArgumentError and
    took a TP4 boot down with it.
    """
    monkeypatch.setenv(ENV, "1")
    fake = _FakeNvml(nvlink_ok=False, pcie_ok=True)
    fake.NVML_P2P_CAPS_INDEX_READ = (0,)
    _install(monkeypatch, fake)
    _no_torch_peer(monkeypatch)

    assert cuda_platform._nvml_caps_index("NVML_P2P_CAPS_INDEX_READ") == 0
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True
    pcie_indices = {
        index
        for _, _, index in fake.queries
        if index != fake.NVML_P2P_CAPS_INDEX_NVLINK
    }
    assert pcie_indices == {0, 1}
    assert all(isinstance(index, int) for index in pcie_indices)


@pytest.mark.parametrize("value", [None, "0", 1.0, True, (0, 1), ()])
def test_caps_index_rejects_anything_that_is_not_one_int(monkeypatch, value):
    fake = _FakeNvml(nvlink_ok=False, pcie_ok=True)
    fake.NVML_P2P_CAPS_INDEX_READ = value
    _install(monkeypatch, fake)
    assert cuda_platform._nvml_caps_index("NVML_P2P_CAPS_INDEX_READ") is None


def test_vendored_pynvml_caps_indices_resolve_to_ints():
    """Against the real pynvml this tree ships, not a fake.

    This is the check that would have caught the `(0,)` typo before a boot did.
    """
    for name in ("NVML_P2P_CAPS_INDEX_READ", "NVML_P2P_CAPS_INDEX_WRITE"):
        index = cuda_platform._nvml_caps_index(name)
        assert isinstance(index, int), f"{name} is not usable as a ctypes index"
    assert cuda_platform._nvml_caps_index("NVML_P2P_CAPS_INDEX_READ") == 0
    assert cuda_platform._nvml_caps_index("NVML_P2P_CAPS_INDEX_WRITE") == 1


def test_non_nvml_exception_on_the_pcie_leg_falls_back_to_torch(monkeypatch):
    """ctypes raises ArgumentError, not NVMLError; it must not reach the boot."""
    import ctypes

    monkeypatch.setenv(ENV, "1")
    fake = _FakeNvml(nvlink_ok=False, pcie_ok=True)

    def boom(a, b, index):
        if index == fake.NVML_P2P_CAPS_INDEX_NVLINK:
            return fake.NVML_P2P_STATUS_NOT_SUPPORTED
        raise ctypes.ArgumentError("argument 3: TypeError")

    fake.nvmlDeviceGetP2PStatus = boom
    _install(monkeypatch, fake)

    _torch_peer(monkeypatch, True)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True

    _torch_peer(monkeypatch, False)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False


ALLOC = "PYTORCH_CUDA_ALLOC_CONF"


@pytest.mark.parametrize(
    "conf,allowed",
    [
        ("expandable_segments:True", False),
        ("expandable_segments:False", True),
        ("expandable_segments:True,max_split_size_mb:128", False),
        ("max_split_size_mb:128", True),
        ("", True),
    ],
)
def test_expandable_segments_blocks_the_pcie_custom_allreduce(
    monkeypatch, conf, allowed
):
    """CustomAllreduce needs legacy CUDA IPC handles; VMM allocations have none.

    Unguarded, the pair crashes every worker at CUDA-graph capture with only
    "Cuda error custom_all_reduce.cuh:164 'invalid argument'". The flag has to
    stand down instead, so leaving it set is safe.
    """
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(ALLOC, conf)
    assert cuda_platform.pcie_p2p_custom_allreduce_allowed() is allowed


def test_allocator_conf_is_irrelevant_when_the_env_is_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(ALLOC, "expandable_segments:False")
    assert cuda_platform.pcie_p2p_custom_allreduce_allowed() is False


def test_expandable_segments_makes_the_group_not_fully_connected(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(ALLOC, "expandable_segments:True")
    _install(monkeypatch, _FakeNvml(nvlink_ok=False, pcie_ok=True))
    _torch_peer(monkeypatch, True)
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is False

    monkeypatch.setenv(ALLOC, "expandable_segments:False")
    assert cuda_platform.NvmlCudaPlatform.is_fully_connected(IDS) is True


def test_host_shm_keeps_serving_when_expandable_segments_blocks_the_handover(
    monkeypatch,
):
    """The guard must reach both gates, or the group silently falls to NCCL."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(ALLOC, "expandable_segments:True")
    monkeypatch.setattr(hsm, "_p2p_unavailable", lambda ws: False)
    assert hsm._stand_aside_for_custom_allreduce(4) is False

    monkeypatch.setenv(ALLOC, "expandable_segments:False")
    assert hsm._stand_aside_for_custom_allreduce(4) is True


def test_single_device_is_never_pcie_fully_connected(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    _torch_peer(monkeypatch, True)
    assert cuda_platform._pcie_p2p_fully_connected([0], use_nvml=False) is False


def test_non_nvml_platform_uses_torch_only(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    _torch_peer(monkeypatch, True)
    assert cuda_platform.NonNvmlCudaPlatform.is_fully_connected(IDS) is True

    monkeypatch.setenv(ENV, "0")
    assert cuda_platform.NonNvmlCudaPlatform.is_fully_connected(IDS) is False


# --------------------------------------------------------------------------
# the probe stays mandatory
# --------------------------------------------------------------------------
def _patch_can_p2p(monkeypatch, *, probe_result, peer_result=True):
    calls = {"probe": 0, "peer": 0}

    def probe(rank, i):
        calls["probe"] += 1
        return probe_result

    def peer(a, b):
        calls["peer"] += 1
        return peer_result

    monkeypatch.setattr(car, "gpu_p2p_access_check", probe)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", peer)
    monkeypatch.setattr(
        car.current_platform, "logical_device_id_to_visible_device_id", lambda i: i
    )
    return calls


def test_skip_p2p_check_is_ignored_when_pcie_p2p_is_allowed(monkeypatch):
    """The opt-in invites a driver that lies; the real probe must still run."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    calls = _patch_can_p2p(monkeypatch, probe_result=True)
    assert car._can_p2p(0, 4) is True
    assert calls["probe"] == 3  # every peer, not one
    assert calls["peer"] == 0


def test_failed_probe_vetoes_even_with_pcie_p2p_allowed(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    calls = _patch_can_p2p(monkeypatch, probe_result=False, peer_result=True)
    assert car._can_p2p(0, 4) is False
    assert calls["probe"] == 1


def test_skip_p2p_check_still_works_with_the_env_off(monkeypatch):
    """Upstream behaviour is untouched by default."""
    monkeypatch.setenv(ENV, "0")
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    calls = _patch_can_p2p(monkeypatch, probe_result=False, peer_result=True)
    assert car._can_p2p(0, 4) is True
    assert calls["probe"] == 0
    assert calls["peer"] == 1


# --------------------------------------------------------------------------
# host-shm stand-aside rule
# --------------------------------------------------------------------------
@pytest.mark.parametrize("peer_works", [False, True])
def test_host_shm_keeps_working_with_the_env_off(monkeypatch, peer_works):
    """Installing a P2P-capable driver must not silently disable the host path."""
    monkeypatch.setenv(ENV, "0")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: peer_works)
    assert hsm._stand_aside_for_custom_allreduce(4) is False


def test_host_shm_stands_aside_when_the_env_is_on_and_p2p_works(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    assert hsm._stand_aside_for_custom_allreduce(4) is True


def test_host_shm_serves_when_the_env_is_on_but_p2p_is_absent(monkeypatch):
    """CustomAllreduce would refuse too, so something has to serve."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: False)
    assert hsm._stand_aside_for_custom_allreduce(4) is False


def test_host_shm_constructor_honours_the_stand_aside_rule(monkeypatch):
    """The two fast paths are exclusive by choice, end to end."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(hsm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(hsm.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(hsm.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(hsm.dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(ps, "in_the_same_node_as", lambda group, source_rank=0: [True] * 4)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setenv(ENV, "1")

    loaded = {"n": 0}

    def _should_not_load():
        loaded["n"] += 1
        raise AssertionError("standing aside must not compile the extension")

    monkeypatch.setattr(hsm, "_load_module", _should_not_load)
    ar = hsm.HostShmAllreduce(group=object(), device="cuda:0")
    assert ar.disabled is True
    assert loaded["n"] == 0
