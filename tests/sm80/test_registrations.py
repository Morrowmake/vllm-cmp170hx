# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's registrations reach the upstream objects they replace.

Each override is looked up by an upstream name (a platform plugin, a backend
enum slot, a layer class name). An upstream rename would silently disable
it, so every test here asks upstream's own lookup and checks that it lands on
the fork's class.
"""

import pytest

import vllm.platforms as platforms
from vllm_sm80 import plugin


def test_platform_resolution_picks_the_fork_platform(monkeypatch):
    builtin = "vllm.platforms.cuda.CudaPlatform"
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    for name in platforms.builtin_platform_plugins:
        monkeypatch.setitem(platforms.builtin_platform_plugins, name, lambda: None)
    monkeypatch.setitem(platforms.builtin_platform_plugins, "cuda", lambda: builtin)
    monkeypatch.setattr(platforms, "cuda_platform_plugin", lambda: builtin)
    monkeypatch.setattr(
        platforms,
        "load_plugins_by_group",
        lambda group: {"sm80": plugin.platform_plugin}
        if group == platforms.PLATFORM_PLUGINS_GROUP
        else {},
    )
    assert platforms.resolve_current_platform_cls_qualname() == (
        plugin.PLATFORM_QUALNAME
    )


@pytest.mark.parametrize("nvml", [True, False])
def test_fork_platform_subclasses_the_upstream_variant(nvml):
    from vllm.platforms import cuda as cuda_platform
    from vllm_sm80 import platform as sm80_platform

    fork_cls = (
        sm80_platform.Sm80NvmlCudaPlatform
        if nvml
        else sm80_platform.Sm80NonNvmlCudaPlatform
    )
    upstream_cls = (
        cuda_platform.NvmlCudaPlatform if nvml else cuda_platform.NonNvmlCudaPlatform
    )
    assert issubclass(fork_cls, upstream_cls)
    assert fork_cls.__mro__[2] is upstream_cls
    assert fork_cls._enum == upstream_cls._enum
    assert fork_cls.device_name == upstream_cls.device_name


def test_custom_backend_slot_holds_triton_mla_sparse():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    plugin.register()
    plugin.register()
    assert AttentionBackendEnum.CUSTOM.get_path() == plugin.TRITON_MLA_SPARSE_BACKEND
    assert AttentionBackendEnum.CUSTOM.get_class().get_name() == "TRITON_MLA_SPARSE"


@pytest.mark.parametrize("major,expected", [(8, True), (9, True), (10, False)])
def test_fork_platform_offers_triton_mla_sparse_last(monkeypatch, major, expected):
    from unittest.mock import MagicMock

    import torch

    from vllm.platforms.interface import DeviceCapability
    from vllm_sm80 import platform as sm80_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

    ok = MagicMock()
    ok.validate_configuration.return_value = ["rejected"]
    monkeypatch.setattr("vllm.platforms.cuda._get_attn_backend_class", lambda b: ok)
    config = AttentionSelectorConfig(
        head_size=576, dtype=torch.bfloat16, kv_cache_dtype=None, block_size=64,
        use_mla=True, use_sparse=True,
    )
    cap = DeviceCapability(major=major, minor=0)
    valid, invalid = sm80_platform.Sm80CudaPlatform.get_valid_backends(
        device_capability=cap, attn_selector_config=config, num_heads=64
    )
    upstream_valid, upstream_invalid = sm80_platform.Sm80CudaPlatform.__mro__[
        2
    ].get_valid_backends(device_capability=cap, attn_selector_config=config, num_heads=64)
    assert (AttentionBackendEnum.CUSTOM in invalid) is expected
    if expected:
        assert invalid[AttentionBackendEnum.CUSTOM][0] == len(upstream_invalid)
    assert len(invalid) == len(upstream_invalid) + int(expected)


@pytest.mark.parametrize(
    "switch",
    [
        None,
        "VLLM_SM80_ALLREDUCE",
        "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P",
    ],
)
def test_communicator_class_follows_the_switches(monkeypatch, switch):
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm_sm80 import platform as sm80_platform
    from vllm_sm80.distributed import communicator as comm
    from vllm.utils.import_utils import resolve_obj_by_qualname

    for name in (
        "VLLM_SM80_ALLREDUCE",
        "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P",
    ):
        monkeypatch.delenv(name, raising=False)
    if switch is not None:
        monkeypatch.setenv(switch, "1")
    cls = resolve_obj_by_qualname(
        sm80_platform.Sm80CudaPlatform.get_device_communicator_cls()
    )
    if switch is None:
        # stateless_coordinator asserts exactly this for elastic EP groups.
        assert cls == CudaCommunicator
    else:
        assert cls is comm.Sm80CudaCommunicator
        assert issubclass(cls, CudaCommunicator)


def test_stateless_coordinator_still_asserts_upstreams_class():
    """The limitation the fork communicator documents: if this assertion goes
    away upstream, the switch-off restriction can go too."""
    import inspect

    from vllm.distributed import stateless_coordinator

    src = inspect.getsource(stateless_coordinator)
    assert "assert device_comm_cls == CudaCommunicator" in src
