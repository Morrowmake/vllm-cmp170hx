# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's CUDA platform, selected through the ``vllm.platform_plugins``
entry point ``sm80`` (``vllm_sm80.plugin.platform_plugin``).

The classes subclass upstream's two CUDA platforms and differ from them only
where the fork needs it:

- ``get_valid_backends``: for MLA on every compute capability other than
  10.x and 12.x, the Triton sparse-MLA backend
  (``vllm_sm80.attention.triton_mla_sparse``) is one more candidate after
  upstream's, through the ``AttentionBackendEnum.CUSTOM`` slot. The vendor
  sparse-MLA kernels reject sm_80, so it is the one that serves there.
- ``is_fully_connected``: with ``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1`` a group
  whose every pair has working PCIe peer-to-peer counts as fully connected
  (``vllm_sm80.distributed.pcie_p2p``). Without the flag the answer is
  upstream's.
- ``get_device_communicator_cls``: the fork communicator
  (``vllm_sm80.distributed.communicator``) while a fork all-reduce switch is
  set, else upstream's.
- ``check_and_update_config``: upstream's, then with
  ``VLLM_CUDAGRAPH_ESTIMATE_SPLIT=1``,
  ``VLLM_SM80_SHARD_ARGMAX=1`` or its hot pair the fork GPU worker
  (``vllm_sm80.worker.graph_mem_worker``) in place of upstream's.
- ``check_and_update_config``: upstream's, then, with ``VLLM_DEV_AB_SWITCH``
  or ``VLLM_HOT_SWAP`` set, the development A/B setter's worker extension
  (``vllm_sm80.dev_ab``).
- ``check_and_update_config``: last, the fork scheduler
  (``vllm_sm80.sched.scheduler``) when no scheduler class is configured.
- ``apply_config_platform_defaults``: upstream's, then the adaptive_k
  settings check (``vllm_sm80.spec_decode.adaptive_k.check_settings``).

``Sm80CudaPlatform`` is chosen the way ``vllm.platforms.cuda.CudaPlatform``
is: the NVML variant when NVML is available, else the non-NVML one.
"""

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import cuda as cuda_platform
from vllm.platforms.cuda import (
    NonNvmlCudaPlatform,
    NvmlCudaPlatform,
    nvml_available,
    with_nvml_context,
)
from vllm.platforms.interface import DeviceCapability
from vllm_sm80.distributed.pcie_p2p import pcie_p2p_fully_connected
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


class _Sm80CudaPlatformMixin:
    """What both fork CUDA platforms add; listed before the upstream base."""

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config,
        num_heads: int | None = None,
    ):
        """Upstream's candidates, then the Triton sparse-MLA backend.

        Upstream ranks every backend of ``_get_backend_priorities`` by its
        position and validates each one; the sparse-MLA candidate takes the
        next position and is validated the same way, so it is chosen only
        where no upstream candidate is valid.
        """
        valid, invalid = super().get_valid_backends(  # type: ignore[misc]
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        if not attn_selector_config.use_mla or device_capability.major in (10, 12):
            return valid, invalid

        from vllm_sm80.plugin import register_triton_mla_sparse_backend

        # The general plugin registers it too; a process can resolve
        # backends without having loaded general plugins.
        register_triton_mla_sparse_backend()
        backend = AttentionBackendEnum.CUSTOM
        priority = len(valid) + len(invalid)
        try:
            backend_class = cuda_platform._get_attn_backend_class(backend)
            reasons = backend_class.validate_configuration(
                device_capability=device_capability,
                **attn_selector_config._asdict(),
            )
        except (ImportError, OSError) as e:
            logger.debug(
                "Attention backend %s is unavailable", backend.name, exc_info=True
            )
            reasons = [f"{type(e).__name__}: {e}"]
        if reasons:
            invalid[backend] = (priority, reasons)
        else:
            candidate = cuda_platform._BackendCandidate(backend_class, backend, priority)
            valid.append(candidate)
        return valid, invalid

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """The fork communicator while a fork all-reduce switch is set.

        Only then: ``vllm.distributed.stateless_coordinator`` asserts that this
        is upstream's ``CudaCommunicator`` (see
        ``vllm_sm80.distributed.communicator``).
        """
        from vllm_sm80.distributed.communicator import (
            COMMUNICATOR_QUALNAME,
            needs_fork_communicator,
        )

        if needs_fork_communicator():
            return COMMUNICATOR_QUALNAME
        return super().get_device_communicator_cls()  # type: ignore[misc]

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config) -> None:
        """Upstream defaults, then the adaptive_k settings check."""
        super().apply_config_platform_defaults(vllm_config)  # type: ignore[misc]
        from vllm_sm80.spec_decode.adaptive_k import check_settings

        check_settings(vllm_config)

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        """Upstream config, optional graph-memory worker, A/B extension, then
        the fork scheduler."""
        super().check_and_update_config(vllm_config)  # type: ignore[misc]
        from vllm_sm80.worker.cudagraph_mem import select_worker
        from vllm_sm80.dev_ab import configure
        from vllm_sm80.sched import configure_scheduler

        select_worker(vllm_config.parallel_config)
        configure(vllm_config)
        configure_scheduler(vllm_config)


class Sm80NvmlCudaPlatform(_Sm80CudaPlatformMixin, NvmlCudaPlatform):
    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """Upstream's NVLink test, then the opt-in PCIe peer-to-peer leg.

        Every pair must report NVML ``P2P_CAPS_INDEX_READ`` and ``_WRITE``
        OK, or -- if this pynvml does not expose those indices -- peer access
        both ways under torch. That relaxation is what lets CustomAllreduce
        past its "more than two PCIe-only GPUs" gate; it is not a correctness
        claim, and the ``gpu_p2p_access_check`` probe stays mandatory behind
        it.
        """
        if super().is_fully_connected(physical_device_ids):
            return True
        return pcie_p2p_fully_connected(physical_device_ids, use_nvml=True)


class Sm80NonNvmlCudaPlatform(_Sm80CudaPlatformMixin, NonNvmlCudaPlatform):
    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        if envs.VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:
            # No NVML context here, so the PCIe leg has only torch's peer query.
            return pcie_p2p_fully_connected(physical_device_ids, use_nvml=False)
        return super().is_fully_connected(physical_device_ids)


Sm80CudaPlatform = Sm80NvmlCudaPlatform if nvml_available else Sm80NonNvmlCudaPlatform

PLATFORM_CLASSES = (Sm80NvmlCudaPlatform, Sm80NonNvmlCudaPlatform)