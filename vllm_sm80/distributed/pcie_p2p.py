# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The opt-in PCIe peer-to-peer leg of the custom all-reduce topology check.

Upstream's ``is_fully_connected`` answers yes only for NVLink, and
``CustomAllreduce`` refuses more than two GPUs that are not fully connected.
With ``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1`` the fork's CUDA platform
(``vllm_sm80.platform``) also counts a group whose every pair reports working
PCIe peer-to-peer. The bytes-moving ``gpu_p2p_access_check`` probe stays
mandatory behind it: ``VLLM_SKIP_P2P_CHECK`` is not honoured under the flag.

NVML handles, the device count and the visible-to-physical map are read from
``vllm.platforms.cuda`` at call time, so this module imports no CUDA code.
"""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def pcie_p2p_custom_allreduce_allowed() -> bool:
    """Whether the opt-in PCIe peer-to-peer custom all-reduce may be used.

    VLLM_CUSTOM_ALLREDUCE_PCIE_P2P asks for it. This also refuses when
    PyTorch's expandable_segments allocator is on, because the two cannot work
    together: CustomAllreduce takes legacy CUDA IPC handles on its captured
    CUDA-graph buffers (``cudaIpcGetMemHandle`` in ``get_graph_buffer_ipc_meta``),
    and that call returns ``cudaErrorInvalidValue`` for VMM-backed expandable
    allocations. Left unguarded the combination kills every worker part-way
    through startup, reporting only ``Cuda error ... 'invalid argument'`` from
    custom_all_reduce.cuh with nothing to connect it to the allocator.

    Refusing here instead leaves the host-staged path serving (with
    VLLM_SM80_ALLREDUCE=1; NCCL otherwise), so the flag is safe to leave
    set: it simply does nothing until the allocator allows it.
    """
    if not envs.VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:
        return False
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        logger.warning_once(
            "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 is ignored because "
            "PYTORCH_CUDA_ALLOC_CONF contains expandable_segments:True. The "
            "custom all-reduce needs legacy CUDA IPC handles for its captured "
            "graph buffers and the VMM allocator cannot provide them. Unset "
            "expandable_segments to use the device-memory path; until then the "
            "host-staged all-reduce keeps serving."
        )
        return False
    return True


def nvml_caps_index(name: str) -> int | None:
    """The integer value of a pynvml P2P caps-index constant, or None.

    ``nvmlDeviceGetP2PStatus`` passes this straight to ctypes, so it has to be
    a real int. vLLM's vendored pynvml declares
    ``NVML_P2P_CAPS_INDEX_READ = (0,)`` -- a 1-tuple, from a stray trailing
    comma -- which ctypes refuses to convert. Unwrap that spelling rather than
    depend on the typo being fixed, and treat anything else as absent so the
    caller falls back to torch.
    """
    from vllm.platforms import cuda as cuda_platform

    value = getattr(cuda_platform.pynvml, name, None)
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _nvml_pcie_p2p_fully_connected(physical_device_ids: list[int]) -> bool:
    """True when NVML reports READ *and* WRITE P2P OK for every ordered pair.

    Returns False -- not an error -- when this pynvml does not expose the READ
    and WRITE caps indices, so the caller can fall back to torch's peer query.
    """
    from vllm.platforms import cuda as cuda_platform

    pynvml = cuda_platform.pynvml
    read_index = nvml_caps_index("NVML_P2P_CAPS_INDEX_READ")
    write_index = nvml_caps_index("NVML_P2P_CAPS_INDEX_WRITE")
    if read_index is None or write_index is None:
        logger.debug(
            "pynvml exposes no usable NVML_P2P_CAPS_INDEX_READ/_WRITE; using"
            " torch.cuda.can_device_access_peer for the PCIe P2P check."
        )
        return False
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i == j:
                    continue
                for caps_index in (read_index, write_index):
                    status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, caps_index
                    )
                    if status != pynvml.NVML_P2P_STATUS_OK:
                        return False
    except Exception as exc:
        # Deliberately broad. Besides NVMLError this has to survive a pynvml
        # whose constants or signatures do not match the loaded libnvidia-ml,
        # which surfaces as ctypes.ArgumentError rather than NVMLError. The
        # torch fallback below answers the same question; a driver quirk here
        # must never be the reason a server fails to start.
        logger.debug("NVML PCIe P2P query failed (%s).", exc)
        return False
    return True


def _torch_p2p_fully_connected(physical_device_ids: list[int]) -> bool:
    """True when torch reports peer access both ways for every pair.

    ``can_device_access_peer`` takes visible ordinals, so the physical IDs are
    translated back through the device-control env var first.
    """
    from vllm.platforms import cuda as cuda_platform

    try:
        physical_to_visible = {
            cuda_platform.CudaPlatformBase.visible_device_id_to_physical_device_id(
                visible
            ): visible
            for visible in range(cuda_platform._cuda_device_count_stateless())
        }
        ordinals = [physical_to_visible[pid] for pid in physical_device_ids]
    except Exception as exc:
        logger.debug("PCIe P2P check: cannot map physical device ids (%s).", exc)
        return False
    for i, a in enumerate(ordinals):
        for j, b in enumerate(ordinals):
            if i == j:
                continue
            try:
                if not torch.cuda.can_device_access_peer(a, b):
                    return False
            except Exception as exc:
                logger.debug("PCIe P2P check failed for %d->%d (%s).", a, b, exc)
                return False
    return True


def pcie_p2p_fully_connected(
    physical_device_ids: list[int], *, use_nvml: bool
) -> bool:
    """The opt-in PCIe leg of ``is_fully_connected``.

    Gated on VLLM_CUSTOM_ALLREDUCE_PCIE_P2P. A True here only says the
    driver *advertises* peer access; it never stands alone. CustomAllreduce
    still has to pass ``gpu_p2p_access_check``, the probe that actually moves
    bytes between the cards, and the same env makes that probe mandatory.
    """
    if not pcie_p2p_custom_allreduce_allowed():
        return False
    if len(physical_device_ids) < 2:
        return False
    connected = use_nvml and _nvml_pcie_p2p_fully_connected(physical_device_ids)
    if not connected:
        connected = _torch_p2p_fully_connected(physical_device_ids)
    # `*_once` de-duplicates on the arguments, so they have to be hashable.
    device_list = ", ".join(str(i) for i in physical_device_ids)
    if connected:
        logger.info_once(
            "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 and every pair of devices"
            " [%s] reports working PCIe peer-to-peer, so this group counts as"
            " fully connected for the custom all-reduce. The"
            " gpu_p2p_access_check probe still has to pass before it is used.",
            device_list,
        )
    else:
        logger.warning_once(
            "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1, but PCIe peer-to-peer is not"
            " available for every pair of devices [%s]. The custom all-reduce"
            " stays disabled for more than two GPUs.",
            device_list,
        )
    return connected
