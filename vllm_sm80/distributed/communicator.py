# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's CUDA device communicator.

``vllm_sm80.platform`` returns :class:`Sm80CudaCommunicator` from
``get_device_communicator_cls`` only while ``VLLM_SM80_ALLREDUCE`` or
``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`` is set (:func:`needs_fork_communicator`);
otherwise every group gets upstream's ``CudaCommunicator``. On top of
upstream it adds:

- the host-staged all-reduce (``VLLM_SM80_ALLREDUCE``,
  ``host_shm_all_reduce.py``), TP groups only, tried before every upstream
  backend in ``all_reduce``;
- the flags-in-data two-shot path (``VLLM_SM80_ALLREDUCE``,
  ``custom_all_reduce_flags.py``) inside the custom all-reduce: ``ca_comm``
  becomes a :class:`Sm80CustomAllreduce`, whose ``custom_all_reduce`` tries
  the flags kernel first, at the same point upstream's own kernel is chosen;
  with ``VLLM_SM80_ALLREDUCE_COMPRESS`` at TP4 its compressed wire format
  takes the BF16 messages above allreduce/flags_raw_max_bytes that the flags
  kernel or upstream's two-stage kernel would take, and its one-shot kernel
  those up to allreduce/flags_one_shot_max_bytes (bitwise the same sums);
- with ``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`` the bytes-moving
  ``gpu_p2p_access_check`` probe stays mandatory: ``VLLM_SKIP_P2P_CHECK`` is
  read as 0 while upstream builds the custom all-reduce;
- the HOSTSHM entry and the PCIe and transport notes in the
  backend-selection log line.

With ``VLLM_SM80_ALLREDUCE`` each message of a TP all-reduce takes the first
of these transports that takes it (world size, bytes, dtype, CUDA graph
capture):

1. the host-staged all-reduce: BF16/FP32 up to its size limit, TP2/TP4 on one
   node; built only where it does not stand aside for working PCIe peer
   access under ``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P``;
2. the custom all-reduce (PCIe peer access): the flags-in-data kernel for
   BF16 up to its size limit at TP2/TP4 (at TP4 with
   ``VLLM_SM80_ALLREDUCE_COMPRESS``: the one-shot kernel up to
   allreduce/flags_one_shot_max_bytes, the compressed one above
   allreduce/flags_raw_max_bytes, also in place of upstream's two-stage
   kernel), else upstream's kernel up to its buffer size;
3. upstream's remaining backends (NCCL).

Layer-owned sums of eager prefill-sized messages (DeepSeek-V4.1's wo_b and
MoE outputs) go, ahead of 3., to the copy-engine ring
(``ce_ring_all_reduce.py``; round-once sums take its owner sum); the model's
own all-reduce helpers make that choice. The ring is built only under
``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`` and where every rank can map the others'
device IPC buffers; elsewhere those sums take 1.-3.

Limitation: ``vllm.distributed.stateless_coordinator`` asserts that the
platform's communicator class is upstream's ``CudaCommunicator``. Stateless
groups (elastic EP) therefore cannot be built while a fork all-reduce switch
is set; with the switches off they get upstream's class and work as before.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

import vllm.envs as envs
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.all_reduce_utils import (
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

COMMUNICATOR_QUALNAME = "vllm_sm80.distributed.communicator.Sm80CudaCommunicator"


def needs_fork_communicator() -> bool:
    """Whether a fork all-reduce switch is set that this communicator serves."""
    return bool(envs.VLLM_SM80_ALLREDUCE or envs.VLLM_CUSTOM_ALLREDUCE_PCIE_P2P)


@contextmanager
def p2p_probe_mandatory() -> Iterator[None]:
    """Read ``VLLM_SKIP_P2P_CHECK`` as 0 under ``VLLM_CUSTOM_ALLREDUCE_PCIE_P2P``.

    ``gpu_p2p_access_check`` is the only check that moves real bytes between
    the cards; ``can_device_access_peer`` is the driver's word for it, and
    VLLM_SKIP_P2P_CHECK trades one for the other (upstream ``_can_p2p``). That
    trade is not on offer when PCIe P2P is what put us on this path at all: a
    driver that advertises peer access the root complex will not route turns
    a wrong answer into a hang. Process environment of the worker during the
    communicator's construction only; the env cache, if already enabled, is
    rebuilt from the restored environment.
    """
    if not (envs.VLLM_CUSTOM_ALLREDUCE_PCIE_P2P and envs.VLLM_SKIP_P2P_CHECK):
        yield
        return
    cached = envs._is_envs_cache_enabled()
    if cached:
        envs.disable_envs_cache()
    old = os.environ.get("VLLM_SKIP_P2P_CHECK")
    os.environ["VLLM_SKIP_P2P_CHECK"] = "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ["VLLM_SKIP_P2P_CHECK"]
        else:
            os.environ["VLLM_SKIP_P2P_CHECK"] = old
        if cached:
            envs.enable_envs_cache()


def allreduce_max_bytes(param: str, world_size: int, default: int) -> int:
    """The allreduce/``param`` row of the sm80 tuning table for a group of
    ``world_size`` ranks, else ``default``. The key is the world size and,
    when the communicator is built inside an engine (under the current vLLM
    config), the model's hidden size: the width of its activation messages."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm_sm80 import tuning

    key: dict[str, int | str] = {"world_size": int(world_size)}
    config = get_current_vllm_config_or_none()
    model_config = None if config is None else config.model_config
    if model_config is not None:
        key["hidden"] = int(model_config.get_hidden_size())
    return tuning.lookup("allreduce", param, key, default)


# The hot-swap pair of VLLM_SM80_ALLREDUCE_COMPRESS (vllm_sm80/hot_swap.py).
COMPRESS_HOT_PAIR = "allreduce_compress"

# Upstream's kernel choice at four fully connected ranks with
# VLLM_CUSTOM_ALLREDUCE_ALGO unset (csrc/custom_all_reduce.cuh): one-stage
# below this many bytes, two-stage from here on.
UPSTREAM_TWO_STAGE_MIN_BYTES = 512 * 1024


class Sm80CustomAllreduce(car.CustomAllreduce):
    """Upstream's custom all-reduce plus the flags-in-data two-shot path.

    Not constructed directly: :class:`Sm80CudaCommunicator` swaps the class
    of the ``ca_comm`` upstream built and then calls :meth:`_init_flags`.
    """

    # Flags-in-data two-shot path (custom_all_reduce_flags.py); None = off.
    _flags = None
    # Its compressed wire format (VLLM_SM80_ALLREDUCE_COMPRESS, or carried by
    # the hot pair COMPRESS_HOT_PAIR); None = off. It takes bf16 messages over
    # _flags_z_min_bytes up to the flags path's limit, and from
    # _flags_z_upstream_from bytes up to _flags_z_max_bytes. The flags path's
    # one-shot kernel takes bf16 messages up to _flags_one_shot_max_bytes
    # (0 = never).
    _flags_z = None
    _flags_z_hot = False
    _flags_z_min_bytes = 0
    _flags_z_upstream_from = 0
    _flags_z_max_bytes = 0
    _flags_one_shot_max_bytes = 0

    def _init_flags(self, same_node: bool) -> None:
        """Set up the flags-in-data two-shot path, or log why it stays off.

        Every rank reaches the same decision: the gate inputs are the same on
        all ranks, and the extension load is agreed before any collective.
        """
        from vllm_sm80.distributed import custom_all_reduce_flags as flags
        from vllm_sm80.distributed.pcie_p2p import pcie_p2p_custom_allreduce_allowed

        capability = current_platform.get_device_capability()
        reason = flags.gate_reason(
            self.world_size,
            same_node,
            None if capability is None else tuple(capability),
            current_platform.is_cuda(),
            pcie_p2p_custom_allreduce_allowed(),
        )
        if reason is None:
            loaded = True
            try:
                flags.load_module()
            except Exception as error:  # compiler missing or failing
                logger.warning("Flags-in-data all-reduce extension: %s", error)
                loaded = False
            if not car._all_ranks_true(self.group, loaded):
                reason = "the extension did not build on every rank"
        if reason is not None:
            logger.info_once(
                "Custom all-reduce flags-in-data path requested "
                "(VLLM_SM80_ALLREDUCE=1) but off: %s%s",
                reason,
                "; so is its compressed wire format (VLLM_SM80_ALLREDUCE_COMPRESS)"
                if envs.VLLM_SM80_ALLREDUCE_COMPRESS else "",
                scope="global",
            )
            return
        self._flags = flags.FlagsAllreduce(
            self.group,
            self.device,
            self.rank,
            self.world_size,
            allreduce_max_bytes("flags_max_bytes", self.world_size, 262144),
            self.create_shared_buffer,
            self.free_shared_buffer,
        )
        logger.info_once(
            "Custom all-reduce flags-in-data two-shot kernel on for bf16 "
            "messages up to %d bytes (VLLM_SM80_ALLREDUCE=1); larger "
            "messages and other dtypes keep the existing kernel; a peer wait over "
            "%.0f s (custom_all_reduce_flags.WAIT_S) traps and ends the process",
            self._flags.max_bytes,
            self._flags.limit_ns / 1e9,
            scope="global",
        )
        hot = False
        if envs.VLLM_HOT_SWAP:
            from vllm_sm80 import hot_swap

            hot = hot_swap.listed(COMPRESS_HOT_PAIR)
        if envs.VLLM_SM80_ALLREDUCE_COMPRESS or hot:
            self._init_flags_z(flags, hot)

    def _init_flags_z(self, flags, hot: bool) -> None:
        """Set up the compressed flags kernel and the one-shot kernel (four
        ranks), or log why not.

        Both replace only kernels that sum in two-shot owner order as they
        do: the flags kernel, and upstream's cross_device_reduce_2stage where
        upstream would run it (not cross_device_reduce_1stage, rank order).
        """
        switch = (
            f"VLLM_HOT_SWAP={COMPRESS_HOT_PAIR}" if hot
            else "VLLM_SM80_ALLREDUCE_COMPRESS=1"
        )
        if self.world_size != flags.Z_WORLD_SIZE:
            logger.info_once(
                "Custom all-reduce compressed flags kernel requested (%s) but "
                "off: world size %d (it serves %d ranks)",
                switch, self.world_size, flags.Z_WORLD_SIZE, scope="global",
            )
            return
        flags_max = self._flags.max_bytes
        algo = envs.VLLM_CUSTOM_ALLREDUCE_ALGO
        if algo in ("1stage", "oneshot"):
            upstream_from = None
        elif algo in ("2stage", "twoshot"):
            upstream_from = flags_max + 16
        else:
            upstream_from = max(flags_max + 16, UPSTREAM_TWO_STAGE_MIN_BYTES)
        # should_custom_ar admits messages below max_size, in 16-byte packs.
        max_bytes = flags_max if upstream_from is None else self.max_size - 16
        self._flags_z = flags.FlagsAllreduce(
            self.group,
            self.device,
            self.rank,
            self.world_size,
            max_bytes,
            self.create_shared_buffer,
            self.free_shared_buffer,
            compressed=True,
        )
        self._flags_z_hot = hot
        self._flags_z_max_bytes = max_bytes
        self._flags_z_min_bytes = allreduce_max_bytes(
            "flags_raw_max_bytes", self.world_size, 32768
        )
        self._flags_z_upstream_from = (
            max_bytes + 16 if upstream_from is None else upstream_from
        )
        self._flags_one_shot_max_bytes = allreduce_max_bytes(
            "flags_one_shot_max_bytes", self.world_size, 8192
        )
        if upstream_from is None:
            upstream = "; cross_device_reduce_1stage (rank order) keeps the rest"
        else:
            upstream = (
                f", and from {upstream_from} to {max_bytes} bytes in place of "
                "cross_device_reduce_2stage"
            )
        logger.info_once(
            "Custom all-reduce compressed flags kernel on (%s) for bf16 "
            "messages over %d bytes up to %d bytes in place of the flags kernel%s "
            "(VLLM_CUSTOM_ALLREDUCE_ALGO=%r), and the one-shot kernel for those "
            "up to %d bytes; lossless, bitwise the kernels they replace; %d MiB "
            "of buffers per rank",
            switch,
            self._flags_z_min_bytes,
            flags_max,
            upstream,
            algo,
            self._flags_one_shot_max_bytes,
            self._flags_z.buffer_bytes >> 20,
            scope="global",
        )

    def _flags_z_takes(self, inp: torch.Tensor) -> bool:
        # The flags path's wire format: off while the flags path is (dev_ab.py's
        # custom_allreduce_flags toggle sets _flags to None).
        flags = self._flags
        if flags is None:
            return False
        n = inp.nbytes
        return (
            n > self._flags_z_min_bytes
            and (n <= flags.max_bytes or self._flags_z_upstream_from <= n <= self._flags_z_max_bytes)
            and self._flags_z.eligible(inp)
        )

    def _one_shot_takes(self, inp: torch.Tensor) -> bool:
        flags = self._flags
        return (
            flags is not None
            and inp.nbytes <= self._flags_one_shot_max_bytes
            and flags.eligible(inp)
        )

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        flags_z = self._flags_z
        if flags_z is None:
            return self._custom_all_reduce_base(input)
        if self._flags_z_takes(input):
            run = flags_z.all_reduce
        elif self._one_shot_takes(input):
            flags = self._flags

            def run(inp, out=None):
                return flags.all_reduce(inp, out, one_shot=True)
        else:
            return self._custom_all_reduce_base(input)
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
            # warm-up before capture: mimic the allocation pattern
            return torch.empty_like(input)
        if self._flags_z_hot:
            from vllm_sm80 import hot_swap

            return hot_swap.pair(
                COMPRESS_HOT_PAIR,
                lambda: self._custom_all_reduce_base(input),
                lambda out: run(input, out),
            )
        return run(input)

    def _custom_all_reduce_base(self, input: torch.Tensor) -> torch.Tensor | None:
        # Checks disabled/should_custom_ar once on either path (super() does it
        # for the existing kernel), as the inline version did.
        if self._flags is not None and self._flags.eligible(input):
            if self.disabled or not self.should_custom_ar(input):
                return None
            if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
                # warm-up before capture: mimic the allocation pattern
                return torch.empty_like(input)
            # Reads `input` in place: no IPC registration and no staging copy.
            return self._flags.all_reduce(input)
        return super().custom_all_reduce(input)

    def close(self):
        closing = not self.disabled and self._ptr
        super().close()
        if closing and self._flags is not None:
            self._flags.close()
            self._flags = None
        if closing and self._flags_z is not None:
            self._flags_z.close()
            self._flags_z = None


class Sm80CudaCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name: str = "",
        global_ranks=None,
        global_world_size=None,
        tcp_store_group=None,
        use_all2all: bool = False,
    ):
        from vllm_sm80.distributed.host_shm_all_reduce import HostShmAllreduce

        self.hostshm_comm: HostShmAllreduce | None = None
        # Upstream logs the backend selection at the end of its __init__,
        # before the host-staged path exists here; log once that is known.
        self._defer_backend_log = True
        with p2p_probe_mandatory():
            super().__init__(
                cpu_group,
                device,
                device_group,
                unique_name,
                global_ranks,
                global_world_size,
                tcp_store_group,
                use_all2all,
            )
        self._defer_backend_log = False

        ca_comm = self.ca_comm
        if envs.VLLM_SM80_ALLREDUCE and ca_comm is not None and (
            not ca_comm.disabled
        ):
            ca_comm.__class__ = Sm80CustomAllreduce
            ca_comm._init_flags(not ca_comm.mnnvl_only)

        # Host-staged all-reduce for PCIe-only nodes. Opt-in and TP-only. It
        # stands aside only when VLLM_CUSTOM_ALLREDUCE_PCIE_P2P is set and
        # GPU peer access works; otherwise it serves even where peers are
        # advertised, since ca_comm refuses PCIe above two GPUs without that flag.
        if (
            envs.VLLM_SM80_ALLREDUCE
            and unique_name.split(":")[0] == "tp"
            and self.world_size > 1
            and current_platform.is_cuda()
        ):
            self.hostshm_comm = HostShmAllreduce(
                group=self.cpu_group,
                device=self.device,
                max_size=allreduce_max_bytes(
                    "host_max_bytes", self.world_size, 512 * 1024
                ),
            )
            if self.hostshm_comm.disabled:
                self.hostshm_comm = None
            else:
                self._check_nothing_dispatches_before_hostshm()

        if self.world_size > 1:
            self._log_all_reduce_backend_selection()

    def _check_nothing_dispatches_before_hostshm(self) -> None:
        """Host-staged all-reduce is tried first in ``all_reduce``; upstream
        backends that its dispatch tries before ``ca_comm`` must be off."""
        ahead = []
        if self.fi_pcie_ipc_ar_comm is not None:
            ahead.append("FLASHINFER_PCIE_IPC")
        if self.fi_ar_comm is not None and not self.fi_ar_comm.disabled:
            ahead.append("FLASHINFER")
        if (
            self.pynccl_comm is not None
            and not self.pynccl_comm.disabled
            and is_symmetric_memory_enabled()
        ):
            ahead.append("NCCL_SYMM_MEM")
        if self.qr_comm is not None and not self.qr_comm.disabled:
            ahead.append("QUICK_REDUCE")
        if self.use_aiter_allreduce and self.aiter_ar_comm is not None:
            ahead.append("AITER_CUSTOM")
        if ahead:
            raise RuntimeError(
                "The host-staged all-reduce (VLLM_SM80_ALLREDUCE=1) is tried before "
                f"every other all-reduce backend, but {ahead} would have to come "
                "first; turn those off or unset VLLM_SM80_ALLREDUCE."
            )

    def _log_all_reduce_backend_selection(self) -> None:
        """Upstream's backend-selection line, with HOSTSHM in its dispatch
        place and the PCIe peer-to-peer note. Startup-log checks parse it, so
        the format is upstream's."""
        if self._defer_backend_log:
            return
        all_potential_ar_backends = [
            "FLASHINFER_PCIE_IPC",
            "FLASHINFER",
            "NCCL_SYMM_MEM",
            "QUICK_REDUCE",
            "AITER_CUSTOM",
            "HOSTSHM",
            "CUSTOM",
            "SYMM_MEM",
            "PYNCCL",
        ]
        enabled_ar_backends: list[str] = []
        if (
            self.fi_pcie_ipc_ar_comm is not None
            and not self.fi_pcie_ipc_ar_comm.disabled
        ):
            enabled_ar_backends.append("FLASHINFER_PCIE_IPC")
        if self.fi_ar_comm is not None and not self.fi_ar_comm.disabled:
            enabled_ar_backends.append("FLASHINFER")
        nccl_symm_ws_ok = self.world_size >= NCCL_SYMM_MEM_ALL_REDUCE_CONFIG[
            "min_world_size"
        ] and (
            self.world_size
            in NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["custom_ar_preferred_ranges"]
            or self.world_size
            > NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
        )
        if (
            self.pynccl_comm is not None
            and not self.pynccl_comm.disabled
            and is_symmetric_memory_enabled()
            and not envs.VLLM_BATCH_INVARIANT
            and nccl_symm_ws_ok
        ):
            enabled_ar_backends.append("NCCL_SYMM_MEM")
        if self.qr_comm is not None and not self.qr_comm.disabled:
            enabled_ar_backends.append("QUICK_REDUCE")
        if (
            self.use_aiter_allreduce
            and self.aiter_ar_comm is not None
            and not self.aiter_ar_comm.disabled
        ):
            enabled_ar_backends.append("AITER_CUSTOM")
        if self.hostshm_comm is not None and not self.hostshm_comm.disabled:
            enabled_ar_backends.append("HOSTSHM")
        if self.ca_comm is not None and not self.ca_comm.disabled:
            enabled_ar_backends.append("CUSTOM")
        if self.symm_mem_comm is not None and not self.symm_mem_comm.disabled:
            enabled_ar_backends.append("SYMM_MEM")
        if self.pynccl_comm is not None and not self.pynccl_comm.disabled:
            enabled_ar_backends.append("PYNCCL")

        # HOSTSHM and CUSTOM are mutually exclusive by the operator's choice
        # (see host_shm_all_reduce._stand_aside_for_custom_allreduce), so this
        # one line says which of the two PCIe fast paths won.
        pcie_p2p_note = ""
        if envs.VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:
            pcie_p2p_note = (
                " VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1: PCIe peer-to-peer counts"
                " as fully connected for the custom all-reduce, and the"
                " host-staged path stands aside wherever peer access is present."
            )
        transport_note = ""
        if envs.VLLM_SM80_ALLREDUCE:
            transport_note = (
                " VLLM_SM80_ALLREDUCE=1: the fork's all-reduce transports take"
                " each message by world size, bytes, dtype and CUDA graph"
                " capture."
            )
        logger.info_once(
            "Using %s all-reduce backends (in dispatch order) for group "
            "'%s' out of potential backends: %s.%s%s",
            "[" + ", ".join(f"'{b}'" for b in enabled_ar_backends) + "]",
            self.unique_name or "<unnamed>",
            "[" + ", ".join(f"'{b}'" for b in all_potential_ar_backends) + "]",
            pcie_p2p_note,
            transport_note,
            scope="global",
        )

    def all_reduce(self, input_):
        # Before every upstream backend on purpose: hostshm exists only when it
        # did not stand aside for the P2P custom all-reduce, so when it exists
        # it is the path the operator chose (and the backends upstream tries
        # before ca_comm are checked off at construction).
        hostshm_comm = self.hostshm_comm
        if hostshm_comm is not None and hostshm_comm.should_host_ar(input_):
            out = hostshm_comm.host_all_reduce(input_)
            assert out is not None
            return out
        return super().all_reduce(input_)

    def destroy(self):
        if self.hostshm_comm is not None:
            self.hostshm_comm.close()
            self.hostshm_comm = None
        super().destroy()
