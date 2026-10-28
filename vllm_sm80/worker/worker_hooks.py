# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The calls vLLM's GPU worker (``vllm/v1/worker/gpu_worker.py``) makes into
the plugin from inside ``determine_available_memory`` and ``execute_model``,
where no extension point exists. The rest of the fork worker is the worker
subclass (``vllm_sm80.worker.graph_mem_worker``).

- :func:`kv_accounting`: the KV-cache memory accounting line
  (VLLM_MEM_ATTRIBUTION).
- :func:`hop_recv` / :func:`hop_send`: the packed pipeline-parallel hop
  (``VLLM_PP_PACKED_HOP``, ``vllm_sm80.worker.pp_hop``) in place of the
  intermediate tensors' ``irecv_tensor_dict`` / ``isend_tensor_dict``; None
  when the hop is off, so the worker takes vLLM's transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def kv_accounting(worker: Any, profile_result: Any) -> None:
    """``determine_available_memory``, once the KV cache size is known."""
    from vllm_sm80.worker import mem_attribution

    estimate = worker.cudagraph_memory_estimate
    mem_attribution.log_kv_accounting(
        worker.requested_memory,
        profile_result,
        estimate,
        estimate if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS else 0,
        worker.available_kv_cache_memory_bytes,
    )


def _pp_hop(worker: Any) -> Any:
    """The packed PP hop when VLLM_PP_PACKED_HOP=1 and the configuration
    allows it, else None. Resolved once per worker, on its first hop."""
    if getattr(worker, "_pp_hop_resolved", False):
        return worker._pp_hop
    worker._pp_hop_resolved = True
    worker._pp_hop = None
    if not envs.VLLM_PP_PACKED_HOP:
        return None
    from vllm.distributed import get_pp_group, get_tp_group

    pp = get_pp_group()
    if pp.world_size <= 1:
        return None
    reason = None
    if not worker.use_v2_model_runner:
        reason = "needs the V2 model runner"
    elif get_tp_group().world_size > 1:
        reason = "TP > 1 (the default hop slices tensors across TP ranks)"
    elif getattr(pp, "use_cpu_custom_send_recv", False):
        reason = "custom CPU send/recv communicator"
    if reason is not None:
        logger.warning("VLLM_PP_PACKED_HOP ignored: %s", reason)
        return None
    from vllm_sm80.worker.pp_hop import PackedHop

    no_metadata = envs.VLLM_PP_HOP_NO_METADATA
    worker._pp_hop = PackedHop(pp, no_metadata=no_metadata)
    logger.info(
        "PP packed hop on: one device transfer per hop%s",
        ", no per-step metadata" if no_metadata else "",
    )
    return worker._pp_hop


def _hop_rows(runner: Any, scheduler_output: SchedulerOutput) -> int | None:
    """Rows a packed PP hop carries this step without metadata: the
    scheduled token count, which every stage reads from the same scheduler
    output. None where a stage may run a different token count (adaptive
    verification, prefill context parallelism, micro-batching); the hop
    then sends the metadata for the step."""
    if (
        runner.adaptive_verification is not None
        or runner.pcp_manager is not None
        or runner.ubatch_runner is not None
    ):
        return None
    return scheduler_output.total_num_scheduled_tokens


def hop_recv(worker: Any, scheduler_output: SchedulerOutput) -> tuple | None:
    """``execute_model`` on a non-first stage with tokens to run: the
    intermediate tensors through the packed hop, as ``irecv_tensor_dict``
    returns them (tensors, device handles, postprocess), or None."""
    hop = _pp_hop(worker)
    if hop is None:
        return None
    layout = getattr(worker, "_pp_hop_layout", None)
    if layout is None:
        from vllm_sm80.worker.pp_hop import HopLayout

        layout = worker._pp_hop_layout = HopLayout.from_tensors(
            worker.model_runner.intermediate_tensors.tensors
        )
    tensor_dict, comm_handles = hop.recv(
        layout,
        _hop_rows(worker.model_runner, scheduler_output),
        device=worker.device,
    )
    return tensor_dict, comm_handles, []


def hop_send(
    worker: Any, scheduler_output: SchedulerOutput, tensors: dict[str, torch.Tensor]
) -> list | None:
    """``execute_model`` on a non-last stage: one flat device send of the
    intermediate tensors through the packed hop, as ``isend_tensor_dict``
    returns its handles (the metadata handle first; the hop retains its
    buffer and metadata, so it has none), or None."""
    hop = _pp_hop(worker)
    if hop is None:
        return None
    return [None, *hop.send(tensors, _hop_rows(worker.model_runner, scheduler_output))]
