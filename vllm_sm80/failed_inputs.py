# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail only the requests whose model inputs a worker could not produce.

A model module may report, after a forward, a bool mask over the forward's
tokens whose inputs it could not produce (for example an EXL3 PLE row that
storage failed to read): ``take_failed_tokens() -> torch.Tensor | None``,
which also clears the mask. The GPU model runner collects the masks with
:func:`take_failed_tokens`, ``AsyncOutput`` maps them to request IDs with
:func:`failed_request_ids` into ``ModelRunnerOutput.failed_req_ids``, and the
scheduler calls :func:`handle_failed_inputs`: those requests finish with
FINISHED_ERROR, and the KV computed from their failed step on leaves the
prefix cache together with every running request that reused it.

Upstream files keep only the calls (scheduler.py, outputs.py,
gpu/async_utils.py, gpu/model_runner.py); the KVCacheManager internals read
here are pinned in tests/sm80/seams.yaml.
"""

import functools
import weakref
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)

# Model -> its modules that report failed tokens, found once per model.
_SOURCES: "weakref.WeakKeyDictionary[nn.Module, tuple[nn.Module, ...]]" = (
    weakref.WeakKeyDictionary()
)


def take_failed_tokens(model: nn.Module) -> torch.Tensor | None:
    """OR of the failed-token masks the model's modules report for the last
    forward, or None when none failed. Taking the masks clears them."""
    sources = _SOURCES.get(model)
    if sources is None:
        sources = tuple(
            module
            for module in model.modules()
            if callable(getattr(module, "take_failed_tokens", None))
        )
        _SOURCES[model] = sources
    if not sources:
        return None
    masks = [
        mask
        for mask in (source.take_failed_tokens() for source in sources)
        if mask is not None
    ]
    return functools.reduce(torch.logical_or, masks) if masks else None


def failed_request_ids(
    failed_tokens: np.ndarray, query_start_loc: np.ndarray, req_ids: list[str]
) -> set[str]:
    """Map a per-token failure mask to the requests owning those tokens.

    Tokens past the last request (padding) belong to no request.
    """
    num_reqs = len(req_ids)
    tokens = np.flatnonzero(failed_tokens[: query_start_loc[num_reqs]])
    if tokens.size == 0:
        return set()
    owners = np.searchsorted(query_start_loc[1 : num_reqs + 1], tokens, side="right")
    return {req_ids[i] for i in np.unique(owners).tolist()}


def evict_blocks_from(
    kv_cache_manager: "KVCacheManager", request_id: str, num_tokens: int
) -> tuple[set[int], ...]:
    """Evict a request's blocks holding tokens at or after `num_tokens`
    from the prefix cache.

    Returns, per KV cache group, the evicted block IDs that another
    request can hold only if it reused those tokens. Groups without
    positionally stable blocks lose all of the request's blocks; they
    return no IDs when a positional group exists, because reusing their
    later states implies reusing that group's later blocks.
    """
    coordinator = kv_cache_manager.coordinator
    managers = coordinator.single_type_managers
    any_stable = any(m.has_positionally_stable_blocks for m in managers)
    reused = []
    for manager, blocks in zip(managers, coordinator.get_blocks(request_id)):
        stable = manager.has_positionally_stable_blocks
        first = num_tokens // manager.block_size if stable else 0
        block_ids = {b.block_id for b in blocks[first:] if not b.is_null}
        manager.block_pool.evict_blocks(block_ids)
        reused.append(block_ids if stable or not any_stable else set())
    return tuple(reused)


def handle_failed_inputs(scheduler: "Scheduler", failed_req_ids: set[str]) -> set[str]:
    """Handle requests whose model inputs failed on a worker.

    KV computed from the failed step on is wrong. Drop it from the prefix
    cache and also fail running requests that already reused it, so no
    request is served from those inputs.

    Returns:
        Set of request IDs to skip in update_from_output and finish with
        FINISHED_ERROR.
    """
    kv_cache_manager = scheduler.kv_cache_manager
    affected: set[str] = set()
    reused: list[set[int]] = []
    for req_id in failed_req_ids:
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished():
            continue
        affected.add(req_id)
        # num_in_flight_tokens still counts the failed step's tokens.
        first_token = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
        group_ids = evict_blocks_from(kv_cache_manager, req_id, first_token)
        reused = (
            [a | b for a, b in zip(reused, group_ids)] if reused else list(group_ids)
        )
    if any(reused):
        for request in scheduler.running:
            if request.request_id not in affected and any(
                not group.isdisjoint(ids)
                for group, ids in zip(
                    reused, kv_cache_manager.get_block_ids(request.request_id)
                )
            ):
                affected.add(request.request_id)
    if affected:
        logger.error(
            "Failing %d request(s) whose model inputs could not be produced "
            "on a worker, or that reused KV computed from them: %s",
            len(affected),
            sorted(affected),
        )
    return affected
