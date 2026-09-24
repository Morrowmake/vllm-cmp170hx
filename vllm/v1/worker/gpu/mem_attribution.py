# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time and profiling-peak memory attribution (VLLM_GLM5_MEM_ATTRIBUTION).

With the flag set, one boot logs, per rank:

* after weight loading, the resident bytes of every module group of the target
  model and the draft model (layer indices folded to ``*``, storages counted
  once), and what ``memory_allocated`` holds beyond them;
* during ``profile_run``, the peak and end allocation of each stage (the
  multimodal encoder profile, the LM dummy run, the sampler) over the profile
  start, with the stage that owns the peak and its margin over the next one --
  only memory live at that high-water point turns into KV cache;
* after profiling, the worker's accounting (weights, torch peak increase,
  non-torch, CUDA-graph estimate, available KV) in one line;
* the allocations made by the attention metadata builders after the KV cache
  is sized, which the profile never sees.

The overall peak is restored after the per-stage resets, so the profile result
and the KV cache size are the same as without the flag. Logging only; nothing
here changes an allocation the model uses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_MiB = float(1 << 20)
_TAG = "[MEM-ATTR]"
_LAYER_INDEX = re.compile(r"\.\d+(?=\.|$)")


def enabled() -> bool:
    return bool(envs.VLLM_GLM5_MEM_ATTRIBUTION)


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        return torch.accelerator.current_device_index()


def group_key(name: str) -> str:
    """Module path of a parameter or buffer with layer indices folded."""
    module = name.rsplit(".", 1)[0] if "." in name else name
    return _LAYER_INDEX.sub(".*", module)


def module_inventory(
    models: Iterable[tuple[str, nn.Module | None]],
    include_cpu: bool = False,
) -> tuple[dict[str, int], int]:
    """Bytes per ``<tag>:<group>`` over the device tensors of the given models,
    each storage counted once (tied weights, shared caches)."""
    seen: set[int] = set()
    groups: dict[str, int] = defaultdict(int)
    total = 0
    for tag, model in models:
        if model is None:
            continue
        tensors = list(model.named_parameters(remove_duplicate=False)) + list(
            model.named_buffers(remove_duplicate=False)
        )
        for name, t in tensors:
            if t is None or t.device.type == "meta":
                continue
            if t.device.type == "cpu" and not include_cpu:
                continue
            storage = t.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in seen or ptr == 0:
                continue
            seen.add(ptr)
            nbytes = storage.nbytes()
            groups[f"{tag}:{group_key(name)}"] += nbytes
            total += nbytes
    return dict(groups), total


def log_load_inventory(
    models: Iterable[tuple[str, nn.Module | None]],
    consumed_bytes: int,
    top: int = 60,
) -> None:
    if not enabled():
        return
    torch.accelerator.synchronize()
    groups, total = module_inventory(models)
    allocated = torch.accelerator.memory_allocated()
    rank = _rank()
    ranked = sorted(groups.items(), key=lambda kv: -kv[1])
    if rank == 0:
        for key, nbytes in ranked[:top]:
            logger.info(
                "%s load rank=%d %10.1f MiB  %s", _TAG, rank, nbytes / _MiB, key
            )
        rest = sum(n for _, n in ranked[top:])
        if rest:
            logger.info(
                "%s load rank=%d %10.1f MiB  (%d smaller groups)",
                _TAG,
                rank,
                rest / _MiB,
                len(ranked) - top,
            )
    logger.info(
        "%s load rank=%d modules=%.1f MiB loading_consumed=%.1f MiB "
        "allocated=%.1f MiB allocated_not_in_modules=%.1f MiB",
        _TAG,
        rank,
        total / _MiB,
        consumed_bytes / _MiB,
        allocated / _MiB,
        (allocated - total) / _MiB,
    )


def workspace_bytes() -> int:
    try:
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        if not is_workspace_manager_initialized():
            return 0
        mgr = current_workspace_manager()
        return sum(mgr._workspace_size_bytes(ws) for ws in mgr._current_workspaces)
    except Exception:  # noqa: BLE001
        return 0


class ProfileStages:
    """Per-stage peaks inside ``profile_run``; a no-op without the flag."""

    def __init__(self) -> None:
        self.on = enabled()
        self.stages: list[tuple[str, float, float, float]] = []
        if not self.on:
            return
        torch.accelerator.synchronize()
        self._start = torch.accelerator.memory_allocated()
        self._stage_start = self._start
        self._overall = torch.accelerator.max_memory_allocated()
        torch.accelerator.reset_peak_memory_stats()

    def stage(self, name: str) -> None:
        if not self.on:
            return
        torch.accelerator.synchronize()
        peak = torch.accelerator.max_memory_allocated()
        end = torch.accelerator.memory_allocated()
        self.stages.append(
            (
                name,
                (self._stage_start - self._start) / _MiB,
                (peak - self._start) / _MiB,
                (end - self._start) / _MiB,
            )
        )
        logger.info(
            "%s profile rank=%d stage=%s start=+%.1f peak=+%.1f end=+%.1f MiB "
            "(over profile start %.1f MiB) workspace=%.1f MiB",
            _TAG,
            _rank(),
            name,
            (self._stage_start - self._start) / _MiB,
            (peak - self._start) / _MiB,
            (end - self._start) / _MiB,
            self._start / _MiB,
            workspace_bytes() / _MiB,
        )
        self._overall = max(self._overall, peak)
        self._stage_start = end
        torch.accelerator.reset_peak_memory_stats()

    def finish(self) -> None:
        """Restore the overall peak so the caller's profile is unchanged, and
        name the stage that owns it."""
        if not self.on:
            return
        cur = torch.accelerator.memory_allocated()
        if self._overall > cur:
            pad = torch.empty(self._overall - cur, dtype=torch.uint8, device="cuda")
            del pad
        torch.accelerator.synchronize()
        if self.stages:
            ranked = sorted(self.stages, key=lambda s: -s[2])
            owner = ranked[0]
            margin = owner[2] - ranked[1][2] if len(ranked) > 1 else owner[2]
            logger.info(
                "%s profile rank=%d peak owner=%s at +%.1f MiB; next=%s at +%.1f "
                "MiB; margin %.1f MiB (a cut in the owner stage frees KV only up "
                "to this margin)",
                _TAG,
                _rank(),
                owner[0],
                owner[2],
                ranked[1][0] if len(ranked) > 1 else "-",
                ranked[1][2] if len(ranked) > 1 else 0.0,
                margin,
            )


def log_kv_accounting(
    requested: int,
    profile_result,
    cudagraph_estimate: int,
    cudagraph_applied: int,
    available: int,
) -> None:
    """The worker's KV sizing in one line: available = requested - non_kv -
    applied CUDA-graph estimate, non_kv = total_consumed + transient headroom."""
    if not enabled():
        return
    r = profile_result
    logger.info(
        "%s kv rank=%d requested=%.1f weights=%.1f torch_peak_increase=%.1f "
        "non_torch_increase=%.1f total_consumed=%.1f transient_headroom=%.1f "
        "non_kv=%.1f cudagraph_estimate=%.1f (applied %.1f) available_kv=%.1f "
        "MiB workspace=%.1f MiB",
        _TAG,
        _rank(),
        requested / _MiB,
        r.weights_memory / _MiB,
        r.torch_peak_increase / _MiB,
        r.non_torch_increase / _MiB,
        r.total_consumed / _MiB,
        r.transient_peak_headroom / _MiB,
        r.non_kv_cache_memory / _MiB,
        cudagraph_estimate / _MiB,
        cudagraph_applied / _MiB,
        available / _MiB,
        workspace_bytes() / _MiB,
    )


class AfterKvAllocations:
    """Bytes allocated by a block that runs after KV sizing (metadata builders)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.on = enabled()

    def __enter__(self) -> AfterKvAllocations:
        if self.on:
            torch.accelerator.synchronize()
            self._before = torch.accelerator.memory_allocated()
        return self

    def __exit__(self, *exc) -> None:
        if not self.on or exc[0] is not None:
            return
        torch.accelerator.synchronize()
        delta = torch.accelerator.memory_allocated() - self._before
        logger.info(
            "%s builders rank=%d %s: attention metadata builders allocated "
            "%.1f MiB (outside the memory profile)",
            _TAG,
            _rank(),
            self.label,
            delta / _MiB,
        )
