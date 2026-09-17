# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group

# Optional redirection of the tensor-parallel all-reduce.
#
# Normally ``None``, in which case ``tensor_model_parallel_all_reduce`` behaves
# exactly as it always has. A model that wants to schedule the collective
# itself -- e.g. the GLM-5.3-Flash prefill comm/compute overlap in
# ``vllm/models/glm5next/common/overlap.py``, which issues it on a side CUDA
# stream so it runs concurrently with the next micro-batch's compute -- installs
# a callable here for the duration of a narrow, well-understood region and
# restores the previous value afterwards.
#
# The interceptor must return a tensor of the same shape/dtype/device that
# *will* hold the reduced result; it is the installer's responsibility to make
# the consuming stream wait for the collective before that tensor is read.
_tp_all_reduce_interceptor: Callable[[torch.Tensor], torch.Tensor] | None = None


def set_tp_all_reduce_interceptor(
    fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Install (or clear) the all-reduce interceptor; returns the previous one.

    Callers must restore the returned value in a ``finally`` block.
    """
    global _tp_all_reduce_interceptor
    previous = _tp_all_reduce_interceptor
    _tp_all_reduce_interceptor = fn
    return previous


def get_tp_all_reduce_interceptor() -> Callable[[torch.Tensor], torch.Tensor] | None:
    return _tp_all_reduce_interceptor


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    interceptor = _tp_all_reduce_interceptor
    if interceptor is not None:
        return interceptor(input_)
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
