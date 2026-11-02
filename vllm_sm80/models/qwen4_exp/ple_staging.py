# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE storage selection and per-step row staging.

Two PLE storages of the fork fill a persistent GPU row buffer before the
forward instead of looking rows up inside it:

- the file-backed BF16/FP16 table (``VLLM_SM80_PLE_MMAP``,
  :mod:`vllm_sm80.models.qwen4_exp.mmap_ple`), on every step;
- the compressed EXL3 table (:mod:`vllm_sm80.models.qwen4_exp.exl3_ple`) for
  steps that replay a FULL CUDA graph, which cannot hold its SSD reads.
  PIECEWISE and eager steps keep its overlapping reader.

Upstream's ``Qwen4ExpNGramEmbedding`` drives both through the prefetch
protocol it already has (``supports_prefetch``, ``start_prefetch``,
``finalize_prefetch``): a staged step's ``start_prefetch`` does nothing and
``finalize_prefetch`` returns the staged rows. What stays outside it:

- :func:`embedding_cls`: ``Qwen4ExpNGramEmbedding.__init__`` (queue hook),
  the storage class;
- :func:`configure`: the fork platform's ``check_and_update_config``, the
  loader settings the file-backed table needs;
- :func:`finalize`: after the model loads (``runner_hooks.after_drafter_load``),
  the storages check their tables and allocate their row buffers before any
  graph capture;
- :func:`stage_rows`: ``execute_model``, after the model inputs are prepared
  (``runner_hooks.stage_ple_rows``), the step's rows.
"""

import weakref
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import vllm.envs as envs
from vllm_sm80.plugin import register_fork_env_vars

# embedding_cls reads VLLM_SM80_PLE_MMAP for every Qwen4Exp n-gram embedding,
# also in a process where no fork entry point ran.
register_fork_env_vars()

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

# Model -> its n-gram embedding modules whose storage stages rows.
_STAGED: "weakref.WeakKeyDictionary[nn.Module, tuple[nn.Module, ...]]" = (
    weakref.WeakKeyDictionary()
)


def embedding_cls(quant_config: Any, default: type) -> type:
    """The PLE storage class of one ``Qwen4ExpNGramEmbedding``: the
    quantization config's (``ple_embedding_cls``, EXL3), else the file-backed
    table with ``VLLM_SM80_PLE_MMAP``, else upstream's ``default``."""
    quant_cls = getattr(quant_config, "ple_embedding_cls", None)
    if not envs.VLLM_SM80_PLE_MMAP:
        return default if quant_cls is None else quant_cls
    if quant_cls is not None:
        raise ValueError(
            "VLLM_SM80_PLE_MMAP keeps BF16/FP16 PLE tables file-backed; this "
            f"checkpoint's quantization ({type(quant_config).__name__}) brings "
            "its own PLE storage. Unset VLLM_SM80_PLE_MMAP."
        )
    from vllm_sm80.models.qwen4_exp.mmap_ple import Qwen4ExpPLEMmapEmbedding

    return Qwen4ExpPLEMmapEmbedding


def configure(vllm_config: "VllmConfig") -> None:
    """``VLLM_SM80_PLE_MMAP``: check the model and layout, and select the lazy
    safetensors loader, which keeps checkpoint shards memory-mapped."""
    if not envs.VLLM_SM80_PLE_MMAP:
        return
    model_config = vllm_config.model_config
    if model_config is None:
        return
    if not getattr(model_config.hf_text_config, "ple_layer_ids", None):
        raise ValueError("VLLM_SM80_PLE_MMAP needs a Qwen4Exp model with PLE layers")
    if model_config.quantization == "exl3":
        raise ValueError(
            "VLLM_SM80_PLE_MMAP is not for EXL3 checkpoints, whose PLE tables "
            "use their own storage"
        )
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError("VLLM_SM80_PLE_MMAP requires tensor_parallel_size=1")
    load_config = vllm_config.load_config
    if load_config.load_format == "auto":
        load_config.load_format = "safetensors"
    elif load_config.load_format != "safetensors":
        raise ValueError(
            "VLLM_SM80_PLE_MMAP requires the safetensors loader, not "
            f"{load_config.load_format!r}"
        )
    if load_config.safetensors_load_strategy is None:
        load_config.safetensors_load_strategy = "lazy"
    elif load_config.safetensors_load_strategy != "lazy":
        raise ValueError(
            "VLLM_SM80_PLE_MMAP requires safetensors_load_strategy='lazy' so "
            "PLE tables stay memory-mapped, not "
            f"{load_config.safetensors_load_strategy!r}"
        )


def staged_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    """The model's n-gram embedding modules whose storage stages rows (this
    pipeline stage's only)."""
    if not isinstance(model, nn.Module):
        return ()
    modules = _STAGED.get(model)
    if modules is None:
        modules = tuple(
            module
            for module in model.modules()
            if getattr(
                getattr(module, "ngram_embedding", None), "staging_enabled", False
            )
        )
        _STAGED[model] = modules
    return modules


def finalize(model: nn.Module) -> None:
    """After loading: each staging storage checks its table and allocates its
    row buffer, which graph capture then reads."""
    for module in staged_modules(model):
        module.ngram_embedding.finalize_staging()


def stage_rows(
    model: nn.Module,
    input_batch: "InputBatch",
    req_states: "RequestState",
    model_inputs: dict[str, Any],
    cg_mode: "CUDAGraphMode",
    dummy_run: bool,
) -> None:
    """Stage this step's PLE rows before the forward.

    A dummy or profiling batch (no registered request) zeroes the rows of
    every staging storage, so no previous request's rows or read failures
    carry into it.

    Raises:
        RuntimeError: A batch mixes registered and unknown requests.
    """
    modules = staged_modules(model)
    if not modules:
        return
    padded = input_batch.num_tokens_after_padding
    known = [req_id in req_states.req_id_to_index for req_id in input_batch.req_ids]
    if not dummy_run and any(known) and not all(known):
        raise RuntimeError("PLE batch contains unknown active request IDs")
    if dummy_run or not all(known):
        for module in modules:
            module.ngram_embedding.zero_rows(padded)
        return
    num_tokens = input_batch.num_tokens
    for module in modules:
        embedding = module.ngram_embedding
        if not embedding.stages(cg_mode):
            continue
        ids = module._prefetch_ids[:num_tokens]
        if num_tokens:
            module.compute_ngram_ids(
                input_batch.input_ids[:num_tokens],
                model_inputs["query_start_loc"],
                model_inputs["ngram_context"],
                output=ids,
            )
        embedding.prepare(ids, padded)


def forward_is_staged() -> bool:
    """Whether the running forward replays, or is captured into, a FULL
    graph. The runner captures FULL graphs with runtime mode NONE, so a NONE
    forward on a capturing stream counts."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    mode = (
        get_forward_context().cudagraph_runtime_mode
        if is_forward_context_available()
        else None
    )
    return mode == CUDAGraphMode.FULL or (
        mode == CUDAGraphMode.NONE
        and torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    )
