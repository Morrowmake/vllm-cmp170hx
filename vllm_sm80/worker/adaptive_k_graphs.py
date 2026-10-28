# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs for the adaptive-k draft counts (``vllm_sm80.spec_decode.adaptive_k``).

The scheduler picks each step's draft count on the host, so every count it may
pick needs its own uniform decode graph in the target model's graph manager
(vLLM's ``ModelCudaGraphManager``; drafter managers keep their one query
length):

- :func:`decode_query_lens`: the per-count query lengths and request caps
  ``CudaGraphManager._init_candidates`` captures (its one call into the
  plugin);
- :func:`warn_missing`: after capture, a startup warning for a count no
  capture size covers (the runner subclass's ``capture_model``);
- :func:`note_graph_miss`: a decode step that ran below the configured decode
  mode (the runner's dispatch hook).
"""

from __future__ import annotations

from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm_sm80.spec_decode.adaptive_k import config_of

logger = init_logger(__name__)


def adaptive_k_config(vllm_config: Any) -> Any:
    """The adaptive-k configuration (``AdaptiveKConfig``), or None."""
    return config_of(vllm_config)


def draft_counts(manager: Any) -> tuple[int, ...]:
    """Draft counts the adaptive-k policy may choose, or () for a manager
    other than the target model's or a config without adaptive-k counts."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    if not isinstance(manager, ModelCudaGraphManager):
        return ()
    config = adaptive_k_config(manager.vllm_config)
    return tuple(config.allowed) if config is not None else ()


def decode_query_lens(manager: Any) -> tuple[list[int], dict[int, int]] | None:
    """The uniform decode query lengths to capture, one per draft count, and
    per length the most requests a uniform decode batch of it can hold
    (load-adaptive depth verifies deep drafts only when few requests run);
    None without adaptive-k counts.

    Without a graph per count, dispatch() silently falls back to eager;
    :func:`warn_missing` checks at startup that the capture sizes actually
    cover the set. Branches on the counts themselves rather than on
    uses_adaptive_k(), so a config that reports the feature on but names no
    counts still lands on the ordinary single-query-length path."""
    counts = draft_counts(manager)
    if not counts:
        return None
    vllm_config = manager.vllm_config
    num_new_sampled_tokens_per_step = (
        manager.decode_query_len - vllm_config.num_speculative_tokens
    )
    lens = sorted(num_spec + num_new_sampled_tokens_per_step for num_spec in counts)
    config = adaptive_k_config(vllm_config)
    max_reqs = {
        num_spec + num_new_sampled_tokens_per_step: (
            config.max_reqs_for(num_spec, manager.max_num_reqs)
        )
        for num_spec in counts
    }
    return lens, max_reqs


def warn_missing(manager: Any) -> None:
    """Check every adaptive draft count has a single-request decode graph.

    ``dispatch`` returns CUDAGraphMode.NONE on a miss and the step quietly
    runs eager, which would read as a mysterious slowdown rather than a
    misconfiguration, so surface it once at startup.
    """
    counts = draft_counts(manager)
    if not counts:
        return
    if not manager.cudagraph_mode or not manager.cudagraph_mode.decode_mode():
        return

    num_new_sampled_tokens_per_step = (
        manager.decode_query_len - manager.vllm_config.num_speculative_tokens
    )
    covered, missing = [], []
    for num_spec in counts:
        query_len = num_spec + num_new_sampled_tokens_per_step
        desc = manager.dispatch(
            num_reqs=1,
            num_tokens=query_len,
            uniform_token_count=query_len,
            num_active_loras=0,
            max_query_len=query_len,
        )
        (covered if desc.cg_mode != CUDAGraphMode.NONE else missing).append(num_spec)
    if missing:
        logger.warning(
            "Adaptive SD: no decode CUDA graph for draft count(s) %s "
            "(need capture sizes that are multiples of %s); steps that "
            "choose them will run eager. Captured: %s.",
            missing,
            [k + num_new_sampled_tokens_per_step for k in missing],
            covered,
        )
    else:
        logger.info(
            "Adaptive SD: decode CUDA graphs captured for draft counts %s.",
            covered,
        )


def is_decode_graph_miss(manager: Any, desc: Any) -> bool:
    """Whether a uniform decode batch dispatched below the configured decode
    mode: PIECEWISE or eager when FULL decode graphs were asked for, eager
    when PIECEWISE."""
    return desc.cg_mode != manager.cudagraph_mode.decode_mode()


def note_graph_miss(
    runner: Any, batch_desc: Any, num_reqs: int, num_toks: int, uniform_tok_count: int
) -> None:
    """Count a uniform decode step that missed its decode CUDA graph mode and
    warn on the 1st, 2nd, 4th, 8th, ... such step, so an unregistered draft
    count surfaces instead of a silent PIECEWISE or eager fallback."""
    manager = runner.cudagraph_manager
    if not is_decode_graph_miss(manager, batch_desc):
        return
    runner._adaptive_k_cg_misses = getattr(runner, "_adaptive_k_cg_misses", 0) + 1
    misses = runner._adaptive_k_cg_misses
    if misses & (misses - 1) == 0:
        logger.warning(
            "Adaptive SD: uniform decode batch (num_reqs=%d, "
            "num_tokens=%d, tokens/req=%d) has no captured CUDA graph "
            "in %s decode mode and ran %s (%d such steps so far).",
            num_reqs,
            num_toks,
            uniform_tok_count,
            manager.cudagraph_mode.decode_mode().name,
            batch_desc.cg_mode.name,
            misses,
        )
