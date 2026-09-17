# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-capture warmup for the sm_80 thin-M BF16 GEMM.

WHY THIS MUST RUN BEFORE CUDA GRAPH CAPTURE. ``thin_gemm`` keeps its FP32
split-K partials and its arrival counters in an allocate-once module-level
cache (``thin_gemm._WORKSPACE``), which is what makes a graph replay correct:
the pointer the graph recorded stays valid for the life of the process, and the
kernel leaves the counters at zero so a buffer is reusable across replays and
shapes. But a first call made *during* capture would put that allocation in the
graph's private pool, and a later capture would then reuse freed memory. Triton
also compiles on first launch, which cannot happen inside a capture at all.

So this runs from ``vllm/model_executor/warmup/kernel_warmup.py``, which
``Worker.compile_or_warm_up_model`` calls immediately before
``capture_model()``.

Shapes are discovered from the live model rather than hard-coded, so the warmup
follows TP size, the checkpoint's ignore list and the drafter architecture
without being told. ``_TABLE_NK`` is a backstop for shapes whose module tree the
walk cannot reach (a drafter held somewhere this does not look); warming a shape
that never runs costs one compile of an already-compiled config.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Small M values an eager (non-captured) decode step can hit.
_EXTRA_MS = (1, 2, 3, 4, 8)

# (N, K) per TP rank at TP=4 for GLM-5.3-Flash-W4A16-MTP + the DFlash drafter,
# from the standalone thin-GEMM shape inventory. Backstop only -- the live walk
# below is authoritative and covers any TP size.
_TABLE_NK = (
    (6416, 4096), (2048, 128), (4096, 2048), (2048, 4096), (4096, 1536),
    (8192, 512), (4096, 4096), (160, 4096), (128, 4096), (6144, 4096),
    (4096, 3072), (288, 4096), (1024, 4096), (4096, 512), (38720, 4096),
    (4096, 20480), (2560, 4096), (1536, 4096), (4096, 1024),
)


def _token_sizes(capture_sizes, bound: int) -> list[int]:
    sizes = {int(s) for s in capture_sizes if 1 <= int(s) <= bound}
    sizes.update(m for m in _EXTRA_MS if m <= bound)
    return sorted(sizes)


def _candidate_models(worker):
    """The target model plus any drafter reachable from the runner."""
    models = []
    try:
        models.append(worker.get_model())
    except Exception:  # pragma: no cover - worker shape varies by backend
        pass
    runner = getattr(worker, "model_runner", None)
    for attr in ("drafter", "speculator"):
        obj = getattr(runner, attr, None)
        if obj is None:
            continue
        # The drafter may be the module itself or a wrapper holding `.model`.
        for inner in (getattr(obj, "model", None), obj):
            if isinstance(inner, torch.nn.Module):
                models.append(inner)
                break
    return models


def _discover_nk(worker) -> set[tuple[int, int]]:
    """Every (N, K) that will reach the unquantized BF16 GEMM path.

    A module qualifies when it owns a 2-D BF16 ``weight`` that is K-contiguous
    and its quant method is one of the unquantized ones -- exactly the
    predicate ``thin_gemm_supported`` applies at run time, minus M.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    unquant = (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)
    found: set[tuple[int, int]] = set()
    for model in _candidate_models(worker):
        for _name, module in model.named_modules():
            if not isinstance(getattr(module, "quant_method", None), unquant):
                continue
            w = getattr(module, "weight", None)
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 2
                and w.dtype is torch.bfloat16
                and w.stride(1) == 1
            ):
                found.add((int(w.shape[0]), int(w.shape[1])))
    return found


def warmup_ampere_thin_gemm(worker, capture_sizes) -> None:
    """Compile and preallocate for every (M, N, K) the graphs can replay."""
    from vllm.ampere_thin_gemm import dispatch_threshold, use_ampere_thin_gemm

    if not use_ampere_thin_gemm():
        return

    from vllm.ampere_thin_gemm.thin_gemm import warmup

    bound = dispatch_threshold()
    ms = _token_sizes(capture_sizes, bound)
    if not ms:
        return

    # Only sizes <= the bound are warmed: a captured graph at M=40..64 takes the
    # cuBLAS branch, so the kernel is never in it and needs nothing preallocated.
    # If the bound is raised by env, this follows it automatically.
    nk = _discover_nk(worker)
    discovered = len(nk)
    nk.update(_TABLE_NK)
    device = worker.device

    warmup(sorted(nk), ms, device=device)
    logger.info(
        "Warmed up the sm_80 thin-M BF16 GEMM: %d shapes (%d discovered on the "
        "live model, %d from the shape-table backstop) x M in %s (bound %d).",
        len(nk), discovered, len(nk) - discovered, ms, bound,
    )
