# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pre-capture warmup of the sm_80 kernels, called once from
``vllm/model_executor/warmup/kernel_warmup.py`` (its one call into this
package), after the upstream model warmups and before ``capture_model()``.

Each family below either compiles its Triton/Gluon kernels before the first
request or allocates its allocate-once scratch before any CUDA-graph capture
(see ``warmup_utils``), and is a no-op unless its switch is set. The order is
the order the families ran in when they were called from kernel_warmup.py.
Model packages are not imported here: a model's warmup is looked up among
the loaded modules (its layers import it when their switch is set).
"""

import sys

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_sm80 import hot_swap

logger = init_logger(__name__)


def _log_prefill_switches() -> None:
    """The startup lines of the prefill switches whose gates are resolved per
    call: the mHC pre-norm GEMM and the sparse-MLA attention."""
    if envs.VLLM_SM80_MHC_PREFILL:
        logger.info_once(
            "sm_80 mHC prefill pre-norm GEMM on (VLLM_SM80_MHC_PREFILL=1): "
            "steps from the mhc/prefill_min_rows row on sm_80.")
    if envs.VLLM_SM80_SPARSE_MLA_PREFILL:
        logger.info_once(
            "sm_80 sparse-MLA prefill on (VLLM_SM80_SPARSE_MLA_PREFILL=1): "
            "steps and contexts from the attention_prefill/sparse_mla_min_rows "
            "and sparse_mla_min_ctx rows on sm_80.")


def warmup_from_worker(worker, cudagraph_capture_sizes) -> None:
    # The compiled Marlin decode (its own switch, on TP and PP workers alike):
    # its persistent scratch must exist before profiling/capture use the ops.
    if envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
        from vllm_sm80.kernels.marlin_moe.decode import (
            open_late_scratch,
            warmup_from_worker,
        )

        # kernel_warmup runs after the KV cache is sized: the whole-expert
        # decode scratch may be allocated from here on.
        open_late_scratch()
        warmup_from_worker(worker)

    # The sm_80 decode kernels, in this order: mHC (vllm_sm80/kernels/mhc/),
    # MoE routing (vllm_sm80/kernels/moe/), the GLM KDA decode step
    # (vllm_sm80/models/glm5next/ops/kda_decode.py) and the fused MoE router.
    # This MUST run before capture_model(): routing._scratch() and
    # kda_decode._counter() are allocate-once module-level buffers, and an
    # allocation made during a capture lands in that graph's private pool, so
    # a second capture would reuse freed memory. Triton also compiles on first
    # launch, which cannot happen inside a capture at all. No-op unless
    # VLLM_AMPERE_DECODE_KERNELS; the compiled Marlin decode above is warmed
    # independently of it.
    if envs.VLLM_AMPERE_DECODE_KERNELS:
        from vllm_sm80.kernels.mhc.warmup import (
            warmup_from_worker as warmup_mhc_decode,
        )
        from vllm_sm80.kernels.moe.warmup import (
            warmup_route_from_worker,
            warmup_routing_from_worker,
        )

        kda_decode = sys.modules.get("vllm_sm80.models.glm5next.ops.kda_decode")
        warmup_mhc_decode(worker, cudagraph_capture_sizes)
        warmup_routing_from_worker(worker, cudagraph_capture_sizes)
        if kda_decode is not None:
            kda_decode.warmup_from_worker(worker, cudagraph_capture_sizes)
        warmup_route_from_worker(worker, cudagraph_capture_sizes)
    # The sm_80 KDA chunked prefill at 64 or 16 heads
    # (vllm_sm80/models/glm5next/ops/kda_prefill.py): compile its kernels
    # before the first prefill. Prefill runs eagerly, so this is about
    # first-request latency, not capture.
    # Only with VLLM_SM80_KDA_PREFILL=1, and a no-op unless a KDA layer took
    # the path.
    kda_prefill = sys.modules.get("vllm_sm80.models.glm5next.ops.kda_prefill")
    if kda_prefill is not None and envs.VLLM_SM80_KDA_PREFILL:
        kda_prefill.warmup_from_worker(worker)
    # The sm_80 thin-M BF16 GEMM (vllm_sm80/kernels/thin_gemm/). This MUST run before
    # capture_model(): thin_gemm keeps its FP32 split-K partials and arrival
    # counters in an allocate-once module-level cache, and an allocation made
    # during a capture lands in that graph's private pool, so a second capture
    # would reuse freed memory. Triton also compiles on first launch, which
    # cannot happen inside a capture at all. No-op unless VLLM_THIN_GEMM.
    from vllm_sm80.kernels.thin_gemm.warmup import warmup_ampere_thin_gemm

    warmup_ampere_thin_gemm(worker, cudagraph_capture_sizes)
    _log_prefill_switches()
    if envs.VLLM_SM80_SPARSE_MLA_PREFILL or hot_swap.listed("sparse_mla_prefill"):
        # The 64-head sparse-MLA prefill kernel (sparse_prefill_mla_pp.py).
        # Prefill runs eagerly, but compiling here keeps the Gluon JIT off the
        # first request. No-op unless this rank runs 64 heads on sm_80.
        from vllm_sm80.kernels.attention.sparse_prefill_mla_pp import (
            warmup_from_worker as warmup_sparse_mla_prefill,
        )

        if warmup_sparse_mla_prefill(worker):
            logger.info("Compiled the PP sparse-MLA prefill kernel.")

    # The deterministic MoE block alignment (vllm_sm80/kernels/moe/align.py).
    # Same reason: its per-(device, E) scratch is allocate-once and Triton
    # compiles on first launch, neither of which may happen inside a capture.
    # No-op unless VLLM_MOE_DETERMINISTIC_ALIGN selects the kernel path.
    from vllm_sm80.kernels.moe.align import (
        warmup_from_worker as warmup_moe_align,
    )

    warmed_experts = warmup_moe_align(worker)
    if warmed_experts:
        logger.info(
            "Warmed up the deterministic MoE alignment kernels for %d experts.",
            warmed_experts,
        )

    # The split-block Marlin MoE prefill (vllm_sm80/kernels/marlin_moe/
    # prefill.py). Same reason: its block-list buffers are
    # allocate-once per (device, E) and its Triton kernels compile on first
    # launch. Only with VLLM_SM80_MARLIN_PREFILL.
    if envs.VLLM_SM80_MARLIN_PREFILL:
        from vllm_sm80.kernels.marlin_moe.prefill import (
            warmup_from_worker as warmup_pp_marlin_prefill,
        )

        if warmup_pp_marlin_prefill(worker):
            logger.info("Warmed up the split-block Marlin MoE prefill lists.")
