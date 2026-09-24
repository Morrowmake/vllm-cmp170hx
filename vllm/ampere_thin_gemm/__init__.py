# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 thin-M BF16 GEMM for the layers GLM-5.3-Flash leaves unquantized.

The W4A16 checkpoint quantizes only the 288 routed experts per sparse layer.
Everything else -- every ``o_proj``, every KDA projection, every MLA/indexer
projection, the three dense MLPs, all 42 shared experts and routers, ``lm_head``
and the whole DFlash drafter -- is in ``quantization_config.ignore`` and runs as
a plain BF16 ``F.linear`` through cuBLASLt. At decode the batch is
``M = num_seqs * (1 + num_spec)``, i.e. 4 at concurrency 1 with k=3, and those
GEMMs are the largest kernel-side cost in the step: 5.06 ms of a 21.0 ms step in
the trace, against a 2.83 ms weight-streaming roofline.

``thin_gemm`` is tuned for these shapes. Measured per-shape, cuBLAS and the
candidate timed in the same process with CUDA-graph replay and an L2-defeating
weight rotation, it takes the target forward's BF16 GEMM time from
**4.310 to 3.727 ms/step at c1** (M=4) and **4.398 to 3.813 ms at c4** (M=16),
plus 0.074 ms/step across ``lm_head`` (twice per step under DFlash) and the six
DFlash drafter GEMMs. Every shape in the table beats cuBLAS at both M.

Semantics are exactly ``F.linear(x, w)`` with no bias: BF16 in, FP32 accumulate,
BF16 out. There is no precision reduction of any kind and no quantization; the
standalone gate holds the kernel to cuBLAS-equivalent error against an exact
FP32 recomputation and requires bitwise run-to-run determinism.

**Default OFF.** With ``VLLM_GLM5_THIN_GEMM`` unset nothing here is imported and
``dispatch_unquantized_gemm`` returns the upstream callable itself, so the
unflagged path is byte-identical rather than merely equivalent.

The M bound
-----------
``VLLM_GLM5_THIN_GEMM_MAX_TOKENS`` defaults to **32**, which is measured, not
assumed. Count-weighted step time over the whole 16-shape table:

===== ============ ========= ======== ==============
    M    candidate    cuBLAS    ratio  shapes losing
===== ============ ========= ======== ==============
   32      4.061 ms  4.699 ms   1.157x          0/16
   40      5.722 ms  4.573 ms   0.799x         10/16
   48      6.065 ms  4.635 ms   0.764x         11/16
   64      6.719 ms  4.817 ms   0.717x         11/16
===== ============ ========= ======== ==============

This build captures graphs at ``M in [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]``,
so 32 is a capture size and the bound lands exactly on one. Raising it one size
costs ~43 % of the step. It is not an artefact of the schedule table stopping at
32 either: re-fitting the configs at M=40 and M=64 over a ~250-point grid
recovers most of the gap but never crosses parity where it matters -- at M=64
``kda_in_proj_qkvbfg_a``, 43 % of all weight traffic, tops out at 0.884x. Past
M=32 cuBLAS switches to a compute-bound tile and a kernel built for the
weight-bandwidth-bound regime has nothing left to trade. Do not raise this
without re-running the token-bound measurements.

M=24 is a capture size inside the bound whose *performance* was never measured
(the bench sweeps M in 1/2/3/4/8/16/32). It is bracketed by wins at 16 and 32
and is correctness-tested, so it is allowed; it is the one M in range without
its own number.
"""

import functools

import torch

from vllm import envs
from vllm.platforms import current_platform

# Below this K the weight is too small to amortise the launch and cuBLAS wins
# outright; also keeps the split-K heuristic out of its degenerate corner.
_MIN_K = 128

__all__ = [
    "ampere_thin_gemm",
    "dispatch_threshold",
    "thin_gemm_supported",
    "thin_linear",
    "use_ampere_thin_gemm",
]


@functools.cache
def _device_supported() -> bool:
    """GA100-class sm_80 only. Cached: this is asked once per layer build."""
    return current_platform.is_cuda() and current_platform.is_device_capability(80)


def use_ampere_thin_gemm() -> bool:
    """True only when the operator opted in AND the part is sm_80."""
    if not envs.VLLM_GLM5_THIN_GEMM:
        return False
    return _device_supported()


def dispatch_threshold() -> int:
    """Largest M dispatched to the kernel; see the table in the module docstring."""
    return envs.VLLM_GLM5_THIN_GEMM_MAX_TOKENS


# (N, K) -> smallest M sent to cuBLAS instead. Measured under pipeline
# parallel, where kda_o_proj runs at full width (K = 64 heads * 128): past
# M = 16 no thin-M schedule beats cuBLAS on it (M=24 46.7 vs 46.0 us, M=32
# 49.3 vs 46.3 us, best of the sweep). No TP=4 shape is listed.
_CUBLAS_FROM_M: dict[tuple[int, int], int] = {(4096, 8192): 17}


def thin_gemm_supported(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Shape/dtype gate. Pure, allocation-free, no device query.

    Deliberately narrow: 2-D only, so nothing here can trigger the reshape copy
    a >2-D activation would need, which would be an allocation on the captured
    path. Everything else -- prefill (M=2048), the vision tower, any bias, any
    non-BF16 or K-strided weight -- falls through to cuBLAS by construction.
    """
    return (
        bias is None
        and x.ndim == 2
        and weight.ndim == 2
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        # The layout UnquantizedLinearMethod.create_weights produces: [N, K]
        # row-major, i.e. K-contiguous.
        and weight.stride(1) == 1
        and x.shape[1] == weight.shape[1]
        and x.shape[1] >= _MIN_K
        and x.shape[0] <= dispatch_threshold()
        and x.shape[0] < _CUBLAS_FROM_M.get(tuple(weight.shape), 1 << 30)
    )


def ampere_thin_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``default_unquantized_gemm`` with the sm_80 thin-M path spliced in.

    Signature matches ``default_unquantized_gemm`` exactly so it can be returned
    from ``dispatch_unquantized_gemm`` in its place.

    The branch is resolved while a CUDA graph is being *recorded*, not on replay:
    each captured graph has one fixed M, so whichever side was taken at capture
    is what the graph contains. A graph captured at size 4 and one captured at
    size 64 legitimately hold different kernels -- that is the bound doing its
    job, not a bug.
    """
    if thin_gemm_supported(x, weight, bias):
        from vllm.ampere_thin_gemm.thin_gemm import thin_gemm

        return thin_gemm(x, weight)
    return torch.nn.functional.linear(x, weight, bias)


def thin_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` for a bare ``nn.Parameter`` weight.

    For call sites that never build a ``LinearBase`` and so never reach
    ``dispatch_unquantized_gemm`` -- the indexer's
    ``index_kpool_compress_gate``, a ``[head_dim, hidden_size]`` parameter
    consumed by ``F.linear`` directly (N=128, K=4096, 11 calls/step, 1.27x).
    """
    if use_ampere_thin_gemm() and thin_gemm_supported(x, weight, None):
        from vllm.ampere_thin_gemm.thin_gemm import thin_gemm

        return thin_gemm(x, weight)
    return torch.nn.functional.linear(x, weight)
