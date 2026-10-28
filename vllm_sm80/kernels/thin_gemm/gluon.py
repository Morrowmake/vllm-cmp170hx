# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gluon thin GEMM for the Qwen3.8 decode shapes (``VLLM_THIN_GEMM_GLUON``).

``thin_gemm_gluon`` carries a schedule measured on the 74-SM sm_80 part for
the seven unquantized (N, K) of Qwen3.8-Flash-Next at tensor-parallel size 2
(``SHAPES``), at every row count up to 32: split-K Gluon kernels with the
reduction in the last-arriving CTA, a K-sliced variant, a persistent
balanced-column kernel, the Triton thin GEMM for the remaining rows and a GEMV
for N = 1. Against cuBLAS it is ~1.41x on the served mix of these calls.

This module wraps the ``VLLM_THIN_GEMM`` dispatch (``ampere_thin_gemm``).
While the gate is open, only calls being captured into a CUDA graph reach the
Gluon kernels. A call made outside capture takes ``F.linear``, the call the
dispatch makes with both flags off, whatever its shape or row count (the
calls ``VLLM_THIN_GEMM`` alone would send to its Triton kernel included),
and that test comes first: launched eagerly (the drafter's decode passes
after the first, which run outside graphs at exactly the request count;
steps larger than the largest graph; the warm-up forward before each
capture), a Gluon call takes 1.7-3.2x the
time of cuBLAS's at 1-8 rows on the 74-SM part, its launch cost being host
time that a graph pays once, at capture. The test is a count of the graph
captures under way in the process (``_CAPTURE_DEPTH``), kept by wrapping
``torch.cuda.CUDAGraph.capture_begin`` and ``capture_end`` when the gate
opens (``count_captures``): every capture in this tree goes through those two
methods, FULL graphs through ``torch.cuda.graph`` (their forward runs with
runtime mode NONE inside it) and breakable PIECEWISE graphs through the
segments of ``BreakableCUDAGraphCapture``, so a call is counted as captured
on a side stream joined to the capture too, and the eager segments of a
breakable graph, which run between one segment's ``capture_end`` and the
next one's ``capture_begin`` and are replayed outside any capture, read 0.
An eager call pays one list read for it; asking the stream
(``cudaStreamIsCapturing``) cost ~0.3 us a call there. The count is
process-wide: a call made on another thread while one captures counts as
captured (vLLM issues a process's model GEMMs from one thread).

A captured call goes to ``thin_gemm_gluon.thin_gemm`` when its (N, K) is in
``SHAPES``, it has at most ``MAX_M`` rows and ``thin_gemm_supported`` admits
it, and its operands are in the layout the warmup compiled for
(``gluon_layout_ok``: unit K stride, row stride a multiple of 16 elements,
16-byte aligned base); every other captured call takes the wrapped function
unchanged, except a table call in another layout, which takes cuBLAS
(``F.linear``): the kernels specialize on those properties, so a different
layout would compile a new variant at serve time, and the cp.async loads of
the Gluon kernels do not compile for a row stride that is not a multiple of
16 elements. The row count decides, never the phase: prefill and mixed steps
with at most 32 rows run the same schedule as decode steps of that size.

The gate is decided once, before CUDA-graph capture, by ``warmup_table``
(called from ``warmup.py``): sm_80 with 74 SMs (the schedule is measured
there), compilation mode NONE (no torch.compile tracing through the Gluon
launches), CUDA graphs for decode and mixed steps (no ``enforce_eager``, no
graph mode that runs either eagerly at unpadded row counts) and the Triton
Gluon ampere API importable. Until then, and whenever it stays closed, the
wrapped dispatch serves every call; a closed gate logs one warning with the
reason. Once those hold, the warmup compiles the kernel variant and allocates
the workspace of every row count up to ``MAX_M``, whatever row counts the
server captures at: a captured call must find both ready, and no eager
forward runs the kernels before a capture (the warm-up forward takes
``F.linear``). A failure there fails the boot rather than leaving this
process on a different kernel from the other ranks.

``VLLM_THIN_GEMM_CONCAT`` pairs only the shipped thin GEMM's calls, so the
two flags do not combine: setting both raises.

Arithmetic: BF16 operands, FP32 accumulation, one BF16 rounding, bitwise
reproducible run to run; the summation order differs from cuBLAS's and from
``thin_gemm_v74``'s.
"""

import functools

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm_sm80.kernels.thin_gemm import impl as shipped_impl
from vllm_sm80.kernels.thin_gemm import thin_gemm_supported

logger = init_logger(__name__)

# (N, K) of the schedule table: hyper-connection down, final mixer down,
# hyper-connection up, MoE router, shared-expert gate, GDN in_proj_ba (TP2
# shard), PLE kv_proj.
SHAPES = frozenset({
    (336, 10240), (320, 10240), (10240, 320), (512, 2560), (1, 2560),
    (48, 2560), (12800, 2560),
})
# Largest row count of the table; more rows keep the wrapped dispatch.
MAX_M = 32
# The SM count the schedule was measured on.
SM_COUNT = 74
# Row counts with a schedule row (the forced-schedule diagnostic's values).
SERVED_M = (1, 2, 4, 8, 16, 24, 32)

# Graph captures begun and not yet ended in this process (count_captures), and
# the switch-off GEMM bound once: an eager call pays one list read for its
# test and no attribute lookups (it runs on every drafter decode pass).
_CAPTURE_DEPTH = [0]
_linear = torch.nn.functional.linear

# Set by warmup_table once every check passed and every variant compiled.
_LIVE = [False]


def count_captures(graph_cls=None) -> None:
    """Wrap ``graph_cls.capture_begin`` and ``capture_end`` (default
    ``torch.cuda.CUDAGraph``) once, so that ``_CAPTURE_DEPTH`` counts the
    captures begun and not yet ended. A ``capture_begin`` that raises counts
    nothing; ``capture_end`` counts down only a graph whose begin was counted,
    also when it raises (the capture is over or unusable either way)."""
    cls = torch.cuda.CUDAGraph if graph_cls is None else graph_cls
    if getattr(cls, "_thin_gemm_counted", False):
        return
    begin, end = cls.capture_begin, cls.capture_end

    @functools.wraps(begin)
    def capture_begin(self, *args, **kwargs):
        begin(self, *args, **kwargs)
        self._thin_gemm_capturing = True
        _CAPTURE_DEPTH[0] += 1

    @functools.wraps(end)
    def capture_end(self, *args, **kwargs):
        try:
            return end(self, *args, **kwargs)
        finally:
            if self.__dict__.pop("_thin_gemm_capturing", False):
                _CAPTURE_DEPTH[0] -= 1

    cls.capture_begin = capture_begin
    cls.capture_end = capture_end
    cls._thin_gemm_counted = True


def set_but_off(reason: str) -> None:
    logger.warning_once(
        "VLLM_THIN_GEMM_GLUON set but off: %s; the VLLM_THIN_GEMM path serves "
        "every call", reason)


def gluon_api_missing() -> str:
    """'' when the Triton Gluon ampere API the kernels use imports, else why."""
    try:
        from triton.experimental import gluon  # noqa: F401
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.language.nvidia.ampere import (  # noqa: F401
            async_copy,
            mma_v2,
        )
        for name in ("NVMMADistributedLayout", "DotOperandLayout",
                     "SwizzledSharedLayout", "allocate_shared_memory"):
            getattr(gl, name)
    except Exception as exc:  # noqa: BLE001 - any failure closes the gate
        return f"Triton Gluon ampere API missing ({type(exc).__name__}: {exc})"
    return ""


def force_m() -> int:
    """VLLM_THIN_GEMM_GLUON_FORCE_M if it names a served row count, else 0."""
    value = envs.VLLM_THIN_GEMM_GLUON_FORCE_M
    if value and value not in SERVED_M:
        logger.warning_once(
            "VLLM_THIN_GEMM_GLUON_FORCE_M=%d ignored: not one of %s",
            value, SERVED_M)
        return 0
    return value


@functools.cache
def impl():
    """``thin_gemm_gluon``, sharing the shipped module's workspace lane (a
    caller that selects a lane for a second stream selects it for both)."""
    from vllm_sm80.kernels.thin_gemm import thin_gemm_gluon

    thin_gemm_gluon._LANE = shipped_impl()._LANE
    thin_gemm_gluon._FORCE_M[0] = force_m()
    return thin_gemm_gluon


def closed_reason(device, vllm_config) -> str:
    """'' when this process may open the gate, else the reason it may not.

    The same on every rank of a launch (device model, config, venv)."""
    from vllm.config import CompilationMode, CUDAGraphMode

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms != SM_COUNT:
        return f"{sms} SMs, the schedule is measured at {SM_COUNT}"
    compilation = vllm_config.compilation_config
    if compilation.mode != CompilationMode.NONE:
        return "compilation mode is not NONE"
    if vllm_config.model_config.enforce_eager:
        return "enforce_eager runs every step eagerly (unpadded row counts)"
    graphs = compilation.cudagraph_mode
    if graphs is None or CUDAGraphMode.NONE in (graphs.decode_mode(),
                                                graphs.mixed_mode()):
        name = getattr(graphs, "name", graphs)
        return (f"cudagraph_mode {name} runs steps eagerly (unpadded row "
                "counts)")
    return gluon_api_missing()


def warmup_table(worker, ms) -> frozenset:
    """Decide the gate and, if it opens, compile and allocate every variant.

    -> the (N, K) this module now serves at M <= MAX_M (empty when closed).
    The Gluon module is warmed at every M from 1 to MAX_M, not only the ones
    ``ms`` holds: a captured call at any of them (a capture size the shape
    sees halved, a capture list this process was not told about) must not
    compile or allocate inside the capture. The shipped thin GEMM is warmed
    here for those shapes at any M above MAX_M that ``ms`` holds; the caller
    warms it for everything else. A warmup failure raises: one rank must not
    serve a different kernel from the others (the replicated router and
    hyper-connection GEMMs would differ between ranks).
    """
    _LIVE[0] = False
    reason = closed_reason(worker.device, worker.vllm_config)
    if reason:
        set_but_off(reason)
        return frozenset()
    table_ms = list(range(1, MAX_M + 1))
    try:
        impl().warmup(sorted(SHAPES), table_ms, device=worker.device)
    except Exception as exc:
        raise RuntimeError(
            "VLLM_THIN_GEMM_GLUON=1: the Gluon thin GEMM warmup failed "
            f"({type(exc).__name__}: {exc}); unset VLLM_THIN_GEMM_GLUON to "
            "serve without it") from exc
    over = [m for m in ms if m > MAX_M]
    if over:
        shipped_impl().warmup(sorted(SHAPES), over, device=worker.device)
    # Before any capture (the warmup runs ahead of them): the dispatch's
    # captured-call test reads the count.
    count_captures()
    _LIVE[0] = True
    pinned = impl()._FORCE_M[0]
    logger.info_once(
        "Gluon thin GEMM active [VLLM_THIN_GEMM_GLUON=1]: %d (N, K) x M 1-%d "
        "in captured calls, F.linear outside capture%s", len(SHAPES), MAX_M,
        f", schedule pinned to M={pinned} [VLLM_THIN_GEMM_GLUON_FORCE_M]"
        if pinned else "")
    return SHAPES


def gluon_layout_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``x`` and ``weight`` have the layout the warmup compiled for.

    Triton specializes a kernel on each pointer's 16-byte alignment, on an
    integer argument being 1 or a multiple of 16, so a contiguous operand of
    the table's K (all multiples of 16) and any view with a unit K stride, a
    row stride that is a multiple of 16 elements and a 16-byte aligned base
    run the warmed kernels. Host-side reads only, no allocation."""
    return (
        x.stride(1) == 1
        and x.stride(0) % 16 == 0
        and x.data_ptr() % 16 == 0
        and weight.stride(0) % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def wrap_dispatch(shipped):
    """``shipped`` (an ``ampere_thin_gemm``) with the table's captured calls
    sent to ``thin_gemm_gluon`` and every call outside capture to
    ``F.linear`` while the gate is open. Same signature."""
    if envs.VLLM_THIN_GEMM_CONCAT:
        raise ValueError(
            "VLLM_THIN_GEMM_GLUON and VLLM_THIN_GEMM_CONCAT are both set: the "
            "concat groups pair only the shipped thin GEMM's calls; set one "
            "of them")

    def ampere_thin_gemm(
        layer: torch.nn.Module,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _LIVE[0]:
            return shipped(layer, x, weight, bias)
        if not _CAPTURE_DEPTH[0]:
            return _linear(x, weight, bias)
        if (
            x.shape[0] <= MAX_M
            and tuple(weight.shape) in SHAPES
            and thin_gemm_supported(x, weight, bias)
        ):
            if gluon_layout_ok(x, weight):
                return impl().thin_gemm(x, weight)
            logger.warning_once(
                "Gluon thin GEMM: a (N, K) = %s call with x strides %s and "
                "base offset %d mod 16 bytes is not in the layout its kernels "
                "were compiled for; such calls take cuBLAS",
                tuple(weight.shape), tuple(x.stride()), x.data_ptr() % 16)
            return _linear(x, weight)
        return shipped(layer, x, weight, bias)

    ampere_thin_gemm.shipped = shipped
    return ampere_thin_gemm
