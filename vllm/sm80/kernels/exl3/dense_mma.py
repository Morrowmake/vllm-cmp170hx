# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-core dense EXL3 launches of 1..32 rows (VLLM_EXL3_DENSE_MMA).

The kernels are part of the shipped library (csrc/exl3/dense_mma.cu) and run
inside linear_out_splitk, which hands them every launch they cover: launches
of 1..32 rows of the dense matrices of Qwen3.8-Flash-Next at TP2/PP1 or
TP1/PP2 (the library's served shapes, including both LM-head widths).
resolve() is called once per process by ops.ensure_extension, before any
launch or graph capture: with the switch on and its gate open (the split-K
operator active, sm_80 with 74 multiprocessors, and a supported parallel
layout) it checks the library, applies the diagnostic plan override,
prepares every kernel instantiation on the current device (shared-memory
attributes and occupancy), sets the
library's process switch and logs the banner.
"""

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# The multiprocessor count the plans were measured and validated at.
VALIDATED_SMS = 74
# (tensor, pipeline) parallel layouts covered by the matrix allowlist.
VALIDATED_LAYOUTS = ((2, 1), (1, 2))

# Library functions resolve() needs; a library without them is refused.
_OPERATORS = (
    "set_dense_mma",
    "set_dense_mma_shape_override",
    "dense_mma_prepare",
    "linear_mma_plan",
)

# The library's switch as resolve() set it.
active = False


def closed_reason(
    splitk: bool, capability: tuple[int, int] | None, sms: int | None, vllm_config
) -> str | None:
    """Why the tensor-core blocks must not serve this process, or None.

    `splitk` is the split-K operator's resolved gate (the blocks run inside
    linear_out_splitk, so they serve nothing while it is off)."""
    if not splitk:
        return (
            "the split-K operator is not active (VLLM_EXL3_DENSE_SPLITK off or its "
            "gate closed; the tensor-core blocks run inside linear_out_splitk)"
        )
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    if sms != VALIDATED_SMS:
        return f"{sms} multiprocessors, not {VALIDATED_SMS}"
    if vllm_config is None:
        return "no vLLM config is set, so the parallel layout is unknown"
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    if layout not in VALIDATED_LAYOUTS:
        return f"(tensor, pipeline) parallel size {layout}, not in {VALIDATED_LAYOUTS}"
    return None


def parse_shape_override(value: str) -> tuple[int, int, int] | None:
    """VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE as (mt, kw, splits), or None."""
    if not value:
        return None
    try:
        mt, kw, splits = (int(part) for part in value.split(","))
    except ValueError:
        mt = kw = splits = 0
    if mt not in (2, 4) or kw not in (1, 2, 4, 8) or splits not in (1, 2):
        raise ValueError(
            f"VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE={value!r}: expected MT,KW,SPLITS "
            "with MT 2 or 4, KW 1, 2, 4 or 8 and SPLITS 1 or 2"
        )
    return mt, kw, splits


def _library():
    """torch.ops._exl3_C, after checking that it has the functions above."""
    for name in _OPERATORS:
        try:
            torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
        except RuntimeError as exc:
            raise RuntimeError(
                "VLLM_EXL3_DENSE_MMA needs a vllm._exl3_C built with "
                f"csrc/exl3/dense_mma.cu: rebuild it ({exc})"
            ) from exc
    return torch.ops._exl3_C


def resolve() -> None:
    """VLLM_EXL3_DENSE_MMA for this process, with its log line."""
    global active
    if not envs.VLLM_EXL3_DENSE_MMA:
        if envs.VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE:
            logger.info_once(
                "VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE is set but VLLM_EXL3_DENSE_MMA "
                "is off; the override is ignored.",
                scope="process",
            )
        return
    override = parse_shape_override(envs.VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE)
    from vllm.config import get_current_vllm_config_or_none
    from vllm.sm80.kernels.exl3 import ops

    capability, sms = None, None
    if torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
        sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count
    reason = closed_reason(
        ops._resolve_dense_splitk(),
        capability,
        sms,
        get_current_vllm_config_or_none(),
    )
    if reason is not None:
        logger.info_once(
            "VLLM_EXL3_DENSE_MMA is set but the gate is closed: %s; every dense "
            "EXL3 launch keeps its current path.",
            reason,
            scope="process",
        )
        return
    library = _library()
    if override is not None:
        library.set_dense_mma_shape_override(*override)
    library.dense_mma_prepare()
    library.set_dense_mma(True)
    active = True
    logger.info_once(
        "EXL3 dense MMA active (VLLM_EXL3_DENSE_MMA): dense launches of 1..32 "
        "rows of the validated TP2/PP1 or TP1/PP2 matrices (bits_x2 10/12/14, "
        "codebook 2; the LM head included) run on tensor-core blocks inside "
        "linear_out_splitk; "
        "other launches keep the split-K operator's path.",
        scope="process",
    )
    if override is not None:
        logger.info_once(
            "EXL3 dense MMA plan forced by VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE: "
            "mt=%d kw=%d splits=%d on every tensor-core launch (diagnostic).",
            *override,
            scope="process",
        )
