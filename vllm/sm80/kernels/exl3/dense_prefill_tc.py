# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-core dense EXL3 linear for prefill steps (VLLM_EXL3_DENSE_PREFILL_TC).

The operator ``linear_out_prefill`` (csrc/exl3/dense_prefill_tc.cu) computes
what ``linear_out`` computes, on the tensor cores: exact integer-grid main
products, FP32 sums in k order, and, for the last part-empty wave of tiles,
a k split whose partial tiles the last arriving block adds in k order
(arithmetic in csrc/exl3/dense_prefill_tc.h). Its results differ from
``linear_out``'s in the last bits, and a row's bits depend on the row count
of its launch and the row's position in it (which tiles the schedule
splits).

- the flag is set and ``resolve`` found the gate open: sm_80 with 74
  multiprocessors, a validated layout (tensor parallel size 2 or 4,
  pipeline parallel size 1, or tensor parallel size 1 with pipeline
  parallel size 2), and either no speculative decoding, an MTP /
  EAGLE drafter without parallel drafting (the drafters ``step_phase``
  covers), or a drafter whose decode steps carry at most MAX_SHIPPED_ROWS
  rows in every forward pass (``step_phase.max_decode_step_rows``), so the
  row floor below keeps every decode step on today's kernels; at the
  (1, 2) layout the BF16 launches take the cheaper two-product arithmetic
  (``set_prefill_arithmetic``, validated against that deployment);
- the running step is a prefill step by the scheduler (``step_phase``: some
  request has more than 1 + num_speculative_tokens tokens scheduled), never
  judged from a row count, and the stream is not capturing a CUDA graph;
- the matrix is one of VALIDATED_MATRICES (padded_in, padded_out, bits_x2;
  codebook 2), the call has more than MAX_SHIPPED_ROWS rows, and the launch
  has more than 64 rows, at least 16 output tiles of 128 x 128 and a 16-byte
  aligned trellis;
- with VLLM_EXL3_DENSE_SPLITK on, ``linear_out_splitk`` would not split the
  launch's K: every launch split-K splits keeps split-K.
Every other launch runs the operator ``_linear`` picked before (linear_out or
linear_out_splitk), unchanged. Row chunks, the workspace and its size are
``_linear``'s; the operator places its stream-K partial tiles and their
counters inside ``linear_workspace_size`` of the launch and zeroes the
counters on every call.
"""

import functools

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.sm80.kernels.exl3 import step_phase

logger = init_logger(__name__)

# Launches of at most this many rows keep today's kernels (the decode bound).
MAX_SHIPPED_ROWS = 64
# The multiprocessor count the schedule was validated at.
VALIDATED_SMS = 74
# The layouts the matrices were validated in: (tensor, pipeline) parallel size.
VALIDATED_PARALLEL = ((2, 1), (4, 1), (1, 2))
# Preserve the upstream shape allowlist at its existing TP2/TP4 layouts.
# PP2 gets its own full-width Qwen allowlist; it must not enable those
# additional shapes in the existing tensor-parallel configurations.
_UPSTREAM_MATRICES = frozenset(
    {
        # Qwen3.8-Flash-Next at TP2.
        (2560, 256, 14),  # attention k/v
        (2560, 640, 14),  # shared expert gate/up
        (2560, 640, 10),  # sparse-attention indexer q/k
        (2560, 1280, 12),  # drafter embedding / hidden-state projections
        (640, 2560, 14),  # shared expert down
        (3072, 2560, 14),  # output projections
        (2560, 3072, 14),  # linear-attention z
        (2560, 5120, 14),  # linear-attention qkv
        (2560, 6144, 14),  # attention q (with output gate)
        # DeepSeek-V4.1-Flash at TP4 (every rank), with its DSpark drafter.
        (5120, 512, 10),  # kv, compressor kv/gate, shared expert gate/up
        (5120, 512, 12),  # kv; drafter kv, shared expert gate/up
        (5120, 640, 10),  # shared expert gate/up
        (5120, 640, 12),  # drafter shared expert gate/up
        (512, 5120, 10),  # shared expert down
        (512, 5120, 12),  # drafter shared expert down
        (640, 5120, 10),  # shared expert down
        (640, 5120, 12),  # drafter shared expert down
        (5120, 1280, 10),  # q down-projection
        (5120, 1280, 12),  # q down-projection; drafter
        (1280, 8192, 10),  # q up-projection
        (1280, 8192, 12),  # drafter q up-projection
        (1280, 4096, 10),  # indexer q
        (4096, 1024, 10),  # output projection groups
        (4096, 1024, 12),  # drafter output projection groups
        (2048, 5120, 10),  # output projection
        (2048, 5120, 12),  # drafter output projection
        (6144, 6400, 10),  # Engram kv
        (15360, 5120, 12),  # drafter hidden-state projection
    }
)
VALIDATED_MATRICES = {
    (2, 1): _UPSTREAM_MATRICES,
    (4, 1): _UPSTREAM_MATRICES,
    (1, 2): frozenset(
        {
            (2560, 512, 14),  # attention k/v
            (2560, 640, 14),  # replicated shared expert gate/up
            (2560, 640, 10),  # replicated indexer q/k
            (2560, 2560, 12),  # MTP embedding / hidden-state projections
            (640, 2560, 14),  # replicated shared expert down
            (6144, 2560, 14),  # output projections
            (2560, 6144, 14),  # linear-attention z
            (2560, 10240, 14),  # linear-attention qkv
            (2560, 12288, 14),  # attention q with output gate
        }
    ),
}
VALIDATED_CODEBOOK = 2
# Speculative methods whose decode steps step_phase reads as decode steps.
COVERED_DRAFTERS = ("mtp", "eagle", "eagle3")
# VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE -> set_prefill_schedule mode.
SCHEDULES = {"planned": 0, "whole": 1}
# The library operators this path needs (csrc/exl3/dense_prefill_tc.cu).
OPERATORS = (
    "linear_out_prefill",
    "linear_prefill_applies",
    "set_prefill_schedule",
    "set_prefill_arithmetic",
)
# A warm-up launch per covered format: (rows, padded_in, padded_out). Its plan
# runs both kernels (whole tiles and the k split) at 74 multiprocessors.
WARMUP_LAUNCH = (482, 2560, 2560)

# Fixed by resolve(), before any launch or graph capture.
active = False
active_layout: tuple[int, int] | None = None
_served_matrices: frozenset[tuple[int, int, int]] = frozenset()
_max_decode_query_len = 0
_sms = 0
_fake_registered = False


def closed_reason(
    capability: tuple[int, int] | None, sms: int | None, vllm_config
) -> str | None:
    """Why linear_out_prefill must not serve this process, or None."""
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    if sms != VALIDATED_SMS:
        return f"{sms} multiprocessors, not {VALIDATED_SMS}"
    if vllm_config is None:
        return "no vLLM config is set, so the decode query length is unknown"
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    if layout not in VALIDATED_PARALLEL:
        return (
            f"(tensor, pipeline) parallel size {layout}, not one of "
            f"{', '.join(map(str, VALIDATED_PARALLEL))}"
        )
    speculative = vllm_config.speculative_config
    parallel_drafting = bool(getattr(speculative, "parallel_drafting", False))
    if (
        speculative is not None
        and (speculative.method not in COVERED_DRAFTERS or parallel_drafting)
        and (rows := step_phase.max_decode_step_rows(vllm_config)) > MAX_SHIPPED_ROWS
    ):
        return (
            f"speculative method {speculative.method!r} (parallel drafting "
            f"{parallel_drafting}): its decode steps can schedule more than 1 + "
            "num_speculative_tokens tokens per request, and a decode step's "
            f"forward pass can carry {rows} rows, more than the "
            f"{MAX_SHIPPED_ROWS} rows at or below which every call keeps "
            "today's kernels"
        )
    return None


def _warmup(sms: int) -> None:
    """Run linear_out_prefill once per covered format and output dtype (BF16,
    FP32) before any capture, so no request's first prefill loads the kernels
    or sets their attributes.
    The temporary buffers are freed again. ops.ensure_extension runs this
    before it marks the library loaded, so the workspace size comes from the
    library itself: ops.linear_workspace_size would call ensure_extension."""
    device = torch.device("cuda")
    rows, padded_in, padded_out = WARMUP_LAUNCH

    x = torch.ones((rows, padded_in), dtype=torch.bfloat16, device=device)
    outs = [
        torch.empty((rows, padded_out), dtype=dtype, device=device)
        for dtype in (torch.bfloat16, torch.float32)
    ]
    workspace = torch.empty(
        torch.ops._exl3_C.linear_workspace_size(rows, padded_in, padded_out),
        dtype=torch.float32,
        device=device,
    )
    suh = torch.ones(padded_in, dtype=torch.float16, device=device)
    svh = torch.ones(padded_out, dtype=torch.float16, device=device)
    for bits_x2 in sorted(
        {b for matrices in VALIDATED_MATRICES.values() for _, _, b in matrices}
    ):
        trellis = torch.zeros(
            (padded_in // 16, padded_out // 16, 8 * bits_x2),
            dtype=torch.int16,
            device=device,
        )
        for out in outs:
            linear_out_prefill(
                x,
                trellis,
                suh,
                svh,
                bits_x2,
                VALIDATED_CODEBOOK,
                padded_in,
                padded_out,
                out,
                workspace,
            )
    torch.cuda.synchronize(device)


def resolve() -> None:
    """VLLM_EXL3_DENSE_PREFILL_TC for this process, with its log line.

    Called once by ops.ensure_extension, after the library loaded and before
    ensure_extension returns: nothing here may call an ops function that calls
    ensure_extension."""
    global active, _max_decode_query_len, _sms, _fake_registered
    global active_layout, _served_matrices
    if not envs.VLLM_EXL3_DENSE_PREFILL_TC:
        return
    from vllm.config import get_current_vllm_config_or_none

    capability, sms = None, None
    if torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
        sms = torch.cuda.get_device_properties().multi_processor_count
    vllm_config = get_current_vllm_config_or_none()
    reason = closed_reason(capability, sms, vllm_config)
    if reason is not None:
        logger.info_once(
            "VLLM_EXL3_DENSE_PREFILL_TC is set but the gate is closed: %s; every "
            "dense EXL3 launch keeps today's kernels.",
            reason,
            scope="process",
        )
        return
    assert vllm_config is not None and sms is not None
    layout = (
        vllm_config.parallel_config.tensor_parallel_size,
        vllm_config.parallel_config.pipeline_parallel_size,
    )
    schedule = envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE
    for name in OPERATORS:
        try:
            torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
        except RuntimeError as exc:
            raise RuntimeError(
                f"VLLM_EXL3_DENSE_PREFILL_TC needs _exl3_C::{name}: rebuild the "
                "EXL3 extension from this tree (python "
                "csrc/exl3/build_standalone.py --out vllm), or unset the flag. "
                f"Detail: {exc}"
            ) from exc
    if not _fake_registered:
        from vllm.sm80.kernels.exl3.ops import _linear_fake

        torch.library.register_fake("_exl3_C::linear_out_prefill")(_linear_fake)
        _fake_registered = True
    set_prefill_schedule(SCHEDULES[schedule])
    set_prefill_arithmetic(int(layout == (1, 2)))
    _warmup(sms)
    _max_decode_query_len = step_phase.max_decode_query_len(vllm_config)
    _sms = sms
    logger.info_once(
        "EXL3 dense prefill tensor-core path active (VLLM_EXL3_DENSE_PREFILL_TC): "
        "in prefill steps (a request with more than %d tokens scheduled), dense "
        "launches of more than %d rows and at least 16 output tiles of the %d "
        "validated matrices (bits_x2 10/12/14, codebook 2) at TP%d/PP%d take "
        "linear_out_prefill, schedule %s; split-K launches, other matrices, "
        "decode steps and graph captures keep today's kernels.",
        _max_decode_query_len,
        MAX_SHIPPED_ROWS,
        len(VALIDATED_MATRICES[layout]),
        layout[0],
        layout[1],
        schedule,
        scope="process",
    )
    active_layout = layout
    _served_matrices = VALIDATED_MATRICES[layout]
    active = True


def linear_prefill_applies(
    rows: int, padded_in: int, padded_out: int, bits_x2: int, codebook: int
) -> bool:
    """Whether linear_out_prefill covers this launch shape (host arithmetic)."""
    return torch.ops._exl3_C.linear_prefill_applies(
        rows, padded_in, padded_out, bits_x2, codebook
    )


def set_prefill_schedule(mode: int) -> None:
    """The operator's schedule for this process: 0 planned, 1 whole tiles."""
    torch.ops._exl3_C.set_prefill_schedule(mode)


def set_prefill_arithmetic(mode: int) -> None:
    """The arithmetic of this process's BF16 prefill GEMMs: 0 exact dominant
    term (three products), 1 the two-product combine of the TP1/PP2 full-width
    matrices."""
    torch.ops._exl3_C.set_prefill_arithmetic(mode)


@functools.cache
def takes_launch(
    rows: int,
    padded_in: int,
    padded_out: int,
    bits_x2: int,
    codebook: int,
    splitk: bool,
    sms: int,
    matrices: frozenset[tuple[int, int, int]],
) -> bool:
    """Whether a prefill-step launch of this shape goes to linear_out_prefill:
    a validated matrix the operator covers at this row count
    and, with split-K on, split-K does not split it."""
    from vllm.sm80.kernels.exl3.ops import linear_splitk_splits

    if (
        codebook != VALIDATED_CODEBOOK
        or (padded_in, padded_out, bits_x2) not in matrices
    ):
        return False
    if splitk and linear_splitk_splits(rows, padded_out, sms, bits_x2, codebook) == 2:
        return False
    return linear_prefill_applies(rows, padded_in, padded_out, bits_x2, codebook)


def linear_out_prefill(
    x,
    trellis,
    suh,
    svh,
    bits_x2: int,
    codebook: int,
    logical_in: int,
    logical_out: int,
    out,
    workspace,
) -> None:
    """linear_out's contract for the launches linear_prefill_applies covers."""
    torch.ops._exl3_C.linear_out_prefill(
        x,
        trellis,
        suh,
        svh,
        bits_x2,
        codebook,
        logical_in,
        logical_out,
        out,
        workspace,
    )


def _launch(
    default,
    splitk: bool,
    x,
    trellis,
    suh,
    svh,
    bits_x2: int,
    codebook: int,
    logical_in: int,
    logical_out: int,
    out,
    workspace,
) -> None:
    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    operator = default
    if trellis.data_ptr() % 16 == 0 and takes_launch(
        x.shape[0],
        padded_in,
        padded_out,
        bits_x2,
        codebook,
        splitk,
        _sms,
        _served_matrices,
    ):
        operator = linear_out_prefill
    operator(
        x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace
    )


def route(default, rows: int, splitk: bool):
    """The operator _linear calls for each row chunk of a call of `rows` rows:
    `default` itself outside prefill steps, during graph capture and for calls
    of at most MAX_SHIPPED_ROWS rows; otherwise a per-launch choice between
    linear_out_prefill and `default`."""
    if (
        rows <= MAX_SHIPPED_ROWS
        or torch.cuda.is_current_stream_capturing()
        or not step_phase.prefill_step(_max_decode_query_len)
    ):
        return default
    return functools.partial(_launch, default, splitk)
