# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings for the shipped BF16 EXL3 extension."""

import importlib
import weakref
from functools import cache

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.sm80 import capture
from vllm.sm80.kernels.exl3 import capture_points, dense_prefill_tc
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_loaded = False
# The base fake implementations, registered once per process: a retry after a
# resolve that raised must not register them again (torch refuses a second).
_fakes_registered = False
# VLLM_EXL3_DENSE_SPLITK with its gate open: _linear calls linear_out_splitk.
# Fixed by ensure_extension, before any launch or graph capture.
_dense_splitk = False

# Fixed FP32 scratch per stream role: 32 MiB normally, 128 MiB when the
# validated PP2 prefill path is active, to amortize its full-width row chunks.
LINEAR_SCRATCH_ELEMENTS = 8 << 20

# VLLM_EXL3_MOE_PREFILL_UNCHUNKED (default off): moe_out_prefill_v2's calls of
# the whole-expert configuration (hidden 5120, intermediate 2304, top-6, 96 of
# 384 experts local, bits_x2 6 or 4, 65..2048 rows) run
# _exl3_C::moe_out_prefill_try instead (csrc/exl3/moe_prefill_try.cu: the same
# arithmetic, bitwise equal, in one launch sequence per call), with its
# workspace size. Every other call is unchanged.
TRY_MOE_PREFILL = envs.VLLM_EXL3_MOE_PREFILL_UNCHUNKED
TRY_MOE_PREFILL_ROWS = (65, 2048)

# moe_out_v2 (csrc/exl3/moe_v2.cu) covers calls of 1..MOE_V2_MAX_ROWS rows of
# these configurations: (hidden, intermediate, top-k, local experts, global
# experts) -> the bits_x2 values its kernels are built for. Every routed matrix
# of a layer is stored with one of them and codebook 2 (moe_v2_expert_meta).
MOE_V2_MAX_ROWS = 64
MOE_V2_CONFIGS = {
    # Half the experts local (tensor parallel size 2 with expert parallelism).
    (2560, 640, 10, 256, 512): (10,),
    # Whole experts local (tensor parallel size 1: pipeline stages split
    # layers, not experts; validated with the same checkpoint at PP2, see
    # quant._prefill_layout_closed_reason for the layout cross-check).
    (2560, 640, 10, 512, 512): (10,),
    # Whole experts, a quarter of them local (expert parallel size 4).
    (5120, 2304, 6, 96, 384): (6, 4),
    # Tensor-parallel routed experts: every expert local, each rank a 640- or
    # 512-column chunk of the 2304-column intermediate.
    (5120, 640, 6, 384, 384): (6, 4),
    (5120, 512, 6, 384, 384): (6, 4),
    # The DSpark drafter's experts: top-3 of 128, a quarter of them local
    # (expert parallel size 4); kernels and stages of their own (moe_v2.cu
    # DraftTiling: 16-row tiles, the hi/lo split in the MMA's C operand).
    (5120, 2304, 3, 32, 128): (8,),
}
# The MOE_V2_CONFIGS entries only VLLM_EXL3_MOE_V2_DRAFT opens (the DSpark
# drafter's layers); VLLM_EXL3_MOE_V2 opens the others. The drafter's calls
# are never folded (moe_out_v2_fold refuses bits_x2 8). The drafter's experts
# split across tensor-parallel ranks (intermediate 640 or 512, all 128 local)
# keep moe_out: moe_out_v2 has bits_x2 8 kernels for those shapes, but their
# output has not been checked through the drafter's consumer.
MOE_V2_DRAFT_CONFIGS = frozenset({(5120, 2304, 3, 32, 128)})
# moe_out_prefill (csrc/exl3/moe_prefill.cu) covers the hidden-2560
# configurations (its served_shape accepts either local-expert count); it
# serves calls of MOE_PREFILL_MIN_ROWS to MOE_PREFILL_MAX_ROWS rows (the
# validated range: one 3460-token prefill chunk).
MOE_PREFILL_CONFIGS = {
    (2560, 640, 10, 256, 512): (10,),
    (2560, 640, 10, 512, 512): (10,),
}
MOE_PREFILL_MIN_ROWS = MOE_V2_MAX_ROWS + 1
MOE_PREFILL_MAX_ROWS = 3460
# moe_out_prefill_v2 (csrc/exl3/moe_prefill_v2.cu) covers the hidden-5120
# configuration: the DeepSeek-V4.1-Flash routed experts at expert parallel
# size 4, bits_x2 6 or 4. It serves calls of MOE_PREFILL_MIN_ROWS to
# MOE_PREFILL_V2_MAX_ROWS rows (the validated range: one 2048-token prefill
# chunk).
MOE_PREFILL_V2_CONFIGS = {
    (5120, 2304, 6, 96, 384): (6, 4),
    # Tensor-parallel routed experts: every expert local, each rank a 640- or
    # 512-column chunk of the 2304-column intermediate.
    (5120, 640, 6, 384, 384): (6, 4),
    (5120, 512, 6, 384, 384): (6, 4),
}
MOE_PREFILL_V2_MAX_ROWS = 2048


# Output dtypes of the dense linear: BF16 rounds the FP32 result once; FP32
# stores it as the kernels computed it.
LINEAR_OUT_DTYPES = (torch.bfloat16, torch.float32)


def _linear_fake(
    x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace
):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(out.dtype in LINEAR_OUT_DTYPES)
    torch._check(out.shape[0] == x.shape[0])
    torch._check(out.shape[1] == logical_out)
    torch._check(x.shape[1] == logical_in)
    torch._check(workspace.dtype == torch.float32)


def _moe_fake(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size,
    out,
    workspace,
    swiglu_limit=0.0,
):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(out.dtype == torch.bfloat16)
    torch._check(out.shape == x.shape)
    torch._check(topk_ids.shape == topk_weights.shape)
    torch._check(workspace.dtype == torch.float32)


def _moe_v2_fake(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size,
    bits_x2,
    out,
    workspace,
    swiglu_limit=0.0,
):
    _moe_fake(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        out,
        workspace,
        swiglu_limit,
    )


def _ple_fake(
    packed_rows,
    global_ids,
    head_bias,
    head_offsets,
    head_vocab_sizes,
    owner_lo,
    owner_hi,
    bits,
    out,
):
    torch._check(out.dtype == torch.bfloat16)
    torch._check(out.shape[0] == packed_rows.shape[0])
    torch._check(out.shape[1] == head_bias.shape[1])
    torch._check(global_ids.numel() == packed_rows.shape[0])


# 3: moe_out_v2 takes bits_x2. 2: the MoE operators take swiglu_limit (a clamp
# before SiLU(gate) * up, 0 for none).
_ABI_VERSION = 3
# Operators this tree calls; a library without one of them is refused.
_OPERATORS = (
    "linear_out",
    "linear_out_splitk",
    "moe_out",
    "moe_out_v2",
    "moe_out_prefill",
    "ple_decode_out",
    "ple_read_rows",
    "linear_workspace_size",
    "linear_splitk_splits",
    "moe_workspace_size",
    "moe_workspace_size_v2",
    "moe_workspace_size_prefill",
)
# Classes this tree constructs; a library without one of them is refused.
_CLASSES = ("PleHostReader", "EngramHostFetch")


def _runtime_build_info() -> dict:
    cuda = torch.version.cuda
    major, minor = (int(v) for v in cuda.split(".")[:2]) if cuda else (0, 0)
    return {
        "abi_version": _ABI_VERSION,
        "torch_version": torch.__version__.split("+")[0],
        "cuda_version": major * 1000 + minor * 10,
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def _validate_extension(module) -> None:
    """module.cpp registers through TORCH_LIBRARY, which is bound to the
    PyTorch build: refuse a library built against another one."""
    built = module.build_info()
    for key, expected in _runtime_build_info().items():
        actual = built.get(key)
        # CUDA minor toolkits within one major share the runtime ABI.
        if key == "cuda_version" and isinstance(actual, int):
            actual, expected = actual // 1000, expected // 1000
        if actual != expected:
            raise ValueError(
                f"{key} mismatch: extension {built.get(key)!r}, runtime {expected!r}"
            )
    for name in _OPERATORS:
        torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
    for name in _CLASSES:
        getattr(torch.classes._exl3_C, name)


def dense_splitk_closed_reason(capability: tuple[int, int] | None) -> str | None:
    """Why linear_out_splitk must not serve this process, or None when it may.

    The operator itself keeps linear_out for every launch outside the
    configuration its kernels cover (formats, block shapes, grid sizes).
    """
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    return None


def _resolve_dense_splitk() -> bool:
    """VLLM_EXL3_DENSE_SPLITK for this process, with its log line."""
    if not envs.VLLM_EXL3_DENSE_SPLITK:
        return False
    capability = (
        tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    )
    reason = dense_splitk_closed_reason(capability)
    if reason is not None:
        logger.info_once(
            "VLLM_EXL3_DENSE_SPLITK is set but the gate is closed: %s; every "
            "dense EXL3 call keeps linear_out.",
            reason,
            scope="process",
        )
        return False
    logger.info_once(
        "EXL3 dense split-K active (VLLM_EXL3_DENSE_SPLITK): dense launches of "
        "at most 1024 tiny or small blocks (bits_x2 10/12/14, codebook 2) split "
        "K in two on warp-specialized blocks; other launches keep linear_out.",
        scope="process",
    )
    return True


def ensure_extension() -> None:
    global _loaded, _dense_splitk, _fakes_registered
    if _loaded:
        return
    try:
        module = importlib.import_module("vllm._exl3_C")
        _validate_extension(module)
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "EXL3 requires the compiled vllm._exl3_C extension, built for the "
            "running PyTorch: build it with python csrc/exl3/build_standalone.py "
            f"--out vllm (in the vLLM source tree). Detail: {exc}"
        ) from exc
    if not _fakes_registered:
        for name, fake in (
            ("linear_out", _linear_fake),
            ("linear_out_splitk", _linear_fake),
            ("moe_out", _moe_fake),
            ("moe_out_v2", _moe_v2_fake),
            ("moe_out_prefill", _moe_fake),
            ("ple_decode_out", _ple_fake),
        ):
            torch.library.register_fake(f"_exl3_C::{name}")(fake)
        _fakes_registered = True
    from vllm.sm80.kernels.exl3 import dense_mma

    dense_mma.resolve()
    _dense_splitk = _resolve_dense_splitk()
    dense_prefill_tc.resolve()
    _loaded = True


@cache
def linear_workspace_size(max_rows: int, padded_in: int, padded_out: int) -> int:
    ensure_extension()
    return torch.ops._exl3_C.linear_workspace_size(max_rows, padded_in, padded_out)


def linear_scratch_elements() -> int:
    wide_pp2 = dense_prefill_tc.active and dense_prefill_tc.active_layout == (1, 2)
    return LINEAR_SCRATCH_ELEMENTS * (4 if wide_pp2 else 1)


@cache
def _linear_row_chunk(padded_in: int, padded_out: int, budget: int) -> int:
    """Return the most rows whose native workspace fits this fixed budget."""
    rows, limit = 0, budget
    while rows < limit:
        middle = (rows + limit + 1) // 2
        if linear_workspace_size(middle, padded_in, padded_out) <= budget:
            rows = middle
        else:
            limit = middle - 1
    if rows == 0:
        raise ValueError("One EXL3 matrix row exceeds the fixed linear scratch")
    return rows


def linear_row_chunk(padded_in: int, padded_out: int) -> int:
    ensure_extension()
    return _linear_row_chunk(padded_in, padded_out, linear_scratch_elements())


@cache
def moe_workspace_size(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


@cache
def moe_workspace_size_v2(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size_v2(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


def moe_v2_expert_meta(hidden_size: int, intermediate_size: int, bits_x2: int):
    """[padded_in, padded_out, bits_x2, codebook] of gate, up and down of
    every local expert of a layer the MoE tensor-core operators take."""
    return [
        [hidden_size, intermediate_size, bits_x2, 2],
        [hidden_size, intermediate_size, bits_x2, 2],
        [intermediate_size, hidden_size, bits_x2, 2],
    ]


def _moe_closed_reason(
    configs: dict[tuple[int, ...], tuple[int, ...]],
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
) -> str | None:
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    if dtype != torch.bfloat16:
        return f"activation dtype {dtype}, not bfloat16"
    shape = (hidden_size, intermediate_size, top_k, local_experts, global_experts)
    rates = configs.get(shape)
    if rates is None:
        return (
            f"(hidden, intermediate, top_k, local experts, global experts) = "
            f"{shape}, not one of {sorted(configs)}"
        )
    if len(metadata) != local_experts:
        return f"{len(metadata)} expert entries for {local_experts} local experts"
    bits_x2 = metadata[0][0][2]
    if bits_x2 not in rates:
        return f"local expert 0 stores bits_x2 {bits_x2}, not one of {rates}"
    expected = moe_v2_expert_meta(hidden_size, intermediate_size, bits_x2)
    for local, meta in enumerate(metadata):
        if [list(m) for m in meta] != expected:
            return (
                f"local expert {local} stores [padded_in, padded_out, bits_x2, "
                f"codebook] {[list(m) for m in meta]}, not {expected}"
            )
    return None


def moe_v2_closed_reason(
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
    draft: bool = False,
) -> str | None:
    """Why moe_out_v2 must not serve this layer, or None when it may (then
    every routed matrix has bits_x2 ``metadata[0][0][2]``).

    Checked once per layer from host-side values (``metadata`` is the
    per-expert [padded_in, padded_out, bits_x2, codebook] list the layer's
    ``exl3_meta`` buffer is built from), never from device tensors.
    ``draft``: the configurations of MOE_V2_DRAFT_CONFIGS
    (VLLM_EXL3_MOE_V2_DRAFT), else the others (VLLM_EXL3_MOE_V2).
    """
    return _moe_closed_reason(
        {
            shape: rates
            for shape, rates in MOE_V2_CONFIGS.items()
            if (shape in MOE_V2_DRAFT_CONFIGS) == draft
        },
        capability,
        dtype,
        hidden_size,
        intermediate_size,
        top_k,
        local_experts,
        global_experts,
        metadata,
    )


def moe_prefill_closed_reason(
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
) -> str | None:
    """moe_v2_closed_reason for moe_out_prefill (MOE_PREFILL_CONFIGS)."""
    return _moe_closed_reason(
        MOE_PREFILL_CONFIGS,
        capability,
        dtype,
        hidden_size,
        intermediate_size,
        top_k,
        local_experts,
        global_experts,
        metadata,
    )


def moe_prefill_v2_closed_reason(
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
) -> str | None:
    """moe_v2_closed_reason for moe_out_prefill_v2 (MOE_PREFILL_V2_CONFIGS)."""
    return _moe_closed_reason(
        MOE_PREFILL_V2_CONFIGS,
        capability,
        dtype,
        hidden_size,
        intermediate_size,
        top_k,
        local_experts,
        global_experts,
        metadata,
    )


def linear_out(
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
    ensure_extension()
    torch.ops._exl3_C.linear_out(
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


def linear_out_splitk(
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
    """linear_out's contract (arguments, checks, workspace); see
    csrc/exl3/dense_splitk.h for the launches it changes."""
    ensure_extension()
    torch.ops._exl3_C.linear_out_splitk(
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


def linear_splitk_splits(
    rows: int, padded_out: int, sms: int, bits_x2: int, codebook: int
) -> int:
    """K split factor linear_out_splitk uses for this launch: 2 on the
    split-K path, 1 where it runs linear_out."""
    ensure_extension()
    return torch.ops._exl3_C.linear_splitk_splits(
        rows, padded_out, sms, bits_x2, codebook
    )


def moe_out(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
) -> None:
    """Routed experts; ``swiglu_limit`` > 0 clamps the FP32 gate from above
    and up on both sides before SiLU(gate) * up (csrc/exl3/moe.cu)."""
    ensure_extension()
    torch.ops._exl3_C.moe_out(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        out,
        workspace,
        swiglu_limit,
    )


def moe_out_v2(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
    *,
    bits_x2: int,
) -> None:
    """moe_out's contract on the tensor cores; see csrc/exl3/moe_v2.cu.
    ``bits_x2`` is the one bit width of every routed matrix of the layer."""
    ensure_extension()
    torch.ops._exl3_C.moe_out_v2(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
        swiglu_limit,
    )


_moe_v2_fold_fake_registered = False


def require_moe_v2_fold() -> None:
    """Refuse a library without moe_out_v2_fold (built from a tree before
    it) and register the operator's fake implementation once per process."""
    global _moe_v2_fold_fake_registered
    ensure_extension()
    try:
        torch._C._dispatch_find_schema_or_throw("_exl3_C::moe_out_v2_fold", "")
    except RuntimeError as exc:
        raise RuntimeError(
            "VLLM_EXL3_MOE_V2_FOLD needs _exl3_C::moe_out_v2_fold: rebuild the "
            "EXL3 extension from this tree (python csrc/exl3/build_standalone.py "
            f"--out vllm), or unset the flag. Detail: {exc}"
        ) from exc
    if not _moe_v2_fold_fake_registered:
        torch.library.register_fake("_exl3_C::moe_out_v2_fold")(_moe_v2_fake)
        _moe_v2_fold_fake_registered = True


def moe_out_v2_fold(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
    *,
    bits_x2: int,
) -> None:
    """moe_out_v2's arguments and output bits in three launches: the
    activation and the combine run inside the gate/up and down lane launches
    (csrc/exl3/moe_v2.cu FoldArgs). Callers check require_moe_v2_fold first."""
    ensure_extension()
    torch.ops._exl3_C.moe_out_v2_fold(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
        swiglu_limit,
    )


# VLLM_EXL3_MOE_V2_RING=1 (default off): the target model's
# expert-parallel moe_out_v2 calls of exactly this configuration (hidden,
# intermediate, top-k, local experts, global experts), bits_x2 6 or 4, 1..64
# rows, take _exl3_C::moe_out_v2_try (csrc/exl3/moe_v2_try.cu) instead of
# moe_out_v2 or moe_out_v2_fold: same arguments and output bits.
MOE_V2_TRY_CONFIG = (5120, 2304, 6, 96, 384)
MOE_V2_TRY_BITS = (6, 4)
_moe_v2_try_fake_registered = False


def moe_v2_try_open(
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    bits_x2: int,
) -> bool:
    """VLLM_EXL3_MOE_V2_RING is set and the layer is the configuration it covers."""
    return (
        envs.VLLM_EXL3_MOE_V2_RING
        and (hidden_size, intermediate_size, top_k, local_experts, global_experts)
        == MOE_V2_TRY_CONFIG
        and bits_x2 in MOE_V2_TRY_BITS
    )


def require_moe_v2_try() -> None:
    """Refuse a library without moe_out_v2_try and register the operator's
    fake implementation once per process."""
    global _moe_v2_try_fake_registered
    ensure_extension()
    try:
        torch._C._dispatch_find_schema_or_throw("_exl3_C::moe_out_v2_try", "")
    except RuntimeError as exc:
        raise RuntimeError(
            "VLLM_EXL3_MOE_V2_RING needs _exl3_C::moe_out_v2_try: rebuild the EXL3 "
            "extension from this tree (python csrc/exl3/build_standalone.py "
            f"--out vllm), or unset the flag. Detail: {exc}"
        ) from exc
    if not _moe_v2_try_fake_registered:
        torch.library.register_fake("_exl3_C::moe_out_v2_try")(_moe_v2_fake)
        _moe_v2_try_fake_registered = True


@cache
def moe_workspace_size_v2_try(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size_v2_try(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


def moe_out_v2_try(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
    *,
    bits_x2: int,
) -> None:
    """moe_out_v2's arguments and output bits (VLLM_EXL3_MOE_V2_RING; callers check
    moe_v2_try_open and require_moe_v2_try first)."""
    ensure_extension()
    torch.ops._exl3_C.moe_out_v2_try(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
        swiglu_limit,
    )


def require_moe_v2_draft() -> None:
    """Refuse a library without the drafter's bits_x2 8 moe_out_v2 kernels
    (built from a tree before them; its moe_v2_draft_force_rows marks them)."""
    ensure_extension()
    try:
        torch._C._dispatch_find_schema_or_throw("_exl3_C::moe_v2_draft_force_rows", "")
    except RuntimeError as exc:
        raise RuntimeError(
            "VLLM_EXL3_MOE_V2_DRAFT needs the bits_x2 8 kernels of "
            "_exl3_C::moe_out_v2: rebuild the EXL3 extension from this tree "
            "(python csrc/exl3/build_standalone.py --out vllm), or unset the "
            f"flag. Detail: {exc}"
        ) from exc


def moe_v2_draft_force_rows(rows: int) -> None:
    """VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS for this process: 0 (each drafter
    row tile takes the lane kernel of its own rows) or 8 or 16 (every tile at
    least that tile's). Set before the warm-up and any graph capture."""
    if rows not in (0, 8, 16):
        raise ValueError(
            f"VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS={rows}: not one of 0, 8, 16"
        )
    require_moe_v2_draft()
    torch.ops._exl3_C.moe_v2_draft_force_rows(rows)


# (device, hidden, intermediate, bits_x2, fold, trial) warmed by warmup_moe_v2.
_moe_v2_warm: set[tuple] = set()


def warmup_moe_v2(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    global_expert_ids: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    bits_x2: int,
    swiglu_limit: float = 0.0,
    fold: bool = False,
    trial: bool = False,
) -> None:
    """Run moe_out_v2 (fold: moe_out_v2_fold; trial: moe_out_v2_try) per
    device, shape and bits_x2 before any CUDA graph capture.

    Each lane kernel instance's first call queries the SM count and the
    kernel's occupancy and keeps them (moe_v2.cu launch_lane). The operator
    picks short or long K slices from rows x top_k (160 routed assignments at
    hidden 2560; hidden 5120 has one layout), so one call of one row and one
    of MOE_V2_MAX_ROWS rows, all with local routes, launch every kernel it can
    pick for the layer's shape and bits_x2, and no capture is a first call.
    The temporary buffers are freed again.
    """
    device = packed_ptrs.device
    key = (device, hidden_size, intermediate_size, bits_x2, fold, trial)
    if key in _moe_v2_warm:
        return
    operator = moe_out_v2_try if trial else moe_out_v2_fold if fold else moe_out_v2
    rows = MOE_V2_MAX_ROWS
    ids = torch.tensor(
        [global_expert_ids[:top_k]] * rows, dtype=torch.int32, device=device
    )
    weights = torch.full((rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    x = torch.ones((rows, hidden_size), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    sizes = [
        moe_workspace_size_v2(
            rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
        )
    ]
    if trial:
        sizes.append(
            moe_workspace_size_v2_try(
                rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
            )
        )
    workspace = torch.empty(max(sizes), dtype=torch.float32, device=device)
    for count in (1, rows):
        operator(
            x[:count],
            ids[:count],
            weights[:count],
            packed_ptrs,
            packed_meta,
            expert_map,
            intermediate_size,
            out[:count],
            workspace,
            swiglu_limit=swiglu_limit,
            bits_x2=bits_x2,
        )
    torch.cuda.synchronize(device)
    _moe_v2_warm.add(key)


@cache
def moe_workspace_size_prefill(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size_prefill(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


def moe_out_prefill(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
) -> None:
    """moe_out's contract for calls of at least MOE_PREFILL_MIN_ROWS rows of a
    served configuration (moe_prefill_closed_reason); see
    csrc/exl3/moe_prefill.cu."""
    ensure_extension()
    torch.ops._exl3_C.moe_out_prefill(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        out,
        workspace,
        swiglu_limit,
    )


_moe_prefill_warm_devices: set[torch.device] = set()


def warmup_moe_prefill(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    global_expert_ids: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    swiglu_limit: float = 0.0,
) -> None:
    """Run moe_out_prefill once per device before any CUDA graph capture.

    The operator sets its GEMM kernels' shared-memory limit on its first call
    and launches the same kernels at every row count (the row tiles of an
    expert are chosen on the device), so one call of MOE_PREFILL_MIN_ROWS
    rows, all with local routes, loads every kernel it can launch. The
    temporary buffers are freed again.
    """
    device = packed_ptrs.device
    if device in _moe_prefill_warm_devices:
        return
    rows = MOE_PREFILL_MIN_ROWS
    ids = torch.tensor(
        [global_expert_ids[:top_k]] * rows, dtype=torch.int32, device=device
    )
    weights = torch.full((rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    x = torch.ones((rows, hidden_size), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    workspace = torch.empty(
        moe_workspace_size_prefill(
            rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
        ),
        dtype=torch.float32,
        device=device,
    )
    moe_out_prefill(
        x,
        ids,
        weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        out,
        workspace,
        swiglu_limit=swiglu_limit,
    )
    torch.accelerator.synchronize(device)
    _moe_prefill_warm_devices.add(device)


_moe_prefill_v2_fake_registered = False


def require_moe_prefill_v2() -> None:
    """Refuse a library without moe_out_prefill_v2 (built from a tree before
    it) and register the operator's fake implementation once per process."""
    global _moe_prefill_v2_fake_registered
    ensure_extension()
    for name in ("moe_out_prefill_v2", "moe_workspace_size_prefill_v2"):
        try:
            torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
        except RuntimeError as exc:
            raise RuntimeError(
                f"VLLM_DEEPSEEK_V41_MOE_PREFILL_TC needs _exl3_C::{name}: rebuild "
                "the EXL3 extension from this tree (python "
                "csrc/exl3/build_standalone.py --out vllm), or unset the flag. "
                f"Detail: {exc}"
            ) from exc
    if not _moe_prefill_v2_fake_registered:
        torch.library.register_fake("_exl3_C::moe_out_prefill_v2")(_moe_v2_fake)
        if TRY_MOE_PREFILL:
            for name in ("moe_out_prefill_try", "moe_workspace_size_prefill_try"):
                try:
                    torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"VLLM_EXL3_MOE_PREFILL_UNCHUNKED=1 needs _exl3_C::{name}: "
                        "rebuild the EXL3 extension from this tree, "
                        f"or unset it. Detail: {exc}"
                    ) from exc
            torch.library.register_fake("_exl3_C::moe_out_prefill_try")(_moe_v2_fake)
            logger.info_once(
                "VLLM_EXL3_MOE_PREFILL_UNCHUNKED: moe_out_prefill_v2 calls of "
                "65..2048 rows of the whole-expert configuration "
                "(hidden 5120, intermediate 2304, top-6, "
                "96 of 384 experts local, bits_x2 6 or 4) run moe_out_prefill_try."
            )
        _moe_prefill_v2_fake_registered = True


def _try_moe_prefill(hidden: int, intermediate: int, topk: int, local: int) -> bool:
    """VLLM_EXL3_MOE_PREFILL_UNCHUNKED is set and the shape is the one it covers."""
    return TRY_MOE_PREFILL and (hidden, intermediate, topk, local) == (
        5120,
        2304,
        6,
        96,
    )


@cache
def moe_workspace_size_prefill_v2(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    """FP32 workspace elements of moe_out_prefill_v2 for calls of up to
    ``max_rows`` rows (they run in row chunks of at most 512 rows)."""
    ensure_extension()
    size = torch.ops._exl3_C.moe_workspace_size_prefill_v2(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )
    if _try_moe_prefill(hidden_size, intermediate_size, topk, local_experts):
        size = max(
            size,
            torch.ops._exl3_C.moe_workspace_size_prefill_try(
                max_rows, hidden_size, intermediate_size, topk, local_experts
            ),
        )
    return size


def moe_out_prefill_v2(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
    *,
    bits_x2: int,
) -> None:
    """moe_out's contract for calls of at least MOE_PREFILL_MIN_ROWS rows of
    a MOE_PREFILL_V2_CONFIGS configuration; see csrc/exl3/moe_prefill_v2.cu.
    ``bits_x2`` is the one bit width of every routed matrix of the layer.
    A call whose experts are stored in another format is refused on the host
    (_check_moe_prefill_v2_formats), never launched."""
    ensure_extension()
    _check_moe_prefill_v2_formats(
        packed_ptrs, packed_meta, x.shape[1], intermediate_size, bits_x2
    )
    op = torch.ops._exl3_C.moe_out_prefill_v2
    if (
        _try_moe_prefill(
            x.shape[1], intermediate_size, topk_ids.shape[1], packed_ptrs.shape[0]
        )
        and expert_map.numel() == 384
        and bits_x2 in (6, 4)
        and TRY_MOE_PREFILL_ROWS[0] <= x.shape[0] <= TRY_MOE_PREFILL_ROWS[1]
    ):
        op = torch.ops._exl3_C.moe_out_prefill_try
    op(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
        swiglu_limit,
    )


# (id(packed_meta), id(packed_ptrs)) -> (their data pointers, the
# (hidden, intermediate, bits_x2) their formats were checked against); an entry
# goes when either tensor does.
_moe_prefill_v2_checked: dict[tuple[int, int], tuple[int, int, set]] = {}


def _check_moe_prefill_v2_formats(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    bits_x2: int,
) -> None:
    """Refuse on the host a moe_out_prefill_v2 call whose local experts are not
    every one stored as (hidden_size, intermediate_size, bits_x2), codebook 2,
    with 16-byte aligned trellis bases: the kernels hold that layout at compile
    time and would stop the device on any other (their route kernel's check).
    The two tensors are read once per pair and format, on the first call, which
    must not be under CUDA graph capture (warmup_moe_prefill_v2 checks every
    layer's pair before any capture)."""
    key = (id(packed_meta), id(packed_ptrs))
    fmt = (hidden_size, intermediate_size, bits_x2)
    entry = _moe_prefill_v2_checked.get(key)
    pointers = (packed_meta.data_ptr(), packed_ptrs.data_ptr())
    if entry is not None and entry[:2] == pointers and fmt in entry[2]:
        return
    if packed_meta.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "EXL3 MoE prefill: unsupported call under CUDA graph capture: the "
            "experts' formats were not checked on the host first "
            "(warmup_moe_prefill_v2)"
        )
    want = moe_v2_expert_meta(hidden_size, intermediate_size, bits_x2)
    meta = packed_meta.tolist()
    trellis = packed_ptrs[..., 0].tolist()
    for local, (stored, bases) in enumerate(zip(meta, trellis)):
        for projection in range(3):
            if stored[projection] != want[projection]:
                raise RuntimeError(
                    f"EXL3 MoE prefill: unsupported call: local expert {local} "
                    f"projection {projection} is stored as {stored[projection]}, "
                    f"not {want[projection]}"
                )
            if bases[projection] % 16:
                raise RuntimeError(
                    f"EXL3 MoE prefill: unsupported call: local expert {local} "
                    f"projection {projection} trellis base is not 16-byte aligned"
                )
    if entry is None or entry[:2] != pointers:
        entry = (*pointers, set())
        if key not in _moe_prefill_v2_checked:
            for tensor in (packed_meta, packed_ptrs):
                weakref.finalize(tensor, _moe_prefill_v2_checked.pop, key, None)
        _moe_prefill_v2_checked[key] = entry
    entry[2].add(fmt)


# (device, hidden, intermediate, bits_x2) warmed by warmup_moe_prefill_v2.
_moe_prefill_v2_warm: set[tuple] = set()


def warmup_moe_prefill_v2(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    global_expert_ids: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    bits_x2: int,
    swiglu_limit: float = 0.0,
) -> None:
    """Run moe_out_prefill_v2 once per device, shape and bits_x2 before any
    CUDA graph capture.
    Each GEMM kernel sets its shared-memory limit on its first call, and the
    operator launches the same kernels at every row count (route tiles are
    chosen on the device), so one call of MOE_PREFILL_MIN_ROWS rows, all with
    local routes, loads every kernel it can launch for the layer's bits_x2.
    The temporary buffers are freed again. Every layer's experts have their
    formats checked on the host here, warmed key or not.
    """
    _check_moe_prefill_v2_formats(
        packed_ptrs, packed_meta, hidden_size, intermediate_size, bits_x2
    )
    device = packed_ptrs.device
    key = (device, hidden_size, intermediate_size, bits_x2)
    if key in _moe_prefill_v2_warm:
        return
    rows_list = [MOE_PREFILL_MIN_ROWS]
    if _try_moe_prefill(hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]):
        # moe_out_prefill_try launches other GEMM shapes above 1024 rows.
        rows_list.append(1025)
    for rows in rows_list:
        ids = torch.tensor(
            [global_expert_ids[:top_k]] * rows, dtype=torch.int32, device=device
        )
        weights = torch.full(
            (rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device
        )
        x = torch.ones((rows, hidden_size), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(x)
        workspace = torch.empty(
            moe_workspace_size_prefill_v2(
                rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
            ),
            dtype=torch.float32,
            device=device,
        )
        moe_out_prefill_v2(
            x,
            ids,
            weights,
            packed_ptrs,
            packed_meta,
            expert_map,
            intermediate_size,
            out,
            workspace,
            swiglu_limit=swiglu_limit,
            bits_x2=bits_x2,
        )
    torch.cuda.synchronize(device)
    _moe_prefill_v2_warm.add(key)


def ple_decode_out(
    packed_rows,
    global_ids,
    head_bias,
    head_offsets,
    head_vocab_sizes,
    owner_lo: int,
    owner_hi: int,
    bits: int,
    out,
) -> None:
    ensure_extension()
    torch.ops._exl3_C.ple_decode_out(
        packed_rows,
        global_ids,
        head_bias,
        head_offsets,
        head_vocab_sizes,
        owner_lo,
        owner_hi,
        bits,
        out,
    )


def ple_read_rows(
    fd: int,
    ids: torch.Tensor,
    dest: torch.Tensor,
    count: int,
    base: int,
    row_bytes: int,
    owner_lo: int,
    owner_hi: int,
    head_offsets: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    retries: int,
    max_threads: int,
) -> torch.Tensor:
    """Read PLE rows from storage in one host call that releases the GIL
    (csrc/exl3/ple_read.cpp); returns the [index, errno, offset] failures."""
    ensure_extension()
    return torch.ops._exl3_C.ple_read_rows(
        fd,
        ids,
        dest,
        count,
        base,
        row_bytes,
        owner_lo,
        owner_hi,
        head_offsets,
        head_vocab_sizes,
        retries,
        max_threads,
    )


def ple_host_reader(
    fd: int,
    ids: torch.Tensor,
    rows: torch.Tensor,
    base: int,
    row_bytes: int,
    owner_lo: int,
    owner_hi: int,
    head_offsets: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    retries: int,
    max_threads: int,
    slots: int,
):
    """A PLE row reader of ids into rows whose reads run as host functions
    on a CUDA stream (csrc/exl3/ple_read.cpp, PleHostReader)."""
    ensure_extension()
    return torch.classes._exl3_C.PleHostReader(
        fd,
        ids,
        rows,
        base,
        row_bytes,
        owner_lo,
        owner_hi,
        head_offsets,
        head_vocab_sizes,
        retries,
        max_threads,
        slots,
    )


def engram_host_fetch(
    fd: int,
    base: int,
    row_bytes: int,
    owner_lo: int,
    owner_hi: int,
    scales: torch.Tensor,
    table: torch.Tensor,
    ids: torch.Tensor,
    values: torch.Tensor,
    scales_out: torch.Tensor,
    cache_bytes: int,
    ways: int,
    retries: int,
    max_threads: int,
    slots: int,
):
    """The native fetch of one rank's DeepSeek V4.1 Engram rows into the
    values/scales staging (csrc/exl3/engram_fetch.cpp, EngramHostFetch)."""
    ensure_extension()
    return torch.classes._exl3_C.EngramHostFetch(
        fd,
        base,
        row_bytes,
        owner_lo,
        owner_hi,
        scales,
        table,
        ids,
        values,
        scales_out,
        cache_bytes,
        ways,
        retries,
        max_threads,
        slots,
    )


def _linear_scratch(device: torch.device) -> torch.Tensor:
    from vllm.utils.torch_utils import aux_stream, current_stream
    from vllm.v1.worker.workspace import current_workspace_manager

    ensure_extension()
    budget = linear_scratch_elements()

    def allocate():
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 scratch must be allocated before graph capture")
        # The compute and graph-capture streams never run model work at the
        # same time and share one scratch. The shared-expert aux stream
        # overlaps the compute stream, so it gets its own. Both are allocated
        # on first use, before memory profiling ends, and never resized.
        count = 1 if aux_stream() is None else 2
        return [
            torch.empty(budget, dtype=torch.float32, device=device)
            for _ in range(count)
        ]

    scratch = current_workspace_manager().get_persistent_resource(
        "exl3_linear_scratch", allocate
    )
    aux = aux_stream()
    return scratch[1] if aux is not None and current_stream() == aux else scratch[0]


def _linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits_x2: int,
    codebook: int,
    logical_out: int,
    row_chunk: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    workspace = _linear_scratch(x.device)
    out = torch.empty((x.shape[0], logical_out), dtype=out_dtype, device=x.device)
    operator = linear_out_splitk if _dense_splitk else linear_out
    if dense_prefill_tc.active:
        operator = dense_prefill_tc.route(operator, x.shape[0], _dense_splitk)
    for start in range(0, x.shape[0], row_chunk):
        operator(
            x[start : start + row_chunk],
            trellis,
            suh,
            svh,
            bits_x2,
            codebook,
            x.shape[1],
            logical_out,
            out[start : start + row_chunk],
            workspace,
        )
    if capture.active:
        capture_points.linear(x, trellis, out, operator)
    return out


def _linear_meta(
    x,
    trellis,
    suh,
    svh,
    bits_x2,
    codebook,
    logical_out,
    row_chunk,
    out_dtype=torch.bfloat16,
):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(x.dim() == 2)
    torch._check(out_dtype in LINEAR_OUT_DTYPES)
    return x.new_empty((x.shape[0], logical_out), dtype=out_dtype)


direct_register_custom_op(
    op_name="exl3_linear",
    op_func=_linear,
    mutates_args=[],
    fake_impl=_linear_meta,
)
