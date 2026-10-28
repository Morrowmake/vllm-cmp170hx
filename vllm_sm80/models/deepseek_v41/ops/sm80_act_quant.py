# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software emulation of DeepSeek-V4.1's activation quantize-dequantize.

The reference model rounds several activations through FP8 or FP4 and writes
the dequantized BF16 values back in place (``act_quant(..., inplace=True)`` and
``fp4_act_quant(..., inplace=True)`` in its kernel module). sm_80 has no FP8 or
FP4 instructions and Triton cannot emit ``fp8e4nv`` there, so these functions
reproduce the rounding arithmetically. Every result is exact: scales are powers
of two or E4M3 values, the element grids are E4M3 or E2M1, and their products
fit BF16, so the BF16 outputs equal the reference bit for bit.

``act_quant_qdq_fp8(x, block=32)`` -- the reference ``act_quant`` with UE8M0
scales: per block of ``block`` values along the last dim,
``amax = max(absmax, 1e-4)``,
``s = 2 ** ceil_log2(amax * fp32(1/448))`` (ceil_log2 in its bit form
``exp - 127 + (mantissa != 0)``),
``y = bf16(e4m3(clamp(x / s, -448, 448)) * s)`` with round-to-nearest-even.

``fp4_qdq(x, block, scale)`` -- the reference ``fp4_act_quant``:
``scale="e8m0"``: ``amax = max(absmax, 6 * 2**-126)``, a power-of-two scale
from ``amax * fp32(1/6)`` as above; ``scale="e4m3"``:
``amax = max(absmax, 6 * 2**-9)``, ``s = e4m3(amax / 6)`` (saturating at 448
like the CUDA ``satfinite`` cast); then ``y = bf16(e2m1(clamp(x / s, -6, 6))
* s)`` with E2M1 round-to-nearest-even on {0, .5, 1, 1.5, 2, 3, 4, 6}.

All divisions are IEEE (round-to-nearest) fp32 divisions.

The torch functions are the definition and run on any device. The ``*_triton``
launchers compute the same bits on CUDA; :func:`qdq_fp8` / :func:`qdq_fp4`
dispatch between the two by device.
"""

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton

FP8_MAX = 448.0
FP4_MAX = 6.0
# Reciprocals as the reference kernel holds them: fp32 constants it multiplies
# by. The Triton code below spells the same fp32 values out as literals.
_FP8_MAX_INV = float(torch.tensor(1.0 / FP8_MAX, dtype=torch.float32))
_FP4_MAX_INV = float(torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
assert _FP8_MAX_INV == 0.0022321429569274187
assert _FP4_MAX_INV == 0.1666666716337204
_FP8_AMAX_FLOOR = 1e-4
_FP4_E8M0_AMAX_FLOOR = 6 * 2.0**-126
_FP4_E4M3_AMAX_FLOOR = 6 * 2.0**-9

SCALE_FORMATS = ("e8m0", "e4m3")


# ---------------------------------------------------------------------------
# torch definition
# ---------------------------------------------------------------------------


def _pow2_ceil(a: torch.Tensor) -> torch.Tensor:
    """``2 ** (exp - 127 + (mantissa != 0))`` of positive normal fp32 ``a``."""
    bits = a.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (e << 23).view(torch.float32)


def _grid_spacing(a: torch.Tensor, min_exp: int, mant_bits: int) -> torch.Tensor:
    """fp32 holding the grid spacing at ``a`` (>= 0) of a format with
    ``mant_bits`` mantissa bits and smallest normal exponent ``min_exp``."""
    exp = ((a.view(torch.int32) >> 23) & 0xFF) - 127
    ulp_exp = torch.clamp(exp, min=min_exp) - mant_bits
    return ((ulp_exp + 127) << 23).view(torch.float32)


def round_e4m3(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 on the E4M3 grid, round-to-nearest-even; ``|v| <= 448``.

    Normals have 3 mantissa bits from 2**-6 up, denormals step by 2**-9. The
    division by the power-of-two spacing is exact and ``torch.round`` rounds
    half to even, so this is the hardware cast for every finite input in range.
    """
    a = v.abs()
    ulp = _grid_spacing(a, -6, 3)
    return torch.copysign(torch.round(a / ulp) * ulp, v)


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 on the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, RNE;
    ``|v| <= 6``."""
    a = v.abs()
    ulp = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.copysign(torch.round(a / ulp) * ulp, v)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    n = x.shape[-1]
    if n % block:
        raise ValueError(f"last dim {n} is not a multiple of the block {block}")
    return x.float().unflatten(-1, (n // block, block))


def act_quant_qdq_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Reference ``act_quant(x, block, "ue8m0", inplace=True)``; returns a new
    tensor of ``x``'s dtype."""
    xf = _blocks(x, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(_FP8_AMAX_FLOOR)
    s = _pow2_ceil(amax * _FP8_MAX_INV)
    q = round_e4m3((xf / s).clamp(-FP8_MAX, FP8_MAX))
    return (q * s).flatten(-2).to(x.dtype)


def fp4_scale(amax: torch.Tensor, scale: str) -> torch.Tensor:
    """The per-block fp32 scale of :func:`fp4_qdq` from the block absmax."""
    if scale == "e8m0":
        return _pow2_ceil(amax.clamp_min(_FP4_E8M0_AMAX_FLOOR) * _FP4_MAX_INV)
    if scale == "e4m3":
        a = amax.clamp_min(_FP4_E4M3_AMAX_FLOOR) / FP4_MAX
        return round_e4m3(a.clamp_max(FP8_MAX))
    raise ValueError(f"scale must be one of {SCALE_FORMATS}, got {scale!r}")


def fp4_qdq(x: torch.Tensor, block: int, scale: str = "e8m0") -> torch.Tensor:
    """Reference ``fp4_act_quant(x, block, inplace=True, scale_dtype=...)``;
    returns a new tensor of ``x``'s dtype."""
    xf = _blocks(x, block)
    s = fp4_scale(xf.abs().amax(-1, keepdim=True), scale)
    q = round_e2m1((xf / s).clamp(-FP4_MAX, FP4_MAX))
    return (q * s).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------


@triton.jit
def _pow2_ceil_tl(a):
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (e << 23).to(tl.float32, bitcast=True)


@triton.jit
def _with_sign_of(r, v):
    """``r`` (>= 0) carrying the sign bit of ``v``, -0.0 included. Triton's
    unary minus is ``0 - r``, which turns a rounded-away -0.0 into +0.0; the
    hardware casts keep it."""
    sign = (v.to(tl.int32, bitcast=True) >> 31) << 31
    return (r.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _round_grid_tl(v, MIN_EXP: tl.constexpr, MANT_BITS: tl.constexpr):
    a = tl.abs(v)
    exp = ((a.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    ulp_exp = tl.maximum(exp, MIN_EXP) - MANT_BITS
    ulp = ((ulp_exp + 127) << 23).to(tl.float32, bitcast=True)
    r = libdevice.rint(tl.div_rn(a, ulp)) * ulp
    return _with_sign_of(r, v)


@triton.jit
def round_e4m3_tl(v):
    """Triton twin of :func:`round_e4m3` (fp32 tile in, fp32 tile out)."""
    return _round_grid_tl(v, -6, 3)


@triton.jit
def round_e2m1_tl(v):
    """Triton twin of :func:`round_e2m1`."""
    a = tl.abs(v)
    ulp = tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    r = libdevice.rint(tl.div_rn(a, ulp)) * ulp
    return _with_sign_of(r, v)


@triton.jit
def qdq_fp8_tile(x):
    """FP8/UE8M0 qdq of an fp32 ``[rows, BLOCK]`` tile, one block per row.
    Returns fp32 values that are exact in BF16."""
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    s = _pow2_ceil_tl(amax * 0.0022321429569274187)  # fp32(1/448)
    v = tl.minimum(tl.maximum(tl.div_rn(x, s[:, None]), -448.0), 448.0)
    return round_e4m3_tl(v) * s[:, None]


@triton.jit
def qdq_fp4_tile(x, E4M3_SCALE: tl.constexpr):
    """FP4 qdq of an fp32 ``[rows, BLOCK]`` tile, one block per row."""
    amax = tl.max(tl.abs(x), axis=1)
    if E4M3_SCALE:
        # 0.01171875 == 6 * 2**-9
        a = tl.div_rn(tl.maximum(amax, 0.01171875), 6.0)
        s = round_e4m3_tl(tl.minimum(a, 448.0))
    else:
        # 7.052966104933725e-38 == 6 * 2**-126; 0.1666666716337204 == fp32(1/6)
        s = _pow2_ceil_tl(tl.maximum(amax, 7.052966104933725e-38) * 0.1666666716337204)
    v = tl.minimum(tl.maximum(tl.div_rn(x, s[:, None]), -6.0), 6.0)
    return round_e2m1_tl(v) * s[:, None]


@triton.jit(do_not_specialize=["num_blocks"])
def _qdq_kernel(
    x_ptr,
    y_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,  # 0 fp8/ue8m0, 1 fp4/e8m0, 2 fp4/e4m3
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = (rows < num_blocks)[:, None]
    offs = rows[:, None] * BLOCK + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if MODE == 0:
        y = qdq_fp8_tile(x)
    else:
        y = qdq_fp4_tile(x, MODE == 2)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def _qdq_triton(x: torch.Tensor, block: int, mode: int) -> torch.Tensor:
    if x.shape[-1] % block:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {block}")
    src = x.contiguous()
    out = torch.empty_like(src)
    num_blocks = src.numel() // block
    if num_blocks:
        rows = 32
        _qdq_kernel[(triton.cdiv(num_blocks, rows),)](
            src, out, num_blocks, BLOCK=block, ROWS=rows, MODE=mode, num_warps=4
        )
    return out


def act_quant_qdq_fp8_triton(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """CUDA twin of :func:`act_quant_qdq_fp8` (same bits)."""
    return _qdq_triton(x, block, 0)


def fp4_qdq_triton(x: torch.Tensor, block: int, scale: str = "e8m0") -> torch.Tensor:
    """CUDA twin of :func:`fp4_qdq` (same bits)."""
    if scale not in SCALE_FORMATS:
        raise ValueError(f"scale must be one of {SCALE_FORMATS}, got {scale!r}")
    return _qdq_triton(x, block, 2 if scale == "e4m3" else 1)


def qdq_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """:func:`act_quant_qdq_fp8`, through Triton on CUDA tensors."""
    if x.is_cuda:
        return act_quant_qdq_fp8_triton(x, block)
    return act_quant_qdq_fp8(x, block)


def qdq_fp4(x: torch.Tensor, block: int, scale: str = "e8m0") -> torch.Tensor:
    """:func:`fp4_qdq`, through Triton on CUDA tensors."""
    if x.is_cuda:
        return fp4_qdq_triton(x, block, scale)
    return fp4_qdq(x, block, scale)
