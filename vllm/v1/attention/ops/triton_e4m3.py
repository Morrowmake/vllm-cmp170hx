# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software float8_e4m3fn <-> float32 conversion for Triton kernels.

Triton refuses ``tl.float8e4nv`` on devices below SM89/SM90 ("supported fp8
dtypes are fp8e4b15, fp8e5"), but the DSA indexer K cache is a byte cache
holding e4m3 values, so kernels that read or write it on Ampere need a
bit-manipulation conversion. Both helpers operate on integer views:

* :func:`e4m3_bits_to_f32` maps a ``uint8`` tensor holding e4m3 bit patterns
  to ``float32`` (exact; NaN for ``0x7F`` / ``0xFF``);
  :func:`e4m3_bits_to_f32_fast` is the 8-op variant the logits kernels use
  (identical except that the two NaN patterns come out as ``+-480``).
* :func:`f32_to_e4m3_bits` maps ``float32`` to the e4m3 bit pattern as
  ``uint8`` with the exact semantics of ``torch.Tensor.to(torch.float8_e4m3fn)``
  on CUDA (round-to-nearest-even, denormals, saturation to ``+-448`` for
  anything larger, NaN preserved). The kpool kernels clamp to ``[-448, 448]``
  before converting anyway.

The host-side predicate :func:`triton_fp8_e4m3_native` says whether a kernel
may instead store through a ``float8_e4m3fn`` pointer and let Triton emit the
hardware ``cvt`` (Hopper and newer).
"""

import functools

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

@functools.cache
def triton_fp8_e4m3_native() -> bool:
    """Whether Triton on this device can cast to ``tl.float8e4nv`` natively.

    Only Hopper+ takes the native path; SM89 (Ada) can in principle, but it
    is left on the software path so the two are never mixed by accident.
    """
    return current_platform.is_cuda() and current_platform.has_device_capability(90)


@triton.jit
def e4m3_bits_to_f32_fast(x):
    """``uint8`` e4m3fn bit patterns -> ``float32``; NaN patterns give +-480.

    Places sign and the 7 magnitude bits at fp32 bits 31 and [20, 27) so the
    e4m3 exponent lands in the fp32 exponent field, then multiplies by 2^120
    to rebias (127 - 7). Normals come out exact; e4m3 denormals go through
    fp32 denormals (m * 2^-129 * 2^120 = m * 2^-9), also exact -- Triton emits
    plain ``mul.f32`` (no ftz), which the exhaustive test asserts. 8 integer /
    float ops per element, so it is what the logits kernels use.
    """
    xi = x.to(tl.int32)
    bits = ((xi & 0x7F) << 20) | ((xi & 0x80) << 24)
    return bits.to(tl.float32, bitcast=True) * 1.3292279957849159e36  # 2^120


@triton.jit
def e4m3_bits_to_bf16_raw(x):
    """``uint8`` e4m3fn bit patterns -> ``bfloat16`` holding ``value * 2^-120``.

    Pure bit placement: sign to bit 15, the 7 magnitude bits to bits [4, 11),
    so the e4m3 exponent lands in the bf16 exponent field with bias 127
    instead of 7. Normals and denormals are exact (bf16 denormals are
    representable and never touched by arithmetic here); the two NaN patterns
    become finite (+-480 * 2^-120), like :func:`e4m3_bits_to_f32_fast`.
    Callers fold the 2^120 back in elsewhere (e.g. into the other MMA operand
    and the per-row weights), which is exact and keeps every intermediate in
    the fp32 normal range.

    Four bytes per PTX instance (``prmt`` spreads them into two 16-bit lanes
    each), ~2.5 instructions per element versus 8 for the fp32 route -- the
    logits kernels' K-tile dequant was issue-bound on sm_80.
    """
    return tl.inline_asm_elementwise(
        """
        {
        .reg .b32 a, b, m, s;
        prmt.b32 a, $2, 0, 0x4140;
        prmt.b32 b, $2, 0, 0x4342;
        shl.b32 m, a, 4;
        and.b32 m, m, 0x07F007F0;
        shl.b32 s, a, 8;
        and.b32 s, s, 0x80008000;
        or.b32 $0, m, s;
        shl.b32 m, b, 4;
        and.b32 m, m, 0x07F007F0;
        shl.b32 s, b, 8;
        and.b32 s, s, 0x80008000;
        or.b32 $1, m, s;
        }
        """,
        "=r,=r,r",
        [x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )

@triton.jit
def e4m3_bits_to_f32(x):
    """``uint8`` e4m3fn bit patterns -> exact ``float32`` values (NaN for
    ``0x7F`` / ``0xFF``)."""
    xi = x.to(tl.int32)
    v = e4m3_bits_to_f32_fast(x)
    return tl.where((xi & 0x7F) == 0x7F, float("nan"), v)


@triton.jit
def f32_to_e4m3_bits(x):
    """``float32`` -> ``uint8`` e4m3fn bits, matching torch's CUDA conversion.

    Round-to-nearest-even in both the normal and the denormal range (the
    ``c10::detail::fp8e4m3fn_from_fp32_value`` bit tricks), but with the
    saturating-finite behaviour of the CUDA cast torch uses on GPU: every
    finite magnitude that would round past 448 (including Inf) becomes
    ``+-448`` (``0x7E``); NaN input stays NaN (``0x7F``). Sign is kept.
    """
    # fp32 bit patterns used below: 0x3C800000 = 2^-6 (smallest e4m3 normal),
    # 0x43F00000 = 480 (first magnitude past 448), 0x46800000 = 2^14 (whose
    # fp32 ulp is the e4m3 denormal ulp 2^-9), 0x7F800000 = +Inf.
    fb = x.to(tl.int32, bitcast=True)
    sign = (fb >> 31) & 1
    fb = fb & 0x7FFFFFFF
    # Denormal / underflow range (|x| < 2^-6): let the fp32 adder round to a
    # multiple of 2^-9 by adding 2^14, then read off the units.
    dn = (fb.to(tl.float32, bitcast=True) + 16384.0).to(tl.int32, bitcast=True)
    dn = dn - 1182793728  # 0x46800000
    # Normal range: rebias the exponent, add 0x7FFFF plus the LSB of the kept
    # mantissa (round-half-to-even at bit 20), then shift the result down.
    mant_odd = (fb >> 20) & 1
    nb = fb + (-1006632960) + 0x7FFFF + mant_odd  # (-120 << 23)
    nrm = (nb >> 20) & 0xFF
    nrm = tl.where(nrm == 0x7F, 0x7E, nrm)  # [464, 480) rounds up: saturate
    r = tl.where(fb < 1015021568, dn, nrm)  # 0x3C800000 == 2^-6
    r = tl.where(fb >= 1139802112, 0x7E, r)  # 0x43F00000 == 480.0: saturate
    r = tl.where(fb > 2139095040, 0x7F, r)  # 0x7F800000: NaN stays NaN
    r = r | (sign << 7)
    return r.to(tl.uint8)


@triton.jit
def store_fp8_e4m3(ptr, value, mask, FP8_NATIVE: tl.constexpr):
    """Store fp32 ``value`` as e4m3 through ``ptr``.

    With ``FP8_NATIVE`` the pointer is a ``float8_e4m3fn`` pointer and Triton
    emits the hardware conversion; otherwise it is the ``uint8`` byte view of
    the same memory and the bits are computed in software.
    """
    if FP8_NATIVE:
        tl.store(ptr, value, mask=mask)
    else:
        tl.store(ptr, f32_to_e4m3_bits(value), mask=mask)
