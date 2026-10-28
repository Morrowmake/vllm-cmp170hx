# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram row dequantization on sm_80: E4M3 values x E8M0 block scales.

The reference (``ParallelEngramEmbedding.forward`` of DeepSeek's
``inference/model.py``) computes ``values.float() * scales.float()`` per
32-wide block and rounds the FP32 product to BF16. Triton cannot convert
``float8e4nv`` on sm_80, so the kernel decodes both byte formats in
software:

- E4M3 through :func:`e4m3_bits_to_f32_fast` (exact for every finite
  pattern), with the two NaN patterns ``0x7F``/``0xFF`` restored to NaN;
- E8M0 byte ``b`` as the FP32 ``2**(b - 127)``: the byte shifted into the
  exponent field for ``b >= 1``, the FP32 denormal ``2**-127`` for ``b == 0``
  and NaN for ``b == 0xFF``, as ``torch.float8_e8m0fnu.float()`` gives.

Both factors are exact in FP32, so the single FP32 product and its
round-to-nearest-even BF16 conversion equal the reference bit for bit.

A NaN product (an E4M3 NaN value or the E8M0 NaN scale, e.g. a poisoned
row) is written as the BF16 bits ``0xFFFF``: torch's CPU float-to-BF16
conversion, which the reference uses, gives that one pattern for every NaN
whatever its sign and payload, while Triton's conversion keeps them. The
kernel stores BF16 bits through an int16 view of the output.
"""

import torch

from vllm_sm80.kernels.attention.triton_e4m3 import e4m3_bits_to_f32_fast
from vllm.triton_utils import tl, triton

ENGRAM_QUANT_BLOCK = 32


@triton.jit(do_not_specialize=["num_rows"])
def _engram_dequant_kernel(
    values,
    scales,
    out,
    num_rows,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < num_rows
    cols = tl.arange(0, DIM)
    mask = valid[:, None]
    bits = tl.load(values + rows[:, None] * DIM + cols[None, :], mask=mask, other=0)
    value = e4m3_bits_to_f32_fast(bits)
    value = tl.where((bits & 0x7F) == 0x7F, float("nan"), value)
    scale_bits = tl.load(
        scales + rows[:, None] * (DIM // QUANT_BLOCK) + (cols // QUANT_BLOCK)[None, :],
        mask=mask,
        other=127,
    ).to(tl.int32)
    scale = (scale_bits << 23).to(tl.float32, bitcast=True)
    # 2**-127 is the FP32 denormal with only mantissa bit 22 set.
    min_scale = tl.full((), 0x400000, tl.int32).to(tl.float32, bitcast=True)
    scale = tl.where(scale_bits == 0, min_scale, scale)
    scale = tl.where(scale_bits == 0xFF, float("nan"), scale)
    product = value * scale
    out_bits = product.to(tl.bfloat16).to(tl.int16, bitcast=True)
    out_bits = tl.where(product != product, -1, out_bits)  # NaN -> 0xFFFF
    tl.store(out + rows[:, None] * DIM + cols[None, :], out_bits, mask=mask)


def dequant_engram_rows_reference(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """The reference formula: ``[n, dim]`` E4M3 bytes and ``[n, dim / 32]``
    E8M0 bytes (both uint8) to ``[n, dim]`` BF16."""
    rows, dim = values.shape
    product = values.view(torch.float8_e4m3fn).float().view(
        rows, -1, ENGRAM_QUANT_BLOCK
    ) * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return product.view(rows, dim).to(torch.bfloat16)


def dequant_engram_rows(
    values: torch.Tensor, scales: torch.Tensor, out: torch.Tensor
) -> None:
    """Dequantize staged rows into ``out`` ([n, dim] BF16, contiguous).

    CUDA tensors take the Triton kernel; CPU tensors the reference formula.
    """
    rows, dim = values.shape
    if (
        values.dtype != torch.uint8
        or scales.dtype != torch.uint8
        or out.dtype != torch.bfloat16
        or scales.shape != (rows, dim // ENGRAM_QUANT_BLOCK)
        or out.shape != (rows, dim)
        or not (values.is_contiguous() and scales.is_contiguous())
        or not out.is_contiguous()
    ):
        raise ValueError("Engram dequant needs contiguous uint8 rows and BF16 output")
    if rows == 0:
        return
    if not values.is_cuda:
        out.copy_(dequant_engram_rows_reference(values, scales))
        return
    block_r = 16
    _engram_dequant_kernel[(triton.cdiv(rows, block_r),)](
        values,
        scales,
        out.view(torch.int16),
        rows,
        DIM=dim,
        QUANT_BLOCK=ENGRAM_QUANT_BLOCK,
        BLOCK_R=block_r,
    )
