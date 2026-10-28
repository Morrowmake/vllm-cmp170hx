# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Portions adapted from GNmac-lab/deepseek-v41-flash-1017-cmp170hx
# (https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) at bbaf367,
# patches/dsv41_wo_a.py (the E4M3 decode and the grouped tile loop):
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 GNmac-lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""DeepSeek-V4.1 grouped FP8 ``wo_a`` as one Triton W8A16 GEMM (sm_80).

:func:`wo_a_grouped` projects the attention output ``x`` [T, G, D] (BF16,
unit stride along D, any row and group stride) with the block-diagonal MXFP8
``wo_a``: output group g is ``x[:, g] @ W_g^T``. ``weight`` [G * R, D] holds
the E4M3 codes as bytes and ``scale`` [G * R, D / 32] the E8M0 exponent of
every 32 inputs of a row, as the hybrid checkpoint's ``ModelOptLinearMethod``
holds them before Marlin's repack. One launch covers every group and writes
the [T, G * R] result that ``wo_b`` reads; no per-group GEMMs, no concat.

Arithmetic, the same as Marlin's MXFP8 W8A16 GEMM up to the order of the FP32
sums: each weight is decoded to BF16 in registers by one inline-PTX block,
bytes to scaled weight (the E4M3 code by bit moves and an exact multiply by
2**120; the scale as BF16 bits ``e << 7``, 2**(e-127) for e in 1..254 and 0
for e = 0, as Marlin's ``dequant_fp8_scales``; the product with a power of
two is exact outside underflow), multiplied with the BF16 inputs on the BF16
tensor cores with FP32 accumulation, and rounded once to BF16.

Shape of the launch: one program per (row block, output column tile), the
column tiles of all groups in one grid (a tile never crosses a group:
``R % BN == 0``), the column index fastest so neighbouring programs share the
row block. Calls of at most 64 rows take a single row block, so the weight is
read once; their 32-column tiles give 64 programs for G * R = 2048. Such a
block is up to four 16-row MMA tiles over one decoded weight tile: one tile
of 16 rows keeps the MMA's warps side by side along the columns, so no two
warps decode the same weights (a taller tile would stack warps along the
rows, each decoding the columns it multiplies).

:class:`WoAGroupedGemmMethod` is the grouped method of such a ``wo_a``
(DeepSeek-V4.1 attention adopts it with
``VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM``): after loading it keeps the codes and
scales as loaded (no Marlin repack, no per-group sub-layers; the same bytes
as Marlin's copy) and projects through :func:`wo_a_grouped`. For the dev A/B
switch it also builds the Marlin sub-layers of its parent, and ``gemm``
selects the arm.

CUDA graphs and the first eager prefill: :data:`WO_A_GROUPED_WARMUP` compiles
the kernel at one row count of each :data:`CONFIGS` entry for every adopted
``wo_a`` shape, target and drafter, before capture (registered by
:meth:`WoAGroupedGemmMethod.adopt`).
"""

import weakref
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm_sm80 import capture
from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

logger = init_logger(__name__)

# (max rows, BM, MT, BN, BK, warps, stages), first match by row count: a row
# block is MT tiles of BM rows.
CONFIGS = (
    (16, 16, 1, 32, 256, 4, 3),
    (32, 16, 2, 32, 256, 4, 3),
    (48, 16, 3, 32, 128, 4, 3),
    (64, 16, 4, 32, 128, 4, 3),
    (256, 64, 1, 64, 64, 4, 3),
    (None, 128, 1, 128, 32, 8, 3),
)
# Every tile width divides R and every K step divides D for a supported shape.
_MAX_BN = max(c[3] for c in CONFIGS)
_MAX_BK = max(c[4] for c in CONFIGS)

# Four E4M3 bytes ($2) and their four E8M0 scale bytes ($3) to four scaled
# BF16 weights ($0: elements 0-1, $1: elements 2-3). Sign and the 7 exponent
# and mantissa bits move into BF16's top bits (exponent bias 7 instead of
# 127); each scale byte becomes the exponent field of a BF16 (``e << 7``).
# Then two exact multiplies with -0 as the addend (so subnormal codes and the
# sign of zero come out exact): by 2**120 (0x7B80) and by the scale.
_SCALED_E4M3_TO_BF16_ASM = tl.constexpr("""{
        .reg .b32 hi, lo, t, bias, nz, zero, s;
        mov.b32 bias, 0x7B807B80;
        mov.b32 nz, 0x80008000;
        mov.b32 zero, 0;
        and.b32 t, $2, 0x80008000;
        and.b32 hi, $2, 0x7F007F00;
        shr.b32 hi, hi, 4;
        or.b32 hi, hi, t;
        shl.b32 lo, $2, 8;
        and.b32 t, lo, 0x80008000;
        and.b32 lo, lo, 0x7F007F00;
        shr.b32 lo, lo, 4;
        or.b32 lo, lo, t;
        prmt.b32 $0, lo, hi, 0x5410;
        prmt.b32 $1, lo, hi, 0x7632;
        fma.rn.bf16x2 $0, $0, bias, nz;
        fma.rn.bf16x2 $1, $1, bias, nz;
        prmt.b32 s, $3, zero, 0x7170;
        shl.b32 s, s, 7;
        fma.rn.bf16x2 $0, $0, s, nz;
        prmt.b32 s, $3, zero, 0x7372;
        shl.b32 s, s, 7;
        fma.rn.bf16x2 $1, $1, s, nz;
        }""")


@triton.jit
def _scaled_e4m3_to_bf16(w_u8, e8m0):
    return tl.inline_asm_elementwise(
        asm=_SCALED_E4M3_TO_BF16_ASM,
        constraints="=r,=r,r,r",
        args=[w_u8, e8m0],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _x_tile(x_rows, k, rows_ok):
    return tl.load(x_rows + k[None, :], mask=rows_ok[:, None], other=0.0)


@triton.jit
def _store_tile(o_ptr, offs_m, offs_n, acc, M, N: tl.constexpr):
    o = o_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    tl.store(o, acc.to(o_ptr.dtype.element_ty), mask=(offs_m < M)[:, None])


@triton.jit(do_not_specialize=["M"])
def _wo_a_grouped_kernel(
    x_ptr, w_ptr, s_ptr, o_ptr, M, stride_xm, stride_xg,
    G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, MT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # x: [M, G, D] bf16 (unit stride on D); w: [G * R, D] e4m3 bytes;
    # s: [G * R, D / 32] e8m0 bytes; o: [M, G * R] contiguous.
    NB: tl.constexpr = G * R // BN
    pid = tl.program_id(0)
    pid_m = pid // NB
    pid_n = pid % NB
    g = pid_n // (R // BN)
    m0 = pid_m * BM * MT + tl.arange(0, BM)  # rows of tile 0; tile i: m0 + i * BM
    offs_n = pid_n * BN + tl.arange(0, BN)  # rows of w, columns of o
    offs_k = tl.arange(0, BK)
    offs_kb = tl.arange(0, BK // 32)
    x_rows = x_ptr + m0.to(tl.int64)[:, None] * stride_xm + g * stride_xg
    x_tile = BM * stride_xm
    w_cols = w_ptr + offs_n[None, :] * D
    s_cols = s_ptr + offs_n[None, :] * (D // 32)
    acc0 = tl.zeros((BM, BN), dtype=tl.float32)
    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    acc3 = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        wu = tl.load(w_cols + (k0 + offs_k)[:, None])  # [BK, BN]
        e = tl.load(s_cols + (k0 // 32 + offs_kb)[:, None])  # [BK / 32, BN]
        e = tl.reshape(tl.broadcast_to(e[:, None, :], (BK // 32, 32, BN)), (BK, BN))
        w = _scaled_e4m3_to_bf16(wu, e)
        k = k0 + offs_k
        acc0 = tl.dot(_x_tile(x_rows, k, m0 < M), w, acc0)
        if MT > 1:
            acc1 = tl.dot(_x_tile(x_rows + x_tile, k, m0 + BM < M), w, acc1)
        if MT > 2:
            acc2 = tl.dot(_x_tile(x_rows + 2 * x_tile, k, m0 + 2 * BM < M), w, acc2)
        if MT > 3:
            acc3 = tl.dot(_x_tile(x_rows + 3 * x_tile, k, m0 + 3 * BM < M), w, acc3)
    _store_tile(o_ptr, m0, offs_n, acc0, M, G * R)
    if MT > 1:
        _store_tile(o_ptr, m0 + BM, offs_n, acc1, M, G * R)
    if MT > 2:
        _store_tile(o_ptr, m0 + 2 * BM, offs_n, acc2, M, G * R)
    if MT > 3:
        _store_tile(o_ptr, m0 + 3 * BM, offs_n, acc3, M, G * R)


def config_for(rows: int) -> tuple[int, int, int, int, int, int]:
    """(BM, MT, BN, BK, warps, stages) of a call of ``rows`` rows."""
    for max_rows, *cfg in CONFIGS:
        if max_rows is None or rows <= max_rows:
            return tuple(cfg)
    raise AssertionError("the last config takes every row count")


# The smallest row count of each CONFIGS entry: the kernel does not specialise
# on M, so its config is all a call's row count puts into the cache key.
WARMUP_ROWS = (1, *(c[0] + 1 for c in CONFIGS if c[0] is not None))


def supports(groups: int, rows_per_group: int, inputs: int) -> str | None:
    """Why the kernel cannot take this ``wo_a`` shape, or None."""
    if groups < 1 or rows_per_group % _MAX_BN:
        return f"{rows_per_group} outputs per group are not a multiple of {_MAX_BN}"
    if inputs % _MAX_BK:
        return f"{inputs} inputs per group are not a multiple of {_MAX_BK}"
    return None


def grid_for(rows: int, outputs: int) -> tuple[int]:
    bm, mt, bn = config_for(rows)[:3]
    return (triton.cdiv(rows, bm * mt) * (outputs // bn),)


def wo_a_grouped(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``x`` [T, G, D] BF16 -> [T, G * R] (``weight`` [G * R, D] E4M3 bytes,
    ``scale`` [G * R, D / 32] E8M0 bytes, both contiguous)."""
    rows, groups, inputs = x.shape
    outputs = weight.shape[0]
    if x.stride(2) != 1:
        raise ValueError("wo_a_grouped reads unit-stride input rows")
    out = torch.empty(rows, outputs, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    bm, mt, bn, bk, warps, stages = config_for(rows)
    _wo_a_grouped_kernel[grid_for(rows, outputs)](
        x, weight, scale, out, rows, x.stride(0), x.stride(1),
        G=groups, R=outputs // groups, D=inputs, BM=bm, MT=mt, BN=bn, BK=bk,
        num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return out


def unsupported(layer: nn.Module, groups: int) -> str | None:
    """Why the linear ``layer`` (built, not loaded) cannot run as
    :class:`WoAGroupedGemmMethod` with ``groups`` groups, or None."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static

    method = layer.quant_method
    if not isinstance(method, ModelOptLinearMethod) or method.spec.weight != kMxfp8Static:
        return f"wo_a is not an MXFP8 ModelOpt linear ({type(method).__name__})"
    if method.input_dtype != torch.bfloat16 or method.out_dtype != torch.bfloat16:
        return "wo_a does not take and return BF16"
    if getattr(layer, "bias", None) is not None:
        return "wo_a has a bias"
    if layer.output_size_per_partition % groups:
        return f"{layer.output_size_per_partition} outputs do not split into {groups} groups"
    return supports(
        groups, layer.output_size_per_partition // groups, layer.input_size_per_partition
    )


class WoAGroupedGemmMethod(GroupedLinearMethod):
    """The grouped MXFP8 ``wo_a`` of DeepSeek-V4.1 attention as one
    :func:`wo_a_grouped` launch.

    ``gemm`` selects the kernel (else the parent's per-group Marlin GEMMs,
    which exist only with ``keep_groups``: the dev A/B switch's off arm)."""

    def __init__(
        self,
        inner,
        groups: int,
        gemm: bool = True,
        keep_groups: bool = False,
    ):
        super().__init__(inner, groups)
        self.gemm = gemm
        self.keep_groups = keep_groups

    @classmethod
    def adopt(
        cls,
        layer: nn.Module,
        groups: int,
        gemm: bool = True,
        keep_groups: bool = False,
    ) -> None:
        reason = unsupported(layer, groups)
        if reason is not None:
            raise ValueError(f"VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM: {reason}")
        if capture.captures("hybrid"):
            # The hybrid capture records the grouped wo_a through
            # GroupedLinearMethod.apply_grouped, which this method replaces:
            # its wo_a cases would vanish.
            raise ValueError(
                "VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM: the hybrid capture records "
                "the per-group Marlin GEMMs, not this kernel; capture with the flag off"
            )
        layer.quant_method = cls(layer.quant_method, groups, gemm, keep_groups)
        if gemm:
            _ADOPTED.add(layer)
            WO_A_GROUPED_WARMUP.register_warmup()
        logger.info_once(
            "DeepSeek-V4.1 sm_80: grouped MXFP8 wo_a as one Triton GEMM "
            "(VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM=%d%s)",
            int(gemm),
            ", per-group Marlin kept for the dev A/B switch" if keep_groups else "",
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        params = dict(layer.named_parameters(recurse=False))
        if set(params) != {"weight", "weight_scale"}:
            raise ValueError(f"MXFP8 wo_a parameters are {sorted(params)}")
        weight, scale = params["weight"].data, params["weight_scale"].data
        rows, inputs = layer.output_size_per_partition, layer.input_size_per_partition
        if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (rows, inputs):
            raise ValueError(f"MXFP8 wo_a weight {weight.dtype} {list(weight.shape)}")
        if scale.element_size() != 1 or tuple(scale.shape) != (rows, inputs // 32):
            raise ValueError(f"MXFP8 wo_a scale {scale.dtype} {list(scale.shape)}")
        if self.keep_groups:
            # Copies each group's rows into its Marlin sub-layer and drops
            # the parameters; weight and scale stay referenced here.
            super().process_weights_after_loading(layer)
        else:
            for name in params:
                delattr(layer, name)
        layer.register_buffer(
            "wo_a_codes", weight.view(torch.uint8).contiguous(), persistent=False
        )
        layer.register_buffer(
            "wo_a_scales", scale.view(torch.uint8).contiguous(), persistent=False
        )

    def apply_grouped(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """``x`` [tokens, groups, input] -> [tokens, groups * output]."""
        if not self.gemm:
            if not self.keep_groups:
                raise RuntimeError(
                    "the per-group Marlin wo_a was not built (boot with the "
                    "dev A/B switch dsv41_wo_a_grouped_gemm)"
                )
            return super().apply_grouped(layer, x)
        if x.dim() != 3 or x.shape[1] != self.groups:
            raise ValueError(f"Grouped input must be [tokens, {self.groups}, input]")
        return wo_a_grouped(x, layer.wo_a_codes, layer.wo_a_scales)


# The wo_a layers running wo_a_grouped, the target's and the drafter's: the
# drafter loads outside the runner's warmup registry, so the one registration
# (made while the target loads) reads the set at warmup time.
_ADOPTED: "weakref.WeakSet[nn.Module]" = weakref.WeakSet()


def _cuda(device: torch.device) -> bool:
    return device.type == "cuda"


class WoAGroupedWarmup(VllmJitKernel["WoAGroupedWarmup.Key"]):
    """Boot-time compile of :func:`wo_a_grouped` (the entry
    :meth:`WoAGroupedGemmMethod.apply_grouped` calls) for each adopted ``wo_a``
    shape at :data:`WARMUP_ROWS`, on the layer's codes and scales and a BF16
    input of the served [T, G, D] layout (unit stride along D, row and group
    strides multiples of 16 elements)."""

    @dataclass(frozen=True)
    class Key:
        rows: int
        groups: int
        outputs: int
        inputs: int
        aligned: tuple
        layer: object = field(compare=False, hash=False, repr=False)

    def get_warmup_keys(self) -> list["WoAGroupedWarmup.Key"]:
        keys: dict = {}
        for layer in list(_ADOPTED):
            codes, scales = layer.wo_a_codes, layer.wo_a_scales
            if not _cuda(codes.device):
                continue
            aligned = (codes.data_ptr() % 16 == 0, scales.data_ptr() % 16 == 0)
            for rows in WARMUP_ROWS:
                key = self.Key(rows, layer.quant_method.groups, codes.shape[0],
                               codes.shape[1], aligned, layer)
                keys.setdefault(key, None)
        return list(keys)

    def compile(self, compile_key: "WoAGroupedWarmup.Key") -> None:
        layer = compile_key.layer
        dev = layer.wo_a_codes.device
        x = torch.zeros(compile_key.rows, compile_key.groups, compile_key.inputs,
                        dtype=torch.bfloat16, device=dev)
        wo_a_grouped(x, layer.wo_a_codes, layer.wo_a_scales)
        torch.cuda.synchronize(dev)


WO_A_GROUPED_WARMUP = WoAGroupedWarmup()
