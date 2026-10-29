# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exhaustive checks of the sm_80 FP8/FP4 quantize-dequantize emulation.

Every finite bf16 bit pattern is placed at the start, the middle and the end
of a block, alone (it sets the scale) and next to fixed companions (the
companion sets the scale and the pattern lands anywhere on the grid, down into
the E4M3 denormals and up to saturation). The module's output must equal, bit
for bit, an independent implementation: the scale exponent from ``frexp``, the
E4M3 element from ``torch.Tensor.to(torch.float8_e4m3fn)`` and the E2M1
element from an explicit nearest-value table with ties to the even code.

    pytest -q tests/models/deepseek_v41/sm80/test_act_quant.py
"""

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import (
    act_quant_qdq_fp8,
    act_quant_qdq_fp8_triton,
    fp4_qdq,
    fp4_qdq_triton,
)

_INV448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
_INV6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _all_finite_bf16() -> torch.Tensor:
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16)
    return x[torch.isfinite(x.float())]


def _pow2_ceil_frexp(a: torch.Tensor) -> torch.Tensor:
    mant, exp = torch.frexp(a)  # a = mant * 2**exp, mant in [0.5, 1)
    k = torch.where(mant == 0.5, exp - 1, exp)
    return torch.exp2(k.double()).float()


def _e2m1_table(v: torch.Tensor) -> torch.Tensor:
    a = v.double().abs().unsqueeze(-1)
    dist = (a - _E2M1).abs()
    nearest = dist == dist.min(-1, keepdim=True).values
    even = torch.arange(8) % 2 == 0
    tie = nearest.sum(-1, keepdim=True) > 1
    pick = torch.where(tie, nearest & even, nearest)
    mag = _E2M1[pick.to(torch.int8).argmax(-1)]
    return torch.copysign(mag.float(), v)


def ref_fp8(x: torch.Tensor, block: int) -> torch.Tensor:
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil_frexp(amax * _INV448)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s).flatten(-2).to(x.dtype)


def ref_fp4(x: torch.Tensor, block: int, scale: str) -> torch.Tensor:
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True)
    if scale == "e8m0":
        s = _pow2_ceil_frexp(amax.clamp_min(6 * 2.0**-126) * _INV6)
    else:
        a = (amax.clamp_min(6 * 2.0**-9) / 6.0).clamp_max(448.0)
        s = a.to(torch.float8_e4m3fn).float()
    q = _e2m1_table((xf / s).clamp(-6.0, 6.0))
    return (q * s).flatten(-2).to(x.dtype)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _assert_bits_equal(got: torch.Tensor, want: torch.Tensor) -> None:
    if not _bits_equal(got, want):
        bad = (got.view(torch.int16) != want.view(torch.int16)).nonzero()
        r = bad[0].tolist()
        raise AssertionError(
            f"{bad.shape[0]} mismatches, first at {r}: got "
            f"{got[tuple(r)].item()!r} want {want[tuple(r)].item()!r}"
        )


def _sweep(block: int, pos: int, companion: float | None, cpos: int):
    patterns = _all_finite_bf16()
    x = torch.zeros(patterns.numel(), block, dtype=torch.bfloat16)
    if companion is not None:
        x[:, cpos] = companion
    x[:, pos] = patterns
    return x


_POSITIONS = ("first", "middle", "last")


def _pos(block: int, which: str) -> tuple[int, int]:
    p = {"first": 0, "middle": block // 2, "last": block - 1}[which]
    return p, (block - 1 if p == 0 else 0)


@pytest.mark.parametrize("which", _POSITIONS)
@pytest.mark.parametrize(
    "companion", [None, 1.0, 448.0, 3.0, 2.0**-10, 1.5e-4, 65536.0, -0.75]
)
def test_fp8_every_bf16_pattern(which: str, companion: float | None) -> None:
    pos, cpos = _pos(32, which)
    x = _sweep(32, pos, companion, cpos)
    _assert_bits_equal(act_quant_qdq_fp8(x, 32), ref_fp8(x, 32))


@pytest.mark.parametrize("scale,block", [("e8m0", 32), ("e4m3", 16)])
@pytest.mark.parametrize("which", _POSITIONS)
@pytest.mark.parametrize(
    "companion", [None, 6.0, 1.0, 5.0, 2.0**-9, 3000.0, 6.375, -0.1]
)
def test_fp4_every_bf16_pattern(
    scale: str, block: int, which: str, companion: float | None
) -> None:
    pos, cpos = _pos(block, which)
    x = _sweep(block, pos, companion, cpos)
    _assert_bits_equal(fp4_qdq(x, block, scale), ref_fp4(x, block, scale))


@pytest.mark.parametrize("scale,block", [("e8m0", 32), ("e4m3", 16)])
def test_e2m1_ties_round_to_even(scale: str, block: int) -> None:
    # amax 6 gives scale exactly 1 for both formats, so x / s is x itself.
    ties = {0.25: 0.0, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0, 3.5: 4.0, 5.0: 4.0}
    for v, want in ties.items():
        for sign in (1.0, -1.0):
            x = torch.zeros(1, block, dtype=torch.bfloat16)
            x[0, 0] = 6.0
            x[0, 1] = sign * v
            y = fp4_qdq(x, block, scale)
            assert y[0, 1].item() == sign * want, (scale, v, sign, y[0, 1].item())
            assert y[0, 0].item() == 6.0
            assert ref_fp4(x, block, scale)[0, 1].item() == sign * want


def test_all_zero_blocks() -> None:
    x = torch.zeros(3, 64, dtype=torch.bfloat16)
    for y in (
        act_quant_qdq_fp8(x, 32),
        fp4_qdq(x, 32, "e8m0"),
        fp4_qdq(x[:, :48], 16, "e4m3"),
    ):
        assert torch.equal(y.view(torch.int16), torch.zeros_like(y.view(torch.int16)))
    # Negative zeros keep their sign, as the hardware casts do.
    nz = torch.full((1, 32), -0.0, dtype=torch.bfloat16)
    assert torch.signbit(act_quant_qdq_fp8(nz, 32).float()).all()
    assert torch.signbit(fp4_qdq(nz, 32, "e8m0").float()).all()


def test_fp8_subnormal_results() -> None:
    # amax 448 -> scale 1; E4M3 denormals step by 2**-9.
    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    x[0, 0] = 448.0
    cases = {
        3 * 2.0**-9: 3 * 2.0**-9,  # exact denormal
        2.0**-10: 0.0,  # half the smallest denormal: tie to even (0)
        3 * 2.0**-10: 2.0**-8,  # 1.5 denormal steps: tie to even (2 steps)
        2.0**-6: 2.0**-6,  # smallest normal
        7 * 2.0**-9: 7 * 2.0**-9,  # largest denormal
    }
    for v, want in cases.items():
        x[0, 1] = v
        assert act_quant_qdq_fp8(x, 32)[0, 1].item() == want, v
        assert ref_fp8(x, 32)[0, 1].item() == want, v


def test_fp4_saturation() -> None:
    # e4m3(6.375 / 6 = 1.0625) ties down to 1.0, so the block max lands at
    # 6.375 / 1 and saturates to 6.
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    x[0, 0] = 6.375
    assert fp4_qdq(x, 16, "e4m3")[0, 0].item() == 6.0
    # The E4M3 scale itself saturates at 448 (satfinite): 3000 / 6 = 500.
    x[0, 0] = 3000.0
    assert fp4_qdq(x, 16, "e4m3")[0, 0].item() == 6.0 * 448.0
    _assert_bits_equal(fp4_qdq(x, 16, "e4m3"), ref_fp4(x, 16, "e4m3"))


def test_shapes_and_dtype() -> None:
    x = torch.randn(2, 3, 512, dtype=torch.bfloat16)
    assert act_quant_qdq_fp8(x).shape == x.shape
    assert fp4_qdq(x, 16, "e4m3").dtype == torch.bfloat16
    with pytest.raises(ValueError):
        act_quant_qdq_fp8(x[..., :40], 32)
    with pytest.raises(ValueError):
        fp4_qdq(x, 32, "ue8m0")


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@_cuda
@pytest.mark.parametrize("which", _POSITIONS)
@pytest.mark.parametrize("companion", [None, 1.0, 448.0, 2.0**-10, 6.375, 3000.0])
def test_triton_matches_torch_every_pattern(which: str, companion: float | None) -> None:
    for block, fn, tfn in (
        (32, lambda x: act_quant_qdq_fp8(x, 32), act_quant_qdq_fp8_triton),
        (32, lambda x: fp4_qdq(x, 32, "e8m0"), lambda x: fp4_qdq_triton(x, 32)),
        (
            16,
            lambda x: fp4_qdq(x, 16, "e4m3"),
            lambda x: fp4_qdq_triton(x, 16, "e4m3"),
        ),
    ):
        pos, cpos = _pos(block, which)
        x = _sweep(block, pos, companion, cpos)
        _assert_bits_equal(tfn(x.cuda()).cpu(), fn(x))


@_cuda
def test_triton_random_rows() -> None:
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(257, 8192, generator=g) * 3).to(torch.bfloat16)
    x[5] = 0
    xc = x.cuda()
    _assert_bits_equal(act_quant_qdq_fp8_triton(xc).cpu(), act_quant_qdq_fp8(x))
    _assert_bits_equal(fp4_qdq_triton(xc, 32).cpu(), fp4_qdq(x, 32))
    _assert_bits_equal(
        fp4_qdq_triton(xc, 16, "e4m3").cpu(), fp4_qdq(x, 16, "e4m3")
    )
