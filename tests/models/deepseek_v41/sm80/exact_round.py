# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference rounding for the all-reduce bits checks: the exact FP64 sum of
the ranks' partials and its correct rounding to BF16 or FP16, decided on
exact FP64 differences (the two neighbours of the sum in the target format,
round to nearest, ties to even)."""

import torch

BF16, FP16 = torch.bfloat16, torch.float16
_INT = {BF16: torch.int16, FP16: torch.int16}


def _step(v: torch.Tensor, up: bool) -> torch.Tensor:
    """The next representable value of v's format above (``up``) or below."""
    i = v.view(_INT[v.dtype]).long()
    width = 8 * v.element_size()
    neg = i < 0
    mag = torch.where(neg, i + (1 << (width - 1)), i)  # magnitude bits
    signed = torch.where(neg, -mag, mag)  # ordered integer: -0 -> 0
    signed = signed + (1 if up else -1)
    sign = signed < 0
    mag = signed.abs()
    i = torch.where(sign, mag - (1 << (width - 1)), mag)
    return i.to(_INT[v.dtype]).view(v.dtype)


def reference_round(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """FP64 ``x`` rounded to nearest, ties to even, in ``dtype``."""
    b = x.to(dtype)
    lo = torch.where(b.double() > x, _step(b, up=False), b)
    hi = _step(lo, up=True)
    d_lo = x - lo.double()
    d_hi = hi.double() - x
    even_lo = (lo.view(_INT[dtype]) & 1) == 0
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & even_lo)
    out = torch.where(pick_lo, lo, hi)
    # a nonzero sum that rounds to zero keeps its sign (IEEE), an exact zero
    # is the FP64 sum's own zero
    signed_zero = torch.where(x < 0, torch.zeros_like(out).neg(), torch.zeros_like(out))
    zero = torch.where(x == 0, x.to(dtype), signed_zero)
    return torch.where(out == 0, zero, out)


def exact_sum(parts) -> torch.Tensor:
    """The FP64 sum of the ranks' partials (exact for these inputs), added
    from the first partial on so that -0 + -0 stays -0 as IEEE says (a
    reduction from a +0 identity would give +0)."""
    acc = parts[0].double()
    for p in parts[1:]:
        acc = acc + p.double()
    return acc
