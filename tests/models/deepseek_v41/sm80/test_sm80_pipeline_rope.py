# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pipeline's RoPE (``ops/sm80_attn_pipeline._rope_row``) rounds as the
reference ``sm80_attn_core.rope`` does on CUDA: its complex64 multiply is
compiled with fused multiply-adds (``real = fma(re, cos, -(im * sin))``,
``imag = fma(re, sin, im * cos)``). Capture r3's FP64 run found the pipeline's
three-rounding form (two products and their sum each rounded) above V2's mean
bound on 11 of 348 ``query`` cases; an emulation of both forms reproduced the
worst case's 1366x ratio exactly.

CPU: the compiled sm_80 code of the row rotation carries no separately rounded
add. GPU: the rotation equals the reference's bytes."""
import re

import pytest
import torch
import triton
import triton.language as tl

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe


@triton.jit
def _rope_probe(x, pos, freqs, out, D: tl.constexpr, RD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, D)
    y = pipe._rope_row(tl.load(x + t * D + d).to(tl.float32), tl.load(pos + t), freqs, D, RD)
    tl.store(out + t * D + d, y.to(tl.bfloat16))


def test_the_rotation_has_no_separately_rounded_add():
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sig = {"x": "*bf16", "pos": "*i64", "freqs": "*fp32", "out": "*bf16",
           "D": "constexpr", "RD": "constexpr"}
    for d in (512, 128):
        compiled = triton.compile(
            ASTSource(_rope_probe, sig, constexprs={"D": d, "RD": 64}),
            target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
        ptx = compiled.asm["ptx"]
        assert not re.search(r"\badd\.rn\b", ptx), "a product sum rounded on its own"
        assert re.search(r"\bfma\.rn\b", ptx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("d", [512, 128])
def test_the_rotation_is_the_reference_bytes(d):
    g = torch.Generator(device="cuda").manual_seed(3)
    rows = 4096
    x = (torch.randn(rows, d, generator=g, device="cuda") * 3).to(torch.bfloat16)
    pos = torch.randint(0, 262144, (rows,), generator=g, device="cuda")
    pos[:64] = torch.arange(64, device="cuda")
    freqs = core.rope_freqs(64, 65536, 10000.0, 16.0, 32, 1, device="cuda")
    out = torch.empty_like(x)
    _rope_probe[(rows,)](x, pos, freqs, out, d, 64)
    want = core.rope(x, pos, freqs)
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))
