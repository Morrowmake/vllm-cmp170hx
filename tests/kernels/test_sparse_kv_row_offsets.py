# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the KV row-offset arithmetic of the sparse-MLA / DSA kernels.

A Triton kernel cannot be executed on a CPU, so this file pins the two things
that can be checked without a device:

* the index arithmetic itself, against a pure-Python mirror that reproduces
  32- and 64-bit wraparound, including the cache size at which a 32-bit row
  product first goes negative;
* the presence of the widening cast in the kernel source, so the fix cannot be
  lost to a later edit.

In Triton, ``ptr + row * stride + col`` evaluates ``row * stride`` *before* the
pointer addition promotes anything, so an ``i32 * i32`` row term wraps while
the address arithmetic around it is 64-bit.  The failure is silent: the load
lands on a wrong cache row and attention returns a plausible wrong answer.

The kernels themselves are covered by the GPU test at the bottom, which skips
when no device is visible.

    pytest -q tests/kernels/test_sparse_kv_row_offsets.py
"""

import inspect

import pytest
import torch

from vllm.ampere_prefill import sparse_prefill_mla
from vllm.v1.attention.ops import triton_mla_sparse, triton_mqa_logits

INT32_MAX = 2**31 - 1

# GLM-5.3-Flash as deployed here: kv_lora_rank 512, qk_rope_head_dim 0 (the
# NoPE layout), one KV head, so the flat KV view is [num_rows, 1, 512] and
# kv.stride(0) is 512 bf16 elements.
MLA_ROW_STRIDE_ELEMS = 512
# The DeepSeek 576 layout (kv_lora_rank 512 + qk_rope_head_dim 64) is the
# kernel's other supported shape and is reached through the BLOCK_DPE > 0 arm.
MLA_ROPE_ROW_STRIDE_ELEMS = 576
# The live engine's KV cache at max_model_len 262144, TP4, 0.95 utilisation.
KV_CACHE_TOKENS = 1_160_192


def row_offset(row: int, stride: int, *, bits: int) -> int:
    """``row * stride`` as Triton evaluates it, at the given integer width."""
    modulus = 1 << bits
    product = (row * stride) & (modulus - 1)
    if product >= modulus >> 1:
        product -= modulus
    return product


def first_overflowing_row_count(stride: int) -> int:
    """Smallest KV cache size (in rows) whose last row wraps a 32-bit product."""
    return INT32_MAX // stride + 2


# ------------------------------------------------------------------ mirror


def test_mirror_reproduces_int32_wraparound():
    assert row_offset(1, 512, bits=32) == 512
    assert row_offset(4_194_303, 512, bits=32) == 2_147_483_136
    # One row further and the product is exactly 2**31: it comes back negative.
    assert row_offset(4_194_304, 512, bits=32) == -2_147_483_648
    assert row_offset(4_194_304, 512, bits=64) == 2_147_483_648


def test_int32_row_products_break_at_four_mebirows():
    assert first_overflowing_row_count(MLA_ROW_STRIDE_ELEMS) == 4_194_305
    last_good = first_overflowing_row_count(MLA_ROW_STRIDE_ELEMS) - 2
    assert row_offset(last_good, MLA_ROW_STRIDE_ELEMS, bits=32) > 0
    assert row_offset(last_good + 1, MLA_ROW_STRIDE_ELEMS, bits=32) < 0
    # ... and the widened arithmetic keeps going.
    assert row_offset(last_good + 1, MLA_ROW_STRIDE_ELEMS, bits=64) == (
        (last_good + 1) * MLA_ROW_STRIDE_ELEMS
    )


def test_rope_layout_breaks_earlier():
    """The 576-wide layout has a proportionally lower row ceiling."""
    assert first_overflowing_row_count(MLA_ROPE_ROW_STRIDE_ELEMS) == 3_728_272


def test_deployment_kv_cache_is_below_the_int32_row_limit():
    """Today's cache is ~3.6x under the threshold: this is a latent fix, and
    the margin is what this test exists to watch."""
    last_row = KV_CACHE_TOKENS - 1
    product = last_row * MLA_ROW_STRIDE_ELEMS
    assert product == 594_017_792
    assert product * 2 == 1_188_035_584  # bytes, bf16
    assert row_offset(last_row, MLA_ROW_STRIDE_ELEMS, bits=32) == product
    assert KV_CACHE_TOKENS < first_overflowing_row_count(MLA_ROW_STRIDE_ELEMS)


# ------------------------------------------------------------ source guard


def kernel_source(fn) -> str:
    """Source of a ``triton.jit`` function (JITFunction keeps it on ``.src``)."""
    return getattr(fn, "src", None) or inspect.getsource(fn)


@pytest.mark.parametrize(
    "fn,stride_name",
    [
        (triton_mla_sparse._mla_sparse_kernel, "stride_k_token"),
        (triton_mla_sparse._mla_sparse_kernel, "stride_v_token"),
        (sparse_prefill_mla._sparse_mla_kernel, "stride_k_token"),
    ],
)
def test_every_kv_row_multiply_is_widened(fn, stride_name):
    lines = [
        line
        for line in kernel_source(fn).splitlines()
        if f" * {stride_name}" in line
    ]
    assert lines, f"no {stride_name} multiply found in {fn}"
    for line in lines:
        assert ".to(tl.int64)" in line, line


def test_prefill_mqa_logits_gather_row_is_widened():
    src = kernel_source(triton_mqa_logits._mqa_logits_kernel)
    assert "cols[:, None].to(tl.int64) * D" in src


def test_paged_mqa_logits_page_offset_is_widened():
    """The decode path was already correct; keep it that way."""
    src = kernel_source(triton_mqa_logits._paged_mqa_logits_kernel)
    assert "blk = tl.load(bt_ptr + b * stride_bt + tile).to(tl.int64)" in src
    assert "k_base = blk * PAGE_BYTES" in src


# ------------------------------------------------------------------- GPU


def sparse_mla_reference(q, kv, indices, sm_scale, d_v):
    num_tokens, num_heads, _ = q.shape
    out = torch.zeros(num_tokens, num_heads, d_v, dtype=torch.float32)
    q32 = q.float()
    kv32 = kv[:, 0, :].float()
    for t in range(num_tokens):
        rows = indices[t, 0]
        rows = rows[(rows >= 0) & (rows < kv.shape[0])].long()
        k = kv32[rows]
        logits = (q32[t] @ k.T) * sm_scale
        probs = torch.softmax(logits, dim=-1)
        out[t] = probs @ k[:, :d_v]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_sparse_mla_matches_reference_after_widening():
    torch.manual_seed(0)
    num_tokens, num_heads, dim, rows, topk = 4, 16, 512, 1024, 64
    q = torch.randn(num_tokens, num_heads, dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(rows, 1, dim, dtype=torch.bfloat16, device="cuda")
    idx = torch.stack(
        [torch.randperm(rows)[:topk] for _ in range(num_tokens)]
    ).to(torch.int32).unsqueeze(1).cuda()
    sm_scale = dim**-0.5

    out, _, _ = triton_mla_sparse.triton_mla_sparse_fwd(q, kv, idx, sm_scale, d_v=dim)
    ref = sparse_mla_reference(q.cpu(), kv.cpu(), idx.cpu(), sm_scale, dim)
    torch.testing.assert_close(
        out.float().cpu(), ref, rtol=3e-2, atol=3e-3
    )
