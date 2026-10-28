# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ple_decode_out against an FP64 reference decoder written from the format.

Row format: one FP16 scale word, then a little-endian ring of dim * K bits
(K bits per channel). Channel c's 16-bit trellis state takes bit m from ring
bit ((c - m // K) mod dim) * K + m % K. The mul1 codebook maps a state s to
FP16((1024 + bytesum(s * 0x83DCD12D mod 2**32)) * k_inv + k_bias) with k_inv
and k_bias the FP16 values 0x1EEE and 0xC931. The output is
codebook * scale + head_bias[head] for owned rows inside a head, else zero.

Set EXL3_PLE_CHECKPOINT to an EXL3 Qwen3.8 checkpoint directory to also
decode real rows.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="ple_decode_out needs CUDA"),
]

_MUL1 = 0x83DCD12D
_K_INV = float(np.array([0x1EEE], np.uint16).view(np.float16)[0])
_K_BIAS = float(np.array([0xC931], np.uint16).view(np.float16)[0])
_NAN_SCALE = 0x7E00

# (K, channels, head vocab sizes): the real Qwen3.8 row layout and synthetic
# layouts with other K and channel counts.
_LAYOUTS = [
    pytest.param(6, 160, (97, 40, 13), id="k6x160"),
    pytest.param(3, 32, (5, 9), id="k3x32"),
    pytest.param(5, 16, (3, 3, 3, 3), id="k5x16"),
]


def _reference(packed, ids, bias, offsets, sizes, owner_lo, owner_hi, bits):
    """Return (FP64 reference, FP32-step emulation of the kernel), [R, dim]."""
    rows, _ = packed.shape
    dim = bias.shape[1]
    words = packed.view(np.uint16).astype(np.uint64)
    ring = ((words[:, 1:, None] >> np.arange(16, dtype=np.uint64)) & 1).reshape(
        rows, -1
    )
    channels = np.arange(dim)
    state = np.zeros((rows, dim), dtype=np.uint64)
    for m in range(16):
        source = ((channels - m // bits) % dim) * bits + m % bits
        state |= ring[:, source] << np.uint64(m)
    product = (state * np.uint64(_MUL1)) & np.uint64(0xFFFFFFFF)
    byte_sum = sum((product >> np.uint64(s)) & np.uint64(255) for s in (0, 8, 16, 24))
    codebook = ((1024 + byte_sum).astype(np.float64) * _K_INV + _K_BIAS).astype(
        np.float16
    )
    scale = packed[:, 0].view(np.float16)
    offsets = np.asarray(offsets, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    head = np.searchsorted(offsets, ids, side="right") - 1
    clipped = np.clip(head, 0, len(offsets) - 1)
    valid = (
        (ids >= owner_lo)
        & (ids < owner_hi)
        & (head >= 0)
        & (ids - offsets[clipped] < sizes[clipped])
    )
    row_bias = bias[clipped]
    reference = codebook.astype(np.float64) * scale.astype(np.float64)[
        :, None
    ] + row_bias.astype(np.float64)
    emulated = (codebook.astype(np.float32) * scale.astype(np.float32)[:, None]) + (
        row_bias.astype(np.float32)
    )
    reference[~valid] = 0.0
    emulated[~valid] = 0.0
    return reference, emulated, valid, codebook, scale, row_bias


def _decode(packed, ids, bias, offsets, sizes, owner_lo, owner_hi, bits):
    from vllm_sm80.kernels.exl3.ops import ple_decode_out

    device = torch.device("cuda")
    out = torch.full(
        (len(ids), bias.shape[1]), 123.0, dtype=torch.bfloat16, device=device
    )
    ple_decode_out(
        torch.from_numpy(packed).to(device),
        torch.tensor(ids, dtype=torch.int64, device=device),
        torch.from_numpy(bias).to(device),
        torch.tensor(offsets, dtype=torch.int64, device=device),
        torch.tensor(sizes, dtype=torch.int64, device=device),
        owner_lo,
        owner_hi,
        bits,
        out,
    )
    torch.cuda.synchronize()
    return out.cpu()


def _check(packed, ids, bias, offsets, sizes, owner_lo, owner_hi, bits):
    out = _decode(packed, ids, bias, offsets, sizes, owner_lo, owner_hi, bits)
    reference, emulated, valid, codebook, scale, row_bias = _reference(
        packed, ids, bias, offsets, sizes, owner_lo, owner_hi, bits
    )
    # Masked rows (not owned, negative, past the table, head padding) are 0.
    masked = ~torch.from_numpy(valid)
    assert torch.equal(out[masked], torch.zeros_like(out[masked]))
    # Bit-exact against FP32 multiply, FP32 add, BF16 round-to-nearest.
    assert torch.equal(out, torch.from_numpy(emulated).to(torch.bfloat16))
    # Within BF16 rounding (unit roundoff 2^-8: 8 significand bits) of the
    # FP32 result, whose two steps add at most 2^-24 each.
    got = out.double().numpy()
    terms = np.abs(codebook.astype(np.float64) * scale[:, None]) + np.abs(row_bias)
    bound = 2.0**-8 * np.abs(reference) + 2.0**-22 * terms + 1e-30
    assert np.all(np.abs(got - reference) <= bound)
    return out, valid


def _synthetic(bits, dim, sizes, seed, extra_ids=()):
    rng = np.random.default_rng(seed)
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    end = sum(sizes)
    rows = end + 5  # head padding rows
    ids = [
        *range(rows),
        -1,
        -(1 << 40),
        rows,
        rows + 1000,
        *rng.integers(0, end, 8).tolist(),
        *extra_ids,
    ]
    words = 1 + dim * bits // 16
    # Random packed words for every position, including masked ones: the
    # decoder must not read them.
    packed = rng.integers(-(1 << 15), 1 << 15, (len(ids), words), dtype=np.int16)
    magnitude = rng.uniform(0.05, 2.0, len(ids)) * rng.choice((-1, 1), len(ids))
    packed[:, 0] = magnitude.astype(np.float16).view(np.int16)
    bias = rng.normal(0, 0.5, (len(sizes), dim)).astype(np.float16)
    return packed, ids, bias, offsets, list(sizes), rows


@pytest.mark.parametrize("bits,dim,sizes", _LAYOUTS)
@pytest.mark.parametrize("owner", ["all", "lower", "upper"])
def test_ple_decode_matches_fp64_reference(bits, dim, sizes, owner):
    packed, ids, bias, offsets, sizes, rows = _synthetic(bits, dim, sizes, seed=bits)
    middle = sizes[0] + 1
    owner_lo, owner_hi = {
        "all": (0, rows),
        "lower": (0, middle),
        "upper": (middle, rows),
    }[owner]
    _, valid = _check(packed, ids, bias, offsets, sizes, owner_lo, owner_hi, bits)
    assert valid.any() and not valid.all()


@pytest.mark.parametrize("bits,dim,sizes", _LAYOUTS)
def test_ple_decode_owner_shards_sum_to_the_full_row(bits, dim, sizes):
    packed, ids, bias, offsets, sizes, rows = _synthetic(bits, dim, sizes, seed=7)
    middle = rows // 2
    full = _decode(packed, ids, bias, offsets, sizes, 0, rows, bits)
    lower = _decode(packed, ids, bias, offsets, sizes, 0, middle, bits)
    upper = _decode(packed, ids, bias, offsets, sizes, middle, rows, bits)
    assert torch.equal(lower + upper, full)


@pytest.mark.parametrize("bits,dim,sizes", _LAYOUTS)
def test_ple_decode_poisoned_scale_gives_nan_only_for_owned_rows(bits, dim, sizes):
    packed, ids, bias, offsets, sizes, rows = _synthetic(bits, dim, sizes, seed=3)
    packed[:, 0] = _NAN_SCALE
    out = _decode(packed, ids, bias, offsets, sizes, 0, sizes[0], bits)
    owned = torch.tensor([0 <= row < sizes[0] for row in ids])
    assert out[owned].isnan().all()
    assert torch.equal(out[~owned], torch.zeros_like(out[~owned]))


def test_ple_decode_rows_beyond_one_grid_pass():
    packed, ids, bias, offsets, sizes, rows = _synthetic(5, 16, (3, 3), seed=11)
    repeat = 70000 // len(ids) + 1
    _check(np.tile(packed, (repeat, 1)), ids * repeat, bias, offsets, sizes, 0, rows, 5)


@pytest.mark.skipif(
    not os.environ.get("EXL3_PLE_CHECKPOINT"),
    reason="set EXL3_PLE_CHECKPOINT to an EXL3 Qwen3.8 checkpoint directory",
)
def test_ple_decode_real_checkpoint_rows():
    from vllm_sm80.kernels.exl3.utils import (
        ple_layout,
        read_headers,
        read_metadata,
    )
    from vllm_sm80.models.qwen4_exp.exl3_ple import Exl3PLERowReader

    path = Path(os.environ["EXL3_PLE_CHECKPOINT"]) / "ngram_embedding.safetensors"
    headers = read_headers(path)
    (key,) = [name for name in headers if name.endswith(".trellis")]
    layout = ple_layout(headers, read_metadata(path), key.removesuffix(".trellis"))
    reader = Exl3PLERowReader(str(path), key, 0, layout.rows)
    try:
        offsets, sizes = layout.head_offsets, layout.head_vocab_sizes
        end = offsets[-1] + sizes[-1]
        rng = np.random.default_rng(0)
        ids = sorted(
            {
                *offsets,
                *(o + s - 1 for o, s in zip(offsets, sizes)),
                *rng.integers(0, end, 2000).tolist(),
                *range(end, layout.rows),
            }
        )
        target = bytearray(len(ids) * layout.row_bytes)
        assert reader.read_rows(ids, memoryview(target)) == []
        packed = np.frombuffer(target, dtype=np.int16).reshape(len(ids), -1).copy()
        bias = np.frombuffer(
            reader.read_tensor(layout.auxiliaries[0]), dtype=np.float16
        ).reshape(layout.heads, layout.dim)
        # Unsharded, then the two halves of a TP=2 vocabulary split.
        middle = layout.rows // 2
        for owner_lo, owner_hi in (
            (0, layout.rows),
            (0, middle),
            (middle, layout.rows),
        ):
            _check(packed, ids, bias, offsets, sizes, owner_lo, owner_hi, layout.bits)
    finally:
        reader.close()
