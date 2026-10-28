# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 row-blocked kernels against the scalar reference kernels, bit for bit.

`linear_out_reference` and `moe_out_reference` are the original scalar
kernels: one sequential FP32 FMA chain per output element with the weight
decoded in place. The row-blocked kernels must reproduce every bit of their
outputs and of the FP32 intermediates both keep, at every block shape (forced
with `set_gemm_shape_override`) and every row count, so a row's result never
depends on the other rows in a launch.

Set VLLM_EXL3_TEST_CHECKPOINT to an EXL3 checkpoint directory to also compare
one real matrix of every rate and codebook found in it.
"""

import json
import os
import struct
from pathlib import Path

import pytest
import torch

from vllm_sm80.kernels.exl3 import ops as exl3_ops

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="EXL3 kernels are built for SM 8.0",
    ),
]

FORMATS = [(bits_x2, codebook) for codebook in (1, 2) for bits_x2 in range(2, 17, 2)]
FORMATS += [(3, 2), (5, 2), (7, 2)]
SHAPES = [-1, 0, 1, 2, 3]  # heuristic, then tiny, small, medium, large
CHECKPOINT = os.environ.get("VLLM_EXL3_TEST_CHECKPOINT")
H128_SCALE = 0.08838834764831845


def _ops():
    exl3_ops.ensure_extension()
    return torch.ops._exl3_C


def _h128(values: torch.Tensor) -> torch.Tensor:
    """The kernels' butterfly in FP32, one IEEE operation at a time:
    distances 1..64, lower = lo + hi, upper = lo - hi, then the scale."""
    shape = values.shape
    v = values.reshape(-1, 128)
    distance = 1
    while distance < 128:
        v = v.reshape(v.shape[0], 128 // (2 * distance), 2, distance)
        lo, hi = v[:, :, 0, :], v[:, :, 1, :]
        v = torch.stack((lo + hi, lo - hi), dim=2)
        distance *= 2
    scale = torch.tensor(H128_SCALE, dtype=torch.float32, device=v.device)
    return (v.reshape(-1, 128) * scale).reshape(shape)


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, name: str):
    view = torch.int16 if actual.element_size() == 2 else torch.int32
    a = actual.contiguous().view(view)
    b = expected.contiguous().view(view)
    mismatch = (a != b).nonzero()
    assert mismatch.numel() == 0, (
        f"{name}: {mismatch.shape[0]} of {a.numel()} elements differ, first at "
        f"{mismatch[0].tolist()}: {actual[tuple(mismatch[0])].item()!r} vs "
        f"{expected[tuple(mismatch[0])].item()!r}"
    )


@pytest.fixture
def shape_override():
    ops = _ops()
    yield ops.set_gemm_shape_override
    ops.set_gemm_shape_override(-1)


def _random_trellis(generator, padded_in, padded_out, bits_x2, device, unaligned=False):
    count = (padded_in // 16) * (padded_out // 16) * 8 * bits_x2
    words = torch.randint(
        -32768, 32768, (count,), generator=generator, dtype=torch.int32
    )
    base = torch.empty(count + 1, dtype=torch.int16, device=device)
    # A storage offset of one halfword leaves the base 2-byte aligned only.
    trellis = (base[1:] if unaligned else base[:count]).view(
        padded_in // 16, padded_out // 16, 8 * bits_x2
    )
    trellis.copy_(words.to(torch.int16).view_as(trellis))
    return trellis


def _random_scales(generator, size, device):
    signs = torch.where(torch.rand(size, generator=generator) < 0.5, -1.0, 1.0)
    return (signs * (0.5 + torch.rand(size, generator=generator))).half().to(device)


def _check_dense(ops, set_shape, x, trellis, suh, svh, bits_x2, codebook, logical_out):
    rows, logical_in = x.shape
    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    size = ops.linear_workspace_size(rows, padded_in, padded_out)
    args = (trellis, suh, svh, bits_x2, codebook, logical_in, logical_out)
    ref_ws = torch.full((size,), float("nan"), device=x.device)
    ref_out = torch.empty(rows, logical_out, dtype=torch.bfloat16, device=x.device)
    ops.linear_out_reference(x, *args, ref_out, ref_ws)
    split = rows * padded_in
    end = split + rows * padded_out
    for shape in SHAPES:
        set_shape(shape)
        ws = torch.full_like(ref_ws, float("nan"))
        out = torch.full_like(ref_out, float("nan"))
        ops.linear_out(x, *args, out, ws)
        name = f"rows={rows} shape={shape}"
        _assert_bitwise(out, ref_out, f"{name} out")
        _assert_bitwise(ws[:split], ref_ws[:split], f"{name} transformed input")
        _assert_bitwise(ws[split:end], ref_ws[split:end], f"{name} core result")


@pytest.mark.parametrize("bits_x2,codebook", FORMATS)
@pytest.mark.parametrize("rows", [1, 3, 17, 130])
def test_dense_matches_reference(bits_x2, codebook, rows, shape_override):
    ops = _ops()
    generator = torch.Generator().manual_seed(1000 * bits_x2 + 10 * codebook + rows)
    device = torch.device("cuda")
    padded_in, padded_out = 384, 512
    trellis = _random_trellis(generator, padded_in, padded_out, bits_x2, device)
    suh = _random_scales(generator, padded_in, device)
    svh = _random_scales(generator, padded_out, device)
    # Logical widths below the padding exercise masked input and output.
    x = torch.randn(rows, 379, generator=generator).to(torch.bfloat16).to(device)
    _check_dense(ops, shape_override, x, trellis, suh, svh, bits_x2, codebook, 509)


@pytest.mark.parametrize("bits_x2,codebook", [(3, 2), (10, 2), (16, 1)])
def test_dense_unaligned_trellis_matches_reference(bits_x2, codebook, shape_override):
    ops = _ops()
    generator = torch.Generator().manual_seed(77 + bits_x2)
    device = torch.device("cuda")
    trellis = _random_trellis(generator, 256, 384, bits_x2, device, unaligned=True)
    assert trellis.data_ptr() % 4 == 2
    suh = _random_scales(generator, 256, device)
    svh = _random_scales(generator, 384, device)
    x = torch.randn(5, 256, generator=generator).to(torch.bfloat16).to(device)
    _check_dense(ops, shape_override, x, trellis, suh, svh, bits_x2, codebook, 384)


def _expert_tables(generator, device, hidden, intermediate, local):
    """Local experts whose nine matrices cycle through every format; experts 2
    and 5 store an extra padded block, expert 3 an unaligned trellis base."""
    hpad = (hidden + 127) // 128 * 128
    ipad = (intermediate + 127) // 128 * 128
    owners, pointers, metadata = [], [], []
    index = 0
    for expert in range(local):
        ptrs, meta = [], []
        for projection in range(3):
            bits_x2, codebook = FORMATS[index % len(FORMATS)]
            index += 1
            padded_in, padded_out = (hpad, ipad) if projection < 2 else (ipad, hpad)
            if expert in (2, 5):
                padded_in, padded_out = padded_in + 128, padded_out + 128
            trellis = _random_trellis(
                generator, padded_in, padded_out, bits_x2, device, unaligned=expert == 3
            )
            suh = _random_scales(generator, padded_in, device)
            svh = _random_scales(generator, padded_out, device)
            owners.append((trellis, suh, svh))
            ptrs.append([trellis.data_ptr(), suh.data_ptr(), svh.data_ptr()])
            meta.append([padded_in, padded_out, bits_x2, codebook])
        pointers.append(ptrs)
        metadata.append(meta)
    return (
        owners,
        torch.tensor(pointers, dtype=torch.int64, device=device),
        torch.tensor(metadata, dtype=torch.int64, device=device),
    )


def _routes(scenario, rows, topk, global_experts, generator):
    if scenario == "empty":  # odd ids are not local here
        pool = torch.tensor([1, 3, 5, -1, global_experts])
        ids = pool[torch.randint(len(pool), (rows, topk), generator=generator)]
    elif scenario == "concentrated":
        ids = torch.zeros(rows, topk, dtype=torch.int64)
    elif scenario == "skewed":
        ids = torch.randint(-1, global_experts + 1, (rows, topk), generator=generator)
        ids[torch.rand(rows, topk, generator=generator) < 0.9] = 6
    else:
        ids = torch.randint(-1, global_experts + 1, (rows, topk), generator=generator)
    weights = torch.rand(rows, topk, generator=generator) + 0.01
    return ids, weights / weights.sum(-1, keepdim=True)


@pytest.mark.parametrize("scenario", ["empty", "concentrated", "skewed", "uniform"])
@pytest.mark.parametrize("rows", [1, 5, 64])
def test_moe_matches_reference(scenario, rows, shape_override):
    ops = _ops()
    generator = torch.Generator().manual_seed(len(scenario) * 100 + rows)
    device = torch.device("cuda")
    hidden, intermediate, topk, global_experts, local = 380, 250, 4, 16, 8
    hpad, ipad = 384, 256
    owners, ptrs, meta = _expert_tables(generator, device, hidden, intermediate, local)
    # Even global experts are local (g // 2); odd ones belong to another rank.
    expert_map = torch.tensor(
        [g // 2 if g % 2 == 0 else -1 for g in range(global_experts)],
        dtype=torch.int32,
        device=device,
    )
    ids, weights = _routes(scenario, rows, topk, global_experts, generator)
    ids, weights = ids.to(device), weights.to(device)
    x = torch.randn(rows, hidden, generator=generator).to(torch.bfloat16).to(device)
    args = (ids, weights, ptrs, meta, expert_map, intermediate)
    size = ops.moe_workspace_size(rows, hidden, intermediate, topk, local)
    ref_ws = torch.full((size,), float("nan"), device=device)
    ref_out = torch.empty_like(x)
    ops.moe_out_reference(x, *args, ref_out, ref_ws)

    flat = ids.flatten()
    valid = (flat >= 0) & (flat < global_experts)
    mapped = torch.where(
        valid, expert_map[flat.clamp(0, global_experts - 1)].long(), -1
    )
    local_rows = (mapped >= 0).nonzero().flatten()
    experts = mapped[local_rows]
    svh_up = torch.stack([owners[3 * e + 1][2][:ipad] for e in range(local)]).float()
    svh_down = torch.stack([owners[3 * e + 2][2][:hpad] for e in range(local)]).float()
    assignments = rows * topk

    def regions(ws):
        """FP32 rows [assignments x hpad | x ipad (gate) | x ipad (up)]."""
        input_, gate, up = ws[: assignments * (hpad + 2 * ipad)].split(
            [assignments * hpad, assignments * ipad, assignments * ipad]
        )
        return (
            input_.view(assignments, hpad),
            gate.view(assignments, ipad),
            up.view(assignments, ipad),
        )

    ref_input, ref_gate, ref_up = regions(ref_ws)
    for shape in SHAPES:
        shape_override(shape)
        ws = torch.full_like(ref_ws, float("nan"))
        out = torch.full_like(ref_out, float("nan"))
        ops.moe_out(x, *args, out, ws)
        name = f"{scenario} rows={rows} shape={shape}"
        _assert_bitwise(out, ref_out, f"{name} out")
        if local_rows.numel() == 0:
            continue
        new_input, new_gate, new_up = regions(ws)
        # Both keep the down input in the gate rows; the new kernel keeps raw
        # up/down core results where the reference keeps transformed outputs.
        _assert_bitwise(
            new_gate[local_rows], ref_gate[local_rows], f"{name} down input"
        )
        _assert_bitwise(
            _h128(new_up[local_rows]) * svh_up[experts],
            ref_up[local_rows],
            f"{name} up",
        )
        _assert_bitwise(
            _h128(new_input[local_rows]) * svh_down[experts],
            ref_input[local_rows],
            f"{name} down",
        )


def _checkpoint_matrices(root):
    """First matrix of every (rate, codebook) in the checkpoint."""
    found = {}
    for path in sorted(Path(root).glob("*.safetensors")):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        for key, item in header.items():
            if key.endswith(".trellis") and len(item["shape"]) == 3:
                prefix = key.removesuffix(".trellis")
                codebook = 2 if prefix + ".mul1" in header else 1
                found.setdefault(
                    (item["shape"][-1] // 8, codebook),
                    (path, 8 + length, prefix, header),
                )
    return found


def _read(path, base, header, key):
    item = header[key]
    begin, end = item["data_offsets"]
    with open(path, "rb") as stream:
        stream.seek(base + begin)
        raw = bytearray(stream.read(end - begin))
    dtype = {"I16": torch.int16, "F16": torch.float16}[item["dtype"]]
    return torch.frombuffer(raw, dtype=dtype).reshape(item["shape"])


@pytest.mark.skipif(CHECKPOINT is None, reason="set VLLM_EXL3_TEST_CHECKPOINT")
@pytest.mark.parametrize("rows", [1, 17])
def test_checkpoint_tiles_match_reference(rows, shape_override):
    ops = _ops()
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(rows)
    for (bits_x2, codebook), (path, base, prefix, header) in _checkpoint_matrices(
        CHECKPOINT
    ).items():
        trellis = _read(path, base, header, prefix + ".trellis")
        suh = _read(path, base, header, prefix + ".suh")
        svh = _read(path, base, header, prefix + ".svh")
        # Whole 128-column output blocks are independent under H128.
        columns = min(trellis.shape[1] * 16, 1024)
        trellis = trellis[:, : columns // 16].contiguous().to(device)
        svh = svh[:columns].contiguous().to(device)
        x = torch.randn(rows, suh.numel(), generator=generator).to(torch.bfloat16)
        _check_dense(
            ops,
            shape_override,
            x.to(device),
            trellis,
            suh.to(device),
            svh,
            bits_x2,
            codebook,
            columns,
        )
