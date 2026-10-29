# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 native operators against an independent FP64 reference.

The reference decodes the format with NumPy in float64 from the packed words:
the MSB-first bit ring of each 16x16 tile, the encoder lane order, the mcg and
mul1 codebooks (whose values are FP16 numbers by definition) and the
128-wide Sylvester Hadamard transforms. It shares no code with the CUDA path.

Set VLLM_EXL3_TEST_CHECKPOINT to an EXL3 checkpoint directory to also check
tiles of every rate and codebook found in it.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.sm80.kernels.exl3 import ops as exl3_ops

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="EXL3 kernels are built for SM 8.0",
    ),
]

# Every integer rate for both codebooks, and the half-integer mul1 rates.
FORMATS = [(bits_x2, codebook) for codebook in (1, 2) for bits_x2 in range(2, 17, 2)]
FORMATS += [(3, 2), (5, 2), (7, 2)]
CHECKPOINT = os.environ.get("VLLM_EXL3_TEST_CHECKPOINT")


def _lane_layout():
    """Tile (row, column) of each of the 256 values in encoder lane order.

    Lane l holds rows 2q, 2q+1, 2q+8, 2q+9 (q = l % 4) of columns l // 4 and
    l // 4 + 8, rows varying fastest: the tensor-core fragment order.
    """
    rows, columns = np.empty(256, np.int64), np.empty(256, np.int64)
    for lane in range(32):
        q = lane % 4
        for c, column in enumerate((lane // 4, lane // 4 + 8)):
            for r, row in enumerate((2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9)):
                rows[lane * 8 + c * 4 + r] = row
                columns[lane * 8 + c * 4 + r] = column
    return rows, columns


_ROWS, _COLUMNS = _lane_layout()


def _states(trellis: np.ndarray, bits_x2: int) -> np.ndarray:
    """16-bit trellis state of each tile value, in encoder lane order."""
    halves = trellis.astype(np.uint16).astype(np.uint64)
    words = halves[..., 0::2] | (halves[..., 1::2] << np.uint64(16))
    shifts = np.arange(31, -1, -1, dtype=np.uint64)
    ring = ((words[..., None] >> shifts) & np.uint64(1)).reshape(*words.shape[:2], -1)
    # Value i ends after (i + 1) * bits / 2 ring bits; for half-integer rates
    # every second value takes the extra bit. The 16-bit window wraps.
    ends = np.arange(1, 257) * bits_x2 // 2
    window = (ends[:, None] - 16 + np.arange(16)) % ring.shape[-1]
    weights = np.uint64(1) << np.arange(15, -1, -1, dtype=np.uint64)
    return (ring[..., window] * weights).sum(-1)


def _codebook(states: np.ndarray, codebook: int) -> np.ndarray:
    """Exact FP16 codebook value of each state, as float64."""
    states = states.astype(np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    if codebook == 1:  # mcg: sum of the two FP16 halves of a hashed word
        product = (states * np.uint64(0xCBAC1FED)) & mask
        product = (product & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
        low = (product & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
        high = (product >> np.uint64(16)).astype(np.uint16).view(np.float16)
        exact = low.astype(np.float64) + high.astype(np.float64)
    else:  # mul1: byte sum of the hash, affine map evaluated once in FP16
        product = (states * np.uint64(0x83DCD12D)) & mask
        total = sum(
            (product >> np.uint64(shift)) & np.uint64(0xFF) for shift in (0, 8, 16, 24)
        ).astype(np.float64)
        scale = float(np.array(0x1EEE, np.uint16).view(np.float16))
        bias = float(np.array(0xC931, np.uint16).view(np.float16))
        exact = (1024.0 + total) * scale + bias
    return exact.astype(np.float16).astype(np.float64)


def _hadamard() -> np.ndarray:
    index = np.arange(128)
    signs = np.vectorize(lambda v: bin(v).count("1") & 1)(index[:, None] & index)
    return (1.0 - 2.0 * signs) / np.sqrt(128.0)


_H128 = _hadamard()


def _blocks(size: int) -> np.ndarray:
    return np.kron(np.eye(size // 128), _H128)


def core_weight(trellis, bits_x2, codebook) -> np.ndarray:
    """Decoded codebook values, float64 [padded_in, padded_out]."""
    k_tiles, n_tiles = trellis.shape[:2]
    tiles = np.zeros((k_tiles, n_tiles, 16, 16))
    tiles[:, :, _ROWS, _COLUMNS] = _codebook(_states(trellis, bits_x2), codebook)
    return tiles.transpose(0, 2, 1, 3).reshape(k_tiles * 16, n_tiles * 16)


def effective_weight(trellis, suh, svh, bits_x2, codebook) -> np.ndarray:
    """Effective float64 weight W [padded_in, padded_out], so that y = x @ W."""
    core = core_weight(trellis, bits_x2, codebook)
    padded_in, padded_out = core.shape
    left = suh.astype(np.float64)[:, None] * _blocks(padded_in)
    right = _blocks(padded_out) * svh.astype(np.float64)[None, :]
    return left @ core @ right


def _assert_close_to_fp64(actual: torch.Tensor, expected: np.ndarray, name: str):
    actual = actual.double().cpu().numpy()
    error = np.abs(actual - expected)
    scale = float(np.sqrt(np.mean(expected**2)))
    # The BF16 result is within 2^-8 of its FP32 value. FP32 transforms and
    # the K-ordered FMA chain add about 2^-24 * sqrt(K) of the output scale,
    # far inside the 2^-12 allowance; a wrong decode is off by O(scale).
    bound = 2.0**-8 * np.abs(expected) + 2.0**-12 * scale
    worst = int(np.argmax(error - bound))
    assert (error <= bound).all(), (
        f"{name}: |error| {error.flat[worst]:.3g} > bound {bound.flat[worst]:.3g}"
    )
    assert np.sqrt(np.mean(error**2)) <= 2.0**-8 * scale, name


def _random_matrix(rng, padded_in, padded_out, bits_x2):
    trellis = rng.integers(
        -(2**15), 2**15, (padded_in // 16, padded_out // 16, 8 * bits_x2)
    ).astype(np.int16)
    suh = rng.standard_normal(padded_in).astype(np.float16)
    svh = (rng.standard_normal(padded_out) * 0.05).astype(np.float16)
    return trellis, suh, svh


def _linear(trellis, suh, svh, bits_x2, codebook, x, logical_out):
    device = torch.device("cuda")
    rows = x.shape[0]
    out = torch.empty(rows, logical_out, dtype=torch.bfloat16, device=device)
    scratch = torch.empty(
        exl3_ops.linear_workspace_size(rows, len(suh), len(svh)),
        dtype=torch.float32,
        device=device,
    )
    exl3_ops.linear_out(
        x.to(device),
        torch.from_numpy(trellis).to(device),
        torch.from_numpy(suh).to(device),
        torch.from_numpy(svh).to(device),
        bits_x2,
        codebook,
        x.shape[1],
        logical_out,
        out,
        scratch,
    )
    return out


def _expected_linear(x, trellis, suh, svh, bits_x2, codebook, logical_out):
    weight = effective_weight(trellis, suh, svh, bits_x2, codebook)
    return x.double().numpy() @ weight[: x.shape[1], :logical_out]


@pytest.mark.parametrize("bits_x2,codebook", FORMATS)
@pytest.mark.parametrize("rows", [1, 3, 17])
def test_linear_matches_fp64_for_every_rate_and_codebook(bits_x2, codebook, rows):
    rng = np.random.default_rng(1000 * bits_x2 + 10 * codebook + rows)
    # Logical sizes below the padded H128 blocks exercise both crops.
    trellis, suh, svh = _random_matrix(rng, 256, 384, bits_x2)
    logical_in, logical_out = 250, 300
    x = torch.from_numpy(rng.standard_normal((rows, logical_in))).to(torch.bfloat16)
    actual = _linear(trellis, suh, svh, bits_x2, codebook, x, logical_out)
    expected = _expected_linear(x, trellis, suh, svh, bits_x2, codebook, logical_out)
    _assert_close_to_fp64(actual, expected, f"bits_x2={bits_x2} codebook={codebook}")


def _moe_case(rng, experts, hidden, intermediate, formats):
    device = torch.device("cuda")
    matrices, owners = [], []
    for expert in range(experts):
        projections = []
        # Packed matrices are stored at whole H128 blocks; the logical
        # intermediate width is cropped by the operator.
        hpad, ipad = -(-hidden // 128) * 128, -(-intermediate // 128) * 128
        for projection, (k, n) in enumerate(((hpad, ipad), (hpad, ipad), (ipad, hpad))):
            bits_x2, codebook = formats[(expert + projection) % len(formats)]
            trellis, suh, svh = _random_matrix(rng, k, n, bits_x2)
            projections.append((trellis, suh, svh, bits_x2, codebook))
        matrices.append(projections)
    pointers: list[list[list[int]]] = []
    metadata: list[list[list[int]]] = []
    for projections in matrices:
        pointers.append([])
        metadata.append([])
        for trellis, suh, svh, bits_x2, codebook in projections:
            tensors = [torch.from_numpy(v).to(device) for v in (trellis, suh, svh)]
            owners.append(tensors)
            pointers[-1].append([t.data_ptr() for t in tensors])
            metadata[-1].append([len(suh), len(svh), bits_x2, codebook])
    return (
        matrices,
        owners,
        torch.tensor(pointers, dtype=torch.int64, device=device),
        torch.tensor(metadata, dtype=torch.int64, device=device),
    )


def _expected_moe(x, ids, weights, expert_map, matrices, intermediate):
    hidden = x.shape[1]
    x = x.double().numpy()
    out = np.zeros_like(x)
    decoded = {}
    for token in range(x.shape[0]):
        for route in range(ids.shape[1]):
            global_id = int(ids[token, route])
            if not 0 <= global_id < len(expert_map):
                continue
            local = int(expert_map[global_id])
            if not 0 <= local < len(matrices):
                continue
            if local not in decoded:
                decoded[local] = [effective_weight(*m) for m in matrices[local]]
            gate, up, down = decoded[local]
            g = x[token] @ gate[:hidden, :intermediate]
            u = x[token] @ up[:hidden, :intermediate]
            act = g / (1.0 + np.exp(-g)) * u
            out[token] += float(weights[token, route]) * (
                act @ down[:intermediate, :hidden]
            )
    return out


@pytest.mark.parametrize("rows", [1, 5, 33])
def test_moe_matches_fp64_with_expert_map_and_masked_routes(rows):
    rng = np.random.default_rng(rows)
    hidden, intermediate, local_experts, global_experts, topk = 256, 200, 4, 8, 3
    formats = [(10, 2), (7, 2), (6, 1), (16, 2)]
    matrices, owners, pointers, metadata = _moe_case(
        rng, local_experts, hidden, intermediate, formats
    )
    # Global experts 1, 3, 4, 6 are local; the rest belong to another rank.
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32)
    expert_map[torch.tensor([1, 3, 4, 6])] = torch.arange(4, dtype=torch.int32)
    ids = torch.from_numpy(rng.integers(-1, global_experts + 1, (rows, topk)))
    ids[0] = torch.tensor([3, -1, global_experts])  # local, padding, out of range
    weights = torch.from_numpy(rng.uniform(0.05, 0.5, (rows, topk)).astype(np.float32))
    x = torch.from_numpy(rng.standard_normal((rows, hidden))).to(torch.bfloat16)
    device = torch.device("cuda")
    out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
    scratch = torch.empty(
        exl3_ops.moe_workspace_size(rows, hidden, intermediate, topk, local_experts),
        dtype=torch.float32,
        device=device,
    )
    exl3_ops.moe_out(
        x.to(device),
        ids.to(device),
        weights.to(device),
        pointers,
        metadata,
        expert_map.to(device),
        intermediate,
        out,
        scratch,
    )
    expected = _expected_moe(x, ids, weights, expert_map, matrices, intermediate)
    _assert_close_to_fp64(out, expected, f"moe rows={rows}")
    del owners


def _checkpoint_tiles():
    from vllm.sm80.kernels.exl3.utils import Exl3Manifest

    manifest = Exl3Manifest(CHECKPOINT)
    chosen: dict[tuple[int, int], str] = {}
    for prefix, matrix in manifest.matrices.items():
        chosen.setdefault((matrix.bits_x2, matrix.codebook), prefix)
    for (bits_x2, codebook), prefix in sorted(chosen.items()):
        trellis_header = manifest.tensors[f"{prefix}.trellis"]
        trellis = np.memmap(
            trellis_header.path,
            dtype="<i2",
            mode="r",
            offset=trellis_header.offset,
            shape=trellis_header.shape,
        )
        # A 128-aligned block of a packed matrix is itself a packed matrix:
        # the transforms are block diagonal.
        k_tiles, n_tiles = min(trellis.shape[0], 16), min(trellis.shape[1], 24)
        scales = []
        for suffix, count in (("suh", k_tiles * 16), ("svh", n_tiles * 16)):
            header = manifest.tensors[f"{prefix}.{suffix}"]
            scales.append(
                np.memmap(header.path, "<f2", "r", header.offset, header.shape)[:count]
            )
        yield (
            prefix,
            np.array(trellis[:k_tiles, :n_tiles]),
            np.array(scales[0]),
            np.array(scales[1]),
            bits_x2,
            codebook,
        )


@pytest.mark.skipif(
    CHECKPOINT is None or not Path(CHECKPOINT).is_dir(),
    reason="set VLLM_EXL3_TEST_CHECKPOINT to an EXL3 checkpoint directory",
)
def test_linear_matches_fp64_on_checkpoint_tiles():
    tiles = list(_checkpoint_tiles())
    assert tiles
    rng = np.random.default_rng(1006)
    for prefix, trellis, suh, svh, bits_x2, codebook in tiles:
        logical_in, logical_out = len(suh), len(svh)
        x = torch.from_numpy(rng.standard_normal((5, logical_in))).to(torch.bfloat16)
        actual = _linear(trellis, suh, svh, bits_x2, codebook, x, logical_out)
        expected = _expected_linear(
            x, trellis, suh, svh, bits_x2, codebook, logical_out
        )
        _assert_close_to_fp64(actual, expected, prefix)


@pytest.mark.parametrize("prefill", [False, True])
def test_linear_scratch_is_shared_with_capture_and_separate_on_aux(
    monkeypatch, prefill
):
    from vllm.utils.torch_utils import aux_stream
    from vllm.v1.worker import workspace

    exl3_ops.ensure_extension()
    monkeypatch.setattr(exl3_ops.dense_prefill_tc, "active", prefill)
    monkeypatch.setattr(exl3_ops.dense_prefill_tc, "active_layout", (1, 2))
    device = torch.device("cuda")
    monkeypatch.setattr(workspace, "_manager", None)
    workspace.init_workspace_manager(device)
    rng = np.random.default_rng(7)
    trellis, suh, svh = _random_matrix(rng, 256, 384, 10)
    tensors = [torch.from_numpy(v).to(device) for v in (trellis, suh, svh)]
    chunk = 2  # several launches per call
    x = torch.from_numpy(rng.standard_normal((5, 256))).to(torch.bfloat16).to(device)

    def call():
        return torch.ops.vllm.exl3_linear(x, *tensors, 10, 2, 384, chunk)

    eager = call()
    main = exl3_ops._linear_scratch(device)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        assert exl3_ops._linear_scratch(device).data_ptr() == main.data_ptr()
        with torch.cuda.graph(graph, stream=capture_stream):
            captured = call()
    torch.cuda.current_stream().wait_stream(capture_stream)
    main.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(captured, eager)
    with torch.cuda.stream(aux_stream()):
        aux = exl3_ops._linear_scratch(device)
    assert aux.data_ptr() != main.data_ptr()
    assert aux.numel() == main.numel() == exl3_ops.linear_scratch_elements()
    expected = _expected_linear(x.cpu(), trellis, suh, svh, 10, 2, 384)
    _assert_close_to_fp64(eager, expected, "chunked op")
    workspace.reset_workspace_manager()
