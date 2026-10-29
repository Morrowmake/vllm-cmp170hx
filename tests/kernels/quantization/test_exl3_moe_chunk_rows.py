# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_routed_gemm/exl3_chunk_rows (the sm80 tuning table): moe_out runs calls
of more than N rows in row chunks of at most N, with a workspace sized for N
rows.

CPU: the row's default and its check, the launch-row rule
(quant.moe_out_launch_rows), and Exl3Experts.apply's chunks: consecutive row
views of the same tensors covering every row once, each at most N rows, the
same workspace and the other arguments unchanged; calls of at most N rows,
moe_out_v2 calls and moe_out_prefill calls go through whole, with the
original tensors.

GPU (SM 8.0): Exl3Experts.apply with chunks of 512 (and 100) rows, through a
workspace sized for one chunk, gives bit for bit moe_out's output of the whole
call through a whole-call workspace, at hidden 5120 / intermediate 2304 /
top-6 / 96 of 384 experts (bits_x2 6) and hidden 2560 / intermediate 640 /
top-10 / 256 of 512 experts (bits_x2 10), with the SwiGLU clamp at hidden
5120. moe_out has no CPU kernel, so bitwise identity is a GPU check; the CPU
tests check that every row reaches exactly one launch with its own inputs.
"""

import pytest
import torch

from tests.kernels.quantization.test_exl3_moe_v2 import (
    _apply,
    _bitwise,
    needs_sm80,
)
from tests.sm80.tuning_rows import use_rows
from vllm_sm80.kernels.exl3 import ops as exl3_ops
from vllm_sm80.kernels.exl3 import quant

HIDDEN, TOPK = 64, 6


def _chunk_rows(monkeypatch, hidden, rows):
    use_rows(
        monkeypatch, ("moe_routed_gemm", "exl3_chunk_rows", {"hidden": hidden}, rows)
    )


def test_row_defaults_off_and_sets_the_launch_rows(monkeypatch):
    use_rows(monkeypatch)
    assert quant.moe_out_launch_rows(2048, 5120) == 2048
    _chunk_rows(monkeypatch, 5120, 512)
    assert quant.moe_out_launch_rows(2048, 5120) == 512
    assert quant.moe_out_launch_rows(300, 5120) == 300
    # Keyed by the hidden size: another model's layers keep whole calls.
    assert quant.moe_out_launch_rows(2048, 2560) == 2048
    _chunk_rows(monkeypatch, 5120, -1)
    with pytest.raises(ValueError, match="0 or positive"):
        quant.moe_out_launch_rows(2048, 5120)


def _experts(moe_out_rows, max_rows=2048, moe_v2_rows=0, prefill_decode_len=0):
    experts = object.__new__(quant.Exl3Experts)
    experts.intermediate_size = 32
    experts.expert_map = "layer map"
    experts.workspace_elements = 0
    experts.max_rows = max_rows
    experts.moe_out_rows = moe_out_rows
    experts.moe_v2_rows = moe_v2_rows
    experts.moe_v2_bits_x2 = 10
    experts.prefill_decode_len = prefill_decode_len
    experts.prefill_v2_bits_x2 = 0
    experts.swiglu_limit = 7.0
    return experts


def _record(monkeypatch):
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_prefill"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)),
        )
    return calls


def _row_tensors(rows):
    x = torch.randn(rows, HIDDEN).to(torch.bfloat16)
    ids = torch.randint(0, 64, (rows, TOPK), dtype=torch.int32)
    weights = torch.rand(rows, TOPK)
    out = torch.empty_like(x)
    return x, ids, weights, out


@pytest.mark.parametrize("rows", [1, 512, 513, 1024, 1500, 2048])
def test_apply_splits_moe_out_calls_into_row_chunks(monkeypatch, rows):
    calls = _record(monkeypatch)
    experts = _experts(moe_out_rows=512)
    x, ids, weights, out = _row_tensors(rows)
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    _apply(experts, x, ids, weights, "ptrs", "meta", out, workspace2)
    assert {name for name, _, _ in calls} == {"moe_out"}
    if rows <= 512:
        ((_, args, _),) = calls
        assert args[0] is x and args[1] is ids and args[2] is weights
        assert args[7] is out
    start = 0
    for _, args, kwargs in calls:
        part_x, part_ids, part_weights = args[0], args[1], args[2]
        part_out = args[7]
        n = part_x.shape[0]
        assert 0 < n <= 512
        for part, whole in (
            (part_x, x),
            (part_ids, ids),
            (part_weights, weights),
            (part_out, out),
        ):
            # A view of the call's own rows start..start + n.
            assert (
                part.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
            )
            assert part.data_ptr() == whole[start].data_ptr()
            assert part.shape[0] == n and part.is_contiguous()
        assert args[3:7] == ("ptrs", "meta", "layer map", 32)
        assert args[8].data_ptr() == workspace2.data_ptr()
        assert args[8].dtype == torch.float32
        assert kwargs == {"swiglu_limit": 7.0}
        start += n
    assert start == rows
    assert len(calls) == -(-rows // 512)


def test_apply_keeps_v2_and_prefill_calls_whole(monkeypatch):
    from vllm_sm80.kernels.exl3 import step_phase

    calls = _record(monkeypatch)
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    # moe_out_v2 takes calls of up to 64 rows, even above the chunk size.
    experts = _experts(moe_out_rows=16, moe_v2_rows=64)
    x, ids, weights, out = _row_tensors(64)
    _apply(experts, x, ids, weights, "ptrs", "meta", out, workspace2)
    assert [(name, args[0] is x) for name, args, _ in calls] == [("moe_out_v2", True)]
    # moe_out_prefill takes prefill-step calls of 65..3460 rows whole.
    calls.clear()
    monkeypatch.setattr(step_phase, "prefill_step", lambda decode_len: True)
    experts = _experts(moe_out_rows=512, prefill_decode_len=4)
    x, ids, weights, out = _row_tensors(1500)
    _apply(experts, x, ids, weights, "ptrs", "meta", out, workspace2)
    assert [(name, args[0] is x) for name, args, _ in calls] == [
        ("moe_out_prefill", True)
    ]


# ---------------------------------------------------------------- GPU


def _whole_and_chunked(
    experts_set, hidden, intermediate, topk, local, x, ids, w, limit, chunk
):
    """moe_out on the whole call, and Exl3Experts.apply in chunks of `chunk`
    rows through a one-chunk workspace."""
    rows = x.shape[0]
    whole_ws = torch.full(
        (exl3_ops.moe_workspace_size(rows, hidden, intermediate, topk, local),),
        float("nan"),
        device=x.device,
    )
    whole = torch.full_like(x, float("nan"))
    exl3_ops.moe_out(
        x,
        ids,
        w,
        experts_set.pointers,
        experts_set.meta,
        experts_set.expert_map,
        intermediate,
        whole,
        whole_ws,
        swiglu_limit=limit,
    )
    size = exl3_ops.moe_workspace_size(chunk, hidden, intermediate, topk, local)
    assert size < whole_ws.numel() or rows <= chunk
    experts = _experts(moe_out_rows=chunk, max_rows=max(rows, chunk))
    experts.intermediate_size = intermediate
    experts.expert_map = experts_set.expert_map
    experts.swiglu_limit = limit
    # workspace2 is BF16 storage viewed as FP32 (Exl3Experts.workspace_shapes).
    workspace2 = torch.full(
        (2 * size,), float("nan"), dtype=torch.bfloat16, device=x.device
    )
    chunked = torch.full_like(x, float("nan"))
    _apply(
        experts, x, ids, w, experts_set.pointers, experts_set.meta, chunked, workspace2
    )
    return whole, chunked


@needs_sm80
@pytest.mark.parametrize("chunk", [512, 100])
@pytest.mark.parametrize("rows", [513, 1000, 2048])
def test_chunked_moe_out_is_bitwise_the_whole_call_hidden5120(rows, chunk):
    from tests.kernels.quantization import test_exl3_moe_v2_hidden5120 as h5120

    experts_set = _hidden5120_experts()
    x, ids, w = h5120._case("uniform", rows, 3000 + rows, 16.0)
    whole, chunked = _whole_and_chunked(
        experts_set,
        h5120.HIDDEN,
        h5120.INTERMEDIATE,
        h5120.TOPK,
        h5120.LOCAL,
        x,
        ids,
        w,
        h5120.SWIGLU_LIMIT,
        chunk,
    )
    assert torch.isfinite(whole.float()).all()
    _bitwise(chunked, whole, f"hidden 5120 rows={rows} chunk={chunk}")


@needs_sm80
@pytest.mark.parametrize("rows", [513, 2048])
def test_chunked_moe_out_is_bitwise_the_whole_call_hidden2560(served_2560, rows):
    from tests.kernels.quantization import test_exl3_moe_v2 as v2

    x, ids, w = v2._case(served_2560, "uniform", rows, 4000 + rows)
    whole, chunked = _whole_and_chunked(
        served_2560,
        v2.HIDDEN,
        v2.INTERMEDIATE,
        v2.TOPK,
        v2.LOCAL,
        x,
        ids,
        w,
        0.0,
        512,
    )
    assert torch.isfinite(whole.float()).all()
    _bitwise(chunked, whole, f"hidden 2560 rows={rows}")


_CACHE = {}


def _hidden5120_experts():
    from tests.kernels.quantization import test_exl3_moe_v2_hidden5120 as h5120

    if "h5120" not in _CACHE:
        _CACHE["h5120"] = h5120._synthetic(6)
    return _CACHE["h5120"]


@pytest.fixture(scope="module")
def served_2560():
    from tests.kernels.quantization import test_exl3_moe_v2 as v2

    return v2.served_experts(None)
