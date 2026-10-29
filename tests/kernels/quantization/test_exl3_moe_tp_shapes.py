# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE tensor-core operators (moe_out_v2, moe_out_prefill_v2) at the
tensor-parallel routed-expert shapes: hidden 5120, every one of the 384
experts local, each of 4 ranks holding a 640- or 512-column chunk (5, 5, 4 or
4 H128 blocks) of the 2304-column intermediate, bits_x2 6 or 4.

CPU: the gates open for those shapes with metadata sliced to the rank's chunk
and stay closed for other widths, for whole-expert metadata and for other
expert counts; the native workspace functions accept the shapes (rows 1..64
for moe_out_v2, 65..2048 for moe_out_prefill_v2) and need no more than the
expert-parallel 2304-column, 96-expert configuration.
"""

import pytest
import torch

from vllm_sm80.kernels.exl3 import ops as exl3_ops

HIDDEN, TOPK, EXPERTS = 5120, 6, 384
TP_WIDTHS = (640, 512)
BITS = (6, 4)
EP_INTERMEDIATE, EP_LOCAL = 2304, 96
GATES = (exl3_ops.moe_v2_closed_reason, exl3_ops.moe_prefill_v2_closed_reason)


def _extension_or_skip():
    try:
        exl3_ops.ensure_extension()
    except RuntimeError as exc:
        pytest.skip(f"vllm_sm80._exl3_C unavailable: {exc}")


def _meta(intermediate, bits_x2, local=EXPERTS):
    return [exl3_ops.moe_v2_expert_meta(HIDDEN, intermediate, bits_x2)] * local


def _reason(gate, intermediate, bits_x2, **changes):
    args = dict(
        capability=(8, 0),
        dtype=torch.bfloat16,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        top_k=TOPK,
        local_experts=EXPERTS,
        global_experts=EXPERTS,
        metadata=_meta(intermediate, bits_x2),
    )
    args.update(changes)
    return gate(**args)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("width", TP_WIDTHS)
@pytest.mark.parametrize("bits_x2", BITS)
def test_gate_opens_for_rank_chunks(gate, width, bits_x2):
    assert _meta(width, bits_x2)[0] == [
        [HIDDEN, width, bits_x2, 2],
        [HIDDEN, width, bits_x2, 2],
        [width, HIDDEN, bits_x2, 2],
    ]
    assert _reason(gate, width, bits_x2) is None


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("width", [640])
def test_gate_stays_closed_otherwise(gate, width):
    closed = {
        "width 576": _reason(gate, 576, 6),
        "width 768": _reason(gate, 768, 6),
        "whole-expert metadata": _reason(gate, width, 6, metadata=_meta(2304, 6)),
        "drafter bits_x2 8": _reason(gate, width, 8),
        "expert parallel": _reason(
            gate, width, 6, local_experts=EP_LOCAL, metadata=_meta(width, 6, EP_LOCAL)
        ),
        "drafter experts": _reason(
            gate, width, 6, local_experts=128, global_experts=128,
            metadata=_meta(width, 6, 128),
        ),
    }
    for name, why in closed.items():
        assert why is not None, name


@pytest.mark.parametrize("width", TP_WIDTHS)
def test_decode_workspace(width):
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_v2
    for rows in range(1, 65):
        tp = size(rows, HIDDEN, width, TOPK, EXPERTS)
        assert 0 < tp <= size(rows, HIDDEN, EP_INTERMEDIATE, TOPK, EP_LOCAL), rows
    with pytest.raises(RuntimeError, match="2304, 640 or 512 only"):
        size(1, HIDDEN, 768, TOPK, EXPERTS)


@pytest.mark.parametrize("width", TP_WIDTHS)
def test_prefill_workspace(width):
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_prefill_v2
    for rows in (65, 97, 511, 512, 513, 1024, 2048):
        tp = size(rows, HIDDEN, width, TOPK, EXPERTS)
        assert 0 < tp <= size(rows, HIDDEN, EP_INTERMEDIATE, TOPK, EP_LOCAL), rows
