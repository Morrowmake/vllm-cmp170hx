# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 routed experts with the SwiGLU clamp (``swiglu_limit``) on sm_80.

``moe_out`` with a limit is bitwise equal to the scalar oracle
``moe_out_reference`` with the same limit, a limit no activation reaches is
bitwise equal to no limit, and a reaching limit changes the output. Shapes
are the DeepSeek-V4.1 routed experts' (hidden 5120, intermediate 2304,
3 bpw, mul1), on random packed weights.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 0),
        reason="EXL3 kernels run on SM 8.0 GPUs",
    ),
]

HIDDEN, INTERMEDIATE, EXPERTS, TOPK, BITS_X2, MUL1 = 5120, 2304, 8, 6, 6, 2


def _experts(device):
    generator = torch.Generator(device="cpu").manual_seed(0)
    keep, pointers, metadata = [], [], []
    for _ in range(EXPERTS):
        ptrs, meta = [], []
        for k, n in ((HIDDEN, INTERMEDIATE), (HIDDEN, INTERMEDIATE), (INTERMEDIATE, HIDDEN)):
            trellis = torch.randint(
                -(2**15), 2**15, (k // 16, n // 16, 8 * BITS_X2), dtype=torch.int16,
                generator=generator,
            ).to(device)
            suh = (torch.rand(k, generator=generator) * 0.2 + 0.9).half().to(device)
            svh = (torch.rand(n, generator=generator) * 0.2 + 0.9).half().to(device)
            keep += [trellis, suh, svh]
            ptrs.append([trellis.data_ptr(), suh.data_ptr(), svh.data_ptr()])
            meta.append([k, n, BITS_X2, MUL1])
        pointers.append(ptrs)
        metadata.append(meta)
    return (
        keep,
        torch.tensor(pointers, dtype=torch.int64, device=device),
        torch.tensor(metadata, dtype=torch.int64, device=device),
    )


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_moe_clamp_matches_the_oracle(rows):
    from vllm_sm80.kernels.exl3 import ops

    ops.ensure_extension()
    device = torch.device("cuda")
    keep, pointers, metadata = _experts(device)
    generator = torch.Generator(device="cpu").manual_seed(rows)
    x = (torch.randn(rows, HIDDEN, generator=generator) * 4).bfloat16().to(device)
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(rows)]
    ).int().to(device)
    weights = torch.rand(rows, TOPK, generator=generator).to(device)
    expert_map = torch.arange(EXPERTS, dtype=torch.int32, device=device)
    workspace = torch.empty(
        ops.moe_workspace_size(rows, HIDDEN, INTERMEDIATE, TOPK, EXPERTS),
        dtype=torch.float32,
        device=device,
    )
    args = (ids, weights, pointers, metadata, expert_map, INTERMEDIATE)

    def run(limit, operator=ops.moe_out):
        out = torch.empty_like(x)
        operator(x, *args, out, workspace, swiglu_limit=limit)
        return out

    def oracle(limit):
        out = torch.empty_like(x)
        torch.ops._exl3_C.moe_out_reference(x, *args, out, workspace, limit)
        return out

    for limit in (0.0, 10.0, 0.5):
        assert torch.equal(run(limit), oracle(limit)), limit
    assert torch.equal(run(1e30), run(0.0))
    assert not torch.equal(run(0.5), run(0.0))
    del keep
