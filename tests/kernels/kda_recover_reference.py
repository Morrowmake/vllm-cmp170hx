# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference of the KDA recover commit
(vllm_sm80/kernels/kda/recover.py)."""

import torch


def commit_reference(states: torch.Tensor, records: torch.Tensor, src: int, n: int,
                     row: int) -> torch.Tensor:
    """State of one layer after replaying `n` tokens of record row `row` over
    states[src]: h = fma(c, k, h * e), fp32 fma emulated in fp64 (the product
    of two fp32 values is exact in fp64)."""
    h = states[src].clone()                          # [H, V, K]
    c, k, e = records[0, row], records[1, row], records[2, row]   # [WS_T, H, D]
    for t in range(n):
        he = h * e[t][:, None, :]
        h = (c[t][:, :, None].double() * k[t][:, None, :].double()
             + he.double()).to(torch.float32)
    return h
