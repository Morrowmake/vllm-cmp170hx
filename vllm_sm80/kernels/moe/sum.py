# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The "moesum" decode fold of ``VLLM_AMPERE_DECODE_IDX_GLUE``: the routed
``moe_sum`` fused with the shared-expert add that follows it.

The shared MoE runner and the Marlin experts call this; whether the fold is
on is decided by the idx-glue gate (``use_idx_glue("moesum")``).
"""

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# moesum: routed moe_sum + shared-expert add
# ---------------------------------------------------------------------------
@triton.jit
def _moe_sum_add_kernel(out_ptr, in_ptr, sh_ptr, H, s_tok, s_topk, s_h,
                        sh_s0, sh_s1, TOPK: tl.constexpr, BLOCK_H: tl.constexpr):
    """``out = bf16(f32(bf16(sum_k in[t, k])) + f32(shared[t]))``.

    The sum is ``routing._moe_sum_kernel`` (fp32, slot order) rounded to the
    output dtype as that kernel stores it; the add is the out-of-place bf16
    ``shared + routed`` (fp32 opmath, one rounding).
    """
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h < H
    base = in_ptr + t * s_tok + h * s_h
    acc = tl.zeros([BLOCK_H], tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(base + k * s_topk, mask=mask, other=0.0).to(tl.float32)
    routed = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    sh = tl.load(sh_ptr + t * sh_s0 + h * sh_s1, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * H + h, (sh + routed).to(out_ptr.dtype.element_ty),
             mask=mask)


def moe_sum_add(inp: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """``shared + moe_sum(inp)`` as one kernel; ``inp`` [M, topk, H]."""
    M, topk, H = inp.shape
    assert shared.shape == (M, H) and shared.dtype == inp.dtype
    out = torch.empty((M, H), dtype=inp.dtype, device=inp.device)
    if M == 0:
        return out
    block_h = 256 if H >= 256 else max(16, triton.next_power_of_2(H))
    _moe_sum_add_kernel[(M, triton.cdiv(H, block_h))](
        out, inp, shared, H, inp.stride(0), inp.stride(1), inp.stride(2),
        shared.stride(0), shared.stride(1),
        TOPK=topk, BLOCK_H=block_h, num_warps=4, num_stages=1,
    )
    return out


class _MoeSumDeferral:
    """Single-threaded hand-off between ``MarlinExperts.moe_sum`` and the MoE
    runner's shared+routed add. The runner arms it around its experts call;
    the first ``moe_sum`` inside consumes the arm and leaves its input here
    instead of summing; the runner takes it back and either fuses or flushes.
    """

    def __init__(self) -> None:
        self.armed = False
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def arm(self) -> None:
        self.armed = True
        self.pending = None

    def offer(self, inp: torch.Tensor, out: torch.Tensor) -> bool:
        if not self.armed:
            return False
        self.armed = False
        self.pending = (inp, out)
        return True

    def take(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        self.armed = False
        p, self.pending = self.pending, None
        return p


MOE_SUM_DEFERRAL = _MoeSumDeferral()