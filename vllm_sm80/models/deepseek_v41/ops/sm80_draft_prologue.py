# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's context-KV prologue on sm_80 as one kernel
(``VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED``).

``DeepseekV41Sm80Attention.sm80_store_context_kv`` writes, for each draft
layer and every ``slots[t] >= 0`` of the first ``n`` rows,
``store_rows(cache, slots, window_kv(rms_norm(kv[:n], kv_norm, eps),
positions, freqs))`` (``sm80_attn_core``; ``kv`` is the wkv part of the
projection): 26 launches per layer. :func:`store_context_kv` writes the same
bytes in one launch, with the torch ops' arithmetic in their order:

* RMSNorm: each square rounded on its own, their sum in the order of torch's
  CUDA mean over a contiguous FP32 row (:func:`torch_cuda_mean_sum`), times
  the mean's factor ``1 / width`` (a power of two, so exact), ``+ eps`` in
  FP32, ``rsqrt`` (the approximate instruction ``rsqrtf`` compiles to), the
  product with the row, then with the weight, rounded to BF16;
* RoPE: ``sm80_attn_pipeline._rope_row`` (``polar`` of the FP32 angle, the
  complex64 multiply as CUDA compiles it), rounded to BF16;
* FP8 (32, UE8M0) qdq: ``sm80_act_quant.qdq_fp8_tile``, the tile
  ``_qdq_kernel`` runs;
* the row is written only where the slot is valid. The torch path also
  rewrites the null block's first row with the value it holds, which leaves
  the same bytes.

The sum order depends on ``n`` through torch's launch configuration
(:func:`mean_block_width`), a launch-time constant. The kernel compiles
without FMA contraction and with libdevice's non-FTZ forms, as nvcc builds
torch. Programs of invalid slots return at once. No host sync and no
data-dependent host branch; the launch captures in CUDA graphs.

Validated configuration (:func:`supports`): a 512-wide KV row with a 64-dim
RoPE tail, BF16 activations and cache, sm_80.
"""

import torch
from triton.language.extra import libdevice

from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import qdq_fp8_tile
from vllm_sm80.models.deepseek_v41.ops.sm80_attn_pipeline import _rope_row
from vllm.triton_utils import tl, triton

HEAD_DIM = 512
ROPE_DIM = 64
# ReduceConfig: at most 512 threads per block for an FP32 reduction; the
# input is loaded as vectors of 4 when the reduced dim is contiguous and at
# least 128 wide.
_MAX_REDUCE_THREADS = 512
_INPUT_VEC = 4
_WARP = 32


def _last_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def mean_block_width(rows: int, width: int) -> int:
    """Threads per row (``block.x``) of torch's CUDA ``mean(-1)`` over a
    contiguous ``[rows, width]`` FP32 tensor: ``ReduceConfig``'s
    ``set_block_dimension`` on ``dim0 = width / 4`` vectors and ``dim1 =
    rows`` outputs. ``width`` is a power of two of at least 128."""
    if rows < 1:
        raise ValueError(f"rows must be positive, got {rows}")
    if width < 128 or width & (width - 1):
        raise ValueError(f"width must be a power of two >= 128, got {width}")
    dim0 = width // _INPUT_VEC
    dim0_pow2 = _last_pow2(dim0) if dim0 < _MAX_REDUCE_THREADS else _MAX_REDUCE_THREADS
    dim1_pow2 = _last_pow2(rows) if rows < _MAX_REDUCE_THREADS else _MAX_REDUCE_THREADS
    block_width = min(dim0_pow2, _WARP)
    block_height = min(dim1_pow2, _MAX_REDUCE_THREADS // block_width)
    return min(dim0_pow2, _MAX_REDUCE_THREADS // block_height)


@triton.jit
def _fold(v, H: tl.constexpr):
    """``v[:H] + v[H:]`` of a ``[2H]`` tile, element by element."""
    lo, hi = tl.split(tl.trans(tl.reshape(v, (2, H))))
    return lo + hi


@triton.jit
def torch_cuda_mean_sum(v, D: tl.constexpr, BW: tl.constexpr):
    """The sum of the ``[D]`` FP32 tile ``v`` in the order of torch's CUDA
    reduction over a contiguous row of ``D`` with ``BW`` lanes
    (:func:`mean_block_width`), as a ``[1]`` tile.

    Lane ``x`` loads the vectors ``x, x + BW, ...`` of 4 values and adds
    element ``i`` of each into accumulator ``i`` (identity 0) in that order;
    its value is ``((acc0 + acc1) + acc2) + acc3``. The lanes then fold in
    halves (``v[x] += v[x + BW / 2]``, ... down to 1): through shared memory
    to 32 lanes, then by warp shuffles of decreasing offset."""
    M: tl.constexpr = 4 * BW
    K: tl.constexpr = D // M
    tl.static_assert(K * M == D)
    tl.static_assert(K <= 4)
    tl.static_assert(BW >= 32)
    tl.static_assert(BW <= 128)
    if K == 1:
        acc = 0.0 + v
    elif K == 2:
        v0, v1 = tl.split(tl.trans(tl.reshape(v, (2, M))))
        acc = (0.0 + v0) + v1
    else:
        # (k1, k0, m) with k = 2 * k1 + k0, moved to (m, k1, k0)
        even, odd = tl.split(tl.permute(tl.reshape(v, (2, 2, M)), (2, 0, 1)))
        v0, v2 = tl.split(even)
        v1, v3 = tl.split(odd)
        acc = (((0.0 + v0) + v1) + v2) + v3
    # (x, i1, i0) with i = 2 * i1 + i0
    even, odd = tl.split(tl.reshape(acc, (BW, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    lane = ((a0 + a1) + a2) + a3
    if BW >= 128:
        lane = _fold(lane, 64)
    if BW >= 64:
        lane = _fold(lane, 32)
    lane = _fold(lane, 16)
    lane = _fold(lane, 8)
    lane = _fold(lane, 4)
    lane = _fold(lane, 2)
    return _fold(lane, 1)


@triton.jit
def rms_norm_row(x, w, eps, D: tl.constexpr, BW: tl.constexpr):
    """``sm80_attn_core.rms_norm`` of the ``[D]`` FP32 row ``x`` (BF16
    values) with the FP32 weight ``w``, as torch computes it on CUDA in a
    call whose row count gives ``BW`` lanes; BF16."""
    # The squares are their own kernel in torch: no contraction into the sum.
    mean = torch_cuda_mean_sum(libdevice.mul_rn(x, x), D, BW) * (1.0 / D)
    inv = libdevice.rsqrt(mean + eps)
    return (w * (x * inv)).to(tl.bfloat16)


@triton.jit
def _draft_prologue_kernel(
    kv_ptr, norm_ptr, positions_ptr, freqs_ptr, cache_ptr, slots_ptr, eps,
    stride_kt: tl.constexpr, stride_kd: tl.constexpr, stride_n: tl.constexpr,
    stride_pos: tl.constexpr, stride_cb: tl.constexpr, stride_ce: tl.constexpr,
    stride_cd: tl.constexpr, stride_slot: tl.constexpr,
    D: tl.constexpr, E: tl.constexpr, RD: tl.constexpr, BW: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + t * stride_slot).to(tl.int64)
    if slot >= 0:
        d = tl.arange(0, D)
        x = tl.load(kv_ptr + t * stride_kt + d * stride_kd).to(tl.float32)
        w = tl.load(norm_ptr + d * stride_n).to(tl.float32)
        kv = rms_norm_row(x, w, eps, D, BW)
        row = _rope_row(kv, tl.load(positions_ptr + t * stride_pos), freqs_ptr, D, RD)
        row = tl.reshape(qdq_fp8_tile(tl.reshape(row, (D // 32, 32))), (D,))
        off = (slot // E) * stride_cb + (slot % E) * stride_ce
        tl.store(cache_ptr + off + d * stride_cd, row.to(tl.bfloat16))


def supports(
    head_dim: int,
    rope_dim: int,
    kv_dtype: torch.dtype,
    capability: tuple[int, int] | None,
) -> str | None:
    """None when a layer has the validated configuration, else why not."""
    if capability is None or capability[0] != 8:
        return f"needs compute capability 8.x, got {capability}"
    if (head_dim, rope_dim) != (HEAD_DIM, ROPE_DIM):
        return (
            f"validated for (head dim, rope dim) = {(HEAD_DIM, ROPE_DIM)}, "
            f"got {(head_dim, rope_dim)}"
        )
    if kv_dtype != torch.bfloat16:
        return f"validated for BF16 caches, got {kv_dtype}"
    return None


def store_context_kv(
    kv: torch.Tensor,
    kv_norm: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """For ``n = min(len(slots), len(kv))`` rows: the window row of
    ``rms_norm(kv[t], kv_norm, eps)`` at ``positions[t]`` written to
    ``cache`` at ``slots[t]`` where ``slots[t] >= 0`` (module docstring).
    ``kv`` is ``[T, D]`` BF16 (any strides), ``cache`` ``[blocks, entries,
    D]`` BF16, ``freqs`` the contiguous FP32 ``[rope_dim / 2]``
    frequencies."""
    n = min(slots.shape[0], kv.shape[0])
    if n == 0:
        return
    d = cache.shape[-1]
    if kv.shape[1] != d or kv_norm.shape != (d,):
        raise ValueError(
            f"rows of {kv.shape[1]} and a weight of {tuple(kv_norm.shape)} "
            f"for cache rows of {d}"
        )
    if kv.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError(
            f"needs BF16 rows and cache, got {kv.dtype} and {cache.dtype}"
        )
    if freqs.dtype != torch.float32 or not freqs.is_contiguous():
        raise ValueError("needs contiguous FP32 RoPE frequencies")
    _draft_prologue_kernel[(n,)](
        kv, kv_norm, positions, freqs, cache, slots, eps,
        kv.stride(0), kv.stride(1), kv_norm.stride(0), positions.stride(0),
        cache.stride(0), cache.stride(1), cache.stride(2), slots.stride(0),
        D=d, E=cache.shape[1], RD=2 * freqs.numel(), BW=mean_block_width(n, d),
        num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False,
    )
