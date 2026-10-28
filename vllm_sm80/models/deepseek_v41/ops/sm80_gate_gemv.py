# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE gate GEMM for decode-sized calls as one Triton kernel (sm_80).

:func:`gate_gemv` returns the router logits ``x @ weight.T`` for BF16 ``x``
[M, K] (1 to :data:`MAX_ROWS` rows) and the FP32 gate weight [N, K] as FP32
[M, N], in place of ``x.float()``, the FP32 GEMM and its split-K reduce. It
reads ``x`` in BF16 (no FP32 copy) and the weight once. A BF16 weight (the
gate weight held once, in BF16, with VLLM_DEEPSEEK_V41_ROUTER_MMA) is read
as int32 pairs, as ``x`` is, and widened exactly: the logits are the bytes
of the same values held in FP32.

Arithmetic: FP64. A BF16 value times an FP32 value needs at most 8 + 24
significant bits, so every product is exact in FP64; the products are summed
in FP64 on the FP64 tensor cores (``mma.m8n8k4.f64``) and the sum is rounded
once to FP32 (round to nearest even). With ``p_k = x_k * w_k`` the FP64 sum
``S`` is within ``K * 2**-53 * sum(|p_k|)`` of the exact ``sum(p_k)``, so the
logit is the correctly rounded FP32 value of the exact dot product unless the
exact value lies within that distance of a rounding midpoint between two FP32
numbers. The FP32 GEMM it replaces differs from the exact value by its own
summation error, orders of magnitude larger. FP64 costs nothing over FP32
accumulation here: the FP64 tensor cores of this part run at its FP32 FMA rate
(13.0 against 13.2 TFLOP/s), and a decode call is bound by reading the weight
(a 6-row call pads to 8 tokens: 31 MFLOP, 2.4 us of FP64 MMA at peak, against
5.5 us to read 384 x 5120 FP32 weights at 1.42 TB/s).

Shape of the launch: grid (expert blocks of :data:`BLOCK_E`, ``SPLIT`` slices
of K, token blocks of at most 32). Program (e, s, t) multiplies the weight
rows of block e over K slice s with the tokens of block t: experts on the
MMA's M side, tokens on its N side (a token tile of 8, 16 or 32, so a 6-row
call pads to 8, not 16). It writes its FP64 partial to a scratch slab and
counts itself in with an acquire-release atomic on the counter of (e, t); the
program that arrives last adds the ``SPLIT`` partials of its tile in the fixed
order 0 .. SPLIT-1, rounds them to FP32, writes the logits and sets the
counter back to zero. The result does not depend on the order the programs
run in, and the counters are zero between launches (what makes a graph
replay correct). ``SPLIT`` is the largest divisor of ``K / BLOCK_K`` keeping
the expert blocks times ``SPLIT`` within two programs per SM.

The partial scratch and the counters are allocated once per (device, N,
SPLIT) and shared by every gate of that shape: the gates run one after the
other on one stream. The first call on a device must not be inside a CUDA
graph capture (vLLM's capture warms each graph eagerly first). No host
synchronisation.

Above :data:`MAX_ROWS` rows (prefill) the caller keeps the FP32 GEMM. Decode
and verify calls of this model have at most 48 rows; prefill calls have
hundreds to thousands, where the GEMM's tiles reuse each weight across many
rows and reading the weight no longer bounds the call.
"""

import torch

from vllm.triton_utils import tl, triton

__all__ = ["MAX_ROWS", "gate_gemv", "serves", "supports"]

MAX_ROWS = 64
BLOCK_E = 64
BLOCK_K = 64
# Larger token tiles than this run as more token blocks (registers: the
# FP64 accumulator of a 32-token tile is 32 registers per thread).
MAX_BLOCK_T = 32
_PROGRAMS_PER_SM = 2
# The SM count of the sm_80 part this serves; used to plan a launch on a
# device without SM properties (CPU tensors in the Triton interpreter).
_DEFAULT_SMS = 74
_SCRATCH: dict = {}
_SMS: dict = {}
# Gate weight dtypes the kernel reads.
_WEIGHT_DTYPES = (torch.float32, torch.bfloat16)


@triton.jit(do_not_specialize=["M"])
def _gate_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    part_ptr,
    count_ptr,
    M,
    stride_xm,
    stride_wn,
    stride_om,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    W_PAIRS: tl.constexpr,
):
    """``x_ptr``: BF16 x read as int32 pairs (``stride_xm`` in pairs);
    ``w_ptr``: the FP32 weight, or with ``W_PAIRS`` the BF16 weight read as
    int32 pairs (``stride_wn`` in pairs);
    ``part_ptr``: FP64 scratch [SPLIT, token blocks, NPAD, BLOCK_T];
    ``count_ptr``: int32 [token blocks, expert blocks], zero on entry and on
    exit."""
    pid_e = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    KS: tl.constexpr = K // SPLIT
    HALF: tl.constexpr = BLOCK_K // 2
    NB: tl.constexpr = (N + BLOCK_E - 1) // BLOCK_E
    TILE: tl.constexpr = NB * BLOCK_E * BLOCK_T
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_e = offs_e < N
    mask_t = offs_t < M
    # Rows past N or M read weight row 0 or zero tokens; their results are
    # never stored.
    rows_e = tl.where(mask_e, offs_e, 0)
    rows_t = tl.where(mask_t, offs_t, 0)
    k0 = pid_s * KS
    # Even and odd K positions as two halves: w_even[:, j] = w[:, k + 2j],
    # w_odd[:, j] = w[:, k + 2j + 1], and x[k + 2j], x[k + 2j + 1] from one
    # int32 (a BF16 weight's too).
    if W_PAIRS:
        w_ptrs = w_ptr + rows_e[:, None] * stride_wn + (k0 // 2 + tl.arange(0, HALF))[None, :]
    else:
        w_ptrs = w_ptr + rows_e[:, None] * stride_wn + (k0 + 2 * tl.arange(0, HALF))[None, :]
    x_ptrs = x_ptr + rows_t[:, None] * stride_xm + (k0 // 2 + tl.arange(0, HALF))[None, :]
    acc = tl.zeros([BLOCK_E, BLOCK_T], dtype=tl.float64)
    for _ in range(KS // BLOCK_K):
        if W_PAIRS:
            # The FP32 values of the BF16 weights, as x's below: exactly the
            # values an FP32-held weight loads.
            w_pairs = tl.load(w_ptrs)
            w_even = (w_pairs << 16).to(tl.float32, bitcast=True).to(tl.float64)
            w_odd = (w_pairs & -65536).to(tl.float32, bitcast=True).to(tl.float64)
            w_ptrs += HALF
        else:
            w_even = tl.load(w_ptrs).to(tl.float64)
            w_odd = tl.load(w_ptrs + 1).to(tl.float64)
            w_ptrs += BLOCK_K
        pairs = tl.load(x_ptrs, mask=mask_t[:, None], other=0)
        # A pair holds x[2j] in its low and x[2j + 1] in its high half; a BF16
        # value is the high half of its FP32 value. Read as int32 so that no
        # MMA operand comes from a 16-bit load (Triton has no FP64 MMA for one).
        x_even = (pairs << 16).to(tl.float32, bitcast=True).to(tl.float64)
        x_odd = (pairs & -65536).to(tl.float32, bitcast=True).to(tl.float64)
        acc = tl.dot(w_even, tl.trans(x_even), acc, input_precision="ieee",
                     out_dtype=tl.float64)
        acc = tl.dot(w_odd, tl.trans(x_odd), acc, input_precision="ieee",
                     out_dtype=tl.float64)
        x_ptrs += HALF
    tile = (pid_t * NB * BLOCK_E + offs_e[:, None]) * BLOCK_T + tl.arange(0, BLOCK_T)[None, :]
    tl.store(part_ptr + pid_s * TILE * tl.num_programs(2) + tile, acc, cache_modifier=".cg")
    # Every thread's partial is written before one thread's release
    # increment; that thread's acquire is ordered before every thread's
    # loads below by the barrier that shares the atomic's result.
    tl.debug_barrier()
    counter = count_ptr + pid_t * NB + pid_e
    arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT - 1:
        total = tl.zeros([BLOCK_E, BLOCK_T], dtype=tl.float64)
        for s in range(SPLIT):
            total += tl.load(part_ptr + s * TILE * tl.num_programs(2) + tile,
                             cache_modifier=".cg")
        tl.store(out_ptr + offs_t[None, :] * stride_om + offs_e[:, None],
                 total.to(tl.float32), mask=mask_e[:, None] & mask_t[None, :])
        tl.store(counter, 0)


def supports(num_experts: int, hidden: int) -> str | None:
    """None when :func:`gate_gemv` serves a gate weight [num_experts, hidden],
    else the reason."""
    if num_experts <= 0:
        return f"{num_experts} experts"
    if hidden <= 0 or hidden % BLOCK_K:
        return f"hidden size {hidden} is not a multiple of {BLOCK_K}"
    return None


def serves(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`gate_gemv` takes this call: a CUDA BF16 ``x`` of 1 to
    :data:`MAX_ROWS` rows whose rows are whole int32 pairs, and a supported
    contiguous FP32 or BF16 weight. Host metadata only."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and _pairs(x)
        and weight.dtype in _WEIGHT_DTYPES
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.shape[1] == x.shape[1]
        and supports(weight.shape[0], weight.shape[1]) is None
    )


def _pairs(x: torch.Tensor) -> bool:
    """``x`` can be read as int32 pairs (``x.view(torch.int32)``)."""
    return x.stride(1) == 1 and x.stride(0) % 2 == 0 and x.storage_offset() % 2 == 0


def _sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return _DEFAULT_SMS
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SMS[key]


def split_for(num_experts: int, hidden: int, sms: int) -> int:
    """The number of K slices: the largest divisor of ``hidden / BLOCK_K``
    that keeps ``cdiv(num_experts, BLOCK_E) * SPLIT`` within two programs per
    SM."""
    blocks = triton.cdiv(num_experts, BLOCK_E)
    chunks = hidden // BLOCK_K
    limit = max(1, _PROGRAMS_PER_SM * sms // blocks)
    return max(d for d in range(1, chunks + 1) if chunks % d == 0 and d <= limit)


def token_tile(rows: int) -> tuple[int, int]:
    """(token tile, token blocks) of a call of ``rows`` rows."""
    tile = min(MAX_BLOCK_T, max(8, triton.next_power_of_2(rows)))
    return tile, triton.cdiv(rows, tile)


def _scratch(device: torch.device, num_experts: int, split: int):
    """(FP64 partials, zeroed int32 counters), once per (device, N, SPLIT),
    large enough for :data:`MAX_ROWS` rows."""
    key = (str(device), num_experts, split)
    if key not in _SCRATCH:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the gate GEMV's scratch is allocated by its first call, which "
                "must run outside a CUDA graph capture"
            )
        blocks = triton.cdiv(num_experts, BLOCK_E)
        _SCRATCH[key] = (
            torch.empty(split * blocks * BLOCK_E * MAX_ROWS, dtype=torch.float64,
                        device=device),
            torch.zeros(blocks * triton.cdiv(MAX_ROWS, MAX_BLOCK_T),
                        dtype=torch.int32, device=device),
        )
    return _SCRATCH[key]


def gate_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 logits ``x @ weight.T`` [M, N] computed in FP64 (module
    docstring) for BF16 ``x`` [M, K], M at most :data:`MAX_ROWS`, and the
    FP32 or BF16 ``weight`` [N, K]. No host synchronisation (graph-safe)."""
    M, K = x.shape
    N = weight.shape[0]
    assert x.dtype == torch.bfloat16 and weight.dtype in _WEIGHT_DTYPES, (x.dtype, weight.dtype)
    assert weight.shape[1] == K and weight.is_contiguous(), weight.shape
    assert M <= MAX_ROWS and _pairs(x), (x.shape, x.stride(), x.storage_offset())
    reason = supports(N, K)
    assert reason is None, reason
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    if M == 0:
        return out
    split = split_for(N, K, _sm_count(x.device))
    part, count = _scratch(x.device, N, split)
    block_t, token_blocks = token_tile(M)
    w_pairs = weight.dtype == torch.bfloat16
    _gate_gemv_kernel[(triton.cdiv(N, BLOCK_E), split, token_blocks)](
        x.view(torch.int32),
        weight.view(torch.int32) if w_pairs else weight,
        out,
        part,
        count,
        M,
        x.stride(0) // 2,
        weight.stride(0) // 2 if w_pairs else weight.stride(0),
        out.stride(0),
        N=N,
        K=K,
        SPLIT=split,
        BLOCK_T=block_t,
        BLOCK_E=BLOCK_E,
        BLOCK_K=BLOCK_K,
        W_PAIRS=w_pairs,
        num_warps=4,
        num_stages=3,
    )
    return out
