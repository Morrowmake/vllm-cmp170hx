# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored kernel: do not edit here; keep in sync with the standalone source:
# edit the standalone harness, re-run its validate.py/bench.py, then re-port.
"""Sparse-MLA (DSA) prefill at 64 heads on sm_80 (pipeline parallel, TP=1).

    sparse_mla_prefill(q, kv, indices, sm_scale, d_v=512, out=None)
        -> (out, max_logits, lse)          # the contract of sparse_mla_fwd

    q        [T, 64, 512] bf16          NoPE latent query (block_dpe 0), any row/head strides
    kv       [rows, 1, 512] bf16        flat paged latent cache; V = the whole row (d_v = 512)
    indices  [T, 1, 2176] int32         2048 top-k slots + a 128-slot local section;
                                        -1 or any row outside [0, rows) is masked out
    out      [T, 64, 512] bf16          written in place when given
    max_logits, lse  [T, 64] fp32       natural-log convention

Taken from ``sparse_prefill_mla.sparse_mla_fwd`` when
``VLLM_GLM5_PP_SPARSE_MLA_PREFILL=1`` and :func:`closed` finds nothing
against the call; every other call keeps the Triton kernel there.

Gluon kernel (explicit layouts, mma.sync m16n8k16 bf16 -> fp32, cp.async), one
CTA per query row covering all 64 heads, 8 warps:

- One gather per query: the [32, 512] tile of cache rows goes to shared memory
  once (cp.async, predicated: invalid slots fetch nothing, double-buffered) and
  feeds all 64 heads. The Triton kernel gathers every row twice, once per
  32-head block.
- d-split: warp (s, g), s < NS = 2, g < GH = 4, owns latent dims
  [256 s, 256 s + 256) of heads [16 g, 16 g + 16). QK is split-K over the two
  dim halves (q held in registers); the two fp32 partial score tiles are summed
  through shared memory in a fixed order. P.V is split-N over the same halves,
  so each warp reads only its own half of the tile for both MMAs and keeps a
  [16, 256] fp32 accumulator.
- Softmax is reduce-scattered: each warp sums the partials of its own 8 heads,
  does the online softmax for them (fp32, exp2), and publishes P (rounded to
  bf16 for the P.V MMA, as every tensor-core attention kernel does) and the
  rescale factor through shared memory.
- Index compaction: the prologue scans the row's index list, prefix-sums the
  valid flags and scatters the valid slots (in list order) into a
  shared-memory list padded with -1; the loop walks ceil(n_valid / 32) dense
  blocks of it, so no step is spent on invalid slots (about half of a causal
  first chunk, and the ~98 %-empty local section).
- Register budget (255, maxnreg pinned: ptxas -O3 can otherwise collapse to 32
  registers and a large stack): no index vectors are carried across a step
  (the index list is L1-resident and loaded where used), which pays for two
  independent QK accumulators (one per K half, summed after) and a P.V
  accumulator split into two independent dim halves: twice the MMA chains in
  flight in latency-bound phases.

Measured on one CMP 170HX (70 SMs, CUDA-graph replay over a 4.73M-row paged
cache, L2 rotated, us per call, Triton kernel with the predicated gather ->
this kernel): first2304 8,999 -> 4,261; cont2304 9,339 -> 7,475; cont2304b
9,346 -> 7,489; tail1282 5,359 -> 4,261; call-weighted geomean 1.43x, 363.5 ->
258.4 ms of sparse-MLA per 8.2K-token prompt over the four stages.

Accuracy: against an fp64 recomputation, out error on 33 captured prefill
records is 0.993x mean / 1.000x max of the fp32 reference that rounds P to
bf16 before P.V, and 1.004x / 1.000x of the Triton kernel's. Not bitwise equal
to the Triton kernel (different accumulation order); bitwise reproducible run
to run (no atomics, fixed reduction order).

A row with no valid slot returns out == 0, max_logits = -1e30 ln 2 and a
finite lse (the Triton kernel's lse differs there by ln(2176) on a -6.9e29
value, which fp32 does not resolve). The prefill path returns lse only with
DCP.

Graph safety: the launch allocates only the returned stats (as
sparse_mla_fwd does), no host sync, no scratch; the dispatch is a pure
function of shapes, dtypes and strides. :func:`warmup_from_worker` compiles
the kernel before CUDA-graph capture.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

logger = init_logger(__name__)

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453

NS = 2          # latent-dim slices (split-K of QK, split-N of P.V)
GH = 4          # head groups
HW = 16         # heads per warp (one m16 row tile)
BLOCK_H = GH * HW
BLOCK_N = 32

# The configuration the kernel was validated on (see closed()).
HEADS = 64
DIM = 512
INDEX_WIDTH = 2176      # 2048 top-k + a 128-slot local section
MAX_TOKENS = 2312       # the largest prefill chunk under pipeline parallel
_I32 = 2**31


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _gather(kn_buf, kv_ptr, ib, seq_kv, stride_kv_t, dcol):
    """cp.async the rows of index vector ib into kn_buf [NS, BLOCK_N, DS]; invalid slots skip."""
    vb = (ib >= 0) & (ib < seq_kv)
    rows = gl.where(vb, ib, 0).to(gl.int64) * stride_kv_t
    async_copy.async_copy_global_to_shared(kn_buf, kv_ptr + rows[None, :, None] + dcol,
                                           mask=vb[None, :, None])
    async_copy.commit_group()


@gluon.jit
def _smla_prefill_kernel(
    q_ptr, kv_ptr, idx_ptr, out_ptr, lse_ptr, mx_ptr,
    seq_kv, h_q,
    stride_q_t, stride_q_h, stride_kv_t, stride_o_t, stride_o_h, stride_lse, stride_i_t,
    sm_scale,
    W: gl.constexpr, NS: gl.constexpr, GH: gl.constexpr, HW: gl.constexpr,
    BLOCK_N: gl.constexpr, DS: gl.constexpr, WP: gl.constexpr, LN2: gl.constexpr,
):
    BH: gl.constexpr = GH * HW
    NW: gl.constexpr = NS * GH
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[NS, GH, 1],
                                                  instr_shape=[1, 16, 8])
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
    # [NS, rows, DS] global loads: warp (s, g) covers slice s
    blk: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 32 // (DS // 8), DS // 8], [NS, GH, 1],
                                         [2, 1, 0])
    swz: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[2, 1, 0])
    L2: gl.constexpr = gl.SliceLayout(0, mma)                      # [BH, BLOCK_N] mma rows
    LHm: gl.constexpr = gl.SliceLayout(1, L2)                       # [BH] mma rows
    # [NS, BH, BLOCK_N] partial scores with the NS partials in registers; the NW warps split
    # the BH rows (8 each), a row's BLOCK_N keys over 4 lanes
    red: gl.constexpr = gl.BlockedLayout([NS, 1, BLOCK_N // 4], [1, 8, 4], [1, NW, 1], [2, 1, 0])
    LR: gl.constexpr = gl.SliceLayout(0, red)                       # [BH, BLOCK_N] owned rows
    LRk: gl.constexpr = gl.SliceLayout(0, LR)
    LRh: gl.constexpr = gl.SliceLayout(1, LR)
    LKb: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blk))   # dim 1 of blk
    LSb: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blk))   # dim 0 of blk
    LDb: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blk))   # dim 2 of blk
    pro: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
    flat: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])

    cur_q = gl.program_id(0)
    hb = gl.program_id(1)

    # ---- q -> two register-resident K halves per warp (A operands), staged once via smem
    offs_s = gl.arange(0, NS, layout=LSb)
    offs_d = gl.arange(0, DS, layout=LDb)
    offs_hb = hb * BH + gl.arange(0, BH, layout=LKb)
    mask_hb = (offs_hb < h_q)[None, :, None]
    dcol = offs_s[:, None, None] * DS + offs_d[None, None, :]
    q_blk = gl.load(q_ptr + cur_q * stride_q_t + offs_hb[None, :, None] * stride_q_h + dcol,
                    mask=mask_hb, other=0.0)
    q_tmp = gl.allocate_shared_memory(gl.bfloat16, [NS, BH, DS], swz, q_blk)
    q_a0 = q_tmp.slice(0, DS // 2, 2).load(dot_a)
    q_a1 = q_tmp.slice(DS // 2, DS // 2, 2).load(dot_a)
    q_tmp._keep_alive()

    offs_hr = hb * BH + gl.arange(0, BH, layout=LRh)
    mask_hr = offs_hr < h_q

    # ---- compacted index list: the valid slots of this row, in list order, in smem
    # (positions [n_valid, 32 * n_blocks) hold -1; invalid slots are parked above WP)
    idx_base = idx_ptr + cur_q * stride_i_t
    lin: gl.constexpr = gl.BlockedLayout([WP // (32 * NW)], [32], [NW], [0])
    offs_w = gl.arange(0, WP, layout=lin)
    all_idx = gl.load(idx_base + offs_w, mask=offs_w < W, other=-1)
    vflag = ((all_idx >= 0) & (all_idx < seq_kv)).to(gl.int32)
    incl = gl.associative_scan(vflag, 0, _add)
    n_valid = gl.sum(vflag, 0)
    n_live = (n_valid + BLOCK_N - 1) // BLOCK_N
    cidx = gl.allocate_shared_memory(gl.int32, [2 * WP], flat)
    cidx.slice(0, WP).store(gl.full([WP], -1, gl.int32, lin))
    gl.barrier()
    cidx.scatter(all_idx, gl.where(vflag != 0, incl - 1, WP + offs_w), 0)
    gl.barrier()

    offs_nb = gl.arange(0, BLOCK_N, layout=LKb)
    offs_nr = gl.arange(0, BLOCK_N, layout=LRk)

    kn_smem = gl.allocate_shared_memory(gl.bfloat16, [2, NS, BLOCK_N, DS], swz)
    s_smem = gl.allocate_shared_memory(
        gl.float32, [NS, BH, BLOCK_N],
        gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 4]], [NS, BH, BLOCK_N], [2, 1, 0]))
    p_smem = gl.allocate_shared_memory(
        gl.bfloat16, [BH, BLOCK_N],
        gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BH, BLOCK_N], [1, 0]))
    r_smem = gl.allocate_shared_memory(gl.float32, [BH], flat)

    # block 0 in flight (a row with no valid slot gathers nothing: all -1)
    _gather(kn_smem.index(0), kv_ptr, cidx.gather(offs_nb, 0), seq_kv, stride_kv_t, dcol)

    # finite sentinel
    e_max = gl.full([BH], -1.0e30, gl.float32, LRh)
    e_sum = gl.full([BH], 0.0, gl.float32, LRh)
    # P.V accumulator as two independent dim halves (smaller B operand per MMA call,
    # the rescale of one half can overlap the other's MMAs)
    acc_a = gl.zeros([NS, BH, DS // 2], gl.float32, mma)
    acc_b = gl.zeros([NS, BH, DS // 2], gl.float32, mma)

    for k in range(0, n_live):
        stage = k % 2
        # tile k landed (own copies) and every warp is done with tile k - 1: refill its stage
        async_copy.wait_group(0)
        gl.barrier()
        # index rows are L1-resident (the prologue scanned the whole list): load at issue time
        # next block (past the end: -1s, fetches nothing)
        _gather(kn_smem.index(1 - stage), kv_ptr,
                cidx.gather(gl.minimum(k + 1, n_live) * BLOCK_N + offs_nb, 0),
                seq_kv, stride_kv_t, dcol)

        kn_s = kn_smem.index(stage)
        z = gl.zeros([NS, BH, BLOCK_N], gl.float32, mma)
        s_0 = mma_v2(q_a0, kn_s.slice(0, DS // 2, 2).permute([0, 2, 1]).load(dot_b), z)
        s_1 = mma_v2(q_a1, kn_s.slice(DS // 2, DS // 2, 2).permute([0, 2, 1]).load(dot_b), z)
        s_part = s_0 + s_1
        s_smem.store(s_part)
        ik = cidx.gather(k * BLOCK_N + offs_nr, 0)
        valid = (ik >= 0) & (ik < seq_kv)
        gl.barrier()

        # own rows: sum the NS partials (fixed order), online softmax
        qk = gl.sum(s_smem.load(red), 0) * sm_scale
        qk = gl.where(mask_hr[:, None] & valid[None, :], qk, -1.0e30)
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2(e_max - n_e_max)
        p = gl.exp2(qk - n_e_max[:, None])
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p_smem.store(gl.where(valid[None, :], p, 0.0).to(gl.bfloat16))
        r_smem.store(re_scale)
        gl.barrier()

        re3 = gl.expand_dims(gl.expand_dims(r_smem.load(LHm), 1), 0)
        pa = gl.convert_layout(
            gl.expand_dims(p_smem.load(L2), 0).broadcast_to([NS, BH, BLOCK_N]), dot_a)
        acc_a = mma_v2(pa, kn_s.slice(0, DS // 2, 2).load(dot_b), acc_a * re3)
        acc_b = mma_v2(pa, kn_s.slice(DS // 2, DS // 2, 2).load(dot_b), acc_b * re3)

    async_copy.wait_group(0)
    # a row with no valid slot: acc = 0, e_sum = 0 -> out exactly 0, finite lse
    e_sum = gl.where(e_sum > 0, e_sum, 1.0)
    max_logits = e_max * LN2
    lse = max_logits + gl.log2(e_sum) * LN2
    offs_l = cur_q * stride_lse + offs_hr
    gl.store(lse_ptr + offs_l, lse, mask=mask_hr)
    gl.store(mx_ptr + offs_l, max_logits, mask=mask_hr)
    gl.barrier()
    r_smem.store(e_sum)
    gl.barrier()
    den = gl.expand_dims(gl.expand_dims(r_smem.load(LHm), 1), 0)
    blkh: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 32 // (DS // 16), DS // 16], [NS, GH, 1],
                                          [2, 1, 0])
    offs_sh = gl.arange(0, NS, layout=gl.SliceLayout(1, gl.SliceLayout(2, blkh)))
    offs_dh = gl.arange(0, DS // 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, blkh)))
    offs_hh = hb * BH + gl.arange(0, BH, layout=gl.SliceLayout(0, gl.SliceLayout(2, blkh)))
    o_ptrs = (out_ptr + cur_q * stride_o_t + offs_hh[None, :, None] * stride_o_h
              + offs_sh[:, None, None] * DS + offs_dh[None, None, :])
    mask_hh = (offs_hh < h_q)[None, :, None]
    gl.store(o_ptrs, gl.convert_layout((acc_a / den).to(out_ptr.dtype.element_ty), blkh),
             mask=mask_hh)
    gl.store(o_ptrs + DS // 2, gl.convert_layout((acc_b / den).to(out_ptr.dtype.element_ty), blkh),
             mask=mask_hh)


def sparse_mla_prefill(q, kv, indices, sm_scale, d_v=512, out=None):
    T, H, D = q.shape
    assert kv.dim() == 3 and kv.shape[1] == 1 and kv.shape[2] == D
    assert indices.dtype == torch.int32 and indices.shape[0] == T and indices.shape[1] == 1
    assert d_v == D == 512, "NoPE latent layout only: V is the whole row (d_v == dim_qk == 512)"
    assert kv.stride(2) == 1 and q.stride(2) == 1 and indices.stride(2) == 1
    W = indices.shape[2]
    if out is None:
        out = torch.empty((T, H, d_v), dtype=q.dtype, device=q.device)
    stats = torch.empty((2, T, H), dtype=torch.float32, device=q.device)
    max_logits, lse = stats[0], stats[1]
    if T == 0:
        return out, max_logits, lse
    grid = (T, triton.cdiv(H, BLOCK_H), 1)
    _smla_prefill_kernel[grid](
        q, kv, indices, out, lse, max_logits,
        kv.shape[0], H,
        q.stride(0), q.stride(1), kv.stride(0), out.stride(0), out.stride(1), H,
        indices.stride(0),
        sm_scale * LOG2E,
        W=W, NS=NS, GH=GH, HW=HW, BLOCK_N=BLOCK_N, DS=D // NS,
        WP=max(triton.next_power_of_2(W), 32 * NS * GH), LN2=LOGE2,
        num_warps=NS * GH, maxnreg=255,
    )
    return out, max_logits, lse


# ------------------------------------------------------------------ dispatch
def closed(q, kv, indices, d_v, block_dpe, out):
    """Why this kernel does not apply to the call, or '' when it does.

    Open exactly on the configuration it was validated on: sm_80, all 64
    query heads on the card, one latent kv head, bf16 q/kv/out, the 512-wide
    NoPE layout with V read from the same row (block_dpe 0), int32 indices
    of width 2176, 1..2312 query rows, unit stride on the last dim of every
    tensor and 32-bit row offsets."""
    if torch.cuda.get_device_capability(q.device) != (8, 0):
        return "not sm_80"
    T, H, D = q.shape
    if H != HEADS:
        return f"{H} query heads on this rank (needs {HEADS})"
    if kv.dim() != 3 or kv.shape[1] != 1:
        return f"kv shape {tuple(kv.shape)} (needs [rows, 1, {DIM}])"
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return f"dtype {q.dtype}/{kv.dtype} (needs bfloat16)"
    dpe = D - d_v if block_dpe is None else block_dpe
    if D != DIM or d_v != DIM or dpe != 0 or kv.shape[2] != DIM:
        return f"layout dim_qk {D} block_dpe {dpe} d_v {d_v} (needs {DIM}/0/{DIM})"
    if (indices.dtype != torch.int32 or indices.dim() != 3 or indices.shape[1] != 1
            or indices.shape[2] != INDEX_WIDTH):
        return (f"indices {indices.dtype} {tuple(indices.shape)} "
                f"(needs int32 [T, 1, {INDEX_WIDTH}])")
    if not 1 <= T <= MAX_TOKENS:
        return f"{T} query rows (needs 1..{MAX_TOKENS})"
    if out is not None and (out.dtype != torch.bfloat16 or tuple(out.shape) != (T, H, DIM)
                            or out.stride(2) != 1):
        return f"out {out.dtype} {tuple(out.shape)} stride {out.stride()}"
    if q.stride(2) != 1 or kv.stride(2) != 1 or indices.stride(2) != 1:
        return "a last dim that is not contiguous"
    ostride = out.stride(0) if out is not None else H * DIM
    if (kv.shape[0] >= _I32 or kv.stride(0) >= _I32
            or T * max(q.stride(0), ostride, indices.stride(0)) >= _I32):
        return "offsets beyond 32 bits"
    return ""


def use(q, kv, indices, d_v, block_dpe, out):
    """True when the call goes to this kernel. Logs once why not otherwise."""
    why = closed(q, kv, indices, d_v, block_dpe, out)
    if why:
        logger.info_once(
            "VLLM_GLM5_PP_SPARSE_MLA_PREFILL=1 but the PP sparse-MLA prefill "
            "kernel is off for this call: %s; using the Triton kernel", why)
        return False
    logger.info_once(
        "PP sparse-MLA prefill kernel active (VLLM_GLM5_PP_SPARSE_MLA_PREFILL=1, "
        "%d heads, %d index slots, sm_80)", HEADS, INDEX_WIDTH)
    return True


def warmup(device="cuda:0"):
    """Compile the kernel (one variant: every launch parameter the dispatch
    admits is fixed, and the int arguments keep the specialisation of the
    production cache, whose row count is a multiple of 16)."""
    dev = torch.device(device)
    q = torch.zeros(1, HEADS, DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.zeros(16, 1, DIM, dtype=torch.bfloat16, device=dev)
    idx = torch.full((1, 1, INDEX_WIDTH), -1, dtype=torch.int32, device=dev)
    idx[0, 0, 0] = 0
    sparse_mla_prefill(q, kv, idx, 0.0625)
    torch.cuda.synchronize(dev)


def warmup_from_worker(worker) -> bool:
    """Called before CUDA-graph capture. Compiles the kernel when the flag is
    on and this rank runs 64 heads on sm_80; returns whether it did."""
    from vllm import envs

    if not (envs.VLLM_GLM5_PP_SPARSE_MLA_PREFILL and envs.VLLM_GLM5_PREFILL_KERNELS):
        return False
    device = worker.device
    if torch.cuda.get_device_capability(device) != (8, 0):
        return False
    heads = worker.model_config.get_num_attention_heads(worker.parallel_config)
    if heads != HEADS:
        return False
    warmup(device)
    return True
