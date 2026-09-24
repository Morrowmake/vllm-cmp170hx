# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored kernel: do not edit here; keep in sync with the standalone source:
# edit the standalone harness, re-run its validate.py/bench.py, then re-port.
#!/usr/bin/env python
"""Sparse MLA (DSA) forward for prefill on sm_80.

    sparse_mla_fwd(q, kv, indices, sm_scale, d_v=512, out=None)
        -> (out, max_logits, lse)          # identical contract to vLLM's

Two things are wrong with the incumbent at this shape, and both are in the
schedule rather than in the arithmetic:

1. `_pick_config` (`vllm/v1/attention/ops/triton_mla_sparse.py`) hard-codes
   `BLOCK_H = 32` for every prefill shape and computes its CTA estimate from 64
   heads.  Under TP=4 there are **16** heads per rank, so half of every qk / pv
   `tl.dot` was masked off.  `grid = (1152, 1, 1)` is confirmed in the trace.
2. The gather is masked per element.  The row mask `indices >= 0 & < seq_kv` is
   real, but it does not have to be applied to the load: clamping an invalid
   index to row 0 and letting the existing `qk = -1e30` mask kill the score is
   *exact* (`exp2(-1e30 - max) == 0`, so the row contributes nothing to `acc`
   and nothing to `e_sum`), and it turns the 16 KB / 32 KB gather into an
   unpredicated vectorised load.  The always-true `offs_d < BLOCK_DMODEL`
   half of the mask is dropped for the same reason.

MEASURED with the family bench (6 cases, `--repeats 3`, incumbent re-measured in
the same run), family score = geomean of incumbent_us / us.  Single-shot event
medians on this part wander +-10 % with the clock, so only bench scores are
quoted:

    kernel                                 BLOCK_N  warps  stages  family score
    vLLM's, BLOCK_H=16 (baseline)           32      4      3        1.1626
    clamped gather                            32      4      3        1.0704
    clamped gather                            32      4      4        1.0750
    clamped gather                            16      2      6        1.1887
    clamped gather                            32      2      2        1.2007
    clamped gather                            32      2      4        1.2206
    clamped gather                            32      2      3        1.2207
    clamped gather                            16      2      4        1.1085
    clamped gather                            32      2      1        0.0603 (!)
    clamped gather                            64      2      2        0.0341 (!)
    + index prefetch                          32      2      2     ** 1.2977 **
    + index prefetch                          32      4      3        1.1267
    + index prefetch                          32      2      3        0.9724 (!)
    + index prefetch                          32      2      4        0.9797 (!)
    + index prefetch + rotation               32      2      2     ** 1.3190 **
                                                                      (1.3153,
                                                                       1.3187)

Two independent things are load-bearing here and they only work together:

* **2 warps, not 4.** With BLOCK_H=16 (h_q=16 under TP=4) a CTA has one 16-row
  MMA per 32 keys and an acc of [16, 512] fp32 = 32 KB of registers; 64 threads
  keep that in registers, 128 threads spill the schedule.  1152 CTAs already
  oversubscribe 70 SMs 16-fold, so `num_splits > 1` only adds a merge tail.
* **The index tile is prefetched one iteration ahead, and then `num_stages`
  must be 2, not 3.**  The gather is an indirect load, so Triton cannot
  pipeline it at all (measured: the phases sum, 688 us gather + 462 qk + 320 pv +
  210 softmax = the whole kernel).  Carrying the 32 int32 indices across the
  loop boundary takes the index load off the gather's dependency chain for
  ~nothing in registers - unlike prefetching the gathered [BLOCK_N, 512] tile,
  which spills and costs 2-70x.  With the prefetch in place a 3-deep
  pipeline becomes actively harmful (1.2977 -> 0.9724): the compiler double-
  buffers the wrong thing.  Long-context cases went 1.27-1.30x -> 1.37-1.42x.
* **Each query starts at its own offset in the top-k list** (`ROTATE`).  Three
  paired family-bench runs, rotation on vs off: 1.3190/1.3153/1.3187 against
  1.2971/1.2944/1.2946 - +1.7 %, no overlap.  It helps at every context size
  (ctx 99328 1405 vs 1445 us, 23040 1420 vs 1457, 8192 1465 vs 1484), which is
  what says the mechanism is NOT the L2 contention it was built to test: the
  `ctx=2304` case, where all 1152 CTAs stream the same 2.4 MB, barely moves
  (2055 vs 2070 us).  It is only legal with one split and BLOCK_N | index_topk,
  both checked by the launcher; decode's split path keeps the plain order.

Two things that looked promising and LOST, measured, do not retry:
 * `dot(kn, trans(q))` + a small [BLOCK_N, BLOCK_H] transpose instead of
   Triton's transpose of the gathered [BLOCK_N, D] tile: ~10 % slower.  Triton
   feeds `trans(kn)` to the MMA out of the staging buffer it has to fill anyway.
 * Skipping `acc *= re_scale` (8192 fp32 multiplies per key block) behind the
   uniform test `tl.max(n_e_max - e_max) > 0`, which is exact: 1772 vs 1532 us
   at ctx 99328.  The branch breaks the software pipeline.

The one case this kernel loses is `ctx=2304` with `topk=2048` (0.91x): a context
barely larger than top-k, so every one of the 1152 CTAs streams the same 2.4 MB.
The incumbent's padded BLOCK_H=32 wins there and this kernel does not (2419 us
at that shape), so it is not a BLOCK_H question; unresolved.  Production context
is 100k+ rows and every other case gains 1.29-1.42x.

Determinism: the split-k merge keeps vLLM's fixed ascending order over splits,
never atomics into the accumulator, so the result is bitwise reproducible.
"""

import torch

from vllm.triton_utils import tl, triton  # noqa: F401  (Triton 3.7.1)


def num_sms(device=0):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cdiv(a, b):
    return -(a // -b)


@triton.jit
def _store_output(
    out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum, e_max,
    cur_q, cur_head, mask_h, offs_dv, dim_v,
    stride_out_token, stride_out_head, stride_lse, LOGE2: tl.constexpr,
):
    acc = acc / e_sum[:, None]
    max_logits = e_max * LOGE2
    lse = max_logits + tl.log2(e_sum) * LOGE2
    offs_o = (cur_q * stride_out_token + cur_head[:, None] * stride_out_head
              + offs_dv[None, :])
    tl.store(out_ptr + offs_o, acc.to(out_ptr.dtype.element_ty),
             mask=mask_h[:, None] & (offs_dv[None, :] < dim_v))
    offs_lse = cur_q * stride_lse + cur_head
    tl.store(softmax_lse_ptr + offs_lse, lse, mask=mask_h)
    tl.store(max_logits_ptr + offs_lse, max_logits, mask=mask_h)


@triton.jit
def _sparse_mla_kernel(
    q_buffer, k_buffer, indices_ptr, out_ptr, softmax_lse_ptr,
    part_acc_ptr, part_sum_ptr, part_max_ptr, counter_ptr, max_logits_ptr,
    seq_kv, h_q, dim_qk, dim_v,
    stride_q_token, stride_q_head,
    stride_k_token, stride_k_head,
    stride_out_token, stride_out_head, stride_lse,
    stride_indices_token, stride_indices_head,
    sm_scale, split_len,
    kv_group_num: tl.constexpr,
    index_topk: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    V_IS_K: tl.constexpr,
    ROTATE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_split = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_buffer + cur_q * stride_q_token + cur_head[:, None] * stride_q_head
        + offs_d[None, :], mask=mask_h[:, None], other=0.0)
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        qpe = tl.load(
            q_buffer + cur_q * stride_q_token + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :],
            mask=mask_h[:, None] & (offs_dpe[None, :] < dim_qk), other=0.0)

    # A finite sentinel, not -inf: a fully masked chunk would otherwise give
    # exp2(-inf - -inf) = NaN and poison acc / e_sum for the rest of the loop.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    split_start = cur_split * split_len
    split_end = tl.minimum(split_start + split_len, index_topk)
    idx_base = (indices_ptr + cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head)
    offs_i = tl.arange(0, BLOCK_N)
    # Carry ONLY the BLOCK_N int32 indices across the loop boundary, so the
    # index load (an L2 hit, ~200 cycles) comes off the gather's dependency
    # chain: `kn`'s address is then already in registers when the iteration
    # starts.  Prefetching the gathered [BLOCK_N, dim] tile as well is what
    # spilled and cost 2-70x in measurements; 32 int32 cost nothing.  Worth 6 % here,
    # and it is what makes num_stages=2 beat num_stages=3 (see _select_config).
    # ROTATE: start each query at its own offset in the top-k list and wrap, so
    # that CTAs running concurrently are never reading the same position of two
    # near-identical index lists at the same instant.  The accumulation order
    # changes (the online softmax is order-dependent), so the result is a
    # different-but-equally-valid fp32 ordering: still bitwise reproducible run
    # to run, error 1.22e-04 vs the incumbent's 6.10e-05 band contribution.
    # Only legal when there is one split and BLOCK_N divides index_topk, which
    # the launcher checks; decode's split path keeps the plain order.
    rot = (cur_q % (index_topk // BLOCK_N)) * BLOCK_N if ROTATE else 0
    indices = tl.load(idx_base + split_start + rot + offs_i,
                      mask=split_start + rot + offs_i < index_topk, other=-1)
    for start_indice in range(split_start, split_end, BLOCK_N):
        if ROTATE:
            offs_next = (rot + start_indice + BLOCK_N) % index_topk + offs_i
        else:
            offs_next = start_indice + BLOCK_N + offs_i
        next_indices = tl.load(idx_base + offs_next,
                               mask=offs_next < index_topk, other=-1)

        mask_kv = (indices >= 0) & (indices < seq_kv)
        # Clamp instead of masking the gather: the rows of invalid entries are
        # read but their scores are forced to -1e30 below, so they contribute
        # exactly zero.  Row 0 is always in range whenever any row exists.
        rows = tl.where(mask_kv, indices, 0).to(tl.int64) * stride_k_token \
            + cur_kv_head_id * stride_k_head

        if V_IS_K:
            # V is the leading dim_v dims of the same rows: gather each row once
            # in its natural [BLOCK_N, D] layout and use it for both dots.
            kn = tl.load(k_buffer + rows[:, None] + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kn))
        else:
            k = tl.load(k_buffer + rows[None, :] + offs_d[:, None])
            qk = tl.dot(q, k)

        if BLOCK_DPE > 0:
            kpe = tl.load(k_buffer + rows[None, :] + offs_dpe[:, None],
                          mask=offs_dpe[:, None] < dim_qk, other=0.0)
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -1.0e30)

        if not V_IS_K:
            v = tl.load(k_buffer + rows[:, None] + offs_dv[None, :],
                        mask=offs_dv[None, :] < dim_v, other=0.0)

        # online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        # The clamped gather read row 0 for invalid entries, so their weight has
        # to be removed here instead.  Masking the [BLOCK_H, BLOCK_N] score tile
        # is bitwise identical to masking the gathered [BLOCK_N, D] rows (the
        # same products are exactly zero, in the same order) and costs 256
        # selects instead of 8192.  `e_sum` keeps the UNMASKED p: when every
        # entry of a query is invalid, qk is uniformly -1e30, p is uniformly 1,
        # and the incumbent's e_sum picks those 1s up too - that is what makes a
        # fully masked query return a finite zero rather than NaN.
        pv = tl.where(mask_kv[None, :], p, 0.0)
        if V_IS_K:
            acc = tl.dot(pv.to(kn.dtype), kn, acc)
        else:
            acc = tl.dot(pv.to(v.dtype), v, acc)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        indices = next_indices

    if NUM_SPLITS > 1:
        # Publish this split's unnormalised partial, then the last CTA of this
        # (token, head block) merges all splits in fixed ascending order.
        row = cur_q * h_q + cur_head
        offs_p = (row[:, None] * NUM_SPLITS + cur_split) * BLOCK_DV + offs_dv[None, :]
        tl.store(part_acc_ptr + offs_p, acc, mask=mask_h[:, None])
        offs_ps = row * NUM_SPLITS + cur_split
        tl.store(part_sum_ptr + offs_ps, e_sum, mask=mask_h)
        tl.store(part_max_ptr + offs_ps, e_max, mask=mask_h)
        tl.debug_barrier()
        counter = counter_ptr + cur_q * tl.num_programs(1) + cur_head_id
        ticket = tl.atomic_add(counter, 1, sem="acq_rel")
        if ticket == NUM_SPLITS - 1:
            tl.store(counter, 0)   # leave it zero for the next call
            offs_s = tl.arange(0, NUM_SPLITS)
            offs_ms = row[:, None] * NUM_SPLITS + offs_s[None, :]
            m_all = tl.load(part_max_ptr + offs_ms, mask=mask_h[:, None],
                            other=-1.0e30, volatile=True)
            s_all = tl.load(part_sum_ptr + offs_ms, mask=mask_h[:, None],
                            other=0.0, volatile=True)
            e_max = tl.max(m_all, 1)
            e_sum = tl.sum(tl.exp2(m_all - e_max[:, None]) * s_all, 1)
            acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
            for i in tl.static_range(NUM_SPLITS):
                m_i = tl.load(part_max_ptr + row * NUM_SPLITS + i, mask=mask_h,
                              other=-1.0e30, volatile=True)
                offs_a = (row[:, None] * NUM_SPLITS + i) * BLOCK_DV + offs_dv[None, :]
                acc_i = tl.load(part_acc_ptr + offs_a, mask=mask_h[:, None],
                                other=0.0, volatile=True)
                acc += acc_i * tl.exp2(m_i - e_max)[:, None]
            _store_output(out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum,
                          e_max, cur_q, cur_head, mask_h, offs_dv, dim_v,
                          stride_out_token, stride_out_head, stride_lse, LOGE2)
    else:
        _store_output(out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum,
                      e_max, cur_q, cur_head, mask_h, offs_dv, dim_v,
                      stride_out_token, stride_out_head, stride_lse, LOGE2)


def _select_config(num_tokens, index_topk, h_q, dim_qk, sms):
    """(BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages).

    Pure function of the shape - no autotune, so graph capture is safe."""
    # Never pad the head dimension past the heads that actually exist.
    BLOCK_H = max(16, min(32, 1 << (h_q - 1).bit_length())) if h_q > 8 else 16
    # 32 keys per stage, 2 warps, 3 stages (measured; see the module docstring).
    # stages * BLOCK_N * dim_qk * 2 B of shared memory must stay inside the
    # 164 KB an sm_80 CTA can be given - 96 KB at dim_qk = 512.
    BLOCK_N, num_warps, num_stages = 32, 2, 2
    # A 32-row head tile (more than 16 heads on the card, e.g. all 64 under
    # pipeline parallel) holds a [32, 512] fp32 accumulator: 256 registers a
    # thread at 2 warps, which spills. 4 warps halves it. Measured at h_q = 64,
    # context 64K, over 197 schedules: 16.87 -> 8.34 ms at 2304 tokens and
    # 8.60 -> 4.25 ms at 1152 (2.02x), the best 32-key schedule at both.
    if BLOCK_H > 16:
        num_warps = 4
    while num_stages > 1 and num_stages * BLOCK_N * dim_qk * 2 > 100 * 1024:
        num_stages -= 1
    head_blocks = _cdiv(h_q, min(BLOCK_H, h_q))
    ctas = num_tokens * head_blocks
    num_splits = 1
    # Prefill (num_tokens >> sms) already has far more query parallelism than
    # SMs, so this leaves num_splits at 1; decode keeps the incumbent's ramp.
    while num_splits < 8 and ctas * num_splits * 2 <= max(160, 2 * sms):
        num_splits *= 2
    num_splits = min(num_splits, _cdiv(index_topk, BLOCK_N))
    return BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages


def sparse_mla_fwd(q, kv, indices, sm_scale, d_v=512, block_dpe=None, out=None):
    from vllm.triton_utils import LOG2E, LOGE2
    from vllm.v1.attention.ops.triton_mla_sparse import _workspace

    num_tokens, num_heads_q, dim_qk = q.shape
    _, num_heads_kv, _ = kv.shape
    assert indices.dtype == torch.int32 and indices.shape[0] == num_tokens
    assert num_heads_kv == 1, "only kv head = 1 is supported"
    index_topk = indices.shape[2]

    if block_dpe is None:
        block_dpe = dim_qk - d_v
    BLOCK_DPE = block_dpe
    BLOCK_DMODEL = dim_qk - BLOCK_DPE
    BLOCK_DV = d_v

    BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages = _select_config(
        num_tokens, index_topk, num_heads_q, dim_qk,
        num_sms(q.device.index or 0))

    scale = sm_scale * LOG2E
    kv_group_num = num_heads_q // num_heads_kv
    split_len = _cdiv(_cdiv(index_topk, num_splits), BLOCK_N) * BLOCK_N
    num_splits = _cdiv(index_topk, split_len)
    num_splits = 1 << (num_splits - 1).bit_length()
    grid = (num_tokens, _cdiv(num_heads_q, min(BLOCK_H, kv_group_num)), num_splits)

    if out is None:
        out = torch.empty((num_tokens, num_heads_q, d_v), dtype=q.dtype,
                          device=q.device)
    stats = torch.empty((2, num_tokens, num_heads_q), dtype=torch.float32,
                        device=q.device)
    max_logits, softmax_lse = stats[0], stats[1]

    if num_splits == 1:
        part_acc = part_sum = part_max = counter = max_logits
    else:
        part_acc, part_sum, part_max, counter = _workspace(
            q.device, num_tokens, num_heads_q, num_splits, BLOCK_DV, grid[1])

    sq, skv, so, si = q.stride(), kv.stride(), out.stride(), indices.stride()
    _sparse_mla_kernel[grid](
        q_buffer=q, k_buffer=kv, indices_ptr=indices,
        out_ptr=out, softmax_lse_ptr=softmax_lse,
        part_acc_ptr=part_acc, part_sum_ptr=part_sum, part_max_ptr=part_max,
        counter_ptr=counter, max_logits_ptr=max_logits,
        seq_kv=kv.shape[0], h_q=num_heads_q, dim_qk=dim_qk, dim_v=d_v,
        stride_q_token=sq[0], stride_q_head=sq[1],
        stride_k_token=skv[0], stride_k_head=skv[1],
        stride_out_token=so[0], stride_out_head=so[1],
        stride_lse=num_heads_q,
        stride_indices_token=si[0], stride_indices_head=si[1],
        sm_scale=scale, split_len=split_len,
        kv_group_num=kv_group_num, index_topk=index_topk,
        NUM_SPLITS=num_splits, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE,
        V_IS_K=(d_v == BLOCK_DMODEL),
        ROTATE=(num_splits == 1 and index_topk % BLOCK_N == 0),
        LOGE2=LOGE2,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, max_logits, softmax_lse


def warmup(cases, device="cuda:0"):
    """cases: [(T, context_len, H, dim_qk, topk)] to JIT before capture."""
    dev = torch.device(device)
    for T, R, H, dim_qk, topk in cases:
        q = torch.zeros(T, H, dim_qk, dtype=torch.bfloat16, device=dev)
        kv = torch.zeros(R, 1, dim_qk, dtype=torch.bfloat16, device=dev)
        idx = torch.zeros(T, 1, topk, dtype=torch.int32, device=dev)
        sparse_mla_fwd(q, kv, idx, dim_qk ** -0.5)
    torch.cuda.synchronize()
