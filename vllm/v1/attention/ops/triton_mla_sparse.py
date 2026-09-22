# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse (top-k indexed) MQA attention for MLA on CUDA.

Adapted from ``vllm/v1/attention/ops/xpu_mla_sparse.py`` so that sparse MLA
(DeepSeek V3.2 DSA and GLM-5.3-Flash) can run on GPUs that have none of the
vendor kernels (FlashMLA / FA3 / FlashInfer-sparse), e.g. Ampere (sm_80).

Layout conventions (shared with the other sparse MLA backends):

* ``q``       -- ``[num_tokens, num_heads_q, dim_qk]`` bf16/fp16, where
  ``dim_qk = kv_lora_rank + qk_rope_head_dim`` (512 for NoPE GLM-5.3-Flash,
  576 for DeepSeek).
* ``kv``      -- ``[num_rows, 1, dim_qk]``, a flat row view of the paged
  latent cache (``flat_kv_row_view``). The first ``d_v`` dims double as V.
* ``indices`` -- ``[num_tokens, 1, topk]`` int32 global cache rows; ``-1``
  (or any out-of-range row) is masked out of the softmax.
"""

import functools

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton


def _cdiv(a: int, b: int) -> int:
    """Ceiling division. ``triton.cdiv`` is a ``ConstexprFunction``: it costs
    ~0.9 us of host time per call, which matters on the decode path."""
    return -(a // -b)


@triton.jit
def _mla_sparse_kernel(
    q_buffer,
    k_buffer,
    v_buffer,
    indices_ptr,
    out_ptr,
    softmax_lse_ptr,
    part_acc_ptr,  # fp32 [T, H, NUM_SPLITS, BLOCK_DV] workspace (NUM_SPLITS > 1)
    part_sum_ptr,  # fp32 [T, H, NUM_SPLITS]
    part_max_ptr,  # fp32 [T, H, NUM_SPLITS]
    counter_ptr,  # int32 [T, head blocks], zero on entry and on exit
    max_logits_ptr,
    seq_kv,
    h_q,
    dim_qk,
    dim_v,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_out_token,
    stride_out_head,
    stride_lse,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    split_len,
    kv_group_num: tl.constexpr,
    index_topk: tl.constexpr,
    NUM_SPLITS: tl.constexpr,  # >1: write unnormalised partials for _mla_merge_kernel
    BLOCK_H: tl.constexpr,  # block size for num heads
    BLOCK_N: tl.constexpr,  # block size for indices
    BLOCK_DV: tl.constexpr,  # block size for dim_v
    BLOCK_DMODEL: tl.constexpr,  # block size for dim_nope
    BLOCK_DPE: tl.constexpr,  # block size for positional embedding
    V_IS_K: tl.constexpr,  # v_buffer rows == k_buffer nope rows (d_v == BLOCK_DMODEL)
    LOGE2: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_split = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < h_q)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    off_q = cur_q * stride_q_token + cur_head[:, None] * stride_q_head + offs_d[None, :]
    mask_dmodel = offs_d < BLOCK_DMODEL
    q = tl.load(
        q_buffer + off_q, mask=(mask_h[:, None]) & (mask_dmodel[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :]
        )
        # dim_qk == BLOCK_DMODEL + BLOCK_DPE (checked by the launcher)
        mask_dpe = offs_dpe < dim_qk
        qpe = tl.load(
            q_buffer + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    # Use a finite sentinel rather than -inf: if every key in a chunk is
    # masked (e.g. leading -1 padding in `indices`) an -inf running max gives
    # re_scale = exp2(-inf - -inf) = NaN, permanently poisoning acc / e_sum.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    split_start = cur_split * split_len
    split_end = tl.minimum(split_start + split_len, index_topk)
    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < index_topk
        indices = tl.load(
            indices_ptr
            + (
                cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice
            ),
            mask=mask_indice,
            other=-1,
        )

        mask_kv = (indices >= 0) & (indices < seq_kv)
        mask_kv_d = mask_dmodel

        if V_IS_K:
            # V is the leading dim_v == BLOCK_DMODEL dims of the same rows: gather
            # each row once in its natural [BLOCK_N, D] layout and reuse it for
            # both dots instead of reading the row twice from L2.
            offs_kn = (
                indices[:, None].to(tl.int64) * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_d[None, :]
            )
            kn = tl.load(
                k_buffer + offs_kn, mask=(mask_kv[:, None]) & (mask_kv_d[None, :]), other=0.0
            ).to(q.dtype)
            qk = tl.dot(q, tl.trans(kn))
        else:
            offs_k = (
                indices[None, :].to(tl.int64) * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_d[:, None]
            )
            # q_nope @ k_nope
            k = tl.load(
                k_buffer + offs_k, mask=(mask_kv[None, :]) & (mask_kv_d[:, None]), other=0.0
            )
            qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            # q_rope @ k_rope
            offs_kpe = (
                indices[None, :].to(tl.int64) * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_dpe[:, None]
            )
            mask_k_dpe = offs_dpe < dim_qk
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=(mask_kv[None, :]) & (mask_k_dpe[:, None]),
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))

        # apply scaling (sm_scale already carries log2(e))
        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -1.0e30)

        if not V_IS_K:
            # load v
            mask_v_d = offs_dv < dim_v
            offs_v = (
                indices[:, None].to(tl.int64) * stride_v_token
                + cur_kv_head_id * stride_v_head
                + offs_dv[None, :]
            )
            v = tl.load(
                v_buffer + offs_v, mask=(mask_kv[:, None]) & (mask_v_d[None, :]), other=0.0
            )

        # online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]

        # score @ v
        if V_IS_K:
            acc += tl.dot(p.to(kn.dtype), kn)
        else:
            acc += tl.dot(p.to(v.dtype), v)

        # update global sum and max
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    if NUM_SPLITS > 1:
        # Publish this split's unnormalised partial (acc, e_sum, base-2 max),
        # then the last CTA of this (token, head block) to arrive merges all
        # NUM_SPLITS partials with the lse and writes the final result.
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
            # every split has arrived; leave the counter zero for the next call
            tl.store(counter, 0)
            offs_s = tl.arange(0, NUM_SPLITS)
            offs_ms = row[:, None] * NUM_SPLITS + offs_s[None, :]
            m_all = tl.load(
                part_max_ptr + offs_ms, mask=mask_h[:, None], other=-1.0e30, volatile=True
            )
            s_all = tl.load(
                part_sum_ptr + offs_ms, mask=mask_h[:, None], other=0.0, volatile=True
            )
            e_max = tl.max(m_all, 1)
            e_sum = tl.sum(tl.exp2(m_all - e_max[:, None]) * s_all, 1)
            acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
            for i in tl.static_range(NUM_SPLITS):
                m_i = tl.load(
                    part_max_ptr + row * NUM_SPLITS + i, mask=mask_h, other=-1.0e30, volatile=True
                )
                offs_a = (row[:, None] * NUM_SPLITS + i) * BLOCK_DV + offs_dv[None, :]
                acc_i = tl.load(
                    part_acc_ptr + offs_a, mask=mask_h[:, None], other=0.0, volatile=True
                )
                acc += acc_i * tl.exp2(m_i - e_max)[:, None]
            _mla_store_output(
                out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum, e_max,
                cur_q, cur_head, mask_h, offs_dv, dim_v,
                stride_out_token, stride_out_head, stride_lse, LOGE2,
            )
    else:
        _mla_store_output(
            out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum, e_max,
            cur_q, cur_head, mask_h, offs_dv, dim_v,
            stride_out_token, stride_out_head, stride_lse, LOGE2,
        )


@triton.jit
def _mla_store_output(
    out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum, e_max,
    cur_q, cur_head, mask_h, offs_dv, dim_v,
    stride_out_token, stride_out_head, stride_lse, LOGE2: tl.constexpr,
):
    # rescaling
    acc /= e_sum[:, None]

    max_logits = e_max * LOGE2
    # calculate lse (natural log, matching FlashMLA / FA conventions)
    lse = max_logits + tl.log2(e_sum) * LOGE2

    # write output
    offs_o = (
        cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :]
    )
    mask_out_d = offs_dv < dim_v
    tl.store(
        out_ptr + offs_o,
        acc.to(out_ptr.dtype.element_ty),
        mask=(mask_h[:, None]) & (mask_out_d[None, :]),
    )

    offs_lse = cur_q * stride_lse + cur_head
    tl.store(softmax_lse_ptr + offs_lse, lse, mask=mask_h)
    tl.store(max_logits_ptr + offs_lse, max_logits, mask=mask_h)


@functools.lru_cache
def _smem_budget(device_index: int) -> int:
    """Shared memory an sm_80 CTA may opt into, read from the device.

    BLOCK_N=64 at two stages asks for 128 KB at dim_qk=512, which GA100 grants
    and the 100 KB parts (sm_86 / sm_89) do not, so the tile is chosen against
    the real limit rather than against a constant.
    """
    try:
        return torch.cuda.get_device_properties(
            device_index
        ).shared_memory_per_block_optin
    except Exception:
        return 101376


@functools.lru_cache
def _num_sms(device_index: int) -> int:
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        return 108


def _pick_config(
    num_tokens: int,
    index_topk: int,
    num_heads_q: int = 64,
    dim_qk: int = 512,
    device_index: int = 0,
) -> tuple[int, int, int, int, int]:
    """(BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages).

    Measured on sm_80 (GA100, 70 SMs) at the deployed shape -- 16 heads per
    rank, top-k 2048, dim_qk 512 -- over 550 schedules at 4 and 16 query rows
    and context 8K / 64K / 200K, each against this kernel's previous schedule
    in the same run (graph replay, rotated caches, median of 20):

        rows  previous            us     new                 us     gain
        4     H16 N32 S8 W4 P3   32.1    H16 N64 S8 W4 P2   30.3    1.06x
        16    H32 N32 S4 W4 P3   59.8    H16 N64 S4 W4 P2   50.2    1.19x

    The previous schedule ranked 6th of 550 at 4 rows and 19th at 16. Three
    things changed:

    * **64 keys per tile, not 32.** Worth most of both numbers. This kernel is
      issue-bound rather than gather-bound -- a 12x change in the KV working
      set moves it under 1 % -- so what pays is halving the trip count over the
      top-k list, not touching the bytes. It costs shared memory, which is why
      the tile is sized against the device below.
    * **Two pipeline stages, not three.** The gather is an indirect load that
      Triton cannot pipeline, so a third stage double-buffers the wrong thing.
    * **The head tile is sized to the heads that exist.** The old rule took 32
      above 8 query rows, reasoning about a 64-head rank; a rank owning 16
      heads then builds a 32-row tile with half its rows masked and throws away
      half of every qk and pv dot. Worth 1.04x on its own at 16 rows (59.8 ->
      57.4 us), and nothing at 4, where the old rule already chose 16.
      Repartitioning heads across CTAs cannot change a result bit -- each
      head's softmax reduction is independent.

    The split ramp is also computed from the real head-block count rather than
    from a hard-coded 64 heads, and fills one wave of SMs rather than a fixed
    160 CTAs. At the two shapes above both forms agree (8 splits at 4 rows, 4
    at 16); they differ at 8 query rows, where the old estimate double-counts
    the CTAs and stops at 4. Splits stay <= 8 because the last-CTA merge reads
    splits x BLOCK_H x 2 KB, and past the wave the merge tail costs more than
    the parallelism it buys (16 rows: 4 splits 50.2 us, 8 splits 60.8, 16
    splits 81.4).

    Set VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY=1 to get the previous schedule back.
    """
    from vllm import envs

    if envs.VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY:
        BLOCK_N = 32
        BLOCK_H = 16 if num_tokens <= 8 else 32
        ctas = num_tokens * (64 // BLOCK_H)
        num_splits = 1
        while num_splits < 8 and ctas * num_splits * 2 <= 160:
            num_splits *= 2
        return BLOCK_H, BLOCK_N, min(num_splits, _cdiv(index_topk, BLOCK_N)), 4, 3

    # Never build a head tile wider than the heads the rank holds; 32 stays the
    # ceiling, so a rank with more than 16 heads keeps today's tile.
    BLOCK_H = min(32, max(16, 1 << (num_heads_q - 1).bit_length()))

    # stages * BLOCK_N * dim_qk * 2 B of staging must fit what the device will
    # grant one CTA. Give up the wide tile before the second stage: the tile is
    # worth more than the pipelining on a kernel whose loads are indirect.
    smem = _smem_budget(device_index)
    BLOCK_N, num_stages = 64, 2
    while BLOCK_N > 16 and num_stages * BLOCK_N * dim_qk * 2 > smem:
        BLOCK_N //= 2
    while num_stages > 1 and num_stages * BLOCK_N * dim_qk * 2 > smem:
        num_stages -= 1

    head_blocks = _cdiv(num_heads_q, min(BLOCK_H, num_heads_q))
    ctas = num_tokens * head_blocks
    num_splits = 1
    while num_splits < 8 and ctas * num_splits * 2 <= _num_sms(device_index):
        num_splits *= 2
    num_splits = min(num_splits, _cdiv(index_topk, BLOCK_N))
    return BLOCK_H, BLOCK_N, num_splits, 4, num_stages


_WORKSPACE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _workspace(device, num_tokens, num_heads, num_splits, block_dv, head_blocks):
    """Split-k scratch, cached per shape. The arrival counter is zeroed once here
    and left zero by the kernel (the merging CTA resets it), so no per-call memset."""
    key = (device, num_tokens, num_heads, num_splits, block_dv, head_blocks)
    ws = _WORKSPACE.get(key)
    if ws is None:
        acc = torch.empty(
            (num_tokens, num_heads, num_splits, block_dv), dtype=torch.float32, device=device
        )
        stats = torch.empty((2, num_tokens, num_heads, num_splits), dtype=torch.float32, device=device)
        counter = torch.zeros((num_tokens, head_blocks), dtype=torch.int32, device=device)
        ws = (acc, stats[0], stats[1], counter)
        _WORKSPACE[key] = ws
    return ws


def _use_ampere_prefill_sparse_mla(num_tokens: int, seq_kv: int,
                                   index_topk: int) -> bool:
    """Gate for vllm/ampere_prefill/sparse_prefill_mla.py.

    Two conditions, both measured (CMP 170HX, 70 SMs, TP=4 so h_q = 16):

      * num_tokens: below the threshold this is decode, where the upstream
        schedule was tuned and where the captured CUDA graphs live.
      * seq_kv vs index_topk: the tuned kernel wins by rotating each query's
        start offset so concurrent CTAs stop marching through the cache in
        lockstep, which turns the gather into L2 hits (1752 GB/s at ctx 99328,
        above the 1.66 TB/s pure-read HBM ceiling). When the context is barely
        larger than top-k, almost every row is selected by almost every query,
        the working set is L2-resident anyway and the rotation only adds index
        arithmetic: measured 0.93x at ctx 2304, 1.39x at ctx 8192, topk 2048.
        The default multiple of 2.0 puts the gate at 4096 rows, between the two.
    """
    from vllm import envs
    from vllm.platforms import current_platform

    if not envs.VLLM_GLM5_PREFILL_KERNELS:
        return False
    if num_tokens < envs.VLLM_GLM5_PREFILL_MIN_TOKENS:
        return False
    if seq_kv < envs.VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT * index_topk:
        return False
    return current_platform.is_cuda() and current_platform.is_device_capability(80)


def triton_mla_sparse_fwd(
    q: torch.Tensor,  # [num_tokens, num_heads_q, dim_qk]
    kv: torch.Tensor,  # [num_rows, num_heads_kv, dim_qk]
    indices: torch.Tensor,  # [num_tokens, num_heads_kv, topk] int32
    sm_scale: float,
    d_v: int = 512,
    block_dpe: int | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse MQA attention over ``indices``-selected cache rows.

    Returns ``(out, max_logits, lse)``:

    * ``out``        -- ``[num_tokens, num_heads_q, d_v]`` in ``q.dtype``
    * ``max_logits`` -- ``[num_tokens, num_heads_q]`` fp32
    * ``lse``        -- ``[num_tokens, num_heads_q]`` fp32, natural-log
      logsumexp of the (scaled) logits over the valid indices.

    Args:
        block_dpe: width of the RoPE tail of ``dim_qk`` (``qk_rope_head_dim``).
            Defaults to ``dim_qk - d_v``: 0 for the NoPE 512 layout, 64 for
            the DeepSeek 576 layout.
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    _, num_heads_kv, _ = kv.shape
    assert dim_qk == kv.shape[2], "q and kv have different head dimensions"
    assert indices.dtype == torch.int32
    assert indices.shape[0] == num_tokens
    _, _, index_topk = indices.shape

    if block_dpe is None:
        block_dpe = dim_qk - d_v

    if _use_ampere_prefill_sparse_mla(num_tokens, kv.shape[0], index_topk):
        from vllm.ampere_prefill.sparse_prefill_mla import sparse_mla_fwd

        return sparse_mla_fwd(
            q, kv, indices, sm_scale, d_v=d_v, block_dpe=block_dpe, out=out
        )

    BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages = _pick_config(
        num_tokens, index_topk, num_heads_q, dim_qk, q.device.index or 0
    )
    BLOCK_DPE = block_dpe
    BLOCK_DMODEL = dim_qk - BLOCK_DPE
    BLOCK_DV = d_v
    assert BLOCK_DV & (BLOCK_DV - 1) == 0, "d_v must be a power of two"
    assert BLOCK_DMODEL & (BLOCK_DMODEL - 1) == 0, "dim_nope must be a power of two"
    assert BLOCK_DPE == 0 or BLOCK_DPE & (BLOCK_DPE - 1) == 0, (
        "block_dpe must be 0 or a power of two"
    )
    assert d_v <= BLOCK_DMODEL, "V is read from the leading dims of the latent"
    assert num_heads_kv == 1, "only support kv head = 1 for now"
    assert index_topk > 0

    sm_scale = sm_scale * LOG2E

    kv_group_num = num_heads_q // num_heads_kv
    split_len = _cdiv(_cdiv(index_topk, num_splits), BLOCK_N) * BLOCK_N
    num_splits = _cdiv(index_topk, split_len)
    # merge kernel iterates tl.arange(0, NUM_SPLITS) -> power of two
    num_splits = 1 << (num_splits - 1).bit_length()
    grid = (
        num_tokens,
        _cdiv(num_heads_q, min(BLOCK_H, kv_group_num)),
        num_splits,
    )

    if out is None:
        out = torch.empty(
            (num_tokens, num_heads_q, d_v), dtype=q.dtype, device=q.device
        )
    else:
        assert out.shape == (num_tokens, num_heads_q, d_v)
        assert out.dtype == q.dtype
    # One allocation for both [num_tokens, num_heads_q] fp32 statistics: a
    # torch.empty costs ~2.6 us of host time, which is a fifth of a decode
    # launch. The two returned tensors are ordinary contiguous views.
    stats = torch.empty(
        (2, num_tokens, num_heads_q), dtype=torch.float32, device=q.device
    )
    max_logits, softmax_lse = stats[0], stats[1]

    # kv[..., :d_v] would have kv's pointer and strides anyway (d_v <=
    # BLOCK_DMODEL, and the kernel masks the V columns), so skip the slice.
    k = v = kv

    if num_splits == 1:
        # unused; any tensor works as a placeholder pointer
        part_acc = part_sum = part_max = counter = max_logits
    else:
        part_acc, part_sum, part_max, counter = _workspace(
            q.device, num_tokens, num_heads_q, num_splits, BLOCK_DV, grid[1]
        )

    sq, skv, so, si = q.stride(), kv.stride(), out.stride(), indices.stride()
    _mla_sparse_kernel[grid](
        q_buffer=q,
        k_buffer=k,
        v_buffer=v,
        indices_ptr=indices,
        out_ptr=out,
        softmax_lse_ptr=softmax_lse,
        part_acc_ptr=part_acc,
        part_sum_ptr=part_sum,
        part_max_ptr=part_max,
        counter_ptr=counter,
        max_logits_ptr=max_logits,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        dim_qk=dim_qk,
        dim_v=d_v,
        stride_q_token=sq[0],
        stride_q_head=sq[1],
        stride_k_token=skv[0],
        stride_k_head=skv[1],
        stride_v_token=skv[0],
        stride_v_head=skv[1],
        stride_out_token=so[0],
        stride_out_head=so[1],
        stride_lse=num_heads_q,  # a contiguous [num_tokens, num_heads_q] row
        stride_indices_token=si[0],
        stride_indices_head=si[1],
        sm_scale=sm_scale,
        split_len=split_len,
        kv_group_num=kv_group_num,
        index_topk=index_topk,
        NUM_SPLITS=num_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        V_IS_K=(d_v == BLOCK_DMODEL),
        LOGE2=LOGE2,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, max_logits, softmax_lse
