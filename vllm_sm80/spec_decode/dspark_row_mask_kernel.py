# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark adaptive depth: the per-step decision kernel.

Before the target forward of a uniform speculative decode step (every request
verifies K > 0 drafts in K + 1 rows, no prefill rows), one Triton program per
request turns its draft-step features, the policy weights and the cost table
into ``live_len`` and ``row_live`` (see ``dspark_row_mask``). For request r of
a b-request step, slot ``s = idx_mapping[r]``, in FP32::

    x_i       = [1, logit(c_i), logit(t_i), e_i]
    p_i       = 1 / (1 + exp(-(sum_f w[i-1][f] * x_i[f])))      (f in order)
    E(k)      = 1 + sum_{i <= k} prod_{j <= i} p_j
    cost_r(k) = T_b(K) - (T_b(K) - T_b(k)) / b
    live_len  = argmax_{k = 0 .. K} E(k) / cost_r(k)     (ties: larger k)
    or, when tau_b > 0 (b >= 2 with a "tau" in the policy):
    live_len  = max {k <= K : prod_{j <= k} p_j >= tau_b}   (0 if none)

with ``c_i`` the confidence head's output, ``t_i`` and ``e_i`` the draft
logits' top-1 probability and entropy (read only when the policy uses them),
``logit(q) = log(q') - log(1 - q')`` for ``q' = clamp(q, 1e-6, 1 - 1e-6)``.
K is this step's verified width (it may change every step); the cost row is
this step's b. Rows ``r*(K+1) + j`` are live iff ``j <= live_len``. Rows past
the step's ``b*(K+1)`` up to the widest step the state can mask are set live
(an earlier wider step may have masked them); rows past that are never
written.

With ``p_hat`` (a policy with a ``host_width`` block, ``dspark_host_width``)
the kernel also writes each request's ``p_i`` for positions ``1 .. W``,
``W = p_hat.shape[1]``, whatever K is: positions past K read the features of
the drafted block (0 past the drafted width), and they change nothing in the
decision. The host reads them a step later to choose a later step's K.

Any other step leaves every row live (``RowMaskState.reset_live``). The kernel
runs eagerly on the current stream, after the previous step's draft (which
wrote the features on the same stream) and before the target forward or
graph replay that reads ``row_live`` in place. No host synchronisation.
"""

from __future__ import annotations

import os

import torch

from vllm_sm80.spec_decode.dspark_row_mask import (
    FEATURES,
    PROB_EPS,
    RowMaskState,
    get_row_mask_state,
)
from vllm.triton_utils import tl, triton

__all__ = [
    "MAX_DRAFTS",
    "begin_step",
    "begin_target_step",
    "compute_row_mask",
    "reference_row_mask",
    "uniform_num_drafts",
]

# Drafts per request the kernel serves (its unrolled loop).
MAX_DRAFTS = 8
_BLOCK_ROWS = triton.next_power_of_2(MAX_DRAFTS + 1)
_NF = len(FEATURES)
# The Triton interpreter (CPU tests) runs the kernel on CPU tensors and
# launches without the CUDA compile options.
_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _mask_span(state: RowMaskState) -> int:
    """Rows any step can mask: max_num_reqs requests of max_draft + 1 rows."""
    return min(state.max_num_reqs * (state.max_draft + 1), state.max_num_tokens)


@triton.jit
def _log_odds(q, EPS: tl.constexpr):
    q = tl.minimum(tl.maximum(q, EPS), 1.0 - EPS)
    return tl.log(q) - tl.log(1.0 - q)


@triton.jit(do_not_specialize=["num_reqs", "num_drafts", "p_width"])
def _row_mask_kernel(
    conf_ptr,
    conf_stride,
    stats_ptr,
    stats_stride,
    w_ptr,
    tau_ptr,
    cost_ptr,
    cost_stride,
    idx_ptr,
    live_len_ptr,
    row_live_ptr,
    p_hat_ptr,
    p_hat_stride,
    num_reqs,
    num_drafts,
    p_width,
    span,
    MAX_K: tl.constexpr,
    NF: tl.constexpr,
    USE_STATS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    r = tl.program_id(0)
    rows_per_req = num_drafts + 1
    if r < num_reqs:
        slot = tl.load(idx_ptr + r).to(tl.int64)
        conf_row = conf_ptr + slot * conf_stride
        stats_row = stats_ptr + slot * stats_stride
        cost_row = cost_ptr + num_reqs * cost_stride
        tau = tl.load(tau_ptr + num_reqs)
        use_tau = (tau > 0.0) & (num_reqs > 1)
        b = num_reqs.to(tl.float32)
        t_full = tl.load(cost_row + num_drafts)
        t_zero = tl.load(cost_row)
        best = tl.math.div_rn(1.0, t_full - tl.math.div_rn(t_full - t_zero, b))
        best_k = tl.zeros((), dtype=tl.int32)
        prod = tl.full((), 1.0, dtype=tl.float32)
        expected = tl.full((), 1.0, dtype=tl.float32)
        for k in tl.static_range(1, MAX_K + 1):
            ok = k <= num_drafts
            # Positions past K only feed p_hat: the decision takes k <= K.
            out = k <= p_width
            need = ok | out
            w = w_ptr + (k - 1) * NF
            c = tl.load(conf_row + (k - 1), mask=need, other=0.5)
            z = tl.load(w, mask=need, other=0.0)
            z = z + tl.load(w + 1, mask=need, other=0.0) * _log_odds(c, EPS)
            if USE_STATS:
                top1 = tl.load(stats_row + (k - 1), mask=need, other=0.5)
                ent = tl.load(stats_row + MAX_K + (k - 1), mask=need, other=0.0)
                z = z + tl.load(w + 2, mask=need, other=0.0) * _log_odds(top1, EPS)
                z = z + tl.load(w + 3, mask=need, other=0.0) * ent
            p = tl.math.div_rn(1.0, 1.0 + tl.exp(-z))
            tl.store(p_hat_ptr + r * p_hat_stride + (k - 1), p, mask=out)
            t_k = tl.load(cost_row + k, mask=ok, other=1.0)
            prod = prod * p
            expected = expected + prod
            cost = t_full - tl.math.div_rn(t_full - t_k, b)
            score = tl.math.div_rn(expected, cost)
            take = ok & tl.where(use_tau, prod >= tau, score >= best)
            best = tl.where(take, score, best)
            best_k = tl.where(take, k, best_k)
        tl.store(live_len_ptr + r, best_k)
        j = tl.arange(0, BLOCK_ROWS)
        live = (j <= best_k).to(tl.int8)
        tl.store(row_live_ptr + r * rows_per_req + j, live, mask=j < rows_per_req)
    else:
        # Rows past this step that a wider earlier step may have masked.
        offs = num_reqs * rows_per_req + tl.arange(0, BLOCK_TAIL)
        ones = tl.full((BLOCK_TAIL,), 1, dtype=tl.int8)
        tl.store(row_live_ptr + offs, ones, mask=offs < span)


def _check(
    state: RowMaskState,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_drafts: int,
    p_hat: torch.Tensor | None,
):
    assert 0 < num_reqs <= state.max_num_reqs, (num_reqs, state.max_num_reqs)
    assert 0 < num_drafts <= min(state.max_draft, MAX_DRAFTS), (num_drafts, state.max_draft)
    assert num_reqs * (num_drafts + 1) <= state.max_num_tokens
    assert idx_mapping.dim() == 1 and idx_mapping.shape[0] >= num_reqs, idx_mapping.shape
    if p_hat is not None:
        assert p_hat.dtype == torch.float32 and p_hat.dim() == 2, (p_hat.dtype, p_hat.shape)
        assert p_hat.shape[0] >= num_reqs and p_hat.stride(1) == 1, p_hat.shape
        assert 0 < p_hat.shape[1] <= state.max_draft, (p_hat.shape, state.max_draft)


def _log_odds_torch(q: torch.Tensor) -> torch.Tensor:
    q = q.clamp(PROB_EPS, 1.0 - PROB_EPS)
    return torch.log(q) - torch.log(1.0 - q)


def reference_row_mask(
    state: RowMaskState,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_drafts: int,
    p_hat: torch.Tensor | None = None,
) -> None:
    """:func:`compute_row_mask` in torch, the same FP32 operations in the same
    order."""
    _check(state, idx_mapping, num_reqs, num_drafts, p_hat)
    n, k_full = num_reqs, num_drafts
    p_width = 0 if p_hat is None else p_hat.shape[1]
    slots = idx_mapping[:n].long()
    conf = state.conf_by_slot[slots, : max(k_full, p_width)]
    stats = state.stats_by_slot[slots, :, : max(k_full, p_width)]
    w = state.weights
    tau = state.tau[n]
    use_tau = bool(tau > 0) and n > 1
    row = state.cost[n]
    b = torch.tensor(float(n), dtype=torch.float32, device=row.device)
    t_full = row[k_full]
    one = torch.ones(n, dtype=torch.float32, device=row.device)
    best = torch.div(one, t_full - torch.div(t_full - row[0], b))
    best_k = torch.zeros(n, dtype=torch.int32, device=row.device)
    prod = one.clone()
    expected = one.clone()
    for k in range(1, max(k_full, p_width) + 1):
        wk = w[k - 1]
        z = wk[0] + wk[1] * _log_odds_torch(conf[:, k - 1])
        if state.uses_stats:
            z = z + wk[2] * _log_odds_torch(stats[:, 0, k - 1])
            z = z + wk[3] * stats[:, 1, k - 1]
        p = torch.div(one, 1.0 + torch.exp(-z))
        if k <= p_width:
            assert p_hat is not None
            p_hat[:n, k - 1] = p
        if k > k_full:
            continue
        prod = prod * p
        expected = expected + prod
        cost = t_full - torch.div(t_full - row[k], b)
        score = torch.div(expected, cost)
        take = (prod >= tau) if use_tau else (score >= best)
        best = torch.where(take, score, best)
        best_k = torch.where(take, torch.full_like(best_k, k), best_k)
    state.live_len[:n] = best_k
    j = torch.arange(k_full + 1, device=best_k.device, dtype=torch.int32)
    end = n * (k_full + 1)
    state.row_live[:end] = (j[None, :] <= best_k[:, None]).to(torch.int8).reshape(-1)
    state.row_live[end : _mask_span(state)] = 1
    state.active = True


def compute_row_mask(
    state: RowMaskState,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_drafts: int,
    p_hat: torch.Tensor | None = None,
) -> None:
    """Write ``live_len[:num_reqs]`` and ``row_live`` for a uniform step of
    ``num_reqs`` requests verifying ``num_drafts`` drafts each; ``idx_mapping``
    gives each request's state slot (device, batch order); with ``p_hat``
    ([>= num_reqs, W] float32, rows contiguous) also each request's ``p_i``
    for i = 1 .. W. One launch, no host synchronisation; the torch reference
    on CPU tensors."""
    if not (state.row_live.is_cuda or _INTERPRET):
        reference_row_mask(state, idx_mapping, num_reqs, num_drafts, p_hat)
        return
    _check(state, idx_mapping, num_reqs, num_drafts, p_hat)
    span = _mask_span(state)
    options = {} if _INTERPRET else {"enable_fp_fusion": False}
    _row_mask_kernel[(num_reqs + 1,)](
        state.conf_by_slot,
        state.conf_by_slot.stride(0),
        state.stats_by_slot,
        state.stats_by_slot.stride(0),
        state.weights,
        state.tau,
        state.cost,
        state.cost.stride(0),
        idx_mapping,
        state.live_len,
        state.row_live,
        state.weights if p_hat is None else p_hat,  # never written without p_hat
        0 if p_hat is None else p_hat.stride(0),
        num_reqs,
        num_drafts,
        0 if p_hat is None else p_hat.shape[1],
        span,
        MAX_K=state.max_draft,
        NF=_NF,
        USE_STATS=state.uses_stats,
        EPS=PROB_EPS,
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_TAIL=triton.next_power_of_2(span),
        num_warps=1,
        **options,
    )
    state.active = True


def uniform_num_drafts(state: RowMaskState, input_batch, dummy_run: bool) -> int:
    """K when the step is a uniform speculative decode the mask serves (every
    request verifies the same K > 0 drafts in K + 1 rows, no prefill rows,
    no GPU-side draft reallocation), else 0. Host facts only."""
    if dummy_run or input_batch.has_prefill or input_batch.max_query_len is not None:
        # max_query_len is set only when drafts are reallocated on the GPU.
        return 0
    drafts = input_batch.num_draft_tokens_per_req
    num_reqs = input_batch.num_reqs
    if drafts is None or not 0 < num_reqs <= state.max_num_reqs:
        return 0
    k = int(drafts[0])
    if not 0 < k <= min(state.max_draft, MAX_DRAFTS):
        return 0
    if input_batch.num_tokens != num_reqs * (k + 1) or not (drafts[:num_reqs] == k).all():
        return 0
    return k


def begin_step(state: RowMaskState, input_batch, dummy_run: bool) -> None:
    """The mask for this step's target forward: computed on a uniform
    speculative decode step, every row live on any other. With the host
    width on, ``state.p_hat_step`` is the step's new ``p_hat`` (batch order)
    on a uniform step and None on any other."""
    k = uniform_num_drafts(state, input_batch, dummy_run)
    state.p_hat_step = None
    if k:
        p_hat = None
        if state.p_hat_width:
            p_hat = torch.empty(
                input_batch.num_reqs,
                state.p_hat_width,
                dtype=torch.float32,
                device=state.row_live.device,
            )
        compute_row_mask(state, input_batch.idx_mapping, input_batch.num_reqs, k, p_hat)
        state.p_hat_step = p_hat
    else:
        state.reset_live()


def begin_target_step(input_batch, dummy_run: bool) -> None:
    """Runner entry, right before the target forward: :func:`begin_step` on
    the shared state; nothing when adaptive depth is off."""
    state = get_row_mask_state()
    if state is not None:
        begin_step(state, input_batch, dummy_run)

