# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 attention for sm_80 over paged, chunked, batched KV.

This is DeepSeek's reference ``Attention.forward`` (with ``_window_kv``,
``_compress_kv``, ``Compressor``, ``Indexer`` and ``select_candidate_blocks``)
restated per token, so that a token's result does not depend on how its
request was chunked or batched. For a query at absolute position ``p``:

* window: positions ``[max(0, p - window + 1), p]`` of this layer's sliding
  cache (the SWA metadata's slot lists);
* compressed: ``n_vis = (p + 1) // ratio`` compressed positions are visible;
  the index source keeps ``min(index_topk, n_vis)`` of them by score (all of
  them when ``n_vis <= index_topk``), restricted to the candidate blocks for
  indexers above the candidate source.

The rounding points are the reference's: window KV is RoPE'd then FP8 (32,
UE8M0) quantize-dequantized before it is cached; the compressed latent is
RoPE'd then FP4 (16, E4M3 scale) qdq'd; index keys are ``k_norm(wk(latent))``,
RoPE'd, FP4 (32, E8M0) qdq'd; the index query is FP4 (32, E8M0) qdq'd; every
per-head index score is rounded to BF16, rectified, weighted and rounded to
BF16 again; heads are summed in the reference's TP groups (fp32 in head
order, rounded to BF16 per group) and the groups added in rank order with
BF16 rounding, which is what the reference's sharded heads plus all-reduce
compute. Attention output has the inverse RoPE on its tail and is FP8 (32,
UE8M0) qdq'd per group before ``wo_a``. Caches hold plain BF16 rows of those
exact values. RoPE is the reference's own arithmetic (``torch.polar`` of the
fp32 angles, complex multiply in fp32).

Selections are canonical: ties at the top-k boundary go to the lowest
position (score descending, index ascending), and the kept positions are
sorted ascending. Visible positions masked out by the candidate blocks score
``-FLT_MAX`` and invisible ones ``-inf``, so a masked visible position is only
ever picked ahead of an invisible one, as in the reference.

Everything here is the torch definition and runs on any device; on CUDA the
index scores and the sparse attention run as Triton kernels
(``sm80_index_scores``, ``sm80_sparse_attn``) with the same rounding points,
and the top-k uses ``torch.topk`` repaired by the shared tie fix. Paged
addressing goes through tensor indexing (int64) here and int64 offsets in the
kernels.

No step reads a device value on the host: every tensor is shaped by the
step's token count and host metadata (``Sm80AttnBatch.max_seq_len``), rows
that write nothing are computed and then dropped by masked writes (which
target the null block's first row with its own value), and the indexer's
score width is the batch's host-side bound on visible compressed positions.
Only the CPU fallbacks of the Triton kernels (``index_scores_torch``,
``sparse_attention_torch``) still index by data.
"""

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import qdq_fp4, qdq_fp8

FLT_MAX = torch.finfo(torch.float32).max
# Upper bound of one index-score chunk ([rows, visible positions] fp32).
DEFAULT_SCORE_CHUNK_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Sm80AttnConfig:
    """Static description of one attention layer on this rank."""

    n_local_heads: int
    head_dim: int
    rope_dim: int
    n_local_groups: int
    window_size: int
    compress_ratio: int
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidates: bool
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_topk_blocks: int
    candidate_block_size: int
    tp_size: int
    eps: float

    @property
    def owns_k(self) -> bool:
        return self.is_kv_source and self.is_index_source

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def index_weight_scale(self) -> float:
        # Reference: weights_proj(x) * (softmax_scale * n_heads ** -0.5).
        return self.index_head_dim**-0.5 * self.index_n_heads**-0.5

    @property
    def index_heads_per_rank(self) -> int:
        return self.index_n_heads // self.tp_size


@dataclass
class Sm80AttnBatch:
    """Per-step token metadata; T = number of real tokens.

    Slot ids address ``cache[slot // E, slot % E]`` with ``E = cache.shape[1]``.
    """

    positions: torch.Tensor  # [T] int64 absolute positions
    token_to_req: torch.Tensor  # [T] int
    query_start_loc: torch.Tensor  # [R + 1] int
    swa_slots: torch.Tensor  # [T] int64 slot of the token's own window row, -1
    swa_indices: torch.Tensor  # [T, W] int32 window slots to attend to, -1 pad
    comp_slots: torch.Tensor | None = None  # [T] int64 slot of the closed group
    comp_block_table: torch.Tensor | None = None  # [R, nb] int32
    idxk_slots: torch.Tensor | None = None  # [T] int64
    idxk_block_table: torch.Tensor | None = None  # [R, nb] int32
    ring_slots: torch.Tensor | None = None  # [T] int64 ratio-2 state ring slot
    # Host-side upper bound on any request's sequence length this step (after
    # its new tokens): bounds the visible compressed positions without a sync.
    max_seq_len: int = 0

    @property
    def num_tokens(self) -> int:
        return self.positions.shape[0]


@dataclass
class Sm80AttnCaches:
    swa: torch.Tensor  # [blocks, block_size, head_dim] bf16
    topk: torch.Tensor  # [>= T, index_topk] int32, shared by the layers
    comp: torch.Tensor | None = None  # [blocks, E, head_dim] bf16 (kv source's)
    idxk: torch.Tensor | None = None  # [blocks, E, index_head_dim] bf16
    ring: torch.Tensor | None = None  # [blocks, capacity, 2 * head_dim] fp32
    candidates: torch.Tensor | None = None  # [>= T, candidate_topk_blocks] int32


@dataclass
class Sm80AttnWeights:
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    attn_sink: torch.Tensor  # [n_local_heads] fp32
    freqs: torch.Tensor  # [rope_dim // 2] fp32 rotary frequencies
    wq_b: Callable[[torch.Tensor], torch.Tensor]
    comp_norm: torch.Tensor | None = None
    index_wq_b: Callable[[torch.Tensor], torch.Tensor] | None = None
    wk: Callable[[torch.Tensor], torch.Tensor] | None = None
    k_norm: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Elementwise pieces
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference ``RMSNorm.forward``: fp32 statistics, product with the
    weight in fp32, one rounding to ``x``'s dtype."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight * xf).to(dtype)


def rope_freqs(
    dim: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Reference ``precompute_freqs_cis`` up to the outer product: the fp32
    frequency of each rotated pair, with YaRN when ``original_seq_len > 0``."""
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Reference ``apply_rotary_emb`` on the last ``2 * len(freqs)`` dims of
    ``x`` ([T, ..., D], adjacent pairs as complex numbers), out of place.

    The rotation is the reference's: ``polar(1, fp32(position) * freq)``
    (the rows of its outer-product table), a complex64 multiply (by the
    conjugate for ``inverse``), one rounding to ``x``'s dtype."""
    rd = 2 * freqs.numel()
    angles = positions.to(torch.float32).unsqueeze(-1) * freqs.to(positions.device)
    cis = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        cis = cis.conj()
    cis = cis.view((positions.shape[0],) + (1,) * (x.dim() - 2) + (rd // 2,))
    tail = torch.view_as_complex(x[..., -rd:].float().unflatten(-1, (-1, 2)))
    out = x.clone()
    out[..., -rd:] = torch.view_as_real(tail * cis).flatten(-2).to(x.dtype)
    return out


def _paged(cache: torch.Tensor, slots: torch.Tensor):
    entries = cache.shape[1]
    slots = slots.to(torch.int64)
    return slots // entries, slots % entries


def store_rows(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """``cache[slot] = row`` for every ``slot >= 0`` (valid slots are
    distinct), without a host sync: the other rows write slot 0 -- the null
    block, which no request owns -- with the value it already holds."""
    keep = slots >= 0
    blk, off = _paged(cache, torch.where(keep, slots, 0))
    keep = keep.view((-1,) + (1,) * (rows.dim() - 1))
    cache[blk, off] = torch.where(keep, rows.to(cache.dtype), cache[blk, off])


def gather_compressed_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    req: torch.Tensor,
    local: torch.Tensor,
) -> torch.Tensor:
    """Rows of compressed positions ``local`` (>= 0) of requests ``req``."""
    entries = cache.shape[1]
    local = local.to(torch.int64)
    blk = block_table[req.to(torch.int64), local // entries].to(torch.int64)
    return cache[blk, local % entries]


# ---------------------------------------------------------------------------
# KV production
# ---------------------------------------------------------------------------


def window_kv(
    kv: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Reference ``_window_kv`` before the cache write: RoPE, then FP8 qdq
    over all 512 dims (RoPE tail included)."""
    return qdq_fp8(rope(kv, positions, freqs), 32)


def compress(
    cfg: Sm80AttnConfig,
    batch: Sm80AttnBatch,
    kv_score: torch.Tensor,
    norm_weight: torch.Tensor,
    ring: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`compress_rows` of one layer's step."""
    assert batch.comp_slots is not None
    return compress_rows(
        kv_score,
        norm_weight,
        ring,
        batch.positions,
        batch.token_to_req,
        batch.query_start_loc,
        batch.comp_slots,
        batch.ring_slots,
        cfg.compress_ratio,
        cfg.head_dim,
        cfg.eps,
    )


def compress_rows(
    kv_score: torch.Tensor,
    norm_weight: torch.Tensor,
    ring: torch.Tensor | None,
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    comp_slots: torch.Tensor,
    ring_slots: torch.Tensor | None,
    compress_ratio: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``Compressor.forward`` for every group a token closes.

    Returns ``(closes, latent)``, both over all T tokens: whether the token
    closes a group (and so writes a compressed row), and the normalized BF16
    latent (pre-RoPE) of the group it would close; rows that close nothing
    hold values no one stores. Ratio 1 is ``norm(wkv(x))`` with the BF16
    projection. Ratio 2 pools the closing token with its predecessor -- from
    this chunk, or from the request's state ring when the chunk starts on
    the second token of a group -- by an fp32 softmax over the gate, rounds
    the pooled row to BF16 and normalizes it; then the last ``capacity`` raw
    ``[kv | score]`` rows of each request go to the ring. The arguments are
    plain tensors and numbers so that a call can be recorded and replayed as
    it ran.
    """
    d = head_dim
    pos = positions
    T = pos.shape[0]
    if compress_ratio == 1:
        kv = kv_score[:T, :d].to(torch.bfloat16)
        return comp_slots >= 0, rms_norm(kv, norm_weight, eps)

    assert compress_ratio == 2 and ring is not None and ring_slots is not None
    assert kv_score.dtype == torch.float32 and kv_score.shape[1] == 2 * d
    closes = (comp_slots >= 0) & ((pos + 1) % 2 == 0)
    t = torch.arange(T, device=pos.device)
    req = token_to_req.to(torch.int64)
    qsl = query_start_loc.to(torch.int64)
    capacity = ring.shape[1]
    # A group a chunk's first token closes started in an earlier step: its
    # first row is in the ring (a closing token always has a ring slot).
    ring_slot = ring_slots.to(torch.int64).clamp_min(0)
    from_ring = ring[ring_slot // capacity, (pos - 1) % capacity]
    first = (t == qsl[req]).unsqueeze(-1)
    prev = torch.where(first, from_ring, kv_score[(t - 1).clamp_min(0)])
    cur = kv_score[:T]
    kv = torch.stack((prev[:, :d], cur[:, :d]), dim=1)
    score = torch.stack((prev[:, d:], cur[:, d:]), dim=1)
    pooled = (kv * score.softmax(dim=1)).sum(dim=1)
    latent = rms_norm(pooled.to(torch.bfloat16), norm_weight, eps)

    # Ring tail: only the last `capacity` tokens of each request, so no two
    # writes land on one ring row. A FULL graph's padding tokens lie past the
    # last request's query and map to request 0 with a ring slot of its block;
    # they are not in any request's query and write nothing.
    end = qsl[req + 1]
    tail = (ring_slots >= 0) & (t >= end - capacity) & (t < end)
    store_rows(ring, torch.where(tail, ring_slots, -1), kv_score[:T])
    return closes, latent


def index_keys(
    cfg: Sm80AttnConfig,
    weights: Sm80AttnWeights,
    latent: torch.Tensor,
    group_pos: torch.Tensor,
) -> torch.Tensor:
    """Reference index-K append: ``k_norm(wk(latent))``, RoPE at the group's
    first position, FP4 (32, E8M0) qdq."""
    assert weights.wk is not None and weights.k_norm is not None
    return index_key_rows(
        weights.wk(latent), weights.k_norm, weights.freqs, group_pos, cfg.eps
    )


def index_key_rows(
    k: torch.Tensor,
    k_norm: torch.Tensor,
    freqs: torch.Tensor,
    group_pos: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """:func:`index_keys` after ``wk``: ``k_norm``, RoPE at ``group_pos``,
    FP4 (32, E8M0) qdq (plain arguments, recordable as a call)."""
    k = rms_norm(k, k_norm, eps)
    return qdq_fp4(rope(k, group_pos, freqs), 32, "e8m0")


def compressed_kv(
    latent: torch.Tensor, group_pos: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Reference compressed-cache row: RoPE, then FP4 (16, E4M3 scale) qdq."""
    return qdq_fp4(rope(latent, group_pos, freqs), 16, "e4m3")


def store_kv(
    cfg: Sm80AttnConfig,
    weights: Sm80AttnWeights,
    caches: Sm80AttnCaches,
    batch: Sm80AttnBatch,
    kv: torch.Tensor | None,
    kv_score: torch.Tensor | None,
    window: bool = True,
) -> None:
    """Every KV write of one layer: window row, compressed row, index key.

    ``kv`` is ``kv_norm(wkv(x))`` (BF16, pre-RoPE). Without ``window`` the
    window rows are already written (the fused prologue) and ``kv`` is unused."""
    if window:
        rows = window_kv(kv, batch.positions, weights.freqs)
        store_rows(caches.swa, batch.swa_slots, rows)
    if not cfg.is_kv_source:
        return
    assert kv_score is not None and caches.comp is not None
    assert weights.comp_norm is not None
    closes, latent = compress(cfg, batch, kv_score, weights.comp_norm, caches.ring)
    group_pos = batch.positions + 1 - cfg.compress_ratio
    if cfg.owns_k:
        assert caches.idxk is not None and batch.idxk_slots is not None
        store_rows(
            caches.idxk,
            torch.where(closes, batch.idxk_slots, -1),
            index_keys(cfg, weights, latent, group_pos),
        )
    assert batch.comp_slots is not None
    store_rows(
        caches.comp,
        torch.where(closes, batch.comp_slots, -1),
        compressed_kv(latent, group_pos, weights.freqs),
    )


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


def index_query(
    cfg: Sm80AttnConfig,
    weights: Sm80AttnWeights,
    qr: torch.Tensor,
    index_weights: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index query (RoPE then FP4 (32, E8M0) qdq) and the per-head BF16
    weights ``weights_proj(x) * (index_head_dim * index_n_heads) ** -0.5``."""
    assert weights.index_wq_b is not None
    q = weights.index_wq_b(qr).view(-1, cfg.index_n_heads, cfg.index_head_dim)
    q = qdq_fp4(rope(q, positions, weights.freqs), 32, "e8m0")
    # A Python float multiplies an fp32 tensor as its fp32 rounding.
    w = (index_weights.float() * cfg.index_weight_scale).to(torch.bfloat16)
    return q, w


def combine_head_scores(
    per_head: torch.Tensor, weights: torch.Tensor, heads_per_rank: int
) -> torch.Tensor:
    """``[..., H, N]`` fp32 dot products -> ``[..., N]`` index scores (fp32
    holding BF16 values), with the reference's rounding: each head's score to
    BF16, ReLU, times its BF16 weight to BF16; each rank's heads summed in
    fp32 (head order) and rounded; the ranks added in order, rounding each
    sum to BF16 (the all-reduce)."""
    bf16 = torch.bfloat16
    s = per_head.to(bf16).float().clamp_min(0.0)
    s = (s * weights.float().unsqueeze(-1)).to(bf16).float()
    n_heads = s.shape[-2]
    total = None
    for r0 in range(0, n_heads, heads_per_rank):
        part = s[..., r0, :]
        for h in range(r0 + 1, r0 + heads_per_rank):
            part = part + s[..., h, :]
        part = part.to(bf16).float()
        total = part if total is None else (total + part).to(bf16).float()
    assert total is not None
    return total


def index_scores_torch(
    q: torch.Tensor,
    w: torch.Tensor,
    n_vis: torch.Tensor,
    req: torch.Tensor,
    idxk: torch.Tensor,
    block_table: torch.Tensor,
    n_cols: int,
    heads_per_rank: int,
) -> torch.Tensor:
    """``[n, n_cols]`` index scores of compressed positions ``< n_vis`` (the
    rest ``-inf``) for queries ``q`` [n, H, D] of requests ``req``."""
    n = q.shape[0]
    out = torch.full((n, n_cols), float("-inf"), dtype=torch.float32, device=q.device)
    for r in torch.unique(req).tolist():
        sel = torch.nonzero(req == r).flatten()
        width = int(n_vis[sel].max())
        if width == 0:
            continue
        cols = torch.arange(width, device=q.device)
        k = gather_compressed_rows(
            idxk, block_table, torch.full_like(cols, r), cols
        ).float()
        dots = torch.einsum("thd,nd->thn", q[sel].float(), k)
        s = combine_head_scores(dots, w[sel], heads_per_rank)
        s = torch.where(cols[None, :] < n_vis[sel, None], s, float("-inf"))
        out[sel, :width] = s
    return out


def index_scores(q, w, n_vis, req, idxk, block_table, n_cols, heads_per_rank):
    if q.is_cuda:
        from vllm_sm80.models.deepseek_v41.ops.sm80_index_scores import (
            index_scores_triton,
        )

        return index_scores_triton(
            q, w, n_vis, req, idxk, block_table, n_cols, heads_per_rank
        )
    return index_scores_torch(
        q, w, n_vis, req, idxk, block_table, n_cols, heads_per_rank
    )


def _mask_invisible(scores: torch.Tensor, n_vis: torch.Tensor) -> torch.Tensor:
    """Columns at or past a row's visible count -> -inf (contiguous fp32)."""
    cols = torch.arange(scores.shape[1], device=scores.device)
    return torch.where(cols[None, :] < n_vis[:, None], scores, float("-inf"))


def topk_canonical(scores: torch.Tensor, n_vis: torch.Tensor, k: int) -> torch.Tensor:
    """``min(k, n_vis)`` best columns per row (score descending, index
    ascending), sorted ascending, ``-1`` padded to width ``k``."""
    n, n_cols = scores.shape
    out = torch.full((n, k), -1, dtype=torch.int32, device=scores.device)
    width = min(k, n_cols)
    if n == 0 or width == 0:
        return out
    scores = _mask_invisible(scores, n_vis)
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :width]
    keep = torch.arange(width, device=scores.device)[None, :] < n_vis[:, None]
    picked = torch.where(keep, order, n_cols)
    picked = torch.sort(picked, dim=-1).values
    out[:, :width] = torch.where(picked == n_cols, -1, picked).to(torch.int32)
    return out


def topk_canonical_cuda(
    scores: torch.Tensor, n_vis: torch.Tensor, k: int
) -> torch.Tensor:
    """:func:`topk_canonical` on CUDA: ``torch.topk`` returns an exact top-k
    whose ties are placed arbitrarily; the shared tie fix makes the set
    canonical and sorts it (one more read of the scores). The tie fix runs on
    the whole ``[n, k]`` buffer, whose columns past the width are -1, so its
    kernel compiles once per ``k``, not once per chunk width."""
    from vllm_sm80.kernels.indexer.topk_tiefix import topk_tiefix_

    n, n_cols = scores.shape
    out = torch.full((n, k), -1, dtype=torch.int32, device=scores.device)
    width = min(k, n_cols)
    if n == 0 or width == 0:
        return out
    scores = _mask_invisible(scores, n_vis)
    idx = torch.topk(scores, width, dim=-1, sorted=False).indices.to(torch.int32)
    idx = torch.where(idx < n_vis[:, None], idx, -1)
    out[:, :width] = idx
    topk_tiefix_(scores, out, row_ends=n_vis.to(torch.int32), sort=True)
    return out


def select_topk(scores: torch.Tensor, n_vis: torch.Tensor, k: int) -> torch.Tensor:
    if scores.is_cuda:
        return topk_canonical_cuda(scores, n_vis, k)
    return topk_canonical(scores, n_vis, k)


def select_candidate_blocks(
    scores: torch.Tensor, n_vis: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Reference ``select_candidate_blocks`` as sorted block ids: block score
    is its best position, the block holding the newest visible position is
    pinned in, the best ``topk_blocks`` reachable blocks are kept (canonical
    ties), ``-1`` padded to width ``topk_blocks``."""
    n, n_cols = scores.shape
    pad = -n_cols % block_size
    blocks = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = blocks.shape[-1]
    last = torch.div(n_vis - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(
        torch.arange(num_blocks, device=scores.device)[None, :] == last[:, None],
        float("inf"),
    )
    out = torch.full((n, topk_blocks), -1, dtype=torch.int32, device=scores.device)
    width = min(topk_blocks, num_blocks)
    if n == 0 or width == 0:
        return out
    top = torch.sort(blocks, dim=-1, descending=True, stable=True)
    ids, vals = top.indices[:, :width], top.values[:, :width]
    ids = torch.where(vals > float("-inf"), ids, num_blocks)
    ids = torch.sort(ids, dim=-1).values
    out[:, :width] = torch.where(ids == num_blocks, -1, ids).to(torch.int32)
    return out


def mask_to_candidates(
    scores: torch.Tensor, candidates: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Level two: visible positions outside the candidate blocks -> -FLT_MAX."""
    n, n_cols = scores.shape
    num_blocks = -(-n_cols // block_size)
    keep = torch.zeros((n, num_blocks + 1), dtype=torch.bool, device=scores.device)
    ids = torch.where(
        (candidates >= 0) & (candidates < num_blocks), candidates, num_blocks
    ).to(torch.int64)
    keep.scatter_(1, ids, True)
    keep = keep[:, :num_blocks].repeat_interleave(block_size, dim=1)[:, :n_cols]
    return torch.where(keep | (scores == float("-inf")), scores, -FLT_MAX)


def run_indexer(
    cfg: Sm80AttnConfig,
    caches: Sm80AttnCaches,
    batch: Sm80AttnBatch,
    q: torch.Tensor,
    w: torch.Tensor,
    valid: torch.Tensor,
    chunk_bytes: int = DEFAULT_SCORE_CHUNK_BYTES,
) -> None:
    """Fill ``caches.topk[:T]`` (and the candidate buffer at the candidate
    source) for every token; rows are scored in chunks of bounded size.

    The score width is the host bound ``max_seq_len // ratio`` on any row's
    visible count; columns past a row's own count score ``-inf`` and are
    never selected, so the width changes no selection."""
    T = batch.num_tokens
    assert caches.idxk is not None and batch.idxk_block_table is not None
    n_vis = torch.where(
        valid,
        torch.div(batch.positions + 1, cfg.compress_ratio, rounding_mode="floor"),
        0,
    )
    req = batch.token_to_req.to(torch.int64)
    topk = caches.topk
    topk[:T].fill_(-1)
    if cfg.is_candidate_source:
        assert caches.candidates is not None
        caches.candidates[:T].fill_(-1)
    if batch.max_seq_len <= 0:
        raise ValueError("Sm80AttnBatch.max_seq_len must bound the sequences")
    width = max(batch.max_seq_len // cfg.compress_ratio, 1)
    rows_per_chunk = max(chunk_bytes // (width * 4), 1)
    for start in range(0, T, rows_per_chunk):
        rows = slice(start, min(start + rows_per_chunk, T))
        scores = index_scores(
            q[rows],
            w[rows],
            n_vis[rows],
            req[rows],
            caches.idxk,
            batch.idxk_block_table,
            width,
            cfg.index_heads_per_rank,
        )
        if cfg.is_candidate_source:
            assert caches.candidates is not None
            caches.candidates[rows] = select_candidate_blocks(
                scores,
                n_vis[rows],
                cfg.candidate_topk_blocks,
                cfg.candidate_block_size,
            )
        elif cfg.uses_candidates:
            assert caches.candidates is not None
            scores = mask_to_candidates(
                scores, caches.candidates[rows], cfg.candidate_block_size
            )
        topk[rows] = select_topk(scores, n_vis[rows], cfg.index_topk)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def sparse_attention_torch(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    comp_cache: torch.Tensor | None,
    comp_block_table: torch.Tensor | None,
    comp_local: torch.Tensor | None,
    req: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference ``sparse_attn`` over the window slots plus the selected
    compressed positions: fp32 scores times ``scale``, softmax with the
    per-head sink in the denominator, probabilities rounded to BF16 before
    they weight the rows; a query with no valid row gets zeros."""
    n, h, d = q.shape
    rows = []
    valid = []
    blk, off = _paged(swa_cache, swa_indices.clamp_min(0))
    rows.append(swa_cache[blk, off])
    valid.append(swa_indices >= 0)
    if comp_local is not None and comp_local.shape[1]:
        assert comp_cache is not None and comp_block_table is not None
        r = req.to(torch.int64)[:, None].expand_as(comp_local)
        rows.append(
            gather_compressed_rows(
                comp_cache, comp_block_table, r, comp_local.clamp_min(0)
            )
        )
        valid.append(comp_local >= 0)
    kv = torch.cat(rows, dim=1).float()
    mask = torch.cat(valid, dim=1)
    s = torch.einsum("nhd,nkd->nhk", q.float(), kv) * scale
    s = s.masked_fill(~mask[:, None, :], float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.full_like(m, -1e30))
    p = torch.exp(s - m)
    denom = p.sum(-1) + torch.exp(attn_sink[None, :h].float() - m.squeeze(-1))
    o = torch.einsum("nhk,nkd->nhd", p.to(torch.bfloat16).float(), kv)
    return (o / denom.unsqueeze(-1)).to(torch.bfloat16)


def sparse_attention(
    q, swa_cache, swa_indices, comp_cache, comp_block_table, comp_local, req, sink, scale
):
    args = (
        q, swa_cache, swa_indices, comp_cache, comp_block_table, comp_local,
        req, sink, scale,
    )
    if q.is_cuda:
        from vllm_sm80.models.deepseek_v41.ops.sm80_sparse_attn import (
            sparse_attention_triton,
        )

        return sparse_attention_triton(*args)
    return sparse_attention_torch(*args)


def output_qdq(
    o: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Inverse RoPE on the output tail, then FP8 (32, UE8M0) qdq of each
    group's ``[heads_per_group * head_dim]`` slice (the ``wo_a`` input)."""
    n, h, d = o.shape
    o = rope(o, positions, freqs, inverse=True)
    return qdq_fp8(o.view(n, n_groups, (h // n_groups) * d), 32).view(n, h, d)


# ---------------------------------------------------------------------------
# One layer
# ---------------------------------------------------------------------------


def pad_batch(batch: Sm80AttnBatch, num_rows: int) -> Sm80AttnBatch:
    """``batch`` extended to ``num_rows`` tokens with padding tokens that
    write nothing and attend to nothing: position 0, request 0 (past every
    request's query), PAD slots and window rows.

    The layer runs every row-wise op (norms, ``wq_b``, the compressor, the
    index projections, qdq) on the padded row count of its input, whatever
    the metadata covers: a FULL graph's metadata covers the padded tokens, a
    piecewise step's only the scheduled ones, and kernels such as reductions
    and GEMMs may pick their configuration from the row count, which would
    change the live rows' last bits between the two."""
    extra = num_rows - batch.num_tokens
    if extra <= 0:
        return batch

    def ext(t: torch.Tensor | None, value: int) -> torch.Tensor | None:
        if t is None:
            return None
        return torch.cat([t, t.new_full((extra,) + tuple(t.shape[1:]), value)])

    return dataclasses.replace(
        batch,
        positions=ext(batch.positions, 0),
        token_to_req=ext(batch.token_to_req, 0),
        swa_slots=ext(batch.swa_slots, -1),
        swa_indices=ext(batch.swa_indices, -1),
        comp_slots=ext(batch.comp_slots, -1),
        idxk_slots=ext(batch.idxk_slots, -1),
        ring_slots=ext(batch.ring_slots, -1),
    )


def prepare_layer(
    cfg: Sm80AttnConfig,
    weights: Sm80AttnWeights,
    caches: Sm80AttnCaches,
    batch: Sm80AttnBatch,
    qr_kv: torch.Tensor,
    kv_score: torch.Tensor | None,
    index_weights: torch.Tensor | None,
    chunk_bytes: int = DEFAULT_SCORE_CHUNK_BYTES,
    fused: bool = False,
    pipeline: bool = False,
    workspace=None,
) -> torch.Tensor:
    """Everything before attention: norms, ``wq_b`` and the query RoPE, this
    step's KV writes, and at index sources the top-k (and candidate) buffers.
    Returns the post-RoPE query [T, n_local_heads, head_dim].

    ``qr_kv`` [>= T, q_lora + head_dim] BF16; ``kv_score`` [>= T, ratio *
    head_dim] fp32 at kv sources; ``index_weights`` [>= T, index_n_heads]
    BF16 (``weights_proj`` output) at index sources.

    ``fused``: the norms and the window row run as one launch
    (``sm80_attn_fused.prologue``) and the returned query is ``wq_b``'s output
    BEFORE RoPE, which :func:`attend_layer` with ``fused`` rotates.
    ``pipeline`` is mutually exclusive with ``fused``: it uses reserved scratch,
    fuses compressor/query/key/selection work and returns POST-RoPE queries."""
    if pipeline:
        if fused or workspace is None:
            raise ValueError("pipeline needs workspace and excludes pre-RoPE fused mode")
        from vllm_sm80.models.deepseek_v41.ops.sm80_attn_pipeline import prepare

        return prepare(cfg, weights, caches, batch, qr_kv, kv_score, index_weights, workspace)
    T = batch.num_tokens
    pos = batch.positions
    d = cfg.head_dim
    if fused:
        from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused

        qr = sm80_attn_fused.prologue(
            qr_kv, weights.q_norm, weights.kv_norm, pos, weights.freqs,
            caches.swa, batch.swa_slots, cfg.eps,
        )
        kv = None
        q = weights.wq_b(qr).view(T, cfg.n_local_heads, d)
    else:
        qr, kv = qr_kv[:T].split([qr_kv.shape[1] - d, d], dim=-1)
        qr = rms_norm(qr, weights.q_norm, cfg.eps)
        kv = rms_norm(kv, weights.kv_norm, cfg.eps)
        q = rope(weights.wq_b(qr).view(T, cfg.n_local_heads, d), pos, weights.freqs)
    store_kv(
        cfg,
        weights,
        caches,
        batch,
        kv,
        None if kv_score is None else kv_score[:T],
        window=not fused,
    )
    if cfg.compress_ratio and cfg.is_index_source:
        assert index_weights is not None
        iq, iw = index_query(cfg, weights, qr, index_weights[:T], pos)
        run_indexer(cfg, caches, batch, iq, iw, batch.swa_slots >= 0, chunk_bytes)
    return q


def attend_layer(
    cfg: Sm80AttnConfig,
    weights: Sm80AttnWeights,
    caches: Sm80AttnCaches,
    batch: Sm80AttnBatch,
    q: torch.Tensor,
    out: torch.Tensor,
    fused: bool = False,
    pipeline: bool = False,
    workspace=None,
) -> None:
    """Sparse attention of the post-RoPE ``q`` over the window slots and the
    selected compressed positions (``caches.topk``), then the output RoPE and
    qdq. Writes ``out[:T]`` (the ``wo_a`` input, [n_local_heads, head_dim]
    BF16 rows) and zeros the padding rows ``out[T:]``.

    ``fused``: ``q`` is the pre-RoPE query of :func:`prepare_layer` with
    ``fused``; the query RoPE, attention and output run as
    ``sm80_attn_fused.attend``. ``pipeline`` consumes the POST-RoPE query
    and writes inverse-RoPE/QDQ output in the ordered-value kernel itself."""
    T = batch.num_tokens
    if pipeline:
        if fused or workspace is None:
            raise ValueError("pipeline needs workspace and excludes pre-RoPE fused mode")
        from vllm_sm80.models.deepseek_v41.ops.sm80_attn_fused import sparse_attention_triton

        sparse_attention_triton(q[:T], batch.positions, weights.freqs, caches.swa,
            batch.swa_indices, caches.comp, batch.comp_block_table,
            caches.topk[:T] if cfg.compress_ratio else None, batch.token_to_req,
            weights.attn_sink, cfg.softmax_scale, post_rope=True, final_out=out,
            scores_out=workspace.attention_scores)
        return
    if fused:
        from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused

        sm80_attn_fused.attend(
            q,
            batch.positions,
            weights.freqs,
            caches.swa,
            batch.swa_indices,
            caches.comp,
            batch.comp_block_table,
            caches.topk[:T] if cfg.compress_ratio else None,
            batch.token_to_req,
            weights.attn_sink,
            cfg.softmax_scale,
            cfg.n_local_groups,
            out,
        )
        return
    o = sparse_attention(
        q[:T],
        caches.swa,
        batch.swa_indices,
        caches.comp,
        batch.comp_block_table,
        caches.topk[:T] if cfg.compress_ratio else None,
        batch.token_to_req,
        weights.attn_sink,
        cfg.softmax_scale,
    )
    out[:T] = output_qdq(o, batch.positions, weights.freqs, cfg.n_local_groups)
    out[T:] = 0


def forward_layer(
    cfg: Sm80AttnConfig,
    weights: Sm80AttnWeights,
    caches: Sm80AttnCaches,
    batch: Sm80AttnBatch,
    qr_kv: torch.Tensor,
    kv_score: torch.Tensor | None,
    index_weights: torch.Tensor | None,
    out: torch.Tensor,
    chunk_bytes: int = DEFAULT_SCORE_CHUNK_BYTES,
) -> None:
    """One attention layer from the ``wq_a | wkv`` output to the ``wo_a``
    input: :func:`prepare_layer` then :func:`attend_layer`, on the batch
    padded to ``qr_kv``'s rows (:func:`pad_batch`)."""
    batch = pad_batch(batch, qr_kv.shape[0])
    q = prepare_layer(
        cfg, weights, caches, batch, qr_kv, kv_score, index_weights, chunk_bytes
    )
    attend_layer(cfg, weights, caches, batch, q, out)
