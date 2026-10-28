# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp32 reference of one DeepSeek-V4.1 attention layer, and test harnesses.

``RefAttention`` restates DeepSeek's reference ``Attention`` (with
``_window_kv``, ``_compress_kv``, ``Compressor``, ``Indexer`` and
``select_candidate_blocks``) in plain torch for one request at a time and one
tensor-parallel rank, with the reference's own state: a window ring buffer,
contiguous compressed and index-key caches, prefill only at ``start_pos == 0``
and one token per decode step. Where the reference leaves an order open this
file fixes the one the fork defines, and says so: ties in every top-k go to
the lowest index, each rank's index heads are summed in head order, the
ranks' partial scores are added in rank order (BF16 after each add, as an
all-reduce does), and linears accumulate exactly (fp64) before their one
rounding, so that shapes cannot change a result.

``PagedRun`` drives the sm_80 layer (``sm80_attn_core.forward_layer``) the way
vLLM does: several requests per step, prompts split into chunks, decode steps
mixed in, block tables with high, shuffled and recycled block ids, CUDA-graph
style padding rows. ``reference_from_capture`` recomputes a captured layer
call in fp32 from its saved inputs (see the module's ``__main__``).
"""

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import (
    act_quant_qdq_fp8,
    fp4_qdq,
)

BF16 = torch.bfloat16


def exact_linear(x: torch.Tensor, w: torch.Tensor, dtype=None) -> torch.Tensor:
    """``F.linear`` with exact accumulation and one rounding to ``dtype``."""
    return F.linear(x.double(), w.double()).to(dtype or x.dtype)


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (
                2 * math.log(base)
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """In place on ``x`` ([b, s, d] or [b, s, h, d]), as the reference."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def act_quant_(x: torch.Tensor) -> None:
    x.copy_(act_quant_qdq_fp8(x, 32))


def fp4_act_quant_(x: torch.Tensor, block: int, scale: str) -> None:
    x.copy_(fp4_qdq(x, block, scale))


def canonical_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``scores.topk(k).indices`` with ties to the lowest index."""
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]


def get_window_topk_idxs(window_size, seqlen, start_pos):
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(1, -1, -1).contiguous()


def ref_sparse_attn(q, kv, attn_sink, topk_idxs, scale, block=64):
    """The reference ``sparse_attn`` kernel's arithmetic: 64-row blocks in
    index order, online softmax from a -1e30 floor, BF16 probabilities in the
    value product, fp32 running sum, sink added at the end."""
    b, s, h, d = q.shape
    k = topk_idxs.shape[-1]
    pad = -k % block
    idxs = F.pad(topk_idxs, (0, pad), value=-1).long()
    qf = q.float()
    m = torch.full((b, s, h), -1e30)
    lsum = torch.zeros((b, s, h))
    acc = torch.zeros((b, s, h, d))
    for t0 in range(0, idxs.shape[-1], block):
        ix = idxs[..., t0 : t0 + block]
        ok = ix >= 0
        rows = torch.gather(
            kv.float().unsqueeze(1).expand(b, s, -1, d),
            2,
            ix.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, d),
        )
        rows = torch.where(ok.unsqueeze(-1), rows, 0.0)
        sc = torch.einsum("bshd,bskd->bshk", qf, rows) * scale
        sc = sc.masked_fill(~ok.unsqueeze(2), float("-inf"))
        m_new = torch.maximum(m, sc.amax(-1))
        alpha = torch.exp(m - m_new)
        p = torch.exp(sc - m_new.unsqueeze(-1))
        lsum = lsum * alpha + p.sum(-1)
        acc = acc * alpha.unsqueeze(-1) + torch.einsum(
            "bshk,bskd->bshd", p.to(BF16).float(), rows
        )
        m = m_new
    lsum = lsum + torch.exp(attn_sink.float().view(1, 1, h) - m)
    return (acc / lsum.unsqueeze(-1)).to(BF16)


@dataclass
class RefArgs:
    dim: int = 64
    q_lora_rank: int = 32
    n_heads: int = 16
    head_dim: int = 512
    rope_head_dim: int = 64
    window_size: int = 16
    o_groups: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 24
    candidate_topk_blocks: int = 3
    candidate_block_size: int = 4
    norm_eps: float = 1e-20
    world_size: int = 4
    rank: int = 1
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 64
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    max_seq_len: int = 512

    def freqs_params(self, ratio: int) -> tuple:
        if ratio:
            return (
                self.rope_head_dim,
                self.original_seq_len,
                self.compress_rope_theta,
                self.rope_factor,
                self.beta_fast,
                self.beta_slow,
            )
        return (self.rope_head_dim, 0, self.rope_theta, self.rope_factor,
                self.beta_fast, self.beta_slow)


@dataclass
class LayerSpec:
    ratio: int
    kv_source: bool = False
    index_source: bool = False
    candidate_source: bool = False
    uses_candidates: bool = False


def make_weights(args: RefArgs, spec: LayerSpec, seed: int) -> dict:
    """Random global (all-rank) weights with the checkpoint's shapes."""
    g = torch.Generator().manual_seed(seed)

    def mat(o, i, scale=1.0):
        return (torch.randn(o, i, generator=g) * scale / math.sqrt(i)).to(BF16)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(BF16)

    w = {
        "wq_a": mat(args.q_lora_rank, args.dim),
        "wkv": mat(args.head_dim, args.dim),
        "q_norm": norm(args.q_lora_rank),
        "kv_norm": norm(args.head_dim),
        "wq_b": mat(args.n_heads * args.head_dim, args.q_lora_rank),
        "attn_sink": torch.randn(args.n_heads, generator=g),
    }
    if spec.kv_source:
        w["c_wkv"] = mat(args.head_dim, args.dim)
        if spec.ratio > 1:
            w["c_wgate"] = mat(args.head_dim, args.dim, 3.0)
        w["c_norm"] = norm(args.head_dim)
    if spec.index_source:
        w["i_wq_b"] = mat(args.index_n_heads * args.index_head_dim, args.q_lora_rank)
        w["i_weights_proj"] = mat(args.index_n_heads, args.dim, 4.0)
        if spec.kv_source:
            w["i_wk"] = mat(args.index_head_dim, args.head_dim)
            w["i_k_norm"] = norm(args.index_head_dim)
    return w


def rank_rows(t: torch.Tensor, args: RefArgs, per_rank: int) -> torch.Tensor:
    return t[args.rank * per_rank : (args.rank + 1) * per_rank]


@dataclass
class RefShared:
    """``SharedAttentionRuntime`` of one request."""

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


class RefAttention:
    """The reference ``Attention`` of one layer, rank ``args.rank``, for one
    request (bsz 1)."""

    def __init__(self, args: RefArgs, spec: LayerSpec, w: dict):
        self.args, self.spec, self.w = args, spec, w
        self.n_local_heads = args.n_heads // args.world_size
        self.n_local_groups = args.o_groups // args.world_size
        self.ratio = spec.ratio
        self.freqs_cis = precompute_freqs_cis(
            args.rope_head_dim, args.max_seq_len, *args.freqs_params(spec.ratio)[1:]
        )
        self.window_kv_cache = torch.zeros(1, args.window_size, args.head_dim, dtype=BF16)
        if spec.kv_source:
            n = args.max_seq_len // self.ratio
            self.compress_kv_cache = torch.zeros(1, n, args.head_dim, dtype=BF16)
            if self.ratio > 1:
                shape = (1, self.ratio, args.head_dim)
                self.kv_state = torch.zeros(shape)
                self.score_state = torch.full(shape, float("-inf"))
            if spec.index_source:
                self.k_cache = torch.zeros(1, n, args.index_head_dim, dtype=BF16)
        self.wq_b = rank_rows(w["wq_b"], args, self.n_local_heads * args.head_dim)
        self.attn_sink = rank_rows(w["attn_sink"], args, self.n_local_heads)

    # -- Compressor ---------------------------------------------------------
    def compressor(self, x, start_pos):
        w, eps, ratio = self.w, self.args.norm_eps, self.ratio
        bsz, seqlen, _ = x.size()
        dtype = x.dtype
        if ratio == 1:
            return ref_rms_norm(exact_linear(x, w["c_wkv"]), w["c_norm"], eps)
        x = x.float()
        kv = exact_linear(x, w["c_wkv"].float())
        score = exact_linear(x, w["c_wgate"].float())
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split(
                    [cutoff, remainder], dim=1
                )
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(
                    dim=1, keepdim=True
                )
        if not should_compress:
            return None
        return ref_rms_norm(kv.to(dtype), w["c_norm"], eps)

    # -- Indexer ------------------------------------------------------------
    def indexer(self, x, qr, latent, start_pos, offset, shared: RefShared):
        a, w, ratio = self.args, self.w, self.ratio
        bsz, seqlen, _ = x.size()
        rd, end_pos = a.rope_head_dim, start_pos + seqlen
        if self.spec.kv_source and latent is not None:
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = ref_rms_norm(exact_linear(latent, w["i_wk"]), w["i_k_norm"], a.norm_eps)
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant_(k, 32, "e8m0")
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        if self.spec.kv_source:
            shared.index_k = self.k_cache

        index_k = shared.index_k[:bsz, : end_pos // ratio]
        hpr = a.index_n_heads // a.world_size
        total = None
        # Every rank's partial score; the all-reduce adds them in rank order.
        for r in range(a.world_size):
            wq = w["i_wq_b"][r * hpr * a.index_head_dim : (r + 1) * hpr * a.index_head_dim]
            q = exact_linear(qr, wq).unflatten(-1, (hpr, a.index_head_dim))
            apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
            fp4_act_quant_(q, 32, "e8m0")
            wp = w["i_weights_proj"][r * hpr : (r + 1) * hpr]
            weights = (
                exact_linear(x, wp).float()
                * torch.tensor(a.index_head_dim**-0.5 * a.index_n_heads**-0.5)
            ).to(BF16)
            score = torch.einsum("bshd,btd->bsht", q.double(), index_k.double()).to(BF16)
            score = (score.relu() * weights.unsqueeze(-1)).float()
            part = score[:, :, 0]
            for h in range(1, hpr):
                part = part + score[:, :, h]
            part = part.to(BF16)
            total = part if total is None else (total.float() + part.float()).to(BF16)
        index_score = total.float()

        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
            index_score.masked_fill_(
                torch.arange(seqlen // ratio) >= compress_lens, float("-inf")
            )
        else:
            compress_lens = end_pos // ratio
        if self.spec.candidate_source:
            shared.candidates = ref_select_candidate_blocks(
                index_score, compress_lens, a.candidate_topk_blocks, a.candidate_block_size
            )
        elif self.spec.uses_candidates:
            index_score = index_score.masked_fill(~shared.candidates, float("-inf"))
        topk = min(a.index_topk, end_pos // ratio)
        idxs = canonical_topk_indices(index_score, topk).sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    # -- Attention ----------------------------------------------------------
    def forward(self, x: torch.Tensor, start_pos: int, shared: RefShared):
        """x: [1, seqlen, dim] BF16. Returns the ``wo_a`` input
        [1, seqlen, n_local_heads, head_dim] (after its FP8 qdq)."""
        a, w = self.args, self.w
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = a.rope_head_dim
        qr = ref_rms_norm(exact_linear(x, w["wq_a"]), w["q_norm"], a.norm_eps)
        q = exact_linear(qr, self.wq_b).unflatten(-1, (self.n_local_heads, a.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # _window_kv
        win = a.window_size
        kv = ref_rms_norm(exact_linear(x, w["wkv"]), w["kv_norm"], a.norm_eps)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_(kv)
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                (
                    self.window_kv_cache[:bsz, cutoff:win],
                    self.window_kv_cache[:bsz, :cutoff],
                ) = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            window_kv = kv
        else:
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        topk_idxs = get_window_topk_idxs(win, seqlen, start_pos)
        kv = window_kv

        if self.ratio:
            # _compress_kv
            ratio = self.ratio
            offset = kv.size(1)
            compress_len = (start_pos + seqlen) // ratio
            latent = None
            if self.spec.kv_source:
                latent = self.compressor(x, start_pos)
                shared.compress_kv = self.compress_kv_cache
            if not self.spec.index_source:
                idxs = shared.topk_idxs
            elif compress_len == 0:
                idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32)
            else:
                idxs = self.indexer(x, qr, latent, start_pos, offset, shared)
            if self.spec.index_source:
                shared.topk_idxs = idxs
            if latent is not None:
                freqs = (
                    self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                    if start_pos == 0
                    else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
                )
                apply_rotary_emb(latent[..., -rd:], freqs)
                fp4_act_quant_(latent, 16, "e4m3")
                self.compress_kv_cache[
                    :bsz, start_pos // ratio : start_pos // ratio + latent.size(1)
                ] = latent
            compress_kv = shared.compress_kv[:bsz, :compress_len]
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, idxs], dim=-1)

        o = ref_sparse_attn(q, kv, self.attn_sink, topk_idxs, a.head_dim**-0.5)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        act_quant_(o)
        return o.view(bsz, seqlen, self.n_local_heads, a.head_dim)


def ref_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, float("inf"))
    k = min(topk_blocks, num_blocks)
    top = torch.sort(scores, dim=-1, descending=True, stable=True)
    ids, vals = top.indices[..., :k], top.values[..., :k]
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, vals > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# ---------------------------------------------------------------------------
# The sm_80 layer, driven like vLLM drives it
# ---------------------------------------------------------------------------


def core_config(args: RefArgs, spec: LayerSpec) -> core.Sm80AttnConfig:
    return core.Sm80AttnConfig(
        n_local_heads=args.n_heads // args.world_size,
        head_dim=args.head_dim,
        rope_dim=args.rope_head_dim,
        n_local_groups=args.o_groups // args.world_size,
        window_size=args.window_size,
        compress_ratio=spec.ratio,
        is_kv_source=spec.kv_source,
        is_index_source=spec.index_source,
        is_candidate_source=spec.candidate_source,
        uses_candidates=spec.uses_candidates,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
        candidate_topk_blocks=args.candidate_topk_blocks,
        candidate_block_size=args.candidate_block_size,
        tp_size=args.world_size,
        eps=args.norm_eps,
    )


def core_weights(args: RefArgs, spec: LayerSpec, w: dict, device) -> core.Sm80AttnWeights:
    h = args.n_heads // args.world_size
    wq_b = rank_rows(w["wq_b"], args, h * args.head_dim).to(device)
    out = core.Sm80AttnWeights(
        q_norm=w["q_norm"].to(device),
        kv_norm=w["kv_norm"].to(device),
        attn_sink=rank_rows(w["attn_sink"], args, h).float().to(device),
        freqs=core.rope_freqs(*args.freqs_params(spec.ratio), device=device),
        wq_b=lambda qr: exact_linear(qr, wq_b),
    )
    if spec.kv_source:
        out.comp_norm = w["c_norm"].to(device)
    if spec.index_source:
        i_wq_b = w["i_wq_b"].to(device)
        out.index_wq_b = lambda qr: exact_linear(qr, i_wq_b)
        if spec.kv_source:
            i_wk = w["i_wk"].to(device)
            out.wk = lambda latent: exact_linear(latent, i_wk)
            out.k_norm = w["i_k_norm"].to(device)
    return out


def layer_inputs(args: RefArgs, spec: LayerSpec, w: dict, x: torch.Tensor):
    """The projections vLLM computes before the attention region."""
    qr_kv = torch.cat([exact_linear(x, w["wq_a"]), exact_linear(x, w["wkv"])], dim=-1)
    kv_score = None
    if spec.kv_source:
        if spec.ratio > 1:
            xf = x.float()
            kv_score = torch.cat(
                [exact_linear(xf, w["c_wkv"].float()), exact_linear(xf, w["c_wgate"].float())],
                dim=-1,
            )
        else:
            kv_score = exact_linear(x, w["c_wkv"]).float()
    index_w = exact_linear(x, w["i_weights_proj"]) if spec.index_source else None
    return qr_kv, kv_score, index_w


class BlockPool:
    """Block ids handed out from a shuffled high range and recycled when
    freed, so tables are non-monotonic and ids are reused across requests."""

    def __init__(self, num_blocks: int, low: int, rng: random.Random):
        self.free = list(range(low, num_blocks))
        rng.shuffle(self.free)
        self.rng = rng

    def take(self) -> int:
        return self.free.pop()

    def give(self, block: int) -> None:
        self.free.insert(self.rng.randrange(len(self.free) + 1), block)


@dataclass
class _Req:
    rid: int
    length: int  # tokens fed so far
    swa: dict = field(default_factory=dict)  # logical block -> physical
    comp: dict = field(default_factory=dict)
    idxk: dict = field(default_factory=dict)
    ring: int = -1


class PagedRun:
    """Paged caches and per-step metadata for a stack of layers sharing one
    compressed/index-key source, as vLLM lays them out."""

    def __init__(
        self,
        args: RefArgs,
        specs: list[LayerSpec],
        device="cpu",
        swa_block: int = 8,
        comp_block_tokens: int = 16,
        num_blocks: int = 1200,
        low_block: int = 700,
        ring_capacity: int = 8,
        seed: int = 0,
    ):
        self.args, self.specs, self.device = args, specs, device
        self.rng = random.Random(seed)
        self.swa_block = swa_block
        ratio = max(s.ratio for s in specs)
        self.ratio = ratio
        self.comp_e = comp_block_tokens // max(ratio, 1)
        self.pools = {
            k: BlockPool(num_blocks, low_block, self.rng)
            for k in ("swa", "comp", "idxk", "ring")
        }
        self.swa_caches = [
            torch.zeros(num_blocks, swa_block, args.head_dim, dtype=BF16, device=device)
            for _ in specs
        ]
        self.comp = torch.zeros(
            num_blocks, self.comp_e, args.head_dim, dtype=BF16, device=device
        )
        self.idxk = torch.zeros(
            num_blocks, self.comp_e, args.index_head_dim, dtype=BF16, device=device
        )
        self.ring = torch.zeros(
            num_blocks, ring_capacity, 2 * args.head_dim, device=device
        )
        self.max_tokens = 512
        self.topk = torch.full(
            (self.max_tokens, args.index_topk), -7, dtype=torch.int32, device=device
        )
        self.candidates = torch.full(
            (self.max_tokens, args.candidate_topk_blocks), -7, dtype=torch.int32,
            device=device,
        )
        self.reqs: dict[int, _Req] = {}

    def _slot(self, table: dict, pool: str, entries: int, idx: int) -> int:
        blk = idx // entries
        if blk not in table:
            table[blk] = self.pools[pool].take()
        return table[blk] * entries + idx % entries

    def _recycle_window(self, req: _Req) -> None:
        lowest = req.length - self.args.window_size  # oldest position still read
        for blk in [b for b in req.swa if (b + 1) * self.swa_block <= lowest]:
            self.pools["swa"].give(req.swa.pop(blk))

    def build(self, sched: list[tuple[int, int]]) -> core.Sm80AttnBatch:
        """Metadata for one step: ``sched`` lists ``(request, new tokens)``."""
        a, ratio = self.args, self.ratio
        pos, req_ix, qsl = [], [], [0]
        swa_slots, swa_idx, comp_slots, idxk_slots, ring_slots = [], [], [], [], []
        reqs = []
        for i, (rid, n) in enumerate(sched):
            req = self.reqs.setdefault(rid, _Req(rid, 0))
            if ratio > 1 and req.ring < 0:
                req.ring = self.pools["ring"].take()
            reqs.append(req)
            for p in range(req.length, req.length + n):
                pos.append(p)
                req_ix.append(i)
                swa_slots.append(self._slot(req.swa, "swa", self.swa_block, p))
                lo = max(0, p - a.window_size + 1)
                row = [
                    self._slot(req.swa, "swa", self.swa_block, q)
                    for q in range(lo, p + 1)
                ]
                swa_idx.append(row + [-1] * (a.window_size - len(row)))
                if ratio and (p + 1) % ratio == 0:
                    j = p // ratio
                    comp_slots.append(self._slot(req.comp, "comp", self.comp_e, j))
                    idxk_slots.append(self._slot(req.idxk, "idxk", self.comp_e, j))
                else:
                    comp_slots.append(-1)
                    idxk_slots.append(-1)
                cap = self.ring.shape[1]
                ring_slots.append(req.ring * cap + p % cap if req.ring >= 0 else -1)
            qsl.append(len(pos))
            req.length += n
        nb = 1 + max(
            [0] + [max([*r.comp, *r.idxk], default=0) for r in reqs]
        )

        def table(attr: str) -> torch.Tensor:
            t = torch.zeros(len(reqs), nb, dtype=torch.int32)
            for i, r in enumerate(reqs):
                for blk, phys in getattr(r, attr).items():
                    t[i, blk] = phys
            return t.to(self.device)

        dev = self.device
        batch = core.Sm80AttnBatch(
            positions=torch.tensor(pos, dtype=torch.int64, device=dev),
            token_to_req=torch.tensor(req_ix, dtype=torch.int32, device=dev),
            query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=dev),
            swa_slots=torch.tensor(swa_slots, dtype=torch.int64, device=dev),
            swa_indices=torch.tensor(swa_idx, dtype=torch.int32, device=dev),
            comp_slots=torch.tensor(comp_slots, dtype=torch.int64, device=dev),
            comp_block_table=table("comp"),
            idxk_slots=torch.tensor(idxk_slots, dtype=torch.int64, device=dev),
            idxk_block_table=table("idxk"),
            ring_slots=torch.tensor(ring_slots, dtype=torch.int64, device=dev),
            max_seq_len=max(r.length for r in reqs),
        )
        self.last_reqs = reqs
        return batch

    def after_step(self) -> None:
        for req in self.last_reqs:
            self._recycle_window(req)

    def caches(self, layer: int) -> core.Sm80AttnCaches:
        spec = self.specs[layer]
        return core.Sm80AttnCaches(
            swa=self.swa_caches[layer],
            topk=self.topk,
            comp=self.comp if spec.ratio else None,
            idxk=self.idxk if spec.ratio else None,
            ring=self.ring if spec.ratio > 1 else None,
            candidates=self.candidates,
        )

    def window_row(self, layer: int, rid: int, p: int) -> torch.Tensor:
        req = self.reqs[rid]
        blk = req.swa[p // self.swa_block]
        return self.swa_caches[layer][blk, p % self.swa_block]

    def comp_row(self, rid: int, j: int, idxk: bool = False) -> torch.Tensor:
        req = self.reqs[rid]
        table = req.idxk if idxk else req.comp
        cache = self.idxk if idxk else self.comp
        return cache[table[j // self.comp_e], j % self.comp_e]


# ---------------------------------------------------------------------------
# Captured inputs (RUNBOOK 01 step 4a)
# ---------------------------------------------------------------------------


def reference_from_capture(path: str) -> dict:
    """Recompute one captured sm_80 attention call in fp32 from its inputs.

    The capture is a ``torch.save`` dict with: ``config`` (the fields of
    ``Sm80AttnConfig``), ``batch`` (the fields of ``Sm80AttnBatch``),
    ``swa_cache_before`` / ``comp_cache_after`` / ``idxk_cache_after`` (the
    cache tensors; the compressed and index-key caches after the layer's own
    writes), ``q`` ([T, H, D] the post-RoPE query), ``topk`` ([T, k] local
    compressed positions the layer attended to, -1 padded), ``attn_sink``,
    ``kv_window`` ([T, D] the rows the layer wrote to its window cache) and
    ``out`` ([T, H, D] the ``wo_a`` input the layer produced) plus ``freqs``.

    Returns ``{"out": recomputed wo_a input, "max_abs_diff": ...,
    "row_mismatch": [T] bool}``: the window cache is updated with the
    captured rows, attention is recomputed by ``ref_sparse_attn`` over
    exactly the captured window slots and selected compressed positions,
    then the inverse RoPE and FP8 qdq are applied -- padding rows included,
    so a padding row that is not zero shows up too.
    """
    cap = torch.load(path, map_location="cpu")
    cfg = core.Sm80AttnConfig(**cap["config"])
    batch = core.Sm80AttnBatch(**cap["batch"])
    swa = cap["swa_cache_before"].clone()
    core.store_rows(swa, batch.swa_slots, cap["kv_window"])
    q = cap["q"]
    T = q.shape[0]
    outs = []
    for t in range(T):
        rows, idx = [], []
        for slot in batch.swa_indices[t].tolist():
            if slot >= 0:
                rows.append(swa[slot // swa.shape[1], slot % swa.shape[1]])
        if cfg.compress_ratio:
            r = int(batch.token_to_req[t])
            for j in cap["topk"][t].tolist():
                if j >= 0:
                    rows.append(
                        core.gather_compressed_rows(
                            cap["comp_cache_after"],
                            batch.comp_block_table,
                            torch.tensor([r]),
                            torch.tensor([j]),
                        )[0]
                    )
        kv = torch.stack(rows) if rows else torch.zeros(1, cfg.head_dim, dtype=BF16)
        idx = torch.arange(len(rows)).int().view(1, 1, -1) if rows else (
            torch.full((1, 1, 1), -1, dtype=torch.int32)
        )
        outs.append(
            ref_sparse_attn(
                q[t].view(1, 1, *q.shape[1:]), kv.unsqueeze(0), cap["attn_sink"],
                idx, cfg.softmax_scale,
            )[0, 0]
        )
    o = torch.stack(outs)
    o = core.output_qdq(o, batch.positions, cap["freqs"], cfg.n_local_groups)
    got = cap["out"][:T]
    diff = (o.float() - got.float()).abs()
    return {
        "out": o,
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
        "row_mismatch": (diff > 0).flatten(1).any(-1),
        "padding_nonzero": bool(cap["out"][T:].abs().sum() > 0),
    }


if __name__ == "__main__":
    import sys

    res = reference_from_capture(sys.argv[1])
    print(
        f"max_abs_diff={res['max_abs_diff']:.3g} "
        f"rows_differing={int(res['row_mismatch'].sum())}/{res['row_mismatch'].numel()} "
        f"padding_nonzero={res['padding_nonzero']}"
    )
