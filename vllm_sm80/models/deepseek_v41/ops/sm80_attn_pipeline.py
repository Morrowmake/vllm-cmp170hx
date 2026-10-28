# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream-ordered attention preparation and exact index selection on sm_80.

All old-ring readers finish before the combined compressed/ring store. Main
and index projections remain ordinary linears. The query returned by prepare
is POST-RoPE; the ordered attention epilogue consumes it without rotating twice.
Scratch belongs to the runner workspace lane AND CUDA stream, never a layer
or a module global. Shared top-k/candidate buffers are not scratch.
"""
import torch
from triton.language.extra import libdevice

from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import qdq_fp4_tile
from vllm_sm80.models.deepseek_v41.ops.sm80_index_scores import score_tile
from vllm_sm80.models.deepseek_v41.ops.sm80_index_topk import (
    SEGMENT, SelectionWorkspace, float_key, select,
)
from vllm.triton_utils import tl, triton

# The most padded rows a call may have: the bound the selection and the
# per-stream scratch are built and validated for.
MAX_ROWS = 48


@triton.jit
def _rope_row(x, pos, freqs, D: tl.constexpr, RD: tl.constexpr):
    d = tl.arange(0, D)
    r = tl.arange(0, RD)
    a = tl.gather(x, D - RD + r, 0).to(tl.float32)
    b = tl.gather(x, D - RD + (r ^ 1), 0).to(tl.float32)
    angle = pos.to(tl.float32) * tl.load(freqs + r // 2)
    sine = libdevice.sin(angle)
    cosine = libdevice.cos(angle)
    # The reference's complex64 multiply (sm80_attn_core.rope) as CUDA
    # compiles c10::complex: real = fma(re, cos, -(im * sin)), imag =
    # fma(re, sin, im * cos) -- one product rounded, then one fused rounding.
    # Even r holds the real part (a = re, b = im), odd r the imaginary (a = im,
    # b = re).
    rotated = tl.where(r % 2 == 0,
        libdevice.fma(a, cosine, -libdevice.mul_rn(b, sine)),
        libdevice.fma(b, sine, libdevice.mul_rn(a, cosine)))
    rotated = rotated.to(tl.bfloat16).to(tl.float32)
    return tl.where(d >= D - RD,
        tl.gather(rotated, tl.maximum(d - (D - RD), 0), 0), x.to(tl.float32))


@triton.jit
def _compress_latent_kernel(raw, norm, ring, pos, req, qsl, slots, ring_slots,
                            latent, closes, eps, xs: tl.constexpr,
                            rb: tl.constexpr, re: tl.constexpr,
                            CAP: tl.constexpr, D: tl.constexpr, RATIO: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    p = tl.load(pos + t)
    slot = tl.load(slots + t)
    cur = tl.load(raw + t * xs + d).to(tl.float32)
    close = slot >= 0
    if RATIO == 2:
        close = close & ((p + 1) % 2 == 0)
        r = tl.load(req + t).to(tl.int64)
        first = t == tl.load(qsl + r)
        rs = tl.maximum(tl.load(ring_slots + t), 0).to(tl.int64)
        # Triton uses signed remainder; position zero must wrap within the ring.
        ro = (rs // CAP) * rb + ((p + CAP - 1) % CAP) * re
        if first:
            prev = tl.load(ring + ro + d)
            ps = tl.load(ring + ro + D + d)
        else:
            prev = tl.load(raw + (t - 1) * xs + d)
            ps = tl.load(raw + (t - 1) * xs + D + d)
        cs = tl.load(raw + t * xs + D + d)
        maximum = tl.maximum(ps, cs)
        pe, ce = tl.exp(ps - maximum), tl.exp(cs - maximum)
        den = pe + ce
        # Preserve the two softmax products and their addition before BF16.
        cur = libdevice.add_rn(libdevice.mul_rn(prev, tl.div_rn(pe, den)),
                              libdevice.mul_rn(cur, tl.div_rn(ce, den)))
    cur = cur.to(tl.bfloat16).to(tl.float32)
    inv = libdevice.rsqrt(tl.sum(cur * cur, 0) / D + eps)
    w = tl.load(norm + d).to(tl.float32)
    y = (w * (cur * inv)).to(tl.bfloat16)
    tl.store(latent + t * D + d, y)
    tl.store(closes + t, close)


@triton.jit
def _compressed_ring_store_kernel(latent, raw, closes, pos, req, qsl,
                                  comp, comp_slots, ring, ring_slots, freqs,
                                  xs: tl.constexpr, cb: tl.constexpr,
                                  ce: tl.constexpr, E: tl.constexpr,
                                  rb: tl.constexpr, re: tl.constexpr,
                                  CAP: tl.constexpr, D: tl.constexpr,
                                  RD: tl.constexpr, RATIO: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 0:
        if tl.load(closes + t):
            d = tl.arange(0, D)
            x = tl.load(latent + t * D + d).to(tl.float32)
            x = _rope_row(x, tl.load(pos + t) + 1 - RATIO, freqs, D, RD)
            x = tl.reshape(qdq_fp4_tile(tl.reshape(x, (D // 16, 16)), True), (D,))
            slot = tl.load(comp_slots + t).to(tl.int64)
            tl.store(comp + (slot // E) * cb + (slot % E) * ce + d, x.to(tl.bfloat16))
    elif RATIO == 2:
        # Only unique last-capacity raw rows may update the state ring.
        r = tl.load(req + t).to(tl.int64)
        end = tl.load(qsl + r + 1)
        slot = tl.load(ring_slots + t).to(tl.int64)
        if (slot >= 0) & (t >= end - CAP) & (t < end):
            d = tl.arange(0, 2 * D)
            tl.store(ring + (slot // CAP) * rb + (slot % CAP) * re + d,
                tl.load(raw + t * xs + d))


@triton.jit
def _query_key_post_kernel(q, iq, iw, k, knorm, pos, freqs, closes, kslots,
                           kcache, qo, iqo, iwo, eps, weight_scale,
                           qs: tl.constexpr, qh: tl.constexpr,
                           iqs: tl.constexpr, iqh: tl.constexpr,
                           iws: tl.constexpr, ks: tl.constexpr,
                           kb: tl.constexpr, ke: tl.constexpr, E: tl.constexpr,
                           H: tl.constexpr, IH: tl.constexpr, D: tl.constexpr,
                           ID: tl.constexpr, RD: tl.constexpr,
                           RATIO: tl.constexpr, INDEX: tl.constexpr, OWNS_K: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    p = tl.load(pos + t)
    if part < H:
        d = tl.arange(0, D)
        x = tl.load(q + t * qs + part * qh + d).to(tl.float32)
        x = _rope_row(x, p, freqs, D, RD)
        tl.store(qo + (t * H + part) * D + d, x.to(tl.bfloat16))
    elif INDEX and part < H + IH:
        h = part - H
        d = tl.arange(0, ID)
        x = tl.load(iq + t * iqs + h * iqh + d).to(tl.float32)
        x = _rope_row(x, p, freqs, ID, RD)
        x = tl.reshape(qdq_fp4_tile(tl.reshape(x, (ID // 32, 32)), False), (ID,))
        tl.store(iqo + (t * IH + h) * ID + d, x.to(tl.bfloat16))
        weight = tl.load(iw + t * iws + h).to(tl.float32) * weight_scale
        tl.store(iwo + t * IH + h, weight.to(tl.bfloat16))
    elif OWNS_K:
        if tl.load(closes + t):
            d = tl.arange(0, ID)
            x = tl.load(k + t * ks + d).to(tl.float32)
            inv = libdevice.rsqrt(tl.sum(x * x, 0) / ID + eps)
            x = (tl.load(knorm + d).to(tl.float32) * (x * inv)).to(tl.bfloat16).to(tl.float32)
            x = _rope_row(x, p + 1 - RATIO, freqs, ID, RD)
            x = tl.reshape(qdq_fp4_tile(tl.reshape(x, (ID // 32, 32)), False), (ID,))
            slot = tl.load(kslots + t).to(tl.int64)
            if slot >= 0:
                tl.store(kcache + (slot // E) * kb + (slot % E) * ke + d, x.to(tl.bfloat16))


@triton.jit(do_not_specialize=["T"])
def _visibility_kernel(pos, slots, nvis, bnvis, RATIO: tl.constexpr,
                       CB: tl.constexpr, T, B: tl.constexpr):
    t = tl.program_id(0) * B + tl.arange(0, B)
    p = tl.load(pos + t, t < T, -1)
    slot = tl.load(slots + t, t < T, -1)
    n = tl.where(slot >= 0, (p + 1) // RATIO, 0).to(tl.int32)
    tl.store(nvis + t, n, t < T)
    tl.store(bnvis + t, (n + CB - 1) // CB, t < T)


@triton.jit(do_not_specialize=["width", "bts"])
def _index_score_stats_kernel(q, w, nvis, req, cache, bt, candidates, scores,
                              maxima, hist, width, qs, qh, ws, kb, ke, bts,
                              ss: tl.constexpr, ms: tl.constexpr,
                              cs: tl.constexpr, NS: tl.constexpr,
                              E: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                              HPR: tl.constexpr, CB: tl.constexpr,
                              CK: tl.constexpr, MASK: tl.constexpr,
                              SOURCE: tl.constexpr, SEARCH_STEPS: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    visible = tl.load(nvis + t)
    histogram = tl.zeros((256,), tl.int32)
    if seg * B >= visible:
        # No visible column: the same -inf scores, block maxima and empty
        # histogram the tile loop below writes, without scoring, candidate
        # searches or histograms.
        cols = seg * B + tl.arange(0, B)
        tl.store(scores + t * ss + cols, tl.full((B,), float('-inf'), tl.float32),
                 cols < width)
        if SOURCE:
            block_ids = seg * B // CB + tl.arange(0, B // CB)
            best = tl.where(block_ids == (visible - 1) // CB, float('inf'), float('-inf'))
            tl.store(maxima + t * ms + block_ids, best, block_ids < (width + CB - 1) // CB)
        tl.store(hist + (t * NS + seg) * 256 + tl.arange(0, 256), histogram)
    else:
        for tile in range(B // 64):
            cols = seg * B + tile * 64 + tl.arange(0, 64)
            total = tl.full((64,), float('-inf'), tl.float32)
            if seg * B + tile * 64 < visible:
                total = score_tile(q, w, req, cache, bt, t, cols, visible,
                    qs, qh, ws, kb, ke, bts, E, H, D, HPR, 64)
            if MASK:
                # Candidate IDs are sorted and -1 padded. Binary search the
                # sorted valid prefix using -1 as +infinity, without a mask spool.
                block = cols // CB
                lo = tl.full((64,), 0, tl.int32)
                hi = tl.full((64,), CK, tl.int32)
                for _ in range(SEARCH_STEPS):
                    mid = (lo + hi) // 2
                    value = tl.load(candidates + t * cs + mid, mid < CK, -1)
                    value = tl.where(value < 0, 2147483647, value)
                    less = value < block
                    lo = tl.where(less, mid + 1, lo)
                    hi = tl.where(less, hi, mid)
                found = tl.load(candidates + t * cs + lo, lo < CK, -1) == block
                total = tl.where((cols < visible) & ~found, -3.4028234663852886e38, total)
            tl.store(scores + t * ss + cols, total, cols < width)
            if SOURCE:
                block_ids = (seg * B + tile * 64) // CB + tl.arange(0, 64 // CB)
                best = tl.max(tl.reshape(total, (64 // CB, CB)), 1)
                best = tl.where(block_ids == (visible - 1) // CB, float('inf'), best)
                tl.store(maxima + t * ms + block_ids, best, block_ids < (width + CB - 1) // CB)
            byte = (float_key(total) >> 24).to(tl.int32)
            histogram += tl.histogram(tl.where((cols < width) & (cols < visible), byte, 256), 256)
        tl.store(hist + (t * NS + seg) * 256 + tl.arange(0, 256), histogram)


class PipelineWorkspace:
    def __init__(self, rows, width, qdim, device):
        self.rows, self.width, self.qdim = rows, width, qdim
        bf16 = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.qr = bf16((rows, qdim))
        self.latent = bf16((rows, 512))
        self.q = bf16((rows, 16, 512))
        self.iq = bf16((rows, 32, 128))
        self.iw = bf16((rows, 32))
        self.closes = torch.empty(rows, dtype=torch.bool, device=device)
        self.nvis = torch.empty(rows, dtype=torch.int32, device=device)
        self.bnvis = torch.empty_like(self.nvis)
        self.scores = torch.empty((rows, width), dtype=torch.float32, device=device)
        self.maxima = torch.empty((rows, triton.cdiv(width, 8)), dtype=torch.float32, device=device)
        self.selection = SelectionWorkspace(rows, width, device)
        self.attention_scores = torch.empty((rows, 16, 640), dtype=torch.float32, device=device)

    @staticmethod
    def bytes(rows, width, qdim):
        return (rows * ((qdim + 512 + 16 * 512 + 32 * 128 + 32) * 2 + 9
                        + (width + triton.cdiv(width, 8) + 16 * 640) * 4)
                + SelectionWorkspace.bytes(rows, width))


def workspace(rows, width, qdim, device, owner, *, execution_stream=None):
    from vllm.v1.worker.workspace import current_workspace_manager

    stream = (torch.cuda.current_stream(device).cuda_stream
              if execution_stream is None else execution_stream)
    key = ('deepseek_attention_pipeline', owner, stream, rows, width, qdim)
    return current_workspace_manager().get_persistent_resource(key,
        lambda: PipelineWorkspace(rows, width, qdim, device))


def supports(cfg, max_rows, max_seq_len, qdim, budget):
    if cfg.compress_ratio not in (0, 1, 2):
        return 'compress ratio is not 0, 1 or 2'
    if cfg.is_index_source and (cfg.index_n_heads, cfg.index_head_dim,
            cfg.candidate_topk_blocks, cfg.candidate_block_size) != (32, 128, 2048, 8):
        return 'index layout is not (32 heads, 128 dim, 2048 blocks, 8 positions)'
    if max_rows < 1 or max_rows > MAX_ROWS or max_seq_len < 1:
        return 'row or context capacity is outside the supported bound'
    if PipelineWorkspace.bytes(max_rows, max_seq_len, qdim) > budget:
        return 'complete pipeline scratch exceeds the configured score budget'
    return None


def compress_rows(kv_score, norm_weight, ring, positions, token_to_req,
                  query_start_loc, comp_slots, ring_slots, compress_ratio,
                  head_dim, eps, *, latent, closes):
    """Read-only compressor stage; the tail store is a subsequent launch."""
    t = positions.shape[0]
    if t:
        r = ring if ring is not None else kv_score
        _compress_latent_kernel[(t,)](kv_score, norm_weight, r, positions,
            token_to_req, query_start_loc, comp_slots,
            ring_slots if ring_slots is not None else comp_slots, latent, closes,
            eps, kv_score.stride(0), r.stride(0), r.stride(1),
            ring.shape[1] if ring is not None else 1, head_dim, compress_ratio,
            enable_fp_fusion=False)
    return closes[:t], latent[:t]


def store_compressed(cfg, weights, caches, batch, raw, ws):
    t = batch.num_tokens
    if not t:
        return
    ring = caches.ring if caches.ring is not None else raw
    _compressed_ring_store_kernel[(t, 2 if cfg.compress_ratio == 2 else 1)](
        ws.latent, raw, ws.closes, batch.positions, batch.token_to_req,
        batch.query_start_loc, caches.comp, batch.comp_slots, ring,
        batch.ring_slots if batch.ring_slots is not None else batch.comp_slots,
        weights.freqs, raw.stride(0), caches.comp.stride(0), caches.comp.stride(1),
        caches.comp.shape[1], ring.stride(0), ring.stride(1),
        caches.ring.shape[1] if caches.ring is not None else 1,
        cfg.head_dim, cfg.rope_dim, cfg.compress_ratio, enable_fp_fusion=False)


def query_key_post(cfg, weights, caches, batch, q, iq, iw, k, ws):
    t = batch.num_tokens
    if not t:
        return ws.q[:0]
    indexed = bool(cfg.compress_ratio and cfg.is_index_source)
    ih = cfg.index_n_heads if indexed else 0
    iq = iq if iq is not None else q
    iw = iw if iw is not None else q
    k = k if k is not None else q
    kc = caches.idxk if cfg.owns_k else caches.swa
    _query_key_post_kernel[(t, cfg.n_local_heads + ih + int(cfg.owns_k))](
        q, iq, iw, k, weights.k_norm if cfg.owns_k else weights.q_norm,
        batch.positions, weights.freqs, ws.closes,
        batch.idxk_slots if cfg.owns_k else batch.swa_slots, kc,
        ws.q, ws.iq, ws.iw, cfg.eps, cfg.index_weight_scale if indexed else 1.0,
        q.stride(0), q.stride(1), iq.stride(0), iq.stride(1), iw.stride(0),
        k.stride(0), kc.stride(0), kc.stride(1), kc.shape[1],
        cfg.n_local_heads, ih, cfg.head_dim, 128, cfg.rope_dim, cfg.compress_ratio,
        indexed, cfg.owns_k, enable_fp_fusion=False)
    return ws.q[:t]


def run_indexer(cfg, caches, batch, q, w, ws):
    t = batch.num_tokens
    if t == 0:
        return
    width = max(ws.width // cfg.compress_ratio, 1)
    if width > ws.width or t > ws.rows:
        raise ValueError('index call exceeds reserved pipeline workspace')
    _visibility_kernel[(triton.cdiv(t, 128),)](batch.positions, batch.swa_slots,
        ws.nvis, ws.bnvis, cfg.compress_ratio, cfg.candidate_block_size, t, 128)
    ns = triton.cdiv(width, SEGMENT)
    scores = ws.scores[:t, :width]
    candidates = caches.candidates if caches.candidates is not None else caches.topk
    _index_score_stats_kernel[(t, ns)](q, w, ws.nvis, batch.token_to_req,
        caches.idxk, batch.idxk_block_table, candidates, scores, ws.maxima,
        ws.selection.hist, width, q.stride(0), q.stride(1), w.stride(0),
        caches.idxk.stride(0), caches.idxk.stride(1), batch.idxk_block_table.stride(0),
        scores.stride(0), ws.maxima.stride(0), candidates.stride(0), ns,
        caches.idxk.shape[1], cfg.index_n_heads, cfg.index_head_dim,
        cfg.index_heads_per_rank, cfg.candidate_block_size,
        cfg.candidate_topk_blocks, cfg.uses_candidates and not cfg.is_candidate_source,
        cfg.is_candidate_source, cfg.candidate_topk_blocks.bit_length(),
        SEGMENT, num_warps=4, enable_fp_fusion=False)
    # Token selection must consume the producer histogram before the candidate
    # selector reuses it. Both use the source's unmasked scores at a source.
    select(scores, ws.nvis[:t], caches.topk[:t], ws.selection, first_histogram=True)
    if cfg.is_candidate_source:
        select(ws.maxima[:t, :triton.cdiv(width, cfg.candidate_block_size)],
            ws.bnvis[:t], caches.candidates[:t], ws.selection)


def prepare(cfg, weights, caches, batch, qr_kv, kv_score, index_weights, ws):
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused as fused

    t = batch.num_tokens
    qr = fused.fused_prologue(qr_kv, weights.q_norm, weights.kv_norm,
        batch.positions, weights.freqs, caches.swa, batch.swa_slots, cfg.eps,
        qr_out=ws.qr[:t])
    q = weights.wq_b(qr).view(t, cfg.n_local_heads, cfg.head_dim)
    iq = iw = k = None
    if cfg.is_kv_source:
        compress_rows(kv_score, weights.comp_norm, caches.ring, batch.positions,
            batch.token_to_req, batch.query_start_loc, batch.comp_slots,
            batch.ring_slots, cfg.compress_ratio, cfg.head_dim, cfg.eps,
            latent=ws.latent, closes=ws.closes)
        if cfg.owns_k:
            k = weights.wk(ws.latent[:t])
        store_compressed(cfg, weights, caches, batch, kv_score, ws)
    if cfg.compress_ratio and cfg.is_index_source:
        iq = weights.index_wq_b(qr).view(t, cfg.index_n_heads, cfg.index_head_dim)
        iw = index_weights
    q = query_key_post(cfg, weights, caches, batch, q, iq, iw, k, ws)
    if cfg.compress_ratio and cfg.is_index_source:
        run_indexer(cfg, caches, batch, ws.iq[:t], ws.iw[:t], ws)
    return q


from dataclasses import dataclass, field
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel


class PipelineWarmup(VllmJitKernel["PipelineWarmup.Key"]):
    @dataclass(frozen=True)
    class Key:
        cfg: object
        rows: int
        width: int
        qdim: int
        strides: tuple
        layer: object = field(compare=False, hash=False, repr=False)

    def get_warmup_keys(self, layer):
        caches = layer._sm80_caches()
        if not caches.swa.is_cuda or caches.swa.numel() == 0:
            return []
        strides = tuple(None if x is None else (x.shape[1], x.stride())
                        for x in (caches.swa, caches.comp, caches.idxk, caches.ring))
        return [self.Key(layer.sm80_cfg, layer._sm80_pipeline_max_rows,
                         layer.max_model_len, layer.q_lora_rank, strides, layer)]

    def compile(self, key):
        from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

        layer, cfg = key.layer, key.cfg
        actual = layer._sm80_caches()
        dev, n = actual.swa.device, key.rows
        ws = layer._sm80_pipeline_workspace(dev)
        def isolated(cache):
            if cache is None:
                return None
            blocks = 2 + triton.cdiv(n, cache.shape[1])
            return torch.empty_strided((blocks, *cache.shape[1:]), cache.stride(),
                                       dtype=cache.dtype, device=dev).zero_()
        caches = core.Sm80AttnCaches(
            swa=isolated(actual.swa), comp=isolated(actual.comp),
            idxk=isolated(actual.idxk), ring=isolated(actual.ring),
            topk=torch.full((n, cfg.index_topk), -1, dtype=torch.int32, device=dev),
            candidates=torch.full((n, cfg.candidate_topk_blocks), -1,
                                  dtype=torch.int32, device=dev))
        pos = torch.arange(n, dtype=torch.int64, device=dev) + 1
        req = torch.zeros(n, dtype=torch.int32, device=dev)
        slots = torch.arange(n, dtype=torch.int64, device=dev) + actual.swa.shape[1]
        indices = torch.full((n, cfg.window_size), -1, dtype=torch.int32, device=dev)
        indices[:, 0] = slots
        comp_slots = (torch.arange(n, dtype=torch.int64, device=dev) + caches.comp.shape[1]
                      if cfg.is_kv_source else None)
        ring_slots = (caches.ring.shape[1] + pos % caches.ring.shape[1]
                      if caches.ring is not None else None)
        bt = torch.zeros((1, triton.cdiv(key.width, caches.comp.shape[1])),
                         dtype=torch.int32, device=dev) if caches.comp is not None else None
        ibt = torch.zeros((1, triton.cdiv(key.width, caches.idxk.shape[1])),
                          dtype=torch.int32, device=dev) if caches.idxk is not None else None
        batch = core.Sm80AttnBatch(pos, req, torch.tensor([0, n], dtype=torch.int32, device=dev),
            slots, indices, comp_slots, bt,
            comp_slots if cfg.owns_k else None, ibt, ring_slots, key.width)
        weights = layer._sm80_weights(dev)
        query = torch.zeros((n, cfg.n_local_heads * cfg.head_dim), dtype=torch.bfloat16, device=dev)
        iquery = torch.zeros((n, 32 * 128), dtype=torch.bfloat16, device=dev)
        ikey = torch.zeros((n, 128), dtype=torch.bfloat16, device=dev)
        weights.wq_b = lambda x: query
        weights.index_wq_b = lambda x: iquery
        weights.wk = lambda x: ikey
        x = torch.zeros((n, key.qdim + cfg.head_dim), dtype=torch.bfloat16, device=dev)
        raw = torch.zeros((n, cfg.compress_ratio * cfg.head_dim), dtype=torch.float32, device=dev)
        iw = torch.zeros((n, 32), dtype=torch.bfloat16, device=dev)
        q = prepare(cfg, weights, caches, batch, x, raw, iw, ws)
        out = torch.empty_like(ws.q)
        core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)


PIPELINE_WARMUP = PipelineWarmup()
