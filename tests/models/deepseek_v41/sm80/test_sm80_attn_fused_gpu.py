# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU checks of the fused sm_80 DeepSeek-V4.1 attention core
(``ops/sm80_attn_fused.py``, ``VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE``); skips
without CUDA.

- FP64 rule (V2) at the row counts of decode, speculative verify and draft
  steps (1 to 48), with and without compressed rows: the prologue's ``qr``
  and window rows and the attend's ``out`` against FP64 (``fp64_cases``
  families ``attn_prologue`` and ``attn_attend``) within 1.10x (mean) and
  1.25x (max) of the served path's error, exact where it is exact; every
  row gives the same bits run alone.
- The layer functions with ``fused`` run exactly the two fused calls (bit for
  bit), write only the window rows of the step's slots, and zero the padding
  rows of ``out``.
- Both calls of a layer captured in a CUDA graph replay the eager result bit
  for bit on new activations and cache rows, with no allocation growth.
- Two calls give the same bits.
The same families run on served captures with ``fp64_cases.py --cases``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_attn_fused_gpu.py
"""

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused as fused

from . import fp64_cases as F

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

BF16 = torch.bfloat16
H, D, Q, E = 16, 512, 1280, 64


def _layer_inputs(T, compressed, seed, long_ctx=False):
    """A decode-sized layer call in the served layout: ``tpr`` tokens per
    request, each attending to its request's last window rows (its own
    included) and, at compressed layers, up to 512 selected positions."""
    g = torch.Generator().manual_seed(seed)
    tpr = 6 if T % 6 == 0 else (5 if T % 5 == 0 else 1)
    R = T // tpr
    base = torch.randint(4000 if long_ctx else 40, 8000 if long_ctx else 300, (R,), generator=g)
    pos = torch.cat([base[r] + torch.arange(tpr) for r in range(R)])
    req = torch.arange(R).repeat_interleave(tpr)
    blocks_per_req = 3
    swa = (torch.randn(1 + R * blocks_per_req, E, D, generator=g) * 0.5).to(BF16)
    swa = fused_qdq_rows(swa)
    slot_of = lambda r, p: E * (1 + r * blocks_per_req) + p % (blocks_per_req * E)  # noqa: E731
    slots = torch.tensor([slot_of(int(r), int(p)) for r, p in zip(req, pos)])
    idx = torch.full((T, 128), -1, dtype=torch.int32)
    for t in range(T):
        p, r = int(pos[t]), int(req[t])
        first = max(0, p - 127)
        idx[t, : p - first + 1] = torch.tensor(
            [slot_of(r, x) for x in range(first, p + 1)], dtype=torch.int32
        )
    a = dict(
        qr_kv=(torch.randn(T, Q + D, generator=g) * 2).to(BF16),
        q_norm=(1 + 0.1 * torch.randn(Q, generator=g)).to(BF16),
        kv_norm=(1 + 0.1 * torch.randn(D, generator=g)).to(BF16),
        positions=pos,
        freqs=core.rope_freqs(64, 0, 10000.0, 1.0, 32, 1),
        swa_cache=swa,
        swa_slots=slots,
        eps=1e-20,
        q=(torch.randn(T, H, D, generator=g) * 0.3).to(BF16),
        swa_indices=idx,
        req=req.to(torch.int32),
        attn_sink=torch.randn(H, generator=g),
        scale=D**-0.5,
        n_groups=2,
        out=torch.zeros(T, H, D, dtype=BF16),
        comp_cache=None,
        comp_block_table=None,
        comp_local=None,
    )
    if compressed:
        nb = 10
        comp = fused_qdq_rows((torch.randn(R * nb, E, D, generator=g) * 0.5).to(BF16))
        table = torch.arange(R * nb, dtype=torch.int32).view(R, nb)
        local = torch.full((T, 512), -1, dtype=torch.int32)
        for t in range(T):
            n_vis = min(512, (int(pos[t]) + 1) // 4, nb * E)
            local[t, :n_vis] = torch.randperm(nb * E, generator=g)[:n_vis].sort().values
        a.update(comp_cache=comp, comp_block_table=table, comp_local=local)
    return a


def fused_qdq_rows(rows):
    """Cache rows as served: FP8 (32, UE8M0) quantize-dequantized BF16."""
    from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import act_quant_qdq_fp8

    return act_quant_qdq_fp8(rows, 32).to(BF16)


def _within_v2(call, ref_exact=None):
    cs, cm, n = F.abs_errors(call.cand, call.exact)
    rs, rm, _ = F.abs_errors(call.ref, call.exact if ref_exact is None else ref_exact)
    assert n > 0
    if rm == 0.0:
        assert cm == 0.0
    else:
        assert cs / n <= 1.10 * rs / n, (cs / n, rs / n)
        assert cm <= 1.25 * rm, (cm, rm)
    assert call.row_invariant


ROWS = (1, 5, 8, 18, 48)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("T", ROWS)
def test_fp64_within_v2_of_the_served_path(T, compressed):
    dev = torch.device("cuda")
    for seed, long_ctx in ((T, False), (100 + T, True)):
        a = _layer_inputs(T, compressed, seed, long_ctx)
        for output in F.ATTN_PROLOGUE_OUTPUTS:
            _within_v2(F.attn_prologue(a, dev, F.Source({}, {}, output)))
        b = dict(a)
        # attend reads the cache the prologue leaves
        cache = a["swa_cache"].cuda()
        core.store_rows(
            cache, a["swa_slots"].cuda(),
            core.window_kv(
                core.rms_norm(a["qr_kv"][:, Q:].cuda(), a["kv_norm"].cuda(), 1e-20),
                a["positions"].cuda(), a["freqs"].cuda(),
            ),
        )
        b["swa_cache"] = cache.cpu()
        _within_v2(F.attn_attend(b, dev, F.Source({}, {}, None)))


def _layer_objects(a, compressed, dev):
    cfg = core.Sm80AttnConfig(
        n_local_heads=H, head_dim=D, rope_dim=64, n_local_groups=2, window_size=128,
        compress_ratio=4 if compressed else 0, is_kv_source=False,
        is_index_source=False, is_candidate_source=False, uses_candidates=False,
        index_n_heads=0, index_head_dim=0, index_topk=512 if compressed else 0,
        candidate_topk_blocks=0, candidate_block_size=0, tp_size=4, eps=1e-20,
    )
    g = torch.Generator().manual_seed(9)
    w = ((torch.randn(H * D, Q, generator=g) / 40).to(BF16)).to(dev)
    weights = core.Sm80AttnWeights(
        q_norm=a["q_norm"].to(dev), kv_norm=a["kv_norm"].to(dev),
        attn_sink=a["attn_sink"].to(dev), freqs=a["freqs"].to(dev),
        wq_b=lambda x: torch.nn.functional.linear(x, w),
    )
    T = a["positions"].shape[0]
    caches = core.Sm80AttnCaches(
        swa=a["swa_cache"].to(dev),
        topk=(a["comp_local"].to(dev) if compressed
              else torch.empty(T, 0, dtype=torch.int32, device=dev)),
        comp=a["comp_cache"].to(dev) if compressed else None,
    )
    batch = core.Sm80AttnBatch(
        positions=a["positions"].to(dev), token_to_req=a["req"].to(dev),
        query_start_loc=torch.zeros(1, dtype=torch.int32, device=dev),
        swa_slots=a["swa_slots"].to(dev), swa_indices=a["swa_indices"].to(dev),
        comp_block_table=a["comp_block_table"].to(dev) if compressed else None,
    )
    return cfg, weights, caches, batch, w


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("T", (1, 8, 18, 48))
def test_layer_functions_run_the_fused_calls(T, compressed):
    dev = torch.device("cuda")
    a = _layer_inputs(T, compressed, 7)
    cfg, weights, caches, batch, w = _layer_objects(a, compressed, dev)
    before = caches.swa.clone()
    qr_kv = a["qr_kv"].to(dev)
    # padded rows past T in out must come back zero
    out = torch.full((T + 3, H, D), 7.0, dtype=BF16, device=dev)
    q = core.prepare_layer(cfg, weights, caches, batch, qr_kv, None, None, fused=True)
    core.attend_layer(cfg, weights, caches, batch, q, out[:T], fused=True)
    out_pad = torch.full((T + 3, H, D), 7.0, dtype=BF16, device=dev)
    T_pad = core.pad_batch(batch, T + 3)
    qr_kv_pad = torch.cat([qr_kv, torch.zeros(3, Q + D, dtype=BF16, device=dev)])
    caches_pad = core.Sm80AttnCaches(
        swa=before.clone(),
        topk=torch.cat([caches.topk, torch.full((3, caches.topk.shape[1]), -1,
                                                dtype=torch.int32, device=dev)]),
        comp=caches.comp,
    )
    q_pad = core.prepare_layer(cfg, weights, caches_pad, T_pad, qr_kv_pad, None, None,
                               fused=True)
    core.attend_layer(cfg, weights, caches_pad, T_pad, q_pad, out_pad, fused=True)
    assert bool((out_pad[T:] == 0).all())

    # the same two calls made directly
    cache = before.clone()
    qr = fused.prologue(qr_kv, weights.q_norm, weights.kv_norm, batch.positions,
                        weights.freqs, cache, batch.swa_slots, cfg.eps)
    want_q = torch.nn.functional.linear(qr, w).view(T, H, D)
    want = torch.empty(T, H, D, dtype=BF16, device=dev)
    fused.attend(want_q, batch.positions, weights.freqs, cache, batch.swa_indices,
                 caches.comp, batch.comp_block_table,
                 caches.topk[:T] if compressed else None, batch.token_to_req,
                 weights.attn_sink, cfg.softmax_scale, 2, want)
    assert torch.equal(q.view(torch.int16), want_q.view(torch.int16))
    assert torch.equal(caches.swa.view(torch.int16), cache.view(torch.int16))
    assert torch.equal(out[:T].view(torch.int16), want.view(torch.int16))
    # only the step's own slots changed
    changed = (caches.swa != before).flatten(1).any(1)
    s = batch.swa_slots
    allowed = torch.zeros_like(changed)
    allowed[(s[s >= 0] // E).unique()] = True
    assert not bool((changed & ~allowed).any())


@pytest.mark.parametrize("T", (1, 8, 48))
def test_graph_replay_equals_eager_without_allocation_growth(T):
    dev = torch.device("cuda")
    a = _layer_inputs(T, True, 21)
    cfg, weights, caches, batch, _ = _layer_objects(a, True, dev)
    qr_kv = a["qr_kv"].to(dev)
    out = torch.zeros(T, H, D, dtype=BF16, device=dev)
    swa0 = caches.swa.clone()

    def layer():
        q = core.prepare_layer(cfg, weights, caches, batch, qr_kv, None, None, fused=True)
        core.attend_layer(cfg, weights, caches, batch, q, out, fused=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer()  # compiles every kernel before capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer()
    torch.cuda.synchronize()
    b = _layer_inputs(T, True, 22)
    new_act, new_q_norm = b["qr_kv"].to(dev), b["q_norm"].to(dev)
    qr_kv.copy_(new_act)
    weights.q_norm.copy_(new_q_norm)
    caches.swa.copy_(swa0)
    graph.replay()
    torch.cuda.synchronize()
    got_out, got_swa = out.clone(), caches.swa.clone()
    mem = torch.cuda.memory_allocated()
    for _ in range(5):
        caches.swa.copy_(swa0)
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == mem
    caches.swa.copy_(swa0)
    out.zero_()
    layer()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), got_out.view(torch.int16))
    assert torch.equal(caches.swa.view(torch.int16), got_swa.view(torch.int16))


@pytest.mark.parametrize("T", (1, 8, 48))
def test_two_calls_give_the_same_bits(T):
    dev = torch.device("cuda")
    a = _layer_inputs(T, True, 33)
    results = []
    for _ in range(2):
        cfg, weights, caches, batch, _ = _layer_objects(a, True, dev)
        out = torch.zeros(T, H, D, dtype=BF16, device=dev)
        q = core.prepare_layer(cfg, weights, caches, batch, a["qr_kv"].to(dev), None,
                               None, fused=True)
        core.attend_layer(cfg, weights, caches, batch, q, out, fused=True)
        results.append((out, caches.swa))
    for x, y in zip(*results):
        assert torch.equal(x.view(torch.int16), y.view(torch.int16))
