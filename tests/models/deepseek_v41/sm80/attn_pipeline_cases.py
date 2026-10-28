# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coupled attention replay, exact selection and actual grouped output consumer.

The coupled source contains every pre-state and official projection fingerprint.
Old preselected-attention sources are accepted ONLY by the epilogue screen,
not as evidence for compressor or index selection. The FP64 chain retains all
BF16 storage boundaries, the control's exact selected IDs, and official weights.
"""
import dataclasses
import json

import torch

from tests.models.deepseek_v41.sm80 import fp64_cases as F
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

BF16 = torch.bfloat16
OUTPUTS = ('qr', 'query', 'latent', 'index_query', 'scaled_index_weights',
           'scores', 'swa', 'comp', 'idxk', 'ring', 'topk', 'candidates', 'out')


def _on(value, device):
    return value.to(device, copy=True) if isinstance(value, torch.Tensor) else value


def _objects(a, src, device):
    from tests.models.deepseek_v41.sm80.hybrid_linear_cases import prepared, reference_settings

    reference_settings()
    cfg = core.Sm80AttnConfig(**src.meta['cfg'])
    batch = core.Sm80AttnBatch(**{f.name: _on(a['batch.' + f.name], device)
                                for f in dataclasses.fields(core.Sm80AttnBatch)})
    caches = core.Sm80AttnCaches(**{f.name: _on(a.get(f.name), device)
                                  for f in dataclasses.fields(core.Sm80AttnCaches)})
    facts = src.meta['projections']
    layers, exact = {}, {}
    for name in ('wq_b', 'index_wq_b', 'wk', 'wo_a'):
        if name not in facts:
            continue
        layer, weight = prepared(src.checkpoint['hybrid'], json.dumps(facts[name], sort_keys=True), str(device))
        layers[name], exact[name] = layer, weight
    def apply(name):
        layer = layers[name]
        return lambda x: layer.quant_method.apply(layer, x, None)
    weights = core.Sm80AttnWeights(
        **{k: _on(a.get(k), device) for k in
           ('q_norm', 'kv_norm', 'comp_norm', 'k_norm', 'freqs', 'attn_sink')},
        wq_b=apply('wq_b'), index_wq_b=apply('index_wq_b') if cfg.is_index_source else None,
        wk=apply('wk') if cfg.owns_k else None)
    return cfg, weights, caches, batch, layers, exact


def _physical(a, name, tensor):
    ids = a.get('blocks.' + name)
    if ids is None:
        return tensor
    first = torch.searchsorted(ids, ids) == torch.arange(ids.numel())
    return tensor[first.to(tensor.device)]


CACHES = ('swa', 'comp', 'idxk', 'ring')


def touched(a, name, ratio):
    """The (page, entry) slots of a captured cache that the call reads or
    writes: the window rows it attends and writes, the compressed/index-key
    entries below each row's visibility (n_vis = (pos + 1) // ratio) through
    its request's block table plus the entries it writes, the ring rows of
    its requests. The captured pages also hold slots no call reads (null and
    write-guard pages, unwritten entries); those stay covered by the write
    guards' bit check and carry whatever memory the page held."""
    cache = a.get(name)
    if cache is None:
        return None
    pages, entries = cache.shape[:2]
    mask = torch.zeros(pages, entries, dtype=torch.bool)

    def mark(slots):
        slots = slots.flatten().long()
        slots = slots[slots >= 0]
        mask[slots // entries, slots % entries] = True

    if name == 'swa':
        mark(a['batch.swa_slots'])
        if a.get('batch.swa_indices') is not None:
            mark(a['batch.swa_indices'])
    elif name in ('comp', 'idxk'):
        if a.get(f'batch.{name}_slots') is not None:
            mark(a[f'batch.{name}_slots'])
        table = a.get(f'batch.{name}_block_table')
        if table is not None:
            pos, req = a['batch.positions'].long(), a['batch.token_to_req'].long()
            visible = torch.where(a['batch.swa_slots'] >= 0, (pos + 1) // max(ratio, 1), 0)
            for r in range(table.shape[0]):
                n = int(visible[req == r].max()) if bool((req == r).any()) else 0
                k = torch.arange(n)
                mark(table[r, k // entries].long() * entries + k % entries)
    elif a.get('batch.ring_slots') is not None:
        rows = a['batch.ring_slots'].long()
        mask[(rows[rows >= 0] // entries).unique()] = True
    return mask


def finite_where_read(a, cfg, name, value):
    """Whether ``value`` (one call's output ``name``) is finite where the
    call reads or writes it; caches are judged on their touched slots."""
    if name in CACHES:
        mask = touched(a, name, cfg.compress_ratio)
        value = value[mask.to(value.device)]
    return bool(torch.isfinite(value).all())


def _write_guards(a, cfg, caches, batch):
    """Every unowned slot, including null/adjacent guards, remains byte exact."""
    closes = batch.comp_slots >= 0 if cfg.is_kv_source else None
    if cfg.is_kv_source and cfg.compress_ratio == 2:
        closes &= (batch.positions + 1) % 2 == 0
    for name in ('swa', 'comp', 'idxk', 'ring'):
        value = getattr(caches, name)
        if value is None:
            continue
        before = a[name].to(value.device)
        allowed = torch.zeros(value.shape[:2], dtype=torch.bool, device=value.device)
        if name == 'swa':
            slots = batch.swa_slots
        elif name == 'comp' and cfg.is_kv_source:
            slots = torch.where(closes, batch.comp_slots, -1)
        elif name == 'idxk' and cfg.owns_k:
            slots = torch.where(closes, batch.idxk_slots, -1)
        elif name == 'ring' and cfg.is_kv_source and cfg.compress_ratio == 2:
            t = torch.arange(batch.num_tokens, device=value.device)
            end = batch.query_start_loc[batch.token_to_req.long() + 1]
            slots = torch.where((t >= end - value.shape[1]) & (t < end), batch.ring_slots, -1)
        else:
            slots = torch.empty(0, dtype=torch.int64, device=value.device)
        slots = slots[slots >= 0].long()
        allowed[slots // value.shape[1], slots % value.shape[1]] = True
        if not F.bits_equal(value[~allowed], before[~allowed]):
            raise ValueError(f'pipeline wrote outside owned {name} slots')


def _execute(a, src, device, pipeline):
    from vllm_sm80.models.deepseek_v41.ops.sm80_attn_pipeline import PipelineWorkspace

    cfg, weights, caches, batch, layers, exact = _objects(a, src, device)
    t = batch.num_tokens
    ws = PipelineWorkspace(max(t, 1), max(batch.max_seq_len, 1),
                           a['qr_kv'].shape[1] - cfg.head_dim, device) if pipeline else None
    out = torch.empty_like(a['out'], device=device)
    def run():
        q = core.prepare_layer(cfg, weights, caches, batch, _on(a['qr_kv'], device),
            _on(a.get('kv_score'), device), _on(a.get('index_weights'), device),
            fused=not pipeline and src.meta.get('fused_control', False),
            pipeline=pipeline, workspace=ws)
        core.attend_layer(cfg, weights, caches, batch, q, out,
            fused=not pipeline and src.meta.get('fused_control', False),
            pipeline=pipeline, workspace=ws)
        return q
    query = run()
    _write_guards(a, cfg, caches, batch)
    values = {f.name: getattr(caches, f.name) for f in dataclasses.fields(caches)}
    values.update(out=out, query=query)
    if pipeline:
        # Workspace rows a layer does not produce hold earlier memory: latent
        # is a kv source's, the index rows an index source's.
        values.update(qr=ws.qr[:t])
        if cfg.is_kv_source:
            values.update(latent=ws.latent[:t])
        if cfg.is_index_source:
            values.update(index_query=ws.iq[:t], scaled_index_weights=ws.iw[:t],
                          scores=ws.scores[:t, :max(batch.max_seq_len // max(cfg.compress_ratio, 1), 1)])
    elif src.meta.get('fused_control', False):
        values['query'] = core.rope(query, batch.positions, weights.freqs)
    return values, (cfg, weights, caches, batch, layers, exact)


def _ideal(a, src, objects, reference):
    from tests.models.deepseek_v41.sm80.hybrid_linear_cases import matmul64
    from tests.models.deepseek_v41.sm80.test_act_quant import ref_fp4, ref_fp8
    from tests.models.deepseek_v41.sm80.test_sm80_kernels_gpu import _gather_rows, _fp64_attention

    cfg, _, _, _, _, weight = objects
    dev = torch.device('cpu')
    batch = core.Sm80AttnBatch(**{f.name: a['batch.' + f.name]
                                for f in dataclasses.fields(core.Sm80AttnBatch)})
    caches = core.Sm80AttnCaches(**{f.name: _on(a.get(f.name), dev)
                                  for f in dataclasses.fields(core.Sm80AttnCaches)})
    t, d, pos = batch.num_tokens, cfg.head_dim, batch.positions
    qr = F._rms_fp64(a['qr_kv'][:t, :-d], a['q_norm'], cfg.eps)
    kv = F._rms_fp64(a['qr_kv'][:t, -d:], a['kv_norm'], cfg.eps).to(BF16)
    window = ref_fp8(F._rope_fp64(kv, pos, a['freqs']).to(BF16), 32).to(BF16)
    core.store_rows(caches.swa, batch.swa_slots, window)
    projected = matmul64(qr.to(BF16), weight['wq_b'], dev).to(BF16).view(t, cfg.n_local_heads, d)
    query = F._rope_fp64(projected, pos, a['freqs']).to(BF16)
    values = dict(qr=qr, query=query)
    if cfg.is_kv_source:
        raw = a['kv_score'][:t]
        closes = batch.comp_slots >= 0
        if cfg.compress_ratio == 1:
            pooled = raw[:, :d].to(BF16)
        else:
            closes &= (pos + 1) % 2 == 0
            rows = torch.arange(t)
            first = rows == batch.query_start_loc[batch.token_to_req.long()]
            prev = raw[(rows - 1).clamp_min(0)].double()
            cap = caches.ring.shape[1]
            rs = batch.ring_slots.clamp_min(0).long()
            prev[first] = caches.ring[rs[first] // cap, (pos[first] - 1) % cap].double()
            pair = torch.stack((prev[:, :d], raw[:, :d].double()), 1)
            gates = torch.stack((prev[:, d:], raw[:, d:].double()), 1).softmax(1)
            pooled = (pair * gates).sum(1).to(BF16)
            end = batch.query_start_loc[batch.token_to_req.long() + 1]
            tail = (rows >= end - cap) & (rows < end) & (batch.ring_slots >= 0)
            core.store_rows(caches.ring, torch.where(tail, batch.ring_slots, -1), raw)
        latent = F._rms_fp64(pooled, a['comp_norm'], cfg.eps)
        values['latent'] = latent
        group_pos = pos + 1 - cfg.compress_ratio
        comp = ref_fp4(F._rope_fp64(latent.to(BF16), group_pos, a['freqs']).to(BF16), 16, 'e4m3').to(BF16)
        core.store_rows(caches.comp, torch.where(closes, batch.comp_slots, -1), comp)
        if cfg.owns_k:
            key = matmul64(latent.to(BF16), weight['wk'], dev).to(BF16)
            key = F._rms_fp64(key, a['k_norm'], cfg.eps).to(BF16)
            key = ref_fp4(F._rope_fp64(key, group_pos, a['freqs']).to(BF16), 32, 'e8m0').to(BF16)
            core.store_rows(caches.idxk, torch.where(closes, batch.idxk_slots, -1), key)
    if cfg.is_index_source:
        iq = matmul64(qr.to(BF16), weight['index_wq_b'], dev).to(BF16).view(t, cfg.index_n_heads, cfg.index_head_dim)
        iq = ref_fp4(F._rope_fp64(iq, pos, a['freqs']).to(BF16), 32, 'e8m0').to(BF16)
        iw = (a['index_weights'].float() * cfg.index_weight_scale).to(BF16)
        values.update(index_query=iq, scaled_index_weights=iw)
        visible = torch.where(batch.swa_slots >= 0, (pos + 1) // cfg.compress_ratio, 0)
        width = max(batch.max_seq_len // cfg.compress_ratio, 1)
        scores = torch.full((t, width), float('-inf'), dtype=torch.float64)
        for row in range(t):
            for start in range(0, int(visible[row]), 1024):
                cols = torch.arange(start, min(start + 1024, int(visible[row])))
                keys = core.gather_compressed_rows(caches.idxk, batch.idxk_block_table,
                    torch.full_like(cols, int(batch.token_to_req[row])), cols)
                dots = iq[row].double() @ keys.double().t()
                scores[row, cols] = (dots.clamp_min(0) * iw[row].double()[:, None]).sum(0)
        if cfg.uses_candidates and not cfg.is_candidate_source:
            scores = core.mask_to_candidates(scores, a['candidates'], cfg.candidate_block_size)
        values['scores'] = scores
    # Selection is a discrete contract. Candidate must equal control exactly;
    # the continuous FP64 chain uses those same selected positions.
    caches.topk = reference['topk'].cpu()
    caches.candidates = None if reference['candidates'] is None else reference['candidates'].cpu()
    rows, valid = _gather_rows(cfg, caches, batch)
    o = _fp64_attention(query, rows, valid, a['attn_sink'], cfg.softmax_scale).to(BF16)
    o = F._rope_fp64(o, pos, a['freqs'], inverse=True).to(BF16)
    values['out'] = ref_fp8(o.reshape(t, cfg.n_local_groups, -1), 32).reshape_as(o).to(BF16)
    values.update({f.name: getattr(caches, f.name) for f in dataclasses.fields(caches)})
    return values


def _subrequest(a, request):
    qsl = a['batch.query_start_loc']
    start, end = int(qsl[request]), int(qsl[request + 1])
    if start == end:
        return None
    b = dict(a)
    token_names = ('qr_kv', 'kv_score', 'index_weights', 'topk', 'candidates', 'out',
        'batch.positions', 'batch.swa_slots', 'batch.swa_indices', 'batch.comp_slots',
        'batch.idxk_slots', 'batch.ring_slots')
    for name in token_names:
        if b.get(name) is not None:
            b[name] = b[name][start:end]
    b['batch.token_to_req'] = torch.zeros(end - start, dtype=a['batch.token_to_req'].dtype)
    b['batch.query_start_loc'] = torch.tensor([0, end - start], dtype=qsl.dtype)
    for name in ('batch.comp_block_table', 'batch.idxk_block_table'):
        if b.get(name) is not None:
            b[name] = b[name][request:request + 1]
    return b, start, end


# The last replayed calls: fp64_cases runs every output of one captured call
# (up to six coupled sources) in a row on the same loaded arguments, and the
# replay does not depend on the output judged.
_REPLAYED: dict[int, tuple] = {}


def pipeline_outputs(a, device, src):
    hit = _REPLAYED.get(id(a))
    if hit is None or hit[0] is not a:
        try:
            result = _pipeline_outputs(a, device, src)
        except ValueError as exc:
            result = exc
        hit = _REPLAYED[id(a)] = (a, result)
        while len(_REPLAYED) > 8:
            del _REPLAYED[next(iter(_REPLAYED))]
    if isinstance(hit[1], ValueError):
        raise hit[1]
    return hit[1]


def _pipeline_outputs(a, device, src):
    if device.type != 'cuda':
        raise ValueError('coupled pipeline replay requires CUDA and actual loaded linears')
    cand, objects = _execute(a, src, device, True)
    ref, _ = _execute(a, src, device, False)
    for name in ('out', 'swa', 'comp', 'idxk', 'ring', 'topk', 'candidates'):
        recorded = a.get('out' if name == 'out' else 'post.' + name)
        if recorded is not None and not F.bits_equal(_physical(a, name, ref[name]), _physical(a, name, recorded)):
            raise ValueError(f'qualified-control replay differs from capture: {name}')
    for name in ('topk', 'candidates', 'ring'):
        if ref[name] is not None and not F.bits_equal(cand[name], ref[name]):
            raise ValueError(f'unchanged pipeline boundary differs: {name}')
    ideal = _ideal(a, src, objects, ref)
    for name, recorded in (('qr', 'qr'), ('latent', 'latent'), ('index_query', 'index_query'),
                           ('scaled_index_weights', 'scaled_index_weights'), ('scores', 'index_scores_unmasked')):
        if recorded in a:
            ref[name] = a[recorded].to(device)
    for name in ('qr', 'swa'):
        if not F.bits_equal(cand[name], ref[name]):
            raise ValueError(f'unchanged seed prologue boundary differs: {name}')
    # Reference score capture is unmasked at a source and later sources mask
    # only after the producer. Keep that same policy for the numerical rows.
    cfg = objects[0]
    if cfg.uses_candidates and not cfg.is_candidate_source and 'scores' in ref:
        ref['scores'] = core.mask_to_candidates(ref['scores'], a['candidates'].to(device), cfg.candidate_block_size)
    if cfg.is_index_source:
        visible = torch.where(a['batch.swa_slots'] >= 0,
            (a['batch.positions'] + 1) // cfg.compress_ratio, 0)
        for values in (cand, ref, ideal):
            scores = values['scores']
            mask = torch.arange(scores.shape[1], device=scores.device)[None, :] < visible.to(scores.device)[:, None]
            if not torch.isfinite(scores[mask]).all():
                raise ValueError('nonfinite visible index arithmetic')
    for name in OUTPUTS:
        for values in (cand, ref, ideal):
            value = values.get(name)
            if (value is not None and name not in ('scores', 'topk', 'candidates')
                    and not finite_where_read(a, cfg, name, value)):
                raise ValueError(f'nonfinite pipeline {name}')
    invariant = True
    for request in sorted({0, a['batch.query_start_loc'].numel() - 2}):
        subset = _subrequest(a, request)
        if subset is not None:
            b, start, end = subset
            alone, _ = _execute(b, src, device, True)
            invariant &= F.bits_equal(alone['out'], cand['out'][start:end])
            invariant &= F.bits_equal(alone['topk'], cand['topk'][start:end])
    return cand, ref, ideal, invariant, objects


def attn_pipeline(a, device, src):
    if src.output not in OUTPUTS:
        raise ValueError(f'attn_pipeline needs a named output from {OUTPUTS}')
    cand, ref, ideal, invariant, objects = pipeline_outputs(a, device, src)
    name = src.output
    if any(values.get(name) is None for values in (cand, ref, ideal)):
        raise ValueError(f'output {name} is not reached by this layer')
    if name in CACHES:
        mask = touched(a, name, objects[0].compress_ratio)
        return F.Call(*[v[name][mask.to(v[name].device)] for v in (cand, ref, ideal)], invariant)
    return F.Call(*[_physical(a, name, v[name]) for v in (cand, ref, ideal)], invariant)


def attn_wo_a_consumer(a, device, src):
    from tests.models.deepseek_v41.sm80.hybrid_linear_cases import matmul64

    cand, ref, ideal, invariant, objects = pipeline_outputs(a, device, src)
    cfg, _, _, _, layers, weights = objects
    layer = layers['wo_a']
    if not hasattr(layer.quant_method, 'apply_grouped'):
        raise ValueError('wo_a consumer is not the actual grouped loader path')
    def apply(value):
        return layer.quant_method.apply_grouped(layer, value.to(device).reshape(value.shape[0], cfg.n_local_groups, -1))
    x64 = ideal['out'].reshape(ideal['out'].shape[0], cfg.n_local_groups, -1)
    y64 = matmul64(x64, weights['wo_a'], device, cfg.n_local_groups)
    y, r = apply(cand['out']), apply(ref['out'])
    if 'wo_a_input' not in a or not F.bits_equal(a['wo_a_input'], ref['out'].cpu().reshape_as(a['wo_a_input'])):
        raise ValueError('actual wo_a input is not byte-linked to the attention capture')
    if not F.bits_equal(r, a['wo_a_output']):
        raise ValueError('actual hybrid wo_a does not replay captured output')
    if not all(torch.isfinite(v).all() for v in (y, r, y64)):
        raise ValueError('nonfinite grouped attention consumer')
    return F.Call(y, r, y64, invariant)


def index_select(scores, n_vis, k, candidates=None, block_size=8):
    """Recordable selector call; production writes directly via select()."""
    from vllm_sm80.models.deepseek_v41.ops.sm80_index_topk import SelectionWorkspace, select

    if candidates is not None:
        scores = core.mask_to_candidates(scores, candidates, block_size)
    if not scores.is_cuda:
        return core.topk_canonical(scores, n_vis, k)
    out = torch.empty((scores.shape[0], k), dtype=torch.int32, device=scores.device)
    ws = SelectionWorkspace(max(scores.shape[0], 1), max(scores.shape[1], 1), scores.device)
    select(scores, n_vis, out, ws)
    return out


def index_select_case(a, device, src):
    scores, nvis = a['scores'], a['n_vis']
    visible = torch.arange(scores.shape[1])[None, :] < nvis[:, None]
    if not torch.isfinite(scores[visible]).all():
        raise ValueError('nonfinite arithmetic in visible index scores')
    args = (int(a['k']), a.get('candidates'), int(a.get('block_size', 8)))
    ref = index_select(scores, nvis, *args)
    cand = index_select(scores.to(device), nvis.to(device), args[0],
                        None if args[1] is None else args[1].to(device), args[2])
    invariant = all(F.bits_equal(cand[i:i + 1], index_select(scores[i:i + 1].to(device),
        nvis[i:i + 1].to(device), args[0], None if args[1] is None else args[1][i:i + 1].to(device), args[2]))
        for i in F.picks(scores.shape[0]))
    return F.Call(cand, ref, ref.double(), invariant)
