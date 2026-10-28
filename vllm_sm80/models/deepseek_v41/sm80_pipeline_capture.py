# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coupled attention snapshots for the existing diagnostic capture service.

Install together with hybrid_capture, never during timing. Graphs preserve only
pre-write slots. At the synchronized capture drain, unique visible cache blocks
are gathered and their pre-state reconstructed before the next execution. No
full-capacity cache shadows are retained in each captured graph.
"""
import dataclasses
import functools
from contextvars import ContextVar

import torch

_current = ContextVar('deepseek_attention_capture', default=None)
_installed = False


CACHE_FIELDS = {
    'swa': (None, 'swa_slots', 'swa_indices'),
    'comp': ('comp_block_table', 'comp_slots', None),
    'idxk': ('idxk_block_table', 'idxk_slots', None),
    'ring': (None, 'ring_slots', None),
}


def defer_cache(name, cache, batch, ratio):
    """Retain bounded pre-write rows, materialize paged state at capture drain."""
    table_name, slot_name, index_name = CACHE_FIELDS[name]
    table = getattr(batch, table_name) if table_name else None
    slots = getattr(batch, slot_name)
    indices = getattr(batch, index_name) if index_name else None
    # Advanced indexing makes one fixed-size copy, including invalid-slot
    # shadows that are discarded on the host. Ring readers are not disturbed.
    old = None
    if cache is not None and slots is not None:
        e = cache.shape[1]
        s = slots.long().clamp_min(0)
        old = cache[s // e, s % e]

    def materialize():
        out = {name: None, 'post.' + name: None, 'blocks.' + name: None}
        out.update({'batch.' + key: None for key in (table_name, slot_name, index_name) if key})
        if cache is None:
            return out
        e = cache.shape[1]
        bt = None if table is None else table.cpu().clone()
        ss = None if slots is None else slots.cpu()
        ii = None if indices is None else indices.cpu()
        if bt is not None:
            pos, req, valid = batch.positions.cpu(), batch.token_to_req.cpu().long(), batch.swa_slots.cpu() >= 0
            visible = torch.where(valid, (pos + 1) // ratio, 0).long()
            per_req = torch.zeros(bt.shape[0], dtype=torch.int64)
            per_req.scatter_reduce_(0, req, visible, reduce='amax')
            used = torch.arange(bt.shape[1])[None, :] < ((per_req + e - 1) // e)[:, None]
            # Invisible table entries are never read; map them to the captured
            # null block rather than copying the entire configured capacity.
            bt = torch.where(used, bt, 0)
        blocks = [torch.zeros(1, dtype=torch.int64)]
        if bt is not None:
            blocks.append(bt.flatten().long().clamp_min(0))
        for value in (ss, ii):
            if value is not None:
                blocks.append(value.flatten().long().clamp_min(0) // e)
        if ss is not None:
            blocks.append((ss.flatten().long().clamp_min(0) // e + 1) % cache.shape[0])
        ids = torch.unique(torch.cat(blocks), sorted=True)
        after = cache[ids.to(cache.device)].cpu()
        before = after.clone()
        def remap(value, is_slot):
            if value is None:
                return None
            block = value.long().clamp_min(0) // e if is_slot else value.long().clamp_min(0)
            mapped = torch.searchsorted(ids, block.contiguous())
            if is_slot:
                mapped = mapped * e + value.long().clamp_min(0) % e
            return torch.where(value >= 0, mapped, -1).to(value.dtype)
        mapped_slots = remap(ss, True)
        if old is not None:
            valid = ss >= 0
            dst = mapped_slots[valid].long()
            before[dst // e, dst % e] = old.cpu()[valid]
        out.update({name: before, 'post.' + name: after, 'blocks.' + name: ids})
        for key, value in ((table_name, remap(bt, False)), (slot_name, mapped_slots),
                           (index_name, remap(ii, True))):
            if key:
                out['batch.' + key] = value
        return out
    return materialize


def install():
    global _installed
    if _installed:
        return
    from vllm_sm80 import capture
    if not capture.active:
        return
    from vllm.forward_context import get_forward_context
    from vllm_sm80.models.deepseek_v41 import sm80_attention
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    _installed = True
    cls = sm80_attention.DeepseekV41Sm80Attention
    original = cls._sm80_attention

    @functools.wraps(original)
    def attention(self, qr_kv, kv_score, index_weights, positions, out):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        # The control is captured, the new body is replayed offline. This also
        # retains the independent score/latent boundaries, not a preselected
        # compressed list masquerading as full pipeline input.
        if self._sm80_attn_pipeline_enabled:
            raise ValueError('coupled capture requires the qualified control arm')
        if qr_kv.shape[0] > self._sm80_pipeline_max_rows:
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        if not self._sm80_is_draft_layer and sm80_attention.prefill_step(self._sm80_max_decode_query_len):
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        rec = capture.begin('attn', 'attn_pipeline', self.prefix + '.pipeline',
                            qr_kv.shape[0], cfg=dataclasses.asdict(self.sm80_cfg),
                            layer_prefix=self.prefix, drafter=self._sm80_is_draft_layer,
                            point_phase='draft' if self._sm80_is_draft_layer else None)
        if rec is None:
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        batch = dataclasses.replace(core.pad_batch(self._sm80_batch(metadata, positions), qr_kv.shape[0]))
        caches = self._sm80_caches()
        weights = self._sm80_weights(qr_kv.device)
        projections = {'wq_b': self.wq_b, 'wo_a': self.wo_a}
        if self.indexer is not None:
            projections['index_wq_b'] = self.indexer.wq_b
            if self.indexer.owns_k:
                projections['wk'] = self.indexer.wk
        facts = {}
        for name, module in projections.items():
            fact = getattr(module, '_hybrid_capture', None)
            if fact is None:
                raise ValueError(f'coupled attention capture lacks loaded projection facts: {name}')
            facts[name] = fact
        rec.meta['projections'] = facts
        rec.meta['fused_control'] = self._sm80_fused_attn_call(qr_kv.shape[0])
        rec.put(qr_kv=qr_kv, kv_score=kv_score, index_weights=index_weights,
                topk=caches.topk[:batch.num_tokens],
                candidates=None if caches.candidates is None else caches.candidates[:batch.num_tokens])
        deferred_fields = {key for fields in CACHE_FIELDS.values() for key in fields if key}
        for name in CACHE_FIELDS:
            rec.defer(defer_cache(name, getattr(caches, name), batch, self.sm80_cfg.compress_ratio))
        rec.put(**{f'batch.{f.name}': getattr(batch, f.name) for f in dataclasses.fields(batch)
                   if f.name not in deferred_fields})
        for name in ('q_norm', 'kv_norm', 'comp_norm', 'k_norm', 'freqs', 'attn_sink'):
            rec.put(**{name: getattr(weights, name)})
        token = _current.set(rec)
        try:
            result = original(self, qr_kv, kv_score, index_weights, positions, out)
            rec.put(out=out, **{'post.topk': caches.topk[:batch.num_tokens],
                'post.candidates': None if caches.candidates is None else caches.candidates[:batch.num_tokens]})
            rec.done()
            return result
        finally:
            _current.reset(token)

    cls._sm80_attention = attention

    def wrap(name, outputs, module=core):
        fn = getattr(module, name)
        @functools.wraps(fn)
        def record(*args, **kwargs):
            result = fn(*args, **kwargs)
            rec = _current.get()
            if rec is not None:
                values = result if isinstance(result, tuple) else (result,)
                rec.put(**dict(zip(outputs, values)))
            return result
        setattr(module, name, record)

    wrap('compress_rows', ('closes', 'latent'))
    wrap('index_query', ('index_query', 'scaled_index_weights'))
    wrap('index_key_rows', ('index_keys',))
    wrap('index_scores', ('index_scores_unmasked',))
    wrap('prepare_layer', ('query',))
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused
    wrap('prologue', ('qr',), sm80_attn_fused)
