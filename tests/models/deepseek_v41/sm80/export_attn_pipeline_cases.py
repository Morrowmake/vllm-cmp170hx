# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export one coupled hybrid capture into served_cases/1, with byte-linked wo_a.

No tensor slicing/padding or invented source state. Missing dimensions and
consumers refuse the export. A declared non-empty manifest is required by dry
preflight before any captured validation row may run.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path

import torch


OUTPUTS = ('qr', 'query', 'latent', 'index_query', 'scaled_index_weights',
           'scores', 'swa', 'comp', 'idxk', 'ring', 'topk', 'candidates', 'out')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bitwise(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and torch.equal(
        x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def source_records(root):
    seen_paths, seen_hashes = set(), set()
    for index in sorted(root.glob('r*/index.jsonl')):
        for line in index.read_text().splitlines():
            rec = json.loads(line)
            if rec.get('point') != 'attn_pipeline' and not (
                    rec.get('point') == 'hybrid_linear' and '.wo_a.grouped.' in rec.get('prefix', '')):
                continue
            if not rec.get('saved'):
                raise ValueError(f'capture quota omitted a required source: {rec}')
            path = Path(rec['path'])
            if path in seen_paths or rec['sha256'] in seen_hashes:
                raise ValueError(f'duplicate captured ID/hash: {path}')
            seen_paths.add(path)
            seen_hashes.add(rec['sha256'])
            if digest(path) != rec['sha256']:
                raise ValueError(f'capture hash changed: {path}')
            rec['rank'] = int(index.parent.name.removeprefix('r'))
            yield rec


def load(rec):
    obj = torch.load(rec['path'], map_location='cpu', weights_only=True)
    if obj.get('schema') != 'lab-point-capture/1':
        raise ValueError(f'wrong capture schema: {rec["path"]}')
    obj['meta']['context'] = rec['context']
    return obj


def save_once(path, value):
    buf = io.BytesIO()
    torch.save(value, buf)
    data = buf.getbuffer()
    if path.exists():
        if digest(path) != hashlib.sha256(data).hexdigest():
            raise ValueError(f'refusing to replace frozen source {path}')
        return
    with path.open('xb') as f:
        f.write(data)


def compact(obj):
    a = dict(obj['tensors'])
    for name, table_name, slot_name, index_name in (
            ('swa', None, 'swa_slots', 'swa_indices'),
            ('comp', 'comp_block_table', 'comp_slots', None),
            ('idxk', 'idxk_block_table', 'idxk_slots', None),
            ('ring', None, 'ring_slots', None)):
        ids = a.get('blocks.' + name)
        if ids is None:
            continue
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        first = torch.full((unique.numel(),), ids.numel(), dtype=torch.int64)
        first.scatter_reduce_(0, inverse, torch.arange(ids.numel()), reduce='amin')
        for key in (name, 'post.' + name):
            cache = a[key]
            if not bitwise(cache[first][inverse], cache):
                raise ValueError(f'aliased captured {key} blocks disagree')
            a[key] = cache[first]
        entries = a[name].shape[1]
        for key, slot in ((table_name, False), (slot_name, True), (index_name, True)):
            if key is None or a.get('batch.' + key) is None:
                continue
            value = a['batch.' + key]
            blocks = value.long().clamp_min(0) // entries if slot else value.long().clamp_min(0)
            mapped = inverse[blocks]
            if slot:
                mapped = mapped * entries + value.long().clamp_min(0) % entries
            a['batch.' + key] = torch.where(value >= 0, mapped, -1).to(value.dtype)
        a['blocks.' + name] = unique
    return dict(obj, tensors=a)


def _key(meta):
    return (meta['rank'], meta['step'], meta['owner'], meta['rows'])


def _outputs(cfg):
    names = ['qr', 'query', 'swa', 'out']
    if cfg['compress_ratio']:
        names += ['topk', 'comp']
    if cfg['is_kv_source']:
        names += ['latent']
        if cfg['compress_ratio'] == 2:
            names += ['ring']
    if cfg['is_index_source']:
        names += ['index_query', 'scaled_index_weights', 'scores', 'idxk']
    if cfg['is_candidate_source'] or cfg['uses_candidates']:
        names += ['candidates']
    return names


def export(root, out, checkpoint):
    grouped = collections.defaultdict(list)
    pipeline = []
    provenance = []
    for rec in source_records(root):
        if rec['point'] == 'attn_pipeline':
            pipeline.append(rec)
        else:
            grouped[_key(rec)].append(rec)
    if not pipeline:
        raise ValueError('no coupled attention captures; preselected r5 attention is not sufficient')
    ranks, roles, phases, contexts, ratios, kinds = set(), set(), set(), set(), set(), set()
    pools = collections.defaultdict(list)
    by_rank = collections.defaultdict(set)
    reached = collections.defaultdict(set)
    censuses = {}
    out.mkdir(parents=True, exist_ok=True)
    sources = out / 'sources'
    sources.mkdir(exist_ok=True)
    for ordinal, rec in enumerate(pipeline):
        obj = compact(load(rec))
        a, meta = obj['tensors'], obj['meta']
        cfg = meta['cfg']
        if meta['rows'] > 48 or (meta['phase'] == 'prefill' and not meta['drafter']):
            raise ValueError('capture is outside the supported scheduler-phase domain')
        required = {'qr_kv', 'qr', 'query', 'swa', 'out', 'batch.positions', 'post.swa', 'post.topk'}
        if cfg['is_kv_source']:
            required |= {'kv_score', 'latent', 'closes', 'comp', 'post.comp'}
        if cfg['compress_ratio'] == 2 and cfg['is_kv_source']:
            required |= {'ring', 'post.ring', 'batch.ring_slots'}
        if cfg['is_index_source']:
            required |= {'index_weights', 'index_query', 'scaled_index_weights', 'index_scores_unmasked', 'idxk', 'post.idxk'}
        if not required <= a.keys():
            raise ValueError(f'missing coupled fields: {sorted(required - a.keys())}')
        layer = meta['layer_prefix']
        matches = []
        for r in grouped[_key(meta)]:
            if not r['prefix'].startswith(layer + '.wo_a.grouped.'):
                continue
            o = load(r)
            if o['meta']['linear']['prefix'] == layer + '.wo_a' and bitwise(
                    o['tensors']['x'], a['out'].reshape(meta['rows'], cfg['n_local_groups'], -1)):
                matches.append((r, o))
        if len(matches) != 1:
            raise ValueError(f'{rec["path"]}: expected one byte-linked actual grouped wo_a, found {len(matches)}')
        consumer_rec, consumer = matches[0]
        if consumer['meta']['linear'] != meta['projections']['wo_a']:
            raise ValueError('attention and successor disagree on actual wo_a loader facts')
        ranks.add(meta['rank']); roles.add(bool(meta['drafter'])); phases.add(meta['phase'])
        contexts.add(meta['context']); ratios.add(cfg['compress_ratio'])
        kind = ('candidate_source' if cfg['is_candidate_source'] else 'candidate_consumer'
                if cfg['uses_candidates'] else 'index_source' if cfg['is_index_source'] else 'shared')
        kinds.add(kind)
        shape = (meta['phase'], meta['rows'], meta['rank'], meta['drafter'], meta['context'],
                 cfg['compress_ratio'], kind, meta['owner'], len(meta['query_lens']))
        by_rank[meta['rank']].add(shape)
        reached[meta['rank']].add((meta['phase'], meta['rows']))
        source = sources / f'pipeline-{ordinal}.pt'
        save_once(source, obj)
        for output in _outputs(cfg):
            pools[('attn_pipeline', output, shape)].append(str(source))
        joined = dict(obj, family='attn_wo_a_consumer', tensors=dict(a,
            wo_a_input=consumer['tensors']['x'], wo_a_output=consumer['tensors']['out']))
        cpath = sources / f'consumer-{ordinal}.pt'
        save_once(cpath, joined)
        pools[('attn_wo_a_consumer', None, shape)].append(str(cpath))
        provenance.append({'id': ordinal, 'attention': rec['path'], 'attention_sha256': rec['sha256'],
                           'wo_a': consumer_rec['path'], 'wo_a_sha256': consumer_rec['sha256']})
        if cfg['is_index_source']:
            nvis = torch.where(a['batch.swa_slots'] >= 0, (a['batch.positions'] + 1) // cfg['compress_ratio'], 0)
            select_args = dict(scores=a['index_scores_unmasked'], n_vis=nvis, k=cfg['index_topk'],
                               candidates=a['candidates'] if cfg['uses_candidates'] and not cfg['is_candidate_source'] else None,
                               block_size=cfg['candidate_block_size'])
            spath = sources / f'select-{ordinal}.pt'
            save_once(spath, dict(obj, family='index_select', tensors=select_args))
            pools[('index_select', None, shape)].append(str(spath))
    expected = [ranks == {0, 1, 2, 3}, roles == {False, True}, {'decode', 'verify', 'draft'} <= phases,
                {'short', 'long'} <= contexts, {0, 1, 2} <= ratios,
                {'candidate_source', 'candidate_consumer', 'shared'} <= kinds]
    if not all(expected):
        raise ValueError(f'incomplete coupled coverage: ranks={ranks}, roles={roles}, phases={phases}, '
                         f'contexts={contexts}, ratios={ratios}, kinds={kinds}')
    # Saved samples are not a census: every reached supported padded shape
    # must have source evidence on each rank, including eager draft calls.
    for rank in range(4):
        census_path = root / f'r{rank}' / 'rows_seen.json'
        census = json.loads(census_path.read_text())
        if census.get('schema') != 'capture-rows-seen/1':
            raise ValueError(f'invalid row census {census_path}')
        censuses[str(census_path)] = digest(census_path)
        expected_rows = set()
        for record in census['eager_calls'] + census['graph_replays']:
            if record.get('family', 'attn') != 'attn':
                continue
            phase, count = record['phase'], record['rows']
            if phase in ('decode', 'verify', 'draft') and 0 < count <= 48:
                expected_rows.add((phase, count))
        if not expected_rows or not expected_rows <= reached[rank]:
            raise ValueError(f'missing supported reached row counts r{rank}: {expected_rows - reached[rank]}')
        shapes = by_rank[rank]
        for axis, required, label in ((0, {'decode', 'verify', 'draft'}, 'phase'),
                (3, {False, True}, 'target/draft'), (4, {'short', 'long'}, 'context'),
                (5, {0, 1, 2}, 'compression ratio'),
                (6, {'candidate_source', 'candidate_consumer', 'shared'}, 'source role')):
            found = {shape[axis] for shape in shapes}
            if not required <= found:
                raise ValueError(f'missing pipeline {label} r{rank}: {required - found}')
        for draft in (False, True):
            found = {shape[4] for shape in shapes if shape[3] == draft}
            if found != {'short', 'long'}:
                raise ValueError(f'missing context coverage r{rank}, draft={draft}: {found}')
    cases = []
    for (family, output, shape), paths in sorted(pools.items(), key=str):
        cases.append(dict(id=f'{family}/{output or "out"}/' + '/'.join(map(str, shape)),
            family=family, **({'output': output} if output else {}),
            phase='prefill' if shape[0] == 'prefill' else 'decode', scheduler_phase=shape[0],
            concurrency=shape[8], rows=shape[1],
            rank=shape[2], drafter=shape[3], context=shape[4], captured_sources=paths,
            captured_sha256={path: 'sha256:' + digest(Path(path)) for path in paths}))
    for name, keep in [('attention-pipeline', lambda c: c['family'] != 'attn_wo_a_consumer'),
                       ('attention-wo-a', lambda c: c['family'] == 'attn_wo_a_consumer')]:
        doc = dict(schema='served_cases/1', checkpoint={'hybrid': str(checkpoint)},
                   cases=[c for c in cases if keep(c)], provenance=provenance, row_censuses=censuses,
                   scope='coupled preparation/cache/selection/output from the unified hybrid capture')
        path = out / (name + '.json')
        text = json.dumps(doc, indent=2) + '\n'
        if path.exists() and path.read_text() != text:
            raise ValueError(f'refusing to replace frozen case file {path}')
        if not path.exists():
            path.write_text(text)
    return cases


def validate(path):
    doc = json.loads(path.read_text())
    if doc.get('schema') != 'served_cases/1' or not doc.get('cases') or not doc.get('provenance'):
        raise ValueError('missing nonempty coupled served_cases manifest/provenance')
    ids, hashes = set(), {}
    for record in doc['provenance']:
        for role in ('attention', 'wo_a'):
            source_path = record[role]
            if source_path not in hashes:
                hashes[source_path] = digest(Path(source_path))
            if hashes[source_path] != record[role + '_sha256']:
                raise ValueError(f'changed raw capture: {source_path}')
    for source_path, expected in doc['row_censuses'].items():
        if digest(Path(source_path)) != expected:
            raise ValueError(f'changed reached-row census: {source_path}')
    for case in doc['cases']:
        if case['id'] in ids:
            raise ValueError('duplicate case ID')
        ids.add(case['id'])
        for path in case['captured_sources']:
            expected = case['captured_sha256'][path].removeprefix('sha256:')
            if path not in hashes:
                hashes[path] = digest(Path(path))
            actual = hashes[path]
            if actual != expected:
                raise ValueError(f'changed source hash: {path}')
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--validate', type=Path)
    args = parser.parse_args()
    if args.validate:
        print(f'validated {validate(args.validate)} declared captured cases')
    else:
        if not all((args.capture, args.out, args.checkpoint)):
            parser.error('--capture, --out and --checkpoint are required')
        print(f'exported {len(export(args.capture, args.out, args.checkpoint))} coupled cases')


if __name__ == '__main__':
    main()
