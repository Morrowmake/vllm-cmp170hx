# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full preparation/cache/attention, ring barriers and graph lifetime checks.

Generated tensors complement, never replace, captured pipeline/wo_a V2 rows.
"""
import dataclasses

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
from . import fp64_cases as F
from .attn_pipeline_cases import _ideal, _write_guards
from .test_sm80_index_topk import config
from .test_sm80_attn_fused_gpu import _within_v2

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')]
BF16 = torch.bfloat16


def inputs(rows, ratio, role, device='cuda'):
    g = torch.Generator().manual_seed(100 + rows)
    def rand(shape, dtype=BF16):
        return (torch.randn(shape, generator=g) * .2).to(dtype).to(device)
    source = role != 'shared'
    cfg = config(compress_ratio=ratio, is_kv_source=bool(ratio and source),
        is_index_source=bool(ratio and source), is_candidate_source=bool(ratio and role == 'source'),
        uses_candidates=bool(ratio and role == 'consumer'))
    requests = 2 if rows >= 2 else 1
    lengths = [rows // requests, rows - rows // requests] if requests == 2 else [rows]
    pos = torch.cat([torch.arange(33 + 17 * r, 33 + 17 * r + size) for r, size in enumerate(lengths)]).long().to(device)
    req = torch.cat([torch.full((size,), r) for r, size in enumerate(lengths)]).int().to(device)
    qsl = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    entries, blocks, width, qdim = 64, 8, 512, 128
    swa = rand((1 + requests * 4, entries, 512))
    slots = (1 + req.long() * 4) * entries + pos % (4 * entries)
    wind = torch.full((rows, 128), -1, dtype=torch.int32, device=device)
    for t in range(rows):
        p, r = int(pos[t]), int(req[t])
        history = torch.arange(max(0, p - 127), p + 1, device=device)
        wind[t, :history.numel()] = (1 + r * 4) * entries + history % (4 * entries)
    table = torch.arange(1, 1 + requests * blocks, dtype=torch.int32, device=device).view(requests, blocks)
    comp = rand((1 + requests * blocks, entries, 512)) if ratio else None
    idxk = rand((1 + requests * blocks, entries, 128)) if ratio else None
    ring = rand((requests + 1, 4, 1024), torch.float32) if ratio == 2 else None
    comp_slots = ((1 + req.long() * blocks) * entries + (pos + 1) // ratio - 1) if ratio and source else None
    ring_slots = (req.long() + 1) * 4 + pos % 4 if ring is not None else None
    topk = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    if ratio:
        for t in range(rows):
            n = int((pos[t] + 1) // ratio)
            topk[t, :n] = torch.arange(n, device=device)
        candidates[:, :3] = torch.tensor([0, 2, 4], device=device)
    caches = core.Sm80AttnCaches(swa, topk, comp, idxk, ring, candidates)
    batch = core.Sm80AttnBatch(pos, req, qsl, slots, wind, comp_slots,
        table if ratio else None, comp_slots if ratio and source else None,
        table if ratio else None, ring_slots, width)
    matrices = {'wq_b': rand((16 * 512, qdim)), 'index_wq_b': rand((32 * 128, qdim)),
                'wk': rand((128, 512))}
    weights = core.Sm80AttnWeights(rand((qdim,)), rand((512,)), rand((16,), torch.float32),
        core.rope_freqs(64, 0, 10000., 1., 32, 1, device=device),
        lambda x: torch.nn.functional.linear(x, matrices['wq_b']),
        comp_norm=rand((512,)), index_wq_b=lambda x: torch.nn.functional.linear(x, matrices['index_wq_b']),
        wk=lambda x: torch.nn.functional.linear(x, matrices['wk']), k_norm=rand((128,)))
    x = rand((rows, qdim + 512))
    raw = rand((rows, max(ratio, 1) * 512), torch.float32) if cfg.is_kv_source else None
    iw = rand((rows, 32)) if cfg.is_index_source else None
    return cfg, weights, caches, batch, matrices, x, raw, iw


def clone(caches):
    return core.Sm80AttnCaches(**{f.name: None if getattr(caches, f.name) is None else getattr(caches, f.name).clone()
                                 for f in dataclasses.fields(caches)})


def snapshot(cfg, weights, caches, batch, x, raw, iw):
    a = {f.name: None if getattr(caches, f.name) is None else getattr(caches, f.name).cpu().clone()
         for f in dataclasses.fields(caches)}
    a.update({'batch.' + f.name: getattr(batch, f.name).cpu() if isinstance(getattr(batch, f.name), torch.Tensor)
              else getattr(batch, f.name) for f in dataclasses.fields(batch)})
    a.update({name: getattr(weights, name).cpu() for name in ('q_norm', 'kv_norm', 'comp_norm', 'k_norm', 'freqs', 'attn_sink')})
    a.update(qr_kv=x.cpu(), kv_score=None if raw is None else raw.cpu(),
             index_weights=None if iw is None else iw.cpu())
    return a


@pytest.mark.parametrize('rows', [0, 1, 2, 5, 6, 8, 18, 40, 48])
@pytest.mark.parametrize('ratio,role', [(0, 'shared'), (1, 'source'), (2, 'source'), (2, 'consumer'), (2, 'shared')])
def test_full_pipeline_rounding_state_and_selection(rows, ratio, role):
    cfg, weights, caches, batch, matrices, x, raw, iw = inputs(rows, ratio, role)
    before = snapshot(cfg, weights, caches, batch, x, raw, iw)
    refcache = clone(caches)
    ws = pipe.PipelineWorkspace(max(rows, 1), 512, 128, 'cuda')
    out = torch.full((rows + 3, 16, 512), 17, dtype=BF16, device='cuda')
    q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
    core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
    assert bool((out[rows:] == 0).all())
    _write_guards(before, cfg, caches, batch)
    if not rows:
        return
    qr = core.prepare_layer(cfg, weights, refcache, batch, x, raw, iw, fused=True)
    refout = torch.empty_like(out[:rows])
    core.attend_layer(cfg, weights, refcache, batch, qr, refout, fused=True)
    # The discrete/storage contracts must not be hidden by an output tolerance.
    assert F.bits_equal(caches.topk, refcache.topk)
    assert F.bits_equal(caches.candidates, refcache.candidates)
    if caches.ring is not None:
        assert F.bits_equal(caches.ring, refcache.ring)
    reference = {f.name: getattr(refcache, f.name) for f in dataclasses.fields(refcache)}
    reference['out'] = refout
    ideal = _ideal(before, F.Source({}, {}), (cfg, None, None, None, None,
                    {k: v.cpu().double() for k, v in matrices.items()}), reference)
    _within_v2(F.Call(out[:rows], refout, ideal['out'], True))
    for name in ('swa', 'comp', 'idxk'):
        if getattr(caches, name) is not None:
            _within_v2(F.Call(getattr(caches, name), getattr(refcache, name), ideal[name], True))


def _row_local(weights, matrices):
    """Projections computed one row at a time, so a row's value does not
    depend on the row count (served Marlin GEMMs tile by it)."""
    def per_row(name):
        return lambda x: torch.cat([torch.nn.functional.linear(x[i:i + 1], matrices[name])
                                    for i in range(x.shape[0])]) if x.shape[0] else x @ matrices[name].t()
    return dataclasses.replace(weights, wq_b=per_row('wq_b'), index_wq_b=per_row('index_wq_b'),
                               wk=per_row('wk'))


def _alone(caches, batch, r, s, e):
    def rows(v):
        return None if v is None else v[s:e]
    def table(v):
        return None if v is None else v[r:r + 1]
    alone = dataclasses.replace(batch, positions=batch.positions[s:e],
        token_to_req=torch.zeros(e - s, dtype=batch.token_to_req.dtype, device=batch.positions.device),
        query_start_loc=torch.tensor([0, e - s], dtype=batch.query_start_loc.dtype, device=batch.positions.device),
        swa_slots=rows(batch.swa_slots), swa_indices=rows(batch.swa_indices), comp_slots=rows(batch.comp_slots),
        comp_block_table=table(batch.comp_block_table), idxk_slots=rows(batch.idxk_slots),
        idxk_block_table=table(batch.idxk_block_table), ring_slots=rows(batch.ring_slots))
    own = clone(caches)
    own = dataclasses.replace(own, topk=own.topk[s:e].clone(), candidates=own.candidates[s:e].clone())
    return own, alone


@pytest.mark.parametrize('rows', [18, 40])
@pytest.mark.parametrize('ratio,role', [(0, 'shared'), (1, 'source'), (2, 'source'), (2, 'consumer'), (2, 'shared')])
def test_a_request_alone_matches_its_rows_of_the_batch(rows, ratio, role):
    """The pipeline's own kernels read only a row's own request: with
    row-local projections, each request replayed alone is bit-equal to its
    rows of the batch. (Captured 18-48-row calls differ alone only through
    the Marlin projections' row-count tiling, as the exp12 path does.)"""
    cfg, weights, caches, batch, matrices, x, raw, iw = inputs(rows, ratio, role)
    weights = _row_local(weights, matrices)
    pre = clone(caches)
    ws = pipe.PipelineWorkspace(rows, 512, 128, 'cuda')
    out = torch.empty((rows, 16, 512), dtype=BF16, device='cuda')
    q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
    core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
    qsl = batch.query_start_loc.tolist()
    for r in range(len(qsl) - 1):
        s, e = qsl[r], qsl[r + 1]
        own, alone = _alone(pre, batch, r, s, e)
        ws1 = pipe.PipelineWorkspace(e - s, 512, 128, 'cuda')
        out1 = torch.empty((e - s, 16, 512), dtype=BF16, device='cuda')
        q1 = core.prepare_layer(cfg, weights, own, alone, x[s:e], None if raw is None else raw[s:e],
                                None if iw is None else iw[s:e], pipeline=True, workspace=ws1)
        core.attend_layer(cfg, weights, own, alone, q1, out1, pipeline=True, workspace=ws1)
        assert F.bits_equal(out1, out[s:e])
        assert F.bits_equal(own.topk, caches.topk[s:e])
        assert F.bits_equal(own.candidates, caches.candidates[s:e])


def capture_dummy_inputs(rows, ratio, role, slots_mode, device='cuda'):
    """``inputs`` with what the runner's graph-capture warm-up passes: every
    position 0 (input_batch zeros them), all-zero block tables at the
    capacity width (block_table zeros the dummy table), one token per
    request, the window holding only the token's own slot, and slot
    mappings either all padding (-1) or the null block's row 0."""
    cfg, weights, caches, batch, matrices, x, raw, iw = inputs(rows, ratio, role, device)
    dev = x.device
    width = 262144
    pos = torch.zeros(rows, dtype=torch.int64, device=dev)
    req = torch.arange(rows, dtype=torch.int32, device=dev)
    qsl = torch.arange(rows + 1, dtype=torch.int32, device=dev)
    slot = torch.full((rows,), -1 if slots_mode == 'padding' else 0, dtype=torch.int64, device=dev)
    wind = torch.full((rows, 128), -1, dtype=torch.int32, device=dev)
    if slots_mode == 'null':
        wind[:, 0] = 0
    table = torch.zeros((max(rows, 1), width // 64), dtype=torch.int32, device=dev)
    source = cfg.is_kv_source
    batch = dataclasses.replace(batch, positions=pos, token_to_req=req, query_start_loc=qsl,
        swa_slots=slot, swa_indices=wind,
        comp_slots=slot.clone() if ratio and source else None,
        comp_block_table=table if ratio else None,
        idxk_slots=slot.clone() if ratio and source else None,
        idxk_block_table=table if ratio else None,
        ring_slots=slot.clone() if caches.ring is not None else None,
        max_seq_len=width)
    return cfg, weights, caches, batch, x, raw, iw


@pytest.mark.parametrize('slots_mode', ['padding', 'null'])
@pytest.mark.parametrize('rows', [4, 6, 48])
@pytest.mark.parametrize('ratio,role', [(0, 'shared'), (1, 'source'), (2, 'source'),
                                        (2, 'consumer'), (2, 'shared')])
def test_capture_warmup_inputs_eager_then_graph(rows, ratio, role, slots_mode):
    """The production capture path with the capture warm-up's dummy inputs
    (zero positions and block tables, capacity width 262144, production
    scratch shape): the eager warm-up call, then the captured call and its
    replay. Run under compute-sanitizer memcheck, any out-of-range access in
    the pipeline kernels (prepare, index scoring, selection, attention) is an
    error; without the sanitizer the outputs must stay finite and the
    selections in range."""
    cfg, weights, caches, batch, x, raw, iw = capture_dummy_inputs(rows, ratio, role, slots_mode)
    ws = pipe.PipelineWorkspace(48, 262144, 128, 'cuda')
    out = torch.zeros((rows, 16, 512), dtype=BF16, device='cuda')

    def run():
        q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
        core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    if slots_mode == 'padding':
        # With null-block slots every row writes its window row into the same
        # null-block row and reads it back: which row's value lands there is a
        # race the dummy batch tolerates, so only the padding form repeats.
        assert F.bits_equal(out, eager)
    if ratio:
        width = 262144 // ratio
        assert bool(((caches.topk >= -1) & (caches.topk < width)).all())


def test_graph_eager_repeat_alternating_sizes_and_workspace_ownership():
    from vllm.v1.worker.workspace import WorkspaceManager

    manager = WorkspaceManager(torch.device('cuda'), num_lanes=2)
    ws = manager.get_persistent_resource('attention', lambda: pipe.PipelineWorkspace(48, 512, 128, 'cuda'))
    other = manager.get_persistent_resource('draft', lambda: pipe.PipelineWorkspace(48, 512, 128, 'cuda'))
    assert ws.q.data_ptr() != other.q.data_ptr()
    pointers = [v.data_ptr() for v in vars(ws).values() if isinstance(v, torch.Tensor)]
    for rows in (1, 48, 6, 18, 1):
        cfg, weights, caches, batch, _, x, raw, iw = inputs(rows, 2, 'source')
        state = clone(caches)
        out = torch.empty((rows, 16, 512), dtype=BF16, device='cuda')
        def reset():
            for f in dataclasses.fields(caches):
                value = getattr(caches, f.name)
                if value is not None:
                    value.copy_(getattr(state, f.name))
        def run():
            q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
            core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
        run()
        eager = out.clone()
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        allocated = torch.cuda.memory_allocated()
        for _ in range(3):
            reset()
            graph.replay()
            assert F.bits_equal(out, eager)
            assert torch.cuda.memory_allocated() == allocated
        assert pointers == [v.data_ptr() for v in vars(ws).values() if isinstance(v, torch.Tensor)]
    manager.lock()
    assert manager.get_persistent_resource('attention', lambda: None) is ws
    with pytest.raises(AssertionError, match='not allocated'):
        manager.get_persistent_resource('missing', lambda: None)


def test_forward_progress_with_delayed_producer_stream():
    cfg, weights, caches, batch, _, x, raw, iw = inputs(48, 2, 'source')
    ws = pipe.PipelineWorkspace(48, 512, 128, 'cuda')
    out = torch.empty((48, 16, 512), dtype=BF16, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1000000)
        q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
        core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert torch.isfinite(out).all()


def test_padding_no_host_reads_and_no_cross_layer_selection_reuse():
    from .test_no_host_sync import _SyncCounter

    cfg, weights, caches, batch, _, x, raw, iw = inputs(6, 2, 'source')
    before = clone(caches)
    batch = core.pad_batch(batch, 8)
    x = torch.cat((x, torch.zeros(2, x.shape[1], dtype=x.dtype, device=x.device)))
    raw = torch.cat((raw, torch.zeros(2, raw.shape[1], dtype=raw.dtype, device=raw.device)))
    iw = torch.cat((iw, torch.zeros(2, iw.shape[1], dtype=iw.dtype, device=iw.device)))
    caches.topk = torch.cat((caches.topk, torch.full((2, 512), -1, dtype=torch.int32, device='cuda')))
    caches.candidates = torch.cat((caches.candidates, torch.full((2, 2048), -1, dtype=torch.int32, device='cuda')))
    ws = pipe.PipelineWorkspace(48, 512, 128, 'cuda')
    out = torch.empty(8, 16, 512, dtype=BF16, device='cuda')
    counter = _SyncCounter()
    with counter:
        q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
        core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
    assert counter.syncs == []
    assert bool((out[6:] == 0).all())
    assert bool((caches.topk[6:] == -1).all())
    assert bool((caches.candidates[6:] == -1).all())
    assert F.bits_equal(caches.ring[0], before.ring[0])
    selected, candidates = caches.topk.clone(), caches.candidates.clone()
    # A downstream sharing layer must leave the source-owned selections live.
    shared = dataclasses.replace(cfg, is_kv_source=False, is_index_source=False,
                                 is_candidate_source=False, uses_candidates=False)
    with counter:
        q = core.prepare_layer(shared, weights, caches, batch, x, None, None,
                               pipeline=True, workspace=ws)
        core.attend_layer(shared, weights, caches, batch, q, out, pipeline=True, workspace=ws)
    assert counter.syncs == []
    assert F.bits_equal(caches.topk, selected)
    assert F.bits_equal(caches.candidates, candidates)


def test_preprofile_runner_scratch_survives_capture_streams_without_growth(monkeypatch):
    from types import SimpleNamespace
    from vllm.v1.worker import workspace as wm
    from vllm_sm80.models.deepseek_v41.sm80_attention import DeepseekV41Sm80Attention

    manager = wm.WorkspaceManager(torch.device('cuda'), num_lanes=2)
    monkeypatch.setattr(wm, 'current_workspace_manager', lambda: manager)
    layer = SimpleNamespace(_sm80_pipeline_execution_stream=None,
        _sm80_pipeline_max_rows=48, max_model_len=512, q_lora_rank=128,
        _sm80_pipeline_owner=('runner', False))
    get = lambda: DeepseekV41Sm80Attention._sm80_pipeline_workspace(layer, 'cuda')
    original = get()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # An independently executable direct caller gets disjoint storage.
        independent = pipe.workspace(48, 512, 128, 'cuda', ('runner', False))
        assert independent.scores.data_ptr() != original.scores.data_ptr()
    with wm.use_workspace_lane(1):
        assert get().scores.data_ptr() != original.scores.data_ptr()
    manager.lock()
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            # Profiling and real capture are sequential transports of the
            # already-reserved runner/lane execution context, even when locked.
            assert get() is original
        torch.cuda.current_stream().wait_stream(capture_stream)
        assert torch.cuda.memory_allocated() == allocated


def test_coupled_cache_graph_snapshot_retains_only_prewrite_rows():
    from types import SimpleNamespace
    from vllm_sm80.capture import Record
    from vllm_sm80.models.deepseek_v41.sm80_pipeline_capture import defer_cache

    cache = torch.arange(6 * 4 * 8, device='cuda').view(6, 4, 8)
    slots = torch.tensor([8, 21], device='cuda')
    batch = SimpleNamespace(comp_block_table=torch.tensor([[2, 5]], device='cuda'),
        comp_slots=slots, positions=torch.tensor([9, 10], device='cuda'),
        token_to_req=torch.tensor([0, 0], device='cuda'), swa_slots=slots)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        record = Record('attn', 'attn_pipeline', 'test', 2, {})
        record.defer(defer_cache('comp', cache, batch, 2))
        cache[slots // 4, slots % 4] += 1
    for _ in range(3):
        before = cache.cpu().clone()
        graph.replay()
        captured = record.materialize()
        ids = captured['blocks.comp']
        assert torch.equal(captured['comp'], before[ids])
        assert torch.equal(captured['post.comp'], cache.cpu()[ids])


def test_padding_ring_read_after_profile_pool_retirement(monkeypatch):
    import gc
    from vllm.v1.worker import workspace as wm

    manager = wm.WorkspaceManager(torch.device('cuda'))
    monkeypatch.setattr(wm, 'current_workspace_manager', lambda: manager)
    execution_stream = torch.cuda.current_stream().cuda_stream
    ws = pipe.workspace(48, 262144, 1280, 'cuda', 'target',
                        execution_stream=execution_stream)
    pointer = ws.latent.data_ptr()
    raw = torch.zeros((1, 1024), dtype=torch.float32, device='cuda')
    norm = torch.ones(512, dtype=BF16, device='cuda')
    pos = torch.zeros(1, dtype=torch.int64, device='cuda')
    req = torch.zeros(1, dtype=torch.int32, device='cuda')
    qsl = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    slots = torch.full((1,), -1, dtype=torch.int64, device='cuda')
    for _ in range(2):
        # Each phase owns a fresh cache and graph pool, like profiling then
        # real capture. The manager-owned pipeline scratch remains live.
        ring = torch.zeros((2, 8, 1024), dtype=torch.float32, device='cuda')
        ring[0, 7, :512] = 2
        _, expected = core.compress_rows(raw, norm, ring.clone(), pos, req,
            qsl, slots, slots, 2, 512, 1e-6)
        def run():
            pipe.compress_rows(raw, norm, ring, pos, req, qsl, slots, slots,
                2, 512, 1e-6, latent=ws.latent, closes=ws.closes)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            run()
        graph.replay()
        torch.cuda.synchronize()
        assert F.bits_equal(ws.latent[:1], expected)
        assert not bool(ws.closes[0])
        assert ws.latent.data_ptr() == pointer
        del graph, pool, ring, expected, run, stream
        gc.collect()
        torch.cuda.empty_cache()
