# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA contracts; prepared only, run explicitly on a reserved card."""
import dataclasses

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
from vllm_sm80.models.deepseek_v41.ops.sm80_index_topk import SelectionWorkspace, select
from .test_sm80_index_topk import config
from .test_sm80_kernels_gpu import _index_case, _huge_view

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')]


@pytest.mark.parametrize('width', [0, 1, 511, 512, 513, 2047, 2048, 2049, 8191, 16385, 262144])
@pytest.mark.parametrize('k', [512, 2048])
def test_radix_exact_at_widths_ties_zeros_and_sentinels(width, k):
    g = torch.Generator().manual_seed(29)
    scores = torch.randint(-8, 9, (5, width), generator=g).float()
    visible = torch.tensor([0, min(width, k - 1), min(width, k), min(width, k + 1), width])
    if width:
        scores[:, 0] = -0.0
        scores[:, width // 2] = -core.FLT_MAX
    expected = core.topk_canonical(scores, visible, k)
    workspace = SelectionWorkspace(5, max(width, 1), 'cuda')
    out = torch.empty((5, k), dtype=torch.int32, device='cuda')
    select(scores.cuda(), visible.cuda(), out, workspace)
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize('source,masked', [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('width', [1, 511, 512, 513, 16383, 16384, 16385])
def test_score_histogram_and_both_selectors_are_bit_exact(source, masked, width):
    g = torch.Generator().manual_seed(31)
    q, w, visible, req, keys, table = _index_case(g, n=4, blocks=(width + 63) // 64)
    visible[:] = torch.tensor([0, min(1, width), max(width - 1, 0), width])
    scores = core.index_scores_torch(q, w, visible, req, keys, table, width, 8)
    candidates = torch.full((4, 2048), -1, dtype=torch.int32)
    if masked:
        candidates[:, :2] = torch.tensor([0, max(0, (width - 1) // 8)])
        expected_scores = core.mask_to_candidates(scores, candidates, 8)
    else:
        expected_scores = scores
    cfg = config(compress_ratio=1, is_candidate_source=source, uses_candidates=masked)
    batch = core.Sm80AttnBatch(visible.cuda() - 1, req.cuda(), torch.zeros(1, device='cuda'),
        torch.where(visible > 0, 0, -1).cuda(), torch.empty(4, 0, dtype=torch.int32, device='cuda'),
        idxk_block_table=table.cuda(), max_seq_len=width)
    caches = core.Sm80AttnCaches(swa=torch.empty(0, device='cuda'), idxk=keys.cuda(),
        topk=torch.empty((4, 512), dtype=torch.int32, device='cuda'), candidates=candidates.cuda())
    ws = pipe.PipelineWorkspace(4, width, 1280, 'cuda')
    pipe.run_indexer(cfg, caches, batch, q.cuda(), w.cuda(), ws)
    assert torch.equal(ws.scores.cpu(), expected_scores)
    assert torch.equal(caches.topk.cpu(), core.topk_canonical(expected_scores, visible, 512))
    if source:
        assert torch.equal(caches.candidates.cpu(), core.select_candidate_blocks(scores, visible, 2048, 8))
    else:
        assert torch.equal(caches.candidates.cpu(), candidates)


@pytest.mark.parametrize('source,masked', [(True, False), (False, False), (False, True)])
def test_capacity_width_with_short_rows_is_bit_exact_on_poisoned_scratch(source, masked):
    """The served shape: capacity width (max_model_len) far above the rows'
    visible counts. Segments past a row's visible count are skipped by the
    selectors and take the producer's fast path; poisoned scratch and a NaN
    score spool must not change a byte of the scores, maxima or selections."""
    width = 262144
    g = torch.Generator().manual_seed(37)
    q, w, visible, req, keys, table = _index_case(g, n=4, blocks=(width + 63) // 64)
    visible[:] = torch.tensor([0, 5, 1033, 4097])
    scores = core.index_scores_torch(q, w, visible, req, keys, table, width, 8)
    candidates = torch.full((4, 2048), -1, dtype=torch.int32)
    if masked:
        candidates[:, :3] = torch.tensor([0, 2, 129])
        expected_scores = core.mask_to_candidates(scores, candidates, 8)
    else:
        expected_scores = scores
    cfg = config(compress_ratio=1, is_candidate_source=source, uses_candidates=masked)
    batch = core.Sm80AttnBatch(visible.cuda() - 1, req.cuda(), torch.zeros(1, device='cuda'),
        torch.where(visible > 0, 0, -1).cuda(), torch.empty(4, 0, dtype=torch.int32, device='cuda'),
        idxk_block_table=table.cuda(), max_seq_len=width)
    caches = core.Sm80AttnCaches(swa=torch.empty(0, device='cuda'), idxk=keys.cuda(),
        topk=torch.empty((4, 512), dtype=torch.int32, device='cuda'), candidates=candidates.cuda())
    ws = pipe.PipelineWorkspace(4, width, 1280, 'cuda')
    ws.scores.fill_(float('nan'))
    ws.maxima.fill_(float('nan'))
    for name in ('hist', 'totals', 'counts', 'prefixes', 'state'):
        getattr(ws.selection, name).fill_(-12345)
    pipe.run_indexer(cfg, caches, batch, q.cuda(), w.cuda(), ws)
    assert torch.equal(ws.scores.cpu(), expected_scores)
    assert torch.equal(caches.topk.cpu(), core.topk_canonical(expected_scores, visible, 512))
    if source:
        assert torch.equal(caches.candidates.cpu(), core.select_candidate_blocks(scores, visible, 2048, 8))
    else:
        assert torch.equal(caches.candidates.cpu(), candidates)


def test_score_producer_addresses_past_int32():
    g = torch.Generator().manual_seed(17)
    keys = _huge_view(3, 4, 128, 2**30 + 512)
    from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import fp4_qdq
    keys[:] = fp4_qdq(torch.randn(3, 4, 128, generator=g), 32).to(torch.bfloat16).cuda()
    q = fp4_qdq(torch.randn(2, 32, 128, generator=g), 32).to(torch.bfloat16).cuda()
    w = torch.ones((2, 32), dtype=torch.bfloat16, device='cuda')
    req = torch.zeros(2, dtype=torch.int32, device='cuda')
    visible = torch.tensor([12, 5], device='cuda')
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device='cuda')
    batch = core.Sm80AttnBatch(visible - 1, req, req, req.long(),
        torch.empty((2, 0), dtype=torch.int32, device='cuda'), idxk_block_table=table, max_seq_len=12)
    caches = core.Sm80AttnCaches(torch.empty(0, device='cuda'),
        torch.empty((2, 512), dtype=torch.int32, device='cuda'), idxk=keys,
        candidates=torch.empty((2, 2048), dtype=torch.int32, device='cuda'))
    ws = pipe.PipelineWorkspace(2, 12, 1280, 'cuda')
    pipe.run_indexer(config(compress_ratio=1), caches, batch, q, w, ws)
    expected = core.index_scores_torch(q, w, visible, req, keys, table, 12, 8)
    assert torch.equal(ws.scores, expected)
    assert torch.equal(caches.topk, core.topk_canonical(expected, visible, 512))


def test_selector_graph_replay_reuses_storage_and_does_not_mix_rows():
    width, k = 16385, 2048
    scores = torch.zeros((4, width), dtype=torch.float32, device='cuda')
    visible = torch.tensor([0, 2047, 2049, width], device='cuda')
    out = torch.empty((4, k), dtype=torch.int32, device='cuda')
    ws = SelectionWorkspace(4, width, 'cuda')
    pointers = tuple(v.data_ptr() for v in vars(ws).values() if isinstance(v, torch.Tensor))
    select(scores, visible, out, ws)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select(scores, visible, out, ws)
    allocated = torch.cuda.memory_allocated()
    for seed in range(4):
        scores.copy_(torch.randint(-5, 6, scores.shape, device='cuda').float())
        graph.replay()
        assert torch.equal(out.cpu(), core.topk_canonical(scores.cpu(), visible.cpu(), k))
        assert torch.cuda.memory_allocated() == allocated
    assert pointers == tuple(v.data_ptr() for v in vars(ws).values() if isinstance(v, torch.Tensor))
