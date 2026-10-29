# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA half of ``test_sm80_select_one_launch.py``; prepared only, run
explicitly on a reserved card.

- ``select`` with ``VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH`` writes the bytes of
  the fifteen-launch path (and ``topk_canonical``'s) over the served capacity
  width, for one and two-level rows, ``k`` 512 and 2048, with and without the
  producer's first histogram, on poisoned scratch.
- ``run_indexer`` gives the same scores, token ids and candidate blocks with
  the switch on as off, at every index role.
- A captured graph of the one-launch call replays with new scores and visible
  counts (its tickets reset themselves) and stays equal to the other path.
"""
import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
from vllm_sm80.models.deepseek_v41.ops import sm80_index_topk as topk
from .test_sm80_index_topk import config
from .test_sm80_kernels_gpu import _index_case
from .test_sm80_select_one_launch import FLAG, POISON, producer_histogram

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
CAPACITY = 262144


def _both(monkeypatch, scores, visible, k, ws, first):
    got = []
    for arm in ('0', '1'):
        monkeypatch.setenv(FLAG, arm)
        for name in ('totals', 'counts', 'prefixes', 'state', 'cand_keys', 'cand_cols', 'kept'):
            getattr(ws, name).fill_(POISON)
        out = torch.full((scores.shape[0], k), POISON, dtype=torch.int32, device='cuda')
        if first:
            ns = triton_cdiv(scores.shape[1], topk.SEGMENT)
            rows = scores.shape[0]
            ws.hist.view(-1)[:rows * ns * 256] = producer_histogram(
                scores.cpu(), visible.cpu(), scores.shape[1], topk.SEGMENT).view(-1).cuda()
        else:
            ws.hist.fill_(POISON)
        topk.select(scores, visible, out, ws, first_histogram=first)
        got.append(out.cpu())
        assert bool((ws.tickets == 0).all())
    return got


def triton_cdiv(a, b):
    return -(-max(a, 1) // b)


@pytest.mark.parametrize('first', [False, True])
@pytest.mark.parametrize('k', [512, 2048])
@pytest.mark.parametrize('width', [1, 511, 2049, 8191, 8192, 8193, 16385, 65537, CAPACITY])
def test_one_launch_bytes_over_capacity_width(monkeypatch, width, k, first):
    if k == 2048 and triton_cdiv(width, topk.PART) * k > topk.SelectionWorkspace(1, CAPACITY, 'cpu').slots:
        pytest.skip('more candidate slots than the served workspace holds')
    g = torch.Generator().manual_seed(width + k)
    rows = 6
    scores = torch.randint(-8, 9, (rows, width), generator=g).float()
    scores[1] = torch.randn(width, generator=g)
    scores[2, ::3] = float('-inf')
    if width > 1:
        scores[3, : width // 2] = -0.0
        scores[4, width // 2] = -core.FLT_MAX
    visible = torch.tensor([width, width, max(width - 1, 0), min(width, k - 1),
                            min(width, topk.PART + 1), min(width, 3 * topk.PART + 5)],
                           dtype=torch.int32)
    col = torch.arange(width)[None, :]
    scores = torch.where(col < visible[:, None].long(), scores, float('nan'))
    ws = topk.SelectionWorkspace(48, CAPACITY, 'cuda')
    off, on = _both(monkeypatch, scores.cuda(), visible.cuda(), k, ws, first)
    clean = torch.where(col < visible[:, None].long(), scores, float('-inf'))
    assert torch.equal(off, core.topk_canonical(clean, visible.long(), k))
    assert torch.equal(on, off)


@pytest.mark.parametrize('source,masked', [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('width', [513, 16385])
def test_run_indexer_same_bytes(monkeypatch, source, masked, width):
    g = torch.Generator().manual_seed(43)
    q, w, visible, req, keys, table = _index_case(g, n=4, blocks=(width + 63) // 64)
    visible[:] = torch.tensor([0, 5, min(width, 9000), width])
    candidates = torch.full((4, 2048), -1, dtype=torch.int32)
    if masked:
        candidates[:, :3] = torch.tensor([0, 1, (width - 1) // 8])
    cfg = config(compress_ratio=1, is_candidate_source=source, uses_candidates=masked)
    got = []
    for arm in ('0', '1'):
        monkeypatch.setenv(FLAG, arm)
        batch = core.Sm80AttnBatch(visible.cuda() - 1, req.cuda(), torch.zeros(1, device='cuda'),
            torch.where(visible > 0, 0, -1).cuda(), torch.empty(4, 0, dtype=torch.int32, device='cuda'),
            idxk_block_table=table.cuda(), max_seq_len=width)
        caches = core.Sm80AttnCaches(swa=torch.empty(0, device='cuda'), idxk=keys.cuda(),
            topk=torch.full((4, 512), POISON, dtype=torch.int32, device='cuda'),
            candidates=candidates.cuda())
        ws = pipe.PipelineWorkspace(4, CAPACITY, 1280, 'cuda')
        pipe.run_indexer(cfg, caches, batch, q.cuda(), w.cuda(), ws)
        got.append((ws.scores[:4, :width].cpu(), caches.topk.cpu(), caches.candidates.cpu()))
    for a, b in zip(*got):
        assert torch.equal(a, b)


def test_one_launch_graph_replay(monkeypatch):
    monkeypatch.setenv(FLAG, '1')
    width, k, rows = 40000, 512, 6
    scores = torch.zeros((rows, width), dtype=torch.float32, device='cuda')
    visible = torch.zeros(rows, dtype=torch.int32, device='cuda')
    out = torch.empty((rows, k), dtype=torch.int32, device='cuda')
    ws = topk.SelectionWorkspace(48, CAPACITY, 'cuda')
    topk.select(scores, visible, out, ws)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk.select(scores, visible, out, ws)
    g = torch.Generator().manual_seed(47)
    for step in range(6):
        scores.copy_(torch.randint(-5, 6, (rows, width), generator=g).float())
        visible.copy_(torch.randint(0, width + 1, (rows,), generator=g, dtype=torch.int32))
        visible[0] = width if step % 2 else 300
        graph.replay()
        torch.cuda.synchronize()
        assert bool((ws.tickets == 0).all())
        want = core.topk_canonical(scores.cpu(), visible.cpu().long(), k)
        assert torch.equal(out.cpu(), want), step
