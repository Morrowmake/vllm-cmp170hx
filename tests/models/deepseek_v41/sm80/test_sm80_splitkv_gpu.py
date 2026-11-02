# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DEEPSEEK_V41_SPLITKV_DECODE: the pipeline's split-KV decode attention
against the shipped ordered-value pass, the fused control and FP64."""
import dataclasses

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused as fused
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
from . import fp64_cases as F
from .attn_pipeline_cases import _ideal
from .test_sm80_attn_fused_gpu import _within_v2
from .test_sm80_attn_pipeline_gpu import clone, inputs, snapshot

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')]
BF16 = torch.bfloat16


def _attend(cfg, weights, caches, batch, x, raw, iw, rows, split, monkeypatch):
    monkeypatch.setenv('VLLM_DEEPSEEK_V41_SPLITKV_DECODE', '1' if split else '0')
    ws = pipe.PipelineWorkspace(max(rows, 1), 512, 128, 'cuda')
    assert (ws.split_scratch is not None) == split
    out = torch.full((rows + 3, 16, 512), 17, dtype=BF16, device='cuda')
    q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
    core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
    return q, out, ws


@pytest.mark.parametrize('rows', [1, 2, 5, 6, 8, 12, 16, 18, 24, 32, 40, 48])
@pytest.mark.parametrize('ratio,role', [(0, 'shared'), (1, 'source'), (2, 'source'), (2, 'consumer'), (2, 'shared')])
def test_split_within_v2_of_shipped_and_bit_stable(rows, ratio, role, monkeypatch):
    cfg, weights, caches, batch, matrices, x, raw, iw = inputs(rows, ratio, role)
    before = snapshot(cfg, weights, caches, batch, x, raw, iw)
    shipped_caches, again_caches = clone(caches), clone(caches)
    q, out, ws = _attend(cfg, weights, caches, batch, x, raw, iw, rows, True, monkeypatch)
    assert bool((out[rows:] == 0).all())
    _, again, _ = _attend(cfg, weights, again_caches, batch, x, raw, iw, rows, True, monkeypatch)
    assert F.bits_equal(out, again)
    _, shipped, _ = _attend(cfg, weights, shipped_caches, batch, x, raw, iw, rows, False, monkeypatch)
    for f in dataclasses.fields(caches):
        if getattr(caches, f.name) is not None:
            assert F.bits_equal(getattr(caches, f.name), getattr(shipped_caches, f.name)), f.name
    reference = {f.name: getattr(shipped_caches, f.name) for f in dataclasses.fields(shipped_caches)}
    reference['out'] = shipped[:rows]
    ideal = _ideal(before, F.Source({}, {}), (cfg, None, None, None, None,
                    {k: v.cpu().double() for k, v in matrices.items()}), reference)
    _within_v2(F.Call(out[:rows], shipped[:rows], ideal['out'], True))


def test_split_counts_are_a_function_of_the_shape():
    tiles = 4 + 512 // 32
    for rows in range(1, fused.MAX_ROWS + 1):
        for t in (1, 4, tiles):
            s = fused.split_count(rows, t)
            assert 1 <= s <= min(t, 10)
            assert s == fused.split_count(rows, t)
        assert rows * fused.split_count(rows, tiles) <= max(150, 3 * rows)
    assert fused.split_scratch_rows(fused.MAX_ROWS) == 150


def test_switch_off_allocates_nothing(monkeypatch):
    monkeypatch.setenv('VLLM_DEEPSEEK_V41_SPLITKV_DECODE', '0')
    off = pipe.PipelineWorkspace.bytes(48, 262144, 1280)
    assert pipe.PipelineWorkspace(4, 512, 128, 'cuda').split_scratch is None
    monkeypatch.setenv('VLLM_DEEPSEEK_V41_SPLITKV_DECODE', '1')
    assert pipe.PipelineWorkspace.bytes(48, 262144, 1280) - off == 150 * 16 * 514 * 4


def test_hot_pair_runs_the_listed_arm(monkeypatch):
    """VLLM_HOT_SWAP=splitkv_decode: with the switch off the workspace holds the
    split scratch, and an eager call runs the shipped pass when the pair is
    off and split-KV when it is on, each bit-equal to its own served path."""
    from vllm_sm80 import hot_swap

    rows, ratio, role = 8, 2, 'source'
    outs = {}
    for name, split, listed, on in (('shipped', False, (), frozenset()),
                                    ('split', True, (), frozenset()),
                                    ('hot_off', False, ('splitkv_decode',), frozenset()),
                                    ('hot_on', False, ('splitkv_decode',), frozenset({'splitkv_decode'}))):
        monkeypatch.setattr(hot_swap._STATE, 'listed', listed)
        monkeypatch.setattr(hot_swap._STATE, 'on', on)
        cfg, weights, caches, batch, matrices, x, raw, iw = inputs(rows, ratio, role)
        _, out, ws = _attend(cfg, weights, caches, batch, x, raw, iw, rows, split, monkeypatch) \
            if not listed else _attend_hot(cfg, weights, caches, batch, x, raw, iw, rows, monkeypatch)
        outs[name] = out
    assert F.bits_equal(outs['hot_off'], outs['shipped'])
    assert F.bits_equal(outs['hot_on'], outs['split'])
    assert not F.bits_equal(outs['shipped'], outs['split'])


def _attend_hot(cfg, weights, caches, batch, x, raw, iw, rows, monkeypatch):
    monkeypatch.setenv('VLLM_DEEPSEEK_V41_SPLITKV_DECODE', '0')
    ws = pipe.PipelineWorkspace(max(rows, 1), 512, 128, 'cuda')
    assert ws.split_scratch is not None
    out = torch.full((rows + 3, 16, 512), 17, dtype=BF16, device='cuda')
    q = core.prepare_layer(cfg, weights, caches, batch, x, raw, iw, pipeline=True, workspace=ws)
    core.attend_layer(cfg, weights, caches, batch, q, out, pipeline=True, workspace=ws)
    return q, out, ws
