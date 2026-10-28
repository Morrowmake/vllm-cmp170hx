# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
from vllm_sm80.models.deepseek_v41.sm80_pipeline_capture import defer_cache


def config(**kwargs):
    cfg = core.Sm80AttnConfig(16, 512, 64, 2, 128, 2, True, True, True, False,
                            32, 128, 512, 2048, 8, 4, 1e-6)
    return dataclasses.replace(cfg, **kwargs)


def test_complete_workspace_bound_and_static_shape_gate():
    cfg = config()
    need = pipe.PipelineWorkspace.bytes(48, 262144, 1280)
    assert pipe.supports(cfg, 48, 262144, 1280, need) is None
    assert pipe.supports(cfg, 48, 262144, 1280, need - 1)
    assert pipe.supports(cfg, 49, 262144, 1280, 1 << 30)
    assert pipe.supports(config(compress_ratio=4), 48, 262144, 1280, 1 << 30)
    assert pipe.supports(config(candidate_block_size=16), 48, 262144, 1280, 1 << 30)


def test_pipeline_and_pre_rope_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match='excludes pre-RoPE'):
        core.prepare_layer(None, None, None, None, None, None, None,
                           fused=True, pipeline=True, workspace=object())
    batch = type('Batch', (), {'num_tokens': 0})()
    with pytest.raises(ValueError, match='excludes pre-RoPE'):
        core.attend_layer(None, None, None, batch, None, None,
                          fused=True, pipeline=True, workspace=object())


def test_deferred_cache_preserves_prestate_aliases_and_ignores_unused_capacity():
    from types import SimpleNamespace
    from vllm_sm80.capture import Record

    cache = torch.arange(6 * 4 * 8).view(6, 4, 8)
    original = cache.clone()
    table = torch.full((2, 65536), 9999, dtype=torch.int32)
    table[:, :2] = torch.tensor([[2, 4], [2, 5]], dtype=torch.int32)
    slots = torch.tensor([8, 9, 21, -1])
    batch = SimpleNamespace(comp_block_table=table, comp_slots=slots,
        positions=torch.tensor([9, 10, 11, 0]), token_to_req=torch.tensor([0, 0, 1, 1]),
        swa_slots=slots)
    record = Record('attn', 'attn_pipeline', 'test', 4, {})
    record.defer(defer_cache('comp', cache, batch, 2))
    valid = slots[slots >= 0]
    cache[valid // 4, valid % 4] += 1000
    out = record.materialize()
    ids, before, after = out['blocks.comp'], out['comp'], out['post.comp']
    assert ids.unique().numel() == ids.numel() <= cache.shape[0]
    assert torch.equal(before, original[ids])
    assert torch.equal(after, cache[ids])
    bt, ss = out['batch.comp_block_table'], out['batch.comp_slots']
    assert bt[0, 0] == bt[1, 0]
    assert ss[-1] == -1 and bool((bt[:, 2:] == 0).all())
    for slot, mapped in zip(slots[:-1], ss[:-1]):
        assert torch.equal(original[slot // 4, slot % 4], before[mapped // 4, mapped % 4])
    with pytest.raises(ValueError, match='duplicate deferred'):
        record.put(comp=torch.empty(0)).materialize()


def test_deferred_window_keeps_negative_indices_and_slot_aliases():
    from types import SimpleNamespace

    cache = torch.arange(6 * 4 * 8).view(6, 4, 8)
    slots = torch.tensor([8, 9, 21, -1])
    indices = torch.tensor([[8, 9, -1], [8, 21, -1]], dtype=torch.int32)
    batch = SimpleNamespace(swa_slots=slots, swa_indices=indices)
    out = defer_cache('swa', cache, batch, 0)()
    ss, ii = out['batch.swa_slots'], out['batch.swa_indices']
    assert ss[0] == ii[0, 0] == ii[1, 0]
    assert ss[-1] == -1 and bool((ii[:, -1] == -1).all())


def test_index_select_family_rejects_nonfinite_visible_arithmetic():
    from .attn_pipeline_cases import index_select_case
    from .fp64_cases import Source
    a = dict(scores=torch.tensor([[float('nan'), 0.0]]), n_vis=torch.tensor([1]), k=512)
    with pytest.raises(ValueError, match='nonfinite arithmetic'):
        index_select_case(a, torch.device('cpu'), Source({}, {}))


def test_index_select_family_exact_ties_and_padding():
    from .attn_pipeline_cases import index_select_case
    from .fp64_cases import Source
    a = dict(scores=torch.tensor([[0., -0., -core.FLT_MAX, 1.], [9., 9., 9., 9.]]),
             n_vis=torch.tensor([4, 0]), k=3)
    call = index_select_case(a, torch.device('cpu'), Source({}, {}))
    assert call.row_invariant
    assert torch.equal(call.cand, torch.tensor([[0, 1, 3], [-1, -1, -1]], dtype=torch.int32))
    assert torch.equal(call.cand.double(), call.exact)
