# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the GLM-5.3-Flash kpool tail ring sizing under speculative
decoding (no GPU required).

A spec-verify step stashes 1 + num_spec rows into the per-request tail ring
before acceptance is known. The ring must hold the open pool's committed keys
plus those rows, or the rows of a rejected pool-completing draft overwrite
keys that the redone completion reads.
"""

import math
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.models.glm5next.common.attention import (
    Glm5NextTailCache,
    kpool_tail_ring_size,
)
from vllm.v1.kv_cache_interface import KpoolTailSpec

HEAD_DIM = 128


@pytest.mark.parametrize(
    "kpool,num_spec,ring",
    [
        (4, 0, 4),
        (4, 1, 8),
        (4, 3, 8),
        (4, 4, 8),
        (4, 5, 12),
        (4, 7, 12),
        (4, 8, 12),
        (4, 9, 16),
        (16, 0, 16),
        (16, 3, 32),
        (16, 7, 32),
    ],
)
def test_ring_size(kpool, num_spec, ring):
    got = kpool_tail_ring_size(kpool, num_spec)
    assert got == ring
    assert got % kpool == 0 and got >= kpool + num_spec
    assert kpool_tail_ring_size(kpool, num_spec, legacy=True) == kpool


def _spec_for(num_spec, monkeypatch, legacy=False):
    monkeypatch.setattr(
        envs, "VLLM_GLM5_KPOOL_TAIL_LEGACY_RING", legacy, raising=False
    )
    cache = SimpleNamespace(_index_kpool=4, head_dim=HEAD_DIM)
    vllm_config = SimpleNamespace(num_speculative_tokens=num_spec)
    return Glm5NextTailCache.get_kv_cache_spec(cache, vllm_config)


@pytest.mark.parametrize("num_spec,ring", [(0, 4), (3, 8), (7, 12)])
def test_tail_spec_uses_the_ring(num_spec, ring, monkeypatch):
    spec = _spec_for(num_spec, monkeypatch)
    assert isinstance(spec, KpoolTailSpec)
    assert spec.block_size == ring
    assert spec.sliding_window == ring
    assert spec.num_kv_heads == 2 and spec.head_size == HEAD_DIM
    # One ring block per request whatever the length.
    assert spec.max_num_blocks_per_req(None, 262144) == 1


@pytest.mark.parametrize("num_spec", [0, 3, 7])
def test_legacy_kill_switch_restores_one_pool_ring(num_spec, monkeypatch):
    spec = _spec_for(num_spec, monkeypatch, legacy=True)
    assert spec.block_size == 4
    assert spec.sliding_window == 4


@pytest.mark.parametrize("block_size", [1152, 4608])
@pytest.mark.parametrize("num_spec", range(0, 13))
def test_ring_fits_the_aliased_indexer_page(block_size, num_spec):
    """The GLM-5.3-Flash layout aliases each tail tensor into its indexer
    layer's page and gives up the layout when the tail page is larger. The
    scheduler block size is the LCM of every group's block size, so the ring
    must also divide the cache block size to leave it unchanged."""
    kpool = 4
    ring = kpool_tail_ring_size(kpool, num_spec)
    tail = KpoolTailSpec(
        block_size=ring,
        num_kv_heads=2,
        head_size=HEAD_DIM,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=ring,
    )
    # Indexer page: block_size // kpool pooled entries of HEAD_DIM fp8 bytes
    # plus a 4-byte scale.
    idx_page = (block_size // kpool) * (HEAD_DIM + 4)
    assert tail.page_size_bytes == ring * 2 * HEAD_DIM * 2
    assert tail.page_size_bytes <= idx_page
    assert math.lcm(block_size, ring) == block_size


def _simulate(kpool, ring, num_spec, steps, rng):
    """Mirror the stash / completion addressing of the tail kernels
    (``pos % ring`` for stashes, ``(pool_start + s) % ring`` for the pool
    read) over a run of spec-verify steps with random acceptance. Returns the
    number of accepted pool completions that read a slot which does not hold
    the committed key of the position it stands for."""
    stored = [None] * ring  # (position, is_committed)
    prompt = rng.randrange(kpool, 4 * kpool)
    # Prefill seed: the prompt's last kpool tokens.
    for p in range(prompt - kpool, prompt):
        stored[p % ring] = (p, True)
    nxt = prompt  # first row of the next verify step (always a real token)
    bad = 0
    for _ in range(steps):
        accepted = rng.randint(0, num_spec)
        rows = range(nxt, nxt + 1 + num_spec)
        for p in rows:
            committed = p <= nxt + accepted
            if p % kpool == kpool - 1:
                start = p - (kpool - 1)
                ok = all(
                    stored[q % ring] == (q, True) for q in range(start, p)
                )
                # Only a completion by an accepted row is final; a rejected
                # one is overwritten when its position is verified again.
                if committed and not ok:
                    bad += 1
            stored[p % ring] = (p, committed)
        nxt = nxt + accepted + 1
    return bad


@pytest.mark.parametrize("kpool", [4, 16])
@pytest.mark.parametrize("num_spec", range(0, 9))
def test_ring_survives_rejected_drafts(kpool, num_spec):
    rng = random.Random(1000 * kpool + num_spec)
    ring = kpool_tail_ring_size(kpool, num_spec)
    assert _simulate(kpool, ring, num_spec, steps=4000, rng=rng) == 0


@pytest.mark.parametrize("num_spec", [2, 3, 7])
def test_one_pool_ring_is_corrupted_by_rejected_drafts(num_spec):
    """The layout before the fix (ring == kpool) with the production kpool."""
    rng = random.Random(num_spec)
    assert _simulate(4, 4, num_spec, steps=4000, rng=rng) > 0


def test_one_pool_ring_is_safe_without_speculation():
    rng = random.Random(0)
    assert _simulate(4, 4, 0, steps=4000, rng=rng) == 0
