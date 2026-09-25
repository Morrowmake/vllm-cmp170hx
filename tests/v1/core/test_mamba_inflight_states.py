# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align-mode Mamba states held for prefill chunks in flight
(VLLM_KV_MAMBA_INFLIGHT_STATES). CPU only."""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

FLAG = "VLLM_KV_MAMBA_INFLIGHT_STATES"
# Another report-only flag that also moves blocks into the running-only part;
# the expected numbers below are with it off.
OTHER_FLAGS = ("VLLM_KV_SWA_INFLIGHT_SCRATCH",)
MAX_LEN = 262144
SPEC = 3

# (max_concurrent_batches, max_num_batched_tokens, Mamba block): production.
TP4 = (2, 3460, 1152)
PP4 = (5, 2312, 4608)


def _vllm_config(
    max_concurrent_batches=2,
    max_num_batched_tokens=3460,
    mamba_cache_mode="align",
    max_num_seqs=8,
    kv_transfer_config=None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_concurrent_batches=max_concurrent_batches,
        max_in_flight_tokens=max_concurrent_batches * max_num_batched_tokens,
        kv_transfer_config=kv_transfer_config,
    )


def _mamba_spec(block_size, mode="align", ckpt=0):
    return MambaSpec(
        block_size=block_size,
        shapes=((4, 64),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_speculative_blocks=SPEC,
        num_prefill_checkpoint_blocks=ckpt,
    )


@pytest.fixture(autouse=True)
def _other_flags_off(monkeypatch):
    for flag in OTHER_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def _pages(spec, nbytes):
    return cdiv(nbytes, spec.page_size_bytes)


def test_flag_off_is_unchanged(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    for mcb, budget, block in (TP4, PP4):
        vc = _vllm_config(mcb, budget)
        spec = _mamba_spec(block)
        assert spec.inflight_state_blocks(vc) == 0
        assert _pages(spec, spec.max_memory_usage_bytes(vc)) == 2 + SPEC
        assert _pages(spec, spec.speculative_scratch_bytes(vc)) == SPEC


@pytest.mark.parametrize(
    "mcb, budget, block, extra",
    [
        (*TP4, 1),  # 2 chunk ends in flight -> seed + 2 states
        (*PP4, 2),  # 5 sub-block chunks span at most 3 blocks
        (5, 2312, 1152, 4),  # PP4 with small blocks: one state per batch
        (1, 3460, 1152, 0),  # synchronous: the two reserved states suffice
        (4, 2312, 4608, 2),  # PP4 without the V2 runner
    ],
)
def test_in_flight_state_blocks(monkeypatch, mcb, budget, block, extra):
    monkeypatch.setenv(FLAG, "1")
    vc = _vllm_config(mcb, budget)
    spec = _mamba_spec(block)
    assert spec.inflight_state_blocks(vc) == extra
    assert _pages(spec, spec.max_memory_usage_bytes(vc)) == 2 + SPEC + extra
    # Held only while chunks are in flight: running-only, like the spec pages.
    assert _pages(spec, spec.speculative_scratch_bytes(vc)) == SPEC + extra
    ck = _mamba_spec(block, ckpt=1)
    assert _pages(ck, ck.max_memory_usage_bytes(vc)) == 3 + SPEC + extra


@pytest.mark.parametrize("mode", ["none", "all"])
def test_other_cache_modes_unchanged(monkeypatch, mode):
    monkeypatch.setenv(FLAG, "1")
    vc = _vllm_config(*TP4[:2], mamba_cache_mode=mode)
    spec = _mamba_spec(TP4[2], mode=mode)
    assert spec.inflight_state_blocks(vc) == 0


def test_kv_connector_charges_per_slot(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    vc = _vllm_config(*PP4[:2], kv_transfer_config=object())
    spec = _mamba_spec(PP4[2])
    assert _pages(spec, spec.max_memory_usage_bytes(vc)) == 2 + SPEC + 2
    assert _pages(spec, spec.speculative_scratch_bytes(vc)) == SPEC


def _config(num_blocks, block, num_mamba, num_swa):
    """A GLM-5.3-Flash-shaped scheduler config: one full-attention group
    (cdiv(262144, block) blocks per request), a kpool tail, `num_mamba` KDA
    groups and `num_swa` drafter sliding-window groups (window 2048, block
    1152: 9 blocks per request at TP4, 13 at PP4)."""
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=block,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.bfloat16,
            ),
        ),
        KVCacheGroupSpec(
            ["tail"],
            KpoolTailSpec(
                block_size=4,
                num_kv_heads=2,
                head_size=128,
                head_size_v=0,
                dtype=torch.bfloat16,
                sliding_window=4,
            ),
        ),
    ]
    for g in range(num_mamba):
        groups.append(KVCacheGroupSpec([f"kda{g}"], _mamba_spec(block)))
    for g in range(num_swa):
        groups.append(
            KVCacheGroupSpec(
                [f"swa{g}"],
                SlidingWindowSpec(
                    block_size=1152,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    sliding_window=2048,
                ),
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


def test_concurrency_report_tp4(monkeypatch):
    """TP4 release: 1,156 blocks, 228 MLA + 1 tail + 4 x 5 KDA + 9 drafter
    = 258 per request (1,174,567 tokens). The pool holds fewer than
    max_num_seqs full requests, so the whole footprint is charged."""
    vc = _vllm_config(*TP4[:2])
    config = _config(1156, TP4[2], 4, 1)
    monkeypatch.delenv(FLAG, raising=False)
    off = get_max_concurrency_for_kv_cache_config(vc, config)
    assert off == pytest.approx(1156 / 258)
    assert int(off * MAX_LEN) == 1174567
    monkeypatch.setenv(FLAG, "1")
    on = get_max_concurrency_for_kv_cache_config(vc, config)
    assert on == pytest.approx(1156 / 262)
    assert int(on * MAX_LEN) == 1156635


def test_concurrency_report_pp4(monkeypatch):
    """PP4 13,11,11,10: 1,017 blocks, 57 MLA + 1 tail + 5 x 5 KDA + 2 x 13
    drafter = 109 per request (2,501,523 tokens, the boot log). More than
    max_num_seqs full requests fit, so spec pages (and, with the flag, the
    in-flight states) are set aside once per running request."""
    vc = _vllm_config(*PP4[:2])
    config = _config(1017, PP4[2], 5, 2)
    monkeypatch.delenv(FLAG, raising=False)
    off = get_max_concurrency_for_kv_cache_config(vc, config)
    assert off == pytest.approx((1017 - 8 * 15) / (109 - 15))
    assert int(off * MAX_LEN) == 2501523
    monkeypatch.setenv(FLAG, "1")
    on = get_max_concurrency_for_kv_cache_config(vc, config)
    assert on == pytest.approx((1017 - 8 * 25) / (119 - 25))
    assert int(on * MAX_LEN) == 2278421


def _chunk_end(start, budget, prompt, block):
    """The alignment rule of Scheduler._mamba_block_aligned_split for a
    request without prefix hits: non-final chunk ends are block-aligned when
    that leaves a non-empty chunk (or the block fits the budget), and a chunk
    starting mid-block stops at the next boundary."""
    end = min(start + budget, prompt)
    if end < prompt:
        aligned = end // block * block
        if aligned > start or block <= budget:
            end = aligned
    nxt = (start // block + 1) * block
    if start % block and start < nxt < end:
        end = nxt
    return end


def _held(manager, rid):
    return sum(1 for b in manager.req_to_blocks.get(rid, ()) if not b.is_null)


@pytest.mark.parametrize(
    "mcb, budget, block, tight",
    [
        (*TP4, True),
        (*PP4, True),
        (5, 2312, 1152, True),
        (1, 3460, 1152, True),
        # Blocks wider than the chunk with two batches: two consecutive
        # sub-block chunk ends share a block, so the bound is not reached.
        (2, 3460, 4608, False),
    ],
)
@pytest.mark.parametrize("prompt", [262000, 100003, 9000])
def test_real_manager_pipelined_prefill(
    monkeypatch, mcb, budget, block, tight, prompt
):
    """A real MambaManager fed one prefill whose chunks fill every in-flight
    batch (the per-request worst case), freeing on the processed basis as
    allocate_slots does. The held states never exceed the new reservation,
    reach it on long prompts, and exceed the old one whenever more than one
    chunk end can be in flight."""
    monkeypatch.setenv(FLAG, "1")
    spec = _mamba_spec(block)
    vc = _vllm_config(mcb, budget)
    old = 2 + SPEC
    new = _pages(spec, spec.max_memory_usage_bytes(vc))
    pool = BlockPool(num_gpu_blocks=2000, enable_caching=False, hash_block_size=block)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block,
    )
    initial_free = pool.get_num_free_blocks()
    computed, inflight, peak = 0, [], 0
    while computed < prompt or inflight:
        if len(inflight) == mcb or computed >= prompt:
            inflight.pop(0)
        if computed < prompt:
            end = _chunk_end(computed, budget, prompt, block)
            in_flight = sum(inflight)
            manager.remove_skipped_blocks("r", max(0, computed - in_flight))
            n = manager.get_num_blocks_to_allocate(
                "r", end + 4, [], computed, computed, end
            )
            manager.allocate_new_blocks("r", end + 4, end)
            assert n >= 0
            inflight.append(end - computed)
            computed = end
        held = _held(manager, "r")
        assert held <= new, (held, new)
        assert initial_free - pool.get_num_free_blocks() == held
        peak = max(peak, held)
    if tight and prompt >= mcb * budget + 2 * block:
        assert peak == new
        extra = min(mcb, cdiv(mcb * budget, block)) - 1
        assert (peak > old) == (extra > 0)
    manager.free("r")
    assert pool.get_num_free_blocks() == initial_free


def test_banner_lines(monkeypatch):
    seen = []
    monkeypatch.setattr(
        kv_cache_utils.logger, "info_once", lambda msg, *args, **kw: seen.append(msg)
    )
    config = _config(1017, PP4[2], 5, 2)
    monkeypatch.setenv(FLAG, "1")
    get_max_concurrency_for_kv_cache_config(_vllm_config(*PP4[:2]), config)
    assert any(m.startswith("Mamba in-flight states") for m in seen)
    seen.clear()
    get_max_concurrency_for_kv_cache_config(_vllm_config(1, 3460), config)
    assert any("no align-mode Mamba group" in m for m in seen)
    seen.clear()
    monkeypatch.delenv(FLAG, raising=False)
    get_max_concurrency_for_kv_cache_config(_vllm_config(*PP4[:2]), config)
    assert not any("Mamba" in m for m in seen)
