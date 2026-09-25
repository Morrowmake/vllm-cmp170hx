# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sliding-window in-flight blocks charged once per running request
(VLLM_KV_SWA_INFLIGHT_SCRATCH). CPU only."""

import random
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

FLAG = "VLLM_KV_SWA_INFLIGHT_SCRATCH"
# The other in-flight accounting flag; the expected numbers are with it off.
OTHER_FLAGS = ("VLLM_KV_MAMBA_INFLIGHT_STATES",)
WINDOW = 2048
BLOCK = 1152
MAX_LEN = 262144


@pytest.fixture(autouse=True)
def _other_flags_off(monkeypatch):
    for flag in OTHER_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def _vllm_config(
    max_num_seqs=8,
    max_concurrent_batches=5,
    max_num_batched_tokens=2312,
    kv_transfer_config=None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=max_concurrent_batches * max_num_batched_tokens,
        kv_transfer_config=kv_transfer_config,
    )


def _swa_spec(block_size=BLOCK, window=WINDOW):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=window,
    )


def _full_spec(block_size=BLOCK):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )


def _config(num_blocks, num_swa_groups=2):
    groups = [KVCacheGroupSpec(["full"], _full_spec())]
    for g in range(num_swa_groups):
        layers = {f"swa{g}.{i}": _swa_spec() for i in range(2)}
        groups.append(
            KVCacheGroupSpec(list(layers), UniformTypeKVCacheSpecs(block_size=BLOCK, kv_cache_specs=layers))
        )
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def _blocks(vllm_config, config):
    return [
        cdiv(g.kv_cache_spec.max_memory_usage_bytes(vllm_config), g.kv_cache_spec.page_size_bytes)
        for g in config.kv_cache_groups
    ]


def test_scratch_zero_with_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    vc = _vllm_config()
    assert _swa_spec().speculative_scratch_bytes(vc) == 0


def test_scratch_is_the_in_flight_part(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    vc = _vllm_config()
    spec = _swa_spec()
    full = spec.max_admission_blocks_per_request(vc.max_in_flight_tokens, MAX_LEN)
    settled = spec.max_admission_blocks_per_request(0, MAX_LEN)
    # PP4 production shape: window 2048, block 1152, 5 x 2312 in flight.
    assert (full, settled) == (13, 3)
    assert spec.speculative_scratch_bytes(vc) == 10 * spec.page_size_bytes
    # TP4 shape: 2 x 3460 in flight.
    vc_tp = _vllm_config(max_concurrent_batches=2, max_num_batched_tokens=3460)
    assert spec.speculative_scratch_bytes(vc_tp) == 6 * spec.page_size_bytes
    # The footprint itself is unchanged.
    assert spec.max_memory_usage_bytes(vc) == 13 * spec.page_size_bytes


def test_gate_closed_with_kv_connector(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    vc = _vllm_config(kv_transfer_config=object())
    assert _swa_spec().speculative_scratch_bytes(vc) == 0


def test_kpool_tail_has_no_scratch(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    tail = KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
    )
    assert tail.speculative_scratch_bytes(_vllm_config()) == 0


def test_short_max_model_len_has_no_scratch(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    vc = _vllm_config()
    vc.model_config.max_model_len = 2048
    # Both caps clamp at max_model_len: nothing is in-flight-only.
    assert _swa_spec().speculative_scratch_bytes(vc) == 0


@pytest.mark.parametrize("num_blocks", [300, 832, 833, 1017, 4000])
def test_concurrency_report(monkeypatch, num_blocks):
    vc = _vllm_config()
    config = _config(num_blocks)
    monkeypatch.delenv(FLAG, raising=False)
    per_group = _blocks(vc, config)
    assert per_group == [228, 13, 13]
    per_request = sum(per_group)
    off = get_max_concurrency_for_kv_cache_config(vc, config)
    assert off == num_blocks / per_request

    monkeypatch.setenv(FLAG, "1")
    on = get_max_concurrency_for_kv_cache_config(vc, config)
    scratch = 2 * 10
    if num_blocks / per_request <= vc.scheduler_config.max_num_seqs:
        # Every resident request may be running: full charge, unchanged.
        assert on == off
    else:
        assert on == (num_blocks - 8 * scratch) / (per_request - scratch)
        assert on > off
        # What is set aside is exactly max_num_seqs running requests' worst
        # case, so a full batch of running requests still fits.
        assert num_blocks - on * (per_request - scratch) == pytest.approx(8 * scratch)
    # Never below the old report.
    assert on >= off


def _drive(manager, rid, computed, in_flight, new, lookahead):
    """What allocate_slots does for one request: free on the processed basis,
    then allocate the slots for this step."""
    manager.remove_skipped_blocks(rid, max(0, computed - in_flight))
    need = computed + new + lookahead
    n = manager.get_num_blocks_to_allocate(rid, need, [], computed, 0, computed + new)
    got = manager.allocate_new_blocks(rid, need, computed + new)
    assert len(got) == n


def _held(manager, rid):
    return sum(1 for b in manager.req_to_blocks.get(rid, ()) if not b.is_null)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_held_blocks_within_cap_and_settled(seed):
    """Real SlidingWindowManager under a pipelined schedule: a request never
    holds more than the full cap, holds at most the settled footprint once its
    steps have settled, and a request that is not running holds nothing."""
    rng = random.Random(seed)
    spec = _swa_spec()
    batches, budget, k, lookahead = 5, 2312, 3, 4
    cap = spec.max_admission_blocks_per_request(batches * budget, MAX_LEN)
    settled = spec.max_admission_blocks_per_request(0, MAX_LEN)
    pool = BlockPool(num_gpu_blocks=2000, enable_caching=False, hash_block_size=BLOCK)
    manager = SlidingWindowManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
    )
    initial_free = pool.get_num_free_blocks()
    reqs = {
        f"r{i}": dict(prompt=rng.randrange(3000, 60000), computed=0, in_flight=0, out=0)
        for i in range(8)
    }
    inflight = []
    peak_sum = 0
    for _ in range(3000):
        if len(inflight) == batches or (inflight and rng.random() < 0.3):
            for rid, n, kind in inflight.pop(0):
                st = reqs[rid]
                st["in_flight"] -= n
                if kind == "decode":
                    st["computed"] -= k - rng.randrange(0, k + 1)
                    st["out"] += 1
        room = budget
        step = []
        for rid, st in reqs.items():
            if st["out"] >= 20:
                continue
            left = st["prompt"] - st["computed"]
            if left > 0:
                n, kind = min(left, room), "chunk"
            elif st["in_flight"] == 0:
                n, kind = 1 + k, "decode"
                if n > room:
                    continue
            else:
                continue
            if n <= 0:
                continue
            _drive(manager, rid, st["computed"], st["in_flight"], n, lookahead)
            if kind == "decode":
                # Previous steps have settled: only the window is held.
                assert _held(manager, rid) <= settled
            st["computed"] += n
            st["in_flight"] += n
            room -= n
            step.append((rid, n, kind))
        if step:
            inflight.append(step)
        total = 0
        for rid, st in reqs.items():
            h = _held(manager, rid)
            assert h <= cap
            total += h
            if st["out"] >= 20 and st["in_flight"] == 0 and h:
                manager.free(rid)
        peak_sum = max(peak_sum, total)
        assert total <= len(reqs) * cap
        if all(st["out"] >= 20 for st in reqs.values()) and not inflight:
            break
    for rid in reqs:
        manager.free(rid)
    assert pool.get_num_free_blocks() == initial_free
    assert peak_sum > len(reqs) * settled  # the in-flight part was exercised


def test_banner_lines(monkeypatch):
    vc = _vllm_config()
    config = _config(1017)
    monkeypatch.setenv(FLAG, "1")
    seen = []
    monkeypatch.setattr(
        kv_cache_utils.logger, "info_once", lambda msg, *args, **kw: seen.append(msg)
    )
    get_max_concurrency_for_kv_cache_config(vc, config)
    assert any(m.startswith("SWA in-flight scratch") for m in seen)
    seen.clear()
    vc_conn = _vllm_config(kv_transfer_config=object())
    get_max_concurrency_for_kv_cache_config(vc_conn, config)
    assert any("KV connector" in m for m in seen)
