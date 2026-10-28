# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba speculative scratch as the fork's KV planning charges it
(vllm_sm80/kv_planning.py): only running requests hold it, and it is exactly
the scratch part of the spec's footprint."""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm_sm80.kv_planning import speculative_scratch_bytes

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
def test_mamba_speculative_scratch_bounded_by_running_requests(
    num_speculative_blocks,
):
    """Runtime scratch never exceeds max_num_seqs * num_speculative_blocks.

    Capacity planning reserves exactly that many blocks per Mamba group
    (see `vllm_sm80.kv_planning.pool_concurrency_limit`), so the allocator
    must not let the scratch footprint grow with the number of requests
    *served*, only with the number of requests running at once.
    """
    block_size = 1152
    max_num_seqs = 8
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )
    pool = BlockPool(num_gpu_blocks=4096, enable_caching=False, hash_block_size=128)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    initial_free = pool.get_num_free_blocks()
    request_ids = [f"r{i}" for i in range(max_num_seqs)]
    peak_held = 0

    # Two full batches back to back: prefill in chunks, then decode. The
    # second batch must not inflate the peak -- freed scratch is reusable.
    for _round in range(2):
        for target in range(block_size, 16 * block_size + 1, block_size):
            for request_id in request_ids:
                computed = target - block_size
                manager.remove_skipped_blocks(request_id, computed)
                manager.get_num_blocks_to_allocate(
                    request_id, target, [], computed, computed, target
                )
                manager.allocate_new_blocks(request_id, target, target)
                peak_held = max(peak_held, initial_free - pool.get_num_free_blocks())
        # Decode steps: the speculative scratch is relocated, not re-allocated.
        for step in range(1, 40):
            target = 16 * block_size + step
            for request_id in request_ids:
                manager.remove_skipped_blocks(request_id, target - 1)
                manager.get_num_blocks_to_allocate(
                    request_id, target, [], target - 1, target - 1, target
                )
                manager.allocate_new_blocks(request_id, target, target)
                peak_held = max(peak_held, initial_free - pool.get_num_free_blocks())
        for request_id in request_ids:
            manager.free(request_id)
        assert pool.get_num_free_blocks() == initial_free

    # 2 live state blocks per request, plus the speculative scratch.
    assert peak_held == max_num_seqs * (2 + num_speculative_blocks)
    scratch_held = peak_held - max_num_seqs * 2
    assert scratch_held == max_num_seqs * num_speculative_blocks


def test_mamba_speculative_scratch_bytes_matches_spec():
    """`speculative_scratch_bytes` is exactly the scratch part of the footprint."""
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig

    model_config = ModelConfig(
        "Qwen/Qwen1.5-7B", runner="generate", dtype="float16", max_model_len=32768
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            max_model_len=32768,
            is_encoder_decoder=model_config.is_encoder_decoder,
            async_scheduling=False,
        ),
        cache_config=CacheConfig(block_size=16, mamba_cache_mode="align"),
    )
    for num_speculative_blocks in (0, 1, 3):
        spec = MambaSpec(
            block_size=1152,
            shapes=((8, 128),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=num_speculative_blocks,
        )
        scratch = speculative_scratch_bytes(spec, vllm_config)
        assert scratch == num_speculative_blocks * spec.page_size_bytes
        assert (
            spec.max_memory_usage_bytes(vllm_config) - scratch
            == 2 * spec.page_size_bytes
        )
    # Specs without speculative scratch report zero.
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    assert speculative_scratch_bytes(full, vllm_config) == 0
