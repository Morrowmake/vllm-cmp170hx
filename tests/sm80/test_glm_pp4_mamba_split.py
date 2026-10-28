# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align-mode prefill chunks of GLM-5.3-Flash at PP4 end on the KDA state grid.

At PP4 the MLA and KDA groups use 4608-token blocks while the DFlash drafter's
sliding-window groups use 1152, so ``cache_config.block_size`` (the minimum
over prefix-cacheable groups) is 1152. The worker writes a chunk's end state
into KDA column ``(end - 1) // 4608``. A split on the 1152 grid let a chunk that
started on a 1152 multiple run past a 4608 boundary; the block left behind was
then hashed as that boundary's state, and a later request sharing the prefix
resumed from a state up to 3456 tokens short. The scheduler now splits on the
Mamba group's own block size (upstream vllm-project/vllm#54076, first part).

The sm_80 KDA prefill exports no internal checkpoints (no checkpoint blocks)
and RecoverSSM keeps no speculative state blocks; the DFlash boundary cache
drops no Eagle block. TP4 (every group 1152) is the unchanged control.

With the default prefix-cache retention (0: only the replay boundaries and
shared-prefix junctions keep a state) the full block below a replay boundary
that is not on the 4608 grid is still retained, so a prompt of 12000 or 16000
tokens published a stale state there too (14000's replay boundary, 13824, is on
the grid). A periodic retention interval or dense retention (None) publishes
the interior boundaries as well. The cases run all of them and three prompt
lengths.
"""

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

HIT_UNIT = 1152
PP4_BLOCK = 4608
MAX_BATCHED = 2312
FAIR_CHUNK = 384
KDA_GROUP = 1
PROMPT_LEN = 14000


# Prefix-cache retention intervals: dense, the default, and periodic.
RETENTIONS = [None, 0, PP4_BLOCK, 2 * PP4_BLOCK]


def _scheduler(kda_block: int, retention: int | None = None) -> Scheduler:
    """A scheduler with GLM's group geometry: MLA and KDA at ``kda_block``, the
    drafter's sliding-window group at 1152 (the hit unit at both layouts)."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="facebook/opt-125m",
            dtype="float16",
            seed=0,
            skip_tokenizer_init=True,
            # Room for the long prompts; only the scheduler reads it.
            hf_overrides={"max_position_embeddings": 65536},
            max_model_len=65536,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=MAX_BATCHED,
            max_model_len=65536,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        cache_config=CacheConfig(
            block_size=HIT_UNIT,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    num_blocks = 2000
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        prefix_cache_retention_interval=retention,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["mla"],
                FullAttentionSpec(
                    block_size=kda_block,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["kda"],
                MambaSpec(
                    block_size=kda_block,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=0,
                    num_prefill_checkpoint_blocks=0,
                    prefill_checkpoint_alignment=None,
                ),
            ),
            KVCacheGroupSpec(
                ["drafter"],
                SlidingWindowSpec(
                    block_size=HIT_UNIT,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=2048,
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=kda_block,
        hash_block_size=HIT_UNIT,
        log_stats=False,
    )


def _prefill(scheduler: Scheduler, request, first_chunk_limit: int | None):
    """Prefill ``request`` chunk by chunk through the scheduler's split and the
    KV cache manager, the first chunk narrowed as fair prefill narrows it when
    another prefill shares the step.

    Returns ``(state_at, chunks, stale)``: KDA block id -> the token count of the
    state the worker leaves in it (the chunk end lands in column
    ``(end - 1) // kda_block``); the chunks; and every KDA block that was hashed
    while holding another state than its hash claims, checked after each step
    so blocks later swapped out of the request but still cached count too. A
    freed id reused by another group carries that group's hash and is skipped.
    """
    manager = scheduler.kv_cache_manager
    kda = manager.coordinator.single_type_managers[KDA_GROUP]
    kda_block = kda.block_size
    state_at: dict[int, int] = {}
    chunks: list[tuple[int, int]] = []
    stale: set[tuple[int, int, int]] = set()
    limit = first_chunk_limit
    while request.num_computed_tokens < request.num_tokens:
        start = request.num_computed_tokens
        budget = min(request.num_tokens - start, limit or MAX_BATCHED)
        num_new = scheduler._mamba_block_aligned_split(
            request, budget, max_prefill_tokens=limit
        )
        limit = None
        assert num_new > 0, f"no progress at {start}"
        assert manager.allocate_slots(request, num_new) is not None
        request.num_computed_tokens = start + num_new
        chunks.append((start, request.num_computed_tokens))
        blocks = kda.req_to_blocks[request.request_id]
        # A freed KDA id handed out again (to this request, a copy target or
        # another group) no longer holds the state recorded for it.
        live = {block.block_id for block in blocks}
        for block_id in list(state_at):
            if (
                block_id not in live
                and manager.block_pool.blocks[block_id].ref_cnt > 0
            ):
                del state_at[block_id]
        column = cdiv(request.num_computed_tokens, kda_block) - 1
        state_at[blocks[column].block_id] = request.num_computed_tokens
        for block_id, held in state_at.items():
            block = manager.block_pool.blocks[block_id]
            if (
                block.block_hash is not None
                and get_group_id(block.block_hash) == KDA_GROUP
                and block.block_hash_num_tokens != held
            ):
                stale.add((block_id, block.block_hash_num_tokens, held))
    return state_at, chunks, sorted(stale)


def _request(req_id: str, num_tokens: int):
    (request,) = create_requests(
        1,
        num_tokens=num_tokens,
        block_size=HIT_UNIT,
        same_prompt=True,
        req_ids=[req_id],
    )
    return request


def test_split_grid_is_the_kda_block() -> None:
    pp4 = _scheduler(PP4_BLOCK)
    assert pp4.cache_config.block_size == HIT_UNIT
    assert pp4.mamba_state_block_size == PP4_BLOCK
    tp4 = _scheduler(HIT_UNIT)
    assert tp4.mamba_state_block_size == tp4.cache_config.block_size == HIT_UNIT


@pytest.mark.parametrize("prompt_len", [12000, PROMPT_LEN, 16000])
@pytest.mark.parametrize("retention", RETENTIONS)
@pytest.mark.parametrize("first_chunk_limit", [None, FAIR_CHUNK, 1000, 1500])
def test_pp4_hashed_kda_blocks_hold_their_boundary_state(
    first_chunk_limit: int | None, retention: int | None, prompt_len: int
) -> None:
    scheduler = _scheduler(PP4_BLOCK, retention)
    request = _request("a", prompt_len)
    _, chunks, stale = _prefill(scheduler, request, first_chunk_limit)
    assert stale == [], f"KDA blocks hashed with the wrong state {stale}; {chunks}"


@pytest.mark.parametrize("retention", [None, PP4_BLOCK])
def test_pp4_branching_request_resumes_from_the_right_state(
    retention: int | None,
) -> None:
    scheduler = _scheduler(PP4_BLOCK, retention)
    state_at, _, _ = _prefill(scheduler, _request("a", PROMPT_LEN), FAIR_CHUNK)
    branch = _request("b", 6000)
    blocks, num_hit, _ = scheduler.kv_cache_manager.get_computed_blocks(branch)
    assert num_hit > 0
    kda_block = blocks.blocks[KDA_GROUP][-1]
    assert state_at.get(kda_block.block_id) == num_hit


@pytest.mark.parametrize("resume_at", [1152, 2304, 3456, 5760, 8064])
def test_pp4_resume_never_runs_past_a_kda_boundary(resume_at: int) -> None:
    """A follow-up turn hit lands on a 1152 multiple; its chunks must end on
    the 4608 grid before crossing it."""
    scheduler = _scheduler(PP4_BLOCK)
    request = _request("r", 16000)
    pos = resume_at
    while pos < request.num_tokens:
        request.num_computed_tokens = pos
        n = scheduler._mamba_block_aligned_split(
            request, min(request.num_tokens - pos, MAX_BATCHED)
        )
        assert n > 0
        end = pos + n
        if end < request.num_tokens:
            assert end // PP4_BLOCK == pos // PP4_BLOCK or end % PP4_BLOCK == 0, (
                f"chunk [{pos}, {end}) crosses a {PP4_BLOCK} boundary mid-chunk"
            )
        pos = end


@pytest.mark.parametrize("prompt_len", [12000, PROMPT_LEN, 16000])
@pytest.mark.parametrize("retention", RETENTIONS)
@pytest.mark.parametrize("first_chunk_limit", [None, FAIR_CHUNK, 1000])
def test_tp4_control(
    first_chunk_limit: int | None, retention: int | None, prompt_len: int
) -> None:
    scheduler = _scheduler(HIT_UNIT, retention)
    _, chunks, stale = _prefill(scheduler, _request("a", prompt_len), first_chunk_limit)
    assert stale == [], f"{stale}; {chunks}"
