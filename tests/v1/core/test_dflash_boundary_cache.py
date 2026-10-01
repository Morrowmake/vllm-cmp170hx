# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash context cache boundaries and the matching Mamba prefill stops."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.prefix_cache.test_mamba_eagle_resume_checkpoint import _prefill
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test
BLOCK = 1152
BOUNDARY = 3 * BLOCK
PREFIX = list(range(1, 5 * BLOCK + 1))


def _manager(enabled=False, *, target_eagle=False, draft_eagle=True):
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target"],
                FullAttentionSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=target_eagle,
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=3,
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=2048,
                ),
                is_eagle_group=draft_eagle,
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=8 * BLOCK,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        use_eagle=True,
        use_dflash_boundary=enabled,
    )


def _stub(manager):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=2 * BLOCK,
        use_eagle_block_drop=True,
        mamba_eagle_block_drop=bool(manager.coordinator.eagle_group_ids),
        hash_block_size=BLOCK,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tail", [0, 1, 2, 3, 1151])
def test_warm_hit_and_split_at_cached_boundary(enabled, tail):
    manager = _manager(enabled)
    stub = _stub(manager)
    tokens = PREFIX[: BOUNDARY + tail]
    cold = make_request("cold", tokens, BLOCK, sha256)
    assert manager.get_computed_blocks(cold)[1] == 0
    ends = _prefill(manager, stub, cold)
    manager.cache_blocks(cold, cold.num_computed_tokens)
    # At an exact prompt edge, the sparse Mamba checkpoint is at the
    # prompt end, past the last-logit replay limit; no older checkpoint exists.
    expected = (BOUNDARY if enabled else BOUNDARY - BLOCK) if tail else 0
    # The last token is always evaluated again to obtain the target logits.
    warm = make_request("warm", tokens, BLOCK, sha256)
    blocks, hit, _ = manager.get_computed_blocks(warm)
    assert hit == expected
    if enabled and tail:
        assert BOUNDARY in ends
        assert len(blocks.blocks[0]) == 3
        assert len(blocks.blocks[2]) == 3
        assert blocks.blocks[2][-1].block_hash is not None
    warm_ends = _prefill(manager, stub, warm)
    assert warm_ends[-1] == len(tokens)
    assert warm.num_computed_tokens - hit == len(tokens) - expected


def test_every_prompt_tail_reuses_exactly_one_more_block():
    # Each mode keeps its own sparse checkpoint; all physical-page tails
    # use the same finalized cold prefix.
    managers = [_manager(False), _manager(True)]
    for manager in managers:
        cold = make_request("cold", PREFIX[: BOUNDARY + BLOCK - 1], BLOCK, sha256)
        _prefill(manager, _stub(manager), cold)
        manager.cache_blocks(cold, cold.num_computed_tokens)
    for tail in range(BLOCK):
        hits = []
        for manager in managers:
            warm = make_request(
                f"warm-{tail}", PREFIX[: BOUNDARY + tail], BLOCK, sha256
            )
            hits.append(manager.get_computed_blocks(warm)[1])
        if tail:
            assert hits == [BOUNDARY - BLOCK, BOUNDARY]
        else:
            assert hits == [0, 0]


@pytest.mark.parametrize("enabled", [False, True])
def test_target_eagle_keeps_lookahead_and_mamba_backoff(enabled):
    manager = _manager(enabled, target_eagle=True)
    assert 0 in manager.coordinator.eagle_group_ids
    assert manager.coordinator.single_type_managers[0].use_eagle
    assert manager.coordinator.single_type_managers[1].drop_eagle_checkpoint_block
    request = make_request("cold", PREFIX[: BOUNDARY + 1], BLOCK, sha256)
    assert (
        Scheduler._mamba_block_aligned_split(
            _stub(manager), request, request.num_tokens
        )
        == BOUNDARY - BLOCK
    )


def test_unannotated_groups_keep_conservative_eagle_fallback():
    manager = _manager(True, draft_eagle=False)
    assert manager.coordinator.eagle_group_ids == {0, 1, 2}
    assert not manager.coordinator.dflash_boundary_group_ids


def test_context_boundary_never_hashes_query_or_mask_positions():
    manager = _manager(True)
    cold = make_request("cold", PREFIX[:BOUNDARY], BLOCK, sha256)
    _prefill(manager, _stub(manager), cold)
    manager.cache_blocks(cold, cold.num_computed_tokens)
    draft = manager.coordinator.single_type_managers[2]
    # DFlash query positions are last_valid_target_position + 1 + offset.
    # They are outside all complete context pages, including at an exact edge.
    for tail in range(BLOCK):
        context_end = BOUNDARY + tail
        cached_end = context_end // BLOCK * BLOCK
        query_positions = range(context_end, context_end + 8)
        assert all(position >= cached_end for position in query_positions)
    assert not draft.use_eagle
    assert (
        manager.coordinator.single_type_managers[1].drop_eagle_checkpoint_block is False
    )
    assert all(
        block.block_hash is None for block in draft.req_to_blocks[cold.request_id][3:]
    )


@pytest.mark.parametrize(
    ("flag", "method", "prefix_caching", "active", "banner"),
    [
        (None, "dflash", True, False, None),
        ("0", "dflash", True, False, None),
        ("1", "dflash", True, True, "active"),
        ("1", "ngram", True, False, "inactive"),
        ("1", "dflash", False, False, "inactive"),
    ],
)
def test_scheduler_reads_boundary_flag_and_logs_gate(
    monkeypatch, caplog_vllm, flag, method, prefix_caching, active, banner
):
    import vllm.envs as envs
    import vllm.logger as vl
    from tests.v1.core.utils import create_scheduler
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.structured_output import StructuredOutputManager

    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    name = "VLLM_GLM5_DFLASH_BOUNDARY_CACHE"
    if flag is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, flag)
    assert envs.VLLM_GLM5_DFLASH_BOUNDARY_CACHE is (flag == "1")
    base = create_scheduler(
        block_size=BLOCK,
        max_num_batched_tokens=2 * BLOCK,
        max_model_len=8 * BLOCK,
        enable_prefix_caching=prefix_caching,
        num_speculative_tokens=3,
    )
    # Keep the validated CPU target config and supply the drafter metadata the
    # scheduler reads. Model loading and speculative model validation are
    # outside this scheduler integration test.
    spec = base.vllm_config.speculative_config
    assert isinstance(spec, SpeculativeConfig)
    spec.method = method
    spec.model = "dflash2"
    spec.target_model_config = base.vllm_config.model_config
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="dflash",
            architectures=["DFlash2Qwen3ForCausalLM"],
        )
    )
    base.vllm_config.cache_config.mamba_cache_mode = "align"
    caplog_vllm.clear()
    vl._print_info_once.cache_clear()
    scheduler = Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=_manager().kv_cache_config,
        block_size=BLOCK,
        hash_block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )
    coordinator = scheduler.kv_cache_manager.coordinator
    assert bool(coordinator.dflash_boundary_group_ids) is active
    messages = [record.getMessage() for record in caplog_vllm.records]
    boundary_messages = [
        message for message in messages if "DFlash boundary cache lookup" in message
    ]
    if banner is None:
        assert boundary_messages == []
    else:
        assert len(boundary_messages) == 1
        assert f"lookup {banner} " in boundary_messages[0]
        assert name in boundary_messages[0]
    if method == "dflash" and prefix_caching:
        assert scheduler.use_eagle_block_drop
        assert scheduler.mamba_eagle_block_drop is (not active)
        cold = make_request("cold", PREFIX[: BOUNDARY + 1], BLOCK, sha256)
        ends = _prefill(scheduler.kv_cache_manager, scheduler, cold)
        scheduler.kv_cache_manager.cache_blocks(cold, cold.num_computed_tokens)
        warm = make_request("warm", PREFIX[: BOUNDARY + 1], BLOCK, sha256)
        expected = BOUNDARY if active else BOUNDARY - BLOCK
        assert expected in ends
        assert scheduler.kv_cache_manager.get_computed_blocks(warm)[1] == expected
