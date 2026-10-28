# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning rules of the sm80 fork (vllm_sm80/kv_planning.py),
exercised through the upstream entry points that call them."""

import logging
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.core.kv_cache_planning as kv_cache_planning
from tests.v1.core.utils import get_kv_cache_configs, new_kv_cache_spec, new_mamba_spec
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm_sm80.kv_planning import fit_packed_kv_cache_groups, pool_concurrency_limit
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_capacity,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

# --------------------------------------------------------------------------
# Speculative scratch in max concurrency
# --------------------------------------------------------------------------


def _mamba_scratch_vllm_config(
    max_num_seqs: int, max_model_len: int = 32768
) -> VllmConfig:
    """A config whose Mamba groups run in `align` mode, as GLM-5.3-Flash does."""
    model_config = ModelConfig(
        "Qwen/Qwen1.5-7B",
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
        async_scheduling=False,
    )
    cache_config = CacheConfig(block_size=16, mamba_cache_mode="align")
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )


def _mamba_scratch_kv_cache_config(
    num_blocks: int, num_speculative_blocks: int, num_mamba_groups: int = 3
) -> KVCacheConfig:
    """One attention group aliased with `num_mamba_groups` Mamba groups."""
    block_size = 1152
    attn_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    groups = [KVCacheGroupSpec([f"attn_{i}" for i in range(12)], attn_spec)]
    for group_id in range(num_mamba_groups):
        mamba_spec = MambaSpec(
            block_size=block_size,
            shapes=((8, 128),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=num_speculative_blocks,
        )
        groups.append(
            KVCacheGroupSpec([f"mamba_{group_id}_{i}" for i in range(12)], mamba_spec)
        )
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


def _blocks_per_group(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> list[int]:
    return [
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    ]


def test_mamba_speculative_scratch_excluded_from_concurrency():
    """Speculative scratch is charged once per running slot, not per slot."""
    max_num_seqs = 8
    num_blocks = 873
    num_speculative_blocks = 3
    num_mamba_groups = 3
    vllm_config = _mamba_scratch_vllm_config(max_num_seqs)
    config = _mamba_scratch_kv_cache_config(
        num_blocks, num_speculative_blocks, num_mamba_groups
    )

    per_group = _blocks_per_group(vllm_config, config)
    # align mode: 2 live state blocks + num_speculative_blocks scratch blocks.
    assert per_group[1:] == [2 + num_speculative_blocks] * num_mamba_groups
    total_per_request = sum(per_group)
    scratch_per_request = num_mamba_groups * num_speculative_blocks
    resident_per_request = total_per_request - scratch_per_request

    # Plain accounting: every concurrency slot pays for the scratch.
    old_concurrency = num_blocks / total_per_request
    # One scratch pool of max_num_seqs * scratch blocks.
    expected = (
        num_blocks - max_num_seqs * scratch_per_request
    ) / resident_per_request
    assert expected > old_concurrency

    actual = get_max_concurrency_for_kv_cache_config(vllm_config, config)
    assert actual == expected

    # The reserved pool is exactly what a full batch of running requests holds.
    reserved = num_blocks - actual * resident_per_request
    assert reserved == max_num_seqs * scratch_per_request

    num_tokens, reported = get_kv_cache_capacity(vllm_config, config)
    assert reported == actual
    assert num_tokens == int(actual * vllm_config.model_config.max_model_len)


def test_mamba_concurrency_unchanged_without_spec_decode():
    """With speculative decoding off the accounting is bit-identical."""
    max_num_seqs = 8
    num_blocks = 873
    vllm_config = _mamba_scratch_vllm_config(max_num_seqs)
    config = _mamba_scratch_kv_cache_config(num_blocks, num_speculative_blocks=0)

    per_group = _blocks_per_group(vllm_config, config)
    assert per_group[1:] == [2, 2, 2]
    assert get_max_concurrency_for_kv_cache_config(
        vllm_config, config
    ) == num_blocks / sum(per_group)


def test_mamba_scratch_not_discounted_below_max_num_seqs():
    """A pool that holds fewer requests than max_num_seqs keeps the old cost."""
    max_num_seqs = 64
    num_blocks = 873
    vllm_config = _mamba_scratch_vllm_config(max_num_seqs)
    config = _mamba_scratch_kv_cache_config(num_blocks, num_speculative_blocks=3)

    total_per_request = sum(_blocks_per_group(vllm_config, config))
    # 873 / 45 = 19.4 resident requests < 64 running slots: every resident
    # request may be running, so it is charged its scratch.
    assert get_max_concurrency_for_kv_cache_config(
        vllm_config, config
    ) == num_blocks / total_per_request


def test_pool_concurrency_limit_edge_cases():
    limit = pool_concurrency_limit
    # No scratch at all -> the plain ratio.
    assert limit(873, 45, 0, 8) == 873 / 45
    # Scratch, but the pool holds fewer requests than max_num_seqs.
    assert limit(360, 45, 9, 8) == 8.0
    # Continuous at the crossover: both branches agree at exactly max_num_seqs.
    assert limit(45 * 8, 45, 9, 8) == 8.0
    # Above the crossover the scratch pool is reserved only once.
    assert limit(873, 45, 9, 8) == (873 - 72) / 36
    # Monotone in the pool size and never below the plain formula.
    for blocks in range(1, 2000):
        new = limit(blocks, 45, 9, 8)
        assert new >= blocks / 45
    # Degenerate spec whose footprint is entirely scratch.
    assert limit(100, 4, 4, 8) == 25.0


# --------------------------------------------------------------------------
# Packed groups of an MLA + Mamba hybrid with a DFlash drafter
# --------------------------------------------------------------------------

MLA_BLOCK = 1024
NUM_TARGET_LAYERS = 45


def _glm5_like_specs(
    num_draft_layers: int = 0, draft_kv_heads: int = 9
) -> dict[str, KVCacheSpec]:
    """A GLM-5.3-Flash-shaped hybrid: MLA attention, indexer and ring tail on
    every fourth layer (3, 7, ..., 43), KDA state on the other 34, in layer
    order, plus DFlash drafter sliding-window layers numbered after them,
    marked ``is_draft`` as the worker marks a drafter's layers.

    The MLA page is 1152 B/token; the default drafter (9 KV heads of 128,
    bf16) is 4608 B/token, so its exact-fit block is 1024 / 4 = 256 tokens.
    """
    specs: dict[str, KVCacheSpec] = {}
    for i in range(NUM_TARGET_LAYERS):
        if i % 4 == 3:
            specs[f"layers.{i}.attn"] = MLAAttentionSpec(
                block_size=MLA_BLOCK,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
            )
            specs[f"layers.{i}.indexer"] = MLAAttentionSpec(
                block_size=MLA_BLOCK,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                tokens_per_state=4,
            )
            specs[f"layers.{i}.tail"] = CircularBufferSpec(
                block_size=4,
                num_kv_heads=2,
                head_size=128,
                head_size_v=0,
                dtype=torch.bfloat16,
            )
        else:
            specs[f"layers.{i}.kda"] = MambaSpec(
                shapes=((3, 6144), (16, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
                block_size=MLA_BLOCK,
            )
    for j in range(num_draft_layers):
        specs[f"layers.{NUM_TARGET_LAYERS + j}.attn"] = SlidingWindowSpec(
            block_size=MLA_BLOCK,
            num_kv_heads=draft_kv_heads,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=512,
            is_draft=True,
        )
    return specs


def _glm5_config(
    monkeypatch, partition: str | None = None, block_drop: bool = True
) -> VllmConfig:
    config = VllmConfig(model_config=ModelConfig(max_model_len=8192))
    config.cache_config.kv_cache_layout = "BLHNC"
    if partition is not None:
        monkeypatch.setattr(
            ModelConfig, "get_total_num_hidden_layers", lambda self: NUM_TARGET_LAYERS
        )
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
        # Set after construction: ParallelConfig validates the world size
        # against the visible GPUs, and only the grouping reads it here.
        config.parallel_config.pipeline_parallel_size = len(partition.split(","))
    config.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=3,
        parallel_drafting=True,
        disable_eagle_block_drop=not block_drop,
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: block_drop,
        use_multi_module_mtp=lambda: False,
        use_dflash=lambda: True,
    )
    return config


def _is_draft(name: str) -> bool:
    return int(name.split(".")[1]) >= NUM_TARGET_LAYERS


def _draft_groups(groups: list[KVCacheGroupSpec]) -> list[KVCacheGroupSpec]:
    return [g for g in groups if g.layer_names and _is_draft(g.layer_names[0])]


def _split_specs_by_stage(
    specs: dict[str, KVCacheSpec], partition: str
) -> list[dict[str, KVCacheSpec]]:
    parts = [int(x) for x in partition.split(",")]
    bounds = [sum(parts[:r]) for r in range(len(parts) + 1)]
    workers: list[dict[str, KVCacheSpec]] = [{} for _ in parts]
    for name, spec in specs.items():
        index = int(name.split(".")[1])
        # The drafter runs on the last stage.
        rank = next(
            (r for r in range(len(parts)) if index < bounds[r + 1]), len(parts) - 1
        )
        workers[rank][name] = spec
    return workers


def _attention_bytes(worker: dict[str, KVCacheSpec]) -> int:
    """Bytes per block of the worker's MLA + indexer layers."""
    return sum(
        spec.page_size_bytes
        for spec in worker.values()
        if type(spec) is MLAAttentionSpec
    )


@pytest.mark.parametrize("block_drop", [True, False])
def test_drafter_rides_the_mla_block_without_pp(monkeypatch, block_drop):
    """One stage: the drafter gets the exact-fit block, one group, and the
    block is exactly as wide as without a drafter."""
    config = _glm5_config(monkeypatch, block_drop=block_drop)
    target_groups = kv_cache_planning.get_kv_cache_groups(config, _glm5_like_specs())
    specs = _glm5_like_specs(num_draft_layers=5)
    mla_page = specs["layers.3.attn"].page_size_bytes
    groups = kv_cache_planning.get_kv_cache_groups(config, specs)

    names = [name for group in groups for name in group.layer_names]
    assert sorted(names) == sorted(specs)
    draft_groups = _draft_groups(groups)
    assert [g.layer_names for g in draft_groups] == [
        [n for n in specs if _is_draft(n)]
    ]
    spec = draft_groups[0].kv_cache_spec
    assert type(spec) is SlidingWindowSpec
    assert spec.block_size == MLA_BLOCK // 4
    assert spec.page_size_bytes == mla_page
    assert [g.is_eagle_group for g in groups] == [
        block_drop and g in draft_groups for g in groups
    ]
    assert kv_cache_planning._pool_bytes_per_block(
        config, groups
    ) == kv_cache_planning._pool_bytes_per_block(config, target_groups)


@pytest.mark.parametrize(
    "partition",
    [
        # GLM-5.3-Flash + DFlash2 under PP4: MLA layers per stage 3, 3, 2, 3.
        "13,11,11,10",
        # The last stage owns only 2 MLA layers (39, 43).
        "14,12,11,8",
        "12,11,11,11",
    ],
)
def test_pp_groups_fit_every_stage(monkeypatch, partition):
    """Under PP no stage's share of a drafter or KDA group widens the block:
    at memory for N blocks of the widest stage's attention, every worker
    gets N blocks."""
    config = _glm5_config(monkeypatch, partition)
    specs = _glm5_like_specs(num_draft_layers=5)
    workers = _split_specs_by_stage(specs, partition)
    anchor = max(_attention_bytes(worker) for worker in workers)
    mla_page = specs["layers.3.attn"].page_size_bytes
    mamba_page = specs["layers.0.kda"].page_size_bytes

    groups = kv_cache_planning.get_kv_cache_groups(config, specs)
    names = [name for group in groups for name in group.layer_names]
    assert sorted(names) == sorted(specs)

    # Drafter groups: near-equal chunks in layer order, at most
    # anchor // mla_page (= 3 MLA pages of the widest stage) wide.
    draft_groups = _draft_groups(groups)
    assert [len(g.layer_names) for g in draft_groups] == [3, 2]
    assert [n for g in draft_groups for n in g.layer_names] == [
        n for n in specs if _is_draft(n)
    ]
    assert all(g.is_eagle_group for g in draft_groups)
    assert not any(g.is_eagle_group for g in groups if g not in draft_groups)

    # KDA groups: no stage holds more states of one group than fit the block.
    per_block = anchor // mamba_page
    for group in groups:
        spec = group.kv_cache_spec
        if not isinstance(getattr(spec, "first_spec", spec), MambaSpec):
            continue
        for worker in workers:
            assert sum(name in worker for name in group.layer_names) <= per_block

    num_blocks = 100
    configs = get_kv_cache_configs(
        config, workers, [anchor * num_blocks + 1] * len(workers)
    )
    assert [c.num_blocks for c in configs] == [num_blocks] * len(workers)
    for kv_cache_config in configs:
        assert (
            kv_cache_planning._pool_bytes_per_block(
                config, kv_cache_config.kv_cache_groups
            )
            <= anchor
        )
    # Every worker's cache tensors are exactly its own layers, each in one of
    # its groups: the drafter's layers live on the last stage only.
    for worker, kv_cache_config in zip(workers, configs):
        grouped = {n for g in kv_cache_config.kv_cache_groups for n in g.layer_names}
        tensors = [n for t in kv_cache_config.kv_cache_tensors for n in t.layers]
        assert sorted(tensors) == sorted(worker)
        assert grouped == set(worker)

    # The scheduler keeps the drafter annotation although worker 0 (whose
    # config it copies) owns no drafter layer.
    sched = generate_scheduler_kv_cache_config(configs)
    assert [g.is_eagle_group for g in sched.kv_cache_groups] == [
        g in draft_groups for g in groups
    ]


def test_pp_without_block_drop_flags_nothing(monkeypatch):
    partition = "13,11,11,10"
    config = _glm5_config(monkeypatch, partition, block_drop=False)
    specs = _glm5_like_specs(num_draft_layers=5)
    groups = kv_cache_planning.get_kv_cache_groups(config, specs)
    assert not any(g.is_eagle_group for g in groups)
    assert [len(g.layer_names) for g in _draft_groups(groups)] == [3, 2]
    workers = _split_specs_by_stage(specs, partition)
    configs = get_kv_cache_configs(config, workers, [1 << 34] * len(workers))
    sched = generate_scheduler_kv_cache_config(configs)
    assert not any(g.is_eagle_group for g in sched.kv_cache_groups)


def test_misaligned_drafter_keeps_its_block_and_warns(monkeypatch, caplog):
    """A drafter whose exact-fit block is not a multiple of 64 keeps the
    packed grouping, says so, and is still annotated."""
    config = _glm5_config(monkeypatch)
    # 8 KV heads: 4096 B/token -> 1152 * 1024 / 4096 = 288 tokens.
    specs = _glm5_like_specs(num_draft_layers=5, draft_kv_heads=8)
    with caplog.at_level(logging.WARNING):
        groups = kv_cache_planning.get_kv_cache_groups(config, specs)
    assert "not a multiple of 64" in caplog.text
    draft_groups = _draft_groups(groups)
    assert draft_groups
    assert all(g.is_eagle_group for g in draft_groups)
    for group in draft_groups:
        assert type(group.kv_cache_spec) is SlidingWindowSpec
        assert group.kv_cache_spec.block_size == MLA_BLOCK


def test_other_models_are_left_alone():
    """A full-attention + Mamba hybrid with a sliding-window drafter is not
    an MLA + Mamba hybrid: its groups pass through untouched."""
    specs: dict[str, KVCacheSpec] = {
        "layers.0.attn": new_kv_cache_spec(block_size=64),
        "layers.1.mamba": new_mamba_spec(block_size=64),
        "layers.2.attn": SlidingWindowSpec(
            block_size=64,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float32,
            sliding_window=128,
        ),
    }
    groups = [KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        speculative_config=SimpleNamespace(
            use_eagle=lambda: True, use_eagle_block_drop=lambda: True
        ),
    )
    assert fit_packed_kv_cache_groups(config, specs, groups) is groups
    assert not any(group.is_eagle_group for group in groups)


@pytest.mark.parametrize("enabled", [False, True])
def test_kv_plan_log(monkeypatch, caplog, enabled):
    """VLLM_KV_PLAN_LOG logs the plan the concurrency comes from, one entry
    per group; it changes nothing else."""
    monkeypatch.setenv("VLLM_KV_PLAN_LOG", "1" if enabled else "0")
    vllm_config = _mamba_scratch_vllm_config(max_num_seqs=8)
    # A pool size of its own, so the once-only log is not deduplicated away.
    num_blocks = 877 if enabled else 879
    config = _mamba_scratch_kv_cache_config(num_blocks, num_speculative_blocks=3)
    with caplog.at_level(logging.INFO):
        concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, config)
    per_group = _blocks_per_group(vllm_config, config)
    assert concurrency == pool_concurrency_limit(num_blocks, sum(per_group), 9, 8)
    lines = [
        r.getMessage() for r in caplog.records if "KV cache plan" in r.getMessage()
    ]
    if not enabled:
        assert lines == []
        return
    assert len(lines) == 1
    assert f"num_blocks={num_blocks}, {sum(per_group)} blocks per request" in lines[0]
    assert "of which 9 scratch" in lines[0]
    assert f"concurrency {concurrency:.4f}" in lines[0]
    assert "[0] FullAttentionSpec layers=12" in lines[0]
    for group_id in (1, 2, 3):
        assert (
            f"[{group_id}] MambaSpec layers=12 block=1152"
            f" page={config.kv_cache_groups[group_id].kv_cache_spec.page_size_bytes}"
            f" charged={per_group[group_id]} scratch=3" in lines[0]
        )
