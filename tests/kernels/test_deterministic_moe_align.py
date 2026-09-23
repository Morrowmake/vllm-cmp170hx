# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the deterministic moe_align_block_size path.

``csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`` takes its deterministic
single-kernel path only when ``topk_ids.numel() < 1024`` *and*
``num_experts <= 64``. GLM-5.3-Flash has 288 routed experts, so every call
takes the two-kernel path, whose ``count_and_sort_expert_tokens_kernel`` ranks
tokens inside an expert segment with a global ``atomicAdd`` -- the order is
whatever the threads arrived in, and the fused Marlin MoE accumulates each
expert in that order.

``deterministic_moe_align_block_size`` ranks by ascending flat routed-row
index instead. These tests pin the output contract on CPU; no CUDA, no engine.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    deterministic_moe_align_block_size,
    use_deterministic_moe_align,
)


def reference_moe_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    buf_len: int,
    num_blocks: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Independent reference alignment, ranking by ascending flat index."""
    numel = topk_ids.numel()
    flat = topk_ids.reshape(-1).tolist()
    per_expert: list[list[int]] = [[] for _ in range(num_experts)]
    for i, e in enumerate(flat):
        if e < 0 or e >= num_experts:
            continue
        if expert_map is not None:
            e = int(expert_map[e])
            if e < 0:
                continue
        per_expert[e].append(i)

    sorted_ids = torch.full((buf_len,), numel, dtype=torch.int32)
    expert_ids = torch.full((num_blocks,), -1, dtype=torch.int32)
    cursor = 0
    for e in range(num_experts):
        n = len(per_expert[e])
        padded = -(-n // block_size) * block_size
        for j, tok in enumerate(per_expert[e]):
            sorted_ids[cursor + j] = tok
        for b in range(padded // block_size):
            expert_ids[cursor // block_size + b] = e
        cursor += padded
    return sorted_ids, expert_ids, cursor


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_DETERMINISTIC_MOE_ALIGN is False
    use_deterministic_moe_align.cache_clear()
    assert use_deterministic_moe_align() is False


def test_flag_is_read_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    use_deterministic_moe_align.cache_clear()
    try:
        assert use_deterministic_moe_align() is True
    finally:
        use_deterministic_moe_align.cache_clear()


NUM_EXPERTS = 288  # GLM-5.3-Flash n_routed_experts
TOPK = 8  # num_experts_per_tok


def _run_align(topk_ids, block_size, num_experts=NUM_EXPERTS, expert_map=None):
    numel = topk_ids.numel()
    buf_len = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        buf_len = min(numel * block_size, buf_len)
    num_blocks = -(-buf_len // block_size)
    sorted_ids = torch.empty(buf_len, dtype=torch.int32)
    expert_ids = torch.empty(num_blocks, dtype=torch.int32)
    ntpp = torch.empty(1, dtype=torch.int32)
    scatter = torch.empty((numel, 1), dtype=torch.int32)
    deterministic_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        ntpp,
        expert_map,
        scatter,
    )
    return sorted_ids, expert_ids, ntpp, scatter, buf_len, num_blocks


def _routing(m: int, seed: int, num_experts: int = NUM_EXPERTS) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(num_experts, generator=g)[:TOPK] for _ in range(m)]
    ).to(torch.int32)


@pytest.mark.parametrize("m,block_size", [(4, 8), (16, 8), (64, 16), (1152, 64)])
def test_alignment_matches_the_reference(m, block_size):
    topk_ids = _routing(m, seed=100 + m)
    sorted_ids, expert_ids, ntpp, _, buf_len, num_blocks = _run_align(
        topk_ids, block_size
    )
    ref_sorted, ref_experts, ref_total = reference_moe_align(
        topk_ids, NUM_EXPERTS, block_size, buf_len, num_blocks
    )
    assert torch.equal(sorted_ids, ref_sorted)
    assert torch.equal(expert_ids, ref_experts)
    assert int(ntpp[0]) == ref_total


@pytest.mark.parametrize("m,block_size", [(4, 8), (16, 8), (1152, 64)])
def test_alignment_is_valid(m, block_size):
    """Every routed row appears exactly once, in its own expert's segment,
    and every other slot holds the padding sentinel."""
    topk_ids = _routing(m, seed=200 + m)
    numel = topk_ids.numel()
    sorted_ids, expert_ids, ntpp, scatter, _, _ = _run_align(topk_ids, block_size)
    total = int(ntpp[0])
    assert total % block_size == 0

    live = sorted_ids[:total]
    real = live[live != numel]
    assert sorted(real.tolist()) == list(range(numel)), "not a permutation"
    assert (sorted_ids[total:] == numel).all(), "tail is not sentinel"

    flat = topk_ids.reshape(-1)
    for blk in range(total // block_size):
        e = int(expert_ids[blk])
        assert e >= 0
        seg = sorted_ids[blk * block_size : (blk + 1) * block_size]
        for tok in seg.tolist():
            if tok != numel:
                assert int(flat[tok]) == e, "token filed under the wrong expert"
    assert (expert_ids[total // block_size :] == -1).all()
    assert scatter.reshape(-1).tolist() == list(range(numel))


@pytest.mark.parametrize("m,block_size", [(4, 8), (16, 8), (1152, 64)])
def test_within_segment_order_is_ascending_flat_index(m, block_size):
    """The property the atomicAdd path does not have."""
    topk_ids = _routing(m, seed=300 + m)
    numel = topk_ids.numel()
    sorted_ids, expert_ids, ntpp, _, _, _ = _run_align(topk_ids, block_size)
    total = int(ntpp[0])
    seen: dict[int, list[int]] = {}
    for blk in range(total // block_size):
        e = int(expert_ids[blk])
        for tok in sorted_ids[blk * block_size : (blk + 1) * block_size].tolist():
            if tok != numel:
                seen.setdefault(e, []).append(tok)
    for toks in seen.values():
        assert toks == sorted(toks)


@pytest.mark.parametrize("m,block_size", [(4, 8), (16, 8), (1152, 64)])
def test_align_repeated_calls_are_identical(m, block_size):
    topk_ids = _routing(m, seed=400 + m)
    first = [t.clone() for t in _run_align(topk_ids, block_size)[:4]]
    for _ in range(19):
        again = _run_align(topk_ids, block_size)[:4]
        for a, b in zip(first, again):
            assert torch.equal(a, b)


def test_invalid_expert_ids_are_dropped():
    topk_ids = _routing(8, seed=500)
    topk_ids[0, 0] = -1
    topk_ids[3, 2] = NUM_EXPERTS  # out of range
    numel = topk_ids.numel()
    sorted_ids, _, ntpp, scatter, _, _ = _run_align(topk_ids, 8)
    total = int(ntpp[0])
    live = sorted_ids[:total]
    real = sorted(live[live != numel].tolist())
    dropped = {0 * TOPK + 0, 3 * TOPK + 2}
    assert real == [i for i in range(numel) if i not in dropped]
    flat_scatter = scatter.reshape(-1).tolist()
    for i in range(numel):
        assert flat_scatter[i] == (-1 if i in dropped else i)


def test_expert_map_shards_and_drops():
    num_experts = 64
    topk_ids = _routing(16, seed=600, num_experts=num_experts)
    # Local shard owns the even experts; the odd ones map to -1.
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
    local = torch.arange(0, num_experts, 2)
    expert_map[local] = torch.arange(local.numel(), dtype=torch.int32)

    sorted_ids, expert_ids, ntpp, scatter, buf_len, num_blocks = _run_align(
        topk_ids, 8, num_experts=num_experts, expert_map=expert_map
    )
    ref_sorted, ref_experts, ref_total = reference_moe_align(
        topk_ids, num_experts, 8, buf_len, num_blocks, expert_map=expert_map
    )
    assert torch.equal(sorted_ids, ref_sorted)
    assert torch.equal(expert_ids, ref_experts)
    assert int(ntpp[0]) == ref_total
    flat = topk_ids.reshape(-1)
    for i in range(topk_ids.numel()):
        expected = -1 if int(expert_map[int(flat[i])]) < 0 else i
        assert int(scatter.reshape(-1)[i]) == expected


def test_all_tokens_to_one_expert():
    topk_ids = torch.full((32, TOPK), 7, dtype=torch.int32)
    sorted_ids, expert_ids, ntpp, _, _, _ = _run_align(topk_ids, 16)
    total = int(ntpp[0])
    assert total == 256  # 32 * 8 routes, already a multiple of 16
    assert sorted_ids[:total].tolist() == list(range(256))
    assert (expert_ids[: total // 16] == 7).all()
    assert (expert_ids[total // 16 :] == -1).all()


def test_wrapper_takes_the_deterministic_path(monkeypatch):
    """moe_align_block_size() is wired.

    ops.moe_align_block_size is CUDA-only, so a result on CPU proves the
    deterministic branch ran instead.
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    use_deterministic_moe_align.cache_clear()
    try:
        topk_ids = _routing(16, seed=700)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            topk_ids, 8, NUM_EXPERTS, None, False, True
        )
        ref_sorted, ref_experts, ref_total = reference_moe_align(
            topk_ids, NUM_EXPERTS, 8, sorted_ids.numel(), expert_ids.numel()
        )
        assert torch.equal(sorted_ids, ref_sorted)
        assert torch.equal(expert_ids, ref_experts)
        assert int(ntpp[0]) == ref_total
    finally:
        use_deterministic_moe_align.cache_clear()


def test_no_cuda_was_initialised():
    assert torch.cuda.is_initialized() is False
