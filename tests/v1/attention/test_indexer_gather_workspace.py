# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the sparse-indexer gather workspace sizing formula.

The indexer's shared K-gather workspace holds one prefill chunk of compressed
index-K (``head_dim`` fp8 bytes + a 4-byte fp32 scale per row). It used to be
sized by ``get_max_prefill_buffer_size`` alone -- a fixed ``40 * max_model_len``
heuristic that is independent of ``max_num_seqs`` and of the indexer's KV
compression ratio. ``get_indexer_gather_workspace_size`` clamps it to what a
single step can actually gather.

These tests pin the formula, its safety bound (never below one full-context
request), and the invariant that the metadata builder's chunker never forms a
chunk wider than the clamped workspace.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    get_indexer_gather_workspace_size,
    get_max_prefill_buffer_size,
)

# Bytes per gathered row on the FP8 path: head_dim fp8 values + fp32 scale.
ROW_BYTES = 128 + 4
MIB = 1024 * 1024

# The GLM-5.3-Flash deployment: 256K context, 8 runner slots, kpool 4.
DEPLOY_MAX_MODEL_LEN = 262144
DEPLOY_MAX_NUM_SEQS = 8
DEPLOY_KPOOL = 4


def _config(max_model_len: int, max_num_seqs: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_deployment_config_sizing():
    cfg = _config(DEPLOY_MAX_MODEL_LEN, DEPLOY_MAX_NUM_SEQS)

    rows = get_indexer_gather_workspace_size(cfg, DEPLOY_KPOOL)
    assert rows == DEPLOY_MAX_NUM_SEQS * (DEPLOY_MAX_MODEL_LEN // DEPLOY_KPOOL)
    assert rows == 524288

    legacy_rows = get_max_prefill_buffer_size(cfg)
    assert legacy_rows == 40 * DEPLOY_MAX_MODEL_LEN

    # 1320 MiB -> 66 MiB of shared workspace on every rank.
    assert legacy_rows * ROW_BYTES == 1320 * MIB
    assert rows * ROW_BYTES == 66 * MIB


def test_never_exceeds_legacy_heuristic():
    for max_model_len in (4096, 32768, 131072, 262144, 524288, 1048576):
        for max_num_seqs in (1, 8, 64, 256, 1024):
            for compress_ratio in (1, 2, 4, 16):
                cfg = _config(max_model_len, max_num_seqs)
                assert get_indexer_gather_workspace_size(
                    cfg, compress_ratio
                ) <= get_max_prefill_buffer_size(cfg)


def test_never_below_one_full_context_request():
    """``_split_indexer_prefill_chunks`` admits one request whatever the limit,
    so the workspace must always hold a single full-context request."""
    for max_model_len in (4096, 32768, 131072, 262144, 524288, 1048576):
        for max_num_seqs in (1, 8, 64, 256, 1024):
            for compress_ratio in (1, 2, 4, 16):
                cfg = _config(max_model_len, max_num_seqs)
                rows = get_indexer_gather_workspace_size(cfg, compress_ratio)
                assert rows >= cdiv(max_model_len, compress_ratio)


def test_default_compress_ratio_is_uncompressed():
    cfg = _config(DEPLOY_MAX_MODEL_LEN, 1024)
    # max_num_seqs * max_model_len dwarfs the heuristic, so it is unchanged.
    assert get_indexer_gather_workspace_size(cfg) == get_max_prefill_buffer_size(cfg)


def test_scales_with_max_num_seqs_and_compression():
    base = _config(DEPLOY_MAX_MODEL_LEN, DEPLOY_MAX_NUM_SEQS)
    doubled_seqs = _config(DEPLOY_MAX_MODEL_LEN, 2 * DEPLOY_MAX_NUM_SEQS)
    assert get_indexer_gather_workspace_size(
        doubled_seqs, DEPLOY_KPOOL
    ) == 2 * get_indexer_gather_workspace_size(base, DEPLOY_KPOOL)

    assert get_indexer_gather_workspace_size(
        base, 2 * DEPLOY_KPOOL
    ) * 2 == get_indexer_gather_workspace_size(base, DEPLOY_KPOOL)


@pytest.mark.parametrize("num_prefills", [1, 2, 4, 8])
@pytest.mark.parametrize("query_len", [1, 256, 2048])
def test_chunker_never_exceeds_clamped_workspace(num_prefills, query_len):
    """Every chunk the splitter forms must fit the clamped gather workspace.

    The builder and the indexer op size themselves from the same helper, so a
    chunk wider than the workspace would silently truncate the gathered K.
    """
    cfg = _config(DEPLOY_MAX_MODEL_LEN, DEPLOY_MAX_NUM_SEQS)
    workspace_rows = get_indexer_gather_workspace_size(cfg, DEPLOY_KPOOL)
    # Worst case: every admitted request sits at full context.
    pool_len = DEPLOY_MAX_MODEL_LEN // DEPLOY_KPOOL
    compressed_seq_lens = torch.full((num_prefills,), pool_len, dtype=torch.int32)
    query_lens = torch.full((num_prefills,), query_len, dtype=torch.int32)
    max_logits_bytes = 512 * MIB

    chunks = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        compressed_seq_lens,
        query_lens,
        workspace_rows,
        max_logits_bytes,
    )

    assert chunks
    for req_slice, query_slice in chunks:
        chunk_n = int(compressed_seq_lens[req_slice].sum())
        assert chunk_n <= workspace_rows
        chunk_m = query_slice.stop - query_slice.start
        assert chunk_m * chunk_n * 4 <= max_logits_bytes

    # Each request group's query rows are tiled exactly once, and every
    # request appears in exactly one group.
    groups: dict[tuple[int, int], list[slice]] = {}
    for req_slice, query_slice in chunks:
        groups.setdefault((req_slice.start, req_slice.stop), []).append(query_slice)
    seen = [r for start, stop in groups for r in range(start, stop)]
    assert sorted(seen) == list(range(num_prefills))
    for (start, stop), query_slices in groups.items():
        chunk_m = int(query_lens[start:stop].sum())
        offset = 0
        for query_slice in sorted(query_slices, key=lambda s: s.start):
            assert query_slice.start == offset
            offset = query_slice.stop
        assert offset == chunk_m


def test_chunker_unchanged_by_the_clamp_at_deployment_config():
    """At this config the clamp is not the binding constraint, so chunking is
    bit-for-bit what the unclamped heuristic produced."""
    cfg = _config(DEPLOY_MAX_MODEL_LEN, DEPLOY_MAX_NUM_SEQS)
    pool_len = DEPLOY_MAX_MODEL_LEN // DEPLOY_KPOOL
    compressed_seq_lens = torch.full(
        (DEPLOY_MAX_NUM_SEQS,), pool_len, dtype=torch.int32
    )
    query_lens = torch.full((DEPLOY_MAX_NUM_SEQS,), 256, dtype=torch.int32)
    max_logits_bytes = 512 * MIB

    split = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks
    clamped = split(
        compressed_seq_lens,
        query_lens,
        get_indexer_gather_workspace_size(cfg, DEPLOY_KPOOL),
        max_logits_bytes,
    )
    legacy = split(
        compressed_seq_lens,
        query_lens,
        get_max_prefill_buffer_size(cfg),
        max_logits_bytes,
    )
    assert clamped == legacy


def test_kill_switch_restores_legacy_heuristic(monkeypatch):
    """``VLLM_GLM5_INDEXER_GATHER_CLAMP=0`` gives back the unclamped size for
    both the op workspace and the builder chunk limit (same helper)."""
    cfg = _config(DEPLOY_MAX_MODEL_LEN, DEPLOY_MAX_NUM_SEQS)
    monkeypatch.setenv("VLLM_GLM5_INDEXER_GATHER_CLAMP", "0")
    assert get_indexer_gather_workspace_size(
        cfg, DEPLOY_KPOOL
    ) == get_max_prefill_buffer_size(cfg)
    monkeypatch.setenv("VLLM_GLM5_INDEXER_GATHER_CLAMP", "1")
    assert get_indexer_gather_workspace_size(cfg, DEPLOY_KPOOL) == 524288
