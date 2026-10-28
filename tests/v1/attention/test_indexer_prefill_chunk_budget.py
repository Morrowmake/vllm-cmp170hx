# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DSA indexer's prefill logits budget.

``_split_indexer_prefill_chunks`` is what stops the fp32 MQA-logits transient
from scaling with the prompt: the buffer is ``[chunk_rows, compressed_context]``
fp32, and the splitter caps ``chunk_rows * compressed_context * 4`` at
``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB``, sub-chunking on the query axis when one
request alone exceeds it.  Without that cap a single 262144-token prompt would
ask for a 64 GiB tensor.

The numbers below are the live deployment: max_model_len 262144, kpool 4 (so
the indexer's context is token_count // 4), max_num_batched_tokens 2048, and
the 512 MiB default budget.

    pytest -q tests/v1/attention/test_indexer_prefill_chunk_budget.py
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

split = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks

MiB = 1024 * 1024
DEFAULT_BUDGET_BYTES = 512 * MiB
KPOOL = 4
MAX_MODEL_LEN = 262_144
# vllm/v1/attention/backends/mla/indexer.py: 40 * max_model_len gathered rows.
WORKSPACE_ROWS = 40 * MAX_MODEL_LEN


def plan(seq_lens, query_lens, budget=DEFAULT_BUDGET_BYTES, workspace=WORKSPACE_ROWS):
    return split(
        torch.tensor(seq_lens, dtype=torch.int64),
        torch.tensor(query_lens, dtype=torch.int64),
        workspace,
        budget,
    )


def logits_elems(chunks, seq_lens):
    """Peak fp32 element count of each emitted chunk's logits buffer."""
    out = []
    for req_slice, query_slice in chunks:
        rows = query_slice.stop - query_slice.start
        cols = sum(seq_lens[req_slice])
        out.append(rows * cols)
    return out


def covers_every_query_row_once(chunks, query_lens):
    seen: dict[tuple[int, int], int] = {}
    for req_slice, query_slice in chunks:
        key = (req_slice.start, req_slice.stop)
        for row in range(query_slice.start, query_slice.stop):
            assert (key, row) not in seen
            seen[(key, row)] = 1
    per_group: dict[tuple[int, int], int] = {}
    for (key, _row) in seen:
        per_group[key] = per_group.get(key, 0) + 1
    for (start, stop), count in per_group.items():
        assert count == sum(query_lens[start:stop])
    return True


# ------------------------------------------------------- the live shapes


def test_full_context_prefill_chunk_sits_exactly_on_the_default_budget():
    """262144 tokens of context, kpool 4, a full 2048-token scheduler chunk."""
    seq_lens = [MAX_MODEL_LEN // KPOOL]  # 65536 compressed rows
    query_lens = [2048]
    chunks = plan(seq_lens, query_lens)

    assert len(chunks) == 1
    (req_slice, query_slice), = chunks
    assert req_slice == slice(0, 1)
    assert query_slice == slice(0, 2048)
    assert logits_elems(chunks, seq_lens) == [2048 * 65536]
    assert 2048 * 65536 * 4 == DEFAULT_BUDGET_BYTES


def test_two_hundred_k_prefill_stays_under_the_default_budget():
    seq_lens = [200_000 // KPOOL]  # 50000
    query_lens = [2048]
    chunks = plan(seq_lens, query_lens)

    assert len(chunks) == 1
    assert logits_elems(chunks, seq_lens) == [2048 * 50_000]
    assert 2048 * 50_000 * 4 < DEFAULT_BUDGET_BYTES


@pytest.mark.parametrize("context", [131_072, 200_000, 262_144])
@pytest.mark.parametrize("query", [1, 512, 2048])
def test_every_chunk_respects_the_budget(context, query):
    seq_lens = [context // KPOOL]
    query_lens = [query]
    chunks = plan(seq_lens, query_lens)

    assert chunks
    assert covers_every_query_row_once(chunks, query_lens)
    for elems in logits_elems(chunks, seq_lens):
        assert elems * 4 <= DEFAULT_BUDGET_BYTES
        assert elems <= 2**31 - 1


# --------------------------------------------------------- sub-chunking


def test_a_tighter_budget_sub_chunks_the_query_axis():
    seq_lens = [MAX_MODEL_LEN // KPOOL]
    query_lens = [2048]
    chunks = plan(seq_lens, query_lens, budget=128 * MiB)

    assert len(chunks) == 4
    assert [c[1] for c in chunks] == [
        slice(0, 512),
        slice(512, 1024),
        slice(1024, 1536),
        slice(1536, 2048),
    ]
    assert covers_every_query_row_once(chunks, query_lens)
    for elems in logits_elems(chunks, seq_lens):
        assert elems * 4 <= 128 * MiB


def test_a_budget_below_one_row_still_emits_single_row_chunks():
    """The splitter must make progress rather than divide by zero: one row per
    chunk is the floor, even when that single row exceeds the budget."""
    seq_lens = [MAX_MODEL_LEN // KPOOL]
    query_lens = [3]
    chunks = plan(seq_lens, query_lens, budget=4)

    assert [c[1] for c in chunks] == [slice(0, 1), slice(1, 2), slice(2, 3)]
    assert covers_every_query_row_once(chunks, query_lens)


def test_multiple_requests_are_grouped_until_the_budget_binds():
    seq_lens = [4096, 4096, 4096, 4096]
    query_lens = [1024, 1024, 1024, 1024]
    # 4 requests together: 4096 rows x 16384 cols = 67,108,864 elems = 256 MiB.
    chunks = plan(seq_lens, query_lens, budget=256 * MiB)
    assert len(chunks) == 1
    assert chunks[0][0] == slice(0, 4)

    # Halve the budget and the group must split.
    chunks = plan(seq_lens, query_lens, budget=128 * MiB)
    assert len(chunks) > 1
    assert covers_every_query_row_once(chunks, query_lens)
    for elems in logits_elems(chunks, seq_lens):
        assert elems * 4 <= 128 * MiB


def test_the_gathered_kv_workspace_also_bounds_a_group():
    """N is capped by the workspace even when the logits budget is generous."""
    seq_lens = [3000, 3000, 3000]
    query_lens = [8, 8, 8]
    chunks = plan(seq_lens, query_lens, budget=1 << 40, workspace=6000)

    assert [c[0] for c in chunks] == [slice(0, 2), slice(2, 3)]
    for req_slice, _ in chunks:
        assert sum(seq_lens[req_slice]) <= 6000


def test_the_prefill_logits_are_released_before_the_next_chunk():
    """The per-chunk buffer is up to the whole budget, so two of them must not
    be live across the loop edge."""
    import inspect

    from vllm.models.glm5next.nvidia import sparse_indexer

    src = inspect.getsource(sparse_indexer.sparse_attn_indexer_kpool)
    assert "del logits" in src
    top_k = src.index("top_k_per_row_prefill")
    assert src.index("del logits", top_k) > top_k
