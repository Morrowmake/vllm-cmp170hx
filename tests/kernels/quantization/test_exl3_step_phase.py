# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler phase helper of the EXL3 prefill kernels (CPU only)."""

from types import SimpleNamespace

import pytest

import vllm.forward_context as forward_context
from vllm_sm80.kernels.exl3.step_phase import (
    max_decode_query_len,
    max_decode_step_rows,
    prefill_step,
)

# One sampled token plus three draft tokens under verification.
DECODE_QUERY_LEN = 4


def _context(monkeypatch, attn_metadata):
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(attn_metadata=attn_metadata),
    )


def _attention(max_query_len):
    return SimpleNamespace(max_query_len=max_query_len, num_actual_tokens=0)


def test_max_decode_query_len_counts_the_verified_draft():
    assert max_decode_query_len(SimpleNamespace(num_speculative_tokens=3)) == 4
    assert max_decode_query_len(SimpleNamespace(num_speculative_tokens=0)) == 1


def _rows_config(max_num_seqs, k=None, schedule=None, adaptive_k=None, method=None):
    speculative = (
        None
        if k is None
        else SimpleNamespace(
            num_speculative_tokens=k,
            num_speculative_tokens_per_batch_size=schedule,
            adaptive_k=adaptive_k,
            method=method,
        )
    )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=speculative,
    )


def test_max_decode_step_rows_bounds_verify_and_draft_slots():
    # No speculation: one token per request.
    assert max_decode_step_rows(_rows_config(8)) == 8
    # Each request: the sampled token, k verified drafts and k draft slots.
    assert max_decode_step_rows(_rows_config(8, k=5)) == 88
    assert max_decode_step_rows(_rows_config(16, k=3)) == 112
    # A batch-size schedule: k = 5 for 1-3 requests, none for 4-8.
    schedule = [[1, 3, 5], [4, 8, 0]]
    assert max_decode_step_rows(_rows_config(8, k=5, schedule=schedule)) == 33
    # Adaptive k may pick any depth up to num_speculative_tokens.
    config = _rows_config(8, k=5, schedule=schedule, adaptive_k={"by_load": [5]})
    assert max_decode_step_rows(config) == 88


@pytest.mark.parametrize("method", ["dspark", "dflash"])
def test_block_drafters_bound_one_forward_pass_not_two(method):
    # The block runs in its own forward pass: b requests of at most 1 + k
    # rows, beside the target's b * (1 + k(b)); never their sum.
    assert max_decode_step_rows(_rows_config(8, k=5, method=method)) == 48
    assert max_decode_step_rows(_rows_config(8, k=7, method=method)) == 64
    schedule = [[1, 2, 7], [3, 3, 5], [4, 8, 3]]
    config = _rows_config(8, k=7, schedule=schedule, method=method)
    assert max_decode_step_rows(config) == 64
    config = _rows_config(8, k=7, adaptive_k={"by_load": [7, 7, 5, 3]}, method=method)
    assert max_decode_step_rows(config) == 64


def test_block_drafter_k7_keeps_the_dense_prefill_gate_open():
    from vllm_sm80.kernels.exl3 import dense_prefill_tc

    config = _rows_config(8, k=7, method="dspark")
    config.speculative_config.parallel_drafting = True
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=4, pipeline_parallel_size=1
    )
    assert dense_prefill_tc.closed_reason((8, 0), 74, config) is None
    # Eight requests of eight rows is the bound; nine would close it.
    config.scheduler_config.max_num_seqs = 9
    assert "rows" in dense_prefill_tc.closed_reason((8, 0), 74, config)


@pytest.mark.parametrize(
    "max_query_len, expected",
    [(1, False), (4, False), (5, True), (309, True), (3460, True)],
)
def test_phase_follows_the_longest_scheduled_request(
    monkeypatch, max_query_len, expected
):
    # A linear-attention layer's metadata has no max_query_len; the first
    # layer that has one decides.
    linear = SimpleNamespace(num_prefills=0, num_decodes=8)
    _context(
        monkeypatch,
        {
            "layers.0.linear_attn": linear,
            "layers.3.self_attn": _attention(max_query_len),
        },
    )
    assert prefill_step(DECODE_QUERY_LEN) is expected


def test_many_decode_requests_stay_a_decode_step(monkeypatch):
    # 64 requests verifying 3 drafts each: 256 rows, still decode.
    _context(monkeypatch, {"layers.3.self_attn": _attention(DECODE_QUERY_LEN)})
    assert prefill_step(DECODE_QUERY_LEN) is False


@pytest.mark.parametrize(
    "attn_metadata",
    [None, {}, [{"layers.3.self_attn": _attention(3460)}], {"a": _attention(None)}],
    ids=["none", "empty", "microbatches", "no_int"],
)
def test_missing_or_unusable_metadata_counts_as_decode(monkeypatch, attn_metadata):
    _context(monkeypatch, attn_metadata)
    assert prefill_step(DECODE_QUERY_LEN) is False


def test_metadata_without_query_len_counts_as_decode_and_says_so(monkeypatch):
    from vllm_sm80.kernels.exl3 import step_phase

    warned = []
    monkeypatch.setattr(step_phase.logger, "warning_once", warned.append)
    _context(monkeypatch, {"layers.0.linear_attn": SimpleNamespace(num_prefills=1)})
    assert prefill_step(DECODE_QUERY_LEN) is False
    assert len(warned) == 1 and "max_query_len" in warned[0]
    # No metadata at all (capture, profiling): decode without a warning.
    _context(monkeypatch, {})
    assert prefill_step(DECODE_QUERY_LEN) is False
    assert len(warned) == 1


def test_no_forward_context_counts_as_decode(monkeypatch):
    monkeypatch.setattr(forward_context, "_forward_context", None)
    assert prefill_step(DECODE_QUERY_LEN) is False
