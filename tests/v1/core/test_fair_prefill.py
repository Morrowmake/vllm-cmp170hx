# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fair chunked prefill: the decode-aware prefill budget and the even split
between concurrent prefills.

Both knobs are opt-in. The first two tests pin the unchanged (upstream)
schedule so the rest read as deltas against it.

`facebook/opt-125m` caps max_model_len at 2048, so the budgets here are scaled
down from the production ones (2048 batched tokens / a 512-token cap); the
ratios are what the assertions are about.
"""

import pytest
import torch

from vllm.config import SchedulerConfig
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MAX_LEN = 2048


def _step(scheduler, sampled=None):
    """Schedule one step, feed the result back, and report prefill tokens.

    Returns `(scheduler_output, num_prefill_tokens)`. Prefill vs decode is
    decided from a snapshot taken before `schedule()`, because `schedule()`
    itself advances `num_computed_tokens`.
    """
    before = {
        req_id: (request.num_computed_tokens, request.num_tokens)
        for req_id, request in scheduler.requests.items()
    }
    output = scheduler.schedule()

    prefill_tokens = 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        num_computed, total = before[req_id]
        # Same predicate the scheduler uses: a request only replaying its last
        # token is decoding, anything else is prefill work.
        if num_computed < total - 1:
            prefill_tokens += num_tokens

    req_ids = list(output.num_scheduled_tokens.keys())
    if sampled is None:
        sampled = [
            [] if scheduler.requests[r].is_prefill_chunk else [100] for r in req_ids
        ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return output, prefill_tokens


def _start_decode_streams(scheduler, num_streams: int, prompt_len: int = 8):
    """Bring `num_streams` short requests to the decode phase."""
    requests = create_requests(
        num_requests=num_streams,
        num_tokens=prompt_len,
        max_tokens=512,
        ignore_eos=True,
        req_ids=[f"dec{i}" for i in range(num_streams)],
    )
    for request in requests:
        scheduler.add_request(request)
    _step(scheduler)
    for request in requests:
        assert not request.is_prefill_chunk
    return requests


# --------------------------------------------------------------------------
# 1. Options off -> the schedule is exactly what it was.
# --------------------------------------------------------------------------


def test_options_off_keeps_the_upstream_schedule():
    """Default config: a long prompt takes the whole remaining budget on every
    step and the decode streams get one token each, so a decode step only
    happens as often as a full-size prefill chunk completes."""
    assert SchedulerConfig.prefill_chunk_with_decodes == 0
    assert SchedulerConfig.max_num_partial_prefills == 0

    scheduler = create_scheduler(
        max_num_batched_tokens=256, max_num_seqs=8, max_model_len=MAX_LEN
    )
    assert not scheduler.fair_prefill

    _start_decode_streams(scheduler, 3)
    (long_req,) = create_requests(
        num_requests=1, num_tokens=1800, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)

    # 3 decode tokens + everything else to the prefill.
    output, prefill_tokens = _step(scheduler)
    assert output.num_scheduled_tokens["long"] == 256 - 3
    assert prefill_tokens == 253

    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["long"] == 253


def test_options_off_two_prompts_are_strictly_fcfs():
    """Default config: the second prompt gets nothing until the first one's
    prefill is finished."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256, max_num_seqs=8, max_model_len=MAX_LEN
    )
    a, b = create_requests(
        num_requests=2, num_tokens=1000, max_tokens=4, req_ids=["a", "b"]
    )
    scheduler.add_request(a)
    scheduler.add_request(b)

    for _ in range(2):
        output, _ = _step(scheduler)
        assert output.num_scheduled_tokens["a"] == 256
        assert "b" not in output.num_scheduled_tokens


# --------------------------------------------------------------------------
# 2a. Decode-aware prefill chunk budget.
# --------------------------------------------------------------------------


def test_decode_streams_scheduled_every_step_under_the_cap():
    """With the cap, every step carries at most `prefill_chunk_with_decodes`
    prefill tokens, and every decode stream is still scheduled every step."""
    cap = 32
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=cap,
    )
    assert scheduler.fair_prefill
    decodes = _start_decode_streams(scheduler, 3)

    (long_req,) = create_requests(
        num_requests=1, num_tokens=1800, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)

    for _ in range(6):
        output, prefill_tokens = _step(scheduler)
        assert output.num_scheduled_tokens["long"] == cap
        assert prefill_tokens == cap
        for request in decodes:
            assert output.num_scheduled_tokens[request.request_id] == 1
        assert sum(output.num_scheduled_tokens.values()) == cap + len(decodes)

    assert long_req.is_prefill_chunk


def test_cap_does_not_apply_without_decodes():
    """Nothing decoding -> the full `max_num_batched_tokens` budget is used, so
    a prompt arriving into an idle engine is not slowed down."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=32,
    )
    (long_req,) = create_requests(
        num_requests=1, num_tokens=1800, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)

    for _ in range(2):
        output, _ = _step(scheduler)
        assert output.num_scheduled_tokens["long"] == 256


def test_cap_lifted_again_once_the_decodes_finish():
    """The cap is re-evaluated every step: when the last decode stream stops,
    the long prefill goes back to full-size chunks."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=32,
    )
    (decode_req,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=2, req_ids=["dec"]
    )
    scheduler.add_request(decode_req)
    _step(scheduler)

    (long_req,) = create_requests(
        num_requests=1, num_tokens=1800, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)
    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["long"] == 32
    # max_tokens=2 -> the decode request finished on that step.
    assert decode_req.is_finished()

    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["long"] == 256


def test_cap_is_a_step_budget_not_a_per_request_cap():
    """Several in-flight prefills share one capped budget -- which is what
    `long_prefill_token_threshold` alone does not give: it caps each request
    separately, so N concurrent prefills still produce N chunks of work."""
    cap = 96
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=cap,
        max_num_partial_prefills=3,
    )
    _start_decode_streams(scheduler, 2)
    for request in create_requests(
        num_requests=3, num_tokens=900, max_tokens=4, req_ids=["p0", "p1", "p2"]
    ):
        scheduler.add_request(request)

    for _ in range(4):
        output, prefill_tokens = _step(scheduler)
        assert prefill_tokens <= cap
        assert sum(output.num_scheduled_tokens.values()) <= 256


def test_prefix_cache_hit_only_charges_recomputed_tokens():
    """A cached prefix is not prefill work, so it is not charged against the
    prefill budget: what the cap limits is the tokens actually recomputed."""
    cap = 64
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        enable_prefix_caching=True,
        block_size=block_size,
        prefill_chunk_with_decodes=cap,
    )
    warm, cold = create_requests(
        num_requests=2,
        num_tokens=640,
        max_tokens=512,
        ignore_eos=True,
        same_prompt=True,
        block_size=block_size,
        req_ids=["warm", "cold"],
    )
    # Warm the cache with the full prompt (nothing decoding -> full budget).
    scheduler.add_request(warm)
    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["warm"] == 640
    assert not warm.is_prefill_chunk

    # `warm` is now decoding; `cold` shares its prompt, so nearly all of it is
    # a cache hit and it reaches the decode phase in a single step despite the
    # 64-token cap.
    scheduler.add_request(cold)
    output, prefill_tokens = _step(scheduler)
    assert cold.num_computed_tokens == 640
    assert output.num_scheduled_tokens["cold"] <= cap
    assert prefill_tokens == output.num_scheduled_tokens["cold"]
    assert not cold.is_prefill_chunk


# --------------------------------------------------------------------------
# 2b. Even split between concurrent prefills.
# --------------------------------------------------------------------------


def test_multiple_prefills_interleave():
    """Two long prompts arriving together each get half the prefill budget on
    every step instead of the second waiting for the first to finish."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        max_num_partial_prefills=2,
    )
    a, b = create_requests(
        num_requests=2, num_tokens=1000, max_tokens=4, req_ids=["a", "b"]
    )
    scheduler.add_request(a)
    scheduler.add_request(b)

    for _ in range(3):
        output, _ = _step(scheduler)
        assert output.num_scheduled_tokens["a"] == 128
        assert output.num_scheduled_tokens["b"] == 128
        assert sum(output.num_scheduled_tokens.values()) == 256


def test_split_capped_by_max_num_partial_prefills():
    """The split is bounded: with three prompts waiting and a limit of two,
    only two advance per step, so slices stay usefully large."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        max_num_partial_prefills=2,
    )
    for request in create_requests(
        num_requests=3, num_tokens=1000, max_tokens=4, req_ids=["a", "b", "c"]
    ):
        scheduler.add_request(request)

    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["a"] == 128
    assert output.num_scheduled_tokens["b"] == 128
    assert "c" not in output.num_scheduled_tokens


def test_split_not_applied_to_a_lone_prefill():
    """The divisor is the number of prefills actually in play, so a single
    prompt still gets the whole budget."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        max_num_partial_prefills=4,
    )
    (only,) = create_requests(
        num_requests=1, num_tokens=1000, max_tokens=4, req_ids=["only"]
    )
    scheduler.add_request(only)
    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["only"] == 256


def test_short_prompt_does_not_wait_behind_a_long_one():
    """The fairness win that matters for an interactive coding client: a small prompt is not
    stuck behind 30K tokens of someone else's prefill."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        max_num_partial_prefills=2,
    )
    (long_req,) = create_requests(
        num_requests=1, num_tokens=1800, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)
    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["long"] == 256

    (short_req,) = create_requests(
        num_requests=1, num_tokens=40, max_tokens=4, req_ids=["short"]
    )
    scheduler.add_request(short_req)
    output, _ = _step(scheduler)
    # The short prompt prefills completely on the step it arrives.
    assert output.num_scheduled_tokens["short"] == 40
    assert not short_req.is_prefill_chunk
    assert output.num_scheduled_tokens["long"] == 128


def test_both_knobs_together():
    """Cap and split compose: the capped budget is what gets divided."""
    scheduler = create_scheduler(
        max_num_batched_tokens=512,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=128,
        max_num_partial_prefills=2,
    )
    decodes = _start_decode_streams(scheduler, 3)
    a, b = create_requests(
        num_requests=2, num_tokens=1000, max_tokens=4, req_ids=["a", "b"]
    )
    scheduler.add_request(a)
    scheduler.add_request(b)

    for _ in range(3):
        output, prefill_tokens = _step(scheduler)
        assert output.num_scheduled_tokens["a"] == 64
        assert output.num_scheduled_tokens["b"] == 64
        assert prefill_tokens == 128
        for request in decodes:
            assert output.num_scheduled_tokens[request.request_id] == 1


# --------------------------------------------------------------------------
# Spec decode, budget invariants, hybrid/Mamba alignment, config validation.
# --------------------------------------------------------------------------


def test_spec_decode_tokens_are_accounted_outside_the_prefill_budget():
    """Speculative decodes keep their 1 + k rows while a capped prefill runs,
    and the drafts are not charged against the prefill budget."""
    num_spec = 3
    cap = 32
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=cap,
    )
    decodes = _start_decode_streams(scheduler, 3)
    for request in decodes:
        request.spec_token_ids = [1, 2, 3]

    (long_req,) = create_requests(
        num_requests=1, num_tokens=1800, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)

    output = scheduler.schedule()
    for request in decodes:
        assert output.num_scheduled_tokens[request.request_id] == 1 + num_spec
        assert output.scheduled_spec_decode_tokens[request.request_id] == [1, 2, 3]
    # The prefill chunk is exactly the cap: the verified draft rows come out of
    # the ordinary token budget, not the prefill budget.
    assert output.num_scheduled_tokens["long"] == cap
    # A prefill chunk never carries drafts.
    assert "long" not in output.scheduled_spec_decode_tokens
    assert sum(output.num_scheduled_tokens.values()) == cap + len(decodes) * (
        1 + num_spec
    )


@pytest.mark.parametrize(
    "cap,partial,num_spec",
    [
        (0, 0, 0),
        (128, 0, 0),
        (0, 3, 0),
        (128, 2, 3),
        (7, 3, 1),
        (1, 0, 0),
        (256, 4, 3),
    ],
)
def test_no_budget_overshoot(cap: int, partial: int, num_spec: int):
    """Whatever the settings, a step never exceeds `max_num_batched_tokens`,
    never exceeds the prefill cap in prefill tokens, and always makes
    progress."""
    max_batched = 256
    scheduler = create_scheduler(
        max_num_batched_tokens=max_batched,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        prefill_chunk_with_decodes=cap,
        max_num_partial_prefills=partial,
        num_speculative_tokens=num_spec or None,
    )
    _start_decode_streams(scheduler, 2)
    for request in create_requests(
        num_requests=4,
        num_tokens=900,
        max_tokens=64,
        req_ids=[f"p{i}" for i in range(4)],
    ):
        scheduler.add_request(request)

    prefill_limit = max(cap, scheduler.num_prefill_lookahead + 1)
    steps_with_work = 0
    for _ in range(60):
        output, prefill_tokens = _step(scheduler)
        assert sum(output.num_scheduled_tokens.values()) <= max_batched
        if cap:
            assert prefill_tokens <= prefill_limit
        steps_with_work += bool(output.num_scheduled_tokens)
    assert steps_with_work == 60
    assert scheduler.requests["p0"].num_computed_tokens > 0


def test_mamba_alignment_still_advances_under_a_sub_block_cap():
    """Hybrid (Mamba / linear-attention) models align prefill chunks to the KV
    block. A cap smaller than the block must not clip every chunk to zero: the
    cap is handed to the alignment split so it falls back to a sub-block
    advance, exactly as `long_prefill_token_threshold` already does."""
    block_size = 256
    cap = 64
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        block_size=block_size,
        prefill_chunk_with_decodes=cap,
        kv_cache_spec=MambaSpec(
            shapes=((1,),),
            dtypes=(torch.float32,),
            block_size=block_size,
            page_size_padded=None,
            mamba_type="mamba2",
            num_speculative_blocks=0,
        ),
    )
    scheduler.need_mamba_block_aligned_split = True
    _start_decode_streams(scheduler, 1, prompt_len=4)

    (long_req,) = create_requests(
        num_requests=1,
        num_tokens=1800,
        max_tokens=4,
        block_size=block_size,
        req_ids=["long"],
    )
    scheduler.add_request(long_req)

    for _ in range(5):
        output, _ = _step(scheduler)
        assert 0 < output.num_scheduled_tokens["long"] <= cap
    assert long_req.num_computed_tokens > 0


def test_config_rejects_a_cap_smaller_than_the_split():
    with pytest.raises(ValueError, match="prefill_chunk_with_decodes"):
        SchedulerConfig.default_factory(
            max_num_batched_tokens=2048,
            prefill_chunk_with_decodes=2,
            max_num_partial_prefills=4,
        )


def test_config_ignores_the_knobs_without_chunked_prefill():
    config = SchedulerConfig.default_factory(
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=False,
        prefill_chunk_with_decodes=512,
        max_num_partial_prefills=2,
    )
    assert config.prefill_chunk_with_decodes == 0
    assert config.max_num_partial_prefills == 0


# --------------------------------------------------------------------------
# Regression: the 2026-09-17 benchmark investigation.
#
# The first GPU benchmark read as if only ~1 step in 4 carried the decode
# streams. It did not: the probe counted SSE deltas, and with spec decode one
# delta carries a whole step's accepted tokens. These tests pin the property
# the benchmark could not see -- that every single step of a long chunked
# prefill also carries every decode stream -- at a KV block size the cap does
# not divide, which is the shape that produced the ragged 512/512/128 chunks
# on GLM-5.3-Flash (block 1152, cap 512).
# --------------------------------------------------------------------------

# 288/128 has the same block-to-cap ratio (2.25) as the production 1152/512,
# so it reproduces the same "two full chunks then a short remainder" pattern
# within opt-125m's 2048-token limit.
ODD_BLOCK = 288
ODD_CAP = 128


def _mamba_scheduler(block_size: int, **kwargs):
    scheduler = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        max_model_len=MAX_LEN,
        block_size=block_size,
        enable_prefix_caching=True,
        kv_cache_spec=MambaSpec(
            shapes=((1,),),
            dtypes=(torch.float32,),
            block_size=block_size,
            page_size_padded=None,
            mamba_type="mamba2",
            num_speculative_blocks=0,
        ),
        **kwargs,
    )
    scheduler.need_mamba_block_aligned_split = True
    return scheduler


def _drive_long_prefill(scheduler, prompt_len=1800, num_spec=0):
    """Run a long prefill to completion next to 3 decode streams.

    Returns `(num_steps, num_steps_carrying_all_decodes, chunk_sizes)`.
    """
    decodes = _start_decode_streams(scheduler, 3)
    (long_req,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=4,
        block_size=scheduler.block_size,
        req_ids=["long"],
    )
    scheduler.add_request(long_req)

    steps = 0
    all_decode_steps = 0
    chunks = []
    while long_req.num_computed_tokens < prompt_len and steps < 200:
        if num_spec:
            for request in decodes:
                if not request.is_prefill_chunk:
                    request.spec_token_ids = list(range(1, num_spec + 1))
        output, _ = _step(scheduler)
        steps += 1
        if all(
            output.num_scheduled_tokens.get(r.request_id, 0) > 0 for r in decodes
        ):
            all_decode_steps += 1
        if "long" in output.num_scheduled_tokens:
            chunks.append(output.num_scheduled_tokens["long"])
    return steps, all_decode_steps, chunks


@pytest.mark.parametrize(
    "knobs",
    [
        {"prefill_chunk_with_decodes": ODD_CAP, "max_num_partial_prefills": 2},
        {"prefill_chunk_with_decodes": ODD_CAP},
        # The existing upstream knob takes the identical path, and is pinned
        # here so the two cannot drift apart.
        {"long_prefill_token_threshold": ODD_CAP},
    ],
    ids=["fair", "cap-only", "long_prefill_token_threshold"],
)
def test_every_prefill_step_also_carries_every_decode(knobs):
    """No step of a capped long prefill is decode-free, including the short
    remainder steps the block alignment inserts."""
    scheduler = _mamba_scheduler(ODD_BLOCK, **knobs)
    steps, all_decode_steps, chunks = _drive_long_prefill(scheduler)

    # The cap makes the prefill take many more, smaller steps...
    assert steps >= 15
    assert max(chunks) <= ODD_CAP
    # ...and the alignment inserts short remainder chunks (128/128/32).
    assert min(chunks) < ODD_CAP
    # ...but every one of them still carries all three decode streams.
    assert all_decode_steps == steps


def test_every_prefill_step_carries_decodes_with_spec_decode():
    """Same, with speculative decoding: the decodes keep their 1 + k rows on
    every chunk, which is what the benchmark's SpecDecoding counters showed."""
    num_spec = 3
    scheduler = _mamba_scheduler(
        ODD_BLOCK,
        num_speculative_tokens=num_spec,
        prefill_chunk_with_decodes=ODD_CAP,
        max_num_partial_prefills=2,
    )
    steps, all_decode_steps, chunks = _drive_long_prefill(scheduler, num_spec=num_spec)
    assert steps >= 15
    assert all_decode_steps == steps


def test_every_prefill_step_carries_decodes_under_async_scheduling():
    """Async scheduling replaces the decodes' sampled tokens with output
    placeholders, which changes `is_prefill_chunk`; the decode-aware cap must
    still see them as decodes and keep scheduling them."""
    scheduler = _mamba_scheduler(
        ODD_BLOCK,
        async_scheduling=True,
        prefill_chunk_with_decodes=ODD_CAP,
        max_num_partial_prefills=2,
    )
    steps, all_decode_steps, chunks = _drive_long_prefill(scheduler)
    assert steps >= 15
    assert max(chunks) <= ODD_CAP
    assert all_decode_steps == steps


def test_uncapped_prefill_is_block_aligned_not_batched_token_sized():
    """Baseline the investigation turned up: with `mamba_cache_mode='align'`
    the *uncapped* chunk is floored to a whole number of KV blocks, so it is
    never `max_num_batched_tokens`. On the real model (budget 2048, block
    1152) that floors to one block, which is why the measured default ran a
    20K prompt in 18 steps rather than the 10 a 2048-token chunk implies."""
    prompt_len = 1800
    scheduler = _mamba_scheduler(ODD_BLOCK)
    steps, all_decode_steps, chunks = _drive_long_prefill(scheduler, prompt_len)

    tail = prompt_len % ODD_BLOCK
    assert all(c % ODD_BLOCK == 0 or c == tail for c in chunks), chunks
    # Far fewer, far larger steps than the capped runs above -- that is the
    # whole difference the cap makes, and the decodes ride along either way.
    assert steps <= 5
    assert all_decode_steps == steps
