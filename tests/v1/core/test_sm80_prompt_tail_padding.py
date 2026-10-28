# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uniform-decode padding under the fork scheduler: a prompt tail over cached
state has no generated token or drafts to verify, so it keeps its real
prefill shape instead of placeholder drafts; a cold one-token admission
beside a decode is still padded, and a padding the Mamba align split clips is
dropped, not shortened."""

import pytest

from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from .sm80_utils import create_requests, create_scheduler
from .utils import mock_kv

pytestmark = pytest.mark.cpu_test


def _model_output(scheduler, output, sampled):
    """Feed `sampled` (per-request list) back to the scheduler."""
    req_ids = list(output.num_scheduled_tokens.keys())
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


def test_cached_prompt_tail_keeps_real_shape_with_running_decode():
    """A cached prompt tail is prefill, even beside a speculative decode."""
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    # Two identical 33-token prompts: 2 full blocks (32 tokens) get cached, so a
    # second identical request hits num_computed == num_prompt_tokens - 1.
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    # Drive r1 through prefill so its prompt blocks are cached, then give it real
    # drafts so it is a running speculative decode (1 + num_spec shape).
    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # r2 arrives with one real prompt token left after the prefix hit.
    scheduler.add_request(r2)
    out = scheduler.schedule()

    # r1 verifies its real drafts.
    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    # r2 has no generated token or drafts to verify yet.
    assert out.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in out.scheduled_spec_decode_tokens


def test_restored_prompt_tail_keeps_real_shape_without_running_requests():
    """A synchronous external hit leaves prefill, not a placeholder verify row."""
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=False),
    )
    (resumed,) = create_requests(num_requests=1, num_tokens=33, max_tokens=16)

    scheduler.add_request(resumed)
    out = scheduler.schedule()

    assert out.num_scheduled_tokens[resumed.request_id] == 1
    assert resumed.request_id not in out.scheduled_spec_decode_tokens


@pytest.mark.parametrize("aligned_tokens", [1, 2, 3])
def test_spec_decode_padding_dropped_when_recurrent_alignment_clips(
    aligned_tokens: int,
):
    """A clipped cold-admission padding must be dropped, not shortened.

    The sampler derives a request's row window from its draft count, so a
    padded request that keeps fewer than 1 + num_spec query rows would select
    the preceding request's hidden states.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    (r1,) = create_requests(
        num_requests=1, num_tokens=33, max_tokens=16, req_ids=["decode"]
    )
    (r2,) = create_requests(
        num_requests=1, num_tokens=1, max_tokens=16, req_ids=["cold"]
    )

    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # Cold one-token admissions retain padding beside an existing decode.
    # Model a recurrent split clipping that padded row to each shorter span.
    scheduler.need_mamba_block_aligned_split = True
    original_split = scheduler._mamba_block_aligned_split

    def align_prompt_tail(request, num_new_tokens, *args):
        if request is r2:
            assert num_new_tokens == 1 + num_spec
            return aligned_tokens
        return original_split(request, num_new_tokens, *args)

    scheduler._mamba_block_aligned_split = align_prompt_tail
    scheduler.add_request(r2)
    out = scheduler.schedule()

    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    assert out.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in out.scheduled_spec_decode_tokens
