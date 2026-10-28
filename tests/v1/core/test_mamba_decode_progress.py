# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A persistent decode budget must not starve an aligned hybrid prefill."""

import pytest
import torch

from tests.v1.core.test_fair_prefill import _start_decode_streams, _step
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import SpeculativeConfig
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "num_spec,draft_slots",
    [(0, 0), (3, 0), (3, 3), (7, 7)],
    ids=["decode", "sequential-spec", "draft-slots3", "draft-slots7"],
)
@pytest.mark.parametrize(
    "budget,cap",
    [(256, 0), (512, 0), (256, 128)],
    ids=["one-block-budget", "two-block-budget", "fair-cap"],
)
def test_long_prompt_advances_beside_decodes(
    budget, cap, num_spec, draft_slots, monkeypatch
):
    # Isolate the scheduler's input-slot contract without loading a drafter.
    # Apply before configuration construction: slots must count from admission,
    # not just after requests have already become running decodes.
    monkeypatch.setattr(
        SpeculativeConfig, "max_num_new_slots_for_drafting",
        property(lambda self: draft_slots),
    )
    block_size = 256
    scheduler = create_scheduler(
        max_num_batched_tokens=budget,
        max_num_seqs=8,
        max_model_len=2048,
        block_size=block_size,
        enable_prefix_caching=True,
        num_speculative_tokens=num_spec or None,
        prefill_chunk_with_decodes=cap,
        kv_cache_spec=MambaSpec(
            block_size=block_size, shapes=((1,),), dtypes=(torch.float32,),
            mamba_cache_mode="align", num_speculative_blocks=num_spec,
        ),
    )
    scheduler.need_mamba_block_aligned_split = True
    spec = scheduler.vllm_config.speculative_config
    assert (spec.max_num_new_slots_for_drafting if spec else 0) == draft_slots
    decodes = _start_decode_streams(scheduler, 4)
    (prompt,) = create_requests(
        1, num_tokens=1800, max_tokens=4, block_size=block_size, req_ids=["long"]
    )
    scheduler.add_request(prompt)
    for _ in range(4):
        for request in decodes:
            request.spec_token_ids = list(range(1, num_spec + 1))
        before = prompt.num_computed_tokens
        output, _ = _step(scheduler)
        assert output.num_scheduled_tokens.get("long", 0) > 0, (
            "alignment discarded the remaining budget behind continuing decodes"
        )
        assert all(output.num_scheduled_tokens[r.request_id] > 0 for r in decodes)
        after = prompt.num_computed_tokens
        # Sub-block progress must still stop at the next cache boundary.
        boundary = (before // block_size + 1) * block_size
        assert not before < boundary < after


@pytest.mark.parametrize("draft_slots", [0, 3, 7])
def test_temporary_prefill_fragment_still_waits_for_a_whole_block(
    draft_slots, monkeypatch
):
    monkeypatch.setattr(
        SpeculativeConfig, "max_num_new_slots_for_drafting",
        property(lambda self: draft_slots),
    )
    scheduler = create_scheduler(
        max_num_batched_tokens=512, max_num_seqs=8, max_model_len=2048,
        block_size=256, enable_prefix_caching=True,
        num_speculative_tokens=draft_slots or None,
        kv_cache_spec=MambaSpec(
            block_size=256, shapes=((1,),), dtypes=(torch.float32,),
            mamba_cache_mode="align", num_speculative_blocks=draft_slots,
        ),
    )
    scheduler.need_mamba_block_aligned_split = True
    for req_id, length in (("short", 128), ("block", 256), ("long", 1800)):
        (request,) = create_requests(
            1, num_tokens=length, max_tokens=4, block_size=256, req_ids=[req_id]
        )
        scheduler.add_request(request)
    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens == {"short": 128, "block": 256}
    # The leftover is transient prefill occupancy, not a reason to fragment
    # the next prompt's state block. Its next step can fund an aligned chunk.
    output, _ = _step(scheduler)
    assert output.num_scheduled_tokens["long"] == 256
