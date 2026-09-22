# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the invariant that keeps kpool-tail padding harmless.

``compute_kpool_tail_slot_mapping`` rewrites the real tokens of the batch onto
each request's circular tail block and *copies the caller's padding through
untouched*.  That is only safe because of a chain of facts elsewhere in the
tree, and the failure it would cause is an illegal memory access under FULL
cudagraph replay -- not a wrong answer -- so it is worth pinning:

1. ``KpoolTailSpec.uses_slot_mapping`` is False, so the generic slot-mapping
   kernel never writes real slots for this group;
2. that kernel writes ``PAD_SLOT_ID`` for mapping-disabled groups and clears
   the rest of the buffer to ``PAD_SLOT_ID``;
3. ``PAD_SLOT_ID`` is negative;
4. the tail-seed kernel returns early on a negative slot.

Break any one of them and padded rows start addressing real pool-cache slots.
This file asserts all four, and demonstrates that the helper itself provides
no second line of defence.

    pytest -q tests/v1/attention/test_kpool_tail_padding_invariant.py
"""

import inspect

import torch

from vllm.models.glm5next.nvidia.ops import kpool_compress
from vllm.v1.attention.backends.mla.indexer import compute_kpool_tail_slot_mapping
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KpoolTailSpec
from vllm.v1.worker.gpu import block_table

KPOOL = 4


def tail_spec():
    return KpoolTailSpec(
        block_size=KPOOL,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=KPOOL,
    )


def test_the_pad_sentinel_is_negative():
    """Every downstream guard is a sign test, not an equality test."""
    assert PAD_SLOT_ID < 0


def test_the_tail_group_never_gets_a_generic_slot_mapping():
    assert tail_spec().uses_slot_mapping is False


def test_the_generic_kernel_pads_disabled_groups_with_the_sentinel():
    src = inspect.getsource(block_table)
    assert "slot_ids = tl.where(mapping_enabled & is_real_req, slot_ids, PAD_ID)" in src
    assert (
        "tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)"
        in src
    )
    assert "PAD_ID=PAD_SLOT_ID" in src


def test_the_tail_seed_kernel_skips_negative_slots():
    src = getattr(
        kpool_compress._kpool_tail_seed_kernel, "src", None
    ) or inspect.getsource(kpool_compress._kpool_tail_seed_kernel)
    body = src[src.index("t = tl.load(tslot_ptr") :]
    assert "if t < 0:" in body
    assert body.index("return") < body.index("blk = t // KPOOL")


def test_the_helper_copies_padding_through_verbatim():
    """The helper is not the guard: whatever the caller left in the padded tail
    is what the tail kernels will see. If this ever needs to change, change the
    helper *and* this test together."""
    positions = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int64)
    block_table_tensor = torch.zeros(2, 8, dtype=torch.int32)
    block_table_tensor[:, 0] = torch.tensor([5, 9], dtype=torch.int32)
    num_actual, num_reqs = 6, 2

    poisoned = torch.arange(10, dtype=torch.int64) + 1000
    out = compute_kpool_tail_slot_mapping(
        poisoned,
        block_table_tensor,
        query_start_loc,
        positions,
        num_actual,
        num_reqs,
        KPOOL,
    )

    # Real tokens land in their own request's circular block.
    assert out[:4].tolist() == [5 * KPOOL + i for i in range(4)]
    assert out[4:6].tolist() == [9 * KPOOL, 9 * KPOOL + 1]
    # Padding is whatever the caller supplied -- here, poison.
    assert out[6:].tolist() == poisoned[6:].tolist()


def test_production_padding_stays_at_the_sentinel():
    """The same call with the padding the runner actually supplies."""
    positions = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int64)
    block_table_tensor = torch.zeros(2, 8, dtype=torch.int32)
    block_table_tensor[:, 0] = torch.tensor([5, 9], dtype=torch.int32)

    slot_mapping = torch.full((10,), PAD_SLOT_ID, dtype=torch.int64)
    out = compute_kpool_tail_slot_mapping(
        slot_mapping, block_table_tensor, query_start_loc, positions, 6, 2, KPOOL
    )

    assert (out[6:] == PAD_SLOT_ID).all()
    assert (out[:6] >= 0).all()
