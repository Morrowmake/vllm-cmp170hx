# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the invariant that keeps kpool-tail padding harmless.

``compute_kpool_tail_slot_mapping`` rewrites the real tokens of the batch onto
each request's circular tail block and *copies the caller's padding through
untouched*.  That is only safe because of a chain of facts elsewhere in the
tree, and the failure it would cause is an illegal memory access under FULL
cudagraph replay -- not a wrong answer -- so it is worth pinning:

1. ``CircularBufferSpec.uses_slot_mapping`` is False, so the generic
   slot-mapping kernel never writes real slots for the tail group;
2. that kernel writes ``PAD_SLOT_ID`` for mapping-disabled groups and clears
   the rest of the buffer to ``PAD_SLOT_ID``;
3. ``PAD_SLOT_ID`` is negative;
4. the tail-seed kernel returns early on a negative slot.

Break any one of them and padded rows start addressing real pool-cache slots.
This file pins 1 and 3 on CPU; tests/v1/worker/test_gpu_kpool_tail_slot_mapping.py
runs the generic kernel for 2.

    pytest -q tests/v1/attention/test_kpool_tail_padding_invariant.py
"""

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import CircularBufferSpec

KPOOL = 4


def tail_spec():
    return CircularBufferSpec(
        block_size=KPOOL,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
    )


def test_the_pad_sentinel_is_negative():
    """Every downstream guard is a sign test, not an equality test."""
    assert PAD_SLOT_ID < 0


def test_the_tail_group_never_gets_a_generic_slot_mapping():
    assert tail_spec().uses_slot_mapping is False
