# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream behaviour the fork's KDA speculative decoding relies on.

A decode row with no scheduled draft tokens stays a speculative row of the
Kimi-K3 KDA metadata: it still consumes the previous verify's accepted-state
offset, so it is not turned into a regular decode. The GDN builder's side
(which the GLM KDA builder, ``vllm_sm80/models/glm5next/sm80_kda.py``,
inherits) is tests/v1/attention/test_gdn_zero_draft_spec_rows.py.
"""

import torch

from tests.models.kimi_k3.test_kda_metadata import BLOCK_SIZE, DEVICE, _make_builder
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder


def test_zero_draft_rows_stay_speculative(monkeypatch):
    # CPU metadata construction must not initialize CUDA's pinned allocator.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[1, 1])
    draft_tokens = torch.tensor([0, 0], dtype=torch.int32)
    # Distinct counts: each row keeps its own accepted-state offset.
    accepted = torch.tensor([3, 2], dtype=torch.int32, device=DEVICE)
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor([False, False], dtype=torch.bool))
    metadata = _make_builder(KimiK3KDAMetadataBuilder, 2, False).build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=draft_tokens,
        num_accepted_tokens=accepted,
    )
    assert metadata.num_spec_decodes == 2
    torch.testing.assert_close(metadata.num_accepted_tokens, accepted)
