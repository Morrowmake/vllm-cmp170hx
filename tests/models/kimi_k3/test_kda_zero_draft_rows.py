# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream behaviour the fork's KDA speculative decoding relies on.

A decode row with no scheduled draft tokens stays a speculative row of the
GDN and Kimi-K3 KDA metadata: it still consumes the previous verify's
accepted-state offset, so it is not turned into a regular decode. The GLM KDA
builder (``vllm_sm80/models/glm5next/sm80_kda.py``) inherits this from the
GDN builder.
"""

import pytest
import torch

from tests.models.kimi_k3.test_kda_metadata import BLOCK_SIZE, DEVICE, _make_builder
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


_CASES = [
    pytest.param(
        BatchSpec(seq_lens=[50, 30], query_lens=[3, 3]),
        [2, 2],
        [1, 1],
        [False, False],
        id="pure-spec-decode",
    ),
    pytest.param(
        BatchSpec(seq_lens=[100, 65, 20], query_lens=[50, 1, 3]),
        [-1, -1, 2],
        [1, 1, 1],
        [True, False, False],
        id="mixed-prefill-and-spec-decode",
        # The builders stage mixed prefill/decode metadata through CUDA.
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="mixed build needs CUDA"
        ),
    ),
    pytest.param(
        BatchSpec(seq_lens=[40, 30], query_lens=[1, 1]),
        [0, 0],
        [3, 2],
        [False, False],
        id="no-scheduled-draft-tokens",
    ),
    pytest.param(
        BatchSpec(seq_lens=[50, 30], query_lens=[3, 2]),
        [2, 1],
        [1, 1],
        [False, False],
        id="ragged-spec-decode",
    ),
]


def _params():
    for builder_cls in (GDNAttentionMetadataBuilder, KimiK3KDAMetadataBuilder):
        for case in _CASES:
            yield pytest.param(
                builder_cls,
                *case.values,
                id=f"{builder_cls.__name__}-{case.id}",
                marks=case.marks,
            )


@pytest.mark.parametrize(
    (
        "builder_cls",
        "batch",
        "num_decode_draft_tokens",
        "num_accepted_tokens",
        "is_prefilling",
    ),
    list(_params()),
)
def test_zero_draft_rows_stay_speculative(
    builder_cls,
    batch: BatchSpec,
    num_decode_draft_tokens: list[int],
    num_accepted_tokens: list[int],
    is_prefilling: list[bool],
    monkeypatch,
):
    # CPU metadata construction must not initialize CUDA's pinned allocator.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    draft_tokens = torch.tensor(num_decode_draft_tokens, dtype=torch.int32)
    accepted = torch.tensor(num_accepted_tokens, dtype=torch.int32, device=DEVICE)
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool))
    metadata = _make_builder(builder_cls, 2, False).build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=draft_tokens,
        num_accepted_tokens=accepted,
    )
    spec_mask = draft_tokens >= 0
    assert metadata.num_spec_decodes == int(spec_mask.sum())
    torch.testing.assert_close(metadata.num_accepted_tokens, accepted[spec_mask])
