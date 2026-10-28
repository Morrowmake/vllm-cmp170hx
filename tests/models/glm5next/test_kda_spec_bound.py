# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM KDA metadata builder (``vllm_sm80/models/glm5next/sm80_kda.py``).

It is the GDN builder plus the step's bound on the tokens of a speculative
request (``spec_max_query_len``, the fused KDA decode kernel's token tile):
every GDN field equals the GDN builder's, and the bound is the batch's
max_query_len capped at num_spec + 1, or None without speculative rows. A
decode row with no scheduled draft tokens still counts as speculative.
"""

import dataclasses

import pytest
import torch

from tests.models.kimi_k3.test_kda_metadata import BLOCK_SIZE, DEVICE, _make_builder
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm_sm80.models.glm5next.sm80_kda import Glm5NextGDNAttentionMetadataBuilder
from vllm_sm80.kernels.kda.recover import Glm5NextGDNAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)


def _assert_same(got, want, name: str) -> None:
    if isinstance(want, torch.Tensor):
        assert torch.equal(got, want), name
    elif dataclasses.is_dataclass(want):
        assert type(got) is type(want), name
        for field in dataclasses.fields(want):
            _assert_same(
                getattr(got, field.name), getattr(want, field.name), name
            )
    elif isinstance(want, dict):
        assert got.keys() == want.keys(), name
        for key in want:
            _assert_same(got[key], want[key], name)
    else:
        assert got == want, name


@pytest.mark.parametrize(
    (
        "batch",
        "num_decode_draft_tokens",
        "num_accepted_tokens",
        "num_speculative_tokens",
        "is_prefilling",
        "expected_spec_max_query_len",
    ),
    [
        pytest.param(
            BatchSpec(seq_lens=[50, 30], query_lens=[3, 3]),
            [2, 2],
            [1, 1],
            2,
            [False, False],
            3,
            id="pure-spec-decode",
        ),
        pytest.param(
            BatchSpec(seq_lens=[100, 65, 20], query_lens=[50, 1, 3]),
            [-1, -1, 2],
            [1, 1, 1],
            2,
            [True, False, False],
            3,
            id="mixed-prefill-and-spec-decode",
            # The builder stages mixed prefill/decode metadata through CUDA.
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="mixed build needs CUDA"
            ),
        ),
        pytest.param(
            BatchSpec(seq_lens=[40, 30], query_lens=[1, 1]),
            None,
            None,
            0,
            [False, False],
            None,
            id="regular-decode",
        ),
        pytest.param(
            BatchSpec(seq_lens=[40, 30], query_lens=[1, 1]),
            [0, 0],
            [3, 2],
            2,
            [False, False],
            1,
            id="no-scheduled-draft-tokens",
        ),
        pytest.param(
            BatchSpec(seq_lens=[50, 30], query_lens=[3, 2]),
            [2, 1],
            [1, 1],
            2,
            [False, False],
            3,
            id="ragged-spec-decode",
        ),
    ],
)
def test_glm_kda_builder_adds_the_spec_bound(
    batch: BatchSpec,
    num_decode_draft_tokens: list[int] | None,
    num_accepted_tokens: list[int] | None,
    num_speculative_tokens: int,
    is_prefilling: list[bool],
    expected_spec_max_query_len: int | None,
    monkeypatch,
):
    # CPU metadata construction must not initialize CUDA's pinned allocator.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    kwargs: dict[str, torch.Tensor] = {}
    if num_decode_draft_tokens is not None:
        assert num_accepted_tokens is not None
        kwargs = {
            "num_decode_draft_tokens_cpu": torch.tensor(
                num_decode_draft_tokens, dtype=torch.int32
            ),
            "num_accepted_tokens": torch.tensor(
                num_accepted_tokens, dtype=torch.int32, device=DEVICE
            ),
        }
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool))

    reference = _make_builder(
        GDNAttentionMetadataBuilder, num_speculative_tokens, False
    ).build(0, common_attn_metadata, **kwargs)
    actual = _make_builder(
        Glm5NextGDNAttentionMetadataBuilder, num_speculative_tokens, False
    ).build(0, common_attn_metadata, **kwargs)

    assert type(actual) is Glm5NextGDNAttentionMetadata
    assert actual.spec_max_query_len == expected_spec_max_query_len
    for field in dataclasses.fields(GDNAttentionMetadata):
        _assert_same(
            getattr(actual, field.name), getattr(reference, field.name), field.name
        )


def test_glm_kda_builder_retypes_the_gdn_metadata_in_place(monkeypatch):
    """No per-build copy: the object the GDN builder returns is the one the
    GLM builder returns, with the GLM class and the bound set."""
    sentinel = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=6,
        num_actual_tokens=6,
    )
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "build", lambda self, *args, **kwargs: sentinel
    )
    builder = object.__new__(Glm5NextGDNAttentionMetadataBuilder)
    builder.num_spec = 2
    builder._kda_recover = None
    common = type("Common", (), {"max_query_len": 3})()
    out = builder.build(0, common)
    assert out is sentinel
    assert type(out) is Glm5NextGDNAttentionMetadata
    assert out.spec_max_query_len == 3
    # The GLM class adds exactly one field to the GDN layout.
    gdn = [f.name for f in dataclasses.fields(GDNAttentionMetadata)]
    glm = [f.name for f in dataclasses.fields(Glm5NextGDNAttentionMetadata)]
    assert glm == gdn + ["spec_max_query_len"]
    assert not hasattr(GDNAttentionMetadata, "__slots__")
