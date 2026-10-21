# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MTP placeholders must not pass raw IDs through on the final PP rank."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils


def _inner(embed):
    inner = nn.Module()
    inner.embed_tokens = embed
    return inner


def test_shared_mtp_embedding_resolves_on_last_target_stage(monkeypatch):
    monkeypatch.setattr(
        eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=2)
    )
    target_embed = nn.Embedding.from_pretrained(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    draft_inner = _inner(PPMissingLayer())
    # Qwen's shared-mode placeholder has no has_own_embed_tokens attribute;
    # the legacy PP1 proposer also relies on that absence.
    eagle_utils.maybe_share_target_embed(nn.Module(), draft_inner, _inner(target_embed))
    torch.testing.assert_close(
        draft_inner.embed_tokens(torch.tensor([2, 0])),
        torch.tensor([[5.0, 6.0], [1.0, 2.0]]),
    )


def test_shared_mtp_embedding_cannot_use_a_missing_target(monkeypatch):
    monkeypatch.setattr(
        eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=2)
    )
    with pytest.raises(RuntimeError, match="needs the target input embedding"):
        eagle_utils.maybe_share_target_embed(
            nn.Module(), _inner(PPMissingLayer()), _inner(PPMissingLayer())
        )
