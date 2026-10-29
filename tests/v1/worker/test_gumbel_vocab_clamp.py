# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU test for the vocabulary clamp on the sampler's block argmax.

Every argmax in the sampler works on a ``BLOCK_SIZE``-wide tile of the logits
row.  The last tile of a vocabulary that is not a multiple of ``BLOCK_SIZE``
has out-of-vocab tail lanes, which are loaded as ``-inf`` and masked out of the
result.  Masking does not remove them from the *reduction*: if every in-vocab
lane of that tile is also ``-inf`` or NaN -- a fully-masked row, a residual
distribution that rejection cancelled to nothing, a bad logits processor -- the
argmax can settle on a tail lane and emit a token id at or beyond
``vocab_size``.  Nothing downstream bounds a sampled token id.

In such a tile every lane is equally arbitrary, so clamping to ``vocab_size-1``
cannot change a well-defined result.  Refs vllm-project/vllm#50843.

    pytest -q tests/v1/worker/test_gumbel_vocab_clamp.py
"""

import pytest
import torch

from vllm.v1.worker.gpu.sample import gumbel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_all_masked_row_samples_an_in_vocab_token():
    vocab_size = 154_880
    logits = torch.full((1, vocab_size), float("-inf"), device="cuda")
    expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device="cuda")
    temperature = torch.ones(1, device="cuda")
    seed = torch.zeros(1, dtype=torch.int64, device="cuda")
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")

    sampled = gumbel.gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        positions,
        apply_temperature=True,
        is_drafting=False,
    )
    assert int(sampled.max()) < vocab_size
    assert int(sampled.min()) >= 0
