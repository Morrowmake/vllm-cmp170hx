# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the vocabulary clamp on the sampler's block argmax.

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

A Triton kernel cannot run on a CPU, so the property is checked against a
mirror of the tile arithmetic and the clamp is pinned in the kernel sources.

    pytest -q tests/v1/worker/test_gumbel_vocab_clamp.py
"""

import inspect

import pytest
import torch

from vllm.v1.worker.gpu.sample import gumbel
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils

BARE = "token_id = block_idx * BLOCK_SIZE + idx"
CLAMPED = "token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)"


def kernel_source(fn) -> str:
    return getattr(fn, "src", None) or inspect.getsource(fn)


# ------------------------------------------------------------------ mirror


def tile_token_id(block_idx: int, block_size: int, lane: int, vocab_size: int) -> int:
    """What the kernels now compute for the winning lane of a tile."""
    return min(block_idx * block_size + lane, vocab_size - 1)


@pytest.mark.parametrize("vocab_size", [151_936, 154_880, 200_064, 4099])
@pytest.mark.parametrize("block_size", [1024, 8192])
def test_every_lane_of_the_last_tile_stays_in_vocab(vocab_size, block_size):
    """The clamp has to hold for *any* lane the reduction might pick, because
    a degenerate tile gives no control over which one wins."""
    last_block = (vocab_size - 1) // block_size
    overflowed = 0
    for lane in range(block_size):
        bare = last_block * block_size + lane
        overflowed += bare >= vocab_size
        assert 0 <= tile_token_id(last_block, block_size, lane, vocab_size) < vocab_size
    if vocab_size % block_size:
        assert overflowed == block_size - vocab_size % block_size
    else:
        assert overflowed == 0


def test_the_clamp_is_inert_for_a_well_defined_winner():
    """A real argmax never selects a masked lane, so nothing moves."""
    vocab_size, block_size = 154_880, 8192
    for block_idx in range((vocab_size + block_size - 1) // block_size):
        last_real = min(block_size, vocab_size - block_idx * block_size) - 1
        for lane in (0, last_real // 2, last_real):
            assert tile_token_id(block_idx, block_size, lane, vocab_size) == (
                block_idx * block_size + lane
            )


# ------------------------------------------------------------ source guard


@pytest.mark.parametrize(
    "fn",
    [
        gumbel._gumbel_sample_kernel,
        rejection_sampler_utils._compute_local_logits_stats_kernel,
        rejection_sampler_utils._resample_kernel,
    ],
)
def test_no_kernel_emits_an_unclamped_token_id(fn):
    src = kernel_source(fn)
    assert BARE not in src, f"{fn} still emits an unclamped token id"


def test_the_clamp_is_present_in_every_argmax_kernel():
    sources = [
        kernel_source(gumbel._gumbel_sample_kernel),
        kernel_source(rejection_sampler_utils._resample_kernel),
    ]
    for src in sources:
        assert CLAMPED in src


def test_the_module_sources_carry_no_bare_token_id():
    """Catches a new argmax kernel added without the clamp."""
    for module in (gumbel, rejection_sampler_utils):
        src = inspect.getsource(module)
        assert BARE not in src, module.__name__


# ------------------------------------------------------------------- GPU


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
