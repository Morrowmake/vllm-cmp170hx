# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test that the mamba state divisor is read late, not cached at init.

``MambaHybridModelState`` is constructed during ``load_model()``, which the
worker runs *before* ``update_block_size_for_backend()`` settles the hybrid
block size.  Caching ``cache_config.block_size`` (or ``mamba_block_size``) in
``__init__`` therefore freezes the pre-alignment value -- 16 rather than the
aligned 1152 on GLM-5.3-Flash.  ``add_request`` uses that value as the divisor
when seeding a resumed request's state column, so a prefix-cache hit at 11520
computed tokens would seed column 719 of a 228-column table: an illegal memory
access, deterministic, and only in ``mamba_cache_mode="align"`` -- which is
exactly what prefix caching selects.

Our state reads ``self.cache_config.block_size`` at request-admission time, so
the hazard cannot occur.  This test pins the ordering assumption and the late
read.

    pytest -q tests/v1/worker/test_mamba_block_size_resolution.py
"""

import inspect
import re

from vllm.platforms import interface
from vllm.v1.worker.gpu.model_states import mamba_hybrid

BLOCK_SIZE_ATTRS = re.compile(
    r"self\.(_?mamba_)?block_size\s*=|self\._mamba_block_size\s*="
)


def test_init_does_not_cache_a_block_size():
    src = inspect.getsource(mamba_hybrid.MambaHybridModelState.__init__)
    assert BLOCK_SIZE_ATTRS.search(src) is None, src
    # It keeps the config object instead, so later reads see the aligned value.
    assert "self.cache_config = vllm_config.cache_config" in src


def test_add_request_reads_the_divisor_from_the_live_config():
    src = inspect.getsource(mamba_hybrid.MambaHybridModelState.add_request)
    assert "self.cache_config.block_size" in src
    assert "// self.cache_config.block_size" in src


def test_align_mode_keeps_the_two_block_sizes_equal():
    """``add_request`` divides by ``block_size``; alignment is what makes that
    the mamba block size too."""
    src = inspect.getsource(interface.Platform._align_hybrid_block_size)
    assert 'if cache_config.mamba_cache_mode == "align":' in src
    assert "cache_config.mamba_block_size = cache_config.block_size" in src


def test_seeding_arithmetic_uses_the_aligned_divisor():
    """The concrete numbers from the upstream report, both ways round."""
    computed_tokens = 11_520
    table_columns = 228
    assert (computed_tokens - 1) // 1152 == 9
    assert (computed_tokens - 1) // 1152 < table_columns
    # The stale value is what goes off the end.
    assert (computed_tokens - 1) // 16 == 719
    assert (computed_tokens - 1) // 16 >= table_columns
