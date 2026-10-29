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

Our state reads ``self.cache_config.mamba_block_size`` at request-admission
time, so the hazard cannot occur.  ``block_size`` is not the divisor: it is the
minimum over prefix-cacheable groups, which a DFlash drafter group with its own
block pulls below the Mamba block.  This test pins the late read.

    pytest -q tests/v1/worker/test_mamba_block_size_resolution.py
"""

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


def test_add_request_reads_the_divisor_from_the_live_config(monkeypatch):
    """The state column is seeded with the Mamba block size the config holds
    at admission, not one read at construction or the minimum block size."""

    def base_init(self, vllm_config, model, encoder_cache, device):
        self.device = device
        self.max_num_reqs = 2
        self.rope_state = None
        self.prompt_embeds_state = None

    monkeypatch.setattr(DefaultModelState, "__init__", base_init)
    cache_config = SimpleNamespace(
        block_size=16,
        mamba_block_size=16,
        mamba_cache_mode="align",
        use_kda_recoverssm=False,
    )
    state = mamba_hybrid.MambaHybridModelState(
        SimpleNamespace(cache_config=cache_config),
        model=None,
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    # update_block_size_for_backend() aligns the Mamba block after load_model();
    # a DFlash drafter group keeps its own block, so block_size stays below it.
    cache_config.mamba_block_size = 1152
    state.add_request(1, SimpleNamespace(num_computed_tokens=11_520))
    assert state._mamba_state_idx_gpu[1].item() == (11_520 - 1) // 1152
