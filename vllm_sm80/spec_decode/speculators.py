# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculator factory: vLLM's ``init_speculator``
(``vllm/v1/worker/gpu/spec_decode/__init__.py``) asks :func:`create_speculator`
first and builds its own speculator when it returns None.

- ``dspark``: :func:`vllm_sm80.spec_decode.dspark_width.create_dspark_speculator`
  (draft widths, shard argmax, the DSpark switches).
- ``dflash`` with a plain DFlash2 drafter (``DFlash2DraftModel``,
  ``DFlash2BeamDraftModel``; not XPress or LiLiCorr, which vLLM's chain
  picks first): :class:`Sm80DFlash2Speculator` (load-following draft width,
  split draft tail).

Everything else is vLLM's own speculator.
"""

import torch

from vllm.config import VllmConfig
from vllm.config.speculative import XPRESS_ARCHITECTURES

_DFLASH2_ARCHITECTURES = frozenset({"DFlash2DraftModel", "DFlash2BeamDraftModel"})


def create_speculator(vllm_config: VllmConfig, device: torch.device):
    """The plugin's speculator for this config, or None (vLLM builds its own)."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dspark":
        from vllm_sm80.spec_decode.dspark_width import create_dspark_speculator

        return create_dspark_speculator(vllm_config, device)
    if speculative_config.method != "dflash":
        return None
    architectures = speculative_config.draft_model_config.architectures
    if (
        any(arch in XPRESS_ARCHITECTURES for arch in architectures)
        or "LiLiCorrDraftModel" in architectures
        or not _DFLASH2_ARCHITECTURES.intersection(architectures)
    ):
        return None
    from vllm_sm80.spec_decode.dflash2_speculator import Sm80DFlash2Speculator

    return Sm80DFlash2Speculator(vllm_config, device)
