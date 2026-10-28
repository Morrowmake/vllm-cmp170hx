# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LiLiCorr drafter classes of the sm_80 fork.

The fork plugin (``vllm_sm80.plugin``) registers :class:`LiLiCorrForCausalLM`
in vLLM's model registry in place of the upstream class in
``vllm/model_executor/models/lilicorr.py`` (``LiLiCorrDraftModel``). LiLiCorr
builds DFlash2 decoder layers when its convolution is on; with the
load-following draft width (``adaptive_k.draft_by_load``) their grouped
convolutions read the block length from a device scalar, as DFlash2's do
(``vllm_sm80/spec_decode/dflash2_model.py``). The drafter also takes the
DFlash-family additions of ``dflash_draft.py``. Both classes subclass the
upstream ones and keep their names.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models import lilicorr
from vllm_sm80.spec_decode.dflash2_model import adopt_grouped_convs
from vllm_sm80.spec_decode.dflash_draft import DFlashDraftMixin


class LiLiCorr(lilicorr.LiLiCorr):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ):
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        adopt_grouped_convs(self, vllm_config)


class LiLiCorrForCausalLM(DFlashDraftMixin, lilicorr.LiLiCorrForCausalLM):
    model_cls = LiLiCorr
