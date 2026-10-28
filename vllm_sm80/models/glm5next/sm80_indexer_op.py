# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash sparse attention indexer op of the sm_80 fork.

``vllm_sm80.plugin.register_glm_layers`` registers
:class:`SparseAttnIndexerKpool` out of tree in place of upstream's op of the
same name (``vllm/models/glm5next/nvidia/sparse_indexer.py``), so every GLM
indexer builds this subclass. Where DeepGEMM is not usable, the op's
``sparse_attn_indexer_kpool`` scores with the Triton fp8 MQA-logits kernels
(``sm80_indexer.py``), which read neither the MXFP4 cache nor the MX-scaled
(SM100) one: this class refuses them at construction.
"""

from vllm.logger import init_logger
from vllm.models.glm5next.nvidia import sparse_indexer as upstream
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported, mqa_logits_use_mx_scales

logger = init_logger(__name__)


class SparseAttnIndexerKpool(upstream.SparseAttnIndexerKpool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if current_platform.is_cuda() and not is_deep_gemm_supported():
            if self.use_fp4_cache or mqa_logits_use_mx_scales():
                raise RuntimeError(
                    "Sparse Attention Indexer MXFP4 or MX-scaled (SM100) cache "
                    "requires DeepGEMM."
                )
            logger.info_once(
                "DeepGEMM is not usable on this device; the sparse attention "
                "indexer will use the Triton fp8 MQA-logits kernels."
            )
