# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-capture warmup of the sm_80 mHC decode kernels (see
``vllm_sm80/kernels/warmup_utils.py`` for why it must run before capture)."""

from vllm.logger import init_logger
from vllm_sm80.kernels.mhc import decode_v2_max_rows, use_ampere_mhc_decode_v3
from vllm_sm80.kernels.warmup_utils import find_module, token_sizes

logger = init_logger(__name__)


def warmup_from_worker(worker, capture_sizes) -> None:
    """No-op unless VLLM_AMPERE_DECODE_KERNELS and VLLM_GLM5_DECODE_MHC_V2."""
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MHC_V2:
        return
    _warmup_mhc_v3(
        worker.get_model(), worker.device, capture_sizes, use_ampere_mhc_decode_v3
    )


def _warmup_mhc_v3(model, device, capture_sizes, gate) -> None:
    """Compile the v3 mHC kernels for every capture size its gate accepts.

    The v3 kernels keep no module-level buffers (outputs and split partials
    are allocated per call, from the graph pool during capture), so compiling
    before capture is all that is needed.
    """
    from vllm_sm80.kernels.mhc import decode as mhc_decode

    layer = find_module(model, "mhc_fused_post_pre_op")
    if layer is None:
        return
    hidden = int(layer.hidden_size)
    hc = int(layer.n)
    sinkhorn = int(layer.mhc_sinkhorn_iterations)
    bound = decode_v2_max_rows(hc, hidden)
    ms = [m for m in token_sizes(capture_sizes, bound) if gate(m, hc, hidden)]
    if not ms:
        logger.info("sm_80 mHC decode v3 requested but its gate is closed "
                    "(hc=%d, hidden=%d); using the previous paths.", hc, hidden)
        return
    mhc_decode.warmup(ms, hidden=hidden, hc=hc, sinkhorn=sinkhorn, device=device)
    logger.info("sm_80 mHC decode v3 on for M <= %d; warmed up M in %s.", bound, ms)
