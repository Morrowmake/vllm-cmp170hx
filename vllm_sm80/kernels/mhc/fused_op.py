# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 mHC decode kernel (``decode.py``) behind ``MHCFusedPostPreOp``.

``vllm_sm80.plugin`` registers :class:`Sm80MHCFusedPostPreOp` out of tree for
``MHCFusedPostPreOp`` (``vllm/model_executor/layers/mhc.py``) while
``VLLM_AMPERE_DECODE_KERNELS`` and ``VLLM_GLM5_DECODE_MHC_V2`` are set. Its
``forward_cuda`` calls the custom op ``vllm::sm80_mhc_fused_post_pre`` in
place of ``vllm::mhc_fused_post_pre_tilelang``, with the same arguments and
the same fake implementation. The op body decides per call, on the host:
for 1 <= M <= the mhc/decode_v2_max_rows row of the sm80 tuning table it
runs the v3 decode kernel (same signature, same four outputs, same dtypes),
else upstream's TileLang body unchanged.
"""

import torch

from vllm.model_executor.kernels.mhc import tilelang as upstream
from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_sm80.kernels.mhc import use_ampere_mhc_decode_v3


def sm80_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_fused_post_pre_tilelang``, with the v3 decode kernel for the
    shapes :func:`use_ampere_mhc_decode_v3` admits. ``n_splits`` / ``tile_n``
    are part of the shared MHC operator API; v3 picks its split from the
    runtime shape."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.view(-1, hc_mult, hidden_size).shape[0]
    if not use_ampere_mhc_decode_v3(
        num_tokens, hc_mult, hidden_size, norm_weight=norm_weight
    ):
        return upstream.mhc_fused_post_pre_tilelang(
            x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps,
        )
    # The norm weight exactly as the TileLang body hands it to its kernels.
    assert norm_weight is not None and norm_weight.shape == (hidden_size,)
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()
    from vllm_sm80.kernels.mhc.decode import mhc_fused_post_pre as _mhc_v3

    return _mhc_v3(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )


direct_register_custom_op(
    op_name="sm80_mhc_fused_post_pre",
    op_func=sm80_mhc_fused_post_pre,
    mutates_args=[],
    fake_impl=upstream._mhc_fused_post_pre_tilelang_fake,
)


class Sm80MHCFusedPostPreOp(MHCFusedPostPreOp):
    """``MHCFusedPostPreOp`` whose CUDA path is ``vllm::sm80_mhc_fused_post_pre``."""

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.sm80_mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )


def register() -> None:
    """Register :class:`Sm80MHCFusedPostPreOp` out of tree (idempotent)."""
    from vllm.model_executor.custom_op import op_registry_oot

    registered = op_registry_oot.get("MHCFusedPostPreOp")
    if registered is None:
        MHCFusedPostPreOp.register_oot(Sm80MHCFusedPostPreOp)
    elif registered is not Sm80MHCFusedPostPreOp:
        raise RuntimeError(
            f"MHCFusedPostPreOp is already replaced by {registered}"
        )
