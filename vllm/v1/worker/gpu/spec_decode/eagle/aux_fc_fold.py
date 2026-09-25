# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fold a DFlash/EAGLE3 drafter's input projection into the pipeline stages.

The drafter's first op is ``fc(cat(aux_0, ..., aux_{n-1}))`` -- one linear map
over the concatenated aux hidden states, i.e. ``sum_i aux_i @ W_i^T`` with
``W_i`` the i-th column block of ``fc.weight``. Under pipeline parallelism the
aux states are produced on several stages and, by default, relayed one by one
to the last stage (``PPHandler.relay_aux_hidden_states``).

With ``VLLM_GLM5_PP_FOLD_DRAFT_FC=1`` every stage multiplies its own aux states
by their column blocks (bf16 inputs, fp32 output) and adds them to a running
fp32 partial sum that travels on the boundary under one key instead of the
aux states; the last stage hands the finished sum to the drafter in place of
the concatenation, and the drafter skips its ``fc``. The products and the sum
are the same numbers the single ``fc`` GEMM accumulates in fp32 before its one
rounding to the model dtype, so the result is identical up to the fp32
summation order. Each stage loads only its own ``W_i`` from the drafter
checkpoint (hidden_size x hidden_size per aux layer).
"""

import glob
import os
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

AUX_FC_PARTIAL_KEY = "aux_fc_partial"
_FC_WEIGHT = "fc.weight"


def _inner(model: nn.Module) -> nn.Module | None:
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import _inner_decoder

    return _inner_decoder(model)


def _draft_checkpoint_files(spec_config: Any) -> list[str]:
    path = spec_config.draft_model_config.model
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            path, allow_patterns=["*.safetensors"], local_files_only=True
        )
    return sorted(glob.glob(os.path.join(path, "*.safetensors")))


def load_fc_column_blocks(
    files: list[str], indices: list[int], hidden_size: int, num_aux: int
) -> dict[int, torch.Tensor]:
    """``{i: fc.weight[:, i*H:(i+1)*H]}`` for the requested aux indices."""
    from safetensors import safe_open

    for f in files:
        with safe_open(f, framework="pt") as st:
            names = set(st.keys())
            if _FC_WEIGHT not in names:
                continue
            sl = st.get_slice(_FC_WEIGHT)
            shape = tuple(sl.get_shape())
            if shape != (hidden_size, num_aux * hidden_size):
                raise ValueError(
                    f"{_FC_WEIGHT} has shape {shape}, expected "
                    f"({hidden_size}, {num_aux * hidden_size})"
                )
            return {
                i: sl[:, i * hidden_size : (i + 1) * hidden_size].contiguous()
                for i in indices
            }
    raise FileNotFoundError(f"{_FC_WEIGHT} not found in {files}")


def maybe_configure_aux_fc_fold(
    model: nn.Module,
    spec_config: Any,
    pp_handler: Any,
    load_dummy_weights: bool = False,
    files: list[str] | None = None,
    dtype: torch.dtype | None = None,
) -> bool:
    """Enable the fold on this stage if VLLM_GLM5_PP_FOLD_DRAFT_FC=1 and the
    model supports it. Must run after the aux layers and the relay are
    configured and before the persistent receive buffer is allocated.

    `dtype` is the activation dtype the aux states are captured in (the
    model config's dtype); the weight blocks are cast to it. Without it the
    first floating-point parameter of the stage that is not fp32 is used: a
    stage's first parameter can be an fp32 tensor (mHC mixing weights, router
    bias), which would make the blocks fp32 while the aux states are bf16."""
    if not envs.VLLM_GLM5_PP_FOLD_DRAFT_FC:
        return False
    from vllm.distributed.parallel_state import get_pp_group

    pp = get_pp_group()
    inner = _inner(model)
    reason = None
    if pp.world_size < 2:
        reason = "no pipeline parallelism"
    elif spec_config is None or spec_config.method != "dflash":
        reason = "only the DFlash drafter is supported"
    elif inner is None or not hasattr(inner, "enable_aux_fc_fold"):
        reason = f"{type(inner).__name__} does not support it"
    elif getattr(inner, "is_sequence_parallel", False):
        reason = "sequence parallelism"
    if reason is not None:
        logger.warning("VLLM_GLM5_PP_FOLD_DRAFT_FC ignored: %s", reason)
        return False

    hidden = inner.config.hidden_size
    num_aux = len(inner.aux_hidden_state_layers)
    base = inner._aux_slot_base_cached
    # Aux id i is captured after layer i - 1, so this stage owns the ids in
    # (start_layer, end_layer]; they follow the upstream ones in slot order.
    num_local = sum(
        inner.start_layer < i <= inner.end_layer for i in inner.aux_hidden_state_layers
    )
    local = list(range(base, base + num_local))
    device = next(inner.parameters()).device
    if dtype is None:
        dtype = next(
            (
                p.dtype
                for p in inner.parameters()
                if p.is_floating_point() and p.dtype != torch.float32
            ),
            torch.bfloat16,
        )
    if load_dummy_weights:
        blocks = {i: torch.zeros(hidden, hidden, dtype=dtype) for i in local}
    else:
        files = files if files is not None else _draft_checkpoint_files(spec_config)
        blocks = load_fc_column_blocks(files, local, hidden, num_aux)
    inner.enable_aux_fc_fold(
        {i: w.to(device=device, dtype=dtype) for i, w in blocks.items()}
    )

    # The boundary now carries the fp32 partial sum instead of aux states.
    if not pp.is_first_rank:
        base_make = inner.make_empty_intermediate_tensors

        def make_empty_with_partial(batch_size, dtype, device):
            tensors = base_make(batch_size, dtype, device)
            tensors[AUX_FC_PARTIAL_KEY] = torch.zeros(
                (batch_size, hidden), dtype=torch.float32, device=device
            )
            return tensors

        model.make_empty_intermediate_tensors = make_empty_with_partial
    if pp_handler is not None:
        pp_handler.aux_hidden_state_relay_keys = ()
    logger.info(
        "DFlash fc folded into the pipeline: this stage applies aux block(s) %s "
        "of %d and forwards an fp32 partial sum",
        local,
        num_aux,
    )
    return True


def mark_drafter_aux_fc_folded(speculator: Any) -> None:
    """Tell the last stage's drafter that its aux input is already the fc
    output (see DFlashQwen3ForCausalLM.combine_hidden_states)."""
    draft_model = getattr(speculator, "model", None)
    if draft_model is None:
        raise RuntimeError("fc fold: the speculator has no draft model")
    draft_model.aux_fc_folded = True
