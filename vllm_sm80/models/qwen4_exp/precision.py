# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp precision settings of the CMP 170HX (sm_80) fork.

Both are fields of the target model's text config, which the checkpoint does
not set; set them with ``--hf-overrides``, e.g.
``'{"text_config": {"moe_router_dtype": "float32", "mtp_hidden_norm": "per_stream"}}'``.

- ``moe_router_dtype``: ``"float32"`` makes the MoE router GEMM return FP32
  logits, read as upstream reads it for DeepSeek-V2 (``_get_moe_router_dtype``).
  Absent or any other value: the model dtype.
- ``mtp_hidden_norm``: ``"flat"`` (absent: the default) normalizes the MTP
  hidden input over the flattened hyper-connection stack, ``"per_stream"``
  normalizes every stream on its own.

Both are read from the target config, also for the MTP drafter: its layers are
built from the draft model config, which dict ``--hf-overrides`` do not reach.
Both are read once, when the owning module is constructed.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype

logger = init_logger(__name__)

_MTP_HIDDEN_NORMS = ("flat", "per_stream")


def target_text_config(vllm_config: VllmConfig):
    """The target model's text config, also when ``vllm_config`` carries the
    MTP drafter's model config."""
    target = getattr(vllm_config.speculative_config, "target_model_config", None)
    model_config = target if target is not None else vllm_config.model_config
    return model_config.hf_text_config


def apply_router_precision(vllm_config: VllmConfig, gate: nn.Module) -> None:
    """Select the router logits dtype for a Qwen4Exp sparse MoE block."""
    config = target_text_config(vllm_config)
    router_dtype = _get_moe_router_dtype(config)
    if router_dtype is not None:
        gate.set_out_dtype(router_dtype)
    logger.info_once(
        "Qwen4Exp router logits: %s (moe_router_dtype=%s)",
        "FP32" if router_dtype is not None else "model dtype",
        getattr(config, "moe_router_dtype", None),
    )


def mtp_hidden_norm_mode(vllm_config: VllmConfig) -> str:
    mode = getattr(target_text_config(vllm_config), "mtp_hidden_norm", "flat")
    if mode not in _MTP_HIDDEN_NORMS:
        raise ValueError(f"mtp_hidden_norm={mode!r} is not one of {_MTP_HIDDEN_NORMS}")
    return mode


class PerStreamGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm over ``[..., hc_count * hidden]`` that normalizes every
    hyper-connection stream on its own, with that stream's slice of the
    checkpoint's affine weight. The parameter is the flat norm's, so weights
    load unchanged.
    """

    def __init__(self, hidden_size: int, hc_count: int, eps: float) -> None:
        super().__init__(hidden_size * hc_count, eps=eps)
        self.hc_count = hc_count
        self.stream_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Like GemmaRMSNorm, apply (1 + weight) in FP32 before one cast back.
        streams = x.unflatten(-1, (self.hc_count, self.stream_size)).float()
        variance = streams.square().mean(dim=-1, keepdim=True)
        streams = streams * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.float().view(self.hc_count, self.stream_size)
        return (streams * (1.0 + weight)).to(x.dtype).flatten(-2)


def mtp_hidden_norm(
    vllm_config: VllmConfig, hidden_size: int, hc_count: int, eps: float
) -> GemmaRMSNorm:
    """Pre-FC norm of the MTP hidden input ``[T, hc_count * hidden]``: the flat
    GemmaRMSNorm, or the per-stream one under ``mtp_hidden_norm="per_stream"``.
    """
    mode = mtp_hidden_norm_mode(vllm_config)
    logger.info_once("Qwen4Exp MTP hidden norm: %s (mtp_hidden_norm)", mode)
    if mode == "per_stream":
        return PerStreamGemmaRMSNorm(hidden_size, hc_count, eps)
    return GemmaRMSNorm(hidden_size * hc_count, eps=eps)
