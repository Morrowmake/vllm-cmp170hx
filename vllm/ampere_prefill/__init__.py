# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill kernels tuned for NVIDIA CMP 170HX (GA100, sm_80, 70 SMs).

Vendored prefill kernels. Every one of
these is gated behind ``VLLM_GLM5_PREFILL_KERNELS``; with the flag unset nothing
in this package is imported or called. Two exceptions have their own switches
and are imported only when those are set: ``kda_prefill`` (KDA chunked prefill
with 64 heads per card, ``VLLM_GLM5_PP_KDA_PREFILL``, or 16 under tensor
parallel 4, ``VLLM_GLM5_TP4_KDA_PREFILL``) and the split-block
Marlin MoE prefill (``pp_marlin_prefill``, ``moe_split_align``,
``VLLM_GLM5_PP_MARLIN_PREFILL``, or ``VLLM_GLM5_TP4_MARLIN_PREFILL`` for the
N=512 shards of tensor parallel 4).

The two measured port hazards are the per-call ``fn`` prepack and sparse
MLA's short-context regression.
"""
