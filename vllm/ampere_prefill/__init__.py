# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill kernels tuned for NVIDIA CMP 170HX (GA100, sm_80, 70 SMs).

Vendored prefill kernels. Every one of
these is gated behind ``VLLM_GLM5_PREFILL_KERNELS``; with the flag unset nothing
in this package is imported or called. The exception is ``kda_prefill``
(KDA chunked prefill with 64 heads per card), which has its own switch,
``VLLM_GLM5_PP_KDA_PREFILL``, and is imported only when that is set.

The two measured port hazards are the per-call ``fn`` prepack and sparse
MLA's short-context regression.
"""
