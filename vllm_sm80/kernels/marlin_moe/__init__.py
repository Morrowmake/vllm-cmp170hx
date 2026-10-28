# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 Marlin MoE paths called from the shared Marlin experts.

  ``extension``    loads and validates the optional compiled library
                   ``vllm_sm80._ampere_marlin_C`` (sources in
                   csrc/ampere_marlin)
  ``decode``       the compiled Marlin decode (``VLLM_AMPERE_MARLIN_DECODE_CUDA``)
  ``prefill``      the split-block Marlin prefill and its compiled tiles for
                   whole experts (``VLLM_SM80_MARLIN_PREFILL``)
  ``split_align``  the block lists ``prefill`` launches over
"""
