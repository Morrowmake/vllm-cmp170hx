# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code the CMP 170HX (sm_80) fork adds to vLLM and that more than one model
can reach.

- ``kernels``: kernels dispatched from shared vLLM ops, one subpackage per
  family (attention, indexer, kda, marlin_moe, mhc, moe, thin_gemm), and
  ``exl3``, the EXL3 quantization method and load format.
- ``attention``: the Triton sparse-MLA attention backend.
- ``distributed``: the host-staged and the flags-in-data custom all-reduce,
  and the opt-in PCIe peer-to-peer topology leg (``pcie_p2p``).
- ``fla``: autotune pins for the vendored flash-linear-attention kernels.
- ``spec_decode``: adaptive draft depth and the PP drafter fc fold.
- ``worker``: the PP draft tail and packed hop, and the memory attribution
  and state-index diagnostics.
- ``kv_planning``: KV cache planning rules: packed-group refit for MLA +
  Mamba hybrids (DFlash drafter fit, PP-aware state groups), speculative
  scratch in KV capacity, and the scheduler's eagle-group flags.
- ``failed_inputs``: fails only the requests whose model inputs a worker
  could not produce (an EXL3 PLE row that storage failed to read).
- ``table_segment``: a holder process that keeps checkpoint files (the
  DeepSeek V4.1 Engram tables) in sealed memory across server restarts,
  and the client a server maps them with (``VLLM_ENGRAM_SEGMENT``).
- ``parser_recovery``: provisional tool-call candidates for the streaming
  parser engine, held until validated; ``deepseek_v41_recovery`` recovers
  the damaged DeepSeek V4.1 tool calls with it (vllm/parser/deepseek_v41.py
  wraps its parser class).
- ``envs``: the fork's environment variables; ``vllm/envs.py`` registers
  them at the end of its own import.
- ``plugin`` and ``guard``: the ``vllm.general_plugins`` entry point
  (distribution in ``plugins/sm80``), which registers ``kernels.exl3``, and
  the startup check that it loaded.
- ``platform``: the fork's CUDA platform, selected by the
  ``vllm.platform_plugins`` entry point of the same distribution; the guard
  also checks that it is the platform in use on CUDA.

Kernels only one model calls live under ``vllm/models/<model>/nvidia/ops``.
Importing this package does no work: the upstream files that use a module
here import that module directly, and ``plugin`` imports the EXL3 modules it
registers.
"""
