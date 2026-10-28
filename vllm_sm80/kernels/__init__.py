# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels tuned for NVIDIA sm_80 parts (GA100 class, e.g. the CMP 170HX).

Code dispatched from shared vLLM ops with shape gates any sm_80 model could
hit lives here, one subpackage per kernel family. Kernels only one model
calls live in that model's ``vllm/models/<model>/nvidia/ops/`` instead.

Importing this package does no work. Every family dispatched from shared
ops is opt-in behind its own switch, and its kernel modules are imported only
when that switch is on.

``exl3`` is different: it is the quantization method and load format "exl3",
registered by vllm_sm80/plugin.py. Its ``quant``, ``loader`` and ``utils``
modules are imported at plugin load in every process, and a checkpoint
quantized with EXL3 always selects it (there is no switch). ``ops``,
which loads the compiled ``vllm_sm80._exl3_C``, is imported only when
Exl3ModelLoader builds an EXL3 model or an EXL3 layer processes its weights.
"""
