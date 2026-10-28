# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 checkpoints on sm_80, registered by the fork plugin (vllm_sm80/plugin.py).
  ``quant``   quantization method "exl3": the config, the linear and MoE
             methods, and the PLE hooks the Qwen4Exp model calls
  ``loader``  load format "exl3": the model loader and the weight loading
  ``ops``     loads the compiled library ``vllm_sm80._exl3_C`` (sources in
             csrc/exl3) and wraps its ops
  ``utils``   the checkpoint manifest, tensor headers, the PLE layout
             and the load ledger
The Qwen4Exp PLE storage built on these stays in
vllm_sm80/models/qwen4_exp/exl3_ple.py.
"""
