# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 attention kernels: Triton sparse-MLA decode and prefill (with the
PP-shape prefill), the fp8 MQA-logits indexer kernels, the e4m3 helpers and
the MLA KV remap-and-cache kernel. The backend that calls them is
``vllm_sm80/attention/triton_mla_sparse.py``; the GLM sparse indexer calls
the logits kernels.
"""
