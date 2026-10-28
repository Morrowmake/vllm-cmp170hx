# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer top-k determinism: canonical (lowest-index) choice on exact ties at
the top-k boundary (VLLM_SM80_TOPK_CANONICAL_TIES) and each row in ascending
order (VLLM_SM80_TOPK_SORTED). The kernels in ``topk_tiefix.py`` are shared by
the GLM kpool sparse indexer (``vllm/models/glm5next/nvidia/sparse_indexer.py``)
and the Qwen QSA indexer (``vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py``),
both through the switches and dispatch in ``topk_tiefix_gate.py``.
"""
