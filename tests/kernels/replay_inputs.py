# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Captured served kernel inputs for the replay tests.

VLLM_SM80_REPLAY_DIR names a directory with one subdirectory per capture
family. A replay test reads its family's records and skips without them:

- ``kda``: KDA commit payloads (``*-cm-*.pt``), test_ampere_kda_fold.py.
- ``kda_pre``: KDA chunked prefill calls (``p<pp>_L<layer>_b<T>_<n>.pt``),
  test_ampere_pp_kda_prefill.py and test_ampere_tp4_kda_prefill.py.
- ``smla_pre``: sparse-MLA prefill calls (``p*_b*_*.pt``),
  test_ampere_pp_sparse_mla_prefill.py and
  test_ampere_smla_prefill_pred_load.py.
- ``idx_glue``: indexer dual-GEMM decode calls (``*.pt`` holding x, w and
  head_dim), test_ampere_idx_glue.py.
- ``moe_pre``: Marlin MoE prefill calls (``*.pt``) whose ``weights`` name a
  file under the replay directory (``once/``),
  test_ampere_pp_marlin_prefill.py and test_ampere_tp4_marlin_prefill.py.

VLLM_SM80_REPLAY_LIMIT caps the records a test reads; 0 reads every record.
Unset, each test reads its own number (limit's default).
"""

import glob
import os

import vllm.envs as envs


def root() -> str:
    """The replay directory ('' when VLLM_SM80_REPLAY_DIR is unset)."""
    return envs.VLLM_SM80_REPLAY_DIR


def files(family: str, pattern: str = "*.pt") -> list[str]:
    """The family's records matching ``pattern``, sorted by path; none
    without a replay directory."""
    if not root():
        return []
    return sorted(glob.glob(os.path.join(root(), family, pattern)))


def limit(default: int) -> int:
    """VLLM_SM80_REPLAY_LIMIT, else ``default``; 0 means every record."""
    value = envs.VLLM_SM80_REPLAY_LIMIT
    return default if value is None else value
