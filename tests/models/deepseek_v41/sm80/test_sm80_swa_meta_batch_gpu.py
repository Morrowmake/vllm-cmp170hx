# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_DEEPSEEK_V41_SWA_META_BATCH`` on one CUDA device: the cases of
``test_sm80_swa_meta_batch.py`` with the compiled kernel, launches counted
by the CUDA profiler (the torch ge and fill kernels by name).

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_swa_meta_batch_gpu.py
"""

import pytest
import torch

from tests.models.deepseek_v41.sm80 import swa_meta_batch_cases as cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_batched_metadata_is_the_per_group_build_in_one_launch():
    cases.check(cases.run_all(torch.device("cuda")))
