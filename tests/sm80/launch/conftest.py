# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAUNCH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "vllm_sm80", "launch")
for p in (_LAUNCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
