# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 device probe every sm_80 kernel gate ends with."""

import torch

# Cached device-capability answer. Resolved lazily so that importing this
# module never touches a CUDA device (the CPU tests import it with
# CUDA_VISIBLE_DEVICES="" and assert torch.cuda.is_initialized() is False),
# and cached so the 166 gate evaluations per decode step cost one bool load.
# The CPU tests set it directly; see tests/kernels/test_ampere_decode.py.
_SM80_CACHE: bool | None = None


def _is_sm80() -> bool:
    """True on a GA100-class part, False anywhere else and with no device.

    Guarded: a missing or broken CUDA install must make the gate return False,
    never raise, and never initialise CUDA on a CPU-only box.
    """
    global _SM80_CACHE
    if _SM80_CACHE is None:
        try:
            from vllm.platforms import current_platform

            _SM80_CACHE = bool(
                current_platform.is_cuda()
                and torch.cuda.is_available()
                and current_platform.is_device_capability(80)
            )
        except Exception:  # no driver, no device, stub platform
            _SM80_CACHE = False
    return _SM80_CACHE
