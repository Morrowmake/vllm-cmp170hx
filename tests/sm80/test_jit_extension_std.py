# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plugin's JIT-built CUDA extensions compile at torch's C++ standard.

torch's headers refuse a standard below their minimum, and a ``-std`` passed
by an extension overrides the one torch.utils.cpp_extension adds. The minimum
is read from torch's own header, so the test follows torch upgrades.
"""

import importlib
import os
import re

import pytest
import torch
import torch.utils.cpp_extension

EXTENSIONS = [
    ("vllm_sm80.distributed.custom_all_reduce_flags", "load_module"),
    ("vllm_sm80.distributed.host_shm_all_reduce", "_load_module"),
]

_CPLUSPLUS = {"11": 201103, "14": 201402, "17": 201703, "20": 202002,
              "23": 202302, "26": 202400}


def torch_min_cplusplus() -> int:
    header = os.path.join(os.path.dirname(torch.__file__), "include", "ATen",
                          "ATen.h")
    with open(header) as f:
        m = re.search(r"__cplusplus\s*<\s*(\d+)L", f.read())
    assert m, f"no __cplusplus check in {header}"
    return int(m.group(1))


def _captured_load(monkeypatch, tmp_path, module, func):
    mod = importlib.import_module(module)
    calls = []
    monkeypatch.setattr(torch.utils.cpp_extension, "load",
                        lambda **kw: calls.append(kw) or object())
    monkeypatch.setattr(mod, "_MODULE", None)
    monkeypatch.setattr(mod, "_build_dir", lambda: str(tmp_path))
    getattr(mod, func)()
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("module,func", EXTENSIONS)
def test_extension_flags_meet_torch_minimum_standard(monkeypatch, tmp_path,
                                                    module, func):
    kw = _captured_load(monkeypatch, tmp_path, module, func)
    minimum = torch_min_cplusplus()
    for key in ("extra_cflags", "extra_cuda_cflags"):
        for flag in kw.get(key) or []:
            m = re.fullmatch(r"-std=(?:c|gnu)\+\+(\d+)", flag)
            if m:
                assert _CPLUSPLUS[m.group(1)] >= minimum, (
                    f"{module} {key} {flag} is below torch's "
                    f"__cplusplus >= {minimum}")


def test_caller_std_flag_is_refused(tmp_path):
    from vllm_sm80 import jit_extension

    with pytest.raises(ValueError, match="-std"):
        jit_extension.load_cuda_extension("ext", "src", str(tmp_path),
                                          extra_cflags=["-std=c++17"],
                                          extra_cuda_cflags=[])
