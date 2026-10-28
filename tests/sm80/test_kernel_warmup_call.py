# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kernel_warmup.py (upstream-owned) makes exactly one call into the sm80
package, vllm_sm80/kernels/warmup.py, which warms every sm_80 kernel family
before capture."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import kernel_warmup as kw
from vllm_sm80.kernels import warmup as sm80_warmup


def test_one_call_into_the_sm80_package():
    tree = ast.parse(Path(kw.__file__).read_text())
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("vllm_sm80")
    ]
    assert [(n.module, [a.name for a in n.names]) for n in imports] == [
        ("vllm_sm80.kernels.warmup", ["warmup_from_worker"])
    ]
    src = Path(kw.__file__).read_text()
    assert src.count("sm80_warmup(") == 1
    assert "VLLM_AMPERE_" not in src and "VLLM_SM80_" not in src


def test_kernel_warmup_hands_the_capture_sizes_over(monkeypatch):
    seen = []
    monkeypatch.setattr(sm80_warmup, "warmup_from_worker",
                        lambda worker, sizes: seen.append((worker, sizes)))
    for name in ("qwen_triton_warmup", "qwen_vl_triton_warmup",
                 "mamba_triton_warmup"):
        monkeypatch.setattr(kw, name, lambda *a, **k: None)
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_jit_warmup=False),
            model_config=None,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2])))
    try:
        kw.kernel_warmup(worker)
    except Exception:  # later upstream warmups need a real worker
        pass
    assert seen == [(worker, [1, 2])]


@pytest.mark.parametrize("mhc,smla", [("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")])
def test_prefill_switches_log_a_startup_line_each(monkeypatch, mhc, smla):
    monkeypatch.setenv("VLLM_SM80_MHC_PREFILL", mhc)
    monkeypatch.setenv("VLLM_SM80_SPARSE_MLA_PREFILL", smla)
    lines = []
    monkeypatch.setattr(sm80_warmup.logger, "info_once",
                        lambda msg, *a, **k: lines.append(msg % a))
    sm80_warmup._log_prefill_switches()
    want = []
    if mhc == "1":
        want.append("sm_80 mHC prefill pre-norm GEMM on (VLLM_SM80_MHC_PREFILL=1): "
                    "steps from the mhc/prefill_min_rows row on sm_80.")
    if smla == "1":
        want.append("sm_80 sparse-MLA prefill on (VLLM_SM80_SPARSE_MLA_PREFILL=1): "
                    "steps and contexts from the attention_prefill/sparse_mla_min_rows "
                    "and sparse_mla_min_ctx rows on sm_80.")
    assert lines == want
