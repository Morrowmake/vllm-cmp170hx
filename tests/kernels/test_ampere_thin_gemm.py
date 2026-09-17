# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the sm_80 thin-M BF16 GEMM integration.

Covers the two things that must hold on any machine, with or without a GPU:
the dispatch predicate admits exactly the shapes it was measured on, and with
``VLLM_GLM5_THIN_GEMM`` unset the unquantized-linear path is the upstream one --
the same object, not a wrapper that forwards.

The kernel's numerics are NOT tested here; they belong to
the standalone GPU correctness suite, which holds the kernel to
cuBLAS-equivalent error against an exact FP32 reference on a real sm_80 part.

No pytest in this deployment's venvs, so the file runs standalone. Every test is
a zero-argument ``test_*`` function, so pytest collects it unchanged if it is
ever available.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python tests/kernels/test_ampere_thin_gemm.py
"""

import contextlib
import sys

import torch

from vllm import envs
from vllm.ampere_thin_gemm import (
    ampere_thin_gemm,
    dispatch_threshold,
    thin_gemm_supported,
    thin_linear,
    use_ampere_thin_gemm,
)
from vllm.model_executor.layers.utils import (
    default_unquantized_gemm,
    dispatch_unquantized_gemm,
)

BF16 = torch.bfloat16
KERNEL_MOD = "vllm.ampere_thin_gemm.thin_gemm"


@contextlib.contextmanager
def setattr_temporarily(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _xw(M, N, K, dtype=BF16, wdtype=None):
    return (
        torch.zeros(M, K, dtype=dtype),
        torch.zeros(N, K, dtype=wdtype or dtype),
    )


# --------------------------------------------------------------- OFF path
def test_off_by_default():
    """The flag is opt-in. A default build must not enable this."""
    assert envs.VLLM_GLM5_THIN_GEMM is False
    assert use_ampere_thin_gemm() is False


def test_dispatch_is_upstream_object_when_off():
    """Not 'behaves the same' -- literally the upstream callable.

    This is what makes the unflagged path byte-identical: no extra frame, no
    predicate evaluated per GEMM, nothing imported from the package.
    """
    with setattr_temporarily(envs, "VLLM_GLM5_THIN_GEMM", False):
        assert dispatch_unquantized_gemm("auto") is default_unquantized_gemm


def test_kpool_gate_is_plain_linear_when_off():
    with setattr_temporarily(envs, "VLLM_GLM5_THIN_GEMM", False):
        x = torch.randn(4, 256, dtype=BF16)
        w = torch.randn(128, 256, dtype=BF16)
        assert torch.equal(thin_linear(x, w), torch.nn.functional.linear(x, w))


def test_kernel_module_not_imported_when_off():
    """The triton-importing module must stay out of a default process."""
    with setattr_temporarily(envs, "VLLM_GLM5_THIN_GEMM", False):
        sys.modules.pop(KERNEL_MOD, None)
        dispatch_unquantized_gemm("auto")
        thin_linear(torch.randn(4, 256, dtype=BF16),
                    torch.randn(128, 256, dtype=BF16))
        assert KERNEL_MOD not in sys.modules


# ------------------------------------------------------- dispatch predicate
def test_threshold_is_the_measured_value():
    """32, and it is a cudagraph capture size."""
    assert dispatch_threshold() == 32


def test_supported_in_win_region():
    for M in (1, 2, 3, 4, 8, 16, 24, 32):
        x, w = _xw(M, 4096, 4096)
        assert thin_gemm_supported(x, w, None), M


def test_rejected_above_threshold():
    """M=40..64 are capture sizes; they must fall through to cuBLAS.

    At M=40 the count-weighted step time is 0.799x -- dispatching there would
    be a 43 % regression, so this bound is load-bearing, not cosmetic.
    """
    for M in (33, 40, 48, 56, 64, 128, 2048):
        x, w = _xw(M, 4096, 4096)
        assert not thin_gemm_supported(x, w, None), M


def test_threshold_follows_env():
    with setattr_temporarily(envs, "VLLM_GLM5_THIN_GEMM_MAX_TOKENS", 8):
        x, w = _xw(16, 4096, 4096)
        assert not thin_gemm_supported(x, w, None)
        x, w = _xw(8, 4096, 4096)
        assert thin_gemm_supported(x, w, None)


def test_rejects_bias():
    x, w = _xw(4, 4096, 4096)
    assert not thin_gemm_supported(x, w, torch.zeros(4096, dtype=BF16))


def test_rejects_non_bf16():
    """No precision path but BF16 exists, in either operand."""
    for dt in (torch.float16, torch.float32):
        x, w = _xw(4, 4096, 4096, dtype=dt)
        assert not thin_gemm_supported(x, w, None), dt
        x, w = _xw(4, 4096, 4096, dtype=BF16, wdtype=dt)
        assert not thin_gemm_supported(x, w, None), dt


def test_rejects_3d_activation():
    """>2-D would need a reshape, i.e. a possible allocation on a graph path."""
    x = torch.zeros(2, 4, 4096, dtype=BF16)
    w = torch.zeros(4096, 4096, dtype=BF16)
    assert not thin_gemm_supported(x, w, None)


def test_rejects_k_strided_weight():
    """create_weights makes [N, K] K-contiguous; anything else is not ours."""
    x, w = _xw(4, 4096, 4096)
    assert not thin_gemm_supported(x, w.t().contiguous().t(), None)


def test_rejects_tiny_k():
    x, w = _xw(4, 2048, 64)
    assert not thin_gemm_supported(x, w, None)


def test_accepts_non_contiguous_activation():
    """vLLM hands F.linear a view of a padded cudagraph input buffer."""
    xb, w = _xw(8, 4096, 4096)
    assert thin_gemm_supported(xb[2:6], w, None)


def test_accepts_every_shape_in_the_table():
    """The distinct (N, K) the step actually runs, at the c1 batch M=4."""
    from vllm.ampere_thin_gemm.warmup import _TABLE_NK

    for N, K in _TABLE_NK:
        x, w = _xw(4, N, K)
        assert thin_gemm_supported(x, w, None) is (K >= 128), (N, K)


def test_kda_f_b_proj_sits_on_the_min_k_boundary():
    """K=128 f_b/g_b_proj are exactly at _MIN_K and must be included."""
    x, w = _xw(4, 2048, 128)
    assert thin_gemm_supported(x, w, None)


# ------------------------------------------------------------- fallthrough
def test_falls_back_without_touching_the_kernel():
    """An unsupported shape must take F.linear and never import triton."""
    sys.modules.pop(KERNEL_MOD, None)
    x = torch.randn(4, 64, dtype=BF16)          # K below _MIN_K
    w = torch.randn(128, 64, dtype=BF16)
    out = ampere_thin_gemm(None, x, w, None)
    assert torch.equal(out, torch.nn.functional.linear(x, w))
    assert KERNEL_MOD not in sys.modules


def test_warmup_is_a_noop_when_off():
    with setattr_temporarily(envs, "VLLM_GLM5_THIN_GEMM", False):
        from vllm.ampere_thin_gemm.warmup import warmup_ampere_thin_gemm

        warmup_ampere_thin_gemm(object(), [1, 2, 4, 8])  # must not raise


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - a report, not a handler
            failed += 1
            print(f"[FAIL] {name}: {exc!r}")
    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'} "
          f"({len(tests)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
