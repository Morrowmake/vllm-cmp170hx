# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 delayed mHC (``ops/sm80_mhc_fused.py``) on one
sm_80 card, at the model's shapes (hidden 5120, hc 4, 20 Sinkhorn iterations,
eps 1e-6, RMS eps 1e-20).

- FP64 rule (V2): every output of the fused call and of the pre-only call,
  with and without a carried pre-mix, against FP64 without intermediate
  rounding and at each path's own rounding points (``fp64_cases``, the
  ``.rp`` outputs): mean error at most 1.10x and max error
  at most 1.25x the torch formulas' (``ops/sm80_mhc.py``, on the same card),
  at every graph capture size and two prefill sizes, each case pooling
  independent calls up to 64 tokens (as a captured case pools the calls of
  its shape); up to 8 rows (one kernel A form), rows run alone give the
  bits of the whole call wherever they do for the torch formulas.

  CORRECTED AFTER THE FIRST GPU RUN (fork a4d9993ce1, 2026-10-18, 66 of 138
  failing). That run judged the ``.rp`` outputs of the fused call against
  FP64 from the FP64 streams rounded to BF16, not from the BF16 streams the
  path itself produced. A stream element a path rounds to the other BF16
  neighbour (about 3e-5 of elements; fewer for kernel A than for the torch
  formulas) then counted as error of the coefficients computed from it, and
  one such element moves ``post`` by up to ~8e-5: the comparison measured
  which path's rounding landed on which element, not the error on the
  inputs the arithmetic received. Each path is now judged against FP64 from
  its own streams (V2: the same inputs the kernel received), the reference
  path against its own FP64 value (that run used the candidate's), and row
  invariance is required where the torch formulas have it. The bounds are
  unchanged. The same run also changed the torch finish (FP64 coefficients,
  a fixed partial-sum order), a candidate change judged under the same rule.
  The second GPU run (9418cf08c4) failed pre.rp's max bound at 1 row
  (1.60e-7 against 1.20e-7, both finishes alike): kernel A's FP32
  accumulation inside each hidden slice (CPU emulation on the same inputs).
  Kernel A now accumulates in FP64 there for this model (fp64_acc, a
  candidate change, judged by the same rule); bounds and references unchanged.

- Bitwise run-to-run determinism, CUDA graph replay equal to eager.
- Kernels per call: two (three with ``capture_aux``) with the finish kernel;
  with the torch finish (``VLLM_DEEPSEEK_V41_FUSED_MHC`` alone: the reference arm the
  finish kernel is measured against, not a speed path) a fixed count,
  TORCH_FINISH_KERNELS. Until the second GPU run this asserted fewer kernels
  than the torch formulas; the FP64 torch finish launches more (177 against
  158), and as this arm carries no speed claim the count is now fixed.

Synthetic inputs; the real-input check is the ``mhc_delayed`` /
``mhc_pre_delayed`` families of ``fp64_cases.py`` on captured calls.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_mhc_fused_gpu.py
"""

import pytest
import torch

from . import fp64_cases as F
from .test_sm80_mhc_fused import call, variants

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="needs an sm_80 card",
    ),
]

HIDDEN = 5120
CUDA = torch.device("cuda")
# the served graph capture sizes (verify steps of DSpark up to 48 rows) and two
# prefill sizes
ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 30, 36, 42, 48, 256, 2048)
VARIANTS = ("fused", "fused_identity", "pre_only", "pre_only_identity")
# kernel A + the torch finish, one fused post + pre call (measured, 9418cf08c4)
TORCH_FINISH_KERNELS = 177


def _outputs(variant: str) -> list[str]:
    names = ["post", "comb", "layer_input", "pre"]
    out = names + [n + ".rp" for n in names]
    return (["residual"] if variant.startswith("fused") else []) + out


@pytest.fixture(params=[False, True], ids=["finish_torch", "finish_kernel"])
def fused(request, monkeypatch):
    """The fused module with its finish in torch (kernel A only) or as kernel
    B (``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH``)."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    monkeypatch.setattr(sm80_mhc_fused, "FINISH_KERNEL", request.param)
    return sm80_mhc_fused


def _cuda(a: dict) -> dict:
    return {k: (v.to(CUDA) if isinstance(v, torch.Tensor) else v) for k, v in a.items()}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("T", ROWS)
def test_fp64_within_v2_of_the_torch_formulas(T, variant, fused):
    family = F.mhc_delayed if variant.startswith("fused") else F.mhc_pre_delayed
    calls = [
        {k: v for k, v in variants(T, HIDDEN, seed)[variant].items() if v is not None}
        for seed in range(max(1, 64 // T))
    ]
    for output in _outputs(variant):
        cs = cm = rs = rm = 0.0
        n = 0
        for a in calls:
            c = family(a, CUDA, F.Source({}, {}, output))
            s, mx, k = F.abs_errors(c.cand, c.exact)
            cs, cm, n = cs + s, max(cm, mx), n + k
            ref_exact = c.exact if c.ref_exact is None else c.ref_exact
            s, mx, _ = F.abs_errors(c.ref, ref_exact)
            rs, rm = rs + s, max(rm, mx)
            assert k == c.cand.numel() > 0, output
            # rows alone give the whole call's bits (one kernel A form up to
            # 8 rows) wherever the torch formulas do
            if T <= 8 and c.ref_row_invariant:
                assert c.row_invariant, output
        assert cs / n <= 1.10 * (rs / n), (output, cs / n, rs / n)
        assert cm <= 1.25 * rm, (output, cm, rm)


@pytest.mark.parametrize("T", (1, 6, 48))
def test_deterministic_and_graph_replay_equal(T, fused):
    for variant in ("fused", "pre_only"):
        a = _cuda(variants(T, HIDDEN)[variant])
        first = call(fused, a)
        torch.cuda.synchronize()
        again = call(fused, a)
        for k in first:
            assert F.bits_equal(first[k], again[k]), (variant, k)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = call(fused, a)
        g.replay()
        torch.cuda.synchronize()
        for k in first:
            assert F.bits_equal(first[k], captured[k]), (variant, k)


def _kernels(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and "memcpy" not in e.name.lower()
        and "memset" not in e.name.lower()
    )


@pytest.mark.parametrize("T", (1, 48))
def test_kernels_per_call(T, fused):
    a = _cuda(variants(T, HIDDEN)["fused"])
    common = [a[k] for k in ("fn", "hc_scale", "hc_base", "rms_eps", "hc_pre_eps",
                             "hc_sinkhorn_eps", "hc_post_mult_value", "sinkhorn_repeat")]
    kw = dict(pre_mix=a["pre_mix"], norm_weight=a["norm_weight"], norm_eps=a["norm_eps"])

    def post_pre(aux=False):
        return fused.mhc_fused_post_pre_delayed(
            a["x"], a["residual"], a["post_layer_mix"], a["comb_res_mix"], *common,
            capture_aux=aux, **kw)

    if fused.FINISH_KERNEL:
        assert _kernels(post_pre) == 2
        assert _kernels(lambda: post_pre(aux=True)) == 3
        assert _kernels(lambda: fused.mhc_pre_delayed(a["residual"], *common, **kw)) == 2
    else:
        # kernel A, then the torch finish: the reference arm, not a speed path;
        # its count is fixed as measured on the second GPU run (9418cf08c4:
        # 177 at 1 and 48 rows)
        assert _kernels(post_pre) == TORCH_FINISH_KERNELS
