# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 delayed mHC finish v2 (``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2``)
on one sm_80 card, at the model's shapes (hidden 5120, hc 4, 20 Sinkhorn
iterations, eps 1e-6, RMS eps 1e-20).

- Kernel A's staging reuse gives the bits of kernel A without it (every
  output, both forms, this model's options and GLM-5.3-Flash's), from 17 rows
  (the reuse) through the prefill chunk.
- FP64 rule (V2) against the served finish (the switch off, the finish kernel
  on) on the same inputs: every output of the fused and the pre-only call,
  with and without a carried pre-mix, against FP64 without intermediate
  rounding and at each path's own BF16 rounding points (the ``.rp``
  outputs: a stream element a path rounds to the other BF16 neighbour is not
  counted as error of the coefficients computed from it), pooled per row
  count over calls up to 64 tokens: mean error at most 1.10x and max error at
  most 1.25x the served finish's; where the served finish is exact the
  finish v2 must be too.
- The calls launch the kernels the switch names (profiler kernel names, as a
  step profile files them): off ``_delayed_finish_kernel``, on
  ``_delayed_finish_v2_kernel``.
- On: bitwise run-to-run determinism; CUDA graph replay equal to eager, no
  allocation growth over replays; after the warmup a capture compiles
  nothing.

Synthetic inputs; the real-input check is the ``mhc_delayed`` /
``mhc_pre_delayed`` families of ``fp64_cases.py`` run with the switch on.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_mhc_finish_v2_gpu.py
"""

from types import SimpleNamespace

import pytest
import torch

from . import fp64_cases as F
from .test_sm80_mhc_fused import call, exacts, variants

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 card",
)

HIDDEN = 5120
CUDA = torch.device("cuda")
# every served decode / verify / draft row count class and the prefill sizes
ROWS = (1, 2, 5, 6, 8, 10, 12, 16, 17, 18, 20, 24, 28, 32, 40, 48, 267, 2048)
VARIANTS = ("fused", "fused_identity", "pre_only", "pre_only_identity")


@pytest.fixture
def fused(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    monkeypatch.setattr(sm80_mhc_fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(sm80_mhc_fused, "FINISH_V2", False)
    return sm80_mhc_fused


def _cuda(a: dict) -> dict:
    return {k: (v.to(CUDA) if isinstance(v, torch.Tensor) else v) for k, v in a.items()}


def _cpu(d: dict) -> dict:
    return {k: v.cpu() for k, v in d.items()}


@pytest.mark.parametrize("options", ["dsv41", "glm53"])
@pytest.mark.parametrize("has_post", [True, False], ids=["post", "pre_only"])
@pytest.mark.parametrize("M", [17, 24, 32, 48, 267, 2048])
def test_kernel_a_staging_reuse_gives_the_same_bits(M, has_post, options):
    from vllm_sm80.kernels.mhc.decode import HB_GLU, post_prenorm

    hidden, opts = ((HIDDEN, dict(round_mix=True, fp64_acc=True)) if options == "dsv41"
                    else (4096, {}))
    if options == "glm53" and not has_post:
        pytest.skip("GLM-5.3-Flash calls kernel A with the post step only")
    g = torch.Generator(device=CUDA).manual_seed(M)
    x = torch.randn(M, hidden, device=CUDA, generator=g).to(torch.bfloat16)
    res = (torch.randn(M, 4, hidden, device=CUDA, generator=g) * 4).to(torch.bfloat16)
    post = torch.rand(M, 4, device=CUDA, generator=g) * 2
    comb = torch.rand(M, 4, 4, device=CUDA, generator=g)
    fn = torch.randn(24, 4 * hidden, device=CUDA, generator=g) * 0.02
    out = []
    for reuse in (False, True):
        ksb = hidden // HB_GLU
        rc = torch.empty_like(res)
        mix = torch.empty(ksb, M, 24, device=CUDA)
        sqr = torch.empty(ksb, M, device=CUDA)
        post_prenorm(x, res, post, comb, fn, rc, mix, sqr, has_post=has_post,
                     hs_reuse=reuse, **opts)
        out.append((rc, mix, sqr))
    for off, on in zip(*out):
        assert F.bits_equal(off, on)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("T", ROWS)
def test_fp64_within_v2_of_the_served_finish(T, variant, fused):
    pools = {}
    for seed in range(max(1, 64 // T)):
        a = {k: v for k, v in variants(T, HIDDEN, seed)[variant].items()}
        fused.FINISH_V2 = False
        inc = _cpu(call(fused, _cuda(a)))
        fused.FINISH_V2 = True
        got = _cpu(call(fused, _cuda(a)))
        exact = exacts(a, got, inc)
        for o in got:
            assert got[o].shape == inc[o].shape and got[o].dtype == inc[o].dtype, o
            for rp in (False, True):
                if o == "residual" and rp:
                    continue
                p = pools.setdefault(o + (".rp" if rp else ""), [0.0, 0.0, 0.0, 0.0, 0])
                cs, cm, n = F.abs_errors(got[o], exact[rp][0][o])
                rs, rm, _ = F.abs_errors(inc[o], exact[rp][1][o])
                p[0], p[1], p[2], p[3], p[4] = p[0] + cs, max(p[1], cm), p[2] + rs, max(p[3], rm), p[4] + n
    for name, (cs, cm, rs, rm, n) in pools.items():
        assert n > 0, name
        if rm == 0:
            assert cm == 0, (name, cm)
        else:
            assert cs / n <= 1.10 * (rs / n), (name, cs / n, rs / n)
            assert cm <= 1.25 * rm, (name, cm, rm)


def _kernel_names(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and "memcpy" not in e.name.lower() and "memset" not in e.name.lower()]


@pytest.mark.parametrize("T,kernel_a", [(6, "_glu_post_prenorm_tb_kernel"),
                                        (32, "_glu_post_prenorm_kernel")])
def test_calls_launch_the_kernels_the_switch_names(T, kernel_a, fused):
    a = _cuda(variants(T, HIDDEN)["fused"])
    for on, finish in ((False, "_delayed_finish_kernel"), (True, "_delayed_finish_v2_kernel")):
        fused.FINISH_V2 = on
        assert _kernel_names(lambda: call(fused, a)) == [kernel_a, finish]
        pre = dict(a, x=None, post_layer_mix=None, comb_res_mix=None)
        assert _kernel_names(lambda: call(fused, pre)) == [kernel_a, finish]


def _compiled(kernels) -> int:
    return sum(len(cache[0]) for k in kernels for cache in k.device_caches.values())


@pytest.mark.parametrize("T", (1, 5, 6, 17, 32, 40))
def test_deterministic_graph_replay_equal_and_warmed(T, fused):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
    from vllm_sm80.kernels.mhc import decode

    fused.FINISH_V2 = True
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=[T]))
    finish_v2.FINISH_V2_WARMUP.warmup(vllm_config=cfg)
    kernels = (finish_v2._delayed_finish_v2_kernel, decode._glu_post_prenorm_kernel,
               decode._glu_post_prenorm_tb_kernel)
    compiled = _compiled(kernels)
    for variant in ("fused", "pre_only", "pre_only_identity"):
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
        before = torch.cuda.memory_allocated()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        for k in first:
            assert F.bits_equal(first[k], captured[k]), (variant, k)
    assert _compiled(kernels) == compiled
