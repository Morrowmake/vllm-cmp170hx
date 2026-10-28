# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 delayed mHC (``ops/sm80_mhc_fused.py``) on CPU.

- The Triton finish kernel runs in the Triton interpreter (a subprocess with
  ``TRITON_INTERPRET=1``), fed by a torch stand-in for kernel A (the Gluon
  kernel needs a GPU: the post step, then FP32 partials over 32-wide hidden
  slices of the BF16 streams). Every output, for the fused and the pre-only
  call, with and without a carried pre-mix and norm, against FP64
  (``fp64_cases.mhc_delayed_fp64``) within the V2 bounds of the torch
  formulas (``ops/sm80_mhc.py``) on the same inputs, at a hidden size that
  is not a multiple of the collapse chunk.
- The torch finish (kernel A only, ``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH`` off) on
  the same stand-in, in process, within the same bounds.
- The FP64 reference against the torch formulas at FP32 accuracy.
- ``mhc_ops``: the switch picks the module the layers call, refuses shapes
  the kernels do not serve; both switches are off by default.
- Every kernel the fused path launches compiles for sm_80 at the model's
  hidden size (no GPU needed).

The GPU half (kernel A, the whole path, CUDA graphs, determinism, kernel
count) is ``test_sm80_mhc_fused_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_mhc_fused.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_mhc

from .fp64_cases import abs_errors, mhc_delayed_fp64

ROOT = Path(__file__).resolve().parents[4]
BF16 = torch.bfloat16
HC = 4
EPS, RMS_EPS, NORM_EPS, MULT, ITERS = 1e-6, 1e-20, 1e-20, 2.0, 20


def make_inputs(T: int, hidden: int, seed: int = 0) -> dict:
    """One call's arguments with the served magnitudes' shape: per-token
    stream scales over four decades, post in (0, 2), a doubly stochastic
    comb, coefficients near the checkpoint's range."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.logspace(-1, 2, T)[:, None, None]
    comb = torch.rand(T, HC, HC, generator=g) + 0.1
    for _ in range(10):
        comb = comb / comb.sum(-1, keepdim=True)
        comb = comb / comb.sum(-2, keepdim=True)
    return dict(
        x=(torch.randn(T, hidden, generator=g) * scale[:, 0]).to(BF16),
        residual=(torch.randn(T, HC, hidden, generator=g) * scale).to(BF16),
        post_layer_mix=2 * torch.sigmoid(torch.randn(T, HC, 1, generator=g)),
        comb_res_mix=comb,
        fn=torch.randn((2 + HC) * HC, HC * hidden, generator=g) * 0.02,
        hc_scale=torch.tensor([0.8, 1.2, 1.6]),
        hc_base=torch.randn((2 + HC) * HC, generator=g) * 0.5,
        rms_eps=RMS_EPS,
        hc_pre_eps=EPS,
        hc_sinkhorn_eps=EPS,
        hc_post_mult_value=MULT,
        sinkhorn_repeat=ITERS,
        pre_mix=torch.sigmoid(torch.randn(T, HC, generator=g)) + EPS,
        norm_weight=(1 + 0.1 * torch.randn(hidden, generator=g)).to(BF16),
        norm_eps=NORM_EPS,
    )


def call(mod, a: dict) -> dict:
    """One path's outputs by name (``x`` None: the pre-only call)."""
    common = [a[k] for k in ("fn", "hc_scale", "hc_base", "rms_eps", "hc_pre_eps",
                             "hc_sinkhorn_eps", "hc_post_mult_value", "sinkhorn_repeat")]
    kw = dict(pre_mix=a.get("pre_mix"), norm_weight=a.get("norm_weight"),
              norm_eps=a.get("norm_eps", 1e-6))
    if a.get("x") is None:
        post, comb, li, pre = mod.mhc_pre_delayed(a["residual"], *common, **kw)
        return dict(post=post, comb=comb, layer_input=li, pre=pre)
    streams, post, comb, li, pre, _ = mod.mhc_fused_post_pre_delayed(
        a["x"], a["residual"], a["post_layer_mix"], a["comb_res_mix"], *common, **kw)
    return dict(residual=streams, post=post, comb=comb, layer_input=li, pre=pre)


def variants(T: int, hidden: int, seed: int = 0) -> dict:
    a = make_inputs(T, hidden, seed)
    return {
        "fused": a,
        "fused_identity": dict(a, pre_mix=None),
        "fused_no_norm": dict(a, norm_weight=None),
        "pre_only": dict(a, x=None, post_layer_mix=None, comb_res_mix=None),
        "pre_only_identity": dict(a, x=None, post_layer_mix=None, comb_res_mix=None,
                                  pre_mix=None),
    }


def _stand_in_kernel_a(x, res, post, comb, fn, rc, mix, sqr, *, has_post=True,
                       round_mix=False, fp64_acc=False, hs_reuse=False):
    """kernel A's contract in torch: the streams (stored BF16), FP32 partial
    projections and squared sums over 32-wide hidden slices of the streams
    as stored (``round_mix``; the only mode this model uses). ``hs_reuse``
    changes only kernel A's shared-memory staging, not its results."""
    assert round_mix and fp64_acc  # this model's options
    T, hc, hidden = res.shape
    streams = sm80_mhc.mhc_post(x, res, post, comb) if has_post else res
    rc.copy_(streams)
    sf = streams.float().view(T, hc, hidden // 32, 32)
    fv = fn.view(fn.shape[0], hc, hidden // 32, 32)
    mix.copy_(torch.einsum("tjsk,njsk->stn", sf, fv))
    sqr.copy_(sf.square().sum(dim=(1, 3)).T)


def exacts(a: dict, got: dict, want: dict) -> dict:
    """Per FP64 form (False: no intermediate rounding; True: at each path's
    own rounding points) the FP64 outputs each path is judged against:
    {rp: (for the fused path, for the torch formulas)}."""
    plain = mhc_delayed_fp64(a)
    if a.get("x") is None:
        own = mhc_delayed_fp64(a, rounding=True)
        return {False: (plain, plain), True: (own, own)}
    return {False: (plain, plain),
            True: (mhc_delayed_fp64(a, True, got["residual"]),
                   mhc_delayed_fp64(a, True, want["residual"]))}


def interpreted_errors(T: int = 5, hidden: int = 1056, calls: int = 6) -> dict:
    """Run in the Triton interpreter: per variant and output, the FP64 errors
    (sum, max, count) of the fused path and of the torch formulas over
    ``calls`` independent calls (pooled, as a captured case pools its calls),
    without and with the reference rounding points."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    sm80_mhc_fused.post_prenorm = _stand_in_kernel_a
    sm80_mhc_fused.FINISH_KERNEL = True
    out = {}
    for seed in range(calls):
        for name, a in variants(T, hidden, seed).items():
            got, want = call(sm80_mhc_fused, a), call(sm80_mhc, a)
            exact = exacts(a, got, want)
            for o in got:
                assert got[o].shape == want[o].shape, (name, o)
                assert got[o].dtype == want[o].dtype, (name, o)
                for rp in (False, True):
                    if o == "residual" and rp:
                        continue
                    key = f"{name}/{o}{'.rp' if rp else ''}"
                    pooled = out.setdefault(key, dict(cand=[0.0, 0.0, 0], ref=[0.0, 0.0, 0]))
                    for i, (path, value) in enumerate((("cand", got[o]), ("ref", want[o]))):
                        s, mx, n = abs_errors(value, exact[rp][i][o])
                        p = pooled[path]
                        p[0], p[1], p[2] = p[0] + s, max(p[1], mx), p[2] + n
    return out


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_mhc_fused import "
            "interpreted_errors as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_finish_kernel_within_v2_of_the_torch_formulas(interpreted):
    # fused calls: residual + 4 outputs x 2 FP64 forms; pre-only: 4 x 2
    assert len(interpreted) == 3 * 9 + 2 * 8
    for key, e in interpreted.items():
        (cs, cm, n), (rs, rm, _) = e["cand"], e["ref"]
        assert n > 0, key
        cand_mean, ref_mean = cs / n, rs / n
        assert cand_mean <= 1.10 * ref_mean, (key, cand_mean, ref_mean)
        assert cm <= 1.25 * rm, (key, cm, rm)


def test_torch_finish_within_v2_of_the_torch_formulas(monkeypatch):
    """Kernel A (its stand-in) with the torch finish: the reference formulas
    on the reduced partials instead of one SGEMM."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    monkeypatch.setattr(sm80_mhc_fused, "post_prenorm", _stand_in_kernel_a)
    monkeypatch.setattr(sm80_mhc_fused, "FINISH_KERNEL", False)
    pooled = {}
    for seed in range(6):
        for name, a in variants(5, 1056, seed).items():
            got, want = call(sm80_mhc_fused, a), call(sm80_mhc, a)
            exact = exacts(a, got, want)
            for o in got:
                assert got[o].shape == want[o].shape and got[o].dtype == want[o].dtype
                for rp in (False, True):
                    if o == "residual" and rp:
                        continue
                    p = pooled.setdefault((name, o, rp), [0.0, 0.0, 0.0, 0.0, 0])
                    cs, cm, n = abs_errors(got[o], exact[rp][0][o])
                    rs, rm, _ = abs_errors(want[o], exact[rp][1][o])
                    p[0], p[1], p[2], p[3] = p[0] + cs, max(p[1], cm), p[2] + rs, max(p[3], rm)
                    p[4] += n
    assert len(pooled) == 3 * 9 + 2 * 8
    for key, (cs, cm, rs, rm, n) in pooled.items():
        assert n > 0, key
        assert cs / n <= 1.10 * (rs / n), (key, cs / n, rs / n)
        assert cm <= 1.25 * rm, (key, cm, rm)


def test_fp64_reference_is_the_torch_formulas_at_fp32_accuracy():
    for name, a in variants(3, 256).items():
        want = call(sm80_mhc, a)
        exact = mhc_delayed_fp64(a, rounding=True)
        for o, v in want.items():
            e = exact[o]
            if v.dtype == BF16:
                # rounded outputs: within one BF16 step of the FP64 value
                tol = e.abs() * 2.0**-8 + 1e-30
            else:
                tol = e.abs() * 1e-5 + 1e-7
            assert bool(((v.double() - e).abs() <= tol).all()), (name, o)


def test_mhc_ops_switch(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80.models.deepseek_v41 import sm80_model
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_FUSED_MHC", raising=False)
    monkeypatch.delenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_FUSED_MHC is False and envs.VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH is False
    cfg = SimpleNamespace(hc_mult=4, hidden_size=5120)
    assert sm80_model.mhc_ops(cfg) is sm80_mhc
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_MHC", "1")
    assert sm80_model.mhc_ops(cfg) is sm80_mhc_fused
    for bad in (SimpleNamespace(hc_mult=2, hidden_size=5120),
                SimpleNamespace(hc_mult=4, hidden_size=5100)):
        with pytest.raises(ValueError, match="VLLM_DEEPSEEK_V41_FUSED_MHC"):
            sm80_model.mhc_ops(bad)
    # both modules carry every op the layers call
    for mod in (sm80_mhc, sm80_mhc_fused):
        for op in ("mhc_pre_delayed", "mhc_fused_post_pre_delayed", "mhc_post"):
            assert callable(getattr(mod, op)), (mod, op)
    assert sm80_mhc_fused.mhc_post is sm80_mhc.mhc_post


def test_fused_path_refuses_what_the_kernels_do_not_serve():
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    a = make_inputs(2, 256)
    with pytest.raises(ValueError, match="one hc_eps"):
        call(sm80_mhc_fused, dict(a, hc_sinkhorn_eps=2e-6))
    with pytest.raises(ValueError, match="multiple of 32"):
        b = make_inputs(2, 80)
        call(sm80_mhc_fused, dict(b, residual=b["residual"][..., :72], x=b["x"][:, :72],
                                  fn=b["fn"][:, : 4 * 72], norm_weight=b["norm_weight"][:72]))


def test_kernels_compile_for_sm80():
    """Kernel A in both forms with this model's options (post + pre and
    pre-only, BF16 streams into the projection) and the finish kernel in
    every variant the wrappers launch, at hidden 5120."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.experimental.gluon._runtime import GluonASTSource

    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused
    from vllm_sm80.kernels.mhc import decode

    hidden, target = 5120, GPUTarget("cuda", 80, 32)
    a_types = {"x_ptr": "*bf16", "res_ptr": "*bf16", "post_ptr": "*fp32",
               "comb_ptr": "*fp32", "fn_ptr": "*fp32", "rc_ptr": "*bf16",
               "mix_ptr": "*fp32", "sqr_ptr": "*fp32", "M": "i32"}
    f_types = {p: "*fp32" for p in ("mix_ptr", "sqr_ptr", "scale_ptr", "base_ptr",
                                    "pre_in_ptr", "post_ptr", "comb_ptr", "pre_ptr")}
    f_types.update(rc_ptr="*bf16", nw_ptr="*bf16", li_ptr="*bf16", M="i32",
                   rms_eps="fp32", hc_eps="fp32", post_mult="fp32", norm_eps="fp32")

    def build(source, kern, types, consts):
        sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
        c = triton.compile(source(kern, sig, constexprs=consts), target=target,
                           options={"num_warps": 4})
        assert c.asm["cubin"]

    base = dict(HIDDEN=hidden, NOUT=24, HB=decode.HB_GLU, NP2=32, ROUND_MIX=True,
                FP64_ACC=True)
    for has_post in (True, False):
        build(GluonASTSource, decode._glu_post_prenorm_tb_kernel, a_types,
              dict(base, BT=decode.BT_T, HAS_POST=has_post))
        build(GluonASTSource, decode._glu_post_prenorm_kernel, a_types,
              dict(base, BM=decode.BM_A, NW=decode.WARPS_GLU, HAS_POST=has_post))
    ks = hidden // decode.HB_GLU
    for has_pre, has_norm in ((True, True), (False, True), (True, False)):
        # PART given: a compile from source takes no default (the launch's
        # does); the parts launched apart are test_sm80_mhc_side.py's
        build(ASTSource, fused._delayed_finish_kernel, f_types,
              dict(HIDDEN=hidden, KS=ks, KSP2=triton.next_power_of_2(ks), SINK=20,
                   HAS_PRE_IN=has_pre, HAS_NORM=has_norm, BH=fused._BH, ACCURATE=True, PART=0))
