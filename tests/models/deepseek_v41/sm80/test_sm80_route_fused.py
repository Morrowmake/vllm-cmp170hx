# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 MoE routing (``ops/sm80_route.py``) on CPU.

- The kernel in the Triton interpreter (a subprocess with
  ``TRITON_INTERPRET=1``) against ``sm80_moe.route`` on the same FP32
  logits, for the target's router (384 experts, top 6) and the drafter's
  (128, top 3), with and without image tokens, renormalised or not: the
  same expert ids in the same order and the weights within 4 ulp (the
  interpreter has no libdevice, so its softplus is ``log(1 + exp(x))``, and
  torch on CPU sums a row in another order than its CUDA kernel).
- In the interpreter, with every logit above softplus' threshold (where
  softplus is the identity), the weights are bitwise the CUDA torch path's
  arithmetic emulated in NumPy FP32: correctly rounded sqrt, the bias add,
  the top-k, and the row sum in the order of torch's CUDA reduction kernel
  (:func:`torch_cuda_row_sum`, an independent model of
  ``ATen/native/cuda/Reduce.cuh`` for a short contiguous row), ``+ 1e-20``,
  the divide and the scale; and a sequential sum would not be.
- The kernel compiles for sm_80 for both routers, with and without the
  verification row mask (``row_live``; its behaviour is
  ``test_row_mask_moe.py``) (no GPU needed).
- The switch is off by default; ``supports`` refuses what the kernel does
  not serve.

The GPU half (bytes equal to the torch path on CUDA, the reduction order
against ``torch.sum`` on CUDA, graph replay) is
``test_sm80_route_fused_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_route_fused.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SENTINEL, IMAGE_IDS = 1000, 5
ROUTERS = ((384, 6), (128, 3))


def torch_cuda_row_sum(row: np.ndarray) -> np.float32:
    """``torch.sum`` of one contiguous FP32 row of k (1..32) values as the
    CUDA reduction kernel orders it: block width = largest power of two not
    above k; lane x loads x, x + width, ... into its four accumulators
    (identity 0), combines them in order, then the lanes fold by warp
    shuffles of decreasing offset."""
    k = len(row)
    width = 1 << (k.bit_length() - 1)
    lanes = []
    for x in range(width):
        acc = [np.float32(0.0)] * 4
        idx, slot = x, 0
        while idx < k and slot < 4:
            acc[slot] = np.float32(acc[slot] + row[idx])
            idx += width
            slot += 1
        assert idx >= k
        value = acc[0]
        for slot in range(1, 4):
            value = np.float32(value + acc[slot])
        lanes.append(value)
    offset = width // 2
    while offset:
        for x in range(offset):
            lanes[x] = np.float32(lanes[x] + lanes[x + offset])
        offset //= 2
    return lanes[0]


def emulated_cuda_route(logits, bias, top_k, scale):
    """The CUDA torch path's FP32 arithmetic for logits above 20 (softplus is
    the identity), no image tokens, renormalised."""
    s = np.sqrt(logits.astype(np.float32))
    biased = (s + bias.astype(np.float32)).astype(np.float32)
    ids = np.argsort(-biased, axis=1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(s, ids, axis=1)
    out = np.empty_like(weights)
    for t in range(weights.shape[0]):
        total = np.float32(torch_cuda_row_sum(weights[t]) + np.float32(1e-20))
        out[t] = (weights[t] / total).astype(np.float32) * np.float32(scale)
    return out.astype(np.float32), ids.astype(np.int32)


def make_case(tokens, experts, seed, *, high=False):
    g = torch.Generator().manual_seed(seed)
    if high:
        logits = 21.0 + 30.0 * torch.rand(tokens, experts, generator=g)
    else:
        # The gate's range, with tails past both softplus branches.
        logits = torch.randn(tokens, experts, generator=g) * 3.0
        logits[:, :4] = torch.tensor([25.0, -30.0, 19.99, 20.01])
    bias = torch.randn(experts, generator=g) * 0.1
    bias_vl = torch.randn(experts, generator=g) * 0.1
    input_ids = torch.randint(0, 2000, (tokens,), generator=g, dtype=torch.int32)
    input_ids[::3] = SENTINEL + (torch.arange(0, tokens, 3) % IMAGE_IDS).int()
    return logits.float().contiguous(), bias, bias_vl, input_ids


def interpreted_results() -> dict:
    """Run in the Triton interpreter: per case, the fused outputs next to the
    torch path's (and the emulation's for the high-logit cases)."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.models.deepseek_v41.ops.sm80_route import route_fused
    from vllm_sm80.models.deepseek_v41.sm80_moe import image_token_mask, route

    out = {}
    for experts, top_k in ROUTERS:
        for tokens in (1, 6, 33):
            for images in (False, True):
                for renorm in (True, False):
                    logits, bias, bias_vl, ids_in = make_case(tokens, experts, tokens + experts)
                    lo = SENTINEL if images else 0
                    got_w, got_i = route_fused(
                        logits, bias, top_k, bias_vl=bias_vl, input_ids=ids_in,
                        sentinel_lo=lo, sentinel_hi=lo + IMAGE_IDS,
                        renormalize=renorm, routed_scaling_factor=1.5,
                    )
                    want_w, want_i = route(
                        logits, bias, top_k, bias_vl=bias_vl,
                        image_mask=image_token_mask(ids_in.long(), lo),
                        renormalize=renorm, routed_scaling_factor=1.5,
                    )
                    key = f"{experts}x{top_k}/T{tokens}/img{int(images)}/renorm{int(renorm)}"
                    out[key] = dict(
                        ids_equal=bool(torch.equal(got_i, want_i.int())),
                        dtypes=[str(got_w.dtype), str(got_i.dtype)],
                        shapes=[list(got_w.shape), list(got_i.shape)],
                        max_ulp=float(((got_w - want_w).abs() / torch.finfo(torch.float32).eps
                                       / want_w.abs().clamp_min(1e-30)).max()),
                    )
        for tokens in (1, 6, 33):
            logits, bias, _, _ = make_case(tokens, experts, 7 + tokens, high=True)
            got_w, got_i = route_fused(logits, bias, top_k, routed_scaling_factor=1.5)
            want_w, want_i = emulated_cuda_route(logits.numpy(), bias.numpy(), top_k, 1.5)
            seq = []
            for t in range(tokens):
                row = got_w[t].numpy()
                s = np.float32(0.0)
                for v in np.take_along_axis(np.sqrt(logits.numpy()[t]), want_i[t].astype(np.int64), 0):
                    s = np.float32(s + v)
                seq.append(s)
            out[f"emulated/{experts}x{top_k}/T{tokens}"] = dict(
                weights_bitwise=bool(np.array_equal(got_w.numpy().view(np.int32),
                                                    want_w.view(np.int32))),
                ids_equal=bool(np.array_equal(got_i.numpy(), want_i)),
            )
    # NaN gate logits: one NaN expert, and a whole NaN row. torch.topk ranks
    # NaN above every number, so the NaN expert comes first and every id stays
    # below the expert count; the weights of such a row are NaN.
    for experts, top_k in ROUTERS:
        logits, bias, _, _ = make_case(3, experts, 91)
        logits[0, 7] = float("nan")
        logits[1, :] = float("nan")
        got_w, got_i = route_fused(logits, bias, top_k, routed_scaling_factor=1.5)
        out[f"nan/{experts}x{top_k}"] = dict(
            in_range=bool(((got_i >= 0) & (got_i < experts)).all()),
            nan_first=int(got_i[0, 0]) == 7,
            nan_rows=bool(torch.isnan(got_w[:2]).all()),
            all_nan_distinct=len(set(got_i[1].tolist())) == top_k,
            finite_row=bool(torch.isfinite(got_w[2]).all()),
        )
    # The order matters: some row of k values sums differently sequentially.
    rng = np.random.default_rng(0)
    differs = {}
    for k in (3, 6):
        rows = (rng.standard_normal((4096, k)) ** 2 * 10.0 ** rng.integers(-3, 3, (4096, k))).astype(np.float32)
        differs[k] = int(sum(
            torch_cuda_row_sum(r) != np.float32(sum(np.float32(v) for v in r)) for r in rows))
    out["order_sensitive"] = differs
    return out


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_route_fused import "
            "interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_interpreted_kernel_routes_like_the_torch_path(interpreted):
    cases = {k: v for k, v in interpreted.items() if "/T" in k and not k.startswith("emulated")}
    assert len(cases) == len(ROUTERS) * 3 * 2 * 2
    for key, c in cases.items():
        k = int(key.split("/")[0].split("x")[1])
        t = int(key.split("/T")[1].split("/")[0])
        assert c["ids_equal"], key
        assert c["dtypes"] == ["torch.float32", "torch.int32"], key
        assert c["shapes"] == [[t, k], [t, k]], key
        assert c["max_ulp"] <= 4.0, (key, c["max_ulp"])


def test_interpreted_kernel_is_the_cuda_arithmetic_bitwise(interpreted):
    cases = {k: v for k, v in interpreted.items() if k.startswith("emulated")}
    assert len(cases) == len(ROUTERS) * 3
    for key, c in cases.items():
        assert c["ids_equal"], key
        assert c["weights_bitwise"], key
    # A sequential sum would differ from the CUDA order on some rows.
    assert all(n > 0 for n in interpreted["order_sensitive"].values()), interpreted


def test_nan_logits_keep_expert_ids_in_range(interpreted):
    cases = {k: v for k, v in interpreted.items() if k.startswith("nan/")}
    assert len(cases) == len(ROUTERS)
    for key, c in cases.items():
        assert all(c.values()), (key, c)


def test_row_sum_model_is_the_installed_torch_reduction():
    """The row-sum order (:func:`torch_cuda_row_sum`) was read from torch's
    ``ATen/native/cuda/Reduce.cuh``; pin the two statements it rests on in the
    installed torch's header: accumulators combined in order, and the warp fold
    with decreasing offsets. A torch upgrade that changes either fails here
    (the GPU half checks the model against ``torch.sum`` on CUDA)."""
    import os

    header = os.path.join(os.path.dirname(torch.__file__),
                          "include/ATen/native/cuda/Reduce.cuh")
    src = open(header).read()
    assert "value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);" in src
    assert "for (int offset = dim_x >> 1; offset > 0; offset >>= 1) {" in src
    assert "#if defined(USE_ROCM)\n    for (int offset = 1; offset < dim_x; offset <<= 1) {" in src
    assert "dim0 >= 128 && iter.num_reduce_dims() == 1" in src


def test_row_sum_model_matches_the_documented_order():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = (rng.standard_normal(6) ** 2 * 10.0 ** rng.integers(-4, 4, 6)).astype(np.float32)
        f = np.float32
        assert torch_cuda_row_sum(w) == f(f(f(w[0] + w[4]) + w[2]) + f(f(w[1] + w[5]) + w[3]))
        v = w[:3]
        assert torch_cuda_row_sum(v) == f(f(v[0] + v[2]) + v[1])


def test_kernel_compiles_for_sm80():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from vllm_sm80.models.deepseek_v41.ops import sm80_route

    types = {"logits_ptr": "*fp32", "logits_stride": "i64", "bias_ptr": "*fp32",
             "bias_vl_ptr": "*fp32", "token_ptr": "*i32", "weights_ptr": "*fp32",
             "ids_ptr": "*i32", "row_live_ptr": "*i8", "num_experts": "i32",
             "sentinel_lo": "i32", "sentinel_hi": "i32", "scale": "fp32", "eps": "fp32"}
    kern = sm80_route._route_kernel
    for experts, top_k in ROUTERS:
        w = 1 << (top_k.bit_length() - 1)
        for has_vl, row_mask in ((True, False), (False, False), (True, True)):
            consts = dict(K=top_k, W=w, LOG2W=w.bit_length() - 1,
                          BLOCK_E=triton.next_power_of_2(experts), HAS_VL=has_vl,
                          RENORM=True, ACCURATE=True, ROW_MASK=row_mask)
            sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
            ptx = {}
            for ftz in (False, True):
                c = triton.compile(ASTSource(kern, sig, constexprs=consts),
                                   target=GPUTarget("cuda", 80, 32),
                                   options={"num_warps": 4, "enable_reflect_ftz": ftz,
                                            "enable_fp_fusion": False})
                assert c.asm["cubin"]
                ptx[ftz] = c.asm["ptx"]
            # IEEE divide and square root; libdevice's exp and log1p in their
            # non-FTZ form (as nvcc builds torch): the only FTZ instruction left
            # is the ex2.approx.ftz inside libdevice's expf, which both forms use.
            assert "div.rn.f32" in ptx[False] and "sqrt.rn.f32" in ptx[False]
            assert "div.full.f32" not in ptx[False] and "sqrt.approx" not in ptx[False]
            assert set(re.findall(r"[a-z0-9.]+\.ftz\.f32", ptx[False])) == {"ex2.approx.ftz.f32"}
            assert "fma.rn.ftz.f32" in ptx[True]
            # Without the mask the kernel never reads row_live (one byte load).
            assert ("ld.global.b8" in ptx[False] or "ld.global.u8" in ptx[False]
                    or "ld.global.s8" in ptx[False]) is row_mask


def test_switch_default_and_supports(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80.models.deepseek_v41.ops import sm80_route
    from vllm_sm80.models.deepseek_v41.sm80_moe import Sm80DeepseekV41Router

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_FUSED_ROUTE", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_FUSED_ROUTE is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_ROUTE", "1")
    assert envs.VLLM_DEEPSEEK_V41_FUSED_ROUTE is True
    assert Sm80DeepseekV41Router.fused_route is False
    for experts, top_k in ROUTERS:
        assert sm80_route.supports(experts, top_k, 1.0) is None
    assert "gate_temp" in sm80_route.supports(384, 6, 0.5)
    assert "experts" in sm80_route.supports(2048, 6, 1.0)
    assert "top_k" in sm80_route.supports(384, 17, 1.0)
    assert "top_k" in sm80_route.supports(4, 6, 1.0)
