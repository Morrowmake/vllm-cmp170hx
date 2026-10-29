# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 DeepSeek-V4.1 MoE gate GEMV (``ops/sm80_gate_gemv.py``) on CPU.

- The kernel in the Triton interpreter (a subprocess with
  ``TRITON_INTERPRET=1``) for the target's gate (384 x 5120) and the
  drafter's (128 x 5120), at the edges of every token tile and token-block
  count (:data:`ROWS`), on BF16
  activations with outlier channels, zeros and subnormals and an FP32 weight
  holding FP16 values (as loaded), plus a weight of full FP32 values: every
  logit is the FP32 rounding of the FP64 product ``x.double() @ w.double().T``
  except where that value lies within the FP64 summation error of an FP32
  rounding midpoint, and within half an FP32 ulp plus that error of it. The
  error of torch's FP32 ``x.float() @ w.T`` is reported beside it.
- A BF16 weight (the gate weight VLLM_DEEPSEEK_V41_ROUTER_MMA holds once, in
  BF16) gives the bytes of the same values held in FP32, both gates.
- A row's logits do not depend on the other rows of the call; a strided
  input gives the same bytes; the counters are zero after every call and the
  scratch is allocated once.
- A second call issues no host read and allocates only its output.
- The kernel compiles for sm_80 (no GPU needed), for an FP32 and a BF16
  weight: FP64 MMA, no redundant MMA across warps, one FP64 -> FP32
  rounding, acquire-release arrival, no register spills.
- The switch is off by default; adoption makes the gate a
  ``Sm80DeepseekV41Gate`` that takes the kernel only with the switch on and
  for calls ``serves`` accepts; the dev A/B switch toggles every gate.

The GPU half (CUDA bytes against FP64, route ids against the FP32 GEMM on
random and near-tie logits, graph replay) is ``test_sm80_gate_gemv_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_gate_gemv.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
HIDDEN = 5120
GATES = (384, 128)
# Both edges of every token tile (8, 16, 32) and token-block count (1, 2),
# a 6-row decode call and the 48-row verify call.
ROWS = (1, 6, 8, 9, 16, 17, 32, 33, 48, 64)
# Row counts of the BF16-weight case: every token tile and token-block count.
BF16_ROWS = (1, 6, 8, 9, 17, 33, 48, 64)


def activations(rows: int, seed: int) -> torch.Tensor:
    """BF16 gate inputs: normalised rows with channel scales, a few outlier
    channels, and in row 2 exact zeros and BF16 subnormals."""
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 + 0.5 * torch.randn(HIDDEN, generator=g).abs()
    scale[torch.randint(0, HIDDEN, (16,), generator=g)] *= 30.0
    x = torch.randn(rows, HIDDEN, generator=g) * scale
    if rows > 2:
        x[2, :64] = 0.0
        x[2, 64:96] = torch.randn(32, generator=g) * 2.0**-130
    return x.to(torch.bfloat16)


def gate_weight(experts: int, seed: int, *, fp16_values: bool = True) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(experts, HIDDEN, generator=g) * 0.01
    w *= 1.0 + torch.rand(experts, 1, generator=g)
    return w.half().float() if fp16_values else w


def judge(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> dict:
    """``out`` against the FP64 product. ``bound``: the FP64 summation error
    of the kernel plus that of the reference, K * 2**-53 * sum |x_k w_k|
    each."""
    ref = x.double() @ w.double().T
    bound = 2 * HIDDEN * 2.0**-53 * (x.double().abs() @ w.double().abs().T)
    r32 = ref.float()
    side = torch.where(ref >= r32.double(), torch.inf, -torch.inf).float()
    mid = (r32.double() + torch.nextafter(r32, side).double()) / 2
    near = (ref - mid).abs() <= bound
    mismatch = out.view(torch.int32) != r32.view(torch.int32)
    half_ulp = (torch.nextafter(r32.abs(), torch.full_like(r32, torch.inf)).double()
                - r32.abs().double()) / 2
    err = (out.double() - ref).abs()
    torch_err = ((x.float() @ w.T).double() - ref).abs()
    big = ref.abs() > 1e-3
    return {
        "numel": out.numel(),
        "mismatch": int(mismatch.sum()),
        "mismatch_not_near": int((mismatch & ~near).sum()),
        "near": int(near.sum()),
        "over_bound": int((err > half_ulp + bound).sum()),
        "max_abs": float(err.max()),
        "max_rel": float((err[big] / ref.abs()[big]).max()),
        "torch_max_abs": float(torch_err.max()),
        "torch_max_rel": float((torch_err[big] / ref.abs()[big]).max()),
    }


def _interpreted() -> dict:
    """Run in a ``TRITON_INTERPRET=1`` subprocess; returns JSON."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv as g

    result = {"cases": {}, "invariant": {}, "bf16_weight": {}, "counters_zero": True}
    for experts in GATES:
        w = gate_weight(experts, experts)
        x_all = activations(max(ROWS), experts + 1)
        whole = None
        rows_alone = {}
        for rows in ROWS:
            out = g.gate_gemv(x_all[:rows], w)
            result["cases"][f"{experts}x{rows}"] = judge(out, x_all[:rows], w)
            rows_alone[rows] = out
            whole = out
            for part, count in g._SCRATCH.values():
                result["counters_zero"] &= bool((count == 0).all())
        # Row r of every call equals row r of the 64-row call (and so of
        # every other call): the result does not depend on the other rows.
        result["invariant"][str(experts)] = all(
            torch.equal(rows_alone[r], whole[:r]) for r in ROWS
        )
        # A full-FP32 weight (not FP16 values): the products stay exact.
        w32 = gate_weight(experts, experts + 7, fp16_values=False)
        out = g.gate_gemv(x_all[:6], w32)
        result["cases"][f"{experts}x6_fp32w"] = judge(out, x_all[:6], w32)
        # A strided input (a row-slice of a wider buffer) reads the same rows.
        wide = torch.zeros(6, HIDDEN + 64, dtype=torch.bfloat16)
        wide[:, 32 : 32 + HIDDEN] = x_all[:6]
        strided = wide[:, 32 : 32 + HIDDEN]
        result["strided_" + str(experts)] = torch.equal(
            g.gate_gemv(strided, w), rows_alone[6]
        )
        # The BF16-held weight: the bytes of its values held in FP32.
        w16 = w.to(torch.bfloat16)
        result["bf16_weight"][str(experts)] = all(
            torch.equal(g.gate_gemv(x_all[:r], w16).view(torch.int32),
                        g.gate_gemv(x_all[:r], w16.float()).view(torch.int32))
            for r in BF16_ROWS
        )
        for part, count in g._SCRATCH.values():
            result["counters_zero"] &= bool((count == 0).all())

    scratch = dict(g._SCRATCH)
    for experts in GATES:
        for rows in (6, 33):
            g.gate_gemv(activations(rows, 3), gate_weight(experts, 5))
    result["scratch_reused"] = all(g._SCRATCH[k] is v for k, v in scratch.items()) and (
        set(g._SCRATCH) == set(scratch)
    )
    return result


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_gate_gemv import "
            "_interpreted; print('RESULT' + json.dumps(_interpreted()))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def test_interpreted_logits_are_the_rounded_fp64_product(interpreted):
    cases = interpreted["cases"]
    assert set(cases) == {f"{e}x{r}" for e in GATES for r in ROWS} | {
        f"{e}x6_fp32w" for e in GATES
    }
    for name, c in cases.items():
        assert c["mismatch_not_near"] == 0, (name, c)
        assert c["over_bound"] == 0, (name, c)
        # FP32 accumulation is the larger error by far.
        assert c["max_abs"] <= c["torch_max_abs"], (name, c)
    for gate in (384, 128):
        rows = [cases[f"{gate}x{r}"] for r in ROWS]
        print(f"gate {gate} x {HIDDEN}, rows {ROWS}: "
              f"mismatches with RN32(FP64) {sum(c['mismatch'] for c in rows)} of "
              f"{sum(c['numel'] for c in rows)} (near a midpoint: "
              f"{sum(c['near'] for c in rows)}); vs FP64 max abs "
              f"{max(c['max_abs'] for c in rows):.3e}, max rel "
              f"{max(c['max_rel'] for c in rows):.3e}; torch FP32 max abs "
              f"{max(c['torch_max_abs'] for c in rows):.3e}, max rel "
              f"{max(c['torch_max_rel'] for c in rows):.3e}")
        print(f"gate {gate}, full FP32 weight, 6 rows: {cases[f'{gate}x6_fp32w']}")


def test_interpreted_bf16_weight_gives_the_bytes_of_its_fp32_values(interpreted):
    assert interpreted["bf16_weight"] == {"384": True, "128": True}


def test_interpreted_rows_are_independent_and_strided_input_reads_the_same(interpreted):
    assert interpreted["invariant"] == {"384": True, "128": True}
    assert interpreted["strided_384"] and interpreted["strided_128"]


def test_interpreted_counters_reset_and_scratch_allocated_once(interpreted):
    assert interpreted["counters_zero"]
    assert interpreted["scratch_reused"]


def test_a_call_reads_nothing_on_the_host_and_allocates_its_output_only(monkeypatch):
    """The host side of a call (the kernel replaced by a recorder): after the
    first call no host read, no allocation but the output, and one launch;
    the scratch is never allocated during a graph capture."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from tests.models.deepseek_v41.sm80.test_no_host_sync import _SYNC_OPS
    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv as g

    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.append(func._schema.name.split("::")[-1])
            return func(*args, **(kwargs or {}))

    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, kwargs))

    monkeypatch.setattr(g, "_gate_gemv_kernel", Kernel())
    monkeypatch.setattr(g, "_SCRATCH", {})
    x, x48, w = activations(6, 3), activations(48, 3), gate_weight(384, 384)
    g.gate_gemv(x, w)
    with Ops() as ops:
        out = g.gate_gemv(x, w)
    assert out.shape == (6, 384) and out.dtype == torch.float32

    def allocations(names):
        assert not _SYNC_OPS & set(names), names
        return [n for n in names if n.startswith(("empty", "zeros", "ones", "full", "new_",
                                                  "clone", "_to_copy", "copy"))]

    assert allocations(ops.names) == ["empty"], ops.names
    assert len(launches) == 2
    with Ops() as ops:
        g.gate_gemv(x48, w)
    assert allocations(ops.names) == ["empty"], ops.names
    assert len(launches) == 3
    # A BF16 weight: still no allocation but the output.
    w16 = w.to(torch.bfloat16)
    with Ops() as ops:
        g.gate_gemv(x, w16)
    assert allocations(ops.names) == ["empty"], ops.names
    assert len(launches) == 4
    # A first call inside a capture refuses before allocating anything.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside a CUDA graph capture"):
        g._scratch(torch.device("cuda", 0), 384, 20)
    assert list(g._SCRATCH) == [("cpu", 384, 20)]


def _ptxas():
    import triton

    return os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")


@pytest.mark.parametrize("w_pairs", [False, True])
def test_kernel_compiles_for_sm80(tmp_path, w_pairs):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv as g

    types = {"x_ptr": "*i32", "w_ptr": "*i32" if w_pairs else "*fp32", "out_ptr": "*fp32",
             "part_ptr": "*fp64", "count_ptr": "*i32", "M": "i32", "stride_xm": "i32",
             "stride_wn": "i32", "stride_om": "i32"}
    kern = g._gate_gemv_kernel
    sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
    for experts in GATES:
        split = g.split_for(experts, HIDDEN, 74)
        for block_t in (8, 16, 32):
            consts = dict(N=experts, K=HIDDEN, SPLIT=split, BLOCK_T=block_t,
                          BLOCK_E=g.BLOCK_E, BLOCK_K=g.BLOCK_K, W_PAIRS=w_pairs)
            c = triton.compile(ASTSource(kern, sig, constexprs=consts),
                               target=GPUTarget("cuda", 80, 32),
                               options={"num_warps": 4, "num_stages": 3})
            ptx = c.asm["ptx"]
            mma = re.findall(r"mma\.sync\.aligned\.([a-z0-9.]+)", ptx)
            assert set(mma) == {"m8n8k4.row.col.f64.f64.f64.f64"}
            # Per warp (4 warps): BLOCK_E x BLOCK_T x BLOCK_K of m8n8k4 tiles,
            # none computed twice.
            assert len(mma) * 4 == (g.BLOCK_E // 8) * (block_t // 8) * (g.BLOCK_K // 4)
            assert "cvt.rn.f32.f64" in ptx and "ftz" not in ptx
            assert "atom.global.gpu.acq_rel.add.u32" in ptx
            assert "cp.async" in ptx  # the weight loads are pipelined
            path = tmp_path / f"k{experts}_{block_t}.ptx"
            path.write_text(ptx)
            info = subprocess.run([_ptxas(), "-arch=sm_80", "-v", str(path), "-o",
                                   str(path.with_suffix(".cubin"))],
                                  capture_output=True, text=True, check=True).stderr
            assert "0 bytes spill stores, 0 bytes spill loads" in info, info


def test_supports_both_gates_and_refuses_partial_k_blocks():
    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv as g

    assert g.supports(384, HIDDEN) is None and g.supports(128, HIDDEN) is None
    assert "multiple" in g.supports(384, 5000)


def test_serves_takes_decode_calls_only():
    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv as g

    w = torch.empty(384, HIDDEN)
    x = torch.empty(6, HIDDEN, dtype=torch.bfloat16)
    assert not g.serves(x, w)  # CPU tensors keep the FP32 GEMM

    class Cuda:
        """The metadata of a CUDA tensor."""

        def __init__(self, t, rows=None):
            self.t, self.is_cuda, self.dtype = t, True, t.dtype
            self.shape = t.shape if rows is None else (rows, t.shape[1])

        def __getattr__(self, name):
            return getattr(self.t, name)

    assert g.serves(Cuda(x), w)
    assert g.serves(Cuda(x, rows=64), w) and not g.serves(Cuda(x, rows=65), w)
    assert not g.serves(Cuda(x, rows=0), w)
    assert not g.serves(Cuda(x.float()), w)
    assert not g.serves(Cuda(torch.empty(6, HIDDEN + 1, dtype=torch.bfloat16)[:, 1:]), w)
    assert g.serves(Cuda(x), w.bfloat16())  # the BF16-held weight
    assert not g.serves(Cuda(x), w.half())
    assert not g.serves(Cuda(x), torch.empty(HIDDEN, 384).T)
    assert not g.serves(Cuda(x), torch.empty(HIDDEN, 384, dtype=torch.bfloat16).T)


def _gate(monkeypatch, experts=8, hidden=64):
    from tests.models.deepseek_v41.sm80 import construct
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    construct.set_groups(monkeypatch, 1, 0)
    gate = GateLinear(hidden, experts, bias=False, out_dtype=torch.float32,
                      prefix="layers.0.ffn.gate")
    assert type(gate) is GateLinear
    return gate


def test_switch_default_off_and_adoption(monkeypatch):
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_GATE_GEMV", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_GATE_GEMV is False
    assert sm80_moe.Sm80DeepseekV41Gate.gemv is False
    for switch in (False, True):
        monkeypatch.setenv("VLLM_DEEPSEEK_V41_GATE_GEMV", "1" if switch else "0")
        assert envs.VLLM_DEEPSEEK_V41_GATE_GEMV is switch
        gate = _gate(monkeypatch)
        gate.e_score_correction_bias = torch.zeros(8)
        gate.tid2eid = None
        router = object.__new__(FusedTopKBiasRouter)
        router.num_fused_shared_experts = 0
        ffn = SimpleNamespace(use_native_mega_moe=False, gate=gate,
                              scoring_func="sqrtsoftplus",
                              experts=SimpleNamespace(router=router))
        sm80_moe.adopt_moe(ffn, SimpleNamespace())
        assert type(gate) is sm80_moe.Sm80DeepseekV41Gate
        assert gate.weight.dtype == torch.float32
        assert gate.gemv is switch


def test_hybrid_capture_refuses_the_kernel(monkeypatch, tmp_path):
    """The hybrid capture records gate calls through GateLinear.forward,
    which the kernel bypasses, so a capture boot with the flag on refuses."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_GATE_GEMV", "1")
    spec = tmp_path / "capture.toml"
    spec.write_text('dir = "capture"\nfamilies = ["attn", "hybrid"]\n')
    monkeypatch.setenv("VLLM_SM80_CAPTURE", str(spec))
    gate = _gate(monkeypatch)
    gate.e_score_correction_bias = torch.zeros(8)
    gate.tid2eid = None
    router = object.__new__(FusedTopKBiasRouter)
    router.num_fused_shared_experts = 0
    ffn = SimpleNamespace(use_native_mega_moe=False, gate=gate, scoring_func="sqrtsoftplus",
                          experts=SimpleNamespace(router=router))
    with pytest.raises(ValueError, match="hybrid capture"):
        sm80_moe.adopt_moe(ffn, SimpleNamespace())


def test_gate_forward_takes_the_kernel_only_when_on_and_served(monkeypatch):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm_sm80.models.deepseek_v41 import sm80_moe
    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv

    gate = _gate(monkeypatch)
    sm80_moe._fp32_gate_weight(gate)
    gate.weight.data.copy_(torch.randn(8, 64))
    x = torch.randn(6, 64).to(torch.bfloat16)
    fallback = GateLinear.forward(gate, x)[0]
    assert torch.equal(fallback, x.float() @ gate.weight.T)
    calls = []
    sentinel = torch.full((6, 8), 7.0)
    monkeypatch.setattr(sm80_gate_gemv, "gate_gemv",
                        lambda x, w: calls.append((x, w)) or sentinel)
    # Off: the FP32 fallback, even for a call the kernel serves.
    monkeypatch.setattr(sm80_gate_gemv, "serves", lambda x, w: True)
    assert torch.equal(gate(x)[0], fallback) and not calls
    gate.gemv = True
    out, bias = gate(x)
    assert out is sentinel and bias is None and calls == [(x, gate.weight)]
    gate.return_bias = False
    assert gate(x) is sentinel
    # On, but not served (prefill rows, CPU): the fallback.
    monkeypatch.setattr(sm80_gate_gemv, "serves", lambda x, w: False)
    assert torch.equal(gate(x), fallback)


def test_dev_ab_switch_toggles_every_gate(monkeypatch):
    from vllm_sm80.models.deepseek_v41 import sm80_moe
    from vllm_sm80 import dev_ab

    switch = dev_ab.SWITCHES["dsv41_gate_gemv"]
    assert switch.graph_bound and not switch.output_identical
    assert switch.env == "VLLM_DEEPSEEK_V41_GATE_GEMV"
    target = torch.nn.ModuleList([_gate(monkeypatch, hidden=HIDDEN) for _ in range(2)])
    draft = torch.nn.ModuleList([_gate(monkeypatch, experts=4, hidden=HIDDEN)])
    gates = [*target, *draft]
    runner = SimpleNamespace(model=target, speculator=SimpleNamespace(model=draft))
    assert "no sm_80 DeepSeek-V4.1 MoE gates" in switch.unavailable(runner)
    for gate in gates:
        sm80_moe._fp32_gate_weight(gate)
    assert switch.unavailable(runner) is None and switch.current(runner) is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_GATE_GEMV", "0")
    switch.apply(runner, True)
    assert switch.current(runner) is True and all(g.gemv for g in gates)
    assert dev_ab.os.environ["VLLM_DEEPSEEK_V41_GATE_GEMV"] == "1"
    switch.apply(runner, False)
    assert switch.current(runner) is False and not any(g.gemv for g in gates)
    assert dev_ab.os.environ["VLLM_DEEPSEEK_V41_GATE_GEMV"] == "0"
