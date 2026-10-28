# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 DeepSeek-V4.1 MoE gate GEMV on CUDA (one card; skipped without).

- The logits of both gates for every row count from 1 to 64 against the FP64
  product: the FP32 rounding of it except within the FP64 summation error of
  a rounding midpoint, and within half an FP32 ulp plus that error.
- Route ids (``sm80_moe.route``) from the kernel's logits against those from
  the FP32 GEMM the gate ran before (``x.float() @ w.T``, cuBLAS): on random
  inputs every row whose experts differ has a near tie at the k-th place; on
  inputs built with a near tie at the k-th place the kernel never orders the
  two candidates against the exact order, while the FP32 GEMM may. The
  disagreement counts are printed (``-s``).
- Graph replay equals eager (new inputs copied in, two gates in one graph,
  eager calls between replays); repeated calls are bitwise equal; a warm call
  does not synchronise with the host.
- The adopted gate takes the kernel up to 64 rows and the FP32 GEMM above.
- The gate weight held once, in BF16 (VLLM_DEEPSEEK_V41_ROUTER_MMA): the
  kernel's logits at 1 to 64 rows and the FP32 fallback's (cuBLAS) at 65 to
  3000 rows are the bytes of the FP32-held gate's, both gates.

    pytest -q -s tests/models/deepseek_v41/sm80/test_sm80_gate_gemv_gpu.py
"""

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops.sm80_gate_gemv import MAX_ROWS, gate_gemv
from vllm_sm80.models.deepseek_v41.sm80_moe import route

from .test_sm80_gate_gemv import GATES, HIDDEN, activations, gate_weight, judge

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

TOP_K = {384: 6, 128: 3}


def _cuda(experts, rows, seed):
    return (activations(rows, seed).cuda(), gate_weight(experts, seed + 1).cuda())


@pytest.mark.parametrize("experts", GATES)
def test_logits_are_the_rounded_fp64_product(experts):
    x_all, w = _cuda(experts, MAX_ROWS, experts)
    totals = dict(numel=0, mismatch=0, near=0, max_abs=0.0, torch_max_abs=0.0)
    for rows in range(1, MAX_ROWS + 1):
        c = judge(gate_gemv(x_all[:rows], w), x_all[:rows], w)
        assert c["mismatch_not_near"] == 0 and c["over_bound"] == 0, (rows, c)
        for key in ("numel", "mismatch", "near"):
            totals[key] += c[key]
        for key in ("max_abs", "torch_max_abs"):
            totals[key] = max(totals[key], c[key])
    print(f"gate {experts}: {totals}")


def _scores(logits, bias):
    return torch.nn.functional.softplus(logits.double()).sqrt() + bias.double()


@pytest.mark.parametrize("experts", GATES)
def test_route_ids_agree_with_the_fp32_gemm_but_at_near_ties(experts):
    top_k = TOP_K[experts]
    g = torch.Generator().manual_seed(experts)
    differ = rows_total = 0
    for seed in range(20):
        x, w = _cuda(experts, 48, 100 * seed + experts)
        bias = (torch.randn(experts, generator=g) * 0.1).cuda()
        ours = gate_gemv(x, w)
        gemm = x.float() @ w.T
        _, ids_ours = route(ours, bias, top_k, routed_scaling_factor=1.5)
        _, ids_gemm = route(gemm, bias, top_k, routed_scaling_factor=1.5)
        rows = (ids_ours.sort(-1).values != ids_gemm.sort(-1).values).any(-1)
        exact = _scores(x.double() @ w.double().T, bias).sort(-1, descending=True).values
        gap = exact[:, top_k - 1] - exact[:, top_k]
        # A different set only where the k-th and (k+1)-th exact scores are
        # within the FP32 GEMM's error of each other.
        assert bool((gap[rows] < 1e-4).all()), gap[rows]
        differ += int(rows.sum())
        rows_total += rows.numel()
    print(f"gate {experts}/top {top_k}, random: {differ} of {rows_total} rows route "
          f"to other experts than with the FP32 GEMM")


@pytest.mark.parametrize("experts", GATES)
def test_near_ties_follow_the_exact_order(experts):
    """Experts a and b compete for the k-th place (the k-1 places above are
    taken by bias, the rest pushed down); b's weight row is a's but for one
    column j, where a holds 0 and b holds n * 2**-20 (an FP16 value), so the
    two logits differ by exactly x[:, j] * n * 2**-20: 0 to ~1e-5."""
    top_k = TOP_K[experts]
    a, b = top_k - 1, top_k
    rows = MAX_ROWS
    gen = torch.Generator().manual_seed(7)
    picks = torch.randint(0, HIDDEN, (rows,), generator=gen)
    w = gate_weight(experts, 3).cuda()
    bias = torch.full((experts,), -10.0)
    bias[: top_k - 1] = 10.0
    bias[a] = bias[b] = 0.0
    bias = bias.cuda()
    ours_flips = gemm_flips = ours_ties = disagree = 0
    for seed in range(20):
        x = activations(rows, 1000 + seed).cuda()
        j = int(picks[seed])
        wb = w.clone()
        wb[a, j] = 0.0
        wb[b] = wb[a]
        wb[b, j] = 2.0**-20 * (seed - 10)
        exact = x.double() @ wb.double().T
        ours = gate_gemv(x, wb)
        gemm = x.float() @ wb.T
        want = exact[:, b] > exact[:, a]
        ties = ours[:, b] == ours[:, a]
        # Rounding is monotone: a strict order of the kernel's logits is the
        # exact order.
        assert bool((((ours[:, b] > ours[:, a]) == want) | ties).all())
        ours_ties += int(ties.sum())
        gemm_flips += int(((gemm[:, b] > gemm[:, a]) != want).sum())
        _, ids_ours = route(ours, bias, top_k)
        _, ids_gemm = route(gemm, bias, top_k)
        disagree += int((ids_ours.sort(-1).values != ids_gemm.sort(-1).values).any(-1).sum())
        ours_flips += int((((ours[:, b] > ours[:, a]) != want) & ~ties).sum())
    assert ours_flips == 0
    print(f"gate {experts}/top {top_k}, near ties over {20 * rows} rows: FP32 GEMM orders "
          f"{gemm_flips} against the exact order; kernel 0 (and {ours_ties} FP32 ties); "
          f"routes differ in {disagree} rows")


def test_graph_replay_equals_eager_and_no_host_sync():
    x384, w384 = _cuda(384, 6, 1)
    x128, w128 = _cuda(128, 6, 2)
    call = lambda: (gate_gemv(x384, w384), gate_gemv(x128, w128))  # noqa: E731
    eager = call()
    assert all(torch.equal(a, b) for a, b in zip(eager, call()))
    torch.cuda.set_sync_debug_mode("error")
    try:
        call()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for seed in range(3):
        x384.copy_(activations(6, 10 + seed).cuda())
        x128.copy_(activations(6, 20 + seed).cuda())
        want = call()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(captured, want)), seed
        # An eager call of another shape between replays.
        gate_gemv(activations(48, seed).cuda(), w384)


def test_adopted_gate_takes_the_kernel_up_to_64_rows(monkeypatch):
    from tests.models.deepseek_v41.sm80 import construct
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    construct.set_groups(monkeypatch, 1, 0)
    with torch.device("cuda"):
        gate = GateLinear(HIDDEN, 384, bias=False, out_dtype=torch.float32,
                          prefix="layers.0.ffn.gate")
    sm80_moe._fp32_gate_weight(gate)
    gate.weight.data.copy_(gate_weight(384, 9))
    gate.gemv = True
    x = activations(MAX_ROWS + 1, 9).cuda()
    for rows in (1, 6, 32, MAX_ROWS):
        assert torch.equal(gate(x[:rows])[0], gate_gemv(x[:rows], gate.weight))
    assert torch.equal(gate(x)[0], x.float() @ gate.weight.T)


@pytest.mark.parametrize("experts", GATES)
def test_bf16_held_gate_gives_the_fp32_held_gates_bytes(monkeypatch, experts):
    from tests.models.deepseek_v41.sm80 import construct
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    construct.set_groups(monkeypatch, 1, 0)
    # BF16 values (the hybrid checkpoint's gate dtype).
    ck = gate_weight(experts, 11).to(torch.bfloat16)
    gates = {}
    for dtype in (torch.float32, torch.bfloat16):
        with torch.device("cuda"):
            gate = GateLinear(HIDDEN, experts, bias=False, out_dtype=torch.float32,
                              prefix="layers.0.ffn.gate")
        sm80_moe._hold_gate_weight(gate, dtype)
        gate.weight.data.copy_(ck)
        gate.gemv = True
        gates[dtype] = gate
    held = gates[torch.bfloat16]
    assert held.weight.dtype == torch.bfloat16
    x_all = activations(3000, 12).cuda()
    for rows in (*range(1, MAX_ROWS + 1), 65, 2047, 2048, 2049, 3000):
        want = gates[torch.float32](x_all[:rows])[0]
        got = held(x_all[:rows])[0]
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), rows
