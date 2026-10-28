# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The BF16 tensor-core MoE gate and route (``VLLM_DEEPSEEK_V41_ROUTER_MMA``,
``ops/sm80_router_mma.py``) against the served fused route
(``ops/sm80_route.py``) on one sm_80 card.

- Selection: given the same FP32 logits, the route picks the experts
  ``route_fused`` picks, bit for bit (int32 ids), for the target's router
  (384 experts, top 6) and the drafter's (128, top 3), at 1, 33 and 64 rows
  (the Gluon route) and 65, 257 and 2047 rows (the Triton routes), with
  image rows (``bias_vl``), forced ties (two experts with the same gate row
  and bias) and NaN rows. The logits are the gate's, written in full
  (``sm80_gate_mma.launch`` without DEFER: the FP64 partials summed in the
  order the deferred route sums them, rounded once).
- Weights: within one FP32 ulp of the FP64 weights of the logits the route
  takes them from (the exact x @ W^T up to ``EXACT_ROWS`` rows, the gate's
  FP32 logits above); NaN rows give NaN weights, as the served route.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_router_mma_gpu.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 card",
)

ROWS = (1, 33, 64, 65, 257, 2047)
NAN_ROW, TIE = 0, (2, 3)


def _case(rows: int, experts: int, seed: int):
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma as rm

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, rm.HIDDEN, generator=g)
    x[1::5] *= 20.0  # logits above softplus's threshold of 20
    x[NAN_ROW, 7] = float("nan")
    w = torch.randn(experts, rm.HIDDEN, generator=g) * 0.02
    w[TIE[1]] = w[TIE[0]]
    bias = torch.randn(experts, generator=g) * 0.1
    bias_vl = torch.randn(experts, generator=g) * 0.1
    # The tied experts rank first on most rows, so their order is tested.
    bias[list(TIE)] = 4.0
    bias_vl[list(TIE)] = 4.0
    input_ids = torch.randint(0, 2000, (rows,), generator=g, dtype=torch.int32)
    input_ids[::3] = rm.SENTINEL_LO + (torch.arange(0, rows, 3) % 5).int()
    return (x.to(torch.bfloat16).cuda(), w.to(torch.bfloat16).cuda(), bias.cuda(),
            bias_vl.cuda(), input_ids.cuda())


def _gate_logits(x, w16):
    """The gate's FP32 logits, written in full (no DEFER)."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma as rm

    M, K = x.shape
    E = w16.shape[0]
    sms = rm._sms(x.device)
    logits = torch.full((M, E), float("nan"), dtype=torch.float32, device=x.device)
    part = torch.empty(max(rm.G.part_numel(M, E, K, sms), 1), dtype=torch.float64,
                       device=x.device)
    rm.G.launch(x, w16, logits, part, rm._locks(x.device), sms, False)
    return logits


def _weights64(logits64, ids, scale):
    s = (logits64.clamp(min=0) + torch.log1p(torch.exp(-logits64.abs()))).sqrt()
    chosen = s.gather(1, ids.long())
    return chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * scale


def _within_one_ulp(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    nan = got.isnan() & want.isnan()
    ref = want.float().abs()
    ulp = (torch.nextafter(ref, torch.full_like(ref, float("inf"))) - ref).double()
    return nan | ((got.double() - want).abs() <= ulp)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("experts", [384, 128])
def test_route_selects_what_the_fused_route_selects(rows, experts):
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma as rm
    from vllm_sm80.models.deepseek_v41.ops.sm80_route import route_fused

    top_k = rm.EXPERTS_TOP_K[experts]
    for seed in range(2):
        x, w16, bias, bias_vl, input_ids = _case(rows, experts, seed)
        assert rm.serves(x)
        logits = _gate_logits(x, w16)
        assert not logits[1:].isnan().any() and logits[NAN_ROW].isnan().all()
        for vision in (False, True):
            lo = rm.SENTINEL_LO if vision else 0
            want_w, want_ids = route_fused(
                logits, bias, top_k, bias_vl=bias_vl, input_ids=input_ids,
                sentinel_lo=lo, sentinel_hi=rm.SENTINEL_HI if vision else 0,
                routed_scaling_factor=rm.SCALE,
            )  # fmt: skip
            got_w, got_ids = rm.gate_route(x, w16, bias, bias_vl, top_k,
                                           input_ids=input_ids, sentinel_lo=lo)
            torch.cuda.synchronize()
            where = f"{experts} experts, {rows} rows, seed {seed}, images {vision}"
            assert got_ids.dtype == torch.int32, where
            assert torch.equal(got_ids, want_ids), where
            assert torch.equal(got_ids[NAN_ROW].cpu(), torch.arange(top_k, dtype=torch.int32))
            if rows > 1:
                # The tied pair, where chosen, in expert order.
                row = got_ids[1:].tolist()
                pairs = [r.index(TIE[0]) < r.index(TIE[1]) for r in row
                         if TIE[0] in r and TIE[1] in r]
                assert pairs and all(pairs), where
            exact = rows <= rm.EXACT_ROWS
            logits64 = x.double() @ w16.double().T if exact else logits.double()
            want64 = _weights64(logits64, got_ids, rm.SCALE)
            ok = _within_one_ulp(got_w, want64)
            assert ok.all(), (where, int((~ok).sum()))
            assert got_w[NAN_ROW].isnan().all() and want_w[NAN_ROW].isnan().all(), where
