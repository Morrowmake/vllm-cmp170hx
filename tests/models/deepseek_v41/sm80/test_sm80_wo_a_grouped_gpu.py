# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grouped MXFP8 wo_a GEMM of DeepSeek-V4.1 on CUDA (one card; skipped
without).

Two loaded MXFP8 wo_a layers of the same codes and 32 x 32 block scales (the
hybrid checkpoint's): the per-group Marlin W8A16 GEMMs (GroupedLinearMethod)
and the grouped GEMM (WoAGroupedGemmMethod), at 2 and 4 local groups (TP4,
TP2) of 1024 outputs and 4096 inputs.

- Against the FP64 product of the decoded weights, for every row count from
  1 to 48 and 64 and for prefill row counts: every output within half a BF16
  ulp plus the FP32 summation bound of its dot product; the mean error at most
  1.10x Marlin's on the same inputs at every row count, the max error at most
  1.25x Marlin's over the decode row counts together and at every prefill row
  count. The share of outputs equal to Marlin's bytes is printed (``-s``).
- A row's bytes do not depend on the call's other rows (every row tile sums
  its K in the same order); repeated calls are bitwise equal.
- A row-padded attention output is read in place: the contiguous input's
  bytes.
- Graph replay equals eager.
- With the dev A/B switch's layers, the off arm is the per-group Marlin
  path's bytes.

    pytest -q -s tests/models/deepseek_v41/sm80/test_sm80_wo_a_grouped_gpu.py
"""

import pytest
import torch
from torch import nn

from tests.models.deepseek_v41.sm80.hybrid_linear_cases import decode_fp8
from vllm_sm80.models.deepseek_v41.ops import sm80_wo_a_grouped as wg
from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BF16 = torch.bfloat16
RANK_OUT, K = 1024, 4096
DECODE_ROWS = [*range(1, 49), 64]
PREFILL_ROWS = [65, 127, 128, 200, 256, 257, 1000, 2048, 8193]


def _codes(groups, seed):
    """Codes from a normal weight (subnormal to 448) and E8M0 block scales
    around 1, generated on the host (GPU RNG depends on the SM count)."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(groups * RANK_OUT, K, generator=g) * 96).clamp(-448, 448)
    blocks = torch.randint(118, 131, (groups * RANK_OUT // 32, K // 32), generator=g)
    return w.to(torch.float8_e4m3fn).view(torch.uint8), blocks.to(torch.uint8)


def _layer(monkeypatch, codes, blocks, adopt):
    """A loaded MXFP8 wo_a on CUDA, as the hybrid checkpoint's, adopted by
    ``adopt(layer)`` and processed."""
    from tests.models.deepseek_v41.sm80.construct import set_groups
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    n, k = codes.shape
    layer = nn.Module()
    layer.register_parameter("bias", None)
    method = ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
        CkptCtx(scale_block_size=(32, 32)),
    )
    method.input_dtype = method.out_dtype = BF16

    def load(param, loaded, *args):
        param.data.copy_(loaded)

    set_groups(monkeypatch, 1, 0)
    with torch.device("cuda"):
        method.create_weights(layer, k, [n], k, n, BF16, weight_loader=load)
    assert type(method.kernel).__name__ == "MarlinMxfp8LinearKernel"
    layer.weight.weight_loader(layer.weight, codes.cuda().view(torch.float8_e4m3fn))
    layer.weight_scale.weight_loader(
        layer.weight_scale, blocks.cuda().view(torch.float8_e8m0fnu)
    )
    layer.quant_method = method
    adopt(layer)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _marlin(groups):
    """The per-group Marlin GEMMs."""
    return lambda layer: GroupedLinearMethod.adopt(layer, groups)


def _gemm(groups, **kwargs):
    """The grouped GEMM."""
    return lambda layer: wg.WoAGroupedGemmMethod.adopt(layer, groups, **kwargs)


@pytest.fixture(scope="module", params=[2, 4], ids=lambda g: f"groups{g}")
def layers(request):
    """(groups, Marlin layer, grouped GEMM layer, FP64 weights [G, R, K])."""
    groups = request.param
    codes, blocks = _codes(groups, groups)
    with pytest.MonkeyPatch.context() as mp:
        marlin = _layer(mp, codes, blocks, _marlin(groups))
        gemm = _layer(mp, codes, blocks, _gemm(groups))
    w64 = decode_fp8(codes.view(torch.float8_e4m3fn), blocks.repeat_interleave(32, 0))
    return groups, marlin, gemm, w64.view(groups, RANK_OUT, K).cuda()


def _x(rows, groups, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, groups * K, generator=g).to(BF16).cuda().view(rows, groups, K)


def _half_ulp(v):
    """Half a BF16 ulp at |v| (FP64), subnormals included."""
    e = torch.floor(torch.log2(v.abs().clamp_min(2.0**-126)))
    return torch.ldexp(torch.ones_like(v), (e - 8).to(torch.int64))


def _errors(out, x, w64):
    """|out - FP64 product| and its bound: half a BF16 ulp plus K * 2**-23 *
    sum |x_k w_k| (FP32 sums of exact products, each add off by at most one
    FP32 ulp, as for a tensor core's truncating adds)."""
    xd = x.double()
    ref = torch.einsum("tgk,grk->tgr", xd, w64).flatten(1)
    absdot = torch.einsum("tgk,grk->tgr", xd.abs(), w64.abs()).flatten(1)
    err = (out.double() - ref).abs()
    half_ulp = _half_ulp(torch.maximum(ref.abs(), out.double().abs()))
    bound = half_ulp + K * 2.0**-23 * absdot
    return err, bound


def test_fp64_error_within_marlins(layers):
    groups, marlin, gemm, w64 = layers
    totals = {"ours": [], "marlin": [], "equal": 0, "numel": 0}
    decode_max = {"ours": 0.0, "marlin": 0.0}
    for rows in DECODE_ROWS + PREFILL_ROWS:
        x = _x(rows, groups, rows)
        ours = gemm.quant_method.apply_grouped(gemm, x)
        theirs = marlin.quant_method.apply_grouped(marlin, x)
        assert ours.shape == theirs.shape == (rows, groups * RANK_OUT)
        assert ours.dtype == BF16
        e_ours, bound = _errors(ours, x, w64)
        e_marlin, _ = _errors(theirs, x, w64)
        assert int((e_ours > bound).sum()) == 0, rows
        mean = (e_ours.mean().item(), e_marlin.mean().item())
        peak = (e_ours.max().item(), e_marlin.max().item())
        assert mean[0] <= 1.10 * mean[1], (rows, mean)
        if rows in DECODE_ROWS:
            decode_max["ours"] = max(decode_max["ours"], peak[0])
            decode_max["marlin"] = max(decode_max["marlin"], peak[1])
        else:
            assert peak[0] <= 1.25 * peak[1], (rows, peak)
        totals["equal"] += int((ours.view(torch.int16) == theirs.view(torch.int16)).sum())
        totals["numel"] += ours.numel()
        totals["ours"].append(mean[0])
        totals["marlin"].append(mean[1])
    assert decode_max["ours"] <= 1.25 * decode_max["marlin"], decode_max
    print(f"groups {groups}: decode max err {decode_max}; "
          f"mean err sum ours {sum(totals['ours']):.6g} marlin {sum(totals['marlin']):.6g}; "
          f"equal bytes {totals['equal']}/{totals['numel']}")


def test_rows_do_not_depend_on_the_call(layers):
    groups, _, gemm, _ = layers
    x = _x(max(PREFILL_ROWS), groups, 99)
    whole = wg.wo_a_grouped(x, gemm.wo_a_codes, gemm.wo_a_scales)
    for rows in DECODE_ROWS + PREFILL_ROWS:
        part = wg.wo_a_grouped(x[:rows], gemm.wo_a_codes, gemm.wo_a_scales)
        assert torch.equal(part.view(torch.int16), whole[:rows].view(torch.int16)), rows
        again = wg.wo_a_grouped(x[:rows], gemm.wo_a_codes, gemm.wo_a_scales)
        assert torch.equal(again.view(torch.int16), part.view(torch.int16)), rows


def test_padded_rows_are_read_in_place(layers):
    groups, _, gemm, _ = layers
    for rows in (1, 6, 33, 64, 300):
        x = _x(rows, groups, rows + 7)
        buf = torch.full((rows, groups * K + 64), float("nan"), dtype=BF16, device="cuda")
        buf[:, : groups * K] = x.flatten(1)
        padded = buf[:, : groups * K].unflatten(1, (groups, K))
        got = wg.wo_a_grouped(padded, gemm.wo_a_codes, gemm.wo_a_scales)
        want = wg.wo_a_grouped(x, gemm.wo_a_codes, gemm.wo_a_scales)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), rows


def test_graph_replay_equals_eager(layers):
    groups, _, gemm, _ = layers
    method = gemm.quant_method
    for rows in (1, 6, 24, 48):
        static = _x(rows, groups, 5).clone()
        method.apply_grouped(gemm, static)  # compile outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = method.apply_grouped(gemm, static)
        for seed in (11, 12):
            fresh = _x(rows, groups, seed)
            static.copy_(fresh)
            graph.replay()
            want = method.apply_grouped(gemm, fresh)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16)), rows


def test_dev_ab_off_arm_is_the_marlin_path(monkeypatch):
    groups = 2
    codes, blocks = _codes(groups, 3)
    marlin = _layer(monkeypatch, codes, blocks, _marlin(groups))
    both = _layer(monkeypatch, codes, blocks,
                  _gemm(groups, gemm=False, keep_groups=True))
    gemm = _layer(monkeypatch, codes, blocks, _gemm(groups))
    for rows in (1, 6, 48, 300):
        x = _x(rows, groups, rows)
        both.quant_method.gemm = False
        off = both.quant_method.apply_grouped(both, x)
        want = marlin.quant_method.apply_grouped(marlin, x)
        assert torch.equal(off.view(torch.int16), want.view(torch.int16)), rows
        both.quant_method.gemm = True
        on = both.quant_method.apply_grouped(both, x)
        want = gemm.quant_method.apply_grouped(gemm, x)
        assert torch.equal(on.view(torch.int16), want.view(torch.int16)), rows
