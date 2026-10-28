# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The replicated Engram wkv on the real Marlin W8A16 GEMM: the DeepSeek-V4.1
wkv (6144 inputs, 25600 outputs, MXFP8 with 32 x 32 scale blocks) at TP4,
all four ranks on one card.

Each rank's replica slot is its processed shard, so the replicated row is the
gathered row byte for byte at every row count, and smaller steps project
from the rank's own slot with the bytes of today's shard.
"""

import pytest
import torch

import vllm.model_executor.layers.linear as linear
from tests.models.deepseek_v41.sm80.construct import set_groups
from tests.models.deepseek_v41.sm80.test_engram_wkv_replica import _Collective
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm_sm80.models.deepseek_v41 import sm80_engram_wkv as W

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin runs on CUDA")

BF16 = torch.bfloat16
TP, K, N = 4, 6144, 25600
MIN_ROWS = 49


def _checkpoint():
    generator = torch.Generator().manual_seed(1)
    weight = (torch.rand(N, K, generator=generator) * 2 - 1).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 130, (N // 32, K // 32), generator=generator).to(torch.uint8)
    return weight.cuda(), scale.cuda().view(torch.float8_e8m0fnu)


def _rank(monkeypatch, rank, checkpoint, adopt):
    """Rank ``rank``'s loaded MXFP8 wkv on Marlin, as the hybrid checkpoint's."""
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    set_groups(monkeypatch, TP, rank)
    with torch.device("cuda"):
        layer = ColumnParallelLinear(
            K, N, bias=False, gather_output=True, return_bias=False,
            params_dtype=BF16, prefix="layers.1.engram.wkv",
        )  # fmt: skip
    del layer.weight
    method = ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
        CkptCtx(scale_block_size=(32, 32)),
    )
    method.input_dtype = method.out_dtype = BF16

    def load(param, loaded, *args):
        param.data.copy_(loaded)

    with torch.device("cuda"):
        method.create_weights(layer, K, [N // TP], K, N, BF16, weight_loader=load)
    assert type(method.kernel).__name__ == "MarlinMxfp8LinearKernel"
    layer.quant_method = method
    weight, scale = checkpoint
    rows = slice(rank * N // TP, (rank + 1) * N // TP)
    blocks = slice(rank * N // TP // 32, (rank + 1) * N // TP // 32)
    layer.weight.weight_loader(layer.weight, weight[rows])
    layer.weight_scale.weight_loader(layer.weight_scale, scale[blocks])
    if adopt:
        W.ReplicatedPrefillWkv.adopt(layer, MIN_ROWS)
    else:
        method.process_weights_after_loading(layer)
    return layer


def test_marlin_replica_rows_are_the_gathered_bytes(monkeypatch):
    checkpoint = _checkpoint()
    plain = [_rank(monkeypatch, r, checkpoint, adopt=False) for r in range(TP)]
    adopted = [_rank(monkeypatch, r, checkpoint, adopt=True) for r in range(TP)]
    collective = _Collective()
    monkeypatch.setattr(W, "_gather_tp", collective.gather)
    monkeypatch.setattr(W, "get_tp_group", collective.group)
    collective.run(adopted)
    torch.cuda.synchronize()

    gathered = []

    def all_gather(output, dim=-1):
        gathered.append(output)
        return torch.cat(sharded, dim=-1)

    monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", all_gather)
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Decode and capture sizes (sharded), then prefill sizes past Marlin's
    # 64-row launches and parallel row slices, up to a 2048-row chunk.
    for rows in [1, 6, 40, 48, 49, 64, 65, 127, 128, 200, 1000, 2047, 2048]:
        x = torch.randn(rows, K, generator=generator, device="cuda").to(BF16)
        sharded = [p.quant_method.apply(p, x) for p in plain]
        want = torch.cat(sharded, dim=-1)
        for rank, layer in enumerate(adopted):
            gathered.clear()
            got = layer(x)
            assert got.shape == (rows, N)
            if rows < MIN_ROWS:
                assert len(gathered) == 1
                assert torch.equal(gathered[0].view(torch.int16), sharded[rank].view(torch.int16))
            else:
                assert not gathered
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (rows, rank)
    torch.cuda.synchronize()
