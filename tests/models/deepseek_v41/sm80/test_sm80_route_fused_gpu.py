# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 MoE routing (``ops/sm80_route.py``) on one
sm_80 card.

- Bytes: weights and int32 ids equal to ``sm80_moe.route`` + ``.to(int32)``
  on CUDA, for the target's router (384 experts, top 6) and the drafter's
  (128, top 3), at every decode graph size and two prefill sizes, with and
  without image tokens, renormalised or not, on logits spanning both
  softplus branches and exp's subnormal range.
- The CPU half's model of torch's CUDA row reduction
  (``test_sm80_route_fused.torch_cuda_row_sum``) is ``torch.sum`` on CUDA,
  bitwise, for rows of 1 to 16 values.
- CUDA graph replay equals eager; two runs are bitwise equal.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_route_fused_gpu.py
"""

import numpy as np
import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops.sm80_route import route_fused
from vllm_sm80.models.deepseek_v41.sm80_moe import image_token_mask, route

from .test_sm80_route_fused import IMAGE_IDS, ROUTERS, SENTINEL, torch_cuda_row_sum

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80 card",
)

TOKENS = (1, 4, 5, 6, 7, 8, 12, 16, 18, 24, 28, 32, 40, 48, 2048, 2044)


def cuda_case(tokens, experts, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=g) * 4.0
    # Both softplus branches and exp's subnormal range.
    logits[:, :6] = torch.tensor([25.0, 20.0, 19.999, -88.0, -95.0, -103.0])
    bias = torch.randn(experts, generator=g) * 0.1
    bias_vl = torch.randn(experts, generator=g) * 0.1
    input_ids = torch.randint(0, 2000, (tokens,), generator=g, dtype=torch.int32)
    input_ids[::3] = SENTINEL + (torch.arange(0, tokens, 3) % IMAGE_IDS).int()
    return [t.cuda() for t in (logits, bias, bias_vl, input_ids)]


def both(logits, bias, bias_vl, input_ids, top_k, images, renorm):
    lo = SENTINEL if images else 0
    got = route_fused(logits, bias, top_k, bias_vl=bias_vl, input_ids=input_ids,
                      sentinel_lo=lo, sentinel_hi=lo + IMAGE_IDS, renormalize=renorm,
                      routed_scaling_factor=1.5)
    w, i = route(logits, bias, top_k, bias_vl=bias_vl,
                 image_mask=image_token_mask(input_ids, lo), renormalize=renorm,
                 routed_scaling_factor=1.5)
    return got, (w, i.to(torch.int32))


@pytest.mark.parametrize("experts,top_k", ROUTERS)
@pytest.mark.parametrize("images", (False, True))
@pytest.mark.parametrize("renorm", (True, False))
def test_bytes_equal_the_torch_path(experts, top_k, images, renorm):
    for tokens in TOKENS:
        args = cuda_case(tokens, experts, tokens * 7 + experts)
        (gw, gi), (ww, wi) = both(*args, top_k, images, renorm)
        assert gi.dtype == torch.int32 and gw.dtype == torch.float32
        assert torch.equal(gi, wi), tokens
        assert torch.equal(gw.view(torch.int32), ww.view(torch.int32)), tokens


@pytest.mark.parametrize("experts,top_k", ROUTERS)
def test_nan_rows_rank_nan_first_like_torch(experts, top_k):
    logits, bias, bias_vl, input_ids = cuda_case(8, experts, 5)
    logits[0, 7] = float("nan")
    logits[1, 3] = float("nan")
    logits[1, experts - 1] = float("nan")
    logits[2, :] = float("nan")
    (gw, gi), (ww, wi) = both(logits, bias, bias_vl, input_ids, top_k, False, True)
    assert bool(((gi >= 0) & (gi < experts)).all())
    assert torch.equal(torch.isnan(gw), torch.isnan(ww))
    # One NaN: torch and the kernel take it first, then the same finite ids.
    assert torch.equal(gi[0], wi[0])
    # Finite rows keep the torch path's bytes.
    assert torch.equal(gi[3:], wi[3:])
    assert torch.equal(gw[3:].view(torch.int32), ww[3:].view(torch.int32))


@pytest.mark.parametrize("experts,top_k", ROUTERS)
def test_row_mask_with_nan_rows(experts, top_k):
    # The row mask (VLLM_DSPARK_ADAPTIVE_DEPTH) on finite and NaN rows: masked rows
    # take id -1 and weight +0.0 whatever their logits; live rows, NaN ones
    # included, keep the unmasked kernel's bytes.
    logits, bias, bias_vl, input_ids = cuda_case(12, experts, 9)
    logits[0, 7] = float("nan")
    logits[2, :] = float("nan")
    logits[5, 3] = float("nan")
    logits[7, :] = float("nan")
    live = torch.ones(16, dtype=torch.int8, device="cuda")
    live[torch.tensor([1, 2, 3, 7, 11])] = 0
    call = lambda row_live: route_fused(  # noqa: E731
        logits, bias, top_k, bias_vl=bias_vl, input_ids=input_ids,
        sentinel_lo=SENTINEL, sentinel_hi=SENTINEL + IMAGE_IDS,
        routed_scaling_factor=1.5, row_live=row_live,
    )
    plain_w, plain_i = call(None)
    w, i = call(live)
    keep = live[:12].bool()
    assert torch.equal(i[keep], plain_i[keep])
    assert torch.equal(w[keep].view(torch.int32), plain_w[keep].view(torch.int32))
    assert bool((i[~keep] == -1).all())
    assert torch.equal(w[~keep].view(torch.int32), torch.zeros_like(w[~keep]).view(torch.int32))
    # An all-live mask is the unmasked kernel, byte for byte.
    w1, i1 = call(torch.ones(16, dtype=torch.int8, device="cuda"))
    assert torch.equal(i1, plain_i) and torch.equal(w1.view(torch.int32), plain_w.view(torch.int32))


def test_row_sum_model_is_torch_sum_on_cuda():
    g = torch.Generator().manual_seed(5)
    for k in range(1, 17):
        rows = (torch.randn(4096, k, generator=g) ** 2
                * 10.0 ** torch.randint(-3, 3, (4096, k), generator=g)).float()
        got = rows.cuda().sum(dim=-1).cpu().numpy()
        want = np.array([torch_cuda_row_sum(r) for r in rows.numpy()], dtype=np.float32)
        assert np.array_equal(got.view(np.int32), want.view(np.int32)), k


def test_graph_replay_and_determinism():
    for experts, top_k in ROUTERS:
        logits, bias, bias_vl, ids = cuda_case(6, experts, 11)
        call = lambda: route_fused(logits, bias, top_k, bias_vl=bias_vl, input_ids=ids,  # noqa: E731
                                   sentinel_lo=SENTINEL, sentinel_hi=SENTINEL + IMAGE_IDS,
                                   routed_scaling_factor=1.5)
        eager = call()
        again = call()
        assert all(torch.equal(a, b) for a, b in zip(eager, again))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(eager, captured))
