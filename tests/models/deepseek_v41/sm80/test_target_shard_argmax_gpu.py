# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy target sampling with a per-rank argmax
(``vllm_sm80/worker/target_argmax.py``) on one GPU.

- The fork runner's ``sample`` with the switch off and on in four gloo
  ranks sharing one card, at the served shape (4 ranks of 32320 columns, 129280
  tokens, 4096-column chunks), with the stock Sampler and RejectionSampler
  and the compiled kernels (the Triton top-k/top-p kernel included): the
  checks of the CPU test.
- The on path reads no device value on the host and replays from a CUDA
  graph with the same tokens as eager and as torch.argmax (one rank).

    CUDA_VISIBLE_DEVICES=<card> python -m pytest -q \
        tests/models/deepseek_v41/sm80/test_target_shard_argmax_gpu.py
"""

from types import SimpleNamespace

import pytest
import torch

from .test_target_shard_argmax import check_ranks, gloo_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB, WIDTH = 129280, 32320


def test_sample_matches_the_gathered_path_on_four_ranks(tmp_path):
    check_ranks(gloo_cases(str(tmp_path), "cuda", VOCAB, WIDTH, None))


def test_no_host_read_and_graph_replay():
    from vllm_sm80.worker import target_argmax as ta

    torch.manual_seed(0)
    hidden, rows, padded = 256, 16, VOCAB + 64
    weight = torch.randn(hidden, padded, device="cuda")
    weight[:, VOCAB:] = float("nan")  # padding columns are never read
    model = SimpleNamespace(
        compute_logits_local=lambda h: h @ weight,
        language_model=SimpleNamespace(
            lm_head=SimpleNamespace(tp_rank=0, tp_size=1),
            logits_processor=SimpleNamespace(org_vocab_size=VOCAB),
        ),
    )
    h = torch.randn(rows, hidden, device="cuda")
    eager = ta.argmax_rows(model, h)  # compiles the kernels
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = ta.argmax_rows(model, h)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    want = torch.argmax((h @ weight)[:, :VOCAB], -1)
    assert torch.equal(torch.argmax(eager, -1), want)

    static = h.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = ta.argmax_rows(model, static)
    for seed in range(3):
        torch.manual_seed(seed + 1)
        static.copy_(torch.randn(rows, hidden, device="cuda"))
        static[seed, :] = 0.0  # an all-zero row: every column ties
        graph.replay()
        torch.cuda.synchronize()
        want = torch.argmax((static @ weight)[:, :VOCAB], -1)
        assert torch.equal(torch.argmax(replayed, -1), want)
        assert torch.equal(replayed, ta.argmax_rows(model, static))
        assert int(want[seed]) == 0
