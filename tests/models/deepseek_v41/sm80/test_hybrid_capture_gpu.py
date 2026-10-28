# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only proof of replay-fresh consumer snapshots; run under a GPU lease."""

import pytest
import torch

from vllm_sm80.models.deepseek_v41 import hybrid_capture as hybrid
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.capture import Record

from .test_hybrid_capture import attention_args, index_args

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("op", ["index_scores", "sparse_attention"])
def test_hybrid_deferred_reader_graph_replay_refreshes_metadata_and_cache(op):
    device = "cuda"
    if op == "index_scores":
        args = index_args(device)
        args.update(
            q=torch.arange(4 * 32 * 128, device=device).reshape(4, 32, 128).to(torch.bfloat16) / 256,
            w=torch.full((4, 32), 0.03125, device=device),
            idxk=torch.arange(6 * 4 * 128, device=device).reshape(6, 4, 128).to(torch.bfloat16) / 4096,
            heads_per_rank=8,
        )
        run = core.index_scores
        table_name, cache_names = "block_table", ("idxk",)
    else:
        args = attention_args(device=device)
        cache = torch.arange(6 * 4 * 512, device=device).reshape(6, 4, 512).to(torch.bfloat16) / 16384
        indices = torch.full((2, 128), -1, dtype=torch.int32, device=device)
        indices[:, :4] = args["swa_indices"]
        local = torch.full((2, 512), -1, dtype=torch.int32, device=device)
        local[:, :4] = args["comp_local"]
        args.update(
            q=torch.arange(2 * 16 * 512, device=device).reshape(2, 16, 512).to(torch.bfloat16) / 8192,
            swa_cache=cache, swa_indices=indices, comp_cache=cache.clone() + 1,
            comp_local=local, sink=torch.zeros(16, device=device), scale=0.25,
        )
        run = core.sparse_attention
        table_name, cache_names = "comp_block_table", ("swa_cache", "comp_cache")
    run(**args)  # Compile the actual consumer before graph capture.
    torch.cuda.synchronize()
    record = Record("hybrid", "hybrid_linear_consumer", "layers.8.attn", args["q"].shape[0], {}, in_graph=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hybrid._defer_reader(record, op, args)
        record.put(out=run(**args))
    pointers = {name: value.data_ptr() for name, value in record.tensors.items() if isinstance(value, torch.Tensor)}
    previous = None
    for iteration in range(3):
        args[table_name][:, 0] = 2 + iteration % 2
        args["q"].add_(0.125)
        for name in cache_names:
            args[name].add_(0.25)
        if op == "index_scores":
            args["n_vis"].copy_(torch.tensor([4 + iteration, 3, 7, 0], device=device))
        else:
            args["swa_indices"][0, 0] = 8 + 4 * (iteration % 2)
            args["comp_local"][0, 0] = iteration
        expected = run(**args)
        graph.replay()
        torch.cuda.synchronize()
        saved = record.materialize()
        observed = saved.pop("out")
        assert torch.equal(observed, expected.cpu())
        # The remapped paged inputs must reproduce the real GPU consumer,
        # not just satisfy a size or serialization assertion.
        replay = {name: value.to(device) if isinstance(value, torch.Tensor) else value
                  for name, value in saved.items()}
        assert torch.equal(run(**replay).cpu(), observed)
        assert all(record.tensors[name].data_ptr() == pointer for name, pointer in pointers.items())
        assert all(saved[name].shape[0] <= args[name].shape[0] for name in cache_names)
        if previous is not None:
            assert not torch.equal(observed, previous)
        previous = observed
