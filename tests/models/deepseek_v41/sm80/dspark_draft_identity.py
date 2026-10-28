# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual sequential draft IDs on identical fixed logits, before target verification.

The GPU entry uses the served TP communicator and both eager and graph replay.
The CPU entry uses Gloo and the Triton interpreter. These are synthetic sampling
inputs, not captured checkpoint trajectories or an acceptance-rate proxy.
"""
import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

SWITCH = "VLLM_SM80_SHARD_ARGMAX"
CASES = ("cross_rank_ties", "nan_ties", "signed_zero", "sequential_markov")
SHAPES = ((1, 5), (4, 3))


def assert_identity(records, world, modes):
    """Gate every actual draft ID, repeat, rank and arm; never target output."""
    indexed = {}
    for row in records:
        key = (row["rank"], row["case"], row["requests"], row["steps"], row["mode"], row["on"])
        assert key not in indexed, ("duplicate draft case", key)
        indexed[key] = row
    expected = {(r, c, n, k, mode, on) for r in range(world) for c in CASES
                for n, k in SHAPES for mode in modes for on in (False, True)}
    assert set(indexed) == expected, "incomplete rank/arm/draft fixture coverage"
    for key, row in indexed.items():
        ids, confidence = row["draft_ids"], row["confidence"]
        assert len(ids) >= 2 and len(confidence) == len(ids), ("missing draft repeats", key)
        assert all(v == ids[0] for v in ids), ("draft repeatability", key)
        assert all(v == confidence[0] for v in confidence), ("confidence repeatability", key)
        n, k = row["requests"], row["steps"]
        assert len(ids[0]) == n and all(len(v) == k for v in ids[0]), ("draft shape", key)
        ref = indexed[(0, row["case"], n, k, "eager", False)]
        assert ids[0] == ref["draft_ids"][0], ("draft IDs differ", key)
        assert confidence[0] == ref["confidence"][0], ("draft confidence differs", key)
        if row["expected_tie_id"] is not None:
            assert all(t == row["expected_tie_id"] for req in ids[0] for t in req), ("lowest-index tie", key)


def _fixture(rank, world, device, case, requests, steps, small, gather):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    width = 25 if small else 32320
    vocab = width * world - (3 if small else 0)
    count = requests * steps
    g = torch.Generator().manual_seed(641)
    base = torch.randint(-3, 4, (count + 4, vocab), generator=g).float()
    tie_id = None
    if case in ("cross_rank_ties", "nan_ties"):
        base.zero_()
        ties = [width - 1, width, 2 * width + 1, 3 * width + 2]
        base[:, ties] = 7.0 if case == "cross_rank_ties" else float("nan")
        tie_id = ties[0]
    elif case == "signed_zero":
        base.zero_()
        base[:, ::2] = -0.0
        tie_id = 0
    # Invalid padded vocabulary columns must never win the local argmax.
    padded = torch.full((count + 4, width * world), float("nan"))
    padded[:, :vocab] = base
    local = padded[:, rank * width:(rank + 1) * width].contiguous().to(device)
    columns = torch.arange(vocab, device=device)

    def local_logits(hidden):
        return local[hidden[:, 0].long()]

    def markov_bias(previous):
        if case == "sequential_markov":
            return ((previous[:, 0, None].long() + columns[None, :]) % 7 - 3).float()
        return torch.zeros((previous.shape[0], vocab), device=device)

    model = SimpleNamespace(
        lm_head=SimpleNamespace(tp_rank=rank, tp_size=world),
        logits_processor=SimpleNamespace(org_vocab_size=vocab),
        draft_id_to_target_id=None,
        compute_logits_local=local_logits,
        compute_draft_logits=lambda hidden: gather(local_logits(hidden))[:, :vocab],
        markov_embed=lambda ids: ids[:, None].float(), markov_bias=markov_bias,
        map_draft_to_target=lambda ids: ids,
        compute_confidence=lambda hidden, previous: hidden[:, 0] * 1000 + previous[:, 0],
    )
    hidden = torch.arange(count + 4, dtype=torch.float32, device=device)[:, None]
    indices = torch.randperm(count + 4, generator=g)[:count].to(device)
    input_ids = torch.randint(0, vocab, (count,), generator=g).to(device)

    def make(on):
        os.environ[SWITCH] = str(int(on))
        s = object.__new__(DSparkSpeculator)
        s.model = model
        s.num_speculative_steps = steps
        s.sample_indices = indices.clone()
        s.input_buffers = SimpleNamespace(input_ids=input_ids.clone())
        s._anchor_idx = torch.arange(requests, device=device) * steps
        s.sample_idx_mapping = torch.arange(count, device=device)
        s.sample_pos = torch.arange(count, device=device) + 1
        s._step_cols = torch.arange(steps, dtype=torch.int32, device=device)
        s.draft_tokens = torch.full((requests, steps), -1, dtype=torch.int64, device=device)
        s.draft_token_confidence_probs = torch.zeros((requests, steps), device=device)
        s.draft_logits = None
        s._draft_topk = None
        s.acceptance_estimator = None
        s.use_confidence_head = True
        return s
    return make, hidden, tie_id


def run_cases(rank, world, device, cpu):
    from vllm_sm80.spec_decode import shard_argmax as sa

    if cpu:
        sa.BLOCK = 8
        def reduce(slots):
            fp32 = slots.float()
            dist.all_reduce(fp32)
            return fp32.to(slots.dtype)
        sa.tensor_model_parallel_all_reduce = reduce
        def gather(local):
            pieces = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(pieces, local)
            return torch.cat(pieces, dim=-1)
    else:
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.distributed.parallel_state import graph_capture
        gather = tensor_model_parallel_all_gather
    records = []
    for case in CASES:
        for requests, steps in SHAPES:
            make, hidden, tie_id = _fixture(rank, world, device, case, requests, steps, cpu, gather)
            for on in (False, True):
                s = make(on)
                s._sample_sequential(requests, hidden)
                assert s._shard_argmax is on, "draft switch did not select the requested arm"
                for mode in (("eager",) if cpu else ("eager", "graph")):
                    if mode == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with graph_capture(device) as context:
                            s._sample_sequential(requests, hidden)
                            context.stream.synchronize()
                            with torch.cuda.graph(graph, stream=context.stream):
                                s._sample_sequential(requests, hidden)
                        replay = graph.replay
                    else:
                        replay = lambda: s._sample_sequential(requests, hidden)
                    ids, confidence = [], []
                    for _ in range(2):
                        s.draft_tokens.fill_(-1)
                        replay()
                        ids.append(s.draft_tokens.cpu().tolist())
                        confidence.append(s.draft_token_confidence_probs.cpu().tolist())
                    records.append(dict(rank=rank, case=case, requests=requests, steps=steps,
                                        mode=mode, on=on, draft_ids=ids, confidence=confidence,
                                        expected_tie_id=tie_id))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 4
    if args.cpu:
        assert os.environ.get("TRITON_INTERPRET") == "1"
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        from tests.models.deepseek_v41.sm80.wo_b_allreduce_bits import _engine_like_setup, _refuse_unless_served
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.distributed.parallel_state import get_tp_group
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        _engine_like_setup()
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
                                         distributed_init_method="env://", backend="nccl")
            ensure_model_parallel_initialized(world, 1)
        assert _refuse_unless_served(get_tp_group().device_communicator) is None
    records = run_cases(rank, world, device, args.cpu)
    all_records = [None] * world
    dist.all_gather_object(all_records, records)
    records = [row for rows in all_records for row in rows]
    modes = ("eager",) if args.cpu else ("eager", "graph")
    assert_identity(records, world, modes)
    if rank == 0:
        with open(args.out, "w") as file:
            json.dump(dict(verdict="PASS", world=world, modes=modes, records=records), file, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
