# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real bits harness's oracle/verdict with CPU transport fixtures."""
import copy
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from .exact_round import exact_sum, reference_round
from . import wo_b_allreduce_bits as bits


def _run(monkeypatch, tmp_path, values, bad_rank=None):
    import vllm.config as config
    import vllm.distributed as distributed
    import vllm.distributed.parallel_state as parallel
    import vllm_sm80.models.deepseek_v41.sm80_attention as attention

    parts = torch.tensor(values, dtype=torch.bfloat16)[:, None, None].expand(4, 1, 8)
    good = reference_round(exact_sum(parts), torch.bfloat16)
    out = tmp_path / "bits.json"
    monkeypatch.setattr(sys, "argv", ["bits", "--out", str(out), "--seeds", "1"])
    for key, value in {"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0",
                       "VLLM_SM80_ALLREDUCE_ROUND_ONCE": "1"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(bits, "ROWS", [1])
    monkeypatch.setattr(bits, "HIDDEN", 8)
    monkeypatch.setattr(bits, "_engine_like_setup", lambda: None)
    monkeypatch.setattr(bits, "_refuse_unless_served", lambda _: None)
    monkeypatch.setattr(bits, "_backend", lambda *_: "nccl")
    monkeypatch.setattr(bits, "_time", lambda *_: 0.0)
    monkeypatch.setattr(config, "VllmConfig", lambda: None)
    monkeypatch.setattr(config, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(distributed, "init_distributed_environment", lambda **_: None)
    monkeypatch.setattr(distributed, "ensure_model_parallel_initialized", lambda *_: None)
    monkeypatch.setattr(parallel, "get_tp_group", lambda: SimpleNamespace(device_communicator=None))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    device = torch.device
    monkeypatch.setattr(torch, "device", lambda *args: device("cpu"))
    monkeypatch.setattr(torch, "randint", lambda lo, hi, shape, **_: torch.zeros(shape, dtype=torch.int64))
    monkeypatch.setattr(torch, "randn", lambda *shape, **_: parts[0].float().clone())
    monkeypatch.setattr(distributed, "tensor_model_parallel_all_reduce",
                        lambda t: parts.float().sum(0).to(t.dtype))
    monkeypatch.setattr(attention, "wo_b_all_reduce", lambda *_: good.clone())
    monkeypatch.setattr(attention, "custom_all_reduce_takes", lambda _: False)
    monkeypatch.setattr(attention, "custom_all_reduce_pieces", lambda _: [1])

    def gather(outputs, value):
        for rank, output in enumerate(outputs):
            if value.dtype == torch.uint8:
                output.copy_(good.view(torch.uint8))
                if rank == bad_rank:
                    output.flatten()[0] += 1
            else:
                output.copy_(parts[rank])

    def gather_objects(outputs, value):
        for rank in range(len(outputs)):
            outputs[rank] = copy.deepcopy(value)
            for case in outputs[rank]:
                case["rank"] = rank
    monkeypatch.setattr(bits.dist, "all_gather", gather)
    monkeypatch.setattr(bits.dist, "all_gather_object", gather_objects)
    monkeypatch.setattr(bits.dist, "all_reduce", lambda *_, **__: None)
    monkeypatch.setattr(bits.dist, "barrier", lambda: None)
    monkeypatch.setattr(bits.dist, "destroy_process_group", lambda: None)
    bits.main()
    return json.loads(out.read_text())


def test_midpoint_oracle_does_not_double_round(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path, [1, 2.0**-8, 2.0**-30, 0])
    assert data["cases"][0]["switched_ne_exact"] == 0
    assert data["switched_no_less_accurate"] is True


def test_wrong_switched_bits_on_nonzero_rank_fail(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path, [1, 1, 1, 1], bad_rank=3)
    assert data["switched_no_less_accurate"] is False
