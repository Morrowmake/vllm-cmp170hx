# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GLM5_MEM_ATTRIBUTION: grouping and no-op behaviour (CPU)."""

import torch
from torch import nn

from vllm.v1.worker.gpu import mem_attribution as ma


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.register_buffer("cache", torch.zeros(16), persistent=False)


class _Model(nn.Module):
    def __init__(self, shared: nn.Parameter | None = None):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(3)])
        self.embed = nn.Embedding(10, 8)
        self.head = nn.Linear(8, 10, bias=False)
        self.head.weight = self.embed.weight  # tied
        if shared is not None:
            self.shared = shared


def test_group_key_folds_layer_indices():
    assert ma.group_key("model.layers.12.mlp.experts.w13_weight") == (
        "model.layers.*.mlp.experts"
    )
    assert ma.group_key("layers.0.self_attn.rotary_emb.cos_sin_cache") == (
        "layers.*.self_attn.rotary_emb"
    )
    assert ma.group_key("lm_head.weight") == "lm_head"


def test_inventory_counts_each_storage_once():
    target = _Model()
    drafter = _Model(shared=target.embed.weight)  # drafter shares the embedding
    groups, total = ma.module_inventory(
        [("target", target), ("drafter", drafter)], include_cpu=True
    )
    f32 = 4
    per_model = 3 * (8 * 8 + 16) * f32 + 10 * 8 * f32  # layers + tied embed/head
    # Each model has its own layers and tied embed/head; the drafter's
    # reference to the target embedding (``shared``) is not counted again.
    assert total == 2 * per_model
    assert groups["target:layers.*.proj"] == 3 * 8 * 8 * f32
    assert groups["target:layers.*"] == 3 * 16 * f32
    assert "drafter:shared" not in groups
    assert "target:head" not in groups  # tied to target:embed


def test_cpu_tensors_are_skipped_by_default():
    groups, total = ma.module_inventory([("target", _Model())])
    assert total == 0 and groups == {}


def test_disabled_hooks_do_nothing(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_MEM_ATTRIBUTION", "0")
    stages = ma.ProfileStages()
    stages.stage("x")
    stages.finish()
    assert stages.stages == []
    ma.log_load_inventory([("target", _Model())], 0)
    ma.log_kv_accounting(0, None, 0, 0, 0)
    with ma.AfterKvAllocations("serving KV"):
        pass
