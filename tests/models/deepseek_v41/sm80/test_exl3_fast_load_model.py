# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_EXL3_FAST_LOAD loads the same bytes: the sm_80 DeepSeek-V4.1 target
and DSpark drafter built with CPU parameters (the meta construction of
``construct`` with the platform's device set to the CPU) from the tiny
checkpoint of ``exl3_fixture``, loaded end to end by ``load_exl3_weights``
with the switch off and on, for every TP4 rank with whole-expert EP4 and
with tensor-split routed experts: every parameter's bytes are equal.
"""

import pytest
import torch

from tests.models.deepseek_v41.sm80 import construct
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)
from vllm.config import set_current_vllm_config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def _load(monkeypatch, checkpoint, rank, ep, fast):
    """{(model, parameter name): CPU copy} after load_exl3_weights."""
    from vllm_sm80.kernels.exl3 import loader

    monkeypatch.setenv("VLLM_EXL3_FAST_LOAD", "1" if fast else "0")
    construct.set_groups(monkeypatch, 4, rank)
    construct.stub_engram_tables(monkeypatch, 4)
    target_config, draft_config = construct.vllm_configs(str(checkpoint), 4)
    # The target and the drafter share the parallel config.
    target_config.parallel_config.enable_expert_parallel = ep
    params = {}
    for role, config in (("target", target_config), ("draft", draft_config)):
        model = construct.build_model(config)
        assert all(p.device.type == "cpu" for p in model.parameters())
        with set_current_vllm_config(config):
            loader.load_exl3_weights(model, config.model_config)
        params.update(
            {(role, name): p.detach().clone() for name, p in model.named_parameters()}
        )
    return params


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("ep", [True, False], ids=["ep4", "split"])
def test_fast_load_loads_the_same_bytes(
    monkeypatch,
    meta_platform,  # noqa: F811
    checkpoint,
    ep,
    rank,
):
    from vllm.platforms import current_platform
    from vllm_sm80 import plugin
    from vllm_sm80.kernels.exl3 import loader, ops

    platform_cls = type(current_platform)
    monkeypatch.setattr(platform_cls, "device_type", "cpu")
    monkeypatch.setattr(
        platform_cls,
        "current_device",
        staticmethod(lambda: torch.device("cpu")),
        raising=False,
    )
    plugin._register_exl3()
    monkeypatch.setattr(ops, "ensure_extension", lambda: None)
    monkeypatch.setattr(loader, "require_cuda_parameters", lambda model: None)
    base = _load(monkeypatch, checkpoint, rank, ep, fast=False)
    fast = _load(monkeypatch, checkpoint, rank, ep, fast=True)
    assert list(base) == list(fast)
    for key, tensor in base.items():
        assert torch.equal(
            tensor.view(-1).view(torch.uint8), fast[key].view(-1).view(torch.uint8)
        ), key
