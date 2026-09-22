# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests that GLM-5.3-Flash routes its MoE gate in fp32 on sm_80.

Expert selection reads the router logits, so the gate's output dtype is part of
the model's definition, not a performance knob.  The chain that delivers it is:

    text_config.moe_router_dtype == "float32"
      -> _get_moe_router_dtype()            -> torch.float32
      -> Glm5NextMoE.router_dtype           -> GateLinear(out_dtype=...)
      -> GateLinear tier 4                  -> torch.mm(..., out_dtype=fp32)

Tier 4 is the only tier reachable below SM90, and upstream #54048 removed its
architecture gate so that it fires on every CUDA device.  If ``out_dtype`` were
``None`` the gate would drop to the ReplicatedLinear fallback and route on bf16
logits instead, which can change which experts are selected.

No GPU and no weights: the config is read from disk and the layer is built on
CPU against a stubbed single-rank parallel state.

    pytest -q tests/models/glm5next/test_router_logits_dtype.py
"""

import contextlib
import json
import pathlib
from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype

CHECKPOINT = pathlib.Path("/home/ba/models/GLM-5.3-Flash-W4A16-MTP/config.json")
GATE_MODULE = "vllm.model_executor.layers.fused_moe.router.gate_linear"


class FakePlatform:
    """A CUDA device that is neither Hopper nor Blackwell, i.e. sm_80."""

    def __init__(self, cuda=True):
        self._cuda = cuda

    def is_cuda(self):
        return self._cuda

    def is_rocm(self):
        return False

    def is_device_capability(self, capability):
        return False

    def is_device_capability_family(self, family):
        return False


@contextlib.contextmanager
def single_rank(cuda=True):
    """Build a ReplicatedLinear without a distributed environment."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(f"{GATE_MODULE}.current_platform", FakePlatform(cuda))
        )
        for module in (
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.parameter",
        ):
            stack.enter_context(
                mock.patch(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
            )
            with contextlib.suppress(AttributeError):
                stack.enter_context(
                    mock.patch(
                        f"{module}.get_tensor_model_parallel_world_size", lambda: 1
                    )
                )
        yield


# ------------------------------------------------------------ the config


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="deployed checkpoint not present")
def test_deployed_checkpoint_asks_for_fp32_router_logits():
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

    text_config = Glm5NextConfig(**json.loads(CHECKPOINT.read_text())).text_config

    assert text_config.model_type == "glm5_next_text"
    assert text_config.moe_router_dtype == "float32"
    assert _get_moe_router_dtype(text_config) is torch.float32


def test_router_dtype_resolution():
    from types import SimpleNamespace

    assert (
        _get_moe_router_dtype(
            SimpleNamespace(moe_router_dtype="float32", model_type="glm5_next_text")
        )
        is torch.float32
    )
    # Absent field: no fp32 promise. This is the branch GLM must not take.
    assert (
        _get_moe_router_dtype(SimpleNamespace(model_type="glm5_next_text")) is None
    )


# -------------------------------------------------------------- the gate


def test_fp32_out_dtype_reaches_the_cublas_tier_on_a_non_hopper_device():
    with single_rank():
        gate = GateLinear(
            4096, 288, out_dtype=torch.float32, params_dtype=torch.bfloat16
        )

    assert gate.out_dtype is torch.float32
    assert gate.weight.dtype is torch.bfloat16
    # Tiers 1-3 are all SM90+/gfx950 only, so sm_80 must land on tier 4.
    assert gate.allow_specialized_router_gemm is False
    assert gate.allow_fp32_router_gemm is False
    assert gate.allow_bf16x3_router_gemm is False
    assert gate.allow_ll_bf16_gemm is False
    assert gate.allow_cublas_router_gemm is True


def test_without_out_dtype_the_gate_would_route_on_bf16():
    """The negative control: this is what a missing moe_router_dtype costs."""
    with single_rank():
        gate = GateLinear(4096, 288, out_dtype=None, params_dtype=torch.bfloat16)

    assert gate.allow_cublas_router_gemm is False


def test_set_out_dtype_reenables_the_cublas_tier():
    with single_rank():
        gate = GateLinear(4096, 288, out_dtype=None, params_dtype=torch.bfloat16)
        gate.set_out_dtype(torch.float32)

    assert gate.allow_cublas_router_gemm is True


def test_the_cublas_tier_has_no_architecture_gate():
    """#54048: the fp32-out router GEMM must not be Hopper-only. A capability
    term reappearing here would silently demote sm_80 to bf16 routing."""
    import inspect

    src = inspect.getsource(GateLinear.__init__)
    predicate = src[src.index("self._router_gemm_cublas_capable") :]
    predicate = predicate[: predicate.index("self.allow_ll_bf16_gemm")]
    assert "is_hopper" not in predicate
    assert "is_blackwell" not in predicate
    assert "can_use_specialized_kernels" not in predicate
