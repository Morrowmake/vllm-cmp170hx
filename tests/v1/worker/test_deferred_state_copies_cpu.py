# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute deferred recurrent-state copies on CPU with the Triton interpreter."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_test


def _exercise(model_name, operation, dim_first):
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    from tests.v1.worker.test_mamba_utils import _COPY_FUNCS, _MockCpuGpuBuffer
    from vllm_sm80.models.glm5next.model_state import Glm5NextModelState
    from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.worker.gpu.model_states import mamba_hybrid
    from vllm.v1.worker import mamba_utils

    state_cls = {"glm": Glm5NextModelState, "qwen": Qwen4ExpModelState}[model_name]
    state = object.__new__(state_cls)
    state._align_mode = True
    state._mamba_group_ids = []
    state._mamba_spec = None
    state._mamba_state_copy_funcs = _COPY_FUNCS
    state.max_num_reqs = 8
    state.device = torch.device("cpu")
    conv_shape = (8, 4) if dim_first else (4, 8)
    spec = MambaSpec(
        block_size=32,
        shapes=(conv_shape, (8,)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["recurrent"], spec)],
    )
    conv = torch.arange(32 * 32, dtype=torch.float16).reshape(32, *conv_shape)
    temporal = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    state.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "recurrent": SimpleNamespace(kv_cache=(conv, temporal))
            }
        )
    )
    request_rows = torch.arange(1, 25, dtype=torch.int32).reshape(8, 3)
    # A later microbatch overwrote the same-address gathered table.
    gathered_rows = request_rows.roll(1, dims=0)
    tables = SimpleNamespace(
        block_tables=[SimpleNamespace(gpu=request_rows)],
        input_block_tables=[gathered_rows],
    )
    with (
        patch.object(mamba_hybrid, "CpuGpuBuffer", _MockCpuGpuBuffer),
        patch.object(mamba_utils, "is_conv_state_dim_first", lambda: dim_first),
    ):
        state.initialize_kv_cache(config, tables)
        ctx = state._mamba_ctx
        assert ctx.block_table_ptrs.tolist() == [request_rows.data_ptr()]
        assert ctx.aligned_index_block_table_ptrs.tolist() == [gathered_rows.data_ptr()]
        mapping = torch.tensor([5, -1, 2], dtype=torch.int32)
        before = [conv.clone(), temporal.clone()]
        expected = [tensor.clone() for tensor in before]
        src, dst = (0, 1) if operation == "precopy" else (1, 0)
        for req_idx in (5, 2):
            for out, old in zip(expected, before):
                out[request_rows[req_idx, dst]].copy_(old[request_rows[req_idx, src]])

        def copy():
            if operation == "precopy":
                ctx.run_fused_precopy(
                    3, torch.full((8,), dst, dtype=torch.int32),
                    torch.full((8,), src, dtype=torch.int32),
                    torch.zeros(8, dtype=torch.int32), mapping,
                )
            else:
                ctx.run_fused_postprocess_align(
                    3, torch.ones(8, dtype=torch.int32),
                    torch.full((8,), src, dtype=torch.int32),
                    torch.full((8,), 32, dtype=torch.int32), mapping,
                )

        copy()
        for actual, want in zip((conv, temporal), expected):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
        # Positive control: the old gathered-table binding must corrupt the
        # owner, even though its storage address stays constant across steps.
        for actual, old in zip((conv, temporal), before):
            actual.copy_(old)
        ctx.block_table_ptrs[0] = gathered_rows.data_ptr()
        copy()
        assert any(not torch.equal(a, b) for a, b in zip((conv, temporal), expected))


@pytest.mark.parametrize("model_name", ["glm", "qwen"])
@pytest.mark.parametrize("operation", ["precopy", "postprocess"])
@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
def test_deferred_copies_preserve_request_slot_owner(model_name, operation, dim_first):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-m", __name__, model_name, operation, str(int(dim_first))],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise(sys.argv[1], sys.argv[2], bool(int(sys.argv[3])))
