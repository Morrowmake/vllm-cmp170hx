# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream's sparse-MLA decode-correctness test, run on the fork's Triton
sparse-MLA backend (vllm_sm80/attention/triton_mla_sparse.py).

The parameter grid is the upstream test's, minus its backend list. Upstream
skips any backend whose supported block sizes are ``MultipleOf`` entries, so
this test applies the ``MultipleOf`` check itself and then shows the upstream
body the one concrete block size of the case.
"""

import pytest
import torch

if not torch.cuda.is_available():
    # The upstream module initialises CUDA while it is imported.
    pytest.skip("needs a CUDA device", allow_module_level=True)

import tests.v1.attention.test_sparse_mla_backends as upstream  # noqa: E402
from vllm_sm80.attention.triton_mla_sparse import TritonMLASparseBackend  # noqa: E402
from vllm.v1.attention.backend import MultipleOf  # noqa: E402

_UPSTREAM = upstream.test_sparse_backend_decode_correctness


def test_triton_mla_sparse_decode_correctness(
    default_vllm_config,
    dist_init,
    batch_name,
    kv_cache_dtype,
    tensor_parallel_size,
    block_size,
    workspace_init,
    q_scale,
    k_scale,
    total_num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    monkeypatch,
):
    sizes = TritonMLASparseBackend.get_supported_kernel_block_sizes()
    if not any(
        block_size % s.base == 0 if isinstance(s, MultipleOf) else block_size == s
        for s in sizes
    ):
        pytest.skip(f"TRITON_MLA_SPARSE does not support block_size={block_size}")
    monkeypatch.setattr(
        TritonMLASparseBackend,
        "get_supported_kernel_block_sizes",
        staticmethod(lambda kv_cache_spec=None: [block_size]),
    )
    _UPSTREAM(
        default_vllm_config=default_vllm_config,
        dist_init=dist_init,
        backend_cls=TritonMLASparseBackend,
        batch_name=batch_name,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tensor_parallel_size,
        block_size=block_size,
        workspace_init=workspace_init,
        q_scale=q_scale,
        k_scale=k_scale,
        total_num_heads=total_num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        monkeypatch=monkeypatch,
    )


test_triton_mla_sparse_decode_correctness.pytestmark = [
    mark
    for mark in _UPSTREAM.pytestmark
    if not (mark.name == "parametrize" and mark.args[0] == "backend_cls")
]
