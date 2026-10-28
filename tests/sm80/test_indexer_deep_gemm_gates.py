# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA indexer's DeepGEMM gates on sm_80.

The vendored DeepGEMM package makes ``has_deep_gemm()`` true on sm_80 although
DeepGEMM cannot run there (``is_deep_gemm_supported()`` is false). Only the
metadata builder must ask ``is_deep_gemm_supported()``: it would otherwise
build DeepGEMM schedule metadata that the Triton paged MQA logits kernel never
reads. The capability helpers require SM90 or SM100 anyway, so on sm_80 they
give the same answer with either predicate.
"""

import inspect

import pytest

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import indexer


def _set_sm80(monkeypatch, has_deep_gemm: bool):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == 8,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: has_deep_gemm)
    monkeypatch.setattr(indexer, "is_deep_gemm_supported", lambda: False)


@pytest.mark.parametrize("has_deep_gemm", [True, False])
def test_capability_helpers_on_sm80_whatever_deep_gemm_reports(
    monkeypatch, has_deep_gemm
):
    _set_sm80(monkeypatch, has_deep_gemm)
    assert not indexer._supports_varlen_paged_mqa_logits()
    assert not indexer._supports_flattened_device_query_lens()
    assert [indexer._supports_native_decode(n) for n in range(1, 9)] == [
        n in (1, 2) for n in range(1, 9)
    ]


def test_every_deep_gemm_predicate_the_module_calls_is_imported():
    # Each name the module calls must be bound at import, not only on the
    # paths a CPU test reaches.
    assert callable(indexer.has_deep_gemm)
    assert callable(indexer.is_deep_gemm_supported)


def test_metadata_builder_asks_whether_deep_gemm_can_run():
    # The one fork edit left in this upstream file: with has_deep_gemm() here,
    # sm_80 would build DeepGEMM schedule metadata on every decode step.
    source = inspect.getsource(indexer.DeepseekV32IndexerMetadataBuilder.build)
    assert "is_deep_gemm_supported()" in source
    assert "has_deep_gemm()" not in source
    assert source.count("get_paged_mqa_logits_metadata(") == 1
