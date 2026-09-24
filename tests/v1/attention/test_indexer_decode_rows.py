# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GLM5_INDEXER_DECODE_ROWS: the indexer's decode block-table buffers.

``expanded_block_table_buffer`` and ``indexer_decode_block_table_buffer`` hold
one row per decode token, but were sized by ``max_num_batched_tokens``. A
decode request has at most ``decode_threshold = next_n`` query tokens and a
step at most ``max_num_seqs`` requests, so the flag sizes them to
``max_num_seqs * next_n`` rows (at 262,144 context, 3,460-token steps and 8 x 4
decode rows: 54 + 27 MiB -> 0.5 + 0.25 MiB per rank).

These tests build real decode metadata on CPU (Triton kernels under
``TRITON_INTERPRET=1``) with the flag off and on and require identical
metadata, including a batch that needs more rows than the narrow sizing, which
must grow the buffers and keep the old ones alive.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= \
        pytest -q tests/v1/attention/test_indexer_decode_rows.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.indexer as indexer_mod
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs CUDA or TRITON_INTERPRET=1 for the metadata kernels",
)

MAX_MODEL_LEN = 4096
NEXT_N = 4  # 3 speculative tokens
KERNEL_BLOCK = 64


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _vllm_config(max_num_seqs, max_num_batched_tokens):
    from tests.v1.attention.utils import create_vllm_config

    try:
        vc = create_vllm_config(
            model_name="Qwen/Qwen3-0.6B",
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no local model config for the test VllmConfig: {e}")
    object.__setattr__(
        vc,
        "speculative_config",
        SimpleNamespace(
            num_speculative_tokens=NEXT_N - 1, enable_adaptive_verification=False
        ),
    )
    return vc


def _builder(
    monkeypatch,
    flag,
    *,
    max_num_seqs=8,
    max_num_batched_tokens=512,
    use_flattening=None,
):
    monkeypatch.setenv("VLLM_GLM5_INDEXER_DECODE_ROWS", flag)
    if not torch.cuda.is_available():
        monkeypatch.setattr(indexer_mod, "num_compute_units", lambda _i=None: 70)
    vc = _vllm_config(max_num_seqs, max_num_batched_tokens)
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    width = get_block_table_width(
        spec.max_num_blocks_per_req(vc, MAX_MODEL_LEN), spec.block_size
    )
    b = indexer_mod.DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer"],
        vllm_config=vc,
        device=_device(),
        block_table_width=width,
    )
    b.kernel_block_size = KERNEL_BLOCK
    if use_flattening is not None:
        b.use_flattening = use_flattening
    return b, width


def _common(query_lens, seq_lens, width):
    dev = _device()
    n = len(query_lens)
    ql = torch.tensor(query_lens, dtype=torch.int32)
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(ql, 0)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    bt = (torch.arange(n * width, dtype=torch.int32) % 997).view(n, width)
    ntok = int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(dev),
        query_start_loc_cpu=qsl,
        seq_lens=seq.to(dev),
        seq_lens_cpu_upper_bound=seq,
        num_reqs=n,
        num_actual_tokens=ntok,
        max_query_len=int(ql.max()),
        max_seq_len=int(seq.max()),
        block_table_tensor=bt.to(dev),
        slot_mapping=torch.zeros(ntok, dtype=torch.int64, device=dev),
        causal=True,
    )


def _decode_fields(md):
    d = md.decode
    assert d is not None
    out = {}
    for name, value in vars(d).items():
        if name == "schedule_metadata":
            # torch.empty scratch that only the DeepGEMM path fills.
            continue
        if isinstance(value, torch.Tensor):
            out[name] = value.detach().cpu().clone()
        elif not hasattr(value, "__dict__"):
            out[name] = value
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].shape == b[k].shape, k
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


BATCHES = [
    # 8 spec decodes of next_n tokens: the largest decode step.
    ([4] * 8, [300, 1000, 17, 4000, 2500, 256, 511, 64]),
    # mixed decode lengths (partial acceptance) then a prefill.
    ([4, 1, 3, 2, 200], [900, 40, 1500, 77, 700]),
    # plain decodes.
    ([1, 1, 1], [5, 3000, 128]),
]


@pytest.mark.parametrize("use_flattening", [True, False])
@pytest.mark.parametrize("query_lens,seq_lens", BATCHES)
def test_narrow_rows_give_identical_decode_metadata(
    monkeypatch, use_flattening, query_lens, seq_lens
):
    wide, width = _builder(monkeypatch, "0", use_flattening=use_flattening)
    ref = _decode_fields(wide.build(0, _common(query_lens, seq_lens, width)))

    narrow, _ = _builder(monkeypatch, "1", use_flattening=use_flattening)
    assert narrow.expanded_block_table_buffer.shape[0] == 8 * NEXT_N
    assert wide.expanded_block_table_buffer.shape[0] == 512
    got = _decode_fields(narrow.build(0, _common(query_lens, seq_lens, width)))
    _assert_same(ref, got)
    assert narrow._retired_decode_buffers == []
    if narrow.indexer_decode_block_table_buffer is not None:
        assert narrow.indexer_decode_block_table_buffer.shape[0] == 8 * NEXT_N


def test_oversized_decode_batch_grows_and_keeps_old_buffers(monkeypatch):
    """max_num_seqs=2 sizes 8 rows; an 8-request decode step needs 32. The
    builder must grow to the full size, keep the replaced buffers referenced
    and still produce the reference metadata."""
    query_lens, seq_lens = BATCHES[0]
    wide, width = _builder(monkeypatch, "0", max_num_seqs=2)
    ref = _decode_fields(wide.build(0, _common(query_lens, seq_lens, width)))

    narrow, _ = _builder(monkeypatch, "1", max_num_seqs=2)
    # Prime the lazy compressed buffer at the narrow size first.
    narrow.build(0, _common([4, 4], [300, 1000], width))
    old_expanded = narrow.expanded_block_table_buffer
    old_compressed = narrow.indexer_decode_block_table_buffer
    assert old_expanded.shape[0] == 2 * NEXT_N
    got = _decode_fields(narrow.build(0, _common(query_lens, seq_lens, width)))
    _assert_same(ref, got)
    assert narrow.expanded_block_table_buffer.shape[0] == 512
    assert any(t is old_expanded for t in narrow._retired_decode_buffers)
    if old_compressed is not None:
        assert any(t is old_compressed for t in narrow._retired_decode_buffers)


def test_deployment_sizes():
    """Bytes freed per rank at the live shapes: W(L) = cdiv(L, 1152) * 18."""
    MiB = 1024 * 1024
    width = -(-262_144 // 1152) * 18
    assert width == 4104
    for batched, c2_mib in ((2312, 36.2), (3460, 54.2)):
        c2 = batched * width * 4
        c3 = batched * (width // 2) * 4
        assert round(c2 / MiB, 1) == c2_mib
        narrow = 8 * NEXT_N * (width + width // 2) * 4
        assert narrow < MiB
        assert c2 + c3 - narrow > 50 * MiB
