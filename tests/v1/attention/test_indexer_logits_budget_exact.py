# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prefill logits budget changes how rows are chunked, never what a row
selects.

``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`` bounds the fp32 ``[rows, context]``
logits transient. Lowering it (512 -> 128) makes
``_split_indexer_prefill_chunks`` group fewer requests per chunk and cut more
query sub-chunks. This test runs the prefill indexer's chunk loop end to end
on CPU for two budgets: the real splitter, the real chunk-metadata builder
(``build_prefill_chunk_metadata``, its Triton kernel under
``TRITON_INTERPRET=1``), the real canonical top-k, and the torch reference for
the MQA logits. It asserts that every query row gets bit-identical logits over
its visible window and the same selected pools, with a plan that really
differs.

The data are exactly representable (e4m3 values in {0, +-0.5, +-1, +-2},
power-of-two scales and weights), so every fp32 sum is exact and the result
cannot depend on how a matmul is blocked; what is tested is the bookkeeping:
row coverage, the per-row [ks, ke) bounds, the K gather per request slice and
the reuse of the gathered K by later query sub-chunks. On the GPU the Triton
kernel computes each output element from its own row of Q, its own column of
K and the row's weights, with a fixed [BLOCK_M*H, D] x [D, BLOCK_N] tile shape
and every tile inside [ks, ke) visited, so a row's logits do not depend on
which rows share its chunk either; the prefill top-k launcher picks its
algorithm by row index only past 12,288 rows, above max_num_batched_tokens.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= \
        pytest -q tests/v1/attention/test_indexer_logits_budget_exact.py
"""

import os

import pytest
import torch

from vllm.model_executor.layers.indexer_topk import canonical_topk
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    build_prefill_chunk_metadata,
)
from vllm.v1.attention.ops.triton_mqa_logits import fp8_mqa_logits_torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs CUDA or TRITON_INTERPRET=1 for the chunk-metadata kernel",
)

split = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks

KPOOL = 4
HEADS = 8
HEAD_DIM = 64
TOPK_TOKENS = 64  # select_k = 16 pools per row
SELECT_K = TOPK_TOKENS // KPOOL


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _exact_fp8(shape, gen):
    vals = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
    idx = torch.randint(0, len(vals), shape, generator=gen)
    return vals[idx].to(torch.float8_e4m3fn)


def _pow2(shape, choices, gen):
    vals = torch.tensor(choices, dtype=torch.float32)
    return vals[torch.randint(0, len(vals), shape, generator=gen)]


def _batch(seq_lens, query_lens, seed=0):
    """Prefill requests (context incl. the new tokens, new tokens) and their
    compressed index-K, queries and head weights."""
    gen = torch.Generator().manual_seed(seed)
    k_per_req = [
        (
            _exact_fp8((s // KPOOL, HEAD_DIM), gen),
            _pow2((s // KPOOL,), [0.5, 1.0, 2.0], gen),
        )
        for s in seq_lens
    ]
    total_q = sum(query_lens)
    q = _exact_fp8((total_q, HEADS, HEAD_DIM), gen)
    w = _pow2((total_q, HEADS), [0.25, 0.5, 1.0], gen)
    return k_per_req, q, w


def _run(seq_lens, query_lens, budget_bytes, workspace_rows, data):
    """The prefill loop of ``sparse_attn_indexer_kpool`` with the torch MQA
    reference and canonical top-k. Returns (plan, per-row window logits,
    per-row selected pools)."""
    dev = _device()
    k_per_req, q, w = data
    q, w = q.to(dev), w.to(dev)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    comp_cpu = seq_lens_t // KPOOL
    qlens = torch.tensor(query_lens, dtype=torch.int64)
    qsl_cpu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.cumsum(qlens, 0).to(torch.int32)
    block_table = torch.zeros((len(seq_lens), 1), dtype=torch.int32, device=dev)

    plan = split(comp_cpu.to(torch.int64), qlens, workspace_rows, budget_bytes)
    window_logits: dict[int, torch.Tensor] = {}
    selected: dict[int, torch.Tensor] = {}
    gathered = None
    gathered_for = None
    for req_slice, query_slice in plan:
        md = build_prefill_chunk_metadata(
            req_slice.start,
            req_slice.stop,
            qsl_cpu.to(dev),
            qsl_cpu,
            seq_lens_t.to(dev),
            comp_cpu.to(dev),
            comp_cpu,
            block_table,
            KPOOL,
            query_slice=query_slice,
            skip_kv_gather=query_slice.start > 0,
        )
        if md is None:
            continue
        assert md.total_seq_lens <= workspace_rows
        if not md.skip_kv_gather:
            # cp_gather_indexer_k_quant_cache: the chunk's requests' compressed
            # rows, packed back to back in cu_seq_lens order.
            parts = [k_per_req[r] for r in range(req_slice.start, req_slice.stop)]
            gathered = (
                torch.cat([p[0] for p in parts]).to(dev),
                torch.cat([p[1] for p in parts]).to(dev),
            )
            gathered_for = (req_slice.start, req_slice.stop)
            cu = md.cu_seq_lens.cpu().tolist()
            assert cu[-1] == gathered[0].shape[0]
        else:
            # A later query sub-chunk reuses the K its first sub-chunk gathered.
            assert gathered_for == (req_slice.start, req_slice.stop)
        assert gathered is not None
        k_quant = gathered[0][: md.total_seq_lens]
        k_scale = gathered[1][: md.total_seq_lens]
        rows = slice(md.token_start, md.token_end)
        logits = fp8_mqa_logits_torch(
            q[rows], (k_quant, k_scale), w[rows], md.cu_seqlen_ks, md.cu_seqlen_ke
        )
        assert logits.numel() * 4 <= budget_bytes or logits.shape[0] == 1
        out = torch.empty((logits.shape[0], SELECT_K), dtype=torch.int32, device=dev)
        # canonical_topk asserts k <= row width; a chunk of short requests can
        # be narrower than select_k (the CUDA top-k handles that with its
        # short-window shortcut). Pad with -inf columns no row can see.
        topk_in = logits
        if logits.shape[1] < SELECT_K:
            topk_in = torch.full((logits.shape[0], SELECT_K), float("-inf"), device=dev)
            topk_in[:, : logits.shape[1]] = logits
        canonical_topk(
            topk_in,
            SELECT_K,
            row_starts=md.cu_seqlen_ks,
            row_ends=md.cu_seqlen_ke,
            out=out,
            relative=True,
            identity_when_short=True,
        )
        ks = md.cu_seqlen_ks.cpu().tolist()
        ke = md.cu_seqlen_ke.cpu().tolist()
        for i in range(logits.shape[0]):
            tok = md.token_start + i
            assert tok not in selected, "a query row was processed twice"
            window_logits[tok] = logits[i, ks[i] : ke[i]].cpu().clone()
            selected[tok] = out[i].cpu().clone()
    assert sorted(selected) == list(range(int(qsl_cpu[-1])))
    return plan, window_logits, selected


# (seq_lens incl. new tokens, query_lens): short prompts, a long single
# prompt that needs query sub-chunks, and a mix that regroups requests.
CASES = [
    ([64, 96, 128, 40], [64, 96, 128, 40]),
    ([2048], [256]),
    ([1200, 64, 1600, 300, 88], [200, 64, 180, 300, 88]),
]


@pytest.mark.parametrize("seq_lens,query_lens", CASES)
def test_lower_budget_selects_the_same_pools(seq_lens, query_lens):
    data = _batch(seq_lens, query_lens)
    workspace = 8 * max(seq_lens) // KPOOL
    # "Large" fits the whole step in one chunk; "small" is 1/4 of the largest
    # row set, the same ratio as 512 -> 128 MiB, and forces both request
    # regrouping and query sub-chunks.
    total_n = sum(s // KPOOL for s in seq_lens)
    large = sum(query_lens) * total_n * 4
    small = max(4, large // 16)
    plan_l, logits_l, sel_l = _run(seq_lens, query_lens, large, workspace, data)
    plan_s, logits_s, sel_s = _run(seq_lens, query_lens, small, workspace, data)

    assert len(plan_s) > len(plan_l), "the small budget must change the plan"
    for tok in sel_l:
        assert torch.equal(logits_l[tok], logits_s[tok]), tok
        assert torch.equal(sel_l[tok], sel_s[tok]), tok


def test_deployment_budgets_plan_the_same_rows():
    """At the live shapes (262,144 context, kpool 4, 3,460-token steps, the
    clamped 524,288-row workspace) 128 MiB covers every query row exactly once
    and keeps each logits buffer under 128 MiB."""
    MiB = 1024 * 1024
    workspace = 8 * (262_144 // KPOOL)
    scenarios = [
        ([262_144 // KPOOL], [3460]),
        ([65_536 // KPOOL, 200_000 // KPOOL], [1000, 2460]),
        ([4_000 // KPOOL] * 8, [432] * 8),
    ]
    for comp, qlens in scenarios:
        for budget in (512 * MiB, 128 * MiB):
            plan = split(
                torch.tensor(comp, dtype=torch.int64),
                torch.tensor(qlens, dtype=torch.int64),
                workspace,
                budget,
            )
            rows = {}
            for req_slice, query_slice in plan:
                n = sum(comp[req_slice])
                m = query_slice.stop - query_slice.start
                assert m * n * 4 <= budget or m == 1
                assert n <= workspace
                for r in range(query_slice.start, query_slice.stop):
                    key = (req_slice.start, req_slice.stop, r)
                    assert key not in rows
                    rows[key] = 1
            assert len(rows) == sum(qlens)
