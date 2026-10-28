# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU checks of the sm_80 DeepSeek-V4.1 attention kernels (skip without CUDA).

* index scores (Triton) equal the torch definition bit for bit: FP4-qdq'd
  queries and keys make every dot product exact in fp32, so only the
  rounding points matter and those are the same;
* the top-k on CUDA (``torch.topk`` + the shared tie fix) equals the stable
  CPU definition on tie-heavy scores;
* sparse attention (Triton) matches the torch definition to BF16 rounding;
* both kernels address rows past 2**31 elements correctly (int64 offsets);
* the whole layer on CUDA passes the same reference comparison as on CPU.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_kernels_gpu.py
"""

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops.sm80_act_quant import fp4_qdq

from . import test_attn_layers as layers
from .dsv41_ref import LayerSpec, RefArgs, make_weights, ref_sparse_attn

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

BF16 = torch.bfloat16


def _index_case(g: torch.Generator, n: int = 37, blocks: int = 60, entries: int = 64):
    idxk = fp4_qdq(torch.randn(blocks, entries, 128, generator=g), 32).to(BF16)
    q = fp4_qdq(torch.randn(n, 32, 128, generator=g) * 2, 32).to(BF16)
    w = (torch.randn(n, 32, generator=g) / 8).to(BF16)
    reqs = 3
    table = torch.stack([torch.randperm(blocks, generator=g) for _ in range(reqs)])
    req = torch.randint(0, reqs, (n,), generator=g)
    n_vis = torch.randint(0, blocks * entries, (n,), generator=g)
    n_vis[:3] = torch.tensor([0, 1, blocks * entries])
    return q, w, n_vis, req, idxk, table.to(torch.int32)


@pytest.mark.parametrize("heads_per_rank", [8, 32])
def test_index_scores_triton_bitwise(heads_per_rank: int) -> None:
    from vllm_sm80.models.deepseek_v41.ops.sm80_index_scores import (
        index_scores_triton,
    )

    g = torch.Generator().manual_seed(1)
    q, w, n_vis, req, idxk, table = _index_case(g)
    n_cols = int(n_vis.max())
    want = core.index_scores_torch(q, w, n_vis, req, idxk, table, n_cols, heads_per_rank)
    got = index_scores_triton(
        q.cuda(), w.cuda(), n_vis.cuda(), req.cuda(), idxk.cuda(), table.cuda(),
        n_cols, heads_per_rank,
    ).cpu()
    assert torch.equal(got, want)


def test_topk_cuda_canonical_on_ties() -> None:
    g = torch.Generator().manual_seed(2)
    n, cols = 64, 3000
    # Few distinct values: the k-th score is tied across many positions.
    scores = torch.randint(-4, 5, (n, cols), generator=g).float() / 4
    n_vis = torch.randint(0, cols + 1, (n,), generator=g)
    n_vis[:2] = torch.tensor([cols, 5])
    scores[3, :100] = -core.FLT_MAX  # masked-but-visible positions
    for k in (512, 24):
        want = core.topk_canonical(scores, n_vis, k)
        got = core.topk_canonical_cuda(scores.cuda(), n_vis.cuda(), k).cpu()
        assert torch.equal(got, want), k


def _attn_case(g: torch.Generator, n: int = 29, h: int = 16):
    swa = torch.randn(40, 32, 512, generator=g).to(BF16)
    comp = torch.randn(50, 64, 512, generator=g).to(BF16)
    q = (torch.randn(n, h, 512, generator=g) * 0.3).to(BF16)
    swa_idx = torch.randint(0, 40 * 32, (n, 128), generator=g).int()
    swa_idx[:, 100:] = -1
    swa_idx[0] = -1  # no valid row at all
    table = torch.stack([torch.randperm(50, generator=g) for _ in range(2)]).int()
    req = torch.randint(0, 2, (n,), generator=g)
    local = torch.randint(0, 50 * 64, (n, 512), generator=g).int()
    local[:, 400:] = -1
    local[0] = -1
    sink = torch.randn(h, generator=g)
    return q, swa, swa_idx, comp, table, local, req, sink


@pytest.mark.parametrize("with_comp", [True, False])
def test_sparse_attention_triton(with_comp: bool) -> None:
    from vllm_sm80.models.deepseek_v41.ops.sm80_sparse_attn import (
        sparse_attention_triton,
    )

    g = torch.Generator().manual_seed(3)
    q, swa, swa_idx, comp, table, local, req, sink = _attn_case(g)
    if not with_comp:
        local = None
    want = core.sparse_attention_torch(
        q, swa, swa_idx, comp, table, local, req, sink, 512**-0.5
    )
    cu = lambda t: None if t is None else t.cuda()  # noqa: E731
    got = sparse_attention_triton(
        cu(q), cu(swa), cu(swa_idx), cu(comp), cu(table), cu(local), cu(req),
        cu(sink), 512**-0.5,
    ).cpu()
    assert not got[0].any(), "a query without rows gives zeros"
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-3)


def _huge_view(rows: int, entries: int, dim: int, block_stride: int):
    """A [3, entries, dim] cache view whose block 2 starts past 2**31 elements."""
    need = 2 * block_stride + entries * dim
    free, _ = torch.cuda.mem_get_info()
    if free < need * 2 + (8 << 30):
        pytest.skip("not enough free GPU memory for a >2**31-element cache")
    storage = torch.zeros(need, dtype=BF16, device="cuda")
    return torch.as_strided(storage, (3, entries, dim), (block_stride, dim, 1))


def test_kernels_int64_offsets() -> None:
    from vllm_sm80.models.deepseek_v41.ops.sm80_index_scores import (
        index_scores_triton,
    )
    from vllm_sm80.models.deepseek_v41.ops.sm80_sparse_attn import (
        sparse_attention_triton,
    )

    g = torch.Generator().manual_seed(4)
    stride = 2**30 + 512
    # Index keys: block table points at physical block 2 (offset > 2**31).
    idxk = _huge_view(3, 4, 128, stride)
    keys = fp4_qdq(torch.randn(3, 4, 128, generator=g), 32).to(BF16).cuda()
    idxk[:] = keys
    q = fp4_qdq(torch.randn(2, 32, 128, generator=g), 32).to(BF16).cuda()
    w = (torch.randn(2, 32, generator=g) / 8).to(BF16).cuda()
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda")
    n_vis = torch.tensor([12, 5], device="cuda")
    req = torch.zeros(2, dtype=torch.int64, device="cuda")
    got = index_scores_triton(q, w, n_vis, req, idxk, table, 12, 8)
    want = core.index_scores_torch(q, w, n_vis, req, idxk, table, 12, 8)
    assert torch.equal(got, want)
    del idxk
    torch.cuda.empty_cache()

    swa = _huge_view(3, 4, 512, stride)
    swa[:] = torch.randn(3, 4, 512, generator=g).to(BF16).cuda()
    qa = (torch.randn(2, 16, 512, generator=g) * 0.3).to(BF16).cuda()
    swa_idx = torch.tensor([[8, 9, 10, 11], [8, 1, -1, -1]], dtype=torch.int32).cuda()
    sink = torch.randn(16, generator=g).cuda()
    got = sparse_attention_triton(
        qa, swa, swa_idx, None, None, None, req, sink, 512**-0.5
    )
    want = core.sparse_attention_torch(
        qa, swa, swa_idx, None, None, None, req, sink, 512**-0.5
    )
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-3)

def _gather_rows(cfg, caches, batch):
    """The rows each token attends to (window slots, then selected
    compressed positions) and their validity: [T, K, D], [T, K]."""
    T = batch.num_tokens
    blk, off = core._paged(caches.swa, batch.swa_indices.clamp_min(0))
    rows, valid = [caches.swa[blk, off]], [batch.swa_indices >= 0]
    if cfg.compress_ratio:
        local = caches.topk[:T]
        req = batch.token_to_req.to(torch.int64)[:, None].expand_as(local)
        rows.append(
            core.gather_compressed_rows(
                caches.comp, batch.comp_block_table, req, local.clamp_min(0)
            )
        )
        valid.append(local >= 0)
    return torch.cat(rows, dim=1), torch.cat(valid, dim=1)


def _fp64_attention(q, rows, valid, sink, scale):
    """Sink attention in FP64: softmax over the valid rows with the sink in
    the denominator; a token without valid rows gets zeros."""
    s = torch.einsum("thd,tkd->thk", q.double(), rows.double()) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    sink = sink.double()[None, : q.shape[1]]
    m = torch.maximum(s.amax(-1), sink)
    p = torch.exp(s - m[..., None])
    denom = p.sum(-1) + torch.exp(sink - m)
    return torch.einsum("thk,tkd->thd", p, rows.double()) / denom[..., None]


def _inverse_rope_fp64(o, positions, freqs):
    """The output's inverse RoPE in FP64 (same fp32 angles as the layer)."""
    rd = 2 * freqs.numel()
    angles = (positions.to(torch.float32)[:, None] * freqs).double()
    cis = torch.polar(torch.ones_like(angles), angles).conj()[:, None, :]
    out = o.double().clone()
    tail = torch.view_as_complex(out[..., -rd:].contiguous().unflatten(-1, (-1, 2)))
    out[..., -rd:] = torch.view_as_real(tail * cis).flatten(-2)
    return out


class _Fp64Errors:
    """Pooled |x - fp64| of every attend_layer call (all layers, steps and
    tokens of the test, padding rows excluded), measured BEFORE the FP8 qdq:
    on the BF16 attention output after its inverse RoPE, the tensor the qdq
    takes. ``sm80`` is the layer's own CUDA path (Triton sparse attention);
    ``reference`` is the reference kernel's arithmetic (``ref_sparse_attn``,
    torch on CUDA) on the same q, rows, selections and sink."""

    def __init__(self):
        self.sum = {"sm80": 0.0, "reference": 0.0}
        self.max = {"sm80": 0.0, "reference": 0.0}
        self.count = 0

    def mean(self, name: str) -> float:
        return self.sum[name] / max(self.count, 1)

    def measure(self, cfg, weights, caches, batch, q) -> None:
        T = batch.num_tokens
        if T == 0:
            return
        q = q[:T]
        pos, freqs = batch.positions, weights.freqs
        rows, valid = _gather_rows(cfg, caches, batch)
        exact = _inverse_rope_fp64(
            _fp64_attention(q, rows, valid, weights.attn_sink, cfg.softmax_scale),
            pos,
            freqs,
        )
        sm80 = core.sparse_attention(
            q, caches.swa, batch.swa_indices, caches.comp, batch.comp_block_table,
            caches.topk[:T] if cfg.compress_ratio else None,
            batch.token_to_req, weights.attn_sink, cfg.softmax_scale,
        )
        idx = torch.where(
            valid, torch.arange(valid.shape[1], device=valid.device), -1
        ).int()[:, None, :]
        with torch.device(q.device):
            ref = ref_sparse_attn(
                q[:, None], rows, weights.attn_sink[: q.shape[1]], idx,
                cfg.softmax_scale,
            )[:, 0]
        for name, o in (("sm80", sm80), ("reference", ref)):
            err = (core.rope(o, pos, freqs, inverse=True).double() - exact).abs()
            self.sum[name] += float(err.sum())
            self.max[name] = max(self.max[name], float(err.max()))
        self.count += exact.numel()

    def report(self) -> str:
        return (
            f"wo_a input before FP8 qdq vs FP64 over {self.count} elements: "
            f"sm80 mean {self.mean('sm80'):.4g} max {self.max['sm80']:.4g}; "
            f"reference mean {self.mean('reference'):.4g} "
            f"max {self.max['reference']:.4g}; ratios mean "
            f"{self.mean('sm80') / max(self.mean('reference'), 1e-30):.3f} "
            f"max {self.max['sm80'] / max(self.max['reference'], 1e-30):.3f}"
        )

    def assert_within(self) -> None:
        print(self.report())
        assert self.count > 0
        assert self.mean("sm80") <= 1.10 * self.mean("reference"), self.report()
        assert self.max["sm80"] <= 1.25 * self.max["reference"], self.report()


def _spy_attend_layer(monkeypatch, errors: _Fp64Errors) -> None:
    attend = core.attend_layer

    def spy(cfg, weights, caches, batch, q, out):
        attend(cfg, weights, caches, batch, q, out)
        errors.measure(cfg, weights, caches, batch, q)

    monkeypatch.setattr(core, "attend_layer", spy)



@pytest.mark.parametrize(
    "specs",
    [
        [LayerSpec(0)],
        [LayerSpec(2, kv_source=True, index_source=True)],
        [
            LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
            LayerSpec(1),
            LayerSpec(1, index_source=True, uses_candidates=True),
        ],
    ],
)
def test_layer_on_cuda_matches_reference(specs, monkeypatch) -> None:
    """The sm_80 layer on CUDA against the reference run on CUDA: cache rows
    bitwise and top-k sets equal. Attention outputs are judged against FP64
    (pooled over the test): the sm80 path's mean error at most 1.10x and its
    max error at most 1.25x those of the reference kernel's arithmetic on
    the same inputs. Run with -s to see both errors."""
    args = RefArgs()
    weights = [make_weights(args, s, 100 + i) for i, s in enumerate(specs)]
    xs = layers.make_inputs(args, len(specs), 5)
    schedule = layers.make_schedule(11)
    errors = _Fp64Errors()
    _spy_attend_layer(monkeypatch, errors)
    run, outs, topks = layers.run_sm80(args, specs, weights, xs, schedule, "cuda")
    ref = layers.run_reference(args, specs, weights, xs, "cuda")
    layers.check(args, specs, run, outs, topks, *ref, compare_outputs=False)
    errors.assert_within()

    # Against the reference run on CPU the selections may differ in a rare
    # near-tie of index scores: the two devices' libm (cos/sin/exp/rsqrt)
    # and fp32 reduction orders differ by ulps, which a BF16 or FP4/FP8
    # rounding can turn into one grid step.
    _, _, cpu_topks = layers.run_reference(args, specs, weights, xs)
    same = sum(topks[k] == v for k, v in cpu_topks.items())
    print(f"top-k sets equal to the CPU reference: {same}/{len(cpu_topks)}")
    assert same >= 0.95 * len(cpu_topks), f"{same}/{len(cpu_topks)} top-k sets equal"
