# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 DeepSeek-V4.1 attention layer against an fp32 reference.

For each layer type -- ratio 0 (sliding window only), ratio 2 kv + index
source, and the ratio-1 stack of the candidate source (layer 20), a pure
consumer (21) and an index source that reads the candidates (24) -- several
requests run through the paged sm_80 layer with chunked prefill, decode steps
of one and of several tokens, recycled high block ids and padding rows. The
reference (``dsv41_ref.RefAttention``) runs each request alone, prefill then
one token per step. Every cache row the layer writes (window, compressed,
index key) must match the reference bit for bit, every top-k selection must
be the same set, and the ``wo_a`` input must match up to the BF16 rounding of
softmax probabilities (the reference rounds them per 64-row block against a
running max).

    pytest -q tests/models/deepseek_v41/sm80/test_attn_layers.py
"""

import random

import pytest
import torch

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

from .dsv41_ref import (
    LayerSpec,
    PagedRun,
    RefArgs,
    RefAttention,
    RefShared,
    core_config,
    core_weights,
    layer_inputs,
    make_weights,
)

BF16 = torch.bfloat16

# (prompt length, total length) per request.
REQUESTS = {0: (70, 79), 1: (23, 41), 2: (101, 105)}


def make_schedule(seed: int) -> list[list[tuple[int, int]]]:
    """Prefill chunks of uneven sizes, then decode steps of 1-3 tokens, with
    requests interleaved in a different order every step."""
    rng = random.Random(seed)
    fed = dict.fromkeys(REQUESTS, 0)
    prefill_chunks = {0: [30, 25, 15], 1: [23], 2: [40, 33, 28]}
    steps = []
    while any(fed[r] < REQUESTS[r][1] for r in REQUESTS):
        step = []
        for r in rng.sample(sorted(REQUESTS), len(REQUESTS)):
            prompt, total = REQUESTS[r]
            if fed[r] >= total or (not steps and r == 2):
                continue
            if fed[r] < prompt:
                n = prefill_chunks[r].pop(0)
            else:
                n = min(rng.choice([1, 1, 2, 3]), total - fed[r])
            step.append((r, n))
            fed[r] += n
        steps.append(step)
    for r, (prompt, _) in REQUESTS.items():
        assert not prefill_chunks[r]
    return steps


def make_inputs(args: RefArgs, n_layers: int, seed: int) -> list[dict[int, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [
        {
            r: torch.randn(total, args.dim, generator=g).to(BF16)
            for r, (_, total) in REQUESTS.items()
        }
        for _ in range(n_layers)
    ]


def run_sm80(args, specs, weights, xs, schedule, device="cpu"):
    run = PagedRun(args, specs, device=device)
    cfgs = [core_config(args, s) for s in specs]
    cws = [core_weights(args, s, w, device) for s, w in zip(specs, weights)]
    h = args.n_heads // args.world_size
    outs = {}  # (layer, rid, pos) -> [H, D]
    topks = {}  # (layer, rid, pos) -> sorted list of local positions
    g = torch.Generator().manual_seed(7)
    for sched in schedule:
        starts = {r: (run.reqs[r].length if r in run.reqs else 0) for r, _ in sched}
        batch = run.build(sched)
        T = batch.num_tokens
        tokens = [(r, starts[r] + i) for r, n in sched for i in range(n)]
        for li, spec in enumerate(specs):
            x = torch.cat([xs[li][r][starts[r] : starts[r] + n] for r, n in sched])
            qr_kv, kv_score, iw = layer_inputs(args, spec, weights[li], x)
            pad = 3  # CUDA-graph padding rows carry garbage
            qr_kv = torch.cat([qr_kv, torch.randn(pad, qr_kv.shape[1], generator=g).to(BF16)])
            if kv_score is not None:
                kv_score = torch.cat([kv_score, torch.randn(pad, kv_score.shape[1], generator=g)])
            if iw is not None:
                iw = torch.cat([iw, torch.randn(pad, iw.shape[1], generator=g).to(BF16)])
            out = torch.full((T + pad, h, args.head_dim), 3.0, dtype=BF16, device=device)
            core.forward_layer(
                cfgs[li],
                cws[li],
                run.caches(li),
                batch,
                qr_kv.to(device),
                None if kv_score is None else kv_score.to(device),
                None if iw is None else iw.to(device),
                out,
            )
            assert not out[T:].any(), "padding rows must come back zero"
            for t, (r, p) in enumerate(tokens):
                outs[(li, r, p)] = out[t].cpu()
                if spec.ratio:
                    row = run.topk[t].cpu().tolist()
                    topks[(li, r, p)] = [j for j in row if j >= 0]
        run.after_step()
    return run, outs, topks


def run_reference(args, specs, weights, xs, device="cpu"):
    """The reference, every request alone, on ``device`` (its tensors are
    created under that default device, so its libm, reductions and RoPE
    table are the device's)."""
    outs, topks, layers = {}, {}, {}
    weights = [{k: v.to(device) for k, v in w.items()} for w in weights]
    with torch.device(device):
        for r, (prompt, total) in REQUESTS.items():
            refs = [RefAttention(args, s, w) for s, w in zip(specs, weights)]
            shared = RefShared()
            layers[r] = refs
            steps = [(0, prompt)] + [(p, 1) for p in range(prompt, total)]
            for start, n in steps:
                for li, ref in enumerate(refs):
                    x = xs[li][r][start : start + n].unsqueeze(0).to(device)
                    o = ref.forward(x, start, shared).cpu()
                    for i in range(n):
                        outs[(li, r, start + i)] = o[0, i]
                    if specs[li].ratio:
                        idxs = shared.topk_idxs.cpu()
                        offset = n if start == 0 else args.window_size
                        for i in range(n):
                            row = idxs[0, i].tolist() if idxs.shape[-1] else []
                            topks[(li, r, start + i)] = [
                                j - offset for j in row if j >= 0
                            ]
    return layers, outs, topks


def check(
    args, specs, run, outs, topks, ref_layers, ref_outs, ref_topks,
    compare_outputs: bool = True,
):
    """Caches bitwise and top-k sets equal; with ``compare_outputs`` also the
    wo_a inputs to the per-element bound and the <= 2 % mismatch gate (CPU
    against CPU, where the reference kernel's arithmetic is the sm_80 one
    up to the BF16 probability rounding). The GPU test compares outputs
    against an FP64 reference instead."""
    # Top-k selections: identical sets (both sorted ascending).
    for key, want in ref_topks.items():
        assert topks[key] == want, f"top-k differs at (layer, req, pos) {key}"
    # The schedule exercises both regimes of every selection: rows that see
    # fewer compressed positions than index_topk (all kept) and rows that
    # must choose.
    for li, spec in enumerate(specs):
        if spec.ratio:
            sizes = {len(v) for k, v in ref_topks.items() if k[0] == li}
            assert args.index_topk in sizes, sizes
            assert min(sizes) < args.index_topk // 4, sizes
            assert len(sizes) > 3, sizes
    assert len(topks) == len(ref_topks)
    # Cache rows: bit for bit.
    for r, (_, total) in REQUESTS.items():
        for li, spec in enumerate(specs):
            ref = ref_layers[r][li]
            for p in range(max(0, total - args.window_size), total):
                got = run.window_row(li, r, p).cpu()
                want = ref.window_kv_cache[0, p % args.window_size].cpu()
                assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (
                    f"window row layer {li} req {r} pos {p}"
                )
            if spec.kv_source:
                for j in range(total // spec.ratio):
                    got = run.comp_row(r, j).cpu()
                    want = ref.compress_kv_cache[0, j].cpu()
                    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (
                        f"compressed row req {r} pos {j}"
                    )
                    if spec.index_source:
                        got = run.comp_row(r, j, idxk=True).cpu()
                        want = ref.k_cache[0, j].cpu()
                        assert torch.equal(
                            got.view(torch.int16), want.view(torch.int16)
                        ), f"index key req {r} pos {j}"
    # wo_a inputs: the same values except where a BF16 probability rounding
    # changed the pre-quant output. That change is one BF16 step relative to
    # the terms being summed, which have the magnitude of the element's FP8
    # block, not of the element: near-zero elements may move by one BF16
    # ulp of their block's amax. Larger elements may cross one FP8 rounding
    # boundary (an E4M3 step, at most 12.5 % of the element).
    nonzero = sum(int((v != 0).sum()) for v in outs.values())
    n_total = sum(v.numel() for v in outs.values())
    assert nonzero > 0.9 * n_total, "outputs are mostly zero"
    assert all(bool(torch.isfinite(v.float()).all()) for v in outs.values())
    if not compare_outputs:
        return
    n_diff = n_total = 0
    for key, want in ref_outs.items():
        got = outs[key]
        assert torch.isfinite(got.float()).all()
        diff = got.float() != want.float()
        n_diff += int(diff.sum())
        n_total += diff.numel()
        w = want.float().flatten()
        amax = w.view(-1, 32).abs().amax(-1, keepdim=True)
        bf16_ulp = torch.where(
            amax > 0, torch.exp2(torch.floor(torch.log2(amax)) - 7), 0.0
        )
        tol = bf16_ulp.expand(-1, 32).flatten() + 0.13 * w.abs()
        err = (got.float().flatten() - w).abs()
        worst = int(torch.argmax(err - tol))
        assert bool((err <= tol).all()), (
            f"{key} element {worst}: |diff| {err[worst].item():.4g} > "
            f"{tol[worst].item():.4g} (block amax {amax[worst // 32].item():.4g})"
        )
    assert n_diff <= 0.02 * n_total, f"{n_diff}/{n_total} wo_a input elements differ"


@pytest.mark.parametrize(
    "name,specs",
    [
        ("ratio0", [LayerSpec(0)]),
        ("ratio2_source", [LayerSpec(2, kv_source=True, index_source=True)]),
        (
            "ratio1_candidate_stack",
            [
                LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
                LayerSpec(1),
                LayerSpec(1, index_source=True, uses_candidates=True),
            ],
        ),
    ],
)
def test_layer_matches_reference(name: str, specs: list[LayerSpec]) -> None:
    torch.manual_seed(0)
    args = RefArgs()
    weights = [make_weights(args, s, 100 + i) for i, s in enumerate(specs)]
    xs = make_inputs(args, len(specs), 5)
    schedule = make_schedule(11)
    run, outs, topks = run_sm80(args, specs, weights, xs, schedule)
    ref_layers, ref_outs, ref_topks = run_reference(args, specs, weights, xs)
    check(args, specs, run, outs, topks, ref_layers, ref_outs, ref_topks)


def test_candidate_mask() -> None:
    """Level two keeps the candidate blocks' scores, sends the other visible
    positions to -FLT_MAX and leaves invisible ones at -inf; with fewer
    candidate positions than k the selection then takes the lowest-index
    masked ones, as the reference's top-k over -inf does with ties to the
    lowest index."""
    s = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    s[0, 9] = float("-inf")  # not visible yet
    cands = torch.tensor([[2, -1]])  # positions 8..9 with block size 4
    m = core.mask_to_candidates(s, cands, 4)
    want = [-core.FLT_MAX] * 8 + [8.0, float("-inf")]
    assert m[0].tolist() == want
    got = core.topk_canonical(m, torch.tensor([9]), 3)
    assert got.tolist() == [[0, 1, 8]]
    cands = torch.tensor([[0, 2]])
    m = core.mask_to_candidates(s, cands, 4)
    assert core.topk_canonical(m, torch.tensor([9]), 3).tolist() == [[2, 3, 8]]


def test_topk_canonical_ties() -> None:
    scores = torch.tensor(
        [[1.0, 2.0, 2.0, 2.0, 0.5, float("-inf")], [3.0, -core.FLT_MAX, 1.0, 1.0, 1.0, 1.0]]
    )
    n_vis = torch.tensor([5, 6])
    got = core.topk_canonical(scores, n_vis, 3)
    assert got.tolist() == [[1, 2, 3], [0, 2, 3]]
    got = core.topk_canonical(scores, torch.tensor([2, 1]), 3)
    assert got.tolist() == [[0, 1, -1], [0, -1, -1]]


def test_candidate_blocks_pin_newest_block() -> None:
    # Block 2 (positions 8..9, newest) scores lowest but is pinned in.
    s = torch.full((1, 10), 0.0)
    s[0, :4] = 5.0
    s[0, 4:8] = 4.0
    s[0, 8:] = -3.0
    got = core.select_candidate_blocks(s, torch.tensor([10]), 2, 4)
    assert got.tolist() == [[0, 2]]
    # Unreachable blocks (all -inf) are never kept.
    s[0, 4:] = float("-inf")
    got = core.select_candidate_blocks(s, torch.tensor([4]), 3, 4)
    assert got.tolist() == [[0, -1, -1]]


def test_paged_offsets_use_int64() -> None:
    """Slots whose element offset passes 2**31 address the right rows."""
    cache = torch.zeros(2, 4, 8, dtype=BF16)
    big = torch.as_strided(cache, (2**27, 4, 8), (0, 8, 1))  # block id 2**26 + 1
    slots = torch.tensor([(2**26 + 1) * 4 + 3], dtype=torch.int64)
    blk, off = core._paged(big, slots)
    assert blk.item() == 2**26 + 1 and off.item() == 3
    assert blk.dtype == torch.int64 and blk.item() * 4 * 8 > 2**31
