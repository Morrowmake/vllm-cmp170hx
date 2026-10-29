# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a FULL CUDA graph of the sm_80 DeepSeek-V4.1 attention relies on.

* The indexer's score width is fixed per graph (``max_model_len // ratio``):
  scoring at that width gives the same caches, selections and outputs, bit
  for bit, as scoring at each step's own bound.
* A decode step's batch reads the metadata builders' buffers themselves, so
  a replay sees the contents the builders wrote in place; in a FULL graph the
  width bound is ``max_model_len``.

    pytest -q tests/models/deepseek_v41/sm80/test_full_graph_shapes.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

from . import test_attn_layers as layers
from .dsv41_ref import (
    LayerSpec,
    PagedRun,
    RefArgs,
    core_config,
    core_weights,
    layer_inputs,
    make_weights,
)


@pytest.mark.parametrize(
    "specs",
    [
        [LayerSpec(2, kv_source=True, index_source=True), LayerSpec(2)],
        [
            LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
            LayerSpec(1),
            LayerSpec(1, index_source=True, uses_candidates=True),
        ],
    ],
)
def test_fixed_score_width_changes_nothing(monkeypatch, specs) -> None:
    args = RefArgs()
    weights = [make_weights(args, s, 100 + i) for i, s in enumerate(specs)]
    xs = layers.make_inputs(args, len(specs), 5)
    schedule = layers.make_schedule(11)
    variable = layers.run_sm80(args, specs, weights, xs, schedule)

    build = PagedRun.build

    def fixed_width(self, sched):
        batch = build(self, sched)
        batch.max_seq_len = args.max_seq_len  # the graph's bound
        return batch

    monkeypatch.setattr(PagedRun, "build", fixed_width)
    fixed = layers.run_sm80(args, specs, weights, xs, schedule)

    (run_v, outs_v, topk_v), (run_f, outs_f, topk_f) = variable, fixed
    assert topk_v == topk_f
    for key in outs_v:
        assert torch.equal(outs_v[key].view(torch.int16), outs_f[key].view(torch.int16))
    for a, b in zip(
        [run_v.comp, run_v.idxk, run_v.ring, *run_v.swa_caches],
        [run_f.comp, run_f.idxk, run_f.ring, *run_f.swa_caches],
    ):
        assert torch.equal(a, b)


def _layer(**attrs) -> sm80.DeepseekV41Sm80Attention:
    layer = object.__new__(sm80.DeepseekV41Sm80Attention)
    nn.Module.__init__(layer)
    for k, v in attrs.items():
        object.__setattr__(layer, k, v)
    return layer


def _metadata(T: int, cap: int = 16):
    """Decode-step metadata as the builders hold it: slices of persistent
    buffers sized for the largest batch."""
    buf = dict(
        swa_idx=torch.zeros(cap, 1, 8, dtype=torch.int32),
        t2r=torch.zeros(cap, dtype=torch.int32),
        qsl=torch.zeros(cap + 1, dtype=torch.int32),
        slots=torch.zeros(cap, dtype=torch.int64),
        comp_slots=torch.zeros(cap, dtype=torch.int64),
        idxk_slots=torch.zeros(cap, dtype=torch.int64),
        ring_slots=torch.zeros(cap, dtype=torch.int64),
        bt=torch.zeros(cap, 4, dtype=torch.int32),
    )
    md = {
        "swa": SimpleNamespace(
            num_decode_tokens=T,
            num_prefill_tokens=0,
            decode_swa_indices=buf["swa_idx"][:T],
            prefill_swa_indices=None,
            token_to_req_indices=buf["t2r"][:T],
            query_start_loc=buf["qsl"][: T + 1],
            slot_mapping=buf["slots"][:T],
        ),
        "comp": SimpleNamespace(
            block_table=buf["bt"], slot_mapping=buf["comp_slots"][:T], max_seq_len=37
        ),
        "idxk": SimpleNamespace(block_table=buf["bt"], slot_mapping=buf["idxk_slots"][:T]),
        "ring": SimpleNamespace(slot_mapping=buf["ring_slots"][:T]),
    }
    return md, buf


@pytest.mark.parametrize("full", [False, True])
def test_decode_batch_reads_persistent_buffers(monkeypatch, full) -> None:
    monkeypatch.setattr(sm80, "_in_full_graph", lambda: full)
    layer = _layer(
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        compress_ratio=2,
        compressed_cache_prefix="comp",
        is_kv_source=True,
        max_model_len=4096,
        indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix="idxk")),
        compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="ring")),
    )
    positions = torch.arange(16, dtype=torch.int64)
    md, buf = _metadata(8)
    first = layer._sm80_batch(md, positions)
    # Every tensor the core reads is (a view of) a builder buffer: what a
    # replay reads after the builders wrote the next step in place.
    pairs = [
        (first.swa_indices, buf["swa_idx"]),
        (first.token_to_req, buf["t2r"]),
        (first.query_start_loc, buf["qsl"]),
        (first.swa_slots, buf["slots"]),
        (first.comp_slots, buf["comp_slots"]),
        (first.idxk_slots, buf["idxk_slots"]),
        (first.ring_slots, buf["ring_slots"]),
        (first.comp_block_table, buf["bt"]),
        (first.idxk_block_table, buf["bt"]),
        (first.positions, positions),
    ]
    for view, base in pairs:
        assert view.data_ptr() == base.data_ptr()
    assert first.swa_indices.shape == (8, 8)
    # The score-width bound: the graph's in FULL, the step's otherwise.
    assert first.max_seq_len == (4096 if full else 37)

    md2, _ = _metadata(8)
    md2["comp"].max_seq_len = 1000
    for name in ("swa", "comp", "idxk", "ring"):
        for attr, value in vars(md[name]).items():
            if isinstance(value, torch.Tensor):
                setattr(md2[name], attr, value)
    second = layer._sm80_batch(md2, positions)
    assert second.max_seq_len == (4096 if full else 1000)
    assert all(
        getattr(first, f).data_ptr() == getattr(second, f).data_ptr()
        for f in ("swa_indices", "swa_slots", "comp_slots", "token_to_req")
    )


def test_full_graph_flag_follows_the_forward_context(monkeypatch) -> None:
    from vllm.config import CUDAGraphMode

    monkeypatch.setattr(sm80, "is_forward_context_available", lambda: False)
    assert not sm80._in_full_graph()
    monkeypatch.setattr(sm80, "is_forward_context_available", lambda: True)
    for mode, want in (
        (CUDAGraphMode.FULL, True),
        (CUDAGraphMode.PIECEWISE, False),
        (CUDAGraphMode.NONE, False),
    ):
        monkeypatch.setattr(
            sm80,
            "get_forward_context",
            lambda mode=mode: SimpleNamespace(cudagraph_runtime_mode=mode),
        )
        assert sm80._in_full_graph() is want



def _full_graph_padding(build, pad: int):
    """``PagedRun.build`` with ``pad`` padding tokens appended the way a FULL
    graph replays a smaller batch: the metadata covers the padded token count,
    the padding tokens fall past the last request's query and so map to
    request 0 (``compute_token_to_req_indices``), their window, compressed and
    index-key slots are PAD, their positions are stale (here: request 0's own,
    the worst case), and the ring slot kernel gives them request 0's ring
    block. Their input rows are the garbage rows ``run_sm80`` appends."""

    def padded(self, sched):
        return _pad_like_full(self, build(self, sched), pad)

    return padded


def _pad_like_full(run: PagedRun, batch, pad: int):
    """``batch`` with ``pad`` padding tokens as a FULL graph's metadata has
    them (see :func:`_full_graph_padding`)."""
    if pad <= 0:
        return batch
    dev = batch.positions.device
    p0 = batch.positions[:1].expand(pad)
    minus = torch.full((pad,), -1, dtype=torch.int64, device=dev)
    batch.positions = torch.cat([batch.positions, p0])
    batch.token_to_req = torch.cat(
        [
            batch.token_to_req,
            torch.zeros(pad, dtype=batch.token_to_req.dtype, device=dev),
        ]
    )
    batch.swa_slots = torch.cat([batch.swa_slots, minus])
    batch.swa_indices = torch.cat(
        [batch.swa_indices, torch.full_like(batch.swa_indices[:1], -1).expand(pad, -1)]
    )
    for name in ("comp_slots", "idxk_slots"):
        setattr(batch, name, torch.cat([getattr(batch, name), minus]))
    req0 = run.last_reqs[0]
    cap = run.ring.shape[1]
    ring = req0.ring * cap + p0 % cap if req0.ring >= 0 else minus
    batch.ring_slots = torch.cat([batch.ring_slots, ring])
    return batch


@pytest.mark.parametrize(
    "specs",
    [
        [LayerSpec(0)],
        [LayerSpec(2, kv_source=True, index_source=True), LayerSpec(2)],
        [
            LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
            LayerSpec(1),
            LayerSpec(1, index_source=True, uses_candidates=True),
        ],
    ],
)
def test_full_graph_padding_rows_change_nothing(monkeypatch, specs) -> None:
    """A FULL graph replays a batch of fewer requests with padding tokens in
    the attention region (PIECEWISE runs it on the real tokens only): the
    padding must change no cache row, selection or output."""
    args = RefArgs()
    weights = [make_weights(args, s, 100 + i) for i, s in enumerate(specs)]
    xs = layers.make_inputs(args, len(specs), 5)
    schedule = layers.make_schedule(11)
    plain = layers.run_sm80(args, specs, weights, xs, schedule)
    monkeypatch.setattr(PagedRun, "build", _full_graph_padding(PagedRun.build, 3))
    padded = layers.run_sm80(args, specs, weights, xs, schedule)
    (run_a, outs_a, topk_a), (run_b, outs_b, topk_b) = plain, padded
    assert topk_a == topk_b
    for key in outs_a:
        assert torch.equal(outs_a[key].view(torch.int16), outs_b[key].view(torch.int16)), key
    for name, a, b in zip(
        ["comp", "idxk", "ring", "swa"],
        [run_a.comp, run_a.idxk, run_a.ring, run_a.swa_caches[0]],
        [run_b.comp, run_b.idxk, run_b.ring, run_b.swa_caches[0]],
    ):
        assert torch.equal(a, b), name


_STACKS = [
    [LayerSpec(0)],
    [LayerSpec(2, kv_source=True, index_source=True), LayerSpec(2)],
    [
        LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
        LayerSpec(1),
        LayerSpec(1, index_source=True, uses_candidates=True),
    ],
]


def _decode_run(args, specs, weights, live: int, rows: int, full: bool, device):
    """Prefill ``live`` requests, then decode them one token a step with the
    layer's input padded to ``rows`` rows. ``full``: the metadata covers the
    padding tokens too, as in a FULL graph; else only the live tokens, as in
    a piecewise step. Returns the live outputs and the caches."""
    run = PagedRun(args, specs, device=device)
    cfgs = [core_config(args, s) for s in specs]
    cws = [core_weights(args, s, w, device) for s, w in zip(specs, weights)]
    h = args.n_heads // args.world_size
    g = torch.Generator().manual_seed(3)
    prefill = [(r, 40 + 9 * r) for r in range(live)]
    schedule = [prefill] + [[(r, 1) for r in range(live)] for _ in range(12)]
    outs = []
    for step, sched in enumerate(schedule):
        batch = run.build(sched)
        T = batch.num_tokens
        n = T if step == 0 else rows
        if full:
            batch = _pad_like_full(run, batch, n - T)
        for li, spec in enumerate(specs):
            x = torch.randn(n, args.dim, generator=g).to(torch.bfloat16)
            qr_kv, kv_score, iw = layer_inputs(args, spec, weights[li], x)
            out = torch.full((n, h, args.head_dim), 3.0, dtype=torch.bfloat16)
            core.forward_layer(
                cfgs[li],
                cws[li],
                run.caches(li),
                batch,
                qr_kv.to(device),
                None if kv_score is None else kv_score.to(device),
                None if iw is None else iw.to(device),
                out_dev := out.to(device),
            )
            outs.append(out_dev[:T].cpu())
        run.after_step()
    caches = [run.comp, run.idxk, run.ring, run.topk, *run.swa_caches]
    return outs, [c.cpu() for c in caches]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("specs", _STACKS)
def test_full_and_piecewise_row_counts_agree(specs) -> None:
    """A decode step of ``live`` requests replayed by a FULL graph of
    ``rows`` tokens (metadata over the padded tokens) gives the outputs and
    caches of the piecewise step (metadata over the live tokens, input
    padded to ``rows``), bit for bit, on the CUDA kernels."""
    device, live, rows = "cuda", 5, 8
    args = RefArgs()
    weights = [make_weights(args, s, 300 + i) for i, s in enumerate(specs)]
    pw_out, pw_caches = _decode_run(args, specs, weights, live, rows, False, device)
    full_out, full_caches = _decode_run(args, specs, weights, live, rows, True, device)
    for i, (a, b) in enumerate(zip(pw_out, full_out)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), i
    for i, (a, b) in enumerate(zip(pw_caches, full_caches)):
        if i == 3:  # top-k buffer: the live rows of the last index source
            a, b = a[:live], b[:live]
        assert torch.equal(a, b), i
