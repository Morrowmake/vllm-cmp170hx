# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention pipeline's index scoring and selection touch only the
segments that hold a visible column (``ops/sm80_index_topk.py``,
``ops/sm80_attn_pipeline._index_score_stats_kernel``), on CPU in the Triton
interpreter (a subprocess with ``TRITON_INTERPRET=1``).

- ``select`` gives ``core.topk_canonical``'s ids when every selection scratch
  buffer is poisoned beforehand and the scores past each row's visible count
  are garbage (NaN, +inf); the histogram and count scratch of segments past
  the visible count are left untouched (never written, so never read).
The interpreter's ``tl.histogram`` (``np.histogram`` over ``[0, bins]``)
counts a value equal to ``bins`` in the last bin, where the GPU lowering
drops it; the selectors use ``bins`` as their "not counted" value, so the
subprocess installs the GPU semantics (:func:`_gpu_histogram`) first. With
the interpreter's own histogram the selection is wrong before and after this
change alike.

- ``run_indexer`` writes the same score row, block maxima and producer
  histogram for segments past the visible count as the tile loop does
  (-inf scores, -inf maxima but +inf at the visible tail block, an empty
  histogram), and its selections are ``topk_canonical`` /
  ``select_candidate_blocks`` of its own scores. The visible scores'
  arithmetic is unchanged and judged bitwise on CUDA only (the interpreter
  does not reproduce the tile's BF16 dot); here they need only be finite.

The CUDA half (bytes against the torch path at widths up to 262144) is in
``test_sm80_index_topk_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_index_visible.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
POISON = -12345


def _gpu_histogram(self, data, bins, mask):
    """``tl.histogram`` as compiled for the GPU: values outside
    ``[0, bins)`` and masked lanes are not counted."""
    import numpy as np
    from triton.runtime import interpreter
    from triton.language import int32

    keep = (data.data >= 0) & (data.data < bins)
    if mask is not None:
        keep &= mask.data
    counts = np.bincount(np.where(keep, data.data, 0).astype(np.int64),
                         weights=keep.astype(np.int64), minlength=bins)[:bins]
    return interpreter.TensorHandle(counts.astype(np.int32), int32)


def interpreted_results() -> dict:
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from triton.runtime import interpreter

    interpreter.InterpreterBuilder.create_histogram = _gpu_histogram
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline as pipe
    from vllm_sm80.models.deepseek_v41.ops.sm80_index_topk import (
        SEGMENT, SelectionWorkspace, select,
    )

    from .test_sm80_index_topk import config
    from .test_sm80_kernels_gpu import _index_case

    out = {}
    g = torch.Generator().manual_seed(7)
    for width in (1, SEGMENT, SEGMENT + 1, 3 * SEGMENT + 5):
        for k in (8, 512):
            rows = 6
            scores = torch.randint(-6, 7, (rows, width), generator=g).float()
            visible = torch.tensor([0, 1, min(width, k - 1), min(width, k), min(width, SEGMENT + 3), width])
            col = torch.arange(width)[None, :]
            garbage = torch.where(col % 2 == 0, float("nan"), float("inf")).expand(rows, width)
            scores = torch.where(col < visible[:, None], scores, garbage).contiguous()
            ws = SelectionWorkspace(rows, width, "cpu")
            for name in ("hist", "totals", "counts", "prefixes"):
                getattr(ws, name).fill_(POISON)
            ws.state.fill_(POISON)
            got = torch.empty((rows, k), dtype=torch.int32)
            select(scores, visible.int(), got, ws)
            clean = torch.where(col < visible[:, None], scores, float("-inf"))
            want = core.topk_canonical(clean, visible, k)
            used = (visible.clamp(max=width) + SEGMENT - 1) // SEGMENT
            untouched = all(
                bool((ws.hist[r, used[r]:] == POISON).all() and (ws.counts[r, used[r]:] == POISON).all())
                for r in range(rows))
            out[f"select/w{width}/k{k}"] = dict(equal=bool(torch.equal(got, want)), untouched=untouched)

    for source, masked in ((True, False), (False, False), (False, True)):
        width = 2 * SEGMENT + 17
        q, w, visible, req, keys, table = _index_case(g, n=4, blocks=(width + 63) // 64)
        visible[:] = torch.tensor([0, 5, SEGMENT + 9, width])
        candidates = torch.full((4, 2048), -1, dtype=torch.int32)
        if masked:
            candidates[:, :3] = torch.tensor([0, 1, (SEGMENT + 8) // 8])
        cfg = config(compress_ratio=1, is_candidate_source=source, uses_candidates=masked)
        batch = core.Sm80AttnBatch(visible - 1, req, torch.zeros(1), torch.where(visible > 0, 0, -1),
                                   torch.empty(4, 0, dtype=torch.int32), idxk_block_table=table,
                                   max_seq_len=width)
        caches = core.Sm80AttnCaches(swa=torch.empty(0), idxk=keys,
                                     topk=torch.empty((4, 512), dtype=torch.int32),
                                     candidates=candidates.clone())
        ws = pipe.PipelineWorkspace(4, width, 1280, "cpu")
        ws.scores.fill_(float("nan"))
        ws.maxima.fill_(float("nan"))
        ws.selection.hist.fill_(POISON)
        pipe.run_indexer(cfg, caches, batch, q, w, ws)
        sc = ws.scores[:, :width]
        col = torch.arange(width)[None, :]
        seg_start = (torch.arange(width) // SEGMENT * SEGMENT)[None, :]
        far = seg_start >= visible[:, None]
        torch_scores = core.index_scores_torch(q, w, visible, req, keys, table, width, 8)
        if masked:
            torch_scores = core.mask_to_candidates(torch_scores, candidates, 8)
        vis = col < visible[:, None]
        blocks = (width + 7) // 8
        bcol = torch.arange(blocks)[None, :]
        bfar = (bcol * 8 // SEGMENT * SEGMENT) >= visible[:, None]
        tail = bcol == torch.div(visible - 1, 8, rounding_mode="trunc")[:, None]
        mx = ws.maxima[:, :blocks]
        ns = (width + SEGMENT - 1) // SEGMENT
        hist_far = all(
            bool((ws.selection.hist[r, s] == 0).all())
            for r in range(4) for s in range(ns) if s * SEGMENT >= int(visible[r]))
        key = f"indexer/source{int(source)}/masked{int(masked)}"
        out[key] = dict(
            far_neg_inf=bool((sc[far] == float("-inf")).all()),
            invisible_neg_inf=bool((sc[~vis] == float("-inf")).all()),
            visible_finite=bool(torch.isfinite(sc[vis & (torch_scores > -3e38)]).all()),
            far_maxima=(not source) or bool(
                (mx[bfar & ~tail] == float("-inf")).all() and (mx[bfar & tail] == float("inf")).all()),
            # At the candidate source the block selector reuses the same
            # histogram scratch after the producer, so only the other roles
            # still hold the producer's histogram.
            hist_far_empty=source or hist_far,
            topk=bool(torch.equal(caches.topk, core.topk_canonical(sc, visible, 512))),
            candidates=bool(torch.equal(caches.candidates,
                                        core.select_candidate_blocks(sc, visible, 2048, 8)))
            if source else bool(torch.equal(caches.candidates, candidates)),
        )
    return out


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_index_visible import "
            "interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_select_ignores_and_skips_invisible_segments(interpreted):
    cases = {k: v for k, v in interpreted.items() if k.startswith("select/")}
    assert len(cases) == 8
    for key, c in cases.items():
        assert c == dict(equal=True, untouched=True), key


def test_score_producer_fast_path_writes_the_tile_loop_values(interpreted):
    cases = {k: v for k, v in interpreted.items() if k.startswith("indexer/")}
    assert len(cases) == 3
    for key, c in cases.items():
        assert all(c.values()), (key, c)
