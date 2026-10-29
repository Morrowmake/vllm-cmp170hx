# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-launch index selection (``VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH``,
``ops/sm80_index_topk._select_one_launch``) on CPU.

- In the Triton interpreter (a subprocess with ``TRITON_INTERPRET=1`` and
  the GPU's ``tl.histogram`` semantics, see ``test_sm80_index_visible``):
  ``select`` with the switch on writes the same bytes as with it off (the
  fifteen-launch path) on random, tied (both zeros, -FLT_MAX), -inf-heavy and
  all-equal rows, rows of fewer visible columns than ``k``, empty rows, a
  visible count past the width, scores past the visible count that are NaN
  or +inf, and rows wider than one part (the two-level merge), with and
  without the producer's first histogram; both equal ``topk_canonical``.
  Its candidate scratch is poisoned beforehand and its tickets are back to
  zero afterwards. With the switch on a call is one launch.
- The kernel compiles for sm_80 (no GPU needed) with the release/acquire
  ticket, the CTA barrier before it and L2 loads of the other parts' slots.
- The switch is off by default and gates on rows, slots and ``k``.

The CUDA half (bytes at widths up to 262144, graph replay) is
``test_sm80_select_one_launch_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_select_one_launch.py
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
FLAG = "VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH"
# Columns per program of the one-launch path (``sm80_index_topk.PART``).
PART = 8192
KERNELS = ("_selection_init", "_radix_hist", "_hist_reduce", "_radix_cutoff",
           "_selection_counts", "_selection_prefix", "_selection_scatter",
           "_select_one_launch")


def producer_histogram(scores, n_vis, width, segment):
    """The score producer's high-byte histogram per segment (columns below
    the visible count), in the selection workspace layout of this width."""
    rows = scores.shape[0]
    ns = -(-max(width, 1) // segment)
    bits = torch.where(scores == 0, 0.0, scores).view(torch.int32).to(torch.int64) & 0xffffffff
    key = bits ^ torch.where(bits >= 0x80000000, 0xffffffff, 0x80000000)
    col = torch.arange(scores.shape[1])
    bins = (torch.arange(rows)[:, None] * ns + col // segment) * 256 + (key >> 24)
    seen = col[None, :] < n_vis[:, None].long().clamp(min=0, max=width)
    hist = torch.bincount(bins[seen], minlength=rows * ns * 256)
    return hist.view(rows, ns, 256).int()


def cases():
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    g = torch.Generator().manual_seed(41)
    wide = 2 * PART + 1000
    out = []

    def add(name, scores, n_vis, k, ws_width=None):
        out.append((name, scores.contiguous(), torch.tensor(n_vis, dtype=torch.int32), k,
                    ws_width or scores.shape[1]))

    w = 3000
    add("random", torch.randn((6, w), generator=g), [w, 2999, 1500, 513, 512, 7], 512)
    tied = torch.randint(-3, 4, (6, w), generator=g).float()
    tied[:, ::7] = -0.0
    tied[:, 5::11] = -core.FLT_MAX
    add("tied", tied, [w, w, 2047, 1025, 600, 513], 512)
    add("tied_k8", tied, [w, 9, 8, 7, 1, 0], 8)
    neg = torch.full((5, w), float("-inf"))
    neg[:, ::97] = torch.randn((5, len(range(0, w, 97))), generator=g)
    neg[1] = float("-inf")
    neg[2, 100:110] = -core.FLT_MAX
    add("neg_inf_heavy", neg, [w, w, w, 1000, 600], 512)
    add("all_equal", torch.full((4, w), 1.5), [w, 1000, 512, 100], 512)
    add("n_vis_below_k", torch.randn((4, w), generator=g), [511, 100, 1, 2], 512)
    add("n_vis_zero", torch.randn((3, w), generator=g), [0, 0, 0], 512)
    add("n_vis_past_width", torch.randn((2, 700), generator=g), [5000, 700], 512)
    add("width_below_k", torch.randn((2, 300), generator=g), [5000, 300], 512)
    rnd = torch.randn((5, wide), generator=g)
    rnd[1] = torch.randint(-2, 3, (wide,), generator=g).float()
    rnd[2] = 0.25
    rnd[3, ::2] = float("-inf")
    add("wide", rnd, [wide, wide, wide - 1, PART + 1, PART], 512)
    add("wide_k2048", rnd[:3], [wide, PART * 2, PART + 3], 2048, ws_width=12 * PART)
    blocks = torch.randn((3, PART + 77), generator=g)
    blocks[:, -1] = float("inf")
    add("wide_inf_tail", blocks, [PART + 77, PART + 70, 4000], 2048, ws_width=8 * PART)
    return out


def interpreted_results() -> dict:
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from triton.runtime import interpreter

    from .test_sm80_index_visible import _gpu_histogram

    interpreter.InterpreterBuilder.create_histogram = _gpu_histogram
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
    from vllm_sm80.models.deepseek_v41.ops import sm80_index_topk as topk

    launches = []

    class Counted:
        def __init__(self, name, fn):
            self.name, self.fn = name, fn

        def __getitem__(self, grid):
            launches.append(self.name)
            return self.fn[grid]

    for name in KERNELS:
        if hasattr(topk, name):
            setattr(topk, name, Counted(name, getattr(topk, name)))

    out = {}
    for name, scores, n_vis, k, ws_width in cases():
        rows, width = scores.shape
        col = torch.arange(width)[None, :]
        shown = col < n_vis[:, None].long()
        garbage = torch.where(col % 2 == 0, float("nan"), float("inf")).expand(rows, width)
        dirty = torch.where(shown, scores, garbage).contiguous()
        clean = torch.where(shown, scores, float("-inf"))
        for first in (False, True):
            got = {}
            for arm in ("0", "1"):
                os.environ[FLAG] = arm
                ws = topk.SelectionWorkspace(rows, ws_width, "cpu")
                for buf in ("hist", "totals", "counts", "prefixes", "state",
                            "cand_keys", "cand_cols", "kept"):
                    if hasattr(ws, buf):
                        getattr(ws, buf).fill_(POISON)
                if first:
                    ns = -(-max(width, 1) // topk.SEGMENT)
                    ws.hist.view(-1)[:rows * ns * 256] = producer_histogram(
                        clean, n_vis, width, topk.SEGMENT).view(-1)
                ids = torch.full((rows, k), POISON, dtype=torch.int32)
                launches.clear()
                topk.select(dirty, n_vis, ids, ws, first_histogram=first)
                tickets = ws.tickets.tolist() if hasattr(ws, "tickets") else None
                # Rows of more than one part went through the merge: every
                # part of theirs wrote its kept count.
                parts = (n_vis.long().clamp(min=0, max=width) + PART - 1) // PART
                merged = [bool((ws.kept[r, :p] != POISON).all()) for r, p in
                          enumerate(parts.tolist()) if p > 1] if hasattr(ws, "kept") else None
                got[arm] = (ids, list(launches), tickets, merged)
            want = core.topk_canonical(clean, n_vis.long().clamp(max=width), k)
            out[f"{name}/first{int(first)}"] = dict(
                same_bytes=bool(torch.equal(got["0"][0], got["1"][0])),
                canonical=bool(torch.equal(got["1"][0], want)),
                multi_canonical=bool(torch.equal(got["0"][0], want)),
                launches_on=got["1"][1],
                tickets=got["1"][2] == [0] * rows,
                merged=got["1"][3],
                wide=bool((n_vis.long().clamp(max=width) > PART).any()),
            )
    return out


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_select_one_launch import "
            "interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_one_launch_bytes_equal_the_multi_launch_path(interpreted):
    assert len(interpreted) == 2 * 12
    for key, c in interpreted.items():
        assert c["multi_canonical"], key
        assert c["same_bytes"] and c["canonical"], (key, c)
        assert c["launches_on"] == ["_select_one_launch"], (key, c)


def test_one_launch_is_one_launch_and_resets_its_tickets(interpreted):
    assert sum(c["wide"] for c in interpreted.values()) == 2 * 3
    for key, c in interpreted.items():
        assert c["launches_on"] == ["_select_one_launch"], (key, c)
        assert c["tickets"], (key, c)
        assert c["merged"] and all(c["merged"]) if c["wide"] else c["merged"] == [], (key, c)


def test_kernel_compiles_for_sm80():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from vllm_sm80.models.deepseek_v41.ops import sm80_index_topk as topk

    kern = topk._select_one_launch
    types = {"scores": "*fp32", "nvis": "*i32", "hist": "*i32", "keys": "*i32",
             "cols": "*i32", "kept": "*i32", "tickets": "*i32", "out": "*i32",
             "width": "i32", "k": "i32"}
    sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
    for first in (False, True):
        consts = dict(ss=262144, os=512, NS=256, SLOTS=32 * topk.PART_SLOTS, PARTS=32,
                      BP=32, FIRST_HIST=first, B=topk.BLOCK)
        c = triton.compile(ASTSource(kern, sig, constexprs=consts),
                           target=GPUTarget("cuda", 80, 32), options={"num_warps": 8})
        assert c.asm["cubin"]
        ptx = c.asm["ptx"]
        assert "atom.global.gpu.acq_rel.add" in ptx
        assert "bar.sync" in ptx
        assert "ld.global.cg" in ptx
        assert "__nv_" not in ptx


def test_switch_default_and_gate(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80.models.deepseek_v41.ops import sm80_index_topk as topk

    monkeypatch.delenv(FLAG, raising=False)
    assert envs.VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH is False
    assert topk.PART == PART
    ws = topk.SelectionWorkspace(48, 262144, "cpu")
    assert not topk.one_launch(6, 262144, 512, ws)
    monkeypatch.setenv(FLAG, "1")
    assert envs.VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH is True
    # The served shapes: token ids over the capacity width, candidate blocks
    # over an eighth of it.
    assert topk.one_launch(6, 262144, 512, ws)
    assert topk.one_launch(48, 262144 // 8, 2048, ws)
    assert not topk.one_launch(49, 262144, 512, ws)
    assert not topk.one_launch(6, 262144, 513, ws)
    # A single part needs no slots.
    assert topk.one_launch(6, topk.PART, 1 << 20, topk.SelectionWorkspace(6, topk.PART, "cpu"))
    assert topk.SelectionWorkspace.bytes(48, 262144) == sum(
        t.numel() * t.element_size() for t in vars(ws).values() if isinstance(t, torch.Tensor))


def test_the_switch_needs_an_enabled_pipeline(monkeypatch):
    """select() runs only in the enabled pipeline's run_indexer, so a switch
    armed on eligible but disabled layers would screen two identical arms."""
    from types import SimpleNamespace

    from vllm_sm80 import dev_ab

    def layers(eligible, enabled):
        return [SimpleNamespace(_sm80_attn_pipeline_eligible=eligible,
                                _sm80_attn_pipeline_enabled=enabled)]

    monkeypatch.setattr(dev_ab, "_pipeline_layers", lambda runner: layers(True, False))
    assert "enabled" in dev_ab._select_unavailable(None)
    monkeypatch.setattr(dev_ab, "_pipeline_layers", lambda runner: layers(False, False))
    assert dev_ab._select_unavailable(None) is not None
    monkeypatch.setattr(dev_ab, "_pipeline_layers", lambda runner: layers(True, True))
    assert dev_ab._select_unavailable(None) is None
