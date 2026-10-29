# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE v2 for the DSpark drafter's routed experts
(VLLM_EXL3_MOE_V2_DRAFT, csrc/exl3/moe_v2.cu): the drafter layers mtp.0-2 of
DeepSeek-V4.1-Flash, top-3 of 128 experts, every routed matrix at bits_x2 8
with codebook 2, SwiGLU clamp 10; the drafter's graphs dispatch 5..40 rows (5
per request).
- Expert parallel size 4 (hidden 5120, intermediate 2304, 32 whole experts
  local): stages of their own (DraftTiling): route_and_gate_up_group_input
  (one 2^e per 640 gate/up columns), moe_lane_kernel with DraftLanes (row
  tiles of up to 16 rows, the low part of 5..16-row tiles in the MMA's C
  operand, bits_x2 8 decoded from two ring words), activate_down_slice (one
  2^e per down K slice) and combine_routes.
- Tensor-parallel experts (every expert local, a 640- or 512-column chunk of
  the intermediate per rank): the gate stays closed (moe_out); moe_out_v2's
  kernels for them (the tensor-parallel shapes' stages at bits_x2 8) are
  tested on the GPU below.

CPU:
- both switches default off; the drafter's configurations open only for
  VLLM_EXL3_MOE_V2_DRAFT (draft=True), never for VLLM_EXL3_MOE_V2, whose
  configurations and closed reasons are unchanged;
- Exl3MoEMethod._moe_v2(draft=True) sets the row tile floor before warming
  bits_x2 8, sizes the workspace and logs its banner through the vLLM logger;
  with VLLM_EXL3_MOE_V2_FOLD it logs that the fold does not apply and neither
  requires nor warms the folded operator; a closed gate logs one line;
- the row tile floor accepts 0, 8 and 16 only, checked before the
  extension is asked (Exl3Experts.apply's bits_x2 dispatch: see
  test_exl3_moe_v2.py and test_exl3_moe_prefill_v2.py);
- the extension registers moe_v2_draft_force_rows;
- moe_v2.cu compiled for sm_80 (nvcc, no GPU): every drafter kernel exists for
  each row tile floor, the other bits_x2 8 kernels are the tensor-parallel
  shapes' unfolded stages only (no folded bits_x2 8 kernel), and the
  drafter's kernels touch no local memory.

GPU (SM 8.0), each drafter shape:
- synthetic drafter experts (random trellis; one of the four weight sets only
  2-byte aligned): moe_out_v2 holds moe_out's error against an FP64 reference
  (pooled mean <= 1.10x, max <= 1.25x, exact where moe_out is exact, finite)
  with the BF16 tie rule (elements above the max budget are forgiven only if
  every one is a genuine BF16 rounding tie with moe_out and they are at most
  1e-3 of the elements) over 1..64 rows with 1..4, 5..8, 9..16-row and
  several tiles per expert, masked and int64 ids and the reached clamp;
- rows without local routes are exact zero; calls repeat bitwise, also with a
  reused workspace over 40, 5, 13 and 40 rows;
- moe_out_v2_fold refuses bits_x2 8;
- a captured call replays bitwise like the eager one (5, 16 and 40 rows, new
  inputs too) without allocator events;
- expert parallel: with the row tile floor at 8 or 16 a row's output does not
  depend on the other rows of its call;
- target path unchanged: with VLLM_EXL3_TEST_BASE_TREE naming a tree without
  this change (its EXL3 extension built), moe_out_v2 and moe_out_v2_fold at
  hidden 5120 (bits_x2 6, 4) and 2560 (bits_x2 10), and moe_out at the
  drafter's configuration, return the same bytes from both trees
  (exl3_moe_v2_digests.py, one process per tree);
- the real experts of mtp.0, mtp.1 and mtp.2 as expert-parallel rank 0 holds
  them, when VLLM_EXL3_TEST_CHECKPOINT names the checkpoint: the FP64 rule on
  rows drawn like the layer's MoE input at 5..40 rows.
"""

import logging
import os
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization import test_exl3_moe_v2_hidden5120 as h5120
from tests.kernels.quantization.test_exl3_moe_v2 import (
    _allocations,
    _bitwise,
    _extension_or_skip,
    needs_sm80,
)
from tests.kernels.quantization.test_exl3_moe_v2_fold import sm80_sass  # noqa: F401
from vllm_sm80.kernels.exl3 import ops as exl3_ops

HIDDEN, INTERMEDIATE, TOPK = 5120, 2304, 3
LOCAL, GLOBAL = 32, 128
BITS_X2 = 8
SWIGLU_LIMIT = 10.0
DRAFTER = (HIDDEN, INTERMEDIATE, TOPK, LOCAL, GLOBAL)
# (intermediate, local experts): expert parallel, then the tensor-parallel
# rank chunks.
SHAPES = {"ep": (INTERMEDIATE, LOCAL), "tp640": (640, GLOBAL), "tp512": (512, GLOBAL)}
# The drafter's graph rows (5 per request, 1..8 requests).
GRAPH_ROWS = (5, 10, 15, 20, 25, 30, 35, 40)
REAL_LAYERS = ("mtp.0", "mtp.1", "mtp.2")
TIE_CAP_FRACTION = 1e-3


def _meta(bits_x2=BITS_X2, intermediate=INTERMEDIATE, local=LOCAL):
    return [exl3_ops.moe_v2_expert_meta(HIDDEN, intermediate, bits_x2)] * local


def _reason(draft=True, **changes):
    args = dict(
        capability=(8, 0),
        dtype=torch.bfloat16,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        local_experts=LOCAL,
        global_experts=GLOBAL,
        metadata=_meta(),
    )
    args.update(changes)
    return exl3_ops.moe_v2_closed_reason(**args, draft=draft)


def _split(width, **changes):
    args = dict(
        intermediate_size=width,
        local_experts=GLOBAL,
        metadata=_meta(intermediate=width, local=GLOBAL),
    )
    args.update(changes)
    return args


# ---------------------------------------------------------------- CPU tests


def test_switches_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_MOE_V2_DRAFT", raising=False)
    monkeypatch.delenv("VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS", raising=False)
    assert envs.VLLM_EXL3_MOE_V2_DRAFT is False
    assert envs.VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS == 0
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_DRAFT", "1")
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS", "16")
    assert envs.VLLM_EXL3_MOE_V2_DRAFT is True
    assert envs.VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS == 16


def test_gate_opens_for_the_drafter_configurations_only_with_their_switch():
    assert _reason() is None
    closed = {
        "VLLM_EXL3_MOE_V2": _reason(draft=False),
        "VLLM_EXL3_MOE_V2 split": _reason(draft=False, **_split(640)),
        "bits_x2 6": _reason(metadata=_meta(6)),
        "split bits_x2 6": _reason(
            **_split(512, metadata=_meta(6, intermediate=512, local=GLOBAL))
        ),
        "mixed widths": _reason(
            metadata=h5120._meta_with(8, 3, 2, [2304, 5120, 6, 2])[:LOCAL]
        ),
        "codebook": _reason(
            metadata=h5120._meta_with(8, 31, 0, [5120, 2304, 8, 1])[:LOCAL]
        ),
        "top_k": _reason(top_k=6),
        "local": _reason(local_experts=96, metadata=_meta() * 3),
        "global": _reason(global_experts=384),
        "capability": _reason(capability=(8, 6)),
        "dtype": _reason(dtype=torch.float16),
        "entries": _reason(metadata=_meta()[:-1]),
        "split width 576": _reason(**_split(576)),
        "split whole-expert metadata": _reason(
            **_split(640, metadata=_meta(local=GLOBAL))
        ),
        "target layers": _reason(
            top_k=6, local_experts=96, global_experts=384, metadata=h5120._meta(6)
        ),
    }
    for name, reason in closed.items():
        assert reason, name
    assert "local expert 3" in closed["mixed widths"]
    assert "(8, 6)" in closed["capability"]
    # VLLM_EXL3_MOE_V2 alone keeps its configurations and its reasons.
    assert closed["VLLM_EXL3_MOE_V2"].endswith(
        "not one of [(2560, 640, 10, 256, 512), (5120, 512, 6, 384, 384), "
        "(5120, 640, 6, 384, 384), (5120, 2304, 6, 96, 384)]"
    )
    assert h5120._reason() is None
    assert h5120._reason(metadata=h5120._meta(4)) is None
    assert {DRAFTER} == exl3_ops.MOE_V2_DRAFT_CONFIGS
    for shape in exl3_ops.MOE_V2_DRAFT_CONFIGS:
        assert exl3_ops.MOE_V2_CONFIGS[shape] == (BITS_X2,)


@pytest.mark.parametrize("width", [640, 512])
def test_split_drafter_layers_keep_moe_out_with_the_switch(width):
    # Not checked through the drafter's consumer: closed for both switches.
    for draft in (True, False):
        reason = _reason(draft=draft, **_split(width))
        assert reason and f"(5120, {width}, 3, 128, 128)" in reason, (draft, reason)
    assert (HIDDEN, width, TOPK, GLOBAL, GLOBAL) not in exl3_ops.MOE_V2_CONFIGS


def _method(monkeypatch, calls):
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = LOCAL
    method.global_expert_ids = list(range(LOCAL))
    method.swiglu_limit = SWIGLU_LIMIT
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(LOCAL, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(_meta()),
        exl3_expert_map=torch.zeros(GLOBAL, dtype=torch.int32),
        top_k=TOPK,
        global_num_experts=GLOBAL,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        exl3_ops, "moe_workspace_size_v2", lambda rows, *args: 1000 * rows
    )
    monkeypatch.setattr(
        exl3_ops,
        "warmup_moe_v2",
        lambda *args, **kwargs: calls.append(("warmup", args, kwargs)),
    )
    monkeypatch.setattr(
        exl3_ops, "moe_v2_draft_force_rows", lambda rows: calls.append(("floor", rows))
    )
    monkeypatch.setattr(
        exl3_ops, "require_moe_v2_fold", lambda: calls.append(("require fold",))
    )
    return method, layer


@pytest.fixture
def vllm_log(monkeypatch, caplog):
    """Records of quant through the real vLLM logger."""
    import vllm.logger as vllm_logger
    from vllm_sm80.kernels.exl3 import quant

    monkeypatch.setattr(quant, "logger", vllm_logger.init_logger(quant.__name__))
    # The vllm logger does not propagate to the root, where caplog listens.
    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    vllm_logger._print_info_once.cache_clear()
    try:
        yield lambda: sorted({r.getMessage() for r in caplog.records})
    finally:
        root.removeHandler(caplog.handler)


@pytest.mark.parametrize("fold, floor", [(False, 16), (True, 0)])
def test_method_opens_the_drafter_layers_unfolded(monkeypatch, vllm_log, fold, floor):
    calls = []
    method, layer = _method(monkeypatch, calls)
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_FOLD", "1" if fold else "0")
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS", str(floor))
    assert method._moe_v2(layer, _meta(), 40, 7, draft=True) == (40, BITS_X2, 40000)
    assert method._moe_v2(layer, _meta(), 2048, 10**9, draft=True) == (
        64,
        BITS_X2,
        10**9,
    )
    # The floor is set before each warm-up; bits_x2 8, never the fold.
    assert [c[0] for c in calls] == ["floor", "warmup", "floor", "warmup"]
    assert calls[0] == ("floor", floor)
    _, args, kwargs = calls[1]
    assert args[4:9] == (HIDDEN, INTERMEDIATE, TOPK, BITS_X2, SWIGLU_LIMIT)
    assert kwargs == {}
    lines = vllm_log()
    assert (
        "EXL3 MoE v2 drafter kernels active (VLLM_EXL3_MOE_V2_DRAFT): tensor-core "
        "routed experts of the DSpark drafter (bits_x2 8, 16-row tiles; row tile "
        f"floor {floor}) for MoE calls of 1..40 rows; larger calls keep moe_out."
    ) in lines
    fold_line = (
        "VLLM_EXL3_MOE_V2_FOLD does not apply to the DSpark drafter's layers "
        "(VLLM_EXL3_MOE_V2_DRAFT): their moe_out_v2 calls keep five launches."
    )
    assert (fold_line in lines) is fold
    assert not any("kernel active (VLLM_EXL3_MOE_V2)" in m for m in lines)


def test_method_closed_gate_logs_one_line_and_warms_nothing(monkeypatch, vllm_log):
    calls = []
    method, layer = _method(monkeypatch, calls)
    # The drafter's layers without the switch (draft=False: VLLM_EXL3_MOE_V2).
    assert method._moe_v2(layer, _meta(), 40, 7) == (0, 0, 7)
    # The switch on target-shaped layers.
    method.local_experts = 96
    layer.top_k, layer.global_num_experts = 6, 384
    assert method._moe_v2(layer, h5120._meta(6), 40, 7, draft=True) == (0, 0, 7)
    assert calls == []
    lines = vllm_log()
    assert len([m for m in lines if "gate is closed" in m]) == 2
    assert any(
        m.startswith("VLLM_EXL3_MOE_V2 is set but the gate is closed: ") for m in lines
    )
    assert any(
        m.startswith("VLLM_EXL3_MOE_V2_DRAFT is set but the gate is closed: ")
        and "(5120, 2304, 6, 96, 384)" in m
        for m in lines
    )


@pytest.mark.parametrize("rows", [12])
def test_row_tile_floor_refuses_other_values(monkeypatch, rows):
    monkeypatch.setattr(
        exl3_ops, "require_moe_v2_draft", lambda: pytest.fail("checked the value first")
    )
    with pytest.raises(ValueError, match="VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS"):
        exl3_ops.moe_v2_draft_force_rows(rows)


def test_extension_registers_the_row_tile_floor():
    _extension_or_skip()
    exl3_ops.require_moe_v2_draft()
    torch._C._dispatch_find_schema_or_throw("_exl3_C::moe_v2_draft_force_rows", "")


DRAFT_KERNELS = {
    *(
        f"moe_lane_kernel<4, {kt}, 8, DraftLanes<{floor}>>"
        for kt in (20, 18)
        for floor in (4, 8, 16)
    ),
    "route_and_gate_up_group_input<8, 640, 2>",
    "activate_down_slice<16, 4>",
}
# The tensor-parallel shapes' unfolded stages at bits_x2 8 (gate/up and down
# K slices of 20 steps; down slices of 16 at intermediate 512).
SPLIT_KERNELS = {
    "moe_lane_kernel<4, 20, 8, NoFold>",
    "moe_lane_kernel<4, 16, 8, NoFold>",
    "route_and_gate_up_input<8, 2, 5120, 16, false>",
}


def test_sm80_build_has_the_drafter_kernels_and_no_folded_bits_x2_8(sm80_sass):  # noqa: F811
    names = {
        re.sub(r"\s+>", ">", re.sub(r"\(.*", "", n).replace("void ", "")): n
        for n in sm80_sass
    }
    assert set(names) >= DRAFT_KERNELS | SPLIT_KERNELS, sorted(
        (DRAFT_KERNELS | SPLIT_KERNELS) - set(names)
    )
    for name in names:
        bits8 = bool(
            re.match(
                r"(moe_lane_kernel<\d+, \d+, 8\b|route_and_gate_up_input<8\b)", name
            )
        )
        draft = "DraftLanes" in name or "group_input" in name or "down_slice" in name
        # bits_x2 8 kernels are the drafter's (no folded or other stage), and
        # the drafter's stages exist for bits_x2 8 only.
        assert not bits8 or name in DRAFT_KERNELS | SPLIT_KERNELS, name
        assert not draft or name in DRAFT_KERNELS, name
    for name in DRAFT_KERNELS:
        local = [t for _, t in sm80_sass[names[name]] if re.search(r"\b(LDL|STL)\b", t)]
        assert not local, (name, local[:4])


# ---------------------------------------------------------------- GPU tests


class _DraftExperts(h5120._Experts):
    """A drafter layer's local experts (h5120._Experts with the drafter's
    expert counts and intermediate width)."""

    def __init__(self, parts, sets_of, device, cache, intermediate, local):
        super().__init__(parts, sets_of, BITS_X2, device, cache)
        self.intermediate = intermediate
        self.local_experts = local
        self.meta = torch.tensor(
            _meta(intermediate=intermediate, local=local),
            dtype=torch.int64,
            device=device,
        )
        expert_map = torch.full((GLOBAL,), -1, dtype=torch.int32)
        expert_map[:local] = torch.arange(local, dtype=torch.int32)
        self.expert_map = expert_map.to(device)


DISTINCT = 4
UNALIGNED_SET = 3


@pytest.fixture(scope="module", params=sorted(SHAPES))
def synthetic(request):
    _extension_or_skip()
    intermediate, local = SHAPES[request.param]
    device = torch.device("cuda")
    rng = np.random.default_rng(1022 + intermediate)
    parts, owners = [], []
    for index in range(DISTINCT):
        matrices = []
        for k, n in (
            (HIDDEN, intermediate),
            (HIDDEN, intermediate),
            (intermediate, HIDDEN),
        ):
            trellis = rng.integers(
                -(2**15), 2**15, (k // 16, n // 16, 8 * BITS_X2)
            ).astype(np.int16)
            suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)).astype(
                np.float16
            )
            svh = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64).astype(
                np.float16
            )
            base = torch.empty(trellis.size + 1, dtype=torch.int16, device=device)
            offset = 1 if index == UNALIGNED_SET else 0
            t = base[offset : offset + trellis.size].view(trellis.shape)
            t.copy_(torch.from_numpy(trellis))
            owners.append(base)
            matrices.append(
                (t, torch.from_numpy(suh).to(device), torch.from_numpy(svh).to(device))
            )
        parts.append(matrices)
    experts = _DraftExperts(
        parts, [e % DISTINCT for e in range(local)], device, True, intermediate, local
    )
    experts.owners = owners
    experts.name = request.param
    yield experts
    experts.release()


@pytest.fixture
def row_tile_floor():
    """Sets the process's drafter row tile floor; back to 0 afterwards."""
    try:
        yield exl3_ops.moe_v2_draft_force_rows
    finally:
        exl3_ops.moe_v2_draft_force_rows(0)


def _workspace(size_fn, experts, rows, device):
    return torch.full(
        (size_fn(rows, HIDDEN, experts.intermediate, TOPK, experts.local_experts),),
        float("nan"),
        device=device,
    )


def _run(op, experts, x, ids, weights, size_fn, workspace=None):
    out = torch.full_like(x, float("nan"))
    if workspace is None:
        workspace = _workspace(size_fn, experts, x.shape[0], x.device)
    op(
        x,
        ids,
        weights,
        experts.pointers,
        experts.meta,
        experts.expert_map,
        experts.intermediate,
        out,
        workspace,
        swiglu_limit=SWIGLU_LIMIT,
    )
    return out


def _v1(experts, x, ids, weights):
    return _run(exl3_ops.moe_out, experts, x, ids, weights, exl3_ops.moe_workspace_size)


def _v2(experts, x, ids, weights, workspace=None):
    return _run(
        partial(exl3_ops.moe_out_v2, bits_x2=BITS_X2),
        experts,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
        workspace,
    )


def _routes(scenario, rows, generator, local=LOCAL):
    ids = torch.stack(
        [torch.randperm(GLOBAL, generator=generator)[:TOPK] for _ in range(rows)]
    )
    if scenario == "local":  # every route local
        ids = torch.stack(
            [torch.randperm(local, generator=generator)[:TOPK] for _ in range(rows)]
        )
    elif scenario == "masked":  # padding, out of range, another rank's
        ids[:, 0] = -1
        ids[:, 1] = GLOBAL
        ids[:, 2] = LOCAL + 7 if local < GLOBAL else -1
    elif scenario == "nonlocal":
        ids = (
            LOCAL + ids % (GLOBAL - LOCAL) if local < GLOBAL else -torch.ones_like(ids)
        )
    elif scenario == "hot":  # two experts in every row: tiles of 16 rows and a rest
        ids[:, 0] = 5
        ids[:, 1] = 4 * 5 + UNALIGNED_SET
    elif scenario == "single":
        ids[:] = -1
        ids[0, 2] = UNALIGNED_SET
    weights = torch.rand(rows, TOPK, generator=generator) + 0.01
    weights = weights / weights.sum(-1, keepdim=True)
    ids = ids if scenario == "int64" else ids.to(torch.int32)
    return ids, weights


def _case(experts, scenario, rows, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    ids, weights = _routes(scenario, rows, generator, experts.local_experts)
    x = (torch.randn(rows, HIDDEN, generator=generator) * scale).to(torch.bfloat16)
    return x.cuda(), ids.cuda(), weights.cuda()


SCENARIOS = [
    *(("uniform", rows, 1.0) for rows in GRAPH_ROWS),
    *(("local", rows, 1.0) for rows in (1, 2, 4, 5, 8, 9, 16, 17, 33, 64)),
    *(("hot", rows, 1.0) for rows in (5, 13, 16, 40, 64)),
    ("masked", 9, 1.0),
    ("single", 2, 1.0),
    ("int64", 7, 1.0),
    ("uniform", 40, 16.0),  # the gate/up clamp at 10 is reached
    ("local", 64, 16.0),
]


def _tie_mask(cand, inc, ref):
    """Genuine BF16 rounding ties: one ulp from moe_out (same sign) and the
    FP64 value within 1/16 ulp of their midpoint."""
    ci = cand.contiguous().view(torch.int16).to(torch.int32)
    ii = inc.contiguous().view(torch.int16).to(torch.int32)
    adjacent = ((ci - ii).abs() == 1) & ((ci < 0) == (ii < 0))
    c, i = cand.double(), inc.double()
    return adjacent & ((ref - 0.5 * (c + i)).abs() <= (c - i).abs() / 16)


class _TiePool:
    """moe_out_v2's FP64 gate pooled over calls (test_exl3_moe_v2._Fp64Pool)
    with the BF16 tie rule for the max."""

    def __init__(self):
        self.n = 0
        self.c_sum = self.i_sum = 0.0
        self.c_max = self.i_max = 0.0
        self.parts = []

    def add(self, cand, inc, ref, name):
        assert torch.isfinite(cand).all(), name
        ec, ei = (cand.double() - ref).abs(), (inc.double() - ref).abs()
        exact = ei == 0
        assert (ec[exact] == 0).all(), f"{name}: inexact where moe_out is"
        self.n += ec.numel()
        self.c_sum += float(ec.sum())
        self.i_sum += float(ei.sum())
        self.c_max = max(self.c_max, float(ec.max()))
        self.i_max = max(self.i_max, float(ei.max()))
        self.parts.append((ec, _tie_mask(cand, inc, ref)))

    def check(self):
        mean_ratio = self.c_sum / self.i_sum
        max_ratio = self.c_max / self.i_max
        forgiven = 0
        if max_ratio > 1.25:
            budget = 1.25 * self.i_max
            over = sum(int((e > budget).sum()) for e, _ in self.parts)
            forgiven = sum(int(((e > budget) & t).sum()) for e, t in self.parts)
            if over == forgiven and forgiven <= TIE_CAP_FRACTION * self.n:
                rest = max(float(torch.where(t, 0.0, e).max()) for e, t in self.parts)
                max_ratio = rest / self.i_max
        assert mean_ratio <= 1.10, f"mean error x{mean_ratio:.3f} of moe_out"
        assert max_ratio <= 1.25, f"max error x{max_ratio:.3f} of moe_out"
        return mean_ratio, max_ratio, forgiven


def _pool_rule(experts, cases, name):
    refs = real.moe_fp64(
        cases, experts.expert_map, experts.decode, experts.intermediate, SWIGLU_LIMIT
    )
    pool = _TiePool()
    for (x, ids, w), ref in zip(cases, refs):
        pool.add(
            _v2(experts, x, ids, w), _v1(experts, x, ids, w), ref,
            f"{name} rows={x.shape[0]}",
        )  # fmt: skip
    return pool


@needs_sm80
def test_draft_meets_the_fp64_gate_of_moe_out(synthetic, record_property):
    cases = [
        _case(synthetic, scenario, rows, 300 + index, scale)
        for index, (scenario, rows, scale) in enumerate(SCENARIOS)
    ]
    mean, worst, forgiven = _pool_rule(synthetic, cases, synthetic.name).check()
    record_property("mean_ratio", mean)
    record_property("max_ratio", worst)
    record_property("ties_forgiven", forgiven)
    print(
        f"synthetic drafter {synthetic.name} bits_x2=8: mean x{mean:.3f} max "
        f"x{worst:.3f} of moe_out, ties forgiven {forgiven}"
    )


@needs_sm80
def test_draft_nonlocal_rows_are_exact_zero_and_calls_repeat_bitwise(synthetic):
    x, ids, weights = _case(synthetic, "nonlocal", 16, 7)
    assert (_v2(synthetic, x, ids, weights) == 0).all()
    x, ids, weights = _case(synthetic, "hot", 40, 8)
    first = _v2(synthetic, x, ids, weights)
    for _ in range(3):
        _bitwise(_v2(synthetic, x, ids, weights), first, "repeat")
    # One workspace for calls of 40, 5, 13 and 40 rows (sized for 40).
    workspace = _workspace(exl3_ops.moe_workspace_size_v2, synthetic, 40, x.device)
    for rows in (40, 5, 13, 40):
        x, ids, weights = _case(synthetic, "local", rows, 20 + rows)
        _bitwise(
            _v2(synthetic, x, ids, weights, workspace),
            _v2(synthetic, x, ids, weights),
            f"reused workspace, {rows} rows",
        )


@needs_sm80
@pytest.mark.parametrize("floor", [8, 16])
def test_draft_row_tile_floor_makes_rows_independent(synthetic, row_tile_floor, floor):
    if synthetic.name != "ep":
        pytest.skip("the row tile floor applies to the expert-parallel stages")
    row_tile_floor(floor)
    x, ids, weights = _case(synthetic, "hot", 40, 31)
    whole = _v2(synthetic, x, ids, weights)
    for row in (0, 7, 16, 39):
        alone = _v2(
            synthetic, x[row : row + 1], ids[row : row + 1], weights[row : row + 1]
        )
        _bitwise(alone, whole[row : row + 1], f"row {row} alone vs in 40 rows")


@needs_sm80
def test_draft_fold_is_refused(synthetic):
    exl3_ops.require_moe_v2_fold()
    x, ids, weights = _case(synthetic, "uniform", 5, 9)
    with pytest.raises(RuntimeError, match="no folded bits_x2 8"):
        _run(
            partial(exl3_ops.moe_out_v2_fold, bits_x2=BITS_X2),
            synthetic,
            x,
            ids,
            weights,
            exl3_ops.moe_workspace_size_v2,
        )


@needs_sm80
@pytest.mark.parametrize("rows", [5, 16, 40])
def test_draft_graph_replay_is_bitwise_eager_without_allocations(synthetic, rows):
    exl3_ops.warmup_moe_v2(
        synthetic.pointers,
        synthetic.meta,
        synthetic.expert_map,
        list(range(synthetic.local_experts)),
        HIDDEN,
        synthetic.intermediate,
        TOPK,
        BITS_X2,
        SWIGLU_LIMIT,
    )
    x, ids, weights = _case(synthetic, "hot", rows, 11)
    out = torch.empty_like(x)
    workspace = _workspace(exl3_ops.moe_workspace_size_v2, synthetic, rows, x.device)
    args = (
        synthetic.pointers,
        synthetic.meta,
        synthetic.expert_map,
        synthetic.intermediate,
    )

    def call():
        exl3_ops.moe_out_v2(
            x, ids, weights, *args, out, workspace, swiglu_limit=SWIGLU_LIMIT,
            bits_x2=BITS_X2,
        )  # fmt: skip

    torch.cuda.synchronize()
    before = _allocations()
    call()
    torch.cuda.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = _allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert _allocations() - before <= calibration, "capture allocated"

    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, eager, "replay vs eager")

    # New inputs in the captured buffers: the replay recomputes them.
    x2, ids2, weights2 = _case(synthetic, "local", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _v2(synthetic, x2, ids2, weights2), "replay with new inputs")


@needs_sm80
def test_target_operators_return_the_base_trees_bytes(tmp_path):
    base = os.environ.get("VLLM_EXL3_TEST_BASE_TREE")
    if not base:
        pytest.skip("VLLM_EXL3_TEST_BASE_TREE names no tree to compare against")
    _extension_or_skip()
    script = Path(__file__).with_name("exl3_moe_v2_digests.py")
    here = Path(__file__).resolve().parents[3]
    digests = {}
    for name, tree in (("base", Path(base).resolve()), ("this", here)):
        out = tmp_path / f"{name}.json"
        env = dict(os.environ, PYTHONPATH=str(tree))
        subprocess.run(
            [sys.executable, str(script), str(out)], check=True, env=env, cwd=tmp_path
        )
        digests[name] = out.read_text()
    assert digests["base"] == digests["this"]


def _real_layer(layer):
    """The drafter layer's experts as expert-parallel rank 0 holds them, and
    its ffn_norm weight (float32, CUDA)."""
    from safetensors import safe_open

    _extension_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    device = torch.device("cuda")
    parts = []
    for expert in range(LOCAL):
        matrices = []
        for role in ("w1", "w3", "w2"):  # gate, up, down
            name = f"{layer}.ffn.experts.{expert}.{role}"
            matrix = checkpoint.matrix(name)
            assert (matrix.bits_x2, matrix.codebook) == (BITS_X2, 2), name
            matrices.append(tuple(t.to(device) for t in checkpoint.whole(name)))
        parts.append(matrices)
    norm = f"{layer}.ffn_norm.weight"
    header = checkpoint.manifest.tensors[norm]
    with safe_open(header.path, framework="pt") as handle:
        norm_weight = handle.get_tensor(norm).float().to(device)
    experts = _DraftExperts(
        parts, list(range(LOCAL)), device, False, INTERMEDIATE, LOCAL
    )
    return experts, norm_weight


@needs_sm80
@pytest.mark.parametrize("layer", REAL_LAYERS)
def test_draft_holds_moe_outs_error_on_real_layer_weights(layer, record_property):
    experts, norm_weight = _real_layer(layer)
    cases = []
    for rows in GRAPH_ROWS:
        generator = torch.Generator().manual_seed(rows)
        ids, weights = _routes("uniform", rows, generator)
        h = torch.randn(rows, HIDDEN, generator=generator, dtype=torch.float64)
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        x = (h.cuda().float() * norm_weight).to(torch.bfloat16)
        cases.append((x, ids.cuda(), weights.cuda()))
    mean, worst, forgiven = _pool_rule(experts, cases, layer).check()
    record_property("mean_ratio", mean)
    record_property("max_ratio", worst)
    print(
        f"{layer} bits_x2=8 rows {GRAPH_ROWS}: mean x{mean:.3f} max x{worst:.3f} "
        f"of moe_out, ties forgiven {forgiven}"
    )
    experts.release()
