# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE v2 (VLLM_EXL3_MOE_V2, csrc/exl3/moe_v2.cu), hidden-5120
configuration: the routed experts of DeepSeek-V4.1-Flash (EXL3 2.9 bpw) at
expert parallel size 4. Hidden 5120, intermediate 2304, top-6, 96 of 384
whole experts local, every routed matrix of a layer at bits_x2 6 (layers 18-22:
4), codebook 2, SwiGLU clamp 10. test_exl3_moe_v2.py holds the hidden-2560
configuration and the helpers used here.

CPU: the gate opens for both bit widths and closes for other layouts (mixed
widths, the drafter's bits_x2 8, other expert counts, top-k or intermediate
sizes); moe_out_prefill's gate stays closed for this shape;
Exl3MoEMethod._moe_v2 hands the layer's bits_x2 to the warm-up and the
experts; and the workspace contract
(the v2 workspace at 64 rows fits in moe_out's at the served 2048 rows, so the
switch does not grow the MoE workspace).

GPU (SM 8.0), each at bits_x2 6 and 4:
- synthetic experts (random trellis; one of the four weight sets has trellis
  bases that are only 2-byte aligned): moe_out_v2 holds moe_out's error
  against an FP64 reference (test_exl3_moe_v2._Fp64Pool: pooled mean <= 1.10x,
  max <= 1.25x with no element excused, exact where moe_out is exact; the
  ratios are printed and recorded as junit properties) over 1..64 rows and
  awkward routings, with the clamp reached; rows without
  local routes are exact zero; calls repeat bitwise; a row's output is the
  same bits in a call of any size up to 64 rows (one K-slice layout); a
  captured call replays bitwise like the eager one without allocator
  events;
- the real experts of layers 10 (bits_x2 6) and 20 (bits_x2 4) of the
  checkpoint named by VLLM_EXL3_TEST_CHECKPOINT, as expert-parallel rank 0
  holds them (global experts 0..95): the same rule on rows drawn like the
  layer's MoE input (unit-RMS Gaussian rows times the layer's ffn_norm
  weight) with uniform top-6 routes, at the decode and draft-verify row counts
  of REAL_ROWS. One line per layer is printed. Captured served calls are
  checked by tests/models/deepseek_v41/sm80/fp64_cases.py (family exl3_moe).
"""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization.test_exl3_moe_v2 import (
    _allocations,
    _bitwise,
    _extension_or_skip,
    _Fp64Pool,
    needs_sm80,
)
from vllm_sm80.kernels.exl3 import ops as exl3_ops

HIDDEN, INTERMEDIATE, TOPK = 5120, 2304, 6
LOCAL, GLOBAL = 96, 384
SWIGLU_LIMIT = 10.0
BITS = (6, 4)
# Decode rows of 1-8 requests and draft-verify rows (6 per request at depth 5)
# of 1-8 requests, plus the 64-row limit.
REAL_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 18, 24, 30, 36, 42, 48, 54, 64)
REAL_LAYERS = {10: 6, 20: 4}


def _meta(bits_x2):
    return [exl3_ops.moe_v2_expert_meta(HIDDEN, INTERMEDIATE, bits_x2)] * LOCAL


def _meta_with(bits_x2, local, projection, entry):
    metadata = [list(map(list, meta)) for meta in _meta(bits_x2)]
    metadata[local][projection] = entry
    return metadata


def _reason(fn=exl3_ops.moe_v2_closed_reason, **changes):
    args = dict(
        capability=(8, 0),
        dtype=torch.bfloat16,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        local_experts=LOCAL,
        global_experts=GLOBAL,
        metadata=_meta(6),
    )
    args.update(changes)
    return fn(**args)


# ---------------------------------------------------------------- CPU tests


def test_gate_opens_for_both_bit_widths_only():
    assert _reason() is None
    assert _reason(metadata=_meta(4)) is None
    closed = {
        "drafter bits_x2 8": _reason(metadata=_meta(8)),
        "mixed widths": _reason(metadata=_meta_with(6, 40, 2, [2304, 5120, 4, 2])),
        "codebook": _reason(metadata=_meta_with(4, 95, 0, [5120, 2304, 4, 1])),
        "local": _reason(local_experts=128, metadata=_meta(6) + _meta(6)[:32]),
        "global": _reason(global_experts=256),
        "top_k": _reason(top_k=8),
        "intermediate split": _reason(intermediate_size=640),
        "entries": _reason(metadata=_meta(6)[:-1]),
    }
    for name, reason in closed.items():
        assert reason, name
    assert "local expert 40" in closed["mixed widths"]
    assert "bits_x2 8" in closed["drafter bits_x2 8"]


def test_prefill_gate_stays_closed_for_this_shape():
    reason = _reason(fn=exl3_ops.moe_prefill_closed_reason)
    assert reason and "(5120, 2304, 6, 96, 384)" in reason


def test_method_hands_the_layer_bits_to_warmup_and_experts(monkeypatch):
    from vllm_sm80.kernels.exl3.quant import Exl3MoEMethod

    bits_x2 = 4  # not the configuration's first width (6)
    warmed = []
    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = LOCAL
    method.global_expert_ids = list(range(LOCAL))
    method.swiglu_limit = SWIGLU_LIMIT
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(LOCAL, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(_meta(bits_x2)),
        exl3_expert_map=torch.zeros(GLOBAL, dtype=torch.int32),
        top_k=TOPK,
        global_num_experts=GLOBAL,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        exl3_ops, "moe_workspace_size_v2", lambda rows, *args: 1000 * rows
    )
    monkeypatch.setattr(exl3_ops, "warmup_moe_v2", lambda *args: warmed.append(args))
    assert method._moe_v2(layer, _meta(bits_x2), 2048, 7) == (64, bits_x2, 64000)
    (args,) = warmed
    assert args[4:9] == (HIDDEN, INTERMEDIATE, TOPK, bits_x2, SWIGLU_LIMIT)


def _route_ints(assignments):
    """RouteLayout::ints (moe_v2.cu) for a multiple of 4 assignments: header,
    sorted, tiles, experts, by-assignment, tile trellis (int64), tile columns."""
    return 4 + assignments * (1 + 4 + 1 + 1 + 2 * 3 + 3)


def test_workspace_contract():
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_v2
    a = 64 * TOPK
    # Slice partials (gate/up 2 x 16 x 2304, down 8 x 5120), FP16 hi/lo
    # operands (2 x 5120 + 2304 floats), 2^-e per (row, slice), the table, then
    # per row the folded call's counters (one per H128 block) and route count.
    lane = a * (2 * 16 * 2304 + 8 * 5120 + 2 * 5120 + 2304) + a * (2 * 16 + 8)
    fold = 64 * (5120 // 128 + 1)
    assert size(64, HIDDEN, INTERMEDIATE, TOPK, LOCAL) == lane + _route_ints(a) + fold
    assert size(64, HIDDEN, INTERMEDIATE, TOPK, 0) == 0
    # The served moe_out workspace (2048 rows) already holds it.
    v1 = torch.ops._exl3_C.moe_workspace_size(2048, HIDDEN, INTERMEDIATE, TOPK, LOCAL)
    assert size(64, HIDDEN, INTERMEDIATE, TOPK, LOCAL) < v1
    with pytest.raises(RuntimeError, match="hidden 5120 with intermediate 2304"):
        size(1, HIDDEN, 768, TOPK, LOCAL)
    with pytest.raises(RuntimeError, match="1024 routed assignments"):
        size(171, HIDDEN, INTERMEDIATE, TOPK, LOCAL)


# ---------------------------------------------------------------- GPU helpers


def _expert_map(device):
    """Expert-parallel rank 0 of 4: global experts 0..95 are local 0..95."""
    return torch.tensor(
        [g if g < LOCAL else -1 for g in range(GLOBAL)],
        dtype=torch.int32,
        device=device,
    )


class _Experts:
    """One layer's local experts: per local expert (trellis, suh, svh) of gate,
    up and down on the GPU, the pointer table, the metadata, and their FP64
    decode (cached per weight set when ``cache``: synthetic layers share four)."""

    def __init__(self, parts, sets_of, bits_x2, device, cache):
        self.parts = parts  # per weight set: [(trellis, suh, svh)] * 3
        self.sets_of = sets_of  # local expert -> weight set
        self.bits_x2 = bits_x2
        self.cache = cache
        self.pointers = torch.tensor(
            [
                [[t.data_ptr(), s.data_ptr(), v.data_ptr()] for t, s, v in parts[k]]
                for k in sets_of
            ],
            dtype=torch.int64,
            device=device,
        )
        self.meta = torch.tensor(_meta(bits_x2), dtype=torch.int64, device=device)
        self.expert_map = _expert_map(device)
        self._decoded = {}

    def decode(self, local):
        k = self.sets_of[local]
        if k in self._decoded:
            return self._decoded[k]
        decoded = [
            (real.core_weight(t, self.bits_x2, 2), s, v) for t, s, v in self.parts[k]
        ]
        if self.cache:
            self._decoded[k] = decoded
        return decoded

    def release(self):
        self._decoded.clear()


DISTINCT = 4
UNALIGNED_SET = 3


def _synthetic(bits_x2):
    _extension_or_skip()
    device = torch.device("cuda")
    rng = np.random.default_rng(1018 + bits_x2)
    parts, owners = [], []
    for index in range(DISTINCT):
        matrices = []
        for k, n in (
            (HIDDEN, INTERMEDIATE),
            (HIDDEN, INTERMEDIATE),
            (INTERMEDIATE, HIDDEN),
        ):
            trellis = rng.integers(
                -(2**15), 2**15, (k // 16, n // 16, 8 * bits_x2)
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
    experts = _Experts(
        parts, [e % DISTINCT for e in range(LOCAL)], bits_x2, device, cache=True
    )
    experts.owners = owners
    return experts


@pytest.fixture(scope="module", params=BITS, ids=lambda b: f"bits_x2={b}")
def synthetic(request):
    experts = _synthetic(request.param)
    yield experts
    experts.release()


def _run(op, experts, x, ids, weights, size_fn, **kwargs):
    out = torch.full_like(x, float("nan"))
    workspace = torch.full(
        (size_fn(x.shape[0], HIDDEN, INTERMEDIATE, TOPK, LOCAL),),
        float("nan"),
        device=x.device,
    )
    op(
        x,
        ids,
        weights,
        experts.pointers,
        experts.meta,
        experts.expert_map,
        INTERMEDIATE,
        out,
        workspace,
        swiglu_limit=SWIGLU_LIMIT,
        **kwargs,
    )
    return out


def _v1(experts, x, ids, weights):
    return _run(exl3_ops.moe_out, experts, x, ids, weights, exl3_ops.moe_workspace_size)


def _v2(experts, x, ids, weights):
    return _run(
        partial(exl3_ops.moe_out_v2, bits_x2=experts.bits_x2),
        experts,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
    )


def _routes(scenario, rows, generator):
    ids = torch.stack(
        [torch.randperm(GLOBAL, generator=generator)[:TOPK] for _ in range(rows)]
    )
    if scenario == "local":  # every route local: up to 64 x 6 local assignments
        ids = torch.stack(
            [torch.randperm(LOCAL, generator=generator)[:TOPK] for _ in range(rows)]
        )
    elif scenario == "masked":  # padding, out of range, other rank
        ids[:, 0] = -1
        ids[:, 1] = GLOBAL
        ids[:, 2] = LOCAL + 7
    elif scenario == "nonlocal":
        ids = LOCAL + ids % (GLOBAL - LOCAL)
    elif scenario == "hot":  # two experts in every row: several 8-row tiles each
        ids[:, 0] = 5
        ids[:, 1] = 4 * 7 + UNALIGNED_SET
    elif scenario == "single":
        ids[:] = -1
        ids[0, 3] = UNALIGNED_SET
    weights = torch.rand(rows, TOPK, generator=generator) + 0.01
    weights = weights / weights.sum(-1, keepdim=True)
    ids = ids if scenario == "int64" else ids.to(torch.int32)
    return ids, weights


def _case(scenario, rows, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    ids, weights = _routes(scenario, rows, generator)
    x = (torch.randn(rows, HIDDEN, generator=generator) * scale).to(torch.bfloat16)
    return x.cuda(), ids.cuda(), weights.cuda()


SCENARIOS = [
    ("uniform", 1, 1.0),
    ("uniform", 2, 1.0),
    ("uniform", 6, 1.0),
    ("uniform", 8, 1.0),
    ("uniform", 13, 1.0),
    ("uniform", 48, 1.0),
    ("uniform", 53, 1.0),
    ("uniform", 54, 1.0),
    ("uniform", 64, 1.0),
    ("local", 4, 1.0),
    ("local", 48, 1.0),
    ("local", 64, 1.0),
    ("masked", 9, 1.0),
    ("hot", 40, 1.0),
    ("single", 2, 1.0),
    ("int64", 7, 1.0),
    ("uniform", 8, 16.0),  # the gate/up clamp at 10 is reached
    ("local", 64, 16.0),
]


def _pool_rule(experts, cases, name):
    """moe_out_v2 and moe_out on every case, one FP64 reference pass over all
    of them (each local expert decoded once)."""
    calls = [(x, ids, w) for x, ids, w in cases]
    refs = real.moe_fp64(
        calls, experts.expert_map, experts.decode, INTERMEDIATE, SWIGLU_LIMIT
    )
    pool = _Fp64Pool()
    for (x, ids, w), ref in zip(cases, refs):
        pool.add(
            _v2(experts, x, ids, w),
            _v1(experts, x, ids, w),
            ref,
            f"{name} rows={x.shape[0]}",
        )
    return pool


# ---------------------------------------------------------------- GPU tests


@needs_sm80
def test_v2_meets_the_fp64_gate_of_moe_out(synthetic, record_property):
    cases = [
        _case(scenario, rows, 200 + index, scale)
        for index, (scenario, rows, scale) in enumerate(SCENARIOS)
    ]
    mean, worst = _pool_rule(synthetic, cases, f"bits_x2={synthetic.bits_x2}").check()
    record_property("mean_ratio", mean)
    record_property("max_ratio", worst)
    print(
        f"synthetic bits_x2={synthetic.bits_x2}: mean x{mean:.3f} "
        f"max x{worst:.3f} of moe_out"
    )


@needs_sm80
def test_v2_nonlocal_rows_are_exact_zero_and_calls_repeat_bitwise(synthetic):
    x, ids, weights = _case("nonlocal", 16, 7)
    assert (_v2(synthetic, x, ids, weights) == 0).all()
    x, ids, weights = _case("hot", 40, 8)
    first = _v2(synthetic, x, ids, weights)
    for _ in range(3):
        _bitwise(_v2(synthetic, x, ids, weights), first, "repeat")


@needs_sm80
def test_v2_refuses_a_bit_width_without_kernels(synthetic):
    # A width with kernels for this shape but not the layer's would trap on the
    # device (route_block); the gate hands the operator the layer's own.
    x, ids, weights = _case("uniform", 4, 9)
    with pytest.raises(RuntimeError, match="no bits_x2 10 kernels"):
        _run(
            partial(exl3_ops.moe_out_v2, bits_x2=10),
            synthetic,
            x,
            ids,
            weights,
            exl3_ops.moe_workspace_size_v2,
        )


@needs_sm80
def test_v2_rows_do_not_depend_on_the_call_size(synthetic):
    # The first m rows of a 64-row call are bitwise the m-row call, so a CUDA
    # graph padded to a larger captured size gives the same rows.
    x, ids, weights = _case("local", 64, 13)
    whole = _v2(synthetic, x, ids, weights)
    for m in (1, 6, 8, 13, 48, 53, 54):
        part = _v2(synthetic, x[:m], ids[:m], weights[:m])
        _bitwise(part, whole[:m], f"rows {m} of 64")


@needs_sm80
def test_v2_graph_replay_is_bitwise_eager_without_allocations(synthetic):
    rows = 64
    exl3_ops.warmup_moe_v2(
        synthetic.pointers,
        synthetic.meta,
        synthetic.expert_map,
        list(range(LOCAL)),
        HIDDEN,
        INTERMEDIATE,
        TOPK,
        synthetic.bits_x2,
        SWIGLU_LIMIT,
    )
    x, ids, weights = _case("uniform", rows, 11)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_v2(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        device=x.device,
    )
    args = (synthetic.pointers, synthetic.meta, synthetic.expert_map, INTERMEDIATE)

    def call():
        exl3_ops.moe_out_v2(
            x,
            ids,
            weights,
            *args,
            out,
            workspace,
            swiglu_limit=SWIGLU_LIMIT,
            bits_x2=synthetic.bits_x2,
        )

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
    x2, ids2, weights2 = _case("masked", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _v2(synthetic, x2, ids2, weights2), "replay with new inputs")


def _real_layer(layer, bits_x2):
    """The layer's experts as expert-parallel rank 0 holds them, and its
    ffn_norm weight (float32, CUDA)."""
    from safetensors import safe_open

    _extension_or_skip()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    device = torch.device("cuda")
    parts = []
    for expert in range(LOCAL):
        matrices = []
        for role in ("w1", "w3", "w2"):  # gate, up, down
            name = f"layers.{layer}.ffn.experts.{expert}.{role}"
            matrix = checkpoint.matrix(name)
            assert (matrix.bits_x2, matrix.codebook) == (bits_x2, 2), name
            matrices.append(tuple(t.to(device) for t in checkpoint.whole(name)))
        parts.append(matrices)
    norm = f"layers.{layer}.ffn_norm.weight"
    header = checkpoint.manifest.tensors[norm]
    with safe_open(header.path, framework="pt") as handle:
        norm_weight = handle.get_tensor(norm).float().to(device)
    return _Experts(
        parts, list(range(LOCAL)), bits_x2, device, cache=False
    ), norm_weight


@needs_sm80
@pytest.mark.parametrize("layer", sorted(REAL_LAYERS))
def test_v2_holds_moe_outs_error_on_real_layer_weights(layer, record_property):
    experts, norm_weight = _real_layer(layer, REAL_LAYERS[layer])
    cases = []
    for rows in REAL_ROWS:
        generator = torch.Generator().manual_seed(1000 * layer + rows)
        ids, weights = _routes("uniform", rows, generator)
        h = torch.randn(rows, HIDDEN, generator=generator, dtype=torch.float64)
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        x = (h.cuda().float() * norm_weight).to(torch.bfloat16)
        cases.append((x, ids.cuda(), weights.cuda()))
    mean, worst = _pool_rule(experts, cases, f"layer {layer}").check()
    record_property("mean_ratio", mean)
    record_property("max_ratio", worst)
    print(
        f"layer {layer} bits_x2={experts.bits_x2} rows {REAL_ROWS}: "
        f"mean x{mean:.3f} max x{worst:.3f} of moe_out"
    )
    experts.release()
