# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 routed experts split across the tensor-parallel ranks with per-expert
chunk rotation (VLLM_EXL3_MOE_EXPERT_ROTATION): rank r holds chunk
(r + L + e) mod the TP size of expert e in layer L (quant.
tensor_parallel_expert_split) instead of chunk (r + L) mod size of every
expert, so in every layer each rank holds the 640-column chunk of half the
experts and the 512-column chunk of the other half (DeepSeek-V4.1-Flash at
TP4), and its calls take the mixed-width operators (ops.MOE_MIXED_SHAPE).

CPU:
- the switch is declared and off by default; off, the split is unchanged;
- the rotation balances every layer: each rank holds every chunk of an equal
  share of the experts (96 of 384, 32 of 128 per chunk at TP4), so every
  rank holds the same columns per layer, at most today's busiest rank's; each
  expert's ranks tile its intermediate once;
- the method under the switch: every expert's columns, the widest chunk as
  the rank's intermediate, mixed widths;
- through the real loader each rank holds exactly the checkpoint's bytes of
  each expert's own chunk and the ranks reassemble every expert bitwise;
- the gates open for metadata of the two widths only with ``mixed`` and only
  at MOE_MIXED_SHAPE; the wrappers take the ``_mixed`` operators; the experts
  pass ``mixed`` to every operator they call, and never fold;
- the regrouping, emulated: given the same chunk products (the GPU tests prove
  the operators give them bit for bit), both layouts sum every (row, route,
  chunk) term exactly once over the ranks, in the same per-rank combine
  arithmetic; only which terms a rank's partial groups before its BF16
  rounding differs.

GPU (SM 8.0), synthetic layers mixing 640- and 512-column experts in the
rotation's pattern, bits_x2 6 and 4 (target, top-6 of 384) and 8 (drafter,
top-3 of 128): every route's output of moe_out_v2_mixed, moe_out_v2_try_mixed,
moe_out_prefill_try_mixed and moe_out_mixed equals the base operator's at its
expert's width bit for bit (one route per row, the others masked); a call of
experts of one width gives the base operator's bytes; a row alone gives its
bits of the whole call; calls repeat bitwise. On four synthetic ranks of whole
experts, the all-reduced output's FP64 error with the rotation is within 1.10x
(mean) and 1.25x (max) of today's split.

    pytest -q tests/kernels/quantization/test_exl3_expert_rotation.py
"""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.envs as envs
from tests.kernels.quantization import exl3_moe_v2_digests as digests
from tests.kernels.quantization import exl3_real_cases as real
from tests.kernels.quantization import test_exl3_moe_v2 as h2560
from tests.kernels.quantization.test_exl3_expert_split import (
    EXPERTS,
    HIDDEN,
    LAYERS,
    ROLES,
    _build_and_load,
    _check_rank,
    _Layer,
    _moe_config,
    _reassemble,
    _whole,
    synthetic,  # noqa: F401 (the module-scoped synthetic checkpoint fixture)
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm_sm80.kernels.exl3 import ops as exl3_ops
from vllm_sm80.kernels.exl3 import quant
from vllm_sm80.kernels.exl3.utils import block_shard

SWITCH = "VLLM_EXL3_MOE_EXPERT_ROTATION"
WHOLE, SIZE = 2304, 4
TARGET = SimpleNamespace(experts=384, top_k=6, bits=(6, 4))
DRAFTER = SimpleNamespace(experts=128, top_k=3, bits=(8,))
needs_sm80 = h2560.needs_sm80


def _chunk(rank, layer, expert, rotate=True, size=SIZE, whole=WHOLE):
    return block_shard(whole, whole, rank, size, layer + (expert if rotate else 0))


def _width(c):
    return c.stop - c.start


# ---------------------------------------------------------------- CPU tests


def test_switch_declared_and_off_by_default(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert getattr(envs, SWITCH) is False
    monkeypatch.setenv(SWITCH, "1")
    assert getattr(envs, SWITCH) is True


@pytest.mark.parametrize("model", [TARGET, DRAFTER], ids=["target", "drafter"])
def test_rotation_balances_every_layer(model):
    widths = (640, 640, 512, 512)
    for layer in range(43):
        held, today = [], []
        for rank in range(SIZE):
            chunks = [_chunk(rank, layer, e) for e in range(model.experts)]
            index = [(rank + layer + e) % SIZE for e in range(model.experts)]
            # Each chunk of an equal share of the experts.
            assert [index.count(c) for c in range(SIZE)] == [model.experts // SIZE] * SIZE
            assert all(_width(c) == widths[i] for c, i in zip(chunks, index))
            held.append(sum(map(_width, chunks)))
            today.append(model.experts * _width(_chunk(rank, layer, 0, rotate=False)))
            # Within one H128 block per pair of experts of today's columns.
            assert abs(held[-1] - today[-1]) <= 128 * model.experts // 2
        # Every rank the same columns per layer, at most today's busiest rank.
        assert held == [model.experts * WHOLE // SIZE] * SIZE
        assert max(held) <= max(today) and sum(held) == sum(today)
        # Each expert's ranks tile its intermediate once.
        for e in range(model.experts):
            spans = sorted((c.start, c.stop) for c in
                           (_chunk(r, layer, e) for r in range(SIZE)))
            assert spans[0][0] == 0 and spans[-1][1] == WHOLE
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


@pytest.mark.parametrize("layer", [0, 1, 7, 40])
def test_method_holds_each_experts_own_chunk(monkeypatch, layer):
    monkeypatch.setenv(SWITCH, "1")
    for rank in range(SIZE):
        moe = _moe_config(WHOLE, SIZE, rank, hidden=5120, experts=384)
        method = quant.Exl3MoEMethod(None, f"model.layers.{layer}.ffn.experts", moe)
        intermediate, columns = method.expert_split
        assert intermediate == WHOLE
        assert columns == tuple(_chunk(rank, layer, e) for e in range(384))
        assert method.mixed_widths
        assert method.maybe_roundup_sizes(
            5120, 576, torch.bfloat16, moe.moe_parallel_config
        ) == (5120, 640)
    monkeypatch.setenv(SWITCH, "0")
    method = quant.Exl3MoEMethod(
        None, f"model.layers.{layer}.ffn.experts", _moe_config(WHOLE, SIZE, 1, hidden=5120,
                                                                experts=384)
    )
    assert not method.mixed_widths
    assert set(method.expert_split[1]) == {_chunk(1, layer, 0, rotate=False)}


def test_expert_split_widths():
    assert quant.expert_split_widths((slice(0, 640),) * 3) == (640, False)
    assert quant.expert_split_widths((slice(0, 640), slice(640, 1152))) == (640, True)
    with pytest.raises(ValueError, match="widths"):
        quant.expert_split_widths((slice(0, 640), slice(0, 384)))


@pytest.mark.parametrize("contiguous", [False, True])
def test_synthetic_ranks_hold_their_experts_chunks_and_reassemble(
    monkeypatch, synthetic, contiguous  # noqa: F811
):
    config, intermediate, size = synthetic
    monkeypatch.setenv(SWITCH, "1")
    for layer_index in range(LAYERS):
        prefix = f"model.layers.{layer_index}.mlp.experts"
        parts = {}
        for rank in range(size):
            moe = _moe_config(intermediate, size, rank)
            method, layer = _build_and_load(
                monkeypatch, config, prefix, moe, _Layer(EXPERTS), contiguous
            )
            columns = [
                _chunk(rank, layer_index, e, size=size, whole=intermediate)
                for e in range(EXPERTS)
            ]
            assert method.mixed_widths
            assert method.intermediate_size == max(map(_width, columns))
            for expert, weights in enumerate(layer.exl3_experts):
                for role, weight in zip(ROLES, weights):
                    whole = _whole(config.manifest, f"{prefix}.{expert}.{role}")
                    _check_rank(weight, role, whole, columns[expert], HIDDEN)
                    parts.setdefault((expert, role), []).append((columns[expert], weight))
        for (expert, role), matrix_parts in parts.items():
            _reassemble(matrix_parts, role, _whole(config.manifest,
                                                   f"{prefix}.{expert}.{role}"))


def _mixed_meta(widths, bits_x2, hidden=5120):
    return [exl3_ops.moe_v2_expert_meta(hidden, w, bits_x2) for w in widths]


def _rotated_widths(experts, rank=1, layer=2):
    return [_width(_chunk(rank, layer, e)) for e in range(experts)]


@pytest.mark.parametrize("model", [TARGET, DRAFTER], ids=["target", "drafter"])
def test_gates_open_for_two_widths_only_when_mixed(model):
    widths = _rotated_widths(model.experts)
    draft = model is DRAFTER
    gates = [partial(exl3_ops.moe_v2_closed_reason, draft=draft)]
    if not draft:
        gates.append(exl3_ops.moe_prefill_v2_closed_reason)
    for gate in gates:
        for bits_x2 in model.bits:
            args = dict(capability=(8, 0), dtype=torch.bfloat16, hidden_size=5120,
                        intermediate_size=640, top_k=model.top_k,
                        local_experts=model.experts, global_experts=model.experts,
                        metadata=_mixed_meta(widths, bits_x2))
            assert gate(**args, mixed=True) is None
            assert "local expert" in gate(**args)
            # A width other than 640 or 512 stays closed.
            bad = list(widths)
            bad[3] = 384
            assert "local expert 3" in gate(**dict(args, metadata=_mixed_meta(bad, bits_x2)),
                                            mixed=True)
            # Experts of one width keep opening, mixed or not.
            assert gate(**dict(args, metadata=_mixed_meta([640] * model.experts, bits_x2)),
                        mixed=True) is None
            # Two widths only at MOE_MIXED_SHAPE.
            narrow = dict(args, intermediate_size=512,
                          metadata=_mixed_meta([512] * model.experts, bits_x2))
            assert "two widths" in gate(**narrow, mixed=True)


def _fake_library(monkeypatch):
    """torch.ops._exl3_C replaced by recorders: {operator: [args]}."""
    calls = {}

    def recorder(name):
        return lambda *args: calls.setdefault(name, []).append(args)

    names = ("moe_out", "moe_out_mixed", "moe_out_v2", "moe_out_v2_mixed",
             "moe_out_v2_try", "moe_out_v2_try_mixed", "moe_out_prefill_v2",
             "moe_out_prefill_try", "moe_out_prefill_try_mixed")
    library = SimpleNamespace(**{n: recorder(n) for n in names})
    monkeypatch.setattr(torch.ops, "_exl3_C", library, raising=False)
    monkeypatch.setattr(exl3_ops, "ensure_extension", lambda: None)
    return calls


def test_wrappers_take_the_mixed_operators(monkeypatch):
    calls = _fake_library(monkeypatch)
    rows, top_k, experts = 70, 6, 384
    x = torch.zeros(rows, 5120, dtype=torch.bfloat16)
    ids = torch.zeros(rows, top_k, dtype=torch.int32)
    w = torch.zeros(rows, top_k)
    meta = torch.tensor(_mixed_meta(_rotated_widths(experts), 6))
    ptrs = torch.full((experts, 3, 3), 256, dtype=torch.int64)
    emap = torch.arange(experts, dtype=torch.int32)
    args = (x, ids, w, ptrs, meta, emap, 640, torch.empty_like(x), torch.empty(4))
    for mixed in (False, True):
        exl3_ops.moe_out(*args, mixed=mixed)
        exl3_ops.moe_out_v2(*args, bits_x2=6, mixed=mixed)
        exl3_ops.moe_out_v2_try(*args, bits_x2=6, mixed=mixed)
    assert {n: len(c) for n, c in calls.items()} == {
        n: 1 for n in ("moe_out", "moe_out_mixed", "moe_out_v2", "moe_out_v2_mixed",
                       "moe_out_v2_try", "moe_out_v2_try_mixed")
    }
    # Prefill: the unchunked operator's mixed form only.
    monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", True)
    exl3_ops.moe_out_prefill_v2(*args, bits_x2=6, mixed=True)
    assert len(calls["moe_out_prefill_try_mixed"]) == 1
    monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", False)
    with pytest.raises(RuntimeError, match="moe_out_prefill_try_mixed only"):
        exl3_ops.moe_out_prefill_v2(*args, bits_x2=6, mixed=True)
    # A prefill call whose experts are not of the two widths is refused on the host.
    monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", True)
    bad = meta.clone()
    bad[5, :, :2] = torch.tensor([[5120, 384], [5120, 384], [384, 5120]])
    with pytest.raises(RuntimeError, match="local expert 5"):
        exl3_ops.moe_out_prefill_v2(x, ids, w, ptrs, bad, emap, 640, args[7], args[8],
                                    bits_x2=6, mixed=True)


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("rows", [6, 65, 300])
def test_experts_pass_mixed_to_every_operator(monkeypatch, ring, rows):
    seen = []

    def record(name):
        def call(*a, mixed=False, **kw):
            seen.append((name, mixed))

        return call

    for name in ("moe_out", "moe_out_v2", "moe_out_v2_try", "moe_out_v2_fold"):
        monkeypatch.setattr(exl3_ops, name, record(name))
    moe = _moe_config(WHOLE, SIZE, 0, hidden=5120, experts=384)
    identity = torch.arange(384, dtype=torch.int32)
    for mixed in (False, True):
        experts = quant.Exl3Experts(
            moe, FusedMoEQuantConfig.make(), 640, identity, workspace=8, max_rows=512,
            moe_out_rows=128, moe_v2_rows=64, moe_v2_bits_x2=6, moe_v2_try=ring,
            mixed_widths=mixed,
        )
        g = torch.Generator().manual_seed(rows)
        ids = torch.rand(rows, 384, generator=g).topk(6, dim=-1)[1].int()
        weights = torch.rand(rows, 6, generator=g)
        x = torch.zeros(rows, 5120, dtype=torch.bfloat16)
        seen.clear()
        experts.apply(
            torch.empty_like(x), x, None, None, weights, ids, None, 384, identity, None,
            None, None, torch.empty(16, dtype=torch.bfloat16), None, False,
        )
        want = ("moe_out_v2_try" if ring else "moe_out_v2") if rows <= 64 else "moe_out"
        chunks = 1 if rows <= 128 else -(-rows // 128)
        assert seen == [(want, mixed)] * chunks


# The per-rank combine of every operator: FP64 route sum of each route's chunk
# product times its weight, in route order, rounded to FP32 then BF16
# (moe_out's FP32 combine differs only in the accumulator; the same holds).
def _combine(products, weights):
    """products [rows, topk, hidden] FP32, weights [rows, topk] FP32 -> BF16."""
    acc = torch.zeros(products.shape[0], products.shape[2], dtype=torch.float64)
    for k in range(products.shape[1]):
        acc = acc + products[:, k].double() * weights[:, k:k + 1].double()
    return acc.float().bfloat16()


def _all_reduce(partials):
    """The four-rank flag all-reduce: element of partition p summed in FP32 in
    rank order p, p + 1, ... (custom_all_reduce_flags.py), rounded once."""
    stacked = torch.stack([p.float() for p in partials])  # [ranks, rows, hidden]
    flat = stacked.reshape(len(partials), -1)
    n = flat.shape[1]
    out = torch.empty(n, dtype=torch.float32)
    bounds = [n * q // len(partials) for q in range(len(partials) + 1)]
    for p in range(len(partials)):
        lo, hi = bounds[p], bounds[p + 1]
        acc = flat[p, lo:hi].clone()
        for step in range(1, len(partials)):
            acc = acc + flat[(p + step) % len(partials), lo:hi]
        out[lo:hi] = acc
    return out.bfloat16().reshape(partials[0].shape)


def test_rotation_only_regroups_the_same_chunk_terms():
    """Same chunk products (rows x routes x chunks) through both layouts: each
    (row, route, chunk) term reaches the all-reduce exactly once in both; a
    rank's combine is the same arithmetic on another grouping; the FP64 sums
    before any rounding agree to FP64 rounding; the rounded results differ
    only in the last bits, with the same error distribution."""
    g = torch.Generator().manual_seed(1030)
    rows, topk, hidden, experts, layer = 512, 6, 1024, 384, 5
    ids = torch.stack([torch.randperm(experts, generator=g)[:topk] for _ in range(rows)])
    weights = (torch.rand(rows, topk, generator=g) + 0.01)
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    # One FP32 product per (row, route, chunk), the expert's chunks of
    # different scale (as 640- and 512-column chunks of real experts are).
    products = (torch.randn(rows, topk, SIZE, hidden, generator=g)
                * torch.tensor([1.0, 0.9, 0.7, 0.8])[None, None, :, None]).float()

    def chunk_of(rank, rotate):
        return torch.tensor([[(rank + layer + (int(e) if rotate else 0)) % SIZE
                              for e in row] for row in ids])

    terms, exact_sums, outputs = {}, {}, {}
    for rotate in (False, True):
        seen = torch.zeros(rows, topk, SIZE, dtype=torch.int64)
        partials, fp64 = [], torch.zeros(rows, hidden, dtype=torch.float64)
        for rank in range(SIZE):
            c = chunk_of(rank, rotate)
            seen.scatter_add_(2, c[..., None], torch.ones_like(c[..., None]))
            mine = torch.gather(products, 2, c[..., None, None].expand(-1, -1, 1, hidden))[:, :, 0]
            partials.append(_combine(mine, weights))
            fp64 += (mine.double() * weights[..., None].double()).sum(1)
        terms[rotate], exact_sums[rotate] = seen, fp64
        outputs[rotate] = _all_reduce(partials)
    assert torch.equal(terms[False], torch.ones_like(terms[False]))
    assert torch.equal(terms[True], torch.ones_like(terms[True]))
    truth = (products.double() * weights[..., None, None].double()).sum((1, 2))
    for rotate in (False, True):
        assert torch.allclose(exact_sums[rotate], truth, rtol=0, atol=1e-12)
    err = {r: (outputs[r].double() - truth).abs() for r in (False, True)}
    assert not torch.equal(outputs[False], outputs[True])  # the regrouping shows
    assert err[True].mean() <= 1.10 * err[False].mean()
    assert err[True].max() <= 1.25 * err[False].max()


# ---------------------------------------------------------------- GPU tests


def _require():
    h2560._extension_or_skip()
    try:
        exl3_ops.require_moe_mixed()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _merge(narrow, wide, wide_mask):
    """A layer's pointer or meta table: each expert from the 640-column layer
    when wide, else from the 512-column one."""
    mask = wide_mask.view(-1, *([1] * (wide.dim() - 1))).to(wide.device)
    return torch.where(mask, wide, narrow)


@pytest.fixture(
    scope="module",
    params=[("target", 6), ("target", 4), ("drafter", 8)],
    ids=lambda p: f"{p[0]}-bits_x2={p[1]}",
)
def mixed_layer(request):
    """Synthetic 640- and 512-column layers of every expert (the identity
    map) and their merge in the rotation's pattern of rank 1, layer 2."""
    _require()
    name, bits_x2 = request.param
    model = TARGET if name == "target" else DRAFTER
    layers = {
        w: digests._layer(5120, w, model.experts, bits_x2, 1030 + w + bits_x2, "cuda")
        for w in (640, 512)
    }
    wide = torch.tensor([w == 640 for w in _rotated_widths(model.experts)])
    pointers = _merge(layers[512][1], layers[640][1], wide)
    meta = _merge(layers[512][2], layers[640][2], wide)
    yield SimpleNamespace(
        name=name, model=model, bits_x2=bits_x2, layers=layers, wide=wide.cuda(),
        pointers=pointers, meta=meta,
        expert_map=torch.arange(model.experts, dtype=torch.int32, device="cuda"),
    )


# family -> (operator, workspace size, rows to test, prefill)
def _families(layer):
    v2 = (exl3_ops.moe_out_v2, exl3_ops.moe_workspace_size_v2, (1, 6, 13, 48, 64))
    out = {"v2": v2, "moe_out": (exl3_ops.moe_out, exl3_ops.moe_workspace_size,
                                 (6, 65, 200))}
    if layer.name == "target":
        out["ring"] = (exl3_ops.moe_out_v2_try, exl3_ops.moe_workspace_size_v2_try,
                       (1, 6, 13, 48, 64))
        out["prefill"] = (None, None, (65, 517, 1025, 2048))
    return out


def _call(layer, family, x, ids, w, width, pointers, meta, mixed):
    out = torch.full_like(x, float("nan"))
    top_k, experts = ids.shape[1], layer.model.experts
    if family == "prefill":
        args = (x.shape[0], 5120, width, top_k, experts)
        size = torch.ops._exl3_C.moe_workspace_size_prefill_try(*args)
        op = getattr(torch.ops._exl3_C,
                     "moe_out_prefill_try_mixed" if mixed else "moe_out_prefill_try")
        op(x, ids, w, pointers, meta, layer.expert_map, width, layer.bits_x2, out,
           torch.full((size,), float("nan"), device="cuda"), digests.SWIGLU_LIMIT)
        return out
    fn, size_fn, _ = _families(layer)[family]
    workspace = torch.full((size_fn(x.shape[0], 5120, width, top_k, experts),),
                           float("nan"), device="cuda")
    extra = {} if family == "moe_out" else {"bits_x2": layer.bits_x2}
    fn(x, ids, w, pointers, meta, layer.expert_map, width, out, workspace,
       swiglu_limit=digests.SWIGLU_LIMIT, mixed=mixed, **extra)
    return out


def _mixed(layer, family, x, ids, w):
    return _call(layer, family, x, ids, w, 640, layer.pointers, layer.meta, True)


def _native(layer, family, width, x, ids, w):
    _, pointers, meta = layer.layers[width]
    return _call(layer, family, x, ids, w, width, pointers, meta, False)


def _case(layer, rows, seed, scenario="uniform"):
    model = layer.model
    return digests._case(scenario, rows, model.top_k, model.experts, model.experts, 5120,
                         seed)


def _families_of(layer):
    if layer.bits_x2 == 8 and layer.name == "drafter":
        return ("v2", "moe_out")
    return tuple(_families(layer))


@needs_sm80
def test_every_route_is_the_base_operator_at_its_experts_width(mixed_layer):
    layer = mixed_layer
    for family in _families_of(layer):
        for rows in _families(layer)[family][2]:
            for scenario in ("uniform", "hot"):
                x, ids, w = _case(layer, rows, 7 * rows + len(family), scenario)
                for k in range(ids.shape[1]):
                    only = torch.full_like(ids, -1)
                    only[:, k] = ids[:, k]
                    got = _mixed(layer, family, x, only, w)
                    wide_row = layer.wide[ids[:, k].long()]
                    for width, select in ((640, wide_row), (512, ~wide_row)):
                        masked = torch.where(select[:, None], only, torch.full_like(only, -1))
                        want = _native(layer, family, width, x, masked, w)
                        h2560._bitwise(got[select], want[select],
                                       f"{family} rows={rows} {scenario} route {k} w{width}")
                    assert not torch.isnan(got).any(), (family, rows, k)


@needs_sm80
def test_experts_of_one_width_give_the_base_operators_bytes(mixed_layer):
    layer = mixed_layer
    wide = torch.nonzero(layer.wide).flatten()
    narrow = torch.nonzero(~layer.wide).flatten()
    for family in _families_of(layer):
        rows = _families(layer)[family][2][-1]
        x, ids, w = _case(layer, rows, 11 * rows)
        for width, pool in ((640, wide), (512, narrow)):
            one = pool[ids.long() % pool.numel()].to(ids.dtype)
            h2560._bitwise(_mixed(layer, family, x, one, w),
                           _native(layer, family, width, x, one, w),
                           f"{family} rows={rows} all {width}")


@needs_sm80
def test_mixed_rows_stand_alone_and_calls_repeat(mixed_layer):
    layer = mixed_layer
    for family in _families_of(layer):
        rows = _families(layer)[family][2][-1]
        x, ids, w = _case(layer, rows, 5 * rows + 1)
        whole = _mixed(layer, family, x, ids, w)
        h2560._bitwise(_mixed(layer, family, x, ids, w), whole, f"{family} repeat")
        if family == "prefill":
            lo = exl3_ops.MOE_PREFILL_MIN_ROWS
            subs = [slice(0, lo), slice(rows - lo, rows)]
        else:
            subs = [slice(i, i + 1) for i in sorted({0, rows // 2, rows - 1})]
        for s in subs:
            h2560._bitwise(_mixed(layer, family, x[s], ids[s], w[s]), whole[s],
                           f"{family} rows {s}")


def _whole_experts(bits_x2, experts, seed):
    """Synthetic whole 2304-column experts (two weight sets) and each rank's
    view of them: {chunk: (owners, pointers, meta)} for the four chunks."""
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(2):
        parts = []
        for k, n in ((5120, WHOLE), (5120, WHOLE), (WHOLE, 5120)):
            trellis = rng.integers(-(2**15), 2**15, (k // 16, n // 16, 8 * bits_x2))
            suh = rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)
            svh = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64
            parts.append((torch.from_numpy(trellis.astype(np.int16)).cuda(),
                          torch.from_numpy(suh.astype(np.float16)).cuda(),
                          torch.from_numpy(svh.astype(np.float16)).cuda()))
        sets.append(parts)
    chunks = {}
    for index in range(SIZE):
        c = block_shard(WHOLE, WHOLE, index, SIZE, 0)
        owners, ptrs, meta = [], [], []
        for parts in sets:
            sliced = []
            for role, (t, s, v) in zip(ROLES, parts):
                if role == "down_proj":
                    sliced.append((t[c.start // 16:c.stop // 16].contiguous(),
                                   s[c.start:c.stop].contiguous(), v))
                else:
                    sliced.append((t[:, c.start // 16:c.stop // 16].contiguous(), s,
                                   v[c.start:c.stop].contiguous()))
            owners.append(sliced)
            ptrs.append([[p.data_ptr() for p in tensors] for tensors in sliced])
            meta.append([[5120, _width(c), bits_x2, 2], [5120, _width(c), bits_x2, 2],
                         [_width(c), 5120, bits_x2, 2]])
        chunks[index] = (owners, ptrs, meta)

    decoded = {}

    def decode(expert):
        if expert % 2 not in decoded:
            decoded[expert % 2] = [(real.core_weight(t, bits_x2, 2), s, v)
                                   for t, s, v in sets[expert % 2]]
        return decoded[expert % 2]

    return chunks, decode


@needs_sm80
@pytest.mark.parametrize("family,bits_x2,rows", [
    ("ring", 6, 48), ("ring", 4, 64), ("prefill", 6, 517), ("moe_out", 8, 65),
])
def test_all_reduced_fp64_error_is_no_worse_than_todays_split(family, bits_x2, rows):
    """Four ranks of the same whole experts, today's split (rank r: chunk
    (r + L) mod 4 of every expert) against the rotation (chunk (r + L + e) mod 4
    of expert e), each rank on its operator, the partials summed as the
    four-rank all-reduce does: the rotation's error against the whole experts'
    FP64 output within 1.10x (mean) and 1.25x (max) of today's."""
    _require()
    model = DRAFTER if bits_x2 == 8 else TARGET
    layer_index = 3
    chunks, decode = _whole_experts(bits_x2, model.experts, 1030 + bits_x2)
    g = torch.Generator().manual_seed(rows)
    ids = torch.stack([torch.randperm(model.experts, generator=g)[:model.top_k]
                       for _ in range(rows)]).int().cuda()
    w = torch.rand(rows, model.top_k, generator=g) + 0.01
    w = (w / w.sum(-1, keepdim=True)).cuda()
    x = torch.randn(rows, 5120, generator=g).bfloat16().cuda()
    layer = SimpleNamespace(name="target" if model is TARGET else "drafter", model=model,
                            bits_x2=bits_x2,
                            expert_map=torch.arange(model.experts, dtype=torch.int32,
                                                    device="cuda"))
    truth = real.moe_fp64([(x, ids, w)], layer.expert_map.cpu(), decode, WHOLE,
                          digests.SWIGLU_LIMIT)[0]
    errors = {}
    for rotate in (False, True):
        partials = []
        for rank in range(SIZE):
            index = [(rank + layer_index + (e if rotate else 0)) % SIZE
                     for e in range(model.experts)]
            ptrs = torch.tensor([chunks[i][1][e % 2] for e, i in enumerate(index)],
                                dtype=torch.int64, device="cuda")
            meta = torch.tensor([chunks[i][2][e % 2] for e, i in enumerate(index)],
                                dtype=torch.int64, device="cuda")
            widths = {int(m) for m in meta[:, 0, 1].tolist()}
            partials.append(_call(layer, family, x, ids, w, max(widths), ptrs, meta,
                                  len(widths) > 1).cpu())
        out = _all_reduce(partials)
        errors[rotate] = (out.double() - truth.cpu()).abs()
    mean = {r: float(e.mean()) for r, e in errors.items()}
    worst = {r: float(e.max()) for r, e in errors.items()}
    print(f"{family} bits_x2={bits_x2} rows={rows}: today mean {mean[False]:.4g} max "
          f"{worst[False]:.4g}, rotation mean {mean[True]:.4g} max {worst[True]:.4g}")
    assert mean[True] <= 1.10 * mean[False]
    assert worst[True] <= 1.25 * worst[False]
