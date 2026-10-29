# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE v2 (VLLM_EXL3_MOE_V2, csrc/exl3/moe_v2.cu), hidden-2560 configuration
(test_exl3_moe_v2_hidden5120.py holds the hidden-5120 one).

CPU: the hardware, matrix and layout/partition gates, bounded workspace,
and row dispatch in Exl3Experts.apply with the switch off and on.

GPU (SM 8.0 only): at the served shape (hidden 2560, intermediate 640, top-10,
256 local experts at TP2 or all 512 at PP2, bits_x2 10 / codebook 2), the off
path is bitwise moe_out and the scalar reference; moe_out_v2 is held to moe_out's
error against
an FP64 reference (pooled mean <= 1.10x, max <= 1.25x with no element excused,
exact where moe_out is exact) over awkward routings; it is
deterministic; and a captured call replays bitwise like the eager call
without any allocator event.
"""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sm80.kernels.exl3 import ops as exl3_ops

needs_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="EXL3 kernels are built for SM 8.0",
)

HIDDEN, INTERMEDIATE, TOPK, BITS_X2 = 2560, 640, 10, 10
LOCAL, GLOBAL = 256, 512
LOCAL_ALL = GLOBAL  # tensor parallel size 1: every global expert is local
SERVED_META = [[[2560, 640, 10, 2], [2560, 640, 10, 2], [640, 2560, 10, 2]]] * LOCAL


def _extension_or_skip():
    try:
        exl3_ops.ensure_extension()
    except RuntimeError as exc:
        pytest.skip(f"vllm._exl3_C unavailable: {exc}")


# ---------------------------------------------------------------- CPU tests


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_MOE_V2", raising=False)
    assert envs.VLLM_EXL3_MOE_V2 is False
    monkeypatch.setenv("VLLM_EXL3_MOE_V2", "1")
    assert envs.VLLM_EXL3_MOE_V2 is True


def _reason(**changes):
    args = dict(
        capability=(8, 0),
        dtype=torch.bfloat16,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        local_experts=LOCAL,
        global_experts=GLOBAL,
        metadata=SERVED_META,
    )
    args.update(changes)
    return exl3_ops.moe_v2_closed_reason(**args)


def _meta_with(local, projection, entry):
    metadata = [list(map(list, meta)) for meta in SERVED_META]
    metadata[local][projection] = entry
    return metadata


def test_gate_opens_only_for_the_served_configuration():
    assert _reason() is None
    closed = {
        "capability": _reason(capability=(8, 6)),
        "no capability": _reason(capability=None),
        "dtype": _reason(dtype=torch.float16),
        "hidden": _reason(hidden_size=4096),
        "intermediate": _reason(intermediate_size=512),
        "top_k": _reason(top_k=8),
        "local": _reason(local_experts=128, metadata=SERVED_META[:128]),
        "global": _reason(global_experts=256),
        "entries": _reason(metadata=SERVED_META[:-1]),
        "rate": _reason(metadata=_meta_with(7, 1, [2560, 640, 8, 2])),
        "codebook": _reason(metadata=_meta_with(255, 2, [640, 2560, 10, 1])),
        "padding": _reason(metadata=_meta_with(0, 0, [2688, 640, 10, 2])),
    }
    for name, reason in closed.items():
        assert reason, name
    assert "local expert 7" in closed["rate"]
    assert "(8, 6)" in closed["capability"]


@pytest.mark.parametrize(
    "local, layout, eligible",
    [
        (LOCAL, (2, 1), True),
        (LOCAL_ALL, (1, 2), True),
        (LOCAL, (1, 2), False),
        (LOCAL_ALL, (2, 1), False),
        (LOCAL_ALL, (1, 1), False),
        (LOCAL_ALL, (2, 2), False),
        (LOCAL_ALL, (1, 4), False),
        (128, (4, 1), False),
        (LOCAL_ALL, None, False),
        (LOCAL, None, False),
    ],
)
def test_decode_gate_pairs_the_expert_partition_with_its_layout(
    monkeypatch, local, layout, eligible
):
    warmed: list[tuple] = []
    method, layer = _method(monkeypatch, warmed, local=local, layout=layout)
    result = method._moe_v2(layer, layer.exl3_meta.tolist(), 16, 7)
    assert result == ((16, BITS_X2, 16000) if eligible else (0, 0, 7))
    assert bool(warmed) is eligible


def test_pp2_decode_gate_checks_the_last_local_expert():
    metadata = [
        [list(projection) for projection in SERVED_META[0]] for _ in range(LOCAL_ALL)
    ]
    assert _reason(local_experts=LOCAL_ALL, metadata=metadata) is None
    metadata[-1][2][2] = 8
    assert _reason(local_experts=LOCAL_ALL, metadata=metadata) is not None


def _method(monkeypatch, warmed, local=LOCAL, layout=(2, 1)):
    from vllm.sm80.kernels.exl3 import quant
    from vllm.sm80.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = local
    method.global_expert_ids = (
        list(range(GLOBAL)) if local == LOCAL_ALL else list(range(0, GLOBAL, 2))
    )
    method.swiglu_limit = 0.0
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(local, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(SERVED_META[:1] * local),
        exl3_expert_map=torch.zeros(GLOBAL, dtype=torch.int32),
        top_k=TOPK,
        global_num_experts=GLOBAL,
    )
    config = (
        SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=layout[0], pipeline_parallel_size=layout[1]
            )
        )
        if layout is not None
        else None
    )
    monkeypatch.setattr(quant, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        exl3_ops, "moe_workspace_size_v2", lambda rows, *args: 1000 * rows
    )
    monkeypatch.setattr(exl3_ops, "warmup_moe_v2", lambda *args: warmed.append(args))
    return method, layer


def test_gate_caps_rows_preserves_larger_workspace_and_rejects_invalid_weights(
    monkeypatch,
):
    warmed: list[tuple] = []
    method, layer = _method(monkeypatch, warmed)
    assert method._moe_v2(layer, SERVED_META, 3200, 7) == (64, 10, 64000)
    assert method._moe_v2(layer, SERVED_META, 3200, 10**9) == (64, 10, 10**9)
    assert method._moe_v2(layer, SERVED_META, 16, 7) == (16, 10, 16000)
    assert len(warmed) == 3
    wrong = _meta_with(3, 0, [2560, 640, 8, 2])
    assert method._moe_v2(layer, wrong, 3200, 7) == (0, 0, 7)
    assert len(warmed) == 3


def _experts(moe_v2_rows, max_rows=4096, workspace=0, expert_map="layer map"):
    """Exl3Experts without its constructor. The CPU tests keep a placeholder
    expert map, which they only compare; GPU tests pass the real one."""
    from vllm.sm80.kernels.exl3.quant import Exl3Experts

    experts = object.__new__(Exl3Experts)
    experts.intermediate_size = INTERMEDIATE
    experts.expert_map = expert_map
    experts.workspace_elements = workspace
    experts.max_rows = max_rows
    experts.moe_v2_rows = moe_v2_rows
    experts.moe_out_rows = max_rows
    experts.moe_v2_bits_x2 = BITS_X2
    experts.prefill_decode_len = 0
    experts.prefill_v2_bits_x2 = 0
    experts.swiglu_limit = 0.0
    return experts


def _apply(experts, x, ids, weights, ptrs, meta, out, workspace2):
    experts.apply(
        out,
        x,
        ptrs,
        meta,
        weights,
        ids,
        None,
        GLOBAL,
        None,
        None,
        None,
        None,
        workspace2,
        None,
        False,
    )


@pytest.mark.parametrize("moe_v2_rows", [0, 64])
def test_apply_dispatches_by_rows(monkeypatch, moe_v2_rows):
    calls = []
    for name in ("moe_out", "moe_out_v2"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append(
                (_name, args[0].shape[0], args, kwargs)
            ),
        )
    experts = _experts(moe_v2_rows)
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    for rows in (1, 4, 63, 64, 65, 3200):
        x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        _apply(experts, x, "ids", "weights", "ptrs", "meta", "out", workspace2)
    expected = [
        ("moe_out_v2" if moe_v2_rows and rows <= 64 else "moe_out", rows)
        for rows in (1, 4, 63, 64, 65, 3200)
    ]
    assert [(name, rows) for name, rows, _, _ in calls] == expected
    for name, _, args, kwargs in calls:
        # Same arguments for both operators: layer map, FP32 view of workspace2.
        assert args[1:6] == ("ids", "weights", "ptrs", "meta", "layer map")
        assert args[6] == INTERMEDIATE and args[7] == "out"
        assert args[8].dtype == torch.float32
        # moe_out_v2 also gets the layer's bits_x2.
        expected_kwargs = {"swiglu_limit": 0.0}
        if name == "moe_out_v2":
            expected_kwargs["bits_x2"] = BITS_X2
        assert kwargs == expected_kwargs


def test_operator_list_and_workspace_contract():
    assert {"moe_out_v2", "moe_workspace_size_v2"} <= set(exl3_ops._OPERATORS)
    _extension_or_skip()
    size = torch.ops._exl3_C.moe_workspace_size_v2
    # 640 assignments: slice partials (gate/up 2 x 8 x 640, down 2 x 2560),
    # FP16 hi/lo operands (2 x 2560 + 640 floats), 2^-e per (row, slice)
    # rounded to 4, then the routing table (header, sorted, tiles, the expert
    # of each sorted entry and of each assignment, then per row tile and
    # projection the int64 trellis address and the tile columns), then per row
    # the folded call's counters (one per H128 block) and local-route count.
    lane = 640 * (2 * 8 * 640 + 2 * 2560 + 2 * 2560 + 640) + 640 * 18
    table = 4 + 640 + 4 * 640 + 640 + 640 + 2 * 3 * 640 + 3 * 640
    fold = 64 * (2560 // 128 + 1)
    assert size(64, HIDDEN, INTERMEDIATE, TOPK, LOCAL) == lane + table + fold
    assert size(1, 2500, 600, TOPK, LOCAL) == size(1, HIDDEN, INTERMEDIATE, TOPK, LOCAL)
    assert size(64, HIDDEN, INTERMEDIATE, TOPK, 0) == 0
    with pytest.raises(RuntimeError, match="hidden 2560"):
        size(1, 4096, INTERMEDIATE, TOPK, LOCAL)
    with pytest.raises(RuntimeError, match="1024 routed assignments"):
        size(103, HIDDEN, INTERMEDIATE, TOPK, LOCAL)


# ---------------------------------------------------------------- GPU tests

DISTINCT = 4  # distinct weight sets; local expert e uses set e % DISTINCT
UNALIGNED_SET = 3  # its trellis bases are only 2-byte aligned


@pytest.fixture(scope="module", params=[LOCAL, LOCAL_ALL], ids=["tp2", "pp2"])
def served(request):
    """Both expert partitions over distinct weight sets and FP64 references."""
    return served_experts(UNALIGNED_SET, local=request.param)


def served_experts(unaligned_set, local=LOCAL):
    """The ``served`` expert sets with ``local`` local experts on one rank; set
    ``unaligned_set`` (None: no set) has trellis bases that are only 2-byte
    aligned. ``mapping[g]`` is the local expert of global expert g (-1: another
    rank's); the served partitions are LOCAL experts holding the even globals
    (tensor parallel size 2) and LOCAL_ALL holding every global (tensor
    parallel size 1, where pipeline parallelism splits the layers)."""
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _extension_or_skip()
    if local == LOCAL_ALL:
        mapping = list(range(GLOBAL))
    elif local == LOCAL:
        mapping = [g // 2 if g % 2 == 0 else -1 for g in range(GLOBAL)]
    else:
        raise ValueError(f"no served expert partition has {local} local experts")
    device = torch.device("cuda")
    rng = np.random.default_rng(1013)
    owners, sets, dense = [], [], []
    for index in range(DISTINCT):
        ptrs, mats = [], []
        for k, n in (
            (HIDDEN, INTERMEDIATE),
            (HIDDEN, INTERMEDIATE),
            (INTERMEDIATE, HIDDEN),
        ):
            trellis = rng.integers(-(2**15), 2**15, (k // 16, n // 16, 80)).astype(
                np.int16
            )
            suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)).astype(
                np.float16
            )
            svh = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64).astype(
                np.float16
            )
            base = torch.empty(trellis.size + 1, dtype=torch.int16, device=device)
            offset = 1 if index == unaligned_set else 0
            t = base[offset : offset + trellis.size].view(trellis.shape)
            t.copy_(torch.from_numpy(trellis))
            s, v = torch.from_numpy(suh).to(device), torch.from_numpy(svh).to(device)
            owners.append((base, s, v))
            ptrs.append([t.data_ptr(), s.data_ptr(), v.data_ptr()])
            mats.append(effective_weight(trellis, suh, svh, 10, 2))
        sets.append(ptrs)
        dense.append([torch.from_numpy(m).to(device) for m in mats])
    pointers = torch.tensor(
        [sets[e % DISTINCT] for e in range(local)], dtype=torch.int64, device=device
    )
    meta = torch.tensor(
        [list(map(list, SERVED_META[0]))] * local, dtype=torch.int64, device=device
    )
    expert_map = torch.tensor(mapping, dtype=torch.int32, device=device)
    return SimpleNamespace(
        owners=owners,
        pointers=pointers,
        meta=meta,
        expert_map=expert_map,
        dense=dense,
        mapping=mapping,
    )


def _routes(scenario, rows, generator):
    ids = torch.randint(0, GLOBAL, (rows, TOPK), generator=generator)
    if scenario == "masked":  # padding, out of range, other rank
        ids[:, 0] = -1
        ids[:, 1] = GLOBAL
        ids[:, 2] = 1
    elif scenario == "nonlocal":
        ids = ids | 1
    elif scenario == "hot":  # one expert gets every row (40+ rows of one tile set)
        ids[:, 0] = 6
        ids[:, 1] = 8 + 2 * DISTINCT
    elif scenario == "single":
        ids[:] = -1
        ids[0, 3] = 2 * UNALIGNED_SET
    elif scenario == "int64":
        pass
    elif scenario == "all_experts":
        ids = torch.arange(rows * TOPK).reshape(rows, TOPK) % GLOBAL
    elif scenario == "upper_half":
        ids = torch.randint(LOCAL, GLOBAL, (rows, TOPK), generator=generator)
    weights = torch.rand(rows, TOPK, generator=generator) + 0.01
    weights = weights / weights.sum(-1, keepdim=True)
    ids = ids if scenario == "int64" else ids.to(torch.int32)
    return ids, weights


def _case(served, scenario, rows, seed):
    generator = torch.Generator().manual_seed(seed)
    ids, weights = _routes(scenario, rows, generator)
    x = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
    return x.cuda(), ids.cuda(), weights.cuda()


def _run(op, served, x, ids, weights, size_fn):
    local = served.pointers.shape[0]
    out = torch.full_like(x, float("nan"))
    workspace = torch.full(
        (size_fn(x.shape[0], HIDDEN, INTERMEDIATE, TOPK, local),),
        float("nan"),
        device=x.device,
    )
    op(
        x,
        ids,
        weights,
        served.pointers,
        served.meta,
        served.expert_map,
        INTERMEDIATE,
        out,
        workspace,
    )
    return out


def _v1(served, x, ids, weights):
    return _run(exl3_ops.moe_out, served, x, ids, weights, exl3_ops.moe_workspace_size)


def _v2(served, x, ids, weights):
    return _run(
        partial(exl3_ops.moe_out_v2, bits_x2=BITS_X2),
        served,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
    )


def _reference(served, x, ids, weights):
    """FP64: out[t] = sum_j w[t, j] down(SiLU(gate x_t) * up x_t), local routes."""
    x64 = x.double()
    out = torch.zeros_like(x64)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            g = int(ids[t, j])
            if not 0 <= g < GLOBAL or served.mapping[g] < 0:
                continue
            gate, up, down = served.dense[served.mapping[g] % DISTINCT]
            a = x64[t] @ gate
            act = a / (1.0 + torch.exp(-a)) * (x64[t] @ up)
            out[t] += float(weights[t, j]) * (act @ down)
    return out


def _bitwise(a, b, name):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name


@needs_sm80
@pytest.mark.parametrize("rows", [1, 4, 32, 64, 65])
def test_off_path_is_moe_out_and_the_scalar_reference(served, rows):
    x, ids, weights = _case(served, "uniform", rows, rows)
    expected = _v1(served, x, ids, weights)
    reference = _run(
        torch.ops._exl3_C.moe_out_reference,
        served,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size,
    )
    _bitwise(expected, reference, f"moe_out vs scalar reference rows={rows}")
    for moe_v2_rows in (0, 64) if rows > 64 else (0,):
        size = exl3_ops.moe_workspace_size(
            rows, HIDDEN, INTERMEDIATE, TOPK, served.pointers.shape[0]
        )
        experts = _experts(moe_v2_rows, workspace=size, expert_map=served.expert_map)
        out = torch.full_like(x, float("nan"))
        workspace2 = torch.empty(2 * size, dtype=torch.bfloat16, device=x.device)
        _apply(experts, x, ids, weights, served.pointers, served.meta, out, workspace2)
        _bitwise(out, expected, f"off path rows={rows} moe_v2_rows={moe_v2_rows}")


SCENARIOS = [
    ("uniform", 1),
    ("uniform", 4),
    ("uniform", 8),
    ("uniform", 16),
    ("uniform", 32),
    ("uniform", 64),
    ("uniform", 3),
    ("uniform", 5),
    ("uniform", 13),
    ("uniform", 33),
    ("uniform", 63),
    ("masked", 9),
    ("hot", 48),
    ("single", 2),
    ("int64", 7),
    ("all_experts", 64),
    ("upper_half", 32),
]


class _Fp64Pool:
    """moe_out_v2's FP64 gate, pooled over calls: mean error <= 1.10x and max
    error <= 1.25x moe_out's (every element counts; no tolerance floor), exact
    where moe_out is exact, all finite."""

    def __init__(self):
        self.n = 0
        self.c_sum = self.i_sum = 0.0
        self.c_max = self.i_max = 0.0

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

    def ratios(self):
        return self.c_sum / self.i_sum, self.c_max / self.i_max

    def check(self, name="moe_out_v2"):
        """Assert the gate; return (mean ratio, max ratio)."""
        mean_ratio, max_ratio = self.ratios()
        assert mean_ratio <= 1.10, f"{name}: mean error x{mean_ratio:.3f} of moe_out"
        assert max_ratio <= 1.25, f"{name}: max error x{max_ratio:.3f} of moe_out"
        return mean_ratio, max_ratio


@needs_sm80
def test_v2_meets_the_fp64_gate_of_moe_out(served):
    pool = _Fp64Pool()
    for index, (scenario, rows) in enumerate(SCENARIOS):
        x, ids, weights = _case(served, scenario, rows, 100 + index)
        cand, inc = _v2(served, x, ids, weights), _v1(served, x, ids, weights)
        ref = _reference(served, x, ids, weights)
        pool.add(cand, inc, ref, f"{scenario} rows={rows}")
    pool.check()


@needs_sm80
def test_v2_rows_without_local_routes_are_exact_zero_and_repeat_bitwise(served):
    x, ids, weights = _case(served, "nonlocal", 16, 7)
    if served.pointers.shape[0] == LOCAL_ALL:
        ids.fill_(-1)  # PP2 has no remote experts; mask every route instead.
    assert (_v2(served, x, ids, weights) == 0).all()
    x, ids, weights = _case(served, "hot", 40, 8)
    first = _v2(served, x, ids, weights)
    for _ in range(3):
        _bitwise(_v2(served, x, ids, weights), first, "repeat")


@needs_sm80
def test_v2_refuses_shapes_outside_its_configuration(served):
    x, ids, weights = _case(served, "uniform", 4, 9)
    out = torch.empty_like(x)
    workspace = torch.empty(1 << 24, device=x.device)
    with pytest.raises(RuntimeError, match="hidden 2560"):
        exl3_ops.moe_out_v2(
            x,
            ids,
            weights,
            served.pointers,
            served.meta,
            served.expert_map,
            512,
            out,
            workspace,
            bits_x2=BITS_X2,
        )
    with pytest.raises(RuntimeError, match="1024 routed assignments"):
        big = x.new_zeros(103, HIDDEN)
        exl3_ops.moe_out_v2(
            big,
            ids.new_zeros(103, TOPK),
            weights.new_zeros(103, TOPK),
            served.pointers,
            served.meta,
            served.expert_map,
            INTERMEDIATE,
            torch.empty_like(big),
            workspace,
            bits_x2=BITS_X2,
        )
    with pytest.raises(RuntimeError, match="no bits_x2 6 kernels"):
        exl3_ops.moe_out_v2(
            x,
            ids,
            weights,
            served.pointers,
            served.meta,
            served.expert_map,
            INTERMEDIATE,
            out,
            workspace,
            bits_x2=6,
        )


def _allocations():
    return torch.accelerator.memory_stats()["allocation.all.allocated"]


@needs_sm80
@pytest.mark.parametrize("rows", [4, 32])  # short and long K slices
def test_v2_graph_replay_is_bitwise_eager_without_allocations(served, rows):
    exl3_ops.warmup_moe_v2(
        served.pointers,
        served.meta,
        served.expert_map,
        [g for g, local in enumerate(served.mapping) if local >= 0],
        HIDDEN,
        INTERMEDIATE,
        TOPK,
        BITS_X2,
    )
    x, ids, weights = _case(served, "uniform", rows, 11)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_v2(
            rows, HIDDEN, INTERMEDIATE, TOPK, served.pointers.shape[0]
        ),
        device=x.device,
    )
    args = (served.pointers, served.meta, served.expert_map, INTERMEDIATE)

    def call():
        exl3_ops.moe_out_v2(x, ids, weights, *args, out, workspace, bits_x2=BITS_X2)

    torch.accelerator.synchronize()
    before = _allocations()
    call()
    torch.accelerator.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
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
    torch.accelerator.synchronize()
    _bitwise(out, eager, "replay vs eager")

    # New inputs in the captured buffers: the replay recomputes them.
    x2, ids2, weights2 = _case(served, "masked", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.accelerator.synchronize()
    _bitwise(out, _v2(served, x2, ids2, weights2), "replay with new inputs")
