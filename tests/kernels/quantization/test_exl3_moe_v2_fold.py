# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE v2 folded (VLLM_EXL3_MOE_V2_FOLD, csrc/exl3/moe_v2.cu FoldArgs):
moe_out_v2_fold runs the activation at the end of the gate/up lane launch and
the combine at the end of the down lane launch (completion counters), three
launches instead of five, and must return moe_out_v2's output bits.

CPU:
- the switch defaults off; Exl3Experts.apply takes moe_out_v2_fold for the
  calls moe_out_v2 serves only when the layer's experts are folded;
  Exl3MoEMethod._moe_v2 requires the operator and warms both arms;
- an exact emulation of the completion counting (groups_over_block and the
  expected counts) over the route tiling: every (row tile, H128 block),
  row tile and (token, H128 block) counter is completed by exactly one item,
  under random completion orders, for every launch configuration of every
  open configuration (its top-k and local experts);
- the workspace at the split shapes holds the folded call's counters, and its
  gate/up scratch fits the down partials' region;
- the extension registers moe_out_v2_fold;
- moe_v2.cu compiled for sm_80 (nvcc, no GPU): the fold instantiations of
  every shape and bit width LaneTiling lists and the folded route launches
  exist, and no lane kernel's GEMM loop touches local memory.

GPU (SM 8.0): moe_out_v2_fold's output bytes equal moe_out_v2's at hidden
5120 (bits_x2 6 and 4: the DeepSeek-V4.1 target layers; its drafter's
bits_x2 8 experts take moe_out, which the switch does not change) and
hidden 2560 (bits_x2 10, short and long K slices), for every row count 1..48
and 64 over routings with rows without local routes, hot experts (several
8-row tiles), masked and int64 ids and the reached SwiGLU clamp, with NaN in
the output and workspace before each call; calls repeat bitwise; a captured
call replays bitwise without allocator events; and the real experts of
layers 10 and 20 when VLLM_EXL3_TEST_CHECKPOINT names the checkpoint. The
same at the tensor-split shapes, for every configuration the gate opens with
every expert local (synthetic experts with one 2-byte aligned weight set, the
identity expert map, rows 1..48 and 64, the same routings with padding and
out-of-range ids in place of other ranks' experts), and on the real experts
of layers 10 and 20 as rank 0 of the split holds them. The DSpark drafter's
configurations (MOE_V2_DRAFT_CONFIGS, bits_x2 8) are never folded: no fold
kernel is built for them, and they are left out here.
"""

import os
import random
import re
import shutil
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.kernels.quantization import test_exl3_moe_v2 as h2560
from tests.kernels.quantization import test_exl3_moe_v2_hidden5120 as h5120
from vllm_sm80.kernels.exl3 import ops as exl3_ops

needs_sm80 = h2560.needs_sm80
ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------- CPU tests


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_MOE_V2_FOLD", raising=False)
    assert envs.VLLM_EXL3_MOE_V2_FOLD is False
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_FOLD", "1")
    assert envs.VLLM_EXL3_MOE_V2_FOLD is True


def test_apply_takes_the_folded_operator_for_the_v2_calls(monkeypatch):
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_v2_fold"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append(
                (_name, args[0].shape[0], kwargs)
            ),
        )
    experts = h2560._experts(64)
    experts.moe_v2_fold = True
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    for rows in (1, 48, 64, 65):
        x = torch.zeros(rows, h2560.HIDDEN, dtype=torch.bfloat16)
        h2560._apply(experts, x, "ids", "weights", "ptrs", "meta", "out", workspace2)
    assert [(name, rows) for name, rows, _ in calls] == [
        ("moe_out_v2_fold", 1),
        ("moe_out_v2_fold", 48),
        ("moe_out_v2_fold", 64),
        ("moe_out", 65),
    ]
    for name, _, kwargs in calls[:3]:
        assert kwargs == {"swiglu_limit": 0.0, "bits_x2": h2560.BITS_X2}


def test_method_requires_and_warms_the_folded_operator(monkeypatch):
    warmed, required = [], []
    method, layer = h2560._method(monkeypatch, warmed)
    monkeypatch.setattr(
        exl3_ops,
        "warmup_moe_v2",
        lambda *args, **kwargs: warmed.append((args, kwargs)),
    )
    monkeypatch.setattr(exl3_ops, "require_moe_v2_fold", lambda: required.append(1))
    monkeypatch.delenv("VLLM_EXL3_MOE_V2_FOLD", raising=False)
    assert method._moe_v2(layer, h2560.SERVED_META, 3200, 7) == (64, 10, 64000)
    assert [kwargs for _, kwargs in warmed] == [{}] and not required
    warmed.clear()
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_FOLD", "1")
    assert method._moe_v2(layer, h2560.SERVED_META, 3200, 7) == (64, 10, 64000)
    # Both arms warm (a dev A/B server switches between them), fold last.
    assert [kwargs for _, kwargs in warmed] == [{}, {"fold": True}]
    assert warmed[0][0] == warmed[1][0] and required == [1]


# The launch configurations moe_v2.cu builds, read from its LaneTiling
# specializations: per (hidden, intermediate) the column tiles, the gate/up and
# down K slice tiles (short and long), the scale groups, the bits_x2 list.
_LONG_NEVER = "std::numeric_limits<int64_t>::max()"


def _lane_tilings(source: str) -> dict:
    tilings = {}
    for h, i, body in re.findall(
        r"struct LaneTiling<(\d+), (\d+)> \{(.*?)\n\};", source, re.S
    ):
        values = {}
        for name, value in re.findall(r"constexpr \w+ (k\w+) = ([^;]+);", body):
            value = value.strip()
            values[name] = (
                None if value == _LONG_NEVER
                else values[value] if value in values else int(value)
            )  # fmt: skip
        bits = re.search(r"integer_sequence<int, ([\d, ]+)>", body).group(1)
        values["bits"] = tuple(int(b) for b in bits.split(","))
        tilings[(int(h), int(i))] = values
    return tilings


TILINGS = _lane_tilings((ROOT / "csrc" / "exl3" / "moe_v2.cu").read_text())
# Tensor-parallel routed experts (every expert local): each rank's 640- or
# 512-column chunk of the 2304-column intermediate.
SPLIT_SHAPES = ((5120, 640), (5120, 512))


def _launches(hidden, intermediate):
    """(column tiles, gate/up slice tiles, gate/up slices, down slice tiles,
    down slices) of every launch configuration of the shape: short K slices,
    and long ones where a call can reach them."""
    t = TILINGS[(hidden, intermediate)]
    configs = [("Short", "Short")]
    if t["kLongSliceAssignments"] is not None:
        configs.append(("Long", "Long"))
    out = []
    for gate_up, down in configs:
        gate_up_tiles = t[f"kGateUp{gate_up}SliceTiles"]
        down_tiles = t[f"kDown{down}SliceTiles"]
        out.append(
            (
                t["kColumnTiles"],
                gate_up_tiles,
                hidden // (16 * gate_up_tiles),
                down_tiles,
                intermediate // (16 * down_tiles),
            )
        )
    return out


def _v2_configs():
    """The MOE_V2_CONFIGS entries the fold serves, as (hidden, intermediate,
    top_k, local, global, bits): not the DSpark drafter's
    (MOE_V2_DRAFT_CONFIGS), whose calls are never folded."""
    draft = getattr(exl3_ops, "MOE_V2_DRAFT_CONFIGS", frozenset())
    return [
        (h, i, k, local, total, bits)
        for (h, i, k, local, total), widths in exl3_ops.MOE_V2_CONFIGS.items()
        if (h, i, k, local, total) not in draft
        for bits in widths
    ]


def _folded_bits(hidden, intermediate):
    """The bits_x2 the fold serves at a shape (its fold kernels are built for
    these only)."""
    shape = (hidden, intermediate)
    return sorted({b for h, i, *_, b in _v2_configs() if (h, i) == shape})


# Completion counting, mirrored from moe_v2.cu (FoldGateUp, FoldDown,
# fold_activate, groups_over_block), for every launch configuration of every
# shape the gate opens (local experts and top-k from its configuration).
COUNTING = sorted(
    {
        (h, i, k, local, *launch)
        for h, i, k, local, _, _ in _v2_configs()
        for launch in _launches(h, i)
    }
)


def _groups_over_block(block, group_tiles, n_tiles):
    return min(8 * block + 7, n_tiles - 1) // group_tiles - 8 * block // group_tiles + 1


def _items(n_tiles, group_tiles, projections, slices):
    """Per item of one row tile: the H128 blocks it counts in."""
    for _ in range(projections * slices):
        for n0 in range(0, n_tiles, group_tiles):
            yield list(range(n0 // 8, (n0 + group_tiles - 1) // 8 + 1))


def _last_once(counts, expected, order):
    """Count `order` (counter keys) atomically: the arrivals that see the
    expected count minus one, per counter."""
    seen, last = Counter(), Counter()
    for key in order:
        if seen[key] == expected[key] - 1:
            last[key] += 1
        seen[key] += 1
    assert seen == counts
    return last


@pytest.mark.parametrize("shape", COUNTING, ids=lambda s: "x".join(map(str, s)))
def test_completion_counters_fire_once_per_block(shape):
    hidden, intermediate, topk, local, group_tiles = shape[:5]
    gate_up_slices, down_slices = shape[6], shape[8]
    rows = 48
    rng = random.Random(hash(shape) & 0xFFFF)
    # Gate/up: one row tile's items over the intermediate's blocks.
    n_tiles = intermediate // 16
    row_blocks = intermediate // 128
    order = [b for item in _items(n_tiles, group_tiles, 2, gate_up_slices) for b in item]
    counts = Counter(order)
    expected = {
        b: 2 * gate_up_slices * _groups_over_block(b, group_tiles, n_tiles)
        for b in range(row_blocks)
    }
    assert counts == Counter(expected)
    rng.shuffle(order)
    assert _last_once(counts, expected, order) == Counter(range(row_blocks))
    # Down: route tiles of up to 8 rows per local expert (route_block), items
    # counting (token, block) per row; ids outside 0..local-1 are not local.
    ids = [[rng.randrange(-1, 2 * local) for _ in range(topk)] for _ in range(rows)]
    ids[0] = [local + 1] * topk  # a token without a local route
    ids[1] = [3] * topk  # one expert's tile from one token
    by_expert = {}
    for a in range(rows * topk):
        e = ids[a // topk][a % topk]
        if 0 <= e < local:
            by_expert.setdefault(e, []).append(a)
    tiles = [
        members[i : i + 8]
        for _, members in sorted(by_expert.items())
        for i in range(0, len(members), 8)
    ]
    n_tiles = hidden // 16
    routes = Counter(a // topk for tile in tiles for a in tile)
    order = [
        (a // topk, b)
        for tile in tiles
        for item in _items(n_tiles, group_tiles, 1, down_slices)
        for a in tile
        for b in item
    ]
    counts = Counter(order)
    expected = {
        (t, b): routes[t] * down_slices * _groups_over_block(b, group_tiles, n_tiles)
        for t in routes
        for b in range(hidden // 128)
    }
    assert counts == Counter(expected)
    rng.shuffle(order)
    assert _last_once(counts, expected, order) == Counter(expected.keys())
    assert 0 not in routes  # its row is the route launch's exact zero


def _route_table_ints(a):
    """RouteLayout::ints (moe_v2.cu) for `a` assignments."""
    experts = 4 + (a + 3) // 4 * 4 + 4 * a
    tile_trellis = (experts + a + a + 1) // 2 * 2
    return tile_trellis + 2 * 3 * a + 3 * a


@pytest.mark.parametrize("shape", SPLIT_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_split_workspace_holds_the_fold_counters(shape):
    """moe_workspace_size_v2 at a split shape: the slice partials, operands,
    2^-e, the routing table, then one int per (row, H128 block of the hidden
    size) plus one per row for the folded call; the folded gate/up stage's
    counters, values and peaks fit in the down partials' region."""
    h2560._extension_or_skip()
    hidden, intermediate = shape
    configs = [c for c in _v2_configs() if c[:2] == shape and c[3] == c[4]]
    _, _, topk, experts, _, _ = configs[0]
    ((nt, _, gate_up_slices, _, down_slices),) = _launches(hidden, intermediate)
    row_blocks = intermediate // 128
    size = torch.ops._exl3_C.moe_workspace_size_v2
    for rows in range(1, 65):
        a = rows * topk
        partials = 2 * gate_up_slices * intermediate + down_slices * hidden
        operands = 2 * hidden + intermediate
        unscale = (a * (2 * gate_up_slices + down_slices) + 3) // 4 * 4
        lane = a * (partials + operands) + unscale
        table = _route_table_ints(a)
        fold = rows * (hidden // 128 + 1)
        got = size(rows, hidden, intermediate, topk, experts)
        assert got == lane + table + fold, rows
        counters = (a * (row_blocks + 1) + 3) // 4 * 4
        scratch = counters + a * intermediate + a * row_blocks
        assert scratch <= down_slices * a * hidden, rows


def test_extension_registers_the_folded_operator():
    h2560._extension_or_skip()
    exl3_ops.require_moe_v2_fold()
    torch._C._dispatch_find_schema_or_throw("_exl3_C::moe_out_v2_fold", "")


# The lane kernels of every shape moe_v2.cu builds: unfolded at every bits_x2
# LaneTiling lists, folded at the bits_x2 the fold serves there; and the
# folded route launches.
LANE_KERNELS = {
    f"moe_lane_kernel<{nt}, {kt}, {bits}, NoFold>"
    for (h, i), tiling in TILINGS.items()
    for nt, gate_up_kt, _, down_kt, _ in _launches(h, i)
    for bits in tiling["bits"]
    for kt in (gate_up_kt, down_kt)
}
FOLD_KERNELS = {
    f"moe_lane_kernel<{nt}, {kt}, {bits}{fold}>"
    for (h, i) in TILINGS
    for nt, gate_up_kt, gate_up_slices, down_kt, down_slices in _launches(h, i)
    for bits in _folded_bits(h, i)
    for kt, fold in (
        (gate_up_kt, f", FoldGateUp<{gate_up_slices}, {i // 128}>"),
        (down_kt, f", FoldDown<{down_slices}>"),
    )
}
LANE_KERNELS |= FOLD_KERNELS
ROUTE_KERNELS = {
    f"route_and_gate_up_input<{bits}, {h // 128 // 20}, {scale_k}, "
    f"{scale_k // (16 * tiles)}, true>"
    for (h, i), tiling in TILINGS.items()
    for bits in _folded_bits(h, i)
    for scale_k, tiles in {
        (tiling["kGateUpShortScaleK"], tiling["kGateUpShortSliceTiles"]),
        (tiling["kGateUpLongScaleK"], tiling["kGateUpLongSliceTiles"]),
    }
}


@pytest.fixture(scope="module")
def sm80_sass():
    """moe_v2.cu as an sm_80 cubin (build_standalone.py's CUDA flags), SASS
    per demangled kernel name."""
    from torch.utils import cpp_extension

    home = cpp_extension.CUDA_HOME
    nvcc = home and Path(home, "bin", "nvcc")
    cuobjdump = home and Path(home, "bin", "cuobjdump")
    if not (nvcc and nvcc.exists() and cuobjdump.exists() and shutil.which("c++filt")):
        pytest.skip("needs the CUDA toolkit (nvcc, cuobjdump) and c++filt")
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    cubin = tmp / f"moe_v2_sm80_{os.getpid()}.cubin"
    includes = []
    for path in cpp_extension.include_paths(device_type="cuda"):
        includes += ["-isystem", path]
    import sysconfig

    command = [
        str(nvcc), "-cubin", "-arch=sm_80", "-O3", "-std=c++20", "-DNDEBUG",
        "-DUSE_CUDA", "-DPy_LIMITED_API=3", "-DTORCH_EXTENSION_NAME=_exl3_C",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-I", str(ROOT / "csrc"), "-I", str(ROOT / "csrc" / "exl3"), *includes,
        "-isystem", sysconfig.get_paths()["include"],
        "-o", str(cubin), str(ROOT / "csrc" / "exl3" / "moe_v2.cu"),
    ]  # fmt: skip
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        sass = subprocess.run(
            [str(cuobjdump), "-sass", str(cubin)],
            check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"{exc.cmd[0]} failed:\n{exc.stderr[-4000:]}")
    finally:
        cubin.unlink(missing_ok=True)
    functions, name = {}, None
    mangled = re.findall(r"Function : (\S+)", sass)
    demangled = subprocess.run(
        ["c++filt"], input="\n".join(mangled), capture_output=True, text=True,
        check=True,
    ).stdout.split("\n")  # fmt: skip
    names = dict(zip(mangled, demangled))
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = names[m.group(1)].replace("exl3::moe_v2::(anonymous namespace)::", "")
            functions[name] = []
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
        if name and m:
            functions[name].append((int(m.group(1), 16), m.group(2)))
    return functions


def _gemm_loops(instructions):
    """The innermost loops (backward branches) that hold an HMMA."""
    loops = []
    for address, text in instructions:
        m = re.search(r"\bBRA\b.*?0x([0-9a-f]+)\s*$", text)
        if m and int(m.group(1), 16) < address:
            begin = int(m.group(1), 16)
            body = [t for a, t in instructions if begin <= a <= address]
            if any(t.startswith("HMMA") or " HMMA" in t for t in body):
                loops.append((begin, address, body))
    return [
        (b, e, body)
        for b, e, body in loops
        if not any(b <= b2 and e2 <= e and (b2, e2) != (b, e) for b2, e2, _ in loops)
    ]


def test_sm80_build_has_every_fold_kernel_and_spill_free_gemm_loops(sm80_sass):
    lane = {}
    for name, instructions in sm80_sass.items():
        if "moe_lane_kernel<" in name:
            key = re.sub(r"\(.*", "", name).replace("void ", "")
            lane[re.sub(r"\s+>", ">", key)] = instructions
    assert LANE_KERNELS <= set(lane), sorted(LANE_KERNELS - set(lane))
    # No fold kernel beyond the bits_x2 the fold serves (the drafter's are
    # never folded).
    folded = {name for name in lane if ", Fold" in name}
    assert folded == FOLD_KERNELS, sorted(folded ^ FOLD_KERNELS)
    routes = {
        re.sub(r"\(.*", "", n).replace("void ", "")
        for n in sm80_sass
        if "route_and_gate_up_input<" in n and ", true>" in n
    }
    assert routes == ROUTE_KERNELS, (sorted(routes), sorted(ROUTE_KERNELS))
    # The kernel body's GEMM loops (lane_tiles inlined: aligned trellis
    # bases, every served layer) stay free of local memory with the fold
    # epilogue; out-of-line callees (lane_tiles_unaligned, the fold stages:
    # from a CALL target to its RET) are not the body.
    for name in LANE_KERNELS:
        instructions = lane[name]
        # (A call to the next instruction is a convergence idiom, not a callee.)
        targets = {
            int(m.group(1), 16)
            for address, text in instructions
            if (m := re.search(r"\bCALL\.\S*\s+0x([0-9a-f]+)", text))
            and int(m.group(1), 16) > address + 0x20
        }
        callees = [
            (t, next(a for a, x in instructions if a >= t and x.startswith("RET")))
            for t in targets
        ]
        loops = [
            (b, e, body)
            for b, e, body in _gemm_loops(instructions)
            if not any(t <= b and e <= r for t, r in callees)
        ]
        assert len(loops) >= 2, name  # one per row-group count (1 and 2)
        for begin, end, body in loops:
            local = [t for t in body if re.search(r"\b(LDL|STL)\b", t)]
            assert not local, (name, hex(begin), hex(end), local[:4])

# ---------------------------------------------------------------- GPU tests

ROWS = (*range(1, 49), 64)


def _fold_5120(experts, x, ids, weights):
    return h5120._run(
        partial(exl3_ops.moe_out_v2_fold, bits_x2=experts.bits_x2),
        experts,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
    )


def _fold_2560(served, x, ids, weights):
    return h2560._run(
        partial(exl3_ops.moe_out_v2_fold, bits_x2=h2560.BITS_X2),
        served,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
    )


@pytest.fixture(scope="module", params=h5120.BITS, ids=lambda b: f"bits_x2={b}")
def synthetic(request):
    experts = h5120._synthetic(request.param)
    exl3_ops.require_moe_v2_fold()
    yield experts
    experts.release()


@pytest.fixture(scope="module")
def served():
    served = h2560.served_experts(h2560.UNALIGNED_SET)
    exl3_ops.require_moe_v2_fold()
    return served


SCENARIOS_5120 = ["uniform", "local", "masked", "nonlocal", "hot", "single", "int64"]
SCENARIOS_2560 = ["uniform", "masked", "nonlocal", "hot", "single", "int64"]


@needs_sm80
@pytest.mark.parametrize("scenario", SCENARIOS_5120)
def test_fold_is_bitwise_v2_hidden5120(synthetic, scenario):
    for rows in ROWS:
        for scale in (1.0, 16.0) if scenario in ("uniform", "local") else (1.0,):
            x, ids, w = h5120._case(scenario, rows, 7 * rows + 1, scale)
            v2 = h5120._v2(synthetic, x, ids, w)
            fold = _fold_5120(synthetic, x, ids, w)
            h2560._bitwise(fold, v2, f"{scenario} rows={rows} scale={scale}")
            assert not torch.isnan(fold).any()


@needs_sm80
@pytest.mark.parametrize("scenario", SCENARIOS_2560)
def test_fold_is_bitwise_v2_hidden2560(served, scenario):
    # 1..15 rows: short K slices (< 160 routed assignments); 16..64: long.
    for rows in ROWS:
        x, ids, w = h2560._case(served, scenario, rows, 7 * rows + 2)
        v2 = h2560._v2(served, x, ids, w)
        fold = _fold_2560(served, x, ids, w)
        h2560._bitwise(fold, v2, f"{scenario} rows={rows}")
        assert not torch.isnan(fold).any()


@needs_sm80
def test_fold_repeats_bitwise_with_a_reused_workspace(synthetic):
    """The counters are zeroed per call: a workspace left by calls of other
    sizes (and garbage) gives the same bits."""
    rows = 48
    workspace = torch.full(
        (exl3_ops.moe_workspace_size_v2(64, h5120.HIDDEN, h5120.INTERMEDIATE,
                                        h5120.TOPK, h5120.LOCAL),),
        float("nan"),
        device="cuda",
    )  # fmt: skip
    first = None
    for count in (rows, 64, 1, rows, rows):
        x, ids, w = h5120._case("uniform", count, 99 if count == rows else count)
        out = torch.full_like(x, float("nan"))
        exl3_ops.moe_out_v2_fold(
            x, ids, w, synthetic.pointers, synthetic.meta, synthetic.expert_map,
            h5120.INTERMEDIATE, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
            bits_x2=synthetic.bits_x2,
        )  # fmt: skip
        if count == rows:
            if first is None:
                first = out.clone()
                h2560._bitwise(out, h5120._v2(synthetic, x, ids, w), "fold vs v2")
            h2560._bitwise(out, first, "repeat")
        workspace.view(torch.int32)[::7] = -12345  # garbage between calls


@needs_sm80
def test_fold_graph_replay_is_bitwise_without_allocations(synthetic):
    rows = 48
    exl3_ops.warmup_moe_v2(
        synthetic.pointers,
        synthetic.meta,
        synthetic.expert_map,
        list(range(h5120.LOCAL)),
        h5120.HIDDEN,
        h5120.INTERMEDIATE,
        h5120.TOPK,
        synthetic.bits_x2,
        h5120.SWIGLU_LIMIT,
        fold=True,
    )
    x, ids, weights = h5120._case("uniform", rows, 11)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_v2(
            rows, h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK, h5120.LOCAL
        ),
        device=x.device,
    )

    def call():
        exl3_ops.moe_out_v2_fold(
            x, ids, weights, synthetic.pointers, synthetic.meta, synthetic.expert_map,
            h5120.INTERMEDIATE, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
            bits_x2=synthetic.bits_x2,
        )  # fmt: skip

    torch.cuda.synchronize()
    before = h2560._allocations()
    call()
    torch.cuda.synchronize()
    assert h2560._allocations() == before, "eager call allocated"
    h2560._bitwise(out, h5120._v2(synthetic, x, ids, weights), "eager fold vs v2")
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = h2560._allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert h2560._allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, eager, "replay vs eager")
    x2, ids2, weights2 = h5120._case("masked", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, h5120._v2(synthetic, x2, ids2, weights2), "replay new inputs")


@needs_sm80
@pytest.mark.parametrize("layer", sorted(h5120.REAL_LAYERS))
def test_fold_is_bitwise_v2_on_real_layer_weights(layer):
    experts, norm_weight = h5120._real_layer(layer, h5120.REAL_LAYERS[layer])
    exl3_ops.require_moe_v2_fold()
    for rows in ROWS:
        generator = torch.Generator().manual_seed(1000 * layer + rows)
        ids, weights = h5120._routes("uniform", rows, generator)
        h = torch.randn(rows, h5120.HIDDEN, generator=generator, dtype=torch.float64)
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        x = (h.cuda().float() * norm_weight).to(torch.bfloat16)
        ids, weights = ids.cuda(), weights.cuda()
        h2560._bitwise(
            _fold_5120(experts, x, ids, weights),
            h5120._v2(experts, x, ids, weights),
            f"layer {layer} rows={rows}",
        )
    experts.release()


# ------------------------------------------------- GPU: tensor-split shapes

# Every configuration the gate opens with every expert local (the routed
# experts split across the tensor-parallel ranks): (intermediate, top_k,
# experts, bits_x2).
SPLIT_CONFIGS = sorted(
    (i, k, local, bits)
    for h, i, k, local, total, bits in _v2_configs()
    if h == 5120 and local == total
)


class _SplitLayer:
    """One rank's experts at a split shape: per expert (trellis, suh, svh) of
    gate, up (the rank's columns) and down (its rows) on the GPU, the pointer
    table, the metadata and the identity expert map."""

    def __init__(self, parts, sets_of, intermediate, topk, experts, bits_x2):
        device = torch.device("cuda")
        self.parts = parts
        self.intermediate, self.topk, self.experts = intermediate, topk, experts
        self.bits_x2 = bits_x2
        self.pointers = torch.tensor(
            [
                [[t.data_ptr(), s.data_ptr(), v.data_ptr()] for t, s, v in parts[k]]
                for k in sets_of
            ],
            dtype=torch.int64,
            device=device,
        )
        self.meta = torch.tensor(
            [exl3_ops.moe_v2_expert_meta(5120, intermediate, bits_x2)] * experts,
            dtype=torch.int64,
            device=device,
        )
        self.expert_map = torch.arange(experts, dtype=torch.int32, device=device)

    def run(self, op, x, ids, weights, rows_capacity=None, workspace=None):
        out = torch.full_like(x, float("nan"))
        if workspace is None:
            workspace = torch.full(
                (
                    exl3_ops.moe_workspace_size_v2(
                        rows_capacity or x.shape[0], 5120, self.intermediate,
                        self.topk, self.experts,
                    ),
                ),
                float("nan"),
                device=x.device,
            )  # fmt: skip
        op(
            x, ids, weights, self.pointers, self.meta, self.expert_map,
            self.intermediate, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
            bits_x2=self.bits_x2,
        )  # fmt: skip
        return out

    def v2(self, x, ids, weights, **kwargs):
        return self.run(exl3_ops.moe_out_v2, x, ids, weights, **kwargs)

    def fold(self, x, ids, weights, **kwargs):
        return self.run(exl3_ops.moe_out_v2_fold, x, ids, weights, **kwargs)


def _split_synthetic(config):
    """h5120._synthetic at a split shape: four weight sets over every expert,
    set UNALIGNED_SET with only 2-byte aligned trellis bases."""
    intermediate, topk, experts, bits_x2 = config
    h2560._extension_or_skip()
    exl3_ops.require_moe_v2_fold()
    device = torch.device("cuda")
    rng = np.random.default_rng(1022 + intermediate + bits_x2)
    parts, owners = [], []
    for index in range(h5120.DISTINCT):
        matrices = []
        for k, n in ((5120, intermediate), (5120, intermediate), (intermediate, 5120)):
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
            offset = 1 if index == h5120.UNALIGNED_SET else 0
            t = base[offset : offset + trellis.size].view(trellis.shape)
            t.copy_(torch.from_numpy(trellis))
            owners.append(base)
            matrices.append(
                (t, torch.from_numpy(suh).to(device), torch.from_numpy(svh).to(device))
            )
        parts.append(matrices)
    sets_of = [e % h5120.DISTINCT for e in range(experts)]
    layer = _SplitLayer(parts, sets_of, intermediate, topk, experts, bits_x2)
    layer.owners = owners
    return layer


@pytest.fixture(
    scope="module",
    params=SPLIT_CONFIGS,
    ids=lambda c: f"i{c[0]}-top{c[1]}-e{c[2]}-bits_x2={c[3]}",
)
def split_layer(request):
    return _split_synthetic(request.param)


# h5120's routings with every expert local: rows without a local route are
# rows whose ids are all padding or out of range.
SPLIT_SCENARIOS = ["uniform", "masked", "nonlocal", "hot", "single", "int64"]


def _split_case(layer, scenario, rows, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    experts, topk = layer.experts, layer.topk
    ids = torch.stack(
        [torch.randperm(experts, generator=generator)[:topk] for _ in range(rows)]
    )
    if scenario == "masked":  # padding and out of range
        ids[:, 0] = -1
        ids[:, 1] = experts
    elif scenario == "nonlocal":  # no route of any row is served
        ids[:, ::2] = -1
        ids[:, 1::2] = experts + 7
    elif scenario == "hot":  # two experts in every row: several 8-row tiles each
        ids[:, 0] = 5
        ids[:, 1] = 4 * 7 + h5120.UNALIGNED_SET
    elif scenario == "single":
        ids[:] = -1
        ids[0, 3] = h5120.UNALIGNED_SET
    weights = torch.rand(rows, topk, generator=generator) + 0.01
    weights = weights / weights.sum(-1, keepdim=True)
    ids = ids if scenario == "int64" else ids.to(torch.int32)
    x = (torch.randn(rows, 5120, generator=generator) * scale).to(torch.bfloat16)
    return x.cuda(), ids.cuda(), weights.cuda()


@needs_sm80
@pytest.mark.parametrize("scenario", SPLIT_SCENARIOS)
def test_fold_is_bitwise_v2_at_split_shapes(split_layer, scenario):
    for rows in ROWS:
        for scale in (1.0, 16.0) if scenario in ("uniform", "hot") else (1.0,):
            x, ids, w = _split_case(split_layer, scenario, rows, 7 * rows + 3, scale)
            v2 = split_layer.v2(x, ids, w)
            fold = split_layer.fold(x, ids, w)
            h2560._bitwise(fold, v2, f"{scenario} rows={rows} scale={scale}")
            assert not torch.isnan(fold).any()
            if scenario == "nonlocal":
                assert not fold.view(torch.int16).any(), "rows without a route"


@needs_sm80
def test_fold_repeats_bitwise_at_split_shapes(split_layer):
    """A workspace sized for 64 rows, reused by calls of other sizes and
    garbled between calls, gives the same bits."""
    rows = 48
    workspace = torch.full(
        (exl3_ops.moe_workspace_size_v2(64, 5120, split_layer.intermediate,
                                        split_layer.topk, split_layer.experts),),
        float("nan"),
        device="cuda",
    )  # fmt: skip
    first = None
    for count in (rows, 64, 1, rows, rows):
        seed = 99 if count == rows else count
        x, ids, w = _split_case(split_layer, "uniform", count, seed)
        out = split_layer.fold(x, ids, w, workspace=workspace)
        if count == rows:
            if first is None:
                first = out.clone()
                h2560._bitwise(out, split_layer.v2(x, ids, w), "fold vs v2")
            h2560._bitwise(out, first, "repeat")
        workspace.view(torch.int32)[::7] = -12345


@needs_sm80
def test_fold_graph_replay_is_bitwise_at_split_shapes(split_layer):
    rows = 48
    layer = split_layer
    exl3_ops.warmup_moe_v2(
        layer.pointers, layer.meta, layer.expert_map, list(range(layer.experts)), 5120,
        layer.intermediate, layer.topk, layer.bits_x2, h5120.SWIGLU_LIMIT, fold=True,
    )  # fmt: skip
    x, ids, weights = _split_case(layer, "uniform", rows, 11)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_v2(
            rows, 5120, layer.intermediate, layer.topk, layer.experts
        ),
        device=x.device,
    )

    def call():
        exl3_ops.moe_out_v2_fold(
            x, ids, weights, layer.pointers, layer.meta, layer.expert_map,
            layer.intermediate, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
            bits_x2=layer.bits_x2,
        )  # fmt: skip

    torch.cuda.synchronize()
    before = h2560._allocations()
    call()
    torch.cuda.synchronize()
    assert h2560._allocations() == before, "eager call allocated"
    h2560._bitwise(out, layer.v2(x, ids, weights), "eager fold vs v2")
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = h2560._allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert h2560._allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, eager, "replay vs eager")
    x2, ids2, weights2 = _split_case(layer, "masked", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, layer.v2(x2, ids2, weights2), "replay new inputs")


# Real layers as one rank of the split holds them: (layer, rank). Rank 0
# holds chunk (0 + layer) % 4 of the 640/640/512/512 columns: layer 10 a
# 512-column chunk at bits_x2 6, layer 20 a 640-column one at bits_x2 4.
SPLIT_REAL = [(10, 0), (20, 0)]


def _split_real_layer(layer, rank, size=4):
    from tests.kernels.quantization import exl3_real_cases as real
    from vllm_sm80.kernels.exl3.utils import block_shard

    h2560._extension_or_skip()
    exl3_ops.require_moe_v2_fold()
    checkpoint = real.Checkpoint(real.checkpoint_or_skip())
    matrices = checkpoint.manifest.matrices
    experts = 0
    while f"layers.{layer}.ffn.experts.{experts}.w1" in matrices:
        experts += 1
    if not experts:
        pytest.skip(f"the checkpoint has no routed experts in layer {layer}")
    first = checkpoint.matrix(f"layers.{layer}.ffn.experts.0.w1")
    hidden, whole = first.padded_in, first.padded_out
    columns = block_shard(whole, whole, rank, size, layer)
    width = columns.stop - columns.start
    configs = [
        c for c in SPLIT_CONFIGS
        if c[0] == width and c[2] == experts and c[3] == first.bits_x2
    ]  # fmt: skip
    if hidden != 5120 or not configs:
        pytest.skip(
            f"no open split configuration for layer {layer}: hidden {hidden}, "
            f"width {width}, {experts} experts, bits_x2 {first.bits_x2}"
        )
    device = torch.device("cuda")
    chunk = (columns.start, columns.stop)
    parts = []
    for expert in range(experts):
        prefix = f"layers.{layer}.ffn.experts.{expert}"
        parts.append(
            [
                tuple(
                    t.to(device)
                    for t in checkpoint.shard(f"{prefix}.{role}", (0, hidden), [chunk])
                )
                for role in ("w1", "w3")  # gate, up: the chunk's columns
            ]
            + [
                tuple(
                    t.to(device)
                    for t in checkpoint.shard(f"{prefix}.w2", chunk, [(0, hidden)])
                )  # down: the chunk's rows
            ]
        )
    from safetensors import safe_open

    # The layer's MoE input scale (unit weights where the layer has none).
    header = checkpoint.manifest.tensors.get(f"layers.{layer}.ffn_norm.weight")
    norm_weight = torch.ones(hidden, device=device)
    if header is not None:
        with safe_open(header.path, framework="pt") as handle:
            norm_weight = handle.get_tensor(header.name).float().to(device)
    _, topk, _, bits_x2 = configs[0]
    layer_experts = _SplitLayer(
        parts, list(range(experts)), width, topk, experts, bits_x2
    )
    return layer_experts, norm_weight


@needs_sm80
@pytest.mark.parametrize("layer,rank", SPLIT_REAL, ids=lambda v: str(v))
def test_fold_is_bitwise_v2_on_real_split_layer_weights(layer, rank):
    experts, norm_weight = _split_real_layer(layer, rank)
    for rows in ROWS:
        generator = torch.Generator().manual_seed(1000 * layer + rows)
        ids = torch.stack(
            [
                torch.randperm(experts.experts, generator=generator)[: experts.topk]
                for _ in range(rows)
            ]
        ).to(torch.int32)
        weights = torch.rand(rows, experts.topk, generator=generator) + 0.01
        weights = weights / weights.sum(-1, keepdim=True)
        h = torch.randn(rows, 5120, generator=generator, dtype=torch.float64)
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        x = (h.cuda().float() * norm_weight).to(torch.bfloat16)
        ids, weights = ids.cuda(), weights.cuda()
        h2560._bitwise(
            experts.fold(x, ids, weights),
            experts.v2(x, ids, weights),
            f"layer {layer} rank {rank} width {experts.intermediate} rows={rows}",
        )
