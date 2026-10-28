# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused DSpark context-KV prologue (``ops/sm80_draft_prologue.py``) on
CPU.

- In the Triton interpreter (a subprocess with ``TRITON_INTERPRET=1``), the
  kernel's RMSNorm sum (``torch_cuda_mean_sum``) is bitwise
  :func:`cuda_mean_sum_model`, an independent NumPy model of torch's CUDA
  reduction (``ATen/native/cuda/Reduce.cuh``) of a contiguous FP32 row of 512
  for each lane count torch picks; the lane counts give different sums on
  some rows, and so does a sequential sum.
- The lane count follows ``ReduceConfig::set_block_dimension`` for every row
  count, and no row count splits a row across warps or blocks; the header
  statements the model rests on are pinned in the installed torch.
- The kernel compiles for sm_80 at each lane count, without FTZ forms and
  with ``rsqrt.approx.f32``; the squares do not contract into the sum even
  with FMA contraction on (where a plain product would).
- The switch is off by default; ``supports`` refuses what the kernel does
  not serve; ``sm80_store_context_kv`` takes the kernel when the layer's
  switch is on and the torch path otherwise.

The libdevice parts (rsqrt, the RoPE's sin/cos/fma, the qdq's rint) do not
run in the interpreter. The GPU half (cache bytes equal to the torch path's,
the sum model against ``mean`` on CUDA, graph replay) is
``test_sm80_draft_prologue_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_draft_prologue.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

from vllm_sm80.models.deepseek_v41.ops import sm80_draft_prologue as dp

ROOT = Path(__file__).resolve().parents[4]
WIDTH = 512
LANES = (128, 64, 32)


@triton.jit
def _sum_probe(x_ptr, out_ptr, D: tl.constexpr, BW: tl.constexpr):
    t = tl.program_id(0)
    s = dp.torch_cuda_mean_sum(tl.load(x_ptr + t * D + tl.arange(0, D)), D, BW)
    tl.store(out_ptr + t + tl.arange(0, 1), s)


@triton.jit
def _guarded_squares(x_ptr, out_ptr, D: tl.constexpr, BW: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, D))
    tl.store(out_ptr + tl.arange(0, 1),
             dp.torch_cuda_mean_sum(libdevice.mul_rn(x, x), D, BW))


@triton.jit
def _plain_squares(x_ptr, out_ptr, D: tl.constexpr, BW: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, D))
    tl.store(out_ptr + tl.arange(0, 1), dp.torch_cuda_mean_sum(x * x, D, BW))


def cuda_mean_sum_model(rows: np.ndarray, lanes: int) -> np.ndarray:
    """The sum of each contiguous FP32 row of ``rows`` [R, W] as torch's CUDA
    reduction kernel orders it with ``lanes`` threads per row and inputs
    vectorized by 4: lane x loads vectors x, x + lanes, ... and adds element
    i of each into accumulator i (from 0); it combines its accumulators in
    order; then lanes x and x + offset are added for offset = lanes / 2, ...,
    1 (shared memory down to 32 lanes, then warp shuffles of decreasing
    offset)."""
    f32 = np.float32
    r, w = rows.shape
    vecs = rows.reshape(r, w // 4, 4).astype(f32)
    acc = np.zeros((r, lanes, 4), dtype=f32)
    idx = 0
    while idx < w // 4:
        acc = (acc + vecs[:, idx:idx + lanes, :]).astype(f32)
        idx += lanes
    lane = acc[:, :, 0]
    for i in range(1, 4):
        lane = (lane + acc[:, :, i]).astype(f32)
    offset = lanes // 2
    while offset:
        lane = (lane[:, :offset] + lane[:, offset:2 * offset]).astype(f32)
        offset //= 2
    return lane[:, 0]


def square_rows(count: int, seed: int) -> np.ndarray:
    """Squares of BF16 values (exact in FP32) spanning six decades."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(count, WIDTH, generator=g)
    x = x * 10.0 ** torch.randint(-3, 3, (count, WIDTH), generator=g).float()
    xf = x.to(torch.bfloat16).float().numpy()
    return (xf * xf).astype(np.float32)


def interpreted_results() -> dict:
    """Run in the Triton interpreter: the kernel's sum against the model."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    rows = square_rows(256, 0)
    out = {}
    sums = {}
    for lanes in LANES:
        got = torch.empty(rows.shape[0], dtype=torch.float32)
        _sum_probe[(rows.shape[0],)](torch.from_numpy(rows), got, WIDTH, lanes)
        want = cuda_mean_sum_model(rows, lanes)
        sums[lanes] = want
        out[f"bitwise/{lanes}"] = bool(
            np.array_equal(got.numpy().view(np.int32), want.view(np.int32)))
    seq = rows[:, 0].copy()
    for j in range(1, WIDTH):
        seq = (seq + rows[:, j]).astype(np.float32)
    out["differs"] = {
        "128-64": int((sums[128] != sums[64]).sum()),
        "64-32": int((sums[64] != sums[32]).sum()),
        "128-32": int((sums[128] != sums[32]).sum()),
        "sequential": int(min((sums[n] != seq).sum() for n in LANES)),
    }
    return out


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_draft_prologue "
            "import interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_interpreted_sum_is_the_cuda_reduction_order(interpreted):
    for lanes in LANES:
        assert interpreted[f"bitwise/{lanes}"], lanes
    # Each lane count, and a sequential sum, gives other bits on some rows.
    assert all(n > 0 for n in interpreted["differs"].values()), interpreted


def test_model_matches_the_documented_order():
    rng = np.random.default_rng(1)
    f = np.float32
    rows = (rng.standard_normal((50, 128)) ** 2).astype(f)
    got = cuda_mean_sum_model(rows, 32)
    for r, row in enumerate(rows):
        lanes = []
        for x in range(32):
            a = [f(row[4 * x + i]) for i in range(4)]
            lanes.append(f(f(f(a[0] + a[1]) + a[2]) + a[3]))
        offset = 16
        while offset:
            lanes = [f(lanes[x] + lanes[x + offset]) for x in range(offset)]
            offset //= 2
        assert got[r] == lanes[0]


def _splits_across_warps(rows: int, lanes: int) -> bool:
    """Whether setReduceConfig would also split the row across warps (and
    then possibly blocks): values per thread against its threshold."""
    height = min(1 << (rows.bit_length() - 1) if rows < 512 else 512, 512 // lanes)
    values = -(-WIDTH // lanes)
    return values >= min(height * 16, 256)


def test_lane_count_follows_reduce_config():
    want = {1: 128, 2: 128, 3: 128, 4: 128, 7: 128, 8: 64, 9: 64, 15: 64,
            16: 32, 17: 32, 40: 32, 511: 32, 512: 32, 4096: 32}
    for rows, lanes in want.items():
        assert dp.mean_block_width(rows, WIDTH) == lanes, rows
        assert not _splits_across_warps(rows, lanes), rows
    with pytest.raises(ValueError):
        dp.mean_block_width(0, WIDTH)
    with pytest.raises(ValueError):
        dp.mean_block_width(4, 384)


def test_reduction_model_is_the_installed_torch_reduction():
    """The model was read from torch's CUDA reduction headers; pin the
    statements it rests on in the installed torch. A torch upgrade that
    changes one fails here (the GPU half checks the model against ``mean``
    on CUDA)."""
    inc = Path(torch.__file__).parent / "include/ATen/native"
    src = (inc / "cuda/Reduce.cuh").read_text()
    for stmt in (
        "static constexpr int MAX_NUM_THREADS = 512;",
        "block_width = std::min(dim0_pow2, int(at::cuda::warp_size()));",
        "block_height = std::min(dim1_pow2, int(max_num_threads / block_width));",
        "block_width = std::min(dim0_pow2, int(max_num_threads / block_height));",
        "dim0 >= 128 && iter.num_reduce_dims() == 1",
        "dim0 /= input_vec_size;",
        "config.input_mult[0] = config.split_input(block_width);",
        "std::min<int>(block_height * 16, max_values_per_thread);",
        "while (idx * input_vec_size + input_vec_size - 1 < end) {",
        "value_list[i] = ops.reduce(value_list[i], values_vec.val[i], "
        "shift + idx * input_vec_size + i);",
        "value_list[0] = ops.combine(value_list[0], value_list[i]);",
        "for (int offset = dim_x/2; offset >= C10_WARP_SIZE; offset >>= 1) {",
        "for (int offset = dim_x >> 1; offset > 0; offset >>= 1) {",
    ):
        assert stmt in src, stmt
    assert re.search(r"template <typename scalar_t, typename ops_t, typename index_t, "
                     r"typename out_scalar_t=scalar_t, int vt0=4, int input_vec_size=vt0>",
                     src)
    ops = (inc / "SharedReduceOps.h").read_text()
    mean = ops[ops.index("struct MeanOps {"):]
    mean = mean[:mean.index("};")]
    assert "return a + b;" in mean and "return a * factor;" in mean


def _compile(kernel, sig, consts, **options):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    return triton.compile(ASTSource(kernel, sig, constexprs=consts),
                          target=GPUTarget("cuda", 80, 32),
                          options={"num_warps": 4, **options})


@pytest.mark.parametrize("lanes", LANES)
def test_kernel_compiles_for_sm80(lanes):
    kern = dp._draft_prologue_kernel
    types = {"kv_ptr": "*bf16", "norm_ptr": "*bf16", "positions_ptr": "*i64",
             "freqs_ptr": "*fp32", "cache_ptr": "*bf16", "slots_ptr": "*i64",
             "eps": "fp32"}
    consts = dict(stride_kt=1024, stride_kd=1, stride_n=1, stride_pos=1,
                  stride_cb=64 * 512, stride_ce=512, stride_cd=1, stride_slot=1,
                  D=512, E=64, RD=64, BW=lanes)
    sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
    c = _compile(kern, sig, consts, enable_fp_fusion=False, enable_reflect_ftz=False)
    assert c.asm["cubin"]
    ptx = c.asm["ptx"]
    assert "rsqrt.approx.f32" in ptx
    assert "mul.rn.f32" in ptx and "div.rn.f32" in ptx
    assert not re.findall(r"[a-z0-9.]+\.ftz\.f32", ptx)


def test_squares_do_not_contract_into_the_sum():
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "D": "constexpr", "BW": "constexpr"}
    contracted = []
    for lanes in LANES:
        consts = {"D": WIDTH, "BW": lanes}
        assert "fma" not in _compile(_guarded_squares, sig, consts).asm["ptx"]
        if "fma.rn.f32" in _compile(_plain_squares, sig, consts).asm["ptx"]:
            contracted.append(lanes)
    # The control: unguarded squares fuse into the sum, so the check above can
    # fail. Whether they fuse at a width depends on the layout Triton picks:
    # Triton 3.8 moves the 128-lane squares through shared memory first.
    assert contracted


def test_switch_default_and_supports(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED", "1")
    assert envs.VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED is True
    assert dp.supports(512, 64, torch.bfloat16, (8, 0)) is None
    assert dp.supports(512, 64, torch.bfloat16, (8, 6)) is None
    assert "capability" in dp.supports(512, 64, torch.bfloat16, (9, 0))
    assert "capability" in dp.supports(512, 64, torch.bfloat16, None)
    assert "head dim" in dp.supports(256, 64, torch.bfloat16, (8, 0))
    assert "head dim" in dp.supports(512, 128, torch.bfloat16, (8, 0))
    assert "BF16" in dp.supports(512, 64, torch.float8_e4m3fn, (8, 0))


def _context_layer(fused: bool, seed: int = 0):
    from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    g = torch.Generator().manual_seed(seed)
    q_lora, rows = 64, 6
    qr_kv = (torch.randn(rows, q_lora + WIDTH, generator=g) * 3).to(torch.bfloat16)
    layer = object.__new__(sm80.DeepseekV41Sm80Attention)
    nn.Module.__init__(layer)
    weight = (torch.rand(WIDTH, generator=g) + 0.5).to(torch.bfloat16)
    attrs = dict(
        _fused_wqa_wkv_gemm=lambda x: qr_kv,
        q_lora_rank=q_lora,
        eps=1e-6,
        kv_norm=SimpleNamespace(weight=weight),
        q_norm=SimpleNamespace(weight=None),
        attn_sink=None,
        wq_b=None,
        compressor=None,
        indexer=None,
        _sm80_freqs=core.rope_freqs(64, 65536, 10000.0, 16.0, 32, 1),
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(3, 4, WIDTH, dtype=torch.bfloat16)),
        _sm80_draft_prologue=fused,
    )
    for k, v in attrs.items():
        object.__setattr__(layer, k, v)
    return layer, qr_kv


def test_store_context_kv_takes_the_kernel_when_switched_on(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    calls = []
    monkeypatch.setattr(dp, "store_context_kv", lambda *a: calls.append(a))
    positions = torch.arange(6) + 100
    slots = torch.tensor([5, -1, 6, 7, 1, -1])

    layer, qr_kv = _context_layer(fused=False)
    layer.sm80_store_context_kv(None, positions, slots)
    assert not calls
    want = torch.zeros(3, 4, WIDTH, dtype=torch.bfloat16)
    kv = core.rms_norm(qr_kv[:, 64:], layer.kv_norm.weight, 1e-6)
    core.store_rows(want, slots, core.window_kv(kv, positions, layer._sm80_freqs))
    cache = layer.swa_cache_layer.kv_cache
    assert torch.equal(cache.view(torch.int16), want.view(torch.int16))
    assert cache.abs().sum() > 0

    layer, qr_kv = _context_layer(fused=True)
    layer.sm80_store_context_kv(None, positions, slots)
    assert len(calls) == 1
    kv, weight, eps, pos, freqs, cache, slot_mapping = calls[0]
    assert kv.data_ptr() == qr_kv[:, 64:].data_ptr() and kv.shape == (6, WIDTH)
    assert kv.stride() == qr_kv.stride()
    assert weight is layer.kv_norm.weight and eps == 1e-6
    assert pos is positions and slot_mapping is slots
    assert freqs is layer._sm80_freqs and cache is layer.swa_cache_layer.kv_cache
    assert cache.abs().sum() == 0

    layer.sm80_store_context_kv(None, positions, None)
    assert len(calls) == 1


def test_dev_ab_switch_sets_the_drafter_layers(monkeypatch):
    from vllm_sm80 import dev_ab

    env = "VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED"
    monkeypatch.setenv(env, "0")
    eligible = SimpleNamespace(_sm80_draft_prologue_eligible=True, _sm80_draft_prologue=False)
    closed = SimpleNamespace(_sm80_draft_prologue_eligible=False, _sm80_draft_prologue=False)
    target = SimpleNamespace(_sm80_draft_prologue_eligible=True, _sm80_draft_prologue=False)
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [target]),
        speculator=SimpleNamespace(model=SimpleNamespace(modules=lambda: [eligible, closed])))
    switch = dev_ab.SWITCHES["dsv41_draft_prologue_fused"]
    assert switch.env == env and switch.graph_bound and switch.output_identical
    assert switch.unavailable(runner) is None
    assert not switch.current(runner)
    switch.apply(runner, True)
    assert eligible._sm80_draft_prologue and switch.current(runner)
    assert not closed._sm80_draft_prologue and not target._sm80_draft_prologue
    assert os.environ[env] == "1"
    switch.apply(runner, False)
    assert not eligible._sm80_draft_prologue and os.environ[env] == "0"
    assert "no eligible" in switch.unavailable(SimpleNamespace(model=None))
    closed_only = SimpleNamespace(speculator=SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [closed])))
    assert "no eligible" in switch.unavailable(closed_only)
