# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-bit e4m3 K dequant in the DSA indexer decode logits (sm_80).

``VLLM_GLM5_INDEXER_DECODE_RAW_K`` (default 1) makes the Triton paged logits
kernel place the e4m3 K bits straight into bf16 (value * 2^-120) instead of
going through fp32, with 2^60 folded into Q and into the head weights, as the
prefill kernel already does. The logits are bitwise the fp32-dequant path's
when the MMA keeps bf16 subnormal inputs (14 of the 256 codes land there),
which the PTX ISA leaves unspecified, hence the sm_80 gate and the device
check here.

CPU tests (``CUDA_VISIBLE_DEVICES=""``): the flag (default on, 0 = kill
switch), the gate, the banners through the real logger, the bit placement
over all 256 codes (exact value * 2^-120, NaN codes -> +-480 * 2^-120) and
the range argument (every product and partial sum stays in the fp32 normal
range), and the sm_80 SASS of both variants. GPU tests (skip without sm_80):
the tensor cores on all 256 raw codes against exact values (subnormals not
flushed), on vs off bitwise on production shapes including caches holding
every code, and CUDA-graph replay.
"""

import logging
import math

import pytest
import torch

from tests.kernels import ampere_sass
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops import triton_mqa_logits as tml
from vllm.v1.attention.ops.triton_e4m3 import e4m3_bits_to_bf16_raw

FLAG = "VLLM_GLM5_INDEXER_DECODE_RAW_K"
H, D, BS = 32, 128, 64
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")
needs_cuobjdump = pytest.mark.skipif(not ampere_sass.available(),
                                     reason="needs Triton's cuobjdump")


# --------------------------------------------------------------------- CPU
def test_flag_declared_default_on(monkeypatch):
    from vllm import envs

    monkeypatch.delenv(FLAG, raising=False)
    assert FLAG in envs.environment_variables
    assert envs.environment_variables[FLAG]() is True
    monkeypatch.setenv(FLAG, "0")
    assert envs.environment_variables[FLAG]() is False
    assert f"    {FLAG}: bool = True\n" in open(envs.__file__).read()


def _banners(fn):
    import vllm.logger as vlog
    from vllm.logger import init_logger

    seen = []

    class Rec(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    logger = init_logger(tml.__name__)
    h = Rec(level=logging.DEBUG)
    old = logger.level
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    vlog._print_info_once.cache_clear()
    try:
        out = fn()
    finally:
        logger.removeHandler(h)
        logger.setLevel(old)
    return out, seen


@pytest.mark.parametrize("cap,flag,want,msg", [
    ((8, 0), None, True, "raw-bit e4m3 K dequant active"),
    ((8, 0), "1", True, "raw-bit e4m3 K dequant active"),
    ((8, 0), "0", False, None),
    ((9, 0), None, False, "not sm_80"),
    ((8, 6), "1", False, "not sm_80"),
])
def test_gate_and_banners(monkeypatch, cap, flag, want, msg):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: cap)
    tml._sm80.cache_clear()
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    try:
        got, seen = _banners(lambda: tml._decode_raw_k(torch.device("cuda", 0)))
    finally:
        tml._sm80.cache_clear()
    assert got is want
    if msg is None:
        assert seen == []
    else:
        assert any(msg in m for m in seen), seen


def _raw_bits_emulated(code: int) -> int:
    # e4m3_bits_to_bf16_raw per byte: magnitude bits << 4, sign << 8
    return ((code << 4) & 0x07F0) | ((code << 8) & 0x8000)


def _e4m3_values():
    codes = torch.arange(256, dtype=torch.int32)
    v = codes.to(torch.uint8).view(torch.float8_e4m3fn).double()
    nan480 = torch.where(codes < 128, 480.0, -480.0).double()
    return codes, torch.where(torch.isnan(v), nan480, v)


def test_bit_placement_exact_for_all_256_codes():
    codes, val = _e4m3_values()
    raw = torch.tensor([_raw_bits_emulated(int(c)) for c in codes], dtype=torch.int32)
    rawf = raw.to(torch.int16).view(torch.bfloat16).double()
    assert torch.equal(rawf, val * 2.0 ** -120)
    # the fp32 path of the default kernel gives the same values unscaled
    sub = [int(c) for c in codes if (c & 0x78) == 0 and (c & 7)]
    assert sub == [1, 2, 3, 4, 5, 6, 7, 129, 130, 131, 132, 133, 134, 135]
    bf16_min_normal = 2.0 ** -126
    for c in sub:
        assert 0 < abs(float(rawf[c])) < bf16_min_normal


def test_scaled_products_stay_in_the_fp32_normal_range():
    """q * 2^60 (bf16, exact) times raw k: every nonzero product is a multiple
    of 2^-78 and at most 480^2 * 2^-60, so any partial sum over D = 128 is 0
    or in [2^-78, 2^-35]: far inside the fp32 normal range, so scaling by
    2^-60 commutes with every rounding of the accumulation."""
    _, val = _e4m3_values()
    nz = val[val != 0].abs()
    qmin, qmax = float(nz.min()) * 2.0 ** 60, float(nz.max()) * 2.0 ** 60
    kmin, kmax = float(nz.min()) * 2.0 ** -120, float(nz.max()) * 2.0 ** -120
    assert math.log2(qmin * kmin) == -78
    assert math.log2(D * qmax * kmax) < -34
    assert qmax < 2.0 ** 128 and qmin >= 2.0 ** -126  # Q stays normal in bf16
    # every e4m3 value is a multiple of 2^-9
    assert torch.equal(torch.round(val * 512), val * 512)


def _args():
    return dict(q_ptr="*u8", kv_u8_ptr="*u8", kv_f32_ptr="*fp32", w_ptr="*fp32",
                ctx_ptr="*i32", bt_ptr="*i32", out_ptr="*fp32",
                max_model_len=32768, tiles_per_prog=4, stride_qb=H * D,
                stride_qt=H * D, stride_qh=D, stride_wm=H, stride_ctx_b=1,
                stride_ctx_t=1, stride_bt=512, stride_om=32768, NEXT_N=(1,),
                NEXT_N_P2=(1,), H=(H,), D=(D,), BLOCK_SIZE=(BS,),
                PAGE_BYTES=(BS * (D + 4),))


@needs_cuobjdump
def test_sass_raw_variant(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    base = ampere_sass.compile_sm80(tml._paged_mqa_logits_kernel,
                                    dict(_args(), RAW_K=(False,)), 2, 2)
    raw = ampere_sass.compile_sm80(tml._paged_mqa_logits_kernel,
                                   dict(_args(), RAW_K=(True,)), 2, 2)
    b_ops, r_ops = base[2], raw[2]
    assert ampere_sass.count(r_ops, "HMMA") == ampere_sass.count(b_ops, "HMMA") > 0
    assert "HMMA.16816.F32.BF16" in raw[0]
    assert ampere_sass.count(r_ops, "F2FP") < ampere_sass.count(b_ops, "F2FP")
    assert sum(r_ops.values()) < sum(b_ops.values())
    for ops in (b_ops, r_ops):
        assert ampere_sass.count(ops, "LDL") == ampere_sass.count(ops, "STL") == 0


def test_raw_k_default_matches_base_signature():
    """With RAW_K at its default the kernel is the original one."""
    import inspect

    src = inspect.getsource(tml._paged_mqa_logits_kernel.fn)
    assert "RAW_K: tl.constexpr = False" in src


# --------------------------------------------------------------------- GPU
@triton.jit
def _probe_dot(q_ptr, k_ptr, o_ptr, DD: tl.constexpr, NQ: tl.constexpr,
               NK: tl.constexpr):
    d = tl.arange(0, DD)
    q = tl.load(q_ptr + tl.arange(0, NQ)[:, None] * DD + d[None, :])
    k = e4m3_bits_to_bf16_raw(
        tl.load(k_ptr + tl.arange(0, NK)[:, None] * DD + d[None, :]))
    tl.store(o_ptr + tl.arange(0, NQ)[:, None] * NK + tl.arange(0, NK)[None, :],
             tl.dot(q, tl.trans(k)))


@needs_sm80
def test_tensor_cores_keep_bf16_subnormal_inputs():
    """All 256 raw-placed codes through tl.dot against exact values."""
    DD, NK = 16, 256
    k = torch.zeros(NK, DD, dtype=torch.uint8)
    for j in range(NK):
        k[j, j % DD] = j
    q = torch.eye(DD) * 2.0 ** 60
    out = torch.empty(DD, NK, device="cuda")
    _probe_dot[(1,)](q.to(torch.bfloat16).cuda(), k.cuda(), out, DD=DD, NQ=DD, NK=NK,
                num_warps=1)
    torch.cuda.synchronize()
    _, val = _e4m3_values()
    got = torch.tensor([float(out[j % DD, j]) for j in range(NK)], dtype=torch.float64)
    bad = [j for j in range(NK) if got[j] != val[j] * 2.0 ** -60]
    assert bad == [], bad


def _case(B, ctx, every_code, seed):
    g = torch.Generator().manual_seed(seed)
    nb = -(-ctx // BS)
    if every_code:
        kb = torch.randint(0, 256, (B * nb, BS * D), generator=g, dtype=torch.uint8)
    else:
        kb = (torch.randn(B * nb, BS * D, generator=g) * 0.5).to(
            torch.float8_e4m3fn).view(torch.uint8)
    sc = (torch.rand(B * nb, BS, generator=g) + 0.5).view(torch.uint8)
    kv = torch.cat([kb, sc], dim=1).view(B * nb, BS, D + 4).contiguous()
    q = (torch.randn(B, 1, H, D, generator=g) * 0.5).to(torch.float8_e4m3fn)
    w = torch.randn(B, H, generator=g).float()
    ctxl = torch.randint(1, ctx + 1, (B, 1), generator=g, dtype=torch.int32)
    ctxl[0, 0] = ctx
    bt = torch.arange(B * nb, dtype=torch.int32).view(B, nb)
    return [t.cuda() for t in (q, kv, w, ctxl, bt)]


def _valid(y, ctxl):
    """Entries the kernel defines: row b, columns [0, ctx_b) (columns past
    the visited tiles are never written, as with DeepGEMM's clean_logits off)."""
    cols = torch.arange(y.shape[1], device=y.device)
    return cols[None, :] < ctxl.view(-1, 1)


def _logits(monkeypatch, on, args, ml):
    monkeypatch.setenv(FLAG, "1" if on else "0")
    return tml.fp8_paged_mqa_logits_triton(*args, ml)


@needs_sm80
@pytest.mark.parametrize("B,ctx,every", [(1, 4096, False), (4, 8192, True),
                                         (8, 32768, False), (32, 4096, True)])
def test_on_equals_off_bitwise(monkeypatch, B, ctx, every):
    args = _case(B, ctx, every, seed=B + ctx)
    a = _logits(monkeypatch, False, args, ctx)
    b = _logits(monkeypatch, True, args, ctx)
    torch.cuda.synchronize()
    m = _valid(a, args[3])
    diff = int(((a.view(torch.int32) != b.view(torch.int32)) & m).sum())
    assert diff == 0, (diff, int(m.sum()))
    assert torch.isfinite(a[m]).all()


@needs_sm80
def test_graph_replay_equals_eager(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    q, kv, w, ctxl, bt = _case(8, 8192, True, seed=3)
    ref = tml.fp8_paged_mqa_logits_triton(q, kv, w, ctxl, bt, 8192)
    torch.cuda.synchronize()
    out = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out["y"] = tml.fp8_paged_mqa_logits_triton(q, kv, w, ctxl, bt, 8192)
    out["y"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    m = _valid(ref, ctxl)
    assert torch.equal(out["y"][m].view(torch.int32), ref[m].view(torch.int32))
