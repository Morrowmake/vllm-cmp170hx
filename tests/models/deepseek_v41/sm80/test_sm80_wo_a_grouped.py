# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grouped MXFP8 wo_a GEMM of DeepSeek-V4.1 on sm_80
(``ops/sm80_wo_a_grouped.py``, VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM), on CPU.

- The inline-PTX block that turns E4M3 code bytes and E8M0 scale bytes into
  scaled BF16 weights, run by an emulator of its own instructions over every
  finite code in every lane: each weight is the FP64 value of the code times
  2**(e-127) (0 with the code's sign for e = 0, Marlin's convention), and
  every multiply in it is exact.
- The kernel compiled for sm_80 at every row tile: BF16 MMA with FP32
  accumulation and none computed twice, every weight of a decode-sized call
  decoded once per K step (twice in the prefill tiles, whose warps are
  stacked along the rows), one rounding to BF16 at the store, no flush to
  zero, no spills.
- The host side: one launch over every group with the input's row and group
  strides (a row-padded or strided attention output is read in place), the
  output the only allocation, the row tiles and grid.
- The method: the loaded codes and scales kept as loaded (no copy, no Marlin
  repack, no per-group layers) and the kernel applied; with the dev A/B
  switch the per-group Marlin layers too, and the arm chosen per call; the
  refusals; the switch default and the dev A/B switch.

The GPU half (the kernel against Marlin and FP64) is
``test_sm80_wo_a_grouped_gpu.py``.
"""

import os
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from tests.models.deepseek_v41.sm80.hybrid_linear_cases import decode_fp8
from vllm_sm80.models.deepseek_v41.ops import sm80_wo_a_grouped as wg
from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

BF16 = torch.bfloat16
# E4M3FN codes other than the two NaNs.
FINITE = np.array([c for c in range(256) if c & 0x7F != 0x7F], dtype=np.uint32)


# -- the decode block, emulated ------------------------------------------------


def _asm():
    """(opcode, operands) of the decode block's instructions, in order."""
    body = wg._SCALED_E4M3_TO_BF16_ASM.value.strip()
    assert body[0] == "{" and body[-1] == "}"
    for line in body[1:-1].split(";"):
        line = line.strip()
        if line and not line.startswith(".reg"):
            op, operands = line.split(None, 1)
            yield op, [a.strip() for a in operands.split(",")]


def _prmt(a, b, sel):
    """PTX prmt.b32, default mode: byte i of d is byte (sel_i & 7) of
    {b, a}, or its sign replicated when sel_i & 8."""
    pool = (b.astype(np.uint64) << np.uint64(32)) | a.astype(np.uint64)
    out = np.zeros_like(a, dtype=np.uint64)
    ff, zero, sign = np.uint64(0xFF), np.uint64(0), np.uint64(0x80)
    for i in range(4):
        s = (sel >> np.uint64(4 * i)) & np.uint64(0xF)
        byte = (pool >> (np.uint64(8) * (s & np.uint64(7)))) & ff
        replicated = np.where((byte & sign) != 0, ff, zero)
        byte = np.where((s & np.uint64(8)) != 0, replicated, byte)
        out |= byte.astype(np.uint64) << np.uint64(8 * i)
    return out


def _bf16_values(halves):
    return (halves.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


def _bf16_bits_exact(values):
    """BF16 bits of FP64 values that BF16 holds exactly (asserted)."""
    f32 = values.astype(np.float32)
    bits = f32.view(np.uint32)
    exact = (f32.astype(np.float64) == values) & ((bits & 0xFFFF) == 0)
    assert exact.all(), "an inexact BF16 result"
    return (bits >> 16).astype(np.uint64)


def _fma_bf16x2(a, b, c):
    """PTX fma.rn.bf16x2 of operations that are exact (asserted): the FP64
    a * b + c of each half, which then needs no rounding."""
    out = np.zeros_like(a, dtype=np.uint64)
    for shift in (0, 16):
        s = np.uint64(shift)
        x, y, z = (_bf16_values((v >> s) & np.uint64(0xFFFF)) for v in (a, b, c))
        with np.errstate(over="raise", under="ignore"):
            out |= _bf16_bits_exact(x * y + z) << s
    return out


def run_decode_block(codes: np.ndarray, scales: np.ndarray):
    """The decode block on 32-bit registers of four code bytes and four scale
    bytes: the two output registers (BF16 pairs)."""
    regs = {"$2": codes.astype(np.uint64), "$3": scales.astype(np.uint64)}

    def value(operand):
        if operand in regs:
            return regs[operand]
        return np.full(codes.shape, int(operand, 0), dtype=np.uint64)

    mask = np.uint64(0xFFFFFFFF)
    for op, (dst, *srcs) in _asm():
        v = [value(s) for s in srcs]
        if op == "mov.b32":
            r = v[0]
        elif op == "and.b32":
            r = v[0] & v[1]
        elif op == "or.b32":
            r = v[0] | v[1]
        elif op == "shl.b32":
            r = (v[0] << v[1]) & mask
        elif op == "shr.b32":
            r = v[0] >> v[1]
        elif op == "prmt.b32":
            r = _prmt(v[0], v[1], v[2])
        elif op == "fma.rn.bf16x2":
            r = _fma_bf16x2(*v)
        else:
            raise AssertionError(f"no emulation of {op}")
        regs[dst] = r & mask
    return regs["$0"], regs["$1"]


def _lanes(out0, out1):
    """The four BF16 results of a block: lanes 0-1 in $0, 2-3 in $1."""
    m = np.uint64(0xFFFF)
    return [out0 & m, out0 >> np.uint64(16), out1 & m, out1 >> np.uint64(16)]


@pytest.mark.parametrize("lane", range(4))
def test_decode_block_scales_every_code_exactly(lane):
    rng = np.random.default_rng(lane)
    # Every finite code in this lane against scales over the whole range
    # where the scaled weights are BF16 numbers (2**-133 to below 2**128:
    # e = 3..246), both ends included; random finite codes in the others.
    exps = np.concatenate([[3, 4, 100, 127, 245, 246], rng.integers(3, 247, 58)])
    code = np.repeat(FINITE, exps.size)
    exp = np.tile(exps, FINITE.size).astype(np.uint32)
    codes = [rng.choice(FINITE, code.size) for _ in range(4)]
    scale = [rng.integers(3, 247, code.size).astype(np.uint32) for _ in range(4)]
    codes[lane], scale[lane] = code, exp
    word = sum(codes[i] << (8 * i) for i in range(4))
    sword = sum(scale[i] << (8 * i) for i in range(4))
    got = _lanes(*run_decode_block(word, sword))
    for i in range(4):
        w = torch.from_numpy(codes[i].astype(np.uint8)).view(torch.float8_e4m3fn)
        s = torch.from_numpy(scale[i].astype(np.uint8))
        want = decode_fp8(w.view(-1, 1).expand(-1, 32), s.view(-1, 1))[:, 0].numpy()
        assert np.array_equal(got[i], _bf16_bits_exact(want)), i
    # Signed zeros: code 0x80 is -0 and stays -0.
    assert (got[lane][code == 0x80] == 0x8000).all()
    assert (got[lane][code == 0x00] == 0).all()


def test_decode_block_zero_scale_is_marlins_zero():
    """E8M0 0 (2**-127) decodes as BF16 0, as Marlin's dequant_fp8_scales
    does; the product keeps the code's sign."""
    word = np.array([c | (c << 8) | (c << 16) | (c << 24) for c in FINITE], np.uint32)
    got = _lanes(*run_decode_block(word, np.zeros_like(word)))
    sign = np.where(FINITE & 0x80, 0x8000, 0).astype(np.uint64)
    for lane in got:
        assert np.array_equal(lane, sign)


# -- the kernel compiled for sm_80 ---------------------------------------------


def _ptxas():
    import triton

    return os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")


# A known Triton 3.8 regression: the second layout pass rematerialises the
# weight decode into one dot's operand layout at MT=3 and MT=4.
_DECODED_TWICE = pytest.mark.xfail(
    strict=True,
    reason="release-blocker:dsv41-flash: Triton 3.8 decodes the weights twice "
    "per K step at 33-64 rows (layout rematerialisation); blocks the next "
    "dsv41-flash release",
)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            c,
            id=f"rows{c[0]}",
            marks=_DECODED_TWICE if c[0] in (48, 64) else (),
        )
        for c in wg.CONFIGS
    ],
)
def test_kernel_compiles_for_sm80(tmp_path, config):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    _, bm, mt, bn, bk, warps, stages = config
    kern = wg._wo_a_grouped_kernel
    types = {"x_ptr": "*bf16", "w_ptr": "*u8", "s_ptr": "*u8", "o_ptr": "*bf16",
             "M": "i32", "stride_xm": "i32", "stride_xg": "i32"}
    sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
    consts = dict(G=2, R=1024, D=4096, BM=bm, MT=mt, BN=bn, BK=bk)
    c = triton.compile(ASTSource(kern, sig, constexprs=consts),
                       target=GPUTarget("cuda", 80, 32),
                       options={"num_warps": warps, "num_stages": stages})
    ptx = c.asm["ptx"]
    mma = re.findall(r"mma\.sync\.aligned\.([a-z0-9.]+)", ptx)
    assert set(mma) == {"m16n8k16.row.col.f32.bf16.bf16.f32"}
    # One K step of the row block in m16n8k16 MMAs over the warps, none twice.
    assert len(mma) * warps == (bm * mt // 16) * (bn // 8) * (bk // 16)
    # The weights of the step, 4 per decode block per thread: decoded once
    # for 16-row tiles (warps side by side along the columns), twice for the
    # prefill tiles.
    decodes = ptx.count("mov.b32 bias, 0x7B807B80;") * 4 * 32 * warps
    assert decodes == bk * bn * (1 if bm == 16 else 2)
    assert "fma.rn.bf16x2" in ptx and "fma.rn.bf16 " not in ptx
    # One rounding, at the store; no flush to zero, no atomics.
    assert re.search(r"cvt\.rn\.bf16(x2)?\.f32", ptx)
    assert "ftz" not in ptx and "atom." not in ptx
    path = tmp_path / f"wo_a_{bm}_{mt}.ptx"
    path.write_text(ptx)
    info = subprocess.run([_ptxas(), "-arch=sm_80", "-v", str(path), "-o",
                           str(path.with_suffix(".cubin"))],
                          capture_output=True, text=True, check=True).stderr
    assert "0 bytes spill stores, 0 bytes spill loads" in info, info


# -- the host side -------------------------------------------------------------


def test_row_tiles_grid_and_supported_shapes():
    # (BM, MT, BN, BK): decode-sized calls are one block of 16-row tiles.
    assert [wg.config_for(r)[:4] for r in (1, 16, 17, 32, 33, 48, 49, 64)] == [
        (16, 1, 32, 256), (16, 1, 32, 256), (16, 2, 32, 256), (16, 2, 32, 256),
        (16, 3, 32, 128), (16, 3, 32, 128), (16, 4, 32, 128), (16, 4, 32, 128),
    ]  # fmt: skip
    assert wg.config_for(65)[:4] == wg.config_for(256)[:4] == (64, 1, 64, 64)
    assert wg.config_for(257)[:4] == wg.config_for(8193)[:4] == (128, 1, 128, 32)
    # TP4: 2 groups of 1024 outputs. Decode calls: 64 programs, one row tile.
    assert {wg.grid_for(r, 2048) for r in range(1, 65)} == {(64,)}
    assert wg.grid_for(65, 2048) == (2 * 32,)
    assert wg.grid_for(2048, 2048) == (16 * 16,)
    assert wg.grid_for(8193, 2048) == (65 * 16,)
    assert wg.supports(2, 1024, 4096) is None and wg.supports(4, 1024, 4096) is None
    assert "outputs" in wg.supports(2, 1000, 4096)
    assert "inputs" in wg.supports(2, 1024, 4000)


class _Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(func._schema.name.split("::")[-1])
        return func(*args, **(kwargs or {}))


class _Recorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize("rows", [1, 300])
@pytest.mark.parametrize("pad", [0, 64])
def test_one_launch_reads_the_attention_output_in_place(monkeypatch, rows, pad):
    groups, rank_out, k = 2, 1024, 4096
    recorder = _Recorder()
    monkeypatch.setattr(wg, "_wo_a_grouped_kernel", recorder)
    # The attention output [T, groups * k] (a row slice of a wider buffer
    # when padded) seen as [T, groups, k].
    buf = torch.zeros(rows, groups * k + pad, dtype=BF16)
    x = buf[:, : groups * k].unflatten(1, (groups, k))
    weight = torch.zeros(groups * rank_out, k, dtype=torch.uint8)
    scale = torch.zeros(groups * rank_out, k // 32, dtype=torch.uint8)
    with _Ops() as ops:
        out = wg.wo_a_grouped(x, weight, scale)
    assert ops.names == ["empty"]  # the output only: no copy, no concat
    assert out.shape == (rows, groups * rank_out) and out.dtype == BF16
    assert out.is_contiguous()
    [(grid, args, kwargs)] = recorder.launches
    assert grid == wg.grid_for(rows, groups * rank_out)
    assert args[0] is x and args[1] is weight and args[2] is scale and args[3] is out
    assert args[4:] == (rows, x.stride(0), k)
    if rows > 1:
        # (torch reports a size-1 dimension's stride as if contiguous)
        assert x.stride(0) == groups * k + pad
    bm, mt, bn, bk, warps, stages = wg.config_for(rows)
    assert kwargs == dict(G=groups, R=rank_out, D=k, BM=bm, MT=mt, BN=bn, BK=bk,
                          num_warps=warps, num_stages=stages)


def test_no_rows_no_launch_and_unit_stride_needed(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(wg, "_wo_a_grouped_kernel", recorder)
    w = torch.zeros(256, 512, dtype=torch.uint8)
    s = torch.zeros(256, 16, dtype=torch.uint8)
    out = wg.wo_a_grouped(torch.zeros(0, 2, 512, dtype=BF16), w, s)
    assert out.shape == (0, 256) and not recorder.launches
    with pytest.raises(ValueError, match="unit-stride"):
        wg.wo_a_grouped(torch.zeros(3, 512, 2, dtype=BF16).transpose(1, 2), w, s)


# -- the method -----------------------------------------------------------------


def _dequant(codes, scales):
    """FP64 weights of loaded codes and per-row E8M0 scales (independent
    definitions)."""
    return decode_fp8(codes.view(torch.float8_e4m3fn), scales)


def _mxfp8_layer(monkeypatch, groups, rank_out, k, seed=2):
    """A built and loaded MXFP8 wo_a (official 32 x 32 block scales), as the
    hybrid checkpoint's, with Marlin's repack and GEMM replaced by recorders
    (CPU). Returns (layer, processed shapes, official codes, per-row scales)."""
    from tests.models.deepseek_v41.sm80.construct import set_groups
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    set_groups(monkeypatch, 1, 0)
    monkeypatch.setattr(MarlinMxfp8LinearKernel, "is_supported",
                        classmethod(lambda cls, c=None: (True, None)))
    processed = []
    monkeypatch.setattr(MarlinMxfp8LinearKernel, "process_weights_after_loading",
                        lambda self, layer: processed.append(tuple(layer.weight.shape)))
    monkeypatch.setattr(
        MarlinMxfp8LinearKernel, "apply_weights",
        lambda self, layer, x, bias=None: (
            x.double() @ _dequant(layer.weight.view(torch.uint8), layer.weight_scale).t()
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.rand(groups * rank_out, k, generator=generator) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    blocks = torch.randint(118, 130, (groups * rank_out // 32, k // 32),
                           generator=generator).to(torch.uint8)
    layer = nn.Module()
    layer.register_parameter("bias", None)
    method = ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
        CkptCtx(scale_block_size=(32, 32)),
    )
    method.input_dtype = method.out_dtype = BF16

    def load(param, loaded, *args):
        param.data.copy_(loaded)

    method.create_weights(layer, k, [groups * rank_out], k, groups * rank_out, BF16,
                          weight_loader=load)
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale.weight_loader(layer.weight_scale, blocks.view(torch.float8_e8m0fnu))
    layer.quant_method = method
    return layer, processed, weight.view(torch.uint8), blocks.repeat_interleave(32, 0)


def _official(x, codes, scales, groups):
    w = _dequant(codes, scales)
    return torch.einsum("tgd,grd->tgr", x.double(), w.view(groups, -1, w.shape[1])).flatten(1)


def _kernel_stand_in(monkeypatch):
    """wo_a_grouped replaced by its FP64 definition; records its inputs."""
    calls = []

    def fp64(x, weight, scale):
        calls.append((x, weight, scale))
        return _official(x, weight, scale, x.shape[1])

    monkeypatch.setattr(wg, "wo_a_grouped", fp64)
    return calls


def test_method_keeps_the_loaded_codes_and_runs_the_kernel(monkeypatch):
    groups, rank_out, k = 2, 128, 256
    layer, processed, codes, scales = _mxfp8_layer(monkeypatch, groups, rank_out, k)
    storage = (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    wg.WoAGroupedGemmMethod.adopt(layer, groups)
    method = layer.quant_method
    assert type(method) is wg.WoAGroupedGemmMethod and method.groups == groups
    assert method.gemm and not method.keep_groups
    method.process_weights_after_loading(layer)
    # No Marlin repack, no per-group layers, the parameters gone; the codes
    # and scales are the loaded tensors themselves (no copy).
    assert processed == [] and not hasattr(layer, "grouped")
    assert not hasattr(layer, "weight") and not hasattr(layer, "weight_scale")
    assert (layer.wo_a_codes.data_ptr(), layer.wo_a_scales.data_ptr()) == storage
    assert layer.wo_a_codes.dtype == layer.wo_a_scales.dtype == torch.uint8
    assert torch.equal(layer.wo_a_codes, codes) and torch.equal(layer.wo_a_scales, scales)
    assert dict(layer.named_buffers()).keys() == {"wo_a_codes", "wo_a_scales"}
    assert not list(layer.parameters())

    calls = _kernel_stand_in(monkeypatch)
    x = torch.randn(5, groups * k).to(BF16).view(5, groups, k)
    got = method.apply_grouped(layer, x)
    [(seen, w, s)] = calls
    assert seen is x and w is layer.wo_a_codes and s is layer.wo_a_scales
    torch.testing.assert_close(got, _official(x, codes, scales, groups), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Grouped input"):
        method.apply_grouped(layer, x.view(5, groups * k))


def test_dev_ab_adoption_builds_both_arms(monkeypatch):
    groups, rank_out, k = 2, 128, 256
    layer, processed, codes, scales = _mxfp8_layer(monkeypatch, groups, rank_out, k)
    wg.WoAGroupedGemmMethod.adopt(layer, groups, gemm=False, keep_groups=True)
    method = layer.quant_method
    method.process_weights_after_loading(layer)
    assert processed == [(rank_out, k)] * groups and len(layer.grouped) == groups
    assert torch.equal(layer.wo_a_codes, codes) and torch.equal(layer.wo_a_scales, scales)
    calls = _kernel_stand_in(monkeypatch)
    x = torch.randn(4, groups * k).to(BF16).view(4, groups, k)
    want = _official(x, codes, scales, groups)
    # Off arm: the per-group Marlin GEMMs, no kernel call.
    # (FP64 stand-ins summing per group; the einsum may sum in another order.)
    torch.testing.assert_close(method.apply_grouped(layer, x), want,
                               rtol=1e-12, atol=1e-12)
    assert not calls
    method.gemm = True
    torch.testing.assert_close(method.apply_grouped(layer, x), want, rtol=0, atol=0)
    assert len(calls) == 1


def test_off_arm_without_the_marlin_layers_refuses(monkeypatch):
    layer, *_ = _mxfp8_layer(monkeypatch, 2, 128, 256)
    wg.WoAGroupedGemmMethod.adopt(layer, 2)
    layer.quant_method.process_weights_after_loading(layer)
    layer.quant_method.gemm = False
    with pytest.raises(RuntimeError, match="dsv41_wo_a_grouped_gemm"):
        layer.quant_method.apply_grouped(layer, torch.zeros(1, 2, 256, dtype=BF16))


def test_adoption_refusals(monkeypatch, tmp_path):
    layer, *_ = _mxfp8_layer(monkeypatch, 2, 64, 256)
    with pytest.raises(ValueError, match="64 outputs per group"):
        wg.WoAGroupedGemmMethod.adopt(layer, 2)
    layer, *_ = _mxfp8_layer(monkeypatch, 2, 128, 256)
    layer.quant_method.input_dtype = torch.float16
    with pytest.raises(ValueError, match="BF16"):
        wg.WoAGroupedGemmMethod.adopt(layer, 2)
    layer, *_ = _mxfp8_layer(monkeypatch, 2, 128, 256)
    layer.bias = nn.Parameter(torch.zeros(256), requires_grad=False)
    with pytest.raises(ValueError, match="bias"):
        wg.WoAGroupedGemmMethod.adopt(layer, 2)
    plain = SimpleNamespace(quant_method=object(), output_size_per_partition=256,
                            input_size_per_partition=256)
    with pytest.raises(ValueError, match="not an MXFP8"):
        wg.WoAGroupedGemmMethod.adopt(plain, 2)
    layer, *_ = _mxfp8_layer(monkeypatch, 2, 128, 256)
    spec = tmp_path / "capture.toml"
    spec.write_text('dir = "capture"\nfamilies = ["dense", "hybrid"]\n')
    monkeypatch.setenv("VLLM_SM80_CAPTURE", str(spec))
    with pytest.raises(ValueError, match="hybrid capture"):
        wg.WoAGroupedGemmMethod.adopt(layer, 2)
    spec.write_text('dir = "capture"\nfamilies = ["dense", "attn"]\n')
    wg.WoAGroupedGemmMethod.adopt(layer, 2)


def test_process_refuses_unexpected_parameters(monkeypatch):
    layer, *_ = _mxfp8_layer(monkeypatch, 2, 128, 256)
    wg.WoAGroupedGemmMethod.adopt(layer, 2)
    layer.register_parameter("extra", nn.Parameter(torch.zeros(1), requires_grad=False))
    with pytest.raises(ValueError, match="parameters"):
        layer.quant_method.process_weights_after_loading(layer)


def test_switch_default(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", "1")
    assert envs.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM is True


def _attention(wo_a_method):
    from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80

    layer = object.__new__(sm80.DeepseekV41Sm80Attention)
    nn.Module.__init__(layer)
    object.__setattr__(layer, "wo_a", SimpleNamespace(quant_method=wo_a_method))
    return layer


def test_dev_ab_switch_sets_every_grouped_gemm_wo_a(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80 import dev_ab

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", "0")
    switch = dev_ab.SWITCHES["dsv41_wo_a_grouped_gemm"]
    assert switch.env == "VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM"
    assert switch.graph_bound and not switch.output_identical
    target = [wg.WoAGroupedGemmMethod(object(), 2, gemm=False, keep_groups=True)
              for _ in range(2)]
    draft = wg.WoAGroupedGemmMethod(object(), 2, gemm=False, keep_groups=True)
    shared = _attention(target[0])
    runner = SimpleNamespace(
        model=nn.ModuleList([shared, _attention(target[1])]),
        speculator=SimpleNamespace(model=nn.ModuleList([shared, _attention(draft)])),
    )
    assert switch.unavailable(runner) is None and switch.current(runner) is False
    switch.apply(runner, True)
    assert [m.gemm for m in (*target, draft)] == [True] * 3
    assert switch.current(runner) is True
    assert envs.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM is True
    switch.apply(runner, False)
    assert [m.gemm for m in (*target, draft)] == [False] * 3
    assert envs.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM is False
    # Booted without the Marlin layers, or without the grouped GEMM method.
    alone = SimpleNamespace(
        model=nn.ModuleList([_attention(wg.WoAGroupedGemmMethod(object(), 2))]),
        speculator=None,
    )
    assert "VLLM_DEV_AB_SWITCH=dsv41_wo_a_grouped_gemm" in switch.unavailable(alone)
    plain = SimpleNamespace(
        model=nn.ModuleList([_attention(GroupedLinearMethod(object(), 2))]),
        speculator=None,
    )
    assert switch.unavailable(plain)
