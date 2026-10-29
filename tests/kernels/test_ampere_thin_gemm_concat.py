# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the thin-GEMM concat groups (``VLLM_THIN_GEMM_CONCAT``).

CPU:
    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_thin_gemm_concat.py

- Byte identity of the group launch, in the Triton interpreter (one child
  process per check, ``TRITON_INTERPRET=1``): ``thin_gemm_group_split`` and
  ``thin_gemm_group`` against solo ``thin_gemm`` calls, bitwise, for every
  served (group, M) at TP4 and PP4 plus ragged / tiled / three-member groups,
  on order-revealing and on random inputs. Negative controls: a concatenated
  launch whose rotation shift comes from the global N tile index, and one that
  runs every column on member 0's schedule, both differ from the solo calls.
  The Triton 3.7 interpreter feeds bf16 ``tl.dot`` operands to numpy as raw
  uint16 bits. The child replaces its ``create_dot`` with one that widens bf16
  exactly and sums each block in fp64, rounding once to fp32. Each column's
  value then depends on its block order, split and rotation, which are the
  properties under test.
- The group launch keeps every member's solo schedule (BLOCK_N, BLOCK_K,
  SPLIT_K, EVEN_K/EVEN_N, rotation) and output strides at every M 1..65.
- The group kernel compiles for sm_80 at every served (group, M), within the
  shared-memory limit.
- Dispatch: flag off, the package exports its own functions and never imports
  the pairing module. Pairing hit, miss (another input, a bumped version),
  a stale stash when member 1 is skipped, stash cleared after use, M outside
  the group, traced calls, set-but-off reasons, registry from a fake MLA /
  indexer module tree.

GPU (sm_80, skipped otherwise): the same byte-identity checks on the device,
and the pairing through a captured CUDA graph (replay bitwise eager, zero
allocation growth, no miss).
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.sm80.tuning_rows import use_rows
from vllm import envs

BF16 = torch.bfloat16
PKG = "vllm_sm80.kernels.thin_gemm"
CONCAT = "vllm_sm80.kernels.thin_gemm.concat"

# Every CUDA-graph capture size of the served TP4 configuration up to the
# thin-GEMM bound (1-8 under deep drafting at one request, then 10-32).
TP4_MS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)
PP4_MS = (4, 8)
# (Ns, K, Ms). The qkva_kpool members are replicated, so TP4 and PP4 share them.
SERVED = {
    "qb_wqb_tp4": ((4096, 4096), 1536, TP4_MS),
    "qkva_kpool": ((2048, 128), 4096, TP4_MS),
    "qb_wqb_pp4": ((16384, 4096), 1536, PP4_MS),
}
# (M, Ns, K, x row stride or None)
AWKWARD = {
    "tiled_m65_split_k": (65, (2048, 128), 4096, None),
    "m33": (33, (4096, 4096), 1536, None),
    "odd_n": (5, (17, 33), 4096, None),
    "three_members": (16, (2048, 128, 160), 4096, None),
    "ragged_k": (4, (512, 64), 200, None),
    "strided_x": (24, (4096, 4096), 1536, 2048),
}


def _v74():
    from vllm_sm80.kernels.thin_gemm import thin_gemm_v74
    return thin_gemm_v74


# --------------------------------------------------------------- inputs ----
BIG_X = 2.0 ** 16
BIG_W = 2.0 ** 8


def _cancel_inputs(M, Ns, K, seed, dev):
    """Every column gets +2^24 s at k = 0 and -2^24 s at k = K-1. While the big
    term sits in the fp32 accumulator, each small term added rounds to its ulp,
    so any change of block order, split or rotation shows in most outputs."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g)
    x[:, 0] = BIG_X
    x[:, K - 1] = BIG_X
    ws = []
    for N in Ns:
        w = torch.randn(N, K, generator=g)
        s = torch.where(torch.rand(N, generator=g) < 0.5, 1.0, 1.5)
        w[:, 0] = BIG_W * s
        w[:, K - 1] = -BIG_W * s
        ws.append(w.to(BF16).to(dev))
    return x.to(BF16).to(dev), ws


def _random_inputs(M, Ns, K, seed, dev):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g).to(BF16)
    ws = [(torch.randn(N, K, generator=g) * K ** -0.5).to(BF16).to(dev) for N in Ns]
    return x.to(dev), ws


def _frac_diff(a, b):
    return float((a.view(torch.int16) != b.view(torch.int16)).float().mean())


def _strided(x, row_stride):
    if row_stride is None:
        return x
    buf = torch.zeros(x.shape[0], row_stride, dtype=x.dtype, device=x.device)
    buf[:, :x.shape[1]] = x
    return buf[:, :x.shape[1]]


def _sync(dev):
    if dev != "cpu":
        torch.cuda.synchronize()


# --------------------------------------------------------------- checks ----
# Each runs on "cpu" (Triton interpreter, in a child process) or "cuda".

def _assert_group_is_solo(v74, x, ws, dev, tag, accuracy):
    solo = [v74.thin_gemm(x, w) for w in ws]
    split = v74.thin_gemm_group_split(x, ws)
    cat = v74.thin_gemm_group(x, ws)
    _sync(dev)
    for i, (a, b) in enumerate(zip(solo, split)):
        assert b.is_contiguous() and b.shape == a.shape, (tag, i)
        assert torch.equal(a, b), (tag, "member", i, _frac_diff(a, b))
    assert torch.equal(torch.cat(solo, 1), cat), (tag, "concatenated output")
    if accuracy:
        # The solo results are a real GEMM (guards against equal garbage).
        ref = torch.cat([x.double() @ w.double().t() for w in ws], 1).cpu()
        err = (torch.cat(solo, 1).double().cpu() - ref).abs().max().item()
        assert err <= 0.02 * max(1.0, ref.abs().max().item()), (tag, err)


def _check_served(dev, name):
    v74 = _v74()
    Ns, K, ms = SERVED[name]
    for M in ms:
        for kind, make in (("cancel", _cancel_inputs), ("random", _random_inputs)):
            x, ws = make(M, Ns, K, seed=M * 7 + K, dev=dev)
            _assert_group_is_solo(v74, x, ws, dev, (name, M, kind), kind == "random")
        print(f"{name} M{M}: group == solo (cancel, random)")


def _check_awkward(dev, name):
    v74 = _v74()
    M, Ns, K, stride = AWKWARD[name]
    for kind, make in (("cancel", _cancel_inputs), ("random", _random_inputs)):
        x, ws = make(M, Ns, K, seed=M + K, dev=dev)
        _assert_group_is_solo(v74, _strided(x, stride), ws, dev, (name, kind),
                              kind == "random")
    print(f"{name}: group == solo")


def _naive_concat_launch(v74, x, ws, cfg):
    """The solo kernel over cat(ws) on one schedule: what a plain N-concatenated
    launch computes. Its rotation shift comes from the global N tile index."""
    M, K = x.shape
    w = torch.cat(ws, 0)
    N = w.shape[0]
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = cfg
    out = torch.empty(M, N, dtype=BF16, device=x.device)
    even_k = K % (BLOCK_K * SPLIT_K) == 0
    tiles_n = -(-N // BLOCK_N)
    if SPLIT_K == 1:
        p, sp, lock = v74._dummy_fp32(x.device), (0, 0, 0), v74._locks(1, x.device)
    else:
        p = v74._partials(SPLIT_K, M, N, x.device)
        sp, lock = p.stride(), v74._locks(tiles_n, x.device)
    v74._thin_gemm_kernel[(tiles_n, SPLIT_K, 1)](
        x, w, out, p, lock, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1), sp[0], sp[1], sp[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
        EVEN_K=even_k, EVEN_N=N % BLOCK_N == 0, TILED_M=False,
        ROTATE_K=v74._rotate_k(even_k, SPLIT_K, BLOCK_M, BLOCK_N),
        num_warps=num_warps, num_stages=num_stages)
    return out


def _check_negative_global_rotation(dev):
    """Both members rotate with one schedule ((128, 128, 1) at M 4, 32 K blocks),
    and member 0 has 48 tiles, so a global tile index shifts member 1's start by
    144 % 32 = 16 blocks."""
    v74 = _v74()
    M, Ns, K = 4, (6144, 6144), 4096
    cfg = v74._select_config(M, Ns[0], K)
    assert v74._rotate_k(True, cfg[3], cfg[0], cfg[1]), cfg
    for kind, make in (("cancel", _cancel_inputs), ("random", _random_inputs)):
        x, ws = make(M, Ns, K, seed=11, dev=dev)
        solo = [v74.thin_gemm(x, w) for w in ws]
        split = v74.thin_gemm_group_split(x, ws)
        naive = _naive_concat_launch(v74, x, ws, cfg)
        _sync(dev)
        assert all(torch.equal(a, b) for a, b in zip(solo, split)), kind
        assert torch.equal(naive[:, :Ns[0]], solo[0]), kind
        frac = _frac_diff(naive[:, Ns[0]:], solo[1])
        print(f"global-index rotation, {kind} inputs: {frac:.3f} of member 1 differs")
        if kind == "cancel":
            assert frac > 0.5, frac


def _check_negative_one_schedule(dev):
    """qkva_kpool at M 4 on member 0's schedule: the kpool columns lose their
    16-way split."""
    v74 = _v74()
    M, Ns, K = 4, (2048, 128), 4096
    cfg = v74._select_config(M, Ns[0], K)
    assert v74._select_config(M, Ns[1], K)[3] == 16
    x, ws = _cancel_inputs(M, Ns, K, seed=5, dev=dev)
    solo = [v74.thin_gemm(x, w) for w in ws]
    naive = _naive_concat_launch(v74, x, ws, cfg)
    _sync(dev)
    assert torch.equal(naive[:, :Ns[0]], solo[0])
    frac = _frac_diff(naive[:, Ns[0]:], solo[1])
    print(f"one schedule: {frac:.3f} of member 1 differs")
    assert frac > 0.5, frac


CHECKS = {f"served_{n}": (lambda dev, n=n: _check_served(dev, n)) for n in SERVED}
CHECKS.update({f"awkward_{n}": (lambda dev, n=n: _check_awkward(dev, n)) for n in AWKWARD})
CHECKS["negative_global_rotation"] = _check_negative_global_rotation
CHECKS["negative_one_schedule"] = _check_negative_one_schedule


def _exact_bf16_dot():
    """Replace the interpreter's tl.dot with one that widens bf16 exactly and
    sums each block in fp64, rounded once to fp32 (see the module docstring)."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter

    def wide(h):
        if h.dtype == tl.bfloat16:
            return (h.data.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return h.data.astype(np.float64)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        out = np.matmul(wide(a), wide(b)).astype(np.float32) + d.data
        return interpreter.TensorHandle(out.astype(d.data.dtype), d.dtype.scalar)

    interpreter.InterpreterBuilder.create_dot = create_dot


def _interpret(name):
    import vllm
    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(
                   p for p in (root, os.environ.get("PYTHONPATH")) if p))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), name], env=env,
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0 and f"PASS {name}" in r.stdout, (
        r.stdout[-3000:] + r.stderr[-6000:])
    print(r.stdout)


@pytest.mark.parametrize("name", list(CHECKS))
def test_group_is_bitwise_solo_interpreted(name):
    _interpret(name)


# ------------------------------------------------------- launch schedule ----

class _Recorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))
        return launch


@pytest.fixture
def v74_cpu(monkeypatch):
    v74 = _v74()
    arg_names = list(v74._thin_gemm_group_kernel.arg_names)
    monkeypatch.setattr(v74, "_num_sms_cache", 74)
    monkeypatch.setattr(v74, "_WORKSPACE", {})
    rec = _Recorder()
    monkeypatch.setattr(v74, "_thin_gemm_group_kernel", rec)
    return v74, rec, arg_names


@pytest.mark.parametrize("Ns,K", [((4096, 4096), 1536), ((2048, 128), 4096),
                                  ((16384, 4096), 1536), ((17, 33), 4096),
                                  ((2048, 128, 160), 4096)])
def test_group_launch_keeps_each_solo_schedule(v74_cpu, Ns, K):
    v74, rec, arg_names = v74_cpu
    for M in range(1, 66):
        x = torch.empty(M, K, dtype=BF16)
        ws = [torch.empty(N, K, dtype=BF16) for N in Ns]
        rec.launches.clear()
        outs = v74.thin_gemm_group_split(x, ws)
        assert len(rec.launches) == 1, M
        grid, args, kw = rec.launches[0]
        a = dict(zip(arg_names, args))
        members = kw["NUM_MEMBERS"]
        assert members == len(Ns)
        starts = [0, a["C1"], a["C2"]]
        cta = 0
        for i in range(members):
            w, y = a[f"W{i}"], a[f"Y{i}"]
            j = next(k for k, ww in enumerate(ws) if ww is w)
            bm, bn, bk, sk, _, _ = v74._select_config(M, Ns[j], K)
            even_k = K % (bk * sk) == 0
            assert kw["BLOCK_M"] == bm and kw["TILED_M"] == (M > bm), M
            assert (kw[f"BN{i}"], kw[f"BK{i}"], kw[f"SK{i}"]) == (bn, bk, sk), (M, j)
            assert kw[f"EK{i}"] == even_k and kw[f"EN{i}"] == (Ns[j] % bn == 0)
            assert kw[f"RK{i}"] == v74._rotate_k(even_k, sk, bm, bn), (M, j)
            assert y is outs[j] and (a[f"sym{i}"], a[f"syn{i}"]) == y.stride()
            assert a[f"N{i}"] == Ns[j] and a[f"T{i}"] == -(-Ns[j] // bn)
            assert starts[i] == cta, (M, i)
            cta += a[f"T{i}"] * sk
            if sk > 1:  # its own counters: one per (N tile, M tile)
                assert a[f"L{i}"].numel() == a[f"T{i}"] * grid[1]
        assert grid[0] == cta and grid[1] == -(-M // kw["BLOCK_M"])


def _compile_shared(kernel, named_args, consts, num_warps, num_stages):
    """Compile for sm_80 without a GPU (as the JIT specialises) -> shared bytes."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    fn = kernel  # the JITFunction: ASTSource wants it, with arg_names
    sig, const, attrs = {}, {}, {}
    for i, name in enumerate(fn.arg_names):
        if name in consts:
            sig[name], const[name] = "constexpr", consts[name]
            continue
        v = named_args[name]
        if isinstance(v, torch.Tensor):
            sig[name] = {BF16: "*bf16", torch.float32: "*fp32", torch.int32: "*i32"}[v.dtype]
            attrs[(i,)] = [["tt.divisibility", 16]]
        elif v == 1:
            sig[name], const[name] = "constexpr", 1
        else:
            sig[name] = "i32"
            if v % 16 == 0:
                attrs[(i,)] = [["tt.divisibility", 16]]
    src = ASTSource(fn=fn, signature=sig, constexprs=const, attrs=attrs)
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32),
                       options={"num_warps": num_warps, "num_stages": num_stages})
    return k.metadata.shared


@pytest.mark.parametrize("name", list(SERVED))
def test_group_kernel_compiles_for_sm80(name, monkeypatch):
    from tests.kernels import ampere_sass
    if not ampere_sass.available():
        pytest.skip("no Triton CUDA toolchain")
    v74 = _v74()
    real = v74._thin_gemm_group_kernel
    arg_names = list(real.arg_names)
    monkeypatch.setattr(v74, "_num_sms_cache", 74)
    monkeypatch.setattr(v74, "_WORKSPACE", {})
    rec = _Recorder()
    monkeypatch.setattr(v74, "_thin_gemm_group_kernel", rec)
    Ns, K, ms = SERVED[name]
    report = []
    for M in ms:
        rec.launches.clear()
        v74.thin_gemm_group_split(torch.empty(M, K, dtype=BF16),
                                  [torch.empty(N, K, dtype=BF16) for N in Ns])
        _, args, kw = rec.launches[0]
        kw = dict(kw)
        warps, stages = kw.pop("num_warps"), kw.pop("num_stages")
        shared = _compile_shared(real, dict(zip(arg_names, args)), kw, warps, stages)
        report.append((M, warps, stages, shared))
        # sm_80 opt-in maximum per block; the forward-progress check of the
        # standalone gate also needs more than 8 KB.
        assert 8 * 1024 < shared <= 163 * 1024, (name, M, shared)
    print(name, "M, num_warps, num_stages, shared bytes:", report)


# --------------------------------------------------------------- dispatch ----

def test_flag_is_declared_off():
    assert envs.VLLM_THIN_GEMM_CONCAT is False
    assert os.environ.get("VLLM_THIN_GEMM_CONCAT", "0") in ("", "0")


def test_flag_off_exports_the_base_dispatch(monkeypatch):
    """Not a wrapper that checks a flag: the package's own functions, the same
    objects the dispatch hands out and the indexer binds."""
    import vllm_sm80.kernels.thin_gemm as pkg
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    from vllm_sm80.models.glm5next import sm80_indexer as indexer

    assert pkg.ampere_thin_gemm.__module__ == PKG
    assert pkg.thin_linear.__module__ == PKG
    assert indexer.thin_linear is pkg.thin_linear
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM", True, raising=False)
    monkeypatch.setattr(pkg, "use_ampere_thin_gemm", lambda: True)
    assert dispatch_unquantized_gemm("auto") is pkg.ampere_thin_gemm


def test_flag_off_never_consults_the_registry(monkeypatch):
    import vllm_sm80.kernels.thin_gemm as pkg
    concat = _concat()
    calls = []
    fake = SimpleNamespace(
        thin_gemm=lambda x, w: calls.append("solo") or F.linear(x, w),
        thin_gemm_group_split=lambda x, ws: calls.append("group"))
    monkeypatch.setattr(pkg, "impl", lambda: fake)
    monkeypatch.setattr(pkg, "use_ampere_thin_gemm", lambda: True)
    x = torch.randn(4, 256, dtype=BF16)
    w = torch.randn(64, 256, dtype=BF16)
    group = concat._Group(concat._Slot("qb_wqb"), "l0", (w, w.clone()), [4])
    monkeypatch.setitem(concat._MEMBERS, w.data_ptr(), (group, 0))
    assert torch.equal(pkg.ampere_thin_gemm(None, x, w, None), F.linear(x, w))
    assert torch.equal(pkg.thin_linear(x, w), F.linear(x, w))
    assert calls == ["solo", "solo"] and group.slot.stash is None


def test_flag_on_dispatch_follows_the_package_impl(monkeypatch):
    """The flag-on functions resolve ``impl`` through the package at call
    time, as the base dispatch does, so a re-resolved or patched impl is the
    one that runs, solo and grouped."""
    import vllm_sm80.kernels.thin_gemm as pkg
    concat = _concat()
    calls = []
    fake = SimpleNamespace(
        thin_gemm=lambda x, w: calls.append("solo") or "fake",
        thin_gemm_group_split=lambda x, ws: calls.append("group") or ("y0", "y1"))
    monkeypatch.setattr(pkg, "impl", lambda: fake)
    monkeypatch.setattr(pkg, "use_ampere_thin_gemm", lambda: True)
    x = torch.zeros(4, 4096, dtype=BF16)
    w = torch.zeros(1024, 4096, dtype=BF16)
    assert concat.ampere_thin_gemm(None, x, w, None) == "fake"
    assert concat.thin_linear(x, w) == "fake"
    w1 = torch.zeros(1024, 4096, dtype=BF16)
    group = concat._Group(concat._Slot("qb_wqb"), "l0", (w, w1), [4])
    monkeypatch.setitem(concat._MEMBERS, w.data_ptr(), (group, 0))
    monkeypatch.setitem(concat._MEMBERS, w1.data_ptr(), (group, 1))
    assert concat.ampere_thin_gemm(None, x, w, None) == "y0"
    assert concat.thin_linear(x, w1) == "y1"
    assert calls == ["solo", "solo", "group"]


_CHILD = """
import sys
import types
from vllm import envs
import vllm_sm80.kernels.thin_gemm as pkg
from vllm_sm80.kernels.thin_gemm.warmup import warmup_ampere_thin_gemm
from vllm_sm80.models.glm5next import sm80_indexer as indexer
warmup_ampere_thin_gemm(types.SimpleNamespace(device="cpu"), [1, 4])
print("FLAG", envs.VLLM_THIN_GEMM_CONCAT)
print("CONCAT_IMPORTED", "vllm_sm80.kernels.thin_gemm.concat" in sys.modules)
print("DISPATCH", pkg.ampere_thin_gemm.__module__, pkg.thin_linear.__module__,
      indexer.thin_linear is pkg.thin_linear)
"""


@pytest.mark.parametrize("flag", ["unset", "1"])
def test_flag_selects_the_exported_dispatch_in_a_fresh_process(flag):
    import vllm
    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(
                   p for p in (root, os.environ.get("PYTHONPATH")) if p))
    env.pop("VLLM_THIN_GEMM_CONCAT", None)
    env.pop("VLLM_THIN_GEMM", None)
    if flag != "unset":
        env["VLLM_THIN_GEMM_CONCAT"] = flag
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env, cwd=root,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    out = r.stdout
    if flag == "unset":
        assert "FLAG False" in out
        assert "CONCAT_IMPORTED False" in out
        assert f"DISPATCH {PKG} {PKG} True" in out
    else:
        assert "FLAG True" in out
        assert f"DISPATCH {CONCAT} {CONCAT} True" in out
        # VLLM_THIN_GEMM is off in the child: the gate says so, once.
        assert "VLLM_THIN_GEMM_CONCAT set but off: VLLM_THIN_GEMM is off" in (
            out + r.stderr)


def test_warmup_hook_registers_only_with_the_flag(monkeypatch):
    from vllm_sm80.kernels.thin_gemm import warmup as wu
    concat = _concat()
    seen = []
    monkeypatch.setattr(concat, "register_groups",
                        lambda worker, sizes: seen.append(tuple(sizes)))
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM", False, raising=False)
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_CONCAT", False, raising=False)
    wu.warmup_ampere_thin_gemm(SimpleNamespace(device="cpu"), [4])
    assert seen == []
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_CONCAT", True, raising=False)
    wu.warmup_ampere_thin_gemm(SimpleNamespace(device="cpu"), [4])
    assert seen == [(4,)]


# ----------------------------------------------------- fake module tree ----

def _concat():
    import importlib
    return importlib.import_module(CONCAT)


HID, QL, NQ, NWQ, NQKV, GATE = 256, 128, 64, 64, 192, 16


class _Lin(nn.Module):
    def __init__(self, n, k, seed, gemm):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n, k, generator=g).to(BF16),
                                   requires_grad=False)
        self.bias = None
        self.quant_method = SimpleNamespace(_gemm_impl=gemm)

    def forward(self, x):
        return self.quant_method._gemm_impl(self, x, self.weight, None)


def _plain_gemm(layer, x, w, bias):
    return F.linear(x, w, bias)


class _Indexer(nn.Module):
    def __init__(self, seed, dims):
        super().__init__()
        hid, ql, _, nwq, _, gate = dims
        self.wq_b = _Lin(nwq, ql, seed + 1, _concat().ampere_thin_gemm)
        g = torch.Generator().manual_seed(seed + 2)
        self.index_kpool_compress_gate = nn.Parameter(
            torch.randn(gate, hid, generator=g).to(BF16), requires_grad=False)

    def forward(self, hidden_states, q_c):
        q = self.wq_b(q_c)
        gate = _concat().thin_linear(hidden_states, self.index_kpool_compress_gate)
        return q, gate


class _MLA(nn.Module):
    """The MLA wrapper's call order (``mla.py``): fused_qkv_a_proj(hidden_states),
    q_c, q_b_proj(q_c), then the indexer (wq_b(q_c) first, the kpool gate on
    hidden_states last) unless the layer skips it."""

    def __init__(self, seed, dims=(HID, QL, NQ, NWQ, NQKV, GATE), skip_topk=False,
                 quantized_q_b=False):
        super().__init__()
        hid, ql, nq, _, nqkv, _ = dims
        gemm = _concat().ampere_thin_gemm
        self.ql = ql
        self.fused_qkv_a_proj = _Lin(nqkv, hid, seed, gemm)
        self.q_b_proj = _Lin(nq, ql, seed + 3, _plain_gemm if quantized_q_b else gemm)
        self.indexer = _Indexer(seed + 4, dims)
        self.is_sparse = True
        self.skip_topk = skip_topk

    def forward(self, hidden_states):
        qkv = self.fused_qkv_a_proj(hidden_states)
        q_c = qkv[:, :self.ql] * 1  # a new tensor, as q_a_layernorm returns
        q = self.q_b_proj(q_c)
        iq = gate = None
        if self.is_sparse and not self.skip_topk:
            iq, gate = self.indexer(hidden_states, q_c)
        return qkv, q, iq, gate


class _Attention(nn.Module):
    """Holds the same indexer without the wrapper's is_sparse / skip_topk."""

    def __init__(self, mla):
        super().__init__()
        self.mla = mla
        self.indexer = mla.indexer


def _reference(mla, hs):
    qkv = F.linear(hs, mla.fused_qkv_a_proj.weight)
    q_c = qkv[:, :mla.ql] * 1
    return (qkv, F.linear(q_c, mla.q_b_proj.weight),
            F.linear(q_c, mla.indexer.wq_b.weight),
            F.linear(hs, mla.indexer.index_kpool_compress_gate))


class _Log:
    def __init__(self):
        self.lines = []

    def _add(self, msg, *args, **kwargs):
        line = msg % args if args else msg
        if line not in self.lines:
            self.lines.append(line)

    info = info_once = warning_once = warning = _add


class _FakeImpl:
    def __init__(self):
        self.calls = []

    def thin_gemm(self, x, w):
        self.calls.append(("solo", int(w.shape[0])))
        return F.linear(x, w)

    def thin_gemm_group_split(self, x, ws):
        self.calls.append(("group", tuple(int(w.shape[0]) for w in ws)))
        return [F.linear(x, w) for w in ws]

    def warmup(self, shapes, ms, device=None, groups=()):
        self.calls.append(("warmup", tuple(groups), tuple(ms)))


def _worker(model, mode=None, device="cpu"):
    from vllm.config import CompilationMode
    mode = CompilationMode.NONE if mode is None else mode
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(mode=mode)),
        device=torch.device(device), get_model=lambda: model, model_runner=None)


@pytest.fixture
def concat_env(monkeypatch):
    import vllm_sm80.kernels.thin_gemm as pkg
    concat = _concat()
    fake, log = _FakeImpl(), _Log()
    monkeypatch.setattr(pkg, "impl", lambda: fake)
    monkeypatch.setattr(pkg, "use_ampere_thin_gemm", lambda: True)
    monkeypatch.setattr(concat, "_device_supported", lambda: True)
    monkeypatch.setattr(concat, "logger", log)
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM", True, raising=False)
    use_rows(monkeypatch)
    yield concat, fake, log
    concat._MEMBERS.clear()
    concat._SLOTS.clear()


def _model():
    layers = nn.ModuleList([
        _MLA(0), _MLA(10), _MLA(20, skip_topk=True), _MLA(30, quantized_q_b=True)])
    model = nn.Module()
    model.layers = layers
    model.attn = _Attention(layers[0])
    return model


def test_registry_from_a_fake_mla_tree(concat_env):
    concat, fake, log = concat_env
    model = _model()
    concat.register_groups(_worker(model), [4, 16])
    L = model.layers
    qb = {id(L[i]) for i in (0, 1)}
    qkva = {id(L[i]) for i in (0, 1, 3)}
    got = {}
    for (group, i) in concat._MEMBERS.values():
        got.setdefault(group.slot.name, set()).add(group)
    assert {g.slot.name for g in got["qb_wqb"]} == {"qb_wqb"}
    owners = {name: {id(m) for g in gs for m in L if m.q_b_proj.weight is g.ws[0]
                     or m.fused_qkv_a_proj.weight is g.ws[0]} for name, gs in got.items()}
    assert owners == {"qb_wqb": qb, "qkva_kpool": qkva}
    for name, gs in got.items():
        for g in gs:
            assert g.ms == frozenset({1, 2, 3, 4, 8, 16})
            for i, w in enumerate(g.ws):
                assert concat._MEMBERS[w.data_ptr()] == (g, i)
    assert ("warmup", (((NQ, NWQ), QL),), (1, 2, 3, 4, 8, 16)) in fake.calls
    assert ("warmup", (((NQKV, GATE), HID),), (1, 2, 3, 4, 8, 16)) in fake.calls
    assert ("sm_80 thin GEMM concat groups active (qb_wqb, qkva_kpool) "
            "[VLLM_THIN_GEMM_CONCAT=1]") in log.lines
    assert not any("set but off" in line for line in log.lines)


def _registered(concat, capture_sizes=(4, 16)):
    model = _model()
    concat.register_groups(_worker(model), list(capture_sizes))
    return model


def test_pairing_hit_returns_the_solo_bytes(concat_env):
    concat, fake, log = concat_env
    model = _registered(concat)
    hs = torch.randn(4, HID, dtype=BF16)
    fake.calls.clear()
    out = model.layers[0](hs)
    for a, b in zip(out, _reference(model.layers[0], hs)):
        assert torch.equal(a, b)
    assert fake.calls == [("group", (NQKV, GATE)), ("group", (NQ, NWQ))]
    for name in ("qb_wqb", "qkva_kpool"):
        slot = concat._SLOTS[name]
        assert (slot.launches, slot.hits, slot.misses) == (1, 1, 0)
        assert slot.stash is None  # cleared after use
    assert not any("ran alone" in line for line in log.lines)


def test_inference_tensors_pair_on_identity(concat_env):
    concat, fake, _ = concat_env
    model = _registered(concat)
    with torch.inference_mode():
        hs = torch.randn(4, HID, dtype=BF16)
        out = model.layers[1](hs)
    for a, b in zip(out, _reference(model.layers[1], hs)):
        assert torch.equal(a, b)
    assert [c[0] for c in fake.calls[-2:]] == ["group", "group"]
    assert concat._SLOTS["qb_wqb"].hits == 1


def test_miss_on_another_input_object(concat_env):
    concat, fake, log = concat_env
    model = _registered(concat)
    mla = model.layers[0]
    hs = torch.randn(4, HID, dtype=BF16)
    mla.fused_qkv_a_proj(hs)
    fake.calls.clear()
    gate = concat.thin_linear(hs.clone(), mla.indexer.index_kpool_compress_gate)
    assert torch.equal(gate, F.linear(hs, mla.indexer.index_kpool_compress_gate))
    assert fake.calls == [("solo", GATE)]
    assert concat._SLOTS["qkva_kpool"].misses == 1
    assert log.lines.count("VLLM_THIN_GEMM_CONCAT: group qkva_kpool member 1 ran "
                           "alone (input not the group's)") == 1


def test_miss_on_a_bumped_version(concat_env):
    concat, fake, _ = concat_env
    model = _registered(concat)
    mla = model.layers[0]
    q_c = torch.randn(4, QL, dtype=BF16)
    mla.q_b_proj(q_c)
    q_c.mul_(2)  # an in-place write between the members
    fake.calls.clear()
    q = mla.indexer.wq_b(q_c)
    assert torch.equal(q, F.linear(q_c, mla.indexer.wq_b.weight))
    assert fake.calls == [("solo", NWQ)]
    assert concat._SLOTS["qb_wqb"].misses == 1


def test_stale_stash_when_member_1_is_skipped(concat_env):
    """A layer that skips its indexer leaves its stash; the next layer's
    member 0 overwrites it and that layer still pairs."""
    concat, fake, _ = concat_env
    model = _registered(concat)
    a, b = model.layers[0], model.layers[1]
    hs_a = torch.randn(4, HID, dtype=BF16)
    hs_b = torch.randn(4, HID, dtype=BF16)
    a.fused_qkv_a_proj(hs_a)            # member 1 of layer a never runs
    b.fused_qkv_a_proj(hs_b)
    fake.calls.clear()
    gate_b = concat.thin_linear(hs_b, b.indexer.index_kpool_compress_gate)
    assert fake.calls == []             # hit
    assert torch.equal(gate_b, F.linear(hs_b, b.indexer.index_kpool_compress_gate))
    gate_a = concat.thin_linear(hs_a, a.indexer.index_kpool_compress_gate)
    assert fake.calls == [("solo", GATE)]  # a's stash is gone: miss
    assert torch.equal(gate_a, F.linear(hs_a, a.indexer.index_kpool_compress_gate))
    slot = concat._SLOTS["qkva_kpool"]
    assert (slot.launches, slot.hits, slot.misses) == (2, 1, 1)


def test_stash_is_used_once(concat_env):
    concat, fake, _ = concat_env
    model = _registered(concat)
    mla = model.layers[0]
    q_c = torch.randn(4, QL, dtype=BF16)
    mla.q_b_proj(q_c)
    mla.indexer.wq_b(q_c)
    fake.calls.clear()
    q = mla.indexer.wq_b(q_c)  # a second member-1 call on the same input
    assert fake.calls == [("solo", NWQ)]
    assert torch.equal(q, F.linear(q_c, mla.indexer.wq_b.weight))


def test_m_outside_the_group_runs_solo_without_a_miss(concat_env):
    concat, fake, log = concat_env
    model = _registered(concat, capture_sizes=(4, 16))
    hs = torch.randn(5, HID, dtype=BF16)  # M 5 is not a warmed size
    fake.calls.clear()
    out = model.layers[0](hs)
    for x, y in zip(out, _reference(model.layers[0], hs)):
        assert torch.equal(x, y)
    assert [c[0] for c in fake.calls] == ["solo"] * 4
    assert concat._SLOTS["qb_wqb"].misses == 0
    assert concat._SLOTS["qkva_kpool"].misses == 0
    hs = torch.randn(40, HID, dtype=BF16)  # past dense_gemm/thin_max_rows
    fake.calls.clear()
    model.layers[0](hs)
    assert fake.calls == []


def test_traced_calls_never_pair(concat_env, monkeypatch):
    concat, fake, _ = concat_env
    model = _registered(concat)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    hs = torch.randn(4, HID, dtype=BF16)
    fake.calls.clear()
    model.layers[0](hs)
    assert [c[0] for c in fake.calls] == ["solo"] * 4


def test_unregistered_weights_take_the_solo_path(concat_env):
    concat, fake, _ = concat_env
    _registered(concat)
    x = torch.randn(4, HID, dtype=BF16)
    w = torch.randn(32, HID, dtype=BF16)
    fake.calls.clear()
    assert torch.equal(concat.ampere_thin_gemm(None, x, w, None), F.linear(x, w))
    assert torch.equal(concat.thin_linear(x, w), F.linear(x, w))
    assert fake.calls == [("solo", 32), ("solo", 32)]


@pytest.mark.parametrize("case", ["thin_gemm_off", "not_sm80", "compiled",
                                  "no_group", "no_size"])
def test_gate_closed_says_why(concat_env, monkeypatch, case):
    from vllm.config import CompilationMode
    concat, fake, log = concat_env
    model, mode, sizes = _model(), None, [4]
    reason = {
        "thin_gemm_off": "VLLM_THIN_GEMM is off",
        "not_sm80": "not an sm_80 device",
        "compiled": "compilation mode VLLM_COMPILE, needs NONE",
        "no_group": "no MLA layer with an indexer on the thin GEMM",
        "no_size": "no token size within dense_gemm/thin_max_rows",
    }[case]
    if case == "thin_gemm_off":
        monkeypatch.setattr(envs, "VLLM_THIN_GEMM", False, raising=False)
    elif case == "not_sm80":
        monkeypatch.setattr(concat, "_device_supported", lambda: False)
    elif case == "compiled":
        mode = CompilationMode.VLLM_COMPILE
    elif case == "no_group":
        model = nn.Sequential(_Lin(64, HID, 0, concat.ampere_thin_gemm))
    else:
        use_rows(monkeypatch, ("dense_gemm", "thin_max_rows", {}, 0))
        sizes = []
    concat.register_groups(_worker(model, mode), sizes)
    assert concat._MEMBERS == {}
    assert log.lines == [f"VLLM_THIN_GEMM_CONCAT set but off: {reason}"]
    assert not any(c[0] == "warmup" for c in fake.calls)


def test_register_groups_lines_through_the_vllm_logger(concat_env, monkeypatch,
                                                       caplog):
    """The banner and a gate-closed line, emitted through vLLM's own logger
    (its ``info_once`` / ``warning_once``), not a stand-in."""
    import logging

    import vllm.logger as vllm_logger
    from vllm.config import CompilationMode
    concat, _, _ = concat_env
    monkeypatch.setattr(concat, "logger", vllm_logger.init_logger(CONCAT))
    # The vllm logger does not propagate to the root, where caplog listens.
    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    try:
        for once in (vllm_logger._print_info_once, vllm_logger._print_warning_once):
            once.cache_clear()
        concat.register_groups(_worker(_model()), [4, 16])
        concat.register_groups(_worker(_model(), CompilationMode.VLLM_COMPILE), [4])
    finally:
        root.removeHandler(caplog.handler)
    lines = [(r.levelno, r.name, r.getMessage()) for r in caplog.records]
    assert (logging.INFO, CONCAT,
            "sm_80 thin GEMM concat groups active (qb_wqb, qkva_kpool) "
            "[VLLM_THIN_GEMM_CONCAT=1]") in lines
    assert (logging.WARNING, CONCAT,
            "VLLM_THIN_GEMM_CONCAT set but off: compilation mode VLLM_COMPILE, "
            "needs NONE") in lines
    assert concat._MEMBERS == {}



# -------------------------------------------------------------------- GPU ----

def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 0)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU")


@gpu
@pytest.mark.parametrize("name", list(CHECKS))
def test_gpu_group_is_bitwise_solo(name):
    CHECKS[name]("cuda")


TP4_DIMS = (4096, 1536, 4096, 4096, 2048, 128)  # hidden, q_lora, q_b, wq_b, qkv_a, gate


@gpu
@pytest.mark.parametrize("M", [4, 16, 32])
def test_gpu_pairing_capture_replay(monkeypatch, M):
    """The fake MLA tree at TP4 shapes on the real kernels: eager == solo calls,
    a captured graph replays bitwise with no allocation growth and no miss."""
    concat = _concat()
    v74 = _v74()
    log = _Log()
    monkeypatch.setattr(concat, "logger", log)
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM", True, raising=False)
    use_rows(monkeypatch)
    try:
        layers = nn.ModuleList([_MLA(s, TP4_DIMS) for s in (0, 10)]).cuda()
        model = nn.Module()
        model.layers = layers
        concat.register_groups(_worker(model, device="cuda"), [M])
        assert any("concat groups active (qb_wqb, qkva_kpool)" in line
                   for line in log.lines), log.lines
        hs = torch.randn(M, TP4_DIMS[0], device="cuda").to(BF16)

        def forward():
            outs = []
            for mla in layers:
                outs += list(mla(hs))
            return outs

        def solo():
            outs = []
            for mla in layers:
                qkv = v74.thin_gemm(hs, mla.fused_qkv_a_proj.weight)
                q_c = qkv[:, :mla.ql] * 1
                outs += [qkv, v74.thin_gemm(q_c, mla.q_b_proj.weight),
                         v74.thin_gemm(q_c, mla.indexer.wq_b.weight),
                         v74.thin_gemm(hs, mla.indexer.index_kpool_compress_gate)]
            return outs

        ref = solo()
        eager = forward()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(eager, ref))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            captured = forward()
        a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(captured, eager))
        assert (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) == (a0, r0)
        for name in ("qb_wqb", "qkva_kpool"):
            slot = concat._SLOTS[name]
            assert slot.misses == 0 and slot.hits == slot.launches == 4, name
        assert not any("ran alone" in line for line in log.lines)
    finally:
        concat._MEMBERS.clear()
        concat._SLOTS.clear()


if __name__ == "__main__":
    # Child of _interpret (TRITON_INTERPRET=1, no GPU) or a GPU run by hand.
    name = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    if dev == "cpu":
        _exact_bf16_dot()
        _v74()._num_sms_cache = 74
    CHECKS[name](dev)
    print(f"PASS {name}")
