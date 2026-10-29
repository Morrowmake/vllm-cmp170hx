# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mHC coefficients beside the sublayer's all-reduce
(``VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM``, ``ops/sm80_mhc_side.py``) on CPU.

- The switch is declared and off by default; ``layer_scope`` takes only the
  fused path with the finish kernel and refuses microbatching; taking it
  marks the scope in use; a decoder layer's forward runs inside its scope.
- The order of work, with the streams, the event and the finish kernels
  replaced by recorders: inside a layer's forward a finish launches its
  layer-input CTAs in place and defers its coefficient CTAs; the all-reduce
  forks them onto the side stream after the current stream's work and joins
  them after its own; without an all-reduce the next mHC call (before its
  kernel A) or the end of the forward launches them inline; outside a
  forward, or above ``MAX_ROWS`` rows, the one launch of before; an
  all-reduce or flush on another stream launches them inline on the stream
  they were deferred on, and waits for it.
- The bits: the Triton finish launched apart (layer-input CTAs, then the
  coefficient CTAs; outputs filled with NaN first) writes the bits of its one
  launch, in the Triton interpreter, for every call form.
- The code (Triton 3.7.1, sm_80, no device): ``PART`` 0 compiles to the SASS
  of both finish kernels before the option (pinned digests), and the
  coefficient-only and layer-input-only kernels hold between them exactly
  the PART 0 kernel's floating-point arithmetic (PTX ops with their rounding
  suffixes; SASS FP instructions, ptxas's contractions included).
- On an sm_80 card (skipped without one): the served finish v2, compiled
  for the card, launched apart (PART 2 in place, PART 1 on the side stream
  beside an all-reduce, or inline at the forward's end) writes the bits of
  its one PART 0 launch, every output NaN-filled first, at 1..40 rows.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_mhc_side.py
"""

import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from .test_sm80_mhc_finish_v2 import _args, _cuobjdump
from .test_sm80_mhc_fused import ROOT, _stand_in_kernel_a, call, variants

# sha256 of every SASS instruction line of the finish kernels before PART,
# compiled for sm_80 by Triton 3.7.1 (no device) at hidden 5120: keyed by
# (kernel, HAS_PRE_IN, HAS_NORM).
BASE_SASS = {
    ("v1", True, True): "dc01aaf12f270c047163bf548db2bca16eafd07cf7322060306bfd94992f3e15",
    ("v1", True, False): "301e5fb7298680ed25ad1ece2d14fba0bffe1da4585f5462b64e405339b3706a",
    ("v1", False, True): "18e4f2d62fbb66e693a46127bfe4186fa7353bd33fd0bc3726cc5fd3bf23e8cc",
    ("v1", False, False): "a6f2968292e7e2644a2d3bf42bf8505811f4565a9b1d92d402ba2276061d734a",
    ("v2", True, True): "52d40315f701b0021c4f71007e80912a2ac89ea3df33dfa233eb6523f6e4a82e",
    ("v2", False, True): "42aec4e71e6c85a847855b52d87a256836f59268cfbc0050bab7886b79b74e6c",
}


def test_switch_declared_and_off_by_default(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM", "1")
    assert envs.VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM is True
    assert isinstance(sm80_mhc_side.ACTIVE, bool)


def test_layer_scope_takes_only_the_fused_finish_kernel(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side as side

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    assert side.layer_scope(fused) is side.LAYER
    assert side.layer_scope(sm80_mhc) is None
    assert side.layer_scope(SimpleNamespace()) is None  # the kernel call capture's wrapper
    monkeypatch.setattr(fused, "FINISH_KERNEL", False)
    assert side.layer_scope(fused) is None


def test_layer_scope_refuses_microbatching_and_marks_the_warmup(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side as side

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(side, "IN_USE", False)
    monkeypatch.setattr(side, "_microbatching", lambda: True)
    with pytest.raises(ValueError, match="microbatching"):
        side.layer_scope(fused)
    assert side.layer_scope(sm80_mhc) is None  # inert: nothing to refuse
    assert side.IN_USE is False
    monkeypatch.setattr(side, "_microbatching", lambda: False)
    assert side.layer_scope(fused) is side.LAYER and side.IN_USE is True


def test_layer_forward_runs_inside_its_scope(monkeypatch):
    from vllm_sm80.models.deepseek_v41 import sm80_model
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side as side

    seen = []
    monkeypatch.setattr(side, "_deferred", None)
    main = object()
    monkeypatch.setattr(side, "current_stream", lambda: main)
    layer = SimpleNamespace(_mhc_side=side.LAYER)
    layer._forward_layer = lambda *a, **k: (seen.append((side._open, a, k)),
                                            side.defer(lambda: seen.append("flushed")))
    sm80_model.DeepseekV4DecoderLayer.forward(layer, 1, 2, capture_previous_aux=True)
    assert seen == [(True, (1, 2), {"capture_previous_aux": True}), "flushed"]
    assert not side._open and side._deferred is None
    layer = SimpleNamespace(_mhc_side=None, _forward_layer=lambda *a: seen.append(side._open))
    sm80_model.DeepseekV4DecoderLayer.forward(layer)
    assert seen[-1] is False


class _Stream:
    def __init__(self, name, log):
        self.name, self.log, self.device = name, log, torch.device("cuda", 0)

    def wait_stream(self, other):
        self.log.append(("wait_stream", self.name, other.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))


class _Event:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record(self, stream):
        self.log.append(("record", stream.name, self.name))


@pytest.fixture
def side(monkeypatch):
    """``sm80_mhc_side`` with a recorded current stream, side stream and event
    (CPU tensors standing for the device's), ``sm80_mhc_fused`` with kernel
    A's stand-in and the finish kernels recorded with the stream they are
    launched on."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side as mhc_side

    log = []
    cur = [_Stream("main", log)]

    @contextmanager
    def stream(s):
        prev, cur[0] = cur[0], s
        try:
            yield
        finally:
            cur[0] = prev

    class Finish:
        def __getitem__(self, grid):
            return lambda *a, **k: log.append(("finish", cur[0].name, grid, k.get("PART")))

    def kernel_a(*args, **kwargs):
        log.append(("kernel_a", cur[0].name))
        return _stand_in_kernel_a(*args, **kwargs)

    monkeypatch.setattr(mhc_side, "current_stream", lambda: cur[0])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(mhc_side, "_side", {0: (_Stream("side", log), _Event("done", log))})
    monkeypatch.setattr(mhc_side, "defers", lambda rows, device: (
        mhc_side._open and 0 < rows <= mhc_side.MAX_ROWS))
    monkeypatch.setattr(mhc_side, "_deferred", None)
    monkeypatch.setattr(fused, "post_prenorm", kernel_a)
    monkeypatch.setattr(fused, "_delayed_finish_kernel", Finish())
    monkeypatch.setattr(finish_v2, "delayed_finish_v2", lambda *a, **k: log.append(
        ("finish_v2", cur[0].name, k.get("part"))))
    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(fused, "FINISH_V2", False)

    def all_reduce(t):
        log.append(("all_reduce", cur[0].name))
        return t

    return SimpleNamespace(fused=fused, side=mhc_side, log=log, all_reduce=all_reduce)


FORK_JOIN = [("wait_stream", "side", "main"), ("finish", "side", (6, 1), 1),
             ("record", "side", "done"), ("all_reduce", "main"), ("wait_event", "main", "done")]


def test_coefficients_go_beside_the_all_reduce(side):
    with side.side.LAYER:
        call(side.fused, _args(6))
        out = side.side.beside(side.all_reduce, "partial")
        assert out == "partial"
        # nothing waits for an all-reduce without a deferred finish
        side.side.beside(side.all_reduce, "partial")
    assert side.log == [("kernel_a", "main"), ("finish", "main", (6, 1), 2), *FORK_JOIN,
                        ("all_reduce", "main")]
    assert side.side._deferred is None


def test_without_an_all_reduce_the_next_call_or_the_forward_end_launches_them(side):
    with side.side.LAYER:
        call(side.fused, _args(6))
        call(side.fused, _args(6))
    assert side.log == [
        ("kernel_a", "main"), ("finish", "main", (6, 1), 2),
        ("finish", "main", (6, 1), 1),  # the next call, before its kernel A
        ("kernel_a", "main"), ("finish", "main", (6, 1), 2),
        ("finish", "main", (6, 1), 1),  # the end of the forward
    ]


def test_one_launch_outside_a_forward_or_above_max_rows(side):
    rows = side.side.MAX_ROWS + 1
    call(side.fused, _args(6))
    with side.side.LAYER:
        call(side.fused, _args(rows))
        side.side.beside(side.all_reduce, "partial")
    assert side.log == [("kernel_a", "main"), ("finish", "main", (6, 2), None),
                        ("kernel_a", "main"), ("finish", "main", (rows, 2), None),
                        ("all_reduce", "main")]


def test_finish_v2_goes_apart_the_same_way(side):
    side.fused.FINISH_V2 = True
    with side.side.LAYER:
        call(side.fused, _args(6))
        side.side.beside(side.all_reduce, "partial")
    call(side.fused, _args(6))
    fork = [("finish_v2", "side", 1) if e[0] == "finish" else e for e in FORK_JOIN]
    assert side.log == [("kernel_a", "main"), ("finish_v2", "main", 2), *fork,
                        ("kernel_a", "main"), ("finish_v2", "main", None)]


def test_another_streams_all_reduce_launches_them_inline_where_deferred(side):
    import torch as t

    other = _Stream("other", side.log)
    with side.side.LAYER:
        call(side.fused, _args(6))
        with t.cuda.stream(other):
            side.side.beside(side.all_reduce, "partial")
        assert side.side._deferred is None
    assert side.log == [("kernel_a", "main"), ("finish", "main", (6, 1), 2),
                        ("finish", "main", (6, 1), 1), ("wait_stream", "other", "main"),
                        ("all_reduce", "other")]


def test_a_flush_on_another_stream_launches_them_where_deferred(side):
    import torch as t

    other = _Stream("other", side.log)
    with side.side.LAYER:
        call(side.fused, _args(6))
        with t.cuda.stream(other):
            side.side.flush()
    assert side.log == [("kernel_a", "main"), ("finish", "main", (6, 1), 2),
                        ("finish", "main", (6, 1), 1), ("wait_stream", "other", "main")]


def interpreted_split_bits(T: int = 5, hidden: int = 1056, calls: int = 3) -> dict:
    """In the Triton interpreter: per call form and output, whether the Triton
    finish launched apart (layer-input CTAs, then the coefficient CTAs) wrote
    the bits of its one launch, every output filled with NaN before."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side as side

    fused.post_prenorm = _stand_in_kernel_a
    fused.FINISH_KERNEL, fused.FINISH_V2 = True, False
    side.defers = lambda rows, device: side._open
    side.current_stream = lambda: None  # CPU tensors: no device stream
    empty = torch.empty

    def nan_empty(*args, **kwargs):
        t = empty(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    torch.empty = nan_empty
    out = {}
    for seed in range(calls):
        for name, a in variants(T, hidden, seed).items():
            one = call(fused, a)
            with side.LAYER:
                apart = call(fused, a)
                assert side._deferred is not None
            for o, v in one.items():
                w = apart[o]
                bits = torch.int16 if v.dtype == torch.bfloat16 else torch.int32
                same = (not bool(v.isnan().any()) and v.shape == w.shape
                        and torch.equal(v.view(bits), w.view(bits)))
                out[f"{name}/{o}"] = out.get(f"{name}/{o}", True) and same
    return out


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.models.deepseek_v41.sm80.test_sm80_mhc_side import "
            "interpreted_split_bits as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_finish_apart_writes_the_bits_of_one_launch(interpreted):
    # fused forms: residual + 4 outputs; pre-only forms: 4 outputs
    assert len(interpreted) == 3 * 5 + 2 * 4
    assert all(interpreted.values()), {k: v for k, v in interpreted.items() if not v}


_FP_SASS = {"DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "FADD", "FMUL", "FFMA", "FSETP", "FMNMX",
            "FSEL", "FRND", "MUFU", "F2F", "F2I", "I2F", "F2FP", "HADD2", "HMUL2", "HFMA2", "FCHK"}
_FP_PTX = re.compile(r"^(?:@%p\d+ )?((?:add|sub|mul|fma|div|rcp|sqrt|rsqrt|ex2|lg2|cvt|neg|abs"
                     r"|min|max|selp|setp|testp|copysign)\.[\w.]+)")


def _compile(kern, consts, gluon):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.experimental.gluon._runtime import GluonASTSource

    types = {p: "*fp32" for p in ("mix_ptr", "sqr_ptr", "scale_ptr", "base_ptr", "pre_in_ptr",
                                  "post_ptr", "comb_ptr", "pre_ptr")}
    types.update(rc_ptr="*bf16", nw_ptr="*bf16", li_ptr="*bf16", M="i32", rms_eps="fp32",
                 hc_eps="fp32", post_mult="fp32", norm_eps="fp32")
    sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, p in enumerate(kern.arg_names)
             if p.endswith("_ptr")}
    source = (GluonASTSource if gluon else ASTSource)(kern, sig, constexprs=consts, attrs=attrs)
    return triton.compile(source, target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})


def _sass(compiled) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(compiled.asm["cubin"])
        f.flush()
        out = subprocess.run([_cuobjdump(), "-sass", f.name], capture_output=True, text=True,
                             check=True).stdout
    return [re.sub(r"\s*/\*[0-9a-f]+\*/\s*", " ", ln).strip() for ln in out.splitlines()
            if re.search(r"/\*[0-9a-f]{4,}\*/", ln)]


def _sass_fp(lines) -> collections.Counter:
    """FP instructions with their modifiers; constants materialised into a
    register (``HFMA2.MMA Rd, -RZ, RZ, ...``) are moves, not arithmetic."""
    ops = collections.Counter()
    for ln in lines:
        tok = ln.split()
        if tok and tok[0].startswith("@"):
            tok = tok[1:]
        if not tok or tok[0].split(".")[0] not in _FP_SASS:
            continue
        if tok[0] == "HFMA2.MMA" and tok[2:4] == ["-RZ,", "RZ,"]:
            continue
        ops[tok[0]] += 1
    return ops


def _ptx_fp(compiled) -> collections.Counter:
    ops = collections.Counter()
    for ln in compiled.asm["ptx"].splitlines():
        m = _FP_PTX.match(re.sub(r"\s+", " ", ln.strip()))
        if m and re.search(r"\.(f64|f32|bf16|bf16x2|f16|rn|rz|approx|ftz)\b", m.group(1)):
            ops[m.group(1)] += 1
    return ops


@pytest.mark.parametrize("key", sorted(BASE_SASS))
def test_parts_keep_the_finish_code(key):
    import triton

    if triton.__version__ != "3.7.1" or _cuobjdump() is None:
        pytest.skip("the digests are of Triton 3.7.1's sm_80 code")
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

    kernel, has_pre, has_norm = key
    if kernel == "v1":
        kern, gluon = fused._delayed_finish_kernel, False
        consts = dict(HIDDEN=5120, KS=160, KSP2=256, SINK=20, BH=fused._BH, ACCURATE=True)
    else:
        kern, gluon = finish_v2._delayed_finish_v2_kernel, True
        consts = dict(HIDDEN=5120, KS=160, KA=128, KB=32, SINK=20, BH=finish_v2.BH_FIN)
    consts.update(HAS_PRE_IN=has_pre, HAS_NORM=has_norm)
    both, coefficients, layer_input = (_compile(kern, dict(consts, PART=p), gluon)
                                       for p in (0, 1, 2))
    lines = [_sass(c) for c in (both, coefficients, layer_input)]
    assert hashlib.sha256("\n".join(lines[0]).encode()).hexdigest() == BASE_SASS[key]
    assert _ptx_fp(both) == _ptx_fp(coefficients) + _ptx_fp(layer_input)
    assert _sass_fp(lines[0]) == _sass_fp(lines[1]) + _sass_fp(lines[2])
    assert sum(_sass_fp(lines[1]).values()) and sum(_sass_fp(lines[2]).values())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 card",
)
@pytest.mark.parametrize("T", [1, 6, 16, 32, 40])
def test_finish_v2_apart_on_the_card_writes_the_bits_of_one_launch(monkeypatch, T):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_side as side

    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(fused, "FINISH_V2", True)
    monkeypatch.setattr(side, "_deferred", None)
    empty = torch.empty

    def nan_empty(*args, **kwargs):
        t = empty(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    monkeypatch.setattr(torch, "empty", nan_empty)

    def joined(a):
        # PART 2 in place, PART 1 on the side stream beside an all-reduce.
        with side.LAYER:
            out = call(fused, a)
            assert side._deferred is not None
            side.beside(lambda x: x, None)
            assert side._deferred is None
        return out

    def flushed(a):
        # PART 2 in place, PART 1 inline at the end of the forward.
        with side.LAYER:
            out = call(fused, a)
            assert side._deferred is not None
        return out

    checked = 0
    for seed in range(2):
        for name, a in variants(T, 5120, seed).items():
            a = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in a.items()}
            one = call(fused, a)
            for how in (joined, flushed):
                apart = how(a)
                torch.cuda.synchronize()
                for o, v in one.items():
                    w = apart[o]
                    bits = torch.int16 if v.dtype == torch.bfloat16 else torch.int32
                    where = f"{name}/{o} rows {T} seed {seed} {how.__name__}"
                    assert not v.isnan().any(), where
                    assert v.shape == w.shape and torch.equal(v.view(bits), w.view(bits)), where
                    checked += 1
    # fused forms: residual + 4 outputs; pre-only forms: 4 outputs
    assert checked == 2 * 2 * (3 * 5 + 2 * 4)
