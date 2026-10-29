# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 delayed mHC finish v2 (``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2``,
``ops/sm80_mhc_finish_v2.py``) on CPU.

- The switch is declared and off by default. Off, ``_delayed`` launches
  kernel A without the staging reuse, then ``_delayed_finish_kernel``
  (recorders in place of the kernels).
- The gate: the finish v2 takes exactly the calls inside hidden 5120, 20
  Sinkhorn iterations, 1..2048 rows and a norm, with the finish kernel selected,
  in both forms (fused and pre only); every other call takes the served finish
  or the torch finish. ``finish_v2_unavailable`` names each reason;
  ``mhc_ops`` registers the pre-capture warmup and logs the banner through
  vLLM's logger when the gate is open, clears the switch and says why when it
  is closed.
- Kernel A with the staging reuse off compiles to the SASS of its callers
  that pass no option (GLM-5.3-Flash's), for GLM-5.3-Flash's and this
  model's options; with it on it takes 24 KB of shared memory instead of 32
  (Triton 3.7.1). The finish v2 compiles for sm_80 in every variant the
  wrapper launches.
- The warmup's compile keys: the graph capture sizes up to 2048 rows and one
  row count of each kernel specialisation; it refuses to run with the switch
  off.

The GPU half (FP64 rule, bits of the off path and of kernel A, CUDA graphs)
is ``test_sm80_mhc_finish_v2_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_mhc_finish_v2.py
"""

import hashlib
import logging
import re
import subprocess
import tempfile
from types import SimpleNamespace

import pytest
import torch

from .test_sm80_mhc_fused import _stand_in_kernel_a, call

HIDDEN = 5120
BF16 = torch.bfloat16
# `_glu_post_prenorm_kernel` compiles: (options, HAS_POST, M divisible by 16).
KERNEL_A_KEYS = (
    ("dsv41", False, False), ("dsv41", False, True), ("dsv41", True, False),
    ("dsv41", True, True), ("glm53", True, False), ("glm53", True, True),
)
OPTIONS = {"glm53": dict(HIDDEN=4096, ROUND_MIX=False, FP64_ACC=False),
           "dsv41": dict(HIDDEN=HIDDEN, ROUND_MIX=True, FP64_ACC=True)}


def _args(T: int, hidden: int = HIDDEN, sink: int = 20, *, fused: bool = True,
          carried: bool = True, norm: bool = True) -> dict:
    """One call's arguments (zeros: the dispatch, not the values, is tested)."""
    hc = 4
    a = dict(
        x=torch.zeros(T, hidden, dtype=BF16) if fused else None,
        residual=torch.zeros(T, hc, hidden, dtype=BF16),
        post_layer_mix=torch.ones(T, hc, 1) if fused else None,
        comb_res_mix=torch.full((T, hc, hc), 0.25) if fused else None,
        fn=torch.zeros((2 + hc) * hc, hc * hidden),
        hc_scale=torch.ones(3),
        hc_base=torch.zeros((2 + hc) * hc),
        rms_eps=1e-20,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=sink,
        pre_mix=torch.full((T, hc), 0.25) if carried else None,
        norm_weight=torch.ones(hidden, dtype=BF16) if norm else None,
        norm_eps=1e-6,
    )
    return a


class _Launch:
    """Stands in for a kernel launched as ``kernel[grid](...)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


@pytest.fixture
def recorded(monkeypatch):
    """``sm80_mhc_fused`` with kernel A replaced by its torch stand-in (the
    keyword arguments recorded), both finish kernels by recorders."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

    rec = SimpleNamespace(kernel_a=[], served=_Launch(), v2=[])

    def kernel_a(*args, **kwargs):
        rec.kernel_a.append(kwargs)
        return _stand_in_kernel_a(*args, **kwargs)

    monkeypatch.setattr(fused, "post_prenorm", kernel_a)
    monkeypatch.setattr(fused, "_delayed_finish_kernel", rec.served)
    monkeypatch.setattr(finish_v2, "delayed_finish_v2",
                        lambda *args, **kwargs: rec.v2.append((args, kwargs)))
    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(fused, "FINISH_V2", False)
    rec.fused = fused
    return rec


def test_switch_declared_and_off_by_default(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2 is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2", "1")
    assert envs.VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2 is True
    assert isinstance(sm80_mhc_fused.FINISH_V2, bool)


@pytest.mark.parametrize("fused_form", [True, False], ids=["fused", "pre_only"])
def test_off_path_launches_are_unchanged(recorded, fused_form):
    call(recorded.fused, _args(17, fused=fused_form))
    has_post = {} if fused_form else {"has_post": False}
    assert recorded.kernel_a == [dict(has_post, round_mix=True, fp64_acc=True, hs_reuse=False)]
    assert not recorded.v2 and len(recorded.served.calls) == 1


@pytest.mark.parametrize("fused_form", [True, False], ids=["fused", "pre_only"])
@pytest.mark.parametrize("carried", [True, False], ids=["carried", "identity"])
@pytest.mark.parametrize("T", [1, 2048])
def test_gate_takes_the_validated_calls(recorded, T, fused_form, carried):
    recorded.fused.FINISH_V2 = True
    call(recorded.fused, _args(T, fused=fused_form, carried=carried))
    assert [k["hs_reuse"] for k in recorded.kernel_a] == [True]
    assert not recorded.served.calls
    (args, kwargs), = recorded.v2
    assert kwargs == dict(has_pre_in=carried, has_norm=True)
    mix, sqr, *_ = args
    assert mix.shape == (160, T, 24) and sqr.shape == (160, T)
    assert args[-5:] == (1e-20, 1e-6, 2.0, 1e-6, 20)


@pytest.mark.parametrize("case", [dict(T=2049), dict(T=6, hidden=1056), dict(T=6, sink=19),
                                  dict(T=6, norm=False)],
                         ids=["rows", "hidden", "sinkhorn", "no_norm"])
def test_gate_leaves_every_other_call_to_the_served_finish(recorded, case):
    recorded.fused.FINISH_V2 = True
    T = case["T"]
    call(recorded.fused, _args(T, case.get("hidden", HIDDEN), case.get("sink", 20),
                               norm=case.get("norm", True)))
    assert [k["hs_reuse"] for k in recorded.kernel_a] == [False]
    assert not recorded.v2 and len(recorded.served.calls) == 1


def test_switch_without_the_finish_kernel_keeps_the_torch_finish(recorded):
    recorded.fused.FINISH_V2 = True
    recorded.fused.FINISH_KERNEL = False
    out = call(recorded.fused, _args(6))
    assert [k["hs_reuse"] for k in recorded.kernel_a] == [False]
    assert not recorded.v2 and not recorded.served.calls
    assert out["layer_input"].shape == (6, HIDDEN)


def test_finish_v2_unavailable_names_each_reason(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    assert fused.finish_v2_unavailable(4, HIDDEN, 20) is None
    assert "hc_mult 2" in fused.finish_v2_unavailable(2, HIDDEN, 20)
    assert "hidden size 4096" in fused.finish_v2_unavailable(4, 4096, 20)
    assert "Sinkhorn" in fused.finish_v2_unavailable(4, HIDDEN, 10)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    assert "sm_80" in fused.finish_v2_unavailable(4, HIDDEN, 20)
    monkeypatch.setattr(fused, "FINISH_KERNEL", False)
    assert "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1" in fused.finish_v2_unavailable(4, HIDDEN, 20)


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_mhc_ops_registers_the_warmup_and_logs_the_banner(monkeypatch):
    import vllm.logger
    from vllm_sm80.models.deepseek_v41 import sm80_model
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

    registered = []
    monkeypatch.setattr(finish_v2.FINISH_V2_WARMUP, "register_warmup",
                        lambda *a, **k: registered.append((a, k)))
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_MHC", "1")
    monkeypatch.setattr(fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(fused, "FINISH_V2", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    lines = _Lines()
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(lines)
    level = vllm_logger.level
    vllm_logger.setLevel(logging.INFO)
    try:
        vllm.logger._print_info_once.cache_clear()
        cfg = SimpleNamespace(hc_mult=4, hidden_size=HIDDEN, hc_sinkhorn_iters=20)
        assert sm80_model.mhc_ops(cfg) is fused
        assert fused.FINISH_V2 is True and registered == [((), {})]
        assert any("finish v2 live (VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2=1)" in ln
                   for ln in lines.lines)
        # closed: the switch is cleared, nothing registered, the line says why
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
        assert sm80_model.mhc_ops(cfg) is fused
        assert fused.FINISH_V2 is False and len(registered) == 1
        assert any("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2 is set but its gate is closed "
                   "(validated on sm_80 only)" in ln for ln in lines.lines)
        # off (the default): no line, no registration
        n = len(lines.lines)
        assert sm80_model.mhc_ops(cfg) is fused
        assert len(registered) == 1
        assert not any("finish v2" in ln for ln in lines.lines[n:])
    finally:
        vllm_logger.removeHandler(lines)
        vllm_logger.setLevel(level)


def test_warmup_keys_and_refusal(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

    w = finish_v2.FINISH_V2_WARMUP
    sizes = [4, 5, 6, 7, 8, 12, 16, 18, 24, 28, 32, 40, 4096]
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes))
    rows = [k.rows for k in w.get_warmup_keys(cfg)]
    assert rows == sorted(set(sizes[:-1]) | set(finish_v2.WARMUP_ROWS))
    none = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=None))
    assert [k.rows for k in w.get_warmup_keys(none)] == sorted(finish_v2.WARMUP_ROWS)
    # one of each specialisation: kernel A's transposed form (<= 8 rows), its
    # 16-row form without (9..16) and with (17+) the reuse; M == 1, M % 16 == 0
    assert {1, 2, 9, 16, 17, 32} <= set(finish_v2.WARMUP_ROWS)
    monkeypatch.setattr(fused, "FINISH_V2", False)
    with pytest.raises(RuntimeError, match="finish v2 selected"):
        w.compile(w.Key(6))


def _cuobjdump():
    import os

    import triton

    path = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
    return path if os.path.exists(path) else None


def _compile(kern, types, consts, M=None):
    """``kern`` compiled for sm_80 without a device, the JIT's specialisation
    of 16-byte aligned pointers and of ``M`` when it is divisible by 16."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource

    sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, p in enumerate(kern.arg_names)
             if p.endswith("_ptr")}
    if M is not None and M % 16 == 0:
        attrs[(kern.arg_names.index("M"),)] = [["tt.divisibility", 16]]
    return triton.compile(GluonASTSource(kern, sig, constexprs=consts, attrs=attrs),
                          target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})


def _sass_digest(compiled) -> str:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(compiled.asm["cubin"])
        f.flush()
        out = subprocess.run([_cuobjdump(), "-sass", f.name], capture_output=True, text=True,
                             check=True).stdout
    lines = [re.sub(r"\s*/\*[0-9a-f]+\*/\s*", " ", ln).strip() for ln in out.splitlines()
             if re.search(r"/\*[0-9a-f]{4}\*/", ln)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


A_TYPES = {"x_ptr": "*bf16", "res_ptr": "*bf16", "post_ptr": "*fp32", "comb_ptr": "*fp32",
           "fn_ptr": "*fp32", "rc_ptr": "*bf16", "mix_ptr": "*fp32", "sqr_ptr": "*fp32",
           "M": "i32"}


@pytest.mark.parametrize("key", KERNEL_A_KEYS)
def test_kernel_a_without_the_reuse_keeps_its_sass(key):
    import triton

    if triton.__version__ != "3.7.1" or _cuobjdump() is None:
        pytest.skip("the shared-memory sizes are of Triton 3.7.1's sm_80 code")
    from vllm_sm80.kernels.mhc import decode

    options, has_post, div16 = key
    M = 32 if div16 else 17
    consts = dict(OPTIONS[options], NOUT=24, HB=decode.HB_GLU, BM=decode.BM_A, NP2=32,
                  NW=decode.WARPS_GLU, HAS_POST=has_post)
    off = _compile(decode._glu_post_prenorm_kernel, A_TYPES, dict(consts, HS_REUSE=False), M)
    # the GLM-5.3-Flash callers pass no HS_REUSE
    default = _compile(decode._glu_post_prenorm_kernel, A_TYPES, consts, M)
    assert _sass_digest(off) == _sass_digest(default)
    on = _compile(decode._glu_post_prenorm_kernel, A_TYPES, dict(consts, HS_REUSE=True), M)
    assert (off.metadata.shared, on.metadata.shared) == (32768, 24576)
    assert _sass_digest(on) != _sass_digest(off)


def test_finish_v2_compiles_for_sm80():
    """Every variant the wrapper launches (carried pre-mix or not, always
    with the norm; M divisible by 16 is a specialisation) at hidden 5120."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2

    types = {p: "*fp32" for p in ("mix_ptr", "sqr_ptr", "scale_ptr", "base_ptr", "pre_in_ptr",
                                  "post_ptr", "comb_ptr", "pre_ptr")}
    types.update(rc_ptr="*bf16", nw_ptr="*bf16", li_ptr="*bf16", M="i32", rms_eps="fp32",
                 hc_eps="fp32", post_mult="fp32", norm_eps="fp32")
    for has_pre in (True, False):
        for M in (6, 32):
            # PART given: a compile from source takes no default (the launch's
            # does); the parts launched apart are test_sm80_mhc_side.py's
            c = _compile(finish_v2._delayed_finish_v2_kernel, types,
                         dict(HIDDEN=HIDDEN, KS=160, KA=128, KB=32, SINK=20, HAS_PRE_IN=has_pre,
                              HAS_NORM=True, BH=finish_v2.BH_FIN, PART=0), M)
            assert c.asm["cubin"]
