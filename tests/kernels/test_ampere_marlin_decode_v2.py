# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_AMPERE_MARLIN_DECODE_V2: dispatch, gate, banners, the optional op, and
on sm_80 bitwise equality with decode_gemm_orig (eager and CUDA-graph replay)."""
import importlib.util
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.kernels.marlin_moe import decode, extension

from tests.kernels import test_ampere_pp_marlin_prefill as pp_helpers

DECODE = "VLLM_AMPERE_MARLIN_DECODE_CUDA"
V2 = "VLLM_AMPERE_MARLIN_DECODE_V2"
ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "csrc/ampere_marlin"
E, TOPK, K, N = 288, 8, 4096, 512

# decode_gemm_v2's measured per-row-count codes (w13, w2): 2-3 rows cfg 2, 4-7 rows
# cfg 3, 32 rows cfg 9 capped at 3 CTAs per SM, otherwise decode_gemm_orig's codes.
V2_CONFIG = {m: ((2, 2) if 2 <= m < 4 else (3, 3) if 4 <= m < 8 else
                 (57, 57) if m >= 32 else
                 (49 if m >= 16 else 0, 49 if m > 16 else 0))
             for m in range(1, 33)}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in (DECODE, V2, "VLLM_AMPERE_DECODE_IDX_GLUE",
                 "VLLM_AMPERE_DECODE_KERNELS"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(extension, "_OPS", None)
    monkeypatch.setattr(extension, "_DECODE_V2", None)
    from vllm.logger import _print_info_once
    _print_info_once.cache_clear()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _logged(fn, logger_name=decode.logger.name):
    """Run fn with a real vLLM logger (decode.py's by default); return
    (result, lines)."""
    handler = _Capture()
    lg = logging.getLogger(logger_name)
    level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        return fn(), handler.lines
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)


# ------------------------------------------------------------------ CPU ----
def test_flag_default_and_declaration(monkeypatch):
    import vllm.envs as envs
    from vllm_sm80 import envs as fork_envs

    monkeypatch.delenv(V2, raising=False)
    assert envs.VLLM_AMPERE_MARLIN_DECODE_V2 is False
    assert V2 in fork_envs.FORK_ENV_VARS
    assert f"    {V2}: bool = False\n" in open(fork_envs.__file__).read()
    monkeypatch.setenv(V2, "1")
    assert envs.VLLM_AMPERE_MARLIN_DECODE_V2 is True


def test_config_v2_table():
    assert {m: decode._config_v2(m) for m in range(1, 33)} == V2_CONFIG
    # rows outside 2..7 and 32 keep decode_gemm_orig's codes
    for m in (1, 8, 15, 16, 17, 31):
        assert decode._config_v2(m) == decode._config(m)


def _bf16(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize("value,ok", [
    (0.0, True), (-0.0, True), (2.0 ** -110, True), (-(2.0 ** -110), True),
    (2.0 ** -111, False), (2.0 ** -126, False), (1e-40, False),
    (1016.0, True), (-1016.0, True), (2.0 ** 10, False), (-(2.0 ** 10), False),
    (float("inf"), False), (float("-inf"), False), (float("nan"), False),
    (0.0034, True),
])
def test_scale_range_boundaries(value, ok):
    s = _bf16([0.0034, value, 0.002])
    assert decode._scales_in_v2_range(s) is ok


def test_scale_range_spans_chunks_and_dtype(monkeypatch):
    monkeypatch.setattr(decode, "_V2_SCALE_CHUNK", 5)
    s = torch.full((4, 7), 0.003, dtype=torch.bfloat16)
    assert decode._scales_in_v2_range(s)
    s[3, 6] = 2.0 ** 12  # last element, last chunk
    assert not decode._scales_in_v2_range(s)
    assert not decode._scales_in_v2_range(torch.full((4,), 0.003, dtype=torch.float16))


class _FakeOps:
    def __init__(self):
        self.calls = []

    def gemm(self, name):
        def call(*args):
            self.calls.append((name, args[11], args[13]))  # w13, cfg
            return 4
        return call

    @property
    def decode_gemm_orig(self):
        return self.gemm("orig")

    def decode_act_orig(self, *args):
        self.calls.append(("act",))


def _dispatch_env(monkeypatch, M, n=N):
    """maybe_apply with the shape gate, scratch, alignment and ops replaced."""
    ops = _FakeOps()
    v2 = ops.gemm("v2")
    monkeypatch.setattr(decode, "gate_reason", lambda *a, **k: None)
    ws = {"part": torch.empty(4 * M * TOPK * 2 * n), "h": torch.empty(M * TOPK * n),
          "c3": torch.empty(M * TOPK * K), "ctr": torch.empty(2)}
    monkeypatch.setattr(decode, "_workspaces", lambda *a, **k: ws)
    monkeypatch.setattr(decode, "_align", lambda ids, ws: (None, None, None))
    monkeypatch.setattr(decode, "_late_scratch_open", True)
    monkeypatch.setattr(extension, "require_extension", lambda: ops)
    monkeypatch.setattr(extension, "decode_v2_op", lambda: (v2, None))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    layer = SimpleNamespace(
        w1_scale=torch.full((2, 3), 0.003, dtype=torch.bfloat16),
        w2_scale=torch.full((2, 3), 0.004, dtype=torch.bfloat16),
        activation_config=SimpleNamespace(clamp_limit=10.0),
        moe_sum=lambda *a: None)
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    w1 = torch.empty(E, K // 16, 2 * n * 2, dtype=torch.int32)
    w2 = torch.empty(E, n // 16, K * 2, dtype=torch.int32)
    ti = torch.zeros(M, TOPK, dtype=torch.int32)
    tw = torch.zeros(M, TOPK)

    def apply():
        return decode.maybe_apply(layer, torch.empty_like(x), x, w1, w2, tw, ti,
                                  None, E, None, False)
    return ops, layer, apply


@pytest.mark.parametrize("M", list(range(1, 33)))
def test_switch_off_runs_orig_with_its_configs(monkeypatch, M):
    monkeypatch.setenv(DECODE, "1")
    ops, _, apply = _dispatch_env(monkeypatch, M)
    monkeypatch.setattr(extension, "decode_v2_op",
                        lambda: pytest.fail("switch off must not load the v2 op"))
    assert apply()
    c13, c2 = decode._config(M)
    assert ops.calls == [("orig", True, c13), ("act",), ("orig", False, c2)]


@pytest.mark.parametrize("M", list(range(1, 33)))
def test_switch_on_runs_v2_with_its_configs(monkeypatch, M):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    ops, _, apply = _dispatch_env(monkeypatch, M)
    taken, lines = _logged(apply)
    assert taken
    c13, c2 = V2_CONFIG[M]
    assert ops.calls == [("v2", True, c13), ("act",), ("v2", False, c2)]
    assert any("Compiled Marlin MoE decode v2 active (VLLM_AMPERE_MARLIN_DECODE_V2)"
               in line for line in lines), lines


def _closed(lines, text):
    return any("VLLM_AMPERE_MARLIN_DECODE_V2 is set but the gate is closed" in line
               and text in line for line in lines)


def test_missing_op_closes_the_gate(monkeypatch):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    ops, _, apply = _dispatch_env(monkeypatch, 4)
    monkeypatch.setattr(extension, "decode_v2_op",
                        lambda: (None, "vllm_sm80._ampere_marlin_C decode_gemm_v2 unavailable: x"))
    taken, lines = _logged(apply)
    assert taken and [c[0] for c in ops.calls] == ["orig", "act", "orig"]
    assert _closed(lines, "decode_gemm_v2 unavailable"), lines


def test_whole_expert_width_closes_the_gate(monkeypatch):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    ops, _, apply = _dispatch_env(monkeypatch, 4, n=2048)
    taken, lines = _logged(apply)
    assert taken and [c[0] for c in ops.calls] == ["orig", "act", "orig"]
    assert _closed(lines, "expert width N=2048 per rank"), lines


def test_out_of_range_scale_closes_the_gate_and_is_cached(monkeypatch):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    ops, layer, apply = _dispatch_env(monkeypatch, 4)
    layer.w2_scale[1, 2] = 2048.0
    checks = []
    real = decode._scales_in_v2_range
    monkeypatch.setattr(decode, "_scales_in_v2_range",
                        lambda s: checks.append(s) or real(s))
    taken, lines = _logged(apply)
    assert taken and [c[0] for c in ops.calls] == ["orig", "act", "orig"]
    assert _closed(lines, "outside [2^-110, 2^10)"), lines
    n = len(checks)
    apply()
    assert len(checks) == n  # cached per layer
    layer.w2_scale = torch.full((2, 3), 0.004, dtype=torch.bfloat16)  # new tensor: rechecked
    ops.calls.clear()
    apply()
    assert len(checks) > n
    assert [c[0] for c in ops.calls] == ["v2", "act", "v2"]


def test_unchecked_scales_under_capture_fall_back_without_caching(monkeypatch):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    ops, layer, apply = _dispatch_env(monkeypatch, 4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(decode, "_scales_in_v2_range",
                        lambda s: pytest.fail("no host sync under capture"))
    taken, lines = _logged(apply)
    assert taken and [c[0] for c in ops.calls] == ["orig", "act", "orig"]
    assert _closed(lines, "not checked before CUDA capture"), lines
    assert getattr(layer, "_ampere_marlin_decode_v2_scales", None) is None


def test_checked_scales_take_v2_under_capture(monkeypatch):
    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    ops, _, apply = _dispatch_env(monkeypatch, 4)
    apply()  # eager: checks and caches
    ops.calls.clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(decode, "_scales_in_v2_range",
                        lambda s: pytest.fail("no host sync under capture"))
    apply()
    assert [c[0] for c in ops.calls] == ["v2", "act", "v2"]


def _worker(width):
    cfg = SimpleNamespace(n_routed_experts=288, num_experts_per_tok=8,
                          moe_intermediate_size=2048)
    return SimpleNamespace(
        device="cuda:0",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=cfg, hf_config=cfg),
            parallel_config=SimpleNamespace(tensor_parallel_size=2048 // width)))


@pytest.mark.parametrize("compiled,v2,logged", [
    ("0", "0", False), ("0", "1", True), ("1", "0", False), ("1", "1", False),
])
def test_switch_without_compiled_decode_logs_closed_gate(monkeypatch, compiled, v2,
                                                         logged):
    """The plugin's register() logs once per process why
    VLLM_AMPERE_MARLIN_DECODE_V2 alone has no effect, and nothing otherwise."""
    from vllm_sm80 import plugin

    monkeypatch.setenv(DECODE, compiled)
    monkeypatch.setenv(V2, v2)
    for name in ("register_models", "register_drafter_models", "_log_registered",
                 "register_triton_mla_sparse_backend", "register_moe_layers"):
        monkeypatch.setattr(plugin, name, lambda: None)
    monkeypatch.setattr(plugin, "_registered", False)
    _, lines = _logged(lambda: (plugin.register(), plugin.register()),
                       plugin.__name__)
    assert lines == ([
        "VLLM_AMPERE_MARLIN_DECODE_V2 is set but the gate is closed "
        "(VLLM_AMPERE_MARLIN_DECODE_CUDA=0); using released Marlin kernels.",
    ] if logged else []), lines


@pytest.mark.parametrize("v2", ["0", "1"])
def test_off_path_startup_is_silent(monkeypatch, v2):
    monkeypatch.setenv(V2, v2)
    _, lines = _logged(lambda: decode.warmup_from_worker(_worker(512)))
    assert not [line for line in lines if V2 in line]


class _Stop(Exception):
    pass


@pytest.mark.parametrize("compiled,v2,called", [
    ("0", "0", False), ("0", "1", False), ("1", "0", True), ("1", "1", True),
])
def test_kernel_warmup_reaches_the_decode_banner(monkeypatch, compiled, v2, called):
    """kernel_warmup calls warmup_from_worker only under
    VLLM_AMPERE_MARLIN_DECODE_CUDA; VLLM_AMPERE_MARLIN_DECODE_V2 does not matter."""
    from vllm.model_executor.warmup import kernel_warmup as kw

    monkeypatch.setenv(DECODE, compiled)
    monkeypatch.setenv(V2, v2)
    for name in ("qwen_triton_warmup", "qwen_vl_triton_warmup", "mamba_triton_warmup"):
        monkeypatch.setattr(kw, name, lambda *a, **k: None)
    monkeypatch.setattr(decode, "open_late_scratch", lambda: None)

    def reached(_worker):
        raise _Stop
    monkeypatch.setattr(decode, "warmup_from_worker", reached)
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_jit_warmup=False),
            model_config=None,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[])))
    try:
        kw.kernel_warmup(worker)
        stopped = False
    except _Stop:
        stopped = True
    except Exception:  # later warmups need a real worker; only the branch matters
        stopped = False
    assert stopped is called


def _fake_registry(monkeypatch, schemas):
    def find(qualified, _overload):
        name = qualified.split("::", 1)[1]
        if name not in schemas:
            raise RuntimeError(f"no operator {qualified}")
        args, types, returns = schemas[name]
        schema = SimpleNamespace(
            arguments=[SimpleNamespace(name=a, type=t)
                       for a, t in zip(args.split(), types.split())],
            returns=[SimpleNamespace(type=t) for t in returns.split()])
        return SimpleNamespace(schema=lambda: schema)
    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", find)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key",
                        lambda _qualified, key: key == "CUDA")


def _fake_library(monkeypatch):
    info = extension._runtime_build_info()
    monkeypatch.setattr(importlib, "import_module",
                        lambda _name: SimpleNamespace(build_info=lambda: info))


def test_decode_v2_op_present(monkeypatch):
    _fake_library(monkeypatch)
    schemas = dict(extension._SCHEMA_ARGUMENTS)
    schemas["decode_gemm_v2"] = extension._SCHEMA_ARGUMENTS["decode_gemm_orig"]
    _fake_registry(monkeypatch, schemas)
    sentinel = object()
    monkeypatch.setattr(torch.ops, "_ampere_marlin_C",
                        SimpleNamespace(decode_gemm_v2=sentinel), raising=False)
    assert extension.decode_v2_op() == (sentinel, None)
    assert extension.decode_v2_op()[0] is sentinel  # cached


@pytest.mark.parametrize("case", ["missing", "schema", "import"])
def test_decode_v2_op_absent_never_raises(monkeypatch, case):
    schemas = dict(extension._SCHEMA_ARGUMENTS)
    if case == "schema":
        args, types, returns = schemas["decode_gemm_orig"]
        schemas["decode_gemm_v2"] = (args, types, "")
    if case == "import":
        def missing(_name):
            raise ModuleNotFoundError("optional module missing")
        monkeypatch.setattr(importlib, "import_module", missing)
    else:
        _fake_library(monkeypatch)
    _fake_registry(monkeypatch, schemas)
    op, why = extension.decode_v2_op()
    assert op is None and "decode_gemm_v2 unavailable" in why


def test_warmup_launches_every_v2_variant(monkeypatch):
    seen = set()

    def op(*args):
        assert int(args[5][0]) == 0  # ntpp = 0: no work items
        seen.add((args[11], args[13] & 15))
        return 4
    monkeypatch.setattr(extension, "decode_v2_op", lambda: (op, None))
    monkeypatch.setattr(decode, "_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda _device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ws = decode._workspaces(torch.device("cpu"), N, 32, create=True)
    decode._warm_v2(torch.device("cpu"), ws)
    want = {(w13, c[i] & 15) for c in V2_CONFIG.values()
            for i, w13 in ((0, True), (1, False))}
    assert seen == want


def test_v2_source_registers_one_op_for_the_n512_shard():
    src = (LIB / "decode_v2.cu").read_text()
    assert "namespace ampere_marlin_decode_v2" in src
    assert re.findall(r'm\.def\(\s*"(\w+)\(', src) == ["decode_gemm_v2"]
    assert re.findall(r'm\.impl\("(\w+)"', src) == ["decode_gemm_v2"]
    shapes = set(re.findall(r"AMPERE_MARLIN_CFGS\((\d+), (true|false), (\d+), (\d+)\)", src))
    assert shapes == {("64", "true", "4096", "1024"), ("32", "false", "512", "4096")}
    cases = re.findall(r"AMPERE_MARLIN_CASE\(RR, WW, (\d+), \d+, \d+, (?:true|false), KK, NN\)",
                       src)
    built = {int(c) for c in cases}
    used = {c[i] & 15 for c in V2_CONFIG.values() for i in (0, 1)}
    assert used <= built
    # decode_orig.cu keeps its own namespace and operators
    orig = (LIB / "decode_orig.cu").read_text()
    assert "namespace ampere_marlin_decode_orig" in orig and "decode_gemm_v2" not in orig


# ------------------------------------------------------------------ GPU ----
def _gpu_ok():
    if not (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (8, 0)
            and importlib.util.find_spec("vllm_sm80._ampere_marlin_C") is not None):
        return False
    extension._DECODE_V2 = None
    ok = extension.decode_v2_op()[0] is not None
    extension._DECODE_V2 = None
    return ok


gpu = pytest.mark.skipif(not _gpu_ok(),
                         reason="needs sm_80 and the optional library with decode_gemm_v2")


@pytest.fixture(scope="module")
def weights():
    """Packed int4 weights and production-magnitude scales, generated on the CPU."""
    g = torch.Generator().manual_seed(0)
    info = torch.iinfo(torch.int32)

    def packed(*shape):
        return torch.randint(info.min, info.max, shape, generator=g, dtype=torch.int32)

    def scales(*shape):
        return ((torch.rand(*shape, generator=g) * 0.2 + 0.9) * 0.0034).to(torch.bfloat16)

    dev = torch.device("cuda:0")
    wd = {"w1": packed(E, K // 16, 2 * N * 2), "w2": packed(E, N // 16, K * 2),
          "w1_scale": scales(E, K // 128, 2 * N), "w2_scale": scales(E, N // 128, K)}
    wd = {k: v.to(dev) for k, v in wd.items()}
    yield wd
    del wd
    torch.cuda.empty_cache()


def _inputs(M, real, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16)
    ti = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, TOPK, generator=g)
    tw = tw / tw.sum(dim=1, keepdim=True)
    ti[real:] = -1
    dev = torch.device("cuda:0")
    return x.to(dev), tw.to(dev), ti.to(dev)


def _poison():
    """Fill the current stream's decode scratch so a path cannot inherit another's results."""
    ws = decode._workspaces(torch.device("cuda:0"), N, 32, create=False)
    if ws is not None:
        for key in ("part", "h", "c3"):
            ws[key].fill_(-7.0)


def _apply(monkeypatch, layer, wd, x, tw, ti, v2, out=None):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    monkeypatch.setenv(V2, str(int(v2)))
    if out is None:
        out = torch.empty_like(x)
    if not torch.cuda.is_current_stream_capturing():
        _poison()
    assert decode.maybe_apply(layer, out, x, wd["w1"], wd["w2"], tw, ti,
                              MoEActivation.SILU, E, None, False)
    return out


@pytest.fixture
def gpu_env(monkeypatch):
    from vllm_sm80.kernels.moe.align import deterministic_moe_align_mode

    monkeypatch.setenv(DECODE, "1")
    monkeypatch.setenv(V2, "1")
    monkeypatch.setenv("VLLM_MOE_DETERMINISTIC_ALIGN", "1")
    deterministic_moe_align_mode.cache_clear()
    monkeypatch.setattr(decode, "_late_scratch_open", True)
    decode.warmup(torch.device("cuda:0"), N, 32)
    yield
    deterministic_moe_align_mode.cache_clear()


CASES = [(m, m) for m in range(1, 33)] + [(16, 12), (24, 20), (32, 28), (8, 4)]


def _same(a, b):
    """Bit-pattern equality of bf16 tensors (+0 and -0 differ)."""
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def _layer_for(wd):
    layer = pp_helpers._layer(device="cuda", n=N, w1_scale=wd["w1_scale"],
                              w2_scale=wd["w2_scale"])
    layer._lora_context = None
    return layer


def _bitwise_case(monkeypatch, wd, M, real, seed):
    layer = _layer_for(wd)
    x, tw, ti = _inputs(M, real, seed)
    orig = _apply(monkeypatch, layer, wd, x, tw, ti, False).clone()
    new = _apply(monkeypatch, layer, wd, x, tw, ti, True).clone()
    again = _apply(monkeypatch, layer, wd, x, tw, ti, True)
    torch.cuda.synchronize()
    assert layer._ampere_marlin_decode_v2_scales[2] is True
    assert torch.isfinite(new[:real]).all()
    assert _same(new[:real], orig[:real])
    assert _same(again[:real], new[:real])


@gpu
@pytest.mark.parametrize("M,real", CASES)
def test_v2_is_bitwise_equal_to_orig(weights, monkeypatch, gpu_env, M, real):
    _bitwise_case(monkeypatch, weights, M, real, 100 + M)


@gpu
@pytest.mark.parametrize("M", [1, 4, 32])
def test_v2_is_bitwise_equal_across_the_scale_range(weights, monkeypatch, gpu_env, M):
    """Group scales drawn from 0, -0, +-2^-60, +-0.0034 and +-1016 (inside the
    checked range, products far above 2^-110)."""
    g = torch.Generator().manual_seed(M)
    vals = torch.tensor([0.0, -0.0, 2.0 ** -60, -(2.0 ** -60), 0.0034, -0.0034,
                         1016.0, -1016.0])

    def mixed(s):
        pick = torch.randint(0, len(vals), s.shape, generator=g)
        return vals[pick].to(torch.bfloat16).to(s.device)
    wd = dict(weights, w1_scale=mixed(weights["w1_scale"]), w2_scale=mixed(weights["w2_scale"]))
    _bitwise_case(monkeypatch, wd, M, M, 500 + M)


@gpu
def test_out_of_range_scales_take_orig_on_device(weights, monkeypatch, gpu_env):
    w2s = weights["w2_scale"].clone()
    w2s[0, 0, 0] = 4096.0
    wd = dict(weights, w2_scale=w2s)
    layer = _layer_for(wd)
    x, tw, ti = _inputs(4, 4, 7)
    _apply(monkeypatch, layer, wd, x, tw, ti, True)
    assert layer._ampere_marlin_decode_v2_scales[2] is False


def _capture(monkeypatch, layer, wd, x, tw, ti, v2, out, stream):
    """Warm, eager and captured call of one path on ``stream``; returns
    (graph, eager output, memory allocated before the eager call)."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _apply(monkeypatch, layer, wd, x, tw, ti, v2, out)  # stream scratch, scale check
        stream.synchronize()
        before = torch.cuda.memory_allocated()
        _apply(monkeypatch, layer, wd, x, tw, ti, v2, out)
        eager = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _apply(monkeypatch, layer, wd, x, tw, ti, v2, out)
    stream.synchronize()
    return graph, eager, before


def _replay(graph, out, stream):
    with torch.cuda.stream(stream):
        _poison()
        out.fill_(-3.0)
    stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()


@gpu
@pytest.mark.parametrize("M,real", [(4, 4), (8, 8), (16, 12), (32, 32), (32, 28)])
def test_v2_graph_replay_equals_eager(weights, monkeypatch, gpu_env, M, real):
    layer = _layer_for(weights)
    x, tw, ti = _inputs(M, real, 300 + M)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    paths = {}
    for v2 in (False, True):
        graph, eager, before = _capture(monkeypatch, layer, weights, x, tw, ti, v2, out, stream)
        for _ in range(3):
            _replay(graph, out, stream)
            assert _same(out[:real], eager[:real]), v2
        paths[v2] = (graph, eager, torch.cuda.memory_allocated() - before)
    assert _same(paths[True][1][:real], paths[False][1][:real])
    # warm, eager, capture and replays allocate exactly what the original path does
    assert paths[True][2] == paths[False][2]
    # new contents in the captured buffers: both replays follow them
    x2, tw2, ti2 = _inputs(M, real, 400 + M)
    want = _apply(monkeypatch, layer, weights, x2, tw2, ti2, False).clone()
    x.copy_(x2)
    tw.copy_(tw2)
    ti.copy_(ti2)
    for v2 in (True, False):
        _replay(paths[v2][0], out, stream)
        assert _same(out[:real], want[:real]), v2
