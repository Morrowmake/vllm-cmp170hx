# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_FLA_PIN_AUTOTUNE: one pinned config per autotune key.
VLLM_FLA_AUTOTUNE_SKIP_SPILLING: no over-limit build is launched on sm_80.

CPU only. The flag is read when the kernel modules are imported, so every
case that depends on it runs in a fresh interpreter.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

triton = pytest.importorskip("triton")

REPO = Path(__file__).resolve().parents[2]
# The vendored FLA and KDA kernels are engine files (the vllm package's tree).
ENGINE = Path(importlib.util.find_spec("vllm").origin).parents[1]
KERNEL_DIRS = (
    ENGINE / "vllm/third_party/flash_linear_attention/ops",
    ENGINE / "vllm/models/glm5next/nvidia/ops/third_party/kda",
)
KERNEL_MODULES = (
    "vllm.third_party.flash_linear_attention.ops.chunk_delta_h",
    "vllm.third_party.flash_linear_attention.ops.chunk_o",
    "vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt",
    "vllm.third_party.flash_linear_attention.ops.cumsum",
    "vllm.third_party.flash_linear_attention.ops.kda",
    "vllm.third_party.flash_linear_attention.ops.l2norm",
    "vllm.third_party.flash_linear_attention.ops.solve_tril",
    "vllm.third_party.flash_linear_attention.ops.wy_fast",
    "vllm.models.glm5next.nvidia.ops.third_party.kda.kernels",
)

_PRELUDE = """
import importlib, json, os, sys
{patch}
for m in {modules!r}:
    importlib.import_module(m)
from triton.runtime.autotuner import Autotuner
import vllm_sm80.fla.pinned_autotune as pa
from vllm_sm80.fla import autotune_pins
"""

# check_shared_mem() is true on a CMP 170HX and false on a CPU-only host. The
# ops package imports its kernels (and so builds their config lists) as soon
# as anything under it is imported, so load utils on its own first, patch it,
# and let the package pick up the patched module.
_SMEM_TRUE = """
import importlib.util
from pathlib import Path
import vllm.third_party.flash_linear_attention as _fla
_name = "vllm.third_party.flash_linear_attention.ops.utils"
_spec = importlib.util.spec_from_file_location(
    _name, Path(_fla.__file__).parent / "ops" / "utils.py"
)
_u = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _u
_spec.loader.exec_module(_u)
_u.check_shared_mem = lambda *a, **k: True
"""


def _run(body: str, flag: str | None, smem: bool = False, extra_env=None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("VLLM_FLA_PIN_AUTOTUNE", None)
    env.pop("VLLM_FLA_AUTOTUNE_SKIP_SPILLING", None)
    if flag is not None:
        env["VLLM_FLA_PIN_AUTOTUNE"] = flag
    env.update(extra_env or {})
    code = (
        _PRELUDE.format(patch=_SMEM_TRUE if smem else "", modules=KERNEL_MODULES)
        + body
    )
    # A file, not ``-c``: @triton.jit needs the kernel source.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        proc = subprocess.run(
            [sys.executable, f.name],
            env=env,
            cwd="/",
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        os.unlink(f.name)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _source_decorators():
    counts = {"pinned": 0, "plain": 0}
    for d in KERNEL_DIRS:
        for f in sorted(d.glob("*.py")):
            if f.name == "pinned_autotune.py":
                continue
            s = f.read_text()
            counts["pinned"] += len(re.findall(r"^@pinned_autotune\(", s, re.M))
            counts["plain"] += len(re.findall(r"triton\.autotune\(", s))
    return counts


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_FLA_PIN_AUTOTUNE", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_FLA_PIN_AUTOTUNE is False


_DESCRIBE = """
out = {}
for name, t in sorted(pa.REGISTRY.items()):
    out[name] = {
        "cls": type(t).__module__ + "." + type(t).__qualname__,
        "run_is_triton": type(t).run is Autotuner.run,
        "configs": [str(c) for c in t.configs],
        "keys": list(t.keys),
        "early_prune": t.early_config_prune is not None,
    }
print(json.dumps(out))
"""


@pytest.mark.parametrize("flag", [None, "0"])
def test_flag_off_keeps_triton_autotuner(flag):
    off = _run(_DESCRIBE, flag)
    on = _run(_DESCRIBE, "1")
    assert off.keys() == on.keys()
    for name, d in off.items():
        assert d["cls"] == "triton.runtime.autotuner.Autotuner", name
        assert d["run_is_triton"], name
        assert not d["early_prune"], name
        # The flag never changes the config list or the key.
        assert d["configs"] == on[name]["configs"], name
        assert d["keys"] == on[name]["keys"], name
        assert len(d["configs"]) > 1, name
    for name, d in on.items():
        assert d["cls"].endswith("pinned_autotune.PinnedAutotuner"), name


def test_every_autotune_is_pinnable_and_in_table():
    counts = _source_decorators()
    # A new @triton.autotune in these directories must use pinned_autotune.
    assert counts["plain"] == 0, counts
    got = _run(
        "print(json.dumps({'registry': sorted(pa.REGISTRY),"
        " 'defaults': sorted(autotune_pins.DEFAULTS),"
        " 'pins': sorted({n for n, _, _ in autotune_pins.PINS})}))",
        None,
    )
    assert len(got["registry"]) == counts["pinned"] == 23
    assert got["registry"] == got["defaults"]
    assert set(got["pins"]) <= set(got["registry"])


_VALIDATE = """
bad = []
b = autotune_pins.T_BUCKETS
if not (isinstance(b, tuple) and all(isinstance(x, int) and x > 0 for x in b)
        and list(b) == sorted(set(b))):
    bad.append(("T_BUCKETS", repr(b)))
for name, spec in autotune_pins.DEFAULTS.items():
    if pa.find_config(pa.REGISTRY[name].configs, spec) is None:
        bad.append(("default", name, spec))
for (name, key, H), per in autotune_pins.PINS.items():
    t = pa.REGISTRY.get(name)
    if not isinstance(key, tuple) or not (H is None or isinstance(H, int)):
        bad.append(("key", name, repr(key), repr(H)))
    if not per:
        bad.append(("empty", name, repr(key)))
    for bucket, spec in per.items():
        if bucket is not None and bucket not in b:
            bad.append(("bucket", name, repr(key), bucket))
        if t is None or pa.find_config(t.configs, spec) is None:
            bad.append(("pin", name, repr(key), bucket, spec))
print(json.dumps(bad))
"""


@pytest.mark.parametrize("smem", [True, False])
def test_table_entries_are_valid_configs(smem):
    # smem=True is the sm_80 config list (check_shared_mem() is true there).
    assert _run(_VALIDATE, None, smem=smem) == []


_PRUNE = """
import torch
from triton.runtime.autotuner import Autotuner

pa._gate = True
name = "glm5_kda.chunk_gla_fwd_kernel_o"
t = pa.REGISTRY[name]
assert isinstance(t, pa.PinnedAutotuner)

def no_bench(*a, **k):
    raise AssertionError("benchmarked")
t._bench = no_bench

class Launch:
    def __init__(self):
        self.calls = []
    def run(self, *args, **kwargs):
        self.calls.append({k: v for k, v in kwargs.items()
                           if k in ("BK", "BV", "num_warps", "num_stages")})
t.fn = Launch()

x = torch.empty(1, dtype=torch.bfloat16)
f = torch.empty(1, dtype=torch.float32)
common = dict(q=x, v=x, g=f, h=x, o=x, A=f, cu_seqlens=None, chunk_indices=None,
              scale=1.0, K=128, V=128, BT=64)
key = pa.pin_lookup_key(t, (), dict(common, T=40, H=16), ())[0]
autotune_pins.T_BUCKETS = (64, 1152)
spec = {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 8, "num_stages": 2}
spec_open = {"kwargs": {"BK": 64, "BV": 128}, "num_warps": 2, "num_stages": 4}
autotune_pins.PINS.clear()
autotune_pins.PINS[(name, key, 16)] = {64: spec, None: spec_open}
t.run(**common, T=40, H=16)    # bucket 64: table
t.run(**common, T=64, H=16)    # bucket 64: table (cached)
t.run(**common, T=1000, H=16)  # bucket 1152: not in the table -> default
t.run(**common, T=5000, H=16)  # open bucket: table
t.run(**common, T=40, H=64)    # other H: default
try:
    t.run(**common, H=16)      # no T: must not guess
    missing_t = "ran"
except AssertionError:
    missing_t = "asserted"
print(json.dumps({
    "calls": t.fn.calls,
    "n_cache": len(t.cache),
    "default": autotune_pins.DEFAULTS[name],
    "spec": spec, "spec_open": spec_open, "missing_t": missing_t,
    "buckets": [pa.t_bucket(v, (64, 1152)) for v in (1, 64, 65, 1152, 1153)],
}))
"""


def test_pinned_run_uses_one_config_per_bucket_without_benchmark():
    got = _run(_PRUNE, "1")

    def flat(s):
        return {**s["kwargs"], "num_warps": s["num_warps"], "num_stages": s["num_stages"]}

    spec, spec_open, default = got["spec"], got["spec_open"], got["default"]
    assert got["calls"] == [flat(spec), flat(spec), flat(default), flat(spec_open),
                            flat(default)]
    assert got["n_cache"] == 4
    assert got["missing_t"] == "asserted"
    assert got["buckets"] == [64, 64, 1152, 1152, None]


_T_ARGS = """
print(json.dumps(sorted(n for n, t in pa.REGISTRY.items() if "T" not in t.arg_names)))
"""


def test_only_l2norm_kernel1_has_no_T_argument():
    # Every other kernel's pin depends on its T argument; this one has none
    # and always uses the open bucket.
    assert _run(_T_ARGS, None) == ["fla.l2norm.l2norm_fwd_kernel1"]


_GATE_CLOSED = """
pa._gate = False
name = "glm5_kda.chunk_gla_fwd_kernel_o"
t = pa.REGISTRY[name]
called = []
def base_run(self, *a, **k):
    called.append(1)
Autotuner.run = base_run
t.run(T=1)
print(json.dumps(called))
"""


def test_gate_closed_falls_back_to_autotuning():
    assert _run(_GATE_CLOSED, "1") == [1]


_KEY = """
import torch
import triton
import triton.language as tl
from triton.runtime.autotuner import Autotuner

@triton.jit
def k(a, b, cu, T, H: tl.constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr, BK: tl.constexpr):
    pass

t = triton.autotune(
    configs=[triton.Config({"BK": 32}), triton.Config({"BK": 64})],
    key=["H", "BT", "IS_VARLEN"],
)(k)

class Launch:
    def run(self, *a, **kw):
        return None
t.fn = Launch()
t._bench = lambda *a, config, **kw: [float(config.kwargs["BK"])]
a = torch.empty(1, dtype=torch.bfloat16)
b = torch.empty(1, dtype=torch.float32)
cu = torch.empty(1, dtype=torch.int32)
args = (a,)
kwargs = dict(b=b, cu=cu, T=7, H=16, BT=64, IS_VARLEN=True)
t.run(*args, **kwargs)
ours = pa.pin_lookup_key(t, args, kwargs, ())[0]
print(json.dumps([repr(list(t.cache)[0]), repr(ours)]))
"""


def test_pin_lookup_key_matches_triton():
    # No on-disk autotune cache: the fake launcher has no source to hash.
    triton_key, ours = _run(_KEY, None, extra_env={"TRITON_CACHE_AUTOTUNING": "0"})
    assert triton_key == ours


_INTERP = """
import torch
import vllm.third_party.flash_linear_attention.ops.index as idx
idx.async_tensor_h2d = lambda t, device=None, dtype=None, **kw: t.to(device=device, dtype=dtype)
from vllm.models.glm5next.nvidia.ops.third_party.kda import kernels as kk

pa._gate = True
t = pa.REGISTRY["glm5_kda.kda_gate_cumsum_fwd_kernel"]
t._bench = None  # must not be used
torch.manual_seed(0)
H, D, lens = 2, 128, [70, 17]
T = sum(lens)
raw_g = torch.randn(1, T, H, D, dtype=torch.bfloat16)
A_log = torch.randn(H, dtype=torch.float32)
g_bias = torch.randn(H * D, dtype=torch.float32)
cu = torch.tensor([0, lens[0], T], dtype=torch.int32)
outs = []
for _ in range(2):
    y = kk.fused_kda_gate_chunk_cumsum(raw_g, A_log=A_log, g_bias=g_bias,
        cu_seqlens=cu, safe_gate=True, lower_bound=-5.0)
    outs.append(y)
print(json.dumps({
    "finite": bool(torch.isfinite(outs[0]).all()),
    "equal": bool(torch.equal(outs[0], outs[1])),
    "cache": [[repr(k), pa.config_spec(c)] for k, c in t.cache.items()],
    "default": autotune_pins.DEFAULTS["glm5_kda.kda_gate_cumsum_fwd_kernel"],
}))
"""


def test_pinned_kernel_runs_in_interpreter():
    got = _run(_INTERP, "1", extra_env={"TRITON_INTERPRET": "1"})
    assert got["finite"] and got["equal"]
    assert len(got["cache"]) == 1
    key, spec = got["cache"][0]
    assert key.startswith("(2, 128, 64, True, ")
    assert spec == got["default"]


_BANNER = """
import types
from vllm.platforms import current_platform
cls = type(current_platform)
cls.get_device_capability = classmethod(lambda c, device_id=0: types.SimpleNamespace(major=8, minor=0))
cls.is_cuda = classmethod(lambda c: True)
pa._gate = None
opened = pa._gate_open()
# a table entry that is not a config of its kernel falls back to the default
name = "glm5_kda.chunk_gla_fwd_kernel_o"
t = pa.REGISTRY[name]
key = (64, "torch.bfloat16")
autotune_pins.PINS[(name, key, 16)] = {None: {"kwargs": {"BK": 7}, "num_warps": 3, "num_stages": 9}}
cfg, src = pa.resolve(name, key, 16, None, t.configs)
cls.get_device_capability = classmethod(lambda c, device_id=0: types.SimpleNamespace(major=9, minor=0))
pa._gate = None
closed = pa._gate_open()
print(json.dumps([opened, src, pa.config_spec(cfg) == autotune_pins.DEFAULTS[name], closed]))
"""


def test_banners_and_bad_pin_fallback_log_cleanly():
    # The log calls dedupe on their arguments, which must be hashable.
    assert _run(_BANNER, "1") == [True, "default", True, False]


# VLLM_FLA_AUTOTUNE_SKIP_SPILLING: on sm_80 a benchmarking autotuner builds
# each config first and never launches one above LOCAL_BYTES_LIMIT bytes of
# local memory per thread.

_SKIP_NAME = "fla.chunk_o.chunk_fwd_kernel_o"
_SKIP_ENV = {"VLLM_FLA_AUTOTUNE_SKIP_SPILLING": "1"}

# chunk_fwd_kernel_o's configs as an sm_80 build sees them: the local memory
# per thread of each, from an sm_80 build on the CPU (cuobjdump STACK), at the
# Qwen3.8 TP2 GDN prefill shape (K = V = 128, BT = 64, 24 value and 8 key
# heads, gate and varlen on).
_SM80_STACK = """
import re, subprocess, tempfile
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import JITFunction
NAME = "fla.chunk_o.chunk_fwd_kernel_o"
fn = pa.REGISTRY[NAME].fn
while not isinstance(fn, JITFunction):
    fn = fn.fn
cuobjdump = os.path.join(
    os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump"
)
ptr = {"q": "*bf16", "k": "*bf16", "v": "*bf16", "h": "*bf16", "g": "*fp32",
       "o": "*bf16", "cu_seqlens": "*i32", "chunk_indices": "*i32"}


def sm80_local_bytes(c):
    const = dict(H=24, Hg=8, K=128, V=128, BT=64, USE_G=True, IS_VARLEN=True,
                 **c.kwargs)
    sig = {a: "constexpr" if a in const else ptr.get(a, "i32") for a in fn.arg_names}
    sig["scale"] = "fp32"
    attrs = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(fn.arg_names)
             if a in ptr}
    k = triton.compile(ASTSource(fn, sig, const, attrs),
                       target=GPUTarget("cuda", 80, 32),
                       options={"num_warps": c.num_warps, "num_stages": c.num_stages})
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        res = subprocess.run([cuobjdump, "--dump-resource-usage", f.name],
                             capture_output=True, text=True, check=True).stdout
    return int(re.search(r"STACK:(\\d+)", res).group(1))
"""

# Drive SkipSpillingAutotuner.run through a real benchmark: the launch, the
# build and do_bench are stubbed, the local sizes come from ``SIZES``.
_SKIP_BENCH = """
t = pa.REGISTRY[NAME]
launched, built = [], []
class Fn:
    arg_names = t.fn.arg_names
    def run(self, *args, warmup=False, **kw):
        cfg = (kw["BK"], kw["BV"], kw["num_warps"], kw["num_stages"])
        if warmup:
            built.append(cfg)
            return cfg
        launched.append(cfg)
t.fn = Fn()
pa.local_bytes = lambda kernel: SIZES[kernel]
DO_BENCH = globals().get("DO_BENCH", [1.0, 1.0, 1.0])
CACHE_HIT = globals().get("CACHE_HIT")
t.do_bench = lambda call, quantiles: (call(), list(DO_BENCH))[1]
t.cache_results = CACHE_HIT is not None
if CACHE_HIT is not None:
    # Stand-in for a hit in Triton's autotune cache (timings of an earlier boot).
    hit = next(c for c in t.configs
               if (c.kwargs["BK"], c.kwargs["BV"], c.num_warps, c.num_stages) == CACHE_HIT)
    def cached(self, key, configs, bench_fn):
        self.cache[key] = hit
        return True
    Autotuner.check_disk_cache = cached
t.cache.clear()
x = torch.empty(1, dtype=torch.bfloat16)
f = torch.empty(1, dtype=torch.float32)
args = dict(q=x, k=x, v=x, h=x, g=f, o=x, cu_seqlens=None, chunk_indices=None,
            scale=1.0, T=8, H=24, Hg=8, K=128, V=128, BT=64)
err = None
try:
    t.run(grid=(1,), warmup=False, **args)
except RuntimeError as e:
    err = str(e)
key = next(iter(t.cache), None)
print(json.dumps({
    "cls": type(t).__qualname__,
    "configs": [list(c) for c in SIZES],
    "launched": sorted(set(launched)),
    "built": sorted(set(built)),
    "timings": {str(pa.config_spec(c)): v[0] for c, v in getattr(t, "configs_timings", {}).items()},
    "best": None if key is None else pa.config_spec(t.cache[key]),
    "error": err,
}))
"""

_SIZES_FROM_BUILD = """
import torch
pa._skip_gate = True
SIZES = {(c.kwargs["BK"], c.kwargs["BV"], c.num_warps, c.num_stages): sm80_local_bytes(c)
         for c in pa.REGISTRY[NAME].configs}
"""

_SIZES_ALL_OVER = """
import torch
pa._skip_gate = True
SIZES = {(c.kwargs["BK"], c.kwargs["BV"], c.num_warps, c.num_stages): 2048
         for c in pa.REGISTRY[NAME].configs}
"""

_SIZES_GATE_CLOSED = """
import torch
pa._skip_gate = False
SIZES = {(c.kwargs["BK"], c.kwargs["BV"], c.num_warps, c.num_stages): 4096
         for c in pa.REGISTRY[NAME].configs}
"""


def test_skip_spilling_flag_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_FLA_AUTOTUNE_SKIP_SPILLING", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_FLA_AUTOTUNE_SKIP_SPILLING is False


def test_skip_spilling_never_launches_an_over_limit_build():
    r = _run(_SM80_STACK + _SIZES_FROM_BUILD + _SKIP_BENCH, None, smem=True,
             extra_env=_SKIP_ENV)
    assert r["cls"] == "SkipSpillingAutotuner"
    configs = [tuple(c) for c in r["configs"]]
    # The real sm_80 list: 2 x 2 tiles of 64/128, 2/4/8 warps, 2/3/4 stages.
    assert {c[:2] for c in configs} == {(64, 64), (64, 128), (128, 64), (128, 128)}
    assert len(configs) == 36
    over = {(128, 128, 2, s) for s in (2, 3, 4)}
    assert {tuple(c) for c in r["built"]} == set(configs)
    assert {tuple(c) for c in r["launched"]} == set(configs) - over
    inf = [k for k, v in r["timings"].items() if v == float("inf")]
    assert len(inf) == 3
    assert r["best"] is not None and not (
        r["best"]["kwargs"] == {"BK": 128, "BV": 128} and r["best"]["num_warps"] == 2
    )
    assert r["error"] is None


def test_skip_spilling_all_configs_over_limit_raises():
    r = _run(_SM80_STACK + _SIZES_ALL_OVER + _SKIP_BENCH, None, smem=True,
             extra_env=_SKIP_ENV)
    assert r["launched"] == []
    assert "no autotune config of fla.chunk_o.chunk_fwd_kernel_o has a finite" in r["error"]


_ALL_INF = """
DO_BENCH = [float("inf")] * 3
"""


def test_skip_spilling_no_finite_time_raises_before_any_over_limit_launch():
    r = _run(_SM80_STACK + _SIZES_FROM_BUILD + _ALL_INF + _SKIP_BENCH, None,
             smem=True, extra_env=_SKIP_ENV)
    assert "has a finite time" in r["error"] and "3 need more than 1024 B" in r["error"]
    assert not {tuple(c) for c in r["launched"]} & {(128, 128, 2, s) for s in (2, 3, 4)}


def test_skip_spilling_cache_hit_over_limit_benchmarks_again():
    r = _run(_SM80_STACK + _SIZES_FROM_BUILD + "CACHE_HIT = (128, 128, 2, 2)\n"
             + _SKIP_BENCH, None, smem=True, extra_env=_SKIP_ENV)
    over = {(128, 128, 2, s) for s in (2, 3, 4)}
    assert (128, 128, 2, 2) in {tuple(c) for c in r["built"]}
    assert not {tuple(c) for c in r["launched"]} & over
    assert len(r["launched"]) == 33
    assert r["best"] is not None and r["error"] is None
    assert (r["best"]["kwargs"]["BK"], r["best"]["kwargs"]["BV"],
            r["best"]["num_warps"]) != (128, 128, 2)


def test_skip_spilling_cache_hit_under_limit_builds_only_the_winner():
    r = _run(_SM80_STACK + _SIZES_FROM_BUILD + "CACHE_HIT = (128, 64, 4, 2)\n"
             + _SKIP_BENCH, None, smem=True, extra_env=_SKIP_ENV)
    assert [tuple(c) for c in r["built"]] == [(128, 64, 4, 2)]
    assert [tuple(c) for c in r["launched"]] == [(128, 64, 4, 2)]
    assert r["error"] is None


def test_skip_spilling_gate_closed_builds_nothing_extra():
    r = _run(_SM80_STACK + _SIZES_GATE_CLOSED + _SKIP_BENCH, None, smem=True,
             extra_env=_SKIP_ENV)
    assert r["built"] == []
    assert len(r["launched"]) == len(r["configs"]) == 36
    assert r["error"] is None


_SKIP_GATE = """
import types
from vllm.platforms import current_platform
cls = type(current_platform)
cls.is_cuda = classmethod(lambda c: True)
out = []
for major, minor in ((8, 0), (8, 6), (9, 0)):
    cls.get_device_capability = classmethod(
        lambda c, device_id=0, m=major, n=minor: types.SimpleNamespace(major=m, minor=n))
    pa._skip_gate = None
    out.append(pa._skip_gate_open())
print(json.dumps(out))
"""


def test_skip_spilling_gate_is_sm80_only_and_logs_cleanly():
    # The banners dedupe on their arguments, which must be hashable.
    assert _run(_SKIP_GATE, None, extra_env=_SKIP_ENV) == [True, False, False]


def test_skip_spilling_covers_every_kernel_and_leaves_pinning_alone():
    skip = _run(_DESCRIBE, None, extra_env=_SKIP_ENV)
    off = _run(_DESCRIBE, None)
    assert skip.keys() == off.keys()
    for name, d in skip.items():
        assert d["cls"].endswith("pinned_autotune.SkipSpillingAutotuner"), name
        # The config list and the key (hence Triton's autotune cache key) are
        # unchanged.
        assert d["configs"] == off[name]["configs"], name
        assert d["keys"] == off[name]["keys"], name
    both = _run(_DESCRIBE, "1", extra_env=_SKIP_ENV)
    for name, d in both.items():
        assert d["cls"].endswith("pinned_autotune.PinnedAutotuner"), name
