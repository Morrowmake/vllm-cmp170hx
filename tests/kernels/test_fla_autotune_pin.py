# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GLM5_FLA_PIN_AUTOTUNE: one pinned config per autotune key.

CPU only. The flag is read when the kernel modules are imported, so every
case that depends on it runs in a fresh interpreter.
"""

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
KERNEL_DIRS = (
    REPO / "vllm/third_party/flash_linear_attention/ops",
    REPO / "vllm/models/glm5next/nvidia/ops/third_party/kda",
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
import vllm.third_party.flash_linear_attention.ops.pinned_autotune as pa
from vllm.third_party.flash_linear_attention.ops import autotune_pins
"""

_SMEM_TRUE = (
    "import vllm.third_party.flash_linear_attention.ops.utils as _u\n"
    "_u.check_shared_mem = lambda *a, **k: True\n"
)


def _run(body: str, flag: str | None, smem: bool = False, extra_env=None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("VLLM_GLM5_FLA_PIN_AUTOTUNE", None)
    if flag is not None:
        env["VLLM_GLM5_FLA_PIN_AUTOTUNE"] = flag
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
    monkeypatch.delenv("VLLM_GLM5_FLA_PIN_AUTOTUNE", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_GLM5_FLA_PIN_AUTOTUNE is False


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
        " 'pins': sorted({n for n, _ in autotune_pins.PINS})}))",
        None,
    )
    assert len(got["registry"]) == counts["pinned"] == 23
    assert got["registry"] == got["defaults"]
    assert set(got["pins"]) <= set(got["registry"])


_VALIDATE = """
bad = []
for name, spec in autotune_pins.DEFAULTS.items():
    if pa.find_config(pa.REGISTRY[name].configs, spec) is None:
        bad.append(("default", name, spec))
for (name, key), spec in autotune_pins.PINS.items():
    t = pa.REGISTRY.get(name)
    if t is None or pa.find_config(t.configs, spec) is None:
        bad.append(("pin", name, repr(key), spec))
    if not isinstance(key, tuple):
        bad.append(("key", name, repr(key)))
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
              scale=1.0, T=100, H=16, K=128, V=128)
listed_key = pa.tuning_key(t, (), dict(common, BT=64))
spec = {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 8, "num_stages": 2}
autotune_pins.PINS[(name, listed_key)] = spec
t.run(**common, BT=64)
t.run(**common, BT=64)
t.run(**common, BT=32)
print(json.dumps({
    "calls": t.fn.calls,
    "cache": {repr(k): pa.config_spec(c) for k, c in t.cache.items()},
    "listed_key": repr(listed_key),
    "default": autotune_pins.DEFAULTS[name],
    "spec": spec,
}))
"""


def test_pinned_run_uses_one_config_without_benchmark():
    got = _run(_PRUNE, "1")
    spec, default = got["spec"], got["default"]

    def flat(s):
        return {**s["kwargs"], "num_warps": s["num_warps"], "num_stages": s["num_stages"]}

    assert got["calls"] == [flat(spec), flat(spec), flat(default)]
    assert got["cache"][got["listed_key"]] == spec
    assert len(got["cache"]) == 2
    assert [v for k, v in got["cache"].items() if k != got["listed_key"]] == [default]


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
print(json.dumps([repr(list(t.cache)[0]), repr(pa.tuning_key(t, args, kwargs))]))
"""


def test_tuning_key_matches_triton():
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
