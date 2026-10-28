# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 DeepSeek-V4.1 path against DeepSeek's own reference source.

DeepSeek publishes the model's reference implementation with the checkpoint
(``inference/model.py``, ``engram.py``, ``kernel.py`` and the two modules
``model.py`` imports, at revision 1bae0e1088993b9b82de08f59de5be5a34e42170).
These tests import that source unchanged and run its functions and modules
on the official weights, next to the port's functions on the same weights and
inputs. Nothing of the official source is restated here: where a piece needs
one of ``kernel.py``'s TileLang CUDA kernels and this host cannot run it, the
piece is skipped, never replaced.

Inputs (absolute paths in ``LAB_INPUT_<NAME>``, pinned and checked by the
caller; this file only reads them):

- ``OFFICIAL_MODEL_PY``, ``OFFICIAL_ENGRAM_PY``, ``OFFICIAL_KERNEL_PY``,
  ``OFFICIAL_VISION_PY``, ``OFFICIAL_IMAGE_PROCESSOR_PY``: the reference
  source; ``OFFICIAL_CONFIG_JSON``: its ``inference/config.json``
  (``ModelArgs`` field names).
- ``OFFICIAL_INDEX_JSON``: the official ``model.safetensors.index.json``;
  ``OFFICIAL_SHARD_<NNNNN>``: official shard NNNNN (00002 embedding, 00003
  layer 0, 00005 layer 2, 00023 layer 20, 00046 DSpark stage 2, 00047 the
  layer-1 Engram); ``OFFICIAL_TOKENIZER_JSON``.
- ``HF_CONFIG_JSON``: the served checkpoint's ``config.json`` (the port reads
  its parameters from it).
- ``EXL3_INDEX_JSON``: the served EXL3 checkpoint's index; the checkpoint is
  its directory, read as the served loader reads it (``Exl3Manifest``, plus
  the override sidecar the served settings name in ``VLLM_EXL3_OVERRIDE`` /
  ``VLLM_EXL3_OVERRIDE_SHA256``, through ``apply_override``).
- ``PROMPT_CORPUS``: a token corpus (``kld-corpus`` format) cut with the
  served tokenizer: the real prompt.
- ``ACTIVATIONS``: the activation spec (schema
  ``dsv41-reference-activations/1``): which corpus documents form the prompt
  and how the hidden-state streams of layers past the first are made while
  no captured served activations are available (``synthetic: true``).

A skip reason starts with one of these keys:

- ``input``: an input above (or the served override setting) is not given;
- ``kernel.<symbol>``: the piece needs ``kernel.<symbol>`` (TileLang, CUDA)
  and this host has no CUDA device with TileLang, or TileLang cannot run it
  on this card;
- ``cuda``: the port's side of the piece is a CUDA kernel and this host has
  no CUDA device.

Weights: the official FP8 (E4M3 x UE8M0 32x32 blocks) linears and FP4
(E2M1 x UE8M0 per 32) tensors are expanded to BF16 exactly (each product of
an FP8/FP4 value and a power of two is a BF16 value), and the official
modules are built with ``ModelArgs.dtype = "bf16"``, the reference's own BF16
path, where ``linear`` is ``F.linear``. The reference's activation FP8
quantisation before FP8/FP4 GEMMs is therefore not part of these
comparisons (a documented port property). The port's side gets the same
BF16 weights.

FP64: an official module or function is also run with every floating
parameter and buffer in float64 and its ``.float()`` / ``.bfloat16()`` /
``.to(bfloat16 | float32 | float16)`` casts mapped to float64
(:func:`official_fp64`): the reference's formula without its rounding, the
value both sides are measured against.
"""

import contextlib
import copy
import dataclasses
import functools
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc
from vllm_sm80.models.deepseek_v41.sm80_moe import route

BF16 = torch.bfloat16
CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Tolerances, stated before any data is read
# ---------------------------------------------------------------------------
#
# "exact": the port's output equals the official output bit for bit.
# ("fp64", factor): with err(y) = |y - official FP64 value| over the finite
#     FP64 values, max err(port) <= factor * max err(official) and
#     mean err(port) <= factor * mean err(official); an official output with
#     zero error requires a port output with zero error. Factor 2: the two
#     sides may sum in other orders.
# ("selection", ulps): top-k picks may differ only where the scores they are
#     ranked by tie within `ulps` BF16 ulps of the row's k-th score (both
#     sides then hold a valid top-k of the same scores).
# ("fp8_step", n): each element of the port's output is within n steps of
#     the FP8 E4M3 grid (at the larger magnitude of the pair) of the official
#     output: both sides quantise the same value, and a pre-quantisation
#     difference below BF16 rounding moves it by at most one grid step.
# ("rel", r): max |port - official| <= r * max |official| (official FP32
#     TileLang arithmetic with no FP64 counterpart; r = 1e-5 is about 100
#     FP32 ulps, far below any formula change).
# "documented_loss": the EXL3 export's own F16 copies (no override) route
#     some tokens to other experts than the official tensors; recorded, and
#     a failure if none do (the override would then be unneeded and its
#     documentation stale).
TOLERANCES = {
    "rope.table": "exact",
    "rope.rotate": "exact",
    "compressor.latent": ("fp64", 2.0),
    "candidate_blocks": ("selection", 0),
    "gate.weights": ("fp64", 2.0),
    "gate.indices": ("selection", 0),
    "gate.served_tensors": "exact",
    "gate.export_f16": "documented_loss",
    "engram.layout": "exact",
    "engram.hash_ids": "exact",
    "engram.dequant": "exact",
    "engram.gate": ("fp64", 2.0),
    "mhc.collapse": ("fp64", 2.0),
    "mhc.post": ("fp64", 2.0),
    "mhc.mixes": ("rel", 1e-5),
    "mhc.delayed_block": ("rel", 1e-5),
    "qdq": "exact",
    "attention.window_kv": "exact",
    "attention.compressed_kv": "exact",
    "attention.index_k": "exact",
    "attention.topk": ("selection", 2),
    "attention.wo_a_input": ("fp8_step", 1),
    "dspark.markov_bias": ("fp64", 2.0),
    "dspark.confidence": ("fp64", 2.0),
}

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

ENV_PREFIX = "LAB_INPUT_"
SOURCE_FILES = {
    "model": "OFFICIAL_MODEL_PY",
    "engram": "OFFICIAL_ENGRAM_PY",
    "kernel": "OFFICIAL_KERNEL_PY",
    "vision": "OFFICIAL_VISION_PY",
    "image_processor": "OFFICIAL_IMAGE_PROCESSOR_PY",
}
KERNEL_SYMBOLS = (
    "act_quant",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "hc_split_sinkhorn",
    "sparse_attn",
)


def input_path(name: str) -> Path:
    value = os.environ.get(ENV_PREFIX + name)
    if not value:
        pytest.skip(f"input: {ENV_PREFIX}{name} is not set")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"input: {ENV_PREFIX}{name}={value} does not exist")
    return path


def need_cuda(what: str) -> torch.device:
    if not CUDA:
        pytest.skip(f"cuda: {what} runs only on a CUDA device")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# The official source
# ---------------------------------------------------------------------------


class OfficialKernelUnavailable(RuntimeError):
    def __init__(self, symbol: str, why: str):
        super().__init__(f"kernel.{symbol}: {why}")
        self.symbol = symbol


def _stub_kernel_module(why: str) -> types.ModuleType:
    """``kernel`` for a host that cannot run it: every symbol model.py
    imports raises, so a piece reaching one is skipped, never emulated."""
    mod = types.ModuleType("kernel")

    def stub(symbol):
        def call(*args, **kwargs):
            raise OfficialKernelUnavailable(symbol, why)

        call.__name__ = symbol
        return call

    for symbol in KERNEL_SYMBOLS:
        setattr(mod, symbol, stub(symbol))
    return mod


def _tag_kernel_errors(mod: types.ModuleType) -> None:
    """A TileLang build or launch failure names its symbol."""
    gpu = torch.cuda.get_device_name()

    def tagged(symbol, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - becomes the skip reason
                raise OfficialKernelUnavailable(
                    symbol, f"TileLang could not run it on {gpu}: {type(exc).__name__}: {exc}"
                ) from exc

        return call

    for symbol in KERNEL_SYMBOLS:
        setattr(mod, symbol, tagged(symbol, getattr(mod, symbol)))


def _exec_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclasses.dataclass
class Official:
    model: types.ModuleType
    engram: types.ModuleType
    kernel: types.ModuleType


@functools.cache
def _official() -> Official:
    work = Path(tempfile.mkdtemp(prefix="dsv41-official-"))
    for name, env in SOURCE_FILES.items():
        shutil.copyfile(input_path(env), work / f"{name}.py")
    names = ("kernel", "engram", "image_processor", "vision")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        if CUDA:
            try:
                kernel = _exec_module("kernel", work / "kernel.py")
                _tag_kernel_errors(kernel)
            except ImportError as exc:
                kernel = _stub_kernel_module(f"kernel.py does not import here: {exc}")
        else:
            kernel = _stub_kernel_module("a TileLang CUDA kernel; this host has no CUDA device")
        sys.modules["kernel"] = kernel
        engram = _exec_module("engram", work / "engram.py")
        _exec_module("image_processor", work / "image_processor.py")
        _exec_module("vision", work / "vision.py")
        model = _exec_module("dsv41_official_model", work / "model.py")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    return Official(model=model, engram=engram, kernel=kernel)


def official() -> Official:
    for env in SOURCE_FILES.values():
        input_path(env)
    return _official()


@contextlib.contextmanager
def official_kernels():
    """Skip, naming the symbol, when a piece reaches an official kernel this
    host cannot run."""
    try:
        yield
    except OfficialKernelUnavailable as exc:
        pytest.skip(str(exc))


@functools.cache
def _config() -> dict:
    return json.loads(input_path("OFFICIAL_CONFIG_JSON").read_text())


def model_args(max_seq_len: int):
    M = official().model
    cfg = _config()
    known = {f.name for f in dataclasses.fields(M.ModelArgs)}
    unknown = sorted(set(cfg) - known)
    assert not unknown, f"inference/config.json keys ModelArgs lacks: {unknown}"
    cfg = {k: v for k, v in cfg.items() if k != "dtype"}
    return M.ModelArgs(**cfg, max_batch_size=1, max_seq_len=max_seq_len, dtype="bf16")


@functools.cache
def hf_text_config() -> types.SimpleNamespace:
    cfg = json.loads(input_path("HF_CONFIG_JSON").read_text())
    return types.SimpleNamespace(**cfg.get("text_config", cfg))


@contextlib.contextmanager
def official_build():
    """Module construction as generate.py does it (BF16 default dtype), with
    the BF16 linear path selected (``ModelArgs.dtype = "bf16"``)."""
    M = official().model
    prev_dtype, prev_default = torch.get_default_dtype(), M.default_dtype
    torch.set_default_dtype(BF16)
    M.default_dtype = BF16
    try:
        yield M
    finally:
        torch.set_default_dtype(prev_dtype)
        M.default_dtype = prev_default


_LOW = (torch.bfloat16, torch.float32, torch.float16)


@contextlib.contextmanager
def official_fp64():
    """Run official code with its float casts mapped to float64."""
    orig_float, orig_to, orig_bf16 = torch.Tensor.float, torch.Tensor.to, torch.Tensor.bfloat16

    def to(self, *args, **kwargs):
        args = tuple(torch.float64 if a in _LOW else a for a in args)
        if kwargs.get("dtype") in _LOW:
            kwargs["dtype"] = torch.float64
        return orig_to(self, *args, **kwargs)

    def as_double(self, *args, **kwargs):
        return orig_to(self, torch.float64)

    torch.Tensor.float = as_double
    torch.Tensor.bfloat16 = as_double
    torch.Tensor.to = to
    try:
        yield
    finally:
        torch.Tensor.float, torch.Tensor.to, torch.Tensor.bfloat16 = orig_float, orig_to, orig_bf16


def fp64_copy(module: torch.nn.Module) -> torch.nn.Module:
    clone = copy.deepcopy(module)
    for t in list(clone.parameters()) + list(clone.buffers()):
        if t.is_floating_point():
            t.data = t.data.double()
    return clone


# ---------------------------------------------------------------------------
# Official weights
# ---------------------------------------------------------------------------

FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


@functools.cache
def _official_index() -> dict:
    return json.loads(input_path("OFFICIAL_INDEX_JSON").read_text())["weight_map"]


def official_tensor(name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    """An official checkpoint tensor, or rows ``rows`` of a 2-D one."""
    from safetensors import safe_open

    number = _official_index()[name].split("-")[1]
    with safe_open(str(input_path(f"OFFICIAL_SHARD_{number}")), "pt") as f:
        if rows is None:
            return f.get_tensor(name)
        sl = f.get_slice(name)
        return torch.stack([sl[r : r + 1][0] for r in rows.tolist()])


def expand_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 weight with UE8M0 32x32 block scales -> the same values in BF16
    (convert.py's dequantisation of wo_a)."""
    out_f, in_f = weight.shape
    s = scale.float().repeat_interleave(32, 0)[:out_f].repeat_interleave(32, 1)[:, :in_f]
    w = weight.float() * s
    w16 = w.to(BF16)
    assert torch.equal(w16.float(), w), "an FP8 block product is not a BF16 value"
    return w16


def expand_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 (low nibble first, as convert.py reads it) with UE8M0 per
    32 along K -> the same values in BF16."""
    b = packed.view(torch.uint8)
    vals = torch.stack([FP4_VALUES[(b & 0x0F).long()], FP4_VALUES[(b >> 4).long()]], -1).flatten(1)
    w = vals * scale.float().repeat_interleave(32, 1)
    w16 = w.to(BF16)
    assert torch.equal(w16.float(), w)
    return w16


def official_weight(name: str) -> torch.Tensor:
    """A checkpoint tensor as the BF16-path module holds it."""
    t = official_tensor(name)
    scale_name = name.rsplit(".", 1)[0] + ".scale"
    if t.dtype == torch.float8_e4m3fn and scale_name in _official_index():
        return expand_fp8(t, official_tensor(scale_name))
    if t.dtype == torch.int8 and scale_name in _official_index():
        return expand_fp4(t, official_tensor(scale_name))
    return t


def load_official(module: torch.nn.Module, prefix: str, skip=()) -> None:
    """Copy the checkpoint values into ``module``'s parameters, each in the
    parameter's own dtype, as ``load_state_dict`` does."""
    for name, param in module.named_parameters():
        if name.endswith(".scale") or any(name.startswith(s) for s in skip):
            continue
        value = official_weight(prefix + name)
        assert value.shape == param.shape, (name, value.shape, param.shape)
        with torch.no_grad():
            param.copy_(value)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@functools.cache
def activation_spec() -> dict:
    spec = json.loads(input_path("ACTIVATIONS").read_text())
    assert spec.get("schema") == "dsv41-reference-activations/1", spec.get("schema")
    return spec


@functools.cache
def prompt_ids(n: int) -> torch.Tensor:
    """The first ``n`` tokens of the corpus documents the spec names,
    concatenated: a real prompt in the served tokenizer."""
    corpus = json.loads(input_path("PROMPT_CORPUS").read_text())
    ids: list[int] = []
    for d in activation_spec()["prompt"]["docs"]:
        ids += corpus["docs"][d]["ids"]
    assert len(ids) >= n, (len(ids), n)
    return torch.tensor(ids[:n], dtype=torch.int64)


@functools.cache
def embedding(n: int) -> torch.Tensor:
    """[n, dim] BF16: the official ``ParallelEmbedding`` of the prompt."""
    with official_build() as M:
        emb = M.ParallelEmbedding(_config()["vocab_size"], _config()["dim"])
    load_official(emb, "embed.")
    with torch.inference_mode():
        out = emb(prompt_ids(n)[None])[0]
    return out.clone()  # an ordinary tensor: callers use it outside inference mode


def stream(layer: int, n: int) -> torch.Tensor:
    """[n, hc, dim] BF16 hidden-state stream entering ``layer``. Layer 0's is
    the real one (the embedding repeated, as ``Transformer.forward`` builds
    it); later layers' are synthetic per the spec: the embedding with a
    random per-copy gain plus Gaussian noise at the spec's RMS ratio."""
    e = embedding(n)
    hc = _config()["hc_mult"]
    if layer == 0:
        return e[:, None, :].expand(-1, hc, -1).contiguous()
    spec = activation_spec()
    assert spec.get("synthetic") is True
    s = spec["stream"]
    g = torch.Generator().manual_seed(int(spec["seed"]) * 1000 + layer)
    gain = 1.0 + float(s["copy_gain_std"]) * torch.randn(hc, generator=g)
    noise = torch.randn(n, hc, e.shape[-1], generator=g)
    rms = e.float().square().mean().sqrt()
    ratio = float(s["noise_rms_ratio"][str(layer)])
    return (e.float()[:, None, :] * gain[None, :, None] + noise * rms * ratio).to(BF16)


def pre_mix(layer: int, n: int) -> torch.Tensor:
    """[n, hc] FP32 collapse weights: the identity one-hot at layer 0
    (``make_identity_pre_mix``), synthetic positive mixes elsewhere."""
    hc = _config()["hc_mult"]
    if layer == 0:
        m = torch.zeros(n, hc)
        m[:, 0] = 1.0
        return m
    g = torch.Generator().manual_seed(int(activation_spec()["seed"]) * 1000 + 500 + layer)
    return torch.sigmoid(torch.randn(n, hc, generator=g)) + 1e-6


def sublayer_input(layer: int, n: int, norm: str) -> torch.Tensor:
    """[n, dim] BF16: official ``hc_pre`` then ``RMSNorm`` (``attn_norm`` or
    ``ffn_norm``) of the layer's stream."""
    M = official().model
    with official_build():
        rms = M.RMSNorm(_config()["dim"], _config()["norm_eps"])
    load_official(rms, f"layers.{layer}.{norm}.")
    with torch.inference_mode():
        x = M.Block.hc_pre(None, stream(layer, n)[None], pre_mix(layer, n)[None])
        return rms(x)[0]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def _errors(y: torch.Tensor, exact: torch.Tensor) -> tuple[float, float]:
    """(max, mean) of |y - exact| over the finite FP64 values; infinite when
    a non-finite FP64 value is not matched exactly."""
    y, exact = y.detach().double().cpu(), exact.detach().double().cpu()
    assert y.shape == exact.shape, (y.shape, exact.shape)
    finite = torch.isfinite(exact)
    a, b = y[~finite], exact[~finite]
    if not bool(((a == b) | (a.isnan() & b.isnan())).all()):
        return math.inf, math.inf
    d = (y[finite] - exact[finite]).abs()
    return (float(d.max()), float(d.mean())) if d.numel() else (0.0, 0.0)


def check_fp64(piece: str, ours, theirs, exact, record_property):
    kind, factor = TOLERANCES[piece]
    assert kind == "fp64"
    om, oa = _errors(ours, exact)
    tm, ta = _errors(theirs, exact)
    for k, v in (("ours_max", om), ("ours_mean", oa), ("official_max", tm), ("official_mean", ta)):
        record_property(f"{piece}.{k}", v)
    scale = float(exact.double().abs().max()) if exact.numel() else 0.0
    print(f"{piece}: vs FP64 ours max {om:.3e} mean {oa:.3e} | official max {tm:.3e} "
          f"mean {ta:.3e} | max |FP64| {scale:.3e}")
    assert om <= factor * tm, f"{piece}: max error {om:.3e} > {factor} x official {tm:.3e}"
    assert oa <= factor * ta, f"{piece}: mean error {oa:.3e} > {factor} x official {ta:.3e}"


def check_exact(piece: str, ours: torch.Tensor, theirs: torch.Tensor, record_property):
    assert TOLERANCES[piece] == "exact"
    ours, theirs = ours.detach().cpu(), theirs.detach().cpu()
    assert ours.shape == theirs.shape, (piece, ours.shape, theirs.shape)
    if ours.is_floating_point():
        same = ours.dtype == theirs.dtype and torch.equal(
            ours.contiguous().view(torch.uint8), theirs.contiguous().view(torch.uint8)
        )
        diff = (ours.double() - theirs.double()).abs()
        n_diff = int((diff > 0).sum())
        record_property(f"{piece}.max_abs", float(diff.max()) if diff.numel() else 0.0)
    else:
        same = torch.equal(ours.long(), theirs.long())
        n_diff = int((ours.long() != theirs.long()).sum())
    record_property(f"{piece}.n_diff", n_diff)
    print(f"{piece}: " + ("bitwise equal" if same else f"{n_diff} of {ours.numel()} differ")
          + f" ({ours.numel()} elements)")
    assert same, f"{piece}: {n_diff} of {ours.numel()} elements differ"


def check_rel(piece: str, ours: torch.Tensor, theirs: torch.Tensor, record_property):
    kind, r = TOLERANCES[piece]
    assert kind == "rel"
    d = float((ours.double().cpu() - theirs.double().cpu()).abs().max())
    scale = float(theirs.double().abs().max())
    record_property(f"{piece}.max_abs", d)
    record_property(f"{piece}.max_rel", d / scale if scale else d)
    print(f"{piece}: max |port - official| {d:.3e}, max |official| {scale:.3e}")
    assert d <= r * scale, f"{piece}: {d:.3e} > {r} x {scale:.3e}"


def bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    a = v.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def check_selection(piece: str, scores: torch.Tensor, ours: list, theirs: list, record_property):
    """Per row: picks outside the other side's set must tie, within the
    tolerance's BF16 ulps, with the row's k-th score under ``scores``."""
    kind, ulps = TOLERANCES[piece]
    assert kind == "selection"
    rows_diff = bad = 0
    for r, (a, b) in enumerate(zip(ours, theirs)):
        a, b = set(a), set(b)
        if a == b:
            continue
        rows_diff += 1
        row = scores[r].double().cpu()
        if len(a) != len(b):
            bad += abs(len(a) - len(b))
        kth = min(float(row[sorted(a)].min()) if a else math.inf,
                  float(row[sorted(b)].min()) if b else math.inf)
        band = ulps * float(bf16_ulp(torch.tensor(kth))) if math.isfinite(kth) else 0.0
        bad += sum(1 for i in a ^ b if not abs(float(row[i]) - kth) <= band)
    record_property(f"{piece}.rows_differing", rows_diff)
    record_property(f"{piece}.picks_outside_tie_band", bad)
    print(f"{piece}: {rows_diff} of {len(ours)} rows pick other sets; "
          f"{bad} differing picks outside the tie band")
    assert bad == 0, f"{piece}: {bad} picks differ by more than a tie"


def check_fp8_step(piece: str, ours: torch.Tensor, theirs: torch.Tensor, record_property):
    kind, n = TOLERANCES[piece]
    assert kind == "fp8_step"
    ours, theirs = ours.float().cpu(), theirs.float().cpu()
    d = (ours - theirs).abs()
    mag = torch.maximum(ours.abs(), theirs.abs()).clamp_min(2.0**-126)
    step = torch.exp2(torch.floor(torch.log2(mag)) - 3)
    worst = float((d / step).max()) if d.numel() else 0.0
    n_diff = int((d > 0).sum())
    record_property(f"{piece}.n_diff", n_diff)
    record_property(f"{piece}.max_steps", worst)
    record_property(f"{piece}.max_abs", float(d.max()) if d.numel() else 0.0)
    print(f"{piece}: {n_diff} of {d.numel()} differ, worst {worst:.3g} FP8 steps, "
          f"max abs {float(d.max()):.3e}")
    assert worst <= n, f"{piece}: an element differs by {worst:.3g} FP8 steps"


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

LAYERS_BY_RATIO = {0: 0, 2: 2, 1: 20}


def _port_freqs(ratio: int, rope_dim: int, device=None) -> torch.Tensor:
    from vllm_sm80.models.deepseek_v41.sm80_attention import _rope_params

    return core.rope_freqs(*_rope_params(hf_text_config(), ratio, rope_dim), device=device)


@pytest.mark.parametrize("ratio", [0, 2, 1])
def test_rope_table(ratio, record_property):
    """``Attention.freqs_cis`` (``precompute_freqs_cis`` as each layer type
    calls it) against the port's ``rope_freqs`` with the parameters it reads
    from the served config, as ``polar(1, position * freq)``."""
    n = 4096
    args = model_args(n)
    with official_build() as M:
        attn = M.Attention(LAYERS_BY_RATIO[ratio], args)
    freqs = _port_freqs(ratio, args.rope_head_dim)
    pos = torch.arange(n, dtype=torch.float32)
    ours = torch.polar(torch.ones(n, freqs.numel()), pos[:, None] * freqs)
    check_exact("rope.table", torch.view_as_real(ours), torch.view_as_real(attn.freqs_cis), record_property)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("ratio", [0, 2])
def test_rope_rotate(ratio, inverse, record_property):
    """``apply_rotary_emb`` on the RoPE tail of a [1, s, h, 512] BF16 tensor
    against the port's ``rope``."""
    n, h = 1024, 4
    args = model_args(n)
    with official_build() as M:
        attn = M.Attention(LAYERS_BY_RATIO[ratio], args)
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(1, n, h, args.head_dim, generator=g) * 3).to(BF16)
    theirs = x.clone()
    M.apply_rotary_emb(theirs[..., -args.rope_head_dim :], attn.freqs_cis[:n], inverse)
    ours = core.rope(x[0], torch.arange(n), _port_freqs(ratio, args.rope_head_dim), inverse=inverse)
    check_exact("rope.rotate", ours, theirs[0], record_property)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


def _single_request(positions: torch.Tensor, ratio: int) -> core.Sm80AttnBatch:
    """One request at ``positions``: closed group j at compressed slot j, a
    ratio-2 state ring of 8 rows in ring block 1 (slot ``8 + position % 8``):
    block 0 is the null block, which the port's masked writes target and no
    request owns."""
    T = positions.shape[0]
    return core.Sm80AttnBatch(
        positions=positions,
        token_to_req=torch.zeros(T, dtype=torch.int32),
        query_start_loc=torch.tensor([0, T], dtype=torch.int32),
        swa_slots=torch.full((T,), -1, dtype=torch.int64),
        swa_indices=torch.empty(T, 0, dtype=torch.int32),
        comp_slots=torch.where((positions + 1) % ratio == 0, positions // ratio, -1),
        ring_slots=8 + positions % 8 if ratio == 2 else None,
    )


@pytest.mark.parametrize("layer,prefill,decode", [(2, 1537, 5), (20, 1536, 3)])
def test_compressor(layer, prefill, decode, record_property):
    """``Compressor.forward`` (a prefill from 0 ending in a partial group,
    then one-token decode steps) against the port's ``compress`` fed the
    projection the reference computes (``wkv``, ``wgate`` of the same input):
    the latent of every closed group."""
    n = prefill + decode
    x = sublayer_input(layer, n, "attn_norm")
    args = model_args(n + 8)
    with official_build() as M:
        comp = M.Compressor(args, layer)
    load_official(comp, f"layers.{layer}.attn.compressor.")
    comp64 = fp64_copy(comp)
    ratio = comp.compress_ratio
    steps = [(0, prefill)] + [(prefill + i, 1) for i in range(decode)]
    theirs, exact, ours = [], [], []
    ring = torch.zeros(2, 8, 2 * comp.head_dim)
    cfg = types.SimpleNamespace(compress_ratio=ratio, head_dim=comp.head_dim, eps=comp.norm.eps)
    with torch.inference_mode():
        for start, length in steps:
            xs = x[None, start : start + length]
            out = comp(xs, start)
            with official_fp64():
                out64 = comp64(xs.double(), start)
            if out is not None:
                theirs.append(out[0])
                exact.append(out64[0])
            # The port's input: the layer's [kv | score] projection in FP32.
            if ratio > 1:
                xf = xs[0].float()
                kv_score = torch.cat([F.linear(xf, comp.wkv.weight), F.linear(xf, comp.wgate.weight)], -1)
            else:
                kv_score = F.linear(xs[0], comp.wkv.weight).float()
            batch = _single_request(torch.arange(start, start + length), ratio)
            closes, latent = core.compress(cfg, batch, kv_score, comp.norm.weight, ring)
            ours.append(latent[closes])
    theirs, exact, ours = torch.cat(theirs), torch.cat(exact), torch.cat(ours)
    check_fp64("compressor.latent", ours, theirs, exact, record_property)


# ---------------------------------------------------------------------------
# Candidate blocks (layer 20, the candidate source)
# ---------------------------------------------------------------------------


def test_candidate_blocks(record_property):
    """``select_candidate_blocks`` on index scores over more compressed
    positions than ``candidate_topk_blocks * candidate_block_size`` (so the
    selection bites), against the port's ``select_candidate_blocks`` on the
    same scores.

    The scores are the port's (``index_scores_torch`` over index keys and
    queries the port makes from the official weights): the official score
    path needs ``kernel.fp4_act_quant``; both selections read one matrix."""
    layer, n = 20, 18432
    args = model_args(n)
    assert args.compress_ratios[layer] == 1 and args.candidate_source_layer == layer
    blocks, bsize = args.candidate_topk_blocks, args.candidate_block_size
    assert n > blocks * bsize
    x = sublayer_input(layer, n, "attn_norm")
    with official_build() as M:
        attn = M.Attention(layer, args)
    load_official(attn, f"layers.{layer}.attn.", skip=("wo_a", "wo_b", "wq_b", "attn_sink"))
    freqs = _port_freqs(1, args.rope_head_dim)
    eps = args.norm_eps
    ind = attn.indexer
    with torch.inference_mode():
        latent = core.rms_norm(F.linear(x, attn.compressor.wkv.weight), attn.compressor.norm.weight, eps)
        weights = types.SimpleNamespace(
            wk=lambda t: F.linear(t, ind.wk.weight), k_norm=ind.k_norm.weight, freqs=freqs,
            index_wq_b=lambda qr: F.linear(qr, ind.wq_b.weight),
        )
        cfg = types.SimpleNamespace(
            eps=eps, index_n_heads=args.index_n_heads, index_head_dim=args.index_head_dim,
            index_weight_scale=args.index_head_dim**-0.5 * args.index_n_heads**-0.5,
        )
        keys = core.index_keys(cfg, weights, latent, torch.arange(n))
        rows = torch.tensor([blocks * bsize - 1, blocks * bsize, blocks * bsize + 3, n - 9, n - 2, n - 1])
        qr = core.rms_norm(F.linear(x[rows], attn.wq_a.weight), attn.q_norm.weight, eps)
        iq, iw = core.index_query(cfg, weights, qr, F.linear(x[rows], ind.weights_proj.weight), rows)
        n_vis = rows + 1
        scores = core.index_scores_torch(
            iq, iw, n_vis, torch.zeros(len(rows), dtype=torch.int64), keys.view(1, n, -1),
            torch.zeros(1, 1, dtype=torch.int32), n, args.index_n_heads,
        )
        theirs = M.select_candidate_blocks(scores, n_vis[:, None], blocks, bsize)
        ids = core.select_candidate_blocks(scores, n_vis, blocks, bsize)
    block_scores = F.pad(scores, (0, -n % bsize), value=float("-inf")).unflatten(-1, (-1, bsize)).amax(-1)
    block_scores[torch.arange(len(rows)), (n_vis - 1) // bsize] = float("inf")  # pinned, both sides
    ours_sets = [sorted(i for i in r.tolist() if i >= 0) for r in ids]
    theirs_sets = [sorted(set((theirs[r].nonzero().flatten() // bsize).tolist())) for r in range(len(rows))]
    record_property("candidate_blocks.kept_per_row", [len(s) for s in theirs_sets])
    check_selection("candidate_blocks", block_scores, ours_sets, theirs_sets, record_property)


# ---------------------------------------------------------------------------
# MoE routing
# ---------------------------------------------------------------------------


def _official_gate(layer: int):
    args = model_args(64)
    with official_build() as M:
        gate = M.Gate(layer, args)
    load_official(gate, f"layers.{layer}.ffn.gate.", skip=("bias_vl",))
    return gate, args


def _sorted_by_id(w: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    return torch.gather(w, -1, torch.argsort(i, -1))


@pytest.mark.parametrize("layer", [0, 2, 20])
def test_gate(layer, record_property):
    """``Gate.forward`` (sqrt-softplus scores, bias-steered top-6, weights
    from the unbiased scores renormalised + 1e-20 and times route_scale)
    against the port's ``route`` on the FP32 logits its gate computes, with
    the same weights."""
    n = 1536
    x = sublayer_input(layer, n, "ffn_norm")
    gate, args = _official_gate(layer)
    with torch.inference_mode():
        w_t, i_t = gate(x)
        g64 = fp64_copy(gate)
        with official_fp64():
            w_64, i_64 = g64(x.double())
        logits = F.linear(x.float(), gate.weight.float())
        w_o, i_o = route(logits, gate.bias, gate.topk, renormalize=args.norm_topk_prob,
                         routed_scaling_factor=args.route_scale)
        select64 = F.softplus(F.linear(x.double(), gate.weight.double())).sqrt() + gate.bias.double()
    check_selection("gate.indices", select64, [r.tolist() for r in i_o], [r.tolist() for r in i_t],
                    record_property)
    same = (torch.sort(i_o, -1).values == torch.sort(i_t, -1).values).all(-1)
    same &= (torch.sort(i_64, -1).values == torch.sort(i_t, -1).values).all(-1)
    check_fp64("gate.weights", _sorted_by_id(w_o, i_o)[same], _sorted_by_id(w_t, i_t)[same],
               _sorted_by_id(w_64, i_64)[same], record_property)


_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


@functools.cache
def _exl3_manifest(override: bool):
    """The served checkpoint as the EXL3 loader opens it; with ``override``
    the sidecar the served settings name is applied as the loader applies it
    (``Exl3Config`` construction in vllm_sm80/kernels/exl3/quant.py)."""
    from vllm import envs
    from vllm_sm80.kernels.exl3.override import apply_override
    from vllm_sm80.kernels.exl3.utils import Exl3Manifest

    manifest = Exl3Manifest(input_path("EXL3_INDEX_JSON").parent)
    if override:
        if not envs.VLLM_EXL3_OVERRIDE:
            pytest.skip("input: VLLM_EXL3_OVERRIDE (a served setting, settings/model.toml) is not set")
        apply_override(manifest, envs.VLLM_EXL3_OVERRIDE, envs.VLLM_EXL3_OVERRIDE_SHA256)
    return manifest


def _served_tensor(name: str, override: bool) -> torch.Tensor:
    h = _exl3_manifest(override).tensors[name]
    return torch.frombuffer(bytearray(h.read_bytes()), dtype=_ST_DTYPES[h.dtype]).reshape(h.shape)


def _served_routing_flips(layer: int, override: bool, record_property, piece: str):
    """Official ``Gate.forward`` with the official tensors against the
    port's ``route`` with the gate weight and bias the served loader reads:
    (tokens routed to another expert set, the sorted expert ids of each)."""
    n = 1536
    x = sublayer_input(layer, n, "ffn_norm")
    gate, args = _official_gate(layer)
    weight = _served_tensor(f"layers.{layer}.ffn.gate.weight", override).float()
    bias = _served_tensor(f"layers.{layer}.ffn.gate.bias", override).float()
    with torch.inference_mode():
        _, i_t = gate(x)
        _, i_s = route(F.linear(x.float(), weight), bias, gate.topk, renormalize=args.norm_topk_prob,
                       routed_scaling_factor=args.route_scale)
    i_t, i_s = torch.sort(i_t, -1).values, torch.sort(i_s, -1).values
    flips = int((i_t != i_s).any(-1).sum())
    record_property(f"{piece}.tokens_routed_differently", flips)
    record_property(f"{piece}.bias_max_abs", float((bias.double() - gate.bias.double()).abs().max()))
    record_property(f"{piece}.weight_max_abs", float((weight.double() - gate.weight.double()).abs().max()))
    print(f"{piece} L{layer}: {flips} of {n} tokens pick another expert set")
    return flips, i_s, i_t


@pytest.mark.parametrize("layer", [0, 2, 20])
def test_gate_served_tensors(layer, record_property):
    """Routing with the gate tensors the served loader reads (the export
    plus the override sidecar) equals the official routing."""
    _, ours, theirs = _served_routing_flips(layer, True, record_property, "gate.served_tensors")
    check_exact("gate.served_tensors", ours, theirs, record_property)


@pytest.mark.parametrize("layer", [0, 2, 20])
def test_gate_export_f16_documented(layer, record_property):
    """Documentation of why the override exists: the export alone holds the
    gate correction bias (F32 in the official checkpoint) and the gate weight
    (BF16) in F16, and with them some tokens route to other experts."""
    assert TOLERANCES["gate.export_f16"] == "documented_loss"
    flips, _, _ = _served_routing_flips(layer, False, record_property, "gate.export_f16")
    assert flips > 0, "the export's F16 gate tensors route like the official ones: the override is stale"


# ---------------------------------------------------------------------------
# Engram
# ---------------------------------------------------------------------------


@functools.cache
def _tokenizer():
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(input_path("OFFICIAL_TOKENIZER_JSON")))


@functools.cache
def _official_hash_state(max_seq_len: int):
    o = official()
    args = model_args(max_seq_len)
    layout = o.engram.EngramLayout.from_args(args)
    return layout, o.engram.NgramHashState(args, layout, _tokenizer())


def test_engram_layout(record_property):
    """Bucket primes and offsets, hash multipliers, the compressed token map
    and its pad id: official ``EngramLayout`` / ``NgramHashState`` against
    the port's."""
    from vllm.models.deepseek_v41.common import engram as ours

    layout, state = _official_hash_state(64)
    cfg = hf_text_config()
    mine = ours.EngramLayout(cfg)
    check_exact("engram.layout", torch.tensor(mine.primes), torch.tensor(layout.primes), record_property)
    check_exact("engram.layout", mine.offsets, state.offsets, record_property)
    mult = ours.compute_hash_multipliers(mine.layer_ids, mine.max_ngram_size, mine.compressed_vocab_size)
    check_exact("engram.layout", mult, state.multipliers, record_property)
    token_map, size = ours.build_compressed_token_map(_tokenizer())
    assert size == cfg.engram_compressed_vocab_size
    check_exact("engram.layout", torch.tensor(token_map), state.token_map, record_property)
    assert token_map[cfg.engram_pad_token_id] == state.pad_id


@functools.cache
def _official_hashes(n: int) -> torch.Tensor:
    _, state = _official_hash_state(n)
    with torch.inference_mode():
        out = state(prompt_ids(n)[None], 0)[0]  # [n, layers, cols]
    return out.clone()


def test_engram_hash_ids(record_property):
    """``NgramHashState.forward`` on the real prompt against the port's hash
    kernel (one request, prefill from position 0, no lookback)."""
    dev = need_cuda("the port's n-gram hash (_hash_ids_kernel, Triton)")
    from vllm.models.deepseek_v41.common import engram as ours

    n = 2048
    theirs = _official_hashes(n)
    layout = ours.EngramLayout(hf_text_config())
    token_map, _ = ours.build_compressed_token_map(_tokenizer())
    state = ours.NgramHashState.__new__(ours.NgramHashState)
    torch.nn.Module.__init__(state)
    state.layout, state.block_size, state.use_slot_cache, state._cache = layout, 64, False, None
    state.lookback_depth = layout.max_ngram_size - 1
    state.pad_id = token_map[layout.pad_token_id]
    state.token_map = torch.tensor(token_map, dtype=torch.int32, device=dev)
    state.primes = torch.tensor(layout.primes, device=dev)
    state.offsets = layout.offsets.to(dev)
    state.multipliers = ours.compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size).to(dev)
    ids = prompt_ids(n).to(dev)
    lookback = torch.full((1, state.lookback_depth), -1, dtype=ids.dtype, device=dev)
    with torch.inference_mode():
        mine = state(ids, torch.arange(n, device=dev), torch.tensor([0, n], dtype=torch.int32, device=dev),
                     torch.zeros(n, dtype=torch.bool, device=dev), lookback,
                     torch.zeros_like(lookback, dtype=torch.bool), None, None)
    check_exact("engram.hash_ids", mine.long(), theirs, record_property)


def _engram_rows(layer_index: int, n: int):
    """The official table rows the prompt's first ``n`` tokens look up, as a
    small table plus remapped ids (a lookup does not depend on the size of
    the table)."""
    layer = _config()["engram_layer_ids"][layer_index]
    hashes = _official_hashes(n)[:, layer_index]
    uniq, inv = torch.unique(hashes, return_inverse=True)
    values = official_tensor(f"layers.{layer}.engram.embed.weight", uniq)
    scales = official_tensor(f"layers.{layer}.engram.embed.scale", uniq)
    return layer, inv, values, scales


def _small_engram_embedding(values, scales):
    M = official().model
    emb = M.ParallelEngramEmbedding(values.shape[0], values.shape[1])
    emb.weight.data, emb.scale.data = values, scales
    return emb


def test_engram_dequant(record_property):
    """``ParallelEngramEmbedding.forward`` (E4M3 rows x UE8M0 per 32 -> BF16)
    against the port's row dequant (its torch definition, and its Triton
    kernel on CUDA)."""
    from vllm_sm80.models.deepseek_v41.ops.sm80_engram_dequant import (
        dequant_engram_rows,
        dequant_engram_rows_reference,
    )

    n = 512
    _, inv, values, scales = _engram_rows(0, n)
    table = _small_engram_embedding(values, scales)
    with torch.inference_mode():
        theirs = table(inv)  # [n, cols, head_dim]
    v8 = values.view(torch.uint8)[inv.flatten()]
    s8 = scales.view(torch.uint8)[inv.flatten()]
    check_exact("engram.dequant", dequant_engram_rows_reference(v8, s8).view(theirs.shape), theirs,
                record_property)
    if CUDA:
        out = torch.empty(v8.shape, dtype=BF16, device="cuda")
        dequant_engram_rows(v8.cuda().contiguous(), s8.cuda().contiguous(), out)
        check_exact("engram.dequant", out.view(theirs.shape), theirs, record_property)


def test_engram_gate(record_property):
    """``Engram.forward`` after its ``wkv`` (per-copy normalised stream . key
    gate, signed sqrt, sigmoid, the shared value added per copy) against the
    port's fused post-wkv kernel on the same ``wkv`` output."""
    dev = need_cuda("the port's Engram gate (_fused_engram_post_wkv_kernel, Triton)")
    from vllm.models.deepseek_v41.common import engram as ours

    n = 512
    layer, inv, values, scales = _engram_rows(0, n)
    args = model_args(n)
    layout = official().engram.EngramLayout.from_args(args)
    small = dataclasses.replace(layout, num_embeddings=tuple(values.shape[0] for _ in layout.layer_ids))
    with official_build() as M:
        eng = M.Engram(args, layer, small)
    eng.embed = _small_engram_embedding(values, scales)
    load_official(eng, f"layers.{layer}.engram.", skip=("embed",))
    h = stream(layer, n)
    with torch.inference_mode():
        theirs = eng(h[None], inv[None])[0]
        kv = eng.wkv(eng.embed(inv).flatten(-2))
        e64 = fp64_copy(eng)
        with official_fp64():
            exact = e64(h[None].double(), inv[None])[0]
        hd, kvd = h.to(dev), kv.to(dev).contiguous()
        qw, kw = eng.q_weight.to(dev), eng.k_weight.to(dev)
        out = torch.empty_like(hd)
        ours._fused_engram_post_wkv_kernel[(n * args.hc_mult,)](
            hd, kvd, qw, kw, hd, out, n, *hd.stride(), *kvd.stride(), *qw.stride(), *kw.stride(), 0,
            *out.stride(), eng.eps, eng.clamp_value, DIM=args.dim, HC_MULT=args.hc_mult,
            BLOCK_SIZE=1 << (args.dim - 1).bit_length(), HAS_MASK=False, num_warps=8,
        )
    check_fp64("engram.gate", out, theirs, exact, record_property)


# ---------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer", [0, 2])
def test_mhc_collapse_and_post(layer, record_property):
    """``Block.hc_pre`` and ``Block.hc_post`` against the port's
    ``hc_collapse`` and ``mhc_post``."""
    M = official().model
    n = 1536
    s, pm = stream(layer, n), pre_mix(layer, n)
    g = torch.Generator().manual_seed(11)
    x = (torch.randn(n, s.shape[-1], generator=g) * 0.5).to(BF16)
    post = 2 * torch.sigmoid(torch.randn(n, s.shape[1], generator=g))
    comb = torch.softmax(torch.randn(n, s.shape[1], s.shape[1], generator=g), -1)
    with torch.inference_mode():
        theirs_pre = M.Block.hc_pre(None, s[None], pm[None])[0]
        theirs_post = M.Block.hc_post(None, x[None], s[None], post[None], comb[None])[0]
        with official_fp64():
            exact_pre = M.Block.hc_pre(None, s[None].double(), pm[None].double())[0]
            exact_post = M.Block.hc_post(None, x[None].double(), s[None].double(), post[None].double(),
                                         comb[None].double())[0]
        ours_pre = sm80_mhc.hc_collapse(s, pm)
        ours_post = sm80_mhc.mhc_post(x, s, post.unsqueeze(-1), comb)
    check_fp64("mhc.collapse", ours_pre, theirs_pre, exact_pre, record_property)
    check_fp64("mhc.post", ours_post, theirs_post, exact_post, record_property)


def _hc_self(args):
    return types.SimpleNamespace(norm_eps=args.norm_eps, hc_mult=args.hc_mult,
                                 hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps)


def _hc_params(layer: int, which: str, device):
    return tuple(official_tensor(f"layers.{layer}.hc_{which}_{k}").float().to(device)
                 for k in ("fn", "scale", "base"))


@pytest.mark.parametrize("layer", [0, 2])
def test_mhc_mixes(layer, record_property):
    """``Block.hc_mixes`` (RMS-scaled projection of the flattened stream,
    then ``kernel.hc_split_sinkhorn``) against the port's ``hc_mixes``."""
    M = official().model
    args = model_args(64)
    dev = torch.device("cuda") if CUDA else torch.device("cpu")
    n = 1536
    s = stream(layer, n).to(dev)
    fn, scale, base = _hc_params(layer, "attn", dev)
    with torch.inference_mode(), official_kernels():
        theirs = M.Block.hc_mixes(_hc_self(args), s[None], fn, scale, base)
    with torch.inference_mode():
        ours = sm80_mhc.hc_mixes(s, fn, scale, base, args.norm_eps, args.hc_eps, args.hc_sinkhorn_iters)
    for a, b in zip(ours, theirs):
        check_rel("mhc.mixes", a, b[0], record_property)


def test_mhc_delayed_block(record_property):
    """``Block.forward``'s wiring (attention collapses with the carried
    pre-mix, the FFN with the attention's, the FFN's mix carried out) against
    the port's decoder layer forward and the next boundary's post, with both
    sides' attention and FFN the same fixed linear maps (only the mHC is
    under test)."""
    dev = need_cuda("the official Block.forward needs kernel.hc_split_sinkhorn")
    from vllm_sm80.models.deepseek_v41 import sm80_model

    M = official().model
    args = model_args(64)
    layer, n = 2, 256
    s = stream(layer, n).to(dev)
    pm = pre_mix(layer, n).to(dev)
    attn_p, ffn_p = _hc_params(layer, "attn", dev), _hc_params(layer, "ffn", dev)
    norms = {k: official_tensor(f"layers.{layer}.{k}.weight").to(dev) for k in ("attn_norm", "ffn_norm")}
    g = torch.Generator().manual_seed(5)
    wa = (torch.randn(args.dim, args.dim, generator=g) / args.dim**0.5).to(BF16).to(dev)
    wf = (torch.randn(args.dim, args.dim, generator=g) / args.dim**0.5).to(BF16).to(dev)

    blk = _hc_self(args)
    blk.hc_mixes = functools.partial(M.Block.hc_mixes, blk)
    blk.hc_pre = functools.partial(M.Block.hc_pre, blk)
    blk.hc_post = functools.partial(M.Block.hc_post, blk)
    blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base = attn_p
    blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base = ffn_p
    with official_build():
        blk.attn_norm = M.RMSNorm(args.dim, args.norm_eps).to(dev)
        blk.ffn_norm = M.RMSNorm(args.dim, args.norm_eps).to(dev)
    blk.attn_norm.weight.data, blk.ffn_norm.weight.data = norms["attn_norm"], norms["ffn_norm"]
    blk.attn = lambda x, start_pos: F.linear(x, wa)
    blk.ffn = lambda x, image_mask: F.linear(x, wf)
    with torch.inference_mode(), official_kernels():
        out_t, next_t = M.Block.forward(blk, s[None], 0, pm[None], None)

    lay = sm80_model.DeepseekV4DecoderLayer.__new__(sm80_model.DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(lay)
    lay.hc_attn_fn, lay.hc_attn_scale, lay.hc_attn_base = attn_p
    lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base = ffn_p
    lay.rms_norm_eps, lay.hc_eps, lay.hc_post_alpha = args.norm_eps, args.hc_eps, 2.0
    lay.hc_sinkhorn_iters, lay.hc_mult, lay.engram = args.hc_sinkhorn_iters, args.hc_mult, None
    lay.attn_norm = types.SimpleNamespace(weight=norms["attn_norm"], variance_epsilon=args.norm_eps)
    lay.ffn_norm = types.SimpleNamespace(weight=norms["ffn_norm"], variance_epsilon=args.norm_eps)
    lay.attn = lambda positions, x, _: F.linear(x, wa)
    lay.ffn = lambda x, input_ids, meta: F.linear(x, wf)
    with torch.inference_mode():
        x, residual, post, res, ffn_pre, _ = lay(s, torch.arange(n, device=dev), None, pm, None, None, s)
        out_o = sm80_mhc.mhc_post(x, residual, post, res)  # the next boundary's post
    check_rel("mhc.delayed_block", out_o, out_t[0], record_property)
    check_rel("mhc.delayed_block", ffn_pre, next_t[0], record_property)


# ---------------------------------------------------------------------------
# Quantisation points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fp8_ue8m0_32", "fp4_e8m0_32", "fp4_e4m3_16"])
def test_qdq(mode, record_property):
    """``kernel.act_quant(.., "ue8m0", inplace=True)`` and
    ``kernel.fp4_act_quant(.., inplace=True)`` against the port's qdq."""
    dev = torch.device("cuda") if CUDA else torch.device("cpu")
    o = official()
    x = sublayer_input(2, 512, "attn_norm")[:, :512]
    g = torch.Generator().manual_seed(3)
    x = (x.float() * (1 + 4 * torch.rand(x.shape, generator=g))).to(BF16).to(dev)
    theirs = x.clone()
    with official_kernels():
        if mode == "fp8_ue8m0_32":
            o.kernel.act_quant(theirs, 32, "ue8m0", torch.float8_e8m0fnu, True)
            ours = core.qdq_fp8(x, 32)
        elif mode == "fp4_e8m0_32":
            o.kernel.fp4_act_quant(theirs, 32, True)
            ours = core.qdq_fp4(x, 32, "e8m0")
        else:
            o.kernel.fp4_act_quant(theirs, 16, True, scale_dtype=torch.float8_e4m3fn)
            ours = core.qdq_fp4(x, 16, "e4m3")
    check_exact("qdq", ours, theirs, record_property)


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------


def _port_index_scores(run, cfg, weights, attn, x, positions, args) -> torch.Tensor:
    """The port's index scores ([rows, compressed positions]) of the queries
    at ``positions`` over the request's final index-key cache."""
    req = run.reqs[0]
    nb = 1 + max(req.idxk)
    table = torch.zeros(1, nb, dtype=torch.int32, device=x.device)
    for blk, phys in req.idxk.items():
        table[0, blk] = phys
    qr = core.rms_norm(F.linear(x[positions], attn.wq_a.weight), attn.q_norm.weight, cfg.eps)
    iw = F.linear(x[positions], attn.indexer.weights_proj.weight)
    iq, w = core.index_query(cfg, weights, qr, iw, positions)
    n_vis = torch.div(positions + 1, cfg.compress_ratio, rounding_mode="floor")
    width = int(n_vis.max())
    return core.index_scores(iq, w, n_vis, torch.zeros_like(positions), run.idxk, table, width,
                             cfg.index_heads_per_rank)


@pytest.mark.parametrize("layer,prefill,decode", [(0, 300, 2), (2, 2050, 2), (20, 1100, 2)])
def test_attention_layer(layer, prefill, decode, record_property):
    """``Attention.forward`` (window KV, compressor, indexer, candidate
    blocks at layer 20, sparse attention with sinks, output RoPE and FP8
    qdq) against the port's ``forward_layer`` on the same BF16 projections:
    the caches each writes, the compressed positions each selects and the
    ``wo_a`` input, over a prefill from 0 and one-token decode steps."""
    dev = need_cuda("the official attention needs kernel.act_quant, fp4_act_quant and sparse_attn")
    from tests.models.deepseek_v41.sm80 import dsv41_ref as R

    M = official().model
    n = prefill + decode
    args = model_args(n + 8)
    x = sublayer_input(layer, n, "attn_norm").to(dev)
    with official_build():
        attn = M.Attention(layer, args)
    load_official(attn, f"layers.{layer}.attn.")
    attn = attn.to(dev)

    recorded, idx_out = [], []
    real_act_quant = M.act_quant

    def recording_act_quant(t, *a, **k):
        out = real_act_quant(t, *a, **k)
        if t.dim() == 4:  # the per-group wo_a input (model.py:786-787)
            recorded.append(t.clone())
        return out

    if attn.indexer is not None:
        attn.indexer.register_forward_hook(lambda m, i, o: idx_out.append(o.clone()))
    steps = [(0, prefill)] + [(prefill + i, 1) for i in range(decode)]
    M.act_quant = recording_act_quant
    try:
        with torch.inference_mode(), official_kernels():
            for start, length in steps:
                attn(x[None, start : start + length], start)
    finally:
        M.act_quant = real_act_quant

    ratio = attn.compress_ratio
    spec = R.LayerSpec(ratio=ratio, kv_source=attn.is_kv_source, index_source=attn.is_index_source,
                       candidate_source=layer == args.candidate_source_layer,
                       uses_candidates=0 <= args.candidate_source_layer < layer)
    ra = R.RefArgs(dim=args.dim, q_lora_rank=args.q_lora_rank, n_heads=args.n_heads, head_dim=args.head_dim,
                   rope_head_dim=args.rope_head_dim, window_size=args.window_size, o_groups=args.o_groups,
                   index_n_heads=args.index_n_heads, index_head_dim=args.index_head_dim,
                   index_topk=args.index_topk, candidate_topk_blocks=args.candidate_topk_blocks,
                   candidate_block_size=args.candidate_block_size, norm_eps=args.norm_eps, world_size=1,
                   rank=0, max_seq_len=n + 8)
    run = R.PagedRun(ra, [spec], device=dev, num_blocks=6000, low_block=100)
    run.max_tokens = max(prefill, 512)
    run.topk = torch.full((run.max_tokens, ra.index_topk), -7, dtype=torch.int32, device=dev)
    run.candidates = torch.full((run.max_tokens, ra.candidate_topk_blocks), -7, dtype=torch.int32,
                                device=dev)
    cfg = R.core_config(ra, spec)
    weights = core.Sm80AttnWeights(
        q_norm=attn.q_norm.weight, kv_norm=attn.kv_norm.weight, attn_sink=attn.attn_sink.float(),
        freqs=_port_freqs(ratio, args.rope_head_dim, dev), wq_b=lambda qr: F.linear(qr, attn.wq_b.weight),
    )
    if attn.compressor is not None:
        weights.comp_norm = attn.compressor.norm.weight
    if attn.indexer is not None:
        weights.index_wq_b = lambda qr: F.linear(qr, attn.indexer.wq_b.weight)
        if attn.indexer.owns_k:
            weights.wk = lambda t: F.linear(t, attn.indexer.wk.weight)
            weights.k_norm = attn.indexer.k_norm.weight
    ours_o, ours_topk = [], []
    with torch.inference_mode():
        for start, length in steps:
            xs = x[start : start + length]
            batch = run.build([(0, length)])
            qr_kv = torch.cat([F.linear(xs, attn.wq_a.weight), F.linear(xs, attn.wkv.weight)], -1)
            kv_score = None
            if attn.compressor is not None:
                c = attn.compressor
                kv_score = (torch.cat([F.linear(xs.float(), c.wkv.weight), F.linear(xs.float(), c.wgate.weight)], -1)
                            if ratio > 1 else F.linear(xs, c.wkv.weight).float())
            iw = F.linear(xs, attn.indexer.weights_proj.weight) if attn.indexer is not None else None
            out = torch.empty(length, args.n_heads, args.head_dim, dtype=BF16, device=dev)
            caches = run.caches(0)
            core.forward_layer(cfg, weights, caches, batch, qr_kv, kv_score, iw, out)
            run.after_step()
            ours_o.append(out.view(length, args.o_groups, -1))
            if ratio:
                ours_topk.append(caches.topk[:length].clone())

    pos = torch.arange(max(0, n - args.window_size), n)
    theirs_win = attn.window_kv_cache[0, pos % args.window_size]
    ours_win = torch.stack([run.window_row(0, 0, int(p)) for p in pos])
    check_exact("attention.window_kv", ours_win, theirs_win, record_property)
    if attn.is_kv_source:
        groups = n // ratio
        ours_c = torch.stack([run.comp_row(0, j) for j in range(groups)])
        check_exact("attention.compressed_kv", ours_c, attn.compress_kv_cache[0, :groups], record_property)
        if attn.indexer is not None and attn.indexer.owns_k:
            ours_k = torch.stack([run.comp_row(0, j, idxk=True) for j in range(groups)])
            check_exact("attention.index_k", ours_k, attn.indexer.k_cache[0, :groups], record_property)
    if ratio and idx_out:
        theirs_sets, ours_sets, positions = [], [], []
        for (start, length), t_idx, o_idx in zip(steps, idx_out, ours_topk):
            offset = length if start == 0 else args.window_size
            for r in range(length):
                tr = t_idx[0, r]
                theirs_sets.append(sorted((tr[tr >= 0] - offset).tolist()))
                ours_sets.append(sorted(o_idx[r][o_idx[r] >= 0].tolist()))
                positions.append(start + r)
        differing = [i for i, (a, b) in enumerate(zip(ours_sets, theirs_sets)) if a != b]
        scores = torch.zeros(len(positions), 1)
        if differing and attn.indexer is not None:
            with torch.inference_mode():
                rows = _port_index_scores(run, cfg, weights, attn, x,
                                          torch.tensor([positions[i] for i in differing], device=dev), args)
            scores = torch.full((len(positions), rows.shape[1]), float("-inf"))
            scores[differing] = rows.float().cpu()
        check_selection("attention.topk", scores, ours_sets, theirs_sets, record_property)
    theirs_o = torch.cat([t[0] for t in recorded])
    check_fp8_step("attention.wo_a_input", torch.cat(ours_o), theirs_o, record_property)


# ---------------------------------------------------------------------------
# DSpark heads (the last stage: mtp.2)
# ---------------------------------------------------------------------------


def served_head_dtype() -> torch.dtype:
    """ModelConfig.head_dtype of the served target and drafter: the
    checkpoint's config (no head_dtype of its own) after the model-config
    hook the plugin registers for DeepseekV41ForCausalLM."""
    from transformers import PretrainedConfig

    from vllm.config.model import _get_head_dtype
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm_sm80 import plugin

    architecture = "DeepseekV41ForCausalLM"
    saved = MODELS_CONFIG_MAP.get(architecture)
    try:
        plugin.register_hf_config_defaults()
        hf = PretrainedConfig(architectures=[architecture])
        MODELS_CONFIG_MAP[architecture].verify_and_update_model_config(
            types.SimpleNamespace(hf_config=hf, hf_text_config=hf)
        )
    finally:
        MODELS_CONFIG_MAP[architecture] = saved
    return _get_head_dtype(config=hf, dtype=BF16, runner_type="generate")


def test_dspark_heads(record_property, monkeypatch):
    """``DSparkMarkovHead.forward`` (the previous token's rank-256 embedding
    and its vocab bias) and ``DSparkConfidenceHead.forward`` against the
    port's DSpark Markov and confidence heads, built in the model dtype, with
    the same weights and inputs, on one rank (no process group). The bias goes
    through the drafter's logits processor as served (FP32 head dtype)."""
    from tests.models.deepseek_v41.sm80.construct import set_groups
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
    from vllm_sm80.kernels.exl3.quant import Exl3HeadLogitsProcessor
    from vllm.utils.torch_utils import set_default_torch_dtype

    set_groups(monkeypatch, 1, 0)
    args = model_args(64)
    stage = f"mtp.{args.n_mtp_layers - 1}."
    rank, vocab = args.dspark_markov_rank, args.vocab_size
    with official_build() as M:
        markov = M.DSparkMarkovHead(vocab, rank)
        conf = M.DSparkConfidenceHead(args.dim + rank)
    load_official(markov, stage + "markov_head.")
    load_official(conf, stage + "confidence_head.")
    tokens = prompt_ids(256)
    g = torch.Generator().manual_seed(9)
    hidden = torch.randn(len(tokens), args.dim, generator=g).to(BF16)
    with torch.inference_mode():
        bias_t, emb_t = markov(tokens)
        conf_t = conf(hidden, emb_t)
        m64, c64 = fp64_copy(markov), fp64_copy(conf)
        with official_fp64():
            bias_64, _ = m64(tokens)
            conf_64 = c64(hidden.double(), emb_t.double())

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_default_torch_dtype(BF16), set_current_vllm_config(VllmConfig()):
        port_markov = DSparkMarkovHead(vocab, vocab, rank, prefix="markov_head")
        port_conf = DSparkConfidenceHead(args.dim + rank, prefix="confidence_head")
        processor = LogitsProcessor(vocab)
    # The drafter's processor: the head dtype the plugin's model-config hook
    # gives the checkpoint's config, served by the fork's EXL3-head processor
    # (sm80_model.DSparkDeepseekV4ForCausalLM).
    processor.head_dtype = served_head_dtype()
    Exl3HeadLogitsProcessor.adopt(processor)
    for param, name in ((port_markov.markov_w1.weight, "markov_head.embed.weight"),
                        (port_markov.markov_w2.weight, "markov_head.head.weight"),
                        (port_conf.proj.weight, "confidence_head.proj.weight")):
        param.data = official_tensor(stage + name).to(param.dtype)
    with torch.inference_mode():
        emb_o = port_markov.embed(tokens)
        bias_o = port_markov.bias(emb_o, processor)
        conf_o = port_conf(hidden, emb_o)
    embed_equal = emb_o.dtype == emb_t.dtype and torch.equal(emb_o, emb_t)
    record_property("dspark.markov_embed.equal", embed_equal)
    record_property("dspark.markov_bias.port_dtype", str(bias_o.dtype))
    assert embed_equal, "the Markov embeddings differ"
    check_fp64("dspark.confidence", conf_o, conf_t, conf_64, record_property)
    check_fp64("dspark.markov_bias", bias_o, bias_t, bias_64, record_property)
