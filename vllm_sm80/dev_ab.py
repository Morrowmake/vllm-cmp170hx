# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Both arms of one engine switch inside one running server (development).

``VLLM_DEV_AB_SWITCH=<name>`` names one switch of :data:`SWITCHES`. Then:

- the platform sets ``parallel_config.worker_extension_cls`` to
  :class:`DevABWorkerExtension` (:func:`configure`);
- Model Runner V2 captures every CUDA graph once per arm, the boot arm first,
  and ends on the boot arm. Each graph store (descriptor -> graph) becomes an
  :class:`ArmGraphStore`, so replay is keyed by (descriptor, arm)
  (:func:`with_ab_capture`, wrapping
  ``GPUModelRunner.capture_model``). The memory-profiling capture takes both
  arms too, so the KV cache is sized with both arms' graphs reserved;
- ``POST /collective_rpc`` (``VLLM_SERVER_DEV_MODE=1``) with method
  ``set_ab_arm`` and args ``["on" | "off", generation]`` switches every
  worker between engine steps. The ranks agree first (no request resident,
  a newer generation, graphs captured for the arm), then acknowledge a commit
  or a rollback on every rank. Unrecoverable consistency faults terminate the
  workers instead of escaping as ordinary RPC exceptions; each safe reply reports its state. The caller drains requests before the
  call and resets the prefix cache after it.

Unset (the default), :func:`configure` and :func:`capture_model_per_arm`
return at once and nothing else here runs.

An arm is the switch's value: "on" or "off". The boot arm is the value the
server started with. Switches bound at construction (the flags-in-data
all-reduce, the KDA prefill layer gate) must boot on; their off arm sets the
constructed choice aside. DeepSeek MoE prefill also must boot on: its loaded
workspace is retained while the off arm disables the eligible expert gates.
MoE chunk rows are not arms because they determine load-time buffer sizes.

Static DSpark batch-size K tables are supported without changing their policy.
The complete target descriptors, each existing draft-width manager, and decoder
replay's breakable stores are arm-keyed. Identical token counts with different
batch/query lengths remain distinct. Capture lists and dispatch maps are frozen
through capture and checked again before a switch. Adaptive verification and
acceptance/load-adaptive K remain refused. The graph-memory profiler sees the
union, and workspace locks only after both arms pass completeness checks.

For same-arm controls the setter accepts a third boolean, ``replay`` (default
true). False bypasses graph dispatch on every manager while retaining captures;
returning to true reuses them, never recaptures. Both operations require a drained
server and a newer generation. RPC replies include the full graph inventory,
K-table dispatch coverage and per-arm capture bytes. These are allocation deltas,
not standalone graph-pool measurements; shared pool reuse is charged once.

Diagnostic ``VLLM_SM80_CAPTURE`` cannot be combined with the A/B switch:
its graph-store bindings do not preserve both arm stores. Configuration refuses
this composition before installing either A/B extension.

With server dev mode enabled, the scheduler extension also accepts declared
completion arrays via ``vllm_xargs.dev_ab_batch`` (a JSON manifest with ``id``
and ``rows`` containing token counts and SHA256 hashes). ``begin_ab_batch``
installs actual per-step execution tracing against an established same-arm
reference; ``end_ab_batch`` validates completeness and removes tracing. Both
RPCs and ordered admission require both development gates. They record actual
row order, verification/draft K, input tensors and target/draft descriptors,
not merely the graphs captured at boot. Pin mismatches never count as identity.
These synchronizing controls are for correctness, not timing.

Several switches in one server, without a graph set per arm, are
:mod:`vllm_sm80.hot_swap` (``VLLM_HOT_SWAP``): its pairs share the boot's
allocations and graph shapes, and this extension carries its
``set_hot_arms`` and ``get_hot_arms``. A switch whose arms capture different
graphs, and the ordered batches above, stay here.
"""

import functools
import gc
import importlib
import os
import sys
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, NoReturn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_sm80 import hot_swap

logger = init_logger(__name__)

ARMS = ("off", "on")
EXTENSION_QUALNAME = "vllm_sm80.dev_ab.DevABWorkerExtension"


@dataclass(frozen=True)
class Switch:
    """One allowlisted switch.

    ``unavailable(runner)`` returns why the switch cannot be armed in this
    server, or None; ``current(runner)`` its boot value; ``apply(runner, on)``
    sets it in this process, outside CUDA capture, warming the kernels the
    arm needs.
    """

    name: str
    env: str
    graph_bound: bool
    output_identical: bool
    unavailable: Callable[[Any], str | None]
    current: Callable[[Any], bool]
    apply: Callable[[Any, bool], None]


@dataclass
class _ArmState:
    switch: Switch | None = None
    boot: bool | None = None
    arm: bool | None = None
    generation: int = 0
    captured: tuple[bool, ...] = ()
    capture_bytes: dict[str, dict[str, int]] = field(default_factory=dict)
    in_capture: bool = False
    replay: bool = True
    saved: dict[str, Any] = field(default_factory=dict)


_STATE = _ArmState()


def _arm_name(on: bool | None) -> str | None:
    return None if on is None else ARMS[int(on)]


# ---------------------------------------------------------------------------
# Switch adapters
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return bool(int(os.environ.get(name, "0") or "0"))


def _set_env(name: str, on: bool) -> None:
    """Set ``name`` in this process and rebuild vLLM's env cache if it is on."""
    cached = envs._is_envs_cache_enabled()
    if cached:
        envs.disable_envs_cache()
    os.environ[name] = "1" if on else "0"
    if cached:
        envs.enable_envs_cache()






_QSA_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
_QSA_MODEL_MODULE = "vllm.models.qwen4_exp.nvidia.indexer_qsa"
_TOPK_GATE = "vllm_sm80.kernels.indexer.topk_tiefix_gate"
_TOPK_ENV = "VLLM_SM80_TOPK_CANONICAL_TIES"


def _topk_gate():
    """The shared indexer top-k gate, or None when the served model has no
    indexer that calls it. The GLM sparse indexer imports it with the model;
    the Qwen model imports its QSA ops module (and with it the gate) lazily
    (first attention step, or the warmup after KV sizing), which is after the
    profiling capture resolves the switch, so a loaded QSA model has its ops
    module imported here; the gate reads the same boot value."""
    if _TOPK_GATE not in sys.modules and _QSA_MODEL_MODULE in sys.modules:
        try:
            importlib.import_module(_QSA_MODULE)
        except ImportError:
            return None
    return sys.modules.get(_TOPK_GATE)


def _topk_unavailable(runner) -> str | None:
    if _topk_gate() is None:
        return "the GLM or Qwen sparse indexer (shared top-k gate) is not loaded"
    return None


def _topk_apply(runner, on: bool) -> None:
    """Both arms keep VLLM_SM80_TOPK_SORTED as booted: off sorts alone (when
    on), on sorts inside the tie-fix launch."""
    gate = _topk_gate()
    _set_env(_TOPK_ENV, on)
    for cached in (gate.use_sorted_topk, gate.use_tiefix_topk, gate.tiefix_split_rows):
        cached.cache_clear()
    gate.use_tiefix_topk()  # warms the kernels when on


def _marlin_unavailable(runner) -> str | None:
    if not envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
        return "needs VLLM_AMPERE_MARLIN_DECODE_CUDA=1"
    from vllm_sm80.kernels.marlin_moe.extension import decode_v2_op

    op, why = decode_v2_op()
    return why if op is None else None


def _marlin_apply(runner, on: bool) -> None:
    _set_env("VLLM_AMPERE_MARLIN_DECODE_V2", on)
    if not on:
        return
    import torch

    from vllm_sm80.kernels.marlin_moe import decode

    for (device, _stream, n), ws in list(decode._WORKSPACES.items()):
        if n == decode._V2_N:
            decode._warm_v2(torch.device(device), ws)


def _custom_allreduce():
    from vllm.distributed.parallel_state import get_tp_group

    return getattr(get_tp_group().device_communicator, "ca_comm", None)


def _flags_unavailable(runner) -> str | None:
    if getattr(_custom_allreduce(), "_flags", None) is None:
        return (
            "boot with VLLM_SM80_ALLREDUCE=1 and the flags-in-data path "
            "on; its off arm runs the existing kernel"
        )
    return None


def _flags_current(runner) -> bool:
    return getattr(_custom_allreduce(), "_flags", None) is not None


def _flags_apply(runner, on: bool) -> None:
    ca_comm = _custom_allreduce()
    flags = _STATE.saved.setdefault("flags", ca_comm._flags)
    ca_comm._flags = flags if on else None


def _kda_layers(runner) -> list:
    return [m for m in runner.model.modules() if hasattr(m, "_pp_kda_prefill")]


def _kda_unavailable(runner) -> str | None:
    if not any(m._pp_kda_prefill for m in _kda_layers(runner)):
        return (
            "boot with VLLM_SM80_KDA_PREFILL so a layer takes the kernel; its "
            "off arm clears the layer gate"
        )
    return None


def _kda_current(runner) -> bool:
    return any(m._pp_kda_prefill for m in _kda_layers(runner))


def _kda_apply(runner, on: bool) -> None:
    layers = _STATE.saved.setdefault(
        "kda_prefill", [(m, m._pp_kda_prefill) for m in _kda_layers(runner)]
    )
    for module, boot in layers:
        module._pp_kda_prefill = boot if on else False


def _speculator(runner):
    return getattr(runner, "speculator", None)


def _argmax_unavailable(runner) -> str | None:
    speculator = _speculator(runner)
    if not hasattr(speculator, "use_local_argmax_reduction"):
        return "the server has no draft model speculator"
    if speculator.draft_logits is not None:
        return "probabilistic drafting samples full draft logits"
    if getattr(speculator, "acceptance_estimator", None) is not None:
        return "adaptive verification estimates acceptance from draft logits"
    if not hasattr(speculator.model, "get_top_tokens"):
        return f"draft model {type(speculator.model).__name__} has no get_top_tokens"
    return None


def _argmax_current(runner) -> bool:
    return bool(_speculator(runner).use_local_argmax_reduction)


def _argmax_apply(runner, on: bool) -> None:
    speculator = _speculator(runner)
    speculator.use_local_argmax_reduction = on
    speculator._validate_local_argmax_reduction()


_TARGET_ARGMAX_ENV = "VLLM_SM80_SHARD_ARGMAX"


def _target_argmax_unavailable(runner) -> str | None:
    from vllm_sm80.worker import target_argmax

    return target_argmax.refusal(runner)


def _target_argmax_current(runner) -> bool:
    from vllm_sm80.worker import target_argmax

    return target_argmax.enabled(runner)


def _target_argmax_apply(runner, on: bool) -> None:
    """Eager sampling only (no graph holds it); the on arm compiles the slot
    kernels here, on one dummy row, so its first greedy step does not. The
    arm is the switch's target half: the drafts keep the boot value (DSpark
    reads the switch once, before its draft graphs are captured)."""
    from vllm_sm80.worker import target_argmax

    target_argmax.set_enabled(runner, on)
    _set_env(_TARGET_ARGMAX_ENV, on)
    target_argmax.warm(runner)
_QWEN4_MODEL_MODULE = "vllm.models.qwen4_exp.nvidia.model"
# Every GateLinear attribute that set_out_dtype writes.
_ROUTER_GATE_ATTRS = (
    "out_dtype",
    "allow_cublas_router_gemm",
    "allow_rocm_bf16x3_router_gemm",
    "allow_ll_bf16_gemm",
    "allow_ll_fp32w_gemm",
)


def _router_gates(runner) -> list:
    """The router gate of every Qwen4Exp MoE block, target and draft model.

    With the text config field ``moe_router_dtype="float32"``, the block calls
    ``set_out_dtype`` at construction; ``GateLinear.forward`` reads the output
    dtype and the tier flags on every call, so the two arms differ only in
    those fields.
    """
    module = sys.modules.get(_QWEN4_MODEL_MODULE)
    if module is None:
        return []
    speculator = getattr(runner, "speculator", None)
    gates = {}
    for model in (getattr(runner, "model", None), getattr(speculator, "model", None)):
        if model is None:
            continue
        for m in model.modules():
            if isinstance(m, module.Qwen4ExpSparseMoeBlock):
                gates[id(m.gate)] = m.gate
    return list(gates.values())


def _router_unavailable(runner) -> str | None:
    gates = _router_gates(runner)
    if not gates:
        return "the served model has no Qwen4Exp MoE router gates"
    for gate in gates:
        if gate.out_dtype is not None:
            return (
                f"router gate {gate.prefix} already has out_dtype "
                f"{gate.out_dtype}; boot without moe_router_dtype in "
                "--hf-overrides, the on arm sets the FP32 router logits"
            )
        if gate._rocm_bf16x3_weight_eligible:
            return (
                f"router gate {gate.prefix} would need the ROCm BF16x3 weight "
                "split, which is built only at load"
            )
    return None


def _router_current(runner) -> bool:
    import torch

    return any(gate.out_dtype == torch.float32 for gate in _router_gates(runner))


def _router_apply(runner, on: bool) -> None:
    import torch

    gates = _STATE.saved.setdefault(
        "router_fp32",
        [
            (gate, {attr: getattr(gate, attr) for attr in _ROUTER_GATE_ATTRS})
            for gate in _router_gates(runner)
        ],
    )
    for gate, boot in gates:
        for attr, value in boot.items():
            setattr(gate, attr, value)
        if on:
            gate.set_out_dtype(torch.float32)



def _model_modules(runner):
    """Unique modules of the target and the drafter, including shared layers."""
    seen = set()
    for model in (getattr(runner, "model", None),
                  getattr(_speculator(runner), "model", None)):
        if model is not None:
            for module in model.modules():
                if id(module) not in seen:
                    seen.add(id(module))
                    yield module


_ATTN_ENV = "VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE"
_ATTN_FIELDS = ("_sm80_fused_attn", "_sm80_fused_max_rows",
                "_sm80_max_decode_query_len", "_sm80_is_draft_layer")
_ABSENT = object()


def _restore_fields(module, fields):
    for name, value in fields.items():
        if value is _ABSENT:
            if hasattr(module, name):
                delattr(module, name)
        else:
            setattr(module, name, value)


def _attn_unavailable(runner) -> str | None:
    layers = [m for m in _model_modules(runner) if hasattr(m, "_sm80_fused_attn")]
    if not layers:
        return "the served target/drafter has no sm_80 DeepSeek attention layers"
    saved = []
    boot_env = _env_flag(_ATTN_ENV)
    try:
        _set_env(_ATTN_ENV, True)
        for module in layers:
            boot = {a: getattr(module, a, _ABSENT) for a in _ATTN_FIELDS}
            try:
                module._sm80_fused_attn = module._sm80_fused_attn_gate(runner.vllm_config)
                enabled = {a: getattr(module, a, _ABSENT) for a in _ATTN_FIELDS}
                saved.append((module, boot, enabled))
            finally:
                _restore_fields(module, boot)
    finally:
        _set_env(_ATTN_ENV, boot_env)
    if not any(enabled["_sm80_fused_attn"] for _, _, enabled in saved):
        return "no DeepSeek attention layer is eligible for the fused path"
    _STATE.saved["dsv41_attention"] = saved
    return None


def _attn_apply(runner, on: bool) -> None:
    _set_env(_ATTN_ENV, on)
    for module, boot, enabled in _STATE.saved["dsv41_attention"]:
        _restore_fields(module, enabled if on else boot)
        module._sm80_fused_attn = enabled["_sm80_fused_attn"] if on else False


def _pipeline_layers(runner):
    return [m for m in _model_modules(runner)
            if hasattr(m, "_sm80_attn_pipeline_eligible")
            and hasattr(m, "_sm80_attn_pipeline_enabled")]


def _pipeline_current(runner) -> bool:
    return any(m._sm80_attn_pipeline_enabled for m in _pipeline_layers(runner))


def _pipeline_unavailable(runner) -> str | None:
    layers = _pipeline_layers(runner)
    if not any(m._sm80_attn_pipeline_eligible for m in layers):
        return "no loaded DeepSeek attention pipeline implementation is eligible"
    _STATE.saved["dsv41_pipeline"] = (
        _pipeline_current(runner),
        [(m, m._sm80_attn_pipeline_eligible, m._sm80_attn_pipeline_enabled)
         for m in layers],
    )
    return None


def _pipeline_apply(runner, on: bool) -> None:
    _set_env("VLLM_DEEPSEEK_V41_ATTN_PIPELINE", on)
    boot, layers = _STATE.saved["dsv41_pipeline"]
    for module, eligible, active in layers:
        module._sm80_attn_pipeline_enabled = active if on == boot else on and eligible


_SELECT_ENV = "VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH"


def _select_unavailable(runner) -> str | None:
    # select() runs only inside the enabled pipeline (run_indexer): with the
    # pipeline off both arms would launch the same kernels.
    if not any(m._sm80_attn_pipeline_eligible and m._sm80_attn_pipeline_enabled
               for m in _pipeline_layers(runner)):
        return "no loaded DeepSeek attention pipeline is enabled"
    return None


def _select_apply(runner, on: bool) -> None:
    # select() reads the switch per call: an arm's graphs capture its launches,
    # and its eager warmup call compiles the one-launch kernel.
    _set_env(_SELECT_ENV, on)


def _moepf_experts(runner):
    seen = set()
    for module in _model_modules(runner):
        method = getattr(module, "quant_method", None)
        kernel = getattr(method, "moe_kernel", None)
        experts = getattr(kernel, "fused_experts", None)
        if hasattr(experts, "prefill_v2_bits_x2") and id(experts) not in seen:
            seen.add(id(experts))
            yield experts


def _moepf_unavailable(runner) -> str | None:
    experts = list(_moepf_experts(runner))
    if not any(e.prefill_v2_bits_x2 for e in experts):
        return ("boot with VLLM_DEEPSEEK_V41_MOE_PREFILL_TC=1: "
                "eligible EXL3 experts need their load-time workspace and warmup")
    _STATE.saved["dsv41_moepf"] = [(e, e.prefill_v2_bits_x2) for e in experts]
    return None


def _moepf_apply(runner, on: bool) -> None:
    _set_env("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", on)
    for experts, bits in _STATE.saved["dsv41_moepf"]:
        experts.prefill_v2_bits_x2 = bits if on else 0


def _moe_v2_target_experts(runner) -> list:
    """Experts moe_out_v2 can serve, except the DSpark drafter's
    (VLLM_EXL3_MOE_V2_DRAFT has its own switch)."""
    return [
        e
        for e in _moepf_experts(runner)
        if hasattr(e, "moe_v2_rows") and not getattr(e, "moe_v2_draft", False)
    ]


def _moe_v2_unavailable(runner) -> str | None:
    experts = _moe_v2_target_experts(runner)
    if not any(e.moe_v2_rows for e in experts):
        return ("boot with VLLM_EXL3_MOE_V2=1: "
                "eligible experts need their load-time workspace and warmup")
    _STATE.saved["exl3_moe_v2"] = [(e, e.moe_v2_rows) for e in experts]
    return None


def _moe_v2_current(runner) -> bool:
    return any(e.moe_v2_rows for e in _moe_v2_target_experts(runner))


def _moe_v2_apply(runner, on: bool) -> None:
    _set_env("VLLM_EXL3_MOE_V2", on)
    for experts, rows in _STATE.saved["exl3_moe_v2"]:
        experts.moe_v2_rows = rows if on else 0


def _moe_v2_draft_experts(runner) -> list:
    return [e for e in _moepf_experts(runner) if getattr(e, "moe_v2_draft", False)]


def _moe_v2_draft_unavailable(runner) -> str | None:
    experts = _moe_v2_draft_experts(runner)
    if not any(e.moe_v2_rows for e in experts):
        return ("boot with VLLM_EXL3_MOE_V2_DRAFT=1: the drafter's experts "
                "need their load-time workspace and warmup")
    _STATE.saved["exl3_moe_v2_draft"] = [(e, e.moe_v2_rows) for e in experts]
    return None


def _moe_v2_draft_current(runner) -> bool:
    return any(e.moe_v2_rows for e in _moe_v2_draft_experts(runner))


def _moe_v2_draft_apply(runner, on: bool) -> None:
    _set_env("VLLM_EXL3_MOE_V2_DRAFT", on)
    for experts, rows in _STATE.saved["exl3_moe_v2_draft"]:
        experts.moe_v2_rows = rows if on else 0


def _moe_v2_fold_experts(runner) -> list:
    return [
        e
        for e in _moepf_experts(runner)
        if getattr(e, "moe_v2_rows", 0) and not getattr(e, "moe_v2_draft", False)
    ]


def _moe_v2_fold_unavailable(runner) -> str | None:
    from vllm_sm80.kernels.exl3 import ops

    if not _moe_v2_fold_experts(runner):
        return ("boot with VLLM_EXL3_MOE_V2=1: the fold changes how moe_out_v2 "
                "calls run")
    try:
        ops.require_moe_v2_fold()
    except RuntimeError as exc:
        return str(exc)
    return None


def _moe_v2_fold_apply(runner, on: bool) -> None:
    """The folded kernels' first call (SM count and occupancy, moe_v2.cu
    launch_lane) is the arm's eager warmup run, before its graphs are
    captured."""
    _set_env("VLLM_EXL3_MOE_V2_FOLD", on)
    for experts in _moe_v2_fold_experts(runner):
        experts.moe_v2_fold = on


def _splitk_unavailable(runner) -> str | None:
    from vllm_sm80.kernels.exl3 import ops
    if not ops._loaded:
        return "no loaded EXL3 extension"
    import torch
    return ops.dense_splitk_closed_reason(tuple(torch.cuda.get_device_capability()))


def _splitk_apply(runner, on: bool) -> None:
    from vllm_sm80.kernels.exl3 import ops
    _set_env("VLLM_EXL3_DENSE_SPLITK", on)
    ops._dense_splitk = on


def _mhc_bindings(runner):
    for layer in _model_modules(runner):
        bound = getattr(layer, "_mhc_ops", None)
        if bound is not None:
            # Preserve the diagnostic wrapper and its layer ownership.
            holder, attr = (bound, "_ops") if hasattr(bound, "_ops") else (layer, "_mhc_ops")
            yield layer, holder, attr


def _mhc_unavailable(runner) -> str | None:
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused

    bindings = list(_mhc_bindings(runner))
    if not bindings:
        return "the served model has no sm_80 DeepSeek delayed mHC layers"
    modes = set()
    for layer, holder, attr in bindings:
        current = getattr(holder, attr)
        if current is not sm80_mhc and current is not sm80_mhc_fused:
            return "a DeepSeek layer has an unsupported delayed mHC implementation"
        reason = sm80_mhc_fused.supports(layer.hc_mult, layer.hidden_size)
        if reason is not None:
            return reason
        modes.add(current is sm80_mhc_fused)
    if len(modes) != 1:
        return "DeepSeek delayed mHC layers have inconsistent active implementations"
    _STATE.saved["dsv41_mhc"] = bindings
    return None


def _mhc_current(runner) -> bool:
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    return any(getattr(holder, attr) is sm80_mhc_fused
               for _, holder, attr in _mhc_bindings(runner))


def _mhc_apply(runner, on: bool) -> None:
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused

    _set_env("VLLM_DEEPSEEK_V41_FUSED_MHC", on)
    for _, holder, attr in _STATE.saved["dsv41_mhc"]:
        setattr(holder, attr, sm80_mhc_fused if on else sm80_mhc)


def _mhc_finish_unavailable(runner) -> str | None:
    reason = _mhc_unavailable(runner)
    if reason is not None:
        return reason
    if not _mhc_current(runner):
        return "boot with VLLM_DEEPSEEK_V41_FUSED_MHC=1 for the fused finish"
    return None


def _mhc_finish_current(runner) -> bool:
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    return sm80_mhc_fused.FINISH_KERNEL


def _mhc_finish_apply(runner, on: bool) -> None:
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    _set_env("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH", on)
    sm80_mhc_fused.FINISH_KERNEL = on


def _mhc_finish_v2_unavailable(runner) -> str | None:
    reason = _mhc_finish_unavailable(runner)
    if reason is not None:
        return reason
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    for layer, _, _ in _STATE.saved["dsv41_mhc"]:
        reason = sm80_mhc_fused.finish_v2_unavailable(
            layer.hc_mult, layer.hidden_size, layer.hc_sinkhorn_iters
        )
        if reason is not None:
            return reason
    return None


def _mhc_finish_v2_current(runner) -> bool:
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused

    return sm80_mhc_fused.FINISH_V2


def _mhc_finish_v2_apply(runner, on: bool) -> None:
    """The on arm compiles its kernels here, before its graphs are captured."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused
    from vllm_sm80.models.deepseek_v41.ops.sm80_mhc_finish_v2 import FINISH_V2_WARMUP

    _set_env("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2", on)
    sm80_mhc_fused.FINISH_V2 = on
    if on:
        FINISH_V2_WARMUP.warmup(vllm_config=runner.vllm_config)


def _route_routers(runner):
    from vllm_sm80.models.deepseek_v41.sm80_moe import Sm80DeepseekV41Router

    seen = set()
    for module in _model_modules(runner):
        router = getattr(module, "router", None)
        if isinstance(router, Sm80DeepseekV41Router) and id(router) not in seen:
            seen.add(id(router))
            yield router


def _route_unavailable(runner) -> str | None:
    from vllm_sm80.models.deepseek_v41.ops import sm80_route

    routers = list(_route_routers(runner))
    if not routers:
        return "the served target/drafter has no sm_80 DeepSeek-V4.1 MoE routers"
    for router in routers:
        reason = sm80_route.supports(
            router.global_num_experts, router.top_k, router.gate_temp
        )
        if reason is not None:
            return reason
    return None


def _route_apply(runner, on: bool) -> None:
    _set_env("VLLM_DEEPSEEK_V41_FUSED_ROUTE", on)
    for router in _route_routers(runner):
        router.fused_route = on


_DRAFT_PROLOGUE_ENV = "VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED"


def _draft_prologue_layers(runner) -> list:
    """The drafter's sm_80 DeepSeek-V4.1 attention layers whose context-KV
    prologue may run as one kernel."""
    model = getattr(_speculator(runner), "model", None)
    if model is None:
        return []
    return [m for m in model.modules()
            if getattr(m, "_sm80_draft_prologue_eligible", False)]


def _draft_prologue_unavailable(runner) -> str | None:
    if not _draft_prologue_layers(runner):
        return "the served drafter has no eligible sm_80 DeepSeek-V4.1 attention layers"
    return None


def _draft_prologue_apply(runner, on: bool) -> None:
    _set_env(_DRAFT_PROLOGUE_ENV, on)
    for layer in _draft_prologue_layers(runner):
        layer._sm80_draft_prologue = on


def _gemv_gates(runner) -> list:
    """Every adopted DeepSeek-V4.1 MoE gate of the target and the drafter."""
    from vllm_sm80.models.deepseek_v41.sm80_moe import Sm80DeepseekV41Gate

    return [m for m in _model_modules(runner) if isinstance(m, Sm80DeepseekV41Gate)]


def _gate_gemv_unavailable(runner) -> str | None:
    from vllm_sm80.models.deepseek_v41.ops import sm80_gate_gemv

    gates = _gemv_gates(runner)
    if not gates:
        return "the served target/drafter has no sm_80 DeepSeek-V4.1 MoE gates"
    for gate in gates:
        reason = sm80_gate_gemv.supports(*gate.weight.shape)
        if reason is not None:
            return reason
    return None


def _gate_gemv_apply(runner, on: bool) -> None:
    """The kernel compiles and its scratch is allocated by the arm's eager
    warmup call, before its graphs are captured."""
    _set_env("VLLM_DEEPSEEK_V41_GATE_GEMV", on)
    for gate in _gemv_gates(runner):
        gate.gemv = on


_SWA_BATCH_ENV = "VLLM_DEEPSEEK_V41_SWA_META_BATCH"


def _swa_batch_unavailable(runner) -> str | None:
    from vllm_sm80.models.deepseek_v41.sm80_attention import Sm80SWAMetadataBuilder

    for groups in getattr(runner, "attn_groups", None) or ():
        for group in groups:
            for builder in group.metadata_builders:
                if isinstance(builder, Sm80SWAMetadataBuilder):
                    return None
    return "the served model has no sm_80 DeepSeek-V4.1 sliding-window builders"


def _swa_batch_apply(runner, on: bool) -> None:
    from vllm_sm80.models.deepseek_v41 import sm80_swa_meta_batch

    # Eager step preparation only: graphs read the builders' buffers, which
    # both arms fill with the same bytes, so one capture serves both.
    _set_env(_SWA_BATCH_ENV, on)
    sm80_swa_meta_batch.reset()


def _dense_prefill_unavailable(runner) -> str | None:
    from vllm_sm80.kernels.exl3 import dense_prefill_tc, ops

    if not ops._loaded or not dense_prefill_tc.active:
        return ("boot with VLLM_EXL3_DENSE_PREFILL_TC=1: "
                "the dense prefill gate and kernel warmup must already be complete")
    return None


def _dense_prefill_current(runner) -> bool:
    from vllm_sm80.kernels.exl3 import dense_prefill_tc

    return dense_prefill_tc.active


def _dense_prefill_apply(runner, on: bool) -> None:
    from vllm_sm80.kernels.exl3 import dense_prefill_tc

    _set_env("VLLM_EXL3_DENSE_PREFILL_TC", on)
    dense_prefill_tc.active = on


def _wo_a_methods(runner):
    """The grouped wo_a methods of the target's and the drafter's sm_80
    DeepSeek-V4.1 attention layers."""
    from vllm_sm80.models.deepseek_v41.sm80_attention import (
        DeepseekV41Sm80Attention,
    )
    from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

    seen = set()
    for module in _model_modules(runner):
        if not isinstance(module, DeepseekV41Sm80Attention):
            continue
        method = getattr(getattr(module, "wo_a", None), "quant_method", None)
        if isinstance(method, GroupedLinearMethod) and id(method) not in seen:
            seen.add(id(method))
            yield method


def _wo_a_gemm_methods(runner):
    """The grouped wo_a methods that hold the grouped GEMM's codes."""
    from vllm_sm80.models.deepseek_v41.ops.sm80_wo_a_grouped import (
        WoAGroupedGemmMethod,
    )

    for method in _wo_a_methods(runner):
        if isinstance(method, WoAGroupedGemmMethod):
            yield method


def _wo_a_gemm_unavailable(runner) -> str | None:
    methods = list(_wo_a_gemm_methods(runner))
    if not methods or not all(m.keep_groups for m in methods):
        return ("boot with VLLM_DEV_AB_SWITCH=dsv41_wo_a_grouped_gemm on an MXFP8 "
                "wo_a: every grouped wo_a needs its codes and its per-group Marlin "
                "layers")
    return None


def _wo_a_gemm_apply(runner, on: bool) -> None:
    # apply_grouped reads the arm per call: each arm's graphs capture its
    # launches (the grouped GEMM or the per-group Marlin GEMMs).
    _set_env("VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", on)
    for method in _wo_a_gemm_methods(runner):
        method.gemm = on


_WKV_REPLICA_ENV = "VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL"


def _wkv_replicas(runner):
    """The served Engram wkv layers that hold a prefill replica."""
    from vllm_sm80.models.deepseek_v41.sm80_engram_wkv import ReplicatedPrefillWkv

    return [m for m in _model_modules(runner) if isinstance(m, ReplicatedPrefillWkv)]


def _wkv_replica_unavailable(runner) -> str | None:
    if not _wkv_replicas(runner):
        return (
            f"boot with {_WKV_REPLICA_ENV}=1 so each Engram wkv holds its "
            "replica; its off arm keeps the sharded path"
        )
    return None


def _wkv_replica_apply(runner, on: bool) -> None:
    layers = _STATE.saved.setdefault(
        "dsv41_engram_wkv_replicated_prefill",
        [(m, m.min_rows) for m in _wkv_replicas(runner)],
    )
    _set_env(_WKV_REPLICA_ENV, on)
    for module, boot in layers:
        module.min_rows = boot if on else None


SWITCHES: dict[str, Switch] = {
    s.name: s
    for s in (
        Switch(
            "dsv41_fused_mhc", "VLLM_DEEPSEEK_V41_FUSED_MHC", True, False,
            _mhc_unavailable, _mhc_current, _mhc_apply,
        ),
        Switch(
            "dsv41_fused_mhc_finish", "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH", True, False,
            _mhc_finish_unavailable, _mhc_finish_current, _mhc_finish_apply,
        ),
        Switch(
            "dsv41_fused_mhc_finish_v2", "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2", True, False,
            _mhc_finish_v2_unavailable, _mhc_finish_v2_current, _mhc_finish_v2_apply,
        ),
        Switch(
            "dsv41_fused_route", "VLLM_DEEPSEEK_V41_FUSED_ROUTE", True, True,
            _route_unavailable,
            lambda runner: any(r.fused_route for r in _route_routers(runner)),
            _route_apply,
        ),
        Switch(
            "dsv41_draft_prologue_fused", _DRAFT_PROLOGUE_ENV, True, True,
            _draft_prologue_unavailable,
            lambda runner: any(m._sm80_draft_prologue for m in _draft_prologue_layers(runner)),
            _draft_prologue_apply,
        ),
        Switch(
            "dsv41_gate_gemv", "VLLM_DEEPSEEK_V41_GATE_GEMV", True, False,
            _gate_gemv_unavailable,
            lambda runner: any(g.gemv for g in _gemv_gates(runner)), _gate_gemv_apply,
        ),
        Switch(
            "dsv41_wo_a_grouped_gemm", "VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", True, False,
            _wo_a_gemm_unavailable,
            lambda runner: any(m.gemm for m in _wo_a_gemm_methods(runner)),
            _wo_a_gemm_apply,
        ),
        Switch(
            "dsv41_engram_wkv_replicated_prefill", _WKV_REPLICA_ENV, False, True,
            _wkv_replica_unavailable,
            lambda runner: any(m.min_rows is not None for m in _wkv_replicas(runner)),
            _wkv_replica_apply,
        ),
        Switch(
            "dsv41_swa_meta_batch", _SWA_BATCH_ENV, False, True,
            _swa_batch_unavailable, lambda runner: _env_flag(_SWA_BATCH_ENV),
            _swa_batch_apply,
        ),
        Switch(
            "exl3_dense_prefill_tc", "VLLM_EXL3_DENSE_PREFILL_TC", True, False,
            _dense_prefill_unavailable, _dense_prefill_current, _dense_prefill_apply,
        ),
        Switch(
            "dsv41_attn_pipeline", "VLLM_DEEPSEEK_V41_ATTN_PIPELINE", True, False,
            _pipeline_unavailable, _pipeline_current, _pipeline_apply,
        ),
        Switch(
            "dsv41_select_one_launch", _SELECT_ENV, True, True,
            _select_unavailable, lambda runner: _env_flag(_SELECT_ENV), _select_apply,
        ),
        Switch(
            "dsv41_fused_attn_decode", "VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", True, False,
            _attn_unavailable, lambda runner: _env_flag(_ATTN_ENV), _attn_apply,
        ),
        Switch(
            "dsv41_moe_prefill_tc", "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", True, False,
            _moepf_unavailable,
            lambda runner: any(e.prefill_v2_bits_x2 for e in _moepf_experts(runner)),
            _moepf_apply,
        ),
        Switch(
            "exl3_moe_v2", "VLLM_EXL3_MOE_V2", True, False,
            _moe_v2_unavailable, _moe_v2_current, _moe_v2_apply,
        ),
        Switch(
            "exl3_moe_v2_draft", "VLLM_EXL3_MOE_V2_DRAFT", True, False,
            _moe_v2_draft_unavailable, _moe_v2_draft_current, _moe_v2_draft_apply,
        ),
        Switch(
            "exl3_moe_v2_fold", "VLLM_EXL3_MOE_V2_FOLD", True, True,
            _moe_v2_fold_unavailable,
            lambda runner: any(e.moe_v2_fold for e in _moe_v2_fold_experts(runner)),
            _moe_v2_fold_apply,
        ),
        Switch(
            "exl3_dense_splitk", "VLLM_EXL3_DENSE_SPLITK", True, False,
            _splitk_unavailable,
            lambda runner: _env_flag("VLLM_EXL3_DENSE_SPLITK"), _splitk_apply,
        ),
        Switch(
            "topk_canonical_ties", _TOPK_ENV, True, False,
            _topk_unavailable,
            lambda runner: _env_flag(_TOPK_ENV), _topk_apply,
        ),
        Switch(
            "marlin_decode_v2", "VLLM_AMPERE_MARLIN_DECODE_V2", True, True,
            _marlin_unavailable,
            lambda runner: _env_flag("VLLM_AMPERE_MARLIN_DECODE_V2"), _marlin_apply,
        ),
        Switch(
            "custom_allreduce_flags", "VLLM_SM80_ALLREDUCE", True, False,
            _flags_unavailable, _flags_current, _flags_apply,
        ),
        Switch(
            "kda_prefill", "VLLM_SM80_KDA_PREFILL",
            False, False,
            _kda_unavailable, _kda_current, _kda_apply,
        ),
        Switch(
            "local_argmax_reduction",
            "speculative_config.use_local_argmax_reduction",
            True, True,
            _argmax_unavailable, _argmax_current, _argmax_apply,
        ),
        Switch(
            "target_shard_argmax", _TARGET_ARGMAX_ENV, False, True,
            _target_argmax_unavailable, _target_argmax_current,
            _target_argmax_apply,
        ),
        Switch(
            "qwen4_router_fp32", "hf_overrides.text_config.moe_router_dtype",
            True, False,
            _router_unavailable, _router_current, _router_apply,
        ),
    )
}


# ---------------------------------------------------------------------------
# Configuration (every process that builds a VllmConfig)
# ---------------------------------------------------------------------------

def configure(vllm_config) -> None:
    """Inject :class:`DevABWorkerExtension` when VLLM_DEV_AB_SWITCH or
    VLLM_HOT_SWAP (:mod:`vllm_sm80.hot_swap`) is set."""
    name = envs.VLLM_DEV_AB_SWITCH
    hot = hot_swap.configure(vllm_config)
    if not name and not hot:
        return
    what = f"VLLM_HOT_SWAP={envs.VLLM_HOT_SWAP}" if hot else f"VLLM_DEV_AB_SWITCH={name}"
    if not hot:
        if envs.VLLM_SM80_CAPTURE:
            raise ValueError(
                "VLLM_DEV_AB_SWITCH cannot be combined with VLLM_SM80_CAPTURE"
            )
        if name not in SWITCHES:
            raise ValueError(
                f"VLLM_DEV_AB_SWITCH={name!r} is not one of: "
                + ", ".join(sorted(SWITCHES))
            )
    parallel_config = vllm_config.parallel_config
    if parallel_config.worker_extension_cls not in ("", EXTENSION_QUALNAME):
        raise ValueError(
            f"{what} needs worker_extension_cls for its own setter; "
            f"{parallel_config.worker_extension_cls!r} is already set"
        )
    parallel_config.worker_extension_cls = EXTENSION_QUALNAME
    if envs.VLLM_SERVER_DEV_MODE and not hot:
        from vllm_sm80.dev_ab_batch import configure as configure_ordered_batches
        configure_ordered_batches(vllm_config)
    if not envs.VLLM_SERVER_DEV_MODE:
        logger.warning_once(
            "%s without VLLM_SERVER_DEV_MODE=1: "
            "POST /collective_rpc is off, so the arm cannot be changed.", what
        )
    logger.warning_once(
        "%s: development server, both arms captured; not for serving.", what
    )


# ---------------------------------------------------------------------------
# Graph stores keyed by arm
# ---------------------------------------------------------------------------


class ArmGraphStore(MutableMapping):
    """A graph store (descriptor -> entry) with one dict per arm. Every access
    goes to the active arm's dict, so a dispatcher keyed by descriptor is
    keyed by (descriptor, arm). ``clear`` empties every arm, as releasing
    graphs must."""

    __slots__ = ("by_arm",)

    def __init__(self, by_arm: dict[bool, dict]):
        self.by_arm = by_arm

    def _active(self) -> dict:
        return self.by_arm.setdefault(_STATE.arm, {})

    def __getitem__(self, key):
        return self._active()[key]

    def __setitem__(self, key, value) -> None:
        self._active()[key] = value

    def __delitem__(self, key) -> None:
        del self._active()[key]

    def __contains__(self, key) -> bool:
        return key in self._active()

    def __iter__(self) -> Iterator:
        return iter(self._active())

    def __len__(self) -> int:
        return len(self._active())

    def get(self, key, default=None):
        return self._active().get(key, default)

    def clear(self) -> None:
        for entries in self.by_arm.values():
            entries.clear()


def _graph_stores() -> list[tuple[Any, str]]:
    """(owner, attribute) of every CUDA graph store in this process."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    stores = [(w, "entries") for w in list(BreakableCUDAGraphWrapper._all_instances)]
    stores += [
        (w, "concrete_cudagraph_entries")
        for w in list(CUDAGraphWrapper._all_instances)
    ]
    stores += [(m, "graphs") for m in gc.get_objects() if isinstance(m, CudaGraphManager)]
    return stores


def _key_stores(arm: bool) -> None:
    """Key every store still plain by ``arm``, the arm its graphs belong to."""
    for owner, attr in _graph_stores():
        entries = getattr(owner, attr)
        if not isinstance(entries, ArmGraphStore):
            setattr(owner, attr, ArmGraphStore({arm: entries}))


def _check_complete(arms: tuple[bool, ...]) -> None:
    """Every store holds the same descriptors under every arm."""
    uneven = []
    for owner, attr in _graph_stores():
        store = getattr(owner, attr)
        keys = [set(store.by_arm.get(arm, ())) for arm in arms]
        if any(k != keys[0] for k in keys[1:]):
            uneven.append(
                f"{type(owner).__name__}.{attr} "
                + "/".join(f"{_arm_name(a)} {len(k)}" for a, k in zip(arms, keys))
            )
    if uneven:
        raise RuntimeError(
            "dev A/B: CUDA graphs differ between arms: " + "; ".join(uneven)
        )


# ---------------------------------------------------------------------------
# Capture (Model Runner V2) and the per-process arm
# ---------------------------------------------------------------------------


def _resolve(runner) -> Switch:
    """The switch, checked against this server; records the boot arm."""
    if _STATE.switch is None:
        name = envs.VLLM_DEV_AB_SWITCH
        if name not in SWITCHES:
            raise ValueError(f"VLLM_DEV_AB_SWITCH={name!r} is not allowlisted")
        switch = SWITCHES[name]
        why = switch.unavailable(runner)
        if why is not None:
            raise RuntimeError(f"VLLM_DEV_AB_SWITCH={name}: {why}")
        _STATE.switch = switch
        _STATE.boot = _STATE.arm = bool(switch.current(runner))
        logger.info(
            "Dev A/B %s (%s): boot arm %s", name, switch.env, _arm_name(_STATE.boot)
        )
    return _STATE.switch


def _synchronize() -> None:
    import torch

    torch.accelerator.synchronize()


def _apply(runner, arm: bool) -> None:
    _synchronize()
    _STATE.switch.apply(runner, arm)
    _STATE.arm = arm
    _synchronize()



def _capture_base(runner) -> int:
    # Re-enter the hooked body, not a subclass's profiling wrapper: nesting a
    # memory window would lose the first arm's persistent-buffer allocation.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    return GPUModelRunner.capture_model(runner, profile_only=True)


def _lock_workspace() -> None:
    from vllm.v1.worker.workspace import lock_workspace
    lock_workspace()


def _dynamic_schedule(runner) -> tuple[int, ...]:
    spec = runner.vllm_config.speculative_config
    if spec is None or not spec.uses_dynamic_speculative_decoding():
        return ()
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
    return tuple(build_dynamic_sd_schedule_lookup(
        spec.num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=runner.max_num_reqs,
        vllm_num_speculative_tokens=runner.vllm_config.num_speculative_tokens,
    )[1:])


@dataclass
class _ManagerPlan:
    manager: Any
    descs: dict = field(init=False)
    candidates: dict = field(init=False)
    original_descs: dict = field(init=False)
    original_candidates: dict = field(init=False)
    decode_query_len: int = field(init=False)
    cudagraph_mode: Any = field(init=False)

    def __post_init__(self):
        self.original_descs = self.manager._capture_descs
        self.original_candidates = self.manager._candidates
        self.descs = {k: tuple(v) for k, v in self.original_descs.items()}
        self.candidates = {k: tuple(v) for k, v in self.original_candidates.items()}
        self.decode_query_len = self.manager.decode_query_len
        self.cudagraph_mode = self.manager.cudagraph_mode

    def unchanged(self) -> bool:
        m = self.manager
        return (m._capture_descs is self.original_descs
                and m._candidates is self.original_candidates
                and {k: tuple(v) for k, v in m._capture_descs.items()} == self.descs
                and {k: tuple(v) for k, v in m._candidates.items()} == self.candidates
                and m.decode_query_len == self.decode_query_len
                and m.cudagraph_mode == self.cudagraph_mode)

    def restore(self) -> None:
        if self.unchanged():
            return
        self.original_descs.clear()
        self.original_descs.update({k: list(v) for k, v in self.descs.items()})
        self.original_candidates.clear()
        self.original_candidates.update({k: list(v) for k, v in self.candidates.items()})
        self.manager._capture_descs = self.original_descs
        self.manager._candidates = self.original_candidates
        self.manager.decode_query_len = self.decode_query_len
        self.manager.cudagraph_mode = self.cudagraph_mode

    def check_graphs(self, arms) -> None:
        from vllm.config import CUDAGraphMode
        m = self.manager
        full = self.descs.get(CUDAGraphMode.FULL, ())
        limit = m._max_full_descs_to_capture
        expected = set(full if limit is None else full[:limit])
        for arm in arms:
            if set(m.graphs.by_arm.get(arm, ())) != expected:
                raise RuntimeError(f"dev A/B: incomplete {type(m).__name__} FULL graphs")
        # Decoder replay uses this wrapper, not manager.graphs (which is empty
        # for PIECEWISE). Check every configured size, not just arm equality.
        wrapper = m.breakable_cg_runner
        if m.use_breakable_cg and self.descs.get(CUDAGraphMode.PIECEWISE):
            expected_sizes = {d.num_tokens for d in self.descs[CUDAGraphMode.PIECEWISE]}
            if wrapper is None or not isinstance(wrapper.entries, ArmGraphStore):
                raise RuntimeError("dev A/B: missing breakable graph store")
            for arm in arms:
                sizes = {d.num_tokens for d in wrapper.entries.by_arm.get(arm, ())}
                if not expected_sizes <= sizes:
                    raise RuntimeError("dev A/B: incomplete decoder/breakable replay sizes")


def _manager_plans() -> list[_ManagerPlan]:
    return [_ManagerPlan(owner) for owner, attr in _graph_stores() if attr == "graphs"]


def _capture_policy(runner):
    spec = runner.vllm_config.speculative_config
    draft = _speculator(runner)
    config = getattr(runner.vllm_config, "compilation_config", None)
    return (
        _dynamic_schedule(runner),
        tuple(getattr(config, "cudagraph_capture_sizes", ()) or ()),
        tuple(getattr(config, "decoder_replay_cudagraph_capture_sizes", ()) or ()),
        getattr(spec, "num_speculative_tokens", None),
        tuple(getattr(draft, "widths", ())),
        tuple((width, id(m)) for width, m in getattr(draft, "_cg_managers", {}).items()),
    )


def _check_policy(runner, policy, plans) -> None:
    if _capture_policy(runner) != policy or any(not p.unchanged() for p in plans):
        raise RuntimeError("dev A/B: capture policy or cached descriptors re-specialized")


def _dynamic_coverage(runner) -> list[dict]:
    """Record real dispatch for every table row, without changing draft policy.

    A block drafter may deliberately generate its maximum width and truncate
    to K. Do not enable new widths here: that changes its attention semantics.
    Both that actual draft shape and the target's (batch, K+1) must be captured.
    """
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import _is_compatible

    config = getattr(runner.vllm_config, "compilation_config", None)
    mode = getattr(config, "cudagraph_mode", CUDAGraphMode.NONE)
    full_decode = mode.decode_mode() == CUDAGraphMode.FULL

    def descriptor(manager, batch, query_len):
        if manager.cudagraph_mode == CUDAGraphMode.NONE:
            if full_decode:
                raise RuntimeError("dev A/B: FULL decode requires target and draft graphs, not eager fallback")
            return "eager"
        candidates = manager._candidates.get((batch * query_len, 0), ())
        for desc in candidates:
            if _is_compatible(desc, batch, batch * query_len, query_len, 0, None, 1):
                if full_decode and desc.cg_mode != CUDAGraphMode.FULL:
                    raise RuntimeError("dev A/B: FULL decode dispatched a non-FULL target or draft descriptor")
                return repr(desc)
        raise RuntimeError(f"dev A/B: missing batch={batch}, query_len={query_len} capture")

    coverage = []
    draft = _speculator(runner)
    for batch, k in enumerate(_dynamic_schedule(runner), 1):
        target = descriptor(runner.cudagraph_manager, batch, k + 1)
        width = k if k in draft._cg_managers else draft.max_width
        manager = draft._cg_managers[width]
        draft_desc = (descriptor(manager, batch, manager.decode_query_len)
                      if k else "context-only")
        coverage.append({"batch": batch, "k": k, "target": target,
                         "target_store": "target", "draft_store": f"draft:{width}" if k else None,
                         "draft_width": width if k else 0, "draft": draft_desc})
    return coverage


def _graph_inventory(runner) -> list[dict]:
    from vllm.config import CUDAGraphMode

    # A descriptor in another manager is not evidence for this dispatch path.
    # Stable roles link each K-table row to its exact target/draft graph store.
    roles = {id(getattr(runner, "cudagraph_manager", None)): "target"}
    roles.update({id(m): f"draft:{width}"
                  for width, m in getattr(_speculator(runner), "_cg_managers", {}).items()})
    inventory = []
    for owner, attr in _graph_stores():
        entry = {"owner": type(owner).__name__, "store": attr,
                 "role": roles.get(id(owner)) if attr == "graphs" else None,
                 "arms": {_arm_name(a): sorted(map(repr, entries))
                          for a, entries in getattr(owner, attr).by_arm.items()}}
        if attr == "graphs":
            entry["expected_full"] = sorted(map(repr, owner._capture_descs.get(CUDAGraphMode.FULL, ())))
        inventory.append(entry)
    return inventory


def _capture_unsupported(runner) -> str | None:
    model_state = runner.model_state
    if model_state.supports_mm_inputs and model_state.encoder_runner.has_cudagraph():
        return "encoder CUDA graphs are not keyed by arm"
    spec = runner.vllm_config.speculative_config
    if getattr(spec, "adaptive_k_config", None) is not None:
        return "acceptance/load-adaptive speculation is not covered"
    if getattr(runner, "adaptive_verification", None) is not None:
        return "adaptive verification profiles step costs during capture"
    draft = _speculator(runner)
    if getattr(draft, "enable_adaptive_verification", False):
        return "adaptive verification changes drafter state"
    if spec is not None and spec.uses_dynamic_speculative_decoding():
        if getattr(spec, "method", None) != "dspark" or not hasattr(draft, "_cg_managers"):
            return "only the static DSpark per-batch K table is covered"
    return None


def with_ab_capture(capture):
    """Keep the ordinary capture body intact behind a default-neutral wrapper."""
    @functools.wraps(capture)
    def wrapped(runner, *, profile_only=False):
        if hot_swap.active():
            return hot_swap.capture(runner, capture, profile_only)
        per_arm = capture_model_per_arm(runner, profile_only)
        if per_arm is not None:
            return per_arm
        return capture(runner, profile_only=profile_only)

    return wrapped


def capture_model_per_arm(runner, profile_only: bool) -> int | None:
    """Capture the union of both arms, preserving the qualified K policy."""
    if not envs.VLLM_DEV_AB_SWITCH or _STATE.in_capture:
        return None
    switch = _resolve(runner)
    why = _capture_unsupported(runner)
    if why is not None:
        raise RuntimeError(f"VLLM_DEV_AB_SWITCH={switch.name}: {why}")
    arms = (_STATE.boot, not _STATE.boot)
    sizes: dict[str, int] = {}
    manager = getattr(runner, "cudagraph_manager", None)
    samples = getattr(manager, "_capture_mem_samples", None)
    plans = _manager_plans()
    policy = _capture_policy(runner)
    coverage = _dynamic_coverage(runner)
    draft = _speculator(runner)
    width = getattr(draft, "width", None)
    _STATE.captured = ()
    _STATE.in_capture = True
    try:
        for index, arm in enumerate(arms):
            _apply(runner, arm)
            _check_policy(runner, policy, plans)
            first_sample = len(samples) if samples is not None else 0
            # Never lock or nest a subclass's profiling window mid-capture.
            size = int(_capture_base(runner))
            if samples is not None and index > 0:
                size += _full_graph_extra(manager, samples, first_sample)
            sizes[_arm_name(arm)] = size
            _key_stores(arm)
            _check_policy(runner, policy, plans)
        _check_complete(arms)
        for plan in plans:
            plan.check_graphs(arms)
    finally:
        for plan in plans:
            plan.restore()
        if width is not None:
            draft.set_width(width)
        try:
            _apply(runner, _STATE.boot)
        finally:
            _STATE.in_capture = False
    if not profile_only:
        _lock_workspace()
        _STATE.captured = arms
        _STATE.saved["capture_contract"] = (policy, plans)
        _STATE.saved["dynamic_coverage"] = coverage
        _STATE.saved["graph_inventory"] = _graph_inventory(runner)
    phase = "profiling" if profile_only else "serving"
    _STATE.capture_bytes[phase] = sizes
    logger.info(
        "Dev A/B %s: %s CUDA graphs captured per arm: %s",
        switch.name, phase,
        ", ".join(f"{a} {b / (1 << 30):.3f} GiB" for a, b in sizes.items()),
    )
    return sum(sizes.values())


def _full_graph_extra(manager, samples: list[int], first: int) -> int:
    """Memory profiling samples only the largest FULL graphs and extrapolates
    their total from the samples it is given, which are the boot arm's. Take a
    later arm's samples out of that list and return what its own
    extrapolation adds to its measured size."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import _extrapolate_full_graph_memory

    arm_samples = samples[first:]
    del samples[first:]
    if not arm_samples:
        return 0
    total = len(manager._capture_descs.get(CUDAGraphMode.FULL, []))
    return _extrapolate_full_graph_memory(arm_samples, total) - sum(arm_samples)


def _resident_requests(runner) -> int:
    states = getattr(runner, "req_states", None)
    if states is not None:
        return len(states.req_id_to_index)
    return len(getattr(runner, "requests", ()))


def _graphs_on(runner) -> bool:
    from vllm.config import CUDAGraphMode

    return runner.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE


def _all_ranks_agree(ok: bool) -> bool:
    import torch
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_world_group

    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_world_group().cpu_group)
    return bool(flag.item())


def _fail_stop(reason: str) -> NoReturn:
    # WorkerProc catches Exception and resumes serving. An unproven rollback or
    # broken control collective must instead terminate the worker. Every rank
    # reaches this path after a failed rollback vote; transport failures also
    # terminate locally, letting the executor's worker-death handling stop peers.
    try:
        logger.critical("Dev A/B fail-stop: %s", reason)
    finally:
        os._exit(1)


def _transaction_vote(ok: bool) -> bool:
    try:
        return _all_ranks_agree(ok)
    except Exception as error:
        _fail_stop(f"control collective failed: {error}")


def _all_ranks_same_state() -> bool:
    try:
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_world_group

        group = get_world_group().cpu_group
        state = (_STATE.arm, _STATE.generation, _STATE.replay)
        states = [None] * dist.get_world_size(group)
        dist.all_gather_object(states, state, group=group)
        return all(other == state for other in states)
    except Exception as error:
        _fail_stop(f"state acknowledgement failed: {error}")


def _report(worker, ok: bool, reason: str) -> dict[str, Any]:
    from vllm_sm80.dev_ab_batch import capability
    switch = _STATE.switch
    return {
        "ok": ok,
        "reason": reason,
        "rank": worker.rank,
        "switch": switch.name,
        "env": switch.env,
        "output_identical": switch.output_identical,
        "arm": _arm_name(_STATE.arm),
        "boot_arm": _arm_name(_STATE.boot),
        "generation": _STATE.generation,
        "replay": _STATE.replay,
        "captured_arms": [_arm_name(a) for a in _STATE.captured],
        "graph_bytes": _STATE.capture_bytes,
        "dynamic_coverage": _STATE.saved.get("dynamic_coverage", []),
        "graph_inventory": _STATE.saved.get("graph_inventory", []),
        "ordered_batch_protocol": capability(worker.model_runner),
    }


def _set_arm(worker, arm: str, generation: int, replay: bool = True) -> dict[str, Any]:
    runner = worker.model_runner
    _resolve(runner)
    # /collective_rpc passes JSON values; a decimal string is a generation too.
    if isinstance(generation, str) and generation.isdecimal():
        generation = int(generation)
    reason = None
    if type(replay) is not bool:
        reason = "replay must be a boolean"
    elif arm not in ARMS:
        reason = f"arm {arm!r} is not one of {ARMS}"
    elif type(generation) is not int or generation <= _STATE.generation:
        reason = f"generation {generation!r} is not newer than {_STATE.generation}"
    elif "batch_trace" in _STATE.saved:
        reason = "end the ordered batch before changing arm or replay"
    else:
        resident = _resident_requests(runner)
        if resident:
            reason = f"{resident} requests resident; drain first"
        elif _graphs_on(runner) and (arm == "on") not in _STATE.captured:
            reason = "CUDA graphs were not captured for this arm"
    previous = (_STATE.arm, _STATE.generation, _STATE.replay)
    graph_flags = []
    if reason is None and "capture_contract" in _STATE.saved:
        policy, plans = _STATE.saved["capture_contract"]
        try:
            _check_policy(runner, policy, plans)
            _check_complete(_STATE.captured)
            for plan in plans:
                plan.check_graphs(_STATE.captured)
            graph_flags = [(p.manager, p.manager._graphs_captured) for p in plans]
        except Exception as error:
            reason = str(error)
    # Readiness is not a commit: adapters and device synchronizations can fail
    # locally after this vote. No rank may return to serving until all commit,
    # or all restore the prior arm, replay flags and generation.
    agreed = _transaction_vote(reason is None)
    if reason is None and not agreed:
        reason = "another rank refused"
    if reason is not None:
        return _report(worker, False, reason)
    if not _all_ranks_same_state():
        _fail_stop("rank states disagree before switching")
    failure = None
    try:
        _apply(runner, arm == "on")
        for manager, _ in graph_flags:
            manager._graphs_captured = replay
        _STATE.replay = replay
        _STATE.generation = generation
    except Exception as error:
        failure = str(error)
    committed = _transaction_vote(failure is None)
    if committed:
        committed = _all_ranks_same_state()
    if not committed:
        restored = False
        try:
            _apply(runner, previous[0])
            for manager, value in graph_flags:
                manager._graphs_captured = value
            _STATE.arm, _STATE.generation, _STATE.replay = previous
            restored = True
        except Exception as error:
            logger.error("Dev A/B rollback failed: %s", error)
        if not _transaction_vote(restored) or not _all_ranks_same_state():
            _fail_stop("cannot acknowledge a consistent rollback on every rank")
        return _report(worker, False, "switch rolled back: " + (failure or "another rank failed to commit"))
    logger.info(
        "Dev A/B %s: arm %s (generation %d)", _STATE.switch.name, arm, generation
    )
    return _report(worker, True, "")


class DevABWorkerExtension:
    """Worker methods for ``POST /collective_rpc``; injected only with
    VLLM_DEV_AB_SWITCH or VLLM_HOT_SWAP set."""

    def set_ab_arm(self, arm: str, generation: int, replay: bool = True) -> dict[str, Any]:
        """Switch while drained; replay=False supplies a same-arm eager control."""
        return _set_arm(self, arm, generation, replay)

    def get_ab_arm(self) -> dict[str, Any]:
        """This worker's switch state."""
        _resolve(self.model_runner)
        return _report(self, True, "")

    def set_hot_arms(self, on: list[str], generation: int) -> dict[str, Any]:
        """Run the candidates of the VLLM_HOT_SWAP pairs in ``on``, the
        incumbents of the rest; drained, on every rank or on none."""
        return hot_swap.set_arms(self, on, generation)

    def get_hot_arms(self) -> dict[str, Any]:
        """This worker's hot pairs and the candidates running."""
        return hot_swap.get_arms(self)


    def begin_ab_batch(self, manifest: dict, reference: str, establish: bool) -> dict:
        """Begin exclusive ordered input and execution-shape attestation."""
        from vllm_sm80.dev_ab_batch import begin
        return begin(self, manifest, reference, establish)

    def end_ab_batch(self) -> dict:
        """Remove instrumentation and return actual per-step batch evidence."""
        from vllm_sm80.dev_ab_batch import end
        return end(self)
