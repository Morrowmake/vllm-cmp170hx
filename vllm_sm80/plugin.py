# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's vLLM general plugin (``vllm.general_plugins`` entry point ``sm80``).

vLLM calls :func:`register` in every process that loads general plugins: the
API server, the engine core and each worker. It may be called more than once
in one process and must stay idempotent. It records that it ran, so
``vllm_sm80.guard`` can tell a loaded plugin from a missing one. The fork's
op, layer and model registrations are added here as they move out of
upstream files.
"""

_registered = False


# ---------------------------------------------------------------------------
# Model code (GLM, DeepSeek-V4.1, Qwen4Exp)
# ---------------------------------------------------------------------------

# Architectures whose fork classes replace the upstream ones in vLLM's model
# registry. The "<module>:<class>" form is resolved lazily: nothing is
# imported until vLLM resolves the architecture, so processes that never
# build the model (the API server, other models) do not load the model code.
MODEL_OVERRIDES = {
    "Glm5NextForCausalLM": (
        "vllm_sm80.models.glm5next.sm80_model:Glm5NextForCausalLM"
    ),
    "Glm5NextForConditionalGeneration": (
        "vllm_sm80.models.glm5next.sm80_model:Glm5NextForConditionalGeneration"
    ),
    # The DeepSeek-V4.1 vision wrapper (its text model is built inside it)
    # and its DSpark drafter, whose class lives in the same model package.
    "DeepseekV41ForCausalLM": (
        "vllm_sm80.models.deepseek_v41.sm80_model:DeepseekV41ForCausalLM"
    ),
    "DSparkV41DraftModel": (
        "vllm_sm80.models.deepseek_v41.sm80_model:DSparkDeepseekV4ForCausalLM"
    ),
    # Qwen4Exp's text model, its vision wrapper (which builds the text model
    # itself) and its MTP drafter: router precision, the quantized LM head,
    # the MTP norm and shared embeddings, compute_logits_local and the greedy
    # target argmax.
    "Qwen4ExpForCausalLM": (
        "vllm_sm80.models.qwen4_exp.sm80_model:Qwen4ExpForCausalLM"
    ),
    "Qwen4ExpForConditionalGeneration": (
        "vllm_sm80.models.qwen4_exp.sm80_model:Qwen4ExpForConditionalGeneration"
    ),
    "Qwen4ExpMTP": (
        "vllm_sm80.models.qwen4_exp.sm80_model:Qwen4ExpMTP"
    ),
}


def register_models() -> None:
    """Register :data:`MODEL_OVERRIDES` (re-registering overwrites, so this
    is idempotent).

    Each entry is then given the model-information cache key of its whole
    model package (``vllm_sm80.model_registry``), so a change to the
    upstream model code also invalidates the cached information.
    """
    from vllm.model_executor.models.registry import (
        ModelRegistry,
        _LazyRegisteredModel,
    )
    from vllm_sm80.model_registry import PackageHashedLazyModel

    for architecture, model_cls in MODEL_OVERRIDES.items():
        ModelRegistry.register_model(architecture, model_cls)
        entry = ModelRegistry.models[architecture]
        assert type(entry) is _LazyRegisteredModel, entry
        ModelRegistry.models[architecture] = PackageHashedLazyModel(
            entry.module_name, entry.class_name
        )


# HF config attributes the fork's models default (architecture -> attribute
# -> value), set by vLLM's model-config hook unless the checkpoint or
# --hf-overrides sets them. DeepSeek-V4.1's reference computes FP32 logits
# (ParallelHead: an FP32 weight applied to x.float()), so its LM head, the
# DSpark Markov head and the draft logits run in FP32 (ModelConfig.head_dtype).
HF_CONFIG_DEFAULTS: dict[str, dict[str, object]] = {
    "DeepseekV41ForCausalLM": {"head_dtype": "float32"},
}


def register_hf_config_defaults() -> None:
    """Wrap the ``MODELS_CONFIG_MAP`` entry of each architecture in
    :data:`HF_CONFIG_DEFAULTS` so it also sets the defaults (idempotent: an
    entry wrapped in this process already is left as it is)."""
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm_sm80.model_registry import with_hf_config_defaults

    for architecture, defaults in HF_CONFIG_DEFAULTS.items():
        hook = MODELS_CONFIG_MAP.get(architecture)
        if getattr(hook, "hf_config_defaults", None) is not None:
            continue
        MODELS_CONFIG_MAP[architecture] = with_hf_config_defaults(hook, defaults)


# DFlash-family drafter code shared by the models (vllm_sm80/spec_decode/).
# Architecture -> ("<module>:<class>" of the fork class, the upstream module
# whose class it extends). Resolved lazily, like MODEL_OVERRIDES.
DRAFTER_OVERRIDES = {
    "DFlashDraftModel": (
        "vllm_sm80.spec_decode.dflash_draft:DFlashQwen3ForCausalLM",
        "vllm.model_executor.models.qwen3_dflash",
    ),
    "DFlash2DraftModel": (
        "vllm_sm80.spec_decode.dflash2_model:DFlash2Qwen3ForCausalLM",
        "vllm.model_executor.models.qwen3_dflash2",
    ),
    "LiLiCorrDraftModel": (
        "vllm_sm80.spec_decode.lilicorr_model:LiLiCorrForCausalLM",
        "vllm.model_executor.models.lilicorr",
    ),
}


def register_drafter_models() -> None:
    """Register :data:`DRAFTER_OVERRIDES` (re-registering overwrites, so
    this is idempotent).

    Each entry is then given a model-information cache key over the fork
    module and the upstream module its class extends
    (``vllm_sm80.model_registry``), so a change to the upstream drafter code
    also invalidates the cached information.
    """
    from vllm.model_executor.models.registry import (
        ModelRegistry,
        _LazyRegisteredModel,
    )
    from vllm_sm80.model_registry import UpstreamHashedLazyModel

    for architecture, (model_cls, upstream_module) in DRAFTER_OVERRIDES.items():
        ModelRegistry.register_model(architecture, model_cls)
        entry = ModelRegistry.models[architecture]
        assert type(entry) is _LazyRegisteredModel, entry
        ModelRegistry.models[architecture] = UpstreamHashedLazyModel(
            entry.module_name, entry.class_name, upstream_module
        )


# Upstream pluggable layers the fork's GLM classes replace (layer class name ->
# "<module>:<class>" of the replacement).
GLM_LAYER_OVERRIDES = {
    "Glm5NextLinearAttention": (
        "vllm_sm80.models.glm5next.sm80_kda:Glm5NextLinearAttention"
    ),
    "MultiHeadLatentAttentionWrapper": (
        "vllm_sm80.models.glm5next.sm80_mla:MultiHeadLatentAttentionWrapper"
    ),
}

# Upstream custom ops the fork's GLM classes replace (op class name ->
# "<module>:<class>" of the replacement).
GLM_OP_OVERRIDES = {
    "SparseAttnIndexerKpool": (
        "vllm_sm80.models.glm5next.sm80_indexer_op:SparseAttnIndexerKpool"
    ),
}


def register_glm_layers() -> None:
    """Register :data:`GLM_LAYER_OVERRIDES` as out-of-tree pluggable layers
    and :data:`GLM_OP_OVERRIDES` as out-of-tree custom ops.

    A layer override needs the replacement class itself, so this imports the
    GLM model code. It therefore runs when the fork GLM model module is
    imported, i.e. when vLLM resolves a GLM architecture to the classes of
    :data:`MODEL_OVERRIDES`, which build every KDA and MLA layer (the MTP
    drafter's too), and not at plugin load in every process. Idempotent.
    """
    import importlib

    from vllm.model_executor.custom_op import (
        CustomOp,
        PluggableLayer,
        op_registry_oot,
    )

    for overrides, base in (
        (GLM_LAYER_OVERRIDES, PluggableLayer),
        (GLM_OP_OVERRIDES, CustomOp),
    ):
        for name, qualified in overrides.items():
            module_name, _, class_name = qualified.partition(":")
            layer_cls = getattr(importlib.import_module(module_name), class_name)
            registered = op_registry_oot.get(name)
            if registered is None and base is CustomOp:
                # register_oot names the op by its table key; keep upstream's
                # op name, which the custom_ops switch reads.
                upstream_name = layer_cls.name
                base.register_oot(layer_cls, name=name)
                layer_cls.name = upstream_name
            elif registered is None:
                base.register_oot(layer_cls, name=name)
            elif registered is not layer_cls:
                raise RuntimeError(
                    f"pluggable layer {name!r} is already replaced by {registered}"
                )


def register() -> None:
    """Entry point named in ``plugins/sm80/pyproject.toml``."""
    register_fork_env_vars()
    register_models()
    register_drafter_models()
    register_hf_config_defaults()
    global _registered
    if not _registered:
        _register_exl3()
        _log_registered()
    _registered = True
    register_triton_mla_sparse_backend()
    register_moe_layers()
    log_marlin_decode_v2_gate()
    register_execution_trace()
    register_serial_trace_export()
    register_light_timing()
    register_fork_parsers()
    register_mhc_decode_op()


def register_fork_parsers() -> None:
    """The GLM-4.7 and DeepSeek V4.1 reasoning and tool parsers and the
    DeepSeek V4.1 tokenizer under their upstream names (``vllm_sm80/parsers``,
    ``vllm_sm80/tokenizers``)."""
    from vllm_sm80.parsers import register_parsers
    from vllm_sm80.tokenizers import register_tokenizers

    register_parsers()
    register_tokenizers()


# ---------------------------------------------------------------- layers: mHC
def register_mhc_decode_op() -> None:
    """Build ``MHCFusedPostPreOp`` as the fork's subclass, whose CUDA path
    runs the sm_80 mHC decode kernel (``vllm_sm80/kernels/mhc/fused_op.py``),
    while VLLM_AMPERE_DECODE_KERNELS and VLLM_GLM5_DECODE_MHC_V2 are set; with
    either off the op stays upstream's. Idempotent."""
    import vllm.envs as envs

    if not (envs.VLLM_AMPERE_DECODE_KERNELS and envs.VLLM_GLM5_DECODE_MHC_V2):
        return
    from vllm_sm80.kernels.mhc import fused_op

    fused_op.register()


def register_execution_trace() -> None:
    """Execution traces beside the torch profiler's, with
    VLLM_SM80_EXECUTION_TRACE (vllm_sm80/worker/execution_trace.py)."""
    from vllm_sm80.worker import execution_trace

    execution_trace.install()


def register_serial_trace_export() -> None:
    """Rank-ordered torch profiler trace export, with
    VLLM_SM80_SERIAL_TRACE_EXPORT (vllm_sm80/worker/trace_export.py)."""
    from vllm_sm80.worker import trace_export

    trace_export.install()


def register_light_timing() -> None:
    """GPU timing of CUDA-graph forwards, with VLLM_SM80_LIGHT_TIMING
    (:mod:`vllm_sm80.worker.light_timing`)."""
    from vllm_sm80.worker import light_timing

    light_timing.install()


def is_registered() -> bool:
    """Whether :func:`register` ran in this process."""
    return _registered


def log_marlin_decode_v2_gate() -> None:
    """Log once per process why ``VLLM_AMPERE_MARLIN_DECODE_V2`` alone has no
    effect: decode_gemm_v2 runs only inside the compiled Marlin decode
    (``VLLM_AMPERE_MARLIN_DECODE_CUDA``), which is not reached while that
    switch is off. With it on, ``vllm_sm80/kernels/marlin_moe/decode.py`` logs
    the v2 gate itself.
    """
    import vllm.envs as envs

    if envs.VLLM_AMPERE_MARLIN_DECODE_CUDA or not envs.VLLM_AMPERE_MARLIN_DECODE_V2:
        return
    from vllm.logger import init_logger

    init_logger(__name__).info_once(
        "VLLM_AMPERE_MARLIN_DECODE_V2 is set but the gate is closed "
        "(VLLM_AMPERE_MARLIN_DECODE_CUDA=0); using released Marlin kernels."
    )


def _log_registered() -> None:
    """Log once per process that the plugin is active, with its version."""
    from importlib.metadata import PackageNotFoundError, version

    from vllm.logger import init_logger
    from vllm_sm80.guard import DISTRIBUTION

    try:
        installed = version(DISTRIBUTION)
    except PackageNotFoundError:
        installed = "unknown"
    init_logger(__name__).info(
        "sm80 plugin registered (%s %s)", DISTRIBUTION, installed
    )


# ---------------------------------------------------------------------------
# EXL3 checkpoints
# ---------------------------------------------------------------------------


def _register_exl3() -> None:
    """EXL3 checkpoints: quantization method "exl3" (a checkpoint whose
    quant_method is "exl3" selects it) and load format "exl3"."""
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.model_loader import register_model_loader
    from vllm_sm80.kernels.exl3.loader import Exl3ModelLoader
    from vllm_sm80.kernels.exl3.quant import Exl3Config

    register_quantization_config("exl3")(Exl3Config)
    register_model_loader("exl3")(Exl3ModelLoader)


# ---------------------------------------------------------------------------
# sm_80 kernels, layers and platform
# ---------------------------------------------------------------------------

PLATFORM_QUALNAME = "vllm_sm80.platform.Sm80CudaPlatform"


def platform_plugin() -> str | None:
    """``vllm.platform_plugins`` entry point ``sm80`` (plugins/sm80/pyproject.toml).

    Returns the fork's CUDA platform (``vllm_sm80.platform``) exactly when
    vLLM's builtin CUDA plugin would activate, else None. vLLM prefers a
    single activated out-of-tree platform plugin over the builtin one
    (``vllm.platforms.resolve_current_platform_cls_qualname``).
    """
    register_fork_env_vars()
    from vllm.platforms import cuda_platform_plugin

    if cuda_platform_plugin() is None:
        return None
    return PLATFORM_QUALNAME


TRITON_MLA_SPARSE_BACKEND = (
    "vllm_sm80.attention.triton_mla_sparse.TritonMLASparseBackend"
)


def register_triton_mla_sparse_backend() -> None:
    """Put the Triton sparse-MLA backend in ``AttentionBackendEnum.CUSTOM``.

    ``vllm_sm80.platform`` offers CUSTOM as a candidate after upstream's MLA
    backends. Idempotent (a dict assignment in the backend registry).
    """
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(AttentionBackendEnum.CUSTOM, TRITON_MLA_SPARSE_BACKEND)


# Upstream class name -> (the fork class built in its place, the switches that
# need it). vllm_sm80/layers/moe.
MOE_LAYERS = {
    "MoERunner": (
        "vllm_sm80.layers.moe.runner.Sm80MoERunner",
        (
            "VLLM_SHARED_EXPERT_REORDER",
            "VLLM_MOE_DECODE_ROUTE_V2",
            # The fused decode routing; its own VLLM_GLM5_DECODE_MOE_ROUTING
            # defaults on.
            "VLLM_AMPERE_DECODE_KERNELS",
            "VLLM_AMPERE_DECODE_IDX_GLUE",
        ),
    ),
    "RoutedExperts": (
        "vllm_sm80.layers.moe.routed_experts.Sm80RoutedExperts",
        ("VLLM_MOE_MASK_PADDING",),
    ),
    "GateLinear": (
        "vllm_sm80.layers.moe.gate.Sm80GateLinear",
        ("VLLM_MOE_DECODE_ROUTE_V2",),
    ),
}


def register_moe_layers() -> None:
    """Build ``MoERunner``, ``RoutedExperts`` and ``GateLinear`` as the fork's
    subclasses (``PluggableLayer`` out-of-tree lookup by class name), each only
    while one of the switches that needs it is set (``MOE_LAYERS``).

    With every switch off nothing is registered and every class stays
    upstream's: an out-of-tree class also replaces the upstream one in
    ``maybe_get_oot_by_class`` checks (MoE LoRA's ``can_replace_layer``),
    which subclasses of the upstream class would no longer match. See
    ``vllm_sm80/layers/moe``. Idempotent: a name already holding the fork
    class is skipped (``register_oot`` asserts on a duplicate name).
    """
    import vllm.envs as envs
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    from vllm.utils.import_utils import resolve_obj_by_qualname

    for name, (qualname, switches) in MOE_LAYERS.items():
        if not any(getattr(envs, switch) for switch in switches):
            continue
        layer_cls = resolve_obj_by_qualname(qualname)
        if op_registry_oot.get(name) is not layer_cls:
            PluggableLayer.register_oot(layer_cls, name=name)


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

_env_vars_registered = False


def register_fork_env_vars() -> None:
    """Add the fork's variables (``vllm_sm80/envs.py``) to vLLM's table
    ``vllm.envs.environment_variables``, which ``vllm.envs.__getattr__`` reads
    lazily. Both entry points call it first: the platform plugin, which vLLM
    resolves before any fork platform code runs, and the general plugin,
    which vLLM loads in every process before engine and model code
    (``validate_environ`` runs after it). The fork modules that read a
    variable when imported call it too: an upstream file can import them in a
    process where neither entry point ran (``VLLM_PLUGINS`` without sm80).
    Once per process."""
    global _env_vars_registered
    if _env_vars_registered:
        return
    import vllm.envs as envs
    from vllm_sm80.envs import register_env_vars

    register_env_vars(envs.environment_variables)
    _env_vars_registered = True
