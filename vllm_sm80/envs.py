# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables of the CMP 170HX (sm_80) fork.

The plugin's entry points call :func:`register_env_vars` first
(``vllm_sm80.plugin.register_fork_env_vars``), so every name below is an
ordinary vLLM environment variable in every process that loaded the plugin:
``envs.X``, ``envs.is_set``, ``envs.compile_factors`` and the
unknown-variable check in ``Platform.validate_environ`` all see it.
"""

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type and default of each variable in FORK_ENV_VARS. Readers use
    # ``vllm.envs.X``, which resolves through vllm.envs.__getattr__.
    VLLM_AMPERE_MARLIN_BUILD: bool = False
    VLLM_CUSTOM_ALLREDUCE_PCIE_P2P: bool = False
    VLLM_CUSTOM_ALLREDUCE_ALGO: str = ""
    VLLM_SM80_MHC_PREFILL: bool = False
    VLLM_SM80_SPARSE_MLA_PREFILL: bool = False
    VLLM_SM80_ALLREDUCE: bool = False
    VLLM_INDEXER_DECODE_RAW_K: bool = False
    VLLM_INDEXER_DSA_REF: bool = False
    VLLM_GLM5_SMLA_PREFILL_PRED_LOAD: bool = False
    VLLM_SM80_KDA_PREFILL: bool = False
    VLLM_STATE_INDEX_CHECK: bool = False
    VLLM_SM80_EXECUTION_TRACE: bool = False
    VLLM_SM80_SERIAL_TRACE_EXPORT: bool = False
    VLLM_SM80_LIGHT_TIMING: str = ""
    VLLM_AMPERE_DECODE_KERNELS: bool = False
    VLLM_GLM5_DECODE_MOE_ROUTING: bool = True
    VLLM_GLM5_DECODE_MHC_V2: bool = False
    VLLM_MOE_DECODE_ROUTE_V2: bool = False
    VLLM_AMPERE_KDA_FOLD: bool = False
    VLLM_AMPERE_KDA_RECOVER_OFF: bool = False
    VLLM_FLA_PIN_AUTOTUNE: bool = False
    VLLM_FLA_AUTOTUNE_SKIP_SPILLING: bool = False
    VLLM_AMPERE_DECODE_IDX_GLUE: bool = False
    VLLM_AMPERE_KDA_SMALL_STACK: bool = False
    VLLM_GLM5_DECODE_IDX_GLUE_PARTS: str = "weights,glue,fwht,cache,moesum"
    VLLM_MOE_DETERMINISTIC_ALIGN: int = 0
    VLLM_SM80_TOPK_SORTED: bool = False
    VLLM_SM80_TOPK_CANONICAL_TIES: bool = False
    VLLM_MOE_MASK_PADDING: bool = False
    VLLM_GLM5_MOE_ROUTE_V2_MASK: bool = True
    VLLM_THIN_GEMM: bool = False
    VLLM_GLM5_DRAFTER_ROPE_FIT: bool = False
    VLLM_TOOL_CHOICE_NONE_MASK: bool = False
    VLLM_DSPARK_PER_K_WIDTH: bool = False
    VLLM_DSPARK_CONFIDENCE_DUMP: str = ""
    VLLM_DSPARK_ADAPTIVE_DEPTH: str = ""
    VLLM_DSPARK_DEPTH_HINT: str = ""
    VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE: str = ""
    VLLM_DFLASH2_SELECTOR_SHARD: bool = False
    VLLM_MEM_ATTRIBUTION: bool = False
    VLLM_CUDAGRAPH_ESTIMATE_SPLIT: int = 0
    VLLM_KV_MAMBA_INFLIGHT_STATES: bool = False
    VLLM_KV_PLAN_LOG: bool = False
    VLLM_THIN_GEMM_CONCAT: bool = False
    VLLM_THIN_GEMM_GLUON: bool = False
    VLLM_THIN_GEMM_GLUON_FORCE_M: int = 0
    VLLM_SHARED_EXPERT_REORDER: bool = False
    VLLM_FAIR_PREFILL_SHORT_BYPASS: bool = False
    VLLM_PP_SPREAD_DECODES: bool = False
    VLLM_PP_PACKED_HOP: bool = False
    VLLM_PP_HOP_NO_METADATA: bool = False
    VLLM_PP_SPLIT_DRAFT_EVENT: bool = False
    VLLM_PP_DRAFT_TAIL_STAGE: int = -1
    VLLM_GLM5_PP_FOLD_DRAFT_FC: bool = False
    VLLM_SM80_MARLIN_PREFILL: bool = False
    VLLM_AMPERE_MARLIN_DECODE_CUDA: bool = False
    VLLM_AMPERE_MARLIN_DECODE_V2: bool = False
    VLLM_EXL3_DENSE_SPLITK: bool = False
    VLLM_EXL3_DENSE_MMA: bool = False
    VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE: str = ""
    VLLM_EXL3_PLE_PLACEMENT: str = "ssd"
    VLLM_EXL3_OVERRIDE: str = ""
    VLLM_EXL3_OVERRIDE_SHA256: str = ""
    VLLM_EXL3_HYBRID: bool = False
    VLLM_SM80_ALLREDUCE_ROUND_ONCE: bool = False
    VLLM_SM80_PREFILL_REDUCE_OVERLAP: bool = False
    VLLM_DEEPSEEK_V41_FUSED_MHC: bool = False
    VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH: bool = False
    VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2: bool = False
    VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM: bool = False
    VLLM_DEEPSEEK_V41_ROUTER_MMA: bool = False
    VLLM_DEEPSEEK_V41_FUSED_ROUTE: bool = False
    VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED: bool = False
    VLLM_DEEPSEEK_V41_GATE_GEMV: bool = False
    VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM: bool = False
    VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE: bool = False
    VLLM_DEEPSEEK_V41_MOE_PREFILL_TC: bool = False
    VLLM_DEEPSEEK_V41_ATTN_PIPELINE: bool = False
    VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH: bool = False
    VLLM_SM80_SHARD_ARGMAX: bool = False
    VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER: bool = False
    VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL: bool = False
    VLLM_DEEPSEEK_V41_SWA_META_BATCH: bool = False
    VLLM_EXL3_PLE_COLD: bool = False
    VLLM_EXL3_PLE_READ_RETRIES: int = 3
    VLLM_EXL3_PLE_TIMING: bool = False
    VLLM_EXL3_PLE_OVERLAP: bool = False
    VLLM_EXL3_DENSE_PREFILL_TC: bool = False
    VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE: str = "planned"
    VLLM_EXL3_PLE_NATIVE_READ: bool = False
    VLLM_EXL3_PLE_HOSTFN_READ: bool = False
    VLLM_ENGRAM_PLACEMENT: str = "ssd"
    VLLM_ENGRAM_ROW_CACHE_GIB: float = 1.0
    VLLM_ENGRAM_ROW_CACHE_PRELOAD: bool = False
    VLLM_ENGRAM_DIRECT_READ: bool = False
    VLLM_ENGRAM_SEGMENT: str = ""
    VLLM_ENGRAM_TELEMETRY_STEPS: int = 0
    VLLM_EXL3_MOE_V2: bool = False
    VLLM_EXL3_MOE_V2_FOLD: bool = False
    VLLM_EXL3_MOE_V2_DRAFT: bool = False
    VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS: int = 0
    VLLM_EXL3_MOE_V2_RING: bool = False
    VLLM_EXL3_MOE_PREFILL_TC: bool = False
    VLLM_EXL3_MOE_PREFILL_UNCHUNKED: bool = False
    VLLM_EXL3_EXPERTS_CONTIGUOUS: bool = False
    VLLM_SM80_TUNING_FILE: str = ""
    VLLM_SM80_CAPTURE: str = ""
    VLLM_SM80_REPLAY_DIR: str = ""
    VLLM_SM80_REPLAY_LIMIT: int | None = None
    VLLM_DEV_AB_SWITCH: str = ""
    VLLM_HOT_SWAP: str = ""

# Same logger name as vllm/envs.py, so the retired-name warnings keep their
# logger.
logger = logging.getLogger("vllm.envs")


def _env_with_choices(*args: Any, **kwargs: Any) -> Callable[[], str | None]:
    """``vllm.envs.env_with_choices``, imported when a getter runs: this module
    is imported while ``vllm.envs`` is still loading."""
    from vllm.envs import env_with_choices

    return env_with_choices(*args, **kwargs)


def _replay_limit() -> int | None:
    """VLLM_SM80_REPLAY_LIMIT; None when unset (each replay test then reads
    its own number of records)."""
    raw = os.getenv("VLLM_SM80_REPLAY_LIMIT", "")
    if not raw:
        return None
    return int(raw)


def _custom_allreduce_algo() -> str:
    """Mirror of the ``std::getenv`` parse in ``csrc/custom_all_reduce.cuh``.

    The C++ side compares the raw bytes against four spellings and throws on
    anything else, so this neither strips nor lowercases. Unset is the empty
    string, which is what the kernel's own size heuristic means.
    """
    raw = os.getenv("VLLM_CUSTOM_ALLREDUCE_ALGO")
    if raw is None:
        return ""
    if raw not in ("1stage", "oneshot", "2stage", "twoshot"):
        raise ValueError(
            f"Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: {raw}. "
            "Valid values: 1stage, oneshot, 2stage, twoshot"
        )
    return raw


FORK_ENV_VARS: dict[str, Callable[[], Any]] = {
    # Must be 1 for the standalone builder of the optional sm_80 Marlin decode
    # and prefill library (_ampere_marlin_C, build_standalone.py beside its
    # sources); see docs/features/quantization/README.md.
    "VLLM_AMPERE_MARLIN_BUILD": lambda: (
        os.getenv("VLLM_AMPERE_MARLIN_BUILD", "0") == "1"
    ),
    # Let the custom all-reduce treat *PCIe* peer-to-peer as "fully
    # connected". Upstream only counts NVLink, so on a PCIe-only node with
    # more than two GPUs CustomAllreduce disables itself before it ever runs
    # the P2P probe. With this set, every pair must report NVML
    # P2P_CAPS_INDEX_READ and _WRITE OK (or peer access both ways under
    # torch) -- and the gpu_p2p_access_check probe, which is the only check
    # that moves real bytes, becomes mandatory regardless of
    # VLLM_SKIP_P2P_CHECK. It is also the switch that hands the fast path
    # from the host-staged all-reduce to the device-memory one: see
    # vllm_sm80/distributed/host_shm_all_reduce.py.
    # Off by default; a driver that advertises peer access it cannot route
    # turns a wrong answer here into a hang, so opt in only after measuring.
    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": lambda: bool(
        int(os.getenv("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", "0"))
    ),
    # Which CustomAllreduce kernel runs: unset (the default) leaves the
    # in-kernel size heuristic alone, "1stage"/"oneshot" and
    # "2stage"/"twoshot" force one. Nothing in Python consumes it -- it is
    # read with std::getenv inside csrc/custom_all_reduce.cuh -- and
    # upstream declares it nowhere, so an operator who sets it (serve.sh
    # does, to 2stage) gets one 'Unknown vLLM environment variable
    # detected' per worker at start. It is declared here for that reason
    # alone. The accessor mirrors the C++ parse exactly: an exact byte
    # compare against the four spellings, no case folding and no
    # whitespace stripping, raising on anything else the way the kernel
    # throws at its first all-reduce.
    "VLLM_CUSTOM_ALLREDUCE_ALGO": lambda: _custom_allreduce_algo(),
    # The fork's tensor-parallel all-reduce transports, each message sent on
    # the one its world size, bytes, dtype and the CUDA graph state select
    # (the order is listed in vllm_sm80/distributed/communicator.py):
    # - the host-staged all-reduce (host_shm_all_reduce.py), TP groups only;
    #   it stands aside where VLLM_CUSTOM_ALLREDUCE_PCIE_P2P is set and peer
    #   access works;
    # - inside the PCIe peer-to-peer custom all-reduce, the flags-in-data
    #   two-shot kernel (custom_all_reduce_flags.py) for BF16 messages up to
    #   allreduce/flags_max_bytes of the sm80 tuning table at TP2 or TP4,
    #   bitwise identical to cross_device_reduce_2stage at TP4 and to
    #   upstream's kernels at TP2;
    # - for the sums a model layer owns (DeepSeek-V4.1's wo_b and MoE outputs),
    #   on eager steps, the copy-engine ring (ce_ring_all_reduce.py) for
    #   messages the two paths above do not take (device memory per rank: one
    #   scratch slot per ring step for a max_num_batched_tokens x hidden FP32
    #   message); round-once sums take its owner sum (FP32 in rank order).
    # VLLM_CUSTOM_ALLREDUCE_PCIE_P2P stays the hardware gate of every
    # peer-to-peer path. Default OFF: unset, every group gets upstream's
    # communicator and dispatch.
    "VLLM_SM80_ALLREDUCE": lambda: bool(
        int(os.getenv("VLLM_SM80_ALLREDUCE", "0"))
    ),
    # The sm_80 mHC pre-norm projection GEMM at prefill sizes
    # (vllm_sm80/kernels/mhc/prenorm.py) in place of the TileLang kernel, for
    # steps of at least the mhc/prefill_min_rows row of the sm80 tuning table
    # (GA100-class parts only; measured on CMP 170HX, 70 SMs). Default OFF:
    # with this unset the kernel is not imported and the call site keeps the
    # upstream path.
    "VLLM_SM80_MHC_PREFILL": lambda: bool(
        int(os.getenv("VLLM_SM80_MHC_PREFILL", "0"))
    ),
    # The sm_80 sparse-MLA (DSA) prefill attention in place of the Triton
    # sparse-MLA kernel's prefill schedule, for steps of at least
    # the attention_prefill/sparse_mla_min_rows row of the sm80 tuning table
    # and contexts of at least its sparse_mla_min_ctx row (twice index_topk
    # without one). The head count per
    # card picks the kernel: all 64 heads on one card (pipeline parallel,
    # TP=1) take a Gluon kernel that gathers each cache row once for all heads
    # and compacts the valid index slots before the loop
    # (vllm_sm80/kernels/attention/sparse_prefill_mla_pp.py; bf16, the
    # 512-wide NoPE layout, 2176 index slots and 1..2312 query rows), every
    # other call the retuned Triton kernel of sparse_prefill_mla.py. Default
    # OFF: with this unset neither is imported and the call site keeps the
    # upstream path.
    "VLLM_SM80_SPARSE_MLA_PREFILL": lambda: bool(
        int(os.getenv("VLLM_SM80_SPARSE_MLA_PREFILL", "0"))
    ),
    # DSA indexer decode logits (vllm_sm80/kernels/attention/triton_mqa_logits.py,
    # Triton path) on sm_80: dequantize the e4m3 K tiles by bit placement
    # (raw bf16 = value * 2^-120, 2^60 folded into Q and the head weights)
    # instead of through fp32. Bitwise identical logits on sm_80, where the
    # tensor cores were verified to keep the bf16 subnormal inputs this
    # produces; other devices keep the fp32 dequant. Off by default (fp32
    # dequant); the GLM launcher sets 1.
    "VLLM_INDEXER_DECODE_RAW_K": lambda: bool(
        int(os.getenv("VLLM_INDEXER_DECODE_RAW_K", "0"))
    ),
    # Debug: 1 swaps the DSA indexer's Triton fp8 MQA-logits kernels
    # (vllm_sm80/kernels/attention/triton_mqa_logits.py) for their pure-torch
    # references.
    "VLLM_INDEXER_DSA_REF": lambda: os.getenv("VLLM_INDEXER_DSA_REF", "0") == "1",
    # Predicate the row gather of the sm_80 sparse-MLA prefill kernel
    # (vllm_sm80/kernels/attention/sparse_prefill_mla.py) on the index mask, so that
    # invalid (-1) top-k slots fetch nothing instead of all reading cache row
    # 0. The output is bitwise unchanged. Taken only with all 64 heads on one
    # card (pipeline parallel, TP=1) or 16 heads per rank (tensor parallel 4)
    # at the 512-wide NoPE layout, 2176 index slots and the schedule that head
    # count gets; every other shape keeps the plain gather.
    # Needs VLLM_SM80_SPARSE_MLA_PREFILL=1 as well. Off by default.
    "VLLM_GLM5_SMLA_PREFILL_PRED_LOAD": lambda: bool(
        int(os.getenv("VLLM_GLM5_SMLA_PREFILL_PRED_LOAD", "0"))
    ),
    # sm_80 KDA chunked prefill (vllm_sm80/models/glm5next/ops/kda_prefill.py)
    # in place of the upstream chunk_kda_with_fused_gate: six launches instead
    # of nine, fixed launch configurations, fp32-exact intermediates where the
    # recurrence needs them. The layer's head count picks the shape: 64 heads
    # per card (pipeline parallel, TP=1) or 32 heads per card (PP=2 with TP=2
    # inside each stage) with chunks of up to 2312 tokens, or 16 heads per
    # card (tensor parallel 4) with chunks of up to 3460 tokens (the TP token
    # budget). Taken only on sm_80 with the Triton prefill
    # backend, head dim 128, bf16 activations, fp32 state, the safe gate at
    # -5 and at most 16 sequences per chunk; everything else keeps the
    # upstream path. Off by default.
    "VLLM_SM80_KDA_PREFILL": lambda: bool(
        int(os.getenv("VLLM_SM80_KDA_PREFILL", "0"))
    ),
    # Debug: range-check every KDA/Mamba state index (GDN metadata, align-mode
    # pre-/post-copy block columns and block ids) on the host before the
    # kernels use it; raises instead of addressing outside the state pool.
    # Synchronizes every step. Off by default.
    "VLLM_STATE_INDEX_CHECK": lambda: bool(
        int(os.getenv("VLLM_STATE_INDEX_CHECK", "0"))
    ),
    # Diagnostic: every torch profiler trace also gets torch's execution trace
    # (op inputs and outputs by tensor and storage id) beside it as
    # <stem>.et.json[.gz] (vllm_sm80/worker/execution_trace.py). Ops replayed
    # inside CUDA graphs are not in it. Off by default.
    "VLLM_SM80_EXECUTION_TRACE": lambda: bool(
        int(os.getenv("VLLM_SM80_EXECUTION_TRACE", "0"))
    ),
    # Inside the stop_profile request, the workers of an engine stop their
    # torch profiler sessions one after another (pipeline stage, then
    # tensor-parallel rank), each releasing its trace's host memory before the
    # next starts, so the host holds one rank's trace at a time; a session
    # stopped by its iteration limit inside a model step is not ordered
    # (vllm_sm80/worker/trace_export.py). Off by default.
    "VLLM_SM80_SERIAL_TRACE_EXPORT": lambda: bool(
        int(os.getenv("VLLM_SM80_SERIAL_TRACE_EXPORT", "0"))
    ),
    # A JSON object (vllm_sm80.worker.light_timing): every forward of the
    # named root modules captured into a CUDA graph stamps the GPU timer at
    # each decoder layer and named scope into a device ring, dumped to files
    # on request, with no host work per step. Diagnostic; empty: off.
    "VLLM_SM80_LIGHT_TIMING": lambda: os.getenv("VLLM_SM80_LIGHT_TIMING", ""),
    # Opt in to the sm_80 decode kernels (mHC and MoE routing in
    # vllm_sm80/kernels/, the KDA decode step in vllm/models/glm5next/nvidia/ops/;
    # GA100-class parts only; measured on CMP 170HX, 70 SMs). Default OFF: with
    # this unset no such kernel is imported or called and every call site keeps
    # the upstream code path exactly. The per-family switches below are only
    # consulted when this is on, so a single family can be disabled on the
    # server without a rebuild -- which is how a regression gets bisected. The
    # fused KDA decode step has no family switch: this one turns it off.
    #
    "VLLM_AMPERE_DECODE_KERNELS": lambda: bool(
        int(os.getenv("VLLM_AMPERE_DECODE_KERNELS", "0"))
    ),
    "VLLM_GLM5_DECODE_MOE_ROUTING": lambda: bool(
        int(os.getenv("VLLM_GLM5_DECODE_MOE_ROUTING", "1"))
    ),
    # Opt in to the sm_80 MLA-indexer decode glue folds in
    # vllm_sm80/models/glm5next/ops/idx_glue.py (GA100-class parts only).
    # Default OFF: with this unset every call site keeps its existing code path
    # exactly.
    # VLLM_GLM5_DECODE_IDX_GLUE_PARTS selects the folds (comma list, only read
    # when the switch is on): weights (fp32 head weights from the wk+weights
    # thin GEMM), glue (indexer casts/fill/copy folded into kernels), fwht
    # (weight scale folded into the query Hadamard+quant, more CTAs), cache
    # (MLA latent cache write merged with the top-k index remap), moesum
    # (routed moe_sum fused with the shared-expert add).
    "VLLM_AMPERE_DECODE_IDX_GLUE": lambda: bool(
        int(os.getenv("VLLM_AMPERE_DECODE_IDX_GLUE", "0"))
    ),
    "VLLM_GLM5_DECODE_IDX_GLUE_PARTS": lambda: os.getenv(
        "VLLM_GLM5_DECODE_IDX_GLUE_PARTS", "weights,glue,fwht,cache,moesum"
    ),
    # mHC v3 (vllm_sm80/kernels/mhc/decode.py): one implementation for
    # 1 <= M <= the mhc/decode_v2_max_rows row of the sm80 tuning table that
    # replaces TileLang in that range. Off by default; needs
    # VLLM_AMPERE_DECODE_KERNELS=1 as well.
    "VLLM_GLM5_DECODE_MHC_V2": lambda: bool(
        int(os.getenv("VLLM_GLM5_DECODE_MHC_V2", "0"))
    ),
    # Opt in to vllm_sm80/kernels/moe/route.py: the MoE router GEMV, the
    # sigmoid/bias top-k and the Marlin block alignment as one op (one launch
    # for 2 <= M <= 64), replacing GateLinear's cuBLAS split-K GEMM + reduce and
    # the routing kernels. Needs VLLM_AMPERE_DECODE_KERNELS=1 as well. Default
    # OFF: with it unset nothing in vllm_sm80/kernels/moe/route.py is imported
    # or called. Its row bound is the moe_router/route_v2_max_rows row of the
    # sm80 tuning table (at most 64).
    "VLLM_MOE_DECODE_ROUTE_V2": lambda: bool(
        int(os.getenv("VLLM_MOE_DECODE_ROUTE_V2", "0"))
    ),
    # RecoverSSM commit fold (TP4, vllm_sm80/kernels/kda/recover.py): the
    # replay of a step's accepted tokens over the KDA state runs at the start
    # of the next verify instead of in a separate commit after sampling; steps
    # whose state is read before then (block boundaries) commit as before.
    # Bitwise the same state. Off by default; no effect where RecoverSSM is off.
    "VLLM_AMPERE_KDA_FOLD": lambda: bool(int(os.getenv("VLLM_AMPERE_KDA_FOLD", "0"))),
    # Diagnostic, never for serving: RecoverSSM off (TP4,
    # vllm_sm80/kernels/kda/recover.py) where its gate holds. The KDA layers
    # then keep one state per draft position, as wherever that gate is closed
    # (draft-position state pages, the per-position verify; no records, commit
    # or commit fold): the same-tree reference without state recovery. Off by
    # default.
    "VLLM_AMPERE_KDA_RECOVER_OFF": lambda: bool(
        int(os.getenv("VLLM_AMPERE_KDA_RECOVER_OFF", "0"))
    ),
    # Pin every @triton.autotune'd kernel of the vendored flash-linear-attention
    # ops and the GLM-5 KDA chunked-prefill kernels to ONE config per autotune
    # key on sm_80 (table in
    # vllm_sm80/fla/autotune_pins.py), so a boot
    # never benchmarks and the chosen config, hence the reduction order, is a
    # pure function of the shape. Read when those modules are imported; other
    # devices keep normal autotuning. Off by default.
    "VLLM_FLA_PIN_AUTOTUNE": lambda: bool(
        int(os.getenv("VLLM_FLA_PIN_AUTOTUNE", "0"))    ),
    # With pinning off, the vendored flash-linear-attention autotuners on sm_80
    # build each config before timing it and skip (never launch) one whose
    # build needs more than the context's default 1,024 B of local memory per
    # thread (chunk_fwd_kernel_o at two warps and the 128 x 128 tile: 11,664
    # B). Timing one in the memory profile makes the driver keep that size x
    # 2,048 x SMs bytes reserved (about 1.5 GiB on a 74-SM card), which the KV
    # cache loses. Only boots that benchmark (an empty or stale autotune cache)
    # are affected; the kernels are unchanged. Read when those modules are
    # imported. Off by default.
    "VLLM_FLA_AUTOTUNE_SKIP_SPILLING": lambda: bool(
        int(os.getenv("VLLM_FLA_AUTOTUNE_SKIP_SPILLING", "0"))
    ),
    # RecoverSSM verify of the fused KDA decode (TP4, kda_decode.py; the commit
    # fold's included): nseq 7-8 take one 16-row warp per CTA (twice the
    # CTAs) instead of one 32-row warp that spills 2,184 B of stack per thread
    # (2,624 B with the fold; at most 872 B at 16 rows). The driver reserves
    # the largest stack launched (at least its default 1,024 B) for every
    # thread an SM can hold and the verify's first launch falls in the
    # CUDA-graph memory profile, so the smaller stack leaves that memory to
    # the KV cache. The same warp layouts, hence bitwise the same outputs and
    # states. Off by default; no effect where RecoverSSM is off.
    "VLLM_AMPERE_KDA_SMALL_STACK": lambda: bool(
        int(os.getenv("VLLM_AMPERE_KDA_SMALL_STACK", "0"))
    ),
    # Rank tokens inside each expert segment of moe_align_block_size by
    # ascending flat routed-row index instead of by atomic arrival order.
    # num_experts is 288 for GLM-5.3-Flash, so the CUDA launcher always
    # takes its two-kernel path, whose count_and_sort kernel ranks with a
    # global atomicAdd; the fused Marlin MoE then accumulates each expert
    # in whatever order that produced. 0 = upstream CUDA op (default),
    # 1 = deterministic Triton kernel; any other value is rejected.
    "VLLM_MOE_DETERMINISTIC_ALIGN": lambda: int(
        _env_with_choices("VLLM_MOE_DETERMINISTIC_ALIGN", "0", ["0", "1"])()
    ),
    # Put each row of a sparse indexer's selected top-k (token, pool or block
    # ids) in ascending order, -1 fill last, before attention reads it: the
    # GLM kpool and Qwen QSA indexers (vllm_sm80/kernels/indexer/). The stock
    # top-k kernels return the right set in an arrival-dependent order, and
    # sparse attention accumulates its online softmax in index order, so the
    # same prompt otherwise gets a different attention output from run to run
    # (last-bit differences that the rest of the model amplifies). Only the
    # order changes, never the set. With VLLM_SM80_TOPK_CANONICAL_TIES the
    # sort runs inside the tie-fix launch, otherwise as a launch of its own.
    # Default OFF: unset, the indexers keep the top-k kernel's order.
    "VLLM_SM80_TOPK_SORTED": lambda: bool(
        int(os.getenv("VLLM_SM80_TOPK_SORTED", "0"))
    ),
    # Keep the fast indexer top-k kernels but make the selected SET a function
    # of the scores alone: where entries tied with the k-th score straddle the
    # selection boundary, the tied slots go to the lowest indices (the GLM
    # kpool and Qwen QSA indexers, vllm_sm80/kernels/indexer/). The stock
    # kernels resolve such ties by atomic arrival order, and which kernel
    # runs depends on the row count and length, so the set otherwise depends
    # on batch shape and timing. One extra read of the logits per row;
    # uncontested rows are left untouched.
    # Default OFF: unset, the indexers keep the top-k kernel's set.
    "VLLM_SM80_TOPK_CANONICAL_TIES": lambda: bool(
        int(os.getenv("VLLM_SM80_TOPK_CANONICAL_TIES", "0"))
    ),
    # Route the padding rows of a padded batch (CUDA-graph sizes) to no
    # expert (topk_ids = -1) before the fused MoE. Their hidden states come
    # from stale input buffers, and where they land in the expert blocks
    # changes how Marlin splits K for the real rows next to them, so the same
    # request otherwise gets last-bit different MoE outputs depending on what
    # ran before. Uses the forward context's is_padding mask; batches of up to
    # the moe_router/fused_route_max_rows row of the sm80 tuning table are left
    # unmasked. Default OFF.
    "VLLM_MOE_MASK_PADDING": lambda: bool(
        int(os.getenv("VLLM_MOE_MASK_PADDING", "0"))
    ),
    # With VLLM_MOE_MASK_PADDING=1 and the sm_80 fused router v2
    # (VLLM_MOE_DECODE_ROUTE_V2=1): route the padding rows to no expert
    # inside the router's own launch (read from the device-side is_padding
    # mask), so its block alignment is already that of the masked ids and is
    # kept, instead of being dropped and recomputed on every padded batch of
    # 9..32 rows. No effect unless both of those are on. Default ON; set 0
    # for the recompute path.
    "VLLM_GLM5_MOE_ROUTE_V2_MASK": lambda: bool(
        int(os.getenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1"))
    ),
    # Opt in to the sm_80 thin-M BF16 GEMM in vllm_sm80/kernels/thin_gemm/ for the
    # layers GLM-5.3-Flash's W4A16 recipe leaves unquantized (GA100-class parts
    # only; measured on CMP 170HX, 70 SMs). Default OFF: with this unset the
    # kernel module is never imported, triton is never touched, and
    # dispatch_unquantized_gemm returns the upstream callable itself, so the
    # unflagged path is byte-identical rather than merely equivalent.
    "VLLM_THIN_GEMM": lambda: bool(int(os.getenv("VLLM_THIN_GEMM", "0"))),
    # Size the DFlash drafter's RoPE cos/sin cache to the reachable positions
    # (max_model_len plus a small query margin) instead of the drafter config's
    # max_position_embeddings. Rows are position-indexed, so every reachable
    # row is unchanged; the unreachable tail is simply not allocated.
    "VLLM_GLM5_DRAFTER_ROPE_FIT": lambda: bool(
        int(os.getenv("VLLM_GLM5_DRAFTER_ROPE_FIT", "0"))
    ),
    # GLM-4.7-format parser (glm47): with tool_choice="none", forbid the
    # tool-call start token in decoding (draft and target). Off by default:
    # requests are adjusted as upstream adjusts them; the GLM launcher sets 1.
    "VLLM_TOOL_CHOICE_NONE_MASK": lambda: bool(
        int(os.getenv("VLLM_TOOL_CHOICE_NONE_MASK", "0"))
    ),
    # DSpark drafts the scheduled K rather than drafting the maximum and
    # truncating. Each positive width has persistent anchors and CUDA graphs.
    "VLLM_DSPARK_PER_K_WIDTH": lambda: bool(
        int(os.getenv("VLLM_DSPARK_PER_K_WIDTH", "0"))
    ),
    # Diagnostics: a directory. The DSpark drafter computes its confidence
    # head in every draft step (without adaptive verification) and TP rank 0
    # appends each request's per-position confidences with the accepted
    # length of the round they predicted (dspark_confidence.py). Empty: off.
    "VLLM_DSPARK_CONFIDENCE_DUMP": lambda: os.getenv(
        "VLLM_DSPARK_CONFIDENCE_DUMP", ""
    ),
    # A dspark-depth-policy/1 JSON file: each DSpark verify step keeps per
    # request only the drafts the policy (draft-step features and the
    # request's acceptance history, against the step cost table) makes worth
    # verifying; the rows past them take no routed expert and their drafts
    # count as rejected (dspark_row_mask.py). Greedy output is the unmasked
    # output truncated at the kept drafts. A host_width block in the file also
    # has the scheduler choose each step's verified width from the requests'
    # earlier rounds (dspark_host_width.py). Empty: off.
    "VLLM_DSPARK_ADAPTIVE_DEPTH": lambda: os.getenv("VLLM_DSPARK_ADAPTIVE_DEPTH", ""),
    # Trial: DSpark verify width from each request's last verified round, per
    # load (vllm_sm80/spec_decode/dspark_depth_hint.py); empty is off.
    "VLLM_DSPARK_DEPTH_HINT": lambda: os.getenv("VLLM_DSPARK_DEPTH_HINT", ""),
    # Diagnostics (step-cost calibration): a file holding a depth; while it
    # holds one inside the adaptive_k load width, every step verifies that
    # depth.
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE": lambda: os.getenv(
        "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE", ""
    ),
    # Shard the DFlash2 candidate selector's two (vocab, rank) codebooks by
    # vocab rows across the tensor-parallel ranks instead of replicating them.
    # Each rank looks up the rows it owns, zero elsewhere, and one all-reduce
    # of the two gathered row sets rebuilds them exactly (x + 0 = x).
    "VLLM_DFLASH2_SELECTOR_SHARD": lambda: bool(
        int(os.getenv("VLLM_DFLASH2_SELECTOR_SHARD", "0"))
    ),
    # Log where device memory goes at start-up: resident bytes per module
    # group after loading, the peak of each profile_run stage, the KV sizing
    # terms, and what the attention metadata builders allocate after the KV
    # cache is sized. Logging only; the KV cache size is unchanged.
    "VLLM_MEM_ATTRIBUTION": lambda: bool(
        int(os.getenv("VLLM_MEM_ATTRIBUTION", "0"))
    ),
    # CUDA-graph memory estimate as the bytes the real capture keeps: the
    # throwaway graph pool, the graph objects, the buffers the capture creates
    # and what profiling leaves allocated, instead of the profiling capture's
    # whole free-memory drop (which also holds one-time costs of the first
    # capture and the warmups' allocator cache). The profiling capture captures
    # every FULL graph rather than vLLM's two largest plus a linear
    # extrapolation. Logs the split once per rank,
    # and after the real capture what it keeps against the estimate. Runs the
    # fork GPU worker (vllm_sm80/worker/graph_mem_worker.py, V2 model runner
    # only; vllm_sm80/worker/cudagraph_mem.py). 2 also records the allocation
    # history of the profiling run and lists what it leaves allocated by
    # call stack (diagnostics; slower capture). Default 0 (OFF).
    "VLLM_CUDAGRAPH_ESTIMATE_SPLIT": lambda: int(
        os.getenv("VLLM_CUDAGRAPH_ESTIMATE_SPLIT", "0")
    ),
    # Align-mode Mamba KV reservation under async scheduling or pipeline
    # parallelism: also reserve the state blocks a request holds for the
    # prefill chunks it has in flight (min(max_concurrent_batches,
    # cdiv(max_in_flight_tokens, block)) - 1 per group beyond the two of
    # synchronous scheduling), charged once per running request. Changes the
    # reported KV capacity and the single-request fit check; the block pool
    # and the admission cap are unchanged. Default OFF.
    "VLLM_KV_MAMBA_INFLIGHT_STATES": lambda: bool(
        int(os.getenv("VLLM_KV_MAMBA_INFLIGHT_STATES", "0"))
    ),
    # Diagnostic: log the scheduler's KV cache plan once at startup, one line
    # per group (spec, layers, block size, page bytes, blocks charged per
    # request and their scratch part) with the pool size and concurrency.
    # Logging only. Default OFF.
    "VLLM_KV_PLAN_LOG": lambda: bool(int(os.getenv("VLLM_KV_PLAN_LOG", "0"))),
    # Re-order the multi-stream shared-expert overlap in the MoE runner.
    # Upstream enqueues the shared experts on the aux stream *before* the gate
    # and the routed dispatch, then joins after the routed kernels. Because the
    # aux branch is submitted first it is already resident when the main stream
    # finally reaches the routed Marlin kernels, so on CUDA under breakable
    # cudagraphs almost none of it actually overlaps (a decode trace measured
    # ~4%). With this on, the runner instead marks the aux stream's start point
    # up front, enqueues the routed experts, and only then runs the shared
    # experts on the aux stream, joining before the output is read.
    # Enqueue order only: the same kernels read the same tensors and the join
    # still precedes every read of the shared-expert output, so the numerics
    # are unchanged. It engages only where the multi-stream order was already
    # chosen, i.e. at or below VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    # tokens (decode), never for the 1152-token prefill chunks.
    # Default OFF until it has its own A/B on the cards.
    "VLLM_SHARED_EXPERT_REORDER": lambda: bool(
        int(os.getenv("VLLM_SHARED_EXPERT_REORDER", "0"))
    ),
    # Fair chunked prefill (`--prefill-chunk-with-decodes CAP`): a prefill
    # whose whole remaining prompt is at most CAP tokens is scheduled in full
    # within the ordinary token budget, not in capped slices, and is not
    # charged to the per-step prefill cap or counted in the partial-prefill
    # split (long prompts keep their share). A burst of short prompts beside a
    # decoding stream is then admitted in one step instead of several. Long
    # prompts keep the cap until their last <= CAP tokens. No effect without
    # the cap or on Mamba align-mode models. Default OFF until measured.
    "VLLM_FAIR_PREFILL_SHORT_BYPASS": lambda: bool(
        int(os.getenv("VLLM_FAIR_PREFILL_SHORT_BYPASS", "0"))
    ),
    # Pipeline parallelism: cap each scheduling step at ceil(decoding
    # requests / pp_size) decode requests so the pp_size in-flight
    # micro-batches share the decodes instead of one taking them all and the
    # others idling (a decode request is ineligible until its results return
    # pp_size steps later). No effect with pipeline_parallel_size 1 or with
    # at most one decoding request. Default OFF until measured on the cards.
    "VLLM_PP_SPREAD_DECODES": lambda: bool(
        int(os.getenv("VLLM_PP_SPREAD_DECODES", "0"))
    ),
    # Pipeline parallelism (V2 runner, TP=1): send each stage hand-off as one
    # flat device transfer instead of one per tensor (the GLM-5.3-Flash
    # boundary carries the residual streams plus drafter aux states). The
    # per-step metadata still travels and is checked. Default OFF.
    "VLLM_PP_PACKED_HOP": lambda: bool(int(os.getenv("VLLM_PP_PACKED_HOP", "0"))),
    # With VLLM_PP_PACKED_HOP: after a first-hop handshake, send no per-step
    # metadata; the receiver's persistent buffer gives the layout and the
    # scheduler output gives the row count (only the scheduled rows travel).
    # Default OFF.
    "VLLM_PP_HOP_NO_METADATA": lambda: bool(
        int(os.getenv("VLLM_PP_HOP_NO_METADATA", "0"))
    ),
    # Pipeline parallelism with speculative decoding: the first stages wait
    # only for the sampled tokens and counts before preparing the next step,
    # and for the draft tokens (sent after the drafter ran) only right before
    # the forward. Default OFF.
    "VLLM_PP_SPLIT_DRAFT_EVENT": lambda: bool(
        int(os.getenv("VLLM_PP_SPLIT_DRAFT_EVENT", "0"))
    ),
    # Pipeline parallelism with a DFlash2 drafter: run the drafter's tail
    # (candidate lm_head pass, top-k, selector, walk) on this earlier stage
    # instead of the last one, which then only runs the drafter's layers.
    # Needs no KV cache on that stage, one lm_head + selector copy. Draft
    # tokens are bit-identical. -1 (default) = off.
    "VLLM_PP_DRAFT_TAIL_STAGE": lambda: int(
        os.getenv("VLLM_PP_DRAFT_TAIL_STAGE", "-1")
    ),
    # GLM-5.3-Flash + DFlash under PP: each stage multiplies its own aux
    # hidden states by their slice of the drafter's input projection (fc) and
    # forwards one fp32 partial sum instead of relaying the aux states; the
    # last stage uses the sum as the fc output. Same arithmetic up to the fp32
    # summation order. Default OFF.
    "VLLM_GLM5_PP_FOLD_DRAFT_FC": lambda: bool(
        int(os.getenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "0"))
    ),
    # Split-block Marlin W4A16 MoE prefill on sm_80 for 288 routed experts
    # (vllm_sm80/kernels/marlin_moe/prefill.py): align the routed rows into
    # one block list per size 64/48/32/16 (cheapest cover per expert) instead
    # of padding every expert to one block size, and run each Marlin GEMM once
    # per list. The expert width per card picks the shape: whole experts
    # (N=2048, pipeline parallel with TP=1), with the list GEMMs on the
    # optional library's prefill_tile_gemm tiles ((64,512) on the 64/48/32-row
    # lists of both GEMMs, the optimistic fast dequant with tiles above a 2^-5
    # scale recomputed by the regular kernel; without the library or its op the
    # released kernels); or tensor-parallel shards on the released kernels:
    # N=512 (tensor parallel 4) and N=1024 (PP=2 with TP=2 inside each stage).
    # Taken for M >= the moe_prefill/marlin_min_rows row of the sm80
    # tuning table at that N. The split cover changes stream-K boundaries and
    # reduction order, so output is not bitwise equal to fused_marlin_moe.
    # Everything else keeps fused_marlin_moe. Off by default.
    "VLLM_SM80_MARLIN_PREFILL": lambda: bool(
        int(os.getenv("VLLM_SM80_MARLIN_PREFILL", "0"))
    ),
    # Optional prebuilt sm_80 Marlin decode kernels. Enabling the flag
    # requires vllm_sm80._ampere_marlin_C; startup fails if it is missing or
    # incompatible.
    "VLLM_AMPERE_MARLIN_DECODE_CUDA": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_DECODE_CUDA", "0"))
    ),
    # With VLLM_AMPERE_MARLIN_DECODE_CUDA=1: run the tensor-parallel (N=512)
    # decode GEMMs through the library's decode_gemm_v2 (one-FMA exact
    # dequant, 4-row chunks at 2-7 rows, zero padding fragments at 32 rows).
    # Same per-output summation order as decode_gemm_orig, so the output is
    # bitwise equal to it while every weight scale lies in [2^-110, 2^10) or
    # is zero (checked once per layer) and no nonzero weight-activation
    # product is below 2^-110 (vllm_sm80/kernels/marlin_moe/decode.py).
    # Without the op, at other widths, or out of that scale range,
    # decode_gemm_orig runs. Off by default.
    "VLLM_AMPERE_MARLIN_DECODE_V2": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_DECODE_V2", "0"))
    ),
    # Run each MLA layer's two thin-GEMM concat groups (q_b_proj + the
    # indexer's wq_b on q_c; fused_qkv_a_proj + the indexer's kpool gate on
    # hidden_states) as one launch per group, paired inside the thin-GEMM
    # dispatch (vllm_sm80/kernels/thin_gemm/concat.py). Every member keeps
    # its solo schedule, so each output is bitwise the separate call. Needs
    # VLLM_THIN_GEMM, sm_80 and compilation mode NONE. Default OFF: unset,
    # the pairing module is never imported.
    "VLLM_THIN_GEMM_CONCAT": lambda: bool(
        int(os.getenv("VLLM_THIN_GEMM_CONCAT", "0"))
    ),
    # Send the thin-GEMM calls of the measured Qwen3.8 decode shapes (seven
    # (N, K): hyper-connection down/up and final down, router, shared-expert
    # gate, GDN in_proj_ba, PLE kv_proj) at M <= 32 that are captured into a
    # CUDA graph to the Gluon thin GEMM (vllm_sm80/kernels/thin_gemm/gluon.py,
    # thin_gemm_gluon.py); once it is active, every call outside capture, of
    # any shape, takes F.linear (VLLM_THIN_GEMM's Triton calls too). Needs
    # VLLM_THIN_GEMM, sm_80 with 74 SMs, the Triton Gluon ampere API,
    # compilation mode NONE and CUDA graphs for decode and mixed steps (no
    # enforce_eager); anything else keeps the VLLM_THIN_GEMM path. A warmup
    # failure fails the boot. Raises with VLLM_THIN_GEMM_CONCAT (the two do
    # not combine).
    # Default OFF: unset, the module is never imported.
    "VLLM_THIN_GEMM_GLUON": lambda: bool(
        int(os.getenv("VLLM_THIN_GEMM_GLUON", "0"))
    ),
    # Diagnostic for VLLM_THIN_GEMM_GLUON: 0 (off) or a served row count
    # (1, 2, 4, 8, 16, 24, 32). Every Gluon thin-GEMM call with at most that
    # many rows takes that row count's schedule, so a row's result does not
    # depend on how many rows the call holds. Default 0.
    "VLLM_THIN_GEMM_GLUON_FORCE_M": lambda: int(
        os.getenv("VLLM_THIN_GEMM_GLUON_FORCE_M", "0")
    ),
    # Dense EXL3 launches of the tiny or small block shape with at most 1024
    # blocks, for bits_x2 10/12/14 with codebook 2 on sm_80, take
    # linear_out_splitk (csrc/exl3/dense_splitk.cu): K split in two halves
    # added atomically on blocks with separate decode warps, and an FP64
    # output transform. Results are deterministic but differ from
    # linear_out's in the last bits. Set once per process before any launch
    # (vllm_sm80/kernels/exl3/ops.py). Default OFF: unset, every dense call
    # takes linear_out.
    "VLLM_EXL3_DENSE_SPLITK": lambda: bool(
        int(os.getenv("VLLM_EXL3_DENSE_SPLITK", "0"))
    ),
    # With VLLM_EXL3_DENSE_SPLITK on, dense EXL3 launches of 1..32 rows of
    # the validated matrices (one Qwen3.8-Flash-Next rank at TP2, bits_x2
    # 10/12/14 with codebook 2; the LM head included) on sm_80 with 74 SMs at
    # TP2 and PP1 run on tensor-core
    # blocks inside linear_out_splitk (csrc/exl3/dense_mma.cu): activations
    # split into FP16 hi + lo, FP32 accumulation, K split inside and across
    # blocks with deterministic adds. Results differ from linear_out's in
    # the last bits. Set once per process before any launch
    # (vllm_sm80/kernels/exl3/dense_mma.py). Default OFF: unset, every launch
    # keeps the split-K operator's path.
    "VLLM_EXL3_DENSE_MMA": lambda: bool(int(os.getenv("VLLM_EXL3_DENSE_MMA", "0"))),
    # Diagnostic for VLLM_EXL3_DENSE_MMA: "MT,KW,SPLITS" (MT 2 or 4 column
    # tiles per warp, KW 1/2/4/8 warps along K per column group, SPLITS 1 or
    # 2 blocks along K) forces that plan on every launch the tensor-core
    # blocks take, in place of the per-shape choice. Unset: the shipped
    # choice.
    "VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE": lambda: os.getenv(
        "VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", ""
    ),
    # Qwen EXL3 checkpoints (vllm_sm80/kernels/exl3/quant.py).
    "VLLM_EXL3_PLE_PLACEMENT": lambda: _env_with_choices(
        "VLLM_EXL3_PLE_PLACEMENT", "ssd", ["ssd", "ram", "device"]
    )(),
    # An override sidecar's manifest (tools/sm80/build_override_sidecar.py):
    # the plain tensors it lists load from its safetensors file, in the source
    # checkpoint's dtype, in place of an EXL3 export's lower-precision copies
    # (vllm_sm80/kernels/exl3/override.py). Unset: the export's tensors.
    "VLLM_EXL3_OVERRIDE": lambda: os.getenv("VLLM_EXL3_OVERRIDE", ""),
    # sha256 of the VLLM_EXL3_OVERRIDE manifest; required when it is set.
    "VLLM_EXL3_OVERRIDE_SHA256": lambda: os.getenv(
        "VLLM_EXL3_OVERRIDE_SHA256", ""
    ),
    # Load an EXL3 checkpoint whose quantization_config routes module classes
    # to EXL3, unquantized or a second registered method (module_quantization,
    # non_routed_quantization; vllm_sm80/kernels/exl3/hybrid.py), its tensors
    # selected by its index. Unset: such a config is refused; EXL3
    # checkpoints without those keys load as before either way.
    "VLLM_EXL3_HYBRID": lambda: bool(int(os.getenv("VLLM_EXL3_HYBRID", "0"))),
    # Sum the BF16 partials of the TP ranks that a layer all-reduces itself
    # (DeepSeek-V4.1's attention wo_b) with one rounding per element: each
    # element accumulated in FP32 in a fixed order and rounded once to BF16,
    # as the switch-off FP32 all-reduce does, at half its bytes. The message
    # stays BF16 wherever its transport rounds once: the host-staged or custom
    # all-reduce when it takes the message whole (decode), else on eager steps
    # the copy-engine ring's owner sum (VLLM_SM80_ALLREDUCE) or the custom
    # all-reduce in pieces below its buffer (prefill). Any other message is
    # all-reduced in FP32 and rounded once (CUDA graph capture of a message
    # too large for the custom all-reduce, a backend dispatched before the
    # custom all-reduce). The FP32 sum of four BF16 partials is exact when
    # their magnitudes lie within a factor 2^14 of each other, so such
    # elements get the FP32 path's bits in any order; others can differ in
    # the last BF16 bit where a transport sums in another order than the FP32
    # path's (the ring's owner sum, the pieces). The function
    # (sm80_attention.round_once_all_reduce) takes any consumer's partial.
    # DeepSeek-V4.1's MoE output partial (routed experts plus the shared-expert
    # shard) is also summed by the layer (skip_final_all_reduce): on eager
    # steps a message too large for the custom all-reduce whole goes over the
    # copy-engine ring's owner sum (VLLM_SM80_ALLREDUCE) or the custom
    # all-reduce in pieces, rounded once, instead of NCCL's BF16 ring (a TP4
    # element rounded three times); every other MoE message takes the
    # communicator's usual path. Default OFF: unset, wo_b partials are
    # all-reduced in FP32 and rounded once and the runner all-reduces the MoE
    # output. Read in vllm_sm80/models/deepseek_v41/sm80_attention.py and
    # sm80_moe.py.
    "VLLM_SM80_ALLREDUCE_ROUND_ONCE": lambda: bool(
        int(os.getenv("VLLM_SM80_ALLREDUCE_ROUND_ONCE", "0"))
    ),
    # In eager steps of at least 1024 rows, run each target decoder layer's
    # wo_b, FFN and both TP all-reduces in two row slices, the all-reduces on
    # a side stream, so one slice's all-reduce runs beside the other slice's
    # compute (DeepSeek-V4.1: sm80_split_reduce.py). Each slice's message
    # takes the path the whole message would; steps whose slices the
    # host-staged or custom all-reduce would take run whole. Not bit-equal to
    # the unsplit step (NCCL's ring and the two-stage custom all-reduce both
    # sum an element in an order set by its place in the message); judged by
    # the V2 prefill dNLL check. Default OFF: unset, every layer runs whole.
    # Read in vllm_sm80/models/deepseek_v41/sm80_model.py and sm80_moe.py.
    "VLLM_SM80_PREFILL_REDUCE_OVERLAP": lambda: bool(
        int(os.getenv("VLLM_SM80_PREFILL_REDUCE_OVERLAP", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: run the decoder layers' delayed mHC (the fused
    # post + pre at every sublayer boundary and the pre-only entries) through
    # the sm_80 mHC kernel A (ops/sm80_mhc_fused.py: post step, projection and
    # squared-sum partials) instead of the reference's FP32 torch formulas
    # (ops/sm80_mhc.py, about 155 kernels per call), with the reference's BF16
    # rounding points; the finish stays the torch formulas unless
    # VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH. Default OFF: unset, the torch formulas run.
    # Read in vllm_sm80/models/deepseek_v41/sm80_model.py.
    "VLLM_DEEPSEEK_V41_FUSED_MHC": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_FUSED_MHC", "0"))
    ),
    # With VLLM_DEEPSEEK_V41_FUSED_MHC: the finish after kernel A (coefficients,
    # Sinkhorn, collapse, RMSNorm) as one Triton kernel, two kernels per call.
    # Default OFF. Read in vllm_sm80/models/deepseek_v41/ops/sm80_mhc_fused.py.
    "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH", "0"))
    ),
    # With VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH: the finish v2 in its validated
    # configuration (hidden 5120, 20 Sinkhorn iterations, 1..2048 rows, with the
    # norm, sm_80): kernel A with its staging reuse from 17 rows, then the Gluon finish
    # (ops/sm80_mhc_finish_v2.py: FP64 reciprocals by MUFU approximation and
    # Newton steps, one-pass collapse) instead of the Triton finish kernel.
    # Order-changing. Default OFF. Read in
    # vllm_sm80/models/deepseek_v41/ops/sm80_mhc_fused.py.
    "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2", "0"))
    ),
    # With VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH: inside a decoder layer's forward,
    # the mHC finish's coefficient CTAs (post, comb with its Sinkhorn, the next
    # pre-mix: first read at the next sublayer boundary) of up to 64 rows run on
    # a side stream beside the sublayer's TP all-reduce, forked as it starts and
    # joined as it returns; the layer-input CTAs stay in place. Same kernels and
    # inputs, so the same bytes. Default OFF. Read in
    # vllm_sm80/models/deepseek_v41/ops/sm80_mhc_side.py.
    "VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the MoE gate and routing of calls of 1 to 2047
    # BF16 rows on the BF16 tensor cores (ops/sm80_router_mma.py): the gate
    # weight held once, in BF16 (its BF16-valued checkpoint values exactly;
    # the gate GEMV and the FP32 GEMM widen it exactly, so their bytes are
    # unchanged), with exact BF16 products, FP32 sums
    # per K chunk and FP64 across chunks, rounded once to FP32; the top-k
    # selected in FP32 as the fused route, the weights in FP64 rounded once,
    # from exactly recomputed logits for calls of up to 64 rows. Not the FP32
    # GEMM's bytes: closer to the FP64 routing weights. Other calls keep the
    # gate GEMV or the FP32 gate and the route. Default OFF. Read in
    # vllm_sm80/models/deepseek_v41/sm80_moe.py.
    "VLLM_DEEPSEEK_V41_ROUTER_MMA": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_ROUTER_MMA", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the MoE routing after the FP32 gate GEMM
    # (sqrt-softplus, bias select and add, top-k, gather, renormalisation,
    # scale, int32 ids: fifteen torch kernels per layer) as one Triton kernel
    # (ops/sm80_route.py) with the torch ops' arithmetic and order, so the
    # same bytes. Default OFF. Read in vllm_sm80/models/deepseek_v41/sm80_moe.py.
    "VLLM_DEEPSEEK_V41_FUSED_ROUTE": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_FUSED_ROUTE", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the DSpark drafter's context-KV prologue per
    # draft layer (kv_norm RMSNorm, RoPE, FP8 qdq and the window-cache write
    # after the wkv projection: 26 torch and Triton kernels per layer) as one
    # Triton kernel (ops/sm80_draft_prologue.py) with the torch ops'
    # arithmetic and order (the RMSNorm sum in torch's CUDA reduction order),
    # so the same cache bytes. Default OFF. Read in
    # vllm_sm80/models/deepseek_v41/sm80_attention.py.
    "VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the MoE gate GEMM of calls of at most 64 rows
    # (decode, speculative verify, draft) as one Triton kernel
    # (ops/sm80_gate_gemv.py) that reads the BF16 input as is and computes the
    # logits in FP64, rounded once to FP32, in place of the FP32 copy of the
    # input, the FP32 GEMM and its split-K reduce. Not the FP32 GEMM's bytes:
    # the correctly rounded FP32 logits, except where the exact value lies
    # within the FP64 summation error of an FP32 rounding midpoint. Default
    # OFF. Read in vllm_sm80/models/deepseek_v41/sm80_moe.py.
    "VLLM_DEEPSEEK_V41_GATE_GEMV": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_GATE_GEMV", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the grouped MXFP8 wo_a (one Marlin GEMM per
    # group and a concat) as one grouped Triton W8A16 GEMM over all groups
    # that writes the [T, groups * o_lora_rank] result directly
    # (ops/sm80_wo_a_grouped.py). It holds the loaded FP8 codes and E8M0
    # scales instead of Marlin's repack (the same bytes); the decoded weights
    # are Marlin's, the FP32 sums run in another order, so outputs change
    # within rounding. Default OFF. Read in
    # vllm_sm80/models/deepseek_v41/sm80_attention.py.
    "VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the attention core of decode, speculative verify
    # and draft steps (the norms and window row of prepare_layer, the query
    # RoPE, sparse attention, output RoPE and qdq of attend_layer) as two
    # fused calls per layer (ops/sm80_attn_fused.py), same rounding points.
    # Default OFF. Read in vllm_sm80/models/deepseek_v41/sm80_attention.py.
    "VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the routed experts of the decoder layers run MoE
    # calls of 65..2048 rows in prefill steps on the tensor cores
    # (csrc/exl3/moe_prefill_v2.cu, ops.moe_out_prefill_v2) at tensor parallel
    # size 4, with every routed matrix of a layer at bits_x2 6 or 4, codebook 2,
    # and 16-byte-aligned trellis bases; with a parallel drafter (DFlash,
    # DSpark) a step counts as a prefill step only when some request has more
    # than 1 + 2k tokens, its decode pass's widest query
    # (vllm_sm80/kernels/exl3/quant.py _prefill_v2_decode_len). Other calls
    # and every call of a decode step keep their kernels. Results differ from
    # moe_out in the last bits. Default OFF: unset, prefill calls take moe_out.
    # Read in vllm_sm80/models/deepseek_v41/sm80_moe.py.
    "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_MOE_PREFILL_TC", "0"))
    ),
    # Complete decode/verify attention preparation, stable index selection,
    # and ordered-value output epilogue, for calls of up to 48 padded rows
    # (sm80_attn_pipeline.MAX_ROWS). Independent of the qualified seed.
    "VLLM_DEEPSEEK_V41_ATTN_PIPELINE": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_ATTN_PIPELINE", "0"))
    ),
    # DeepSeek-V4.1 on sm_80, the attention pipeline's exact index selection:
    # a call of at most 48 rows as one Triton launch (one program per
    # 8192-column part of a row, the last part to finish merging the parts'
    # candidates) in place of fifteen, with the same output bytes. Default
    # OFF. Read in vllm_sm80/models/deepseek_v41/ops/sm80_index_topk.py.
    "VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH", "0"))
    ),
    # Greedy argmax over a vocabulary-parallel head without the all-gather of
    # the logits, for greedy target sampling and greedy drafting. Each TP rank
    # takes the argmax of its own vocabulary shard, the ranks exchange one
    # (value, index) pair per 4096-column chunk through 16-byte all-reduce
    # slots, and the pick is torch.argmax's over the gathered row, ties and
    # NaN included (the first index of the largest value, the first NaN, -0.0
    # equal to +0.0). Precision: none changes.
    # - Target: when every request of a step is greedy (temperature 0) and
    #   nothing reads the logits beyond their argmax (no logprobs, logits
    #   processing, grammar bitmask, top-k/top-p, NaN count, sampling mask or
    #   watermarking), the samplers get rows holding 0.0 at the token and -inf
    #   elsewhere; every other step gathers. A greedy-only step with
    #   batch-sharded sampling samples unsharded from those rows. Needs a
    #   target class with ShardArgmaxLogits (DeepSeek-V4.1, GLM-5.3, Qwen4Exp
    #   on sm_80) and TP > 1. It selects the fork GPU worker
    #   (vllm_sm80/worker/graph_mem_worker.py, V2 runner; another worker or
    #   runner is refused at boot) and warms the kernels in the dummy sampler
    #   run.
    # - Drafts: the DSpark Markov sampler takes its greedy drafts per rank
    #   (no probabilistic drafting, draft top-k, acceptance estimator or
    #   reduced draft vocabulary); so does a greedy drafter sampling through
    #   vLLM's sample_draft (MTP, EAGLE, DFlash) whose draft model has
    #   compute_logits_local (Qwen4Exp's MTP on sm_80), set up at load and
    #   refused with speculative_config.use_local_argmax_reduction (two
    #   per-rank argmaxes) or an acceptance estimator; a candidate drafter
    #   (DFlash2) has no draft argmax over the vocabulary.
    # A config it cannot serve is refused at boot with the reason. Default
    # OFF: unset, the full-vocabulary all-gather and torch.argmax run. Read in
    # vllm_sm80/worker/target_argmax.py and vllm_sm80/spec_decode/shard_argmax.py.
    "VLLM_SM80_SHARD_ARGMAX": lambda: bool(
        int(os.getenv("VLLM_SM80_SHARD_ARGMAX", "0"))
    ),
    # Drop the compressed PLE table from the page cache at setup so a run
    # starts cold (reads come from storage). Residency is logged either way.
    "VLLM_EXL3_PLE_COLD": lambda: bool(int(os.getenv("VLLM_EXL3_PLE_COLD", "0"))),
    # Retries (1 ms backoff, doubling) for a failed or truncated PLE row read
    # before the requests using that row are finished with an error.
    "VLLM_EXL3_PLE_READ_RETRIES": lambda: int(
        os.getenv("VLLM_EXL3_PLE_READ_RETRIES", "3")
    ),
    # Diagnostic: time the host side of every SSD PLE prefetch (the wait for
    # the token ids copied to the host, then the row reads) and log a summary
    # per rank every 256 prefetches and at close. Off by default.
    "VLLM_EXL3_PLE_TIMING": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_TIMING", "0"))
    ),
    # Read the SSD PLE rows at the PLE layer instead of before the layer
    # ahead of it, so the host wait for the token ids and the row reads
    # overlap that layer on the GPU. Same rows, kernels and stream order for
    # the consumer. Off by default.
    "VLLM_EXL3_PLE_OVERLAP": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_OVERLAP", "0"))
    ),
    # Dense EXL3 launches of prefill steps with more than 64 rows and at
    # least 16 output tiles, for bits_x2 10/12/14 with codebook 2 on sm_80
    # with 74 multiprocessors, take linear_out_prefill
    # (csrc/exl3/dense_prefill_tc.cu): tensor-core products on an exact
    # integer grid, FP32 sums in k order, a deterministic k split of the last
    # wave of tiles. Launches split-K splits keep split-K. Results differ from
    # linear_out's in the last bits. Set once per process before any launch
    # (vllm_sm80/kernels/exl3/dense_prefill_tc.py). Default OFF: unset, every
    # dense call keeps today's kernels.
    "VLLM_EXL3_DENSE_PREFILL_TC": lambda: bool(
        int(os.getenv("VLLM_EXL3_DENSE_PREFILL_TC", "0"))
    ),
    # Diagnostic for VLLM_EXL3_DENSE_PREFILL_TC: "planned" (default) splits
    # the last wave's k range when the planner predicts a gain; "whole" runs
    # whole tiles only, to isolate the k split in a replay.
    "VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE": lambda: _env_with_choices(
        "VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", "planned", ["planned", "whole"]
    )(),
    # Read the SSD PLE rows of a step in one native call that releases the
    # GIL (csrc/exl3/ple_read.cpp) instead of a Python loop on a thread pool.
    # Same rows, bytes and kernels for the consumer; its failures are a
    # subset of the pool path's (one extra read attempt for rows not in the
    # page cache). Off by default.
    "VLLM_EXL3_PLE_NATIVE_READ": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_NATIVE_READ", "0"))
    ),
    # Read the SSD PLE rows of a step in a host function queued on the PLE
    # stream (csrc/exl3/ple_read.cpp, the native reader's core), between the
    # copy of the ids to the host and the copy of the rows to the device, so
    # the host never waits for the ids. Same rows and bytes as the native
    # reader; failed rows reach the consumer poisoned in the same step and
    # are logged at the next step. Takes precedence over
    # VLLM_EXL3_PLE_OVERLAP and VLLM_EXL3_PLE_NATIVE_READ for SSD tables.
    # Off by default.
    "VLLM_EXL3_PLE_HOSTFN_READ": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_HOSTFN_READ", "0"))
    ),
    # Start Engram lookups, row gathers and WKV projections before decoder
    # layers on the registered auxiliary stream. Its EXL3 scratch is disjoint
    # from the main stream and ordered before shared-expert GEMMs. TP only;
    # no sequence parallelism, batched Engram projection or microbatching.
    "VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER", "0"))
    ),
    # Every TP rank holds every rank's Engram wkv shard (gathered once after
    # loading; the layer's own shard is a view of it) and projects steps of
    # more rows than any decode step or CUDA graph capture from all shards,
    # without the all-gather of the output: the same GEMM per shard, the
    # gathered bytes. Smaller steps keep the sharded GEMM and the all-gather.
    # Costs the other shards' bytes on every card. TP only; no sequence
    # parallelism or batched Engram projection
    # (vllm_sm80/models/deepseek_v41/sm80_engram_wkv.py). Default OFF.
    "VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL", "0"))
    ),
    # DeepSeek-V4.1 on sm_80: the sliding-window metadata of every SWA KV
    # cache group of a step (one group per layer) in one Triton launch
    # instead of three to five launches per group, the same bytes in the same
    # builder buffers (vllm_sm80/models/deepseek_v41/sm80_swa_meta_batch.py).
    # Eager step preparation only; captures and other shapes build per group.
    # Default OFF. Read in vllm_sm80/models/deepseek_v41/sm80_attention.py.
    "VLLM_DEEPSEEK_V41_SWA_META_BATCH": lambda: bool(
        int(os.getenv("VLLM_DEEPSEEK_V41_SWA_META_BATCH", "0"))
    ),
    # Where the sm_80 Engram storage keeps the E4M3 values of its owned table
    # rows (vllm_sm80/models/deepseek_v41/sm80_engram.py): "ssd" reads them
    # on demand from the base-checkpoint shards through a bounded host row
    # cache; "ram" reads them whole at startup, past the page cache, into
    # host memory of their exact size (~45.8 GiB per rank of DeepSeek V4.1
    # Flash at TP4) and refuses to start when the host or a cgroup memory
    # limit cannot hold them. Scales are resident host memory either way.
    "VLLM_ENGRAM_PLACEMENT": lambda: _env_with_choices(
        "VLLM_ENGRAM_PLACEMENT", "ssd", ["ssd", "ram"]
    )(),
    # Host row cache of the sm_80 Engram SSD placement, in GiB per rank,
    # split evenly over the Engram tables; includes its tags. 0 disables the
    # cache (every row is read from the file). Unused by "ram".
    "VLLM_ENGRAM_ROW_CACHE_GIB": lambda: float(
        os.getenv("VLLM_ENGRAM_ROW_CACHE_GIB", "1")
    ),
    # Fill the host row cache of the sm_80 Engram SSD placement at startup:
    # one sequential pass over each rank's value extent per table, read past
    # the page cache, stores rows in empty cache ways only and stops once
    # every way holds a row (vllm_sm80/models/deepseek_v41/sm80_engram.py).
    # Startup reads up to the whole extent (~24.6 GB per table and rank of
    # DeepSeek V4.1 Flash at TP4); the fetch counters do not count the
    # stored rows. Unused by "ram" and by a zero cache. Off by default.
    "VLLM_ENGRAM_ROW_CACHE_PRELOAD": lambda: bool(
        int(os.getenv("VLLM_ENGRAM_ROW_CACHE_PRELOAD", "0"))
    ),
    # Read the rows the sm_80 Engram row cache misses with O_DIRECT through
    # aligned 4 KiB spans instead of buffered 256 B reads, so the misses
    # leave no pages in the page cache (about 4 KiB of page cache per missed
    # row otherwise). A file system that refuses O_DIRECT is refused at
    # startup; there is no buffered fallback. Unused by "ram". Off by
    # default.
    "VLLM_ENGRAM_DIRECT_READ": lambda: bool(
        int(os.getenv("VLLM_ENGRAM_DIRECT_READ", "0"))
    ),
    # Socket of a table segment holder (vllm_sm80.table_segment) that
    # holds the Engram checkpoint files in memory: with placement "ram" each
    # rank maps its rows of the held files instead of reading them from
    # storage. A missing holder, or a file that is not the one it read, is
    # refused at startup; there is no fallback to reading. Refused with
    # "ssd". Default empty: off.
    "VLLM_ENGRAM_SEGMENT": lambda: os.getenv("VLLM_ENGRAM_SEGMENT", ""),
    # Diagnostic: log the counters of each sm_80 Engram table (rows, cache
    # hits and misses, page-cache and storage bytes, host read and wait
    # time) every N steps, and at close and exit
    # (vllm_sm80/models/deepseek_v41/sm80_engram.py). The counters are
    # kept either way. Default 0: never logged.
    "VLLM_ENGRAM_TELEMETRY_STEPS": lambda: int(
        os.getenv("VLLM_ENGRAM_TELEMETRY_STEPS", "0")
    ),
    # EXL3 routed experts on the tensor cores (csrc/exl3/moe_v2.cu) for calls
    # of at most 64 rows of the layers ops.MOE_V2_CONFIGS lists (hidden 2560,
    # intermediate 640, top-10, 256 of 512 experts local, bits_x2 10; hidden
    # 5120, intermediate 2304, top-6, 96 of 384 experts local, bits_x2 6 or 4),
    # every local expert matrix of a layer with one bits_x2 and codebook 2, on
    # sm_80 (vllm_sm80/kernels/exl3/quant.py). Results differ from the default
    # kernels in the last bits. Default OFF: unset, every call takes moe_out.
    "VLLM_EXL3_MOE_V2": lambda: bool(int(os.getenv("VLLM_EXL3_MOE_V2", "0"))),
    # With VLLM_EXL3_MOE_V2: its calls take moe_out_v2_fold, which runs the
    # activation at the end of the gate/up launch and the combine at the end
    # of the down launch (completion counters; csrc/exl3/moe_v2.cu FoldArgs):
    # three launches per call instead of five, the same output bits. Needs an
    # EXL3 extension built from this tree. Default OFF. Read in
    # vllm_sm80/kernels/exl3/quant.py.
    "VLLM_EXL3_MOE_V2_FOLD": lambda: bool(
        int(os.getenv("VLLM_EXL3_MOE_V2_FOLD", "0"))
    ),
    # EXL3 routed experts on the tensor cores (csrc/exl3/moe_v2.cu,
    # DraftTiling) for calls of at most 64 rows of the configuration
    # ops.MOE_V2_DRAFT_CONFIGS names: the DSpark drafter's experts of
    # DeepSeek-V4.1-Flash (hidden 5120, intermediate 2304, top-3, 32 of 128
    # experts local, every local expert matrix bits_x2 8 with codebook 2), on
    # sm_80; independent of VLLM_EXL3_MOE_V2, which keeps the other
    # configurations, and never folded (VLLM_EXL3_MOE_V2_FOLD does not apply).
    # The drafter's experts split across tensor-parallel ranks (intermediate
    # 640 or 512, all 128 local) keep moe_out with the flag on. Results
    # differ from moe_out in the last bits. Needs an EXL3 extension
    # built from this tree. Default OFF: unset, those calls take moe_out. Read
    # in vllm_sm80/kernels/exl3/quant.py.
    "VLLM_EXL3_MOE_V2_DRAFT": lambda: bool(
        int(os.getenv("VLLM_EXL3_MOE_V2_DRAFT", "0"))
    ),
    # Diagnostic for VLLM_EXL3_MOE_V2_DRAFT: 0 (off), 8 or 16. Every drafter
    # row tile takes at least the lane kernel of a tile of that many rows
    # (moe_v2.cu DraftLanes; by default a tile of 1..4 rows sums the low part
    # of its activations outside the MMA, a larger one inside), so a row's
    # result does not depend on how many rows of the call share its expert.
    # Set once per process (before the warm-up and graph capture). Default 0.
    "VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS": lambda: int(
        os.getenv("VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS", "0")
    ),
    # With VLLM_EXL3_MOE_V2: the moe_out_v2 calls of the whole-expert layers
    # of ops.MOE_V2_TRY_CONFIG (hidden 5120, intermediate 2304, top-6, 96 of
    # 384 experts local, bits_x2 6 or 4, 1..64 rows) take moe_out_v2_try
    # (csrc/exl3/moe_v2_try.cu) in place of moe_out_v2 or moe_out_v2_fold:
    # the lane GEMMs' trellis words through a per-warp cp.async shared ring,
    # the same arguments, workspace contract and output bits. Needs an EXL3
    # extension built from this tree. Default OFF. Read in
    # vllm_sm80/kernels/exl3/ops.py (moe_v2_try_open).
    "VLLM_EXL3_MOE_V2_RING": lambda: bool(
        int(os.getenv("VLLM_EXL3_MOE_V2_RING", "0"))
    ),
    # EXL3 routed experts on the tensor cores (csrc/exl3/moe_prefill.cu) for
    # calls of 65..3460 rows in prefill steps (some request scheduled more
    # than 1 + num_speculative_tokens tokens), with the hidden-2560
    # configuration VLLM_EXL3_MOE_V2 covers (ops.MOE_PREFILL_CONFIGS) and
    # 16-byte-aligned expert trellis bases
    # (vllm_sm80/kernels/exl3/quant.py). Other calls
    # and every call of a decode step keep their kernels. Results
    # differ from moe_out in the last bits. Default OFF: unset, prefill calls
    # take moe_out.
    "VLLM_EXL3_MOE_PREFILL_TC": lambda: bool(
        int(os.getenv("VLLM_EXL3_MOE_PREFILL_TC", "0"))
    ),
    # With VLLM_DEEPSEEK_V41_MOE_PREFILL_TC: moe_out_prefill_v2's calls of the
    # whole-expert configuration (hidden 5120, intermediate 2304, top-6, 96
    # of 384 experts local, bits_x2 6 or 4, 65..2048 rows) run
    # moe_out_prefill_try (csrc/exl3/moe_prefill_try.cu) as one launch
    # sequence per call instead of one per 512-row chunk, so each touched
    # expert is decoded once per call; the same arithmetic, bitwise equal.
    # Needs an EXL3 extension built from this tree. Default OFF. Read once
    # at import in vllm_sm80/kernels/exl3/ops.py.
    "VLLM_EXL3_MOE_PREFILL_UNCHUNKED": lambda: bool(
        int(os.getenv("VLLM_EXL3_MOE_PREFILL_UNCHUNKED", "0"))
    ),
    # EXL3 routed experts: the trellis, suh and svh of a MoE layer's local
    # expert matrices are views into one device buffer per tensor kind
    # (vllm_sm80/kernels/exl3/quant.py expert_buffer_views) instead of one
    # allocation each, which leaves no allocator holes between them. Same
    # bytes and the same kernels. Default OFF: one allocation per tensor.
    "VLLM_EXL3_EXPERTS_CONTIGUOUS": lambda: bool(
        int(os.getenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", "0"))
    ),

    # Diagnostic: the path of a size cut-off table (schema sm80-tuning/1, see
    # vllm_sm80/tuning.py) read instead of the card's own table in
    # vllm_sm80/tuning/, for tuning runs. Unset by default.
    "VLLM_SM80_TUNING_FILE": lambda: os.getenv("VLLM_SM80_TUNING_FILE", ""),
    # Diagnostic: the path of a capture spec, a TOML file naming the
    # directory to which selected served kernel calls (inputs and outputs)
    # are saved and which calls (vllm_sm80/capture.py: its keys and their
    # defaults). Saving waits for the stream: diagnostic boots only. Default
    # empty: nothing is captured and nothing is patched.
    "VLLM_SM80_CAPTURE": lambda: os.getenv("VLLM_SM80_CAPTURE", ""),
    # Tests only: a directory of captured kernel inputs with one subdirectory
    # per capture family (kda, kda_pre, smla_pre, idx_glue, moe_pre with its
    # once weights), which the replay tests read
    # (tests/kernels/replay_inputs.py). Default empty: those tests skip.
    "VLLM_SM80_REPLAY_DIR": lambda: os.getenv("VLLM_SM80_REPLAY_DIR", ""),
    # Tests only: the most records a replay test reads from its family; 0
    # reads every record. Default unset: each test's own number.
    "VLLM_SM80_REPLAY_LIMIT": _replay_limit,
    # Development server only: the name of one switch in
    # vllm_sm80/dev_ab.py SWITCHES. Both of its arms are captured at boot and
    # POST /collective_rpc set_ab_arm (VLLM_SERVER_DEV_MODE=1) changes the arm
    # between requests. Costs graph memory and boot time; not for serving.
    # Unset by default.
    "VLLM_DEV_AB_SWITCH": lambda: os.getenv("VLLM_DEV_AB_SWITCH", ""),
    # Development server only: comma-separated pairs of
    # vllm_sm80.hot_swap.HOT_PAIRS, each a switch booted off. Every CUDA
    # graph holds each pair's graph sites in both variants side by side, and
    # POST /collective_rpc set_hot_arms (VLLM_SERVER_DEV_MODE=1) selects the
    # candidates between requests. Not for serving. Unset by default.
    "VLLM_HOT_SWAP": lambda: os.getenv("VLLM_HOT_SWAP", ""),
}


# Environment variables retired in this release, with what replaces them. A
# retired name never supplies a value (no aliases): setting one only logs a
# warning, once per process, when this module is imported. Remove this table,
# warn_retired_env_vars() and its call below in the next release.
RETIRED_ENV_VARS: dict[str, str] = {
    # Renamed: shared code is named after the feature, not the model.
    "VLLM_GLM5_DECODE_MOE_MAX_TOKENS": (
        "moved to the sm80 tuning table, moe_router/fused_route_max_rows"
    ),
    "VLLM_GLM5_FLA_PIN_AUTOTUNE": "renamed to VLLM_FLA_PIN_AUTOTUNE",
    "VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE": (
        "moved to the sm80 tuning table, allreduce/host_max_bytes"
    ),
    "VLLM_GLM5_MEM_ATTRIBUTION": "renamed to VLLM_MEM_ATTRIBUTION",
    "VLLM_GLM5_MOE_MASK_PADDING": (
        "renamed to VLLM_MOE_MASK_PADDING. Batches of up to "
        "moe_router/fused_route_max_rows rows of the sm80 tuning table (default "
        "8) are now left unmasked even with VLLM_AMPERE_DECODE_KERNELS off; a "
        "row of 0 masks every batch"
    ),
    "VLLM_GLM5_SHARED_EXPERT_REORDER": "renamed to VLLM_SHARED_EXPERT_REORDER",
    "VLLM_GLM5_STATE_INDEX_CHECK": "renamed to VLLM_STATE_INDEX_CHECK",
    "VLLM_GLM5_THIN_GEMM": "renamed to VLLM_THIN_GEMM",
    "VLLM_GLM5_THIN_GEMM_MAX_TOKENS": (
        "moved to the sm80 tuning table, dense_gemm/thin_max_rows"
    ),
    "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE": (
        "renamed to VLLM_CUSTOM_ALLREDUCE_PCIE_P2P"
    ),
    "VLLM_BUILD_AMPERE_MARLIN": "renamed to VLLM_AMPERE_MARLIN_BUILD",
    "VLLM_DSA_INDEXER_REF": "renamed to VLLM_INDEXER_DSA_REF",
    "VLLM_GLM5_DECODE_IDX_GLUE": "renamed to VLLM_AMPERE_DECODE_IDX_GLUE",
    "VLLM_GLM5_DECODE_KERNELS": "renamed to VLLM_AMPERE_DECODE_KERNELS",
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2": "renamed to VLLM_MOE_DECODE_ROUTE_V2",
    "VLLM_GLM5_DETERMINISTIC_MOE_ALIGN": "renamed to VLLM_MOE_DETERMINISTIC_ALIGN",
    "VLLM_GLM5_INDEXER_DECODE_RAW_K": "renamed to VLLM_INDEXER_DECODE_RAW_K",
    "VLLM_GLM5_MARLIN_DECODE_CUDA": "renamed to VLLM_AMPERE_MARLIN_DECODE_CUDA",
    "VLLM_GLM5_PP_KDA_PREFILL": "merged into VLLM_SM80_KDA_PREFILL",
    "VLLM_GLM5_PP_MARLIN_PREFILL": "merged into VLLM_SM80_MARLIN_PREFILL",
    "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED": "merged into VLLM_SM80_MARLIN_PREFILL",
    "VLLM_GLM5_PP_SPARSE_MLA_PREFILL": "merged into VLLM_SM80_SPARSE_MLA_PREFILL",
    "VLLM_GLM5_TOOL_CHOICE_NONE_MASK": "renamed to VLLM_TOOL_CHOICE_NONE_MASK",
    "VLLM_GLM5_TP4_KDA_PREFILL": "merged into VLLM_SM80_KDA_PREFILL",
    "VLLM_GLM5_TP4_MARLIN_PREFILL": "merged into VLLM_SM80_MARLIN_PREFILL",
    # Merged: one switch per operation across models.
    "VLLM_GLM5_TOPK_TIEFIX": (
        "merged into VLLM_SM80_TOPK_CANONICAL_TIES (all sparse indexers)"
    ),
    "VLLM_QWEN4_QSA_TOPK_CANONICAL": (
        "merged into VLLM_SM80_TOPK_CANONICAL_TIES (all sparse indexers)"
    ),
    "VLLM_GLM5_TOPK_SORTED": (
        "merged into VLLM_SM80_TOPK_SORTED (all sparse indexers; no longer "
        "needs the tie fix)"
    ),
    "VLLM_QWEN_QSA_TOPK_SORTED": (
        "merged into VLLM_SM80_TOPK_SORTED (all sparse indexers; default off)"
    ),
    # The shape picks the kernel.
    "VLLM_AMPERE_KDA_PREFILL_WIDE": (
        "merged into VLLM_SM80_KDA_PREFILL (64 or 16 heads per card, picked "
        "by the layer's head count)"
    ),
    "VLLM_AMPERE_KDA_PREFILL_NARROW": (
        "merged into VLLM_SM80_KDA_PREFILL (64 or 16 heads per card, picked "
        "by the layer's head count)"
    ),
    "VLLM_AMPERE_MARLIN_PREFILL_WIDE": (
        "merged into VLLM_SM80_MARLIN_PREFILL (whole experts or TP4 shards, "
        "picked by the expert width per card)"
    ),
    "VLLM_AMPERE_MARLIN_PREFILL_NARROW": (
        "merged into VLLM_SM80_MARLIN_PREFILL (whole experts or TP4 shards, "
        "picked by the expert width per card)"
    ),
    "VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED": (
        "merged into VLLM_SM80_MARLIN_PREFILL (the whole-expert lists run on "
        "the compiled tiles whenever the library has them)"
    ),
    "VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE": (
        "merged into VLLM_SM80_SPARSE_MLA_PREFILL (the 64-head kernel is "
        "picked by the head count per card)"
    ),
    # Split: one switch per operation.
    "VLLM_GLM5_PREFILL_KERNELS": (
        "split into VLLM_SM80_MHC_PREFILL (the mHC pre-norm GEMM) and "
        "VLLM_SM80_SPARSE_MLA_PREFILL (the sparse-MLA prefill attention)"
    ),
    "VLLM_GLM5_DECODE_KDA_WIDE": (
        "merged into VLLM_AMPERE_DECODE_KERNELS (64 heads per card take the "
        "fused KDA decode step at every measured shape)"
    ),
    "VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS": (
        "merged into VLLM_AMPERE_MARLIN_DECODE_CUDA (whole experts take the "
        "compiled decode at 4..8 rows)"
    ),
    "VLLM_GLM5_MARLIN_DECODE_PP_MULTI": (
        "merged into VLLM_AMPERE_MARLIN_DECODE_CUDA (whole experts take the "
        "compiled decode for multi-request rows up to the sm80 tuning table "
        "row moe_routed_gemm/marlin_multi_max_rows)"
    ),
    "VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE": (
        "renamed to VLLM_SM80_PREFILL_REDUCE_OVERLAP"
    ),
    "VLLM_GLM5_HOST_ALLREDUCE": (
        "merged into VLLM_SM80_ALLREDUCE, which picks the transport per message"
    ),
    "VLLM_HOST_ALLREDUCE": (
        "merged into VLLM_SM80_ALLREDUCE, which picks the transport per message"
    ),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS": (
        "merged into VLLM_SM80_ALLREDUCE, which picks the transport per message"
    ),
    "VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE": (
        "merged into VLLM_SM80_ALLREDUCE, which picks the transport per message"
    ),
    "VLLM_DEEPSEEK_V41_WO_B_BF16_ALLREDUCE": (
        "merged into VLLM_SM80_ALLREDUCE_ROUND_ONCE (wo_b partials summed with "
        "one rounding, in BF16 where the transport rounds once)"
    ),
    "VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE": (
        "merged into VLLM_SM80_ALLREDUCE_ROUND_ONCE (wo_b partials summed with "
        "one rounding, in BF16 where the transport rounds once)"
    ),
    "VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE": (
        "merged into VLLM_SM80_ALLREDUCE_ROUND_ONCE"
    ),
    "VLLM_GLM5_DRAFTER_SELECTOR_SHARD": (
        "renamed to VLLM_DFLASH2_SELECTOR_SHARD (read by the shared DFlash2 "
        "drafter code)"
    ),
    "VLLM_DSPARK_SHARD_ARGMAX": (
        "merged into VLLM_SM80_SHARD_ARGMAX (greedy target sampling and "
        "greedy drafting over a vocabulary-parallel head)"
    ),
    "VLLM_SM80_TARGET_SHARD_ARGMAX": (
        "merged into VLLM_SM80_SHARD_ARGMAX (greedy target sampling and "
        "greedy drafting over a vocabulary-parallel head)"
    ),
    "VLLM_DEEPSEEK_V41_DSPARK_SHARD_ARGMAX": (
        "merged into VLLM_SM80_SHARD_ARGMAX (greedy target sampling and "
        "greedy drafting over a vocabulary-parallel head)"
    ),
    "VLLM_DEEPSEEK_V41_TARGET_SHARD_ARGMAX": (
        "merged into VLLM_SM80_SHARD_ARGMAX (greedy target sampling and "
        "greedy drafting over a vocabulary-parallel head)"
    ),
    "VLLM_SM80_CAPTURE_DIR": (
        "merged into VLLM_SM80_CAPTURE, the path of a capture spec whose dir "
        "key names the directory"
    ),
    "VLLM_SM80_CAPTURE_FAMILIES": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's families list)"
    ),
    "VLLM_SM80_CAPTURE_MATCH": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's match)"
    ),
    "VLLM_SM80_CAPTURE_ROWS": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's rows list)"
    ),
    "VLLM_SM80_CAPTURE_PER_KEY": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's per_key)"
    ),
    "VLLM_SM80_CAPTURE_PREFILL_MATCH": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's prefill_match)"
    ),
    "VLLM_SM80_CAPTURE_PREFILL_CALLS": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's prefill_calls list)"
    ),
    "VLLM_SM80_CAPTURE_PREFILL_PER_KEY": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's prefill_per_key)"
    ),
    "VLLM_SM80_CAPTURE_PREFILL_MIN_ROWS": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's prefill_min_rows)"
    ),
    "VLLM_SM80_CAPTURE_MAX_MB": (
        "merged into VLLM_SM80_CAPTURE (the capture spec's max_mb)"
    ),
    "KDA_FOLD_REPLAY_DIR": (
        "merged into VLLM_SM80_REPLAY_DIR (its kda subdirectory)"
    ),
    "KDA_FOLD_REPLAY_STEPS": "merged into VLLM_SM80_REPLAY_LIMIT",
    "KDA_PRE_CAPTURE_DIR": (
        "merged into VLLM_SM80_REPLAY_DIR (its kda_pre subdirectory)"
    ),
    "KDA_PRE_CAPTURE_MAX": "merged into VLLM_SM80_REPLAY_LIMIT",
    "SMLA_PRE_CAPTURE_DIR": (
        "merged into VLLM_SM80_REPLAY_DIR (its smla_pre subdirectory)"
    ),
    "VLLM_GLM5_IDX_GLUE_CAPTURE": (
        "merged into VLLM_SM80_REPLAY_DIR (its idx_glue subdirectory)"
    ),
    "VLLM_PP_MARLIN_CAPTURE_DIR": (
        "merged into VLLM_SM80_REPLAY_DIR (its moe_pre and once "
        "subdirectories)"
    ),
    "VLLM_PP_MARLIN_CAPTURE_LIMIT": "merged into VLLM_SM80_REPLAY_LIMIT",
    "VLLM_TP4_MARLIN_CAPTURE_DIR": (
        "merged into VLLM_SM80_REPLAY_DIR (its moe_pre and once "
        "subdirectories)"
    ),
    "VLLM_TP4_MARLIN_CAPTURE_LIMIT": "merged into VLLM_SM80_REPLAY_LIMIT",
    # Replaced by a vLLM option or a standard location.
    "VLLM_GLM5_DFLASH_ADAPTIVE_K": (
        "replaced by --speculative-config adaptive_k (by_load)"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS": (
        "replaced by --speculative-config adaptive_k.by_load"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH": (
        "replaced by --speculative-config adaptive_k.draft_by_load"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT": (
        "replaced by --speculative-config adaptive_k.accept"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS": (
        "replaced by --speculative-config adaptive_k.accept.costs"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI": (
        "replaced by --speculative-config adaptive_k.accept.costs_multi"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_HYST": (
        "replaced by --speculative-config adaptive_k.accept.hysteresis"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_PRIOR": (
        "replaced by --speculative-config adaptive_k.accept.prior"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG": (
        "replaced by --speculative-config adaptive_k.log_interval"
    ),
    "VLLM_GLM5_LOCAL_LOGITS": "replaced by --enable-batch-sharded-sampling",
    "VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR": (
        "replaced by TORCH_EXTENSIONS_DIR (default "
        "~/.cache/vllm/host_shm_all_reduce)"
    ),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR": (
        "replaced by $VLLM_CACHE_ROOT/custom_all_reduce_flags"
    ),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S": (
        "replaced by the constant WAIT_S (60 s) in "
        "vllm_sm80/distributed/custom_all_reduce_flags.py"
    ),
    "VLLM_GLM5_AUX_HIDDEN_TENSOR": (
        "replaced by the model config field aux_hidden_state_tensor; for the "
        "branch tensor pass --hf-overrides "
        '{"text_config": {"aux_hidden_state_tensor": "branch"}}'
    ),
    "VLLM_QWEN4_ROUTER_FP32": (
        "replaced by --hf-overrides "
        '{"text_config": {"moe_router_dtype": "float32"}} (the model config '
        "field moe_router_dtype, also read by the MTP drafter)"
    ),
    "VLLM_QWEN4_MTP_PER_STREAM_NORM": (
        "replaced by --hf-overrides "
        '{"text_config": {"mtp_hidden_norm": "per_stream"}} (the model config '
        'field mtp_hidden_norm, default "flat")'
    ),
    # Removed with the code they selected.
    "VLLM_ROUND_ONCE_ALLREDUCE_GROUPED": (
        "removed with the grouped NCCL round-once all-reduce: the copy-engine "
        "ring's owner sum (VLLM_SM80_ALLREDUCE) takes every message it could "
        "take"
    ),
    "VLLM_GLM5_DECODE_MHC": (
        "removed with the v1 sm_80 mHC decode kernel; the v3 kernel, opted in "
        "with VLLM_GLM5_DECODE_MHC_V2, covers the same shapes"
    ),
    "VLLM_GLM5_DECODE_MHC_MAX_TOKENS": (
        "removed with the v1 sm_80 mHC decode kernel; the v3 bound is "
        "mhc/decode_v2_max_rows in the sm80 tuning table"
    ),
    "VLLM_GLM5_DECODE_MHC_V3": (
        "removed; v3 is the only sm_80 mHC decode kernel, opted in with "
        "VLLM_GLM5_DECODE_MHC_V2"
    ),
    "VLLM_GLM5_DECODE_MHC_V2_FN_BF16": (
        "removed; sm_80 mHC decode always reads the fp32 weights"
    ),
    "VLLM_GLM5_THIN_GEMM_V74": (
        "removed; the 74-SM schedule is the only thin GEMM"
    ),
    "VLLM_GLM5_PREFILL_PACK_BF16X2": (
        "removed; the paired-rounding pack is the only mHC pre-norm pack"
    ),
    "VLLM_GLM5_MARLIN_DECODE_VARIANT": (
        "removed; compiled Marlin decode always uses the split-K order "
        "(the former default, orig)"
    ),
    "VLLM_GLM5_CUSTOM_ALLREDUCE_MAX_SIZE": (
        "removed; the custom all-reduce uses the 8 MiB default cap"
    ),
    "THIN_GEMM_MAX_M": (
        "removed; it was read only by uncalled code. The thin GEMM bound is "
        "dense_gemm/thin_max_rows in the sm80 tuning table"
    ),
    "VLLM_KV_SWA_INFLIGHT_SCRATCH": (
        "removed; the KV capacity report charges sliding-window in-flight "
        "blocks per concurrency slot, as upstream"
    ),
    "VLLM_GLM5_INDEXER_DECODE_ROWS": (
        "removed; the indexer decode block tables are sized by "
        "max_num_batched_tokens, as upstream"
    ),
    "VLLM_GLM5_INDEXER_GATHER_CLAMP": (
        "removed; the indexer K-gather workspace and prefill chunk limit are "
        "sized as upstream (40 * max_model_len, in compressed rows)"
    ),
    "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED": (
        "removed; the TP4 split-list Marlin MoE prefill runs on the released "
        "Marlin kernels"
    ),
    # The TP prefill comm/compute overlap and its chunk-size observer.
    "VLLM_GLM5_PREFILL_OVERLAP": (
        "removed; the TP all-reduces of a prefill run unsplit on the main "
        "stream, as upstream"
    ),
    "VLLM_GLM5_PREFILL_OVERLAP_SPLITS": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_DEBUG": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_BACKEND": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OBSERVE": (
        "removed with VLLM_GLM5_PREFILL_OVERLAP (the prefill chunk-size "
        "observer lived in the overlap module)"
    ),
    "VLLM_PP_DRAFT_TAIL_VERIFY": (
        "removed; the last stage no longer reruns the draft tail to compare it "
        "with the drafts VLLM_PP_DRAFT_TAIL_STAGE sends"
    ),
    "VLLM_GLM5_HOST_ALLREDUCE_TEST_FAIL_RANK": (
        "removed; the host-staged all-reduce no longer has a forced "
        "registration failure for testing"
    ),
    "VLLM_HOST_ALLREDUCE_TEST_FAIL_RANK": (
        "removed; the host-staged all-reduce no longer has a forced "
        "registration failure for testing"
    ),
    "VLLM_EXL3": (
        "removed; an EXL3 checkpoint always loads with the EXL3 method"
    ),
    "VLLM_ENGRAM_NATIVE_READ": (
        "removed; the sm_80 Engram rows are always fetched by one native host "
        "function (csrc/exl3/engram_fetch.cpp)"
    ),
    "VLLM_GLM5_DFLASH_BOUNDARY_CACHE": (
        "removed; DFlash and DSpark never drop the last prefix-cache block "
        "(upstream vLLM #54163)"
    ),
    "VLLM_DFLASH_BOUNDARY_CACHE": (
        "removed; DFlash and DSpark never drop the last prefix-cache block "
        "(upstream vLLM #54163)"
    ),
    "VLLM_EXL3_EXPERT_PLACEMENT": (
        "removed; the routed experts keep the checkpoint placement"
    ),
    "VLLM_EXL3_EXPERT_PLACEMENT_SHA256": (
        "removed; the routed experts keep the checkpoint placement"
    ),
    "VLLM_EXL3_EXPERT_REPLICAS": "removed; no expert is replicated",
    "VLLM_EXL3_EXPERT_REPLICAS_SHA256": "removed; no expert is replicated",
    "VLLM_DEEPSEEK_V41_WO_A_STRIDED": (
        "removed; the grouped wo_a GEMM reads the attention output directly"
    ),
    # Size cut-offs: tuned data per card and shape (vllm_sm80/tuning.py).
    "VLLM_THIN_GEMM_MAX_TOKENS": (
        "moved to the sm80 tuning table, dense_gemm/thin_max_rows"
    ),
    "VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS": (
        "moved to the sm80 tuning table, mhc/decode_v2_max_rows"
    ),
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS": (
        "moved to the sm80 tuning table, moe_router/route_v2_max_rows"
    ),
    "VLLM_MOE_MASK_PADDING_MAX_TOKENS": (
        "moved to the sm80 tuning table, moe_router/fused_route_max_rows"
    ),
    "VLLM_GLM5_PREFILL_MIN_TOKENS": (
        "moved to the sm80 tuning table, attention_prefill/sparse_mla_min_rows "
        "and mhc/prefill_min_rows"
    ),
    "VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS": (
        "moved to the sm80 tuning table, moe_prefill/marlin_min_rows"
    ),
    "VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS": (
        "moved to the sm80 tuning table, moe_prefill/marlin_min_rows"
    ),
    "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS": (
        "moved to the sm80 tuning table, moe_routed_gemm/marlin_multi_max_rows"
    ),
    "VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT": (
        "moved to the sm80 tuning table, attention_prefill/sparse_mla_min_ctx"
    ),
    "VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS": (
        "moved to the sm80 tuning table, topk_select/split_rows"
    ),
    "VLLM_EXL3_MOE_CHUNK_ROWS": (
        "moved to the sm80 tuning table, moe_routed_gemm/exl3_chunk_rows"
    ),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES": (
        "moved to the sm80 tuning table, allreduce/flags_max_bytes"
    ),
    "VLLM_HOST_ALLREDUCE_MAX_SIZE": (
        "moved to the sm80 tuning table, allreduce/host_max_bytes"
    ),
    "VLLM_CUDAGRAPH_CAPTURE_MAX_SPLIT_MB": (
        "moved to the sm80 tuning table, cudagraph/capture_max_split_mb"
    ),
    # Support limits, now code constants.
    "VLLM_DEEPSEEK_V41_ATTN_PIPELINE_MAX_TOKENS": (
        "removed; the attention pipeline takes calls of up to 48 padded rows, "
        "its supported bound (sm80_attn_pipeline.MAX_ROWS)"
    ),
    "VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE_MAX_TOKENS": (
        "removed; the fused attention core takes calls of up to 48 padded "
        "rows, its validated bound (sm80_attn_fused.MAX_ROWS)"
    ),
    "KDA_FOLD_ONLY": (
        "removed; select the KDA fold test cases with pytest -k, or name them "
        "on the command line when running tests/kernels/test_ampere_kda_fold.py "
        "as a script"
    ),
}
_warned_retired_env_vars: set[str] = set()


def warn_retired_env_vars() -> None:
    """Warn once per process for every retired name that is set."""
    for name, replacement in RETIRED_ENV_VARS.items():
        if name in os.environ and name not in _warned_retired_env_vars:
            _warned_retired_env_vars.add(name)
            logger.warning("%s is retired and ignored: %s.", name, replacement)


def register_env_vars(environment_variables: dict[str, Callable[[], Any]]) -> None:
    """Add the fork's variables to vLLM's table, then warn about retired names.

    Raises if upstream vLLM declares one of the fork's names itself, so a
    rebase that brings such a name cannot silently swap the reader.
    """
    for name, getter in FORK_ENV_VARS.items():
        existing = environment_variables.get(name)
        if existing is not None and existing is not getter:
            raise ValueError(
                f"{name} is declared by vLLM and by vllm_sm80/envs.py; "
                "keep one declaration"
            )
    environment_variables.update(FORK_ENV_VARS)
    warn_retired_env_vars()
