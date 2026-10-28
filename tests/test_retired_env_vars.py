# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables retired in this release (vllm_sm80/envs.py).

A retired name is never read for a value; setting it only logs one warning
per process when vllm.envs is imported. vLLM's generic unknown-variable check
may also name it. Remove this file with the table in the next release.
"""

import os
import subprocess
import sys

import pytest

import vllm.envs as envs
from vllm_sm80 import envs as fork_envs

RENAMED = {
    "VLLM_GLM5_FLA_PIN_AUTOTUNE": "VLLM_FLA_PIN_AUTOTUNE",
    "VLLM_GLM5_MEM_ATTRIBUTION": "VLLM_MEM_ATTRIBUTION",
    "VLLM_GLM5_MOE_MASK_PADDING": "VLLM_MOE_MASK_PADDING",
    "VLLM_GLM5_SHARED_EXPERT_REORDER": "VLLM_SHARED_EXPERT_REORDER",
    "VLLM_GLM5_STATE_INDEX_CHECK": "VLLM_STATE_INDEX_CHECK",
    "VLLM_GLM5_THIN_GEMM": "VLLM_THIN_GEMM",
    "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE": "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P",
    "VLLM_BUILD_AMPERE_MARLIN": "VLLM_AMPERE_MARLIN_BUILD",
    "VLLM_DSA_INDEXER_REF": "VLLM_INDEXER_DSA_REF",
    "VLLM_GLM5_DECODE_IDX_GLUE": "VLLM_AMPERE_DECODE_IDX_GLUE",
    "VLLM_GLM5_DECODE_KERNELS": "VLLM_AMPERE_DECODE_KERNELS",
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2": "VLLM_MOE_DECODE_ROUTE_V2",
    "VLLM_GLM5_DETERMINISTIC_MOE_ALIGN": "VLLM_MOE_DETERMINISTIC_ALIGN",
    "VLLM_GLM5_INDEXER_DECODE_RAW_K": "VLLM_INDEXER_DECODE_RAW_K",
    "VLLM_GLM5_MARLIN_DECODE_CUDA": "VLLM_AMPERE_MARLIN_DECODE_CUDA",
    "VLLM_GLM5_TOOL_CHOICE_NONE_MASK": "VLLM_TOOL_CHOICE_NONE_MASK",
    "VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE": "VLLM_SM80_PREFILL_REDUCE_OVERLAP",
    "VLLM_GLM5_DRAFTER_SELECTOR_SHARD": "VLLM_DFLASH2_SELECTOR_SHARD",
}

MERGED = {
    "VLLM_GLM5_HOST_ALLREDUCE": "VLLM_SM80_ALLREDUCE",
    "VLLM_HOST_ALLREDUCE": "VLLM_SM80_ALLREDUCE",
    "VLLM_CUSTOM_ALLREDUCE_FLAGS": "VLLM_SM80_ALLREDUCE",
    "VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE": "VLLM_SM80_ALLREDUCE",
    "VLLM_DEEPSEEK_V41_WO_B_BF16_ALLREDUCE": "VLLM_SM80_ALLREDUCE_ROUND_ONCE",
    "VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE": (
        "VLLM_SM80_ALLREDUCE_ROUND_ONCE"
    ),
    "VLLM_GLM5_TOPK_TIEFIX": "VLLM_SM80_TOPK_CANONICAL_TIES",
    "VLLM_QWEN4_QSA_TOPK_CANONICAL": "VLLM_SM80_TOPK_CANONICAL_TIES",
    "VLLM_GLM5_TOPK_SORTED": "VLLM_SM80_TOPK_SORTED",
    "VLLM_QWEN_QSA_TOPK_SORTED": "VLLM_SM80_TOPK_SORTED",
    "VLLM_GLM5_PP_KDA_PREFILL": "VLLM_SM80_KDA_PREFILL",
    "VLLM_GLM5_TP4_KDA_PREFILL": "VLLM_SM80_KDA_PREFILL",
    "VLLM_AMPERE_KDA_PREFILL_WIDE": "VLLM_SM80_KDA_PREFILL",
    "VLLM_AMPERE_KDA_PREFILL_NARROW": "VLLM_SM80_KDA_PREFILL",
    "VLLM_GLM5_PP_MARLIN_PREFILL": "VLLM_SM80_MARLIN_PREFILL",
    "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED": "VLLM_SM80_MARLIN_PREFILL",
    "VLLM_GLM5_TP4_MARLIN_PREFILL": "VLLM_SM80_MARLIN_PREFILL",
    "VLLM_AMPERE_MARLIN_PREFILL_WIDE": "VLLM_SM80_MARLIN_PREFILL",
    "VLLM_AMPERE_MARLIN_PREFILL_NARROW": "VLLM_SM80_MARLIN_PREFILL",
    "VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED": "VLLM_SM80_MARLIN_PREFILL",
    "VLLM_GLM5_PP_SPARSE_MLA_PREFILL": "VLLM_SM80_SPARSE_MLA_PREFILL",
    "VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE": "VLLM_SM80_SPARSE_MLA_PREFILL",
    "VLLM_GLM5_DECODE_KDA_WIDE": "VLLM_AMPERE_DECODE_KERNELS",
    "VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS": "VLLM_AMPERE_MARLIN_DECODE_CUDA",
    "VLLM_GLM5_MARLIN_DECODE_PP_MULTI": "VLLM_AMPERE_MARLIN_DECODE_CUDA",
    "VLLM_DSPARK_SHARD_ARGMAX": "VLLM_SM80_SHARD_ARGMAX",
    "VLLM_SM80_TARGET_SHARD_ARGMAX": "VLLM_SM80_SHARD_ARGMAX",
    "VLLM_DEEPSEEK_V41_DSPARK_SHARD_ARGMAX": "VLLM_SM80_SHARD_ARGMAX",
    "VLLM_DEEPSEEK_V41_TARGET_SHARD_ARGMAX": "VLLM_SM80_SHARD_ARGMAX",
    "VLLM_SM80_CAPTURE_DIR": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_FAMILIES": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_MATCH": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_ROWS": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_PER_KEY": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_PREFILL_MATCH": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_PREFILL_CALLS": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_PREFILL_PER_KEY": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_PREFILL_MIN_ROWS": "VLLM_SM80_CAPTURE",
    "VLLM_SM80_CAPTURE_MAX_MB": "VLLM_SM80_CAPTURE",
    "KDA_FOLD_REPLAY_DIR": "VLLM_SM80_REPLAY_DIR",
    "KDA_FOLD_REPLAY_STEPS": "VLLM_SM80_REPLAY_LIMIT",
    "KDA_PRE_CAPTURE_DIR": "VLLM_SM80_REPLAY_DIR",
    "KDA_PRE_CAPTURE_MAX": "VLLM_SM80_REPLAY_LIMIT",
    "SMLA_PRE_CAPTURE_DIR": "VLLM_SM80_REPLAY_DIR",
    "VLLM_GLM5_IDX_GLUE_CAPTURE": "VLLM_SM80_REPLAY_DIR",
    "VLLM_PP_MARLIN_CAPTURE_DIR": "VLLM_SM80_REPLAY_DIR",
    "VLLM_PP_MARLIN_CAPTURE_LIMIT": "VLLM_SM80_REPLAY_LIMIT",
    "VLLM_TP4_MARLIN_CAPTURE_DIR": "VLLM_SM80_REPLAY_DIR",
    "VLLM_TP4_MARLIN_CAPTURE_LIMIT": "VLLM_SM80_REPLAY_LIMIT",
}

# Split into one switch per operation.
SPLIT = {
    "VLLM_GLM5_PREFILL_KERNELS": ("VLLM_SM80_MHC_PREFILL",
                                  "VLLM_SM80_SPARSE_MLA_PREFILL"),
}

REPLACED_OR_REMOVED = {
    "VLLM_GLM5_DFLASH_ADAPTIVE_K",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS",
    "VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_HYST",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_PRIOR",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG",
    "VLLM_GLM5_LOCAL_LOGITS",
    "VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR",
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR",
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S",
    "VLLM_GLM5_AUX_HIDDEN_TENSOR",
    "VLLM_QWEN4_ROUTER_FP32",
    "VLLM_QWEN4_MTP_PER_STREAM_NORM",
    "VLLM_GLM5_DECODE_MHC",
    "VLLM_GLM5_DECODE_MHC_MAX_TOKENS",
    "VLLM_GLM5_DECODE_MHC_V3",
    "VLLM_GLM5_DECODE_MHC_V2_FN_BF16",
    "VLLM_GLM5_THIN_GEMM_V74",
    "VLLM_GLM5_PREFILL_PACK_BF16X2",
    "VLLM_GLM5_MARLIN_DECODE_VARIANT",
    "VLLM_GLM5_CUSTOM_ALLREDUCE_MAX_SIZE",
    "THIN_GEMM_MAX_M",
    "VLLM_KV_SWA_INFLIGHT_SCRATCH",
    "VLLM_GLM5_INDEXER_DECODE_ROWS",
    "VLLM_GLM5_INDEXER_GATHER_CLAMP",
    "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED",
    "VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_SPLITS",
    "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS",
    "VLLM_GLM5_PREFILL_OVERLAP_DEBUG",
    "VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA",
    "VLLM_GLM5_PREFILL_OVERLAP_BACKEND",
    "VLLM_GLM5_PREFILL_OBSERVE",
    "VLLM_PP_DRAFT_TAIL_VERIFY",
    "VLLM_GLM5_HOST_ALLREDUCE_TEST_FAIL_RANK",
    "VLLM_HOST_ALLREDUCE_TEST_FAIL_RANK",
    "VLLM_EXL3",
    "VLLM_ENGRAM_NATIVE_READ",
    "VLLM_GLM5_DFLASH_BOUNDARY_CACHE",
    "VLLM_DFLASH_BOUNDARY_CACHE",
    "VLLM_EXL3_EXPERT_PLACEMENT",
    "VLLM_EXL3_EXPERT_PLACEMENT_SHA256",
    "VLLM_EXL3_EXPERT_REPLICAS",
    "VLLM_EXL3_EXPERT_REPLICAS_SHA256",
    "VLLM_DEEPSEEK_V41_WO_A_STRIDED",
    "VLLM_DEEPSEEK_V41_ATTN_PIPELINE_MAX_TOKENS",
    "VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE_MAX_TOKENS",
    "VLLM_ROUND_ONCE_ALLREDUCE_GROUPED",
    "KDA_FOLD_ONLY",
}

# Size cut-offs moved to the sm80 tuning table (vllm_sm80/tuning.py), with
# the (op, param) rows that replace them and the shape key and default each
# reader passes.
MOVED_TO_TABLE = {
    "VLLM_THIN_GEMM_MAX_TOKENS": (
        "dense_gemm/thin_max_rows", {"gemm_n": 4096, "gemm_k": 4096}, 32),
    "VLLM_GLM5_THIN_GEMM_MAX_TOKENS": (
        "dense_gemm/thin_max_rows", {"gemm_n": 4096, "gemm_k": 4096}, 32),
    "VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS": (
        "mhc/decode_v2_max_rows", {"hidden": 4096, "hc_mult": 4}, 32),
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS": (
        "moe_router/route_v2_max_rows",
        {"n_experts": 288, "top_k": 8, "hidden": 4096}, 32),
    "VLLM_MOE_MASK_PADDING_MAX_TOKENS": (
        "moe_router/fused_route_max_rows", {"top_k": 8}, 8),
    "VLLM_GLM5_DECODE_MOE_MAX_TOKENS": (
        "moe_router/fused_route_max_rows", {"top_k": 8}, 8),
    "VLLM_GLM5_PREFILL_MIN_TOKENS": (
        "attention_prefill/sparse_mla_min_rows", {"head_dim": 512}, 512),
    "VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS": (
        "moe_prefill/marlin_min_rows", {"gemm_n": 2048}, 384),
    "VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS": (
        "moe_prefill/marlin_min_rows", {"gemm_n": 512}, 384),
    "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS": (
        "moe_routed_gemm/marlin_multi_max_rows", {"gemm_n": 2048}, 32),
    "VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT": (
        "attention_prefill/sparse_mla_min_ctx",
        {"head_dim": 512, "index_width": 2176}, 4352),
    "VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS": (
        "topk_select/split_rows", {"top_k": 512}, 0),
    "VLLM_EXL3_MOE_CHUNK_ROWS": (
        "moe_routed_gemm/exl3_chunk_rows", {"hidden": 5120}, 0),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES": (
        "allreduce/flags_max_bytes", {"world_size": 4, "hidden": 5120}, 262144),
    "VLLM_HOST_ALLREDUCE_MAX_SIZE": (
        "allreduce/host_max_bytes", {"world_size": 4, "hidden": 5120}, 524288),
    "VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE": (
        "allreduce/host_max_bytes", {"world_size": 4, "hidden": 5120}, 524288),
    "VLLM_CUDAGRAPH_CAPTURE_MAX_SPLIT_MB": (
        "cudagraph/capture_max_split_mb", {}, 0),
}


def test_table_lists_every_retired_name():
    assert set(fork_envs.RETIRED_ENV_VARS) == (
        set(RENAMED) | set(MERGED) | set(SPLIT) | REPLACED_OR_REMOVED
        | set(MOVED_TO_TABLE)
    )
    assert all(fork_envs.RETIRED_ENV_VARS.values())


@pytest.mark.parametrize(
    "name",
    sorted(
        set(RENAMED) | set(MERGED) | set(SPLIT) | REPLACED_OR_REMOVED
        | set(MOVED_TO_TABLE)
    ),
)
def test_retired_name_is_not_declared(name):
    assert name not in envs.environment_variables


@pytest.mark.parametrize("old,new", sorted(RENAMED.items()))
def test_renamed_name_points_at_its_replacement(old, new):
    assert new in envs.environment_variables
    assert fork_envs.RETIRED_ENV_VARS[old].startswith(f"renamed to {new}")


@pytest.mark.parametrize("old,new", sorted(MERGED.items()))
def test_merged_name_points_at_its_replacement(old, new):
    assert new in envs.environment_variables
    assert fork_envs.RETIRED_ENV_VARS[old].startswith(f"merged into {new}")


@pytest.mark.parametrize("old,row", sorted(MOVED_TO_TABLE.items()))
def test_moved_name_points_at_its_table_row(old, row):
    message = fork_envs.RETIRED_ENV_VARS[old]
    assert message.startswith(f"moved to the sm80 tuning table, {row[0]}")


def test_prefill_min_tokens_names_both_rows():
    assert fork_envs.RETIRED_ENV_VARS["VLLM_GLM5_PREFILL_MIN_TOKENS"] == (
        "moved to the sm80 tuning table, attention_prefill/sparse_mla_min_rows "
        "and mhc/prefill_min_rows"
    )


@pytest.mark.parametrize("old,row", sorted(MOVED_TO_TABLE.items()))
def test_moved_value_is_never_used(monkeypatch, old, row):
    from tests.sm80.tuning_rows import use_rows
    from vllm_sm80 import tuning

    use_rows(monkeypatch)
    op_param, key, default = row
    op, param = op_param.split("/")
    # Not a valid value: a reader of the old name would raise or change it.
    monkeypatch.setenv(old, "not-a-valid-value")
    assert tuning.lookup(op, param, key, default) == default


@pytest.mark.parametrize("old,new", sorted(SPLIT.items()))
def test_split_name_points_at_its_replacements(old, new):
    assert all(name in envs.environment_variables for name in new)
    assert fork_envs.RETIRED_ENV_VARS[old].startswith(f"split into {new[0]} ")
    assert all(name in fork_envs.RETIRED_ENV_VARS[old] for name in new)


@pytest.mark.parametrize(
    "old,new",
    sorted((RENAMED | MERGED).items())
    + [(old, name) for old, names in sorted(SPLIT.items()) for name in names],
)
def test_retired_value_is_never_used(monkeypatch, old, new):
    monkeypatch.delenv(old, raising=False)
    monkeypatch.delenv(new, raising=False)
    default = envs.environment_variables[new]()
    # Not a valid value for any of the new names: an alias would raise.
    monkeypatch.setenv(old, "not-a-valid-value")
    assert envs.environment_variables[new]() == default


def test_warns_once_per_process(monkeypatch):
    retired = []
    monkeypatch.setattr(fork_envs, "_warned_retired_env_vars", set())
    monkeypatch.setattr(
        fork_envs.logger,
        "warning",
        lambda msg, *args: retired.append(msg % args),
    )
    for name in fork_envs.RETIRED_ENV_VARS:
        monkeypatch.setenv(name, "1")

    fork_envs.warn_retired_env_vars()
    fork_envs.warn_retired_env_vars()

    for name, replacement in fork_envs.RETIRED_ENV_VARS.items():
        assert retired.count(f"{name} is retired and ignored: {replacement}.") == 1
    assert len(retired) == len(fork_envs.RETIRED_ENV_VARS)


def test_process_startup_warns_once_and_ignores_the_value():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("VLLM_") and k not in fork_envs.RETIRED_ENV_VARS
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["VLLM_GLM5_MOE_MASK_PADDING"] = "1"
    env["VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR"] = "/nonexistent"
    code = (
        "import vllm.envs as e\n"
        "from vllm.platforms.interface import Platform\n"
        "Platform.validate_environ(False)\n"
        "assert e.VLLM_MOE_MASK_PADDING is False\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    for name in ("VLLM_GLM5_MOE_MASK_PADDING", "VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR"):
        line = f"{name} is retired and ignored: {fork_envs.RETIRED_ENV_VARS[name]}."
        assert out.count(line) == 1, out
        # The generic unknown-variable warning may follow; it is not a second
        # retired-name warning and the value is still ignored.
