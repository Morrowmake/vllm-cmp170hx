# Fork map

<!-- Generated from git by the fork's map tool. Do not edit by hand. -->

This repository is Morrowmake's fork of vLLM for the NVIDIA CMP 170HX and other sm_80 (GA100-class) GPUs. It is a general vLLM: it serves the same models as upstream. The license is Apache-2.0, as upstream.

Upstream base: 330c5bffeeb48725c0502a351e8ead2b54bd81eb

All numbers compare the working tree with this commit. A fork-owned path does not exist at the base. An upstream-owned path does. This file is not listed in its own tables.

The maintainers regenerate it from git whenever the fork changes.

## 1. Fork-owned code

Files added since the base, by location. Lines are counted by git.

### `vllm/models/<model>/nvidia/ops/`

Kernels used by one model only. Each model keeps its own copy.

| Path | Lines |
|---|---:|
| `vllm/models/deepseek_v41/nvidia/ops/sm80_act_quant.py` | 259 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_attn_core.py` | 932 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_attn_fused.py` | 614 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_attn_pipeline.py` | 453 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_draft_prologue.py` | 212 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_engram_dequant.py` | 112 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_gate_gemv.py` | 281 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_gate_mma.py` | 122 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_index_scores.py` | 220 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_index_topk.py` | 383 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_mhc.py` | 202 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_mhc_finish_v2.py` | 420 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_mhc_fused.py` | 590 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_mhc_side.py` | 207 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_route.py` | 225 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_route_gluon.py` | 199 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_route_mma.py` | 86 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_router_mma.py` | 337 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_sparse_attn.py` | 180 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_swa_meta.py` | 249 |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_wo_a_grouped.py` | 403 |
| `vllm/models/glm5next/nvidia/ops/idx_glue.py` | 538 |
| `vllm/models/glm5next/nvidia/ops/idx_glue_gates.py` | 99 |
| `vllm/models/glm5next/nvidia/ops/kda_decode.py` | 1,048 |
| `vllm/models/glm5next/nvidia/ops/kda_prefill.py` | 1,393 |
| **25 files** | **9,764** |

### `vllm/models/<model>/nvidia/`

Model code of one model that the fork plugin registers in place of the upstream classes: model, layer and metadata subclasses.

| Path | Lines |
|---|---:|
| `vllm/models/deepseek_v41/nvidia/dspark_depth_policy.json` | 300 |
| `vllm/models/deepseek_v41/nvidia/hybrid_capture.py` | 367 |
| `vllm/models/deepseek_v41/nvidia/sm80_attention.py` | 1,171 |
| `vllm/models/deepseek_v41/nvidia/sm80_engram.py` | 1,380 |
| `vllm/models/deepseek_v41/nvidia/sm80_engram_prefetch.py` | 107 |
| `vllm/models/deepseek_v41/nvidia/sm80_engram_projection.py` | 18 |
| `vllm/models/deepseek_v41/nvidia/sm80_engram_wkv.py` | 196 |
| `vllm/models/deepseek_v41/nvidia/sm80_exl3.py` | 104 |
| `vllm/models/deepseek_v41/nvidia/sm80_model.py` | 528 |
| `vllm/models/deepseek_v41/nvidia/sm80_moe.py` | 537 |
| `vllm/models/deepseek_v41/nvidia/sm80_pipeline_capture.py` | 184 |
| `vllm/models/deepseek_v41/nvidia/sm80_split_reduce.py` | 252 |
| `vllm/models/deepseek_v41/nvidia/sm80_swa_meta_batch.py` | 445 |
| `vllm/models/glm5next/nvidia/model_state.py` | 113 |
| `vllm/models/glm5next/nvidia/sm80_kda.py` | 224 |
| `vllm/models/glm5next/nvidia/sm80_mla.py` | 123 |
| `vllm/models/glm5next/nvidia/sm80_model.py` | 470 |
| `vllm/models/qwen4_exp/nvidia/exl3_layout.py` | 190 |
| `vllm/models/qwen4_exp/nvidia/exl3_ple.py` | 948 |
| `vllm/models/qwen4_exp/nvidia/precision.py` | 93 |
| `vllm/models/qwen4_exp/nvidia/sm80_model.py` | 44 |
| **21 files** | **7,794** |

### `vllm/sm80/kernels/`

Kernels shared by several models on sm_80, one directory per family.

| Path | Lines |
|---|---:|
| `vllm/sm80/kernels/__init__.py` | 19 |
| `vllm/sm80/kernels/_platform.py` | 33 |
| `vllm/sm80/kernels/attention/__init__.py` | 8 |
| `vllm/sm80/kernels/attention/sparse_prefill_mla.py` | 480 |
| `vllm/sm80/kernels/attention/sparse_prefill_mla_pp.py` | 380 |
| `vllm/sm80/kernels/attention/triton_e4m3.py` | 142 |
| `vllm/sm80/kernels/attention/triton_mla_remap_cache.py` | 168 |
| `vllm/sm80/kernels/attention/triton_mla_sparse.py` | 612 |
| `vllm/sm80/kernels/attention/triton_mqa_logits.py` | 628 |
| `vllm/sm80/kernels/exl3/__init__.py` | 13 |
| `vllm/sm80/kernels/exl3/capture_points.py` | 111 |
| `vllm/sm80/kernels/exl3/dense_mma.py` | 151 |
| `vllm/sm80/kernels/exl3/dense_prefill_tc.py` | 354 |
| `vllm/sm80/kernels/exl3/hybrid.py` | 133 |
| `vllm/sm80/kernels/exl3/layout.py` | 108 |
| `vllm/sm80/kernels/exl3/loader.py` | 230 |
| `vllm/sm80/kernels/exl3/ops.py` | 1,383 |
| `vllm/sm80/kernels/exl3/override.py` | 128 |
| `vllm/sm80/kernels/exl3/quant.py` | 1,794 |
| `vllm/sm80/kernels/exl3/step_phase.py` | 98 |
| `vllm/sm80/kernels/exl3/utils.py` | 401 |
| `vllm/sm80/kernels/idx_glue_gate.py` | 75 |
| `vllm/sm80/kernels/indexer/__init__.py` | 9 |
| `vllm/sm80/kernels/indexer/topk_tiefix.py` | 571 |
| `vllm/sm80/kernels/indexer/topk_tiefix_gate.py` | 259 |
| `vllm/sm80/kernels/kda/__init__.py` | 17 |
| `vllm/sm80/kernels/kda/commit_plan.py` | 225 |
| `vllm/sm80/kernels/kda/recover.py` | 1,053 |
| `vllm/sm80/kernels/marlin_moe/__init__.py` | 12 |
| `vllm/sm80/kernels/marlin_moe/decode.py` | 420 |
| `vllm/sm80/kernels/marlin_moe/extension.py` | 170 |
| `vllm/sm80/kernels/marlin_moe/prefill.py` | 501 |
| `vllm/sm80/kernels/marlin_moe/split_align.py` | 224 |
| `vllm/sm80/kernels/mhc/__init__.py` | 137 |
| `vllm/sm80/kernels/mhc/decode.py` | 953 |
| `vllm/sm80/kernels/mhc/prenorm.py` | 470 |
| `vllm/sm80/kernels/mhc/prenorm_workspace.py` | 118 |
| `vllm/sm80/kernels/mhc/warmup.py` | 46 |
| `vllm/sm80/kernels/moe/__init__.py` | 400 |
| `vllm/sm80/kernels/moe/align.py` | 325 |
| `vllm/sm80/kernels/moe/route.py` | 1,109 |
| `vllm/sm80/kernels/moe/routing.py` | 709 |
| `vllm/sm80/kernels/moe/sum.py` | 84 |
| `vllm/sm80/kernels/moe/warmup.py` | 110 |
| `vllm/sm80/kernels/thin_gemm/__init__.py` | 206 |
| `vllm/sm80/kernels/thin_gemm/concat.py` | 280 |
| `vllm/sm80/kernels/thin_gemm/gluon.py` | 299 |
| `vllm/sm80/kernels/thin_gemm/thin_gemm_gluon.py` | 968 |
| `vllm/sm80/kernels/thin_gemm/thin_gemm_v74.py` | 1,037 |
| `vllm/sm80/kernels/thin_gemm/warmup.py` | 154 |
| `vllm/sm80/kernels/warmup.py` | 132 |
| `vllm/sm80/kernels/warmup_utils.py` | 37 |
| **52 files** | **18,484** |

### `vllm/sm80/`

The rest of the shared sm_80 code: the sparse MLA attention backend, the host-staged and flags-in-data all-reduce, FLA autotune pins, speculative-decoding helpers and pipeline-parallel worker helpers.

| Path | Lines |
|---|---:|
| `vllm/sm80/__init__.py` | 41 |
| `vllm/sm80/attention/__init__.py` | 2 |
| `vllm/sm80/attention/triton_mla_sparse.py` | 386 |
| `vllm/sm80/capture.py` | 703 |
| `vllm/sm80/deepseek_v41_recovery.py` | 344 |
| `vllm/sm80/dev_ab.py` | 1,670 |
| `vllm/sm80/dev_ab_batch.py` | 367 |
| `vllm/sm80/distributed/__init__.py` | 2 |
| `vllm/sm80/distributed/ce_ring_all_reduce.py` | 406 |
| `vllm/sm80/distributed/communicator.py` | 397 |
| `vllm/sm80/distributed/custom_all_reduce_flags.py` | 680 |
| `vllm/sm80/distributed/host_shm_all_reduce.py` | 1,155 |
| `vllm/sm80/distributed/pcie_p2p.py` | 183 |
| `vllm/sm80/distributed/step_errors.py` | 167 |
| `vllm/sm80/envs.py` | 1,656 |
| `vllm/sm80/failed_inputs.py` | 144 |
| `vllm/sm80/fla/__init__.py` | 2 |
| `vllm/sm80/fla/autotune_pins.py` | 210 |
| `vllm/sm80/fla/pinned_autotune.py` | 384 |
| `vllm/sm80/guard.py` | 68 |
| `vllm/sm80/hot_swap.py` | 807 |
| `vllm/sm80/jit_extension.py` | 150 |
| `vllm/sm80/kv_planning.py` | 588 |
| `vllm/sm80/launch/__init__.py` | 7 |
| `vllm/sm80/launch/canon.py` | 110 |
| `vllm/sm80/launch/checkpoint.py` | 428 |
| `vllm/sm80/launch/data/build_inputs.toml` | 48 |
| `vllm/sm80/launch/data/link_written.toml` | 12 |
| `vllm/sm80/launch/data/native_inputs.toml` | 107 |
| `vllm/sm80/launch/data/tree_exempt.toml` | 16 |
| `vllm/sm80/launch/defaults.toml` | 160 |
| `vllm/sm80/launch/envclass.py` | 45 |
| `vllm/sm80/launch/errors.py` | 20 |
| `vllm/sm80/launch/main.py` | 578 |
| `vllm/sm80/launch/model.py` | 150 |
| `vllm/sm80/launch/registry.py` | 421 |
| `vllm/sm80/launch/resolver.py` | 1,098 |
| `vllm/sm80/launch/settings_dir.py` | 237 |
| `vllm/sm80/launch/treestate.py` | 822 |
| `vllm/sm80/launch/validated.py` | 292 |
| `vllm/sm80/layers/__init__.py` | 4 |
| `vllm/sm80/layers/grouped_linear.py` | 94 |
| `vllm/sm80/layers/moe/__init__.py` | 19 |
| `vllm/sm80/layers/moe/gate.py` | 44 |
| `vllm/sm80/layers/moe/routed_experts.py` | 103 |
| `vllm/sm80/layers/moe/router.py` | 129 |
| `vllm/sm80/layers/moe/runner.py` | 177 |
| `vllm/sm80/layers/moe/shared_experts.py` | 134 |
| `vllm/sm80/model_registry.py` | 78 |
| `vllm/sm80/parser_recovery.py` | 321 |
| `vllm/sm80/platform.py` | 156 |
| `vllm/sm80/plugin.py` | 356 |
| `vllm/sm80/spec_decode/__init__.py` | 2 |
| `vllm/sm80/spec_decode/adaptive_k.py` | 366 |
| `vllm/sm80/spec_decode/aux_fc_fold.py` | 184 |
| `vllm/sm80/spec_decode/dflash2_model.py` | 425 |
| `vllm/sm80/spec_decode/dspark_confidence.py` | 369 |
| `vllm/sm80/spec_decode/dspark_depth_hint.py` | 175 |
| `vllm/sm80/spec_decode/dspark_host_width.py` | 470 |
| `vllm/sm80/spec_decode/dspark_row_mask.py` | 364 |
| `vllm/sm80/spec_decode/dspark_row_mask_conf.py` | 272 |
| `vllm/sm80/spec_decode/dspark_row_mask_kernel.py` | 331 |
| `vllm/sm80/spec_decode/dspark_row_mask_reject.py` | 48 |
| `vllm/sm80/spec_decode/dspark_width.py` | 104 |
| `vllm/sm80/spec_decode/lilicorr_model.py` | 31 |
| `vllm/sm80/spec_decode/shard_argmax.py` | 627 |
| `vllm/sm80/spec_decode/warmup.py` | 184 |
| `vllm/sm80/table_segment.py` | 707 |
| `vllm/sm80/tuning.py` | 278 |
| `vllm/sm80/tuning/nvidia-cmp-170hx.toml` | 331 |
| `vllm/sm80/worker/__init__.py` | 2 |
| `vllm/sm80/worker/cudagraph_mem.py` | 528 |
| `vllm/sm80/worker/execution_trace.py` | 150 |
| `vllm/sm80/worker/graph_mem_worker.py` | 96 |
| `vllm/sm80/worker/mem_attribution.py` | 276 |
| `vllm/sm80/worker/pp_draft_tail.py` | 1,017 |
| `vllm/sm80/worker/pp_hop.py` | 210 |
| `vllm/sm80/worker/state_index_check.py` | 299 |
| `vllm/sm80/worker/target_argmax.py` | 278 |
| `vllm/sm80/worker/trace_export.py` | 200 |
| **80 files** | **25,002** |

### `csrc/libtorch_stable/moe/ampere_marlin/`

An optional native library for the Marlin MoE kernels. vLLM runs without it.

| Path | Lines |
|---|---:|
| `csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py` | 101 |
| `csrc/libtorch_stable/moe/ampere_marlin/common_tiles.h` | 19 |
| `csrc/libtorch_stable/moe/ampere_marlin/decode_orig.cu` | 416 |
| `csrc/libtorch_stable/moe/ampere_marlin/decode_v2.cu` | 404 |
| `csrc/libtorch_stable/moe/ampere_marlin/kernel_selector_tiles.h` | 35 |
| `csrc/libtorch_stable/moe/ampere_marlin/kernels_tiles_sm80.cu` | 25 |
| `csrc/libtorch_stable/moe/ampere_marlin/marlin_moe_kernel.h` | 75 |
| `csrc/libtorch_stable/moe/ampere_marlin/marlin_moe_ops.cuh` | 770 |
| `csrc/libtorch_stable/moe/ampere_marlin/marlin_moe_template.h` | 2,238 |
| `csrc/libtorch_stable/moe/ampere_marlin/module.cpp` | 28 |
| `csrc/libtorch_stable/moe/ampere_marlin/prefill_tiles.cu` | 30 |
| **11 files** | **4,141** |

### Other fork-owned files

Files added elsewhere, grouped by top-level directory.

#### Repository root

| Path | Lines |
|---|---:|
| `CITATION.cff` | 7 |
| `COMMUNITY_RECIPES.md` | 68 |
| `NOTICE` | 9 |
| **3 files** | **84** |

#### `.github/`

| Path | Lines |
|---|---:|
| `.github/ISSUE_TEMPLATE/050-engine-bug.yml` | 43 |
| **1 file** | **43** |

#### `csrc/`

| Path | Lines |
|---|---:|
| `csrc/exl3/build_standalone.py` | 117 |
| `csrc/exl3/codec.cuh` | 211 |
| `csrc/exl3/codec_moe_v2.cuh` | 119 |
| `csrc/exl3/codec_prefill.cuh` | 68 |
| `csrc/exl3/common.h` | 114 |
| `csrc/exl3/dense.cu` | 151 |
| `csrc/exl3/dense_mma.cu` | 862 |
| `csrc/exl3/dense_mma.h` | 24 |
| `csrc/exl3/dense_prefill_tc.cu` | 631 |
| `csrc/exl3/dense_prefill_tc.h` | 41 |
| `csrc/exl3/dense_splitk.cu` | 618 |
| `csrc/exl3/dense_splitk.h` | 26 |
| `csrc/exl3/engram_fetch.cpp` | 862 |
| `csrc/exl3/engram_fetch.h` | 41 |
| `csrc/exl3/gemm.cu` | 489 |
| `csrc/exl3/gemm.cuh` | 82 |
| `csrc/exl3/module.cpp` | 103 |
| `csrc/exl3/moe.cu` | 409 |
| `csrc/exl3/moe_prefill.cu` | 1,045 |
| `csrc/exl3/moe_prefill.h` | 28 |
| `csrc/exl3/moe_prefill_try.cu` | 1,283 |
| `csrc/exl3/moe_prefill_try.h` | 23 |
| `csrc/exl3/moe_prefill_v2.cu` | 1,175 |
| `csrc/exl3/moe_prefill_v2.h` | 30 |
| `csrc/exl3/moe_v2.cu` | 2,143 |
| `csrc/exl3/moe_v2.h` | 39 |
| `csrc/exl3/moe_v2_try.cu` | 1,410 |
| `csrc/exl3/moe_v2_try/codec.cuh` | 223 |
| `csrc/exl3/moe_v2_try/codec_moe_v2.cuh` | 92 |
| `csrc/exl3/moe_v2_try/common.h` | 114 |
| `csrc/exl3/moe_v2_try/ops.h` | 31 |
| `csrc/exl3/ops.h` | 48 |
| `csrc/exl3/ple.cu` | 234 |
| `csrc/exl3/ple.h` | 53 |
| `csrc/exl3/ple_read.cpp` | 373 |
| `csrc/exl3/reference.cu` | 450 |
| `csrc/exl3/row_read.h` | 279 |
| **37 files** | **14,041** |

#### `docs/`

| Path | Lines |
|---|---:|
| `docs/THIRD_PARTY.md` | 85 |
| `docs/UPSTREAM_README.md` | 111 |
| `docs/features/quantization/exl3.md` | 146 |
| `docs/features/speculative_decoding/adaptive_k.md` | 168 |
| **4 files** | **510** |

#### `plugins/`

| Path | Lines |
|---|---:|
| `plugins/sm80/licenses/exl3/LICENSE` | 21 |
| `plugins/sm80/licenses/exl3/NOTICE` | 33 |
| `plugins/sm80/pyproject.toml` | 35 |
| **3 files** | **89** |

#### `tests/`

| Path | Lines |
|---|---:|
| `tests/distributed/test_custom_all_reduce_flags.py` | 963 |
| `tests/distributed/test_host_shm_all_reduce.py` | 614 |
| `tests/distributed/test_host_shm_setup_agreement.py` | 246 |
| `tests/distributed/test_pcie_p2p_gate.py` | 510 |
| `tests/kernels/ampere_sass.py` | 81 |
| `tests/kernels/interpreter_child.py` | 107 |
| `tests/kernels/kda_fold_world.py` | 614 |
| `tests/kernels/kda_recover_reference.py` | 20 |
| `tests/kernels/moe/test_moe_mask_padding.py` | 209 |
| `tests/kernels/moe/test_moe_mask_padding_route_v2.py` | 419 |
| `tests/kernels/moe/test_moe_route_v2_mask.py` | 410 |
| `tests/kernels/moe/test_shared_experts_reorder.py` | 490 |
| `tests/kernels/moe_align_reference.py` | 87 |
| `tests/kernels/quantization/exl3_moe_v2_digests.py` | 165 |
| `tests/kernels/quantization/exl3_real_cases.py` | 392 |
| `tests/kernels/quantization/test_exl3_dense_mma.py` | 596 |
| `tests/kernels/quantization/test_exl3_dense_mma_real_inputs.py` | 295 |
| `tests/kernels/quantization/test_exl3_dense_prefill_tc.py` | 911 |
| `tests/kernels/quantization/test_exl3_dense_prefill_tc_real_inputs.py` | 281 |
| `tests/kernels/quantization/test_exl3_dense_splitk.py` | 468 |
| `tests/kernels/quantization/test_exl3_dense_splitk_real_inputs.py` | 322 |
| `tests/kernels/quantization/test_exl3_expert_buffers.py` | 295 |
| `tests/kernels/quantization/test_exl3_expert_split.py` | 509 |
| `tests/kernels/quantization/test_exl3_fp32_out.py` | 213 |
| `tests/kernels/quantization/test_exl3_gemm_bitexact.py` | 324 |
| `tests/kernels/quantization/test_exl3_kernels.py` | 369 |
| `tests/kernels/quantization/test_exl3_moe_chunk_rows.py` | 280 |
| `tests/kernels/quantization/test_exl3_moe_prefill.py` | 536 |
| `tests/kernels/quantization/test_exl3_moe_prefill_real_inputs.py` | 262 |
| `tests/kernels/quantization/test_exl3_moe_prefill_v2.py` | 768 |
| `tests/kernels/quantization/test_exl3_moe_swiglu_clamp.py` | 86 |
| `tests/kernels/quantization/test_exl3_moe_tp_shapes.py` | 102 |
| `tests/kernels/quantization/test_exl3_moe_v2.py` | 618 |
| `tests/kernels/quantization/test_exl3_moe_v2_draft.py` | 824 |
| `tests/kernels/quantization/test_exl3_moe_v2_fold.py` | 1,029 |
| `tests/kernels/quantization/test_exl3_moe_v2_hidden5120.py` | 594 |
| `tests/kernels/quantization/test_exl3_moe_v2_real_inputs.py` | 345 |
| `tests/kernels/quantization/test_exl3_moe_v2_switches.py` | 415 |
| `tests/kernels/quantization/test_exl3_step_phase.py` | 147 |
| `tests/kernels/quantization/test_sm80_capture.py` | 563 |
| `tests/kernels/replay_inputs.py` | 46 |
| `tests/kernels/test_ampere_bundle_interactions.py` | 240 |
| `tests/kernels/test_ampere_decode.py` | 473 |
| `tests/kernels/test_ampere_decode_mhc_v3.py` | 398 |
| `tests/kernels/test_ampere_idx_glue.py` | 838 |
| `tests/kernels/test_ampere_indexer_decode_raw_k.py` | 270 |
| `tests/kernels/test_ampere_kda_dispatch.py` | 785 |
| `tests/kernels/test_ampere_kda_fold.py` | 1,035 |
| `tests/kernels/test_ampere_kda_prefill_metadata.py` | 579 |
| `tests/kernels/test_ampere_kda_recover.py` | 1,007 |
| `tests/kernels/test_ampere_kda_small_stack.py` | 504 |
| `tests/kernels/test_ampere_kda_step_tile.py` | 185 |
| `tests/kernels/test_ampere_marlin_decode_v2.py` | 625 |
| `tests/kernels/test_ampere_marlin_optional.py` | 237 |
| `tests/kernels/test_ampere_marlin_pp_mid_rows.py` | 406 |
| `tests/kernels/test_ampere_marlin_prefill_tiles.py` | 862 |
| `tests/kernels/test_ampere_moe_route.py` | 405 |
| `tests/kernels/test_ampere_pp_kda_prefill.py` | 452 |
| `tests/kernels/test_ampere_pp_marlin_prefill.py` | 706 |
| `tests/kernels/test_ampere_pp_sparse_mla_prefill.py` | 618 |
| `tests/kernels/test_ampere_prefill.py` | 955 |
| `tests/kernels/test_ampere_prefill_pack_bf16x2.py` | 149 |
| `tests/kernels/test_ampere_smla_prefill_pred_load.py` | 430 |
| `tests/kernels/test_ampere_thin_gemm.py` | 241 |
| `tests/kernels/test_ampere_thin_gemm_concat.py` | 964 |
| `tests/kernels/test_ampere_thin_gemm_gluon.py` | 995 |
| `tests/kernels/test_ampere_thin_gemm_gluon_real_inputs.py` | 242 |
| `tests/kernels/test_ampere_thin_gemm_v74.py` | 286 |
| `tests/kernels/test_ampere_tp4_kda_prefill.py` | 365 |
| `tests/kernels/test_ampere_tp4_marlin_prefill.py` | 408 |
| `tests/kernels/test_deterministic_moe_align.py` | 411 |
| `tests/kernels/test_fla_autotune_pin.py` | 629 |
| `tests/kernels/test_replay_inputs.py` | 56 |
| `tests/kernels/test_sorted_topk.py` | 262 |
| `tests/kernels/test_sparse_kv_row_offsets.py` | 173 |
| `tests/kernels/test_topk_reference.py` | 114 |
| `tests/kernels/test_topk_tiefix.py` | 458 |
| `tests/kernels/test_topk_tiefix_split.py` | 320 |
| `tests/kernels/thin_gemm_real_cases.py` | 148 |
| `tests/kernels/topk_reference.py` | 59 |
| `tests/model_executor/layers/test_exl3_layers.py` | 615 |
| `tests/model_executor/model_loader/test_exl3_manifest.py` | 250 |
| `tests/model_executor/model_loader/test_exl3_override.py` | 247 |
| `tests/models/deepseek_v41/sm80/__init__.py` | 0 |
| `tests/models/deepseek_v41/sm80/attn_pipeline_cases.py` | 408 |
| `tests/models/deepseek_v41/sm80/construct.py` | 209 |
| `tests/models/deepseek_v41/sm80/dspark_draft_identity.py` | 201 |
| `tests/models/deepseek_v41/sm80/dsv41_ref.py` | 764 |
| `tests/models/deepseek_v41/sm80/exact_round.py` | 52 |
| `tests/models/deepseek_v41/sm80/exl3_fixture.py` | 284 |
| `tests/models/deepseek_v41/sm80/export_attn_pipeline_cases.py` | 306 |
| `tests/models/deepseek_v41/sm80/export_mhc_consumer_cases.py` | 225 |
| `tests/models/deepseek_v41/sm80/fixtures/config.json` | 184 |
| `tests/models/deepseek_v41/sm80/fixtures/exl3_headers.json` | 25,297 |
| `tests/models/deepseek_v41/sm80/fp64_cases.py` | 1,661 |
| `tests/models/deepseek_v41/sm80/hybrid_consumer_cases.py` | 289 |
| `tests/models/deepseek_v41/sm80/hybrid_fixture.py` | 189 |
| `tests/models/deepseek_v41/sm80/hybrid_linear_cases.py` | 224 |
| `tests/models/deepseek_v41/sm80/lifecycle_contracts.py` | 1,508 |
| `tests/models/deepseek_v41/sm80/mhc_consumer_cases.py` | 148 |
| `tests/models/deepseek_v41/sm80/swa_meta_batch_cases.py` | 519 |
| `tests/models/deepseek_v41/sm80/test_act_quant.py` | 249 |
| `tests/models/deepseek_v41/sm80/test_attn_layers.py` | 309 |
| `tests/models/deepseek_v41/sm80/test_boot_warmup.py` | 511 |
| `tests/models/deepseek_v41/sm80/test_dspark_draft_identity.py` | 66 |
| `tests/models/deepseek_v41/sm80/test_dspark_draft_identity_gpu.py` | 35 |
| `tests/models/deepseek_v41/sm80/test_dspark_shard_argmax.py` | 422 |
| `tests/models/deepseek_v41/sm80/test_dspark_shard_argmax_stats.py` | 111 |
| `tests/models/deepseek_v41/sm80/test_engram_early_gather.py` | 226 |
| `tests/models/deepseek_v41/sm80/test_engram_storage.py` | 1,900 |
| `tests/models/deepseek_v41/sm80/test_engram_wkv_replica.py` | 459 |
| `tests/models/deepseek_v41/sm80/test_engram_wkv_replica_gpu.py` | 113 |
| `tests/models/deepseek_v41/sm80/test_exl3_construction.py` | 129 |
| `tests/models/deepseek_v41/sm80/test_fp32_outputs.py` | 379 |
| `tests/models/deepseek_v41/sm80/test_fp64_cases.py` | 1,286 |
| `tests/models/deepseek_v41/sm80/test_full_graph_shapes.py` | 336 |
| `tests/models/deepseek_v41/sm80/test_hybrid_capture.py` | 349 |
| `tests/models/deepseek_v41/sm80/test_hybrid_capture_gpu.py` | 78 |
| `tests/models/deepseek_v41/sm80/test_hybrid_checkpoint.py` | 495 |
| `tests/models/deepseek_v41/sm80/test_lifecycle_contracts.py` | 126 |
| `tests/models/deepseek_v41/sm80/test_mhc_consumer_cases.py` | 115 |
| `tests/models/deepseek_v41/sm80/test_no_host_sync.py` | 117 |
| `tests/models/deepseek_v41/sm80/test_official_reference.py` | 1,402 |
| `tests/models/deepseek_v41/sm80/test_round_once_identity.py` | 279 |
| `tests/models/deepseek_v41/sm80/test_row_mask_moe.py` | 641 |
| `tests/models/deepseek_v41/sm80/test_sm80_attention_wiring.py` | 526 |
| `tests/models/deepseek_v41/sm80/test_sm80_attn_fused.py` | 224 |
| `tests/models/deepseek_v41/sm80/test_sm80_attn_fused_gpu.py` | 279 |
| `tests/models/deepseek_v41/sm80/test_sm80_attn_pipeline_gpu.py` | 439 |
| `tests/models/deepseek_v41/sm80/test_sm80_ce_allreduce_dispatch.py` | 124 |
| `tests/models/deepseek_v41/sm80/test_sm80_decoder_replay.py` | 37 |
| `tests/models/deepseek_v41/sm80/test_sm80_draft_prologue.py` | 366 |
| `tests/models/deepseek_v41/sm80/test_sm80_draft_prologue_gpu.py` | 162 |
| `tests/models/deepseek_v41/sm80/test_sm80_gate_gemv.py` | 458 |
| `tests/models/deepseek_v41/sm80/test_sm80_gate_gemv_gpu.py` | 200 |
| `tests/models/deepseek_v41/sm80/test_sm80_index_topk.py` | 98 |
| `tests/models/deepseek_v41/sm80/test_sm80_index_topk_gpu.py` | 139 |
| `tests/models/deepseek_v41/sm80/test_sm80_index_visible.py` | 177 |
| `tests/models/deepseek_v41/sm80/test_sm80_kernels_gpu.py` | 312 |
| `tests/models/deepseek_v41/sm80/test_sm80_mhc_finish_v2.py` | 349 |
| `tests/models/deepseek_v41/sm80/test_sm80_mhc_finish_v2_gpu.py` | 185 |
| `tests/models/deepseek_v41/sm80/test_sm80_mhc_fused.py` | 302 |
| `tests/models/deepseek_v41/sm80/test_sm80_mhc_fused_gpu.py` | 176 |
| `tests/models/deepseek_v41/sm80/test_sm80_mhc_routing.py` | 237 |
| `tests/models/deepseek_v41/sm80/test_sm80_mhc_side.py` | 480 |
| `tests/models/deepseek_v41/sm80/test_sm80_moe_allreduce.py` | 141 |
| `tests/models/deepseek_v41/sm80/test_sm80_pipeline_lifecycle.py` | 126 |
| `tests/models/deepseek_v41/sm80/test_sm80_pipeline_rope.py` | 59 |
| `tests/models/deepseek_v41/sm80/test_sm80_route_fused.py` | 302 |
| `tests/models/deepseek_v41/sm80/test_sm80_route_fused_gpu.py` | 146 |
| `tests/models/deepseek_v41/sm80/test_sm80_router_mma_gpu.py` | 118 |
| `tests/models/deepseek_v41/sm80/test_sm80_router_mma_switch.py` | 340 |
| `tests/models/deepseek_v41/sm80/test_sm80_select_one_launch.py` | 262 |
| `tests/models/deepseek_v41/sm80/test_sm80_select_one_launch_gpu.py` | 157 |
| `tests/models/deepseek_v41/sm80/test_sm80_split_reduce.py` | 471 |
| `tests/models/deepseek_v41/sm80/test_sm80_stage_entry_engram.py` | 64 |
| `tests/models/deepseek_v41/sm80/test_sm80_swa_meta_batch.py` | 68 |
| `tests/models/deepseek_v41/sm80/test_sm80_swa_meta_batch_gpu.py` | 19 |
| `tests/models/deepseek_v41/sm80/test_sm80_wo_a_grouped.py` | 548 |
| `tests/models/deepseek_v41/sm80/test_sm80_wo_a_grouped_gpu.py` | 227 |
| `tests/models/deepseek_v41/sm80/test_target_shard_argmax.py` | 779 |
| `tests/models/deepseek_v41/sm80/test_target_shard_argmax_gpu.py` | 72 |
| `tests/models/deepseek_v41/sm80/test_topk_width.py` | 54 |
| `tests/models/deepseek_v41/sm80/test_wo_b_allreduce_bits_gpu.py` | 113 |
| `tests/models/deepseek_v41/sm80/test_wo_b_bits_verdict.py` | 83 |
| `tests/models/deepseek_v41/sm80/wo_b_allreduce_bits.py` | 363 |
| `tests/models/fixtures/native_checkpoint_configs/glm53_dflash2/config.json` | 61 |
| `tests/models/fixtures/native_checkpoint_configs/glm53_flash/config.json` | 2,636 |
| `tests/models/fixtures/native_checkpoint_configs/provenance.json` | 19 |
| `tests/models/fixtures/native_checkpoint_configs/qwen38_flash_next/config.json` | 168 |
| `tests/models/glm5next/test_aux_hidden_states_pp.py` | 227 |
| `tests/models/glm5next/test_drafter_rope_fit.py` | 199 |
| `tests/models/glm5next/test_drafter_selector_shard.py` | 180 |
| `tests/models/glm5next/test_kda_spec_bound.py` | 173 |
| `tests/models/glm5next/test_pp_fold_draft_fc.py` | 189 |
| `tests/models/glm5next/test_router_logits_dtype.py` | 156 |
| `tests/models/kimi_k3/test_kda_zero_draft_rows.py` | 101 |
| `tests/models/qwen4_exp/fixtures/exl3_names.json` | 451 |
| `tests/models/qwen4_exp/test_exl3_construction.py` | 273 |
| `tests/models/qwen4_exp/test_exl3_layout.py` | 220 |
| `tests/models/qwen4_exp/test_exl3_ple_decode.py` | 233 |
| `tests/models/qwen4_exp/test_exl3_ple_reader.py` | 1,683 |
| `tests/models/qwen4_exp/test_mtp_local_argmax.py` | 224 |
| `tests/models/qwen4_exp/test_ple_replay_cpu.py` | 153 |
| `tests/models/qwen4_exp/test_precision_switches.py` | 231 |
| `tests/models/qwen4_exp/test_qsa_canonical_topk.py` | 402 |
| `tests/models/qwen4_exp/test_qsa_canonical_topk_gpu.py` | 239 |
| `tests/models/qwen4_exp/test_qsa_sorted_selection.py` | 182 |
| `tests/models/test_glm5next_local_logits.py` | 73 |
| `tests/models/test_native_checkpoint_config_meta.py` | 416 |
| `tests/models/test_native_checkpoint_wrapper_meta.py` | 268 |
| `tests/parser/engine/test_deepseek_v41_tool_args.py` | 340 |
| `tests/parser/engine/test_glm47_tool_choice_none.py` | 276 |
| `tests/plugins_tests/conftest.py` | 17 |
| `tests/sm80/gpu_lock_guard.py` | 86 |
| `tests/sm80/gpu_lock_guard_cases.py` | 25 |
| `tests/sm80/launch/conftest.py` | 10 |
| `tests/sm80/launch/helpers.py` | 510 |
| `tests/sm80/launch/test_launch_ambient.py` | 281 |
| `tests/sm80/launch/test_launch_checkpoint.py` | 365 |
| `tests/sm80/launch/test_launch_dependencies.py` | 154 |
| `tests/sm80/launch/test_launch_engine_declarations.py` | 79 |
| `tests/sm80/launch/test_launch_guards.py` | 328 |
| `tests/sm80/launch/test_launch_guards2.py` | 269 |
| `tests/sm80/launch/test_launch_guards3.py` | 1,124 |
| `tests/sm80/launch/test_launch_hash_cache.py` | 50 |
| `tests/sm80/launch/test_launch_knobs.py` | 109 |
| `tests/sm80/launch/test_launch_real_parser.py` | 28 |
| `tests/sm80/launch/test_launch_registry.py` | 154 |
| `tests/sm80/launch/test_launch_resolve.py` | 352 |
| `tests/sm80/launch/test_launch_single_card_model.py` | 227 |
| `tests/sm80/launch/test_launch_validated.py` | 483 |
| `tests/sm80/seams.yaml` | 1,645 |
| `tests/sm80/test_ce_ring_all_reduce.py` | 288 |
| `tests/sm80/test_dev_ab.py` | 697 |
| `tests/sm80/test_dev_ab_atomic.py` | 137 |
| `tests/sm80/test_dev_ab_batch.py` | 330 |
| `tests/sm80/test_dev_ab_dynamic.py` | 487 |
| `tests/sm80/test_drafter_registrations.py` | 126 |
| `tests/sm80/test_exl3_platform_verification.py` | 65 |
| `tests/sm80/test_exl3_plugin.py` | 110 |
| `tests/sm80/test_glm_pp4_draft_fc_staging.py` | 86 |
| `tests/sm80/test_glm_pp4_kv_plan.py` | 89 |
| `tests/sm80/test_glm_pp4_mamba_split.py` | 263 |
| `tests/sm80/test_glm_registrations.py` | 309 |
| `tests/sm80/test_gpu_lock_guard.py` | 74 |
| `tests/sm80/test_hot_swap.py` | 701 |
| `tests/sm80/test_hot_swap_gpu.py` | 124 |
| `tests/sm80/test_import_cuda_init.py` | 159 |
| `tests/sm80/test_indexer_deep_gemm_gates.py` | 57 |
| `tests/sm80/test_jit_extension_cache.py` | 160 |
| `tests/sm80/test_jit_extension_std.py` | 68 |
| `tests/sm80/test_kernel_warmup_call.py` | 69 |
| `tests/sm80/test_kv_planning.py` | 492 |
| `tests/sm80/test_modification_notices.py` | 80 |
| `tests/sm80/test_moe_layers.py` | 297 |
| `tests/sm80/test_mrv2_prepare_inputs_padding.py` | 126 |
| `tests/sm80/test_plugin_banner.py` | 36 |
| `tests/sm80/test_plugin_distribution.py` | 112 |
| `tests/sm80/test_plugin_guard.py` | 299 |
| `tests/sm80/test_profile_retirement.py` | 185 |
| `tests/sm80/test_registrations.py` | 137 |
| `tests/sm80/test_seams.py` | 57 |
| `tests/sm80/test_setup_envs_load.py` | 83 |
| `tests/sm80/test_shard_argmax.py` | 384 |
| `tests/sm80/test_step_error_copy.py` | 66 |
| `tests/sm80/test_step_error_dispatch.py` | 72 |
| `tests/sm80/test_table_segment.py` | 283 |
| `tests/sm80/test_triton_mla_sparse_block_stride.py` | 151 |
| `tests/sm80/test_tuning.py` | 288 |
| `tests/sm80/tuning_rows.py` | 39 |
| `tests/sm80/update_seams.py` | 173 |
| `tests/test_retired_env_vars.py` | 319 |
| `tests/v1/attention/test_glm_metadata_reuse.py` | 228 |
| `tests/v1/attention/test_indexer_logits_budget_exact.py` | 224 |
| `tests/v1/attention/test_indexer_prefill_chunk_budget.py` | 177 |
| `tests/v1/attention/test_kpool_tail_padding_invariant.py` | 117 |
| `tests/v1/attention/test_sm80_triton_mla_sparse_decode.py` | 76 |
| `tests/v1/core/test_fair_prefill.py` | 938 |
| `tests/v1/core/test_mamba_decode_progress.py` | 101 |
| `tests/v1/core/test_mamba_inflight_states.py` | 291 |
| `tests/v1/core/test_pp_spread_decodes.py` | 79 |
| `tests/v1/core/test_scheduler_failed_inputs.py` | 108 |
| `tests/v1/kv_connector/conftest.py` | 17 |
| `tests/v1/spec_decode/dspark_host_width_cases.json` | 1 |
| `tests/v1/spec_decode/test_adaptive_k.py` | 223 |
| `tests/v1/spec_decode/test_dspark_confidence_dump.py` | 308 |
| `tests/v1/spec_decode/test_dspark_depth_hint.py` | 162 |
| `tests/v1/spec_decode/test_dspark_host_width.py` | 443 |
| `tests/v1/spec_decode/test_dspark_host_width_gpu.py` | 124 |
| `tests/v1/spec_decode/test_dspark_row_mask_conf.py` | 476 |
| `tests/v1/spec_decode/test_dspark_row_mask_gpu.py` | 294 |
| `tests/v1/spec_decode/test_dspark_row_mask_kernel.py` | 568 |
| `tests/v1/spec_decode/test_dspark_row_mask_reject.py` | 402 |
| `tests/v1/spec_decode/test_dspark_width.py` | 268 |
| `tests/v1/spec_decode/test_dspark_zero_draft.py` | 392 |
| `tests/v1/spec_decode/test_glm5_dflash_accept_depth.py` | 378 |
| `tests/v1/spec_decode/test_glm5_dflash_adaptive_k.py` | 453 |
| `tests/v1/spec_decode/test_glm5_dflash_draft_width.py` | 562 |
| `tests/v1/worker/test_deferred_state_copies_cpu.py` | 120 |
| `tests/v1/worker/test_gpu_async_failed_tokens.py` | 74 |
| `tests/v1/worker/test_gumbel_vocab_clamp.py` | 129 |
| `tests/v1/worker/test_mamba_block_size_resolution.py` | 92 |
| `tests/v1/worker/test_mem_attribution.py` | 70 |
| `tests/v1/worker/test_pp_deferred_block_tables.py` | 219 |
| `tests/v1/worker/test_pp_draft_tail.py` | 1,206 |
| `tests/v1/worker/test_pp_hop.py` | 187 |
| `tests/v1/worker/test_pp_split_draft_event.py` | 127 |
| `tests/v1/worker/test_sm80_cudagraph_mem.py` | 571 |
| `tests/v1/worker/test_sm80_execution_trace.py` | 124 |
| `tests/v1/worker/test_sm80_profile_retirement_gpu.py` | 161 |
| `tests/v1/worker/test_sm80_trace_export.py` | 216 |
| `tests/v1/worker/test_state_index_check.py` | 351 |
| **293 files** | **123,514** |

#### `tools/`

| Path | Lines |
|---|---:|
| `tools/sm80/build_hybrid_checkpoint.py` | 235 |
| `tools/sm80/build_override_sidecar.py` | 370 |
| `tools/sm80/export_hybrid_cases.py` | 262 |
| `tools/sm80/upstream_edit_exceptions.yaml` | 221 |
| **4 files** | **1,088** |

## 2. Patched upstream files

Every upstream-owned path that differs from the base. Hunks are counted at zero context lines, so one hunk is one edit site. Tests, docs and `.github` are paths under `tests/`, `docs/` or `.github/` and Markdown files. Everything else is product code.

### Product code

| Path | Change | + | - | Hunks |
|---|---|---:|---:|---:|
| `vllm/config/scheduler.py` | modified | 59 | 0 | 4 |
| `vllm/config/speculative.py` | modified | 59 | 0 | 5 |
| `vllm/config/vllm.py` | modified | 47 | 5 | 4 |
| `vllm/engine/arg_utils.py` | modified | 14 | 0 | 5 |
| `vllm/entrypoints/anthropic/serving.py` | modified | 9 | 1 | 3 |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | modified | 4 | 0 | 2 |
| `vllm/entrypoints/openai/responses/protocol.py` | modified | 3 | 0 | 3 |
| `vllm/envs.py` | modified | 9 | 0 | 2 |
| `vllm/model_executor/kernels/mhc/tilelang.py` | modified | 45 | 0 | 4 |
| `vllm/model_executor/layers/attention/mla_attention.py` | modified | 13 | 5 | 3 |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` | modified | 64 | 0 | 4 |
| `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` | modified | 42 | 10 | 4 |
| `vllm/model_executor/layers/utils.py` | modified | 12 | 0 | 2 |
| `vllm/model_executor/models/mimo_v2_omni.py` | modified | 4 | 1 | 3 |
| `vllm/model_executor/models/qwen3_dflash.py` | modified | 72 | 1 | 5 |
| `vllm/model_executor/models/qwen3_next.py` | modified | 4 | 0 | 2 |
| `vllm/model_executor/warmup/kernel_warmup.py` | modified | 8 | 0 | 2 |
| `vllm/models/deepseek_v41/common/engram.py` | modified | 5 | 4 | 4 |
| `vllm/models/deepseek_v41/nvidia/dspark.py` | modified | 3 | 0 | 3 |
| `vllm/models/deepseek_v41/nvidia/model.py` | modified | 3 | 0 | 3 |
| `vllm/models/glm5next/common/attention.py` | modified | 46 | 16 | 8 |
| `vllm/models/glm5next/common/kda.py` | modified | 270 | 14 | 19 |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | modified | 48 | 14 | 23 |
| `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` | modified | 8 | 6 | 8 |
| `vllm/models/glm5next/nvidia/sparse_indexer.py` | modified | 123 | 16 | 18 |
| `vllm/models/kimi_k3/nvidia/kda_metadata.py` | modified | 3 | 0 | 2 |
| `vllm/models/kimi_k3/nvidia/ops/recoverssm.py` | modified | 3 | 1 | 3 |
| `vllm/models/qwen4_exp/common/ngram_embedding.py` | modified | 4 | 0 | 2 |
| `vllm/models/qwen4_exp/nvidia/model.py` | modified | 4 | 0 | 4 |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | modified | 27 | 4 | 7 |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | modified | 2 | 0 | 2 |
| `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` | modified | 3 | 0 | 3 |
| `vllm/parser/abstract_parser.py` | modified | 4 | 0 | 2 |
| `vllm/parser/deepseek_v41.py` | modified | 5 | 0 | 3 |
| `vllm/parser/glm47_moe.py` | modified | 19 | 0 | 3 |
| `vllm/parser/parser_manager.py` | modified | 9 | 0 | 2 |
| `vllm/renderers/online_renderer.py` | modified | 15 | 0 | 4 |
| `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py` | modified | 3 | 1 | 3 |
| `vllm/third_party/flash_linear_attention/ops/chunk_o.py` | modified | 3 | 1 | 3 |
| `vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py` | modified | 3 | 1 | 3 |
| `vllm/third_party/flash_linear_attention/ops/cumsum.py` | modified | 4 | 2 | 4 |
| `vllm/third_party/flash_linear_attention/ops/index.py` | modified | 13 | 2 | 3 |
| `vllm/third_party/flash_linear_attention/ops/kda.py` | modified | 8 | 6 | 8 |
| `vllm/third_party/flash_linear_attention/ops/l2norm.py` | modified | 5 | 2 | 4 |
| `vllm/third_party/flash_linear_attention/ops/solve_tril.py` | modified | 5 | 3 | 5 |
| `vllm/third_party/flash_linear_attention/ops/utils.py` | modified | 38 | 9 | 9 |
| `vllm/third_party/flash_linear_attention/ops/wy_fast.py` | modified | 3 | 1 | 3 |
| `vllm/tokenizers/deepseek_v41.py` | modified | 5 | 0 | 2 |
| `vllm/v1/attention/backends/mla/indexer.py` | modified | 6 | 2 | 4 |
| `vllm/v1/attention/backends/recoverssm_metadata.py` | modified | 5 | 0 | 2 |
| `vllm/v1/core/kv_cache_planning.py` | modified | 3 | 0 | 3 |
| `vllm/v1/core/kv_cache_utils.py` | modified | 7 | 19 | 5 |
| `vllm/v1/core/sched/scheduler.py` | modified | 486 | 12 | 41 |
| `vllm/v1/executor/multiproc_executor.py` | modified | 3 | 2 | 3 |
| `vllm/v1/kv_cache_interface.py` | modified | 70 | 1 | 5 |
| `vllm/v1/outputs.py` | modified | 6 | 0 | 2 |
| `vllm/v1/worker/gpu/async_utils.py` | modified | 25 | 0 | 7 |
| `vllm/v1/worker/gpu/block_table.py` | modified | 11 | 3 | 4 |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | modified | 141 | 21 | 17 |
| `vllm/v1/worker/gpu/model_runner.py` | modified | 209 | 13 | 35 |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | modified | 44 | 1 | 7 |
| `vllm/v1/worker/gpu/model_states/recoverssm.py` | modified | 4 | 9 | 3 |
| `vllm/v1/worker/gpu/pp_utils.py` | modified | 222 | 8 | 15 |
| `vllm/v1/worker/gpu/sample/gumbel.py` | modified | 9 | 1 | 2 |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | modified | 4 | 3 | 3 |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | modified | 100 | 19 | 6 |
| `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | modified | 13 | 3 | 4 |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | modified | 282 | 0 | 7 |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | modified | 3 | 0 | 3 |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | modified | 8 | 2 | 3 |
| `vllm/v1/worker/gpu/warmup.py` | modified | 3 | 0 | 3 |
| `vllm/v1/worker/gpu_model_runner.py` | modified | 7 | 0 | 3 |
| `vllm/v1/worker/gpu_worker.py` | modified | 85 | 4 | 10 |
| **73 files** | | **2,918** | **249** | **424** |

### Tests, docs and .github

| Path | Change | + | - | Hunks |
|---|---|---:|---:|---:|
| `.github/ISSUE_TEMPLATE/100-documentation.yml` | deleted | 0 | 29 | 1 |
| `.github/ISSUE_TEMPLATE/200-installation.yml` | deleted | 0 | 47 | 1 |
| `.github/ISSUE_TEMPLATE/300-usage.yml` | deleted | 0 | 45 | 1 |
| `.github/ISSUE_TEMPLATE/400-bug-report.yml` | deleted | 0 | 108 | 1 |
| `.github/ISSUE_TEMPLATE/450-ci-failure.yml` | deleted | 0 | 69 | 1 |
| `.github/ISSUE_TEMPLATE/500-feature-request.yml` | deleted | 0 | 38 | 1 |
| `.github/ISSUE_TEMPLATE/700-performance-discussion.yml` | deleted | 0 | 59 | 1 |
| `.github/ISSUE_TEMPLATE/750-RFC.yml` | deleted | 0 | 52 | 1 |
| `.github/ISSUE_TEMPLATE/config.yml` | modified | 12 | 2 | 3 |
| `README.md` | modified | 292 | 77 | 30 |
| `docs/features/quantization/README.md` | modified | 61 | 0 | 2 |
| `docs/features/speculative_decoding/README.md` | modified | 3 | 0 | 2 |
| `tests/conftest.py` | modified | 9 | 0 | 3 |
| `tests/entrypoints/anthropic/test_anthropic_messages_conversion.py` | modified | 43 | 0 | 1 |
| `tests/models/kimi_k3/test_kda_metadata.py` | modified | 3 | 1 | 1 |
| `tests/parser/engine/test_deepseek_v41.py` | modified | 759 | 1 | 4 |
| `tests/test_envs.py` | modified | 62 | 0 | 3 |
| `tests/tokenizers_/test_deepseek_v41.py` | modified | 57 | 0 | 1 |
| `tests/v1/core/test_kv_cache_utils.py` | modified | 2 | 0 | 2 |
| `tests/v1/core/test_scheduler.py` | modified | 19 | 20 | 9 |
| `tests/v1/core/test_single_type_kv_cache_manager.py` | modified | 105 | 0 | 2 |
| `tests/v1/core/utils.py` | modified | 5 | 0 | 3 |
| `tests/v1/worker/test_gpu_input_batch_v2.py` | modified | 28 | 0 | 2 |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | modified | 4 | 1 | 4 |
| `tests/v1/worker/test_gpu_warmup_blocks.py` | modified | 9 | 1 | 5 |
| `tests/v1/worker/test_grammar_invalid_drafts.py` | modified | 6 | 0 | 2 |
| `tests/v1/worker/test_mamba_hybrid_model_state.py` | modified | 2 | 0 | 1 |
| **27 files** | | **1,481** | **550** | **88** |

### Totals

| Group | Files | + | - | Hunks |
|---|---:|---:|---:|---:|
| Product code | 73 | 2,918 | 249 | 424 |
| Tests, docs and .github | 27 | 1,481 | 550 | 88 |
| All | 100 | 4,399 | 799 | 512 |

## 3. Fork environment variables

Declared in `vllm/sm80/envs.py`, which `vllm/envs.py` registers into `environment_variables`. Retired names (`RETIRED_ENV_VARS`) are not listed: they are ignored and only warn.

The default is the value the accessor uses when the variable is unset, as written in the code (environment strings are quoted). Type and declared default come from `if TYPE_CHECKING:`.

| Name | Type | Declared default | Default |
|---|---|---|---|
| `VLLM_AMPERE_DECODE_IDX_GLUE` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_DECODE_KERNELS` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_KDA_FOLD` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_KDA_RECOVER_OFF` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_KDA_SMALL_STACK` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_MARLIN_BUILD` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_MARLIN_DECODE_CUDA` | `bool` | `False` | `'0'` |
| `VLLM_AMPERE_MARLIN_DECODE_V2` | `bool` | `False` | `'0'` |
| `VLLM_CUDAGRAPH_ESTIMATE_SPLIT` | `int` | `0` | `'0'` |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | `str` | `''` | `None` |
| `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_ATTN_PIPELINE` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_FUSED_MHC` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_FUSED_ROUTE` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_GATE_GEMV` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_MOE_PREFILL_TC` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_ROUTER_MMA` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_SWA_META_BATCH` | `bool` | `False` | `'0'` |
| `VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM` | `bool` | `False` | `'0'` |
| `VLLM_DEV_AB_SWITCH` | `str` | `''` | `''` |
| `VLLM_DFLASH2_SELECTOR_SHARD` | `bool` | `False` | `'0'` |
| `VLLM_DSPARK_ADAPTIVE_DEPTH` | `str` | `''` | `''` |
| `VLLM_DSPARK_CONFIDENCE_DUMP` | `str` | `''` | `''` |
| `VLLM_DSPARK_DEPTH_HINT` | `str` | `''` | `''` |
| `VLLM_DSPARK_PER_K_WIDTH` | `bool` | `False` | `'0'` |
| `VLLM_ENGRAM_DIRECT_READ` | `bool` | `False` | `'0'` |
| `VLLM_ENGRAM_PLACEMENT` | `str` | `'ssd'` | `'ssd'` |
| `VLLM_ENGRAM_ROW_CACHE_GIB` | `float` | `1.0` | `'1'` |
| `VLLM_ENGRAM_ROW_CACHE_PRELOAD` | `bool` | `False` | `'0'` |
| `VLLM_ENGRAM_SEGMENT` | `str` | `''` | `''` |
| `VLLM_ENGRAM_TELEMETRY_STEPS` | `int` | `0` | `'0'` |
| `VLLM_EXL3_DENSE_MMA` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE` | `str` | `''` | `''` |
| `VLLM_EXL3_DENSE_PREFILL_TC` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE` | `str` | `'planned'` | `'planned'` |
| `VLLM_EXL3_DENSE_SPLITK` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_EXPERTS_CONTIGUOUS` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_HYBRID` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_MOE_PREFILL_TC` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_MOE_PREFILL_UNCHUNKED` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_MOE_V2` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_MOE_V2_DRAFT` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS` | `int` | `0` | `'0'` |
| `VLLM_EXL3_MOE_V2_FOLD` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_MOE_V2_RING` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_OVERRIDE` | `str` | `''` | `''` |
| `VLLM_EXL3_OVERRIDE_SHA256` | `str` | `''` | `''` |
| `VLLM_EXL3_PLE_COLD` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_PLE_HOSTFN_READ` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_PLE_NATIVE_READ` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_PLE_OVERLAP` | `bool` | `False` | `'0'` |
| `VLLM_EXL3_PLE_PLACEMENT` | `str` | `'ssd'` | `'ssd'` |
| `VLLM_EXL3_PLE_READ_RETRIES` | `int` | `3` | `'3'` |
| `VLLM_EXL3_PLE_TIMING` | `bool` | `False` | `'0'` |
| `VLLM_FAIR_PREFILL_SHORT_BYPASS` | `bool` | `False` | `'0'` |
| `VLLM_FLA_AUTOTUNE_SKIP_SPILLING` | `bool` | `False` | `'0'` |
| `VLLM_FLA_PIN_AUTOTUNE` | `bool` | `False` | `'0'` |
| `VLLM_GLM5_DECODE_IDX_GLUE_PARTS` | `str` | `'weights,glue,fwht,cache,moesum'` | `'weights,glue,fwht,cache,moesum'` |
| `VLLM_GLM5_DECODE_MHC_V2` | `bool` | `False` | `'0'` |
| `VLLM_GLM5_DECODE_MOE_ROUTING` | `bool` | `True` | `'1'` |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE` | `str` | `''` | `''` |
| `VLLM_GLM5_DRAFTER_ROPE_FIT` | `bool` | `False` | `'0'` |
| `VLLM_GLM5_MOE_ROUTE_V2_MASK` | `bool` | `True` | `'1'` |
| `VLLM_GLM5_PP_FOLD_DRAFT_FC` | `bool` | `False` | `'0'` |
| `VLLM_GLM5_SMLA_PREFILL_PRED_LOAD` | `bool` | `False` | `'0'` |
| `VLLM_HOT_SWAP` | `str` | `''` | `''` |
| `VLLM_INDEXER_DECODE_RAW_K` | `bool` | `False` | `'0'` |
| `VLLM_INDEXER_DSA_REF` | `bool` | `False` | `'0'` |
| `VLLM_KV_MAMBA_INFLIGHT_STATES` | `bool` | `False` | `'0'` |
| `VLLM_KV_PLAN_LOG` | `bool` | `False` | `'0'` |
| `VLLM_MEM_ATTRIBUTION` | `bool` | `False` | `'0'` |
| `VLLM_MOE_DECODE_ROUTE_V2` | `bool` | `False` | `'0'` |
| `VLLM_MOE_DETERMINISTIC_ALIGN` | `int` | `0` | `'0'` |
| `VLLM_MOE_MASK_PADDING` | `bool` | `False` | `'0'` |
| `VLLM_PP_DRAFT_TAIL_STAGE` | `int` | `-1` | `'-1'` |
| `VLLM_PP_HOP_NO_METADATA` | `bool` | `False` | `'0'` |
| `VLLM_PP_PACKED_HOP` | `bool` | `False` | `'0'` |
| `VLLM_PP_SPLIT_DRAFT_EVENT` | `bool` | `False` | `'0'` |
| `VLLM_PP_SPREAD_DECODES` | `bool` | `False` | `'0'` |
| `VLLM_SHARED_EXPERT_REORDER` | `bool` | `False` | `'0'` |
| `VLLM_SM80_ALLREDUCE` | `bool` | `False` | `'0'` |
| `VLLM_SM80_ALLREDUCE_ROUND_ONCE` | `bool` | `False` | `'0'` |
| `VLLM_SM80_CAPTURE` | `str` | `''` | `''` |
| `VLLM_SM80_EXECUTION_TRACE` | `bool` | `False` | `'0'` |
| `VLLM_SM80_KDA_PREFILL` | `bool` | `False` | `'0'` |
| `VLLM_SM80_MARLIN_PREFILL` | `bool` | `False` | `'0'` |
| `VLLM_SM80_MHC_PREFILL` | `bool` | `False` | `'0'` |
| `VLLM_SM80_PREFILL_REDUCE_OVERLAP` | `bool` | `False` | `'0'` |
| `VLLM_SM80_REPLAY_DIR` | `str` | `''` | `''` |
| `VLLM_SM80_REPLAY_LIMIT` | `int \| None` | `None` | - |
| `VLLM_SM80_SERIAL_TRACE_EXPORT` | `bool` | `False` | `'0'` |
| `VLLM_SM80_SHARD_ARGMAX` | `bool` | `False` | `'0'` |
| `VLLM_SM80_SPARSE_MLA_PREFILL` | `bool` | `False` | `'0'` |
| `VLLM_SM80_TOPK_CANONICAL_TIES` | `bool` | `False` | `'0'` |
| `VLLM_SM80_TOPK_SORTED` | `bool` | `False` | `'0'` |
| `VLLM_SM80_TUNING_FILE` | `str` | `''` | `''` |
| `VLLM_STATE_INDEX_CHECK` | `bool` | `False` | `'0'` |
| `VLLM_THIN_GEMM` | `bool` | `False` | `'0'` |
| `VLLM_THIN_GEMM_CONCAT` | `bool` | `False` | `'0'` |
| `VLLM_THIN_GEMM_GLUON` | `bool` | `False` | `'0'` |
| `VLLM_THIN_GEMM_GLUON_FORCE_M` | `int` | `0` | `'0'` |
| `VLLM_TOOL_CHOICE_NONE_MASK` | `bool` | `False` | `'0'` |
| **111 variables** | | | |

## 4. Unreferenced fork code

Functions, methods and classes in fork-added product files whose name occurs nowhere under `vllm/` except where it is defined (a class's methods are checked only when the class is used). A name that matches any other identifier or string counts as used, so this is a lower bound. 'Used by tests' means only tests name it.

| Path | Line | Name | Lines | Used by tests |
|---|---:|---|---:|---|
| `vllm/models/deepseek_v41/nvidia/ops/sm80_attn_core.py` | 914 | `forward_layer` | 19 | yes |
| `vllm/models/deepseek_v41/nvidia/ops/sm80_router_mma.py` | 205 | `gate_route` | 14 | yes |
| `vllm/models/deepseek_v41/nvidia/sm80_engram.py` | 295 | `engram_telemetry` | 4 | yes |
| `vllm/sm80/dev_ab.py` | 1647 | `get_ab_arm` | 4 | yes |
| `vllm/sm80/distributed/ce_ring_all_reduce.py` | 98 | `emulate_owner` | 7 | yes |
| `vllm/sm80/spec_decode/dspark_row_mask.py` | 362 | `_reset_for_tests` | 3 | yes |
| **6 definitions** | | | **51** | |

## 5. Health numbers

Fork-added product code is new files under `vllm/` or `csrc/` plus new build files, without tests or Markdown.

| Measure | Value |
|---|---:|
| Fork-added product files | 227 |
| Fork-added product lines | 79,261 |
| Patched upstream files | 100 |
| Patched upstream lines | +4,399 / -799 |
| Patched upstream product files | 73 |
| Patched upstream product lines | +2,918 / -249 |
| Product lines added (new files and patches) | 82,179 |
| Fork environment variables (flags) | 111 |
| Unreferenced fork definitions | 6 (51 lines) |
