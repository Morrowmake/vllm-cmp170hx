// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// _ampere_marlin_C prefill_tile_gemm: this directory's copy of vLLM's Marlin
// MoE GEMM host code (marlin_moe_ops.cuh) compiled with the settings of
// common_tiles.h and registered as torch.ops._ampere_marlin_C.prefill_tile_gemm.
// Explicit thread_k / thread_n / blocks_per_sm, or -1 for Marlin's own choice;
// caller-owned float32 c_tmp and an optional fast-dequant redo area. A tile
// outside the instantiated set (kernel_selector_tiles.h) raises an error.
#include "ampere_marlin/common_tiles.h"
#include "ampere_marlin/marlin_moe_ops.cuh"

STABLE_TORCH_LIBRARY(_ampere_marlin_C, m) {
  m.def(
      "prefill_tile_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm, Tensor(c!) c_tmp, Tensor(r!)? redo=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_ampere_marlin_C, CUDA, m) {
  m.impl("prefill_tile_gemm", TORCH_BOX(&ampere_marlin_prefill_tile_gemm));
}
