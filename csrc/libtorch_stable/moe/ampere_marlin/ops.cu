// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// _ampere_marlin_C: vLLM's Marlin MoE GEMM host code (../marlin_moe_wna16/
// ops.cu) compiled with the settings of common.h and registered as
// torch.ops._ampere_marlin_C.prefill_gemm (and registers prefill_tile_gemm of
// prefill_tiles.cu). Its final argument is caller-owned
// float32 reduction scratch. Explicit K128/N128 tiles are built for 32/48
// rows and K128/N256 for 64 rows; any other request fails
// with "Unsupported shapes".
#include "libtorch_stable/moe/ampere_marlin/common.h"

#define MARLIN_MOE_NO_MOE_C_IMPL
// keep the host entry point out of the way of _moe_C's symbol of that name
#define moe_wna16_marlin_gemm ampere_marlin_prefill_gemm
#include "libtorch_stable/moe/marlin_moe_wna16/ops.cu"
#undef moe_wna16_marlin_gemm

// prefill_tiles.cu: the same host code over the explicit-tile kernel set.
torch::stable::Tensor ampere_marlin_prefill_tile_gemm(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none,
    torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    torch::stable::Tensor& workspace, torch::stable::Tensor& sorted_token_ids,
    torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& num_tokens_past_padded,
    torch::stable::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, vllm::ScalarTypeId const& b_type_id, int64_t size_m,
    int64_t size_n, int64_t size_k, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm, torch::stable::Tensor& c_tmp);

STABLE_TORCH_LIBRARY(_ampere_marlin_C, m) {
  m.def(
      "prefill_gemm(Tensor! a, Tensor? c_or_none,"
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
      "int thread_k, int thread_n, int blocks_per_sm, Tensor(c!) c_tmp) -> Tensor");
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
      "int thread_k, int thread_n, int blocks_per_sm, Tensor(c!) c_tmp) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_ampere_marlin_C, CUDA, m) {
  m.impl("prefill_gemm", TORCH_BOX(&ampere_marlin_prefill_gemm));
  m.impl("prefill_tile_gemm", TORCH_BOX(&ampere_marlin_prefill_tile_gemm));
}


