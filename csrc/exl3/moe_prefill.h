#pragma once

#include <torch/types.h>

#include <cstdint>

namespace exl3 {

// moe_out on the tensor cores for prefill-sized calls of one served shape
// (moe_prefill.cu): hidden 2560, intermediate 640, every routed matrix
// bits_x2 10 / codebook 2, at least kMoePrefillMinRows rows per call. Same
// arguments and output as moe_out; results differ from it in the last bits
// (hi/lo FP16 MMAs, FP64 route sums).
constexpr int64_t kMoePrefillMinRows = 65;
constexpr int64_t kMoePrefillMaxExperts = 4096;

int64_t moe_workspace_size_prefill(int64_t max_rows, int64_t hidden_size,
                                   int64_t intermediate_size, int64_t topk,
                                   int64_t local_experts);

void moe_out_prefill(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     torch::Tensor& out, torch::Tensor& workspace, double swiglu_limit);

}  // namespace exl3
