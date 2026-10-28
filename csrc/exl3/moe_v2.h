#pragma once

#include <torch/types.h>
#include <cstdint>

namespace exl3 {

// moe_out on the tensor cores for the served shapes (moe_v2.cu): padded hidden
// 2560 with intermediate 640 (bits_x2 10) and padded hidden 5120 with
// intermediate 2304 (bits_x2 6 or 4; 8: the DSpark drafter's experts, with
// stages of their own), codebook 2, at most 1024 routed assignments per call.
// moe_out's arguments plus bits_x2, the one bit width of every routed matrix
// of the layer; same output, which differs from moe_out's in the last bits
// (FP64 side stages, hi/lo FP16 MMAs).
int64_t moe_workspace_size_v2(int64_t max_rows, int64_t hidden_size,
                              int64_t intermediate_size, int64_t topk,
                              int64_t local_experts);
void moe_out_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                const torch::Tensor& topk_weights,
                const torch::Tensor& packed_ptrs,
                const torch::Tensor& packed_meta,
                const torch::Tensor& expert_map, int64_t intermediate_size,
                int64_t bits_x2, torch::Tensor& out, torch::Tensor& workspace,
                double swiglu_limit);
// moe_out_v2's arguments and output bits; the activation and combine stages
// run inside the gate/up and down launches (three launches, not five).
void moe_out_v2_fold(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     int64_t bits_x2, torch::Tensor& out, torch::Tensor& workspace,
                     double swiglu_limit);
// The smallest row tile kernel (8 or 16 rows) every bits_x2 8 row tile takes;
// 0: each tile's by its rows (VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS). Set before
// the first call and any graph capture.
void moe_v2_draft_force_rows(int64_t rows);

}  // namespace exl3
