#pragma once

#include <torch/types.h>
#include <cstdint>

namespace exl3 {

int64_t linear_workspace_size(int64_t max_rows, int64_t padded_in,
                              int64_t padded_out);
int64_t moe_workspace_size(int64_t max_rows, int64_t hidden_size,
                           int64_t intermediate_size, int64_t topk,
                           int64_t local_experts);

// out is BF16, or FP32 for the unrounded result (check_linear_out).
void linear_out(const torch::Tensor& x, const torch::Tensor& trellis,
                const torch::Tensor& suh, const torch::Tensor& svh,
                int64_t bits_x2, int64_t codebook, int64_t logical_in,
                int64_t logical_out, torch::Tensor& out,
                torch::Tensor& workspace);

// swiglu_limit > 0 clamps the FP32 gate from above and up on both sides
// before SiLU(gate) * up; 0 leaves them unclamped.
void moe_out(const torch::Tensor& x, const torch::Tensor& topk_ids,
             const torch::Tensor& topk_weights,
             const torch::Tensor& packed_ptrs,
             const torch::Tensor& packed_meta,
             const torch::Tensor& expert_map, int64_t intermediate_size,
             torch::Tensor& out, torch::Tensor& workspace, double swiglu_limit);

// Test-only: today's scalar kernels, the bit-exact oracle for the operators
// above (same signatures; their workspace needs never exceed the sizes above).
void linear_out_reference(const torch::Tensor& x, const torch::Tensor& trellis,
                          const torch::Tensor& suh, const torch::Tensor& svh,
                          int64_t bits_x2, int64_t codebook, int64_t logical_in,
                          int64_t logical_out, torch::Tensor& out,
                          torch::Tensor& workspace);
void moe_out_reference(const torch::Tensor& x, const torch::Tensor& topk_ids,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& packed_ptrs,
                       const torch::Tensor& packed_meta,
                       const torch::Tensor& expert_map, int64_t intermediate_size,
                       torch::Tensor& out, torch::Tensor& workspace, double swiglu_limit);
// Test-only: force one GEMM block shape (0 tiny .. 3 large) for every later
// launch in the process, or -1 for the row-count heuristic. Any shape gives
// bitwise identical rows.
void set_gemm_shape_override(int64_t shape);

}  // namespace exl3
