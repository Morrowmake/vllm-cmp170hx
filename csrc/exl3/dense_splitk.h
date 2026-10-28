#pragma once
// Split-K dense EXL3 linear (VLLM_EXL3_DENSE_SPLITK; dense_splitk.cu).

#include <torch/types.h>

#include <cstdint>

namespace exl3 {

// Same signature, checks and workspace (linear_workspace_size) as
// linear_out. Launches whose split factor (below) is 2 run linear_input,
// which also zeroes the result rows, the split-K pipelined blocks and an
// FP64 linear_output; every other launch is linear_out itself.
void linear_out_splitk(const torch::Tensor& x, const torch::Tensor& trellis,
                       const torch::Tensor& suh, const torch::Tensor& svh,
                       int64_t bits_x2, int64_t codebook, int64_t logical_in,
                       int64_t logical_out, torch::Tensor& out,
                       torch::Tensor& workspace);

// K split factor (1 or 2) that linear_out_splitk uses for `rows` rows of a
// matrix with `padded_out` columns stored as (bits_x2, codebook) on a device
// with `sms` multiprocessors. Host arithmetic only.
int64_t linear_splitk_splits(int64_t rows, int64_t padded_out, int64_t sms,
                             int64_t bits_x2, int64_t codebook);

}  // namespace exl3
