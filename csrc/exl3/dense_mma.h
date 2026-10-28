#pragma once
// Tensor-core dense EXL3 launches of 1..32 rows (VLLM_EXL3_DENSE_MMA;
// dense_mma.cu), reached through linear_out_splitk.
#include <torch/types.h>

#include <cstdint>

namespace exl3 {

// Called by linear_out_splitk after its checks, under its device guard.
// With the process switch on (set_dense_mma) and a launch the tensor-core
// blocks cover (one of the served matrices, bits_x2 10/12/14 with codebook
// 2, 1..32 rows, a 16-byte aligned trellis, a 4-byte aligned out holding
// the block exponents), runs the launch and returns true; otherwise
// launches nothing and returns false. Uses linear_out's workspace
// (linear_workspace_size) and, until its last kernel overwrites it, the
// first 2 * (rows rounded up to 8) * padded_in / 128 bytes of out.
bool linear_out_mma(const torch::Tensor& x, const torch::Tensor& trellis,
                    const torch::Tensor& suh, const torch::Tensor& svh,
                    int64_t bits_x2, int64_t codebook, int64_t logical_in,
                    int64_t logical_out, torch::Tensor& out,
                    torch::Tensor& workspace, int sms);

}  // namespace exl3
