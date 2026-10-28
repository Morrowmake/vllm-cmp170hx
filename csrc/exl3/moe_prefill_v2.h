#pragma once
// EXL3 routed-expert MoE on the tensor cores for prefill-sized calls of the
// DeepSeek-V4.1-Flash experts (VLLM_DEEPSEEK_V41_MOE_PREFILL_TC;
// moe_prefill_v2.cu): hidden 5120, intermediate 2304, every routed matrix of a
// layer stored with one bits_x2 of 6 or 4 and codebook 2; the caller passes the
// layer's own bits_x2 (the kernels stop the device on an expert stored with
// another one, which they read from packed_meta). Registered into
// _exl3_C by its own library fragment. Same arguments and output as
// moe_out_v2; results differ from moe_out in the last bits (hi/lo FP16 MMAs,
// FP64 route sums).

#include <torch/types.h>

#include <cstdint>

namespace exl3 {

// FP32 workspace elements one call of up to max_rows rows needs: calls run in
// row chunks of at most 512 rows, so the size stops growing at 512 rows.
int64_t moe_workspace_size_prefill_v2(int64_t max_rows, int64_t hidden_size,
                                      int64_t intermediate_size, int64_t topk,
                                      int64_t local_experts);

void moe_out_prefill_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                        const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
                        const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
                        int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
                        torch::Tensor& workspace, double swiglu_limit);

}  // namespace exl3
