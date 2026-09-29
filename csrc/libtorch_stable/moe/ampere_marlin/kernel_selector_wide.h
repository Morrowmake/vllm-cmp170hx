// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// clang-format off
// The instantiations in kernels_sm80.cu: bf16 activations and output,
// uint4b8 weights, bf16 group-128 scales, thread_k_blocks 8.
// 32/48-row lists use N128/128 threads; 64-row lists use N256/256 threads.
// The 48-row tile uses 3 stages, the other tiles use 4 stages.
if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 8, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 3 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 8, false, 3, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 8, false, 4, 8, false>;
