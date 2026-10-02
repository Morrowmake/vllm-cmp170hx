// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// clang-format off
// The 21 instantiations of kernels_tiles_sm80.cu (prefill_tile_gemm): bf16
// activations and output, uint4b8 weights, bf16 group-128 scales, 4 stages.
// m_blocks <= 1: (128,128,256) (64,128,128) (128,64,128) as (k, n, threads);
// m_blocks  > 1: (64,256,256) (64,128,128) (128,64,128) (64,512,256) (64,256,128).
if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 32 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 32, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 16, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 32 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 32, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 16, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 32 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 32, 4, false, 4, 8, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 16, 4, false, 4, 8, false>;
