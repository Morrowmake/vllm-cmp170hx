// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// clang-format off
// The Marlin instantiations of prefill_tile_gemm (kernel_selector_tiles.h).
#include "libtorch_stable/moe/ampere_marlin/common_tiles.h"
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 32, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 32, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 32, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

}
