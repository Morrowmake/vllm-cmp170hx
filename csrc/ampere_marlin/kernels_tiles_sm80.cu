// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// clang-format off
// The Marlin instantiations of prefill_tile_gemm (kernel_selector_tiles.h).
#include "ampere_marlin/common_tiles.h"
#include "ampere_marlin/marlin_moe_kernel.h"
#include "ampere_marlin/marlin_moe_template.h"

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 8, false, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 8, false, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 8, false, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 32, 4, false, 4, 8, false, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 32, 4, false, 4, 8, false, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 32, 4, false, 4, 8, false, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 8, false, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 8, false, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 8, false, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 32, 4, false, 4, 8, false, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 32, 4, false, 4, 8, false, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 32, 4, false, 4, 8, false, true>( MARLIN_KERNEL_PARAMS );

}
