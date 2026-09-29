// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// clang-format off
// The three Marlin instantiations of _ampere_marlin_C (kernel_selector_wide.h).
#include "libtorch_stable/moe/ampere_marlin/common.h"
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 8, false, 3, 8, false>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

}
