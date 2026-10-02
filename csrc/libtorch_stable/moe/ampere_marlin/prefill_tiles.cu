// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// _ampere_marlin_C prefill_tile_gemm: vLLM's Marlin MoE GEMM host code
// (../marlin_moe_wna16/ops.cu) compiled with the settings of common_tiles.h.
// Same arguments as prefill_gemm (explicit thread_k / thread_n /
// blocks_per_sm, or -1 for Marlin's own choice; caller-owned float32 c_tmp).
// Registered in ops.cu.
#include "libtorch_stable/moe/ampere_marlin/common_tiles.h"

#define MARLIN_MOE_NO_MOE_C_IMPL
#define moe_wna16_marlin_gemm ampere_marlin_prefill_tile_gemm
#include "libtorch_stable/moe/marlin_moe_wna16/ops.cu"
#undef moe_wna16_marlin_gemm
