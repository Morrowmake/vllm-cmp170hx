// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Settings shared by the prefill translation units of _ampere_marlin_C:
// MoE GEMM of ../marlin_moe_wna16/ compiled under its own namespace, with
// (thread_k 128, thread_n 256) tiles at 256 threads for whole-expert prefill
// on sm_80, and an early return for a launch whose block list is empty.
// Nothing here changes _moe_C_stable_libtorch: it is built without these
// macros.
#pragma once

// The decode translation unit uses ATen and is version-locked by build_info().
// Only these prefill translation units use the stable libtorch interface.
#define TORCH_TARGET_VERSION 0x020B000000000000ULL
#define MARLIN_MOE_PREALLOCATED_SCRATCH

#define MARLIN_NAMESPACE_NAME ampere_marlin
#define MARLIN_MOE_EMPTY_LIST_RETURN
#define MARLIN_MOE_MAX_THREADS 256
#define MARLIN_MOE_KERNEL_SELECTOR \
  "libtorch_stable/moe/ampere_marlin/kernel_selector_wide.h"
