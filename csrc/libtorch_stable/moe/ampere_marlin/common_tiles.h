// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Settings of the explicit-tile prefill translation units of _ampere_marlin_C
// (prefill_tiles.cu, kernels_tiles_sm80.cu): the unmodified Marlin MoE GEMM
// template of ../marlin_moe_wna16/ under its own namespace, with
// caller-owned fp32 scratch and two host-side rules for explicit tiles:
// thread_k 64 tiles put every warp along N (thread_n / 2 threads, 128 to 256),
// and a launch asking for more than one CTA per SM checks that occupancy
// allows it (the stream-K locks need every CTA resident). The kernel set is
// kernel_selector_tiles.h: Marlin's own exec-config choices for 8..64-row
// blocks plus the (64,512) and (64,256) all-warps-along-N tiles.
// Nothing here changes _moe_C_stable_libtorch or the K128 prefill_gemm units.
#pragma once

#define TORCH_TARGET_VERSION 0x020B000000000000ULL
#define MARLIN_MOE_PREALLOCATED_SCRATCH
#define MARLIN_MOE_TILE_THREADS_ALONG_N
#define MARLIN_MOE_CHECK_RESIDENT_BLOCKS

#define MARLIN_NAMESPACE_NAME ampere_marlin_tiles
#define MARLIN_MOE_KERNEL_SELECTOR \
  "libtorch_stable/moe/ampere_marlin/kernel_selector_tiles.h"
