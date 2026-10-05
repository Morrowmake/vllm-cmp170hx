// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Settings of the explicit-tile prefill translation units of _ampere_marlin_C
// (prefill_tiles.cu, kernels_tiles_sm80.cu): the Marlin MoE GEMM template of
// ../marlin_moe_wna16/ under its own namespace, with caller-owned fp32
// scratch, thread_k 64 tiles with every warp along N (thread_n / 2 threads,
// 128 to 256), and stream-K CTA indices handed out in start order
// (MARLIN_MOE_ORDERED_STREAM_K): a CTA only ever waits for CTAs that have
// already started, so a kernel on another stream holding SMs can slow the GEMM
// but never stall it; schedule and arithmetic are those of blockIdx.x order.
// The kernel set is kernel_selector_tiles.h: Marlin's own exec-config choices
// for 8..64-row blocks plus the (64,512) and (64,256) all-warps-along-N tiles,
// and the fast_dequant twins of the PP whole-expert set with their redo
// protocol (MARLIN_MOE_FAST_DEQUANT_REDO, ../marlin_moe_wna16/kernel.h).
// Nothing here changes _moe_C_stable_libtorch or the K128 prefill_gemm units.
#pragma once

#define TORCH_TARGET_VERSION 0x020B000000000000ULL
#define MARLIN_MOE_PREALLOCATED_SCRATCH
#define MARLIN_MOE_TILE_THREADS_ALONG_N
#define MARLIN_MOE_ORDERED_STREAM_K
#define MARLIN_MOE_FAST_DEQUANT_REDO

#define MARLIN_NAMESPACE_NAME ampere_marlin_tiles
#define MARLIN_MOE_KERNEL_SELECTOR \
  "libtorch_stable/moe/ampere_marlin/kernel_selector_tiles.h"
