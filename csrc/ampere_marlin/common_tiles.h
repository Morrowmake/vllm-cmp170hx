// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Settings of the explicit-tile prefill translation units of _ampere_marlin_C
// (prefill_tiles.cu, kernels_tiles_sm80.cu): this directory's copy of vLLM's
// Marlin MoE GEMM (marlin_moe_kernel.h, marlin_moe_template.h,
// marlin_moe_ops.cuh) under its own namespace, with caller-owned fp32
// scratch, thread_k 64 tiles with every warp along N (thread_n / 2 threads,
// 128 to 256), and stream-K CTA indices handed out in start order: a CTA only
// ever waits for CTAs that have already started, so a kernel on another stream
// holding SMs can slow the GEMM but never stall it; schedule and arithmetic
// are those of blockIdx.x order. The kernel set is kernel_selector_tiles.h:
// Marlin's own exec-config choices for 16-row blocks, the (64,512)
// all-warps-along-N tile for 32..64-row blocks, and the fast_dequant twin of
// each for the PP whole-expert (N=2048) lists with their redo protocol
// (marlin_moe_kernel.h). vLLM's own MoE library does not use these files.
#pragma once

#define TORCH_TARGET_VERSION 0x020B000000000000ULL
#define MARLIN_NAMESPACE_NAME ampere_marlin_tiles
