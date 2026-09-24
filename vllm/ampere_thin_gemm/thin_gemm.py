# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored kernel: keep in sync with the standalone kernel source.
# Revalidate changes with the standalone correctness and performance checks.
# Keep changes synchronized: the standalone repo owns the
# correctness gate (validate.py) and the benchmark (bench.py) that qualify this
# file, so a change made here and not there is a change that nothing measures.
# To update, re-copy the file and re-run the standalone gate.
"""Thin-M BF16 GEMM for sm_80 (candidate 1: split-K, FP32 partials).

    thin_gemm(x[M, K] bf16, w[N, K] bf16) -> y[M, N] bf16      == F.linear(x, w)

Keep this kernel in sync with the standalone source.
The contract with the rest of the
tree is only the `thin_gemm` signature and its semantics (BF16 x BF16, FP32
accumulate, BF16 out; no precision reduction of any kind).

Why split-K
-----------
At M <= 32 the op is weight-bandwidth bound: it reads N*K*2 bytes of weight and
does 2*M*N*K flops, so the arithmetic intensity is ~M flop/byte - 4 flop/byte at
M=4 against ~75 flop/byte of bf16 balance on this part.  The only thing that
matters is streaming the weight once at full HBM rate, which needs enough
concurrent CTAs to saturate the memory system.  A plain (M, N) tiling gives
ceil(N/BLOCK_N) CTAs; for the narrow shapes in shapes.json that is 1..8 CTAs on
a 70-SM part.  Splitting K across CTAs multiplies the CTA count without reading
any weight byte twice - each CTA owns a disjoint set of K-blocks.

The partial sums are FP32 and are reduced in the same launch: each CTA stores
its partial and bumps a per-output-tile arrival counter, and the CTA that
arrives last sums the SPLIT_K partials in fixed order 0..SPLIT_K-1 and writes
the output.  No CTA spins, and the counter only elects the reducer (it never
accumulates values), so the result is bitwise deterministic run to run.

Constraints (see ../README.md): sm_80 only, Triton 3.7.1, no TMA/wgmma, no fp8,
must be CUDA-graph capturable (see `warmup()`).
"""

import os
import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Tuning knobs.  _select_config() below is where the schedule is tuned.
# --------------------------------------------------------------------------

# Target CTA count as a multiple of the SM count.  Measured on this part
# (70 SMs, sm_80): what a weight-streaming schedule wants is *many small* CTAs,
# not few large ones.  A tile that is narrow in N reads exactly the same weight
# bytes as a wide one, gives the scheduler finer-grained work so the tail of the
# grid does not idle SMs, and lets SPLIT_K stay at 1 - which drops the whole
# reduction launch.  1.8 is the smallest target that reproduces the measured
# winner for the wide shapes without over-splitting the narrow ones.
_CTA_PER_SM_TARGET = 1.8

# Never split K so finely that a CTA reads fewer than this many K elements;
# below it the fixed per-CTA cost and the reduction pass dominate.
_MIN_K_PER_SPLIT = 256

# Largest shared-memory allocation a CTA may request (sm_80 allows 163 KiB
# opt-in).  num_stages is trimmed rather than failing to compile at a large M.
_MAX_SMEM = 150 * 1024

# Measured overrides, keyed by (N, K, M) with an (N, K, None) any-M fallback.
# Value is (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) - BLOCK_M is
# never stored here because it MUST cover M (the kernel masks rows against
# BLOCK_M, so a pinned BLOCK_M would silently drop rows at a larger M).
#
# Produced by a configuration sweep: for every (N, K) in shapes.json it
# times a pruned schedule grid with bench.py's own methodology (512 MB weight
# rotation, one CUDA graph, median of 12 replays) and keeps the winner.  The
# quoted us/cuBLAS numbers are from that sweep, not from a remembered run.
#
# Re-measured after the reduction was fused into the last CTA: the first table
# was tuned when a K split cost an extra kernel launch, so it was biased toward
# small SPLIT_K.  With the launch gone, splitting further is cheap and the
# narrow-N shapes want more, smaller CTAs.
#
# The M = 16 and M = 32 rows were then re-measured on their own with
# num_warps = 8 and BLOCK_K = 256 in the grid, which the earlier sweeps never
# reached.  At BLOCK_M = 32 the operand tiles double, so a config fitted at
# BLOCK_M = 16 can starve the machine: BLOCK_N = 128 at BLOCK_M = 32 needs
# 123 KiB of shared memory for 3 stages, i.e. one CTA per SM.  Large M wants a
# NARROWER N tile and MORE warps than small M, and it wants SPLIT_K = 1 sooner,
# because BLOCK_M = 32 already doubles the work per CTA.
#
# Finally the remaining M = 1, 2, 4, 8 rows were re-measured with the same
# widened grid.  The lesson generalised past large M: three shapes were
# on a config the widened grid beats at EVERY M, because the earlier sweeps'
# num_warps came from a rule rather than from measurement.  mla_fused_qkv_a_proj
# in particular wanted SPLIT_K = 1 with BLOCK_N = 32 at every M, not SPLIT_K = 4
# with BLOCK_N = 64 -- worth 6-7 % on its own.
#
# Every row below is a measured winner, ranked as the min of two separate
# timings; a candidate under ~2.5 % better than the incumbent was NOT taken,
# because a single timing on this part is worth +/-3 %.
_CONFIG_OVERRIDES: dict = {
    # kda_in_proj_qkvbfg_a  N=6416 K=4096
    (6416, 4096, 1): (32, 128, 1, 2, 3),  # 34.82 us vs cuBLAS 40.03 (1.150x), separate timing 34.82 us
    (6416, 4096, 2): (32, 128, 1, 2, 3),  # 35.19 us vs cuBLAS 38.35 (1.090x), separate timing 35.19 us
    (6416, 4096, 4): (32, 128, 1, 2, 3),  # 35.28 us vs cuBLAS 38.35 (1.087x), separate timing 35.28 us
    (6416, 4096, 8): (32, 128, 1, 2, 3),  # 35.75 us vs cuBLAS 38.82 (1.086x), separate timing 35.75 us
    (6416, 4096, 16): (32, 128, 1, 2, 3),  # 36.03 us vs cuBLAS 39.19 (1.088x), separate timing 36.03 us
    (6416, 4096, 24): (128, 128, 1, 4, 3),  # 37.33 us vs cuBLAS 44.78 (1.200x), fallback was 38.73
    (6416, 4096, 32): (128, 128, 1, 4, 3),  # 37.42 us vs cuBLAS 45.71 (1.221x), separate timing 37.42 us
    (6416, 4096, None): (32, 128, 1, 2, 3),
    # dense_mlp_gate_up  N=6144 K=4096
    (6144, 4096, 1): (32, 128, 1, 2, 3),  # 33.98 us vs cuBLAS 37.52 (1.104x), separate timing 33.98 us
    (6144, 4096, 2): (32, 128, 1, 2, 3),  # 34.26 us vs cuBLAS 37.33 (1.090x), separate timing 34.26 us
    (6144, 4096, 4): (32, 128, 1, 2, 3),  # 34.44 us vs cuBLAS 37.70 (1.095x), separate timing 34.44 us
    (6144, 4096, 8): (128, 128, 1, 4, 3),  # 34.54 us vs cuBLAS 37.79 (1.094x), separate timing 34.54 us
    (6144, 4096, 16): (128, 128, 1, 4, 3),  # 34.72 us vs cuBLAS 38.07 (1.097x), separate timing 34.72 us
    (6144, 4096, 24): (128, 128, 1, 4, 3),  # 35.56 us vs cuBLAS 44.22 (1.243x), fallback was 37.89
    (6144, 4096, 32): (128, 128, 1, 4, 3),  # 36.40 us vs cuBLAS 44.78 (1.230x), separate timing 36.40 us
    (6144, 4096, None): (32, 128, 1, 2, 3),
    # mla_o_proj  N=4096 K=4096
    (4096, 4096, 1): (32, 128, 1, 2, 4),  # 23.55 us vs cuBLAS 25.02 (1.062x), separate timing 23.55 us
    (4096, 4096, 2): (32, 128, 1, 4, 4),  # 23.42 us vs cuBLAS 25.79 (1.101x), separate timing 24.26 us
    (4096, 4096, 4): (32, 128, 1, 4, 4),  # 23.42 us vs cuBLAS 25.66 (1.096x), separate timing 24.19 us
    (4096, 4096, 8): (32, 128, 1, 4, 4),  # 23.68 us vs cuBLAS 25.79 (1.089x), separate timing 24.51 us
    (4096, 4096, 16): (32, 128, 1, 4, 4),  # 24.13 us vs cuBLAS 26.05 (1.080x), separate timing 24.19 us
    (4096, 4096, 24): (64, 128, 1, 8, 5),  # 24.45 us vs cuBLAS 25.73 (1.052x), fallback was 25.79
    (4096, 4096, 32): (64, 128, 1, 8, 5),  # 24.58 us vs cuBLAS 25.98 (1.057x), separate timing 27.97 us
    (4096, 4096, None): (32, 128, 1, 4, 4),
    # dense_mlp_down  N=4096 K=3072
    (4096, 3072, 1): (32, 128, 1, 2, 4),  # 18.85 us vs cuBLAS 20.20 (1.072x), separate timing 18.85 us
    (4096, 3072, 2): (64, 256, 1, 4, 3),  # 18.66 us vs cuBLAS 20.57 (1.102x), separate timing 19.55 us
    (4096, 3072, 4): (64, 256, 1, 4, 3),  # 18.71 us vs cuBLAS 20.53 (1.097x), separate timing 19.60 us
    (4096, 3072, 8): (32, 128, 1, 4, 4),  # 19.08 us vs cuBLAS 20.81 (1.090x), separate timing 19.97 us
    (4096, 3072, 16): (32, 128, 1, 4, 4),  # 19.55 us vs cuBLAS 21.09 (1.079x), separate timing 19.69 us
    (4096, 3072, 24): (64, 128, 1, 8, 4),  # 20.15 us vs cuBLAS 20.99 (1.042x), fallback was 21.18
    (4096, 3072, 32): (64, 128, 1, 8, 4),  # 20.48 us vs cuBLAS 21.22 (1.036x), separate timing 20.53 us
    (4096, 3072, None): (32, 128, 1, 4, 4),
    # mla_fused_qkv_a_proj  N=2048 K=4096
    (2048, 4096, 1): (32, 256, 1, 4, 4),  # 13.73 us vs cuBLAS 16.67 (1.214x), separate timing 14.69 us
    (2048, 4096, 2): (32, 128, 1, 4, 5),  # 13.70 us vs cuBLAS 16.54 (1.208x), separate timing 14.53 us
    (2048, 4096, 4): (32, 128, 1, 4, 5),  # 13.79 us vs cuBLAS 17.86 (1.295x), separate timing 14.82 us
    (2048, 4096, 8): (32, 128, 1, 4, 5),  # 13.89 us vs cuBLAS 17.02 (1.226x), separate timing 14.91 us
    (2048, 4096, 16): (32, 256, 1, 4, 4),  # 13.95 us vs cuBLAS 17.38 (1.245x), separate timing 15.36 us
    (2048, 4096, 32): (32, 128, 1, 4, 5),  # 15.74 us vs cuBLAS 18.02 (1.144x), separate timing 16.93 us
    (2048, 4096, None): (32, 128, 1, 4, 5),
    # kda_o_proj  N=4096 K=2048
    (4096, 2048, 1): (64, 128, 1, 4, 4),  # 13.15 us vs cuBLAS 14.75 (1.122x), separate timing 13.15 us
    (4096, 2048, 2): (16, 128, 1, 2, 4),  # 13.70 us vs cuBLAS 15.04 (1.098x), separate timing 13.70 us
    (4096, 2048, 4): (16, 128, 1, 2, 4),  # 13.70 us vs cuBLAS 15.01 (1.096x), separate timing 13.70 us
    (4096, 2048, 8): (64, 256, 1, 4, 4),  # 13.86 us vs cuBLAS 15.17 (1.095x), separate timing 13.89 us
    (4096, 2048, 16): (64, 128, 1, 8, 4),  # 13.66 us vs cuBLAS 15.20 (1.112x), separate timing 13.76 us
    (4096, 2048, 24): (64, 128, 1, 8, 4),  # 14.66 us vs cuBLAS 15.39 (1.050x), separate timing 14.53 us, fallback was 19.55
    (4096, 2048, 32): (64, 128, 1, 4, 4),  # 14.50 us vs cuBLAS 15.52 (1.071x), separate timing 14.50 us
    (4096, 2048, None): (16, 128, 1, 2, 4),
    # mla_q_b_proj+idx_wq_b  N=4096 K=1536
    (4096, 1536, 1): (64, 128, 1, 4, 4),  # 10.53 us vs cuBLAS 12.31 (1.170x), separate timing 10.53 us
    (4096, 1536, 2): (64, 256, 1, 4, 3),  # 11.05 us vs cuBLAS 12.48 (1.129x), separate timing 11.05 us
    (4096, 1536, 4): (64, 256, 1, 4, 3),  # 11.10 us vs cuBLAS 12.53 (1.129x), separate timing 11.10 us
    (4096, 1536, 8): (64, 256, 1, 4, 3),  # 11.24 us vs cuBLAS 12.69 (1.129x), separate timing 11.24 us
    (4096, 1536, 16): (64, 128, 1, 8, 4),  # 11.31 us vs cuBLAS 12.91 (1.141x), separate timing 11.43 us
    (4096, 1536, 24): (64, 128, 1, 8, 4),  # 12.12 us vs cuBLAS 12.88 (1.063x), fallback was 12.69
    (4096, 1536, 32): (64, 128, 1, 4, 4),  # 12.15 us vs cuBLAS 13.00 (1.071x), separate timing 12.15 us
    (4096, 1536, None): (64, 256, 1, 4, 3),
    # shared_gate_up  N=1024 K=4096
    (1024, 4096, 1): (32, 512, 2, 2, 3),  # 9.28 us vs cuBLAS 10.98 (1.183x), separate timing 9.58 us
    (1024, 4096, 2): (64, 128, 4, 4, 4),  # 9.41 us vs cuBLAS 11.04 (1.173x), separate timing 9.79 us
    (1024, 4096, 4): (64, 64, 4, 4, 5),  # 9.68 us vs cuBLAS 10.82 (1.117x), separate timing 9.98 us
    (1024, 4096, 8): (64, 64, 4, 4, 5),  # 9.89 us vs cuBLAS 11.02 (1.115x), separate timing 10.02 us
    (1024, 4096, 16): (32, 64, 4, 2, 5),  # 9.94 us vs cuBLAS 11.23 (1.130x), separate timing 9.94 us
    (1024, 4096, 24): (64, 256, 4, 4, 3),  # 10.51 us vs cuBLAS 11.38 (1.082x), fallback was 11.42
    (1024, 4096, 32): (64, 64, 4, 4, 5),  # 11.36 us vs cuBLAS 11.65 (1.025x), separate timing 12.54 us
    (1024, 4096, None): (64, 64, 4, 4, 5),
    # mla_kv_b_proj  N=8192 K=512
    (8192, 512, 1): (32, 64, 1, 2, 4),  # 8.11 us vs cuBLAS 9.57 (1.179x), separate timing 8.11 us
    (8192, 512, 2): (64, 64, 1, 4, 4),  # 8.26 us vs cuBLAS 11.92 (1.444x), separate timing 8.26 us
    (8192, 512, 4): (64, 64, 1, 4, 4),  # 8.27 us vs cuBLAS 12.00 (1.451x), separate timing 8.27 us
    (8192, 512, 8): (64, 64, 1, 4, 4),  # 8.42 us vs cuBLAS 12.18 (1.447x), separate timing 8.42 us
    (8192, 512, 16): (64, 64, 1, 4, 4),  # 8.62 us vs cuBLAS 12.34 (1.430x), separate timing 8.62 us
    (8192, 512, 32): (64, 64, 1, 4, 4),  # 9.42 us vs cuBLAS 10.26 (1.088x), separate timing 9.42 us
    (8192, 512, None): (64, 64, 1, 4, 4),
    # shared_down  N=4096 K=512
    (4096, 512, 1): (32, 64, 1, 2, 5),  # 5.33 us vs cuBLAS 6.60 (1.239x), separate timing 5.33 us
    (4096, 512, 2): (32, 64, 1, 2, 5),  # 5.47 us vs cuBLAS 6.62 (1.211x), separate timing 5.47 us
    (4096, 512, 4): (32, 64, 1, 2, 5),  # 5.50 us vs cuBLAS 6.60 (1.201x), separate timing 5.50 us
    (4096, 512, 8): (32, 64, 1, 2, 5),  # 5.60 us vs cuBLAS 6.70 (1.196x), separate timing 5.60 us
    (4096, 512, 16): (32, 64, 1, 2, 5),  # 5.71 us vs cuBLAS 6.80 (1.190x), separate timing 5.71 us
    (4096, 512, 24): (64, 64, 1, 8, 5),  # 6.02 us vs cuBLAS 7.04 (1.169x), fallback was 6.53
    (4096, 512, 32): (64, 64, 1, 8, 5),  # 6.19 us vs cuBLAS 7.20 (1.163x), separate timing 6.37 us
    (4096, 512, None): (32, 64, 1, 2, 5),
    # moe_router_gate  N=288 K=4096
    (288, 4096, 1): (16, 128, 8, 2, 3),  # 5.86 us vs cuBLAS 6.30 (1.076x), separate timing 6.27 us
    (288, 4096, 2): (16, 64, 8, 2, 5),  # 5.94 us vs cuBLAS 7.86 (1.324x), separate timing 6.24 us
    (288, 4096, 4): (16, 64, 8, 2, 5),  # 5.82 us vs cuBLAS 7.98 (1.371x), separate timing 6.39 us
    (288, 4096, 8): (16, 64, 8, 2, 5),  # 5.79 us vs cuBLAS 8.12 (1.401x), separate timing 6.01 us
    (288, 4096, 16): (16, 64, 8, 2, 5),  # 6.00 us vs cuBLAS 8.35 (1.391x), separate timing 6.51 us
    (288, 4096, 24): (32, 256, 4, 4, 3),  # 7.01 us vs cuBLAS 8.36 (1.193x), fallback was 7.25
    (288, 4096, 32): (32, 256, 4, 4, 3),  # 7.16 us vs cuBLAS 8.65 (1.208x), separate timing 7.50 us
    (288, 4096, None): (16, 64, 8, 2, 5),
    # idx_wk_weights_proj  N=160 K=4096
    (160, 4096, 1): (16, 128, 8, 2, 4),  # 4.96 us vs cuBLAS 5.50 (1.108x), separate timing 5.18 us
    (160, 4096, 2): (32, 128, 8, 4, 4),  # 5.12 us vs cuBLAS 7.01 (1.369x), separate timing 5.26 us
    (160, 4096, 4): (32, 128, 8, 4, 4),  # 5.19 us vs cuBLAS 6.53 (1.257x), separate timing 5.41 us
    (160, 4096, 8): (32, 128, 8, 4, 4),  # 5.19 us vs cuBLAS 6.71 (1.294x), separate timing 5.40 us
    (160, 4096, 16): (16, 128, 8, 2, 4),  # 5.49 us vs cuBLAS 6.90 (1.257x), separate timing 5.80 us
    (160, 4096, 32): (32, 64, 8, 2, 5),  # 6.07 us vs cuBLAS 9.10 (1.499x), separate timing 6.23 us
    (160, 4096, None): (32, 128, 8, 4, 4),
    # idx_kpool_compress_gate  N=128 K=4096
    (128, 4096, 1): (16, 128, 16, 2, 3),  # 5.00 us vs cuBLAS 5.42 (1.085x), separate timing 5.15 us
    (128, 4096, 2): (16, 128, 8, 4, 4),  # 4.85 us vs cuBLAS 7.04 (1.450x), separate timing 5.16 us
    (128, 4096, 4): (16, 128, 16, 4, 4),  # 4.94 us vs cuBLAS 6.30 (1.275x), separate timing 5.08 us
    (128, 4096, 8): (16, 128, 8, 4, 4),  # 4.96 us vs cuBLAS 6.39 (1.290x), separate timing 5.08 us
    (128, 4096, 16): (16, 128, 8, 4, 4),  # 5.03 us vs cuBLAS 6.62 (1.315x), separate timing 5.20 us
    (128, 4096, 32): (16, 128, 8, 4, 4),  # 5.76 us vs cuBLAS 7.11 (1.233x), separate timing 6.14 us
    (128, 4096, None): (16, 128, 8, 4, 4),
    # dflash_fc  N=4096 K=20480 (DFlash drafter input projection, replicated).
    # One schedule wins or ties at every M; M=16 is within noise of the old
    # fallback but kept equal to the any-M row below.
    (4096, 20480, 1): (128, 128, 2, 4, 4),  # 105.73 us vs cuBLAS 111.23 (1.052x), separate timing 105.86 us, fallback was 109.31
    (4096, 20480, 2): (128, 128, 2, 4, 4),  # 106.50 us vs cuBLAS 112.64 (1.058x), separate timing 106.50 us, fallback was 110.98
    (4096, 20480, 4): (128, 128, 2, 4, 4),  # 107.01 us vs cuBLAS 112.77 (1.054x), separate timing 107.01 us, fallback was 110.21
    (4096, 20480, 8): (128, 128, 2, 4, 4),  # 107.52 us vs cuBLAS 113.02 (1.051x), separate timing 107.65 us, fallback was 111.23
    (4096, 20480, 16): (128, 128, 2, 4, 4),  # 108.54 us vs cuBLAS 113.66 (1.047x), separate timing 108.54 us, fallback was 109.95
    (4096, 20480, 24): (128, 128, 2, 4, 4),  # 112.51 us vs cuBLAS 150.78 (1.340x), separate timing 112.64 us, fallback was 130.05
    (4096, 20480, 32): (128, 128, 2, 4, 4),  # 112.77 us vs cuBLAS 151.94 (1.347x), separate timing 112.64 us, fallback was 142.59
    (4096, 20480, None): (128, 128, 2, 4, 4),
    # lm_head_verify+lm_head_dflash_draft  N=38720 K=4096
    (38720, 4096, 1): (64, 256, 1, 4, 3),  # 193.66 us vs cuBLAS 213.89 (1.104x), fallback was 199.94
    (38720, 4096, 2): (64, 256, 1, 4, 3),  # 193.79 us vs cuBLAS 211.71 (1.092x), fallback was 200.06
    (38720, 4096, 4): (64, 256, 1, 4, 3),  # 195.07 us vs cuBLAS 214.27 (1.098x), fallback was 201.47
    (38720, 4096, 8): (64, 256, 1, 4, 3),  # 196.86 us vs cuBLAS 218.50 (1.110x), fallback was 203.26
    (38720, 4096, 16): (128, 128, 1, 4, 4),  # 199.68 us vs cuBLAS 223.74 (1.121x), fallback was 208.26
    (38720, 4096, 24): (64, 128, 1, 4, 3),  # 202.75 us vs cuBLAS 216.58 (1.068x), fallback was 213.38
    (38720, 4096, 32): (64, 128, 1, 4, 4),  # 210.56 us vs cuBLAS 220.16 (1.046x), fallback was 216.83
    (38720, 4096, None): (64, 256, 1, 4, 3),
    # dflash_qkv  N=1536 K=4096 (M <= 8 keeps the fallback schedule, which ties)
    (1536, 4096, 16): (32, 64, 4, 2, 5),  # 12.93 us vs cuBLAS 15.05 (1.164x), fallback was 14.19
    (1536, 4096, 24): (32, 64, 4, 4, 4),  # 14.19 us vs cuBLAS 15.38 (1.084x), fallback was 17.46
    (1536, 4096, 32): (32, 128, 4, 2, 3),  # 14.91 us vs cuBLAS 15.55 (1.043x), fallback was 19.00
    # dflash_o  N=4096 K=1024
    (4096, 1024, 1): (64, 128, 1, 4, 4),  # 7.74 us vs cuBLAS 9.09 (1.174x), fallback was 8.00
    (4096, 1024, 2): (32, 64, 1, 2, 5),  # 8.16 us vs cuBLAS 9.42 (1.155x), fallback was 8.51
    (4096, 1024, 4): (32, 64, 1, 2, 5),  # 8.22 us vs cuBLAS 9.47 (1.152x), fallback was 8.54
    (4096, 1024, 8): (32, 64, 1, 2, 5),  # 8.32 us vs cuBLAS 9.54 (1.146x), fallback was 8.72
    (4096, 1024, 16): (64, 128, 1, 4, 4),  # 8.42 us vs cuBLAS 9.73 (1.156x), fallback was 8.80
    (4096, 1024, 24): (64, 64, 1, 8, 4),  # 9.20 us vs cuBLAS 9.89 (1.075x), fallback was 10.88
    (4096, 1024, 32): (64, 128, 1, 4, 3),  # 9.25 us vs cuBLAS 10.05 (1.087x), fallback was 11.49
    (4096, 1024, None): (64, 128, 1, 4, 4),
    # kda_f_b_proj+kda_g_b_proj  N=2048 K=128
    (2048, 128, 1): (32, 64, 1, 2, 5),  # 2.50 us vs cuBLAS 3.39 (1.353x), separate timing 2.50 us
    (2048, 128, 2): (32, 128, 1, 4, 2),  # 2.55 us vs cuBLAS 3.49 (1.370x), separate timing 2.57 us
    (2048, 128, 4): (32, 128, 1, 4, 3),  # 2.55 us vs cuBLAS 3.48 (1.364x), separate timing 2.60 us
    (2048, 128, 8): (32, 128, 1, 4, 2),  # 2.61 us vs cuBLAS 3.55 (1.358x), separate timing 2.63 us
    (2048, 128, 16): (32, 128, 1, 4, 2),  # 2.66 us vs cuBLAS 3.59 (1.349x), separate timing 2.70 us
    (2048, 128, 32): (32, 128, 1, 4, 4),  # 2.85 us vs cuBLAS 3.60 (1.262x), separate timing 2.90 us
    (2048, 128, None): (32, 128, 1, 4, 3),
}

_num_sms_cache = None


def num_sms() -> int:
    global _num_sms_cache
    if _num_sms_cache is None:
        _num_sms_cache = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _num_sms_cache


def _pow2_at_least(x, lo=16):
    v = lo
    while v < x:
        v *= 2
    return v


def _heuristic(M, N, K, BLOCK_M):
    """Fallback for any (N, K) the sweep did not measure.

    Same principle the measured table encodes: pick the *largest* BLOCK_N that
    still yields ~_CTA_PER_SM_TARGET CTAs per SM from the N dimension alone, so
    SPLIT_K stays 1 and no reduction kernel runs; only split K when N cannot
    fill the machine by itself.
    """
    BLOCK_K = 128 if K >= 1024 else _pow2_at_least(min(K, 64), 16)
    want = max(1, int(_CTA_PER_SM_TARGET * num_sms()))

    BLOCK_N = 16
    for cand in (128, 64, 32):
        if triton.cdiv(N, cand) >= want:
            BLOCK_N = cand
            break

    tiles_n = triton.cdiv(N, BLOCK_N)
    max_split = max(1, min(triton.cdiv(K, BLOCK_K), K // _MIN_K_PER_SPLIT))
    SPLIT_K = max(1, min(max_split, -(-want // tiles_n)))

    num_warps = 4 if BLOCK_N >= 64 else 2
    num_stages = 4 if BLOCK_N * BLOCK_K <= 8192 else 3
    return BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages


def _select_config(M, N, K):
    """-> (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)."""
    # tl.dot needs M >= 16 on Ampere (m16n8k16), so M is padded up to 16 and
    # masked.  The cap at 64 is a shared-memory bound, NOT a bound on M: the
    # kernel tiles M over grid dim 2, so any M is covered.
    BLOCK_M = min(64, _pow2_at_least(M, 16))

    cfg = _CONFIG_OVERRIDES.get((N, K, M))
    if cfg is None:
        cfg = _CONFIG_OVERRIDES.get((N, K, None))
    if cfg is None:
        cfg = _heuristic(M, N, K, BLOCK_M)
    BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = cfg

    # A pinned config was measured at a small BLOCK_M; a larger M grows the
    # pipelined operand tiles, so trim the pipeline depth instead of asking for
    # more shared memory than an SM has.
    while num_stages > 2 and \
            num_stages * (BLOCK_N + BLOCK_M) * BLOCK_K * 2 > _MAX_SMEM:
        num_stages -= 1
    return BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

@triton.jit
def _thin_gemm_kernel(
    X, W, Y, P, LOCK,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    TILED_M: tl.constexpr,
):
    """One CTA per (M tile, N tile, K split).  Reads its K-blocks of W once.

    When SPLIT_K > 1 the FP32 partials are reduced by whichever CTA of the
    column arrives last, not by a second launch: the second launch cost 2-3.5 us
    per call (a bare in-graph launch is ~0.9 us on this part, and the rest is
    the drain plus one full HBM latency round with nothing left to overlap it).

    The reduction stays bitwise deterministic because the *order* is fixed -
    always partial 0, then 1, ... SPLIT_K-1 - even though *which* CTA performs
    it varies run to run.  That is the whole reason this is not an atomic_add
    into Y, which would not be reproducible.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # BLOCK_M is capped by shared memory, so a caller with more rows than one
    # tile gets several M tiles over grid dim 2.  Masking alone is NOT enough:
    # with a pinned BLOCK_M and no pid_m, rows past BLOCK_M were never written.
    # Every M the model uses (1..64) is a single tile, and there the row offset
    # must stay a compile-time zero -- a runtime `pid_m * BLOCK_M` in the
    # address arithmetic costs 0.5-4.5 % on every shape (measured), so the
    # one-tile case is specialized rather than paying for generality.
    if TILED_M:
        pid_m = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        pid_m = 0
        offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    # Clamping the row/column index keeps every address inside the tensor, so
    # the masked-off lanes of a ragged tile cannot fault even before the mask
    # is applied.
    x_m = tl.where(m_mask, offs_m, 0)
    w_n = tl.where(n_mask, offs_n, 0)

    x_ptrs = X + x_m[:, None] * stride_xm + \
        (pid_k * BLOCK_K + offs_k)[None, :] * stride_xk
    w_ptrs = W + w_n[:, None] * stride_wn + \
        (pid_k * BLOCK_K + offs_k)[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step = BLOCK_K * SPLIT_K
    n_iter = tl.cdiv(K, step)
    for i in range(n_iter):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        else:
            k_now = i * step + pid_k * BLOCK_K + offs_k
            k_mask = k_now < K
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        x_ptrs += step * stride_xk
        w_ptrs += step * stride_wk

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    if SPLIT_K == 1:
        if EVEN_N:
            tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=m_mask[:, None])
        else:
            tl.store(y_ptrs, acc.to(Y.dtype.element_ty),
                     mask=m_mask[:, None] & n_mask[None, :])
    else:
        p_mask = m_mask[:, None] & n_mask[None, :]
        p_off = offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        # ".cg" keeps the partial out of the non-coherent per-SM L1 so the
        # reducing CTA, which runs on a different SM, can see it in L2.
        tl.store(P + pid_k * stride_pk + p_off, acc, mask=p_mask,
                 cache_modifier=".cg")
        # Release the store, then count arrivals for this N column.  No CTA
        # ever spins, so this cannot deadlock however the grid is scheduled.
        # One counter per (M tile, N tile) output block; grid dim 0 is the
        # number of N tiles, so this indexes the column within this M tile.
        if TILED_M:
            lock = LOCK + pid_m * tl.num_programs(0) + pid_n
        else:
            lock = LOCK + pid_n
        arrived = tl.atomic_add(lock, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                tot += tl.load(P + k * stride_pk + p_off, mask=p_mask,
                               other=0.0, cache_modifier=".cv")
            tl.store(y_ptrs, tot.to(Y.dtype.element_ty), mask=p_mask)
            # Leave the counter at zero for the next launch.  Launches on one
            # stream are serialized, so no later CTA can race this reset.
            tl.atomic_xchg(lock, 0, sem="release", scope="gpu")


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
# Buffers are cached forever and never shrunk.  That matters for CUDA-graph
# capture: a graph records the pointer it was captured with, so the buffer must
# outlive the graph.  Call warmup() before capture to force every size the
# model needs to exist first.

_WORKSPACE: dict = {}


def _partials(split_k, M, N, device):
    key = (device.index, split_k * M * N)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.empty(split_k * M * N, dtype=torch.float32, device=device)
        _WORKSPACE[key] = buf
    return buf.view(split_k, M, N)


def _locks(tiles_n, device):
    """Zeroed arrival counters, one per N tile.

    Cached like every other workspace buffer so the pointer a captured graph
    recorded stays valid.  The kernel always leaves the counters at zero, so a
    buffer can be reused by any shape and across replays.
    """
    key = (device.index, "lock", tiles_n)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.zeros(tiles_n, dtype=torch.int32, device=device)
        _WORKSPACE[key] = buf
    return buf


def _dummy_fp32(device):
    key = (device.index, "dummy")
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _WORKSPACE[key] = buf
    return buf


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def thin_gemm(x: torch.Tensor, w: torch.Tensor,
              out: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ w.T for BF16 x [M, K] and BF16 w [N, K], FP32 accumulate."""
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, \
        "thin_gemm is BF16-only; no quantized or FP8 path exists by design"
    assert x.ndim == 2 and w.ndim == 2 and x.shape[1] == w.shape[1]
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M == 0 or N == 0:
        return out

    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = _select_config(M, N, K)
    even_k = (K % (BLOCK_K * SPLIT_K)) == 0
    even_n = (N % BLOCK_N) == 0

    tiles_n = triton.cdiv(N, BLOCK_N)
    # BLOCK_M is bounded by shared memory, so a caller with M > BLOCK_M needs
    # more than one M tile.  Every benchmarked M (1..32) gives tiles_m == 1.
    tiles_m = triton.cdiv(M, BLOCK_M)
    if SPLIT_K == 1:
        # Unused by the kernel (the SPLIT_K branch is constexpr-folded away),
        # but the arguments still have to be real pointers.
        p = _dummy_fp32(x.device)
        sp = (0, 0, 0)
        lock = _locks(1, x.device)
    else:
        p = _partials(SPLIT_K, M, N, x.device)
        sp = (p.stride(0), p.stride(1), p.stride(2))
        lock = _locks(tiles_n * tiles_m, x.device)

    _thin_gemm_kernel[(tiles_n, SPLIT_K, tiles_m)](
        x, w, out, p, lock,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        sp[0], sp[1], sp[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
        EVEN_K=even_k, EVEN_N=even_n, TILED_M=(tiles_m > 1),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def warmup(shapes, ms, device=None) -> None:
    """JIT-compile and allocate for every (M, N, K) that will be replayed.

    Triton compiles on first launch and the workspace is allocated lazily, so
    neither may happen inside a `torch.cuda.graph` capture.  vLLM calls this
    once during its capture warm-up; see INTEGRATION.md.
    """
    device = device or torch.device("cuda", torch.cuda.current_device())
    for N, K in shapes:
        w = torch.empty((N, K), dtype=torch.bfloat16, device=device)
        for M in ms:
            x = torch.empty((M, K), dtype=torch.bfloat16, device=device)
            thin_gemm(x, w)
    torch.cuda.synchronize(device)


# Measured, not assumed.  INTEGRATION.md gates the vLLM side on
# `x.shape[0] <= THIN_GEMM_MAX_M`, and that build captures graphs at
# M in [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], so the only question the
# threshold has to answer is which of those sizes thin_gemm still wins.
#
# Count-weighted step time over the whole shape table, candidate vs cuBLAS
# timed in the same process with bench.py's rotation and graph replay:
#
#     M     candidate   cuBLAS    ratio   shapes losing
#     32      4.061 ms  4.699 ms  1.157x    0/16
#     40      5.722 ms  4.573 ms  0.799x   10/16
#     48      6.065 ms  4.635 ms  0.764x   11/16
#     56      6.258 ms  4.701 ms  0.751x   11/16
#     64      6.719 ms  4.817 ms  0.717x   11/16
#
# 32 is therefore the right answer, and it is not merely an artefact of the
# schedule table stopping there.  Re-fitting the configs at M = 40 and 64 from
# a ~250-point grid recovers most of the gap but does not cross parity where it
# matters: at M=40 the best config found is 0.89-1.00x on the four dominant
# shapes, and at M=64 `kda_in_proj_qkvbfg_a` -- 43 % of all weight traffic --
# still only reaches 0.884x.  M=40 is the worst case of all because BLOCK_M is
# capped at 64, so 24 of every 64 accumulator rows are padding the tensor core
# computes and discards; cuBLAS switches to a compute-bound tile there and this
# kernel, which is built for the weight-bandwidth-bound regime, has nothing
# left to trade.
#
# Raising this to the next capture size would cost ~43 % of the step. Do not
# raise it without re-running the token-bound measurements.
_DISPATCH_MAX_M = 32


def dispatch_threshold() -> int:
    """Largest M for which thin_gemm should be preferred over cuBLAS.

    Measured (see the table above); `INTEGRATION.md` explains how the vLLM-side
    dispatch uses it.  Overridable for performance measurements.
    """
    return int(os.environ.get("THIN_GEMM_MAX_M", str(_DISPATCH_MAX_M)))
