# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored kernel: keep in sync with the standalone kernel source.
# Revalidate changes with the standalone correctness and performance checks.
# Keep changes synchronized: the standalone repo owns the
# correctness gate (validate.py) and the benchmark (bench.py) that qualify this
# file, so a change made here and not there is a change that nothing measures.
# To update, re-copy the file and re-run the standalone gate.
"""Thin-M BF16 GEMM for sm_80, 74-SM schedule (split-K, FP32 partials).

The schedule is tuned on the 74-SM part: measured schedule rows (tagged
"74 SMs" in the table where the 74-SM re-sweep changed them), a per-N-tile
rotation of the K start for wide tiles (``_rotate_k``; summation order
changes, deterministically), and a split-K arrival that only releases, with
the last arriver acquiring through the counter reset. Every output is bitwise
reproducible run to run.

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

import contextlib

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
#
# Rows tagged "74 SMs" were re-swept on the 74-SM part (bench.py rotation, each
# config interleaved with the previous schedule) and kept only when bench.py
# itself confirmed them; the ratio quoted is over the previous schedule.
_CONFIG_OVERRIDES: dict = {
    # kda_in_proj_qkvbfg_a  N=6416 K=4096
    (6416, 4096, 1): (32, 128, 1, 2, 3),  # 34.82 us vs cuBLAS 40.03 (1.150x), separate timing 34.82 us
    (6416, 4096, 2): (32, 128, 1, 2, 3),  # 35.19 us vs cuBLAS 38.35 (1.090x), separate timing 35.19 us
    (6416, 4096, 4): (128, 128, 1, 4, 3),  # 74 SMs + ROTATE_K: 1.022-1.025x over (32, 128, 1, 2, 3) (3 processes)
    (6416, 4096, 8): (32, 128, 1, 2, 3),  # 35.75 us vs cuBLAS 38.82 (1.086x), separate timing 35.75 us
    (6416, 4096, 16): (128, 128, 1, 4, 3),  # 74 SMs + ROTATE_K: 1.05-1.12x over (32, 128, 1, 2, 3); placement-robust
    (6416, 4096, 24): (128, 128, 1, 4, 3),  # 37.33 us vs cuBLAS 44.78 (1.200x), fallback was 38.73
    (6416, 4096, 32): (128, 128, 1, 4, 3),  # 37.42 us vs cuBLAS 45.71 (1.221x), separate timing 37.42 us
    (6416, 4096, None): (32, 128, 1, 2, 3),
    # dense_mlp_gate_up  N=6144 K=4096
    (6144, 4096, 1): (32, 128, 1, 2, 3),  # 33.98 us vs cuBLAS 37.52 (1.104x), separate timing 33.98 us
    (6144, 4096, 2): (32, 128, 1, 2, 3),  # 34.26 us vs cuBLAS 37.33 (1.090x), separate timing 34.26 us
    (6144, 4096, 4): (128, 128, 1, 4, 3),  # 74 SMs + ROTATE_K: 1.035-1.038x over (32, 128, 1, 2, 3) (3 processes)
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
    (4096, 4096, 16): (32, 128, 1, 4, 4),  # 24.13 us vs cuBLAS 26.05 (1.080x); (64,128,1,8,5) won only cold, 0.989x warm
    (4096, 4096, 24): (64, 128, 1, 8, 5),  # 24.45 us vs cuBLAS 25.73 (1.052x), fallback was 25.79
    (4096, 4096, 32): (64, 128, 1, 8, 5),  # 24.58 us vs cuBLAS 25.98 (1.057x), separate timing 27.97 us
    (4096, 4096, None): (32, 128, 1, 4, 4),
    # dense_mlp_down  N=4096 K=3072
    (4096, 3072, 1): (32, 128, 1, 2, 4),  # 18.85 us vs cuBLAS 20.20 (1.072x), separate timing 18.85 us
    (4096, 3072, 2): (64, 256, 1, 4, 3),  # 18.66 us vs cuBLAS 20.57 (1.102x), separate timing 19.55 us
    (4096, 3072, 4): (64, 256, 1, 4, 3),  # 18.71 us vs cuBLAS 20.53 (1.097x), separate timing 19.60 us
    (4096, 3072, 8): (32, 128, 1, 4, 4),  # 19.08 us vs cuBLAS 20.81 (1.090x), separate timing 19.97 us
    (4096, 3072, 16): (64, 128, 1, 8, 4),  # 74 SMs: 1.01-1.08x over (32, 128, 1, 4, 4) (3 processes)
    (4096, 3072, 24): (64, 128, 1, 8, 4),  # 20.15 us vs cuBLAS 20.99 (1.042x), fallback was 21.18
    (4096, 3072, 32): (64, 128, 1, 8, 4),  # 20.48 us vs cuBLAS 21.22 (1.036x), separate timing 20.53 us
    (4096, 3072, None): (32, 128, 1, 4, 4),
    # mla_fused_qkv_a_proj  N=2048 K=4096
    (2048, 4096, 1): (32, 256, 1, 4, 4),  # 13.73 us vs cuBLAS 16.67 (1.214x), separate timing 14.69 us
    (2048, 4096, 2): (32, 128, 1, 4, 5),  # 13.70 us vs cuBLAS 16.54 (1.208x), separate timing 14.53 us
    (2048, 4096, 4): (32, 128, 1, 4, 5),  # 13.79 us vs cuBLAS 17.86 (1.295x), separate timing 14.82 us
    (2048, 4096, 8): (32, 128, 1, 4, 5),  # 13.89 us vs cuBLAS 17.02 (1.226x), separate timing 14.91 us
    (2048, 4096, 16): (32, 256, 1, 4, 4),  # 13.95 us vs cuBLAS 17.38 (1.245x), separate timing 15.36 us
    (2048, 4096, 24): (64, 256, 2, 4, 3),  # 74 SMs, cheaper split-K tail: 1.075x over (32, 128, 1, 4, 5)
    (2048, 4096, 32): (32, 128, 1, 4, 5),  # 15.74 us vs cuBLAS 18.02 (1.144x); split-K (64,256,2,4,3) won only cold, 0.96x warm
    (2048, 4096, None): (32, 128, 1, 4, 5),
    # kda_o_proj  N=4096 K=2048
    (4096, 2048, 1): (64, 128, 1, 4, 4),  # 13.15 us vs cuBLAS 14.75 (1.122x), separate timing 13.15 us
    (4096, 2048, 2): (16, 128, 1, 2, 4),  # 13.70 us vs cuBLAS 15.04 (1.098x), separate timing 13.70 us
    (4096, 2048, 4): (64, 128, 1, 8, 5),  # 74 SMs: 1.091x over (16, 128, 1, 2, 4)
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
    (1024, 4096, 16): (64, 128, 4, 2, 4),  # 74 SMs: 1.062x over (32, 64, 4, 2, 5)
    (1024, 4096, 24): (64, 256, 4, 4, 3),  # 10.51 us vs cuBLAS 11.38 (1.082x), fallback was 11.42
    (1024, 4096, 32): (64, 64, 4, 8, 4),  # 74 SMs: 1.038x over (64, 64, 4, 4, 5)
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
    (4096, 512, 24): (64, 64, 1, 2, 4),  # 74 SMs: 1.046x over (64, 64, 1, 8, 5)
    (4096, 512, 32): (64, 64, 1, 2, 4),  # 74 SMs: 1.105x over (64, 64, 1, 8, 5)
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
    # M=4 and M=16 have their own 74-SM schedules; every other M runs the
    # any-M schedule below.
    (4096, 20480, 1): (128, 128, 2, 4, 4),  # 105.73 us vs cuBLAS 111.23 (1.052x), separate timing 105.86 us, fallback was 109.31
    (4096, 20480, 2): (128, 128, 2, 4, 4),  # 106.50 us vs cuBLAS 112.64 (1.058x), separate timing 106.50 us, fallback was 110.98
    (4096, 20480, 4): (32, 128, 1, 4, 5),  # 74 SMs: 1.290x over (128, 128, 2, 4, 4)
    (4096, 20480, 8): (128, 128, 2, 4, 4),  # 107.52 us vs cuBLAS 113.02 (1.051x), separate timing 107.65 us, fallback was 111.23
    (4096, 20480, 16): (64, 128, 1, 8, 4),  # 74 SMs: 1.278x over (128, 128, 2, 4, 4)
    (4096, 20480, 24): (128, 128, 2, 4, 4),  # 112.51 us vs cuBLAS 150.78 (1.340x), separate timing 112.64 us, fallback was 130.05
    (4096, 20480, 32): (128, 128, 2, 4, 4),  # 112.77 us vs cuBLAS 151.94 (1.347x), separate timing 112.64 us, fallback was 142.59
    (4096, 20480, None): (128, 128, 2, 4, 4),
    # lm_head_verify+lm_head_dflash_draft  N=38720 K=4096
    (38720, 4096, 1): (256, 64, 1, 4, 2),  # 74 SMs (M 3-12): 1.11-1.13x over (64, 256, 1, 4, 3)
    (38720, 4096, 2): (256, 64, 1, 4, 2),
    (38720, 4096, 3): (256, 64, 1, 4, 2),
    (38720, 4096, 4): (256, 64, 1, 4, 2),
    (38720, 4096, 8): (256, 64, 1, 4, 2),
    (38720, 4096, 12): (256, 64, 1, 4, 2),
    (38720, 4096, 16): (256, 64, 1, 4, 2),  # 74 SMs: 1.058x over (128, 128, 1, 4, 4)
    (38720, 4096, 18): (64, 128, 1, 8, 4),  # 74 SMs: 1.080x over (64, 256, 1, 4, 3)
    (38720, 4096, 24): (64, 128, 1, 8, 3),  # 74 SMs: 1.128x over (64, 128, 1, 4, 3)
    (38720, 4096, 32): (64, 128, 1, 8, 3),  # 74 SMs: 1.100x over (64, 128, 1, 4, 4)
    (38720, 4096, None): (64, 256, 1, 4, 3),
    # dflash_qkv  N=1536 K=4096
    (1536, 4096, 4): (64, 128, 3, 8, 4),  # 74 SMs: 1.060x over the heuristic (16, 128, 2, 2, 4)
    (1536, 4096, 16): (64, 128, 3, 8, 4),  # 74 SMs: 1.038x over (32, 64, 4, 2, 5)
    (1536, 4096, 24): (64, 128, 3, 8, 4),  # 74 SMs: 1.036x over (32, 64, 4, 4, 4)
    (1536, 4096, 32): (64, 128, 3, 8, 3),  # 74 SMs: 1.056x over (32, 128, 4, 2, 3)
    # dflash_o  N=4096 K=1024
    (4096, 1024, 1): (64, 128, 1, 4, 4),  # 7.74 us vs cuBLAS 9.09 (1.174x), fallback was 8.00
    (4096, 1024, 2): (32, 64, 1, 2, 5),  # 8.16 us vs cuBLAS 9.42 (1.155x), fallback was 8.51
    (4096, 1024, 4): (32, 64, 1, 2, 5),  # 8.22 us vs cuBLAS 9.47 (1.152x), fallback was 8.54
    (4096, 1024, 8): (32, 64, 1, 2, 5),  # 8.32 us vs cuBLAS 9.54 (1.146x), fallback was 8.72
    (4096, 1024, 16): (64, 128, 1, 4, 4),  # 8.42 us vs cuBLAS 9.73 (1.156x), fallback was 8.80
    (4096, 1024, 24): (64, 128, 1, 8, 4),  # 74 SMs: 1.088x over (64, 64, 1, 8, 4)
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
    #
    # Pipeline parallel (TP=1): one card runs whole layers, so the column- and
    # row-parallel projections are at full width. Only the (N, K) that exist
    # under PP alone are listed; an (N, K) shared with TP=4 keeps its rows
    # above. Same sweep and acceptance rule (>= 2.5 % over the fallback
    # schedule); an M not listed keeps the heuristic.
    # PP4 lm_head_verify+lm_head_dflash_draft  N=154880 K=4096
    (154880, 4096, 1): (128, 128, 1, 4, 4),  # 747.39 us vs cuBLAS 948.35 (1.269x), was 887.94
    (154880, 4096, 4): (64, 64, 1, 8, 2),  # 74 SMs: 1.186x over the heuristic (128, 128, 1, 4, 3)
    (154880, 4096, 8): (64, 64, 1, 8, 2),  # 74 SMs: 1.230x over the heuristic (128, 128, 1, 4, 3)
    (154880, 4096, 32): (128, 256, 1, 4, 2),  # 793.34 us vs cuBLAS 1051.52 (1.325x), was 1491.46
    # PP4 kda_in_proj_qkvbfg_a  N=24896 K=4096
    (24896, 4096, 1): (128, 128, 2, 4, 3),  # 129.15 us vs cuBLAS 144.77 (1.121x), was 132.48
    (24896, 4096, 4): (64, 64, 1, 8, 2),  # 74 SMs: 1.117x over the heuristic (128, 128, 1, 4, 3)
    (24896, 4096, 8): (128, 128, 1, 4, 4),  # 130.43 us vs cuBLAS 130.69 (1.002x), was 134.66
    (24896, 4096, 16): (128, 128, 1, 4, 4),  # 131.20 us vs cuBLAS 132.99 (1.014x), was 136.96
    (24896, 4096, 24): (128, 128, 1, 4, 2),  # 132.99 us vs cuBLAS 148.48 (1.116x), was 141.70
    (24896, 4096, 32): (128, 128, 1, 4, 2),  # 133.89 us vs cuBLAS 149.76 (1.119x), was 144.51
    # PP4 dense_mlp_gate_up+dflash_gate_up  N=24576 K=4096
    (24576, 4096, 1): (128, 128, 1, 4, 4),  # 126.85 us vs cuBLAS 141.44 (1.115x), was 131.07
    (24576, 4096, 4): (128, 128, 1, 4, 4),  # 128.00 us vs cuBLAS 128.38 (1.003x), was 132.22
    (24576, 4096, 8): (128, 128, 1, 4, 4),  # 74 SMs + ROTATE_K: 1.04-1.08x over (64, 64, 1, 8, 2) (3 processes)
    (24576, 4096, 16): (128, 128, 1, 4, 4),  # 129.41 us vs cuBLAS 131.20 (1.014x), was 136.58
    (24576, 4096, 24): (128, 128, 1, 4, 2),  # 131.33 us vs cuBLAS 147.07 (1.120x), was 140.16
    (24576, 4096, 32): (128, 128, 1, 4, 2),  # 131.84 us vs cuBLAS 148.61 (1.127x), was 142.59
    # PP4 mla_o_proj  N=4096 K=16384
    (4096, 16384, 4): (32, 128, 1, 4, 4),  # 74 SMs: 1.040x over the heuristic (16, 128, 1, 2, 4)
    (4096, 16384, 8): (64, 128, 1, 8, 5),  # 74 SMs: 1.060x over the heuristic (16, 128, 1, 2, 4)
    (4096, 16384, 24): (32, 128, 1, 4, 4),  # 88.32 us vs cuBLAS 118.78 (1.345x), was 105.09
    (4096, 16384, 32): (128, 64, 2, 4, 4),  # 91.01 us vs cuBLAS 119.17 (1.309x), was 114.69
    # PP4 dense_mlp_down+dflash_down  N=4096 K=12288
    (4096, 12288, 8): (64, 128, 1, 8, 5),  # 74 SMs: 1.138x over the heuristic (16, 128, 1, 2, 4)
    (4096, 12288, 16): (64, 128, 1, 8, 5),  # 74 SMs: 1.415x over the heuristic (16, 128, 1, 2, 4)
    (4096, 12288, 24): (128, 128, 2, 4, 4),  # 70.14 us vs cuBLAS 96.90 (1.381x), was 80.00
    (4096, 12288, 32): (128, 128, 2, 4, 3),  # 70.40 us vs cuBLAS 98.05 (1.393x), was 87.55
    # PP4 kda_o_proj  N=4096 K=8192
    (4096, 8192, 4): (64, 128, 1, 8, 4),  # 74 SMs: 1.133x over the heuristic (16, 128, 1, 2, 4)
    (4096, 8192, 8): (64, 128, 1, 8, 5),  # 74 SMs: 1.209x over the heuristic (16, 128, 1, 2, 4)
    # PP4 dflash_ctx_kv  N=10240 K=4096
    (10240, 4096, 1): (32, 128, 1, 2, 2),  # 54.91 us vs cuBLAS 64.90 (1.182x), was 62.72
    (10240, 4096, 2): (32, 128, 1, 2, 2),  # 55.30 us vs cuBLAS 65.54 (1.185x), was 64.90
    (10240, 4096, 4): (32, 128, 1, 2, 2),  # 55.81 us vs cuBLAS 66.43 (1.190x), was 65.41
    (10240, 4096, 8): (32, 128, 1, 2, 2),  # 55.81 us vs cuBLAS 67.07 (1.202x), was 65.79
    (10240, 4096, 16): (32, 128, 1, 2, 3),  # 56.58 us vs cuBLAS 69.12 (1.222x), was 66.94
    (10240, 4096, 24): (64, 128, 1, 4, 3),  # 59.78 us vs cuBLAS 76.54 (1.281x), was 76.80
    (10240, 4096, 32): (64, 128, 1, 4, 3),  # 60.42 us vs cuBLAS 75.90 (1.256x), was 75.39
    # PP4 mla_q_b_proj  N=16384 K=1536
    (16384, 1536, 8): (128, 128, 1, 4, 4),  # 35.10 us vs cuBLAS 39.66 (1.130x), was 36.03
    # PP4 mla_kv_b_proj  N=32768 K=512
    (32768, 512, 4): (64, 256, 1, 4, 3),  # 24.90 us vs cuBLAS 29.06 (1.167x), was 25.54
    (32768, 512, 8): (64, 256, 1, 4, 3),  # 25.22 us vs cuBLAS 29.57 (1.173x), was 26.05
    (32768, 512, 16): (128, 128, 1, 4, 3),  # 26.05 us vs cuBLAS 30.21 (1.160x), was 27.07
    (32768, 512, 24): (128, 128, 1, 4, 3),  # 27.58 us vs cuBLAS 31.42 (1.139x), was 28.99
    (32768, 512, 32): (128, 128, 1, 4, 3),  # 27.52 us vs cuBLAS 30.85 (1.121x), was 29.50
    # PP4 kda_f_b_proj+kda_g_b_proj  N=8192 K=128
    (8192, 128, 1): (64, 32, 1, 4, 4),  # 3.84 us vs cuBLAS 5.08 (1.320x), was 3.95
    (8192, 128, 24): (128, 32, 1, 4, 4),  # 4.29 us vs cuBLAS 5.43 (1.265x), was 4.65
    (8192, 128, 32): (128, 32, 1, 4, 4),  # 4.42 us vs cuBLAS 5.48 (1.240x), was 4.74
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
    ROTATE_K: tl.constexpr,
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
    if ROTATE_K:
        # Every CTA streams the same x.  In lock step they would all read the
        # same x K-block from the same L2 lines at once; starting each N tile
        # at a different K-block spreads those reads (summation order changes,
        # deterministically: the start depends only on the tile).  Only taken
        # when EVEN_K, so every K-block is whole.
        shift = (pid_n * 3) % n_iter
        for i in range(n_iter):
            j = i + shift
            j = tl.where(j >= n_iter, j - n_iter, j)
            x = tl.load(x_ptrs + j * step * stride_xk, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs + j * step * stride_wk, mask=n_mask[:, None], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    else:
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
        # The arrival only needs release (it publishes this CTA's partial).
        # Only the last arriver needs acquire, and it gets it from the counter
        # reset: an acquire RMW on the same counter, after the others' release
        # RMWs in its modification order, synchronizes with all of them.  The
        # non-last CTAs thus skip the acquire fence (measured +3-10 % on the
        # split-K shapes vs one acq_rel arrival).  Launches on one stream are
        # serialized, so resetting before the reads cannot race a later CTA.
        arrived = tl.atomic_add(lock, 1, sem="release", scope="gpu")
        if arrived == SPLIT_K - 1:
            tl.atomic_xchg(lock, 0, sem="acquire", scope="gpu")
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                tot += tl.load(P + k * stride_pk + p_off, mask=p_mask,
                               other=0.0, cache_modifier=".cv")
            tl.store(y_ptrs, tot.to(Y.dtype.element_ty), mask=p_mask)


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
# Buffers are cached forever and never shrunk.  That matters for CUDA-graph
# capture: a graph records the pointer it was captured with, so the buffer must
# outlive the graph.  Call warmup() before capture to force every size the
# model needs to exist first.

_WORKSPACE: dict = {}

# Workspace lanes.  The shared partials and arrival counters are safe because
# launches on one stream are serialized.  A caller that runs thin GEMMs on a
# second stream while the first one is busy selects another lane and gets
# buffers of its own.  Lane 0, the default, keeps the keys every other caller
# has always used.  A plain module-level slot (not a ContextVar) so that the
# lookup stays traceable inside torch.compile'd regions; a compiled region
# keeps the buffers of the lane it was traced in.
_LANE = [0]


@contextlib.contextmanager
def workspace_lane(lane: int):
    """Run the enclosed thin GEMMs on workspace lane ``lane`` (>= 0)."""
    if lane < 0:
        raise ValueError(f"workspace lane must be non-negative, got {lane}")
    prev = _LANE[0]
    _LANE[0] = lane
    try:
        yield
    finally:
        _LANE[0] = prev


def _lane_key(*key):
    lane = _LANE[0]
    return key if lane == 0 else ("lane", lane, *key)


def _partials(split_k, M, N, device):
    key = _lane_key(device.index, split_k * M * N)
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
    key = _lane_key(device.index, "lock", tiles_n)
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


def _rotate_k(even_k: bool, split_k: int, block_m: int, block_n: int) -> bool:
    """Whether a launch rotates its K start per N tile (``ROTATE_K``).

    Shared with ``vllm_sm80.models.glm5next.ops.idx_glue`` so its merged-GEMM
    columns stay bitwise what ``thin_gemm`` stores for the same config.
    """
    return even_k and split_k == 1 and (block_m >= 32 or block_n >= 128)

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
        # Rotation measured on the 74-SM part: +2-5 % at BLOCK_M >= 32 (x tile
        # as large as a W tile row block) and on BLOCK_N >= 128 tiles at any M
        # (it also evens out the per-launch CTA finish spread); -0.5 to -3 %
        # on narrow BLOCK_N < 128 tiles at BLOCK_M = 16 and on split-K shapes.
        ROTATE_K=_rotate_k(even_k, SPLIT_K, BLOCK_M, BLOCK_N),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def warmup(shapes, ms, device=None, groups=()) -> None:
    """JIT-compile and allocate for every (M, N, K) that will be replayed.

    Triton compiles on first launch and the workspace is allocated lazily, so
    neither may happen inside a `torch.cuda.graph` capture.  vLLM calls this
    once before capture (``vllm_sm80/kernels/thin_gemm/warmup.py``).
    ``groups``: ((N_1, ..., N_g), K) concat groups served by thin_gemm_group.
    """
    device = device or torch.device("cuda", torch.cuda.current_device())
    for N, K in shapes:
        w = torch.empty((N, K), dtype=torch.bfloat16, device=device)
        for M in ms:
            x = torch.empty((M, K), dtype=torch.bfloat16, device=device)
            thin_gemm(x, w)
    for Ns, K in groups:
        ws = [torch.empty((N, K), dtype=torch.bfloat16, device=device) for N in Ns]
        for M in ms:
            x = torch.empty((M, K), dtype=torch.bfloat16, device=device)
            thin_gemm_group(x, ws)
            thin_gemm_group_split(x, ws)
    torch.cuda.synchronize(device)


# --------------------------------------------------------------------------
# Concat groups: one launch for projections that read the same x
# --------------------------------------------------------------------------
# Column n of the output reads only x and the weight row n, and its value is
# fixed by its summation order: the split-K partition (BLOCK_K, SPLIT_K), the
# rotation start (`_rotate_k` and the N tile index) and the fixed partial
# order.  A group launch runs every member's CTAs with the member's own
# `_select_config` schedule, its member-local tile numbering and its own
# workspace, so each member's output is bitwise its separate `thin_gemm` call.
# BLOCK_N only decides which CTA computes a column (and, through the tile
# index, the rotation start, which stays member-local); num_warps and
# num_stages are per launch and do not change the order of any column.
# `_gemm_tile` is `_thin_gemm_kernel`'s body with the tile coordinates and the
# arrival counter passed in; the solo kernel above is left as it is.


@triton.jit
def _gemm_tile(
    X, W, Y, P, LOCK,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_pk, stride_pm, stride_pn,
    pid_n, pid_k, pid_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    TILED_M: tl.constexpr,
    ROTATE_K: tl.constexpr,
):
    """One (M tile, N tile, K split) output block.  Reads its K-blocks of W once.

    When SPLIT_K > 1 the FP32 partials are reduced by whichever CTA of the
    column arrives last, not by a second launch: the second launch cost 2-3.5 us
    per call (a bare in-graph launch is ~0.9 us on this part, and the rest is
    the drain plus one full HBM latency round with nothing left to overlap it).

    The reduction stays bitwise deterministic because the *order* is fixed -
    always partial 0, then 1, ... SPLIT_K-1 - even though *which* CTA performs
    it varies run to run.  That is the whole reason this is not an atomic_add
    into Y, which would not be reproducible.  ``LOCK`` is this output block's
    arrival counter (unused when SPLIT_K == 1).
    """
    # BLOCK_M is capped by shared memory, so a caller with more rows than one
    # tile gets several M tiles.  Masking alone is NOT enough: with a pinned
    # BLOCK_M and no pid_m, rows past BLOCK_M were never written.
    # Every M the model uses (1..64) is a single tile, and there the row offset
    # must stay a compile-time zero -- a runtime `pid_m * BLOCK_M` in the
    # address arithmetic costs 0.5-4.5 % on every shape (measured), so the
    # one-tile case is specialized rather than paying for generality.
    if TILED_M:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
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
    if ROTATE_K:
        # Every CTA streams the same x.  In lock step they would all read the
        # same x K-block from the same L2 lines at once; starting each N tile
        # at a different K-block spreads those reads (summation order changes,
        # deterministically: the start depends only on the tile).  Only taken
        # when EVEN_K, so every K-block is whole.
        shift = (pid_n * 3) % n_iter
        for i in range(n_iter):
            j = i + shift
            j = tl.where(j >= n_iter, j - n_iter, j)
            x = tl.load(x_ptrs + j * step * stride_xk, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs + j * step * stride_wk, mask=n_mask[:, None], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    else:
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
        # Release the store, then count arrivals for this output block.  No CTA
        # ever spins, so this cannot deadlock however the grid is scheduled.
        # The arrival only needs release (it publishes this CTA's partial).
        # Only the last arriver needs acquire, and it gets it from the counter
        # reset: an acquire RMW on the same counter, after the others' release
        # RMWs in its modification order, synchronizes with all of them.  The
        # non-last CTAs thus skip the acquire fence (measured +3-10 % on the
        # split-K shapes vs one acq_rel arrival).  Launches on one stream are
        # serialized, so resetting before the reads cannot race a later CTA.
        arrived = tl.atomic_add(LOCK, 1, sem="release", scope="gpu")
        if arrived == SPLIT_K - 1:
            tl.atomic_xchg(LOCK, 0, sem="acquire", scope="gpu")
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                tot += tl.load(P + k * stride_pk + p_off, mask=p_mask,
                               other=0.0, cache_modifier=".cv")
            tl.store(y_ptrs, tot.to(Y.dtype.element_ty), mask=p_mask)


@triton.jit
def _group_member(
    X, W, Y, P, LOCK,
    M, N, K, local, pid_m, tiles_n,
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
    ROTATE_K: tl.constexpr,
):
    # Same CTA -> (pid_n, pid_k) numbering as the member's own thin_gemm grid
    # (dim 0 = N tile, fastest), so the member-local tile index drives the
    # rotation shift exactly as in a separate call.
    pid_n = local % tiles_n
    pid_k = local // tiles_n
    if TILED_M:
        lock = LOCK + pid_m * tiles_n + pid_n
    else:
        lock = LOCK + pid_n
    _gemm_tile(X, W, Y, P, lock, M, N, K,
               stride_xm, stride_xk, stride_wn, stride_wk,
               stride_ym, stride_yn, stride_pk, stride_pm, stride_pn,
               pid_n, pid_k, pid_m,
               BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, EVEN_K, EVEN_N, TILED_M, ROTATE_K)


@triton.jit
def _thin_gemm_group_kernel(
    X, M, K, stride_xm, stride_xk,
    W0, Y0, P0, L0, N0, T0, swn0, swk0, sym0, syn0, spk0, spm0, spn0,
    W1, Y1, P1, L1, N1, T1, swn1, swk1, sym1, syn1, spk1, spm1, spn1,
    W2, Y2, P2, L2, N2, T2, swn2, swk2, sym2, syn2, spk2, spm2, spn2,
    C1, C2,
    BLOCK_M: tl.constexpr, TILED_M: tl.constexpr, NUM_MEMBERS: tl.constexpr,
    BN0: tl.constexpr, BK0: tl.constexpr, SK0: tl.constexpr,
    EK0: tl.constexpr, EN0: tl.constexpr, RK0: tl.constexpr,
    BN1: tl.constexpr, BK1: tl.constexpr, SK1: tl.constexpr,
    EK1: tl.constexpr, EN1: tl.constexpr, RK1: tl.constexpr,
    BN2: tl.constexpr, BK2: tl.constexpr, SK2: tl.constexpr,
    EK2: tl.constexpr, EN2: tl.constexpr, RK2: tl.constexpr,
):
    """One launch for a concat group: CTAs [0, C1) are member 0's grid, [C1, C2)
    member 1's, [C2, ...) member 2's; each member keeps its own schedule
    (BLOCK_N, BLOCK_K, SPLIT_K, rotation), workspace, arrival counters and
    output (pointer and strides)."""
    pid = tl.program_id(0)
    if TILED_M:
        pid_m = tl.program_id(1)
    else:
        pid_m = 0
    if pid < C1:
        _group_member(X, W0, Y0, P0, L0, M, N0, K, pid, pid_m, T0,
                      stride_xm, stride_xk, swn0, swk0, sym0, syn0,
                      spk0, spm0, spn0,
                      BLOCK_M, BN0, BK0, SK0, EK0, EN0, TILED_M, RK0)
    elif pid < C2:
        _group_member(X, W1, Y1, P1, L1, M, N1, K, pid - C1, pid_m, T1,
                      stride_xm, stride_xk, swn1, swk1, sym1, syn1,
                      spk1, spm1, spn1,
                      BLOCK_M, BN1, BK1, SK1, EK1, EN1, TILED_M, RK1)
    else:
        if NUM_MEMBERS == 3:
            _group_member(X, W2, Y2, P2, L2, M, N2, K, pid - C2, pid_m, T2,
                          stride_xm, stride_xk, swn2, swk2, sym2, syn2,
                          spk2, spm2, spn2,
                          BLOCK_M, BN2, BK2, SK2, EK2, EN2, TILED_M, RK2)


def _group_partials(numel, device):
    """Flat FP32 partials for every split-K member of one group launch."""
    key = _lane_key(device.index, "group", numel)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.empty(max(1, numel), dtype=torch.float32, device=device)
        _WORKSPACE[key] = buf
    return buf


def thin_gemm_group(x: torch.Tensor, ws, out: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ cat(ws).T for weights that read the same input x (a concat group).

    ws: the members' own [N_i, K] bf16 weights (separate tensors, read in place:
    no concatenated copy).  Member i lands in columns [off_i, off_i + N_i) of one
    [M, sum N_i] bf16 output (out, if given, may be a strided view).  Up to three
    members run in ONE launch (two serialized launches cost 2.1-3.6 us more than
    one); each member keeps the schedule its own thin_gemm would use, so every
    column has the same K order as a separate call.
    """
    assert x.dtype == torch.bfloat16 and x.ndim == 2
    M, K = x.shape
    Ns = [w.shape[0] for w in ws]
    for w in ws:
        assert w.dtype == torch.bfloat16 and w.ndim == 2 and w.shape[1] == K
    if out is None:
        out = torch.empty((M, sum(Ns)), dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return out
    members = []
    off = 0
    for w, n in zip(ws, Ns):
        if n > 0:
            members.append((w, out[:, off:off + n]))
        off += n
    _group_run(x, members)
    return out


def thin_gemm_group_split(x: torch.Tensor, ws) -> list[torch.Tensor]:
    """``thin_gemm_group`` with member i in its own contiguous [M, N_i] tensor.

    Bitwise ``[thin_gemm(x, w) for w in ws]``, in one launch per (up to) three
    members.
    """
    assert x.dtype == torch.bfloat16 and x.ndim == 2
    M, K = x.shape
    for w in ws:
        assert w.dtype == torch.bfloat16 and w.ndim == 2 and w.shape[1] == K
    outs = [torch.empty((M, w.shape[0]), dtype=torch.bfloat16, device=x.device)
            for w in ws]
    if M > 0:
        _group_run(x, [(w, y) for w, y in zip(ws, outs) if w.shape[0] > 0])
    return outs


def _group_run(x, members) -> None:
    for i in range(0, len(members), 3):
        chunk = members[i:i + 3]
        if len(chunk) == 1:
            thin_gemm(x, chunk[0][0], out=chunk[0][1])
        else:
            _group_launch(x, chunk)


def _group_launch(x, members) -> None:
    """One _thin_gemm_group_kernel launch over 2-3 (w, out) members."""
    M, K = x.shape
    # Split-K members' CTAs go first (stable otherwise): their reduction tail
    # then hides under the unsplit member's streaming instead of trailing the
    # launch (qkva_kpool M4: 0.96x -> 1.35x over two launches; at M24, where
    # both members split, the given order measured better).  Members own their
    # output columns, so the CTA order does not change any result.
    cfgs = [_select_config(M, w.shape[0], K) for w, _ in members]
    order = sorted(range(len(members)), key=lambda i: cfgs[i][3] == 1)
    members = [members[i] for i in order]
    cfgs = [cfgs[i] for i in order]
    BLOCK_M = cfgs[0][0]
    tiles_m = triton.cdiv(M, BLOCK_M)
    # num_warps / num_stages are per launch: take the member with the most
    # weight bytes, then trim the pipeline until every member's tiles fit.
    lead = max(range(len(members)), key=lambda i: members[i][0].shape[0])
    num_warps, num_stages = cfgs[lead][4], cfgs[lead][5]
    while num_stages > 2 and any(
            num_stages * (c[1] + BLOCK_M) * c[2] * 2 > _MAX_SMEM for c in cfgs):
        num_stages -= 1

    part_sizes, lock_sizes = [], []
    for (w, _), c in zip(members, cfgs):
        n, split = w.shape[0], c[3]
        part_sizes.append(split * M * n if split > 1 else 0)
        lock_sizes.append(triton.cdiv(n, c[1]) * tiles_m if split > 1 else 0)
    p_all = _group_partials(sum(part_sizes), x.device)
    l_all = _locks(max(1, sum(lock_sizes)), x.device)

    args, consts = [], {}
    starts, cta = [], 0
    p_off = l_off = 0
    for i in range(3):
        if i < len(members):
            w, y = members[i]
            _, BLOCK_N, BLOCK_K, SPLIT_K, _, _ = cfgs[i]
            n = w.shape[0]
            even_k = (K % (BLOCK_K * SPLIT_K)) == 0
            tiles_n = triton.cdiv(n, BLOCK_N)
            if SPLIT_K > 1:
                p = p_all[p_off:p_off + part_sizes[i]].view(SPLIT_K, M, n)
                sp = (p.stride(0), p.stride(1), p.stride(2))
                lock = l_all[l_off:l_off + lock_sizes[i]]
            else:
                p, sp, lock = p_all, (0, 0, 0), l_all
            p_off += part_sizes[i]
            l_off += lock_sizes[i]
            args += [w, y, p, lock, n, tiles_n, w.stride(0), w.stride(1),
                     y.stride(0), y.stride(1), *sp]
            consts.update({
                f"BN{i}": BLOCK_N, f"BK{i}": BLOCK_K, f"SK{i}": SPLIT_K,
                f"EK{i}": even_k, f"EN{i}": (n % BLOCK_N) == 0,
                f"RK{i}": _rotate_k(even_k, SPLIT_K, BLOCK_M, BLOCK_N),
            })
            if i > 0:
                starts.append(cta)
            cta += tiles_n * SPLIT_K
        else:
            w, y = members[0]
            args += [w, y, p_all, l_all, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            consts.update({f"BN{i}": 16, f"BK{i}": 16, f"SK{i}": 1,
                           f"EK{i}": True, f"EN{i}": True, f"RK{i}": False})
            if i > 0:
                starts.append(cta)
    _thin_gemm_group_kernel[(cta, tiles_m)](
        x, M, K, x.stride(0), x.stride(1),
        *args, *starts,
        BLOCK_M=BLOCK_M, TILED_M=(tiles_m > 1), NUM_MEMBERS=len(members),
        **consts, num_warps=num_warps, num_stages=num_stages,
    )
