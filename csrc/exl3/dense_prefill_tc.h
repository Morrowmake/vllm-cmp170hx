#pragma once
// Tensor-core dense EXL3 linear for prefill launches (VLLM_EXL3_DENSE_PREFILL_TC;
// dense_prefill_tc.cu). Registered into _exl3_C by its own library fragment.
//
// Arithmetic (differs from linear_out's FP32 chain in the last bits):
//   a = H128(x * suh) exactly as linear_out's linear_input (FP32);
//   per (row, H128 block of k) a power-of-two scale s puts the block's max |a| in [2^10, 2^11);
//   s * a = hi + lo + r with hi = rint(s * a) (an integer, exact in FP16), lo = fp16_rn(s * a - hi) (|lo| <= 1/2,
//   |r| <= 2^-13); W (exact FP16, |W| < 4) = wh + wl with wh on the 2^-7 grid, wl = W - wh (|wl| <= 2^-8).
//   The tensor cores truncate (round toward zero) every inexact sum, which biases |result| down coherently
//   over k; so the dominant term is made exact: for every k16 step, dh = mma(wh^T, hi) from C = 0 is a sum of
//   integer multiples of 2^-7 below 2^24 of them (exact in FP32); the small terms dl = mma(W^T, lo) and
//   dw = mma(wl^T, hi), each from C = 0 (independent sums: a dependent C chain stalls the tensor pipe), then
//   acc = fma(dh + (dl + dw), 1 / s, acc) in FP32 round-to-nearest, k16 steps in increasing k; when the
//   schedule splits a tile's k range over blocks (stream-K), each segment accumulates from 0 and the segments
//   are summed in k order in FP32;
//   out = H128(acc) * svh in FP32, rounded to BF16 for a BF16 out, exactly as linear_out's linear_output (a block
//   owns whole H128 column blocks).
// Deterministic: the schedule and the k split points are a function of the launch shape and the device's
// multiprocessor count only, the split tiles' partial sums are added in a fixed (k) order whichever block
// finishes the tile, and no row's value depends on another row's. A row's bits do depend on the row count of
// its launch and the row's position in it: the schedule follows the row count, and a tile in the split region
// sums k segments that each start from 0 where a whole tile runs one chain.
#include <torch/types.h>
#include <cstdint>
namespace exl3 {
// Same signature, checks and workspace (linear_workspace_size) as linear_out. Refuses a launch for which
// linear_prefill_applies (below) is false: the caller keeps those on linear_out / linear_out_splitk.
void linear_out_prefill(const torch::Tensor& x, const torch::Tensor& trellis, const torch::Tensor& suh,
                        const torch::Tensor& svh, int64_t bits_x2, int64_t codebook, int64_t logical_in,
                        int64_t logical_out, torch::Tensor& out, torch::Tensor& workspace);
// Whether linear_out_prefill serves a launch of `rows` rows of a (padded_in x padded_out) matrix stored as
// (bits_x2, codebook): more than 64 rows, at least 16 output tiles of 128 x 128, a format it covers (bits_x2
// 10/12/14, codebook 2), and its activation scratch within linear_workspace_size. The trellis must also be
// 16-byte aligned (checked by the operator). Host arithmetic only.
bool linear_prefill_applies(int64_t rows, int64_t padded_in, int64_t padded_out, int64_t bits_x2,
                            int64_t codebook);
// Diagnostic: 0 = the planned schedule (stream-K for the last wave when it predicts a gain), 1 = whole tiles
// only. Process-wide; set before any launch.
void set_prefill_schedule(int64_t mode);
}  // namespace exl3
