#include "ops.h"
#include "common.h"
#include "codec.cuh"
#include "gemm.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

namespace exl3 {
namespace {

constexpr int kWarpsPerBlock = 8;

unsigned int warp_blocks(int64_t tasks) {
  return launch_blocks((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

// One warp per (row, 128-column block): x * suh, then H128.
__global__ void linear_input(const __nv_bfloat16* x, const half* suh, int64_t rows,
                             int64_t logical_in, int64_t padded_in, float* transformed) {
  const int lane = threadIdx.x & 31;
  const int64_t blocks = padded_in / 128;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < rows * blocks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t row = task / blocks;
    const int64_t base = (task % blocks) * 128 + lane;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      v[i] = col < logical_in ? __fmul_rn(__bfloat162float(x[row * logical_in + col]),
                                          __half2float(suh[col]))
                              : 0.0f;
    }
    hadamard128_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) transformed[row * padded_in + base + 32 * i] = v[i];
  }
}

// One warp per (row, 128-column block): H128, then svh, stored as Out
// (store_output).
template <typename Out>
__global__ void linear_output(const float* result, const half* svh, int64_t rows,
                              int64_t padded_out, int64_t logical_out, Out* out) {
  const int lane = threadIdx.x & 31;
  const int64_t blocks = padded_out / 128;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < rows * blocks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t row = task / blocks;
    const int64_t base = (task % blocks) * 128 + lane;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = result[row * padded_out + base + 32 * i];
    hadamard128_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      if (col < logical_out) {
        store_output(out + row * logical_out + col,
                     __fmul_rn(v[i], __half2float(svh[col])));
      }
    }
  }
}

}  // namespace

int64_t linear_workspace_size(int64_t max_rows, int64_t padded_in,
                              int64_t padded_out) {
  TORCH_CHECK(max_rows >= 0, "EXL3 max_rows must be nonnegative");
  check_padded(padded_in);
  check_padded(padded_out);
  return workspace_elements(checked_mul(max_rows, checked_add(padded_in, padded_out)));
}

void linear_out(const torch::Tensor& x, const torch::Tensor& trellis,
                const torch::Tensor& suh, const torch::Tensor& svh,
                int64_t bits_x2, int64_t codebook, int64_t logical_in,
                int64_t logical_out, torch::Tensor& out,
                torch::Tensor& workspace) {
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_tensor(trellis, x, at::kShort, "trellis");
  check_tensor(suh, x, at::kHalf, "suh");
  check_tensor(svh, x, at::kHalf, "svh");
  check_linear_out(out, x);
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(x.dim() == 2 && trellis.dim() == 3 && suh.dim() == 1 &&
                  svh.dim() == 1 && out.dim() == 2,
              "EXL3 linear requires rank-2 x/out, rank-3 trellis and rank-1 scales");
  TORCH_CHECK(trellis.size(2) == 8 * bits_x2, "EXL3 packed word stride mismatch");
  const int64_t padded_in = checked_mul(trellis.size(0), 16);
  const int64_t padded_out = checked_mul(trellis.size(1), 16);
  check_padded(padded_in);
  check_padded(padded_out);
  TORCH_CHECK(logical_in > 0 && logical_in <= padded_in &&
                  logical_out > 0 && logical_out <= padded_out,
              "EXL3 logical dimensions must lie within the packed geometry");
  TORCH_CHECK(x.size(1) == logical_in && out.size(0) == x.size(0) &&
                  out.size(1) == logical_out && suh.numel() == padded_in &&
                  svh.numel() == padded_out,
              "EXL3 linear input/output/scale geometry mismatch");
  const int64_t rows = x.size(0);
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "EXL3 linear has too many rows");
  TORCH_CHECK(workspace.numel() >= linear_workspace_size(rows, padded_in, padded_out),
              "EXL3 linear workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 linear workspace must be 16-byte aligned");
  check_mutations(out, workspace, {&x, &trellis, &suh, &svh});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const auto* suh_ptr = reinterpret_cast<const half*>(suh.data_ptr<at::Half>());
  const auto* svh_ptr = reinterpret_cast<const half*>(svh.data_ptr<at::Half>());
  float* transformed = workspace.data_ptr<float>();
  float* result = transformed + rows * padded_in;
  linear_input<<<warp_blocks(rows * (padded_in / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), suh_ptr, rows,
      logical_in, padded_in, transformed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  GemmProblem problem{};
  problem.a = transformed;
  problem.c = result;
  problem.a_stride = padded_in;
  problem.c_stride = padded_out;
  problem.k = static_cast<int>(padded_in);
  problem.n = static_cast<int>(padded_out);
  problem.trellis = reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>());
  problem.tiles_n = padded_out / 16;
  problem.format = format_key(static_cast<int>(bits_x2), static_cast<int>(codebook));
  problem.rows = rows;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const GemmShape shape = dense_gemm_shape(rows, padded_out, sms);
  const int64_t block_rows = gemm_block_rows(shape);
  launch_gemm(problem, shape,
              (rows + block_rows - 1) / block_rows * (padded_out / gemm_block_cols(shape)),
              stream);

  with_linear_out(out, [&](auto* out_ptr) {
    linear_output<<<warp_blocks(rows * (padded_out / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
        result, svh_ptr, rows, padded_out, logical_out, out_ptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
