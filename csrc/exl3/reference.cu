/*
 * Reference EXL3 kernels: the original scalar decoder, kept unchanged as the
 * bit-exact oracle for the row-blocked kernels in dense.cu/moe.cu/gemm.cuh.
 * Test-only operators; serving never calls them.
 *
 * EXL3 codec adapted from turboderp-org/exllamav3, revision
 * d3739fd393337b1ff4d6c2a342b12f0c87a9592f:
 * quant/{pack.cu,frac.cu,codebook.cuh,exl3_dq.cuh,hadamard_inner.cuh}
 * and modules/quant/exl3_lib/quantize.py.
 * Modifications: scalar generic ring decoder, inverse lane permutation,
 * BF16 operator integration, and FP32 normalized H128 arithmetic.
 *
 * MIT License
 *
 * Copyright (c) 2025 Turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "ops.h"
#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace exl3 {
namespace {

constexpr float kH128Scale = 0.08838834764831845f;

struct Matrix {
  const uint16_t* trellis;
  const half* suh;
  const half* svh;
  int64_t padded_in;
  int64_t padded_out;
  int bits_x2;
  int codebook;
};

// The codebook is stored/procedurally generated in FP16 by the format. Only
// its decoded value is widened; activation transforms and dots remain FP32.
__device__ __forceinline__ float codebook_value(uint32_t state, int codebook) {
  if (codebook == 1) {
    uint32_t product = state * 0xCBAC1FEDu;
    product = (product & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__hadd(__ushort_as_half(product & 0xffffu),
                              __ushort_as_half(product >> 16)));
  }
  const uint32_t product = state * 0x83DCD12Du;
  const uint32_t sum = (product & 255u) + ((product >> 8) & 255u) +
                       ((product >> 16) & 255u) + (product >> 24);
  return __half2float(__hfma(__ushort_as_half(0x6400u + sum),
                            __ushort_as_half(0x1eee),
                            __ushort_as_half(0xc931)));
}

__device__ __forceinline__ float core_value(const Matrix& matrix,
                                            int64_t row, int64_t col) {
  const int r = row & 15;
  const int c = col & 15;
  // Invert tensor_core_perm: lane owns eight interleaved row/column values.
  const int lane = ((c & 7) * 4) + ((r & 7) >> 1);
  const int index = lane * 8 + (r & 1) + ((r >> 3) * 2) + ((c >> 3) * 4);
  const int whole = matrix.bits_x2 / 2;
  const int end = (index + 1) * whole +
                  ((matrix.bits_x2 & 1) ? (index + 1) / 2 : 0);
  const int ring_bits = 128 * matrix.bits_x2;
  const int start = (end - 16 + ring_bits) % ring_bits;
  const int words = ring_bits / 32;
  const int64_t tile = (row / 16) * (matrix.padded_out / 16) + col / 16;
  const uint16_t* packed = matrix.trellis + tile * (8 * matrix.bits_x2);
  // pack_trellis swaps 16-bit pairs, leaving MSB-first uint32 ring words.
  // Build words explicitly so an I16 view need not have uint32 alignment.
  const int lo = start / 32;
  const int hi = (lo + 1) % words;
  const uint32_t a = uint32_t(packed[2 * lo]) |
                     (uint32_t(packed[2 * lo + 1]) << 16);
  const uint32_t b = uint32_t(packed[2 * hi]) |
                     (uint32_t(packed[2 * hi + 1]) << 16);
  const uint64_t window = (uint64_t(a) << 32) | b;
  const uint32_t state = (window >> (48 - (start & 31))) & 0xffffu;
  return codebook_value(state, matrix.codebook);
}

// One 128-thread block. Standard symmetric Sylvester ordering, not sequency
// ordering. Every caller participates even for masked routes/padded channels.
__device__ __forceinline__ float hadamard128(float value, float* shared) {
  const int lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  #pragma unroll
  for (int distance = 1; distance < 128; distance <<= 1) {
    const float other = shared[lane ^ distance];
    const float next = (lane & distance) ? __fsub_rn(other, value)
                                         : __fadd_rn(value, other);
    __syncthreads();
    shared[lane] = next;
    value = next;
    __syncthreads();
  }
  return __fmul_rn(value, kH128Scale);
}

__global__ void linear_input(const __nv_bfloat16* x, Matrix matrix,
                             int64_t rows, int64_t logical_in, float* transformed) {
  __shared__ float shared[128];
  const int64_t blocks = matrix.padded_in / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col = (task % blocks) * 128 + threadIdx.x;
    const float value = col < logical_in
        ? __fmul_rn(__bfloat162float(x[row * logical_in + col]),
                     __half2float(matrix.suh[col])) : 0.0f;
    transformed[row * matrix.padded_in + col] = hadamard128(value, shared);
  }
}

__global__ void linear_core(const float* input, Matrix matrix, int64_t rows,
                            float* result) {
  __shared__ float shared_input[128];
  const int64_t blocks = matrix.padded_out / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col = (task % blocks) * 128 + threadIdx.x;
    float sum = 0.0f;
    for (int64_t base = 0; base < matrix.padded_in; base += 128) {
      shared_input[threadIdx.x] = input[row * matrix.padded_in + base + threadIdx.x];
      __syncthreads();
      for (int k = 0; k < 128; ++k) {
        sum = __fmaf_rn(shared_input[k], core_value(matrix, base + k, col), sum);
      }
      __syncthreads();
    }
    result[row * matrix.padded_out + col] = sum;
  }
}

__global__ void linear_output(const float* result, Matrix matrix, int64_t rows,
                              int64_t logical_out, __nv_bfloat16* out) {
  __shared__ float shared[128];
  const int64_t blocks = matrix.padded_out / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col = (task % blocks) * 128 + threadIdx.x;
    const float value = hadamard128(result[row * matrix.padded_out + col], shared);
    if (col < logical_out) {
      out[row * logical_out + col] = __float2bfloat16_rn(
          __fmul_rn(value, __half2float(matrix.svh[col])));
    }
  }
}

struct Routing {
  const void* ids;
  const void* expert_map;
  const int64_t* pointers;
  const int64_t* metadata;
  int64_t global_experts;
  int64_t local_experts;
  int64_t topk;
  bool ids64;
  bool map64;
};

__device__ __forceinline__ int64_t integer_at(const void* ptr, bool wide, int64_t i) {
  return wide ? static_cast<const int64_t*>(ptr)[i]
              : static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ int64_t local_expert(const Routing& routing, int64_t assignment) {
  const int64_t global = integer_at(routing.ids, routing.ids64, assignment);
  if (global < 0 || global >= routing.global_experts) return -1;
  const int64_t local = integer_at(routing.expert_map, routing.map64, global);
  return local >= 0 && local < routing.local_experts ? local : -1;
}
__device__ __forceinline__ Matrix expert_matrix(const Routing& routing,
                                               int64_t expert, int projection) {
  const int64_t* ptrs = routing.pointers + expert * 9 + projection * 3;
  const int64_t* meta = routing.metadata + expert * 12 + projection * 4;
  return {reinterpret_cast<const uint16_t*>(ptrs[0]),
          reinterpret_cast<const half*>(ptrs[1]),
          reinterpret_cast<const half*>(ptrs[2]), meta[0], meta[1],
          static_cast<int>(meta[2]), static_cast<int>(meta[3])};
}

__global__ void expert_input(const __nv_bfloat16* x, Routing routing,
                             int64_t assignments, int64_t hidden,
                             int64_t hpad, int projection, float* input) {
  __shared__ float shared[128];
  const int64_t blocks = hpad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float value = 0.0f;
    if (expert >= 0 && col < hidden) {
      const Matrix matrix = expert_matrix(routing, expert, projection);
      value = __fmul_rn(__bfloat162float(x[(assignment / routing.topk) * hidden + col]),
                         __half2float(matrix.suh[col]));
    }
    input[assignment * hpad + col] = hadamard128(value, shared);
  }
}

__global__ void expert_core(const float* input, Routing routing,
                            int64_t assignments, int64_t kpad, int64_t npad,
                            int projection, float* result) {
  __shared__ float shared_input[128];
  const int64_t blocks = npad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float sum = 0.0f;
    if (expert >= 0) {
      const Matrix matrix = expert_matrix(routing, expert, projection);
      // H128 is block-local: additional entirely padded input/output blocks
      // are zero or cropped. Use actual metadata strides to address the core.
      for (int64_t base = 0; base < kpad; base += 128) {
        shared_input[threadIdx.x] = input[assignment * kpad + base + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < 128; ++k) {
          sum = __fmaf_rn(shared_input[k], core_value(matrix, base + k, col), sum);
        }
        __syncthreads();
      }
    }
    result[assignment * npad + col] = sum;
  }
}

__global__ void expert_output(Routing routing, int64_t assignments,
                              int64_t npad, int projection, float* result) {
  __shared__ float shared[128];
  const int64_t blocks = npad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float value = hadamard128(result[assignment * npad + col], shared);
    if (expert >= 0) {
      const Matrix matrix = expert_matrix(routing, expert, projection);
      value = __fmul_rn(value, __half2float(matrix.svh[col]));
    } else {
      value = 0.0f;
    }
    result[assignment * npad + col] = value;
  }
}

__global__ void activate_down_input(Routing routing, int64_t assignments,
                                    int64_t intermediate, int64_t ipad,
                                    float* gate, const float* up, float limit) {
  __shared__ float shared[128];
  const int64_t blocks = ipad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float value = 0.0f;
    if (expert >= 0 && col < intermediate) {
      const int64_t index = assignment * ipad + col;
      float g = gate[index];
      float u = up[index];
      swiglu_clamp(g, u, limit);
      const float silu = g / __fadd_rn(1.0f, expf(-g));
      const Matrix down = expert_matrix(routing, expert, 2);
      value = __fmul_rn(__fmul_rn(silu, u), __half2float(down.suh[col]));
    }
    gate[assignment * ipad + col] = hadamard128(value, shared);
  }
}

__global__ void combine_routes(const float* down, const float* weights,
                               Routing routing, int64_t rows, int64_t hidden,
                               int64_t hpad, __nv_bfloat16* out) {
  const int64_t elements = rows * hidden;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / hidden;
    const int64_t col = index % hidden;
    float sum = 0.0f;
    for (int64_t route = 0; route < routing.topk; ++route) {
      const int64_t assignment = row * routing.topk + route;
      if (local_expert(routing, assignment) >= 0) {
        // Exactly one routing-weight application, after the complete down op.
        sum = __fmaf_rn(down[assignment * hpad + col], weights[assignment], sum);
      }
    }
    out[index] = __float2bfloat16_rn(sum);
  }
}

}  // namespace
// Today's dense operator, unchanged except for its name.
void linear_out_reference(const torch::Tensor& x, const torch::Tensor& trellis,
                          const torch::Tensor& suh, const torch::Tensor& svh,
                          int64_t bits_x2, int64_t codebook, int64_t logical_in,
                          int64_t logical_out, torch::Tensor& out,
                          torch::Tensor& workspace) {
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_tensor(trellis, x, at::kShort, "trellis");
  check_tensor(suh, x, at::kHalf, "suh");
  check_tensor(svh, x, at::kHalf, "svh");
  check_tensor(out, x, at::kBFloat16, "out");
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
  TORCH_CHECK(workspace.numel() >= checked_mul(rows, checked_add(padded_in, padded_out)),
              "EXL3 linear workspace is too small");
  check_mutations(out, workspace, {&x, &trellis, &suh, &svh});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const Matrix matrix{reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
                      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
                      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
                      padded_in, padded_out, static_cast<int>(bits_x2),
                      static_cast<int>(codebook)};
  float* transformed = workspace.data_ptr<float>();
  float* result = transformed + rows * padded_in;
  linear_input<<<launch_blocks(rows * (padded_in / 128)), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      matrix, rows, logical_in, transformed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_core<<<launch_blocks(rows * (padded_out / 128)), 128, 0, stream>>>(
      transformed, matrix, rows, result);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_output<<<launch_blocks(rows * (padded_out / 128)), 128, 0, stream>>>(
      result, matrix, rows, logical_out,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Today's MoE operator, unchanged except for its name. Its workspace need,
// assignments * (hpad + 2 * ipad), never exceeds moe_workspace_size.
void moe_out_reference(const torch::Tensor& x, const torch::Tensor& topk_ids,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& packed_ptrs,
                       const torch::Tensor& packed_meta,
                       const torch::Tensor& expert_map, int64_t intermediate_size,
                       torch::Tensor& out, torch::Tensor& workspace, double swiglu_limit) {
  const float limit = checked_swiglu_limit(swiglu_limit);
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_integer(topk_ids, x, "topk_ids");
  check_integer(expert_map, x, "expert_map");
  check_tensor(topk_weights, x, at::kFloat, "topk_weights");
  check_tensor(packed_ptrs, x, at::kLong, "packed_ptrs");
  check_tensor(packed_meta, x, at::kLong, "packed_meta");
  check_tensor(out, x, at::kBFloat16, "out");
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && topk_ids.dim() == 2 &&
                  topk_weights.dim() == 2 && expert_map.dim() == 1 &&
                  packed_ptrs.dim() == 3 && packed_meta.dim() == 3,
              "EXL3 MoE tensor ranks mismatch");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  const int64_t hpad = padded_size(hidden);
  const int64_t ipad = padded_size(intermediate_size);
  TORCH_CHECK(topk_ids.size(0) == rows && topk_weights.sizes() == topk_ids.sizes() &&
                  out.sizes() == x.sizes(), "EXL3 MoE row/topk/output geometry mismatch");
  TORCH_CHECK(packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == experts && packed_meta.size(1) == 3 &&
                  packed_meta.size(2) == 4, "EXL3 MoE pointer/metadata geometry mismatch");
  const int64_t assignments = checked_mul(rows, topk);
  const int64_t required = experts == 0 ? 0 :
      checked_mul(assignments, checked_add(hpad, checked_mul(2, ipad)));
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  check_mutations(out, workspace,
                  {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  Routing routing{topk_ids.data_ptr(), expert_map.data_ptr(),
                  packed_ptrs.data_ptr<int64_t>(), packed_meta.data_ptr<int64_t>(),
                  expert_map.numel(), experts, topk,
                  topk_ids.scalar_type() == at::kLong, expert_map.scalar_type() == at::kLong};
  float* input = workspace.data_ptr<float>();
  if (experts > 0 && assignments > 0) {
    float* gate = input + assignments * hpad;
    float* up = gate + assignments * ipad;
    const auto* bf16_input = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    for (int projection = 0; projection < 2; ++projection) {
      float* result = projection == 0 ? gate : up;
      expert_input<<<launch_blocks(assignments * (hpad / 128)), 128, 0, stream>>>(
          bf16_input, routing, assignments, hidden, hpad, projection, input);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      expert_core<<<launch_blocks(assignments * (ipad / 128)), 128, 0, stream>>>(
          input, routing, assignments, hpad, ipad, projection, result);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      expert_output<<<launch_blocks(assignments * (ipad / 128)), 128, 0, stream>>>(
          routing, assignments, ipad, projection, result);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    activate_down_input<<<launch_blocks(assignments * (ipad / 128)), 128, 0, stream>>>(
        routing, assignments, intermediate_size, ipad, gate, up, limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expert_core<<<launch_blocks(assignments * (hpad / 128)), 128, 0, stream>>>(
        gate, routing, assignments, ipad, hpad, 2, input);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expert_output<<<launch_blocks(assignments * (hpad / 128)), 128, 0, stream>>>(
        routing, assignments, hpad, 2, input);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    routing.topk = 0;
  }
  combine_routes<<<launch_blocks(checked_add(checked_mul(rows, hidden), 127) / 128), 128, 0, stream>>>(
      input, topk_weights.data_ptr<float>(), routing, rows, hidden, hpad,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
