/*
 * SPDX-License-Identifier: MIT
 *
 * Tail-biting ring mapping and mul1 constants adapted from ExLlamaV3
 * exllamav3/modules/quant/exl3_lib/ngram_codec.py, revision
 * d3739fd393337b1ff4d6c2a342b12f0c87a9592f.
 * Modifications: allocation-free selected-row CUDA decoder, generic integer-bit
 * layouts, global-row ownership/head masks, and BF16 output with explicitly
 * separated FP32 multiplication and addition. No upstream runtime is used.
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

#include "ple.h"

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

// Return the last head starting at or before id, then validate its real vocab
// extent. Gaps and table-padding rows must not read packed data or head bias.
__device__ __forceinline__ int64_t ple_head_for_id(
    int64_t id, const int64_t* offsets, const int64_t* vocab_sizes,
    int64_t heads) {
  int64_t lo = 0;
  int64_t hi = heads;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (offsets[mid] <= id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo == 0) {
    return -1;
  }
  const int64_t head = lo - 1;
  const int64_t start = offsets[head];
  const int64_t size = vocab_sizes[head];
  // Subtraction avoids overflowing a signed 64-bit start + size endpoint.
  return start >= 0 && size > 0 && id >= start && id - start < size
             ? head
             : -1;
}

__device__ __forceinline__ float ple_mul1_value(uint32_t state) {
  // Unsigned arithmetic supplies the codec's modulo-2^32 product. The two
  // constants are stored FP16 bit patterns, not approximated decimal literals.
  const uint32_t product = state * uint32_t{0x83DCD12D};
  const uint32_t byte_sum = (product & 255u) + ((product >> 8) & 255u) +
                            ((product >> 16) & 255u) + (product >> 24);
  const float k_inv = __half2float(__ushort_as_half(0x1eee));
  const float k_bias = __half2float(__ushort_as_half(0xc931));
  const float value = __fadd_rn(
      __fmul_rn(static_cast<float>(1024u + byte_sum), k_inv), k_bias);
  // The codebook value itself must round to FP16 before applying row scale.
  return __half2float(__float2half_rn(value));
}

__global__ void ple_decode_kernel(
    const uint16_t* packed_rows, const int64_t* global_ids,
    const __half* head_bias, const int64_t* head_offsets,
    const int64_t* head_vocab_sizes, int64_t rows, int64_t words,
    int64_t heads, int64_t dim, int64_t owner_lo, int64_t owner_hi,
    int bits, __nv_bfloat16* out) {
  __shared__ int64_t row_head;
  __shared__ float row_scale;

  // A bounded grid also handles row counts beyond CUDA's grid-x limit.
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    if (threadIdx.x == 0) {
      const int64_t id = global_ids[row];
      row_head = -1;
      row_scale = 0.0f;
      if (id >= 0 && id >= owner_lo && id < owner_hi) {
        row_head = ple_head_for_id(id, head_offsets, head_vocab_sizes, heads);
        if (row_head >= 0) {
          row_scale = __half2float(__ushort_as_half(packed_rows[row * words]));
        }
      }
    }
    __syncthreads();

    for (int64_t channel = threadIdx.x; channel < dim;
         channel += blockDim.x) {
      if (row_head < 0) {
        out[row * dim + channel] = __float2bfloat16_rn(0.0f);
        continue;
      }

      uint32_t state = 0;
#pragma unroll
      for (int m = 0; m < 16; ++m) {
        // The independent ring proof maps state bit m to
        // ((channel - floor(m / bits)) mod dim) * bits + m % bits.
        // dim*bits is a nonempty whole-word ring, so floor(m/bits)<dim.
        const int64_t back = m / bits;
        const int64_t source_channel =
            channel >= back ? channel - back : dim - (back - channel);
        const int64_t ring_bit = source_channel * bits + m % bits;
        const uint16_t word = packed_rows[row * words + 1 + ring_bit / 16];
        state |= static_cast<uint32_t>((word >> (ring_bit % 16)) & 1u) << m;
      }

      // Do not replace these intrinsics with an expression the compiler may
      // contract into FMA: the FP32 product must round before bias addition.
      const float scaled = __fmul_rn(ple_mul1_value(state), row_scale);
      const float biased = __fadd_rn(
          scaled, __half2float(head_bias[row_head * dim + channel]));
      out[row * dim + channel] = __float2bfloat16_rn(biased);
    }
    // Prevent early threads from replacing shared row metadata while another
    // thread is still processing the preceding row's channels.
    __syncthreads();
  }
}

void ple_check_tensor(const torch::Tensor& tensor, const char* name,
                      at::ScalarType dtype, int64_t ndim,
                      const torch::Device& device) {
  TORCH_CHECK(tensor.defined(), "ple_decode_out: ", name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), "ple_decode_out: ", name,
              " must be a CUDA tensor (use a CUDA/UVA view for pinned rows)");
  TORCH_CHECK(tensor.device() == device, "ple_decode_out: ", name,
              " must be on the output device");
  TORCH_CHECK(tensor.scalar_type() == dtype, "ple_decode_out: ", name,
              " has an invalid dtype");
  TORCH_CHECK(tensor.dim() == ndim, "ple_decode_out: ", name,
              " must have ", ndim, " dimensions");
  TORCH_CHECK(tensor.is_contiguous(), "ple_decode_out: ", name,
              " must be contiguous");
  TORCH_CHECK(tensor.layout() == torch::kStrided, "ple_decode_out: ", name,
              " must have strided layout");
}

}  // namespace

void ple_decode_out(const torch::Tensor& packed_rows,
                    const torch::Tensor& global_ids,
                    const torch::Tensor& head_bias,
                    const torch::Tensor& head_offsets,
                    const torch::Tensor& head_vocab_sizes, int64_t owner_lo,
                    int64_t owner_hi, int64_t bits, torch::Tensor& out) {
  TORCH_CHECK(out.defined() && out.is_cuda(),
              "ple_decode_out: out must be a defined CUDA tensor");
  const auto device = out.device();
  ple_check_tensor(out, "out", at::kBFloat16, 2, device);
  ple_check_tensor(packed_rows, "packed_rows", at::kShort, 2, device);
  ple_check_tensor(global_ids, "global_ids", at::kLong, 1, device);
  ple_check_tensor(head_bias, "head_bias", at::kHalf, 2, device);
  ple_check_tensor(head_offsets, "head_offsets", at::kLong, 1, device);
  ple_check_tensor(head_vocab_sizes, "head_vocab_sizes", at::kLong, 1, device);
  TORCH_CHECK(bits >= 1 && bits <= 16,
              "ple_decode_out: bits must be an integer in [1, 16]");
  TORCH_CHECK(owner_lo >= 0 && owner_hi >= owner_lo,
              "ple_decode_out: owner range must satisfy 0 <= owner_lo <= owner_hi");

  const int64_t rows = global_ids.size(0);
  const int64_t dim = out.size(1);
  const int64_t heads = head_bias.size(0);
  constexpr int64_t max_index = std::numeric_limits<int64_t>::max();
  TORCH_CHECK(dim > 0 && dim <= (max_index - 256) / bits,
              "ple_decode_out: dimension is empty or ring geometry overflows");
  const int64_t ring_bits = dim * bits;
  TORCH_CHECK(ring_bits % 16 == 0,
              "ple_decode_out: dim * bits must be divisible by 16");
  const int64_t words = 1 + ring_bits / 16;
  TORCH_CHECK(rows <= (max_index - 65535) / words,
              "ple_decode_out: packed row geometry overflows 64-bit indexing");
  TORCH_CHECK(rows <= max_index / dim && heads <= max_index / dim,
              "ple_decode_out: output or bias geometry overflows 64-bit indexing");
  TORCH_CHECK(out.size(0) == rows && packed_rows.size(0) == rows &&
                  packed_rows.size(1) == words,
              "ple_decode_out: packed_rows must be [R, 1 + dim*bits/16] and "
              "out must be [R, dim], with R = global_ids.numel()");
  TORCH_CHECK(heads > 0 && head_bias.size(1) == dim &&
                  head_offsets.size(0) == heads &&
                  head_vocab_sizes.size(0) == heads,
              "ple_decode_out: head_bias must be [heads, dim] and offsets/"
              "vocab sizes must be [heads], with heads > 0");

  at::assert_no_overlap(out, packed_rows);
  at::assert_no_overlap(out, global_ids);
  at::assert_no_overlap(out, head_bias);
  at::assert_no_overlap(out, head_offsets);
  at::assert_no_overlap(out, head_vocab_sizes);
  if (rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(device);
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  const unsigned int blocks =
      static_cast<unsigned int>(std::min<int64_t>(rows, 65535));
  ple_decode_kernel<<<blocks, 256, 0, stream.stream()>>>(
      reinterpret_cast<const uint16_t*>(packed_rows.data_ptr<int16_t>()),
      global_ids.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(head_bias.data_ptr<at::Half>()),
      head_offsets.data_ptr<int64_t>(), head_vocab_sizes.data_ptr<int64_t>(),
      rows, words, heads, dim, owner_lo, owner_hi, static_cast<int>(bits),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
