#pragma once

// Row-blocked EXL3 core: result[row][col] = sum over k of input[row][k] *
// W[k][col], where W is decoded from the trellis. Each block decodes a
// BK x BN slab of W into shared memory once and applies it to up to BM rows.
//
// Bit-exactness contract with the scalar reference kernels: every output
// element is one sequential chain sum = __fmaf_rn(input[k], W[k][col], sum)
// for k = 0, 1, ..., K - 1 in increasing order, starting from 0.0f, with the
// identical FP32 weight value. The block shape only decides which thread runs
// a chain and which rows share a decoded slab; it never splits or reorders a
// chain, so a row's result is independent of the other rows in the launch.

#include "codec.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace exl3 {

// MoE routing table in the int32 tail of the MoE workspace, written by the
// route kernel each call (moe.cu):
//   [0] local assignment count, [1] row-tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted index, rows, 0}.
struct RouteLayout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() { return kHeader; }
  __host__ __device__ static constexpr int64_t tiles_offset(int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
};

struct GemmProblem {
  const float* a;      // activation rows, row stride a_stride
  float* c;            // raw core results, row stride c_stride
  int64_t a_stride;
  int64_t c_stride;
  int k;               // reduction length, multiple of 128
  int n;               // computed output columns, multiple of 128
  // Dense: one matrix applied to rows 0 .. rows - 1.
  // MoE: rows is the assignment count (the routing table's capacity).
  const uint16_t* trellis;
  int64_t tiles_n;     // tile columns of the packed matrix (padded_out / 16)
  int format;          // format_key(bits_x2, codebook)
  int64_t rows;
  // MoE (route != nullptr): rows and matrix come from the routing table.
  const int* route;
  const int64_t* pointers;   // [local][3 projections][trellis, suh, svh]
  const int64_t* metadata;   // [local][3 projections][padded_in, padded_out, bits_x2, codebook]
  int projection;
  // MoE gate and up in one launch (x != nullptr; tiny and small shapes): the
  // activation of projection q for assignment a is H128(x[a / topk] * suh_q),
  // formed in the block from the BF16 rows; gate results go to c, up to c_up.
  const __nv_bfloat16* x;
  int64_t hidden;
  int64_t topk;
  float* c_up;
};

enum class GemmShape { kTiny, kSmall, kMedium, kLarge };

// Whether a shape forms MoE gate/up activations in the block (see x above).
bool gemm_fuses_input(GemmShape shape);

int gemm_block_rows(GemmShape shape);
int gemm_block_cols(GemmShape shape);
// Rows per block are chosen from host-known sizes only; any choice gives
// bitwise identical rows.
GemmShape dense_gemm_shape(int64_t rows, int64_t n, int sms);
GemmShape moe_gemm_shape(int64_t assignments, int64_t global_experts);
// Whether moe_out runs gate and up in one launch with in-block activations.
bool moe_fuses_gate_up(GemmShape shape, int64_t assignments);
// Launches a fixed grid of min(max_items, limit) blocks; blocks stride over
// the items the problem actually has (MoE: known on the device only).
void launch_gemm(const GemmProblem& problem, GemmShape shape, int64_t max_items,
                 cudaStream_t stream);

}  // namespace exl3
