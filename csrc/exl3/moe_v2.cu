// EXL3 routed-expert MoE on the tensor cores for decode-sized calls of the
// served shapes (LaneTiling below), every expert matrix of a layer stored with
// one bit width and the mul1 codebook:
//   - hidden 2560, intermediate 640, bits_x2 10;
//   - hidden 5120, intermediate 2304, bits_x2 6 or 4;
//   - hidden 5120, intermediate 640 or 512, bits_x2 6, 4 or 8: tensor-parallel
//     routed experts, each rank holding a 640- or 512-column chunk of every
//     expert's 2304-column intermediate (bits_x2 8: the DSpark drafter's
//     experts, VLLM_EXL3_MOE_V2_DRAFT);
//   - hidden 5120, intermediate 2304, bits_x2 8: the DSpark drafter's routed
//     experts (VLLM_EXL3_MOE_V2_DRAFT), with stages of their own (DraftTiling).
// The caller (vllm_sm80/kernels/exl3/quant.py) takes this operator only for
// the configurations ops.moe_v2_closed_reason opens and at most 64 rows; every
// other call stays on moe_out (moe.cu), which this file does not change.
//
// Arithmetic differs from moe_out: the expert GEMMs run m16n8k16 FP16 MMAs on
// exactly decoded weights and a hi/lo FP16 split of the scaled activations,
// and the input, activation and combine stages run in FP64. Results are
// deterministic and a row never depends on other rows (the drafter's stages:
// see DraftTiling).
//
// moe_out_v2_fold (VLLM_EXL3_MOE_V2_FOLD) computes the same bits in three
// launches instead of five: the activation and the combine run at the end of
// the gate/up and down lane launches (FoldArgs).
//
// The hidden-5120 shape (two H128 blocks per input warp, column groups of four
// n16 tiles, lane kernels per bit width) follows Curtis White's port of this
// file to that shape (github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE
// at 88eb6e1, Apache-2.0); see plugins/sm80/licenses/exl3/NOTICE.

#include "moe_v2.h"
#include "ops.h"
#include "common.h"
#include "codec_moe_v2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <atomic>
#include <limits>
#include <type_traits>
#include <utility>

namespace exl3 {
namespace moe_v2 {
namespace {

// The route + gate/up operand launch (block 0 routes with these threads; the
// other blocks run one warp per H128 block of a row).
constexpr int kRouteInputThreads = 640;
// Route kernel shared memory: three int arrays per local expert.
constexpr int64_t kMaxRouteExperts = 4096;

// Integers the route kernel writes into the workspace each call:
//   [0] local assignment count, [1] row-tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted index, rows, 0},
//   experts[assignments]: the local expert of each sorted entry,
//   by_assignment[assignments]: the local expert of each assignment (-1: none),
//   tile_trellis[assignments][3] (int64), tile_columns[assignments][3]: per row
//   tile and projection the trellis address and padded_out / 16.
struct RouteLayout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() { return kHeader; }
  __host__ __device__ static constexpr int64_t tiles_offset(int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t experts_offset(int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
  __host__ __device__ static constexpr int64_t by_assignment_offset(int64_t assignments) {
    return experts_offset(assignments) + assignments;
  }
  // Per row tile and projection: the expert's trellis address (int64, 8-byte
  // aligned) and its tile columns (padded_out / 16).
  __host__ __device__ static constexpr int64_t tile_trellis_offset(int64_t assignments) {
    return (by_assignment_offset(assignments) + assignments + 1) / 2 * 2;
  }
  __host__ __device__ static constexpr int64_t tile_columns_offset(int64_t assignments) {
    return tile_trellis_offset(assignments) + 2 * 3 * assignments;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return tile_columns_offset(assignments) + 3 * assignments;
  }
};

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
// [trellis, suh, svh] of one projection of one local expert.
__device__ __forceinline__ const int64_t* expert_pointers(const Routing& routing,
                                                          int64_t expert, int projection) {
  return routing.pointers + expert * 9 + projection * 3;
}
__device__ __forceinline__ const half* expert_suh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[1]);
}
__device__ __forceinline__ const half* expert_svh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[2]);
}

// Exclusive block-wide scan of two counters; also returns the totals.
__device__ __forceinline__ void block_scan2(int a, int b, int& a_before, int& b_before,
                                            int& a_total, int& b_total) {
  __shared__ int warp_a[32];
  __shared__ int warp_b[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  int x = a;
  int y = b;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int xu = __shfl_up_sync(0xffffffffu, x, d);
    const int yu = __shfl_up_sync(0xffffffffu, y, d);
    if (lane >= d) {
      x += xu;
      y += yu;
    }
  }
  if (lane == 31) {
    warp_a[warp] = x;
    warp_b[warp] = y;
  }
  __syncthreads();
  if (warp == 0) {
    int wx = lane < warps ? warp_a[lane] : 0;
    int wy = lane < warps ? warp_b[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int xu = __shfl_up_sync(0xffffffffu, wx, d);
      const int yu = __shfl_up_sync(0xffffffffu, wy, d);
      if (lane >= d) {
        wx += xu;
        wy += yu;
      }
    }
    warp_a[lane] = wx;
    warp_b[lane] = wy;
  }
  __syncthreads();
  a_before = x - a + (warp > 0 ? warp_a[warp - 1] : 0);
  b_before = y - b + (warp > 0 ? warp_b[warp - 1] : 0);
  a_total = warp_a[warps - 1];
  b_total = warp_b[warps - 1];
}

// Groups the local assignments by local expert into tiles of up to
// block_rows rows (one block, every call), records each sorted entry's
// expert and each assignment's expert, and each row tile's trellis address
// and tile columns per projection. Order inside an expert is arbitrary: each
// assignment's result depends only on its own row, never on its position or
// neighbours. kOrdered: increasing assignment order inside an expert (the
// drafter's stages; the dynamic shared memory then holds one more int per
// assignment). A matrix not stored as bits_x2 kBitsX2 / codebook 2 traps.
template <int kBitsX2, bool kOrdered = false>
__device__ __forceinline__ void route_block(const Routing& routing, int64_t assignments,
                                            int block_rows, int* table) {
  extern __shared__ int route_shared[];
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;              // counts, then fill cursors
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;   // experts + 1 entries
  int* expert_of = tile_begin + experts + 1;   // kOrdered: assignments entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) cursor[e] = 0;
  __syncthreads();
  int* by_assignment = table + RouteLayout::by_assignment_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    by_assignment[a] = static_cast<int>(e);
    if constexpr (kOrdered) expert_of[a] = static_cast<int>(e);
    if (e >= 0) atomicAdd(&cursor[e], 1);
  }
  __syncthreads();
  const int per = (experts + blockDim.x - 1) / blockDim.x;
  const int e0 = min(experts, static_cast<int>(threadIdx.x) * per);
  const int e1 = min(experts, e0 + per);
  int rows = 0;
  int tiles = 0;
  for (int e = e0; e < e1; ++e) {
    rows += cursor[e];
    tiles += (cursor[e] + block_rows - 1) / block_rows;
  }
  int row_at, tile_at, row_total, tile_total;
  block_scan2(rows, tiles, row_at, tile_at, row_total, tile_total);
  for (int e = e0; e < e1; ++e) {
    const int count = cursor[e];
    row_begin[e] = row_at;
    tile_begin[e] = tile_at;
    cursor[e] = row_at;
    row_at += count;
    tile_at += (count + block_rows - 1) / block_rows;
  }
  if (threadIdx.x == 0) {
    tile_begin[experts] = tile_total;
    table[0] = row_total;
    table[1] = tile_total;
  }
  __syncthreads();
  int4* tile_table = reinterpret_cast<int4*>(table + RouteLayout::tiles_offset(assignments));
  for (int t = threadIdx.x; t < tile_total; t += blockDim.x) {
    // Last expert whose first tile is at or before t (empty experts share
    // their successor's start, so the search lands on the owner).
    int lo = 0;
    int hi = experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (tile_begin[mid] <= t) lo = mid; else hi = mid;
    }
    const int first = row_begin[lo] + (t - tile_begin[lo]) * block_rows;
    const int end = lo + 1 < experts ? row_begin[lo + 1] : row_total;
    tile_table[t] = make_int4(lo, first, min(block_rows, end - first), 0);
    int64_t* tile_trellis = reinterpret_cast<int64_t*>(
        table + RouteLayout::tile_trellis_offset(assignments));
    int* tile_columns = table + RouteLayout::tile_columns_offset(assignments);
    for (int q = 0; q < 3; ++q) {
      const int64_t* meta = routing.metadata + int64_t(lo) * 12 + q * 4;
      // The launch's format only (kBitsX2, codebook 2); anything else traps.
      if (meta[2] != kBitsX2 || meta[3] != 2) __trap();
      tile_trellis[t * 3 + q] = routing.pointers[int64_t(lo) * 9 + q * 3];
      tile_columns[t * 3 + q] = static_cast<int>(meta[1] / 16);
    }
  }
  int* sorted = table + RouteLayout::sorted_offset();
  int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    if constexpr (kOrdered) {
      // The rank among the expert's earlier assignments, not an atomic
      // arrival order.
      const int e = expert_of[a];
      if (e >= 0) {
        int rank = 0;
        for (int64_t b = 0; b < a; ++b) rank += expert_of[b] == e;
        const int at = row_begin[e] + rank;
        sorted[at] = static_cast<int>(a);
        sorted_experts[at] = e;
      }
    } else {
      const int64_t e = local_expert(routing, a);
      if (e >= 0) {
        const int at = atomicAdd(&cursor[e], 1);
        sorted[at] = static_cast<int>(a);
        sorted_experts[at] = static_cast<int>(e);
      }
    }
  }
}

__device__ __forceinline__ double silu(double g) { return g / __dadd_rn(1.0, exp(-g)); }

// Lane order of a K-long row (MMA B operand): k step k / 16, then lane t's k
// values 2t, 2t + 1, 2t + 8, 2t + 9 (t = (k % 8) / 2).
__device__ __forceinline__ int64_t lane_position(int64_t k) {
  const int kk = static_cast<int>(k & 15);
  return (k & ~int64_t(15)) + ((kk & 7) >> 1) * 4 + ((kk >> 3) << 1) + (kk & 1);
}

// The tensor-core B operand of one row: e per scale group (the peak over the
// group's warps) puts max |v| 2^e in [2^14, 2^15) (2^e and 2^-e stay normal
// FP32), hi = f16(v 2^e) (exact scaling), lo = f16(v 2^e - hi), stored at
// lane_position(k) of the row's high and low FP16 rows.
template <int kVals>
__device__ __forceinline__ float warp_peak(const float (&v)[kVals]) {
  float peak = 0.0f;
#pragma unroll
  for (int i = 0; i < kVals; ++i) peak = fmaxf(peak, fabsf(v[i]));
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
  return peak;
}
__device__ __forceinline__ int halves_exponent(float peak) {
  const int biased = static_cast<int>(__float_as_uint(peak) >> 23);
  return peak == 0.0f ? 0 : min(141 - biased, 126);
}
// The lane's values at k = first + lane + 32 i, scaled by 2^e.
template <int kVals>
__device__ __forceinline__ void store_scaled_halves(const float (&v)[kVals], int lane, int first,
                                                    int e, __half* hi, __half* lo) {
  const float scale = __uint_as_float(uint32_t(127 + e) << 23);
#pragma unroll
  for (int i = 0; i < kVals; ++i) {
    const int64_t at = lane_position(first + lane + 32 * i);
    const float y = __fmul_rn(v[i], scale);
    const __half h = __float2half_rn(y);
    hi[at] = h;
    lo[at] = __float2half_rn(__fsub_rn(y, __half2float(h)));
  }
}

// Where the down GEMM wants its input: per assignment [hi | lo] FP16 rows of
// ipad halves, and one 2^-e per (assignment, down K slice) (the same value
// for every slice: one exponent per row).
struct DownHalves {
  __half* halves;
  float* unscale;
};

// One task: the routing entry at sorted position pos, 128-column block
// `block`; returns false past the routed count, else the transformed values
// (lane holds columns block * 128 + lane + 32 i) and the assignment. Gate and
// up arrive as kSlices K-slice partials (slice_stride apart), summed in
// increasing slice order; then the gate and up output transforms (H128, svh),
// SiLU(gate) * up, down suh and the down H128, all in FP64. Three dependent
// memory round trips: the routing table (count, sorted entry and its expert
// together), then all partials with the expert's pointers, then the svh/suh
// values. Above 8 slices the partials are summed as they load (the same
// additions in the same order), which keeps 2 x kSlices x 4 of them out of
// the registers. kL2: the partials were written by other warps of the same
// launch (the folded call) and are read past L1 (load_partial).
template <bool kL2>
__device__ __forceinline__ float load_partial(const float* at) {
  if constexpr (kL2) {
    return __ldcg(at);
  } else {
    return *at;
  }
}
template <int kSlices, bool kL2 = false>
__device__ __forceinline__ bool activation_block(const Routing& routing, const int* table,
                                                 int64_t assignments, int64_t intermediate,
                                                 int64_t ipad, const float* gate, const float* up,
                                                 int64_t slice_stride, int64_t pos, int64_t block,
                                                 int lane, float limit, double (&v)[4],
                                                 int64_t& assignment) {
  const int* sorted = table + RouteLayout::sorted_offset();
  const int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  const int count = __ldg(table);
  assignment = __ldg(sorted + pos);
  const int64_t expert = __ldg(sorted_experts + pos);
  if (pos >= count) return false;
  const int64_t base = block * 128 + lane;
  const half* gate_svh = expert_svh(routing, expert, 0);
  const half* up_svh = expert_svh(routing, expert, 1);
  const half* down_suh = expert_suh(routing, expert, 2);
  float gate_scale[4];
  float up_scale[4];
  float down_scale[4];
  double g[4];
  double u[4];
  if constexpr (kSlices <= 8) {
    float gs[kSlices][4];
    float us[kSlices][4];
#pragma unroll
    for (int s = 0; s < kSlices; ++s) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        gs[s][i] = load_partial<kL2>(gate + s * slice_stride + assignment * ipad + base + 32 * i);
        us[s][i] = load_partial<kL2>(up + s * slice_stride + assignment * ipad + base + 32 * i);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      gate_scale[i] = __half2float(gate_svh[base + 32 * i]);
      up_scale[i] = __half2float(up_svh[base + 32 * i]);
      down_scale[i] = __half2float(down_suh[base + 32 * i]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      g[i] = gs[0][i];
      u[i] = us[0][i];
#pragma unroll
      for (int s = 1; s < kSlices; ++s) {
        g[i] = __dadd_rn(g[i], double(gs[s][i]));
        u[i] = __dadd_rn(u[i], double(us[s][i]));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      gate_scale[i] = __half2float(gate_svh[base + 32 * i]);
      up_scale[i] = __half2float(up_svh[base + 32 * i]);
      down_scale[i] = __half2float(down_suh[base + 32 * i]);
      g[i] = load_partial<kL2>(gate + assignment * ipad + base + 32 * i);
      u[i] = load_partial<kL2>(up + assignment * ipad + base + 32 * i);
    }
#pragma unroll
    for (int s = 1; s < kSlices; ++s) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        g[i] = __dadd_rn(g[i], double(load_partial<kL2>(gate + s * slice_stride +
                                                        assignment * ipad + base + 32 * i)));
        u[i] = __dadd_rn(u[i], double(load_partial<kL2>(up + s * slice_stride +
                                                        assignment * ipad + base + 32 * i)));
      }
    }
  }
  hadamard128_warp_f64(g, lane);
  hadamard128_warp_f64(u, lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t col = base + 32 * i;
    double gv = __dmul_rn(g[i], double(gate_scale[i]));
    double uv = __dmul_rn(u[i], double(up_scale[i]));
    v[i] = 0.0;
    if (col < intermediate) {
      swiglu_clamp(gv, uv, limit);
      v[i] = __dmul_rn(__dmul_rn(silu(gv), uv), double(down_scale[i]));
    }
  }
  hadamard128_warp_f64(v, lane);
  return true;
}

// The activation stage: rounded once to FP32 and written as the down GEMM's
// tensor-core operand with one 2^e per row (down_slices copies of 2^-e). One
// thread block per sorted position (kRowBlocks warps = ipad / 128, one per
// H128 block), the row peak from the warps' peaks in shared memory.
template <int kSlices, int kRowBlocks>
__global__ void __launch_bounds__(32 * kRowBlocks)
activate_down_row(Routing routing, const int* table, int64_t assignments, int64_t intermediate,
                  int64_t ipad, const float* gate, const float* up, int64_t slice_stride,
                  DownHalves halves, int down_slices, float limit) {
  __shared__ float warp_peaks[kRowBlocks];
  const int lane = threadIdx.x & 31;
  const int block = threadIdx.x >> 5;
  // Uniform trip count per thread block (barriers inside).
  for (int64_t pos = blockIdx.x; pos < assignments; pos += gridDim.x) {
    double v[4];
    int64_t assignment;
    if (!activation_block<kSlices>(routing, table, assignments, intermediate, ipad, gate, up,
                                   slice_stride, pos, block, lane, limit, v, assignment)) {
      break;
    }
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __double2float_rn(v[i]);
    const float peak = warp_peak(f);
    if (lane == 0) warp_peaks[block] = peak;
    __syncthreads();
    float row_peak = 0.0f;
#pragma unroll
    for (int w = 0; w < kRowBlocks; ++w) row_peak = fmaxf(row_peak, warp_peaks[w]);
    const int e = halves_exponent(row_peak);
    __half* hi = halves.halves + assignment * 2 * ipad;
    store_scaled_halves(f, lane, block * 128, e, hi, hi + ipad);
    if (block == 0 && lane < down_slices) {
      halves.unscale[assignment * down_slices + lane] = __uint_as_float(uint32_t(127 - e) << 23);
    }
    __syncthreads();
  }
}

// The drafter's activation stage: rounded once to FP32 and written as the
// down GEMM's tensor-core operand with one 2^e per (row, down K slice). One
// thread block per (sorted position, down K slice): warp w takes the slice's
// w-th overlapping H128 block (a block on a slice boundary is formed by both
// of its slices, identically), the slice peak covers the slice's own columns
// only (warp peaks in shared memory), and each slice stores only its columns,
// so a row's slices spread over the SMs.
template <int kSlices, int kSliceBlocks>
__global__ void __launch_bounds__(32 * kSliceBlocks)
activate_down_slice(Routing routing, const int* table, int64_t assignments, int64_t intermediate,
                    int64_t ipad, const float* gate, const float* up, int64_t slice_stride,
                    DownHalves halves, int down_slices, float limit) {
  __shared__ float warp_peaks[kSliceBlocks];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t width = ipad / down_slices;
  const int count = __ldg(table);
  // Uniform trip count per thread block (barriers inside); tasks run in
  // increasing sorted position, so the first one past the count ends the loop.
  for (int64_t task = blockIdx.x; task < assignments * down_slices; task += gridDim.x) {
    const int64_t pos = task / down_slices;
    if (pos >= count) break;
    const int slice = static_cast<int>(task % down_slices);
    const int64_t c0 = slice * width;
    const int64_t c1 = c0 + width;
    const int64_t block = c0 / 128 + warp;
    const bool active = block * 128 < c1;
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t assignment = 0;
    if (active) {
      double v[4];
      activation_block<kSlices>(routing, table, assignments, intermediate, ipad, gate, up,
                                slice_stride, pos, block, lane, limit, v, assignment);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = block * 128 + lane + 32 * i;
        if (col >= c0 && col < c1) f[i] = __double2float_rn(v[i]);
      }
    }
    const float peak = warp_peak(f);
    if (lane == 0) warp_peaks[warp] = peak;
    __syncthreads();
    float slice_peak = 0.0f;
#pragma unroll
    for (int w = 0; w < kSliceBlocks; ++w) slice_peak = fmaxf(slice_peak, warp_peaks[w]);
    const int e = halves_exponent(slice_peak);
    if (active) {
      __half* hi = halves.halves + assignment * 2 * ipad;
      const float scale = __uint_as_float(uint32_t(127 + e) << 23);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = block * 128 + lane + 32 * i;
        if (col >= c0 && col < c1) {
          const int64_t at = lane_position(col);
          const float y = __fmul_rn(f[i], scale);
          const __half h = __float2half_rn(y);
          hi[at] = h;
          hi[ipad + at] = __float2half_rn(__fsub_rn(y, __half2float(h)));
        }
      }
      if (threadIdx.x == 0) {
        halves.unscale[assignment * down_slices + slice] = __uint_as_float(uint32_t(127 - e) << 23);
      }
    }
    __syncthreads();
  }
}

// One block per (token, 128-column block), one warp per route (blockDim =
// 32 * min(topk, 32), at least 4 warps). Warp w handles routes w, w + warps,
// ...: the local route's down output (the sum of kSlices K-slice partials,
// slice_stride apart, in increasing slice order) transform (H128, svh) in
// FP64 into shared memory, every route's loads in flight at once. Then
// threads 0..127 apply exactly one routing-weight FMA per local route in
// route order (FP64) and round to FP32, then to BF16. Local experts come from
// the routing table's by-assignment array. Dynamic shared memory:
// combine_shared_bytes(topk).
constexpr int kCombineMaxThreads = 1024;
size_t combine_shared_bytes(int64_t topk) {
  return size_t(topk) * (128 * sizeof(double) + sizeof(float) + sizeof(int));
}
template <int kSlices>
__global__ void __launch_bounds__(kCombineMaxThreads)
combine_routes(const float* down, const float* weights, Routing routing, const int* table,
               int64_t rows, int64_t hidden, int64_t hpad, int64_t slice_stride,
               __nv_bfloat16* out) {
  extern __shared__ __align__(16) unsigned char combine_shared[];
  const int64_t topk = routing.topk;
  double* values = reinterpret_cast<double*>(combine_shared);      // [topk][128]
  float* route_weight = reinterpret_cast<float*>(values + topk * 128);
  int* route_local = reinterpret_cast<int*>(route_weight + topk);
  const int* by_assignment = table + RouteLayout::by_assignment_offset(rows * topk);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t blocks = hpad / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col0 = (task % blocks) * 128;
    for (int64_t r = warp; r < topk; r += blockDim.x / 32) {
      const int64_t assignment = row * topk + r;
      const int64_t expert = __ldg(by_assignment + assignment);
      if (lane == 0) {
        route_local[r] = expert >= 0;
        route_weight[r] = weights[assignment];
      }
      if (expert < 0) continue;
      const half* svh = expert_svh(routing, expert, 2);
      float v[kSlices][4];
#pragma unroll
      for (int s = 0; s < kSlices; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          v[s][i] = down[s * slice_stride + assignment * hpad + col0 + lane + 32 * i];
        }
      }
      float scale[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) scale[i] = __half2float(svh[col0 + lane + 32 * i]);
      double x[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[i] = v[0][i];
#pragma unroll
        for (int s = 1; s < kSlices; ++s) x[i] = __dadd_rn(x[i], double(v[s][i]));
      }
      hadamard128_warp_f64(x, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[r * 128 + lane + 32 * i] = __dmul_rn(x[i], double(scale[i]));
      }
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      double sum = 0.0;
      for (int64_t r = 0; r < topk; ++r) {
        if (route_local[r]) {
          sum = __fma_rn(values[r * 128 + threadIdx.x], double(route_weight[r]), sum);
        }
      }
      const int64_t col = col0 + threadIdx.x;
      if (col < hidden) out[row * hidden + col] = __float2bfloat16_rn(__double2float_rn(sum));
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Lane-tile expert GEMM (codebook 2, one bits_x2 per instance).
//
// An item is one warp's kNT output n16 tiles x one K slice of one route tile;
// no shared memory and no block barriers. The warp decodes its own 16x16
// trellis tiles straight into m16n8k16 A fragments (A = W^T: 16 output
// columns x 16 k; decode_lane_fragment_mul1) and multiplies them on the tensor
// cores (FP16 in, FP32 out, C = 0 per k16 step) with B = 16 k x 8 columns: per
// group of four route rows, columns 0..3 hold the rows' high FP16 parts and
// 4..7 the low parts of their activations scaled by a power of two per row
// and scale group (store_scaled_halves; produced once per call by
// route_and_gate_up_input and activate_down_row, read straight from global
// memory). The k16 results are summed in FP32 in increasing k per K slice,
// high and low apart; the slice partial (hi + lo) 2^-e goes to the workspace
// and the consumer kernel sums the slices in increasing slice order (FP64).
// Everything is fixed-order: a row's result never depends on other rows.
constexpr int kLaneRows = 8;                    // route rows per tile
constexpr int kDraftLaneRows = 16;              // the drafter's (DraftLanes)
constexpr int64_t kLaneMaxAssignments = 1024;   // routed rows the workspace layout covers
constexpr int kLaneThreads = 256;
constexpr int kLaneAhead = 3;                   // k steps prefetched to L2

struct LaneProblem {
  const int* table;
  int64_t assignments;       // routing table capacity
  int k_tiles;               // K / 16
  int n_tiles;               // N / 16
  int projections;           // 2: gate (out0) and up (out1); 1: down (out0)
  int first_projection;
  float* out0;               // [slice][assignment][N] partials
  float* out1;
  int64_t slice_stride;
  // B operand: per (assignment, projection) row [hi | lo] FP16 rows of K
  // halves in lane order, and 2^-e per (assignment, projection, K slice).
  const __half* halves;
  const float* unscale;
};

__device__ __forceinline__ void mma_f16_f32(const uint32_t (&a)[4], uint2 b, float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.0f));
}
// C-accumulating form: d = a b + d (the tensor core's own, truncating, add).
__device__ __forceinline__ void mma_f16_f32_acc(const uint32_t (&a)[4], uint2 b, float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}

// One warp: kNT output n16 tiles x kKT k16 steps of one route tile with
// kGroups n8 row groups (row_of: lane r holds tile row r's assignment), k
// outer: the kNT tiles of a k step are adjacent in the trellis (immediate
// offsets from three per-lane word pointers); the next k step's tiles and B
// fragments are in flight while this one decodes.
// kDraft (the drafter's DraftLanes): kGroups 2 (5..8 rows) and 4 (9..16 rows)
// split hi and lo per pair of groups: groups 2q and 2q + 1 hold the high and
// low parts of rows 8q..8q+7; high sums in FP32 outside the MMA, low sums in
// the MMA's C operand (lo is ~2^-11 of hi, so the truncating add is far below
// the FP32 rounding of the result); a row gets the same arithmetic in either.
// bits_x2 8 decodes from two ring words (WordWindow).
template <int kGroups, int kNT, int kKT, int kBitsX2, bool kAligned, bool kDraft = false>
__device__ __forceinline__ void lane_tiles(const LaneProblem& p, int pi, int slice, int row_of,
                                           int rows, const uint16_t* trellis, int64_t tiles_n,
                                           int n0, int lane, const LaneWindow<kBitsX2>& window,
                                           float* out) {
  using Ring = TileRing<kBitsX2>;
  constexpr int kTileWords = Ring::kTileHalfwords / 2;
  const int kt0 = slice * kKT;
  const int slices = p.k_tiles / kKT;
  const int64_t row_halves = 2 * int64_t(p.k_tiles) * 16;
  const int t = lane & 3;
  const int gq = lane >> 2;
  constexpr bool kSplit = kDraft && (kGroups == 2 || kGroups == 4);
  // B source of this lane: column gq of each group = hi (gq < 4) or lo of
  // row 4g + gq % 4 (split: row 8 (g / 2) + gq, part g % 2), k values 2t,
  // 2t + 1, 2t + 8, 2t + 9 of each k step.
  const uint2* bsrc[kGroups];
  bool bok[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    const int r = kSplit ? 8 * (g / 2) + gq : 4 * g + (gq & 3);
    const int part = kSplit ? g % 2 : (gq >> 2);
    bok[g] = r < rows;
    const int64_t assignment = __shfl_sync(0xffffffffu, row_of, bok[g] ? r : 0);
    bsrc[g] = reinterpret_cast<const uint2*>(
                  p.halves + (assignment * p.projections + pi) * row_halves +
                  part * (row_halves / 2) + kt0 * 16) + t;
  }
  // 2^-e of the rows this lane stores: 4g + 2t + c (lanes t < 2); split:
  // 8 (g / 2) + 2t + c (entries of odd g unused).
  float unscale[kGroups][2];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = kSplit ? 8 * (g / 2) + 2 * t + c : 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
      unscale[g][c] = kSplit && g % 2 == 1
                          ? 0.0f
                          : __ldg(p.unscale + (assignment * p.projections + pi) * slices + slice);
    }
  }
  const uint16_t* step = trellis + (int64_t(kt0) * tiles_n + n0) * Ring::kTileHalfwords;
  const int64_t stride = tiles_n * Ring::kTileHalfwords;
  constexpr bool kWordWindow = kDraft && WordWindow<kBitsX2>::kValue;
  constexpr int kUsedWords = kWordWindow ? 2 : Ring::kLoadWords;
  const uint32_t* wp[kUsedWords];
#pragma unroll
  for (int i = 0; i < kUsedWords; ++i) {
    wp[i] = reinterpret_cast<const uint32_t*>(step) + window.word[i];
  }
  uint32_t words[kNT][Ring::kLoadWords];
  auto fetch = [&](int j) {
    if constexpr (kAligned) {
#pragma unroll
      for (int i = 0; i < kUsedWords; ++i) words[j][i] = __ldg(wp[i] + j * kTileWords);
    } else if constexpr (kWordWindow) {
#pragma unroll
      for (int i = 0; i < kUsedWords; ++i) {
        words[j][i] = ring_word(step + j * Ring::kTileHalfwords, window.word[i], false);
      }
    } else {
      load_lane_words<kBitsX2>(step + j * Ring::kTileHalfwords, window, false, words[j]);
    }
  };
#pragma unroll
  for (int j = 0; j < kNT; ++j) fetch(j);
  uint2 bnext[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) bnext[g] = bok[g] ? __ldg(bsrc[g]) : make_uint2(0u, 0u);
  float acc[kNT][kGroups][4];
#pragma unroll
  for (int j = 0; j < kNT; ++j) {
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[j][g][i] = 0.0f;
    }
  }
  // L2 prefetch kLaneAhead k steps ahead: lane = 128-byte line of the step.
  constexpr int kStepBytes = kNT * Ring::kTileHalfwords * 2;
  const char* ahead = reinterpret_cast<const char*>(step + kLaneAhead * stride) +
                      min(lane * 128, kStepBytes - 1);
#pragma unroll 1
  for (int kt = 0; kt < kKT; ++kt) {
    if (lane * 128 < kStepBytes + 127 && kt + kLaneAhead < kKT) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(ahead));
    }
    ahead += stride * 2;
    const bool more = kt + 1 < kKT;
    uint2 b[kGroups];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      b[g] = bnext[g];
      if (more && bok[g]) bnext[g] = __ldg(bsrc[g] + 4 * (kt + 1));
    }
    if constexpr (kAligned) {
#pragma unroll
      for (int i = 0; i < kUsedWords; ++i) wp[i] += stride / 2;
    } else {
      step += stride;
    }
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
      uint32_t a[4];
      if constexpr (kWordWindow) {
        decode_lane_fragment_mul1_w32(words[j][0], words[j][1], a);
      } else {
        decode_lane_fragment_mul1<kBitsX2>(words[j], window.shift, a);
      }
      if (more) fetch(j);
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        if (kSplit && g % 2 == 1) {
          mma_f16_f32_acc(a, b[g], acc[j][g]);
        } else {
          float d[4];
          mma_f16_f32(a, b[g], d);
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[j][g][i] = __fadd_rn(acc[j][g][i], d[i]);
        }
      }
    }
  }
  // Lanes t < 2 hold the high sums of rows 4g + 2t + {0, 1}; lane t + 2 the
  // low sums of the same rows and columns (split: every lane holds both sums
  // of rows 8q + 2t + {0, 1}).
  const int64_t n_stride = int64_t(p.n_tiles) * 16;
  if constexpr (kSplit) {
#pragma unroll
    for (int q = 0; q < kGroups / 2; ++q) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = 8 * q + 2 * t + c;
        const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
#pragma unroll
        for (int j = 0; j < kNT; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int i = 2 * h + c;
            if (r < rows) {
              out[assignment * n_stride + (n0 + j) * 16 + gq + 8 * h] =
                  __fmul_rn(__fadd_rn(acc[j][2 * q][i], acc[j][2 * q + 1][i]), unscale[2 * q][c]);
            }
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = 4 * g + 2 * (t & 1) + c;
        const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
#pragma unroll
        for (int j = 0; j < kNT; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int i = 2 * h + c;
            const float low = __shfl_xor_sync(0xffffffffu, acc[j][g][i], 2);
            if (t < 2 && r < rows) {
              out[assignment * n_stride + (n0 + j) * 16 + gq + 8 * h] =
                  __fmul_rn(__fadd_rn(acc[j][g][i], low), unscale[g][c]);
            }
          }
        }
      }
    }
  }
}

// The unaligned-trellis variant, out of line: inlining it beside the aligned
// one costs the hot loops about 3 % (register allocation and code layout).
template <int kGroups, int kNT, int kKT, int kBitsX2, bool kDraft = false>
__device__ __noinline__ void lane_tiles_unaligned(LaneProblem p, int pi, int slice, int row_of,
                                                  int rows, const uint16_t* trellis,
                                                  int64_t tiles_n, int n0, int lane,
                                                  float* out) {
  const LaneWindow<kBitsX2> window(lane);
  lane_tiles<kGroups, kNT, kKT, kBitsX2, false, kDraft>(p, pi, slice, row_of, rows, trellis,
                                                        tiles_n, n0, lane, window, out);
}

// ---------------------------------------------------------------------------
// The folded call (moe_out_v2_fold): three launches instead of five. The
// activation runs at the end of the gate/up lane launch and the combine at
// the end of the down lane launch, each started by the warp whose item
// completes its inputs (completion counters); the arithmetic and its order
// are the unfolded stages' (activation_block, activate_down_row's row peak
// and store_scaled_halves, combine_routes' sums), so the output is the same
// bits.
//   - Gate/up: an item counts once per H128 block of the intermediate its
//     column group overlaps, per row tile. The warp that completes (tile,
//     block) runs activation_block for the tile's rows there and stores the
//     FP32 values and the warp peak; the warp that activates a tile's last
//     block takes each row's peak over its blocks, stores the down operand
//     (store_scaled_halves) and its 2^-e.
//   - Down: an item counts once per (token of each of its rows, H128 block of
//     the hidden size it overlaps). The warp that completes (token, block)
//     sums the token's local routes there in route order and writes the
//     BF16 output.
// Counters are zeroed by the route launch, which also counts each token's
// local routes and writes the rows without one as exact zero (no combine
// runs for them). Producers fence their stores before counting; the
// completing warp fences and reads the other warps' values from L2 (__ldcg).
// The gate/up counters, values and peaks live in the down partials' region
// (unused until the down launch); the token counters after the routing table
// (fold_ints).
struct FoldArgs {
  Routing routing;
  int* tile_counters;       // [tiles][row blocks] items, then [tiles] blocks activated
  int64_t tile_counter_ints;
  float* values;            // [assignment][ipad] activation values, FP32
  float* peaks;             // [assignment][row blocks] their warp peaks
  int* token_counters;      // [token][hpad / 128] down items
  int64_t token_counter_ints;
  int* token_routes;        // [token] local routes
  DownHalves down;
  int down_slices;
  int64_t intermediate;
  float limit;
  const float* weights;
  __nv_bfloat16* out;
  int64_t rows;
  int64_t hidden;
};

// Token counters and local-route counts after the routing table.
__host__ __device__ constexpr int64_t fold_ints(int64_t rows, int64_t hpad) {
  return rows * (hpad / 128 + 1);
}

// Column groups of `group_tiles` n16 tiles that overlap H128 block `block`
// (n16 tiles 8 block .. 8 block + 7) of an n_tiles-wide output.
__device__ __forceinline__ int groups_over_block(int block, int group_tiles, int n_tiles) {
  return min(8 * block + 7, n_tiles - 1) / group_tiles - 8 * block / group_tiles + 1;
}

// Fence-fence synchronization through the relaxed counter atomics (PTX
// memory model): acq_rel at GPU scope, not __threadfence's sequentially
// consistent fence, which the pattern does not need.
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
// Every lane's stores visible before a lane counts the warp's item: the
// barrier orders the other lanes' stores before the counting lane's fence,
// which then releases them cumulatively.
__device__ __forceinline__ void release_item() {
  __syncwarp();
  fence_acq_rel_gpu();
}
// After a lane saw the last count: the other warps' stores, for every lane.
__device__ __forceinline__ void acquire_item() {
  __syncwarp();
  fence_acq_rel_gpu();
  __syncwarp();
}

// The route launch's part (blocks 1..): zero the counters, count each
// token's local routes, write the rows without one as exact zero.
__device__ __forceinline__ void fold_route_prologue(const Routing& routing, const FoldArgs& f) {
  const int64_t thread = int64_t(blockIdx.x - 1) * blockDim.x + threadIdx.x;
  const int64_t threads = int64_t(gridDim.x - 1) * blockDim.x;
  for (int64_t i = thread; i < f.tile_counter_ints; i += threads) f.tile_counters[i] = 0;
  for (int64_t i = thread; i < f.token_counter_ints; i += threads) f.token_counters[i] = 0;
  const int lane = threadIdx.x & 31;
  for (int64_t token = thread >> 5; token < f.rows; token += threads >> 5) {
    int local = 0;
    for (int64_t r = lane; r < routing.topk; r += 32) {
      local += local_expert(routing, token * routing.topk + r) >= 0;
    }
    local = __reduce_add_sync(0xffffffffu, local);
    if (lane == 0) f.token_routes[token] = local;
    if (local == 0) {
      for (int64_t col = lane; col < f.hidden; col += 32) {
        f.out[token * f.hidden + col] = __float2bfloat16_rn(0.0f);
      }
    }
  }
}

// The down operand of a row tile whose blocks are all activated:
// activate_down_row's row peak, 2^e, halves and 2^-e copies. The values load
// kChunk blocks at a time (registers). Out of line.
template <int kRowBlocks>
__device__ __noinline__ void fold_down_operand(LaneProblem p, FoldArgs f, int4 tile, int lane) {
  constexpr int kChunk = kRowBlocks < 6 ? kRowBlocks : 6;
  const int64_t ipad = int64_t(p.n_tiles) * 16;
  const int* sorted = p.table + RouteLayout::sorted_offset();
  for (int r = 0; r < tile.z; ++r) {
    const int64_t assignment = __ldg(sorted + tile.y + r);
    float row_peak = 0.0f;
#pragma unroll
    for (int w = 0; w < kRowBlocks; ++w) {
      row_peak = fmaxf(row_peak, __ldcg(f.peaks + assignment * kRowBlocks + w));
    }
    const int e = halves_exponent(row_peak);
    __half* hi = f.down.halves + assignment * 2 * ipad;
#pragma unroll 1
    for (int b0 = 0; b0 < kRowBlocks; b0 += kChunk) {
      float v[kChunk][4];
#pragma unroll
      for (int b = 0; b < kChunk; ++b) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          v[b][i] = b0 + b < kRowBlocks
                        ? __ldcg(f.values + assignment * ipad + (b0 + b) * 128 + lane + 32 * i)
                        : 0.0f;
        }
      }
#pragma unroll
      for (int b = 0; b < kChunk; ++b) {
        if (b0 + b < kRowBlocks) store_scaled_halves(v[b], lane, (b0 + b) * 128, e, hi, hi + ipad);
      }
    }
    if (lane < f.down_slices) {
      f.down.unscale[assignment * f.down_slices + lane] = __uint_as_float(uint32_t(127 - e) << 23);
    }
  }
}

// The completed (tile, block) pairs of `ready` (bit j: block b0 + j):
// activate_down_row's per-block work for every row of the tile; the warp
// that activates the tile's last block also stores its down operand. Out of
// line, like lane_tiles_unaligned, to keep the lane loop's registers.
template <int kSlices, int kRowBlocks>
__device__ __noinline__ void fold_activate(LaneProblem p, FoldArgs f, int64_t tile_index,
                                           int4 tile, int b0, unsigned ready, int lane) {
  acquire_item();
  const int64_t ipad = int64_t(p.n_tiles) * 16;
  int* blocks_done = f.tile_counters + p.assignments * kRowBlocks;
  while (ready != 0) {
    const int block = b0 + __ffs(ready) - 1;
    ready &= ready - 1;
    for (int r = 0; r < tile.z; ++r) {
      double v[4];
      int64_t assignment;
      activation_block<kSlices, true>(f.routing, p.table, p.assignments, f.intermediate, ipad,
                                      p.out0, p.out1, p.slice_stride, tile.y + r, block, lane,
                                      f.limit, v, assignment);
      float values[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) values[i] = __double2float_rn(v[i]);
      const float peak = warp_peak(values);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        f.values[assignment * ipad + block * 128 + lane + 32 * i] = values[i];
      }
      if (lane == 0) f.peaks[assignment * kRowBlocks + block] = peak;
    }
    release_item();
    int last = 0;
    if (lane == 0) last = atomicAdd(blocks_done + tile_index, 1) == kRowBlocks - 1;
    if (__shfl_sync(0xffffffffu, last, 0)) {
      acquire_item();
      fold_down_operand<kRowBlocks>(p, f, tile, lane);
    }
  }
}

// combine_routes for the completed (token, block) pairs: lane l of `lanes`
// holds the token of its row (row_of / topk) and in `mine` its blocks (bit j:
// block b0 + j). Out of line.
template <int kSlices>
__device__ __noinline__ void fold_combine(LaneProblem p, FoldArgs f, unsigned lanes,
                                          unsigned mine, int row_of, int b0, int lane) {
  acquire_item();
  const int64_t hpad = int64_t(p.n_tiles) * 16;
  const int64_t topk = f.routing.topk;
  const int* by_assignment = p.table + RouteLayout::by_assignment_offset(p.assignments);
  while (lanes != 0) {
    const int source = __ffs(lanes) - 1;
    lanes &= lanes - 1;
    const unsigned blocks = __shfl_sync(0xffffffffu, mine, source);
    const int64_t token = __shfl_sync(0xffffffffu, row_of, source) / topk;
    for (unsigned bits = blocks; bits != 0; bits &= bits - 1) {
      const int64_t col0 = int64_t(b0 + __ffs(bits) - 1) * 128;
      double sum[4] = {0.0, 0.0, 0.0, 0.0};
      for (int64_t r = 0; r < topk; ++r) {
        const int64_t assignment = token * topk + r;
        const int64_t expert = __ldg(by_assignment + assignment);
        if (expert < 0) continue;
        const half* svh = expert_svh(f.routing, expert, 2);
        float v[kSlices][4];
#pragma unroll
        for (int s = 0; s < kSlices; ++s) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            v[s][i] = __ldcg(p.out0 + s * p.slice_stride + assignment * hpad + col0 + lane +
                             32 * i);
          }
        }
        float scale[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) scale[i] = __half2float(svh[col0 + lane + 32 * i]);
        double x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          x[i] = v[0][i];
#pragma unroll
          for (int s = 1; s < kSlices; ++s) x[i] = __dadd_rn(x[i], double(v[s][i]));
        }
        hadamard128_warp_f64(x, lane);
        const double weight = double(f.weights[assignment]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sum[i] = __fma_rn(__dmul_rn(x[i], double(scale[i])), weight, sum[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = col0 + lane + 32 * i;
        if (col < f.hidden) {
          f.out[token * f.hidden + col] = __float2bfloat16_rn(__double2float_rn(sum[i]));
        }
      }
    }
  }
}

// The lane kernel's per-item end: nothing (unfolded), or the gate/up and
// down launches' completion counting.
struct NoFold {
  template <int kNT>
  __device__ __forceinline__ static void item_done(const LaneProblem&, const FoldArgs&, int64_t,
                                                   int4, int, int, int, int, int) {}
};

template <int kSlices, int kRowBlocks>
struct FoldGateUp {
  template <int kNT>
  __device__ __forceinline__ static void item_done(const LaneProblem& p, const FoldArgs& f,
                                                   int64_t tile_index, int4 tile, int n0,
                                                   int row_of, int rows, int slices, int lane) {
    release_item();
    const int b0 = n0 / 8;
    bool last = false;
    if (lane <= (n0 + kNT - 1) / 8 - b0) {
      const int block = b0 + lane;
      const int expected = p.projections * slices * groups_over_block(block, kNT, p.n_tiles);
      last = atomicAdd(f.tile_counters + tile_index * kRowBlocks + block, 1) == expected - 1;
    }
    const unsigned ready = __ballot_sync(0xffffffffu, last);
    if (ready != 0) fold_activate<kSlices, kRowBlocks>(p, f, tile_index, tile, b0, ready, lane);
  }
};

template <int kSlices>
struct FoldDown {
  template <int kNT>
  __device__ __forceinline__ static void item_done(const LaneProblem& p, const FoldArgs& f,
                                                   int64_t, int4, int n0, int row_of, int rows,
                                                   int slices, int lane) {
    release_item();
    const int blocks = p.n_tiles / 8;
    const int b0 = n0 / 8;
    unsigned mine = 0;
    if (lane < rows) {
      const int64_t token = row_of / f.routing.topk;
      const int routes = f.token_routes[token];
      for (int j = 0; j <= (n0 + kNT - 1) / 8 - b0; ++j) {
        const int expected = routes * slices * groups_over_block(b0 + j, kNT, p.n_tiles);
        if (atomicAdd(f.token_counters + token * blocks + b0 + j, 1) == expected - 1) {
          mine |= 1u << j;
        }
      }
    }
    const unsigned lanes = __ballot_sync(0xffffffffu, mine != 0);
    if (lanes != 0) fold_combine<kSlices>(p, f, lanes, mine, row_of, b0, lane);
  }
};

// The drafter's lane kernels (run_lanes with DraftShape): row tiles of up to
// kDraftLaneRows rows, nothing after an item. A tile of 1..4 rows takes one
// n8 row group, of 5..8 rows a hi/lo pair of groups, of 9..16 two pairs
// (lane_tiles kDraft), the 1..4-row group with the low part summed outside
// the MMA. kMinRows 8 or 16 (VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS) gives every
// tile at least the kernel of that many rows, so a row's result no longer
// depends on how many rows of the call share its expert.
template <int kMinRows>
struct DraftLanes {
  template <int kNT>
  __device__ __forceinline__ static void item_done(const LaneProblem&, const FoldArgs&, int64_t,
                                                   int4, int, int, int, int, int) {}
};
// DraftLanes' kMinRows; 0 for the other lane kernels.
template <class Fold>
struct DraftMinRows : std::integral_constant<int, 0> {};
template <int kMinRows>
struct DraftMinRows<DraftLanes<kMinRows>> : std::integral_constant<int, kMinRows> {};

// Warp items: (route tile, projection, K slice, column group of kNT n16
// tiles), column groups fastest so a block's warps share B rows in L1.
template <int kNT, int kKT, int kBitsX2, class Fold = NoFold>
__global__ void __launch_bounds__(kLaneThreads, 2)
moe_lane_kernel(LaneProblem p, FoldArgs fold) {
  const int lane = threadIdx.x & 31;
  const LaneWindow<kBitsX2> window(lane);
  const int slices = p.k_tiles / kKT;
  const int groups_n = p.n_tiles / kNT;
  const int64_t per_tile = int64_t(p.projections) * slices * groups_n;
  const int64_t items = int64_t(p.table[1]) * per_tile;
  const int* sorted = p.table + RouteLayout::sorted_offset();
  const int4* tiles = reinterpret_cast<const int4*>(p.table + RouteLayout::tiles_offset(p.assignments));
  const long long* tile_trellis = reinterpret_cast<const long long*>(
      p.table + RouteLayout::tile_trellis_offset(p.assignments));
  const int* tile_columns = p.table + RouteLayout::tile_columns_offset(p.assignments);
  const int64_t warps = int64_t(gridDim.x) * (kLaneThreads / 32);
  for (int64_t item = int64_t(blockIdx.x) * (kLaneThreads / 32) + (threadIdx.x >> 5);
       item < items; item += warps) {
    const int64_t tile_index = item / per_tile;
    int rest = static_cast<int>(item % per_tile);
    const int column_group = rest % groups_n;
    rest /= groups_n;
    const int slice = rest % slices;
    const int pi = rest / slices;
    const int projection = p.first_projection + pi;
    // One round trip for the tile, its trellis address and columns (written
    // by the route kernel, which also checked the format).
    const int4 tile = tiles[tile_index];
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>(
        __ldg(tile_trellis + tile_index * 3 + projection));
    const int64_t tiles_n = __ldg(tile_columns + tile_index * 3 + projection);
    const int rows = tile.z;
    const int row_of = lane < rows ? sorted[tile.y + lane] : 0;
    const int n0 = column_group * kNT;
    const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & 3) == 0;
    float* out = (pi == 0 ? p.out0 : p.out1) + slice * p.slice_stride;
    constexpr int kDraftMin = DraftMinRows<Fold>::value;
    constexpr bool kDraft = kDraftMin > 0;
#define EXL3_LANE_GROUPS(G)                                                                     \
    if (aligned) {                                                                              \
      lane_tiles<G, kNT, kKT, kBitsX2, true, kDraft>(p, pi, slice, row_of, rows, trellis,       \
                                                     tiles_n, n0, lane, window, out);           \
    } else {                                                                                    \
      lane_tiles_unaligned<G, kNT, kKT, kBitsX2, kDraft>(p, pi, slice, row_of, rows, trellis,   \
                                                         tiles_n, n0, lane, out);               \
    }
    if constexpr (!kDraft) {
      if (rows <= 4) {
        EXL3_LANE_GROUPS(1);
      } else {
        EXL3_LANE_GROUPS(2);
      }
    } else if (kDraftMin <= 4 && rows <= 4) {
      EXL3_LANE_GROUPS(1);
    } else if (kDraftMin <= 8 && rows <= 8) {
      EXL3_LANE_GROUPS(2);
    } else {
      // 9..16 rows: four row groups (two hi/lo pairs) decoded once, half a
      // column group at a time (the accumulators of a kGroups 2 call).
      constexpr int kHalf = kNT / 2;
      if (aligned) {
        lane_tiles<4, kHalf, kKT, kBitsX2, true, true>(p, pi, slice, row_of, rows, trellis,
                                                       tiles_n, n0, lane, window, out);
        lane_tiles<4, kNT - kHalf, kKT, kBitsX2, true, true>(
            p, pi, slice, row_of, rows, trellis, tiles_n, n0 + kHalf, lane, window, out);
      } else {
        lane_tiles_unaligned<4, kHalf, kKT, kBitsX2, true>(p, pi, slice, row_of, rows, trellis,
                                                           tiles_n, n0, lane, out);
        lane_tiles_unaligned<4, kNT - kHalf, kKT, kBitsX2, true>(
            p, pi, slice, row_of, rows, trellis, tiles_n, n0 + kHalf, lane, out);
      }
    }
#undef EXL3_LANE_GROUPS
    Fold::template item_done<kNT>(p, fold, tile_index, tile, n0, row_of, rows, slices, lane);
  }
}

// One launch: block 0 routes (route_block) while the other blocks form the
// gate/up tensor-core B operand straight from topk_ids (independent of the
// routing table). One thread block per local assignment, each warp
// kBlocksPerWarp consecutive H128 blocks of the row (kRouteInputThreads / 32 x
// kBlocksPerWarp = hpad / 128), both projections: x * suh and H128 in FP64,
// rounded once to FP32; the 2^e of a (projection, kScaleK-column group) comes
// from the peak over the group's H128 blocks (shared memory), then the halves
// are stored by store_scaled_halves. A group spans kSplit GEMM K slices, each
// given its own copy of 2^-e. One block per warp keeps its own body: the same
// arithmetic in the instruction order the hidden-2560 shape was tuned with.
// kFold: the folded call's prologue first (fold_route_prologue).
template <int kBitsX2, int kBlocksPerWarp, int kScaleK, int kSplit, bool kFold = false>
__global__ void __launch_bounds__(kRouteInputThreads)
route_and_gate_up_input(Routing routing, int64_t assignments, int block_rows, int* table,
                        const __nv_bfloat16* x, int64_t hidden, int64_t hpad, __half* halves,
                        float* unscale, FoldArgs fold) {
  if (blockIdx.x == 0) {
    route_block<kBitsX2>(routing, assignments, block_rows, table);
    return;
  }
  if constexpr (kFold) fold_route_prologue(routing, fold);
  constexpr int kBlocks = kScaleK / 128;
  constexpr int kWarps = kRouteInputThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t slices = hpad / kScaleK * kSplit;
  if constexpr (kBlocksPerWarp == 1) {
    static_assert(kWarps % kBlocks == 0 && kSplit <= 32, "input warps must tile whole groups");
    __shared__ float warp_peaks[2][kWarps];
    const int group = warp / kBlocks;
    // Uniform trip count per thread block (barriers inside).
    for (int64_t assignment = blockIdx.x - 1; assignment < assignments;
         assignment += gridDim.x - 1) {
      const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
      float xs[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = int64_t(warp) * 128 + lane + 32 * i;
        xs[i] = col < hidden ? __bfloat162float(row[col]) : 0.0f;
      }
      const int64_t expert = local_expert(routing, assignment);
      float v[2][4] = {};
      if (expert >= 0) {
        const half* suh[2] = {expert_suh(routing, expert, 0), expert_suh(routing, expert, 1)};
        float scale[2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int64_t col = int64_t(warp) * 128 + lane + 32 * i;
            scale[q][i] = col < hidden ? __half2float(suh[q][col]) : 0.0f;
          }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          double block[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) block[i] = __dmul_rn(xs[i], scale[q][i]);
          hadamard128_warp_f64(block, lane);
#pragma unroll
          for (int i = 0; i < 4; ++i) v[q][i] = __double2float_rn(block[i]);
        }
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const float peak = warp_peak(v[q]);
        if (lane == 0) warp_peaks[q][warp] = peak;
      }
      __syncthreads();
      if (expert >= 0 && warp < hpad / 128) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          float group_peak = 0.0f;
#pragma unroll
          for (int w = 0; w < kBlocks; ++w) {
            group_peak = fmaxf(group_peak, warp_peaks[q][group * kBlocks + w]);
          }
          const int e = halves_exponent(group_peak);
          __half* hi = halves + (assignment * 2 + q) * 2 * hpad + int64_t(group) * kScaleK;
          store_scaled_halves(v[q], lane, (warp % kBlocks) * 128, e, hi, hi + hpad);
          if (warp % kBlocks == 0 && lane < kSplit) {
            unscale[(assignment * 2 + q) * slices + group * kSplit + lane] =
                __uint_as_float(uint32_t(127 - e) << 23);
          }
        }
      }
      __syncthreads();
    }
  } else {
    constexpr int kWarpsPerGroup = kBlocks / kBlocksPerWarp;
    static_assert(kBlocks % kBlocksPerWarp == 0 && kWarps % kWarpsPerGroup == 0 && kSplit <= 32,
                  "input warps must tile whole groups");
    __shared__ float block_peaks[2][kWarps][kBlocksPerWarp];
    const int group = warp / kWarpsPerGroup;
    const int member = warp % kWarpsPerGroup;
    // Uniform trip count per thread block (barriers inside).
    for (int64_t assignment = blockIdx.x - 1; assignment < assignments;
         assignment += gridDim.x - 1) {
      const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
      const int64_t expert = local_expert(routing, assignment);
      float v[2][kBlocksPerWarp][4] = {};
#pragma unroll
      for (int b = 0; b < kBlocksPerWarp; ++b) {
        const int64_t first = (int64_t(warp) * kBlocksPerWarp + b) * 128;
        float xs[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int64_t col = first + lane + 32 * i;
          xs[i] = col < hidden ? __bfloat162float(row[col]) : 0.0f;
        }
        if (expert >= 0) {
          const half* suh[2] = {expert_suh(routing, expert, 0), expert_suh(routing, expert, 1)};
          float scale[2][4];
#pragma unroll
          for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int64_t col = first + lane + 32 * i;
              scale[q][i] = col < hidden ? __half2float(suh[q][col]) : 0.0f;
            }
          }
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            double block[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) block[i] = __dmul_rn(xs[i], scale[q][i]);
            hadamard128_warp_f64(block, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[q][b][i] = __double2float_rn(block[i]);
          }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const float peak = warp_peak(v[q][b]);
          if (lane == 0) block_peaks[q][warp][b] = peak;
        }
      }
      __syncthreads();
      if (expert >= 0 && warp * kBlocksPerWarp < hpad / 128) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          float group_peak = 0.0f;
#pragma unroll
          for (int w = 0; w < kWarpsPerGroup; ++w) {
#pragma unroll
            for (int b = 0; b < kBlocksPerWarp; ++b) {
              group_peak = fmaxf(group_peak, block_peaks[q][group * kWarpsPerGroup + w][b]);
            }
          }
          const int e = halves_exponent(group_peak);
          __half* hi = halves + (assignment * 2 + q) * 2 * hpad + int64_t(group) * kScaleK;
#pragma unroll
          for (int b = 0; b < kBlocksPerWarp; ++b) {
            store_scaled_halves(v[q][b], lane, (member * kBlocksPerWarp + b) * 128, e, hi,
                                hi + hpad);
          }
          if (member == 0 && lane < kSplit) {
            unscale[(assignment * 2 + q) * slices + group * kSplit + lane] =
                __uint_as_float(uint32_t(127 - e) << 23);
          }
        }
      }
      __syncthreads();
    }
  }
}

// The drafter's route + gate/up operand launch: block 0 routes in increasing
// assignment order inside an expert (route_block kOrdered) while the other
// blocks form the gate/up tensor-core B operand straight from topk_ids. One
// thread block per (assignment, kScaleK-column scale group), one warp per
// H128 block of the group, both projections: x * suh and H128 in FP64,
// rounded once to FP32; the 2^e of a (projection, group) comes from the peak
// over the group's H128 blocks (shared memory), then the halves are stored by
// store_scaled_halves. A group spans kSplit GEMM K slices, each given its own
// copy of 2^-e. A row's groups spread over the SMs.
template <int kBitsX2, int kScaleK, int kSplit>
__global__ void __launch_bounds__(32 * (kScaleK / 128))
route_and_gate_up_group_input(Routing routing, int64_t assignments, int block_rows, int* table,
                              const __nv_bfloat16* x, int64_t hidden, int64_t hpad,
                              __half* halves, float* unscale) {
  if (blockIdx.x == 0) {
    route_block<kBitsX2, true>(routing, assignments, block_rows, table);
    return;
  }
  constexpr int kBlocks = kScaleK / 128;
  static_assert(kBlocks <= 32 && kSplit <= 32, "one warp per H128 block of a scale group");
  __shared__ float warp_peaks[2][kBlocks];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t groups = hpad / kScaleK;
  const int64_t slices = groups * kSplit;
  // Uniform trip count per thread block (barriers inside).
  for (int64_t task = blockIdx.x - 1; task < assignments * groups; task += gridDim.x - 1) {
    const int64_t assignment = task / groups;
    const int64_t group = task - assignment * groups;
    const int64_t first = group * kScaleK + int64_t(warp) * 128;
    const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
    float xs[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = first + lane + 32 * i;
      xs[i] = col < hidden ? __bfloat162float(row[col]) : 0.0f;
    }
    const int64_t expert = local_expert(routing, assignment);
    float v[2][4] = {};
    if (expert >= 0) {
      const half* suh[2] = {expert_suh(routing, expert, 0), expert_suh(routing, expert, 1)};
      float scale[2][4];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int64_t col = first + lane + 32 * i;
          scale[q][i] = col < hidden ? __half2float(suh[q][col]) : 0.0f;
        }
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        double block[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) block[i] = __dmul_rn(xs[i], scale[q][i]);
        hadamard128_warp_f64(block, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[q][i] = __double2float_rn(block[i]);
      }
    }
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const float peak = warp_peak(v[q]);
      if (lane == 0) warp_peaks[q][warp] = peak;
    }
    __syncthreads();
    if (expert >= 0) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        float group_peak = 0.0f;
#pragma unroll
        for (int w = 0; w < kBlocks; ++w) group_peak = fmaxf(group_peak, warp_peaks[q][w]);
        const int e = halves_exponent(group_peak);
        __half* hi = halves + (assignment * 2 + q) * 2 * hpad + group * kScaleK;
        store_scaled_halves(v[q], lane, warp * 128, e, hi, hi + hpad);
        if (warp == 0 && lane < kSplit) {
          unscale[(assignment * 2 + q) * slices + group * kSplit + lane] =
              __uint_as_float(uint32_t(127 - e) << 23);
        }
      }
    }
    __syncthreads();
  }
}

// The served shapes, by padded hidden and intermediate size. Items cover
// kColumnTiles n16 tiles. Gate/up: calls below kLongSliceAssignments routed
// assignments give each (row, projection) one 2^e over the whole K and split
// it into slices of kGateUpShortSliceTiles k16 steps; larger calls one 2^e per
// kGateUpLongScaleK columns and slices of kGateUpLongSliceTiles steps. The
// down operand has one 2^e per row: slices of kDownShortSliceTiles steps below
// kLongDownAssignments assignments, of kDownLongSliceTiles steps from there
// on. Bits lists the bits_x2 values the lane kernels are built for.
template <int64_t kHidden, int64_t kIntermediate>
struct LaneTiling;

// Hidden 2560, intermediate 640: gate/up 8 slices of 20 steps (4 of 40, one
// 2^e per five H128 blocks), down 2 slices of 20 (the whole K, one item per
// row tile and column group).
template <>
struct LaneTiling<2560, 640> {
  using Bits = std::integer_sequence<int, 10>;
  static constexpr int kColumnTiles = 5;
  static constexpr int kGateUpShortScaleK = 2560;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = 640;
  static constexpr int kGateUpLongSliceTiles = 40;
  static constexpr int kDownShortSliceTiles = 20;
  static constexpr int kDownLongSliceTiles = 40;
  static constexpr int64_t kLongSliceAssignments = 160;
  static constexpr int64_t kLongDownAssignments = 160;
};

// Hidden 5120, intermediate 2304 (whole experts per rank; at expert parallel
// size 4 a quarter of the routes is local): column groups of 4 n16 tiles
// divide 144 and 320; gate/up 16 slices of 20 steps, down 8 slices of 18. One
// slice layout for every call of up to 64 rows (384 assignments; the long
// parameters repeat the short ones and never apply), so a row's result does
// not depend on how many rows share the call.
template <>
struct LaneTiling<5120, 2304> {
  using Bits = std::integer_sequence<int, 6, 4>;
  static constexpr int kColumnTiles = 4;
  static constexpr int kGateUpShortScaleK = 5120;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = kGateUpShortScaleK;
  static constexpr int kGateUpLongSliceTiles = kGateUpShortSliceTiles;
  static constexpr int kDownShortSliceTiles = 18;
  static constexpr int kDownLongSliceTiles = kDownShortSliceTiles;
  static constexpr int64_t kLongSliceAssignments = std::numeric_limits<int64_t>::max();
  static constexpr int64_t kLongDownAssignments = std::numeric_limits<int64_t>::max();
};

// Hidden 5120, intermediate 640 (tensor-parallel routed experts: every expert
// local, this rank's 640-column chunk of the 2304 intermediate): gate/up as
// for intermediate 2304 (16 slices of 20 steps), down 2 slices of 20. One
// slice layout for every call of up to 64 rows. bits_x2 8: the DSpark
// drafter's experts split the same way (VLLM_EXL3_MOE_V2_DRAFT; these
// stages, not DraftTiling's).
template <>
struct LaneTiling<5120, 640> {
  using Bits = std::integer_sequence<int, 6, 4, 8>;
  static constexpr int kColumnTiles = 4;
  static constexpr int kGateUpShortScaleK = 5120;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = kGateUpShortScaleK;
  static constexpr int kGateUpLongSliceTiles = kGateUpShortSliceTiles;
  static constexpr int kDownShortSliceTiles = 20;
  static constexpr int kDownLongSliceTiles = kDownShortSliceTiles;
  static constexpr int64_t kLongSliceAssignments = std::numeric_limits<int64_t>::max();
  static constexpr int64_t kLongDownAssignments = std::numeric_limits<int64_t>::max();
};

// Hidden 5120, intermediate 512 (tensor-parallel routed experts, this rank's
// 512-column chunk): gate/up 16 slices of 20 steps, down 2 slices of 16. One
// slice layout for every call of up to 64 rows. bits_x2 8: as for 640.
template <>
struct LaneTiling<5120, 512> {
  using Bits = std::integer_sequence<int, 6, 4, 8>;
  static constexpr int kColumnTiles = 4;
  static constexpr int kGateUpShortScaleK = 5120;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = kGateUpShortScaleK;
  static constexpr int kGateUpLongSliceTiles = kGateUpShortSliceTiles;
  static constexpr int kDownShortSliceTiles = 16;
  static constexpr int kDownLongSliceTiles = kDownShortSliceTiles;
  static constexpr int64_t kLongSliceAssignments = std::numeric_limits<int64_t>::max();
  static constexpr int64_t kLongDownAssignments = std::numeric_limits<int64_t>::max();
};

// The DSpark drafter's routed experts (hidden 5120, intermediate 2304, bits_x2
// 8; VLLM_EXL3_MOE_V2_DRAFT): the hidden-5120 K slices (one layout for every
// call of up to 64 rows) with stages of their own: one gate/up 2^e per 640
// columns, five H128 blocks (route_and_gate_up_group_input: one thread block
// per (row, group)); one down 2^e per (row, K slice) (activate_down_slice);
// row tiles of up to 16 rows, the low part of 5..16-row tiles summed in the
// MMA's C operand (DraftLanes). A row's result depends on whether its tile
// holds 1..4 rows or more (how many rows of the call share its expert) unless
// VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS pins the tile kernel.
struct DraftTiling {
  using Bits = std::integer_sequence<int, 8>;
  static constexpr int kColumnTiles = 4;
  static constexpr int kGateUpShortScaleK = 640;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = kGateUpShortScaleK;
  static constexpr int kGateUpLongSliceTiles = kGateUpShortSliceTiles;
  static constexpr int kDownShortSliceTiles = 18;
  static constexpr int kDownLongSliceTiles = kDownShortSliceTiles;
  static constexpr int64_t kLongSliceAssignments = std::numeric_limits<int64_t>::max();
  static constexpr int64_t kLongDownAssignments = std::numeric_limits<int64_t>::max();
};

template <int64_t kH, int64_t kI, class kTiling = LaneTiling<kH, kI>>
struct LaneShape : kTiling {
  using Tiling = kTiling;
  // The drafter's stages (run_lanes).
  static constexpr bool kDraft = std::is_same_v<Tiling, DraftTiling>;
  static constexpr int64_t kHidden = kH;
  static constexpr int64_t kIntermediate = kI;
  static constexpr int kGateUpShortSlices = int(kH / (16 * Tiling::kGateUpShortSliceTiles));
  static constexpr int kGateUpLongSlices = int(kH / (16 * Tiling::kGateUpLongSliceTiles));
  static constexpr int kDownShortSlices = int(kI / (16 * Tiling::kDownShortSliceTiles));
  static constexpr int kDownLongSlices = int(kI / (16 * Tiling::kDownLongSliceTiles));
  // Workspace layout: room for the larger slice counts.
  static constexpr int kGateUpSlices = kGateUpShortSlices;
  static constexpr int kDownSlices = kDownShortSlices;
  // H128 blocks of a row per input warp, and the activation stage's warps.
  static constexpr int kInputBlocksPerWarp = int(kH / 128) / (kRouteInputThreads / 32);
  static constexpr int kRowBlocks = int(kI / 128);
  // The drafter's activation warps per down K slice: the H128 blocks a slice
  // of either width can overlap.
  static constexpr int kSliceBlocks =
      int((16 * std::max(Tiling::kDownShortSliceTiles, Tiling::kDownLongSliceTiles) + 127) /
          128) + 1;
  static_assert(kH % (16 * Tiling::kGateUpShortSliceTiles) == 0 &&
                    kH % (16 * Tiling::kGateUpLongSliceTiles) == 0 &&
                    kGateUpShortSlices >= kGateUpLongSlices &&
                    Tiling::kGateUpShortScaleK % (16 * Tiling::kGateUpShortSliceTiles) == 0 &&
                    Tiling::kGateUpLongScaleK % (16 * Tiling::kGateUpLongSliceTiles) == 0 &&
                    kH % Tiling::kGateUpShortScaleK == 0 && kH % Tiling::kGateUpLongScaleK == 0 &&
                    kI % (16 * Tiling::kDownShortSliceTiles) == 0 &&
                    kI % (16 * Tiling::kDownLongSliceTiles) == 0 &&
                    kDownShortSlices >= kDownLongSlices && kDownShortSlices <= 32 &&
                    kH % (16 * Tiling::kColumnTiles) == 0 &&
                    kI % (16 * Tiling::kColumnTiles) == 0,
                "column groups, K slices and scale groups must divide the shape; a "
                "gate/up scale group holds whole K slices");
  static_assert(kH % (128 * (kRouteInputThreads / 32)) == 0 && kI % 128 == 0 &&
                    kRowBlocks * 32 <= 1024,
                "the input warps cover whole H128 blocks of a row; one activation warp "
                "per H128 block of the intermediate");
  static_assert(!kDraft || (Tiling::kGateUpShortScaleK % 128 == 0 &&
                            Tiling::kGateUpShortScaleK / 128 <= 32 &&
                            Tiling::kGateUpLongScaleK == Tiling::kGateUpShortScaleK &&
                            kSliceBlocks * 32 <= 1024),
                "the drafter's input: one warp per H128 block of a gate/up scale group; one "
                "activation warp per H128 block of a down K slice");
};

using DraftShape = LaneShape<5120, 2304, DraftTiling>;
// Apart from its tiling the drafter's shape is the hidden-5120 one: its calls
// use that workspace (moe_workspace_size_v2; the route table holds fewer row
// tiles, of 16 rows, in the same layout) and row geometry.
static_assert(DraftShape::kGateUpSlices == LaneShape<5120, 2304>::kGateUpSlices &&
                  DraftShape::kDownSlices == LaneShape<5120, 2304>::kDownSlices &&
                  DraftShape::kInputBlocksPerWarp == LaneShape<5120, 2304>::kInputBlocksPerWarp &&
                  DraftShape::kRowBlocks == LaneShape<5120, 2304>::kRowBlocks,
              "the drafter's workspace layout and row geometry must be the hidden-5120 ones");

constexpr const char* kShapeMessage =
    "EXL3 MoE v2 supports padded hidden 2560 with intermediate 640 and padded hidden 5120 "
    "with intermediate 2304, 640 or 512 only";

// f(LaneShape<hpad, ipad>{}) for a served shape; false for any other.
template <class F>
bool with_shape(int64_t hpad, int64_t ipad, F&& f) {
  if (hpad == 2560 && ipad == 640) {
    f(LaneShape<2560, 640>{});
    return true;
  }
  if (hpad == 5120 && ipad == 2304) {
    f(LaneShape<5120, 2304>{});
    return true;
  }
  if (hpad == 5120 && ipad == 640) {
    f(LaneShape<5120, 640>{});
    return true;
  }
  if (hpad == 5120 && ipad == 512) {
    f(LaneShape<5120, 512>{});
    return true;
  }
  return false;
}

// f(std::integral_constant<int, bits_x2>{}) if bits_x2 is one of kBits.
template <int... kBits, class F>
bool with_bits(std::integer_sequence<int, kBits...>, int64_t bits_x2, F&& f) {
  return ((bits_x2 == kBits ? (f(std::integral_constant<int, kBits>{}), true) : false) || ...);
}

void check_assignments(int64_t assignments) {
  TORCH_CHECK(assignments <= kLaneMaxAssignments,
              "EXL3 MoE v2 supports at most 1024 routed assignments per call");
}

template <int kNT, int kKT, int kBitsX2, class Fold = NoFold>
void launch_lane(const LaneProblem& p, const FoldArgs& fold, int64_t max_tiles,
                 cudaStream_t stream) {
  auto* kernel = &moe_lane_kernel<kNT, kKT, kBitsX2, Fold>;
  static const int64_t resident = [kernel] {
    int device = 0;
    int sms = 0;
    int per_sm = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                                 kLaneThreads, 0));
    return int64_t(std::max(sms, 1)) * std::max(per_sm, 1);
  }();
  const int64_t per_tile = int64_t(p.projections) * (p.k_tiles / kKT) * (p.n_tiles / kNT);
  constexpr int64_t kWarps = kLaneThreads / 32;
  const int64_t grid = std::min((max_tiles * per_tile + kWarps - 1) / kWarps, resident);
  if (grid <= 0) return;
  kernel<<<static_cast<unsigned int>(grid), kLaneThreads, 0, stream>>>(p, fold);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FP32 workspace elements before the routing table: gate, up and down slice
// partials; the gate/up and down B operands (FP16 hi and lo rows) and their
// 2^-e per (row, slice), rounded up to whole 16-byte groups.
template <class S>
int64_t lane_workspace(int64_t assignments, int64_t hpad, int64_t ipad) {
  const int64_t gate_up = checked_mul(2 * S::kGateUpSlices, ipad);
  const int64_t down = checked_mul(S::kDownSlices, hpad);
  const int64_t operands = checked_add(checked_mul(2, hpad), ipad);   // FP16 hi + lo
  const int64_t unscale = (assignments * (2 * S::kGateUpSlices + S::kDownSlices) + 3) / 4 * 4;
  return checked_add(
      checked_mul(assignments, checked_add(checked_add(gate_up, down), operands)), unscale);
}

// One call's arguments, checked by moe_out_v2.
struct LaneCall {
  Routing routing;
  const __nv_bfloat16* x;
  const float* topk_weights;
  __nv_bfloat16* out;
  float* workspace;
  int64_t rows;
  int64_t hidden;
  int64_t hpad;
  int64_t ipad;
  int64_t intermediate;
  int64_t experts;
  int64_t assignments;
  float limit;
  bool fold;                 // moe_out_v2_fold: three launches (FoldArgs)
  int draft_min_rows;        // DraftShape: DraftLanes' kMinRows (4, 8 or 16)
  cudaStream_t stream;
};

template <class S, int kBitsX2>
void run_lanes(const LaneCall& c) {
  Routing routing = c.routing;
  const int64_t assignments = c.assignments;
  const int64_t hpad = c.hpad;
  const int64_t ipad = c.ipad;
  float* input = c.workspace;
  const float* down_out = input;
  int64_t down_stride = 0;
  const int* route_table = nullptr;   // the routing table combine_routes reads
  FoldArgs fold{};
  if (c.experts > 0 && assignments > 0) {
    const int64_t gate_up_stride = assignments * ipad;
    down_stride = assignments * hpad;
    float* gate = input;
    float* up = gate + S::kGateUpSlices * gate_up_stride;
    float* down = up + S::kGateUpSlices * gate_up_stride;
    // B operands: [assignment][gate, up][hi, lo][hpad] and [assignment][hi, lo][ipad]
    // FP16, then 2^-e per (assignment, projection, slice).
    __half* gate_up_halves = reinterpret_cast<__half*>(down + S::kDownSlices * down_stride);
    __half* down_halves = gate_up_halves + assignments * 2 * 2 * hpad;
    float* gate_up_unscale = reinterpret_cast<float*>(down_halves + assignments * 2 * ipad);
    float* down_unscale = gate_up_unscale + assignments * 2 * S::kGateUpSlices;
    // The int4 route tiles need a 16-byte aligned table.
    int* table = reinterpret_cast<int*>(
        gate_up_unscale + (assignments * (2 * S::kGateUpSlices + S::kDownSlices) + 3) / 4 * 4);
    route_table = table;
    down_out = down;
    // Down operand with one 2^e per row; long or short K slices.
    const bool long_down = assignments >= S::kLongDownAssignments;
    const int down_used = long_down ? S::kDownLongSlices : S::kDownShortSlices;
    // The drafter's calls (bits_x2 8, run_moe_out_v2 refuses the fold) build
    // no folded kernels.
    constexpr bool kFoldable = !S::kDraft && kBitsX2 != 8;
    if constexpr (kFoldable) {
      if (c.fold) {
        static_assert(S::kDownSlices * S::kHidden >= S::kIntermediate + 2 * S::kRowBlocks + 5,
                      "the folded gate/up stage keeps its counters, values and peaks in the "
                      "down partials' region");
        int* tile_counters = reinterpret_cast<int*>(down);
        const int64_t tile_counter_ints = assignments * (S::kRowBlocks + 1);
        float* values = down + (tile_counter_ints + 3) / 4 * 4;
        int* token_counters = table + RouteLayout::ints(assignments);
        const int64_t token_counter_ints = c.rows * (hpad / 128);
        fold = FoldArgs{routing, tile_counters, tile_counter_ints, values,
                        values + assignments * ipad, token_counters, token_counter_ints,
                        token_counters + token_counter_ints,
                        DownHalves{down_halves, down_unscale}, down_used, c.intermediate,
                        c.limit, c.topk_weights, c.out, c.rows, c.hidden};
      }
    }
    // The drafter's lane kernels: DraftLanes<c.draft_min_rows>.
    const auto draft_lane = [&](auto slice_tiles, const LaneProblem& problem,
                                int64_t tiles) {
      constexpr int kSliceTiles = decltype(slice_tiles)::value;
      if (c.draft_min_rows == 16) {
        launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2, DraftLanes<16>>(problem, fold, tiles,
                                                                          c.stream);
      } else if (c.draft_min_rows == 8) {
        launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2, DraftLanes<8>>(problem, fold, tiles,
                                                                         c.stream);
      } else {
        launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2, DraftLanes<4>>(problem, fold, tiles,
                                                                         c.stream);
      }
    };
    // Routing (block 0) and the gate/up B operand (other blocks) in one launch,
    // then the gate/up GEMM and the activation, with short or long K slices.
    const auto gate_up = [&](auto scale_k, auto slice_tiles) {
      constexpr int kScaleK = decltype(scale_k)::value;
      constexpr int kSplit = kScaleK / (16 * decltype(slice_tiles)::value);
      if constexpr (S::kDraft) {
        route_and_gate_up_group_input<kBitsX2, kScaleK, kSplit>
            <<<launch_blocks(1 + assignments * (hpad / kScaleK)), 32 * (kScaleK / 128),
               (3 * c.experts + 1 + assignments) * sizeof(int), c.stream>>>(
                routing, assignments, kDraftLaneRows, table, c.x, c.hidden, hpad,
                gate_up_halves, gate_up_unscale);
      } else {
        const auto launch = [&](auto folded) {
          route_and_gate_up_input<kBitsX2, S::kInputBlocksPerWarp, kScaleK, kSplit,
                                  decltype(folded)::value>
              <<<launch_blocks(1 + assignments),
                 kRouteInputThreads, (3 * c.experts + 1) * sizeof(int), c.stream>>>(
                  routing, assignments, kLaneRows, table, c.x, c.hidden, hpad, gate_up_halves,
                  gate_up_unscale, fold);
        };
        if constexpr (kFoldable) {
          if (c.fold) {
            launch(std::true_type{});
          } else {
            launch(std::false_type{});
          }
        } else {
          launch(std::false_type{});
        }
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    const bool long_slices = assignments >= S::kLongSliceAssignments;
    if (long_slices) {
      gate_up(std::integral_constant<int, S::kGateUpLongScaleK>{},
              std::integral_constant<int, S::kGateUpLongSliceTiles>{});
    } else {
      gate_up(std::integral_constant<int, S::kGateUpShortScaleK>{},
              std::integral_constant<int, S::kGateUpShortSliceTiles>{});
    }
    // Row tiles: sum over experts of ceil(rows_e / tile rows).
    constexpr int64_t kTileRows = S::kDraft ? kDraftLaneRows : kLaneRows;
    const int64_t max_tiles = std::min<int64_t>(
        assignments, assignments / kTileRows + std::min<int64_t>(c.experts, assignments) + 1);
    LaneProblem p{};
    p.table = table;
    p.assignments = assignments;
    p.k_tiles = static_cast<int>(hpad / 16);
    p.n_tiles = static_cast<int>(ipad / 16);
    p.projections = 2;
    p.first_projection = 0;
    p.out0 = gate;
    p.out1 = up;
    p.slice_stride = gate_up_stride;
    p.halves = gate_up_halves;
    p.unscale = gate_up_unscale;
    const auto activate = [&](auto slices) {
      if constexpr (S::kDraft) {
        activate_down_slice<decltype(slices)::value, S::kSliceBlocks>
            <<<launch_blocks(assignments * down_used), 32 * S::kSliceBlocks, 0, c.stream>>>(
                routing, table, assignments, c.intermediate, ipad, gate, up, gate_up_stride,
                DownHalves{down_halves, down_unscale}, down_used, c.limit);
      } else {
        activate_down_row<decltype(slices)::value, S::kRowBlocks>
            <<<launch_blocks(assignments), 32 * S::kRowBlocks, 0, c.stream>>>(
                routing, table, assignments, c.intermediate, ipad, gate, up, gate_up_stride,
                DownHalves{down_halves, down_unscale}, down_used, c.limit);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    // Folded: the activation at the end of the gate/up launch.
    const auto gate_up_lane = [&](auto slice_tiles, auto slices) {
      constexpr int kSliceTiles = decltype(slice_tiles)::value;
      constexpr int kSlices = decltype(slices)::value;
      if constexpr (S::kDraft) {
        draft_lane(std::integral_constant<int, kSliceTiles>{}, p, max_tiles);
        activate(std::integral_constant<int, kSlices>{});
      } else {
        if constexpr (kFoldable) {
          if (c.fold) {
            launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2,
                        FoldGateUp<kSlices, S::kRowBlocks>>(p, fold, max_tiles, c.stream);
            return;
          }
        }
        launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2>(p, fold, max_tiles, c.stream);
        activate(std::integral_constant<int, kSlices>{});
      }
    };
    if (long_slices) {
      gate_up_lane(std::integral_constant<int, S::kGateUpLongSliceTiles>{},
                   std::integral_constant<int, S::kGateUpLongSlices>{});
    } else {
      gate_up_lane(std::integral_constant<int, S::kGateUpShortSliceTiles>{},
                   std::integral_constant<int, S::kGateUpShortSlices>{});
    }
    p.k_tiles = static_cast<int>(ipad / 16);
    p.n_tiles = static_cast<int>(hpad / 16);
    p.projections = 1;
    p.first_projection = 2;
    p.out0 = down;
    p.out1 = nullptr;
    p.slice_stride = down_stride;
    p.halves = down_halves;
    p.unscale = down_unscale;
    // Folded: the combine at the end of the down launch.
    const auto down_lane = [&](auto slice_tiles, auto slices) {
      constexpr int kSliceTiles = decltype(slice_tiles)::value;
      if constexpr (S::kDraft) {
        draft_lane(std::integral_constant<int, kSliceTiles>{}, p, max_tiles);
      } else {
        if constexpr (kFoldable) {
          if (c.fold) {
            launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2,
                        FoldDown<decltype(slices)::value>>(p, fold, max_tiles, c.stream);
            return;
          }
        }
        launch_lane<S::kColumnTiles, kSliceTiles, kBitsX2>(p, fold, max_tiles, c.stream);
      }
    };
    if (long_down) {
      down_lane(std::integral_constant<int, S::kDownLongSliceTiles>{},
                std::integral_constant<int, S::kDownLongSlices>{});
    } else {
      down_lane(std::integral_constant<int, S::kDownShortSliceTiles>{},
                std::integral_constant<int, S::kDownShortSlices>{});
    }
    if (c.fold) return;
  } else {
    // No local expert or no route: every row is exactly zero.
    routing.topk = 0;
  }
  const size_t shared = combine_shared_bytes(routing.topk);
  TORCH_CHECK(shared <= 48 * 1024, "EXL3 MoE topk too large for the combine stage");
  const auto combine = [&](auto slices) {
    combine_routes<decltype(slices)::value>
        <<<launch_blocks(c.rows * (hpad / 128)),
           32 * static_cast<int>(std::min<int64_t>(std::max<int64_t>(routing.topk, 4), 32)),
           shared, c.stream>>>(down_out, c.topk_weights, routing, route_table, c.rows,
                               c.hidden, hpad, down_stride, c.out);
  };
  if (assignments >= S::kLongDownAssignments) {
    combine(std::integral_constant<int, S::kDownLongSlices>{});
  } else {
    combine(std::integral_constant<int, S::kDownSlices>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
}  // namespace moe_v2

int64_t moe_workspace_size_v2(int64_t max_rows, int64_t hidden_size,
                              int64_t intermediate_size, int64_t topk,
                              int64_t local_experts) {
  using namespace moe_v2;
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  int64_t lane = 0;
  TORCH_CHECK(with_shape(hpad, ipad,
                         [&](auto shape) {
                           check_assignments(assignments);
                           lane = lane_workspace<decltype(shape)>(assignments, hpad, ipad);
                         }),
              kShapeMessage);
  // The routing table, then the folded call's token counters (fold_ints).
  return workspace_elements(checked_add(
      lane, checked_add(RouteLayout::ints(assignments), fold_ints(max_rows, hpad))));
}

namespace {

// The drafter's DraftLanes<kMinRows> (moe_v2_draft_force_rows,
// VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS): 4 picks each tile's kernel by its rows;
// 8 or 16 gives every tile at least that tile's kernel. Set before the
// warm-up and graph capture.
std::atomic<int> draft_min_rows{4};

// moe_out_v2 (fold false) and moe_out_v2_fold: the same checks and stages;
// folded, the activation and combine run inside the lane launches.
void run_moe_out_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                    const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
                    const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
                    int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
                    torch::Tensor& workspace, double swiglu_limit, bool fold) {
  using namespace moe_v2;
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
  TORCH_CHECK(experts <= kMaxRouteExperts, "EXL3 MoE has too many local experts");
  const int64_t assignments = checked_mul(rows, topk);
  bool built = false;
  // The drafter's experts (DraftShape) take their own stages, never folded.
  const bool draft = hpad == DraftShape::kHidden && ipad == DraftShape::kIntermediate &&
                     with_bits(DraftShape::Bits{}, bits_x2, [](auto) {});
  TORCH_CHECK(with_shape(hpad, ipad,
                         [&](auto shape) {
                           check_assignments(assignments);
                           built = draft || with_bits(typename decltype(shape)::Bits{},
                                                      bits_x2, [](auto) {});
                         }),
              kShapeMessage);
  TORCH_CHECK(built, "EXL3 MoE v2 has no bits_x2 ", bits_x2, " kernels for padded hidden ",
              hpad, " and intermediate ", ipad);
  TORCH_CHECK(!(fold && bits_x2 == 8),
              "EXL3 MoE v2 has no folded bits_x2 8 kernels (the DSpark drafter's experts)");
  // The drafter's ordered routing keeps one more int per assignment in the
  // route block's dynamic shared memory.
  TORCH_CHECK(!draft || (3 * experts + 1 + assignments) * int64_t(sizeof(int)) <= 48 * 1024,
              "EXL3 MoE v2 bits_x2 8: too many local experts and assignments for the route "
              "block's shared memory");
  const int64_t required =
      moe_workspace_size_v2(rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(out, workspace,
                  {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const LaneCall call{
      Routing{topk_ids.data_ptr(), expert_map.data_ptr(), packed_ptrs.data_ptr<int64_t>(),
              packed_meta.data_ptr<int64_t>(), expert_map.numel(), experts, topk,
              topk_ids.scalar_type() == at::kLong, expert_map.scalar_type() == at::kLong},
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      topk_weights.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      workspace.data_ptr<float>(),
      rows,
      hidden,
      hpad,
      ipad,
      intermediate_size,
      experts,
      assignments,
      limit,
      fold,
      draft_min_rows.load(),
      stream};
  if (draft) {
    with_bits(DraftShape::Bits{}, bits_x2, [&](auto bits) {
      run_lanes<DraftShape, decltype(bits)::value>(call);
    });
  } else {
    with_shape(hpad, ipad, [&](auto shape) {
      with_bits(typename decltype(shape)::Bits{}, bits_x2, [&](auto bits) {
        run_lanes<decltype(shape), decltype(bits)::value>(call);
      });
    });
  }
}

}  // namespace

void moe_out_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                const torch::Tensor& topk_weights,
                const torch::Tensor& packed_ptrs,
                const torch::Tensor& packed_meta,
                const torch::Tensor& expert_map, int64_t intermediate_size,
                int64_t bits_x2, torch::Tensor& out, torch::Tensor& workspace,
                double swiglu_limit) {
  run_moe_out_v2(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
                 intermediate_size, bits_x2, out, workspace, swiglu_limit, false);
}

void moe_out_v2_fold(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     int64_t bits_x2, torch::Tensor& out, torch::Tensor& workspace,
                     double swiglu_limit) {
  run_moe_out_v2(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
                 intermediate_size, bits_x2, out, workspace, swiglu_limit, true);
}

void moe_v2_draft_force_rows(int64_t rows) {
  TORCH_CHECK(rows == 0 || rows == 8 || rows == 16,
              "EXL3 MoE v2 drafter tile rows to force must be 0 (none), 8 or 16, not ", rows);
  draft_min_rows.store(rows == 0 ? 4 : static_cast<int>(rows));
}

}  // namespace exl3

// moe_out_v2_fold: moe_out_v2's schema and output bits in three launches.
TORCH_LIBRARY_FRAGMENT(_exl3_C, m) {
  m.def("moe_out_v2_fold(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, int bits_x2, Tensor(a!) out, "
        "Tensor(b!) workspace, float swiglu_limit=0.0) -> ()");
  // The drafter's (bits_x2 8) row tile kernel floor; 0: by each tile's rows.
  m.def("moe_v2_draft_force_rows(int rows) -> ()", &exl3::moe_v2_draft_force_rows);
}

TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("moe_out_v2_fold", &exl3::moe_out_v2_fold);
}
