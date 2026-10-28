// Tensor-core dense EXL3 linear for prefill launches (dense_prefill_tc.h has the arithmetic).
#include "dense_prefill_tc.h"
#include "codec_prefill.cuh"
#include "common.h"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <atomic>
#include <limits>

namespace exl3 {
namespace dense_prefill_tc {
namespace {

// Launches of at most this many rows keep the shipped kernels.
constexpr int64_t kMaxShippedRows = 64;
// Fewer output tiles than this leave most of the 74 SMs idle: one tile's full-k pass then takes longer than
// the shipped kernel's whole launch (k/v at 482 rows: 8 tiles), so such launches keep the shipped kernels.
constexpr int64_t kMinTiles = 16;

#define EXL3_PREFILL_TC_FORMATS(X) X(10, 2) X(12, 2) X(14, 2)

constexpr int kSplitWarps = 8;

// One warp per (row, H128 block): linear_out's linear_input values, then a fixed-point hi/lo split on the
// block's power-of-two grid, stored in mma B-fragment order: for k16 step s and lane quad t, one 16-byte unit
// {hi(2t, 2t+1), hi(2t+8, 2t+9), lo(2t, 2t+1), lo(2t+8, 2t+9)} of k = 16s + ..., at unit row * (k / 4) + 4s + t.
__global__ void split_input(const __nv_bfloat16* x, const half* suh, int64_t rows, int64_t logical_in,
                            int64_t padded_in, uint4* act, float* inv_scale, int* counters, int64_t num_counters) {
  __shared__ __align__(16) half staged[kSplitWarps][2][128];
  // The GEMM's stream-K fixup counters start at zero on every call.
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < num_counters;
       i += int64_t(gridDim.x) * blockDim.x) {
    counters[i] = 0;
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t blocks = padded_in / 128;
  for (int64_t task = int64_t(blockIdx.x) * kSplitWarps + warp; task < rows * blocks;
       task += int64_t(gridDim.x) * kSplitWarps) {
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
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    int e = 0;
    if (m > 0.0f) {
      int ex;
      frexpf(m, &ex);              // m = f * 2^ex, f in [0.5, 1): m * 2^(11 - ex) in [2^10, 2^11)
      e = min(max(11 - ex, -126), 126);
    }
    const float s = ldexpf(1.0f, e);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float a = __fmul_rn(v[i], s);   // exact (power of two), |a| < 2^11
      const float h = rintf(a);              // integer in [-2^11, 2^11]: exact in FP16
      staged[warp][0][lane + 32 * i] = __float2half_rn(h);
      staged[warp][1][lane + 32 * i] = __float2half_rn(__fsub_rn(a, h));   // exact difference, |lo| <= 1/2
    }
    __syncwarp();
    const int k0 = 16 * (lane >> 2) + 2 * (lane & 3);
    const uint32_t* hi = reinterpret_cast<const uint32_t*>(staged[warp][0]);
    const uint32_t* lo = reinterpret_cast<const uint32_t*>(staged[warp][1]);
    act[task * 32 + lane] = make_uint4(hi[k0 / 2], hi[k0 / 2 + 4], lo[k0 / 2], lo[k0 / 2 + 4]);
    if (lane == 0) inv_scale[task] = ldexpf(1.0f, -e);
  }
}

// Block tile: 128 rows x 128 output columns (one H128 output block), k in stages of 64 (4 k16 steps).
// Shared memory: activation fragments, scales and raw trellis words in 3-stage cp.async rings; decoded W
// fragments double-buffered (stage i + 1 is decoded while stage i is multiplied).
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 64;
constexpr int kKTiles = kBK / 16;
constexpr int kStages = 3;
constexpr int kActBytes = kBM * kBK * 4;                     // hi + lo B fragments: kBK * 4 bytes per row
constexpr int kInvBytes = kBM * 4;
constexpr int kTrellisBytes = kKTiles * (kBN / 16) * 16 * 16;  // raw tiles of one stage at bits_x2 <= 16
constexpr int kWBytes = kKTiles * (kBN / 16) * 32 * 16;       // decoded A fragments, one uint4 per lane and tile
constexpr int kInvOffset = kStages * kActBytes;
constexpr int kTrellisOffset = kInvOffset + kStages * kInvBytes;
constexpr int kWOffset = kTrellisOffset + kStages * kTrellisBytes;
constexpr int kSmemBytes = kWOffset + 2 * kWBytes;
constexpr int kOutStride = kBN + 4;
static_assert(kBM * kOutStride * 4 <= kSmemBytes, "epilogue tile must fit the pipeline buffers");
static_assert(kSmemBytes <= 163 * 1024, "sm_80 opt-in shared memory");
static_assert(kBN / 16 == kWarps && kKTiles == 4, "warp w decodes tile column w; one decode per row group");
static_assert(128 % kBK == 0, "a stage lies inside one H128 block (one scale per row)");

// Work: 128 x 128 output tiles (row block, column block). A rectangle of row blocks [0, dp_rows) x column blocks
// [0, dp_cols) runs whole, one tile per block (2-D grid). The rest, row blocks [sk_row0, row blocks) x column
// blocks [sk_col0, column blocks) (sk_row0 = dp_rows with dp_cols = all columns, or sk_col0 = dp_cols with dp_rows
// = all rows), is numbered row-major (ts) and its k-stages are split contiguously over blocks [0, sk_ctas) of a
// second launch (stream-K). A tile split between blocks is finished by the block that arrives last at its
// counter, which sums every segment's partial in segment (k) order: deterministic, and no block ever waits for
// another.
struct PrefillProblem {
  const uint4* act;         // [rows][k / 4] B-fragment units (split_input)
  const float* inv_scale;   // [rows][k / 128]
  const uint16_t* trellis;
  const half* svh;
  float* partials;          // [2 * sk_ctas][kBM * kBN]: a block's first and last split segment
  int* counters;            // [sk_tiles], zeroed by split_input
  int64_t tiles_n;
  int64_t sk_row0;
  int64_t sk_col0;
  int64_t sk_width;         // column blocks - sk_col0
  int64_t sk_tiles;
  int64_t rows;
  int64_t logical_out;
  int sk_ctas;
  int k;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Byte offset of 16-byte unit c (4 * k16 step + quad t) of activation row r in a stage buffer. Rows are
// 256 bytes; the swizzle keeps each quarter warp (rows 2i, 2i + 1, quads 0..3) on distinct bank groups.
__device__ __forceinline__ int act_offset(int r, int c) { return r * (kBK * 4) + ((c ^ ((r & 1) << 2)) << 4); }

// D = A * B from C = 0 (one k16 step; no running accumulator inside the tensor core).
__device__ __forceinline__ void mma_k16(const uint4& a, uint32_t b0, uint32_t b1, float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "f"(0.0f));
}

// W = wh + wl exactly: wh = W rounded to the 2^-7 grid ((W + 12) - 12 in FP16; |W| < 4 for both codebooks, so
// W + 12 lies in [8, 16) where the FP16 ulp is 2^-7), wl = W - wh (|wl| <= 2^-8, exact in FP16).
__device__ __forceinline__ uint32_t grid_part(uint32_t w) {
  const __half2 twelve = __float2half2_rn(12.0f);
  const __half2 h = __hsub2(__hadd2(*reinterpret_cast<const __half2*>(&w), twelve), twelve);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t rest_part(uint32_t w, uint32_t wh) {
  const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&w), *reinterpret_cast<const __half2*>(&wh));
  return *reinterpret_cast<const uint32_t*>(&r);
}


// kSplit = false: whole tile (blockIdx.y, blockIdx.x). kSplit = true: block blockIdx.x's stream-K share.
template <int kBitsX2, int kCodebook, typename Out, bool kSplit>
__global__ void __launch_bounds__(kThreads, 1) prefill_gemm(PrefillProblem p, Out* out) {
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ int finishes_tile;
  using Ring = TileRing<kBitsX2>;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wn = warp & 1;    // 64-column half of the block
  const int wm = warp >> 1;   // 32-row quarter of the block
  const int t = lane & 3;
  const int g = lane >> 2;
  int64_t col0 = 0;   // current tile (set_split_tile)
  int64_t row0 = 0;
  int rows = 0;
  const int k = p.k;
  const int stages = k / kBK;   // k >= 128: at least two
  const int k_blocks = k / 128;
  const LaneWindow<kBitsX2> window(lane);
  const uint32_t smem0 = smem_addr(smem);

  // Activation fragments and the row scales of stage s.
  auto load_act = [&](int s) {
    const uint32_t base = smem0 + (s % kStages) * kActBytes;
    const int64_t k0 = int64_t(s) * kBK;
    constexpr int kUnits = kBK / 4;   // 16-byte units per row and stage
#pragma unroll
    for (int v = 0; v < kBM * kUnits / kThreads; ++v) {
      const int id = tid + v * kThreads;
      const int r = id / kUnits;
      const int c = id % kUnits;
      const bool valid = r < rows;
      const uint4* src = p.act + (valid ? (row0 + r) * (k / 4) + k0 / 4 + c : 0);
      cp_async16(base + act_offset(r, c), src, valid);
    }
    if (tid < kBM) {
      const bool valid = tid < rows;
      const float* src = p.inv_scale + (valid ? (row0 + tid) * k_blocks + k0 / 128 : 0);
      cp_async4(smem0 + kInvOffset + (s % kStages) * kInvBytes + tid * 4, src, valid);
    }
  };
  // Raw trellis of stage s: per k16 step, the block's 8 tiles are contiguous (16-byte aligned, host-checked).
  constexpr int kTileBytes = Ring::kTileHalfwords * 2;
  constexpr int kStepUnits = (kBN / 16) * kTileBytes / 16;
  auto load_trellis = [&](int s) {
    const uint32_t base = smem0 + kTrellisOffset + (s % kStages) * kTrellisBytes;
#pragma unroll
    for (int v = 0; v < (kKTiles * kStepUnits + kThreads - 1) / kThreads; ++v) {
      const int id = tid + v * kThreads;
      if (id < kKTiles * kStepUnits) {
        const int kt = id / kStepUnits;
        const int c = id % kStepUnits;
        const int64_t tile = (int64_t(s) * kKTiles + kt) * p.tiles_n + col0 / 16;
        const uint4* src = reinterpret_cast<const uint4*>(p.trellis + tile * Ring::kTileHalfwords) + c;
        cp_async16(base + (kt * kStepUnits + c) * 16, src, true);
      }
    }
  };
  // Warp w decodes tile column w of k16 step kt of the stage in trellis buffer `from` into W buffer `to`.
  auto decode_store = [&](int from, int to, int kt) {
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(
        smem + kTrellisOffset + from * kTrellisBytes + (kt * (kBN / 16) + warp) * kTileBytes);
    uint32_t words[Ring::kLoadWords];
#pragma unroll
    for (int i = 0; i < Ring::kLoadWords; ++i) words[i] = tile[window.word[i]];
    uint32_t h[4];
    decode_lane_half2<kBitsX2, kCodebook>(words, window.shift, h);
    // A = W^T (m = output column, k): {(g, 2t), (g + 8, 2t), (g, 2t + 8), (g + 8, 2t + 8)}.
    reinterpret_cast<uint4*>(smem + kWOffset + to * kWBytes)[(kt * kWarps + warp) * 32 + lane] =
        make_uint4(h[0], h[2], h[1], h[3]);
  };

  float acc[4][4][4];
  auto acc_at = [&](int f) -> float& { return acc[f >> 4][(f >> 2) & 3][f & 3]; };   // f unrolled: registers

  // Stream-K tile ts of the split region (row-major over its sk_width columns).
  auto set_split_tile = [&](int64_t ts) {
    row0 = (p.sk_row0 + ts / p.sk_width) * kBM;
    col0 = (p.sk_col0 + ts % p.sk_width) * kBN;
    rows = static_cast<int>(min(int64_t(kBM), p.rows - row0));
  };

  // acc = the current tile's k-stages [s0, s1), from zero. On return no copy is in flight and every warp is past
  // its last shared-memory read.
  auto mainloop = [&](int s0, int s1) {
#pragma unroll
    for (int f = 0; f < 64; ++f) acc_at(f) = 0.0f;
    // Group i holds act(i) and trellis(i + 1) (group s0 also trellis(s0)): at the top of iteration i, waiting
    // for all but the newest group gives stage i's activations and the trellis that iteration i decodes.
    load_act(s0);
    load_trellis(s0);
    if (s0 + 1 < s1) load_trellis(s0 + 1);
    cp_async_commit();
    if (s0 + 1 < s1) load_act(s0 + 1);
    if (s0 + 2 < s1) load_trellis(s0 + 2);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
#pragma unroll
    for (int kt = 0; kt < kKTiles; ++kt) decode_store(s0 % kStages, s0 & 1, kt);

    for (int i = s0; i < s1; ++i) {
      cp_async_wait<1>();
      __syncthreads();
      if (i + 2 < s1) load_act(i + 2);
      if (i + 3 < s1) load_trellis(i + 3);
      cp_async_commit();
      const unsigned char* act = smem + (i % kStages) * kActBytes;
      const uint4* wfrag = reinterpret_cast<const uint4*>(smem + kWOffset + (i & 1) * kWBytes);
      const float* inv = reinterpret_cast<const float*>(smem + kInvOffset + (i % kStages) * kInvBytes);
      float2 scale[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) scale[j] = *reinterpret_cast<const float2*>(inv + 32 * wm + 8 * j + 2 * t);
#pragma unroll
      for (int ks = 0; ks < kKTiles; ++ks) {
        uint4 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          b[j] = *reinterpret_cast<const uint4*>(act + act_offset(32 * wm + 8 * j + g, 4 * ks + t));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 a = wfrag[(ks * kWarps + 4 * wn + q) * 32 + lane];
          const uint4 wh = make_uint4(grid_part(a.x), grid_part(a.y), grid_part(a.z), grid_part(a.w));
          const uint4 wl = make_uint4(rest_part(a.x, wh.x), rest_part(a.y, wh.y), rest_part(a.z, wh.z),
                                      rest_part(a.w, wh.w));
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            // wh * hi: integers on the 2^-7 grid, |sum| <= 16 * 2^11 * 2^9: exact in FP32, no truncation.
            float dh[4], dl[4], dw[4];
            mma_k16(wh, b[j].x, b[j].y, dh);
            // The small terms W * lo and wl * hi (|lo| <= 1/2, |wl| <= 2^-8): independent tensor-core sums.
            mma_k16(a, b[j].z, b[j].w, dl);
            mma_k16(wl, b[j].x, b[j].y, dw);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              acc[q][j][e] = __fmaf_rn(__fadd_rn(dh[e], __fadd_rn(dl[e], dw[e])), e & 1 ? scale[j].y : scale[j].x,
                                       acc[q][j][e]);
            }
          }
        }
        // The next stage's decode, interleaved with this stage's MMAs (past the last stage: unused buffer).
        decode_store((i + 1) % kStages, (i + 1) & 1, ks);
      }
    }
    cp_async_wait<0>();
    __syncthreads();
  };

  // linear_out's linear_output on the current tile's H128 column block: H128, then svh.
  auto epilogue = [&]() {
    float* tile = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = (4 * wn + q) * 16 + g;
        const int r = 32 * wm + 8 * j + 2 * t;
        tile[r * kOutStride + c] = acc[q][j][0];
        tile[(r + 1) * kOutStride + c] = acc[q][j][1];
        tile[r * kOutStride + c + 8] = acc[q][j][2];
        tile[(r + 1) * kOutStride + c + 8] = acc[q][j][3];
      }
    }
    __syncthreads();
    for (int r = warp; r < rows; r += kThreads / 32) {
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = tile[r * kOutStride + lane + 32 * i];
      hadamard128_warp(v, lane);
      const int64_t row = row0 + r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = col0 + lane + 32 * i;
        if (col < p.logical_out) {
          store_output(out + row * p.logical_out + col, __fmul_rn(v[i], __half2float(p.svh[col])));
        }
      }
    }
    __syncthreads();   // the tile buffer is the next mainloop's pipeline memory
  };

  if constexpr (!kSplit) {
    // Whole tiles: grid (dp_cols, dp_rows), coordinates straight from blockIdx (cheap to rematerialise: this path
    // sits at the register limit).
    col0 = int64_t(blockIdx.x) * kBN;
    row0 = int64_t(blockIdx.y) * kBM;
    rows = static_cast<int>(min(int64_t(kBM), p.rows - row0));
    mainloop(0, stages);
    epilogue();
  } else {
    // Stream-K: block c owns units [start(c), start(c + 1)) of the ts * stages + stage sequence.
    const int64_t units = p.sk_tiles * stages;
    const int64_t base = units / p.sk_ctas;   // >= 1 (host-checked)
    const int64_t rem = units % p.sk_ctas;
    auto start = [&](int64_t c) { return c * base + min(c, rem); };
    auto owner = [&](int64_t u) {
      const int64_t cut = rem * (base + 1);
      return u < cut ? u / (base + 1) : rem + (u - cut) / base;
    };
    // A block has at most two split segments: in its first tile and in its last tile.
    auto slot = [&](int64_t c, int64_t ts) {
      const bool second = ts == (start(c + 1) - 1) / stages && ts != start(c) / stages;
      return p.partials + (2 * c + (second ? 1 : 0)) * int64_t(kBM * kBN) + tid;
    };
    const int64_t c = blockIdx.x;
    for (int64_t u = start(c), end = start(c + 1); u < end;) {
      const int64_t ts = u / stages;
      const int s0 = static_cast<int>(u % stages);
      const int s1 = static_cast<int>(min(int64_t(stages), s0 + (end - u)));
      u += s1 - s0;
      set_split_tile(ts);
      mainloop(s0, s1);
      if (s0 > 0 || s1 < stages) {
        float* mine = slot(c, ts);
#pragma unroll
        for (int f = 0; f < 64; ++f) __stcg(mine + f * kThreads, acc_at(f));
        __threadfence();
        __syncthreads();
        const int64_t first = owner(ts * stages);
        const int segments = static_cast<int>(owner(ts * stages + stages - 1) - first + 1);
        if (tid == 0) finishes_tile = atomicAdd(p.counters + ts, 1) == segments - 1;
        __syncthreads();
        if (!finishes_tile) continue;
        __threadfence();
        // Every segment's partial (this block's included), summed in k order.
        for (int j = 0; j < segments; ++j) {
          const float* src = slot(first + j, ts);
#pragma unroll
          for (int f = 0; f < 64; ++f) {
            const float v = __ldcg(src + f * kThreads);
            acc_at(f) = j == 0 ? v : __fadd_rn(acc_at(f), v);
          }
        }
      }
      epilogue();
    }
  }
}

template <int kBitsX2, int kCodebook, typename Out, bool kSplit>
void launch_gemm_mode(const PrefillProblem& p, Out* out, dim3 grid, cudaStream_t stream) {
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(prefill_gemm<kBitsX2, kCodebook, Out, kSplit>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(prefill_gemm<kBitsX2, kCodebook, Out, kSplit>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    return true;
  }();
  (void)configured;
  prefill_gemm<kBitsX2, kCodebook, Out, kSplit><<<grid, kThreads, kSmemBytes, stream>>>(p, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Whole tiles first (grid = dp_grid: column blocks x row blocks of the whole-tile rectangle), then the split
// tiles (grid = sk_ctas) in a second launch.
template <int kBitsX2, int kCodebook, typename Out>
void launch_gemm(const PrefillProblem& p, Out* out, dim3 dp_grid, cudaStream_t stream) {
  if (dp_grid.x > 0 && dp_grid.y > 0) launch_gemm_mode<kBitsX2, kCodebook, Out, false>(p, out, dp_grid, stream);
  if (p.sk_ctas > 0) launch_gemm_mode<kBitsX2, kCodebook, Out, true>(p, out, dim3(p.sk_ctas), stream);
}

struct Schedule {
  int64_t dp_rows;   // whole-tile rectangle: row blocks [0, dp_rows) x column blocks [0, dp_cols)
  int64_t dp_cols;
  int64_t sk_row0;   // split region: row blocks [sk_row0, ..) x column blocks [sk_col0, ..)
  int64_t sk_col0;
  int sk_ctas;
};

// One block per SM (kSmemBytes). Whole tiles alone leave the last wave part-empty (e.g. 105 tiles on 74 SMs: 2
// waves for 1.42 waves of work). Candidates: the whole-tile rectangle is the row blocks, or the column blocks,
// that fit in all full waves but one (or in all full waves); the rest is split by k over the blocks, as many as
// the free workspace holds two partial tiles for. Predicted time in k-stages per block: whole waves + the largest
// split share + a fixup of 1 + (segments per split tile) / 2 (the finishing block re-reads every segment's
// partial). Kept only when it beats whole tiles by 5 %. A pure function of the shape: deterministic.
Schedule plan_schedule(int64_t row_blocks, int64_t col_blocks, int64_t stages, int64_t sms, int64_t free_floats) {
  const int64_t tiles = row_blocks * col_blocks;
  const Schedule all_whole{row_blocks, col_blocks, row_blocks, 0, 0};
  const int64_t whole = (tiles + sms - 1) / sms * stages;
  Schedule best = all_whole;
  int64_t best_time = whole;
  for (int64_t waves = std::max<int64_t>(tiles / sms - 1, 0); waves <= tiles / sms; ++waves) {
    for (const bool by_rows : {true, false}) {
      const int64_t dp_rows = by_rows ? waves * sms / col_blocks : row_blocks;
      const int64_t dp_cols = by_rows ? col_blocks : std::min(col_blocks, waves * sms / row_blocks);
      const int64_t split = tiles - dp_rows * dp_cols;
      if (split <= 0) continue;
      const int64_t units = split * stages;
      const int64_t room = free_floats - split;   // counters first
      const int64_t ctas = std::min({sms, units, room > 0 ? room / (2 * kBM * kBN) : int64_t(0)});
      if (ctas < 1) continue;
      const int64_t segments = (stages + units / ctas - 1) / (units / ctas) + 1;
      const int64_t time = (dp_rows * dp_cols + sms - 1) / sms * stages + (units + ctas - 1) / ctas +
                           (2 + segments) / 2;
      if (time < best_time) {
        best = by_rows ? Schedule{dp_rows, col_blocks, dp_rows, 0, static_cast<int>(ctas)}
                       : Schedule{row_blocks, dp_cols, 0, dp_cols, static_cast<int>(ctas)};
        best_time = time;
      }
    }
  }
  return best.sk_ctas > 0 && best_time * 20 > whole * 19 ? all_whole : best;
}

// 0: plan_schedule; 1: whole tiles only (set_prefill_schedule).
std::atomic<int> schedule_mode{0};

bool format_covered(int64_t bits_x2, int64_t codebook) {
#define EXL3_PREFILL_TC_COVERED(B, C) \
  if (bits_x2 == B && codebook == C) return true;
  EXL3_PREFILL_TC_FORMATS(EXL3_PREFILL_TC_COVERED)
#undef EXL3_PREFILL_TC_COVERED
  return false;
}

// FP32 elements of the activation scratch: hi + lo FP16 B fragments (4 bytes per (row, k)) and one scale per
// (row, H128 block).
int64_t activation_elements(int64_t rows, int64_t padded_in) {
  return checked_add(checked_mul(rows, padded_in), checked_mul(rows, padded_in / 128));
}

template <typename Out>
void prefill_linear(const __nv_bfloat16* x, const half* suh, const half* svh, const uint16_t* trellis,
                    int format, int64_t rows, int64_t logical_in, int64_t padded_in, int64_t logical_out,
                    int64_t padded_out, float* workspace, int64_t workspace_elements, int sms, Out* out,
                    cudaStream_t stream) {
  uint4* act = reinterpret_cast<uint4*>(workspace);
  float* inv_scale = workspace + rows * padded_in;
  float* partials = inv_scale + rows * (padded_in / 128);
  const int64_t col_blocks = padded_out / kBN;
  const int64_t row_blocks = (rows + kBM - 1) / kBM;
  const Schedule plan =
      schedule_mode.load(std::memory_order_relaxed) == 1
          ? Schedule{row_blocks, col_blocks, row_blocks, 0, 0}
          : plan_schedule(row_blocks, col_blocks, padded_in / kBK, sms, workspace + workspace_elements - partials);
  const int64_t sk_tiles = plan.sk_ctas > 0 ? (row_blocks - plan.sk_row0) * (col_blocks - plan.sk_col0) : 0;
  int* counters = reinterpret_cast<int*>(partials + int64_t(2) * plan.sk_ctas * kBM * kBN);

  const int64_t tasks = rows * (padded_in / 128);
  split_input<<<launch_blocks((tasks + kSplitWarps - 1) / kSplitWarps), 32 * kSplitWarps, 0, stream>>>(
      x, suh, rows, logical_in, padded_in, act, inv_scale, counters, sk_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  PrefillProblem p{};
  p.act = act;
  p.inv_scale = inv_scale;
  p.trellis = trellis;
  p.svh = svh;
  p.partials = partials;
  p.counters = counters;
  p.tiles_n = padded_out / 16;
  p.sk_row0 = plan.sk_row0;
  p.sk_col0 = plan.sk_col0;
  p.sk_width = col_blocks - plan.sk_col0;
  p.sk_tiles = sk_tiles;
  p.rows = rows;
  p.logical_out = logical_out;
  p.sk_ctas = plan.sk_ctas;
  p.k = static_cast<int>(padded_in);
  const dim3 dp_grid(static_cast<unsigned int>(plan.dp_cols), static_cast<unsigned int>(plan.dp_rows));
  switch (format) {
#define EXL3_PREFILL_TC_CASE(B, C) \
    case format_key(B, C): return launch_gemm<B, C, Out>(p, out, dp_grid, stream);
    EXL3_PREFILL_TC_FORMATS(EXL3_PREFILL_TC_CASE)
#undef EXL3_PREFILL_TC_CASE
  }
  TORCH_CHECK(false, "Unsupported EXL3 rate/codebook combination for the prefill path");
}

}  // namespace
}  // namespace dense_prefill_tc

bool linear_prefill_applies(int64_t rows, int64_t padded_in, int64_t padded_out, int64_t bits_x2,
                            int64_t codebook) {
  using namespace dense_prefill_tc;
  check_padded(padded_in);
  check_padded(padded_out);
  if (rows <= kMaxShippedRows || !format_covered(bits_x2, codebook)) return false;
  if ((padded_out / kBN) * ((rows + kBM - 1) / kBM) < kMinTiles) return false;
  return activation_elements(rows, padded_in) <= linear_workspace_size(rows, padded_in, padded_out);
}

void set_prefill_schedule(int64_t mode) {
  TORCH_CHECK(mode == 0 || mode == 1, "EXL3 prefill schedule must be 0 (planned) or 1 (whole tiles)");
  dense_prefill_tc::schedule_mode.store(static_cast<int>(mode), std::memory_order_relaxed);
}

void linear_out_prefill(const torch::Tensor& x, const torch::Tensor& trellis, const torch::Tensor& suh,
                        const torch::Tensor& svh, int64_t bits_x2, int64_t codebook, int64_t logical_in,
                        int64_t logical_out, torch::Tensor& out, torch::Tensor& workspace) {
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
  const int64_t needed = linear_workspace_size(rows, padded_in, padded_out);
  TORCH_CHECK(workspace.numel() >= needed, "EXL3 linear workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 linear workspace must be 16-byte aligned");
  check_mutations(out, workspace, {&x, &trellis, &suh, &svh});
  TORCH_CHECK(linear_prefill_applies(rows, padded_in, padded_out, bits_x2, codebook),
              "EXL3 prefill path does not cover this launch (linear_prefill_applies is false)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(trellis.data_ptr()) % 16 == 0,
              "EXL3 prefill path needs a 16-byte aligned trellis");
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  // The planner places stream-K partials and counters only inside linear_out's workspace bound.
  with_linear_out(out, [&](auto* out_ptr) {
    dense_prefill_tc::prefill_linear(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
        format_key(static_cast<int>(bits_x2), static_cast<int>(codebook)), rows, logical_in,
        padded_in, logical_out, padded_out, workspace.data_ptr<float>(), needed,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount, out_ptr, stream);
  });
}

}  // namespace exl3

TORCH_LIBRARY_FRAGMENT(_exl3_C, m) {
  m.def("linear_out_prefill(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int bits_x2, int codebook, int logical_in, int logical_out, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  // Host arithmetic and a process setting: catch-all CPU functions that never touch a device.
  m.def("linear_prefill_applies(int rows, int padded_in, int padded_out, int bits_x2, "
        "int codebook) -> bool", &exl3::linear_prefill_applies);
  m.def("set_prefill_schedule(int mode) -> ()", &exl3::set_prefill_schedule);
}

TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("linear_out_prefill", &exl3::linear_out_prefill);
}
