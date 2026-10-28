// Split-K dense EXL3 linear (VLLM_EXL3_DENSE_SPLITK).
//
// For dense launches that today run the tiny or small block shape with at
// most kMaxBlocks blocks, linear_out_splitk replaces linear_out's three
// launches with:
// - linear_input, which also zeroes the FP32 result rows;
// - warp-specialized blocks (pipe_kernel): separate decode warps fill a ring
//   of decoded chunks ahead of the FMA threads, and each (row tile, column
//   tile) runs as two items, one per half of K, each atomically added onto
//   the zeroed result;
// - linear_output with the output H128 and svh in FP64, rounded once to FP32
//   and then, for a BF16 out, to BF16.
// Every other launch, and every format outside kServedFormats, is linear_out.
//
// Arithmetic (not linear_out's; the results differ in the last bits). Per K
// half, each BK chunk's FMA chain (k in increasing order) starts from 0.0f
// and the chunk sums are added in chunk order in FP32. The two half sums are
// added atomically onto 0.0f: fl(0 + a + b) == fl(0 + b + a), so results are
// deterministic. Which path a launch takes depends on its row count, so a
// row's result may depend on the rows that share its launch.
//
// The workspace is linear_out's (linear_workspace_size): the halves add into
// its result rows, so no partial buffers are needed.
#include "dense_splitk.h"

#include "common.h"
#include "codec.cuh"
#include "dense_mma.h"
#include "gemm.cuh"
#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

namespace exl3 {
namespace dense_splitk {
namespace {

// gemm.cu's tiny and small block configurations (the same tile geometry and
// micro-tile mapping, so dense_gemm_shape's choice describes these blocks).
template <int Threads, int BM, int BN, int BK, int TM, int TN, int MinBlocks, bool FuseInput>
struct GemmConfig {
  static constexpr int kThreads = Threads;
  static constexpr int kBM = BM;
  static constexpr int kBN = BN;
  static constexpr int kBK = BK;
  static constexpr int kTM = TM;
  static constexpr int kTN = TN;
  static constexpr int kMinBlocks = MinBlocks;
  static constexpr bool kFuseInput = FuseInput;
  static constexpr int kWarps = Threads / 32;
  static constexpr int kRowsPerWarp = (BM + kWarps - 1) / kWarps;
  // Activations are stored k-major; rows are read as float4 when TM >= 4.
  static constexpr int kAStride = TM >= 4 ? BM + 4 : BM + 1;
  // BN + 4 is 4 (mod 16): the eight stores of a decoding warp hit 32 banks.
  static constexpr int kWStride = BN + 4;
  static constexpr int kColThreads = BN / TN;
  static constexpr int kFmaThreads = (BM / TM) * kColThreads;
  static constexpr int kTileRows = BK / 16;
  static constexpr int kTileCols = BN / 16;
  static constexpr int kTasksPerThread = kTileRows * kTileCols * 32 / Threads;
  static constexpr int kAVectors = BM * BK / 4;
  static constexpr int kAPerThread = (kAVectors + Threads - 1) / Threads;
  static_assert(Threads % 32 == 0, "whole warps");
  static_assert(kTileRows * kTileCols * 32 % Threads == 0, "even decode split");
  static_assert(kFmaThreads <= Threads, "micro-tiles exceed the block");
  static_assert(BK % 16 == 0 && 128 % BK == 0 && BN % 16 == 0 && 128 % BN == 0,
                "K/N tiles must divide the H128 blocks");
  static_assert((TM == 1 || TM == 4 || TM == 8) && (TN == 1 || TN == 4 || TN == 8),
                "micro-tile widths");
};

using TinyConfig = GemmConfig<128, 4, 32, 64, 1, 1, 4, true>;
using SmallConfig = GemmConfig<128, 16, 32, 32, 4, 1, 4, true>;

// Offset of micro-tile element i for thread t. Width 8 is split into two
// groups of four, half a block apart, so a warp's float4 reads are contiguous.
template <int TX, int BX>
__device__ __forceinline__ int micro_offset(int t, int i) {
  if constexpr (TX == 8) {
    return (i < 4 ? 0 : BX / 2) + t * 4 + (i & 3);
  } else if constexpr (TX == 4) {
    return t * 4 + i;
  } else {
    return t;
  }
}

template <int TX, int BX>
__device__ __forceinline__ void load_micro(const float* row, int t, float (&v)[TX]) {
  if constexpr (TX == 1) {
    v[0] = row[t];
  } else {
#pragma unroll
    for (int g = 0; g < TX / 4; ++g) {
      const float4 q = *reinterpret_cast<const float4*>(row + micro_offset<TX, BX>(t, 4 * g));
      v[4 * g] = q.x;
      v[4 * g + 1] = q.y;
      v[4 * g + 2] = q.z;
      v[4 * g + 3] = q.w;
    }
  }
}

// A dense problem plus its K split factor. The kernel parameter has the
// GemmProblem fields first and `splits` after them.
struct SplitProblem : GemmProblem {
  // 2: the first and second halves of the chunks run as separate items, each
  // added atomically onto c, which the caller zeroes first. 0 or 1: no split.
  int splits;
};

// Producer groups and FMA threads in one block. The block keeps the base
// configuration's kThreads FMA threads (threads 0 .. Cfg::kThreads - 1, the
// same micro-tile mapping) and adds kGroups producer groups of Cfg::kThreads
// threads with the base fetch and stash mapping. Producers fill a ring of
// kStages decoded chunks ahead of the FMA threads: chunk c sits in stage
// c % kStages and comes from group c % kGroups. Named barriers pair the FMA
// threads with one stage's group: full[stage] (producers arrive, FMA threads
// wait) and empty[stage] (FMA threads arrive once a later chunk needs the
// stage, producers wait). Dense problems only: the format is known on the
// host, so each format gets its own kernel and the producers run without a
// device-side format switch.
template <class Cfg, int Groups, int MinBlocks>
struct PipeConfig {
  using Base = Cfg;
  static constexpr int kGroups = Groups;
  static constexpr int kStages = 2 * Groups;
  static constexpr int kThreads = Cfg::kThreads * (1 + Groups);
  static constexpr int kMinBlocks = MinBlocks;
  // FMA threads plus one producer group meet at each named barrier.
  static constexpr int kPairThreads = 2 * Cfg::kThreads;
  // Barrier 0 is __syncthreads; full[s] = 1 + s, empty[s] = 1 + kStages + s.
  static_assert(1 + 2 * kStages <= 16, "sm_80 has 16 named barriers");
  static_assert(Cfg::kFmaThreads <= Cfg::kThreads, "FMA threads exceed their group");
  __device__ static constexpr int full(int stage) { return 1 + stage; }
  __device__ static constexpr int empty(int stage) { return 1 + kStages + stage; }
};

using TinyPipe = PipeConfig<TinyConfig, 2, 2>;
using SmallPipe = PipeConfig<SmallConfig, 2, 2>;

template <class P>
struct PipeSmem {
  using Cfg = typename P::Base;
  float a[P::kStages][Cfg::kBK][Cfg::kAStride];
  float w[P::kStages][Cfg::kBK][Cfg::kWStride];
  int row[Cfg::kBM];
};

// bar.arrive / bar.sync on one named barrier: the producer-consumer pattern
// of the PTX barrier section (shared stores before arrive are visible to the
// threads that complete the matching sync).
__device__ __forceinline__ void barrier_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void barrier_arrive(int id, int threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// Producer group `group`: chunks group, group + kGroups, ... of the item's
// `chunks` chunks starting at chunk `first`, with the base fetch (thread
// index ptid in place of threadIdx.x) and stash into the chunk's stage. Two
// register sets keep the next chunk's loads in flight while the current one
// is decoded.
template <int kBitsX2, int kCodebook, class P>
__device__ __forceinline__ void produce_tile(const SplitProblem& p, PipeSmem<P>& s,
                                             const uint16_t* trellis, int64_t tiles_n,
                                             int64_t col0, int first, int chunks, int group,
                                             int ptid) {
  using Cfg = typename P::Base;
  using Ring = TileRing<kBitsX2>;
  using Words = uint32_t[Cfg::kTasksPerThread][Ring::kLoadWords];
  using Staged = float4[Cfg::kAPerThread];
  constexpr int BK = Cfg::kBK;
  constexpr int T = Cfg::kThreads;
  constexpr int S = P::kStages;
  constexpr int G = P::kGroups;
  const int lane = ptid & 31;
  const LaneWindow<kBitsX2> window(lane);
  const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & 3) == 0;
  // Chunk indices below are local to the item; first + chunk addresses K.
  const int64_t tile_col0 = col0 / 16;
  auto fetch = [&](int chunk, Words& words, Staged& staged) {
#pragma unroll
    for (int t = 0; t < Cfg::kTasksPerThread; ++t) {
      const int tile = (ptid + t * T) >> 5;
      const int tile_row = tile / Cfg::kTileCols;
      const int tile_col = tile % Cfg::kTileCols;
      const int64_t index = (int64_t(first + chunk) * Cfg::kTileRows + tile_row) * tiles_n +
                            tile_col0 + tile_col;
      load_lane_words<kBitsX2>(trellis + index * Ring::kTileHalfwords, window, aligned,
                               words[t]);
    }
#pragma unroll
    for (int v = 0; v < Cfg::kAPerThread; ++v) {
      const int index = ptid + v * T;
      staged[v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (index < Cfg::kAVectors) {
        const int r = index / (BK / 4);
        const int k4 = index % (BK / 4);
        const int row = s.row[r];
        if (row >= 0) {
          staged[v] = *reinterpret_cast<const float4*>(
              p.a + int64_t(row) * p.a_stride + int64_t(first + chunk) * BK + 4 * k4);
        }
      }
    }
  };
  auto put = [&](int chunk, const Words& words, const Staged& staged) {
    const int stage = chunk % S;
    // The stage last held chunk - S; wait until the FMA threads are done.
    if (chunk >= S) barrier_sync(P::empty(stage), P::kPairThreads);
#pragma unroll
    for (int t = 0; t < Cfg::kTasksPerThread; ++t) {
      const int tile = (ptid + t * T) >> 5;
      const int tile_row = tile / Cfg::kTileCols;
      const int tile_col = tile % Cfg::kTileCols;
      float value[8];
      decode_lane<kBitsX2, kCodebook>(words[t], window.shift, value);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s.w[stage][tile_row * 16 + lane_tile_row(lane, j)]
           [tile_col * 16 + lane_tile_col(lane, j)] = value[j];
      }
    }
#pragma unroll
    for (int v = 0; v < Cfg::kAPerThread; ++v) {
      const int index = ptid + v * T;
      if (index < Cfg::kAVectors) {
        const int r = index / (BK / 4);
        const int k4 = index % (BK / 4);
        s.a[stage][4 * k4][r] = staged[v].x;
        s.a[stage][4 * k4 + 1][r] = staged[v].y;
        s.a[stage][4 * k4 + 2][r] = staged[v].z;
        s.a[stage][4 * k4 + 3][r] = staged[v].w;
      }
    }
    barrier_arrive(P::full(stage), P::kPairThreads);
  };
  if (group >= chunks) return;
  Words words0, words1;
  Staged staged0, staged1;
  fetch(group, words0, staged0);
  for (int chunk = group; chunk < chunks; chunk += 2 * G) {
    const bool next = chunk + G < chunks;
    if (next) fetch(chunk + G, words1, staged1);
    put(chunk, words0, staged0);
    if (!next) break;
    if (chunk + 2 * G < chunks) fetch(chunk + 2 * G, words0, staged0);
    put(chunk + G, words1, staged1);
  }
}

// FMA threads: the base micro-tile chain over the item's `chunks` chunks and
// stores (or, with `accumulate`, float atomic adds into a zeroed result),
// operands from the ring.
template <class P>
__device__ __forceinline__ void consume_tile(const SplitProblem& p, PipeSmem<P>& s,
                                             int64_t col0, int rows, float* c, int chunks,
                                             bool accumulate) {
  using Cfg = typename P::Base;
  constexpr int BK = Cfg::kBK;
  constexpr int S = P::kStages;
  const int tid = threadIdx.x;
  const int tn = tid % Cfg::kColThreads;
  const int tm = tid / Cfg::kColThreads;
  const bool computes = tid < Cfg::kFmaThreads &&
                        micro_offset<Cfg::kTM, Cfg::kBM>(tm, 0) < rows;
  // Blocked summation: each chunk's FMA chain starts from 0.0f (BK terms in k
  // order) and the chunk sums are added in chunk order in FP32: one rounding
  // per chunk at the running sum's magnitude instead of one per k.
  float acc[Cfg::kTM][Cfg::kTN];
#pragma unroll
  for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
    for (int j = 0; j < Cfg::kTN; ++j) acc[i][j] = 0.0f;
  }
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int stage = chunk % S;
    barrier_sync(P::full(stage), P::kPairThreads);
    if (computes) {
      float part[Cfg::kTM][Cfg::kTN];
#pragma unroll
      for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
        for (int j = 0; j < Cfg::kTN; ++j) part[i][j] = 0.0f;
      }
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float av[Cfg::kTM];
        float wv[Cfg::kTN];
        load_micro<Cfg::kTM, Cfg::kBM>(&s.a[stage][kk][0], tm, av);
        load_micro<Cfg::kTN, Cfg::kBN>(&s.w[stage][kk][0], tn, wv);
#pragma unroll
        for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
          for (int j = 0; j < Cfg::kTN; ++j) {
            part[i][j] = __fmaf_rn(av[i], wv[j], part[i][j]);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
        for (int j = 0; j < Cfg::kTN; ++j) acc[i][j] = __fadd_rn(acc[i][j], part[i][j]);
      }
    }
    // Release the stage only when a later chunk will reuse it, so every
    // arrive meets exactly one producer wait within this item.
    if (chunk + S < chunks) barrier_arrive(P::empty(stage), P::kPairThreads);
  }
  if (computes) {
#pragma unroll
    for (int i = 0; i < Cfg::kTM; ++i) {
      const int r = micro_offset<Cfg::kTM, Cfg::kBM>(tm, i);
      if (r < rows) {
        float* out = c + int64_t(s.row[r]) * p.c_stride + col0;
        if (accumulate) {
#pragma unroll
          for (int j = 0; j < Cfg::kTN; ++j) {
            atomicAdd(out + micro_offset<Cfg::kTN, Cfg::kBN>(tn, j), acc[i][j]);
          }
        } else if constexpr (Cfg::kTN == 1) {
          out[tn] = acc[i][0];
        } else {
#pragma unroll
          for (int g = 0; g < Cfg::kTN / 4; ++g) {
            *reinterpret_cast<float4*>(out + micro_offset<Cfg::kTN, Cfg::kBN>(tn, 4 * g)) =
                make_float4(acc[i][4 * g], acc[i][4 * g + 1], acc[i][4 * g + 2],
                            acc[i][4 * g + 3]);
          }
        }
      }
    }
  }
}

// Item loop for one dense matrix of one format; item = (row tile, column
// tile) x split, the split varying fastest. With two splits each half-K sum
// is added atomically onto the zeroed result: fl(0 + a + b) is the same in
// either order, so the result stays bitwise deterministic.
template <class P, int kBitsX2, int kCodebook>
__global__ void __launch_bounds__(P::kThreads, P::kMinBlocks)
pipe_kernel(SplitProblem p) {
  using Cfg = typename P::Base;
  __shared__ __align__(16) PipeSmem<P> s;
  const int64_t col_tiles = p.n / Cfg::kBN;
  const int splits = p.splits > 1 ? 2 : 1;
  const int total_chunks = p.k / Cfg::kBK;
  const int64_t items = (p.rows + Cfg::kBM - 1) / Cfg::kBM * col_tiles * splits;
  for (int64_t item = blockIdx.x; item < items; item += gridDim.x) {
    const int split = static_cast<int>(item % splits);
    const int64_t tile = item / splits;
    const int64_t row0 = tile / col_tiles * Cfg::kBM;
    const int64_t col0 = (tile % col_tiles) * Cfg::kBN;
    const int first = static_cast<int>(int64_t(split) * total_chunks / splits);
    const int chunks = static_cast<int>(int64_t(split + 1) * total_chunks / splits) - first;
    const int rows = static_cast<int>(std::min<int64_t>(Cfg::kBM, p.rows - row0));
    for (int i = threadIdx.x; i < Cfg::kBM; i += P::kThreads) {
      s.row[i] = i < rows ? static_cast<int>(row0 + i) : -1;
    }
    __syncthreads();
    if (threadIdx.x < Cfg::kThreads) {
      consume_tile<P>(p, s, col0, rows, p.c, chunks, splits > 1);
    } else {
      const int ptid = threadIdx.x - Cfg::kThreads;
      produce_tile<kBitsX2, kCodebook, P>(p, s, p.trellis, p.tiles_n, col0, first, chunks,
                                          ptid / Cfg::kThreads, ptid % Cfg::kThreads);
    }
    __syncthreads();
  }
}

// The formats this path serves (bits_x2, codebook): 5, 6 and 7 bits per
// weight with the mul1 codebook. Any other format keeps linear_out.
#define EXL3_SPLITK_FORMATS(X) X(10, 2) X(12, 2) X(14, 2)

bool served_format(int64_t bits_x2, int64_t codebook) {
#define EXL3_SERVED(B, C) if (bits_x2 == B && codebook == C) return true;
  EXL3_SPLITK_FORMATS(EXL3_SERVED)
#undef EXL3_SERVED
  return false;
}

template <class P>
void launch_pipelined(const SplitProblem& problem, int64_t max_items, cudaStream_t stream) {
  if (max_items <= 0) return;
  const unsigned int grid =
      static_cast<unsigned int>(std::min<int64_t>(max_items, std::numeric_limits<int>::max()));
  switch (problem.format) {
#define EXL3_CASE(B, C)                                                 \
    case format_key(B, C):                                              \
      pipe_kernel<P, B, C><<<grid, P::kThreads, 0, stream>>>(problem); \
      break;
    EXL3_SPLITK_FORMATS(EXL3_CASE)
#undef EXL3_CASE
    default:
      TORCH_CHECK(false, "EXL3 split-K has no kernel for this rate/codebook combination");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Tiny and small grids above this many blocks (the LM head's 3,880 to 7,760)
// keep linear_out: the warp-specialized blocks lose there.
constexpr int64_t kMaxBlocks = 1024;

// 2 for the tiny and small shapes with at most kMaxBlocks blocks, else 1.
int shape_splits(GemmShape shape, int64_t rows, int64_t n) {
  if (rows <= 0) return 1;
  if (shape != GemmShape::kTiny && shape != GemmShape::kSmall) return 1;
  const int64_t bm = gemm_block_rows(shape);
  const int64_t blocks = (rows + bm - 1) / bm * (n / gemm_block_cols(shape));
  return blocks <= kMaxBlocks ? 2 : 1;
}

constexpr int kWarpsPerBlock = 8;

unsigned int warp_blocks(int64_t tasks) {
  return launch_blocks((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

// One warp per (row, 128-column block): x * suh, then H128 (linear_out's
// arithmetic). With `zeroed` != nullptr the launch also zeroes zero_vectors
// float4s there (the split-K accumulation target).
__global__ void linear_input(const __nv_bfloat16* x, const half* suh, int64_t rows,
                             int64_t logical_in, int64_t padded_in, float* transformed,
                             float4* zeroed, int64_t zero_vectors) {
  if (zeroed != nullptr) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < zero_vectors;
         i += int64_t(gridDim.x) * blockDim.x) {
      zeroed[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
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


// hadamard128_warp's butterfly (same element layout and order) in FP64,
// scaled by 1/sqrt(128) rounded to FP64.
__device__ __forceinline__ void hadamard128_warp_f64(double (&v)[4], int lane) {
#pragma unroll
  for (int distance = 1; distance < 32; distance <<= 1) {
    const bool upper = (lane & distance) != 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const double other = __shfl_xor_sync(0xffffffffu, v[i], distance);
      v[i] = upper ? other - v[i] : v[i] + other;
    }
  }
#pragma unroll
  for (int distance = 1; distance < 4; distance <<= 1) {
    double next[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const double other = v[i ^ distance];
      next[i] = (i & distance) ? other - v[i] : v[i] + other;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = next[i];
  }
  constexpr double kScale = 0.088388347648318440550;  // 1 / sqrt(128)
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= kScale;
}

// One warp per (row, 128-column block): H128, then svh, in FP64 and rounded
// once to FP32, then stored as Out (store_output).
template <typename Out>
__global__ void linear_output(const float* result, const half* svh, int64_t rows,
                              int64_t padded_out, int64_t logical_out, Out* out) {
  const int lane = threadIdx.x & 31;
  const int64_t blocks = padded_out / 128;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < rows * blocks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t row = task / blocks;
    const int64_t base = (task % blocks) * 128 + lane;
    double v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = result[row * padded_out + base + 32 * i];
    hadamard128_warp_f64(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      if (col < logical_out) {
        store_output(out + row * logical_out + col,
                     static_cast<float>(v[i] * double(__half2float(svh[col]))));
      }
    }
  }
}

}  // namespace
}  // namespace dense_splitk

int64_t linear_splitk_splits(int64_t rows, int64_t padded_out, int64_t sms,
                             int64_t bits_x2, int64_t codebook) {
  TORCH_CHECK(rows >= 0, "EXL3 rows must be nonnegative");
  TORCH_CHECK(sms > 0 && sms <= std::numeric_limits<int>::max(),
              "EXL3 multiprocessor count must be a positive int");
  check_padded(padded_out);
  if (!dense_splitk::served_format(bits_x2, codebook)) return 1;
  return dense_splitk::shape_splits(dense_gemm_shape(rows, padded_out, static_cast<int>(sms)),
                                    rows, padded_out);
}

void linear_out_splitk(const torch::Tensor& x, const torch::Tensor& trellis,
                       const torch::Tensor& suh, const torch::Tensor& svh,
                       int64_t bits_x2, int64_t codebook, int64_t logical_in,
                       int64_t logical_out, torch::Tensor& out,
                       torch::Tensor& workspace) {
  using namespace dense_splitk;
  // linear_out's checks.
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
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // With VLLM_EXL3_DENSE_MMA the tensor-core blocks take the launches they
  // cover first (dense_mma.h); everything else continues below unchanged.
  if (linear_out_mma(x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out,
                     workspace, sms)) {
    return;
  }
  const GemmShape shape = dense_gemm_shape(rows, padded_out, sms);
  if (!served_format(bits_x2, codebook) || shape_splits(shape, rows, padded_out) != 2) {
    linear_out(x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace);
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const auto* suh_ptr = reinterpret_cast<const half*>(suh.data_ptr<at::Half>());
  const auto* svh_ptr = reinterpret_cast<const half*>(svh.data_ptr<at::Half>());
  float* transformed = workspace.data_ptr<float>();
  // rows * padded_out floats, 16-byte aligned (padded_in is a multiple of 128).
  float* result = transformed + rows * padded_in;
  linear_input<<<warp_blocks(rows * (padded_in / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), suh_ptr, rows,
      logical_in, padded_in, transformed, reinterpret_cast<float4*>(result),
      rows * padded_out / 4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  SplitProblem problem{};
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
  problem.splits = 2;
  const int64_t block_rows = gemm_block_rows(shape);
  const int64_t items =
      (rows + block_rows - 1) / block_rows * (padded_out / gemm_block_cols(shape)) * 2;
  if (shape == GemmShape::kTiny) {
    launch_pipelined<TinyPipe>(problem, items, stream);
  } else {
    launch_pipelined<SmallPipe>(problem, items, stream);
  }

  with_linear_out(out, [&](auto* out_ptr) {
    linear_output<<<warp_blocks(rows * (padded_out / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
        result, svh_ptr, rows, padded_out, logical_out, out_ptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
