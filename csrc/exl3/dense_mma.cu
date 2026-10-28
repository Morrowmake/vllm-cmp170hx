// Tensor-core dense EXL3 linear for 1..32 rows (VLLM_EXL3_DENSE_MMA).
//
// With the process switch on (set_dense_mma, set once by the Python side
// before any launch or graph capture), linear_out_splitk hands every launch
// to linear_out_mma first. A launch the tensor-core blocks cover (formats
// in EXL3_DENSE_MMA_FORMATS, a matrix in kServedShapes, 1..32 rows, a
// 16-byte aligned trellis, room for the block exponents in out; see
// served_plan and dense_mma_plan) runs:
// - linear_input_split: x * suh, H128 (linear_input's FP32 values), then
//   each H128 block of each row scaled by a power of two and split into
//   FP16 hi + lo in the m16n8k16 B-fragment order; with two K splits it
//   also zeroes the result;
// - mma_kernel: trellis tiles decoded in registers straight into m16n8k16 A
//   fragments, the products on the tensor cores (FP32 accumulate);
// - linear_output with the output H128 and svh in FP64, rounded once to
//   FP32 and then, for a BF16 out, to BF16 (linear_out_splitk's).
// Every other launch, including every launch of more than 32 rows, stays
// with linear_out_splitk (its split-K blocks or linear_out).
//
// Arithmetic (not linear_out's; the results differ in the last bits). W is
// exact in FP16. Each H128 block of each activation row is scaled by
// 2^(14 - e) (its max |v| in [2^14, 2^15)) and split into hi = rn16(v) and
// lo = rn16(v - hi). Per k16 slice D = W^T lo + 0, then D += W^T hi on the
// tensor core, then acc = fma(D, 2^(e - 14) of the slice's block, acc) in
// FP32 in slice order over a warp's K range (the product by a power of two
// is exact: one rounding per slice). The kw range sums of a block are added
// in K order in FP32; with two splits both halves are added atomically onto
// the zeroed result: fl(0 + a + b) == fl(0 + b + a), so results are
// deterministic. The plan, and with it a row's result, depends on the row
// count of the launch.
//
// The workspace is linear_out's (linear_workspace_size): the split
// activations take exactly the transformed input's rows * padded_in floats,
// the result follows them. The block scale exponents (rows rounded up to 8
// times padded_in / 128 int16) live at the start of out until linear_output
// overwrites it.
#include "dense_mma.h"

#include "common.h"
#include "codec.cuh"
#include "gemm.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace exl3 {
namespace dense_mma {
namespace {

// The formats this path serves (bits_x2, codebook): 5, 6 and 7 bits per
// weight with the mul1 codebook. Any other format keeps linear_out_splitk.
#define EXL3_DENSE_MMA_FORMATS(X) X(10, 2) X(12, 2) X(14, 2)

bool served_format(int64_t bits_x2, int64_t codebook) {
#define EXL3_SERVED(B, C) if (bits_x2 == B && codebook == C) return true;
  EXL3_DENSE_MMA_FORMATS(EXL3_SERVED)
#undef EXL3_SERVED
  return false;
}

// The matrices this path serves (padded_in, padded_out, bits_x2), codebook
// 2: the dense matrices of one tensor-parallel rank of Qwen3.8-Flash-Next at
// TP2, the only shapes its plans were measured and validated on (the Python
// gate also requires TP2, PP1 and 74 SMs). Any other matrix keeps
// linear_out_splitk.
constexpr int64_t kServedShapes[][3] = {
    {2560, 256, 14},     // attention k/v
    {2560, 640, 14},     // shared expert gate/up
    {2560, 640, 10},     // sparse-attention indexer q/k
    {2560, 1280, 12},    // drafter embedding / hidden-state projections
    {2560, 3072, 14},    // linear-attention z
    {2560, 5120, 14},    // linear-attention qkv
    {2560, 6144, 14},    // attention q (with output gate)
    {640, 2560, 14},     // shared expert down
    {3072, 2560, 14},    // output projections
    {2560, 124160, 12},  // LM head (half the vocabulary)
};

bool served_shape(int64_t k, int64_t n, int64_t bits_x2) {
  for (const auto& shape : kServedShapes) {
    if (shape[0] == k && shape[1] == n && shape[2] == bits_x2) return true;
  }
  return false;
}

// Shared-memory attributes and occupancy are per device: each instantiation
// sets and queries them once per device index below this bound.
constexpr int kMmaMaxDevices = 64;

// A dense problem plus the tensor-core fields. The kernel parameter has the
// GemmProblem fields first and these after them.
struct MmaProblem : GemmProblem {
  // 2: a second block takes the second half of K for the same column
  // groups; both halves are added atomically onto c, which the caller
  // zeroes first. 0 or 1: no split.
  int splits;
  // The split activations, one uint4 per (row, k16 slice, q) as
  // linear_input_split stores them (k / 16 slices per row), per (H128
  // block, row) the biased FP32 exponent of the block's inverse scale
  // (block-major, rows rounded up to 8 per block), and the warps per column
  // group (K parts inside a block; splits: across blocks).
  const uint4* split;
  const int16_t* block_exp;
  int mma_kw;
};

// How a launch runs on the tensor-core blocks: mt = 0 (not at all), or
// 2 / 4 column tiles per warp, kw warps per column group along K inside a
// block and splits (1 or 2) blocks along K.
struct MmaPlan {
  int mt;
  int kw;
  int splits;
};

// Two mul1 codebook values at once, as the FP16 pair (low half = s0's
// value): the format's scalar arithmetic, FP16 fused multiply-add with
// round-to-nearest. The sum of the four product bytes, 0..1020, sits in the
// FP16 mantissa of 1024.0 (0x6400): exactly 1024 + sum. The second sum lands
// in the high half first; the first adds onto it (no carry: sums < 2^16).
__device__ __forceinline__ uint32_t mul1_pair_h2(uint32_t s0, uint32_t s1) {
  const uint32_t high = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0u) * 0x10000u + 0x64006400u;
  const uint32_t packed = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, high);
  const uint32_t scale = 0x1eee1eeeu;
  const uint32_t bias = 0xc931c931u;
  const __half2 value = __hfma2(*reinterpret_cast<const __half2*>(&packed),
                                *reinterpret_cast<const __half2*>(&scale),
                                *reinterpret_cast<const __half2*>(&bias));
  return *reinterpret_cast<const uint32_t*>(&value);
}

// Encoder lane `lane`'s eight FP16 values as four pairs: pair i holds values
// 2i and 2i + 1. Value j sits at tile row 2 * (lane & 3) + (j & 1) +
// 8 * ((j >> 1) & 1) and tile column (lane >> 2) + 8 * (j >> 2): the inverse
// of the tensor-core lane permutation (codec.cuh decode_lane's order).
template <int kBitsX2, int kCodebook>
__device__ __forceinline__ void decode_lane_h2(
    const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords], int shift, uint32_t (&pair)[4]) {
  static_assert(kCodebook == 2, "the tensor-core blocks serve the mul1 codebook only");
  using Ring = TileRing<kBitsX2>;
  uint32_t u[Ring::kWindowWords];
#pragma unroll
  for (int i = 0; i < Ring::kWindowWords; ++i) {
    const uint32_t next = i + 1 < Ring::kLoadWords ? words[i + 1] : 0u;
    u[i] = __funnelshift_l(next, words[i], shift);
  }
  uint32_t state[8];
  state[0] = window_state<Ring::rel(0), Ring::kWindowWords>(u);
  state[1] = window_state<Ring::rel(1), Ring::kWindowWords>(u);
  state[2] = window_state<Ring::rel(2), Ring::kWindowWords>(u);
  state[3] = window_state<Ring::rel(3), Ring::kWindowWords>(u);
  state[4] = window_state<Ring::rel(4), Ring::kWindowWords>(u);
  state[5] = window_state<Ring::rel(5), Ring::kWindowWords>(u);
  state[6] = window_state<Ring::rel(6), Ring::kWindowWords>(u);
  state[7] = window_state<Ring::rel(7), Ring::kWindowWords>(u);
#pragma unroll
  for (int i = 0; i < 4; ++i) pair[i] = mul1_pair_h2(state[2 * i], state[2 * i + 1]);
}

// Tensor-core dense blocks at decode row counts (rows <= kMmaMaxRows;
// dense_mma_plan). A warp owns MT adjacent 16-column tiles (a column group)
// over one range of K and steps through it in k16 slices: each lane decodes
// its share of the slice's MT trellis tiles straight into m16n8k16 A
// fragments (decode_lane_h2 already yields the fragment order; A = W^T, 16
// output columns x 16 k), and the activation rows form B (n8 = 8 rows per
// fragment, NT fragments). A block of max(kw, 4) warps holds its warps / kw
// column groups, each split over kw warps along K; with splits == 2 a second
// block takes the second half of K for the same groups.
// Most warps per block (kw <= kMmaWarps); kw < 4 runs blocks of 4 warps.
constexpr int kMmaWarps = 8;
constexpr int kMmaMinBlockWarps = 4;
constexpr int kMmaThreads = 32 * kMmaWarps;
constexpr int kMmaMaxRows = 32;

__device__ __forceinline__ void mma_f16_16816(float (&d)[4], const uint32_t (&a)[4],
                                              uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Each warp streams its k16 slices through a private ring of kMmaStages
// stages (cp.async, kMmaStages - 1 slices in flight): the MT trellis tiles
// (contiguous in memory), the lane-ordered split activations and the lanes'
// block scale exponents.
constexpr int kMmaStages = 4;

template <int kBitsX2, int NT, int MT>
struct MmaStage {
  static constexpr int kTileWords = TileRing<kBitsX2>::kWords;
  uint32_t words[MT * kTileWords];
  uint4 act[NT][32];
  // Per lane the biased exponents of its result rows 2q, 2q + 1 for the
  // slice's H128 block.
  uint32_t scale[NT][32];
};

// Dynamic shared memory of a block of `warps` warps: the warps' rings, reused
// after the K loop for the kw-way reduction (MT * NT * 4 floats per lane and
// warp, which is never more than a ring).
template <int kBitsX2, int NT, int MT>
constexpr int mma_smem_bytes(int warps) {
  return warps * kMmaStages * int(sizeof(MmaStage<kBitsX2, NT, MT>));
}

__device__ __forceinline__ void cp_async16(uint32_t smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t smem) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem) : "memory");
  return v;
}
__device__ __forceinline__ uint4 lds128(uint32_t smem) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(smem)
               : "memory");
  return v;
}

// A value the compiler cannot recompute (keeps lane constants in registers).
__device__ __forceinline__ uint32_t opaque_u32(uint32_t v) {
  asm volatile("" : "+r"(v));
  return v;
}

template <int kBitsX2, int kCodebook, int NT, int MT>
__global__ void __launch_bounds__(kMmaThreads) mma_kernel(MmaProblem p) {
  using Ring = TileRing<kBitsX2>;
  using Stage = MmaStage<kBitsX2, NT, MT>;
  constexpr int S = kMmaStages;
  static_assert(MT * NT * 4 * 32 * 4 <= S * int(sizeof(Stage)), "reduction exceeds a ring");
  // 16-byte pieces of the MT tiles of one slice (a tile is 16 * bits_x2 bytes).
  constexpr int kPieces = MT * kBitsX2;
  extern __shared__ __align__(16) unsigned char mma_smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int q = lane & 3;
  constexpr uint32_t kStageBytes = sizeof(Stage);
  constexpr uint32_t kActOffset = offsetof(Stage, act);
  constexpr uint32_t kScaleOffset = offsetof(Stage, scale);
  // Lane constants as shared addresses into stage 0 of the warp's ring,
  // opaque to the compiler so that they stay in registers instead of being
  // recomputed every slice: the issue targets (trellis piece lane, B values),
  // the lane window's words of tile 0, and the window shift.
  const uint32_t ring = static_cast<uint32_t>(__cvta_generic_to_shared(mma_smem)) +
                        static_cast<uint32_t>(warp) * S * kStageBytes;
  const uint32_t issue_words = opaque_u32(ring + 16 * lane);
  const uint32_t act_lane = opaque_u32(ring + kActOffset + 16 * lane);
  const uint32_t scale_lane = opaque_u32(ring + kScaleOffset + 4 * lane);
  uint32_t read_word[Ring::kLoadWords];
  int shift;
  {
    const LaneWindow<kBitsX2> window(lane);
#pragma unroll
    for (int i = 0; i < Ring::kLoadWords; ++i) read_word[i] = opaque_u32(ring + 4 * window.word[i]);
    shift = static_cast<int>(opaque_u32(static_cast<uint32_t>(window.shift)));
  }
  const int kw = p.mma_kw;
  const int splits = p.splits > 1 ? 2 : 1;
  const int steps = p.k / 16;
  const int64_t groups = p.n / (16 * MT);
  // Block item: (block of column groups, split); warp: (group in block, K part).
  const int split = static_cast<int>(blockIdx.x % splits);
  const int64_t group = int64_t(blockIdx.x / splits) * int(blockDim.x / 32 / kw) + warp / kw;
  const int part = split * kw + warp % kw;
  const int first = static_cast<int>(int64_t(part) * steps / (kw * splits));
  const int last = static_cast<int>(int64_t(part + 1) * steps / (kw * splits));
  const bool active = group < groups;
  static_assert(kActOffset % 16 == 0 && kStageBytes % 16 == 0, "16-byte ring pieces");
  // This lane's B rows (row g of each fragment), and its result rows' scale
  // exponents: block_exp holds per H128 block one int16 per row of the row
  // count rounded up to 8, so rows 2q and 2q + 1 of a fragment are one word.
  const int rows8 = (static_cast<int>(p.rows) + 7) & ~7;
  const uint4* arow[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int row = nt * 8 + g;
    arow[nt] = row < p.rows ? p.split + int64_t(row) * steps * 4 + q : nullptr;
  }
  const uint32_t* esrc = reinterpret_cast<const uint32_t*>(p.block_exp) + q;
  const int64_t ct0 = group * MT;
  // Running sources of the next slice to issue and the ring stage it fills.
  const uint16_t* tsrc =
      p.trellis + (int64_t(first) * p.tiles_n + ct0) * Ring::kTileHalfwords + 8 * lane;
  const int64_t tstride = p.tiles_n * Ring::kTileHalfwords;
  const uint4* asrc[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) asrc[nt] = arow[nt] != nullptr ? arow[nt] + 4 * first : nullptr;
  uint32_t issue_offset = 0;
  int issue_slice = first;
  auto issue = [&]() {
#pragma unroll
    for (int j = 0; j < (kPieces + 31) / 32; ++j) {
      if (kPieces % 32 == 0 || lane + 32 * j < kPieces) {
        cp_async16(issue_words + issue_offset + 512 * j, tsrc + 256 * j);
      }
    }
    const uint32_t* eblock = esrc + (issue_slice >> 3) * (rows8 / 2);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      if (asrc[nt] != nullptr) {
        cp_async16(act_lane + issue_offset + 512 * nt, asrc[nt]);
        asrc[nt] += 4;
      }
      cp_async4(scale_lane + issue_offset + 128 * nt, eblock + 4 * nt);
    }
    ++issue_slice;
    tsrc += tstride;
    issue_offset = issue_offset + kStageBytes == S * kStageBytes ? 0 : issue_offset + kStageBytes;
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;
    }
  }
  if (active) {
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
      if (first + s < last) issue();
      cp_async_commit();
    }
    uint32_t read_offset = 0;
    for (int s = first; s < last; ++s) {
      // Every lane has finished reading the stage that slice s + S - 1 reuses.
      __syncwarp();
      if (s + S - 1 < last) issue();
      cp_async_commit();
      cp_async_wait<S - 1>();
      __syncwarp();
      const uint32_t offset = read_offset;
      read_offset = read_offset + kStageBytes == S * kStageBytes ? 0 : read_offset + kStageBytes;
      // b[nt]: hi pairs (k 2q.., 2q+8..), then lo pairs; zero past the rows.
      // inv[nt]: 2^(e - 14) of the slice's H128 block for rows 2q, 2q + 1
      // (garbage past the rows, whose D columns are zero and never stored).
      uint4 b[NT];
      float inv[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        b[nt] = arow[nt] != nullptr ? lds128(act_lane + offset + 512 * nt) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t exps = lds32(scale_lane + offset + 128 * nt);
        inv[nt][0] = __int_as_float(static_cast<int>((exps & 0xffffu) << 23));
        inv[nt][1] = __int_as_float(static_cast<int>((exps >> 16) << 23));
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t words[Ring::kLoadWords];
#pragma unroll
        for (int i = 0; i < Ring::kLoadWords; ++i) {
          words[i] = lds32(read_word[i] + offset + 4 * mt * Stage::kTileWords);
        }
        uint32_t pair[4];
        decode_lane_h2<kBitsX2, kCodebook>(words, shift, pair);
        // Pairs: 0 (k 2q.., col g), 1 (k 2q+8.., col g), 2 (k 2q.., col g+8),
        // 3 (k 2q+8.., col g+8); A regs: (g, k), (g+8, k), (g, k+8), (g+8, k+8).
        const uint32_t frag[4] = {pair[0], pair[2], pair[1], pair[3]};
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          mma_f16_16816(d, frag, b[nt].z, b[nt].w);
          mma_f16_16816(d, frag, b[nt].x, b[nt].y);
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[mt][nt][i] = __fmaf_rn(d[i], inv[nt][i & 1], acc[mt][nt][i]);
        }
      }
    }
    cp_async_wait<0>();
  }
  if (kw > 1) {
    // Range sums of the group's kw warps, added in K order by its first warp.
    float* red = reinterpret_cast<float*>(mma_smem);
    constexpr int kValues = MT * NT * 4;
    __syncthreads();
    if (warp % kw != 0) {
#pragma unroll
      for (int v = 0; v < kValues; ++v) red[(warp * kValues + v) * 32 + lane] = (&acc[0][0][0])[v];
    }
    __syncthreads();
    if (warp % kw == 0) {
      for (int w = warp + 1; w < warp + kw; ++w) {
#pragma unroll
        for (int v = 0; v < kValues; ++v) {
          (&acc[0][0][0])[v] = __fadd_rn((&acc[0][0][0])[v], red[(w * kValues + v) * 32 + lane]);
        }
      }
    }
  }
  if (!active || warp % kw != 0) return;
  // D column 2q + j is row 2q + j of the fragment.
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int row = nt * 8 + 2 * q + j;
      if (row >= p.rows) continue;
      float* out = p.c + int64_t(row) * p.c_stride + ct0 * 16 + g;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float lo = acc[mt][nt][j];
        const float hi = acc[mt][nt][2 + j];
        if (splits > 1) {
          atomicAdd(out + 16 * mt, lo);
          atomicAdd(out + 16 * mt + 8, hi);
        } else {
          out[16 * mt] = lo;
          out[16 * mt + 8] = hi;
        }
      }
    }
  }
}

// `device`, or the current device when it is negative.
int mma_device(int device) {
  if (device < 0) C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device < kMmaMaxDevices, "EXL3 MMA blocks need a device index below ",
              kMmaMaxDevices);
  return device;
}

// The kernel of one instantiation with its shared-memory attributes set
// (once per device; `device` is the current device): the rings are its whole
// shared memory, so it asks for the full carveout and lets registers bound
// the blocks per SM.
template <int kBitsX2, int kCodebook, int NT, int MT>
auto prepared_mma_kernel(int device) {
  auto* kernel = mma_kernel<kBitsX2, kCodebook, NT, MT>;
  static std::once_flag attributed[kMmaMaxDevices];
  std::call_once(attributed[device], [kernel] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        mma_smem_bytes<kBitsX2, NT, MT>(kMmaWarps)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
  });
  return kernel;
}

// Resident blocks per SM of an instantiation for blocks of `warps` warps
// (kMmaMinBlockWarps or kMmaWarps), queried once each per device (`device`
// is the current device).
template <int kBitsX2, int kCodebook, int NT, int MT>
int mma_blocks_per_sm(int warps, int device) {
  struct Occupancy {
    std::once_flag queried;
    int small = 0;
    int large = 0;
  };
  static Occupancy occupancy[kMmaMaxDevices];
  Occupancy& o = occupancy[device];
  std::call_once(o.queried, [&o, device] {
    auto* kernel = prepared_mma_kernel<kBitsX2, kCodebook, NT, MT>(device);
    auto query = [kernel](int w) {
      int blocks = 0;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, kernel, 32 * w, mma_smem_bytes<kBitsX2, NT, MT>(w)));
      return blocks;
    };
    o.small = query(kMmaMinBlockWarps);
    o.large = query(kMmaWarps);
  });
  return warps == kMmaWarps ? o.large : o.small;
}

// Calls f.template operator()<bits_x2, codebook, NT, MT>() for the
// instantiation of a served format, row count and column group width.
template <int MT, class F>
void mma_visit_rows(int format, int64_t rows, F&& f) {
  auto by_format = [&]<int NT>() {
    switch (format) {
#define EXL3_CASE(B, C) \
      case format_key(B, C): return f.template operator()<B, C, NT, MT>();
      EXL3_DENSE_MMA_FORMATS(EXL3_CASE)
#undef EXL3_CASE
      default:
        TORCH_CHECK(false, "EXL3 MMA blocks have no kernel for this rate/codebook combination");
    }
  };
  switch ((rows + 7) / 8) {
    case 1: return by_format.template operator()<1>();
    case 2: return by_format.template operator()<2>();
    case 3: return by_format.template operator()<3>();
    default: return by_format.template operator()<4>();
  }
}
template <class F>
void mma_visit(int format, int64_t rows, int mt, F&& f) {
  if (mt == 4) return mma_visit_rows<4>(format, rows, f);
  TORCH_CHECK(mt == 2, "EXL3 MMA column group must be 2 or 4 tiles");
  return mma_visit_rows<2>(format, rows, f);
}

int mma_block_warps(int kw) { return std::max(kw, kMmaMinBlockWarps); }

int64_t mma_blocks(int64_t n, int mt, int kw, int splits) {
  const int64_t groups = n / (16 * mt);
  const int per_block = mma_block_warps(kw) / kw;
  return (groups + per_block - 1) / per_block * splits;
}

// The LM-head regime: at least this many 4-tile column groups per SM need no
// K split; the fewest k16 slices per K part otherwise.
constexpr int kMmaWideGroupsPerSm = 8;
constexpr int kMmaMinSteps = 8;

// Every dense launch of 1..32 rows (16-byte aligned trellis).
// Wide matrices (>= kMmaWideGroupsPerSm 4-tile groups per SM) run 4-tile
// groups over all of K. Otherwise 2-tile groups and the most K parts (16 =
// 8 warps x 2 blocks, then 8, 4, 2) with >= kMmaMinSteps slices per part
// whose blocks all fit on the SMs at once (measured on the Qwen TP2 shapes:
// a second wave or an SM with one block more than the others costs more than
// the extra parallelism gains). The 2-tile plans query the occupancy on
// `device` (the current device when negative), which must be current.
MmaPlan dense_mma_plan(int64_t rows, int64_t k, int64_t n, int format, int sms,
                       const void* trellis, int device) {
  MmaPlan plan{};
  if (rows <= 0 || rows > kMmaMaxRows || k % 128 != 0 || n % 64 != 0) return plan;
  if (reinterpret_cast<uintptr_t>(trellis) % 16 != 0) return plan;
  if (n / 64 >= int64_t(kMmaWideGroupsPerSm) * sms) return MmaPlan{4, 1, 1};
  device = mma_device(device);
  const int64_t steps = k / 16;
  for (int parts = 16; parts >= 1; parts /= 2) {
    if (parts > 1 && steps / parts < kMmaMinSteps) continue;
    const int kw = std::min(parts, kMmaWarps);
    const int splits = parts / kw;
    int per_sm = 0;
    mma_visit(format, rows, 2, [&]<int B, int C, int NT, int MT>() {
      per_sm = mma_blocks_per_sm<B, C, NT, MT>(mma_block_warps(kw), device);
    });
    if (parts == 1 || mma_blocks(n, 2, kw, splits) <= int64_t(sms) * per_sm) {
      plan = MmaPlan{2, kw, splits};
      break;
    }
  }
  return plan;
}

// `device` is the current device.
void launch_mma(const MmaProblem& problem, int mt, cudaStream_t stream, int device) {
  TORCH_CHECK(problem.route == nullptr && problem.rows > 0 && problem.rows <= kMmaMaxRows &&
                  problem.n % 64 == 0 && problem.k % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(problem.trellis) % 16 == 0 &&
                  problem.mma_kw >= 1 && problem.mma_kw <= kMmaWarps &&
                  kMmaWarps % problem.mma_kw == 0,
              "EXL3 MMA blocks need a dense problem of 1..32 rows and a 16-byte aligned trellis");
  const int splits = problem.splits > 1 ? 2 : 1;
  const int warps = mma_block_warps(problem.mma_kw);
  const unsigned int grid =
      static_cast<unsigned int>(mma_blocks(problem.n, mt, problem.mma_kw, splits));
  mma_visit(problem.format, problem.rows, mt, [&]<int B, int C, int NT, int MT>() {
    auto* kernel = prepared_mma_kernel<B, C, NT, MT>(device);
    kernel<<<grid, 32 * warps, mma_smem_bytes<B, NT, MT>(warps), stream>>>(problem);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int kWarpsPerBlock = 8;

unsigned int warp_blocks(int64_t tasks) {
  return launch_blocks((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

// Measured (graph replay, rows 1-32, K 2560-3072): 4-warp blocks of one warp
// per (row, H128 block) take 2.5-3.4 us; one block of K / 128 warps per row
// with a row-wide max took 4.1-4.9 us.
constexpr int kSplitWarpsPerBlock = 4;

// The tensor-core blocks' input: one warp per (row, 128-column block),
// kSplitWarpsPerBlock warps per block, with linear_input's exact FP32
// values. The block's max |v| sets a power-of-two scale 2^(14 - e) (max
// scaled into [2^14, 2^15)); each scaled value is split into FP16
// hi = rn(v) and lo = rn(v - hi) (the residual is exact in FP32). Stored per
// (row, k16 slice, q = 0..3) as one uint4 in the m16n8k16 B-fragment order:
// hi pairs (k 2q, 2q + 1), (2q + 8, 2q + 9), then the lo pairs.
// block_exp[block * rows8 + row] = 113 + e (rows8: rows rounded up to 8),
// the biased FP32 exponent of 2^(e - 14), which the GEMM multiplies the
// block's slice sums by. With `zeroed` != nullptr the launch also zeroes
// zero_vectors float4s there (the split GEMM's accumulation target).
__global__ void linear_input_split(const __nv_bfloat16* x, const half* suh, int64_t rows,
                                   int64_t logical_in, int64_t padded_in, uint32_t* split,
                                   int16_t* block_exp, float4* zeroed, int64_t zero_vectors) {
  if (zeroed != nullptr) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < zero_vectors;
         i += int64_t(gridDim.x) * blockDim.x) {
      zeroed[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  const int lane = threadIdx.x & 31;
  const int64_t blocks = padded_in / 128;
  const int64_t task = int64_t(blockIdx.x) * kSplitWarpsPerBlock + (threadIdx.x >> 5);
  if (task >= rows * blocks) return;
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
  for (int distance = 16; distance > 0; distance >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, distance));
  }
  // Zero and subnormal maxima clamp to e = -112, so that both powers of two stay normal.
  const int e = max((__float_as_int(m) >> 23) - 127, -112);
  const float scale = __int_as_float((141 - e) << 23);
  if (lane == 0) {
    block_exp[(task % blocks) * ((rows + 7) & ~int64_t(7)) + row] = static_cast<int16_t>(113 + e);
  }
  const int64_t steps = padded_in / 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float s = __fmul_rn(v[i], scale);
    const __half hi = __float2half_rn(s);
    const __half lo = __float2half_rn(__fsub_rn(s, __half2float(hi)));
    // Lanes lane and lane ^ 1 hold k and k ^ 1: the even lane stores the hi
    // pair, the odd one the lo pair.
    const uint16_t mine = __half_as_ushort((lane & 1) ? hi : lo);
    const uint16_t other = static_cast<uint16_t>(__shfl_xor_sync(0xffffffffu, uint32_t(mine), 1));
    const uint16_t own = __half_as_ushort((lane & 1) ? lo : hi);
    const uint32_t pair = (lane & 1) ? (uint32_t(other) | (uint32_t(own) << 16))
                                     : (uint32_t(own) | (uint32_t(other) << 16));
    const int64_t k = base + 32 * i;
    const int kk = static_cast<int>(k & 15);
    const int q = (kk & 7) >> 1;
    const int word = (kk >> 3) + ((lane & 1) ? 2 : 0);
    split[((row * steps + (k >> 4)) * 4 + q) * 4 + word] = pair;
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

// The process switch and the diagnostic plan override (0: none, else
// mt << 8 | kw << 4 | splits), both set before any launch or graph capture.
std::atomic<bool> g_enabled{false};
std::atomic<int> g_override{0};

// dense_mma_plan for a served format and matrix (kServedShapes) whose block
// exponents fit in out ((rows rounded up to 8) * k / 128 int16 within rows *
// logical_out elements, each BF16 or FP32 and so at least one int16; out
// 4-byte aligned: the GEMM copies word pairs), with the
// diagnostic override applied to every launch the plan puts on the
// tensor-core blocks.
MmaPlan served_plan(int64_t rows, int64_t k, int64_t n, int64_t logical_out, int64_t bits_x2,
                    int64_t codebook, int sms, const void* trellis, const void* out,
                    int device) {
  if (!served_format(bits_x2, codebook)) return MmaPlan{};
  if (!served_shape(k, n, bits_x2)) return MmaPlan{};
  const bool fits_exps = (rows + 7) / 8 * 8 * (k / 128) <= rows * logical_out &&
                         reinterpret_cast<uintptr_t>(out) % 4 == 0;
  if (!fits_exps) return MmaPlan{};
  const MmaPlan plan = dense_mma_plan(
      rows, k, n, format_key(static_cast<int>(bits_x2), static_cast<int>(codebook)), sms,
      trellis, device);
  const int forced = g_override.load(std::memory_order_relaxed);
  if (plan.mt == 0 || forced == 0) return plan;
  return MmaPlan{forced >> 8, (forced >> 4) & 15, forced & 15};
}

void set_dense_mma(bool enabled) { g_enabled.store(enabled, std::memory_order_relaxed); }

void set_dense_mma_shape_override(int64_t mt, int64_t kw, int64_t splits) {
  if (mt == 0 && kw == 0 && splits == 0) {
    g_override.store(0, std::memory_order_relaxed);
    return;
  }
  TORCH_CHECK((mt == 2 || mt == 4) && (kw == 1 || kw == 2 || kw == 4 || kw == 8) &&
                  (splits == 1 || splits == 2),
              "EXL3 MMA plan override needs mt 2 or 4, kw 1, 2, 4 or 8 and splits 1 or 2 "
              "(or 0, 0, 0 for none)");
  g_override.store(static_cast<int>(mt << 8 | kw << 4 | splits), std::memory_order_relaxed);
}

// Sets every instantiation's shared-memory attributes and queries its
// occupancy on the current device, so that no launch (and no graph capture)
// is an instantiation's first call there. Another device is prepared on its
// first plan or launch.
void dense_mma_prepare() {
  const int device = mma_device(-1);
#define EXL3_KEY(B, C) format_key(B, C),
  for (const int format : {EXL3_DENSE_MMA_FORMATS(EXL3_KEY)}) {
    for (const int64_t rows : {1, 9, 17, 25}) {
      for (const int mt : {2, 4}) {
        mma_visit(format, rows, mt, [&]<int B, int C, int NT, int MT>() {
          (void)mma_blocks_per_sm<B, C, NT, MT>(kMmaWarps, device);
        });
      }
    }
  }
#undef EXL3_KEY
}

// [mt, kw, splits] of a launch of `rows` rows of a 16-byte aligned matrix
// whose output is unpadded (logical_out == padded_out) and 4-byte aligned
// (mt 0: not on the tensor-core blocks). Wide matrices (the LM head) and
// launches outside the blocks need no device; the 2-tile plans query the
// occupancy of the current device.
std::vector<int64_t> linear_mma_plan(int64_t rows, int64_t padded_in, int64_t padded_out,
                                     int64_t sms, int64_t bits_x2, int64_t codebook) {
  TORCH_CHECK(rows >= 0, "EXL3 rows must be nonnegative");
  TORCH_CHECK(sms > 0 && sms <= std::numeric_limits<int>::max(),
              "EXL3 multiprocessor count must be a positive int");
  check_padded(padded_in);
  check_padded(padded_out);
  const MmaPlan plan = served_plan(rows, padded_in, padded_out, padded_out, bits_x2, codebook,
                                   static_cast<int>(sms), nullptr, nullptr, -1);
  return {plan.mt, plan.kw, plan.splits};
}

}  // namespace
}  // namespace dense_mma

bool linear_out_mma(const torch::Tensor& x, const torch::Tensor& trellis,
                    const torch::Tensor& suh, const torch::Tensor& svh,
                    int64_t bits_x2, int64_t codebook, int64_t logical_in,
                    int64_t logical_out, torch::Tensor& out,
                    torch::Tensor& workspace, int sms) {
  using namespace dense_mma;
  if (!g_enabled.load(std::memory_order_relaxed)) return false;
  const int64_t rows = x.size(0);
  const int64_t padded_in = trellis.size(0) * 16;
  const int64_t padded_out = trellis.size(1) * 16;
  const int device = mma_device(x.get_device());
  const MmaPlan plan = served_plan(rows, padded_in, padded_out, logical_out, bits_x2, codebook,
                                   sms, trellis.data_ptr(), out.data_ptr(), device);
  if (plan.mt == 0) return false;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  float* transformed = workspace.data_ptr<float>();
  // rows * padded_out floats, 16-byte aligned (padded_in is a multiple of 128).
  float* result = transformed + rows * padded_in;
  int16_t* block_exp = reinterpret_cast<int16_t*>(out.data_ptr());
  float4* zeroed = plan.splits > 1 ? reinterpret_cast<float4*>(result) : nullptr;
  const int64_t warps = rows * (padded_in / 128);
  linear_input_split<<<static_cast<unsigned int>((warps + kSplitWarpsPerBlock - 1) /
                                                 kSplitWarpsPerBlock),
                       32 * kSplitWarpsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()), rows, logical_in, padded_in,
      reinterpret_cast<uint32_t*>(transformed), block_exp, zeroed, rows * padded_out / 4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  MmaProblem problem{};
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
  problem.splits = plan.splits;
  problem.split = reinterpret_cast<const uint4*>(transformed);
  problem.block_exp = block_exp;
  problem.mma_kw = plan.kw;
  launch_mma(problem, plan.mt, stream, device);

  const auto* svh_ptr = reinterpret_cast<const half*>(svh.data_ptr<at::Half>());
  with_linear_out(out, [&](auto* out_ptr) {
    linear_output<<<warp_blocks(rows * (padded_out / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
        result, svh_ptr, rows, padded_out, logical_out, out_ptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}  // namespace exl3

// No tensor arguments: catch-all host functions. set_dense_mma and the
// override only store process-wide values; dense_mma_prepare and the 2-tile
// plans call the CUDA runtime on the current device.
TORCH_LIBRARY_FRAGMENT(_exl3_C, m) {
  m.def("set_dense_mma(bool enabled) -> ()", &exl3::dense_mma::set_dense_mma);
  m.def("set_dense_mma_shape_override(int mt, int kw, int splits) -> ()",
        &exl3::dense_mma::set_dense_mma_shape_override);
  m.def("dense_mma_prepare() -> ()", &exl3::dense_mma::dense_mma_prepare);
  m.def("linear_mma_plan(int rows, int padded_in, int padded_out, int sms, int bits_x2, "
        "int codebook) -> int[]",
        &exl3::dense_mma::linear_mma_plan);
}
