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
constexpr int kRouteThreads = 1024;
// Route kernel shared memory: three int arrays per local expert.
constexpr int64_t kMaxRouteExperts = 4096;

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

unsigned int warp_blocks(int64_t tasks) {
  return launch_blocks((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
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

// Groups the local assignments by local expert (one block, every step).
// Order inside an expert is arbitrary: each assignment's result depends only
// on its own row, never on its position or neighbours.
__global__ void __launch_bounds__(kRouteThreads)
route_assignments(Routing routing, int64_t assignments, int block_rows, int* table) {
  extern __shared__ int route_shared[];
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;              // counts, then fill cursors
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;   // experts + 1 entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) cursor[e] = 0;
  __syncthreads();
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
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
  }
  int* sorted = table + RouteLayout::sorted_offset();
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    if (e >= 0) sorted[atomicAdd(&cursor[e], 1)] = static_cast<int>(a);
  }
}

// One warp per (local assignment, 128-column block): x * suh, then H128.
__global__ void expert_input(const __nv_bfloat16* x, Routing routing, const int* table,
                             int64_t hidden, int64_t hpad, int projection, float* input) {
  const int lane = threadIdx.x & 31;
  const int* sorted = table + RouteLayout::sorted_offset();
  const int64_t blocks = hpad / 128;
  const int64_t tasks = int64_t(table[0]) * blocks;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < tasks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t assignment = sorted[task / blocks];
    const int64_t base = (task % blocks) * 128 + lane;
    const half* suh = expert_suh(routing, local_expert(routing, assignment), projection);
    const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      v[i] = col < hidden ? __fmul_rn(__bfloat162float(row[col]), __half2float(suh[col]))
                          : 0.0f;
    }
    hadamard128_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) input[assignment * hpad + base + 32 * i] = v[i];
  }
}

// One warp per (local assignment, 128-column block): gate and up output
// transforms (H128, svh), SiLU(gate) * up, down suh, then the down H128,
// written over the gate rows; gate and up clamped first by swiglu_clamp.
__global__ void activate_down_input(Routing routing, const int* table,
                                    int64_t intermediate, int64_t ipad, float* gate,
                                    const float* up, float limit) {
  const int lane = threadIdx.x & 31;
  const int* sorted = table + RouteLayout::sorted_offset();
  const int64_t blocks = ipad / 128;
  const int64_t tasks = int64_t(table[0]) * blocks;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < tasks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t assignment = sorted[task / blocks];
    const int64_t base = (task % blocks) * 128 + lane;
    const int64_t expert = local_expert(routing, assignment);
    const half* gate_svh = expert_svh(routing, expert, 0);
    const half* up_svh = expert_svh(routing, expert, 1);
    const half* down_suh = expert_suh(routing, expert, 2);
    float g[4];
    float u[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      g[i] = gate[assignment * ipad + base + 32 * i];
      u[i] = up[assignment * ipad + base + 32 * i];
    }
    hadamard128_warp(g, lane);
    hadamard128_warp(u, lane);
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      float gv = __fmul_rn(g[i], __half2float(gate_svh[col]));
      float uv = __fmul_rn(u[i], __half2float(up_svh[col]));
      v[i] = 0.0f;
      if (col < intermediate) {
        swiglu_clamp(gv, uv, limit);
        const float silu = gv / __fadd_rn(1.0f, expf(-gv));
        v[i] = __fmul_rn(__fmul_rn(silu, uv), __half2float(down_suh[col]));
      }
    }
    hadamard128_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) gate[assignment * ipad + base + 32 * i] = v[i];
  }
}

// One warp per (token, 128-column block): each local route's down output
// transform (H128, svh), then exactly one routing-weight FMA per route in
// route order, then BF16 rounding.
__global__ void combine_routes(const float* down, const float* weights, Routing routing,
                               int64_t rows, int64_t hidden, int64_t hpad,
                               __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  const int64_t blocks = hpad / 128;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < rows * blocks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t row = task / blocks;
    const int64_t base = (task % blocks) * 128 + lane;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int64_t route = 0; route < routing.topk; ++route) {
      const int64_t assignment = row * routing.topk + route;
      const int64_t expert = local_expert(routing, assignment);
      if (expert < 0) continue;
      const half* svh = expert_svh(routing, expert, 2);
      const float weight = weights[assignment];
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = down[assignment * hpad + base + 32 * i];
      hadamard128_warp(v, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float value = __fmul_rn(v[i], __half2float(svh[base + 32 * i]));
        sum[i] = __fmaf_rn(value, weight, sum[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      if (col < hidden) out[row * hidden + col] = __float2bfloat16_rn(sum[i]);
    }
  }
}

}  // namespace

int64_t moe_workspace_size(int64_t max_rows, int64_t hidden_size,
                           int64_t intermediate_size, int64_t topk,
                           int64_t local_experts) {
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  // FP32 rows (input/down result, gate, up) plus the int32 routing table.
  return workspace_elements(checked_add(
      checked_mul(assignments, checked_add(hpad, checked_mul(2, ipad))),
      RouteLayout::ints(assignments)));
}

void moe_out(const torch::Tensor& x, const torch::Tensor& topk_ids,
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
  TORCH_CHECK(experts <= kMaxRouteExperts, "EXL3 MoE has too many local experts");
  const int64_t assignments = checked_mul(rows, topk);
  TORCH_CHECK(assignments <= std::numeric_limits<int>::max(),
              "EXL3 MoE has too many routed assignments");
  const int64_t required = moe_workspace_size(rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
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
    int* table = reinterpret_cast<int*>(up + assignments * ipad);
    const GemmShape shape = moe_gemm_shape(assignments, expert_map.numel());
    const int64_t block_rows = gemm_block_rows(shape);
    const int64_t block_cols = gemm_block_cols(shape);
    route_assignments<<<1, kRouteThreads, (3 * experts + 1) * sizeof(int), stream>>>(
        routing, assignments, static_cast<int>(block_rows), table);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Row tiles: sum over experts of ceil(rows_e / block_rows).
    const int64_t max_tiles = std::min<int64_t>(
        assignments, assignments / block_rows + std::min<int64_t>(experts, assignments) + 1);
    GemmProblem problem{};
    problem.rows = assignments;
    problem.route = table;
    problem.pointers = routing.pointers;
    problem.metadata = routing.metadata;
    const auto* bf16_input = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    problem.c_stride = ipad;
    problem.k = static_cast<int>(hpad);
    problem.n = static_cast<int>(ipad);
    if (moe_fuses_gate_up(shape, assignments)) {
      // Few rows: gate and up in one launch, each block forming its rows'
      // x * suh H128 activations itself.
      problem.x = bf16_input;
      problem.hidden = hidden;
      problem.topk = topk;
      problem.c = gate;
      problem.c_up = up;
      problem.projection = 0;
      launch_gemm(problem, shape, 2 * max_tiles * (ipad / block_cols), stream);
      problem.x = nullptr;
    } else {
      for (int projection = 0; projection < 2; ++projection) {
        expert_input<<<warp_blocks(assignments * (hpad / 128)), 32 * kWarpsPerBlock, 0,
                       stream>>>(bf16_input, routing, table, hidden, hpad, projection, input);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        problem.a = input;
        problem.a_stride = hpad;
        problem.c = projection == 0 ? gate : up;
        problem.projection = projection;
        launch_gemm(problem, shape, max_tiles * (ipad / block_cols), stream);
      }
    }
    activate_down_input<<<warp_blocks(assignments * (ipad / 128)), 32 * kWarpsPerBlock, 0,
                          stream>>>(routing, table, intermediate_size, ipad, gate, up, limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    problem.a = gate;
    problem.a_stride = ipad;
    problem.c = input;
    problem.c_stride = hpad;
    problem.k = static_cast<int>(ipad);
    problem.n = static_cast<int>(hpad);
    problem.projection = 2;
    launch_gemm(problem, shape, max_tiles * (hpad / block_cols), stream);
  } else {
    routing.topk = 0;
  }
  combine_routes<<<warp_blocks(rows * (hpad / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
      input, topk_weights.data_ptr<float>(), routing, rows, hidden, hpad,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
