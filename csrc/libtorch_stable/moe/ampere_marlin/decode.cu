// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * Weight-streaming W4A16 MoE decode kernels.
 *
 * Reads vLLM's Marlin-packed uint4b8 / group-128 / bf16-scale layout directly:
 *   packed int32 [E][K/16][2N]: k16 row r, 64-col tile t, lane l, int32 j at
 *   ((t*32 + l)*4 + j); with c = l%4, g = l/4 the nibbles hold
 *     nib0 (k 2c,   n g)   nib4 (2c+1, g)   nib1 (2c+8, g)   nib5 (2c+9, g)
 *     nib2 (2c,   g+8)     nib6 (2c+1, g+8) nib3 (2c+8, g+8) nib7 (2c+9, g+8)
 *   (n relative to 64t + 16j) - exactly the mma.m16n8k16 A fragment of a
 *   16(n) x 16(k) block, so each lane's 16-byte load is its own fragment.
 *   scales bf16 [E][K/128][N]: inside each 64-col chunk, col c sits at
 *   8*(c%8) + c/8, so lane (g) needs the 8 contiguous scales 8g..8g+7.
 *
 * Sharded W13 and W2 share one GEMM kernel; whole-expert GEMMs preserve Marlin's
 * reduction grouping in a separate streaming kernel. Tokens (<= 8 per aligned block) are the
 * mma N dimension.  Every warp is independent: it takes warp-items (aligned block b,
 * TW adjacent 64-col tiles, k-slice ks of R k16 rows) from an integer work queue and
 * streams them as one continuous run of CR-row chunks.  Weights go through a per-lane
 * two-slot cp.async ring in shared memory: each lane copies exactly the 16-byte
 * fragments it later consumes, so no barrier or warp sync is needed and the next chunk
 * stays in flight during compute.  Scales and activation fragments (small,
 * L2-resident) are loaded with the same chunk into ping-pong registers.  Per-item
 * base pointers are computed once per item.
 * Dequant as Marlin does it: (q-8) exactly in bf16 (magic-number LOP3), times the
 * group scale in bf16 (one rounding, __hmul2), fp32 mma accumulation - the same
 * weight values and rounding points as Marlin. The reduction grouping is shape-specific.
 *
 *   w13: fp32 part[ks][slot][2N]; TP shards use four fixed splits, whole experts
 *        use the two-chain Marlin grouping and one full-sum plane (no atomics)
 *   act: fixed-order split-K sum -> bf16 (R1) -> clamp-silu-mul (R2, R3) -> h
 *   w2 : TP shards use one K chain; whole experts use the Marlin grouping above
 *        bf16(acc) (R4) * bf16(router w) -> bf16 (R5) -> c3[slot][K]
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

namespace ampere_marlin_decode {

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(c), "n"((0xf0 & 0xcc) | 0xaa));
  return r;
}

// nibbles at bits [0,4) and [16,20) of q -> bf16x2 (nib - 8), exact.
__device__ __forceinline__ __nv_bfloat162 deq2(uint32_t q) {
  uint32_t v = lop3_and_or(q, 0x000f000fu, 0x43004300u);
  const uint32_t sub = 0x43084308u;
  return __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v),
                 *reinterpret_cast<const __nv_bfloat162*>(&sub));
}

__device__ __forceinline__ uint32_t scl(__nv_bfloat162 v, __nv_bfloat162 s) {
  __nv_bfloat162 r = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
               "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

constexpr int WARPS = 4;  // per CTA (warps are independent; the CTA is only a container)

template <int CR, int TW>
struct Frags {            // one chunk's small operands, prefetched in registers
  uint32_t x[CR][2];      // activation B fragments (token g; k 2c..2c+1 and 2c+8..2c+9)
  int4 s[TW];             // per tile: 8 bf16 scales of lane group g (cols g, g+8 of each j)
};

// R: k16 rows per warp-item; CR: k16 rows per chunk (2-slot per-lane cp.async ring);
// TW: adjacent 64-col tiles per warp (they share the activation fragments).
// tgroups_log2 / ksplit_log2: log2 of N/64/TW and K/16/R (both powers of two).
// Items are handed out by an integer work counter (`ctr`, zero on entry): fast warps take
// more items, so per-SM speed differences and item-count quantisation leave no tail.  The
// result of an item does not depend on which warp computes it (deterministic).  Each GEMM
// zeroes the other GEMM's counter (`ctr_other`), which is idle while this one runs.
template <int R, int CR, int TW, int K, int N, bool W13>
__global__ void __launch_bounds__(WARPS * 32)
    moe_dec_gemm(const __nv_bfloat16* __restrict__ A, int64_t a_stride,
                 const int4* __restrict__ W, const int4* __restrict__ S,
                 const int32_t* __restrict__ sorted, const int32_t* __restrict__ eids,
                 const int32_t* __restrict__ ntpp, const float* __restrict__ topk_w,
                 int topk_log2, int n_slots, int tgroups_log2, int ksplit_log2,
                 int* __restrict__ ctr, int* __restrict__ ctr_other,
                 float* __restrict__ part, __nv_bfloat16* __restrict__ out) {
  static_assert(R % CR == 0 && 8 % CR == 0, "chunks inside one scale group");
  constexpr int CH = R / CR;
  constexpr int SLOT = CR * TW * 32;  // int4 per ring slot (all lanes)
  extern __shared__ int4 smem_raw[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, c = lane & 3;
  int4* const ring0 = smem_raw + warp * (2 * SLOT) + lane;
  int4* const ring1 = ring0 + SLOT;
  if (blockIdx.x == 0 && threadIdx.x == 0) *ctr_other = 0;

  constexpr int krows = K >> 4;
  constexpr int wrow = N >> 1, srow = N >> 3;  // compile-time: immediate address offsets
  const int items = (ntpp[0] >> 3) << (tgroups_log2 + ksplit_log2);

  auto split = [&](int item, int& b, int& ks, int& t0) {
    t0 = (item & ((1 << tgroups_log2) - 1)) * TW;
    const int rest = item >> tgroups_log2;
    ks = rest & ((1 << ksplit_log2) - 1);
    b = rest >> ksplit_log2;
  };

  // Load cursor, one chunk ahead of compute.  Per-item bases are computed once per grab.
  int l_item = 0, l_ch = 0;
  const int4* l_w = W;
  const int4* l_s = S;
  const uint32_t* l_x = reinterpret_cast<const uint32_t*>(A);
  auto grab = [&]() {
    int v = 0;
    if (lane == 0) v = atomicAdd(ctr, 1);
    l_item = __shfl_sync(0xffffffffu, v, 0);
    l_ch = 0;
    if (l_item < items) {
      int b, ks, t0;
      split(l_item, b, ks, t0);
      const int e0 = eids[b];
      const int e = e0 < 0 ? 0 : e0;
      const int r0 = ks * R;
      l_w = W + ((int64_t)e * krows + r0) * wrow + t0 * 32 + lane;
      l_s = S + ((int64_t)e * (K >> 7) + (r0 >> 3)) * srow + t0 * 8 + g;
      // Invalid tokens read row 0: their mma columns are independent and never stored.
      const int sid = sorted[b * 8 + g];
      const int arow = e0 >= 0 && sid < n_slots ? (W13 ? sid >> topk_log2 : sid) : 0;
      l_x = reinterpret_cast<const uint32_t*>(A + (int64_t)arow * a_stride + r0 * 16 + 2 * c);
    }
  };
  // chunk (l_item, l_ch): weights -> ring slot (cp.async), scales + activations -> f
  auto load = [&](int4* dst, Frags<CR, TW>& f) {
    const int4* src = l_w + l_ch * (CR * wrow);
#pragma unroll
    for (int i = 0; i < CR; i++)
#pragma unroll
      for (int u = 0; u < TW; u++)
        cp_async16(dst + (i * TW + u) * 32, src + i * wrow + u * 32, true);
    cp_async_commit();
    const int4* sp = l_s + ((l_ch * CR) >> 3) * srow;
#pragma unroll
    for (int u = 0; u < TW; u++) f.s[u] = __ldg(sp + u * 8);
    const uint32_t* ap = l_x + l_ch * CR * 8;
#pragma unroll
    for (int i = 0; i < CR; i++) {
      f.x[i][0] = __ldg(ap + i * 8);
      f.x[i][1] = __ldg(ap + i * 8 + 4);
    }
  };

  float acc[TW][4][4];
#pragma unroll
  for (int u = 0; u < TW; u++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[u][j][v] = 0.f;

  auto epilogue = [&](int item) {  // this warp's tiles of a finished item -> global
    int b, ks, t0;
    split(item, b, ks, t0);
    const bool ok = eids[b] >= 0;
    const int s0 = sorted[b * 8 + 2 * c];
    const int s1 = sorted[b * 8 + 2 * c + 1];
    const bool v0 = ok && s0 < n_slots, v1 = ok && s1 < n_slots;
    if constexpr (W13) {
      float* p0 = part + ((int64_t)ks * n_slots + s0) * N + t0 * 64 + g;
      float* p1 = part + ((int64_t)ks * n_slots + s1) * N + t0 * 64 + g;
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            if (v0) p0[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h];
            if (v1) p1[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h + 1];
          }
    } else {
      const float w0 = v0 ? __bfloat162float(__float2bfloat16(topk_w[s0])) : 0.f;
      const float w1 = v1 ? __bfloat162float(__float2bfloat16(topk_w[s1])) : 0.f;
      __nv_bfloat16* o0 = out + (int64_t)s0 * N + t0 * 64 + g;
      __nv_bfloat16* o1 = out + (int64_t)s1 * N + t0 * 64 + g;
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            if (v0)
              o0[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h])) * w0);
            if (v1)
              o1[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h + 1])) * w1);
          }
    }
#pragma unroll
    for (int u = 0; u < TW; u++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int v = 0; v < 4; v++) acc[u][j][v] = 0.f;
  };

  // One loop body (compact code): slot parity picks the ring slot at run time and the
  // prefetched fragments are copied, instead of a two-way unrolled ping-pong.
  Frags<CR, TW> cur, nxt;
  grab();
  int c_item = l_item, c_ch = 0;  // compute cursor
  if (l_item < items) load(ring0, cur);
  int par = 0;
  while (c_item < items) {
    cp_async_wait<0>();  // own lane's copies of the current chunk have landed
    if (++l_ch == CH) grab();
    const int4* cs = par ? ring1 : ring0;
    if (l_item < items) load(par ? ring0 : ring1, nxt);
#pragma unroll
    for (int i = 0; i < CR; i++) {
#pragma unroll
      for (int u = 0; u < TW; u++) {
        const int4 wv = cs[(i * TW + u) * 32];
        const uint32_t* qv = reinterpret_cast<const uint32_t*>(&wv);
        const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&cur.s[u]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t qj = qv[j];
          const __nv_bfloat162 sl = __low2bfloat162(s2[j]);   // col g   (pos 2j)
          const __nv_bfloat162 sh = __high2bfloat162(s2[j]);  // col g+8 (pos 2j+1)
          mma_bf16(acc[u][j], scl(deq2(qj), sl), scl(deq2(qj >> 8), sh),
                   scl(deq2(qj >> 4), sl), scl(deq2(qj >> 12), sh), cur.x[i][0], cur.x[i][1]);
        }
      }
    }
    if (++c_ch == CH) {
      epilogue(c_item);
      c_ch = 0;
      c_item = l_item;  // the load cursor wrapped to this item one chunk ago
    }
    cur = nxt;
    par ^= 1;
  }
  cp_async_wait<0>();
}

// Whole-expert GEMMs use the block-8 Marlin arithmetic partition: 128 output
// columns, 64 K values per tile, two K chains, and three reference CTAs per SM.
// The streaming launch geometry is independent of this reduction geometry.
template <int TW, bool W13>
__global__ void __launch_bounds__(WARPS * 32)
    moe_dec_whole(const __nv_bfloat16* __restrict__ A, int64_t a_stride,
                const int4* __restrict__ W, const int4* __restrict__ S,
                const int32_t* __restrict__ sorted, const int32_t* __restrict__ eids,
                const int32_t* __restrict__ ntpp, const float* __restrict__ topk_w,
                int n_slots, int topk_log2, int reference_grid, int* __restrict__ ctr,
                int* __restrict__ ctr_other, float* __restrict__ part,
                __nv_bfloat16* __restrict__ out) {
  constexpr int K = W13 ? 4096 : 2048, N = 4096, CR = 8;
  constexpr int K_TILES = K / 64, N_TILES = N / 128;
  constexpr int GROUPS = N / 64 / TW;
  constexpr int SLOT = CR * TW * 32;
  extern __shared__ int4 smem_raw[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, c = lane & 3;
  int4* const ring0 = smem_raw + warp * (2 * SLOT) + lane;
  int4* const ring1 = ring0 + SLOT;
  if (blockIdx.x == 0 && threadIdx.x == 0) *ctr_other = 0;

  const int blocks = ntpp[0] >> 3;
  const int tiles = blocks * N_TILES;
  int sk_tiles = tiles;
  if (tiles > reference_grid) {
    sk_tiles = tiles % reference_grid;
    if (3 * sk_tiles <= reference_grid) sk_tiles += reference_grid;
  }
  const int dp_tiles = tiles - sk_tiles;
  const int raw_iters = (K_TILES * sk_tiles + reference_grid - 1) / reference_grid;
  const int iters = 2 * ((raw_iters + 1) / 2);  // Keep group-128 boundaries.

  while (true) {
    int item = 0;
    if (lane == 0) item = atomicAdd(ctr, 1);
    item = __shfl_sync(0xffffffffu, item, 0);
    if (item >= blocks * GROUPS) break;
    const int b = item / GROUPS;
    const int t0 = (item % GROUPS) * TW;
    const int tile = b * N_TILES + t0 / 2;
    const int e0 = eids[b], e = e0 < 0 ? 0 : e0;
    const int sid = sorted[b * 8 + g];
    const int arow = e0 >= 0 && sid < n_slots ? (W13 ? sid >> topk_log2 : sid) : 0;
    const int4* const weights = W + (int64_t)e * (K / 16) * (N / 2) + t0 * 32 + lane;
    const int4* const scales = S + (int64_t)e * (K / 128) * (N / 8) + t0 * 8 + g;
    const uint32_t* const input =
        reinterpret_cast<const uint32_t*>(A + (int64_t)arow * a_stride + 2 * c);
    const int tile_start = (tile - dp_tiles) * K_TILES;
    float total[TW][4][4];

    // Marlin's global reduction starts with the highest-K stripe. Each lower
    // stripe then adds the accumulated higher stripes after its two-chain sum.
    int end = K_TILES;
    while (end > 0) {
      const int stripe_start = tile < dp_tiles ? 0 :
          ((tile_start + end - 1) / iters) * iters - tile_start;
      const int begin = stripe_start > 0 ? stripe_start : 0;
      const int row = begin * 4;
      const int chunks = (end - begin) * 4 / CR;
      float acc[2][TW][4][4];
#pragma unroll
      for (int q = 0; q < 2; q++)
#pragma unroll
        for (int u = 0; u < TW; u++)
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int v = 0; v < 4; v++) acc[q][u][j][v] = 0.f;

      auto load = [&](int chunk, int4* dst, Frags<CR, TW>& f) {
        const int r = row + chunk * CR;
        const int4* src = weights + r * (N / 2);
#pragma unroll
        for (int i = 0; i < CR; i++)
#pragma unroll
          for (int u = 0; u < TW; u++)
            cp_async16(dst + (i * TW + u) * 32, src + i * (N / 2) + u * 32, true);
        cp_async_commit();
#pragma unroll
        for (int u = 0; u < TW; u++) f.s[u] = __ldg(scales + (r / 8) * (N / 8) + u * 8);
#pragma unroll
        for (int i = 0; i < CR; i++) {
          f.x[i][0] = __ldg(input + (r + i) * 8);
          f.x[i][1] = __ldg(input + (r + i) * 8 + 4);
        }
      };
      Frags<CR, TW> cur, nxt;
      load(0, ring0, cur);
      for (int chunk = 0; chunk < chunks; chunk++) {
        cp_async_wait<0>();
        const int4* cs = chunk & 1 ? ring1 : ring0;
        if (chunk + 1 < chunks) load(chunk + 1, chunk & 1 ? ring0 : ring1, nxt);
#pragma unroll
        for (int i = 0; i < CR; i++) {
#pragma unroll
          for (int u = 0; u < TW; u++) {
            const int4 wv = cs[(i * TW + u) * 32];
            const uint32_t* qv = reinterpret_cast<const uint32_t*>(&wv);
            const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&cur.s[u]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              const uint32_t qj = qv[j];
              const __nv_bfloat162 sl = __low2bfloat162(s2[j]);
              const __nv_bfloat162 sh = __high2bfloat162(s2[j]);
              mma_bf16(acc[(i / 2) % 2][u][j], scl(deq2(qj), sl),
                       scl(deq2(qj >> 8), sh), scl(deq2(qj >> 4), sl),
                       scl(deq2(qj >> 12), sh), cur.x[i][0], cur.x[i][1]);
            }
          }
        }
        if (chunk + 1 < chunks) cur = nxt;
      }
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const float sum = acc[0][u][j][v] + acc[1][u][j][v];
            total[u][j][v] = end == K_TILES ? sum : sum + total[u][j][v];
          }
      end = begin;
    }

    const int s0 = sorted[b * 8 + 2 * c], s1 = sorted[b * 8 + 2 * c + 1];
    const bool v0 = e0 >= 0 && s0 < n_slots, v1 = e0 >= 0 && s1 < n_slots;
    float w0 = 0.f, w1 = 0.f;
    if constexpr (!W13) {
      w0 = v0 ? __bfloat162float(__float2bfloat16(topk_w[s0])) : 0.f;
      w1 = v1 ? __bfloat162float(__float2bfloat16(topk_w[s1])) : 0.f;
    }
#pragma unroll
    for (int u = 0; u < TW; u++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int col = t0 * 64 + g + 64 * u + 16 * j + 8 * h;
          if constexpr (W13) {
            if (v0) part[(int64_t)s0 * N + col] = total[u][j][2 * h];
            if (v1) part[(int64_t)s1 * N + col] = total[u][j][2 * h + 1];
          } else {
            if (v0) out[(int64_t)s0 * N + col] = __float2bfloat16(
                __bfloat162float(__float2bfloat16(total[u][j][2 * h])) * w0);
            if (v1) out[(int64_t)s1 * N + col] = __float2bfloat16(
                __bfloat162float(__float2bfloat16(total[u][j][2 * h + 1])) * w1);
          }
        }
  }
}

// h[slot][n] = bf16(bf16(silu(min(G, L))) * clamp(U, -L, L)), G/U = bf16(sum_ks part).
// One block per slot, one thread per 4 columns: all 2 * KS float4 partial loads are issued
// before the fixed-order (ks = 0..KS-1) sums, so the kernel pays one L2 round trip.
template <int KS>
__global__ void moe_dec_act(const float* __restrict__ part, const int32_t* __restrict__ ids,
                            int n_slots, int Nh, float limit, __nv_bfloat16* __restrict__ h) {
  const int slot = blockIdx.x;
  if (ids[slot] < 0) return;
  const int n = threadIdx.x * 4;
  if (n >= Nh) return;
  const int64_t row = (int64_t)n_slots * 2 * Nh;
  const float* p = part + (int64_t)slot * 2 * Nh + n;
  float4 gv[KS], uv[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    gv[ks] = __ldg(reinterpret_cast<const float4*>(p + ks * row));
    uv[ks] = __ldg(reinterpret_cast<const float4*>(p + ks * row + Nh));
  }
  float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    if constexpr (KS == 1) {
      g[0] = gv[ks].x; g[1] = gv[ks].y; g[2] = gv[ks].z; g[3] = gv[ks].w;
      u[0] = uv[ks].x; u[1] = uv[ks].y; u[2] = uv[ks].z; u[3] = uv[ks].w;
    } else {
      g[0] += gv[ks].x; g[1] += gv[ks].y; g[2] += gv[ks].z; g[3] += gv[ks].w;
      u[0] += uv[ks].x; u[1] += uv[ks].y; u[2] += uv[ks].z; u[3] += uv[ks].w;
    }
  }
  __nv_bfloat16 r[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float gf = fminf(__bfloat162float(__float2bfloat16(g[i])), limit);
    const float uf = fmaxf(fminf(__bfloat162float(__float2bfloat16(u[i])), limit), -limit);
    const float sl = __bfloat162float(__float2bfloat16(gf / (1.0f + expf(-gf))));
    r[i] = __float2bfloat16(sl * uf);
  }
  *reinterpret_cast<uint2*>(h + (int64_t)slot * Nh + n) = *reinterpret_cast<const uint2*>(r);
}

static int log2_exact(int64_t v) {
  TORCH_CHECK(v > 0 && (v & (v - 1)) == 0, "expected a power of two, got ", v);
  return __builtin_ctzll((unsigned long long)v);
}

template <int R, int CR, int TW, int KK, int NN, bool W13>
static int64_t launch(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                      at::Tensor const& sorted, at::Tensor const& eids, at::Tensor const& ntpp,
                      at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int64_t K,
                      int64_t N, int cap, at::Tensor& ctr, at::Tensor& out) {
  constexpr int T = WARPS * 32;
  constexpr int SMEM = WARPS * 2 * CR * TW * 32 * (int)sizeof(int4);
  TORCH_CHECK(K == KK && N == NN, "shape not built: K=", K, " N=", N);
  auto kern = moe_dec_gemm<R, CR, TW, KK, NN, W13>;
  static const int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T, SMEM);
    return n > 0 ? n : 1;
  }();
  const int ksplit = (int)(K / 16 / R);
  TORCH_CHECK(ksplit * R * 16 == K, "K must be a multiple of R k16 rows");
  if (W13) {
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.numel() >= ksplit * n_slots * N,
                "w13 partial buffer");
  } else {
    TORCH_CHECK(ksplit == 1 && out.scalar_type() == at::kBFloat16 &&
                    out.numel() >= n_slots * N,
                "w2 output");
  }
  TORCH_CHECK((N / 64) % TW == 0, "N must be a multiple of TW tiles");
  const int tg_log2 = log2_exact(N / 64 / TW), ksplit_log2 = log2_exact(ksplit);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_blk = std::min<int64_t>(sorted.numel() / 8, n_slots);
  const int64_t max_items = max_blk * ksplit * (N / 64 / TW);
  const int64_t grid = std::min<int64_t>((max_items + WARPS - 1) / WARPS, (int64_t)sms * (cap > 0 ? std::min(cap, occ) : occ));
  if (grid > 0)
    kern<<<grid, T, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)a.data_ptr(), a.stride(0), (const int4*)w.data_ptr(),
        (const int4*)s.data_ptr(), sorted.data_ptr<int32_t>(), eids.data_ptr<int32_t>(),
        ntpp.data_ptr<int32_t>(), topk_w.data_ptr<float>(), log2_exact(topk), (int)n_slots,
        tg_log2, ksplit_log2, ctr.data_ptr<int32_t>() + (W13 ? 0 : 1),
        ctr.data_ptr<int32_t>() + (W13 ? 1 : 0), W13 ? out.data_ptr<float>() : nullptr,
        W13 ? nullptr : (__nv_bfloat16*)out.data_ptr());
  return ksplit;
}

template <int TW, bool W13>
static int64_t launch_whole(at::Tensor const& a, at::Tensor const& w,
                                at::Tensor const& s, at::Tensor const& sorted,
                                at::Tensor const& eids, at::Tensor const& ntpp,
                                at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int cap,
                                at::Tensor& ctr, at::Tensor& out) {
  constexpr int N = 4096, CR = 8, T = WARPS * 32;
  constexpr int SMEM = WARPS * 2 * CR * TW * 32 * (int)sizeof(int4);
  TORCH_CHECK(out.scalar_type() == (W13 ? at::kFloat : at::kBFloat16) &&
                  out.numel() >= n_slots * N,
              "whole-expert output buffer");
  auto kern = moe_dec_whole<TW, W13>;
  static const int occ = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T, SMEM);
    return n > 0 ? n : 1;
  }();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_blk = std::min<int64_t>(sorted.numel() / 8, n_slots);
  const int64_t max_items = max_blk * (N / 64 / TW);
  const int64_t grid = std::min<int64_t>((max_items + WARPS - 1) / WARPS,
      (int64_t)sms * (cap > 0 ? std::min(cap, occ) : occ));
  if (grid > 0)
    kern<<<grid, T, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)a.data_ptr(), a.stride(0), (const int4*)w.data_ptr(),
        (const int4*)s.data_ptr(), sorted.data_ptr<int32_t>(), eids.data_ptr<int32_t>(),
        ntpp.data_ptr<int32_t>(), topk_w.data_ptr<float>(), (int)n_slots, log2_exact(topk),
        3 * sms, ctr.data_ptr<int32_t>() + (W13 ? 0 : 1),
        ctr.data_ptr<int32_t>() + (W13 ? 1 : 0), W13 ? out.data_ptr<float>() : nullptr,
        W13 ? nullptr : (__nv_bfloat16*)out.data_ptr());
  return 1;
}

}  // namespace ampere_marlin_decode

using namespace ampere_marlin_decode;

// w13: part [ksplit, n_slots, N] fp32;  w2: out [n_slots, N] bf16.  Returns ksplit.
// rows = k16 rows per warp item (w13: 64; w2: K/16).  cfg & 15 picks (CR, TW)
// from AMPERE_MARLIN_CFGS; cfg >> 4, if non-zero, caps the resident CTAs per SM.
int64_t ampere_marlin_decode_gemm(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                           at::Tensor const& sorted, at::Tensor const& eids,
                           at::Tensor const& ntpp, at::Tensor const& topk_w, int64_t topk,
                           int64_t n_slots, int64_t K, int64_t N, bool w13, int64_t rows,
                           int64_t cfg, at::Tensor& ctr, at::Tensor& out) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.stride(1) == 1 && a.stride(0) % 2 == 0,
              "a: bf16, unit inner stride, even row stride");
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous(), "weights contiguous");
  TORCH_CHECK(K % 128 == 0 && N % 64 == 0, "K % 128, N % 64");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && sorted.scalar_type() == at::kInt &&
                  eids.scalar_type() == at::kInt && ntpp.scalar_type() == at::kInt,
              "routing dtypes");
  TORCH_CHECK(ctr.scalar_type() == at::kInt && ctr.numel() >= 2 && ctr.is_contiguous(),
              "ctr: 2 int32 work counters, zero before the first call");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  if (w13 && K == 4096 && N == 4096 && rows == 64) {
    if ((cfg & 15) == 0)
      return launch_whole<1, true>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots,
                                 (int)(cfg >> 4), ctr, out);
    if ((cfg & 15) == 1)
      return launch_whole<2, true>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots,
                                 (int)(cfg >> 4), ctr, out);
  }
#define AMPERE_MARLIN_CASE(RR, WW, CFG, CR, TW, KK, NN)                                                      \
  if (K == KK && N == NN && rows == RR && w13 == WW && (cfg & 15) == CFG)                                           \
    return launch<RR, CR, TW, KK, NN, WW>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots, K, N,  \
                                  (int)(cfg >> 4), ctr, out);
#define AMPERE_MARLIN_CFGS(RR, WW, KK, NN)                                                           \
  AMPERE_MARLIN_CASE(RR, WW, 0, 8, 1, KK, NN)                                                     \
  AMPERE_MARLIN_CASE(RR, WW, 1, 8, 2, KK, NN)
  AMPERE_MARLIN_CFGS(64, true, 4096, 1024)
  AMPERE_MARLIN_CFGS(32, false, 512, 4096)
  if (!w13 && K == 2048 && N == 4096 && rows == 128) {
    if ((cfg & 15) == 0)
      return launch_whole<1, false>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots,
                                    (int)(cfg >> 4), ctr, out);
    if ((cfg & 15) == 1)
      return launch_whole<2, false>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots,
                                    (int)(cfg >> 4), ctr, out);
  }
#undef AMPERE_MARLIN_CFGS
#undef AMPERE_MARLIN_CASE
  TORCH_CHECK(false, "unsupported rows=", rows, " w13=", w13, " cfg=", cfg);
}

void ampere_marlin_decode_act(at::Tensor const& part, at::Tensor const& ids, int64_t ksplit,
                       int64_t n_slots, int64_t Nh, double limit, at::Tensor& h) {
  const at::cuda::OptionalCUDAGuard guard(part.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((ksplit == 1 || ksplit == 4) && Nh % 4 == 0 && Nh / 4 <= 1024,
              "act: ksplit 1 or 4, Nh % 4 == 0");
  if (n_slots > 0) {
    if (ksplit == 1)
      moe_dec_act<1><<<n_slots, (int)(Nh / 4), 0, stream>>>(
          part.data_ptr<float>(), ids.data_ptr<int32_t>(), (int)n_slots, (int)Nh, (float)limit,
          (__nv_bfloat16*)h.data_ptr());
    else
      moe_dec_act<4><<<n_slots, (int)(Nh / 4), 0, stream>>>(
          part.data_ptr<float>(), ids.data_ptr<int32_t>(), (int)n_slots, (int)Nh, (float)limit,
          (__nv_bfloat16*)h.data_ptr());
  }
}

TORCH_LIBRARY_FRAGMENT(_ampere_marlin_C, m) {
  m.def(
      "decode_gemm(Tensor a, Tensor w, Tensor s, Tensor sorted, Tensor eids, Tensor ntpp, "
      "Tensor topk_w, int topk, int n_slots, int K, int N, bool w13, int rows, int cfg, "
      "Tensor(c!) ctr, Tensor(o!) out) -> int");
  m.def(
      "decode_act(Tensor part, Tensor ids, int ksplit, int n_slots, int Nh, float limit, "
      "Tensor(h!) h) -> ()");
}

TORCH_LIBRARY_IMPL(_ampere_marlin_C, CUDA, m) {
  m.impl("decode_gemm", &ampere_marlin_decode_gemm);
  m.impl("decode_act", &ampere_marlin_decode_act);
}
