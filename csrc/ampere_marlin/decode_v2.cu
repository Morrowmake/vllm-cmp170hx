// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * Weight-streaming W4A16 MoE decode GEMM, second revision (tensor-parallel shard N=512).
 *
 * Registered as decode_gemm_v2; the activation step stays decode_act_orig.  Same item
 * split, plane count and per-output k order as decode_gemm_orig, so for scales with
 * 2^-110 <= |s| < 2^10 (or s = 0) the output is bitwise equal to it (see the dequant note
 * below for the one further condition); the changes are the
 * one-FMA dequant below, the first work item per warp taken without an atomic, 4-row
 * chunk instantiations, and zero activation fragments for padding tokens (ZB).  The w13
 * projection is split four ways along K, so its fp32 sums are formed in a different
 * order from the released Marlin kernels.
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
 * One GEMM kernel serves both projections.  Tokens (<= 8 per aligned block) are the
 * mma N dimension.  Every warp is independent: it takes warp-items (aligned block b,
 * TW adjacent 64-col tiles, k-slice ks of R k16 rows) from an integer work queue and
 * streams them as one continuous run of CR-row chunks.  Weights go through a per-lane
 * two-slot cp.async ring in shared memory: each lane copies exactly the 16-byte
 * fragments it later consumes, so no barrier or warp sync is needed and the next chunk
 * stays in flight during compute.  Scales and activation fragments (small,
 * L2-resident) are loaded with the same chunk into ping-pong registers.  Per-item
 * base pointers are computed once per item.
 * Dequant to Marlin's weight values bf16((q-8)*s), exactly, with one bf16x2 FMA per two
 * weights (dq): a bf16 bit pattern b < 256 is exactly b * 2^-133 (subnormal/normal
 * boundary), so a nibble ANDed into bits [0,4) or [4,8) of each half is q * 2^-133 or
 * 16q * 2^-133, and fma(that, s * 2^117 (or 2^113), -8s * 2^-16) is (q-8)*s*2^-16 rounded
 * once - bitwise bf16((q-8)*s) * 2^-16 for every scale 2^-110 <= |s| < 2^10 (or s = 0), up to
 * the sign of zero weights.  The fp32 mma accumulation is therefore the incumbent's scaled by
 * exactly 2^-16 as long as no nonzero product falls below 2^-110 and the mma keeps fp32
 * subnormals unflushed; the epilogues undo it exactly.  fp32 sums run in a different order
 * from the released Marlin kernels (see above), so the result stays allclose to them, not
 * bitwise.
 *
 *   w13: fp32 split-K partials part[ks][slot][2N] (deterministic, no atomics)
 *   act: fixed-order split-K sum -> bf16 (R1) -> clamp-silu-mul (R2, R3) -> h
 *   w2 : one k-slice; bf16(acc) (R4) * bf16(router w) -> bf16 (R5) -> c3[slot][K]
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

namespace ampere_marlin_decode_v2 {

__device__ __forceinline__ __nv_bfloat162 as_bf2(uint32_t v) {
  return *reinterpret_cast<const __nv_bfloat162*>(&v);
}
__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<const uint32_t*>(&v);
}

// bf16x2 power-of-two factors (exact scalings): 2^117, 2^113, -2^-13.
constexpr uint32_t kP117 = 0x7A007A00u, kP113 = 0x78007800u, kNeg8x2m16 = 0xB900B900u;
constexpr float kUnscale = 65536.0f;  // 2^16: undoes the dequant's 2^-16

// Per-scale constants of one bf16x2 scale pair (s_lo, s_hi): b1 = s * 2^117 (nibble at
// bits [0,4)), b16 = s * 2^113 (nibble at bits [4,8)), c = -8s * 2^-16.
struct DqScale {
  __nv_bfloat162 b1, b16, c;
};
__device__ __forceinline__ DqScale dq_scale(__nv_bfloat162 s) {
  return {__hmul2(s, as_bf2(kP117)), __hmul2(s, as_bf2(kP113)), __hmul2(s, as_bf2(kNeg8x2m16))};
}
// two weights: nibble bits v (already masked into [0,4) or [4,8) of each half) -> bf16x2
// bf16((q-8)*s) * 2^-16 via one fma.rn (b = matching b1/b16 half, c = matching c half).
__device__ __forceinline__ uint32_t dq(uint32_t v, __nv_bfloat162 b, __nv_bfloat162 c) {
  return as_u32(__hfma2(as_bf2(v), b, c));
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
// ZB: padding tokens get zero activation fragments (cfg 9; see grab).
template <int R, int CR, int TW, int K, int N, bool W13, bool ZB>
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
  // Every warp's first item is its global warp id; later items come from the counter, offset
  // by the warp count (claims stay on demand).
  const int nwarps = gridDim.x * WARPS;
  int l_item = blockIdx.x * WARPS + warp, l_ch = 0;
  const int4* l_w = W;
  const int4* l_s = S;
  const uint32_t* l_x = reinterpret_cast<const uint32_t*>(A);
  bool l_xok = false;
  auto grab = [&](bool first) {
    if (!first) {
      int v = 0;
      if (lane == 0) v = atomicAdd(ctr, 1) + nwarps;
      l_item = __shfl_sync(0xffffffffu, v, 0);
    }
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
      // ZB: they get zero B fragments instead (no load; zero operands cut tensor-core
      // switching energy, which matters under the power cap at large M).
      const int sid = sorted[b * 8 + g];
      l_xok = e0 >= 0 && sid < n_slots;
      const int arow = l_xok ? (W13 ? sid >> topk_log2 : sid) : 0;
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
    const uint32_t xm = (!ZB || l_xok) ? 0xffffffffu : 0u;
#pragma unroll
    for (int i = 0; i < CR; i++) {
      f.x[i][0] = __ldg(ap + i * 8) & xm;
      f.x[i][1] = __ldg(ap + i * 8 + 4) & xm;
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
            if (v0) p0[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h] * kUnscale;
            if (v1) p1[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h + 1] * kUnscale;
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
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h] * kUnscale)) * w0);
            if (v1)
              o1[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h + 1] * kUnscale)) * w1);
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
  grab(true);
  int c_item = l_item, c_ch = 0;  // compute cursor
  if (l_item < items) load(ring0, cur);
  int par = 0;
  while (c_item < items) {
    cp_async_wait<0>();  // own lane's copies of the current chunk have landed
    if (++l_ch == CH) grab(false);
    const int4* cs = par ? ring1 : ring0;
    if (l_item < items) load(par ? ring0 : ring1, nxt);
    DqScale ds[TW][4];  // this chunk's scale group (CR <= 8 rows: one group per chunk)
#pragma unroll
    for (int u = 0; u < TW; u++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        ds[u][j] = dq_scale(reinterpret_cast<const __nv_bfloat162*>(&cur.s[u])[j]);
#pragma unroll
    for (int i = 0; i < CR; i++) {
#pragma unroll
      for (int u = 0; u < TW; u++) {
        const int4 wv = cs[(i * TW + u) * 32];
        const uint32_t* qv = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          // nibbles: [0,4) k 2c (col g), [4,8) k 2c+8 (g), [8,12) k 2c (g+8), [12,16) k 2c+8 (g+8);
          // the high half holds k + 1 of each.  Low half of dq_scale = col g, high = col g+8.
          const uint32_t qj = qv[j], qh = qj >> 8;
          const DqScale& d = ds[u][j];
          const __nv_bfloat162 b1l = __low2bfloat162(d.b1), b1h = __high2bfloat162(d.b1);
          const __nv_bfloat162 b16l = __low2bfloat162(d.b16), b16h = __high2bfloat162(d.b16);
          const __nv_bfloat162 cl = __low2bfloat162(d.c), ch = __high2bfloat162(d.c);
          mma_bf16(acc[u][j], dq(qj & 0x000f000fu, b1l, cl), dq(qh & 0x000f000fu, b1h, ch),
                   dq(qj & 0x00f000f0u, b16l, cl), dq(qh & 0x00f000f0u, b16h, ch),
                   cur.x[i][0], cur.x[i][1]);
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

static int log2_exact(int64_t v) {
  TORCH_CHECK(v > 0 && (v & (v - 1)) == 0, "expected a power of two, got ", v);
  return __builtin_ctzll((unsigned long long)v);
}

template <int R, int CR, int TW, int KK, int NN, bool W13, bool ZB>
static int64_t launch(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
                      at::Tensor const& sorted, at::Tensor const& eids, at::Tensor const& ntpp,
                      at::Tensor const& topk_w, int64_t topk, int64_t n_slots, int64_t K,
                      int64_t N, int cap, at::Tensor& ctr, at::Tensor& out) {
  constexpr int T = WARPS * 32;
  constexpr int SMEM = WARPS * 2 * CR * TW * 32 * (int)sizeof(int4);
  TORCH_CHECK(K == KK && N == NN, "shape not built: K=", K, " N=", N);
  auto kern = moe_dec_gemm<R, CR, TW, KK, NN, W13, ZB>;
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

}  // namespace ampere_marlin_decode_v2

using namespace ampere_marlin_decode_v2;

// w13: part [ksplit, n_slots, N] fp32;  w2: out [n_slots, N] bf16.  Returns ksplit.
// rows = k16 rows per warp item (w13: 64; w2: K/16).  cfg & 15 picks (CR, TW, ZB)
// from AMPERE_MARLIN_CFGS; cfg >> 4, if non-zero, caps the resident CTAs per SM.
// Built for the N=512 shard only (w13 K=4096 N=1024, w2 K=512 N=4096).
int64_t ampere_marlin_decode_v2_gemm(at::Tensor const& a, at::Tensor const& w, at::Tensor const& s,
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
#define AMPERE_MARLIN_CASE(RR, WW, CFG, CR, TW, ZB, KK, NN)                                          \
  if (K == KK && N == NN && rows == RR && w13 == WW && (cfg & 15) == CFG)                           \
    return launch<RR, CR, TW, KK, NN, WW, ZB>(a, w, s, sorted, eids, ntpp, topk_w, topk, n_slots, K, \
                                              N, (int)(cfg >> 4), ctr, out);
#define AMPERE_MARLIN_CFGS(RR, WW, KK, NN)                    \
  AMPERE_MARLIN_CASE(RR, WW, 0, 8, 1, false, KK, NN)          \
  AMPERE_MARLIN_CASE(RR, WW, 1, 8, 2, false, KK, NN)          \
  AMPERE_MARLIN_CASE(RR, WW, 2, 4, 1, false, KK, NN)          \
  AMPERE_MARLIN_CASE(RR, WW, 3, 4, 2, false, KK, NN)          \
  AMPERE_MARLIN_CASE(RR, WW, 9, 8, 2, true, KK, NN)
  AMPERE_MARLIN_CFGS(64, true, 4096, 1024)
  AMPERE_MARLIN_CFGS(32, false, 512, 4096)
#undef AMPERE_MARLIN_CFGS
#undef AMPERE_MARLIN_CASE
  TORCH_CHECK(false, "unsupported rows=", rows, " w13=", w13, " cfg=", cfg);
}

TORCH_LIBRARY_FRAGMENT(_ampere_marlin_C, m) {
  m.def(
      "decode_gemm_v2(Tensor a, Tensor w, Tensor s, Tensor sorted, Tensor eids, Tensor ntpp, "
      "Tensor topk_w, int topk, int n_slots, int K, int N, bool w13, int rows, int cfg, "
      "Tensor(c!) ctr, Tensor(o!) out) -> int");
}

TORCH_LIBRARY_IMPL(_ampere_marlin_C, CUDA, m) {
  m.impl("decode_gemm_v2", &ampere_marlin_decode_v2_gemm);
}
