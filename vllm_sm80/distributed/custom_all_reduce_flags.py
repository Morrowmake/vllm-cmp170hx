# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write-based two-shot all-reduce with the ready signal carried in the data.

Why this exists
---------------
On PCIe-only GPUs with peer access (the NVIDIA CMP 170HX on PCIe Gen2 x16
is the case this was built for), `cross_device_reduce_2stage` spends most of a
small decode all-reduce in its two flag barriers: each barrier is a fenced
flag write to every peer and a poll, and a fenced round trip costs ~8 us on this
link. At four ranks a 4096-wide bf16 row (8 KiB) took ~20 us, of which the
barriers alone were ~10 us. Upstream's default kernel at two ranks,
`cross_device_reduce_1stage`, has the same two barriers.

Here no separate flag exists. Each rank pushes its data into the peers'
receive buffers with plain 16-byte stores, and a receiver knows a 16-byte pack
has arrived once none of its four 32-bit words is the sentinel 0xFFFFFFFF (the
buffers are pre-filled with it). Two shots, as in the incumbent:

  1. reduce-scatter by push: partition q of my input goes to rank q;
  2. rank q reduces its partition and pushes the result to every peer;
  3. every rank gathers the other partitions from its own buffer.

Three buffer stages rotate. The stage the previous call used is re-filled with
the sentinel during the current call: a peer can be at most one call ahead
(it cannot finish a call without my data), so it can only be writing into the
next stage. A per-communicator device word holds the stage; CTA 0 advances it
after every CTA has read it, so a captured CUDA graph replays correctly.

Measured on 4x CMP 170HX against `cross_device_reduce_2stage` at 512 threads,
block limit 36, CUDA-graph replay of back-to-back calls, this launch rule
(256 threads, one per pack of the largest partition, at most 36 blocks):
8 KiB 20.5 -> 10.1 us, 32 KiB 23.9 -> 13.2, 64 KiB 30.9 -> 20.3,
128 KiB 48.4 -> 36.0, 256 KiB 86.4 -> 66.3. Two ranks use the same protocol
and launch rule; they have not been timed.

Exactness
---------
Four ranks: bitwise identical to `cross_device_reduce_2stage`: the element in
partition p is accumulated in fp32 in the order p, p+1, p+2, p+3 and rounded
once, as there, and the partitions are the same.

Two ranks: bitwise identical to `cross_device_reduce_1stage`, which upstream
runs at world size 2 unless VLLM_CUSTOM_ALLREDUCE_ALGO forces a kernel, and to
`cross_device_reduce_2stage`. Every kernel converts both bf16 inputs to fp32
exactly, adds them once in fp32 and rounds once to bf16; only the operand
order differs (partition p adds x_p + x_(1-p), the one-stage kernel x_0 + x_1),
and a single IEEE fp32 addition is commutative.

The sentinel is a pair of negative NaNs with full payload. An input word equal
to it is sent as a pair of canonical NaNs; any NaN input makes the fp32 sum the
canonical NaN, so the result bits do not change. A reduced word can never be
the sentinel (bf16 rounding of a NaN gives the canonical NaN), and is
sanitised anyway.

Only bf16, two or four ranks, compute capability 8.0 and the opt-in PCIe
peer-to-peer custom all-reduce. Messages above
allreduce/flags_max_bytes in the sm80 tuning table (default 256 KiB, the
largest measured) and other dtypes keep the incumbent kernel.

Compressed wire format and one-shot (four ranks)
------------------------------------------------
``two_shot_z`` (``FlagsAllreduce(..., compressed=True)``, taken by
``VLLM_SM80_ALLREDUCE_COMPRESS``, communicator.py) runs the same two shots
and the same owner-order FP32 sums, but moves each partition as warp blocks of
31 packs (248 values) in a lossless format (v2): the block's largest exponent
field T and a mask of wide lanes (5 header bytes), per value the sign and
7-bit mantissa (one byte), and per 8-value lane a code stream of 3 bits per
value (narrow: exponent T - c, every value within 7 octaves of T and none
with exponent field 0) or 4 bits (wide: 0 = exponent field 0, i.e. +-0 and
subnormals, c > 0 = exponent T + 1 - c). The slot carries 22 to 24 whole
16-byte quads (raw: 32); a block with exponent 255 (Inf, NaN), a value more
than 14 octaves below T, or an encoded word equal to the sentinel goes raw.
Served data: ~27 % fewer link bytes than raw. Every partial a rank receives
is the bits its peer sent, so every sum, and every output bit, is the
uncompressed kernel's. One warp encodes, or receives and decodes, one
(block, peer) pair through a per-warp shared-memory scratch.

``one_shot<4>`` (``all_reduce(..., one_shot=True)``, taken by the same switch
for messages up to allreduce/flags_one_shot_max_bytes): every rank pushes its
whole sanitized message to every peer and sums all four partials of every
element itself in the element's two-shot owner order (owner = the partition
holding it), rounded once: the two-shot's bits with one flag latency instead
of two. It shares the flags-in-data buffers, stage rotation and refill
bookkeeping of ``two_shot`` (it uses the stage prefix [0, 4 size)).

Source: a standalone TP4 all-reduce kernel, measured on captured served
all-reduce calls: KERNEL_SRC from "Lossless compressed wire format v2"
on is its wire.cuh (less Pack, defined above) and flags.cu (store_block,
recv_blocks2, one_shot, two_shot_z) verbatim; ``all_reduce_z`` and the
one-shot launch rule are its bindings'. The uncompressed kernel's text is
unchanged.

On 4x CMP 170HX (CUDA-graph replay, per call): uncompressed two-shot ->
one-shot 640 B 8.5 -> 6.2 us, 4 KiB 9.2 -> 7.2, 8 KiB 9.6 -> 9.1, 10 KiB
10.0 -> 10.2 (the one-shot's 3x egress); compressed v2 against uncompressed
loses at 1-2 rows of 5120, ties at 3-5 and wins from 6: below
allreduce/flags_raw_max_bytes (32 KiB) the uncompressed kernels stay.

Safety
------
A wait never gives up quietly. Every wait is bounded by
:data:`WAIT_S` (60 s, measured on the GPU's global timer, far above any
normal skew between ranks). On overrun the
first timed-out thread writes rank, call, stage, size, phase, the awaited
peer and the wait time into a host-mapped error record and the kernel
executes a trap, so the CUDA context fails and no partial result is ever
returned. A host thread polls the record (and it is checked again every 256
eager calls and at exit): it logs the record and ends the process. The other
ranks then time out the same way. The incumbent kernel waits without limit.

The extension is compiled on first use with torch's JIT loader and cached
outside the source tree, in $VLLM_CACHE_ROOT/custom_all_reduce_flags.
"""

import atexit
import ctypes
import math
import os
import sys
import threading

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

WORLD_SIZES = (2, 4)  # ranks the kernel is instantiated for
THREADS = 256  # launch rule chosen from a threads x blocks sweep (1..32 rows)
MAX_BLOCKS = 36
EPOCH_WORDS = 16
CHECK_EVERY = 256  # eager calls between host checks of the error record
EXIT_CODE = 70
Z_WORLD_SIZE = 4  # two_shot_z serves four ranks only
Z_BLOCK_PACKS = 31  # packs per warp block of the compressed format (lane 31: header)
Z_SLOT_WORDS = 128  # 32-bit words per block slot (32 lanes x 16 bytes)

KERNEL_SRC = r"""
#include <cuda_bf16.h>
#include <cstdint>

namespace arflags {

constexpr int kMaxRanks = 4;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kCanonicalPair = 0x7FFF7FFFu;

// Peer buffers; at two ranks p[2] and p[3] are null and never selected.
struct __align__(16) Ptrs {
  void* p[kMaxRanks];
};

// w[0] stage (0..2), w[1] CTAs that read w[0] this call, w[2..4] packs
// (two_shot) or block slots per area (two_shot_z) last written into each
// stage, w[8] calls completed, w[14] error record published, w[15] timed-out
// waits.
struct __align__(16) Epoch {
  uint32_t w[16];
};

// Owner of pack idx in cross_device_reduce_2stage with R ranks: part = size / R,
// ranks 0..R-2 own [r*part, (r+1)*part), rank R-1 the rest (all of it when
// part == 0).
template <int R>
__host__ __device__ __forceinline__ int part_len(int q, int size) {
  int part = size / R;
  return q == R - 1 ? size - (R - 1) * part : part;
}

__host__ __device__ __forceinline__ uint32_t sanitize_word(uint32_t w) {
  return w == kSentinel ? kCanonicalPair : w;
}

#define AFD __device__ __forceinline__

AFD uint4 ld_volatile16(const void* ptr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

AFD void st_volatile16(void* ptr, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

AFD bool dirty(const uint4& v) {
  return v.x == kSentinel || v.y == kSentinel || v.z == kSentinel ||
         v.w == kSentinel;
}

AFD uint4 sanitize(uint4 v) {
  v.x = sanitize_word(v.x);
  v.y = sanitize_word(v.y);
  v.z = sanitize_word(v.z);
  v.w = sanitize_word(v.w);
  return v;
}

// Error record in host-mapped memory (int64 words): [0] set, [1] rank,
// [2] stage, [3] call, [4] size in packs, [5] phase (1 reduce-scatter wait,
// 2 all-gather wait), [6] awaited peer, [7] waited ns, [8] block, [9] thread,
// [10] limit ns, [11] host acknowledgement (set by the host after logging).
struct WaitCtx {
  Epoch* ep;
  long long* err;
  unsigned long long limit_ns;
  int rank;
  int stage;
  int size;
};

AFD unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

AFD void st_release_sys_u64(long long* p, long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

AFD long long ld_acquire_sys_u64(const long long* p) {
  long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// Overrun. The first timed-out thread writes the record, publishes it with a
// system-scope release, then waits (at most kAckNs) for the host to
// acknowledge that it has logged it (word 11). Every other timed-out thread
// waits (at most kAckNs + 1 s) until the record is published, so no thread
// traps before the record is complete. Then all trap: the CUDA context fails
// and the caller never sees a partial result.
constexpr unsigned long long kAckNs = 2000000000ull;
__device__ __noinline__ void wait_failed(WaitCtx c, int phase, int peer,
                                        unsigned long long waited) {
  volatile unsigned int* published = &c.ep->w[14];
  if (atomicAdd(&c.ep->w[15], 1u) == 0u) {
    volatile long long* e = c.err;
    e[1] = c.rank;
    e[2] = c.stage;
    e[3] = c.ep->w[8];
    e[4] = c.size;
    e[5] = phase;
    e[6] = peer;
    e[7] = (long long)waited;
    e[8] = blockIdx.x;
    e[9] = threadIdx.x;
    e[10] = (long long)c.limit_ns;
    __threadfence_system();
    st_release_sys_u64(c.err, 1);
    __threadfence_system();
    *published = 1u;
    __threadfence();
    const unsigned long long t0 = now_ns();
    while (ld_acquire_sys_u64(c.err + 11) == 0 && now_ns() - t0 < kAckNs) {
    }
  } else {
    const unsigned long long t0 = now_ns();
    while (*published == 0u && now_ns() - t0 < kAckNs + 1000000000ull) {
    }
  }
  __trap();
}

// Buffer pointers in registers, selected by static index only (a dynamically
// indexed kernel parameter would be copied to local memory).
template <int R>
struct Regs {
  static_assert(R == 2 || R == 4, "two or four ranks");
  void* p0;
  void* p1;
  void* p2;
  void* p3;
  AFD explicit Regs(const Ptrs& b)
      : p0(b.p[0]), p1(b.p[1]), p2(b.p[2]), p3(b.p[3]) {}
  AFD void* operator[](int q) const {
    if constexpr (R == 2) {
      return q == 0 ? p0 : p1;
    } else {
      return q == 0 ? p0 : (q == 1 ? p1 : (q == 2 ? p2 : p3));
    }
  }
};

template <int R>
AFD uint4 pick(const uint4 (&v)[R], int q) {
  if constexpr (R == 2) {
    return q == 0 ? v[0] : v[1];
  } else {
    return q == 0 ? v[0] : (q == 1 ? v[1] : (q == 2 ? v[2] : v[3]));
  }
}

union Pack {
  uint4 u;
  __nv_bfloat16 h[8];
};

// The incumbent's packed_reduce over pointers rotated to the owner.
template <int R>
AFD uint4 reduce_owner_order(const uint4 (&v)[R], int owner) {
  float acc[8];
  Pack x;
  x.u = pick<R>(v, owner);
#pragma unroll
  for (int e = 0; e < 8; e++) acc[e] = __bfloat162float(x.h[e]);
#pragma unroll
  for (int j = 1; j < R; j++) {
    x.u = pick<R>(v, (owner + j) & (R - 1));
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] += __bfloat162float(x.h[e]);
  }
#pragma unroll
  for (int e = 0; e < 8; e++) x.h[e] = __float2bfloat16(acc[e]);
  return x.u;
}

// Waits for the R - 1 peer packs of one owned index.
template <int R>
AFD void wait_peers(const uint4* base, int rank_stride, int rank,
                    uint4 (&v)[R], const WaitCtx& c) {
  bool ready[R];
#pragma unroll
  for (int s = 0; s < R; s++) ready[s] = (s == rank);
  int remaining = R - 1;
  const unsigned long long t0 = now_ns();
  while (remaining) {
#pragma unroll
    for (int s = 0; s < R; s++) {
      if (!ready[s]) {
        uint4 x = ld_volatile16(base + s * rank_stride);
        if (!dirty(x)) {
          v[s] = x;
          ready[s] = true;
          --remaining;
        }
      }
    }
    if (remaining) {
      unsigned long long waited = now_ns() - t0;
      if (waited > c.limit_ns) {
        int peer = 0;
#pragma unroll
        for (int s = R - 1; s >= 0; s--)
          if (!ready[s]) peer = s;
        wait_failed(c, 1, peer, waited);
      }
    }
  }
}

AFD uint4 wait1(const uint4* ptr, int peer, const WaitCtx& c) {
  uint4 x = ld_volatile16(ptr);
  const unsigned long long t0 = now_ns();
  while (dirty(x)) {
    unsigned long long waited = now_ns() - t0;
    if (waited > c.limit_ns) wait_failed(c, 2, peer, waited);
    x = ld_volatile16(ptr);
  }
  return x;
}

// R ranks, R a power of two. size: message length in 16-byte packs. Layout per
// stage (L = largest partition): reduce-scatter slot s at s*L, all-gather
// slot q at (R+q)*L. stage_packs >= 2*R*L.
template <int R>
__global__ void __launch_bounds__(512, 1)
    two_shot(Ptrs bufs, const uint4* __restrict__ in, uint4* __restrict__ out,
             Epoch* ep, int rank, int size, int stage_packs,
             unsigned long long limit_ns, long long* err) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int stage = ep->w[0];
  const int dirty_stage = (stage + 2) % 3;
  const int dirty_size = ep->w[2 + dirty_stage];
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&ep->w[1], 1u);
  const Regs<R> b(bufs);
  const WaitCtx wc{ep, err, limit_ns, rank, stage, size};

  const int part = size / R;
  const int L = part_len<R>(R - 1, size);

  for (int idx = tid; idx < L; idx += stride) {
#pragma unroll
    for (int i = 1; i < R; i++) {
      int q = (rank + i) & (R - 1);
      if (idx < part_len<R>(q, size)) {
        uint4* dst = reinterpret_cast<uint4*>(b[q]) + stage * stage_packs +
                     rank * L + idx;
        st_volatile16(dst, sanitize(in[q * part + idx]));
      }
    }
  }

  uint4* self = reinterpret_cast<uint4*>(b[rank]);
  uint4* dirty_base = self + dirty_stage * stage_packs;
  const uint4 sent = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
  for (int idx = tid; idx < dirty_size; idx += stride) dirty_base[idx] = sent;

  const uint4* cur = self + stage * stage_packs;
  const int mine = part_len<R>(rank, size);
  for (int idx = tid; idx < mine; idx += stride) {
    uint4 own = sanitize(in[rank * part + idx]);
    uint4 v[R];
#pragma unroll
    for (int s = 0; s < R; s++) v[s] = own;
    wait_peers<R>(cur + idx, L, rank, v, wc);
    uint4 red = sanitize(reduce_owner_order<R>(v, rank));
    out[rank * part + idx] = red;
#pragma unroll
    for (int i = 1; i < R; i++) {
      int q = (rank + i) & (R - 1);
      uint4* dst = reinterpret_cast<uint4*>(b[q]) + stage * stage_packs +
                   (R + rank) * L + idx;
      st_volatile16(dst, red);
    }
  }

  for (int idx = tid; idx < L; idx += stride) {
#pragma unroll
    for (int i = 1; i < R; i++) {
      int q = (rank + i) & (R - 1);
      if (idx < part_len<R>(q, size))
        out[q * part + idx] = wait1(cur + (R + q) * L + idx, q, wc);
    }
  }

  if (tid == 0) {
    volatile uint32_t* arrived = &ep->w[1];
    while (*arrived < gridDim.x) {
    }
    ep->w[2 + stage] = 2 * R * L;
    ep->w[1] = 0;
    ep->w[8] += 1;
    ep->w[0] = (stage + 1) % 3;
  }
}

template __global__ void two_shot<2>(Ptrs, const uint4* __restrict__,
                                     uint4* __restrict__, Epoch*, int, int, int,
                                     unsigned long long, long long*);
template __global__ void two_shot<4>(Ptrs, const uint4* __restrict__,
                                     uint4* __restrict__, Epoch*, int, int, int,
                                     unsigned long long, long long*);

// Lossless compressed wire format v2 and the kernels that use it (four ranks,
// VLLM_SM80_ALLREDUCE_COMPRESS). From here to the end of the namespace: the
// all-reduce search's wire.cuh (less Pack, defined above) and its flags.cu
// store_block, recv_blocks2, one_shot and two_shot_z, verbatim.
constexpr int kLanes = 32;
constexpr int kBlockPacks = 31;
constexpr int kSlotWords = 4 * kLanes;

AFD uint4 shfl4(const uint4& v, int src) {
  return make_uint4(__shfl_sync(0xFFFFFFFFu, v.x, src), __shfl_sync(0xFFFFFFFFu, v.y, src),
                    __shfl_sync(0xFFFFFFFFu, v.z, src), __shfl_sync(0xFFFFFFFFu, v.w, src));
}

AFD uint32_t half_of(const uint4& v, int i) {
  const uint32_t w = (i >> 1) == 0 ? v.x : (i >> 1) == 1 ? v.y : (i >> 1) == 2 ? v.z : v.w;
  return (i & 1) ? (w >> 16) : (w & 0xFFFFu);
}

// This lane's 16-byte share of a block slot (index `at`) and whether it is stored.
struct Quad {
  uint4 v;
  int at;
  bool store;
};

AFD void add_pack(float (&acc)[8], uint4 u) {
  Pack x;
  x.u = u;
#pragma unroll
  for (int e = 0; e < 8; e++) acc[e] += __bfloat162float(x.h[e]);
}

// Format v2: one block of 31 packs per warp, in a slot of 128 words (a byte
// stream):
//   compressed: byte 0 = T, the block's largest exponent field (0..254, so
//     word 0 is never the sentinel), bytes 1-4 = the wide-lane mask (bit l:
//     lane l codes in 4 bits; bit 31 clear = compressed), words 2 + 2l and
//     3 + 2l = lane l's sign / 7-bit mantissa bytes, and the code stream: the
//     code fields in lane order, 3 bytes (8 x 3 bits) for a narrow lane, 4
//     (8 x 4 bits) for a wide one; stream bytes 0-2 sit at slot bytes 5-7, byte
//     k >= 3 at slot byte 253 + k. Wide code 0 = exponent field 0 (+-0,
//     subnormals), c > 0 = exponent T + 1 - c; narrow code c = exponent T - c
//     (a lane is narrow when it has no zero exponent field and every value is
//     within 7 octaves of T). The slot carries z2_quads() whole quads
//     (22..24; the tail is zero-padded: partial 16-byte stores cost more link
//     time than the padding).
//   raw (mask bit 31 set: quad 0 = (0, 0x80, 0, 0)): quad (l + 1) & 31 = lane
//     l's sanitized pack. Taken when a value has exponent 255, a code would
//     exceed 15, or a compressed word equals the sentinel.
// The receiver checks every word it reads for the sentinel.
constexpr uint32_t kZ2Sentinel = 0xFFFFFFFFu;
constexpr uint32_t kZ2RawW1 = 0x80u;                              // word 1 of a raw slot's quad 0
constexpr int kZ2FixedBytes = 256 + 3 * kBlockPacks - 3;         // 346: every lane narrow
constexpr int kZ2PollQuads = (kZ2FixedBytes + kBlockPacks + 15) / 16;   // 24: the most a compressed slot has
constexpr int kZ2Words = 128;                                   // warp scratch words

AFD bool z2_raw(uint32_t w1) { return w1 & kZ2RawW1; }
AFD uint32_t z2_mask(uint32_t w0, uint32_t w1) { return (w0 >> 8) | ((w1 & 0x7Fu) << 24); }
AFD int z2_quads(uint32_t w0, uint32_t w1) {
  return z2_raw(w1) ? kLanes : (kZ2FixedBytes + __popc(z2_mask(w0, w1)) + 15) >> 4;
}
AFD int z2_pos(int k) { return k < 3 ? 5 + k : 253 + k; }

// Warp-collective. v: this lane's sanitized pack (zero for lane 31 and past
// the partition); sm: the warp's kZ2Words scratch words. A compressed slot
// longer than MaxQuads goes raw instead (raw word 0 is free for the caller).
template <int MaxQuads = kZ2PollQuads>
AFD Quad encode2(const uint4& v, int lane, uint32_t* sm) {
  uint32_t T = 0u;
#pragma unroll
  for (int i = 0; i < 8; i++) T = max(T, (half_of(v, i) >> 7) & 0xFFu);
  T = __reduce_max_sync(0xFFFFFFFFu, T);
  uint32_t maxc = 0u, c3 = 0u, c4 = 0u, b0 = 0u, b1 = 0u;
  bool zero = false;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint32_t h = half_of(v, i);
    const uint32_t e = (h >> 7) & 0xFFu;
    const uint32_t c = e ? T + 1u - e : 0u;
    const uint32_t byte = ((h >> 8) & 0x80u) | (h & 0x7Fu);
    maxc = max(maxc, c);
    zero |= e == 0u;
    c3 |= ((c - 1u) & 7u) << (3 * i);
    c4 |= c << (4 * i);
    if (i < 4) b0 |= byte << (8 * i);
    else b1 |= byte << (8 * (i - 4));
  }
  if (T != 255u && !__any_sync(0xFFFFFFFFu, maxc > 15u)) {
    const bool wide = zero || maxc > 8u;
    const uint32_t mask = __ballot_sync(0xFFFFFFFFu, wide) & 0x7FFFFFFFu;
    sm[64 + lane] = 0u;
    if (lane == kLanes - 1) sm[1] = mask >> 24;
    __syncwarp();
    if (lane < kBlockPacks) {
      sm[2 + 2 * lane] = b0;
      sm[3 + 2 * lane] = b1;
      const int k = 3 * lane + __popc(mask & ((1u << lane) - 1u));
      uint8_t* const by = reinterpret_cast<uint8_t*>(sm);
      const uint32_t code = wide ? c4 : c3;
      by[z2_pos(k)] = code & 0xFFu;
      by[z2_pos(k + 1)] = (code >> 8) & 0xFFu;
      by[z2_pos(k + 2)] = (code >> 16) & 0xFFu;
      if (wide) by[z2_pos(k + 3)] = code >> 24;
    } else {
      sm[0] = T | (mask << 8);
    }
    __syncwarp();
    const int nq = (kZ2FixedBytes + __popc(mask) + 15) >> 4;
    const bool store = lane < nq;
    uint4 q = make_uint4(0u, 0u, 0u, 0u);
    if (store) q = reinterpret_cast<const uint4*>(sm)[lane];
    const bool bad = store && (q.x == kZ2Sentinel || q.y == kZ2Sentinel ||
                               q.z == kZ2Sentinel || q.w == kZ2Sentinel);
    __syncwarp();
    if (nq <= MaxQuads && !__any_sync(0xFFFFFFFFu, bad)) return Quad{q, lane, store};
  }
  if (lane == kLanes - 1) return Quad{make_uint4(0u, kZ2RawW1, 0u, 0u), 0, true};
  return Quad{v, lane + 1, true};
}

// Warp-collective: lane l's pack from the received slot (lane j holds quad j;
// w0, w1 = its words 0 and 1). sm: the warp's scratch.
AFD uint4 decode2(const uint4& got, uint32_t w0, uint32_t w1, int lane, uint32_t* sm) {
  if (z2_raw(w1)) return shfl4(got, (lane + 1) & (kLanes - 1));
  reinterpret_cast<uint4*>(sm)[lane] = got;
  __syncwarp();
  uint32_t o4[4] = {0u, 0u, 0u, 0u};
  if (lane < kBlockPacks) {
    const uint32_t T = w0 & 0xFFu, mask = z2_mask(w0, w1);
    const uint32_t b0 = sm[2 + 2 * lane], b1 = sm[3 + 2 * lane];
    const bool wide = (mask >> lane) & 1u;
    uint32_t code;
    if (lane == 0) {
      code = (w1 >> 8) | (static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(sm)[256]) << 24);
    } else {
      const int o = 253 + 3 * lane + __popc(mask & ((1u << lane) - 1u));
      code = __funnelshift_r(sm[o >> 2], sm[(o >> 2) + 1], 8 * (o & 3));
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t byte = ((i < 4 ? b0 : b1) >> (8 * (i & 3))) & 0xFFu;
      uint32_t e;
      if (wide) {
        const uint32_t c = (code >> (4 * i)) & 0xFu;
        e = c ? T + 1u - c : 0u;
      } else {
        e = T - ((code >> (3 * i)) & 0x7u);
      }
      const uint32_t h = ((byte & 0x80u) << 8) | (e << 7) | (byte & 0x7Fu);
      o4[i >> 1] |= h << (16 * (i & 1));
    }
  }
  __syncwarp();
  return make_uint4(o4[0], o4[1], o4[2], o4[3]);
}

AFD void store_block(uint32_t* slot, const Quad& q) {
  if (q.store) st_volatile16(reinterpret_cast<uint4*>(slot) + q.at, q.v);
}

// Format v2 receive (above): polls quads 0..kZ2PollQuads-1 of every active
// source (a compressed slot has at most that many), then, once the header
// (words 0, 1) is in, exactly the slot's quads.
template <int N>
AFD void recv_blocks2(const uint32_t* const (&src)[N], const bool (&active)[N],
                      const int (&peer)[N], int lane, int phase, uint4 (&out)[N],
                      uint32_t* sm, const WaitCtx& c) {
  uint4 got[N];
  bool need[N];
  uint32_t hdr[N], mask[N];
  bool known[N];
#pragma unroll
  for (int p = 0; p < N; p++) {
    need[p] = active[p] && lane < kZ2PollQuads;
    known[p] = !active[p];
    hdr[p] = mask[p] = 0u;
    got[p] = make_uint4(0u, 0u, 0u, 0u);
  }
  const unsigned long long t0 = now_ns();
  while (true) {
#pragma unroll
    for (int p = 0; p < N; p++) {
      if (need[p]) {
        const uint4 x = ld_volatile16(reinterpret_cast<const uint4*>(src[p]) + lane);
        if (!dirty(x)) {
          got[p] = x;
          need[p] = false;
        }
      }
    }
    bool pending = false;
#pragma unroll
    for (int p = 0; p < N; p++) {
      if (!known[p]) {
        const bool ready = __shfl_sync(0xFFFFFFFFu, !need[p], 0);
        const uint32_t h = __shfl_sync(0xFFFFFFFFu, got[p].x, 0);
        const uint32_t m = __shfl_sync(0xFFFFFFFFu, got[p].y, 0);
        if (ready) {
          known[p] = true;
          hdr[p] = h;
          mask[p] = m;
          const int nq = z2_quads(h, m);
          if (lane >= nq) need[p] = false;
          else if (lane >= kZ2PollQuads) need[p] = true;
        }
      }
      pending |= need[p] || !known[p];
    }
    if (!__any_sync(0xFFFFFFFFu, pending)) break;
    const unsigned long long waited = now_ns() - t0;
    if (waited > c.limit_ns) {
      int who = 0;
#pragma unroll
      for (int p = N - 1; p >= 0; p--)
        if (need[p] || !known[p]) who = peer[p];
      wait_failed(c, phase, who, waited);
    }
  }
#pragma unroll
  for (int p = 0; p < N; p++)
    if (active[p]) out[p] = decode2(got[p], hdr[p], mask[p], lane, sm);
}


// One-shot for tiny messages (one flag latency instead of two): every rank
// pushes its whole sanitized message to every peer (area s at s*size) and sums
// all R partials of every element itself, in the element's two-shot owner
// order (owner = partition of the element, part_len), rounded once: the same
// bits as two_shot. Shares the stage rotation and refill bookkeeping of
// two_shot (the used region is the prefix [0, R*size) of the stage), so both
// kernels may run on one buffer. stage_packs >= R*size.
template <int R>
__global__ void __launch_bounds__(512, 1)
    one_shot(Ptrs bufs, const uint4* __restrict__ in, uint4* __restrict__ out,
             Epoch* ep, int rank, int size, int stage_packs,
             unsigned long long limit_ns, long long* err) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int stage = ep->w[0];
  const int dirty_stage = (stage + 2) % 3;
  const int dirty_size = ep->w[2 + dirty_stage];
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&ep->w[1], 1u);
  const Regs<R> b(bufs);
  const WaitCtx wc{ep, err, limit_ns, rank, stage, size};

  for (int idx = tid; idx < size; idx += stride) {
    const uint4 v = sanitize(in[idx]);
#pragma unroll
    for (int i = 1; i < R; i++) {
      const int q = (rank + i) & (R - 1);
      st_volatile16(reinterpret_cast<uint4*>(b[q]) + stage * stage_packs + rank * size + idx, v);
    }
  }

  uint4* self = reinterpret_cast<uint4*>(b[rank]);
  uint4* dirty_base = self + dirty_stage * stage_packs;
  const uint4 sent = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
  for (int idx = tid; idx < dirty_size; idx += stride) dirty_base[idx] = sent;

  const uint4* cur = self + stage * stage_packs;
  const int part = size / R;
  for (int idx = tid; idx < size; idx += stride) {
    const int owner = part ? min(idx / part, R - 1) : R - 1;
    const uint4 own = sanitize(in[idx]);
    uint4 v[R];
#pragma unroll
    for (int s = 0; s < R; s++) v[s] = own;
    wait_peers<R>(cur + idx, size, rank, v, wc);
    out[idx] = sanitize(reduce_owner_order<R>(v, owner));
  }

  if (tid == 0) {
    volatile uint32_t* arrived = &ep->w[1];
    while (*arrived < gridDim.x) {
    }
    ep->w[2 + stage] = R * size;
    ep->w[1] = 0;
    ep->w[8] += 1;
    ep->w[0] = (stage + 1) % 3;
  }
}

template __global__ void one_shot<4>(Ptrs, const uint4* __restrict__,
                                     uint4* __restrict__, Epoch*, int, int, int,
                                     unsigned long long, long long*);

// R = 4 ranks. size: message length in 16-byte packs. Per stage: 2R areas
// (reduce-scatter from sender s = area s, all-gather from owner q = area R+q)
// of lb_max block slots of kSlotWords words; stage_words = 2R * lb_max * 128.
// CTA c owns a contiguous range of block indices in every phase on every rank
// (its waits depend only on CTA c of the peers, as in the plugin kernel). In
// the reduce-scatter and the all-gather one warp handles one (block, peer)
// pair: encode and decode are dependent ALU chains that few warps cannot hide.
constexpr int kRanks = 4;
__global__ void __launch_bounds__(512, 1)
    two_shot_z(Ptrs bufs, const uint4* __restrict__ in, uint4* __restrict__ out,
               Epoch* ep, int rank, int size, int stage_words, int lb_max,
               unsigned long long limit_ns, long long* err) {
  const int lane = threadIdx.x & (kLanes - 1);
  const int warp = threadIdx.x / kLanes;
  const int warps = blockDim.x / kLanes;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int stage = ep->w[0];
  const int dirty_stage = (stage + 2) % 3;
  const int dirty_blocks = ep->w[2 + dirty_stage];
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&ep->w[1], 1u);
  // Peer pointers in registers, selected by value (a dynamically indexed
  // array would live in local memory).
  uint32_t* const b0 = reinterpret_cast<uint32_t*>(bufs.p[0]) + stage * stage_words;
  uint32_t* const b1 = reinterpret_cast<uint32_t*>(bufs.p[1]) + stage * stage_words;
  uint32_t* const b2 = reinterpret_cast<uint32_t*>(bufs.p[2]) + stage * stage_words;
  uint32_t* const b3 = reinterpret_cast<uint32_t*>(bufs.p[3]) + stage * stage_words;
  const WaitCtx wc{ep, err, limit_ns, rank, stage, size};
  __shared__ __align__(16) uint32_t scratch[512 / kLanes][kZ2Words];
  uint32_t* const sm = scratch[warp];

  const int part = size / kRanks;
  const int L = part_len<kRanks>(kRanks - 1, size);
  const int nb_all = (L + kBlockPacks - 1) / kBlockPacks;
  const int per_cta = (nb_all + gridDim.x - 1) / gridDim.x;
  const int blk_lo = blockIdx.x * per_cta;
  const int blk_n = max(0, min(nb_all, blk_lo + per_cta) - blk_lo);
  auto len = [&](int q) { return q == kRanks - 1 ? L : part; };
  auto nb = [&](int q) { return (len(q) + kBlockPacks - 1) / kBlockPacks; };
  auto slot = [&](int q, int area, int b) {
    uint32_t* const base = q == 0 ? b0 : (q == 1 ? b1 : (q == 2 ? b2 : b3));
    return base + (area * lb_max + b) * kSlotWords;
  };

  // Reduce-scatter: partition q's blocks to owner q, area `rank`.
  for (int t = warp; t < (kRanks - 1) * blk_n; t += warps) {
    const int b = blk_lo + t / (kRanks - 1);
    const int q = (rank + 1 + t % (kRanks - 1)) & (kRanks - 1);
    if (b < nb(q)) {
      const int idx = b * kBlockPacks + lane;
      const bool valid = lane < kBlockPacks && idx < len(q);
      const uint4 v = valid ? sanitize(in[q * part + idx]) : make_uint4(0u, 0u, 0u, 0u);
      store_block(slot(q, rank, b), encode2(v, lane, sm));
    }
  }

  // Owner: FP32 sum in owner order, rounded once, pushed to every peer.
  for (int b = blk_lo + warp; b < min(blk_lo + blk_n, nb(rank)); b += warps) {
    const int idx = b * kBlockPacks + lane;
    const bool valid = lane < kBlockPacks && idx < len(rank);
    const uint4 own = valid ? sanitize(in[rank * part + idx]) : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t* src[kRanks - 1];
    bool active[kRanks - 1];
    int peer[kRanks - 1];
    uint4 v[kRanks - 1];
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      peer[i - 1] = (rank + i) & (kRanks - 1);
      src[i - 1] = slot(rank, peer[i - 1], b);
      active[i - 1] = true;
    }
    recv_blocks2<kRanks - 1>(src, active, peer, lane, 1, v, sm, wc);
    float acc[8];
    {
      Pack x;
      x.u = own;
#pragma unroll
      for (int e = 0; e < 8; e++) acc[e] = __bfloat162float(x.h[e]);
    }
#pragma unroll
    for (int i = 0; i < kRanks - 1; i++) add_pack(acc, v[i]);
    Pack red;
#pragma unroll
    for (int e = 0; e < 8; e++) red.h[e] = __float2bfloat16(acc[e]);
    const uint4 r = valid ? sanitize(red.u) : make_uint4(0u, 0u, 0u, 0u);
    if (valid) out[rank * part + idx] = r;
    const Quad qd = encode2(r, lane, sm);
#pragma unroll
    for (int i = 1; i < kRanks; i++)
      store_block(slot((rank + i) & (kRanks - 1), kRanks + rank, b), qd);
  }
  uint4* dirty_base = reinterpret_cast<uint4*>(slot(rank, 0, 0) + (dirty_stage - stage) * stage_words);
  const uint4 sent = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
  const int area_quads = dirty_blocks * (kSlotWords / 4);
  const int lb_quads = lb_max * (kSlotWords / 4);
  for (int t = tid; t < 2 * kRanks * area_quads; t += stride)
    dirty_base[(t / area_quads) * lb_quads + t % area_quads] = sent;

  // All-gather: every other owner's reduced blocks.
  for (int t = warp; t < (kRanks - 1) * blk_n; t += warps) {
    const int b = blk_lo + t / (kRanks - 1);
    const int q = (rank + 1 + t % (kRanks - 1)) & (kRanks - 1);
    if (b < nb(q)) {
      const int idx = b * kBlockPacks + lane;
      const uint32_t* src[1] = {slot(rank, kRanks + q, b)};
      const bool active[1] = {true};
      const int peer[1] = {q};
      uint4 v[1];
      recv_blocks2<1>(src, active, peer, lane, 2, v, sm, wc);
      if (lane < kBlockPacks && idx < len(q)) out[q * part + idx] = v[0];
    }
  }

  if (tid == 0) {
    volatile uint32_t* arrived = &ep->w[1];
    while (*arrived < gridDim.x) {
    }
    ep->w[2 + stage] = nb_all;
    ep->w[1] = 0;
    ep->w[8] += 1;
    ep->w[0] = (stage + 1) % 3;
  }
}

#undef AFD

}  // namespace arflags
"""

BIND_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void all_reduce(torch::Tensor inp, torch::Tensor out, std::vector<int64_t> bufs,
                int64_t epoch_ptr, int64_t rank, int64_t stage_packs,
                int64_t threads, int64_t max_blocks, int64_t limit_ns,
                int64_t err_dev_ptr, bool one_shot) {
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(inp.numel() == out.numel());
  TORCH_CHECK(inp.nbytes() % 16 == 0 && inp.nbytes() > 0);
  const int ranks = static_cast<int>(bufs.size());
  TORCH_CHECK(ranks == 2 || ranks == 4, "flags all-reduce: 2 or 4 ranks");
  TORCH_CHECK(0 <= rank && rank < ranks);
  TORCH_CHECK(limit_ns > 0 && err_dev_ptr != 0);
  int size = static_cast<int>(inp.nbytes() / 16);
  int L = ranks == 2 ? arflags::part_len<2>(1, size)
                     : arflags::part_len<4>(3, size);
  // one_shot (four ranks): the one-shot kernel, every rank's whole message in
  // the stage prefix [0, 4 size); else the two-shot.
  TORCH_CHECK(!one_shot || ranks == 4, "the one-shot kernel serves 4 ranks");
  const int work = one_shot ? size : L;
  TORCH_CHECK((one_shot ? ranks * size : 2 * ranks * L) <= stage_packs,
              "message larger than the flags buffer");
  int blocks = static_cast<int>(
      std::min<int64_t>(max_blocks, (work + threads - 1) / threads));
  blocks = std::max(blocks, 1);
  arflags::Ptrs p;
  for (int i = 0; i < arflags::kMaxRanks; i++)
    p.p[i] = i < ranks ? reinterpret_cast<void*>(bufs[i]) : nullptr;
  auto kernel = one_shot ? arflags::one_shot<4>
                         : (ranks == 2 ? arflags::two_shot<2> : arflags::two_shot<4>);
  kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      p, reinterpret_cast<const uint4*>(inp.data_ptr()),
      reinterpret_cast<uint4*>(out.data_ptr()),
      reinterpret_cast<arflags::Epoch*>(epoch_ptr), static_cast<int>(rank), size,
      static_cast<int>(stage_packs), static_cast<unsigned long long>(limit_ns),
      reinterpret_cast<long long*>(err_dev_ptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void all_reduce_z(torch::Tensor inp, torch::Tensor out, std::vector<int64_t> bufs,
                int64_t epoch_ptr, int64_t rank, int64_t stage_words, int64_t lb_max,
                int64_t threads, int64_t max_blocks, int64_t limit_ns,
                int64_t err_dev_ptr) {
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(inp.numel() == out.numel());
  TORCH_CHECK(inp.nbytes() % 16 == 0 && inp.nbytes() > 0);
  TORCH_CHECK(bufs.size() == 4, "flags all-reduce: 4 ranks");
  TORCH_CHECK(0 <= rank && rank < 4);
  TORCH_CHECK(limit_ns > 0 && err_dev_ptr != 0);
  TORCH_CHECK(threads % arflags::kLanes == 0 && threads <= 512);
  int size = static_cast<int>(inp.nbytes() / 16);
  int L = arflags::part_len<4>(3, size);
  int nb = (L + arflags::kBlockPacks - 1) / arflags::kBlockPacks;
  TORCH_CHECK(nb <= lb_max && 2 * 4 * lb_max * arflags::kSlotWords <= stage_words,
              "message larger than the flags buffer");
  const int warps_per_block = static_cast<int>(threads) / arflags::kLanes;
  // One warp per (block, peer) task of the reduce-scatter.
  int blocks = static_cast<int>(
      std::min<int64_t>(max_blocks, (3 * nb + warps_per_block - 1) / warps_per_block));
  blocks = std::max(blocks, 1);
  arflags::Ptrs p;
  for (int i = 0; i < arflags::kMaxRanks; i++) p.p[i] = reinterpret_cast<void*>(bufs[i]);
  arflags::two_shot_z<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      p, reinterpret_cast<const uint4*>(inp.data_ptr()),
      reinterpret_cast<uint4*>(out.data_ptr()),
      reinterpret_cast<arflags::Epoch*>(epoch_ptr), static_cast<int>(rank), size,
      static_cast<int>(stage_words), static_cast<int>(lb_max),
      static_cast<unsigned long long>(limit_ns),
      reinterpret_cast<long long*>(err_dev_ptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_sentinel(int64_t ptr, int64_t bytes) {
  // 0xFF bytes = the 0xFFFFFFFF sentinel word everywhere.
  C10_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(ptr), 0xFF, bytes));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

// Host-mapped, portable error record (16 int64 words, zeroed). Returns the host
// address and the device address; never freed (lives as long as the process).
std::vector<int64_t> alloc_error_record() {
  void* h = nullptr;
  C10_CUDA_CHECK(cudaHostAlloc(&h, 16 * sizeof(int64_t),
                               cudaHostAllocMapped | cudaHostAllocPortable));
  memset(h, 0, 16 * sizeof(int64_t));
  void* d = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&d, h, 0));
  return {reinterpret_cast<int64_t>(h), reinterpret_cast<int64_t>(d)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("all_reduce", &all_reduce, "flags-in-data two-shot (or one-shot) all-reduce");
  m.def("all_reduce_z", &all_reduce_z,
        "flags-in-data two-shot all-reduce, compressed wire format");
  m.def("fill_sentinel", &fill_sentinel, "fill a buffer with the sentinel");
  m.def("alloc_error_record", &alloc_error_record,
        "host-mapped error record for timed-out waits");
}
"""

CUDA_SRC = KERNEL_SRC + BIND_SRC

_MODULE = None


def _build_dir() -> str:
    """Base of the JIT build directories; never inside the source tree."""
    d = os.path.join(envs.VLLM_CACHE_ROOT, "custom_all_reduce_flags")
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def load_module():
    """Compile (or reuse) the extension. Frozen across calls."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from vllm_sm80.jit_extension import load_cuda_extension

    _MODULE = load_cuda_extension(
        "custom_ar_flags",
        CUDA_SRC,
        _build_dir(),
        extra_cflags=["-O3"],
        # No --use_fast_math: the reduction must round exactly as the incumbent.
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_80,code=sm_80"],
    )
    return _MODULE


# Longest a flags-in-data all-reduce waits for a peer before it records the
# overrun and traps (the process then exits with the record logged).
WAIT_S = 60.0


def wait_limit_ns() -> int:
    """:data:`WAIT_S` in nanoseconds, read at call time."""
    return int(WAIT_S * 1e9)


_PHASES = {1: "reduce-scatter", 2: "all-gather"}


def describe(rec) -> str | None:
    """The error message for a record (sequence of 16 ints), or None if unset."""
    if not rec[0]:
        return None
    return (
        "Custom all-reduce flags-in-data wait timed out on rank %d after %.1f s "
        "(limit %.1f s, custom_all_reduce_flags.WAIT_S): call %d, stage %d, "
        "%d packs, %s wait for rank %d (block %d, thread %d). The kernel trapped "
        "instead of returning a partial result; rank %d never delivered its data."
        % (rec[1], rec[7] / 1e9, rec[10] / 1e9, rec[3], rec[2], rec[4],
           _PHASES.get(rec[5], "?"), rec[6], rec[8], rec[9], rec[6])
    )


ACK_WORD = 11


class ErrorRecord:
    """Host-mapped record the kernel fills before it traps, and its watchers.

    A daemon thread polls it (host memory: no CUDA call, works after the
    context has failed), it is checked again at interpreter exit, and callers
    may check it. When set: the message is logged and printed, the record is
    acknowledged (the kernel holds its trap until then, at most 2 s), and the
    process ends with EXIT_CODE.
    """

    def __init__(self, module, poll_s: float = 0.05, on_error=None, exit_fn=None):
        self.host_ptr, self.dev_ptr = module.alloc_error_record()
        self._words = (ctypes.c_int64 * 16).from_address(self.host_ptr)
        self._on_error = on_error or _report
        self._exit = exit_fn if exit_fn is not None else os._exit
        self._stop = threading.Event()
        self._reported = False
        self._thread = threading.Thread(
            target=self._poll, args=(poll_s,), name="flags-ar-watch", daemon=True
        )
        self._thread.start()
        atexit.register(self.check)

    def read(self) -> list[int]:
        return list(self._words)

    def check(self) -> None:
        msg = describe(self.read())
        if msg is not None and not self._reported:
            self._reported = True
            self._on_error(msg)
            self._words[ACK_WORD] = 1  # lets the kernel trap now
            self._exit(EXIT_CODE)

    def _poll(self, poll_s: float) -> None:
        while not self._stop.wait(poll_s):
            self.check()


def _report(msg: str) -> None:
    logger.error(msg)
    print(msg, file=sys.stderr, flush=True)


def largest_part_packs(nbytes: int, world_size: int) -> int:
    """Packs in the largest partition: the last rank's, as in the kernel."""
    size = nbytes // 16
    return size - (world_size - 1) * (size // world_size)


def stage_packs(max_bytes: int, world_size: int) -> int:
    """Packs per buffer stage: a reduce-scatter and an all-gather slot per rank."""
    return 2 * world_size * largest_part_packs(max_bytes, world_size)


def launch_blocks(nbytes: int, world_size: int, one_shot: bool = False) -> int:
    """One thread per pack of the largest partition (the one-shot: of the
    whole message), :data:`THREADS` per block, at most :data:`MAX_BLOCKS`."""
    work = nbytes // 16 if one_shot else largest_part_packs(nbytes, world_size)
    return max(1, min(MAX_BLOCKS, math.ceil(work / THREADS)))


def z_block_slots(max_bytes: int) -> int:
    """Block slots per area of the compressed kernel's stage: the blocks of
    the largest partition (the last rank's) of a ``max_bytes`` message."""
    return -(-largest_part_packs(max_bytes, Z_WORLD_SIZE) // Z_BLOCK_PACKS)


def z_stage_words(max_bytes: int) -> int:
    """Words per compressed stage: a reduce-scatter and an all-gather area per
    rank, each of :func:`z_block_slots` slots."""
    return 2 * Z_WORLD_SIZE * z_block_slots(max_bytes) * Z_SLOT_WORDS


def gate_reason(
    world_size: int,
    same_node: bool,
    capability: tuple[int, int] | None,
    is_cuda: bool,
    pcie_p2p_allowed: bool,
) -> str | None:
    """Why the flags path stays off for this communicator (None: it may run)."""
    if not is_cuda:
        return "not a CUDA platform"
    if world_size not in WORLD_SIZES:
        return f"world size {world_size} (the kernel serves 2 or 4 ranks)"
    if not same_node:
        return "the group spans nodes"
    if capability != (8, 0):
        return f"compute capability {capability} (validated on 8.0 only)"
    if not pcie_p2p_allowed:
        return (
            "the PCIe peer-to-peer custom all-reduce is not enabled "
            "(VLLM_CUSTOM_ALLREDUCE_PCIE_P2P with expandable_segments off)"
        )
    return None


class FlagsAllreduce:
    """Per-communicator state: shared stage buffers and the epoch block.

    ``compressed``: the compressed wire format (``two_shot_z``, four ranks);
    its own buffers, epoch and error record, sized for ``max_bytes``.
    Uncompressed at four ranks, :meth:`all_reduce` may run the one-shot
    kernel on the same buffers and epoch (``one_shot``).
    Construction is collective over `group` (IPC handle exchange, barrier).
    """

    def __init__(self, group, device: torch.device, rank: int, world_size: int,
                 max_bytes: int, create_shared_buffer, free_shared_buffer,
                 compressed: bool = False):
        self.module = load_module()
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.max_bytes = int(max_bytes)
        self.compressed = compressed
        if compressed:
            if world_size != Z_WORLD_SIZE:
                raise ValueError(
                    f"the compressed flags kernel serves {Z_WORLD_SIZE} ranks, not {world_size}"
                )
            self.lb_max = z_block_slots(self.max_bytes)
            self.stage_words = z_stage_words(self.max_bytes)
            self.buffer_bytes = 3 * self.stage_words * 4
        else:
            self.stage_packs = stage_packs(self.max_bytes, world_size)
            self.buffer_bytes = 3 * self.stage_packs * 16
        self._free = free_shared_buffer
        self.ptrs = create_shared_buffer(self.buffer_bytes, group=group)
        self.module.fill_sentinel(self.ptrs[rank], self.buffer_bytes)
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=device)
        self.limit_ns = wait_limit_ns()
        self.errors = ErrorRecord(self.module)
        self._calls = 0
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def eligible(self, inp: torch.Tensor) -> bool:
        return inp.dtype == torch.bfloat16 and 0 < inp.nbytes <= self.max_bytes

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None,
                   one_shot: bool = False):
        """``inp``'s sum into ``out`` (a new tensor if None). ``one_shot``:
        the one-shot kernel (uncompressed, four ranks; the two-shot's bits)."""
        if out is None:
            out = torch.empty_like(inp)
        if self.compressed:
            self.module.all_reduce_z(
                inp,
                out,
                self.ptrs,
                self.epoch.data_ptr(),
                self.rank,
                self.stage_words,
                self.lb_max,
                THREADS,
                MAX_BLOCKS,
                self.limit_ns,
                self.errors.dev_ptr,
            )
        else:
            self.module.all_reduce(
                inp,
                out,
                self.ptrs,
                self.epoch.data_ptr(),
                self.rank,
                self.stage_packs,
                THREADS,
                launch_blocks(inp.nbytes, self.world_size, one_shot),
                self.limit_ns,
                self.errors.dev_ptr,
                one_shot,
            )
        self._calls += 1
        if self._calls % CHECK_EVERY == 0 and not torch.cuda.is_current_stream_capturing():
            self.errors.check()
        return out

    def close(self) -> None:
        self.errors.check()
        if self.ptrs is not None:
            self._free(self.ptrs, rank=self.rank)
            self.ptrs = None
