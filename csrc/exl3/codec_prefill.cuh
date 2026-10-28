#pragma once
// The EXL3 codec as FP16 pairs for the tensor-core prefill kernels
// (dense_prefill_tc.cu): a lane's eight decoded tile values, exact in FP16,
// before any widening. Built on codec.cuh, which this header leaves unchanged.
#include "codec.cuh"
namespace exl3 {
namespace dense_prefill_tc {
// Two codebook values at once. Each lane computes exactly the scalar
// codebook arithmetic of the format: FP16 add (mcg) or FP16 fused
// multiply-add (mul1) with round-to-nearest. The FP16 pair is exact; it is
// codebook_pair before the widening to FP32.
template <int kCodebook>
__device__ __forceinline__ __half2 codebook_half2(uint32_t s0, uint32_t s1) {
  if constexpr (kCodebook == 1) {
    const uint32_t m0 = ((s0 * 0xCBAC1FEDu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint32_t m1 = ((s1 * 0xCBAC1FEDu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint32_t lo = __byte_perm(m0, m1, 0x5410);
    const uint32_t hi = __byte_perm(m0, m1, 0x7632);
    return __hadd2(*reinterpret_cast<const __half2*>(&lo), *reinterpret_cast<const __half2*>(&hi));
  } else {
    // Sum of the four product bytes, 0..1020, placed in the FP16 mantissa
    // of 1024.0 (0x6400): exactly 1024 + sum.
    const uint32_t sum0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0u);
    const uint32_t sum1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0u);
    const uint32_t packed = (sum0 | (sum1 << 16)) + 0x64006400u;
    const uint32_t scale = 0x1eee1eeeu;
    const uint32_t bias = 0xc931c931u;
    return __hfma2(*reinterpret_cast<const __half2*>(&packed),
                   *reinterpret_cast<const __half2*>(&scale),
                   *reinterpret_cast<const __half2*>(&bias));
  }
}

// Encoder lane `lane`'s eight 16-bit states, from its loaded window words.
template <int kBitsX2>
__device__ __forceinline__ void lane_states(
    const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords], int shift, uint32_t (&state)[8]) {
  using Ring = TileRing<kBitsX2>;
  uint32_t u[Ring::kWindowWords];
#pragma unroll
  for (int i = 0; i < Ring::kWindowWords; ++i) {
    const uint32_t next = i + 1 < Ring::kLoadWords ? words[i + 1] : 0u;
    u[i] = __funnelshift_l(next, words[i], shift);
  }
  state[0] = window_state<Ring::rel(0), Ring::kWindowWords>(u);
  state[1] = window_state<Ring::rel(1), Ring::kWindowWords>(u);
  state[2] = window_state<Ring::rel(2), Ring::kWindowWords>(u);
  state[3] = window_state<Ring::rel(3), Ring::kWindowWords>(u);
  state[4] = window_state<Ring::rel(4), Ring::kWindowWords>(u);
  state[5] = window_state<Ring::rel(5), Ring::kWindowWords>(u);
  state[6] = window_state<Ring::rel(6), Ring::kWindowWords>(u);
  state[7] = window_state<Ring::rel(7), Ring::kWindowWords>(u);
}

// decode_lane's eight values as FP16 pairs (values 2i, 2i + 1 in pair i), exact.
template <int kBitsX2, int kCodebook>
__device__ __forceinline__ void decode_lane_half2(
    const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords], int shift, uint32_t (&pair)[4]) {
  uint32_t state[8];
  lane_states<kBitsX2>(words, shift, state);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 h = codebook_half2<kCodebook>(state[2 * i], state[2 * i + 1]);
    pair[i] = *reinterpret_cast<const uint32_t*>(&h);
  }
}
}  // namespace dense_prefill_tc
}  // namespace exl3
