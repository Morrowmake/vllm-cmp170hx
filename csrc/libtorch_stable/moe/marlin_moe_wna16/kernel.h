
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "libtorch_stable/quantization/marlin/marlin.cuh"
#include "libtorch_stable/quantization/marlin/marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#ifdef MARLIN_MOE_FAST_DEQUANT_REDO
#define MARLIN_KERNEL_PARAMS                                                 \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                    \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                        \
      const int4 *__restrict__ b_bias_ptr,                                   \
      const float *__restrict__ a_scales_ptr,                                \
      const int4 *__restrict__ scales_ptr,                                   \
      const float *__restrict__ global_scale_ptr,                            \
      const int4 *__restrict__ zp_ptr,                                       \
      const int32_t *__restrict__ sorted_token_ids_ptr,                      \
      const int32_t *__restrict__ expert_ids_ptr,                            \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,                \
      const float *__restrict__ topk_weights_ptr, int top_k,                 \
      bool mul_topk_weights, int prob_m, int prob_n, int prob_k, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce,              \
      int *__restrict__ redo
#else
#define MARLIN_KERNEL_PARAMS                                                 \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                    \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                        \
      const int4 *__restrict__ b_bias_ptr,                                   \
      const float *__restrict__ a_scales_ptr,                                \
      const int4 *__restrict__ scales_ptr,                                   \
      const float *__restrict__ global_scale_ptr,                            \
      const int4 *__restrict__ zp_ptr,                                       \
      const int32_t *__restrict__ sorted_token_ids_ptr,                      \
      const int32_t *__restrict__ expert_ids_ptr,                            \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,                \
      const float *__restrict__ topk_weights_ptr, int top_k,                 \
      bool mul_topk_weights, int prob_m, int prob_n, int prob_k, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce
#endif

namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
#ifdef MARLIN_MOE_FAST_DEQUANT_REDO
          const bool is_zp_float,  // is zero point of float16 type?
          const bool fast_dequant  // one-HFMA2 subnormal dequant (see below)
#else
          const bool is_zp_float   // is zero point of float16 type?
#endif
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);
#ifdef MARLIN_MOE_FAST_DEQUANT_REDO

// fast_dequant takes the even nibbles unshifted (q & 0xf = q * 2^-133 as bf16
// bits) with scale s * 2^133 and the odd ones as (q << 4) & 0xf0 = q * 2^-129
// with s * 2^129, then one HFMA2 with -8 s per bf16x2.  Exact (bit-identical
// to Marlin's two-step dequant, an exact zero up to its sign) iff s * 2^133 is
// finite, i.e. |s| bits <= kFastDequantMax (every finite scale up to it
// checked exhaustively on sm_80).
constexpr int kFastDequantMax = 0x3cff;

// Optimistic dequant with redo (ops.cu launches the fast_dequant kernel, then
// the regular one, with the same `redo` area; no scale pass, no host sync):
//   redo[0]       count of tiles the fast kernel could not compute exactly
//   redo[1 + i]   their m-n tile index (moe block * n_tiles + n tile); a tile
//                 split over several CTAs may appear more than once.
// The fast kernel appends every tile in which any scale exceeds
// kFastDequantMax (its output for that tile is garbage); the regular kernel
// with a non-null `redo` only recomputes the listed tiles, whole (no k-split),
// and exits if the count is 0.  With redo == nullptr the regular kernel
// computes everything.  redo[0] is zeroed by the caller before the pair of
// launches.
#endif

}
