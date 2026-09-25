# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Pinned autotune configs for the fla / GLM-5 KDA Triton kernels on sm_80.

Used by ``pinned_autotune`` when ``VLLM_GLM5_FLA_PIN_AUTOTUNE=1``.

``PINS[(kernel name, autotune key, H)][T bucket]`` is the config for one
launch shape. The autotune key is the tuple ``Autotuner.run`` caches under
(the ``key`` argument values, then the dtype string of every tensor
argument); ``H`` is the kernel's ``H`` argument (None if it has none); the T
bucket is the first bound in ``T_BUCKETS`` that is >= the kernel's ``T``
argument (the launch's total token count), or None past the last bound.
``DEFAULTS`` gives each kernel the config for any shape not in ``PINS``. An
entry is ``{"kwargs": {...}, "num_warps": w, "num_stages": s}`` and must equal
one of the kernel's autotune configs.

Recorded on NVIDIA CMP 170HX 64GB (sm_80), 5 fresh-process runs x 23 cases, heads [16, 64].
T buckets (upper bounds, last open): [64, 200, 512, 1152, 2304, 3456, 4608].
Pick per (kernel, key, H, T bucket): majority over the (run, case)
samples in the bucket. DEFAULTS: the pick for H=16 in the
bucket holding T=1152.
Kernels not launched by the KDA prefill path keep the rule-of-thumb default
(num_warps 4, num_stages 3, block sizes 64): fla.chunk_o.chunk_fwd_kernel_o,
fla.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
fla.cumsum.chunk_local_cumsum_scalar_kernel,
fla.cumsum.chunk_local_cumsum_vector_kernel, fla.kda.chunk_gla_fwd_kernel_o,
fla.kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter,
fla.kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra,
fla.kda.kda_gate_cumsum_fwd_kernel, fla.kda.kda_gate_fwd_kernel,
fla.kda.recompute_w_u_fwd_kernel, fla.l2norm.l2norm_fwd_kernel,
fla.l2norm.l2norm_fwd_kernel1,
fla.solve_tril.merge_16x16_to_32x32_inverse_kernel,
fla.solve_tril.solve_tril_16x16_kernel, fla.wy_fast.recompute_w_u_fwd_kernel,
glm5_kda.kda_gate_fwd_kernel.
"""

# Generated file: regenerate, do not edit by hand.
PROVISIONAL = False

T_BUCKETS: tuple[int, ...] = (64, 200, 512, 1152, 2304, 3456, 4608)

DEFAULTS: dict[str, dict] = {
    "fla.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
    "fla.chunk_o.chunk_fwd_kernel_o": {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 4, "num_stages": 3},
    "fla.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel": {"kwargs": {"BK": 64}, "num_warps": 4, "num_stages": 3},
    "fla.cumsum.chunk_local_cumsum_scalar_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.cumsum.chunk_local_cumsum_vector_kernel": {"kwargs": {"BS": 32}, "num_warps": 4, "num_stages": 3},
    "fla.kda.chunk_gla_fwd_kernel_o": {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 4, "num_stages": 3},
    "fla.kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": {"kwargs": {"BK": 64}, "num_warps": 4, "num_stages": 3},
    "fla.kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.kda.kda_gate_cumsum_fwd_kernel": {"kwargs": {"BD": 64}, "num_warps": 4, "num_stages": 3},
    "fla.kda.kda_gate_fwd_kernel": {"kwargs": {"BT": 64}, "num_warps": 4, "num_stages": 3},
    "fla.kda.recompute_w_u_fwd_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.l2norm.l2norm_fwd_kernel": {"kwargs": {"BT": 64}, "num_warps": 4, "num_stages": 3},
    "fla.l2norm.l2norm_fwd_kernel1": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.solve_tril.merge_16x16_to_32x32_inverse_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.solve_tril.merge_16x16_to_64x64_inverse_kernel": {"kwargs": {}, "num_warps": 2, "num_stages": 3},
    "fla.solve_tril.solve_tril_16x16_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.wy_fast.recompute_w_u_fwd_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.chunk_gla_fwd_kernel_o": {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
    "glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
    "glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra": {"kwargs": {}, "num_warps": 1, "num_stages": 3},
    "glm5_kda.kda_gate_cumsum_fwd_kernel": {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.kda_gate_fwd_kernel": {"kwargs": {"BT": 64}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.recompute_w_u_fwd_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 2},
}

PINS: dict[tuple[str, tuple, int | None], dict[int | None, dict]] = {
    ("fla.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64", (16, 128, 128, 64, "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.float32", "torch.float32", "torch.int32", "torch.int64"), 16): {
        64: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 2},
        200: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 2},
        512: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
        1152: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
        2304: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
        3456: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
        4608: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
        None: {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 3},
    },
    ("fla.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64", (64, 128, 128, 64, "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.float32", "torch.float32", "torch.int32", "torch.int64"), 64): {
        64: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        200: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        512: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        1152: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        2304: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        3456: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        4608: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
        None: {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 2},
    },
    ("fla.solve_tril.merge_16x16_to_64x64_inverse_kernel", (16, 64, True, "torch.float32", "torch.bfloat16", "torch.int32", "torch.int32"), 16): {
        64: {"kwargs": {}, "num_warps": 4, "num_stages": 3},
        200: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        512: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        1152: {"kwargs": {}, "num_warps": 2, "num_stages": 3},
        2304: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
        3456: {"kwargs": {}, "num_warps": 2, "num_stages": 3},
        4608: {"kwargs": {}, "num_warps": 2, "num_stages": 5},
        None: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
    },
    ("fla.solve_tril.merge_16x16_to_64x64_inverse_kernel", (64, 64, True, "torch.float32", "torch.bfloat16", "torch.int32", "torch.int32"), 64): {
        64: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        200: {"kwargs": {}, "num_warps": 2, "num_stages": 3},
        512: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
        1152: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
        2304: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
        3456: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
        4608: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
        None: {"kwargs": {}, "num_warps": 2, "num_stages": 4},
    },
    ("glm5_kda.chunk_gla_fwd_kernel_o", (64, "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.int32", "torch.int32"), 16): {
        64: {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 8, "num_stages": 2},
        200: {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 8, "num_stages": 2},
        512: {"kwargs": {"BK": 64, "BV": 128}, "num_warps": 8, "num_stages": 2},
        1152: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
        2304: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
        3456: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 3},
        4608: {"kwargs": {"BK": 64, "BV": 128}, "num_warps": 4, "num_stages": 2},
        None: {"kwargs": {"BK": 64, "BV": 128}, "num_warps": 4, "num_stages": 4},
    },
    ("glm5_kda.chunk_gla_fwd_kernel_o", (64, "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.int32", "torch.int32"), 64): {
        64: {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 8, "num_stages": 2},
        200: {"kwargs": {"BK": 64, "BV": 128}, "num_warps": 8, "num_stages": 2},
        512: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
        1152: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
        2304: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 4},
        3456: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
        4608: {"kwargs": {"BK": 32, "BV": 128}, "num_warps": 4, "num_stages": 2},
        None: {"kwargs": {"BK": 64, "BV": 128}, "num_warps": 4, "num_stages": 2},
    },
    ("glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter", (16, "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.float32", "torch.float32", "torch.float32", "torch.int32", "torch.int32"), 16): {
        64: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 3},
        200: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 3},
        512: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        1152: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        2304: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        3456: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        4608: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        None: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
    },
    ("glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter", (16, "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.float32", "torch.float32", "torch.float32", "torch.int32", "torch.int32"), 64): {
        64: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 3},
        200: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        512: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        1152: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        2304: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        3456: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        4608: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
        None: {"kwargs": {"BK": 32}, "num_warps": 1, "num_stages": 2},
    },
    ("glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra", (128, 64, "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.float32", "torch.float32", "torch.float32", "torch.int32", "torch.int32"), 16): {
        64: {"kwargs": {}, "num_warps": 8, "num_stages": 3},
        200: {"kwargs": {}, "num_warps": 4, "num_stages": 3},
        512: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        1152: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        2304: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        3456: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        4608: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        None: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
    },
    ("glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra", (128, 64, "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.float32", "torch.float32", "torch.float32", "torch.int32", "torch.int32"), 64): {
        64: {"kwargs": {}, "num_warps": 4, "num_stages": 3},
        200: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        512: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        1152: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        2304: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        3456: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        4608: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
        None: {"kwargs": {}, "num_warps": 1, "num_stages": 3},
    },
    ("glm5_kda.kda_gate_cumsum_fwd_kernel", (16, 128, 64, True, "torch.bfloat16", "torch.float32", "torch.float32", "torch.float32", "torch.int32", "torch.int32"), 16): {
        64: {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
        200: {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
        512: {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
        1152: {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
        2304: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        3456: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        4608: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        None: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
    },
    ("glm5_kda.kda_gate_cumsum_fwd_kernel", (64, 128, 64, True, "torch.bfloat16", "torch.float32", "torch.float32", "torch.float32", "torch.int32", "torch.int32"), 64): {
        64: {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
        200: {"kwargs": {"BD": 32}, "num_warps": 4, "num_stages": 3},
        512: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        1152: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        2304: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        3456: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        4608: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
        None: {"kwargs": {"BD": 64}, "num_warps": 8, "num_stages": 3},
    },
    ("glm5_kda.recompute_w_u_fwd_kernel", (16, 128, 128, 64, 64, 64, True, "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.int32", "torch.int32"), 16): {
        64: {"kwargs": {}, "num_warps": 4, "num_stages": 3},
        200: {"kwargs": {}, "num_warps": 4, "num_stages": 3},
        512: {"kwargs": {}, "num_warps": 8, "num_stages": 2},
        1152: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        2304: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        3456: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        4608: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        None: {"kwargs": {}, "num_warps": 2, "num_stages": 2},
    },
    ("glm5_kda.recompute_w_u_fwd_kernel", (64, 128, 128, 64, 64, 64, True, "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16", "torch.bfloat16", "torch.float32", "torch.int32", "torch.int32"), 64): {
        64: {"kwargs": {}, "num_warps": 8, "num_stages": 3},
        200: {"kwargs": {}, "num_warps": 8, "num_stages": 2},
        512: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        1152: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        2304: {"kwargs": {}, "num_warps": 4, "num_stages": 2},
        3456: {"kwargs": {}, "num_warps": 2, "num_stages": 2},
        4608: {"kwargs": {}, "num_warps": 2, "num_stages": 2},
        None: {"kwargs": {}, "num_warps": 2, "num_stages": 2},
    },
}
