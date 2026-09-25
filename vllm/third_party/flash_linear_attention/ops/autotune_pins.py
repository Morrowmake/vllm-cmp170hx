# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Pinned autotune configs for the fla / GLM-5 KDA Triton kernels on sm_80.

Used by ``pinned_autotune`` when ``VLLM_GLM5_FLA_PIN_AUTOTUNE=1``.

``PINS`` maps (kernel name, autotune key) to one config; the key is the tuple
``Autotuner.run`` caches under (the ``key`` argument values, then the dtype
string of every tensor argument). ``DEFAULTS`` gives each kernel the config
used for any key not in ``PINS``. An entry is
``{"kwargs": {...}, "num_warps": w, "num_stages": s}`` and must equal one of
the kernel's autotune configs.

PROVISIONAL: the defaults below are a rule-of-thumb choice (num_warps 4,
num_stages 3, block sizes 64 where offered), not measured winners. They are
replaced by the per-key winners recorded on the target card.
"""

# Generated file: regenerate, do not edit by hand.
PROVISIONAL = True

DEFAULTS: dict[str, dict] = {
    "fla.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {"kwargs": {"BV": 64}, "num_warps": 4, "num_stages": 3},
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
    "fla.solve_tril.merge_16x16_to_64x64_inverse_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.solve_tril.solve_tril_16x16_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "fla.wy_fast.recompute_w_u_fwd_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.chunk_gla_fwd_kernel_o": {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": {"kwargs": {"BK": 64}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.kda_gate_cumsum_fwd_kernel": {"kwargs": {"BD": 64}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.kda_gate_fwd_kernel": {"kwargs": {"BT": 64}, "num_warps": 4, "num_stages": 3},
    "glm5_kda.recompute_w_u_fwd_kernel": {"kwargs": {}, "num_warps": 4, "num_stages": 3},
}

PINS: dict[tuple[str, tuple], dict] = {
}
