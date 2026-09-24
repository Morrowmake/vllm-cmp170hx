# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode kernels tuned for NVIDIA CMP 170HX (GA100, sm_80, 70 SMs).

Vendored decode kernels. Three families:

  ``mhc_decode``   the fused mHC post + pre sublayer boundary, 90 calls/step
  ``moe_routing``  routing + block alignment in one kernel, plus ``moe_sum``
  ``kda_decode``   conv + gated delta rule + gated RMSNorm in one kernel

Every one of these is gated behind ``VLLM_GLM5_DECODE_KERNELS`` plus a
per-family switch; with the master flag unset nothing in this package is
imported or called and every call site keeps the upstream code path exactly.

PER-FAMILY TOKEN BOUNDS. The families do not win over the same M range, and an
average family score hides that. Measured us/call (incumbent -> candidate) on
GPU 2, graph replay:

    M        mhc_decode          moe_routing         kda_decode
    1    12.48 -> 11.42 1.09   13.18 -> 5.92 2.23   11.07 ->  9.86 1.12
    2    12.69 -> 11.87 1.07   13.63 -> 7.20 1.89   12.86 -> 11.45 1.12
    4    12.95 -> 12.74 1.02   13.91 -> 8.86 1.57   17.14 -> 14.49 1.18
    8    16.24 -> 15.28 1.06   14.02 -> 12.17 1.15  19.61 -> 18.57 1.06
   16    18.17 -> 20.29 0.90   14.50 -> 13.66 1.06  26.83 -> 24.65 1.09
   32    25.74 -> 22.70 1.13   14.02 -> 15.14 0.93  43.40 -> 42.42 1.02

M=16 is the concurrency-4 decode shape (4 seqs x (1 + 3) spec tokens), so
mHC's 0.90x there is a live regression and the default bound is 8. mHC wins
again at M=32, but ``M <= 8 or M >= 32`` would be fragile -- M=24 is untested
and the curve is non-monotonic around the kernel's internal ``MMA_FROM_M=16``
path switch -- so each family gets one honest upper bound instead.

CUDA-graph constraints: the warmups in ``vllm/ampere_decode/warmup.py`` must run before
the decode graphs are captured.
"""

import torch

__all__ = [
    "use_ampere_mhc_decode",
    "use_ampere_mhc_decode_v2",
    "use_ampere_moe_routing",
    "use_ampere_kda_decode",
    "use_ampere_kda_decode_v2",
    "marlin_block_size_m",
    "stash_fused_align",
    "take_fused_align",
]

# Sentinel for "the caller has a norm_weight tensor". The mHC gate only needs
# to know whether it is None, and the artifact-link script calls the gate with
# three positional arguments, so the parameter cannot be required.
_PRESENT = object()

# Cached device-capability answer. Resolved lazily so that importing this
# package never touches a CUDA device (the CPU tests import it with
# CUDA_VISIBLE_DEVICES="" and assert torch.cuda.is_initialized() is False),
# and cached so the 166 gate evaluations per decode step cost one bool load.
# The CPU tests set it directly; see tests/kernels/test_ampere_decode.py.
_SM80_CACHE: bool | None = None


def _is_sm80() -> bool:
    """True on a GA100-class part, False anywhere else and with no device.

    Guarded: a missing or broken CUDA install must make the gate return False,
    never raise, and never initialise CUDA on a CPU-only box.
    """
    global _SM80_CACHE
    if _SM80_CACHE is None:
        try:
            from vllm.platforms import current_platform

            _SM80_CACHE = bool(
                current_platform.is_cuda()
                and torch.cuda.is_available()
                and current_platform.is_device_capability(80)
            )
        except Exception:  # no driver, no device, stub platform
            _SM80_CACHE = False
    return _SM80_CACHE


def use_ampere_mhc_decode(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
    *,
    norm_weight: object | None = _PRESENT,
) -> bool:
    """Gate for vllm/ampere_decode/mhc_decode.py::mhc_fused_post_pre.

    Resolved on the host, per call, outside any graph. ``norm_weight`` is only
    inspected for None-ness: the ported kernel asserts the input RMSNorm is
    fused in, so a call without it must fall through to TileLang.
    """
    from vllm import envs

    if not envs.VLLM_GLM5_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MHC:
        return False
    if num_tokens < 1 or num_tokens > envs.VLLM_GLM5_DECODE_MHC_MAX_TOKENS:
        return False
    if norm_weight is None:
        return False
    # Kernel A slices `hidden` by HB_A=128 and the 24 prenorm outputs by
    # NB_A=8; both must divide exactly.
    if hidden_size <= 0 or hidden_size % 128:
        return False
    if hc_mult < 2 or (hc_mult * (hc_mult + 2)) % 8:
        return False
    return _is_sm80()


def use_ampere_mhc_decode_v2(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
    *,
    norm_weight: object | None = _PRESENT,
) -> bool:
    """Gate for vllm/ampere_decode/mhc_decode_v2.py::mhc_fused_post_pre.

    Checked before ``use_ampere_mhc_decode``: when open it replaces both the v1
    kernel (M <= 8) and TileLang (M > 8) for 1 <= M <=
    ``VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS`` (32). Does not require
    ``VLLM_GLM5_DECODE_MHC``. Measured us/call, graph replay, one CMP 170HX,
    against the faster of v1 and TileLang in the same run (best of 3):

        M     v1      TileLang   v2     vs faster   vs production today
        4    12.71    12.94     7.76     1.64x       1.64x (v1)
        8    15.22    16.22     8.27     1.84x       1.84x (v1)
       16    20.28    18.15     8.91     2.04x       2.04x (TileLang)
       32    22.66    25.66    10.66     2.13x       2.41x (TileLang)

    Only the hc == 4, hidden % 1024 == 0 path (two Gluon kernels) was measured
    and validated, so the gate admits nothing else.
    """
    from vllm import envs

    if not envs.VLLM_GLM5_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MHC_V2:
        return False
    if num_tokens < 1 or num_tokens > envs.VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS:
        return False
    if norm_weight is None:
        return False
    if hc_mult != 4 or hidden_size <= 0 or hidden_size % 1024:
        return False
    return _is_sm80()


def use_ampere_moe_routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    scoring_func: str = "sigmoid",
    num_expert_group: int = 1,
    topk_group: int = 1,
    renormalize: bool = True,
) -> bool:
    """Gate for vllm/ampere_decode/moe_routing.py.

    Covers ``fused_route_align`` (routing + block alignment) and ``moe_sum``.
    The keyword arguments are the routing-method assumptions the kernel bakes
    in; their defaults are the GLM-5.3-Flash values, so the three-positional
    call form used by the artifact-link script means "the production router".
    """
    from vllm import envs

    if not envs.VLLM_GLM5_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MOE_ROUTING:
        return False
    if num_tokens < 1 or num_tokens > envs.VLLM_GLM5_DECODE_MOE_MAX_TOKENS:
        return False
    # The fused kernel is a single-CTA sigmoid top-k with no group phase.
    if scoring_func != "sigmoid":
        return False
    if num_expert_group != 1 or topk_group != 1:
        return False
    if not renormalize:
        return False
    if num_experts != 288:
        return False
    # `tl.topk` inside the kernel needs a power-of-two k that fits the expert
    # tile split (E=288 -> 256 + 32).
    if topk < 4 or topk > 32 or (topk & (topk - 1)):
        return False
    return _is_sm80()


def use_ampere_kda_decode(
    num_seqs: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Gate for vllm/ampere_decode/kda_decode.py::kda_decode.

    ``num_heads``/``head_dim`` are pinned to the TP=4 GLM-5.3-Flash rank shape
    (64/4 heads, 128-wide) because that is the only shape the vendored
    ``warmup()`` compiles, and a first launch during CUDA graph capture is
    fatal. Raising this means extending the warmup first.
    """
    from vllm import envs

    if not envs.VLLM_GLM5_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_KDA:
        return False
    if num_tokens < 1 or num_tokens > envs.VLLM_GLM5_DECODE_KDA_MAX_TOKENS:
        return False
    if num_seqs < 1 or num_seqs > num_tokens:
        return False
    if num_heads != 16 or head_dim != 128:
        return False
    # One int32 arrival counter per (sequence, head); _CTR_SLOTS is 8192.
    if num_seqs * num_heads > 8192:
        return False
    return _is_sm80()


# Shapes the fused v2 kernel is validated for (vllm/ampere_decode/
# kda_decode_v2.py): the TP=4 GLM-5.3-Flash rank shape, up to 5 tokens per
# sequence (its gate workspace holds 8 token rows) and up to 8 sequences.
_KDA_V2_HEADS = 16
_KDA_V2_HEAD_DIM = 128
_KDA_V2_MAX_TOKENS_PER_SEQ = 5
_KDA_V2_MAX_SEQS = 8


def use_ampere_kda_decode_v2(
    num_seqs: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    w_f: torch.Tensor | None = None,
    w_g: torch.Tensor | None = None,
) -> bool:
    """Gate for vllm/ampere_decode/kda_decode_v2.py::kda_decode_v2.

    Resolved on the host, once per layer call, before any launch. Outside the
    covered shapes, or with any condition unmet, the caller keeps the
    f_b/g_b GEMMs + kda_decode path. ``w_f``/``w_g`` are the f_b_proj and
    g_b_proj weights: the kernel reads them directly, so they must be plain
    bf16 [H * D, D] row-major tensors (unquantized, as in the W4A16
    checkpoint).
    """
    from vllm import envs

    if not envs.VLLM_GLM5_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_KDA_V2:
        return False
    if num_seqs < 1 or num_seqs > _KDA_V2_MAX_SEQS:
        return False
    if num_tokens < num_seqs or num_tokens % num_seqs:
        return False
    if num_tokens // num_seqs > _KDA_V2_MAX_TOKENS_PER_SEQ:
        return False
    if num_tokens > envs.VLLM_GLM5_DECODE_KDA_MAX_TOKENS:
        return False
    if num_heads != _KDA_V2_HEADS or head_dim != _KDA_V2_HEAD_DIM:
        return False
    for w in (w_f, w_g):
        if w is None:
            continue
        if (
            not isinstance(w, torch.Tensor)
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or tuple(w.shape) != (num_heads * head_dim, head_dim)
            or w.stride(1) != 1
        ):
            return False
    return _is_sm80()


def marlin_block_size_m(
    num_tokens: int,
    topk: int,
    num_experts: int,
    input_dtype: torch.dtype | None = None,
) -> int:
    """``fused_marlin_moe``'s host-side ``block_size_m`` choice, verbatim.

    The fused routing kernel produces the block alignment too, but the routing
    runs in ``GroupedTopKRouter`` and the alignment is requested much later,
    inside ``fused_marlin_moe``, which is where the block size is picked. This
    reproduces that pick so the alignment can be computed early. If the
    prediction is ever wrong the handoff below simply misses and upstream's
    ``moe_align_block_size`` runs as before -- correct, just not fused.
    """
    block_size_m = 8
    for block_size_m in (8, 16, 32, 48, 64):
        if num_tokens * topk / num_experts / block_size_m < 0.9:
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)
    return block_size_m


# --- routing -> alignment handoff -------------------------------------------
# `fused_route_align` replaces `fused_grouped_topk` AND `moe_align_block_size`,
# but those are called from two places a long way apart (the router, then the
# Marlin experts' apply). One slot is enough: the 42 MoE layers run strictly
# one after another, routing then alignment, and the slot holds a reference to
# the `topk_ids` it belongs to, so identity comparison can never alias a freed
# tensor.
_PENDING: tuple | None = None


def stash_fused_align(topk_ids, block_size, num_experts, aligned) -> None:
    """Record the alignment the fused routing kernel already computed."""
    global _PENDING
    _PENDING = (topk_ids, int(block_size), int(num_experts), aligned)


def take_fused_align(
    topk_ids,
    block_size: int,
    num_experts: int,
    expert_map,
    pad_sorted_ids: bool,
):
    """Return the stashed alignment for exactly this request, or None."""
    global _PENDING
    pending = _PENDING
    if pending is None:
        return None
    # `ignore_invalid_experts` deliberately has no guard: upstream passes
    # `expert_map if ignore_invalid_experts else None` down to the kernel and
    # applies the post-hoc gather only `if expert_map is not None and not
    # ignore_invalid_experts`, so once expert_map is None -- which the line
    # below requires -- the flag changes nothing on either branch. Marlin's
    # ignore_invalid_experts=True is therefore covered by the expert_map test.
    if expert_map is not None or pad_sorted_ids:
        return None
    ids, bs, ne, aligned = pending
    if ids is not topk_ids or bs != int(block_size) or ne != int(num_experts):
        return None
    _PENDING = None
    return aligned
