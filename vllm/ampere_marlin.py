# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the optional, prebuilt Marlin extension. Never compile at runtime.

Importing this module does not import or load the optional library. Workers call
``require_extension`` before model loading only when a CUDA Marlin flag is set.
The same binary serves tensor- and pipeline-parallel expert layouts.
"""

import importlib

import torch

_OPS = None
_ABI_VERSION = 1
_SCHEMA_ARGUMENTS = {
    "decode_gemm": (
        "a w s sorted eids ntpp topk_w topk n_slots K N w13 rows cfg ctr out",
        "Tensor Tensor Tensor Tensor Tensor Tensor Tensor int int int int bool int int Tensor Tensor",
        "int",
    ),
    "decode_act": (
        "part ids ksplit n_slots Nh limit h",
        "Tensor Tensor int int int float Tensor",
        "",
    ),
    "prefill_gemm": (
        "a c_or_none b_q_weight b_bias_or_none b_scales a_scales global_scale "
        "b_zeros_or_none workspace sorted_token_ids expert_ids "
        "num_tokens_past_padded topk_weights moe_block_size top_k "
        "mul_topk_weights b_type_id size_m size_n size_k use_atomic_add "
        "use_fp32_reduce is_zp_float thread_k thread_n blocks_per_sm c_tmp",
        "Tensor Optional[Tensor] Tensor Optional[Tensor] Tensor Optional[Tensor] "
        "Optional[Tensor] Optional[Tensor] Tensor Tensor Tensor Tensor Tensor "
        "int int bool int int int int bool bool bool int int int Tensor",
        "Tensor",
    ),
}


def _runtime_build_info() -> dict:
    cuda = torch.version.cuda
    major, minor = (int(v) for v in cuda.split(".")[:2]) if cuda else (0, 0)
    return {
        "abi_version": _ABI_VERSION,
        "torch_version": torch.__version__.split("+")[0],
        "cuda_version": major * 1000 + minor * 10,
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def _validate_extension(module) -> None:
    built = module.build_info()
    for key, expected in _runtime_build_info().items():
        actual = built.get(key)
        # CUDA minor toolkits within one major share the runtime ABI.
        if key == "cuda_version" and isinstance(actual, int):
            actual, expected = actual // 1000, expected // 1000
        if actual != expected:
            raise ValueError(
                f"{key} mismatch: extension {built.get(key)!r}, runtime {expected!r}"
            )
    for name, (names, types, returns) in _SCHEMA_ARGUMENTS.items():
        qualified = f"_ampere_marlin_C::{name}"
        schema = torch._C._dispatch_find_schema_or_throw(qualified, "").schema()
        actual = (
            " ".join(a.name for a in schema.arguments),
            " ".join(str(a.type) for a in schema.arguments),
            " ".join(str(a.type) for a in schema.returns),
        )
        if actual != (names, types, returns):
            raise ValueError(f"incompatible operator schema: {qualified}")
        if not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "CUDA"):
            raise ValueError(f"missing CUDA implementation: {qualified}")


def require_extension():
    """Return validated operators, or fail with installation instructions.

    This explicit API also allows launchers to check an opt-in installation
    without allocating CUDA memory. Success is cached; failures are not.
    """
    global _OPS
    if _OPS is not None:
        return _OPS
    try:
        module = importlib.import_module("vllm._ampere_marlin_C")
        _validate_extension(module)
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "Optional compiled Marlin was requested, but vllm._ampere_marlin_C "
            "is missing or incompatible. Install a matching prebuilt package or "
            "rebuild this vLLM installation with VLLM_BUILD_AMPERE_MARLIN=1 "
            "using the current PyTorch/CUDA environment. To use the released "
            "kernels instead, set VLLM_GLM5_MARLIN_DECODE_CUDA=0 and "
            f"VLLM_GLM5_MARLIN_PREFILL_CUDA=0 before restarting. Detail: {exc}"
        ) from exc
    _OPS = torch.ops._ampere_marlin_C
    return _OPS
