# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused replacements for the eager decode prologue (GLM-5.3-Flash / DFlash).

Opt-in with ``VLLM_GLM5_PROLOGUE_FUSE=1``.  Default **off**: every call site
checks :func:`enabled` before touching a tensor, so with the flag unset the
upstream instruction sequence runs unchanged, byte for byte.

Why
---
A c1 DFlash decode step is two CUDA-graph launches plus ~116 eager kernels of
input/metadata preparation that run *before* the first collective.  Those 116
kernels produce a few hundred bytes of index tensors and cost 0.48 ms on rank 0
and 0.67 ms on rank 3; only ~0.25 ms of that is kernel time, the rest is
launch gaps.  Because the ranks then meet in the embedding all-reduce, the
slowest rank's prologue sets the step time and ranks 0-2 burn ~0.19 ms/step
waiting.  The cure is fewer launches, not faster kernels.

What is fused
-------------
The Mamba and GDN sites are fused here. The GDN fusion also removes
the four ``async_tensor_h2d`` copies those builds make:

==============================================================  ====  ======
site                                                            now   fused
==============================================================  ====  ======
``attention/backends/gdn_attn.py:build`` (x4 KDA groups)          44       4
``attention/backends/utils.py:mamba_get_block_table_tensor``      28       4
``utils/torch_utils.py:async_tensor_h2d`` (the 4 GDN mask H2Ds)    4       0
==============================================================  ====  ======

Exactness
---------
Every fused op has a pure-torch reference (``*_ref``) that is a transcription
of the upstream sequence, and the Triton kernel is a transcription of the
reference.  Both families produce **integer index tensors only** -- no reduction
order, no floating point -- so "exact" here means bit-identical, and the CPU
tests in ``tests/v1/worker/test_prologue_fuse.py`` assert exactly that against
the upstream implementations.  On a non-CUDA tensor (i.e. in those tests) the
reference path runs, so the dispatch wrapper itself is covered too.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import (
        GDNAttentionMetadata,
        GDNAttentionMetadataBuilder,
    )

logger = init_logger(__name__)

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def _env_flag(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean")


@dataclass(frozen=True)
class PrologueFuseSettings:
    """Parsed ``VLLM_GLM5_PROLOGUE_FUSE*`` environment."""

    enabled: bool = False
    # Individually disable one fusion, for bisecting a regression.
    gdn: bool = True
    mamba_block_table: bool = True


def read_settings(env: dict[str, str] | None = None) -> PrologueFuseSettings:
    """Parse the prologue-fusion environment. Pure; safe to call on CPU."""
    env = dict(os.environ) if env is None else env
    on = _env_flag(env, "VLLM_GLM5_PROLOGUE_FUSE", False)
    return PrologueFuseSettings(
        enabled=on,
        gdn=_env_flag(env, "VLLM_GLM5_PROLOGUE_FUSE_GDN", True),
        mamba_block_table=_env_flag(env, "VLLM_GLM5_PROLOGUE_FUSE_MAMBA_BT", True),
    )


@functools.cache
def settings() -> PrologueFuseSettings:
    s = read_settings()
    if s.enabled:
        logger.info(
            "GLM-5 decode prologue fusion active (gdn=%s mamba_bt=%s)",
            s.gdn,
            s.mamba_block_table,
        )
    return s


def enabled() -> bool:
    return settings().enabled


# --------------------------------------------------------------------------- #
# Triton availability. The module must import on a CPU-only box (unit tests),
# so the kernels are defined lazily and the reference path is always available.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised on GPU only
    from vllm.triton_utils import tl, triton

    _HAS_TRITON = triton is not None and tl is not None
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _use_triton(t: torch.Tensor) -> bool:
    return _HAS_TRITON and t.is_cuda


def _flat(*tensors: torch.Tensor) -> bool:
    """Every 1-D tensor the kernels index with a bare offset must be dense."""
    return all(t.is_contiguous() for t in tensors)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


# --------------------------------------------------------------------------- #
# 1. mamba_get_block_table_tensor, "align" mode.        7 kernels -> 1
#
# Upstream (vllm/v1/attention/backends/utils.py):
#     start_indices = (seq_lens - 1) // block_size      # sub, floor_divide
#     start_indices.clamp_(min=0)                       # clamp
#     offsets = torch.arange(1 + num_spec_blocks, ...)  # arange
#     idx = (start_indices.unsqueeze(1) + offsets)      # add
#            .to(torch.int64)                           # cast
#     return torch.gather(block_table, 1, idx)          # gather
# --------------------------------------------------------------------------- #

if _HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _mamba_tail_block_table_kernel(
        block_table_ptr,
        seq_lens_ptr,
        out_ptr,
        bt_row_stride,
        bt_col_stride,
        out_stride,
        num_cols_in,
        MAMBA_BLOCK_SIZE: tl.constexpr,
        NUM_COLS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        r = tl.program_id(0)
        s = tl.load(seq_lens_ptr + r).to(tl.int32)
        # (s - 1) // block_size then clamp(min=0). Triton integer division
        # truncates toward zero while torch floor-divides, but the two differ
        # only for s == 0 (-1 -> 0 vs -1 -> -1) and the clamp maps both to 0.
        start = (s - 1) // MAMBA_BLOCK_SIZE
        start = tl.maximum(start, 0)
        cols = tl.arange(0, BLOCK)
        keep = cols < NUM_COLS
        idx = start + cols
        v = tl.load(
            block_table_ptr + r * bt_row_stride + idx * bt_col_stride,
            mask=keep & (idx < num_cols_in),
            other=0,
        )
        tl.store(out_ptr + r * out_stride + cols, v, mask=keep)


def mamba_tail_block_table_ref(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_speculative_blocks: int,
) -> torch.Tensor:
    """Pure-torch transcription of the upstream "align" branch."""
    start_indices = (seq_lens - 1) // block_size
    start_indices = start_indices.clamp(min=0)
    offsets = torch.arange(
        1 + num_speculative_blocks,
        device=block_table.device,
        dtype=torch.int32,
    )
    indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
    return torch.gather(block_table, 1, indices_to_gather)


def mamba_tail_block_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_speculative_blocks: int,
) -> torch.Tensor:
    """One kernel in place of sub/div/clamp/arange/add/cast/gather."""
    num_cols = 1 + num_speculative_blocks
    if (
        not _use_triton(block_table)
        or block_table.dim() != 2
        or seq_lens.dim() != 1
        or not _flat(seq_lens)
        or seq_lens.shape[0] != block_table.shape[0]
    ):
        # `block_table` itself may be a strided view (e.g. [:, ::2] for a
        # kernel-block-size remap); the kernel handles that via bt_col_stride.
        return mamba_tail_block_table_ref(
            block_table, seq_lens, block_size, num_speculative_blocks
        )
    num_reqs = block_table.shape[0]
    out = torch.empty(
        (num_reqs, num_cols), dtype=block_table.dtype, device=block_table.device
    )
    if num_reqs == 0:
        return out
    _mamba_tail_block_table_kernel[(num_reqs,)](
        block_table,
        seq_lens,
        out,
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        block_table.shape[1],
        MAMBA_BLOCK_SIZE=block_size,  # type: ignore[arg-type]
        NUM_COLS=num_cols,  # type: ignore[arg-type]
        BLOCK=_next_pow2(num_cols),  # type: ignore[arg-type]
    )
    return out


# --------------------------------------------------------------------------- #
# 2. GDNAttentionMetadataBuilder.build, pure spec-decode shape.
#                                                      11 kernels -> 2 (x4)
#
# In steady DFlash decode every row is a spec-decode row and the padded rows
# are zero-length at the back, so upstream's masked selects degenerate into
# leading slices and its six copy_ / five fill_ launches into one write per
# persistent buffer.  Anything else (any prefill, any plain decode, a mask that
# is not a leading run, a batch beyond the capture size) returns None here and
# the unmodified upstream body runs.
# --------------------------------------------------------------------------- #

if _HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _gdn_spec_decode_meta_kernel(
        block_table_ptr,
        bt_row_stride,
        bt_col_stride,
        src_qsl_ptr,
        src_acc_ptr,
        state_idx_ptr,
        state_idx_stride,
        masks_ptr,  # int8 view of the bool buffer
        token_indx_ptr,
        qsl_out_ptr,
        acc_out_ptr,
        num_spec_decodes,
        batch_size,
        spec_token_size,
        NUM_COLS: tl.constexpr,
        NULL_BLOCK_ID: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        which = tl.program_id(0)
        off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        if which == 0:
            # spec_state_indices_tensor[:batch_size, :NUM_COLS]
            keep = off < batch_size * NUM_COLS
            row = off // NUM_COLS
            col = off % NUM_COLS
            is_spec = row < num_spec_decodes
            v = tl.load(
                block_table_ptr + row * bt_row_stride + col * bt_col_stride,
                mask=keep & is_spec,
                other=0,
            )
            v = tl.where(is_spec, v, NULL_BLOCK_ID)
            tl.store(state_idx_ptr + row * state_idx_stride + col, v, mask=keep)
        elif which == 1:
            # spec_sequence_masks[:batch_size]
            keep = off < batch_size
            v = tl.where(off < num_spec_decodes, 1, 0).to(tl.int8)
            tl.store(masks_ptr + off, v, mask=keep)
        elif which == 2:
            # spec_token_indx[:spec_token_size] = arange(spec_token_size)
            keep = off < spec_token_size
            tl.store(token_indx_ptr + off, off.to(tl.int32), mask=keep)
        elif which == 3:
            # spec_query_start_loc[:batch_size + 1], tail held at qsl[S]
            keep = off < batch_size + 1
            idx = tl.minimum(off, num_spec_decodes)
            v = tl.load(src_qsl_ptr + idx, mask=keep, other=0)
            tl.store(qsl_out_ptr + off, v, mask=keep)
        else:
            # num_accepted_tokens[:batch_size], padded rows = 1
            keep = off < batch_size
            is_spec = off < num_spec_decodes
            v = tl.load(src_acc_ptr + off, mask=keep & is_spec, other=0)
            v = tl.where(is_spec, v, 1)
            tl.store(acc_out_ptr + off, v, mask=keep)


def gdn_spec_decode_meta_ref(
    block_table: torch.Tensor,
    src_qsl: torch.Tensor,
    src_acc: torch.Tensor,
    state_idx: torch.Tensor,
    masks: torch.Tensor,
    token_indx: torch.Tensor,
    qsl_out: torch.Tensor,
    acc_out: torch.Tensor,
    num_spec_decodes: int,
    batch_size: int,
    spec_token_size: int,
    num_cols: int,
    null_block_id: int,
) -> None:
    """Pure-torch transcription of the upstream buffer writes."""
    s = num_spec_decodes
    state_idx[:s, :num_cols].copy_(block_table[:s, :num_cols])
    state_idx[s:batch_size, :num_cols].fill_(null_block_id)
    masks[:s].fill_(True)
    masks[s:batch_size].fill_(False)
    token_indx[:spec_token_size].copy_(
        torch.arange(
            spec_token_size, dtype=token_indx.dtype, device=token_indx.device
        )
    )
    qsl_out[: s + 1].copy_(src_qsl[: s + 1])
    qsl_out[s + 1 : batch_size + 1].fill_(src_qsl[s].item())
    acc_out[:s].copy_(src_acc[:s])
    acc_out[s:batch_size].fill_(1)


def _gdn_spec_decode_meta(
    block_table: torch.Tensor,
    src_qsl: torch.Tensor,
    src_acc: torch.Tensor,
    state_idx: torch.Tensor,
    masks: torch.Tensor,
    token_indx: torch.Tensor,
    qsl_out: torch.Tensor,
    acc_out: torch.Tensor,
    num_spec_decodes: int,
    batch_size: int,
    spec_token_size: int,
    num_cols: int,
    null_block_id: int,
) -> None:
    if not _use_triton(block_table) or not _flat(
        src_qsl, src_acc, masks, token_indx, qsl_out, acc_out
    ):
        gdn_spec_decode_meta_ref(
            block_table,
            src_qsl,
            src_acc,
            state_idx,
            masks,
            token_indx,
            qsl_out,
            acc_out,
            num_spec_decodes,
            batch_size,
            spec_token_size,
            num_cols,
            null_block_id,
        )
        return
    block = 128
    longest = max(batch_size * num_cols, batch_size + 1, spec_token_size, 1)
    grid = (5, (longest + block - 1) // block)
    _gdn_spec_decode_meta_kernel[grid](
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        src_qsl,
        src_acc,
        state_idx,
        state_idx.stride(0),
        masks.view(torch.int8),
        token_indx,
        qsl_out,
        acc_out,
        num_spec_decodes,
        batch_size,
        spec_token_size,
        NUM_COLS=num_cols,  # type: ignore[arg-type]
        NULL_BLOCK_ID=null_block_id,  # type: ignore[arg-type]
        BLOCK=block,  # type: ignore[arg-type]
    )


def gdn_spec_decode_plan(
    builder: Any,
    num_reqs: int,
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> dict[str, int] | None:
    """CPU-only eligibility test for the fused GDN path.

    Returns the derived scalars when the batch has the steady DFlash decode
    shape that the fused writer handles, else ``None``.  Touches only small
    CPU tensors, so it launches nothing.
    """
    if not builder.use_spec_decode or num_decode_draft_tokens_cpu is None:
        return None
    if not builder.use_full_cuda_graph:
        return None
    mask = num_decode_draft_tokens_cpu >= 0
    s = int(mask.sum().item())
    if s == 0 or s > num_reqs:
        return None
    # Spec rows must be exactly the leading run: upstream selects them with a
    # boolean mask, the fused writer with a slice, and the two agree only then.
    if not bool(mask[:s].all().item()) or bool(mask[s:num_reqs].any().item()):
        return None
    if int(num_decode_draft_tokens_cpu[:s].sum().item()) == 0:
        return None
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - (
        query_start_loc_cpu[:num_reqs]
    )
    # num_prefills == 0 and num_decodes == 0 <=> every non-spec row is empty.
    if int(query_lens_cpu[s:num_reqs].abs().sum().item()) != 0:
        return None
    total_tokens = int(query_lens_cpu.sum().item())
    if s > builder.decode_cudagraph_max_bs:
        return None
    if total_tokens > builder.decode_cudagraph_max_bs:
        return None
    spec_token_size = min(
        s * (builder.num_spec + 1), int(query_start_loc_cpu[num_reqs].item())
    )
    if spec_token_size > builder.spec_token_indx.numel():
        return None
    # The fused writer indexes the persistent buffers at [:batch_size]; upstream
    # does the same, but only ever reaches here with a batch that fits.
    if num_reqs > builder.decode_cudagraph_max_bs:
        return None
    return {
        "num_spec_decodes": s,
        "num_spec_decode_tokens": total_tokens,
        "spec_token_size": spec_token_size,
    }


def try_build_gdn_spec_decode(
    builder: "GDNAttentionMetadataBuilder",
    m: Any,
    block_table_tensor: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> "GDNAttentionMetadata | None":
    """Fused GDN metadata build, or ``None`` to fall back to upstream."""
    s = settings()
    if not (s.enabled and s.gdn):
        return None
    if num_accepted_tokens is None:
        return None
    batch_size = m.num_reqs
    plan = gdn_spec_decode_plan(
        builder, batch_size, m.query_start_loc_cpu, num_decode_draft_tokens_cpu
    )
    if plan is None:
        return None
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    nsd = plan["num_spec_decodes"]
    num_cols = builder.num_spec + 1
    if block_table_tensor.dim() != 2 or block_table_tensor.shape[1] < num_cols:
        return None
    _gdn_spec_decode_meta(
        block_table_tensor,
        m.query_start_loc,
        num_accepted_tokens,
        builder.spec_state_indices_tensor,
        builder.spec_sequence_masks,
        builder.spec_token_indx,
        builder.spec_query_start_loc,
        builder.num_accepted_tokens,
        nsd,
        batch_size,
        plan["spec_token_size"],
        num_cols,
        NULL_BLOCK_ID,
    )
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=nsd,
        num_spec_decode_tokens=plan["num_spec_decode_tokens"],
        num_actual_tokens=m.num_actual_tokens,
        has_initial_state=None,
        chunk_indices=None,
        chunk_offsets=None,
        prefill_query_start_loc=None,
        prefill_state_indices=None,
        prefill_has_initial_state=None,
        spec_query_start_loc=builder.spec_query_start_loc[: batch_size + 1],
        non_spec_query_start_loc=None,
        spec_state_indices_tensor=builder.spec_state_indices_tensor[:batch_size],
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=builder.spec_sequence_masks[:batch_size],
        spec_token_indx=builder.spec_token_indx[: plan["spec_token_size"]],
        non_spec_token_indx=builder.non_spec_token_indx[:0],
        num_accepted_tokens=builder.num_accepted_tokens[:batch_size],
        nums_dict=None,
        batch_ptr=None,
        token_chunk_offset_ptr=None,
    )
