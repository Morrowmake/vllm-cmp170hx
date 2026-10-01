# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM for GLM-5 KDA layers on the fused sm_80 decode path (TP only).

Speculative verification normally keeps one recurrent state per draft
position per request (`num_speculative_blocks = num_spec` extra state pages per
KDA layer), so that the state at the accepted position can be picked up next
step.  With VLLM_GLM5_KDA_RECOVER=1:

  * the KV spec reserves no draft-position state pages
    (`num_speculative_blocks = 0`); the attention block size is unchanged;
  * the verify step (kda_decode_v2 with `records`) starts from the request's
    single state, writes no state and stores, per token, the correction
    c_t [V], the normalised key k_t [K] and the decay gate e_t [K] (fp32) into
    a per-running-request record buffer (one per KDA layer, allocated once);
  * after sampling, `KDARecoverCommitContext.commit` replays the accepted
    tokens over the state with the same fma sequence (bitwise the per-token
    state) and writes it once, compacts the conv window and, in align mode,
    also writes the block-boundary state, using the commit-plan and
    conv-compaction kernels of the upstream RecoverSSM implementation.

Accepted-token bookkeeping follows upstream RecoverSSM: the verify always
sees num_accepted = 1 (the committed state and conv window are already at the
accepted position) and the align postprocess resets the running state column.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

WS_T = 8                     # token rows per record (kda_decode_v2._WS_T)
HEADS = 16                   # KDA heads per card the verify kernel covers here
HEAD_DIM = 128
_ARCHS = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")

def kda_recover_requested() -> bool:
    from vllm import envs

    return bool(envs.VLLM_GLM5_KDA_RECOVER)


def _closed_reason(vllm_config) -> str | None:
    from vllm import envs
    from vllm.ampere_decode import _is_sm80

    model_config = vllm_config.model_config
    archs = tuple(getattr(model_config.hf_config, "architectures", None) or ())
    if not any(a in _ARCHS for a in archs):
        return f"model {archs} is not GLM-5 Next"
    if not (envs.VLLM_GLM5_DECODE_KERNELS and envs.VLLM_GLM5_DECODE_KDA_V2):
        return "needs VLLM_GLM5_DECODE_KERNELS=1 and VLLM_GLM5_DECODE_KDA_V2=1"
    num_spec = vllm_config.num_speculative_tokens or 0
    if num_spec < 1 or num_spec + 1 > WS_T:
        return f"needs 1..{WS_T - 1} speculative tokens (have {num_spec})"
    pc = vllm_config.parallel_config
    if pc.pipeline_parallel_size != 1:
        return "pipeline parallel is not supported"
    text = model_config.hf_text_config
    heads = int(getattr(text, "linear_num_heads", 0) or 0)
    head_dim = int(getattr(text, "linear_head_dim", 0) or 0)
    tp = pc.tensor_parallel_size
    if head_dim != HEAD_DIM or tp <= 0 or heads % tp or heads // tp != HEADS:
        return f"needs {HEADS} KDA heads of {HEAD_DIM} per card (have {heads}/{tp})"
    if vllm_config.scheduler_config.max_num_seqs > 8:
        return "needs max_num_seqs <= 8"
    if vllm_config.cache_config.mamba_cache_mode not in ("align", "none"):
        return f"mamba cache mode {vllm_config.cache_config.mamba_cache_mode!r}"
    if getattr(vllm_config.cache_config, "use_replayssm", False):
        return "--use-replayssm is set"
    if not _is_sm80():
        return "needs sm_80"
    return None


def kda_recover_enabled(vllm_config) -> bool:
    """VLLM_GLM5_KDA_RECOVER=1 and the configuration it is built for.

    Cheap; called at init by the KDA layers, the metadata builders and the
    model state. Logs the banner (or why the gate is closed) once.
    """
    if not kda_recover_requested():
        return False
    reason = _closed_reason(vllm_config)
    if reason is None:
        logger.info_once(
            "sm_80 KDA recover active: one KDA state per request, accepted "
            "tokens replayed after sampling (VLLM_GLM5_KDA_RECOVER=1)."
        )
        return True
    logger.info_once(
        "VLLM_GLM5_KDA_RECOVER=1 has no effect: %s; keeping one KDA state per "
        "draft position.",
        reason,
    )
    return False


def records_rows(vllm_config) -> int:
    return int(vllm_config.scheduler_config.max_num_seqs)


def records_nbytes(vllm_config, num_kda_layers: int) -> int:
    """Device bytes of the record buffer of one card."""
    return 3 * num_kda_layers * records_rows(vllm_config) * WS_T * HEADS * HEAD_DIM * 4


class RecordPool:
    """One fp32 [layers, 3, rows, WS_T, H, D] tensor per device: (c, k, e) of
    every KDA layer of the card."""

    _pools: dict = {}

    @classmethod
    def get(cls, device, num_layers: int, rows: int) -> torch.Tensor:
        key = (torch.device(device).type, torch.device(device).index)
        t = cls._pools.get(key)
        if t is None:
            t = torch.zeros(num_layers, 3, rows, WS_T, HEADS, HEAD_DIM,
                            dtype=torch.float32, device=device)
            cls._pools[key] = t
        assert t.shape[0] == num_layers and t.shape[2] == rows
        return t


def layer_records(pool: torch.Tensor, index: int):
    return pool[index, 0], pool[index, 1], pool[index, 2]


# ---------------------------------------------------------------------------
# commit: replay the accepted tokens over the state, all layers in one launch
# ---------------------------------------------------------------------------
@triton.jit
def _kda_recover_commit_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    records_ptr,               # [pool layers, 3, rows, WS_T, H, D] fp32
    record_index_ptr,          # [L] int64: pool layer of each committed layer
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_indices,
    stride_rec_layer,
    stride_rec_part,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    WS_T: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_lh = tl.program_id(2)
    i_l = i_lh // H
    i_h = i_lh % H
    src = tl.load(state_indices_ptr + i_b * stride_state_indices).to(tl.int64)
    if src <= null_block_id:
        return
    n = tl.load(commit_lens_ptr + i_b)
    if n == 0:
        return
    fin = tl.load(final_state_indices_ptr + i_b).to(tl.int64)
    if fin <= null_block_id:
        return
    bnd = tl.load(boundary_state_indices_ptr + i_b).to(tl.int64)
    blen = tl.load(boundary_recovery_lens_ptr + i_b)

    base = tl.load(state_base_addrs_ptr + i_l).to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    bstride = tl.load(state_block_strides_ptr + i_l)
    o_v = i_v * BV + tl.arange(0, BV)
    o_d = tl.arange(0, D)
    off = i_h * stride_state_head + o_v[:, None] * stride_state_v + o_d[None, :]
    h = tl.load(base + src * bstride + off)
    i_r = tl.load(record_index_ptr + i_l)
    rb = records_ptr + i_r * stride_rec_layer + (i_b * WS_T * H + i_h) * D
    for t in tl.static_range(WS_T):
        if t < n:
            c = tl.load(rb + t * (H * D) + o_v)
            k = tl.load(rb + stride_rec_part + t * (H * D) + o_d)
            e = tl.load(rb + 2 * stride_rec_part + t * (H * D) + o_d)
            he = h * e[None, :]
            cb, kb = tl.broadcast(c[:, None], k[None, :])
            h = tl.fma(cb, kb, he)
            if ALIGN_MODE:
                if (bnd > null_block_id) & (t + 1 == blen):
                    tl.store(base + bnd * bstride + off, h)
    tl.store(base + fin * bstride + off, h)


COMMIT_BV = 32
COMMIT_WARPS = 4


@dataclasses.dataclass
class KDARecoverCommitContext:
    conv_states: tuple
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    states: tuple
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    records: torch.Tensor          # the card's pool [layers, 3, rows, WS_T, H, D]
    record_index: torch.Tensor     # [L] int64 pool layer of each layer here
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int

    @classmethod
    def create(cls, layers: Sequence[Any], *, spec_query_len: int,
               max_num_reqs: int) -> "KDARecoverCommitContext":
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        if not layers:
            raise ValueError("KDA recover commit needs at least one layer")
        pool = layers[0]._kda_recover_pool
        index = [layer._kda_recover_index for layer in layers]
        if any(layer._kda_recover_pool is not pool for layer in layers):
            raise ValueError("KDA recover layers must share one record pool")
        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        states = [layer.kv_cache[1] for layer in layers]
        ref = states[0]
        if ref.ndim != 4 or ref.dtype != torch.float32 or ref.stride(3) != 1:
            raise ValueError("KDA recover needs fp32 [blocks, H, V, K] states")
        if tuple(ref.shape[1:]) != (HEADS, HEAD_DIM, HEAD_DIM):
            raise ValueError(f"KDA recover state shape {tuple(ref.shape)}")
        for s in states:
            if s.shape != ref.shape or s.stride()[1:] != ref.stride()[1:]:
                raise ValueError("KDA recover layers need matching state layout")
        conv_ref = conv_states[0]
        conv_history_len = conv_ref.shape[2] - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA recover conv state is shorter than its window")
        for s in conv_states:
            if s.shape != conv_ref.shape or s.dtype != conv_ref.dtype:
                raise ValueError("KDA recover layers need matching conv state")
        device = ref.device

        def i64(vals):
            return torch.tensor(list(vals), dtype=torch.int64, device=device)

        if not pool.is_contiguous():
            raise ValueError("KDA recover record pool must be contiguous")
        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=i64(s.data_ptr() for s in conv_states),
            conv_state_block_strides=i64(s.stride(0) for s in conv_states),
            conv_state_dim_strides=i64(s.stride(1) for s in conv_states),
            conv_state_token_strides=i64(s.stride(2) for s in conv_states),
            conv_history_len=conv_history_len,
            states=tuple(states),
            state_base_addrs=i64(s.data_ptr() for s in states),
            state_block_strides=i64(s.stride(0) for s in states),
            records=pool,
            record_index=i64(index),
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            boundary_state_indices=torch.empty(max_num_reqs, dtype=torch.int32,
                                               device=device),
            boundary_recovery_lens=torch.empty(max_num_reqs, dtype=torch.int32,
                                               device=device),
            spec_query_len=spec_query_len,
        )

    def commit(self, num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor,
               query_start_loc: torch.Tensor, request_indices: torch.Tensor | None = None,
               block_table: torch.Tensor | None = None,
               num_computed_tokens: torch.Tensor | None = None,
               mamba_block_size: int | None = None) -> None:
        """Replay the accepted tokens of every spec-decode row in every layer."""
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
            _compact_conv_state_kernel,
            _prepare_commit_plan_kernel,
        )

        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0] or batch > self.records.shape[2]:
            raise ValueError("KDA recover commit batch exceeds its capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA recover commit metadata is incompatible")
        align = block_table is not None
        if align and (num_computed_tokens is None or mamba_block_size is None):
            raise ValueError("KDA recover align metadata is incomplete")
        bt_stride = block_table.stride() if align else (0, 0)
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens, request_indices, state_indices, query_start_loc,
            block_table, num_computed_tokens, self.commit_lens,
            self.final_state_indices, self.boundary_state_indices,
            self.boundary_recovery_lens, NULL_BLOCK_ID, mamba_block_size or 1,
            block_table.shape[1] if align else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0), query_start_loc.stride(0),
            bt_stride[0], bt_stride[1],
            num_computed_tokens.stride(0) if align else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        num_layers = len(self.states)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            conv_ref, self.conv_state_base_addrs, self.conv_state_block_strides,
            self.conv_state_dim_strides, self.conv_state_token_strides,
            state_indices, self.commit_lens, self.final_state_indices,
            self.boundary_state_indices, self.boundary_recovery_lens,
            NULL_BLOCK_ID, conv_dim, self.conv_history_len, state_indices.stride(0),
            BLOCK_D=256, BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align, num_warps=4,
        )
        ref = self.states[0]
        _kda_recover_commit_kernel[(HEAD_DIM // COMMIT_BV, batch, num_layers * HEADS)](
            ref, self.state_base_addrs, self.state_block_strides, self.records,
            self.record_index,
            state_indices, self.commit_lens, self.final_state_indices,
            self.boundary_state_indices, self.boundary_recovery_lens,
            NULL_BLOCK_ID, ref.stride(1), ref.stride(2), state_indices.stride(0),
            self.records.stride(0), self.records.stride(1),
            H=HEADS, D=HEAD_DIM, BV=COMMIT_BV, WS_T=WS_T, ALIGN_MODE=align,
            num_warps=COMMIT_WARPS,
        )


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class KDARecoverCommitMetadata:
    state_indices: torch.Tensor           # [num_spec_decodes] state slots read by the verify
    query_start_loc: torch.Tensor         # [num_spec_decodes + 1]
    request_indices: torch.Tensor | None  # batch row of each spec decode (None: identity)
    block_table: torch.Tensor | None      # align mode only
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


@dataclasses.dataclass
class Glm5KDARecoverMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recover_commit: KDARecoverCommitMetadata | None = None
    recover_context: "KDARecoverCommitContext | None" = dataclasses.field(
        default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        c = self.recover_commit
        if c is None or self.recover_context is None:
            return None
        self.recover_context.commit(
            num_accepted_tokens, c.state_indices, c.query_start_loc,
            request_indices=c.request_indices, block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens, mamba_block_size=c.block_size)
        if c.block_table is None:
            return None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens,
            block_size=c.block_size,
        )


class BuilderRecover:
    """Per GDN metadata builder state for VLLM_GLM5_KDA_RECOVER."""

    def __init__(self, builder) -> None:
        self.builder = builder
        rows = builder.decode_cudagraph_max_bs
        self.ones = torch.ones(rows, dtype=torch.int32, device=builder.device)
        self.context: KDARecoverCommitContext | None = None

    def get_context(self) -> KDARecoverCommitContext:
        if self.context is None:
            b = self.builder
            fc = b.vllm_config.compilation_config.static_forward_context
            layers = [fc[name] for name in b.layer_names]
            self.context = KDARecoverCommitContext.create(
                layers, spec_query_len=1 + b.num_spec,
                max_num_reqs=b.vllm_config.scheduler_config.max_num_seqs)
        return self.context

    def wrap(self, meta: GDNAttentionMetadata, m,
             spec_sequence_masks_cpu: torch.Tensor | None) -> GDNAttentionMetadata:
        """Add the commit metadata; the verify reads num_accepted = 1."""
        n = meta.num_spec_decodes
        if n == 0 or meta.spec_state_indices_tensor is None:
            return meta
        b = self.builder
        rows = meta.num_accepted_tokens.shape[0] if meta.num_accepted_tokens is not None else n
        request_indices = None
        pure = meta.num_prefills == 0 and meta.num_decodes == 0
        if not pure:
            assert spec_sequence_masks_cpu is not None
            idx = spec_sequence_masks_cpu.nonzero().flatten().to(torch.int32)
            request_indices = idx.pin_memory().to(b.device, non_blocking=True) \
                if torch.cuda.is_available() else idx.to(b.device)
        align = b.kv_cache_spec.mamba_cache_mode == "align"
        commit = KDARecoverCommitMetadata(
            state_indices=meta.spec_state_indices_tensor[:n, 0],
            query_start_loc=meta.spec_query_start_loc[: n + 1],
            request_indices=request_indices,
            block_table=m.block_table_tensor if align else None,
            num_computed_tokens=m.compute_num_computed_tokens() if align else None,
            block_size=b.kv_cache_spec.block_size if align else None,
        )
        fields = {f.name: getattr(meta, f.name) for f in dataclasses.fields(meta)}
        fields["num_accepted_tokens"] = self.ones[:rows]
        return Glm5KDARecoverMetadata(**fields, recover_commit=commit,
                                      recover_context=self.get_context())


# ---------------------------------------------------------------------------
# reference (torch, any device): what the commit computes, for tests
# ---------------------------------------------------------------------------
def commit_reference(states: torch.Tensor, records: torch.Tensor, src: int, n: int,
                     row: int) -> torch.Tensor:
    """State of one layer after replaying `n` tokens of record row `row` over
    states[src]: h = fma(c, k, h * e), fp32 fma emulated in fp64 (the product
    of two fp32 values is exact in fp64)."""
    h = states[src].clone()                          # [H, V, K]
    c, k, e = records[0, row], records[1, row], records[2, row]   # [WS_T, H, D]
    for t in range(n):
        he = h * e[t][:, None, :]
        h = (c[t][:, :, None].double() * k[t][:, None, :].double()
             + he.double()).to(torch.float32)
    return h
