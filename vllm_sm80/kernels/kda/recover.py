# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM for GLM-5 KDA layers on the fused sm_80 decode path (TP only).

Speculative verification normally keeps one recurrent state per draft
position per request (`num_speculative_blocks = num_spec` extra state pages per
KDA layer), so that the state at the accepted position can be picked up next
step.  With RecoverSSM (on wherever kda_recover_enabled() holds: tensor
parallel with 16 KDA heads per card, up to 7 drafts, up to 8 sequences; the
diagnostic VLLM_AMPERE_KDA_RECOVER_OFF=1 turns it off there, never for
serving):

  * the KV spec reserves no draft-position state pages
    (`num_speculative_blocks = 0`); the attention block size is unchanged;
  * the verify step (kda_decode_v2 with `records`) starts from the request's
    single state, writes no state and stores, per token, the correction
    c_t [V], the normalised key k_t [K] and the decay gate e_t [K] (fp32) into
    a per-running-request record buffer (one per KDA layer, allocated once);
  * after sampling, `KDARecoverCommitContext.commit` replays the accepted
    tokens over the state with the same fma sequence (bitwise the per-token
    state) and writes it once, compacts the conv window and, in align mode,
    also writes the block-boundary state, using the RecoverSSM commit-plan
    kernel (commit_plan.py) and the conv-compaction kernel of the upstream
    RecoverSSM implementation.

Accepted-token bookkeeping follows upstream RecoverSSM: the verify always
sees num_accepted = 1 (the committed state and conv window are already at the
accepted position) and the align postprocess resets the running state column.

COMMIT FOLD (VLLM_AMPERE_KDA_FOLD=1, off by default; kda_fold_enabled()): the
commit of a row whose state nobody reads before its next verify moves into
that verify.  After sampling, the commit plan (_kda_fold_plan_kernel) splits
the rows:

  * eager (align mode only): the step crosses or ends on a block boundary,
    commits to another block than it read, or the next step's tokens could
    spill into the next block (the align pre-copy would copy the state; it
    would copy the uncommitted state and the next verify would replay it
    there, but the rule keeps every pre-copy on committed states).  These
    commit exactly as without the fold;
  * deferred (every other row): nothing is written; the plan stores the
    commit length the request owes (FoldState.pend, per KDA cache group and
    request slot).

The next verify of a deferred row (kda_decode_v2 with `records.fold`) replays
the owed records over the state it loads with the same fma(c, k, h * e)
sequence, stores that state back to the same slot (the commit's store, so the
slot ends the step holding the committed state) and reads the conv window
shifted by the owed count (the commit's compaction, as the per-draft-slot step
reads it).  Records are kept per request slot instead of per verify row so
that they survive to the next step.  The fold's buffers are static per layer
(BuilderRecover sets them on the layers' LayerRecords before capture).  The
GLM model state (vllm_sm80/models/glm5next/model_state.py) holds FoldHooks
(kda_fold_hooks, None with the fold off) and calls them at three points:

  * prepare_step, after building a step's metadata (FoldState.gather: per
    verify row, the request slot and the owed lengths);
  * add_request: a slot taken by a request owes nothing.  A request id
    re-added in the same step it was removed is a streaming-input session
    whose state is kept (the worker removes and re-adds it with the next
    input chunk, whose prefill reads that state): its owed commit is written
    first (FoldState.flush, the commit's own plan values and kernels);
  * remove_request: remembers the removed request until the next step.  A
    second removal of the same id in one step drops that record: a request
    finished or preempted in finish_requests and re-added in the same step
    (Scheduler.reset_prefix_cache(reset_running_requests=True) preempts and
    resumes running requests in one step) is removed again by add_requests'
    streaming-update removal before it is re-added.

Consumers that read a committed state between steps (align pre-copy and
post-copy, prefix-cache hits on boundary blocks, a streaming session's next
chunk) only ever see eager or flushed rows.  A finished or preempted request
is never flushed, whenever it is re-added: its blocks are freed and may
already belong to another request.

The per-draft-slot chain (fused_recurrent_kda) cannot pay an owed commit; it
is not reached under RecoverSSM: a pure verify takes the recover verify
whatever the shape gate says, a mixed step's spec rows take it whenever the
deferred projections and fp32 states are there, which RecoverSSM's gate
guarantees (and that chain would already read draft-position state pages
RecoverSSM does not allocate).
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

WS_T = 8                     # token rows per record (kda_decode._WS_T)
HEADS = 16                   # KDA heads per card the verify kernel covers here
HEAD_DIM = 128
_ARCHS = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")

def _closed_reason(vllm_config) -> str | None:
    from vllm_sm80.kernels._platform import _is_sm80

    num_spec = vllm_config.num_speculative_tokens or 0
    if num_spec < 1 or num_spec + 1 > WS_T:
        return f"needs 1..{WS_T - 1} speculative tokens (have {num_spec})"
    pc = vllm_config.parallel_config
    if pc.pipeline_parallel_size != 1:
        return "pipeline parallel is not supported"
    text = vllm_config.model_config.hf_text_config
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


def _recover_off_unused(reason: str) -> None:
    from vllm import envs

    if envs.VLLM_AMPERE_KDA_RECOVER_OFF:
        logger.info_once(
            "VLLM_AMPERE_KDA_RECOVER_OFF=1 set but has no effect: RecoverSSM is "
            "off for this configuration anyway (%s).",
            reason,
        )


def kda_recover_enabled(vllm_config) -> bool:
    """RecoverSSM for this configuration: a GLM-5 Next model with the sm_80
    decode kernels on (VLLM_AMPERE_DECODE_KERNELS=1) and the configuration the
    verify kernel is built for, unless VLLM_AMPERE_KDA_RECOVER_OFF=1.

    VLLM_AMPERE_KDA_RECOVER_OFF=1 is a diagnostic, never for serving: it
    returns False where the gate holds, so every caller takes the path a
    closed gate takes (one KDA state per draft position, upstream's
    speculative state handling). It gives a same-tree reference without
    state recovery.

    Cheap; called at init by the KDA layers, the metadata builders and the
    model state. Logs the banner (or why it stays off) once.
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS:
        _recover_off_unused("VLLM_AMPERE_DECODE_KERNELS is off")
        return False
    model_config = vllm_config.model_config
    archs = tuple(getattr(model_config.hf_config, "architectures", None) or ())
    if not any(a in _ARCHS for a in archs):
        return False
    reason = _closed_reason(vllm_config)
    if reason is None:
        if envs.VLLM_AMPERE_KDA_RECOVER_OFF:
            logger.warning_once(
                "sm_80 KDA recover off by VLLM_AMPERE_KDA_RECOVER_OFF=1 (a "
                "diagnostic, not for serving): keeping one KDA state per draft "
                "position."
            )
            return False
        logger.info_once(
            "sm_80 KDA recover active: one KDA state per request, accepted "
            "tokens replayed after sampling."
        )
        return True
    _recover_off_unused(reason)
    logger.info_once(
        "sm_80 KDA recover off: %s; keeping one KDA state per draft position.",
        reason,
    )
    return False


def kda_fold_enabled(vllm_config) -> bool:
    """The RecoverSSM commit fold (module docstring): VLLM_AMPERE_KDA_FOLD=1 on
    a configuration where RecoverSSM is on, without data parallelism (a
    data-parallel dummy batch would run a verify outside the request slots'
    bookkeeping).

    Cheap; called at init by the metadata builders (BuilderRecover: the fold
    buffers), the KDA decode warmup (the fold variants) and the GLM model
    state (kda_fold_hooks). Logs the banner (or why it stays off) once.
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_KDA_FOLD:
        return False
    if not kda_recover_enabled(vllm_config):
        logger.info_once(
            "VLLM_AMPERE_KDA_FOLD=1 set but off: RecoverSSM is off for this "
            "configuration; the KDA commit runs after sampling."
        )
        return False
    dp = int(getattr(vllm_config.parallel_config, "data_parallel_size", 1) or 1)
    if dp != 1:
        logger.info_once(
            "VLLM_AMPERE_KDA_FOLD=1 set but off: needs data parallel size 1 "
            "(have %d); the KDA commit runs after sampling.", dp,
        )
        return False
    logger.info_once(
        "sm_80 KDA commit fold active (VLLM_AMPERE_KDA_FOLD=1): accepted tokens "
        "replayed in the next verify; block-boundary steps commit after "
        "sampling."
    )
    return True


def records_rows(vllm_config) -> int:
    return int(vllm_config.scheduler_config.max_num_seqs)


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


class LayerRecords(tuple):
    """(corrections, keys, gates) of one KDA layer, views of the card's
    RecordPool; `fold` (commit fold, set by BuilderRecover): the layer's
    (owed lengths per verify row, record row per verify row) FoldState
    buffers, or None."""

    fold = None

    def __new__(cls, c, k, e):
        return super().__new__(cls, (c, k, e))


def layer_records(pool: torch.Tensor, index: int) -> LayerRecords:
    return LayerRecords(pool[index, 0], pool[index, 1], pool[index, 2])


MAX_FOLD_GROUPS = 8          # KDA cache groups one FoldState holds (GLM: 4)


class FoldState:
    """Per-card bookkeeping of the commit fold, shared by the KDA cache
    groups (allocated once, before CUDA-graph capture):

      pend [MAX_FOLD_GROUPS, max_num_reqs] int32: per group and request slot,
          the commit length the slot's next verify replays (0: committed);
          written by the commit plan, cleared by reset_slot and flush;
      src [MAX_FOLD_GROUPS, max_num_reqs] int32: per group and request slot,
          the state block of the owed commit (written with pend; flush);
      lens [MAX_FOLD_GROUPS, rows] int32: per group and verify row, the owed
          length of this step (gather), read by the verify;
      record_rows [rows] int32: per verify row, the request slot (-1: no
          request), the record row the verify writes and the commit reads.

    `rows` covers every row a verify can see (BuilderRecover.ones).  Host
    side: the owner (request id) of every slot and the requests removed in
    the current step (FoldHooks).
    """

    _states: dict = {}

    def __init__(self, device, max_num_reqs: int, rows: int) -> None:
        self.device = torch.device(device)
        self.max_num_reqs = int(max_num_reqs)
        self.rows = int(rows)
        # empty + fill (as the builders' own buffers): CPU harnesses that
        # redirect the builders' CUDA allocations cover these too
        self.pend = torch.empty(MAX_FOLD_GROUPS, self.max_num_reqs,
                                dtype=torch.int32, device=device).fill_(0)
        self.lens = torch.empty(MAX_FOLD_GROUPS, self.rows, dtype=torch.int32,
                                device=device).fill_(0)
        self.record_rows = torch.empty(self.rows, dtype=torch.int32,
                                       device=device).fill_(-1)
        self.src = torch.empty(MAX_FOLD_GROUPS, self.max_num_reqs,
                               dtype=torch.int32, device=device).fill_(0)
        # one-row plan of a flush: state block, record row
        self.flush_src = torch.empty(1, dtype=torch.int32, device=device).fill_(0)
        self.flush_row = torch.empty(1, dtype=torch.int32, device=device).fill_(0)
        self.groups: dict = {}
        self.builders: dict = {}   # group -> BuilderRecover (flush)
        self.probe = None      # a layer name of the first group (prepare_step)
        self.owner: list = [None] * self.max_num_reqs
        self.removed: dict = {}    # request id -> slot, removed this step

    @staticmethod
    def _key(device):
        d = torch.device(device)
        index = d.index
        if d.type == "cuda" and index is None:
            index = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
        return (d.type, index)

    @classmethod
    def get(cls, device, max_num_reqs: int, rows: int) -> "FoldState":
        key = cls._key(device)
        s = cls._states.get(key)
        if s is None:
            s = cls(device, max_num_reqs, rows)
            cls._states[key] = s
        if s.max_num_reqs != max_num_reqs or s.rows != rows:
            raise ValueError("KDA commit fold: builders disagree on the row counts")
        return s

    @classmethod
    def peek(cls, device) -> "FoldState | None":
        return cls._states.get(cls._key(device))

    def group(self, layer_names) -> int:
        """The group index of a KDA cache group, stable across builder
        rebuilds (memory profiling builds the builders once more)."""
        key = tuple(layer_names)
        g = self.groups.get(key)
        if g is None:
            g = len(self.groups)
            if g >= MAX_FOLD_GROUPS:
                raise ValueError(f"KDA commit fold: more than {MAX_FOLD_GROUPS} groups")
            self.groups[key] = g
            if self.probe is None:
                self.probe = key[0]
        return g

    def reset_slot(self, slot: int) -> None:
        """A request slot taken by a new (or resumed) request owes nothing."""
        self.pend[:, slot].fill_(0)

    def flush(self, slot: int) -> None:
        """Write the commit `slot` owes in every group now, as the eager
        commit would have written it (the same conv compaction and commit
        kernels over the same plan values: final = source block, no boundary
        state), and clear the debt. No-op on the device for a slot that owes
        nothing."""
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel

        for g, b in sorted(self.builders.items()):
            ctx = b.get_context()
            _kda_fold_flush_plan_kernel[(1,)](
                self.pend[g], self.src[g], slot, ctx.commit_lens, ctx.final_state_indices,
                ctx.boundary_state_indices, ctx.boundary_recovery_lens, self.flush_src,
                self.flush_row, NULL_BLOCK_ID, num_warps=1,
            )
            conv_ref = ctx.conv_states[0]
            conv_dim = conv_ref.shape[1]
            num_layers = len(ctx.states)
            _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), 1, num_layers)](
                conv_ref, ctx.conv_state_base_addrs, ctx.conv_state_block_strides,
                ctx.conv_state_dim_strides, ctx.conv_state_token_strides,
                self.flush_src, ctx.commit_lens, ctx.final_state_indices,
                ctx.boundary_state_indices, ctx.boundary_recovery_lens,
                NULL_BLOCK_ID, conv_dim, ctx.conv_history_len, 1,
                BLOCK_D=256, BLOCK_HISTORY=triton.next_power_of_2(ctx.conv_history_len),
                ALIGN_MODE=False, num_warps=4,
            )
            ref = ctx.states[0]
            _kda_recover_commit_kernel[(HEAD_DIM // COMMIT_BV, 1, num_layers * HEADS)](
                ref, ctx.state_base_addrs, ctx.state_block_strides, ctx.records,
                ctx.record_index, self.flush_row,
                self.flush_src, ctx.commit_lens, ctx.final_state_indices,
                ctx.boundary_state_indices, ctx.boundary_recovery_lens,
                NULL_BLOCK_ID, ref.stride(1), ref.stride(2), 1,
                ctx.records.stride(0), ctx.records.stride(1),
                H=HEADS, D=HEAD_DIM, BV=COMMIT_BV, WS_T=WS_T, ALIGN_MODE=False,
                HAS_RECORD_ROWS=True, num_warps=COMMIT_WARPS,
            )

    def gather(self, idx_mapping: torch.Tensor,
               request_indices: torch.Tensor | None, num_spec: int) -> None:
        """Once per step: per verify row, the request slot and every group's
        owed length."""
        if num_spec > self.rows:
            raise ValueError(f"KDA commit fold: {num_spec} rows exceed {self.rows}")
        _kda_fold_gather_kernel[(1,)](
            idx_mapping, request_indices, self.pend, self.lens, self.record_rows,
            num_spec, self.rows, idx_mapping.shape[0], self.max_num_reqs,
            self.pend.stride(0), self.lens.stride(0),
            G=len(self.groups), BLOCK=triton.next_power_of_2(self.rows),
            num_warps=1,
        )


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _kda_fold_gather_kernel(
    idx_mapping_ptr,           # [num_reqs] batch row -> request slot (-1: none)
    request_indices_ptr,       # [num_spec] verify row -> batch row (None: identity)
    pend_ptr,                  # [MAX_FOLD_GROUPS, max_num_reqs] int32
    lens_ptr,                  # [MAX_FOLD_GROUPS, rows] int32 (out)
    record_rows_ptr,           # [rows] int32 (out)
    num_spec,
    rows,
    num_reqs,
    max_num_reqs,
    stride_pend,
    stride_lens,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
):
    r = tl.arange(0, BLOCK)
    live = r < num_spec
    if HAS_REQUEST_INDICES:
        b = tl.load(request_indices_ptr + r, mask=live, other=-1)
    else:
        b = r
    ok = live & (b >= 0) & (b < num_reqs)
    slot = tl.load(idx_mapping_ptr + b, mask=ok, other=-1)
    ok = ok & (slot >= 0) & (slot < max_num_reqs)
    slot = tl.where(ok, slot, -1)
    in_rows = r < rows
    tl.store(record_rows_ptr + r, slot, mask=in_rows)
    for g in tl.static_range(G):
        n = tl.load(pend_ptr + g * stride_pend + slot, mask=ok, other=0)
        tl.store(lens_ptr + g * stride_lens + r, n, mask=in_rows)


@triton.jit
def _kda_fold_flush_plan_kernel(
    pend_ptr,                  # [max_num_reqs] int32: this group's owed lengths
    src_ptr,                   # [max_num_reqs] int32: this group's owed blocks
    slot,
    commit_lens_ptr,           # the context's one-row plan (out)
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    flush_src_ptr,             # [1] int32 state block (out)
    flush_row_ptr,             # [1] int32 record row (out)
    null_block_id,
):
    n = tl.load(pend_ptr + slot)
    s = tl.load(src_ptr + slot)
    owes = n > 0
    tl.store(commit_lens_ptr, tl.where(owes, n, 0))
    tl.store(final_state_indices_ptr, tl.where(owes, s, null_block_id))
    tl.store(boundary_state_indices_ptr, null_block_id)
    tl.store(boundary_recovery_lens_ptr, 0)
    tl.store(flush_src_ptr, tl.where(owes, s, null_block_id))
    tl.store(flush_row_ptr, slot)
    tl.store(pend_ptr + slot, 0)


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
    record_rows_ptr,           # HAS_RECORD_ROWS: [rows] int32 record row of each row
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
    HAS_RECORD_ROWS: tl.constexpr = False,
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
    if HAS_RECORD_ROWS:
        # commit fold: records are kept per request slot
        i_row = tl.maximum(tl.load(record_rows_ptr + i_b), 0)
    else:
        i_row = i_b
    rb = records_ptr + i_r * stride_rec_layer + (i_row * WS_T * H + i_h) * D
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


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _kda_fold_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    record_rows_ptr,             # [rows] int32: request slot of each row (-1: none)
    pend_ptr,                    # [max_num_reqs] int32: this group's owed lengths
    src_ptr,                     # [max_num_reqs] int32: this group's owed blocks
    commit_lens_ptr,             # eager commit length (0: deferred or nothing)
    true_lens_ptr,               # the commit length (_prepare_commit_plan_kernel's)
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    NEXT_WINDOW: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    """_prepare_commit_plan_kernel (commit_plan.py) line for line, plus the
    eager / deferred split of the commit fold."""
    spec_idx = tl.program_id(0)
    slot = tl.load(record_rows_ptr + spec_idx)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    if (source_state_idx <= null_block_id) | (request_idx < 0):
        tl.store(commit_lens_ptr + spec_idx, 0)
        tl.store(true_lens_ptr + spec_idx, 0)
        tl.store(final_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_recovery_lens_ptr + spec_idx, 0)
        if (request_idx >= 0) & (slot >= 0):
            tl.store(pend_ptr + slot, 0)
        return
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)
    if commit_len <= 0:
        tl.store(commit_lens_ptr + spec_idx, 0)
        tl.store(true_lens_ptr + spec_idx, 0)
        tl.store(final_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_state_indices_ptr + spec_idx, null_block_id)
        tl.store(boundary_recovery_lens_ptr + spec_idx, 0)
        if slot >= 0:
            tl.store(pend_ptr + slot, 0)
        return

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    defer = commit_len > 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        final_state_col = tl.maximum((final_num_computed - 1) // mamba_block_size, 0)
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col,
            mask=final_state_col < block_table_width,
            other=null_block_id,
        ).to(tl.int64)
        tl.device_assert(
            final_state_idx > null_block_id, "RecoverSSM final state has no owner"
        )
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=(
                crosses_boundary
                & (next_boundary // mamba_block_size - 1 >= 0)
                & (next_boundary // mamba_block_size - 1 < block_table_width)
            ),
            other=null_block_id,
        ).to(tl.int64)
        tl.device_assert(
            (~crosses_boundary) | (boundary_state_idx > null_block_id),
            "RecoverSSM boundary state has no owner",
        )
        # Deferred only if the commit stays in the block the verify read, no
        # boundary state is due, and the next step's num_spec + 1 tokens stay in
        # that block (no pre-copy or post-copy reads it before the next verify).
        next_col = (final_num_computed + NEXT_WINDOW - 1) // mamba_block_size
        defer = (
            (final_state_idx == source_state_idx)
            & (~crosses_boundary)
            & (next_col == final_state_col)
        )
    valid = (final_state_idx > null_block_id) & (commit_len > 0)
    defer = defer & valid
    eager = valid & (~defer)
    tl.store(commit_lens_ptr + spec_idx, tl.where(eager, commit_len, 0))
    tl.store(true_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )
    if slot >= 0:
        tl.store(pend_ptr + slot, tl.where(defer, commit_len, 0))
        tl.store(src_ptr + slot, source_state_idx.to(tl.int32))


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
    true_lens: torch.Tensor        # commit fold: the commit length of every row
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
            true_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            spec_query_len=spec_query_len,
        )

    def commit(self, num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor,
               query_start_loc: torch.Tensor, request_indices: torch.Tensor | None = None,
               block_table: torch.Tensor | None = None,
               num_computed_tokens: torch.Tensor | None = None,
               mamba_block_size: int | None = None,
               fold: "tuple[FoldState, int] | None" = None) -> None:
        """Replay the accepted tokens of every spec-decode row in every layer.

        `fold` (commit fold: the card's FoldState and this group's index):
        only the eager rows are replayed here; the deferred rows' lengths go
        to FoldState.pend for their next verify, and `true_lens` holds every
        row's commit length."""
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel
        from vllm_sm80.kernels.kda.commit_plan import _prepare_commit_plan_kernel

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
        record_rows = None
        if fold is None:
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
        else:
            state, group = fold
            record_rows = state.record_rows
            _kda_fold_plan_kernel[(batch,)](
                num_accepted_tokens, request_indices, state_indices, query_start_loc,
                block_table, num_computed_tokens, record_rows, state.pend[group],
                state.src[group],
                self.commit_lens, self.true_lens,
                self.final_state_indices, self.boundary_state_indices,
                self.boundary_recovery_lens, NULL_BLOCK_ID, mamba_block_size or 1,
                block_table.shape[1] if align else 1,
                num_accepted_tokens.stride(0),
                request_indices.stride(0) if request_indices is not None else 0,
                state_indices.stride(0), query_start_loc.stride(0),
                bt_stride[0], bt_stride[1],
                num_computed_tokens.stride(0) if align else 0,
                SPEC_QUERY_LEN=self.spec_query_len,
                NEXT_WINDOW=self.spec_query_len,
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
            self.record_index, record_rows,
            state_indices, self.commit_lens, self.final_state_indices,
            self.boundary_state_indices, self.boundary_recovery_lens,
            NULL_BLOCK_ID, ref.stride(1), ref.stride(2), state_indices.stride(0),
            self.records.stride(0), self.records.stride(1),
            H=HEADS, D=HEAD_DIM, BV=COMMIT_BV, WS_T=WS_T, ALIGN_MODE=align,
            HAS_RECORD_ROWS=record_rows is not None,
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


# The GLM KDA layer's GDN metadata (built by the GDN builder of
# vllm_sm80/models/glm5next/sm80_kda.py). It lives here, not beside that
# builder, because importing any module of the GLM model package imports the
# model and FLA, whose platform check initialises CUDA, and this module must
# import without initialising CUDA (tests/kernels/test_ampere_kda_recover.py).
@dataclasses.dataclass
class Glm5NextGDNAttentionMetadata(GDNAttentionMetadata):
    # Bound on the tokens of any spec-decode request of the step: the batch's
    # max_query_len (what its CUDA graph was captured for), capped at
    # num_spec + 1. None without spec decodes.
    spec_max_query_len: int | None = None


@dataclasses.dataclass
class Glm5KDARecoverMetadata(Glm5NextGDNAttentionMetadata, RecoverSSMMetadata):
    recover_commit: KDARecoverCommitMetadata | None = None
    # the builder's BuilderRecover: the commit context is created on the first
    # commit (after the KV caches are bound), not when metadata is built
    recover_context: "BuilderRecover | None" = dataclasses.field(
        default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        c = self.recover_commit
        if c is None or self.recover_context is None:
            return None
        context = self.recover_context.get_context()
        fold = None
        state = getattr(self.recover_context, "fold", None)
        if state is not None:
            fold = (state, self.recover_context.fold_group)
        context.commit(
            num_accepted_tokens, c.state_indices, c.query_start_loc,
            request_indices=c.request_indices, block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens, mamba_block_size=c.block_size,
            fold=fold)
        if c.block_table is None:
            return None
        lens = context.true_lens if fold is not None else context.commit_lens
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens,
            block_size=c.block_size,
            commit_lens=lens[:self.num_spec_decodes],
            source_state_indices=c.state_indices,
        )


class FoldHooks:
    """The model state's calls into the commit fold (module docstring);
    kda_fold_hooks gives None with the fold off, so the off path makes no
    call. Each call looks up the card's FoldState, which the metadata
    builders create after the model state."""

    def __init__(self, device) -> None:
        self.device = device

    def prepare_step(self, attn_metadata: dict, idx_mapping: torch.Tensor,
                     for_capture: bool) -> None:
        """Once per real step after its metadata is built: map the verify
        rows to request slots and owed lengths (FoldState.gather). Nothing at
        capture and on steps without spec rows."""
        fold = FoldState.peek(self.device)
        if fold is None or for_capture:
            return
        fold.removed.clear()
        md = attn_metadata.get(fold.probe)
        if not isinstance(md, Glm5KDARecoverMetadata):
            return
        c = md.recover_commit
        fold.gather(idx_mapping, c.request_indices if c is not None else None,
                    md.num_spec_decodes)

    def add_request(self, slot: int, req_id: str, num_computed_tokens: int) -> None:
        """A request takes `slot`. A streaming-input session re-added in the
        step it was removed in, with its state kept, first pays what it owes
        (at its old slot, whose records are intact); the slot then owes
        nothing."""
        fold = FoldState.peek(self.device)
        if fold is None:
            return
        old = fold.removed.pop(req_id, None)
        if old is not None and num_computed_tokens > 0:
            fold.flush(old)
        fold.reset_slot(slot)
        fold.owner[slot] = req_id

    def remove_request(self, req_id: str) -> None:
        """Remember a removed request until the next step (add_request). A
        second removal in the step (finished or preempted, then removed again
        by add_requests before its re-add) forgets it: its state was freed."""
        fold = FoldState.peek(self.device)
        if fold is None:
            return
        if req_id not in fold.owner:
            fold.removed.pop(req_id, None)
            return
        slot = fold.owner.index(req_id)
        fold.owner[slot] = None
        fold.removed[req_id] = slot


def kda_fold_hooks(vllm_config, device) -> FoldHooks | None:
    """The model state's FoldHooks, None unless the commit fold is on."""
    return FoldHooks(device) if kda_fold_enabled(vllm_config) else None


class BuilderRecover:
    """Per GDN metadata builder state for RecoverSSM."""

    def __init__(self, builder) -> None:
        self.builder = builder
        # One row per request the verify can see. decode_cudagraph_max_bs is
        # bounded by the largest captured size, which is 0 without CUDA graphs
        # (enforce_eager): the verify would then read num_accepted from an
        # empty tensor.
        rows = max(builder.decode_cudagraph_max_bs,
                   builder.vllm_config.scheduler_config.max_num_seqs)
        self.ones = torch.ones(rows, dtype=torch.int32, device=builder.device)
        self.context: KDARecoverCommitContext | None = None
        # commit fold: the card's FoldState and this cache group's index
        self.fold: FoldState | None = None
        self.fold_group = -1
        if kda_fold_enabled(builder.vllm_config):
            max_seqs = builder.vllm_config.scheduler_config.max_num_seqs
            self.fold = FoldState.get(builder.device, max_seqs,
                                      max_seqs * (1 + builder.num_spec))
            if rows > self.fold.rows:
                raise ValueError(f"KDA commit fold: {rows} rows exceed {self.fold.rows}")
            self.fold_group = self.fold.group(builder.layer_names)
            self.fold.builders[self.fold_group] = self
            # The verify reads the fold buffers from the layer's records
            # (static: allocated once, before CUDA-graph capture).
            fc = builder.vllm_config.compilation_config.static_forward_context
            for name in builder.layer_names:
                records = getattr(fc.get(name), "_kda_recover_records", None)
                if records is not None:
                    records.fold = (self.fold.lens[self.fold_group],
                                    self.fold.record_rows)

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
        if rows > self.ones.shape[0]:
            raise ValueError(f"KDA recover: {rows} rows exceed {self.ones.shape[0]}")
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
                                      recover_context=self)
