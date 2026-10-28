# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy argmax over a vocabulary-parallel head without the all-gather of
the logits: the slot kernels, and greedy DSpark Markov sampling on them.

``VLLM_SM80_SHARD_ARGMAX`` covers greedy target sampling
(``vllm_sm80/worker/target_argmax.py``, which uses these kernels) and greedy
drafting over a vocabulary-parallel head. How each drafter takes its greedy
drafts with the switch on, and which drafters the switch refuses, is
:func:`drafting`.

The DSpark drafter samples its block left to right: at step i the draft
logits of position i are its base logits plus a Markov bias computed from the
token drafted at step i - 1, and a greedy draft is their argmax. Upstream
builds the base logits with the vocabulary-parallel head and all-gathers
them over the TP group (``compute_draft_logits``), only to take that argmax.

With ``VLLM_SM80_SHARD_ARGMAX`` the drafter's
``compute_logits_local`` keeps each rank's own columns of the base logits,
and every step:

1. adds the Markov bias's columns of that shard, with the same torch
   addition the gathered path runs over the whole row;
2. :func:`chunk_slots` reduces each ``BLOCK``-column chunk to the first index
   of its largest value and writes it into the chunk's 16-byte slot of a BF16
   buffer, every other rank's slots zero;
3. the buffer is summed over the TP group (``tensor_model_parallel_all_reduce``);
4. :func:`pick` takes, per row, the slot with the largest value, the lowest
   slot on ties, and returns its token.

When the drafter keeps the draft logits' top-1 probability and entropy (the
DSpark adaptive depth policy's ``logit_top1``/``ent`` features,
``_row_stats``), each rank also writes, behind the chunk slots of the same
buffer, its shard's softmax statistics of the row: ``m`` (the largest value),
``s = sum exp(l - m)`` and ``t = sum exp(l - m) * l``, each FP32 word as four
byte lanes (:func:`shard_stats`). After the sum every rank holds every rank's
words and combines them in rank order (:func:`combine_stats`):
``M = max m_r``, ``S = sum s_r exp(m_r - M)``, ``T = sum t_r exp(m_r - M)``,
top-1 probability ``1 / S`` and entropy ``M + log S - T / S``, the same bits on
every rank. These equal the gathered row's softmax statistics up to the
summation order.

Exactness. The draft is torch.argmax's over the gathered row: ATen's
``GreaterOrNan`` prefers a NaN to any number, the lower index between two
NaNs or two equal values, and compares -0.0 equal to +0.0. The kernels
compare a 32-bit key built from the value's bits alone: -0.0 becomes +0.0,
every NaN the largest key, and otherwise the key keeps the IEEE order. No
floating-point comparison runs, so no flush of subnormals can merge two
values. A chunk takes the lowest index among its largest keys, and the slots
lie in global column order (rank, then chunk), so the lowest slot among equal
keys holds the lowest index. A slot carries the key in four BF16 lanes of one
byte each and the global index in three (the eighth is zero): integers up to
255 are exact in BF16, and adding zeros is exact under any all-reduce
algorithm or accumulation dtype, so the buffer arrives intact. Columns at or
beyond the vocabulary size (padding) are never read, as the gathered path
cuts them off.
"""

import functools
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm_sm80 import hot_swap
from vllm.triton_utils import tl, triton

BLOCK = 4096  # columns per chunk; one slot per chunk
LANES = 8  # BF16 lanes per slot: 4 key bytes, 3 index bytes, 1 zero
STATS_LANES = 16  # BF16 lanes per rank's statistics: 3 FP32 words, 4 bytes each
MAX_VOCAB = 1 << 24  # three index bytes
SWITCH = "VLLM_SM80_SHARD_ARGMAX"


@triton.jit
def _lane_shift(lane):
    # Key bytes in lanes 0-3 and index bytes in lanes 4-6, most significant
    # first; lane 7 carries nothing.
    return tl.where(lane < 4, 24 - 8 * lane, tl.where(lane < 7, 48 - 8 * lane, 0))


@triton.jit
def _chunk_slots_kernel(
    logits_ptr,
    logits_stride,
    slots_ptr,
    row_stride,
    n_cols,
    col_start,
    rank,
    TP: tl.constexpr,
    TP_P2: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_cols
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=valid, other=0.0)
    bits = x.to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)  # -0.0 -> +0.0
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    wide = bits.to(tl.int64)
    # Non-negative floats above every negative one; negative floats ordered
    # by decreasing magnitude; NaN above +inf (0xFF800000).
    key = tl.where(wide >= 0, wide + 2147483648, -1 - wide)
    key = tl.where(is_nan, 4294967295, key)
    key = tl.where(valid, key, -1)
    best = tl.max(key, axis=0)
    first = tl.min(tl.where(key == best, offs, 2147483647), axis=0)
    # A chunk without columns sends key 0, below every value's key (-inf's is
    # 0x007FFFFF), so it is never picked.
    empty = best < 0
    enc_key = tl.where(empty, 0, best)
    index = tl.where(empty, 0, col_start + first.to(tl.int64))
    lane = tl.arange(0, 8)
    shift = _lane_shift(lane).to(tl.int64)
    src = tl.where(lane < 4, enc_key, index)
    byte = tl.where(lane < 7, (src >> shift) & 255, 0)
    r = tl.arange(0, TP_P2)
    vals = tl.where(r[:, None] == rank, byte[None, :], 0)
    ptrs = (
        slots_ptr
        + row * row_stride
        + (r[:, None] * CHUNKS + chunk) * 8
        + lane[None, :]
    )
    tl.store(ptrs, vals.to(tl.float32).to(tl.bfloat16), mask=r[:, None] < TP)


@triton.jit
def _pick_kernel(
    slots_ptr, row_stride, out_ptr, vocab, SLOTS: tl.constexpr, SLOTS_P2: tl.constexpr
):
    row = tl.program_id(0)
    s = tl.arange(0, SLOTS_P2)
    lane = tl.arange(0, 8)
    v = tl.load(
        slots_ptr + row * row_stride + s[:, None] * 8 + lane[None, :],
        mask=s[:, None] < SLOTS,
        other=0.0,
    )
    part = v.to(tl.float32).to(tl.int64) << _lane_shift(lane).to(tl.int64)[None, :]
    key = tl.sum(tl.where(lane[None, :] < 4, part, 0), axis=1)
    index = tl.sum(tl.where((lane[None, :] >= 4) & (lane[None, :] < 7), part, 0), axis=1)
    key = tl.where(s < SLOTS, key, -1)
    best = tl.max(key, axis=0)
    first = tl.min(tl.where(key == best, s, SLOTS_P2), axis=0)
    token = tl.sum(tl.where(s == first, index, 0), axis=0)
    # Summed slots always hold a column below vocab; a CUDA-graph warmup run
    # inside the custom all-reduce's capture context gets an unwritten buffer
    # back (torch.empty_like: the flags-in-data all-reduce's warmup copy,
    # vllm_sm80/distributed/communicator.py custom_all_reduce, and upstream's
    # custom_all_reduce.py), so keep its token a valid embedding row.
    tl.store(out_ptr + row, tl.minimum(tl.maximum(token, 0), vocab - 1))


def chunk_slots(
    logits: torch.Tensor,
    col_start: int,
    rank: int,
    tp_size: int,
    chunks: int,
    stats: bool = False,
) -> torch.Tensor:
    """This rank's slots of one sampling step.

    ``logits`` [rows, n] FP32 holds global columns ``col_start ..
    col_start + n - 1`` (n <= chunks * BLOCK). Returns [rows, tp_size * chunks
    * LANES] BF16 (with ``stats``, ``tp_size * STATS_LANES`` more lanes holding
    :func:`shard_stats`): this rank's ``chunks`` slots filled, every other
    rank's zero, ready to be summed over the TP group.
    """
    rows, n_cols = logits.shape
    if logits.dtype != torch.float32 or logits.stride(-1) != 1:
        raise ValueError("chunk_slots takes FP32 logits with unit column stride")
    if n_cols > chunks * BLOCK or col_start + n_cols > MAX_VOCAB:
        raise ValueError(f"{n_cols} columns from {col_start} exceed the slots")
    width = tp_size * chunks * LANES + (tp_size * STATS_LANES if stats else 0)
    slots = torch.empty(rows, width, dtype=torch.bfloat16, device=logits.device)
    if rows:
        _chunk_slots_kernel[(rows, chunks)](
            logits,
            logits.stride(0),
            slots,
            slots.stride(0),
            n_cols,
            col_start,
            rank,
            TP=tp_size,
            TP_P2=triton.next_power_of_2(tp_size),
            CHUNKS=chunks,
            BLOCK=BLOCK,
            num_warps=8,
        )
        if stats:
            shard_stats(logits, slots, tp_size * chunks * LANES, rank, tp_size, chunks)
    return slots


@triton.jit
def _shard_stats_kernel(
    logits_ptr,
    logits_stride,
    slots_ptr,
    row_stride,
    base,
    n_cols,
    rank,
    TP: tl.constexpr,
    TP_P2: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = tl.full((), float("-inf"), tl.float32)
    for c in tl.static_range(CHUNKS):
        cols = c * BLOCK + offs
        x = tl.load(logits_ptr + row * logits_stride + cols, mask=cols < n_cols,
                    other=float("-inf"))
        m = tl.maximum(m, tl.max(x, axis=0))
    s = tl.zeros((), tl.float32)
    t = tl.zeros((), tl.float32)
    if m > float("-inf"):
        for c in tl.static_range(CHUNKS):
            cols = c * BLOCK + offs
            valid = cols < n_cols
            x = tl.load(logits_ptr + row * logits_stride + cols, mask=valid, other=0.0)
            e = tl.where(valid, tl.exp(x - m), 0.0)
            s += tl.sum(e, axis=0)
            # A -inf column has e = 0 and adds nothing (not 0 * -inf).
            t += tl.sum(tl.where(e > 0.0, e * x, 0.0), axis=0)
    lane = tl.arange(0, 16)
    word = lane // 4
    m_bits = m.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    s_bits = s.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    t_bits = t.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    bits = tl.where(word == 0, m_bits, tl.where(word == 1, s_bits, tl.where(word == 2, t_bits, 0)))
    byte = (bits >> (24 - 8 * (lane % 4)).to(tl.int64)) & 255
    r = tl.arange(0, TP_P2)
    vals = tl.where(r[:, None] == rank, byte[None, :], 0)
    ptrs = slots_ptr + row * row_stride + base + r[:, None] * 16 + lane[None, :]
    tl.store(ptrs, vals.to(tl.float32).to(tl.bfloat16), mask=r[:, None] < TP)


def shard_stats(
    logits: torch.Tensor, slots: torch.Tensor, base: int, rank: int, tp_size: int, chunks: int
) -> None:
    """Write this rank's softmax statistics of each row of ``logits`` (its
    shard's columns) into lanes ``base + rank * STATS_LANES ..`` of ``slots``,
    zeros in every other rank's lanes (see the module docstring)."""
    rows, n_cols = logits.shape
    if rows:
        _shard_stats_kernel[(rows,)](
            logits,
            logits.stride(0),
            slots,
            slots.stride(0),
            base,
            n_cols,
            rank,
            TP=tp_size,
            TP_P2=triton.next_power_of_2(tp_size),
            CHUNKS=chunks,
            BLOCK=BLOCK,
            num_warps=8,
        )


@triton.jit
def _combine_stats_kernel(
    slots_ptr,
    row_stride,
    base,
    out_ptr,
    out_row_stride,
    out_col_stride,
    TP: tl.constexpr,
    TP_P2: tl.constexpr,
):
    row = tl.program_id(0)
    r = tl.arange(0, TP_P2)
    lane = tl.arange(0, 16)
    v = tl.load(slots_ptr + row * row_stride + base + r[:, None] * 16 + lane[None, :],
                mask=r[:, None] < TP, other=0.0)
    byte = v.to(tl.float32).to(tl.int64) << (24 - 8 * (lane % 4)).to(tl.int64)[None, :]
    word = (lane // 4)[None, :]
    m = tl.sum(tl.where(word == 0, byte, 0), axis=1).to(tl.int32).to(tl.float32, bitcast=True)
    s = tl.sum(tl.where(word == 1, byte, 0), axis=1).to(tl.int32).to(tl.float32, bitcast=True)
    t = tl.sum(tl.where(word == 2, byte, 0), axis=1).to(tl.int32).to(tl.float32, bitcast=True)
    live = (r < TP) & (s > 0.0)
    big = tl.max(tl.where(live, m, float("-inf")), axis=0)
    w = tl.where(live, tl.exp(m - big), 0.0)
    total = tl.sum(tl.where(live, s * w, 0.0), axis=0)
    moment = tl.sum(tl.where(live, t * w, 0.0), axis=0)
    top1 = tl.math.div_rn(1.0, total)
    ent = big + tl.log(total) - tl.math.div_rn(moment, total)
    tl.store(out_ptr + row * out_row_stride, top1)
    tl.store(out_ptr + row * out_row_stride + out_col_stride, ent)


def combine_stats(slots: torch.Tensor, base: int, tp_size: int, out: torch.Tensor) -> None:
    """Per row of summed ``slots``, the top-1 probability and entropy from the
    ranks' statistics at lane ``base``, into ``out[row, 0]`` and
    ``out[row, 1]`` (``out`` [rows, 2] FP32, any strides)."""
    rows = slots.shape[0]
    if rows:
        _combine_stats_kernel[(rows,)](
            slots,
            slots.stride(0),
            base,
            out,
            out.stride(0),
            out.stride(1),
            TP=tp_size,
            TP_P2=triton.next_power_of_2(tp_size),
            num_warps=1,
        )


def pick(slots: torch.Tensor, vocab: int, num_slots: int | None = None) -> torch.Tensor:
    """The token of each row of summed slots ([rows, >= slots * LANES] BF16):
    the index in the slot with the largest key, the lowest slot on ties,
    held in 0 .. vocab - 1 (see _pick_kernel). ``num_slots``: the chunk slots
    of a row (default: every lane of the row)."""
    rows = slots.shape[0]
    if num_slots is None:
        num_slots = slots.shape[1] // LANES
    out = torch.empty(rows, dtype=torch.int64, device=slots.device)
    if rows:
        _pick_kernel[(rows,)](
            slots,
            slots.stride(0),
            out,
            vocab,
            SLOTS=num_slots,
            SLOTS_P2=triton.next_power_of_2(num_slots),
            num_warps=1,
        )
    return out


def shard_tokens(
    local: torch.Tensor, rank: int, tp_size: int, vocab: int
) -> torch.Tensor:
    """torch.argmax of each gathered row, from this rank's shard ``local``
    [rows, width] of vocabulary-parallel logits (padding columns kept): the
    gathered row is the ranks' columns in rank order, cut at ``vocab``.
    Every rank calls it with the same rows; with none, no rank enters the
    all-reduce."""
    rows, width = local.shape
    if rows == 0:
        return torch.empty(0, dtype=torch.int64, device=local.device)
    col_start = rank * width
    n_cols = min(max(vocab - col_start, 0), width)
    cols = local[:, :n_cols]
    if cols.dtype != torch.float32:
        cols = cols.float()  # exact; keeps order, ties and NaN
    chunks = triton.cdiv(width, BLOCK)
    slots = chunk_slots(cols, col_start, rank, tp_size, chunks)
    if tp_size > 1:
        slots = tensor_model_parallel_all_reduce(slots)
    return pick(slots, vocab)


def refusal(speculator) -> str | None:
    """Why the per-rank argmax cannot replace this DSpark speculator's sampling."""
    model = speculator.model
    if not hasattr(model, "compute_logits_local"):
        return (
            f"{type(model).__name__} has no vocabulary-shard logits "
            "(compute_logits_local)"
        )
    if speculator.draft_logits is not None:
        return "probabilistic drafting samples the full draft logits"
    if speculator._draft_topk is not None:
        return "draft top-k (dspark_draft_topk) selects on the full draft logits"
    if speculator.acceptance_estimator is not None:
        return "the acceptance estimator reads the full draft logits"
    if model.draft_id_to_target_id is not None:
        return "a reduced draft vocabulary maps draft ids to target ids"
    if model.logits_processor.org_vocab_size > MAX_VOCAB:
        return f"a vocabulary above {MAX_VOCAB} tokens"
    return None


def _markov_sampler(speculator) -> bool:
    """Whether ``speculator`` samples its drafts with the DSpark Markov sampler
    that :func:`shard_argmax_sampling` decorates (also through subclasses that
    override and call it)."""
    return any(
        getattr(vars(cls).get("_sample_sequential"), "shard_argmax_sampler", False)
        for cls in type(speculator).__mro__
    )


def _kind(speculator) -> str:
    """How ``speculator`` drafts (see :func:`drafting`)."""
    if speculator is None:
        return "none"
    if _markov_sampler(speculator):
        return "markov"
    if hasattr(speculator, "candidate_sampler"):
        return "candidates"
    if not hasattr(speculator, "sample_draft"):
        return "headless"
    if speculator.draft_logits is not None:
        return "probabilistic"
    return "sample_draft"


def sample_draft_refusal(speculator) -> str | None:
    """Why the per-rank argmax cannot replace the greedy argmax of this
    speculator's ``sample_draft`` (:func:`adopt_sample_draft`)."""
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    name = type(speculator).__name__
    if not isinstance(speculator, DraftModelSpeculator):
        return f"{name} samples its drafts outside vLLM's draft-model speculator"
    cls = type(speculator)
    if (
        cls.sample_draft is not DraftModelSpeculator.sample_draft
        or cls.compute_draft_logits is not DraftModelSpeculator.compute_draft_logits
    ):
        return (
            f"{name} computes or samples its draft logits its own way "
            "(sample_draft, compute_draft_logits)"
        )
    if speculator.use_local_argmax_reduction:
        return (
            "speculative_config.use_local_argmax_reduction takes the drafts' "
            "per-rank argmax already: unset it with the switch"
        )
    if speculator.acceptance_estimator is not None:
        return "the acceptance estimator reads the full draft logits"
    model = speculator.model
    if not hasattr(model, "compute_logits_local"):
        return (
            f"{type(model).__name__} has no vocabulary-shard logits "
            "(compute_logits_local)"
        )
    if getattr(model, "draft_id_to_target_id", None) is not None:
        return "a reduced draft vocabulary maps draft ids to target ids"
    processor = getattr(model, "logits_processor", None)
    if not hasattr(getattr(model, "lm_head", None), "tp_rank") or processor is None:
        return f"{type(model).__name__} has no vocabulary-parallel LM head"
    if processor.org_vocab_size > MAX_VOCAB:
        return f"a vocabulary above {MAX_VOCAB} tokens"
    return None


def drafting(speculator) -> tuple[str, str | None]:
    """How the greedy drafts of ``speculator`` take their argmax with the
    switch on, and why the switch cannot serve them (None when it can).

    - The DSpark Markov sampler: the per-rank argmax of this module
      (:func:`sample_sequential`), refused as :func:`refusal` says.
    - A drafter that samples through vLLM's ``sample_draft`` (MTP, EAGLE,
      DFlash): the per-rank argmax of its draft model's
      ``compute_logits_local`` (:func:`adopt_sample_draft`), refused as
      :func:`sample_draft_refusal` says.
    - A candidate drafter (DFlash2, LiLiCorr): its selector picks among the
      head's top-k candidates, which ``compute_candidates`` already takes per
      rank (``LogitsProcessor.get_top_k_tokens``). No draft argmax over the
      vocabulary: nothing to serve.
    - Probabilistic drafting, no drafter, or one without a draft head
      (n-gram): no draft argmax, nothing to serve.
    """
    kind = _kind(speculator)
    name = type(speculator).__name__
    if kind == "none":
        return "no drafter", None
    if kind == "markov":
        return "the DSpark Markov sampler's per-rank argmax", refusal(speculator)
    if kind == "candidates":
        return (
            f"{name} selects among the head's per-rank top-k candidates, no "
            "draft argmax",
            None,
        )
    if kind == "headless":
        return f"{name} has no draft head", None
    if kind == "probabilistic":
        return f"{name} samples its drafts, no draft argmax", None
    return f"{name} samples per rank (sample_draft)", sample_draft_refusal(speculator)


def draft_tokens(model, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
    """torch.argmax of ``model.compute_logits(hidden_states, spec_step_idx)``
    without gathering the logits: the draft model's
    ``compute_logits_local(hidden_states, spec_step_idx)`` (this rank's
    vocabulary shard of the same logits, padding columns kept) through
    :func:`shard_tokens`."""
    local = model.compute_logits_local(hidden_states, spec_step_idx)
    head = model.lm_head
    return shard_tokens(
        local, head.tp_rank, head.tp_size, model.logits_processor.org_vocab_size
    )


def adopt_sample_draft(speculator) -> bool:
    """With the switch on, at load and before any draft graph is captured:
    a greedy drafter that samples through vLLM's ``sample_draft`` takes its
    drafts from :func:`draft_tokens` instead of the gathered logits' argmax
    (the same tokens, ties and NaN included). Refuses (ValueError) a drafter
    :func:`sample_draft_refusal` names; returns whether it adopted one (False
    for the other kinds of :func:`drafting`)."""
    if _kind(speculator) != "sample_draft":
        return False
    reason = sample_draft_refusal(speculator)
    if reason is not None:
        raise ValueError(f"{SWITCH}: {reason}")

    def sample_draft(
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
        spec_step_idx: int = 0,
        num_steps: int = 1,
    ) -> torch.Tensor:
        # Greedy: no draft logits are cached, no draw is seeded.
        return draft_tokens(speculator.model, hidden_states, spec_step_idx)

    speculator.sample_draft = sample_draft
    return True


def _enabled(speculator) -> bool:
    on = getattr(speculator, "_shard_argmax", None)
    if on is None:
        on = bool(getattr(envs, SWITCH))
        if on:
            reason = refusal(speculator)
            if reason is not None:
                raise ValueError(f"{SWITCH}: {reason}")
        speculator._shard_argmax = on
    return on


def sample_sequential(speculator, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """``DSparkSpeculator._sample_sequential`` for greedy drafts, with the
    argmax taken per rank (see the module docstring)."""
    s = speculator
    model = s.model
    n_spec = s.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[s.sample_indices[:num_sample]]
    local = model.compute_logits_local(sample_hidden)
    width = local.shape[-1]
    local = local.view(num_reqs, n_spec, width)
    # The gathered row is the ranks' columns in rank order, cut at the
    # vocabulary size.
    rank, tp_size = model.lm_head.tp_rank, model.lm_head.tp_size
    col_start = rank * width
    n_cols = min(max(model.logits_processor.org_vocab_size - col_start, 0), width)
    chunks = triton.cdiv(width, BLOCK)
    confidence_markov_embeds = []
    # The adaptive depth policy's draft-logit features (dspark_row_mask_conf).
    stats = getattr(s, "_row_stats", None)
    base = tp_size * chunks * LANES

    prev = s.input_buffers.input_ids[s._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        markov_embed = model.markov_embed(prev)
        if s.use_confidence_head:
            confidence_markov_embeds.append(markov_embed)
        bias = model.markov_bias(markov_embed)
        logits_i = local[:, i, :n_cols] + bias[:, col_start : col_start + n_cols]
        if logits_i.dtype != torch.float32:
            logits_i = logits_i.float()  # exact; keeps order, ties and NaN
        slots = chunk_slots(
            logits_i, col_start, rank, tp_size, chunks, stats=stats is not None
        )
        if tp_size > 1:
            slots = tensor_model_parallel_all_reduce(slots)
        draft_sampled_i = pick(slots, model.logits_processor.org_vocab_size, tp_size * chunks)
        if stats is not None:
            combine_stats(slots, base, tp_size, stats[:num_reqs, :, i])
        s.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i

    if s.use_confidence_head:
        confidence = model.compute_confidence(
            sample_hidden,
            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
        )
        s.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)


def shard_argmax_sampling(
    sample: Callable[..., None],
) -> Callable[..., None]:
    """Decorates ``DSparkSpeculator._sample_sequential``: with
    ``VLLM_SM80_SHARD_ARGMAX`` the greedy drafts come from
    :func:`sample_sequential`; the switch refuses a speculator it cannot
    serve (:func:`refusal`). With the hot pair ``shard_argmax``
    (``vllm_sm80.hot_swap``) both samplers are one pair site writing the
    same ``draft_tokens``. The mark on the wrapper lets :func:`drafting`
    find this sampler."""

    @functools.wraps(sample)
    def wrapper(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if hot_swap.listed("shard_argmax"):

            def gathered() -> torch.Tensor:
                sample(self, num_reqs, head_hidden)
                return self.draft_tokens

            def per_rank(out: torch.Tensor | None) -> torch.Tensor:
                sample_sequential(self, num_reqs, head_hidden)
                return self.draft_tokens

            hot_swap.pair("shard_argmax", gathered, per_rank)
            return None
        if not _enabled(self):
            return sample(self, num_reqs, head_hidden)
        sample_sequential(self, num_reqs, head_hidden)

    wrapper.shard_argmax_sampler = True
    return wrapper
