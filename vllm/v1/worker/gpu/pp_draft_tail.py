# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the DFlash2 draft tail on an earlier pipeline stage.

Under pipeline parallelism the last stage runs the target's last layers, the
verify lm_head, the rejection sampler, the drafter's layers *and* the drafter's
tail: a second full-vocabulary lm_head pass over the sampled rows (top-k
candidates), the candidate selector and the selector walk. The tail needs no
KV cache, only ``lm_head`` and the selector weights, so it can run on another
stage that has memory to spare and idles while the last stage works
(``VLLM_PP_DRAFT_TAIL_STAGE=<stage>``, default -1 = off).

With the flag on:

* the last stage runs the drafter's layers as before, gathers the sampled rows'
  hidden states plus the per-row inputs of the walk (anchor token, sample
  position, temperature, seed) into one byte buffer (the *payload*), and sends
  it to the tail stage on a dedicated communicator;
* the tail stage runs the same candidate pass, selector and walk on the same
  rows with the same weights (copied from the last stage once, after start-up)
  and becomes the root of the draft-token broadcast for that step;
* the last stage receives the drafts like every other stage and scatters them
  into its request state before a later step reads them.

The draft tokens are bit-identical to the flag-off path: the kernels, their
inputs, the row count (the flag-off path's CUDA-graph padding included) and
the weights are the same. A step falls back to the last stage when the tail
cannot be moved (structured output needs the drafts on the host at once; a
step that samples nothing broadcasts nothing); every stage takes that decision
from the same scheduler output, so the collective order always agrees.

This module holds the pieces that do not need a GPU (gate, payload layout,
the per-step decision, the last stage's pending-draft queue) plus the tail
module a tail stage builds.
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

TAIL_STAGE_OFF = -1

# Elements per fp64 partial sum in weight_checksum: a 128 MiB fp64 transient.
CHECKSUM_CHUNK = 1 << 24


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftTailGate:
    """Start-up decision, taken from the config alone so every stage agrees.

    ``stage`` is the pipeline rank (within the PP group) that runs the tail,
    or -1. ``reason`` says why a requested stage was refused ("" otherwise).
    """

    requested: int
    stage: int
    reason: str

    @property
    def enabled(self) -> bool:
        return self.stage >= 0


def draft_tail_gate(vllm_config: VllmConfig, requested: int) -> DraftTailGate:
    """Whether the configuration supports moving the draft tail.

    Everything checked here is identical on every rank. Features the moved
    tail does not carry (probabilistic draft sampling, adaptive verification,
    acceptance-adaptive draft counts, tensor parallelism within a stage) keep
    the tail on the last stage.
    """

    def off(reason: str) -> DraftTailGate:
        return DraftTailGate(requested, TAIL_STAGE_OFF, reason)

    if requested < 0:
        return DraftTailGate(requested, TAIL_STAGE_OFF, "")
    pc = vllm_config.parallel_config
    pp = pc.pipeline_parallel_size
    if pp < 2:
        return off("needs pipeline parallelism")
    if requested > pp - 2:
        return off(f"stage must be an earlier stage than the last (0..{pp - 2})")
    if pc.tensor_parallel_size != 1:
        return off("needs one card per stage (tensor parallel size 1)")
    if pc.data_parallel_size != 1:
        return off("not supported with data parallelism")
    if getattr(pc, "prefill_context_parallel_size", 1) != 1:
        return off("not supported with prefill context parallelism")
    if not vllm_config.use_v2_model_runner:
        return off("needs the V2 model runner")
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "dflash":
        return off("needs a DFlash2 drafter")
    draft_model_config = spec.draft_model_config
    archs = getattr(draft_model_config, "architectures", None) or []
    if "DFlash2DraftModel" not in archs:
        return off("needs a DFlash2 drafter (candidate selector)")
    if "LiLiCorrDraftModel" in archs:
        return off("the LiLiCorr drafter scores candidates its own way")
    if spec.draft_sample_method != "greedy":
        return off("probabilistic draft sampling keeps the tail on the last stage")
    if spec.enable_adaptive_verification:
        return off("adaptive verification keeps the tail on the last stage")
    if spec.uses_adaptive_k() or spec.uses_dynamic_speculative_decoding():
        return off("variable draft counts keep the tail on the last stage")
    return DraftTailGate(requested, requested, "")


# --------------------------------------------------------------------------
# Per-step decision and roles
# --------------------------------------------------------------------------


def step_uses_remote_tail(
    live: bool,
    need_sampled_mask: np.ndarray | None,
    has_structured_output_reqs: bool,
) -> bool:
    """Whether this step's tail runs on the tail stage.

    Every input is known on every stage from the scheduler output, so every
    stage reaches the same answer (which decides the broadcast root).
    """
    return live and need_sampled_mask is not None and not has_structured_output_reqs


# Draft-broadcast roles for one step.
ROLE_LOCAL_ROOT = "local_root"  # last stage, tail local: broadcast drafts as root
ROLE_SEND_THEN_RECV = "send_then_recv"  # last stage, tail remote
ROLE_TAIL_ROOT = "tail_root"  # tail stage, tail remote: recv payload, root
ROLE_RECV = "recv"  # every other stage: receive drafts


def draft_role(pp_rank: int, pp_size: int, tail_stage: int, remote: bool) -> str:
    last = pp_size - 1
    if not remote:
        return ROLE_LOCAL_ROOT if pp_rank == last else ROLE_RECV
    if pp_rank == last:
        return ROLE_SEND_THEN_RECV
    if pp_rank == tail_stage:
        return ROLE_TAIL_ROOT
    return ROLE_RECV


def draft_broadcast_src(pp_size: int, tail_stage: int, remote: bool) -> int:
    """Pipeline rank (within the PP group) that roots this step's drafts."""
    return tail_stage if remote else pp_size - 1


def tail_rows(num_reqs: int, rows_table: dict[int, int] | None) -> int:
    """Rows the tail runs over: the drafter's (CUDA-graph padded) row count.

    ``rows_table`` maps the batch's request count to the row count the last
    stage's drafter dispatch uses; it is sent once by the last stage.
    """
    if rows_table:
        return rows_table.get(num_reqs, num_reqs)
    return num_reqs


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------

_ALIGN = 16


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


@dataclass
class TailPayloadViews:
    hidden: torch.Tensor  # [rows * k, H] model dtype
    anchor: torch.Tensor  # [rows] input-id dtype
    sample_pos: torch.Tensor  # [rows * k] int64
    row_state: torch.Tensor  # [rows * k] int32; row index, -1 = padding
    temperature: torch.Tensor  # [rows] float32
    seeds: torch.Tensor  # [rows] int64


class TailPayloadLayout:
    """One contiguous byte buffer per step, laid out from the row count, so
    both ends compute the same offsets and sizes without exchanging them."""

    def __init__(
        self,
        max_rows: int,
        num_steps: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        anchor_dtype: torch.dtype,
    ) -> None:
        self.max_rows = max_rows
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.hidden_dtype = hidden_dtype
        self.anchor_dtype = anchor_dtype

    def _regions(self, rows: int) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
        n = rows * self.num_steps
        return [
            ("hidden", self.hidden_dtype, (n, self.hidden_size)),
            ("sample_pos", torch.int64, (n,)),
            ("seeds", torch.int64, (rows,)),
            ("anchor", self.anchor_dtype, (rows,)),
            ("row_state", torch.int32, (n,)),
            ("temperature", torch.float32, (rows,)),
        ]

    def _offsets(
        self, rows: int
    ) -> tuple[list[tuple[str, torch.dtype, tuple, int]], int]:
        out = []
        off = 0
        for name, dtype, shape in self._regions(rows):
            nbytes = int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
            out.append((name, dtype, shape, off))
            off = _align(off + nbytes)
        return out, off

    def nbytes(self, rows: int) -> int:
        return self._offsets(rows)[1]

    def views(self, buf: torch.Tensor, rows: int) -> TailPayloadViews:
        assert buf.dtype == torch.uint8 and buf.dim() == 1
        regions, total = self._offsets(rows)
        assert buf.numel() >= total, (buf.numel(), total)
        parts: dict[str, torch.Tensor] = {}
        for name, dtype, shape, off in regions:
            esize = torch.tensor([], dtype=dtype).element_size()
            n = int(np.prod(shape))
            parts[name] = buf[off : off + n * esize].view(dtype).view(shape)
        return TailPayloadViews(**parts)


def pack_tail_payload(
    views: TailPayloadViews,
    rows: int,
    num_steps: int,
    last_hidden_states: torch.Tensor,
    sample_indices: torch.Tensor,
    input_ids: torch.Tensor,
    anchor_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    row_ids: torch.Tensor,
) -> None:
    """Gather what the tail reads, row for row as the flag-off path reads it.

    Static-shaped for a given ``rows`` and allocation-light, so it can sit in
    the drafter's captured graph. ``temperature``/``seeds`` are indexed by
    request slot on the last stage; the payload carries them per row, and
    ``row_state`` maps each sample row to its row (or -1 for padding), which
    is what the walk indexes them by on the tail stage.
    """
    n = rows * num_steps
    torch.index_select(last_hidden_states, 0, sample_indices[:n], out=views.hidden)
    torch.index_select(input_ids, 0, anchor_indices[:rows], out=views.anchor)
    views.sample_pos.copy_(sample_pos[:n])
    idx = sample_idx_mapping[:n]
    views.row_state.copy_(
        torch.where(idx >= 0, row_ids[:n], torch.full_like(row_ids[:n], -1))
    )
    slots = sample_idx_mapping[0:n:num_steps].clamp(min=0).to(torch.int64)
    torch.index_select(temperature, 0, slots, out=views.temperature)
    torch.index_select(seeds, 0, slots, out=views.seeds)


def copy_payload_rows(
    src: TailPayloadViews, dst: TailPayloadViews, rows: int, num_steps: int
) -> None:
    """Copy the first ``rows`` rows between payloads of different row counts
    (only used if the last stage's row count ever differs from the table)."""
    n = rows * num_steps
    dst.hidden[:n].copy_(src.hidden[:n])
    dst.sample_pos[:n].copy_(src.sample_pos[:n])
    dst.row_state[:n].copy_(src.row_state[:n])
    dst.anchor[:rows].copy_(src.anchor[:rows])
    dst.temperature[:rows].copy_(src.temperature[:rows])
    dst.seeds[:rows].copy_(src.seeds[:rows])


# --------------------------------------------------------------------------
# The tail computation
# --------------------------------------------------------------------------


def run_draft_tail(
    compute_candidates: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    select: Callable[..., torch.Tensor],
    walk: Callable[..., None],
    views: TailPayloadViews,
    rows: int,
    num_steps: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate pass, selector and walk over ``rows`` payload rows.

    Mirrors ``DFlash2Speculator._generate_draft`` after its drafter forward:
    same inputs, same shapes, same order. ``walk`` writes the draft tokens.
    """
    hidden = views.hidden.view(rows, num_steps, -1)
    candidate_ids, unary_logits = compute_candidates(hidden.flatten(0, 1))
    candidate_ids = candidate_ids.view(rows, num_steps, top_k)
    unary_logits = unary_logits.view_as(candidate_ids)
    scores = select(candidate_ids, unary_logits, hidden, views.anchor)
    walk(candidate_ids, scores, views, rows)
    return candidate_ids, scores


# --------------------------------------------------------------------------
# Last stage: drafts that arrive from the tail stage
# --------------------------------------------------------------------------


@dataclass
class RemoteDrafts:
    step: int
    event: Any
    draft_tokens: torch.Tensor  # [num_reqs, k], batch order
    idx_mapping_np: np.ndarray  # [num_reqs]
    keep_np: np.ndarray  # [num_reqs] rows that produce a sample
    gen_np: np.ndarray  # [num_reqs] slot generation at send time
    # VLLM_PP_DRAFT_TAIL_VERIFY: the drafts the last stage computed itself.
    check: torch.Tensor | None = None


class RemoteDraftQueue:
    """Drafts broadcast by the tail stage, not yet in the last stage's state.

    The last stage no longer writes its own drafts; they land later on the
    broadcast stream. Before a step reads draft tokens it applies every entry
    that holds a row of the batch (and every entry before it, in order, so a
    row always ends with its newest drafts), plus any entry older than
    ``max_age`` steps. Entries for other rows stay queued, so the stage never
    waits on a tail it does not need yet. Rows whose request slot was freed
    since the send are skipped.
    """

    REPORT_EVERY = 500

    def __init__(self, max_age: int) -> None:
        self.max_age = max_age
        self.entries: deque[RemoteDrafts] = deque()
        self.step = 0
        # VLLM_PP_DRAFT_TAIL_VERIFY counters (device tensors, read rarely).
        self.checked_rows: torch.Tensor | None = None
        self.differing_rows: torch.Tensor | None = None

    def _verify(self, e: RemoteDrafts, keep: np.ndarray) -> None:
        assert e.check is not None
        keep_t = torch.as_tensor(keep, device=e.draft_tokens.device)
        diff = (e.draft_tokens != e.check).any(dim=1) & keep_t
        if self.checked_rows is None:
            self.checked_rows = torch.zeros((), dtype=torch.int64,
                                            device=e.draft_tokens.device)
            self.differing_rows = torch.zeros_like(self.checked_rows)
        self.checked_rows += keep_t.sum()
        self.differing_rows += diff.sum()

    def verify_report(self) -> tuple[int, int] | None:
        """(rows checked, rows whose drafts differed); a host sync."""
        if self.checked_rows is None:
            return None
        return int(self.checked_rows), int(self.differing_rows)

    def __len__(self) -> int:
        return len(self.entries)

    def push(self, entry: RemoteDrafts) -> None:
        self.entries.append(entry)

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def _live_rows(self, e: RemoteDrafts, gen_now: np.ndarray) -> np.ndarray:
        return e.keep_np & (gen_now[e.idx_mapping_np] == e.gen_np)

    def due(
        self, batch_idx_np: np.ndarray, gen_now: np.ndarray
    ) -> list[RemoteDrafts]:
        last_due = -1
        batch = set(int(i) for i in batch_idx_np)
        for i, e in enumerate(self.entries):
            if e.step <= self.step - self.max_age:
                last_due = i
                continue
            rows = e.idx_mapping_np[self._live_rows(e, gen_now)]
            if any(int(r) in batch for r in rows):
                last_due = i
        out = []
        for _ in range(last_due + 1):
            out.append(self.entries.popleft())
        return out

    def apply(
        self,
        batch_idx_np: np.ndarray,
        gen_now: np.ndarray,
        draft_tokens_to_update: torch.Tensor,
        wait_event: Callable[[Any], None],
        to_device: Callable[[np.ndarray], torch.Tensor],
    ) -> int:
        """Apply the due entries; returns how many were applied."""
        applied = 0
        for e in self.due(batch_idx_np, gen_now):
            keep = self._live_rows(e, gen_now)
            if not keep.any():
                continue
            wait_event(e.event)
            if e.check is not None:
                self._verify(e, keep)
            draft_tokens = e.draft_tokens
            if keep.all():
                idx_np = e.idx_mapping_np
            else:
                keep_t = torch.as_tensor(keep, device=draft_tokens.device)
                draft_tokens = draft_tokens[keep_t]
                idx_np = e.idx_mapping_np[keep]
            draft_tokens_to_update[to_device(idx_np)] = draft_tokens
            applied += 1
        return applied


# --------------------------------------------------------------------------
# Private workspaces for a tail that runs beside the stage's own forward
# --------------------------------------------------------------------------


@contextlib.contextmanager
def private_flashinfer_topk_workspace(
    device: torch.device, holder: dict
) -> Iterator[None]:
    """Give FlashInfer's radix top-k a private row-state buffer.

    The radix top-k keeps one cached, zero-initialised row-state buffer per
    device and relies on stream order to reuse it. The tail stage runs its
    top-k on a side stream while its own forward (whose indexer top-k uses
    the same cache) runs on the main stream, so the tail swaps in its own
    buffer for the duration of the call. Host-side only: the kernel is
    launched with whichever pointer the cache holds at launch time.
    """
    try:
        import flashinfer.utils as fi_utils
    except Exception:
        yield
        return
    cache = getattr(fi_utils, "_cache_buf", None)
    if not isinstance(cache, dict):
        yield
        return
    key = (f"radix_topk_row_states_{device}", device)
    missing = object()
    prev = cache.get(key, missing)
    mine = holder.get("buf")
    if mine is None:
        mine = torch.zeros(1024 * 1024, dtype=torch.uint8, device=device)
    cache[key] = mine
    try:
        yield
    finally:
        # FlashInfer may have grown the buffer: keep whatever it now holds.
        holder["buf"] = cache.get(key, mine)
        if prev is missing:
            cache.pop(key, None)
        else:
            cache[key] = prev


def flashinfer_topk_isolatable() -> bool:
    """True when the candidate top-k either does not use FlashInfer or uses a
    FlashInfer whose row-state cache the tail can swap (see above)."""
    from vllm.model_executor.layers.logits_processor import _flashinfer_topk

    if _flashinfer_topk() is None:
        return True
    try:
        import flashinfer.utils as fi_utils
    except Exception:
        return False
    return isinstance(getattr(fi_utils, "_cache_buf", None), dict)


@contextlib.contextmanager
def tail_side_stream_workspaces(device: torch.device, holder: dict) -> Iterator[None]:
    """Everything the tail needs to run concurrently with the stage's forward:
    a private thin-GEMM workspace lane and a private FlashInfer top-k buffer."""
    from vllm import envs

    with contextlib.ExitStack() as stack:
        if envs.VLLM_GLM5_THIN_GEMM:
            from vllm.ampere_thin_gemm.thin_gemm import workspace_lane

            stack.enter_context(workspace_lane(DRAFT_TAIL_THIN_GEMM_LANE))
        stack.enter_context(private_flashinfer_topk_workspace(device, holder))
        yield


DRAFT_TAIL_THIN_GEMM_LANE = 1


# --------------------------------------------------------------------------
# Tail stage: its own copy of lm_head and the selector
# --------------------------------------------------------------------------


def _find_lm_head_prefix(model: torch.nn.Module) -> str:
    for name, _ in model.named_modules():
        if name == "lm_head" or name.endswith(".lm_head"):
            return name
    return "lm_head"


class DraftTailModule(torch.nn.Module):
    """lm_head, the candidate logits processor and the candidate selector,
    built the way the last stage builds them (same classes, same arguments,
    same compile tag), with weights copied from the last stage after start-up.
    """

    def __init__(self, vllm_config: VllmConfig, target_model: torch.nn.Module) -> None:
        super().__init__()
        from vllm.compilation.backends import set_model_tag
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )
        from vllm.model_executor.models.qwen3_dflash2 import CandidateSelector
        from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
            dflash_draft_vllm_config,
        )

        spec = vllm_config.speculative_config
        assert spec is not None
        draft_hf = spec.draft_model_config.hf_config
        draft_cfg = draft_hf.dflash_config
        target_hf = vllm_config.model_config.hf_text_config
        draft_vllm_config = dflash_draft_vllm_config(vllm_config)
        self.lm_head_prefix = _find_lm_head_prefix(target_model)
        with set_current_vllm_config(vllm_config):
            self.lm_head = ParallelLMHead(
                target_hf.vocab_size,
                target_hf.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=self.lm_head_prefix,
            )
        with set_current_vllm_config(draft_vllm_config):
            softcap = float(draft_cfg.get("final_logit_softcapping") or 0.0)
            self.candidate_logits_processor = LogitsProcessor(
                vllm_config.model_config.get_vocab_size(),
                scale=float(draft_cfg.get("output_multiplier", 1.0)),
                soft_cap=softcap if softcap > 0 else None,
            )
            with set_model_tag("dflash2_candidate_selector"):
                self.candidate_selector = CandidateSelector(
                    hidden_size=draft_hf.hidden_size,
                    vocab_size=draft_hf.vocab_size,
                    rank=int(draft_cfg["selector_rank"]),
                    top_k=int(draft_cfg["selector_top_k"]),
                    params_dtype=vllm_config.model_config.dtype,
                    prefix="model.candidate_selector",
                )
        for module in (self.lm_head, self.candidate_selector.hidden_projection):
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                quant_method.process_weights_after_loading(module)

    @property
    def top_k(self) -> int:
        return self.candidate_selector.top_k

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.candidate_selector.top_k
        )

    def select(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        return self.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )


def tail_tensors(lm_head: torch.nn.Module, selector: torch.nn.Module) -> dict:
    """Name -> tensor for everything the tail reads, in a fixed order."""
    out: dict[str, torch.Tensor] = {}
    for prefix, module in (("lm_head", lm_head), ("candidate_selector", selector)):
        for name, t in sorted(module.named_parameters()):
            out[f"{prefix}.{name}"] = t
        for name, t in sorted(module.named_buffers()):
            out[f"{prefix}.{name}"] = t
    return out


def weight_checksum(t: torch.Tensor, chunk: int | None = None) -> float:
    """fp64 sum of ``t``, taken over fixed slices along dim 0 so the fp64
    transient is at most ``chunk`` elements (8 bytes each) whatever the
    tensor's size or layout. The copy check runs after the KV pool is sized,
    where a whole-tensor ``.double()`` of the full-vocabulary lm_head
    (154,880 x 4,096) would need 4.73 GiB. Both stages sum in the same order,
    so equal bytes give an equal result."""
    chunk = chunk or CHECKSUM_CHUNK
    t = t.detach()
    if t.dim() == 0:
        return float(t.double())
    if t.numel() == 0:
        return 0.0
    row = max(1, t[0].numel())
    total = 0.0
    for part in t.split(max(1, chunk // row), dim=0):
        if part.numel() <= chunk:
            total += float(part.double().sum())
        else:  # one dim-0 row alone exceeds the chunk
            for r in part:
                total += weight_checksum(r, chunk)
    return total


def tensor_signature(tensors: dict) -> list[tuple]:
    return [
        (name, tuple(t.shape), str(t.dtype), tuple(t.stride()))
        for name, t in tensors.items()
    ]


def module_signature(lm_head: torch.nn.Module, logits_processor, selector) -> dict:
    """What must match between the two copies besides the weights."""
    si = lm_head.shard_indices
    return {
        "lm_head_quant": type(lm_head.quant_method).__name__,
        "lm_head_shard": (
            si.org_vocab_start_index,
            si.org_vocab_end_index,
            si.num_org_vocab_padding,
        ),
        "lm_head_tp": lm_head.tp_size,
        "logits_scale": float(logits_processor.scale),
        "logits_soft_cap": logits_processor.soft_cap,
        "logits_head_dtype": str(getattr(logits_processor, "head_dtype", None)),
        "top_k": int(selector.top_k),
        "selector_shard": bool(getattr(selector, "shard_vocab", False)),
    }


# --------------------------------------------------------------------------
# Controller: what the model runner calls
# --------------------------------------------------------------------------


class DraftTailController:
    """Owns the moved draft tail for one pipeline stage.

    Lifecycle: ``create`` at runner init (config gate, identical on every
    stage) -> ``attach_speculator`` (last stage: split the drafter's graphs)
    -> ``load`` (tail stage: allocate the lm_head + selector copy, counted in
    the model's memory) -> ``finalize`` after CUDA-graph capture (every stage:
    agree, copy the weights, warm the tail up, go live). Until ``finalize``
    succeeds every step keeps the tail on the last stage.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 gate: DraftTailGate, pp_rank: int, pp_size: int) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.gate = gate
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_last = pp_rank == pp_size - 1
        self.is_tail = gate.enabled and pp_rank == gate.stage
        self.live = False
        self.rows_table: dict[int, int] | None = None
        self.layout: TailPayloadLayout | None = None
        self.module: DraftTailModule | None = None
        self.speculator = None
        self.handler = None  # the stage's PPHandler (communicator, stream)
        self._workspaces: dict = {}
        self._out_tokens: torch.Tensor | None = None
        self._scores: torch.Tensor | None = None
        self._warned_rows = False

    @classmethod
    def create(cls, vllm_config: VllmConfig, device: torch.device,
               pp_rank: int, pp_size: int) -> DraftTailController | None:
        from vllm import envs

        gate = draft_tail_gate(vllm_config, envs.VLLM_PP_DRAFT_TAIL_STAGE)
        if not gate.enabled:
            if gate.requested >= 0:
                logger.warning_once(
                    "VLLM_PP_DRAFT_TAIL_STAGE=%d set but off: %s; the draft "
                    "tail stays on the last stage.", gate.requested, gate.reason
                )
            return None
        return cls(vllm_config, device, gate, pp_rank, pp_size)

    # ---- set-up -----------------------------------------------------------

    def attach_handler(self, handler) -> None:
        self.handler = handler

    def attach_speculator(self, speculator) -> None:
        """Last stage: capture the drafter without its tail from now on."""
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
            DFlash2Speculator,
        )

        if self.is_last and isinstance(speculator, DFlash2Speculator):
            speculator.enable_split_tail()
            self.speculator = speculator
            self.layout = speculator.tail_layout

    def load(self, target_model: torch.nn.Module, dtype: torch.dtype) -> None:
        """Tail stage: allocate its copy of lm_head and the selector."""
        if not self.is_tail:
            return
        from vllm.utils.torch_utils import set_default_torch_dtype

        with set_default_torch_dtype(dtype), torch.device(self.device):
            self.module = DraftTailModule(self.vllm_config, target_model)
        spec = self.vllm_config.speculative_config
        k = spec.num_speculative_tokens
        max_rows = self.vllm_config.scheduler_config.max_num_seqs
        self.layout = TailPayloadLayout(
            max_rows=max_rows,
            num_steps=k,
            hidden_size=spec.draft_model_config.get_hidden_size(),
            hidden_dtype=dtype,
            anchor_dtype=torch.int32,
        )
        self._out_tokens = torch.zeros(max_rows, k, dtype=torch.int64,
                                       device=self.device)
        self._scores = torch.zeros(max_rows * k * self.module.top_k,
                                   dtype=torch.float32, device=self.device)

    # ---- per step ---------------------------------------------------------

    def remote_step(self, input_batch) -> bool:
        from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

        return step_uses_remote_tail(
            self.live,
            compute_need_sampled_mask(input_batch),
            input_batch.has_structured_output_reqs,
        )

    def step(self, input_batch):
        """Non-last stages: this step's DraftTailStep, or None (tail local)."""
        from vllm.v1.worker.gpu.pp_utils import DraftTailStep

        if not self.remote_step(input_batch):
            return None
        assert self.layout is not None
        num_reqs = input_batch.num_reqs
        rows = tail_rows(num_reqs, self.rows_table)
        compute = None
        if self.is_tail:
            def compute(payload: torch.Tensor, draft_tokens: torch.Tensor) -> None:
                self._run(payload, rows, num_reqs, draft_tokens)
        return DraftTailStep(self.layout.nbytes(rows), compute)

    def check_drafts(self, num_reqs: int) -> torch.Tensor | None:
        """VLLM_PP_DRAFT_TAIL_VERIFY: the tail run here too, for comparison."""
        from vllm import envs

        if not envs.VLLM_PP_DRAFT_TAIL_VERIFY or self.speculator is None:
            return None
        self.speculator.run_tail_local(self.speculator.tail_rows)
        return self.speculator.draft_tokens[:num_reqs].clone()

    def payload(self, num_reqs: int) -> torch.Tensor:
        """Last stage, after a split drafter forward: the bytes to send."""
        spec = self.speculator
        assert spec is not None and self.layout is not None
        rows = spec.tail_rows
        expected = tail_rows(num_reqs, self.rows_table)
        if rows == expected:
            return spec.tail_payload(rows).clone()
        if not self._warned_rows:
            self._warned_rows = True
            logger.warning(
                "PP draft tail: the drafter ran %d rows for %d requests where "
                "the tail stage expects %d; sending %d rows (drafts stay valid "
                "but may differ from a last-stage tail).",
                rows, num_reqs, expected, expected,
            )
        out = torch.zeros(self.layout.nbytes(expected), dtype=torch.uint8,
                          device=self.device)
        dst = self.layout.views(out, expected)
        dst.row_state.fill_(-1)
        copy_payload_rows(spec.tail_staging_views(rows), dst,
                          min(rows, expected), self.layout.num_steps)
        return out

    def _walk(self):
        from vllm.triton_utils import triton
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
            _selector_walk_kernel,
        )

        assert self.module is not None
        top_k = self.module.top_k
        k = self.layout.num_steps
        block_k = triton.next_power_of_2(top_k)
        use_fp64 = self.vllm_config.model_config.use_fp64_gumbel
        out_tokens, scores_out = self._out_tokens, self._scores

        def walk(candidate_ids, scores, views: TailPayloadViews, rows: int) -> None:
            _selector_walk_kernel[(rows,)](
                scores.contiguous(),
                candidate_ids.contiguous(),
                views.sample_pos,
                views.row_state,
                views.temperature,
                views.seeds,
                out_tokens,
                scores_out,
                num_steps=k,
                top_k=top_k,
                BLOCK_K=block_k,
                SAMPLE_PROBABILISTIC=False,
                USE_FP64=use_fp64,
                num_warps=1,
            )

        return walk

    def _run(self, payload: torch.Tensor, rows: int, num_reqs: int,
             draft_tokens: torch.Tensor | None) -> None:
        """Tail stage, on the broadcast stream (beside its own forward)."""
        assert self.module is not None and self.layout is not None
        views = self.layout.views(payload, rows)
        with tail_side_stream_workspaces(self.device, self._workspaces):
            run_draft_tail(
                self.module.compute_candidates,
                self.module.select,
                self._walk(),
                views,
                rows,
                self.layout.num_steps,
                self.module.top_k,
            )
        if draft_tokens is not None:
            draft_tokens.copy_(self._out_tokens[:num_reqs])

    # ---- after capture ----------------------------------------------------

    def _local_info(self) -> dict:
        info: dict = {"ok": False, "reason": ""}
        if self.is_last:
            spec = self.speculator
            if spec is None or not spec.split_tail:
                info["reason"] = "the last stage's drafter is not DFlash2"
                return info
            lm_head = spec.model.lm_head
            selector = spec.model.model.candidate_selector
            info.update(
                ok=True,
                sig=tensor_signature(tail_tensors(lm_head, selector)),
                mod=module_signature(
                    lm_head, spec.model.candidate_logits_processor, selector
                ),
                rows_table=spec.graph_rows_table(),
                layout=(spec.tail_layout.max_rows, spec.tail_layout.num_steps,
                        spec.tail_layout.hidden_size,
                        str(spec.tail_layout.hidden_dtype),
                        str(spec.tail_layout.anchor_dtype)),
                probabilistic=spec.draft_logits is not None,
                adaptive=bool(spec.enable_adaptive_verification),
                fp64=bool(spec.use_fp64_gumbel),
            )
        elif self.is_tail:
            m = self.module
            if m is None:
                info["reason"] = "the tail stage has no lm_head copy"
                return info
            if not flashinfer_topk_isolatable():
                info["reason"] = "cannot give the candidate top-k a private buffer"
                return info
            lay = self.layout
            info.update(
                ok=True,
                sig=tensor_signature(tail_tensors(m.lm_head, m.candidate_selector)),
                mod=module_signature(m.lm_head, m.candidate_logits_processor,
                                     m.candidate_selector),
                layout=(lay.max_rows, lay.num_steps, lay.hidden_size,
                        str(lay.hidden_dtype), str(lay.anchor_dtype)),
                fp64=bool(self.vllm_config.model_config.use_fp64_gumbel),
            )
        return info

    @staticmethod
    def agree(last: dict, tail: dict) -> tuple[bool, str]:
        if not last.get("ok"):
            return False, last.get("reason") or "last stage not ready"
        if not tail.get("ok"):
            return False, tail.get("reason") or "tail stage not ready"
        if last.get("probabilistic") or last.get("adaptive"):
            return False, "the drafter samples probabilistically or adapts"
        for key in ("sig", "mod", "layout", "fp64"):
            if last.get(key) != tail.get(key):
                return False, (
                    f"the two copies differ ({key}: {last.get(key)} vs "
                    f"{tail.get(key)})"
                )
        return True, ""

    def finalize(self) -> None:
        """Every stage, once, after CUDA-graph capture."""
        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
        last_idx = self.pp_size - 1
        tail_idx = self.gate.stage
        mine = self._local_info()
        info_last = pp.broadcast_object(mine if self.is_last else None, src=last_idx)
        info_tail = pp.broadcast_object(mine if self.is_tail else None, src=tail_idx)
        ok, reason = self.agree(info_last, info_tail)
        if ok and (self.is_last or self.is_tail):
            ok, reason = self._copy_weights(pp, tail_idx)
        if ok and self.is_tail:
            self.rows_table = info_last["rows_table"]
            self._warm_up()
        # Everyone learns whether the copy (checked on the tail stage) held.
        verdict = pp.broadcast_object((ok, reason) if self.is_tail else None,
                                      src=tail_idx)
        ok, reason = verdict
        if ok:
            self.rows_table = info_last["rows_table"]
            self.live = True
            logger.info_once(
                "PP draft tail on: stage %d runs the DFlash2 candidate lm_head "
                "pass, top-k, selector and walk and roots the draft broadcast "
                "(VLLM_PP_DRAFT_TAIL_STAGE=%d; rows per batch size %s).",
                tail_idx, tail_idx, str(self.rows_table),  # *_once args must hash
            )
        else:
            if self.is_tail and self.module is not None:
                self.module = None
                torch.cuda.empty_cache()
            logger.warning_once(
                "VLLM_PP_DRAFT_TAIL_STAGE=%d set but off: %s; the draft tail "
                "stays on the last stage.", tail_idx, reason,
            )

    def _copy_weights(self, pp, tail_idx: int) -> tuple[bool, str]:
        group = getattr(self.handler, "tail_group", None)
        if group is None:
            return False, "no draft-tail communicator"
        if self.is_last:
            spec = self.speculator
            tensors = tail_tensors(
                spec.model.lm_head, spec.model.model.candidate_selector
            )
            for t in tensors.values():
                torch.distributed.send(t.detach().contiguous(),
                                       dst=pp.ranks[tail_idx], group=group)
            sums = [weight_checksum(t) for t in tensors.values()]
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            pp.send_object(sums, dst=tail_idx)
            return True, ""
        m = self.module
        tensors = tail_tensors(m.lm_head, m.candidate_selector)
        for t in tensors.values():
            assert t.is_contiguous()
            torch.distributed.recv(t.data, src=pp.ranks[-1], group=group)
        torch.cuda.synchronize(self.device)
        sums = [weight_checksum(t) for t in tensors.values()]
        torch.cuda.empty_cache()
        expect = pp.recv_object(src=self.pp_size - 1)
        if sums != expect:
            return False, "the lm_head/selector copy does not match the last stage"
        return True, ""

    def _warm_up(self) -> None:
        """Compile the selector and allocate the private workspaces for every
        row count, on the broadcast stream the tail runs on."""
        assert self.layout is not None and self.handler is not None
        stream = self.handler.broadcast_stream
        rows_seen = sorted(set(self.rows_table.values()) |
                           set(range(1, self.layout.max_rows + 1)))
        with torch.cuda.stream(stream):
            for rows in rows_seen:
                buf = torch.zeros(self.layout.nbytes(rows), dtype=torch.uint8,
                                  device=self.device)
                self.layout.views(buf, rows).row_state.fill_(-1)
                self._run(buf, rows, 0, None)
        torch.cuda.synchronize(self.device)
