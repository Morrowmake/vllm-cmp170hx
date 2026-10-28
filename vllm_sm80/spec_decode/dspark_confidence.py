# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark confidence head outputs paired with the acceptance they predicted.

With ``VLLM_DSPARK_CONFIDENCE_DUMP`` set to a directory, the drafter computes
its confidence head in every draft step, as adaptive verification does, but
without enabling adaptive verification. Nothing it drafts or the target
verifies changes; the head adds one small FP32 projection per drafted position
to the captured draft step. Where the drafter samples each position from its
gathered draft logits (``_sample_logits``; not with
``VLLM_SM80_SHARD_ARGMAX``, which never gathers them), the draft
step also records, per drafted position, the top-1 probability and the entropy
(nats) of the logits it sampled from, before temperature: a log-sum-exp, a max
and a softmax-weighted sum over each position's ``[requests, vocab]`` logits.

Each ``propose`` call first stages, stream-ordered and without waiting:

* the confidences the previous draft step wrote (``draft_token_confidence_probs``,
  one row per request of the previous batch, one sigmoid output per drafted
  position: the head's probability that this draft is accepted given that the
  ones before it were);
* the draft-logit statistics the same draft step wrote, when it wrote them;
* this step's ``num_sampled`` and ``num_rejected``, the verification result of
  exactly those drafts.

TP rank 0 then appends one JSON line per request found in both batches to
``<directory>/dspark-confidence.<UTC start>-<pid>.rank<rank>.jsonl`` (a new
file per server start)::

    {"step": n, "req": id, "conf": [p_1, ...], "width": drafted positions,
     "verified": drafts the target verified, "accepted": accepted drafts,
     "batch": requests in the verifying step, "t": time.monotonic() when the
     verifying step's draft call began (CLOCK_MONOTONIC: comparable with a
     client's time.monotonic() on the same host, so rows can be assigned to
     benchmark windows),
     "top1": [q_1, ...], "ent": [H_1, ...] (only with the draft-logit
     statistics: top-1 probability and entropy of each drafted position)}

``verified`` and ``accepted`` come from ``num_sampled - 1 + num_rejected`` and
``num_sampled - 1``. Rows that verified nothing are skipped: prefill chunks,
and a step after a zero-draft step, which runs no draft step (the zero-draft
branch of ``DFlashSpeculator.propose``) and so leaves no confidences of its batch. Warm-up
requests (ids ``_warmup*``, ``_v2_mixed_warmup*``) are skipped. A row is
written once its copy event completes, so in steady state the host never
waits on the device; leftover rows are written at interpreter exit.
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Staging slots: a slot is reused only after its copies completed. With one
# slot per step, 32 covers far more steps than the host runs ahead.
NUM_SLOTS = 32

# Request ids of the runner's warm-up batches (vllm/v1/worker/gpu/warmup.py).
WARMUP_PREFIXES = ("_warmup", "_v2_mixed_warmup")


@dataclass
class _Slot:
    conf: torch.Tensor  # [max_num_reqs, max_width] float32
    stats: torch.Tensor  # [max_num_reqs, 2, max_width] float32: top-1, entropy
    num_sampled: torch.Tensor  # [max_num_reqs] int32
    num_rejected: torch.Tensor  # [max_num_reqs] int32


@dataclass
class _Pending:
    event: torch.cuda.Event | None
    slot: int
    step: int
    prev_req_ids: list[str]
    prev_width: int
    has_stats: bool
    req_ids: list[str]
    t: float


def _tp_rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except (AssertionError, RuntimeError):
        # No model-parallel group (single-process tests).
        return 0


class ConfidenceDump:
    """Stages confidences and verification results; writes paired rows."""

    def __init__(
        self,
        directory: str,
        *,
        max_num_reqs: int,
        max_width: int,
        device: torch.device,
        rank: int,
    ) -> None:
        self.rank = rank
        self.writes = rank == 0
        self.path = Path(directory) / (
            f"dspark-confidence.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
            f"-{os.getpid()}.rank{rank}.jsonl"
        )
        self._file = None
        if self.writes:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", buffering=1)  # noqa: SIM115
        pin = device.type == "cuda"
        self._slots = [
            _Slot(
                torch.empty(
                    max_num_reqs, max_width, dtype=torch.float32, pin_memory=pin
                ),
                torch.empty(
                    max_num_reqs, 2, max_width, dtype=torch.float32, pin_memory=pin
                ),
                torch.empty(max_num_reqs, dtype=torch.int32, pin_memory=pin),
                torch.empty(max_num_reqs, dtype=torch.int32, pin_memory=pin),
            )
            for _ in range(NUM_SLOTS if self.writes else 0)
        ]
        self._pending: deque[_Pending] = deque()
        self._next_slot = 0
        self._step = 0
        self._prev: tuple[list[str], int] | None = None
        self.rows_written = 0

    def stage(
        self,
        confidences: torch.Tensor,
        req_ids: list[str],
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        stats: torch.Tensor | None = None,
    ) -> None:
        """Stage the previous draft step's ``confidences`` (and its draft-logit
        ``stats``, [max_num_reqs, 2, max_width], when it wrote them) against
        this step's verification result (``num_sampled``/``num_rejected`` of
        ``req_ids``)."""
        now = time.monotonic()
        self._step += 1
        prev = self._prev
        if not self.writes or prev is None:
            return
        prev_req_ids, prev_width = prev
        slot_idx = self._next_slot
        self._next_slot = (slot_idx + 1) % len(self._slots)
        if self._pending and self._pending[0].slot == slot_idx:
            self._write(self._pending.popleft(), wait=True)
        slot = self._slots[slot_idx]
        n_prev, n = len(prev_req_ids), len(req_ids)
        # Whole rows of the contiguous (max_num_reqs, max_width) buffer: a
        # strided device-to-host copy stages through unpinned memory and waits
        # for the stream. _write reads only the drafted columns.
        slot.conf[:n_prev].copy_(confidences[:n_prev], non_blocking=True)
        if stats is not None:
            slot.stats[:n_prev].copy_(stats[:n_prev], non_blocking=True)
        slot.num_sampled[:n].copy_(num_sampled[:n], non_blocking=True)
        slot.num_rejected[:n].copy_(num_rejected[:n], non_blocking=True)
        event = None
        if confidences.is_cuda:
            event = torch.cuda.Event()
            event.record()
        self._pending.append(
            _Pending(
                event,
                slot_idx,
                self._step,
                prev_req_ids,
                prev_width,
                stats is not None,
                req_ids,
                now,
            )
        )
        self.drain(wait=False)

    def drafted(self, req_ids: list[str], width: int) -> None:
        """Record the batch whose confidences the draft step just wrote."""
        self._prev = (list(req_ids), width)

    def drafted_nothing(self) -> None:
        """The draft step did not run: the confidence buffer holds no row of
        this batch, so the next step pairs nothing."""
        self._prev = None

    def drain(self, wait: bool) -> None:
        while self._pending:
            head = self._pending[0]
            if not wait and head.event is not None and not head.event.query():
                return
            self._write(self._pending.popleft(), wait=True)

    def _write(self, p: _Pending, wait: bool) -> None:
        if wait and p.event is not None:
            p.event.synchronize()
        slot = self._slots[p.slot]
        prev_row = {rid: i for i, rid in enumerate(p.prev_req_ids)}
        assert self._file is not None
        for i, rid in enumerate(p.req_ids):
            j = prev_row.get(rid)
            if j is None or rid.startswith(WARMUP_PREFIXES):
                continue
            accepted = int(slot.num_sampled[i]) - 1
            verified = accepted + int(slot.num_rejected[i])
            if accepted < 0 or verified <= 0:
                continue
            row = {
                "step": p.step,
                "req": rid,
                "conf": [float(x) for x in slot.conf[j, : p.prev_width].tolist()],
                "width": p.prev_width,
                "verified": verified,
                "accepted": accepted,
                "batch": len(p.req_ids),
                "t": p.t,
            }
            if p.has_stats:
                w = p.prev_width
                row["top1"] = [float(x) for x in slot.stats[j, 0, :w].tolist()]
                row["ent"] = [float(x) for x in slot.stats[j, 1, :w].tolist()]
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.rows_written += 1

    def close(self) -> None:
        if self._file is None:
            return
        self.drain(wait=True)
        self._file.close()
        self._file = None


class ConfidenceDumpMixin:
    """Mixed in front of a DSpark speculator class by ``confidence_dump_class``."""

    _confidence_dump: ConfidenceDump | None = None
    # [max_num_reqs, 2, max_width]: per drafted position, the top-1 probability
    # and the entropy of the draft logits; written inside the draft step.
    _draft_stats: torch.Tensor | None = None
    # _sample_logits ran (set while tracing the draft step, so also on capture).
    _draft_stats_written = False

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        head = getattr(model.model, "confidence_head", None)
        if head is None:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_DUMP is set but the DSpark drafter has "
                "no confidence head (the checkpoint ships none, or it did not load)."
            )
        self.use_confidence_head = True
        weight = head.proj.weight
        directory = envs.VLLM_DSPARK_CONFIDENCE_DUMP
        self._confidence_dump = dump = ConfidenceDump(
            directory,
            max_num_reqs=self.max_num_reqs,
            max_width=self.max_width,
            device=self.device,
            rank=_tp_rank(),
        )
        self._draft_stats = torch.full(
            (self.max_num_reqs, 2, self.max_width),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        atexit.register(_close_dump, weakref.ref(dump))
        if dump.writes:
            logger.info(
                "DSpark confidence dump to %s: confidence head %s, weight "
                "L2 norm %.6g, finite %s",
                dump.path,
                tuple(weight.shape),
                float(weight.float().norm()),
                bool(torch.isfinite(weight).all()),
            )
        return model

    def _sample_logits(self, logits, idx_map, sample_pos, step):
        if self._draft_stats is not None:
            write_draft_stats(self._draft_stats, logits, step)
            self._draft_stats_written = True
        return super()._sample_logits(logits, idx_map, sample_pos, step)

    def propose(self, input_batch, *args, **kwargs):
        dump = self._confidence_dump
        if dump is None:
            return super().propose(input_batch, *args, **kwargs)
        bound = _propose_signature().bind(self, input_batch, *args, **kwargs)
        arguments = bound.arguments
        real = not arguments.get("dummy_run") and not arguments.get("is_profile")
        if real:
            dump.stage(
                _full_confidence(self),
                list(input_batch.req_ids),
                arguments["num_sampled"],
                arguments["num_rejected"],
                self._draft_stats if self._draft_stats_written else None,
            )
        result = super().propose(input_batch, *args, **kwargs)
        if real:
            if arguments.get("num_speculative_tokens") == 0:
                # Zero drafts: propose stored only the context K/V.
                dump.drafted_nothing()
            else:
                width = int(getattr(self, "width", self.max_width))
                dump.drafted(input_batch.req_ids, width)
        return result


def write_draft_stats(out: torch.Tensor, logits: torch.Tensor, step: int) -> None:
    """Write column ``step`` of ``out`` [max_num_reqs, 2, max_width]: per row of
    ``logits`` [requests, vocab], its top-1 probability and entropy (nats)."""
    x = logits.float()
    n = x.shape[0]
    lse = torch.logsumexp(x, dim=-1)
    out[:n, 0, step] = torch.exp(x.amax(dim=-1) - lse)
    p = torch.softmax(x, dim=-1)
    # xlogy: a masked (-inf) column has p == 0 and adds 0, not nan.
    out[:n, 1, step] = -torch.special.xlogy(p, p).sum(dim=-1)


def _full_confidence(speculator: Any) -> torch.Tensor:
    # The per-K-width drafter swaps in a narrower view while it samples and
    # restores the full (max_num_reqs, max_width) buffer afterwards.
    by_width = getattr(speculator, "_confidence_by_width", None)
    if by_width:
        return by_width[speculator.max_width]
    return speculator.draft_token_confidence_probs


@cache
def _propose_signature() -> inspect.Signature:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    return inspect.signature(DFlashSpeculator.propose)


def _close_dump(ref: weakref.ref) -> None:
    dump = ref()
    if dump is not None:
        dump.close()


@cache
def confidence_dump_class(base: type) -> type:
    """``base`` with the confidence head on and the per-round dump."""
    return type(f"ConfidenceDump{base.__name__}", (ConfidenceDumpMixin, base), {})
