# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark adaptive depth, host side: the step's verified width K from each
request's ``p_i`` of an earlier round.

A ``VLLM_DSPARK_ADAPTIVE_DEPTH`` policy with a ``host_width`` block (estimate
``stale``) makes a decode step of b requests, b in the block's
``batch_sizes``, verify the K in ``k_choices`` maximising::

    sum_r E(q_r, d_r(K)) / T(b, K | d)
    d_r(K)     = the largest k <= K with prod_{j <= k} q_rj >= tau[b]
    E(q, d)    = 1 + sum_{i <= d} prod_{j <= i} q_j
    T(b, K | d) = t0_ms[b] + b cum_ms[b][K]
                  - (1 - s[b][K]) sum_r (cum_ms[b][K] - cum_ms[b][d_r])

(ties to the smaller K; ``tau`` is the policy's), where q_r is request r's
``p_i`` (positions 1 .. max K) of its verify round ``lag_rounds`` back. A step
takes ``fallback_k`` when any request lacks that round or its ``p_i`` (its
first rounds, placeholder drafts, or a round without the decision kernel:
prefill rows or mixed draft counts). Inside the step the decision kernel
picks each request's live depth as at any b. Steps with a prefill row and
batch sizes outside the block keep the width the load (and
``VLLM_DSPARK_DEPTH_HINT``) gave; the host width chooses among the K at most
that width and the drafts on hand.

Why a round back: under async scheduling the scheduler fixes step n's shape
while step n - 1 runs, with the outputs of steps up to n - 2 processed, so
the ``p_i`` of a request's round two back are on the host without a sync.

No upstream file changes:

* worker: the decision kernel also writes each request's ``p_i`` for
  positions 1 .. max K into a fresh tensor (``RowMaskState.p_hat_step``);
  :func:`install_output_copy` wraps the GPU runner's ``AsyncOutput`` so its
  copy stream copies it to the host before the output's copy event, and
  ``get_output`` sets the rows (``req_ids`` order) on the
  ``ModelRunnerOutput`` as :data:`P_HAT_ATTR`;
* scheduler (:func:`install`, at plugin registration): wraps
  ``Scheduler.schedule`` (each request's verify round: its scheduled drafts,
  placeholder drafts excepted), ``Scheduler.update_from_output`` (the round's
  ``p_i``), ``Scheduler._select_adaptive_k`` (the width) and
  ``AdaptiveKPolicy.mark_padded`` / ``forget``.

Pipeline parallelism is refused, and a forced calibration depth
(``VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE``) is left as forced. It needs
``speculative_config.adaptive_k``. The first step refuses a K choice
without CUDA graphs: one adaptive_k does not capture up to the largest batch
size (``by_load`` lists it), or a ``cudagraph_capture_sizes`` without
b * (K + 1) for a batch size b of the block.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# The ModelRunnerOutput attribute holding a step's p_i rows.
P_HAT_ATTR = "sm80_dspark_p_hat"
ESTIMATES = ("stale",)
_LOG_EVERY = 4096
# Steps whose outputs may still be outstanding (bounded bookkeeping).
_MAX_IN_FLIGHT = 64


@dataclass(frozen=True)
class HostWidth:
    """A loaded ``host_width`` block (per batch size b of ``batch_sizes``)."""

    batch_sizes: frozenset[int]
    lag: int
    fallback_k: int
    # Ascending.
    k_choices: tuple[int, ...]
    t0: dict[int, float]
    # cum[b][k], k = 0 .. max K.
    cum: dict[int, tuple[float, ...]]
    # share[b][K]: the cost share of a masked row at physical width K.
    share: dict[int, dict[int, float]]
    tau: dict[int, float]

    @property
    def width(self) -> int:
        """The positions of p_i the choice reads."""
        return self.k_choices[-1]


def _finite(path, what: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{path}: host_width {what} is not a number: {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{path}: host_width {what} is not finite: {value!r}")
    return v


def _positive_ints(path, what: str, values: Any) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{path}: host_width {what} must be distinct positive integers: {values!r}")
    return tuple(sorted(values))


def parse_host_width(path, raw: dict) -> HostWidth:
    """The policy's ``host_width`` block, checked (ValueError when malformed)."""
    block = raw.get("host_width")
    if not isinstance(block, dict):
        raise ValueError(f"{path}: host_width must be an object")
    if block.get("estimate") not in ESTIMATES:
        raise ValueError(
            f"{path}: host_width estimate {block.get('estimate')!r}: only {ESTIMATES} "
            "(p_i of an earlier round) is served"
        )
    lag = block.get("lag_rounds")
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError(f"{path}: host_width lag_rounds must be a positive integer: {lag!r}")
    ks = _positive_ints(path, "k_choices", block.get("k_choices"))
    fallback = block.get("fallback_k")
    if fallback not in ks:
        raise ValueError(f"{path}: host_width fallback_k {fallback!r} is not in k_choices {list(ks)}")
    sizes = _positive_ints(path, "batch_sizes", block.get("batch_sizes"))
    if sizes[0] < 2:
        raise ValueError(f"{path}: host_width batch_sizes start at 2 (b = 1 takes the cost rule)")
    raw_tau = raw.get("tau")
    if not isinstance(raw_tau, dict):
        raise ValueError(f"{path}: host_width needs the policy's tau")
    t0, cum, share, tau = {}, {}, {}, {}
    for b in sizes:
        key = str(b)
        if key not in raw_tau:
            raise ValueError(f"{path}: host_width batch size {b} has no tau")
        tau[b] = _finite(path, f"tau[{b}]", raw_tau[key])
        if not 0.0 < tau[b] <= 1.0:
            raise ValueError(f"{path}: tau[{b}] = {tau[b]} is not in (0, 1]")
        try:
            t0_b, cum_b, share_b = (
                block["t0_ms"][key], block["cum_ms"][key], block["s"][key])
        except (KeyError, TypeError):
            raise ValueError(f"{path}: host_width needs t0_ms, cum_ms and s at batch size {b}") from None
        t0[b] = _finite(path, f"t0_ms[{b}]", t0_b)
        if t0[b] <= 0.0:
            raise ValueError(f"{path}: host_width t0_ms[{b}] = {t0[b]} is not positive")
        if not isinstance(cum_b, list) or len(cum_b) <= ks[-1]:
            raise ValueError(f"{path}: host_width cum_ms[{b}] needs k = 0 .. {ks[-1]}")
        row = tuple(_finite(path, f"cum_ms[{b}]", v) for v in cum_b)
        if row[0] != 0.0 or any(x > y for x, y in zip(row, row[1:])):
            raise ValueError(f"{path}: host_width cum_ms[{b}] must start at 0 and not decrease: {row}")
        cum[b] = row
        if not isinstance(share_b, dict):
            raise ValueError(f"{path}: host_width s[{b}] must be an object of K -> share")
        share[b] = {}
        for k in ks:
            if str(k) not in share_b:
                raise ValueError(f"{path}: host_width s[{b}] has no K = {k}")
            v = _finite(path, f"s[{b}][{k}]", share_b[str(k)])
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{path}: host_width s[{b}][{k}] = {v} is not in [0, 1]")
            share[b][k] = v
    return HostWidth(frozenset(sizes), lag, fallback, ks, t0, cum, share, tau)


@cache
def configured() -> HostWidth | None:
    """The host width of the policy ``VLLM_DSPARK_ADAPTIVE_DEPTH`` names, or
    None (switch off, or a policy without ``host_width``)."""
    path = envs.VLLM_DSPARK_ADAPTIVE_DEPTH
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    if raw.get("host_width") is None:
        return None
    return parse_host_width(path, raw)


# ----------------------------------------------------------------- the rule


def _depth(q, cap: int, tau: float) -> int:
    prod, k = 1.0, 0
    while k < cap:
        prod *= q[k]
        if prod < tau:
            break
        k += 1
    return k


def _expected(q, d: int) -> float:
    e = prod = 1.0
    for i in range(d):
        prod *= q[i]
        e += prod
    return e


def choose_width(hw: HostWidth, b: int, estimates, cap: int | None = None) -> int:
    """The K of ``hw.k_choices`` (those at most ``cap``) for a b-request step
    whose requests' p_i (each at least ``hw.width`` positions) are
    ``estimates``; ``cap`` when no choice is that narrow."""
    cum, t0, tau, share = hw.cum[b], hw.t0[b], hw.tau[b], hw.share[b]
    choices = [k for k in hw.k_choices if cap is None or k <= cap]
    if not choices:
        assert cap is not None
        return cap
    best, best_k = None, choices[0]
    for k in choices:
        depths = [_depth(q, k, tau) for q in estimates]
        tokens = sum(_expected(q, d) for q, d in zip(estimates, depths))
        ms = t0 + b * cum[k] - (1.0 - share[k]) * sum(cum[k] - cum[d] for d in depths)
        value = tokens / ms
        if best is None or value > best:
            best, best_k = value, k
    return best_k


# ---------------------------------------------------------------- scheduler


class _Round:
    __slots__ = ("p_hat",)

    def __init__(self) -> None:
        self.p_hat: list[float] | None = None


class _Rounds:
    """A scheduler's rounds (kept on its AdaptiveKPolicy): per request its
    last ``lag`` verify rounds, oldest first; per step in flight its rounds
    awaiting p_i."""

    def __init__(self, hw: HostWidth) -> None:
        self.hw = hw
        self.by_req: dict[str, deque[_Round]] = {}
        self.in_flight: dict[int, tuple[Any, dict[str, _Round]]] = {}
        # Requests the step being scheduled gives placeholder drafts.
        self.padded: set[str] = set()
        self.checked = False
        # Chosen K per step since the last log line (0: fallback, no estimates).
        self.chosen: Counter[int] = Counter()

    def scheduled(self, output) -> None:
        step: dict[str, _Round] = {}
        for req_id, drafts in output.scheduled_spec_decode_tokens.items():
            if not drafts:
                continue
            rnd = _Round()
            rounds = self.by_req.get(req_id)
            if rounds is None:
                rounds = self.by_req[req_id] = deque(maxlen=self.hw.lag)
            rounds.append(rnd)
            if req_id not in self.padded:
                step[req_id] = rnd
        if step:
            self.in_flight[id(output)] = (output, step)
            while len(self.in_flight) > _MAX_IN_FLIGHT:
                del self.in_flight[next(iter(self.in_flight))]

    def observed(self, scheduler_output, model_runner_output) -> None:
        entry = self.in_flight.pop(id(scheduler_output), None)
        if entry is None or entry[0] is not scheduler_output:
            return
        p_hat = getattr(model_runner_output, P_HAT_ATTR, None)
        if p_hat is None:
            return
        index = model_runner_output.req_id_to_index
        for req_id, rnd in entry[1].items():
            i = index.get(req_id)
            if i is not None and i < len(p_hat):
                rnd.p_hat = p_hat[i]

    def estimates(self, running) -> list[list[float]] | None:
        lag = self.hw.lag
        out = []
        for request in running:
            rounds = self.by_req.get(request.request_id)
            if rounds is None or len(rounds) < lag or rounds[-lag].p_hat is None:
                return None
            out.append(rounds[-lag].p_hat)
        return out

    def record(self, k: int) -> None:
        self.chosen[k] += 1
        total = sum(self.chosen.values())
        if total >= _LOG_EVERY:
            logger.info(
                "DSpark host width, last %d steps: %s; no estimates %d",
                total,
                ", ".join(f"K{k} {n}" for k, n in sorted(self.chosen.items()) if k),
                self.chosen.get(0, 0),
            )
            self.chosen.clear()


def _rounds(policy, hw: HostWidth) -> _Rounds:
    rounds = policy.__dict__.get("_dspark_host_rounds")
    if rounds is None:
        rounds = policy.__dict__["_dspark_host_rounds"] = _Rounds(hw)
    return rounds


def _check_captured(scheduler, hw: HostWidth) -> None:
    """Refuse a K choice without CUDA graphs at some batch size of the block,
    and pipeline parallelism (a micro-batch decodes part of the running
    requests, which the rule would cost as all of them)."""
    if scheduler.vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise ValueError(
            "VLLM_DSPARK_ADAPTIVE_DEPTH: host_width does not serve pipeline "
            "parallelism (each micro-batch decodes part of the running requests)"
        )
    config = scheduler.adaptive_k.config
    max_reqs = scheduler.max_num_running_reqs
    sizes = [b for b in sorted(hw.batch_sizes) if b <= max_reqs]
    if not sizes:
        return
    missing = [k for k in hw.k_choices if config.max_reqs_for(k, max_reqs) < sizes[-1]]
    if missing:
        raise ValueError(
            f"VLLM_DSPARK_ADAPTIVE_DEPTH: host_width chooses K {missing}, which "
            f"adaptive_k captures no CUDA graphs for up to {sizes[-1]} requests "
            f"(by_load {list(config.by_load)}); list them in adaptive_k.by_load, "
            "past max_num_seqs if no load should serve them"
        )
    capture = scheduler.vllm_config.compilation_config.cudagraph_capture_sizes
    if capture:
        absent = sorted({b * (k + 1) for b in sizes for k in hw.k_choices} - set(capture))
        if absent:
            raise ValueError(
                "VLLM_DSPARK_ADAPTIVE_DEPTH: host_width steps of "
                f"{absent} tokens have no cudagraph_capture_sizes entry"
            )


def install() -> None:
    """Wrap the scheduler when the policy has a host width (plugin
    registration; idempotent)."""
    hw = configured()
    if hw is None:
        return
    from vllm_sm80.spec_decode.adaptive_k import AdaptiveKPolicy
    from vllm.v1.core.sched.scheduler import Scheduler

    select = Scheduler._select_adaptive_k
    if getattr(select, "_dspark_host_width", False):
        return
    schedule, update = Scheduler.schedule, Scheduler.update_from_output
    mark_padded, forget = AdaptiveKPolicy.mark_padded, AdaptiveKPolicy.forget

    @functools.wraps(schedule)
    def _schedule(self, *args, **kwargs):
        if self.adaptive_k is None:
            raise ValueError(
                "VLLM_DSPARK_ADAPTIVE_DEPTH: a policy with host_width needs "
                "speculative_config.adaptive_k"
            )
        rounds = _rounds(self.adaptive_k, hw)
        rounds.padded.clear()
        output = schedule(self, *args, **kwargs)
        rounds.scheduled(output)
        return output

    @functools.wraps(update)
    def _update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs):
        if self.adaptive_k is not None:
            _rounds(self.adaptive_k, hw).observed(scheduler_output, model_runner_output)
        return update(self, scheduler_output, model_runner_output, *args, **kwargs)

    @functools.wraps(select)
    def _select_adaptive_k(self) -> tuple[int, int]:
        k_verify, k_draft = select(self)
        running = self.running
        b = len(running)
        if b not in hw.batch_sizes or any(r.is_prefill_chunk for r in running):
            return k_verify, k_draft
        if self._forced_depth() is not None:
            # Step-cost calibration: the forced depth stands.
            return k_verify, k_draft
        rounds = _rounds(self.adaptive_k, hw)
        if not rounds.checked:
            _check_captured(self, hw)
            rounds.checked = True
        estimates = rounds.estimates(running)
        if estimates is None:
            k = min(hw.fallback_k, k_verify)
        else:
            k = choose_width(hw, b, estimates, cap=k_verify)
        for request in running:
            if len(request.spec_token_ids) > k:
                # Rebind: AsyncScheduler shares one placeholder list.
                request.spec_token_ids = request.spec_token_ids[:k]
        rounds.record(0 if estimates is None else k)
        return k, k_draft

    @functools.wraps(mark_padded)
    def _mark_padded(self, req_id: str) -> None:
        _rounds(self, hw).padded.add(req_id)
        return mark_padded(self, req_id)

    @functools.wraps(forget)
    def _forget(self, req_id: str) -> None:
        _rounds(self, hw).by_req.pop(req_id, None)
        return forget(self, req_id)

    _select_adaptive_k._dspark_host_width = True  # type: ignore[attr-defined]
    Scheduler.schedule = _schedule
    Scheduler.update_from_output = _update_from_output
    Scheduler._select_adaptive_k = _select_adaptive_k
    AdaptiveKPolicy.mark_padded = _mark_padded
    AdaptiveKPolicy.forget = _forget


# ------------------------------------------------------------------- worker


def install_output_copy() -> None:
    """Wrap the GPU runner's AsyncOutput to carry the step's p_i to the host
    (worker side, when the row mask state is created; idempotent)."""
    import torch

    from vllm_sm80.spec_decode.dspark_row_mask import get_row_mask_state
    from vllm.v1.worker.gpu import async_utils

    cls = async_utils.AsyncOutput
    init, get_output = cls.__init__, cls.get_output
    if getattr(init, "_dspark_p_hat", False):
        return
    stream_index = tuple(inspect.signature(init).parameters).index("copy_stream") - 1

    @functools.wraps(init)
    def _init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        state = get_row_mask_state()
        p_hat = None if state is None else state.take_p_hat()
        if p_hat is None:
            return
        copy_stream = kwargs["copy_stream"] if "copy_stream" in kwargs else args[stream_index]
        with torch.cuda.stream(copy_stream):
            # The copy stream already waits for the step; recording the
            # output's event again makes its synchronize cover this copy.
            self._dspark_p_hat_gpu = p_hat
            self._dspark_p_hat = async_utils.async_copy_to_np(p_hat)
            self.copy_event.record(copy_stream)

    @functools.wraps(get_output)
    def _get_output(self):
        output = get_output(self)
        p_hat = self.__dict__.pop("_dspark_p_hat", None)
        if p_hat is not None:
            del self._dspark_p_hat_gpu
            setattr(output, P_HAT_ATTR, p_hat.tolist())
        return output

    _init._dspark_p_hat = True  # type: ignore[attr-defined]
    cls.__init__ = _init
    cls.get_output = _get_output
