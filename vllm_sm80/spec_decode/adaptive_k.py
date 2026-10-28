# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-adaptive speculative draft count (host-side).

Picks the number of draft tokens to *verify* per scheduler step from the
number of requests the server holds (``by_load``), optionally refined by the
recent per-draft acceptance of the running requests (``accept``), and can ask
the drafter to draft only that width (``draft_by_load``). Everything here is
pure Python running on the scheduler's CPU thread: nothing is trimmed on the
device, so it composes with attention backends (e.g. the DSA indexer) that
reject vLLM's device-side ``enable_adaptive_verification``.

Because decode runs under uniform-batch full CUDA graphs, the choice is made
once per step for the whole batch and is always one of the draft counts that
have captured graphs.
"""

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Keys accepted in the ``adaptive_k`` speculative-config mapping.
_CONFIG_KEYS = frozenset({"by_load", "draft_by_load", "accept", "log_interval"})


@dataclass
class AdaptiveKConfig:
    """Validated form of the ``adaptive_k`` speculative-config mapping."""

    # The verification width for 1, 2, ... requests in the server; the last
    # entry covers every larger count.
    by_load: tuple[int, ...]
    # Draft counts that may be chosen, and therefore the counts that decode
    # CUDA graphs must be captured for. Always sorted.
    allowed: tuple[int, ...]
    log_interval: int = 0
    # Also draft only the width the next step verifies (the drafter must
    # support per-width drafting; otherwise it drafts its full block and the
    # scheduler still verifies the load width).
    draft_by_load: bool = False
    # Also weigh recent draft acceptance (acceptance-aware depth).
    # accept_costs[i] is the relative step cost of verifying allowed[i]
    # drafts; the step takes the depth (at most the load width) that
    # maximises expected tokens per unit cost over the batch.
    accept: bool = False
    accept_costs: tuple[float, ...] = ()
    # Costs for steps of two or more requests (empty: accept_costs).
    accept_costs_multi: tuple[float, ...] = ()
    accept_hysteresis: float = 0.03
    accept_prior: float = 0.75
    accept_decay: float = 0.9
    accept_strength: float = 2.0

    def k_for_load(self, num_reqs: int) -> int:
        """Verification width for ``num_reqs`` requests."""
        return self.by_load[min(max(num_reqs, 1), len(self.by_load)) - 1]

    def max_reqs_for(self, k: int, max_num_reqs: int) -> int:
        """Largest request count at which ``k`` is verified, so decode CUDA
        graphs for ``k`` are needed only up to it; 0 when ``k`` is never
        picked."""
        if k not in self.allowed:
            return 0
        if k <= self.by_load[-1]:
            return max_num_reqs
        # A step verifies at most the load width, so depth k runs only while
        # the load width is at least k (exactly the load width without the
        # acceptance-aware choice, which then only lists load widths).
        widest = 0
        for n, value in enumerate(self.by_load, start=1):
            if value >= k:
                widest = n
        return min(widest, max_num_reqs)

    @classmethod
    def from_dict(cls, raw: Any, num_speculative_tokens: int) -> "AdaptiveKConfig":
        """Build from the user mapping.

        ``num_speculative_tokens`` is the hard upper bound: it sizes the
        drafter's buffers and the widest captured decode graph, so no load
        width can exceed it.

        Every count in ``1..num_speculative_tokens`` is legal for both drafter
        families, because the count is applied to *verification*: an
        autoregressive drafter (MTP) proposes into a fixed-width buffer, and a
        DFlash block drafter emits its whole block in one fixed-shape pass
        unless ``draft_by_load`` gives it one graph set per width. In both
        cases a step verifies a prefix of the drafts and discards the tail,
        which is what keeps this CUDA-graph-safe.
        """
        if not isinstance(raw, dict):
            raise ValueError(
                "speculative_config.adaptive_k must be a mapping, e.g. "
                '{"by_load": [7, 5, 3]}.'
            )
        if "by_load" not in raw:
            raise ValueError(
                "speculative_config.adaptive_k needs by_load (the draft count "
                "for 1, 2, ... requests, the last covering the rest); the "
                "acceptance-EMA policy (min/max/ema/margin/quantile/allowed) "
                "has been removed."
            )
        unknown = set(raw) - _CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown adaptive_k keys {sorted(unknown)}; "
                f"expected a subset of {sorted(_CONFIG_KEYS)}."
            )
        if num_speculative_tokens <= 0:
            raise ValueError(
                "adaptive_k requires num_speculative_tokens > 0 (it is the "
                "maximum draft count and sizes the drafter's buffers)."
            )

        log_interval = int(raw.get("log_interval", 0))
        if log_interval < 0:
            raise ValueError("adaptive_k.log_interval must be >= 0.")

        raw_by_load = raw["by_load"]
        if not isinstance(raw_by_load, (list, tuple)) or not raw_by_load:
            raise ValueError(
                "adaptive_k.by_load must be a non-empty list of draft "
                "counts (for 1, 2, ... requests; the last covers the rest)."
            )
        by_load = tuple(int(k) for k in raw_by_load)
        illegal = sorted({k for k in by_load if not 1 <= k <= num_speculative_tokens})
        if illegal:
            raise ValueError(
                f"adaptive_k.by_load contains {illegal}, outside "
                f"[1, num_speculative_tokens={num_speculative_tokens}]."
            )
        allowed = tuple(sorted(set(by_load)))
        accept_kwargs: dict[str, Any] = {}
        raw_accept = raw.get("accept")
        if raw_accept is not None:
            if not isinstance(raw_accept, dict):
                raise ValueError("adaptive_k.accept must be a mapping.")
            # The acceptance-aware choice may pick any depth from the
            # shallowest load width up to the deepest one.
            allowed = tuple(range(min(by_load), max(by_load) + 1))
            costs = tuple(float(c) for c in raw_accept.get("costs", ()))
            if len(costs) != len(allowed) or min(costs, default=0.0) <= 0:
                raise ValueError(
                    f"adaptive_k.accept.costs needs one positive cost per "
                    f"draft count {list(allowed)}, got {list(costs)}."
                )
            costs_multi = tuple(
                float(c) for c in raw_accept.get("costs_multi", ()) or ()
            )
            if costs_multi and (
                len(costs_multi) != len(allowed) or min(costs_multi) <= 0
            ):
                raise ValueError(
                    f"adaptive_k.accept.costs_multi needs one positive cost "
                    f"per draft count {list(allowed)}, got {list(costs_multi)}."
                )
            hysteresis = float(raw_accept.get("hysteresis", 0.03))
            prior = float(raw_accept.get("prior", 0.75))
            decay = float(raw_accept.get("decay", 0.9))
            if not 0.0 <= hysteresis < 1.0:
                raise ValueError("adaptive_k.accept.hysteresis must be in [0, 1).")
            if not 0.0 < prior < 1.0 or not 0.0 <= decay < 1.0:
                raise ValueError(
                    "adaptive_k.accept.prior must be in (0, 1) and decay in [0, 1)."
                )
            accept_kwargs = dict(
                accept=True,
                accept_costs=costs,
                accept_costs_multi=costs_multi,
                accept_hysteresis=hysteresis,
                accept_prior=prior,
                accept_decay=decay,
            )
        return cls(
            by_load=by_load,
            allowed=allowed,
            log_interval=log_interval,
            draft_by_load=bool(raw.get("draft_by_load", False)),
            **accept_kwargs,
        )


@dataclass
class AdaptiveKPolicy:
    """Per-request acceptance counts and the per-step depth choice."""

    config: AdaptiveKConfig
    # Histogram of chosen k, for the periodic summary log line.
    _k_counts: dict[int, int] = field(default_factory=dict)
    _steps_since_log: int = 0
    # req_id -> number of in-flight steps whose "drafts" were the scheduler's
    # ``[-1] * k`` uniform-decode padding rather than real proposals. Those are
    # rejected by construction, so their result is not evidence about
    # acceptance; see :meth:`mark_padded`.
    _pending_padded: dict[str, int] = field(default_factory=dict)
    # Acceptance-aware depth: per request, decayed counts of accepted drafts
    # and of rejections (a geometric model of per-draft acceptance; a step
    # whose drafts were all accepted adds successes and no failure, so no
    # censoring correction is needed). A request's depth depends only on its
    # own history and on the batch it runs in: new requests start from the
    # fixed configured prior and the hysteresis reference is per request, so
    # an identical request run alone repeats exactly whatever ran before it.
    _acc_s: dict[str, float] = field(default_factory=dict)
    _acc_f: dict[str, float] = field(default_factory=dict)
    _accept_last: dict[str, int] = field(default_factory=dict)
    # Model-wide observed counts, for the log line only (never read by the
    # choice).
    _obs_s: float = 0.0
    _obs_f: float = 0.0

    # ---------------------------------------------------------------- feedback

    def mark_padded(self, req_id: str) -> None:
        """Note that this step's "drafts" for ``req_id`` are padding.

        The scheduler pads a request that re-enters decode with
        ``[-1] * k`` placeholder drafts so the batch stays uniform and inside
        its captured CUDA graph. Those placeholders are always rejected, so
        counting the result would be a fabricated zero-acceptance sample that
        drags the depth down for several steps. Count them here and skip the
        matching :meth:`observe`; a request's outputs arrive in the order its
        steps were scheduled, so a counter is enough even with pipeline
        parallelism or async scheduling putting several steps in flight.
        """
        self._pending_padded[req_id] = self._pending_padded.get(req_id, 0) + 1

    def observe(self, req_id: str, num_draft_tokens: int, num_accepted: int) -> None:
        """Record one verification result for a request.

        ``num_accepted`` is the number of *draft* tokens accepted (the bonus
        token is not a draft), so it lies in ``[0, num_draft_tokens]``.
        """
        if num_draft_tokens <= 0:
            return
        pending_padded = self._pending_padded.get(req_id)
        if pending_padded:
            # Placeholder drafts: no information about acceptance.
            if pending_padded == 1:
                del self._pending_padded[req_id]
            else:
                self._pending_padded[req_id] = pending_padded - 1
            return

        if self.config.accept:
            accepted = min(max(num_accepted, 0), num_draft_tokens)
            rejected = 1.0 if accepted < num_draft_tokens else 0.0
            d = self.config.accept_decay
            self._acc_s[req_id] = d * self._acc_s.get(req_id, 0.0) + accepted
            self._acc_f[req_id] = d * self._acc_f.get(req_id, 0.0) + rejected
            gd = 0.999
            self._obs_s = gd * self._obs_s + accepted
            self._obs_f = gd * self._obs_f + rejected

    def forget(self, req_id: str) -> None:
        self._pending_padded.pop(req_id, None)
        self._acc_s.pop(req_id, None)
        self._acc_f.pop(req_id, None)
        self._accept_last.pop(req_id, None)

    # ------------------------------------------------- acceptance-aware depth

    def accept_prior(self) -> float:
        """The fixed per-draft acceptance new requests start from (config,
        never adapted at run time: a model-wide running mean would make a
        request's depth depend on the requests before it)."""
        return self.config.accept_prior

    def observed_acceptance(self) -> float:
        """Model-wide observed per-draft acceptance (logging only)."""
        total = self._obs_s + self._obs_f
        return self._obs_s / total if total else float("nan")

    def accept_rate(self, req_id: str) -> float:
        """This request's per-draft acceptance, shrunk towards the fixed
        prior."""
        a = self.config.accept_strength
        p0 = self.accept_prior()
        s = self._acc_s.get(req_id, 0.0)
        f = self._acc_f.get(req_id, 0.0)
        return (s + a * p0) / (s + f + a)

    @staticmethod
    def expected_tokens(p: float, k: int) -> float:
        """Tokens a step commits at depth k (bonus included) when each draft
        is accepted with probability p given the earlier ones were."""
        p = min(max(p, 0.0), 0.999)
        return (1.0 - p ** (k + 1)) / (1.0 - p)

    def select_by_acceptance(self, req_ids: list[str], cap: int) -> int:
        """The depth (a captured count, at most ``cap``, the load width) that
        maximises the batch's expected tokens per unit step cost. Keeps the
        current depth unless another beats it by the hysteresis margin."""
        cfg = self.config
        table = (
            cfg.accept_costs_multi
            if len(req_ids) > 1 and cfg.accept_costs_multi
            else cfg.accept_costs
        )
        candidates = [(k, cost) for k, cost in zip(cfg.allowed, table) if k <= cap]
        if not candidates or not req_ids:
            return cap
        rates = [self.accept_rate(r) for r in req_ids]
        scores = {
            k: sum(self.expected_tokens(p, k) for p in rates) / cost
            for k, cost in candidates
        }
        best = max(scores, key=lambda k: (scores[k], -k))
        # Hysteresis against the depth these requests last ran at (only when
        # they agree on one; a request's own history, never another's).
        lasts = {self._accept_last.get(r) for r in req_ids}
        last = lasts.pop() if len(lasts) == 1 else None
        if (
            last is not None
            and last in scores
            and last != best
            and scores[best] <= scores[last] * (1.0 + cfg.accept_hysteresis)
        ):
            best = last
        for r in req_ids:
            self._accept_last[r] = best
        return best

    # ------------------------------------------------------------------ policy

    def snap(self, k: int) -> int:
        """Round ``k`` down to the nearest draft count that has a CUDA graph."""
        best = self.config.allowed[0]
        for value in self.config.allowed:
            if value > k:
                break
            best = value
        return best

    # ----------------------------------------------------------------- logging

    def record_choice(self, k: int) -> None:
        self._k_counts[k] = self._k_counts.get(k, 0) + 1
        self._steps_since_log += 1

    def maybe_log(self) -> None:
        interval = self.config.log_interval
        if not interval or self._steps_since_log < interval:
            return
        total = sum(self._k_counts.values()) or 1
        hist = " ".join(
            f"k={k}:{n}({100 * n / total:.0f}%)"
            for k, n in sorted(self._k_counts.items())
        )
        mean_k = sum(k * n for k, n in self._k_counts.items()) / total
        if self.config.accept:
            logger.info(
                "Adaptive SD: %d steps, mean k=%.2f, per-draft acceptance "
                "observed=%.3f (prior %.2f), %s",
                total,
                mean_k,
                self.observed_acceptance(),
                self.accept_prior(),
                hist,
            )
        else:
            logger.info("Adaptive SD: %d steps, mean k=%.2f, %s", total, mean_k, hist)
        self._k_counts.clear()
        self._steps_since_log = 0
