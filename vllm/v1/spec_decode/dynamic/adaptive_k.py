# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-adaptive speculative draft count (host-side).

Picks the number of draft tokens to *verify* per scheduler step from an
exponential moving average of how many drafts each running request has
recently had accepted. Everything here is pure Python running on the
scheduler's CPU thread: nothing is trimmed on the device, so it composes with
attention backends (e.g. the DSA indexer) that reject vLLM's device-side
``enable_adaptive_verification``.

Because decode runs under uniform-batch full CUDA graphs, the choice is made
once per step for the whole batch (a low quantile over the per-request EMAs,
so no request in the batch is over-drafted), and the result is snapped to the
set of draft counts that have captured graphs.
"""

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Keys accepted in the ``adaptive_k`` speculative-config mapping.
_CONFIG_KEYS = frozenset(
    {"min", "max", "ema", "margin", "quantile", "allowed", "log_interval", "by_load",
     "draft_by_load", "accept"}
)


@dataclass
class AdaptiveKConfig:
    """Validated form of the ``adaptive_k`` speculative-config mapping."""

    min_k: int = 1
    max_k: int = 1
    ema: float = 0.8
    margin: float = 1.0
    quantile: float = 0.5
    # Draft counts that may be chosen, and therefore the counts that decode
    # CUDA graphs must be captured for. Always sorted, always ends at max_k.
    allowed: tuple[int, ...] = ()
    log_interval: int = 0
    # Load mode: the verification width for 1, 2, ... requests in the server;
    # the last entry covers every larger count. Empty: acceptance mode.
    by_load: tuple[int, ...] = ()
    # Load mode: also draft only the width the next step verifies (the
    # drafter must support per-width drafting; otherwise it drafts its full
    # block and the scheduler still verifies the load width).
    draft_by_load: bool = False
    # Load mode: also weigh recent draft acceptance (acceptance-aware depth).
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

    @property
    def load_mode(self) -> bool:
        return bool(self.by_load)

    def k_for_load(self, num_reqs: int) -> int:
        """Verification width for ``num_reqs`` requests (load mode)."""
        assert self.by_load
        return self.by_load[min(max(num_reqs, 1), len(self.by_load)) - 1]

    def max_reqs_for(self, k: int, max_num_reqs: int) -> int:
        """Largest request count at which ``k`` is verified, so decode CUDA
        graphs for ``k`` are needed only up to it. Every count in acceptance
        mode; 0 when load mode never picks ``k``."""
        if not self.by_load:
            return max_num_reqs
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
        drafter's buffers and the widest captured decode graph, so ``max`` can
        never exceed it.

        Every count in ``1..num_speculative_tokens`` is legal for both drafter
        families, because k is applied to *verification*, not to drafting: an
        autoregressive drafter (MTP) proposes into a fixed-width buffer, and
        DFlash2 emits its whole block in one fixed-shape pass whose graph must
        not change. In both cases a step verifies a prefix of the drafts and
        discards the tail, which is what keeps this CUDA-graph-safe.
        ``allowed`` exists so a deployment can restrict the set (e.g. DFlash2
        with ``[2, 4, 7]``) and keep the number of captured graphs small.
        """
        if not isinstance(raw, dict):
            raise ValueError(
                "speculative_config.adaptive_k must be a mapping, e.g. "
                '{"min": 1, "max": 3, "ema": 0.8}.'
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

        min_k = int(raw.get("min", 1))
        max_k = int(raw.get("max", num_speculative_tokens))
        ema = float(raw.get("ema", 0.8))
        margin = float(raw.get("margin", 1.0))
        quantile = float(raw.get("quantile", 0.5))
        log_interval = int(raw.get("log_interval", 0))

        if max_k > num_speculative_tokens:
            raise ValueError(
                f"adaptive_k.max ({max_k}) must be <= num_speculative_tokens "
                f"({num_speculative_tokens}); raise num_speculative_tokens "
                "instead, it sizes the drafter buffers and CUDA graphs."
            )
        if min_k < 1:
            raise ValueError("adaptive_k.min must be >= 1.")
        if min_k > max_k:
            raise ValueError(f"adaptive_k.min ({min_k}) must be <= max ({max_k}).")
        if not 0.0 <= ema < 1.0:
            raise ValueError(f"adaptive_k.ema ({ema}) must be in [0, 1).")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"adaptive_k.quantile ({quantile}) must be in [0, 1].")
        if log_interval < 0:
            raise ValueError("adaptive_k.log_interval must be >= 0.")

        raw_by_load = raw.get("by_load")
        by_load: tuple[int, ...] = ()
        if raw_by_load is not None:
            if not isinstance(raw_by_load, (list, tuple)) or not raw_by_load:
                raise ValueError(
                    "adaptive_k.by_load must be a non-empty list of draft "
                    "counts (for 1, 2, ... requests; the last covers the rest)."
                )
            by_load = tuple(int(k) for k in raw_by_load)
            illegal = sorted(
                {k for k in by_load if not 1 <= k <= num_speculative_tokens}
            )
            if illegal:
                raise ValueError(
                    f"adaptive_k.by_load contains {illegal}, outside "
                    f"[1, num_speculative_tokens={num_speculative_tokens}]."
                )
            if "allowed" in raw or "min" in raw or "max" in raw:
                raise ValueError(
                    "adaptive_k.by_load names the draft counts itself; drop "
                    "allowed/min/max."
                )
            allowed = tuple(sorted(set(by_load)))
            accept_kwargs: dict[str, Any] = {}
            raw_accept = raw.get("accept")
            if raw_accept is not None:
                if not isinstance(raw_accept, dict):
                    raise ValueError("adaptive_k.accept must be a mapping.")
                # The acceptance-aware choice may pick any depth from the
                # shallowest load width up to the deepest one.
                min_depth = int(raw_accept.get("min_depth", min(by_load)))
                if not 1 <= min_depth <= min(by_load):
                    raise ValueError(
                        "adaptive_k.accept.min_depth must be between 1 "
                        "and the shallowest load width"
                    )
                allowed = tuple(range(min_depth, max(by_load) + 1))
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
                min_k=allowed[0],
                max_k=max(by_load),
                ema=ema,
                margin=margin,
                quantile=quantile,
                allowed=allowed,
                log_interval=log_interval,
                by_load=by_load,
                draft_by_load=bool(raw.get("draft_by_load", False)),
                **accept_kwargs,
            )

        if raw.get("draft_by_load"):
            raise ValueError("adaptive_k.draft_by_load needs adaptive_k.by_load.")
        if raw.get("accept") is not None:
            raise ValueError("adaptive_k.accept needs adaptive_k.by_load.")
        raw_allowed = raw.get("allowed")
        if raw_allowed is None:
            candidates = list(range(1, num_speculative_tokens + 1))
        else:
            if not isinstance(raw_allowed, (list, tuple)) or not raw_allowed:
                raise ValueError("adaptive_k.allowed must be a non-empty list of ints.")
            candidates = [int(k) for k in raw_allowed]
            illegal = sorted(
                k for k in candidates if not 1 <= k <= num_speculative_tokens
            )
            if illegal:
                raise ValueError(
                    f"adaptive_k.allowed contains {illegal}, outside "
                    f"[1, num_speculative_tokens={num_speculative_tokens}]."
                )

        allowed = sorted({k for k in candidates if min_k <= k <= max_k})
        if not allowed:
            raise ValueError(
                f"adaptive_k.allowed has no value inside [min={min_k}, max={max_k}]."
            )

        return cls(
            min_k=allowed[0],
            max_k=allowed[-1],
            ema=ema,
            margin=margin,
            quantile=quantile,
            allowed=tuple(allowed),
            log_interval=log_interval,
        )


@dataclass
class AdaptiveKPolicy:
    """Per-request acceptance EMAs and the per-step draft-count choice."""

    config: AdaptiveKConfig
    # req_id -> EMA of the number of *draft* tokens accepted per step.
    _ema: dict[str, float] = field(default_factory=dict)
    # Model-wide running mean of accepted drafts, used to seed new requests so
    # they start near the typical k instead of at min_k.
    _prior: float = 0.0
    _prior_weight: int = 0
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

    def __post_init__(self) -> None:
        # Seed the prior so the very first steps behave like fixed k=max
        # rather than like k=min; it converges to the real mean within a few
        # tens of steps at the default decay.
        self._prior = max(self.config.max_k - self.config.margin, 0.0)

    # ---------------------------------------------------------------- feedback

    def mark_padded(self, req_id: str) -> None:
        """Note that this step's "drafts" for ``req_id`` are padding.

        The scheduler pads a request that re-enters decode with
        ``[-1] * k`` placeholder drafts so the batch stays uniform and inside
        its captured CUDA graph. Those placeholders are always rejected, so
        feeding the result to the EMA would be a fabricated zero-acceptance
        sample that drags k down for several steps. Count them here and skip
        the matching :meth:`observe`; a request's outputs arrive in the order
        its steps were scheduled, so a counter is enough even with pipeline
        parallelism or async scheduling putting several steps in flight.
        """
        self._pending_padded[req_id] = self._pending_padded.get(req_id, 0) + 1

    def observe(self, req_id: str, num_draft_tokens: int, num_accepted: int) -> None:
        """Record one verification result for a request.

        ``num_accepted`` is the number of *draft* tokens accepted (the bonus
        token is not a draft), so it lies in ``[0, num_draft_tokens]``.

        A step that offered k drafts and had all k accepted is a
        *right-censored* observation: acceptance is at least k, and how much
        more is unobservable at that width. Recording k as though it were the
        truth biases the average down exactly where k should be growing, which
        makes every width its own self-fulfilling equilibrium -- measured on
        GLM-5.3-Flash / DFlash2, k latched at 2 and gave up 16% of aggregate
        throughput against a fixed k=3. Such a sample is therefore
        extrapolated by ``margin``, the same step the policy aims one beyond
        the expected acceptance.
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
        sample = float(min(max(num_accepted, 0), num_draft_tokens))
        if num_accepted >= num_draft_tokens:
            # Right-censored (see the docstring): extrapolate past the ceiling
            # this width imposed, or k can never grow out of it.
            sample += self.config.margin

        decay = self.config.ema
        prev = self._ema.get(req_id)
        self._ema[req_id] = (
            sample if prev is None else decay * prev + (1.0 - decay) * sample
        )

        # Running mean over all observations, used to seed new requests.
        self._prior_weight = min(self._prior_weight + 1, 10_000)
        self._prior += (sample - self._prior) / self._prior_weight

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
        self._ema.pop(req_id, None)
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

    def _k_for(self, req_id: str) -> int:
        ema = self._ema.get(req_id, self._prior)
        k = int(round(ema + self.config.margin))
        return min(max(k, self.config.min_k), self.config.max_k)

    def snap(self, k: int) -> int:
        """Round ``k`` down to the nearest draft count that has a CUDA graph."""
        best = self.config.allowed[0]
        for value in self.config.allowed:
            if value > k:
                break
            best = value
        return best

    def select_k(self, req_ids: list[str]) -> int:
        """Choose the draft count for a step covering ``req_ids``.

        Decode runs as a uniform batch under full CUDA graphs, so one k must
        serve every request in the step, taken as a quantile of the
        per-request choices and snapped to a captured graph size.

        The default is the median rather than the minimum. The minimum reads
        as the safe choice -- nobody is over-drafted -- but it is a minimum
        over a growing sample, so it falls as concurrency rises even when
        acceptance does not: on the GLM-5.3-Flash / DFlash2 deployment it
        picked k=2 for 87-98% of steps at concurrency 4-8 and cost 16% of
        aggregate throughput against a fixed k=3. Over-drafting one straggler
        wastes a verification row; under-drafting the whole batch wastes a
        step.
        """
        if not req_ids:
            return self.config.max_k
        ks = sorted(self._k_for(req_id) for req_id in req_ids)
        idx = int(self.config.quantile * (len(ks) - 1))
        return self.snap(ks[idx])

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
            logger.info(
                "Adaptive SD: %d steps, mean k=%.2f, accepted-draft prior=%.2f, %s",
                total,
                mean_k,
                self._prior,
                hist,
            )
        self._k_counts.clear()
        self._steps_since_log = 0
