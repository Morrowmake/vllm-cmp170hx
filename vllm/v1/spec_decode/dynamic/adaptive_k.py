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
    {"min", "max", "ema", "margin", "quantile", "allowed", "log_interval"}
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

    def forget(self, req_id: str) -> None:
        self._ema.pop(req_id, None)
        self._pending_padded.pop(req_id, None)

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
        logger.info(
            "Adaptive SD: %d steps, mean k=%.2f, accepted-draft prior=%.2f, %s",
            total,
            mean_k,
            self._prior,
            hist,
        )
        self._k_counts.clear()
        self._steps_since_log = 0
