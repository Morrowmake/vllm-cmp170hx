# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step state of the fork scheduler inside ``Scheduler.schedule()``.

One :class:`Sm80Step` is made per step (``vllm_sm80.sched.hooks.sm80_step``)
and carries what the step's two request loops share:

* fair chunked prefill: a second budget that only prefill tokens draw on,
  with a per-request cap, charged when a request is scheduled and refunded
  when a scheduled prefill is preempted. Decode tokens never draw on it.
  With both knobs at 0 it equals the token budget and never binds;
* the Mamba align-mode chunk ceiling of a prefill: the widest chunk the
  budgets could give it once other prefills' charges are restored, so
  decode occupancy does not shrink it;
* PP decode spreading: the decodes this micro-batch has taken against its
  share;
* the uniform-decode padding exemption of a cached prompt tail.

The request the loop is sizing is held between :meth:`cap` / :meth:`admit`
and :meth:`charge`; :meth:`charge` for any other request charges no prefill.
"""

import logging
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_sm80.sched.scheduler import Sm80SchedulerMixin

# The upstream scheduler's logger: VLLM_LOGGING_CONFIG_PATH can enable just
# this one for the per-step fair-prefill line.
logger = init_logger("vllm.v1.core.sched.scheduler")


class Sm80Step:
    __slots__ = (
        "scheduler",
        "draft_slots",
        "threshold",
        "prefill_token_budget",
        "prefill_chunk_cap",
        "num_prefill_tokens",
        "num_prefill_reqs",
        "pp_decode_cap",
        "num_pp_decodes",
        "request",
        "req_is_prefill",
        "is_prefill",
        "prefill_chunk_limit",
    )

    def __init__(
        self,
        scheduler: "Sm80SchedulerMixin",
        token_budget: int,
        draft_slots: int,
        long_prefill_token_threshold: int,
    ) -> None:
        self.scheduler = scheduler
        self.draft_slots = draft_slots
        self.threshold = long_prefill_token_threshold
        # When fair prefill is off the prefill budget equals `token_budget`
        # and can never bind, so the schedule is unchanged.
        self.prefill_token_budget = token_budget
        self.prefill_chunk_cap = token_budget
        if scheduler.fair_prefill:
            self.prefill_token_budget, self.prefill_chunk_cap = (
                scheduler._fair_prefill_budget()
            )
        # Prefill charges, refunded when scheduled prefills are preempted.
        self.num_prefill_tokens = 0
        self.num_prefill_reqs = 0
        self.pp_decode_cap = scheduler._pp_decode_cap()
        self.num_pp_decodes = 0
        self.request: Any = None
        self.req_is_prefill = False
        self.is_prefill = False
        self.prefill_chunk_limit: int | None = None

    def skip(self, request: Any) -> bool:
        """Running loop: whether this micro-batch already has its share of
        decodes under PP decode spreading (the request stays eligible for the
        next one)."""
        return (
            self.pp_decode_cap is not None
            and not request.is_prefill_chunk
            and self.num_pp_decodes >= self.pp_decode_cap
        )

    def _cap(self, request: Any, num_remaining: int, num_new_tokens: int) -> int:
        scheduler = self.scheduler
        self.request = request
        # A short prompt the bypass exempts is neither capped nor charged.
        self.is_prefill = (
            scheduler.fair_prefill
            and self.req_is_prefill
            and not scheduler._short_prefill_bypass(num_remaining)
        )
        self.prefill_chunk_limit = None
        if self.is_prefill:
            limit = min(self.prefill_chunk_cap, self.prefill_token_budget)
            if self.threshold > 0:
                limit = min(limit, self.threshold)
            self.prefill_chunk_limit = limit
            num_new_tokens = min(num_new_tokens, limit)
        return num_new_tokens

    def cap(self, request: Any, num_new_tokens: int) -> int:
        """Running loop: the request's chunk under the prefill budget."""
        self.req_is_prefill = request.is_prefill_chunk
        return self._cap(
            request, request.num_tokens - request.num_computed_tokens, num_new_tokens
        )

    def admit(self, request: Any, num_computed_tokens: int, num_new_tokens: int) -> int:
        """Waiting loop: the request's chunk under the prefill budget, or 0
        when no prefill budget is left this step (the request stays at the
        head of the queue; ``num_new_tokens`` is positive here).

        Same predicate the DP prefill throttle uses: a request only replaying
        its last token is a decode, not prefill work. A prefix-cache hit has
        already been folded into ``num_computed_tokens``, so only the tokens
        actually recomputed are charged.
        """
        self.req_is_prefill = num_computed_tokens < request.num_tokens - 1
        num_new_tokens = self._cap(
            request, request.num_tokens - num_computed_tokens, num_new_tokens
        )
        if self.prefill_chunk_limit is not None and self.prefill_chunk_limit <= 0:
            return 0
        return num_new_tokens

    def ceil(self, token_budget: int, input_budget: int) -> int | None:
        """Mamba align-mode chunk ceiling of the request being sized.

        Decodes can occupy the budgets indefinitely, so a prefill's ceiling
        restores other prefills' charges, not decode charges; a temporary
        prefill fragment still waits for a whole block on the next step.
        """
        if not self.req_is_prefill:
            return self.prefill_chunk_limit
        max_tokens = self.scheduler.max_num_scheduled_tokens
        return min(
            self.prefill_chunk_limit
            if self.prefill_chunk_limit is not None
            else max_tokens,
            self.threshold or max_tokens,
            token_budget + self.num_prefill_tokens,
            input_budget
            + self.num_prefill_tokens
            + (self.num_prefill_reqs - 1) * self.draft_slots,
        )

    def refund(self, request: Any, restored: int) -> None:
        """A request scheduled this step was preempted: refund exactly what
        :meth:`charge` charged."""
        if request.is_prefill_chunk:
            scheduler = self.scheduler
            if scheduler.fair_prefill and not scheduler._short_prefill_bypass(
                request.num_tokens - request.num_computed_tokens
            ):
                self.prefill_token_budget += restored
            self.num_prefill_tokens -= restored
            self.num_prefill_reqs -= 1

    def charge(self, request: Any, num_new_tokens: int) -> None:
        """The request was scheduled with ``num_new_tokens`` tokens."""
        if self.pp_decode_cap is not None and not request.is_prefill_chunk:
            self.num_pp_decodes += 1
        if request is not self.request:
            return
        if self.is_prefill:
            self.prefill_token_budget -= num_new_tokens
        if self.req_is_prefill:
            self.num_prefill_tokens += num_new_tokens
            self.num_prefill_reqs += 1
        self.request = None

    def prompt_tail(self, request: Any, num_computed_tokens: int) -> bool:
        """A prompt tail over cached state has no generated token or drafts to
        verify: it keeps its real prefill shape, not the uniform-decode
        padding, for every cache source and boundary policy. Resumed decodes
        keep padding."""
        return (
            num_computed_tokens > 0
            and request.num_output_tokens == 0
            and num_computed_tokens < request.num_prompt_tokens
        )

    def finish(
        self, total_num_scheduled_tokens: int, num_scheduled_tokens: dict[str, int]
    ) -> None:
        assert self.prefill_token_budget >= 0
        if logger.isEnabledFor(logging.DEBUG):
            # One line per step, enough to answer "did the decode streams ride
            # along with every prefill chunk?" from a server log.
            logger.debug(
                "fairprefill step=%d total=%d prefill=%d/%dreq "
                "decode=%d/%dreq cap=%d left=%d",
                self.scheduler.current_step,
                total_num_scheduled_tokens,
                self.num_prefill_tokens,
                self.num_prefill_reqs,
                total_num_scheduled_tokens - self.num_prefill_tokens,
                len(num_scheduled_tokens) - self.num_prefill_reqs,
                self.prefill_chunk_cap,
                self.prefill_token_budget,
            )
