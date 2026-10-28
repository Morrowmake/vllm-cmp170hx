# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``Sm80Scheduler`` / ``Sm80AsyncScheduler``: upstream's two schedulers
with the fork's scheduling features, selected through
``SchedulerConfig.scheduler_cls`` (``vllm_sm80.sched.select_scheduler``,
called by the fork's CUDA platform when no scheduler class is configured).

* Fair chunked prefill (``prefill_chunk_with_decodes``,
  ``max_num_partial_prefills``; ``VLLM_FAIR_PREFILL_SHORT_BYPASS``): a
  decode-aware prefill budget split between concurrent prefills.
* PP decode spreading (``VLLM_PP_SPREAD_DECODES``).
* Load-adaptive draft count (``speculative_config.adaptive_k``): each step
  verifies one width chosen before any request is sized, optionally refined
  by the DSpark depth hint (``VLLM_DSPARK_DEPTH_HINT``) and host width
  (``VLLM_DSPARK_ADAPTIVE_DEPTH`` with a ``host_width`` block).
* The Mamba align-mode state grid: the Mamba group's own block size.

The per-step decisions inside ``schedule()``'s request loops are
``vllm_sm80.sched.step.Sm80Step``, reached from the upstream file through
``vllm_sm80.sched.hooks``. The upstream methods overridden here are pinned in
tests/sm80/seams.yaml.
"""

import itertools
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import PauseState, Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request
from vllm_sm80.sched.step import Sm80Step
from vllm_sm80.spec_decode import dspark_depth_hint, dspark_host_width
from vllm_sm80.spec_decode.adaptive_k import AdaptiveKPolicy

logger = init_logger(__name__)


class Sm80SchedulerMixin:
    """The fork's scheduling features; listed before the upstream base."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        vllm_config = self.vllm_config
        speculative_config = vllm_config.speculative_config

        self.adaptive_k: AdaptiveKPolicy | None = None
        if speculative_config is not None and speculative_config.uses_adaptive_k():
            self.adaptive_k = AdaptiveKPolicy(speculative_config.adaptive_k_config)
            by_load = self.adaptive_k.config.by_load
            logger.info_once(
                "GLM-5 load-adaptive DFlash depth active "
                "(adaptive_k.by_load): drafts %d per step, "
                "verifies %s requests",
                self.num_spec_tokens,
                ", ".join(
                    f"{k} at {n}" if n < len(by_load) else f"{k} at >= {n}"
                    for n, k in enumerate(by_load, start=1)
                ),
            )
            if self.adaptive_k.config.accept:
                cfg = self.adaptive_k.config
                logger.info_once(
                    "GLM-5 acceptance-aware DFlash depth active "
                    "(adaptive_k.accept): each step's depth "
                    "within the load width maximises expected tokens per unit "
                    "step cost; costs %s for depths %s, hysteresis %.2f",
                    str(cfg.accept_costs),
                    str(cfg.allowed),
                    cfg.accept_hysteresis,
                )
        # DSpark refinements of the adaptive width (each off unless its
        # switch is set): the depth hint, then the host width on top of it.
        self._depth_hint = dspark_depth_hint.enabled()
        self._host_width = dspark_host_width.configured()

        # PP decode spreading (VLLM_PP_SPREAD_DECODES): a decode request is not
        # eligible again until its results return pp_size steps later, so a
        # step that takes every eligible decode leaves the next pp_size - 1
        # micro-batches with none and the stages idle in turn. Capping each
        # step at ceil(decoding / pp_size) decodes keeps every micro-batch
        # busy with its share instead.
        self.pp_spread_decodes = self.use_pp and envs.VLLM_PP_SPREAD_DECODES
        if self.pp_spread_decodes:
            logger.info(
                "PP decode spreading on: at most ceil(decoding requests / %d) "
                "decode requests per micro-batch",
                self.parallel_config.pipeline_parallel_size,
            )

        # Fair chunked prefill (opt-in). Both knobs at 0 -> `fair_prefill` is
        # False and every fair-prefill path is bypassed, so the schedule is
        # identical to upstream.
        self.prefill_chunk_with_decodes = (
            self.scheduler_config.prefill_chunk_with_decodes
        )
        self.max_num_partial_prefills = self.scheduler_config.max_num_partial_prefills
        self.fair_prefill = self.scheduler_config.enable_chunked_prefill and bool(
            self.prefill_chunk_with_decodes or self.max_num_partial_prefills
        )
        # VLLM_FAIR_PREFILL_SHORT_BYPASS (default off): a prompt whose whole
        # remaining prefill fits in `prefill_chunk_with_decodes` is scheduled
        # in one step, outside the capped prefill budget and the split. Off ->
        # `_short_prefill_bypass` is always False and nothing changes.
        self.fair_prefill_short_bypass = (
            self.fair_prefill
            and self.prefill_chunk_with_decodes > 0
            and envs.VLLM_FAIR_PREFILL_SHORT_BYPASS
        )

        # The align-mode state grid is the Mamba group's own block size: the
        # worker writes a chunk's end state into Mamba column
        # (end - 1) // MambaSpec.block_size. `cache_config.block_size` is the
        # minimum over prefix-cacheable groups and can be finer (a drafter
        # group with a smaller block); splitting on it lets a chunk run past a
        # Mamba boundary, and the block left behind is then hashed as that
        # boundary's state. Read by vllm_sm80.sched.hooks.mamba_block_size.
        mamba_state_block_sizes = {
            group.kv_cache_spec.block_size
            for group in self.kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        }
        assert len(mamba_state_block_sizes) <= 1, (
            "Mamba align scheduling needs a single Mamba state block size, "
            f"got {sorted(mamba_state_block_sizes)}"
        )
        self.mamba_state_block_size = (
            next(iter(mamba_state_block_sizes)) if mamba_state_block_sizes else None
        )

    # ------------------------------------------------------------------ step

    def _sm80_new_step(
        self, token_budget: int, draft_slots: int, long_prefill_token_threshold: int
    ) -> Sm80Step:
        """vllm_sm80.sched.hooks.sm80_step: this step's loop state."""
        return Sm80Step(self, token_budget, draft_slots, long_prefill_token_threshold)

    def schedule(self, *args: Any, **kwargs: Any) -> SchedulerOutput:
        """Pick the step's draft widths, then upstream's schedule.

        The adaptive width is chosen once, on the host, before any request
        is sized, so the uniform-decode padding and the truncation of
        already-proposed drafts agree on one width for the whole step --
        which is what keeps the decode batch inside a captured full CUDA
        graph. ``cur_num_spec_tokens`` is what this step verifies,
        ``next_num_spec_tokens`` what the drafter produces for the next one.
        """
        adaptive_k = self.adaptive_k
        host_rounds = None
        if self._host_width is not None:
            if adaptive_k is None:
                raise ValueError(
                    "VLLM_DSPARK_ADAPTIVE_DEPTH: a policy with host_width needs "
                    "speculative_config.adaptive_k"
                )
            host_rounds = dspark_host_width.rounds_of(adaptive_k, self._host_width)
            host_rounds.padded.clear()
        self.cur_num_spec_tokens = self.num_spec_tokens
        self.next_num_spec_tokens = self.num_spec_tokens
        running_ids: set[str] | None = None
        if adaptive_k is not None and self.num_spec_tokens > 0:
            running_ids = {request.request_id for request in self.running}
            self.cur_num_spec_tokens, self.next_num_spec_tokens = (
                self._select_adaptive_k()
            )
        output = super().schedule(*args, **kwargs)  # type: ignore[misc]
        if running_ids is not None:
            # Requests the waiting loop scheduled with drafts were padded with
            # placeholder drafts, rejected by construction: keep them out of
            # the acceptance counts. (A request running at the start of the
            # step never reaches the waiting loop: it only runs when nothing
            # was preempted.)
            for req_id in output.scheduled_spec_decode_tokens:
                if req_id not in running_ids:
                    self._mark_padded(req_id)
        if host_rounds is not None:
            host_rounds.scheduled(output)
        return output

    def update_from_output(
        self, scheduler_output, model_runner_output, *args, **kwargs
    ):
        if self._host_width is not None and self.adaptive_k is not None:
            dspark_host_width.rounds_of(self.adaptive_k, self._host_width).observed(
                scheduler_output, model_runner_output
            )
        return super().update_from_output(  # type: ignore[misc]
            scheduler_output, model_runner_output, *args, **kwargs
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ):
        """Upstream's, after feeding the adaptive-k acceptance counts (called
        once per request whose drafts were verified this step)."""
        adaptive_k = self.adaptive_k
        if adaptive_k is not None:
            # Grammar-invalidated drafts were never really proposed, so they
            # must not count against the acceptance counts.
            num_invalid = (
                num_invalid_spec_tokens.get(request_id, 0)
                if num_invalid_spec_tokens
                else 0
            )
            num_drafts = num_draft_tokens - num_invalid
            if self._depth_hint:
                dspark_depth_hint.observe(
                    adaptive_k, request_id, num_drafts, num_accepted_tokens
                )
            adaptive_k.observe(request_id, num_drafts, num_accepted_tokens)
        return super().make_spec_decoding_stats(  # type: ignore[misc]
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )

    def _free_request(self, request: Request, *args: Any, **kwargs: Any):
        adaptive_k = self.adaptive_k
        if adaptive_k is not None:
            request_id = request.request_id
            if self._host_width is not None:
                dspark_host_width.forget(adaptive_k, self._host_width, request_id)
            if self._depth_hint:
                dspark_depth_hint.forget(adaptive_k, request_id)
            adaptive_k.forget(request_id)
        return super()._free_request(request, *args, **kwargs)  # type: ignore[misc]

    def _mark_padded(self, req_id: str) -> None:
        assert self.adaptive_k is not None
        if self._host_width is not None:
            dspark_host_width.rounds_of(self.adaptive_k, self._host_width).padded.add(
                req_id
            )
        self.adaptive_k.mark_padded(req_id)

    # --------------------------------------------------------- fair prefill

    def _fair_prefill_budget(self) -> tuple[int, int]:
        """Fair chunked prefill: this step's prefill budget and per-request cap.

        Returns ``(total, per_request)``, both counted in prefill tokens only.
        Decode tokens draw on ``token_budget`` / ``input_budget`` as usual and
        are never charged here, which is what lets a decode stream keep its
        slot in every step while someone else's prompt is being prefilled.

        * ``total`` is ``max_num_scheduled_tokens``, narrowed to
          ``prefill_chunk_with_decodes`` whenever at least one running request
          is past its prefill (i.e. would decode this step). With nothing
          decoding the full budget is kept, so a solo prompt is unaffected.
        * ``per_request`` splits ``total`` evenly between up to
          ``max_num_partial_prefills`` prefill-phase requests (running chunks
          plus the waiting requests that could be admitted this step), so
          several prompts advance together instead of one at a time. Prompts
          `_short_prefill_bypass` exempts are not counted.
        """
        total = self.max_num_scheduled_tokens
        if self.prefill_chunk_with_decodes > 0 and any(
            not r.is_prefill_chunk for r in self.running
        ):
            total = min(total, self.prefill_chunk_with_decodes)

        per_request = total
        if self.max_num_partial_prefills > 0:
            # Short prompts the bypass exempts do not draw on `total`, so they
            # take no share of it.
            num_running_prefills = sum(
                1
                for r in self.running
                if r.is_prefill_chunk
                and not self._short_prefill_bypass(r.num_tokens - r.num_computed_tokens)
            )
            free_slots = max(
                0,
                self.max_num_active_reqs
                - len(self.running)
                - self.num_waiting_for_streaming_input,
            )
            num_waiting = (
                min(len(self.waiting), free_slots)
                if self._pause_state == PauseState.UNPAUSED
                else 0
            )
            if self.fair_prefill_short_bypass and num_waiting:
                # Before the prefix-cache lookup: a hit only shortens a prompt,
                # so this can count a prompt that ends up short, never the
                # other way round.
                num_waiting = sum(
                    not self._short_prefill_bypass(r.num_tokens - r.num_computed_tokens)
                    for r in itertools.islice(self.waiting, num_waiting)
                )
            num_prefills = min(
                self.max_num_partial_prefills, num_running_prefills + num_waiting
            )
            per_request = total // max(num_prefills, 1)

        # Floor both: `_reserve_prefill_lookahead` refuses to end a chunk within
        # `num_prefill_lookahead` of the prefill end, so a cap at or below that
        # would zero every chunk and stall the prefill forever. Non-binding for
        # any sane setting (the lookahead is a handful of tokens).
        floor = self.num_prefill_lookahead + 1
        return max(total, floor), max(per_request, floor)

    def _short_prefill_bypass(self, num_remaining: int) -> bool:
        """Whether a prefill with ``num_remaining`` tokens left is neither
        capped nor charged to the prefill budget, so a long prompt beside a
        short burst keeps its full share. Read per call (not folded into the
        init flag) so align-mode Mamba, whose chunk sizing derives from the
        prefill limit, is always excluded.
        """
        return (
            self.fair_prefill_short_bypass
            and not self.need_mamba_block_aligned_split
            and num_remaining <= self.prefill_chunk_with_decodes
        )

    # ------------------------------------------------------ PP spreading

    def _pp_decode_cap(self) -> int | None:
        """Decode requests this micro-batch may take under PP decode spreading
        (None: no cap). Counts every running request past its prefill,
        including those whose previous step is still in flight."""
        if not self.pp_spread_decodes:
            return None
        num_decoding = sum(1 for r in self.running if not r.is_prefill_chunk)
        return max(1, cdiv(num_decoding, self.parallel_config.pipeline_parallel_size))

    # ------------------------------------------------------- adaptive k

    def _select_adaptive_k(self) -> tuple[int, int]:
        """This step's verification width and the next drafting width: the
        load width (:meth:`_select_load_width`), then the DSpark depth hint
        and host width where their switches are set."""
        k_verify, k_draft = self._select_load_width()
        if self._depth_hint:
            k_verify, k_draft = dspark_depth_hint.select_step(self, k_verify, k_draft)
        if self._host_width is not None:
            k_verify, k_draft = dspark_host_width.select(
                self, self._host_width, k_verify, k_draft
            )
        return k_verify, k_draft

    def _select_load_width(self) -> tuple[int, int]:
        """Pick this step's verification width and the next drafting width.

        The width follows the load: the number of requests the server holds,
        waiting ones included, picks the ``by_load`` entry, and with
        ``accept`` the step takes the depth within that width that maximises
        the batch's expected tokens per unit step cost. The drafter is asked
        for its full block, or with ``draft_by_load`` for exactly the chosen
        width. What this step can *verify* is additionally capped by the
        drafts already on hand: every request in a uniform decode batch must
        present the same number of drafts, so the batch minimum wins, and
        longer draft lists are truncated here -- before
        ``num_tokens_with_spec`` is read, so every downstream size follows.
        Keeping the two separate is what lets the width climb again: a step
        verifies the drafts it has while asking for more, and gets them on
        the next step.
        """
        assert self.adaptive_k is not None
        running = self.running
        if not running:
            return self.num_spec_tokens, self.num_spec_tokens
        config = self.adaptive_k.config
        # The drafter produces its full block (a DFlash block is one
        # fixed-shape pass), so drafts are never the limit after the first
        # step -- unless draft_by_load narrows the block to the width verified
        # next, trading a one-step lag when load drops for a cheaper drafter
        # under load.
        num_reqs = len(running) + len(self.waiting) + len(self.kv_holding_waiting)
        k_want = load_width = config.k_for_load(num_reqs)
        if config.accept:
            # Acceptance-aware: within the load width, the depth that
            # maximises the batch's expected tokens per unit step cost.
            decoding = [r.request_id for r in running if not r.is_prefill_chunk]
            if decoding:
                k_want = self.adaptive_k.select_by_acceptance(decoding, k_want)
        forced = self._forced_depth()
        if forced is not None and forced <= load_width:
            k_want = forced
        k_draft = k_want if config.draft_by_load else self.num_spec_tokens

        available = min(
            (len(r.spec_token_ids) for r in running if r.spec_token_ids),
            default=k_draft,
        )
        k_verify = k_want if available >= k_want else self.adaptive_k.snap(available)

        for request in running:
            if len(request.spec_token_ids) > k_verify:
                # Rebind rather than mutate: AsyncScheduler hands every
                # request the same placeholder list object.
                request.spec_token_ids = request.spec_token_ids[:k_verify]

        if config.log_interval:
            # The chosen-k histogram exists only for the periodic log line.
            self.adaptive_k.record_choice(k_verify)
            self.adaptive_k.maybe_log()
        return k_verify, k_draft

    def _forced_depth(self) -> int | None:
        """VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE (diagnostics, step-cost
        calibration): a depth written in that file, re-read every 64 steps,
        replaces the chosen one while it is a captured depth."""
        path = getattr(self, "_force_file", None)
        if path is None:
            from vllm import envs as _envs

            path = self._force_file = _envs.VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE
            self._force_value = None
            self._force_countdown = 0
        if not path:
            return None
        self._force_countdown -= 1
        if self._force_countdown <= 0:
            self._force_countdown = 64
            try:
                with open(path) as f:
                    value = int(f.read().strip() or 0)
            except (OSError, ValueError):
                value = 0
            assert self.adaptive_k is not None
            self._force_value = (
                value if value in self.adaptive_k.config.allowed else None
            )
        return self._force_value


class Sm80Scheduler(Sm80SchedulerMixin, Scheduler):
    pass


class Sm80AsyncScheduler(Sm80SchedulerMixin, AsyncScheduler):
    pass
