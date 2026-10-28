# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark verify width from each request's last verification (trial).

``VLLM_DSPARK_DEPTH_HINT`` holds two width lists indexed by the number of
requests the server holds (the last entry repeats), for example
``full=7,7,5,5;base=5,5,5,3``:

* ``full``: the width of a step whose every decoding request accepted all the
  drafts of its last verified round, or offers tools (``tool_choice`` not
  ``"none"``) or asks for a JSON ``response_format`` (the chat endpoint marks
  those in the request's sampling ``extra_args``, ``mark``);
* ``base``: every other step, and a request's first decoding step.

A decode batch verifies one width, so a single request without a full round
keeps the step at ``base``. The step's width is capped by the adaptive_k load
width; the drafter drafts that width, or a 5-wide block when the width is
below 5, as the served table drafts 5 and verifies 3.

The plugin (``install``) wraps ``AdaptiveKPolicy.observe``/``forget`` to keep
each request's last round, and ``Scheduler._select_adaptive_k``: the wrapped
method has already truncated every request's drafts to at most the load
width, and this one truncates them further to its own width. It needs
``speculative_config.adaptive_k`` with ``draft_by_load`` and a ``by_load`` at
least as wide as ``full`` at each load that lists every width either list
uses (adaptive_k captures CUDA graphs only for the ``by_load`` widths, and a
step of another width runs eager; an entry past ``max_num_seqs``, as in
``[7, 7, 5, 5, 5, 5, 5, 5, 3]``, captures 3 without serving it at any load),
and ``VLLM_DSPARK_PER_K_WIDTH`` (one drafter query block per width). The first
step refuses a table with an uncaptured width. Empty, nothing changes.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

KEY = "dspark_depth_hint"
DRAFT_FLOOR = 5
_JSON_FORMATS = frozenset({"json_object", "json_schema", "structural_tag"})


@cache
def widths() -> dict[str, tuple[int, ...]] | None:
    raw = os.environ.get("VLLM_DSPARK_DEPTH_HINT", "").strip()
    if not raw:
        return None
    table: dict[str, tuple[int, ...]] = {}
    for part in raw.split(";"):
        name, _, values = part.partition("=")
        name = name.strip()
        if name not in ("full", "base"):
            raise ValueError(f"VLLM_DSPARK_DEPTH_HINT: unknown list {name!r}")
        table[name] = tuple(int(v) for v in values.split(","))
    if set(table) != {"full", "base"}:
        raise ValueError("VLLM_DSPARK_DEPTH_HINT needs both full= and base=")
    return table


def enabled() -> bool:
    return widths() is not None


def is_certain(request: Any) -> bool:
    """A chat request whose output is a tool call or JSON by its own fields."""
    tools = getattr(request, "tools", None)
    if tools and getattr(request, "tool_choice", None) != "none":
        return True
    fmt = getattr(request, "response_format", None)
    return getattr(fmt, "type", None) in _JSON_FORMATS


def mark(request: Any, extra_args: dict[str, Any]) -> None:
    if enabled() and is_certain(request):
        extra_args[KEY] = "full"


def _width(name: str, load: int) -> int:
    table = widths()
    assert table is not None
    values = table[name]
    return values[min(max(load, 1), len(values)) - 1]


def select(
    running: list[Any], full_rounds: dict[str, bool], num_reqs: int, load_width: int
) -> tuple[int, int]:
    """(verify width, draft width) of this step; ``full_rounds``: per request,
    whether its last verified round accepted every draft."""
    decoding = [r for r in running if not r.is_prefill_chunk]
    full = bool(decoding) and all(
        full_rounds.get(r.request_id, False)
        or ((r.sampling_params.extra_args or {}) if r.sampling_params else {}).get(KEY)
        == "full"
        for r in decoding
    )
    verify = min(_width("full" if full else "base", num_reqs), load_width)
    return verify, max(verify, DRAFT_FLOOR)


def install() -> None:
    """Wrap the adaptive-k policy and Scheduler._select_adaptive_k when the
    hint is set (plugin registration; idempotent)."""
    if not enabled():
        return
    from vllm_sm80.spec_decode.adaptive_k import AdaptiveKPolicy
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler._select_adaptive_k
    if getattr(original, "_dspark_depth_hint", False):
        return
    observe, forget = AdaptiveKPolicy.observe, AdaptiveKPolicy.forget

    def _rounds(policy) -> dict[str, bool]:
        rounds = policy.__dict__.get("_dspark_full_rounds")
        if rounds is None:
            rounds = policy.__dict__["_dspark_full_rounds"] = {}
        return rounds

    def _observe(self, req_id: str, num_draft_tokens: int, num_accepted: int):
        # Placeholder drafts (pending padded) carry no acceptance.
        if num_draft_tokens > 0 and not self._pending_padded.get(req_id):
            _rounds(self)[req_id] = num_accepted >= num_draft_tokens
        return observe(self, req_id, num_draft_tokens, num_accepted)

    def _forget(self, req_id: str):
        _rounds(self).pop(req_id, None)
        return forget(self, req_id)

    def _select_adaptive_k(self) -> tuple[int, int]:
        k_verify, k_draft = original(self)
        _check_captured(self.adaptive_k)
        running = self.running
        if not running:
            return k_verify, k_draft
        num_reqs = len(running) + len(self.waiting) + len(self.kv_holding_waiting)
        load_width = self.adaptive_k.config.k_for_load(num_reqs)
        k_want, k_draft = select(
            running, _rounds(self.adaptive_k), num_reqs, load_width
        )
        available = min(
            (len(r.spec_token_ids) for r in running if r.spec_token_ids),
            default=k_draft,
        )
        k_verify = k_want if available >= k_want else self.adaptive_k.snap(available)
        for request in running:
            if len(request.spec_token_ids) > k_verify:
                request.spec_token_ids = request.spec_token_ids[:k_verify]
        return k_verify, k_draft

    _select_adaptive_k._dspark_depth_hint = True  # type: ignore[attr-defined]
    AdaptiveKPolicy.observe = _observe
    AdaptiveKPolicy.forget = _forget
    Scheduler._select_adaptive_k = _select_adaptive_k


def _check_captured(policy) -> None:
    """Refuse (once per policy) a width list with a width adaptive_k has no
    CUDA graphs for: its steps would run eager."""
    if policy.__dict__.get("_dspark_hint_checked"):
        return
    table = widths()
    assert table is not None
    allowed = set(policy.config.allowed)
    missing = sorted({w for v in table.values() for w in v} - allowed)
    if missing:
        raise ValueError(
            f"VLLM_DSPARK_DEPTH_HINT uses widths {missing} that adaptive_k "
            f"captures no CUDA graphs for (by_load widths {sorted(allowed)}); "
            "list them in adaptive_k.by_load, past max_num_seqs if no load "
            "should serve them"
        )
    policy.__dict__["_dspark_hint_checked"] = True

