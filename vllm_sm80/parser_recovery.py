# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provisional tool-call recovery for the streaming parser engine.

A provisional transition opens a tool call the model wrote without its
proper markup (a plain-XML opener, or no opener at all). The engine holds
every event of such a candidate until it can decide:

- the name must pass the parser's validator when the name ends, or, for a
  headless opener (the opener is really the first parameter), some declared
  tool must cover the parameters;
- while its arguments arrive, the parser's shape check must keep accepting
  the held text, so prose that only starts like a call is restored as soon
  as it stops looking like one, not at the end of the output;
- the call must reach a commit transition (its invoke closer). Any other
  transition out of the candidate, an unexpected terminal, or the end of the
  output restores the candidate as the text it was, in the state it came from.

A ``held_only`` transition (a closer only a recovered call may use) is plain
text when no candidate is held. Special tokens the engine drops never enter
the held text. The engine records the tool index of every committed
candidate in ``recovered_tool_indexes``. In ``skip_tool_parsing`` mode (the
reasoning pass of a delegating parser) no candidate is held: tool parsing
belongs to the tool pass. Inside a Markdown code fence of the content (from a
line opening with three or more backticks or tildes to its closing line) no
candidate is held either: an opener there is a quoted example, so it stays
content text.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine_config import ParserState, Transition
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.token_id_scanner import DROP_TERMINAL

# Names are compact; a longer or multi-line "name" is prose quoting an opener.
_NAME_LIMIT = 256


@dataclass(frozen=True, slots=True)
class RecoveryTransition(Transition):
    # Opens a candidate tool call that is held until it is validated.
    provisional: bool = False
    # The opener of a provisional candidate may be its first parameter; the
    # tool is then inferred from the parameters.
    headless: bool = False
    # Commits a held candidate.
    commit: bool = False
    # Applies only while a candidate is held; plain text otherwise.
    held_only: bool = False


# Validates a held candidate's name.
NameValidator = Callable[[str], bool]
# Returns the declared tools whose parameters cover a headless candidate's
# raw text so far.
HeadlessCandidates = Callable[[str], Sequence[str]]
# Returns the tool a complete headless candidate's raw text names, or None
# when it names no tool; a headless candidate commits only with a tool.
HeadlessTool = Callable[[str], str | None]
# Makes the shape check of one held candidate. The check gets the held text
# each time it grows while the arguments arrive; False restores the candidate.
ShapeCheckFactory = Callable[[], Callable[[str], bool]]


class RecoveringParserEngine(StreamingParserEngine):
    """StreamingParserEngine that holds provisional tool-call candidates."""

    name_validator: NameValidator
    headless_candidates: HeadlessCandidates | None
    headless_tool: HeadlessTool | None
    args_terminals: frozenset[str]
    shape_check: ShapeCheckFactory | None
    recovered_tool_indexes: set[int]

    @classmethod
    def adopt(
        cls,
        engine: StreamingParserEngine,
        name_validator: NameValidator,
        headless_candidates: HeadlessCandidates | None = None,
        headless_tool: HeadlessTool | None = None,
        args_terminals: Collection[str] = (),
        shape_check: ShapeCheckFactory | None = None,
    ) -> RecoveringParserEngine:
        """Turn an engine its parser built into a recovering engine.

        ``args_terminals`` are terminals with no transition inside a call's
        arguments (parameter markup) that stay part of a held candidate; any
        other terminal without a transition ends the candidate.
        """
        engine.__class__ = cls
        assert isinstance(engine, RecoveringParserEngine)
        engine.name_validator = name_validator
        engine.headless_candidates = headless_candidates
        engine.headless_tool = headless_tool
        engine.args_terminals = frozenset(args_terminals)
        engine.shape_check = shape_check
        engine.recovered_tool_indexes = set()
        engine._clear_hold()
        engine._clear_fences()
        return engine

    def reset(self, initial_state: ParserState | None = None) -> None:
        super().reset(initial_state)
        self.recovered_tool_indexes = set()
        self._clear_hold()
        self._clear_fences()

    def finish(self) -> list[SemanticEvent]:
        if not self._held:
            return super().finish()
        events = self._process_scanner_items(self._scanner.flush_pending())
        events.extend(self._process_lex_tokens(self._lexer.flush()))
        if self._held:
            events.extend(self._abort_hold())
        self._record_reasoning_tokens(events)
        events.extend(super().finish())
        return events

    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if self._held:
            return self._on_held_terminal(terminal, value, token_count)
        transition = self.config.transitions.get((self.state, terminal))
        if isinstance(transition, RecoveryTransition):
            if transition.held_only:
                return self._emit_for_state(value, token_count)
            if transition.provisional:
                # Only content shows a fence; in the states after a DSML
                # wrapper, text is not emitted, so the opener must still open
                # a candidate there.
                if self._fence is not None and self.state == ParserState.CONTENT:
                    return self._emit_for_state(value, token_count)
                if not self.skip_tool_parsing:
                    return self._begin_hold(transition, value, token_count)
        return super()._on_terminal(terminal, value, token_count)

    def _emit_for_state(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        if not self._held:
            self._track_fences(text)
            return super()._emit_for_state(text, token_count)
        self._hold_raw += text
        self._hold_token_count += token_count
        if self.state == ParserState.TOOL_NAME:
            self._hold_name += text
            if len(self._hold_name) > _NAME_LIMIT or "\n" in self._hold_name:
                return self._abort_hold()
        elif self._hold_shape is not None and not self._hold_shape(self._hold_raw):
            return self._abort_hold()
        self._hold_events.extend(super()._emit_for_state(text, token_count))
        return []

    # ── Code fences ─────────────────────────────────────────────────

    def _clear_fences(self) -> None:
        # The open fence as (character, length of its run), or None.
        self._fence: tuple[str, int] | None = None
        # The current content line while it may still be a fence line, and
        # whether that line has been classified.
        self._fence_line = ""
        self._fence_line_done = False

    def _track_fences(self, text: str) -> None:
        """Follow fence lines through the content text as it is emitted."""
        if self.state != ParserState.CONTENT:
            return
        for i, part in enumerate(text.split("\n")):
            if i:
                if not self._fence_line_done:
                    self._classify_fence_line(line_ended=True)
                self._fence_line, self._fence_line_done = "", False
            if self._fence_line_done or not part:
                continue
            self._fence_line += part
            self._classify_fence_line(line_ended=False)

    def _classify_fence_line(self, line_ended: bool) -> None:
        """Markdown fences: an opening line has a run of three or more
        backticks or tildes after up to three spaces (an info string may
        follow, without a backtick after a backtick run); its closing line has
        a run of the same character at least as long and nothing else. A line
        is classified as soon as it can be: a backtick opener at its end."""
        line = self._fence_line
        head = line.lstrip(" ")
        if len(line) - len(head) > 3 or (head and head[0] not in "`~"):
            self._fence_line_done = True
            return
        if not head:
            return
        char = head[0]
        run = len(head) - len(head.lstrip(char))
        rest = head[run:]
        if not rest and not line_ended:
            return  # the run may go on
        if run < 3:
            self._fence_line_done = True
        elif self._fence is None:
            if char == "`" and not line_ended:
                return  # a backtick later in the line makes it inline code
            if not (char == "`" and "`" in rest):
                self._fence = (char, run)
            self._fence_line_done = True
        elif char != self._fence[0] or run < self._fence[1] or rest.strip():
            self._fence_line_done = True
        elif line_ended:
            self._fence = None
            self._fence_line_done = True

    # ── Held candidate ──────────────────────────────────────────────

    def _clear_hold(self) -> None:
        self._held = False
        self._hold_headless_opener = False
        self._hold_headless = False
        self._hold_shape: Callable[[str], bool] | None = None
        self._hold_events: list[SemanticEvent] = []
        self._hold_raw = ""
        self._hold_token_count = 0
        self._hold_name = ""
        self._hold_prior_state = self.state
        self._hold_prior_tool_index = self.tool_index

    def _begin_hold(
        self,
        transition: RecoveryTransition,
        value: str,
        token_count: int,
    ) -> list[SemanticEvent]:
        prior_state, prior_tool_index = self.state, self.tool_index
        events = self._apply_transition(transition, value, token_count)
        self._held = True
        self._hold_headless_opener = transition.headless
        self._hold_headless = False
        self._hold_shape = self.shape_check() if self.shape_check else None
        self._hold_events = events
        self._hold_raw = value
        self._hold_token_count = token_count
        self._hold_name = ""
        self._hold_prior_state = prior_state
        self._hold_prior_tool_index = prior_tool_index
        return []

    def _on_held_terminal(
        self, terminal: str, value: str, token_count: int
    ) -> list[SemanticEvent]:
        if self._has_drops and terminal == DROP_TERMINAL:
            return []
        transition = self.config.transitions.get((self.state, terminal))
        if transition is None:
            if self.state == ParserState.TOOL_ARGS and terminal in self.args_terminals:
                return self._emit_for_state(value, token_count)
            events = self._abort_hold()
            events.extend(self._on_terminal(terminal, value, token_count))
            return events

        self._hold_raw += value
        self._hold_token_count += token_count
        if self.state == ParserState.TOOL_NAME:
            if self.name_validator(self._hold_name):
                self._hold_events.extend(
                    self._apply_transition(transition, value, token_count)
                )
                return []
            if self._hold_headless_opener and self._headless_candidates():
                return self._continue_headless()
            return self._abort_hold()

        if (
            self.state == ParserState.TOOL_ARGS
            and isinstance(transition, RecoveryTransition)
            and transition.commit
            and (
                not self._hold_headless
                or (self.headless_tool and self.headless_tool(self._hold_raw))
            )
        ):
            events = self._hold_events
            events.extend(self._apply_transition(transition, value, token_count))
            self.recovered_tool_indexes.add(self.tool_index)
            self._clear_hold()
            return events
        return self._abort_hold()

    def _headless_candidates(self) -> Sequence[str]:
        if self.headless_candidates is None:
            return ()
        return self.headless_candidates(self._hold_raw)

    def _continue_headless(self) -> list[SemanticEvent]:
        # The opener and its "name" were the first parameter: drop the name,
        # keep the call open and take the opener text as its arguments.
        self._hold_events = [
            event for event in self._hold_events if event.type != EventType.TOOL_NAME
        ]
        self._hold_headless = True
        self._hold_name = ""
        self.state = ParserState.TOOL_ARGS
        self._reset_args_state()
        self._hold_events.append(
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                value=self._hold_raw,
                tool_index=self.tool_index,
            )
        )
        return []

    def _abort_hold(self) -> list[SemanticEvent]:
        raw, held_tokens = self._hold_raw, self._hold_token_count
        self.state = self._hold_prior_state
        self.tool_index = self._hold_prior_tool_index
        self._reset_args_state()
        self._clear_hold()
        self._track_fences(raw)
        return super()._emit_for_state(raw, held_tokens)
