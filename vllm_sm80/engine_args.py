# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's served engine flags.

``vllm/engine/arg_utils.py`` calls :func:`add_cli_args` at the end of
``EngineArgs.add_cli_args``, which adds ``--prefill-chunk-with-decodes`` and
``--max-num-partial-prefills`` (fair chunked prefill), and
:func:`from_cli_args` at the end of ``EngineArgs.from_cli_args``, which moves
those flags and the ``adaptive_k`` key of ``--speculative-config`` into
``additional_config["sm80"]`` (``vllm_sm80.config``), where the plugin reads
them.
"""

import argparse
from typing import Any

from vllm.utils.argparse_utils import human_readable_int

from vllm_sm80.config import FAIR_PREFILL_KEYS, with_settings

PREFILL_CHUNK_WITH_DECODES_HELP = """\
Decode-aware prefill chunk budget ("fair chunked prefill").

When > 0 and at least one running request is in the decode phase, the total
number of *prefill* tokens the scheduler issues in one step is capped at this
value. Decode tokens are budgeted separately (against
`max_num_batched_tokens`) and are never capped by this, so decode streams get
a scheduler step roughly every `prefill_chunk_with_decodes` tokens of prefill
work instead of every `max_num_batched_tokens` tokens. When no request is
decoding the full `max_num_batched_tokens` budget is used, so a prompt
arriving into an idle engine still prefills at full chunk size.

Unlike `long_prefill_token_threshold`, which caps each request's chunk
individually when requests compete, this is a single budget shared by all
prefill-phase requests in the step, and it only binds while something is
decoding.

0 (the default) disables the cap and reproduces upstream scheduling exactly.
Only takes effect when chunked prefill is enabled."""

MAX_NUM_PARTIAL_PREFILLS_HELP = """\
Round-robin fairness between concurrent chunked prefills.

When > 0, the per-step prefill token budget (`prefill_chunk_with_decodes` when
that is active, otherwise `max_num_batched_tokens`) is divided evenly between
up to this many prefill-phase requests, so a second prompt starts making
progress on the step it is admitted rather than waiting for the first
prompt's prefill to finish. Requests beyond that many wait for a later step in
arrival order, which bounds how thinly the budget is sliced.

Note this trades mean TTFT for worst-case TTFT: with two equal prompts,
first-come-first-served finishes one at T/2 and the other at T, while an even
split finishes both at ~T. It pays off when prompt lengths differ a lot (a
short prompt no longer waits behind a 30K one).

0 (the default) disables the split and reproduces upstream scheduling exactly.
Only takes effect when chunked prefill is enabled."""


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the fair chunked prefill flags."""
    group = parser.add_argument_group(
        "Sm80SchedulerConfig", "Fair chunked prefill (the sm80 fork scheduler)."
    )
    group.add_argument(
        "--prefill-chunk-with-decodes",
        type=human_readable_int,
        default=0,
        help=PREFILL_CHUNK_WITH_DECODES_HELP + f"\n\n{human_readable_int.__doc__}",
    )
    group.add_argument(
        "--max-num-partial-prefills",
        type=int,
        default=0,
        help=MAX_NUM_PARTIAL_PREFILLS_HELP,
    )


def from_cli_args(engine_args: Any, args: argparse.Namespace) -> None:
    """Move the fork's settings from the parsed flags into
    ``engine_args.additional_config``."""
    settings: dict[str, Any] = {}
    for key in FAIR_PREFILL_KEYS:
        value = getattr(args, key, 0)
        if value:
            settings[key] = value
    if settings:
        engine_args.additional_config = with_settings(
            engine_args.additional_config, settings
        )
