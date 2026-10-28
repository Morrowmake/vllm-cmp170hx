# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refusal: the launcher will not resolve or start this launch."""

from __future__ import annotations

EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_CHECKPOINT_INCOMPLETE = 4
EXIT_CHECKPOINT_MISMATCH = 5
EXIT_CHECKPOINT_ABSENT = 6


class Refusal(Exception):
    """One or more reasons the launch is refused, each a plain sentence."""

    def __init__(self, *reasons: str, code: int = EXIT_REFUSED) -> None:
        self.reasons = [r for r in reasons if r]
        self.code = code
        super().__init__("; ".join(self.reasons))
