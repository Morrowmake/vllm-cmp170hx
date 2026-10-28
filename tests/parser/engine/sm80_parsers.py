# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's parser registrations (vllm_sm80.parsers) for one test module,
with vLLM's parser registries restored afterwards so upstream tests in the
same process keep upstream's parsers."""

import pytest

from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm_sm80.parsers import register_parsers

_REGISTRIES = (
    (ToolParserManager, "tool_parsers"),
    (ReasoningParserManager, "reasoning_parsers"),
)


@pytest.fixture(autouse=True, scope="module")
def fork_parsers():
    saved = [
        (manager, attr, dict(getattr(manager, attr)), dict(manager.lazy_parsers))
        for manager, attr in _REGISTRIES
    ]
    register_parsers()
    yield
    for manager, attr, eager, lazy in saved:
        getattr(manager, attr).clear()
        getattr(manager, attr).update(eager)
        manager.lazy_parsers.clear()
        manager.lazy_parsers.update(lazy)
