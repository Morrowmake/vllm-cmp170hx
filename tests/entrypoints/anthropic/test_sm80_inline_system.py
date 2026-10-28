# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline system messages in /v1/messages for a renderer with its own prompt
encoder: kept in place only when the tokenizer declares that its encoder
renders them (``renders_inline_system_messages``), as the DeepSeek V4.1
tokenizer the plugin registers does
(tests/tokenizers_/test_sm80_deepseek_v41.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.entrypoints.anthropic.test_anthropic_messages_conversion import (
    PERMISSIVE_TEMPLATE,
    Q35_TEMPLATE,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages


@pytest.mark.parametrize(
    ("declared", "template", "expected"),
    [
        (True, None, False),  # DeepSeek-V4.1: rendered in place
        (False, None, True),  # an encoder that does not declare it
        (True, PERMISSIVE_TEMPLATE, False),  # a CLI template is probed
        (True, Q35_TEMPLATE, True),
    ],
)
def test_own_encoder_merges_unless_it_declares_inline_system(
    monkeypatch, declared, template, expected
):
    """The encoder is never called to find out."""
    import vllm.entrypoints.anthropic.serving as serving_mod

    mock_logger = MagicMock()
    monkeypatch.setattr(serving_mod, "logger", mock_logger)
    tokenizer = SimpleNamespace(
        apply_chat_template=MagicMock(side_effect=AssertionError("not probed"))
    )
    if declared:
        tokenizer.renders_inline_system_messages = True
    online_renderer = SimpleNamespace(
        renderer=SimpleNamespace(tokenizer=tokenizer),
        chat_template=template,
        model_config=None,
    )
    merge = AnthropicServingMessages._should_merge_inline_system(online_renderer)
    assert merge is expected
    assert mock_logger.warning_once.called is expected
