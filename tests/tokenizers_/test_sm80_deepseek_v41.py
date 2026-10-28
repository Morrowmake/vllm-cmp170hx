# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek V4.1 tokenizer the plugin registers, and the V4.1 prompt
encoding of tool-call turns: drop_thinking drops the reasoning of earlier
turns only, and a user message holding only tool results does not start a
new turn."""

import copy
import pickle
from types import SimpleNamespace

from tests.tokenizers_.test_deepseek_v41 import render, tool_call


class _FakeHfTokenizer:
    all_special_ids = [0]
    all_special_tokens = ["<s>"]

    def get_vocab(self):
        return {"<s>": 0, "a": 1}

    def get_added_vocab(self):
        return {}

    def __len__(self):
        return 2


def test_registered_tokenizer_declares_inline_system_messages(monkeypatch):
    import vllm_sm80.tokenizers.deepseek_v41 as plugin_tokenizer
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm_sm80.tokenizers import register_tokenizers

    monkeypatch.setattr(
        TokenizerRegistry, "tokenizers", dict(TokenizerRegistry.tokenizers)
    )
    register_tokenizers()
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v41") is (
        plugin_tokenizer.Sm80DeepseekV41Tokenizer
    )

    monkeypatch.setattr(
        plugin_tokenizer,
        "TokenizersBackend",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeHfTokenizer()),
    )
    tokenizer = TokenizerRegistry.load_tokenizer("deepseek_v41", "model")
    assert tokenizer.renders_inline_system_messages is True
    # The wrappers rebuild themselves on copy and pickle; the flag survives.
    assert copy.copy(tokenizer).renders_inline_system_messages is True
    assert pickle.loads(pickle.dumps(tokenizer)).renders_inline_system_messages
    # Upstream's V4.1 prompt encoder still renders the prompt.
    assert "<｜User｜>q" in tokenizer.apply_chat_template(
        [{"role": "user", "content": "q"}], tokenize=False
    )


def test_drop_thinking_keeps_current_turn_across_tool_results():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = render(messages)
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")

    next_turn = render(messages + [{"role": "user", "content": "q3"}])
    assert "r2" not in next_turn
    assert next_turn.endswith("q3<｜Assistant｜><think>")


def test_tool_loop_keeps_every_round_of_the_current_turn():
    def round_(n):
        return [
            {
                "role": "assistant",
                "reasoning_content": f"r{n}",
                "tool_calls": [tool_call(f"c{n}")],
            },
            {"role": "tool", "tool_call_id": f"c{n}", "content": f"v{n}"},
        ]

    messages = [{"role": "user", "content": "q"}, *round_(1), *round_(2)]
    prompt = render(messages)
    assert "<｜User｜>q<｜Assistant｜><think>r1</think>" in prompt
    assert "<tool_result>v1</tool_result><｜Assistant｜><think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>v2</tool_result><｜Assistant｜><think>")
    # Declared tools turn drop_thinking off; the turns render the same.
    schema = {"type": "object", "properties": {}}
    tools = [
        {"type": "function", "function": {"name": f"c{n}", "parameters": schema}}
        for n in (1, 2)
    ]
    assert render(messages, tools=tools).endswith(prompt[prompt.index("<｜User｜>q") :])
