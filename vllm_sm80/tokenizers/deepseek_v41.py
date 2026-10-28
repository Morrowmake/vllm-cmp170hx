# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream's DeepSeek V4.1 tokenizer, declaring that its prompt encoder
renders a mid-conversation system message in place. The Anthropic
``/v1/messages`` serving reads ``renders_inline_system_messages`` and then
keeps inline system messages where they are."""

from transformers import TokenizersBackend

from vllm.tokenizers.deepseek_v41 import (
    DeepseekV41Tokenizer,
    get_deepseek_v41_tokenizer,
)
from vllm.tokenizers.hf import HfTokenizer, get_cached_tokenizer


class Sm80DeepseekV41Tokenizer(DeepseekV41Tokenizer):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        # Upstream's from_pretrained, with the attribute set on the HF
        # tokenizer both wrappers copy, so copies and pickles keep it (the
        # wrappers rebuild themselves from that tokenizer).
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        tokenizer.renders_inline_system_messages = True
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
