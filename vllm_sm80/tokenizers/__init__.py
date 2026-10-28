# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's tokenizers, registered under upstream's tokenizer modes.

- ``deepseek_v41``: ``vllm_sm80.tokenizers.deepseek_v41``, upstream's V4.1
  tokenizer declaring that its prompt encoder renders a mid-conversation
  system message in place (``renders_inline_system_messages``), so
  ``/v1/messages`` keeps such messages where they are instead of merging
  them into the leading system prompt.

:func:`register_tokenizers` (plugin registration) points those modes of vLLM's
tokenizer registry at the fork classes. The registry imports a mode's module
on first use, so nothing here is imported until a tokenizer is loaded.
"""

# Tokenizer mode -> (module, class).
TOKENIZERS = {
    "deepseek_v41": (
        "vllm_sm80.tokenizers.deepseek_v41",
        "Sm80DeepseekV41Tokenizer",
    ),
}


def register_tokenizers() -> None:
    """Register :data:`TOKENIZERS` (idempotent), replacing upstream's entry
    for each mode without the registry's overwrite warning."""
    from vllm.tokenizers.registry import TokenizerRegistry

    for mode, (module, class_name) in TOKENIZERS.items():
        TokenizerRegistry.tokenizers.pop(mode, None)
        TokenizerRegistry.register(mode, module, class_name)
