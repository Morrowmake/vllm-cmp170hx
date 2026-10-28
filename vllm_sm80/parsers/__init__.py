# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's reasoning and tool parsers, registered under upstream's names.

- ``glm45`` / ``glm47``: ``vllm_sm80.parsers.glm47``, upstream's GLM-4.7
  parser plus the ``tool_choice="none"`` decode mask
  (``VLLM_TOOL_CHOICE_NONE_MASK``).
- ``ling3``: ``vllm_sm80.parsers.ling3``, upstream's Ling3 parser (built on
  the GLM-4.7 parser) plus the same decode mask.
- ``deepseek_v41``: ``vllm_sm80.parsers.deepseek_v41``, upstream's V4.1 parser
  with damaged-call recovery (``vllm_sm80.deepseek_v41_recovery``).

:func:`register_parsers` (plugin registration) points those names of vLLM's
tool and reasoning parser registries at the fork classes. Both registries
load a name's module on first use, so nothing here is imported until a
server resolves one of these parsers.
"""

# Registry name -> (module, tool parser class, reasoning parser class).
PARSERS = {
    "glm45": (
        "vllm_sm80.parsers.glm47",
        "Sm80Glm47MoeToolParser",
        "Sm80Glm47MoeReasoningParser",
    ),
    "glm47": (
        "vllm_sm80.parsers.glm47",
        "Sm80Glm47MoeToolParser",
        "Sm80Glm47MoeReasoningParser",
    ),
    "ling3": (
        "vllm_sm80.parsers.ling3",
        "Sm80Ling3ToolParser",
        "Sm80Ling3ReasoningParser",
    ),
    "deepseek_v41": (
        "vllm_sm80.parsers.deepseek_v41",
        "Sm80DeepSeekV41ToolParser",
        "Sm80DeepSeekV41ReasoningParser",
    ),
}


def register_parsers() -> None:
    """Register :data:`PARSERS` lazily (idempotent). Importing the registries
    first runs their own lazy registration, which these entries replace; a
    class a registry already loaded for one of the names is dropped, since a
    loaded class wins over a lazy entry."""
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    for name, (module, tool_cls, reasoning_cls) in PARSERS.items():
        ToolParserManager.tool_parsers.pop(name, None)
        ReasoningParserManager.reasoning_parsers.pop(name, None)
        ToolParserManager.register_lazy_module(name, module, tool_cls)
        ReasoningParserManager.register_lazy_module(name, module, reasoning_cls)


def wants_tool_choice_none_mask(parser) -> bool:
    """Whether ``parser`` (a composed parser or a parser engine class), or the
    engine behind one of its adapters, applies the decode mask."""
    classes = (
        parser,
        *(
            getattr(cls, "_parser_engine_cls", None)
            for cls in (parser.reasoning_parser_cls, parser.tool_parser_cls)
        ),
    )
    return any(
        getattr(cls, "adjust_request_when_tool_choice_none", False) for cls in classes
    )


def adjust_on_tool_choice_none(parser) -> bool:
    """Whether the renderer must adjust a ``tool_choice="none"`` request for
    ``parser``: it applies the decode mask and ``VLLM_TOOL_CHOICE_NONE_MASK``
    is on (``OnlineRenderer``'s one call)."""
    if not wants_tool_choice_none_mask(parser):
        return False
    import vllm.envs as envs

    return bool(envs.VLLM_TOOL_CHOICE_NONE_MASK)
