# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ling3 parser with the ``tool_choice="none"`` decode mask of the GLM-4.7
parser it extends (``vllm_sm80.parsers.glm47``, VLLM_TOOL_CHOICE_NONE_MASK).
Registered as ``ling3`` (``vllm_sm80.parsers``)."""

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.ling3 import Ling3Parser
from vllm.reasoning.ling3_reasoning_parser import Ling3ParserReasoningAdapter
from vllm.tool_parsers.ling3_tool_parser import Ling3ToolParser
from vllm_sm80.parsers.glm47 import ToolChoiceNoneMask


class Sm80Ling3Parser(ToolChoiceNoneMask, Ling3Parser):
    pass


# The engine's own adapter attributes, as upstream sets them for its parser.
make_adapters(Sm80Ling3Parser)


class Sm80Ling3ReasoningParser(Ling3ParserReasoningAdapter):  # type: ignore[valid-type, misc]
    _parser_engine_cls = Sm80Ling3Parser


class Sm80Ling3ToolParser(Ling3ToolParser):
    _parser_engine_cls = Sm80Ling3Parser
