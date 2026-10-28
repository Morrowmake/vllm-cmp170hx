# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 parser with damaged-call recovery
(``vllm_sm80.deepseek_v41_recovery``), registered as ``deepseek_v41``
(``vllm_sm80.parsers``)."""

from vllm.parser.deepseek_v41 import DeepSeekV41Parser as UpstreamDeepSeekV41Parser
from vllm.parser.engine.adapters import make_adapters
from vllm.reasoning.deepseek_v41_engine_reasoning_parser import (
    DeepSeekV41ParserReasoningAdapter,
)
from vllm.tool_parsers.deepseekv41_engine_tool_parser import (
    DeepSeekV41EngineToolParser,
)
from vllm_sm80.deepseek_v41_recovery import recovering_parser

DeepSeekV41Parser = recovering_parser(UpstreamDeepSeekV41Parser)

# The engine's own adapter attributes, as upstream sets them for its parser.
make_adapters(DeepSeekV41Parser)


class Sm80DeepSeekV41ReasoningParser(DeepSeekV41ParserReasoningAdapter):  # type: ignore[valid-type, misc]
    _parser_engine_cls = DeepSeekV41Parser


class Sm80DeepSeekV41ToolParser(DeepSeekV41EngineToolParser):  # type: ignore[valid-type, misc]
    _parser_engine_cls = DeepSeekV41Parser
