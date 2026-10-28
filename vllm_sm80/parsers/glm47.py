# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser with the ``tool_choice="none"`` decode mask.

``VLLM_TOOL_CHOICE_NONE_MASK`` (off by default; the GLM launcher sets 1):
with ``tool_choice="none"`` the tool-call start token is a bad word, so the
model cannot open a tool call the request does not allow. Off, a request is
adjusted exactly as upstream adjusts it. Registered as ``glm45`` / ``glm47``
(``vllm_sm80.parsers``); :class:`ToolChoiceNoneMask` also serves upstream's
parsers built on the GLM-4.7 parser (``vllm_sm80.parsers.ling3``).
"""

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.glm47_moe import TOOL_CALL_START, Glm47MoeParser
from vllm.reasoning.glm47_moe_reasoning_parser import Glm47MoeParserReasoningAdapter
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

from vllm import envs

logger = init_logger(__name__)


class ToolChoiceNoneMask:
    """The decode mask over a GLM-4.7-format parser engine (listed before it)."""

    # Read by vllm_sm80.parsers.adjust_on_tool_choice_none.
    adjust_request_when_tool_choice_none = True

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)  # type: ignore[misc]
        if (
            request.tool_choice == "none"
            and envs.VLLM_TOOL_CHOICE_NONE_MASK
            and TOOL_CALL_START not in request.bad_words
        ):
            request.bad_words = [*request.bad_words, TOOL_CALL_START]
        return request


class Sm80Glm47MoeParser(ToolChoiceNoneMask, Glm47MoeParser):
    pass


# The engine's own adapter attributes, as upstream sets them for its parser.
make_adapters(Sm80Glm47MoeParser)


class Sm80Glm47MoeReasoningParser(Glm47MoeParserReasoningAdapter):  # type: ignore[valid-type, misc]
    _parser_engine_cls = Sm80Glm47MoeParser


class Sm80Glm47MoeToolParser(Glm47MoeModelToolParser):
    _parser_engine_cls = Sm80Glm47MoeParser


def log_mask_state() -> None:
    """The startup line naming the mask's state (once per process; logged
    when a server first resolves a GLM or Ling3 parser, which imports this
    module)."""
    logger.info_once(
        "GLM tool_choice=none decode mask: %s (VLLM_TOOL_CHOICE_NONE_MASK)",
        "enabled" if envs.VLLM_TOOL_CHOICE_NONE_MASK else "disabled",
    )


log_mask_state()
