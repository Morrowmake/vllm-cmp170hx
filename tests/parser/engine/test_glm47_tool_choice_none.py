# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.inputs.engine import tokens_input
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.glm47_moe import TOOL_CALL_START, Glm47MoeParser
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import ChatParams
from vllm.v1.sample.ops.bad_words import apply_bad_words, apply_bad_words_with_drafts

TOOL_ID = 7


def _request(api, **kwargs):
    if api == "chat":
        return ChatCompletionRequest(
            model="glm",
            messages=[],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            **kwargs,
        )
    if isinstance(kwargs.get("tool_choice"), dict):
        kwargs["tool_choice"] = {"type": "function", "name": "lookup"}
    return ResponsesRequest(
        input="answer", tools=[{"type": "function", "name": "lookup"}], **kwargs
    )


def _parser():
    tokenizer = make_mock_tokenizer({TOOL_CALL_START: TOOL_ID})
    tokenizer.max_token_id = 11
    tokenizer.encode.side_effect = lambda text, **kwargs: (
        [TOOL_ID] if text.lstrip() == TOOL_CALL_START else [3]
    )
    return Glm47MoeParser(tokenizer)


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "choice",
    [
        "none",
        "auto",
        "required",
        {"type": "function", "function": {"name": "lookup"}},
    ],
)
def test_only_none_gets_decode_mask(api, choice, monkeypatch):
    monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", "1")
    request = _request(api, tool_choice=choice, bad_words=["existing"])
    parser = _parser()
    assert parser.adjust_request(request) is request
    parser.adjust_request(request)
    assert request.bad_words == (
        ["existing", TOOL_CALL_START] if choice == "none" else ["existing"]
    )
    assert request.skip_special_tokens is False


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("flag", [None, "0"])
def test_off_by_default_adjusts_like_upstream(api, flag, monkeypatch):
    """Unset or 0: no decode mask, and the request gets exactly the upstream
    ParserEngine adjustment (special tokens kept for the parser)."""
    if flag is None:
        monkeypatch.delenv("VLLM_TOOL_CHOICE_NONE_MASK", raising=False)
    else:
        monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", flag)
    request = _request(api, tool_choice="none", bad_words=["existing"])
    upstream = _request(api, tool_choice="none", bad_words=["existing"])
    ParserEngine.adjust_request(_parser(), upstream)
    assert _parser().adjust_request(request) is request
    assert request.bad_words == upstream.bad_words == ["existing"]
    assert request.skip_special_tokens is upstream.skip_special_tokens is False


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_mask_reaches_regular_and_draft_sampling(api, monkeypatch):
    monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", "1")
    parser = _parser()
    request = _request(api, tool_choice="none")
    parser.adjust_request(request)
    params = request.to_sampling_params(8, {})
    params.update_from_tokenizer(parser.model_tokenizer)
    assert params.bad_words_token_ids == [[TOOL_ID]]
    # A large positive logit must still lose to an ordinary answer token.
    for drafted in (False, True):
        logits = torch.zeros((3, 12))
        logits[:, TOOL_ID] = 1000
        logits[:, 5] = 1
        if drafted:
            apply_bad_words_with_drafts(
                logits, {0: params.bad_words_token_ids}, [[], [5], [5, 5]], [3]
            )
        else:
            apply_bad_words(logits, {0: params.bad_words_token_ids}, [[]] * 3)
            logits = logits[:1]
        assert torch.isneginf(logits[:, TOOL_ID]).all()
        assert logits.argmax(dim=-1).tolist() == [5] * logits.shape[0]


class PlainParser(Parser):
    def adjust_request(self, request):
        raise AssertionError("tool_choice=none should skip this parser")


class MaskOnlyParser(PlainParser):
    adjust_request_when_tool_choice_none = True


def _preprocess(api, parser_cls, monkeypatch, reuse=False):
    request = _request(api, tool_choice="none")
    monkeypatch.setattr(type(request), "build_tok_params", lambda *a: MagicMock())
    monkeypatch.setattr(
        type(request), "build_chat_params", lambda *a, **kw: ChatParams()
    )
    tokenizer = _parser().model_tokenizer
    renderer = object.__new__(OnlineRenderer)
    renderer.trust_request_mm_kwargs = False
    renderer.reasoning_effort_map = {}
    renderer.model_config = SimpleNamespace(
        multimodal_config=None, enable_prompt_embeds=False, is_encoder_decoder=False
    )
    renderer.renderer = MagicMock()
    renderer.renderer.tokenizer = tokenizer
    renderer.renderer.get_tokenizer.return_value = tokenizer
    renderer.renderer.render_chat_async = AsyncMock(
        return_value=([[]], [tokens_input([1])])
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    if reuse:
        request.kv_transfer_params = {"prompt_token_ids": [1, 2, 3]}
    asyncio.run(
        renderer.preprocess_chat(request, [], None, "string", {}, tools, parser_cls)
    )
    return request, renderer, tools


@pytest.mark.parametrize("parser_cls", [Glm47MoeParser, PlainParser])
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("reuse", [False, True])
def test_renderer_adjusts_none_without_removing_tools(
    api, reuse, parser_cls, monkeypatch
):
    monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", "1")
    request, renderer, tools = _preprocess(api, parser_cls, monkeypatch, reuse)
    assert request.bad_words == (
        [TOOL_CALL_START] if parser_cls is Glm47MoeParser else []
    )
    if not reuse:
        chat_params = renderer.renderer.render_chat_async.call_args.args[1]
        assert chat_params.chat_template_kwargs["tools"] == tools


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("flag", [None, "0"])
def test_renderer_off_calls_adjust_only_where_upstream_does(api, flag, monkeypatch):
    """Off, a parser that asks for tool_choice="none" adjustment only for the
    mask is not called, as upstream does not call it."""
    if flag is None:
        monkeypatch.delenv("VLLM_TOOL_CHOICE_NONE_MASK", raising=False)
    else:
        monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", flag)
    request, _, _ = _preprocess(api, MaskOnlyParser, monkeypatch)
    assert request.bad_words == []


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_startup_banner(enabled, monkeypatch):
    monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", enabled)
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.ParserManager.get_parser",
        lambda **kwargs: Glm47MoeParser,
    )
    banner = MagicMock()
    monkeypatch.setattr("vllm.renderers.online_renderer.logger.info_once", banner)
    OnlineRenderer(
        SimpleNamespace(hf_config=SimpleNamespace(model_type="glm5next"), model="glm"),
        MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
    )
    assert banner.call_args.args[1] == ("enabled" if enabled == "1" else "disabled")


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("configuration", ["tool", "reasoning", "both"])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_managed_parser_and_startup_banner(api, configuration, enabled, monkeypatch):
    monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", enabled)
    tool = "glm47" if configuration in ("tool", "both") else None
    reasoning = "glm47" if configuration in ("reasoning", "both") else None
    banner = MagicMock()
    monkeypatch.setattr("vllm.renderers.online_renderer.logger.info_once", banner)
    tokenizer = _parser().model_tokenizer
    renderer = OnlineRenderer(
        SimpleNamespace(hf_config=SimpleNamespace(model_type="glm5next"), model="glm"),
        MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
        tool_parser=tool,
        reasoning_parser=reasoning,
        enable_auto_tools=True,
    )
    assert renderer.parser.adjust_request_when_tool_choice_none
    assert banner.call_args.args[1] == ("enabled" if enabled == "1" else "disabled")
    renderer.model_config.multimodal_config = None
    renderer.model_config.enable_prompt_embeds = False
    renderer.model_config.is_encoder_decoder = False
    renderer.renderer.tokenizer = tokenizer
    renderer.renderer.get_tokenizer.return_value = tokenizer
    renderer.renderer.render_chat_async = AsyncMock(
        return_value=([[]], [tokens_input([1])])
    )
    request = _request(api, tool_choice="none", bad_words=["existing"])
    monkeypatch.setattr(type(request), "build_tok_params", lambda *a: MagicMock())
    monkeypatch.setattr(
        type(request), "build_chat_params", lambda *a, **kw: ChatParams()
    )
    for _ in range(2):
        asyncio.run(
            renderer.preprocess_chat(
                request, [], None, "string", {}, None, renderer.parser
            )
        )
    assert request.bad_words == (
        ["existing", TOOL_CALL_START] if enabled == "1" else ["existing"]
    )


def test_non_glm_managed_parser_retains_none_opt_out():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="qwen3_xml", enable_auto_tools=True
    )
    assert not parser_cls.adjust_request_when_tool_choice_none


def test_real_logger_emits_managed_startup_banner(monkeypatch, caplog):
    import logging

    from vllm.logger import _print_info_once
    from vllm.renderers.online_renderer import logger

    _print_info_once.cache_clear()
    monkeypatch.setenv("VLLM_TOOL_CHOICE_NONE_MASK", "1")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    caplog.set_level(logging.INFO, logger=logger.name)
    OnlineRenderer(
        SimpleNamespace(hf_config=SimpleNamespace(model_type="glm5next"), model="glm"),
        MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
        tool_parser="glm47",
        reasoning_parser="glm47",
        enable_auto_tools=True,
    )
    assert any(
        record.getMessage() == "GLM tool_choice=none decode mask: enabled "
        "(VLLM_TOOL_CHOICE_NONE_MASK)"
        for record in caplog.records
    )
