# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool arguments pass through the V4.1 parser exactly as the model wrote them.

A client may validate arguments by presence: a shell tool may reject a call
that carries ``justification`` without ``sandbox_permissions``. The parser
must neither drop a parameter the model wrote nor add one it did not, and must
not alter a value (a trailing ``;``, an entity, a ``<``), whether the call is
parsed whole or streamed at any chunking, and whether it is a DSML call or a
recovered one.
"""

import json

import pytest

from tests.parser.engine.test_deepseek_v41 import (
    _stream_calls,
    parser_for,
    tokenizer_for,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_EXEC_PARAMETERS = {
    "type": "object",
    "properties": {
        "cmd": {"type": "string", "description": "Shell command to execute."},
        "justification": {
            "type": "string",
            "description": "User-facing approval question for "
            "`require_escalated`; omit otherwise.",
        },
        "login": {"type": "boolean"},
        "max_output_tokens": {"type": "number"},
        "prefix_rule": {"type": "array", "items": {"type": "string"}},
        "sandbox_permissions": {
            "type": "string",
            "description": "Per-command sandbox override. Defaults to "
            "`use_default`; use `require_escalated` for unsandboxed execution.",
            "enum": ["use_default", "require_escalated"],
        },
        "shell": {"type": "string"},
        "tty": {"type": "boolean"},
        "workdir": {"type": "string"},
        "yield_time_ms": {"type": "number"},
    },
    "required": ["cmd"],
    "additionalProperties": False,
}
_EXEC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Runs a command in a PTY.",
            "parameters": _EXEC_PARAMETERS,
        },
    }
]
_FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]
_CHUNKS = [None, 1, 2, 3, 7, 10000]


def _param(name, value, string="true"):
    attr = "" if string is None else f' string="{string}"'
    return f'<｜DSML｜ parameter name="{name}"{attr}>{value}</｜DSML｜ parameter>\n'


def _dsml(tool, params):
    return (
        f'<｜DSML｜ calls>\n<｜DSML｜ invoke name="{tool}">\n'
        + "".join(params)
        + "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )


def _calls(text, tools, chunk_size, thinking=False):
    """``[(name, arguments dict)]`` of a model output, whole or streamed."""
    if thinking:
        text = "Plan.</think>" + text
    if chunk_size is None:
        tokenizer, tokens = tokenizer_for(text, False)
        parser = parser_for(tokenizer, {"thinking": thinking})
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
        )
        _, content, calls = parser.parse(
            text,
            request,
            enable_auto_tools=True,
            model_output_token_ids=[tid for tid, _ in tokens],
        )
        pairs = [(call.name, call.arguments) for call in calls or []]
    else:
        assert not thinking
        _, output = _stream_calls(text, tools, chunk_size)
        content = output.content
        pairs = [(call["name"], call["arguments"]) for call in output.tool_calls]
    assert not (content or "").strip()
    return [(name, json.loads(args)) for name, args in pairs]


# The parameters as the model writes a sandboxed command: every declared one,
# in schema order, ``justification`` empty.
_FULL_EXEC = [
    _param("cmd", "pwd && ls -la && cat counter.py"),
    _param("justification", ""),
    _param("login", "false", "false"),
    _param("max_output_tokens", "10000", "false"),
    _param("prefix_rule", "[]", "false"),
    _param("sandbox_permissions", "use_default"),
    _param("shell", "zsh"),
    _param("tty", "false", "false"),
    _param("workdir", "/srv/work"),
    _param("yield_time_ms", "10000", "false"),
]
_FULL_EXEC_ARGS = {
    "cmd": "pwd && ls -la && cat counter.py",
    "justification": "",
    "login": False,
    "max_output_tokens": 10000,
    "prefix_rule": [],
    "sandbox_permissions": "use_default",
    "shell": "zsh",
    "tty": False,
    "workdir": "/srv/work",
    "yield_time_ms": 10000,
}


@pytest.mark.parametrize("chunk_size", _CHUNKS)
def test_every_written_parameter_is_kept_in_order(chunk_size):
    calls = _calls(_dsml("exec_command", _FULL_EXEC), _EXEC_TOOLS, chunk_size)
    assert calls == [("exec_command", _FULL_EXEC_ARGS)]
    assert list(calls[0][1]) == list(_FULL_EXEC_ARGS)


def test_every_written_parameter_is_kept_after_thinking():
    calls = _calls(_dsml("exec_command", _FULL_EXEC), _EXEC_TOOLS, None, True)
    assert calls == [("exec_command", _FULL_EXEC_ARGS)]


@pytest.mark.parametrize("chunk_size", _CHUNKS)
def test_no_parameter_is_added(chunk_size):
    """A call the model wrote with ``cmd`` and an empty ``justification`` only
    comes out with exactly those two."""
    text = _dsml("exec_command", [_param("cmd", "ls"), _param("justification", "")])
    assert _calls(text, _EXEC_TOOLS, chunk_size) == [
        ("exec_command", {"cmd": "ls", "justification": ""})
    ]


@pytest.mark.parametrize(
    ("written", "string"),
    [
        ("use_default", "true"),
        ('"use_default"', "false"),
        ("use_default", "false"),
        ("use_default", None),
    ],
)
@pytest.mark.parametrize("chunk_size", _CHUNKS)
def test_enum_parameter_survives_every_typing(written, string, chunk_size):
    text = _dsml(
        "exec_command",
        [
            _param("cmd", "ls"),
            _param("justification", ""),
            _param("sandbox_permissions", written, string),
        ],
    )
    assert _calls(text, _EXEC_TOOLS, chunk_size) == [
        (
            "exec_command",
            {"cmd": "ls", "justification": "", "sandbox_permissions": "use_default"},
        )
    ]


_VALUES = [
    "/srv/app/main.py",
    "/srv/app;",
    "cd /srv/app; ls",
    "cd /srv/app && ls",
    "/srv/a&amp;b",
    "/srv/x&#59;",
    "echo &quot;x&quot; &lt;y&gt;",
    "/srv/<dir>/f",
    "a < b",
    "x <",
    "trailing<",
    "</",
    "</｜DSML｜ par",
    "C:\\Users\\新建文件夹\\a.txt",
    " lead and trail ",
]


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("chunk_size", _CHUNKS)
def test_string_values_are_byte_exact(value, chunk_size):
    text = _dsml("exec_command", [_param("cmd", value), _param("workdir", value)])
    assert _calls(text, _EXEC_TOOLS, chunk_size) == [
        ("exec_command", {"cmd": value, "workdir": value})
    ]


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("chunk_size", _CHUNKS)
def test_file_path_values_are_byte_exact(value, chunk_size):
    text = (
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="Read">\n'
        + _param("file_path", value)
        + "</｜DSML｜ invoke>\n"
        '<｜DSML｜ invoke name="Bash">\n'
        + _param("command", f"ls {value}")
        + _param("description", "List")
        + "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    assert _calls(text, _FILE_TOOLS, chunk_size) == [
        ("Read", {"file_path": value}),
        ("Bash", {"command": f"ls {value}", "description": "List"}),
    ]


def _xml_call(params):
    return (
        'Running:\n<invoke name="exec_command">\n'
        + "".join(
            f'<parameter name="{name}">{value}</parameter>\n' for name, value in params
        )
        + "</invoke>"
    )


@pytest.mark.parametrize("chunk_size", _CHUNKS)
def test_recovered_call_keeps_every_parameter(chunk_size):
    params = [
        ("cmd", "cat a.txt; echo done > out.txt"),
        ("justification", ""),
        ("max_output_tokens", "10000"),
        ("sandbox_permissions", "use_default"),
        ("workdir", "/srv/app;"),
    ]
    text = _xml_call(params)
    if chunk_size is None:
        tokenizer, tokens = tokenizer_for(text, False)
        parser = parser_for(tokenizer, {"thinking": False})
        _, content, calls = parser.parse(
            text,
            ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                tools=_EXEC_TOOLS,
            ),
            enable_auto_tools=True,
            model_output_token_ids=[tid for tid, _ in tokens],
        )
        calls = [(call.name, json.loads(call.arguments)) for call in calls]
    else:
        _, output = _stream_calls(text, _EXEC_TOOLS, chunk_size)
        content = output.content
        calls = [
            (call["name"], json.loads(call["arguments"]))
            for call in output.tool_calls
        ]
    assert content.strip() == "Running:"
    assert calls == [
        (
            "exec_command",
            {
                "cmd": "cat a.txt; echo done > out.txt",
                "justification": "",
                "max_output_tokens": 10000,
                "sandbox_permissions": "use_default",
                "workdir": "/srv/app;",
            },
        )
    ]


def test_responses_request_tools_keep_every_parameter():
    """The same call under a Responses request, whose tools are flat function
    tools."""
    text = _dsml("exec_command", _FULL_EXEC)
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    request = ResponsesRequest(
        model="test-model",
        input="test",
        tools=[
            {
                "type": "function",
                "name": "exec_command",
                "description": "Runs a command in a PTY.",
                "strict": False,
                "parameters": _EXEC_PARAMETERS,
            }
        ],
    )
    _, content, calls = parser.parse(
        text,
        request,
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert not (content or "").strip()
    assert [(call.name, json.loads(call.arguments)) for call in calls] == [
        ("exec_command", _FULL_EXEC_ARGS)
    ]
