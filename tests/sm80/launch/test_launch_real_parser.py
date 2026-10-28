# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launcher's static option inventory against this tree's real serve
parser, built as the command line builds it: every long spelling the parser
accepts (aliases and negative forms included) is read as the option it sets,
and the inventory names no option the parser lacks. Imports vllm."""

from __future__ import annotations

from helpers import ROOT
from registry import read_arg_options
from resolver import arg_destination


def serve_parser():
    from vllm.entrypoints.cli.serve import cmd_init
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    (serve,) = cmd_init()
    return serve.subparser_init(FlexibleArgumentParser().add_subparsers())


def test_every_real_spelling_read_as_the_option_it_sets():
    actions = {s[2:]: a for s, a in serve_parser()._option_string_actions.items() if s.startswith("--")}
    opts = read_arg_options(ROOT)
    first = {s: next(o[2:] for o in a.option_strings if o.startswith("--")) for s, a in actions.items()}
    misread = {s: (arg_destination(s, opts), first[s]) for s in actions if arg_destination(s, opts) != first[s]}
    assert not misread, misread
    assert opts.options <= set(actions), sorted(opts.options - set(actions))
