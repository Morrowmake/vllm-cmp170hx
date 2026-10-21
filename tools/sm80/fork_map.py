#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate FORK.md: what this fork adds to upstream vLLM and what it changes.

The map compares the working tree with an upstream base commit. It reads git
(`git diff`, `git show`, `git rev-parse`) and the working tree, nothing else.
The output is deterministic: paths are sorted and there are no timestamps.

    python tools/sm80/fork_map.py --base 31e2443c90 --write
    python tools/sm80/fork_map.py --check

Without --base the commit in FORK.md's `Upstream base: <sha>` line is used.
Paths count only when git sees them: new files must be added to the index
before they appear in the map.

Unreferenced code is counted by name: a fork-added function, method or class
whose name occurs nowhere in the product tree (`vllm/`) except at definition
sites. A name shared with any other identifier or string counts as referenced,
so the count is a lower bound.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import regex as re

REPO = Path(__file__).resolve().parents[2]
FORK_MD = "FORK.md"
BASE_LINE = re.compile(r"^Upstream base: `?([0-9a-f]{7,40})`?[ \t]*$", re.MULTILINE)

UPSTREAM_ENVS = "vllm/envs.py"
FORK_ENVS = "vllm/sm80/envs.py"

# Fork-owned layout: (heading, path pattern, what lives there). A file is listed
# under the first heading whose pattern matches.
LAYOUT = (
    (
        "`vllm/models/<model>/nvidia/ops/`",
        re.compile(r"vllm/models/[^/]+/nvidia/ops/"),
        "Kernels used by one model only. Each model keeps its own copy.",
    ),
    (
        "`vllm/models/<model>/nvidia/`",
        re.compile(r"vllm/models/[^/]+/nvidia/"),
        "Model code of one model that the fork plugin registers in place of the "
        "upstream classes: model, layer and metadata subclasses.",
    ),
    (
        "`vllm/sm80/kernels/`",
        re.compile(r"vllm/sm80/kernels/"),
        "Kernels shared by several models on sm_80, one directory per family.",
    ),
    (
        "`vllm/sm80/`",
        re.compile(r"vllm/sm80/"),
        "The rest of the shared sm_80 code: the sparse MLA attention backend, "
        "the host-staged and flags-in-data all-reduce, FLA autotune pins, "
        "speculative-decoding helpers and pipeline-parallel worker helpers.",
    ),
    (
        "`csrc/libtorch_stable/moe/ampere_marlin/`",
        re.compile(r"csrc/libtorch_stable/moe/ampere_marlin/"),
        "An optional native library for the Marlin MoE kernels. vLLM runs without it.",
    ),
)

NON_PRODUCT_DIRS = ("tests/", "docs/", ".github/")
BUILD_NAMES = frozenset(
    {"CMakeLists.txt", "setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"}
)
STATUS_WORDS = {"M": "modified", "D": "deleted", "T": "type changed"}
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
DIFF_OPTS = ("--no-ext-diff", "--no-textconv", "--no-renames", "--no-color")


class GitError(RuntimeError):
    pass


def git(*args: str) -> bytes:
    cmd = ["git", "-C", str(REPO), "-c", "core.quotepath=off", *args]
    # Read-only: never refresh or rewrite the index while diffing.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    proc = subprocess.run(cmd, capture_output=True, env=env)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip() or "failed"
        raise GitError(f"git {' '.join(args)}: {msg}")
    return proc.stdout


def resolve_commit(rev: str) -> str:
    try:
        return (
            git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
            .decode()
            .strip()
        )
    except GitError:
        raise GitError(f"base commit {rev!r} is not in this repository") from None


@dataclass(frozen=True)
class Change:
    path: str
    status: str  # A, M, D or T
    added: int | None  # None for binary files
    deleted: int | None
    hunks: int = 0


def is_product(path: str) -> bool:
    return not (path.startswith(NON_PRODUCT_DIRS) or path.endswith(".md"))


def is_build_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        name in BUILD_NAMES
        or name.endswith(".cmake")
        or path.startswith(("cmake/", "requirements/"))
    )


def is_fork_product(path: str) -> bool:
    return is_product(path) and (
        path.startswith(("vllm/", "csrc/")) or is_build_file(path)
    )


def collect_changes(base: str) -> list[Change]:
    fields = git("diff", "-z", "--name-status", *DIFF_OPTS, base).split(b"\0")
    status = {}
    for code, path in zip(fields[0::2], fields[1::2]):
        status[path.decode()] = code.decode()[0]

    counts = {}
    for entry in git("diff", "-z", "--numstat", *DIFF_OPTS, base).split(b"\0"):
        if not entry:
            continue
        added, deleted, path = entry.decode().split("\t", 2)
        counts[path] = (
            None if added == "-" else int(added),
            None if deleted == "-" else int(deleted),
        )

    # Hunks of upstream files, counted at zero context: one hunk per edit site.
    patched = sorted(p for p, s in status.items() if s in STATUS_WORDS)
    hunks = {}
    if patched:
        patch = git("diff", "-U0", *DIFF_OPTS, "--diff-filter=MDT", base)
        files = patch.split(b"\ndiff --git ")
        if len(files) != len(patched):
            raise GitError(
                f"patch has {len(files)} files, name-status has {len(patched)}"
            )
        for path, text in zip(patched, files):
            header = text.split(b"\n", 1)[0].removeprefix(b"diff --git ")
            if header.decode() != f"a/{path} b/{path}":
                raise GitError(f"patch order mismatch at {path!r}: {header!r}")
            hunks[path] = sum(
                1 for line in text.split(b"\n") if line.startswith(b"@@ ")
            )

    return [
        Change(path, code, *counts.get(path, (0, 0)), hunks.get(path, 0))
        for path, code in sorted(status.items())
        if path != FORK_MD
    ]


# ---------------------------------------------------------------- env vars


@dataclass(frozen=True)
class EnvVar:
    name: str
    type: str
    declared: str
    default: str


def env_accessors(tree: ast.Module) -> dict[str, ast.expr]:
    """Map env name -> accessor node.

    Accessors are dict entries `"NAME": lambda: ...` (or a factory call or a
    named accessor function) in a dict literal that holds at least one lambda,
    and `d["NAME"] = lambda: ...` assignments. Tables of strings (retired
    names) are ignored.
    """
    found: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            pairs = [(k, v) for k, v in zip(node.keys, node.values) if k is not None]
            if (
                pairs
                and all(
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and ENV_NAME.match(k.value)
                    and isinstance(v, (ast.Lambda, ast.Call, ast.Name))
                    for k, v in pairs
                )
                and any(isinstance(v, ast.Lambda) for _, v in pairs)
            ):
                for k, v in pairs:
                    found.setdefault(k.value, v)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            for target in node.targets:
                key = target.slice if isinstance(target, ast.Subscript) else None
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and ENV_NAME.match(key.value)
                ):
                    found.setdefault(key.value, node.value)
    return found


def typing_declarations(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Map env name -> (type, default) from `if TYPE_CHECKING:` blocks."""
    found: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                default = ast.unparse(stmt.value) if stmt.value is not None else ""
                found.setdefault(
                    stmt.target.id, (ast.unparse(stmt.annotation), default)
                )
    return found


def module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def accessor_default(
    name: str, accessor: ast.AST, functions: dict[str, ast.FunctionDef]
) -> str:
    """The default the accessor passes with the name, e.g. getenv's 2nd arg.

    Looks inside the accessor first, then inside module-level helper
    functions the accessor calls without passing the name.
    """
    helpers = []
    for node in ast.walk(accessor):
        if not isinstance(node, ast.Call):
            continue
        if not (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == name
        ):
            if isinstance(node.func, ast.Name) and node.func.id in functions:
                helpers.append(functions[node.func.id])
            continue
        if len(node.args) >= 2:
            return ast.unparse(node.args[1])
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.unparse(kw.value)
        if ast.unparse(node.func) in ("os.getenv", "os.environ.get"):
            return "None"
    for helper in helpers:
        default = accessor_default(name, helper, {})
        if default:
            return default
    return ""


def collect_env_vars(base: str, added: set[str]) -> tuple[str, list[EnvVar]]:
    upstream_now = ast.parse((REPO / UPSTREAM_ENVS).read_text(encoding="utf-8"))
    retired = (
        "Retired names (`RETIRED_ENV_VARS`) are not listed: they are ignored "
        "and only warn."
    )
    if FORK_ENVS in added:
        tree = ast.parse((REPO / FORK_ENVS).read_text(encoding="utf-8"))
        accessors = env_accessors(tree)
        typed = {**typing_declarations(upstream_now), **typing_declarations(tree)}
        source = (
            f"Declared in `{FORK_ENVS}`, which `{UPSTREAM_ENVS}` registers into "
            f"`environment_variables`. {retired}"
        )
    else:
        tree = upstream_now
        upstream_base = ast.parse(git("show", f"{base}:{UPSTREAM_ENVS}").decode())
        before = env_accessors(upstream_base)
        accessors = {k: v for k, v in env_accessors(tree).items() if k not in before}
        typed = typing_declarations(tree)
        source = (
            "The names the fork added to `environment_variables` in "
            f"`{UPSTREAM_ENVS}`. {retired}"
        )
    functions = module_functions(tree)
    rows = [
        EnvVar(
            name, *typed.get(name, ("", "")), accessor_default(name, node, functions)
        )
        for name, node in sorted(accessors.items())
    ]
    return source, rows


# ---------------------------------------------------------------- unreferenced code


@dataclass(frozen=True)
class Unreferenced:
    path: str
    line: int
    name: str
    lines: int
    in_tests: bool


IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
DEF_SITE = re.compile(rb"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def tracked_py(prefix: str) -> list[str]:
    out = git("ls-files", "-z", "--", prefix).split(b"\0")
    return sorted(p.decode() for p in out if p.endswith(b".py"))


def name_counts(paths: list[str]) -> tuple[dict[bytes, int], dict[bytes, int]]:
    """Identifier occurrences and definition sites over the given files."""
    tokens: dict[bytes, int] = {}
    defs: dict[bytes, int] = {}
    for path in paths:
        try:
            data = (REPO / path).read_bytes()
        except OSError:
            continue
        for m in IDENT.finditer(data):
            tokens[m.group()] = tokens.get(m.group(), 0) + 1
        for m in DEF_SITE.finditer(data):
            defs[m.group(1)] = defs.get(m.group(1), 0) + 1
    return tokens, defs


def collect_unreferenced(fork_product: list[Change]) -> list[Unreferenced]:
    product = tracked_py("vllm")
    tokens, defs = name_counts(product)
    test_tokens, test_defs = name_counts(tracked_py("tests"))
    found = []

    def check(path: str, node: ast.AST) -> bool:
        """Record `node` if its name is never used; True when recorded."""
        name = node.name.encode()
        if node.name.startswith("__") and node.name.endswith("__"):
            return False
        if tokens.get(name, 0) > defs.get(name, 0):
            return False
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        found.append(
            Unreferenced(
                path,
                node.lineno,
                node.name,
                node.end_lineno - start + 1,
                test_tokens.get(name, 0) > test_defs.get(name, 0),
            )
        )
        return True

    for change in fork_product:
        if not change.path.endswith(".py"):
            continue
        try:
            tree = ast.parse((REPO / change.path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        # Top-level functions and classes; the methods of a class only when the
        # class itself is used.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                check(change.path, node)
            elif isinstance(node, ast.ClassDef) and not check(change.path, node):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        check(change.path, item)
    return sorted(found, key=lambda u: (u.path, u.line))


# ---------------------------------------------------------------- rendering


def num(value: int | None) -> str:
    return "binary" if value is None else f"{value:,}"


def code(text: str) -> str:
    return f"`{text}`".replace("|", "\\|") if text else "-"


def line_total(changes: list[Change]) -> int:
    return sum(c.added or 0 for c in changes)


def file_table(changes: list[Change]) -> list[str]:
    out = ["| Path | Lines |", "| --- | ---: |"]
    out += [f"| `{c.path}` | {num(c.added)} |" for c in changes]
    noun = "file" if len(changes) == 1 else "files"
    out.append(f"| **{len(changes)} {noun}** | **{line_total(changes):,}** |")
    return out


def patch_totals(changes: list[Change]) -> tuple[int, int, int, int]:
    return (
        len(changes),
        sum(c.added or 0 for c in changes),
        sum(c.deleted or 0 for c in changes),
        sum(c.hunks for c in changes),
    )


def patch_table(changes: list[Change]) -> list[str]:
    if not changes:
        return ["None."]
    out = ["| Path | Change | + | - | Hunks |", "| --- | --- | ---: | ---: | ---: |"]
    for c in changes:
        out.append(
            f"| `{c.path}` | {STATUS_WORDS[c.status]} | {num(c.added)} "
            f"| {num(c.deleted)} | {c.hunks} |"
        )
    n, plus, minus, hunks = patch_totals(changes)
    noun = "file" if n == 1 else "files"
    out.append(f"| **{n} {noun}** | | **{plus:,}** | **{minus:,}** | **{hunks:,}** |")
    return out


def env_table(envs: list[EnvVar]) -> list[str]:
    if not envs:
        return ["None."]
    typed = any(e.type or e.declared for e in envs)
    out = [
        "The default is the value the accessor uses when the variable is unset, "
        "as written in the code (environment strings are quoted).",
    ]
    if typed:
        out[0] += " Type and declared default come from `if TYPE_CHECKING:`."
        out += [
            "",
            "| Name | Type | Declared default | Default |",
            "| --- | --- | --- | --- |",
        ]
        out += [
            f"| `{e.name}` | {code(e.type)} | {code(e.declared)} | {code(e.default)} |"
            for e in envs
        ]
        out.append(f"| **{len(envs)} variables** | | | |")
    else:
        out += ["", "| Name | Default |", "| --- | --- |"]
        out += [f"| `{e.name}` | {code(e.default)} |" for e in envs]
        out.append(f"| **{len(envs)} variables** | |")
    return out


def unreferenced_table(unref: list[Unreferenced]) -> list[str]:
    if not unref:
        return ["None."]
    out = [
        "| Path | Line | Name | Lines | Used by tests |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    out += [
        f"| `{u.path}` | {u.line} | `{u.name}` | {u.lines:,} | "
        f"{'yes' if u.in_tests else 'no'} |"
        for u in unref
    ]
    total = sum(u.lines for u in unref)
    out.append(f"| **{len(unref)} definitions** | | | **{total:,}** | |")
    return out


def render(
    base: str,
    changes: list[Change],
    env_source: str,
    envs: list[EnvVar],
    unref: list[Unreferenced],
) -> str:
    added = [c for c in changes if c.status == "A"]
    patched = [c for c in changes if c.status in STATUS_WORDS]
    patched_product = [c for c in patched if is_product(c.path)]
    patched_other = [c for c in patched if not is_product(c.path)]
    fork_product = [c for c in added if is_fork_product(c.path)]

    out = [
        "# Fork map",
        "",
        "<!-- Generated by tools/sm80/fork_map.py. Do not edit by hand. -->",
        "",
        "This repository is Morrowmake's fork of vLLM for the NVIDIA CMP 170HX "
        "and other sm_80 (GA100-class) GPUs. It is a general vLLM: it serves "
        "the same models as upstream. The license is Apache-2.0, as upstream.",
        "",
        f"Upstream base: {base}",
        "",
        "All numbers compare the working tree with this commit. A fork-owned "
        "path does not exist at the base. An upstream-owned path does. This "
        "file is not listed in its own tables.",
        "",
        "Regenerate with `python tools/sm80/fork_map.py --write`. "
        "Check it with `python tools/sm80/fork_map.py --check`.",
        "",
        "## 1. Fork-owned code",
        "",
        "Files added since the base, by location. Lines are counted by git.",
        "",
    ]

    placed: set[str] = set()
    for heading, pattern, what in LAYOUT:
        files = [c for c in added if c.path not in placed and pattern.match(c.path)]
        placed.update(c.path for c in files)
        out += [f"### {heading}", "", what, ""]
        out += file_table(files) if files else ["No files yet."]
        out.append("")

    out += [
        "### Other fork-owned files",
        "",
        "Files added elsewhere, grouped by top-level directory.",
        "",
    ]
    groups: dict[str, list[Change]] = {}
    for c in added:
        if c.path not in placed:
            top = c.path.split("/", 1)[0] + "/" if "/" in c.path else ""
            groups.setdefault(top, []).append(c)
    if not groups:
        out += ["None.", ""]
    for top in sorted(groups):
        out += [f"#### `{top}`" if top else "#### Repository root", ""]
        out += file_table(groups[top])
        out.append("")

    out += [
        "## 2. Patched upstream files",
        "",
        "Every upstream-owned path that differs from the base. Hunks are counted "
        "at zero context lines, so one hunk is one edit site. Tests, docs and "
        "`.github` are paths under `tests/`, `docs/` or `.github/` and "
        "Markdown files. Everything else is product code.",
        "",
        "### Product code",
        "",
        *patch_table(patched_product),
        "",
        "### Tests, docs and .github",
        "",
        *patch_table(patched_other),
        "",
        "### Totals",
        "",
        "| Group | Files | + | - | Hunks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, group in (
        ("Product code", patched_product),
        ("Tests, docs and .github", patched_other),
        ("All", patched),
    ):
        n, plus, minus, hunks = patch_totals(group)
        out.append(f"| {label} | {n} | {plus:,} | {minus:,} | {hunks:,} |")
    out.append("")

    out += ["## 3. Fork environment variables", "", env_source, ""]
    out += env_table(envs)
    out.append("")

    out += [
        "## 4. Unreferenced fork code",
        "",
        "Functions, methods and classes in fork-added product files whose name "
        "occurs nowhere under `vllm/` except where it is defined (a class's "
        "methods are checked only when the class is used). A name that matches "
        "any other identifier or string counts as used, so this is a lower "
        "bound. 'Used by tests' means only tests name it.",
        "",
        *unreferenced_table(unref),
        "",
    ]

    p_n, p_plus, p_minus, _ = patch_totals(patched)
    pp_n, pp_plus, pp_minus, _ = patch_totals(patched_product)
    out += [
        "## 5. Health numbers",
        "",
        "Fork-added product code is new files under `vllm/` or `csrc/` plus new "
        "build files, without tests or Markdown.",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Fork-added product files | {len(fork_product):,} |",
        f"| Fork-added product lines | {line_total(fork_product):,} |",
        f"| Patched upstream files | {p_n:,} |",
        f"| Patched upstream lines | +{p_plus:,} / -{p_minus:,} |",
        f"| Patched upstream product files | {pp_n:,} |",
        f"| Patched upstream product lines | +{pp_plus:,} / -{pp_minus:,} |",
        f"| Product lines added (new files and patches) | "
        f"{line_total(fork_product) + pp_plus:,} |",
        f"| Fork environment variables (flags) | {len(envs):,} |",
        f"| Unreferenced fork definitions | {len(unref):,} "
        f"({sum(u.lines for u in unref):,} lines) |",
        "",
    ]
    return "\n".join(out)


def summary(
    changes: list[Change], envs: list[EnvVar], unref: list[Unreferenced]
) -> str:
    patched = [c for c in changes if c.status in STATUS_WORDS]
    product = [c for c in patched if is_product(c.path)]
    fork_product = [c for c in changes if c.status == "A" and is_fork_product(c.path)]
    n, plus, minus, hunks = patch_totals(patched)
    pn, pplus, pminus, phunks = patch_totals(product)
    return (
        f"patched upstream files: {n} (+{plus:,}/-{minus:,}, {hunks:,} hunks); "
        f"product: {pn} (+{pplus:,}/-{pminus:,}, {phunks:,} hunks); "
        f"fork-added product: {len(fork_product)} files, "
        f"{line_total(fork_product):,} lines; env vars: {len(envs)}; "
        f"unreferenced definitions: {len(unref)}"
    )


def recorded_base() -> str | None:
    path = REPO / FORK_MD
    if not path.is_file():
        return None
    match = BASE_LINE.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--base",
        help="upstream base commit (default: the 'Upstream base:' line of FORK.md)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write FORK.md")
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit 1 with a unified diff when FORK.md is stale",
    )
    args = parser.parse_args(argv)

    base_rev = args.base or recorded_base()
    if base_rev is None:
        parser.error("--base is required: FORK.md has no 'Upstream base:' line")
    try:
        base = resolve_commit(base_rev)
        changes = collect_changes(base)
        env_source, envs = collect_env_vars(
            base, {c.path for c in changes if c.status == "A"}
        )
        unref = collect_unreferenced(
            [c for c in changes if c.status == "A" and is_fork_product(c.path)]
        )
    except GitError as exc:
        print(f"fork_map: {exc}", file=sys.stderr)
        return 2
    text = render(base, changes, env_source, envs, unref)

    target = REPO / FORK_MD
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current == text:
            print(f"{FORK_MD} is current. {summary(changes, envs, unref)}")
            return 0
        sys.stdout.writelines(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                text.splitlines(keepends=True),
                f"a/{FORK_MD}",
                f"b/{FORK_MD} (regenerated)",
            )
        )
        print(
            f"{FORK_MD} is stale. Run: python tools/sm80/fork_map.py --write",
            file=sys.stderr,
        )
        return 1
    if args.write:
        target.write_text(text, encoding="utf-8")
        print(f"Wrote {FORK_MD}. {summary(changes, envs, unref)}")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
