# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size cut-offs of the sm_80 kernels, read from a per-card table.

A speed crossover (the largest batch a kernel variant wins at, the smallest
prefill it pays off at, a chunk or message size) is data rather than a switch:
``vllm_sm80/tuning/<card>.toml`` holds one row per operation, parameter and
shape, and the code asks :func:`lookup` for the value at its own shape::

    rows = tuning.lookup("dense_gemm", "thin_max_rows",
                         {"gemm_n": n, "gemm_k": k}, 32)

The table (schema ``sm80-tuning/1``)::

    schema = "sm80-tuning/1"
    card = "NVIDIA CMP 170HX"          # the device name, verbatim

    [[row]]
    op = "dense_gemm"                  # an operation id
    param = "thin_max_rows"
    key = {gemm_n = 4096, gemm_k = 4096}
    value = 32
    evidence = "where the value comes from"

A row applies when every field of its key equals that field of the lookup
key; among the rows that apply, the one matching more fields wins. Two
applying rows with equally many fields are a table error. With no applying
row the lookup returns ``default`` and logs one INFO line naming the
operation, parameter and key, so a shape nobody tuned shows at boot.

The table is ``VLLM_SM80_TUNING_FILE`` when that is set (a diagnostic, for
tuning runs), else the file of the current CUDA device's name
(``torch.cuda.get_device_name()``) in ``tuning/``, lower-cased, every run of
other characters than letters and digits replaced by ``-``. A process without
a visible CUDA device or without a file for its device has no rows.

Results are cached per (op, param, key, default) in host Python. Callers ask
when a gate is evaluated (eager steps, warmup, graph capture); a captured
graph replays the kernels the capture chose, so a replay never asks. The
table's rows are not part of the torch.compile cache key: a gate inside a
torch.compile-traced region bakes the value it read into the compiled graph,
so a changed table needs a fresh compile cache there.
"""

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEMA = "sm80-tuning/1"
TABLE_DIR = Path(__file__).with_name("tuning")
_ROW_FIELDS = frozenset({"op", "param", "key", "value", "evidence"})
_TOP_FIELDS = frozenset({"schema", "card", "row"})

Key = tuple[tuple[str, int | str], ...]


class TuningTableError(ValueError):
    """A table that does not follow the sm80-tuning/1 schema."""


@dataclass(frozen=True)
class Row:
    op: str
    param: str
    key: Key
    value: int
    evidence: str


@dataclass(frozen=True)
class Table:
    source: str
    card: str
    rows: dict[tuple[str, str], tuple[Row, ...]]


_NO_TABLE = Table(source="", card="", rows={})


def card_file_name(card: str) -> str:
    """'NVIDIA CMP 170HX' -> 'nvidia-cmp-170hx.toml'."""
    return re.sub(r"[^a-z0-9]+", "-", card.lower()).strip("-") + ".toml"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _key_text(key: Key) -> str:
    return "{" + ", ".join(f"{d}={v!r}" for d, v in key) + "}"


def _row(source: str, i: int, raw: object) -> Row:
    where = f"{source}: row {i + 1}"
    if not isinstance(raw, dict):
        raise TuningTableError(f"{where} is not a table")
    if set(raw) != _ROW_FIELDS:
        raise TuningTableError(
            f"{where} has fields {sorted(raw)}, needs exactly "
            f"{sorted(_ROW_FIELDS)}"
        )
    for name in ("op", "param", "evidence"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise TuningTableError(f"{where}: {name} must be a non-empty string")
    if not _is_int(raw["value"]):
        raise TuningTableError(f"{where}: value must be an integer")
    key = raw["key"]
    if not isinstance(key, dict):
        raise TuningTableError(f"{where}: key must be a table")
    for dim, value in key.items():
        if not (_is_int(value) or isinstance(value, str)):
            raise TuningTableError(
                f"{where}: key field {dim} must be an integer or a string"
            )
    return Row(
        op=raw["op"],
        param=raw["param"],
        key=tuple(sorted(key.items())),
        value=raw["value"],
        evidence=raw["evidence"],
    )


def parse_table(text: str, source: str) -> Table:
    """The rows of one table file; raises TuningTableError on any schema
    fault, so a broken table stops the process instead of serving defaults."""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise TuningTableError(f"{source}: {e}") from e
    if doc.get("schema") != SCHEMA:
        raise TuningTableError(f"{source}: schema must be {SCHEMA!r}")
    extra = set(doc) - _TOP_FIELDS
    if extra:
        raise TuningTableError(f"{source}: unknown fields {sorted(extra)}")
    card = doc.get("card")
    if not isinstance(card, str) or not card:
        raise TuningTableError(f"{source}: card must be a non-empty string")
    raw_rows = doc.get("row", [])
    if not isinstance(raw_rows, list):
        raise TuningTableError(f"{source}: row must be an array of tables")
    by_param: dict[tuple[str, str], list[Row]] = {}
    seen: set[tuple[str, str, Key]] = set()
    for i, raw in enumerate(raw_rows):
        row = _row(source, i, raw)
        ident = (row.op, row.param, row.key)
        if ident in seen:
            raise TuningTableError(
                f"{source}: two rows for {row.op}/{row.param} at "
                f"{_key_text(row.key)}"
            )
        seen.add(ident)
        by_param.setdefault((row.op, row.param), []).append(row)
    return Table(
        source=source,
        card=card,
        rows={k: tuple(v) for k, v in by_param.items()},
    )


def _device_name() -> str | None:
    """The current CUDA device's name, None without a visible device."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name()


@cache
def table() -> Table:
    """The table this process reads (loaded once)."""
    override = envs.VLLM_SM80_TUNING_FILE
    if override:
        path = Path(override)
        loaded = parse_table(path.read_text(), str(path))
        logger.info(
            "sm80 tuning table %s (card %s, %d rows) "
            "[VLLM_SM80_TUNING_FILE; unset to use the card's own table]",
            path,
            loaded.card,
            sum(len(rows) for rows in loaded.rows.values()),
        )
        return loaded
    card = _device_name()
    if card is None:
        return _NO_TABLE
    path = TABLE_DIR / card_file_name(card)
    if not path.is_file():
        logger.info(
            "sm80 tuning: no table for %s (%s); every cut-off takes its default",
            card,
            path,
        )
        return _NO_TABLE
    loaded = parse_table(path.read_text(), str(path))
    if loaded.card != card:
        raise TuningTableError(
            f"{path} is the table of {loaded.card!r}, not of this device "
            f"{card!r}"
        )
    logger.info(
        "sm80 tuning table %s (%d rows)",
        path,
        sum(len(rows) for rows in loaded.rows.values()),
    )
    return loaded


# (op, param, key, default) -> the value lookup returned.
_RESULTS: dict[tuple[str, str, Key, int], int] = {}


def lookup(
    op: str, param: str, key: Mapping[str, int | str], default: int
) -> int:
    """The tuned value of ``param`` of operation ``op`` at shape ``key``,
    else ``default``. After the first call per (op, param, key, default) a
    dictionary read."""
    ident = (op, param, tuple(sorted(key.items())), default)
    value = _RESULTS.get(ident)
    if value is None:
        value = _RESULTS[ident] = _resolve(*ident)
    return value


def _resolve(op: str, param: str, key: Key, default: int) -> int:
    loaded = table()
    fields = dict(key)
    best: Row | None = None
    tied: Row | None = None
    for row in loaded.rows.get((op, param), ()):
        if any(d not in fields or fields[d] != v for d, v in row.key):
            continue
        if best is None or len(row.key) > len(best.key):
            best, tied = row, None
        elif len(row.key) == len(best.key):
            tied = row
    if tied is not None:
        assert best is not None
        raise TuningTableError(
            f"{loaded.source}: {op}/{param} at {_key_text(key)} matches two "
            f"rows equally: {_key_text(best.key)} and {_key_text(tied.key)}"
        )
    if best is None:
        logger.info(
            "sm80 tuning: no row for %s/%s at %s in %s; using the default %d",
            op,
            param,
            _key_text(key),
            loaded.source or "no table",
            default,
        )
        return default
    logger.info(
        "sm80 tuning: %s/%s at %s = %d (row %s; %s)",
        op,
        param,
        _key_text(key),
        best.value,
        _key_text(best.key),
        best.evidence,
    )
    return best.value


def cache_clear() -> None:
    """Forget the loaded table and every cached lookup."""
    _RESULTS.clear()
    table.cache_clear()
