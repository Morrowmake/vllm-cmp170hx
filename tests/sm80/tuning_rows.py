# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test helper: give the sm80 tuning table (vllm_sm80/tuning.py) chosen rows.

``use_rows(monkeypatch, ("dense_gemm", "thin_max_rows", {}, 8))`` makes every
lookup for the rest of the test see exactly these rows (and no others, so a
parameter without a row takes its default even on a card with a table). The
rows go through the table parser; monkeypatch restores the real table and a
lookup cache on teardown.
"""

import json

from vllm_sm80 import tuning


def table_text(rows, card: str = "test card") -> str:
    """An sm80-tuning/1 file holding ``rows`` of (op, param, key, value)."""
    out = [f'schema = "{tuning.SCHEMA}"', f"card = {json.dumps(card)}"]
    for op, param, key, value in rows:
        fields = ", ".join(f"{dim} = {json.dumps(v)}" for dim, v in key.items())
        out += [
            "",
            "[[row]]",
            f"op = {json.dumps(op)}",
            f"param = {json.dumps(param)}",
            f"key = {{{fields}}}",
            f"value = {int(value)}",
            'evidence = "test"',
        ]
    return "\n".join(out) + "\n"


def use_rows(monkeypatch, *rows) -> tuning.Table:
    """For the rest of the test the table holds exactly ``rows``."""
    loaded = tuning.parse_table(table_text(rows), "<test rows>")
    monkeypatch.setattr(tuning, "table", lambda: loaded)
    monkeypatch.setattr(tuning, "_RESULTS", {})
    return loaded
