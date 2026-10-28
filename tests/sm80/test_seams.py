# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the upstream seams pinned in seams.yaml are unchanged.

One test per seam row. A failure means upstream changed (or removed) an
object the sm80 fork code relies on; check the change against the row's
`why`, adapt the fork code if needed, then refresh the pin with
tests/sm80/update_seams.py.

Set SM80_SEAMS_FILE to check a different pin file.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "sm80_update_seams", _HERE / "update_seams.py"
)
seams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seams)

SEAMS_FILE = Path(os.environ.get("SM80_SEAMS_FILE", seams.SEAMS_PATH))
ROWS = seams.load_rows(SEAMS_FILE)


def _drift_message(row: dict, problem: str) -> str:
    return (
        f"seam {row['id']} ({row['target']}) {problem}\n"
        f"  fork code depends on: {row['why']}\n"
        f"  Check the upstream change against that dependency, adapt the fork "
        f"code if needed, then run: python tests/sm80/update_seams.py {row['id']}"
    )


@pytest.mark.parametrize("row", ROWS, ids=[row["id"] for row in ROWS])
def test_seam_pinned(row: dict) -> None:
    try:
        measured = seams.measure(row)
    except Exception as exc:
        pytest.fail(
            _drift_message(row, f"cannot be resolved: {type(exc).__name__}: {exc}"),
            pytrace=False,
        )
    drift = [
        f"  {field}: pinned {row.get(field)!r}, now {value!r}"
        for field, value in measured.items()
        if row.get(field) != value
    ]
    if drift:
        pytest.fail(
            _drift_message(row, "changed upstream:\n" + "\n".join(drift)),
            pytrace=False,
        )
