# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The closed environment class table (defaults.toml ``[[classes]]``)."""

from __future__ import annotations

import fnmatch

from errors import Refusal

CLASSES = ("secret", "ownership", "code-loading", "location",
           "container-runtime", "functional", "inert")
UNCLASSIFIED = "unclassified"

# Code-loading names refused whatever their value, in every policy.
ALWAYS_REFUSED_CODE = ("PYTHONHOME", "PYTHONSTARTUP")


class ClassTable:
    def __init__(self, rows: list[dict], model_prefixes: list[str] | None = None) -> None:
        self.rows = []
        for row in rows:
            cls = row.get("class")
            if cls not in CLASSES:
                raise Refusal(f"defaults: unknown environment class {cls!r}")
            prefixes = list(row.get("prefixes", []))
            if row.get("model_prefixes"):
                prefixes += list(model_prefixes or [])
            self.rows.append((cls, set(row.get("names", [])), tuple(prefixes),
                              tuple(row.get("patterns", []))))

    def classify(self, name: str) -> str:
        for cls, names, prefixes, patterns in self.rows:
            if name in names or (prefixes and name.startswith(prefixes)):
                return cls
            if any(fnmatch.fnmatchcase(name, p) for p in patterns):
                return cls
        return UNCLASSIFIED


def always_refused(name: str) -> bool:
    """PYTHONHOME, PYTHONSTARTUP and every LD_* except LD_LIBRARY_PATH."""
    if name in ALWAYS_REFUSED_CODE:
        return True
    return name.startswith("LD_") and name != "LD_LIBRARY_PATH"
