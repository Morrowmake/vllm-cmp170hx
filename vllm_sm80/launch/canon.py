# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical JSON, hashing and atomic writes shared by the launcher modules.

Stdlib only. Every hash the launcher prints is the sha256 of the canonical
form: UTF-8 JSON with sorted keys and no insignificant whitespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any


def canonical(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(obj: Any) -> str:
    return sha256_bytes(canonical(obj))


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_write(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary name and a rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class HashCache:
    """sha256 of files keyed by (device, inode, size, mtime_ns).

    The cache only avoids re-reading unchanged files; a changed stat always
    re-hashes. ``path`` None keeps the cache in memory only. A save keeps
    only the entries whose path still exists, so trees that are gone do not
    grow the file every launch reads.
    """

    SCHEMA = "launch-hash-cache/1"

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.entries: dict[str, list] = {}
        self.dirty = False
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("schema") == self.SCHEMA:
                    self.entries = data.get("entries", {})
            except (OSError, ValueError):
                self.entries = {}

    @staticmethod
    def _key(st: os.stat_result) -> list:
        return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]

    def sha256(self, path: str) -> str:
        real = os.path.realpath(path)
        st = os.stat(real)
        key = self._key(st)
        hit = self.entries.get(real)
        if hit and hit[:4] == key:
            return hit[4]
        digest = sha256_file(real)
        self.entries[real] = key + [digest]
        self.dirty = True
        return digest

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        self.entries = {p: e for p, e in self.entries.items() if os.path.exists(p)}
        try:
            atomic_write(self.path, canonical({"schema": self.SCHEMA, "entries": self.entries}))
        except OSError:
            pass
