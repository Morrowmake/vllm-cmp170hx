# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The file hash cache keeps only paths that still exist: a save drops the
entries of deleted files, keeps the live ones, and the launch it serves
resolves the same."""

from __future__ import annotations

import json
import os

from canon import HashCache
from helpers import World, write


def read_entries(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]


def test_save_drops_missing_paths_and_keeps_live_ones(tmp_path):
    files = {n: write(str(tmp_path / "files" / n), n) for n in ("kept", "gone", "new")}
    store = str(tmp_path / "hash-cache.json")
    cache = HashCache(store)
    for name in ("kept", "gone"):
        cache.sha256(files[name])
    cache.save()
    assert set(read_entries(store)) == {os.path.realpath(files["kept"]), os.path.realpath(files["gone"])}
    os.remove(files["gone"])
    cache = HashCache(store)
    cache.sha256(files["new"])
    cache.save()
    entries = read_entries(store)
    assert set(entries) == {os.path.realpath(files["kept"]), os.path.realpath(files["new"])}
    assert HashCache(store).sha256(files["kept"]) == entries[os.path.realpath(files["kept"])][4]


def test_launch_prunes_the_cache_and_resolves_the_same(tmp_path):
    w = World(tmp_path / "w")
    first = w.resolve()
    store = os.path.join(w.stamps, "hash-cache.json")
    live = read_entries(store)
    assert live
    rehashed = sorted(live)[0]
    stale = {str(tmp_path / "deleted-tree" / f"f{i}.py"): [1, i, 1, 1, "0" * 64] for i in range(3)}
    seeded = {p: e for p, e in live.items() if p != rehashed}
    with open(store, "w", encoding="utf-8") as f:
        json.dump({"schema": HashCache.SCHEMA, "entries": {**seeded, **stale}}, f)
    assert w.resolve() == first        # the dropped live entry is hashed again, so the cache is saved
    assert read_entries(store) == live
