# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build directories of the plugin's JIT extensions and torch's build lock.

A build directory is shared only by identical builds, and a ``lock`` left by
a killed build is removed instead of being waited on forever. torch's
compiler is replaced by a recorder: nothing is compiled.
"""

import importlib
import os
import subprocess
import sys
import time

import pytest
import torch
import torch.utils.cpp_extension

from vllm_sm80 import jit_extension

EXTENSIONS = [
    ("vllm_sm80.distributed.custom_all_reduce_flags", "load_module"),
    ("vllm_sm80.distributed.host_shm_all_reduce", "_load_module"),
]


@pytest.fixture
def loads(monkeypatch):
    """Each torch load as (kwargs, whether ``lock`` existed at the call)."""
    calls = []

    def load(**kw):
        lock = os.path.join(kw["build_directory"], "lock")
        calls.append((kw, os.path.exists(lock)))
        return object()

    monkeypatch.setattr(torch.utils.cpp_extension, "load", load)
    return calls


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(jit_extension.logger, "warning",
                        lambda msg, *args: seen.append(msg % args))
    return seen


def _load(base, source="src", cflags=("-O3", ), cuda_cflags=("-O3", )):
    jit_extension.load_cuda_extension("ext",
                                      source,
                                      str(base),
                                      extra_cflags=list(cflags),
                                      extra_cuda_cflags=list(cuda_cflags))


def _dir(base):
    """Build directory of the default ``_load``."""
    return jit_extension.build_dir_for("ext",
                                       "src",
                                       str(base),
                                       extra_cflags=["-O3"],
                                       extra_cuda_cflags=["-O3"])[1]


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_build_dir_is_shared_only_by_identical_builds(monkeypatch, tmp_path,
                                                     loads):
    _load(tmp_path)
    _load(tmp_path)
    _load(tmp_path, source="other")
    _load(tmp_path, cflags=["-O2"])
    _load(tmp_path, cuda_cflags=["-O2"])
    monkeypatch.setattr(torch, "__version__", "0.0.0")
    _load(tmp_path)
    monkeypatch.setattr(torch.version, "cuda", "0.0")
    _load(tmp_path)
    dirs = [kw["build_directory"] for kw, _ in loads]
    assert dirs[0] == dirs[1]
    assert len(set(dirs)) == 6
    for kw, _ in loads:
        assert os.path.dirname(kw["build_directory"]) == str(tmp_path)
        assert os.path.basename(kw["build_directory"]) == kw["name"]


@pytest.mark.parametrize("module,func", EXTENSIONS)
def test_extensions_build_in_their_own_keyed_dir(monkeypatch, tmp_path, loads,
                                                 module, func):
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "_MODULE", None)
    monkeypatch.setattr(mod, "_build_dir", lambda: str(tmp_path))
    getattr(mod, func)()
    (kw, _), = loads
    assert kw["build_directory"] == os.path.join(str(tmp_path), kw["name"])


def test_lock_of_dead_builder_is_removed(tmp_path, loads, warnings):
    d = _dir(tmp_path)
    os.makedirs(d)
    pid = _dead_pid()
    with open(os.path.join(d, ".owner"), "w") as f:
        f.write(str(pid))
    open(os.path.join(d, "lock"), "w").close()
    _load(tmp_path)
    assert [held for _, held in loads] == [False]
    assert len(warnings) == 1
    assert d in warnings[0] and str(pid) in warnings[0]


def test_lock_of_live_builder_is_kept(tmp_path, loads, warnings):
    d = _dir(tmp_path)
    os.makedirs(d)
    with open(os.path.join(d, ".owner"), "w") as f:
        f.write(str(os.getppid()))
    lock = os.path.join(d, "lock")
    open(lock, "w").close()
    old = time.time() - 10 * jit_extension.STALE_LOCK_AGE_S
    os.utime(lock, (old, old))
    _load(tmp_path)
    assert [held for _, held in loads] == [True]
    assert os.path.exists(lock)
    assert warnings == []


def test_unowned_lock_is_removed_only_when_old(tmp_path, loads, warnings):
    d = _dir(tmp_path)
    os.makedirs(d)
    lock = os.path.join(d, "lock")
    open(lock, "w").close()
    _load(tmp_path)
    assert os.path.exists(lock)
    assert warnings == []
    old = time.time() - 2 * jit_extension.STALE_LOCK_AGE_S
    os.utime(lock, (old, old))
    _load(tmp_path)
    assert [held for _, held in loads] == [True, False]
    assert len(warnings) == 1 and d in warnings[0]


def test_owner_is_recorded_during_load_and_cleared_after(monkeypatch, tmp_path):
    owner = os.path.join(_dir(tmp_path), ".owner")
    seen = []

    def load(**kw):
        with open(owner) as f:
            seen.append(f.read())
        raise RuntimeError("build failed")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", load)
    with pytest.raises(RuntimeError):
        _load(tmp_path)
    assert seen == [str(os.getpid())]
    with open(owner) as f:
        assert f.read() == ""
