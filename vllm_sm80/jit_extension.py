# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT builds of the plugin's CUDA extensions.

The C++ standard is torch's: ``torch.utils.cpp_extension`` already adds the
``-std`` its own headers require to both the host and the nvcc command lines,
so callers pass none. A ``-std`` from a caller would come later on the command
line and override torch's, which breaks the build as soon as torch raises its
minimum (torch 2.14's headers refuse anything below C++20).

Each build gets its own directory, named by a hash of everything that goes
into it (source, flags, torch version and the CUDA it was built for), so two
source trees or virtualenvs share a directory only when they would produce
the same binary.

torch serialises builds in a directory with a ``lock`` file that it creates
and removes; a process killed mid-build leaves the file behind and every later
load waits on it forever. Loads of one build directory are therefore also
serialised by an ``flock`` on ``.owner``, which records the holder's pid while
it loads. A ``lock`` found under that flock whose recorded holder is dead, or
that has no recorded holder and is older than ``STALE_LOCK_AGE_S``, is left
over from a killed build and is removed.
"""

import errno
import fcntl
import hashlib
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Twice the longest build measured for these extensions (about 52 s).
STALE_LOCK_AGE_S = 120.0


def build_dir_for(
    prefix: str,
    source: str,
    base_dir: str,
    *,
    extra_cflags: list[str],
    extra_cuda_cflags: list[str],
) -> tuple[str, str]:
    """The extension name and build directory for this exact build.

    The C++ standard is not a caller flag (``load_cuda_extension`` refuses
    one); torch picks it, so the torch version covers it.
    """
    key = "\0".join([
        source, *extra_cflags, "", *extra_cuda_cflags, "", torch.__version__,
        str(torch.version.cuda)
    ])
    name = f"{prefix}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    return name, os.path.join(os.path.abspath(base_dir), name)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM
    return True


def _clear_stale_lock(build_dir: str, recorded: str) -> None:
    """Remove torch's ``lock`` in ``build_dir`` if no live build holds it.

    Called with the ``.owner`` flock held; ``recorded`` is the pid the last
    holder left in ``.owner`` (empty when it finished its load).
    """
    lock = os.path.join(build_dir, "lock")
    try:
        age = time.time() - os.stat(lock).st_mtime
    except FileNotFoundError:
        return
    if recorded.isdigit():
        pid = int(recorded)
        if _pid_alive(pid):
            return
        reason = f"its builder (pid {pid}) is no longer running"
    elif age > STALE_LOCK_AGE_S:
        reason = f"it has no recorded builder and is {age:.0f} s old"
    else:
        return
    logger.warning("Removing stale JIT build lock in %s: %s.", build_dir,
                   reason)
    try:
        os.remove(lock)
    except FileNotFoundError:
        pass


def load_cuda_extension(
    prefix: str,
    source: str,
    base_dir: str,
    *,
    extra_cflags: list[str],
    extra_cuda_cflags: list[str],
):
    """Compile (or reuse) the CUDA ``source`` as a torch extension.

    The build lives in ``build_dir_for(...)`` under ``base_dir``.
    """
    for flag in (*extra_cflags, *extra_cuda_cflags):
        if flag.startswith("-std"):
            raise ValueError(
                f"{prefix}: {flag!r} would override the C++ standard torch "
                "requires; torch.utils.cpp_extension sets it")
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    name, build_dir = build_dir_for(prefix,
                                    source,
                                    base_dir,
                                    extra_cflags=extra_cflags,
                                    extra_cuda_cflags=extra_cuda_cflags)
    os.makedirs(build_dir, exist_ok=True)
    src = os.path.join(build_dir, f"{name}.cu")
    if not os.path.exists(src):
        tmp = src + f".{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(source)
        os.replace(tmp, src)
    with open(os.path.join(build_dir, ".owner"), "a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        owner.seek(0)
        _clear_stale_lock(build_dir, owner.read().strip())
        owner.seek(0)
        owner.truncate()
        owner.write(str(os.getpid()))
        owner.flush()
        try:
            return load(
                name=name,
                sources=[src],
                build_directory=build_dir,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=False,
            )
        finally:
            owner.seek(0)
            owner.truncate()
            owner.flush()
