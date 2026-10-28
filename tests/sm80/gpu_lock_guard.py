# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests run only under the GPU lock.

On a machine whose GPUs are shared through a lock, the lock holder's job gets
``GPU_LOCK_TOKENS`` (its lock tokens) in its environment. A pytest run without
it must not reach a card another job holds: CUDA tests started from a CPU test
run on cards in use by a timed job have raised Xid errors there.

``tests/conftest.py`` calls:

* ``pytest_configure(config)`` (imported as its hook): registers the ``cuda``
  marker;
* ``hide_cuda()`` before anything can initialise CUDA: outside the lock it
  empties ``CUDA_VISIBLE_DEVICES``, so no test, worker or child process of the
  run sees a card, and tests that skip without CUDA skip;
* ``refuse_cuda_tests(config, items)`` from ``pytest_collection_modifyitems``:
  tests marked ``cuda`` (they need a CUDA device) refuse the run at collection
  when the caller left a card visible outside the lock, naming them (under
  xdist the workers run nothing and the run stops with that message); when
  the caller hid the cards itself (``CUDA_VISIBLE_DEVICES=``, a CPU run) they
  are skipped.

Inside the lock nothing changes. On a machine without a GPU lock, set
``GPU_LOCK_TOKENS`` to any value.
"""

import os

import pytest

LOCK_ENV = "GPU_LOCK_TOKENS"
MARKER = "cuda"
RULE = "GPU tests run only under the GPU lock"
# The caller's CUDA_VISIBLE_DEVICES ("all" when unset) before hide_cuda emptied
# it, kept in the environment for xdist workers and child processes.
CALLER_ENV = "GPU_LOCK_GUARD_CALLER_DEVICES"


def locked() -> bool:
    return bool(os.environ.get(LOCK_ENV, "").strip())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: needs a CUDA device; outside the GPU lock refused or skipped",
    )


def hide_cuda() -> None:
    if locked():
        return
    os.environ.setdefault(CALLER_ENV, os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def refuse_cuda_tests(config: pytest.Config, items: list[pytest.Item]) -> None:
    if locked():
        return
    marked = [item for item in items if item.get_closest_marker(MARKER)]
    if not marked:
        return
    if os.environ.get(CALLER_ENV, "") == "":
        skip = pytest.mark.skip(reason=f"{RULE} ({LOCK_ENV} is unset)")
        for item in marked:
            item.add_marker(skip)
        return
    names = "\n  ".join(item.nodeid for item in marked[:10])
    more = f"\n  ... and {len(marked) - 10} more" if len(marked) > 10 else ""
    message = (
        f"{RULE}: {LOCK_ENV} is unset and CUDA_VISIBLE_DEVICES was "
        f"{os.environ[CALLER_ENV]!r}, but {len(marked)} collected test(s) are "
        f"marked {MARKER!r} (they need a CUDA device):\n  {names}{more}\n"
        f"Run them under the GPU lock, or run without a card: "
        f"CUDA_VISIBLE_DEVICES= (they are then skipped) or -m 'not {MARKER}'."
    )
    if hasattr(config, "workerinput"):
        # An xdist worker: an error here would only crash it. It runs nothing
        # and stops the run; the controller prints the message.
        run = items[0].session
        config.hook.pytest_deselected(items=list(items))
        items[:] = []
        run.shouldstop = message
        return
    raise pytest.UsageError(message)
