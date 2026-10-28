# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU timing of CUDA-graph steps (``VLLM_SM80_LIGHT_TIMING``).

The flag holds a JSON object:

- ``dir``: absolute directory of the dump requests and the dumps.
- ``roots``: ``{tag: class name}``; a module of that class is one model forward
  (the target model, a drafter). A root called inside another root's forward
  is part of it. A root must be called through ``nn.Module.__call__``: a class
  compiled with ``support_torch_compile`` replaces it and runs no hooks, so
  name the module that calls it instead.
- ``layer``: class name of a decoder layer, or a list of them (a drafter whose
  layers are of another class); ``scopes``: attribute names of a layer's
  children timed on their own (``[]``: layers only; a layer without one of
  them skips it).
- ``rows`` (default 4096) and ``width`` (default 512): the device ring.

Each forward of a root opens the next row of an int64 ``[rows, width]`` ring
on the root's device with a one-thread kernel: column 0 is the root's tag index
plus one, negative for an eager forward, column 1 its ``%globaltimer`` (ns) at
entry, the other columns zero. In a forward captured into a CUDA graph, one
more kernel (``light_timing_stamp`` in a trace) stamps the timer into a column
of that row at each layer's entry, at the entry and exit of each scope and at
the root's exit. A column belongs to one (root tag, module path, edge),
assigned when the first graph that reaches it is captured, so every graph of
every size writes a module's stamps to the same column. Eager forwards
(warmup, prefill) open their row and stamp nothing else: they add one kernel
and mark the step boundary. A stamp runs when the stream reaches it, so the
difference of two stamps on one stream is the GPU time between them, host gaps
and waits included. Nothing reads the ring while the model runs: no
synchronisation and no host work per step. Each rank stamps the layers it
holds: under pipeline parallelism a stage's columns name its own layers, and a
root that runs on one stage only (a drafter on the last) opens rows there
only.

Dumps: each worker's thread polls ``<dir>/request`` every 0.2 s; when its text
changes to N, the worker copies the ring and its counter on a side stream and
writes ``<dir>/rank<global rank>.<N>.npz``: ``counter`` (forwards opened so
far), ``rows`` (the rows opened since the previous dump of this process, oldest
first, at most ``rows``), ``first`` (counter value of the first of them),
``lost`` (rows opened since the previous dump that the ring no longer holds)
and ``meta`` (JSON: rank, device, tags, layer, scopes, columns as [tag, path,
edge]). A request already present at start is not served.

Hooks: a global forward pre-hook finds the roots, then each root, its layers
and their scopes get forward hooks of their own. The finder is removed once
every configured root class has been seen, or else when a root it found starts
its next forward: the first step (the profile run) calls every root the rank
holds, so a rank without one (a pipeline stage without the drafter, a run
without one) keeps no per-module hook past it, and a root first called after
that is never timed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

FLAG = "VLLM_SM80_LIGHT_TIMING"
POLL_S = 0.2
DEFAULT_ROWS = 4096
DEFAULT_WIDTH = 512
FIRST_COLUMN = 2  # columns 0 and 1: tag, entry time


def parse(text: str) -> dict[str, Any]:
    """The flag's settings; ValueError naming what is wrong."""
    try:
        cfg = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{FLAG} is not JSON: {exc}") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{FLAG} must be a JSON object")
    unknown = set(cfg) - {"dir", "roots", "layer", "scopes", "rows", "width"}
    if unknown:
        raise ValueError(f"{FLAG}: unknown keys {sorted(unknown)}")
    if not isinstance(cfg.get("dir"), str) or not os.path.isabs(cfg["dir"]):
        raise ValueError(f"{FLAG}: dir must be an absolute path")
    roots = cfg.get("roots")
    if (
        not isinstance(roots, dict)
        or not roots
        or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in roots.items())
        or len(set(roots.values())) != len(roots)
    ):
        raise ValueError(f"{FLAG}: roots must map tags to distinct class names")
    layer = cfg.get("layer")
    layers = [layer] if isinstance(layer, str) else layer
    if (
        not isinstance(layers, list)
        or not layers
        or not all(isinstance(c, str) and c for c in layers)
    ):
        raise ValueError(f"{FLAG}: layer must be a class name or a list of them")
    scopes = cfg.get("scopes", [])
    if not isinstance(scopes, list) or not all(isinstance(s, str) and s for s in scopes):
        raise ValueError(f"{FLAG}: scopes must be a list of attribute names")
    rows, width = cfg.get("rows", DEFAULT_ROWS), cfg.get("width", DEFAULT_WIDTH)
    for name, v in (("rows", rows), ("width", width)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{FLAG}: {name} must be a positive integer")
    if width <= FIRST_COLUMN:
        raise ValueError(f"{FLAG}: width must exceed {FIRST_COLUMN}")
    return {
        "dir": cfg["dir"],
        "roots": dict(roots),
        "layer": layer if isinstance(layer, str) else list(layers),
        "scopes": list(scopes),
        "rows": rows,
        "width": width,
    }


_stamp_kernel = None


def _kernel():
    """The stamp kernel (built on first use: Triton is imported only with the
    flag on)."""
    global _stamp_kernel
    if _stamp_kernel is None:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import globaltimer

        @triton.jit(do_not_specialize=["column", "tag", "rows", "width"])
        def light_timing_stamp(
            ring, counter, column, tag, rows, width,
            OPEN: tl.constexpr, BLOCK: tl.constexpr,
        ):
            now = globaltimer()
            if OPEN:
                n = tl.load(counter)
                row = (n % rows).to(tl.int64)
                offs = tl.arange(0, BLOCK)
                values = tl.where(offs == 0, tag.to(tl.int64), tl.where(offs == 1, now, 0))
                tl.store(ring + row * width + offs, values, mask=offs < width)
                tl.debug_barrier()
                tl.store(counter, n + 1)
            else:
                n = tl.load(counter) - 1
                row = (n % rows).to(tl.int64)
                tl.store(ring + row * width + column, now)

        _stamp_kernel = light_timing_stamp
    return _stamp_kernel


def _launch(timer: "LightTimer", column: int, tag: int) -> None:
    """Enqueue one stamp on the current stream: open a row with `tag` when
    `column` is 0, else stamp `column` of the open row."""
    kernel = _kernel()
    if column == 0:
        kernel[(1,)](timer.ring, timer.counter, 0, tag, timer.rows, timer.width,
                     OPEN=True, BLOCK=timer.block, num_warps=1)
    else:
        kernel[(1,)](timer.ring, timer.counter, column, 0, timer.rows, timer.width,
                     OPEN=False, BLOCK=1, num_warps=1)


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


class LightTimer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.rows, self.width = cfg["rows"], cfg["width"]
        self.block = 1 << (self.width - 1).bit_length()
        self.tags = list(cfg["roots"])
        self.by_class = {cls: tag for tag, cls in cfg["roots"].items()}
        self.missing = set(self.by_class)
        self.columns: dict[tuple[str, str, str], int] = {}
        self.ring: torch.Tensor | None = None
        self.counter: torch.Tensor | None = None
        self.device: torch.device | None = None
        self.current: str | None = None
        self.depth = 0
        self.stamping = False
        self.attached: set[int] = set()
        self.handle = None
        self.lock = threading.Lock()
        self.dumped = None  # counter value at the previous dump

    # -- attach ---------------------------------------------------------------

    def find_roots(self, module: nn.Module, args: Any) -> None:
        tag = self.by_class.get(type(module).__name__)
        if tag is None:
            return
        if id(module) in self.attached:
            # A root's next forward: every root of this rank has run by now.
            self.stop_finding()
            return
        self.attach(module, tag)
        # The root's own hooks miss this call's entry but not its exit.
        self.enter_root(tag)
        self.missing.discard(type(module).__name__)
        if not self.missing:
            self.stop_finding()

    def stop_finding(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def attach(self, root: nn.Module, tag: str) -> None:
        if _capturing():
            raise RuntimeError(
                f"{FLAG}: root {type(root).__name__} first ran inside a CUDA graph "
                "capture; the ring must be allocated before"
            )
        self.attached.add(id(root))
        if self.ring is None:
            self.allocate(root)
        root.register_forward_pre_hook(lambda m, a: self.enter_root(tag))
        root.register_forward_hook(lambda m, a, o: self.exit_root(tag))
        layer = self.cfg["layer"]
        layer_classes = {layer} if isinstance(layer, str) else set(layer)
        scopes = self.cfg["scopes"]
        for path, module in root.named_modules():
            if type(module).__name__ not in layer_classes:
                continue
            module.register_forward_pre_hook(self.stamper(tag, path, "entry"))
            for name in scopes:
                child = getattr(module, name, None)
                if isinstance(child, nn.Module):
                    child.register_forward_pre_hook(self.stamper(tag, f"{path}.{name}", "entry"))
                    child.register_forward_hook(self.stamper(tag, f"{path}.{name}", "exit"))
        logger.info("Light timing: root %s (%s) attached", tag, type(root).__name__)

    def allocate(self, root: nn.Module) -> None:
        param = next(root.parameters(), None)
        self.device = (
            param.device if param is not None and param.is_cuda
            else torch.device("cuda", torch.cuda.current_device())
        )
        self.ring = torch.zeros((self.rows, self.width), dtype=torch.int64, device=self.device)
        self.counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        with torch.cuda.device(self.device):
            # Build both kernel variants now, outside any capture.
            _launch(self, 0, 0)
            _launch(self, FIRST_COLUMN, 0)
            torch.cuda.synchronize(self.device)
        self.counter.zero_()
        self.ring.zero_()
        os.makedirs(self.cfg["dir"], exist_ok=True)
        threading.Thread(target=self.serve_dumps, name="sm80-light-timing", daemon=True).start()

    # -- hooks ------------------------------------------------------------------

    def enter_root(self, tag: str) -> None:
        self.depth += 1
        if self.depth > 1:
            return
        self.current = tag
        self.stamping = _capturing()
        index = self.tags.index(tag) + 1
        _launch(self, 0, index if self.stamping else -index)

    def exit_root(self, tag: str) -> None:
        self.depth -= 1
        if self.depth > 0:
            return
        if self.stamping:
            _launch(self, self.column(tag, "", "exit"), 0)
        self.current = None
        self.stamping = False

    def stamper(self, tag: str, path: str, edge: str):
        def hook(module, *rest):
            if self.stamping and self.current == tag:
                _launch(self, self.column(tag, path, edge), 0)

        return hook

    def column(self, tag: str, path: str, edge: str) -> int:
        key = (tag, path, edge)
        col = self.columns.get(key)
        if col is None:
            col = FIRST_COLUMN + len(self.columns)
            if col >= self.width:
                raise RuntimeError(
                    f"{FLAG}: width {self.width} holds {self.width - FIRST_COLUMN} "
                    "stamps per forward; a graph needs more"
                )
            self.columns[key] = col
        return col

    # -- dumps ------------------------------------------------------------------

    def request_path(self) -> str:
        return os.path.join(self.cfg["dir"], "request")

    def read_request(self) -> str | None:
        try:
            with open(self.request_path()) as f:
                return f.read().strip() or None
        except FileNotFoundError:
            return None

    def serve_dumps(self) -> None:
        seen = self.read_request()
        while True:
            time.sleep(POLL_S)
            req = self.read_request()
            if req is None or req == seen:
                continue
            seen = req
            try:
                self.dump(req)
            except Exception:
                logger.exception("Light timing: dump %s failed", req)

    def dump(self, name: str) -> str:
        """Write this rank's rows since the previous dump; returns the path."""
        with self.lock:
            host_counter, host_ring = self.copy()
            n = int(host_counter[0])
            start = 0 if self.dumped is None else self.dumped
            lost = max(0, n - start - self.rows)
            first = max(start, n - self.rows)
            idx = np.arange(first, n) % self.rows
            rows = host_ring[idx]
            self.dumped = n
        meta = {
            "rank": _global_rank(),
            "device": str(self.device),
            "tags": self.tags,
            "layer": self.cfg["layer"],
            "scopes": self.cfg["scopes"],
            "width": self.width,
            "columns": [list(k) for k, _ in sorted(self.columns.items(), key=lambda kv: kv[1])],
        }
        out = os.path.join(self.cfg["dir"], f"rank{meta['rank']}.{name}.npz")
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, counter=np.int64(n), first=np.int64(first), lost=np.int64(lost),
                            rows=rows, meta=np.array(json.dumps(meta)))
        os.replace(tmp, out)
        return out

    def copy(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.ring is not None and self.counter is not None
        if self.device is None or self.device.type != "cuda":
            return self.counter.numpy().copy(), self.ring.numpy().copy()
        with torch.cuda.device(self.device):
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                counter = self.counter.to("cpu")
                ring = self.ring.to("cpu")
        return counter.numpy(), ring.numpy()


_timer: LightTimer | None = None


def install() -> bool:
    """Register the root finder when the flag is set; whether it did."""
    global _timer
    import vllm.envs as envs

    text = envs.VLLM_SM80_LIGHT_TIMING
    if not text:
        return False
    if _timer is not None:
        return True
    cfg = parse(text)
    _timer = LightTimer(cfg)
    _timer.handle = nn.modules.module.register_module_forward_pre_hook(_timer.find_roots)
    layer = cfg["layer"]
    logger.warning_once(
        "Light timing on: CUDA-graph forwards of %s stamp layer %s and scopes %s "
        "into a %d x %d ring, dumps on request in %s (%s)",
        ", ".join(f"{t}={c}" for t, c in cfg["roots"].items()),
        layer if isinstance(layer, str) else ", ".join(layer),
        ", ".join(cfg["scopes"]) or "none", cfg["rows"], cfg["width"], cfg["dir"], FLAG,
    )
    return True
