# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic capture of served kernel calls (VLLM_SM80_CAPTURE).

Off by default: with VLLM_SM80_CAPTURE unset ``active`` is False, nothing is
patched and no capture point does any work. With it set to the path of a
capture spec (below), the capture points (the dense EXL3 linear, the EXL3
routed experts, model attention layers, model mHC ops)
hand selected calls to :func:`begin`; the returned :class:`Record` keeps
copies of the call's arguments and outputs, and every kept call becomes one
file

    <dir>/r<rank>/<family>/<prefix-or-key>-M<rows>-<phase>-<owner>-s<step>-<n>.pt
    {"schema": "lab-point-capture/1", "family": <point>,
     "meta": {...}, "tensors": {argument or output: value}}

(``torch.load(..., weights_only=True)``) and one line of
``<dir>/r<rank>/index.jsonl`` (family, point, prefix, rows, phase, owner,
step, bytes, saved, path, sha256). ``point`` names the operator the call
went through (the family id of the FP64 case driver where it has one).

Eager calls (prefill steps, decode steps outside a CUDA graph) copy snapshots
directly to the host at each capture point. FULL CUDA graph calls are recorded
inside the graph: while a graph is captured, a selected call's copies are
captured with it (asynchronous copies into pinned host buffers, in stream
order, so on every replay they hold exactly what the graph's kernels read
and wrote), and after synchronizing a replay the host copies are read. The
graph is the served one plus these copies; nothing it computes changes.
Reading back waits for the stream: diagnostic boots only.

The capture spec is a TOML file; every key but ``dir`` is optional, shown
here with its default, and any other key is refused (:func:`spec`):

    dir = "<directory>"
    families = ["dense", "moe", "attn"]
    match = ""
    rows = []
    per_key = 2
    prefill_match = ""
    prefill_calls = [1]
    prefill_per_key = 0
    prefill_min_rows = 65
    max_mb = 4096

- dir: where the files go; a relative path is taken from the spec's
  directory.
- families: the families recorded (FAMILIES).
- match: a regex searched in the call's prefix (the checkpoint prefix for
  dense matrices, the module prefix otherwise) for every call outside
  prefill steps; empty matches every prefix.
- rows: the row counts whose graph captures get copies.
- per_key: files per (family, prefix, rows, phase, owner, context), where the
  context is ``long`` when some request of the step had computed at least
  LONG_CONTEXT tokens before it, else ``short``.
- prefill_match (empty: match), prefill_calls, prefill_min_rows: in prefill
  steps, the prefixes and the ordinals (per family and prefix, from 1, over
  calls of at least prefill_min_rows rows) that are kept.
- prefill_per_key: default 0 keeps the ordinal policy. Positive N instead
  keeps the first N eligible attempts per (family, point, exact prefix, rows,
  phase, owner, short/long context). Quota refusals also consume an attempt;
  the reached-row census and actual prefill call ordinal continue counting.
  These records use the semantic key hash in their filename so different
  points at the same prefix cannot overwrite one another.
- max_mb: MiB of tensors written per process; past it the index says
  ``saved: false``.

The step phase is the scheduler's, from the step the model runner executes:
a step that schedules some request more than 1 + num_speculative_tokens
tokens is ``prefill``, one token per request ``decode``, otherwise
``verify``; ``meta.query_lens`` holds every request's scheduled tokens. A
record's ``phase`` is the step phase for an eager call; a call replayed in a
CUDA graph gets the graph's own phase (the target model's graphs: ``decode``
for one token per request, else ``verify``; a speculator's graphs:
``draft``), since a drafter graph also runs in prefill steps with draft
rows; ``meta.step_phase`` always holds the step phase. The owner is
``eager`` or the class of the CUDA graph manager that replayed the call,
with the graph's descriptor in ``meta.graph``. Profile runs, warm-ups and dummy steps are never recorded.

Row counts seen, selected or not: ``<dir>/r<rank>/rows_seen.json``
{"schema": "capture-rows-seen/1", "steps", "eager_calls": [{family, phase,
rows, calls}], "graph_replays": [{owner, phase, step_phase, rows,
replays}]}: every eager call of a captured family in a real step, and every
replay of a FULL graph in a real step, rewritten at the start of every
SEEN_EVERY-th step (the last step's own counts land with the next one) and
with every record written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm_sm80.plugin import register_fork_env_vars

register_fork_env_vars()  # read at import below

logger = init_logger(__name__)

SCHEMA = "lab-point-capture/1"
FAMILIES = ("dense", "moe", "attn", "mhc", "hybrid")
# Context, in tokens, from which a step's records count as long-context ones.
LONG_CONTEXT = 4096

active = bool(envs.VLLM_SM80_CAPTURE)

# The capture spec's keys: (type, item type of a list, default); dir has no
# default.
SPEC_KEYS: dict[str, tuple[type, type | None, Any]] = {
    "dir": (str, None, None),
    "families": (list, str, ["dense", "moe", "attn"]),
    "match": (str, None, ""),
    "rows": (list, int, []),
    "per_key": (int, None, 2),
    "prefill_match": (str, None, ""),
    "prefill_calls": (list, int, [1]),
    "prefill_per_key": (int, None, 0),
    "prefill_min_rows": (int, None, 65),
    "max_mb": (int, None, 4096),
}


@dataclass(frozen=True)
class Spec:
    """The settings of a capture spec (module docstring)."""

    path: str
    dir: str
    families: frozenset
    match: str
    rows: frozenset
    per_key: int
    prefill_match: str
    prefill_calls: frozenset
    prefill_per_key: int
    prefill_min_rows: int
    max_mb: int


def _typed(value: Any, kind: type) -> bool:
    # TOML booleans are not integers here.
    return isinstance(value, kind) and not (kind is int and isinstance(value, bool))


def spec(path: str | None = None) -> Spec:
    """The capture spec at ``path`` (default: VLLM_SM80_CAPTURE). An unknown
    key, a value of another type, no ``dir``, an unknown family or a negative
    ``prefill_per_key`` raises ValueError naming the file."""
    path = path or envs.VLLM_SM80_CAPTURE
    where = f"VLLM_SM80_CAPTURE {path}"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = sorted(set(raw) - set(SPEC_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; known: {sorted(SPEC_KEYS)}")
    values = {}
    for key, (kind, item, default) in SPEC_KEYS.items():
        value = raw.get(key, default)
        if value is None or (key == "dir" and not value):
            raise ValueError(f"{where}: {key} is required")
        if not _typed(value, kind) or (
            item is not None and not all(_typed(v, item) for v in value)
        ):
            what = f"a list of {item.__name__}" if item else kind.__name__
            raise ValueError(f"{where}: {key} must be {what}, not {value!r}")
        values[key] = value
    unknown = sorted(set(values["families"]) - set(FAMILIES))
    if unknown:
        raise ValueError(f"{where}: unknown families {unknown}; known: {list(FAMILIES)}")
    if values["prefill_per_key"] < 0:
        raise ValueError(f"{where}: prefill_per_key must be nonnegative")
    base = os.path.dirname(os.path.abspath(path))
    return Spec(
        path=path,
        dir=os.path.join(base, values["dir"]),
        families=frozenset(values["families"]),
        match=values["match"],
        rows=frozenset(values["rows"]),
        per_key=values["per_key"],
        prefill_match=values["prefill_match"],
        prefill_calls=frozenset(values["prefill_calls"]),
        prefill_per_key=values["prefill_per_key"],
        prefill_min_rows=values["prefill_min_rows"],
        max_mb=values["max_mb"],
    )


def captures(family: str) -> bool:
    """Whether VLLM_SM80_CAPTURE is set and its spec records ``family``."""
    return bool(envs.VLLM_SM80_CAPTURE) and family in spec().families


_point_phase = ContextVar("sm80_capture_point_phase", default=None)


@contextmanager
def phase_scope(phase: str):
    """Scope actual model ownership without changing target scheduler metadata."""
    if phase not in ("prefill", "decode", "verify", "draft"):
        raise ValueError(f"unknown capture point phase {phase!r}")
    token = _point_phase.set(phase)
    try:
        yield
    finally:
        _point_phase.reset(token)


@dataclass
class _Config:
    root: str
    families: frozenset
    match: re.Pattern
    prefill_match: re.Pattern
    rows: frozenset
    per_key: int
    prefill_calls: frozenset
    prefill_min_rows: int
    max_bytes: int
    rank: int
    tp_rank: int
    tp_size: int
    prefill_per_key: int


def _ranks() -> tuple[int, int, int]:
    """(global rank, tensor parallel rank, tensor parallel size)."""
    rank = tp_rank = 0
    tp_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
    except (AssertionError, RuntimeError):
        tp_rank = rank
    return rank, tp_rank, tp_size


_cfg: _Config | None = None


def config() -> _Config:
    """The settings, read once in the worker (after it joined its groups,
    at weight loading); the first call installs the hooks."""
    global _cfg
    if _cfg is None:
        settings = spec()
        rank, tp_rank, tp_size = _ranks()
        cfg = _Config(
            root=os.path.join(settings.dir, f"r{rank}"),
            families=settings.families,
            match=re.compile(settings.match),
            prefill_match=re.compile(settings.prefill_match or settings.match),
            rows=settings.rows,
            per_key=settings.per_key,
            prefill_calls=settings.prefill_calls,
            prefill_min_rows=settings.prefill_min_rows,
            max_bytes=settings.max_mb << 20,
            rank=rank,
            tp_rank=tp_rank,
            tp_size=tp_size,
            prefill_per_key=settings.prefill_per_key,
        )
        os.makedirs(cfg.root, exist_ok=True)
        _install(cfg.families)
        _cfg = cfg
        logger.info(
            "Kernel call capture active (VLLM_SM80_CAPTURE=%s): writing %s, "
            "families %s, graph rows %s, %d per key, prefill calls %s, %d "
            "prefill attempts per semantic key (0 uses ordinals), at most %d "
            "MiB; diagnostic boots only.",
            settings.path, cfg.root, ",".join(sorted(cfg.families)),
            sorted(cfg.rows), cfg.per_key, sorted(cfg.prefill_calls),
            cfg.prefill_per_key, settings.max_mb,
        )
    return _cfg


# ------------------------------------------------------------------ step state

# The step the model runner executes (set by its execute_model, kept until the
# next one so the drafter's calls in sample_tokens get the same step).
_step: dict[str, Any] = {
    "real": False, "step": 0, "phase": None, "query_lens": [], "context_lens": [],
}
# Records made while a graph is being captured; the graph manager's storing
# the graph moves them onto the graph object (they live and die with it, so a
# graph torn down after the memory estimate takes its copies with it).
_pending: list[Record] = []
_counts: dict[tuple, int] = {}
_prefill_calls: dict[tuple[str, str], int] = {}
_bytes = [0]
_weights: dict[int, dict] = {}
# Row counts seen (module docstring) and how often they are written.
_seen: dict[tuple, int] = {}
SEEN_EVERY = 1


def _dump_seen() -> None:
    if _cfg is None:
        return
    eager, graphs = [], []
    for key, n in sorted(_seen.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if key[0] == "eager":
            eager.append(dict(zip(("family", "phase", "rows"), key[1:]), calls=n))
        else:
            graphs.append(
                dict(zip(("owner", "phase", "step_phase", "rows"), key[1:]), replays=n)
            )
    path = os.path.join(_cfg.root, "rows_seen.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"schema": "capture-rows-seen/1", "steps": _step["step"],
                   "eager_calls": eager, "graph_replays": graphs}, f, indent=1)
    os.replace(path + ".tmp", path)


def set_step(scheduler_output, decode_len: int, dummy: bool) -> None:
    """Called by the model runner before every step it executes.
    ``context_lens``: the tokens each request had computed before the step
    (new requests first, then the running ones)."""
    if _cfg is not None and "hybrid" in _cfg.families:
        from vllm_sm80.models.deepseek_v41.hybrid_capture import reset_links
        reset_links()
    if dummy:
        _step["real"] = False
        return
    lens = sorted(getattr(scheduler_output, "num_scheduled_tokens", {}).values())
    context = [
        getattr(r, "num_computed_tokens", None)
        for r in getattr(scheduler_output, "scheduled_new_reqs", None) or []
    ]
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    context += list(getattr(cached, "num_computed_tokens", None) or [])
    top = lens[-1] if lens else 0
    phase = "prefill" if top > decode_len else "decode" if top == 1 else "verify"
    longest = max((c for c in context if c is not None), default=0)
    _step.update(real=bool(lens), step=_step["step"] + 1, phase=phase,
                 query_lens=lens, context_lens=context,
                 context="long" if longest >= LONG_CONTEXT else "short")
    if _step["step"] % SEEN_EVERY == 0:
        _dump_seen()


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ------------------------------------------------------------------ records


@dataclass
class Record:
    family: str
    point: str
    prefix: str
    rows: int
    meta: dict
    tensors: dict = field(default_factory=dict)
    in_graph: bool = False
    point_phase: str | None = None
    deferred: list = field(default_factory=list)
    layout_fields: set[str] = field(default_factory=set)

    def put(self, *, preserve_layout: bool = False, **values: Any) -> Record:
        """Snapshot CUDA values directly into host memory in stream order.

        Eager copies use owned pageable memory and complete before returning.
        Graph copies use pinned memory and complete at drain.

        Linked producer outputs may opt into their original strides. Only the
        span from the first to last addressed element is retained, with its
        origin rebased to zero; relative consumer offsets remain valid. This
        includes intervening padding, not the whole backing allocation.
        """
        for name, value in values.items():
            self.layout_fields.discard(name)
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if preserve_layout:
                    self.layout_fields.add(name)
                if value.is_cuda:
                    shape, stride = value.shape, value.stride()
                    # A non-dense D2H copy otherwise packs through a CUDA temporary.
                    # Transfer its flat span and leave any export packing to the host.
                    if not value.is_contiguous():
                        value = _flat_span(value)
                    snapshot = torch.empty_like(value, device="cpu", pin_memory=self.in_graph)
                    snapshot.copy_(value, non_blocking=self.in_graph)
                    value = snapshot.as_strided(shape, stride)
                elif preserve_layout and not value.is_contiguous():
                    value = _flat_span(value).clone().as_strided(value.shape, value.stride())
                else:
                    value = value.clone()
            self.tensors[name] = value
        return self

    def defer(self, materialize) -> Record:
        """Resolve model-owned state after this eager call/graph replay.

        The adapter must retain pre-write values for state the call mutates.
        Never defer an ephemeral intermediate or a state another graph can
        mutate before this record drains. Callbacks return flat tensor fields;
        no callback is serialized. This avoids graph-sized copies of unused
        paged cache capacity and duplicate table aliases.
        """
        self.deferred.append(materialize)
        return self

    def materialize(self) -> dict:
        values = {k: _host(v, preserve_layout=k in self.layout_fields)
                  for k, v in self.tensors.items()}
        for materialize in self.deferred:
            extra = materialize()
            overlap = values.keys() & extra.keys()
            if overlap:
                raise ValueError(f'duplicate deferred capture fields: {overlap}')
            values.update({k: _host(v) for k, v in extra.items()})
        return values

    def done(self) -> None:
        if self.in_graph:
            _pending.append(self)
        else:
            _write([self], _step["phase"], "eager", None)


def begin(
    family: str, point: str, prefix: str, rows: int,
    point_phase: str | None = None, **meta: Any
) -> Record | None:
    """A Record for this call when it is selected, else None (decided before
    anything is copied). ``point_phase`` is an explicit model-owned phase,
    independent of the target scheduler phase (for example an eager draft
    called while the target is prefilling). It is never inferred from rows."""
    if point_phase is None:
        point_phase = _point_phase.get()
    if point_phase not in (None, "prefill", "decode", "verify", "draft"):
        raise ValueError(f"unknown capture point phase {point_phase!r}")
    cfg = config()
    if family not in cfg.families:
        return None
    if _capturing():
        if rows not in cfg.rows or not cfg.match.search(prefix):
            return None
        return Record(family, point, prefix, rows, meta, in_graph=True, point_phase=point_phase)
    if not _step["real"]:
        return None
    phase = point_phase or _step["phase"]
    seen = ("eager", family, phase, rows)
    _seen[seen] = _seen.get(seen, 0) + 1
    if phase == "prefill":
        if rows < cfg.prefill_min_rows or not cfg.prefill_match.search(prefix):
            return None
        key = (family, prefix)
        _prefill_calls[key] = _prefill_calls.get(key, 0) + 1
        if cfg.prefill_per_key:
            if _full(family, point, prefix, rows, phase, "eager"):
                return None
        elif _prefill_calls[key] not in cfg.prefill_calls:
            return None
        meta = dict(meta, prefill_call=_prefill_calls[key])
    elif not cfg.match.search(prefix) or _full(
        family, point, prefix, rows, phase, "eager"
    ):
        return None
    return Record(family, point, prefix, rows, meta, point_phase=point_phase)


def _key(
    family: str, point: str, prefix: str, rows: int, phase: str, owner: str
) -> tuple:
    key = (family, prefix, rows, phase, owner, _step.get("context"))
    if phase == "prefill" and config().prefill_per_key:
        key += (point,)
    return key


def _full(
    family: str, point: str, prefix: str, rows: int, phase: str, owner: str
) -> bool:
    cfg = config()
    if phase == "prefill":
        if not cfg.prefill_per_key:
            return False
        limit = cfg.prefill_per_key
    else:
        limit = cfg.per_key
    key = _key(family, point, prefix, rows, phase, owner)
    return _counts.get(key, 0) >= limit


def _flat_span(value: torch.Tensor) -> torch.Tensor:
    """A view of the minimum storage span, starting at this tensor's origin."""
    count = 0 if not value.numel() else 1 + sum(
        (size - 1) * stride for size, stride in zip(value.shape, value.stride())
    )
    return value.as_strided((count,), (1,))


def _host(value: Any, *, preserve_layout: bool = False) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    # Graph replay overwrites pinned snapshots; materialized records own their data.
    copy = value.device.type == "cpu" and value.is_pinned()
    if preserve_layout and not value.is_contiguous():
        return _flat_span(value).to(device="cpu", copy=copy).as_strided(value.shape, value.stride())
    return value.to(device="cpu", copy=copy or not value.is_contiguous(),
                    memory_format=torch.contiguous_format)


def _write(
    records: list[Record], phase: str, owner: str, graph: dict | None
) -> None:
    cfg = config()
    sm_count = (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if torch.cuda.is_available()
        else 0
    )
    lines = []
    for rec in records:
        record_phase = rec.point_phase or phase
        if _full(rec.family, rec.point, rec.prefix, rec.rows, record_phase, owner):
            continue
        key = _key(rec.family, rec.point, rec.prefix, rec.rows, record_phase, owner)
        bounded_prefill = record_phase == "prefill" and cfg.prefill_per_key > 0
        n = _counts.get(key, 0)
        tensors = rec.materialize()
        size = sum(
            t.untyped_storage().nbytes() if name in rec.layout_fields else t.numel() * t.element_size()
            for name, t in tensors.items()
            if isinstance(t, torch.Tensor)
        )
        line = {
            "family": rec.family, "point": rec.point, "prefix": rec.prefix,
            "rows": rec.rows, "phase": record_phase, "step_phase": _step["phase"],
            "owner": owner,
            "step": _step["step"], "context": _step.get("context"), "bytes": size,
            "saved": False,
        }
        if _bytes[0] + size <= cfg.max_bytes:
            meta = dict(
                rec.meta, module_prefix=rec.prefix, rank=cfg.rank,
                tp_rank=cfg.tp_rank, tp_size=cfg.tp_size, phase=record_phase,
                step_phase=_step["phase"], owner=owner,
                step=_step["step"], query_lens=list(_step["query_lens"]),
                context_lens=list(_step["context_lens"]),
                rows=rec.rows, graph=graph, sm_count=sm_count,
            )
            name = re.sub(r"[^A-Za-z0-9_.]", "_", rec.prefix)
            if bounded_prefill:
                name = hashlib.sha256(repr(key).encode()).hexdigest()
            folder = os.path.join(cfg.root, rec.family)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(
                folder,
                f"{name}-M{rec.rows}-{record_phase}-{owner}-s{_step['step']}-{n}.pt",
            )
            payload = {"schema": SCHEMA, "family": rec.point, "meta": meta,
                       "tensors": tensors}
            torch.save(payload, path + ".tmp")
            os.replace(path + ".tmp", path)
            with open(path, "rb") as f:
                line["sha256"] = hashlib.sha256(f.read()).hexdigest()
            line.update(saved=True, path=path)
            _bytes[0] += size
        if line["saved"] or bounded_prefill:
            _counts[key] = n + 1
        lines.append(json.dumps(line) + "\n")
    if lines:
        with open(os.path.join(cfg.root, "index.jsonl"), "a") as f:
            f.writelines(lines)
        _dump_seen()


# ------------------------------------------------------------------ weights


def register_weight(key: int, **meta: Any) -> None:
    """Facts about a weight (keyed by its device address) that the records
    of its calls carry."""
    _weights[key] = meta


def weight(key: int) -> dict | None:
    return _weights.get(key)


def sha256_of(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ hooks


class _BindingGraphs(dict):
    """A graph manager's ``graphs``: storing a graph binds the records made
    while it was captured to the graph."""

    def __setitem__(self, desc, graph):
        super().__setitem__(desc, graph)
        graph._sm80_capture_records = list(_pending)
        _pending.clear()


def _bind(manager) -> None:
    if _cfg is not None and "hybrid" in _cfg.families:
        from vllm_sm80.models.deepseek_v41.hybrid_capture import reset_links
        reset_links()
    if not isinstance(manager.graphs, _BindingGraphs):
        manager.graphs = _BindingGraphs(manager.graphs)
    # Copies made by a capture that never stored its graph go with it.
    _pending.clear()


def _graph_phase(manager, desc) -> str:
    """The phase of the calls a FULL graph replays: ``draft`` for a
    speculator's graphs, else ``decode`` (one token per request) or
    ``verify``."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    if not isinstance(manager, ModelCudaGraphManager):
        return "draft"
    return "decode" if desc.uniform_token_count == 1 else "verify"


def _drain(manager, desc) -> None:
    if not _step["real"]:
        return
    owner = type(manager).__name__
    phase = _graph_phase(manager, desc)
    seen = ("graph", owner, phase, _step["phase"], desc.num_tokens)
    _seen[seen] = _seen.get(seen, 0) + 1
    records = getattr(manager.graphs.get(desc), "_sm80_capture_records", None)
    if not records:
        return
    if all(
        _full(r.family, r.point, r.prefix, r.rows, r.point_phase or phase, owner)
        for r in records
    ):
        return
    torch.cuda.current_stream().synchronize()
    graph = {
        "num_tokens": desc.num_tokens,
        "num_reqs": desc.num_reqs,
        "uniform_token_count": desc.uniform_token_count,
        "max_query_len": desc.max_query_len,
        "mode": str(desc.cg_mode),
    }
    _write(records, phase, owner, graph)


def _install(families: frozenset) -> None:
    """Patch the model runner (the step's phase) and the CUDA graph managers
    (records inside graphs)."""
    from vllm_sm80.kernels.exl3.step_phase import max_decode_query_len
    from vllm.v1.worker.gpu import cudagraph_utils, model_runner

    manager = cudagraph_utils.CudaGraphManager
    capture, run = manager.capture, manager.run_fullgraph

    def capture_binding(self, *args, **kwargs):
        _bind(self)
        return capture(self, *args, **kwargs)

    def run_draining(self, desc):
        result = run(self, desc)
        _drain(self, desc)
        return result

    manager.capture = capture_binding
    manager.run_fullgraph = run_draining

    runner = model_runner.GPUModelRunner
    execute = runner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        set_step(scheduler_output, max_decode_query_len(self.vllm_config), bool(dummy))
        return execute(self, scheduler_output, *args, **kwargs)

    runner.execute_model = execute_model
    if "hybrid" in families:
        from vllm_sm80.models.deepseek_v41.hybrid_capture import install

        install()
