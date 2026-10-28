# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffer-lifecycle contracts of the DeepSeek-V4.1 sm_80 port, on CPU.

Drives the port's stateful buffers through a seeded generated request history
with tiny shapes and no checkpoint, checks their lifecycle invariants and
writes one contract report (schema 1: group, seed, identity, checks,
coverage, events, failures)::

    python -m tests.models.deepseek_v41.sm80.lifecycle_contracts \\
        --group compressor_ring --seed 1 --out report.json

Groups:

Only invariants a serving run would not show at once are checked: reads of
rows another request (or nobody) wrote, after blocks are freed, recycled or
shared; the compressor ring carry across chunks, steps and preemption; stale
rows after a cache eviction or an early staging reuse.

* ``compressor_ring`` -- the ratio-2 compressor state ring
  (``CompressorStateCache`` capacity, the ring slot mapping kernel): the
  sm_80 ``compress`` carry of a group's first row across chunk and step
  boundaries reads the request's own row, through draft rejection, ring
  block reuse and preemption.
* ``paged_kv`` -- the window rows (window 128), the compressed rows of ratio
  2 and ratio 1 stacks and the index-key rows: every row the window index
  kernel, the indexer and the sparse attention read is the reading request's,
  for that position, written after its block was last allocated; blocks are
  freed behind the window, recycled, shared by prefix hits, at ids whose
  element offsets pass 2**31.
* ``engram_staging`` -- ``Sm80EngramEmbedding`` lookups of every TP rank,
  eager and captured (the one fixed staging of every graph), with the device
  side emulated by an in-order queue: the copies, the native host fetch
  (``EngramHostFetch``, queued as a host function) and the dequant run when
  the queue reaches them; the host only waits through the layer's events.
  Every staging (device ids, host ids, host rows, device rows) is read only
  while it holds the reading lookup's data, the native row cache follows the
  set-associative LRU (hits, misses, inserts, evictions per fetch) and serves
  the table's bytes, and rows it evicted come back as the table's rows.

The attention groups run the history's requests through the real sm_80
attention core (``sm80_attn_core.forward_layer``), with the ring slot mapping
and window index kernels in the Triton interpreter. A history has arrivals,
prefill chunks of uneven size, decode steps with drafts and partial
acceptance, finishes, aborts, preemption (every block freed, the request
recomputed later), prefix-cache hits on full blocks, window blocks freed
behind the window and blocks recycled from a free queue. Reads are checked by
the provenance of the row they address (rows hold random values before first
use); compressed rows, and every row and output of a recomputed request, are
compared bitwise with the same request run alone and uninterrupted (fresh
caches, one prefill chunk, one token per decode step).
"""

import os

# The ring slot mapping and window index kernels run on CPU in the Triton
# interpreter; it must be chosen before any kernel is defined.
os.environ["TRITON_INTERPRET"] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter, deque  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

from tests.models.deepseek_v41.sm80.dsv41_ref import (  # noqa: E402
    LayerSpec,
    RefArgs,
    core_config,
    core_weights,
    layer_inputs,
    make_weights,
)
from tests.models.deepseek_v41.sm80.test_engram_storage import (  # noqa: E402
    DIM as ENGRAM_DIM,
)
from tests.models.deepseek_v41.sm80.test_engram_storage import (  # noqa: E402
    _LruModel,
    _write_table,
)
from vllm.models.deepseek_v41 import compressor as compressor_mod  # noqa: E402
from vllm_sm80.models.deepseek_v41 import sm80_engram  # noqa: E402
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core  # noqa: E402
from vllm_sm80.models.deepseek_v41.ops import sm80_engram_dequant  # noqa: E402
from vllm_sm80.plugin import register_fork_env_vars  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.attention.backends.mla import sparse_swa  # noqa: E402

# The Engram host fetch reads the plugin's variables (VLLM_SM80_CAPTURE) from
# vllm.envs, where a served process has them from the plugin's entry points.
register_fork_env_vars()

BF16 = torch.bfloat16
ROOT = Path(__file__).resolve().parents[4]
# Where a CODE_FILES path lives: upstream's package in the engine tree, the
# plugin package in the plugin tree, else beside these tests (tests/, csrc/).
# All three are ROOT in a single tree; an assembled engine splits them.
PACKAGE_ROOTS = {
    "vllm/": Path(compressor_mod.__file__).resolve().parents[3],
    "vllm_sm80/": Path(sm80_engram.__file__).resolve().parents[3],
}
WINDOW = 128
SWA_BLOCK = 8  # tokens per window block
COMP_BLOCK_TOKENS = 16  # tokens per compressed / index-key block
NUM_SPEC = 3  # drafts per decode step (the served speculative depth)
MAX_RUNNING = 4
PAD_ROWS = 2  # CUDA-graph padding rows of every step
OFFSET_FLOOR = 2**31  # element offset every block of the history passes
MAX_FAILURES = 20  # recorded failures per check

# Only invariants a serving run does not show at once: a request reading
# rows another request (or nobody) wrote, after blocks are freed, recycled or
# shared, and the ring carry across chunks, steps and preemption.
ATTN_CHECKS = (
    "ring_carry_reads_own_row",
    "compressed_rows_match_uninterrupted",
    "recompute_equals_uninterrupted",
    "window_reads_owned",
    "compressed_reads_owned",
    "freed_blocks_never_read",
    "recycled_blocks_rewritten_before_read",
)
ATTN_COVERAGE = (
    "requests",
    "steps",
    "chunk_boundaries",
    "drafts_rejected",
    "aborts",
    "preemptions",
    "recomputes",
    "prefix_hits",
    "recycled_blocks",
    "window_blocks_freed",
    "ring_carries",
    "ring_block_reuses",
    "window_reads",
    "compressed_reads",
    "recompute_rows_compared",
)
# Rows a lookup returns after the row cache evicted them, and the host
# staging refilled only after the device has read it.
ENGRAM_CHECKS = (
    "staging_reuse_ordered",
    "cache_entries_match_table",
    "no_stale_row_after_eviction",
)
ENGRAM_COVERAGE = (
    "lookups",
    "evictions",
    "set_conflicts",
    "refetch_after_eviction",
    "staging_waits",
    "captured_lookups",
)


@dataclass(frozen=True)
class HistoryProfile:
    """What a seeded history emphasises."""

    stacks: dict  # name -> [LayerSpec]
    requests: int
    prompt: tuple[int, int]
    decode: tuple[int, int]
    chunks: tuple[int, ...]  # prefill chunk caps
    budget: int  # tokens per step
    pools: dict  # kind -> blocks
    preempt: tuple[str, ...]  # forced preemptions by phase
    abort: tuple[str, ...]  # forced aborts by phase
    prefix_requests: int
    arrive_span: int


PROFILES = {
    "compressor_ring": HistoryProfile(
        stacks={"ratio2": [LayerSpec(2, kv_source=True, index_source=True)]},
        requests=6,
        prompt=(40, 150),
        decode=(10, 28),
        chunks=(1, 3, 5, 7, 9, 13, 17, 31),
        budget=48,
        pools={"swa": 96, "comp": 48, "idxk": 48, "ring": 3},
        preempt=("prefill", "decode", "decode"),
        abort=("decode",),
        prefix_requests=1,
        arrive_span=12,
    ),
    "paged_kv": HistoryProfile(
        stacks={
            "window_ratio2": [
                LayerSpec(0),
                LayerSpec(2, kv_source=True, index_source=True),
            ],
            "ratio1_candidates": [
                LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
                LayerSpec(1),
                LayerSpec(1, index_source=True, uses_candidates=True),
            ],
        },
        requests=6,
        prompt=(60, 260),
        decode=(6, 16),
        chunks=(7, 16, 24, 40, 64),
        budget=96,
        pools={"swa": 72, "comp": 44, "idxk": 44, "ring": 5},
        preempt=("prefill", "decode"),
        abort=("prefill",),
        prefix_requests=2,
        arrive_span=20,
    ),
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Record:
    """Checks, coverage, events and failures of one report."""

    def __init__(self, checks, coverage):
        self.checks = dict.fromkeys(checks, True)
        self.coverage = Counter(dict.fromkeys(coverage, 0))
        self.events = []
        self.failures = []
        self._per_check = Counter()

    def fail(self, check: str, **detail) -> None:
        self.checks[check] = False
        self._per_check[check] += 1
        if self._per_check[check] <= MAX_FAILURES:
            self.failures.append(
                {"check": check, **{k: _jsonable(v) for k, v in detail.items()}}
            )

    def expect(self, ok: bool, check: str, **detail) -> bool:
        if not ok:
            self.fail(check, **detail)
        return ok

    def event(self, input: dict, transition: dict) -> None:
        self.events.append(
            {
                "input": {k: _jsonable(v) for k, v in input.items()},
                "transition": {k: _jsonable(v) for k, v in transition.items()},
            }
        )

    def report(self, group: str, seed: int) -> dict:
        return {
            "schema": 1,
            "group": group,
            "seed": seed,
            "identity": identity(group),
            "checks": self.checks,
            "coverage": dict(self.coverage),
            "events": self.events,
            "failures": self.failures,
        }


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


CODE_FILES = (
    "tests/models/deepseek_v41/sm80/lifecycle_contracts.py",
    "tests/models/deepseek_v41/sm80/dsv41_ref.py",
    "vllm/models/deepseek_v41/compressor.py",
    "vllm_sm80/models/deepseek_v41/ops/sm80_attn_core.py",
    "vllm_sm80/models/deepseek_v41/ops/sm80_act_quant.py",
    "vllm/v1/attention/backends/mla/sparse_swa.py",
    "vllm_sm80/models/deepseek_v41/sm80_engram.py",
    "csrc/exl3/engram_fetch.cpp",
    "vllm_sm80/models/deepseek_v41/ops/sm80_engram_dequant.py",
)


def identity(group: str) -> dict:
    """The code under test: its files' digest, the shapes of the history."""
    h = hashlib.sha256()
    for rel in CODE_FILES:
        root = next((r for p, r in PACKAGE_ROOTS.items() if rel.startswith(p)),
                    ROOT)
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes())
    ident = {
        "producer": CODE_FILES[0],
        "code_sha256": h.hexdigest(),
        "torch": torch.__version__,
    }
    if group in PROFILES:
        ident.update(window=str(WINDOW), num_speculative_tokens=str(NUM_SPEC))
    return ident


def _stable_seed(*parts) -> int:
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


# ---------------------------------------------------------------------------
# Requests and block pools
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Request:
    rid: int
    stream: int
    prompt: int
    total: int
    arrive: int = 0
    donor: "Request | None" = None
    share: int = 0  # leading positions whose content is the donor's
    status: str = "future"
    length: int = 0  # positions computed and accepted
    known: int = 0  # positions whose content is known (prompt, accepted)
    tables: dict = field(default_factory=dict)  # kind -> {logical: block}
    ring: int = 0  # 0: the null block
    shared: set = field(default_factory=set)  # (kind, block) from prefix hits
    preempted: bool = False

    def key(self, p: int) -> tuple:
        """The final content of position ``p``."""
        if self.donor is not None and p < self.share:
            return self.donor.key(p)
        return (self.stream, p, 0)


class Pool:
    """Blocks of one cache kind: a free queue (taken from the head, freed to
    the tail), reference counts, and the prefix registry of full blocks,
    which stay registered on the free queue until they are taken again."""

    def __init__(self, base: int, size: int, rng: random.Random):
        ids = list(range(base, base + size))
        rng.shuffle(ids)
        self.base, self.size = base, size
        self.free = deque(ids)
        self.ref: dict[int, int] = {}
        self.used: set[int] = set()
        self.key_of: dict[int, tuple] = {}
        self.by_key: dict[tuple, int] = {}
        self.alloc_tick: dict[int, int] = {}
        self.recycled = 0

    def take(self, tick: int) -> int:
        b = self.free.popleft()
        key = self.key_of.pop(b, None)
        if key is not None:
            del self.by_key[key]
        if b in self.used:
            self.recycled += 1
        self.used.add(b)
        self.ref[b] = 1
        self.alloc_tick[b] = tick
        return b

    def hit(self, key: tuple) -> int:
        b = self.by_key[key]
        if b not in self.ref:
            self.free.remove(b)
            self.ref[b] = 0
        self.ref[b] += 1
        return b

    def give(self, b: int) -> None:
        self.ref[b] -= 1
        if self.ref[b] == 0:
            del self.ref[b]
            self.free.append(b)

    def register(self, b: int, key: tuple) -> None:
        if b in self.key_of or key in self.by_key:
            return
        self.key_of[b] = key
        self.by_key[key] = b


# ---------------------------------------------------------------------------
# Paged attention caches and one step
# ---------------------------------------------------------------------------


def ring_capacity() -> int:
    """The ring rows per request of ``CompressorStateCache`` at the served
    speculative depth."""
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        num_speculative_tokens=NUM_SPEC,
    )
    with mock.patch.object(compressor_mod, "get_current_vllm_config", lambda: config):
        cache = compressor_mod.CompressorStateCache(
            state_dim=2 * RefArgs().head_dim, dtype=torch.float32, prefix="ring"
        )
    return cache.block_size


class Stack:
    """One stack of attention layers sharing a compressed / index-key source,
    its paged caches and the checks of every step it runs."""

    def __init__(
        self,
        name: str,
        specs: list,
        seed: int,
        pools: dict,
        high: bool,
        rng: random.Random,
        rec: Record | None,
        oracle: dict,
    ):
        self.name, self.specs, self.seed = name, specs, seed
        self.args = RefArgs(window_size=WINDOW)
        self.rec, self.oracle = rec, oracle  # rec None: record into oracle
        self.weights = [
            make_weights(self.args, s, _stable_seed(seed, name, li) % 2**31)
            for li, s in enumerate(specs)
        ]
        self.cfgs = [core_config(self.args, s) for s in specs]
        self.cws = [
            core_weights(self.args, s, w, "cpu") for s, w in zip(specs, self.weights)
        ]
        self.ratio = max(s.ratio for s in specs)
        self.comp_e = COMP_BLOCK_TOKENS // self.ratio if self.ratio else 0
        self.cap = ring_capacity()
        a = self.args
        shapes = {"swa": ((SWA_BLOCK, a.head_dim), BF16)}
        if self.ratio:
            shapes["comp"] = ((self.comp_e, a.head_dim), BF16)
            shapes["idxk"] = ((self.comp_e, a.index_head_dim), BF16)
        if self.ratio > 1:
            shapes["ring"] = ((self.cap, 2 * a.head_dim), torch.float32)
        self.pools, self.tensors = {}, {}
        g = torch.Generator().manual_seed(_stable_seed(seed, name, "garbage") % 2**31)
        for kind, (tail, dtype) in shapes.items():
            per_block = tail[0] * tail[1]
            base = OFFSET_FLOOR // per_block + 1 + rng.randrange(4096) if high else 1
            pool = Pool(base, pools[kind], rng)
            self.pools[kind] = pool
            names = [f"swa{li}" for li in range(len(specs))] if kind == "swa" else [kind]
            for n in names:
                # Untouched pages below the pool stay unallocated.
                t = torch.empty((base + pool.size, *tail), dtype=dtype)
                t[:base].view(-1)[: per_block].zero_()  # the null block
                t[base:] = (torch.randn((pool.size, *tail), generator=g) * 4).to(dtype)
                self.tensors[n] = t
        self.prov = {n: {} for n in self.tensors}  # name -> {(block, off): key}
        self.wrote = {n: {} for n in self.tensors}  # name -> {(block, off): tick}
        self.topk = torch.full((512, a.index_topk), -7, dtype=torch.int32)
        self.candidates = torch.full(
            (512, a.candidate_topk_blocks), -7, dtype=torch.int32
        )
        self.tick = 0

    # -- helpers ----------------------------------------------------------

    def kind_of(self, name: str) -> str:
        return "swa" if name.startswith("swa") else name

    def x_row(self, li: int, key: tuple) -> torch.Tensor:
        g = torch.Generator().manual_seed(_stable_seed(self.seed, self.name, li, key) % 2**31)
        return torch.randn(self.args.dim, generator=g).to(BF16)

    def caches(self, li: int) -> core.Sm80AttnCaches:
        spec = self.specs[li]
        return core.Sm80AttnCaches(
            swa=self.tensors[f"swa{li}"],
            topk=self.topk,
            comp=self.tensors["comp"] if spec.ratio else None,
            idxk=self.tensors["idxk"] if spec.ratio else None,
            ring=self.tensors["ring"] if spec.ratio > 1 else None,
            candidates=self.candidates,
        )

    def comp_kinds(self) -> list[str]:
        return ["comp", "idxk"] if self.ratio else []

    # -- blocks -----------------------------------------------------------

    def needs(self, req: Request, start: int, n: int) -> Counter:
        need = Counter()
        sw = req.tables.setdefault("swa", {})
        need["swa"] = len({p // SWA_BLOCK for p in range(start, start + n)} - set(sw))
        for kind in self.comp_kinds():
            t = req.tables.setdefault(kind, {})
            need[kind] = len(
                {p // COMP_BLOCK_TOKENS for p in range(start, start + n)} - set(t)
            )
        return need

    def can_allocate(self, need: Counter) -> bool:
        return all(len(self.pools[k].free) >= n for k, n in need.items())

    def allocate(self, req: Request, start: int, n: int) -> None:
        self.tick += 1
        for p in range(start, start + n):
            for kind, size in [("swa", SWA_BLOCK)] + [
                (k, COMP_BLOCK_TOKENS) for k in self.comp_kinds()
            ]:
                t = req.tables.setdefault(kind, {})
                if p // size not in t:
                    t[p // size] = self.pools[kind].take(self.tick)

    def release(self, req: Request) -> dict:
        freed = Counter()
        for kind, t in req.tables.items():
            for b in t.values():
                self.pools[kind].give(b)
                freed[kind] += 1
            t.clear()
        if req.ring:
            self.pools["ring"].give(req.ring)
            freed["ring"] += 1
            req.ring = 0
        req.shared.clear()
        return dict(freed)

    def hit_length(self, req: Request) -> int:
        """Longest block-aligned prefix (< known) whose full blocks are
        registered: compressed and index-key blocks from position 0, window
        blocks covering the first computed position's window."""
        best = 0
        top = (req.known - 1) // COMP_BLOCK_TOKENS * COMP_BLOCK_TOKENS
        for h in range(top, 0, -COMP_BLOCK_TOKENS):
            if all(
                self.block_key(req, k, b) in self.pools[k].by_key
                for k, b in self.hit_blocks(req, h)
            ):
                best = h
                break
        return best

    def hit_blocks(self, req: Request, h: int) -> list[tuple[str, int]]:
        out = [(k, b) for k in self.comp_kinds() for b in range(h // COMP_BLOCK_TOKENS)]
        lo = max(0, h - WINDOW + 1) // SWA_BLOCK
        out += [("swa", b) for b in range(lo, (h - 1) // SWA_BLOCK + 1)]
        return out

    def block_key(self, req: Request, kind: str, b: int) -> tuple:
        size = SWA_BLOCK if kind == "swa" else COMP_BLOCK_TOKENS
        return (kind,) + tuple(req.key(q) for q in range(b * size, (b + 1) * size))

    def take_hit(self, req: Request, h: int) -> int:
        for kind, b in self.hit_blocks(req, h):
            blk = self.pools[kind].hit(self.block_key(req, kind, b))
            req.tables.setdefault(kind, {})[b] = blk
            req.shared.add((kind, blk))
        return len(self.hit_blocks(req, h)) if h else 0

    def register_full(self, req: Request, start: int) -> None:
        for kind, size in [("swa", SWA_BLOCK)] + [
            (k, COMP_BLOCK_TOKENS) for k in self.comp_kinds()
        ]:
            t = req.tables.get(kind, {})
            for b in range(start // size, req.length // size):
                if b in t:
                    self.pools[kind].register(t[b], self.block_key(req, kind, b))

    def free_behind_window(self, req: Request) -> int:
        t = req.tables.get("swa", {})
        gone = [b for b in t if (b + 1) * SWA_BLOCK <= req.length - WINDOW]
        for b in gone:
            blk = t.pop(b)
            self.pools["swa"].give(blk)
            req.shared.discard(("swa", blk))
        return len(gone)

    # -- one step ---------------------------------------------------------

    def step(self, sched: list, step_no: int) -> dict:
        """Run one batch: ``sched`` lists ``(request, start, keys, accepted)``
        where ``keys`` are the contents of positions ``start..``."""
        rec, a = self.rec, self.args
        self.tick += 1
        R = len(sched)
        pos, t2r, keys_t, qsl = [], [], [], [0]
        for i, (req, start, keys, _) in enumerate(sched):
            for k, key in enumerate(keys):
                pos.append(start + k)
                t2r.append(i)
                keys_t.append(key)
            qsl.append(len(pos))
        T = len(pos)
        Tp = T + PAD_ROWS

        def ckey(i: int, q: int) -> tuple:
            req, start, keys, _ = sched[i]
            if start <= q < start + len(keys):
                return keys[q - start]
            return req.key(q)

        def comp_key(i: int, j: int) -> tuple:
            r = self.ratio
            return tuple(ckey(i, q) for q in range(j * r, (j + 1) * r))

        def final(t: int) -> bool:
            req, start, _, acc = sched[t2r[t]]
            return pos[t] < start + acc

        def table(kind: str) -> torch.Tensor:
            nb = 1 + max([0] + [max(r.tables.get(kind, {}) or [0]) for r, *_ in sched])
            out = torch.zeros(R, nb, dtype=torch.int32)
            for i, (req, *_) in enumerate(sched):
                for lb, blk in req.tables.get(kind, {}).items():
                    out[i, lb] = blk
            return out

        pos_t = torch.zeros(Tp, dtype=torch.int64)
        pos_t[:T] = torch.tensor(pos, dtype=torch.int64)
        t2r_t = torch.zeros(Tp, dtype=torch.int32)
        t2r_t[:T] = torch.tensor(t2r, dtype=torch.int32)
        qsl_t = torch.tensor(qsl, dtype=torch.int32)
        swa_bt = table("swa")
        swa_slots = torch.tensor(
            [sched[t2r[t]][0].tables["swa"][p // SWA_BLOCK] * SWA_BLOCK + p % SWA_BLOCK
             for t, p in enumerate(pos)],
            dtype=torch.int64,
        )
        # The upstream window index kernel, as the window metadata builder
        # launches it (no image visibility, no bounded replay).
        swa_idx = torch.full((Tp, WINDOW), -7, dtype=torch.int32)
        swa_lens = torch.zeros(Tp, dtype=torch.int32)
        valid = torch.zeros(Tp, dtype=torch.bool)
        valid[:T] = True
        seq_lens = torch.tensor(
            [start + len(keys) for _, start, keys, _ in sched], dtype=torch.int32
        )
        sparse_swa._compute_swa_indices_and_lens_kernel[(Tp,)](
            swa_idx, swa_idx.stride(0), swa_lens, WINDOW, WINDOW, swa_lens, swa_lens,
            qsl_t, seq_lens, t2r_t, valid, swa_bt, swa_bt.stride(0), SWA_BLOCK,
            torch.zeros(R, dtype=torch.int32), 0,
            HAS_IMAGE=False, TRITON_BLOCK_SIZE=1024,
        )
        batch = core.Sm80AttnBatch(
            positions=pos_t[:T],
            token_to_req=t2r_t[:T],
            query_start_loc=qsl_t,
            swa_slots=swa_slots,
            swa_indices=swa_idx[:T],
            max_seq_len=int(seq_lens.max()),
        )
        if self.ratio:
            e, r = self.comp_e, self.ratio
            for kind in self.comp_kinds():
                slots = [
                    sched[t2r[t]][0].tables[kind][(p // r) // e] * e + (p // r) % e
                    if (p + 1) % r == 0 else -1
                    for t, p in enumerate(pos)
                ]
                setattr(batch, f"{kind}_slots", torch.tensor(slots, dtype=torch.int64))
                setattr(batch, f"{kind}_block_table", table(kind))
        if self.ratio > 1:
            ring_bt = torch.tensor([[req.ring] for req, *_ in sched], dtype=torch.int32)
            ring_slots = torch.empty(Tp, dtype=torch.int64)
            compressor_mod._ring_slot_mapping_kernel[(triton.cdiv(Tp, 256),)](
                ring_slots, ring_bt, ring_bt.stride(0), t2r_t, pos_t, T, Tp,
                CAPACITY=self.cap, BLOCK=256,
            )
            batch.ring_slots = ring_slots[:T]

        transition = Counter()
        g = torch.Generator().manual_seed(_stable_seed(self.seed, self.name, "pad", step_no) % 2**31)
        h = a.n_heads // a.world_size
        for li, spec in enumerate(self.specs):
            names = [f"swa{li}"]
            if spec.kv_source:
                names += ["comp"] + (["idxk"] if spec.index_source else [])
            if spec.kv_source and spec.ratio > 1 and rec is not None:
                self.check_ring_carry(li, spec, sched, ckey, step_no)
            x = torch.stack([self.x_row(li, key) for key in keys_t])
            qr_kv, kv_score, iw = layer_inputs(a, spec, self.weights[li], x)
            qr_kv = torch.cat([qr_kv, torch.randn(PAD_ROWS, qr_kv.shape[1], generator=g).to(BF16)])
            if kv_score is not None:
                kv_score = torch.cat(
                    [kv_score, torch.randn(PAD_ROWS, kv_score.shape[1], generator=g)]
                )
            if iw is not None:
                iw = torch.cat([iw, torch.randn(PAD_ROWS, iw.shape[1], generator=g).to(BF16)])
            out = torch.full((Tp, h, a.head_dim), 3.0, dtype=BF16)
            core.forward_layer(
                self.cfgs[li], self.cws[li], self.caches(li), batch, qr_kv, kv_score, iw, out
            )
            writes = self.expected_writes(li, spec, sched, batch, pos, t2r, keys_t, comp_key)
            for n in names:
                self.check_writes(n, writes.get(n, {}), sched, pos, final, step_no,
                                  transition)
            if rec is None:
                self.record_outputs(li, sched, pos, t2r, out, final)
                continue
            self.check_window_reads(li, sched, swa_idx, pos, t2r, T, ckey, step_no, transition)
            if spec.ratio:
                self.check_compressed_reads(li, spec, sched, batch, pos, t2r, comp_key,
                                            step_no, transition)
            self.check_outputs(li, sched, pos, t2r, out, final, step_no)
        return dict(transition)

    def expected_writes(self, li, spec, sched, batch, pos, t2r, keys_t, comp_key) -> dict:
        """{cache name: {(block, offset): (key, token, request index)}}."""
        out = {}
        sw = {}
        for t, slot in enumerate(batch.swa_slots.tolist()):
            sw[(slot // SWA_BLOCK, slot % SWA_BLOCK)] = (keys_t[t], t, t2r[t])
        out[f"swa{li}"] = sw
        if not spec.kv_source:
            return out
        kinds = ["comp"] + (["idxk"] if spec.index_source else [])
        for kind in kinds:
            d = {}
            for t, slot in enumerate(getattr(batch, f"{kind}_slots").tolist()):
                if slot >= 0:
                    j = pos[t] // self.ratio
                    d[(slot // self.comp_e, slot % self.comp_e)] = (comp_key(t2r[t], j), t, t2r[t])
            out[kind] = d
        return out

    def check_writes(self, name, expected, sched, pos, final, step_no, tr) -> None:
        """Provenance follows the writes; compressed and index-key rows, and
        every row of a recomputed request, equal the uninterrupted run's."""
        rec, kind = self.rec, self.kind_of(name)
        tensor = self.tensors[name]
        for (blk, off), (key, t, i) in expected.items():
            req = sched[i][0]
            self.prov[name][(blk, off)] = key
            self.wrote[name][(blk, off)] = self.tick
            if not final(t):
                continue
            idx = pos[t] if kind == "swa" else pos[t] // self.ratio
            if rec is None:
                self.oracle[(req.rid, self.name, name, idx)] = tensor[blk, off].clone()
                continue
            tr[f"{kind}_rows_written"] += 1
            if kind == "swa" and not req.preempted:
                continue
            ok = torch.equal(tensor[blk, off], self.oracle[(req.rid, self.name, name, idx)])
            detail = dict(step=step_no, request=req.rid, cache=name, index=idx, block=blk)
            if kind != "swa":
                rec.expect(ok, "compressed_rows_match_uninterrupted", **detail)
            if req.preempted:
                rec.coverage["recompute_rows_compared"] += 1
                rec.expect(ok, "recompute_equals_uninterrupted", **detail)

    def check_ring_carry(self, li, spec, sched, ckey, step_no) -> None:
        """A chunk starting on a group's second token pools it with the ring
        row of the first: that row must hold the request's own ``[kv | score]``
        input of that position, in a ring block the request holds."""
        rec, ring = self.rec, self.tensors["ring"]
        for i, (req, start, _, _) in enumerate(sched):
            if start % 2 == 0:
                continue
            rec.coverage["ring_carries"] += 1
            x = self.x_row(li, ckey(i, start - 1))[None]
            want = layer_inputs(self.args, spec, self.weights[li], x)[1][0]
            row = (start - 1) % self.cap
            rec.expect(
                req.ring != 0 and self.pools["ring"].ref.get(req.ring) == 1
                and torch.equal(ring[req.ring, row], want),
                "ring_carry_reads_own_row", step=step_no, request=req.rid, position=start - 1,
                ring_block=req.ring,
            )

    def check_read(self, name, blk, off, req, i, want_key, expect_blk, step_no, check, **detail):
        rec, kind = self.rec, self.kind_of(name)
        pool = self.pools[kind]
        held = pool.ref.get(blk, 0) > 0
        if not held:
            rec.fail("freed_blocks_never_read", step=step_no, request=req.rid, cache=name,
                     block=blk, **detail)
        got = self.prov[name].get((blk, off))
        ok = blk == expect_blk and held and got == want_key
        if not ok and self.wrote[name].get((blk, off), -1) < pool.alloc_tick.get(blk, 0):
            rec.fail("recycled_blocks_rewritten_before_read", step=step_no, request=req.rid,
                     cache=name, block=blk, **detail)
        if not ok:
            rec.fail(check, step=step_no, request=req.rid, cache=name, block=blk,
                     expected_block=expect_blk, holds=got, **detail)

    def check_window_reads(self, li, sched, swa_idx, pos, t2r, T, ckey, step_no, tr) -> None:
        name = f"swa{li}"
        rows = swa_idx.tolist()
        for t, row in enumerate(rows):
            if t >= T:
                self.rec.expect(all(s == -1 for s in row), "window_reads_owned", step=step_no,
                                row=t, reason="padding row has window slots")
                continue
            p, i = pos[t], t2r[t]
            req = sched[i][0]
            lo = max(0, p - WINDOW + 1)
            n = p - lo + 1
            if not self.rec.expect(all(s == -1 for s in row[n:]) and min(row[:n]) >= 0,
                                   "window_reads_owned", step=step_no, request=req.rid,
                                   position=p, reason="window slot count"):
                continue
            table = req.tables["swa"]
            for k in range(n):
                q = lo + k
                s = row[k]
                self.check_read(name, s // SWA_BLOCK, s % SWA_BLOCK, req, i, ckey(i, q),
                                table.get(q // SWA_BLOCK), step_no, "window_reads_owned",
                                position=p, reads=q)
            tr["window_reads"] += n
            self.rec.coverage["window_reads"] += n

    def check_compressed_reads(self, li, spec, sched, batch, pos, t2r, comp_key, step_no, tr):
        e = self.comp_e
        topk = self.topk[: len(pos)].tolist()
        per_req_vis = Counter()
        for t, p in enumerate(pos):
            i = t2r[t]
            req = sched[i][0]
            n_vis = (p + 1) // self.ratio
            per_req_vis[i] = max(per_req_vis[i], n_vis)
            for j in topk[t]:
                if j < 0:
                    continue
                if not self.rec.expect(j < n_vis, "compressed_reads_owned", step=step_no,
                                       request=req.rid, position=p, reads=j,
                                       reason="selected an invisible position"):
                    continue
                blk = req.tables["comp"].get(j // e)
                self.check_read("comp", blk, j % e, req, i, comp_key(i, j), blk, step_no,
                                "compressed_reads_owned", position=p, reads=j)
                tr["compressed_reads"] += 1
                self.rec.coverage["compressed_reads"] += 1
        if not spec.index_source:
            return
        # The indexer scores every index key below the request's widest
        # visible count in the step.
        for i, width in per_req_vis.items():
            req = sched[i][0]
            for j in range(width):
                blk = req.tables["idxk"].get(j // e)
                self.check_read("idxk", blk, j % e, req, i, comp_key(i, j), blk, step_no,
                                "compressed_reads_owned", reads=j)
            tr["index_key_reads"] += width
            self.rec.coverage["compressed_reads"] += width

    def record_outputs(self, li, sched, pos, t2r, out, final) -> None:
        for t, p in enumerate(pos):
            if final(t):
                self.oracle[(sched[t2r[t]][0].rid, self.name, "out", li, p)] = out[t].clone()

    def check_outputs(self, li, sched, pos, t2r, out, final, step_no) -> None:
        """A recomputed request's attention outputs equal the uninterrupted
        run's."""
        rec = self.rec
        for t, p in enumerate(pos):
            req = sched[t2r[t]][0]
            if not (req.preempted and final(t)):
                continue
            rec.coverage["recompute_rows_compared"] += 1
            rec.expect(
                torch.equal(out[t], self.oracle[(req.rid, self.name, "out", li, p)]),
                "recompute_equals_uninterrupted", step=step_no, request=req.rid, layer=li,
                position=p,
            )


# ---------------------------------------------------------------------------
# Histories
# ---------------------------------------------------------------------------


def make_requests(rng: random.Random, prof: HistoryProfile) -> list[Request]:
    reqs = [
        Request(
            rid=i, stream=i, prompt=(p := rng.randint(*prof.prompt)),
            total=p + rng.randint(*prof.decode), arrive=rng.randint(0, prof.arrive_span),
        )
        for i in range(prof.requests)
    ]
    for k in range(prof.prefix_requests):
        donor = rng.choice(reqs[: prof.requests])
        share = COMP_BLOCK_TOKENS * rng.randint(1, (donor.prompt - 1) // COMP_BLOCK_TOKENS)
        prompt = share + rng.randint(17, 60)
        reqs.append(
            Request(rid=len(reqs), stream=len(reqs), prompt=prompt,
                    total=prompt + rng.randint(*prof.decode), donor=donor, share=share)
        )
    return reqs


class History:
    """A scheduler over one stack: admission with prefix hits, chunked
    prefill, decode with drafts, preemption under block pressure and forced
    preemptions and aborts."""

    def __init__(self, stack: Stack, reqs, prof, rng, rec):
        self.stack, self.reqs, self.prof, self.rng, self.rec = stack, reqs, prof, rng, rec
        self.waiting: deque = deque()
        self.running: list = []
        self.attempt = 0
        victims = rng.sample(reqs[: prof.requests], len(prof.preempt) + len(prof.abort))
        self.triggers = [("preempt", v, ph) for v, ph in zip(victims, prof.preempt)]
        self.triggers += [
            ("abort", v, ph) for v, ph in zip(victims[len(prof.preempt):], prof.abort)
        ]
        for _, v, ph in self.triggers:
            if ph == "prefill":  # at least two chunks before the trigger
                v.prompt = max(v.prompt, 2 * prof.budget + 1)
                v.total = max(v.total, v.prompt + prof.decode[0])

    def event(self, op, req, **transition):
        self.rec.event({"op": op, "stack": self.stack.name, "request": req.rid,
                        "length": req.length}, {"status": req.status, **transition})

    def preempt(self, req) -> None:
        freed = self.stack.release(req)
        self.running.remove(req)
        req.length, req.status, req.preempted = 0, "waiting", True
        self.waiting.appendleft(req)
        self.rec.coverage["preemptions"] += 1
        self.event("preempt", req, blocks_freed=freed)

    def end(self, req, status) -> None:
        freed = self.stack.release(req)
        if req in self.running:
            self.running.remove(req)
        if req in self.waiting:
            self.waiting.remove(req)
        req.status = status
        if status == "aborted":
            self.rec.coverage["aborts"] += 1
        self.event("finish" if status == "finished" else "abort", req, blocks_freed=freed)

    def admit(self, req) -> dict | None:
        """Take the ring block and the prefix hit; None without a ring block."""
        st = self.stack
        info = {"ring_block": 0, "ring_reused": False}
        if st.ratio > 1:
            pool = st.pools["ring"]
            if not pool.free:
                return None
            info["ring_reused"] = pool.free[0] in pool.used
            info["ring_block"] = pool.take(st.tick)
        req.ring = info["ring_block"]
        h = st.hit_length(req)
        info["hit_blocks"] = st.take_hit(req, h)
        info["hit_tokens"] = req.length = h
        req.status = "running"
        return info

    def admitted(self, req, info) -> None:
        cov = self.rec.coverage
        cov["ring_block_reuses"] += int(info["ring_reused"])
        if info["hit_tokens"]:
            cov["prefix_hits"] += 1
            cov["prefix_hit_blocks"] += info["hit_blocks"]
        if req.preempted:
            cov["recomputes"] += 1
        self.event("admit", req, known=req.known, **info)

    def plan(self, req, budget):
        rng, prof = self.rng, self.prof
        start = req.length
        if req.length < req.known:
            n = min(req.known - start, budget, rng.choice(prof.chunks))
            return start, [req.key(p) for p in range(start, start + n)], n
        k = rng.choice((0, 1, 2, NUM_SPEC, NUM_SPEC))
        n = min(1 + k, budget)
        acc = min(rng.randint(1, n), req.total - start)
        keys = [req.key(p) for p in range(start, start + acc)]
        for p in range(start + acc, start + n):
            self.attempt += 1
            keys.append((req.stream, p, self.attempt))
        return start, keys, acc

    def fire_triggers(self) -> None:
        for trig in list(self.triggers):
            op, v, ph = trig
            if v.status != "running":
                continue
            if ph == "prefill" and not (0 < v.length < v.known and v.length >= v.prompt // 2):
                continue
            if ph == "decode" and v.length < v.prompt + 2:
                continue
            self.triggers.remove(trig)
            if op == "preempt":
                self.preempt(v)
            else:
                self.end(v, "aborted")

    def run(self) -> None:
        st, rec, prof = self.stack, self.rec, self.prof
        step = 0
        while any(r.status not in ("finished", "aborted") for r in self.reqs):
            if step > 3000:
                raise RuntimeError("history does not terminate")
            for r in self.reqs:
                if r.status != "future":
                    continue
                ready = (r.donor.length >= r.share or r.donor.status in ("finished", "aborted")
                         if r.donor is not None else step >= r.arrive)
                if ready:
                    r.status, r.known = "waiting", r.prompt
                    self.waiting.append(r)
                    rec.coverage["requests"] += 1
                    self.event("arrive", r, prompt=r.prompt, total=r.total,
                               prefix_of=None if r.donor is None else r.donor.rid,
                               shared_tokens=r.share)
            self.fire_triggers()
            sched, budget = [], prof.budget
            for req in list(self.running):
                if req not in self.running or budget <= 0:
                    continue
                start, keys, acc = self.plan(req, budget)
                while not st.can_allocate(st.needs(req, start, len(keys))):
                    victim = self.running[-1]
                    self.preempt(victim)
                    if victim is req:
                        break
                if req not in self.running:
                    continue
                st.allocate(req, start, len(keys))
                sched.append((req, start, keys, acc))
                budget -= len(keys)
            while self.waiting and len(self.running) < MAX_RUNNING and budget > 0:
                req = self.waiting[0]
                info = self.admit(req)
                if info is None:
                    break
                start, keys, acc = self.plan(req, budget)
                if not st.can_allocate(st.needs(req, start, len(keys))):
                    st.release(req)
                    req.length, req.status = 0, "waiting"
                    break
                self.waiting.popleft()
                self.running.append(req)
                self.admitted(req, info)
                st.allocate(req, start, len(keys))
                sched.append((req, start, keys, acc))
                budget -= len(keys)
            if not sched:
                if self.waiting and not self.running:
                    raise RuntimeError("history deadlocked")
                step += 1
                continue
            transition = st.step(sched, step)
            freed = 0
            for req, start, keys, acc in sched:
                if 0 < start < req.known:
                    rec.coverage["chunk_boundaries"] += 1
                rec.coverage["drafts_rejected"] += len(keys) - acc
                req.length = start + acc
                req.known = max(req.known, req.length)
                st.register_full(req, start)
                freed += st.free_behind_window(req)
            rec.coverage["window_blocks_freed"] += freed
            rec.event(
                {"op": "step", "stack": st.name, "step": step,
                 "batch": [[r.rid, s, len(k), acc] for r, s, k, acc in sched]},
                {**transition, "window_blocks_freed": freed},
            )
            for req, *_ in sched:
                if req.status == "running" and req.length >= req.total:
                    self.end(req, "finished")
            rec.coverage["steps"] += 1
            step += 1
        rec.coverage["recycled_blocks"] += sum(p.recycled for p in st.pools.values())


def run_uninterrupted(name, specs, seed, req: Request, oracle: dict) -> None:
    """The request alone in fresh caches: its prompt in one chunk, then one
    token per decode step."""
    alone = copy.copy(req)
    alone.tables, alone.shared, alone.ring = {}, set(), 0
    blocks = req.total // SWA_BLOCK + 2
    pools = {"swa": blocks, "comp": blocks, "idxk": blocks, "ring": 1}
    st = Stack(name, specs, seed, pools, False, random.Random(0), None, oracle)
    if st.ratio > 1:
        alone.ring = st.pools["ring"].take(0)
    start, step = 0, 0
    while start < req.total:
        n = req.prompt if start == 0 else 1
        keys = [req.key(p) for p in range(start, start + n)]
        st.allocate(alone, start, n)
        st.step([(alone, start, keys, n)], step)
        start += n
        step += 1


def run_attention(group: str, seed: int) -> Record:
    prof = PROFILES[group]
    rec = Record(ATTN_CHECKS, ATTN_COVERAGE)
    for name, specs in prof.stacks.items():
        rng = random.Random(_stable_seed(seed, group, name))
        reqs = make_requests(rng, prof)
        oracle: dict = {}
        stack = Stack(name, specs, seed, prof.pools, True, rng, rec, oracle)
        history = History(stack, reqs, prof, rng, rec)  # fixes the requests' lengths
        for req in reqs:
            run_uninterrupted(name, specs, seed, req, oracle)
        history.run()
    return rec


# ---------------------------------------------------------------------------
# Engram staging
# ---------------------------------------------------------------------------

ENGRAM_HEADS = (149,) * 8
ENGRAM_ROWS = sum(ENGRAM_HEADS) + 40
ENGRAM_LAYER = 3
ENGRAM_TP = 4
ENGRAM_MAX_TOKENS = 12
ENGRAM_CACHE_SETS = 2
ENGRAM_CACHE_WAYS = 8
# Token counts of captured lookups: capture sizes and spec-decode counts.
ENGRAM_GRAPH_SIZES = (3, 5, 6, 7, 12)
ENGRAM_LOOKUPS = 40


class DeviceQueue:
    """One in-order device stream on the host: deferred copies, dequant and
    consumers run when the stream reaches them -- when the host waits on an
    event, or when the device makes progress between lookups."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.ops: deque = deque()
        self.queued = 0
        self.done = 0
        self.running = False

    def enqueue(self, fn, kind: str, lookup: int) -> None:
        self.ops.append((fn, kind, lookup))
        self.queued += 1

    def run_until(self, mark: int) -> list:
        ran = []
        self.running = True
        try:
            while self.done < mark:
                fn, kind, lookup = self.ops.popleft()
                fn()
                self.done += 1
                ran.append((kind, lookup))
        finally:
            self.running = False
        return ran

    def progress(self) -> None:
        self.run_until(self.done + self.rng.randint(0, len(self.ops)))

    def drain(self) -> None:
        self.run_until(self.queued)


class QueueEvent:
    def __init__(self, queue: DeviceQueue, on_wait=None):
        self.queue, self.mark, self.on_wait = queue, 0, on_wait

    def record(self, stream=None) -> None:
        self.mark = self.queue.queued

    def synchronize(self) -> None:
        ran = self.queue.run_until(self.mark)
        if self.on_wait is not None:
            self.on_wait(ran)


class QueueStream:
    def wait_event(self, event) -> None:
        # One in-order stream: work queued after the event already waits.
        pass


class DeviceMode(TorchFunctionMode):
    """Defers the copies and fills the lookup issues on the device stream:
    those into device buffers and every non-blocking copy."""

    def __init__(self, queue: DeviceQueue, kinds: dict, state):
        super().__init__()
        self.queue, self.kinds, self.state = queue, kinds, state

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.queue.running and func in (torch.Tensor.copy_, torch.Tensor.fill_):
            dst = args[0]
            kind = self.kinds.get(dst.untyped_storage().data_ptr())
            non_blocking = kwargs.get("non_blocking") or (len(args) > 2 and args[2])
            if kind is not None and (kind.split(":")[-1].startswith("dev") or non_blocking):
                lookup = self.state.lookup
                self.queue.enqueue(
                    lambda: self.state.on_device_op(kind, lookup, func, args, kwargs),
                    kind, lookup,
                )
                return dst
        return func(*args, **kwargs)


class FetchProxy:
    """The native fetch with its queued work on the emulated device queue:
    ``enqueue`` (eager) queues the host function, ``enqueue_fixed``
    (captured) its id copy, host function and row copies, as the stream would
    run them. Everything else is the real ``EngramHostFetch``."""

    def __init__(self, rank: "EngramRank", fetcher):
        self.rank, self.fetcher = rank, fetcher

    def __getattr__(self, name):
        return getattr(self.fetcher, name)

    def enqueue(self, count: int) -> None:
        rank, lookup = self.rank, self.rank.lookup
        rank.queue.enqueue(
            lambda: rank.host_fetch("eager", lookup, count,
                                    lambda: self.fetcher.run(count)),
            "host_fetch", lookup,
        )

    def enqueue_fixed(self, count, dev_ids, dev_values, dev_scales) -> None:
        rank, lookup = self.rank, self.rank.lookup
        fixed = rank.emb.store.fixed
        staging = "fixed"

        def ids_to_host():
            rank.read(staging, "dev_ids", lookup)
            fixed.ids[:count].copy_(dev_ids[:count])
            rank.writer[(staging, "host_ids")] = lookup

        def rows_to_device():
            rank.read(staging, "host_rows", lookup)
            dev_values[:count].copy_(fixed.values[:count])
            dev_scales[:count].copy_(fixed.scales[:count])
            rank.writer[(staging, "dev_values")] = lookup

        rank.queue.enqueue(ids_to_host, "host_ids", lookup)
        rank.queue.enqueue(
            lambda: rank.host_fetch(staging, lookup, count,
                                    lambda: self.fetcher.run_fixed(count)),
            "host_fetch", lookup,
        )
        rank.queue.enqueue(rows_to_device, "dev_values", lookup)


class EngramRank:
    """One TP rank's ``Sm80EngramEmbedding`` driven through a history of
    eager and captured lookups with the device stream emulated."""

    def __init__(self, files, rank, table_values, table_scales, rng, rec: Record):
        self.rec, self.rng, self.rank = rec, rng, rank
        self.values, self.scales = table_values, table_scales
        cache_bytes = ENGRAM_CACHE_SETS * ENGRAM_CACHE_WAYS * (ENGRAM_DIM + 16)
        emb = sm80_engram.Sm80EngramEmbedding(
            ENGRAM_ROWS, ENGRAM_DIM, ENGRAM_HEADS, files, tp_rank=rank, tp_size=ENGRAM_TP,
            max_tokens=ENGRAM_MAX_TOKENS, placement="ssd", cache_bytes=cache_bytes,
            device="cpu", graph_capture=True,
        )
        assert emb.store.fetcher.cache_rows() == ENGRAM_CACHE_SETS * ENGRAM_CACHE_WAYS
        self.emb = emb
        self.queue = DeviceQueue(rng)
        cap = emb.capacity_rows
        emb._dev_ids = torch.empty(cap, dtype=torch.int64)
        emb._dev_values = torch.empty(cap, ENGRAM_DIM, dtype=torch.uint8)
        emb._dev_scales = torch.empty(cap, ENGRAM_DIM // 32, dtype=torch.uint8)
        emb._staged = QueueEvent(self.queue)
        emb._consumed = QueueEvent(self.queue)
        emb.device = torch.device("cuda")
        emb.store.fetcher = FetchProxy(self, emb.store.fetcher)
        self.out = torch.empty(ENGRAM_MAX_TOKENS, emb.part_n_hash_cols, ENGRAM_DIM, dtype=BF16)
        store = emb.store

        def ptr(t):
            return t.untyped_storage().data_ptr()

        self.kinds = {
            ptr(emb._dev_ids): "eager:dev_ids",
            ptr(emb._dev_values): "eager:dev_values",
            ptr(emb._dev_scales): "eager:dev_scales",
            ptr(store.ids): "eager:host_ids",
            ptr(self.out): "out:dev_out",
        }
        dev_ids, dev_values, dev_scales = emb._graph_staging[1]
        self.kinds[ptr(dev_ids)] = "fixed:dev_ids"
        self.kinds[ptr(dev_values)] = "fixed:dev_values"
        self.kinds[ptr(dev_scales)] = "fixed:dev_scales"
        self.lookup = -1
        # (staging, buffer) -> the lookup whose data the buffer holds.
        self.writer: dict = {}
        self.model = _LruModel(ENGRAM_CACHE_SETS, ENGRAM_CACHE_WAYS)
        self.results: dict = {}
        self.expected: dict = {}
        self.hit_rows: dict = {}
        self.refetched: dict = {}
        self.evicted_ids: set = set()
        self.transition: dict = {}
        # lookup -> the staging it used.
        self.staged_by: dict = {}
        self.counters = store.counters()
        lo = emb.vocab_start_idx
        hot = [rng.randrange(emb.part_num_embeddings) for _ in range(6)]
        self.hot = [lo + h for h in hot]

    # -- device side ------------------------------------------------------

    def read(self, staging: str, buffer: str, lookup: int) -> None:
        holds = self.writer.get((staging, buffer), -1)
        self.rec.expect(holds == lookup, "staging_reuse_ordered", rank=self.rank,
                        lookup=lookup, staging=staging, buffer=buffer, holds=holds)

    def on_device_op(self, kind, lookup, func, args, kwargs) -> None:
        staging, buffer = kind.split(":")
        if buffer == "host_ids":
            self.read(staging, "dev_ids", lookup)
        elif buffer in ("dev_values", "dev_scales"):
            self.read(staging, "host_rows", lookup)
        func(*args, **kwargs)
        self.writer[(staging, buffer)] = lookup

    def host_fetch(self, staging: str, lookup: int, count: int, run) -> None:
        """The host function: the native fetch of the staged ids, checked
        against the LRU model of the row cache."""
        rec, store = self.rec, self.emb.store
        self.read(staging, "host_ids", lookup)
        ids = store.ids if staging == "eager" else store.fixed.ids
        ids = ids.numpy()[:count].copy()
        lo, hi = store.owner_lo, store.owner_hi
        local = np.unique(ids[(ids >= lo) & (ids < hi)] - lo)
        tags_before = set(self.model.tags[self.model.tags >= 0].tolist())
        hits, misses, inserted, evicted, overflow = self.model.fetch(local)
        run()
        self.writer[(staging, "host_rows")] = lookup
        tags_after = set(self.model.tags[self.model.tags >= 0].tolist())
        counters = store.counters()
        delta = {k: counters[k] - self.counters[k] for k in counters}
        self.counters = counters
        want = dict(cache_hits=hits, cache_misses=misses, cache_inserts=inserted,
                    cache_evictions=evicted)
        got = {k: delta[k] for k in want}
        rec.expect(got == want, "cache_entries_match_table", rank=self.rank,
                   lookup=lookup, native=got, lru=want)
        gone = tags_before - tags_after
        unique = set(local.tolist())
        self.hit_rows[lookup] = {lo + r for r in unique & tags_before}
        refetch = len(unique & self.evicted_ids)
        self.evicted_ids = (self.evicted_ids | gone) - tags_after
        self.refetched[lookup] = refetch
        cov = rec.coverage
        cov["evictions"] += evicted
        cov["set_conflicts"] += overflow
        cov["refetch_after_eviction"] += refetch
        self.transition = dict(cache_hits=hits, cache_misses=misses,
                               evicted=evicted, refetched=refetch)

    def deferred_dequant(self, values, scales, out) -> None:
        real, lookup = DEQUANT, self.lookup
        staging = self.kinds[values.untyped_storage().data_ptr()].split(":")[0]

        def dequant():
            self.read(staging, "dev_values", lookup)
            real(values, scales, out)

        self.queue.enqueue(dequant, "dequant", lookup)

    # -- host side --------------------------------------------------------

    def expected_rows(self, cols: np.ndarray) -> torch.Tensor:
        lo, hi = self.emb.vocab_start_idx, self.emb.vocab_end_idx
        flat = cols.reshape(-1)
        v = np.zeros((len(flat), ENGRAM_DIM), dtype=np.uint8)
        s = np.full((len(flat), ENGRAM_DIM // 32), 127, dtype=np.uint8)
        own = (flat >= lo) & (flat < hi)
        v[own], s[own] = self.values[flat[own]], self.scales[flat[own]]
        rows = sm80_engram_dequant.dequant_engram_rows_reference(
            torch.from_numpy(v), torch.from_numpy(s)
        )
        return rows.view(cols.shape[0], cols.shape[1], ENGRAM_DIM)

    def ids(self, n: int) -> torch.Tensor:
        """[n, hash heads] global row ids: this rank's columns draw hot rows,
        repeats and fresh rows; other columns are other ranks' rows."""
        rng = self.rng
        out = np.empty((n, len(ENGRAM_HEADS)), dtype=np.int64)
        offsets = np.cumsum((0,) + ENGRAM_HEADS[:-1])
        for c, (off, size) in enumerate(zip(offsets, ENGRAM_HEADS)):
            for t in range(n):
                out[t, c] = off + rng.randrange(size)
        start, part = self.emb.head_start, self.emb.part_n_hash_cols
        for t in range(n):
            for c in range(start, start + part):
                r = rng.random()
                if r < 0.4:
                    out[t, c] = rng.choice(self.hot)
                elif r < 0.5 and t:
                    out[t, c] = out[t - 1, c]
        return torch.from_numpy(out)

    def run(self) -> None:
        rec, emb, q = self.rec, self.emb, self.queue
        mode = DeviceMode(q, self.kinds, self)
        cols_of: dict = {}
        for lookup in range(ENGRAM_LOOKUPS):
            q.progress()
            self.lookup = lookup
            captured = self.rng.random() < 0.4
            if captured:
                n = self.rng.choice(ENGRAM_GRAPH_SIZES)
            else:
                n = self.rng.randint(1, ENGRAM_MAX_TOKENS)
            staging = "fixed" if captured else "eager"
            # The device ordering, not a host wait, protects a staging that
            # an earlier lookup's queued work still uses.
            rec.coverage["staging_waits"] += any(
                kind != "consume" and lk < lookup and self.staged_by.get(lk) == staging
                for _, kind, lk in q.ops
            )
            idx = self.ids(n)
            cols = idx[:, emb.head_start : emb.head_start + emb.part_n_hash_cols].numpy()
            cols_of[lookup] = cols
            self.expected[lookup] = self.expected_rows(cols)
            pending = len(q.ops)
            out = self.out[:n]
            with mock.patch.object(torch.cuda, "current_stream", lambda: QueueStream()), \
                    mock.patch.object(torch.cuda, "is_current_stream_capturing",
                                      lambda c=captured: c), \
                    mock.patch.object(sm80_engram_dequant, "dequant_engram_rows",
                                      self.deferred_dequant), mode:
                emb.lookup(idx, out)
            self.staged_by[lookup] = staging
            q.enqueue(lambda lk=lookup, o=out: self.results.__setitem__(lk, o.clone()),
                      "consume", lookup)
            rec.coverage["lookups"] += 1
            rec.coverage["captured_lookups"] += captured
            rec.event(
                {"op": "lookup", "rank": self.rank, "lookup": lookup, "tokens": n,
                 "staging": staging, "unique_rows": len(np.unique(cols))},
                {"device_ops_pending_before": pending, "device_ops_pending_after": len(q.ops)},
            )
        q.drain()
        emb.close()
        for lookup, want in self.expected.items():
            got = self.results.get(lookup)
            if got is None:
                rec.fail("no_stale_row_after_eviction", rank=self.rank, lookup=lookup,
                         consumed=False)
                continue
            same = (got.view(torch.int16) == want.view(torch.int16)).all(-1).numpy()
            # Rows served from the row cache are the table's rows.
            hit = np.isin(cols_of[lookup], list(self.hit_rows.get(lookup, ())))
            rec.expect(bool(same[hit].all()), "cache_entries_match_table",
                       rank=self.rank, lookup=lookup, rows=int(hit.sum()))
            # A lookup of rows the cache had evicted returns the table's rows.
            if self.refetched.get(lookup):
                rec.expect(bool(same.all()), "no_stale_row_after_eviction",
                           rank=self.rank, lookup=lookup)
        rec.event({"op": "drain", "rank": self.rank},
                  {"lookups_consumed": len(self.results), "device_ops_left": len(q.ops),
                   **self.transition})

DEQUANT = sm80_engram_dequant.dequant_engram_rows


def run_engram(seed: int, work: Path) -> Record:
    rec = Record(ENGRAM_CHECKS, ENGRAM_COVERAGE)
    with tempfile.TemporaryDirectory(dir=work) as d:
        d = Path(d)
        _write_table(d, ENGRAM_LAYER, ENGRAM_ROWS, np.arange(ENGRAM_ROWS),
                     "model-00001-of-00001.safetensors")
        files = sm80_engram.locate_engram_table(d, ENGRAM_LAYER, ENGRAM_ROWS, ENGRAM_DIM)
        raw = np.fromfile(files.values_path, dtype=np.uint8)
        values = raw[files.values_offset : files.values_offset + ENGRAM_ROWS * ENGRAM_DIM]
        scale_bytes = ENGRAM_ROWS * ENGRAM_DIM // 32
        scales = raw[files.scales_offset : files.scales_offset + scale_bytes]
        values = values.reshape(ENGRAM_ROWS, ENGRAM_DIM)
        scales = scales.reshape(ENGRAM_ROWS, ENGRAM_DIM // 32)
        for rank in range(ENGRAM_TP):
            rng = random.Random(_stable_seed(seed, "engram", rank))
            EngramRank(files, rank, values, scales, rng, rec).run()
    return rec


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

GROUPS = ("compressor_ring", "paged_kv", "engram_staging")


def produce(group: str, seed: int, out: Path) -> dict:
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rec = run_engram(seed, out.parent) if group == "engram_staging" else run_attention(group, seed)
    report = rec.report(group, seed)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", required=True, choices=GROUPS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True, help="report path (.json)")
    args = ap.parse_args(argv)
    report = produce(args.group, args.seed, Path(args.out))
    failed = sorted(k for k, v in report["checks"].items() if not v)
    print(f"{args.group} seed {args.seed}: "
          + ("all checks hold" if not failed else "failed " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
