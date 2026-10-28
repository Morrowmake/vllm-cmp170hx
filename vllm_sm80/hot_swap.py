# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine switches swapped between steps inside one server (development).

``VLLM_HOT_SWAP=<pair>[,<pair>...]`` names pairs of :data:`HOT_PAIRS`. A
pair is one engine switch: its incumbent is the switch off, as the server
boots, its candidate the switch on, both in this tree. Then:

- the platform sets ``parallel_config.worker_extension_cls`` to the dev A/B
  worker extension (:func:`vllm_sm80.dev_ab.configure`);
- a pair reaches the engine at graph sites, eagerly, or both:

  - graph sites (:func:`pair`): every CUDA graph is captured once with both
    variants of every pair site side by side (:func:`capture`, called by
    ``with_ab_capture``). A site runs the incumbent, then the candidate
    writing into the incumbent's output tensor, and records the graph nodes
    each variant added (the set difference of the capturing graph's nodes
    around it). The graphs keep their ``cudaGraph_t`` (``keep_graph=True``)
    so node handles stay valid. The candidate's nodes start disabled
    (``cuGraphNodeSetEnabled``): a disabled kernel, memset or memcpy node
    runs as an empty node. Eager calls of a site run the arm's variant;
  - eager state (``HotPair.eager``): code that runs only outside CUDA graphs
    (prefill steps, sampling) reads the arm at each call (:func:`eager_on`,
    which refuses a read while a graph is being captured) or from the
    attributes the pair's ``apply`` sets on the loaded model;

- ``POST /collective_rpc`` (``VLLM_SERVER_DEV_MODE=1``) with method
  ``set_hot_arms`` and args ``[[<pairs running their candidate>],
  generation]`` flips the node sets of every executable graph and the eager
  state on every rank between engine steps, then re-uploads each executable
  graph. The ranks agree first (no request resident, a newer generation,
  every pair carried on some rank), then acknowledge a commit or a rollback
  on every rank, as ``set_ab_arm`` does.

The pairs are independent: any subset runs its candidates, so one server
carries every listed candidate and each A/B costs only its measurement
blocks. Both variants of a graph site are captured, so the memory profile
and the KV cache size see both variants' scratch. Each arm replays the same
topology: the other variant's nodes are present and disabled in both arms.

A pair site never nests inside another. A variant whose capture adds a node
of another type (an event, a host node, a child graph) is refused at capture.

A switch is a pair only when both of its arms run on what the boot built:
the same allocations, process groups and captured graph shapes. Not pairs:

- ``VLLM_SM80_ALLREDUCE``: the communicator builds its transports when it
  is constructed (the host-staged buffers, the flags-in-data buffers sized by
  the allreduce/flags_max_bytes row, the copy-engine ring sized by
  ``max_num_batched_tokens``), and the layers take over their sums at load;
- ``VLLM_SM80_ALLREDUCE_ROUND_ONCE``: decode graphs capture the BF16 or the
  FP32 all-reduce of the ``wo_b`` partial, each into its own output, so the
  candidate cannot write the incumbent's output without a copy the served
  path does not run;
- ``VLLM_SM80_MARLIN_PREFILL``: the fused MoE layer calls the split-block
  prefill only when the switch is set at boot (its warmup sizes the block
  lists then), and that check sits in the upstream layer;
- the sm80 tuning table (``VLLM_SM80_TUNING_FILE``): lookups are cached per
  process; rows read while capturing pick each graph's kernels, rows read at
  load size workspaces and communicator buffers, and the arms carry no table.

An eager candidate allocates its own scratch at its first call, after the KV
cache is sized: the boot's memory profile sees the incumbents' eager steps.

Unset (the default), :func:`listed` is False for every pair,
:func:`capture` is never entered and nothing here runs.
"""

import sys
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class HotPair:
    """One allowlisted pair.

    ``env`` is the switch that serves the candidate alone (it must be off in
    a hot server: the server boots on every incumbent).
    ``unavailable(runner)`` returns why this rank cannot carry the pair, or
    None; it may record what ``apply`` sets (:func:`_targets`).
    ``apply(runner, on)``, when given, sets the pair's eager state on this
    rank, outside CUDA capture. ``eager``: the pair changes steps that run
    outside CUDA graphs, so it is carried without a graph site.
    """

    name: str
    env: str
    unavailable: Callable[[Any], str | None]
    apply: Callable[[Any, bool], None] | None = None
    eager: bool = False


# ---------------------------------------------------------------------------
# Pair adapters
# ---------------------------------------------------------------------------


def _targets(name: str) -> list:
    """What pair ``name`` sets on this rank, recorded when it was resolved."""
    return _STATE.targets.setdefault(name, [])


def _argmax_unavailable(runner) -> str | None:
    """The target half samples eagerly (``target_argmax.step``); the DSpark
    Markov sampler's drafts are a graph site (``shard_argmax``)."""
    from vllm_sm80.spec_decode import shard_argmax
    from vllm_sm80.worker import target_argmax

    if shard_argmax._kind(getattr(runner, "speculator", None)) == "sample_draft":
        return (
            "the drafter takes the per-rank argmax by replacing its "
            "sample_draft at load, with no incumbent beside it"
        )
    reason = target_argmax.refusal(runner)
    if reason is not None:
        return reason
    if getattr(runner, "is_last_pp_rank", True):
        _targets("shard_argmax").append(runner)
    return None


def _argmax_apply(runner, on: bool) -> None:
    from vllm_sm80.worker import target_argmax

    for target in _targets("shard_argmax"):
        target_argmax.set_enabled(target, on)
        target_argmax.warm(target)


_DSV41_MODEL = "vllm_sm80.models.deepseek_v41.sm80_model"
_DSV41_SPLIT = "vllm_sm80.models.deepseek_v41.sm80_split_reduce"


def _split_unavailable(runner) -> str | None:
    model = sys.modules.get(_DSV41_MODEL)
    if model is None:
        return None
    layers = [
        m for m in runner.model.modules()
        if isinstance(m, model.DeepseekV4DecoderLayer)
    ]
    split = sys.modules[_DSV41_SPLIT]
    takers = [layer for layer in layers if split.splits(layer)]
    if layers and not takers:
        return (
            "no decoder layer issues its own all-reduces; boot with "
            "VLLM_SM80_ALLREDUCE=1, which moves the MoE output's all-reduce "
            "into the layer"
        )
    _targets("prefill_reduce_overlap").extend(takers)
    return None


def _split_apply(runner, on: bool) -> None:
    for layer in _targets("prefill_reduce_overlap"):
        layer._split_reduce = on


_KDA_PREFILL = "vllm_sm80.models.glm5next.ops.kda_prefill"


def _kda_unavailable(runner) -> str | None:
    """The layer part of the gate, as the layer resolves it with the switch
    set at construction; every KDA layer boots with its gate closed."""
    layers = [m for m in runner.model.modules() if hasattr(m, "_pp_kda_prefill")]
    if not layers:
        return None
    import importlib

    kda = importlib.import_module(_KDA_PREFILL)
    dtype = runner.vllm_config.model_config.dtype
    _targets("kda_prefill").extend(
        m for m in layers
        if kda.use_for_layer(
            m.kda_prefill_backend, m.local_num_heads, m.head_dim, dtype,
            m.kda_safe_gate, m.kda_lower_bound,
            allowed_heads=kda.supported_heads(),
        )
    )
    return None


def _kda_apply(runner, on: bool) -> None:
    layers = _targets("kda_prefill")
    for layer in layers:
        layer._pp_kda_prefill = on
    if on and layers and "kda_prefill" not in _STATE.warmed:
        import importlib

        import torch

        kda = importlib.import_module(_KDA_PREFILL)
        with torch.inference_mode():
            kda.warmup(
                kda.WARMUP_PLANS, device=runner.device,
                heads=int(layers[0].local_num_heads),
            )
        _STATE.warmed.add("kda_prefill")


_SPARSE_MLA_BACKEND = "vllm_sm80.attention.triton_mla_sparse"


def _sparse_mla_unavailable(runner) -> str | None:
    """The attention layers of this rank on the sm_80 sparse-MLA backend,
    whose prefill calls read the gate (the memory profile skips attention,
    so no call has read it at boot)."""
    _targets("sparse_mla_prefill").extend(
        m for m in runner.model.modules()
        if type(getattr(m, "impl", None)).__module__ == _SPARSE_MLA_BACKEND
    )
    return None


def _topk_unavailable(runner) -> str | None:
    from vllm_sm80.dev_ab import _topk_gate

    gate = _topk_gate()
    if gate is None:
        return "the GLM or Qwen sparse indexer (shared top-k gate) is not loaded"
    gate._warm()
    return None


HOT_PAIRS: dict[str, HotPair] = {
    p.name: p
    for p in (
        HotPair(
            "shard_argmax", "VLLM_SM80_SHARD_ARGMAX",
            _argmax_unavailable, _argmax_apply, eager=True,
        ),
        HotPair(
            "prefill_reduce_overlap", "VLLM_SM80_PREFILL_REDUCE_OVERLAP",
            _split_unavailable, _split_apply, eager=True,
        ),
        HotPair(
            "kda_prefill", "VLLM_SM80_KDA_PREFILL",
            _kda_unavailable, _kda_apply, eager=True,
        ),
        HotPair(
            "mhc_prefill", "VLLM_SM80_MHC_PREFILL",
            lambda runner: None, eager=True,
        ),
        HotPair(
            "sparse_mla_prefill", "VLLM_SM80_SPARSE_MLA_PREFILL",
            _sparse_mla_unavailable, eager=True,
        ),
        HotPair(
            "topk_canonical_ties", "VLLM_SM80_TOPK_CANONICAL_TIES",
            _topk_unavailable,
        ),
        HotPair("topk_sorted", "VLLM_SM80_TOPK_SORTED", _topk_unavailable),
    )
}

# Pairs that share one site: a server carries at most one of each group.
_ONE_SITE = (("topk_canonical_ties", "topk_sorted"),)


def _parse(value: str) -> tuple[str, ...]:
    names = tuple(n.strip() for n in value.split(",") if n.strip())
    unknown = [n for n in names if n not in HOT_PAIRS]
    if unknown:
        raise ValueError(
            f"VLLM_HOT_SWAP: {', '.join(unknown)} not in: " + ", ".join(sorted(HOT_PAIRS))
        )
    if len(set(names)) != len(names):
        raise ValueError(f"VLLM_HOT_SWAP={value!r} names a pair twice")
    for group in _ONE_SITE:
        if len(set(group) & set(names)) > 1:
            raise ValueError(
                f"VLLM_HOT_SWAP: {' and '.join(group)} share one site; "
                "list one of them per server"
            )
    return names


@dataclass
class _Site:
    name: str
    base: tuple[int, ...]
    cand: tuple[int, ...]


@dataclass
class _Exec:
    graph: weakref.ref
    exec_handle: Any
    sites: list[_Site]
    on: frozenset = frozenset()


@dataclass
class _HotState:
    listed: tuple[str, ...] = ()
    on: frozenset = frozenset()
    generation: int = 0
    window: bool = False
    in_pair: bool = False
    resolved: bool = False
    pending: dict[int, list[_Site]] = field(default_factory=dict)
    execs: list[_Exec] = field(default_factory=list)
    kernels: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    toggle_ms: float = 0.0
    # Graphs captured in the window, with or without pair sites.
    graphs_captured: int = 0
    # Pairs whose eager site or gate ran on this rank.
    reached: set[str] = field(default_factory=set)
    # What each pair's apply sets on this rank (_targets).
    targets: dict[str, list] = field(default_factory=dict)
    warmed: set[str] = field(default_factory=set)


_STATE = _HotState(listed=_parse(envs.VLLM_HOT_SWAP) if envs.VLLM_HOT_SWAP else ())


def active() -> bool:
    return bool(_STATE.listed)


def listed(name: str) -> bool:
    """True when the server carries pair ``name`` (its sites must call
    :func:`pair`, its eager gates :func:`eager_on`)."""
    return name in _STATE.listed


def eager_on(name: str) -> bool:
    """Whether the listed pair ``name`` runs its candidate at this call of a
    gate that only eager steps read. A read while a CUDA graph is captured is
    refused: the graph would keep this arm's choice."""
    st = _STATE
    if name not in st.listed:
        return False
    import torch

    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"hot pair {name}: its eager gate was read during CUDA graph capture")
    st.reached.add(name)
    return name in st.on


def configure(vllm_config) -> bool:
    """Validate VLLM_HOT_SWAP for this config; True when it is set."""
    import os

    if not _STATE.listed:
        return False
    if envs.VLLM_DEV_AB_SWITCH:
        raise ValueError("VLLM_HOT_SWAP cannot be combined with VLLM_DEV_AB_SWITCH")
    if envs.VLLM_SM80_CAPTURE:
        raise ValueError("VLLM_HOT_SWAP cannot be combined with VLLM_SM80_CAPTURE")
    for name in _STATE.listed:
        env = HOT_PAIRS[name].env
        if os.environ.get(env, "0") not in ("", "0"):
            raise ValueError(
                f"VLLM_HOT_SWAP={name}: {env} must be off (the server boots on "
                "every incumbent; set_hot_arms selects candidates)"
            )
    return True


# ---------------------------------------------------------------------------
# CUDA driver calls
# ---------------------------------------------------------------------------


def _drv():
    from cuda.bindings import driver

    return driver


def _check(result, what: str):
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"hot swap: {what} failed: {err}")
    return rest


def _capturing_graph() -> int | None:
    """The graph the current stream is capturing into, or None."""
    import torch

    if not torch.cuda.is_current_stream_capturing():
        return None
    drv = _drv()
    stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
    status, _id, graph, *_ = _check(drv.cuStreamGetCaptureInfo(stream), "cuStreamGetCaptureInfo")
    if status != drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        return None
    return int(graph)


def _nodes(graph: int) -> set[int]:
    drv = _drv()
    g = drv.CUgraph(graph)
    _, count = _check(drv.cuGraphGetNodes(g, 0), "cuGraphGetNodes")
    nodes, _ = _check(drv.cuGraphGetNodes(g, count), "cuGraphGetNodes")
    return {int(n) for n in nodes}


_TOGGLEABLE = None


def _node_kinds(nodes) -> list[str]:
    """The kernel name of each kernel node, 'memset'/'memcpy' for those;
    raises on any other node type."""
    global _TOGGLEABLE
    drv = _drv()
    T = drv.CUgraphNodeType
    if _TOGGLEABLE is None:
        _TOGGLEABLE = {T.CU_GRAPH_NODE_TYPE_KERNEL, T.CU_GRAPH_NODE_TYPE_MEMSET,
                       T.CU_GRAPH_NODE_TYPE_MEMCPY}
    kinds = []
    for n in nodes:
        (kind,) = _check(drv.cuGraphNodeGetType(drv.CUgraphNode(n)), "cuGraphNodeGetType")
        if kind not in _TOGGLEABLE:
            raise RuntimeError(f"node type {kind.name} cannot be disabled in a graph")
        if kind == T.CU_GRAPH_NODE_TYPE_KERNEL:
            kinds.append(_kernel_name(n))
        else:
            kinds.append("memset" if kind == T.CU_GRAPH_NODE_TYPE_MEMSET else "memcpy")
    return kinds


def _kernel_name(node: int) -> str:
    drv = _drv()
    try:
        (params,) = _check(drv.cuGraphKernelNodeGetParams(drv.CUgraphNode(node)),
                           "cuGraphKernelNodeGetParams")
        if int(params.func):
            (name,) = _check(drv.cuFuncGetName(params.func), "cuFuncGetName")
        else:
            (name,) = _check(drv.cuKernelGetName(params.kern), "cuKernelGetName")
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # names are a report, never a refusal
        return "?"


def _set_enabled(exec_handle, nodes: tuple[int, ...], on: bool) -> None:
    drv = _drv()
    flag = 1 if on else 0
    for n in nodes:
        _check(drv.cuGraphNodeSetEnabled(exec_handle, drv.CUgraphNode(n), flag),
               "cuGraphNodeSetEnabled")


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def pair(name: str, base: Callable[[], Any], cand: Callable[[Any], Any]):
    """One pair site. ``base()`` returns the incumbent's output tensor;
    ``cand(out)`` computes the candidate into ``out`` (``cand(None)``: into a
    new tensor) and returns it, or returns None when it does not serve this
    call (it then launched nothing). Returns the site's output."""
    st = _STATE
    if not st.window:
        import torch

        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            # The graph would keep only this arm's variant: every graph with a
            # pair site is captured inside the window (capture).
            raise RuntimeError(f"hot pair {name}: captured outside the hot capture window")
        st.reached.add(name)
        if name in st.on:
            out = cand(None)
            if out is not None:
                return out
        return base()
    if st.in_pair:
        raise RuntimeError(f"hot pair {name}: pair sites cannot nest")
    st.in_pair = True
    try:
        graph = _capturing_graph()
        before = _nodes(graph) if graph is not None else None
        out = base()
        middle = _nodes(graph) if graph is not None else None
        got = cand(out)
        after = _nodes(graph) if graph is not None else None
    finally:
        st.in_pair = False
    if got is not None and (got.data_ptr() != out.data_ptr() or got.shape != out.shape):
        raise RuntimeError(f"hot pair {name}: the candidate must write the incumbent's output")
    if graph is not None:
        base_nodes, cand_nodes = middle - before, after - middle
        if got is None:
            if cand_nodes:
                raise RuntimeError(f"hot pair {name}: a declining candidate launched work")
        else:
            st.pending.setdefault(graph, []).append(
                _Site(name, tuple(sorted(base_nodes)), tuple(sorted(cand_nodes))))
    return out


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _hot_graph_class():
    import torch

    original = torch.cuda.CUDAGraph

    class HotCUDAGraph(original):
        """A CUDA graph that keeps its cudaGraph_t and binds the pair sites
        captured into it to its executable graph."""

        def __new__(cls, keep_graph: bool = False):
            return super().__new__(cls, True)

        def __init__(self, keep_graph: bool = False):
            super().__init__(True)

        def capture_end(self) -> None:
            super().capture_end()
            self.instantiate()
            _bind(self)

    return original, HotCUDAGraph


def _bind(graph) -> None:
    st = _STATE
    st.graphs_captured += 1
    sites = st.pending.pop(int(graph.raw_cuda_graph()), [])
    if not sites:
        return
    for site in sites:
        names = st.kernels.setdefault(site.name, {"base": set(), "cand": set()})
        try:
            base_kinds, cand_kinds = _node_kinds(site.base), _node_kinds(site.cand)
        except RuntimeError as error:
            raise RuntimeError(f"hot pair {site.name}: {error}") from None
        if len(names["base"]) < 16:
            names["base"].update(base_kinds)
            names["cand"].update(cand_kinds)
    drv = _drv()
    ex = _Exec(weakref.ref(graph), drv.CUgraphExec(int(graph.raw_cuda_graph_exec())), sites)
    _toggle(ex, st.on, force=True)
    st.execs.append(ex)


def _toggle(ex: _Exec, on: frozenset, force: bool = False) -> None:
    for site in ex.sites:
        cand = site.name in on
        if force or cand != (site.name in ex.on):
            _set_enabled(ex.exec_handle, site.base, not cand)
            _set_enabled(ex.exec_handle, site.cand, cand)
    ex.on = on


def capture(runner, capture_fn, profile_only: bool):
    """Run ``capture_fn`` (the ordinary capture body) with both variants of
    every pair captured side by side."""
    import torch

    st = _STATE
    why = _capture_refusal(runner)
    if why is not None:
        raise RuntimeError(f"VLLM_HOT_SWAP: {why}")
    _resolve(runner)
    original, hot = _hot_graph_class()
    st.window = True
    torch.cuda.CUDAGraph = hot
    try:
        result = capture_fn(runner, profile_only=profile_only)
    finally:
        torch.cuda.CUDAGraph = original
        st.window = False
        stray = sum(len(s) for s in st.pending.values())
        st.pending.clear()
    if stray:
        raise RuntimeError(f"hot swap: {stray} pair sites captured outside a bound graph")
    _prune()
    logger.info(
        "Hot swap: %s capture of %s: %d graphs carry pair sites (%s)",
        "profiling" if profile_only else "serving", ",".join(st.listed),
        len(st.execs), _site_summary(),
    )
    return result


def _capture_refusal(runner) -> str | None:
    """Why the capture cannot carry pairs: policies that profile step costs
    while capturing would tune themselves to the incumbents. (The adaptive K
    table is configuration, not a capture-time profile: it runs as served in
    every arm.)"""
    if getattr(runner, "adaptive_verification", None) is not None:
        return "adaptive verification profiles step costs during capture"
    if getattr(getattr(runner, "speculator", None), "enable_adaptive_verification", False):
        return "adaptive verification changes drafter state"
    return None


def _resolve(runner) -> None:
    st = _STATE
    if st.resolved:
        return
    for name in st.listed:
        why = HOT_PAIRS[name].unavailable(runner)
        if why is not None:
            raise RuntimeError(f"VLLM_HOT_SWAP={name}: {why}")
    st.resolved = True


def _prune() -> None:
    _STATE.execs = [ex for ex in _STATE.execs if ex.graph() is not None]


def _sites() -> dict[str, dict[str, Any]]:
    out = {
        name: {"sites": 0, "base_nodes": 0, "cand_nodes": 0, "base_kernels": [],
               "cand_kernels": [], "eager": _carried_eagerly(name)}
        for name in _STATE.listed
    }
    for ex in _STATE.execs:
        if ex.graph() is None:
            continue
        for site in ex.sites:
            row = out[site.name]
            row["sites"] += 1
            row["base_nodes"] += len(site.base)
            row["cand_nodes"] += len(site.cand)
    for name, kinds in _STATE.kernels.items():
        out[name]["base_kernels"] = sorted(kinds["base"])
        out[name]["cand_kernels"] = sorted(kinds["cand"])
    return out


def _carried_eagerly(name: str) -> bool:
    """Whether pair ``name`` changes an eager step on this rank: its apply
    has something to set, or one of its eager sites or gates has run."""
    return bool(_STATE.targets.get(name)) or name in _STATE.reached


def _site_summary() -> str:
    return "; ".join(
        f"{n}: {r['sites']} sites, {r['base_nodes']}/{r['cand_nodes']} base/cand nodes"
        for n, r in _sites().items()
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _apply(runner, on: frozenset, force: bool = False) -> None:
    """Every executable graph, every eager state and every eager site to
    ``on`` (``force``: rewrite every site's flags and every pair's eager
    state, not only those whose arm changes)."""
    import torch

    drv = _drv()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    _prune()
    stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
    for ex in _STATE.execs:
        _toggle(ex, on, force)
        _check(drv.cuGraphUpload(ex.exec_handle, stream), "cuGraphUpload")
    for name in _STATE.listed:
        apply = HOT_PAIRS[name].apply
        if apply is not None and (force or (name in on) != (name in _STATE.on)):
            apply(runner, name in on)
    torch.accelerator.synchronize()
    _STATE.on = on
    _STATE.toggle_ms = (time.perf_counter() - start) * 1e3


def _report(worker, ok: bool, reason: str) -> dict[str, Any]:
    st = _STATE
    return {
        "ok": ok,
        "reason": reason,
        "rank": worker.rank,
        "pairs": list(st.listed),
        "on": sorted(st.on),
        "generation": st.generation,
        "graphs": sum(1 for ex in st.execs if ex.graph() is not None),
        "sites": _sites(),
        "toggle_ms": round(st.toggle_ms, 3),
    }


def _gather(value) -> list:
    """``value`` of every rank of the world group, in rank order."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_world_group

    group = get_world_group().cpu_group
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _all_ranks_same_state() -> bool:
    from vllm_sm80.dev_ab import _fail_stop

    try:
        state = (sorted(_STATE.on), _STATE.generation)
        return all(other == state for other in _gather(state))
    except Exception as error:
        _fail_stop(f"hot swap state acknowledgement failed: {error}")


def _carried_anywhere() -> set[str]:
    """The listed pairs some rank carries: a captured graph site, or an
    eager change (:func:`_carried_eagerly`). With pipeline parallelism a
    stage may hold none of a pair's layers."""
    from vllm_sm80.dev_ab import _fail_stop

    sites = _sites()
    mine = sorted(
        n for n in _STATE.listed
        if sites[n]["sites"] or (HOT_PAIRS[n].eager and sites[n]["eager"])
    )
    try:
        return {n for names in _gather(mine) for n in names}
    except Exception as error:
        _fail_stop(f"hot swap carried-pair exchange failed: {error}")


def set_arms(worker, on, generation) -> dict[str, Any]:
    """Run the candidates of the pairs in ``on`` and the incumbents of the
    rest, on every rank, or on none."""
    from vllm_sm80.dev_ab import _fail_stop, _resident_requests, _transaction_vote

    st = _STATE
    runner = worker.model_runner
    if isinstance(generation, str) and generation.isdecimal():
        generation = int(generation)
    reason = None
    if not st.listed:
        reason = "VLLM_HOT_SWAP is not set"
    elif not isinstance(on, (list, tuple)) or not all(isinstance(n, str) for n in on):
        reason = "on must be a list of pair names"
    elif unknown := sorted(set(on) - set(st.listed)):
        reason = f"{', '.join(unknown)} not carried (carried: {', '.join(st.listed)})"
    elif type(generation) is not int or generation <= st.generation:
        reason = f"generation {generation!r} is not newer than {st.generation}"
    elif resident := _resident_requests(runner):
        reason = f"{resident} requests resident; drain first"
    else:
        try:
            _resolve(runner)
        except RuntimeError as error:
            reason = str(error)
    agreed = _transaction_vote(reason is None)
    if reason is None and not agreed:
        reason = "another rank refused"
    if reason is not None:
        return _report(worker, False, reason)
    # Both arms would run the same kernels.
    if empty := sorted(set(on) - _carried_anywhere()):
        return _report(
            worker, False,
            f"{', '.join(empty)}: no rank has a captured pair site or an eager change of it",
        )
    if not _all_ranks_same_state():
        _fail_stop("hot swap rank states disagree before switching")
    previous = (st.on, st.generation)
    failure = None
    try:
        _apply(runner, frozenset(on))
        st.generation = generation
    except Exception as error:
        failure = str(error)
    committed = _transaction_vote(failure is None)
    if committed:
        committed = _all_ranks_same_state()
    if not committed:
        restored = False
        try:
            _apply(runner, previous[0], force=True)
            st.on, st.generation = previous
            restored = True
        except Exception as error:
            logger.error("Hot swap rollback failed: %s", error)
        if not _transaction_vote(restored) or not _all_ranks_same_state():
            _fail_stop("hot swap cannot acknowledge a consistent rollback on every rank")
        return _report(worker, False, "switch rolled back: " + (failure or "another rank failed to commit"))
    logger.info("Hot swap: candidates %s (generation %d, %.1f ms)",
                ",".join(sorted(st.on)) or "none", generation, st.toggle_ms)
    return _report(worker, True, "")


def get_arms(worker) -> dict[str, Any]:
    if _STATE.listed:
        try:
            _resolve(worker.model_runner)
        except RuntimeError as error:
            return _report(worker, False, str(error))
    return _report(worker, True, "")
