# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine switches swapped between steps inside one server
(``vllm_sm80.hot_swap``), on CPU: the CUDA driver calls are replaced by a
recording fake, the model by stand-ins."""
import ast
import inspect
import sys
import types
from types import SimpleNamespace

import pytest

from vllm_sm80 import dev_ab, hot_swap

RUNNER = SimpleNamespace(
    vllm_config=SimpleNamespace(speculative_config=None),
    adaptive_verification=None, speculator=None,
)


class _Trap:
    def __getattr__(self, name):
        raise AssertionError(f"touched .{name}")


@pytest.fixture
def applied(monkeypatch):
    """Calls of the eager pair's apply hook."""
    calls = []
    monkeypatch.setitem(hot_swap.HOT_PAIRS, "graph_pair",
                        hot_swap.HotPair("graph_pair", "VLLM_FAKE_GRAPH", lambda runner: None))
    monkeypatch.setitem(hot_swap.HOT_PAIRS, "eager_pair", hot_swap.HotPair(
        "eager_pair", "VLLM_FAKE_EAGER", lambda runner: None,
        lambda runner, on: calls.append(on), eager=True))
    return calls


@pytest.fixture
def state(monkeypatch, applied):
    fresh = hot_swap._HotState(listed=("graph_pair", "eager_pair"))
    monkeypatch.setattr(hot_swap, "_STATE", fresh)
    return fresh


class _Tensor:
    def __init__(self, ptr, shape=(4, 8)):
        self.ptr, self.shape = ptr, shape

    def data_ptr(self):
        return self.ptr


class _Graph:
    """A capturing graph whose node set grows as variants launch."""

    def __init__(self):
        self.nodes: set[int] = set()
        self.next = 1

    def launch(self, n):
        new = set(range(self.next, self.next + n))
        self.next += n
        self.nodes |= new


@pytest.fixture
def capturing(monkeypatch, state):
    graph = _Graph()
    monkeypatch.setattr(hot_swap, "_capturing_graph", lambda: 77)
    monkeypatch.setattr(hot_swap, "_nodes", lambda g: set(graph.nodes))
    state.window = True
    return graph


def test_unset_touches_nothing(monkeypatch):
    monkeypatch.setattr(hot_swap, "_STATE", hot_swap._HotState())
    assert not hot_swap.configure(_Trap())
    assert not hot_swap.listed("shard_argmax")
    assert not hot_swap.eager_on("shard_argmax")
    assert not hot_swap.active()


def test_parse_refusals():
    assert hot_swap._parse("shard_argmax, kda_prefill") == ("shard_argmax", "kda_prefill")
    with pytest.raises(ValueError, match="not in"):
        hot_swap._parse("shard_argmax,nope")
    with pytest.raises(ValueError, match="twice"):
        hot_swap._parse("shard_argmax,shard_argmax")
    with pytest.raises(ValueError, match="share one site"):
        hot_swap._parse("topk_canonical_ties,topk_sorted")


def test_configure_refusals(monkeypatch):
    monkeypatch.setattr(hot_swap, "_STATE", hot_swap._HotState(listed=("shard_argmax",)))
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "kda_prefill")
    with pytest.raises(ValueError, match="VLLM_DEV_AB_SWITCH"):
        hot_swap.configure(_Trap())
    monkeypatch.delenv("VLLM_DEV_AB_SWITCH")
    monkeypatch.setenv("VLLM_SM80_CAPTURE", "spec")
    with pytest.raises(ValueError, match="VLLM_SM80_CAPTURE"):
        hot_swap.configure(_Trap())
    monkeypatch.delenv("VLLM_SM80_CAPTURE")
    monkeypatch.setenv("VLLM_SM80_SHARD_ARGMAX", "1")
    with pytest.raises(ValueError, match="must be off"):
        hot_swap.configure(_Trap())
    monkeypatch.setenv("VLLM_SM80_SHARD_ARGMAX", "0")
    assert hot_swap.configure(_Trap())


def test_dev_ab_injects_the_extension_for_hot_pairs(monkeypatch, state):
    config = SimpleNamespace(parallel_config=SimpleNamespace(worker_extension_cls=""))
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    monkeypatch.delenv("VLLM_DEV_AB_SWITCH", raising=False)
    dev_ab.configure(config)
    assert config.parallel_config.worker_extension_cls == dev_ab.EXTENSION_QUALNAME
    assert hasattr(dev_ab.DevABWorkerExtension, "set_hot_arms")
    assert hasattr(dev_ab.DevABWorkerExtension, "get_hot_arms")


def test_dev_ab_capture_wrapper_routes_to_the_hot_window(monkeypatch, state):
    seen = []
    monkeypatch.setattr(hot_swap, "capture",
                        lambda runner, body, profile_only: seen.append(profile_only) or 7)
    wrapped = dev_ab.with_ab_capture(lambda runner, *, profile_only=False: 0)
    assert wrapped(RUNNER, profile_only=True) == 7 and seen == [True]


def test_eager_site_follows_the_arm(state):
    base = _Tensor(1)
    cand_out = _Tensor(2)
    calls = []

    def run_base():
        calls.append("base")
        return base

    def run_cand(out):
        calls.append(("cand", out))
        return cand_out

    assert hot_swap.pair("graph_pair", run_base, run_cand) is base
    state.on = frozenset({"graph_pair"})
    assert hot_swap.pair("graph_pair", run_base, run_cand) is cand_out
    assert calls == ["base", ("cand", None)]
    # A declining candidate keeps the incumbent.
    assert hot_swap.pair("graph_pair", run_base, lambda out: None) is base
    assert state.reached == {"graph_pair"}


def test_eager_gate_follows_the_arm_and_refuses_capture(monkeypatch, state):
    assert not hot_swap.eager_on("eager_pair")
    state.on = frozenset({"eager_pair"})
    assert hot_swap.eager_on("eager_pair")
    assert not hot_swap.eager_on("not_listed")
    assert state.reached == {"eager_pair"}
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        hot_swap.eager_on("eager_pair")


def test_capture_outside_the_window_is_refused(monkeypatch, state):
    """A graph captured outside the window would hold only one arm's variant."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside the hot capture window"):
        hot_swap.pair("graph_pair", lambda: _Tensor(1), lambda out: out)


def test_capture_records_each_variants_nodes(capturing, state):
    out = _Tensor(5)

    def base():
        capturing.launch(1)
        return out

    def cand(o):
        assert o is out
        capturing.launch(2)
        return _Tensor(5)

    capturing.launch(3)  # work before the site
    assert hot_swap.pair("graph_pair", base, cand) is out
    capturing.launch(1)  # work after it
    hot_swap.pair("eager_pair", base, lambda o: None)  # declines: no site
    (site,) = state.pending[77]
    assert site == hot_swap._Site("graph_pair", (4,), (5, 6))


def test_capture_refusals(capturing, state):
    out = _Tensor(5)
    with pytest.raises(RuntimeError, match="write the incumbent's output"):
        hot_swap.pair("graph_pair", lambda: out, lambda o: _Tensor(6))

    def declining(o):
        capturing.launch(1)
        return None

    with pytest.raises(RuntimeError, match="declining candidate launched"):
        hot_swap.pair("graph_pair", lambda: out, declining)

    def nested(o):
        return hot_swap.pair("eager_pair", lambda: out, lambda o2: o2)

    with pytest.raises(RuntimeError, match="cannot nest"):
        hot_swap.pair("graph_pair", lambda: out, nested)
    assert not state.in_pair


class _FakeDriver:
    """cuda.bindings.driver stand-in recording enable flags per (exec, node)."""

    def __init__(self):
        self.enabled: dict[tuple[int, int], int] = {}
        self.uploads: list[int] = []

    CUgraphNode = staticmethod(int)
    CUgraphExec = staticmethod(int)
    CUstream = staticmethod(int)

    def cuGraphNodeSetEnabled(self, ex, node, flag):
        self.enabled[(ex, node)] = flag
        return (0,)

    def cuGraphUpload(self, ex, stream):
        self.uploads.append(ex)
        return (0,)


class _CudaGraph:
    def __init__(self, handle, exec_handle):
        self.handle, self.exec_handle = handle, exec_handle

    def raw_cuda_graph(self):
        return self.handle

    def raw_cuda_graph_exec(self):
        return self.exec_handle


@pytest.fixture
def driver(monkeypatch):
    fake = _FakeDriver()
    monkeypatch.setattr(hot_swap, "_drv", lambda: fake)
    monkeypatch.setattr(hot_swap, "_node_kinds", lambda nodes: ["k"] * len(nodes))
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
    return fake


def test_bind_starts_on_the_incumbents_and_toggles(state, driver, applied):
    state.pending[10] = [hot_swap._Site("graph_pair", (1,), (2, 3)),
                         hot_swap._Site("graph_pair", (4,), (5,))]
    graph = _CudaGraph(10, 900)
    hot_swap._bind(graph)
    assert driver.enabled == {(900, 1): 1, (900, 2): 0, (900, 3): 0, (900, 4): 1, (900, 5): 0}
    assert not state.pending

    hot_swap._apply(RUNNER, frozenset({"graph_pair"}))
    assert driver.enabled == {(900, 1): 0, (900, 2): 1, (900, 3): 1, (900, 4): 0, (900, 5): 1}
    assert driver.uploads == [900]
    assert state.on == {"graph_pair"} and applied == []
    # Only what changes arm is touched: the eager pair's state, no node.
    driver.enabled.clear()
    hot_swap._apply(RUNNER, frozenset({"graph_pair", "eager_pair"}))
    assert driver.enabled == {} and applied == [True]
    # A forced apply rewrites every site and every eager state.
    hot_swap._apply(RUNNER, frozenset({"graph_pair", "eager_pair"}), force=True)
    assert len(driver.enabled) == 5 and applied == [True, True]
    report = hot_swap._report(SimpleNamespace(rank=0), True, "")
    assert report["sites"]["graph_pair"] == {
        "sites": 2, "base_nodes": 2, "cand_nodes": 3, "base_kernels": ["k"],
        "cand_kernels": ["k"], "eager": False}
    assert report["graphs"] == 1 and report["on"] == ["eager_pair", "graph_pair"]
    # A released graph drops out.
    del graph
    hot_swap._prune()
    assert hot_swap._report(SimpleNamespace(rank=0), True, "")["graphs"] == 0


def test_partial_toggle_is_repaired_by_force(state, driver):
    """A toggle that fails half-way leaves ex.on stale; a forced toggle
    rewrites every site."""
    state.pending[10] = [hot_swap._Site("graph_pair", (1,), (2,))]
    graph = _CudaGraph(10, 900)
    hot_swap._bind(graph)
    (ex,) = state.execs
    driver.enabled[(900, 1)], driver.enabled[(900, 2)] = 0, 0  # half-way: neither runs
    hot_swap._toggle(ex, frozenset(), force=True)
    assert driver.enabled == {(900, 1): 1, (900, 2): 0}


def _worker(resident=0):
    runner = SimpleNamespace(req_states=SimpleNamespace(req_id_to_index=dict.fromkeys(range(resident))))
    return SimpleNamespace(rank=0, model_runner=runner)


_CARRIED_ANYWHERE = hot_swap._carried_anywhere


@pytest.fixture
def transaction(monkeypatch, state):
    applied = []
    state.resolved = True
    state.reached = {"graph_pair", "eager_pair"}
    monkeypatch.setattr(dev_ab, "_transaction_vote", lambda ok: ok)
    monkeypatch.setattr(hot_swap, "_all_ranks_same_state", lambda: True)
    monkeypatch.setattr(hot_swap, "_gather", lambda value: [value])
    monkeypatch.setattr(hot_swap, "_carried_anywhere", lambda: set(state.listed))

    def apply(runner, on, force=False):
        applied.append((on, force) if force else on)
        state.on = on

    monkeypatch.setattr(hot_swap, "_apply", apply)
    return applied


def test_set_arms_commits(transaction, state):
    reply = hot_swap.set_arms(_worker(), ["graph_pair"], 1)
    assert reply["ok"] and reply["on"] == ["graph_pair"] and reply["generation"] == 1
    assert transaction == [frozenset({"graph_pair"})]
    reply = hot_swap.set_arms(_worker(), [], "2")
    assert reply["ok"] and reply["on"] == [] and reply["generation"] == 2


@pytest.mark.parametrize("on, generation, resident, why", [
    (["nope"], 1, 0, "not carried"),
    ("graph_pair", 1, 0, "list of pair names"),
    (["graph_pair"], 0, 0, "not newer"),
    (["graph_pair"], 1, 2, "drain first"),
])
def test_set_arms_refusals(transaction, state, on, generation, resident, why):
    reply = hot_swap.set_arms(_worker(resident), on, generation)
    assert not reply["ok"] and why in reply["reason"]
    assert transaction == [] and state.generation == 0


def test_set_arms_refuses_an_unavailable_pair(monkeypatch, transaction, state):
    state.resolved = False
    monkeypatch.setitem(hot_swap.HOT_PAIRS, "graph_pair", hot_swap.HotPair(
        "graph_pair", "VLLM_FAKE_GRAPH", lambda runner: "no such layer"))
    reply = hot_swap.set_arms(_worker(), ["graph_pair"], 1)
    assert not reply["ok"] and "no such layer" in reply["reason"]
    reply = hot_swap.get_arms(_worker())
    assert not reply["ok"] and "no such layer" in reply["reason"]


def test_set_arms_rolls_back_a_failed_commit(monkeypatch, transaction, state):
    def failing(runner, on, force=False):
        transaction.append((on, force))
        if on:
            raise RuntimeError("boom")
        state.on = on

    monkeypatch.setattr(hot_swap, "_apply", failing)
    reply = hot_swap.set_arms(_worker(), ["graph_pair"], 1)
    assert not reply["ok"] and "rolled back: boom" in reply["reason"]
    assert state.on == frozenset() and state.generation == 0
    # The rollback rewrites every site's flags, not only those it believes changed.
    assert transaction == [(frozenset({"graph_pair"}), False), (frozenset(), True)]


def test_set_arms_refuses_a_pair_no_rank_carries(monkeypatch, transaction, state):
    monkeypatch.setattr(hot_swap, "_carried_anywhere", _CARRIED_ANYWHERE)
    state.reached = set()
    reply = hot_swap.set_arms(_worker(), ["graph_pair"], 1)
    assert not reply["ok"] and "no rank has a captured pair site" in reply["reason"]
    # A graph pair is not carried by an eager call alone ...
    state.reached = {"graph_pair"}
    assert hot_swap._carried_anywhere() == set()
    # ... an eager pair is, and so is one another rank carries.
    state.reached = {"eager_pair"}
    assert hot_swap._carried_anywhere() == {"eager_pair"}
    monkeypatch.setattr(hot_swap, "_gather", lambda value: [value, ["graph_pair"]])
    assert hot_swap._carried_anywhere() == {"eager_pair", "graph_pair"}
    assert transaction == []


@pytest.mark.parametrize("field, value, why", [
    ("adaptive_verification", object(), "adaptive verification profiles"),
    ("speculator", SimpleNamespace(enable_adaptive_verification=True), "drafter state"),
])
def test_capture_refuses_cost_profiling_policies(state, field, value, why):
    runner = SimpleNamespace(**{**vars(RUNNER), field: value})
    with pytest.raises(RuntimeError, match=why):
        hot_swap.capture(runner, lambda r, *, profile_only: 0, False)
    assert not state.window


def test_capture_window_patches_and_restores_the_graph_class(state):
    torch = pytest.importorskip("torch")
    original = torch.cuda.CUDAGraph
    seen = {}

    def body(runner, *, profile_only):
        seen["cls"] = torch.cuda.CUDAGraph
        seen["window"] = state.window
        return 3

    state.resolved = True
    assert hot_swap.capture(RUNNER, body, False) == 3
    assert seen["window"] and issubclass(seen["cls"], original) and seen["cls"] is not original
    assert torch.cuda.CUDAGraph is original and not state.window

    def failing(runner, *, profile_only):
        state.pending[1] = [hot_swap._Site("graph_pair", (1,), (2,))]
        return 0

    with pytest.raises(RuntimeError, match="outside a bound graph"):
        hot_swap.capture(RUNNER, failing, True)
    assert torch.cuda.CUDAGraph is original and not state.pending


def test_hot_pairs_are_ast_readable():
    """The allowlist is read without importing the engine: every HotPair
    call names its pair and switch as string literals."""
    tree = ast.parse(inspect.getsource(hot_swap))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "HotPair"]
    names = {c.args[0].value: c.args[1].value for c in calls}
    assert names == {
        "shard_argmax": "VLLM_SM80_SHARD_ARGMAX",
        "prefill_reduce_overlap": "VLLM_SM80_PREFILL_REDUCE_OVERLAP",
        "kda_prefill": "VLLM_SM80_KDA_PREFILL",
        "mhc_prefill": "VLLM_SM80_MHC_PREFILL",
        "sparse_mla_prefill": "VLLM_SM80_SPARSE_MLA_PREFILL",
        "topk_canonical_ties": "VLLM_SM80_TOPK_CANONICAL_TIES",
        "topk_sorted": "VLLM_SM80_TOPK_SORTED",
    }
    assert set(names) == set(hot_swap.HOT_PAIRS)
    from vllm_sm80 import envs as sm80_envs

    assert set(names.values()) <= set(sm80_envs.FORK_ENV_VARS)
    assert "VLLM_HOT_SWAP" in sm80_envs.FORK_ENV_VARS


# ---------------------------------------------------------------------------
# Pair adapters
# ---------------------------------------------------------------------------


@pytest.fixture
def listed(monkeypatch):
    def make(*names):
        fresh = hot_swap._HotState(listed=names)
        monkeypatch.setattr(hot_swap, "_STATE", fresh)
        return fresh

    return make


class _Module:
    def __init__(self, children=(), **attrs):
        self.children_ = list(children)
        self.__dict__.update(attrs)

    def modules(self):
        yield self
        for child in self.children_:
            yield from child.modules()


def test_split_reduce_pair_toggles_the_layers_that_issue_their_sums(monkeypatch, listed):
    st = listed("prefill_reduce_overlap")
    model = types.ModuleType("fake_model")
    split = types.ModuleType("fake_split")

    class DeepseekV4DecoderLayer(_Module):
        pass

    model.DeepseekV4DecoderLayer = DeepseekV4DecoderLayer
    split.splits = lambda layer: layer.issues
    monkeypatch.setitem(sys.modules, hot_swap._DSV41_MODEL, model)
    monkeypatch.setitem(sys.modules, hot_swap._DSV41_SPLIT, split)
    layers = [DeepseekV4DecoderLayer(issues=i != 1, _split_reduce=False) for i in range(3)]
    runner = SimpleNamespace(model=_Module(layers))
    pair = hot_swap.HOT_PAIRS["prefill_reduce_overlap"]
    assert pair.unavailable(runner) is None
    pair.apply(runner, True)
    assert [layer._split_reduce for layer in layers] == [True, False, True]
    pair.apply(runner, False)
    assert not any(layer._split_reduce for layer in layers)
    assert hot_swap._carried_eagerly("prefill_reduce_overlap")
    # Without the layer-owned all-reduce no layer can split.
    st.targets.clear()
    for layer in layers:
        layer.issues = False
    assert "VLLM_SM80_ALLREDUCE=1" in pair.unavailable(runner)


def test_split_reduce_predicate_is_the_adoption_rule():
    from vllm_sm80.models.deepseek_v41 import sm80_split_reduce as split

    taker = SimpleNamespace(ffn=object.__new__(split.Sm80DeepseekV41MoE),
                            attn=SimpleNamespace(reduces_wo_b=lambda: True))
    other = SimpleNamespace(ffn=object(), attn=SimpleNamespace(reduces_wo_b=lambda: True))
    assert split.splits(taker) and not split.splits(other)
    split.adopt_split_reduce([taker, other])
    assert taker._split_reduce and not hasattr(other, "_split_reduce")


def test_kda_pair_resolves_the_layer_gates_off_and_warms_once(monkeypatch, listed):
    listed("kda_prefill")
    kda = types.ModuleType("fake_kda")
    warmed = []
    kda.supported_heads = lambda: (64, 16)
    kda.use_for_layer = lambda backend, heads, *rest, allowed_heads: (
        backend == "auto" and heads in allowed_heads)
    kda.WARMUP_PLANS = ([64],)
    kda.warmup = lambda plans, device, heads: warmed.append((device, heads))
    monkeypatch.setitem(sys.modules, hot_swap._KDA_PREFILL, kda)
    fields = dict(head_dim=128, kda_safe_gate=True, kda_lower_bound=None, _pp_kda_prefill=False)
    layers = [_Module(kda_prefill_backend="auto", local_num_heads=16, **fields),
              _Module(kda_prefill_backend="flashkda", local_num_heads=16, **fields)]
    runner = SimpleNamespace(
        model=_Module(layers), device="cuda:0",
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype="bf16")))
    pair = hot_swap.HOT_PAIRS["kda_prefill"]
    assert pair.unavailable(runner) is None
    assert not layers[0]._pp_kda_prefill
    pair.apply(runner, True)
    pair.apply(runner, False)
    pair.apply(runner, True)
    assert [layer._pp_kda_prefill for layer in layers] == [True, False]
    assert warmed == [("cuda:0", 16)]


def test_argmax_pair_refuses_drafts_replaced_at_load(monkeypatch, listed):
    from vllm_sm80.spec_decode import shard_argmax
    from vllm_sm80.worker import target_argmax

    listed("shard_argmax")
    pair = hot_swap.HOT_PAIRS["shard_argmax"]
    monkeypatch.setattr(shard_argmax, "_kind", lambda speculator: "sample_draft")
    assert "replacing its sample_draft at load" in pair.unavailable(RUNNER)
    monkeypatch.setattr(shard_argmax, "_kind", lambda speculator: "markov")
    monkeypatch.setattr(target_argmax, "refusal", lambda runner: "TP 1")
    assert pair.unavailable(RUNNER) == "TP 1"
    monkeypatch.setattr(target_argmax, "refusal", lambda runner: None)
    assert pair.unavailable(RUNNER) is None
    calls = []
    monkeypatch.setattr(target_argmax, "set_enabled", lambda runner, on: calls.append(on))
    monkeypatch.setattr(target_argmax, "warm", lambda runner: calls.append("warm"))
    pair.apply(RUNNER, True)
    pair.apply(RUNNER, False)
    assert calls == [True, "warm", False, "warm"]


def test_sparse_mla_pair_targets_the_backend_layers(listed):
    listed("sparse_mla_prefill")
    impl = type("Impl", (), {"__module__": hot_swap._SPARSE_MLA_BACKEND})()
    runner = SimpleNamespace(model=_Module([_Module(impl=impl), _Module(impl=object())]))
    assert hot_swap.HOT_PAIRS["sparse_mla_prefill"].unavailable(runner) is None
    assert len(hot_swap._targets("sparse_mla_prefill")) == 1


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


class _Sites(list):
    """hot_swap.pair calls, each run on ``arm`` (False: the incumbent)."""

    arm = False

    def __call__(self, name, base, cand):
        self.append(name)
        return cand(None) if self.arm else base()


@pytest.fixture
def sites(monkeypatch):
    seen = _Sites()
    monkeypatch.setattr(hot_swap, "pair", seen)
    return seen


@pytest.mark.parametrize("pair, booted, base, cand", [
    ("topk_canonical_ties", (False, True), ["sort"], [("tiefix", True, 8)]),
    ("topk_canonical_ties", (False, False), [], [("tiefix", False, 0)]),
    ("topk_sorted", (True, False), [("tiefix", False, None)], [("tiefix", True, 8)]),
    ("topk_sorted", (False, False), [], ["sort"]),
])
def test_topk_site_runs_the_booted_and_the_candidate_switches(
        monkeypatch, listed, sites, pair, booted, base, cand):
    from vllm_sm80.kernels.indexer import topk_tiefix_gate as gate

    listed(pair)
    calls = []
    monkeypatch.setattr(gate, "use_tiefix_topk", lambda: booted[0])
    monkeypatch.setattr(gate, "use_sorted_topk", lambda: booted[1])
    monkeypatch.setattr(gate.tuning, "lookup", lambda op, param, key, default: 8)
    monkeypatch.setattr(gate, "tiefix_topk_", lambda logits, idx, *, indexer, row_ends,
                        row_starts, relative, sort, split_rows:
                        calls.append(("tiefix", sort, split_rows)))
    topk_tiefix = types.ModuleType("topk_tiefix")
    topk_tiefix.sort_topk_rows_ = lambda idx: calls.append("sort")
    monkeypatch.setitem(sys.modules, "vllm_sm80.kernels.indexer.topk_tiefix", topk_tiefix)
    idx = object()
    gate.canonical_topk_(object(), _IndexRows(idx), indexer="kpool", row_ends=None)
    assert sites == [pair] and calls == base
    calls.clear()
    sites.arm = True
    gate.canonical_topk_(object(), _IndexRows(idx), indexer="kpool", row_ends=None)
    assert calls == cand


class _IndexRows:
    shape = (4, 512)

    def __init__(self, ptr):
        self.ptr = ptr


def test_topk_site_is_absent_without_a_pair(monkeypatch, listed, sites):
    from vllm_sm80.kernels.indexer import topk_tiefix_gate as gate

    listed()
    monkeypatch.setattr(gate, "use_tiefix_topk", lambda: False)
    monkeypatch.setattr(gate, "use_sorted_topk", lambda: False)
    gate.canonical_topk_(object(), _IndexRows(1), indexer="kpool", row_ends=None)
    gate.canonical_decode_topk_(object(), _IndexRows(1), None, 1, indexer="kpool")
    assert sites == []


@pytest.fixture
def sm80_platform(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_device_capability", lambda cap: cap == 80)


def test_mhc_prefill_gate_reads_the_arm_above_the_threshold(monkeypatch, listed, sm80_platform):
    from vllm_sm80.kernels import mhc

    monkeypatch.setenv("VLLM_SM80_MHC_PREFILL", "0")
    monkeypatch.setattr(mhc.tuning, "lookup", lambda op, param, key, default: 384)
    listed()
    assert not mhc.use_ampere_prefill_prenorm(4096)
    st = listed("mhc_prefill")
    assert not mhc.use_ampere_prefill_prenorm(4096)
    assert not mhc.use_ampere_prefill_prenorm(64) and st.reached == {"mhc_prefill"}
    st.on = frozenset({"mhc_prefill"})
    assert mhc.use_ampere_prefill_prenorm(4096)
    assert not mhc.use_ampere_prefill_prenorm(64)


def test_sparse_mla_prefill_gate_reads_the_arm_past_both_rows(monkeypatch, listed, sm80_platform):
    from vllm_sm80.kernels.attention import triton_mla_sparse as tms
    from vllm_sm80 import tuning

    monkeypatch.setenv("VLLM_SM80_SPARSE_MLA_PREFILL", "0")
    monkeypatch.setattr(tuning, "lookup",
                        lambda op, param, key, default: 384 if param.endswith("rows") else 4352)
    st = listed("sparse_mla_prefill")
    assert not tms._use_ampere_prefill_sparse_mla(2048, 8192, 2176, 512)
    st.on = frozenset({"sparse_mla_prefill"})
    assert tms._use_ampere_prefill_sparse_mla(2048, 8192, 2176, 512)
    assert not tms._use_ampere_prefill_sparse_mla(2048, 4000, 2176, 512)
    assert not tms._use_ampere_prefill_sparse_mla(64, 8192, 2176, 512)
    listed()
    assert not tms._use_ampere_prefill_sparse_mla(2048, 8192, 2176, 512)


def test_dspark_drafts_are_one_site(monkeypatch, listed, sites):
    from vllm_sm80.spec_decode import shard_argmax

    calls = []
    monkeypatch.setattr(shard_argmax, "sample_sequential",
                        lambda s, n, hidden: calls.append(("per_rank", n)))
    wrapped = shard_argmax.shard_argmax_sampling(
        lambda s, n, hidden: calls.append(("gathered", n)))
    speculator = SimpleNamespace(draft_tokens="tokens")
    listed("shard_argmax")
    assert wrapped(speculator, 3, None) is None
    sites.arm = True
    wrapped(speculator, 3, None)
    assert sites == ["shard_argmax", "shard_argmax"]
    assert calls == [("gathered", 3), ("per_rank", 3)]
    assert not hasattr(speculator, "_shard_argmax")


def test_hot_argmax_selects_the_fork_worker(monkeypatch, listed):
    from vllm_sm80.worker import cudagraph_mem

    monkeypatch.setenv("VLLM_SM80_SHARD_ARGMAX", "0")
    monkeypatch.setenv("VLLM_CUDAGRAPH_ESTIMATE_SPLIT", "0")
    listed("shard_argmax")
    config = SimpleNamespace(worker_cls=cudagraph_mem.UPSTREAM_WORKER)
    cudagraph_mem.select_worker(config)
    assert config.worker_cls == cudagraph_mem.WORKER
    # Another worker cannot serve the hot pair.
    with pytest.raises(ValueError, match="fork GPU worker"):
        cudagraph_mem.select_worker(SimpleNamespace(worker_cls="my.Worker"))
