# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's MoE layer classes (vllm_sm80/layers/moe) behind vLLM's
out-of-tree layer registry: class identity, idempotent registration, and that
every override is upstream's code path with the fork switches off. CPU only."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.custom_op as custom_op
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm_sm80 import plugin
from vllm_sm80.layers.moe.gate import Sm80GateLinear
from vllm_sm80.layers.moe.routed_experts import Sm80RoutedExperts
from vllm_sm80.layers.moe.router import Sm80GroupedTopKRouter
from vllm_sm80.layers.moe.runner import Sm80MoERunner
from vllm_sm80.layers.moe.shared_experts import Sm80SharedExperts

SWITCHES = (
    "VLLM_SHARED_EXPERT_REORDER",
    "VLLM_MOE_DECODE_ROUTE_V2",
    "VLLM_AMPERE_DECODE_IDX_GLUE",
    "VLLM_GLM5_DECODE_MOE_ROUTING",
    "VLLM_AMPERE_DECODE_KERNELS",
    "VLLM_MOE_MASK_PADDING",
)


@pytest.fixture
def registry(monkeypatch):
    """A private out-of-tree registry holding the fork's registrations."""
    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    monkeypatch.setenv("VLLM_SHARED_EXPERT_REORDER", "1")
    monkeypatch.setenv("VLLM_MOE_DECODE_ROUTE_V2", "1")
    monkeypatch.setenv("VLLM_MOE_MASK_PADDING", "1")
    plugin.register_moe_layers()
    return custom_op.op_registry_oot


def test_switches_off_register_nothing(monkeypatch):
    """With every switch off, every MoE class stays upstream's, including in
    the out-of-tree class checks MoE LoRA uses."""
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA  # noqa: F401

    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    plugin.register_moe_layers()
    plugin.register()
    assert custom_op.op_registry_oot == {}
    for upstream in (MoERunner, RoutedExperts, GateLinear):
        assert custom_op.maybe_get_oot_by_class(upstream) is upstream
        assert type(upstream.__new__(upstream)) is upstream


@pytest.mark.parametrize(
    "switch, names",
    [
        ("VLLM_SHARED_EXPERT_REORDER", {"MoERunner"}),
        ("VLLM_AMPERE_DECODE_KERNELS", {"MoERunner"}),
        ("VLLM_AMPERE_DECODE_IDX_GLUE", {"MoERunner"}),
        ("VLLM_MOE_DECODE_ROUTE_V2", {"MoERunner", "GateLinear"}),
        ("VLLM_MOE_MASK_PADDING", {"RoutedExperts"}),
    ],
)
def test_each_switch_registers_the_classes_it_needs(monkeypatch, switch, names):
    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(switch, "1")
    plugin.register_moe_layers()
    assert set(custom_op.op_registry_oot) == names
    for name, layer_cls in custom_op.op_registry_oot.items():
        assert layer_cls.__name__ == "Sm80" + name
        assert custom_op.maybe_get_oot_by_class(layer_cls.__mro__[1]) is layer_cls


@pytest.fixture
def off(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    import vllm_sm80.layers.moe.routed_experts as re_mod

    monkeypatch.setattr(re_mod, "_MASK_PADDING", None)


def test_registration_is_idempotent(registry, monkeypatch):
    before = dict(registry)
    plugin.register_moe_layers()
    plugin.register()
    assert registry == before
    assert registry == {
        "MoERunner": Sm80MoERunner,
        "RoutedExperts": Sm80RoutedExperts,
        "GateLinear": Sm80GateLinear,
    }


@pytest.mark.parametrize(
    "upstream, fork",
    [(MoERunner, Sm80MoERunner), (RoutedExperts, Sm80RoutedExperts),
     (GateLinear, Sm80GateLinear)],
)
def test_upstream_class_instantiates_the_fork_class(registry, upstream, fork):
    obj = upstream.__new__(upstream)
    assert type(obj) is fork and isinstance(obj, upstream)


def test_runner_subclasses_keep_their_class(registry):
    class OtherRunner(MoERunner):
        pass

    assert type(OtherRunner.__new__(OtherRunner)) is OtherRunner


def _router():
    return GroupedTopKRouter(
        top_k=2, global_num_experts=8, num_expert_group=2, topk_group=1,
        renormalize=True, scoring_func="sigmoid",
        e_score_correction_bias=torch.zeros(8),
    )


def _runner(monkeypatch, gate=None, shared=True):
    """A real Sm80MoERunner.__init__ over light stand-ins."""
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    monkeypatch.setattr(mr_mod, "register_layer_for_moe_forward_op",
                        lambda *a: None)
    monkeypatch.setattr(mr_mod, "get_current_vllm_config", lambda: None)
    layer = torch.nn.Linear(4, 4) if shared else None
    moe_config = SimpleNamespace(moe_parallel_config=None)
    runner = Sm80MoERunner.__new__(Sm80MoERunner)
    Sm80MoERunner.__init__(
        runner, "layers.0.mlp.experts", moe_config, _router(),
        torch.nn.Identity(), gate=gate, shared_experts=layer,
    )
    return runner


def test_runner_swaps_children_in_place(monkeypatch, off):
    runner = _runner(monkeypatch)
    assert type(runner._shared_experts) is Sm80SharedExperts
    assert type(runner.router) is Sm80GroupedTopKRouter
    assert runner._shared_experts._reorder_after_routed is False
    # Same module tree and parameter names as upstream's runner.
    names = [n for n, _ in runner.named_parameters()]
    assert names == ["_shared_experts._layer.weight", "_shared_experts._layer.bias"]
    assert [n for n, _ in runner.named_modules()] == [
        "", "routed_experts", "_shared_experts", "_shared_experts._layer"
    ]


def test_runner_entry_is_upstreams_with_glue_off(monkeypatch, off):
    runner = _runner(monkeypatch)
    upstream = MoERunner._select_forward(runner)
    assert runner._forward_entry is upstream
    assert runner._moe_sum_glue is False


def test_route_v2_gate_gets_its_router_only_with_the_switch(monkeypatch, off):
    gate = torch.nn.Module.__new__(Sm80GateLinear)
    torch.nn.Module.__init__(gate)
    runner = _runner(monkeypatch, gate=gate)
    assert gate._route_v2_router is None
    monkeypatch.setenv("VLLM_MOE_DECODE_ROUTE_V2", "1")
    gate2 = torch.nn.Module.__new__(Sm80GateLinear)
    torch.nn.Module.__init__(gate2)
    runner = _runner(monkeypatch, gate=gate2)
    assert gate2._route_v2_router is runner.router


def test_gate_without_router_is_upstream_forward(monkeypatch):
    calls = []
    monkeypatch.setattr(GateLinear, "forward",
                        lambda self, x: calls.append(x) or ("up", None))
    gate = torch.nn.Module.__new__(Sm80GateLinear)
    torch.nn.Module.__init__(gate)
    x = torch.zeros(2, 4)
    assert gate.forward(x) == ("up", None)
    assert calls == [x]


def test_gate_with_router_falls_back_when_route_v2_declines(monkeypatch):
    import vllm_sm80.kernels.moe as moe_kernels

    monkeypatch.setattr(GateLinear, "forward", lambda self, x: ("up", None))
    monkeypatch.setattr(moe_kernels, "maybe_moe_route_v2", lambda g, r, x: None)
    gate = torch.nn.Module.__new__(Sm80GateLinear)
    torch.nn.Module.__init__(gate)
    gate.return_bias = True
    gate.attach_router(object())
    assert gate.forward(torch.zeros(2, 4)) == ("up", None)
    logits = torch.ones(2, 8)
    monkeypatch.setattr(moe_kernels, "maybe_moe_route_v2",
                        lambda g, r, x: logits)
    out = gate.forward(torch.zeros(2, 4))
    assert out[0] is logits and out[1] is None


def test_router_delegates_with_switches_off(monkeypatch, off):
    seen = []
    monkeypatch.setattr(
        GroupedTopKRouter, "_compute_routing",
        lambda self, h, l, t, *, input_ids=None: seen.append((h, l, t, input_ids))
        or "up",
    )
    fork = Sm80GroupedTopKRouter.adopt(_router())
    h, l = torch.zeros(3, 4), torch.zeros(3, 8)
    assert fork._compute_routing(h, l, None, input_ids="ids") == "up"
    assert seen == [(h, l, None, "ids")]


def test_routed_experts_delegate_with_mask_off(monkeypatch, off):
    seen = []
    monkeypatch.setattr(RoutedExperts, "forward_modular",
                        lambda self, **kw: seen.append(kw) or "out")
    ex = torch.nn.Module.__new__(Sm80RoutedExperts)
    ids = torch.arange(8, dtype=torch.int32).view(4, 2)
    ref = ids.clone()
    kw = dict(x=torch.zeros(4, 2), topk_weights=torch.ones(4, 2), topk_ids=ids,
              shared_experts=None, shared_experts_input=None)
    assert ex.forward_modular(**kw) == "out"
    assert seen == [kw] and seen[0]["topk_ids"] is ids and torch.equal(ids, ref)


def test_shared_experts_delegate_with_reorder_off(monkeypatch, off):
    calls = []
    monkeypatch.setattr(SharedExperts, "maybe_forward_async",
                        lambda self, x: calls.append(("async", x)) or False)
    monkeypatch.setattr(SharedExperts, "wait",
                        lambda self: calls.append(("wait",)))
    shared = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(shared)
    Sm80SharedExperts.adopt(shared)
    x = torch.zeros(1)
    assert shared.maybe_forward_async(x) is False
    shared.wait()
    assert calls == [("async", x), ("wait",)]


def test_moesum_wrapper_returns_the_fused_sum(monkeypatch):
    """Mocked dispatch: the entry offers its unsummed buffer; the wrapper
    returns (None, shared + sum) as forward's own add would compute it."""
    import vllm_sm80.kernels.moe.sum as sum_mod

    monkeypatch.setattr(sum_mod, "moe_sum_add",
                        lambda inp, sh: sh + inp.sum(dim=1))
    runner = torch.nn.Module.__new__(Sm80MoERunner)
    torch.nn.Module.__init__(runner)
    runner._moe_sum_og_pre = None
    monkeypatch.setattr(Sm80MoERunner, "_idx_glue_arm_moe_sum",
                        lambda self, h, og: True)
    inp = torch.randn(3, 4, 6)
    shared = torch.randn(3, 6)
    order = []

    def entry(*args):
        assert sum_mod.MOE_SUM_DEFERRAL.offer(inp, torch.empty(3, 6))
        order.append("entry")
        return shared, torch.empty(3, 6)

    out = runner._moe_sum_forward_entry(entry, torch.zeros(3, 6), None, None,
                                        None, "x", 0)
    assert order == ["entry"] and out[0] is None
    torch.testing.assert_close(out[1], shared + inp.sum(dim=1))
    assert sum_mod.MOE_SUM_DEFERRAL.take() is None


def test_upstream_call_order_the_overrides_rely_on():
    impl = inspect.getsource(MoERunner._forward_impl)
    assert (impl.index(".maybe_forward_async(")
            < impl.index("router_logits, _ = self.gate(hidden_states)")
            < impl.index("self._apply_quant_method("))
    aq = inspect.getsource(MoERunner._apply_quant_method)
    assert (aq.index("self.router.select_experts(")
            < aq.index("self.routed_experts.forward_modular(")
            < aq.index("self._shared_experts.wait()")
            < aq.index("self._shared_experts.output"))
    fwd = inspect.getsource(MoERunner.forward)
    assert (fwd.index("self._maybe_pad_hidden_states(")
            < fwd.index("self._forward_entry(")
            < fwd.index("_unpack(result)")
            < fwd.index("result = shared_output + fused_output"))
    init = inspect.getsource(MoERunner.__init__)
    assert init.count("self._select_forward()") == 1
