# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DEEPSEEK_V41_ROUTER_MMA (vllm_sm80/envs.py, read in
vllm_sm80/models/deepseek_v41/sm80_moe.py) on CPU: declared, off by
default, read through vllm.envs; adopt_moe hands the layer to
_adopt_router_mma only when it is set; the adoption refuses without
VLLM_DEEPSEEK_V41_FUSED_ROUTE and VLLM_DEEPSEEK_V41_GATE_GEMV; a gate call
whose predecessor was never routed (an expert method reading the logits
itself) is refused.

The gate weight with the switch is held once, in BF16:

- the adopted gate has no other weight; a BF16-valued FP16 or BF16
  checkpoint tensor loads exactly (the FP32-held gate's values, and the
  bytes of the BF16 copy the gate made of its FP32 weight before), and a
  value BF16 does not hold is refused before it is copied;
- a call neither kernel serves (prefill rows, CPU) runs the FP32 fallback on
  the widened weight: the FP32-held gate's bytes, at 1 to 3000 rows;
- the tiny EXL3 checkpoint loads through the EXL3 loader, target and
  drafter, into BF16 gate weights whose FP32 values are the FP32-held gates'
  bytes, and its random FP16 gates (not BF16-valued) are refused;
- the bytes held and freed for the served shapes (40 target gates of
  384 x 5120, 3 drafter gates of 128 x 5120), logged by the boot warmup.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_router_mma_switch.py
"""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.models.deepseek_v41.sm80 import construct
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.deepseek_v41.sm80.test_exl3_construction import _load
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)

SWITCH = "VLLM_DEEPSEEK_V41_ROUTER_MMA"
NEEDS = ("VLLM_DEEPSEEK_V41_FUSED_ROUTE", "VLLM_DEEPSEEK_V41_GATE_GEMV")
HIDDEN = 5120


def _ffn(monkeypatch):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    construct.set_groups(monkeypatch, 1, 0)
    gate = GateLinear(64, 8, bias=False, out_dtype=torch.float32,
                      prefix="layers.0.ffn.gate")
    gate.e_score_correction_bias = torch.zeros(8)
    gate.tid2eid = None
    router = object.__new__(FusedTopKBiasRouter)
    router.num_fused_shared_experts = 0
    return SimpleNamespace(use_native_mega_moe=False, gate=gate,
                           scoring_func="sqrtsoftplus",
                           experts=SimpleNamespace(router=router))


def _bf16_valued_fp16(experts: int, hidden: int, seed: int) -> torch.Tensor:
    """An FP16 gate weight holding BF16 values (as the EXL3 export's do): every
    magnitude at least 1e-4, an FP16 normal, whose 8 significant bits FP16
    holds."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(experts, hidden, generator=g) * 0.01
    w = torch.where(w.abs() < 1e-4, torch.full_like(w, 1e-4).copysign(w), w)
    w16 = w.to(torch.bfloat16)
    ck = w16.half()
    assert torch.equal(ck.float(), w16.float())
    return ck


def test_switch_declared_and_off_by_default(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(SWITCH, raising=False)
    assert envs.VLLM_DEEPSEEK_V41_ROUTER_MMA is False
    monkeypatch.setenv(SWITCH, "1")
    assert envs.VLLM_DEEPSEEK_V41_ROUTER_MMA is True
    monkeypatch.setenv(SWITCH, "0")
    assert envs.VLLM_DEEPSEEK_V41_ROUTER_MMA is False


def test_adoption_takes_the_router_only_with_the_switch(monkeypatch):
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    for switch in (None, "0", "1"):
        if switch is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, switch)
        adopted = []
        monkeypatch.setattr(
            sm80_moe, "_adopt_router_mma",
            lambda ffn, router: adopted.append((ffn, router)),
        )  # fmt: skip
        ffn = _ffn(monkeypatch)
        sm80_moe.adopt_moe(ffn, SimpleNamespace())
        if switch == "1":
            assert adopted == [(ffn, ffn.experts.router)]
        else:
            assert adopted == []
        assert type(ffn.experts.router) is sm80_moe.Sm80DeepseekV41Router
        assert ffn.experts.router.mma_gate is None
        assert ffn.gate.mma is False
        held = torch.bfloat16 if switch == "1" else torch.float32
        assert ffn.gate.weight.dtype == held


@pytest.mark.parametrize("route,gemv", [("0", "0"), ("1", "0"), ("0", "1")])
def test_adoption_needs_the_fused_route_and_the_gate_gemv(monkeypatch, route, gemv):
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_ROUTE", route)
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_GATE_GEMV", gemv)
    missing = [n for n, v in (("VLLM_DEEPSEEK_V41_FUSED_ROUTE", route),
                              ("VLLM_DEEPSEEK_V41_GATE_GEMV", gemv)) if v == "0"]
    with pytest.raises(ValueError) as refused:
        sm80_moe._adopt_router_mma(SimpleNamespace(), SimpleNamespace())
    assert str(refused.value).startswith("VLLM_DEEPSEEK_V41_ROUTER_MMA needs ")
    assert all(f"{n}=1" in str(refused.value) for n in missing)
    # With both set the checks go on to the layer (here: no gate at all).
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_ROUTE", "1")
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_GATE_GEMV", "1")
    with pytest.raises(AttributeError):
        sm80_moe._adopt_router_mma(SimpleNamespace(), SimpleNamespace())


def test_a_gate_call_whose_predecessor_was_not_routed_is_refused(monkeypatch):
    from vllm_sm80.models.deepseek_v41 import sm80_moe
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma

    gate = _ffn(monkeypatch).gate
    sm80_moe._hold_gate_weight(gate, torch.bfloat16)
    gate.mma = True
    monkeypatch.setattr(sm80_router_mma, "serves", lambda x: True)
    monkeypatch.setattr(sm80_router_mma, "gate", lambda x, w: (x, "pending"))
    gate.forward(torch.zeros(2, 64))
    assert gate.pending == "pending"
    with pytest.raises(RuntimeError, match="previous gate call was not routed"):
        gate.forward(torch.zeros(2, 64))
    gate.pending = None  # the router consumed it
    gate.forward(torch.zeros(2, 64))


def _adopt_mma(monkeypatch):
    """A layer adopted with the switch (the tensor-core path's shape check
    and warmup registration stubbed: the test gate is 8 x 64)."""
    from vllm_sm80.models.deepseek_v41 import sm80_moe
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma

    for name in (SWITCH, *NEEDS):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(sm80_router_mma, "supports", lambda *args: None)
    monkeypatch.setattr(sm80_router_mma, "register_warmup", lambda router: None)
    ffn = _ffn(monkeypatch)
    ffn.experts.moe_config = SimpleNamespace(dp_size=1, pcp_size=1,
                                             is_sequence_parallel=False)
    router = ffn.experts.router
    router.global_num_experts, router.top_k, router.renormalize = 8, 3, True
    router.routed_scaling_factor, router.image_sentinel_lo = 1.5, 0
    sm80_moe.adopt_moe(ffn, SimpleNamespace())
    return ffn


def test_the_gate_weight_is_held_once_in_bf16_and_loads_exactly(monkeypatch):
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    ffn = _adopt_mma(monkeypatch)
    gate = ffn.gate
    assert gate.mma and ffn.experts.router.mma_gate is gate
    assert {n: p.dtype for n, p in gate.named_parameters()} == {"weight": torch.bfloat16}
    ck = _bf16_valued_fp16(8, 64, 0)
    gate.weight.weight_loader(gate.weight, ck)
    assert torch.equal(gate.weight.float(), ck.float())
    # The FP32-held gate's values, and the bytes of the BF16 copy the gate
    # made of that FP32 weight before.
    ref = _ffn(monkeypatch).gate
    sm80_moe._fp32_gate_weight(ref)
    ref.weight.weight_loader(ref.weight, ck)
    assert torch.equal(gate.weight.float().view(torch.int32), ref.weight.view(torch.int32))
    assert torch.equal(gate.weight.view(torch.int16),
                       ref.weight.to(torch.bfloat16).view(torch.int16))
    # A BF16 checkpoint tensor (the hybrid checkpoint's) loads as it is.
    gate.weight.weight_loader(gate.weight, (-ck).to(torch.bfloat16))
    assert torch.equal(gate.weight.float(), -ck.float())
    # A value BF16 does not hold is refused before the copy rounds it.
    before = gate.weight.detach().clone()
    bad = ck.clone()
    bad[3, 5] = 1.0 + 2.0**-10
    with pytest.raises(ValueError, match="not BF16-valued"):
        gate.weight.weight_loader(gate.weight, bad)
    assert torch.equal(gate.weight, before)


def test_the_fp32_fallback_returns_the_fp32_held_gates_bytes(monkeypatch):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    construct.set_groups(monkeypatch, 1, 0)
    ck = _bf16_valued_fp16(384, HIDDEN, 1)
    gates = {}
    for dtype in (torch.float32, torch.bfloat16):
        gate = GateLinear(HIDDEN, 384, bias=False, out_dtype=torch.float32,
                          prefix="layers.0.ffn.gate")
        sm80_moe._hold_gate_weight(gate, dtype)
        gate.weight.data.copy_(ck)
        gates[dtype] = gate
    held = gates[torch.bfloat16]
    assert held.weight.dtype == torch.bfloat16
    # Both kernels on: on CPU neither serves, so every call is the fallback.
    held.mma = held.gemv = True
    g = torch.Generator().manual_seed(2)
    for rows in (1, 6, 2048, 2049, 3000):
        x = (torch.randn(rows, HIDDEN, generator=g) * 3.0).to(torch.bfloat16)
        want = gates[torch.float32](x)[0]
        got, bias = held(x)
        assert bias is None and got.dtype == torch.float32
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), rows
    x = torch.randn(5, HIDDEN, generator=g)
    assert torch.equal(held(x)[0].view(torch.int32),
                       gates[torch.float32](x)[0].view(torch.int32))
    # Upstream's fallback on the BF16 weight would multiply in BF16.
    x = (torch.randn(2048, HIDDEN, generator=g) * 3.0).to(torch.bfloat16)
    assert not torch.equal(GateLinear.forward(held, x)[0], gates[torch.float32](x)[0])


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory):
    """The tiny EXL3 checkpoint as written (random FP16 gates) and with its
    gate weights rounded to BF16 values (those FP16 does not hold: zero)."""
    raw = tmp_path_factory.mktemp("tiny-raw")
    fixture.write_checkpoint(raw, fixture.tiny_config())
    held = tmp_path_factory.mktemp("tiny-bf16")
    fixture.write_checkpoint(held, fixture.tiny_config())
    rounded = 0
    for shard in sorted(held.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for name, t in tensors.items():
            if name.endswith(".ffn.gate.weight"):
                assert t.dtype == torch.float16
                w16 = t.to(torch.bfloat16)
                h = w16.half()
                h[h.float() != w16.float()] = 0.0
                assert torch.equal(h.to(torch.bfloat16).float(), h.float())
                tensors[name] = h
                rounded += 1
        save_file(tensors, str(shard))
    assert rounded == 11  # 8 target layers, 3 drafter layers
    return raw, held


def _gate_weights(model) -> dict[str, torch.Tensor]:
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    return {n: m.weight.detach().clone() for n, m in model.named_modules()
            if isinstance(m, sm80_moe.Sm80DeepseekV41Gate)}


def test_the_tiny_checkpoint_loads_bf16_gates_with_the_fp32_gates_bytes(
    monkeypatch,
    meta_platform,  # noqa: F811
    tiny_checkpoints,
):
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma
    from vllm_sm80 import plugin

    plugin._register_exl3()
    raw, held = tiny_checkpoints
    # The tiny gates (16 x 512, 8 x 512) are not a tensor-core shape.
    monkeypatch.setattr(sm80_router_mma, "supports", lambda *args: None)
    monkeypatch.setattr(sm80_router_mma, "register_warmup", lambda router: None)
    construct.set_groups(monkeypatch, 1, 0)
    construct.stub_engram_tables(monkeypatch, 1)
    weights = {}
    for switch in ("0", "1"):
        for name in (SWITCH, *NEEDS):
            monkeypatch.setenv(name, switch)
        configs = dict(zip(("target", "draft"), construct.vllm_configs(str(held), 1)))
        for role, config in configs.items():
            model = construct.build_model(config)
            _load(monkeypatch, model, config)
            weights[switch, role] = _gate_weights(model)
    for role, layers in (("target", 8), ("draft", 3)):
        off, on = weights["0", role], weights["1", role]
        assert len(off) == layers and set(on) == set(off)
        for name in off:
            assert off[name].dtype == torch.float32 and on[name].dtype == torch.bfloat16
            assert torch.equal(on[name].float().view(torch.int32),
                               off[name].view(torch.int32)), name
        assert sum(w.nbytes for w in on.values()) * 2 == sum(w.nbytes for w in off.values())
    # The checkpoint as written: random FP16 gates, refused with the switch.
    for name in (SWITCH, *NEEDS):
        monkeypatch.setenv(name, "1")
    config, _ = construct.vllm_configs(str(raw), 1)
    model = construct.build_model(config)
    with pytest.raises(ValueError, match="not BF16-valued"):
        _load(monkeypatch, model, config)


def test_held_and_freed_bytes_of_the_served_gates(monkeypatch):
    from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma as rm

    def router(experts):
        weight = torch.empty(experts, rm.HIDDEN, dtype=torch.bfloat16, device="meta")
        return SimpleNamespace(mma_gate=SimpleNamespace(weight=weight))

    routers = [router(384) for _ in range(40)] + [router(128) for _ in range(3)]
    # Two routers of one gate count it once.
    routers.append(SimpleNamespace(mma_gate=routers[0].mma_gate))
    assert rm.held_gate_bytes(routers) == (43, 161_218_560, 322_437_120)
    lines = []
    monkeypatch.setattr(rm, "logger", SimpleNamespace(
        info_once=lambda msg, *args, **kwargs: lines.append(msg % args)))
    monkeypatch.setattr(rm, "_ADOPTED", routers)
    monkeypatch.setattr(rm, "_sm80", lambda device: False)
    assert rm.RouterWarmup().get_warmup_keys() == []
    assert lines == [
        "DeepSeek-V4.1 sm_80: 43 MoE gate weights held once, in BF16 (161218560 bytes); "
        "322437120 bytes of FP32 gate weights freed on this rank "
        "(VLLM_DEEPSEEK_V41_ROUTER_MMA=1)"
    ]
