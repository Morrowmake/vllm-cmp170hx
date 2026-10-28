# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static K-table, decoder replay and construction-bound arm contracts."""
from types import SimpleNamespace as NS
from weakref import WeakSet

import pytest

from vllm.config import CUDAGraphMode as Mode
from vllm_sm80 import dev_ab as ab
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor as Desc
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from tests.sm80.test_dev_ab import fake_switch, state  # noqa: F401


def manager(descs, cls=CudaGraphManager):
    m = object.__new__(cls)
    m.graphs = {}
    m._capture_descs = {}
    m._candidates = {}
    for d in descs:
        m._capture_descs.setdefault(d.cg_mode, []).append(d)
        m._candidates.setdefault((d.num_tokens, 0), []).append(d)
    m.decode_query_len = descs[0].uniform_token_count or 1
    m._max_full_descs_to_capture = None
    m._capture_mem_samples = None
    m._graphs_captured = False
    m.cudagraph_mode = Mode.FULL_DECODE_ONLY
    m.use_breakable_cg = False
    m.breakable_cg_runner = None
    return m


def dynamic_runner(monkeypatch, boot=False, k_busy=3):
    from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import (
        DecoderReplayCudaGraphManager,
    )
    target = manager([Desc(Mode.FULL, b * (k + 1), b, k + 1)
                      for b in range(1, 9) for k in {5, k_busy}])
    draft = manager([Desc(Mode.FULL, b * 5, b, 5) for b in range(1, 9)])
    replay = manager([Desc(Mode.PIECEWISE, n, None) for n in (128, 256, 1024)],
                     DecoderReplayCudaGraphManager)
    replay.use_breakable_cg = True
    replay.breakable_cg_runner = NS(entries={})
    stores = [(m, "graphs") for m in (target, draft, replay)]
    stores.append((replay.breakable_cg_runner, "entries"))
    monkeypatch.setattr(ab, "_graph_stores", lambda: stores)
    table = [[1, 3, 5], [4, 8, k_busy]]
    spec = NS(method="dspark", num_speculative_tokens=5,
              num_speculative_tokens_per_batch_size=table,
              uses_dynamic_speculative_decoding=lambda: True)
    drafter = NS(_cg_managers={5: draft}, widths=(5,), max_width=5, width=5,
                 query_cudagraph_manager=draft)
    def set_width(width):
        drafter.width = width
        drafter.query_cudagraph_manager = drafter._cg_managers[width]
    drafter.set_width = set_width
    runner = NS(model_state=NS(supports_mm_inputs=False, replay_cudagraphs=replay),
                vllm_config=NS(speculative_config=spec, num_speculative_tokens=5,
                               compilation_config=NS(cudagraph_mode=Mode.FULL_DECODE_ONLY)),
                max_num_reqs=8, speculator=drafter, cudagraph_manager=target,
                req_states=NS(req_id_to_index={}), adaptive_verification=None)
    captured = []
    def capture(r):
        assert r is runner
        captured.append(ab._STATE.arm)
        for m in (target, draft):
            for d in m._capture_descs[Mode.FULL]:
                m.graphs[d] = (ab._STATE.arm, d)
            m._graphs_captured = True
        for d in replay._capture_descs[Mode.PIECEWISE]:
            replay.breakable_cg_runner.entries[d] = (ab._STATE.arm, d)
        replay._graphs_captured = True
        return 70 if ab._STATE.arm else 30
    monkeypatch.setattr(ab, "_capture_base", capture)
    monkeypatch.setattr(ab, "_all_ranks_agree", lambda ok: ok)
    switch = ab.SWITCHES["fake"]
    monkeypatch.setitem(ab.SWITCHES, "fake", ab.Switch(
        switch.name, switch.env, True, True, switch.unavailable,
        lambda _: boot, switch.apply))
    return runner, (target, draft, replay), captured


@pytest.mark.parametrize("boot", [False, True])
@pytest.mark.parametrize("busy", [0, 3])
def test_dynamic_captures_every_shape_and_preserves_policy(monkeypatch, state, fake_switch, boot, busy):
    runner, managers, captured = dynamic_runner(monkeypatch, boot, busy)
    table = runner.vllm_config.speculative_config.num_speculative_tokens_per_batch_size
    plans = [(m._capture_descs, m._candidates) for m in managers]
    locks = []
    monkeypatch.setattr(ab, "_lock_workspace", lambda: locks.append(tuple(captured)))
    assert ab.capture_model_per_arm(runner, False) == 100
    assert captured == [boot, not boot] and locks == [(boot, not boot)]
    assert state.arm is boot and state.captured == (boot, not boot)
    assert runner.vllm_config.speculative_config.num_speculative_tokens_per_batch_size is table
    assert runner.speculator.query_cudagraph_manager is managers[1]
    assert all(m._capture_descs is p[0] and m._candidates is p[1]
               for m, p in zip(managers, plans))
    coverage = state.saved["dynamic_coverage"]
    assert [(r["batch"], r["k"]) for r in coverage] == [(b, 5 if b <= 3 else busy) for b in range(1, 9)]
    assert {r["draft_width"] for r in coverage} == ({5} if busy else {0, 5})
    stores = {s["role"]: s for s in state.saved["graph_inventory"] if s["role"]}
    assert set(stores) == {"target", "draft:5"}
    for row in coverage:
        for path in ("target", "draft") if row["k"] else ("target",):
            store = stores[row[path + "_store"]]
            assert row[path] in store["expected_full"]
            assert all(row[path] in store["arms"][arm] for arm in ("off", "on"))
    for m in managers:
        expected = set(m._capture_descs.get(Mode.FULL, ()))
        for arm in (False, True):
            assert set(m.graphs.by_arm.get(arm, ())) == expected
    # Equal token counts do not alias different batch/K descriptors.
    target = managers[0]
    if busy == 3:
        d1, d2 = Desc(Mode.FULL, 12, 2, 6), Desc(Mode.FULL, 12, 3, 4)
        assert d1 in target.graphs and d2 in target.graphs and d1 != d2
    replay_store = managers[2].breakable_cg_runner.entries
    assert len(replay_store.by_arm[False]) == len(replay_store.by_arm[True]) == 3
    worker = NS(rank=0, model_runner=runner)
    for gen, arm, replay in ((1, "off", False), (2, "on", True), (3, "off", True)):
        reply = ab._set_arm(worker, arm, gen, replay)
        assert reply["ok"] and reply["replay"] is replay
        assert all(m._graphs_captured is replay for m in managers)
    assert state.capture_bytes["serving"] == {"off": 30, "on": 70}


def test_missing_decoder_replay_size_fails_even_when_both_arms_match(monkeypatch, state, fake_switch):
    runner, managers, _ = dynamic_runner(monkeypatch)
    base = ab._capture_base
    def capture(r):
        result = base(r)
        store = managers[2].breakable_cg_runner.entries
        del store[next(d for d in store if d.num_tokens == 256)]
        return result
    monkeypatch.setattr(ab, "_capture_base", capture)
    locks = []
    monkeypatch.setattr(ab, "_lock_workspace", lambda: locks.append(True))
    with pytest.raises(RuntimeError, match="replay sizes"):
        ab.capture_model_per_arm(runner, False)
    assert not locks and state.arm is False and not state.captured


def test_failure_restores_boot_arm_and_descriptor_objects(monkeypatch, state, fake_switch):
    runner, managers, _ = dynamic_runner(monkeypatch)
    original = managers[0]._capture_descs
    base = ab._capture_base
    def capture(r):
        result = base(r)
        if state.arm:
            managers[0]._capture_descs = {}
        return result
    monkeypatch.setattr(ab, "_capture_base", capture)
    with pytest.raises(RuntimeError, match="re-specialized"):
        ab.capture_model_per_arm(runner, False)
    assert managers[0]._capture_descs is original
    assert state.arm is False and state.in_capture is False and not state.captured


def test_set_refuses_respecialized_table_atomically(monkeypatch, state, fake_switch):
    runner, _, _ = dynamic_runner(monkeypatch)
    ab.capture_model_per_arm(runner, False)
    runner.vllm_config.speculative_config.num_speculative_tokens_per_batch_size[1][2] = 0
    before = list(fake_switch)
    reply = ab._set_arm(NS(rank=0, model_runner=runner), "on", 1)
    assert not reply["ok"] and "re-specialized" in reply["reason"]
    assert fake_switch == before and state.generation == 0


@pytest.mark.parametrize("attribute", ["adaptive_verification", "adaptive_k_config", "method"])
def test_stateful_or_other_dynamic_policies_remain_refused(monkeypatch, state, fake_switch, attribute):
    runner, _, captured = dynamic_runner(monkeypatch)
    if attribute == "adaptive_verification":
        runner.adaptive_verification = object()
    else:
        setattr(runner.vllm_config.speculative_config, attribute,
                "dflash" if attribute == "method" else object())
    with pytest.raises(RuntimeError):
        ab.capture_model_per_arm(runner, False)
    assert not captured


def test_discovery_includes_decoder_manager_and_its_actual_store(monkeypatch, state):
    from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import DecoderReplayCudaGraphManager
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    wrapper = object.__new__(BreakableCUDAGraphWrapper)
    wrapper.entries = {}
    replay = object.__new__(DecoderReplayCudaGraphManager)
    replay.graphs = {}
    replay.breakable_cg_runner = wrapper
    monkeypatch.setattr(BreakableCUDAGraphWrapper, "_all_instances", WeakSet([wrapper]))
    stores = ab._graph_stores()
    assert (replay, "graphs") in stores and (wrapper, "entries") in stores
    assert stores.count((replay, "graphs")) == stores.count((wrapper, "entries")) == 1


@pytest.mark.parametrize("boot", [False, True])
def test_attention_adapter_visits_target_and_drafter_and_restores(monkeypatch, state, boot):
    class Layer:
        def __init__(self, eligible=True):
            self._sm80_fused_attn = boot and eligible
            self.eligible = eligible
        def _sm80_fused_attn_gate(self, config):
            self._sm80_fused_max_rows = 48
            self._sm80_max_decode_query_len = 6
            self._sm80_is_draft_layer = False
            return self.eligible
    a, b, unsupported = Layer(), Layer(), Layer(False)
    runner = NS(model=NS(modules=lambda: [a, unsupported]),
                speculator=NS(model=NS(modules=lambda: [a, b])), vllm_config=object())
    monkeypatch.setenv(ab._ATTN_ENV, str(int(boot)))
    switch = ab.SWITCHES["dsv41_fused_attn_decode"]
    assert switch.unavailable(runner) is None
    assert len(state.saved["dsv41_attention"]) == 3
    switch.apply(runner, True)
    assert a._sm80_fused_attn and b._sm80_fused_attn and not unsupported._sm80_fused_attn
    switch.apply(runner, False)
    assert not a._sm80_fused_attn and not b._sm80_fused_attn
    assert not hasattr(a, "_sm80_fused_max_rows")
    switch.apply(runner, True)
    assert a._sm80_fused_max_rows == 48


def test_moe_prefill_requires_load_time_workspace_and_keeps_closed_gates(state):
    eligible, closed = NS(prefill_v2_bits_x2=6), NS(prefill_v2_bits_x2=0)
    def layer(experts):
        return NS(quant_method=NS(moe_kernel=NS(fused_experts=experts)))
    runner = NS(model=NS(modules=lambda: [layer(eligible)]),
                speculator=NS(model=NS(modules=lambda: [layer(closed)])))
    switch = ab.SWITCHES["dsv41_moe_prefill_tc"]
    assert switch.unavailable(runner) is None
    switch.apply(runner, False)
    assert eligible.prefill_v2_bits_x2 == closed.prefill_v2_bits_x2 == 0
    switch.apply(runner, True)
    assert eligible.prefill_v2_bits_x2 == 6 and closed.prefill_v2_bits_x2 == 0
    eligible.prefill_v2_bits_x2 = 0
    assert "boot with" in switch.unavailable(runner)
    assert all(s.env != "VLLM_EXL3_MOE_CHUNK_ROWS" for s in ab.SWITCHES.values())


def test_splitk_adapter_changes_latched_dispatch(monkeypatch, state):
    from vllm_sm80.kernels.exl3 import ops
    monkeypatch.setattr(ops, "_dense_splitk", False)
    ab._splitk_apply(None, True)
    assert ops._dense_splitk
    ab._splitk_apply(None, False)
    assert not ops._dense_splitk

def test_base_capture_does_not_nest_subclass_memory_window(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    calls = []
    monkeypatch.setattr(GPUModelRunner, "capture_model",
                        lambda self, **kw: calls.append((self, kw)) or 17)
    runner = NS(capture_model=lambda **kw: pytest.fail("nested profiler"))
    assert ab._capture_base(runner) == 17
    assert calls == [(runner, {"profile_only": True})]


def test_profiling_does_not_qualify_serving_graphs(monkeypatch, state, fake_switch):
    runner, _, _ = dynamic_runner(monkeypatch)
    assert ab.capture_model_per_arm(runner, True) == 100
    assert state.captured == () and "capture_contract" not in state.saved
    assert state.capture_bytes["profiling"] == {"off": 30, "on": 70}


def test_missing_target_shape_is_refused_before_capture(monkeypatch, state, fake_switch):
    runner, managers, captured = dynamic_runner(monkeypatch)
    managers[0]._candidates.pop((32, 0))
    with pytest.raises(RuntimeError, match="missing batch=8"):
        ab.capture_model_per_arm(runner, False)
    assert not captured


@pytest.mark.parametrize("boot", [False, True])
def test_pipeline_adapter_preserves_construction_eligibility(state, boot):
    def layer(eligible):
        return NS(_sm80_attn_pipeline_eligible=eligible,
                  _sm80_attn_pipeline_enabled=boot and eligible)
    target, draft, closed = layer(True), layer(True), layer(False)
    runner = NS(model=NS(modules=lambda: [target, closed]),
                speculator=NS(model=NS(modules=lambda: [target, draft])))
    switch = ab.SWITCHES["dsv41_attn_pipeline"]
    assert switch.unavailable(runner) is None and switch.current(runner) is boot
    assert switch.graph_bound and not switch.output_identical
    for arm in (not boot, boot):
        switch.apply(runner, arm)
        assert target._sm80_attn_pipeline_enabled is arm
        assert draft._sm80_attn_pipeline_enabled is arm
        assert not closed._sm80_attn_pipeline_enabled
        assert target._sm80_attn_pipeline_eligible and draft._sm80_attn_pipeline_eligible
        assert not closed._sm80_attn_pipeline_eligible
    absent = NS(model=NS(modules=lambda: [object()]), speculator=None)
    assert "no loaded" in switch.unavailable(absent)


@pytest.mark.parametrize("which", [0, 1])
def test_full_decode_rejects_eager_target_or_drafter(monkeypatch, state, fake_switch, which):
    runner, managers, captured = dynamic_runner(monkeypatch)
    managers[which].cudagraph_mode = Mode.NONE
    # A plausible eager manager has no FULL capture plan or candidates.
    managers[which]._capture_descs[Mode.FULL] = []
    managers[which]._candidates.clear()
    with pytest.raises(RuntimeError, match="FULL"):
        ab.capture_model_per_arm(runner, False)
    assert not captured and not state.captured


def test_changed_manager_mode_cannot_silently_enable_eager(monkeypatch, state, fake_switch):
    runner, managers, _ = dynamic_runner(monkeypatch)
    ab.capture_model_per_arm(runner, False)
    managers[1].cudagraph_mode = Mode.NONE
    reply = ab._set_arm(NS(rank=0, model_runner=runner), "on", 1)
    assert not reply["ok"] and "re-specialized" in reply["reason"]
    assert state.arm is False and state.generation == 0


@pytest.mark.parametrize("boot", [False, True])
def test_mhc_adapter_changes_loaded_target_and_draft_without_losing_capture(monkeypatch, state, boot):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused
    from vllm_sm80.models.deepseek_v41.sm80_model import _CapturedMhc

    initial = sm80_mhc_fused if boot else sm80_mhc
    target = NS(_mhc_ops=initial, hc_mult=4, hidden_size=5120)
    draft = NS(hc_mult=4, hidden_size=5120, attn=NS(prefix="draft.layers.0.attn"))
    wrapper = _CapturedMhc(initial, draft)
    draft._mhc_ops = wrapper
    runner = NS(model=NS(modules=lambda: [target]),
                speculator=NS(model=NS(modules=lambda: [target, draft])))
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_MHC", str(int(boot)))
    switch = ab.SWITCHES["dsv41_fused_mhc"]
    assert switch.unavailable(runner) is None
    assert len(state.saved["dsv41_mhc"]) == 2
    assert switch.current(runner) is boot
    for on in (not boot, boot):
        switch.apply(runner, on)
        expected = sm80_mhc_fused if on else sm80_mhc
        assert target._mhc_ops is expected and wrapper._ops is expected
        assert draft._mhc_ops is wrapper and wrapper._layer is draft
        assert switch.current(runner) is on
    target.hc_mult = 3
    assert "hc_mult" in switch.unavailable(runner)
    target.hc_mult = 4
    target._mhc_ops = object()
    assert "unsupported" in switch.unavailable(runner)
    assert "no sm_80" in switch.unavailable(NS())


def test_mhc_finish_requires_fused_parent_and_changes_latched_dispatch(monkeypatch, state):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused

    layer = NS(_mhc_ops=sm80_mhc, hc_mult=4, hidden_size=5120)
    runner = NS(model=NS(modules=lambda: [layer]))
    switch = ab.SWITCHES["dsv41_fused_mhc_finish"]
    assert "boot with" in switch.unavailable(runner)
    layer._mhc_ops = sm80_mhc_fused
    monkeypatch.setattr(sm80_mhc_fused, "FINISH_KERNEL", False)
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH", "0")
    assert switch.unavailable(runner) is None
    for on in (True, False):
        switch.apply(runner, on)
        assert switch.current(runner) is on
        assert layer._mhc_ops is sm80_mhc_fused


def test_mhc_finish_v2_requires_the_finish_kernel_and_warms_its_arm(monkeypatch, state):
    import torch

    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_finish_v2 as finish_v2

    warmed = []
    monkeypatch.setattr(finish_v2.FINISH_V2_WARMUP, "warmup",
                        lambda **kw: warmed.append(kw["vllm_config"]))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    layer = NS(_mhc_ops=sm80_mhc, hc_mult=4, hidden_size=5120, hc_sinkhorn_iters=20)
    config = object()
    runner = NS(model=NS(modules=lambda: [layer]), vllm_config=config)
    switch = ab.SWITCHES["dsv41_fused_mhc_finish_v2"]
    assert switch.graph_bound and not switch.output_identical
    assert switch.env == "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2"
    assert "boot with" in switch.unavailable(runner)
    layer._mhc_ops = sm80_mhc_fused
    monkeypatch.setattr(sm80_mhc_fused, "FINISH_KERNEL", False)
    assert "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1" in switch.unavailable(runner)
    monkeypatch.setattr(sm80_mhc_fused, "FINISH_KERNEL", True)
    monkeypatch.setattr(sm80_mhc_fused, "FINISH_V2", False)
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2", "0")
    assert switch.unavailable(runner) is None and switch.current(runner) is False
    switch.apply(runner, True)
    assert switch.current(runner) is True and warmed == [config]
    assert ab.os.environ["VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2"] == "1"
    switch.apply(runner, False)
    assert switch.current(runner) is False and warmed == [config]
    assert layer._mhc_ops is sm80_mhc_fused
    layer.hidden_size = 4096
    assert "hidden size 4096" in switch.unavailable(runner)


def test_dense_prefill_requires_boot_warmup_and_preserves_route_configuration(monkeypatch, state):
    from vllm_sm80.kernels.exl3 import dense_prefill_tc as dense, ops

    switch = ab.SWITCHES["exl3_dense_prefill_tc"]
    monkeypatch.setattr(ops, "_loaded", True)
    monkeypatch.setattr(dense, "active", False)
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "0")
    assert "boot with" in switch.unavailable(None)
    dense.active = True
    assert switch.unavailable(None) is None
    fixed = dense._max_decode_query_len, dense._sms, dense._fake_registered
    for on in (False, True):
        switch.apply(None, on)
        assert switch.current(None) is on
        assert (dense._max_decode_query_len, dense._sms, dense._fake_registered) == fixed


@pytest.mark.parametrize("name", [
    "VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC",
    "VLLM_DEEPSEEK_V41_ATTN_PIPELINE", "VLLM_EXL3_HYBRID", "VLLM_DSPARK_PER_K_WIDTH",
    "VLLM_SM80_ALLREDUCE_ROUND_ONCE", "VLLM_SM80_ALLREDUCE",
    "VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE",
    "VLLM_SM80_PREFILL_REDUCE_OVERLAP", "VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER",
    "VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL",
    "VLLM_SM80_SHARD_ARGMAX",
])
def test_combined_features_keep_default_off(monkeypatch, name):
    from vllm_sm80.envs import FORK_ENV_VARS

    monkeypatch.delenv(name, raising=False)
    assert FORK_ENV_VARS[name]() is False


@pytest.mark.parametrize("name", ["VLLM_DEV_AB_SWITCH", "VLLM_SM80_CAPTURE",
                                  "VLLM_SM80_REPLAY_DIR"])
def test_combined_optional_paths_keep_empty_default(monkeypatch, name):
    from vllm_sm80.envs import FORK_ENV_VARS

    monkeypatch.delenv(name, raising=False)
    assert FORK_ENV_VARS[name]() == ""


def test_moe_v2_requires_construction_and_restores_only_eligible_rows(monkeypatch, state):
    eligible = NS(prefill_v2_bits_x2=6, moe_v2_rows=64, moe_v2_bits_x2=6, workspace_elements=1234)
    closed = NS(prefill_v2_bits_x2=0, moe_v2_rows=0, moe_v2_bits_x2=0, workspace_elements=5678)
    def layer(experts):
        return NS(quant_method=NS(moe_kernel=NS(fused_experts=experts)))
    runner = NS(model=NS(modules=lambda: [layer(eligible), layer(closed)]))
    monkeypatch.setenv("VLLM_EXL3_MOE_V2", "1")
    switch = ab.SWITCHES["exl3_moe_v2"]
    assert switch.unavailable(runner) is None
    for on in (False, True):
        switch.apply(runner, on)
        assert switch.current(runner) is on
        assert eligible.moe_v2_rows == (64 if on else 0)
        assert closed.moe_v2_rows == 0
        assert eligible.moe_v2_bits_x2 == 6 and closed.moe_v2_bits_x2 == 0
        assert eligible.workspace_elements == 1234 and closed.workspace_elements == 5678
    eligible.moe_v2_rows = 0
    assert "boot with" in switch.unavailable(runner)


def test_moe_v2_draft_switch_toggles_only_the_drafter_experts(monkeypatch, state):
    target = NS(prefill_v2_bits_x2=6, moe_v2_rows=64, moe_v2_bits_x2=6, moe_v2_draft=False)
    drafter = NS(prefill_v2_bits_x2=0, moe_v2_rows=40, moe_v2_bits_x2=8, moe_v2_draft=True)
    def layer(experts):
        return NS(quant_method=NS(moe_kernel=NS(fused_experts=experts)))
    runner = NS(model=NS(modules=lambda: [layer(target), layer(drafter)]))
    monkeypatch.setenv("VLLM_EXL3_MOE_V2", "1")
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_DRAFT", "1")
    draft = ab.SWITCHES["exl3_moe_v2_draft"]
    v2 = ab.SWITCHES["exl3_moe_v2"]
    assert draft.graph_bound and not draft.output_identical
    assert draft.unavailable(runner) is None and v2.unavailable(runner) is None
    for on in (False, True):
        draft.apply(runner, on)
        assert draft.current(runner) is on
        assert drafter.moe_v2_rows == (40 if on else 0) and target.moe_v2_rows == 64
    for on in (False, True):
        v2.apply(runner, on)
        assert v2.current(runner) is on
        assert target.moe_v2_rows == (64 if on else 0) and drafter.moe_v2_rows == 40
    # The fold switch never folds the drafter's calls.
    assert ab._moe_v2_fold_experts(runner) == [target]
    drafter.moe_v2_rows = 0
    assert "boot with VLLM_EXL3_MOE_V2_DRAFT=1" in draft.unavailable(runner)
    target.moe_v2_rows = 0
    drafter.moe_v2_rows = 40
    assert "boot with VLLM_EXL3_MOE_V2=1" in v2.unavailable(runner)
