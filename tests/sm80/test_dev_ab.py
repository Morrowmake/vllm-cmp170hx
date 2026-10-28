# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The development A/B switch setter (vllm_sm80/dev_ab.py)."""
import ast
import hashlib
import pathlib
import textwrap
from types import SimpleNamespace

import pytest

from vllm_sm80 import dev_ab

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_RUNNER = REPO / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
# sha256 of GPUModelRunner.capture_model (decorator to end, dedented) in the
# upstream-derived tree the hook was added to; the hook is the only change.
UPSTREAM_CAPTURE_MODEL_SHA256 = (
    "2dcdeba2e33d7885e14aed9cbbd2a6eda0b814a9ffc661c9f4e096bc0b41ae47"
)
HOOK = "@with_ab_capture\n"


class _Trap:
    """Fails on any attribute access: proves a code path leaves it alone."""

    def __getattr__(self, name):
        raise AssertionError(f"touched .{name}")


@pytest.fixture
def state(monkeypatch):
    fresh = dev_ab._ArmState()
    monkeypatch.setattr(dev_ab, "_STATE", fresh)
    monkeypatch.setattr(dev_ab, "_synchronize", lambda: None)
    monkeypatch.setattr(dev_ab, "_capture_base", lambda r: r.capture_model(profile_only=True))
    monkeypatch.setattr(dev_ab, "_lock_workspace", lambda: None)
    monkeypatch.setattr(dev_ab, "_all_ranks_same_state", lambda: True)
    return fresh


@pytest.fixture
def fake_switch(monkeypatch):
    applied: list[bool] = []
    switch = dev_ab.Switch(
        "fake", "VLLM_FAKE_SWITCH", True, True,
        unavailable=lambda runner: None,
        current=lambda runner: False,
        apply=lambda runner, on: applied.append(on),
    )
    monkeypatch.setitem(dev_ab.SWITCHES, "fake", switch)
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "fake")
    return applied


class _Owner:
    def __init__(self):
        self.entries = {}


class _Runner:
    """Capture through the same wrapper used by Model Runner V2."""

    def __init__(self, owners, late_owner=None):
        self.owners = owners
        self.late_owner = late_owner
        self.calls = []
        self.model_state = SimpleNamespace(supports_mm_inputs=False)
        self.vllm_config = SimpleNamespace(speculative_config=None)
        self.adaptive_verification = None
        self.req_states = SimpleNamespace(req_id_to_index={})

    @dev_ab.with_ab_capture
    def capture_model(self, *, profile_only=False):
        arm = dev_ab._STATE.arm
        self.calls.append((arm, profile_only))
        if arm and self.late_owner is not None:
            self.owners.append(self.late_owner)
        for owner in self.owners:
            owner.entries["desc"] = f"graph-{arm}"
        return 100 if arm else 10


def _method_source(path, cls, name):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    start = min(
                        [d.lineno for d in item.decorator_list] + [item.lineno]
                    )
                    return textwrap.dedent("".join(lines[start - 1:item.end_lineno]))
    raise AssertionError(f"{cls}.{name} not found")


# --- switch unset: the off path --------------------------------------------


@pytest.mark.parametrize("value", [None, ""])
def test_unset_touches_nothing(monkeypatch, state, value):
    if value is None:
        monkeypatch.delenv("VLLM_DEV_AB_SWITCH", raising=False)
    else:
        monkeypatch.setenv("VLLM_DEV_AB_SWITCH", value)
    assert dev_ab.capture_model_per_arm(_Trap(), False) is None
    assert dev_ab.capture_model_per_arm(_Trap(), True) is None
    dev_ab.configure(_Trap())
    assert state == dev_ab._ArmState()


def test_capture_model_is_upstream_plus_the_hook():
    source = _method_source(MODEL_RUNNER, "GPUModelRunner", "capture_model")
    assert source.count(HOOK) == 1
    # The wrapper owns switching; the original capture body is byte-identical.
    assert source.split("\n")[1] == "@with_ab_capture"
    digest = hashlib.sha256(source.replace(HOOK, "").encode()).hexdigest()
    assert digest == UPSTREAM_CAPTURE_MODEL_SHA256


# --- configuration -----------------------------------------------------------


def test_configure_injects_the_extension(monkeypatch, state):
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "exl3_dense_splitk")
    config = SimpleNamespace(parallel_config=SimpleNamespace(worker_extension_cls=""))
    dev_ab.configure(config)
    assert config.parallel_config.worker_extension_cls == dev_ab.EXTENSION_QUALNAME
    dev_ab.configure(config)
    assert config.parallel_config.worker_extension_cls == dev_ab.EXTENSION_QUALNAME


def test_configure_refuses_unlisted_switch(monkeypatch, state):
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "VLLM_ANY_ENV")
    config = SimpleNamespace(parallel_config=SimpleNamespace(worker_extension_cls=""))
    with pytest.raises(ValueError, match="is not one of"):
        dev_ab.configure(config)


def test_configure_refuses_another_extension(monkeypatch, state):
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "exl3_dense_splitk")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_extension_cls="pkg.Other")
    )
    with pytest.raises(ValueError, match="already set"):
        dev_ab.configure(config)


@pytest.mark.parametrize("dev_mode", ["0", "1"])
def test_configure_refuses_diagnostic_capture(monkeypatch, state, dev_mode):
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "exl3_dense_splitk")
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", dev_mode)
    monkeypatch.setenv("VLLM_SM80_CAPTURE", "capture.toml")
    config = SimpleNamespace(parallel_config=SimpleNamespace(worker_extension_cls=""))
    with pytest.raises(ValueError, match="cannot be combined with VLLM_SM80_CAPTURE$"):
        dev_ab.configure(config)
    assert config.parallel_config.worker_extension_cls == ""
    assert state == dev_ab._ArmState()


def test_diagnostic_capture_without_ab_leaves_config_untouched(monkeypatch, state):
    monkeypatch.delenv("VLLM_DEV_AB_SWITCH", raising=False)
    monkeypatch.setenv("VLLM_SM80_CAPTURE", "capture.toml")
    dev_ab.configure(_Trap())
    assert state == dev_ab._ArmState()


def test_allowlist():
    assert set(dev_ab.SWITCHES) == {
        "exl3_dense_splitk",
        "exl3_moe_v2",
        "exl3_moe_v2_draft",
        "exl3_moe_v2_fold",
        "exl3_dense_prefill_tc",
        "dsv41_fused_mhc",
        "dsv41_fused_mhc_finish",
        "dsv41_fused_mhc_finish_v2",
        "dsv41_fused_route",
        "dsv41_draft_prologue_fused",
        "dsv41_gate_gemv",
        "dsv41_wo_a_grouped_gemm",
        "dsv41_engram_wkv_replicated_prefill",
        "dsv41_swa_meta_batch",
        "dsv41_fused_attn_decode",
        "dsv41_attn_pipeline",
        "dsv41_select_one_launch",
        "dsv41_moe_prefill_tc",
        "topk_canonical_ties",
        "marlin_decode_v2",
        "custom_allreduce_flags",
        "kda_prefill",
        "local_argmax_reduction",
        "target_shard_argmax",
        "qwen4_router_fp32",
    }


def test_extension_methods_are_free_on_the_worker():
    gpu_worker = pytest.importorskip("vllm.v1.worker.gpu_worker")
    names = [n for n in dir(dev_ab.DevABWorkerExtension) if not n.startswith("__")]
    assert names == [
        "begin_ab_batch", "end_ab_batch", "get_ab_arm", "get_hot_arms",
        "set_ab_arm", "set_hot_arms",
    ]
    for name in names:
        assert not hasattr(gpu_worker.Worker, name)


# --- graph stores ------------------------------------------------------------


def test_store_routes_by_arm(state):
    state.arm = False
    store = dev_ab.ArmGraphStore({False: {"d": 1}})
    assert store.get("d") == 1 and "d" in store and len(store) == 1
    state.arm = True
    assert store.get("d") is None and "d" not in store and len(store) == 0
    store["d"] = 2
    assert store["d"] == 2
    state.arm = False
    assert store["d"] == 1 and list(store) == ["d"]
    store.clear()
    assert store.by_arm == {False: {}, True: {}}


@pytest.mark.parametrize("profile_only", [False, True])
def test_capture_takes_both_arms_and_ends_on_boot(
    monkeypatch, state, fake_switch, profile_only
):
    owners = [_Owner(), _Owner()]
    monkeypatch.setattr(
        dev_ab, "_graph_stores", lambda: [(o, "entries") for o in owners]
    )
    runner = _Runner(owners)
    assert runner.capture_model(profile_only=profile_only) == 110
    # Boot arm (off) first; only the last capture may lock the workspace.
    assert runner.calls == [(False, True), (True, True)]
    assert fake_switch == [False, True, False]
    assert state.arm is False
    assert state.captured == (() if profile_only else (False, True))
    phase = "profiling" if profile_only else "serving"
    assert state.capture_bytes[phase] == {"off": 10, "on": 100}
    for owner in owners:
        assert isinstance(owner.entries, dev_ab.ArmGraphStore)
        assert owner.entries["desc"] == "graph-False"
        state.arm = True
        assert owner.entries["desc"] == "graph-True"
        state.arm = False


def test_capture_refuses_graphs_of_one_arm(monkeypatch, state, fake_switch):
    owners = [_Owner()]
    monkeypatch.setattr(
        dev_ab, "_graph_stores", lambda: [(o, "entries") for o in owners]
    )
    runner = _Runner(owners, late_owner=_Owner())
    with pytest.raises(RuntimeError, match="differ between arms"):
        runner.capture_model()
    assert state.in_capture is False


def test_capture_refuses_encoder_graphs(monkeypatch, state, fake_switch):
    runner = _Runner([])
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=True,
        encoder_runner=SimpleNamespace(has_cudagraph=lambda: True),
    )
    with pytest.raises(RuntimeError, match="encoder"):
        runner.capture_model()


def test_unavailable_switch_fails_at_boot(monkeypatch, state):
    switch = dev_ab.Switch(
        "fake", "VLLM_FAKE_SWITCH", True, True,
        unavailable=lambda runner: "not in this tree",
        current=lambda runner: False,
        apply=lambda runner, on: None,
    )
    monkeypatch.setitem(dev_ab.SWITCHES, "fake", switch)
    monkeypatch.setenv("VLLM_DEV_AB_SWITCH", "fake")
    with pytest.raises(RuntimeError, match="not in this tree"):
        _Runner([]).capture_model()


# --- the setter ----------------------------------------------------------------


@pytest.fixture
def worker(monkeypatch, state, fake_switch):
    owners = [_Owner()]
    monkeypatch.setattr(
        dev_ab, "_graph_stores", lambda: [(o, "entries") for o in owners]
    )
    monkeypatch.setattr(dev_ab, "_graphs_on", lambda runner: True)
    monkeypatch.setattr(dev_ab, "_all_ranks_agree", lambda ok: ok)
    runner = _Runner(owners)
    runner.capture_model()
    fake_switch.clear()
    return SimpleNamespace(rank=0, model_runner=runner)


def _set(worker, arm, generation):
    return dev_ab.DevABWorkerExtension.set_ab_arm(worker, arm, generation)


def test_set_switches_and_acknowledges(worker, fake_switch):
    reply = _set(worker, "on", 1)
    assert reply["ok"] and reply["arm"] == "on" and reply["generation"] == 1
    assert reply["boot_arm"] == "off" and reply["captured_arms"] == ["off", "on"]
    assert fake_switch == [True]
    assert worker.model_runner.owners[0].entries["desc"] == "graph-True"
    reply = _set(worker, "off", 2)
    assert reply["ok"] and reply["arm"] == "off"
    assert dev_ab.DevABWorkerExtension.get_ab_arm(worker)["generation"] == 2


@pytest.mark.parametrize(
    ("arm", "generation", "reason"),
    [
        ("maybe", 1, "is not one of"),
        ("on", 0, "not newer"),
        ("on", True, "not newer"),
        ("on", "1x", "not newer"),
    ],
)
def test_set_refuses_bad_arguments(worker, fake_switch, arm, generation, reason):
    reply = _set(worker, arm, generation)
    assert not reply["ok"] and reason in reply["reason"]
    assert reply["arm"] == "off" and fake_switch == []


def test_set_takes_a_decimal_string_generation(worker, fake_switch):
    reply = _set(worker, "on", "7")
    assert reply["ok"] and reply["generation"] == 7
    assert fake_switch == [True]


def test_set_refuses_stale_generation(worker):
    assert _set(worker, "on", 5)["ok"]
    reply = _set(worker, "off", 5)
    assert not reply["ok"] and reply["arm"] == "on"


def test_set_refuses_resident_requests(worker, fake_switch):
    worker.model_runner.req_states.req_id_to_index["r1"] = 0
    reply = _set(worker, "on", 1)
    assert not reply["ok"] and "drain first" in reply["reason"]
    assert fake_switch == []


def test_set_refuses_arm_without_graphs(worker, state, fake_switch):
    state.captured = (False,)
    reply = _set(worker, "on", 1)
    assert not reply["ok"] and "not captured" in reply["reason"]


def test_set_switches_no_rank_when_another_refuses(monkeypatch, worker, fake_switch):
    monkeypatch.setattr(dev_ab, "_all_ranks_agree", lambda ok: False)
    reply = _set(worker, "on", 1)
    assert not reply["ok"] and reply["reason"] == "another rank refused"
    assert fake_switch == [] and reply["arm"] == "off"


def test_profiling_extrapolates_each_arms_full_graphs(monkeypatch, state, fake_switch):
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils

    owners = [_Owner()]
    monkeypatch.setattr(
        dev_ab, "_graph_stores", lambda: [(o, "entries") for o in owners]
    )
    samples: list[int] = []
    runner = _Runner(owners)
    runner.cudagraph_manager = SimpleNamespace(
        _capture_mem_samples=samples,
        _capture_descs={CUDAGraphMode.FULL: list(range(10))},
    )
    inner = runner.capture_model

    def capture_model(*, profile_only=False):
        if dev_ab._STATE.in_capture:
            # Each arm samples its two largest FULL graphs.
            samples.extend([7, 3] if dev_ab._STATE.arm else [5, 2])
        return inner(profile_only=profile_only)

    runner.capture_model = capture_model
    total = runner.capture_model(profile_only=True)
    # Only the boot arm's samples stay for profile_cudagraph_memory to
    # extrapolate; the other arm's extrapolation is in its size.
    assert samples == [5, 2]
    extra = cudagraph_utils._extrapolate_full_graph_memory([7, 3], 10) - 10
    assert state.capture_bytes["profiling"] == {"off": 10, "on": 100 + extra}
    assert total == 110 + extra


def test_topk_switch_imports_the_qsa_module(monkeypatch):
    import types

    gate = types.ModuleType(dev_ab._TOPK_GATE)
    monkeypatch.delitem(dev_ab.sys.modules, dev_ab._QSA_MODULE, raising=False)
    monkeypatch.delitem(dev_ab.sys.modules, dev_ab._TOPK_GATE, raising=False)
    imported = []

    def fake_import(name):
        # The QSA ops module imports the shared gate.
        imported.append(name)
        dev_ab.sys.modules[name] = types.ModuleType(name)
        dev_ab.sys.modules[dev_ab._TOPK_GATE] = gate
        return dev_ab.sys.modules[name]

    monkeypatch.setattr(dev_ab.importlib, "import_module", fake_import)
    monkeypatch.setitem(
        dev_ab.sys.modules, dev_ab._QSA_MODEL_MODULE, types.ModuleType("m")
    )
    monkeypatch.setenv(dev_ab._TOPK_ENV, "1")
    switch = dev_ab.SWITCHES["topk_canonical_ties"]
    try:
        assert switch.unavailable(None) is None
        assert switch.current(None) is True
        assert imported == [dev_ab._QSA_MODULE]
    finally:
        dev_ab.sys.modules.pop(dev_ab._QSA_MODULE, None)
        dev_ab.sys.modules.pop(dev_ab._TOPK_GATE, None)


def test_topk_switch_refuses_a_model_without_an_indexer(monkeypatch):
    # Another model (no GLM or QSA indexer built): refuse rather than A/B two
    # equal arms.
    monkeypatch.delitem(dev_ab.sys.modules, dev_ab._QSA_MODEL_MODULE, raising=False)
    monkeypatch.delitem(dev_ab.sys.modules, dev_ab._TOPK_GATE, raising=False)
    imported = []
    monkeypatch.setattr(dev_ab.importlib, "import_module", imported.append)
    why = dev_ab.SWITCHES["topk_canonical_ties"].unavailable(None)
    assert "not loaded" in why and imported == []


def test_topk_switch_apply_rereads_the_gate(monkeypatch):
    """An arm sets the switch, drops the gate's cached reads and warms the
    kernels (use_tiefix_topk) before the arm's capture; the sort switch is
    left as booted."""
    import os
    import types

    calls = []

    class Cached:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            calls.append(self.name)

        def cache_clear(self):
            calls.append(f"clear {self.name}")

    gate = types.ModuleType(dev_ab._TOPK_GATE)
    for name in ("use_sorted_topk", "use_tiefix_topk", "tiefix_split_rows"):
        setattr(gate, name, Cached(name))
    monkeypatch.setitem(dev_ab.sys.modules, dev_ab._TOPK_GATE, gate)
    monkeypatch.setenv(dev_ab._TOPK_ENV, "0")
    monkeypatch.setenv("VLLM_SM80_TOPK_SORTED", "1")
    dev_ab.SWITCHES["topk_canonical_ties"].apply(None, True)
    assert os.environ[dev_ab._TOPK_ENV] == "1"
    assert os.environ["VLLM_SM80_TOPK_SORTED"] == "1"
    assert calls == [
        "clear use_sorted_topk",
        "clear use_tiefix_topk",
        "clear tiefix_split_rows",
        "use_tiefix_topk",
    ]


def test_swa_meta_batch_switch(monkeypatch):
    """Eager step preparation only, the same bytes: no per-arm graphs."""
    from vllm_sm80.models.deepseek_v41 import sm80_swa_meta_batch
    from vllm_sm80.models.deepseek_v41.sm80_attention import Sm80SWAMetadataBuilder

    switch = dev_ab.SWITCHES["dsv41_swa_meta_batch"]
    assert not switch.graph_bound and switch.output_identical
    assert switch.env == "VLLM_DEEPSEEK_V41_SWA_META_BATCH"
    group = SimpleNamespace(metadata_builders=[object()])
    assert "no sm_80" in switch.unavailable(SimpleNamespace(attn_groups=[[group]]))
    builder = object.__new__(Sm80SWAMetadataBuilder)
    runner = SimpleNamespace(
        attn_groups=[[group], [SimpleNamespace(metadata_builders=[builder])]]
    )
    assert switch.unavailable(runner) is None
    monkeypatch.setenv(switch.env, "0")
    resets = []
    monkeypatch.setattr(sm80_swa_meta_batch, "reset", lambda: resets.append(1))
    assert switch.current(runner) is False
    switch.apply(runner, True)
    assert switch.current(runner) is True
    switch.apply(runner, False)
    assert switch.current(runner) is False
    # Learned groups are forgotten at every switch.
    assert resets == [1, 1]


def test_local_argmax_switch_sets_the_speculator():
    module = pytest.importorskip("vllm.v1.worker.gpu.spec_decode.speculator")

    class Drafter:
        def get_top_tokens(self, hidden_states):
            return hidden_states

    class Speculator:
        _validate_local_argmax_reduction = (
            module.DraftModelSpeculator._validate_local_argmax_reduction
        )

        def __init__(self, model, method="greedy"):
            self.model = model
            self.speculative_config = SimpleNamespace(draft_sample_method=method)
            self.use_local_argmax_reduction = False
            self.draft_logits = None if method == "greedy" else object()
            self.acceptance_estimator = None

    switch = dev_ab.SWITCHES["local_argmax_reduction"]
    assert switch.graph_bound and switch.output_identical
    runner = SimpleNamespace(speculator=Speculator(Drafter()))
    assert switch.unavailable(runner) is None and switch.current(runner) is False
    switch.apply(runner, True)
    assert switch.current(runner) is True
    switch.apply(runner, False)
    assert switch.current(runner) is False
    # Servers where the on arm cannot run: refuse at boot.
    for speculator, reason in (
        (None, "no draft model speculator"),
        (Speculator(Drafter(), "probabilistic"), "probabilistic"),
        (Speculator(object()), "no get_top_tokens"),
    ):
        why = switch.unavailable(SimpleNamespace(speculator=speculator))
        assert reason in why
    adaptive = Speculator(Drafter())
    adaptive.acceptance_estimator = object()
    assert "adaptive" in switch.unavailable(SimpleNamespace(speculator=adaptive))
    # The speculator's own check still guards the on arm.
    with pytest.raises(ValueError, match="get_top_tokens"):
        switch.apply(SimpleNamespace(speculator=Speculator(object())), True)
def test_target_shard_argmax_switch(monkeypatch):
    """Eager sampling only, the same tokens: no per-arm graphs; the on arm
    compiles its kernels when it is applied."""
    torch = pytest.importorskip("torch")
    from vllm_sm80.worker import target_argmax

    switch = dev_ab.SWITCHES["target_shard_argmax"]
    assert not switch.graph_bound and switch.output_identical
    assert switch.env == "VLLM_SM80_SHARD_ARGMAX"
    lm = SimpleNamespace(
        lm_head=SimpleNamespace(tp_rank=0, tp_size=4),
        logits_processor=SimpleNamespace(org_vocab_size=129280, logits_as_input=False),
    )
    model = SimpleNamespace(compute_logits_local=lambda h: h, language_model=lm,
                            target_argmax_logits=True)

    def runner_with(**kw):
        fields = dict(model=model, lora_config=None, batch_sharder=None,
                      target_argmax_steps=True, device=torch.device("cpu"),
                      model_config=SimpleNamespace(get_hidden_size=lambda: 8,
                                                   dtype=torch.bfloat16))
        fields.update(kw)
        return SimpleNamespace(**fields)

    warmed = []
    monkeypatch.setattr(target_argmax, "argmax_rows",
                        lambda m, h: warmed.append((m, h.shape, h.dtype)))
    runner = runner_with()
    assert switch.unavailable(runner) is None
    monkeypatch.setenv(switch.env, "0")
    assert switch.current(runner) is False
    switch.apply(runner, True)
    assert switch.current(runner) is True and runner._target_shard_argmax is True
    assert warmed == [(model, (1, 8), torch.bfloat16)]
    switch.apply(runner, False)
    assert switch.current(runner) is False and len(warmed) == 1
    # Servers where the on arm cannot run: refuse at boot.
    assert switch.unavailable(runner_with(batch_sharder=object())) is None
    unsplit = SimpleNamespace(
        compute_logits_local=lambda h: h, target_argmax_logits=True,
        language_model=SimpleNamespace(lm_head=SimpleNamespace(tp_rank=0, tp_size=1),
                                       logits_processor=lm.logits_processor),
    )
    assert "TP 1" in switch.unavailable(runner_with(model=unsplit))
    upstream = runner_with(target_argmax_steps=False)
    assert "does not sample in greedy steps" in switch.unavailable(upstream)
    headless = runner_with(model=SimpleNamespace(target_argmax_logits=True))
    assert "compute_logits_local" in switch.unavailable(headless)
    with pytest.raises(ValueError, match="compute_logits_local"):
        switch.apply(headless, True)


@pytest.fixture
def router_model(monkeypatch):
    """Two target and one draft Qwen4Exp MoE block whose gates carry the
    GateLinear fields set_out_dtype reads and writes, and its method."""
    torch = pytest.importorskip("torch")
    gate_linear = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.router.gate_linear"
    )

    class Gate:
        set_out_dtype = gate_linear.GateLinear.set_out_dtype

        def __init__(self, prefix):
            self.prefix = prefix
            self.weight = torch.empty(4, 8, dtype=torch.bfloat16)
            self.out_dtype = None
            self._router_gemm_cublas_capable = True
            self.allow_cublas_router_gemm = False
            self._rocm_bf16x3_weight_eligible = False
            self.allow_rocm_bf16x3_router_gemm = False
            self.allow_specialized_router_gemm = False
            self.allow_ll_bf16_gemm = False
            self.allow_ll_fp32w_gemm = False

    class Block(torch.nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.gate = Gate(prefix)

    monkeypatch.setitem(
        dev_ab.sys.modules,
        dev_ab._QWEN4_MODEL_MODULE,
        SimpleNamespace(Qwen4ExpSparseMoeBlock=Block),
    )
    target = torch.nn.Sequential(Block("layers.0.mlp.gate"), Block("layers.1.mlp.gate"))
    draft = torch.nn.Sequential(Block("mtp.layers.0.mlp.gate"))
    return SimpleNamespace(
        torch=torch,
        Gate=Gate,
        runner=SimpleNamespace(model=target, speculator=SimpleNamespace(model=draft)),
        gates=[block.gate for block in (*target, *draft)],
    )


def _gate_fields(gate):
    return {attr: getattr(gate, attr) for attr in dev_ab._ROUTER_GATE_ATTRS}


def test_router_fp32_switch_sets_every_gate(state, router_model):
    torch, runner = router_model.torch, router_model.runner
    switch = dev_ab.SWITCHES["qwen4_router_fp32"]
    assert switch.graph_bound and not switch.output_identical
    assert switch.env == "hf_overrides.text_config.moe_router_dtype"
    assert switch.unavailable(runner) is None and switch.current(runner) is False
    # What the block's constructor does when the switch is set at boot.
    reference = router_model.Gate("reference")
    reference.set_out_dtype(torch.float32)
    assert reference.allow_cublas_router_gemm  # bf16 x bf16 -> fp32 GEMM tier
    boot = [_gate_fields(gate) for gate in router_model.gates]
    switch.apply(runner, False)  # the boot arm first, as the per-arm capture
    switch.apply(runner, True)
    assert switch.current(runner) is True
    for gate in router_model.gates:
        assert _gate_fields(gate) == _gate_fields(reference)
    switch.apply(runner, False)
    assert switch.current(runner) is False
    assert [_gate_fields(gate) for gate in router_model.gates] == boot
    # set_out_dtype refuses a second set, so the boot fields came back first.
    switch.apply(runner, True)
    assert switch.current(runner) is True


def test_router_fp32_switch_refuses_at_boot(monkeypatch, router_model):
    torch, runner = router_model.torch, router_model.runner
    switch = dev_ab.SWITCHES["qwen4_router_fp32"]
    other = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(2, 2)))
    assert "no Qwen4Exp MoE router gates" in switch.unavailable(other)
    # A boot with moe_router_dtype="float32" already set the FP32 logits.
    draft_gate = router_model.gates[-1]
    draft_gate.out_dtype = torch.float32
    assert "already has out_dtype" in switch.unavailable(runner)
    assert "moe_router_dtype" in switch.unavailable(runner)
    draft_gate.out_dtype = None
    draft_gate._rocm_bf16x3_weight_eligible = True
    assert "ROCm BF16x3" in switch.unavailable(runner)
    draft_gate._rocm_bf16x3_weight_eligible = False
    assert switch.unavailable(runner) is None
    monkeypatch.delitem(dev_ab.sys.modules, dev_ab._QWEN4_MODEL_MODULE)
    assert "no Qwen4Exp MoE router gates" in switch.unavailable(runner)

@pytest.mark.parametrize("profile_only", [False, True])
@pytest.mark.parametrize("selected", [None, 0, 23])
def test_capture_decorator_preserves_unset_and_zero_byte_results(monkeypatch, profile_only, selected):
    calls = []
    runner = object()
    def original(value, *, profile_only=False):
        assert value is runner
        calls.append(profile_only)
        return 17
    monkeypatch.setattr(dev_ab, "capture_model_per_arm", lambda value, profile: selected)
    wrapped = dev_ab.with_ab_capture(original)
    assert wrapped(runner, profile_only=profile_only) == (17 if selected is None else selected)
    assert calls == ([profile_only] if selected is None else [])
    assert wrapped.__wrapped__ is original
