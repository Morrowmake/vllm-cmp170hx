# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: the prefill overlap never defers an all-reduce the MoE reads inline.

The GLM-5 prefill overlap installs a TP all-reduce interceptor that returns a
destination buffer before the collective has run, and joins it after the MLP
block returns. ``MoERunner`` has two reductions whose result it reads *inside*
``forward`` (the shared-expert all-reduce when the MoE kernel reports
``output_is_reduced``, and the early routed all-reduce ahead of a
non-commutative routed output transform). Under the interceptor either one
would compute on a buffer still in flight: wrong values, no error.

Covered here, on the host, with the real ``MoERunner.forward`` bound onto a
stub that supplies only the attributes it touches, and the real
``PrefillOverlapRegion`` driven by an executor fake whose buffers stay NaN
until joined:

* the production configuration (TP, no SP, ``output_is_reduced`` False,
  shared experts, no routed transform) reports no reason, issues exactly one
  all-reduce, and produces bit-identical output with and without the overlap;
* the unsafe configurations report a reason, and the guard raises instead of
  returning NaN (the unguarded behaviour is reproduced to show the hazard);
* ``maybe_open_region`` asks the model once and stays off for the process when
  there is a reason, and opens exactly as before when there is none;
* ``Glm5NextModel._overlap_unsafe_reason`` scans only MoE layers and names the
  first unsafe one.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import communication_op, parallel_state
from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.glm5next.common import overlap as ov
from vllm.models.glm5next.common.model import Glm5NextModel

WORLD = 4
HIDDEN = 16
TOKENS = 1152  # the production prefill chunk
SCALE = 2.5


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class DeferredExecutor(ov.OverlapExecutor):
    """Side-stream fake: a destination stays NaN until its signal is joined."""

    def __init__(self) -> None:
        self._queue: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._done = 0

    def fork(self) -> None:
        pass

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        self._queue.append((inp.clone(), out))

    def signal(self):
        return len(self._queue)

    def join(self, token) -> None:
        while self._done < token:
            inp, out = self._queue[self._done]
            out.copy_(inp * WORLD)
            self._done += 1

    def keepalive(self, *tensors: torch.Tensor) -> None:
        pass

    def empty_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tensor, float("nan"))


def make_region() -> ov.PrefillOverlapRegion:
    settings = ov.OverlapSettings(enabled=True, splits=2, min_tokens=1)
    region = ov.PrefillOverlapRegion(
        settings, DeferredExecutor(), fallback_all_reduce=lambda t: t * WORLD
    )
    return region


class _Transform(torch.nn.Module):
    """A routed output transform; row-wise, so the test values stay exact."""

    def __init__(self, commutative: bool) -> None:
        super().__init__()
        self.reduce_commutative = commutative

    def forward(self, x):
        return x * 3.0


class _StubRunner:
    """Carries the real forward and reduction logic of ``MoERunner``."""

    forward = MoERunner.forward
    inline_all_reduce_reason = MoERunner.inline_all_reduce_reason
    _refuse_deferred_all_reduce = staticmethod(MoERunner._refuse_deferred_all_reduce)
    _fused_output_is_reduced = MoERunner._fused_output_is_reduced
    _maybe_reduce_shared_expert_output = MoERunner._maybe_reduce_shared_expert_output
    _maybe_reduce_routed_output_before_transform = (
        MoERunner._maybe_reduce_routed_output_before_transform
    )
    _maybe_reduce_final_output = MoERunner._maybe_reduce_final_output
    _maybe_apply_routed_scale_to_output = MoERunner._maybe_apply_routed_scale_to_output
    apply_routed_input_transform = MoERunner.apply_routed_input_transform
    apply_routed_output_transform = MoERunner.apply_routed_output_transform
    _maybe_pad_hidden_states = MoERunner._maybe_pad_hidden_states
    _maybe_add_zero_expert_output = MoERunner._maybe_add_zero_expert_output

    def __init__(
        self,
        *,
        output_is_reduced: bool = False,
        sequence_parallel: bool = False,
        transform: torch.nn.Module | None = None,
        shared: bool = True,
    ) -> None:
        torch.manual_seed(0)
        self.moe_config = SimpleNamespace(
            is_sequence_parallel=sequence_parallel,
            tp_size=WORLD,
            ep_size=1,
            skip_final_all_reduce=False,
            hidden_dim=HIDDEN,
            hidden_dim_unpadded=HIDDEN,
        )
        kernel = SimpleNamespace(output_is_reduced=lambda: output_is_reduced)
        self._quant_method = SimpleNamespace(
            moe_kernel=kernel, skip_forward_padding=True, has_unpadded_output=False
        )
        self._shared_experts = object() if shared else None
        self.routed_input_transform = None
        self.routed_output_transform = transform
        self.routed_scaling_factor = SCALE
        self.router = None
        self.w_shared = torch.randn(HIDDEN, HIDDEN, dtype=torch.float64)
        self.w_routed = torch.randn(HIDDEN, HIDDEN, dtype=torch.float64)
        self.output_is_reduced = output_is_reduced

    def _encode_layer_name(self):
        return "layers.3.mlp.experts"

    def _forward_entry(self, hidden, router_logits, shared_in, input_ids, name, pad):
        fused = hidden @ self.w_routed
        if self.output_is_reduced:
            # The kernel's own combine already summed the ranks.
            fused = fused * WORLD
        if self._shared_experts is None:
            return fused
        return shared_in @ self.w_shared, fused


def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(TOKENS, HIDDEN, dtype=torch.float64)


def expected(runner: _StubRunner, x: torch.Tensor) -> torch.Tensor:
    """What four ranks holding the same partials would return."""
    shared = x @ runner.w_shared * WORLD
    routed = x @ runner.w_routed * WORLD * SCALE
    if runner.routed_output_transform is not None:
        routed = routed * 3.0
    return shared + routed


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ov.reset_for_testing()
    previous = communication_op.set_tp_all_reduce_interceptor(None)
    calls: list[int] = []

    class _Group:
        def all_reduce(self, t):
            calls.append(int(t.shape[0]))
            return t * WORLD

    # Plain TP all-reduce on the main stream when no interceptor is installed.
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: _Group())
    yield calls
    communication_op.set_tp_all_reduce_interceptor(previous)
    ov.reset_for_testing()


# --------------------------------------------------------------------------- #
# Production configuration: unchanged
# --------------------------------------------------------------------------- #


def test_production_config_reports_no_reason():
    assert _StubRunner().inline_all_reduce_reason() is None


def test_production_config_plain_tp_is_one_final_all_reduce(_clean):
    runner = _StubRunner()
    x = inputs()
    out = runner.forward(x, x)
    assert _clean == [TOKENS], "exactly one all-reduce, of the combined output"
    assert torch.equal(out, expected(runner, x))


@pytest.mark.parametrize("splits", [1, 2])
def test_production_config_is_bit_identical_under_the_overlap(_clean, splits):
    runner = _StubRunner()
    x = inputs()
    plain = runner.forward(x, x)

    region = make_region()
    with region.capture(splits) as handles:
        deferred = runner.forward(x, x)
    handle = region.single(handles, "the MLP down projection")
    assert handle is not None and not region.degraded
    assert handle.out is deferred
    handle.join_all()
    assert not torch.isnan(deferred).any()
    assert torch.equal(deferred, plain)


# --------------------------------------------------------------------------- #
# Unsafe configurations: refused, loudly
# --------------------------------------------------------------------------- #


def test_output_is_reduced_reports_a_reason():
    reason = _StubRunner(output_is_reduced=True).inline_all_reduce_reason()
    assert reason is not None and "output_is_reduced" in reason


def test_unguarded_shared_reduce_reads_an_unfinished_buffer(monkeypatch):
    """What the guard prevents: without it the sum is NaN here (garbage on a
    real side stream), and ``region.single`` cannot tell, because there is
    still exactly one intercepted collective."""
    monkeypatch.setattr(
        _StubRunner, "_refuse_deferred_all_reduce", staticmethod(lambda what: None)
    )
    runner = _StubRunner(output_is_reduced=True)
    x = inputs()
    region = make_region()
    with region.capture(2) as handles:
        out = runner.forward(x, x)
    handle = region.single(handles, "the MLP down projection")
    assert handle is not None and not region.degraded
    handle.join_all()
    assert torch.isnan(out).all()


def test_guard_raises_on_the_shared_reduce_under_the_overlap():
    runner = _StubRunner(output_is_reduced=True)
    x = inputs()
    region = make_region()
    with pytest.raises(RuntimeError, match="shared-expert output"):
        with region.capture(2):
            runner.forward(x, x)
    # The interceptor is restored even though the block raised.
    assert communication_op.get_tp_all_reduce_interceptor() is None


def test_guard_is_silent_on_the_same_config_without_an_interceptor(_clean):
    runner = _StubRunner(output_is_reduced=True)
    x = inputs()
    out = runner.forward(x, x)
    assert _clean == [TOKENS], "only the shared-expert reduce; no final reduce"
    assert torch.equal(out, expected(runner, x))


def test_non_commutative_transform_reports_and_raises():
    runner = _StubRunner(transform=_Transform(commutative=False))
    reason = runner.inline_all_reduce_reason()
    assert reason is not None and "transform" in reason
    x = inputs()
    with pytest.raises(RuntimeError, match="routed output"):
        with make_region().capture(2):
            runner.forward(x, x)


def test_commutative_transform_is_safe():
    runner = _StubRunner(transform=_Transform(commutative=True))
    assert runner.inline_all_reduce_reason() is None
    x = inputs()
    region = make_region()
    with region.capture(2) as handles:
        out = runner.forward(x, x)
    region.single(handles, "mlp").join_all()
    assert torch.equal(out, expected(runner, x))


def test_sequence_parallel_reports_no_reason():
    # SP never takes either inline reduction (and the overlap refuses SP).
    runner = _StubRunner(output_is_reduced=True, sequence_parallel=True)
    assert runner.inline_all_reduce_reason() is None


# --------------------------------------------------------------------------- #
# maybe_open_region: ask once, then stay off
# --------------------------------------------------------------------------- #


@pytest.fixture
def openable(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_PREFILL_OVERLAP", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: WORLD
    )
    built: list[ov.PrefillOverlapRegion] = []

    def _build(settings):
        region = make_region()
        built.append(region)
        return region

    monkeypatch.setattr(ov, "_build_region", _build)
    ov.reset_for_testing()
    return built


def _open(**kwargs):
    return ov.maybe_open_region(
        num_tokens=TOKENS, mhc=True, sequence_parallel=False, **kwargs
    )


def test_open_region_stays_off_when_the_model_gives_a_reason(openable, caplog):
    asked: list[int] = []

    def reason():
        asked.append(1)
        return "layer 7: the MoE kernel reduces its own output"

    assert _open(unsafe_reason=reason) is None
    assert _open(unsafe_reason=reason) is None
    assert asked == [1], "asked once, then the feature is off for the process"
    assert openable == [], "no region, no comm stream"
    assert ov.get_settings().enabled is False
    assert ov.get_active_region() is None


def test_open_region_logs_the_reason(openable, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        ov.logger, "warning", lambda msg, *a: seen.append(msg % a if a else msg)
    )
    _open(unsafe_reason=lambda: "layer 7: because")
    assert len(seen) == 1
    assert "prefill overlap stays off" in seen[0] and "layer 7: because" in seen[0]


def test_open_region_opens_as_before_when_there_is_no_reason(openable):
    asked: list[int] = []

    def reason():
        asked.append(1)
        return None

    first = _open(unsafe_reason=reason)
    ov.close_region(first)
    second = _open(unsafe_reason=reason)
    assert first is not None and second is first
    assert asked == [1], "only asked when the region is first built"
    assert len(openable) == 1


def test_open_region_without_the_argument_is_unchanged(openable):
    region = _open()
    assert region is not None and ov.get_active_region() is region


def test_flag_off_never_asks(monkeypatch):
    monkeypatch.delenv("VLLM_GLM5_PREFILL_OVERLAP", raising=False)
    ov.reset_for_testing()

    def reason():
        raise AssertionError("asked with the feature off")

    assert _open(unsafe_reason=reason) is None


# --------------------------------------------------------------------------- #
# The model-side scan
# --------------------------------------------------------------------------- #


def _layer(idx: int, *, moe: bool, reason: str | None = None):
    layer = SimpleNamespace(layer_idx=idx, _mlp_is_moe=moe)
    if moe:
        experts = SimpleNamespace(inline_all_reduce_reason=lambda: reason)
        layer.mlp = SimpleNamespace(experts=experts)
    else:
        layer.mlp = SimpleNamespace()  # a dense MLP has no experts
    return layer


def test_model_scan_production_layout_is_safe():
    model = SimpleNamespace(
        _active_layers=[_layer(0, moe=False)] + [_layer(i, moe=True) for i in range(1, 6)]
    )
    assert Glm5NextModel._overlap_unsafe_reason(model) is None


def test_model_scan_names_the_first_unsafe_layer():
    model = SimpleNamespace(
        _active_layers=[
            _layer(0, moe=False),
            _layer(1, moe=True),
            _layer(2, moe=True, reason="r2"),
            _layer(3, moe=True, reason="r3"),
        ]
    )
    assert Glm5NextModel._overlap_unsafe_reason(model) == "layer 2: r2"


def test_model_forward_hands_the_scan_to_the_region():
    src = inspect.getsource(Glm5NextModel.forward)
    assert "unsafe_reason=self._overlap_unsafe_reason" in src


def test_runner_module_uses_the_live_interceptor_getter():
    # The guard must read the interceptor at call time, not a copy.
    assert mr.get_tp_all_reduce_interceptor is communication_op.get_tp_all_reduce_interceptor
