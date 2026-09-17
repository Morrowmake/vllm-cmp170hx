# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the GLM-5.3-Flash prefill comm/compute overlap.

These cover everything that does not need four GPUs: the environment parsing,
the micro-batch split/merge algebra, the stream-ordering contract (through an
executor fake that *defers* each all-reduce until it is joined, so a driver
that reads a buffer too early sees NaN), the invariant that attention is never
split, and that the feature-off path is the unmodified one.

The GPU-side checks -- bitwise/logprob agreement against the default server and
a profiler capture proving the NCCL kernels really are concurrent -- live in
the standalone GPU overlap validation suite.
"""

import pytest
import torch

from vllm.distributed import communication_op
from vllm.models.glm5next.common import overlap as ov
from vllm.models.glm5next.common.model import Glm5NextDecoderLayer

WORLD = 4


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class DeferredExecutor(ov.OverlapExecutor):
    """Executor fake that models a real side stream.

    An ``all_reduce`` does not touch its destination until the main stream
    joins the matching signal, and destinations start out NaN. A driver that
    consumes a micro-batch before joining its collective therefore produces
    NaN rather than silently passing.
    """

    def __init__(self, world_size: int = WORLD) -> None:
        self.world_size = world_size
        self.log: list[tuple] = []
        self._queue: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._done = 0

    # -- OverlapExecutor ---------------------------------------------------- #

    def fork(self) -> None:
        self.log.append(("fork",))

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        assert inp.shape == out.shape
        self.log.append(("all_reduce", int(inp.shape[0])))
        self._queue.append((inp.clone(), out))

    def signal(self):
        token = len(self._queue)
        self.log.append(("signal", token))
        return token

    def join(self, token) -> None:
        self.log.append(("join", token))
        while self._done < token:
            inp, out = self._queue[self._done]
            out.copy_(inp * self.world_size)
            self._done += 1

    def keepalive(self, *tensors: torch.Tensor) -> None:
        self.log.append(("keepalive", len(tensors)))

    def empty_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tensor, float("nan"))

    def reset_events(self) -> None:
        self.log.append(("reset_events",))

    # -- helpers ------------------------------------------------------------ #

    def ops(self, name: str) -> list[tuple]:
        return [entry for entry in self.log if entry[0] == name]


def make_region(splits: int = 2, world_size: int = WORLD):
    settings = ov.OverlapSettings(enabled=True, splits=splits, min_tokens=1)
    executor = DeferredExecutor(world_size)
    region = ov.PrefillOverlapRegion(
        settings, executor, fallback_all_reduce=lambda t: t * world_size
    )
    return region, executor


HIDDEN = 16
STREAMS = 4


def _rowwise_mix(x, residual, post, comb, fn, scale, base, norm_weight, norm_eps):
    """Stand-in for hc_fused_post_pre: strictly row-wise, same arity/shapes.

    residual [T, n, H] -> [T, n, H]; post [T, n]; comb [T, n, n]; x [T, H].
    """
    residual_out = residual + x.unsqueeze(1) * post.unsqueeze(-1)
    post_out = post * float(scale) + float(base)
    comb_out = comb + post.unsqueeze(-1) * post.unsqueeze(-2)
    layer_input = (residual_out.mean(dim=1) * norm_weight) / (
        residual_out.mean(dim=1).abs().amax(dim=-1, keepdim=True) + norm_eps
    )
    return residual_out, post_out, comb_out, layer_input + float(fn)


class FakeLayer:
    """Minimal stand-in carrying the real ``_forward_attn_ffn_overlapped``."""

    _forward_attn_ffn_overlapped = (
        Glm5NextDecoderLayer._forward_attn_ffn_overlapped
    )

    def __init__(self, *, is_last: bool = False, moe: bool = True) -> None:
        torch.manual_seed(0)
        self.is_sequence_parallel = False
        self._mlp_is_moe = moe
        self.n = STREAMS
        self.layer_idx = 3
        self.num_hidden_layers = 4 if is_last else 40
        self.hc_ffn_fn = 0.25
        self.hc_ffn_scale = 1.5
        self.hc_ffn_base = 0.125
        self.post_attention_layernorm = torch.nn.RMSNorm(HIDDEN)
        self.post_attention_layernorm.variance_epsilon = 1e-6
        self.attn_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float64)
        self.mlp_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float64)
        self.attn_calls: list[int] = []
        self.mlp_calls: list[int] = []

    # the pieces the driver calls
    def self_attn(self, *, hidden_states, positions):
        self.attn_calls.append(int(hidden_states.shape[0]))
        partial = hidden_states @ self.attn_weight
        return communication_op.tensor_model_parallel_all_reduce(partial)

    def hc_fused_post_pre(self, x, residual, post, comb, fn, scale, base,
                          norm_weight, norm_eps):
        return _rowwise_mix(
            x, residual, post, comb, fn, scale, base,
            norm_weight.to(x.dtype), norm_eps,
        )

    def mlp(self, x, already_sequence_parallel=False):
        self.mlp_calls.append(int(x.shape[0]))
        partial = x @ self.mlp_weight
        return communication_op.tensor_model_parallel_all_reduce(partial)

    def hc_post(self, x, residual, post, comb):
        return residual + x.unsqueeze(1) * post.unsqueeze(-1)

    # the unsplit reference, written out so the test does not depend on the
    # production tail staying textually identical
    def reference(self, positions, x, residual, post, comb):
        x = self.self_attn(hidden_states=x, positions=positions)
        residual, post, comb, x = self.hc_fused_post_pre(
            x, residual, post, comb,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
        x = self.mlp(x)
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            return x.mean(dim=1), None, None, None
        return x, residual, post, comb


def make_inputs(num_tokens: int):
    torch.manual_seed(1)
    return (
        torch.arange(num_tokens),
        torch.randn(num_tokens, HIDDEN, dtype=torch.float64),
        torch.randn(num_tokens, STREAMS, HIDDEN, dtype=torch.float64),
        torch.randn(num_tokens, STREAMS, dtype=torch.float64),
        torch.randn(num_tokens, STREAMS, STREAMS, dtype=torch.float64),
    )


@pytest.fixture(autouse=True)
def _clean_hook():
    ov.reset_for_testing()
    previous = communication_op.set_tp_all_reduce_interceptor(None)
    yield
    communication_op.set_tp_all_reduce_interceptor(previous)
    ov.reset_for_testing()


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #


def test_settings_default_off():
    settings = ov.read_settings({})
    assert settings.enabled is False
    assert settings.active is False
    assert settings.splits == 2
    assert settings.min_tokens == 512


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_settings_truthy(raw):
    assert ov.read_settings({"VLLM_GLM5_PREFILL_OVERLAP": raw}).enabled is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", ""])
def test_settings_falsy(raw):
    assert ov.read_settings({"VLLM_GLM5_PREFILL_OVERLAP": raw}).enabled is False


def test_settings_splits_and_min_tokens():
    settings = ov.read_settings(
        {
            "VLLM_GLM5_PREFILL_OVERLAP": "1",
            "VLLM_GLM5_PREFILL_OVERLAP_SPLITS": "4",
            "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS": "1024",
        }
    )
    assert (settings.splits, settings.min_tokens, settings.active) == (4, 1024, True)


def test_settings_splits_one_is_inactive():
    settings = ov.read_settings(
        {"VLLM_GLM5_PREFILL_OVERLAP": "1", "VLLM_GLM5_PREFILL_OVERLAP_SPLITS": "1"}
    )
    assert settings.enabled is True and settings.active is False


@pytest.mark.parametrize(
    "env",
    [
        {"VLLM_GLM5_PREFILL_OVERLAP_SPLITS": "zero"},
        {"VLLM_GLM5_PREFILL_OVERLAP_SPLITS": "0"},
        {"VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS": "-1"},
    ],
)
def test_settings_reject_bad_values(env):
    with pytest.raises(ValueError):
        ov.read_settings(env)


# --------------------------------------------------------------------------- #
# split/merge algebra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("splits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 8, 100, 576, 1152, 2048])
def test_split_bounds_tile_the_range(num_tokens, splits):
    bounds = ov.split_bounds(num_tokens, splits)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == num_tokens
    for (_, hi), (lo, _) in zip(bounds, bounds[1:]):
        assert hi == lo
    assert all(hi > lo for lo, hi in bounds) or num_tokens == 0


def test_split_bounds_production_shape():
    assert ov.split_bounds(1152, 2) == [(0, 576), (576, 1152)]
    assert ov.split_bounds(1152, 4) == [
        (0, 288), (288, 576), (576, 864), (864, 1152)
    ]


def test_split_bounds_alignment_and_degenerate():
    # every slice but the last is a multiple of the alignment
    for lo, hi in ov.split_bounds(1000, 3)[:-1]:
        assert (hi - lo) % ov.TOKEN_ALIGN == 0
    # too few tokens to split -> one slice, never an empty micro-batch
    assert ov.split_bounds(7, 4) == [(0, 7)]
    assert ov.split_bounds(0, 4) == [(0, 0)]
    assert ov.split_bounds(1152, 1) == [(0, 1152)]


@pytest.mark.parametrize("splits", [2, 3, 4])
def test_slice_then_cat_round_trips(splits):
    x = torch.randn(1152, STREAMS, HIDDEN, dtype=torch.float64)
    parts = [x[lo:hi] for lo, hi in ov.split_bounds(1152, splits)]
    assert all(p.is_contiguous() for p in parts)
    assert torch.equal(torch.cat(parts, dim=0), x)


# --------------------------------------------------------------------------- #
# The interceptor and its ordering contract
# --------------------------------------------------------------------------- #


def test_capture_installs_and_restores_the_hook():
    region, _ = make_region()
    sentinel = object()
    communication_op.set_tp_all_reduce_interceptor(sentinel)
    with region.capture(2):
        installed = communication_op.get_tp_all_reduce_interceptor()
        assert installed.__self__ is region
        assert installed.__func__ is ov.PrefillOverlapRegion.submit
    assert communication_op.get_tp_all_reduce_interceptor() is sentinel


def test_capture_restores_the_hook_on_exception():
    region, _ = make_region()
    with pytest.raises(RuntimeError):
        with region.capture(2):
            raise RuntimeError("boom")
    assert communication_op.get_tp_all_reduce_interceptor() is None


def test_submit_slices_the_message():
    region, executor = make_region(splits=4)
    with region.capture(4) as handles:
        out = communication_op.tensor_model_parallel_all_reduce(
            torch.zeros(1152, HIDDEN)
        )
    assert out.shape == (1152, HIDDEN)
    assert len(handles) == 1
    assert handles[0].bounds == ov.split_bounds(1152, 4)
    assert executor.ops("all_reduce") == [("all_reduce", 288)] * 4
    assert region.async_submitted == 1 and region.slices == 4


def test_submit_does_not_reslice_a_micro_batch():
    region, executor = make_region(splits=4)
    with region.capture(1):
        communication_op.tensor_model_parallel_all_reduce(torch.zeros(288, HIDDEN))
    assert executor.ops("all_reduce") == [("all_reduce", 288)]


def test_submit_falls_back_when_too_few_rows():
    region, executor = make_region(splits=4)
    with region.capture(4) as handles:
        out = communication_op.tensor_model_parallel_all_reduce(torch.ones(2, HIDDEN))
    assert handles[0].synchronous is True
    assert executor.ops("all_reduce") == []
    assert torch.equal(out, torch.full((2, HIDDEN), float(WORLD)))
    handles[0].join_all()  # must be a no-op, not an index error


def test_single_joins_and_degrades_on_an_unexpected_collective_count():
    region, executor = make_region(splits=2)
    with region.capture(2) as handles:
        communication_op.tensor_model_parallel_all_reduce(torch.zeros(64, HIDDEN))
        communication_op.tensor_model_parallel_all_reduce(torch.zeros(64, HIDDEN))
    assert region.single(handles, "a two-collective block") is None
    assert region.degraded is True
    # everything was joined, so no buffer is left in flight
    assert executor._done == len(executor._queue)


# --------------------------------------------------------------------------- #
# The layer driver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("is_last", [False, True])
def test_overlapped_layer_matches_the_unsplit_reference(splits, is_last):
    num_tokens = 1152
    region, _ = make_region(splits=splits)

    reference_layer = FakeLayer(is_last=is_last)
    communication_op.set_tp_all_reduce_interceptor(lambda t: t * WORLD)
    expected = reference_layer.reference(*make_inputs(num_tokens))
    communication_op.set_tp_all_reduce_interceptor(None)

    layer = FakeLayer(is_last=is_last)
    got = layer._forward_attn_ffn_overlapped(*make_inputs(num_tokens), region)

    for want, have in zip(expected, got):
        if want is None:
            assert have is None
            continue
        assert not torch.isnan(have).any(), "a buffer was read before its join"
        # Every split operation is row-wise, so this is exact even in float64.
        assert torch.equal(want, have)


@pytest.mark.parametrize("splits", [2, 4])
def test_attention_is_never_split(splits):
    """The KDA scan and the DSA indexer carry state across tokens.

    This locks in the design decision: whatever the split factor, attention is
    called exactly once per layer with the whole chunk.
    """
    region, _ = make_region(splits=splits)
    layer = FakeLayer()
    layer._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert layer.attn_calls == [1152]
    assert layer.mlp_calls == [hi - lo for lo, hi in ov.split_bounds(1152, splits)]
    assert sum(layer.mlp_calls) == 1152


def test_pipeline_order_keeps_the_comm_stream_ahead():
    """AR_a[i+1] must already be queued when the MLP of slice i runs.

    That is the whole point: the attention-output collectives are all enqueued
    before any MLP compute, so the comm stream never runs dry while the main
    stream works through the micro-batches.
    """
    region, executor = make_region(splits=2)
    layer = FakeLayer()
    layer._forward_attn_ffn_overlapped(*make_inputs(1152), region)

    kinds = [entry for entry in executor.log if entry[0] in ("all_reduce", "join")]
    # 2 attention slices, then the per-micro-batch joins/MLP collectives.
    assert kinds[0] == ("all_reduce", 576)
    assert kinds[1] == ("all_reduce", 576)
    assert kinds[2][0] == "join"          # join AR_a[0]
    assert kinds[3] == ("all_reduce", 576)  # AR_m[0], issued behind AR_a[1]
    assert kinds[4][0] == "join"          # join AR_a[1]
    assert kinds[5] == ("all_reduce", 576)  # AR_m[1]
    # the two MLP collectives are joined only after both were issued
    assert [k[0] for k in kinds[6:]] == ["join", "join"]


def test_every_result_is_joined_before_it_is_returned():
    region, executor = make_region(splits=4)
    layer = FakeLayer()
    out = layer._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert executor._done == len(executor._queue)
    for tensor in out:
        if tensor is not None:
            assert not torch.isnan(tensor).any()


def test_splits_of_one_degenerate_to_the_plain_path():
    region, executor = make_region(splits=1)
    layer = FakeLayer()
    layer._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert layer.attn_calls == [1152] and layer.mlp_calls == [1152]
    assert executor.ops("all_reduce") == [("all_reduce", 1152)] * 2


def test_dense_mlp_layer_takes_the_no_kwarg_path():
    region, _ = make_region(splits=2)
    layer = FakeLayer(moe=False)
    layer._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert layer.mlp_calls == [576, 576]


# --------------------------------------------------------------------------- #
# The feature-off path
# --------------------------------------------------------------------------- #


def test_all_reduce_delegates_to_the_tp_group_when_no_hook_is_installed(monkeypatch):
    calls = []

    class Group:
        def all_reduce(self, tensor):
            calls.append(tensor)
            return tensor * WORLD

    monkeypatch.setattr(communication_op, "get_tp_group", Group)
    tensor = torch.ones(4, HIDDEN)
    out = communication_op.tensor_model_parallel_all_reduce(tensor)
    assert len(calls) == 1 and calls[0] is tensor
    assert torch.equal(out, tensor * WORLD)


def test_no_active_region_by_default():
    assert ov.get_active_region() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_tokens=1152, mhc=True, sequence_parallel=False),   # flag off
        dict(num_tokens=4, mhc=True, sequence_parallel=False),      # decode size
        dict(num_tokens=1152, mhc=False, sequence_parallel=False),  # no mHC
        dict(num_tokens=1152, mhc=True, sequence_parallel=True),    # SP
    ],
)
def test_maybe_open_region_declines(kwargs, monkeypatch):
    # Flag on for every case but the first, so each guard is exercised alone.
    if kwargs["num_tokens"] != 1152 or not kwargs["mhc"] or kwargs["sequence_parallel"]:
        monkeypatch.setenv("VLLM_GLM5_PREFILL_OVERLAP", "1")
    ov.reset_for_testing()
    assert ov.maybe_open_region(**kwargs) is None
    assert ov.get_active_region() is None


def test_close_region_clears_the_active_region():
    region, _ = make_region()
    ov._ACTIVE = region
    ov.close_region(region)
    assert ov.get_active_region() is None
    assert region.submitted == 0 and region.slices == 0
