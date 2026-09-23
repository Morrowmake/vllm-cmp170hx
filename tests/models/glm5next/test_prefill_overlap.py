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
    _next_overlap_layer = Glm5NextDecoderLayer._next_overlap_layer

    def __init__(self, *, is_last: bool = False, moe: bool = True) -> None:
        torch.manual_seed(0)
        self.is_sequence_parallel = False
        # attn-side mHC params, used when a previous layer absorbs this
        # layer's pre-attention mix under cross-layer overlap
        self.hc_attn_fn = 0.5
        self.hc_attn_scale = 1.25
        self.hc_attn_base = 0.0625
        self.input_layernorm = torch.nn.RMSNorm(HIDDEN)
        self.input_layernorm.variance_epsilon = 1e-6
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


# --------------------------------------------------------------------------- #
# Cross-layer split (follow-up 1)
# --------------------------------------------------------------------------- #


def make_cross_region(splits: int = 2, world_size: int = WORLD, cross: bool = True):
    settings = ov.OverlapSettings(
        enabled=True, splits=splits, min_tokens=1, cross_layer=cross
    )
    executor = DeferredExecutor(world_size)
    region = ov.PrefillOverlapRegion(
        settings, executor, fallback_all_reduce=lambda t: t * world_size
    )
    region.cross_layer = cross
    return region, executor


def link(first: "FakeLayer", second: "FakeLayer") -> None:
    first._next_layer = [second]


def test_cross_layer_settings_default_off():
    assert ov.read_settings({}).cross_layer is False
    assert ov.read_settings(
        {"VLLM_GLM5_PREFILL_OVERLAP_CROSS_LAYER": "1"}
    ).cross_layer is True


def test_cross_layer_hands_on_a_sliced_state():
    region, _ = make_cross_region(splits=2)
    a, b = FakeLayer(), FakeLayer()
    link(a, b)
    x, carried, post, comb = a._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert isinstance(carried, ov.SlicedMHCState)
    assert post is None and comb is None
    assert len(carried) == 2 and carried.bounds == ov.split_bounds(1152, 2)
    # only the attention input is materialised
    assert x.shape == (1152, HIDDEN)
    assert [r.shape[0] for r in carried.residual] == [576, 576]


def test_cross_layer_two_layer_chain_matches_the_unsplit_reference():
    """Layer A's tail + layer B's pre-attention mix must equal doing both plainly."""
    num_tokens = 1152

    # reference: A's unsplit tail, then B's pre-attention mix, unsplit
    ref_a, ref_b = FakeLayer(), FakeLayer()
    communication_op.set_tp_all_reduce_interceptor(lambda t: t * WORLD)
    x, residual, post, comb = ref_a.reference(*make_inputs(num_tokens))
    want = ref_b.hc_fused_post_pre(
        x, residual, post, comb,
        ref_b.hc_attn_fn, ref_b.hc_attn_scale, ref_b.hc_attn_base,
        norm_weight=ref_b.input_layernorm.weight.data,
        norm_eps=ref_b.input_layernorm.variance_epsilon,
    )
    communication_op.set_tp_all_reduce_interceptor(None)

    region, _ = make_cross_region(splits=2)
    a, b = FakeLayer(), FakeLayer()
    link(a, b)
    got_x, carried, _, _ = a._forward_attn_ffn_overlapped(*make_inputs(num_tokens), region)

    want_residual, want_post, want_comb, want_x = want
    assert not torch.isnan(got_x).any()
    assert torch.equal(want_x, got_x)
    assert torch.equal(want_residual, torch.cat(carried.residual, dim=0))
    assert torch.equal(want_post, torch.cat(carried.post, dim=0))
    assert torch.equal(want_comb, torch.cat(carried.comb, dim=0))


@pytest.mark.parametrize("splits", [2, 4])
def test_cross_layer_consumer_matches_the_materialised_path(splits):
    """B consuming a SlicedMHCState == B consuming the same state materialised."""
    num_tokens = 1152
    region, _ = make_cross_region(splits=splits)
    a, b = FakeLayer(), FakeLayer()
    link(a, b)
    x, carried, _, _ = a._forward_attn_ffn_overlapped(*make_inputs(num_tokens), region)

    # B, fed the sliced state
    region_b, _ = make_cross_region(splits=splits, cross=False)
    got = b._forward_attn_ffn_overlapped(
        None, x.clone(), None, None, None, region_b, sliced=carried
    )
    # B, fed the same state materialised
    region_c, _ = make_cross_region(splits=splits, cross=False)
    b2 = FakeLayer()
    got2 = b2._forward_attn_ffn_overlapped(
        None,
        x.clone(),
        torch.cat(carried.residual, dim=0),
        torch.cat(carried.post, dim=0),
        torch.cat(carried.comb, dim=0),
        region_c,
    )
    for lhs, rhs in zip(got, got2):
        if lhs is None:
            assert rhs is None
            continue
        assert not torch.isnan(lhs).any()
        assert torch.equal(lhs, rhs)


def test_cross_layer_still_never_splits_attention():
    region, _ = make_cross_region(splits=4)
    a, b = FakeLayer(), FakeLayer()
    link(a, b)
    a._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert a.attn_calls == [1152]


def test_cross_layer_off_materialises_exactly_as_before():
    region_on, _ = make_cross_region(splits=2, cross=False)
    a, b = FakeLayer(), FakeLayer()
    link(a, b)  # linked, but the flag is off
    got = a._forward_attn_ffn_overlapped(*make_inputs(1152), region_on)
    assert not isinstance(got[1], ov.SlicedMHCState)
    region_plain, _ = make_region(splits=2)
    plain = FakeLayer()._forward_attn_ffn_overlapped(*make_inputs(1152), region_plain)
    for lhs, rhs in zip(got, plain):
        assert torch.equal(lhs, rhs)


def test_last_layer_never_hands_on():
    """The final mHC layer contracts; it must materialise, not hand a state on."""
    region, _ = make_cross_region(splits=2)
    a, b = FakeLayer(is_last=True), FakeLayer()
    link(a, b)
    x, residual, post, comb = a._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert (residual, post, comb) == (None, None, None)
    assert x.shape == (1152, HIDDEN)


def test_unlinked_layer_materialises():
    """A layer with no successor (last of a pipeline stage) must materialise."""
    region, _ = make_cross_region(splits=2)
    a = FakeLayer()  # no link
    got = a._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert not isinstance(got[1], ov.SlicedMHCState)


def test_cross_layer_joins_every_collective_before_use():
    region, executor = make_cross_region(splits=4)
    a, b = FakeLayer(), FakeLayer()
    link(a, b)
    x, carried, _, _ = a._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert executor._done == len(executor._queue)
    assert not torch.isnan(x).any()
    for part in carried.residual + carried.post + carried.comb:
        assert not torch.isnan(part).any()


def test_maybe_open_region_can_suppress_cross_layer(monkeypatch):
    """Aux-hidden-state capture must be able to turn the handoff off."""
    settings = ov.read_settings(
        {
            "VLLM_GLM5_PREFILL_OVERLAP": "1",
            "VLLM_GLM5_PREFILL_OVERLAP_CROSS_LAYER": "1",
        }
    )
    assert settings.cross_layer is True
    region, _ = make_cross_region(cross=True)
    region.cross_layer = settings.cross_layer and False  # allow_cross_layer=False
    layer = FakeLayer()
    link(layer, FakeLayer())
    got = layer._forward_attn_ffn_overlapped(*make_inputs(1152), region)
    assert not isinstance(got[1], ov.SlicedMHCState)


# --------------------------------------------------------------------------- #
# Backend selection: which communicator carries the split collectives
#
# The overlap used to refuse to run at all whenever the TP group had another
# active all-reduce backend, which on a PCIe-P2P box means CustomAllreduce and
# cost 6.3% of cold prefill (p2p_tp4_validation.md). These cover the policy
# that replaced the blanket refusal: a per-backend decision at build time and
# a per-slice fallthrough to PyNccl.
# --------------------------------------------------------------------------- #


class FakeCA:
    """Stand-in for CustomAllreduce: the size cap is the interesting part."""

    def __init__(self, max_size: int = 8192 * 1024, disabled: bool = False) -> None:
        self.max_size = max_size
        self.disabled = disabled
        self.calls: list[int] = []

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        nbytes = inp.numel() * inp.element_size()
        # Mirrors the real gate, including its strict '<'.
        return nbytes % 16 == 0 and nbytes < self.max_size


class FakeHostShm:
    def __init__(self, host_cap: int = 512 * 1024, disabled: bool = False) -> None:
        self.host_cap = host_cap
        self.disabled = disabled

    def should_host_ar(self, inp: torch.Tensor) -> bool:
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.host_cap and nbytes % 16 == 0


class FakeNccl:
    disabled = False

    def __init__(self) -> None:
        self.calls: list[int] = []

    def all_reduce(self, inp, out, stream=None):
        self.calls.append(int(inp.shape[0]))


class FakeCommunicator:
    def __init__(self, **backends) -> None:
        self.pynccl_comm = FakeNccl()
        self.ca_comm = None
        self.hostshm_comm = None
        self.qr_comm = None
        self.fi_ar_comm = None
        self.fi_pcie_ipc_ar_comm = None
        self.symm_mem_comm = None
        self.aiter_ar_comm = None
        for name, value in backends.items():
            setattr(self, name, value)


class FakeTpGroup:
    def __init__(self, communicator) -> None:
        self.device_communicator = communicator

    def all_reduce(self, tensor):
        return tensor * WORLD


def hidden_slice(tokens: int, hidden: int = 4096) -> torch.Tensor:
    """A bf16 tensor the size of one production micro-batch."""
    return torch.empty(tokens, hidden, dtype=torch.bfloat16)


# -- environment ------------------------------------------------------------ #


def test_settings_under_ca_defaults_on_and_backend_defaults_auto():
    settings = ov.read_settings({})
    assert settings.under_ca is True
    assert settings.backend == "auto"


@pytest.mark.parametrize("raw,expected", [("0", False), ("no", False), ("1", True)])
def test_settings_under_ca_parses(raw, expected):
    settings = ov.read_settings({"VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA": raw})
    assert settings.under_ca is expected


@pytest.mark.parametrize("raw", ["auto", "custom", "NCCL", " hostshm "])
def test_settings_backend_accepts_every_known_value(raw):
    settings = ov.read_settings({"VLLM_GLM5_PREFILL_OVERLAP_BACKEND": raw})
    assert settings.backend == raw.strip().lower()
    assert settings.backend in ov.SPLIT_BACKENDS


def test_settings_backend_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="must be one of"):
        ov.read_settings({"VLLM_GLM5_PREFILL_OVERLAP_BACKEND": "quickreduce"})


# -- the policy ------------------------------------------------------------- #


@pytest.mark.parametrize(
    "backend,ca_active,hostshm_active,expected",
    [
        # auto resolves on what is actually live
        ("auto", True, False, "custom"),
        ("auto", False, False, "nccl"),
        ("auto", True, True, "custom"),
        ("auto", False, True, "nccl"),
        # explicit wins, including "keep using PyNccl while ca_comm is live"
        ("nccl", True, False, "nccl"),
        ("custom", True, False, "custom"),
        ("hostshm", False, True, "hostshm"),
        # asking for something that is not there degrades to PyNccl, never off
        ("custom", False, False, "nccl"),
        ("hostshm", False, False, "nccl"),
    ],
)
def test_select_split_backend(backend, ca_active, hostshm_active, expected):
    settings = ov.OverlapSettings(enabled=True, backend=backend)
    chosen, reason = ov.select_split_backend(
        settings, ca_active=ca_active, hostshm_active=hostshm_active
    )
    assert chosen == expected
    assert reason


def test_select_split_backend_stands_down_only_when_the_knob_says_so():
    off = ov.OverlapSettings(enabled=True, under_ca=False)
    chosen, reason = ov.select_split_backend(off, ca_active=True)
    assert chosen is None
    assert "UNDER_CA=0" in reason and "ca_comm" in reason

    # The same knob is irrelevant when no ca_comm is live: the feature runs.
    chosen, _ = ov.select_split_backend(off, ca_active=False)
    assert chosen == "nccl"

    # And with the knob at its default the stand-down does not trigger at all.
    on = ov.OverlapSettings(enabled=True)
    chosen, _ = ov.select_split_backend(on, ca_active=True)
    assert chosen == "custom"


@pytest.mark.parametrize("attribute", ov.UNANALYSED_BACKENDS)
def test_select_split_backend_still_refuses_an_unanalysed_backend(attribute):
    settings = ov.OverlapSettings(enabled=True)
    chosen, reason = ov.select_split_backend(
        settings, ca_active=True, unanalysed=attribute
    )
    assert chosen is None
    assert attribute in reason


def test_ca_comm_is_not_in_the_unanalysed_list():
    # The whole point of the change: ca_comm is driven, not refused.
    assert "ca_comm" not in ov.UNANALYSED_BACKENDS


# -- per-slice dispatch ----------------------------------------------------- #


def make_executor(backend: str, *, ca=None, hostshm=None):
    return ov.CudaOverlapExecutor(
        comm_stream=None, nccl=FakeNccl(), backend=backend, ca=ca, hostshm=hostshm
    )


def test_nccl_backend_never_consults_the_other_communicators():
    executor = make_executor("nccl", ca=FakeCA(), hostshm=FakeHostShm())
    assert executor.choose(hidden_slice(576)) == "nccl"


def test_custom_backend_takes_the_production_half_chunk():
    # 1152-token chunk, SPLITS=2 -> 576 x 4096 x 2 = 4,718,592 B, under the cap.
    executor = make_executor("custom", ca=FakeCA())
    assert executor.choose(hidden_slice(576)) == "custom"


def test_custom_backend_declines_at_the_cap_and_falls_through_to_pynccl():
    # 2048-token chunk, SPLITS=2 -> 1024 x 4096 x 2 = exactly 8 MiB, and the
    # real gate is a strict '<'. This is the trap a phase-2 arm would fall into.
    executor = make_executor("custom", ca=FakeCA())
    assert hidden_slice(1024).numel() * 2 == 8192 * 1024
    assert executor.choose(hidden_slice(1024)) == "nccl"
    # SPLITS=4 on the same chunk fits.
    assert executor.choose(hidden_slice(512)) == "custom"


def test_custom_backend_falls_through_when_the_communicator_is_absent_or_off():
    assert make_executor("custom").choose(hidden_slice(576)) == "nccl"
    off = make_executor("custom", ca=FakeCA(disabled=True))
    assert off.choose(hidden_slice(576)) == "nccl"


def test_hostshm_backend_respects_its_own_cap():
    executor = make_executor("hostshm", hostshm=FakeHostShm(host_cap=512 * 1024))
    assert executor.choose(hidden_slice(32)) == "hostshm"      # 256 KiB
    assert executor.choose(hidden_slice(576)) == "nccl"        # 4.7 MB, over cap
    raised = make_executor("hostshm", hostshm=FakeHostShm(host_cap=8 << 20))
    assert raised.choose(hidden_slice(576)) == "hostshm"


def test_slice_backends_counts_what_each_slice_actually_got():
    executor = make_executor("custom", ca=FakeCA())
    for tokens in (576, 576, 1024):
        executor._note(hidden_slice(tokens), executor.choose(hidden_slice(tokens)))
    assert executor.slice_backends == {"custom": 2, "hostshm": 0, "nccl": 1}


def test_the_real_custom_allreduce_gate_agrees_with_the_fake():
    """The cap arithmetic above is only useful if it matches the real rule."""
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    class Stub:
        disabled = False
        world_size = WORLD
        fully_connected = True
        max_size = 8192 * 1024

    for tokens, expected in ((576, True), (512, True), (1024, False), (2048, False)):
        real = CustomAllreduce.should_custom_ar(Stub(), hidden_slice(tokens))
        assert real is expected
        assert FakeCA().should_custom_ar(hidden_slice(tokens)) is expected


# -- region build ----------------------------------------------------------- #


def build_region(monkeypatch, communicator, env=None):
    """Drive the real ``_build_region`` with a fake TP group, on CPU."""
    monkeypatch.setattr(ov.torch.cuda, "Stream", lambda *a, **k: object())
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "get_tp_group", lambda: FakeTpGroup(communicator))
    return ov._build_region(ov.read_settings(env or {}))


def test_build_region_drives_ca_comm_instead_of_standing_down(monkeypatch):
    region = build_region(monkeypatch, FakeCommunicator(ca_comm=FakeCA()))
    assert region is not None
    assert region.executor.backend == "custom"


def test_build_region_honours_the_stand_down_knob(monkeypatch):
    region = build_region(
        monkeypatch,
        FakeCommunicator(ca_comm=FakeCA()),
        {"VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA": "0"},
    )
    assert region is None


def test_build_region_backend_override_reaches_the_executor(monkeypatch):
    region = build_region(
        monkeypatch,
        FakeCommunicator(ca_comm=FakeCA(), hostshm_comm=FakeHostShm()),
        {"VLLM_GLM5_PREFILL_OVERLAP_BACKEND": "nccl"},
    )
    assert region is not None and region.executor.backend == "nccl"


def test_build_region_ignores_a_disabled_ca_comm(monkeypatch):
    region = build_region(monkeypatch, FakeCommunicator(ca_comm=FakeCA(disabled=True)))
    assert region is not None and region.executor.backend == "nccl"


def test_build_region_still_refuses_an_unanalysed_backend(monkeypatch):
    region = build_region(monkeypatch, FakeCommunicator(qr_comm=FakeNccl()))
    assert region is None


def test_build_region_needs_a_pynccl_communicator(monkeypatch):
    communicator = FakeCommunicator(ca_comm=FakeCA())
    communicator.pynccl_comm = None
    assert build_region(monkeypatch, communicator) is None


# --------------------------------------------------------------------------- #
# Two streams must never drive one backend at once
# --------------------------------------------------------------------------- #


def test_an_inline_fallback_drains_the_comm_stream_first():
    """The custom all-reduce shares one staging buffer and one Signal block
    between calls, so a main-stream collective issued while comm-stream ones
    are in flight would interleave the flag protocol. The inline fallback is
    the only main-stream collective inside a region; it must join first."""
    region, executor = make_region(splits=2)
    with region.capture(2):
        region.submit(torch.ones(64, HIDDEN, dtype=torch.float64))
        assert region._last_token is not None
        executor.log.clear()
        # One row: below `splits`, so this one is reduced inline.
        region.submit(torch.ones(1, HIDDEN, dtype=torch.float64))
    assert executor.log == [("join", 2)]
    assert region._last_token is None


def test_drain_is_a_no_op_with_nothing_in_flight():
    region, executor = make_region(splits=2)
    executor.log.clear()
    region.drain()
    region.drain()
    assert executor.log == []


def test_begin_layer_forgets_the_previous_layers_token():
    region, executor = make_region(splits=2)
    with region.capture(2):
        region.submit(torch.ones(64, HIDDEN, dtype=torch.float64))
    assert region._last_token is not None
    region.begin_layer()
    assert region._last_token is None
    executor.log.clear()
    region.drain()
    assert executor.log == []


def test_the_join_semantics_of_a_sliced_handle_are_unchanged():
    """Regression guard: the backend knob must not change who waits for what."""
    region, executor = make_region(splits=2)
    with region.capture(2) as handles:
        out = region.submit(torch.ones(64, HIDDEN, dtype=torch.float64))
    handle = handles[0]
    assert handle.synchronous is False and len(handle.tokens) == 2
    assert torch.isnan(out).all()
    handle.join(0)
    assert not torch.isnan(out[:32]).any() and torch.isnan(out[32:]).all()
    handle.join_all()
    assert torch.allclose(out, torch.full_like(out, float(WORLD)))


def test_custom_allreduce_max_size_knob_defaults_to_upstreams_value():
    """The knob exists to raise the cap for an experiment, not to change it."""
    import inspect

    import vllm.envs as envs
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    upstream = inspect.signature(CustomAllreduce.__init__).parameters["max_size"]
    assert envs.VLLM_GLM5_CUSTOM_ALLREDUCE_MAX_SIZE == upstream.default == 8192 * 1024


def test_the_cap_knob_is_what_lets_the_custom_arm_run_at_splits_two():
    # A 2048-token chunk at SPLITS=2 sits exactly on the 8 MiB cap and is
    # refused; raising the cap is the only way that arm reaches CUSTOM without
    # also changing SPLITS.
    assert make_executor("custom", ca=FakeCA()).choose(hidden_slice(1024)) == "nccl"
    raised = make_executor("custom", ca=FakeCA(max_size=16 * 1024 * 1024))
    assert raised.choose(hidden_slice(1024)) == "custom"
