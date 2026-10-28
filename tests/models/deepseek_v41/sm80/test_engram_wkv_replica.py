# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram wkv replicated for prefill steps
(VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL), on CPU.

Four TP ranks of a real ColumnParallelLinear load one checkpoint weight
through the layer's weight loader. A stand-in method processes each rank's
shard into another runtime layout (the weight transposed and packed into
int32, the E8M0 scales transposed, as Marlin's processing changes both) and
computes in float64. The ranks process their weights in threads joined by an
in-memory all-gather. The real Marlin GEMM is
test_engram_wkv_replica_gpu.py.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.linear as linear
from tests.models.deepseek_v41.sm80.construct import set_groups
from vllm.model_executor.layers.linear import ColumnParallelLinear, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm_sm80.models.deepseek_v41 import sm80_engram_wkv as W

BF16 = torch.bfloat16
TP, K, N = 4, 128, 4 * 48
MIN_ROWS = 49  # 8 sequences x (1 + 5 drafts) + 1; the largest capture is 40


class _PackedMethod(LinearMethodBase):
    """Processes a shard into a packed layout and applies it in float64,
    recording which layer (shard) each call reads."""

    def __init__(self):
        self.calls = []

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        n = layer.output_size_per_partition
        replace_parameter(layer, "weight", layer.weight.data.t().contiguous().view(torch.int32))
        replace_parameter(
            layer,
            "weight_scale",
            layer.weight_scale.data.t().contiguous().view(torch.float8_e8m0fnu),
        )
        assert layer.weight.shape == (K, n // 2)

    def apply(self, layer, x, bias=None):
        assert bias is None
        self.calls.append(layer)
        weight = layer.weight.view(BF16).t().double()
        scale = layer.weight_scale.view(torch.uint8).t().double() - 127
        weight = weight * torch.exp2(scale).repeat_interleave(32, dim=1)
        out = (x.double() @ weight.t()).to(x.dtype)
        assert out.shape[-1] == layer.output_size_per_partition
        return out


def _checkpoint():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(N, K, generator=generator).to(BF16)
    scale = torch.randint(120, 134, (N, K // 32), generator=generator).to(torch.uint8)
    return weight, scale


def _rank(monkeypatch, rank, adopt):
    """Rank ``rank``'s wkv with the checkpoint loaded through its own loader
    (its TP slice of the output rows)."""
    set_groups(monkeypatch, TP, rank)
    layer = ColumnParallelLinear(
        K, N, bias=False, gather_output=True, return_bias=False,
        params_dtype=BF16, prefix="layers.1.engram.wkv",
    )  # fmt: skip
    scale = nn.Parameter(
        torch.empty(N // TP, K // 32, dtype=torch.uint8), requires_grad=False
    )
    set_weight_attrs(scale, {"output_dim": 0, "weight_loader": layer.weight_loader})
    layer.register_parameter("weight_scale", scale)
    layer.quant_method = _PackedMethod()
    weight, scale = _checkpoint()
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale.weight_loader(layer.weight_scale, scale)
    rows = slice(rank * N // TP, (rank + 1) * N // TP)
    assert torch.equal(layer.weight.data, weight[rows])
    assert torch.equal(layer.weight_scale.data, scale[rows])
    if adopt:
        W.ReplicatedPrefillWkv.adopt(layer, MIN_ROWS)
    return layer


class _Collective:
    """``W._gather_tp`` and ``W.get_tp_group`` for TP ranks run as threads."""

    def __init__(self):
        self.barrier = threading.Barrier(TP)
        self.slots = [None] * TP
        self.local = threading.local()
        self.gathers = 0

    def group(self):
        return SimpleNamespace(world_size=TP, rank_in_group=self.local.rank)

    def gather(self, tensor):
        self.slots[self.local.rank] = tensor.clone()
        self.barrier.wait()
        out = torch.stack(self.slots)
        self.gathers += 1
        self.barrier.wait()
        return out

    def run(self, layers):
        errors = []

        def process(rank):
            self.local.rank = rank
            try:
                layers[rank].quant_method.process_weights_after_loading(layers[rank])
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                self.barrier.abort()

        threads = [threading.Thread(target=process, args=(r,)) for r in range(TP)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]


@pytest.fixture
def ranks(monkeypatch):
    """The four adopted ranks after processing, and four plain ranks processed
    by the method alone (today's shards)."""
    plain = [_rank(monkeypatch, r, adopt=False) for r in range(TP)]
    for layer in plain:
        layer.quant_method.process_weights_after_loading(layer)
    adopted = [_rank(monkeypatch, r, adopt=True) for r in range(TP)]
    collective = _Collective()
    monkeypatch.setattr(W, "_gather_tp", collective.gather)
    monkeypatch.setattr(W, "get_tp_group", collective.group)
    collective.run(adopted)
    # Two tensors per rank (weight, scales), one gather each.
    assert collective.gathers == 2 * TP
    return SimpleNamespace(plain=plain, adopted=adopted)


def _bytes(tensor):
    return tensor.contiguous().view(-1).view(torch.uint8)


def test_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    name = "VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True


@pytest.mark.parametrize(
    "seqs,drafts,captured,want",
    [
        (8, 5, 40, 49),  # the DeepSeek-V4.1 serving shape: decode 48 > capture 40
        (8, None, 40, 41),
        (8, 5, 512, 513),
        (256, 5, 512, 1537),
        (4, 3, None, 17),
    ],
)
def test_min_rows_exceed_decode_and_captures(seqs, drafts, captured, want):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
        speculative_config=None
        if drafts is None
        else SimpleNamespace(num_speculative_tokens=drafts),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=captured),
    )
    assert W.prefill_min_rows(config) == want


def test_replica_holds_every_rank_shard_without_a_second_copy(ranks):
    for rank, layer in enumerate(ranks.adopted):
        shards = layer.wkv_shards
        assert len(shards) == TP
        for source, shard in enumerate(shards):
            ref = ranks.plain[source]
            # The processed bytes of rank ``source``, in its runtime layout.
            for name in ("weight", "weight_scale"):
                got, want = getattr(shard, name), getattr(ref, name)
                assert got.dtype == want.dtype and got.shape == want.shape
                assert torch.equal(_bytes(got), _bytes(want)), (rank, source, name)
            assert shard.output_size_per_partition == N // TP
            assert shard.input_size_per_partition == K
            assert shard.tp_rank == rank  # the layer's attributes, unchanged
        # The layer's own parameters are its slot of the replica.
        for name in ("weight", "weight_scale"):
            own = getattr(layer, name)
            assert isinstance(own, nn.Parameter)
            assert own.data_ptr() == getattr(shards[rank], name).data_ptr()
            storage = own.untyped_storage()
            assert storage.nbytes() == TP * own.nbytes
            assert all(
                getattr(s, name).untyped_storage().data_ptr() == storage.data_ptr()
                for s in shards
            )
            # The weight loader survives for a later reload.
            assert own.weight_loader.__self__ is layer
        # Outside the module tree: loading and state dicts see one shard.
        assert {n for n, _ in layer.named_parameters()} == {"weight", "weight_scale"}
        assert not list(layer.children())


@pytest.fixture
def gathers(monkeypatch, ranks):
    """Today's all-gather of the sharded outputs, recorded; every torch
    collective fails."""
    calls = []

    def all_gather(output, dim=-1):
        assert dim == -1
        rank = output_rank[0]
        calls.append(rank)
        x = inputs[0]
        parts = [p.quant_method.apply(p, x) for p in ranks.plain]
        assert torch.equal(parts[rank], output)
        return torch.cat(parts, dim=-1)

    def no_collective(*args, **kwargs):
        raise AssertionError("collective on the replicated path")

    inputs, output_rank = [None], [None]
    monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", all_gather)
    for name in ("all_gather", "all_gather_into_tensor", "all_reduce", "broadcast"):
        monkeypatch.setattr(torch.distributed, name, no_collective)
    return SimpleNamespace(calls=calls, inputs=inputs, rank=output_rank)


def _project(layer, rows, gathers, rank):
    generator = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, K, generator=generator).to(BF16)
    gathers.inputs[0], gathers.rank[0] = x, rank
    layer.quant_method.inner.calls.clear()
    return x, layer(x)


@pytest.mark.parametrize("rows", [1, 6, 40, 48, 49, 64, 300])
def test_prefill_rows_skip_the_all_gather(ranks, gathers, rows):
    for rank, layer in enumerate(ranks.adopted):
        gathers.calls.clear()
        x, out = _project(layer, rows, gathers, rank)
        want = torch.cat([p.quant_method.apply(p, x) for p in ranks.plain], dim=-1)
        assert out.shape == (rows, N) and out.dtype == BF16
        assert torch.equal(out.view(torch.int16), want.view(torch.int16))
        calls = layer.quant_method.inner.calls
        if rows < MIN_ROWS:
            # Decode and every capture size: this rank's shard, then the gather.
            assert gathers.calls == [rank]
            assert calls == [layer]
        else:
            # Prefill: every shard here, in rank order, and no gather.
            assert gathers.calls == []
            assert calls == layer.wkv_shards
            assert all(c is not layer for c in calls)


def test_off_arm_keeps_the_sharded_path(ranks, gathers):
    layer = ranks.adopted[2]
    layer.min_rows = None
    _project(layer, 300, gathers, 2)
    assert gathers.calls == [2]
    assert layer.quant_method.inner.calls == [layer]


def test_unprocessed_replica_refuses_prefill(monkeypatch):
    layer = _rank(monkeypatch, 1, adopt=True)
    with pytest.raises(RuntimeError, match="no replica"):
        layer(torch.zeros(MIN_ROWS, K, dtype=BF16))


def test_method_wrapper_is_the_layer_method(monkeypatch):
    layer = _rank(monkeypatch, 0, adopt=True)
    method = layer.quant_method
    assert isinstance(method, QuantizeMethodBase)
    assert isinstance(method.inner, _PackedMethod)
    assert method.requires_device_loading is method.inner.requires_device_loading
    assert method.supports_pre_processed_weights is False
    with pytest.raises(TypeError):
        method.create_weights(layer)


def test_replica_must_match_the_layer(monkeypatch):
    layer = _rank(monkeypatch, 0, adopt=True)
    layer.quant_method.inner.process_weights_after_loading(layer)
    stacked = {"weight": torch.stack([layer.weight.data] * TP)}
    with pytest.raises(ValueError, match="does not match"):
        W.install_replica(layer, stacked)
    stacked["weight_scale"] = torch.stack([layer.weight_scale.data] * (TP - 1))
    with pytest.raises(ValueError, match="is not"):
        W.install_replica(layer, stacked)


def test_rank_must_be_the_tp_rank(monkeypatch):
    layer = _rank(monkeypatch, 1, adopt=True)
    monkeypatch.setattr(
        W, "get_tp_group", lambda: SimpleNamespace(world_size=TP, rank_in_group=2)
    )
    with pytest.raises(ValueError, match="is not TP rank"):
        layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"disable_tp": True}, "TP-sharded"),
        ({"gather_output": False}, "TP-sharded"),
        ({"bias": True}, "without bias"),
        ({"return_bias": True}, "without bias"),
    ],
)
def test_adopt_refuses_other_layers(monkeypatch, kwargs, match):
    set_groups(monkeypatch, TP, 0)
    args = dict(bias=False, gather_output=True, return_bias=False, params_dtype=BF16)
    layer = ColumnParallelLinear(K, N, **(args | kwargs))
    with pytest.raises(ValueError, match=match):
        W.ReplicatedPrefillWkv.adopt(layer, MIN_ROWS)


def _model(engram, batched=2048):
    layer = SimpleNamespace(engram=engram)
    return SimpleNamespace(
        layers=[SimpleNamespace(engram=None), layer],
        start_layer=0,
        end_layer=2,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=8, max_num_batched_tokens=batched
            ),
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=40),
        ),
    )


@pytest.fixture
def no_capture(monkeypatch):
    monkeypatch.delenv("VLLM_SM80_CAPTURE", raising=False)


def _engram(monkeypatch):
    set_groups(monkeypatch, TP, 3)
    wkv = ColumnParallelLinear(
        K, N, bias=False, gather_output=True, return_bias=False, params_dtype=BF16
    )
    return SimpleNamespace(
        use_sequence_parallel=False, _batch=None,
        embed_tokens=SimpleNamespace(dp_size=1), wkv=wkv,
    )  # fmt: skip


def test_adopt_every_local_engram_wkv(monkeypatch, no_capture):
    engram = _engram(monkeypatch)
    wkv = engram.wkv
    W.adopt_replicated_wkv(_model(engram))
    assert type(wkv) is W.ReplicatedPrefillWkv
    assert wkv.min_rows == MIN_ROWS and wkv.wkv_shards is None


def test_adopt_refuses_a_replica_no_step_reaches(monkeypatch, no_capture):
    """With a step budget below min_rows the replica would cost memory and
    never run."""
    engram = _engram(monkeypatch)
    with pytest.raises(ValueError, match="max_num_batched_tokens 48"):
        W.adopt_replicated_wkv(_model(engram, batched=MIN_ROWS - 1))
    assert type(engram.wkv) is ColumnParallelLinear
    W.adopt_replicated_wkv(_model(engram, batched=MIN_ROWS))
    assert type(engram.wkv) is W.ReplicatedPrefillWkv


def test_adopt_refuses_a_hybrid_capture(monkeypatch, no_capture, tmp_path):
    """Every shard copy carries this rank's capture facts: a hybrid capture
    would record four GEMMs as this rank's."""
    engram = _engram(monkeypatch)
    spec = tmp_path / "capture.toml"
    spec.write_text('dir = "capture"\nfamilies = ["attn", "hybrid"]\n')
    monkeypatch.setenv("VLLM_SM80_CAPTURE", str(spec))
    with pytest.raises(ValueError, match="hybrid capture"):
        W.adopt_replicated_wkv(_model(engram))
    assert type(engram.wkv) is ColumnParallelLinear
    spec.write_text('dir = "capture"\nfamilies = ["attn"]\n')
    W.adopt_replicated_wkv(_model(engram))
    assert type(engram.wkv) is W.ReplicatedPrefillWkv


@pytest.mark.parametrize(
    "change,match",
    [
        ({"use_sequence_parallel": True}, "unbatched TP"),
        ({"_batch": object()}, "unbatched TP"),
        ({"embed_tokens": SimpleNamespace(dp_size=2)}, "TP-only"),
    ],
)
def test_adopt_refuses_other_engram_layouts(change, match):
    engram = SimpleNamespace(
        use_sequence_parallel=False, _batch=None,
        embed_tokens=SimpleNamespace(dp_size=1), wkv=None,
    )  # fmt: skip
    for key, value in change.items():
        setattr(engram, key, value)
    with pytest.raises(ValueError, match=match):
        W.adopt_replicated_wkv(_model(engram))


@pytest.mark.parametrize("on", [False, True])
def test_backbone_adoption_follows_the_switch(monkeypatch, on):
    from vllm.models.deepseek_v41.nvidia import model as upstream_model
    from vllm_sm80.models.deepseek_v41 import sm80_model

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL", str(int(on)))
    adopted = []
    monkeypatch.setattr(W, "adopt_replicated_wkv", adopted.append)
    model = object.__new__(upstream_model.DeepseekV4Model)
    nn.Module.__init__(model)
    model.use_sequence_parallel = model.fuse_mhc_all_reduce = False
    model.layers, model.start_layer, model.end_layer = [], 0, 0
    model.config = None
    model.decoder_replay_layers = None
    sm80_model.DeepseekV4Model.adopt(model)
    assert adopted == ([model] if on else [])


def test_dev_ab_switch_sets_every_replica(monkeypatch, ranks):
    import vllm.envs as envs
    from vllm_sm80 import dev_ab

    name = "VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL"
    monkeypatch.setenv(name, "1")
    monkeypatch.setattr(dev_ab, "_STATE", dev_ab._ArmState())
    switch = dev_ab.SWITCHES["dsv41_engram_wkv_replicated_prefill"]
    assert switch.env == name
    # Captures never take the replica; both arms give the gathered bytes.
    assert not switch.graph_bound and switch.output_identical
    runner = SimpleNamespace(model=nn.ModuleList(ranks.adopted[:2]), speculator=None)
    assert switch.unavailable(runner) is None and switch.current(runner) is True
    switch.apply(runner, False)
    assert [m.min_rows for m in ranks.adopted[:2]] == [None, None]
    assert switch.current(runner) is False and envs.VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL is False
    switch.apply(runner, True)
    assert [m.min_rows for m in ranks.adopted[:2]] == [MIN_ROWS, MIN_ROWS]
    assert switch.current(runner) is True and envs.VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL is True
    plain = SimpleNamespace(model=nn.ModuleList(ranks.plain), speculator=None)
    assert "boot with" in switch.unavailable(plain)
