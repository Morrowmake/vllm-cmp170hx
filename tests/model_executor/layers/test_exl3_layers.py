# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of EXL3 sharding, expert placement and scratch sizing.

Geometries are the Qwen3.8-Flash-Next ones; the checkpoint files are sparse
synthetic headers, so no weights are read.
"""

import json
import math
import struct
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import linear as linear_module
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.sm80.kernels.exl3 import ops as exl3_ops
from vllm.sm80.kernels.exl3 import quant as exl3_module

_BYTES = {"I16": 2, "F16": 2, "I32": 4}
_MUL1 = 0x83DCD12D
HIDDEN = 2560
VOCAB = 248320

# prefix -> (padded_in, padded_out, bits_x2)
DENSE = {
    "model.layers.0.linear_attn.in_proj_qkv": (2560, 10240, 14),
    "model.layers.0.linear_attn.in_proj_z": (2560, 6144, 14),
    "model.layers.0.linear_attn.out_proj": (6144, 2560, 14),
    "model.layers.3.self_attn.q_proj": (2560, 12288, 14),
    "model.layers.3.self_attn.k_proj": (2560, 512, 14),
    "model.layers.3.self_attn.v_proj": (2560, 512, 14),
    "model.layers.3.self_attn.o_proj": (6144, 2560, 14),
    "model.layers.0.mlp.shared_expert.gate_proj": (2560, 640, 14),
    "model.layers.0.mlp.shared_expert.down_proj": (640, 2560, 14),
    "lm_head": (2560, VOCAB, 12),
}
QSA_INDEX = 640  # (4 indexer query heads + 1 key head) x 128
for _qsa in ("model.language_model.layers.3.self_attn", "mtp.layers.0.self_attn"):
    DENSE.update(
        {
            f"{_qsa}.q_proj": (2560, 12288, 14),
            f"{_qsa}.k_proj": (2560, 512, 14),
            f"{_qsa}.v_proj": (2560, 512, 14),
            f"{_qsa}.indexer.index_qk_proj": (2560, QSA_INDEX, 14),
        }
    )
EXPERTS, EXPERT_HIDDEN, EXPERT_INTERMEDIATE = 8, 256, 128


def _matrix_entries(prefix, padded_in, padded_out, bits_x2):
    return {
        f"{prefix}.trellis": (
            "I16",
            [padded_in // 16, padded_out // 16, 8 * bits_x2],
            None,
        ),
        f"{prefix}.suh": ("F16", [padded_in], None),
        f"{prefix}.svh": ("F16", [padded_out], None),
        f"{prefix}.mul1": ("I32", [], struct.pack("<I", _MUL1)),
    }


def _write_sparse(path, tensors):
    """Write a safetensors file whose unspecified payloads stay sparse."""
    header, offset, chunks = {}, 0, []
    for name, (dtype, shape, data) in tensors.items():
        size = math.prod(shape) * _BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        if data is not None:
            chunks.append((offset, data))
        offset += size
    encoded = json.dumps(header).encode()
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        base = file.tell()
        for start, data in chunks:
            file.seek(base + start)
            file.write(data)
        file.truncate(base + offset)


@pytest.fixture
def exl3_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(8, 0),
    )
    exl3_module._require_sm80.cache_clear()
    tensors = {}
    for prefix, geometry in DENSE.items():
        tensors.update(_matrix_entries(prefix, *geometry))
    for expert in range(EXPERTS):
        for role, k, n in (
            ("gate_proj", EXPERT_HIDDEN, EXPERT_INTERMEDIATE),
            ("up_proj", EXPERT_HIDDEN, EXPERT_INTERMEDIATE),
            ("down_proj", EXPERT_INTERMEDIATE, EXPERT_HIDDEN),
        ):
            tensors.update(
                _matrix_entries(f"model.layers.0.mlp.experts.{expert}.{role}", k, n, 10)
            )
    _write_sparse(tmp_path / "model.safetensors", tensors)
    config = exl3_module.Exl3Config()
    config.maybe_update_config(str(tmp_path))
    config.packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    yield config
    exl3_module._require_sm80.cache_clear()


def _tp(monkeypatch, rank, size):
    for module in (linear_module, vocab_parallel_embedding):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: size
        )


def _slices(layer):
    return [
        (w.matrix.prefix, w.input_slice, w.output_slices, tuple(w.trellis.shape))
        for w in layer.exl3_weights
    ]


def _covered(parts, length):
    """Count how often each element of [0, length) is selected."""
    counts = [0] * length
    for part in parts:
        for index in range(part.start, part.stop):
            counts[index] += 1
    return counts


@pytest.mark.parametrize("size", [1, 2])
def test_gdn_qkvz_shards_q_k_v_groups_and_z(exl3_config, monkeypatch, size):
    shards = []
    for rank in range(size):
        _tp(monkeypatch, rank, size)
        layer = MergedColumnParallelLinear(
            HIDDEN,
            [2048, 2048, 6144, 6144],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=exl3_config,
            prefix="model.layers.0.linear_attn.in_proj_qkvz",
        )
        q, k, v = 2048 // size, 2048 // size, 6144 // size
        assert _slices(layer) == [
            (
                "model.layers.0.linear_attn.in_proj_qkv",
                slice(0, HIDDEN),
                [
                    slice(rank * q, (rank + 1) * q),
                    slice(2048 + rank * k, 2048 + (rank + 1) * k),
                    slice(4096 + rank * v, 4096 + (rank + 1) * v),
                ],
                (160, (q + k + v) // 16, 112),
            ),
            (
                "model.layers.0.linear_attn.in_proj_z",
                slice(0, HIDDEN),
                [slice(rank * v, (rank + 1) * v)],
                (160, v // 16, 112),
            ),
        ]
        assert [w.logical_out for w in layer.exl3_weights] == [q + k + v, v]
        shards.append(layer)
    for index, width in ((0, 10240), (1, 6144)):
        parts = [p for layer in shards for p in layer.exl3_weights[index].output_slices]
        assert _covered(parts, width) == [1] * width


@pytest.mark.parametrize("size", [2, 4])
def test_qkv_shards_kv_heads_and_their_replicas(exl3_config, monkeypatch, size):
    replicas = max(1, size // 2)
    kv_parts, q_parts = [], []
    for rank in range(size):
        _tp(monkeypatch, rank, size)
        layer = QKVParallelLinear(
            HIDDEN,
            256,
            48,
            2,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=exl3_config,
            prefix="model.layers.3.self_attn.qkv_proj",
        )
        assert layer.num_kv_head_replicas == replicas
        q_width = 12288 // size
        kv_rank, kv_width = rank // replicas, 512 // (size // replicas)
        assert _slices(layer) == [
            (
                "model.layers.3.self_attn.q_proj",
                slice(0, HIDDEN),
                [slice(rank * q_width, (rank + 1) * q_width)],
                (160, q_width // 16, 112),
            ),
            (
                "model.layers.3.self_attn.k_proj",
                slice(0, HIDDEN),
                [slice(kv_rank * kv_width, (kv_rank + 1) * kv_width)],
                (160, kv_width // 16, 112),
            ),
            (
                "model.layers.3.self_attn.v_proj",
                slice(0, HIDDEN),
                [slice(kv_rank * kv_width, (kv_rank + 1) * kv_width)],
                (160, kv_width // 16, 112),
            ),
        ]
        q_parts += layer.exl3_weights[0].output_slices
        kv_parts += layer.exl3_weights[1].output_slices
    assert _covered(q_parts, 12288) == [1] * 12288
    # Each KV head is held by exactly `replicas` ranks.
    assert _covered(kv_parts, 512) == [replicas] * 512


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize(
    "owner,prefix",
    [
        ("Qwen4ExpForConditionalGeneration", "language_model.model.layers.3.self_attn"),
        ("Qwen4ExpMTP", "mtp.layers.0.self_attn"),
    ],
)
def test_qsa_fused_qkv_indexer_shards_heads_and_replicates_the_indexer(
    exl3_config, monkeypatch, owner, prefix, size
):
    import vllm.models.qwen4_exp.nvidia.model as qwen_model
    from vllm.models.qwen4_exp.nvidia import mtp as qwen_mtp
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAQKVIndexerLinear

    # The target's packed mapping lists the indexer, the MTP draft's does not;
    # both build the same layers.
    model_class = getattr(qwen_model, owner, None) or getattr(qwen_mtp, owner)
    monkeypatch.setattr(exl3_config, "is_qwen", True)
    monkeypatch.setattr(
        exl3_config, "packed_modules_mapping", model_class.packed_modules_mapping
    )
    source = exl3_config.source_prefix(prefix)
    replicas = max(1, size // 2)
    q_parts, kv_parts = [], []
    for rank in range(size):
        _tp(monkeypatch, rank, size)
        # Qwen4ExpQSAAttention builds this Q/K/V layer, then fuses the indexer.
        qkv = QKVParallelLinear(
            HIDDEN,
            256,
            48,
            2,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=exl3_config,
            prefix=f"{prefix}.qkv_proj",
        )
        assert [w.matrix.prefix for w in qkv.exl3_weights] == [
            f"{source}.{name}_proj" for name in "qkv"
        ]
        layer = Qwen4ExpQSAQKVIndexerLinear(qkv, QSA_INDEX, exl3_config)
        q_width = 12288 // size
        kv_rank, kv_width = rank // replicas, 512 // (size // replicas)
        kv = [slice(kv_rank * kv_width, (kv_rank + 1) * kv_width)]
        assert _slices(layer) == [
            (
                f"{source}.q_proj",
                slice(0, HIDDEN),
                [slice(rank * q_width, (rank + 1) * q_width)],
                (160, q_width // 16, 112),
            ),
            (f"{source}.k_proj", slice(0, HIDDEN), kv, (160, kv_width // 16, 112)),
            (f"{source}.v_proj", slice(0, HIDDEN), kv, (160, kv_width // 16, 112)),
            (
                f"{source}.indexer.index_qk_proj",
                slice(0, HIDDEN),
                [slice(0, QSA_INDEX)],
                (160, QSA_INDEX // 16, 112),
            ),
        ]
        # Output columns line up with the layer's split of Q/gate+K+V | indexer.
        assert [w.logical_out for w in layer.exl3_weights] == [
            q_width,
            kv_width,
            kv_width,
            QSA_INDEX,
        ]
        assert sum(w.logical_out for w in layer.exl3_weights) == (
            layer.output_size_per_partition
        )
        q_parts += layer.exl3_weights[0].output_slices
        kv_parts += layer.exl3_weights[1].output_slices
    assert _covered(q_parts, 12288) == [1] * 12288
    assert _covered(kv_parts, 512) == [replicas] * 512


def test_row_parallel_shards_inputs_at_h128_blocks(exl3_config, monkeypatch):
    parts = []
    for rank in range(2):
        _tp(monkeypatch, rank, 2)
        layer = RowParallelLinear(
            6144,
            HIDDEN,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=exl3_config,
            prefix="model.layers.3.self_attn.o_proj",
        )
        assert _slices(layer) == [
            (
                "model.layers.3.self_attn.o_proj",
                slice(rank * 3072, (rank + 1) * 3072),
                [slice(0, HIDDEN)],
                (192, 160, 112),
            )
        ]
        assert layer.exl3_weights[0].logical_in == 3072
        parts.append(layer.exl3_weights[0].input_slice)
    assert _covered(parts, 6144) == [1] * 6144


def test_shared_expert_down_cannot_split_inside_h128(exl3_config, monkeypatch):
    # 640 / 2 = 320 is not a multiple of 128, so the model replicates it.
    _tp(monkeypatch, 0, 2)
    with pytest.raises(ValueError, match="H128"):
        RowParallelLinear(
            640,
            HIDDEN,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=exl3_config,
            prefix="model.layers.0.mlp.shared_expert.down_proj",
        )
    layer = ReplicatedLinear(
        640,
        HIDDEN,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=exl3_config,
        prefix="model.layers.0.mlp.shared_expert.down_proj",
    )
    assert _slices(layer)[0][1:3] == (slice(0, 640), [slice(0, HIDDEN)])


def test_lm_head_shards_vocabulary(exl3_config, monkeypatch):
    parts = []
    for rank in range(2):
        _tp(monkeypatch, rank, 2)
        head = ParallelLMHead(
            VOCAB,
            HIDDEN,
            params_dtype=torch.bfloat16,
            quant_config=exl3_config,
            prefix="lm_head",
        )
        (weight,) = head.exl3_weights
        assert weight.input_slice == slice(0, HIDDEN)
        assert weight.output_slices == [slice(rank * 124160, (rank + 1) * 124160)]
        assert weight.logical_out == 124160
        assert tuple(weight.trellis.shape) == (160, 7760, 96)
        parts += weight.output_slices
    assert _covered(parts, VOCAB) == [1] * VOCAB


def test_method_selection_refuses_other_gpus(exl3_config, monkeypatch):
    _tp(monkeypatch, 0, 1)
    layer = ReplicatedLinear(
        2560,
        640,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=exl3_config,
        prefix="model.layers.0.mlp.shared_expert.gate_proj",
    )
    assert isinstance(layer.quant_method, exl3_module.Exl3LinearMethod)
    for major, minor in ((8, 6), (9, 0)):
        monkeypatch.setattr(
            current_platform,
            "get_device_capability",
            lambda device_id=0, c=(major, minor): DeviceCapability(*c),
        )
        exl3_module._require_sm80.cache_clear()
        with pytest.raises(
            ValueError, match=f"SM 8.0 only; this GPU is SM {major}.{minor}"
        ):
            exl3_config.get_quant_method(
                layer, "model.layers.0.mlp.shared_expert.gate_proj"
            )


def _moe_config(ep_rank=0, ep_size=2, dp_size=1, sp_size=1, max_tokens=3460):
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=ep_rank,
        sp_size=sp_size,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=EXPERTS,
        experts_per_token=2,
        hidden_dim=EXPERT_HIDDEN,
        intermediate_size=EXPERT_INTERMEDIATE,
        num_local_experts=EXPERTS // ep_size,
        num_logical_experts=EXPERTS,
        moe_parallel_config=parallel,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        max_num_tokens=max_tokens,
    )


class _RoutedExperts(torch.nn.Module):
    """The two RoutedExperts members create_weights reads, from vLLM's map."""

    def __init__(self, expert_map):
        super().__init__()
        self.global_num_experts = EXPERTS
        self.apply_router_weight_on_input = False
        self.expert_map = expert_map

    def _map_global_expert_id_to_local_expert_id(self, expert):
        return int(self.expert_map[expert])


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_moe_assigns_whole_experts_from_the_ep_map(exl3_config, strategy):
    prefix = "model.layers.0.mlp.experts"
    owned, skipped = [], []
    for rank in range(2):
        local, expert_map, _ = determine_expert_map(
            2, rank, EXPERTS, expert_placement_strategy=strategy
        )
        assert expert_map is not None
        method = exl3_module.Exl3MoEMethod(exl3_config, prefix, _moe_config(rank))
        layer = _RoutedExperts(expert_map)
        method.create_weights(
            layer, local, EXPERT_HIDDEN, EXPERT_INTERMEDIATE, torch.bfloat16
        )
        expected = [e for e in range(EXPERTS) if expert_map[e] >= 0]
        expected.sort(key=lambda e: int(expert_map[e]))
        assert method.global_expert_ids == expected
        for local_id, weights in enumerate(layer.exl3_experts):
            assert [w.matrix.prefix for w in weights] == [
                f"{prefix}.{expected[local_id]}.{role}"
                for role in ("gate_proj", "up_proj", "down_proj")
            ]
            assert [tuple(w.trellis.shape) for w in weights] == [
                (16, 8, 80),
                (16, 8, 80),
                (8, 16, 80),
            ]
        owned += [w.matrix.prefix for e in layer.exl3_experts for w in e]
        skipped += method.nonlocal_keys
    all_keys = {
        key
        for name, matrix in exl3_config.manifest.matrices.items()
        if ".experts." in name
        for key in matrix.keys
    }
    # Every expert is owned by exactly one rank; the other rank skips its keys.
    assert sorted(owned) == sorted(
        name for name in exl3_config.manifest.matrices if ".experts." in name
    )
    assert len(skipped) == len(all_keys) and set(skipped) == all_keys


def test_moe_scratch_rows_follow_the_dispatch_actually_used():
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    # TP2 + EP without DP: no dispatch, so each rank sees only its own rows,
    # not ep_size times as many.
    moe = _moe_config(ep_size=2)
    prepare = maybe_make_prepare_finalize(
        moe=moe, quant_config=FusedMoEQuantConfig.make(), allow_new_interface=True
    )
    assert type(prepare).__name__ == "MoEPrepareAndFinalizeNoDPEPModular"
    assert exl3_module.dispatched_rows(prepare, moe) == 3460
    # All-gather over two DP replicas.
    gather = MoEPrepareAndFinalizeNaiveDPEPModular(num_dispatchers=4)
    assert exl3_module.dispatched_rows(gather, _moe_config(ep_size=4, dp_size=2)) == (
        2 * 3460
    )
    # Sequence-parallel shares are padded to equal size before the gather.
    sequence = MoEPrepareAndFinalizeNaiveDPEPModular(
        is_sequence_parallel=True, num_dispatchers=4
    )
    moe = _moe_config(ep_size=4, dp_size=2, sp_size=2, max_tokens=3461)
    assert exl3_module.dispatched_rows(sequence, moe) == 4 * 1731


def test_linear_row_chunk_keeps_scratch_fixed():
    budget = exl3_ops.LINEAR_SCRATCH_ELEMENTS
    for padded_in, padded_out in ((2560, 124160), (2560, 10240), (128, 128)):
        chunk = exl3_ops.linear_row_chunk(padded_in, padded_out)
        assert exl3_ops.linear_workspace_size(chunk, padded_in, padded_out) <= budget
        assert exl3_ops.linear_workspace_size(chunk + 1, padded_in, padded_out) > budget
    # The TP2 LM head shard runs 66 rows per launch within 32 MiB.
    assert exl3_ops.linear_row_chunk(2560, 124160) == 66


def test_linear_reuses_one_fixed_scratch_for_every_row_count(monkeypatch):
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace

    monkeypatch.setattr(torch_utils, "aux_stream", lambda: None)
    monkeypatch.setattr(workspace, "_manager", None)
    workspace.init_workspace_manager(torch.device("cpu"))
    calls = []

    def native(
        x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, scratch
    ):
        assert scratch.numel() >= exl3_ops.linear_workspace_size(
            x.shape[0], suh.numel(), svh.numel()
        )
        calls.append((x.shape[0], scratch.data_ptr(), scratch.numel()))
        out.copy_(x[:, :1].expand(-1, logical_out))

    monkeypatch.setattr(exl3_ops, "linear_out", native)
    padded_in, padded_out = 2560, 124160
    chunk = exl3_ops.linear_row_chunk(padded_in, padded_out)
    trellis = torch.empty(0, dtype=torch.int16)
    suh = torch.empty(padded_in, dtype=torch.float16)
    svh = torch.empty(padded_out, dtype=torch.float16)
    for rows in (1, chunk, 2 * chunk + 5, 3460):
        calls.clear()
        x = torch.arange(rows, dtype=torch.bfloat16)[:, None].expand(-1, padded_in)
        out = exl3_ops._linear(
            x.contiguous(), trellis, suh, svh, 12, 2, padded_out, chunk
        )
        assert [c[0] for c in calls] == [
            min(chunk, rows - start) for start in range(0, rows, chunk)
        ]
        assert torch.equal(out[:, 0], x[:, 0])
        assert {c[1:] for c in calls} == {
            (calls[0][1], exl3_ops.LINEAR_SCRATCH_ELEMENTS)
        }
        if rows == 1:
            first = calls[0][1]
        assert calls[0][1] == first
    workspace.reset_workspace_manager()


@pytest.mark.parametrize(
    "draft_model,is_draft,needed",
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_target_manifest_outlives_the_target_until_the_draft_loads(
    draft_model, is_draft, needed
):
    from vllm.sm80.kernels.exl3.loader import manifest_still_needed

    spec = SimpleNamespace(draft_model_config=object()) if draft_model else None
    assert manifest_still_needed(
        SimpleNamespace(
            speculative_config=spec,
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        ),
        is_draft,
    ) is (needed)


@pytest.mark.parametrize("is_last_rank", [False, True])
def test_only_the_draft_owning_stage_retains_its_manifest(monkeypatch, is_last_rank):
    from vllm.sm80.kernels.exl3 import loader

    monkeypatch.setattr(
        loader, "get_pp_group", lambda: SimpleNamespace(is_last_rank=is_last_rank)
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object()),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    assert loader.manifest_still_needed(config, is_draft=False) is is_last_rank


def test_release_frees_the_loader_and_layer_configs(exl3_config):
    from vllm.sm80.kernels.exl3.loader import release_manifests

    # The MTP draft loads with the target's config while its MoE layer holds a
    # second, draft-only config; both inventories are freed.
    target = exl3_module.Exl3Config()
    target.maybe_update_config(str(exl3_config.manifest.directory))
    layer = torch.nn.Module()
    layer.quant_method = exl3_module.Exl3MoEMethod(
        exl3_config, "model.layers.0.mlp.experts", _moe_config()
    )
    model = torch.nn.Sequential(layer)
    release_manifests(model, target)
    assert target.manifest is None and exl3_config.manifest is None
