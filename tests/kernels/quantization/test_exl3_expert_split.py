# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 routed experts split across the tensor-parallel ranks (vLLM without
--enable-expert-parallel: MoE tp_size > 1, ep_size 1), on CPU.

Every rank holds every expert, cut into whole H128 blocks of the
intermediate: chunks as even as the blocks allow, the larger ones first, and
rank r of layer L holds chunk (r + L) mod the TP size
(quant.tensor_parallel_expert_split). Gate and up keep the chunk's output
columns (trellis columns, svh) and the whole suh; down keeps the chunk's input
rows (trellis rows, suh) and the whole svh. Checked here:

- the chunks tile the intermediate once per layer, the rotation gives every
  rank the larger chunk in an equal share of the layers;
- the method builds under TP, reports the rank's width to vLLM, holds every
  expert under its own id, and refuses the whole-expert plans and
  dispatchers it cannot serve;
- through the real loader (create_weights, Exl3PackedWeight.load) each rank
  holds exactly the checkpoint's bytes of its chunk, and the ranks' slices
  reassemble every expert bitwise: on synthetic checkpoints (always) and on
  one expert of three layers of the checkpoint named by
  VLLM_EXL3_TEST_CHECKPOINT (DeepSeek-V4.1-Flash: a 3-bit layer, a 2-bit
  layer and a drafter layer);
- the experts hand the router's top-k ids and weights to the operator
  untouched, with the identity expert map and the rank's width.
"""

import json
import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from tests.kernels.quantization.test_exl3_expert_buffers import (
    _check_layout,
    _file_array,
    _namespace,
)
from tests.model_executor.layers.test_exl3_layers import _write_sparse
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
from vllm_sm80.kernels.exl3 import quant
from vllm_sm80.kernels.exl3.utils import Exl3LoadLedger, block_shard

ROLES = ("gate_proj", "up_proj", "down_proj")
KINDS = ("trellis", "suh", "svh")
HIDDEN = 256
EXPERTS = 4
# (intermediate, TP size): DeepSeek-V4.1-Flash's 18 blocks over 4 ranks
# (5, 5, 4, 4) and an odd count over 2 (3, 2).
SHAPES = [(2304, 4), (640, 2)]
LAYERS = 4


def _moe_config(
    intermediate,
    tp_size,
    tp_rank,
    hidden=HIDDEN,
    experts=EXPERTS,
    dp_size=1,
    pcp_size=1,
    sp_size=1,
    enable_eplb=False,
):
    parallel = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=pcp_size,
        dp_size=dp_size,
        ep_size=1,
        tp_rank=tp_rank,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=enable_eplb,
    )
    return FusedMoEConfig(
        num_experts=experts,
        experts_per_token=2,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=experts,
        num_logical_experts=experts,
        moe_parallel_config=parallel,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        max_num_tokens=256,
    )


class _Layer(torch.nn.Module):
    """The RoutedExperts members create_weights reads. ``owned``: the experts
    this test loads (default every expert, vLLM's map without EP)."""

    def __init__(self, global_experts, owned=None):
        super().__init__()
        self.global_num_experts = global_experts
        self.apply_router_weight_on_input = False
        _, self.expert_map, _ = determine_expert_map(1, 0, global_experts)
        self.owned = list(range(global_experts)) if owned is None else owned

    def _map_global_expert_id_to_local_expert_id(self, expert):
        if self.expert_map is not None:
            return int(self.expert_map[expert])
        return self.owned.index(expert) if expert in self.owned else -1


def _build_and_load(
    monkeypatch, config, prefix, moe, layer, contiguous, experts=None
):
    """The method under ``moe``, vLLM's size round-up, create_weights and the
    loader's per-weight load; returns (method, layer)."""
    monkeypatch.setenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", "1" if contiguous else "0")
    method = quant.Exl3MoEMethod(config, prefix, moe)
    hidden, width = method.maybe_roundup_sizes(
        moe.hidden_dim,
        moe.intermediate_size_per_partition,
        moe.in_dtype,
        moe.moe_parallel_config,
    )
    method.create_weights(
        layer,
        len(layer.owned) if experts is None else experts,
        hidden,
        width,
        torch.bfloat16,
    )
    manifest = config.manifest
    ledger = Exl3LoadLedger(manifest)
    paths = sorted(
        {
            manifest.tensors[f"{w.matrix.prefix}.{kind}"].path
            for expert in layer.exl3_experts
            for w in expert
            for kind in KINDS
        }
    )
    with ExitStack() as stack:
        files = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in paths
        }
        for expert in layer.exl3_experts:
            for weight in expert:
                weight.load(files, ledger)
    return method, layer


def _whole(manifest, prefix):
    return {
        kind: torch.from_numpy(np.array(_file_array(manifest, f"{prefix}.{kind}")))
        .view(torch.int16)
        for kind in KINDS
    }


def _check_rank(weight, role, whole, columns, hidden):
    """One rank's matrix holds the checkpoint's bytes of its chunk."""
    n0, n1 = columns.start, columns.stop
    got = {kind: getattr(weight, kind).detach().view(torch.int16) for kind in KINDS}
    if role == "down_proj":
        expected = {
            "trellis": whole["trellis"][n0 // 16 : n1 // 16],
            "suh": whole["suh"][n0:n1],
            "svh": whole["svh"],
        }
        assert (weight.padded_in, weight.padded_out) == (n1 - n0, hidden)
        assert weight.input_slice == columns
    else:
        expected = {
            "trellis": whole["trellis"][:, n0 // 16 : n1 // 16],
            "suh": whole["suh"],
            "svh": whole["svh"][n0:n1],
        }
        assert (weight.padded_in, weight.padded_out) == (hidden, n1 - n0)
        assert weight.output_slices == [columns]
    for kind in KINDS:
        assert torch.equal(got[kind], expected[kind]), f"{weight.matrix.prefix}.{kind}"


def _reassemble(parts, role, whole):
    """The ranks' slices of one matrix, in column order, give it bitwise."""
    parts = sorted(parts, key=lambda p: p[0].start)
    assert parts[0][0].start == 0 and all(
        a.stop == b.start for (a, _), (b, _) in zip(parts, parts[1:])
    )
    sliced, kept = ("suh", "svh") if role == "down_proj" else ("svh", "suh")
    axis = 0 if role == "down_proj" else 1

    def kind_of(weight, kind):
        return getattr(weight, kind).detach().view(torch.int16)

    trellis = torch.cat([kind_of(w, "trellis") for _, w in parts], dim=axis)
    assert torch.equal(trellis, whole["trellis"])
    assert torch.equal(torch.cat([kind_of(w, sliced) for _, w in parts]), whole[sliced])
    for _, weight in parts:
        assert torch.equal(kind_of(weight, kept), whole[kept])


# --------------------------------------------------------------------- the cut


@pytest.mark.parametrize(
    "intermediate,size", SHAPES + [(512, 4), (1408, 4), (2304, 2), (2304, 1)]
)
def test_chunks_tile_the_intermediate_once_per_layer(intermediate, size):
    blocks = intermediate // 128
    base, extra = divmod(blocks, size)
    for layer in range(43):
        chunks = [
            block_shard(intermediate, intermediate, rank, size, layer)
            for rank in range(size)
        ]
        covered = sorted((c.start, c.stop) for c in chunks)
        assert covered[0][0] == 0 and covered[-1][1] == intermediate
        assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
        assert all(c.start % 128 == 0 and c.stop % 128 == 0 for c in chunks)
        widths = sorted((c.stop - c.start) // 128 for c in chunks)
        assert widths == sorted([base + 1] * extra + [base] * (size - extra))
        # Rank r holds chunk (r + layer) mod size, in block order.
        for rank, c in enumerate(chunks):
            index = (rank + layer) % size
            assert c.start == 128 * (index * base + min(index, extra))


def test_every_rank_holds_the_larger_chunk_in_half_the_target_layers():
    heavy = [0] * 4
    for layer in range(40):
        for rank in range(4):
            c = block_shard(2304, 2304, rank, 4, layer)
            heavy[rank] += c.stop - c.start == 640
    assert heavy == [20, 20, 20, 20]
    # The three drafter layers follow the target's numbering (40..42).
    widths = [
        [
            block_shard(2304, 2304, r, 4, layer).stop
            - block_shard(2304, 2304, r, 4, layer).start
            for r in range(4)
        ]
        for layer in (40, 41, 42)
    ]
    assert widths == [[640, 640, 512, 512], [640, 512, 512, 640], [512, 512, 640, 640]]


# ------------------------------------------------------------------ the method


def test_method_builds_under_tp_and_reports_the_rank_width():
    expected = {
        0: [640, 640, 512, 512],
        1: [640, 512, 512, 640],
        7: [512, 640, 640, 512],
    }
    for layer, widths in expected.items():
        for rank in range(4):
            moe = _moe_config(2304, 4, rank, hidden=5120, experts=384)
            # vLLM's even share is not a whole number of H128 blocks.
            assert moe.intermediate_size_per_partition == 576
            method = quant.Exl3MoEMethod(None, f"model.layers.{layer}.ffn.experts", moe)
            intermediate, columns = method.expert_split
            assert intermediate == 2304
            assert columns == block_shard(2304, 2304, rank, 4, layer)
            assert method.maybe_roundup_sizes(
                5120, 576, torch.bfloat16, moe.moe_parallel_config
            ) == (5120, widths[rank])


def test_whole_expert_ep_is_unchanged():
    moe = _moe_config(2304, 1, 0, hidden=5120, experts=384)
    method = quant.Exl3MoEMethod(None, "model.layers.3.ffn.experts", moe)
    assert method.expert_split is None
    assert method.maybe_roundup_sizes(
        5120, 2304, torch.bfloat16, moe.moe_parallel_config
    ) == (5120, 2304)


@pytest.mark.parametrize(
    "option", [{"dp_size": 2}, {"pcp_size": 2}, {"sp_size": 2}, {"enable_eplb": True}]
)
def test_unsupported_dispatch_is_refused_under_tp(option):
    moe = _moe_config(2304, 4, 0, **option)
    with pytest.raises(ValueError, match="tensor-parallel routed experts support no"):
        quant.Exl3MoEMethod(None, "model.layers.0.ffn.experts", moe)
    assert not quant.Exl3Experts._supports_parallel_config(moe.moe_parallel_config)


def test_experts_accept_tp_without_ep():
    supports = quant.Exl3Experts._supports_parallel_config
    assert supports(_moe_config(2304, 4, 1).moe_parallel_config)
    assert supports(_moe_config(2304, 1, 0).moe_parallel_config)


# -------------------------------------------------------- synthetic checkpoint


@pytest.fixture(scope="module", params=SHAPES, ids=lambda s: f"i{s[0]}-tp{s[1]}")
def synthetic(request, tmp_path_factory):
    """LAYERS MoE layers of EXPERTS experts with random payloads: gate and up
    at bits_x2 6, down at 4."""
    intermediate, size = request.param
    directory = tmp_path_factory.mktemp(f"split{intermediate}")
    rng = np.random.default_rng(1022 + intermediate)
    tensors = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            for role, k, n, bits_x2 in (
                ("gate_proj", HIDDEN, intermediate, 6),
                ("up_proj", HIDDEN, intermediate, 6),
                ("down_proj", intermediate, HIDDEN, 4),
            ):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{role}"
                trellis = rng.integers(
                    -(2**15), 2**15, (k // 16, n // 16, 8 * bits_x2)
                )
                tensors[f"{prefix}.trellis"] = (
                    "I16",
                    list(trellis.shape),
                    trellis.astype(np.int16).tobytes(),
                )
                for kind, width in (("suh", k), ("svh", n)):
                    values = rng.uniform(-1.5, 1.5, width).astype(np.float16)
                    tensors[f"{prefix}.{kind}"] = ("F16", [width], values.tobytes())
                tensors[f"{prefix}.mul1"] = (
                    "I32",
                    [],
                    np.uint32(0x83DCD12D).tobytes(),
                )
    _write_sparse(directory / "model.safetensors", tensors)
    config = quant.Exl3Config()
    config.maybe_update_config(str(directory))
    return config, intermediate, size


@pytest.mark.parametrize("contiguous", [False, True])
def test_synthetic_ranks_hold_their_chunk_and_reassemble(
    monkeypatch, synthetic, contiguous
):
    config, intermediate, size = synthetic
    for layer_index in range(LAYERS):
        prefix = f"model.layers.{layer_index}.mlp.experts"
        parts = {}
        for rank in range(size):
            moe = _moe_config(intermediate, size, rank)
            method, layer = _build_and_load(
                monkeypatch, config, prefix, moe, _Layer(EXPERTS), contiguous
            )
            columns = block_shard(intermediate, intermediate, rank, size, layer_index)
            # Every expert under its own id; nothing is left to another rank.
            assert method.global_expert_ids == list(range(EXPERTS))
            assert method.local_experts == EXPERTS
            assert method.nonlocal_keys == []
            assert method.intermediate_size == columns.stop - columns.start
            _check_layout(layer, contiguous)
            for expert, weights in enumerate(layer.exl3_experts):
                for role, weight in zip(ROLES, weights):
                    whole = _whole(config.manifest, f"{prefix}.{expert}.{role}")
                    _check_rank(weight, role, whole, columns, HIDDEN)
                    parts.setdefault((expert, role), []).append((columns, weight))
        for (expert, role), matrix_parts in parts.items():
            whole = _whole(config.manifest, f"{prefix}.{expert}.{role}")
            _reassemble(matrix_parts, role, whole)


def test_rank_width_must_match_the_split(monkeypatch, synthetic):
    config, intermediate, size = synthetic
    method = quant.Exl3MoEMethod(
        config, "model.layers.0.mlp.experts", _moe_config(intermediate, size, 0)
    )
    with pytest.raises(ValueError, match="need this rank's intermediate"):
        method.create_weights(
            _Layer(EXPERTS), EXPERTS, HIDDEN, intermediate // size + 1, torch.bfloat16
        )


# ------------------------------------------------------------- routing


@pytest.mark.parametrize("rows,operator", [(6, "moe_out_v2"), (65, "moe_out")])
def test_experts_pass_routes_untouched_with_the_identity_map(
    monkeypatch, rows, operator
):
    from vllm_sm80.kernels.exl3 import ops

    experts_count, width = 384, 640
    moe = _moe_config(2304, 4, 0, hidden=5120, experts=experts_count)
    identity = torch.arange(experts_count, dtype=torch.int32)
    experts = quant.Exl3Experts(
        moe,
        FusedMoEQuantConfig.make(),
        width,
        identity,
        workspace=8,
        max_rows=128,
        moe_v2_rows=64,
        moe_v2_bits_x2=6,
    )
    seen = []

    def record(name):
        def call(x, ids, weights, w1, w2, expert_map, intermediate, out, *a, **kw):
            seen.append((name, ids, weights, expert_map, intermediate))

        return call

    monkeypatch.setattr(ops, "moe_out_v2", record("moe_out_v2"))
    monkeypatch.setattr(ops, "moe_out", record("moe_out"))
    g = torch.Generator().manual_seed(rows)
    ids = torch.rand(rows, experts_count, generator=g).topk(6, dim=-1)[1].int()
    weights = torch.rand(rows, 6, generator=g)
    ids_before, weights_before = ids.clone(), weights.clone()
    x = torch.zeros(rows, 5120, dtype=torch.bfloat16)
    experts.apply(
        torch.empty_like(x), x, None, None, weights, ids, None, experts_count,
        identity, None, None, None, torch.empty(16, dtype=torch.bfloat16), None,
        False,
    )
    assert len(seen) == 1
    name, got_ids, got_weights, expert_map, intermediate = seen[0]
    assert name == operator
    assert got_ids is ids and got_weights is weights
    assert torch.equal(ids, ids_before) and torch.equal(weights, weights_before)
    assert torch.equal(expert_map, torch.arange(experts_count, dtype=torch.int32))
    assert intermediate == width


# ------------------------------------------------------------ real checkpoint


def _real_config(monkeypatch):
    value = os.environ.get("VLLM_EXL3_TEST_CHECKPOINT")
    if not value or not Path(value).is_dir():
        pytest.skip("set VLLM_EXL3_TEST_CHECKPOINT to an EXL3 MoE checkpoint")
    directory = Path(value)
    raw = json.loads((directory / "config.json").read_text())
    monkeypatch.setenv("VLLM_EXL3_HYBRID", "1")
    config = quant.Exl3Config.from_config(raw.get("quantization_config", {}))
    config.maybe_update_config(str(directory), _namespace(raw))
    return config


# (vLLM layer index, bits_x2 of its routed experts) in DeepSeek-V4.1-Flash
# 2.9 bpw: a 3-bit target layer, a 2-bit one (18..22) and the first drafter
# layer (numbered after the 40 target layers).
REAL_LAYERS = [(10, 6), (20, 4), (40, 8)]
REAL_EXPERT = 7


@pytest.mark.parametrize("layer_index,bits_x2", REAL_LAYERS)
def test_real_expert_ranks_hold_their_chunk_and_reassemble(
    monkeypatch, layer_index, bits_x2
):
    config = _real_config(monkeypatch)
    prefix = f"model.layers.{layer_index}.ffn.experts"
    sources = [
        config.layout.expert_matrix(config, prefix, REAL_EXPERT, role)
        for role in ROLES
    ]
    if any(source not in config.manifest.matrices for source in sources):
        pytest.skip(f"{prefix}: no EXL3 expert {REAL_EXPERT} in the checkpoint")
    gate = config.manifest.matrices[sources[0]]
    hidden, intermediate = gate.padded_in, gate.padded_out
    assert [config.manifest.matrices[s].bits_x2 for s in sources] == [bits_x2] * 3
    wholes = [_whole(config.manifest, source) for source in sources]
    parts = [[] for _ in ROLES]
    for rank in range(4):
        moe = _moe_config(intermediate, 4, rank, hidden=hidden, experts=384)
        # One expert through the real path: the others stay unloaded.
        layer = _Layer(REAL_EXPERT + 1, owned=[REAL_EXPERT])
        layer.expert_map = None
        method, layer = _build_and_load(
            monkeypatch, config, prefix, moe, layer, contiguous=False, experts=1
        )
        columns = block_shard(intermediate, intermediate, rank, 4, layer_index)
        assert method.global_expert_ids == [REAL_EXPERT]
        assert method.intermediate_size == columns.stop - columns.start
        for i, (role, weight) in enumerate(zip(ROLES, layer.exl3_experts[0])):
            assert weight.matrix.prefix == sources[i]
            _check_rank(weight, role, wholes[i], columns, hidden)
            parts[i].append((columns, weight))
    for role, matrix_parts, whole in zip(ROLES, parts, wholes):
        _reassemble(matrix_parts, role, whole)
    chunks = [
        block_shard(intermediate, intermediate, r, 4, layer_index) for r in range(4)
    ]
    print(
        f"{prefix} expert {REAL_EXPERT} ({sources[0]}): bits_x2 {bits_x2}, "
        "intermediate columns of ranks 0..3: "
        f"{[(c.start, c.stop) for c in chunks]}"
    )
