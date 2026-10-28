# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram ``wkv`` replicated for prefill steps
(VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL).

Upstream's Engram ``wkv`` is column-parallel with ``gather_output``: each TP
rank projects its ``1/tp`` of the output columns and an all-gather of the BF16
outputs assembles every row. In a prefill step that gather moves
``rows * output * 2 * (tp - 1) / tp`` bytes per rank and Engram layer.

Here each rank also holds the other ranks' shards: after the layer's own
quantization method has processed this rank's shard, the processed bytes of
every parameter are gathered once over the TP group into one ``[tp, ...]``
replica. The layer's parameters become views of this rank's slot (no second
copy), so a step below :attr:`ReplicatedPrefillWkv.min_rows` rows runs the
sharded GEMM and the all-gather as before, on the same bytes. A step of at
least ``min_rows`` rows applies the method once per shard, each call the
GEMM that shard's rank would run (same input, same shard bytes, same
shapes), and concatenates the outputs in rank order, as the all-gather does:
the gathered row, byte for byte, without the collective.

``min_rows`` (:func:`prefill_min_rows`) is one more than the largest step
that can be decode only (``max_num_seqs * (1 + num_speculative_tokens)``) or
that a CUDA graph captures, so decode steps and every captured graph keep the
sharded path; the replicated path runs in eager prefill steps only.
"""

import copy
from itertools import islice

import torch
from torch import nn

from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.linear import ColumnParallelLinear, LinearMethodBase
from vllm_sm80 import capture


def prefill_min_rows(vllm_config) -> int:
    """Fewest rows of a step that takes the replicated ``wkv``: more than any
    decode-only step and any CUDA graph capture size."""
    scheduler = vllm_config.scheduler_config
    speculative = vllm_config.speculative_config
    drafts = speculative.num_speculative_tokens if speculative is not None else 0
    decode_rows = scheduler.max_num_seqs * (1 + (drafts or 0))
    captured = vllm_config.compilation_config.max_cudagraph_capture_size or 0
    return max(decode_rows, captured) + 1


def _gather_tp(tensor: torch.Tensor) -> torch.Tensor:
    """Every TP rank's ``tensor`` stacked as ``[tp, *tensor.shape]``: a byte
    copy, whatever the dtype."""
    group = get_tp_group()
    flat = tensor.contiguous().view(-1).view(torch.uint8)
    out = flat.new_empty((group.world_size, flat.numel()))
    torch.distributed.all_gather_into_tensor(out, flat, group=group.device_group)
    return out.view(tensor.dtype).view(group.world_size, *tensor.shape)


class ReplicatedWkvMethod(LinearMethodBase):
    """The layer's own method, plus the gathered replica after loading."""

    def __init__(self, inner: LinearMethodBase):
        self.inner = inner

    @property
    def supports_pre_processed_weights(self) -> bool:  # type: ignore[override]
        return self.inner.supports_pre_processed_weights

    @property
    def requires_device_loading(self) -> bool:  # type: ignore[override]
        return self.inner.requires_device_loading

    def create_weights(self, *args, **kwargs):
        raise TypeError("ReplicatedWkvMethod adopts a layer whose weights exist")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.inner.process_weights_after_loading(layer)
        group = get_tp_group()
        if (group.world_size, group.rank_in_group) != (layer.tp_size, layer.tp_rank):
            raise ValueError(
                f"wkv rank {layer.tp_rank}/{layer.tp_size} is not TP rank "
                f"{group.rank_in_group}/{group.world_size}"
            )
        tensors = {
            **dict(layer.named_parameters(recurse=False)),
            **dict(layer.named_buffers(recurse=False)),
        }
        install_replica(layer, {name: _gather_tp(t.data) for name, t in tensors.items()})

    def apply(self, layer, x, bias=None):
        return self.inner.apply(layer, x, bias)


def install_replica(layer: nn.Module, stacked: dict[str, torch.Tensor]) -> None:
    """Point the layer's processed parameters and buffers at its rank's slot
    of ``stacked`` (name -> ``[tp, ...]``) and keep one view of the layer per
    rank in ``layer.wkv_shards``."""
    params = dict(layer.named_parameters(recurse=False))
    buffers = dict(layer.named_buffers(recurse=False))
    if set(stacked) != set(params) | set(buffers):
        raise ValueError(f"replica {sorted(stacked)} does not match the layer")
    for name, tensor in {**params, **buffers}.items():
        if stacked[name].shape != (layer.tp_size, *tensor.shape):
            raise ValueError(f"replica {name} {list(stacked[name].shape)} is not "
                             f"{layer.tp_size} x {list(tensor.shape)}")
    for name, param in params.items():
        param.data = stacked[name][layer.tp_rank]
    for name in buffers:
        layer._buffers[name] = stacked[name][layer.tp_rank]
    shards = []
    for rank in range(layer.tp_size):
        # Every attribute the method reads from the layer; this rank's tensors.
        shard = copy.copy(layer)
        object.__setattr__(shard, "_parameters", {
            name: nn.Parameter(stacked[name][rank], requires_grad=False)
            for name in params
        } | {name: None for name, p in layer._parameters.items() if p is None})
        object.__setattr__(shard, "_buffers", {
            name: stacked[name][rank] for name in buffers
        })
        object.__setattr__(shard, "_modules", {})
        object.__setattr__(shard, "wkv_shards", None)
        shards.append(shard)
    # A plain list: the shards stay out of the module tree (loading, state
    # dicts, device moves).
    object.__setattr__(layer, "wkv_shards", shards)


class ReplicatedPrefillWkv(ColumnParallelLinear):
    """An Engram ``wkv`` that projects steps of ``min_rows`` rows or more
    from its replica, without the all-gather."""

    min_rows: int | None
    wkv_shards: list | None

    @classmethod
    def adopt(cls, layer: ColumnParallelLinear, min_rows: int) -> None:
        if type(layer) is not ColumnParallelLinear:
            raise ValueError(f"replicated wkv adopts a ColumnParallelLinear, "
                             f"not {type(layer).__name__}")
        if not layer.gather_output or layer.tp_size < 2:
            raise ValueError("replicated wkv needs a TP-sharded, gathered output")
        if layer.bias is not None or layer.return_bias:
            raise ValueError("replicated wkv takes a layer without bias")
        if min_rows < 1:
            raise ValueError(f"min_rows {min_rows} must be positive")
        layer.__class__ = cls
        layer.quant_method = ReplicatedWkvMethod(layer.quant_method)
        layer.min_rows = min_rows
        object.__setattr__(layer, "wkv_shards", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.min_rows is None or input_.shape[0] < self.min_rows:
            return super().forward(input_)
        return self.forward_replicated(input_)

    def forward_replicated(self, input_: torch.Tensor) -> torch.Tensor:
        """Every rank's shard GEMM on this rank, outputs in rank order."""
        shards = self.wkv_shards
        if shards is None:
            raise RuntimeError(
                "replicated wkv has no replica: its weights were not processed "
                "after loading"
            )
        inner = self.quant_method.inner
        return torch.cat([inner.apply(shard, input_) for shard in shards], dim=-1)


def adopt_replicated_wkv(model) -> None:
    """Replicate every local Engram layer's ``wkv`` for prefill steps."""
    min_rows = prefill_min_rows(model.vllm_config)
    batched = model.vllm_config.scheduler_config.max_num_batched_tokens
    if min_rows > batched:
        # No step could take the replica: it would only cost memory.
        raise ValueError(
            f"replicated wkv: steps of {min_rows}+ rows (above every decode "
            f"step and capture) exceed max_num_batched_tokens {batched}"
        )
    if capture.captures("hybrid"):
        # Each shard copy carries this rank's capture facts: the capture would
        # record the four shard GEMMs as this rank's.
        raise ValueError(
            "VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL: the hybrid capture "
            "records the sharded wkv GEMM, not the replica; capture with the "
            "flag off"
        )
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        engram = getattr(layer, "engram", None)
        if engram is None:
            continue
        if engram.use_sequence_parallel or engram._batch is not None:
            raise ValueError("replicated wkv requires unbatched TP projection")
        if engram.embed_tokens.dp_size != 1:
            raise ValueError("replicated wkv requires TP-only table sharding")
        ReplicatedPrefillWkv.adopt(engram.wkv, min_rows)
