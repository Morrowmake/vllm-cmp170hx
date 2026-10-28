# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A block-diagonal projection run as one GEMM per group.

Upstream marks a block-diagonal linear with ``is_bmm`` and
``bmm_batch_size`` (output group g is input group g times W_g^T) and leaves
the GEMM to a batched kernel of its quantization method. A method without a
batched kernel on this device would otherwise fall back to a dense BF16 copy.
:class:`GroupedLinearMethod` keeps the layer's own method instead: after
loading, every row-sharded parameter is cut into the groups' rows, each group
becomes a sub-layer that the method processes and applies on its own, and
:meth:`GroupedLinearMethod.apply_grouped` projects ``[tokens, groups, input]``
group by group. Any linear method whose parameters are sharded along the
output rows (``output_dim`` 0: dense, block-scaled or per-row-scaled
weights) works unchanged.

Each group's input rows are copied to a contiguous tensor before the method
applies them.
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearMethodBase

# Layer attributes a linear method reads from the layer it processes/applies.
_LAYER_ATTRS = (
    "input_size_per_partition",
    "input_size",
    "params_dtype",
    "weight_block_size",
)


class GroupedLinearMethod(LinearMethodBase):
    def __init__(self, inner: LinearMethodBase, groups: int):
        self.inner = inner
        self.groups = groups

    @classmethod
    def adopt(cls, layer: nn.Module, groups: int) -> None:
        """Run the already-built ``layer`` (its weights created by its own
        method) as ``groups`` groups."""
        if layer.output_size_per_partition % groups:
            raise ValueError(
                f"{layer.output_size_per_partition} outputs do not split into "
                f"{groups} groups"
            )
        layer.quant_method = cls(layer.quant_method, groups)

    def create_weights(self, *args, **kwargs):
        raise TypeError("GroupedLinearMethod adopts a layer whose weights exist")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        rows = layer.output_size_per_partition
        width = rows // self.groups
        params = dict(layer.named_parameters(recurse=False))
        for name, param in params.items():
            if getattr(param, "output_dim", None) != 0 or param.shape[0] != rows:
                raise ValueError(f"Cannot cut {name} {list(param.shape)} into groups")
        grouped = nn.ModuleList()
        for group in range(self.groups):
            sub = nn.Module()
            for name, param in params.items():
                data = param.data[group * width : (group + 1) * width].clone()
                sub.register_parameter(name, nn.Parameter(data, requires_grad=False))
            if "bias" not in params:
                sub.register_parameter("bias", None)
            for attr in _LAYER_ATTRS:
                if hasattr(layer, attr):
                    setattr(sub, attr, getattr(layer, attr))
            sub.output_size_per_partition = width
            sub.output_size = width
            sub.logical_widths = [width]
            self.inner.process_weights_after_loading(sub)
            grouped.append(sub)
        for name in params:
            delattr(layer, name)
        layer.grouped = grouped

    def apply(self, layer, x, bias=None):
        raise ValueError("Grouped projections run through apply_grouped")

    def apply_grouped(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """``x`` [tokens, groups, input] -> [tokens, groups * output]."""
        if x.dim() != 3 or x.shape[1] != self.groups:
            raise ValueError(f"Grouped input must be [tokens, {self.groups}, input]")
        return torch.cat(
            [
                self.inner.apply(sub, x[:, group].contiguous())
                for group, sub in enumerate(layer.grouped)
            ],
            dim=-1,
        )
