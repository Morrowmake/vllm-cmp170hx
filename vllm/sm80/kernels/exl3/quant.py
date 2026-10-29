# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 packed linear and whole-expert inference with BF16 activations."""

import math
import re
from functools import cache, partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sm80 import capture, tuning
from vllm.sm80.kernels.exl3.hybrid import (
    EXL3,
    UNQUANTIZED,
    HybridSpec,
    check_plain_sources,
    parse_hybrid,
)
from vllm.sm80.kernels.exl3.layout import Exl3Layout, exl3_layout
from vllm.sm80.kernels.exl3.override import apply_override
from vllm.sm80.kernels.exl3.utils import (
    Exl3Manifest,
    PackedMatrix,
    aligned_shard,
    block_shard,
)

logger = init_logger(__name__)


# Expert partitions the EXL3 MoE tensor-core operators (moe_out_v2,
# VLLM_EXL3_MOE_V2; moe_out_prefill, VLLM_EXL3_MOE_PREFILL_TC) are validated
# on for the Qwen3.8-Flash-Next hidden-2560 512-expert shape, each layout
# with the local expert count it produces: at tensor parallel size 2 with
# expert parallelism the experts are split over the two ranks (256 of 512
# local), and at tensor parallel size 1 with pipeline parallel size 2 the
# pipeline splits the layers and every stage holds all 512. Any other
# layout/partition combination keeps moe_out. The shape-keyed DeepSeek
# configurations (ops.MOE_V2_CONFIGS) are validated without a layout gate.
_VALIDATED_MOE_PARTITIONS = {(2, 1): 256, (1, 2): 512}


def _moe_partition_closed_reason(
    vllm_config, hidden_size: int, local_experts: int, global_experts: int
) -> str | None:
    """Why the MoE tensor-core operators must not serve this engine's
    expert partition, or None: the shape-keyed gates in ops accept any
    engine that presents a validated matrix shape, so the layouts and
    expert partitions validated for the 512-expert Qwen shape are checked
    here (an arbitrary topology holding a validated-looking partition is
    not a validated configuration)."""
    if (hidden_size, global_experts) != (2560, 512):
        return None
    validated = _VALIDATED_MOE_PARTITIONS
    if vllm_config is None:
        return "no vLLM config is set to pair a layout with its partition"
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    if layout not in validated:
        first, second = tuple(validated)
        return f"(tensor, pipeline) parallel size {layout}, not {first} or {second}"
    if local_experts != validated[layout]:
        return (
            f"layout {layout} holds {local_experts} experts locally, not the "
            f"{validated[layout]} it is validated with"
        )
    return None


def _verify_launch_options(vllm_config) -> None:
    """Refuse a model built without the EXL3 launch options.

    EXL3 weights load only through the "exl3" load format (Exl3ModelLoader),
    and EXL3 vision towers run data-parallel. Without the options vLLM would
    pick the default loader or shard the vision tower by weights, so fail
    instead. initialize_model sets the current config while a model is
    built; a layer built outside a model (unit tests) has none to check.
    """
    if vllm_config is None:
        return
    load_format = vllm_config.load_config.load_format
    if load_format != "exl3":
        raise ValueError(
            f"EXL3 checkpoints need --load-format exl3, not {load_format!r}"
        )
    model_config = vllm_config.model_config
    multimodal = model_config.multimodal_config
    # The MTP draft config carries no launch options of its own.
    if (
        multimodal is not None
        and model_config.runner_type != "draft"
        and multimodal.mm_encoder_tp_mode != "data"
    ):
        raise ValueError(
            "EXL3 vision towers need --mm-encoder-tp-mode data, not "
            f"{multimodal.mm_encoder_tp_mode!r}"
        )


@cache
def _require_sm80() -> None:
    """The extension is built for SM 8.0 only; refuse other GPUs up front."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is not None and capability.to_int() != 80:
        raise ValueError(
            "EXL3 kernels are built for SM 8.0 only; this GPU is SM "
            f"{capability.major}.{capability.minor}"
        )


class Exl3Config(QuantizationConfig):
    """Registered as "exl3" by vllm/sm80/plugin.py.

    Upstream model code consults the attributes and methods below through
    getattr, so they apply only to this config. The checkpoint's model family
    adds those its own model code reads (Exl3Layout.config_hooks).

    A hybrid checkpoint (vllm/sm80/kernels/exl3/hybrid.py) routes each module
    to EXL3, unquantized or its second method by the config's rules; the
    config then also answers ``weight_block_size`` with the second method's,
    which model code reads to name and pad that method's scales.
    """

    min_transform_size = 128
    # Shared experts of the models that read this: TP shards keep whole
    # transform blocks.
    shared_expert_shard_group_size = min_transform_size

    def __init__(self, hybrid: HybridSpec | None = None):
        super().__init__()
        self.manifest: Exl3Manifest | None = None
        self.mapped_prefixes: dict[str, str] = {}
        # The checkpoint's model family (vllm/sm80/kernels/exl3/layout.py),
        # chosen from the HF config by maybe_update_config.
        self.layout = Exl3Layout()
        self.hybrid = hybrid
        if hybrid is not None:
            block = getattr(hybrid.secondary, "weight_block_size", None)
            if block is not None:
                self.weight_block_size = block

    @classmethod
    def get_name(cls) -> Any:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> None:
        """Refuse EXL3 off CUDA while ModelConfig verifies quantization.

        ModelConfig asks every registered method before it checks the
        platform; EXL3 never replaces the checkpoint's method.
        """
        if hf_quant_cfg.get("quant_method") == "exl3" or user_quant == "exl3":
            from vllm.platforms import current_platform

            if not current_platform.is_cuda():
                raise ValueError("EXL3 execution requires the NVIDIA CUDA platform")
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(parse_hybrid(config))

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        if not Path(model_name).is_dir():
            raise ValueError("Download the EXL3 safetensors checkpoint before loading")
        self.layout = exl3_layout(hf_config)
        self.manifest = Exl3Manifest(
            model_name, select_by_index=self.hybrid is not None
        )
        if envs.VLLM_EXL3_OVERRIDE:
            apply_override(
                self.manifest, envs.VLLM_EXL3_OVERRIDE, envs.VLLM_EXL3_OVERRIDE_SHA256
            )
        self.layout.check_manifest(self.manifest)

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.manifest is None:
            raise ValueError("EXL3 name mapping requires a checkpoint manifest")
        mapped_prefixes: dict[str, str] = {}
        for source in self.manifest.matrices:
            mapped = hf_to_vllm_mapper.map_name(f"{source}.weight")
            if mapped is None:
                continue
            mapped = mapped.removesuffix(".weight")
            if mapped in mapped_prefixes and mapped_prefixes[mapped] != source:
                raise ValueError(f"Conflicting EXL3 module mapping: {mapped}")
            mapped_prefixes[mapped] = source
        self.mapped_prefixes = mapped_prefixes
        if self.hybrid is not None:
            self.hybrid.secondary.apply_vllm_mapper(hf_to_vllm_mapper)

    def __getattr__(self, name: str):
        """The layout hooks the model family's own code looks up with getattr
        (Exl3Layout.config_hooks); reached only for names the config lacks."""
        layout = self.__dict__.get("layout")
        if layout is not None and name in layout.config_hooks:
            return getattr(layout, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def release_manifest(self) -> None:
        """Drop the checkpoint inventory once this config's model is loaded."""
        self.manifest = None
        self.mapped_prefixes = {}

    def source_prefix(self, prefix: str) -> str:
        """Resolve model construction prefixes to checkpoint module names."""
        assert self.manifest is not None, (
            "EXL3 manifest must precede model construction"
        )
        if prefix in self.mapped_prefixes:
            return self.mapped_prefixes[prefix]
        return self.layout.source_prefix(self, prefix)

    def constituents(self, prefix: str, outputs: int = 0) -> list[PackedMatrix]:
        assert self.manifest is not None
        source = self.source_prefix(prefix)
        if source in self.manifest.matrices:
            return [self.manifest.matrices[source]]
        names = self._member_names(source, outputs)
        found = [
            self.manifest.matrices[n] for n in names if n in self.manifest.matrices
        ]
        if found and len(found) != len(names):
            raise ValueError(f"Mixed plain/packed EXL3 fused projection: {prefix}")
        return found

    def _member_names(self, source: str, outputs: int) -> list[str]:
        """Checkpoint module names of the members of the fused projection
        ``source`` with ``outputs`` logical outputs."""
        parent, _, leaf = source.rpartition(".")
        members = self.layout.members(self, leaf, source, outputs)
        if members is None:
            members = self.packed_modules_mapping.get(leaf, [leaf])
        return [
            self.source_prefix(f"{parent}.{member}" if parent else member)
            for member in members
        ]

    def _module_sources(self, prefix: str, outputs: int) -> list[str]:
        """Checkpoint module names a linear module loads from: itself when
        the checkpoint holds it whole, else its fused members."""
        assert self.manifest is not None
        source = self.source_prefix(prefix)
        if (
            source in self.manifest.matrices
            or f"{source}.weight" in self.manifest.tensors
            or self.layout.grouped_members(self, source)
        ):
            return [source]
        return self._member_names(source, outputs)

    def grouped_constituents(self, prefix: str) -> list[PackedMatrix]:
        """Per-group matrices of a block-diagonal projection stored as one
        EXL3 matrix per group (Exl3Layout.grouped_members), in group order."""
        assert self.manifest is not None
        names = self.layout.grouped_members(self, self.source_prefix(prefix))
        missing = [n for n in names if n not in self.manifest.matrices]
        if missing:
            raise ValueError(f"Missing EXL3 group matrices: {missing[:4]}")
        return [self.manifest.matrices[n] for n in names]

    def get_quant_method(self, layer: nn.Module, prefix: str):
        from vllm.config import get_current_vllm_config_or_none

        _verify_launch_options(get_current_vllm_config_or_none())
        _require_sm80()
        if self.hybrid is not None:
            return self._hybrid_quant_method(layer, prefix)
        return self._exl3_quant_method(layer, prefix)

    def _hybrid_quant_method(self, layer: nn.Module, prefix: str):
        """The method the hybrid rules give the module (hybrid.py)."""
        assert self.hybrid is not None and self.manifest is not None
        if capture.active:
            capture.config()
        if isinstance(layer, RoutedExperts):
            sources = [self.source_prefix(prefix)]
        elif isinstance(layer, (LinearBase, ParallelLMHead)):
            outputs = len(getattr(layer, "output_sizes", ()))
            sources = self._module_sources(prefix, outputs)
        else:
            return None
        method = self.hybrid.method_for(sources)
        if method == EXL3:
            quant_method = self._exl3_quant_method(layer, prefix)
            if quant_method is None or isinstance(
                quant_method, UnquantizedLinearMethod
            ):
                raise ValueError(
                    f"{prefix} takes exl3; {sources} hold no EXL3 matrices"
                )
            return quant_method
        if not isinstance(layer, RoutedExperts):
            check_plain_sources(self.manifest, sources, method)
            if capture.active:
                layer._hybrid_capture_sources = [
                    (source, list(self.manifest.tensors[f"{source}.weight"].shape))
                    for source in sources
                ]
        if method == UNQUANTIZED:
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None
        secondary = self.hybrid.secondary
        secondary.packed_modules_mapping = self.packed_modules_mapping
        return secondary.get_quant_method(layer, prefix)

    def _exl3_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(self, prefix, layer.moe_config)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            matrices = self.constituents(
                prefix, len(getattr(layer, "output_sizes", ()))
            )
            if matrices:
                return Exl3LinearMethod(
                    matrices, uneven=self.layout.shards_unevenly(prefix)
                )
            groups = self.grouped_constituents(prefix)
            if groups and isinstance(layer, LinearBase):
                return Exl3GroupedLinearMethod(groups)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
        return None


# Every view of an expert buffer starts at a multiple of this many bytes, the
# alignment the caching allocator gives each block, so the kernels see the
# same alignment as with one allocation per tensor.
EXPERT_VIEW_ALIGN_BYTES = 512


def expert_buffer_views(
    matrices: list[PackedMatrix],
    slices: list[tuple[slice, list[slice]]] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(trellis, suh, svh) views of packed matrices, in ``matrices`` order,
    into one new buffer per tensor kind (int16 trellis, FP16 suh, FP16 svh)
    on the default device. ``slices``: per matrix the (input slice, output
    slices) its Exl3PackedWeight loads (default: the whole matrix). Each view
    is a contiguous tensor of the shape Exl3PackedWeight allocates for them,
    at an offset that is a multiple of EXPERT_VIEW_ALIGN_BYTES
    (VLLM_EXL3_EXPERTS_CONTIGUOUS)."""
    if slices is None:
        slices = [(slice(0, m.padded_in), [slice(0, m.padded_out)]) for m in matrices]
    dims = [
        (rows.stop - rows.start, sum(c.stop - c.start for c in cols), m.bits_x2)
        for m, (rows, cols) in zip(matrices, slices, strict=True)
    ]
    kinds = (
        (torch.int16, lambda k, n, bits_x2: (k // 16, n // 16, 8 * bits_x2)),
        (torch.float16, lambda k, n, bits_x2: (k,)),
        (torch.float16, lambda k, n, bits_x2: (n,)),
    )
    columns = []
    for dtype, shape_of in kinds:
        step = EXPERT_VIEW_ALIGN_BYTES // dtype.itemsize
        shapes = [shape_of(*dim) for dim in dims]
        offsets, total = [], 0
        for shape in shapes:
            offsets.append(total)
            total += -(-math.prod(shape) // step) * step
        buffer = torch.empty(total, dtype=dtype)
        columns.append(
            [
                buffer[offset : offset + math.prod(shape)].view(shape)
                for offset, shape in zip(offsets, shapes)
            ]
        )
    return list(zip(*columns))


class Exl3PackedWeight(nn.Module):
    def __init__(
        self,
        matrix: PackedMatrix,
        logical_in: int,
        logical_out: int,
        input_slice: slice,
        output_slices: list[slice],
        views: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        """``views``: (trellis, suh, svh) tensors to load into instead of new
        allocations (expert_buffer_views); their shapes and dtypes must be the
        ones allocated here."""
        super().__init__()
        for part, limit in [(input_slice, matrix.padded_in)] + [
            (part, matrix.padded_out) for part in output_slices
        ]:
            if (
                part.step is not None
                or not 0 <= part.start < part.stop <= limit
                or part.start % 128
                or part.stop % 128
            ):
                raise ValueError("EXL3 shards must select complete H128 blocks")
        self.matrix = matrix
        self.logical_in = logical_in
        self.logical_out = logical_out
        self.input_slice = input_slice
        self.output_slices = output_slices
        self.padded_in = input_slice.stop - input_slice.start
        self.padded_out = sum(s.stop - s.start for s in output_slices)
        if (
            not 0 < logical_in <= self.padded_in
            or not 0 < logical_out <= self.padded_out
        ):
            raise ValueError("EXL3 logical dimensions exceed assigned packed blocks")
        shapes = (
            (self.padded_in // 16, self.padded_out // 16, 8 * matrix.bits_x2),
            (self.padded_in,),
            (self.padded_out,),
        )
        dtypes = (torch.int16, torch.float16, torch.float16)
        if views is None:
            views = tuple(
                torch.empty(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)
            )
        elif any(
            tuple(view.shape) != shape
            or view.dtype != dtype
            or not view.is_contiguous()
            for view, shape, dtype in zip(views, shapes, dtypes)
        ):
            raise ValueError("EXL3 packed weight views do not match the matrix")
        trellis, suh, svh = views
        self.trellis = nn.Parameter(trellis, requires_grad=False)
        self.suh = nn.Parameter(suh, requires_grad=False)
        self.svh = nn.Parameter(svh, requires_grad=False)
        self.loaded = False

    def load(self, files, ledger):
        manifest = ledger.manifest
        prefix = self.matrix.prefix
        ks = slice(self.input_slice.start // 16, self.input_slice.stop // 16)
        header = manifest.tensors[f"{prefix}.trellis"]
        packed = files[header.path].get_slice(header.name)
        offset = 0
        for output_slice in self.output_slices:
            ns = slice(output_slice.start // 16, output_slice.stop // 16)
            width = ns.stop - ns.start
            self.trellis.data[:, offset : offset + width].copy_(packed[ks, ns, :])
            offset += width
        for suffix, param, slices in (
            ("suh", self.suh, [self.input_slice]),
            ("svh", self.svh, self.output_slices),
        ):
            header = manifest.tensors[f"{prefix}.{suffix}"]
            data = files[header.path].get_slice(header.name)
            offset = 0
            for part in slices:
                width = part.stop - part.start
                param.data[offset : offset + width].copy_(data[part])
                offset += width
        for key in self.matrix.keys:
            ledger.mark(key, "consumed", prefix)
        self.loaded = True


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, matrices: list[PackedMatrix], uneven: bool = False):
        self.matrices = matrices
        # TP shards in whole H128 blocks, the leading ranks taking one more
        # block when the blocks do not divide evenly (Exl3Layout
        # .shards_unevenly); otherwise every rank's shard is equal.
        self.uneven = uneven
        # LM head: (offset, width, partition) of this rank's vocabulary in the
        # H128 blocks it computes; None for other layers.
        self.head_columns: tuple[int, int, int] | None = None
        # Keys of the matrices other ranks hold (grouped projections).
        self.nonlocal_keys: list[str] = []
        # Output dtype (ops.LINEAR_OUT_DTYPES): BF16, or FP32 for a layer whose
        # output the model keeps in FP32 (set_output_dtype; an LM head takes
        # ModelConfig.head_dtype).
        self.out_dtype = torch.bfloat16

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("EXL3 requires BF16 activations")
        if isinstance(layer, ParallelLMHead):
            self._create_head_weights(layer, input_size, sum(output_partition_sizes))
            return
        rank, size = layer.tp_rank, layer.tp_size
        if isinstance(layer, ReplicatedLinear):
            rank, size = 0, 1
        row_parallel = isinstance(layer, RowParallelLinear)
        parts = list(output_partition_sizes)
        gdn = self.matrices[0].prefix.endswith(".in_proj_qkv")
        if gdn:
            if len(parts) != 4 or len(self.matrices) != 2:
                raise ValueError("EXL3 GDN requires separate Q/K/V/Z logical slices")
            groups = [parts[:3], parts[3:]]
        else:
            groups = [[n] for n in parts]
        if len(groups) != len(self.matrices):
            raise ValueError("EXL3 packed constituents do not match linear outputs")
        # The fused QSA layer's K/V follow the KV heads like QKVParallelLinear;
        # its indexer Q/K is replicated on every rank.
        qsa_indexer = len(self.matrices) == 4 and self.matrices[3].prefix.endswith(
            ".indexer.index_qk_proj"
        )
        kv_heads = isinstance(layer, QKVParallelLinear) or qsa_indexer
        shard = block_shard if self.uneven else aligned_shard
        weights = []
        for index, (matrix, group) in enumerate(zip(self.matrices, groups)):
            output_rank, output_size_tp = rank, size
            if row_parallel or (qsa_indexer and index == 3):
                output_rank, output_size_tp = 0, 1
            elif kv_heads and index in (1, 2):
                output_rank //= layer.num_kv_head_replicas
                output_size_tp //= layer.num_kv_head_replicas
            logical_output = sum(group) * output_size_tp
            matrix.validate_logical(input_size, logical_output)
            input_slice = shard(
                input_size,
                matrix.padded_in,
                rank if row_parallel else 0,
                size if row_parallel else 1,
            )
            output_slices = []
            start = 0
            for local_width in group:
                full_width = local_width * output_size_tp
                padded_width = matrix.padded_out if len(group) == 1 else full_width
                part = shard(full_width, padded_width, output_rank, output_size_tp)
                output_slices.append(slice(start + part.start, start + part.stop))
                start += full_width
            logical_in, logical_out = input_size_per_partition, sum(group)
            if self.uneven:
                # The shard widths are the blocks this rank holds, not the
                # layer's equal partition sizes.
                if row_parallel:
                    logical_in = input_slice.stop - input_slice.start
                logical_out = sum(s.stop - s.start for s in output_slices)
            weights.append(
                Exl3PackedWeight(
                    matrix,
                    logical_in,
                    logical_out,
                    input_slice,
                    output_slices,
                )
            )
        layer.exl3_weights = nn.ModuleList(weights)

    def _create_head_weights(self, layer, input_size: int, partition: int) -> None:
        """LM head shard: the H128 blocks that cover this rank's vocabulary
        range, whose columns apply() cuts to the range and pads to the
        partition, so a range that starts or ends inside a block (a
        vocabulary that does not split into equal whole blocks) loads too."""
        if len(self.matrices) != 1:
            raise ValueError("EXL3 LM head requires one packed matrix")
        (matrix,) = self.matrices
        matrix.validate_logical(input_size, layer.org_vocab_size)
        indices = layer.shard_indices
        start, end = indices.org_vocab_start_index, indices.org_vocab_end_index
        if not 0 <= start < end <= partition + start:
            raise ValueError("EXL3 LM head shard has no vocabulary rows")
        low = start // 128 * 128
        high = min(-(-end // 128) * 128, matrix.padded_out)
        self.head_columns = (start - low, end - start, partition)
        # The logits come out in the head dtype, as LogitsProcessor computes
        # them for an unquantized head (Exl3HeadLogitsProcessor).
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            self.out_dtype = _linear_out_dtype(vllm_config.model_config.head_dtype)
        layer.exl3_weights = nn.ModuleList(
            [
                Exl3PackedWeight(
                    matrix,
                    input_size,
                    end - low,
                    slice(0, matrix.padded_in),
                    [slice(low, high)],
                )
            ]
        )

    def process_weights_after_loading(self, layer):
        from vllm.sm80.kernels.exl3 import capture_points
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        self.row_chunks = []
        for weight in layer.exl3_weights:
            if not weight.loaded:
                raise ValueError(f"Unloaded EXL3 matrix: {weight.matrix.prefix}")
            if not all(p.is_cuda for p in weight.parameters()):
                raise ValueError("EXL3 packed weights and scales must remain on CUDA")
            self.row_chunks.append(
                exl3_ops.linear_row_chunk(weight.padded_in, weight.padded_out)
            )
            capture_points.register_linear(weight, self.row_chunks[-1], self.out_dtype)

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).contiguous()
        outputs = [
            torch.ops.vllm.exl3_linear(
                x,
                weight.trellis,
                weight.suh,
                weight.svh,
                weight.matrix.bits_x2,
                weight.matrix.codebook,
                weight.logical_out,
                row_chunk,
                self.out_dtype,
            )
            for weight, row_chunk in zip(layer.exl3_weights, self.row_chunks)
        ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if self.head_columns is not None:
            output = _head_partition(output, *self.head_columns)
        if bias is not None:
            output.add_(bias)
        return output.reshape(*shape, output.shape[-1])


def _head_partition(
    output: torch.Tensor, offset: int, width: int, partition: int
) -> torch.Tensor:
    """This rank's vocabulary columns of its covering blocks, padded to the
    partition (padded vocabulary rows are dropped after the gather)."""
    if offset == 0 and width == partition:
        return output
    output = output[:, offset : offset + width]
    if partition > width:
        return torch.nn.functional.pad(output, (0, partition - width))
    return output.contiguous()


def _linear_out_dtype(dtype: torch.dtype) -> torch.dtype:
    from vllm.sm80.kernels.exl3.ops import LINEAR_OUT_DTYPES

    if dtype not in LINEAR_OUT_DTYPES:
        raise ValueError(f"EXL3 linear outputs are BF16 or FP32, not {dtype}")
    return dtype


class OutDtypeLinearMethod(UnquantizedLinearMethod):
    """An unquantized linear that returns its FP32-accumulated product in
    ``out_dtype``: FP32 keeps the sum of the exact BF16 products unrounded,
    as an FP32 linear of the same BF16 input and weight computes it."""

    def __init__(self, out_dtype: torch.dtype):
        super().__init__()
        self.out_dtype = out_dtype

    def apply(self, layer, x, bias=None):
        shape = x.shape
        output = torch.mm(
            x.reshape(-1, shape[-1]), layer.weight.t(), out_dtype=self.out_dtype
        )
        if bias is not None:
            output = output + bias.to(self.out_dtype)
        return output.reshape(*shape[:-1], output.shape[-1])


def set_output_dtype(layer: nn.Module, dtype: torch.dtype) -> None:
    """Have the EXL3 or unquantized linear ``layer`` return ``dtype`` (BF16
    or FP32). Both accumulate in FP32; FP32 stores that result instead of
    rounding it to BF16 (an unquantized layer takes OutDtypeLinearMethod)."""
    method = getattr(layer, "quant_method", None)
    if isinstance(method, Exl3LinearMethod):
        method.out_dtype = _linear_out_dtype(dtype)
    elif type(method) in (UnquantizedLinearMethod, OutDtypeLinearMethod):
        layer.quant_method = OutDtypeLinearMethod(_linear_out_dtype(dtype))
    else:
        raise TypeError(
            f"{type(layer).__name__} has no EXL3 or unquantized linear method "
            f"({type(method).__name__}) to set an output dtype on"
        )


def set_moe_prefill_v2(layer: nn.Module) -> str | None:
    """Have the EXL3 routed experts of ``layer`` (the module holding the MoE
    quant method) take moe_out_prefill_v2 for prefill calls when its gate
    opens at load (Exl3MoEMethod._moe_prefill_v2). Returns why it cannot, or
    None."""
    method = getattr(layer, "quant_method", None)
    if not isinstance(method, Exl3MoEMethod):
        kind = type(method).__name__
        return f"{type(layer).__name__} has no EXL3 MoE method ({kind})"
    method.prefill_v2 = True
    return None


class Exl3HeadLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that also serves a head_dtype different from the
    model dtype with an EXL3 LM head: the head stores its logits in the head
    dtype (Exl3LinearMethod._create_head_weights), so it runs as it is.
    Every other head takes the upstream path."""

    @classmethod
    def adopt(cls, processor: LogitsProcessor) -> None:
        assert type(processor) is LogitsProcessor, type(processor)
        processor.__class__ = cls

    def _apply_head(
        self,
        lm_head: nn.Module,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        method = lm_head.quant_method
        if not isinstance(method, Exl3LinearMethod):
            return super()._apply_head(lm_head, hidden_states, embedding_bias)
        head_dtype = self.head_dtype or hidden_states.dtype
        if method.out_dtype != head_dtype:
            raise ValueError(
                f"The EXL3 LM head stores {method.out_dtype}, the head dtype is "
                f"{head_dtype}"
            )
        return method.apply(lm_head, hidden_states, bias=embedding_bias)


class Exl3GroupedLinearMethod(Exl3LinearMethod):
    """A block-diagonal projection stored as one EXL3 matrix per group:
    output group g is input group g times W_g^T. TP ranks own whole groups in
    order, so a column-parallel layer of G groups gives each of its tp ranks
    G / tp of them. The layer has no dense weight; callers project through
    :meth:`apply_grouped` (DeepSeek-V4.1 attention ``wo_a``)."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("EXL3 requires BF16 activations")
        groups, size = len(self.matrices), layer.tp_size
        if groups % size or output_size % groups:
            raise ValueError(f"EXL3 cannot split {groups} groups over {size} ranks")
        local, group_out = groups // size, output_size // groups
        mine = range(layer.tp_rank * local, (layer.tp_rank + 1) * local)
        self.nonlocal_keys = [
            key
            for group, matrix in enumerate(self.matrices)
            if group not in mine
            for key in matrix.keys
        ]
        if sum(output_partition_sizes) != local * group_out:
            raise ValueError("EXL3 group widths do not match the layer partition")
        weights = []
        for matrix in self.matrices[
            layer.tp_rank * local : (layer.tp_rank + 1) * local
        ]:
            matrix.validate_logical(input_size, group_out)
            weights.append(
                Exl3PackedWeight(
                    matrix,
                    input_size,
                    group_out,
                    slice(0, matrix.padded_in),
                    [slice(0, matrix.padded_out)],
                )
            )
        layer.exl3_weights = nn.ModuleList(weights)

    def apply(self, layer, x, bias=None):
        raise ValueError("Grouped EXL3 projections run through apply_grouped")

    def apply_grouped(self, layer, x: torch.Tensor) -> torch.Tensor:
        """``x`` [tokens, local groups, input] -> [tokens, local groups *
        output], group by group in this rank's group order."""
        if x.dim() != 3 or x.shape[1] != len(layer.exl3_weights):
            raise ValueError("Grouped EXL3 input must be [tokens, groups, input]")
        outputs = [
            torch.ops.vllm.exl3_linear(
                x[:, group].contiguous(),
                weight.trellis,
                weight.suh,
                weight.svh,
                weight.matrix.bits_x2,
                weight.matrix.codebook,
                weight.logical_out,
                row_chunk,
                self.out_dtype,
            )
            for group, (weight, row_chunk) in enumerate(
                zip(layer.exl3_weights, self.row_chunks)
            )
        ]
        return torch.cat(outputs, dim=-1)


def expert_tp_closed_reason(parallel) -> str | None:
    """Why the routed experts cannot be split across the tensor-parallel
    ranks of ``parallel`` (a FusedMoEParallelConfig with tp_size > 1), or
    None."""
    if (
        parallel.use_ep
        or parallel.dp_size != 1
        or parallel.pcp_size != 1
        or parallel.is_sequence_parallel
        or parallel.enable_eplb
        or parallel.use_all2all_kernels
    ):
        return (
            "EXL3 tensor-parallel routed experts support no expert, data, "
            "context or sequence parallelism, EPLB or all-to-all dispatch"
        )
    return None


def tensor_parallel_expert_split(moe, prefix: str) -> tuple[int, slice]:
    """(whole intermediate, this rank's columns) of every routed expert of
    the MoE layer ``prefix`` at MoE tensor parallel size > 1 (vLLM without
    --enable-expert-parallel). Each rank holds whole H128 blocks of every
    expert's intermediate, in chunks as even as the blocks allow with the
    larger chunks first (block_shard); rank r of layer L holds chunk
    (r + L) mod the tensor parallel size, so the larger chunks move between
    ranks from layer to layer. Gate and up keep the chunk's output columns,
    down its input rows. Routing is unchanged: every rank holds every expert,
    and the MoE's all-reduce sums the ranks' partial outputs."""
    from vllm.model_executor.models.utils import extract_layer_index

    parallel = moe.moe_parallel_config
    reason = expert_tp_closed_reason(parallel)
    if reason is not None:
        raise ValueError(reason)
    intermediate = moe.intermediate_size
    size, rank = parallel.tp_size, parallel.tp_rank
    layer_index = extract_layer_index(prefix)

    def chunk(rotation: int) -> slice:
        return block_shard(intermediate, intermediate, rank, size, rotation)

    columns = chunk(layer_index)
    logger.info_once(
        "EXL3 routed experts of %s are split across %d tensor-parallel ranks "
        "(no expert parallelism): rank %d holds %s of each expert's %d "
        "intermediate columns at layer index mod %d = 0..%d.",
        re.sub(r"\d+", "<L>", prefix),
        size,
        rank,
        tuple(part.stop - part.start for part in map(chunk, range(size))),
        intermediate,
        size,
        size - 1,
    )
    logger.debug(
        "EXL3 routed experts of %s: rank %d holds intermediate columns [%d, %d)",
        prefix,
        rank,
        columns.start,
        columns.stop,
    )
    return intermediate, columns


class Exl3MoEMethod(FusedMoEMethodBase):
    # Routed experts at MoE tensor parallel size > 1: (whole intermediate,
    # this rank's columns of every expert) from tensor_parallel_expert_split;
    # None with whole experts per rank.
    expert_split = None

    def __init__(self, config, prefix, moe):
        super().__init__(moe)
        self.config = config
        self.prefix = prefix
        if moe.moe_parallel_config.tp_size != 1:
            self.expert_split = tensor_parallel_expert_split(moe, prefix)
        if moe.has_bias or moe.activation != MoEActivation.SILU:
            raise ValueError("EXL3 experts require bias-free SiLU gate/up/down")
        if moe.is_lora_enabled:
            raise ValueError("EXL3 experts do not support LoRA")
        if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
            raise ValueError("EXL3 experts support SiLU with an optional clamp only")
        # SwiGLU clamp before SiLU(gate) * up: gate from above, up on both
        # sides (0: none). The native operators take it as swiglu_limit.
        self.swiglu_limit = 0.0 if moe.swiglu_limit is None else float(moe.swiglu_limit)
        if not self.swiglu_limit >= 0.0:
            raise ValueError(f"Invalid EXL3 SwiGLU limit: {moe.swiglu_limit}")
        # Set by set_moe_prefill_v2 before the weights load: this layer's
        # prefill calls may take moe_out_prefill_v2 (its gate decides at load).
        self.prefill_v2 = False

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        """Under tensor parallelism a rank's intermediate is its chunk of
        whole H128 blocks (tensor_parallel_expert_split), not an even share."""
        hidden_size, intermediate = super().maybe_roundup_sizes(
            hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config
        )
        if self.expert_split is not None:
            columns = self.expert_split[1]
            intermediate = columns.stop - columns.start
        return hidden_size, intermediate

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16 or layer.apply_router_weight_on_input:
            raise ValueError("EXL3 MoE requires BF16 and routing weights after down")
        assert self.config.manifest is not None
        if self.expert_split is None:
            intermediate, columns = intermediate_size_per_partition, None
        else:
            intermediate, columns = self.expert_split
            if intermediate_size_per_partition != columns.stop - columns.start:
                raise ValueError(
                    "EXL3 tensor-parallel experts need this rank's intermediate "
                    f"{columns.stop - columns.start}, not "
                    f"{intermediate_size_per_partition}"
                )
        self.intermediate_size = intermediate_size_per_partition
        self.local_experts = num_experts
        self.hidden_size = hidden_size

        self.global_expert_ids = [-1] * num_experts
        self.nonlocal_keys = []
        # (local expert, matrix, (logical in, logical out, input slice, output
        # slices)) in allocation order.
        local_matrices = []
        for expert in range(layer.global_num_experts):
            local = layer._map_global_expert_id_to_local_expert_id(expert)
            for role in ("gate_proj", "up_proj", "down_proj"):
                prefix = self.config.layout.expert_matrix(
                    self.config, self.prefix, expert, role
                )
                matrix = self.config.manifest.matrices.get(prefix)
                if matrix is None:
                    raise ValueError(f"Missing EXL3 expert matrix: {prefix}")
                k, n = hidden_size, intermediate
                if role == "down_proj":
                    k, n = n, k
                matrix.validate_logical(k, n)
                if local < 0:
                    self.nonlocal_keys.extend(matrix.keys)
                    continue
                self.global_expert_ids[local] = expert
                whole_in = slice(0, matrix.padded_in)
                whole_out = [slice(0, matrix.padded_out)]
                if columns is None:
                    part = (k, n, whole_in, whole_out)
                elif role == "down_proj":
                    # Down keeps this rank's input rows and every output.
                    part = (columns.stop - columns.start, n, columns, whole_out)
                else:
                    # Gate and up keep every input and this rank's outputs.
                    part = (k, columns.stop - columns.start, whole_in, [columns])
                local_matrices.append((local, matrix, part))
        views = [None] * len(local_matrices)
        if envs.VLLM_EXL3_EXPERTS_CONTIGUOUS and local_matrices:
            views = expert_buffer_views(
                [matrix for _, matrix, _ in local_matrices],
                [(part[2], part[3]) for _, _, part in local_matrices],
            )
            logger.info_once(
                "EXL3 routed experts in one buffer per MoE layer and tensor kind "
                "(VLLM_EXL3_EXPERTS_CONTIGUOUS): trellis, suh and svh of every "
                "local expert matrix are views."
            )
        weights = [[] for _ in range(num_experts)]
        for (local, matrix, part), view in zip(local_matrices, views):
            weights[local].append(Exl3PackedWeight(matrix, *part, views=view))
        if any(len(expert) != 3 for expert in weights):
            raise ValueError("EXL3 expert assignment is incomplete or duplicated")
        layer.exl3_experts = nn.ModuleList(nn.ModuleList(w) for w in weights)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make()

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        pointers, metadata = [], []
        for expert in layer.exl3_experts:
            ptrs, meta = [], []
            for weight in expert:
                if not weight.loaded or not all(p.is_cuda for p in weight.parameters()):
                    raise ValueError("EXL3 expert weights must be loaded on CUDA")
                ptrs.append(
                    [
                        weight.trellis.data_ptr(),
                        weight.suh.data_ptr(),
                        weight.svh.data_ptr(),
                    ]
                )
                meta.append(
                    [
                        weight.padded_in,
                        weight.padded_out,
                        weight.matrix.bits_x2,
                        weight.matrix.codebook,
                    ]
                )
            pointers.append(ptrs)
            metadata.append(meta)
        device = layer.exl3_experts[0][0].trellis.device
        layer.register_buffer(
            "exl3_ptrs", torch.tensor(pointers, dtype=torch.int64, device=device)
        )
        layer.register_buffer(
            "exl3_meta", torch.tensor(metadata, dtype=torch.int64, device=device)
        )
        from vllm.sm80.kernels.exl3 import capture_points

        capture_points.register_moe(layer.exl3_ptrs, self.prefix)
        if layer.expert_map is None:
            layer.register_buffer(
                "exl3_expert_map",
                torch.arange(
                    layer.global_num_experts, dtype=torch.int32, device=device
                ),
            )
        else:
            layer.register_buffer("exl3_expert_map", layer.expert_map)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
        )
        max_rows = dispatched_rows(prepare, self.moe)
        # Its workspace holds one moe_out launch, which may be a row chunk.
        moe_out_rows = moe_out_launch_rows(max_rows, self.hidden_size)
        workspace = exl3_ops.moe_workspace_size(
            moe_out_rows,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
        )
        moe_v2_rows = moe_v2_bits_x2 = 0
        if envs.VLLM_EXL3_MOE_V2:
            moe_v2_rows, moe_v2_bits_x2, workspace = self._moe_v2(
                layer, metadata, max_rows, workspace
            )
        # VLLM_EXL3_MOE_V2_DRAFT: the DSpark drafter's layers, with kernels of
        # their own (ops.MOE_V2_DRAFT_CONFIGS), never folded.
        moe_v2_draft = False
        if not moe_v2_rows and envs.VLLM_EXL3_MOE_V2_DRAFT:
            moe_v2_rows, moe_v2_bits_x2, workspace = self._moe_v2(
                layer, metadata, max_rows, workspace, draft=True
            )
            moe_v2_draft = bool(moe_v2_rows)
        # VLLM_EXL3_MOE_V2_RING: the target layers' moe_out_v2 calls of the covered
        # configuration take moe_out_v2_try (before moe_out_v2_fold).
        moe_v2_try = False
        if moe_v2_rows and not moe_v2_draft:
            moe_v2_try, workspace = self._moe_v2_try(
                layer, moe_v2_rows, moe_v2_bits_x2, workspace
            )
        prefill_decode_len = 0
        if envs.VLLM_EXL3_MOE_PREFILL_TC:
            prefill_decode_len, workspace = self._moe_prefill(
                layer, metadata, max_rows, workspace
            )
        prefill_v2_bits_x2 = prefill_v2_decode_len = 0
        if self.prefill_v2:
            prefill_v2_bits_x2, prefill_v2_decode_len, workspace = self._moe_prefill_v2(
                layer, metadata, max_rows, workspace
            )
        if moe_out_rows < max_rows:
            logger.info_once(
                "EXL3 moe_out runs MoE calls of more than %d rows in chunks of "
                "at most %d (moe_routed_gemm/exl3_chunk_rows); its workspace "
                "holds %d rows instead of %d.",
                moe_out_rows,
                moe_out_rows,
                moe_out_rows,
                max_rows,
            )
        experts = Exl3Experts(
            self.moe,
            self.moe_quant_config,
            self.intermediate_size,
            layer.exl3_expert_map,
            workspace,
            max_rows,
            moe_out_rows=moe_out_rows,
            moe_v2_rows=moe_v2_rows,
            moe_v2_bits_x2=moe_v2_bits_x2,
            moe_v2_fold=bool(moe_v2_rows)
            and not moe_v2_draft
            and envs.VLLM_EXL3_MOE_V2_FOLD,
            moe_v2_draft=moe_v2_draft,
            moe_v2_try=moe_v2_try,
            prefill_decode_len=prefill_decode_len,
            prefill_v2_bits_x2=prefill_v2_bits_x2,
            prefill_v2_decode_len=prefill_v2_decode_len,
            swiglu_limit=self.swiglu_limit,
        )
        assert prepare is not None
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)

    def _moe_v2(
        self, layer, metadata, max_rows: int, workspace: int, draft: bool = False
    ):
        """Rows up to which this layer's calls take moe_out_v2, the bits_x2
        of its routed matrices, and a workspace size that fits both operators
        (0 rows: the gate is closed). ``draft``: the configurations
        VLLM_EXL3_MOE_V2_DRAFT opens (the DSpark drafter's layers)."""
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        device = layer.exl3_ptrs.device
        config = get_current_vllm_config_or_none()
        reason = exl3_ops.moe_v2_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
            draft=draft,
        )
        if reason is None:
            reason = _moe_partition_closed_reason(
                config, self.hidden_size, self.local_experts, layer.global_num_experts
            )
        if reason is not None:
            if draft:
                logger.info_once(
                    "VLLM_EXL3_MOE_V2_DRAFT is set but the gate is closed: %s; "
                    "these MoE layers keep moe_out.",
                    reason,
                )
            else:
                logger.info_once(
                    "VLLM_EXL3_MOE_V2 is set but the gate is closed: %s; these MoE "
                    "layers keep moe_out.",
                    reason,
                )
            return 0, 0, workspace
        if draft:
            exl3_ops.moe_v2_draft_force_rows(envs.VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS)
        # One bit width for every routed matrix of the layer (the gate checked).
        bits_x2 = metadata[0][0][2]
        rows = min(max_rows, exl3_ops.MOE_V2_MAX_ROWS)
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_v2(
                rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_v2(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            bits_x2,
            self.swiglu_limit,
        )
        if draft:
            logger.info_once(
                "EXL3 MoE v2 drafter kernels active (VLLM_EXL3_MOE_V2_DRAFT): "
                "tensor-core routed experts of the DSpark drafter (bits_x2 %d, "
                "16-row tiles; row tile floor %d) for MoE calls of 1..%d rows; "
                "larger calls keep moe_out.",
                bits_x2,
                envs.VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS,
                rows,
            )
        else:
            logger.info_once(
                "EXL3 MoE v2 kernel active (VLLM_EXL3_MOE_V2): tensor-core routed "
                "experts (bits_x2 %d) for MoE calls of 1..%d rows; larger calls "
                "keep moe_out.",
                bits_x2,
                rows,
            )
        if draft:
            if envs.VLLM_EXL3_MOE_V2_FOLD:
                logger.info_once(
                    "VLLM_EXL3_MOE_V2_FOLD does not apply to the DSpark drafter's "
                    "layers (VLLM_EXL3_MOE_V2_DRAFT): their moe_out_v2 calls keep "
                    "five launches."
                )
        elif envs.VLLM_EXL3_MOE_V2_FOLD:
            exl3_ops.require_moe_v2_fold()
            exl3_ops.warmup_moe_v2(
                layer.exl3_ptrs,
                layer.exl3_meta,
                layer.exl3_expert_map,
                self.global_expert_ids,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                bits_x2,
                self.swiglu_limit,
                fold=True,
            )
            logger.info_once(
                "EXL3 MoE v2 folded (VLLM_EXL3_MOE_V2_FOLD): moe_out_v2_fold runs "
                "the activation and combine inside the expert GEMM launches "
                "(three launches per call, the same output bits).",
            )
        return rows, bits_x2, workspace

    def _moe_v2_try(self, layer, rows: int, bits_x2: int, workspace: int):
        """(True, a workspace size that fits moe_out_v2_try too) when
        VLLM_EXL3_MOE_V2_RING is set and the layer is its configuration, after
        warming the operator; else (False, workspace)."""
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        if not exl3_ops.moe_v2_try_open(
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            bits_x2,
        ):
            return False, workspace
        exl3_ops.require_moe_v2_try()
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_v2_try(
                rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_v2(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            bits_x2,
            self.swiglu_limit,
            trial=True,
        )
        logger.info_once(
            "EXL3 MoE v2 ring kernel active (VLLM_EXL3_MOE_V2_RING): moe_out_v2_try "
            "for MoE calls of 1..%d rows of the covered layers (the same output "
            "bits as moe_out_v2).",
            rows,
        )
        return True, workspace

    def _moe_prefill(self, layer, metadata, max_rows: int, workspace: int):
        """The most tokens one request has in a decode step, above which a
        step is a prefill step whose calls of more than 64 rows take
        moe_out_prefill, and a workspace size that fits every operator (0:
        the gate is closed)."""
        from vllm.config import get_current_vllm_config
        from vllm.sm80.kernels.exl3 import ops as exl3_ops
        from vllm.sm80.kernels.exl3.step_phase import max_decode_query_len

        device = layer.exl3_ptrs.device
        reason = exl3_ops.moe_prefill_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
        )
        if reason is None and max_rows < exl3_ops.MOE_PREFILL_MIN_ROWS:
            reason = f"MoE calls have at most {max_rows} rows"
        if reason is None:
            # Host copy of the layer's pointer table: one device sync, at load.
            reason = _unaligned_trellis_reason(layer.exl3_ptrs.tolist())
        vllm_config = get_current_vllm_config()
        if reason is None:
            reason = _prefill_layout_closed_reason(vllm_config, self.local_experts)
        if reason is not None:
            logger.info_once(
                "VLLM_EXL3_MOE_PREFILL_TC is set but the gate is closed: %s; "
                "these MoE layers keep moe_out in prefill steps.",
                reason,
            )
            return 0, workspace
        rows = min(max_rows, exl3_ops.MOE_PREFILL_MAX_ROWS)
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_prefill(
                rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_prefill(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.swiglu_limit,
        )
        decode_len = max_decode_query_len(vllm_config)
        logger.info_once(
            "EXL3 MoE prefill kernel active (VLLM_EXL3_MOE_PREFILL_TC): "
            "tensor-core routed experts for MoE calls of %d..%d rows in prefill "
            "steps (a request scheduled more than %d tokens); decode steps, "
            "calls of at most %d rows and larger calls keep their kernels.",
            exl3_ops.MOE_PREFILL_MIN_ROWS,
            rows,
            decode_len,
            exl3_ops.MOE_PREFILL_MIN_ROWS - 1,
        )
        compilation = getattr(vllm_config, "compilation_config", None)
        graph_max = getattr(compilation, "max_cudagraph_capture_size", None) or 0
        if graph_max >= exl3_ops.MOE_PREFILL_MIN_ROWS:
            logger.info_once(
                "VLLM_EXL3_MOE_PREFILL_TC: CUDA graphs are captured up to %d "
                "tokens; prefill steps padded to a captured size replay moe_out.",
                graph_max,
            )
        return decode_len, workspace

    def _moe_prefill_v2(self, layer, metadata, max_rows: int, workspace: int):
        """The bits_x2 of this layer's routed matrices, the most tokens one
        request has in a decode step, above which a step is a prefill step
        whose calls of MOE_PREFILL_MIN_ROWS..MOE_PREFILL_V2_MAX_ROWS rows take
        moe_out_prefill_v2, and a workspace size that fits every operator
        (bits_x2 0: the gate is closed)."""
        from vllm.config import get_current_vllm_config
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        device = layer.exl3_ptrs.device
        reason = exl3_ops.moe_prefill_v2_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
        )
        if reason is None and max_rows < exl3_ops.MOE_PREFILL_MIN_ROWS:
            reason = f"MoE calls have at most {max_rows} rows"
        if reason is None:
            # Host copy of the layer's pointer table: one device sync, at load.
            reason = _unaligned_trellis_reason(layer.exl3_ptrs.tolist())
        vllm_config = get_current_vllm_config()
        if reason is None:
            reason = _prefill_v2_layout_closed_reason(vllm_config)
        if reason is not None:
            logger.info_once(
                "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC is set but the gate is closed: "
                "%s; these MoE layers keep their kernels in prefill steps.",
                reason,
            )
            return 0, 0, workspace
        exl3_ops.require_moe_prefill_v2()
        # One bit width for every routed matrix of the layer (the gate checked).
        bits_x2 = metadata[0][0][2]
        rows = min(max_rows, exl3_ops.MOE_PREFILL_V2_MAX_ROWS)
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_prefill_v2(
                rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_prefill_v2(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            bits_x2,
            self.swiglu_limit,
        )
        decode_len = _prefill_v2_decode_len(vllm_config)
        logger.info_once(
            "EXL3 MoE prefill v2 tensor-core path active "
            "(VLLM_DEEPSEEK_V41_MOE_PREFILL_TC): routed experts (bits_x2 6 or 4) "
            "for MoE calls of %d..%d rows in prefill steps (a request scheduled "
            "more than %d tokens); decode steps, graph captures, calls of at "
            "most %d rows and larger calls keep their kernels.",
            exl3_ops.MOE_PREFILL_MIN_ROWS,
            rows,
            decode_len,
            exl3_ops.MOE_PREFILL_MIN_ROWS - 1,
        )
        return bits_x2, decode_len, workspace

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.exl3_ptrs,
            w2=layer.exl3_meta,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=False,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.exl3_expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


class Exl3Experts(mk.FusedMoEExpertsModular):
    # VLLM_EXL3_MOE_V2_FOLD: the moe_out_v2 calls take moe_out_v2_fold.
    moe_v2_fold = False
    # VLLM_EXL3_MOE_V2_DRAFT: the moe_out_v2 calls are the DSpark drafter's
    # (bits_x2 8 kernels of their own, never folded).
    moe_v2_draft = False
    # VLLM_EXL3_MOE_V2_RING: the moe_out_v2 calls take moe_out_v2_try.
    moe_v2_try = False

    def __init__(
        self,
        moe,
        quant,
        intermediate_size,
        expert_map,
        workspace,
        max_rows,
        moe_out_rows=None,
        moe_v2_rows=0,
        moe_v2_bits_x2=0,
        moe_v2_fold=False,
        moe_v2_draft=False,
        moe_v2_try=False,
        prefill_decode_len=0,
        prefill_v2_bits_x2=0,
        prefill_v2_decode_len=0,
        swiglu_limit=0.0,
    ):
        super().__init__(moe, quant)
        self.intermediate_size = intermediate_size
        self.expert_map = expert_map
        self.workspace_elements = workspace
        self.max_rows = max_rows
        # moe_out takes at most this many rows per launch (its workspace size;
        # moe_routed_gemm/exl3_chunk_rows); larger calls run in chunks.
        self.moe_out_rows = max_rows if moe_out_rows is None else moe_out_rows
        # Calls of 1..moe_v2_rows rows take moe_out_v2 (VLLM_EXL3_MOE_V2) with
        # the layer's routed-matrix bits_x2.
        self.moe_v2_rows = moe_v2_rows
        self.moe_v2_bits_x2 = moe_v2_bits_x2
        self.moe_v2_fold = moe_v2_fold
        self.moe_v2_draft = moe_v2_draft
        self.moe_v2_try = moe_v2_try
        # Nonzero (VLLM_EXL3_MOE_PREFILL_TC with its gate open): calls of more
        # than 64 rows in a step that schedules some request more than this
        # many tokens take moe_out_prefill.
        self.prefill_decode_len = prefill_decode_len
        # Nonzero (VLLM_DEEPSEEK_V41_MOE_PREFILL_TC with its gate open): calls
        # of MOE_PREFILL_MIN_ROWS..MOE_PREFILL_V2_MAX_ROWS rows in a step that
        # schedules some request more than prefill_v2_decode_len tokens take
        # moe_out_prefill_v2 with the layer's routed-matrix bits_x2.
        self.prefill_v2_bits_x2 = prefill_v2_bits_x2
        self.prefill_v2_decode_len = prefill_v2_decode_len
        # SwiGLU clamp the native operators apply (0: none).
        self.swiglu_limit = swiglu_limit

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        from vllm.platforms import current_platform

        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config):
        # Whole experts per rank, or every expert split across the
        # tensor-parallel ranks (tensor_parallel_expert_split).
        return (
            moe_parallel_config.tp_size == 1
            or expert_tp_closed_reason(moe_parallel_config) is None
        )

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return (
            w1.shape[0],
            a1.shape[0],
            2 * self.intermediate_size,
            a1.shape[-1],
            topk_ids.shape[1],
        )

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        # workspace2 uses BF16 storage viewed as FP32 by the native operator.
        return (0,), (2 * self.workspace_elements,), (M, K)

    def finalize_weight_and_reduce_impl(self):
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        from vllm.sm80.kernels.exl3.ops import (
            moe_out,
            moe_out_v2,
            moe_out_v2_fold,
            moe_out_v2_try,
        )

        rows = hidden_states.shape[0]
        if rows > self.max_rows:
            raise ValueError("EXL3 MoE exceeds configured dispatched row capacity")
        if (
            apply_router_weight_on_input
            or a1q_scale is not None
            or a2_scale is not None
        ):
            raise ValueError("EXL3 MoE requires unquantized, unweighted BF16 inputs")
        if 0 < rows <= self.moe_v2_rows:
            operator = partial(
                moe_out_v2_try
                if self.moe_v2_try
                else moe_out_v2_fold
                if self.moe_v2_fold
                else moe_out_v2,
                bits_x2=self.moe_v2_bits_x2,
            )
        else:
            operator = moe_out
        if self.prefill_decode_len:
            operator = self._prefill_operator(rows, operator)
        if self.prefill_v2_bits_x2:
            operator = self._prefill_v2_operator(rows, operator)
        # Row-indexed arguments; moe_out takes at most moe_out_rows of them per
        # launch. It computes every row from its own input and routes alone, so
        # row chunks give the bits of the whole call.
        rows_args = (hidden_states, topk_ids, topk_weights, output)
        if operator is moe_out and rows > self.moe_out_rows:
            step = self.moe_out_rows
            chunks = [
                tuple(t[start : start + step] for t in rows_args)
                for start in range(0, rows, step)
            ]
        else:
            chunks = [rows_args]
        for x, ids, weights, out in chunks:
            operator(
                x,
                ids,
                weights,
                w1,
                w2,
                self.expert_map if expert_map is None else expert_map,
                self.intermediate_size,
                out,
                workspace2.view(torch.float32),
                swiglu_limit=self.swiglu_limit,
            )
        if capture.active:
            from vllm.sm80.kernels.exl3 import capture_points

            capture_points.moe(
                operator,
                hidden_states,
                topk_ids,
                topk_weights,
                w1,
                w2,
                self.expert_map if expert_map is None else expert_map,
                self.intermediate_size,
                self.swiglu_limit,
                output,
            )

    def _prefill_operator(self, rows: int, operator):
        """moe_out_prefill for a call of MOE_PREFILL_MIN_ROWS..MOE_PREFILL_MAX_ROWS
        rows in a prefill step (the phase the scheduler gave the step, never
        the row count); otherwise ``operator``."""
        from vllm.sm80.kernels.exl3.ops import (
            MOE_PREFILL_MAX_ROWS,
            MOE_PREFILL_MIN_ROWS,
            moe_out_prefill,
        )
        from vllm.sm80.kernels.exl3.step_phase import prefill_step

        if MOE_PREFILL_MIN_ROWS <= rows <= MOE_PREFILL_MAX_ROWS and prefill_step(
            self.prefill_decode_len
        ):
            return moe_out_prefill
        return operator

    def _prefill_v2_operator(self, rows: int, operator):
        """moe_out_prefill_v2 for a call of
        MOE_PREFILL_MIN_ROWS..MOE_PREFILL_V2_MAX_ROWS rows in a prefill step
        (the phase the scheduler gave the step, never the row count);
        otherwise ``operator``."""
        from vllm.sm80.kernels.exl3.ops import (
            MOE_PREFILL_MIN_ROWS,
            MOE_PREFILL_V2_MAX_ROWS,
            moe_out_prefill_v2,
        )
        from vllm.sm80.kernels.exl3.step_phase import prefill_step

        if MOE_PREFILL_MIN_ROWS <= rows <= MOE_PREFILL_V2_MAX_ROWS and prefill_step(
            self.prefill_v2_decode_len
        ):
            return partial(moe_out_prefill_v2, bits_x2=self.prefill_v2_bits_x2)
        return operator


def _prefill_layout_closed_reason(vllm_config, local_experts: int) -> str | None:
    """Qwen's prefill partition gate and its sequential-drafter requirement."""
    reason = _moe_partition_closed_reason(vllm_config, 2560, local_experts, 512)
    if reason is not None:
        return reason
    spec = vllm_config.speculative_config
    if spec is not None and (
        getattr(spec, "parallel_drafting", False)
        or getattr(spec, "method", None) in ("dflash", "dspark")
    ):
        return f"speculative method {spec.method} drafts with parallel slots"
    return None


def _prefill_v2_layout_closed_reason(vllm_config) -> str | None:
    """Why moe_out_prefill_v2 must not serve this engine, or None: it is
    validated at tensor parallel size 4 without pipeline parallelism."""
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    if layout != (4, 1):
        return f"(tensor, pipeline) parallel size {layout}, not (4, 1)"
    return None


def _prefill_v2_decode_len(vllm_config) -> int:
    """The most tokens one request has in any forward pass of a decode step,
    above which the prefill-step rule (step_phase) calls a step a prefill
    step: 1 + k for the target, and for a drafter with parallel drafting slots
    (DFlash, DSpark) its decode pass, 1 + k verified tokens plus k slots
    (the per-request bound of step_phase.max_decode_step_rows)."""
    from vllm.sm80.kernels.exl3.step_phase import max_decode_query_len

    decode_len = max_decode_query_len(vllm_config)
    spec = vllm_config.speculative_config
    if spec is not None and (
        getattr(spec, "parallel_drafting", False)
        or getattr(spec, "method", None) in ("dflash", "dspark")
    ):
        return 2 * decode_len - 1
    return decode_len


def _unaligned_trellis_reason(pointers) -> str | None:
    """Why moe_out_prefill must not read these experts, or None: it loads
    every expert trellis in 16-byte units and stops the device on a trellis
    base that is not 16-byte aligned. ``pointers`` is the layer's pointer
    table as a host list: per local expert and projection (gate, up, down)
    the [trellis, suh, svh] addresses."""
    roles = ("gate_proj", "up_proj", "down_proj")
    for local, expert in enumerate(pointers):
        for role, (trellis, _suh, _svh) in zip(roles, expert):
            if trellis % 16:
                return f"local expert {local} {role} trellis is not 16-byte aligned"
    return None


def moe_out_launch_rows(max_rows: int, hidden: int) -> int:
    """Most rows one moe_out launch takes for calls of up to ``max_rows``
    rows: the moe_routed_gemm/exl3_chunk_rows row of the sm80 tuning table at
    this hidden size when it is nonzero and smaller (larger calls then run in
    row chunks), else ``max_rows``. Without a row, 0: whole calls."""
    chunk_rows = tuning.lookup(
        "moe_routed_gemm", "exl3_chunk_rows", {"hidden": int(hidden)}, 0
    )
    if chunk_rows < 0:
        raise ValueError(
            f"moe_routed_gemm/exl3_chunk_rows must be 0 or positive, not {chunk_rows}"
        )
    return min(max_rows, chunk_rows) if chunk_rows else max_rows


def dispatched_rows(prepare, moe) -> int:
    """Most rows the experts of one rank receive in one MoE call."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    parallel = moe.moe_parallel_config
    rows = moe.max_num_tokens
    if isinstance(prepare, MoEPrepareAndFinalizeNoDPEPModular):
        # No dispatch (TP + EP without DP): each rank's experts see only the
        # rows of its own forward.
        return rows
    if not isinstance(prepare, MoEPrepareAndFinalizeNaiveDPEPModular):
        raise ValueError("EXL3 EP requires allgather_reducescatter dispatch")
    if parallel.is_sequence_parallel:
        # The EP group gathers every rank's padded sequence-parallel share.
        return parallel.ep_size * -(-rows // parallel.sp_size)
    # The DP (or PCP) group gathers every replica's rows.
    return rows * parallel.dp_size * parallel.pcp_size
