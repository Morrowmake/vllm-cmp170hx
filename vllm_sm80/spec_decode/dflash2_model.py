# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 drafter classes of the sm_80 fork (GLM's drafter; Qwen may use it).

The fork plugin (``vllm_sm80.plugin``) registers :class:`DFlash2Qwen3ForCausalLM`
in vLLM's model registry in place of the upstream class in
``vllm/model_executor/models/qwen3_dflash2.py`` (``DFlash2DraftModel``). The
classes here subclass the upstream ones and keep their names, so the registry
reports the same model information and checkpoints load into the same
parameter names.

What they add, each only while its switch is on (otherwise the modules are
upstream's):

- the load-following draft width (``adaptive_k.draft_by_load``): the drafted
  block length changes per step, so the grouped convolutions of every DFlash2
  decoder layer read it from a device scalar (``block_size_tensor``, set by
  the speculator) instead of a constant, through the
  ``vllm::dflash2_grouped_conv_dyn`` op. The upstream-built convolutions are
  given the fork class (:meth:`DFlashGroupedConv.adopt`), so their modules and
  parameters are exactly the upstream ones. LiLiCorr builds the same decoder
  layers (``lilicorr_model.py``);
- ``VLLM_DFLASH2_SELECTOR_SHARD`` with TP > 1: the candidate selector
  keeps a contiguous vocab shard of its two codebooks per tensor-parallel rank
  and rebuilds the looked-up rows with one all-reduce (exact: each row has one
  non-zero contributor). The PP draft tail builds its selector the same way
  (:func:`build_candidate_selector`).
- the DFlash-family additions of ``dflash_draft.py`` (the fitted RoPE cache,
  the folded aux fc input).
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.models import qwen3_dflash2
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv
from vllm.model_executor.models.utils import maybe_prefix
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_sm80.spec_decode.dflash_draft import DFlashDraftMixin

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Grouped convolution with a runtime block length
# ---------------------------------------------------------------------------


@triton.jit
def _dflash2_grouped_conv_dyn_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    output_ptr,
    block_size_ptr,
    x_stride_row,
    delta_stride_row,
    delta_stride_tap,
    base_stride_tap,
    output_stride_row,
    NUM_CHANNELS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    ELEMENT_BLOCK: tl.constexpr,
) -> None:
    """_dflash2_grouped_conv_kernel with the per-request block length read
    from device memory (load-following draft width): the same arithmetic,
    only the block length is a runtime value instead of a constant."""
    row = tl.program_id(0) // triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    col_block = tl.program_id(0) % triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    channels = col_block * ELEMENT_BLOCK + tl.arange(0, ELEMENT_BLOCK)
    mask = channels < NUM_CHANNELS
    groups = channels // GROUP_SIZE
    position = row % tl.load(block_size_ptr)

    delta_row = delta_ptr + row * delta_stride_row
    x_row = x_ptr + row * x_stride_row
    accumulator = (
        tl.load(base_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta_row + groups, mask=mask, other=0.0).to(tl.float32)
    ) * tl.load(x_row + channels, mask=mask, other=0.0).to(tl.float32)

    for tap in tl.static_range(1, TAPS):
        tap_mask = mask & (position >= tap)
        coefficient = tl.load(
            base_ptr + tap * base_stride_tap + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32) + tl.load(
            delta_row + tap * delta_stride_tap + groups,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + (row - tap) * x_stride_row + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += coefficient * x

    tl.store(
        output_ptr + row * output_stride_row + channels,
        accumulator,
        mask=mask,
    )


def dflash2_grouped_conv_dyn_impl(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    num_rows, num_channels = x.shape
    output = torch.empty_like(x)
    if num_rows == 0:
        return output

    element_block = 1024 if num_rows >= 128 and num_channels % 1024 == 0 else 512
    grid = (num_rows * triton.cdiv(num_channels, element_block),)
    _dflash2_grouped_conv_dyn_kernel[grid](
        x,
        delta,
        base,
        output,
        block_size,
        x.stride(0),
        delta.stride(0),
        delta.stride(1),
        base.stride(0),
        output.stride(0),
        NUM_CHANNELS=num_channels,
        GROUP_SIZE=group_size,
        TAPS=base.shape[0],
        ELEMENT_BLOCK=element_block,
        num_warps=4,
    )
    return output


def dflash2_grouped_conv_dyn_fake(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="dflash2_grouped_conv_dyn",
    op_func=dflash2_grouped_conv_dyn_impl,
    fake_impl=dflash2_grouped_conv_dyn_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _dynamic_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size_tensor: torch.Tensor,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Upstream ``_grouped_conv`` with the block length in a device scalar."""
    if hidden_states.is_cuda:
        return torch.ops.vllm.dflash2_grouped_conv_dyn(
            hidden_states, delta, base, block_size_tensor, group_size
        )
    return _grouped_conv(
        hidden_states,
        delta,
        base,
        int(block_size_tensor.item()),
        num_groups,
        group_size,
        taps,
    )


def _load_following_width(vllm_config: VllmConfig) -> bool:
    """Whether the drafter drafts per-step widths
    (adaptive_k.draft_by_load), so the conv's block length varies."""
    from vllm_sm80.spec_decode.dflash2_speculator import load_following_draft_widths

    return bool(load_following_draft_widths(vllm_config))


class DFlashGroupedConv(qwen3_dflash2.DFlashGroupedConv):
    """Grouped convolution whose block length is the device scalar
    ``block_size_tensor`` (load-following draft width)."""

    block_size_tensor: torch.Tensor

    @classmethod
    def adopt(cls, conv: qwen3_dflash2.DFlashGroupedConv) -> "DFlashGroupedConv":
        """Give an upstream-built convolution this class and its block-length
        buffer (non-persistent: not in the state dict). The block length (bonus
        + drafted tokens per request) changes per step, so the conv reads it
        from this scalar (set by the drafter before each forward and before
        each width's graph capture) rather than a compiled constant."""
        assert type(conv) is qwen3_dflash2.DFlashGroupedConv, type(conv)
        conv.__class__ = cls
        conv.register_buffer(
            "block_size_tensor",
            torch.tensor([conv.block_size], dtype=torch.int32),
            persistent=False,
        )
        return conv

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _dynamic_grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size_tensor,
            self.num_groups,
            self.group_size,
            self.taps,
        )


def adopt_grouped_convs(model: nn.Module, vllm_config: VllmConfig) -> None:
    """With the load-following draft width, give every grouped convolution of
    ``model`` the runtime block length (:meth:`DFlashGroupedConv.adopt`).
    Without it the convolutions stay upstream's."""
    if not _load_following_width(vllm_config):
        return
    for module in model.modules():
        if type(module) is qwen3_dflash2.DFlashGroupedConv:
            DFlashGroupedConv.adopt(module)


# ---------------------------------------------------------------------------
# Candidate selector with vocab-sharded codebooks
# ---------------------------------------------------------------------------


def _predecessor_ids(
    candidate_ids: torch.Tensor, anchor_token_ids: torch.Tensor, top_k: int
) -> torch.Tensor:
    return torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )


def _combine_edges(
    predecessors: torch.Tensor,
    successors: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


def selector_vocab_shard(
    vocab_size: int, tp_rank: int, tp_size: int
) -> tuple[int, int, int]:
    """(first row, end row, padded rows per rank) of a contiguous vocab shard."""
    per_rank = -(-vocab_size // tp_size)
    start = min(vocab_size, tp_rank * per_rank)
    end = min(vocab_size, start + per_rank)
    return start, end, per_rank


def local_codebook_rows(
    table: torch.Tensor, ids: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """Rows of a vocab shard for global ids, zero where another rank owns
    the id. Summing this over all ranks gives ``full_table[ids]`` exactly."""
    owned = (ids >= start) & (ids < end)
    local = torch.where(owned, ids - start, torch.zeros_like(ids))
    return table[local].masked_fill(~owned[..., None], 0)


class CandidateSelector(qwen3_dflash2.CandidateSelector):
    """The DFlash2 candidate selector with each tensor-parallel rank holding
    a contiguous vocab shard of both codebooks (padded to the same rows)."""

    shard_vocab = True

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end, rows = selector_vocab_shard(
            vocab_size, get_tensor_model_parallel_rank(), tp_size
        )
        elem = torch.tensor([], dtype=params_dtype).element_size()
        logger.info_once(
            "VLLM_DFLASH2_SELECTOR_SHARD: selector codebooks %d -> %d "
            "rows per rank (%.1f -> %.1f MiB for both).",
            vocab_size,
            rows,
            2 * vocab_size * rank * elem / 2**20,
            2 * rows * rank * elem / 2**20,
        )
        # The upstream constructor sizes both codebooks by its vocab_size.
        super().__init__(
            hidden_size=hidden_size,
            vocab_size=rows,
            rank=rank,
            top_k=top_k,
            params_dtype=params_dtype,
            prefix=prefix,
        )
        self.vocab_start, self.vocab_end = vocab_start, vocab_end
        for param in (self.predecessor_codebook, self.successor_codebook):
            param.weight_loader = self._load_codebook_shard

    def _load_codebook_shard(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        n = self.vocab_end - self.vocab_start
        param.data[:n].copy_(loaded_weight[self.vocab_start : self.vocab_end])
        param.data[n:].zero_()

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        predecessor_ids = _predecessor_ids(candidate_ids, anchor_token_ids, self.top_k)
        rows = torch.stack(
            (
                local_codebook_rows(
                    self.predecessor_codebook,
                    predecessor_ids,
                    self.vocab_start,
                    self.vocab_end,
                ),
                local_codebook_rows(
                    self.successor_codebook,
                    candidate_ids,
                    self.vocab_start,
                    self.vocab_end,
                ),
            )
        )
        # One rank holds each row and the others add zeros: exact.
        rows = tensor_model_parallel_all_reduce(rows)
        return _combine_edges(rows[0], rows[1], unary_logits, hidden)


def build_candidate_selector(**kwargs) -> qwen3_dflash2.CandidateSelector:
    """The DFlash2 candidate selector: the vocab-sharded one with
    ``VLLM_DFLASH2_SELECTOR_SHARD`` and TP > 1, else upstream's."""
    tp_size = get_tensor_model_parallel_world_size()
    if envs.VLLM_DFLASH2_SELECTOR_SHARD and tp_size > 1:
        return CandidateSelector(**kwargs)
    return qwen3_dflash2.CandidateSelector(**kwargs)


# ---------------------------------------------------------------------------
# Drafter models
# ---------------------------------------------------------------------------


class DFlash2Qwen3Model(qwen3_dflash2.DFlash2Qwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        # The upstream DFlash2Qwen3Model.__init__ with this module's selector
        # (upstream builds its own inline, which would allocate the full
        # codebooks before a replacement could shard them).
        DFlashQwen3Model.__init__(
            self,
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        adopt_grouped_convs(self, vllm_config)
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = build_candidate_selector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )


class DFlash2Qwen3ForCausalLM(DFlashDraftMixin, qwen3_dflash2.DFlash2Qwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model
