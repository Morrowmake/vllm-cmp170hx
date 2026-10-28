# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash model classes of the sm_80 fork.

The fork plugin (``vllm_sm80.plugin``) registers the two architectures of this
module in vLLM's model registry in place of the upstream classes in
``vllm/models/glm5next/common/model.py``. Each class here subclasses the
upstream one and keeps its name, so the registry reports the same model
information and checkpoints load into the same parameter names.

What the subclasses add:

- pipeline parallelism: the boundary tensors (the mHC residual streams) and
  ``make_empty_intermediate_tensors``;
- auxiliary hidden states for EAGLE3/DFlash drafters across pipeline stages,
  the choice of the captured tensor (``aux_hidden_state_tensor``) and the
  drafter fc fold (``vllm_sm80/spec_decode/aux_fc_fold.py``);
- the target embedding on the last pipeline stage when a drafter there
  shares it;
- ``compute_logits_local`` for ``--enable-batch-sharded-sampling``;
- with ``VLLM_SM80_SHARD_ARGMAX``, the argmax rows from ``compute_logits``
  in a greedy-only sampling step of the fork runner
  (``vllm_sm80/worker/target_argmax.py``: ``ShardArgmaxLogits``);
- the model-runner state (``model_state.py``) with the fork's KDA RecoverSSM.

Importing this module also registers the fork's GLM layer overrides
(``vllm_sm80.plugin.register_glm_layers``): the fork KDA layer and its
attention metadata (``sm80_kda.py``).

The backbone (``Glm5NextModel``) is built by the upstream constructor and then
given the fork class, so its modules and parameters are exactly the upstream
ones.
"""

from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.mhc import hc_contract
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.glm5next.common import model as glm_model
from vllm.sequence import IntermediateTensors
from vllm_sm80 import plugin
from vllm_sm80.worker.target_argmax import ShardArgmaxLogits

# Which per-layer tensor is handed to an EAGLE3/DFlash drafter as the
# "hidden state" of an auxiliary layer, from the text config field
# ``aux_hidden_state_tensor`` (absent: "stream_mean"; set it with e.g.
# ``--hf-overrides '{"text_config": {"aux_hidden_state_tensor": "branch"}}'``).
# GLM-5.3-Flash runs mHC with ``hc_mult`` (4) parallel residual streams, so a
# layer has no single [tokens, hidden] output until its deferred ``hc_post`` is
# materialized and the streams are contracted. See
# Glm5NextModel._capture_aux_hidden_state.
#   "stream_mean" (default): hc_contract(hc_post(...)) -- the same tensor the
#       last decoder layer produces, i.e. the model's own definition of "the
#       hidden state after layer i". Matches the drafter's expected width
#       (target_hidden_size == config.hidden_size, one feature block per layer).
#   "branch": the raw FFN branch output before hc_post (same width, no
#       residual). Escape hatch for checkpoints trained on the branch tensor.
_AUX_TENSOR_MODES = ("stream_mean", "branch")


def aux_hidden_state_mode(config) -> str:
    mode = getattr(config, "aux_hidden_state_tensor", "stream_mean")
    if mode not in _AUX_TENSOR_MODES:
        raise ValueError(
            f"aux_hidden_state_tensor={mode!r} is not one of {_AUX_TENSOR_MODES}"
        )
    return mode


def _mm_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w.T`` with fp32 output (fp32 accumulation, no rounding to the
    input dtype), as the drafter fc accumulates before its single rounding."""
    if x.is_cuda:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), w.float().t())


def _drafter_target_embedding(
    vllm_config: VllmConfig, prefix: str
) -> VocabParallelEmbedding | None:
    """The input embedding a non-first pipeline stage holds for its drafter.

    EAGLE/DFlash drafters on the last stage ship no embedding of their own
    and alias the target's (maybe_share_target_embed), which a PPMissingLayer
    cannot provide. The first stage builds the table itself. Built before the
    backbone, so it is the backbone's first module, as on the first stage.
    """
    if get_pp_group().is_first_rank or not spec_decode_needs_target_embed(
        vllm_config
    ):
        return None
    config = vllm_config.model_config.hf_text_config
    return VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        prefix=f"{prefix}.embed_tokens",
    )


class Glm5NextModel(glm_model.Glm5NextModel):
    """Backbone with pipeline-parallel mHC boundaries and drafter aux states.

    Auxiliary hidden states for EAGLE3/DFlash drafters cross pipeline stages
    as extra IntermediateTensors keys next to the mHC residual streams: each
    stage packs the states of its own aux layers
    (EagleModelMixin.pack_local_aux_hidden_states), middle stages forward
    the upstream ones unchanged (PPHandler.relay_aux_hidden_states), and the
    last stage prepends the received ones to its own
    (collect_remote_aux_hidden_states), so the drafter sees them in layer
    order exactly as with PP=1. Each captured state is already the
    contracted [num_tokens, hidden_size] tensor, which is the width the
    relay slots reserve (reserve_aux_intermediate_tensor_slots).

    ``forward`` replaces upstream's, including its aux capture
    (``_aux_hidden_state``): that one has no pipeline boundary, ignores
    ``aux_hidden_state_tensor``, and contracts the streams by
    ``config.mhc_num_residual_streams``, which the GLM-5.3-Flash text config
    does not carry (the stream count is ``config.hc_mult``).
    """

    supports_aux_hidden_states_over_pp: ClassVar[bool] = True
    # Drafter fc fold under PP (enable_aux_fc_fold); None when off.
    aux_fc_blocks: dict[int, torch.Tensor] | None = None

    @classmethod
    def adopt(
        cls,
        model: glm_model.Glm5NextModel,
        embed_tokens: VocabParallelEmbedding | None,
    ) -> "Glm5NextModel":
        """Give an upstream-built backbone this class and its state."""
        assert type(model) is glm_model.Glm5NextModel, type(model)
        model.__class__ = cls
        if embed_tokens is not None:
            # Replaces the PPMissingLayer in place (same attribute slot).
            model.embed_tokens = embed_tokens
        config = model.config
        # Same derivation as Glm5NextDecoderLayer (self.mhc = config.mhc /
        # self.n = config.hc_mult).
        model.mhc = bool(config.mhc)
        if model.mhc:
            model.hc_mult = config.hc_mult
            # PP boundary tensor for mHC: the residual streams themselves,
            # (num_tokens, n_streams, hidden_size). The deferred hc_post state
            # (post/comb) is materialized into the streams before sending, so
            # a single key suffices (no separate "residual" copy).
            model.make_empty_intermediate_tensors = (
                model._make_empty_mhc_intermediate_tensors
            )
        else:
            # Plain (hidden_states, residual) pair of shape
            # (num_tokens, hidden_size), like every other dense-residual model.
            model.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size
                )
            )

        # Only consulted when a drafter asks for auxiliary hidden states.
        model.aux_hidden_state_mode = aux_hidden_state_mode(config)
        # Drafter fc fold under PP (aux_fc_fold.py): aux slot -> this stage's
        # [hidden, hidden] block of the drafter's fc.weight. None when off.
        model.aux_fc_blocks = None
        return model

    def _make_empty_mhc_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (
                        batch_size,
                        self.hc_mult,
                        self.config.hidden_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def enable_aux_fc_fold(self, blocks: dict[int, torch.Tensor]) -> None:
        """Fold the drafter's fc into this stage (VLLM_GLM5_PP_FOLD_DRAFT_FC):
        ``blocks`` maps each local aux slot to its fc.weight column block. The
        boundary then carries one fp32 partial sum instead of aux states."""
        self.aux_fc_blocks = blocks
        # Nothing to collect as separate aux states on the last stage.
        self._aux_upstream_total_cached = 0

    def _fold_aux_fc(
        self,
        partial: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """Return partial + sum_j aux_j @ W_(base+j)^T (products in fp32)."""
        assert self.aux_fc_blocks is not None
        if partial is None:
            ref = aux_hidden_states[0] if aux_hidden_states else None
            partial = torch.zeros(
                (num_tokens, self.config.hidden_size),
                dtype=torch.float32,
                device=ref.device if ref is not None else self.device,
            )
        base = self._aux_slot_base_cached
        for j, aux in enumerate(aux_hidden_states):
            partial = partial + _mm_fp32(aux, self.aux_fc_blocks[base + j])
        return partial

    def _capture_aux_hidden_state(
        self,
        layer: glm_model.Glm5NextDecoderLayer,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> torch.Tensor:
        """The ``[num_tokens, hidden_size]`` output state of ``layer``.

        Non-mHC (and MTP) layers already return the summed residual stream, so
        the layer output is ``hidden_states`` as-is -- unlike the
        ``hidden_states + residual`` convention of
        ``EagleModelMixin._maybe_add_hidden_state``, which assumes layers that
        return the pair unsummed.

        mHC layers defer their ``hc_post`` to the next layer's fused pre-op, so
        ``hidden_states`` here is only the FFN branch output and the four
        residual streams live in ``residual``/``post``/``comb``. Materializing
        that deferred state and contracting the streams (their mean) is exactly
        what the *last* decoder layer does before ``self.norm``, so it is the
        model's own definition of "the hidden state after layer i" and the only
        candidate whose width equals the drafter's ``target_hidden_size``
        (the un-contracted streams are ``n * hidden_size`` wide). ``hc_post``
        is a pure function of the deferred state; calling it here does not
        consume it, so the next layer's fused op is unaffected.
        """
        if (
            not self.mhc
            or self.aux_hidden_state_mode == "branch"
            or residual is None
            or post is None
        ):
            return hidden_states
        assert comb is not None
        streams = layer.hc_post(hidden_states, residual, post, comb)
        return hc_contract(streams, self.hc_mult)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if self.mhc:
                # The previous rank sent the materialized residual streams
                # (num_tokens, n, hidden): its pending hc_post was applied
                # before the send, so there is no deferred state to fuse.
                # With residual=None/post=None this rank's first layer takes
                # the standalone hc_pre path (x = residual = streams), which
                # is exactly the layer-0 path minus hc_expand (layer_idx != 0
                # here, so no re-expansion happens).
                assert hidden_states.dim() == 3, hidden_states.shape
                residual = None
            else:
                residual = intermediate_tensors["residual"]
            post = None
            comb = None

        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        # Aux states captured on earlier pipeline stages (last stage only;
        # empty without an aux-hidden-state drafter or with PP=1).
        remote_aux_hidden_states = self.collect_remote_aux_hidden_states(
            intermediate_tensors
        )
        fold_partial_in: torch.Tensor | None = None
        if self.aux_fc_blocks is not None and not get_pp_group().is_first_rank:
            from vllm_sm80.spec_decode.aux_fc_fold import (
                AUX_FC_PARTIAL_KEY,
            )

            assert intermediate_tensors is not None
            fold_partial_in = intermediate_tensors[AUX_FC_PARTIAL_KEY]
        aux_hidden_states: list[torch.Tensor] = []
        if self.aux_hidden_state_layers:
            for layer in self._active_layers:
                hidden_states, residual, post, comb = layer(
                    positions, hidden_states, residual, post, comb
                )
                # Aux layer ids are "capture after layer i", encoded as
                # i + 1 (see spec_decode get_eagle3_aux_layers_from_config,
                # which adds 1 to the drafter's
                # dflash_config.target_layer_ids).
                if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(
                        self._capture_aux_hidden_state(
                            layer, hidden_states, residual, post, comb
                        )
                    )
        else:
            for layer in self._active_layers:
                hidden_states, residual, post, comb = layer(
                    positions, hidden_states, residual, post, comb
                )

        if not get_pp_group().is_last_rank:
            # This stage's aux states travel with the boundary tensors; the
            # receiver expects full-token rows, as for the streams below.
            if self.is_sequence_parallel and aux_hidden_states:
                aux_hidden_states = [
                    sp_all_gather(aux)[:full_num_tokens] for aux in aux_hidden_states
                ]
            if self.aux_fc_blocks is not None:
                from vllm_sm80.spec_decode.aux_fc_fold import (
                    AUX_FC_PARTIAL_KEY,
                )

                aux_tensors = {
                    AUX_FC_PARTIAL_KEY: self._fold_aux_fc(
                        fold_partial_in, aux_hidden_states, full_num_tokens
                    )
                }
            else:
                aux_tensors = self.pack_local_aux_hidden_states(aux_hidden_states)
            if self.mhc:
                # post/comb are the deferred hc_post inputs of this rank's
                # last mHC layer (normally consumed by the next layer's
                # hc_fused_post_pre). Materialize them here so the boundary
                # carries only the residual streams; the next rank's first
                # layer then runs standalone hc_pre on them, which is what
                # the fused op computes after its internal hc_post.
                if post is not None:
                    assert residual is not None and comb is not None
                    hidden_states = self._active_layers[-1].hc_post(
                        hidden_states, residual, post, comb
                    )
                assert hidden_states.dim() == 3, hidden_states.shape
                if self.is_sequence_parallel:
                    # Send the full token count (the receiver re-shards), as
                    # the last rank does before its final norm. Collectives
                    # expect 2-D, so gather the flattened streams.
                    n, h = hidden_states.shape[1:]
                    hidden_states = sp_all_gather(hidden_states.flatten(1))[
                        :full_num_tokens
                    ].view(-1, n, h)
                return IntermediateTensors(
                    {"hidden_states": hidden_states, **aux_tensors}
                )
            if self.is_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                residual = sp_all_gather(residual)[:full_num_tokens]
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual, **aux_tensors}
            )

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        if self.is_sequence_parallel and aux_hidden_states:
            aux_hidden_states = [
                sp_all_gather(aux)[:full_num_tokens] for aux in aux_hidden_states
            ]
        if self.aux_fc_blocks is not None:
            # The drafter receives the finished fc output (fp32) in place of
            # the concatenated aux states.
            return hidden_states, [
                self._fold_aux_fc(fold_partial_in, aux_hidden_states, full_num_tokens)
            ]
        # Remote states come from earlier layers, so they go first.
        aux_hidden_states = remote_aux_hidden_states + aux_hidden_states
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Glm5NextForCausalLM(ShardArgmaxLogits, glm_model.Glm5NextForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        embed_tokens = _drafter_target_embedding(
            vllm_config, maybe_prefix(prefix, "model")
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        Glm5NextModel.adopt(self.model, embed_tokens)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_model_state_cls(self):
        from vllm_sm80.models.glm5next.model_state import (
            resolve_glm_model_state_cls,
        )

        return resolve_glm_model_state_cls(self.vllm_config, self)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """This rank's vocab shard of the logits, ungathered, for
        ``--enable-batch-sharded-sampling``: each rank samples its own rows
        after one all-to-all instead of all-gathering the full-vocab logits
        (4x fewer wire bytes at TP=4)."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)


class Glm5NextForConditionalGeneration(
    ShardArgmaxLogits, glm_model.Glm5NextForConditionalGeneration
):
    """The multimodal wrapper. Its ``language_model`` is built through the
    registry under ``Glm5NextForCausalLM``, so it is this module's text model
    whenever the fork plugin has registered it."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # For get_model_state_cls (the upstream wrapper keeps model_config only).
        self._sm80_vllm_config = vllm_config
        # PP: the boundary tensors are defined by the text model (mHC residual
        # streams, or the plain hidden/residual pair for non-mHC configs).
        # The inherited Glm4vForConditionalGeneration.forward passes
        # intermediate_tensors straight through to language_model.model and
        # returns its IntermediateTensors unchanged on non-last ranks.
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_model_state_cls(self):
        from vllm_sm80.models.glm5next.model_state import (
            resolve_glm_model_state_cls,
        )

        return resolve_glm_model_state_cls(self._sm80_vllm_config, self)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)


# The upstream decoder layer builds the KDA layers by the upstream class name;
# the layer override makes each of them the fork class (sm80_kda.py).
plugin.register_glm_layers()
