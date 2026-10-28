# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer of the sm_80 fork and its attention metadata.

``vllm_sm80.plugin.register_glm_layers`` registers
:class:`Glm5NextLinearAttention` as the out-of-tree replacement of the
upstream layer of the same name (``vllm/models/glm5next/common/kda.py``), so
every KDA layer the GLM decoder builds is this subclass. It adds:

- the fused sm_80 KDA decode step (``ops/kda_decode.py``,
  ``VLLM_AMPERE_DECODE_KERNELS``) and the KDA chunked prefill
  (``ops/kda_prefill.py``, ``VLLM_SM80_KDA_PREFILL``) in the layer's forward.
  Upstream's ``forward`` and ``_forward`` call this class's
  ``sm80_forward_tail``, ``_kda_mixed`` and ``sm80_chunk_kda`` where present
  (one hook each); everything else in them is upstream's code. The output
  norm runs inside the eager ``_forward`` break, where the step's path (and
  with it whether a fused kernel already applied the norm) is known;
- RecoverSSM (``vllm_sm80/kernels/kda/recover.py``): the per-layer record
  buffer, no draft-position state pages in the KV spec, and the builder's
  commit metadata;
- its own GDN attention backend, whose builder adds the step's bound on the
  tokens of a speculative request (``spec_max_query_len``, the decode
  kernel's token tile) and the RecoverSSM wrap to the GDN metadata. The
  metadata class itself is defined in ``vllm_sm80/kernels/kda/recover.py``,
  which must import without the GLM model package.
"""

import functools
from dataclasses import replace

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.glm5next.common import kda as glm_kda
from vllm_sm80.kernels.kda.recover import Glm5NextGDNAttentionMetadata
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

logger = init_logger(__name__)


class Glm5NextGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The GDN builder enables metadata reuse across groups only for its
        # exact class (subclasses may keep per-group metadata). This builder
        # adds only batch-level fields, so it takes the same choice as the GDN
        # builder; RecoverSSM below turns reuse off again.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner and device.type == "cuda"
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None
        # RecoverSSM (GLM-5 KDA layers, vllm_sm80/kernels/kda/recover.py):
        # commit metadata on every spec-decode step; None otherwise.
        self._kda_recover = None
        if self.use_spec_decode:
            from vllm_sm80.kernels.kda.recover import (
                BuilderRecover,
                kda_recover_enabled,
            )

            if kda_recover_enabled(vllm_config):
                self._kda_recover = BuilderRecover(self)
                # Recovery commit indices and contexts belong to this group,
                # unlike the batch-level metadata shared by ordinary GDN.
                self.supports_update_block_table = False

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        m = common_attn_metadata
        # The GDN builder returns a new GDNAttentionMetadata per build. Give it
        # the GLM class in place (a dataclass subclass adding one defaulted
        # field, no slots), so no copy is made per build.
        assert type(metadata) is GDNAttentionMetadata, type(metadata)
        metadata.__class__ = Glm5NextGDNAttentionMetadata
        metadata.spec_max_query_len = (
            min(int(m.max_query_len), self.num_spec + 1)
            if metadata.num_spec_decodes > 0 and getattr(m, "max_query_len", None)
            else None
        )
        if self._kda_recover is not None:
            return self._kda_recover.wrap(metadata, m, metadata.spec_sequence_masks_cpu)
        return metadata


class Glm5NextGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Glm5NextGDNAttentionMetadataBuilder]:
        return Glm5NextGDNAttentionMetadataBuilder


class Glm5NextLinearAttention(glm_kda.Glm5NextLinearAttention):
    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        # The fused sm_80 KDA decode step
        # (vllm_sm80/models/glm5next/ops/kda_decode.py)
        # whenever the decode kernels are on: forward() then defers f_b_proj /
        # g_b_proj to _forward so the kernel can consume f_a / g_a directly;
        # every other path computes them there.
        from vllm import envs as _envs

        self._kda_v2 = bool(_envs.VLLM_AMPERE_DECODE_KERNELS)
        # 64 heads per card (pipeline parallel): the measured decode shapes on
        # v2, with the wide schedule beyond one short sequence.
        self._kda_wide = bool(self._kda_v2 and self.local_num_heads == 64)
        # Token tile from the step's bound on tokens per request, at every
        # head count.
        self._kda_step_tile = self._kda_v2
        if self._kda_v2:
            # Imported here rather than on first use so that the pre-capture
            # warmup in vllm_sm80/kernels/warmup.py, which does not import model
            # packages, finds it among the loaded modules.
            from vllm_sm80.models.glm5next.ops import kda_decode  # noqa: F401

            logger.info_once(
                "sm_80 KDA decode v2 active: f_b_proj + g_b_proj + conv + "
                "gated delta rule + gated RMSNorm in one kernel, token tile "
                "sized per step."
            )
        if self._kda_wide:
            logger.info_once(
                "sm_80 KDA decode v2 wide: 64 heads, the measured shapes "
                "of up to 8 sequences of up to 8 tokens."
            )

        # RecoverSSM (vllm_sm80/kernels/kda/recover.py): one state per
        # request; the verify stores per-token records instead of states.
        self._kda_recover = False
        if self._kda_v2:
            from vllm_sm80.kernels.kda.recover import kda_recover_enabled

            if kda_recover_enabled(vllm_config):
                self._init_kda_recover(config, vllm_config, prefix)

        # VLLM_SM80_KDA_PREFILL (64 or 16 heads): the layer part of the gate,
        # resolved once.
        self._pp_kda_prefill = False
        if _envs.VLLM_SM80_KDA_PREFILL:
            from vllm_sm80.models.glm5next.ops.kda_prefill import (
                enabled_heads,
                use_for_layer,
            )

            self._pp_kda_prefill = use_for_layer(
                self.kda_prefill_backend,
                self.local_num_heads,
                self.head_dim,
                vllm_config.model_config.dtype,
                self.kda_safe_gate,
                self.kda_lower_bound,
                allowed_heads=enabled_heads(),
            )

        # Set by _kda_deferred_step for one mixed RecoverSSM step and cleared
        # after upstream's _forward has read it.
        self._kda_mixed = None

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Glm5NextGDNAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        if self._kda_recover:
            # Recovery records replace draft-position state pages, not the
            # independent prefill checkpoint reservation.
            spec = replace(spec, num_speculative_blocks=0)
        return spec

    def _init_kda_recover(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        import re

        from vllm_sm80.kernels.kda.recover import (
            RecordPool,
            layer_records,
            records_rows,
        )

        kda_layers = [
            i for i, t in enumerate(config.layer_types) if t == "linear_attention"
        ]
        m = re.search(r"layers\.(\d+)\.", prefix + ".")
        if m is None or int(m.group(1)) not in kda_layers:
            raise ValueError(f"RecoverSSM: no KDA layer index in {prefix!r}")
        self._kda_recover = True
        self._kda_recover_index = kda_layers.index(int(m.group(1)))
        device = torch.empty(0).device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        self._kda_recover_pool = RecordPool.get(
            device, len(kda_layers), records_rows(vllm_config)
        )
        self._kda_recover_records = layer_records(
            self._kda_recover_pool, self._kda_recover_index
        )

    def _fill_deferred_g2(self, g_a: torch.Tensor, g2: torch.Tensor) -> None:
        """g_b_proj(g_a) into the caller's g2 buffer (decode-kernel
        paths that do not fuse the norm): the same GEMM the unfused forward
        runs, copied bitwise into the buffer o_norm reads."""
        g2.copy_(
            self.g_b_proj(g_a)[0].reshape(-1, self.local_num_heads, self.head_dim)
        )

    def sm80_forward_tail(
        self,
        hidden_states: torch.Tensor,
        qkv: torch.Tensor,
        beta: torch.Tensor,
        f_a: torch.Tensor,
        g_a: torch.Tensor,
    ) -> torch.Tensor:
        """Upstream ``forward`` after the input projections, which calls it.

        With the decode kernels on, f_b_proj / g_b_proj are deferred to
        ``_forward``: the fused kernel reads f_a / g_a itself. g1 is only
        consumed inside ``_forward``, so it is computed there when needed. g2
        is also read by the o_norm at the end of ``_forward``, so it is a
        buffer allocated here and filled in ``_forward`` on every path that
        does not fuse the norm (the same static-buffer contract as
        core_attn_out).
        """
        num_tokens = hidden_states.size(0)
        if self._kda_v2:
            g1 = None
            g2 = torch.empty(
                (num_tokens, self.local_num_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            f_a_in, g_a_in = f_a, g_a
        else:
            g1 = self.f_b_proj(f_a)[0]
            g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

            g_proj_states = self.g_b_proj(g_a)[0]
            # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
            g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)
            f_a_in = g_a_in = None

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Call the decorated eager break directly so host-side prefill branches
        # are not captured by PIECEWISE CUDA graphs. It leaves the normalised
        # attention output in core_attn_out.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
            g2=g2,
            f_a=f_a_in,
            g_a=g_a_in,
        )
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor | None,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        g2: torch.Tensor | None = None,
        f_a: torch.Tensor | None = None,
        g_a: torch.Tensor | None = None,
    ) -> None:
        """Core attention and o_norm, written into ``core_attn_out``.

        The step's metadata picks the path here, and with it whether a fused
        sm_80 kernel has already applied o_norm, so the norm is applied here
        too: a host-side decision read after this eager break would be frozen
        into a PIECEWISE graph by the batch it was captured with.
        """
        if self._forward_core(qkv_proj_states, g1, beta, core_attn_out, g2, f_a, g_a):
            return
        normed = self.o_norm(core_attn_out, g2)
        # The CUDA gated norm runs in place; keep the static buffer either way.
        if normed.data_ptr() != core_attn_out.data_ptr():
            core_attn_out.copy_(normed)

    def _forward_core(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor | None,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        g2: torch.Tensor | None,
        f_a: torch.Tensor | None,
        g_a: torch.Tensor | None,
    ) -> bool:
        """Core attention into ``core_attn_out``; True when a fused sm_80
        kernel has already applied o_norm. f_a / g_a are passed only with the
        decode kernels on (see :meth:`sm80_forward_tail`): g1 is then None and
        g2 an unfilled buffer, resolved by :meth:`_kda_deferred_step`. The
        rest is upstream's ``_forward`` (nested in this eager break, so it runs
        inline)."""
        if f_a is not None:
            fused, g1 = self._kda_deferred_step(
                qkv_proj_states, beta, core_attn_out, g2, f_a, g_a
            )
            if fused:
                return True
            if g1 is None:
                # No metadata for this layer (profile / warmup dummy run).
                return False
        try:
            # Explicit, not super(): tests/kernels/test_ampere_kda_dispatch.py
            # binds these methods to a stand-in layer.
            glm_kda.Glm5NextLinearAttention._forward(
                self,
                qkv_proj_states=qkv_proj_states,
                g1=g1,
                beta=beta,
                core_attn_out=core_attn_out,
            )
        finally:
            self._kda_mixed = None
        return False

    def _kda_deferred_step(
        self,
        qkv_proj_states: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        g2: torch.Tensor,
        f_a: torch.Tensor,
        g_a: torch.Tensor,
    ) -> tuple[bool, torch.Tensor | None]:
        """The decode-kernel part of ``_forward`` ahead of upstream's: the step
        metadata and caches as upstream's ``_forward`` reads them, then

        - the fused sm_80 KDA decode step (kda_decode.py): ONE kernel for
          f_b_proj + g_b_proj + causal_conv1d_update + fused_recurrent_kda +
          o_norm on a pure spec-verify step, within the shapes
          use_ampere_kda_decode_v2 covers; RecoverSSM: a pure spec-verify step
          runs the kernel in RECOVER mode whatever the shape gate says (there
          are no draft-position state pages to write to). Returns (True, None);
        - otherwise the deferred projections are materialised (g1 returned,
          g2 filled) and the unfused chain runs upstream's ``_forward``; under
          RecoverSSM the same kernel runs the spec rows of a mixed step
          unnormalised (``_kda_mixed``, read by upstream's ``_forward``).

        Without metadata only g2 is filled: (False, None).
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            self._fill_deferred_g2(g_a, g2)
            return False, None
        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs may omit mamba-family metadata.
            self._fill_deferred_g2(g_a, g2)
            return False, None
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # The fused kernel's token tile from the step's bound (None: the full
        # num_spec + 1 window, without the fused step).
        kda_tile = (
            getattr(attn_metadata_narrowed, "spec_max_query_len", None)
            if self._kda_step_tile
            else None
        )
        lower_bound = self.kda_lower_bound

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = self.kv_cache
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)
        # Upstream's merged q|k|v short-conv weight, built once (upstream's
        # _forward finds it built).
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = torch.cat(
                [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                dim=0,
            ).contiguous()
        conv_weights = self._merged_conv_weight
        conv_bias = self.q_conv1d.bias

        pure_verify = (
            use_spec
            and (non_spec_token_indx is None or non_spec_token_indx.numel() == 0)
            and attn_metadata_narrowed.num_prefills == 0
            and attn_metadata_narrowed.num_decodes == 0
        )
        if self._kda_recover and pure_verify:
            from vllm_sm80.models.glm5next.ops.kda_decode import (
                _kda_tile,
                kda_decode_v2,
            )

            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None and spec_query_start_loc is not None
            kda_decode_v2(
                qkv_proj_states,
                beta,
                f_a[:num_actual_tokens],
                g_a[:num_actual_tokens],
                self.f_b_proj.weight,
                self.g_b_proj.weight,
                conv_state,
                conv_weights,
                conv_bias,
                self.o_norm.weight,
                recurrent_state,
                spec_state_indices_tensor[:, 0][:num_spec_decodes],
                spec_state_indices_tensor,
                num_accepted_tokens,
                spec_query_start_loc[: num_spec_decodes + 1],
                _kda_tile(kda_tile, self.num_spec + 1, keep_same_rows=True),
                self.A_log.view(-1),
                self.dt_bias,
                lower_bound=lower_bound,
                eps=self.o_norm.eps,
                out=core_attn_out[0, :num_actual_tokens].unsqueeze(0),
                records=self._kda_recover_records,
            )
            return True, None

        # Spec rows of a mixed step on the fused kernel: RecoverSSM only (the
        # per-draft-slot step measured slower there than the unfused chain).
        from vllm_sm80.models.glm5next.ops.kda_decode import (
            use_ampere_kda_decode_v2,
            wide_schedule,
        )

        w_f = getattr(self.f_b_proj, "weight", None)
        w_g = getattr(self.g_b_proj, "weight", None)
        v2_ready = (
            use_spec
            and w_f is not None
            and w_g is not None
            and spec_state_indices_tensor is not None
            and num_accepted_tokens is not None
            and spec_query_start_loc is not None
            and recurrent_state.dtype == torch.float32
        )
        if (
            v2_ready
            and pure_verify
            and use_ampere_kda_decode_v2(
                num_spec_decodes,
                num_actual_tokens,
                self.local_num_heads,
                self.head_dim,
                w_f,
                w_g,
            )
        ):
            from vllm_sm80.models.glm5next.ops.kda_decode import (
                _kda_tile,
                kda_decode_v2,
            )

            kda_decode_v2(
                qkv_proj_states,
                beta,
                f_a[:num_actual_tokens],
                g_a[:num_actual_tokens],
                w_f,
                w_g,
                conv_state,
                conv_weights,
                conv_bias,
                self.o_norm.weight,
                recurrent_state,
                spec_state_indices_tensor[:, 0][:num_spec_decodes],
                spec_state_indices_tensor,
                num_accepted_tokens,
                spec_query_start_loc[: num_spec_decodes + 1],
                _kda_tile(kda_tile, spec_state_indices_tensor.size(-1)),
                self.A_log.view(-1),
                self.dt_bias,
                lower_bound=lower_bound,
                eps=self.o_norm.eps,
                out=core_attn_out[0, :num_actual_tokens].unsqueeze(0),
                wide=self._kda_wide
                and wide_schedule(
                    num_spec_decodes,
                    num_actual_tokens // num_spec_decodes,
                    self.local_num_heads,
                ),
            )
            return True, None
        mixed_v2 = (
            v2_ready
            and self._kda_recover
            and not pure_verify
            and spec_token_indx is not None
        )
        # Upstream's _forward cuts g1 to num_actual_tokens.
        g1 = self.f_b_proj(f_a)[0]
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)
        self._fill_deferred_g2(g_a, g2)
        if mixed_v2:
            self._kda_mixed = functools.partial(
                self._kda_mixed_spec_rows,
                f_a=f_a[:num_actual_tokens],
                g_a=g_a[:num_actual_tokens],
                spec_token_indx=spec_token_indx,
                conv_state=conv_state,
                conv_weights=conv_weights,
                conv_bias=conv_bias,
                recurrent_state=recurrent_state,
                spec_state_indices_tensor=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_start_loc=spec_query_start_loc,
                num_spec_decodes=num_spec_decodes,
                kda_tile=kda_tile,
                lower_bound=lower_bound,
            )
        return False, g1

    def _kda_mixed_spec_rows(
        self,
        qkv_spec: torch.Tensor,
        beta_spec: torch.Tensor,
        *,
        f_a: torch.Tensor,
        g_a: torch.Tensor,
        spec_token_indx: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias: torch.Tensor | None,
        recurrent_state: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_spec_decodes: int,
        kda_tile: int | None,
        lower_bound: float,
    ) -> torch.Tensor:
        """Mixed prefill + verify step (``_kda_mixed``, in place of upstream's
        spec-row recurrence; upstream's spec-row conv is skipped): the spec
        rows go through the recover verify (conv + recurrence, records, no
        state writes) and stay unnormalised; ``_forward`` norms the whole
        step."""
        from vllm_sm80.models.glm5next.ops.kda_decode import (
            _kda_tile,
            kda_decode_v2,
        )

        core_attn_out_spec = torch.empty(
            1,
            qkv_spec.shape[0],
            self.local_num_heads,
            self.head_dim,
            dtype=qkv_spec.dtype,
            device=qkv_spec.device,
        )
        kda_decode_v2(
            qkv_spec,
            beta_spec.contiguous(),
            f_a.index_select(0, spec_token_indx),
            g_a.index_select(0, spec_token_indx),
            self.f_b_proj.weight,
            self.g_b_proj.weight,
            conv_state,
            conv_weights,
            conv_bias,
            self.o_norm.weight,
            recurrent_state,
            spec_state_indices_tensor[:, 0][:num_spec_decodes],
            spec_state_indices_tensor,
            num_accepted_tokens,
            spec_query_start_loc[: num_spec_decodes + 1],
            _kda_tile(kda_tile, self.num_spec + 1, keep_same_rows=True),
            self.A_log.view(-1),
            self.dt_bias,
            lower_bound=lower_bound,
            eps=self.o_norm.eps,
            out=core_attn_out_spec,
            records=self._kda_recover_records,
            skip_norm=True,
        )
        return core_attn_out_spec

    def sm80_chunk_kda(self, **kwargs):
        """Upstream's ``chunk_kda_with_fused_gate`` call in ``_forward``, on
        this module's chunked prefill (kda_prefill.py) when
        VLLM_SM80_KDA_PREFILL covers the layer and the call."""
        chunk_fn = glm_kda.chunk_kda_with_fused_gate
        if self._pp_kda_prefill:
            from vllm_sm80.models.glm5next.ops.kda_prefill import select_chunk_fn

            q = kwargs["q"]
            chunk_fn = select_chunk_fn(
                chunk_fn,
                q.shape[1],
                kwargs["cu_seqlens"].shape[0] - 1,
                q.dtype,
                kwargs["raw_g"].dtype,
                self.kv_cache[1].dtype,
                self.A_log.dtype,
                self.dt_bias.dtype,
                num_heads=self.local_num_heads,
            )
        return chunk_fn(**kwargs)
