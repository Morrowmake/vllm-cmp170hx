# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step draft width for vLLM's DFlash-family speculators.

``DraftWidthMixin`` goes in front of a ``DFlashSpeculator`` subclass (the
speculator factory, ``vllm_sm80.spec_decode.speculators``, builds the mixed
classes). The block is drafted at the width the scheduler asks for (the
load-following width of ``adaptive_k.draft_by_load``, DSpark's per-K widths).
Every per-width buffer is a contiguous view of the widest one, and every width
has its own drafter CUDA graphs. With one width (the configured
``num_speculative_tokens``, unless :meth:`enable_draft_widths`) the speculator
runs exactly as vLLM's.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager


class DraftWidthMixin:
    """Draft widths for a ``DFlashSpeculator`` subclass (MRO: mixin first)."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)  # type: ignore[call-arg]
        self.max_width = self.num_speculative_steps
        self.widths: tuple[int, ...] = (self.max_width,)
        self.width = self.max_width
        self._full_draft_tokens = self.draft_tokens
        self._draft_token_views = {self.max_width: self.draft_tokens}
        self._sample_cols = {self.max_width: self.sample_col}
        self._cg_managers: dict[int, DFlashCudaGraphManager] = {}

    def enable_draft_widths(self, widths) -> None:
        """Draft at any of ``widths`` (each <= num_speculative_tokens)."""
        widths = tuple(sorted({int(w) for w in widths} | {self.max_width}))
        assert 1 <= widths[0] and widths[-1] == self.max_width, widths
        self.widths = widths
        flat = self._full_draft_tokens.view(-1)
        for w in widths:
            self._draft_token_views[w] = flat[: self.max_num_reqs * w].view(
                self.max_num_reqs, w
            )
            self._sample_cols[w] = torch.arange(
                w, dtype=torch.int32, device=self.device
            ).repeat(self.max_num_reqs)

    def set_width(self, width: int) -> None:
        """Make ``width`` the block the next forward drafts (host state only)."""
        if width not in self._draft_token_views:
            width = self.max_width
        self.width = width
        self.num_query_per_req += width - self.num_speculative_steps
        self.num_speculative_steps = width
        self.draft_tokens = self._draft_token_views[width]
        self.sample_col = self._sample_cols[width]
        if width in self._cg_managers:
            self.query_cudagraph_manager = self._cg_managers[width]
        self._set_model_block_size(self.num_query_per_req)
        self._on_width(width)

    def _set_model_block_size(self, block_size: int) -> None:
        """Tell a drafter whose layers depend on the per-request block length
        (DFlash2's grouped conv) the length this width drafts. A device
        scalar written on the stream, so eager, compiled and captured
        forwards all read the current value; written only on a change."""
        model = getattr(self, "model", None)
        if not isinstance(model, nn.Module) or len(getattr(self, "widths", ())) < 2:
            return
        tensors = getattr(self, "_block_size_tensors", None)
        if tensors is None:
            tensors = self._block_size_tensors = [
                m.block_size_tensor
                for m in model.modules()
                if getattr(m, "block_size_tensor", None) is not None
            ]
        if tensors and getattr(self, "_model_block_size", None) != block_size:
            for t in tensors:
                t.fill_(block_size)
            self._model_block_size = block_size

    def _on_width(self, width: int) -> None:
        """Subclass hook for per-width state."""

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # vLLM's manager, built once per width for that width's query length
        # (the request's block plus the width's draft positions).
        query_len_base = self.num_query_per_req - self.width
        for width in self.widths:
            self.num_query_per_req = query_len_base + width
            super().init_cudagraph_manager(cudagraph_mode)  # type: ignore[misc]
            self._cg_managers[width] = self.query_cudagraph_manager
        self.num_query_per_req = query_len_base + self.width
        self.set_width(self.max_width)

    def capture(self) -> None:
        # One graph set per draft width; each is captured with that width's
        # host state (block size, sample layout, draft-token view).
        for width in self.widths:
            self.set_width(width)
            super().capture()  # type: ignore[misc]
        self.set_width(self.max_width)

    def retire_width_managers(self) -> None:
        """Drop the per-width managers (the profiling capture's): vLLM's
        profiling retires only the managers held directly as attributes, and
        the real KV initialization rebuilds all of them."""
        self._cg_managers.clear()

    @torch.inference_mode()
    def propose(self, *args: Any, num_speculative_tokens: int | None = None, **kwargs):
        # The block width for this step: what the scheduler will verify next
        # (load-following width), else the configured num_speculative_tokens.
        self.set_width(
            num_speculative_tokens
            if num_speculative_tokens is not None
            else self.max_width
        )
        return super().propose(  # type: ignore[misc]
            *args, num_speculative_tokens=num_speculative_tokens, **kwargs
        )
