# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early Engram gathers and projection beside decoder compute.

The lookup and WKV projection depend only on this step's hashes. Issue them
before decoder compute and join each layer's own completion at injection.

They run on a stream of their own, shared by the Engram layers so their
lookups take turns. The shared-expert projections run on the registered
auxiliary stream; queued there, every Engram layer's lookup, gathers and WKV
would sit in front of the first decoder layer's shared expert, and that
layer's MoE output would wait for the last Engram layer's projection.

An EXL3 WKV projection stays on the registered auxiliary stream: EXL3 picks
its scratch by stream (scratch[1] there, scratch[0] on any other stream), so
on a third stream it would share the main stream's scratch. There the shared
experts cannot overlap it; no third scratch is needed.
"""

from itertools import islice

import torch

from vllm.models.deepseek_v41.common.engram import Engram
from vllm_sm80.models.deepseek_v41.sm80_engram_projection import project_rows
from vllm.utils.torch_utils import aux_stream, current_stream

_STREAMS: dict[int, torch.cuda.Stream] = {}


def _projection_stream(engram: Engram) -> torch.cuda.Stream | None:
    """The stream an Engram's early lookup and projection run on (module doc),
    or None when an EXL3 projection has no registered auxiliary stream."""
    from vllm_sm80.kernels.exl3.quant import Exl3LinearMethod

    if isinstance(getattr(engram.wkv, "quant_method", None), Exl3LinearMethod):
        return aux_stream()
    index = torch.cuda.current_device()
    stream = _STREAMS.get(index)
    if stream is None:
        stream = _STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


class EarlyEngram(Engram):
    """An Engram with one outstanding projection, consumed by its forward."""

    @classmethod
    def adopt(cls, engram: Engram) -> None:
        if engram.use_sequence_parallel or engram._batch is not None:
            raise ValueError("early Engram requires unbatched TP projection")
        if engram.embed_tokens.dp_size != 1:
            raise ValueError("early Engram requires TP-only table sharding")
        stream = _projection_stream(engram)
        if stream is None:
            raise ValueError(
                "early Engram with an EXL3 projection requires the registered "
                "auxiliary stream"
            )
        engram.__class__ = cls
        engram._prefetch_stream = stream
        engram._prefetch_done = torch.cuda.Event()
        engram._projection_done = torch.cuda.Event()
        engram._early_kv = None

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        if self._early_kv is not None:
            raise RuntimeError("previous Engram projection was not consumed")
        stream = self._prefetch_stream
        assert stream is not None
        stream.wait_stream(current_stream())
        hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            # Keep lookup, both gathers and WKV on the projection stream,
            # including large lookups that the upstream helper runs inline.
            super().prepare_embeddings(hash_ids)
            self._early_kv = project_rows(self, hash_ids)
            self._projection_done.record(stream)

    def _project_kv(self, hash_ids: torch.Tensor) -> torch.Tensor:
        kv = self._early_kv
        if kv is None or kv.shape[0] != hash_ids.shape[0]:
            raise RuntimeError("Engram projection missing for this step's rows")
        stream = current_stream()
        stream.wait_event(self._projection_done)
        # The tensor was allocated on the projection stream. Its last reader
        # runs on main; retain its storage through that reader before a later
        # allocation on the projection stream.
        kv.record_stream(stream)
        self._early_kv = None
        return kv


def adopt_early_engram(model) -> None:
    """Enable early gathers only for a topology with one forward in flight."""
    if model.vllm_config.parallel_config.use_ubatching:
        raise ValueError("early Engram does not support microbatching")
    mode = model.vllm_config.compilation_config.cudagraph_mode
    if mode is not None and mode.has_piecewise_cudagraphs():
        raise ValueError(
            "early Engram requires eager or full CUDA graphs; "
            "piecewise capture cannot retain a side-stream fork across breaks"
        )
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        engram = getattr(layer, "engram", None)
        if engram is not None:
            EarlyEngram.adopt(engram)
