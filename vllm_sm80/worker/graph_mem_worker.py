# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's GPU worker with the fork's V2 model runner.

The fork platform names :class:`GraphMemWorker` as the worker
(``cudagraph_mem.select_worker``) when either of its runner's features is
on:

- ``VLLM_CUDAGRAPH_ESTIMATE_SPLIT`` (``cudagraph_mem``):
  ``profile_cudagraph_memory`` and ``capture_model`` are routed through
  ``cudagraph_mem``, which also has the profiling capture capture every FULL
  graph;
- ``VLLM_SM80_SHARD_ARGMAX`` (``target_argmax``): ``sample``
  runs inside ``target_argmax.step``, so a greedy-only step's target logits
  are the per-rank argmax rows; ``load_model`` ends in
  ``target_argmax.load``, which refuses a target, head or drafter the switch
  cannot serve and takes a ``sample_draft`` drafter's greedy drafts per
  rank; and the dummy sampler run at boot warms the slot kernels. The hot
  pair ``shard_argmax`` (``vllm_sm80.hot_swap``) needs the same runner and
  turns the target half on and off between steps (``set_enabled``).

A feature that is off leaves its methods as upstream's; everything else is
upstream's. Other runners (V1, the encoder-only runner) are built unchanged:
they keep vLLM's estimate, and the greedy argmax refuses them.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_sm80 import hot_swap
from vllm_sm80.worker import cudagraph_mem, target_argmax
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@functools.cache
def runner_cls() -> type:
    """vLLM's V2 model runner with the fork's features (imported on first
    use, as ``Worker._make_model_runner`` imports the runner)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

    class GraphMemModelRunner(GPUModelRunnerV2):
        # target_argmax.refusal: sample() runs in target_argmax.step.
        target_argmax_steps = True

        def load_model(self, *args, **kwargs) -> None:
            super().load_model(*args, **kwargs)
            target_argmax.load(self)

        def profile_cudagraph_memory(self) -> int:
            if not cudagraph_mem.enabled():
                return super().profile_cudagraph_memory()
            return cudagraph_mem.profile(
                GPUModelRunnerV2.profile_cudagraph_memory, self
            )

        def capture_model(self, *, profile_only: bool = False) -> int:
            if not cudagraph_mem.enabled():
                return super().capture_model(profile_only=profile_only)
            if profile_only:
                cudagraph_mem.measure_full_graphs(self.cudagraph_manager)
            return cudagraph_mem.capture(super().capture_model, profile_only)

        def _dummy_sampler_run(self, hidden_states):
            super()._dummy_sampler_run(hidden_states)
            target_argmax.warm(self, hidden_states)

        def sample(self, hidden_states, input_batch, grammar_output):
            with target_argmax.step(self, input_batch, grammar_output):
                return super().sample(hidden_states, input_batch, grammar_output)

    return GraphMemModelRunner


class GraphMemWorker(Worker):
    def _make_model_runner(self) -> GPUModelRunner:
        if self.use_v2_model_runner and not self.vllm_config.is_mm_encoder_only:
            return runner_cls()(self.vllm_config, self.device)  # type: ignore[return-value]
        if getattr(envs, target_argmax.SWITCH) or hot_swap.listed("shard_argmax"):
            raise ValueError(
                f"{target_argmax.SWITCH}: the greedy target argmax needs the V2 "
                "model runner of a generative model"
            )
        logger.warning_once(
            "VLLM_CUDAGRAPH_ESTIMATE_SPLIT=1 applies to the V2 model runner "
            "only; this worker keeps vLLM's CUDA-graph memory estimate."
        )
        return super()._make_model_runner()
