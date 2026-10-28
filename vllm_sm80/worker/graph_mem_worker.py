# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's GPU worker with the fork's V2 model runner.

The fork platform names :class:`GraphMemWorker` as the worker
(``cudagraph_mem.select_worker``) whenever the worker is vLLM's GPU worker.
Its V2 runner (:func:`runner_cls`) carries the fork's runner features; each
is a no-op with its switch off:

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
  turns the target half on and off between steps (``set_enabled``);
- the A/B capture of ``VLLM_DEV_AB_SWITCH`` and the hot pairs
  (``dev_ab.with_ab_capture``) around vLLM's ``capture_model``;
- adaptive-k (``adaptive_k_graphs``): the startup check that every draft
  count has a decode graph, and the per-step graph-miss counter;
- the draft width managers of the plugin speculators: the profiling capture's
  per-width managers are dropped before vLLM's profiling teardown;
- the moved draft tail (``VLLM_PP_DRAFT_TAIL_STAGE``): the draft width each
  step's tail runs at, handed to the PP handler before it receives;
- the load inventory (``mem_attribution``, VLLM_MEM_ATTRIBUTION).

The worker adds the Marlin MoE decode extension check
(``VLLM_AMPERE_MARLIN_DECODE_CUDA``) before device init and finalizes the
moved draft tail after warm-up. The calls the runner and worker make from
inside vLLM's methods are ``runner_hooks`` and ``worker_hooks``.

Other runners (V1, the encoder-only runner) are built unchanged: they keep
vLLM's estimate, and the greedy argmax refuses them.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker
from vllm_sm80 import hot_swap
from vllm_sm80.worker import cudagraph_mem, target_argmax

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@functools.cache
def runner_cls() -> type:
    """vLLM's V2 model runner with the fork's features (imported on first
    use, as ``Worker._make_model_runner`` imports the runner)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm_sm80.dev_ab import with_ab_capture
    from vllm_sm80.worker import adaptive_k_graphs, mem_attribution

    # vLLM's capture_model behind the A/B capture, as one inference-mode call.
    ab_capture = torch.inference_mode()(
        with_ab_capture(GPUModelRunnerV2.capture_model)
    )

    class GraphMemModelRunner(GPUModelRunnerV2):
        # target_argmax.refusal: sample() runs in target_argmax.step.
        target_argmax_steps = True

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Adaptive-k SD: the scheduler picks this step's draft count on
            # the host and ships it as num_spec_tokens_to_schedule.
            config = adaptive_k_graphs.adaptive_k_config(self.vllm_config)
            self.adaptive_k_enabled = bool(config is not None and config.allowed)
            # Decode steps that missed their decode CUDA graph mode.
            self._adaptive_k_cg_misses = 0

        def load_model(self, *args, **kwargs) -> None:
            super().load_model(*args, **kwargs)
            mem_attribution.log_load_inventory(
                [
                    ("target", self.model),
                    ("drafter", getattr(self.speculator, "model", None)),
                ],
                self.model_memory_usage,
            )
            target_argmax.load(self)

        def profile_cudagraph_memory(self) -> int:
            if not cudagraph_mem.enabled():
                return super().profile_cudagraph_memory()
            return cudagraph_mem.profile(
                GPUModelRunnerV2.profile_cudagraph_memory, self
            )

        def capture_model(self, *, profile_only: bool = False) -> int:
            if not cudagraph_mem.enabled():
                size = ab_capture(self, profile_only=profile_only)
            else:
                if profile_only:
                    cudagraph_mem.measure_full_graphs(self.cudagraph_manager)
                size = cudagraph_mem.capture(
                    functools.partial(ab_capture, self), profile_only
                )
            if self.cudagraph_manager.needs_capture():
                adaptive_k_graphs.warn_missing(self.cudagraph_manager)
            if profile_only:
                # vLLM's profiling retires the speculator's managers held as
                # attributes; the per-width ones go here, before its teardown
                # releases the profiling pool.
                retire = getattr(self.speculator, "retire_width_managers", None)
                if retire is not None:
                    retire()
            return size

        def _dummy_sampler_run(self, hidden_states):
            super()._dummy_sampler_run(hidden_states)
            target_argmax.warm(self, hidden_states)

        def sample(self, hidden_states, input_batch, grammar_output):
            with target_argmax.step(self, input_batch, grammar_output):
                return super().sample(hidden_states, input_batch, grammar_output)

        def sample_tokens(self, grammar_output):
            # The moved draft tail runs at the width this step drafts.
            state = self.execute_model_state
            if state is not None and self.pp_handler is not None:
                self.pp_handler.next_width = state.num_spec_tokens_to_schedule
            return super().sample_tokens(grammar_output)

    return GraphMemModelRunner


class GraphMemWorker(Worker):
    def init_device(self):
        if envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
            from vllm_sm80.kernels.marlin_moe.extension import require_extension

            require_extension()
        return super().init_device()

    def _make_model_runner(self) -> GPUModelRunner:
        if self.use_v2_model_runner and not self.vllm_config.is_mm_encoder_only:
            return runner_cls()(self.vllm_config, self.device)  # type: ignore[return-value]
        if getattr(envs, target_argmax.SWITCH) or hot_swap.listed("shard_argmax"):
            raise ValueError(
                f"{target_argmax.SWITCH}: the greedy target argmax needs the V2 "
                "model runner of a generative model"
            )
        if cudagraph_mem.enabled():
            logger.warning_once(
                "VLLM_CUDAGRAPH_ESTIMATE_SPLIT=1 applies to the V2 model runner "
                "only; this worker keeps vLLM's CUDA-graph memory estimate."
            )
        return super()._make_model_runner()

    def _compile_or_warm_up_model(self):
        result = super()._compile_or_warm_up_model()
        # VLLM_PP_DRAFT_TAIL_STAGE: every stage, once, after graph capture.
        draft_tail = getattr(self.model_runner, "draft_tail", None)
        if self.use_v2_model_runner and draft_tail is not None:
            draft_tail.finalize()
        return result
