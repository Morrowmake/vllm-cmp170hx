# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot-time compile of the speculative-decoding Triton kernels that no boot
dummy run reaches (model runner V2).

* Rejection sampling (``_compute_local_logits_stats_kernel``,
  ``_rejection_kernel``, ``_resample_kernel``): the dummy sampler run calls the
  plain sampler only, and the runner's warmup steps
  (``vllm/v1/worker/gpu/warmup.py``) verify with
  ``SamplingParams.for_sampler_warmup()``, whose logits processing hands the
  kernels an FP32 copy of the logits. A request without logits processing
  (greedy, or temperature alone) hands them the model's logits tensor itself,
  of another dtype, so those kernels compiled on the first such request.
  Warmed by ``rejection_sample`` on one dummy request with
  ``num_speculative_tokens`` drafts, once with the model's logits and once
  with their FP32 copy, with the speculator's draft-logits buffer, the
  sampler's state buffers and the rejection sampler's own settings.
* DFlash/DSpark input preparation (``_prepare_dflash_inputs_kernel``): its
  ``BLOCK_SIZE`` constexpr is ``min(256, next_pow2(max query length +
  num_query_per_req))``. Dummy and capture batches are uniform decode, so only
  the decode value compiles at boot; a prefill chunk needs a larger one.
  Warmed by preparing a one-request dummy batch for every reachable value.

Called at the end of the runner's warmup steps (``warmup_kernels``, with
JIT warmup on), before graph capture, and runs only on sm_80 with a DSpark
speculator (DeepSeek-V4.1's; other models' drafters keep upstream's boot).
It makes no collective call: the target logits are a local zero tensor of
the shape and dtype ``compute_logits`` returns, so a rank that fails cannot
leave the others waiting. A failure is logged and does not stop the engine:
the kernels then compile on first use, as before.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# The prepare kernel's BLOCK_SIZE cap (prepare_dflash_inputs).
_MAX_PREPARE_BLOCK = 256


def prepare_block_sizes(num_query_per_req: int, max_num_tokens: int) -> list[int]:
    """Every BLOCK_SIZE ``prepare_dflash_inputs`` can pick, with the query
    length that picks it: ``min(256, next_pow2(q + num_query_per_req))`` for
    ``1 <= q <= max_num_tokens``."""
    out = []
    block = triton.next_power_of_2(1 + num_query_per_req)
    while block <= _MAX_PREPARE_BLOCK:
        q = max(1, block // 2 + 1 - num_query_per_req)
        if q > max_num_tokens:
            break
        out.append((block, q))
        block *= 2
    return out


def warmup_spec_decode(runner: "GPUModelRunner") -> None:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    speculator = getattr(runner, "speculator", None)
    if (
        not isinstance(speculator, DSparkSpeculator)
        or not runner.is_last_pp_rank
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(80)
    ):
        return
    try:
        _warm_rejection_sampler(runner, speculator)
        _warm_dflash_prepare(runner, speculator)
    except Exception:
        logger.warning(
            "Speculative-decoding kernel warmup failed; those kernels compile "
            "on first use instead.",
            exc_info=True,
        )
        return
    torch.accelerator.synchronize()


def _warm_rejection_sampler(runner, speculator) -> None:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    sampler = getattr(runner, "rejection_sampler", None)
    if sampler is None:
        return
    n = sampler.num_speculative_steps + 1
    dev = runner.device
    # compute_logits' output (full vocabulary, head dtype), without its
    # TP gather.
    model_config = runner.model_config
    logits = torch.zeros(
        n, model_config.get_vocab_size(), dtype=model_config.head_dtype, device=dev
    )
    idx_mapping = torch.zeros(1, dtype=torch.int64, device=dev)
    expanded_idx_mapping = torch.zeros(n, dtype=torch.int64, device=dev)
    expanded_local_pos = torch.arange(n, dtype=torch.int32, device=dev)
    cu_num_logits = torch.tensor([0, n], dtype=torch.int32, device=dev)
    draft_sampled = runner.input_buffers.input_ids.new_zeros(n)
    pos = runner.input_buffers.positions.new_zeros(n)
    states = sampler.sampler.sampling_states
    processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    # The arguments RejectionSampler._verify passes.
    for target_logits in (logits, processed):
        rejection_sample(
            target_logits,
            speculator.draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            states.temperature.gpu,
            states.seeds.gpu,
            sampler.num_speculative_steps,
            sampler.synthetic_conditional_rates,
            use_fp64=sampler.sampler.use_fp64_gumbel,
            use_block_verification=sampler.use_block_verification,
            **sampler._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )


def _warm_dflash_prepare(runner, speculator) -> None:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        DFlashSpeculator,
        prepare_dflash_inputs,
    )

    if not isinstance(speculator, DFlashSpeculator):
        return
    dev = runner.device
    num_sampled = torch.ones(1, dtype=torch.int32, device=dev)
    num_rejected = torch.zeros(1, dtype=torch.int32, device=dev)
    tables = speculator.block_tables
    for _, q in prepare_block_sizes(
        speculator.num_query_per_req, runner.max_num_tokens
    ):
        batch = InputBatch.make_dummy(1, q, runner.input_buffers)
        # The argument list of DFlashSpeculator.propose.
        for i, gid in enumerate(speculator.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                speculator.input_buffers,
                tables.slot_mappings[gid],
                speculator.context_positions,
                speculator._context_slot_mappings[i],
                speculator.sample_indices,
                speculator.sample_pos,
                speculator.sample_idx_mapping,
                speculator.temperature,
                speculator.seeds,
                batch,
                num_sampled,
                num_rejected,
                runner.req_states.last_sampled_tokens,
                runner.req_states.next_prefill_tokens,
                runner.sampler.sampling_states.temperature.gpu,
                runner.sampler.sampling_states.seeds.gpu,
                tables.input_block_tables[gid],
                tables.block_sizes[gid],
                tables.cp_rank,
                speculator.dcp_size,
                tables.cp_interleave,
                speculator.parallel_drafting_token_id,
                speculator.num_query_per_req,
                speculator.num_speculative_steps,
                speculator.max_num_reqs,
                speculator.max_num_tokens,
                speculator.max_model_len,
                speculator.sample_from_anchor,
            )
