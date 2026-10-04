# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Morrowmake for CMP 170HX support; see repository history.

import contextlib
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _check_draft_skip_warmup(sweep=False):
    """Interpret the real input and shard kernels on warmup scheduler outputs."""
    from unittest.mock import patch

    from tests.v1.worker.test_gpu_warmup_blocks import _attention_group, _make_runner
    from vllm.v1.worker.gpu.sample import batch_shard
    from vllm.v1.worker.gpu.spec_decode.draft_confidence import (
        DraftConfidence,
        FrozenCoefficients,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.warmup import (
        run_mixed_prefill_decode_warmup,
        warmup_kernels,
    )

    def h2d(values, device=None, dtype=None, out=None):
        tensor = torch.as_tensor(values, dtype=dtype)
        return out.copy_(tensor) if out is not None else tensor

    cases = (
        [
            (width, nreq, pad, skip)
            for width in (1, 3, 7)
            for nreq in (1, 8)
            for pad in (0, 16)
            for skip in (True, False)
        ]
        if sweep
        else [(7, 8, 0, True)]
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(model_runner_module, "async_tensor_h2d", h2d)
        mp.setattr(torch.accelerator, "synchronize", lambda: None)
        mp.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
        mp.setenv("VLLM_MOE_SKIP_PADDING", "0")

        def check_case(width, nreq, pad, skip):
            mp.setenv("VLLM_GLM5_DFLASH_SKIP", str(int(skip)))
            warm_runner = _make_runner([_attention_group()], width + 1, width)
            warm_runner.scheduler_config.max_num_seqs = nreq
            warm_runner.max_num_reqs = 8
            confidence = DraftConfidence(
                8,
                width,
                "cpu",
                FrozenCoefficients(1.0, (0.0,) * width, 0.3) if skip else None,
            )
            runner = GPUModelRunner.__new__(GPUModelRunner)
            runner.device = torch.device("cpu")
            runner.max_num_reqs = 8
            runner.decode_query_len = width + 1
            runner.input_buffers = InputBuffers(8, 256, runner.device)
            runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
            runner.model_config = SimpleNamespace(rswa_window=None)
            runner.adaptive_verification = None
            runner.fast_prefill = None
            runner.pcp_manager = None
            runner.pp_handler = None
            runner.speculator = SimpleNamespace(draft_confidence=confidence)
            runner.req_states = SimpleNamespace(
                num_computed_tokens_np=np.zeros(8, dtype=np.int32),
                num_computed_tokens=SimpleNamespace(
                    gpu=torch.zeros(8, dtype=torch.int32)
                ),
                prefill_len=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
                all_token_ids=SimpleNamespace(gpu=torch.arange(128).repeat(8, 1)),
                next_prefill_tokens=torch.zeros(8, 8, dtype=torch.int32),
                last_sampled_tokens=torch.arange(8, dtype=torch.int32) + 20,
                draft_tokens=torch.arange(8 * width, dtype=torch.int32).view(8, width),
            )
            slots = {}
            current = None
            steps = []

            def execute(output):
                nonlocal current
                for req in output.scheduled_new_reqs:
                    slot = slots.setdefault(req.req_id, len(slots))
                    runner.req_states.prefill_len.gpu[slot] = len(req.prompt_token_ids)
                cached = output.scheduled_cached_reqs
                for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
                    slot = slots[req_id]
                    runner.req_states.num_computed_tokens_np[slot] = computed
                    runner.req_states.num_computed_tokens.gpu[slot] = computed
                if not output.num_scheduled_tokens:
                    return
                ids = list(output.num_scheduled_tokens)
                mapping = np.array([slots[r] for r in ids], dtype=np.int32)
                computed = runner.req_states.num_computed_tokens_np[mapping]
                prefill = runner.req_states.prefill_len.gpu[mapping].numpy()
                prefilling = computed < prefill
                state = SimpleNamespace(
                    num_tokens=output.total_num_scheduled_tokens,
                    req_ids=ids,
                    num_scheduled_tokens=np.array(
                        list(output.num_scheduled_tokens.values()), dtype=np.int32
                    ),
                    idx_mapping_np=mapping,
                    has_prefill=bool(prefilling.any()),
                    prefill_len_np=prefill,
                    num_computed_prefill_tokens_np=np.minimum(computed, prefill),
                    is_prefilling_np=prefilling,
                )
                # Distinct caps expose a wrong contiguous slice or owner ordering.
                confidence.positions[mapping, 0] = torch.from_numpy(computed + 1).long()
                confidence.caps[mapping] = torch.from_numpy(mapping % (width + 1))
                if sweep:
                    # One slot carries a prediction from an earlier prefix.
                    confidence.positions[mapping[-1], 0] -= 1
                desc = SimpleNamespace(
                    num_tokens=state.num_tokens + pad, num_reqs=8 if pad else len(ids)
                )
                current = runner.prepare_inputs(output, state, desc, 0)
                expected = []
                for slot, length, is_prefill in zip(
                    mapping, state.num_scheduled_tokens, prefilling
                ):
                    valid = (
                        skip and not is_prefill and not (sweep and slot == mapping[-1])
                    )
                    expected.extend(
                        valid and row > slot % (width + 1) for row in range(length)
                    )
                token_mask = torch.tensor(expected, dtype=torch.bool)
                if skip:
                    assert torch.equal(
                        current.is_padding[: state.num_tokens], token_mask
                    )
                    assert current.is_padding[state.num_tokens :].all()
                    assert torch.equal(
                        current.draft_skip_mask, token_mask[current.logits_indices]
                    )
                steps.append(current.logits_indices.numel())

            def sample(_grammar):
                assert current is not None
                if not sweep and not current.num_draft_tokens:
                    return
                global_mask = current.draft_skip_mask
                for rank in range(4):
                    group = SimpleNamespace(rank_in_group=rank, world_size=4)
                    with patch.object(batch_shard, "get_tp_group", return_value=group):
                        sharder = batch_shard.BatchSharder(8, width + 1, runner.device)
                    local, _, _, _ = sharder.shard_sampler_inputs(current, None)
                    owned = np.flatnonzero(current.idx_mapping_np % 4 == rank)
                    rows = [
                        i
                        for req in owned
                        for i in range(
                            current.cu_num_logits_np[req],
                            current.cu_num_logits_np[req + 1],
                        )
                    ]
                    expected_mask = global_mask[rows] if skip else None
                    assert torch.equal(
                        local.logits_indices, current.logits_indices[rows]
                    )
                    expected = current.input_ids[local.logits_indices].clone()
                    if skip:
                        expected.masked_fill_(expected_mask, -1)

                    def verify(
                        _logits,
                        _batch,
                        _draft_logits,
                        proposals,
                        *_args,
                        expected=expected,
                    ):
                        assert torch.equal(proposals, expected)
                        return (
                            torch.zeros(_batch.num_reqs, width + 1, dtype=torch.int64),
                            torch.ones(_batch.num_reqs, dtype=torch.int32),
                            None,
                        )

                    if local.num_draft_tokens:
                        sampler = RejectionSampler.__new__(RejectionSampler)
                        sampler.sampler = SimpleNamespace(
                            compute_nans=False,
                            sampling_states=SimpleNamespace(
                                max_num_logprobs=lambda _: 0
                            ),
                            req_states=runner.req_states,
                        )
                        sampler._verify_in_chunks = verify
                        sampler(torch.zeros(local.logits_indices.numel(), 64), local)
                    if skip:
                        assert local.draft_skip_mask.shape == local.logits_indices.shape
                        assert torch.equal(local.draft_skip_mask, expected_mask)
                    else:
                        assert local.draft_skip_mask is None
                assert current.draft_skip_mask is global_mask

            warmup_kernels(warm_runner, execute, sample)
            assert steps[1] == nreq * (width + 1)
            if sweep:
                slots.clear()
                runner.req_states.num_computed_tokens_np.fill(0)
                runner.req_states.num_computed_tokens.gpu.zero_()
                assert run_mixed_prefill_decode_warmup(warm_runner, execute, sample, 32)

        for case in cases:
            check_case(*case)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "sweep", [False, True], ids=["warmup_tp4", "batch_shape_sweep"]
)
def test_draft_skip_mask_follows_sampler_logits_on_cpu(sweep):
    """Warmup's actual scheduler batches must survive TP-local sampling."""
    code = (
        "import runpy; "
        f"check = runpy.run_path({__file__!r})['_check_draft_skip_warmup']; "
        f"check(sweep={sweep!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            TRITON_INTERPRET="1",
            CUDA_VISIBLE_DEVICES="",
            VLLM_TARGET_DEVICE="cpu",
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _prepare_padding_batch(monkeypatch, lengths, padding, confidence=None):
    """Run prepare_inputs on CPU; replace only device input-building kernels."""
    nreq, ntok = len(lengths), sum(lengths)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = nreq
    runner.decode_query_len = 8
    runner.input_buffers = InputBuffers(nreq, len(padding), runner.device)
    runner.input_buffers.is_padding.copy_(padding)
    runner.input_buffers.positions[:ntok] = torch.cat(
        [torch.arange(10, 10 + length) for length in lengths]
    )
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.adaptive_verification = None
    runner.fast_prefill = None
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.speculator = SimpleNamespace(draft_confidence=confidence)
    runner.req_states = SimpleNamespace(
        num_computed_tokens_np=np.full(nreq, 10, dtype=np.int32),
        num_computed_tokens=SimpleNamespace(gpu=torch.full((nreq,), 10)),
        last_sampled_tokens=None,
        prefill_len=SimpleNamespace(gpu=torch.zeros(nreq)),
        draft_tokens=None,
    )

    def h2d(values, device=None, dtype=None, out=None):
        tensor = torch.as_tensor(values, dtype=dtype)
        return out.copy_(tensor) if out is not None else tensor

    def expand(mapping, total, boundaries, _decode_len):
        counts = boundaries[1:] - boundaries[:-1]
        return mapping.repeat_interleave(counts), torch.cat(
            [torch.arange(int(count)) for count in counts]
        )

    monkeypatch.setattr(model_runner_module, "async_tensor_h2d", h2d)
    monkeypatch.setattr(model_runner_module, "expand_idx_mapping", expand)
    monkeypatch.setattr(model_runner_module, "prepare_pos_seq_lens", lambda *a: None)
    monkeypatch.setattr(
        model_runner_module,
        "combine_sampled_and_draft_tokens",
        lambda *a: torch.arange(ntok),
    )
    req_ids = [str(i) for i in range(nreq)]
    scheduler = SimpleNamespace(
        scheduled_spec_decode_tokens={
            req_id: [1] * (length - 1)
            for req_id, length in zip(req_ids, lengths)
            if length > 1
        },
        has_structured_output_requests=False,
    )
    state = SimpleNamespace(
        num_tokens=ntok,
        req_ids=req_ids,
        num_scheduled_tokens=np.array(lengths, dtype=np.int32),
        idx_mapping_np=np.arange(nreq),
        has_prefill=False,
        prefill_len_np=np.zeros(nreq, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(nreq, dtype=np.int32),
        is_prefilling_np=np.zeros(nreq, dtype=np.bool_),
    )
    desc = SimpleNamespace(num_tokens=len(padding), num_reqs=nreq)
    return runner, runner.prepare_inputs(scheduler, state, desc, 0)


@pytest.mark.parametrize("generic_skip", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_prepare_inputs_skip_off_preserves_base_padding_and_routes(
    monkeypatch, generic_skip, seed
):
    """MASK_PADDING alone must preserve the pre-skip buffer/routing contract."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    monkeypatch.setenv("VLLM_GLM5_DFLASH_SKIP", "0")
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", "1")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", str(int(generic_skip)))
    monkeypatch.setenv("VLLM_GLM5_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MOE_MAX_TOKENS", "8")
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 9, (1 + seed % 4,), generator=g).tolist()
    ntok = sum(lengths)
    padding = torch.randint(0, 2, (ntok + seed % 5 + 3,), generator=g).bool()
    expected = padding.clone()
    # Before draft skipping, only the upstream switch initialized this buffer.
    if generic_skip:
        expected[:ntok] = False
        expected[ntok:] = True
    runner, batch = _prepare_padding_batch(monkeypatch, lengths, padding)
    assert batch.is_padding.data_ptr() == runner.input_buffers.is_padding.data_ptr()
    assert torch.equal(batch.is_padding, expected)
    assert torch.equal(runner.input_buffers.is_padding, expected)

    monkeypatch.setattr(mr, "_MASK_PADDING", None)
    monkeypatch.setattr(mr, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mr, "get_forward_context", lambda: batch)
    ids = torch.randint(0, 288, (len(padding), 8), generator=g, dtype=torch.int32)
    expected_ids = ids.clone()
    if len(padding) > 8:
        expected_ids.masked_fill_(expected[:, None], -1)
    assert mr.mask_padding_topk_ids(ids) is ids
    assert torch.equal(ids, expected_ids)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("generic_skip", [False, True])
@pytest.mark.parametrize("mask_padding", [False, True])
def test_prepare_inputs_fills_before_applying_draft_skip(
    monkeypatch, generic_skip, mask_padding
):
    """Clear stale masks first, then retain skipped live rows and graph padding."""
    from vllm.v1.worker.gpu.spec_decode.draft_confidence import (
        DraftConfidence,
        FrozenCoefficients,
    )

    monkeypatch.setenv("VLLM_GLM5_DFLASH_SKIP", "1")
    monkeypatch.setenv("VLLM_GLM5_MOE_MASK_PADDING", str(int(mask_padding)))
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", str(int(generic_skip)))
    predictor = DraftConfidence(2, 3, "cpu", FrozenCoefficients(1.0, (0.0,) * 3, 0.3))
    predictor.positions[:2, 0] = 11
    predictor.caps[:2] = torch.tensor([1, 2])
    apply_mask = predictor.apply_mask
    calls = []

    def observe_fill(batch, is_prefilling):
        calls.append(batch.is_padding.clone())
        assert batch.is_padding.tolist() == [False] * 8 + [True] * 4
        apply_mask(batch, is_prefilling)

    monkeypatch.setattr(predictor, "apply_mask", observe_fill)
    runner, batch = _prepare_padding_batch(
        monkeypatch, [4, 4], torch.ones(12, dtype=torch.bool), predictor
    )
    expected = [False, False, True, True, False, False, False, True] + [True] * 4
    assert len(calls) == 1
    assert batch.is_padding.tolist() == expected
    assert runner.input_buffers.is_padding.tolist() == expected
    assert batch.draft_skip_mask.tolist() == expected[:8]
    assert not torch.cuda.is_initialized()


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""
    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
        warn_on_missing_adaptive_graphs=lambda: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []
