# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for DSpark width selection, sampling and graph geometry."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm_sm80.envs import FORK_ENV_VARS
from vllm_sm80.spec_decode import dspark_width as width_module
from vllm_sm80.spec_decode.adaptive_k import AdaptiveKConfig
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as upstream

pytestmark = pytest.mark.cpu_test


def _config(schedule=((1, 3, 5), (4, 8, 3))):
    spec = SimpleNamespace(
        num_speculative_tokens=5,
        num_speculative_tokens_per_batch_size=list(schedule),
        adaptive_k_config=None,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        uses_dynamic_speculative_decoding=lambda: bool(schedule),
    )
    return SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        # Speculator.max_model_len reads the model config.
        model_config=SimpleNamespace(max_model_len=128),
    )


def _init_cpu(self, config, device):
    self.vllm_config = config
    self.device = device
    self.max_num_reqs = 8
    self.max_num_tokens = 128
    self.max_width = self.width = self.num_speculative_steps = 5
    self.widths = (5,)
    self.sample_from_anchor = True
    self.num_query_per_req = 5
    self._step_cols = torch.arange(5, dtype=torch.int32)
    self._anchor_idx = torch.arange(8) * 5
    self.draft_tokens = torch.full((8, 5), -1, dtype=torch.int64)
    self._full_draft_tokens = self.draft_tokens
    self._draft_token_views = {5: self.draft_tokens}
    self.sample_col = torch.arange(5, dtype=torch.int32).repeat(8)
    self._sample_cols = {5: self.sample_col}
    self.draft_token_confidence_probs = torch.full((8, 5), -1.0)
    self.draft_logits = None
    self.draft_logits_num_shards = 1
    self.draft_logits_shard_idx = 0
    self._cg_managers = {}
    self.query_cudagraph_manager = None
    self._draft_topk = None
    self.use_confidence_head = False
    self._context_slot_mappings = torch.zeros((1, 128), dtype=torch.int64)
    self.sample_indices = torch.zeros(40, dtype=torch.int64)
    self.sample_pos = torch.zeros(40, dtype=torch.int64)
    self.sample_idx_mapping = torch.full((40,), -1, dtype=torch.int64)
    self.input_buffers = SimpleNamespace(input_ids=torch.arange(48))
    self.model = SimpleNamespace()


@pytest.fixture
def drafter(monkeypatch):
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", _init_cpu)
    return width_module.PerKDSparkSpeculator(_config(), torch.device("cpu"))


def test_default_off_is_original_class(monkeypatch):
    monkeypatch.delenv("VLLM_DSPARK_PER_K_WIDTH", raising=False)
    assert FORK_ENV_VARS["VLLM_DSPARK_PER_K_WIDTH"]() is False
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", _init_cpu)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", False)
    result = width_module.create_dspark_speculator(_config(), torch.device("cpu"))
    assert type(result) is upstream.DSparkSpeculator
    result.set_width(3)
    assert result.width == 5
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", True)
    result = width_module.create_dspark_speculator(_config(), torch.device("cpu"))
    assert type(result) is width_module.PerKDSparkSpeculator
    assert result.widths == (3, 5)


def test_non_anchor_mode_is_refused_before_base_initialization(monkeypatch):
    config = _config()
    config.speculative_config.draft_model_config.hf_config.sample_from_anchor = False
    base_init = MagicMock(return_value=None)
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", base_init)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", True)
    with pytest.raises(ValueError, match="requires sample_from_anchor=True"):
        width_module.create_dspark_speculator(config, torch.device("cpu"))
    base_init.assert_not_called()
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", False)
    result = width_module.create_dspark_speculator(config, torch.device("cpu"))
    assert type(result) is upstream.DSparkSpeculator
    base_init.assert_called_once()


@pytest.mark.parametrize("schedule,expected", [
    (((1, 3, 5), (4, 8, 3)), (3, 5)),
    (((1, 2, 8), (4, 6, 2), (7, 8, 0)), (2, 5)),
    (((1, 8, 5), (9, 20, 1)), (5,)),
    ((), (5,)),
])
def test_widths_reuse_scheduler_lookup(schedule, expected):
    assert width_module.draft_widths(_config(schedule)) == expected


def test_adaptive_widths_require_draft_by_load():
    cfg = _config(())
    for enabled, expected in ((False, (5,)), (True, (2, 4, 5))):
        cfg.speculative_config.adaptive_k_config = AdaptiveKConfig.from_dict(
            {"by_load": [5, 4, 2], "draft_by_load": enabled}, 5
        )
        assert width_module.draft_widths(cfg) == expected


def test_anchors_and_views_survive_width_transitions(drafter):
    anchors = {w: a.clone() for w, a in drafter._anchor_indices_by_width.items()}
    pointers = {w: a.data_ptr() for w, a in drafter._anchor_indices_by_width.items()}
    logits = drafter.draft_logits = torch.empty(8, 5, 11)
    managers = drafter._cg_managers = {3: object(), 5: object()}
    for requested, selected in ((3, 3), (5, 5), (3, 3), (0, 5), (4, 5)):
        drafter.set_width(requested)
        qlen = selected
        assert drafter.num_query_per_req == qlen
        assert drafter.num_speculative_steps == selected
        assert drafter.query_cudagraph_manager is managers[selected]
        assert torch.equal(drafter._anchor_idx, torch.arange(8) * qlen)
        assert drafter._anchor_idx.data_ptr() == pointers[selected]
        assert drafter.draft_tokens.is_contiguous()
        assert drafter.draft_tokens.shape == (8, selected)
        assert drafter.draft_tokens.data_ptr() == drafter._full_draft_tokens.data_ptr()
        assert drafter.draft_logits is logits and logits.stride() == (55, 11, 1)
        assert drafter.draft_token_confidence_probs.shape == (8, 5)
    for w, snapshot in anchors.items():
        assert torch.equal(drafter._anchor_indices_by_width[w], snapshot)


@pytest.mark.parametrize("width", [3, 5])
@pytest.mark.parametrize("num_reqs", [4, 6, 8])
@pytest.mark.parametrize("topk", [None, 11])
def test_markov_walk_uses_current_anchors_and_k_columns(drafter, width, num_reqs, topk):
    drafter.set_width(width)
    drafter._draft_topk = topk
    drafter.use_confidence_head = True
    drafter.input_buffers.input_ids.fill_(10)
    drafter.input_buffers.input_ids[drafter._anchor_idx] = torch.arange(8)
    sample_count = num_reqs * width
    drafter.sample_indices[:sample_count] = torch.arange(sample_count)
    # Nonidentity request slots, including an inert padded row.
    slots = torch.tensor([7, 2, 5, 0, 3, 6, 1, -1])[:num_reqs]
    drafter.sample_idx_mapping[:sample_count] = slots.repeat_interleave(width)
    drafter.sample_pos[:sample_count] = torch.arange(101, 101 + width).repeat(num_reqs)
    calls = []
    drafter.model.compute_draft_logits = lambda x: torch.zeros(x.shape[0], 11)
    drafter.model.markov_embed = lambda previous: previous[:, None].float()
    drafter.model.markov_bias = lambda emb: -(
        torch.arange(11)[None, :] - (emb + 1) % 11
    ).abs()
    drafter.model.apply_markov_bias_gathered = (
        lambda emb, logits, values, indices: drafter.model.markov_bias(emb)
    )
    drafter.model.compute_confidence = lambda hidden, emb: emb[:, 0] / 11

    def sample(logits, idx, pos, step):
        calls.append((step, idx.clone(), pos.clone()))
        return logits.argmax(-1)

    drafter._sample_logits = sample
    drafter._sample_sequential(num_reqs, torch.zeros(sample_count, 2))
    expected = (torch.arange(num_reqs)[:, None] + torch.arange(1, width + 1)) % 11
    assert torch.equal(drafter.draft_tokens[:num_reqs], expected)
    assert [call[0] for call in calls] == list(range(width))
    for step, idx, pos in calls:
        assert torch.equal(idx, slots)
        assert torch.equal(pos, torch.full((num_reqs,), 101 + step))
    expected_confidence = (expected - 1).remainder(11).float() / 11
    assert torch.equal(drafter.draft_token_confidence_probs[:num_reqs, :width],
                       expected_confidence)
    # Same shape/strides handed to the unchanged adaptive-verification cache.
    published = torch.empty(8, 5)
    published[:num_reqs] = drafter.draft_token_confidence_probs[:num_reqs]
    assert torch.equal(published[:num_reqs, :width], expected_confidence)
    if width < 5:
        assert (drafter._confidence_by_width[5][:, width:] == -1).all()


def test_probabilistic_sampling_keeps_max_width_cache(drafter, monkeypatch):
    drafter.draft_logits = torch.full((8, 5, 11), -99.0)
    drafter._d2t_scatter_index = None
    drafter.draft_watermarker = None
    drafter.temperature = torch.ones(8)
    drafter.seeds = torch.arange(8)
    drafter.use_fp64_gumbel = True
    drafter._maybe_predict_acceptance = lambda *args: None
    seen = []

    def sample(logits, idx, temperature, seeds, positions, **kwargs):
        seen.append((positions.clone(), kwargs))
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(upstream, "gumbel_sample", sample)
    cache = drafter.draft_logits
    for width in (5, 3, 5):
        drafter.set_width(width)
        for step in range(width):
            drafter._sample_logits(torch.zeros(2, 11), torch.tensor([6, 1]),
                                  torch.tensor([100, 200]) + step, step)
            positions, kwargs = seen[-1]
            assert kwargs["logits_cache"] is cache
            assert kwargs["logits_cache_col"].item() == step
            assert kwargs["is_drafting"] and kwargs["use_fp64"]
            assert positions.tolist() == [99 + step, 199 + step]


def test_capture_geometry_and_k5_descriptors_unchanged(drafter, monkeypatch):
    cfg = MagicMock(spec=VllmConfig)
    cfg.scheduler_config = SimpleNamespace(max_num_seqs=8)
    cfg.parallel_config = SimpleNamespace(tensor_parallel_size=4)
    cfg.speculative_config = _config().speculative_config
    cfg.num_speculative_tokens = 5
    cfg.compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[3, 4, 5, 6, 8, 12, 15, 18, 20, 24, 30, 32, 40],
    )
    cfg.compilation_config.max_cudagraph_capture_size = 40
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: SimpleNamespace(
        is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None)
    drafter.vllm_config = cfg
    drafter.attn_cg_support = SimpleNamespace(min_cg_support=AttentionCGSupport.UNIFORM_BATCH)
    drafter.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    original = DFlashCudaGraphManager(cfg, torch.device("cpu"),
                                     CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=5)
    assert drafter._cg_managers[5]._capture_descs == original._capture_descs
    for width, manager in drafter._cg_managers.items():
        manager._graphs_captured = True
        for n in (4, 6, 8):
            desc = manager.dispatch(n, n * width, uniform_token_count=width,
                                    num_active_loras=0)
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert (desc.num_tokens, desc.num_reqs, desc.uniform_token_count) == (
                n * width, n, width)
    captured = []

    def capture(manager, forward, *args, **kwargs):
        captured.append((drafter.width, manager.decode_query_len,
                         drafter._anchor_idx.clone(), drafter.draft_tokens.shape))

    monkeypatch.setattr(DFlashCudaGraphManager, "capture", capture)
    drafter.block_tables = drafter.attn_groups = drafter.kv_cache_config = None
    drafter._group_causal = False
    drafter.capture()
    assert [(w, q) for w, q, _, _ in captured] == [(3, 3), (5, 5)]
    for w, _, anchors, shape in captured:
        assert torch.equal(anchors, torch.arange(8) * w)
        assert shape == (8, w)
    assert drafter.width == 5
    assert drafter.query_cudagraph_manager is drafter._cg_managers[5]
