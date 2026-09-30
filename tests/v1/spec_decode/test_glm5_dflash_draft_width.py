# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-following DFlash draft width (VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH)."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.adaptive_k import AdaptiveKConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import pp_draft_tail as dt
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as sm

FLAG = "VLLM_GLM5_DFLASH_ADAPTIVE_K"
WIDTH_FLAG = "VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH"
BY_LOAD_WIDTH = {"by_load": [5, 4, 3], "draft_by_load": True}
MAX_REQS, TOP_K, H, VOCAB = 8, 4, 16, 97


def _clear_once_caches() -> None:
    import vllm.logger as vllm_logger_module

    for name in ("_print_warning_once", "_print_info_once"):
        fn = getattr(vllm_logger_module, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


# ------------------------------------------------------------------ the flag


def test_flag_declared_default_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(WIDTH_FLAG, raising=False)
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH is False
    monkeypatch.setenv(WIDTH_FLAG, "1")
    assert envs.VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH is True


def _spec(**overrides):
    attrs = dict(method="dflash", num_speculative_tokens=3, adaptive_k=None,
                 num_speculative_tokens_per_batch_size=None,
                 enable_adaptive_verification=False)
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_width_flag_off_leaves_the_load_config_unchanged(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv(WIDTH_FLAG, raising=False)
    monkeypatch.delenv("VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS", raising=False)
    spec = _spec()
    SpeculativeConfig._maybe_enable_glm5_load_adaptive_depth(spec)
    assert spec.adaptive_k == {"by_load": [5, 4, 3], "log_interval": 0}


def test_width_flag_adds_draft_by_load(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(WIDTH_FLAG, "1")
    monkeypatch.delenv("VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS", raising=False)
    spec = _spec()
    SpeculativeConfig._maybe_enable_glm5_load_adaptive_depth(spec)
    assert spec.num_speculative_tokens == 5
    assert spec.adaptive_k["draft_by_load"] is True
    config = AdaptiveKConfig.from_dict(spec.adaptive_k, 5)
    assert config.draft_by_load and config.allowed == (3, 4, 5)


def test_width_flag_alone_says_it_is_off(monkeypatch, caplog_vllm):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setenv(WIDTH_FLAG, "1")
    _clear_once_caches()
    spec = _spec()
    before = dict(vars(spec))
    SpeculativeConfig._maybe_enable_glm5_load_adaptive_depth(spec)
    assert vars(spec) == before
    assert any("needs VLLM_GLM5_DFLASH_ADAPTIVE_K=1" in r.getMessage()
               for r in caplog_vllm.records)


def test_draft_by_load_needs_by_load():
    with pytest.raises(ValueError, match="needs adaptive_k.by_load"):
        AdaptiveKConfig.from_dict({"min": 1, "max": 5, "draft_by_load": True}, 5)


@pytest.mark.parametrize(
    "sample,adaptive,dp,expect",
    [("greedy", False, 1, (3, 4, 5)), ("probabilistic", False, 1, ()),
     ("greedy", True, 1, ()), ("greedy", False, 2, ())],
)
def test_drafter_gate(sample, adaptive, dp, expect):
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            adaptive_k_config=AdaptiveKConfig.from_dict(BY_LOAD_WIDTH, 5),
            draft_sample_method=sample, enable_adaptive_verification=adaptive),
        parallel_config=SimpleNamespace(data_parallel_size=dp))
    assert sm.load_following_draft_widths(cfg) == expect


# ------------------------------------------------------------- the scheduler


def _make_scheduler(adaptive_k):
    base = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192,
                            num_speculative_tokens=5)
    spec = base.vllm_config.speculative_config
    spec.adaptive_k = adaptive_k
    spec.adaptive_k_config = AdaptiveKConfig.from_dict(adaptive_k, 5)
    return Scheduler(vllm_config=base.vllm_config,
                     kv_cache_config=base.kv_cache_config,
                     block_size=base.block_size, log_stats=True,
                     structured_output_manager=StructuredOutputManager(
                         base.vllm_config))


@pytest.mark.parametrize("num_requests,k", [(1, 5), (2, 4), (3, 3), (8, 3)])
def test_scheduler_drafts_the_width_it_verifies(num_requests, k):
    scheduler = _make_scheduler(BY_LOAD_WIDTH)
    requests = create_requests(num_requests=num_requests)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()
    for request in requests:
        request.spec_token_ids = [11, 12, 13, 14, 15][:k]
    output = scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == k
    # A batch of 8 asks the drafter for a width-3 block, not the full 5.
    assert output.num_spec_tokens_to_schedule == k


def test_full_block_without_draft_by_load():
    scheduler = _make_scheduler({"by_load": [5, 4, 3]})
    requests = create_requests(num_requests=8)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()
    for request in requests:
        request.spec_token_ids = [11, 12, 13, 14, 15]
    output = scheduler.schedule()
    assert (scheduler.cur_num_spec_tokens, output.num_spec_tokens_to_schedule) == (3, 5)


def test_width_lags_one_step_when_load_drops():
    scheduler = _make_scheduler(BY_LOAD_WIDTH)
    requests = create_requests(num_requests=3)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()
    for request in requests:
        request.spec_token_ids = [11, 12, 13]
    scheduler.schedule()
    for request in requests[1:]:
        scheduler.running.remove(request)
    # The survivor holds a width-3 block: verify 3, ask for 5 ...
    requests[0].spec_token_ids = [11, 12, 13]
    output = scheduler.schedule()
    assert (scheduler.cur_num_spec_tokens, output.num_spec_tokens_to_schedule) == (3, 5)
    # ... and verify 5 once it has them.
    requests[0].spec_token_ids = [11, 12, 13, 14, 15]
    scheduler.schedule()
    assert scheduler.cur_num_spec_tokens == 5


# ---------------------------------------------------------------- the drafter


class _WalkStub:
    """_selector_walk_kernel[(rows,)](...) in Python, greedy (same indexing)."""

    def __init__(self):
        self.log = []

    def __getitem__(self, grid):
        def run(scores, cand, sample_pos, req_state, temperature, seeds, tokens,
                realized, num_steps, top_k, BLOCK_K, SAMPLE_PROBABILISTIC,
                USE_FP64, num_warps):
            sc, ca = scores.reshape(-1), cand.reshape(-1)
            rs, tok, real = req_state.reshape(-1), tokens.reshape(-1), \
                realized.reshape(-1)
            for row in range(grid[0]):
                st = int(rs[row * num_steps])
                if st < 0:
                    continue
                prev = 0
                for step in range(num_steps):
                    flat = row * num_steps + step
                    base = (flat * top_k + prev) * top_k
                    v = sc[base:base + top_k]
                    idx = int(torch.argmax(v))
                    real[flat * top_k:(flat + 1) * top_k] = v
                    tok[flat] = ca[flat * top_k + idx]
                    self.log.append((row, step, int(sample_pos[flat])))
                    prev = idx
        return run


def _drafter(max_width, widths, width, num_reqs, pad, split, seed=0):
    """A DFlash2Speculator (CPU, stand-in model) whose block is ``max_width``,
    drafting at ``width``. The inputs depend only on (width, rows, seed), so
    two drafters drafting the same width see the same step."""
    k = max_width
    s = object.__new__(sm.DFlash2Speculator)
    s.device = torch.device("cpu")
    s.dtype = torch.float32
    s.hidden_size = H
    s.max_num_reqs = MAX_REQS
    s.num_speculative_steps = k
    s.num_query_per_req = k + 1
    s.top_k = TOP_K
    s.candidate_sampler = sm.CandidateSampler(MAX_REQS, k, TOP_K, s.device)
    s.draft_tokens = torch.zeros(MAX_REQS, k, dtype=torch.int64)
    s.draft_logits = None
    s.use_fp64_gumbel = False
    s.enable_adaptive_verification = False
    s.split_tail = False
    # The width state __init__ sets up.
    s.max_width = s.width = k
    s.widths = (k,)
    s._full_draft_tokens = s.draft_tokens
    s._draft_token_views = {k: s.draft_tokens}
    s.sample_col = torch.arange(k, dtype=torch.int32).repeat(MAX_REQS)
    s._sample_cols = {k: s.sample_col}
    s._cg_managers = {}
    s._tail_layouts, s._tail_row_ids_by_width, s._anchor_indices_by_width = {}, {}, {}
    s.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(MAX_REQS * (k + 1), dtype=torch.int32))
    if widths:
        s.enable_draft_widths(widths)
    if split:
        s.enable_split_tail()
    s.set_width(width)

    w, nq, rows = width, width + 1, num_reqs + pad
    g = torch.Generator().manual_seed(seed * 1000 + w * 10 + rows)
    input_ids = torch.randint(0, VOCAB, (MAX_REQS * nq,), generator=g,
                              dtype=torch.int32)
    s.input_buffers.input_ids[: input_ids.numel()] = input_ids
    slots = torch.randperm(MAX_REQS, generator=g)[:rows].int()
    slots[num_reqs:] = -1
    s.sample_idx_mapping = torch.full((MAX_REQS * k,), -1, dtype=torch.int32)
    s.sample_idx_mapping[:rows * w] = slots.repeat_interleave(w)
    s.sample_indices = torch.zeros(MAX_REQS * k, dtype=torch.int64)
    s.sample_indices[:rows * w] = (torch.arange(rows * w) // w * nq
                                   + torch.arange(rows * w) % w + 1)
    s.sample_pos = torch.zeros(MAX_REQS * k, dtype=torch.int64)
    s.sample_pos[:MAX_REQS * w] = torch.randint(1, 5000, (MAX_REQS * w,), generator=g)
    s.temperature = torch.rand(MAX_REQS, generator=g)
    s.seeds = torch.randint(0, 1 << 30, (MAX_REQS,), generator=g)
    hidden = torch.randn(rows * nq, H, generator=g)
    proj = torch.randn(H, TOP_K, generator=g)

    def compute_candidates(x):
        unary = x @ proj
        ids = (x.abs().sum(-1, keepdim=True) * 1000).long() % VOCAB
        return (ids + torch.arange(TOP_K)) % VOCAB, unary

    def selector(cand, unary, hid, anchor):
        return (unary[..., None, :] + 0.5 * unary[..., :, None]
                + 0.01 * anchor.float()[:, None, None, None]
                + 0.1 * hid.sum(-1)[..., None, None])

    s.model = SimpleNamespace(compute_candidates=compute_candidates,
                              model=SimpleNamespace(candidate_selector=selector))
    s._run_model = lambda *a, **kw: hidden
    return s, rows


def _draft(monkeypatch, s, rows, num_reqs, split):
    stub = _WalkStub()
    monkeypatch.setattr(sm, "_selector_walk_kernel", stub)
    s._generate_draft(rows, rows * s.num_query_per_req, None, None, None)
    if split:
        s.run_tail_local(rows)
    return s.draft_tokens[:num_reqs].clone(), stub.log


def test_width_views_share_the_full_buffer():
    s, _ = _drafter(5, (3, 4, 5), 3, 2, 0, False)
    for w in (3, 4, 5):
        v = s._draft_token_views[w]
        assert v.shape == (MAX_REQS, w) and v.is_contiguous()
        assert v.data_ptr() == s._full_draft_tokens.data_ptr()
        assert torch.equal(s._sample_cols[w][: 2 * w], torch.arange(w).repeat(2).int())
    s.set_width(4)
    assert (s.num_speculative_steps, s.num_query_per_req) == (4, 5)
    assert s.candidate_sampler.num_steps == 4 and s.draft_tokens.shape == (8, 4)
    s.set_width(7)  # not a captured width: the full block
    assert s.width == 5


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("num_reqs,pad", [(1, 0), (3, 1), (8, 0)])
@pytest.mark.parametrize("split", [False, True])
def test_narrow_draft_equals_a_drafter_of_that_width(monkeypatch, width, num_reqs,
                                                      pad, split):
    """Drafting width w in a K=5 drafter is the step a K=w drafter runs: same
    tokens, same walk (so under load the drafter does what a fixed k=3 one
    does)."""
    if num_reqs + pad > MAX_REQS:
        pytest.skip("more rows than requests")
    wide, rows = _drafter(5, (3, 4, 5), width, num_reqs, pad, split)
    plain, _ = _drafter(width, (), width, num_reqs, pad, split)
    out_wide = _draft(monkeypatch, wide, rows, num_reqs, split)
    out_plain = _draft(monkeypatch, plain, rows, num_reqs, split)
    assert out_wide[0].shape == (num_reqs, width)
    assert torch.equal(out_wide[0], out_plain[0])
    assert out_wide[1] == out_plain[1] and out_plain[1]


@pytest.mark.parametrize("width", [3, 4, 5])
def test_split_equals_fused_at_every_width(monkeypatch, width):
    fused, rows = _drafter(5, (3, 4, 5), width, 3, 1, False)
    split, _ = _drafter(5, (3, 4, 5), width, 3, 1, True)
    assert torch.equal(_draft(monkeypatch, fused, rows, 3, False)[0],
                       _draft(monkeypatch, split, rows, 3, True)[0])


@pytest.mark.parametrize("width", [3, 4, 5])
def test_tail_stage_fills_only_the_drafted_columns(monkeypatch, width):
    """The tail stage reads the last stage's width-w payload with its own
    width-w layout and writes the first w columns of the full-width broadcast."""
    num_reqs = 3
    last, rows = _drafter(5, (3, 4, 5), width, num_reqs, 1, True)
    expect, _ = _draft(monkeypatch, last, rows, num_reqs, True)
    payload = last.tail_payload(rows).clone()

    gate = dt.DraftTailGate(requested=2, stage=2, reason="")
    tail = object.__new__(dt.DraftTailController)
    tail.device = torch.device("cpu")
    tail.vllm_config = SimpleNamespace(model_config=SimpleNamespace(
        use_fp64_gumbel=False))
    tail.gate, tail.widths = gate, (3, 4, 5)
    tail.layouts = {w: last.tail_layout_for(w) for w in (3, 4, 5)}
    tail.layout = tail.layouts[5]
    tail.rows_table = {w: {n: n for n in range(1, 9)} for w in (3, 4, 5)}
    tail.module = SimpleNamespace(
        top_k=TOP_K, compute_candidates=last.model.compute_candidates,
        select=last.model.model.candidate_selector)
    tail._out_tokens = torch.zeros(MAX_REQS, 5, dtype=torch.int64)
    tail._scores = torch.zeros(MAX_REQS * 5 * TOP_K)
    tail._workspaces = {}
    monkeypatch.setattr(dt, "tail_side_stream_workspaces",
                        lambda *a, **k: __import__("contextlib").nullcontext())
    monkeypatch.setattr(sm, "_selector_walk_kernel", _WalkStub())
    assert tail._rows(num_reqs, width) == num_reqs
    broadcast = torch.full((num_reqs, 5), -7, dtype=torch.int64)
    tail._run(payload, rows, num_reqs, broadcast, width)
    assert torch.equal(broadcast[:, :width], expect)
    assert (broadcast[:, width:] == -7).all()


def test_tail_rows_table_per_width_and_single_width():
    c = object.__new__(dt.DraftTailController)
    c.rows_table = {3: {1: 1, 2: 2, 3: 4}, 5: {1: 1, 2: 2, 3: 4}}
    assert c._rows(3, 3) == 4 and c._rows(3, 4) == 3
    c.rows_table = {1: 1, 2: 2, 3: 4}  # one width: request count -> rows
    assert c._rows(3, 5) == 4


# ------------------------------------------------- logs on a non-first rank


def _not_local_first_rank(monkeypatch):
    """A pipeline stage other than the first: *_once would print nothing."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "is_local_first_rank", lambda: False)
    monkeypatch.setattr(ps, "is_global_first_rank", lambda: False)


def test_drafter_banner_logs_on_a_non_first_rank(monkeypatch, caplog_vllm):
    _not_local_first_rank(monkeypatch)
    from vllm.logger import init_logger

    # The premise: a *_once line is dropped on this rank ...
    init_logger("vllm.test_draft_width").info_once("once-only line %s", "x")
    # ... while the drafter's banner still prints.
    sm.log_draft_width_banner((3, 4, 5))
    messages = [r.getMessage() for r in caplog_vllm.records]
    assert not any("once-only line" in m for m in messages)
    assert any(
        "GLM-5 load-following DFlash draft width active" in m and "(3, 4, 5)" in m
        for m in messages
    ), messages


def test_drafter_gate_warning_logs_on_a_non_first_rank(monkeypatch, caplog_vllm):
    _not_local_first_rank(monkeypatch)
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            adaptive_k_config=AdaptiveKConfig.from_dict(BY_LOAD_WIDTH, 5),
            draft_sample_method="probabilistic", enable_adaptive_verification=False),
        parallel_config=SimpleNamespace(data_parallel_size=1))
    assert sm.load_following_draft_widths(cfg) == ()
    assert not any("keeps its full block" in r.getMessage() for r in caplog_vllm.records)
    assert sm.load_following_draft_widths(cfg, log=True) == ()
    assert any("keeps its full block: needs greedy draft sampling" in r.getMessage()
               for r in caplog_vllm.records)


# ------------------------------------------- the drafter's block-length conv


def _conv_inputs(rows, hidden=64, groups=8, taps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, hidden, generator=g)
    delta = torch.randn(rows, taps, groups, generator=g)
    base = torch.randn(taps, hidden, generator=g)
    return x, delta, base


@pytest.mark.parametrize("width", [3, 4, 5])
@pytest.mark.parametrize("num_reqs", [1, 2, 8])
def test_dynamic_conv_equals_a_drafter_of_that_width(width, num_reqs):
    """A width-w step in a K=5 drafter convolves each request's 1+w block as a
    K=w drafter does -- the block length follows the step, not the config."""
    from vllm.model_executor.models import qwen3_dflash2 as m

    x, delta, base = _conv_inputs(num_reqs * (1 + width))
    k_w = m._grouped_conv(x, delta, base, 1 + width, 8, 8, 3)
    dyn = m._grouped_conv(x, delta, base, 6, 8, 8, 3,
                          block_size_tensor=torch.tensor([1 + width], dtype=torch.int32))
    assert torch.equal(dyn, k_w)


def test_fixed_block_length_mixes_requests_at_a_narrow_width():
    """The bug the dynamic block fixes: with the configured block (6) a width-3
    step's blocks of 4 are misaligned, so the conv reads across requests."""
    from vllm.model_executor.models import qwen3_dflash2 as m

    x, delta, base = _conv_inputs(2 * 4)
    assert not torch.equal(m._grouped_conv(x, delta, base, 6, 8, 8, 3),
                           m._grouped_conv(x, delta, base, 4, 8, 8, 3))


def test_conv_module_gets_a_block_tensor_only_with_width_mode():
    from vllm.model_executor.models import qwen3_dflash2 as m

    static = object.__new__(m.DFlashGroupedConv)
    torch.nn.Module.__init__(static)
    static.block_size_tensor = None
    assert static.block_size_tensor is None
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            adaptive_k_config=AdaptiveKConfig.from_dict(BY_LOAD_WIDTH, 5),
            draft_sample_method="greedy", enable_adaptive_verification=False),
        parallel_config=SimpleNamespace(data_parallel_size=1))
    assert m._load_following_width(cfg) is True
    cfg.speculative_config.adaptive_k_config = AdaptiveKConfig.from_dict(
        {"by_load": [5, 4, 3]}, 5)
    assert m._load_following_width(cfg) is False


def test_set_width_writes_the_block_length_into_the_drafter():
    s, _ = _drafter(5, (3, 4, 5), 5, 2, 0, False)
    tensors = [torch.tensor([6], dtype=torch.int32) for _ in range(3)]
    s.model = torch.nn.Module()
    for i, t in enumerate(tensors):
        sub = torch.nn.Module()
        sub.block_size_tensor = t
        s.model.add_module(f"conv{i}", sub)
    s._block_size_tensors = None
    s._model_block_size = None
    s.set_width(3)
    assert all(int(t) == 4 for t in tensors)
    s.set_width(4)
    assert all(int(t) == 5 for t in tensors)
    s.set_width(5)
    assert all(int(t) == 6 for t in tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("width", [3, 4, 5])
def test_dynamic_conv_kernel_matches_the_static_kernel(width):
    """GPU: the runtime-block Triton kernel gives the static kernel's output
    bit for bit at every width (both fp32-accumulate the same terms)."""
    from vllm.model_executor.models import qwen3_dflash2 as m

    num_reqs, hidden = 8, 1024
    x, delta, base = (t.cuda().to(torch.bfloat16)
                      for t in _conv_inputs(num_reqs * (1 + width), hidden, 64, 3))
    static = m.dflash2_grouped_conv_impl(x, delta, base, 1 + width, hidden // 64)
    dyn = m.dflash2_grouped_conv_dyn_impl(
        x, delta, base, torch.tensor([1 + width], dtype=torch.int32, device="cuda"),
        hidden // 64)
    assert torch.equal(dyn, static)


@pytest.fixture()
def gloo_tp1():
    """A one-process gloo tensor-parallel group (CPU), for modules whose
    layers ask for the TP group at construction."""
    import contextlib
    import os
    import tempfile

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=1, rank=0,
                                         distributed_init_method=f"file://{path}",
                                         local_rank=0, backend="gloo")
            initialize_model_parallel(1, 1, backend="gloo")
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


@pytest.mark.parametrize("dynamic", [False, True])
def test_conv_module_constructs_both_ways(gloo_tp1, dynamic):
    """The real DFlashGroupedConv.__init__ (not a stub), both modes: with the
    draft width on it owns a non-persistent block-length buffer, otherwise
    none. Forward (prepare + finish) at every width: the dynamic module with
    its block set to 1 + w equals a static module built for block 1 + w."""
    from vllm.model_executor.models import qwen3_dflash2 as m

    def build(block, dyn):
        conv = m.DFlashGroupedConv(hidden_size=64, taps=3, group_size=8,
                                   block_size=block, params_dtype=torch.float32,
                                   prefix="t", dynamic_block=dyn)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            conv.base_kernel.copy_(torch.randn(conv.base_kernel.shape, generator=g))
            w = conv.kernel_projection.weight
            w.copy_(torch.randn(w.shape, generator=g) * 0.1)
        return conv

    conv = build(6, dynamic)
    if dynamic:
        assert int(conv.block_size_tensor) == 6
        assert "block_size_tensor" not in conv.state_dict()
        assert "block_size_tensor" in dict(conv.named_buffers())
    else:
        assert conv.block_size_tensor is None
        assert "block_size_tensor" not in dict(conv.named_buffers())
    for width in (3, 4, 5):
        x = torch.randn(3 * (1 + width), 64, generator=torch.Generator().manual_seed(width))
        if dynamic:
            conv.block_size_tensor.fill_(1 + width)
            ref = build(1 + width, False)
        else:
            ref = build(6, False)
        with torch.no_grad():
            h, c = conv.prepare(x)
            out = conv.finish(h, c)
            h_ref, c_ref = ref.prepare(x)
            out_ref = ref.finish(h_ref, c_ref)
        assert torch.equal(out, out_ref), width


@pytest.mark.parametrize("width", [3, 5, 6, 7])
@pytest.mark.parametrize("split", [False, True])
def test_k7_narrow_draft_equals_a_drafter_of_that_width(monkeypatch, width, split):
    """K_max 7: a width-w block in a K=7 drafter is a K=w drafter's step, fused and
    through the PP draft-tail payload (split)."""
    wide, rows = _drafter(7, (3, 4, 5, 6, 7), width, 3, 1, split)
    plain, _ = _drafter(width, (), width, 3, 1, split)
    out_wide = _draft(monkeypatch, wide, rows, 3, split)
    out_plain = _draft(monkeypatch, plain, rows, 3, split)
    assert torch.equal(out_wide[0], out_plain[0]) and out_wide[1] == out_plain[1]


@pytest.mark.parametrize("width", [3, 6, 7])
def test_k7_tail_stage_fills_only_the_drafted_columns(monkeypatch, width):
    num_reqs = 3
    last, rows = _drafter(7, (3, 4, 5, 6, 7), width, num_reqs, 1, True)
    expect, _ = _draft(monkeypatch, last, rows, num_reqs, True)
    payload = last.tail_payload(rows).clone()
    tail = object.__new__(dt.DraftTailController)
    tail.device = torch.device("cpu")
    tail.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_fp64_gumbel=False))
    tail.widths = (3, 4, 5, 6, 7)
    tail.layouts = {w: last.tail_layout_for(w) for w in tail.widths}
    tail.layout = tail.layouts[7]
    tail.rows_table = {w: {n: n for n in range(1, 9)} for w in tail.widths}
    tail.module = SimpleNamespace(top_k=TOP_K, compute_candidates=last.model.compute_candidates,
                                  select=last.model.model.candidate_selector)
    tail._out_tokens = torch.zeros(MAX_REQS, 7, dtype=torch.int64)
    tail._scores = torch.zeros(MAX_REQS * 7 * TOP_K)
    tail._workspaces = {}
    monkeypatch.setattr(dt, "tail_side_stream_workspaces",
                        lambda *a, **k: __import__("contextlib").nullcontext())
    monkeypatch.setattr(sm, "_selector_walk_kernel", _WalkStub())
    broadcast = torch.full((num_reqs, 7), -7, dtype=torch.int64)
    tail._run(payload, rows, num_reqs, broadcast, width)
    assert torch.equal(broadcast[:, :width], expect)
    assert (broadcast[:, width:] == -7).all()
