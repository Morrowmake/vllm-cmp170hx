# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 attention core (``ops/sm80_attn_fused.py``)
on CPU: the switches and their defaults, the configuration gate and the
per-call rule of :class:`DeepseekV41Sm80Attention` (which calls take the
fused core), and that the layer functions keep the unfused path by default.
The kernels themselves (FP64 bounds, CUDA graphs, determinism, the layer
wiring) are ``test_sm80_attn_fused_gpu.py``.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_attn_fused.py
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.models.deepseek_v41 import sm80_attention as attn
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_fused as fused

VALIDATED = dict(
    n_local_heads=16, head_dim=512, rope_dim=64, n_local_groups=2, window_size=128,
    index_topk=512, compress_ratio=4, kv_dtype=torch.bfloat16, capability=(8, 0),
)


def test_switches_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE is False
    monkeypatch.delenv("VLLM_DEEPSEEK_V41_ATTN_PIPELINE", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_ATTN_PIPELINE is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", "1")
    assert envs.VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE is True


def test_row_bounds_are_the_validated_support_limits():
    """Both paths take calls of up to 48 padded rows, the range they were
    validated on; the bound is code, not a setting."""
    from vllm import envs
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_pipeline

    assert fused.MAX_ROWS == 48
    assert sm80_attn_pipeline.MAX_ROWS == 48
    for name in ("VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE_MAX_TOKENS",
                 "VLLM_DEEPSEEK_V41_ATTN_PIPELINE_MAX_TOKENS"):
        assert name not in envs.environment_variables
    budget = 1 << 62
    cfg = SimpleNamespace(compress_ratio=0, is_index_source=False)
    assert sm80_attn_pipeline.supports(cfg, 48, 1, 128, budget) is None
    assert "outside the supported bound" in sm80_attn_pipeline.supports(
        cfg, 49, 1, 128, budget)


def test_supports_only_the_validated_configuration():
    assert fused.supports(**VALIDATED) is None
    assert fused.supports(**dict(VALIDATED, compress_ratio=0, index_topk=0)) is None
    assert fused.supports(**dict(VALIDATED, capability=(8, 6))) is None
    for change, words in (
        (dict(capability=(9, 0)), "capability"),
        (dict(capability=None), "capability"),
        (dict(n_local_heads=32), "local heads"),
        (dict(head_dim=576), "local heads"),
        (dict(rope_dim=32), "local heads"),
        (dict(n_local_groups=4), "local heads"),
        (dict(window_size=256), "local heads"),
        (dict(index_topk=1024), "top-k"),
        (dict(kv_dtype=torch.float8_e4m3fn), "BF16"),
    ):
        reason = fused.supports(**dict(VALIDATED, **change))
        assert reason is not None and words in reason, (change, reason)


def _layer(draft=False, on=True, max_rows=48):
    return SimpleNamespace(
        _sm80_fused_attn=on,
        _sm80_fused_max_rows=max_rows,
        _sm80_is_draft_layer=draft,
        _sm80_max_decode_query_len=6,
    )


@pytest.mark.parametrize("prefill", [False, True])
def test_which_calls_take_the_fused_core(monkeypatch, prefill):
    seen = []

    def step(max_query_len):
        seen.append(max_query_len)
        return prefill

    monkeypatch.setattr(attn, "prefill_step", step)
    call = attn.DeepseekV41Sm80Attention._sm80_fused_attn_call
    assert call(_layer(on=False), 1) is False
    assert call(_layer(), 49) is False
    assert call(_layer(draft=True), 49) is False
    assert call(_layer(max_rows=16), 17) is False
    # a draft layer's calls are draft blocks, whatever the step
    assert call(_layer(draft=True), 40) is True
    # a target layer: decode and verify steps only
    assert call(_layer(), 48) is (not prefill)
    assert seen and set(seen) == {6}


def _gate_layer(heads=16):
    return SimpleNamespace(
        n_local_heads=heads, head_dim=512, rope_head_dim=64, n_local_groups=2,
        window_size=128, compress_ratio=4, kv_cache_torch_dtype=torch.bfloat16,
        sm80_cfg=SimpleNamespace(index_topk=512), layer_id=41,
    )


def _vllm_config(k=5, layers=40):
    return SimpleNamespace(
        num_speculative_tokens=k,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=layers)
        ),
    )


def test_the_layer_gate(monkeypatch):
    gate = attn.DeepseekV41Sm80Attention._sm80_fused_attn_gate
    monkeypatch.setattr(attn.current_platform, "get_device_capability", lambda: (8, 0))
    monkeypatch.delenv("VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", raising=False)
    layer = _gate_layer()
    assert gate(layer, _vllm_config()) is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE", "1")
    assert gate(layer, _vllm_config()) is True
    assert layer._sm80_is_draft_layer is True  # layer 41 of a 40-layer target
    assert layer._sm80_max_decode_query_len == 6
    assert layer._sm80_fused_max_rows == 48
    target = SimpleNamespace(**dict(vars(_gate_layer()), layer_id=39))
    assert gate(target, _vllm_config()) is True and target._sm80_is_draft_layer is False
    # a configuration the kernels were not validated for stays unfused
    assert gate(_gate_layer(heads=32), _vllm_config()) is False
    monkeypatch.setattr(attn.current_platform, "get_device_capability", lambda: (9, 0))
    assert gate(_gate_layer(), _vllm_config()) is False


def test_the_layer_functions_default_to_the_unfused_path():
    for fn in (core.prepare_layer, core.attend_layer):
        assert inspect.signature(fn).parameters["fused"].default is False
        assert inspect.signature(fn).parameters["pipeline"].default is False
    assert inspect.signature(core.store_kv).parameters["window"].default is True
    assert inspect.signature(core.forward_layer).parameters.get("fused") is None


def test_unfused_layer_on_cpu_is_unchanged():
    """With ``fused`` off, prepare_layer + attend_layer are the reference
    composition (norms, wq_b, RoPE, window write, sparse attention, output
    qdq) bit for bit."""
    g = torch.Generator().manual_seed(3)
    T, H, D, Q = 5, 16, 512, 1280
    cfg = core.Sm80AttnConfig(
        n_local_heads=H, head_dim=D, rope_dim=64, n_local_groups=2, window_size=128,
        compress_ratio=0, is_kv_source=False, is_index_source=False,
        is_candidate_source=False, uses_candidates=False, index_n_heads=0,
        index_head_dim=0, index_topk=0, candidate_topk_blocks=0,
        candidate_block_size=0, tp_size=4, eps=1e-6,
    )
    w = (torch.randn(H * D, Q, generator=g) / 40).to(torch.bfloat16)
    weights = core.Sm80AttnWeights(
        q_norm=(1 + 0.1 * torch.randn(Q, generator=g)).to(torch.bfloat16),
        kv_norm=(1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16),
        attn_sink=torch.randn(H, generator=g),
        freqs=core.rope_freqs(64, 0, 10000.0, 1.0, 32, 1),
        wq_b=lambda x: torch.nn.functional.linear(x, w),
    )
    swa = (torch.randn(4, 64, D, generator=g) * 0.5).to(torch.bfloat16)
    pos = torch.tensor([3, 4, 9, 10, 11])
    slots = torch.tensor([64 + 3, 64 + 4, 128 + 9, 128 + 10, -1])
    idx = torch.full((T, 128), -1, dtype=torch.int32)
    idx[:, :3] = torch.tensor([[64, 65, 67], [65, 67, 68], [128, 137, 138],
                               [137, 138, 139], [128, 129, 130]], dtype=torch.int32)
    batch = core.Sm80AttnBatch(
        positions=pos, token_to_req=torch.tensor([0, 0, 1, 1, 1]),
        query_start_loc=torch.tensor([0, 2, 5]), swa_slots=slots, swa_indices=idx,
    )
    qr_kv = (torch.randn(T, Q + D, generator=g) * 2).to(torch.bfloat16)
    caches = core.Sm80AttnCaches(swa=swa.clone(), topk=torch.empty(T, 0, dtype=torch.int32))
    out = torch.empty(T, H, D, dtype=torch.bfloat16)
    q = core.prepare_layer(cfg, weights, caches, batch, qr_kv, None, None)
    core.attend_layer(cfg, weights, caches, batch, q, out)

    want_cache = swa.clone()
    qr, kv = qr_kv.split([Q, D], dim=-1)
    qr = core.rms_norm(qr, weights.q_norm, cfg.eps)
    kv = core.rms_norm(kv, weights.kv_norm, cfg.eps)
    want_q = core.rope(weights.wq_b(qr).view(T, H, D), pos, weights.freqs)
    core.store_rows(want_cache, slots, core.window_kv(kv, pos, weights.freqs))
    o = core.sparse_attention(
        want_q, want_cache, idx, None, None, None, batch.token_to_req,
        weights.attn_sink, cfg.softmax_scale,
    )
    want_out = core.output_qdq(o, pos, weights.freqs, 2)
    assert torch.equal(q.view(torch.int16), want_q.view(torch.int16))
    assert torch.equal(caches.swa.view(torch.int16), want_cache.view(torch.int16))
    assert torch.equal(out.view(torch.int16), want_out.view(torch.int16))


@pytest.mark.parametrize("draft,prefill", [(False, False), (False, True), (True, False), (True, True)])
def test_pipeline_call_uses_phase_and_static_eligibility(monkeypatch, draft, prefill):
    monkeypatch.setattr(attn, "prefill_step", lambda _: prefill)
    layer = SimpleNamespace(_sm80_attn_pipeline_enabled=True,
        _sm80_attn_pipeline_eligible=True, _sm80_pipeline_max_rows=48,
        _sm80_is_draft_layer=draft, _sm80_max_decode_query_len=6,
        max_model_len=262144, window_size=128, is_kv_source=False, is_index_source=False)
    x = SimpleNamespace(shape=(48, 1792), dtype=torch.bfloat16, is_cuda=True)
    caches = SimpleNamespace(swa=torch.empty(1, 64, 512, dtype=torch.bfloat16),
                             comp=None, idxk=None, ring=None)
    batch = SimpleNamespace(max_seq_len=1024, swa_indices=torch.empty(48, 128))
    call = attn.DeepseekV41Sm80Attention._sm80_pipeline_call
    assert call(layer, x, None, None, batch, caches) is (draft or not prefill)
    x.shape = (49, 1792)
    assert call(layer, x, None, None, batch, caches) is False
    x.shape = (1, 1792)
    layer._sm80_attn_pipeline_eligible = False
    assert call(layer, x, None, None, batch, caches) is False
    layer._sm80_attn_pipeline_eligible = True
    layer._sm80_attn_pipeline_enabled = False
    assert call(layer, x, None, None, batch, caches) is False
