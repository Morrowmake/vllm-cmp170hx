# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused DSpark context-KV prologue (``ops/sm80_draft_prologue.py``) on
one sm_80 card.

- The sum: ``torch_cuda_mean_sum`` times ``1 / 512`` is ``mean(-1)`` of the
  squares on CUDA, bitwise, and the CPU half's NumPy model, at the row
  counts of ``ROWS`` (both sides of every change of torch's block shape;
  all three lane counts).
- RMSNorm: ``rms_norm_row`` is ``sm80_attn_core.rms_norm`` on CUDA, bitwise,
  at the same row counts.
- Cache bytes: after :func:`store_context_kv` the whole cache equals the
  torch path's (``rms_norm``, ``window_kv``, ``store_rows``) at the same row
  counts, with invalid slots, the strided wkv view of a ``[T, q_lora + 512]``
  projection, and more slots than rows or fewer.
- CUDA graph replay on new inputs equals the torch path.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_draft_prologue_gpu.py
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_sm80.models.deepseek_v41.ops import sm80_draft_prologue as dp

from .test_sm80_draft_prologue import _sum_probe, cuda_mean_sum_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8,
    reason="needs an sm_80 card",
)

# Both sides of every change of torch's block shape for ``mean(-1)`` (block
# height 1, 2, 4, 8, 16: lanes 128, 128, 128, 64, 32), a decode size and the
# prefill sizes.
ROWS = (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 40, 64, 2048)
WIDTH = 512
Q_LORA = 1024
EPS = 1e-6
ENTRIES = 64
OPTIONS = dict(enable_fp_fusion=False, enable_reflect_ftz=False)


@triton.jit
def _rms_probe(x_ptr, w_ptr, out_ptr, eps, D: tl.constexpr, BW: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, D)
    x = tl.load(x_ptr + t * D + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    tl.store(out_ptr + t * D + d, dp.rms_norm_row(x, w, eps, D, BW))


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def projection(rows, seed):
    """A ``[rows, Q_LORA + 512]`` BF16 projection whose wkv rows span six
    decades element by element and four row by row."""
    g = _gen(seed)
    x = torch.randn(rows, Q_LORA + WIDTH, generator=g, device="cuda")
    x = x * 10.0 ** torch.randint(-3, 3, x.shape, generator=g, device="cuda").float()
    x = x * 10.0 ** torch.randint(-2, 2, (rows, 1), generator=g, device="cuda").float()
    return x.to(torch.bfloat16)


def norm_weight(seed):
    g = _gen(seed)
    return (torch.randn(WIDTH, generator=g, device="cuda") * 0.5 + 1.0).to(torch.bfloat16)


def freqs():
    return core.rope_freqs(64, 65536, 10000.0, 16.0, 32, 1, device="cuda")


def calls(rows):
    """Independent calls of ``rows`` rows: enough rows in all."""
    return max(1, 256 // rows)


@pytest.mark.parametrize("rows", ROWS)
def test_sum_is_torch_mean(rows):
    lanes = dp.mean_block_width(rows, WIDTH)
    for c in range(calls(rows)):
        kv = projection(rows, 1000 * rows + c)[:, Q_LORA:]
        sq = kv.float().square()
        want = sq.mean(-1)
        got = torch.empty(rows, dtype=torch.float32, device="cuda")
        _sum_probe[(rows,)](sq.contiguous(), got, WIDTH, lanes)
        got = got * (1.0 / WIDTH)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (rows, c)
        model = cuda_mean_sum_model(sq.cpu().numpy(), lanes) * np.float32(1.0 / WIDTH)
        assert np.array_equal(model.view(np.int32), want.cpu().numpy().view(np.int32))


@pytest.mark.parametrize("rows", ROWS)
def test_rms_norm_row_is_torch_rms_norm(rows):
    lanes = dp.mean_block_width(rows, WIDTH)
    for c in range(calls(rows)):
        kv = projection(rows, 2000 * rows + c)[:, Q_LORA:]
        w = norm_weight(c)
        want = core.rms_norm(kv, w, EPS)
        got = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device="cuda")
        _rms_probe[(rows,)](kv.contiguous(), w, got, EPS, WIDTH, lanes, **OPTIONS)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (rows, c)


def context_case(rows, n_slots, seed):
    g = _gen(seed)
    qr_kv = projection(rows, seed)
    positions = torch.randint(0, 1 << 20, (rows,), generator=g, device="cuda")
    positions[: min(rows, 8)] = torch.arange(min(rows, 8), device="cuda")
    blocks = 2 + triton.cdiv(n_slots, ENTRIES)
    cache = torch.randn(blocks, ENTRIES, WIDTH, generator=g, device="cuda").to(torch.bfloat16)
    slots = ENTRIES + torch.randperm((blocks - 1) * ENTRIES, generator=g, device="cuda")[:n_slots]
    slots[::3] = -1
    return qr_kv, positions, cache, slots


def torch_path(kv, weight, positions, rope, cache, slots):
    n = min(slots.shape[0], kv.shape[0])
    rows = core.window_kv(core.rms_norm(kv[:n], weight, EPS), positions[:n], rope)
    core.store_rows(cache, slots[:n], rows)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("extra", ((0, 0), (0, 3), (3, 0)))
def test_cache_bytes_are_the_torch_paths(rows, extra):
    rope = freqs()
    for c in range(min(calls(rows), 8)):
        qr_kv, positions, cache, slots = context_case(
            rows + extra[0], rows + extra[1], 3000 * rows + c)
        kv = qr_kv[:, Q_LORA:]
        w = norm_weight(c)
        want = cache.clone()
        torch_path(kv, w, positions, rope, want, slots)
        dp.store_context_kv(kv, w, EPS, positions, rope, cache, slots)
        assert torch.equal(cache.view(torch.int16), want.view(torch.int16)), (rows, extra, c)


def test_graph_replay_is_the_torch_path():
    rope = freqs()
    rows = 8
    qr_kv, positions, cache, slots = context_case(rows, rows, 7)
    w = norm_weight(7)
    kv = qr_kv[:, Q_LORA:]
    dp.store_context_kv(kv, w, EPS, positions, rope, cache, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dp.store_context_kv(kv, w, EPS, positions, rope, cache, slots)
    for seed in (11, 12):
        new_qr_kv, new_positions, new_cache, new_slots = context_case(rows, rows, seed)
        qr_kv.copy_(new_qr_kv)
        positions.copy_(new_positions)
        cache.copy_(new_cache)
        slots.copy_(new_slots)
        want = cache.clone()
        torch_path(kv, w, positions, rope, want, slots)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cache.view(torch.int16), want.view(torch.int16)), seed
