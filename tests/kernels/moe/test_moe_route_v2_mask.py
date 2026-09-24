# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The padding mask applied inside the sm_80 fused router v2
(VLLM_GLM5_MOE_ROUTE_V2_MASK, vllm/ampere_decode/moe_route.py).

CPU part (runs under TRITON_INTERPRET=1): the routing rows and the alignment
of `_moe_route_kernel` (`_route_align_rows`, one CTA per row, the last one to
arrive building the alignment) are driven through a thin wrapper kernel on
given logits. The tensor-core GEMV half needs inline PTX and is not part of
the change, so it is left out; the routing and alignment code is the code the
server runs. Checked at every padded size 9..32:

  * the alignment equals a fresh deterministic alignment of the masked ids;
  * the ids of real rows and every weight equal the unmasked launch's;
  * padding rows' ids are -1;
  * with no padding (the mask absent, or present and all real) every output
    is byte-identical to the unmasked launch.

GPU part: the whole one-launch op with and without the mask, eager and under
graph replay, against the same references.
"""

import os

import pytest
import torch

E = 288
TOPK = 8
HIDDEN = 4096

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _align(ids, bs):
    """A fresh deterministic alignment of `ids` (the reference)."""
    from vllm.model_executor.layers.fused_moe import moe_align_block_size as mab

    n = ids.numel()
    max_pad = n + E * (bs - 1)
    if n < E:
        max_pad = min(n * bs, max_pad)
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=ids.device)
    expert_ids = torch.empty((max_pad + bs - 1) // bs, dtype=torch.int32,
                             device=ids.device)
    ntp = torch.empty(1, dtype=torch.int32, device=ids.device)
    mab.deterministic_moe_align_block_size(
        ids, E, bs, sorted_ids, expert_ids, ntp, None, None
    )
    return sorted_ids, expert_ids, ntp


def _same_align(a, b, bs):
    """Equal over the whole used range, and every unused slot is a sentinel."""
    ntp = int(a[2])
    if ntp != int(b[2]):
        return False
    nb = ntp // bs
    return torch.equal(a[0][:ntp], b[0][:ntp]) and torch.equal(a[1][:nb], b[1][:nb])


def _tail_is_sentinel(al, numel, bs):
    ntp = int(al[2])
    return bool((al[0][ntp:] == numel).all()) and bool((al[1][ntp // bs:] == -1).all())


# ---------------------------------------------------------------- CPU part
def _rows_launch(logits, bias, bs, is_padding=None):
    """`_moe_route_kernel`'s routing half on given logits: M CTAs, one row
    each, exactly the constexprs `moe_route` passes."""
    import triton

    from vllm.ampere_decode import moe_route as mrt

    M = logits.shape[0]
    numel = M * TOPK
    mnp, nblk = mrt.align_sizes(numel, E, bs)
    (_, be, bk, bn, _, _, _, ea, eb, bit, _) = mrt._config(M, TOPK, E, bs, mnp, nblk)
    dev = logits.device
    tw = torch.empty(M, TOPK, dtype=torch.float32, device=dev)
    ti = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
    s = torch.empty(mnp, dtype=torch.int32, device=dev)
    e = torch.empty(nblk, dtype=torch.int32, device=dev)
    n = torch.empty(1, dtype=torch.int32, device=dev)
    col = torch.zeros(2 * E, dtype=torch.int32, device=dev)
    arrive = torch.zeros(2, dtype=torch.int32, device=dev)
    pad = None if is_padding is None else is_padding.view(torch.uint8)
    _rows_kernel[(M,)](
        logits, bias, tw, ti, s, e, n, col, arrive, logits.stride(0), 2.5,
        M=M, E=E, BS=bs, MNP=mnp, NBLK=nblk, TOPK=TOPK, BLOCK_K=bk,
        BLOCK_E=be, BLOCK_N=bn,
        FILL_P=triton.next_power_of_2(triton.cdiv(max(mnp, nblk), M)),
        EA=ea, EB=eb, BITONIC=bit, RENORM=True,
        BLOCK_J=triton.next_power_of_2(triton.cdiv(M, bs)),
        pad_ptr=pad, HAS_PAD=pad is not None,
    )
    # the scratch is left at zero by every launch (graph-replay invariant)
    assert int(col.abs().sum()) == 0 and int(arrive.abs().sum()) == 0
    return tw, ti, (s, e, n)


try:
    import triton
    import triton.language as tl

    from vllm.ampere_decode.moe_route import _route_align_rows

    @triton.jit
    def _rows_kernel(logits_ptr, bias_ptr, topk_w_ptr, topk_ids_ptr,
                     sorted_ids_ptr, expert_ids_ptr, ntpp_ptr, col_ptr,
                     arrive_ptr, stride_lm, routed_scaling_factor,
                     M: tl.constexpr, E: tl.constexpr, BS: tl.constexpr,
                     MNP: tl.constexpr, NBLK: tl.constexpr,
                     TOPK: tl.constexpr, BLOCK_K: tl.constexpr,
                     BLOCK_E: tl.constexpr, BLOCK_N: tl.constexpr,
                     FILL_P: tl.constexpr, EA: tl.constexpr, EB: tl.constexpr,
                     BITONIC: tl.constexpr, RENORM: tl.constexpr,
                     BLOCK_J: tl.constexpr, pad_ptr=None,
                     HAS_PAD: tl.constexpr = False):
        _route_align_rows(tl.program_id(0), logits_ptr, bias_ptr, topk_w_ptr,
                          topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr,
                          ntpp_ptr, col_ptr, arrive_ptr, stride_lm, 1,
                          routed_scaling_factor, M, E, BS, MNP, NBLK, TOPK,
                          BLOCK_K, 1, BLOCK_E, BLOCK_N, FILL_P, EA, EB,
                          BITONIC, RENORM, M, BLOCK_J, pad_ptr, HAS_PAD)
except Exception:  # pragma: no cover - triton missing
    _rows_kernel = None

cpu = pytest.mark.skipif(not INTERPRET or _rows_kernel is None,
                         reason="needs TRITON_INTERPRET=1")


@pytest.fixture(autouse=True)
def _interpreter_libdevice(monkeypatch):
    """The interpreter does not implement `libdevice`; give the two functions
    the routing rows use numpy equivalents. Every CPU check compares launches
    of the same interpreted code, so the tanh's last ulp does not matter."""
    if not INTERPRET:
        return
    import numpy as np
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle

    from vllm.ampere_decode import moe_route as mrt

    def _wrap(x, data):
        return tl.core.tensor(TensorHandle(data, x.handle.dtype.scalar), x.type)

    class _LibDevice:
        @staticmethod
        def tanh(x, _semantic=None):
            return _wrap(x, np.tanh(x.handle.data))

        @staticmethod
        def popc(x, _semantic=None):
            d = x.handle.data
            return _wrap(x, np.bitwise_count(d.view(np.uint32)).astype(d.dtype))

    monkeypatch.setattr(mrt, "libdevice", _LibDevice)


def _cpu_inputs(m, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(m, E, generator=g) * 3.0
    bias = torch.randn(E, generator=g) * 0.1
    return logits, bias


def _n_real(m):
    return max(1, m - 1 - (m % 5))      # 1..(m-1) padding rows, varied


@cpu
@pytest.mark.parametrize("m", list(range(9, 33)))
def test_cpu_masked_alignment_equals_fresh_alignment(m):
    from vllm.ampere_decode import marlin_block_size_m

    bs = marlin_block_size_m(m, TOPK, E)
    logits, bias = _cpu_inputs(m, 1000 + m)
    n_real = _n_real(m)
    is_padding = torch.arange(m) >= n_real

    tw0, ti0, al0 = _rows_launch(logits, bias, bs)
    tw1, ti1, al1 = _rows_launch(logits, bias, bs, is_padding)

    # real rows unchanged, padding rows routed to no expert, weights as before
    assert torch.equal(ti1[:n_real], ti0[:n_real])
    assert bool((ti1[n_real:] == -1).all())
    assert torch.equal(tw1, tw0)
    # the unmasked launch aligns the unmasked ids (sanity of the reference)
    assert _same_align(al0, _align(ti0.clone(), bs), bs)
    # the masked launch's alignment is a fresh one of the masked ids
    masked = ti0.clone()
    masked.masked_fill_(is_padding[:, None], -1)
    assert torch.equal(ti1, masked)
    ref = _align(masked, bs)
    assert _same_align(al1, ref, bs)
    assert _tail_is_sentinel(al1, m * TOPK, bs)
    # no used slot refers to a padding row
    used = al1[0][: int(al1[2])]
    assert bool((used[used < m * TOPK] < n_real * TOPK).all())


@cpu
@pytest.mark.parametrize("m", [9, 12, 16, 24, 32])
def test_cpu_no_padding_is_byte_identical(m):
    from vllm.ampere_decode import marlin_block_size_m

    bs = marlin_block_size_m(m, TOPK, E)
    logits, bias = _cpu_inputs(m, 2000 + m)
    ref = _rows_launch(logits, bias, bs)
    got = _rows_launch(logits, bias, bs, torch.zeros(m, dtype=torch.bool))
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    for a, b in zip(got[2], ref[2]):
        assert torch.equal(a, b)          # whole buffers, sentinels included


@cpu
def test_cpu_all_padding_but_one_row():
    from vllm.ampere_decode import marlin_block_size_m

    m = 16
    bs = marlin_block_size_m(m, TOPK, E)
    logits, bias = _cpu_inputs(m, 7)
    is_padding = torch.arange(m) >= 1
    _, ti0, _ = _rows_launch(logits, bias, bs)
    _, ti1, al1 = _rows_launch(logits, bias, bs, is_padding)
    masked = ti0.clone()
    masked[1:] = -1
    assert torch.equal(ti1, masked)
    assert _same_align(al1, _align(masked, bs), bs)
    assert int(al1[2]) == TOPK * bs       # 8 experts, one block each


# ------------------------------------------------ host handoff (CPU, no kernel)
def test_pad_supported_range():
    from vllm.ampere_decode.moe_route import pad_supported

    assert not pad_supported(1)
    assert all(pad_supported(m) for m in range(2, 65))
    assert not pad_supported(65)


def test_moe_route_rejects_padding_outside_the_one_launch_path():
    from vllm.ampere_decode.moe_route import moe_route

    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    w = torch.zeros(E, HIDDEN, dtype=torch.bfloat16)
    b = torch.zeros(E)
    with pytest.raises(ValueError):
        moe_route(x, w, b, is_padding=torch.zeros(1, dtype=torch.bool))


# ---------------------------------------------------------------- GPU part
def _has_sm80():
    if INTERPRET or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (8, 0)


gpu = pytest.mark.skipif(not _has_sm80(), reason="needs an sm_80 GPU")


def _gpu_inputs(m, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).cuda()
    w = (torch.randn(E, HIDDEN, generator=g) * 0.02).to(torch.bfloat16).cuda()
    b = (torch.randn(E, generator=g) * 0.1).cuda()
    return x, w, b


@gpu
@pytest.mark.parametrize("m", list(range(9, 33)))
def test_gpu_masked_route_v2(m):
    from vllm.ampere_decode import marlin_block_size_m
    from vllm.ampere_decode.moe_route import moe_route

    bs = marlin_block_size_m(m, TOPK, E)
    x, w, b = _gpu_inputs(m, 300 + m)
    n_real = _n_real(m)
    is_padding = (torch.arange(m) >= n_real).cuda()
    lg0, tw0, ti0, *al0 = moe_route(x, w, b, topk=TOPK, block_size=bs)
    lg1, tw1, ti1, *al1 = moe_route(x, w, b, topk=TOPK, block_size=bs,
                                    is_padding=is_padding)
    torch.cuda.synchronize()
    assert torch.equal(lg1, lg0) and torch.equal(tw1, tw0)
    assert torch.equal(ti1[:n_real], ti0[:n_real])
    assert bool((ti1[n_real:] == -1).all())
    masked = ti0.clone()
    masked.masked_fill_(is_padding[:, None], -1)
    assert _same_align(al1, _align(masked, bs), bs)
    assert _tail_is_sentinel(al1, m * TOPK, bs)
    # all-real mask: byte-identical to the unmasked op
    out = moe_route(x, w, b, topk=TOPK, block_size=bs,
                    is_padding=torch.zeros(m, dtype=torch.bool, device="cuda"))
    for p, q in zip(out, (lg0, tw0, ti0, *al0)):
        assert torch.equal(p, q)


@gpu
@pytest.mark.parametrize("m", [12, 16, 24, 32])
def test_gpu_graph_replay_follows_the_device_mask(m):
    from vllm.ampere_decode import marlin_block_size_m
    from vllm.ampere_decode.moe_route import align_sizes, moe_route, warmup

    bs = marlin_block_size_m(m, TOPK, E)
    warmup((m,), (bs,), topk=TOPK, num_experts=E, hidden=HIDDEN,
           padded_ms=(m,))
    x, w, b = _gpu_inputs(m, 500 + m)
    mnp, nblk = align_sizes(m * TOPK, E, bs)
    out = (torch.empty(m, E, device="cuda"), torch.empty(m, TOPK, device="cuda"),
           torch.empty(m, TOPK, dtype=torch.int32, device="cuda"),
           torch.empty(mnp, dtype=torch.int32, device="cuda"),
           torch.empty(nblk, dtype=torch.int32, device="cuda"),
           torch.empty(1, dtype=torch.int32, device="cuda"))
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    moe_route(x, w, b, topk=TOPK, block_size=bs, out=out, is_padding=pad)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_route(x, w, b, topk=TOPK, block_size=bs, out=out, is_padding=pad)
    ref = [t.clone() for t in moe_route(x, w, b, topk=TOPK, block_size=bs)]
    for n_real in (m, m - 1, max(1, m // 2), 1):
        pad.copy_(torch.arange(m, device="cuda") >= n_real)
        for t in out:
            t.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        masked = ref[2].clone()
        masked.masked_fill_(pad[:, None], -1)
        assert torch.equal(out[2], masked)
        assert torch.equal(out[1], ref[1])
        al = _align(masked, bs)
        assert _same_align(out[3:], al, bs)
        assert _tail_is_sentinel(out[3:], m * TOPK, bs)
        if n_real == m:
            for p, q in zip(out, ref):
                assert torch.equal(p, q)
