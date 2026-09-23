# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in sm_80 MLA indexer / MoE tail folds
(vllm/ampere_decode/idx_glue.py, ``VLLM_GLM5_DECODE_IDX_GLUE``).

Three layers:

* host logic (gates, predicates, the moe_sum hand-off): plain CPU tests;
* every fold against the code it replaces, on CPU through the Triton
  interpreter (each check runs in a subprocess with ``TRITON_INTERPRET=1``);
* the same checks on the GPU plus comparisons with the CUDA ops the folds
  replace (``concat_and_cache_mla``, ``ops.moe_sum``, the cuBLAS fp32 sgemm,
  the compiled weight-scale leaf) -- skipped without CUDA.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_idx_glue.py
    CUDA_VISIBLE_DEVICES=3  pytest -q tests/kernels/test_ampere_idx_glue.py
"""

import os
import subprocess
import sys
import types

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")

N_HEAD = 32
HEAD_DIM = 128
HIDDEN = 4096
KPOOL = 4
TOPK_TOKENS = 2048
BUF_WIDTH = 2176  # topk + kpool - 1 rounded up to 128


# --------------------------------------------------------------------- gates
@pytest.fixture
def glue_on(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_DECODE_IDX_GLUE", "1")
    monkeypatch.delenv("VLLM_GLM5_DECODE_IDX_GLUE_PARTS", raising=False)
    monkeypatch.setattr("vllm.ampere_decode._SM80_CACHE", True)
    return monkeypatch


def test_off_by_default(monkeypatch):
    from vllm.ampere_decode import use_idx_glue

    monkeypatch.delenv("VLLM_GLM5_DECODE_IDX_GLUE", raising=False)
    monkeypatch.setattr("vllm.ampere_decode._SM80_CACHE", True)
    for part in ("weights", "glue", "fwht", "cache", "moesum"):
        assert use_idx_glue(part) is False


def test_off_path_does_not_import_the_fold_module():
    code = (
        "import sys, os\n"
        "os.environ.pop('VLLM_GLM5_DECODE_IDX_GLUE', None)\n"
        "import vllm.ampere_decode as ad\n"
        "ad._SM80_CACHE = True\n"
        "assert ad.use_idx_glue('glue') is False\n"
        "assert 'vllm.ampere_decode.idx_glue' not in sys.modules\n"
        "print('ok')\n"
    )
    import vllm

    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=root)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_all_parts_on_by_default(glue_on):
    from vllm.ampere_decode import use_idx_glue

    for part in ("weights", "glue", "fwht", "cache", "moesum"):
        assert use_idx_glue(part) is True
    assert use_idx_glue("nonsense") is False


def test_parts_selection(glue_on):
    from vllm.ampere_decode import use_idx_glue

    glue_on.setenv("VLLM_GLM5_DECODE_IDX_GLUE_PARTS", "glue, moesum,bogus")
    assert use_idx_glue("glue") and use_idx_glue("moesum")
    assert not use_idx_glue("weights")
    assert not use_idx_glue("fwht")
    assert not use_idx_glue("cache")


def test_not_sm80_means_off(glue_on):
    from vllm.ampere_decode import use_idx_glue

    glue_on.setattr("vllm.ampere_decode._SM80_CACHE", False)
    assert use_idx_glue("glue") is False


def test_production_weight_scale_is_a_power_of_two():
    from vllm.ampere_decode.idx_glue import is_pow2_float

    # softmax_scale * n_head**-0.5 with head_dim 128, n_head 32: 2**-6.
    wscale = HEAD_DIM**-0.5 * N_HEAD**-0.5
    assert is_pow2_float(wscale)
    # not exact in fp64 (0.015625000000000003), exact once rounded to fp32
    assert torch.tensor(wscale, dtype=torch.float32).item() == 2.0**-6
    assert not is_pow2_float(0.3)
    assert not is_pow2_float(0.0)
    assert not is_pow2_float(float("inf"))


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 24, 32])
def test_dual_gemm_supported_at_every_decode_size(m):
    from vllm.ampere_decode.idx_glue import thin_gemm_dual_supported

    x = torch.empty(m, HIDDEN, dtype=torch.bfloat16)
    w = torch.empty(HEAD_DIM + N_HEAD, HIDDEN, dtype=torch.bfloat16)
    assert thin_gemm_dual_supported(x, w, HEAD_DIM)


def test_dual_gemm_rejects_outside_the_thin_region():
    from vllm.ampere_decode.idx_glue import thin_gemm_dual_supported

    w = torch.empty(HEAD_DIM + N_HEAD, HIDDEN, dtype=torch.bfloat16)
    assert not thin_gemm_dual_supported(
        torch.empty(64, HIDDEN, dtype=torch.bfloat16), w, HEAD_DIM)
    assert not thin_gemm_dual_supported(
        torch.empty(4, HIDDEN, dtype=torch.float32), w, HEAD_DIM)
    # a split that does not fall on an N-tile boundary
    assert not thin_gemm_dual_supported(
        torch.empty(4, HIDDEN, dtype=torch.bfloat16), w, 100)


def test_moe_sum_deferral_protocol():
    from vllm.ampere_decode.idx_glue import _MoeSumDeferral

    d = _MoeSumDeferral()
    a, b = torch.zeros(1), torch.zeros(1)
    assert d.offer(a, b) is False  # not armed: the experts sum as usual
    d.arm()
    assert d.offer(a, b) is True
    assert d.offer(a, b) is False  # the arm is consumed by the first offer
    p = d.take()
    assert p[0] is a and p[1] is b
    assert d.take() is None
    d.arm()
    assert d.take() is None and d.offer(a, b) is False  # take disarms


def _indexer_meta(requires_padding=False, max_seq_len=5000, decode=True):
    dm = types.SimpleNamespace(requires_padding=requires_padding) if decode else None
    return types.SimpleNamespace(decode=dm, max_seq_len=max_seq_len)


def test_glue_tail_predicate(glue_on):
    from vllm.models.glm5next.nvidia.sparse_indexer import _use_glue_tail

    pos = torch.arange(4)
    ok = dict(has_decode=True, has_prefill=False, index_kpool=KPOOL,
              positions=pos, use_fp4_cache=False, topk_tokens=TOPK_TOKENS)
    glue_on.setattr(
        "vllm.models.glm5next.nvidia.sparse_indexer.current_platform",
        types.SimpleNamespace(is_cuda_alike=lambda: True),
    )
    assert _use_glue_tail(_indexer_meta(), **ok)
    for k, v in (("has_decode", False), ("has_prefill", True), ("index_kpool", 1),
                 ("positions", None), ("positions", torch.arange(0)),
                 ("use_fp4_cache", True)):
        assert not _use_glue_tail(_indexer_meta(), **dict(ok, **{k: v})), k
    # padded (non-uniform) decode layout and the short-context early return
    assert not _use_glue_tail(_indexer_meta(requires_padding=True), **ok)
    assert not _use_glue_tail(_indexer_meta(max_seq_len=TOPK_TOKENS), **ok)
    assert not _use_glue_tail(_indexer_meta(decode=False), **ok)
    glue_on.setenv("VLLM_GLM5_DECODE_IDX_GLUE_PARTS", "weights")
    assert not _use_glue_tail(_indexer_meta(), **ok)


def _mla_self(**kw):
    impl = types.SimpleNamespace(supports_idx_glue_kv_defer=True, dcp_world_size=1,
                                 _idx_glue_pending_kv=None)
    base = dict(hisparse_cache=None, use_pcp=False, impl=impl)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_mla_cache_defer_predicate(glue_on):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    f = MLAAttention._idx_glue_defer_kv
    kv = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
    meta = types.SimpleNamespace(num_prefills=0)
    assert f(_mla_self(), kv, meta, "auto") is True
    assert f(_mla_self(), kv, types.SimpleNamespace(num_prefills=1), "auto") is False
    assert f(_mla_self(), kv, None, "auto") is False
    assert f(_mla_self(), kv, meta, "fp8") is False
    assert f(_mla_self(), torch.zeros(0), meta, "auto") is False
    assert f(_mla_self(hisparse_cache=object()), kv, meta, "auto") is False
    assert f(_mla_self(use_pcp=True), kv, meta, "auto") is False
    no_backend = _mla_self()
    no_backend.impl.supports_idx_glue_kv_defer = False
    assert f(no_backend, kv, meta, "auto") is False
    dcp = _mla_self()
    dcp.impl.dcp_world_size = 2
    assert f(dcp, kv, meta, "auto") is False
    glue_on.setenv("VLLM_GLM5_DECODE_IDX_GLUE_PARTS", "glue")
    assert f(_mla_self(), kv, meta, "auto") is False


def test_mla_cache_defer_refuses_piecewise_capture(glue_on):
    from vllm.compilation import breakable_cudagraph as bcg
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.attention import mla_attention as ma

    cap = types.SimpleNamespace(_capturing=True)
    glue_on.setattr(bcg.BreakableCUDAGraphCapture, "current",
                    staticmethod(lambda: cap))
    kv = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
    meta = types.SimpleNamespace(num_prefills=0)
    for mode, want in ((CUDAGraphMode.PIECEWISE, False), (CUDAGraphMode.NONE, False),
                       (CUDAGraphMode.FULL, True)):
        glue_on.setattr(ma, "get_forward_context",
                        lambda m=mode: types.SimpleNamespace(
                            cudagraph_runtime_mode=m))
        assert ma.MLAAttention._idx_glue_defer_kv(_mla_self(), kv, meta,
                                                  "auto") is want, mode


def test_mla_flush_writes_and_clears():
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    calls = []
    s = _mla_self()
    s.impl.do_kv_cache_update = lambda *a: calls.append(a)
    MLAAttention._idx_glue_flush_kv(s)
    assert calls == []
    s.impl._idx_glue_pending_kv = (1, 2, 3, 4, 5, 6)
    MLAAttention._idx_glue_flush_kv(s)
    assert calls == [(1, 2, 3, 4, 5, 6)] and s.impl._idx_glue_pending_kv is None


def _runner_self(**kw):
    base = dict(_shared_experts=object(), routed_scaling_factor=1.0,
                routed_output_transform=None, _fused_output_is_reduced=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_moe_runner_arm_predicate(glue_on):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    f = MoERunner._idx_glue_arm_moe_sum
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    assert f(_runner_self(), x, None) is True
    assert f(_runner_self(_shared_experts=None), x, None) is False
    assert f(_runner_self(routed_scaling_factor=2.5), x, None) is False
    assert f(_runner_self(routed_output_transform=abs), x, None) is False
    assert f(_runner_self(_fused_output_is_reduced=True), x, None) is False
    assert f(_runner_self(), x, 4000) is False
    assert f(_runner_self(), torch.zeros(33, HIDDEN, dtype=torch.bfloat16),
             None) is False
    assert f(_runner_self(), x.float(), None) is False
    glue_on.setenv("VLLM_GLM5_DECODE_IDX_GLUE_PARTS", "glue")
    assert f(_runner_self(), x, None) is False
    can = MoERunner._idx_glue_can_fuse_moe_sum
    inp = torch.zeros(4, 8, HIDDEN, dtype=torch.bfloat16)
    assert can(inp, torch.zeros(4, HIDDEN, dtype=torch.bfloat16))
    assert not can(inp, None)
    assert not can(inp, torch.zeros(4, HIDDEN, dtype=torch.float32))
    assert not can(inp, torch.zeros(3, HIDDEN, dtype=torch.bfloat16))


# ------------------------------------------------- fold vs replaced code checks
# Each _check_* takes a device string and raises on any mismatch. On CPU they
# run under the Triton interpreter in a subprocess (the interpreter has to be
# selected before Triton is imported); on a GPU they run in-process.


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _to_bf16(x):
    """fp32 -> bf16 as a Triton kernel rounds it on this device.

    The GPU rounds to nearest even (cvt.rn), like torch. The Triton 3.7
    interpreter truncates, so interpreted kernels are compared with a
    truncating model; the GPU tests compare with torch and the CUDA ops.
    """
    if os.environ.get("TRITON_INTERPRET") == "1":
        return (x.float().contiguous().view(torch.int32) & -65536).view(
            torch.float32).to(torch.bfloat16)
    return x.to(torch.bfloat16)


def _check_expand(dev):
    from vllm.ampere_decode.idx_glue import expand_pools_into_buffer
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kp

    g = _gen(0)
    n_groups = TOPK_TOKENS // KPOOL
    for n, extra in ((4, 0), (16, 0), (12, 4), (1, 3)):
        t_rows = n + extra
        pos = torch.randint(0, 60000, (t_rows + 5,), generator=g, dtype=torch.int64)
        pos[0] = 5  # short: few pools, a partial tail
        pos[-1] = 3000
        pool = torch.full((n, n_groups), -1, dtype=torch.int32)
        for r in range(n):
            npools = int(pos[r] + 1) // KPOOL
            k = min(npools, n_groups)
            if k:
                pool[r, :k] = torch.randperm(npools, generator=g)[:k].to(torch.int32)
        garbage = torch.randint(-(2**31), 2**31 - 1, (t_rows + 3, BUF_WIDTH),
                                generator=g, dtype=torch.int32)
        pos_d, pool_d = pos.to(dev), pool.to(dev)

        ref = garbage.clone().to(dev)
        ref[:t_rows] = -1  # the fill
        seq = pos_d[:n].to(torch.int32) + 1
        out = kp.expand_pools_and_append_tail(pool_d.to(torch.int64), seq, KPOOL)
        ref[: out.shape[0], : out.shape[-1]] = out

        new = garbage.clone().to(dev)
        expand_pools_into_buffer(pool_d, pos_d[:n], new, t_rows, KPOOL)
        assert torch.equal(new.cpu(), ref.cpu()), (n, extra)

        # and against the pure-torch pair it fused in the first place
        tokens = kp.expand_pools_to_tokens(pool.to(torch.int64), pool >= 0,
                                           TOPK_TOKENS, KPOOL)
        seq_c = pos[:n].to(torch.int32) + 1
        want = kp.append_tail_to_topk(tokens, seq_c, seq_c // KPOOL, KPOOL)
        assert torch.equal(new.cpu()[:n, : want.shape[1]], want)
        assert (new.cpu()[:n, want.shape[1]:] == -1).all()


def _check_kpool_pos64(dev):
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kp

    g = _gen(1)
    num_blocks, page, pool = 8, 16, KPOOL
    for b, next_n, base in ((1, 4, (5,)), (2, 4, (7, 30))):
        kv = torch.randint(0, 255, (num_blocks, page, HEAD_DIM + 4), generator=g,
                           dtype=torch.uint8)
        tail = torch.randn(num_blocks, 2, pool, HEAD_DIM, generator=g).to(
            torch.bfloat16)
        key = torch.randn(b, next_n, HEAD_DIM, generator=g).to(torch.bfloat16)
        score = torch.randn(b, next_n, HEAD_DIM, generator=g).to(torch.bfloat16)
        ape = torch.randn(pool, HEAD_DIM, generator=g)
        pos = torch.stack([torch.arange(p, p + next_n) for p in base]).to(torch.int64)
        tail_blocks = torch.tensor([1, 3])[:b, None]
        tslot = (tail_blocks * pool + pos % pool).to(torch.int32)
        slot = torch.where(pos % pool == pool - 1, pos // pool + 2 * page,
                           torch.full_like(pos, -1)).to(torch.int32)
        outs = []
        for p in (pos.to(torch.int32), pos):
            kv_d, tail_d = kv.clone().to(dev), tail.clone().to(dev)
            kp.kpool_decode_update_and_maybe_write_cache_batched(
                kv_d, tail_d, tslot.to(dev), key.to(dev), score.to(dev),
                ape.to(dev), slot.to(dev), p.to(dev), pool, HEAD_DIM,
                round_scale=True)
            outs.append((kv_d.cpu(), tail_d.cpu()))
        assert torch.equal(outs[0][0], outs[1][0])
        assert torch.equal(outs[0][1], outs[1][1])
        assert not torch.equal(outs[0][0], kv)  # a pool was completed


def _check_fwht(dev):
    from vllm.ampere_decode.idx_glue import fwht128_quant_fp8_wscale
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8

    g = _gen(2)
    wscale = HEAD_DIM**-0.5 * N_HEAD**-0.5
    for m in (1, 4, 16, 40):
        q = (torch.randn(m * N_HEAD, HEAD_DIM, generator=g)
             * torch.logspace(-3, 3, m * N_HEAD)[:, None]).to(torch.bfloat16)
        q[0] = 0  # the absmax floor
        wbuf = torch.randn(m, HEAD_DIM + N_HEAD, generator=g) * 3
        weights = wbuf[:, HEAD_DIM:]  # strided like the dual GEMM view
        q_d, w_d = q.to(dev), weights.to(dev)

        q8_ref, qs_ref = fwht128_quant_fp8(q_d)
        w_ref = (w_d.unsqueeze(-1) * qs_ref.view(-1, N_HEAD, 1) * wscale).squeeze(-1)
        q8_new, w_new = fwht128_quant_fp8_wscale(q_d, w_d, wscale)
        assert torch.equal(q8_new.view(torch.uint8).cpu(),
                           q8_ref.view(torch.uint8).cpu()), m
        assert torch.equal(w_new.cpu(), w_ref.cpu()), m
        assert w_new.is_contiguous() and w_new.shape == (m, N_HEAD)


def _check_moe_sum_add(dev):
    from vllm.ampere_decode.idx_glue import moe_sum_add
    from vllm.ampere_decode.moe_routing import moe_sum

    g = _gen(3)
    for m in (1, 4, 16):
        inp = (torch.randn(m * 8, HIDDEN, generator=g) * 4).to(torch.bfloat16)
        inp = inp.view(m, 8, HIDDEN)
        shared = (torch.randn(m, HIDDEN, generator=g) * 4).to(torch.bfloat16)
        inp_d, sh_d = inp.to(dev), shared.to(dev)
        routed = moe_sum(inp_d)  # the Triton sum the fold replaces
        # the out-of-place bf16 add: fp32 opmath, one rounding
        ref = _to_bf16(sh_d.float() + routed.float())
        new = moe_sum_add(inp_d, sh_d)
        assert torch.equal(new.cpu(), ref.cpu()), m
        if dev != "cpu":
            assert torch.equal(new, sh_d + routed), m


def _remap_ref(req_id, block_table, ti, block_size, stride_rows):
    tok = ti.to(torch.int64)
    blk = torch.div(tok, block_size, rounding_mode="floor")
    off = tok - blk * block_size
    valid = (tok >= 0) & (blk < block_table.shape[1])
    safe = torch.where(valid, blk, torch.zeros_like(blk))
    base = block_table.to(torch.int64)[req_id.to(torch.int64)[:, None], safe]
    out = base * stride_rows + off
    return torch.where(valid, out, torch.full_like(out, -1)).to(torch.int32)


def _cache_ref(kv_cache, kv_c, k_pe, slots):
    out = kv_cache.clone()
    d_c = kv_c.shape[1]
    for t in range(slots.shape[0]):
        s = int(slots[t])
        if s < 0:
            continue
        blk, off = divmod(s, kv_cache.shape[1])
        out[blk, off, :d_c] = kv_c[t]
        if k_pe.numel():
            out[blk, off, d_c:] = k_pe[t].reshape(-1)
    return out


def _check_remap_cache(dev):
    from vllm.ampere_decode.idx_glue import remap_and_cache_mla

    g = _gen(4)
    block_size, num_blocks, pad_rows = 64, 16, 8
    for d_pe, n_q, n_w in ((0, 4, 4), (0, 12, 16), (64, 4, 4)):
        width = 512 + d_pe
        store = torch.randn(num_blocks, block_size + pad_rows, width,
                            generator=g).to(torch.bfloat16)
        kv_cache = store[:, :block_size]  # block stride > block_size rows
        stride_rows = kv_cache.stride(0) // width
        n_req = 3
        max_blocks = 6
        block_table = torch.randint(0, num_blocks, (n_req, max_blocks), generator=g,
                                    dtype=torch.int32)
        req_id = torch.randint(0, n_req, (n_q,), generator=g, dtype=torch.int32)
        ti = torch.randint(-1, max_blocks * block_size + 40, (n_q, BUF_WIDTH),
                           generator=g, dtype=torch.int32)
        ti[:, -200:] = -1
        kvc_buf = torch.randn(max(n_w, n_q), 576, generator=g).to(torch.bfloat16)
        kv_c = kvc_buf[:, :512]
        k_pe = torch.randn(max(n_w, n_q), 1, d_pe, generator=g).to(torch.bfloat16)
        slots = torch.randperm(num_blocks * block_size, generator=g)[:n_w].to(
            torch.int64)
        slots[1] = -1

        want_idx = _remap_ref(req_id, block_table, ti, block_size, stride_rows)
        want_cache = _cache_ref(kv_cache, kv_c, k_pe, slots)
        store_d = store.clone().to(dev)
        kv_d = store_d[:, :block_size]
        got = remap_and_cache_mla(
            req_id.to(dev), block_table.to(dev), ti.to(dev), kv_c.to(dev),
            k_pe.to(dev), kv_d, slots.to(dev), block_size, stride_rows)
        assert torch.equal(got.cpu(), want_idx), d_pe
        assert torch.equal(kv_d.cpu(), want_cache), d_pe
        # the padding rows between blocks are never touched
        assert torch.equal(store_d[:, block_size:].cpu(), store[:, block_size:])


def _check_dual_gemm(dev, ms=(1, 4, 16, 32)):
    from vllm.ampere_decode.idx_glue import thin_gemm_dual
    from vllm.ampere_thin_gemm.thin_gemm import thin_gemm

    g = _gen(5)
    n = HEAD_DIM + N_HEAD
    w = (torch.randn(n, HIDDEN, generator=g) * 0.02).to(torch.bfloat16)
    stats = []
    for m in ms:
        x = torch.randn(m, HIDDEN, generator=g).to(torch.bfloat16)
        x_d, w_d = x.to(dev), w.to(dev)
        full = thin_gemm(x_d, w_d)
        lo, hi = thin_gemm_dual(x_d, w_d, HEAD_DIM)
        ref_lo = full[:, :HEAD_DIM]
        assert torch.equal(lo.cpu(), ref_lo.cpu()), m
        assert lo.stride() == ref_lo.stride(), (lo.stride(), ref_lo.stride())
        # the fp32 block is the accumulator the bf16 store rounds
        assert torch.equal(_to_bf16(hi).cpu(), full[:, HEAD_DIM:].cpu()), m
        assert (full.cpu().double() - x.double() @ w.double().t()).abs().max() < 0.05
        truth = x.double() @ w[HEAD_DIM:].double().t()
        old = (x_d.float() @ w_d[HEAD_DIM:].t().contiguous().float()).cpu()
        err_new = (hi.cpu().double() - truth).abs().max().item()
        err_old = (old.double() - truth).abs().max().item()
        diff = (hi.cpu() - old).abs().max().item()
        stats.append((m, err_new, err_old, diff))
        scale = truth.abs().max().item()
        assert err_new <= 1e-5 * scale + 2 * err_old, (m, err_new, err_old)
    return stats


CHECKS = {
    "expand": _check_expand,
    "kpool_pos64": _check_kpool_pos64,
    "fwht": _check_fwht,
    "moe_sum_add": _check_moe_sum_add,
    "remap_cache": _check_remap_cache,
}
# The Triton 3.7 interpreter computes bf16 tl.dot wrongly (the incumbent
# thin_gemm returns values off by ~4e12 under it), so the dual GEMM is checked
# on the GPU only.
GPU_CHECKS = dict(CHECKS, dual_gemm=lambda dev: _check_dual_gemm(dev))


def _interpret(name):
    import vllm

    root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(
                   p for p in (root, os.environ.get("PYTHONPATH")) if p))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), name], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and f"PASS {name}" in r.stdout, (
        r.stdout[-3000:] + r.stderr[-6000:])


@pytest.mark.parametrize("name", list(CHECKS))
def test_fold_matches_replaced_code_interpreted(name):
    _interpret(name)


@cuda_only
@pytest.mark.parametrize("name", list(GPU_CHECKS))
def test_fold_matches_replaced_code_gpu(name):
    GPU_CHECKS[name]("cuda")


@cuda_only
def test_dual_gemm_all_decode_sizes_gpu():
    stats = _check_dual_gemm("cuda", ms=(1, 2, 4, 8, 16, 24, 32))
    for m, err_new, err_old, diff in stats:
        print(f"M={m}: max|new-fp64|={err_new:.3e} max|sgemm-fp64|={err_old:.3e} "
              f"max|new-sgemm|={diff:.3e}")


@cuda_only
def test_weight_scale_matches_compiled_leaf_gpu():
    """The production unfused path is the inductor leaf, not eager torch."""
    from vllm.ampere_decode.idx_glue import fwht128_quant_fp8_wscale
    from vllm.models.glm5next.common.attention import _fused_indexer_weight_scale
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8

    g = _gen(6)
    wscale = HEAD_DIM**-0.5 * N_HEAD**-0.5
    for m in (1, 4, 16):
        q = torch.randn(m * N_HEAD, HEAD_DIM, generator=g).to(torch.bfloat16).cuda()
        w = torch.randn(m, N_HEAD, generator=g).cuda()
        _, qs = fwht128_quant_fp8(q)
        ref = _fused_indexer_weight_scale(w, qs.view(-1, N_HEAD, 1), wscale)
        _, new = fwht128_quant_fp8_wscale(q, w, wscale)
        assert torch.equal(new, ref), m


@cuda_only
def test_cache_write_matches_concat_and_cache_mla_gpu():
    from vllm import _custom_ops as ops
    from vllm.ampere_decode.idx_glue import remap_and_cache_mla
    from vllm.v1.attention.backends.mla.sparse_utils import (
        flat_kv_row_view,
        triton_convert_req_index_to_global_index,
    )

    g = _gen(7)
    block_size, num_blocks = 64, 32
    for n in (4, 16):
        kv = torch.randn(num_blocks, block_size, 512, generator=g).to(
            torch.bfloat16).cuda()
        kv_c = torch.randn(n, 576, generator=g).to(torch.bfloat16).cuda()[:, :512]
        k_pe = torch.empty(n, 1, 0, dtype=torch.bfloat16, device="cuda")
        slots = torch.randperm(num_blocks * block_size, generator=g)[:n].cuda()
        slots[0] = -1
        bt = torch.randint(0, num_blocks, (2, 8), generator=g,
                           dtype=torch.int32).cuda()
        req = torch.randint(0, 2, (n,), generator=g, dtype=torch.int32).cuda()
        ti = torch.randint(-1, 8 * block_size, (n, BUF_WIDTH), generator=g,
                           dtype=torch.int32).cuda()
        ref_kv = kv.clone()
        ops.concat_and_cache_mla(kv_c, k_pe.squeeze(1), ref_kv, slots,
                                 kv_cache_dtype="auto",
                                 scale=torch.ones(1, device="cuda"))
        _, stride_rows = flat_kv_row_view(ref_kv, block_size)
        ref_idx = triton_convert_req_index_to_global_index(
            req, bt, ti, BLOCK_SIZE=block_size, BLOCK_STRIDE_ROWS=stride_rows,
            NUM_TOPK_TOKENS=BUF_WIDTH)
        new_kv = kv.clone()
        new_idx = remap_and_cache_mla(req, bt, ti, kv_c, k_pe, new_kv, slots,
                                      block_size, stride_rows)
        assert torch.equal(new_kv, ref_kv), n
        assert torch.equal(new_idx, ref_idx), n


@cuda_only
def test_moe_sum_add_matches_cuda_moe_sum_gpu():
    from vllm import _custom_ops as ops
    from vllm.ampere_decode.idx_glue import moe_sum_add

    g = _gen(8)
    for m in (4, 16, 32):
        inp = (torch.randn(m, 8, HIDDEN, generator=g) * 4).to(torch.bfloat16).cuda()
        sh = (torch.randn(m, HIDDEN, generator=g) * 4).to(torch.bfloat16).cuda()
        routed = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ops.moe_sum(inp, routed)
        assert torch.equal(moe_sum_add(inp, sh), sh + routed), m


def _main(argv):
    name = argv[1]
    if os.environ.get("TRITON_INTERPRET") == "1":
        # no device to ask: the fp8 stores take the software path, as on sm_80
        import vllm.v1.attention.ops.triton_e4m3 as e4m3

        e4m3.triton_fp8_e4m3_native = lambda: False
        import vllm.models.glm5next.nvidia.ops.kpool_compress as kp

        kp.triton_fp8_e4m3_native = e4m3.triton_fp8_e4m3_native
        import vllm.ampere_decode.idx_glue as ig

        ig.triton_fp8_e4m3_native = e4m3.triton_fp8_e4m3_native
        import vllm.ampere_thin_gemm.thin_gemm as tg

        tg._num_sms_cache = 70
    dev = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    (CHECKS if dev == "cpu" else GPU_CHECKS)[name](dev)
    print(f"PASS {name}")


if __name__ == "__main__":
    _main(sys.argv)
