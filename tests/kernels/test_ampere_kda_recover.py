# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_GLM5_KDA_RECOVER (vllm/ampere_decode/kda_recover.py).

CPU part: env default, the gate, the KV spec (no draft-position state pages)
and the KV capacity it gives through the real KV-sizing code, the record
buffer size, the replay arithmetic.
GPU part (skipped without CUDA): the recover verify is bitwise today's v2
(outputs, conv); the commit gives bitwise today's state at the accepted
position (incl. the align-mode block-boundary state) and today's conv window;
the mixed-step variant leaves the output unnormalised; verify and commit
capture into CUDA graphs with zero allocation growth and replay bitwise.

    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_recover.py
"""

import os
import types

try:
    import pytest
except ImportError:
    pytest = None
import torch

import vllm.ampere_decode as ad
from vllm.ampere_decode import kda_recover as kr

H, D, KA, CONV_K = 16, 128, 128, 4
PROJ = H * D
CONV_DIM = 3 * PROJ
PW = CONV_DIM + H + 2 * KA
ENV = ("VLLM_GLM5_KDA_RECOVER", "VLLM_GLM5_DECODE_KERNELS", "VLLM_GLM5_DECODE_KDA_V2",
       "VLLM_KV_MAMBA_INFLIGHT_STATES", "VLLM_KV_SWA_INFLIGHT_SCRATCH")


class _Env:
    def __init__(self):
        self._saved = {k: os.environ.get(k) for k in ENV}
        self._sm80 = ad._SM80_CACHE

    def set(self, sm80=True, **kv):
        for k, v in kv.items():
            os.environ[k] = str(v)
        ad._SM80_CACHE = sm80
        return self

    def undo(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ad._SM80_CACHE = self._sm80


def _cfg(tp=4, pp=1, num_spec=7, max_seqs=8, mode="align", arch="Glm5NextForConditionalGeneration",
         heads=64, max_len=262_144):
    text = types.SimpleNamespace(linear_num_heads=heads, linear_head_dim=128)
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(architectures=[arch]), hf_text_config=text,
            max_model_len=max_len, dtype=torch.bfloat16),
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=pp,
                                              tensor_parallel_size=tp,
                                              decode_context_parallel_size=1,
                                              prefill_context_parallel_size=1),
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_seqs),
        cache_config=types.SimpleNamespace(mamba_cache_mode=mode, use_replayssm=False,
                                           mamba_block_size=1152,
                                           mamba_page_size_padded=1152 * 1024,
                                           mamba_cache_dtype="auto",
                                           use_kda_recoverssm=False),
        num_speculative_tokens=num_spec, max_concurrent_batches=2,
        max_in_flight_tokens=6926, kv_transfer_config=None)


# ------------------------------------------------------------------ CPU tests

def test_env_default_is_off():
    os.environ.pop("VLLM_GLM5_KDA_RECOVER", None)
    from vllm import envs

    assert envs.VLLM_GLM5_KDA_RECOVER is False
    assert kr.kda_recover_enabled(_cfg()) is False


def test_gate():
    e = _Env()
    try:
        e.set(VLLM_GLM5_KDA_RECOVER=1, VLLM_GLM5_DECODE_KERNELS=1, VLLM_GLM5_DECODE_KDA_V2=1)
        assert kr.kda_recover_enabled(_cfg())
        assert kr.kda_recover_enabled(_cfg(num_spec=3, mode="none"))
        for bad in (_cfg(pp=4, tp=1), _cfg(num_spec=0), _cfg(num_spec=8), _cfg(tp=2),
                    _cfg(max_seqs=16), _cfg(mode="all"), _cfg(arch="KimiLinearForCausalLM")):
            assert not kr.kda_recover_enabled(bad)
        e.set(VLLM_GLM5_DECODE_KDA_V2=0)
        assert not kr.kda_recover_enabled(_cfg())
        e.set(VLLM_GLM5_DECODE_KDA_V2=1, sm80=False)
        assert not kr.kda_recover_enabled(_cfg())
    finally:
        e.undo()


def _kda_spec(recover, num_spec=7):
    """The KV spec through Glm5NextLinearAttention.get_kv_cache_spec."""
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention as K

    layer = K.__new__(K)
    torch.nn.Module.__init__(layer)
    layer.tp_size, layer.num_heads, layer.head_dim = 4, 64, 128
    layer.conv_size, layer.num_spec = 4, num_spec
    cfg = _cfg(num_spec=num_spec)
    layer.model_config, layer.cache_config = cfg.model_config, cfg.cache_config
    layer._kda_recover = recover
    return layer.get_kv_cache_spec(cfg)


def test_kv_spec_has_no_draft_state_pages():
    off, on = _kda_spec(False), _kda_spec(True)
    assert off.num_speculative_blocks == 7 and on.num_speculative_blocks == 0
    assert on.page_size_bytes == off.page_size_bytes == 1152 * 1024
    assert on.shapes == off.shapes and on.block_size == off.block_size == 1152


def _report(spec, records_bytes):
    """GPU KV cache size as get_max_concurrency_for_kv_cache_config reports
    it, for the pool of boot depth7_val/tp4_d7_1 (1,138 blocks of 1,152)."""
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    class Other(FullAttentionSpec):
        """The sliding-window in-flight and indexer-tail groups: 10 blocks per
        request (boot log of depth7_val/tp4_d7_1)."""

        def max_memory_usage_bytes(self, vllm_config):
            return 10 * self.page_size_bytes

    # 1,024 B per token: the MLA page of the boot (1,179,648 B at 1,152)
    mla = FullAttentionSpec(block_size=1152, num_kv_heads=1, head_size=256,
                            dtype=torch.bfloat16)
    other = Other(block_size=1152, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    assert mla.page_size_bytes == spec.page_size_bytes
    bytes_per_block = 11 * 1152 * (1024 + 33)            # 11 MLA + 11 indexer tensors
    pool = 1138 * bytes_per_block                       # lower end of the logged pool
    groups = ([KVCacheGroupSpec([f"kda{g}"], spec) for g in range(4)]
              + [KVCacheGroupSpec(["mla"], mla), KVCacheGroupSpec(["other"], other)])
    out = []
    for p in (pool, pool + bytes_per_block - 1):
        n = (p - records_bytes) // bytes_per_block
        kcfg = KVCacheConfig(num_blocks=n, kv_cache_tensors=[], kv_cache_groups=groups)
        out.append(int(get_max_concurrency_for_kv_cache_config(_cfg(), kcfg) * 262_144))
    return out


def test_kv_capacity_tp4_depth7():
    e = _Env()
    try:
        e.set(VLLM_KV_MAMBA_INFLIGHT_STATES=1, VLLM_KV_SWA_INFLIGHT_SCRATCH=0)
        off = _report(_kda_spec(False), 0)
        on = _report(_kda_spec(True), kr.records_nbytes(_cfg(), 34))
        assert off[0] == 1_073_093, off           # logged, boot tp4_d7_1
        assert 1_189_085 <= on[0] <= on[1] <= 1_190_133, on
        print(f"  KV tokens: off {off[0]:,}  on {on[0]:,}..{on[1]:,} "
              f"(+{100 * (on[0] / off[0] - 1):.1f} %)")
    finally:
        e.undo()


def test_records_bytes():
    assert kr.records_nbytes(_cfg(), 34) == 34 * 3 * 8 * 8 * 16 * 128 * 4 == 53_477_376


def test_commit_reference_replays_per_position_states():
    """CPU: replaying (c, k, e) with fma(c, k, h*e) reproduces every
    per-position state bitwise (the kernel's sequence: he = h*e, c from he,
    h = fma(c, k, he))."""
    g = torch.Generator().manual_seed(0)
    T = 8
    states = torch.randn(2, H, D, D, generator=g) * 0.5
    rec = torch.zeros(3, 1, kr.WS_T, H, D)
    h = states[1].clone()
    per_pos = []
    for t in range(T):
        e = torch.exp(-5 * torch.sigmoid(torch.randn(H, D, generator=g).double())).float()
        k = torch.randn(H, D, generator=g)
        k = (k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).float()
        v = torch.randn(H, D, generator=g)
        beta = torch.sigmoid(torch.randn(H, generator=g))
        he = h * e[:, None, :]
        c = ((v - (he * k[:, None, :]).sum(-1)) * beta[:, None]).float()
        h = (c[:, :, None].double() * k[:, None, :].double() + he.double()).float()
        per_pos.append(h.clone())
        rec[0, 0, t], rec[1, 0, t], rec[2, 0, t] = c, k, e
    for n in range(1, T + 1):
        r = kr.commit_reference(states, rec, 1, n, 0)
        assert torch.equal(r.view(torch.int32), per_pos[n - 1].view(torch.int32)), n


def test_import_does_not_initialise_cuda():
    import importlib

    importlib.reload(kr)
    assert torch.cuda.is_initialized() is False


# ------------------------------------------------------------------ GPU tests

def _inputs(nseq, T, seed, nslot, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)

    def n(shape, mean, std, dtype=torch.float32):
        return torch.empty(shape, device=dev).normal_(mean, std, generator=g).to(dtype)

    M = nseq * T
    buf = n((M, PW), 0.0, 1.0)
    buf[:, CONV_DIM + H:CONV_DIM + H + KA] *= 1.0 / (0.02344 * KA ** 0.5)
    buf[:, CONV_DIM + H + KA:] *= 1.0 / (0.02858 * KA ** 0.5)
    proj = buf.to(torch.bfloat16)
    p = types.SimpleNamespace(
        qkv=proj[:, :CONV_DIM], beta=proj[:, CONV_DIM:CONV_DIM + H].unsqueeze(0),
        fa=proj[:, CONV_DIM + H:CONV_DIM + H + KA], ga=proj[:, CONV_DIM + H + KA:],
        wf=n((PROJ, KA), 0.0, 0.02344, torch.bfloat16),
        wg=n((PROJ, KA), 0.0, 0.02858, torch.bfloat16),
        cw=n((CONV_DIM, CONV_K), 0.0, 0.5), nw=n((D,), 0.1325, 0.0126, torch.bfloat16),
        al=n((H,), 1.527, 0.411), gb=n((PROJ,), -0.815, 1.211),
        conv=n((nslot, CONV_K - 1 + 7, CONV_DIM), 0.0, 1.0, torch.bfloat16),
        rec=n((nslot, H, D, D), 0.0, 0.5),
        qsl=torch.arange(0, nseq + 1, device=dev, dtype=torch.int32) * T)
    return p


def _v2(p, conv, rec, ssm, acc, T, records=None, skip_norm=False, out=None):
    from vllm.ampere_decode.kda_decode_v2 import kda_decode_v2

    nseq = ssm.shape[0]
    if out is None:
        out = torch.empty(1, nseq * T, H, D, device="cuda", dtype=torch.bfloat16)
    return kda_decode_v2(p.qkv, p.beta, p.fa, p.ga, p.wf, p.wg, conv.transpose(-1, -2),
                         p.cw, None, p.nw, rec, ssm[:, 0], ssm, acc, p.qsl, 8, p.al, p.gb,
                         out=out, records=records, skip_norm=skip_norm)


def _context(conv, rec, pool):
    layer = types.SimpleNamespace(kv_cache=(conv, rec), _kda_recover_pool=pool,
                                  _kda_recover_index=0)
    return kr.KDARecoverCommitContext.create([layer], spec_query_len=8, max_num_reqs=8)


def gpu_test_verify_and_commit_bitwise():
    """For T 1..8, nseq 1/2/4/8, previous accepted a and new accepted n:
    recover verify == today's v2 (outputs, conv) bitwise; commit == today's
    state at column n-1 bitwise; compacted conv == today's window at n."""
    from vllm.ampere_decode import kda_decode_v2 as k2

    nfail = 0
    for T in range(1, 9):
        for nseq in (1, 2, 4, 8):
            k2.warmup(plans=((nseq, 8),), recover=True)
            nslot = 2 + nseq * 8
            p = _inputs(nseq, T, 1000 * T + nseq, nslot)
            # today: 8 distinct slots per sequence, state read at column a-1
            ssm8 = torch.as_tensor([[1 + s * 8 + t for t in range(8)] for s in range(nseq)],
                                   device="cuda", dtype=torch.int32)
            p.qsl = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * T
            for a in sorted({1, T}):
                acc = torch.full((nseq,), a, device="cuda", dtype=torch.int32)
                ra, ca = p.rec.clone(), p.conv.clone()
                ya = _v2(p, ca, ra, ssm8, acc, T).clone()
                # recover: one slot per sequence (column 0) holding today's initial state
                rb, cb = p.rec.clone(), p.conv.clone()
                for s in range(nseq):
                    rb[1 + s * 8] = p.rec[1 + s * 8 + a - 1]
                # today's conv window starts at row a-1; recover reads row 0
                cb_shift = cb.clone()
                for s in range(nseq):
                    w = cb[1 + s * 8, a - 1:].clone()
                    cb_shift[1 + s * 8, :w.shape[0]] = w
                cb = cb_shift
                ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
                pool = torch.full((1, 3, 8, kr.WS_T, H, D), float("nan"), device="cuda")
                recs = kr.layer_records(pool, 0)
                rb0 = rb.clone()
                yb = _v2(p, cb, rb, ssm8[:, :1].contiguous(), ones, T, records=recs).clone()
                torch.cuda.synchronize()
                ok = torch.equal(ya, yb) and torch.equal(rb, rb0)
                # conv after verify: today's rows [a-1 ..] == recover rows [0 ..]
                for s in range(nseq):
                    keep = CONV_K - 2 + T
                    ok &= torch.equal(ca[1 + s * 8, :keep], cb[1 + s * 8, :keep])
                ctx = _context(cb, rb, pool)
                src = ssm8[:, 0].contiguous()
                for n_acc in range(1, T + 1):
                    rc, cc = rb.clone(), cb.clone()
                    ctx.conv_states = (cc.transpose(-1, -2),)
                    ctx.conv_state_base_addrs.fill_(cc.data_ptr())
                    ctx.states = (rc,)
                    ctx.state_base_addrs.fill_(rc.data_ptr())
                    nsamp = torch.full((nseq,), n_acc, device="cuda", dtype=torch.int32)
                    ctx.commit(nsamp, src, p.qsl)
                    torch.cuda.synchronize()
                    for s in range(nseq):
                        want = ra[1 + s * 8 + n_acc - 1]
                        got = rc[1 + s * 8]
                        ok &= torch.equal(got.view(torch.int32), want.view(torch.int32))
                        # next step reads conv rows [0, 3): today rows [n-1, n+2)
                        ok &= torch.equal(cc[1 + s * 8, :3], ca[1 + s * 8, n_acc - 1:n_acc + 2])
                if not ok:
                    nfail += 1
                    print(f"  FAIL T={T} nseq={nseq} a={a}")
    assert nfail == 0, nfail


def gpu_test_commit_align_boundary():
    """Align mode: a commit that crosses a block boundary writes the state
    after the boundary tokens to the boundary block and the final state to
    the block of the new position, both bitwise the per-position states."""
    from vllm.ampere_decode import kda_decode_v2 as k2

    T, nseq, bs = 8, 1, 16
    k2.warmup(plans=((nseq, 8),), recover=True)
    nslot = 12
    p = _inputs(nseq, T, 77, nslot)
    ssm8 = torch.arange(1, 9, device="cuda", dtype=torch.int32).view(1, 8)
    acc = torch.ones(1, device="cuda", dtype=torch.int32)
    ra = p.rec.clone()
    _v2(p, p.conv.clone(), ra, ssm8, acc, T)
    pool = torch.zeros(1, 3, 8, kr.WS_T, H, D, device="cuda")
    rb = p.rec.clone()
    _v2(p, p.conv.clone(), rb, ssm8[:, :1].contiguous(), acc, T, records=kr.layer_records(pool, 0))
    # block table: positions 0..47 -> blocks 9, 10, 11 (block size 16); 13 computed tokens
    bt = torch.tensor([[9, 10, 11]], device="cuda", dtype=torch.int32)
    rb[9] = rb[1]
    src = torch.tensor([9], device="cuda", dtype=torch.int32)
    ctx = _context(p.conv.clone(), rb, pool)
    ncomp = torch.tensor([13], device="cuda", dtype=torch.int32)
    nsamp = torch.tensor([6], device="cuda", dtype=torch.int32)   # 13 + 6 = 19: crosses 16
    ctx.commit(nsamp, src, p.qsl, block_table=bt, num_computed_tokens=ncomp,
               mamba_block_size=bs)
    torch.cuda.synchronize()
    # boundary after 3 tokens (13 -> 16) in block 9 (col 0), final after 6 in block 10
    assert torch.equal(rb[9].view(torch.int32), ra[1 + 2].view(torch.int32))
    assert torch.equal(rb[10].view(torch.int32), ra[1 + 5].view(torch.int32))


def gpu_test_skip_norm_is_the_staged_output():
    """The mixed-step variant leaves the recurrence output unnormalised: the
    normalised variant equals gated-RMSNorm(skip_norm output) up to rounding,
    and conv/records are identical between the two."""
    from vllm.ampere_decode import kda_decode_v2 as k2

    T, nseq = 8, 2
    k2.warmup(plans=((nseq, 8),), recover=True)
    p = _inputs(nseq, T, 5, 2 + nseq * 8)
    ssm = torch.tensor([[1], [9]], device="cuda", dtype=torch.int32)
    ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
    outs = []
    for skip in (False, True):
        pool = torch.zeros(1, 3, 8, kr.WS_T, H, D, device="cuda")
        r, c = p.rec.clone(), p.conv.clone()
        y = _v2(p, c, r, ssm, ones, T, records=kr.layer_records(pool, 0), skip_norm=skip)
        outs.append((y.clone(), c, pool))
    (yn, cn, pn), (ys, cs, ps) = outs
    assert torch.equal(cn, cs) and torch.equal(pn, ps)
    assert not torch.equal(yn, ys)
    x = ys.float()
    g2 = (p.ga.float() @ p.wg.float().t()).to(torch.bfloat16).float().view(1, -1, H, D)
    ref = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * p.nw.float() * torch.sigmoid(g2)
    assert float((ref - yn.float()).abs().max()) < 0.05 * float(ref.abs().max())


def gpu_test_graph_zero_growth():
    """Verify and commit capture with no allocation beyond torch's own per-graph
    capture allocation, replays allocate nothing and are bitwise equal to eager."""
    from vllm.ampere_decode import kda_decode_v2 as k2

    T, nseq = 8, 4
    k2.warmup(plans=((nseq, 8),), recover=True)
    p = _inputs(nseq, T, 9, 2 + nseq * 8)
    ssm = torch.tensor([[1 + 8 * s] for s in range(nseq)], device="cuda", dtype=torch.int32)
    ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
    nsamp = torch.tensor([1, 4, 8, 6], device="cuda", dtype=torch.int32)
    src = ssm[:, 0].contiguous()
    pool = torch.zeros(1, 3, 8, kr.WS_T, H, D, device="cuda")
    recs = kr.layer_records(pool, 0)
    rec0, conv0 = p.rec.clone(), p.conv.clone()
    out = torch.empty(1, nseq * T, H, D, device="cuda", dtype=torch.bfloat16)
    ctx = _context(p.conv, p.rec, pool)

    def step():
        _v2(p, p.conv, p.rec, ssm, ones, T, records=recs, out=out)
        ctx.commit(nsamp, src, p.qsl)

    step()
    torch.cuda.synchronize()
    ye, re_, ce = out.clone(), p.rec.clone(), p.conv.clone()
    p.rec.copy_(rec0)
    p.conv.copy_(conv0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    # torch's capture machinery allocates a little per graph by itself
    # (1,024 B for a graph holding one in-place add on this stack, the
    # kda_decode_v2 harness calibration); the capture may allocate exactly
    # that much and nothing more, and replays nothing.
    scratch = torch.zeros(16, device="cuda")
    torch.cuda.synchronize()
    c0 = torch.cuda.memory_allocated()
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        scratch.add_(1.0)
    torch.cuda.synchronize()
    cap_overhead = torch.cuda.memory_allocated() - c0
    del g0
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()
    grow = torch.cuda.memory_allocated() - m0 - cap_overhead
    assert grow <= 0, (grow, cap_overhead)
    m1 = torch.cuda.memory_allocated()
    for _ in range(2):
        p.rec.copy_(rec0)
        p.conv.copy_(conv0)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ye) and torch.equal(p.rec, re_) and torch.equal(p.conv, ce)
    assert torch.cuda.memory_allocated() <= m1, "replay allocated"
    print(f"  capture growth beyond torch's own {cap_overhead} B/graph: {grow} B")
    del g


CPU_TESTS = (test_env_default_is_off, test_gate, test_kv_spec_has_no_draft_state_pages,
             test_kv_capacity_tp4_depth7, test_records_bytes,
             test_commit_reference_replays_per_position_states,
             test_import_does_not_initialise_cuda)
GPU_TESTS = (gpu_test_verify_and_commit_bitwise, gpu_test_commit_align_boundary,
             gpu_test_skip_norm_is_the_staged_output, gpu_test_graph_zero_growth)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an sm_80 GPU")
    test_gpu_verify_and_commit_bitwise = _needs_gpu(gpu_test_verify_and_commit_bitwise)
    test_gpu_commit_align_boundary = _needs_gpu(gpu_test_commit_align_boundary)
    test_gpu_skip_norm_is_the_staged_output = _needs_gpu(gpu_test_skip_norm_is_the_staged_output)
    test_gpu_graph_zero_growth = _needs_gpu(gpu_test_graph_zero_growth)


def _main():
    import sys
    import traceback

    tests = list(CPU_TESTS)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available():
        tests += list(GPU_TESTS)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
